#!/usr/bin/env python3
"""
PyTorch Dataset for Oligogenic Paired Variants

Provides dataset classes for training oligogenic interaction models
using paired variant data from OLIDA or other sources.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import hashlib
import pickle


class OligogenicPairedDataset(Dataset):
    """
    PyTorch Dataset for paired variant oligogenic classification.

    Each sample contains two variants (each with ref/alt sequences)
    and a binary label indicating oligogenic interaction.

    Expected input format:
        List of (variant_dict, label) where variant_dict contains:
        - variant1_ref, variant1_alt: Sequences for variant 1
        - variant2_ref, variant2_alt: Sequences for variant 2
        - Optional: gene1, gene2, disease, olida_id
    """

    def __init__(
        self,
        data: List[Tuple[Dict, int]],
        tokenizer,
        max_length: int = 1024,
        return_metadata: bool = False,
    ):
        """
        Args:
            data: List of (variant_dict, label) tuples
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            return_metadata: Include gene/disease info in output
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        variant_dict, label = self.data[idx]

        # Tokenize all four sequences
        sequences = {
            'variant1_ref': variant_dict.get('variant1_ref', ''),
            'variant1_alt': variant_dict.get('variant1_alt', ''),
            'variant2_ref': variant_dict.get('variant2_ref', ''),
            'variant2_alt': variant_dict.get('variant2_alt', ''),
        }

        tokenized = {}
        for key, seq in sequences.items():
            encoded = self.tokenizer(
                seq,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            )
            tokenized[f'{key}_input_ids'] = encoded['input_ids'].squeeze(0)
            if 'attention_mask' in encoded:
                tokenized[f'{key}_attention_mask'] = encoded['attention_mask'].squeeze(0)

        tokenized['labels'] = torch.tensor(label, dtype=torch.long)

        if self.return_metadata:
            tokenized['gene1'] = variant_dict.get('gene1', '')
            tokenized['gene2'] = variant_dict.get('gene2', '')
            tokenized['disease'] = variant_dict.get('disease', '')

        return tokenized


class OligogenicEmbeddingDataset(Dataset):
    """
    Dataset using pre-computed delta embeddings for efficiency.

    Use this when training with cached embeddings to avoid
    redundant forward passes through the base model.
    """

    def __init__(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor,
        metadata: Optional[List[Dict]] = None,
    ):
        """
        Args:
            embeddings1: Pre-computed delta embeddings for variant 1 (N x hidden_size)
            embeddings2: Pre-computed delta embeddings for variant 2 (N x hidden_size)
            labels: Binary labels (N,)
            metadata: Optional list of metadata dicts
        """
        assert len(embeddings1) == len(embeddings2) == len(labels)
        self.embeddings1 = embeddings1
        self.embeddings2 = embeddings2
        self.labels = labels
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'embedding1': self.embeddings1[idx],
            'embedding2': self.embeddings2[idx],
            'labels': self.labels[idx],
        }
        if self.metadata:
            item['metadata'] = self.metadata[idx]
        return item


class OligogenicEmbeddingCache:
    """
    Cache delta embeddings per variant to avoid redundant computation.

    Useful when the same variant appears in multiple pairs.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        max_memory_items: int = 10000,
    ):
        """
        Args:
            cache_dir: Directory for disk cache (None for memory-only)
            max_memory_items: Maximum items in memory cache
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_memory_items = max_memory_items
        self._memory_cache: Dict[str, torch.Tensor] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, chrom: str, pos: int, ref: str, alt: str) -> str:
        """Create unique key for variant."""
        key_str = f"{chrom}:{pos}:{ref}>{alt}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, chrom: str, pos: int, ref: str, alt: str) -> Optional[torch.Tensor]:
        """Get cached embedding if available."""
        key = self._make_key(chrom, pos, ref, alt)

        # Check memory cache
        if key in self._memory_cache:
            return self._memory_cache[key]

        # Check disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pt"
            if cache_file.exists():
                embedding = torch.load(cache_file, weights_only=True)
                self._add_to_memory(key, embedding)
                return embedding

        return None

    def put(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        embedding: torch.Tensor,
    ) -> None:
        """Cache embedding for variant."""
        key = self._make_key(chrom, pos, ref, alt)

        # Add to memory cache
        self._add_to_memory(key, embedding)

        # Save to disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pt"
            torch.save(embedding.cpu(), cache_file)

    def _add_to_memory(self, key: str, embedding: torch.Tensor) -> None:
        """Add to memory cache with LRU eviction."""
        if len(self._memory_cache) >= self.max_memory_items:
            # Remove oldest item
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
        self._memory_cache[key] = embedding

    def clear(self) -> None:
        """Clear all caches."""
        self._memory_cache.clear()
        if self.cache_dir:
            for f in self.cache_dir.glob("*.pt"):
                f.unlink()


def collate_paired_variants(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for OligogenicPairedDataset.

    Stacks all tensors and handles variable-length metadata.
    """
    collated = {}

    # Get all tensor keys from first item
    tensor_keys = [k for k, v in batch[0].items() if isinstance(v, torch.Tensor)]

    for key in tensor_keys:
        collated[key] = torch.stack([item[key] for item in batch])

    # Handle metadata if present
    if 'gene1' in batch[0]:
        collated['gene1'] = [item['gene1'] for item in batch]
        collated['gene2'] = [item['gene2'] for item in batch]
        collated['disease'] = [item['disease'] for item in batch]

    return collated


def create_oligogenic_dataloaders(
    train_data: List[Tuple[Dict, int]],
    val_data: List[Tuple[Dict, int]],
    tokenizer,
    batch_size: int = 16,
    max_length: int = 1024,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders for oligogenic training.

    Args:
        train_data: Training data
        val_data: Validation data
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = OligogenicPairedDataset(
        train_data, tokenizer, max_length=max_length
    )
    val_dataset = OligogenicPairedDataset(
        val_data, tokenizer, max_length=max_length
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_paired_variants,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_paired_variants,
        pin_memory=True,
    )

    return train_loader, val_loader
