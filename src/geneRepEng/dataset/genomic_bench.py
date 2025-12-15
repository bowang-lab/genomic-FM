"""
Genomic dataset utilities for control vector training
"""

import random
import typing
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path

from ..extract import GenomicDatasetEntry


@dataclass
class GenomicDataset:
    """Dataset class for genomic sequence pairs"""
    entries: List[GenomicDatasetEntry]
    name: str = "genomic_dataset"

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]

    def sample(self, n: int, seed: Optional[int] = None) -> "GenomicDataset":
        """Sample n entries from the dataset"""
        if seed is not None:
            random.seed(seed)
        sampled_entries = random.sample(self.entries, min(n, len(self.entries)))
        return GenomicDataset(sampled_entries, f"{self.name}_sampled_{n}")


class GenomicDatasetAnswers:
    """Container for genomic dataset answers/labels"""

    def __init__(self, answers: Dict[str, Union[int, str, float]]):
        self.answers = answers

    def get_answer(self, question_id: str) -> Union[int, str, float]:
        return self.answers.get(question_id, None)


def create_clinvar_control_dataset(
    data_path: str,
    max_samples: Optional[int] = None,
    pathogenic_only: bool = True,
    seed: int = 42
) -> GenomicDataset:
    """
    Create a control dataset from ClinVar data for training control vectors

    Args:
        data_path: Path to the processed ClinVar data
        max_samples: Maximum number of samples to include
        pathogenic_only: Whether to only include pathogenic vs benign variants
        seed: Random seed for sampling

    Returns:
        GenomicDataset with reference/alternative sequence pairs
    """
    # This would integrate with the existing ClinVar data wrapper
    from ...pack_tunable_model.hf_dataloader import return_clinvar_multitask_dataset
    from transformers import AutoTokenizer

    # Create a dummy tokenizer for data loading
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load ClinVar data
    datasets, task_num_classes, max_seq_len = return_clinvar_multitask_dataset(
        tokenizer=tokenizer,
        target='CLNSIG',  # Clinical significance
        seq_length=1024,
        val_split=0.1,
        test_split=0.1,
        seed=seed
    )

    # Extract training data
    train_dataset = datasets['train']

    entries = []
    for i in range(len(train_dataset)):
        item = train_dataset[i]

        # Decode sequences from tokenized format
        ref_tokens = item['ref_input_ids']
        alt_tokens = item['alt_input_ids']

        # Convert back to sequences (this is a simplification)
        ref_sequence = tokenizer.decode(ref_tokens, skip_special_tokens=True)
        alt_sequence = tokenizer.decode(alt_tokens, skip_special_tokens=True)

        label = item['labels']

        entry = GenomicDatasetEntry(
            ref_sequence=ref_sequence,
            alt_sequence=alt_sequence,
            label=label
        )
        entries.append(entry)

        if max_samples and len(entries) >= max_samples:
            break

    return GenomicDataset(entries, "clinvar_control")


def create_smart_variant_control_dataset(
    csv_path: str,
    fasta_path: Optional[str] = None,
    threshold: float = 54.0,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> GenomicDataset:
    """
    Create a control dataset from SMART variant data

    Args:
        csv_path: Path to the SMART variant CSV file
        fasta_path: Path to FASTA file (if needed)
        threshold: Threshold for binarizing SMART scores
        max_samples: Maximum number of samples to include
        seed: Random seed for sampling

    Returns:
        GenomicDataset with reference/alternative sequence pairs
    """
    # This would integrate with the existing SMART data wrapper
    from ...pack_tunable_model.hf_dataloader import return_smart_dataset
    from transformers import AutoTokenizer

    # Create a dummy tokenizer for data loading
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load SMART data
    datasets, task_num_classes, max_seq_len = return_smart_dataset(
        tokenizer=tokenizer,
        csv_path=csv_path,
        fasta_path=fasta_path,
        threshold=threshold,
        seq_length=1024,
        val_split=0.1,
        test_split=0.1,
        seed=seed,
        all_records=True,
        num_records=max_samples
    )

    # Extract training data
    train_dataset = datasets['train']

    entries = []
    for i in range(len(train_dataset)):
        item = train_dataset[i]

        # Decode sequences from tokenized format
        ref_tokens = item['ref_input_ids']
        alt_tokens = item['alt_input_ids']

        # Convert back to sequences (this is a simplification)
        ref_sequence = tokenizer.decode(ref_tokens, skip_special_tokens=True)
        alt_sequence = tokenizer.decode(alt_tokens, skip_special_tokens=True)

        label = item['labels']

        entry = GenomicDatasetEntry(
            ref_sequence=ref_sequence,
            alt_sequence=alt_sequence,
            label=label
        )
        entries.append(entry)

    return GenomicDataset(entries, "smart_variant_control")


def create_synthetic_control_dataset(
    n_samples: int = 1000,
    seq_length: int = 100,
    mutation_rate: float = 0.01,
    seed: int = 42
) -> GenomicDataset:
    """
    Create a synthetic dataset for testing control vector methods

    Args:
        n_samples: Number of sequence pairs to generate
        seq_length: Length of each sequence
        mutation_rate: Rate of mutations between ref and alt sequences
        seed: Random seed

    Returns:
        GenomicDataset with synthetic sequence pairs
    """
    random.seed(seed)
    np.random.seed(seed)

    nucleotides = ['A', 'T', 'G', 'C']
    entries = []

    for i in range(n_samples):
        # Generate reference sequence
        ref_seq = ''.join(random.choices(nucleotides, k=seq_length))

        # Generate alternative sequence with mutations
        alt_seq = list(ref_seq)
        n_mutations = max(1, int(seq_length * mutation_rate))
        mutation_positions = random.sample(range(seq_length), n_mutations)

        for pos in mutation_positions:
            # Change to a different nucleotide
            current_nuc = alt_seq[pos]
            available_nucs = [n for n in nucleotides if n != current_nuc]
            alt_seq[pos] = random.choice(available_nucs)

        alt_seq = ''.join(alt_seq)

        # Assign random labels for testing
        label = random.choice([0, 1])

        entry = GenomicDatasetEntry(
            ref_sequence=ref_seq,
            alt_sequence=alt_seq,
            label=label
        )
        entries.append(entry)

    return GenomicDataset(entries, "synthetic_control")


def load_genomic_dataset_from_file(
    file_path: str,
    format: str = "csv"
) -> GenomicDataset:
    """
    Load a genomic dataset from file

    Args:
        file_path: Path to the dataset file
        format: File format ("csv", "json", "tsv")

    Returns:
        GenomicDataset loaded from file
    """
    path = Path(file_path)

    if format == "csv":
        df = pd.read_csv(path)
    elif format == "tsv":
        df = pd.read_csv(path, sep='\t')
    elif format == "json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported format: {format}")

    # Expected columns: ref_sequence, alt_sequence, label (optional)
    required_cols = ['ref_sequence', 'alt_sequence']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    entries = []
    for _, row in df.iterrows():
        entry = GenomicDatasetEntry(
            ref_sequence=row['ref_sequence'],
            alt_sequence=row['alt_sequence'],
            label=row.get('label', None)
        )
        entries.append(entry)

    return GenomicDataset(entries, path.stem)


def save_genomic_dataset(dataset: GenomicDataset, file_path: str, format: str = "csv"):
    """
    Save a genomic dataset to file

    Args:
        dataset: GenomicDataset to save
        file_path: Path to save the dataset
        format: File format ("csv", "json", "tsv")
    """
    data = []
    for entry in dataset.entries:
        data.append({
            'ref_sequence': entry.ref_sequence,
            'alt_sequence': entry.alt_sequence,
            'label': entry.label
        })

    df = pd.DataFrame(data)

    if format == "csv":
        df.to_csv(file_path, index=False)
    elif format == "tsv":
        df.to_csv(file_path, sep='\t', index=False)
    elif format == "json":
        df.to_json(file_path, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


# Example usage functions
def create_pathogenicity_control_dataset(max_samples: int = 1000) -> GenomicDataset:
    """Create a dataset focused on pathogenic vs benign variants"""
    # This would be implemented to create pairs where:
    # - Reference: benign variant sequence
    # - Alternative: pathogenic variant sequence
    # This allows training control vectors that steer towards pathogenicity
    pass


def create_tissue_specific_control_dataset(tissue_type: str, max_samples: int = 1000) -> GenomicDataset:
    """Create a dataset for tissue-specific expression control"""
    # This would be implemented to create pairs where:
    # - Reference: sequence with low expression in tissue
    # - Alternative: sequence with high expression in tissue
    pass
