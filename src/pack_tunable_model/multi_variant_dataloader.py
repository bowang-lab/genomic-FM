"""
Multi-Variant Per Patient Dataset and Collator

Handles multiple variants per patient with two processing modes:
1. Local Mode: Variants within same context window (~1kb, same chromosome region)
   - Insert ALL variants into a single ref/alt sequence pair
2. Aggregated Mode: Variants on different chromosomes/genes
   - Process each variant separately, aggregate via attention

Auto-detects mode based on variant positions.
"""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

from ..geneRepEng.dataset.cgc_primary_findings import PatientVariantSample, VariantInfo
from ..sequence_extractor import GenomeSequenceExtractor


@dataclass
class ProcessedPatientSample:
    """Processed patient sample ready for model input."""
    patient_id: str
    mode: str  # "local" or "aggregated"

    # For local mode: single ref/alt pair with all variants
    ref_sequence: Optional[str] = None
    alt_sequence: Optional[str] = None

    # For aggregated mode: list of ref/alt pairs per variant
    variant_ref_sequences: Optional[List[str]] = None
    variant_alt_sequences: Optional[List[str]] = None

    # Labels
    disease_class: Optional[int] = None
    pathogenicity_label: int = 1
    num_variants: int = 1


class MultiVariantPatientDataset(Dataset):
    """
    Dataset for multi-variant per patient training.

    Handles both local and aggregated modes automatically based on variant positions.
    """

    def __init__(
        self,
        patient_samples: List[PatientVariantSample],
        tokenizer: PreTrainedTokenizer,
        genome_extractor: GenomeSequenceExtractor,
        seq_length: int = 1024,
        max_variants: int = 10,
        mode: str = "auto",  # "local", "aggregated", or "auto"
    ):
        """
        Args:
            patient_samples: List of PatientVariantSample objects
            tokenizer: Tokenizer for DNA sequences
            genome_extractor: GenomeSequenceExtractor for sequence extraction
            seq_length: Maximum sequence length
            max_variants: Maximum number of variants per patient
            mode: Processing mode - "local", "aggregated", or "auto"
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.genome_extractor = genome_extractor
        self.seq_length = seq_length
        self.max_variants = max_variants
        self.mode = mode

        # Process all patient samples
        self.processed_samples = []
        self._process_patients(patient_samples)

        print(f"MultiVariantPatientDataset: {len(self.processed_samples)} samples")
        local_count = sum(1 for s in self.processed_samples if s.mode == "local")
        agg_count = sum(1 for s in self.processed_samples if s.mode == "aggregated")
        print(f"  Local mode: {local_count}, Aggregated mode: {agg_count}")

    def _determine_mode(self, sample: PatientVariantSample) -> str:
        """Determine processing mode based on variant positions."""
        if self.mode in ["local", "aggregated"]:
            return self.mode

        # Auto mode: check if variants can fit in local mode
        variants = sample.variants[:self.max_variants]

        if len(variants) <= 1:
            return "local"

        # Check if all on same chromosome
        chroms = set()
        positions = []
        for v in variants:
            chrom = v.chrom if v.chrom.startswith('chr') else f"chr{v.chrom}"
            chroms.add(chrom)
            positions.append(v.pos)

        if len(chroms) > 1:
            return "aggregated"

        # Check span
        variant_span = max(positions) - min(positions)
        if variant_span < self.seq_length:
            return "local"

        return "aggregated"

    def _process_patients(self, patient_samples: List[PatientVariantSample]):
        """Process all patient samples into model-ready format."""
        skipped = 0

        for sample in patient_samples:
            processed = self._process_single_patient(sample)
            if processed is not None:
                self.processed_samples.append(processed)
            else:
                skipped += 1

        if skipped > 0:
            print(f"Skipped {skipped} patients (no valid sequences extracted)")

    def _process_single_patient(self, sample: PatientVariantSample) -> Optional[ProcessedPatientSample]:
        """Process a single patient sample."""
        variants = sample.variants[:self.max_variants]
        if not variants:
            return None

        mode = self._determine_mode(sample)

        if mode == "local":
            return self._process_local_mode(sample, variants)
        else:
            return self._process_aggregated_mode(sample, variants)

    def _process_local_mode(
        self, sample: PatientVariantSample, variants: List[VariantInfo]
    ) -> Optional[ProcessedPatientSample]:
        """Process patient in local mode (all variants in single sequence)."""
        # Convert VariantInfo to dict format for genome_extractor
        variant_dicts = [
            {
                'chrom': v.chrom,
                'pos': v.pos,
                'ref': v.ref,
                'alt': v.alt,
                'variant_id': v.variant_id
            }
            for v in variants
        ]

        ref_seq, alt_seq = self.genome_extractor.extract_multi_variant_sequence(
            variant_dicts, sequence_length=self.seq_length
        )

        if ref_seq is None or alt_seq is None:
            return None

        return ProcessedPatientSample(
            patient_id=sample.patient_id,
            mode="local",
            ref_sequence=ref_seq,
            alt_sequence=alt_seq,
            disease_class=sample.disease_class,
            pathogenicity_label=sample.pathogenicity_label,
            num_variants=len(variants)
        )

    def _process_aggregated_mode(
        self, sample: PatientVariantSample, variants: List[VariantInfo]
    ) -> Optional[ProcessedPatientSample]:
        """Process patient in aggregated mode (separate embeddings per variant)."""
        ref_sequences = []
        alt_sequences = []

        for v in variants:
            variant_dict = {
                'chrom': v.chrom,
                'pos': v.pos,
                'ref': v.ref,
                'alt': v.alt,
                'variant_id': v.variant_id
            }

            ref_seq, alt_seq = self.genome_extractor.extract_multi_variant_sequence(
                [variant_dict], sequence_length=self.seq_length
            )
            if ref_seq is not None and alt_seq is not None:
                ref_sequences.append(ref_seq)
                alt_sequences.append(alt_seq)

        if not ref_sequences:
            return None

        return ProcessedPatientSample(
            patient_id=sample.patient_id,
            mode="aggregated",
            variant_ref_sequences=ref_sequences,
            variant_alt_sequences=alt_sequences,
            disease_class=sample.disease_class,
            pathogenicity_label=sample.pathogenicity_label,
            num_variants=len(ref_sequences)
        )

    def __len__(self):
        return len(self.processed_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample = self.processed_samples[idx]

        if sample.mode == "local":
            return self._get_local_item(sample)
        else:
            return self._get_aggregated_item(sample)

    def _get_local_item(self, sample: ProcessedPatientSample) -> Dict[str, Any]:
        """Get item for local mode sample."""
        # Tokenize single ref/alt pair
        ref_tokens = self.tokenizer(
            sample.ref_sequence,
            return_tensors="pt",
            padding="max_length",
            max_length=self.seq_length,
            truncation=True,
        )
        alt_tokens = self.tokenizer(
            sample.alt_sequence,
            return_tensors="pt",
            padding="max_length",
            max_length=self.seq_length,
            truncation=True,
        )

        item = {
            "mode": "local",
            "ref_input_ids": ref_tokens["input_ids"].squeeze(0),
            "ref_attention_mask": ref_tokens["attention_mask"].squeeze(0),
            "alt_input_ids": alt_tokens["input_ids"].squeeze(0),
            "alt_attention_mask": alt_tokens["attention_mask"].squeeze(0),
            "num_variants": sample.num_variants,
            "pathogenicity_label": sample.pathogenicity_label,
        }

        if sample.disease_class is not None:
            item["disease_class"] = sample.disease_class

        return item

    def _get_aggregated_item(self, sample: ProcessedPatientSample) -> Dict[str, Any]:
        """Get item for aggregated mode sample."""
        num_variants = len(sample.variant_ref_sequences)

        # Tokenize all variant sequences
        ref_input_ids_list = []
        ref_attention_mask_list = []
        alt_input_ids_list = []
        alt_attention_mask_list = []

        for ref_seq, alt_seq in zip(sample.variant_ref_sequences, sample.variant_alt_sequences):
            ref_tokens = self.tokenizer(
                ref_seq,
                return_tensors="pt",
                padding="max_length",
                max_length=self.seq_length,
                truncation=True,
            )
            alt_tokens = self.tokenizer(
                alt_seq,
                return_tensors="pt",
                padding="max_length",
                max_length=self.seq_length,
                truncation=True,
            )

            ref_input_ids_list.append(ref_tokens["input_ids"].squeeze(0))
            ref_attention_mask_list.append(ref_tokens["attention_mask"].squeeze(0))
            alt_input_ids_list.append(alt_tokens["input_ids"].squeeze(0))
            alt_attention_mask_list.append(alt_tokens["attention_mask"].squeeze(0))

        # Stack into tensors
        item = {
            "mode": "aggregated",
            "ref_input_ids": torch.stack(ref_input_ids_list),  # (num_variants, seq_len)
            "ref_attention_mask": torch.stack(ref_attention_mask_list),
            "alt_input_ids": torch.stack(alt_input_ids_list),
            "alt_attention_mask": torch.stack(alt_attention_mask_list),
            "num_variants": num_variants,
            "pathogenicity_label": sample.pathogenicity_label,
        }

        if sample.disease_class is not None:
            item["disease_class"] = sample.disease_class

        return item


class MultiVariantDataCollator:
    """
    Collator for multi-variant patient data.

    Handles batching with variable variant counts and separate collation
    for local vs aggregated mode samples.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_variants: int = 10,
        num_disease_classes: int = 4,
    ):
        self.tokenizer = tokenizer
        self.max_variants = max_variants
        self.num_disease_classes = num_disease_classes
        self.pad_token_id = tokenizer.pad_token_id or 0

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of instances."""
        # Separate by mode
        local_instances = [i for i in instances if i["mode"] == "local"]
        aggregated_instances = [i for i in instances if i["mode"] == "aggregated"]

        batch = {}

        # Process local mode instances
        if local_instances:
            local_batch = self._collate_local(local_instances)
            batch.update({f"local_{k}": v for k, v in local_batch.items()})
            batch["has_local"] = True
            batch["local_batch_size"] = len(local_instances)
        else:
            batch["has_local"] = False
            batch["local_batch_size"] = 0

        # Process aggregated mode instances
        if aggregated_instances:
            agg_batch = self._collate_aggregated(aggregated_instances)
            batch.update({f"agg_{k}": v for k, v in agg_batch.items()})
            batch["has_aggregated"] = True
            batch["agg_batch_size"] = len(aggregated_instances)
        else:
            batch["has_aggregated"] = False
            batch["agg_batch_size"] = 0

        return batch

    def _collate_local(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate local mode instances."""
        ref_input_ids = torch.stack([i["ref_input_ids"] for i in instances])
        ref_attention_mask = torch.stack([i["ref_attention_mask"] for i in instances])
        alt_input_ids = torch.stack([i["alt_input_ids"] for i in instances])
        alt_attention_mask = torch.stack([i["alt_attention_mask"] for i in instances])
        num_variants = torch.tensor([i["num_variants"] for i in instances], dtype=torch.long)
        pathogenicity_labels = torch.tensor(
            [i["pathogenicity_label"] for i in instances], dtype=torch.long
        )

        batch = {
            "ref_input_ids": ref_input_ids,
            "ref_attention_mask": ref_attention_mask,
            "alt_input_ids": alt_input_ids,
            "alt_attention_mask": alt_attention_mask,
            "num_variants": num_variants,
            "pathogenicity_labels": pathogenicity_labels,
        }

        # Disease class (optional)
        if "disease_class" in instances[0]:
            disease_classes = torch.tensor(
                [i.get("disease_class", -100) for i in instances], dtype=torch.long
            )
            batch["disease_labels"] = disease_classes

        return batch

    def _collate_aggregated(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate aggregated mode instances with variable variant counts."""
        batch_size = len(instances)
        max_variants_in_batch = max(i["num_variants"] for i in instances)
        seq_length = instances[0]["ref_input_ids"].shape[-1]

        # Initialize padded tensors
        ref_input_ids = torch.full(
            (batch_size, max_variants_in_batch, seq_length),
            self.pad_token_id,
            dtype=torch.long
        )
        ref_attention_mask = torch.zeros(
            (batch_size, max_variants_in_batch, seq_length),
            dtype=torch.long
        )
        alt_input_ids = torch.full(
            (batch_size, max_variants_in_batch, seq_length),
            self.pad_token_id,
            dtype=torch.long
        )
        alt_attention_mask = torch.zeros(
            (batch_size, max_variants_in_batch, seq_length),
            dtype=torch.long
        )

        # Variant mask (which variant positions are valid)
        variant_mask = torch.zeros((batch_size, max_variants_in_batch), dtype=torch.bool)

        num_variants_list = []
        pathogenicity_labels = []
        disease_labels = []

        for i, instance in enumerate(instances):
            n_vars = instance["num_variants"]
            num_variants_list.append(n_vars)

            ref_input_ids[i, :n_vars] = instance["ref_input_ids"]
            ref_attention_mask[i, :n_vars] = instance["ref_attention_mask"]
            alt_input_ids[i, :n_vars] = instance["alt_input_ids"]
            alt_attention_mask[i, :n_vars] = instance["alt_attention_mask"]
            variant_mask[i, :n_vars] = True

            pathogenicity_labels.append(instance["pathogenicity_label"])
            disease_labels.append(instance.get("disease_class", -100))

        batch = {
            "ref_input_ids": ref_input_ids,
            "ref_attention_mask": ref_attention_mask,
            "alt_input_ids": alt_input_ids,
            "alt_attention_mask": alt_attention_mask,
            "variant_mask": variant_mask,
            "num_variants": torch.tensor(num_variants_list, dtype=torch.long),
            "pathogenicity_labels": torch.tensor(pathogenicity_labels, dtype=torch.long),
        }

        if any(d != -100 for d in disease_labels):
            batch["disease_labels"] = torch.tensor(disease_labels, dtype=torch.long)

        return batch
