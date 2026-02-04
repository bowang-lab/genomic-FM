"""
ClinVar Control Dataset Loader for geneRepEng

Loads benign variants from ClinVar (optionally filtered by disease) to use as
control/reference sequences for training control vectors.
"""

import random
import numpy as np
from typing import List, Optional
from pathlib import Path
from ..extract import GenomicDatasetEntry
from .genomic_bench import GenomicDataset
from ...dataloader.data_wrapper import ClinVarDataWrapper, set_disease_subset_from_file


def load_clinvar_benign_variants(
    disease_subset_file: Optional[str] = None,
    seq_length: int = 1024,
    n_samples: int = 1000,
    seed: int = 42
) -> GenomicDataset:
    """
    Load benign variants from ClinVar for use as control/reference sequences.

    Args:
        disease_subset_file: Path to file with disease names to filter by
                            (e.g., 'heart_related_diseases.txt')
        seq_length: Length of sequences to extract
        n_samples: Number of benign variants to sample
        seed: Random seed for reproducibility

    Returns:
        GenomicDataset with benign variants as ref/alt pairs
    """
    print(f"Loading ClinVar benign variants...")
    if disease_subset_file:
        print(f"  Filtering by diseases in: {disease_subset_file}")
        set_disease_subset_from_file(disease_subset_file)

    # Load raw ClinVar data directly (no tokenization needed)
    clinvar_wrapper = ClinVarDataWrapper(all_records=False, use_default_dir=False, num_records=1000000)
    data = clinvar_wrapper.get_data(Seq_length=seq_length, target='CLNSIG', disease_subset=False)

    print(f"Loaded {len(data)} total ClinVar variants")

    # Filter to benign variants only
    benign_entries = []
    random.seed(seed)
    random.shuffle(data)

    for item in data:
        # item format: [[ref, alt, variant_type], label]
        ref_seq, alt_seq, variant_type = item[0]
        label = item[1]

        # Check if variant is benign
        if label == 'Benign':
            entry = GenomicDatasetEntry(
                ref_sequence=ref_seq,
                alt_sequence=alt_seq,
                label=0  # Benign
            )
            benign_entries.append(entry)

            # Stop if we have enough samples
            if len(benign_entries) >= n_samples:
                break

    print(f"Extracted {len(benign_entries)} benign variants")

    # Shuffle and limit to requested number
    random.seed(seed)
    random.shuffle(benign_entries)
    benign_entries = benign_entries[:n_samples]

    dataset_name = "clinvar_benign"
    if disease_subset_file:
        subset_name = Path(disease_subset_file).stem
        dataset_name = f"clinvar_benign_{subset_name}"

    return GenomicDataset(benign_entries, dataset_name)


def load_cardiac_benign_variants(
    n_samples: int = 1000,
    seq_length: int = 1024,
    seed: int = 42
) -> GenomicDataset:
    """
    Load benign cardiac variants from ClinVar.

    Convenience function that filters ClinVar to cardiac-related diseases.

    Args:
        n_samples: Number of benign variants to sample
        seq_length: Sequence length
        seed: Random seed

    Returns:
        GenomicDataset with cardiac benign variants
    """
    return load_clinvar_benign_variants(
        disease_subset_file="heart_related_diseases.txt",
        seq_length=seq_length,
        n_samples=n_samples,
        seed=seed
    )


def create_balanced_control_dataset(
    pathogenic_dataset: GenomicDataset,
    benign_dataset: GenomicDataset,
    balance_method: str = "upsample",
    seed: int = 42
) -> GenomicDataset:
    """
    Create a balanced dataset with equal numbers of pathogenic and benign variants.

    Args:
        pathogenic_dataset: Dataset with pathogenic variants
        benign_dataset: Dataset with benign variants
        balance_method: How to balance ("upsample", "downsample", or "none")
        seed: Random seed

    Returns:
        Combined balanced dataset
    """
    n_pathogenic = len(pathogenic_dataset)
    n_benign = len(benign_dataset)

    print(f"Balancing datasets: {n_pathogenic} pathogenic, {n_benign} benign")

    random.seed(seed)

    if balance_method == "upsample":
        # Upsample the smaller dataset
        if n_pathogenic < n_benign:
            # Upsample pathogenic
            target_size = n_benign
            pathogenic_entries = pathogenic_dataset.entries
            while len(pathogenic_entries) < target_size:
                pathogenic_entries.extend(
                    random.sample(pathogenic_dataset.entries,
                                min(len(pathogenic_dataset), target_size - len(pathogenic_entries)))
                )
            benign_entries = benign_dataset.entries
        else:
            # Upsample benign
            target_size = n_pathogenic
            benign_entries = benign_dataset.entries
            while len(benign_entries) < target_size:
                benign_entries.extend(
                    random.sample(benign_dataset.entries,
                                min(len(benign_dataset), target_size - len(benign_entries)))
                )
            pathogenic_entries = pathogenic_dataset.entries

    elif balance_method == "downsample":
        # Downsample the larger dataset
        target_size = min(n_pathogenic, n_benign)
        pathogenic_entries = random.sample(pathogenic_dataset.entries, target_size)
        benign_entries = random.sample(benign_dataset.entries, target_size)

    else:  # "none"
        pathogenic_entries = pathogenic_dataset.entries
        benign_entries = benign_dataset.entries

    # Combine entries
    combined_entries = pathogenic_entries + benign_entries

    # Shuffle
    random.shuffle(combined_entries)

    print(f"Created balanced dataset with {len(combined_entries)} total entries")
    print(f"  Pathogenic: {len(pathogenic_entries)}")
    print(f"  Benign: {len(benign_entries)}")

    return GenomicDataset(combined_entries, "balanced_control_dataset")
