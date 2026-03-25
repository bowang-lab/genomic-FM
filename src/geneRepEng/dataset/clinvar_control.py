"""
ClinVar Control Dataset Loader for geneRepEng

Loads benign variants from ClinVar (optionally filtered by disease) to use as
control/reference sequences for training control vectors.
"""

import random
from typing import List, Optional
from pathlib import Path
from ..extract import GenomicDatasetEntry
from .genomic_bench import GenomicDataset
from ...dataloader.data_wrapper import ClinVarDataWrapper, set_disease_subset_from_file, GenomeSequenceExtractor, create_variant_record
from ...datasets.clinvar import load_clinvar


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


def load_benign_by_genes(
    gene_list: List[str],
    n_samples: int = 1000,
    seq_length: int = 1024,
    seed: int = 42,
    genome_fa: str = "root/data/hg19.fa",
) -> GenomicDataset:
    """
    Load benign ClinVar variants filtered to specific genes.

    This function loads benign variants from ClinVar and filters them to only
    include variants in the specified gene list. Useful for creating gene-matched
    control datasets.

    Args:
        gene_list: List of gene symbols to filter by (e.g., ["FLT4", "NOTCH1"])
        n_samples: Maximum number of benign variants to sample
        seq_length: Length of sequences to extract
        seed: Random seed for reproducibility
        genome_fa: Path to reference genome FASTA

    Returns:
        GenomicDataset with benign variants filtered by genes
    """
    print(f"Loading ClinVar benign variants for {len(gene_list)} genes...")

    # Convert gene list to set for faster lookup
    gene_set = set(g.upper() for g in gene_list)

    # Load raw ClinVar records
    clinvar_vcf_path = load_clinvar.download_file(vcf_file_path='./root/data/clinvar_20250409.vcf')
    records = load_clinvar.read_vcf(clinvar_vcf_path, num_records=1000000, all_records=True)
    print(f"Loaded {len(records)} total ClinVar records")

    # Initialize genome extractor
    genome_extractor = GenomeSequenceExtractor(genome_fa)

    # Filter to benign variants in specified genes
    random.seed(seed)
    random.shuffle(records)

    benign_entries = []
    genes_found = set()

    for record in records:
        # Check if benign
        clnsig = record.get('CLNSIG', [])
        if not clnsig or clnsig[0] not in ['Benign', 'Likely_benign', 'Benign/Likely_benign']:
            continue

        # Check if in gene list
        gene_info = record.get('GENEINFO', '')
        if gene_info:
            # GENEINFO format is typically "GENE:ID" or "GENE:ID|GENE2:ID2"
            if isinstance(gene_info, list):
                gene_info = gene_info[0] if gene_info else ''
            gene_names = [g.split(':')[0].upper() for g in str(gene_info).split('|')]
            matching_genes = gene_set.intersection(gene_names)
            if not matching_genes:
                continue
            genes_found.update(matching_genes)
        else:
            continue

        # Extract sequences
        try:
            ref_seq, alt_seq = genome_extractor.extract_sequence_from_record(
                record,
                sequence_length=seq_length
            )

            if ref_seq is None or alt_seq is None:
                continue

            entry = GenomicDatasetEntry(
                ref_sequence=ref_seq,
                alt_sequence=alt_seq,
                label=0  # Benign
            )
            benign_entries.append(entry)

            if len(benign_entries) >= n_samples:
                break

        except Exception as e:
            continue

    print(f"Extracted {len(benign_entries)} benign variants from {len(genes_found)} genes")
    print(f"Genes with variants: {sorted(genes_found)}")

    return GenomicDataset(benign_entries, f"clinvar_benign_gene_matched")
