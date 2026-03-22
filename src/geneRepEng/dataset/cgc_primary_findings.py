"""
CGC Primary Findings Dataset Loader for geneRepEng

Loads pathogenic variants from CGC pediatric cardiac patients and extracts
reference/alternative sequence pairs for control vector training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from ..extract import GenomicDatasetEntry
from .genomic_bench import GenomicDataset
from ...dataloader.data_wrapper import GenomeSequenceExtractor, create_variant_record

# Disease classes for cardiac conditions
DISEASE_CLASSES = ['Aortopathy', 'Cardiomyopathy', 'Arrhythmia', 'Structural defect']


def load_cgc_primary_findings(
    csv_path: str = "root/data/primary_findings_analysis/primary_findings_analysis_results.csv",
    genome_fa: str = "root/data/hg19.fa",
    seq_length: int = 1024,
    min_smart_score: Optional[float] = None,
    max_variants: Optional[int] = None
) -> GenomicDataset:
    """
    Load CGC primary findings (pathogenic variants) and extract sequences.

    Args:
        csv_path: Path to primary findings CSV file
        genome_fa: Path to reference genome FASTA
        seq_length: Length of sequence to extract around variant
        min_smart_score: Minimum SMART score threshold (default: None = all variants)
        max_variants: Maximum number of variants to load (default: None = all)

    Returns:
        GenomicDataset with ref/alt sequence pairs
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} CGC primary findings from {csv_path}")

    # Filter by SMART score if specified
    if min_smart_score is not None:
        df = df[df['smart_score'] >= min_smart_score]
        print(f"Filtered to {len(df)} variants with SMART score >= {min_smart_score}")

    # Limit number of variants if specified
    if max_variants is not None:
        df = df.head(max_variants)
        print(f"Limited to {max_variants} variants")

    # Initialize sequence extractor
    genome_extractor = GenomeSequenceExtractor(genome_fa)
    print(f"Initialized genome extractor with {genome_fa}")

    # Extract sequences for each variant
    entries = []
    skipped = 0

    for idx, row in df.iterrows():
        try:
            # Create variant record
            record = create_variant_record(
                chrom=row['chrom'],
                pos=int(row['pos']),
                ref=row['ref'],
                alt=row['alt'],
                variant_id=row['variant_key']
            )

            # Extract reference and alternative sequences
            ref_seq, alt_seq = genome_extractor.extract_sequence_from_record(
                record,
                sequence_length=seq_length
            )

            if ref_seq is None or alt_seq is None:
                skipped += 1
                continue

            # Create GenomicDatasetEntry
            # Label: 1 for pathogenic (these are all pathogenic findings)
            entry = GenomicDatasetEntry(
                ref_sequence=ref_seq,
                alt_sequence=alt_seq,
                label=1  # Pathogenic
            )
            entries.append(entry)

        except Exception as e:
            print(f"Warning: Skipped variant {row['variant_key']}: {e}")
            skipped += 1
            continue

    print(f"Successfully extracted {len(entries)} sequence pairs")
    if skipped > 0:
        print(f"Skipped {skipped} variants due to extraction errors")

    return GenomicDataset(entries, "cgc_primary_findings_pathogenic")


def load_cgc_by_gene(
    gene_symbol: str,
    csv_path: str = "root/data/primary_findings_analysis/primary_findings_analysis_results.csv",
    **kwargs
) -> GenomicDataset:
    """
    Load CGC variants for a specific gene.

    Args:
        gene_symbol: Gene symbol to filter by (e.g., "FLT4", "NOTCH1")
        csv_path: Path to primary findings CSV
        **kwargs: Additional arguments passed to load_cgc_primary_findings

    Returns:
        GenomicDataset for the specified gene
    """
    df = pd.read_csv(csv_path)

    # Filter to specific gene
    gene_df = df[df['gene_symbol'] == gene_symbol]

    if len(gene_df) == 0:
        print(f"Warning: No variants found for gene {gene_symbol}")
        return GenomicDataset([], f"cgc_{gene_symbol}_empty")

    # Save to temporary file
    temp_path = f"/tmp/cgc_{gene_symbol}.csv"
    gene_df.to_csv(temp_path, index=False)

    # Load using main function
    dataset = load_cgc_primary_findings(csv_path=temp_path, **kwargs)
    dataset.name = f"cgc_{gene_symbol}_pathogenic"

    return dataset


def get_cgc_gene_list(
    csv_path: str = "root/data/primary_findings_analysis/primary_findings_analysis_results.csv"
) -> List[str]:
    """
    Get list of unique genes in CGC primary findings.

    Args:
        csv_path: Path to primary findings CSV

    Returns:
        List of unique gene symbols
    """
    df = pd.read_csv(csv_path)
    return sorted(df['gene_symbol'].unique().tolist())


def get_cgc_case_list(
    csv_path: str = "root/data/primary_findings_analysis/primary_findings_analysis_results.csv"
) -> List[str]:
    """
    Get list of unique case IDs in CGC primary findings.

    Args:
        csv_path: Path to primary findings CSV

    Returns:
        List of unique case IDs
    """
    df = pd.read_csv(csv_path)
    return sorted(df['CaseID'].unique().tolist())


def load_disease_labels(
    disease_labels_path: str = "root/data/cgc_disease_labels.csv"
) -> Dict[str, str]:
    """
    Load disease labels mapping CaseID to disease Class.

    Args:
        disease_labels_path: Path to disease labels CSV

    Returns:
        Dict mapping CaseID to disease Class
    """
    df = pd.read_csv(disease_labels_path)
    return dict(zip(df['CaseID'].astype(str), df['Class']))


def load_cgc_by_disease_class(
    disease_class: str,
    csv_path: str = "root/data/primary_findings_analysis/primary_findings_analysis_results.csv",
    disease_labels_path: str = "root/data/cgc_disease_labels.csv",
    genome_fa: str = "root/data/hg19.fa",
    seq_length: int = 1024,
) -> GenomicDataset:
    """
    Load CGC primary findings filtered by disease class.

    Args:
        disease_class: One of 'Aortopathy', 'Cardiomyopathy', 'Arrhythmia', 'Structural defect'
        csv_path: Path to primary findings CSV file
        disease_labels_path: Path to disease labels CSV mapping CaseID to Class
        genome_fa: Path to reference genome FASTA
        seq_length: Length of sequence to extract around variant

    Returns:
        GenomicDataset with ref/alt sequence pairs for the specified disease
    """
    if disease_class not in DISEASE_CLASSES:
        raise ValueError(f"disease_class must be one of {DISEASE_CLASSES}, got '{disease_class}'")

    # Load disease labels
    case_to_disease = load_disease_labels(disease_labels_path)
    print(f"Loaded disease labels for {len(case_to_disease)} cases")

    # Load primary findings
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} CGC primary findings from {csv_path}")

    # Filter by disease class
    df['CaseID_str'] = df['CaseID'].astype(str)
    df['disease_class'] = df['CaseID_str'].map(case_to_disease)
    df_filtered = df[df['disease_class'] == disease_class]
    print(f"Filtered to {len(df_filtered)} variants for disease class '{disease_class}'")

    if len(df_filtered) == 0:
        print(f"Warning: No variants found for disease class {disease_class}")
        return GenomicDataset([], f"cgc_{disease_class}_pathogenic")

    # Initialize sequence extractor
    genome_extractor = GenomeSequenceExtractor(genome_fa)
    print(f"Initialized genome extractor with {genome_fa}")

    # Extract sequences for each variant
    entries = []
    skipped = 0

    for idx, row in df_filtered.iterrows():
        try:
            # Create variant record
            record = create_variant_record(
                chrom=row['chrom'],
                pos=int(row['pos']),
                ref=row['ref'],
                alt=row['alt'],
                variant_id=row['variant_key']
            )

            # Extract reference and alternative sequences
            ref_seq, alt_seq = genome_extractor.extract_sequence_from_record(
                record,
                sequence_length=seq_length
            )

            if ref_seq is None or alt_seq is None:
                skipped += 1
                continue

            # Create GenomicDatasetEntry
            entry = GenomicDatasetEntry(
                ref_sequence=ref_seq,
                alt_sequence=alt_seq,
                label=1  # Pathogenic
            )
            entries.append(entry)

        except Exception as e:
            print(f"Warning: Skipped variant {row['variant_key']}: {e}")
            skipped += 1
            continue

    print(f"Successfully extracted {len(entries)} sequence pairs for {disease_class}")
    if skipped > 0:
        print(f"Skipped {skipped} variants due to extraction errors")

    return GenomicDataset(entries, f"cgc_{disease_class}_pathogenic")


def get_disease_specific_genes(
    disease_class: str,
    csv_path: str = "root/data/primary_findings_analysis/primary_findings_analysis_results.csv",
    disease_labels_path: str = "root/data/cgc_disease_labels.csv",
) -> List[str]:
    """
    Get unique gene symbols for a specific disease class.

    Args:
        disease_class: One of 'Aortopathy', 'Cardiomyopathy', 'Arrhythmia', 'Structural defect'
        csv_path: Path to primary findings CSV
        disease_labels_path: Path to disease labels CSV

    Returns:
        List of unique gene symbols for the disease class
    """
    if disease_class not in DISEASE_CLASSES:
        raise ValueError(f"disease_class must be one of {DISEASE_CLASSES}, got '{disease_class}'")

    # Load disease labels
    case_to_disease = load_disease_labels(disease_labels_path)

    # Load primary findings
    df = pd.read_csv(csv_path)

    # Filter by disease class
    df['CaseID_str'] = df['CaseID'].astype(str)
    df['disease_class'] = df['CaseID_str'].map(case_to_disease)
    df_filtered = df[df['disease_class'] == disease_class]

    # Get unique gene symbols
    genes = df_filtered['gene_symbol'].dropna().unique().tolist()
    return sorted(genes)


def load_cgc_with_smart_scores(
    disease_class: str,
    csv_path: str = "root/data/primary_findings_analysis/primary_findings_analysis_results.csv",
    disease_labels_path: str = "root/data/cgc_disease_labels.csv",
    genome_fa: str = "root/data/hg19.fa",
    seq_length: int = 1024,
) -> Tuple[GenomicDataset, np.ndarray]:
    """
    Load CGC dataset for a disease class paired with SMART pathogenicity scores.

    Args:
        disease_class: One of 'Aortopathy', 'Cardiomyopathy', 'Arrhythmia', 'Structural defect'
        csv_path: Path to primary findings CSV file
        disease_labels_path: Path to disease labels CSV mapping CaseID to Class
        genome_fa: Path to reference genome FASTA
        seq_length: Length of sequence to extract around variant

    Returns:
        Tuple of (GenomicDataset, smart_scores array)
    """
    if disease_class not in DISEASE_CLASSES:
        raise ValueError(f"disease_class must be one of {DISEASE_CLASSES}, got '{disease_class}'")

    # Load disease labels
    case_to_disease = load_disease_labels(disease_labels_path)

    # Load primary findings
    df = pd.read_csv(csv_path)

    # Filter by disease class
    df['CaseID_str'] = df['CaseID'].astype(str)
    df['disease_class'] = df['CaseID_str'].map(case_to_disease)
    df_filtered = df[df['disease_class'] == disease_class].copy()

    if len(df_filtered) == 0:
        return GenomicDataset([], f"cgc_{disease_class}_pathogenic"), np.array([])

    # Initialize sequence extractor
    genome_extractor = GenomeSequenceExtractor(genome_fa)

    # Extract sequences and smart scores
    entries = []
    smart_scores = []

    for idx, row in df_filtered.iterrows():
        try:
            record = create_variant_record(
                chrom=row['chrom'],
                pos=int(row['pos']),
                ref=row['ref'],
                alt=row['alt'],
                variant_id=row['variant_key']
            )

            ref_seq, alt_seq = genome_extractor.extract_sequence_from_record(
                record,
                sequence_length=seq_length
            )

            if ref_seq is None or alt_seq is None:
                continue

            entry = GenomicDatasetEntry(
                ref_sequence=ref_seq,
                alt_sequence=alt_seq,
                label=1
            )
            entries.append(entry)
            smart_scores.append(row.get('smart_score', 0.0))

        except Exception:
            continue

    return GenomicDataset(entries, f"cgc_{disease_class}_pathogenic"), np.array(smart_scores)


def load_cgc_low_confidence_controls(
    max_smart_score: float = 0.5,
    csv_path: str = "root/data/primary_findings_analysis/primary_findings_analysis_results.csv",
    genome_fa: str = "root/data/hg19.fa",
    seq_length: int = 1024,
    n_samples: Optional[int] = None,
    seed: int = 42
) -> GenomicDataset:
    """
    Load CGC variants below SMART threshold as control/reference sequences.

    Uses variants from the same CGC cohort with low pathogenicity scores
    as controls, avoiding batch effects from using external datasets.

    Args:
        max_smart_score: Maximum SMART score threshold (variants below this are controls)
        csv_path: Path to primary findings CSV file
        genome_fa: Path to reference genome FASTA
        seq_length: Length of sequence to extract around variant
        n_samples: Maximum number of variants to load (None = all below threshold)
        seed: Random seed for sampling

    Returns:
        GenomicDataset with low-confidence variants as controls
    """
    import random

    # Load primary findings
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} CGC variants from {csv_path}")

    # Check if smart_score column exists
    if 'smart_score' not in df.columns:
        raise ValueError("CSV must contain 'smart_score' column for filtering")

    # Filter to low-confidence variants
    df_low = df[df['smart_score'] < max_smart_score].copy()
    print(f"Found {len(df_low)} variants with SMART score < {max_smart_score}")

    if len(df_low) == 0:
        print(f"Warning: No variants below SMART threshold {max_smart_score}")
        return GenomicDataset([], "cgc_low_confidence_controls")

    # Shuffle and limit samples
    random.seed(seed)
    indices = list(df_low.index)
    random.shuffle(indices)
    if n_samples is not None:
        indices = indices[:n_samples]
    df_low = df_low.loc[indices]

    # Initialize sequence extractor
    genome_extractor = GenomeSequenceExtractor(genome_fa)

    # Extract sequences
    entries = []
    skipped = 0

    for idx, row in df_low.iterrows():
        try:
            record = create_variant_record(
                chrom=row['chrom'],
                pos=int(row['pos']),
                ref=row['ref'],
                alt=row['alt'],
                variant_id=row['variant_key']
            )

            ref_seq, alt_seq = genome_extractor.extract_sequence_from_record(
                record,
                sequence_length=seq_length
            )

            if ref_seq is None or alt_seq is None:
                skipped += 1
                continue

            entry = GenomicDatasetEntry(
                ref_sequence=ref_seq,
                alt_sequence=alt_seq,
                label=0  # Control/low-confidence
            )
            entries.append(entry)

        except Exception as e:
            skipped += 1
            continue

    print(f"Extracted {len(entries)} low-confidence control variants")
    if skipped > 0:
        print(f"Skipped {skipped} variants due to extraction errors")

    return GenomicDataset(entries, f"cgc_low_confidence_smart_lt_{max_smart_score}")


def load_controls(
    source: str = "cgc",
    max_smart_score: float = 0.5,
    n_samples: int = 500,
    seq_length: int = 1024,
    seed: int = 42,
    **kwargs
) -> GenomicDataset:
    """
    Load control variants for training control vectors.

    This is the recommended entry point for loading controls. By default uses
    CGC variants below SMART threshold (same cohort, fewer batch effects).

    Args:
        source: Control source - "cgc" (default) or "clinvar"
            - "cgc": CGC variants below SMART threshold (same cohort)
            - "clinvar": ClinVar benign variants (external, confirmed benign)
        max_smart_score: For CGC source, maximum SMART score threshold
        n_samples: Maximum number of control variants
        seq_length: Sequence length to extract
        seed: Random seed
        **kwargs: Additional arguments passed to loader

    Returns:
        GenomicDataset with control variants
    """
    if source == "cgc":
        return load_cgc_low_confidence_controls(
            max_smart_score=max_smart_score,
            n_samples=n_samples,
            seq_length=seq_length,
            seed=seed,
            **kwargs
        )
    elif source == "clinvar":
        from .clinvar_control import load_cardiac_benign_variants
        return load_cardiac_benign_variants(
            n_samples=n_samples,
            seq_length=seq_length,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown control source: {source}. Use 'cgc' or 'clinvar'")
