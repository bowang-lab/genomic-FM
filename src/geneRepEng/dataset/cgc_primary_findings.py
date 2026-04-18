"""
CGC Primary Findings Dataset Loader for geneRepEng

Loads pathogenic variants from CGC pediatric cardiac patients and extracts
reference/alternative sequence pairs for control vector training.
"""

import random
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from ..extract import GenomicDatasetEntry
from .genomic_bench import GenomicDataset
from ...dataloader.data_wrapper import GenomeSequenceExtractor, create_variant_record


@dataclass
class VariantInfo:
    """Information about a single variant."""
    chrom: str
    pos: int
    ref: str
    alt: str
    variant_id: str
    gene_symbol: Optional[str] = None
    smart_score: Optional[float] = None


@dataclass
class PatientVariantSample:
    """Sample containing all variants for a single patient."""
    patient_id: str
    variants: List[VariantInfo]
    disease_class: Optional[int] = None
    disease_name: Optional[str] = None
    pathogenicity_label: int = 1  # Set by loading function

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
    max_smart_score: float = 50.0,
    csv_path: str = "root/data/unfiltered_variants.csv",
    genome_fa: str = "root/data/hg19.fa",
    seq_length: int = 1024,
    n_samples: Optional[int] = 1000,
    seed: int = 42
) -> GenomicDataset:
    """
    Load CGC variants below SMART threshold as control/reference sequences.

    Uses variants from the same CGC cohort with low pathogenicity scores
    as controls, avoiding batch effects from using external datasets.

    Args:
        max_smart_score: Maximum SMART score threshold (default 50.0, variants below are controls)
        csv_path: Path to unfiltered variants CSV file
        genome_fa: Path to reference genome FASTA
        seq_length: Length of sequence to extract around variant
        n_samples: Maximum number of variants to load (default 1000)
        seed: Random seed for sampling

    Returns:
        GenomicDataset with low-confidence variants as controls
    """
    # Load variants
    df = pd.read_csv(csv_path, low_memory=False)
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

    # Extract sequences (using unfiltered_variants.csv column names)
    entries = []
    skipped = 0

    for idx, row in df_low.iterrows():
        try:
            record = create_variant_record(
                chrom=row['CHROM'],
                pos=int(row['start']),
                ref=row['ref_allele'],
                alt=row['alt_allele'],
                variant_id=idx
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
    max_smart_score: float = 50.0,
    n_samples: int = 1000,
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
        max_smart_score: For CGC source, maximum SMART score threshold (default 50.0)
        n_samples: Maximum number of control variants (default 1000)
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


def create_balanced_control_dataset(
    pathogenic_dataset: GenomicDataset,
    benign_dataset: GenomicDataset = None,
    balance_method: str = "upsample",
    seed: int = 42,
    control_source: str = "cgc",
    max_smart_score: float = 50.0,
    n_controls: int = 1000,
    seq_length: int = 1024
) -> GenomicDataset:
    """
    Create a balanced dataset with pathogenic and control variants.

    Args:
        pathogenic_dataset: Dataset with pathogenic variants
        benign_dataset: Dataset with control variants. If None, loads automatically
                       based on control_source
        balance_method: How to balance ("upsample", "downsample", or "none")
        seed: Random seed
        control_source: Source for controls if benign_dataset is None:
            - "cgc" (default): CGC variants below SMART threshold (same cohort)
            - "clinvar": ClinVar benign variants (external, confirmed benign)
        max_smart_score: For CGC source, SMART score threshold
        n_controls: Number of control variants to load if loading automatically
        seq_length: Sequence length for automatic loading

    Returns:
        Combined balanced dataset
    """
    # Load controls if not provided
    if benign_dataset is None:
        print(f"Loading controls from source: {control_source}")
        benign_dataset = load_controls(
            source=control_source,
            max_smart_score=max_smart_score,
            n_samples=n_controls,
            seq_length=seq_length,
            seed=seed
        )

    n_pathogenic = len(pathogenic_dataset)
    n_benign = len(benign_dataset)

    print(f"Balancing datasets: {n_pathogenic} pathogenic, {n_benign} control")

    random.seed(seed)

    if balance_method == "upsample":
        # Upsample the smaller dataset
        if n_pathogenic < n_benign:
            # Upsample pathogenic
            target_size = n_benign
            pathogenic_entries = list(pathogenic_dataset.entries)
            while len(pathogenic_entries) < target_size:
                pathogenic_entries.extend(
                    random.sample(pathogenic_dataset.entries,
                                min(len(pathogenic_dataset), target_size - len(pathogenic_entries)))
                )
            benign_entries = list(benign_dataset.entries)
        else:
            # Upsample benign
            target_size = n_pathogenic
            benign_entries = list(benign_dataset.entries)
            while len(benign_entries) < target_size:
                benign_entries.extend(
                    random.sample(benign_dataset.entries,
                                min(len(benign_dataset), target_size - len(benign_entries)))
                )
            pathogenic_entries = list(pathogenic_dataset.entries)

    elif balance_method == "downsample":
        # Downsample the larger dataset
        target_size = min(n_pathogenic, n_benign)
        pathogenic_entries = random.sample(pathogenic_dataset.entries, target_size)
        benign_entries = random.sample(benign_dataset.entries, target_size)

    else:  # "none"
        pathogenic_entries = list(pathogenic_dataset.entries)
        benign_entries = list(benign_dataset.entries)

    # Combine entries
    combined_entries = pathogenic_entries + benign_entries

    # Shuffle
    random.shuffle(combined_entries)

    print(f"Created balanced dataset with {len(combined_entries)} total entries")
    print(f"  Pathogenic: {len(pathogenic_entries)}")
    print(f"  Control: {len(benign_entries)}")

    return GenomicDataset(combined_entries, f"balanced_{control_source}_control_dataset")


def load_cgc_by_patient(
    csv_path: str = "root/data/primary_findings_analysis/primary_findings_analysis_results.csv",
    disease_labels_path: str = "root/data/cgc_disease_labels.csv",
    max_variants_per_patient: int = 10,
    min_smart_score: Optional[float] = None,
    pathogenicity_threshold: float = 65.0,
) -> Dict[str, PatientVariantSample]:
    """
    Group CGC variants by patient (CaseID) for multi-variant training.

    Args:
        csv_path: Path to primary findings CSV file
        disease_labels_path: Path to disease labels CSV mapping CaseID to Class
        max_variants_per_patient: Maximum number of variants per patient (default 10)
        min_smart_score: Minimum SMART score threshold (default: None = all variants)
        pathogenicity_threshold: SMART score threshold for pathogenicity (default 65.0)

    Returns:
        Dict mapping patient_id to PatientVariantSample with all their variants
    """
    # Load disease labels
    case_to_disease = load_disease_labels(disease_labels_path)
    disease_to_id = {label: idx for idx, label in enumerate(DISEASE_CLASSES)}

    # Load primary findings
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} CGC primary findings from {csv_path}")

    # Filter by SMART score if specified
    if min_smart_score is not None:
        df = df[df['smart_score'] >= min_smart_score]
        print(f"Filtered to {len(df)} variants with SMART score >= {min_smart_score}")

    # Group variants by patient
    patient_variants: Dict[str, List[VariantInfo]] = {}

    for idx, row in df.iterrows():
        patient_id = str(row['CaseID'])

        variant_info = VariantInfo(
            chrom=row['chrom'],
            pos=int(row['pos']),
            ref=row['ref'],
            alt=row['alt'],
            variant_id=row['variant_key'],
            gene_symbol=row.get('gene_symbol'),
            smart_score=row.get('smart_score')
        )

        if patient_id not in patient_variants:
            patient_variants[patient_id] = []

        # Limit variants per patient
        if len(patient_variants[patient_id]) < max_variants_per_patient:
            patient_variants[patient_id].append(variant_info)

    # Create PatientVariantSample objects
    patient_samples: Dict[str, PatientVariantSample] = {}

    for patient_id, variants in patient_variants.items():
        # Get disease class for this patient
        disease_name = case_to_disease.get(patient_id)
        disease_class = disease_to_id.get(disease_name) if disease_name else None

        # Calculate aggregated pathogenicity
        smart_scores = [v.smart_score for v in variants if v.smart_score is not None]
        if smart_scores:
            max_score = max(smart_scores)
            pathogenicity_label = 1 if max_score >= pathogenicity_threshold else 0
        else:
            pathogenicity_label = 1  # Default for primary findings

        sample = PatientVariantSample(
            patient_id=patient_id,
            variants=variants,
            disease_class=disease_class,
            disease_name=disease_name,
            pathogenicity_label=pathogenicity_label,
        )
        patient_samples[patient_id] = sample

    print(f"Grouped variants into {len(patient_samples)} patients")

    # Print statistics
    variant_counts = [len(s.variants) for s in patient_samples.values()]
    print(f"Variants per patient: min={min(variant_counts)}, max={max(variant_counts)}, "
          f"mean={np.mean(variant_counts):.2f}")

    # Disease class distribution
    disease_counts = {}
    for sample in patient_samples.values():
        if sample.disease_name:
            disease_counts[sample.disease_name] = disease_counts.get(sample.disease_name, 0) + 1
    print(f"Disease distribution: {disease_counts}")

    return patient_samples


def load_cgc_controls_by_patient(
    csv_path: str = "root/data/unfiltered_variants.csv",
    max_smart_score: float = 50.0,
    max_variants_per_patient: int = 10,
    n_patients: Optional[int] = None,
    seed: int = 42
) -> Dict[str, PatientVariantSample]:
    """
    Load CGC variants below SMART threshold as control samples, grouped by patient.

    Args:
        csv_path: Path to unfiltered variants CSV file
        max_smart_score: Maximum SMART score threshold (default 50.0)
        max_variants_per_patient: Maximum variants per patient
        n_patients: Maximum number of patients (default None = all)
        seed: Random seed for patient sampling

    Returns:
        Dict mapping patient_id to PatientVariantSample with control variants
    """
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df)} CGC variants from {csv_path}")

    if 'smart_score' not in df.columns:
        raise ValueError("CSV must contain 'smart_score' column for filtering")

    # Filter to low-confidence variants
    df_low = df[df['smart_score'] < max_smart_score].copy()
    print(f"Found {len(df_low)} variants with SMART score < {max_smart_score}")

    # Group by patient
    patient_variants: Dict[str, List[VariantInfo]] = {}

    for idx, row in df_low.iterrows():
        patient_id = str(row.get('CaseID', idx))

        variant_info = VariantInfo(
            chrom=row['CHROM'],
            pos=int(row['start']),
            ref=row['ref_allele'],
            alt=row['alt_allele'],
            variant_id=str(idx),
            gene_symbol=row.get('gene_symbol'),
            smart_score=row.get('smart_score')
        )

        if patient_id not in patient_variants:
            patient_variants[patient_id] = []

        if len(patient_variants[patient_id]) < max_variants_per_patient:
            patient_variants[patient_id].append(variant_info)

    # Sample patients if requested
    patient_ids = list(patient_variants.keys())
    if n_patients is not None and len(patient_ids) > n_patients:
        random.seed(seed)
        patient_ids = random.sample(patient_ids, n_patients)

    # Create PatientVariantSample objects
    patient_samples: Dict[str, PatientVariantSample] = {}

    for patient_id in patient_ids:
        variants = patient_variants[patient_id]
        sample = PatientVariantSample(
            patient_id=patient_id,
            variants=variants,
            disease_class=None,
            disease_name=None,
            pathogenicity_label=0,
        )
        patient_samples[patient_id] = sample

    print(f"Created {len(patient_samples)} control patient samples")
    return patient_samples


def load_all_variants_by_patient(
    csv_path: str = "root/data/unfiltered_variants.csv",
    primary_findings_path: str = "root/data/primary_findings_analysis/primary_findings_analysis_results.csv",
    disease_labels_path: str = "root/data/cgc_disease_labels.csv",
    max_variants_per_patient: int = 100,
    variant_selection: str = "smart_ranked",  # "smart_ranked", "random", "all"
    include_n_top: int = 10,  # Include top N variants by SMART score
    include_n_random: int = 90,  # Include N random other variants
    seed: int = 42,
) -> Dict[str, PatientVariantSample]:
    """
    Load ALL variants per patient from unfiltered data for multi-variant training.

    Pathogenicity is determined by PRIMARY FINDINGS (not SMART score):
    - Patients WITH primary findings → pathogenic (label=1)
    - Patients WITHOUT primary findings → control (label=0)

    Args:
        csv_path: Path to unfiltered variants CSV file
        primary_findings_path: Path to primary findings CSV (determines pathogenic patients)
        disease_labels_path: Path to disease labels CSV mapping case_id to Class
        max_variants_per_patient: Maximum total variants per patient
        variant_selection: How to select variants ("smart_ranked", "random", "all")
        include_n_top: For smart_ranked, include top N by SMART score
        include_n_random: For smart_ranked, include N random other variants
        seed: Random seed

    Returns:
        Dict mapping patient_id to PatientVariantSample with all their variants
    """
    random.seed(seed)

    # Load primary findings to determine which patients are pathogenic
    primary_df = pd.read_csv(primary_findings_path)
    pathogenic_patients = set(primary_df['CaseID'].astype(str).unique())
    print(f"Loaded {len(pathogenic_patients)} patients with primary findings (pathogenic)")

    # Load disease labels if file exists
    if Path(disease_labels_path).exists():
        case_to_disease = load_disease_labels(disease_labels_path)
        disease_to_id = {label: idx for idx, label in enumerate(DISEASE_CLASSES)}
    else:
        print(f"Disease labels file not found: {disease_labels_path}")
        case_to_disease = {}
        disease_to_id = {}

    # Load variants - only needed columns for efficiency
    print(f"Loading variants from {csv_path}...")
    cols_needed = ['case_id', 'CHROM', 'start', 'ref_allele', 'alt_allele',
                   'gene_symbol', 'smart_score']
    df = pd.read_csv(csv_path, usecols=cols_needed, low_memory=False)
    print(f"Loaded {len(df):,} variants for {df['case_id'].nunique()} patients")

    # Remove duplicate variants (same patient, same position) - keep highest SMART score
    df = df.sort_values('smart_score', ascending=False)
    df = df.drop_duplicates(subset=['case_id', 'CHROM', 'start'], keep='first')
    print(f"After deduplication: {len(df):,} unique variants")

    # Group by patient
    patient_samples: Dict[str, PatientVariantSample] = {}

    for patient_id, patient_df in df.groupby('case_id'):
        patient_id = str(patient_id)

        # Select variants based on strategy
        if variant_selection == "smart_ranked":
            # Sort by SMART score (highest first)
            patient_df = patient_df.sort_values('smart_score', ascending=False)

            # Get top N by SMART score
            top_variants = patient_df.head(include_n_top)

            # Get random sample from the rest
            remaining = patient_df.iloc[include_n_top:]
            if len(remaining) > include_n_random:
                random_variants = remaining.sample(n=include_n_random, random_state=seed)
            else:
                random_variants = remaining

            # Combine and shuffle
            selected_df = pd.concat([top_variants, random_variants])
            selected_df = selected_df.sample(frac=1, random_state=seed)

        elif variant_selection == "random":
            if len(patient_df) > max_variants_per_patient:
                selected_df = patient_df.sample(n=max_variants_per_patient, random_state=seed)
            else:
                selected_df = patient_df

        else:  # "all"
            selected_df = patient_df.head(max_variants_per_patient)

        # Create VariantInfo objects
        variants = []
        for idx, row in selected_df.iterrows():
            variant_info = VariantInfo(
                chrom=str(row['CHROM']),
                pos=int(row['start']),
                ref=str(row['ref_allele']),
                alt=str(row['alt_allele']),
                variant_id=str(idx),
                gene_symbol=row.get('gene_symbol'),
                smart_score=row.get('smart_score')
            )
            variants.append(variant_info)

        if not variants:
            continue

        # Pathogenicity based on PRIMARY FINDINGS (not SMART score)
        pathogenicity_label = 1 if patient_id in pathogenic_patients else 0

        # Get disease class (only for pathogenic patients)
        disease_name = case_to_disease.get(patient_id)
        disease_class = disease_to_id.get(disease_name) if disease_name else None

        sample = PatientVariantSample(
            patient_id=patient_id,
            variants=variants,
            disease_class=disease_class,
            disease_name=disease_name,
            pathogenicity_label=pathogenicity_label,
        )
        patient_samples[patient_id] = sample

    # Print statistics
    print(f"\nCreated {len(patient_samples)} patient samples")
    variant_counts = [len(s.variants) for s in patient_samples.values()]
    print(f"Variants per patient: min={min(variant_counts)}, max={max(variant_counts)}, "
          f"mean={np.mean(variant_counts):.1f}")

    # Label distribution
    n_pathogenic = sum(1 for s in patient_samples.values() if s.pathogenicity_label == 1)
    n_benign = sum(1 for s in patient_samples.values() if s.pathogenicity_label == 0)
    print(f"Patient labels: {n_pathogenic} pathogenic, {n_benign} benign")

    # Disease distribution (only for pathogenic patients)
    disease_counts = {}
    for sample in patient_samples.values():
        if sample.disease_name:
            disease_counts[sample.disease_name] = disease_counts.get(sample.disease_name, 0) + 1
    if disease_counts:
        print(f"Disease distribution: {disease_counts}")

    return patient_samples
