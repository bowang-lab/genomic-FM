"""
CGC Primary Findings Dataset Loader for geneRepEng

Loads pathogenic variants from CGC pediatric cardiac patients and extracts
reference/alternative sequence pairs for control vector training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from ..extract import GenomicDatasetEntry
from .genomic_bench import GenomicDataset
from ...dataloader.data_wrapper import GenomeSequenceExtractor, create_variant_record


def load_cgc_primary_findings(
    csv_path: str = "root/data/primary_findings_analysis/primary_findings_analysis_results.csv",
    genome_fa: str = "root/data/hg19.fa",
    seq_length: int = 512,
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
