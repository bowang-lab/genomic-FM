#!/usr/bin/env python3
"""
Cardiac-Filtered OLIDA Loader

Filters OLIDA oligogenic disease pairs based on configurable disease keywords
and gene lists. Supports loading filters from external files.
"""

import pandas as pd
import random
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Union
from tqdm import tqdm

from .load_olida import (
    get_variant_combinations,
    load_and_process_negative_pairs,
)
from src.sequence_extractor import GenomeSequenceExtractor


# Default paths for filter files (can be overridden)
DEFAULT_DISEASE_KEYWORDS_FILE = Path("./root/data/cardiac_disease_keywords.txt")
DEFAULT_GENE_LIST_FILE = Path("./root/data/cardiac_genes.txt")


def load_keywords_from_file(filepath: Union[str, Path]) -> List[str]:
    """
    Load keywords/terms from a text file (one per line).

    Args:
        filepath: Path to text file with one keyword per line

    Returns:
        List of lowercase keywords
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return []

    keywords = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                keywords.append(line.lower())
    return keywords


def load_genes_from_file(filepath: Union[str, Path]) -> Set[str]:
    """
    Load gene symbols from a text file or CSV.

    Supports:
    - Text file with one gene per line
    - CSV with 'Gene' column (like GeneListCGC.csv)

    Args:
        filepath: Path to gene list file

    Returns:
        Set of uppercase gene symbols
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return set()

    genes = set()

    # Check if CSV format
    if filepath.suffix.lower() == '.csv':
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        # Look for gene column
        gene_col = None
        for col in ['Gene', 'gene', 'GENE', 'gene_symbol', 'Symbol']:
            if col in df.columns:
                gene_col = col
                break
        if gene_col:
            genes = set(df[gene_col].dropna().astype(str).str.upper().str.strip())
    else:
        # Text file format
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    genes.add(line.upper())

    return genes


def matches_keywords(text: str, keywords: List[str]) -> bool:
    """
    Check if text contains any of the keywords.

    Args:
        text: Text to search
        keywords: List of keywords to match (lowercase)

    Returns:
        True if any keyword is found in text
    """
    if not text or not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def filter_olida_by_criteria(
    variant_combinations: List[Dict],
    disease_keywords: Optional[List[str]] = None,
    gene_set: Optional[Set[str]] = None,
    min_final_meta: int = 1,
    require_disease_match: bool = True,
    require_gene_match: bool = False,
) -> List[Dict]:
    """
    Filter OLIDA variant combinations by configurable criteria.

    Args:
        variant_combinations: Raw OLIDA variant combinations
        disease_keywords: List of disease keywords to filter by (lowercase)
        gene_set: Set of gene symbols to filter by (uppercase)
        min_final_meta: Minimum FINALmeta score (default 1)
        require_disease_match: If True, disease must match keywords
        require_gene_match: If True, at least one gene must be in gene_set

    Returns:
        Filtered list of variant combinations
    """
    filtered = []

    for combo in variant_combinations:
        # Check FINALmeta score
        if combo.get('FINALmeta', 0) < min_final_meta:
            continue

        # Check disease keywords if required
        if require_disease_match and disease_keywords:
            disease = combo.get('Disease', '')
            if not matches_keywords(disease, disease_keywords):
                continue

        # Check gene set if required
        if require_gene_match and gene_set:
            genes = []
            for var_key in ['Variant_1', 'Variant_2']:
                if var_key in combo and 'Gene' in combo[var_key]:
                    genes.append(combo[var_key]['Gene'].upper())
            if not any(g in gene_set for g in genes):
                continue

        filtered.append(combo)

    return filtered


def load_filtered_olida_paired(
    seq_length: int = 1024,
    disease_keywords: Optional[List[str]] = None,
    disease_keywords_file: Optional[Union[str, Path]] = None,
    gene_set: Optional[Set[str]] = None,
    gene_list_file: Optional[Union[str, Path]] = None,
    min_final_meta: int = 1,
    require_disease_match: bool = True,
    require_gene_match: bool = False,
    negative_ratio: float = 1.0,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> List[Tuple[Dict, int]]:
    """
    Load filtered OLIDA data in paired variant format.

    Filter sources (in order of priority):
    1. Explicit disease_keywords/gene_set parameters
    2. disease_keywords_file/gene_list_file paths
    3. Default file paths if they exist

    Args:
        seq_length: Sequence length for extraction
        disease_keywords: List of disease keywords (lowercase)
        disease_keywords_file: Path to file with disease keywords
        gene_set: Set of gene symbols (uppercase)
        gene_list_file: Path to file with gene list
        min_final_meta: Minimum FINALmeta score
        require_disease_match: Filter by disease keywords
        require_gene_match: Also require matching genes
        negative_ratio: Ratio of negative to positive samples
        limit: Maximum number of samples (None for all)
        verbose: Print progress information

    Returns:
        List of (variant_dict, label) tuples where variant_dict contains:
        - variant1_ref, variant1_alt: Sequences for variant 1
        - variant2_ref, variant2_alt: Sequences for variant 2
        - gene1, gene2: Gene symbols
        - disease: Disease name
    """
    # Load disease keywords
    if disease_keywords is None:
        if disease_keywords_file:
            disease_keywords = load_keywords_from_file(disease_keywords_file)
        elif DEFAULT_DISEASE_KEYWORDS_FILE.exists():
            disease_keywords = load_keywords_from_file(DEFAULT_DISEASE_KEYWORDS_FILE)
        else:
            disease_keywords = []

    # Load gene set
    if gene_set is None:
        if gene_list_file:
            gene_set = load_genes_from_file(gene_list_file)
        elif DEFAULT_GENE_LIST_FILE.exists():
            gene_set = load_genes_from_file(DEFAULT_GENE_LIST_FILE)
        else:
            gene_set = set()

    if verbose:
        print(f"Loaded {len(disease_keywords)} disease keywords, {len(gene_set)} genes for filtering")

    # Load and filter OLIDA data
    all_combinations = get_variant_combinations()
    filtered_combinations = filter_olida_by_criteria(
        all_combinations,
        disease_keywords=disease_keywords,
        gene_set=gene_set,
        min_final_meta=min_final_meta,
        require_disease_match=require_disease_match and len(disease_keywords) > 0,
        require_gene_match=require_gene_match and len(gene_set) > 0,
    )

    if verbose:
        print(f"OLIDA filter: {len(filtered_combinations)}/{len(all_combinations)} "
              f"({100*len(filtered_combinations)/max(len(all_combinations),1):.1f}%) pairs retained")

    # Extract sequences for positive pairs
    genome_extractor = GenomeSequenceExtractor()
    data = []

    iterator = tqdm(filtered_combinations, desc="Processing OLIDA") if verbose else filtered_combinations
    for variant_combo in iterator:
        try:
            variants = {}
            genes = []

            for i, variant_key in enumerate(['Variant_1', 'Variant_2'], 1):
                var_data = variant_combo[variant_key]
                required = ['Chromosome', 'Genomic_Position_Hg38', 'Ref_Allele', 'Alt_Allele']

                # Check required fields
                if not all(key in var_data for key in required):
                    raise ValueError(f"Missing keys in {variant_key}")
                if any(var_data[key] == "N.A." or
                       (isinstance(var_data[key], float) and pd.isna(var_data[key]))
                       for key in required):
                    raise ValueError(f"Invalid values in {variant_key}")

                record = {
                    'Chromosome': var_data['Chromosome'],
                    'Position': int(var_data['Genomic_Position_Hg38']),
                    'Reference Base': var_data['Ref_Allele'],
                    'Alternate Base': var_data['Alt_Allele'],
                    'ID': variant_combo['OLIDA_ID']
                }

                ref_seq, alt_seq = genome_extractor.extract_sequence_from_record(record, seq_length)
                variants[f'variant{i}_ref'] = ref_seq
                variants[f'variant{i}_alt'] = alt_seq
                genes.append(var_data.get('Gene', 'Unknown'))

            variants['gene1'] = genes[0]
            variants['gene2'] = genes[1]
            variants['disease'] = variant_combo['Disease']
            variants['olida_id'] = variant_combo['OLIDA_ID']

            data.append((variants, 1))  # Label 1 = oligogenic

        except Exception:
            continue

    if verbose:
        print(f"Extracted {len(data)} positive oligogenic pairs")

    # Add negative samples from 1000 Genome Project
    if negative_ratio > 0 and len(data) > 0:
        n_negatives = int(len(data) * negative_ratio)
        negative_data = load_and_process_negative_pairs(
            Seq_length=seq_length,
            num_records=n_negatives,
            paired=True,
        )

        for neg_item, label in negative_data:
            if isinstance(neg_item, dict):
                data.append((neg_item, 0))

        if verbose:
            print(f"Added {len(negative_data)} negative pairs from 1000GP")

    random.shuffle(data)

    if limit:
        data = data[:limit]

    if verbose:
        n_pos = sum(1 for _, l in data if l == 1)
        n_neg = sum(1 for _, l in data if l == 0)
        print(f"Final dataset: {len(data)} pairs ({n_pos} positive, {n_neg} negative)")

    return data


def get_olida_filter_stats(
    disease_keywords: Optional[List[str]] = None,
    disease_keywords_file: Optional[Union[str, Path]] = None,
    gene_set: Optional[Set[str]] = None,
    gene_list_file: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    Get statistics about filtered OLIDA data.

    Args:
        disease_keywords: List of disease keywords
        disease_keywords_file: Path to disease keywords file
        gene_set: Set of gene symbols
        gene_list_file: Path to gene list file

    Returns:
        Dict with filtering statistics and distributions
    """
    # Load filters
    if disease_keywords is None and disease_keywords_file:
        disease_keywords = load_keywords_from_file(disease_keywords_file)
    if gene_set is None and gene_list_file:
        gene_set = load_genes_from_file(gene_list_file)

    all_combinations = get_variant_combinations()
    filtered_combinations = filter_olida_by_criteria(
        all_combinations,
        disease_keywords=disease_keywords or [],
        gene_set=gene_set or set(),
        require_disease_match=bool(disease_keywords),
        require_gene_match=bool(gene_set),
    )

    # Count diseases
    disease_counts = {}
    for combo in filtered_combinations:
        disease = combo.get('Disease', 'Unknown')
        disease_counts[disease] = disease_counts.get(disease, 0) + 1

    # Count genes
    gene_counts = {}
    for combo in filtered_combinations:
        for var_key in ['Variant_1', 'Variant_2']:
            if var_key in combo and 'Gene' in combo[var_key]:
                gene = combo[var_key]['Gene']
                gene_counts[gene] = gene_counts.get(gene, 0) + 1

    return {
        'total_olida': len(all_combinations),
        'filtered': len(filtered_combinations),
        'filter_rate': len(filtered_combinations) / max(len(all_combinations), 1),
        'n_disease_keywords': len(disease_keywords) if disease_keywords else 0,
        'n_genes_in_filter': len(gene_set) if gene_set else 0,
        'disease_distribution': dict(sorted(disease_counts.items(), key=lambda x: -x[1])),
        'gene_distribution': dict(sorted(gene_counts.items(), key=lambda x: -x[1])),
    }
