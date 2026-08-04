"""OLIDA oligogenic disease data loaders."""

from .load_olida import (
    get_olida,
    get_variant_combinations,
    get_gene_pairs,
    load_and_process_negative_pairs,
)
from .cardiac_olida import (
    load_filtered_olida_paired,
    filter_olida_by_criteria,
    load_keywords_from_file,
    load_genes_from_file,
    get_olida_filter_stats,
)

__all__ = [
    # Base OLIDA loaders
    'get_olida',
    'get_variant_combinations',
    'get_gene_pairs',
    'load_and_process_negative_pairs',
    # Filtered loaders
    'load_filtered_olida_paired',
    'filter_olida_by_criteria',
    'load_keywords_from_file',
    'load_genes_from_file',
    'get_olida_filter_stats',
]
