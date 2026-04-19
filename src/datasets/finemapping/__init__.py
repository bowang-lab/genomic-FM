"""
Finemapping module for SuSiE-based statistical fine-mapping.

This module provides tools for running SuSiE-inf finemapping on GWAS summary statistics
and generating binary credible set labels for variant prioritization.
"""

from .susie_wrapper import (
    run_susie_finemapping,
    compute_credible_set_labels,
    prepare_sumstats_for_susie,
)
from .ld_matrix import LDMatrix

__all__ = [
    'run_susie_finemapping',
    'compute_credible_set_labels',
    'prepare_sumstats_for_susie',
    'LDMatrix',
]
