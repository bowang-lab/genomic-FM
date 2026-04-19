"""
SuSiE-inf wrapper for statistical fine-mapping.

Provides functions to run SuSiE-inf on GWAS summary statistics
and compute binary credible set labels for training.
"""

import numpy as np
import pandas as pd
import re
from typing import Tuple, List, Dict
from scipy.stats import norm
import susieinf


def run_susie_finemapping(
    z_scores: np.ndarray,
    ld_matrix: np.ndarray,
    n_samples: int,
    L: int = 10,
    coverage: float = 0.95,
    purity: float = 0.5,
    meansq: float = 1.0,
    maxiter: int = 100,
    verbose: bool = False,
) -> Dict:
    """
    Run SuSiE-inf finemapping on GWAS summary statistics.

    Args:
        z_scores: Z-scores for each variant (n_variants,)
        ld_matrix: LD correlation matrix (n_variants x n_variants)
        n_samples: GWAS sample size
        L: Maximum number of causal signals (default: 10)
        coverage: Credible set coverage threshold (default: 0.95)
        purity: Minimum absolute correlation for purity filtering (default: 0.5)
        meansq: Average squared magnitude of phenotype (default: 1.0 for standardized)
        maxiter: Maximum iterations (default: 100)
        verbose: Print progress (default: False)

    Returns:
        Dictionary containing:
            - 'pip': Posterior inclusion probabilities (n_variants,)
            - 'credible_sets': List of credible set indices
            - 'cs_coverages': Coverage of each credible set
            - 'converged': Whether SuSiE converged
    """
    n_variants = len(z_scores)

    # Validate inputs
    if ld_matrix.shape != (n_variants, n_variants):
        raise ValueError(
            f"LD matrix shape {ld_matrix.shape} doesn't match z_scores length {n_variants}"
        )

    # Ensure correct dtypes
    z = np.asarray(z_scores, dtype=np.float64)
    LD = np.asarray(ld_matrix, dtype=np.float64)

    # Run SuSiE-inf
    # API: susie(z, meansq, n, L, LD=None, ...)
    result = susieinf.susie(
        z=z,
        meansq=meansq,
        n=n_samples,
        L=L,
        LD=LD,
        maxiter=maxiter,
        verbose=verbose,
    )

    # Extract PIP matrix (p x L) and aggregate to per-variant PIP
    # PIP for variant i = 1 - product over effects of (1 - PIP[i,l])
    pip_matrix = result['PIP']  # Shape: (n_variants, L)
    pip = 1.0 - np.prod(1.0 - pip_matrix, axis=1)

    # Compute credible sets using susieinf.cred
    credible_sets_raw = susieinf.cred(
        PIP=pip_matrix,
        coverage=coverage,
        purity=purity,
        LD=LD,
        dedup=True,
    )

    # Convert to list of lists (already 0-indexed from susieinf)
    credible_sets = [list(cs) for cs in credible_sets_raw if len(cs) > 0]

    # Coverage is the same for all sets (user-specified)
    cs_coverages = [coverage] * len(credible_sets)

    return {
        'pip': pip,
        'pip_matrix': pip_matrix,
        'credible_sets': credible_sets,
        'cs_coverages': cs_coverages,
        'converged': True,
        'ssq': result.get('ssq', None),
        'sigmasq': result.get('sigmasq', None),
    }


def compute_credible_set_labels(
    pips: np.ndarray,
    credible_sets: List[List[int]],
) -> np.ndarray:
    """
    Compute binary labels based on credible set membership.

    Args:
        pips: Posterior inclusion probabilities (n_variants,)
        credible_sets: List of credible sets, each containing variant indices

    Returns:
        Binary labels: 1 if variant is in any credible set, 0 otherwise
    """
    n_variants = len(pips)
    labels = np.zeros(n_variants, dtype=np.int32)

    for cs in credible_sets:
        for idx in cs:
            if 0 <= idx < n_variants:
                labels[idx] = 1

    return labels


def prepare_sumstats_for_susie(
    gwas_df,
    beta_col: str = 'OR or BETA',
    se_col: str = 'SE',
    ci_col: str = '95% CI (TEXT)',
    pval_col: str = 'P-VALUE',
    pos_col: str = 'CHR_POS',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert GWAS catalog format to z-scores and positions for SuSiE.

    Args:
        gwas_df: DataFrame with GWAS summary statistics
        beta_col: Column name for beta/OR values
        se_col: Column name for standard errors (optional)
        ci_col: Column name for confidence intervals (used if SE not available)
        pval_col: Column name for p-values
        pos_col: Column name for positions

    Returns:
        Tuple of:
            - z_scores: Z-scores for each variant
            - positions: Genomic positions
    """
    df = gwas_df.copy()

    # Get positions
    positions = df[pos_col].values.astype(int)

    # Get betas
    betas = df[beta_col].values.astype(float)

    # Get standard errors
    if se_col in df.columns and df[se_col].notna().any():
        ses = df[se_col].values.astype(float)
    elif ci_col in df.columns:
        # Parse SE from confidence intervals
        ses = df[ci_col].apply(_parse_ci_to_se).values
    else:
        # Estimate SE from p-values and betas
        pvals = df[pval_col].values.astype(float)
        # z = beta / se, and z = qnorm(pval/2)
        # so se = beta / z
        z_from_pval = np.abs(norm.ppf(pvals / 2))
        z_from_pval = np.clip(z_from_pval, 1e-10, 40)  # Avoid inf
        ses = np.abs(betas) / z_from_pval
        ses = np.clip(ses, 1e-10, 100)

    # Handle missing/invalid SE values
    ses = np.nan_to_num(ses, nan=1.0, posinf=1.0, neginf=1.0)
    ses = np.clip(ses, 1e-10, 100)

    # Compute z-scores
    z_scores = betas / ses
    z_scores = np.nan_to_num(z_scores, nan=0.0, posinf=0.0, neginf=0.0)

    return z_scores, positions


def _parse_ci_to_se(ci_text: str) -> float:
    """
    Parse confidence interval text to standard error.

    Assumes 95% CI under normal distribution.

    Args:
        ci_text: CI string like '[1.04-1.16]' or '1.04-1.16'

    Returns:
        Estimated standard error
    """
    if pd.isna(ci_text) or not isinstance(ci_text, str):
        return np.nan

    # Match patterns like [1.04-1.16] or (1.04-1.16) or 1.04-1.16
    match = re.search(r'[\[\(]?(-?\d+\.?\d*)\s*[-–]\s*(-?\d+\.?\d*)[\]\)]?', ci_text)
    if not match:
        return np.nan

    lower = float(match.group(1))
    upper = float(match.group(2))

    # SE = (upper - lower) / (2 * 1.96) for 95% CI
    half_width = (upper - lower) / 2
    se = half_width / 1.96

    return se if se > 0 else np.nan
