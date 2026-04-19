"""
LD matrix computation for SuSiE finemapping.

Computes full pairwise LD correlation matrices from 1000 Genomes VCF files.
"""

import numpy as np
from typing import Tuple
import os
from cyvcf2 import VCF
from numba import jit, prange


class LDMatrixComputer:
    """
    Compute LD correlation matrices from 1000G VCF.

    Follows genomic-FM DataWrapper pattern for consistent interface.
    """

    def __init__(
        self,
        vcf_path: str = './root/data/1000G/EUR.vcf.gz',
        panel: str = 'EUR',
    ):
        """
        Initialize LD matrix computer.

        Args:
            vcf_path: Path to population-specific VCF file
            panel: Population panel name (for reference only)
        """
        self.vcf_path = vcf_path
        self.panel = panel

    def compute(
        self,
        chrom: str,
        start: int,
        end: int,
        min_maf: float = 0.01,
        max_missing: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute full pairwise LD correlation matrix for genomic region.

        Args:
            chrom: Chromosome (e.g., '11' or 'chr11')
            start: Start position (1-based)
            end: End position (1-based)
            min_maf: Minimum minor allele frequency filter
            max_missing: Maximum fraction of missing genotypes allowed

        Returns:
            Tuple of:
                - R: (n_variants x n_variants) LD correlation matrix
                - positions: (n_variants,) genomic positions
                - variant_ids: (n_variants,) variant IDs if available
        """
        if not os.path.exists(self.vcf_path):
            raise FileNotFoundError(
                f"VCF file not found: {self.vcf_path}\n"
                f"Please download 1000 Genomes VCF for {self.panel} population."
            )

        # Load genotypes from VCF
        G, positions, variant_ids = self._load_genotypes(
            chrom, start, end, min_maf, max_missing
        )

        if G.shape[1] == 0:
            return np.array([[]]), np.array([]), np.array([])

        # Compute correlation matrix (Numba-accelerated)
        R = self._compute_correlation_matrix(G)

        return R, positions, variant_ids

    def _load_genotypes(
        self,
        chrom: str,
        start: int,
        end: int,
        min_maf: float = 0.01,
        max_missing: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load 0/1/2 dosage matrix from VCF.

        Args:
            chrom: Chromosome
            start: Start position
            end: End position
            min_maf: Minimum MAF filter
            max_missing: Maximum missing rate

        Returns:
            Tuple of:
                - G: (n_samples, n_variants) genotype dosage matrix
                - positions: (n_variants,) positions
                - variant_ids: (n_variants,) variant IDs
        """
        vcf = VCF(self.vcf_path, gts012=True)

        # Normalize chromosome name
        chrom_query = chrom.replace('chr', '')
        region = f"{chrom_query}:{start}-{end}"

        geno_list = []
        pos_list = []
        id_list = []

        # Try querying the region
        variants = list(vcf(region))

        # If no results, try with chr prefix
        if len(variants) == 0:
            region = f"chr{chrom_query}:{start}-{end}"
            variants = list(vcf(region))

        for var in variants:
            # Keep only biallelic SNPs
            if len(var.REF) != 1 or len(var.ALT) != 1 or len(var.ALT[0]) != 1:
                continue

            gt = var.gt_types  # 0=HOM_REF, 1=HET, 2=HOM_ALT, 3=UNKNOWN

            # Check missing rate
            missing_rate = np.mean(gt == 3)
            if missing_rate > max_missing:
                continue

            # Convert to float and handle missing
            gt_float = gt.astype(np.float32)
            gt_float[gt == 3] = np.nan

            # Check MAF
            valid_gt = gt_float[~np.isnan(gt_float)]
            if len(valid_gt) == 0:
                continue
            maf = np.mean(valid_gt) / 2
            maf = min(maf, 1 - maf)
            if maf < min_maf:
                continue

            # Impute missing with mean
            mean_gt = np.nanmean(gt_float)
            gt_float = np.nan_to_num(gt_float, nan=mean_gt)

            geno_list.append(gt_float)
            pos_list.append(var.POS)
            id_list.append(var.ID if var.ID else f"{var.CHROM}:{var.POS}:{var.REF}:{var.ALT[0]}")

        vcf.close()

        if len(geno_list) == 0:
            return np.array([]).reshape(0, 0), np.array([]), np.array([])

        G = np.stack(geno_list, axis=1)  # (n_samples, n_variants)
        positions = np.array(pos_list, dtype=np.int64)
        variant_ids = np.array(id_list)

        return G, positions, variant_ids

    @staticmethod
    def _compute_correlation_matrix(G: np.ndarray) -> np.ndarray:
        """
        Compute LD correlation matrix using Numba JIT acceleration.

        Args:
            G: (n_samples, n_variants) genotype matrix

        Returns:
            R: (n_variants, n_variants) correlation matrix
        """
        return _numba_correlation(G.astype(np.float32))

    def compute_for_variants(
        self,
        positions: np.ndarray,
        chrom: str,
        window: int = 500000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute LD matrix for a specific set of variant positions.

        Args:
            positions: Array of variant positions to include
            chrom: Chromosome
            window: Window around min/max positions to include

        Returns:
            Tuple of (R, matched_positions, indices_in_input)
        """
        if len(positions) == 0:
            return np.array([[]]), np.array([]), np.array([])

        start = int(np.min(positions)) - window
        end = int(np.max(positions)) + window
        start = max(1, start)

        R, vcf_positions, _ = self.compute(chrom, start, end)

        if len(vcf_positions) == 0:
            return np.array([[]]), np.array([]), np.array([])

        # Match input positions to VCF positions
        matched_indices_input = []
        matched_indices_vcf = []

        pos_to_vcf_idx = {pos: idx for idx, pos in enumerate(vcf_positions)}

        for i, pos in enumerate(positions):
            if pos in pos_to_vcf_idx:
                matched_indices_input.append(i)
                matched_indices_vcf.append(pos_to_vcf_idx[pos])

        if len(matched_indices_vcf) == 0:
            return np.array([[]]), np.array([]), np.array([])

        # Subset R matrix to matched variants
        matched_indices_vcf = np.array(matched_indices_vcf)
        R_subset = R[np.ix_(matched_indices_vcf, matched_indices_vcf)]
        matched_positions = vcf_positions[matched_indices_vcf]

        return R_subset, matched_positions, np.array(matched_indices_input)


@jit(nopython=True, parallel=True, cache=True)
def _numba_correlation(G: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated LD correlation computation.

    Args:
        G: (n_samples, n_variants) genotype matrix (float32)

    Returns:
        R: (n_variants, n_variants) correlation matrix
    """
    n_samples, n_variants = G.shape
    R = np.zeros((n_variants, n_variants), dtype=np.float32)

    # Pre-compute means and stds
    means = np.zeros(n_variants, dtype=np.float32)
    stds = np.zeros(n_variants, dtype=np.float32)

    for j in range(n_variants):
        means[j] = np.mean(G[:, j])
        stds[j] = np.std(G[:, j])
        if stds[j] < 1e-10:
            stds[j] = 1e-10

    # Compute correlations in parallel
    for i in prange(n_variants):
        xi = (G[:, i] - means[i]) / stds[i]
        for j in range(i, n_variants):
            xj = (G[:, j] - means[j]) / stds[j]
            r = np.sum(xi * xj) / n_samples
            R[i, j] = r
            R[j, i] = r

    # Ensure diagonal is exactly 1.0
    for i in range(n_variants):
        R[i, i] = 1.0

    return R
