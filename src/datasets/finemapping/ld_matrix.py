"""
LD matrix computation for finemapping and privacy research.

Supports two modes:
1. Pre-computed HDF5 LD blocks from PRS-CSx (recommended, faster)
2. On-the-fly computation from 1000 Genomes VCF files
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import os
import h5py
from numba import jit, prange


class LDMatrix:
    """
    Load pre-computed LD matrices from PRS-CSx HDF5 reference.

    This uses the pre-computed LD blocks from:
    https://github.com/getian107/PRScsx

    Much faster than computing from VCF since LD is pre-computed.
    """

    def __init__(
        self,
        ld_dir: str = './root/data/1000G/ldblk_1kg_eur',
        panel: str = 'EUR',
    ):
        """
        Initialize LD matrix loader.

        Args:
            ld_dir: Directory containing ldblk_1kg_chr*.hdf5 files
            panel: Population panel name (for reference)
        """
        self.ld_dir = ld_dir
        self.panel = panel
        self._snpinfo = None
        self._snp_to_idx = None

    def _load_snpinfo(self) -> None:
        """Load SNP info file mapping rsIDs to positions."""
        if self._snpinfo is not None:
            return

        snpinfo_path = os.path.join(self.ld_dir, 'snpinfo_1kg_hm3')
        if not os.path.exists(snpinfo_path):
            raise FileNotFoundError(f"SNP info file not found: {snpinfo_path}")

        snps = []
        with open(snpinfo_path) as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                snps.append({
                    'chrom': parts[0],
                    'rsid': parts[1],
                    'pos': int(parts[2]),
                    'a1': parts[3],
                    'a2': parts[4],
                    'maf': float(parts[5]),
                })

        self._snpinfo = snps
        # Build index: (chrom, pos) -> index in snpinfo
        self._snp_to_idx = {
            (s['chrom'], s['pos']): i for i, s in enumerate(snps)
        }
        # Also index by rsID
        self._rsid_to_idx = {s['rsid']: i for i, s in enumerate(snps)}

    def _get_hdf5_path(self, chrom: str) -> str:
        """Get HDF5 path for a specific chromosome."""
        chrom_num = str(chrom).replace('chr', '')
        return os.path.join(self.ld_dir, f'ldblk_1kg_chr{chrom_num}.hdf5')

    def get_snps_in_region(
        self,
        chrom: str,
        start: int,
        end: int,
    ) -> List[Dict]:
        """Get HapMap3 SNPs in a genomic region."""
        self._load_snpinfo()
        chrom_str = str(chrom).replace('chr', '')

        snps = []
        for s in self._snpinfo:
            if s['chrom'] == chrom_str and start <= s['pos'] <= end:
                snps.append(s)

        return sorted(snps, key=lambda x: x['pos'])

    def compute(
        self,
        chrom: str,
        start: int,
        end: int,
        min_maf: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get pre-computed LD matrix for a genomic region.

        Args:
            chrom: Chromosome (e.g., '11' or 'chr11')
            start: Start position (1-based)
            end: End position (1-based)
            min_maf: Minimum MAF filter (applied to HapMap3 SNPs)

        Returns:
            Tuple of:
                - R: (n_variants x n_variants) LD correlation matrix
                - positions: (n_variants,) genomic positions
                - variant_ids: (n_variants,) rsIDs
        """
        self._load_snpinfo()
        chrom_str = str(chrom).replace('chr', '')

        # Get SNPs in region
        region_snps = [
            s for s in self._snpinfo
            if s['chrom'] == chrom_str and start <= s['pos'] <= end and s['maf'] >= min_maf
        ]

        if len(region_snps) == 0:
            return np.array([[]]), np.array([]), np.array([])

        # Get rsIDs and positions
        rsids = [s['rsid'] for s in region_snps]
        positions = np.array([s['pos'] for s in region_snps], dtype=np.int64)

        # Load LD from HDF5
        hdf5_path = self._get_hdf5_path(chrom)
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"LD file not found: {hdf5_path}")

        rsid_set = set(rsids)
        R = self._load_ld_for_snps(hdf5_path, rsid_set, rsids)

        return R, positions, np.array(rsids)

    def _load_ld_for_snps(
        self,
        hdf5_path: str,
        rsid_set: set,
        ordered_rsids: List[str],
    ) -> np.ndarray:
        """Load LD matrix for specific SNPs from HDF5 blocks."""
        n = len(ordered_rsids)
        R = np.eye(n, dtype=np.float32)

        rsid_to_out_idx = {rsid: i for i, rsid in enumerate(ordered_rsids)}

        with h5py.File(hdf5_path, 'r') as f:
            for blk_name in f.keys():
                blk = f[blk_name]
                snplist = [s.decode() if isinstance(s, bytes) else s for s in blk['snplist'][:]]

                # Find which SNPs in this block are in our query
                blk_indices = []
                out_indices = []
                for i, rsid in enumerate(snplist):
                    if rsid in rsid_set:
                        blk_indices.append(i)
                        out_indices.append(rsid_to_out_idx[rsid])

                if len(blk_indices) < 2:
                    continue

                # Extract LD submatrix
                ldblk = blk['ldblk'][:]
                blk_indices = np.array(blk_indices)
                out_indices = np.array(out_indices)

                # Copy LD values
                for i, (bi, oi) in enumerate(zip(blk_indices, out_indices)):
                    for j, (bj, oj) in enumerate(zip(blk_indices, out_indices)):
                        if i <= j:
                            R[oi, oj] = ldblk[bi, bj]
                            R[oj, oi] = ldblk[bi, bj]

        return R

    def compute_ld_groups(
        self,
        chrom: str,
        start: int,
        end: int,
        r2_threshold: float = 0.2,
        min_maf: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute LD-based group assignments for variants in a region.

        Groups variants into connected components where any pair with r² >= threshold
        is in the same group. This captures variants in linkage disequilibrium.

        Args:
            chrom: Chromosome
            start: Start position
            end: End position
            r2_threshold: r² threshold for grouping (default 0.2)
            min_maf: Minimum MAF filter

        Returns:
            Tuple of:
                - group_ids: (n_variants,) group assignment for each variant
                - positions: (n_variants,) genomic positions
                - variant_ids: (n_variants,) rsIDs
        """
        R, positions, variant_ids = self.compute(chrom, start, end, min_maf)

        if len(positions) == 0:
            return np.array([]), np.array([]), np.array([])

        # Convert to r² (squared correlations)
        R2 = R ** 2

        # Find connected components using union-find
        n = len(positions)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Connect variants with r² >= threshold
        for i in range(n):
            for j in range(i + 1, n):
                if R2[i, j] >= r2_threshold:
                    union(i, j)

        # Assign contiguous group IDs
        root_to_group = {}
        group_ids = np.zeros(n, dtype=np.int64)
        next_group = 0

        for i in range(n):
            root = find(i)
            if root not in root_to_group:
                root_to_group[root] = next_group
                next_group += 1
            group_ids[i] = root_to_group[root]

        return group_ids, positions, variant_ids

    def get_ld_for_rsids(
        self,
        rsids: List[str],
        chrom: str,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get LD matrix for a specific list of rsIDs.

        Args:
            rsids: List of rsIDs to get LD for
            chrom: Chromosome (for HDF5 file lookup)

        Returns:
            Tuple of:
                - R: LD correlation matrix for matched SNPs
                - matched_rsids: rsIDs that were found
        """
        self._load_snpinfo()

        # Filter to valid rsIDs
        valid_rsids = [r for r in rsids if r in self._rsid_to_idx]
        if len(valid_rsids) == 0:
            return np.array([[]]), []

        rsid_set = set(valid_rsids)
        R = self._load_ld_for_snps(
            self._get_hdf5_path(chrom),
            rsid_set,
            valid_rsids
        )

        return R, valid_rsids


class LDMatrixVCF:
    """
    Compute LD matrices from raw 1000 Genomes VCF files.

    Use this when you need LD for SNPs not in HapMap3, or need
    to compute LD from a different reference panel.
    """

    def __init__(
        self,
        vcf_dir: str = './root/data/1000G',
        panel: str = 'EUR',
        samples_file: str = None,
    ):
        """
        Initialize LD matrix computer from VCF.

        Args:
            vcf_dir: Directory containing per-chromosome VCF files
            panel: Population panel name
            samples_file: Path to file with sample IDs to include
        """
        self.vcf_dir = vcf_dir
        self.panel = panel
        self.samples = None

        if samples_file is None:
            samples_file = os.path.join(vcf_dir, f'{panel}_samples.txt')
        if os.path.exists(samples_file):
            with open(samples_file) as f:
                self.samples = [line.strip() for line in f if line.strip()]

    def _get_vcf_path(self, chrom: str) -> str:
        """Get VCF path for a specific chromosome."""
        chrom_num = chrom.replace('chr', '')
        vcf_path = os.path.join(self.vcf_dir, f'chr{chrom_num}.vcf.gz')
        if not os.path.exists(vcf_path):
            vcf_path = os.path.join(
                self.vcf_dir,
                f'ALL.chr{chrom_num}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz'
            )
        return vcf_path

    def compute(
        self,
        chrom: str,
        start: int,
        end: int,
        min_maf: float = 0.01,
        max_missing: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute LD correlation matrix from VCF for genomic region.

        Args:
            chrom: Chromosome
            start: Start position
            end: End position
            min_maf: Minimum MAF filter
            max_missing: Maximum missing rate

        Returns:
            Tuple of (R, positions, variant_ids)
        """
        from cyvcf2 import VCF

        vcf_path = self._get_vcf_path(chrom)
        if not os.path.exists(vcf_path):
            raise FileNotFoundError(f"VCF not found: {vcf_path}")

        if self.samples:
            vcf = VCF(vcf_path, gts012=True, samples=self.samples)
        else:
            vcf = VCF(vcf_path, gts012=True)

        chrom_query = chrom.replace('chr', '')
        region = f"{chrom_query}:{start}-{end}"

        geno_list = []
        pos_list = []
        id_list = []

        variants = list(vcf(region))
        if len(variants) == 0:
            region = f"chr{chrom_query}:{start}-{end}"
            variants = list(vcf(region))

        for var in variants:
            if len(var.REF) != 1 or len(var.ALT) != 1 or len(var.ALT[0]) != 1:
                continue

            gt = var.gt_types
            missing_rate = np.mean(gt == 3)
            if missing_rate > max_missing:
                continue

            gt_float = gt.astype(np.float32)
            gt_float[gt == 3] = np.nan

            valid_gt = gt_float[~np.isnan(gt_float)]
            if len(valid_gt) == 0:
                continue
            maf = np.mean(valid_gt) / 2
            maf = min(maf, 1 - maf)
            if maf < min_maf:
                continue

            mean_gt = np.nanmean(gt_float)
            gt_float = np.nan_to_num(gt_float, nan=mean_gt)

            geno_list.append(gt_float)
            pos_list.append(var.POS)
            id_list.append(var.ID if var.ID else f"{var.CHROM}:{var.POS}:{var.REF}:{var.ALT[0]}")

        vcf.close()

        if len(geno_list) == 0:
            return np.array([[]]), np.array([]), np.array([])

        G = np.stack(geno_list, axis=1)
        positions = np.array(pos_list, dtype=np.int64)
        variant_ids = np.array(id_list)

        R = _numba_correlation(G.astype(np.float32))

        return R, positions, variant_ids


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

    means = np.zeros(n_variants, dtype=np.float32)
    stds = np.zeros(n_variants, dtype=np.float32)

    for j in range(n_variants):
        means[j] = np.mean(G[:, j])
        stds[j] = np.std(G[:, j])
        if stds[j] < 1e-10:
            stds[j] = 1e-10

    for i in prange(n_variants):
        xi = (G[:, i] - means[i]) / stds[i]
        for j in range(i, n_variants):
            xj = (G[:, j] - means[j]) / stds[j]
            r = np.sum(xi * xj) / n_samples
            R[i, j] = r
            R[j, i] = r

    for i in range(n_variants):
        R[i, i] = 1.0

    return R
