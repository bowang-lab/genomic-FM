#!/usr/bin/env python3
"""
Generate ClinVar grouped dataset statistics for privacy analysis.
"""

import sys
sys.path.insert(0, '/Users/vallijahsubasri/genomic-FM')

from collections import Counter
from src.dataloader.data_wrapper import ClinVarGroupedDataWrapper


def print_stats(name, data, group_to_id, stats):
    """Print detailed statistics for a dataset."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Basic counts
    print(f"\nTotal samples: {len(data)}")
    print(f"Total groups: {len(group_to_id)}")

    # Label distribution
    labels = [item[1] for item in data]
    label_counts = Counter(labels)
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        pct = 100 * count / len(data)
        label_name = "Benign" if label == 0 else "Pathogenic"
        print(f"  {label_name} ({label}): {count} ({pct:.1f}%)")

    # Group size distribution
    group_ids = [item[2] for item in data]
    group_sizes = Counter(group_ids)
    sizes = list(group_sizes.values())

    print(f"\nGroup size statistics:")
    print(f"  Min group size: {min(sizes)}")
    print(f"  Max group size: {max(sizes)}")
    print(f"  Mean group size: {sum(sizes)/len(sizes):.1f}")

    # Distribution of group sizes
    size_dist = Counter(sizes)
    print(f"\nGroup size distribution (size -> # of groups):")
    for size in sorted(size_dist.keys())[:15]:  # Show first 15
        print(f"  {size} variants: {size_dist[size]} groups")
    if len(size_dist) > 15:
        print(f"  ... ({len(size_dist) - 15} more size categories)")

    # Sample groups
    id_to_group = {v: k for k, v in group_to_id.items()}
    print(f"\nSample groups (first 10):")
    for gid in list(group_to_id.values())[:10]:
        group_name = id_to_group[gid]
        count = group_sizes[gid]
        print(f"  {group_name}: {count} variants")

    return len(data), len(group_to_id)


def main():
    print("="*60)
    print("  ClinVar Grouped Dataset Statistics")
    print("="*60)

    results = {}

    # Test different grouping modes
    grouping_modes = [
        ('gene', 5, 50),           # All genes with 5-50 variants
        ('cardiac_gene', 5, 50),   # CGC cardiac genes only (647 genes)
        ('hcm_gene', 5, 50),       # HCM genes only (168 genes)
        ('cardiac_panel', 5, 500), # Group by cardiac category (4 groups)
    ]

    for grouping, min_var, max_var in grouping_modes:
        print(f"\n\nLoading {grouping} grouping (min={min_var}, max={max_var})...")

        try:
            wrapper = ClinVarGroupedDataWrapper(
                num_records=100000,
                all_records=True,
                use_default_dir=False,
                min_variants_per_gene=min_var,
                max_variants_per_gene=max_var,
                grouping=grouping,
            )

            # Get pathogenicity data
            data, group_to_id, stats = wrapper.get_data(
                Seq_length=1024,
                balance_classes=True,
                target='CLNSIG',
            )

            n_samples, n_groups = print_stats(
                f"{grouping.upper()} Grouping (CLNSIG target)",
                data, group_to_id, stats
            )
            results[f"{grouping}_CLNSIG"] = (n_samples, n_groups)

        except Exception as e:
            print(f"Error loading {grouping}: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"\n{'Grouping':<25} {'Samples':>10} {'Groups':>10}")
    print("-"*45)
    for key, (samples, groups) in results.items():
        print(f"{key:<25} {samples:>10} {groups:>10}")

    print("\n\nDone!")


if __name__ == "__main__":
    main()
