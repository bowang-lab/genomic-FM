#!/usr/bin/env python3
"""
Temporary script to load MAVES data and save as JSONL in the exact format used by training.
"""

import json
import argparse
import numpy as np
from src.dataloader.data_wrapper import MAVEDataWrapper


def main():
    parser = argparse.ArgumentParser(description="Load MAVES data and save as JSONL")
    parser.add_argument("--output", default="maves_data.jsonl", help="Output JSONL file")
    parser.add_argument("--num_records", type=int, default=2000, help="Number of records to load")
    parser.add_argument("--all_records", action="store_true", help="Load all records")
    parser.add_argument("--max_studies", type=int, default=None, help="Maximum number of studies to process")
    parser.add_argument("--seq_length", type=int, default=None, help="Sequence length (None for unlimited)")
    parser.add_argument("--experimental_methods", nargs="*", help="Filter by experimental methods (e.g., DMS)")
    parser.add_argument("--filter_genes", nargs="*", help="Filter by genes")
    parser.add_argument("--coding_only", type=bool, default=None, help="Filter coding sequences only")

    args = parser.parse_args()

    print(f"Loading MAVES data...")
    print(f"Records: {'all' if args.all_records else args.num_records}")
    print(f"Sequence length: {'unlimited' if args.seq_length is None else args.seq_length}")
    if args.experimental_methods:
        print(f"Experimental methods: {args.experimental_methods}")
    if args.filter_genes:
        print(f"Genes: {args.filter_genes}")

    # Load MAVES data using the same wrapper as training
    mave_wrapper = MAVEDataWrapper(
        num_records=args.num_records,
        all_records=args.all_records,
        filter_genes=args.filter_genes,
        experimental_methods=args.experimental_methods,
        coding_only=args.coding_only,
        max_studies=args.max_studies
    )

    # Get all data at once using the existing get_data method
    print("Loading data from MAVE wrapper...")
    data = mave_wrapper.get_data(Seq_length=args.seq_length or float('inf'), target="score")

    all_processed_data = []
    total_skipped = 0

    # Open output file and process all data
    with open(args.output, 'w') as f:
        for seq_pair, score in data:
            try:
                score_float = float(score)
                if np.isnan(score_float) or np.isinf(score_float):
                    total_skipped += 1
                    continue
                # Extract the three components: ref, alt, annotation
                ref, alt, annotation = seq_pair
                record_data = {
                    "reference_sequence": ref,
                    "variant_sequence": alt,
                    "annotation": annotation,
                    "score": score_float
                }
                all_processed_data.append(record_data)

                # Write record immediately
                record = [
                    [ref, alt, annotation],
                    score_float
                ]
                f.write(json.dumps(record) + '\n')

            except (TypeError, ValueError):
                total_skipped += 1
                continue

        print(f"Processed {len(all_processed_data)} records, {total_skipped} skipped")

    if total_skipped > 0:
        print(f"Total skipped {total_skipped} samples with invalid scores")

    # Report score statistics for information
    if len(all_processed_data) > 0:
        scores = [item["score"] for item in all_processed_data]
        scores_array = np.array(scores)
        print(f"Score statistics:")
        print(f"  Mean: {scores_array.mean():.4f} ± {scores_array.std():.4f}")
        print(f"  Range: [{scores_array.min():.4f}, {scores_array.max():.4f}]")

    print(f"Saved {len(all_processed_data)} records to {args.output}")
    print(f"Format matches training pipeline: (ref, alt, annotation) + score")


if __name__ == "__main__":
    main()