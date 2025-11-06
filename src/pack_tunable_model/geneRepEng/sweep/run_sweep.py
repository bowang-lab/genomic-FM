#!/usr/bin/env python3
"""
Utility script to run genomic control vector sweeps with different configurations.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_sweep_command(args):
    """Construct and run the sweep command."""

    # Base command
    cmd = [
        sys.executable,
        "sweep.py",
        "--model_type", args.model_type,
        "--dataset_type", args.dataset_type,
        "--method", args.method,
        "--project_name", args.project_name,
        "--count", str(args.count),
        "--seed", str(args.seed),
        "--max_samples", str(args.max_samples),
        "--eval_samples", str(args.eval_samples),
        "--seq_length", str(args.seq_length),
    ]

    # Add optional arguments
    if args.csv_path:
        cmd.extend(["--csv_path", args.csv_path])

    if args.threshold:
        cmd.extend(["--threshold", str(args.threshold)])

    if args.sweep_id:
        cmd.extend(["--sweep_id", args.sweep_id])

    if args.entity:
        cmd.extend(["--entity", args.entity])

    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Run genomic control vector sweeps")

    # Model and dataset configuration
    parser.add_argument("--model_type", type=str, default="omni_dna",
                        choices=["omni_dna", "nucleotide_transformer"],
                        help="Model type to use")
    parser.add_argument("--dataset_type", type=str, default="smart",
                        choices=["smart", "synthetic"],
                        help="Dataset type to use")
    parser.add_argument("--csv_path", type=str,
                        default="/zehui/genomic_fm/all_clinvar/unfiltered_variants.csv",
                        help="Path to SMART variant CSV file")

    # Experiment configuration
    parser.add_argument("--method", type=str, default="pca_diff",
                        choices=["pca_diff", "mean_diff"],
                        help="Control vector extraction method")
    parser.add_argument("--threshold", type=float, default=54.0,
                        help="Threshold for binarizing SMART scores")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Maximum samples for training")
    parser.add_argument("--eval_samples", type=int, default=1000,
                        help="Number of samples for evaluation")
    parser.add_argument("--seq_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Wandb configuration
    parser.add_argument("--project_name", type=str, default="geneRepEng-sweep",
                        help="Wandb project name")
    parser.add_argument("--entity", type=str, default=None,
                        help="Wandb entity")
    parser.add_argument("--sweep_id", type=str, default=None,
                        help="Existing sweep ID to continue")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of sweep runs")

    # Preset configurations
    parser.add_argument("--preset", type=str, default=None,
                        choices=["quick_test", "full_omni_dna", "full_nt", "pathogenicity"],
                        help="Use a preset configuration")

    args = parser.parse_args()

    # Apply preset configurations
    if args.preset == "quick_test":
        args.dataset_type = "synthetic"
        args.max_samples = 500
        args.eval_samples = 100
        args.count = 10
        args.seq_length = 100
        print("Using quick test preset")

    elif args.preset == "full_omni_dna":
        args.model_type = "omni_dna"
        args.dataset_type = "smart"
        args.max_samples = 100000
        args.eval_samples = 4000
        args.count = 200
        args.method = "pca_diff"
        print("Using full Omni-DNA preset")

    elif args.preset == "full_nt":
        args.model_type = "nucleotide_transformer"
        args.dataset_type = "smart"
        args.max_samples = 10000
        args.eval_samples = 2000
        args.count = 200
        args.method = "pca_diff"
        print("Using full Nucleotide Transformer preset")

    elif args.preset == "pathogenicity":
        args.dataset_type = "smart"
        args.threshold = 54.0
        args.method = "pca_diff"
        args.max_samples = 8000
        args.eval_samples = 1500
        args.count = 150
        print("Using pathogenicity prediction preset")

    # Run the sweep
    exit_code = run_sweep_command(args)

    if exit_code == 0:
        print("Sweep completed successfully!")
    else:
        print(f"Sweep failed with exit code: {exit_code}")

    return exit_code

if __name__ == "__main__":
    sys.exit(main())
