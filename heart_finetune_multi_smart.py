#!/usr/bin/env python3
"""
Script to run smart variant training across multiple thresholds.
Runs training with thresholds: 49, 54, 59, 64, 69
"""

import subprocess
import sys
import argparse

def run_training(threshold, base_args):
    """Run training with a specific threshold"""
    cmd = [
        "accelerate", "launch",
        "--config_file", "configs/ddp.yaml",
        "--main_process_port", str(29500 + int(threshold)),  # Use unique port for each threshold
        "-m", "src.pack_tunable_model.hf_trainer_smart",
        "--threshold", str(threshold)
    ] + base_args
    
    print(f"\n{'='*60}")
    print(f"Running training with threshold: {threshold}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ Training with threshold {threshold} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Training with threshold {threshold} failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run smart variant training across multiple thresholds (49, 54, 59, 64, 69)"
    )
    
    parser.add_argument("--model", type=str, default="nt", help="Model type")
    parser.add_argument("--seed", type=int, default=127, help="Random seed")
    parser.add_argument("--decoder", action="store_true", help="Use decoder architecture")
    parser.add_argument("--test_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--task", type=str, default="CLNDN", choices=["CLNDN", "CLNSIG"], help="Task type")
    parser.add_argument("--learning_rate", type=float, default=0.000005, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue on failure")
    parser.add_argument("--checkpoint_path", type=str, default=None, 
                        help="Path to pre-trained ClinVar checkpoint to load from")
    
    args = parser.parse_args()
    
    thresholds = [49.0, 54.0, 59.0, 64.0, 69.0]
    base_args = []
    if args.model != "nt":
        base_args.extend(["--model", args.model])
    if args.seed != 127:
        base_args.extend(["--seed", str(args.seed)])
    if args.decoder:
        base_args.append("--decoder")
    if args.test_only:
        base_args.append("--test_only")
    if args.task != "CLNDN":
        base_args.extend(["--task", args.task])
    if args.learning_rate != 0.000005:
        base_args.extend(["--learning_rate", str(args.learning_rate)])
    if args.batch_size != 8:
        base_args.extend(["--batch_size", str(args.batch_size)])
    if args.num_epochs != 10:
        base_args.extend(["--num_epochs", str(args.num_epochs)])
    if args.max_grad_norm != 1.0:
        base_args.extend(["--max_grad_norm", str(args.max_grad_norm)])
    if args.num_workers != 8:
        base_args.extend(["--num_workers", str(args.num_workers)])
    if args.checkpoint_path:
        base_args.extend(["--checkpoint_path", args.checkpoint_path])
    
    print(f"Starting multi-threshold training with thresholds: {thresholds}")
    print(f"Base arguments: {' '.join(base_args) if base_args else 'Using default values'}")
    successful_runs = []
    failed_runs = []
    
    for threshold in thresholds:
        success = run_training(threshold, base_args)
        if success:
            successful_runs.append(threshold)
        else:
            failed_runs.append(threshold)
            if not args.continue_on_error:
                print(f"\nStopping execution due to failure at threshold {threshold}")
                break
    
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Successful runs ({len(successful_runs)}): {successful_runs}")
    if failed_runs:
        print(f"Failed runs ({len(failed_runs)}): {failed_runs}")
    print(f"Total thresholds processed: {len(successful_runs) + len(failed_runs)}/{len(thresholds)}")
    
    if failed_runs:
        print(f"\n⚠️  Some training runs failed. Check logs above for details.")
        sys.exit(1)
    else:
        print(f"\n✅ All training runs completed successfully!")

if __name__ == "__main__":
    main()
