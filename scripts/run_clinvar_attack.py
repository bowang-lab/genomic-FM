"""
ClinVar Grouped Attribute Inference Attack
==========================================
Adapted from Emmy's DMS attack (guess_kolter_one_codon.py) for ClinVar classification.

Runs attribute inference attack on grouped ClinVar data to test whether
variants can be re-identified based on model behavior within their
biological group (e.g., gene, exon, cardiac panel).

Usage:
    python scripts/run_clinvar_attack.py --expid 0 --num_experiments 64 --grouping gene
    python scripts/run_clinvar_attack.py --expid 0 --num_experiments 64 --grouping gene --eval_only 1
"""

from __future__ import annotations
import pickle
import argparse
import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attacks import ClinVarAttributeInference
from src.attacks.attribute_inference import run_attack
from src.pack_tunable_model.hf_dataloader import return_clinvar_grouped_lira_dataset
from src.pack_tunable_model.wrap_model import WrappedModelWithClassificationHead
from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    checkpoint_path: str,
    model_type: str = None,
    device: str = 'cuda'
):
    """
    Load a trained model and tokenizer from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint directory
        model_type: Base model type (e.g., 'nt', 'omni_dna_116m', 'dnabert2')
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    # Try to load using WrappedModelWithClassificationHead.from_pretrained
    try:
        model = WrappedModelWithClassificationHead.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        print(f"Loaded wrapped model from {checkpoint_path}")
        return model.to(device), tokenizer
    except Exception as e:
        print(f"Could not load wrapped model directly: {e}")

    # Fall back to loading base model + classification head separately
    if model_type is None:
        raise ValueError("model_type must be provided when loading from separate components")

    # Determine model path
    local_model_base = f"./root/models/{model_type}"
    if model_type == 'omni_dna_116m':
        model_path = local_model_base if os.path.exists(local_model_base) else "zehui127/Omni-DNA-116M"
    elif model_type == 'nt':
        model_path = local_model_base if os.path.exists(local_model_base) else "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    elif model_type == 'dnabert2':
        model_path = local_model_base if os.path.exists(local_model_base) else "zhihan1996/DNABERT-2-117M"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load base model
    use_local = os.path.exists(model_path)
    if model_type in ['gpn-star', 'nt']:
        base_model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=use_local)
    elif model_type == 'omni_dna_116m':
        base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=use_local)
    else:
        base_model = AutoModel.from_pretrained(model_path, trust_remote_code=True, local_files_only=use_local)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=use_local)

    # Create wrapped model with classification head
    model = WrappedModelWithClassificationHead(base_model, num_classes=2)

    # Load checkpoint weights if available
    checkpoint_weights = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(checkpoint_weights):
        state_dict = torch.load(checkpoint_weights, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {checkpoint_weights}")
    else:
        # Try classification head only
        head_weights = os.path.join(checkpoint_path, "classification_head.bin")
        if os.path.exists(head_weights):
            model.classification_head.load_state_dict(torch.load(head_weights, map_location='cpu'))
            print(f"Loaded classification head from {head_weights}")

    return model.to(device), tokenizer


def main():
    parser = argparse.ArgumentParser(description="ClinVar grouped attribute inference attack")

    # Data parameters
    parser.add_argument("--grouping", default="gene",
                        choices=["gene", "exon", "cardiac_panel", "cardiac_gene", "hcm_gene"],
                        help="Grouping mode for variants")
    parser.add_argument("--num_records", type=int, default=100000,
                        help="Number of records to load")
    parser.add_argument("--all_records", type=int, default=1,
                        help="Load all records (1) or use num_records (0)")

    # Model parameters
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default="nt",
                        choices=["nt", "omni_dna_116m", "dnabert2"],
                        help="Base model type")

    # LiRA parameters
    parser.add_argument("--expid", type=int, default=0,
                        help="Experiment ID for LiRA")
    parser.add_argument("--num_experiments", type=int, default=64,
                        help="Total number of LiRA experiments")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Attack parameters
    parser.add_argument("--eval_only", type=int, default=1,
                        help="Only run attack evaluation (no training)")

    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./attack_results",
                        help="Directory to save results")

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    args.output_dir = f"{args.output_dir}/{args.grouping}/exp{args.expid}_{args.num_experiments}"
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Arguments: {args}")

    # Load model and tokenizer
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(
        args.checkpoint,
        model_type=args.model_type,
        device=str(device)
    )

    # Load data with grouped LiRA function
    print(f"\nLoading data with {args.grouping} grouping...")
    datasets, task_num_classes, seq_length, group_to_id, stats = return_clinvar_grouped_lira_dataset(
        tokenizer,
        grouping=args.grouping,
        exp_id=args.expid,
        num_experiments=args.num_experiments,
        seed=args.seed,
        num_records=args.num_records,
        all_records=bool(args.all_records),
    )

    # Get the full dataset
    task_name = f'CLNSIG_{args.grouping}'
    full_dataset = datasets[f'{task_name}_full']
    train_mask = stats['membership_info']['sample_membership']

    print(f"\nDataset loaded:")
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Total groups: {len(group_to_id)}")
    print(f"  Training samples: {sum(train_mask)}")
    print(f"  Validation samples: {len(train_mask) - sum(train_mask)}")

    # Run attack
    if args.eval_only:
        print("\n" + "="*70)
        print("Running Attribute Inference Attack")
        print("="*70)

        results = run_attack(
            model=model,
            tokenizer=tokenizer,
            full_dataset=full_dataset,
            group_to_id=group_to_id,
            train_mask=train_mask,
            device=str(device),
            verbose=True
        )

        # Save results
        results_path = os.path.join(args.output_dir, f"attack_results_{args.grouping}.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {results_path}")

        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Training set accuracy:   {results['train']['overall_accuracy']:.4f} "
              f"(random: {results['train']['random_baseline']:.4f}, "
              f"advantage: {results['train']['advantage']:.4f})")
        print(f"Validation set accuracy: {results['val']['overall_accuracy']:.4f} "
              f"(random: {results['val']['random_baseline']:.4f}, "
              f"advantage: {results['val']['advantage']:.4f})")
        print(f"Full set accuracy:       {results['full']['overall_accuracy']:.4f} "
              f"(random: {results['full']['random_baseline']:.4f}, "
              f"advantage: {results['full']['advantage']:.4f})")
        print("="*70)

    return results


if __name__ == "__main__":
    main()
