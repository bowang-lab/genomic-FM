"""
ClinVar Grouped Attribute Inference Attack
==========================================
Adapted from Emmy's DMS attack (guess_kolter_one_codon.py) for ClinVar classification.

Runs attribute inference attack on grouped ClinVar data to test whether
variants can be re-identified based on model behavior within their
biological group (e.g., gene, exon, cardiac panel).

Prediction targets:
    - CLNSIG: Pathogenicity classification (benign/pathogenic) - 2 classes
    - CLNDN: Disease prediction (multi-class) - N disease classes

Grouping modes:
    - gene: Each gene is a group (all genes in ClinVar)
    - exon: Each exon is a group (gene:exon_number)
    - cardiac_panel: Groups by CGC cardiac category (CM_ARM, AORTOPATHY, CHD, OTHER)
    - cardiac_gene: Only CGC cardiac genes (647 genes), each gene is a group
    - hcm_gene: Only HCM genes (168 genes), each gene is a group

Attack modes:
    - likelihood-based (default): Uses model's softmax probabilities
    - embedding-based: Uses cosine similarity of model embeddings

Usage:
    # Basic pathogenicity attack
    python scripts/run_clinvar_attack.py --checkpoint ./model --grouping gene --target CLNSIG

    # Disease prediction attack
    python scripts/run_clinvar_attack.py --checkpoint ./model --grouping gene --target CLNDN

    # Different grouping modes
    python scripts/run_clinvar_attack.py --checkpoint ./model --grouping exon
    python scripts/run_clinvar_attack.py --checkpoint ./model --grouping cardiac_panel
    python scripts/run_clinvar_attack.py --checkpoint ./model --grouping cardiac_gene
    python scripts/run_clinvar_attack.py --checkpoint ./model --grouping hcm_gene

    # Disease prediction with heart disease subset
    python scripts/run_clinvar_attack.py --checkpoint ./model --grouping cardiac_gene --target CLNDN \\
        --disease_subset_file ./root/data/heart_related_diseases.txt

    # Embedding-based attack (instead of likelihood-based)
    python scripts/run_clinvar_attack.py --checkpoint ./model --use_embedding 1

    # Custom data parameters
    python scripts/run_clinvar_attack.py --checkpoint ./model \\
        --seq_length 512 \\
        --min_variants_per_gene 3 \\
        --max_variants_per_gene 100 \\
        --balance_classes 0

    # LiRA multi-experiment setup
    python scripts/run_clinvar_attack.py --checkpoint ./model --expid 0 --num_experiments 64
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
    model: str,
    num_classes: int = 2,
    device: str = 'cuda'
):
    """
    Load model and tokenizer from HuggingFace or local checkpoint.

    Auto-detects whether path is a local checkpoint or HuggingFace model.

    Args:
        model: HuggingFace model name (e.g., 'InstaDeepAI/nucleotide-transformer-v2-50m-multi-species')
               or local checkpoint path
        num_classes: Number of output classes
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    # Check if it's a local path with a saved model
    is_local = os.path.isdir(model)

    if is_local:
        # Try to load as WrappedModelWithClassificationHead first
        try:
            loaded_model = WrappedModelWithClassificationHead.from_pretrained(model)
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            print(f"Loaded wrapped model from {model}")
            return loaded_model.to(device), tokenizer
        except Exception as e:
            print(f"Could not load as wrapped model: {e}")
            # Try loading as base model
            print(f"Trying to load as base model from {model}...")

    # Load from HuggingFace or local base model
    print(f"Loading model from {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    # Auto-detect model type based on name
    model_lower = model.lower()
    if 'nucleotide-transformer' in model_lower or '/nt' in model_lower:
        base_model = AutoModelForMaskedLM.from_pretrained(model, trust_remote_code=True)
    elif 'omni' in model_lower:
        base_model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
    else:
        # Default: try AutoModel
        try:
            base_model = AutoModel.from_pretrained(model, trust_remote_code=True)
        except Exception:
            base_model = AutoModelForMaskedLM.from_pretrained(model, trust_remote_code=True)

    # Create wrapped model with classification head
    wrapped_model = WrappedModelWithClassificationHead(base_model, num_classes=num_classes)
    print(f"Created model with {num_classes}-class classification head (randomly initialized)")

    return wrapped_model.to(device), tokenizer


def main():
    parser = argparse.ArgumentParser(description="ClinVar grouped attribute inference attack")

    # Data parameters
    parser.add_argument("--grouping", default="gene",
                        choices=["gene", "exon", "cardiac_panel", "cardiac_gene", "hcm_gene"],
                        help="Grouping mode for variants")
    parser.add_argument("--target", default="CLNSIG",
                        choices=["CLNSIG", "CLNDN"],
                        help="Prediction target: CLNSIG (pathogenicity) or CLNDN (disease)")
    parser.add_argument("--disease_subset_file", type=str, default=None,
                        help="Path to file with disease names to filter (for CLNDN target)")
    parser.add_argument("--num_records", type=int, default=100000,
                        help="Number of records to load")
    parser.add_argument("--all_records", type=int, default=1,
                        help="Load all records (1) or use num_records (0)")
    parser.add_argument("--seq_length", type=int, default=1024,
                        help="Sequence context length around variant")
    parser.add_argument("--min_variants_per_gene", type=int, default=5,
                        help="Minimum variants per gene to include")
    parser.add_argument("--max_variants_per_gene", type=int, default=50,
                        help="Maximum variants per gene")
    parser.add_argument("--balance_classes", type=int, default=1,
                        help="Balance pathogenic/benign within groups (1) or not (0)")

    # Model parameters
    parser.add_argument("--model", type=str, default="InstaDeepAI/nucleotide-transformer-500m-human-ref",
                        help="HuggingFace model name or local checkpoint path")

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
    parser.add_argument("--use_embedding", type=int, default=0,
                        help="Use embedding-based attack instead of likelihood-based (0 or 1)")

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
    args.output_dir = f"{args.output_dir}/{args.target}_{args.grouping}/exp{args.expid}_{args.num_experiments}"
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Arguments: {args}")

    # Load model and tokenizer
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(
        model=args.model,
        num_classes=2,  # Default for CLNSIG, will be updated for CLNDN after data loading
        device=str(device)
    )

    # Load data with grouped LiRA function
    print(f"\nLoading data with {args.target} target, {args.grouping} grouping...")
    datasets, task_num_classes, seq_length, group_to_id, stats = return_clinvar_grouped_lira_dataset(
        tokenizer,
        grouping=args.grouping,
        target=args.target,
        seq_length=args.seq_length,
        exp_id=args.expid,
        num_experiments=args.num_experiments,
        seed=args.seed,
        num_records=args.num_records,
        all_records=bool(args.all_records),
        disease_subset_file=args.disease_subset_file,
        min_variants_per_gene=args.min_variants_per_gene,
        max_variants_per_gene=args.max_variants_per_gene,
        balance_classes=bool(args.balance_classes),
    )

    # Get the full dataset
    task_name = f'{args.target}_{args.grouping}'
    full_dataset = datasets[f'{task_name}_full']
    train_mask = stats['membership_info']['sample_membership']
    num_classes = stats['membership_info']['num_classes']

    print(f"\nDataset loaded:")
    print(f"  Target: {args.target}")
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Total groups: {len(group_to_id)}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Training samples: {sum(train_mask)}")
    print(f"  Validation samples: {len(train_mask) - sum(train_mask)}")

    # Run attack
    if args.eval_only:
        attack_type = "Embedding-based" if args.use_embedding else "Likelihood-based"
        print("\n" + "="*70)
        print(f"Running {attack_type} Attribute Inference Attack ({args.target} target)")
        print("="*70)

        results = run_attack(
            model=model,
            tokenizer=tokenizer,
            full_dataset=full_dataset,
            group_to_id=group_to_id,
            train_mask=train_mask,
            device=str(device),
            verbose=True,
            use_embedding=bool(args.use_embedding)
        )

        # Save results
        results_path = os.path.join(args.output_dir, f"attack_results_{args.target}_{args.grouping}.pkl")
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
