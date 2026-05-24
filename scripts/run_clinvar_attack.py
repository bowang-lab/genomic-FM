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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attacks import ClinVarAttributeInference
from src.attacks.attribute_inference import run_attack
from src.pack_tunable_model.hf_dataloader import return_clinvar_grouped_lira_dataset
from src.pack_tunable_model.wrap_model import WrappedModelWithClassificationHead
from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup


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


def collate_fn(batch):
    """Collate function for DataLoader."""
    ref_input_ids = torch.stack([item["ref_input_ids"] for item in batch])
    alt_input_ids = torch.stack([item["alt_input_ids"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    group_ids = torch.tensor([item["group_id"] for item in batch], dtype=torch.long)

    result = {
        "ref_input_ids": ref_input_ids,
        "alt_input_ids": alt_input_ids,
        "labels": labels,
        "group_ids": group_ids,
    }

    if "ref_attention_mask" in batch[0]:
        result["ref_attention_mask"] = torch.stack([item["ref_attention_mask"] for item in batch])
    if "alt_attention_mask" in batch[0]:
        result["alt_attention_mask"] = torch.stack([item["alt_attention_mask"] for item in batch])

    return result


def train_model(
    model,
    train_dataset,
    val_dataset,
    device,
    output_dir,
    epochs=50,
    batch_size=8,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    patience=10,
    freeze_backbone=True,
):
    """
    Train the classification head on the training set.

    Args:
        model: WrappedModelWithClassificationHead
        train_dataset: Training dataset
        val_dataset: Validation dataset
        device: torch device
        output_dir: Directory to save checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for classification head
        weight_decay: Weight decay
        warmup_ratio: Warmup ratio for scheduler
        patience: Early stopping patience
        freeze_backbone: Whether to freeze the backbone (train head only)

    Returns:
        Dict with training results
    """
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.base_model.parameters():
            param.requires_grad = False
        print("Backbone frozen, training classification head only")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Set up optimizer - only train parameters that require grad
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    # Scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    best_val_loss = float("inf")
    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            # Move to device
            ref_input_ids = batch["ref_input_ids"].to(device)
            alt_input_ids = batch["alt_input_ids"].to(device)
            labels = batch["labels"].to(device)
            ref_attention_mask = batch.get("ref_attention_mask")
            alt_attention_mask = batch.get("alt_attention_mask")
            if ref_attention_mask is not None:
                ref_attention_mask = ref_attention_mask.to(device)
            if alt_attention_mask is not None:
                alt_attention_mask = alt_attention_mask.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(
                ref_input_ids=ref_input_ids,
                ref_attention_mask=ref_attention_mask,
                alt_input_ids=alt_input_ids,
                alt_attention_mask=alt_attention_mask,
                labels=labels,
            )

            loss = outputs["loss"]
            logits = outputs["logits"]

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Track metrics
            epoch_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += len(labels)

            pbar.set_postfix({"loss": loss.item(), "acc": epoch_correct / epoch_total})

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                ref_input_ids = batch["ref_input_ids"].to(device)
                alt_input_ids = batch["alt_input_ids"].to(device)
                labels = batch["labels"].to(device)
                ref_attention_mask = batch.get("ref_attention_mask")
                alt_attention_mask = batch.get("alt_attention_mask")
                if ref_attention_mask is not None:
                    ref_attention_mask = ref_attention_mask.to(device)
                if alt_attention_mask is not None:
                    alt_attention_mask = alt_attention_mask.to(device)

                outputs = model(
                    ref_input_ids=ref_input_ids,
                    ref_attention_mask=ref_attention_mask,
                    alt_input_ids=alt_input_ids,
                    alt_attention_mask=alt_attention_mask,
                    labels=labels,
                )

                loss = outputs["loss"]
                logits = outputs["logits"]
                val_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch:>3}/{epochs}  |  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  |  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # Checkpointing and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            no_improve = 0

            # Save checkpoint
            ckpt_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  ✓ Checkpoint saved → {ckpt_path}")
        else:
            no_improve += 1
            if patience and no_improve >= patience:
                print(f"Early stopping after {epoch} epochs (no improvement for {patience} epochs)")
                break

    # Save final model
    final_path = os.path.join(output_dir, "final_model.pt")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
    }, final_path)
    print(f"Final model saved → {final_path}")

    return {
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "final_epoch": epoch,
    }


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

    # Training parameters
    parser.add_argument("--eval_only", type=int, default=1,
                        help="Only run attack evaluation (no training)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--freeze_backbone", type=int, default=1,
                        help="Freeze backbone and only train head (1) or fine-tune all (0)")

    # Attack parameters
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

    # Create train/val splits from the full dataset using train_mask
    train_indices = [i for i, m in enumerate(train_mask) if m]
    val_indices = [i for i, m in enumerate(train_mask) if not m]
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Training or evaluation
    if not args.eval_only:
        # Training mode: train the classification head
        print("\n" + "="*70)
        print(f"Training Classification Head (expid={args.expid}/{args.num_experiments})")
        print("="*70)

        train_results = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            freeze_backbone=bool(args.freeze_backbone),
        )

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best validation loss: {train_results['best_val_loss']:.4f}")
        print(f"Best validation accuracy: {train_results['best_val_acc']:.4f}")
        print(f"Stopped at epoch: {train_results['final_epoch']}")
        print("="*70)

        # Save train mask for later attack evaluation
        mask_path = os.path.join(args.output_dir, "train_mask.npy")
        np.save(mask_path, train_mask)
        print(f"Train mask saved → {mask_path}")

        return train_results

    else:
        # Evaluation mode: load checkpoint if exists, then run attack
        ckpt_path = os.path.join(args.output_dir, "best_model.pt")
        if os.path.exists(ckpt_path):
            print(f"\nLoading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            print(f"Loaded model from epoch {checkpoint['epoch']} (val_acc={checkpoint['val_acc']:.4f})")
        else:
            print(f"\nNo checkpoint found at {ckpt_path}, using randomly initialized head")

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
