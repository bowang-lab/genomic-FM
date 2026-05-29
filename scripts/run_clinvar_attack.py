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
from scipy.stats import norm
from glob import glob

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attacks import ClinVarAttributeInference
from src.attacks.attribute_inference import run_attack
from src.pack_tunable_model.hf_dataloader import return_clinvar_grouped_lira_dataset
from src.pack_tunable_model.wrap_model import WrappedModelWithClassificationHead
from src.pack_tunable_model.checkpoint_utils import load_checkpoint_into_model
from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup


def resolve_base_model_path(model_path: str) -> str:
    """Resolve the base model path - prefer local models over HuggingFace.

    Supported models (from genomic-FM):
        - nt_transformer_v2_500m: NT v2 500m multi-species (default)
        - nt_transformer_human_ref: NT 500m human ref
        - nt_transformer_1000g: NT 500m 1000g
        - dnabert2: DNABERT-2-117M
        - hyenadna variants
        - gena-lm variants
        - grover
    """
    model_lower = model_path.lower()

    # Local paths -> HuggingFace fallbacks
    # Format: (local_path, huggingface_id)
    MODEL_REGISTRY = {
        'nt_v2_500m': ('./root/models/nt', 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'),
        'nt_human_ref': ('./root/models/nucleotide-transformer-500m-human-ref', 'InstaDeepAI/nucleotide-transformer-500m-human-ref'),
        'nt_1000g': ('./root/models/nucleotide-transformer-500m-1000g', 'InstaDeepAI/nucleotide-transformer-500m-1000g'),
        'dnabert2': ('./root/models/dnabert2', 'zhihan1996/DNABERT-2-117M'),
        'gpn': ('./root/models/gpn-msa-sapiens', 'songlab/gpn-msa-sapiens'),
        'lucaone': ('./root/models/lucaone', None),
        'evo2': ('./root/models/evo2', None),
    }

    # Match checkpoint name to model key
    if 'human-ref' in model_lower or 'human_ref' in model_lower:
        key = 'nt_human_ref'
    elif '1000g' in model_lower:
        key = 'nt_1000g'
    elif 'dnabert' in model_lower:
        key = 'dnabert2'
    elif 'gpn' in model_lower:
        key = 'gpn'
    elif 'lucaone' in model_lower or 'luca' in model_lower:
        key = 'lucaone'
    elif 'evo' in model_lower:
        key = 'evo2'
    elif 'nt' in model_lower or 'nucleotide' in model_lower:
        key = 'nt_v2_500m'  # Default NT is v2 multispecies
    else:
        key = 'nt_v2_500m'  # Default

    local_path, hf_id = MODEL_REGISTRY[key]

    # Try local path first (no internet needed)
    if os.path.isdir(local_path):
        print(f"Using local base model: {local_path}")
        return local_path

    # Fallback to HuggingFace
    if hf_id:
        print(f"Local model not found, using HuggingFace: {hf_id}")
        return hf_id

    raise FileNotFoundError(f"Model not found locally at {local_path} and no HuggingFace fallback available")


def find_best_checkpoint(model_dir: Path) -> Path:
    """Find the best checkpoint in a directory based on trainer_state.json."""
    checkpoint_dirs = sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')],
        key=lambda x: int(x.name.split('-')[1])
    )

    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {model_dir}")

    # Try to find best from trainer_state.json
    for ckpt_dir in reversed(checkpoint_dirs):  # Start from latest
        trainer_state = ckpt_dir / "trainer_state.json"
        if trainer_state.exists():
            import json
            with open(trainer_state) as f:
                state = json.load(f)
            if "best_model_checkpoint" in state and state["best_model_checkpoint"]:
                best_path = Path(state["best_model_checkpoint"])
                if best_path.exists():
                    return best_path
                # Try relative path
                best_name = best_path.name
                for d in checkpoint_dirs:
                    if d.name == best_name:
                        return d

    # Default to latest checkpoint
    return checkpoint_dirs[-1]


def load_model_and_tokenizer(
    model: str,
    num_classes: int = 2,
    device: str = 'cuda',
    base_model_id: str = None
):
    """
    Load model and tokenizer from HuggingFace or local checkpoint.

    For local checkpoints without config.json, loads base architecture from HuggingFace
    and then loads checkpoint weights.

    Args:
        model: HuggingFace model name or local checkpoint path
        num_classes: Number of output classes
        device: Device to load model on
        base_model_id: Optional HuggingFace ID for base model (auto-detected if None)

    Returns:
        Tuple of (model, tokenizer)
    """
    is_local = os.path.isdir(model)

    if is_local:
        model_path = Path(model)

        # Try to load as WrappedModelWithClassificationHead first
        try:
            loaded_model = WrappedModelWithClassificationHead.from_pretrained(model)
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            print(f"Loaded wrapped model from {model}")
            return loaded_model.to(device), tokenizer
        except Exception as e:
            print(f"Could not load as wrapped model: {e}")

        # Check if this is a checkpoint directory with subdirectories
        checkpoint_dirs = [d for d in model_path.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]

        if checkpoint_dirs:
            print(f"Found {len(checkpoint_dirs)} checkpoint directories")

            # Resolve base model path (local or HuggingFace)
            hf_id = base_model_id or resolve_base_model_path(model)
            print(f"Using base model: {hf_id}")

            # Load tokenizer and base model architecture
            tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

            # Detect model type (check both path and original model name)
            model_lower = hf_id.lower()
            orig_lower = model.lower()
            is_nt = 'nucleotide-transformer' in model_lower or '/nt' in model_lower or 'root/models/nt' in orig_lower
            is_omni = 'omni' in model_lower or 'omni' in orig_lower

            if is_nt:
                base_model = AutoModelForMaskedLM.from_pretrained(hf_id, trust_remote_code=True)
            elif is_omni:
                base_model = AutoModelForCausalLM.from_pretrained(hf_id, trust_remote_code=True)
            else:
                base_model = AutoModelForMaskedLM.from_pretrained(hf_id, trust_remote_code=True)

            # Find best checkpoint and load weights
            best_ckpt = find_best_checkpoint(model_path)
            print(f"Loading weights from: {best_ckpt}")

            weights_path = best_ckpt / "pytorch_model.bin"
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location="cpu")

                # Filter to matching keys (handle potential prefix mismatches)
                model_keys = set(base_model.state_dict().keys())
                filtered = {}
                for k, v in state_dict.items():
                    # Try original key
                    if k in model_keys:
                        filtered[k] = v
                    # Try without 'base_model.' prefix
                    elif k.replace("base_model.", "", 1) in model_keys:
                        filtered[k.replace("base_model.", "", 1)] = v
                    # Try without 'model.' prefix
                    elif k.replace("model.", "", 1) in model_keys:
                        filtered[k.replace("model.", "", 1)] = v

                if filtered:
                    base_model.load_state_dict(filtered, strict=False)
                    print(f"Loaded {len(filtered)}/{len(state_dict)} weights from checkpoint")
                else:
                    print("Warning: No matching weights found, using base model weights")
            else:
                print(f"Warning: No pytorch_model.bin found at {weights_path}")

            # Create wrapped model with classification head
            wrapped_model = WrappedModelWithClassificationHead(base_model, num_classes=num_classes)
            return wrapped_model.to(device), tokenizer
        else:
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


@torch.no_grad()
def evaluate_for_lira(
    model,
    dataset,
    device,
    batch_size=16,
):
    """
    Evaluate model on full dataset and return per-sample predictions/losses.

    This is needed for LiRA membership inference - we need to track
    each sample's loss when it's IN training vs OUT of training.

    Args:
        model: Trained model
        dataset: Full dataset (all samples)
        device: torch device
        batch_size: Batch size for evaluation

    Returns:
        Dict with per-sample predictions, labels, losses, and logits
    """
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing issues
    )

    all_preds = []
    all_labels = []
    all_losses = []
    all_logits = []
    all_probs = []

    loss_fn = nn.CrossEntropyLoss(reduction='none')  # Per-sample loss

    for batch in tqdm(loader, desc="Evaluating for LiRA"):
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
        )

        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        losses = loss_fn(logits, labels)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_losses.append(losses.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_losses = np.concatenate(all_losses)
    all_logits = np.concatenate(all_logits)
    all_probs = np.concatenate(all_probs)

    # Compute aggregate metrics
    accuracy = (all_preds == all_labels).mean()
    mean_loss = all_losses.mean()

    return {
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_losses": all_losses,
        "all_logits": all_logits,
        "all_probs": all_probs,
        "accuracy": accuracy,
        "mean_loss": mean_loss,
    }


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


def aggregate_lira_results(
    base_dir: str,
    target: str,
    grouping: str,
    model: str = None,
    use_embedding: int = 0,
    freeze_backbone: int = 1,
    min_variants_per_gene: int = 5,
    max_variants_per_gene: int = 50,
    disease_subset_file: str = None,
    num_experiments: int = 64,
    output_dir: str = None,
):
    """
    Aggregate LiRA results from all experiments and generate ROC curves.

    Implements all 5 LiRA variants from Carlini et al.:
    - LiRA (online): per-sample μ, per-sample σ
    - LiRA (online, fixed variance): per-sample μ, global σ
    - LiRA (offline): global μ, per-sample σ
    - LiRA (offline, fixed variance): global μ, global σ
    - Global Threshold: simple threshold on loss

    Args:
        base_dir: Base directory containing experiment results
        target: Prediction target (CLNSIG or CLNDN)
        grouping: Grouping mode used
        model: Model name/path (used to construct experiment directory name)
        use_embedding: Whether embedding-based attack was used (0 or 1)
        min_variants_per_gene: Min variants per gene filter used
        max_variants_per_gene: Max variants per gene filter used
        disease_subset_file: Disease subset file used (if any)
        num_experiments: Number of shadow model experiments
        output_dir: Directory to save aggregated results and plots
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    print("\n" + "="*70)
    print("Aggregating LiRA Results")
    print("="*70)

    # Construct experiment name to match train/eval naming
    if model:
        model_name = os.path.basename(model.rstrip('/'))
        emb_suffix = "_emb" if use_embedding else "_lik"
        freeze_suffix = "_head" if freeze_backbone else "_full"
        data_suffix = ""
        if min_variants_per_gene != 5 or max_variants_per_gene != 50:
            data_suffix += f"_v{min_variants_per_gene}-{max_variants_per_gene}"
        if disease_subset_file:
            subset_name = os.path.basename(disease_subset_file).replace('.txt', '').replace('_related_diseases', '')
            data_suffix += f"_{subset_name}"
        experiment_name = f"{target}_{grouping}_{model_name}{emb_suffix}{freeze_suffix}{data_suffix}"
    else:
        # Fallback for legacy runs without model in path
        experiment_name = f"{target}_{grouping}"

    # Find all experiment directories
    exp_pattern = f"{base_dir}/{experiment_name}/exp*_{num_experiments}"
    exp_dirs = sorted(glob(exp_pattern))

    if not exp_dirs:
        print(f"No experiment directories found matching: {exp_pattern}")
        return None

    print(f"Found {len(exp_dirs)} experiment directories")

    # Load results from each experiment
    all_attack_results = []
    all_full_metrics = []
    all_train_masks = []

    for exp_dir in exp_dirs:
        attack_file = os.path.join(exp_dir, f"attack_results_{target}_{grouping}.pkl")
        metrics_file = os.path.join(exp_dir, f"full_metrics_{target}_{grouping}.pkl")
        mask_file = os.path.join(exp_dir, "train_mask.npy")

        has_attack = os.path.exists(attack_file)
        has_metrics = os.path.exists(metrics_file)
        has_mask = os.path.exists(mask_file)

        if has_attack:
            with open(attack_file, 'rb') as f:
                all_attack_results.append(pickle.load(f))
        if has_metrics:
            with open(metrics_file, 'rb') as f:
                all_full_metrics.append(pickle.load(f))
        if has_mask:
            all_train_masks.append(np.load(mask_file))

    print(f"Loaded {len(all_attack_results)} attack results")
    print(f"Loaded {len(all_full_metrics)} full metrics (for LiRA)")
    print(f"Loaded {len(all_train_masks)} train masks")

    if len(all_full_metrics) < 2 or len(all_train_masks) < 2:
        print("Need at least 2 experiments with full_metrics for LiRA analysis")
        if len(all_attack_results) >= 2:
            print("(Have attack results but missing full_metrics - re-run with --mode eval)")
        return None

    # Extract attack metrics for plotting (if available)
    train_accuracies = [r['train']['overall_accuracy'] for r in all_attack_results] if all_attack_results else []
    val_accuracies = [r['val']['overall_accuracy'] for r in all_attack_results] if all_attack_results else []
    train_advantages = [r['train']['advantage'] for r in all_attack_results] if all_attack_results else []
    val_advantages = [r['val']['advantage'] for r in all_attack_results] if all_attack_results else []

    # =========================================================================
    # LiRA Membership Inference using per-sample losses
    # =========================================================================
    n_samples = len(all_full_metrics[0]['all_losses'])
    n_experiments = len(all_full_metrics)

    print(f"\nComputing LiRA scores for {n_samples} samples across {n_experiments} experiments...")

    # Collect per-sample losses when IN training vs OUT of training
    in_losses = defaultdict(list)   # losses when sample is IN training
    out_losses = defaultdict(list)  # losses when sample is OUT of training

    for exp_idx, (metrics, mask) in enumerate(zip(all_full_metrics, all_train_masks)):
        losses = metrics['all_losses']
        for sample_idx in range(min(len(losses), len(mask))):
            if mask[sample_idx]:  # In training
                in_losses[sample_idx].append(losses[sample_idx])
            else:  # Out of training
                out_losses[sample_idx].append(losses[sample_idx])

    # Compute global statistics for offline variants
    all_in_losses = []
    all_out_losses = []
    for sample_idx in range(n_samples):
        all_in_losses.extend(in_losses.get(sample_idx, []))
        all_out_losses.extend(out_losses.get(sample_idx, []))

    global_in_mean = np.mean(all_in_losses) if all_in_losses else 0
    global_out_mean = np.mean(all_out_losses) if all_out_losses else 0
    global_std = np.std(all_in_losses + all_out_losses) + 1e-6 if (all_in_losses or all_out_losses) else 1.0

    print(f"Global stats: IN mean={global_in_mean:.4f}, OUT mean={global_out_mean:.4f}, std={global_std:.4f}")

    # Compute all 5 LiRA variants
    # Use the LAST experiment as the "target" model being attacked
    target_exp_idx = -1
    target_metrics = all_full_metrics[target_exp_idx]
    target_mask = all_train_masks[target_exp_idx]
    target_losses = target_metrics['all_losses']

    lira_online = []
    lira_online_fixed = []
    lira_offline = []
    lira_offline_fixed = []
    global_threshold = []
    lira_labels = []

    for sample_idx in range(n_samples):
        if sample_idx not in in_losses or sample_idx not in out_losses:
            continue
        if len(in_losses[sample_idx]) < 1 or len(out_losses[sample_idx]) < 1:
            continue

        in_vals = np.array(in_losses[sample_idx])
        out_vals = np.array(out_losses[sample_idx])

        # Per-sample statistics
        in_mean = np.mean(in_vals)
        out_mean = np.mean(out_vals)
        in_std = np.std(in_vals) + 1e-6 if len(in_vals) > 1 else global_std
        out_std = np.std(out_vals) + 1e-6 if len(out_vals) > 1 else global_std

        # Observation: loss from target model
        obs = target_losses[sample_idx]
        is_member = target_mask[sample_idx]

        # 1. LiRA Online: per-sample μ, per-sample σ
        log_p_in = norm.logpdf(obs, in_mean, in_std)
        log_p_out = norm.logpdf(obs, out_mean, out_std)
        lira_online.append(log_p_in - log_p_out)

        # 2. LiRA Online Fixed Variance: per-sample μ, global σ
        log_p_in_fixed = norm.logpdf(obs, in_mean, global_std)
        log_p_out_fixed = norm.logpdf(obs, out_mean, global_std)
        lira_online_fixed.append(log_p_in_fixed - log_p_out_fixed)

        # 3. LiRA Offline: global μ, per-sample σ
        log_p_in_global = norm.logpdf(obs, global_in_mean, in_std)
        log_p_out_global = norm.logpdf(obs, global_out_mean, out_std)
        lira_offline.append(log_p_in_global - log_p_out_global)

        # 4. LiRA Offline Fixed Variance: global μ, global σ
        log_p_in_global_fixed = norm.logpdf(obs, global_in_mean, global_std)
        log_p_out_global_fixed = norm.logpdf(obs, global_out_mean, global_std)
        lira_offline_fixed.append(log_p_in_global_fixed - log_p_out_global_fixed)

        # 5. Global Threshold: negative loss (lower loss = more likely member)
        global_threshold.append(-obs)

        lira_labels.append(1 if is_member else 0)

    # Convert to arrays
    lira_online = np.array(lira_online)
    lira_online_fixed = np.array(lira_online_fixed)
    lira_offline = np.array(lira_offline)
    lira_offline_fixed = np.array(lira_offline_fixed)
    global_threshold = np.array(global_threshold)
    lira_labels = np.array(lira_labels)

    print(f"Computed LiRA scores for {len(lira_labels)} samples")
    print(f"  Members: {sum(lira_labels)}, Non-members: {len(lira_labels) - sum(lira_labels)}")

    # Create output directory
    if output_dir is None:
        output_dir = f"{base_dir}/{experiment_name}/aggregate"
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # Generate plots
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Attack accuracy distribution across experiments
    ax1 = axes[0, 0]
    if train_accuracies:
        ax1.hist(train_accuracies, bins=15, alpha=0.7, label='Training', color='blue')
        ax1.hist(val_accuracies, bins=15, alpha=0.7, label='Validation', color='orange')
        ax1.axvline(np.mean(train_accuracies), color='blue', linestyle='--', label=f'Train mean: {np.mean(train_accuracies):.3f}')
        ax1.axvline(np.mean(val_accuracies), color='orange', linestyle='--', label=f'Val mean: {np.mean(val_accuracies):.3f}')
        ax1.set_xlabel('Attack Accuracy')
        ax1.set_ylabel('Count')
        ax1.set_title('Attribute Inference Attack Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No attack results available', ha='center', va='center')
        ax1.set_title('Attribute Inference Attack Accuracy')

    # Plot 2: Advantage over random
    ax2 = axes[0, 1]
    if train_advantages:
        ax2.hist(train_advantages, bins=15, alpha=0.7, label='Training', color='blue')
        ax2.hist(val_advantages, bins=15, alpha=0.7, label='Validation', color='orange')
        ax2.axvline(0, color='red', linestyle='-', linewidth=2, label='Random baseline')
        ax2.set_xlabel('Advantage over Random')
        ax2.set_ylabel('Count')
        ax2.set_title('Attack Advantage Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No attack results available', ha='center', va='center')
        ax2.set_title('Attack Advantage Distribution')

    # Plot 3: LiRA ROC curves (log scale, like Emmy's plot)
    ax3 = axes[1, 0]

    auc_results = {}

    if len(lira_labels) > 10 and len(np.unique(lira_labels)) > 1:
        # Compute ROC curves for all 5 variants
        # 1. LiRA Online
        fpr, tpr, _ = roc_curve(lira_labels, lira_online)
        auc_val = auc(fpr, tpr)
        auc_results['online'] = auc_val
        ax3.plot(fpr, tpr, 'b-', linewidth=2, label=f'LiRA (online) auc={auc_val:.3f}')

        # 2. LiRA Online Fixed Variance
        fpr, tpr, _ = roc_curve(lira_labels, lira_online_fixed)
        auc_val = auc(fpr, tpr)
        auc_results['online_fixed'] = auc_val
        ax3.plot(fpr, tpr, color='orange', linewidth=2, label=f'LiRA (online, fixed variance) auc={auc_val:.3f}')

        # 3. LiRA Offline
        fpr, tpr, _ = roc_curve(lira_labels, lira_offline)
        auc_val = auc(fpr, tpr)
        auc_results['offline'] = auc_val
        ax3.plot(fpr, tpr, 'g-', linewidth=2, label=f'LiRA (offline) auc={auc_val:.3f}')

        # 4. LiRA Offline Fixed Variance
        fpr, tpr, _ = roc_curve(lira_labels, lira_offline_fixed)
        auc_val = auc(fpr, tpr)
        auc_results['offline_fixed'] = auc_val
        ax3.plot(fpr, tpr, 'r-', linewidth=2, label=f'LiRA (offline, fixed variance) auc={auc_val:.3f}')

        # 5. Global Threshold
        fpr, tpr, _ = roc_curve(lira_labels, global_threshold)
        auc_val = auc(fpr, tpr)
        auc_results['global_threshold'] = auc_val
        ax3.plot(fpr, tpr, color='purple', linewidth=2, label=f'Global Threshold auc={auc_val:.3f}')

        # Random baseline (diagonal)
        ax3.plot([1e-5, 1], [1e-5, 1], 'k--', alpha=0.5)

        # Log scale like Emmy's plot
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlim([1e-5, 1])
        ax3.set_ylim([1e-5, 1])
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title(f'{grouping}')
        ax3.legend(loc='lower right', fontsize=8)
        ax3.grid(True, alpha=0.3, which='both')
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for ROC\n(need more experiments)',
                ha='center', va='center', fontsize=12)
        ax3.set_title('LiRA Membership Inference ROC')

    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    n_exp = len(all_full_metrics)
    summary_text = f"""
    Summary Statistics ({n_exp} experiments)
    {'='*50}

    LiRA Membership Inference:
      Samples analyzed: {len(lira_labels)}
      Members: {sum(lira_labels)} | Non-members: {len(lira_labels) - sum(lira_labels)}
    """

    if auc_results:
        summary_text += f"""
    AUC Scores:
      LiRA (online):              {auc_results.get('online', 0):.3f}
      LiRA (online, fixed var):   {auc_results.get('online_fixed', 0):.3f}
      LiRA (offline):             {auc_results.get('offline', 0):.3f}
      LiRA (offline, fixed var):  {auc_results.get('offline_fixed', 0):.3f}
      Global Threshold:           {auc_results.get('global_threshold', 0):.3f}
    """

    if train_accuracies:
        summary_text += f"""
    Attribute Inference Attack:
      Train accuracy: {np.mean(train_accuracies):.4f} ± {np.std(train_accuracies):.4f}
      Val accuracy:   {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}
      Train advantage:{np.mean(train_advantages):.4f}
      Val advantage:  {np.mean(val_advantages):.4f}
    """

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'lira_analysis_{target}_{grouping}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {plot_path}")

    # Save aggregated results
    agg_results = {
        'n_experiments': n_exp,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_advantages': train_advantages,
        'val_advantages': val_advantages,
        'lira_online': lira_online.tolist() if len(lira_online) > 0 else [],
        'lira_online_fixed': lira_online_fixed.tolist() if len(lira_online_fixed) > 0 else [],
        'lira_offline': lira_offline.tolist() if len(lira_offline) > 0 else [],
        'lira_offline_fixed': lira_offline_fixed.tolist() if len(lira_offline_fixed) > 0 else [],
        'global_threshold': global_threshold.tolist() if len(global_threshold) > 0 else [],
        'lira_labels': lira_labels.tolist() if len(lira_labels) > 0 else [],
        'auc_results': auc_results,
        'summary': {
            'mean_train_accuracy': float(np.mean(train_accuracies)) if train_accuracies else None,
            'std_train_accuracy': float(np.std(train_accuracies)) if train_accuracies else None,
            'mean_val_accuracy': float(np.mean(val_accuracies)) if val_accuracies else None,
            'std_val_accuracy': float(np.std(val_accuracies)) if val_accuracies else None,
            'mean_train_advantage': float(np.mean(train_advantages)) if train_advantages else None,
            'mean_val_advantage': float(np.mean(val_advantages)) if val_advantages else None,
        }
    }

    if auc_results:
        agg_results['summary'].update(auc_results)

    results_path = os.path.join(output_dir, f'aggregate_results_{target}_{grouping}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(agg_results, f)
    print(f"Results saved to: {results_path}")

    # Print summary
    print("\n" + "="*70)
    print("AGGREGATION SUMMARY")
    print("="*70)
    print(f"Experiments analyzed: {n_exp}")
    print(f"Samples with LiRA scores: {len(lira_labels)}")
    print(f"  Members: {sum(lira_labels)}, Non-members: {len(lira_labels) - sum(lira_labels)}")

    if auc_results:
        print("\nLiRA AUC Scores:")
        print(f"  LiRA (online):              {auc_results.get('online', 0):.4f}")
        print(f"  LiRA (online, fixed var):   {auc_results.get('online_fixed', 0):.4f}")
        print(f"  LiRA (offline):             {auc_results.get('offline', 0):.4f}")
        print(f"  LiRA (offline, fixed var):  {auc_results.get('offline_fixed', 0):.4f}")
        print(f"  Global Threshold:           {auc_results.get('global_threshold', 0):.4f}")

    if train_accuracies:
        print("\nAttribute Inference Attack:")
        print(f"  Mean train accuracy: {np.mean(train_accuracies):.4f} ± {np.std(train_accuracies):.4f}")
        print(f"  Mean val accuracy:   {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}")
    print("="*70)

    return agg_results


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
    parser.add_argument("--base_model", type=str, default=None,
                        help="HuggingFace base model ID for loading checkpoint weights (auto-detected if not specified)")

    # LiRA parameters
    parser.add_argument("--expid", type=int, default=0,
                        help="Experiment ID for LiRA")
    parser.add_argument("--num_experiments", type=int, default=64,
                        help="Total number of LiRA experiments")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Mode selection
    parser.add_argument("--mode", type=str, default="eval",
                        choices=["train", "eval", "aggregate"],
                        help="Mode: train (train shadow model), eval (run attack), aggregate (combine results & plot)")
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

    # Handle aggregate mode separately (no model/data needed)
    if args.mode == "aggregate":
        base_output_dir = args.output_dir
        agg_results = aggregate_lira_results(
            base_dir=base_output_dir,
            target=args.target,
            grouping=args.grouping,
            model=args.model,
            use_embedding=args.use_embedding,
            freeze_backbone=args.freeze_backbone,
            min_variants_per_gene=args.min_variants_per_gene,
            max_variants_per_gene=args.max_variants_per_gene,
            disease_subset_file=args.disease_subset_file,
            num_experiments=args.num_experiments,
        )
        return agg_results

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory (include expid for train/eval modes)
    # Extract model name from path (e.g., "./root/models/nt-500m" -> "nt-500m")
    model_name = os.path.basename(args.model.rstrip('/'))
    emb_suffix = "_emb" if args.use_embedding else "_lik"
    freeze_suffix = "_head" if args.freeze_backbone else "_full"

    # Add data filtering parameters to experiment name (only if non-default)
    data_suffix = ""
    if args.min_variants_per_gene != 5 or args.max_variants_per_gene != 50:
        data_suffix += f"_v{args.min_variants_per_gene}-{args.max_variants_per_gene}"
    if args.disease_subset_file:
        subset_name = os.path.basename(args.disease_subset_file).replace('.txt', '').replace('_related_diseases', '')
        data_suffix += f"_{subset_name}"

    base_output_dir = args.output_dir
    experiment_name = f"{args.target}_{args.grouping}_{model_name}{emb_suffix}{freeze_suffix}{data_suffix}"
    args.output_dir = f"{args.output_dir}/{experiment_name}/exp{args.expid}_{args.num_experiments}"
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Arguments: {args}")

    # Load model and tokenizer
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(
        model=args.model,
        num_classes=2,  # Default for CLNSIG, will be updated for CLNDN after data loading
        device=str(device),
        base_model_id=args.base_model
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

    # Update classification head if num_classes differs from default (2)
    if num_classes != 2:
        print(f"\nReinitializing classification head for {num_classes} classes...")
        # Replace the final layer of the classification head
        old_head = model.classification_head
        # Get input features from the second-to-last layer (Linear -> ReLU -> Dropout -> Linear)
        # The last layer is at index -1, it's nn.Linear(128, old_num_classes)
        model.classification_head = nn.Sequential(
            old_head[0],  # nn.Linear(hidden_size, 128)
            old_head[1],  # nn.ReLU()
            old_head[2],  # nn.Dropout(0.1)
            nn.Linear(128, num_classes)  # New final layer with correct num_classes
        )
        # Initialize the new final layer
        nn.init.xavier_uniform_(model.classification_head[-1].weight)
        nn.init.constant_(model.classification_head[-1].bias, 0.0)
        model = model.to(device)
        print(f"Classification head updated: 2 -> {num_classes} classes")

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
    if args.mode == "train":
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

        # Evaluate on full dataset and save per-sample metrics (for LiRA)
        print("\n" + "="*70)
        print("Evaluating on Full Dataset (for LiRA membership inference)")
        print("="*70)

        full_metrics = evaluate_for_lira(
            model=model,
            dataset=full_dataset,
            device=device,
            batch_size=args.batch_size * 2,
        )

        # Save full metrics for LiRA aggregation
        metrics_path = os.path.join(args.output_dir, f"full_metrics_{args.target}_{args.grouping}.pkl")
        with open(metrics_path, 'wb') as f:
            pickle.dump(full_metrics, f)
        print(f"Full metrics saved to: {metrics_path}")
        print(f"  Accuracy: {full_metrics['accuracy']:.4f}")
        print(f"  Mean loss: {full_metrics['mean_loss']:.4f}")
        print(f"  Samples: {len(full_metrics['all_preds'])}")

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
