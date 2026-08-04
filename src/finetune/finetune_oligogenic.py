#!/usr/bin/env python3
"""
Fine-tuning Script for Oligogenic Interaction Prediction

Trains a paired variant classifier on OLIDA data to predict
P(oligogenic interaction) for variant pairs.

Usage:
    python -m src.finetune.finetune_oligogenic --config configs/oligogenic.yaml --dataset olida_cardiac_nt

    # Or with command-line overrides:
    python -m src.finetune.finetune_oligogenic \
        --model nt \
        --seq_length 1024 \
        --combine_mode hadamard \
        --epochs 20 \
        --batch_size 16
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from src.datasets.olida.cardiac_olida import load_filtered_olida_paired
from src.dataloader.oligogenic_dataset import (
    OligogenicPairedDataset,
    collate_paired_variants,
)
from src.pack_tunable_model.wrap_model import WrappedModelWithPairedVariantHead


# Model identifiers
MODEL_REGISTRY = {
    'nt': 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species',
    'dnabert2': 'zhihan1996/DNABERT-2-117M',
    'hyenadna': 'LongSafari/hyenadna-medium-160k-seqlen-hf',
    'caduceus': 'kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16',
    'omni_dna_116m': 'zehui127/Omni-DNA-116M',
    'gena-lm': 'AIRI-Institute/gena-lm-bert-base-t2t',
}

# Models that use masked LM architecture
MASKED_LM_MODELS = {'nt', 'gpn-star'}
# Models that use causal LM architecture
CAUSAL_LM_MODELS = {'omni_dna_116m', 'omni_dna'}
# Decoder models
DECODER_MODELS = {'hyenadna', 'omni_dna_116m', 'omni_dna'}


def load_base_model(model_name: str, local_path: Optional[Path] = None):
    """Load base model and tokenizer."""
    model_id = MODEL_REGISTRY.get(model_name, model_name)

    # Check for local model
    if local_path and local_path.exists():
        model_path = str(local_path)
    else:
        model_path = model_id

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Select appropriate model class
    if model_name in MASKED_LM_MODELS:
        base_model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
    elif model_name in CAUSAL_LM_MODELS:
        base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    else:
        base_model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    return base_model, tokenizer


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        outputs = model(
            variant1_ref_input_ids=batch['variant1_ref_input_ids'],
            variant1_ref_attention_mask=batch.get('variant1_ref_attention_mask'),
            variant1_alt_input_ids=batch['variant1_alt_input_ids'],
            variant1_alt_attention_mask=batch.get('variant1_alt_attention_mask'),
            variant2_ref_input_ids=batch['variant2_ref_input_ids'],
            variant2_ref_attention_mask=batch.get('variant2_ref_attention_mask'),
            variant2_alt_input_ids=batch['variant2_alt_input_ids'],
            variant2_alt_attention_mask=batch.get('variant2_alt_attention_mask'),
            labels=batch['labels'],
        )

        loss = outputs['loss']
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Collect predictions
        with torch.no_grad():
            probs = torch.softmax(outputs['logits'], dim=-1)[:, 1]
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0
    pred_labels = (all_preds > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, pred_labels, average='binary', zero_division=0
    )

    return {
        'loss': avg_loss,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: str) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        outputs = model(
            variant1_ref_input_ids=batch['variant1_ref_input_ids'],
            variant1_ref_attention_mask=batch.get('variant1_ref_attention_mask'),
            variant1_alt_input_ids=batch['variant1_alt_input_ids'],
            variant1_alt_attention_mask=batch.get('variant1_alt_attention_mask'),
            variant2_ref_input_ids=batch['variant2_ref_input_ids'],
            variant2_ref_attention_mask=batch.get('variant2_ref_attention_mask'),
            variant2_alt_input_ids=batch['variant2_alt_input_ids'],
            variant2_alt_attention_mask=batch.get('variant2_alt_attention_mask'),
            labels=batch['labels'],
        )

        total_loss += outputs['loss'].item()
        probs = torch.softmax(outputs['logits'], dim=-1)[:, 1]
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0
    pred_labels = (all_preds > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, pred_labels, average='binary', zero_division=0
    )
    accuracy = (pred_labels == all_labels).mean()

    return {
        'loss': avg_loss,
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def run_training(config: Dict) -> None:
    """Run oligogenic training with given configuration."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load base model
    model_name = config.get('model', 'nt')
    local_path = Path(config.get('local_model_path', '')) if config.get('local_model_path') else None
    base_model, tokenizer = load_base_model(model_name, local_path)

    # Create wrapped model with paired variant head
    is_decoder = model_name in DECODER_MODELS
    model = WrappedModelWithPairedVariantHead(
        base_model=base_model,
        num_classes=2,
        decoder=is_decoder,
        combine_mode=config.get('combine_mode', 'hadamard'),
        pooling=config.get('pooling', 'mean'),
    )
    model.to(device)

    # Load data
    print("Loading OLIDA data...")
    data = load_filtered_olida_paired(
        seq_length=config.get('seq_length', 1024),
        disease_keywords_file=config.get('disease_keywords_file'),
        gene_list_file=config.get('gene_list_file'),
        min_final_meta=config.get('min_final_meta', 1),
        negative_ratio=config.get('negative_ratio', 1.0),
        require_disease_match=config.get('require_disease_match', True),
        require_gene_match=config.get('require_gene_match', False),
    )

    # Split data
    val_ratio = config.get('val_ratio', 0.15)
    n_val = int(len(data) * val_ratio)
    n_train = len(data) - n_val

    train_data, val_data = random_split(data, [n_train, n_val])
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Create datasets
    train_dataset = OligogenicPairedDataset(
        list(train_data), tokenizer, max_length=config.get('seq_length', 1024)
    )
    val_dataset = OligogenicPairedDataset(
        list(val_data), tokenizer, max_length=config.get('seq_length', 1024)
    )

    # Create dataloaders
    batch_size = config.get('batch_size', 16)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_paired_variants,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_paired_variants,
        pin_memory=True,
    )

    # Optimizer and scheduler
    epochs = config.get('epochs', 20)
    lr = config.get('learning_rate', 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    output_dir = Path(config.get('output_dir', './output/oligogenic'))
    output_dir.mkdir(parents=True, exist_ok=True)

    best_auc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics = evaluate(model, val_loader, device)

        print(f"Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_epoch = epoch + 1
            checkpoint_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_auc,
                'config': config,
            }, checkpoint_path)
            print(f"Saved best model (AUC: {best_auc:.4f})")

    print(f"\nTraining complete. Best AUC: {best_auc:.4f} at epoch {best_epoch}")

    # Save final model
    final_path = output_dir / 'final_model.pt'
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, final_path)


def main():
    parser = argparse.ArgumentParser(description='Train oligogenic interaction model')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--dataset', type=str, help='Dataset name from config')
    parser.add_argument('--model', type=str, default='nt', help='Model name')
    parser.add_argument('--seq_length', type=int, default=1024, help='Sequence length')
    parser.add_argument('--combine_mode', type=str, default='hadamard',
                        choices=['concat', 'diff', 'hadamard'], help='Embedding combine mode')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['cls', 'mean', 'last'], help='Pooling strategy')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./output/oligogenic',
                        help='Output directory')
    parser.add_argument('--disease_keywords_file', type=str, help='Path to disease keywords file')
    parser.add_argument('--gene_list_file', type=str, help='Path to gene list file')

    args = parser.parse_args()

    # Load config from YAML if provided
    if args.config:
        with open(args.config, 'r') as f:
            all_configs = yaml.safe_load(f)

        if args.dataset and args.dataset in all_configs:
            config = all_configs[args.dataset]
        else:
            # Use first config if no dataset specified
            config = next(iter(all_configs.values()))
    else:
        config = {}

    # Override with command-line arguments
    for key, value in vars(args).items():
        if value is not None and key not in ['config', 'dataset']:
            config[key] = value

    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    run_training(config)


if __name__ == '__main__':
    main()
