"""
CardioBoost Cardiac Variant Classification Trainer

Trains models on CardioBoost dataset for cardiac disease variant classification.
Supports cardiomyopathy (CM) and arrhythmia (ARM) variants.
"""

import os
import torch
import argparse
import numpy as np
import sklearn
from typing import Optional

from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    TrainingArguments,
    AutoTokenizer,
    Trainer,
    set_seed
)

from .hf_dataloader import (
    return_cardioboost_dataset,
    MultiTaskDataCollator,
)
from .wrap_model import WrappedModelWithClassificationHead


class CardioBoostTrainer(Trainer):
    """Custom trainer with proper evaluation handling."""

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        ignore_keys = ["hidden_states", "ref_outputs", "alt_outputs"]
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        return metrics


def calculate_metrics(predictions: np.ndarray, labels: np.ndarray):
    """Calculate classification metrics."""
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(valid_labels, valid_predictions, average="binary", zero_division=0),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(valid_labels, valid_predictions),
        "precision": sklearn.metrics.precision_score(valid_labels, valid_predictions, average="binary", zero_division=0),
        "recall": sklearn.metrics.recall_score(valid_labels, valid_predictions, average="binary", zero_division=0),
        "roc_auc": sklearn.metrics.roc_auc_score(valid_labels, valid_predictions) if len(np.unique(valid_labels)) > 1 else 0.0,
    }


def preprocess_logits_for_metrics(logits: torch.Tensor, labels: Optional[torch.Tensor] = None):
    """Get predictions from logits."""
    if logits is None:
        if labels is not None:
            return torch.tensor([], device=labels.device)
        return torch.tensor([])
    return torch.argmax(logits, dim=-1)


def compute_metrics(eval_pred):
    """Compute metrics from predictions and labels."""
    predictions, labels = eval_pred
    return calculate_metrics(predictions, labels)


def get_model_and_tokenizer(model_type: str):
    """Load base model and tokenizer."""
    local_model_base = f"./root/models/{model_type}"

    if model_type == 'omni_dna_116m':
        model_path = local_model_base if os.path.exists(local_model_base) else "zehui127/Omni-DNA-116M"
    elif model_type == 'nt':
        model_path = local_model_base if os.path.exists(local_model_base) else "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    elif model_type == 'dnabert2':
        model_path = local_model_base if os.path.exists(local_model_base) else "zhihan1996/DNABERT-2-117M"
    elif model_type == 'hyenadna':
        model_path = "LongSafari/hyenadna-medium-160k-seqlen-hf"
    elif model_type == 'caduceus':
        model_path = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
    elif model_type == 'gena-lm':
        model_path = "AIRI-Institute/gena-lm-bert-base-t2t"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load model with appropriate class
    if model_type in ['nt']:
        model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
    elif model_type == 'omni_dna_116m':
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"Loaded model from {model_path}")

    return model, tokenizer


def train_cardioboost(
    model_type: str = 'omni_dna_116m',
    disease_type: str = 'cm',
    data_dir: str = None,
    seq_length: int = 1024,
    seed: int = 42,
    decoder: bool = False,
    learning_rate: float = 0.00001,
    batch_size: int = 16,
    num_epochs: int = 50,
    include_vus: bool = False,
    checkpoint_path: str = None,
):
    """
    Train a model on CardioBoost dataset.

    Args:
        model_type: Model architecture to use
        disease_type: 'cm' (cardiomyopathy) or 'arm' (arrhythmia)
        data_dir: Path to CardioBoost data directory
        seq_length: DNA sequence length
        seed: Random seed
        decoder: Whether to use decoder-style model
        learning_rate: Learning rate
        batch_size: Batch size per device
        num_epochs: Number of training epochs
        include_vus: Whether to include VUS variants in training
        checkpoint_path: Path to pretrained checkpoint to load
    """
    set_seed(seed)

    # Load model and tokenizer
    base_model, tokenizer = get_model_and_tokenizer(model_type)

    # Load checkpoint weights if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint_dirs = [d for d in os.listdir(checkpoint_path)
                          if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_path, d))]

        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
            latest_checkpoint = checkpoint_dirs[-1]
            checkpoint_weights_path = os.path.join(checkpoint_path, latest_checkpoint, "pytorch_model.bin")

            if os.path.exists(checkpoint_weights_path):
                print(f"Loading checkpoint weights from {checkpoint_weights_path}")
                checkpoint_state = torch.load(checkpoint_weights_path, map_location="cpu")
                base_model.load_state_dict(checkpoint_state, strict=False)
                print("Successfully loaded checkpoint weights")

    # Load dataset
    task_name = f"cardioboost_{disease_type.upper()}"
    datasets, task_num_classes, max_seq_len = return_cardioboost_dataset(
        tokenizer=tokenizer,
        data_dir=data_dir,
        disease_type=disease_type,
        seq_length=seq_length,
        seed=seed,
        include_vus=include_vus,
    )

    num_classes = task_num_classes[task_name]

    # Create wrapped model with classification head
    model = WrappedModelWithClassificationHead(base_model, num_classes, decoder=decoder)

    # Training arguments
    output_dir = f"./root/models/cardioboost_{model_type}_{disease_type}_{seq_length}"
    if include_vus:
        output_dir += "_with_vus"

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        save_total_limit=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="matthews_correlation",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_safetensors=False,
        remove_unused_columns=False,
        logging_steps=10,
        warmup_ratio=0.1,
        weight_decay=0.01,
    )

    # Data collator
    data_collator = MultiTaskDataCollator(tokenizer)

    # Create trainer
    trainer = CardioBoostTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets.get(f"{task_name}_val"),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Train
    print(f"\n{'='*60}")
    print(f"Training CardioBoost {disease_type.upper()} Classifier")
    print(f"Model: {model_type}")
    print(f"Sequence Length: {seq_length}")
    print(f"Include VUS: {include_vus}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    trainer.train()

    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)

    test_dataset = datasets.get(f"{task_name}_test")
    if test_dataset:
        test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
        print("\nTest Results:")
        for key, value in test_results.items():
            print(f"  {key}: {value:.4f}")

    # Save final model
    trainer.save_model(output_dir)
    print(f"\nModel saved to {output_dir}")

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CardioBoost Cardiac Variant Classifier Training")

    parser.add_argument("--model", type=str, default="omni_dna_116m",
                        choices=["omni_dna_116m", "nt", "dnabert2", "hyenadna", "caduceus", "gena-lm"],
                        help="Model architecture to use")
    parser.add_argument("--disease_type", type=str, default="cm",
                        choices=["cm", "arm"],
                        help="Disease type: cm (cardiomyopathy) or arm (arrhythmia)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to CardioBoost data directory")
    parser.add_argument("--seq_length", type=int, default=1024,
                        choices=[512, 1024, 2048, 4096],
                        help="DNA sequence length (default 1024 to match CGC/SMART)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--decoder", action="store_true",
                        help="Use decoder-style model (for autoregressive models)")
    parser.add_argument("--learning_rate", type=float, default=0.00001,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per device")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--include_vus", action="store_true",
                        help="Include VUS variants in training")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to pretrained checkpoint to initialize from")

    args = parser.parse_args()

    train_cardioboost(
        model_type=args.model,
        disease_type=args.disease_type,
        data_dir=args.data_dir,
        seq_length=args.seq_length,
        seed=args.seed,
        decoder=args.decoder,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        include_vus=args.include_vus,
        checkpoint_path=args.checkpoint_path,
    )
