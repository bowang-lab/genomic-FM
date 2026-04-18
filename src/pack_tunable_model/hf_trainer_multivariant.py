"""
Multi-Variant Per Patient Training Script

Trains a model that handles multiple variants per patient within or across
context windows, supporting both disease classification and pathogenicity prediction.

Two processing modes:
1. Local Mode: Variants within same context window (~1kb)
2. Aggregated Mode: Variants on different chromosomes/genes (attention aggregation)
"""

import os
import torch
import argparse
import numpy as np
import sklearn.metrics
import logging
from typing import Dict, Optional, Any

from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    TrainingArguments,
    AutoTokenizer,
    Trainer,
    set_seed,
)

from .multi_variant_dataloader import (
    MultiVariantPatientDataset,
    MultiVariantDataCollator,
)
from .multi_variant_model import WrappedModelWithMultiVariantHead
from ..geneRepEng.dataset.cgc_primary_findings import (
    load_cgc_by_patient,
    load_cgc_controls_by_patient,
    PatientVariantSample,
    DISEASE_CLASSES,
)
from ..sequence_extractor import GenomeSequenceExtractor


class MultiVariantTrainer(Trainer):
    """Custom trainer for multi-variant model with proper metric computation."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss from model outputs."""
        outputs = model(**inputs)
        loss = outputs.get("loss")

        if loss is None:
            raise ValueError("Model did not return a loss")

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step for multi-variant model."""
        model.eval()

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs.get("loss")
        if loss is not None:
            loss = loss.detach()

        if prediction_loss_only:
            return (loss, None, None)

        # Get logits
        disease_logits = outputs.get("disease_logits")
        pathogenicity_logits = outputs.get("pathogenicity_logits")

        # Get labels from inputs
        disease_labels = None
        pathogenicity_labels = None

        if inputs.get("has_local"):
            if "local_disease_labels" in inputs:
                disease_labels = inputs["local_disease_labels"]
            if "local_pathogenicity_labels" in inputs:
                pathogenicity_labels = inputs["local_pathogenicity_labels"]

        if inputs.get("has_aggregated"):
            if "agg_disease_labels" in inputs:
                agg_disease = inputs["agg_disease_labels"]
                disease_labels = torch.cat([disease_labels, agg_disease]) if disease_labels is not None else agg_disease
            if "agg_pathogenicity_labels" in inputs:
                agg_path = inputs["agg_pathogenicity_labels"]
                pathogenicity_labels = torch.cat([pathogenicity_labels, agg_path]) if pathogenicity_labels is not None else agg_path

        # Stack logits and labels for metric computation
        # Use pathogenicity as primary metric target
        logits = pathogenicity_logits if pathogenicity_logits is not None else disease_logits
        labels = pathogenicity_labels if pathogenicity_labels is not None else disease_labels

        if logits is not None:
            logits = logits.detach()
        if labels is not None:
            labels = labels.detach()

        return (loss, logits, labels)


def calculate_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calculate classification metrics."""
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    if len(valid_labels) == 0:
        return {"accuracy": 0.0, "f1": 0.0, "matthews_correlation": 0.0}

    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }


def preprocess_logits_for_metrics(logits: torch.Tensor, labels: Optional[torch.Tensor] = None):
    """Convert logits to predictions."""
    if logits is None:
        if labels is not None:
            return torch.tensor([], device=labels.device)
        return torch.tensor([])
    return torch.argmax(logits, dim=-1)


def compute_metrics(eval_pred):
    """Compute metrics from evaluation predictions."""
    predictions, labels = eval_pred
    return calculate_metrics(predictions, labels)


def get_model_and_tokenizer(model_type: str):
    """Load base model and tokenizer."""
    local_model_base = f"./root/models/{model_type}"

    model_configs = {
        'omni_dna_116m': ("zehui127/Omni-DNA-116M", "causal"),
        'nt': ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", "masked"),
        'dnabert2': ("zhihan1996/DNABERT-2-117M", "encoder"),
        'hyenadna': ("LongSafari/hyenadna-medium-160k-seqlen-hf", "encoder"),
        'caduceus': ("kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16", "encoder"),
        'gena-lm': ("AIRI-Institute/gena-lm-bert-base-t2t", "encoder"),
        'gpn-star': ("songlab/gpn-star-hg38-v100-200m", "masked"),
    }

    if model_type not in model_configs:
        raise ValueError(f"Unsupported model type: {model_type}")

    hf_path, model_class = model_configs[model_type]
    model_path = local_model_base if os.path.exists(local_model_base) else hf_path
    use_local = os.path.exists(model_path)

    print(f"Loading model from {model_path}")

    if model_class == "masked":
        model = AutoModelForMaskedLM.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=use_local
        )
    elif model_class == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=use_local
        )
    else:
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=use_local,
            add_pooling_layer=False
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=use_local
    )

    return model, tokenizer


def load_patient_datasets(
    tokenizer,
    genome_extractor: GenomeSequenceExtractor,
    seq_length: int = 1024,
    max_variants: int = 10,
    mode: str = "auto",
    val_split: float = 0.15,
    test_split: float = 0.15,
    include_controls: bool = True,
    n_controls: int = 500,
    seed: int = 42,
):
    """Load CGC patient-level datasets for multi-variant training."""
    import random
    from sklearn.model_selection import train_test_split

    # Load pathogenic patients
    print("Loading CGC pathogenic patients...")
    pathogenic_samples = load_cgc_by_patient(max_variants_per_patient=max_variants)
    pathogenic_list = list(pathogenic_samples.values())
    print(f"Loaded {len(pathogenic_list)} pathogenic patients")

    # Optionally load control patients
    if include_controls:
        print("Loading CGC control patients...")
        control_samples = load_cgc_controls_by_patient(
            max_variants_per_patient=max_variants,
            n_patients=n_controls,
            seed=seed
        )
        control_list = list(control_samples.values())
        print(f"Loaded {len(control_list)} control patients")
    else:
        control_list = []

    # Combine and shuffle
    all_samples = pathogenic_list + control_list
    random.seed(seed)
    random.shuffle(all_samples)

    print(f"Total patients: {len(all_samples)}")

    # Filter samples with valid disease classes for stratification
    samples_with_disease = [s for s in all_samples if s.disease_class is not None]
    samples_without_disease = [s for s in all_samples if s.disease_class is None]

    # Stratified split on samples with disease labels
    if len(samples_with_disease) > 0:
        disease_labels = [s.disease_class for s in samples_with_disease]

        # Train/temp split
        train_samples, temp_samples = train_test_split(
            samples_with_disease,
            test_size=val_split + test_split,
            stratify=disease_labels,
            random_state=seed
        )

        # Val/test split
        temp_labels = [s.disease_class for s in temp_samples]
        val_samples, test_samples = train_test_split(
            temp_samples,
            test_size=test_split / (val_split + test_split),
            stratify=temp_labels,
            random_state=seed
        )
    else:
        # Random split if no disease labels
        n_total = len(all_samples)
        n_test = int(n_total * test_split)
        n_val = int(n_total * val_split)

        test_samples = all_samples[:n_test]
        val_samples = all_samples[n_test:n_test + n_val]
        train_samples = all_samples[n_test + n_val:]

    # Add samples without disease labels to training
    train_samples.extend(samples_without_disease)
    random.shuffle(train_samples)

    print(f"Splits - Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Create datasets
    train_dataset = MultiVariantPatientDataset(
        train_samples, tokenizer, genome_extractor, seq_length, max_variants, mode
    )
    val_dataset = MultiVariantPatientDataset(
        val_samples, tokenizer, genome_extractor, seq_length, max_variants, mode
    )
    test_dataset = MultiVariantPatientDataset(
        test_samples, tokenizer, genome_extractor, seq_length, max_variants, mode
    )

    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
    }


def run_multivariant_training(
    seed: int = 42,
    model_type: str = 'nt',
    decoder: bool = False,
    learning_rate: float = 5e-6,
    batch_size: int = 4,
    num_epochs: int = 10,
    max_grad_norm: float = 1.0,
    num_workers: int = 4,
    seq_length: int = 1024,
    max_variants: int = 10,
    mode: str = "auto",
    include_controls: bool = True,
    n_controls: int = 500,
    disease_weight: float = 1.0,
    pathogenicity_weight: float = 1.0,
    attention_heads: int = 4,
    dropout: float = 0.1,
    genome_fa: str = "root/data/hg19.fa",
):
    """Run multi-variant per patient training."""
    set_seed(seed)

    path_prefix = "./root/models"
    output_dir = f"{path_prefix}/multivariant_{model_type}_mode_{mode}_seed_{seed}"

    print(f"\n{'='*60}")
    print("Multi-Variant Per Patient Training")
    print(f"{'='*60}")
    print(f"Model: {model_type}")
    print(f"Mode: {mode}")
    print(f"Max variants: {max_variants}")
    print(f"Sequence length: {seq_length}")
    print(f"Include controls: {include_controls}")
    print(f"{'='*60}\n")

    # Load model and tokenizer
    base_model, tokenizer = get_model_and_tokenizer(model_type)

    # Initialize genome extractor
    genome_extractor = GenomeSequenceExtractor(fasta_file=genome_fa)

    # Load datasets
    datasets = load_patient_datasets(
        tokenizer=tokenizer,
        genome_extractor=genome_extractor,
        seq_length=seq_length,
        max_variants=max_variants,
        mode=mode,
        include_controls=include_controls,
        n_controls=n_controls,
        seed=seed,
    )

    # Create wrapped model
    model = WrappedModelWithMultiVariantHead(
        base_model=base_model,
        num_disease_classes=len(DISEASE_CLASSES),
        num_pathogenicity_classes=2,
        max_variants=max_variants,
        decoder=decoder,
        attention_heads=attention_heads,
        dropout=dropout,
        disease_weight=disease_weight,
        pathogenicity_weight=pathogenicity_weight,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
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
        dataloader_num_workers=num_workers,
        ddp_find_unused_parameters=True,  # Required for attention aggregation
        logging_steps=50,
        report_to="none",
    )

    # Data collator
    data_collator = MultiVariantDataCollator(
        tokenizer=tokenizer,
        max_variants=max_variants,
        num_disease_classes=len(DISEASE_CLASSES),
    )

    # Create trainer
    trainer = MultiVariantTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'],
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=datasets['test'])
    print(f"Test metrics: {test_metrics}")

    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to: {output_dir}")

    return trainer, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Multi-Variant Per Patient Training")

    # Model arguments
    parser.add_argument("--model", type=str, default='nt', help="Model type")
    parser.add_argument("--decoder", action="store_true", help="Decoder architecture")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Data arguments
    parser.add_argument("--seq_length", type=int, default=1024, help="Sequence length")
    parser.add_argument("--max_variants", type=int, default=10, help="Max variants per patient")
    parser.add_argument("--mode", type=str, default="auto", choices=["local", "aggregated", "auto"],
                        help="Processing mode")
    parser.add_argument("--include_controls", action="store_true", help="Include control patients")
    parser.add_argument("--n_controls", type=int, default=500, help="Number of control patients")
    parser.add_argument("--genome_fa", type=str, default="root/data/hg19.fa", help="Reference genome path")

    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data workers")

    # Model architecture arguments
    parser.add_argument("--attention_heads", type=int, default=4, help="Attention heads for aggregation")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--disease_weight", type=float, default=1.0, help="Disease loss weight")
    parser.add_argument("--pathogenicity_weight", type=float, default=1.0, help="Pathogenicity loss weight")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    run_multivariant_training(
        seed=args.seed,
        model_type=args.model,
        decoder=args.decoder,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
        seq_length=args.seq_length,
        max_variants=args.max_variants,
        mode=args.mode,
        include_controls=args.include_controls,
        n_controls=args.n_controls,
        disease_weight=args.disease_weight,
        pathogenicity_weight=args.pathogenicity_weight,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        genome_fa=args.genome_fa,
    )


if __name__ == "__main__":
    main()
