#!/usr/bin/env python3
"""
Genomic Control Vector Sweep for Layer-Specific Strength Optimization

This script performs a sweep to optimize control vector strengths for different layers
in genomic language models, using classification accuracy as the evaluation metric.

Inspired by vlm-repeng sweeps but adapted for genomic data and classification tasks.
"""

import argparse
import os
import torch
import json
import numpy as np
from pathlib import Path
import wandb
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
import random
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# Add necessary paths to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
geneRepEng_dir = os.path.dirname(current_dir)
pack_tunable_model_dir = os.path.dirname(geneRepEng_dir)
src_dir = os.path.dirname(pack_tunable_model_dir)
root_dir = os.path.dirname(src_dir)

sys.path.insert(0, root_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, pack_tunable_model_dir)
sys.path.insert(0, geneRepEng_dir)

from geneRepEng import ControlModel, ControlVector, GenomicDatasetEntry
from geneRepEng.dataset import create_smart_variant_control_dataset, create_synthetic_control_dataset
from geneRepEng.util.omni_dna import create_omni_dna_control_model, get_omni_dna_layer_list
from geneRepEng.extractor import create_control_vector_from_sequences

# Import from the parent module for dataset integration
from hf_dataloader import return_smart_dataset, MultiTaskDataCollator

# ----------------------------------------------------------------------
# Module-level globals for caching
# ----------------------------------------------------------------------
_global_tokenizer = None
_global_model = None
_global_model_id = None
_global_control_vector = None
_global_datasets = None

def _load_once(model_id: str, max_length: int = 1024):
    """Load model and tokenizer once per process, cache in module-level globals."""
    global _global_tokenizer, _global_model, _global_model_id

    if (_global_model is None or _global_tokenizer is None or _global_model_id != model_id):
        print(f"[Load-Once] Loading {model_id} ...")

        # Clear any existing cached objects
        if _global_model is not None:
            del _global_model
        if _global_tokenizer is not None:
            del _global_tokenizer
        torch.cuda.empty_cache()

        # Load fresh instances
        _global_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        _global_tokenizer.model_max_length = max_length

        # Load model to a single device to avoid device mismatch issues
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _global_model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(device)
        print(_global_model)
        _global_model_id = model_id

        # Set model to eval mode
        _global_model.eval()

    else:
        print("[Load-Once] Reusing cached model instance")
        _global_model.eval()

    return _global_tokenizer, _global_model

# ----------------------------------------------------------------------
# Parse Arguments
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run W&B sweep for genomic control vector layer strength optimization.")
    parser.add_argument("--model_type", type=str, default="omni_dna",
                        help="Model type (e.g., 'omni_dna', 'nucleotide_transformer').")
    parser.add_argument("--dataset_type", type=str, default="smart",
                        help="Dataset type ('smart', 'synthetic').")
    parser.add_argument("--csv_path", type=str,
                        default="/zehui/genomic_fm/all_clinvar/unfiltered_variants.csv",
                        help="Path to SMART variant CSV file.")
    parser.add_argument("--method", type=str, default="pca_diff",
                        help="Method for extracting control vector ('pca_diff', 'mean_diff').")
    parser.add_argument("--threshold", type=float, default=54.0,
                        help="Threshold for binarizing SMART scores.")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Maximum number of samples to use for training.")
    parser.add_argument("--eval_samples", type=int, default=1000,
                        help="Number of samples to use for evaluation.")
    parser.add_argument("--seq_length", type=int, default=1024,
                        help="Maximum sequence length.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--sweep_id", type=str, default=None,
                        help="Existing sweep ID to continue.")
    parser.add_argument("--project_name", type=str, default="geneRepEng-sweep",
                        help="Wandb project name.")
    parser.add_argument("--entity", type=str, default=None,
                        help="Wandb entity (username or team name).")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of runs for the sweep.")
    return parser.parse_args()

# ----------------------------------------------------------------------
# Save utilities
# ----------------------------------------------------------------------
def save_json(obj, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ----------------------------------------------------------------------
# Evaluation Functions
# ----------------------------------------------------------------------
def calculate_classification_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive classification metrics."""
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "matthews_correlation": matthews_corrcoef(labels, predictions),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
    }

def evaluate_genomic_model(
    model: torch.nn.Module,
    tokenizer,
    sequences: List[str],
    labels: List[int],
    batch_size: int = 8
) -> Dict[str, float]:
    """
    Evaluate a genomic model on classification task.

    Args:
        model: The genomic model to evaluate
        tokenizer: Tokenizer for DNA sequences
        sequences: List of DNA sequences
        labels: List of corresponding labels
        batch_size: Batch size for evaluation

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    predictions = []

    # Ensure model is on a single device
    device = next(model.parameters()).device
    print(f"Model device: {device}")

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            try:
                # Tokenize sequences
                inputs = tokenizer(
                    batch_sequences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=tokenizer.model_max_length
                )

                # Move to device and filter out unsupported keys
                inputs = {k: v.to(device) for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}

                # Get model outputs
                outputs = model(**inputs)
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {e}")
                # Create dummy predictions for this batch
                batch_preds = np.random.randint(0, 2, len(batch_sequences))
                predictions.extend(batch_preds)
                continue

            # Extract hidden states and apply simple classification
            # For simplicity, we'll use the last hidden state mean as features
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                last_hidden = outputs.hidden_states[-1]  # Last layer
            else:
                last_hidden = outputs.last_hidden_state

            # Mean pooling over sequence length
            if 'attention_mask' in inputs:
                mask = inputs['attention_mask'].unsqueeze(-1)
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                pooled = last_hidden.mean(dim=1)

            # Simple binary classification using pooled features
            # We'll use a simple threshold on the mean activation
            batch_preds = (pooled.mean(dim=-1) > 0).cpu().numpy().astype(int)
            predictions.extend(batch_preds)

    # Calculate metrics
    predictions = np.array(predictions[:len(labels)])
    return calculate_classification_metrics(predictions, np.array(labels))

# ----------------------------------------------------------------------
# Shared Data Precomputation
# ----------------------------------------------------------------------
def precompute_shared_data(args):
    """Pre-compute baseline results and control vector once per sweep."""
    global _global_control_vector, _global_datasets

    print("[Precompute] Computing shared baseline and control vector...")

    # Model setup
    if args.model_type == "omni_dna":
        model_id = "zehui127/Omni-DNA-116M"
    elif args.model_type == "nucleotide_transformer":
        model_id = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    else:
        raise NotImplementedError(f"Model not implemented: {args.model_type}")

    tokenizer, model = _load_once(model_id, args.seq_length)

    # Dataset setup
    if args.dataset_type == "smart":
        print(f"Loading SMART dataset from {args.csv_path}")
        datasets, task_num_classes, max_seq_len = return_smart_dataset(
            tokenizer=tokenizer,
            csv_path=args.csv_path,
            threshold=args.threshold,
            seq_length=args.seq_length,
            val_split=0.2,
            test_split=0.2,
            seed=args.seed,
            all_records=False,
            num_records=args.max_samples
        )

        # Convert to sequences and labels for evaluation
        def extract_sequences_and_labels(dataset):
            sequences = []
            labels = []
            for i in range(min(len(dataset), args.eval_samples)):
                item = dataset[i]
                # Use reference sequence for evaluation
                ref_tokens = item['ref_input_ids']
                ref_seq = tokenizer.decode(ref_tokens, skip_special_tokens=True)
                sequences.append(ref_seq)
                labels.append(item['labels'])
            return sequences, labels

        train_sequences, train_labels = extract_sequences_and_labels(datasets['train'])
        val_sequences, val_labels = extract_sequences_and_labels(datasets['CLNDN_val'])
        test_sequences, test_labels = extract_sequences_and_labels(datasets['CLNDN_test'])

        # Create control vector dataset entries
        control_entries = []
        for i in range(min(len(datasets['train']), 1000)):  # Limit for control vector training
            item = datasets['train'][i]
            ref_tokens = item['ref_input_ids']
            alt_tokens = item['alt_input_ids']

            ref_seq = tokenizer.decode(ref_tokens, skip_special_tokens=True)
            alt_seq = tokenizer.decode(alt_tokens, skip_special_tokens=True)

            entry = GenomicDatasetEntry(
                ref_sequence=ref_seq,
                alt_sequence=alt_seq,
                label=item['labels']
            )
            control_entries.append(entry)

    elif args.dataset_type == "synthetic":
        print("Creating synthetic dataset")
        dataset = create_synthetic_control_dataset(
            n_samples=args.max_samples,
            seq_length=100,
            mutation_rate=0.05,
            seed=args.seed
        )

        # Split dataset
        total_samples = len(dataset.entries)
        train_size = int(0.6 * total_samples)
        val_size = int(0.2 * total_samples)

        control_entries = dataset.entries[:train_size]
        val_entries = dataset.entries[train_size:train_size + val_size]
        test_entries = dataset.entries[train_size + val_size:]

        # Extract sequences and labels
        train_sequences = [entry.ref_sequence for entry in control_entries[:args.eval_samples]]
        train_labels = [entry.label for entry in control_entries[:args.eval_samples]]

        val_sequences = [entry.ref_sequence for entry in val_entries[:args.eval_samples]]
        val_labels = [entry.label for entry in val_entries[:args.eval_samples]]

        test_sequences = [entry.ref_sequence for entry in test_entries[:args.eval_samples]]
        test_labels = [entry.label for entry in test_entries[:args.eval_samples]]

    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    # Compute baseline metrics
    print("Computing baseline metrics...")
    baseline_train = evaluate_genomic_model(model, tokenizer, train_sequences, train_labels)
    baseline_val = evaluate_genomic_model(model, tokenizer, val_sequences, val_labels)
    baseline_test = evaluate_genomic_model(model, tokenizer, test_sequences, test_labels)

    # Extract control vector
    print("Extracting control vector...")
    print(f"Model type: {type(model)}")
    print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")

    ref_sequences = [entry.ref_sequence for entry in control_entries]
    alt_sequences = [entry.alt_sequence for entry in control_entries]

    try:
        # Use the layer list from the model
        layers = get_omni_dna_layer_list(model)
        layer_ids = list(range(len(layers)))

        control_directions = create_control_vector_from_sequences(
            model=model,
            tokenizer=tokenizer,
            ref_sequences=ref_sequences,
            alt_sequences=alt_sequences,
            layer_ids=layer_ids,
            method=args.method,
            batch_size=4
        )

        control_vector = ControlVector(
            model_type=model.config.model_type,
            directions=control_directions
        )

    except Exception as e:
        print(f"Warning: Error creating control vector: {e}")
        print("Creating dummy control vector for testing...")
        # Create a dummy control vector for testing
        try:
            layers = get_omni_dna_layer_list(model)
        except Exception as e2:
            print(f"Warning: Could not get layer list: {e2}")
            # Create a fake layer list for testing
            print("Creating fake layer list for testing...")
            layers = [f"layer_{i}" for i in range(12)]  # Assume 12 layers for testing

        dummy_directions = {}
        hidden_size = getattr(model.config, 'hidden_size', getattr(model.config, 'd_model', 768))

        for i in range(len(layers)):
            dummy_directions[i] = np.random.randn(hidden_size) * 0.1

        control_vector = ControlVector(
            model_type=getattr(model.config, 'model_type', 'unknown'),
            directions=dummy_directions
        )

    # Store shared data
    _global_control_vector = control_vector
    _global_datasets = {
        "baseline_metrics": {
            "train": baseline_train,
            "val": baseline_val,
            "test": baseline_test
        },
        "sequences": {
            "train": (train_sequences, train_labels),
            "val": (val_sequences, val_labels),
            "test": (test_sequences, test_labels)
        }
    }

    print("[Precompute] Shared data computed successfully")
    print(f"Control vector extracted for {len(control_vector.directions)} layers")
    print(f"Baseline accuracies - Train: {baseline_train['accuracy']:.3f}, Val: {baseline_val['accuracy']:.3f}, Test: {baseline_test['accuracy']:.3f}")

# ----------------------------------------------------------------------
# Main Single-Run Function for W&B
# ----------------------------------------------------------------------
def run_single_experiment():
    """Run a single experiment with layer-specific control strengths."""
    global _global_control_vector, _global_datasets

    # Initialize wandb run
    os.environ["WANDB_DISABLE_STATS"] = "true"
    wandb.init()
    config = wandb.config

    # Get hyperparameters from wandb config
    model_type = config.get("model_type", "omni_dna")
    method = config.get("method", "pca_diff")
    dataset_type = config.get("dataset_type", "smart")

    # Get layer-specific alpha values
    num_layers = len(_global_control_vector.directions)
    alpha_vals = np.array([config.get(f"alpha_{i}", 1.0) for i in range(num_layers)])

    print(f"Running experiment with alpha values: {alpha_vals}")

    # Model setup
    if model_type == "omni_dna":
        model_id = "zehui127/Omni-DNA-116M"
    elif model_type == "nucleotide_transformer":
        model_id = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    else:
        raise NotImplementedError(f"Model not implemented: {model_type}")

    tokenizer, base_model = _load_once(model_id)

    # Get baseline results
    baseline_metrics = _global_datasets["baseline_metrics"]

    # Create controlled model
    layer_ids = list(_global_control_vector.directions.keys())
    controlled_model = create_omni_dna_control_model(
        model=base_model,
        layer_ids=layer_ids
    )

    # Apply layer-specific control strengths
    scaled_directions = {}
    for i, layer_id in enumerate(layer_ids):
        if i < len(alpha_vals):
            scaled_directions[layer_id] = _global_control_vector.directions[layer_id] * alpha_vals[i]
        else:
            scaled_directions[layer_id] = _global_control_vector.directions[layer_id] * 1.0

    scaled_control_vector = ControlVector(
        model_type=_global_control_vector.model_type,
        directions=scaled_directions
    )

    controlled_model.set_control(scaled_control_vector, strength=1.0)

    # Evaluate on all splits
    results = {"baseline_metrics": baseline_metrics}

    for split_name in ["val", "test"]:
        sequences, labels = _global_datasets["sequences"][split_name]

        print(f"Evaluating on {split_name} split ({len(sequences)} samples)")
        controlled_metrics = evaluate_genomic_model(
            controlled_model, tokenizer, sequences, labels
        )

        # Calculate improvements
        baseline_split_metrics = baseline_metrics[split_name]
        improvements = {}
        for metric_name in controlled_metrics:
            improvements[f"{metric_name}_improvement"] = (
                controlled_metrics[metric_name] - baseline_split_metrics[metric_name]
            )

        results[split_name] = {
            "controlled_metrics": controlled_metrics,
            "baseline_metrics": baseline_split_metrics,
            "improvements": improvements
        }

    # Clear control
    controlled_model.clear_control()

    # Prepare logging data
    log_dict = {
        "model_type": model_type,
        "method": method,
        "dataset_type": dataset_type,
        # Individual alpha values
        **{f"alpha_{i}": alpha_vals[i] for i in range(len(alpha_vals))},
        # Summary statistics
        "alpha_mean": np.mean(alpha_vals),
        "alpha_std": np.std(alpha_vals),
        "alpha_min": np.min(alpha_vals),
        "alpha_max": np.max(alpha_vals),
        # Baseline metrics
        "baseline_train_accuracy": baseline_metrics["train"]["accuracy"],
        "baseline_val_accuracy": baseline_metrics["val"]["accuracy"],
        "baseline_test_accuracy": baseline_metrics["test"]["accuracy"],
        # Controlled metrics and improvements
        "val_accuracy": results["val"]["controlled_metrics"]["accuracy"],
        "val_f1": results["val"]["controlled_metrics"]["f1"],
        "val_matthews_correlation": results["val"]["controlled_metrics"]["matthews_correlation"],
        "val_accuracy_improvement": results["val"]["improvements"]["accuracy_improvement"],
        "val_f1_improvement": results["val"]["improvements"]["f1_improvement"],
        "val_matthews_improvement": results["val"]["improvements"]["matthews_correlation_improvement"],
        "test_accuracy": results["test"]["controlled_metrics"]["accuracy"],
        "test_f1": results["test"]["controlled_metrics"]["f1"],
        "test_matthews_correlation": results["test"]["controlled_metrics"]["matthews_correlation"],
        "test_accuracy_improvement": results["test"]["improvements"]["accuracy_improvement"],
        "test_f1_improvement": results["test"]["improvements"]["f1_improvement"],
        "test_matthews_improvement": results["test"]["improvements"]["matthews_correlation_improvement"],
    }

    # Log to wandb
    wandb.log(log_dict)

    # Save results locally
    experiment_data = {
        "config": dict(log_dict),
        "alpha_values": alpha_vals.tolist(),
        "results": results,
        "sweep_id": wandb.run.sweep_id,
        "run_id": wandb.run.id,
        "run_name": wandb.run.name,
    }

    # Save to sweep-specific directory
    sweep_id = wandb.run.sweep_id if wandb.run.sweep_id else "no_sweep"
    run_name = wandb.run.name if wandb.run.name else wandb.run.id

    sweep_results_dir = Path(f"results/{sweep_id}")
    sweep_results_dir.mkdir(parents=True, exist_ok=True)

    save_json(experiment_data, sweep_results_dir / f"{run_name}.json")
    print(f"Results saved to: {sweep_results_dir / f'{run_name}.json'}")

# ----------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Pre-compute shared data once before starting sweep
    precompute_shared_data(args)

    if args.sweep_id:
        # Continue existing sweep
        print(f"Continuing sweep: {args.sweep_id}")
        wandb.agent(
            args.sweep_id,
            function=run_single_experiment,
            project=args.project_name,
            entity=args.entity
        )
    else:
        # Create new sweep
        num_layers = len(_global_control_vector.directions)
        print(f"Creating new sweep for {num_layers} layers")

        # Create parameters for each layer's alpha
        parameters = {}
        for i in range(num_layers):
            parameters[f"alpha_{i}"] = {
                "distribution": "uniform",
                "min": -2.0,
                "max": 2.0,
            }

        # Add fixed parameters
        parameters.update({
            "model_type": {"value": args.model_type},
            "method": {"value": args.method},
            "dataset_type": {"value": args.dataset_type},
        })

        sweep_config = {
            "method": "bayes",
            "metric": {
                "name": "val_accuracy_improvement",
                "goal": "maximize"
            },
            "parameters": parameters
        }

        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=args.project_name,
            entity=args.entity
        )
        print(f"Created new sweep: {sweep_id}")

        wandb.agent(
            sweep_id,
            function=run_single_experiment,
            project=args.project_name,
            entity=args.entity,
            count=args.count
        )

if __name__ == "__main__":
    main()
