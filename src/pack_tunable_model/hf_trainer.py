import os
import torch
import sys
import csv
import argparse
import numpy as np
import sklearn
import logging
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union
from scipy.stats import spearmanr

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoModelForMaskedLM,
    TrainingArguments,
    AutoTokenizer,
    Trainer,
    set_seed
)
from accelerate import Accelerator
# Import from local modules
from .hf_dataloader import return_clinvar_multitask_dataset, MultiTaskDataCollator, return_smart_dataset, return_maves_dataset
# from .hf_dataloader import return_smart_dataset, MultiTaskDataCollator
# Import our custom wrapper model
from .wrap_model import WrappedModelWithClassificationHead
from .checkpoint_utils import load_checkpoint_into_model


class SafeDistributedTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to handle distributed evaluation properly.
        """
        # If not on main process, return empty dict
        # if not self.is_world_process_zero():
            # Synchronize with other processes but don't do evaluation
            # self.accelerator.wait_for_everyone()
            # return {}
        ignore_keys = ["hidden_states","ref_outputs","alt_outputs"]
        # On main process, perform evaluation and return metrics
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Wait for all processes to sync up after evaluation
        # self.accelerator.wait_for_everyone()
        return metrics


def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray, task_type="classification"):
    valid_mask = labels != -100  # Exclude padding tokens
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    if task_type == "regression":
        # Regression metrics
        # Calculate Spearman correlation safely
        if len(valid_labels) > 1 and len(np.unique(valid_labels)) > 1 and len(np.unique(valid_predictions)) > 1:
            spearman_corr, _ = spearmanr(valid_labels, valid_predictions)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
        else:
            spearman_corr = 0.0

        return {
            "mse": sklearn.metrics.mean_squared_error(valid_labels, valid_predictions),
            "mae": sklearn.metrics.mean_absolute_error(valid_labels, valid_predictions),
            "r2": sklearn.metrics.r2_score(valid_labels, valid_predictions),
            "spearman_correlation": spearman_corr,
        }
    else:
        # Classification metrics
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

def preprocess_logits_for_metrics_old(logits: torch.Tensor, labels: Optional[torch.Tensor] = None):
    """
    Simple function to get predictions from logits.
    """
    predictions = torch.argmax(logits, dim=-1)
    return predictions

def preprocess_logits_for_metrics(logits: torch.Tensor, labels: Optional[torch.Tensor] = None):
    """
    Simple function to get predictions from logits, with None handling.
    """
    if logits is None:
        # Return empty tensor with same device as labels if available
        if labels is not None:
            return torch.tensor([], device=labels.device)
        return torch.tensor([])

    # For regression (1 output), return logits directly; for classification, use argmax
    if logits.shape[-1] == 1:
        # Regression: return raw logits
        predictions = logits.squeeze(-1)
    else:
        # Classification: use argmax
        predictions = torch.argmax(logits, dim=-1)
    return predictions

def compute_metrics(eval_pred, task_type="classification"):
    """
    Compute metrics from predictions and labels.
    """
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels, task_type)

def run_single_task_finetune(task, seed, model_type='nt', decoder=False, test_only=False,
                            learning_rate=0.000005, batch_size=8, num_epochs=10,
                            max_grad_norm=1.0, num_workers=8, gradient_checkpointing=False,
                            filter_genes=None, experimental_methods=None, region_type='all',
                            variant_types=None, seq_length_range=None, max_samples_per_experiment=None,
                            normalize_scores=False, disease_subset_file=None, pretrained_model=None,
                            comparison_mode="delta"):
    set_seed(seed)
    accelerator = Accelerator()
    # Configuration
    path_prefix = "./root/models" # for local use
    # path_prefix = "./root/clinvar_disease_classification"
    results_file = f"{path_prefix}/test_results_clinvar.csv"

    # Model and Tokenizer Selection
    model_path = None
    
    # Check for local models first
    local_model_base = f"./root/models/{model_type}"
    
    if model_type == 'omni_dna_116m':
        # Check if local model exists
        if os.path.exists(local_model_base):
            model_path = local_model_base
            tokenizer_path = model_path
            print(f"Using local model from {model_path}")
        else:
            model_path = "zehui127/Omni-DNA-116M"
            tokenizer_path = "zehui127/Omni-DNA-116M"
            print(f"Using HuggingFace model: {model_path}")
    elif model_type == 'nt':
        # Check if local model exists
        if os.path.exists(local_model_base):
            model_path = local_model_base
            tokenizer_path = model_path
            print(f"Using local model from {model_path}")
        else:
            model_path = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
            tokenizer_path = model_path
            print(f"Using HuggingFace model: {model_path}")
        # if test_only:
            # model_path = "./root/clinvar_disease_classification/checkpoint-55213"
    elif model_type == 'dnabert2':
        # Check if local model exists
        if os.path.exists(local_model_base):
            model_path = local_model_base
            tokenizer_path = model_path
            print(f"Using local model from {model_path}")
        else:
            model_path = "zhihan1996/DNABERT-2-117M"
            tokenizer_path = model_path
            print(f"Using HuggingFace model: {model_path}")
    elif model_type=='hyenadna':
        model_path = "LongSafari/hyenadna-medium-160k-seqlen-hf"
        tokenizer_path = model_path
        print(f"Using HuggingFace HyenaDNA model: {model_path}")
    elif model_type=='caduceus':
        model_path = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
        tokenizer_path = model_path
        print(f"Using HuggingFace Caduceus model: {model_path}")
    elif model_type=='gena-lm':
        model_path = "AIRI-Institute/gena-lm-bert-base-t2t"
        tokenizer_path = model_path
        print(f"Using HuggingFace GENA-LM model: {model_path}")
    elif model_type=='gpn-star':
        # Check if local model exists first
        local_gpn_path = "./root/models/gpn-star-hg38-v100-200m"
        if os.path.exists(local_gpn_path):
            model_path = local_gpn_path
            tokenizer_path = model_path
            print(f"Using local GPN-Star model from {model_path}")
        else:
            model_path = "songlab/gpn-star-hg38-v100-200m"
            tokenizer_path = model_path
            print(f"Using HuggingFace GPN-Star model: {model_path}")
    elif model_type=='luca':
        model_path = "LucaGroup/LucaOne-default-step36M"
        tokenizer_path = model_path
        print(f"Using HuggingFace LucaOne model: {model_path}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if model_type in ['gpn-star', 'nt']:
        base_model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    elif model_type == 'omni_dna_116m':
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    elif model_type == 'luca':
        from lucagplm import LucaGPLMModel, LucaGPLMTokenizer
        base_model = LucaGPLMModel.from_pretrained(model_path)
        tokenizer = LucaGPLMTokenizer.from_pretrained(tokenizer_path)
    else:
        base_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

    if model_type != 'luca':
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )

    # Load checkpoint weights if provided
    checkpoint_suffix = ""
    if pretrained_model:
        # Support both absolute paths and relative paths (directory names in ./root/models/)
        if os.path.isabs(pretrained_model):
            checkpoint_path = pretrained_model
        else:
            checkpoint_path = f"{path_prefix}/{pretrained_model}"
        load_checkpoint_into_model(base_model, checkpoint_path)
        # Add checkpoint suffix for continual learning tracking
        checkpoint_suffix = f"_from_{os.path.basename(checkpoint_path)}"

    ########### Load Dataset ##################
    if task == "MAVES":
        # Load MAVES dataset for regression with filtering
        accelerator.print(f"Loading MAVES dataset with filters:")
        if filter_genes:
            accelerator.print(f"  - Genes: {filter_genes}")
        if experimental_methods:
            accelerator.print(f"  - Methods: {experimental_methods}")
        if region_type != 'all':
            accelerator.print(f"  - Region type: {region_type}")
        if variant_types:
            accelerator.print(f"  - Variant types: {variant_types}")
        if seq_length_range:
            accelerator.print(f"  - Sequence length range: {seq_length_range}")
        if max_samples_per_experiment:
            accelerator.print(f"  - Max samples per experiment: {max_samples_per_experiment}")

        datasets, task_num_classes, max_seq_len = return_maves_dataset(
            tokenizer, target='score', seq_length=1024, seed=seed,
            filter_genes=filter_genes,
            experimental_methods=experimental_methods,
            region_type=region_type,
            variant_types=variant_types,
            seq_length_range=seq_length_range,
            max_samples_per_experiment=max_samples_per_experiment,
            normalize_scores=normalize_scores
        )
        task = "MAVES_score"  # Update task name to match dataset key

        # Log dataset sizes
        for split, dataset in datasets.items():
            accelerator.print(f"  - {split}: {len(dataset)} samples")
    else:
        # Load ClinVar dataset for classification
        datasets, task_num_classes, max_seq_len = return_clinvar_multitask_dataset(
            tokenizer, task, seed=seed, disease_subset_file=disease_subset_file
        )
    tokenizer.model_max_length = max_seq_len
    num_classes = task_num_classes[task]

    # Create wrapped model with classification head
    model = WrappedModelWithClassificationHead(base_model, num_classes, decoder=decoder,
                                                comparison_mode=comparison_mode)
    accelerator.print(f"Using comparison mode: {comparison_mode}")

    # Enable gradient checkpointing if requested
    if gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        accelerator.print("Gradient checkpointing enabled")

    # Load saved model if testing only
    if test_only and os.path.exists(f"{model_path}"):
        state_dict_path = f"{model_path}/pytorch_model.bin"
        if os.path.exists(state_dict_path):
            print(f"Loading weights from {state_dict_path}")
            head_state_dict = torch.load(state_dict_path)
            model.load_state_dict(head_state_dict)
        else:
            print(f"Warning: State dict not found at {state_dict_path}")

    # Create descriptive suffix for model output directory based on filters
    filter_parts = []
    if experimental_methods:
        methods_str = "_".join(experimental_methods[:2])  # Take first 2 to avoid too long names
        filter_parts.append(methods_str)

    if region_type and region_type != 'all':
        filter_parts.append(region_type)

    # Add disease subset identifier if provided
    if disease_subset_file:
        # Extract identifier from filename (e.g., "heart" from "heart_related_diseases.txt")
        subset_name = os.path.basename(disease_subset_file).replace('_related_diseases.txt', '').replace('.txt', '')
        filter_parts.append(subset_name)

    filter_suffix = "_" + "_".join(filter_parts) if filter_parts else ""

    # Set output directory and log it
    output_path = f"{path_prefix}/pretrain_model_{model_type}_{task}{filter_suffix}{checkpoint_suffix}"
    accelerator.print(f"  - Output directory: {output_path}")

    # Prepare Training Arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        run_name=f"{model_type}_{task}{filter_suffix}",  # Add wandb run name with filter suffix
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        save_total_limit=10,
        eval_strategy="epoch",  # Changed from evaluation_strategy
        save_strategy="epoch",
        metric_for_best_model="matthews_correlation",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_safetensors=False,
        remove_unused_columns=False,
        dataloader_num_workers=num_workers,
        ddp_find_unused_parameters=False,  # Set to False for better performance in distributed training
    )
    print(f"Training arguments prediction loss only: {training_args.prediction_loss_only}")
    # Data Collator
    data_collator = MultiTaskDataCollator(tokenizer)

    # Create compute_metrics function for the specific task type
    task_type = "regression" if task.startswith("MAVES") else "classification"

    def compute_metrics_for_task(eval_pred):
        predictions, labels = eval_pred
        return calculate_metric_with_sklearn(predictions, labels, task_type)

    # Update training arguments for regression vs classification
    if task.startswith("MAVES"):
        training_args.metric_for_best_model = "spearman_correlation"  # Use Spearman for regression
        training_args.greater_is_better = True
        training_args.logging_steps = 50  # More frequent logging for debugging
        training_args.warmup_ratio = 0.1  # Add warmup for stable training
        training_args.gradient_accumulation_steps = 2  # Effectively double batch size
    else:
        training_args.metric_for_best_model = "matthews_correlation"  # Use MCC for classification
        training_args.greater_is_better = True

    # Create standard Trainer (no longer need custom trainer)
    trainer = SafeDistributedTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets.get(f"{task}_val"),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics_for_task,
        data_collator=data_collator,
    )

    # Training
    if not test_only:
        trainer.train()
        # Save classification head explicitly (in case save_pretrained doesn't handle it)

    # Evaluation
    test_dataset = datasets.get(f"{task}_test")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print(f"Test Metrics for {task}: {test_metrics}")
    # log the test metrics
    write_header = not os.path.exists(results_file)
    with open(results_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = ["Seed", "Model Type", "Task", "Matthews Correlation"]
            writer.writerow(header)

        row = [seed, model_type, task, test_metrics.get("eval_matthews_correlation", "N/A")]
        writer.writerow(row)

    print(f"Test metrics appended to {results_file}")

def main():
    parser = argparse.ArgumentParser(description="Single-task fine-tune and evaluate model.")
    parser.add_argument("--model", type=str, default='nt',
                        help="Model type (e.g., omni_dna_116m, nt, dnabert2)")
    parser.add_argument("--seed", type=int, default=127,
                        help="Random seed value for training")
    parser.add_argument("--decoder", action="store_true",
                        help="Whether the model has a decoder architecture")
    parser.add_argument("--test_only", action="store_true",
                        help="Only run evaluation on the test set")
    parser.add_argument("--task", type=str, default="CLNSIG",
                        choices=["CLNDN", "CLNSIG", "MAVES"],
                        help="Prediction task: CLNDN (disease classification), CLNSIG (pathogenicity), or MAVES (variant effect regression)")
    parser.add_argument("--disease_subset_file", type=str, default=None,
                        help="Path to text file containing disease names to filter (one per line). E.g., 'heart_related_diseases.txt' for cardiac diseases")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.000005,
                        help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per device for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")

    # Essential MAVES filtering arguments
    parser.add_argument("--filter_genes", type=str, default=None,
                        help="Comma-separated list of genes to filter (e.g., 'BRCA1,TP53,EGFR')")
    parser.add_argument("--experimental_methods", type=str, default=None,
                        help="Comma-separated methods or categories (e.g., 'DMS,MPRA' or 'PROMOTER,ENHANCER')")
    parser.add_argument("--region_type", type=str, default='all', choices=['coding', 'non-coding', 'all'],
                        help="Filter by region type: 'coding', 'non-coding', or 'all' (default)")
    parser.add_argument("--variant_types", type=str, default=None,
                        help="Comma-separated variant types (e.g., 'sub,del,ins')")
    parser.add_argument("--seq_len_min", type=int, default=None,
                        help="Minimum sequence length for filtering")
    parser.add_argument("--seq_len_max", type=int, default=1024,
                        help="Maximum sequence length for filtering")
    parser.add_argument("--max_samples_per_experiment", type=int, default=None,
                        help="Maximum samples to use per experiment (for balanced training)")
    parser.add_argument("--normalize_scores", action="store_true",
                        help="Enable score normalization for MAVE regression tasks")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="Checkpoint to load weights from. Can be a directory name in ./root/models/ (e.g., 'pretrain_model_nt_MAVES_score_DMS') or an absolute path. Automatically uses best checkpoint if multiple exist.")
    parser.add_argument("--comparison_mode", type=str, default="delta",
                        choices=["delta", "concat"],
                        help="How to compare ref/alt embeddings: 'delta' (subtraction) or 'concat' (concatenation like DYNA)")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Parse essential filter arguments
    filter_genes = args.filter_genes.split(',') if args.filter_genes else None

    # Parse experimental methods (simple comma-separated list)
    experimental_methods = args.experimental_methods.split(',') if args.experimental_methods else None

    # Validate experimental methods if provided
    if experimental_methods:
        from ..dataloader.mave_utils import get_method_categories, expand_method_filters
        valid_categories = get_method_categories()
        for method in experimental_methods:
            if method.upper() not in valid_categories:
                # Check if it's a valid individual method by trying to expand
                try:
                    expand_method_filters([method])
                except:
                    print(f"Warning: '{method}' is not a recognized MAVES category or method")
                    print(f"Available categories: {', '.join(valid_categories)}")
        print(f"Using experimental methods filter: {experimental_methods}")

    region_type = args.region_type

    variant_types = None
    if args.variant_types:
        variant_types = [v.strip() for v in args.variant_types.split(',')]
        print(f"Using variant types filter: {variant_types}")

    seq_length_range = None
    if args.seq_len_min is not None or args.seq_len_max is not None:
        seq_length_range = (
            args.seq_len_min if args.seq_len_min is not None else 0,
            args.seq_len_max if args.seq_len_max is not None else float('inf')
        )

    max_samples_per_experiment = args.max_samples_per_experiment
    normalize_scores = args.normalize_scores

    # Run with specified task and essential filters
    run_single_task_finetune(args.task, args.seed, args.model, args.decoder, args.test_only,
                            learning_rate=args.learning_rate,
                            batch_size=args.batch_size,
                            num_epochs=args.num_epochs,
                            max_grad_norm=args.max_grad_norm,
                            num_workers=args.num_workers,
                            gradient_checkpointing=args.gradient_checkpointing,
                            filter_genes=filter_genes,
                            experimental_methods=experimental_methods,
                            region_type=region_type,
                            variant_types=variant_types,
                            seq_length_range=seq_length_range,
                            max_samples_per_experiment=max_samples_per_experiment,
                            normalize_scores=normalize_scores,
                            disease_subset_file=args.disease_subset_file,
                            pretrained_model=args.pretrained_model,
                            comparison_mode=args.comparison_mode)

if __name__ == "__main__":
    main()
