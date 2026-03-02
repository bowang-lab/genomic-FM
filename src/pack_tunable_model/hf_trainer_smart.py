import os
import torch
import sys
import csv
import argparse
import numpy as np
import sklearn
import logging
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import transformers
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    TrainingArguments,
    AutoTokenizer,
    Trainer,
    set_seed
)
from accelerate import Accelerator
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
# Import from local modules
from .hf_dataloader import (
    return_clinvar_multitask_dataset,
    MultiTaskDataCollator,
    return_smart_dataset,
    return_multitask_dataset,
    return_multilabel_dataset,
    MultiLabelDataCollator,
)
from .wrap_model import WrappedModelWithClassificationHead
from ..tunable_model.hf_trainer import MultitaskTrainer, AllHeadsMultitaskTrainer


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


def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -100  # Exclude padding tokens
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
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

    predictions = torch.argmax(logits, dim=-1)
    return predictions

def compute_metrics(eval_pred):
    """
    Compute metrics from predictions and labels.
    """
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)

def get_model_and_tokenizer(model_type: str):
    """Load base model and tokenizer. Uses appropriate loader per model (trainers add their own heads)."""
    from transformers import AutoModelForCausalLM

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
    elif model_type == 'gpn-star':
        local_gpn = "./root/models/gpn-star-hg38-v100-200m"
        model_path = local_gpn if os.path.exists(local_gpn) else "songlab/gpn-star-hg38-v100-200m"
    elif model_type == 'luca':
        from lucagplm import LucaGPLMModel, LucaGPLMTokenizer
        model = LucaGPLMModel.from_pretrained("LucaGroup/LucaOne-default-step36M")
        tokenizer = LucaGPLMTokenizer.from_pretrained("LucaGroup/LucaOne-default-step36M")
        return model, tokenizer
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Use appropriate model class based on what each model supports
    if model_type in ['gpn-star', 'nt']:
        model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    elif model_type == 'omni_dna_116m':
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    else:
        # dnabert2, hyenadna, caduceus, gena-lm support AutoModel
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    print(f"Using model from {model_path}")
    return model, tokenizer


def run_single_task_finetune(task, seed, model_type='nt', decoder=False, test_only=False,
                            learning_rate=0.000005, batch_size=8, num_epochs=10,
                            max_grad_norm=1.0, num_workers=8, threshold=65.0, checkpoint_path=None, checkpoint_step=None,
                            min_samples_per_class=2):
    set_seed(seed)
    accelerator = Accelerator()
    # Configuration
    path_prefix = "./root/models" # for local use
    # path_prefix = "./root/clinvar_disease_classification"
    results_file = f"{path_prefix}/test_results_clinvar.csv"

    # Model and Tokenizer Selection
    model_path = None
    tokenizer_path = None
    checkpoint_weights_path = None  # Path to checkpoint weights to load
    
    # If checkpoint_path is provided, handle it as checkpoint weights to load
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Check if the checkpoint_path contains multiple checkpoint subdirectories
        checkpoint_dirs = [d for d in os.listdir(checkpoint_path) 
                          if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_path, d))]
        
        if checkpoint_dirs:
            if checkpoint_step:
                # Use the specified checkpoint step
                specific_checkpoint = f"checkpoint-{checkpoint_step}"
                if specific_checkpoint in checkpoint_dirs:
                    checkpoint_weights_path = os.path.join(checkpoint_path, specific_checkpoint, "pytorch_model.bin")
                    print(f"Will load ClinVar-trained weights from {checkpoint_weights_path} (step {checkpoint_step})")
                else:
                    available_steps = [int(d.split('-')[1]) for d in checkpoint_dirs]
                    raise ValueError(f"Checkpoint step {checkpoint_step} not found. Available steps: {sorted(available_steps)}")
            else:
                # Sort checkpoint dirs to find the latest one
                checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
                latest_checkpoint = checkpoint_dirs[-1]

                # Try to find the best checkpoint from trainer_state.json inside the latest checkpoint
                trainer_state_path = os.path.join(checkpoint_path, latest_checkpoint, "trainer_state.json")
                best_checkpoint = None

                if os.path.exists(trainer_state_path):
                    import json
                    with open(trainer_state_path, 'r') as f:
                        trainer_state = json.load(f)
                    best_checkpoint_path = trainer_state.get('best_model_checkpoint')
                    if best_checkpoint_path:
                        # Extract just the checkpoint directory name
                        best_checkpoint = os.path.basename(best_checkpoint_path)
                        if best_checkpoint in checkpoint_dirs:
                            checkpoint_weights_path = os.path.join(checkpoint_path, best_checkpoint, "pytorch_model.bin")
                            best_metric = trainer_state.get('best_metric', 'N/A')
                            print(f"Will load ClinVar-trained weights from {checkpoint_weights_path} (best checkpoint, metric={best_metric})")

                if not checkpoint_weights_path:
                    # Fall back to latest checkpoint if no best found
                    checkpoint_weights_path = os.path.join(checkpoint_path, latest_checkpoint, "pytorch_model.bin")
                    print(f"Will load ClinVar-trained weights from {checkpoint_weights_path} (latest from {len(checkpoint_dirs)} checkpoints, no best checkpoint found)")
        else:
            # Check if it's a direct path to a checkpoint directory
            if os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
                checkpoint_weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
                print(f"Will load ClinVar-trained weights from {checkpoint_weights_path}")
            else:
                print(f"Warning: No pytorch_model.bin found in {checkpoint_path}")
    
    # Determine base model path based on model type
    if not model_path:
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

    # Load base model (trainers/wrappers add their own heads)
    from transformers import AutoModelForCausalLM
    if model_type in ['gpn-star', 'nt']:
        base_model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    elif model_type == 'omni_dna_116m':
        base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    elif model_type == 'luca':
        from lucagplm import LucaGPLMModel, LucaGPLMTokenizer
        base_model = LucaGPLMModel.from_pretrained(model_path)
        tokenizer = LucaGPLMTokenizer.from_pretrained(tokenizer_path)
    else:
        base_model = AutoModel.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    
    # Load checkpoint weights if provided
    if checkpoint_weights_path and os.path.exists(checkpoint_weights_path):
        print(f"Loading checkpoint weights from {checkpoint_weights_path}")
        import torch
        checkpoint_state = torch.load(checkpoint_weights_path, map_location="cpu")
        
        # The checkpoint might have a different structure, try to load it
        try:
            # If the checkpoint is a full model state dict
            if isinstance(checkpoint_state, dict) and not any(k.startswith('module.') for k in checkpoint_state.keys()):
                base_model.load_state_dict(checkpoint_state, strict=False)
            else:
                # Try to extract the model state dict from various possible formats
                if 'model_state_dict' in checkpoint_state:
                    base_model.load_state_dict(checkpoint_state['model_state_dict'], strict=False)
                elif 'state_dict' in checkpoint_state:
                    base_model.load_state_dict(checkpoint_state['state_dict'], strict=False)
                else:
                    # Remove 'module.' prefix if present (from DataParallel)
                    state_dict = {k.replace('module.', ''): v for k, v in checkpoint_state.items()}
                    base_model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded checkpoint weights")
        except Exception as e:
            print(f"Warning: Could not load checkpoint weights: {e}")
            print(f"Continuing with base model weights only")
    
    if model_type != 'luca':
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )

    ########### Load Dataset old ##################
        # dangerous zone: so we need to use main_process_first
        # Code in this block is executed by rank-0 first,
        # all other ranks are blocked until rank-0 exits the block.
    # datasets, task_num_classes, max_seq_len = return_clinvar_multitask_dataset(
    #     tokenizer, task, seed=seed
    # )
    # Map task to target for cleaner interface
    target = 'disease' if task == 'CLNDN' else 'score'
    
    datasets, task_num_classes, max_seq_len = return_smart_dataset(
        tokenizer, 'root/data/unfiltered_variants.csv',
        target=target, task_name=task, threshold=threshold, min_samples_per_class=min_samples_per_class
    )
    tokenizer.model_max_length = max_seq_len
        # << all ranks continue here >>
    num_classes = task_num_classes[task]
    ################### Main Process Only ###########################
    # if accelerator.is_main_process:
    #     accelerator.print(f"Loading dataset for task {task} on main process")
    #     datasets, task_num_classes, max_seq_len = return_clinvar_multitask_dataset(
    #         tokenizer, task, seed=seed
    #     )
    #     num_classes = task_num_classes[task]
    # else:
    #     # Dummy values for non-main processes
    #     datasets = {}
    #     task_num_classes = {task: 2}  # Default to binary classification
    #     max_seq_len = 1000
    #     num_classes = 2
    # # Broadcast num_classes from main process to all processes
    # num_classes = accelerator.prepare(torch.tensor([num_classes], device=accelerator.device))[0].item()
    # accelerator.print(f"Loading base model from {model_path}")
    ##############################################

    # Create wrapped model with classification head
    model = WrappedModelWithClassificationHead(base_model, num_classes, decoder=decoder)

    # Model is already loaded with the appropriate checkpoint or base model
    # Prepare Training Arguments
    checkpoint_suffix = f"_from_{os.path.basename(checkpoint_path)}" if checkpoint_path else ""
    training_args = TrainingArguments(
        output_dir=f"{path_prefix}/smart_pretrain_model_{model_type}_{task}_threshold_{int(threshold)}_with_state_dict{checkpoint_suffix}",
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
        save_safetensors=False,  # OmniDNA has weight tying (wte/word_embeddings share memory)
        remove_unused_columns=False,
        dataloader_num_workers=num_workers,
        # ddp_find_unused_parameters=False,  # Set to False for better performance in distributed training
    )
    print(f"Training arguments prediction loss only: {training_args.prediction_loss_only}")
    # Data Collator
    data_collator = MultiTaskDataCollator(tokenizer)

    # Create standard Trainer (no longer need custom trainer)
    trainer = SafeDistributedTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets.get(f"{task}_val"),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
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


def run_multitask_finetune(seed, model_type='nt', decoder=False, learning_rate=0.000005, batch_size=8,
                           num_epochs=10, max_grad_norm=1.0, num_workers=8, threshold=65.0,
                           include_clndn=True, include_clnsig=True, include_maves=False,
                           data_source='smart'):
    """Run multi-task training with any combination of CLNDN, CLNSIG, MAVES (task-routing style)."""
    set_seed(seed)

    path_prefix = "./root/models"
    model, tokenizer = get_model_and_tokenizer(model_type)

    datasets, task_num_classes, _ = return_multitask_dataset(
        tokenizer, data_source=data_source, threshold=threshold, seed=seed,
        include_clndn=include_clndn, include_clnsig=include_clnsig, include_maves=include_maves,
    )

    tasks = list(task_num_classes.keys())
    print(f"Multi-task training with: {tasks}")
    print(f"Decoder mode: {decoder}")

    # Pick first available validation set
    eval_ds = next((datasets.get(f"{t}_val") for t in tasks if datasets.get(f"{t}_val")), None)

    trainer = MultitaskTrainer(
        task_num_classes=task_num_classes,
        decoder=decoder,
        model=model,
        args=TrainingArguments(
            output_dir=f"{path_prefix}/smart_discriminative_multitask_{model_type}_{'_'.join(tasks)}",
            learning_rate=learning_rate, max_grad_norm=max_grad_norm,
            per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs, save_total_limit=5,
            eval_strategy="epoch", save_strategy="epoch",
            metric_for_best_model="matthews_correlation", greater_is_better=True,
            load_best_model_at_end=True, save_safetensors=False,  # Weight tying issue
            remove_unused_columns=False, dataloader_num_workers=num_workers,
        ),
        train_dataset=datasets['train'],
        eval_dataset=eval_ds,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        data_collator=MultiTaskDataCollator(tokenizer),
    )

    trainer.train()

    for task in tasks:
        test_ds = datasets.get(f"{task}_test") or datasets.get(f"MAVES_score_test")
        if test_ds:
            print(f"Test Metrics for {task}: {trainer.evaluate(eval_dataset=test_ds)}")


def run_allheads_multitask_finetune(
    seed,
    model_type='nt',
    decoder=False,
    learning_rate=0.000005,
    batch_size=8,
    num_epochs=10,
    max_grad_norm=1.0,
    num_workers=8,
    threshold=65.0,
    csv_path='root/data/unfiltered_variants.csv',
    data_source='smart',
):
    """
    AlphaGenome-style multitask training: ALL heads evaluated for EVERY sample.

    Uses paired CLNDN + CLNSIG labels from SMART data.
    Both classification heads get gradients from every variant.
    """
    set_seed(seed)

    path_prefix = "./root/models"
    model, tokenizer = get_model_and_tokenizer(model_type)

    # Load multi-label dataset (paired CLNDN + CLNSIG)
    datasets, task_num_classes, task_configs, seq_length = return_multilabel_dataset(
        tokenizer,
        csv_path=csv_path,
        threshold=threshold,
        seed=seed,
    )

    task_names = list(task_num_classes.keys())
    print(f"\n{'='*60}")
    print(f"AlphaGenome-style Multi-Task Training")
    print(f"{'='*60}")
    print(f"Tasks: {task_names}")
    print(f"Task classes: {task_num_classes}")
    print(f"Model: {model_type}")
    print(f"Decoder mode: {decoder}")
    print(f"{'='*60}\n")

    # Create trainer (custom evaluate handles per-task metrics)
    trainer = AllHeadsMultitaskTrainer(
        task_num_classes=task_num_classes,
        decoder=decoder,
        label_smoothing=0.1,
        model=model,
        args=TrainingArguments(
            output_dir=f"{path_prefix}/smart_allheads_multitask_{model_type}_CLNDN_CLNSIG",
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
            save_safetensors=False,  # Weight tying issue with OmniDNA
            remove_unused_columns=False,
            dataloader_num_workers=num_workers,
        ),
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'],
        data_collator=MultiLabelDataCollator(tokenizer, task_names),
    )

    # Train
    trainer.train()

    # Evaluate on test set
    print(f"\n{'='*60}")
    print("Test Set Evaluation")
    print(f"{'='*60}")
    test_metrics = trainer.evaluate(eval_dataset=datasets['test'])
    print(f"Test Metrics: {test_metrics}")

    return trainer, test_metrics


def run_generative_multitask_finetune(
    seed: int,
    model_type: str = 'omni_dna_116m',
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    num_epochs: int = 10,
    max_seq_length: int = 512,
    threshold: float = 65.0,
    include_clndn: bool = True,
    include_clnsig: bool = True,
    data_source: str = 'smart',
    neftune_noise_alpha: float = 5.0,
):
    """
    Generative multitask fine-tuning using SFTTrainer for variant classification.

    Formats variant classification as a text generation task:
    - Input: "Classify variant: [REF_SEQ][SEP][ALT_SEQ][MASK]"
    - Output: Label text (e.g., "Pathogenic", "Benign", "Cardiomyopathy")

    Only works with decoder-only models (OmniDNA).

    Args:
        seed: Random seed
        model_type: Must be 'omni_dna_116m' or 'omni_dna_1b'
        include_clndn: Include disease classification task
        include_clnsig: Include pathogenicity classification task
        neftune_noise_alpha: NEFTune noise for regularization (0 to disable)
    """
    set_seed(seed)

    # Validate model type
    supported_models = ['omni_dna_116m', 'omni_dna_1b']
    if model_type not in supported_models:
        raise ValueError(f"Generative training only supports {supported_models}, got: {model_type}")

    path_prefix = "./root/models"
    tasks_str = '_'.join([t for t, inc in [('CLNDN', include_clndn), ('CLNSIG', include_clnsig)] if inc])
    output_path = f"{path_prefix}/smart_generative_multitask_{model_type}_{tasks_str}"

    # Load model and tokenizer
    local_model_base = f"./root/models/{model_type}"
    if model_type == 'omni_dna_116m':
        model_path = local_model_base if os.path.exists(local_model_base) else "zehui127/Omni-DNA-116M"
    else:
        model_path = local_model_base if os.path.exists(local_model_base) else "zehui127/Omni-DNA-1B"

    use_local = os.path.exists(model_path)

    print(f"\n{'='*60}")
    print(f"Generative Multitask Training (SFTTrainer)")
    print(f"{'='*60}")
    print(f"Model: {model_type}")
    print(f"Model path: {model_path}")
    print(f"Tasks: CLNDN={include_clndn}, CLNSIG={include_clnsig}")
    print(f"NEFTune alpha: {neftune_noise_alpha}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=use_local)

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=use_local)

    # Define label mappings
    disease_labels = ['Aortopathy', 'Arrhythmia', 'Cardiomyopathy', 'Structural_defect']
    pathogenicity_labels = ['Benign', 'Pathogenic']

    # Load raw data
    from ..dataloader.data_wrapper import SmartVariantDataWrapper
    wrapper = SmartVariantDataWrapper(csv_path='root/data/unfiltered_variants.csv', all_records=True)
    raw_data = wrapper.get_multitask_data(Seq_length=max_seq_length, threshold=threshold)

    # Format data for generative training
    formatted_data = []
    max_seq = max_seq_length // 2 - 50  # Leave room for instruction + label

    for ref, alt, disease, pathogenicity in raw_data:
        ref_seq = ref[:max_seq] if len(ref) > max_seq else ref
        alt_seq = alt[:max_seq] if len(alt) > max_seq else alt

        if include_clndn and disease in disease_labels:
            formatted_data.append({
                'instruction': f"{ref_seq}[SEP]{alt_seq}",
                'output': disease.replace(' ', '_'),
                'task': 'CLNDN'
            })

        if include_clnsig and pathogenicity in [0, 1]:
            label = pathogenicity_labels[pathogenicity]
            formatted_data.append({
                'instruction': f"{ref_seq}[SEP]{alt_seq}",
                'output': label,
                'task': 'CLNSIG'
            })

    # Task distribution
    task_counts = {}
    for item in formatted_data:
        task_counts[item['task']] = task_counts.get(item['task'], 0) + 1
    print(f"Total samples: {len(formatted_data)}")
    print(f"Task distribution: {task_counts}")

    # Create dataset and split
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.shuffle(seed=seed)
    split = dataset.train_test_split(test_size=0.1, seed=seed)
    train_dataset = split['train']
    eval_dataset = split['test']

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Formatting function for SFTTrainer (Omni-DNA style)
    def formatting_prompts_func(example):
        return f"{example['instruction']}[MASK]{example['output']}"

    # Completion-only loss (only compute loss on output after [MASK])
    response_template = "[MASK]"
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # Training config
    # Note: save_safetensors=False to avoid DTensor bug in some transformers versions
    training_args = SFTConfig(
        output_dir=output_path,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=4,  # Reduce memory usage
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        max_seq_length=max_seq_length,
        save_total_limit=3,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        save_safetensors=False,
        neftune_noise_alpha=neftune_noise_alpha if neftune_noise_alpha > 0 else None,
        logging_steps=50,
        report_to="none",
        ddp_find_unused_parameters=False,  # Better DDP performance
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting generative training...")
    trainer.train()

    # Save
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"\nModel saved to: {output_path}")

    return trainer


def main():
    parser = argparse.ArgumentParser(description="SMART model fine-tuning (single or multi-task)")
    parser.add_argument("--model", type=str, default='nt', help="Model type")
    parser.add_argument("--seed", type=int, default=127, help="Random seed")
    parser.add_argument("--decoder", action="store_true", help="Decoder architecture")
    parser.add_argument("--test_only", action="store_true", help="Only evaluate")

    # Task selection
    parser.add_argument("--task", type=str, default="CLNDN", choices=["CLNDN", "CLNSIG"],
                        help="Single task mode")
    parser.add_argument("--multitask", action="store_true", help="Enable multi-task mode (discriminative)")
    parser.add_argument("--allheads", action="store_true", help="AlphaGenome-style: all heads per sample")
    parser.add_argument("--generative", action="store_true",
                        help="Generative multitask training (SFTTrainer, OmniDNA only)")
    parser.add_argument("--data_source", type=str, default='smart', choices=['smart', 'clinvar'],
                        help="Data source for classification tasks")
    parser.add_argument("--clndn", action="store_true", help="Include CLNDN (disease) in multi-task")
    parser.add_argument("--clnsig", action="store_true", help="Include CLNSIG (pathogenicity) in multi-task")
    parser.add_argument("--maves", action="store_true", help="Include MAVES (fitness) in multi-task")
    parser.add_argument("--neftune_alpha", type=float, default=5.0,
                        help="NEFTune noise alpha for generative training (0 to disable)")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.000005)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=65.0)
    parser.add_argument("--min_samples_per_class", type=int, default=2)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_step", type=int, default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.generative:
        # Generative multitask training (SFTTrainer, OmniDNA only)
        clndn = args.clndn or (not args.clndn and not args.clnsig)
        clnsig = args.clnsig or (not args.clndn and not args.clnsig)
        run_generative_multitask_finetune(
            seed=args.seed,
            model_type=args.model,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            threshold=args.threshold,
            include_clndn=clndn,
            include_clnsig=clnsig,
            data_source=args.data_source,
            neftune_noise_alpha=args.neftune_alpha,
        )
    elif args.allheads:
        # AlphaGenome-style: all heads evaluated for every sample
        run_allheads_multitask_finetune(
            seed=args.seed,
            model_type=args.model,
            decoder=args.decoder,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            max_grad_norm=args.max_grad_norm,
            num_workers=args.num_workers,
            threshold=args.threshold,
            data_source=args.data_source
        )
    elif args.multitask:
        # Task-routing: each sample goes to one head
        clndn = args.clndn or (not args.clndn and not args.clnsig and not args.maves)
        clnsig = args.clnsig or (not args.clndn and not args.clnsig and not args.maves)
        maves = args.maves
        run_multitask_finetune(
            seed=args.seed,
            model_type=args.model,
            decoder=args.decoder,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            max_grad_norm=args.max_grad_norm,
            num_workers=args.num_workers,
            threshold=args.threshold,
            include_clndn=clndn,
            include_clnsig=clnsig,
            include_maves=maves,
            data_source=args.data_source
        )
    else:
        run_single_task_finetune(
            args.task, args.seed, args.model, args.decoder, args.test_only,
            learning_rate=args.learning_rate, batch_size=args.batch_size,
            num_epochs=args.num_epochs, max_grad_norm=args.max_grad_norm,
            num_workers=args.num_workers, threshold=args.threshold,
            checkpoint_path=args.checkpoint_path, checkpoint_step=args.checkpoint_step,
            min_samples_per_class=args.min_samples_per_class
        )

if __name__ == "__main__":
    main()
