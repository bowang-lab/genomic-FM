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
    TrainingArguments,
    AutoTokenizer,
    Trainer,
    set_seed
)
from accelerate import Accelerator
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
# from .seq_pack import FramePackCausalLM

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
                # Sort by checkpoint number and select the latest one
                checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
                latest_checkpoint = checkpoint_dirs[-1]
                checkpoint_weights_path = os.path.join(checkpoint_path, latest_checkpoint, "pytorch_model.bin")
                print(f"Will load ClinVar-trained weights from {checkpoint_weights_path} (latest from {len(checkpoint_dirs)} checkpoints)")
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
        elif model_type == 'seq_pack':
            # model_path = "zehui127/Omni-DNA-116M"
            model_path = 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'
            # tokenizer_path = "zehui127/Omni-DNA-116M"
            tokenizer_path = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
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
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    # Load base model (trainers/wrappers add their own heads)
    from transformers import AutoModelForCausalLM
    if model_type in ['gpn-star', 'nt']:
        base_model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    elif model_type == 'omni_dna_116m':
        base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
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
        save_safetensors=True,
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
            output_dir=f"{path_prefix}/smart_multitask_{model_type}_{'_'.join(tasks)}",
            learning_rate=learning_rate, max_grad_norm=max_grad_norm,
            per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs, save_total_limit=5,
            eval_strategy="epoch", save_strategy="epoch",
            metric_for_best_model="matthews_correlation", greater_is_better=True,
            load_best_model_at_end=True, save_safetensors=True,
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
            output_dir=f"{path_prefix}/allheads_multitask_{model_type}_CLNDN_CLNSIG",
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
            save_safetensors=True,
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


def main():
    parser = argparse.ArgumentParser(description="SMART model fine-tuning (single or multi-task)")
    parser.add_argument("--model", type=str, default='nt', help="Model type")
    parser.add_argument("--seed", type=int, default=127, help="Random seed")
    parser.add_argument("--decoder", action="store_true", help="Decoder architecture")
    parser.add_argument("--test_only", action="store_true", help="Only evaluate")

    # Task selection
    parser.add_argument("--task", type=str, default="CLNDN", choices=["CLNDN", "CLNSIG"],
                        help="Single task mode")
    parser.add_argument("--multitask", action="store_true", help="Enable multi-task mode")
    parser.add_argument("--allheads", action="store_true", help="AlphaGenome-style: all heads per sample")
    parser.add_argument("--data_source", type=str, default='smart', choices=['smart', 'clinvar'],
                        help="Data source for classification tasks")
    parser.add_argument("--clndn", action="store_true", help="Include CLNDN (disease) in multi-task")
    parser.add_argument("--clnsig", action="store_true", help="Include CLNSIG (pathogenicity) in multi-task")
    parser.add_argument("--maves", action="store_true", help="Include MAVES (fitness) in multi-task")

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

    if args.allheads:
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
