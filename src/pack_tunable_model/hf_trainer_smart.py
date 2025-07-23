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
    AutoModelForSequenceClassification,
    TrainingArguments,
    AutoTokenizer,
    Trainer,
    set_seed
)
from accelerate import Accelerator
# Import from local modules
from .hf_dataloader import return_clinvar_multitask_dataset, MultiTaskDataCollator, return_smart_dataset
# from .hf_dataloader import return_smart_dataset, MultiTaskDataCollator
# Import our custom wrapper model
from .wrap_model import WrappedModelWithClassificationHead
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

def run_single_task_finetune(task, seed, model_type='nt', decoder=False, test_only=False):
    set_seed(seed)
    accelerator = Accelerator()
    # Configuration
    path_prefix = "/mnt/data/genomic_fm/all_clinvar" # for amlt
    # path_prefix = "/home/v-zehuili/repositories/amlt/codes/genomic-FM/root/clinvar_disease_classification"
    results_file = f"{path_prefix}/test_results_clinvar.csv"

    # Model and Tokenizer Selection
    model_path = None
    if model_type == 'olmo':
        model_path = "zehui127/Omni-DNA-116M"
        tokenizer_path = "zehui127/Omni-DNA-116M"
    elif model_type == 'nt':
        model_path = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
        tokenizer_path = model_path
        # if test_only:
            # model_path = "/home/v-zehuili/repositories/genomic-FM/root/clinvar_disease_classification/checkpoint-55213"
    elif model_type == 'seq_pack':
        # model_path = "zehui127/Omni-DNA-116M"
        model_path = 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'
        # tokenizer_path = "zehui127/Omni-DNA-116M"
        tokenizer_path = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    elif model_type == 'dnabert2':
        model_path = "zhihan1996/DNABERT-2-117M"
        tokenizer_path = model_path
    elif model_type=='hyenaDNA':
        model_path = "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf"
        tokenizer_path = model_path
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load base model (not using AutoModelForSequenceClassification anymore)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True
    )

    ########### Load Dataset old ##################
        # dangerous zone: so we need to use main_process_first
        # Code in this block is executed by rank-0 first,
        # all other ranks are blocked until rank-0 exits the block.
    # datasets, task_num_classes, max_seq_len = return_clinvar_multitask_dataset(
    #     tokenizer, task, seed=seed
    # )
    datasets, task_num_classes, max_seq_len = return_smart_dataset(
        tokenizer, '/mnt/data/genomic_fm/all_clinvar/unfiltered_variants.csv'
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

    # Load saved model if testing only
    # if test_only and os.path.exists(f"{model_path}"):
    state_dict = f"/mnt/data/genomic_fm/all_clinvar/pretrain_model_dnabert2_CLNSIG/checkpoint-58330/pytorch_model.bin"
    print(f"Loading weights from {state_dict}")
    head_state_dict = torch.load(f"{state_dict}")
    model.load_state_dict(head_state_dict)
    # Prepare Training Arguments
    training_args = TrainingArguments(
        output_dir=f"{path_prefix}/smart_pretrain_model_{model_type}_{task}_with_state_dict",
        learning_rate=0.000005,
        max_grad_norm=1.0,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        save_total_limit=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="matthews_correlation",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_safetensors=False,
        remove_unused_columns=False,
        dataloader_num_workers=8,
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

def main():
    parser = argparse.ArgumentParser(description="Single-task fine-tune and evaluate model.")
    parser.add_argument("--model", type=str, default='nt',
                        help="Model type (e.g., olmo, nt, dnabert2)")
    parser.add_argument("--seed", type=int, default=127,
                        help="Random seed value for training")
    parser.add_argument("--decoder", action="store_true",
                        help="Whether the model has a decoder architecture")
    parser.add_argument("--test_only", action="store_true",
                        help="Only run evaluation on the test set")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run with single task (CLNDN)
    # run_single_task_finetune('CLNDN', args.seed, args.model, args.decoder, args.test_only)
    # pathegenoic vs. benign
    run_single_task_finetune('CLNDN', args.seed, args.model, args.decoder, args.test_only)

if __name__ == "__main__":
    main()
