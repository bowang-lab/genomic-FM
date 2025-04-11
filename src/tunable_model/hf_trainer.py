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
    set_seed
)
current_dir = os.path.dirname(os.path.abspath(__file__))
test_only = True
# Navigate to the parent of the parent directory
# olmo_repo_path = os.path.abspath(os.path.join(current_dir, "..", "..", "OLmo-GFM"))
# sys.path.append(olmo_repo_path)
# from hf_olmo import *  # Ensure necessary imports from OLMo repo

# Import from local modules
from .hf_dataloader import return_clinvar_multitask_dataset, MultiTaskDataCollator

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

def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], labels: Optional[torch.Tensor] = None):
    # Handle the case where logits is a tuple from the custom loss computation
    if isinstance(logits, tuple):
        logits = logits[0]
    # Ensure logits is a torch.Tensor
    if not isinstance(logits, torch.Tensor):
        raise ValueError("Logits must be a torch.Tensor")
    # Print shape for debugging
    # Get the argmax predictions
    predictions = torch.argmax(logits, dim=-1)
    return predictions

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)

class MultitaskTrainer(transformers.Trainer):
    def __init__(self, task_num_classes, *args, **kwargs):
        """
        Initialize MultitaskTrainer with task-specific number of classes and datasets.

        Args:
            task_num_classes (dict): Dictionary mapping task names to number of classes
            multitask_datasets (dict): Dictionary of datasets for different tasks
        """
        self.task_num_classes = task_num_classes
        super().__init__(*args, **kwargs)

    def _save_state_dict(self):
        state_dict = super()._save_state_dict()
        if hasattr(self.model, 'task_classification_heads'):
            state_dict['task_classification_heads'] = self.model.task_classification_heads.state_dict()
        return state_dict

    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        """
        Custom loss computation that considers task-specific classification heads
        """
        task_names = inputs.pop("task_names", None)
        labels = inputs.get("labels")

        # Dynamic task-specific classification head
        if task_names is not None and len(task_names) > 0:
            unique_tasks = list(set(task_names))

            # Create task-specific classification heads if not exists
            if not hasattr(model, 'task_classification_heads'):
                model.task_classification_heads = torch.nn.ModuleDict()

            for task in unique_tasks:
                if task not in model.task_classification_heads:
                    num_classes = self.task_num_classes.get(task, 2)  # Default to binary if not specified
                    model.task_classification_heads[task] = torch.nn.Linear(
                        model.config.hidden_size,
                        num_classes
                    ).to(model.device)

            # Route inputs to task-specific heads
            outputs_ref = model(
                input_ids=inputs['ref_input_ids'],
                attention_mask=inputs['ref_attention_mask'],
                output_hidden_states=True
            )

            outputs_alt = model(
                input_ids=inputs['alt_input_ids'],
                attention_mask=inputs['alt_attention_mask'],
                output_hidden_states=True
            )

            last_hidden_state_ref = outputs_ref.hidden_states[-1][:,0,:]
            last_hidden_state_alt = outputs_alt.hidden_states[-1][:,0,:]
            last_hidden_state = last_hidden_state_alt - last_hidden_state_ref
            logits = torch.zeros(
                (len(task_names), self.task_num_classes[task_names[0]]),
                device=last_hidden_state.device
            )

            for i, (task, hidden_state) in enumerate(zip(task_names, last_hidden_state)):
                logits[i] = model.task_classification_heads[task](hidden_state)

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

            return (loss, (logits, outputs)) if return_outputs else loss

        # Default to standard trainer loss computation
        return super().compute_loss(model, inputs, return_outputs)
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", task_name=None):
        """
        Extend evaluate method to support multitask evaluation

        Args:
            eval_dataset: Dataset to evaluate
            ignore_keys: Keys to ignore in evaluation
            metric_key_prefix: Prefix for metrics
            task_name: Optional task name for specific task evaluation
        """
        # If no specific dataset is provided, use the first task's test dataset
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        # If a specific task name is provided, filter the dataset
        if task_name is not None:
            eval_dataset = eval_dataset.filter(
                lambda example: example['task_name'] == task_name
            )

        # Call parent's evaluate method
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

        return metrics

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prediction step that uses existing task-specific classification heads
        """
        with torch.no_grad():
            # Extract task names from inputs
            task_names = inputs.pop("task_names", None)

            # Perform forward pass to get hidden states
            outputs_ref = model(
                input_ids=inputs['ref_input_ids'],
                attention_mask=inputs['ref_attention_mask'],
                output_hidden_states=True
            )

            outputs_alt = model(
                input_ids=inputs['alt_input_ids'],
                attention_mask=inputs['alt_attention_mask'],
                output_hidden_states=True
            )
            labels = inputs.get("labels")

            # # If no task names provided, use default trainer prediction step
            # if task_names is None or len(task_names) == 0:
            #     return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

            # Ensure task-specific classification heads exist
            if not hasattr(model, 'task_classification_heads'):
                raise ValueError("Task-specific classification heads not found. They should be created during training.")

            # Get last hidden state
            last_hidden_state_ref = outputs_ref.hidden_states[-1][:,0,:]
            last_hidden_state_alt = outputs_alt.hidden_states[-1][:,0,:]
            last_hidden_state = last_hidden_state_alt - last_hidden_state_ref

            # Prepare logits using task-specific heads
            logits = torch.zeros(
                (len(task_names), self.task_num_classes[task_names[0]]),
                device=last_hidden_state.device
            )

            # Apply task-specific heads
            for i, (task, hidden_state) in enumerate(zip(task_names, last_hidden_state)):
                logits[i] = model.task_classification_heads[task](hidden_state)

            # Compute loss if needed
            loss = None
            if not prediction_loss_only:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)

            # Restore task_names to inputs for potential future use
            if task_names is not None:
                inputs["task_names"] = task_names

            return loss, logits, labels

def run_multitask_finetune(tasks, seed, model_type='nt'):
    set_seed(seed)

    # Configuration
    path_prefix = "/home/v-zehuili/repositories/genomic-FM/root"
    cache_dir = f"{path_prefix}/cache_directory"
    results_file = f"{path_prefix}/test_results_multitask.csv"

    # Model and Tokenizer Selection
    if model_type == 'olmo':
        model = AutoModelForSequenceClassification.from_pretrained(
            "/home/v-zehuili/finetune_nt_150M/step832510-unsharded",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "zehui127/Omni-DNA-116M",
            trust_remote_code=True
        )
    elif model_type == 'nt':
        model = AutoModelForSequenceClassification.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
            trust_remote_code=True
        )
        if test_only:
            model = AutoModelForSequenceClassification.from_pretrained(
               "/home/v-zehuili/repositories/genomic-FM/root/clinvar_disease_classification/checkpoint-55213",
                trust_remote_code=True
            )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    tokenizer.model_max_length = 1000
    # Load Multitask Datasets
    multitask_datasets, task_num_classes, max_seq_len = return_clinvar_multitask_dataset(
        tokenizer,tasks[0], seed=seed
    )
    if test_only:
        state_dict = torch.load("/home/v-zehuili/repositories/genomic-FM/root/clinvar_disease_classification/checkpoint-55213/pytorch_model.bin")
        # Check for task-specific heads in state_dict
        task_head_keys = [k for k in state_dict.keys() if k.startswith('task_classification_heads.')]
        if task_head_keys:
            if not hasattr(model, 'task_classification_heads'):
                model.task_classification_heads = torch.nn.ModuleDict()
            # Create task-specific classification heads for each task
            for task in tasks:
                if task not in model.task_classification_heads:
                    num_classes = task_num_classes.get(task, 2)
                    model.task_classification_heads[task] = torch.nn.Linear(
                        model.config.hidden_size,
                        num_classes
                    ).to(model.device)
            # Filter and load only task-specific head parameters
            task_head_state = {k.replace('task_classification_heads.', ''): v for k, v in state_dict.items() if k.startswith('task_classification_heads.')}
            model.task_classification_heads.load_state_dict(task_head_state)

    # Prepare Training Arguments
    training_args = TrainingArguments(
        output_dir=f"{path_prefix}/clinvar_disease_classification",
        learning_rate=0.000005,
        max_grad_norm=1.0,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        evaluation_strategy="no",
        save_strategy="epoch",
        metric_for_best_model="matthews_correlation",
        greater_is_better=True,
        save_total_limit=2,
        load_best_model_at_end=False,
        save_safetensors=False,
        remove_unused_columns=False,
    )

    # Data Collator
    data_collator = MultiTaskDataCollator(tokenizer)

    # Multitask Trainer
    trainer = MultitaskTrainer(
        task_num_classes=task_num_classes,
        model=model,
        args=training_args,
        train_dataset=multitask_datasets['train'],  # Use the combined train dataset
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    # Training
    if not test_only:
        trainer.train()

    # Evaluation and Logging
    test_metrics = {}
    for task in tasks:
        test_dataset = multitask_datasets.get(f"{task}_test")
        if test_dataset:
            task_metrics = trainer.evaluate(eval_dataset=test_dataset)
            test_metrics[task] = task_metrics
            print(f"Test Metrics for {task}: {task_metrics}")

    # Log results
    write_header = not os.path.exists(results_file)
    with open(results_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = ["Seed", "Model Type", "Tasks"] + [f"{task}_metric" for task in tasks]
            writer.writerow(header)

        row = [seed, model_type, ",".join(tasks)]
        for task in tasks:
            row.append(test_metrics.get(task, {}).get("eval_matthews_correlation", "N/A"))

        writer.writerow(row)

    print(f"Multitask test metrics appended to {results_file}")

def main():
    parser = argparse.ArgumentParser(description="Multi-task Fine-tune and evaluate model.")
    parser.add_argument("--model", type=str, default='nt',
                        help="Model type (e.g., olmo, nt)")
    parser.add_argument("--seed", type=int, default=127,
                        help="Random seed value for training")
    # tasks = [
    #     'H3', 'H4', 'H3K9ac', 'H3K14ac', 'H4ac',
    #     'H3K4me1', 'H3K4me2', 'H3K4me3',
    #     'H3K36me3', 'H3K79me3'
    # ]
    tasks = [
        'CLNDN'
    ]
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    run_multitask_finetune(tasks, args.seed, args.model)

if __name__ == "__main__":
    main()
