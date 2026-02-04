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
test_only = False
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

class AllHeadsMultitaskTrainer(transformers.Trainer):
    """AlphaGenome-style trainer: all heads evaluated for every sample."""

    IGNORE_INDEX = -100

    def __init__(
        self,
        task_num_classes: Dict[str, int],
        decoder: bool = False,
        label_smoothing: float = 0.1,
        *args,
        **kwargs
    ):
        """
        Args:
            task_num_classes: Dict mapping task names to number of classes
                              e.g., {'CLNDN': 4, 'CLNSIG': 2}
            decoder: Whether to use decoder-style (last token) pooling
            label_smoothing: Label smoothing factor for classification
        """
        self.task_num_classes = task_num_classes
        self.task_names = list(task_num_classes.keys())
        self.decoder = decoder
        self.label_smoothing = label_smoothing
        super().__init__(*args, **kwargs)

        # Initialize task heads after model is set
        self._initialize_task_heads()

    def _initialize_task_heads(self):
        """Initialize all task-specific classification heads upfront."""
        model = self.model

        if not hasattr(model, 'task_heads'):
            model.task_heads = torch.nn.ModuleDict()

        hidden_size = model.config.hidden_size

        for task, num_classes in self.task_num_classes.items():
            if task not in model.task_heads:
                # MLP head (similar to wrap_model.py)
                model.task_heads[task] = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, 128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(128, num_classes)
                ).to(model.device)
                print(f"Initialized head for {task}: {num_classes} classes")

    def _get_shared_embeddings(self, model, inputs):
        """Get shared embeddings from ref/alt sequences."""
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

        if not self.decoder:
            # Encoder models: use [CLS] token (first token)
            hidden_ref = outputs_ref.hidden_states[-1][:, 0, :]
            hidden_alt = outputs_alt.hidden_states[-1][:, 0, :]
        else:
            # Decoder models: use last token
            ref_attention_mask = inputs['ref_attention_mask']
            alt_attention_mask = inputs['alt_attention_mask']

            hidden_ref_full = outputs_ref.hidden_states[-1]
            hidden_alt_full = outputs_alt.hidden_states[-1]

            ref_seq_lens = ref_attention_mask.sum(dim=-1)
            alt_seq_lens = alt_attention_mask.sum(dim=-1)
            batch_index = torch.arange(ref_seq_lens.size(0), device=hidden_ref_full.device)

            hidden_ref = hidden_ref_full[batch_index, ref_seq_lens - 1, :]
            hidden_alt = hidden_alt_full[batch_index, alt_seq_lens - 1, :]

        # Compute difference (variant effect representation)
        embeddings = hidden_alt - hidden_ref

        return embeddings, outputs_ref, outputs_alt

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss by evaluating ALL heads and summing losses.

        AlphaGenome-style: L_total = Σ L_h (sum over all heads)
        """
        # Ensure task heads exist
        if not hasattr(model, 'task_heads') or len(model.task_heads) == 0:
            self._initialize_task_heads()

        # Get shared embeddings
        embeddings, outputs_ref, outputs_alt = self._get_shared_embeddings(model, inputs)

        # Evaluate ALL heads and compute losses
        head_losses = []
        all_logits = {}

        for task in self.task_names:
            # Get labels for this task
            labels = inputs.get(f"label_{task}")
            if labels is None:
                continue

            # Forward through task head
            logits = model.task_heads[task](embeddings)
            all_logits[task] = logits

            # Create mask for valid labels (not IGNORE_INDEX)
            valid_mask = labels != self.IGNORE_INDEX

            if valid_mask.sum() > 0:
                # Compute loss only on valid samples
                valid_logits = logits[valid_mask]
                valid_labels = labels[valid_mask]

                # Classification loss with label smoothing
                task_loss = torch.nn.functional.cross_entropy(
                    valid_logits,
                    valid_labels,
                    label_smoothing=self.label_smoothing
                )
                head_losses.append(task_loss)

        # Sum losses across heads (AlphaGenome-style)
        if head_losses:
            total_loss = torch.stack(head_losses).sum()
        else:
            total_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        if return_outputs:
            # Return logits for primary task (first task) for metrics
            primary_task = self.task_names[0]
            primary_logits = all_logits.get(primary_task, embeddings)
            return (total_loss, (primary_logits, outputs_ref))

        return total_loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prediction step with all-heads evaluation."""

        with torch.no_grad():
            # Ensure task heads exist
            if not hasattr(model, 'task_heads') or len(model.task_heads) == 0:
                self._initialize_task_heads()

            # Get shared embeddings
            embeddings, _, _ = self._get_shared_embeddings(model, inputs)

            # Compute all head outputs and losses
            head_losses = []
            all_logits = {}

            for task in self.task_names:
                labels = inputs.get(f"label_{task}")
                if labels is None:
                    continue

                logits = model.task_heads[task](embeddings)
                all_logits[task] = logits

                valid_mask = labels != self.IGNORE_INDEX
                if valid_mask.sum() > 0:
                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask]
                    task_loss = torch.nn.functional.cross_entropy(
                        valid_logits, valid_labels, label_smoothing=self.label_smoothing
                    )
                    head_losses.append(task_loss)

            loss = torch.stack(head_losses).sum() if head_losses else None

            # Return logits and labels for primary task
            primary_task = self.task_names[0]
            primary_logits = all_logits.get(primary_task)
            primary_labels = inputs.get(f"label_{primary_task}")

            return loss, primary_logits, primary_labels

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Evaluate all tasks and return per-task metrics."""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        model = self.model
        model.eval()
        dataloader = self.get_eval_dataloader(eval_dataset)

        # Collect predictions per task
        task_preds = {task: [] for task in self.task_names}
        task_labels = {task: [] for task in self.task_names}
        total_loss = 0.0
        num_batches = 0

        for inputs in dataloader:
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                embeddings, _, _ = self._get_shared_embeddings(model, inputs)
                for task in self.task_names:
                    labels = inputs.get(f"label_{task}")
                    if labels is None:
                        continue
                    logits = model.task_heads[task](embeddings)
                    preds = torch.argmax(logits, dim=-1)

                    valid_mask = labels != self.IGNORE_INDEX
                    task_preds[task].extend(preds[valid_mask].cpu().numpy())
                    task_labels[task].extend(labels[valid_mask].cpu().numpy())

        # Compute per-task metrics
        metrics = {}
        for task in self.task_names:
            if len(task_preds[task]) > 0:
                preds = np.array(task_preds[task])
                labels = np.array(task_labels[task])
                metrics[f"{metric_key_prefix}_{task}_accuracy"] = sklearn.metrics.accuracy_score(labels, preds)
                metrics[f"{metric_key_prefix}_{task}_f1"] = sklearn.metrics.f1_score(labels, preds, average="macro", zero_division=0)
                metrics[f"{metric_key_prefix}_{task}_mcc"] = sklearn.metrics.matthews_corrcoef(labels, preds)

        # Average MCC across tasks for model selection
        mccs = [v for k, v in metrics.items() if k.endswith("_mcc")]
        metrics[f"{metric_key_prefix}_matthews_correlation"] = np.mean(mccs) if mccs else 0.0

        return metrics


class MultitaskTrainer(transformers.Trainer):
    """Task-routing trainer: each sample goes to one head based on task_name."""

    def __init__(self, task_num_classes, decoder=False, *args, **kwargs):
        """
        Initialize MultitaskTrainer with task-specific number of classes and datasets.

        Args:
            task_num_classes (dict): Dictionary mapping task names to number of classes
            decoder (bool): Whether to use decoder-style (last token) or encoder-style ([CLS] token) pooling
        """
        self.task_num_classes = task_num_classes
        self.decoder = decoder
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
            logits = None
            if not self.decoder:
                last_hidden_state_ref = outputs_ref.hidden_states[-1][:,0,:]
                last_hidden_state_alt = outputs_alt.hidden_states[-1][:,0,:]
                last_hidden_state = last_hidden_state_alt - last_hidden_state_ref
                # Use list to handle different output sizes per task
                logits_list = []
                for i, (task, hidden_state) in enumerate(zip(task_names, last_hidden_state)):
                    logits_list.append(model.task_classification_heads[task](hidden_state))
                logits = torch.stack(logits_list, dim=0)
            else:
                # Decoder-only models (HyenaDNA, OmniDNA): use last token
                ref_attention_mask = inputs['ref_attention_mask']
                alt_attention_mask = inputs['alt_attention_mask']

                # shape: (batch_size, seq_len, hidden_dim)
                # Use hidden_states for decoder-only models (HyenaDNA, OmniDNA)
                hidden_ref = outputs_ref.hidden_states[-1]
                hidden_alt = outputs_alt.hidden_states[-1]

                # compute true sequence lengths so we know where the "last" token is
                ref_seq_lens = ref_attention_mask.sum(dim=-1)  # (batch_size,)
                alt_seq_lens = alt_attention_mask.sum(dim=-1)  # (batch_size,)

                # build an index for batch dimension
                batch_index = torch.arange(ref_seq_lens.size(0), device=hidden_ref.device)

                # select the last-token vector for each example
                last_ref = hidden_ref[batch_index, ref_seq_lens - 1, :]  # (batch_size, hidden_dim)
                last_alt = hidden_alt[batch_index, alt_seq_lens - 1, :]  # (batch_size, hidden_dim)

                # take the difference
                last_hidden_state = last_alt - last_ref   # (batch_size, hidden_dim)
                # run each sample's difference vector through its task head
                logits_list = []
                for i, task in enumerate(task_names):
                    logits_list.append(model.task_classification_heads[task](last_hidden_state[i]))
                logits = torch.stack(logits_list, dim=0)

            # Compute per-sample loss based on task type (handles mixed batches)
            losses = []
            for i, task in enumerate(task_names):
                sample_logits = logits[i:i+1]
                sample_label = labels[i:i+1]

                if "MAVES" in task:
                    # Regression task - use Huber loss (robust to outliers)
                    sample_loss = torch.nn.functional.huber_loss(
                        sample_logits.squeeze(-1),
                        sample_label.float(),
                        delta=1.0
                    )
                else:
                    # Classification task - use CrossEntropy with label smoothing
                    sample_loss = torch.nn.functional.cross_entropy(
                        sample_logits,
                        sample_label,
                        label_smoothing=0.1
                    )
                losses.append(sample_loss)

            loss = torch.stack(losses).mean()

            return (loss, (logits, outputs_ref)) if return_outputs else loss

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

            if not self.decoder:
                # Encoder models: use [CLS] token (first token)
                last_hidden_state_ref = outputs_ref.hidden_states[-1][:,0,:]
                last_hidden_state_alt = outputs_alt.hidden_states[-1][:,0,:]
                last_hidden_state = last_hidden_state_alt - last_hidden_state_ref

                # Apply task-specific heads
                logits_list = []
                for i, (task, hidden_state) in enumerate(zip(task_names, last_hidden_state)):
                    logits_list.append(model.task_classification_heads[task](hidden_state))
                logits = torch.stack(logits_list, dim=0)
            else:
                # Decoder-only models: use last token
                ref_attention_mask = inputs['ref_attention_mask']
                alt_attention_mask = inputs['alt_attention_mask']

                # shape: (batch_size, seq_len, hidden_dim)
                hidden_ref = outputs_ref.hidden_states[-1]
                hidden_alt = outputs_alt.hidden_states[-1]

                # compute true sequence lengths so we know where the "last" token is
                ref_seq_lens = ref_attention_mask.sum(dim=-1)  # (batch_size,)
                alt_seq_lens = alt_attention_mask.sum(dim=-1)  # (batch_size,)

                # build an index for batch dimension
                batch_index = torch.arange(ref_seq_lens.size(0), device=hidden_ref.device)

                # select the last-token vector for each example
                last_ref = hidden_ref[batch_index, ref_seq_lens - 1, :]  # (batch_size, hidden_dim)
                last_alt = hidden_alt[batch_index, alt_seq_lens - 1, :]  # (batch_size, hidden_dim)

                # take the difference
                last_hidden_state = last_alt - last_ref   # (batch_size, hidden_dim)
                # run each sample's difference vector through its task head
                logits_list = []
                for i, task in enumerate(task_names):
                    logits_list.append(model.task_classification_heads[task](last_hidden_state[i]))
                logits = torch.stack(logits_list, dim=0)

            # Compute loss if needed (per-sample to handle mixed batches)
            loss = None
            if not prediction_loss_only:
                losses = []
                for i, task in enumerate(task_names):
                    sample_logits = logits[i:i+1]
                    sample_label = labels[i:i+1]

                    if "MAVES" in task:
                        # Regression task
                        sample_loss = torch.nn.functional.huber_loss(
                            sample_logits.squeeze(-1),
                            sample_label.float(),
                            delta=1.0
                        )
                    else:
                        # Classification task
                        sample_loss = torch.nn.functional.cross_entropy(
                            sample_logits,
                            sample_label,
                            label_smoothing=0.1
                        )
                    losses.append(sample_loss)

                loss = torch.stack(losses).mean()

            # Restore task_names to inputs for potential future use
            if task_names is not None:
                inputs["task_names"] = task_names

            return loss, logits, labels

def run_multitask_finetune(tasks, seed, model_type='nt'):
    set_seed(seed)

    # Configuration
    path_prefix = "/mnt/data/genomic_fm/finetune" # for amlt
    results_file = f"{path_prefix}/test_results_multitask.csv"

    # Model and Tokenizer Selection
    if model_type == 'omni_dna_116m':
        model = AutoModelForSequenceClassification.from_pretrained(
            "zehui127/Omni-DNA-116M",
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
    elif model_type=='dnabert2':
        # requiring transformer 4.29.0
        # tiktoken gdown tiktoken datasets wandb
        # pip uninstall triton
        model = AutoModelForSequenceClassification.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    elif model_type=='luca':
        from lucagplm import LucaGPLMModel, LucaGPLMTokenizer
        model = LucaGPLMModel.from_pretrained("LucaGroup/LucaOne-default-step36M")
        tokenizer = LucaGPLMTokenizer.from_pretrained("LucaGroup/LucaOne-default-step36M")
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
        num_train_epochs=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="matthews_correlation",
        greater_is_better=True,
        save_total_limit=2,
        load_best_model_at_end=True,
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
        eval_dataset=multitask_datasets.get("CLNDN_val"),  # Use the combined validation dataset
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
                        help="Model type (e.g., omni_dna_116m, nt, dnabert2, luca)")
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
