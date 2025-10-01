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

# Import from local modules
from .hf_dataloader import return_clinvar_multitask_dataset, MultiTaskDataCollator
from ..pack_tunable_model.wrap_model import WrappedModelWithClassificationHead


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
    # Get the argmax predictions
    predictions = torch.argmax(logits, dim=-1)
    return predictions

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)

class CLNDNTrainer(transformers.Trainer):
    def __init__(self, decoder=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = decoder

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss computation using the classification head attached to model
        """
        labels = inputs.get("labels")
        core = getattr(model, "module", model)
        # Get hidden states from model
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
            # For encoder models
            last_hidden_state_ref = outputs_ref.hidden_states[-1][:,0,:]
            last_hidden_state_alt = outputs_alt.hidden_states[-1][:,0,:]
            last_hidden_state = last_hidden_state_alt - last_hidden_state_ref
            logits = core.classification_head(last_hidden_state)
        else:
            # For decoder models
            ref_attention_mask = inputs['ref_attention_mask']
            alt_attention_mask = inputs['alt_attention_mask']

            # shape: (batch_size, seq_len, hidden_dim)
            hidden_ref = outputs_ref.decoder_hidden_states[-1]
            hidden_alt = outputs_alt.decoder_hidden_states[-1]

            # compute true sequence lengths so we know where the "last" token is
            ref_seq_lens = ref_attention_mask.sum(dim=-1)  # (batch_size,)
            alt_seq_lens = alt_attention_mask.sum(dim=-1)  # (batch_size,)

            # build an index for batch dimension
            batch_index = torch.arange(ref_seq_lens.size(0), device=hidden_ref.device)

            # select the last‐token vector for each example
            last_ref = hidden_ref[batch_index, ref_seq_lens - 1, :]  # (batch_size, hidden_dim)
            last_alt = hidden_alt[batch_index, alt_seq_lens - 1, :]  # (batch_size, hidden_dim)

            # take the difference
            last_hidden_state = last_alt - last_ref   # (batch_size, hidden_dim)
            # Apply classification head
            logits = core.classification_head(last_hidden_state)

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        return (loss, (logits, outputs)) if return_outputs else loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prediction step using the classification head attached to model
        """
        with torch.no_grad():
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
            core = getattr(model, "module", model)
            if not self.decoder:
                # Get last hidden state for encoder models
                last_hidden_state_ref = outputs_ref.hidden_states[-1][:,0,:]
                last_hidden_state_alt = outputs_alt.hidden_states[-1][:,0,:]
                last_hidden_state = last_hidden_state_alt - last_hidden_state_ref
                # Apply classification head
                logits = core.classification_head(last_hidden_state)
            else:
                # For decoder models
                ref_attention_mask = inputs['ref_attention_mask']
                alt_attention_mask = inputs['alt_attention_mask']

                # shape: (batch_size, seq_len, hidden_dim)
                hidden_ref = outputs_ref.decoder_hidden_states[-1]
                hidden_alt = outputs_alt.decoder_hidden_states[-1]

                # compute true sequence lengths
                ref_seq_lens = ref_attention_mask.sum(dim=-1)  # (batch_size,)
                alt_seq_lens = alt_attention_mask.sum(dim=-1)  # (batch_size,)

                # build an index for batch dimension
                batch_index = torch.arange(ref_seq_lens.size(0), device=hidden_ref.device)

                # select the last‐token vector for each example
                last_ref = hidden_ref[batch_index, ref_seq_lens - 1, :]  # (batch_size, hidden_dim)
                last_alt = hidden_alt[batch_index, alt_seq_lens - 1, :]  # (batch_size, hidden_dim)

                # take the difference
                last_hidden_state = last_alt - last_ref   # (batch_size, hidden_dim)
                # Apply classification head
                logits = core.classification_head(last_hidden_state)

            # Compute loss if needed
            loss = None
            if not prediction_loss_only:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)

            return loss, logits, labels

def run_single_task_finetune(task, seed, model_type='nt', decoder=False, test_only=False):
    set_seed(seed)

    # Configuration
    path_prefix = "/mnt/data/genomic_fm/finetune" # for amlt
    # path_prefix = "/home/v-zehuili/repositories/amlt/codes/genomic-FM/root/clinvar_disease_classification" # for amlt
    results_file = f"{path_prefix}/test_results_clinvar.csv"

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
        model = AutoModelForSequenceClassification.from_pretrained(
            f"zhihan1996/DNABERT-2-117M",
            num_labels=2,  # Default for binary classification
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            trust_remote_code=True
        )
    elif model_type=='luca':
        from lucagplm import LucaGPLMModel, LucaGPLMTokenizer
        model = LucaGPLMModel.from_pretrained("LucaGroup/LucaOne-default-step36M")
        tokenizer = LucaGPLMTokenizer.from_pretrained("LucaGroup/LucaOne-default-step36M")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    tokenizer.model_max_length = 1000

    # Load Dataset
    datasets, task_num_classes, max_seq_len = return_clinvar_multitask_dataset(
        tokenizer, task, seed=seed
    )
    # Create and attach classification head to the model
    num_classes = task_num_classes[task]
    model = WrappedModelWithClassificationHead(model, num_classes, decoder=decoder)

    # Load saved classification head if testing
    if test_only:
        state_dict = torch.load("/home/v-zehuili/repositories/genomic-FM/root/clinvar_disease_classification/checkpoint-55213/pytorch_model.bin")
        head_state_dict = {k.replace('task_classification_heads.CLNDN.', ''): v
                         for k, v in state_dict.items()
                         if k.startswith('task_classification_heads.CLNDN.')}
        if head_state_dict:
            # Adapt the keys to match the new structure
            model_head_dict = {}
            for k, v in head_state_dict.items():
                if k.startswith('classifier.'):
                    model_head_dict[k.replace('classifier.', '')] = v
                else:
                    model_head_dict[k] = v

            model.classification_head.load_state_dict(model_head_dict)

    # Make sure the classification head is on the same device as the model
    model.classification_head = model.classification_head.to(model.device)

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

    # Create Trainer
    trainer = CLNDNTrainer(
        decoder=decoder,
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

    # Evaluation
    test_dataset = datasets.get(f"{task}_test")
    if test_dataset:
        test_metrics = trainer.evaluate(eval_dataset=test_dataset)
        print(f"Test Metrics for {task}: {test_metrics}")

        # Log results
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
                        help="Model type (e.g., omni_dna_116m, nt, dnabert2, luca)")
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
    run_single_task_finetune('CLNDN', args.seed, args.model, args.decoder, args.test_only)

if __name__ == "__main__":
    main()
