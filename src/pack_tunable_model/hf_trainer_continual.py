"""
LoRA + Replay Trainer for Continual Learning of Genomic Foundation Models

Key features:
1. LoRA fine-tuning (parameter-efficient, ~1% of params)
2. Simple replay buffer for preventing catastrophic forgetting
3. Optional SLAO (orthogonal init + asymmetric merging) for advanced continual learning

Usage:
    # Simple continual learning with replay
    accelerate launch -m src.pack_tunable_model.hf_trainer_continual \
        --model nt --task CLNSIG --use_lora --use_replay

    # With SLAO for multi-task continual learning
    accelerate launch -m src.pack_tunable_model.hf_trainer_continual \
        --model nt --task CLNDN --use_lora --use_replay --slao --task_number 2 \
        --lora_checkpoint ./root/models/continual_nt_CLNSIG_lora_replay/lora_adapter
"""

import os
import math
import torch
import argparse
import logging
import csv
import numpy as np
from typing import Optional, Dict, List
from scipy.stats import spearmanr
import sklearn.metrics

from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    TrainingArguments,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    set_seed,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from accelerate import Accelerator

from .hf_dataloader import (
    return_clinvar_multitask_dataset,
    return_maves_dataset,
    MultiTaskDataCollator,
)
from .wrap_model import WrappedModelWithClassificationHead
from .replay_buffer import ReplayBuffer, collate_replay_samples


# =============================================================================
# SLAO: Single LoRA Continual Learning (Optional)
# Based on "Single LoRA via orthogonal init + asymmetric merging" (2025)
# =============================================================================

def compute_orthogonal_initialization(prev_lora_B: torch.Tensor, new_rank: int, device=None) -> torch.Tensor:
    """Initialize new LoRA B matrix orthogonal to previous task's B directions."""
    if device is None:
        device = prev_lora_B.device

    out_features = prev_lora_B.shape[0]
    prev_rank = prev_lora_B.shape[1]

    Q, _ = torch.linalg.qr(prev_lora_B.float())

    if out_features > prev_rank:
        null_space_dim = out_features - prev_rank
        random_init = torch.randn(out_features, min(new_rank, null_space_dim), device=device)
        projection = Q @ (Q.T @ random_init)
        orthogonal_init = random_init - projection
        norms = orthogonal_init.norm(dim=0, keepdim=True).clamp(min=1e-8)
        orthogonal_init = orthogonal_init / norms

        if new_rank > null_space_dim:
            extra = torch.randn(out_features, new_rank - null_space_dim, device=device) * 0.01
            orthogonal_init = torch.cat([orthogonal_init, extra], dim=1)

        return orthogonal_init.to(prev_lora_B.dtype)
    else:
        return torch.randn(out_features, new_rank, device=device, dtype=prev_lora_B.dtype) * 0.01


def apply_slao_initialization(model, prev_checkpoint_path: str, task_number: int, accelerator=None):
    """Apply SLAO-style initialization: load previous LoRA and init B orthogonally."""
    if not isinstance(model, PeftModel):
        if accelerator:
            accelerator.print("Warning: SLAO requires PeftModel, skipping")
        return None

    safetensors_path = f"{prev_checkpoint_path}/adapter_model.safetensors"
    bin_path = f"{prev_checkpoint_path}/adapter_model.bin"

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        prev_state = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        prev_state = torch.load(bin_path, map_location="cpu")
    else:
        if accelerator:
            accelerator.print(f"Warning: No adapter weights found at {prev_checkpoint_path}")
        return None

    prev_lora_state = {}

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            key_name = name.replace("base_model.model.", "").replace("base_model.", "")

            if key_name in prev_state:
                prev_lora_state[name] = prev_state[key_name].clone()

                if "lora_B" in name:
                    prev_B = prev_state[key_name]
                    new_rank = param.shape[1] if param.dim() > 1 else param.shape[0]

                    if prev_B.dim() == 2 and prev_B.shape[0] == param.shape[0]:
                        orthog_init = compute_orthogonal_initialization(prev_B, new_rank, device=param.device)
                        with torch.no_grad():
                            if param.shape == orthog_init.shape:
                                param.copy_(orthog_init)
                            else:
                                param.copy_(prev_state[key_name].to(param.device))
                    else:
                        param.copy_(prev_state[key_name].to(param.device))
                else:
                    with torch.no_grad():
                        param.copy_(prev_state[key_name].to(param.device))

    if accelerator:
        accelerator.print(f"SLAO: Loaded {len(prev_lora_state)} LoRA params with orthogonal B init")

    return prev_lora_state


class SLAOMergingCallback(TrainerCallback):
    """Callback for SLAO-style asymmetric merging during training."""

    def __init__(self, prev_lora_state: Dict[str, torch.Tensor], task_number: int,
                 merge_frequency: int = 100, alpha_A: float = 0.5):
        self.prev_lora_state = prev_lora_state
        self.task_number = task_number
        self.merge_frequency = merge_frequency
        self.alpha_A = alpha_A

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.merge_frequency != 0 or self.prev_lora_state is None:
            return

        lambda_t = 1.0 / math.sqrt(max(self.task_number, 1))

        for name, param in model.named_parameters():
            if name not in self.prev_lora_state:
                continue

            prev_weight = self.prev_lora_state[name].to(param.device)
            if prev_weight.shape != param.shape:
                continue

            merge_ratio = lambda_t * self.alpha_A if "lora_A" in name else lambda_t

            with torch.no_grad():
                param.data = param.data - 0.01 * merge_ratio * (param.data - prev_weight)


class ReplayBufferFlushCallback(TrainerCallback):
    """Flush remaining samples from ReplayTrainer's buffer at training end."""

    def __init__(self, trainer: "ReplayTrainer"):
        self.trainer = trainer

    def on_train_end(self, args, state, control, **kwargs):
        if self.trainer._sample_buffer and self.trainer.replay_buffer is not None:
            for sample in self.trainer._sample_buffer:
                self.trainer.replay_buffer.add(*sample, task_id=self.trainer.task_id)
            self.trainer._sample_buffer = []


# =============================================================================
# Metrics
# =============================================================================

def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray, task_type="classification"):
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    if task_type == "regression":
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
        return {
            "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
            "f1": sklearn.metrics.f1_score(valid_labels, valid_predictions, average="macro", zero_division=0),
            "matthews_correlation": sklearn.metrics.matthews_corrcoef(valid_labels, valid_predictions),
            "precision": sklearn.metrics.precision_score(valid_labels, valid_predictions, average="macro", zero_division=0),
            "recall": sklearn.metrics.recall_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        }


def preprocess_logits_for_metrics(logits: torch.Tensor, labels: Optional[torch.Tensor] = None):
    if logits is None:
        if labels is not None:
            return torch.tensor([], device=labels.device)
        return torch.tensor([])
    if logits.shape[-1] == 1:
        predictions = logits.squeeze(-1)
    else:
        predictions = torch.argmax(logits, dim=-1)
    return predictions


# =============================================================================
# Replay Trainer
# =============================================================================

class ReplayTrainer(Trainer):
    """
    Trainer with data-level replay mixing.

    Based on "Replaying pre-training data improves fine-tuning" (Kotha & Liang):
    - Mix replay samples directly into batches (data-level, not loss-level)
    - Higher replay ratios (0.5) improve target task performance
    - Simple random sampling from buffer
    """

    def __init__(
        self,
        replay_buffer: Optional[ReplayBuffer] = None,
        task_id: str = "default",
        replay_ratio: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.replay_buffer = replay_buffer
        self.task_id = task_id
        # Guard against division by zero (replay_ratio must be < 1.0)
        self.replay_ratio = min(replay_ratio, 0.99)
        self._sample_buffer: List[tuple] = []

    def _concat_batches(self, batch1: Dict, batch2: Dict) -> Dict:
        """Concatenate two batches along the batch dimension."""
        combined = {}
        all_keys = set(batch1.keys()) | set(batch2.keys())
        for key in all_keys:
            if key in batch1 and key in batch2:
                if torch.is_tensor(batch1[key]) and torch.is_tensor(batch2[key]):
                    combined[key] = torch.cat([batch1[key], batch2[key]], dim=0)
                else:
                    combined[key] = batch1[key]
            elif key in batch1:
                combined[key] = batch1[key]
            else:
                combined[key] = batch2[key]
        return combined

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Data-level mixing: concatenate replay samples to batch before forward pass."""

        # Add current samples to replay buffer (for future tasks)
        if self.model.training and self.replay_buffer is not None:
            with torch.no_grad():
                batch_size = inputs["labels"].shape[0]
                num_to_add = max(1, batch_size // 4)
                indices = torch.randperm(batch_size)[:num_to_add]

                for idx in indices.tolist():
                    self._sample_buffer.append((
                        inputs["ref_input_ids"][idx].cpu(),
                        inputs["ref_attention_mask"][idx].cpu(),
                        inputs["alt_input_ids"][idx].cpu(),
                        inputs["alt_attention_mask"][idx].cpu(),
                        inputs["labels"][idx].cpu(),
                    ))

                if len(self._sample_buffer) >= 50:
                    for sample in self._sample_buffer[:25]:
                        self.replay_buffer.add(*sample, task_id=self.task_id)
                    self._sample_buffer = self._sample_buffer[25:]

        # Data-level mixing: concatenate replay samples to current batch
        if self.model.training and self.replay_buffer and self.replay_buffer.total_size > 0:
            current_batch_size = inputs["labels"].shape[0]
            # Calculate replay samples to match desired ratio
            # If ratio=0.5, we want equal parts current and replay
            replay_batch_size = int(current_batch_size * self.replay_ratio / (1 - self.replay_ratio))
            replay_batch_size = max(1, min(replay_batch_size, current_batch_size))

            replay_samples = self.replay_buffer.sample(replay_batch_size)

            if replay_samples:
                replay_batch = collate_replay_samples(replay_samples, device=inputs["labels"].device)
                # Concatenate replay samples to current batch
                combined_inputs = self._concat_batches(inputs, replay_batch)

                # Single forward pass on combined batch
                outputs = model(**combined_inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

                return (loss, outputs) if return_outputs else loss

        # No replay available, standard forward pass
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss


# =============================================================================
# LoRA Configuration
# =============================================================================

def get_lora_config(model_type: str, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1) -> LoraConfig:
    """Get LoRA config for different genomic model architectures."""
    if model_type in ["nt", "dnabert2", "gena-lm"]:
        target_modules = ["query", "key", "value", "dense"]
    elif model_type == "hyenadna":
        target_modules = ["in_proj", "out_proj"]
    elif model_type == "caduceus":
        target_modules = ["in_proj", "out_proj", "x_proj"]
    elif model_type == "gpn-star":
        target_modules = ["query", "key", "value", "dense"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )


# =============================================================================
# Main Training Function
# =============================================================================

def run_continual_finetune(
    task: str,
    seed: int,
    model_type: str = "nt",
    decoder: bool = False,
    test_only: bool = False,
    learning_rate: float = 1e-4,
    batch_size: int = 8,
    num_epochs: int = 10,
    max_grad_norm: float = 1.0,
    num_workers: int = 8,
    # LoRA options
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_checkpoint: Optional[str] = None,
    # Replay options (based on Kotha & Liang 2026)
    use_replay: bool = True,
    replay_buffer_path: Optional[str] = None,
    replay_buffer_size: int = 5000,
    replay_ratio: float = 0.5,
    # SLAO options
    slao: bool = False,
    task_number: int = 1,
    slao_alpha_A: float = 0.5,
    slao_merge_frequency: int = 100,
    # MAVES filters
    filter_genes=None,
    experimental_methods=None,
    region_type="all",
    variant_types=None,
    seq_length_range=None,
    max_samples_per_experiment=None,
    normalize_scores: bool = False,
    comparison_mode: str = "delta",
):
    """Run LoRA fine-tuning with optional replay buffer."""
    set_seed(seed)
    accelerator = Accelerator()

    path_prefix = "./root/models"
    results_file = f"{path_prefix}/test_results_continual.csv"

    # =========================================================================
    # Model Loading
    # =========================================================================
    local_model_base = f"./root/models/{model_type}"

    model_paths = {
        "nt": ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", AutoModelForMaskedLM),
        "dnabert2": ("zhihan1996/DNABERT-2-117M", AutoModel),
        "hyenadna": ("LongSafari/hyenadna-medium-160k-seqlen-hf", AutoModel),
        "caduceus": ("kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16", AutoModel),
        "gena-lm": ("AIRI-Institute/gena-lm-bert-base-t2t", AutoModel),
        "gpn-star": ("songlab/gpn-star-hg38-v100-200m", AutoModelForMaskedLM),
        "omni_dna_116m": ("zehui127/Omni-DNA-116M", AutoModel),
        "omni_dna_1b": ("zehui127/Omni-DNA-1B", AutoModel),
        "luca": ("InstaDeepAI/LUCA-GenomeFoundation-v0_5-2B", AutoModel),
    }

    if model_type not in model_paths:
        raise ValueError(f"Unsupported model: {model_type}. Supported: {list(model_paths.keys())}")

    hf_path, model_cls = model_paths[model_type]
    model_path = local_model_base if os.path.exists(local_model_base) else hf_path

    accelerator.print(f"Loading model from: {model_path}")

    base_model = model_cls.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=os.path.exists(model_path)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=os.path.exists(model_path)
    )

    # =========================================================================
    # Load Dataset
    # =========================================================================
    if task == "MAVES":
        datasets, task_num_classes, max_seq_len = return_maves_dataset(
            tokenizer, target="score", seq_length=1024, seed=seed,
            filter_genes=filter_genes,
            experimental_methods=experimental_methods,
            region_type=region_type,
            variant_types=variant_types,
            seq_length_range=seq_length_range,
            max_samples_per_experiment=max_samples_per_experiment,
            normalize_scores=normalize_scores,
        )
        task = "MAVES_score"
    else:
        datasets, task_num_classes, max_seq_len = return_clinvar_multitask_dataset(
            tokenizer, task, seed=seed
        )

    tokenizer.model_max_length = max_seq_len
    num_classes = task_num_classes[task]
    accelerator.print(f"Task: {task}, Classes: {num_classes}")

    # =========================================================================
    # Apply LoRA
    # =========================================================================
    prev_lora_state = None
    if use_lora:
        lora_config = get_lora_config(model_type, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        base_model = get_peft_model(base_model, lora_config)

        if lora_checkpoint and os.path.exists(lora_checkpoint):
            if slao and task_number > 1:
                accelerator.print(f"SLAO: Loading LoRA with orthogonal init from: {lora_checkpoint}")
                prev_lora_state = apply_slao_initialization(
                    base_model, lora_checkpoint, task_number, accelerator
                )
            else:
                accelerator.print(f"Loading LoRA checkpoint: {lora_checkpoint}")
                base_model.load_adapter(lora_checkpoint, adapter_name="default")

        trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in base_model.parameters())
        accelerator.print(f"LoRA trainable params: {trainable_params:,} / {total_params:,} "
                         f"({100 * trainable_params / total_params:.2f}%)")

    # =========================================================================
    # Wrap with Classification Head
    # =========================================================================
    model = WrappedModelWithClassificationHead(
        base_model, num_classes, decoder=decoder, comparison_mode=comparison_mode
    )

    # =========================================================================
    # Setup Replay Buffer
    # =========================================================================
    replay_buffer = None
    if use_replay:
        if replay_buffer_path and os.path.exists(replay_buffer_path):
            replay_buffer = ReplayBuffer.load(replay_buffer_path)
            accelerator.print(f"Loaded replay buffer: {replay_buffer.task_sizes()}")
        else:
            replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
            accelerator.print(f"Created new replay buffer (max_size={replay_buffer_size})")

    # =========================================================================
    # Training Setup
    # =========================================================================
    lora_suffix = "_lora" if use_lora else ""
    replay_suffix = "_replay" if use_replay else ""
    output_path = f"{path_prefix}/continual_{model_type}_{task}{lora_suffix}{replay_suffix}"

    task_type = "regression" if task.startswith("MAVES") else "classification"

    training_args = TrainingArguments(
        output_dir=output_path,
        run_name=f"continual_{model_type}_{task}",
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        save_total_limit=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="spearman_correlation" if task_type == "regression" else "matthews_correlation",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_safetensors=False,
        remove_unused_columns=False,
        dataloader_num_workers=num_workers,
        logging_steps=50,
    )

    if task_type == "regression":
        training_args.warmup_ratio = 0.1
        training_args.gradient_accumulation_steps = 2

    data_collator = MultiTaskDataCollator(tokenizer)

    def compute_metrics_for_task(eval_pred):
        predictions, labels = eval_pred
        return calculate_metric_with_sklearn(predictions, labels, task_type)

    # =========================================================================
    # Create Trainer
    # =========================================================================
    callbacks = []
    if slao and prev_lora_state is not None:
        slao_callback = SLAOMergingCallback(
            prev_lora_state=prev_lora_state,
            task_number=task_number,
            merge_frequency=slao_merge_frequency,
            alpha_A=slao_alpha_A,
        )
        callbacks.append(slao_callback)
        accelerator.print(f"SLAO: Enabled asymmetric merging (task {task_number})")

    if use_replay:
        trainer = ReplayTrainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets.get(f"{task}_val"),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics_for_task,
            data_collator=data_collator,
            callbacks=callbacks if callbacks else None,
            replay_buffer=replay_buffer,
            task_id=task,
            replay_ratio=replay_ratio,
        )
        # Add callback to flush remaining samples at training end
        trainer.add_callback(ReplayBufferFlushCallback(trainer))
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets.get(f"{task}_val"),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics_for_task,
            data_collator=data_collator,
            callbacks=callbacks if callbacks else None,
        )

    # =========================================================================
    # Training
    # =========================================================================
    if not test_only:
        trainer.train()

        # Save LoRA adapter
        if use_lora:
            lora_save_path = f"{output_path}/lora_adapter"
            base_model.save_pretrained(lora_save_path)
            accelerator.print(f"Saved LoRA adapter to: {lora_save_path}")

        # Save replay buffer
        if replay_buffer:
            buffer_save_path = f"{path_prefix}/replay_buffer_{task}.pt"
            replay_buffer.save(buffer_save_path)
            accelerator.print(f"Saved replay buffer to: {buffer_save_path}")

    # =========================================================================
    # Evaluation
    # =========================================================================
    test_dataset = datasets.get(f"{task}_test")
    if test_dataset:
        test_metrics = trainer.evaluate(eval_dataset=test_dataset)
        accelerator.print(f"Test Metrics: {test_metrics}")

        # Log results
        write_header = not os.path.exists(results_file)
        with open(results_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Seed", "Model", "Task", "LoRA", "Replay", "Metric"])
            metric_key = "eval_spearman_correlation" if task_type == "regression" else "eval_matthews_correlation"
            writer.writerow([seed, model_type, task, use_lora, use_replay, test_metrics.get(metric_key, "N/A")])

        return test_metrics

    return {}


def main():
    parser = argparse.ArgumentParser(description="LoRA + Replay Continual Learning")

    # Model and task
    parser.add_argument("--model", type=str, default="nt",
                        help="Model type (e.g., nt, omni_dna_116m, dnabert2)")
    parser.add_argument("--task", type=str, default="CLNSIG",
                        choices=["CLNDN", "CLNSIG", "MAVES"])
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument("--decoder", action="store_true")
    parser.add_argument("--test_only", action="store_true")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)

    # LoRA options
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="Previous LoRA checkpoint for continual learning")

    # Replay options (based on Kotha & Liang 2026)
    parser.add_argument("--use_replay", action="store_true", help="Enable replay buffer")
    parser.add_argument("--replay_buffer", type=str, default=None,
                        help="Path to existing replay buffer")
    parser.add_argument("--replay_buffer_size", type=int, default=5000,
                        help="Max samples in replay buffer")
    parser.add_argument("--replay_ratio", type=float, default=0.5,
                        help="Fraction of combined batch from replay (0.5 = equal mix)")

    # SLAO options (optional advanced continual learning)
    parser.add_argument("--slao", action="store_true",
                        help="Enable SLAO: orthogonal init + asymmetric merging")
    parser.add_argument("--task_number", type=int, default=1,
                        help="Current task number for SLAO time-aware scaling")
    parser.add_argument("--slao_alpha_A", type=float, default=0.5,
                        help="SLAO: preservation factor for A matrices")
    parser.add_argument("--slao_merge_frequency", type=int, default=100,
                        help="SLAO: merge with previous weights every N steps")

    # MAVES options
    parser.add_argument("--filter_genes", type=str, default=None)
    parser.add_argument("--experimental_methods", type=str, default=None)
    parser.add_argument("--region_type", type=str, default="all")
    parser.add_argument("--variant_types", type=str, default=None)
    parser.add_argument("--seq_len_min", type=int, default=None)
    parser.add_argument("--seq_len_max", type=int, default=1024)
    parser.add_argument("--max_samples_per_experiment", type=int, default=None)
    parser.add_argument("--normalize_scores", action="store_true")

    # Other
    parser.add_argument("--comparison_mode", type=str, default="delta", choices=["delta", "concat"])

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Parse filters
    filter_genes = args.filter_genes.split(",") if args.filter_genes else None
    experimental_methods = args.experimental_methods.split(",") if args.experimental_methods else None
    variant_types = args.variant_types.split(",") if args.variant_types else None

    seq_length_range = None
    if args.seq_len_min or args.seq_len_max:
        seq_length_range = (args.seq_len_min or 0, args.seq_len_max or float("inf"))

    run_continual_finetune(
        task=args.task,
        seed=args.seed,
        model_type=args.model,
        decoder=args.decoder,
        test_only=args.test_only,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_checkpoint=args.lora_checkpoint,
        use_replay=args.use_replay,
        replay_buffer_path=args.replay_buffer,
        replay_buffer_size=args.replay_buffer_size,
        replay_ratio=args.replay_ratio,
        slao=args.slao,
        task_number=args.task_number,
        slao_alpha_A=args.slao_alpha_A,
        slao_merge_frequency=args.slao_merge_frequency,
        filter_genes=filter_genes,
        experimental_methods=experimental_methods,
        region_type=args.region_type,
        variant_types=variant_types,
        seq_length_range=seq_length_range,
        max_samples_per_experiment=args.max_samples_per_experiment,
        normalize_scores=args.normalize_scores,
        comparison_mode=args.comparison_mode,
    )


if __name__ == "__main__":
    main()
