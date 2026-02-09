"""
LoRA + Replay + SLAO Trainer for Continual Learning of Genomic Foundation Models

Based on:
- SLAO (2025): Single LoRA continual learning via orthogonal init and asymmetric merging
- ACM Survey: Combining replay + regularization for best CL performance

Key features:
1. LoRA fine-tuning (parameter-efficient, ~1% of params)
2. Surprise-based replay buffer with per-sample loss tracking
3. Optional EWC regularization to protect important weights
4. LoRA checkpoint chaining for continual learning
5. SLAO: Orthogonal initialization + asymmetric A/B treatment + time-aware scaling

Usage:
    # Single task with replay
    accelerate launch -m src.pack_tunable_model.hf_trainer_continual \
        --model nt --task CLNSIG --use_lora --use_replay

    # Continual learning: Task B after Task A
    accelerate launch -m src.pack_tunable_model.hf_trainer_continual \
        --model nt --task CLNDN --use_lora --use_replay \
        --replay_buffer ./root/models/replay_buffer_CLNSIG.pt \
        --lora_checkpoint ./root/models/lora_nt_CLNSIG

    # With EWC regularization (recommended by ACM Survey)
    accelerate launch -m src.pack_tunable_model.hf_trainer_continual \
        --model nt --task CLNDN --use_lora --use_replay --use_ewc \
        --replay_buffer ./root/models/replay_buffer_CLNSIG.pt \
        --ewc_fisher ./root/models/fisher_CLNSIG.pt
"""

import os
import torch
import argparse
import logging
import csv
import numpy as np
from typing import Optional, Dict, Any, List
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
from peft import get_peft_model, LoraConfig, PeftModel, TaskType
from accelerate import Accelerator

from .hf_dataloader import (
    return_clinvar_multitask_dataset,
    return_maves_dataset,
    MultiTaskDataCollator,
)
from .wrap_model import WrappedModelWithClassificationHead
from .replay_buffer import ReplayBuffer, ReplaySample, collate_replay_samples
from copy import deepcopy
import math


# =============================================================================
# SLAO: Single LoRA Continual Learning via Orthogonal Init + Asymmetric Merging
# Based on "Merge Before Forget: A Single LoRA Continual Learning" (2025)
# =============================================================================

def compute_orthogonal_initialization(
    prev_lora_B: torch.Tensor,
    new_rank: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Initialize new LoRA B matrix orthogonal to previous task's B directions.

    SLAO insight: Previous task knowledge lives in the column space of B.
    By initializing new B orthogonal to this space, we avoid overwriting
    old task representations.

    Args:
        prev_lora_B: Previous task's LoRA B matrix [out_features, rank]
        new_rank: Rank for new LoRA (can differ from previous)
        device: Device to place output tensor

    Returns:
        Orthogonally initialized B matrix [out_features, new_rank]
    """
    if device is None:
        device = prev_lora_B.device

    out_features = prev_lora_B.shape[0]
    prev_rank = prev_lora_B.shape[1]

    # QR decomposition to find orthonormal basis of previous B's column space
    Q, R = torch.linalg.qr(prev_lora_B.float())

    # The null space (orthogonal complement) starts after the first prev_rank columns
    # If out_features > prev_rank, we have room for orthogonal directions
    if out_features > prev_rank:
        # Generate random vectors in the null space
        null_space_dim = out_features - prev_rank
        random_init = torch.randn(out_features, min(new_rank, null_space_dim), device=device)

        # Project out any component in Q's column space
        # new_B = random_init - Q @ Q^T @ random_init
        projection = Q @ (Q.T @ random_init)
        orthogonal_init = random_init - projection

        # Normalize columns
        norms = orthogonal_init.norm(dim=0, keepdim=True).clamp(min=1e-8)
        orthogonal_init = orthogonal_init / norms

        # If we need more columns than null space allows, pad with small random
        if new_rank > null_space_dim:
            extra = torch.randn(out_features, new_rank - null_space_dim, device=device) * 0.01
            orthogonal_init = torch.cat([orthogonal_init, extra], dim=1)

        return orthogonal_init.to(prev_lora_B.dtype)
    else:
        # No room for truly orthogonal init, use small random (rare case)
        return torch.randn(out_features, new_rank, device=device, dtype=prev_lora_B.dtype) * 0.01


def asymmetric_lora_merge(
    old_A: torch.Tensor,
    old_B: torch.Tensor,
    new_A: torch.Tensor,
    new_B: torch.Tensor,
    task_number: int,
    alpha_A: float = 0.5,  # Preservation factor for A (higher = more preservation)
) -> tuple:
    """
    Asymmetrically merge old and new LoRA weights using time-aware scaling.

    SLAO insight:
    - A matrices are more similar across tasks (shared feature extraction)
    - B matrices are task-specific (output projections)

    Uses λ(t) = 1/√t time-aware scaling where t = task_number.

    Args:
        old_A: Previous LoRA A matrix
        old_B: Previous LoRA B matrix
        new_A: New task's LoRA A matrix
        new_B: New task's LoRA B matrix
        task_number: Current task number (1-indexed, used for λ(t))
        alpha_A: How much to preserve A vs B (0.5 = symmetric)

    Returns:
        Tuple of (merged_A, merged_B)
    """
    # Time-aware scaling: λ(t) = 1/√t
    # As more tasks are learned, be more conservative with updates
    lambda_t = 1.0 / math.sqrt(max(task_number, 1))

    # Asymmetric treatment:
    # A gets less update (more preserved) because it's more similar across tasks
    # B gets more update (more plastic) because it's task-specific
    lambda_A = lambda_t * alpha_A  # Reduced plasticity for A
    lambda_B = lambda_t            # Full plasticity for B

    # Merge: new = (1 - λ) * old + λ * new
    merged_A = (1 - lambda_A) * old_A + lambda_A * new_A
    merged_B = (1 - lambda_B) * old_B + lambda_B * new_B

    return merged_A, merged_B


def apply_slao_initialization(
    model,
    prev_checkpoint_path: str,
    task_number: int,
    use_orthogonal_init: bool = True,
    accelerator = None,
):
    """
    Apply SLAO-style initialization to a LoRA model.

    1. Load previous LoRA weights
    2. Optionally initialize new B matrices orthogonal to previous
    3. Store previous weights for potential merging during training

    Args:
        model: PeftModel with LoRA
        prev_checkpoint_path: Path to previous task's LoRA adapter
        task_number: Current task number (1 = first task, 2 = second, etc.)
        use_orthogonal_init: Whether to apply orthogonal initialization
        accelerator: Accelerator for printing

    Returns:
        Dict containing previous LoRA state for merging
    """
    from peft import PeftModel

    if not isinstance(model, PeftModel):
        if accelerator:
            accelerator.print("Warning: SLAO requires PeftModel, skipping initialization")
        return None

    # Load the previous adapter weights (handle both safetensors and bin formats)
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

    # Store for later merging
    prev_lora_state = {}

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            # Find corresponding previous weight
            # Convert parameter name to state dict key format
            key_name = name.replace("base_model.model.", "").replace("base_model.", "")

            if key_name in prev_state:
                prev_lora_state[name] = prev_state[key_name].clone()

                if use_orthogonal_init and "lora_B" in name:
                    # Apply orthogonal initialization for B matrices
                    prev_B = prev_state[key_name]
                    new_rank = param.shape[1] if param.dim() > 1 else param.shape[0]

                    # Only apply if dimensions are compatible
                    if prev_B.dim() == 2 and prev_B.shape[0] == param.shape[0]:
                        orthog_init = compute_orthogonal_initialization(
                            prev_B, new_rank, device=param.device
                        )
                        with torch.no_grad():
                            if param.shape == orthog_init.shape:
                                param.copy_(orthog_init)
                                if accelerator:
                                    accelerator.print(f"  Orthogonal init: {name}")
                            else:
                                # Fall back to loading previous weights
                                param.copy_(prev_state[key_name].to(param.device))
                    else:
                        param.copy_(prev_state[key_name].to(param.device))
                else:
                    # For A matrices and when not using orthogonal init, load previous weights
                    with torch.no_grad():
                        param.copy_(prev_state[key_name].to(param.device))

    if accelerator:
        accelerator.print(f"SLAO: Loaded {len(prev_lora_state)} LoRA parameters from task {task_number - 1}")
        if use_orthogonal_init:
            accelerator.print(f"SLAO: Applied orthogonal initialization for B matrices")

    return prev_lora_state


class SLAOMergingCallback(TrainerCallback):
    """
    Callback that performs SLAO-style asymmetric merging during training.

    Periodically merges current LoRA weights with previous task weights
    using time-aware scaling and asymmetric A/B treatment.
    """

    def __init__(
        self,
        prev_lora_state: Dict[str, torch.Tensor],
        task_number: int,
        merge_frequency: int = 100,  # Merge every N steps
        alpha_A: float = 0.5,
    ):
        self.prev_lora_state = prev_lora_state
        self.task_number = task_number
        self.merge_frequency = merge_frequency
        self.alpha_A = alpha_A

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Periodically merge with previous task weights."""
        if state.global_step % self.merge_frequency != 0:
            return

        if self.prev_lora_state is None:
            return

        # Find and merge LoRA A/B pairs
        lora_params = {}
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                lora_params[name] = param

        # Group by layer (matching A and B)
        merged_count = 0
        for name, param in lora_params.items():
            if name not in self.prev_lora_state:
                continue

            prev_weight = self.prev_lora_state[name].to(param.device)

            if prev_weight.shape != param.shape:
                continue

            # Time-aware scaling
            lambda_t = 1.0 / math.sqrt(max(self.task_number, 1))

            # Asymmetric: A gets less update, B gets more
            if "lora_A" in name:
                merge_ratio = lambda_t * self.alpha_A
            else:  # lora_B
                merge_ratio = lambda_t

            # Soft merge: pull current weights toward previous
            with torch.no_grad():
                # Small pull toward previous weights (regularization effect)
                param.data = param.data - 0.01 * merge_ratio * (param.data - prev_weight)

            merged_count += 1


# =============================================================================
# Metrics (reused from hf_trainer.py)
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
# Replay Callback - Adds replay samples to training
# =============================================================================

class ReplayCallback(TrainerCallback):
    """
    Callback that adds high-loss samples to replay buffer during training.

    Integrates with HuggingFace Trainer to:
    1. Track sample losses during training
    2. Add surprising samples to buffer
    3. Optionally mix replay samples into batches
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        task_id: str,
        add_frequency: int = 10,  # Add to buffer every N steps
        top_k_per_batch: int = 2,  # Keep top-k highest loss samples per batch
    ):
        self.replay_buffer = replay_buffer
        self.task_id = task_id
        self.add_frequency = add_frequency
        self.top_k_per_batch = top_k_per_batch
        self._pending_samples: List[tuple] = []

    def on_step_end(self, args, state, control, **kwargs):
        """Called after each training step."""
        # Add pending samples to buffer periodically
        if state.global_step % self.add_frequency == 0 and self._pending_samples:
            # Sort by loss and keep top samples
            self._pending_samples.sort(key=lambda x: x[-1], reverse=True)
            for sample_data in self._pending_samples[:self.top_k_per_batch * self.add_frequency]:
                self.replay_buffer.add(*sample_data, task_id=self.task_id)
            self._pending_samples = []


class ReplayTrainer(Trainer):
    """
    Trainer with replay buffer integration and optional EWC regularization.

    Extends HuggingFace Trainer to:
    1. Compute per-sample losses for replay selection (SuRe-style)
    2. Mix replay samples into training batches
    3. Optional EWC regularization to protect important weights
    4. Track and log replay/continual learning statistics

    Based on ACM Computing Surveys recommendations for combining
    replay with parameter regularization for best CL performance.
    """

    def __init__(
        self,
        replay_buffer: Optional[ReplayBuffer] = None,
        task_id: str = "default",
        replay_ratio: float = 0.2,  # Fraction of batch from replay
        use_ewc: bool = False,  # Enable EWC regularization
        ewc_lambda: float = 0.5,  # EWC regularization strength
        fisher_samples: int = 200,  # Samples for Fisher estimation
        **kwargs
    ):
        super().__init__(**kwargs)
        self.replay_buffer = replay_buffer
        self.task_id = task_id
        self.replay_ratio = replay_ratio
        self._sample_losses: List[tuple] = []

        # EWC (Elastic Weight Consolidation) for parameter regularization
        self.use_ewc = use_ewc
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self._fisher_dict: Optional[Dict[str, torch.Tensor]] = None
        self._optimal_params: Optional[Dict[str, torch.Tensor]] = None

    def compute_fisher_information(self, dataloader):
        """
        Compute Fisher Information Matrix (diagonal approximation).

        Called after training on a task to identify important parameters
        for that task. Used by EWC to prevent forgetting.
        """
        self.model.eval()
        fisher_dict = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

        num_samples = 0
        for batch in dataloader:
            if num_samples >= self.fisher_samples:
                break

            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            self.model.zero_grad()
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_dict[n] += p.grad.data.pow(2)

            num_samples += batch["labels"].shape[0]

        # Normalize by number of samples
        for n in fisher_dict:
            fisher_dict[n] /= num_samples

        self._fisher_dict = fisher_dict
        self._optimal_params = {n: p.clone() for n, p in self.model.named_parameters() if p.requires_grad}

        self.model.train()
        return fisher_dict

    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        if self._fisher_dict is None or self._optimal_params is None:
            return torch.tensor(0.0)

        ewc_loss = 0.0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self._fisher_dict:
                ewc_loss += (self._fisher_dict[n] * (p - self._optimal_params[n]).pow(2)).sum()

        return self.ewc_lambda * ewc_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override to track per-sample losses and mix in replay."""
        # Standard forward pass
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Store sample info for replay buffer (during training only)
        if self.model.training and self.replay_buffer is not None:
            with torch.no_grad():
                # Compute TRUE per-sample losses (SuRe-style)
                batch_size = inputs["labels"].shape[0]
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
                labels = inputs["labels"]

                # Per-sample loss computation
                if logits.shape[-1] == 1:  # Regression
                    per_sample_losses = torch.nn.functional.mse_loss(
                        logits.squeeze(-1), labels.float(), reduction='none'
                    )
                else:  # Classification
                    per_sample_losses = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none'
                    )

                # Add top-k highest loss samples from this batch
                k = min(3, batch_size)
                top_losses, top_indices = torch.topk(per_sample_losses.view(-1), k)

                for idx, sample_loss in zip(top_indices.tolist(), top_losses.tolist()):
                    self._sample_losses.append((
                        inputs["ref_input_ids"][idx],
                        inputs["ref_attention_mask"][idx],
                        inputs["alt_input_ids"][idx],
                        inputs["alt_attention_mask"][idx],
                        inputs["labels"][idx],
                        sample_loss,
                    ))

                # Periodically flush to buffer
                if len(self._sample_losses) >= 20:
                    self._sample_losses.sort(key=lambda x: x[-1], reverse=True)
                    for sample in self._sample_losses[:5]:  # Keep top 5
                        self.replay_buffer.add(*sample[:-1], task_id=self.task_id, loss=sample[-1])
                    self._sample_losses = []

        # Mix in replay samples if available
        if self.model.training and self.replay_buffer and self.replay_buffer.total_size > 0:
            replay_batch_size = max(1, int(inputs["labels"].shape[0] * self.replay_ratio))
            replay_samples = self.replay_buffer.sample(replay_batch_size)

            if replay_samples:
                replay_batch = collate_replay_samples(replay_samples, device=loss.device)

                # Forward pass on replay samples
                replay_outputs = model(**replay_batch)
                replay_loss = replay_outputs["loss"] if isinstance(replay_outputs, dict) else replay_outputs[0]

                # Combined loss (weighted average)
                loss = (1 - self.replay_ratio) * loss + self.replay_ratio * replay_loss

        # Add EWC regularization if enabled (protects important params from previous tasks)
        if self.use_ewc and self._fisher_dict is not None:
            loss = loss + self.ewc_loss()

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# LoRA Configuration for Genomic Models
# =============================================================================

def get_lora_config(
    model_type: str,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> LoraConfig:
    """
    Get LoRA config optimized for different genomic model architectures.

    Args:
        model_type: Model architecture (nt, dnabert2, hyenadna, etc.)
        r: LoRA rank (lower = fewer params, less expressive)
        lora_alpha: Scaling factor (typically 2*r)
        lora_dropout: Dropout for regularization
    """
    # Target modules vary by architecture
    if model_type in ["nt", "dnabert2", "gena-lm"]:
        # BERT-style architectures
        target_modules = ["query", "key", "value", "dense"]
    elif model_type == "hyenadna":
        # Hyena architecture
        target_modules = ["in_proj", "out_proj"]
    elif model_type == "caduceus":
        # Mamba-style
        target_modules = ["in_proj", "out_proj", "x_proj"]
    elif model_type == "gpn-star":
        target_modules = ["query", "key", "value", "dense"]
    else:
        # Default: common attention modules
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

def run_lora_finetune(
    task: str,
    seed: int,
    model_type: str = "nt",
    decoder: bool = False,
    test_only: bool = False,
    learning_rate: float = 1e-4,  # Higher LR for LoRA
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
    # Replay options
    use_replay: bool = True,
    replay_buffer_path: Optional[str] = None,
    replay_buffer_size: int = 1000,
    replay_ratio: float = 0.2,
    # EWC options (ACM Survey: combine replay + regularization for best results)
    use_ewc: bool = False,
    ewc_lambda: float = 0.5,
    ewc_fisher_path: Optional[str] = None,
    # SLAO options (orthogonal init + asymmetric merging)
    use_slao: bool = False,
    task_number: int = 1,  # Current task number (1-indexed)
    slao_alpha_A: float = 0.5,  # Preservation factor for A matrices
    slao_merge_frequency: int = 100,  # Merge with previous weights every N steps
    # MAVES filters
    filter_genes=None,
    experimental_methods=None,
    region_type="all",
    variant_types=None,
    seq_length_range=None,
    max_samples_per_experiment=None,
    normalize_scores: bool = False,
    # Other options
    comparison_mode: str = "delta",
):
    """
    Run LoRA fine-tuning with optional replay buffer.

    This is the 2025 SOTA approach for continual learning:
    - LoRA for parameter-efficient adaptation
    - Surprise-based replay to prevent forgetting
    """
    set_seed(seed)
    accelerator = Accelerator()

    path_prefix = "./root/models"
    results_file = f"{path_prefix}/test_results_lora.csv"

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
        # Decoder-only models (can use SDFT)
        "omni-dna": ("zehui127/Omni-DNA-116M", AutoModel),
        "omni-dna-1b": ("zehui127/Omni-DNA-1B", AutoModel),
    }

    # Track if model is decoder-only (supports SDFT)
    decoder_only_models = {"omni-dna", "omni-dna-1b"}

    if model_type not in model_paths:
        raise ValueError(f"Unsupported model: {model_type}")

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
    # Apply LoRA to Base Model
    # =========================================================================
    # Track previous LoRA state for SLAO merging
    prev_lora_state = None

    if use_lora:
        lora_config = get_lora_config(
            model_type, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        base_model = get_peft_model(base_model, lora_config)

        # Load previous LoRA checkpoint for continual learning
        if lora_checkpoint and os.path.exists(lora_checkpoint):
            if use_slao and task_number > 1:
                # SLAO: Orthogonal initialization + store previous weights for merging
                accelerator.print(f"SLAO: Loading LoRA checkpoint with orthogonal init: {lora_checkpoint}")
                prev_lora_state = apply_slao_initialization(
                    base_model,
                    lora_checkpoint,
                    task_number=task_number,
                    use_orthogonal_init=True,
                    accelerator=accelerator,
                )
            else:
                # Standard loading
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
    output_path = f"{path_prefix}/model_{model_type}_{task}{lora_suffix}{replay_suffix}"

    task_type = "regression" if task.startswith("MAVES") else "classification"

    training_args = TrainingArguments(
        output_dir=output_path,
        run_name=f"{model_type}_{task}{lora_suffix}",
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        save_total_limit=5,
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

    # SLAO merging callback (asymmetric A/B treatment with time-aware scaling)
    if use_slao and prev_lora_state is not None:
        slao_callback = SLAOMergingCallback(
            prev_lora_state=prev_lora_state,
            task_number=task_number,
            merge_frequency=slao_merge_frequency,
            alpha_A=slao_alpha_A,
        )
        callbacks.append(slao_callback)
        accelerator.print(f"SLAO: Enabled asymmetric merging (task {task_number}, α_A={slao_alpha_A})")

    if use_replay or use_ewc:
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
            use_ewc=use_ewc,
            ewc_lambda=ewc_lambda,
        )

        # Load Fisher information from previous task for EWC
        if use_ewc and ewc_fisher_path and os.path.exists(ewc_fisher_path):
            fisher_data = torch.load(ewc_fisher_path)
            trainer._fisher_dict = fisher_data["fisher"]
            trainer._optimal_params = fisher_data["optimal_params"]
            accelerator.print(f"Loaded Fisher information from: {ewc_fisher_path}")
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

        # Save LoRA adapter separately
        if use_lora:
            lora_save_path = f"{output_path}/lora_adapter"
            base_model.save_pretrained(lora_save_path)
            accelerator.print(f"Saved LoRA adapter to: {lora_save_path}")

        # Save replay buffer
        if replay_buffer:
            buffer_save_path = f"{path_prefix}/replay_buffer_{task}.pt"
            replay_buffer.save(buffer_save_path)

        # Compute and save Fisher information for EWC (for next task)
        if use_ewc:
            from torch.utils.data import DataLoader
            train_dataloader = DataLoader(
                datasets["train"], batch_size=batch_size, collate_fn=data_collator
            )
            trainer.compute_fisher_information(train_dataloader)
            fisher_save_path = f"{path_prefix}/fisher_{task}.pt"
            torch.save({
                "fisher": trainer._fisher_dict,
                "optimal_params": trainer._optimal_params,
            }, fisher_save_path)
            accelerator.print(f"Saved Fisher information to: {fisher_save_path}")

    # =========================================================================
    # Evaluation
    # =========================================================================
    test_dataset = datasets.get(f"{task}_test")
    if test_dataset:
        test_metrics = trainer.evaluate(eval_dataset=test_dataset)
        accelerator.print(f"Test Metrics for {task}: {test_metrics}")

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
                        choices=["nt", "dnabert2", "hyenadna", "caduceus", "gena-lm", "gpn-star"])
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

    # Replay options
    parser.add_argument("--use_replay", action="store_true", help="Enable replay buffer")
    parser.add_argument("--replay_buffer", type=str, default=None,
                        help="Path to existing replay buffer")
    parser.add_argument("--replay_buffer_size", type=int, default=1000)
    parser.add_argument("--replay_ratio", type=float, default=0.2,
                        help="Fraction of batch from replay")

    # EWC options (ACM Survey: combine replay + regularization)
    parser.add_argument("--use_ewc", action="store_true",
                        help="Enable EWC regularization (complements replay)")
    parser.add_argument("--ewc_lambda", type=float, default=0.5,
                        help="EWC regularization strength")
    parser.add_argument("--ewc_fisher", type=str, default=None,
                        help="Path to Fisher information from previous task")

    # SLAO options (orthogonal init + asymmetric merging)
    parser.add_argument("--use_slao", action="store_true",
                        help="Enable SLAO: orthogonal init + asymmetric A/B merging")
    parser.add_argument("--task_number", type=int, default=1,
                        help="Current task number for time-aware scaling (1-indexed)")
    parser.add_argument("--slao_alpha_A", type=float, default=0.5,
                        help="SLAO: preservation factor for A matrices (higher = more preserved)")
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

    run_lora_finetune(
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
        use_ewc=args.use_ewc,
        ewc_lambda=args.ewc_lambda,
        ewc_fisher_path=args.ewc_fisher,
        use_slao=args.use_slao,
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
