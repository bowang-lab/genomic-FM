"""
Genomic Language Model Fine-tuning for Variant Effect Prediction
=================================================================
Uses AutoModelForSequenceClassification (num_labels=1) directly,
compatible with models like gena-lm, OmniDNA, Nucleotide Transformer, etc.
that expose a built-in regression/classification head.

Supports three training modes:
  - "head_only"  : freeze backbone, train only the built-in head
  - "peft"       : LoRA adapters on backbone + head (requires `peft`)
  - "full"       : unfreeze everything, end-to-end fine-tuning

"""

# Common head module names across genomic / protein LMs.
# Extend this list if your model uses a different name.
# ESM header: self.classifier
# OMNI-DNA: self.score
# GENA-LM: self.classifier (self.dropout)
# DNABERT2: self.classifier (self.dropout)

# LORA:
# ESM: target_modules=["query", "value"],
# OMNI: target_modules=["att_proj", "attn_out", "ff_proj", "ff_out"],
# GENA target_modules=["query", "value"],
# DNABERT2 target_modules=["query", "value"],

from __future__ import annotations
import pickle
import argparse
import os
import random
from pathlib import Path
from typing import Literal, Optional
from scipy.stats import norm
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, AutoConfig
from collections import defaultdict
import itertools


# Optional — only needed for PEFT mode
try:
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False



def generate_b(a, x, rng=None):
    """
    Generate b such that:
      Pr(a=0, b=0) = x
      Pr(a=1, b=1) = x
      Pr(a=1, b=0) = 0.5 - x
      Pr(a=0, b=1) = 0.5 - x

    Assumes Pr(a=0) = Pr(a=1) = 0.5, and x in (0, 0.5).
    """
    assert 0 < x < 0.5, "x must be in (0, 0.5)"
    if rng is None:
        rng = np.random.default_rng()

    p_match = 2 * x  # Pr(b == a)

    # Draw uniform samples, then b matches a with probability p_match
    u = rng.uniform(size=a.shape)
    b = np.where(u < p_match, a, 1 - a)
    return b


def generate_keep_for_lira(dataset_size, pkeep=0.5, expid=None, num_experiments=None, seed=0, df=None, args=None):
    if not args.split_by_group:
        if not args.dependency_ratio:
            np.random.seed(seed)
            if num_experiments is not None and expid < num_experiments:
                keep = np.random.uniform(0, 1, size=(num_experiments, dataset_size))
                order = keep.argsort(0)
                keep = order < int(pkeep * num_experiments)
                keep = np.array(keep[expid], dtype=bool)
            else:
                keep = np.random.uniform(0, 1, size=dataset_size) <= pkeep
        else:
            pass
            rng = np.random.seed(seed)
            if num_experiments is not None and expid < num_experiments:
                keep = np.random.uniform(0, 1, size=(num_experiments, dataset_size))
                order = keep.argsort(0)
                keep = order < int(pkeep * num_experiments)
                keep = np.array(keep[expid], dtype=bool)
            else:
                keep = np.random.uniform(0, 1, size=dataset_size) <= pkeep
            a = keep[:len(keep)//2]
            b = generate_b(a, args.dependency_ratio, rng=rng)
            dependent_keep = np.zeros_like(keep)

            dependent_keep[:len(a)] = a
            dependent_keep[len(a):len(a)+len(b)] = b
            keep = dependent_keep


    else:
        np.random.seed(seed)
        if num_experiments is not None and expid < num_experiments:
            keep = np.random.uniform(0, 1, size=(num_experiments, len(df['id_mut'].unique())))
            order = keep.argsort(0)
            keep = order < int(pkeep * num_experiments)
            keep = np.array(keep[expid], dtype=bool)
        else:
            keep = np.random.uniform(0, 1, size=len(df['id_mut'].unique())) <= pkeep

        unique_ids = df["id_mut"].unique()
        bool_map = pd.Series(keep, index=unique_ids)

        # 2. Map this Series back to the original column
        df["keep"] = df["id_mut"].map(bool_map)
        keep = df["keep"].to_numpy(dtype=bool)
    return keep


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

class VariantDataset(Dataset):
    """Tokenises genomic sequences on-the-fly and returns (input_ids,
    attention_mask, label) tuples."""

    def __init__(
            self,
            sequences: list[str],
            labels: np.ndarray,
            tokenizer,
            references,
            max_length: int = 512,
            df=None
    ):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reference = references if isinstance(references, list) else [references] * len(sequences)
        self.df = df

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sequences[idx].strip("\n"),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        enc_ref = self.tokenizer(
            self.reference[idx].strip("\n"),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return_res = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": self.labels[idx],
            "input_ids_ref": enc_ref["input_ids"].squeeze(0),
            "attention_mask_ref": enc_ref["attention_mask"].squeeze(0)
        }
        # if self.df is not None:
        #     return_res["start"] = self.df[idx]["start"]
        #     return_res["end"] = self.df[idx]["end"]
        #     return_res["mut_list"] = self.df[idx]["mut_list"]
        #     return_res["ref"] = self.df[idx]["ref"]
        #     return_res["mut"] = self.df[idx]["mut"]
        return return_res


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------

class GenomicRegressionModel(nn.Module):
    """Wraps any HuggingFace encoder with a regression head.

    The [CLS] token representation (position 0) is used as the sequence
    embedding. If your model uses a different pooling strategy (mean pooling,
    last token, etc.) adjust `_pool` accordingly.
    """

    def __init__(
            self,
            backbone: nn.Module,
            hidden_size: int,
            dropout: float = 0.1,
            pooling: Literal["cls", "mean"] = "cls",
            use_delta=False,
            upsample_embedding=False
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.use_delta = use_delta
        self.upsample_embedding = upsample_embedding
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]
        # Mean pooling — ignores padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def upsample(self, emb, target_length):
        # repeat the emb for k*times such that it has the same length as initial input
        batch_size, seq_len, embedding_dim = emb.shape
        # Calculate the repeat factor
        repeat_factor = target_length // seq_len
        remainder = target_length % seq_len
        # Repeat the embeddings along the sequence dimension
        repeated_emb = emb.repeat_interleave(repeat_factor, dim=1)
        # Handle any remainder by slicing and appending
        if remainder > 0:
            extra_emb = emb[:, :remainder, :]
            repeated_emb = torch.cat([repeated_emb, extra_emb], dim=1)
        return repeated_emb

    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        emb = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.upsample_embedding:
            emb = self.upsample(emb, input_ids.size(1))
        return emb

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, input_ids_ref=None,
                attention_mask_ref=None) -> torch.Tensor:
        last_hidden_state = self.get_embeddings(input_ids=input_ids, attention_mask=attention_mask)
        if input_ids_ref is not None and self.use_delta:
            last_hidden_state_ref = self.get_embeddings(input_ids=input_ids_ref, attention_mask=attention_mask_ref)
            try:
                last_hidden_state = last_hidden_state - last_hidden_state_ref
            except:
                breakpoint()
        # upsampled_outputs = self.upsample(outputs.last_hidden_state, input_ids.size(1))
        # upsampled_outputs = upsampled_outputs.squeeze(1)
        pooled = self._pool(last_hidden_state, attention_mask)
        return self.head(pooled).squeeze(-1)  # shape: (batch,)


# ---------------------------------------------------------------------------
# 3. Data loading & preprocessing
# ---------------------------------------------------------------------------

class ScoreNormaliser:
    """
    Fit on training data only; apply the same transform to test data.

    standard : z-score  – good default for unbounded scores
    minmax   : [0, 1]   – use when scores are naturally bounded
    robust   : median / IQR – more resistant to extreme outliers
    log1p    : log(1+x) then z-score – for right-skewed, non-negative scores
    none     : pass through
    """

    def __init__(self, mode="standard"):
        assert mode in ("standard", "minmax", "robust", "log1p", "none")
        self.mode = mode
        self._params: dict = {}

    def fit(self, values: np.ndarray):
        v = values.astype(float)
        if self.mode == "standard":
            self._params = {"mean": v.mean(), "std": v.std() + 1e-8}
        elif self.mode == "minmax":
            self._params = {"min": v.min(), "range": (v.max() - v.min()) + 1e-8}
        elif self.mode == "robust":
            q25, q75 = np.percentile(v, [25, 75])
            self._params = {"median": np.median(v), "iqr": (q75 - q25) + 1e-8}
        elif self.mode == "log1p":
            v_log = np.log1p(v)
            self._params = {"mean": v_log.mean(), "std": v_log.std() + 1e-8}
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        v = values.astype(float)
        if self.mode == "standard":
            return (v - self._params["mean"]) / self._params["std"]
        elif self.mode == "minmax":
            return (v - self._params["min"]) / self._params["range"]
        elif self.mode == "robust":
            return (v - self._params["median"]) / self._params["iqr"]
        elif self.mode == "log1p":
            return (np.log1p(v) - self._params["mean"]) / self._params["std"]
        return v

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        v = values.astype(float)
        if self.mode == "standard":
            return v * self._params["std"] + self._params["mean"]
        elif self.mode == "minmax":
            return v * self._params["range"] + self._params["min"]
        elif self.mode == "robust":
            return v * self._params["iqr"] + self._params["median"]
        elif self.mode == "log1p":
            return np.expm1(v * self._params["std"] + self._params["mean"])
        return v


def load_and_split(
        csv_path: str,
        sequence_col: str,
        score_col: str,
        norm_method = "standard",
        args=None
):
    """
    Returns:
        train_seqs, val_seqs  : lists of raw DNA strings
        train_labels, val_labels : float32 numpy arrays (normalised)
        scaler                : fitted scaler (keep for inverse_transform at inference)
    """
    num_datapoints = args.num_datapoints_to_include
    df = pd.read_csv(csv_path)[:num_datapoints]

    with open(args.ref_file, 'r') as file:
        ref_seq = file.read()  # Read the entire file content into a single string
    # print(ref_seq)

    def process_mutation(row, ref_string):
        # 1. Extract id_mut: split by '_', take second part, strip first/last chars, convert to int
        id_mut = int(row['id'].split('_')[1][1:-1])
        # 2. Calculate coordinates

        # 3. Extract sequences
        mut_indices = []
        mut_seq = row['mutated_sequence_dna']
        for e, (i, j) in enumerate(zip(mut_seq, ref_seq)):
            if i != j: mut_indices.append(e)
        start = mut_indices[0]
        end = mut_indices[-1] + 1
        ref = ref_string[start:end]
        mut = row['mutated_sequence_dna'][start:end]
        mut_list = []
        template = ""
        num_snps = 0
        for r, m in zip(ref, mut):
            if r != m:
                template += "{}"
                num_snps += 1
            else:
                template += r
        for letter in list(itertools.product(["A", "T", "C", "G"], repeat=num_snps)):
            if args.include_ref_genotype:
                mut_list.append(template.format(*letter))
            else:
                if template.format(*letter) != ref:
                    mut_list.append(template.format(*letter))

        # Return as a Series so it maps to new columns
        return pd.Series([id_mut, start, end, ref, mut, mut_list, num_snps])

    # Apply the function to the dataframe
    df[['id_mut', 'start', 'end', 'ref', 'mut', "mut_list", "num_snps"]] = df.apply(
        process_mutation,
        axis=1,
        ref_string=ref_seq
    )

    if not os.path.exists(args.mask_path):
        train_mask = generate_keep_for_lira(len(df), expid=args.expid, num_experiments=args.num_experiments, seed=args.seed, df=df, args=args)
        np.save(args.mask_path, train_mask)
        np.save(args.group_id_path, df["id_mut"].tolist())
    else:
        train_mask = np.load(args.mask_path).astype(bool)

    train_mask = train_mask[:num_datapoints]
    assert sequence_col in df.columns, f"Column '{sequence_col}' not found in CSV."
    assert score_col in df.columns, f"Column '{score_col}' not found in CSV."
    assert len(train_mask) == len(df), (
        f"train_mask length {len(train_mask)} != dataframe length {len(df)}."
    )

    train_mask = np.asarray(train_mask, dtype=bool)
    val_mask = ~train_mask

    train_scores = np.array(list(df.loc[train_mask, score_col]), )
    val_scores = np.array(list(df.loc[val_mask, score_col]))
    scaler = ScoreNormaliser(mode=norm_method)
    scaler.fit(train_scores)

    if scaler is not None:
        train_labels = scaler.transform(train_scores)
        val_labels = scaler.transform(val_scores)
        all_labels = scaler.transform(df[score_col])
    else:
        train_labels = train_scores
        val_labels = val_scores
        all_labels = df[score_col].tolist()

    train_seqs = df.loc[train_mask, sequence_col].tolist()
    val_seqs = df.loc[val_mask, sequence_col].tolist()
    all_seqs = df[sequence_col].tolist()
    df["labels"] = all_labels

    return train_seqs, val_seqs, all_seqs, train_labels, val_labels, all_labels, scaler, ref_seq, df, train_mask


# ---------------------------------------------------------------------------
# 4. Build model for each training mode
# ---------------------------------------------------------------------------

def build_model(
        model_name_or_path: str,
        mode: Literal["head_only", "peft", "full"],
        dropout: float = 0.1,
        pooling: Literal["cls", "mean"] = "cls",
        # LoRA hyperparameters (peft mode only)
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[list[str]] = None,  # e.g. ["query", "value"]
        args=None,
) -> tuple[GenomicRegressionModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
    hidden_size = backbone.config.hidden_size

    # ---- mode: head_only ----
    if mode == "head_only":
        for param in backbone.parameters():
            param.requires_grad = False

    # ---- mode: peft (LoRA) ----
    elif mode == "peft":
        if not PEFT_AVAILABLE:
            raise ImportError("Install the `peft` library: pip install peft")

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            # If None, peft will try to auto-detect attention projection layers.
            # For most transformer backbones use ["query", "value"] or
            # ["q_proj", "v_proj"]. Check your model's named_modules() output.
            target_modules=lora_target_modules,
            bias="none",
        )
        backbone = get_peft_model(backbone, lora_config)
        backbone.print_trainable_parameters()

    # ---- mode: full ----
    elif mode == "full":
        pass  # all parameters already trainable by default

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose head_only | peft | full.")
    try:
        args.max_len = config.max_position_embeddings
    except AttributeError:
        args.max_len = config.max_sequence_length
    model = GenomicRegressionModel(
        backbone=backbone,
        hidden_size=hidden_size,
        dropout=dropout,
        pooling=pooling,
        use_delta=args.use_delta,
        upsample_embedding=args.upsample_embedding
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# 5. Evaluation helpers
# ---------------------------------------------------------------------------

def get_inputs(batch, device, full=False):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    try:
        attention_mask_ref = batch["attention_mask_ref"].to(device)
    except KeyError:
        attention_mask_ref = None
    try:
        input_ids_ref = batch["input_ids_ref"].to(device)
    except KeyError:
        input_ids_ref, attention_mask_ref = None, None
    if not full:
        return input_ids, attention_mask, labels, input_ids_ref, attention_mask_ref
    else:
        start = batch["start"]
        end = batch["end"]
        mut_list = batch["mut_list"]
        ref = batch["ref"]
        mut = batch["mut"]
        return input_ids, attention_mask, labels, input_ids_ref, attention_mask_ref, start, end, mut_list, ref, mut


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    loss_fn = nn.MSELoss()
    # len_loader = len(loader)
    for step, batch in enumerate(loader, 1):
        # print(f"eval | step {step}/{len_loader}")
        input_ids, attention_mask, labels, input_ids_ref, attention_mask_ref = get_inputs(batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            preds = model(input_ids, attention_mask, input_ids_ref, attention_mask_ref)
            loss = loss_fn(preds, labels)

        total_loss += loss.item() * len(labels)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mse = total_loss / len(all_labels)
    pearson_r, _ = pearsonr(all_labels, all_preds)
    spearman_r, _ = spearmanr(all_labels, all_preds)

    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "all_labels": all_labels,
        "all_preds": all_preds
    }


@torch.no_grad()
def model_output(model: nn.Module, batch, device: torch.device, use_amp):
    """
    Run the linear model forward.
    Returns predicted weekly warfarin dose in mg/week.
    """
    model.eval()
    input_ids, attention_mask, labels, input_ids_ref, attention_mask_ref = get_inputs(batch, device)
    with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
        preds = model(input_ids, attention_mask, input_ids_ref, attention_mask_ref)
    return preds

def pi(y_true, y_pred, sigma):
    residual = y_true - y_pred
    return norm.pdf(residual, loc=0, scale=sigma)


def build_feasible_set(df, idx, args, tokenizer,ref_seq=None):
    row = df.loc[idx]
    start, end, mut_list, ref, mut, mutated_sequence_dna, labels = row["start"], row["end"], row["mut_list"], row["ref"], row["mut"], row["mutated_sequence_dna"], row["labels"]
    all_mut_sequences = {}
    all_ref_sequences = []
    labels_list = []
    curr = []

    error_counter = 0

    for mut_codon in mut_list:
        mut_seq = mutated_sequence_dna[:start] + mut_codon + mutated_sequence_dna[end: ]
        ref_seq_new = mutated_sequence_dna[:start] + ref + mutated_sequence_dna[end:]
        if ref_seq is not None:
            ref_seq = ref_seq_new
        if mut_codon in all_mut_sequences:
            error_counter += 1
            all_mut_sequences[f"{mut_codon}_{error_counter}"] = mut_seq.strip("\n")
        else:
            all_mut_sequences[mut_codon] = mut_seq.strip("\n")
        all_ref_sequences.append(ref_seq.strip("\n"))
        labels_list.append(labels)
        curr.append(mut_codon == mut)
    enc = tokenizer(
        list(all_mut_sequences.values()),
        max_length=args.max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    enc_ref = tokenizer(
        all_ref_sequences,
        max_length=args.max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return_res = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": torch.tensor(np.array(labels_list)),
        "input_ids_ref": enc_ref["input_ids"],
        "attention_mask_ref": enc_ref["attention_mask"],
        "curr": curr,
        "mut_list": mut_list
    }
    return return_res


def marginal_weight(candidate):
    """
    calculate the prior
    """
    return 1

def attack_A_pi(df, idx, args, model, tokenizer, device, use_amp=True,sigma=0.2,ref_seq=None,):
    """
    Run the A_pi model inversion attack to infer VKORC1 genotype.

    """
    # Step 1: build feasible set X̂
    candidates = build_feasible_set(df, idx, args, tokenizer,ref_seq)

    # Step 2 & 3: for each VKORC1 value, accumulate weighted score
    scores = defaultdict(float)
    y_pred_list = model_output(model, candidates, device, use_amp).detach().cpu().numpy()
    for i in range(len(candidates["labels"])):
        # Encode the candidate to a feature vector
        y_pred = y_pred_list[i]
        y_true = candidates["labels"][i]

        # pi(y_obs, f(x)) — model fit likelihood
        likelihood = pi(y_true, y_pred, sigma=sigma)

        # p(x) — prior weight (unknown attributes only)
        prior = marginal_weight({k: v[i] for k, v in candidates.items()})

        # Combined weight for this candidate
        w = likelihood * prior
        scores[candidates["mut_list"][i]] += w

    predicted = max(scores, key=scores.get)
    predicted_top_5 = sorted(scores, key=scores.get, reverse=True)[:5]
    # print(f"\nPredicted codon: {predicted} | true codon: {np.array(candidates['mut_list'])[np.array(candidates['curr'])][0]} | predicted top n: {predicted_top_5}")

    return predicted, dict(scores), np.array(candidates['mut_list'])[np.array(candidates['curr'])][0]


def attack_A_pi_list_scale(df, idx, args, model, tokenizer, device, use_amp=True,sigma_list=[0.2],ref_seq=None,):
    """
    Run the A_pi model inversion attack to infer VKORC1 genotype.

    """
    # Step 1: build feasible set X̂
    candidates = build_feasible_set(df, idx, args, tokenizer,ref_seq)

    # Step 2 & 3: for each VKORC1 value, accumulate weighted score
    all_scores = {}
    for sigma in sigma_list:
        scores = defaultdict(float)
        all_scores[sigma] = scores
    y_pred_list = model_output(model, candidates, device, use_amp).detach().cpu().numpy()
    for i in range(len(candidates["labels"])):
        # Encode the candidate to a feature vector
        y_pred = y_pred_list[i]
        y_true = candidates["labels"][i]

        for sigma in sigma_list:
            # pi(y_obs, f(x)) — model fit likelihood
            likelihood = pi(y_true, y_pred, sigma=sigma)

            # p(x) — prior weight (unknown attributes only)
            prior = marginal_weight({k: v[i] for k, v in candidates.items()})

            # Combined weight for this candidate
            w = likelihood * prior
            all_scores[sigma][candidates["mut_list"][i]] += w

    predicted = []
    predicted_top_5 = []
    for sigma in sigma_list:
        scores = all_scores[sigma]
        predicted.append(max(scores, key=scores.get))
        predicted_top_5.append(sorted(scores, key=scores.get, reverse=True)[:5])
    # print(f"\nPredicted codon: {predicted} | true codon: {np.array(candidates['mut_list'])[np.array(candidates['curr'])][0]} | predicted top n: {predicted_top_5}")

    return predicted, dict(all_scores), np.array(candidates['mut_list'])[np.array(candidates['curr'])][0], predicted_top_5
# ---------------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------------

def train(
        backbone_lr_multiplier: float = 0.1,  # lower LR for backbone in full/peft mode
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.06,
        gradient_accumulation_steps: int = 1,
        # ---- model ----
        dropout: float = 0.1,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[list[str]] = None,
        # ---- misc ----
        patience=None,  # early stopping patience
        args=None,
) -> dict:
    """Run the full training loop. Returns a dict with best val metrics."""

    batch_size = args.batch_size
    learning_rate = args.lr
    pooling = args.pooling
    lora_r = args.lora_r
    use_amp = args.use_amp
    output_dir = args.output_dir
    seed = args.seed
    max_grad_norm = args.max_grad_norm

    # --- reproducibility ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.group_id_path = f"{output_dir}/group_id.npy"
    args.mask_path = f"{output_dir}/keep.npy"
    print(f"Using device: {device}")

    # --- data ---
    train_seqs, val_seqs, all_seqs, train_labels, val_labels, all_labels, scaler, ref_seq, df, train_mask = load_and_split(
        args.csv_path, args.sequence_col, args.score_col, args.norm_method, args
    )
    print(f"Train: {len(train_seqs)}  |  Val: {len(val_seqs)}")

    # --- model & tokenizer ---
    model, tokenizer = build_model(
        args.model_name_or_path, args.mode, dropout, pooling,
        lora_r, lora_alpha, lora_dropout, lora_target_modules, args
    )
    model.to(device)

    # --- datasets & loaders ---
    train_ds = VariantDataset(train_seqs, train_labels, tokenizer, ref_seq, args.max_len)
    val_ds = VariantDataset(val_seqs, val_labels, tokenizer, ref_seq, args.max_len)
    full_ds = VariantDataset(all_seqs, all_labels, tokenizer, ref_seq, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size * 16, 1024), shuffle=False,
                            num_workers=4, pin_memory=True)
    full_loader = DataLoader(full_ds, batch_size=min(batch_size * 16, 1024), shuffle=False,
                             num_workers=4, pin_memory=True)

    if args.eval_only:
        model.load_state_dict(
            torch.load(os.path.join(output_dir, f"best_model_{args.mode}.pt"), weights_only=False)["model_state"])

        full_metrics = evaluate(model, full_loader, device, use_amp)
        with open(os.path.join(output_dir, f"best_full_metric_{args.mode}.pkl"),
                  'wb') as f:
            pickle.dump(full_metrics, f)


        try:
            model.load_state_dict(
                torch.load(os.path.join(output_dir, f"final_model_{args.mode}.pt"), weights_only=False)["model_state"])

            full_metrics = evaluate(model, full_loader, device, use_amp)
            with open(os.path.join(output_dir, f"full_metric_{args.mode}.pkl"),
                      'wb') as f:
                pickle.dump(full_metrics, f)
        except FileNotFoundError:
            print("final_model not found")
        exit(0)
        model.eval()
        # train_loader_for_val = DataLoader(train_ds, batch_size=min(batch_size * 16, 1024), shuffle=False,
        #                     num_workers=4, pin_memory=True)
        # scale = []
        #
        # metrics = evaluate(model, train_loader_for_val, device, use_amp)
        # scale.append(metrics["rmse"])
        # print(f"scale: {scale}")
        # metrics = evaluate(model, val_loader, device, use_amp)
        # scale.append(metrics["rmse"])
        # print(f"scale: {scale}")
        # metrics = evaluate(model, full_loader, device, use_amp)
        # scale.append(metrics["rmse"])
        # print(f"scale: {scale}")
        if args.mode == "full":
            scale = [0.19903818679678897, 0.6739997774941628, 0.49651083560236764]
        elif args.mode == "head_only":
            scale = [0.2184214183317421, 0.6812189332491396, 0.5054813629132462]
        all_predicted = defaultdict(list)
        all_true = defaultdict(list)
        all_top_n_predicted = defaultdict(list)
        all_scores = []
        for idx in range(len(df)):
            if train_mask[idx]:
                # print(df.loc[idx, "id"])
                predicted, scores, true_codon, top_n_predicted = attack_A_pi_list_scale(df, idx, args, model, tokenizer, device,
                                                                       use_amp=True,
                                                                       sigma_list=scale, ref_seq=ref_seq)
                all_predicted[df.loc[idx]["num_snps"]].append(predicted)
                all_true[df.loc[idx]["num_snps"]].append(true_codon)
                all_top_n_predicted[df.loc[idx]["num_snps"]].append(top_n_predicted)
                all_scores.append(scores)

        for num_snps, predicted in all_predicted.items():
            accuracy = (np.stack(predicted) == np.array(all_true[num_snps])[:,None]).mean(0)
            top_n_scale_accuracy_list = defaultdict(list)
            print(
                f"accuracy on training set (scale={scale}) | {num_snps} snps: {accuracy} | total data points: {len(predicted)}")
            if num_snps != 1:
                for y, top_n in zip(all_true[num_snps], np.stack(all_top_n_predicted[num_snps])):
                    for e, top_n_scale in enumerate(top_n):
                        top_n_scale_accuracy_list[scale[e]].append(y in top_n_scale)

                for s, top_n_scale in top_n_scale_accuracy_list.items():
                    print(f"top n accuracy on training set (scale={s}) | {num_snps} snps: {np.array(top_n_scale).mean()}")
            print()

        print()
        print()

        # scale = [0.19903822398544255, 0.67]
        all_predicted = defaultdict(list)
        all_true = defaultdict(list)
        all_top_n_predicted = defaultdict(list)
        all_scores = []
        for idx in range(len(df)):
            predicted, scores, true_codon, top_n_predicted = attack_A_pi_list_scale(df, idx, args, model, tokenizer,
                                                                                    device,
                                                                                    use_amp=True,
                                                                                    sigma_list=scale,
                                                                                    ref_seq=ref_seq)
            all_predicted[df.loc[idx]["num_snps"]].append(predicted)
            all_true[df.loc[idx]["num_snps"]].append(true_codon)
            all_top_n_predicted[df.loc[idx]["num_snps"]].append(top_n_predicted)
            all_scores.append(scores)

        for num_snps, predicted in all_predicted.items():
            accuracy = (np.stack(predicted) == np.array(all_true[num_snps])[:, None]).mean(0)
            top_n_scale_accuracy_list = defaultdict(list)
            print(
                f"accuracy on all set (scale={scale}) | {num_snps} snps: {accuracy} | total data points: {len(predicted)}")
            if num_snps != 1:
                for y, top_n in zip(all_true[num_snps], np.stack(all_top_n_predicted[num_snps])):
                    for e, top_n_scale in enumerate(top_n):
                        top_n_scale_accuracy_list[scale[e]].append(y in top_n_scale)

                for s, top_n_scale in top_n_scale_accuracy_list.items():
                    print(
                        f"top n accuracy on all set (scale={s}) | {num_snps} snps: {np.array(top_n_scale).mean()}")
            print()

        print()
        print()


        # scale = [0.19903822398544255, 0.67]
        all_predicted = defaultdict(list)
        all_true = defaultdict(list)
        all_top_n_predicted = defaultdict(list)
        all_scores = []
        for idx in range(len(df)):
            if not train_mask[idx]:
                # print(df.loc[idx, "id"])
                predicted, scores, true_codon, top_n_predicted = attack_A_pi_list_scale(df, idx, args, model, tokenizer, device, use_amp=True,
                                                                       sigma_list=scale, ref_seq=ref_seq)
                all_predicted[df.loc[idx]["num_snps"]].append(predicted)
                all_true[df.loc[idx]["num_snps"]].append(true_codon)
                all_top_n_predicted[df.loc[idx]["num_snps"]].append(top_n_predicted)
                all_scores.append(scores)

        for num_snps, predicted in all_predicted.items():
            accuracy = (np.stack(predicted) == np.array(all_true[num_snps])[:,None]).mean(0)
            top_n_scale_accuracy_list = defaultdict(list)
            print(
                f"accuracy on val set (scale={scale}) | {num_snps} snps: {accuracy} | total data points: {len(predicted)}")
            if num_snps != 1:
                for y, top_n in zip(all_true[num_snps], np.stack(all_top_n_predicted[num_snps])):
                    for e, top_n_scale in enumerate(top_n):
                        top_n_scale_accuracy_list[scale[e]].append(y in top_n_scale)

                for s, top_n_scale in top_n_scale_accuracy_list.items():
                    print(f"top n accuracy on val set (scale={s}) | {num_snps} snps: {np.array(top_n_scale).mean()}")
            print()

        return

    # --- optimiser: separate LR groups for backbone vs head ---
    head_params = list(model.head.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.backbone.parameters()
                       if id(p) not in head_param_ids and p.requires_grad]

    param_groups = [{"params": head_params, "lr": learning_rate}]
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": learning_rate * backbone_lr_multiplier,
        })
    # breakpoint()
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    # --- scheduler ---
    total_steps = (len(train_loader) // gradient_accumulation_steps) * args.epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- loss & AMP scaler ---
    loss_fn = nn.MSELoss()
    amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    # --- training loop ---
    best_val_metrics = None
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        # len_train_loader = len(train_loader)
        for step, batch in enumerate(train_loader, 1):
            # print(f"epoch {epoch} | step {step} / {len_train_loader}")
            input_ids, attention_mask, labels, input_ids_ref, attention_mask_ref = get_inputs(batch, device)

            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                preds = model(input_ids, attention_mask, input_ids_ref, attention_mask_ref)
                loss = loss_fn(preds, labels) / gradient_accumulation_steps

            amp_scaler.scale(loss).backward()

            if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                amp_scaler.unscale_(optimizer)
                if max_grad_norm is not None:
                    # print("clipping norm")
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * gradient_accumulation_steps * len(labels)

        train_mse = epoch_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, device, use_amp)
        val_loss = val_metrics["mse"]

        # full_metrics = evaluate(model, full_loader, device, use_amp)
        # with open(os.path.join(output_dir, f"full_metric_{args.mode}_{epoch}.pkl"),
        #           'wb') as f:
        #     pickle.dump(full_metrics, f)

        print(
            f"Epoch {epoch:>3}/{args.epochs}  |  "
            f"train_mse={train_mse:.4f}  |  "
            f"val_mse={val_loss:.4f}  val_rmse={val_metrics['rmse']:.4f}  "
            f"pearson_r={val_metrics['pearson_r']:.4f}  "
            f"spearman_r={val_metrics['spearman_r']:.4f}"
        )

        # --- checkpoint & early stopping ---
        if val_loss < best_val_loss:
            #
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            no_improve = 0

            ckpt_path = os.path.join(output_dir, f"best_model_{args.mode}.pt")
            if args.save_checkpoint:
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    # "optim_state": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "scaler": (scaler.mode, scaler._params),
                    "norm_method": args.norm_method,
                    "mode":        args.mode,
                }, ckpt_path)
                print(f"  ✓ Checkpoint saved → {ckpt_path}")


        else:
            no_improve += 1
            if patience is not None:
                if no_improve >= patience:
                    print(f"  Early stopping triggered (no improvement for {patience} epochs).")
                    break

    if args.save_checkpoint:
        ckpt_path = os.path.join(output_dir, f"final_model_{args.mode}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            # "optim_state": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "scaler": (scaler.mode, scaler._params),
            "norm_method": args.norm_method,
            "mode": args.mode,
        }, ckpt_path)
        print(f"  ✓ Checkpoint saved → {ckpt_path}")


    model.load_state_dict(torch.load(os.path.join(output_dir, f"best_model_{args.mode}.pt"),weights_only=False)["model_state"])
    full_metrics = evaluate(model, full_loader, device, use_amp)
    with open(os.path.join(output_dir, f"best_full_metric_{args.mode}.pkl"),
              'wb') as f:
        pickle.dump(full_metrics, f)

    model.load_state_dict(
        torch.load(os.path.join(output_dir, f"final_model_{args.mode}.pt"), weights_only=False)["model_state"])
    full_metrics = evaluate(model, full_loader, device, use_amp)
    with open(os.path.join(output_dir, f"full_metric_{args.mode}.pkl"),
              'wb') as f:
        pickle.dump(full_metrics, f)


    # ckpt_path = os.path.join(output_dir, f"model_{args.epochs}.pt")
    # torch.save({
    #     "epoch":       args.epochs,
    #     "model_state": model.state_dict(),
    #     # "optim_state": optimizer.state_dict(),
    #     # "full_metrics": full_metrics,
    #     "scaler": (scaler.mode, scaler._params),
    #     "norm_method": args.norm_method,
    #     "mode":        args.mode,
    # }, ckpt_path)

    print(f"\nBest validation metrics ({args.mode}):")
    for k, v in best_val_metrics.items():
        if k not in ["all_labels", "all_preds"]:
            print(f"  {k}: {v:.4f}")
    return best_val_metrics


# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tasks = {
        "ecoli_protein_dna_dms": {"adkar_2012": "ccdb", "firnberg_2014": "b_lactamase",
                                  "jacquier_2013": "b_lactamase",
                                  "tsuboyama_argr_2023": "argr",
                                  "tsuboyama_feca_2023": "feca",
                                  "tsuboyama_nusa_2023": "nusa",
                                  "tsuboyama_rfah_2023": "rfah",
                                  "tsuboyama_yaia_2023": "yaia",

                                  "weeks_2023": "rnc", "kelsic_2016": "if1", },

        "human_protein_dna_dms": {
            "garvie_2021": "PDE3A",
            "giacomelli_2018": "P53",
            "kotler_2018": "P53",
            "silverstein_2021": "GDI1",
            "sun_2020": "CBS",
        },
        "other_prokaryotic_dna_dms": {"chen_2020": "a4grb6",
                                      "rockah_2015": "haeiii",
                                      },
    }

    tasks_ground_truth_name = {
        "ecoli_protein_dna_dms": {"adkar_2012": ["DMS_score"], "firnberg_2014": ["DMS_score"],
                                  "jacquier_2013": ["DMS_score"],
                                  "tsuboyama_argr_2023": ["DMS_score"],
                                  "tsuboyama_feca_2023": ["DMS_score"],
                                  "tsuboyama_nusa_2023": ["DMS_score"],
                                  "tsuboyama_rfah_2023": ["DMS_score"],
                                  "tsuboyama_yaia_2023": ["DMS_score"],
                                  "weeks_2023": ["DMS_score"], "kelsic_2016": ["DMS_score"], },

        "human_protein_dna_dms": {"garvie_2021": ["DMS_zscore_ratio"],
                                  "giacomelli_2018": ["DMS_WT_Nutlin", "DMS_null_Nutlin", "DMS_null_etoposide"],
                                  "kotler_2018": ["DMS"],
                                  "silverstein_2021": ["DMS"],
                                  "sun_2020": ["DMS_lowB6", "DMS_highB6"], },
        "other_prokaryotic_dna_dms": {"chen_2020": ["DMS"],
                                      "rockah_2015": ["DMS"],
                                      },}
    parser = argparse.ArgumentParser(description="Finetune a genomic LM for variant effect prediction.")
    parser.add_argument("--dataset_name", default="kotler", type=str)
    parser.add_argument("--model_name_or_path", default="InstaDeepAI/nucleotide-transformer-500m-human-ref",
                        help="HuggingFace model name or local path, e.g. InstaDeepAI/nucleotide-transformer-500m-human-ref")
    parser.add_argument("--mode", default="full",
                        choices=["head_only", "peft", "full"])
    parser.add_argument("--norm_method", default="standard",
                        choices=["standard", "minmax", "robust", "log1p", "none"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--global_output_dir", default="./checkpoints")
    parser.add_argument("--pooling", default="mean", choices=["cls", "mean"])
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--use_amp", default=1, choices=[0, 1])
    parser.add_argument("--expid", type=int, default=0)
    parser.add_argument("--num_experiments", type=int, default=16)
    parser.add_argument("--use_delta", type=int, default=0)
    parser.add_argument("--upsample_embedding", type=int, default=0)
    parser.add_argument("--num_datapoints_to_include", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--eval_only", type=int, default=0)
    parser.add_argument("--include_ref_genotype", type=int, default=0)
    parser.add_argument("--save_checkpoint", type=int, default=1)
    parser.add_argument("--split_by_group", type=int, default=0)
    parser.add_argument("--dependency_ratio",
                        default=None,
                        help="dependency ratio must be != 0.25 and between 0 and 0.5")
    args = parser.parse_args()

    tasks_mapping = {
        "kotler": ("variant_effect_human/kotler.csv",
                   "mutated_sequence_dna", "DMS",
                   "variant_effect_human/human_sequences_wt/P53_CDS.txt"
                   ),
        "garvie": ("variant_effect_human/garvie_2021_dms_dna.csv",
                   "mutated_sequence_dna", "DMS_zscore_ratio",
                   "variant_effect_human/human_sequences_wt/PDE3A_CDS.txt"
                   )
    }

    args.csv_path, args.sequence_col, args.score_col, args.ref_file = tasks_mapping[args.dataset_name]

    # parser.add_argument("--csv_path", help="Path to input CSV file",
    #                     default="variant_effect_human/garvie_2021_dms_dna.csv")
    # parser.add_argument("--sequence_col", default="mutated_sequence_dna", help="Name of the sequence column")
    # parser.add_argument("--score_col", help="Name of the score column", default="DMS_zscore_ratio")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir = f"{args.global_output_dir}/{args.dataset_name}/{args.num_datapoints_to_include}_useDelta{args.use_delta}_upsample{args.upsample_embedding}"
    if args.split_by_group:
        args.output_dir += f"_split_by_group"
    if args.dependency_ratio:
        args.output_dir += f"_dependency_ratio{args.dependency_ratio}"
    args.output_dir += f"/exp{args.expid}_{args.num_experiments}"
    print(args)
    train(args = args
    )
    print(args)