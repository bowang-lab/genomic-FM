"""
Genomic Language Model Attribute Inference Attack for ClinVar Grouped Data
===========================================================================
Adapted from Emmy's DMS attack (guess_kolter_one_codon.py) for classification tasks.

Given a model and a sample, infer which variant within the group it is.
Uses likelihood-based scoring similar to the DMS attack.
"""

from __future__ import annotations
import pickle
import argparse
import os
from pathlib import Path
from typing import Literal, Optional, Dict, List, Tuple
from scipy.stats import norm
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.pack_tunable_model.hf_dataloader import return_clinvar_grouped_lira_dataset


# ---------------------------------------------------------------------------
# Likelihood functions (adapted from DMS attack)
# ---------------------------------------------------------------------------

def pi_regression(y_true: float, y_pred: float, sigma: float) -> float:
    """
    Likelihood for regression tasks (DMS-style).
    Uses normal distribution centered at prediction.
    """
    residual = y_true - y_pred
    return norm.pdf(residual, loc=0, scale=sigma)


def pi_classification(y_true: int, probs: np.ndarray) -> float:
    """
    Likelihood for classification tasks.
    Returns softmax probability at true class.
    """
    return probs[y_true]


def marginal_weight(candidate: dict) -> float:
    """
    Calculate the prior weight for a candidate.
    Currently uniform prior (all candidates equally likely a priori).
    """
    return 1.0


# ---------------------------------------------------------------------------
# ClinVar Attribute Inference Attack
# ---------------------------------------------------------------------------

class ClinVarAttributeInference:
    """
    Attribute inference attack on grouped ClinVar data.

    Given a model and a sample, infer which variant within the group it is.
    Adapted from Emmy's DMS attack for classification tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        full_dataset,
        group_to_id: Dict[str, int],
        device: str = 'cuda'
    ):
        """
        Initialize the attribute inference attack.

        Args:
            model: Trained classification model
            tokenizer: Tokenizer used for the model
            full_dataset: ClinVarGroupedDataset containing all variants
            group_to_id: Mapping from group names to group IDs
            device: Device to run inference on
        """
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.full_dataset = full_dataset
        self.group_to_id = group_to_id
        self.id_to_group = {v: k for k, v in group_to_id.items()}
        self.device = device

        # Move model to device
        self.model = self.model.to(device)

        # Index variants by group for fast lookup
        self._build_group_index()

    def _build_group_index(self):
        """Map group_id -> list of sample indices in that group."""
        self.group_to_samples = defaultdict(list)
        for idx in range(len(self.full_dataset)):
            group_id = self.full_dataset.group_ids[idx]
            self.group_to_samples[group_id].append(idx)

    def build_feasible_set(self, idx: int) -> Dict:
        """
        Build feasible set for a target sample (analogous to DMS build_feasible_set).

        Returns all variants in the same group as the target.

        Args:
            idx: Index of the target sample

        Returns:
            Dictionary with batched tensors and metadata for all candidates
        """
        group_id = self.full_dataset.group_ids[idx]
        candidate_indices = self.group_to_samples[group_id]

        # Build batch of all candidates in the group
        batch = {
            'ref_input_ids': torch.stack([self.full_dataset.ref_input_ids[i] for i in candidate_indices]),
            'alt_input_ids': torch.stack([self.full_dataset.alt_input_ids[i] for i in candidate_indices]),
            'ref_attention_mask': torch.stack([self.full_dataset.ref_attention_mask[i] for i in candidate_indices]),
            'alt_attention_mask': torch.stack([self.full_dataset.alt_attention_mask[i] for i in candidate_indices]),
            'labels': torch.tensor([self.full_dataset.labels[i] for i in candidate_indices]),
            'indices': candidate_indices,
            'curr': [i == idx for i in candidate_indices],  # Boolean mask for target
            'group_id': group_id,
            'group_name': self.id_to_group.get(group_id, f"group_{group_id}"),
        }
        return batch

    @torch.no_grad()
    def model_output(self, batch: Dict) -> torch.Tensor:
        """
        Run model forward pass (analogous to DMS model_output).

        Args:
            batch: Dictionary with batched input tensors

        Returns:
            Model predictions (logits or probabilities)
        """
        self.model.eval()
        outputs = self.model(
            ref_input_ids=batch['ref_input_ids'].to(self.device),
            ref_attention_mask=batch['ref_attention_mask'].to(self.device),
            alt_input_ids=batch['alt_input_ids'].to(self.device),
            alt_attention_mask=batch['alt_attention_mask'].to(self.device),
        )
        return outputs['logits']

    def attack_A_pi(self, idx: int) -> Tuple[Optional[int], Dict, Optional[int]]:
        """
        Run the A_pi attribute inference attack (analogous to DMS attack_A_pi).

        For each candidate in the feasible set, compute likelihood-weighted score.
        Predict the candidate with highest score.

        Args:
            idx: Index of the target sample

        Returns:
            Tuple of (predicted_idx, scores_dict, true_idx)
        """
        # Step 1: build feasible set
        candidates = self.build_feasible_set(idx)

        if len(candidates['indices']) < 2:
            return None, {}, None

        # Step 2: get model predictions
        logits = self.model_output(candidates)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Step 3: compute likelihood-weighted scores for each candidate
        scores = defaultdict(float)
        true_label = self.full_dataset.labels[idx]

        for i, cand_idx in enumerate(candidates['indices']):
            # pi(y_obs, f(x)) - model fit likelihood (classification version)
            likelihood = pi_classification(true_label, probs[i])

            # p(x) - prior weight
            prior = marginal_weight({'idx': cand_idx})

            # Combined weight
            w = likelihood * prior
            scores[cand_idx] = w

        # Step 4: predict candidate with highest score
        predicted_idx = max(scores, key=scores.get)
        true_idx = np.array(candidates['indices'])[np.array(candidates['curr'])][0]

        return predicted_idx, dict(scores), true_idx

    def attack_A_pi_top_n(self, idx: int, n: int = 5) -> Tuple[List[int], Dict, int, List[int]]:
        """
        Run attack and return top-N predictions (analogous to DMS attack_A_pi_list_scale).

        Args:
            idx: Index of target sample
            n: Number of top predictions to return

        Returns:
            Tuple of (predicted_idx, scores_dict, true_idx, top_n_predictions)
        """
        predicted_idx, scores, true_idx = self.attack_A_pi(idx)

        if predicted_idx is None:
            return None, {}, None, []

        top_n = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:n]

        return predicted_idx, scores, true_idx, top_n

    def attack_dataset(
        self,
        sample_indices: List[int],
        df=None,
        verbose: bool = True
    ) -> Dict:
        """
        Run attack on dataset (analogous to DMS eval_only attack loop).

        Reports accuracy by group_size (analogous to num_snps in DMS).

        Args:
            sample_indices: List of sample indices to attack
            df: Optional dataframe with metadata
            verbose: Print progress

        Returns:
            Dictionary with attack results
        """
        all_predicted = defaultdict(list)
        all_true = defaultdict(list)
        all_top_n_predicted = defaultdict(list)
        all_scores = []

        for i, idx in enumerate(sample_indices):
            if verbose and (i + 1) % 100 == 0:
                print(f"Attacking sample {i + 1}/{len(sample_indices)}")

            # Get group size for this sample
            group_id = self.full_dataset.group_ids[idx]
            group_size = len(self.group_to_samples[group_id])

            # Run attack
            predicted, scores, true_idx, top_n = self.attack_A_pi_top_n(idx, n=5)

            if predicted is None:
                continue

            all_predicted[group_size].append(predicted)
            all_true[group_size].append(true_idx)
            all_top_n_predicted[group_size].append(top_n)
            all_scores.append(scores)

        # Compute metrics by group_size (analogous to num_snps)
        results_by_size = {}
        total_correct = 0
        total_samples = 0

        for group_size in sorted(all_predicted.keys()):
            predicted = all_predicted[group_size]
            true = all_true[group_size]

            correct = sum(p == t for p, t in zip(predicted, true))
            total = len(predicted)
            accuracy = correct / total if total > 0 else 0
            random_baseline = 1.0 / group_size

            # Top-N accuracy (for groups with size > 1)
            top_n_accuracy = 0.0
            if group_size > 1:
                top_n_correct = sum(
                    t in top_n_preds for t, top_n_preds in zip(true, all_top_n_predicted[group_size])
                )
                top_n_accuracy = top_n_correct / total if total > 0 else 0

            results_by_size[group_size] = {
                'accuracy': accuracy,
                'random_baseline': random_baseline,
                'advantage': accuracy - random_baseline,
                'top_5_accuracy': top_n_accuracy,
                'correct': correct,
                'total': total,
            }

            total_correct += correct
            total_samples += total

        # Overall metrics
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

        # Weighted random baseline
        random_baseline = 0.0
        for group_size, stats in results_by_size.items():
            weight = stats['total'] / total_samples if total_samples > 0 else 0
            random_baseline += weight * stats['random_baseline']

        return {
            'overall_accuracy': overall_accuracy,
            'random_baseline': random_baseline,
            'advantage': overall_accuracy - random_baseline,
            'by_group_size': results_by_size,
            'total_samples': total_samples,
            'all_scores': all_scores,
        }

    def print_results(self, results: Dict, set_name: str = ""):
        """
        Print results in DMS-style format.

        Args:
            results: Results dictionary from attack_dataset
            set_name: Name of the dataset (e.g., "training", "validation")
        """
        print(f"\n{'='*70}")
        print(f"Attribute Inference Attack Results{' - ' + set_name if set_name else ''}")
        print(f"{'='*70}")

        for group_size in sorted(results['by_group_size'].keys()):
            stats = results['by_group_size'][group_size]
            print(
                f"accuracy on {set_name} set | {group_size} variants/group: "
                f"{stats['accuracy']:.4f} (random={stats['random_baseline']:.4f}, "
                f"advantage={stats['advantage']:.4f}) | "
                f"total data points: {stats['total']}"
            )
            if group_size > 1 and stats['top_5_accuracy'] > 0:
                print(f"  top 5 accuracy: {stats['top_5_accuracy']:.4f}")

        print(f"\nOverall: accuracy={results['overall_accuracy']:.4f}, "
              f"random={results['random_baseline']:.4f}, "
              f"advantage={results['advantage']:.4f}, "
              f"n={results['total_samples']}")
        print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Standalone functions for compatibility
# ---------------------------------------------------------------------------

def run_attack(
    model: nn.Module,
    tokenizer,
    full_dataset,
    group_to_id: Dict[str, int],
    train_mask: np.ndarray,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict:
    """
    Run full attack pipeline (training set, validation set, full set).

    Analogous to the eval_only block in DMS attack.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        full_dataset: Full ClinVar grouped dataset
        group_to_id: Group name to ID mapping
        train_mask: Boolean mask indicating training samples
        device: Device to run on
        verbose: Print progress

    Returns:
        Dictionary with results for train, val, and full sets
    """
    attack = ClinVarAttributeInference(
        model=model,
        tokenizer=tokenizer,
        full_dataset=full_dataset,
        group_to_id=group_to_id,
        device=device
    )

    results = {}

    # Attack training set
    train_indices = [i for i, m in enumerate(train_mask) if m]
    if verbose:
        print(f"\nAttacking {len(train_indices)} training samples...")
    results['train'] = attack.attack_dataset(train_indices, verbose=verbose)
    attack.print_results(results['train'], "training")

    # Attack validation set
    val_indices = [i for i, m in enumerate(train_mask) if not m]
    if verbose:
        print(f"\nAttacking {len(val_indices)} validation samples...")
    results['val'] = attack.attack_dataset(val_indices, verbose=verbose)
    attack.print_results(results['val'], "validation")

    # Attack full set
    all_indices = list(range(len(full_dataset)))
    if verbose:
        print(f"\nAttacking {len(all_indices)} total samples...")
    results['full'] = attack.attack_dataset(all_indices, verbose=verbose)
    attack.print_results(results['full'], "full")

    return results
