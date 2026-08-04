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


def pi_classification_confidence(y_true: int, probs: np.ndarray) -> float:
    """
    Likelihood combining correctness with model confidence.
    Trained samples often have higher confidence.
    """
    confidence = np.max(probs)
    correctness = probs[y_true]
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    # Higher score = more likely training sample
    return correctness * confidence * np.exp(-entropy)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


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

    @torch.no_grad()
    def get_embeddings(self, batch: Dict) -> np.ndarray:
        """
        Extract model embeddings before the classification head.

        Args:
            batch: Dictionary with batched input tensors

        Returns:
            Embeddings as numpy array of shape [batch_size, hidden_dim]
        """
        self.model.eval()

        # Try to get embeddings from the model
        # Different model architectures may have different ways to access embeddings
        try:
            # For WrappedModelWithClassificationHead
            if hasattr(self.model, 'base_model'):
                ref_outputs = self.model.base_model(
                    input_ids=batch['ref_input_ids'].to(self.device),
                    attention_mask=batch['ref_attention_mask'].to(self.device),
                    output_hidden_states=True,
                )
                alt_outputs = self.model.base_model(
                    input_ids=batch['alt_input_ids'].to(self.device),
                    attention_mask=batch['alt_attention_mask'].to(self.device),
                    output_hidden_states=True,
                )

                # Get last hidden state and pool (mean over sequence)
                if hasattr(ref_outputs, 'last_hidden_state'):
                    ref_embed = ref_outputs.last_hidden_state.mean(dim=1)
                    alt_embed = alt_outputs.last_hidden_state.mean(dim=1)
                elif hasattr(ref_outputs, 'hidden_states'):
                    ref_embed = ref_outputs.hidden_states[-1].mean(dim=1)
                    alt_embed = alt_outputs.hidden_states[-1].mean(dim=1)
                else:
                    # Fallback: use logits as pseudo-embeddings
                    outputs = self.model(
                        ref_input_ids=batch['ref_input_ids'].to(self.device),
                        ref_attention_mask=batch['ref_attention_mask'].to(self.device),
                        alt_input_ids=batch['alt_input_ids'].to(self.device),
                        alt_attention_mask=batch['alt_attention_mask'].to(self.device),
                    )
                    return outputs['logits'].cpu().numpy()

                # Concatenate ref and alt embeddings
                embeddings = torch.cat([ref_embed, alt_embed], dim=-1)
                return embeddings.cpu().numpy()
            else:
                # Fallback: use logits as pseudo-embeddings
                outputs = self.model(
                    ref_input_ids=batch['ref_input_ids'].to(self.device),
                    ref_attention_mask=batch['ref_attention_mask'].to(self.device),
                    alt_input_ids=batch['alt_input_ids'].to(self.device),
                    alt_attention_mask=batch['alt_attention_mask'].to(self.device),
                )
                return outputs['logits'].cpu().numpy()
        except Exception as e:
            print(f"Warning: Could not extract embeddings: {e}. Using logits as fallback.")
            outputs = self.model(
                ref_input_ids=batch['ref_input_ids'].to(self.device),
                ref_attention_mask=batch['ref_attention_mask'].to(self.device),
                alt_input_ids=batch['alt_input_ids'].to(self.device),
                alt_attention_mask=batch['alt_attention_mask'].to(self.device),
            )
            return outputs['logits'].cpu().numpy()

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

    def attack_embedding(self, idx: int, reference_embeddings: np.ndarray) -> Tuple[Optional[int], Dict, Optional[int]]:
        """
        Run the embedding-based attribute inference attack (analogous to attack_A_pi).

        Uses model's internal representations (embeddings from last hidden state)
        to exploit the model's learned feature space directly. For each candidate
        in the feasible set, compute similarity to training sample embeddings.
        Predict the candidate with highest score.

        Analogous to likelihood-based attack:
        - Likelihood: score = P(y_obs | f(candidate)) - how well candidate explains observed label
        - Embedding: score = sim(embed(candidate), train_embeddings) - how "trained" candidate looks

        Args:
            idx: Index of the target sample
            reference_embeddings: Pre-computed embeddings from training samples.
                                  Shape: [N_train, hidden_dim].

        Returns:
            Tuple of (predicted_idx, scores_dict, true_idx)
        """
        # Step 1: build feasible set
        candidates = self.build_feasible_set(idx)

        if len(candidates['indices']) < 2:
            return None, {}, None

        # Step 2: get model embeddings for all candidates
        embeddings = self.get_embeddings(candidates)

        # Step 3: compute embedding-based scores for each candidate
        scores = {}

        for i, cand_idx in enumerate(candidates['indices']):
            # Compute mean cosine similarity to training sample embeddings
            # Higher similarity indicates the candidate's embedding pattern
            # is more "familiar" to the model (likely seen during training)
            similarities = [
                cosine_similarity(embeddings[i], ref_emb)
                for ref_emb in reference_embeddings
            ]
            scores[cand_idx] = np.mean(similarities)

        # Step 4: predict candidate with highest score
        predicted_idx = max(scores, key=scores.get)
        true_idx = np.array(candidates['indices'])[np.array(candidates['curr'])][0]

        return predicted_idx, dict(scores), true_idx

    def attack_embedding_top_n(
        self, idx: int, n: int, reference_embeddings: np.ndarray
    ) -> Tuple[Optional[int], Dict, Optional[int], List[int]]:
        """
        Run embedding-based attack and return top-N predictions (analogous to attack_A_pi_top_n).

        Args:
            idx: Index of target sample
            n: Number of top predictions to return
            reference_embeddings: Pre-computed embeddings from training samples.
                                  Shape: [N_train, hidden_dim].

        Returns:
            Tuple of (predicted_idx, scores_dict, true_idx, top_n_predictions)
        """
        predicted_idx, scores, true_idx = self.attack_embedding(
            idx, reference_embeddings=reference_embeddings
        )

        if predicted_idx is None:
            return None, {}, None, []

        top_n = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:n]

        return predicted_idx, scores, true_idx, top_n

    def attack_dataset(
        self,
        sample_indices: List[int],
        df=None,
        verbose: bool = True,
        use_embedding: bool = False,
        reference_embeddings: np.ndarray = None
    ) -> Dict:
        """
        Run attack on dataset (analogous to DMS eval_only attack loop).

        Reports accuracy by group_size (analogous to num_snps in DMS).
        Also computes granular metrics: same-label vs cross-label accuracy.

        Args:
            sample_indices: List of sample indices to attack
            df: Optional dataframe with metadata
            verbose: Print progress
            use_embedding: Use embedding-based attack instead of likelihood-based
            reference_embeddings: Pre-computed embeddings from training samples.
                For embedding-based attack, compares candidate embeddings to these
                reference embeddings. Higher similarity indicates memorization.

        Returns:
            Dictionary with attack results including granular metrics
        """
        all_predicted = defaultdict(list)
        all_true = defaultdict(list)
        all_top_n_predicted = defaultdict(list)
        all_scores = []

        # Granular metrics tracking
        same_label_correct = 0
        same_label_total = 0
        cross_label_correct = 0
        cross_label_total = 0
        same_label_group_sizes = []
        cross_label_group_sizes = []

        for i, idx in enumerate(sample_indices):
            if verbose and (i + 1) % 100 == 0:
                print(f"Attacking sample {i + 1}/{len(sample_indices)}")

            # Get group info for this sample
            group_id = self.full_dataset.group_ids[idx]
            group_size = len(self.group_to_samples[group_id])
            target_label = self.full_dataset.labels[idx]

            # Check if all candidates in group have same label (same-label scenario)
            candidate_indices = self.group_to_samples[group_id]
            candidate_labels = [self.full_dataset.labels[c] for c in candidate_indices]
            same_label_candidates = sum(1 for l in candidate_labels if l == target_label)
            is_same_label_group = (same_label_candidates == len(candidate_labels))

            # Run attack
            if use_embedding:
                predicted, scores, true_idx, top_n = self.attack_embedding_top_n(
                    idx, n=5, reference_embeddings=reference_embeddings
                )
            else:
                predicted, scores, true_idx, top_n = self.attack_A_pi_top_n(idx, n=5)

            if predicted is None:
                continue

            all_predicted[group_size].append(predicted)
            all_true[group_size].append(true_idx)
            all_top_n_predicted[group_size].append(top_n)
            all_scores.append(scores)

            # Track granular metrics
            is_correct = (predicted == true_idx)
            if is_same_label_group:
                same_label_total += 1
                same_label_group_sizes.append(group_size)
                if is_correct:
                    same_label_correct += 1
            else:
                cross_label_total += 1
                cross_label_group_sizes.append(group_size)
                if is_correct:
                    cross_label_correct += 1

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

        # Compute granular metrics
        same_label_accuracy = same_label_correct / same_label_total if same_label_total > 0 else 0
        cross_label_accuracy = cross_label_correct / cross_label_total if cross_label_total > 0 else 0

        # Random baselines for granular metrics
        same_label_random = np.mean([1.0 / s for s in same_label_group_sizes]) if same_label_group_sizes else 0
        cross_label_random = np.mean([1.0 / s for s in cross_label_group_sizes]) if cross_label_group_sizes else 0

        return {
            'overall_accuracy': overall_accuracy,
            'random_baseline': random_baseline,
            'advantage': overall_accuracy - random_baseline,
            'by_group_size': results_by_size,
            'num_samples': total_samples,
            'total_samples': total_samples,  # Keep for backwards compatibility
            'all_scores': all_scores,
            # Granular metrics
            'granular': {
                'same_label_accuracy': same_label_accuracy,
                'same_label_random': same_label_random,
                'same_label_advantage': same_label_accuracy - same_label_random,
                'same_label_total': same_label_total,
                'cross_label_accuracy': cross_label_accuracy,
                'cross_label_random': cross_label_random,
                'cross_label_advantage': cross_label_accuracy - cross_label_random,
                'cross_label_total': cross_label_total,
            },
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
              f"n={results['num_samples']}")

        # Print granular metrics if available
        if 'granular' in results:
            g = results['granular']
            print(f"\n{'-'*70}")
            print("Granular Metrics (Same-Label vs Cross-Label):")
            print(f"{'-'*70}")
            print(f"  Same-label accuracy:  {g['same_label_accuracy']:.4f} "
                  f"(random={g['same_label_random']:.4f}, "
                  f"advantage={g['same_label_advantage']:.4f}) "
                  f"n={g['same_label_total']}")
            print(f"  Cross-label accuracy: {g['cross_label_accuracy']:.4f} "
                  f"(random={g['cross_label_random']:.4f}, "
                  f"advantage={g['cross_label_advantage']:.4f}) "
                  f"n={g['cross_label_total']}")

            # Interpretation
            if g['same_label_total'] > 0 and g['cross_label_total'] > 0:
                if g['same_label_advantage'] < 0.05 and g['cross_label_advantage'] > 0.1:
                    print("\n  [!] Attack success primarily from distinguishing different labels,")
                    print("      not from identifying specific variants with same label.")
                elif g['same_label_advantage'] > 0.05:
                    print("\n  [!] Attack shows real variant-level distinguishing ability")
                    print("      (success even within same-label groups).")

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
    verbose: bool = True,
    use_embedding: bool = False,
    reference_embeddings: np.ndarray = None
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
        use_embedding: Use embedding-based attack instead of likelihood-based
        reference_embeddings: Pre-computed training sample embeddings for
            embedding-based attack. If None and use_embedding=True, will
            compute embeddings from training samples.

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

    attack_type = "embedding-based" if use_embedding else "likelihood-based"
    if verbose:
        print(f"\nRunning {attack_type} attribute inference attack...")

    # For embedding-based attack, compute reference embeddings from training samples
    # if not provided. These serve as the "fingerprint" of what trained embeddings look like.
    if use_embedding and reference_embeddings is None:
        if verbose:
            print("Computing reference embeddings from training samples...")
        train_indices = [i for i, m in enumerate(train_mask) if m]
        # Sample subset of training data for efficiency (use up to 1000 samples)
        sample_size = min(1000, len(train_indices))
        sampled_indices = np.random.choice(train_indices, size=sample_size, replace=False)

        # Compute embeddings for sampled training samples
        ref_embeddings_list = []
        for idx in sampled_indices:
            batch = attack.build_feasible_set(idx)
            # Only need the target sample's embedding
            target_batch_idx = batch['curr'].index(True)
            embeddings = attack.get_embeddings(batch)
            ref_embeddings_list.append(embeddings[target_batch_idx])

        reference_embeddings = np.array(ref_embeddings_list)
        if verbose:
            print(f"Computed {len(reference_embeddings)} reference embeddings")

    results = {}

    # Attack training set
    train_indices = [i for i, m in enumerate(train_mask) if m]
    if verbose:
        print(f"\nAttacking {len(train_indices)} training samples...")
    results['train'] = attack.attack_dataset(
        train_indices, verbose=verbose, use_embedding=use_embedding,
        reference_embeddings=reference_embeddings
    )
    attack.print_results(results['train'], "training")

    # Attack validation set
    val_indices = [i for i, m in enumerate(train_mask) if not m]
    if verbose:
        print(f"\nAttacking {len(val_indices)} validation samples...")
    results['val'] = attack.attack_dataset(
        val_indices, verbose=verbose, use_embedding=use_embedding,
        reference_embeddings=reference_embeddings
    )
    attack.print_results(results['val'], "validation")

    # Attack full set
    all_indices = list(range(len(full_dataset)))
    if verbose:
        print(f"\nAttacking {len(all_indices)} total samples...")
    results['full'] = attack.attack_dataset(
        all_indices, verbose=verbose, use_embedding=use_embedding,
        reference_embeddings=reference_embeddings
    )
    attack.print_results(results['full'], "full")

    return results
