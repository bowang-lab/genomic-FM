"""
Utility functions for DNA sequence handling and model creation.

Note: Layer list and splice functions are in control.py (model_layer_list, get_splice_fn).
      Embedding extraction is in extract.py (extract_layer_representations).
"""

import torch
from typing import List, Dict, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer


def create_omni_dna_control_model(
    model: PreTrainedModel,
    layer_ids: Optional[List[int]] = None,
    control_strength: float = 1.0
):
    """
    Create a control model wrapper for genomic models.

    Args:
        model: Base genomic model
        layer_ids: List of layer indices to control (default: middle 50% of layers)
        control_strength: Default control strength

    Returns:
        ControlModel instance
    """
    from ..cv_loader import ControlModel, model_layer_list, get_splice_fn

    if layer_ids is None:
        layers = model_layer_list(model)
        total_layers = len(layers)
        start_layer = total_layers // 4
        end_layer = 3 * total_layers // 4
        layer_ids = list(range(start_layer, end_layer))

    splice_fn = get_splice_fn(model)

    return ControlModel(
        model=model,
        layer_ids=layer_ids,
        splice_fn=splice_fn
    )


def preprocess_dna_sequences(
    tokenizer: PreTrainedTokenizer,
    sequences: List[str],
    max_length: Optional[int] = None,
    padding: bool = True,
    truncation: bool = True,
    return_tensors: str = "pt"
) -> Dict[str, torch.Tensor]:
    """
    Preprocess DNA sequences for genomic models.

    Args:
        tokenizer: DNA tokenizer
        sequences: List of DNA sequences
        max_length: Maximum sequence length
        padding: Whether to pad sequences
        truncation: Whether to truncate sequences
        return_tensors: Format of returned tensors

    Returns:
        Dictionary of tokenized inputs
    """
    if max_length is None:
        max_length = getattr(tokenizer, 'model_max_length', 1024)

    processed_sequences = []
    for seq in sequences:
        seq = seq.upper()
        valid_chars = set('ATGCN')
        seq = ''.join(c if c in valid_chars else 'N' for c in seq)
        processed_sequences.append(seq)

    return tokenizer(
        processed_sequences,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors
    )


def validate_dna_sequence(sequence: str) -> bool:
    """Check if sequence contains only valid DNA characters (ATGCN)."""
    valid_chars = set('ATGCN')
    return all(c.upper() in valid_chars for c in sequence)


def generate_random_dna_sequence(length: int, seed: Optional[int] = None) -> str:
    """Generate a random DNA sequence for testing."""
    import random

    if seed is not None:
        random.seed(seed)

    nucleotides = ['A', 'T', 'G', 'C']
    return ''.join(random.choices(nucleotides, k=length))


def mutate_dna_sequence(
    sequence: str,
    mutation_rate: float = 0.01,
    seed: Optional[int] = None
) -> str:
    """
    Introduce random mutations into a DNA sequence.

    Args:
        sequence: Original DNA sequence
        mutation_rate: Fraction of positions to mutate
        seed: Random seed for reproducibility

    Returns:
        Mutated DNA sequence
    """
    import random

    if seed is not None:
        random.seed(seed)

    nucleotides = ['A', 'T', 'G', 'C']
    sequence = sequence.upper()
    mutated_seq = list(sequence)

    n_mutations = max(1, int(len(sequence) * mutation_rate))
    mutation_positions = random.sample(range(len(sequence)), n_mutations)

    for pos in mutation_positions:
        current_nuc = mutated_seq[pos]
        if current_nuc in nucleotides:
            available_nucs = [n for n in nucleotides if n != current_nuc]
            mutated_seq[pos] = random.choice(available_nucs)

    return ''.join(mutated_seq)
