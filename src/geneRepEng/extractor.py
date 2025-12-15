"""
Utility functions for extracting representations from genomic models
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer
import tqdm

def extract_layer_representations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sequences: List[str],
    layer_ids: List[int],
    batch_size: int = 8,
    max_length: Optional[int] = None,
    device: Optional[torch.device] = None,
    position: str = "last"  # "last", "mean", "cls"
) -> Dict[int, np.ndarray]:
    """
    Extract hidden representations from specific layers of a genomic model

    Args:
        model: The genomic language model
        tokenizer: Tokenizer for DNA sequences
        sequences: List of DNA sequences
        layer_ids: List of layer indices to extract from
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        device: Device to run on
        position: How to aggregate sequence representations

    Returns:
        Dictionary mapping layer_id to representations array
    """
    if device is None:
        device = next(model.parameters()).device

    if max_length is None:
        max_length = getattr(tokenizer, 'model_max_length', 1024)

    model.eval()
    representations = {layer_id: [] for layer_id in layer_ids}

    # Process sequences in batches
    for i in tqdm.tqdm(range(0, len(sequences), batch_size), desc="Extracting representations"):
        batch_sequences = sequences[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        # Move to device and filter out unsupported keys (e.g., token_type_ids for OLMo)
        supported_keys = ['input_ids', 'attention_mask']
        inputs = {k: v.to(device) for k, v in inputs.items() if k in supported_keys}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Handle hidden_states - could be tuple, list, or None
            if hidden_states is None:
                raise ValueError("Model did not return hidden states. Make sure output_hidden_states=True is supported.")

            # Convert to list if needed
            if isinstance(hidden_states, tuple):
                hidden_states = list(hidden_states)

            num_hidden_states = len(hidden_states)

            # Extract representations for each requested layer
            for layer_id in layer_ids:
                # Map layer_id to actual hidden_states index
                # hidden_states typically includes embeddings at index 0, then layer outputs
                # But some models may have different structures
                hs_index = layer_id + 1
                if hs_index >= num_hidden_states:
                    hs_index = num_hidden_states - 1  # Use last available layer
                layer_hidden = hidden_states[hs_index]

                # Check if hidden states are already pooled (2D) or full (3D)
                if layer_hidden.dim() == 2:
                    # Already pooled: [batch_size, hidden_size]
                    batch_representations = layer_hidden.cpu().numpy()
                elif layer_hidden.dim() == 3:
                    # Full hidden states: [batch_size, seq_len, hidden_size]
                    # Aggregate representations based on position
                    if position == "last":
                        # Use the last non-padding token
                        attention_mask = inputs.get('attention_mask')
                        if attention_mask is not None:
                            seq_lengths = attention_mask.sum(dim=1)
                            batch_representations = []
                            for j, seq_len in enumerate(seq_lengths):
                                batch_representations.append(layer_hidden[j, int(seq_len.item()) - 1].cpu().numpy())
                            batch_representations = np.array(batch_representations)
                        else:
                            batch_representations = layer_hidden[:, -1].cpu().numpy()
                    elif position == "mean":
                        # Mean pooling over sequence length
                        attention_mask = inputs.get('attention_mask')
                        if attention_mask is not None:
                            mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden.size()).float()
                            sum_hidden = torch.sum(layer_hidden * mask_expanded, dim=1)
                            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                            batch_representations = (sum_hidden / sum_mask).cpu().numpy()
                        else:
                            batch_representations = layer_hidden.mean(dim=1).cpu().numpy()
                    elif position == "cls":
                        # Use CLS token (first token)
                        batch_representations = layer_hidden[:, 0].cpu().numpy()
                    else:
                        raise ValueError(f"Unknown position: {position}")
                else:
                    raise ValueError(f"Unexpected hidden state dimensions: {layer_hidden.dim()}")

                representations[layer_id].extend(batch_representations)

    # Convert lists to numpy arrays
    return {k: np.array(v) for k, v in representations.items()}

def compute_activation_differences(
    ref_representations: Dict[int, np.ndarray],
    alt_representations: Dict[int, np.ndarray]
) -> Dict[int, np.ndarray]:
    """
    Compute activation differences between reference and alternative sequences

    Args:
        ref_representations: Representations from reference sequences
        alt_representations: Representations from alternative sequences

    Returns:
        Dictionary of activation differences for each layer
    """
    differences = {}

    for layer_id in ref_representations.keys():
        if layer_id in alt_representations:
            ref_reps = ref_representations[layer_id]
            alt_reps = alt_representations[layer_id]

            # Ensure same number of samples
            min_samples = min(ref_reps.shape[0], alt_reps.shape[0])
            ref_reps = ref_reps[:min_samples]
            alt_reps = alt_reps[:min_samples]

            differences[layer_id] = alt_reps - ref_reps

    return differences

def find_principal_directions(
    activation_differences: Dict[int, np.ndarray],
    n_components: int = 1
) -> Dict[int, np.ndarray]:
    """
    Find principal directions of variation using PCA

    Args:
        activation_differences: Dictionary of activation differences
        n_components: Number of principal components to extract

    Returns:
        Dictionary of principal directions for each layer
    """
    from sklearn.decomposition import PCA

    directions = {}

    for layer_id, diffs in activation_differences.items():
        if diffs.shape[0] > n_components:
            pca = PCA(n_components=n_components)
            pca.fit(diffs)

            if n_components == 1:
                directions[layer_id] = pca.components_[0]
            else:
                directions[layer_id] = pca.components_
        else:
            # If we have fewer samples than components, just use the mean
            directions[layer_id] = np.mean(diffs, axis=0)
            if directions[layer_id].ndim == 1:
                directions[layer_id] = directions[layer_id] / np.linalg.norm(directions[layer_id])

    return directions

def create_control_vector_from_sequences(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    ref_sequences: List[str],
    alt_sequences: List[str],
    layer_ids: Optional[List[int]] = None,
    batch_size: int = 8,
    method: str = "pca_diff",
    **kwargs
) -> Dict[int, np.ndarray]:
    """
    Create control vectors from reference and alternative genomic sequences

    Args:
        model: The genomic language model
        tokenizer: Tokenizer for DNA sequences
        ref_sequences: List of reference DNA sequences
        alt_sequences: List of alternative DNA sequences
        layer_ids: List of layer indices (default: all layers)
        batch_size: Batch size for processing
        method: Method for computing control vectors

    Returns:
        Dictionary of control vectors for each layer
    """
    if layer_ids is None:
        # Extract from all layers
        from .control import model_layer_list
        layers = model_layer_list(model)
        layer_ids = list(range(len(layers)))

    # Extract representations
    ref_reps = extract_layer_representations(
        model, tokenizer, ref_sequences, layer_ids, batch_size, **kwargs
    )
    alt_reps = extract_layer_representations(
        model, tokenizer, alt_sequences, layer_ids, batch_size, **kwargs
    )

    # Compute differences
    differences = compute_activation_differences(ref_reps, alt_reps)

    # Find control directions
    if method == "pca_diff":
        directions = find_principal_directions(differences, n_components=1)
    elif method == "mean_diff":
        directions = {k: np.mean(v, axis=0) for k, v in differences.items()}
        directions = {k: v / np.linalg.norm(v) for k, v in directions.items()}
    else:
        raise ValueError(f"Unknown method: {method}")

    return directions
