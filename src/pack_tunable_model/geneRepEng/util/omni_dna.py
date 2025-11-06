"""
Utility functions specific to Omni-DNA model architecture
"""

import torch
from typing import List, Dict, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer


def get_omni_dna_layer_list(model: PreTrainedModel) -> List[torch.nn.Module]:
    """
    Get the list of transformer layers from Omni-DNA model

    Args:
        model: Omni-DNA model

    Returns:
        List of transformer layers
    """
    # Handle OLMoForSequenceCLS model structure
    if hasattr(model, 'model') and hasattr(model.model, 'transformer') and hasattr(model.model.transformer, 'blocks'):
        return model.model.transformer.blocks
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
        return model.transformer.blocks
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
        return model.transformer.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        return model.encoder.layer
    else:
        # Try to find layers/blocks in the model
        for name, module in model.named_modules():
            if ('layers' in name.lower() or 'blocks' in name.lower()) and isinstance(module, torch.nn.ModuleList):
                print(f"Found potential layers at: {name}")
                return module

        # Print model structure for debugging
        print("Model structure:")
        for name, module in model.named_children():
            print(f"  {name}: {type(module)}")
            if hasattr(module, '__dict__'):
                for child_name, child_module in module.named_children():
                    print(f"    {child_name}: {type(child_module)}")

        raise ValueError(f"Could not find transformer layers in Omni-DNA model: {type(model)}")


def get_omni_dna_splice_fn(model: PreTrainedModel):
    """
    Get the function to splice into Omni-DNA model layers

    Args:
        model: Omni-DNA model

    Returns:
        Function to get the target module for wrapping
    """
    model_type = getattr(model.config, 'model_type', '')

    # For OLMo models, wrap the entire layer block
    if 'olmo' in model_type.lower() or 'OLMoForSequenceCLS' in str(type(model)):
        def splice_fn(layer):
            # For OLMo models, wrap the entire layer
            return layer
        return splice_fn

    # For other models, we typically want to wrap the attention or MLP components
    def splice_fn(layer):
        # Try different common layer structures
        # Check if attribute is a module (not a method)
        if hasattr(layer, 'self_attn') and isinstance(getattr(layer, 'self_attn'), torch.nn.Module):
            return layer.self_attn
        elif hasattr(layer, 'attn') and isinstance(getattr(layer, 'attn'), torch.nn.Module):
            return layer.attn
        elif hasattr(layer, 'mlp') and isinstance(getattr(layer, 'mlp'), torch.nn.Module):
            return layer.mlp
        elif hasattr(layer, 'feed_forward') and isinstance(getattr(layer, 'feed_forward'), torch.nn.Module):
            return layer.feed_forward
        else:
            # Default to the layer itself
            return layer

    return splice_fn


def preprocess_dna_sequences(
    tokenizer: PreTrainedTokenizer,
    sequences: List[str],
    max_length: Optional[int] = None,
    padding: bool = True,
    truncation: bool = True,
    return_tensors: str = "pt"
) -> Dict[str, torch.Tensor]:
    """
    Preprocess DNA sequences for Omni-DNA model

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

    # Ensure sequences are properly formatted for DNA tokenization
    processed_sequences = []
    for seq in sequences:
        # Convert to uppercase and validate DNA characters
        seq = seq.upper()
        # Replace any invalid characters with 'N'
        valid_chars = set('ATGCN')
        seq = ''.join(c if c in valid_chars else 'N' for c in seq)
        processed_sequences.append(seq)

    # Tokenize
    inputs = tokenizer(
        processed_sequences,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors
    )

    return inputs


def extract_omni_dna_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sequences: List[str],
    layer_ids: Optional[List[int]] = None,
    pooling_strategy: str = "last_token",
    batch_size: int = 8
) -> Dict[int, torch.Tensor]:
    """
    Extract embeddings from specific layers of Omni-DNA model

    Args:
        model: Omni-DNA model
        tokenizer: DNA tokenizer
        sequences: List of DNA sequences
        layer_ids: List of layer indices to extract from
        pooling_strategy: How to pool sequence representations
        batch_size: Batch size for processing

    Returns:
        Dictionary mapping layer_id to embeddings tensor
    """
    if layer_ids is None:
        layers = get_omni_dna_layer_list(model)
        layer_ids = list(range(len(layers)))

    model.eval()
    device = next(model.parameters()).device

    all_embeddings = {layer_id: [] for layer_id in layer_ids}

    # Process in batches
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]

        # Preprocess batch
        inputs = preprocess_dna_sequences(tokenizer, batch_sequences)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            for layer_id in layer_ids:
                layer_hidden = hidden_states[layer_id + 1]  # +1 for input embeddings

                # Apply pooling strategy
                if pooling_strategy == "last_token":
                    # Use last non-padding token
                    attention_mask = inputs.get('attention_mask')
                    if attention_mask is not None:
                        seq_lengths = attention_mask.sum(dim=1)
                        batch_embeddings = []
                        for j, seq_len in enumerate(seq_lengths):
                            batch_embeddings.append(layer_hidden[j, seq_len - 1])
                        batch_embeddings = torch.stack(batch_embeddings)
                    else:
                        batch_embeddings = layer_hidden[:, -1]

                elif pooling_strategy == "mean_pooling":
                    # Mean pooling over sequence
                    attention_mask = inputs.get('attention_mask')
                    if attention_mask is not None:
                        mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden.size()).float()
                        sum_hidden = torch.sum(layer_hidden * mask_expanded, dim=1)
                        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                        batch_embeddings = sum_hidden / sum_mask
                    else:
                        batch_embeddings = layer_hidden.mean(dim=1)

                elif pooling_strategy == "cls_token":
                    # Use CLS token (first token)
                    batch_embeddings = layer_hidden[:, 0]

                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

                all_embeddings[layer_id].append(batch_embeddings.cpu())

    # Concatenate all batches
    final_embeddings = {}
    for layer_id, embeddings_list in all_embeddings.items():
        final_embeddings[layer_id] = torch.cat(embeddings_list, dim=0)

    return final_embeddings


def create_omni_dna_control_model(
    model: PreTrainedModel,
    layer_ids: Optional[List[int]] = None,
    control_strength: float = 1.0
):
    """
    Create a control model wrapper for Omni-DNA

    Args:
        model: Base Omni-DNA model
        layer_ids: List of layer indices to control
        control_strength: Default control strength

    Returns:
        ControlModel instance
    """
    from ..control import ControlModel

    if layer_ids is None:
        layers = get_omni_dna_layer_list(model)
        # Control middle layers by default
        total_layers = len(layers)
        start_layer = total_layers // 4
        end_layer = 3 * total_layers // 4
        layer_ids = list(range(start_layer, end_layer))

    splice_fn = get_omni_dna_splice_fn(model)

    control_model = ControlModel(
        model=model,
        layer_ids=layer_ids,
        splice_fn=splice_fn
    )

    return control_model


def validate_dna_sequence(sequence: str) -> bool:
    """
    Validate that a sequence contains only valid DNA characters

    Args:
        sequence: DNA sequence string

    Returns:
        True if valid, False otherwise
    """
    valid_chars = set('ATGCN')
    return all(c.upper() in valid_chars for c in sequence)


def generate_random_dna_sequence(length: int, seed: Optional[int] = None) -> str:
    """
    Generate a random DNA sequence for testing

    Args:
        length: Length of sequence to generate
        seed: Random seed for reproducibility

    Returns:
        Random DNA sequence string
    """
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
    Introduce random mutations into a DNA sequence

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
