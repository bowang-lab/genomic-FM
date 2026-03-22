import dataclasses
import os
import typing
import inspect
import warnings
import numpy as np

np.float_ = np.float64

from sklearn.decomposition import PCA
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import tqdm

from .cv_loader import ControlModel

@dataclasses.dataclass
class GenomicDatasetEntry:
    """Dataset entry for genomic data with reference and alternative sequences"""
    ref_sequence: str  # Reference DNA sequence
    alt_sequence: str  # Alternative DNA sequence
    label: typing.Optional[int] = None  # Optional label for supervised training

@dataclasses.dataclass
class CustomFunctions:
    F_batched_get_hiddens: typing.Optional[typing.Callable] = None
    F_batched_preprocess: typing.Optional[typing.Callable] = None

    def __post_init__(self):
        # Set default functions if None
        if self.F_batched_get_hiddens is None:
            self.F_batched_get_hiddens = batched_get_hiddens
        else:
            # Validate signature
            sig = inspect.signature(self.F_batched_get_hiddens)
            actual = list(sig.parameters)
            expected = ["model", "processors", "inputs", "hidden_layers", "batch_size", "custom_functions", "pooling"]
            if actual != expected:
                raise ValueError(f"F_batched_get_hiddens signature must be {expected!r}, got {actual!r}")

        if self.F_batched_preprocess is None:
            self.F_batched_preprocess = batched_preprocess
        else:
            # Validate signature
            sig = inspect.signature(self.F_batched_preprocess)
            actual = list(sig.parameters)
            expected = ["processors", "inputs", "batch_size", "dtype"]
            if actual != expected:
                raise ValueError(f"F_batched_preprocess signature must be {expected!r}, got {actual!r}")

def batched_get_hiddens(
    model,
    processors,
    inputs: list,
    hidden_layers: list[int],
    batch_size: int,
    custom_functions: CustomFunctions,
    pooling: str = "last",
) -> dict[int, np.ndarray]:
    """
    Using the given model and processors, pass the genomic inputs through the model and get the hidden
    states for each layer in `hidden_layers`.

    Args:
        pooling: Pooling strategy - "last" (last token), "mean" (mean pooling), "cls" (first token)

    Returns a dictionary from `hidden_layers` layer id to an numpy array of shape `(n_inputs, hidden_dim)`
    """
    shape_checked = False

    # Determine which preprocess function to use
    preprocess_func = custom_functions.F_batched_preprocess

    # Preprocess inputs using the determined preprocess function
    processed_inputs = preprocess_func(processors, inputs, batch_size, dtype=model.dtype)

    hidden_states = {layer: [] for layer in hidden_layers}
    with torch.no_grad():
        for processed in tqdm.tqdm(processed_inputs, desc="batched_get_hiddens"):
            # Move processed inputs to the model's device
            for key, value in processed.items():
                if isinstance(value, torch.Tensor):
                    processed[key] = value.to(model.device)

            # Get model output with hidden states
            out = model(**processed, output_hidden_states=True)

            # Ensure the layer indices matches that of the model layers
            if not shape_checked:
                from .cv_loader import model_layer_list
                if len(out.hidden_states) != len(model_layer_list(model)) + 1:
                    warnings.warn(
                        f"Length of hidden_states should equal model_layers+1: {len(out.hidden_states)} != {len(model_layer_list(model))+1}"
                    )
                assert len(out.hidden_states[0].shape) in [2, 3], f"layer hidden state shape should be either (sequence_length, hidden_size) or (batch_size, sequence_length, hidden_size), not {out.hidden_states[0].shape}"
                shape_checked = True

            # Extract hidden states for each requested layer
            for layer in hidden_layers:
                hidden_idx = layer + 1 if layer >= 0 else layer
                layer_output = out.hidden_states[hidden_idx]

                # Apply pooling strategy
                if layer_output.ndim == 2:      # shape: (seq_len, hidden_dim)
                    if pooling == "cls":
                        hidden_state = layer_output[0].cpu().float().numpy()
                    elif pooling == "mean":
                        hidden_state = layer_output.mean(dim=0).cpu().float().numpy()
                    else:  # "last"
                        hidden_state = layer_output[-1].cpu().float().numpy()
                else:                           # shape: (batch_size, seq_len, hidden_dim)
                    if pooling == "cls":
                        hidden_state = layer_output[0, 0].cpu().float().numpy()
                    elif pooling == "mean":
                        # Mean pooling with attention mask if available
                        attention_mask = processed.get('attention_mask')
                        if attention_mask is not None:
                            mask = attention_mask[0].unsqueeze(-1).expand(layer_output[0].size()).float()
                            sum_hidden = torch.sum(layer_output[0] * mask, dim=0)
                            sum_mask = torch.clamp(mask.sum(dim=0), min=1e-9)
                            hidden_state = (sum_hidden / sum_mask).cpu().float().numpy()
                        else:
                            hidden_state = layer_output[0].mean(dim=0).cpu().float().numpy()
                    else:  # "last"
                        hidden_state = layer_output[0, -1].cpu().float().numpy()
                hidden_states[layer].append(hidden_state)

            del out

    return {k: np.vstack(v) for k, v in hidden_states.items()}

def batched_preprocess(processors, inputs: list, batch_size: int, dtype=None):
    """ Default function to preprocess a list of genomic inputs in batches """
    assert len(processors) == 1, "by default processors length must be 1, use custom F_batched_preprocess to change"
    assert len(inputs) > 0, "inputs must be a non-empty list"
    processor = processors[0]  # This should be the tokenizer for DNA sequences

    batched_inputs = [
        inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)
    ]

    input_dicts = []
    for batch in tqdm.tqdm(batched_inputs, desc="batched_preprocess"):
        for input_entry in batch:
            # For genomic data, we expect input_entry to have 'sequence' key
            if isinstance(input_entry, dict) and 'sequence' in input_entry:
                sequence = input_entry['sequence']
            elif isinstance(input_entry, GenomicDatasetEntry):
                # For control vector training, we might use either ref or alt sequence
                # This will be determined by the calling code
                sequence = getattr(input_entry, 'sequence', input_entry.ref_sequence)
            else:
                sequence = str(input_entry)

            # Tokenize the DNA sequence
            processed = processor(
                sequence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=getattr(processor, 'model_max_length', 1024)
            )

            # Keep batch dimension - model expects (batch_size, seq_len)
            input_dicts.append(processed)

    return input_dicts

def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    mag = np.linalg.norm(direction)
    assert not np.isinf(mag)
    return (H @ direction) / mag

@dataclasses.dataclass
class ControlVector:
    model_type: str
    directions: dict[int, np.ndarray]

    @classmethod
    def train(
        cls,
        model: "PreTrainedModel | ControlModel",
        processors: list,
        dataset: list[GenomicDatasetEntry],
        custom_functions: typing.Optional[CustomFunctions] = None,
        **kwargs,
    ) -> "ControlVector":
        """
        Train a ControlVector for genomic data using reference and alternative sequences.

        Args:
            model (PreTrainedModel | ControlModel): The Omni-DNA model to train against.
            processors (list): The tokenizers to process the genomic dataset.
            dataset (list[GenomicDatasetEntry]): The genomic dataset with ref/alt sequences.
            custom_functions (CustomFunctions, optional): Custom functions for processing and extracting hidden states.
            **kwargs: Additional keyword arguments.
                max_batch_size (int, optional): The maximum batch size for training.
                    Defaults to 32. Try reducing this if you're running out of memory.
                method (str, optional): The training method to use. Can be either
                    "pca_diff" or "pca_center". Defaults to "pca_diff".
                pooling (str, optional): Pooling strategy - "last" (last token),
                    "mean" (mean pooling), "cls" (first token). Defaults to "last".

        Returns:
            ControlVector: The trained vector for genomic control.
        """
        if isinstance(model, ControlModel):
            model.model.eval()
        else:
            model.eval()

        with torch.inference_mode():
            dirs = read_representations(
                model,
                processors,
                dataset,
                custom_functions=custom_functions,
                **kwargs,
            )
        return cls(model_type=model.config.model_type, directions=dirs)

    def save(self, path: str):
        """Save the control vector to a file"""
        np.savez(path, **{str(k): v for k, v in self.directions.items()}, model_type=self.model_type)

    @classmethod
    def load(cls, path: str) -> "ControlVector":
        """Load a control vector from a file"""
        data = np.load(path, allow_pickle=True)
        model_type = str(data['model_type'])
        directions = {int(k): v for k, v in data.items() if k != 'model_type'}
        return cls(model_type=model_type, directions=directions)

# =============================================================================
# Simplified APIs (raw sequences, no GenomicDatasetEntry required)
# =============================================================================

def extract_layer_representations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sequences: typing.List[str],
    layer_ids: typing.List[int],
    batch_size: int = 8,
    pooling: str = "last",
) -> typing.Dict[int, np.ndarray]:
    """
    Extract hidden representations from specific layers for raw DNA sequences.

    Args:
        model: The genomic language model
        tokenizer: Tokenizer for DNA sequences
        sequences: List of DNA sequences (raw strings)
        layer_ids: List of layer indices to extract from
        batch_size: Batch size for processing
        pooling: Pooling strategy ("last", "mean", "cls")

    Returns:
        Dict mapping layer_id to representations array (n_sequences, hidden_dim)
    """
    custom_functions = CustomFunctions()
    inputs = [{'sequence': seq} for seq in sequences]
    return custom_functions.F_batched_get_hiddens(
        model, [tokenizer], inputs, layer_ids, batch_size, custom_functions, pooling
    )


def create_control_vector_from_sequences(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    ref_sequences: typing.List[str],
    alt_sequences: typing.List[str],
    layer_ids: typing.Optional[typing.List[int]] = None,
    batch_size: int = 8,
    method: str = "pca_diff",
    pooling: str = "last",
) -> typing.Dict[int, np.ndarray]:
    """
    Create control vectors from raw reference and alternative DNA sequences.

    Args:
        model: The genomic language model
        tokenizer: Tokenizer for DNA sequences
        ref_sequences: List of reference DNA sequences
        alt_sequences: List of alternative DNA sequences
        layer_ids: List of layer indices (default: all layers)
        batch_size: Batch size for processing
        method: "pca_diff" or "mean_diff"
        pooling: Pooling strategy ("last", "mean", "cls")

    Returns:
        Dict mapping layer_id to control direction vector
    """
    from .cv_loader import model_layer_list

    if layer_ids is None:
        layers = model_layer_list(model)
        layer_ids = list(range(len(layers)))

    ref_reps = extract_layer_representations(model, tokenizer, ref_sequences, layer_ids, batch_size, pooling)
    alt_reps = extract_layer_representations(model, tokenizer, alt_sequences, layer_ids, batch_size, pooling)

    directions = {}
    for layer_id in layer_ids:
        diff = alt_reps[layer_id] - ref_reps[layer_id]

        if method == "pca_diff":
            if diff.shape[0] > 1:
                pca = PCA(n_components=1)
                pca.fit(diff)
                direction = pca.components_[0]
            else:
                direction = diff[0] / (np.linalg.norm(diff[0]) + 1e-8)
        elif method == "mean_diff":
            direction = np.mean(diff, axis=0)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
        else:
            raise ValueError(f"Unknown method: {method}")

        directions[layer_id] = direction

    return directions


def read_representations(
    model,
    processors,
    dataset: list[GenomicDatasetEntry],
    custom_functions: typing.Optional[CustomFunctions] = None,
    method: str = "pca_diff",
    max_batch_size: int = 32,
    pooling: str = "last",
    **kwargs,
) -> dict[int, np.ndarray]:
    """
    Read hidden representations from genomic sequences and compute control directions.

    For genomic data, we extract representations from both reference and alternative
    sequences and compute the difference vector.

    Args:
        pooling: Pooling strategy - "last" (last token), "mean" (mean pooling), "cls" (first token)
    """
    if custom_functions is None:
        custom_functions = CustomFunctions()

    # Determine which layers to extract from
    from .cv_loader import model_layer_list
    layers = model_layer_list(model)
    hidden_layers = list(range(len(layers)))

    # Prepare reference sequences
    ref_inputs = [{'sequence': entry.ref_sequence} for entry in dataset]
    ref_hiddens = custom_functions.F_batched_get_hiddens(
        model, processors, ref_inputs, hidden_layers, max_batch_size, custom_functions, pooling
    )

    # Prepare alternative sequences
    alt_inputs = [{'sequence': entry.alt_sequence} for entry in dataset]
    alt_hiddens = custom_functions.F_batched_get_hiddens(
        model, processors, alt_inputs, hidden_layers, max_batch_size, custom_functions, pooling
    )

    # Compute control directions
    directions = {}
    for layer in hidden_layers:
        ref_states = ref_hiddens[layer]
        alt_states = alt_hiddens[layer]

        if method == "pca_diff":
            # Compute difference vectors
            diff_vectors = alt_states - ref_states
            # Apply PCA to find the primary direction of variation
            if diff_vectors.shape[0] > 1:
                pca = PCA(n_components=1)
                pca.fit(diff_vectors)
                direction = pca.components_[0]
            else:
                direction = diff_vectors[0] / np.linalg.norm(diff_vectors[0])
        elif method == "mean_diff":
            # Simple mean difference
            direction = np.mean(alt_states - ref_states, axis=0)
            direction = direction / np.linalg.norm(direction)
        else:
            raise ValueError(f"Unknown method: {method}")

        directions[layer] = direction

    return directions
