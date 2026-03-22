"""
Control Module for geneRepEng

Provides ControlModel and ControlModule wrappers for applying control vectors
to transformer layers, enabling representation steering.
"""

import typing
import warnings

import torch
from transformers import PreTrainedModel

if typing.TYPE_CHECKING:
    from .extract import ControlVector


def model_layer_list(model: PreTrainedModel):
    """Get the list of transformer layers from different model architectures"""
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
        # Omni-DNA and similar models
        return model.transformer.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'transformer') and hasattr(model.model.transformer, 'blocks'):
        # OLMoForSequenceCLS models
        return model.model.transformer.blocks
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
        # Direct OLMo transformer models
        return model.transformer.blocks
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-style models
        return model.transformer.h
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # LLaMA-style models
        return model.model.layers
    elif hasattr(model, 'esm') and hasattr(model.esm, 'encoder') and hasattr(model.esm.encoder, 'layer'):
        # ESM models (Nucleotide Transformer) - EsmForMaskedLM structure
        return model.esm.encoder.layer
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        # BERT-style models
        return model.encoder.layer
    elif hasattr(model, 'bert') and hasattr(model.bert, 'encoder') and hasattr(model.bert.encoder, 'layer'):
        # BERT models with bert attribute (BertForMaskedLM, etc.)
        return model.bert.encoder.layer
    elif hasattr(model, 'roberta') and hasattr(model.roberta, 'encoder') and hasattr(model.roberta.encoder, 'layer'):
        # RoBERTa models
        return model.roberta.encoder.layer
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")


def get_olmo_splice_fn():
    """
    Get splice function for OLMo models.
    For OLMo, we wrap the entire layer block (not a submodule).
    """
    def olmo_splice(layer):
        return layer
    return olmo_splice


def get_splice_fn(model: PreTrainedModel):
    """
    Get the appropriate splice function for the model type.

    Args:
        model: The model to get splice function for

    Returns:
        A function that extracts the target module from a layer
    """
    model_type = getattr(model.config, 'model_type', '')

    if 'olmo' in model_type.lower() or 'OLMoForSequenceCLS' in str(type(model)):
        return get_olmo_splice_fn()

    # Default splice function for other models
    def default_splice_fn(layer):
        if hasattr(layer, 'self_attn'):
            return layer.self_attn
        elif hasattr(layer, 'attention'):
            return layer.attention
        elif hasattr(layer, 'attn'):
            return layer.attn
        elif hasattr(layer, 'mlp'):
            return layer.mlp
        elif hasattr(layer, 'feed_forward'):
            return layer.feed_forward
        else:
            return layer
    return default_splice_fn


class ControlModule(torch.nn.Module):
    """
    A wrapper around a transformer layer that can apply control vectors.

    Supports both single control vectors (legacy) and multiple named control vectors
    that can be applied simultaneously.

    Args:
        wrapped_layer: The transformer layer to wrap
        model_type: Model type string from config
        pooling: Pooling strategy for control application:
            - "last": Apply to last token (default, for decoder models)
            - "cls": Apply to CLS/first token (for encoder models)
            - "mean": Apply to all tokens (for mean pooling)
    """

    def __init__(self, wrapped_layer, model_type: str, pooling: str = "last"):
        super().__init__()
        self.wrapped_layer = wrapped_layer
        self.model_type = model_type
        self.pooling = pooling
        # Legacy single control vector (for backward compatibility)
        self.control_vector = None
        self.control_strength = 1.0
        # Named control vectors: Dict[name -> (vector, strength)]
        self.control_vectors: typing.Dict[str, typing.Tuple[torch.Tensor, float]] = {}

    def set_control(self, control_vector: torch.Tensor, strength: float = 1.0, name: str = "default"):
        """
        Set a control vector to apply.

        Args:
            control_vector: The control vector tensor
            strength: The strength to apply (default: 1.0)
            name: Name for this control vector (default: "default")
        """
        # Get device from the wrapped layer's parameters
        device = next(self.wrapped_layer.parameters()).device
        vector = control_vector.to(device)

        # Store in named dict
        self.control_vectors[name] = (vector, strength)

        # Also set legacy attributes for backward compatibility
        if name == "default":
            self.control_vector = vector
            self.control_strength = strength

    def clear_control(self, name: str = None):
        """
        Clear control vector(s).

        Args:
            name: If provided, clear only this named vector.
                  If None, clear all control vectors.
        """
        if name is None:
            # Clear all
            self.control_vectors.clear()
            self.control_vector = None
            self.control_strength = 1.0
        else:
            # Clear specific named vector
            self.control_vectors.pop(name, None)
            if name == "default":
                self.control_vector = None
                self.control_strength = 1.0

    def forward(self, *args, **kwargs):
        # Run the original layer
        output = self.wrapped_layer(*args, **kwargs)

        # Apply control if any vectors are set
        if self.control_vectors:
            # Sum all control vectors with their strengths
            total_control = None
            for vector, strength in self.control_vectors.values():
                scaled = strength * vector
                if total_control is None:
                    total_control = scaled
                else:
                    total_control = total_control + scaled

            if total_control is not None:
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    hidden_states = self._apply_control(hidden_states, total_control)
                    output = (hidden_states,) + output[1:]
                else:
                    output = self._apply_control(output, total_control)

        return output

    def _apply_control(self, hidden_states: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Apply control vector based on pooling strategy."""
        if self.pooling == "cls":
            # Apply to CLS/first token (encoder models)
            if hidden_states.dim() == 3:  # (batch, seq_len, hidden_dim)
                hidden_states[:, 0, :] += control
            elif hidden_states.dim() == 2:  # (seq_len, hidden_dim)
                hidden_states[0, :] += control
        elif self.pooling == "mean":
            # Apply to all tokens (for mean pooling)
            if hidden_states.dim() == 3:  # (batch, seq_len, hidden_dim)
                hidden_states = hidden_states + control.unsqueeze(0).unsqueeze(0)
            elif hidden_states.dim() == 2:  # (seq_len, hidden_dim)
                hidden_states = hidden_states + control.unsqueeze(0)
        else:
            # Default "last": Apply to last token (decoder models)
            if hidden_states.dim() == 3:  # (batch, seq_len, hidden_dim)
                hidden_states[:, -1, :] += control
            elif hidden_states.dim() == 2:  # (seq_len, hidden_dim)
                hidden_states[-1, :] += control
        return hidden_states


class ControlModel(torch.nn.Module):
    """
    **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

    A wrapped genomic language model that can have controls set on its layers with `self.set_control`.

    Args:
        model: The transformer model to wrap
        layer_ids: Which layers to apply control to
        splice_fn: Optional function to extract target module from layer
        pooling: Pooling strategy for control application:
            - "last": Apply to last token (default, for decoder models like Omni-DNA)
            - "cls": Apply to CLS/first token (for encoder models like NT, DNABERT)
            - "mean": Apply to all tokens (for mean pooling)
    """

    def __init__(self, model: PreTrainedModel, layer_ids: typing.List[int], splice_fn=None, pooling: str = "last"):
        super().__init__()
        self.model = model
        self.dtype = model.dtype
        self.layer_ids = layer_ids
        self.pooling = pooling
        self.splice_fn = splice_fn if splice_fn else get_splice_fn(model)
        self._wrapped_layers: typing.Dict[int, ControlModule] = {}

        layers = model_layer_list(model)
        for idx in self.layer_ids:
            target = self.splice_fn(layers[idx])

            if not isinstance(target, ControlModule):
                wrapped = ControlModule(target, model_type=model.config.model_type, pooling=pooling)

                # Inplace overwrite on model tree to wrap sub-module
                if target is layers[idx]:
                    layers[idx] = wrapped
                else:
                    self._assign_back(layers[idx], target, wrapped)

                # Track wrapped layers for per-layer access
                self._wrapped_layers[idx] = wrapped
            else:
                warnings.warn(f"Layer {idx} already wrapped! Skipping.")
                self._wrapped_layers[idx] = target

    @staticmethod
    def _assign_back(layer, old_child, new_child):
        """
        Replace the attribute in `layer` that references `old_child`
        with `new_child` (works for .self_attn, .mlp, etc.).
        """
        for attr_name, mod in layer.named_children():
            if mod is old_child:
                setattr(layer, attr_name, new_child)
                return

        # If we couldn't find it in named_children, try a more thorough search
        for attr_name in dir(layer):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(layer, attr_name)
                    if attr_value is old_child:
                        setattr(layer, attr_name, new_child)
                        return
                except (AttributeError, TypeError):
                    continue

        raise RuntimeError(
            f"Could not locate sub-module to replace. "
            f"Layer type: {type(layer)}, children: {[n for n, _ in layer.named_children()]}"
        )

    def forward(self, *args, **kwargs):
        """Forward pass through the controlled model"""
        return self.model(*args, **kwargs)

    @property
    def config(self):
        """Access to the underlying model's config"""
        return self.model.config

    @property
    def device(self):
        """Access to the underlying model's device"""
        return next(self.model.parameters()).device

    def set_control(self, control_vector: "ControlVector", strength: float = 1.0, name: str = "default"):
        """
        Set control vectors on the specified layers.

        Args:
            control_vector: The ControlVector containing directions for each layer
            strength: The strength to apply the control (default: 1.0)
            name: Name for this control vector (default: "default")
        """
        layers = model_layer_list(self.model)
        for idx in self.layer_ids:
            if idx in control_vector.directions:
                target = self.splice_fn(layers[idx])
                if isinstance(target, ControlModule):
                    direction = torch.tensor(
                        control_vector.directions[idx],
                        dtype=self.dtype,
                        device=next(self.model.parameters()).device
                    )
                    target.set_control(direction, strength, name)
                else:
                    warnings.warn(f"Layer {idx} is not wrapped with ControlModule")

    def set_multi_control(
        self,
        control_vectors: typing.Dict[str, "ControlVector"],
        strengths: typing.Optional[typing.Dict[str, float]] = None
    ):
        """
        Apply multiple named control vectors simultaneously.

        Args:
            control_vectors: Dict mapping names to ControlVectors
            strengths: Optional dict mapping names to strengths.
                       If not provided, all strengths default to 1.0.

        Example:
            model.set_multi_control(
                {"disease1": cv1, "disease2": cv2},
                {"disease1": 1.0, "disease2": -0.5}
            )
        """
        if strengths is None:
            strengths = {}

        for name, cv in control_vectors.items():
            strength = strengths.get(name, 1.0)
            self.set_control(cv, strength=strength, name=name)

    def clear_control(self, name: str = None):
        """
        Clear control vector(s).

        Args:
            name: If provided, clear only this named vector.
                  If None, clear all control vectors.
        """
        layers = model_layer_list(self.model)
        for idx in self.layer_ids:
            target = self.splice_fn(layers[idx])
            if isinstance(target, ControlModule):
                target.clear_control(name)

    def reset(self):
        """Reset to original model state"""
        self.clear_control()

    def generate(self, *args, **kwargs):
        """Generation method that passes through to the underlying model"""
        return self.model.generate(*args, **kwargs)
