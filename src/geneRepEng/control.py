import dataclasses
import typing
import warnings

import torch
from transformers import PretrainedConfig, PreTrainedModel

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
    Get splice function for OLMo models
    For OLMo, we want to wrap the entire layer block (not a method)
    """
    def olmo_splice(layer):
        # For OLMo models, we need to directly return the layer module
        # Don't try to access any attributes or methods, just return the layer itself
        print(f"Debug olmo_splice: layer type before return = {type(layer)}")
        print(f"Debug olmo_splice: layer = {layer}")
        result = layer  # Make sure we're returning the module, not calling anything on it
        print(f"Debug olmo_splice: result type = {type(result)}")
        return result
    return olmo_splice


def get_splice_fn(model):
    """
    Get the appropriate splice function for the model type
    """
    model_type = getattr(model.config, 'model_type', '')

    print(f"Debug get_splice_fn: model_type = '{model_type}'")
    print(f"Debug get_splice_fn: str(type(model)) = '{str(type(model))}'")
    print(f"Debug get_splice_fn: 'olmo' in model_type.lower() = {'olmo' in model_type.lower()}")
    print(f"Debug get_splice_fn: 'OLMoForSequenceCLS' in str(type(model)) = {'OLMoForSequenceCLS' in str(type(model))}")

    if 'olmo' in model_type.lower() or 'OLMoForSequenceCLS' in str(type(model)):
        print("Debug get_splice_fn: Using OLMo splice function")
        return get_olmo_splice_fn()
    else:
        print("Debug get_splice_fn: Using default splice function")
        # Default splice function for other models
        def default_splice_fn(layer):
            # Try different common layer structures
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
                # Default to the layer itself
                return layer
        return default_splice_fn


class ControlModule(torch.nn.Module):
    """
    A wrapper around a transformer layer that can apply control vectors.
    """

    def __init__(self, wrapped_layer, model_type: str):
        super().__init__()
        self.wrapped_layer = wrapped_layer
        self.model_type = model_type
        self.control_vector = None
        self.control_strength = 1.0

    def set_control(self, control_vector: torch.Tensor, strength: float = 1.0):
        """Set the control vector to apply"""
        # Get device from the wrapped layer's parameters
        device = next(self.wrapped_layer.parameters()).device
        self.control_vector = control_vector.to(device)
        self.control_strength = strength

    def clear_control(self):
        """Clear any applied control vector"""
        self.control_vector = None
        self.control_strength = 1.0

    def forward(self, *args, **kwargs):
        # Run the original layer
        output = self.wrapped_layer(*args, **kwargs)

        # Apply control if set
        if self.control_vector is not None:
            if isinstance(output, tuple):
                # If output is a tuple (hidden_states, ...), modify the hidden states
                hidden_states = output[0]
                if hidden_states.dim() == 3:  # (batch, seq_len, hidden_dim)
                    # Apply control to the last token
                    hidden_states[:, -1, :] += self.control_strength * self.control_vector
                elif hidden_states.dim() == 2:  # (seq_len, hidden_dim)
                    hidden_states[-1, :] += self.control_strength * self.control_vector
                output = (hidden_states,) + output[1:]
            else:
                # If output is just hidden states
                if output.dim() == 3:  # (batch, seq_len, hidden_dim)
                    output[:, -1, :] += self.control_strength * self.control_vector
                elif output.dim() == 2:  # (seq_len, hidden_dim)
                    output[-1, :] += self.control_strength * self.control_vector

        return output


class ControlModel(torch.nn.Module):
    """
    **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

    A wrapped genomic language model that can have controls set on its layers with `self.set_control`.
    """

    def __init__(self, model: PreTrainedModel, layer_ids, splice_fn=None):
        super().__init__()
        self.model = model
        self.dtype = model.dtype
        self.layer_ids = layer_ids
        self.splice_fn = splice_fn if splice_fn else get_splice_fn(model)

        layers = model_layer_list(model)
        for idx in self.layer_ids:
            print(f"Debug: splice_fn type = {type(self.splice_fn)}")
            print(f"Debug: calling splice_fn on layer {idx}")
            print(f"Debug: layers[{idx}] = {layers[idx]}")
            print(f"Debug: layers[{idx}] type = {type(layers[idx])}")
            target = self.splice_fn(layers[idx])            # module to wrap
            print(f"Debug: splice_fn returned {type(target)}")
            print(f"Debug: target = {target}")
            print(f"Debug: target is layers[{idx}]: {target is layers[idx]}")

            if not isinstance(target, ControlModule):
                wrapped = ControlModule(target, model_type=model.config.model_type)

                # Debug information
                print(f"Processing layer {idx}")
                print(f"Layer type: {type(layers[idx])}")
                print(f"Target type: {type(target)}")
                print(f"Target is layer: {target is layers[idx]}")
                print(f"Model type: {model.config.model_type}")

                # inplace overwrite on model tree to wrap sub-module
                if target is layers[idx]:
                    print(f"Directly replacing layer {idx}")
                    layers[idx] = wrapped
                else:
                    print(f"Using _assign_back for layer {idx}")
                    self._assign_back(layers[idx], target, wrapped)
            else:
                warnings.warn("Layer already wrapped! Skipping.")

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

        print(f"Warning: Could not locate sub-module to replace. Layer type: {type(layer)}")
        print(f"Old child type: {type(old_child)}, New child type: {type(new_child)}")
        print(f"Layer children: {list(layer.named_children())}")
        raise RuntimeError("Could not locate sub-module to replace")

    def forward(self, *args, **kwargs):
        """Forward pass through the controlled model"""
        return self.model(*args, **kwargs)

    @property
    def config(self):
        """Access to the underlying model's config"""
        return self.model.config

    def set_control(self, control_vector: "ControlVector", strength: float = 1.0):
        """
        Set control vectors on the specified layers.

        Args:
            control_vector: The ControlVector containing directions for each layer
            strength: The strength to apply the control (default: 1.0)
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
                    target.set_control(direction, strength)
                else:
                    warnings.warn(f"Layer {idx} is not wrapped with ControlModule")

    def clear_control(self):
        """Clear all control vectors"""
        layers = model_layer_list(self.model)
        for idx in self.layer_ids:
            target = self.splice_fn(layers[idx])
            if isinstance(target, ControlModule):
                target.clear_control()

    def reset(self):
        """Reset to original model state"""
        self.clear_control()

    def generate(self, *args, **kwargs):
        """Generation method that passes through to the underlying model"""
        return self.model.generate(*args, **kwargs)
