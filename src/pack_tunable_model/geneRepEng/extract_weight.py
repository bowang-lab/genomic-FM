import dataclasses
import typing
import warnings
from pathlib import Path

import torch
import numpy as np
from transformers import PreTrainedModel
from peft import PeftModel, LoraConfig, get_peft_model

@dataclasses.dataclass
class LoraAdapter:
    """Represents a LoRA adapter for genomic models"""
    model_type: str
    adapter_weights: dict[str, torch.Tensor]
    config: typing.Optional[LoraConfig] = None

    @classmethod
    def from_pretrained(cls, model_path: str) -> "LoraAdapter":
        """Load a LoRA adapter from a saved path"""
        # This would typically load from a saved PEFT model
        # Implementation depends on the specific model format
        raise NotImplementedError("Loading from pretrained not yet implemented")

    def save(self, path: str):
        """Save the LoRA adapter"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save adapter weights
        torch.save(self.adapter_weights, save_path / "adapter_weights.pt")

        # Save config if available
        if self.config is not None:
            with open(save_path / "adapter_config.json", "w") as f:
                import json
                json.dump(self.config.to_dict(), f, indent=2)

        # Save metadata
        metadata = {"model_type": self.model_type}
        with open(save_path / "metadata.json", "w") as f:
            import json
            json.dump(metadata, f, indent=2)

def extract_lora_weights(model: PreTrainedModel, target_modules: typing.List[str] = None) -> dict:
    """
    Extract LoRA weights from a PEFT model for genomic applications

    Args:
        model: The PEFT model with LoRA adapters
        target_modules: List of module names to extract from

    Returns:
        Dictionary of LoRA weights
    """
    if not isinstance(model, PeftModel):
        raise ValueError("Model must be a PeftModel with LoRA adapters")

    lora_weights = {}

    # Extract LoRA A and B matrices
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            if target_modules is None or any(target in name for target in target_modules):
                lora_weights[f"{name}.lora_A"] = module.lora_A[module.adapter_name].weight.data.clone()
                lora_weights[f"{name}.lora_B"] = module.lora_B[module.adapter_name].weight.data.clone()

                # Also save scaling factor if available
                if hasattr(module, 'scaling'):
                    lora_weights[f"{name}.scaling"] = torch.tensor(module.scaling[module.adapter_name])

    return lora_weights

def combine_lora_adapters(
    adapters: typing.List[LoraAdapter],
    weights: typing.List[float] = None
) -> LoraAdapter:
    """
    Combine multiple LoRA adapters into a single adapter

    Args:
        adapters: List of LoRA adapters to combine
        weights: List of weights for combining (default: equal weights)

    Returns:
        Combined LoRA adapter
    """
    if not adapters:
        raise ValueError("At least one adapter must be provided")

    if weights is None:
        weights = [1.0 / len(adapters)] * len(adapters)

    if len(weights) != len(adapters):
        raise ValueError("Number of weights must match number of adapters")

    # Ensure all adapters have the same model type
    model_types = set(adapter.model_type for adapter in adapters)
    if len(model_types) > 1:
        warnings.warn(f"Multiple model types found: {model_types}")

    # Get all parameter names
    all_param_names = set()
    for adapter in adapters:
        all_param_names.update(adapter.adapter_weights.keys())

    # Combine weights
    combined_weights = {}
    for param_name in all_param_names:
        param_tensors = []
        param_weights = []

        for adapter, weight in zip(adapters, weights):
            if param_name in adapter.adapter_weights:
                param_tensors.append(adapter.adapter_weights[param_name])
                param_weights.append(weight)

        if param_tensors:
            # Weighted average of the parameters
            combined_param = sum(w * t for w, t in zip(param_weights, param_tensors))
            combined_param = combined_param / sum(param_weights)  # Normalize
            combined_weights[param_name] = combined_param

    return LoraAdapter(
        model_type=adapters[0].model_type,
        adapter_weights=combined_weights,
        config=adapters[0].config
    )

def create_genomic_lora_config(
    r: int = 8,
    lora_alpha: int = 32,
    target_modules: typing.List[str] = None,
    lora_dropout: float = 0.1,
    bias: str = "none",
    task_type: str = "FEATURE_EXTRACTION"
) -> LoraConfig:
    """
    Create a LoRA configuration optimized for genomic models

    Args:
        r: Rank of the LoRA decomposition
        lora_alpha: LoRA scaling parameter
        target_modules: List of module names to apply LoRA to
        lora_dropout: Dropout rate for LoRA layers
        bias: Bias type ("none", "all", or "lora_only")
        task_type: Task type for PEFT

    Returns:
        LoRA configuration
    """
    if target_modules is None:
        # Default target modules for common genomic model architectures
        target_modules = [
            "query", "key", "value", "dense",  # Attention modules
            "intermediate.dense", "output.dense",  # Feed-forward modules
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )

def apply_lora_to_genomic_model(
    model: PreTrainedModel,
    lora_config: LoraConfig = None
) -> PeftModel:
    """
    Apply LoRA to a genomic model

    Args:
        model: The base genomic model
        lora_config: LoRA configuration (creates default if None)

    Returns:
        PEFT model with LoRA adapters
    """
    if lora_config is None:
        lora_config = create_genomic_lora_config()

    peft_model = get_peft_model(model, lora_config)
    return peft_model
