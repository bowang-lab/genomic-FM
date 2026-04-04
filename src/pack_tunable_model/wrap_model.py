import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedModel

class BertPooler(nn.Module):
    """ some models needs a pooler layer applied to hidden states, some models do not need a pooler layer"""
    def __init__(self, config):
        super(BertPooler, self).__init__()
        if hasattr(config,"hidden_size"):
            hidden_size = config.hidden_size
        else:
            hidden_size = config.d_model
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self,
                hidden_states: torch.Tensor,
                pool: Optional[bool] = True) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class WrappedModelWithClassificationHead(nn.Module):
    def __init__(self, base_model, num_classes, decoder=False, hidden_states_pooler=False,
                 comparison_mode="delta", pooling="cls"):
        """
        Args:
            comparison_mode: "delta" (subtraction) or "concat" (concatenation like DYNA)
            pooling: "cls" (first token), "mean" (mean pooling), or "last" (last non-padding token)
        """
        super().__init__()
        self.base_model = base_model
        self.decoder = decoder
        self.comparison_mode = comparison_mode
        self.pooling = pooling

        if pooling not in ["cls", "mean", "last"]:
            raise ValueError(f"Unknown pooling: {pooling}. Use 'cls', 'mean', or 'last'.")

        # Get the hidden size from the base model configuration
        if hasattr(base_model, "config"):
            if hasattr(self.base_model.config,"hidden_size"):
                hidden_size = base_model.config.hidden_size
            else:
                hidden_size = base_model.config.d_model
        else:
            # Fallback if config is not available
            hidden_size = 768  # Default size, adjust as needed

        self.hidden_size = hidden_size

        if hidden_states_pooler:
            # Add a pooler layer if needed
            self.pooler = BertPooler(base_model.config)
        else:
            self.pooler = None

        # Determine input size based on comparison mode
        if comparison_mode == "delta":
            head_input_size = hidden_size
        elif comparison_mode == "concat":
            head_input_size = hidden_size * 2
        else:
            raise ValueError(f"Unknown comparison_mode: {comparison_mode}. Use 'delta' or 'concat'.")

        # MLP head instead of single linear (like DYNA)
        self.classification_head = nn.Sequential(
            nn.Linear(head_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

        # Initialize final layer properly for regression vs classification
        final_layer = self.classification_head[-1]
        if num_classes == 1:
            nn.init.normal_(final_layer.weight, mean=0.0, std=1e-4)
            nn.init.constant_(final_layer.bias, 0.0)
        else:
            nn.init.xavier_uniform_(final_layer.weight)
            nn.init.constant_(final_layer.bias, 0.0)

    def _apply_pooling(self, hidden_states, attention_mask):
        """
        Apply pooling strategy to hidden states.

        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Tensor of shape (batch_size, seq_len)

        Returns:
            Pooled tensor of shape (batch_size, hidden_size)
        """
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        elif self.pooling == "mean":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_hidden / sum_mask
            else:
                return hidden_states.mean(dim=1)
        elif self.pooling == "last":
            if attention_mask is not None:
                # Get the last non-padding token for each sequence
                seq_lengths = attention_mask.sum(dim=1)
                batch_size = hidden_states.size(0)
                batch_indices = torch.arange(batch_size, device=hidden_states.device)
                return hidden_states[batch_indices, seq_lengths - 1, :]
            else:
                return hidden_states[:, -1, :]

    def forward(self,
                input_ids=None,
                attention_mask=None,
                ref_input_ids=None,
                ref_attention_mask=None,
                alt_input_ids=None,
                alt_attention_mask=None,
                labels=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs):
        """
        Forward pass that handles the differential encoding between reference and alternate sequences.
        """
        # Make sure we have the required inputs
        if ref_input_ids is None or alt_input_ids is None:
            raise ValueError("Both ref_input_ids and alt_input_ids must be provided")

        # Process reference sequence
        # Check if model doesn't support attention_mask (HyenaDNA, Caduceus/Mamba)
        model_class_name = str(self.base_model.__class__)
        skip_attention_mask = any(name in model_class_name for name in ['HyenaDNA', 'Caduceus', 'Mamba'])

        if skip_attention_mask:
            outputs_ref = self.base_model(
                input_ids=ref_input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            outputs_ref = self.base_model(
                input_ids=ref_input_ids,
                attention_mask=ref_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Process alternate sequence
        if skip_attention_mask:
            outputs_alt = self.base_model(
                input_ids=alt_input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            outputs_alt = self.base_model(
                input_ids=alt_input_ids,
                attention_mask=alt_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Handle models that return tuples instead of objects (e.g., DNABERT2)
        # Extract hidden states properly based on output type
        if isinstance(outputs_ref, tuple):
            # For models returning tuples, the last hidden state is typically the first element
            hidden_states_ref = (outputs_ref[0],)
        elif hasattr(outputs_ref, 'hidden_states') and outputs_ref.hidden_states is not None:
            hidden_states_ref = outputs_ref.hidden_states
            if not isinstance(hidden_states_ref, tuple):
                hidden_states_ref = (hidden_states_ref,)
        elif hasattr(outputs_ref, 'last_hidden_state'):
            hidden_states_ref = (outputs_ref.last_hidden_state,)
        else:
            raise ValueError(f"Cannot extract hidden states from model output: {type(outputs_ref)}")

        if isinstance(outputs_alt, tuple):
            hidden_states_alt = (outputs_alt[0],)
        elif hasattr(outputs_alt, 'hidden_states') and outputs_alt.hidden_states is not None:
            hidden_states_alt = outputs_alt.hidden_states
            if not isinstance(hidden_states_alt, tuple):
                hidden_states_alt = (hidden_states_alt,)
        elif hasattr(outputs_alt, 'last_hidden_state'):
            hidden_states_alt = (outputs_alt.last_hidden_state,)
        else:
            raise ValueError(f"Cannot extract hidden states from model output: {type(outputs_alt)}")
        if not self.decoder:
            # For encoder models, apply pooling strategy (cls or mean)
            last_hidden_state_ref = self._apply_pooling(hidden_states_ref[-1], ref_attention_mask)
            last_hidden_state_alt = self._apply_pooling(hidden_states_alt[-1], alt_attention_mask)
            if self.pooler is not None:
                last_hidden_state_ref = self.pooler(last_hidden_state_ref)
                last_hidden_state_alt = self.pooler(last_hidden_state_alt)
        else:
            # For decoder models, take the last token from the sequence
            ref_seq_lens = ref_attention_mask.sum(dim=-1)
            alt_seq_lens = alt_attention_mask.sum(dim=-1)
            batch_index = torch.arange(ref_seq_lens.size(0), device=hidden_states_ref[-1].device)
            last_hidden_state_ref = hidden_states_ref[-1][batch_index, ref_seq_lens - 1, :]
            last_hidden_state_alt = hidden_states_alt[-1][batch_index, alt_seq_lens - 1, :]
            if self.pooler is not None:
                last_hidden_state_ref = self.pooler(last_hidden_state_ref)
                last_hidden_state_alt = self.pooler(last_hidden_state_alt)

        # Build features based on comparison mode
        if self.comparison_mode == "delta":
            features = last_hidden_state_alt - last_hidden_state_ref
        elif self.comparison_mode == "concat":
            features = torch.cat([last_hidden_state_ref, last_hidden_state_alt], dim=1)

        # Apply classification head
        logits = self.classification_head(features)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Check if this is a regression task (num_classes = 1)
            if logits.size(-1) == 1:
                # Regression task - use MSELoss
                loss_fct = torch.nn.MSELoss()
                # Squeeze logits to match label dimensions for regression
                loss = loss_fct(logits.squeeze(-1), labels.float())
            else:
                # Classification task - use CrossEntropyLoss
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Return output in dictionary format similar to HuggingFace models
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": (hidden_states_ref, hidden_states_alt) if output_hidden_states else None,
            "ref_outputs": outputs_ref,
            "alt_outputs": outputs_alt,
        }

    def get_input_embeddings(self):
        """Get input embeddings from the base model."""
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set input embeddings for the base model."""
        self.base_model.set_input_embeddings(value)

    def save_pretrained(self, save_directory, **kwargs):
        """Save both the base model and classification head."""
        # First save the base model
        self.base_model.save_pretrained(save_directory, **kwargs)

        # Save the classification head separately
        torch.save(self.classification_head.state_dict(),
                  f"{save_directory}/classification_head.bin")

    @classmethod
    def from_pretrained(cls, model_path, num_classes=None, decoder=False, **kwargs):
        """Load model from pretrained weights."""
        # Load base model
        base_model = PreTrainedModel.from_pretrained(model_path, **kwargs)

        # Determine num_classes from the saved classification head if not provided
        if num_classes is None:
            # Try to infer from the saved classification head
            try:
                head_state_dict = torch.load(f"{model_path}/classification_head.bin")
                # The head is nn.Sequential; the final Linear layer's weight
                # has keys like "3.weight" (index within Sequential)
                final_weight_key = None
                for key in head_state_dict:
                    if key.endswith(".weight"):
                        final_weight_key = key  # last .weight key is the output layer
                if final_weight_key is not None:
                    num_classes = head_state_dict[final_weight_key].shape[0]
                else:
                    raise ValueError("Could not determine num_classes from saved state")
            except (FileNotFoundError, KeyError, AttributeError):
                raise ValueError("num_classes must be provided when loading model")

        # Create instance
        model = cls(base_model, num_classes, decoder=decoder)

        # Load classification head if exists
        try:
            head_state_dict = torch.load(f"{model_path}/classification_head.bin")
            model.classification_head.load_state_dict(head_state_dict)
        except FileNotFoundError:
            print("No classification head found, using initialized weights.")

        return model


class WrappedModelWithPairedVariantHead(nn.Module):
    """
    Model wrapper for oligogenic prediction with paired variant processing.

    Processes each variant in a pair separately, then combines their embeddings
    for classification. This allows the model to learn variant-variant interactions.
    """

    def __init__(self, base_model, num_classes=2, decoder=False, hidden_states_pooler=False,
                 combine_mode="concat", pooling="cls"):
        """
        Args:
            combine_mode: How to combine variant embeddings:
                - "concat": [emb1; emb2] -> 2x hidden_size
                - "diff": emb1 - emb2 -> hidden_size
                - "hadamard": [emb1; emb2; emb1 * emb2] -> 3x hidden_size
            pooling: "cls" (first token), "mean" (mean pooling), or "last" (last non-padding token)
        """
        super().__init__()
        self.base_model = base_model
        self.decoder = decoder
        self.combine_mode = combine_mode
        self.pooling = pooling

        if pooling not in ["cls", "mean", "last"]:
            raise ValueError(f"Unknown pooling: {pooling}. Use 'cls', 'mean', or 'last'.")

        # Get the hidden size from the base model configuration
        if hasattr(base_model, "config"):
            if hasattr(self.base_model.config, "hidden_size"):
                hidden_size = base_model.config.hidden_size
            else:
                hidden_size = base_model.config.d_model
        else:
            hidden_size = 768

        self.hidden_size = hidden_size

        if hidden_states_pooler:
            self.pooler = BertPooler(base_model.config)
        else:
            self.pooler = None

        # Determine input size based on combine mode
        if combine_mode == "concat":
            head_input_size = hidden_size * 2
        elif combine_mode == "diff":
            head_input_size = hidden_size
        elif combine_mode == "hadamard":
            head_input_size = hidden_size * 3
        else:
            raise ValueError(f"Unknown combine_mode: {combine_mode}")

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(head_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

        # Initialize final layer
        final_layer = self.classification_head[-1]
        nn.init.xavier_uniform_(final_layer.weight)
        nn.init.constant_(final_layer.bias, 0.0)

    def _apply_pooling(self, hidden_states, attention_mask):
        """
        Apply pooling strategy to hidden states.

        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Tensor of shape (batch_size, seq_len)

        Returns:
            Pooled tensor of shape (batch_size, hidden_size)
        """
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        elif self.pooling == "mean":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_hidden / sum_mask
            else:
                return hidden_states.mean(dim=1)
        elif self.pooling == "last":
            if attention_mask is not None:
                # Get the last non-padding token for each sequence
                seq_lengths = attention_mask.sum(dim=1)
                batch_size = hidden_states.size(0)
                batch_indices = torch.arange(batch_size, device=hidden_states.device)
                return hidden_states[batch_indices, seq_lengths - 1, :]
            else:
                return hidden_states[:, -1, :]

    def get_variant_embedding(self, ref_input_ids, ref_attention_mask, alt_input_ids, alt_attention_mask):
        """Get differential embedding (alt - ref) for a single variant."""
        model_class_name = str(self.base_model.__class__)
        skip_attention_mask = any(name in model_class_name for name in ['HyenaDNA', 'Caduceus', 'Mamba'])

        if skip_attention_mask:
            outputs_ref = self.base_model(input_ids=ref_input_ids, output_hidden_states=True, return_dict=True)
            outputs_alt = self.base_model(input_ids=alt_input_ids, output_hidden_states=True, return_dict=True)
        else:
            outputs_ref = self.base_model(input_ids=ref_input_ids, attention_mask=ref_attention_mask,
                                          output_hidden_states=True, return_dict=True)
            outputs_alt = self.base_model(input_ids=alt_input_ids, attention_mask=alt_attention_mask,
                                          output_hidden_states=True, return_dict=True)

        # Handle models that return tuples instead of objects (e.g., DNABERT2)
        if isinstance(outputs_ref, tuple):
            hidden_states_ref = (outputs_ref[0],)
        elif hasattr(outputs_ref, 'hidden_states') and outputs_ref.hidden_states is not None:
            hidden_states_ref = outputs_ref.hidden_states
            if not isinstance(hidden_states_ref, tuple):
                hidden_states_ref = (hidden_states_ref,)
        elif hasattr(outputs_ref, 'last_hidden_state'):
            hidden_states_ref = (outputs_ref.last_hidden_state,)
        else:
            raise ValueError(f"Cannot extract hidden states from model output: {type(outputs_ref)}")

        if isinstance(outputs_alt, tuple):
            hidden_states_alt = (outputs_alt[0],)
        elif hasattr(outputs_alt, 'hidden_states') and outputs_alt.hidden_states is not None:
            hidden_states_alt = outputs_alt.hidden_states
            if not isinstance(hidden_states_alt, tuple):
                hidden_states_alt = (hidden_states_alt,)
        elif hasattr(outputs_alt, 'last_hidden_state'):
            hidden_states_alt = (outputs_alt.last_hidden_state,)
        else:
            raise ValueError(f"Cannot extract hidden states from model output: {type(outputs_alt)}")

        if not self.decoder:
            # For encoder models, apply pooling strategy (cls or mean)
            ref_embed = self._apply_pooling(hidden_states_ref[-1], ref_attention_mask)
            alt_embed = self._apply_pooling(hidden_states_alt[-1], alt_attention_mask)
        else:
            ref_seq_lens = ref_attention_mask.sum(dim=-1)
            alt_seq_lens = alt_attention_mask.sum(dim=-1)
            batch_index = torch.arange(ref_seq_lens.size(0), device=ref_input_ids.device)
            ref_embed = hidden_states_ref[-1][batch_index, ref_seq_lens - 1, :]
            alt_embed = hidden_states_alt[-1][batch_index, alt_seq_lens - 1, :]

        if self.pooler is not None:
            ref_embed = self.pooler(ref_embed)
            alt_embed = self.pooler(alt_embed)

        return alt_embed - ref_embed

    def forward(
        self,
        variant1_ref_input_ids=None,
        variant1_ref_attention_mask=None,
        variant1_alt_input_ids=None,
        variant1_alt_attention_mask=None,
        variant2_ref_input_ids=None,
        variant2_ref_attention_mask=None,
        variant2_alt_input_ids=None,
        variant2_alt_attention_mask=None,
        labels=None,
        **kwargs
    ):
        """
        Forward pass processing each variant separately then combining.
        """
        # Get embedding for variant 1
        emb1 = self.get_variant_embedding(
            variant1_ref_input_ids, variant1_ref_attention_mask,
            variant1_alt_input_ids, variant1_alt_attention_mask
        )

        # Get embedding for variant 2
        emb2 = self.get_variant_embedding(
            variant2_ref_input_ids, variant2_ref_attention_mask,
            variant2_alt_input_ids, variant2_alt_attention_mask
        )

        # Combine embeddings
        if self.combine_mode == "concat":
            combined = torch.cat([emb1, emb2], dim=1)
        elif self.combine_mode == "diff":
            combined = emb1 - emb2
        elif self.combine_mode == "hadamard":
            combined = torch.cat([emb1, emb2, emb1 * emb2], dim=1)

        # Classification
        logits = self.classification_head(combined)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            if logits.size(-1) == 1:
                loss = nn.MSELoss()(logits.squeeze(-1), labels.float())
            else:
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "variant1_embedding": emb1,
            "variant2_embedding": emb2,
        }

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)
