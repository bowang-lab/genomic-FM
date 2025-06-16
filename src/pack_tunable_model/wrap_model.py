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
    def __init__(self, base_model, num_classes, decoder=False,hidden_states_pooler=True):
        super().__init__()
        self.base_model = base_model
        self.decoder = decoder

        # Get the hidden size from the base model configuration
        if hasattr(base_model, "config"):
            if hasattr(self.base_model.config,"hidden_size"):
                hidden_size = base_model.config.hidden_size
            else:
                hidden_size = base_model.config.d_model
        else:
            # Fallback if config is not available
            hidden_size = 768  # Default size, adjust as needed
        if hidden_states_pooler:
            # Add a pooler layer if needed
            self.pooler = BertPooler(base_model.config)
        else:
            self.pooler = None
        self.classification_head = nn.Linear(hidden_size, num_classes)
        # self.classification_head = nn.Linear(hidden_size, num_classes)


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
        outputs_ref = self.base_model(
            input_ids=ref_input_ids,
            attention_mask=ref_attention_mask,
            output_hidden_states=True,
            return_dict=True,
            # **kwargs
        )

        # Process alternate sequence
        outputs_alt = self.base_model(
            input_ids=alt_input_ids,
            attention_mask=alt_attention_mask,
            output_hidden_states=True,
            return_dict=True,
            # **kwargs
        )
        # check the shape of outputs_ref.hidden_states and outputs_alt.hidden_states
        # if it does not have the all layers, then add a dummy dimension by nest it in a list
        # if not isinstance(outputs_ref.hidden_states, tuple):
        if not isinstance(outputs_ref.hidden_states, tuple):
            if hasattr(self.base_model.config,"hidden_size"):
                outputs_ref.hidden_states = (outputs_ref.hidden_states,)
        # if not isinstance(outputs_alt.hidden_states, tuple):
        if not isinstance(outputs_alt.hidden_states, tuple):
            if hasattr(self.base_model.config,"hidden_size"):
                outputs_alt.hidden_states = (outputs_alt.hidden_states,)
        if not self.decoder:
            # For encoder models, take the [CLS] token (first token) from the last hidden state
            # print(f"outputs_ref.hidden_states: {outputs_ref.hidden_states.shape}")
            last_hidden_state_ref = outputs_ref.hidden_states[-1][:, 0, :]
            # last_hidden_state_ref = outputs_ref.hidden_states[-1].mean(dim=1)
            last_hidden_state_alt = outputs_alt.hidden_states[-1][:, 0, :]
            # last_hidden_state_alt = outputs_alt.hidden_states[-1].mean(dim=1)
            if self.pooler is not None:
                # Apply pooler if needed
                last_hidden_state_ref = self.pooler(last_hidden_state_ref)
                last_hidden_state_alt = self.pooler(last_hidden_state_alt)
            difference = last_hidden_state_alt - last_hidden_state_ref
        else:
            # For decoder models, take the last token from the sequence
            # compute true sequence lengths

            # ref_seq_lens = outputs_ref.attention_mask.sum(dim=-1)  # (batch_size,)
            # alt_seq_lens = outputs_alt.attention_mask.sum(dim=-1)  # (batch_size,)
            ref_seq_lens = ref_attention_mask.sum(dim=-1)  # (batch_size,)
            alt_seq_lens = alt_attention_mask.sum(dim=-1)  # (batch_size,)

            # build an index for batch dimension
            batch_index = torch.arange(ref_seq_lens.size(0), device=outputs_ref.hidden_states[-1].device)

            # select the last‐token vector for each example
            last_ref = outputs_ref.hidden_states[-1][batch_index, ref_seq_lens - 1, :]
            last_alt = outputs_alt.hidden_states[-1][batch_index, alt_seq_lens - 1, :]
            if self.pooler is not None:
                # Apply pooler if needed
                last_ref = self.pooler(last_ref)
                last_alt = self.pooler(last_alt)
            # take the difference
            difference = last_alt - last_ref

        # Apply classification head
        logits = self.classification_head(difference)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Return output in dictionary format similar to HuggingFace models
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": (outputs_ref.hidden_states, outputs_alt.hidden_states) if output_hidden_states else None,
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
                # Get output dimension from the weight matrix
                weight_shape = head_state_dict.get("weight").shape
                if weight_shape:
                    num_classes = weight_shape[0]
                else:
                    raise ValueError("Could not determine num_classes from saved state")
            except (FileNotFoundError, KeyError):
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
