"""
Multi-Variant Model Wrapper

Supports both local and aggregated modes for handling multiple variants per patient:
- Local mode: Standard delta embedding from single multi-variant sequence
- Aggregated mode: Attention-based aggregation of per-variant embeddings

Multi-task heads for disease classification and pathogenicity prediction.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union
import math


class VariantAttentionAggregator(nn.Module):
    """
    Attention-based aggregation for variable-length variant embeddings.

    Uses cross-attention where a learnable CLS query attends to all variant embeddings.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Learnable CLS token for querying variant embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        nn.init.normal_(self.cls_token, std=0.02)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm and feed-forward
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        variant_embeddings: torch.Tensor,
        variant_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate variant embeddings using attention.

        Args:
            variant_embeddings: (batch_size, max_variants, hidden_size)
            variant_mask: (batch_size, max_variants) - True for valid variants

        Returns:
            Aggregated embedding: (batch_size, hidden_size)
        """
        batch_size = variant_embeddings.size(0)

        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, hidden)

        # Create key padding mask for attention (True = ignore)
        # MultiheadAttention expects True for positions to mask
        key_padding_mask = ~variant_mask  # Invert: True means padding

        # Cross-attention: CLS queries variant embeddings
        attn_output, attn_weights = self.attention(
            query=cls_tokens,
            key=variant_embeddings,
            value=variant_embeddings,
            key_padding_mask=key_padding_mask,
        )

        # Residual + norm (using CLS as residual)
        x = self.norm1(cls_tokens + attn_output)

        # Feed-forward with residual
        x = self.norm2(x + self.ffn(x))

        # Return aggregated embedding (squeeze CLS dimension)
        return x.squeeze(1)  # (batch_size, hidden_size)


class VariantCountEmbedding(nn.Module):
    """
    Learnable embedding for variant count.

    Allows the model to learn from the number of variants per patient.
    """

    def __init__(self, max_variants: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_variants + 1, embedding_dim)

    def forward(self, num_variants: torch.Tensor) -> torch.Tensor:
        """
        Args:
            num_variants: (batch_size,) tensor of variant counts

        Returns:
            Embeddings: (batch_size, embedding_dim)
        """
        return self.embedding(num_variants.clamp(max=self.embedding.num_embeddings - 1))


class WrappedModelWithMultiVariantHead(nn.Module):
    """
    Model wrapper for multi-variant per patient prediction.

    Supports:
    - Local mode: Standard delta embedding (alt - ref)
    - Aggregated mode: Attention-based aggregation of per-variant embeddings
    - Multi-task heads: Disease classification + pathogenicity prediction
    """

    def __init__(
        self,
        base_model,
        num_disease_classes: int = 4,
        num_pathogenicity_classes: int = 2,
        max_variants: int = 10,
        decoder: bool = False,
        hidden_states_pooler: bool = False,
        pooling: str = "cls",
        attention_heads: int = 4,
        dropout: float = 0.1,
        disease_weight: float = 1.0,
        pathogenicity_weight: float = 1.0,
    ):
        """
        Args:
            base_model: Pretrained genomic foundation model
            num_disease_classes: Number of disease classes
            num_pathogenicity_classes: Number of pathogenicity classes (2 for binary)
            max_variants: Maximum variants per patient
            decoder: Whether base model is decoder-only
            hidden_states_pooler: Whether to apply BERT-style pooler
            pooling: Pooling strategy ("cls", "mean", "last")
            attention_heads: Number of attention heads for aggregation
            dropout: Dropout rate
            disease_weight: Weight for disease classification loss
            pathogenicity_weight: Weight for pathogenicity loss
        """
        super().__init__()
        self.base_model = base_model
        self.decoder = decoder
        self.pooling = pooling
        self.max_variants = max_variants
        self.disease_weight = disease_weight
        self.pathogenicity_weight = pathogenicity_weight

        # Get hidden size from base model
        if hasattr(base_model, "config"):
            if hasattr(base_model.config, "hidden_size"):
                hidden_size = base_model.config.hidden_size
            else:
                hidden_size = base_model.config.d_model
        else:
            hidden_size = 768

        self.hidden_size = hidden_size

        # Optional pooler
        if hidden_states_pooler:
            from .wrap_model import BertPooler
            self.pooler = BertPooler(base_model.config)
        else:
            self.pooler = None

        # Attention aggregator for aggregated mode
        self.attention_aggregator = VariantAttentionAggregator(
            hidden_size=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
        )

        # Variant count embedding
        self.variant_count_embedding = VariantCountEmbedding(
            max_variants=max_variants,
            embedding_dim=hidden_size,
        )

        # Projection to combine variant embedding and count embedding
        self.combine_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Multi-task heads
        self.disease_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_disease_classes),
        )

        self.pathogenicity_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_pathogenicity_classes),
        )

        # Initialize heads
        for head in [self.disease_head, self.pathogenicity_head]:
            final_layer = head[-1]
            nn.init.xavier_uniform_(final_layer.weight)
            nn.init.constant_(final_layer.bias, 0.0)

        # Loss functions
        self.disease_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.pathogenicity_loss_fn = nn.CrossEntropyLoss()

    def _apply_pooling(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Apply pooling strategy to hidden states."""
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
                seq_lengths = attention_mask.sum(dim=1)
                batch_size = hidden_states.size(0)
                batch_indices = torch.arange(batch_size, device=hidden_states.device)
                return hidden_states[batch_indices, seq_lengths - 1, :]
            else:
                return hidden_states[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def _get_hidden_states(self, outputs):
        """Extract hidden states from model outputs."""
        if isinstance(outputs, tuple):
            return (outputs[0],)
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states
            return hidden_states if isinstance(hidden_states, tuple) else (hidden_states,)
        elif hasattr(outputs, 'last_hidden_state'):
            return (outputs.last_hidden_state,)
        else:
            raise ValueError(f"Cannot extract hidden states from: {type(outputs)}")

    def _encode_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode a sequence and return pooled representation."""
        # Check if model doesn't support attention_mask
        model_class_name = str(self.base_model.__class__)
        skip_attention_mask = any(
            name in model_class_name for name in ['HyenaDNA', 'Caduceus', 'Mamba']
        )

        if skip_attention_mask:
            outputs = self.base_model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = self._get_hidden_states(outputs)

        if not self.decoder:
            pooled = self._apply_pooling(hidden_states[-1], attention_mask)
        else:
            seq_lens = attention_mask.sum(dim=-1)
            batch_index = torch.arange(seq_lens.size(0), device=input_ids.device)
            pooled = hidden_states[-1][batch_index, seq_lens - 1, :]

        if self.pooler is not None:
            pooled = self.pooler(pooled)

        return pooled

    def _get_delta_embedding(
        self,
        ref_input_ids: torch.Tensor,
        ref_attention_mask: Optional[torch.Tensor],
        alt_input_ids: torch.Tensor,
        alt_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Get delta embedding (alt - ref) for variant effect."""
        ref_embed = self._encode_sequence(ref_input_ids, ref_attention_mask)
        alt_embed = self._encode_sequence(alt_input_ids, alt_attention_mask)
        return alt_embed - ref_embed

    def _forward_local_batch(
        self,
        ref_input_ids: torch.Tensor,
        ref_attention_mask: torch.Tensor,
        alt_input_ids: torch.Tensor,
        alt_attention_mask: torch.Tensor,
        num_variants: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for local mode batch."""
        # Get delta embedding
        delta_embed = self._get_delta_embedding(
            ref_input_ids, ref_attention_mask, alt_input_ids, alt_attention_mask
        )

        # Add variant count information
        count_embed = self.variant_count_embedding(num_variants)
        combined = torch.cat([delta_embed, count_embed], dim=-1)
        patient_embed = self.combine_proj(combined)

        return patient_embed

    def _forward_aggregated_batch(
        self,
        ref_input_ids: torch.Tensor,
        ref_attention_mask: torch.Tensor,
        alt_input_ids: torch.Tensor,
        alt_attention_mask: torch.Tensor,
        variant_mask: torch.Tensor,
        num_variants: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for aggregated mode batch."""
        batch_size, max_variants, seq_len = ref_input_ids.shape

        # Process each variant to get delta embeddings
        # Reshape to process all variants at once
        ref_flat = ref_input_ids.view(batch_size * max_variants, seq_len)
        ref_mask_flat = ref_attention_mask.view(batch_size * max_variants, seq_len)
        alt_flat = alt_input_ids.view(batch_size * max_variants, seq_len)
        alt_mask_flat = alt_attention_mask.view(batch_size * max_variants, seq_len)

        # Get delta embeddings for all variants
        delta_embeds_flat = self._get_delta_embedding(
            ref_flat, ref_mask_flat, alt_flat, alt_mask_flat
        )

        # Reshape back to (batch_size, max_variants, hidden_size)
        variant_embeddings = delta_embeds_flat.view(batch_size, max_variants, self.hidden_size)

        # Aggregate using attention
        aggregated = self.attention_aggregator(variant_embeddings, variant_mask)

        # Add variant count information
        count_embed = self.variant_count_embedding(num_variants)
        combined = torch.cat([aggregated, count_embed], dim=-1)
        patient_embed = self.combine_proj(combined)

        return patient_embed

    def forward(
        self,
        # Local mode inputs
        local_ref_input_ids: Optional[torch.Tensor] = None,
        local_ref_attention_mask: Optional[torch.Tensor] = None,
        local_alt_input_ids: Optional[torch.Tensor] = None,
        local_alt_attention_mask: Optional[torch.Tensor] = None,
        local_num_variants: Optional[torch.Tensor] = None,
        local_disease_labels: Optional[torch.Tensor] = None,
        local_pathogenicity_labels: Optional[torch.Tensor] = None,
        # Aggregated mode inputs
        agg_ref_input_ids: Optional[torch.Tensor] = None,
        agg_ref_attention_mask: Optional[torch.Tensor] = None,
        agg_alt_input_ids: Optional[torch.Tensor] = None,
        agg_alt_attention_mask: Optional[torch.Tensor] = None,
        agg_variant_mask: Optional[torch.Tensor] = None,
        agg_num_variants: Optional[torch.Tensor] = None,
        agg_disease_labels: Optional[torch.Tensor] = None,
        agg_pathogenicity_labels: Optional[torch.Tensor] = None,
        # Batch flags
        has_local: bool = False,
        has_aggregated: bool = False,
        local_batch_size: int = 0,
        agg_batch_size: int = 0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass handling both local and aggregated batches.

        Returns dict with loss, logits, and optional embeddings.
        """
        all_patient_embeds = []
        all_disease_labels = []
        all_pathogenicity_labels = []

        # Process local batch
        if has_local and local_ref_input_ids is not None:
            local_embeds = self._forward_local_batch(
                local_ref_input_ids,
                local_ref_attention_mask,
                local_alt_input_ids,
                local_alt_attention_mask,
                local_num_variants,
            )
            all_patient_embeds.append(local_embeds)

            if local_disease_labels is not None:
                all_disease_labels.append(local_disease_labels)
            if local_pathogenicity_labels is not None:
                all_pathogenicity_labels.append(local_pathogenicity_labels)

        # Process aggregated batch
        if has_aggregated and agg_ref_input_ids is not None:
            agg_embeds = self._forward_aggregated_batch(
                agg_ref_input_ids,
                agg_ref_attention_mask,
                agg_alt_input_ids,
                agg_alt_attention_mask,
                agg_variant_mask,
                agg_num_variants,
            )
            all_patient_embeds.append(agg_embeds)

            if agg_disease_labels is not None:
                all_disease_labels.append(agg_disease_labels)
            if agg_pathogenicity_labels is not None:
                all_pathogenicity_labels.append(agg_pathogenicity_labels)

        if not all_patient_embeds:
            raise ValueError("No valid inputs provided")

        # Concatenate embeddings from both modes
        patient_embeds = torch.cat(all_patient_embeds, dim=0)

        # Apply task heads
        disease_logits = self.disease_head(patient_embeds)
        pathogenicity_logits = self.pathogenicity_head(patient_embeds)

        # Calculate losses
        loss = None
        disease_loss = None
        pathogenicity_loss = None

        if all_disease_labels:
            disease_labels = torch.cat(all_disease_labels, dim=0)
            disease_loss = self.disease_loss_fn(disease_logits, disease_labels)

        if all_pathogenicity_labels:
            pathogenicity_labels = torch.cat(all_pathogenicity_labels, dim=0)
            pathogenicity_loss = self.pathogenicity_loss_fn(
                pathogenicity_logits, pathogenicity_labels
            )

        # Combine losses
        if disease_loss is not None and pathogenicity_loss is not None:
            loss = (
                self.disease_weight * disease_loss
                + self.pathogenicity_weight * pathogenicity_loss
            )
        elif disease_loss is not None:
            loss = disease_loss
        elif pathogenicity_loss is not None:
            loss = pathogenicity_loss

        return {
            "loss": loss,
            "disease_logits": disease_logits,
            "pathogenicity_logits": pathogenicity_logits,
            "disease_loss": disease_loss,
            "pathogenicity_loss": pathogenicity_loss,
            "patient_embeddings": patient_embeds,
        }

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)

    def save_pretrained(self, save_directory, **kwargs):
        """Save the model."""
        import os

        self.base_model.save_pretrained(save_directory, **kwargs)

        # Save additional components
        torch.save({
            'attention_aggregator': self.attention_aggregator.state_dict(),
            'variant_count_embedding': self.variant_count_embedding.state_dict(),
            'combine_proj': self.combine_proj.state_dict(),
            'disease_head': self.disease_head.state_dict(),
            'pathogenicity_head': self.pathogenicity_head.state_dict(),
        }, os.path.join(save_directory, 'multi_variant_heads.bin'))

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        base_model_class,
        num_disease_classes: int = 4,
        num_pathogenicity_classes: int = 2,
        **kwargs,
    ):
        """Load model from pretrained."""
        import os

        base_model = base_model_class.from_pretrained(model_path)
        model = cls(
            base_model,
            num_disease_classes=num_disease_classes,
            num_pathogenicity_classes=num_pathogenicity_classes,
            **kwargs,
        )

        # Load additional components
        heads_path = os.path.join(model_path, 'multi_variant_heads.bin')
        if os.path.exists(heads_path):
            state_dict = torch.load(heads_path, map_location='cpu')
            model.attention_aggregator.load_state_dict(state_dict['attention_aggregator'])
            model.variant_count_embedding.load_state_dict(state_dict['variant_count_embedding'])
            model.combine_proj.load_state_dict(state_dict['combine_proj'])
            model.disease_head.load_state_dict(state_dict['disease_head'])
            model.pathogenicity_head.load_state_dict(state_dict['pathogenicity_head'])

        return model
