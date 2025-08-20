from __future__ import annotations
from typing import Optional, Tuple, Sequence, Union, Type, List
import math

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)
import torch.nn.functional as F

# -------------------------------------------------------------------------
#  Shared utility: Advanced adaptive compressor with global context tokens
# -------------------------------------------------------------------------
class AdaptiveCompressor(nn.Module):
    """
    Enhanced adaptive compressor that adds a global context token for each segment
    in addition to the sampled tokens. This helps retain more contextual information
    during compression.
    """
    def __init__(
        self,
        dim: int,
        gumbel_tau: float = 1.0,
        hidden_dim: int = 32,
        debug: bool = True,
        print_every: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(2 * dim, dim)

        # Global context projector - transforms segment embeddings into a global token
        self.global_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # ---- learnable bias ------------------------------------------------
        # self.bias_mlp = nn.Sequential(
        #     nn.Linear(1, hidden_dim), nn.GELU(),
        #     nn.Linear(hidden_dim, 1, bias=False)
        # )
        self.bias_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        # self.bias_mlp = nn.Sequential(
        #     nn.Linear(1, hidden_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim // 2, 1, bias=False)
        # )
        self.alpha = nn.Parameter(torch.tensor(0.1))  # convex-mix gate
        self.gumbel_tau = gumbel_tau

        # ---- DEBUG infra ---------------------------------------------------
        self._debug = debug
        self._print_every = max(1, print_every)
        self._step = 0
        self._prev = {n: p.detach().clone() for n, p in self.named_parameters()}

        if self._debug:
            self._register_grad_hooks()
        self.freeze_after =5000
        self._frozen = False

    # ------------- debugging helpers ---------------------------------------
    def _register_grad_hooks(self):
        for name, p in self.named_parameters():

            def _hook(grad, n=name):
                if self._step % self._print_every == 0:
                    tag = "None" if grad is None else f"{grad.norm():.4g}"
                    print(f"[compressor] step={self._step:<5} {n:25} grad={tag}")

            p.register_hook(_hook)

    def _report_weight_updates(self):
        if not (self._debug and self._step % self._print_every == 0):
            return
        for n, p in self.named_parameters():
            delta = (p - self._prev[n].to(p.device)).abs().max().item()
            if delta != 0:
                print(f"[compressor] step={self._step:<5} {n:25} Δ={delta:.4e}")
            self._prev[n] = p.detach().clone()

    # ------------- weight mixing: uniform ↔ learned bias -------------------
    def _weights(self, start: int, end: int, device) -> torch.Tensor:
        L = end - start
        if L <= 0:
            return torch.ones(1, device=device)

        uni = torch.full((L,), 1.0 / L, device=device)

        pos = torch.linspace(0, 1, L, device=device).unsqueeze(-1)
        bias = self.bias_mlp(pos).squeeze(-1)
        bias = bias - bias.logsumexp(0)          # log-softmax → prob

        α = torch.sigmoid(self.alpha)
        w = (1 - α) * uni + α * bias.exp()
        return w / w.sum()

    # ------------- deterministic indices (Adaptive style) ------------------
    @staticmethod
    def _pick_indices(L: int, k: int, device) -> torch.Tensor:
        if k == 1:
            return torch.tensor([L // 2], device=device)
        return torch.linspace(0, L - 1, steps=k, device=device).long()

    # ------------- differentiable sampler ----------------------------------
    def _sample(self, seg: torch.Tensor, w: torch.Tensor, k: int) -> torch.Tensor:
        B, L, D = seg.shape
        if k >= L:
            return seg

        # Create global context token that summarizes the entire segment
        global_ctx = self.global_proj(seg.mean(dim=1, keepdim=True))  # [B, 1, D]

        if self.training:
            logits = torch.log(w + 1e-9).expand(k, L)
            g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
            y_soft = F.softmax((logits + g) / self.gumbel_tau, dim=-1)
            hard_idx = y_soft.argmax(-1, keepdim=True)
            y_hard = torch.zeros_like(y_soft).scatter_(-1, hard_idx, 1.0)
            y = (y_hard - y_soft).detach() + y_soft
            sampled = torch.einsum('kl,lbd->kbd', y, seg.transpose(0, 1)).transpose(0, 1)

            # Concatenate sampled tokens with global context token
            return torch.cat([global_ctx, sampled], dim=1)  # [B, k+1, D]

        idx = self._pick_indices(L, k, seg.device)
        sampled = seg[:, idx, :]

        # Concatenate sampled tokens with global context token
        return torch.cat([global_ctx, sampled], dim=1)  # [B, k+1, D]

    def _maybe_freeze(self):
        if (
            self.freeze_after is not None
            and not self._frozen
            and self._step >= self.freeze_after
        ):
            for p in self.parameters():
                p.requires_grad = False
            self._frozen = True
            if self._debug:
                print(f"[compressor] 🔒 parameters frozen at step {self._step}")

    # ------------------------- forward -------------------------------------
    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        self._step += 1
        self._maybe_freeze()
        self._report_weight_updates()

        B, N, D = x.shape
        if N <= target_len:
            return x

        first = x[:, :1, :]
        budget = target_len - 1
        if budget <= 0:
            return first

        # We need to account for the additional global context token per segment
        # Adjust segments calculation to account for the global tokens we'll add
        segments = self._calculate_adaptive_segments(N - 1, budget)

        out, ptr = [first], 1
        for seg_len, want in segments:
            if ptr >= N:
                break
            end = min(ptr + seg_len, N)
            seg = x[:, ptr:end, :]
            L = seg.size(1)

            # Allocate one token less per segment to make room for global token
            k = int(min(max(1, round(want) - 1), L))
            if k <= 0:  # If we can't allocate any regular tokens, still keep the global one
                k = 1

            w = self._weights(ptr, end, x.device)
            segment_out = self._sample(seg, w, k)  # Now returns k+1 tokens (k sampled + 1 global)
            out.append(segment_out)

            ptr = end

        compressed = torch.cat(out, dim=1)
        # Ensure we don't exceed target length
        return compressed[:, :target_len, :]

    # -------- segment scheduler with adjustment for global tokens ----------
    def _calculate_adaptive_segments(self, orig_len: int, target_len: int):
        if target_len >= orig_len:
            return [(orig_len, 1.0)]
        segments, ratio = [], target_len / orig_len
        cur, rem, taken = 1, orig_len, 0
        while rem > 0 and taken < target_len:
            size = min(cur, rem)
            imp = 2.0 - (1.5 * taken / target_len)

            # Need to adjust for the fact that each segment will now use one extra token
            # for the global context representation
            need = max(1.0, size * ratio * imp)

            if size <= 3:
                need = min(size, math.ceil(need))
            segments.append((size, need))
            rem -= size
            taken += math.ceil(need)
            cur *= 2

        # If we've exceeded our budget, reduce tokens from segments
        if taken > target_len:
            extra = taken - target_len
            for i in range(len(segments) - 1, -1, -1):
                size, need = segments[i]
                drop = min(extra, math.ceil(need) - 1)
                if drop:
                    segments[i] = (size, need - drop)
                    extra -= drop
                if not extra:
                    break
        return segments

# -------------------------------------------------------------------------
#  Base mixin with compression logic (not a HF model by itself)
# -------------------------------------------------------------------------
class _FramePackMixin:
    keep_last: int
    native_ctx: int
    compressor: AdaptiveCompressor

    def _infer_ctx_len(self) -> int:
        for field in (
            "max_position_embeddings",
            "n_positions",
            "max_seq_len",
            "block_size",
        ):
            if hasattr(self.base.config, field):
                return int(getattr(self.base.config, field))
        tok_len = getattr(self.tokenizer, "model_max_length", 0)
        return (
            int(tok_len)
            if 0 < tok_len < 10_000_000
            else 2048  # fallback for RoPE / ALiBi
        )

    def _compress(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input_ids.device
        embed_fn = self.base.get_input_embeddings()
        embeds = embed_fn(input_ids)                       # (B,L,D)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        # Split into history and recent tokens
        recent_emb = embeds[:, -self.keep_last:, :]
        recent_mask = attention_mask[:, -self.keep_last:]
        hist_emb = embeds[:, :-self.keep_last, :]
        hist_mask = attention_mask[:, :-self.keep_last]

        # Calculate how much history we can keep
        history_budget = max(0, self.native_ctx - recent_emb.size(1))

        if hist_emb.size(1) > history_budget and history_budget > 0:
            # Use adaptive compressor (which will preserve the first token)
            hist_emb = self.compressor(hist_emb, history_budget)

            # Create corresponding attention mask
            hist_mask = torch.ones((hist_emb.size(0), hist_emb.size(1)),
                                  dtype=attention_mask.dtype,
                                  device=device)

        # Combine compressed history with recent tokens
        emb_final = torch.cat([hist_emb, recent_emb], dim=1) if hist_emb.size(1) > 0 else recent_emb
        mask_final = torch.cat([hist_mask, recent_mask], dim=1) if hist_mask.size(1) > 0 else recent_mask

        return emb_final, mask_final


# -------------------------------------------------------------------------
#  FramePack for CAUSAL-LM  (keeps generate API)
# -------------------------------------------------------------------------
class FramePackCausalLM(_FramePackMixin, PreTrainedModel):
    _no_split_modules = ["AdaptiveCompressor"]

    def __init__(
        self,
        base_model: PreTrainedModel | str,
        keep_last: int = 1024,
        native_ctx_len: Optional[int] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        return_mask = False
    ):
        if isinstance(base_model, str):
            base_model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
        super().__init__(base_model.config)
        self.base = base_model
        self.keep_last = keep_last
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            base_model.config._name_or_path, trust_remote_code=True
        )
        self.dim = self.base.get_input_embeddings().embedding_dim
        self.native_ctx = native_ctx_len or self._infer_ctx_len()
        self.compressor = AdaptiveCompressor(self.dim).to(base_model.device)
        self.return_mask = return_mask

    # HF generation hooks --------------------------------------------------
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs,
    ):
        embeds, mask = self._compress(input_ids, attention_mask)
        print(f"after compress: {embeds.shape}")
        return {
            "inputs_embeds": embeds,
            "attention_mask": mask,
            "past_key_values": None,
            "use_cache": False,
        }

    def _reorder_cache(self, past, beam_idx):
        return past  # no KV cache

    # forward --------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        cache_position: Optional[object] = None,
    ):
        if inputs_embeds is None:
            inputs_embeds, attention_mask = self._compress(
                input_ids, attention_mask
            )
        return self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )


# -------------------------------------------------------------------------
#  FramePack for SEQUENCE-CLASSIFICATION
# -------------------------------------------------------------------------
class FramePackSequenceClassifier(_FramePackMixin, PreTrainedModel):
    _no_split_modules = ["AdaptiveCompressor"]

    def __init__(
        self,
        base_model: PreTrainedModel | str,
        num_labels: Optional[int] = None,
        keep_last: int = 1024,
        native_ctx_len: Optional[int] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        return_mask = False,
    ):
        if isinstance(base_model, str):
            cfg = AutoConfig.from_pretrained(base_model, num_labels=num_labels, trust_remote_code=True)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model, config=cfg, trust_remote_code=True
            )
        super().__init__(base_model.config)
        self.base = base_model
        self.keep_last = keep_last
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            base_model.config._name_or_path, trust_remote_code=True
        )
        self.dim = self.base.get_input_embeddings().embedding_dim
        self.native_ctx = native_ctx_len or self._infer_ctx_len()
        self.compressor = AdaptiveCompressor(self.dim).to(base_model.device)
        self.return_mask = return_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[object] = None,
    ):
        if inputs_embeds is None:
            inputs_embeds, attention_mask = self._compress(
                input_ids, attention_mask
            )
        if self.return_mask:
            out = self.base(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=output_hidden_states,
                )
            out["attention_mask"] = attention_mask
            return out
        return self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )


# -------------------------------------------------------------------------
#  Minimal demos
# -------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- LM demo ---------------------------------------------------
    lm = FramePackCausalLM('zehui127/Omni-DNA-116M', keep_last=1024,native_ctx_len=5096).to(device)
    tok = lm.tokenizer
    prompt = ("GCGTGGAGAG" * 40000).strip()  # ≈ 10k tokens
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    print(f"input_ids: {input_ids.shape}")
    # out = lm.generate(input_ids, max_new_tokens=50)
    out = lm.forward(input_ids)
    print(f"output_ids: {out.logits.shape}")
    print(f"output_ids: {out.loss.shape}")
    # print(tok.decode(out[0, -100:], skip_special_tokens=True))

    # ---------- classifier demo ------------------------------------------
    # clf = FramePackSequenceClassifier(
    #     "zehui127/Omni-DNA-116M", num_labels=2,  keep_last=70,native_ctx_len=256,
    # ).to(device)
    # tok_clf = clf.tokenizer
    # long_text = "AGAGAGATAGAGAAG" * 1000  # > context length
    # ids = tok_clf(long_text, return_tensors="pt").input_ids.to(device)
    # output = clf(ids)
    # logits = clf(ids).logits
    # print("Output:", output.keys())
    # print(f"Logits: {logits.shape}")
    # print(clf.base_model.main_input_name)
