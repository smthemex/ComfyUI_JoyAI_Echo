from enum import Enum
from typing import Protocol

import torch

from .rope_ import LTXRopeType, apply_rotary_emb

memory_efficient_attention = None
flash_attn_interface = None
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None
try:
    # FlashAttention3 and XFormersAttention cannot be used together
    if memory_efficient_attention is None:
        import flash_attn_interface
except ImportError:
    flash_attn_interface = None


def _slice_rope(
    pe: tuple[torch.Tensor, torch.Tensor], start: int, end: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return pe[0][..., start:end, :], pe[1][..., start:end, :]


def update_kv_cache(
    cache: dict,
    start: int,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert one global token range and return the active sink+FIFO window.

    The cache is deliberately inference-only. Repeated denoising forwards for
    the same block replace that block in place; the clean refresh forward then
    replaces it once more before the next block starts.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("causal KV caches are inference-only")
    if k.shape != v.shape or k.ndim != 3:
        raise ValueError("KV tensors must have matching [batch, tokens, dim] shapes")

    length = int(cache["length"])
    old_positions = cache["positions"][:length]
    old_k = cache["k"][:, :length]
    old_v = cache["v"][:, :length]
    end = start + k.shape[1]

    # A denoising step is a transaction over [start, end): discard a previous
    # noisy version of that range while retaining earlier committed history.
    keep_old = old_positions < start
    positions = torch.cat(
        [old_positions[keep_old], torch.arange(start, end, device=k.device)], dim=0
    )
    merged_k = torch.cat([old_k[:, keep_old], k], dim=1)
    merged_v = torch.cat([old_v[:, keep_old], v], dim=1)

    local = int(cache.get("local_attn_size", -1))
    sink = int(cache.get("sink_tokens", 0))
    if local >= 0 and positions.numel() > local:
        if not 0 <= sink < local:
            raise ValueError(f"expected 0 <= sink_tokens < local_attn_size, got {sink}/{local}")
        sink_mask = positions < sink
        recent_budget = local - int(sink_mask.sum())
        recent_start = max(sink, end - recent_budget)
        keep = sink_mask | (positions >= recent_start)
        positions = positions[keep]
        merged_k = merged_k[:, keep]
        merged_v = merged_v[:, keep]

    active = positions.numel()
    if active > cache["k"].shape[1]:
        raise ValueError(f"KV cache overflow: {active} active tokens exceed capacity {cache['k'].shape[1]}")
    cache["k"][:, :active].copy_(merged_k)
    cache["v"][:, :active].copy_(merged_v)
    cache["positions"][:active].copy_(positions)
    cache["length"] = active
    return cache["k"][:, :active].clone(), cache["v"][:, :active].clone()


class AttentionCallable(Protocol):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...


class PytorchAttention(AttentionCallable):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        if mask is not None:
            # add a batch dimension if there isn't already one
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a heads dimension if there isn't already one
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class XFormersAttention(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_efficient_attention is None:
            raise RuntimeError("XFormersAttention was selected but `xformers` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        # xformers expects [B, M, H, K]
        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            # add a singleton batch dimension
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a singleton heads dimension
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            # pad to a multiple of 8
            pad = 8 - mask.shape[-1] % 8
            # the xformers docs says that it's allowed to have a mask of shape (1, Nq, Nk)
            # but when using separated heads, the shape has to be (B, H, Nq, Nk)
            # in flux, this matrix ends up being over 1GB
            # here, we create a mask with the same batch/head size as the input mask (potentially singleton or full)
            mask_out = torch.empty(
                [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device
            )

            mask_out[..., : mask.shape[-1]] = mask
            # doesn't this remove the padding again??
            mask = mask_out[..., : mask.shape[-1]]
            mask = mask.expand(b, heads, -1, -1)

        out = memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, attn_bias=mask, p=0.0)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention3(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flash_attn_interface is None:
            raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            raise NotImplementedError("Mask is not supported for FlashAttention3")

        out = flash_attn_interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_3 = "flash_attention_3"
    DEFAULT = "default"

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self is AttentionFunction.PYTORCH:
            return PytorchAttention()(q, k, v, heads, mask)
        elif self is AttentionFunction.XFORMERS:
            return XFormersAttention()(q, k, v, heads, mask)
        elif self is AttentionFunction.FLASH_ATTENTION_3:
            return FlashAttention3()(q, k, v, heads, mask)
        else:
            # Default behavior: XFormers if installed else - PyTorch
            return (
                XFormersAttention()(q, k, v, heads, mask)
                if memory_efficient_attention is not None
                else PytorchAttention()(q, k, v, heads, mask)
            )


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        attention_function: AttentionCallable | AttentionFunction = AttentionFunction.DEFAULT,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = attention_function

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        # Optional per-head gating
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
        kv_cache: dict | None = None,
        kv_cache_start: int = 0,
        crossattn_cache: dict | None = None,
    ) -> torch.Tensor:
        """Multi-head attention with optional RoPE, perturbation masking, and per-head gating.
        When ``perturbation_mask`` is all zeros, the expensive query/key path
        (linear projections, RMSNorm, RoPE) is skipped entirely and only the
        value projection is used as a pass-through.
        Args:
            x: Query input tensor of shape ``(B, T, query_dim)``.
            context: Key/value context tensor of shape ``(B, S, context_dim)``.
                Falls back to ``x`` (self-attention) when *None*.
            mask: Optional attention mask. Interpretation depends on the attention
                backend (additive bias for xformers/PyTorch SDPA).
            pe: Rotary positional embeddings applied to both ``q`` and ``k``.
            k_pe: Separate rotary positional embeddings for ``k`` only. When
                *None*, ``pe`` is reused for keys.
            perturbation_mask: Optional mask in ``[0, 1]`` that
                blends the attention output with the raw value projection:
                ``out = attn_out * mask + v * (1 - mask)``.
                **1** keeps the full attention output, **0** bypasses attention
                and passes the value projection through unchanged.
                *None* or all-ones means standard attention; all-zeros skips
                the query/key path entirely for efficiency.
            all_perturbed: Whether all perturbations are active for this block.
        Returns:
            Output tensor of shape ``(B, T, query_dim)``.
        """
        context = x if context is None else context
        use_attention = not all_perturbed

        v = self.to_v(context)

        if not use_attention:
            out = v
        else:
            q = self.q_norm(self.to_q(x))
            if crossattn_cache is not None:
                if not crossattn_cache["is_init"]:
                    cached_k = self.k_norm(self.to_k(context))
                    size = cached_k.shape[1]
                    crossattn_cache["k"][:, :size].copy_(cached_k)
                    crossattn_cache["v"][:, :size].copy_(v)
                    crossattn_cache["length"] = size
                    crossattn_cache["is_init"] = True
                size = int(crossattn_cache["length"])
                k = crossattn_cache["k"][:, :size]
                v = crossattn_cache["v"][:, :size]
                if pe is not None:
                    q = apply_rotary_emb(q, pe, self.rope_type)
            else:
                k = self.k_norm(self.to_k(context))
                local_pe = kv_cache.get("local_rope_pe") if kv_cache is not None else None
                local_q_pe = kv_cache.get("local_cross_q_rope_pe") if kv_cache is not None else None
                local_k_pe = kv_cache.get("local_cross_k_rope_pe") if kv_cache is not None else None
                if local_pe is not None:
                    k, v = update_kv_cache(kv_cache, kv_cache_start, k, v)
                    active = k.shape[1]
                    q_len = q.shape[1]
                    q = apply_rotary_emb(q, _slice_rope(local_pe, active - q_len, active), self.rope_type)
                    k = apply_rotary_emb(k, _slice_rope(local_pe, 0, active), self.rope_type)
                elif local_q_pe is not None or local_k_pe is not None:
                    if local_q_pe is None or local_k_pe is None:
                        raise ValueError("cross-modal RoPE rebase requires both query and key templates")
                    new_keys = k.shape[1]
                    k, v = update_kv_cache(kv_cache, kv_cache_start, k, v)
                    query_slice = kv_cache["local_cross_q_slices"].get((kv_cache_start, kv_cache_start + new_keys))
                    if query_slice is None:
                        raise ValueError("missing local cross-modal query RoPE slice")
                    q = apply_rotary_emb(q, _slice_rope(local_q_pe, *query_slice), self.rope_type)
                    k = apply_rotary_emb(k, _slice_rope(local_k_pe, 0, k.shape[1]), self.rope_type)
                else:
                    if pe is not None:
                        q = apply_rotary_emb(q, pe, self.rope_type)
                        k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)
                    if kv_cache is not None:
                        k, v = update_kv_cache(kv_cache, kv_cache_start, k, v)

            out = self.attention_function(q, k, v, self.heads, mask)  # (B, T, H*D)

            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

        # Apply per-head gating if enabled
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)  # (B, T, H)
            b, t, _ = out.shape
            # Reshape to (B, T, H, D) for per-head gating
            out = out.view(b, t, self.heads, self.dim_head)
            # Apply gating: 2 * sigmoid(x) so that zero-init gives identity (2 * 0.5 = 1.0)
            gates = 2.0 * torch.sigmoid(gate_logits)  # (B, T, H)
            out = out * gates.unsqueeze(-1)  # (B, T, H, D) * (B, T, H, 1)
            # Reshape back to (B, T, H*D)
            out = out.view(b, t, self.heads * self.dim_head)

        return self.to_out(out)
