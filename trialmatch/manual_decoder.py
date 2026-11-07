"""Task 3 (and the engine for Tasks 4-5): a from-scratch forward pass for Qwen2-style
decoders (RMSNorm, RoPE, grouped-query attention, SwiGLU MLP) that reuses the HF
model's weights but owns the KV cache, positions, causal masking and attention kernel.

Why: the assignment requires the cache-aware decoding flow to be implemented and
examined, not switched on with use_cache=True. Every step below is explicit.

KV cache layout (per layer): K, V tensors of shape [B, n_kv_heads, capacity, head_dim],
pre-allocated once. `length` is the number of valid positions. Writing T new tokens
copies their K/V into slots [length, length+T); rollback (speculative decoding) is just
setting `length` back, because stale slots are never read (attention only looks at
[0, length)). When capacity is exceeded the buffers grow by doubling (amortised O(1)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .attention import ATTENTION_IMPLS


@dataclass
class KVCache:
    k: List[torch.Tensor]
    v: List[torch.Tensor]
    length: int = 0
    grow_events: int = 0

    @property
    def capacity(self) -> int:
        return self.k[0].shape[2]

    def ensure(self, needed: int):
        if needed <= self.capacity:
            return
        new_cap = max(needed, 2 * self.capacity)
        for i in range(len(self.k)):
            for buf in (self.k, self.v):
                old = buf[i]
                nb = old.new_zeros(old.shape[0], old.shape[1], new_cap, old.shape[3])
                nb[:, :, :self.length] = old[:, :, :self.length]
                buf[i] = nb
        self.grow_events += 1

    def crop(self, n: int):
        """Roll back to the first n positions (used by speculative decoding)."""
        assert 0 <= n <= self.length, (n, self.length)
        self.length = n

    def nbytes_used(self) -> int:
        per = self.k[0].element_size() * self.k[0].shape[0] * self.k[0].shape[1] * self.k[0].shape[3]
        return 2 * len(self.k) * per * self.length

    def nbytes_allocated(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


class ManualDecoder:
    def __init__(self, model, attn_impl: str = "torch_sdpa", block_q: int = 64, block_k: int = 64):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.model = base
        cfg = base.config
        self.cfg = cfg
        if getattr(cfg, "rope_scaling", None) not in (None, {}) and \
                (cfg.rope_scaling or {}).get("rope_type", cfg.rope_scaling.get("type")) not in (None, "default"):
            raise NotImplementedError(f"rope_scaling={cfg.rope_scaling} not supported by ManualDecoder")
        self.inner = base.model
        self.layers = self.inner.layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.groups = self.n_heads // self.n_kv
        self.device = next(base.parameters()).device
        self.dtype = next(base.parameters()).dtype
        theta = getattr(cfg, "rope_theta", 10000.0)
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, device=self.device,
                                                      dtype=torch.float32) / self.head_dim))
        self.set_attention(attn_impl, block_q, block_k)

    def set_attention(self, impl: str, block_q: int = 64, block_k: int = 64):
        if impl not in ATTENTION_IMPLS:
            raise ValueError(impl)
        self.attn_name = impl
        fn = ATTENTION_IMPLS[impl]
        if impl == "tiled":
            self.attn = lambda q, k, v, q_offset: fn(q, k, v, causal=True, q_offset=q_offset,
                                                     block_q=block_q, block_k=block_k)
        else:
            self.attn = lambda q, k, v, q_offset: fn(q, k, v, causal=True, q_offset=q_offset)

    # ---------------------------------------------------------------- cache
    def new_cache(self, batch: int = 1, capacity: int = 512) -> KVCache:
        shape = (batch, self.n_kv, capacity, self.head_dim)
        L = len(self.layers)
        return KVCache(k=[torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in range(L)],
                       v=[torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in range(L)])

    # ---------------------------------------------------------------- RoPE
    def _rope(self, positions: torch.Tensor):
        freqs = positions.float()[:, None] * self.inv_freq[None, :]      # [T, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)                          # [T, D]
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, x, cos, sin):
        return x * cos[None, None] + self._rotate_half(x) * sin[None, None]

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, cache: Optional[KVCache] = None,
                last_only: bool = False) -> torch.Tensor:
        """input_ids [B, T]. With a cache, the new tokens take absolute positions
        cache.length .. cache.length+T-1 and attend to all cached positions plus
        themselves (causal). Without a cache, positions are 0..T-1 (full recompute)."""
        B, T = input_ids.shape
        p0 = cache.length if cache is not None else 0
        if cache is not None:
            cache.ensure(p0 + T)
        positions = torch.arange(p0, p0 + T, device=self.device)
        cos, sin = self._rope(positions)
        h = self.inner.embed_tokens(input_ids)
        for li, layer in enumerate(self.layers):
            att = layer.self_attn
            x = layer.input_layernorm(h)
            q = att.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = att.k_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            v = att.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)     # keys are cached *after* RoPE
            if cache is not None:
                cache.k[li][:, :, p0:p0 + T] = k
                cache.v[li][:, :, p0:p0 + T] = v
                K = cache.k[li][:, :, :p0 + T]
                V = cache.v[li][:, :, :p0 + T]
            else:
                K, V = k, v
            if self.groups > 1:                   # grouped-query attention: share KV heads
                K = K.repeat_interleave(self.groups, dim=1)
                V = V.repeat_interleave(self.groups, dim=1)
            o = self.attn(q, K, V, p0)            # causal with query offset p0
            o = o.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
            h = h + att.o_proj(o)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        if cache is not None:
            cache.length = p0 + T
        if last_only:
            h = h[:, -1:]
        h = self.inner.norm(h)
        return self.model.lm_head(h).float()
