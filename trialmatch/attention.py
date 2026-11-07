"""Task 4: reference scaled dot-product attention and a tiled (blocked) attention with
online softmax, in plain PyTorch. Both support causal masking with a query offset
(queries at absolute positions q_offset .. q_offset+Tq-1 attending to keys 0..Tk-1),
which is exactly what a KV-cached decoder needs.

Shapes: q [B, H, Tq, D], k/v [B, H, Tk, D]  ->  out [B, H, Tq, D]
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def causal_mask(Tq: int, Tk: int, q_offset: int, device) -> torch.Tensor:
    """Boolean [Tq, Tk]; True where attention is NOT allowed (key position > query position)."""
    qpos = torch.arange(Tq, device=device) + q_offset
    kpos = torch.arange(Tk, device=device)
    return kpos[None, :] > qpos[:, None]


def reference_attention(q, k, v, causal: bool = True, q_offset: int = 0,
                        scale: Optional[float] = None) -> torch.Tensor:
    """Materialises the full [Tq, Tk] score matrix. Numerically stable softmax via max
    subtraction, computed in fp32 regardless of input dtype."""
    D = q.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32     # fp64 = ground truth mode
    s = torch.matmul(q.to(acc), k.to(acc).transpose(-1, -2)) * scale       # [B,H,Tq,Tk]
    if causal:
        s = s.masked_fill(causal_mask(q.shape[-2], k.shape[-2], q_offset, q.device), float("-inf"))
    m = s.amax(dim=-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))            # fully-masked rows
    p = torch.exp(s - m)
    l = p.sum(dim=-1, keepdim=True)
    out = torch.matmul(p, v.to(acc)) / l.clamp_min(1e-30)
    return out.to(q.dtype)


def tiled_attention(q, k, v, causal: bool = True, q_offset: int = 0,
                    scale: Optional[float] = None, block_q: int = 64, block_k: int = 64,
                    return_stats: bool = False):
    """Blocked attention with online softmax (FlashAttention-style recurrence).

    For each query block we stream over key blocks and keep, per query row:
      m  running max of scores seen so far
      l  running sum of exp(score - m)
      acc running sum of exp(score - m) * v
    When a new block raises the max to m_new we rescale the old state by
    alpha = exp(m_old - m_new). After the last block, out = acc / l.
    Only a [block_q, block_k] score tile is alive at any time. Key blocks that lie
    entirely in the causal future of the query block are skipped.
    """
    B, H, Tq, D = q.shape
    Tk = k.shape[-2]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    out = torch.empty((B, H, Tq, D), dtype=torch.float32, device=q.device)
    blocks_done = blocks_skipped = 0
    for qs in range(0, Tq, block_q):
        qe = min(qs + block_q, Tq)
        qb = q[:, :, qs:qe].float() * scale
        nq = qe - qs
        m = torch.full((B, H, nq, 1), float("-inf"), device=q.device)
        l = torch.zeros((B, H, nq, 1), device=q.device)
        acc = torch.zeros((B, H, nq, D), device=q.device)
        q_first_abs, q_last_abs = q_offset + qs, q_offset + qe - 1
        for ks in range(0, Tk, block_k):
            ke = min(ks + block_k, Tk)
            if causal and ks > q_last_abs:          # whole key block is in the future
                blocks_skipped += (Tk - ks + block_k - 1) // block_k
                break
            kb = k[:, :, ks:ke].float()
            vb = v[:, :, ks:ke].float()
            s = torch.matmul(qb, kb.transpose(-1, -2))                     # [B,H,nq,nk]
            if causal and (ke - 1) > q_first_abs:     # diagonal block: partially masked
                qpos = torch.arange(qs, qe, device=q.device) + q_offset
                kpos = torch.arange(ks, ke, device=q.device)
                s = s.masked_fill(kpos[None, :] > qpos[:, None], float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
            p = torch.exp(s - m_safe)                                      # masked -> 0
            alpha = torch.exp(m - m_safe)                                  # m=-inf -> 0
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            acc = alpha * acc + torch.matmul(p, vb)
            m = m_new
            blocks_done += 1
        out[:, :, qs:qe] = acc / l.clamp_min(1e-30)
    out = out.to(q.dtype)
    if return_stats:
        return out, {"blocks_computed": blocks_done, "blocks_skipped": blocks_skipped}
    return out


def torch_sdpa(q, k, v, causal: bool = True, q_offset: int = 0, scale: Optional[float] = None):
    """PyTorch's fused scaled_dot_product_attention (flash / mem-efficient / math backend
    chosen by PyTorch). Used only as a speed/accuracy reference, never as 'our' kernel."""
    Tq, Tk = q.shape[-2], k.shape[-2]
    if not causal or (Tq == 1 and q_offset == Tk - 1):
        return F.scaled_dot_product_attention(q, k, v, scale=scale)
    if q_offset == 0 and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    allowed = ~causal_mask(Tq, Tk, q_offset, q.device)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=scale)


ATTENTION_IMPLS = {"reference": reference_attention, "tiled": tiled_attention, "torch_sdpa": torch_sdpa}
