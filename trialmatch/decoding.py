"""Task 3: greedy decoding paths with and without KV-cache reuse, instrumented timing.

  hf_uncached     : HF forward on the FULL sequence every step (reference, O(n^2) recompute)
  hf_cached       : HF forward with an explicit DynamicCache that we pass in and advance
                    ourselves, together with explicit position_ids and attention_mask
  manual_uncached : our ManualDecoder, full recompute every step
  manual_cached   : our ManualDecoder with our own pre-allocated KV cache
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional

import torch

from .manual_decoder import ManualDecoder


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __init__(self):
        _sync()
        self.t = time.perf_counter()

    def lap(self) -> float:
        _sync()
        now = time.perf_counter()
        d, self.t = now - self.t, now
        return d


def _result(tokens, prefill, decode_times, extra=None):
    n = len(tokens)
    dec = sum(decode_times)
    r = {"tokens": tokens, "n_new_tokens": n, "prefill_s": prefill, "decode_s": dec,
         "total_s": prefill + dec,
         # first token comes out of prefill; decode tok/s counts the remaining n-1 tokens
         "decode_tokens_per_s": (n - 1) / dec if n > 1 and dec > 0 else None,
         "e2e_tokens_per_s": n / (prefill + dec) if prefill + dec > 0 else None,
         "per_step_s": decode_times}
    if extra:
        r.update(extra)
    return r


def _stop(tok_id, stops):
    return stops is not None and tok_id in stops


@torch.no_grad()
def hf_uncached(model, input_ids: torch.Tensor, max_new_tokens: int, stops: Optional[Iterable[int]] = None,
                ignore_eos: bool = False) -> Dict:
    ids = input_ids
    out, times = [], []
    tm = Timer()
    prefill = None
    for step in range(max_new_tokens):
        logits = model(input_ids=ids, use_cache=False).logits[:, -1, :]
        nxt = int(logits.argmax(-1)[0])
        dt = tm.lap()
        if step == 0:
            prefill = dt
        else:
            times.append(dt)
        out.append(nxt)
        if not ignore_eos and _stop(nxt, stops):
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return _result(out, prefill, times, {"mode": "hf_uncached"})


@torch.no_grad()
def hf_cached(model, input_ids: torch.Tensor, max_new_tokens: int, stops: Optional[Iterable[int]] = None,
              ignore_eos: bool = False) -> Dict:
    """Prefill once, then feed ONE token per step with the cache from the previous step.
    position_ids are the absolute positions of the new token; attention_mask covers
    cached + new positions (all valid, batch of 1, no padding). We assert the cache grows
    by exactly one position per step."""
    from transformers import DynamicCache
    dev = input_ids.device
    L = input_ids.shape[1]
    cache = DynamicCache()
    tm = Timer()
    o = model(input_ids=input_ids, attention_mask=torch.ones(1, L, dtype=torch.long, device=dev),
              position_ids=torch.arange(L, device=dev)[None], past_key_values=cache, use_cache=True)
    cache = o.past_key_values
    nxt = int(o.logits[:, -1, :].argmax(-1)[0])
    prefill = tm.lap()
    out, times, cache_lens = [nxt], [], [cache.get_seq_length()]
    assert cache_lens[0] == L, (cache_lens[0], L)
    cur = L
    while len(out) < max_new_tokens and (ignore_eos or not _stop(nxt, stops)):
        o = model(input_ids=torch.tensor([[nxt]], device=dev),
                  attention_mask=torch.ones(1, cur + 1, dtype=torch.long, device=dev),
                  position_ids=torch.tensor([[cur]], device=dev),
                  cache_position=torch.tensor([cur], device=dev),
                  past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        cur += 1
        cl = cache.get_seq_length()
        assert cl == cur, f"cache length {cl} != expected {cur}"
        cache_lens.append(cl)
        nxt = int(o.logits[:, -1, :].argmax(-1)[0])
        times.append(tm.lap())
        out.append(nxt)
    return _result(out, prefill, times, {"mode": "hf_cached", "cache_lengths": cache_lens})


@torch.no_grad()
def manual_uncached(dec: ManualDecoder, input_ids, max_new_tokens, stops=None, ignore_eos=False) -> Dict:
    ids = input_ids
    out, times = [], []
    tm = Timer()
    prefill = None
    for step in range(max_new_tokens):
        nxt = int(dec.forward(ids, cache=None, last_only=True)[:, -1].argmax(-1)[0])
        dt = tm.lap()
        if step == 0:
            prefill = dt
        else:
            times.append(dt)
        out.append(nxt)
        if not ignore_eos and _stop(nxt, stops):
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return _result(out, prefill, times, {"mode": f"manual_uncached[{dec.attn_name}]"})


@torch.no_grad()
def manual_cached(dec: ManualDecoder, input_ids, max_new_tokens, stops=None, ignore_eos=False,
                  return_logits: bool = False) -> Dict:
    L = input_ids.shape[1]
    cache = dec.new_cache(1, capacity=L + max_new_tokens)
    tm = Timer()
    logits = dec.forward(input_ids, cache=cache, last_only=True)
    nxt = int(logits[:, -1].argmax(-1)[0])
    prefill = tm.lap()
    out, times, all_logits = [nxt], [], [logits[0, -1].cpu()] if return_logits else None
    while len(out) < max_new_tokens and (ignore_eos or not _stop(nxt, stops)):
        logits = dec.forward(torch.tensor([[nxt]], device=dec.device), cache=cache, last_only=True)
        nxt = int(logits[:, -1].argmax(-1)[0])
        times.append(tm.lap())
        out.append(nxt)
        if return_logits:
            all_logits.append(logits[0, -1].cpu())
    extra = {"mode": f"manual_cached[{dec.attn_name}]", "cache_len_final": cache.length,
             "cache_bytes_used": cache.nbytes_used(), "cache_bytes_allocated": cache.nbytes_allocated(),
             "cache_grow_events": cache.grow_events}
    if return_logits:
        extra["logits"] = all_logits
    return _result(out, prefill, times, extra)


def first_mismatch(a: List[int], b: List[int]) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


@torch.no_grad()
def investigate_mismatch(model, input_ids, ref_tokens, other_tokens, pos) -> Dict:
    """At the first divergent position, compute the reference logits (uncached full
    forward on the shared prefix) and report the top-2 margin. A tiny margin means the
    divergence is a floating-point near-tie (different kernel shapes / accumulation order),
    not a cache bug."""
    prefix = torch.cat([input_ids, torch.tensor([ref_tokens[:pos]], device=input_ids.device,
                                                 dtype=input_ids.dtype)], dim=1) if pos else input_ids
    logits = model(input_ids=prefix, use_cache=False).logits[0, -1].float()
    top2 = torch.topk(logits, 2)
    return {"position": pos, "ref_token": ref_tokens[pos] if pos < len(ref_tokens) else None,
            "other_token": other_tokens[pos] if pos < len(other_tokens) else None,
            "top1_logit": float(top2.values[0]), "top2_logit": float(top2.values[1]),
            "margin": float(top2.values[0] - top2.values[1]),
            "logit_ref_token": float(logits[ref_tokens[pos]]) if pos < len(ref_tokens) else None,
            "logit_other_token": float(logits[other_tokens[pos]]) if pos < len(other_tokens) else None}


def measure(fn, *args, warmup: int = 1, repeats: int = 3, **kw) -> Dict:
    """Warm-up runs (CUDA kernel selection, allocator), then repeated synchronised runs.
    Reports the median of each timing and the peak allocated memory."""
    import statistics
    for _ in range(warmup):
        fn(*args, **kw)
    runs = []
    peak = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    for _ in range(repeats):
        runs.append(fn(*args, **kw))
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
    med = lambda k: statistics.median([r[k] for r in runs if r[k] is not None]) \
        if any(r[k] is not None for r in runs) else None
    r0 = dict(runs[0])
    r0.pop("per_step_s", None)
    r0.pop("logits", None)
    r0.update({k: med(k) for k in ("prefill_s", "decode_s", "total_s", "decode_tokens_per_s", "e2e_tokens_per_s")})
    r0["repeats"] = repeats
    r0["all_total_s"] = [r["total_s"] for r in runs]
    r0["peak_mem_bytes"] = peak
    r0["peak_mem_over_weights_bytes"] = (peak - base) if peak is not None else None
    return r0
