"""Task 5: greedy speculative decoding with explicit cache-state synchronisation.

Both draft and target run through ManualDecoder, so we control their KV caches directly
(rollback = KVCache.crop).

Invariant at the start of every round (S = committed tokens = prompt + generated):
    target_cache.length == len(S) - 1   (last committed token not yet fed to the target)
    draft_cache.length  <= len(S) - 1   (draft may lag by one; it catches up by feeding
                                         the missing committed tokens in one forward)
Round:
  1. Draft: feed the missing committed tokens, then greedily propose k tokens d1..dk.
  2. Verify: feed [S[-1], d1..dk] (k+1 tokens) to the target in ONE forward; its logits at
     row i give the target's greedy choice t_i for the position after d_i (t_0 after S[-1]).
  3. Accept the longest prefix with d_{i+1} == t_i (a tokens). Commit d1..da plus t_a
     (the correction token if a < k, or a free "bonus" token if a == k).
  4. Roll back the target cache to len(S_new) - 1 = old_len + a; crop the draft cache to
     at most that (its entries for rejected draft tokens are discarded).
Every committed token is the target's own argmax given the committed prefix, so the
output equals plain greedy decoding of the target (up to floating-point near-ties
between batched-verification and one-token-at-a-time logits, which we measure).
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch

from .decoding import Timer
from .manual_decoder import ManualDecoder


@torch.no_grad()
def speculative_greedy(target: ManualDecoder, draft: ManualDecoder, input_ids: torch.Tensor,
                       max_new_tokens: int, k: int = 4, stops: Optional[Iterable[int]] = None,
                       ignore_eos: bool = False) -> Dict:
    stops = set(stops or [])
    dev = target.device
    prompt = input_ids[0].tolist()
    L = len(prompt)
    cap = L + max_new_tokens + k + 2
    tc = target.new_cache(1, cap)
    dc = draft.new_cache(1, cap)
    tm = Timer()

    # prefill both models on prompt[:-1]; the last prompt token is fed in round 1
    if L > 1:
        target.forward(input_ids[:, :-1], cache=tc, last_only=True)
        draft.forward(input_ids[:, :-1], cache=dc, last_only=True)
    prefill = tm.lap()

    seq = list(prompt)
    rounds = []
    n_target_fwd = n_draft_fwd = 0
    finished = False
    while not finished and len(seq) - L < max_new_tokens:
        remaining = max_new_tokens - (len(seq) - L)
        kk = max(1, min(k, remaining))     # do not draft past the budget
        # ---- 1. draft k tokens
        missing = seq[dc.length:]          # committed tokens the draft has not consumed (1 or 2)
        logits = draft.forward(torch.tensor([missing], device=dev), cache=dc, last_only=True)
        n_draft_fwd += 1
        drafts = []
        for j in range(kk):
            d = int(logits[0, -1].argmax())
            drafts.append(d)
            if d in stops and not ignore_eos:
                break
            if j < kk - 1:
                logits = draft.forward(torch.tensor([[d]], device=dev), cache=dc, last_only=True)
                n_draft_fwd += 1
        # ---- 2. verify with ONE target forward over [S[-1], d1..dk]
        assert tc.length == len(seq) - 1, (tc.length, len(seq))
        base = tc.length
        vlog = target.forward(torch.tensor([[seq[-1]] + drafts], device=dev), cache=tc)
        n_target_fwd += 1
        tgt = vlog[0].argmax(-1).tolist()          # len(drafts)+1 greedy target tokens
        # ---- 3. accept longest matching prefix
        a = 0
        while a < len(drafts) and drafts[a] == tgt[a]:
            a += 1
        new = drafts[:a] + [tgt[a]]
        # stop handling: cut at the first stop token
        for i, t in enumerate(new):
            if t in stops and not ignore_eos:
                new = new[:i + 1]
                finished = True
                break
        new = new[:remaining]
        seq.extend(new)
        # ---- 4. cache synchronisation
        tc.crop(base + len(new))                   # = len(seq) - 1 ; drops rejected drafts
        dc.crop(min(dc.length, len(seq) - 1))
        rounds.append({"drafted": len(drafts), "accepted": a, "committed": len(new)})
        if len(seq) - L >= max_new_tokens:
            break
    total = tm.lap()
    gen = seq[L:]
    drafted = sum(r["drafted"] for r in rounds)
    accepted = sum(r["accepted"] for r in rounds)
    return {
        "mode": f"speculative[k={k},{target.attn_name}]", "tokens": gen, "n_new_tokens": len(gen),
        "prefill_s": prefill, "decode_s": total, "total_s": prefill + total,
        "e2e_tokens_per_s": len(gen) / (prefill + total) if prefill + total > 0 else None,
        "decode_tokens_per_s": len(gen) / total if total > 0 else None,
        "rounds": len(rounds), "drafted": drafted, "accepted": accepted,
        "acceptance_rate": accepted / drafted if drafted else None,
        "mean_accepted_per_round": accepted / len(rounds) if rounds else None,
        "tokens_per_target_forward": len(gen) / n_target_fwd if n_target_fwd else None,
        "target_forwards": n_target_fwd, "draft_forwards": n_draft_fwd,
        "round_log": rounds,
    }


def check_tokenizer_compat(tok_t, tok_d) -> Dict:
    vt, vd = tok_t.get_vocab(), tok_d.get_vocab()
    same = vt == vd
    probe = "Exclusion: patients with eGFR < 30 mL/min/1.73m² or pregnancy. {\"label\": \"not excluded\"}"
    return {"identical_vocab": same, "vocab_size_target": len(vt), "vocab_size_draft": len(vd),
            "probe_ids_equal": tok_t(probe)["input_ids"] == tok_d(probe)["input_ids"],
            "eos_target": tok_t.eos_token, "eos_draft": tok_d.eos_token}
