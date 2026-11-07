"""Task 3: KV-cache correctness (greedy token equality) and benchmarks.

Correctness: for >=3 (input length, output length) pairs, the four paths
hf_uncached / hf_cached / manual_uncached / manual_cached must produce the same greedy
tokens. Any mismatch is investigated (top-2 logit margin at the first divergence).
Benchmark: prefill time, decode time, tokens/s and peak memory, with warm-up and
synchronised, repeated timing (median reported)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from trialmatch import config as C
from trialmatch.data import read_jsonl
from trialmatch.decoding import (first_mismatch, hf_cached, hf_uncached, investigate_mismatch,
                                 manual_cached, manual_uncached, measure)
from trialmatch.manual_decoder import ManualDecoder
from trialmatch.modeling import describe_model, load_model, stop_token_ids
from trialmatch.prompts import build_prompt


def make_inputs(tok, lengths, dev):
    """Real validation prompts; long inputs are built by concatenating several val notes'
    prompts and cutting to the exact token length (benchmark only)."""
    recs = read_jsonl(C.DATA_DIR / "val.jsonl")
    pool = []
    for r in recs[:40]:
        ids, _ = build_prompt(tok, r["sentences"], r["criterion_type"], r["criterion_text"])
        pool.append(ids)
    pool.sort(key=len)
    flat = [t for p in pool for t in p]
    out = {}
    for L in lengths:
        exact = [p for p in pool if len(p) >= L]
        ids = exact[0][-L:] if exact and len(exact[0]) - L < 64 else flat[:L]
        out[L] = torch.tensor([ids], device=dev)
    return out, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=C.TARGET_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dtype", default=None, help="fp16/bf16/fp32 (default: best for GPU)")
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 1024])
    ap.add_argument("--outputs", type=int, nargs="+", default=[32, 96])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--attn", default="torch_sdpa", help="attention used inside ManualDecoder")
    ap.add_argument("--skip_uncached_above", type=int, default=100000,
                    help="skip uncached benchmark when L+N exceeds this (time budget)")
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "kvcache"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    model, tok = load_model(a.base, a.adapter, dtype=a.dtype)
    dev = next(model.parameters()).device
    dec = ManualDecoder(model, attn_impl=a.attn)
    stops = stop_token_ids(tok)
    info = describe_model(model)
    inputs, pool = make_inputs(tok, a.lengths, dev)

    # ---------------- correctness -------------------------------------------------
    correctness = []
    cases = [("real_prompt_eos", torch.tensor([pool[len(pool) // 2]], device=dev), 64, False)]
    for L in a.lengths:
        for N in a.outputs:
            cases.append((f"L{L}_N{N}_forced", inputs[L], N, True))
    for name, ids, N, ignore in cases:
        runs = {f.__name__: (f(model, ids, N, stops, ignore) if f.__name__.startswith("hf")
                             else f(dec, ids, N, stops, ignore))
                for f in (hf_uncached, hf_cached, manual_uncached, manual_cached)}
        ref = runs["hf_uncached"]["tokens"]
        entry = {"case": name, "input_len": ids.shape[1], "max_new": N, "ignore_eos": ignore,
                 "n_generated": len(ref), "text": tok.decode(ref)[:200], "agreement": {}}
        for k, r in runs.items():
            mm = first_mismatch(ref, r["tokens"])
            entry["agreement"][k] = {"equal": mm is None, "first_mismatch": mm,
                                     "matching_prefix_frac": (mm if mm is not None else len(ref)) / max(1, len(ref))}
            if mm is not None:
                entry["agreement"][k]["investigation"] = investigate_mismatch(model, ids, ref, r["tokens"], mm)
        entry["hf_cache_lengths_ok"] = runs["hf_cached"]["cache_lengths"] == list(
            range(ids.shape[1], ids.shape[1] + len(runs["hf_cached"]["tokens"])))
        correctness.append(entry)
        print(name, {k: v["equal"] for k, v in entry["agreement"].items()})

    # logit-level agreement of the manual cached path vs HF full forward (one case)
    ids = inputs[a.lengths[0]]
    r = manual_cached(dec, ids, 16, stops, True, return_logits=True)
    seq = torch.cat([ids, torch.tensor([r["tokens"][:-1]], device=dev)], dim=1)
    with torch.no_grad():
        hf_logits = model(input_ids=seq, use_cache=False).logits[0, ids.shape[1] - 1:].float().cpu()
    man = torch.stack(r["logits"])
    logit_err = {"max_abs": float((hf_logits - man).abs().max()), "mean_abs": float((hf_logits - man).abs().mean()),
                 "dtype": info["dtype"]}
    print("logit diff manual_cached vs HF full forward:", logit_err)

    # ---------------- benchmark ----------------------------------------------------
    bench = []
    for L in a.lengths:
        for N in a.outputs:
            ids = inputs[L]
            for fname, fn, obj in (("hf_cached", hf_cached, model), ("manual_cached", manual_cached, dec),
                                   ("hf_uncached", hf_uncached, model), ("manual_uncached", manual_uncached, dec)):
                if "uncached" in fname and L + N > a.skip_uncached_above:
                    continue
                m = measure(fn, obj, ids, N, stops, True, warmup=1, repeats=a.repeats)
                m.pop("tokens", None); m.pop("cache_lengths", None)
                m.update({"input_len": L, "output_len": N, "path": fname})
                bench.append(m)
                print(f"L={L:5d} N={N:4d} {fname:16s} prefill={m['prefill_s']*1e3:8.1f}ms "
                      f"decode={m['decode_s']*1e3:9.1f}ms tok/s={m['decode_tokens_per_s'] or 0:7.1f} "
                      f"peak={(m['peak_mem_bytes'] or 0)/2**20:8.0f}MiB")
    # per-step latency profile (shows uncached cost growing with sequence length)
    prof_ids = inputs[a.lengths[-1]]
    prof = {"cached": hf_cached(model, prof_ids, 48, stops, True)["per_step_s"],
            "uncached": hf_uncached(model, prof_ids, 48, stops, True)["per_step_s"]}

    res = {"model": info, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "torch": torch.__version__, "correctness": correctness, "logit_agreement": logit_err,
           "benchmark": bench, "per_step_profile_s": prof,
           "notes": {"timing": "median of repeats after 1 warm-up; torch.cuda.synchronize around every timed step",
                     "decode_tokens_per_s": "(N-1)/decode_s; first token is produced by prefill",
                     "forced_length": "ignore_eos=True so every path generates exactly N tokens"}}
    (out / "kvcache_results.json").write_text(json.dumps(res, indent=2))
    plot(res, out)


def plot(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = res["benchmark"]
    paths = ["hf_cached", "manual_cached", "hf_uncached", "manual_uncached"]
    colors = {"hf_cached": "#3b6ea5", "manual_cached": "#6aa84f", "hf_uncached": "#d1495b", "manual_uncached": "#e69138"}
    Ns = sorted({r["output_len"] for r in b})
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    N = Ns[-1]
    for p in paths:
        rs = sorted([r for r in b if r["path"] == p and r["output_len"] == N], key=lambda r: r["input_len"])
        if not rs:
            continue
        x = [r["input_len"] for r in rs]
        axs[0].plot(x, [r["decode_tokens_per_s"] for r in rs], "o-", color=colors[p], label=p)
        axs[1].plot(x, [r["prefill_s"] * 1e3 for r in rs], "o-", color=colors[p], label=p)
        axs[2].plot(x, [(r["peak_mem_bytes"] or 0) / 2**20 for r in rs], "o-", color=colors[p], label=p)
    for ax, t in zip(axs, ["decode tokens/s", "prefill ms", "peak allocated MiB"]):
        ax.set_xlabel("input tokens"); ax.set_ylabel(t); ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"KV cache benchmark (output = {N} tokens)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "kvcache_benchmark.png", dpi=160); plt.close(fig)

    prof = res["per_step_profile_s"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for k, c in (("cached", "#3b6ea5"), ("uncached", "#d1495b")):
        ax.plot([t * 1e3 for t in prof[k]], color=c, label=k)
    ax.set_xlabel("decode step"); ax.set_ylabel("ms / token"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "kvcache_per_step.png", dpi=160); plt.close(fig)

    lines = ["| path | input | output | prefill ms | decode ms | decode tok/s | e2e tok/s | peak MiB |",
             "|---|---|---|---|---|---|---|---|"]
    for r in b:
        lines.append(f"| {r['path']} | {r['input_len']} | {r['output_len']} | {r['prefill_s']*1e3:.1f} | "
                     f"{r['decode_s']*1e3:.1f} | {r['decode_tokens_per_s'] or 0:.1f} | {r['e2e_tokens_per_s'] or 0:.1f} | "
                     f"{(r['peak_mem_bytes'] or 0)/2**20:.0f} |")
    lines += ["", "Correctness (token equality vs hf_uncached):", ""]
    for c in res["correctness"]:
        lines.append(f"- {c['case']} (L={c['input_len']}, N={c['n_generated']}): " +
                     ", ".join(f"{k}={'OK' if v['equal'] else 'MISMATCH@' + str(v['first_mismatch'])}"
                               for k, v in c["agreement"].items()))
    (out / "kvcache_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
