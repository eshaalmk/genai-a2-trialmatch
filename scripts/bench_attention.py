"""Task 4: validate tiled online-softmax attention against the reference implementation
(and an fp64 ground truth), then benchmark speed and memory against the reference and
PyTorch's fused SDPA. Optionally plug the tiled kernel into the full model decoder."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from trialmatch import config as C
from trialmatch.attention import reference_attention, tiled_attention, torch_sdpa


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def correctness(dev):
    torch.manual_seed(0)
    shapes = [  # (B, H, Tq, Tk, D, q_offset, block_q, block_k)
        (1, 1, 1, 1, 8, 0, 4, 4),
        (1, 2, 7, 7, 16, 0, 3, 2),
        (2, 4, 64, 64, 64, 0, 16, 16),
        (1, 12, 100, 100, 128, 0, 32, 48),     # sizes not divisible by blocks
        (1, 12, 1, 257, 128, 256, 64, 64),     # cached decode step: 1 query, offset
        (1, 12, 5, 300, 128, 295, 64, 64),     # speculative verification chunk
        (2, 8, 333, 333, 64, 0, 128, 128),
        (1, 4, 50, 50, 32, 0, 50, 50),         # single tile
    ]
    rows = []
    for dtype in (torch.float32, torch.float16 if dev == "cuda" else torch.bfloat16):
        for (B, H, Tq, Tk, D, off, bq, bk) in shapes:
            q = torch.randn(B, H, Tq, D, device=dev) * 2
            k = torch.randn(B, H, Tk, D, device=dev) * 2
            v = torch.randn(B, H, Tk, D, device=dev)
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
            gt = reference_attention(q.double(), k.double(), v.double(), True, off).double()
            ref = reference_attention(q, k, v, True, off).double()
            til, st = tiled_attention(q, k, v, True, off, block_q=bq, block_k=bk, return_stats=True)
            sd = torch_sdpa(q, k, v, True, off).double()
            til = til.double()
            tol = 1e-5 if dtype == torch.float32 else 1e-2
            row = {"dtype": str(dtype).replace("torch.", ""), "shape": [B, H, Tq, Tk, D], "q_offset": off,
                   "blocks": [bq, bk], **st,
                   "tiled_vs_ref_max": float((til - ref).abs().max()), "tiled_vs_ref_mean": float((til - ref).abs().mean()),
                   "tiled_vs_fp64_max": float((til - gt).abs().max()), "ref_vs_fp64_max": float((ref - gt).abs().max()),
                   "sdpa_vs_fp64_max": float((sd - gt).abs().max()), "tolerance": tol}
            row["pass"] = row["tiled_vs_ref_max"] <= tol and row["tiled_vs_fp64_max"] <= tol
            rows.append(row)
            print(f"{row['dtype']:8s} {row['shape']} off={off:3d} tiled-ref max={row['tiled_vs_ref_max']:.2e} "
                  f"tiled-fp64={row['tiled_vs_fp64_max']:.2e} sdpa-fp64={row['sdpa_vs_fp64_max']:.2e} "
                  f"{'PASS' if row['pass'] else 'FAIL'}")
    # causal mask check: output at position i must not change when future keys change
    q = torch.randn(1, 2, 32, 16, device=dev); k = torch.randn(1, 2, 32, 16, device=dev); v = torch.randn(1, 2, 32, 16, device=dev)
    o1 = tiled_attention(q, k, v, True, 0, block_q=8, block_k=8)
    k2, v2 = k.clone(), v.clone(); k2[:, :, 20:] += 5; v2[:, :, 20:] -= 3
    o2 = tiled_attention(q, k2, v2, True, 0, block_q=8, block_k=8)
    causal_ok = bool(torch.allclose(o1[:, :, :20], o2[:, :, :20], atol=1e-6))
    # extreme-score stability (would overflow a naive exp)
    qx = torch.randn(1, 1, 16, 8, device=dev) * 300
    kx = torch.randn(1, 1, 16, 8, device=dev) * 300
    vx = torch.randn(1, 1, 16, 8, device=dev)
    stable = bool(torch.isfinite(tiled_attention(qx, kx, vx, True, 0, block_q=4, block_k=4)).all())
    stable_err = float((tiled_attention(qx, kx, vx, True, 0, block_q=4, block_k=4).double()
                        - reference_attention(qx.double(), kx.double(), vx.double(), True, 0)).abs().max())
    return rows, {"future_keys_do_not_leak": causal_ok, "large_logits_finite": stable,
                  "large_logits_max_err": stable_err}


def bench_one(fn, *args, repeats=5, **kw):
    for _ in range(2):
        fn(*args, **kw)
    sync()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    ts = []
    for _ in range(repeats):
        sync(); t = time.perf_counter(); fn(*args, **kw); sync(); ts.append(time.perf_counter() - t)
    peak = (torch.cuda.max_memory_allocated() - base) if torch.cuda.is_available() else None
    return statistics.median(ts), peak


def performance(dev, seqs, H, D, dtype):
    rows = []
    for T in seqs:
        q = torch.randn(1, H, T, D, device=dev, dtype=dtype)
        k = torch.randn(1, H, T, D, device=dev, dtype=dtype)
        v = torch.randn(1, H, T, D, device=dev, dtype=dtype)
        impls = [("reference", reference_attention, {}), ("tiled_b64", tiled_attention, {"block_q": 64, "block_k": 64}),
                 ("tiled_b128", tiled_attention, {"block_q": 128, "block_k": 128}), ("torch_sdpa", torch_sdpa, {})]
        for name, fn, kw in impls:
            try:
                t, peak = bench_one(fn, q, k, v, True, 0, **kw)
            except torch.cuda.OutOfMemoryError:
                t, peak = None, "OOM"
                torch.cuda.empty_cache()
            rows.append({"impl": name, "T": T, "H": H, "D": D, "dtype": str(dtype).replace("torch.", ""),
                         "time_ms": t * 1e3 if t else None, "extra_peak_mib": peak / 2**20 if isinstance(peak, int) else peak,
                         "score_matrix_mib_theoretical": H * T * T * 4 / 2**20})
            print(f"T={T:5d} {name:11s} time={rows[-1]['time_ms'] or float('nan'):9.2f}ms extra_peak={rows[-1]['extra_peak_mib']}")
    return rows


def model_integration(args, dev):
    """Run the full fine-tuned model with each attention kernel inside ManualDecoder."""
    from trialmatch.data import read_jsonl
    from trialmatch.decoding import first_mismatch, manual_cached, measure
    from trialmatch.manual_decoder import ManualDecoder
    from trialmatch.modeling import load_model, stop_token_ids
    from trialmatch.prompts import build_prompt
    model, tok = load_model(args.base, args.adapter)
    stops = stop_token_ids(tok)
    r = read_jsonl(C.DATA_DIR / "val.jsonl")[0]
    ids, _ = build_prompt(tok, r["sentences"], r["criterion_type"], r["criterion_text"])
    ids = torch.tensor([ids], device=dev)
    dec = ManualDecoder(model)
    res = {}
    for impl in ("torch_sdpa", "reference", "tiled"):
        dec.set_attention(impl, 64, 64)
        m = measure(manual_cached, dec, ids, 48, stops, True, warmup=1, repeats=2)
        res[impl] = {k: m[k] for k in ("tokens", "prefill_s", "decode_s", "decode_tokens_per_s", "peak_mem_bytes")}
    ref = res["reference"]["tokens"]
    for impl in res:
        res[impl]["first_mismatch_vs_reference"] = first_mismatch(ref, res[impl]["tokens"])
        res[impl]["text"] = tok.decode(res[impl]["tokens"])
        res[impl].pop("tokens")
        print(impl, {k: v for k, v in res[impl].items() if k != "text"})
    return {"prompt_tokens": ids.shape[1], "new_tokens": 48, "results": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--with_model", action="store_true")
    ap.add_argument("--base", default=C.TARGET_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "attention"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows, checks = correctness(dev)
    dtype = torch.float16 if dev == "cuda" else torch.float32
    perf = performance(dev, a.seqs if dev == "cuda" else [s for s in a.seqs if s <= 1024], a.heads, a.head_dim, dtype)
    res = {"device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "torch": torch.__version__,
           "correctness": rows, "all_pass": all(r["pass"] for r in rows), "property_checks": checks,
           "performance": perf}
    if a.with_model:
        res["model_integration"] = model_integration(a, dev)
    (out / "attention_results.json").write_text(json.dumps(res, indent=2))
    print("ALL CORRECTNESS CHECKS PASS:", res["all_pass"], checks)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    colors = {"reference": "#d1495b", "tiled_b64": "#6aa84f", "tiled_b128": "#38761d", "torch_sdpa": "#3b6ea5"}
    for impl in colors:
        rs = [r for r in perf if r["impl"] == impl and r["time_ms"]]
        if rs:
            axs[0].plot([r["T"] for r in rs], [r["time_ms"] for r in rs], "o-", color=colors[impl], label=impl)
            if isinstance(rs[0]["extra_peak_mib"], (int, float)):
                axs[1].plot([r["T"] for r in rs], [r["extra_peak_mib"] for r in rs], "o-", color=colors[impl], label=impl)
    for ax, t in zip(axs, ["time (ms)", "extra peak memory (MiB)"]):
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("sequence length")
        ax.set_ylabel(t); ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Causal self-attention, H={a.heads}, D={a.head_dim}, {res['device']}", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "attention_benchmark.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
