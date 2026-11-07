"""Task 5: greedy speculative decoding — token equality with plain greedy target decoding,
acceptance statistics, latency, throughput, memory; sweep over draft length k and over
draft models (base vs fine-tuned 0.5B). HF assisted generation is run only as an external
reference point."""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from trialmatch import config as C
from trialmatch.data import read_jsonl
from trialmatch.decoding import first_mismatch, manual_cached, measure
from trialmatch.manual_decoder import ManualDecoder
from trialmatch.modeling import describe_model, load_model, stop_token_ids
from trialmatch.prompts import build_prompt
from trialmatch.speculative import check_tokenizer_compat, speculative_greedy


def hf_greedy(model, ids, N, stops):
    """HF generate reference (natural-EOS setting only: forcing a length in HF needs
    min_new_tokens, which suppresses EOS logits and so changes the argmax)."""
    with torch.no_grad():
        g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=N,
                           do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
                           eos_token_id=sorted(stops), pad_token_id=min(stops))
    return g[0, ids.shape[1]:].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=C.TARGET_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--draft_base", default=C.DRAFT_MODEL)
    ap.add_argument("--draft_adapter", default=None, help="fine-tuned draft adapter (optional)")
    ap.add_argument("--n_prompts", type=int, default=12)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--forced_len", type=int, default=128)
    ap.add_argument("--attn", default="torch_sdpa")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "speculative"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    target, tok = load_model(a.base, a.adapter)
    dev = next(target.parameters()).device
    stops = stop_token_ids(tok)
    tdec = ManualDecoder(target, attn_impl=a.attn)
    drafts = {"draft_base": load_model(a.draft_base, None)}
    if a.draft_adapter:
        drafts["draft_finetuned"] = load_model(a.draft_base, a.draft_adapter)
    compat = {n: check_tokenizer_compat(tok, dt) for n, (dm, dt) in drafts.items()}
    print("tokenizer compatibility:", compat)
    for n, c in compat.items():
        assert c["identical_vocab"] or c["probe_ids_equal"], f"{n}: tokenizer mismatch"

    recs = read_jsonl(C.DATA_DIR / "val.jsonl")
    step = max(1, len(recs) // a.n_prompts)
    prompts = []
    for r in recs[::step][:a.n_prompts]:
        ids, _ = build_prompt(tok, r["sentences"], r["criterion_type"], r["criterion_text"])
        prompts.append(torch.tensor([ids], device=dev))

    settings = [("natural_eos", C.MAX_NEW_TOKENS, False), (f"forced_{a.forced_len}", a.forced_len, True)]
    runs, equality = [], []
    for sname, N, ignore in settings:
        for pi, ids in enumerate(prompts):
            base = measure(manual_cached, tdec, ids, N, stops, ignore, warmup=1, repeats=a.repeats)
            base_tokens = base["tokens"]
            eq_row = {"setting": sname, "prompt": pi, "input_len": ids.shape[1]}
            if not ignore:
                eq_row["greedy_manual_vs_hf_generate"] = first_mismatch(hf_greedy(target, ids, N, stops), base_tokens) is None
            runs.append({"setting": sname, "prompt": pi, "method": "greedy_target", "draft": None, "k": None,
                         **{k: base[k] for k in ("n_new_tokens", "prefill_s", "decode_s", "total_s",
                                                 "e2e_tokens_per_s", "peak_mem_bytes")}})
            for dname, (dmodel, _) in drafts.items():
                ddec = ManualDecoder(dmodel, attn_impl=a.attn)
                for k in a.ks:
                    m = measure(speculative_greedy, tdec, ddec, ids, N, k, stops, ignore, warmup=1, repeats=a.repeats)
                    mm = first_mismatch(base_tokens, m["tokens"])
                    eq_row[f"{dname}_k{k}"] = mm is None
                    if mm is not None:
                        eq_row[f"{dname}_k{k}_first_mismatch"] = mm
                    runs.append({"setting": sname, "prompt": pi, "method": "speculative", "draft": dname, "k": k,
                                 **{x: m[x] for x in ("n_new_tokens", "prefill_s", "decode_s", "total_s",
                                                      "e2e_tokens_per_s", "peak_mem_bytes", "acceptance_rate",
                                                      "mean_accepted_per_round", "tokens_per_target_forward",
                                                      "target_forwards", "draft_forwards", "rounds")}})
                # HF assisted generation, reference only (natural-EOS setting)
                if ignore:
                    continue
                try:
                    import time
                    from trialmatch.decoding import _sync
                    _sync(); t = time.perf_counter()
                    with torch.no_grad():
                        g = target.generate(input_ids=ids, attention_mask=torch.ones_like(ids), assistant_model=dmodel,
                                            max_new_tokens=N, do_sample=False, temperature=None, top_p=None,
                                            top_k=None, eos_token_id=sorted(stops), pad_token_id=min(stops))
                    _sync()
                    eq_row[f"{dname}_hf_assisted_equal"] = first_mismatch(base_tokens, g[0, ids.shape[1]:].tolist()) is None
                    runs.append({"setting": sname, "prompt": pi, "method": "hf_assisted_reference", "draft": dname,
                                 "k": None, "total_s": time.perf_counter() - t,
                                 "n_new_tokens": g.shape[1] - ids.shape[1]})
                except Exception as e:  # reference only; never required
                    eq_row[f"{dname}_hf_assisted_error"] = str(e)[:200]
            equality.append(eq_row)
            print(eq_row)

    # aggregate
    summary = []
    keys = sorted({(r["setting"], r["method"], r["draft"], r["k"]) for r in runs}, key=str)
    for (s, meth, d, k) in keys:
        rs = [r for r in runs if (r["setting"], r["method"], r["draft"], r["k"]) == (s, meth, d, k)]
        agg = lambda f: statistics.mean([r[f] for r in rs if r.get(f) is not None]) if any(r.get(f) is not None for r in rs) else None
        base_t = statistics.mean([r["total_s"] for r in runs if r["setting"] == s and r["method"] == "greedy_target"])
        summary.append({"setting": s, "method": meth, "draft": d, "k": k, "n_prompts": len(rs),
                        "mean_total_s": agg("total_s"), "mean_tokens": agg("n_new_tokens"),
                        "tokens_per_s": (sum(r["n_new_tokens"] for r in rs) / sum(r["total_s"] for r in rs)),
                        "speedup_vs_greedy": base_t / agg("total_s") if agg("total_s") else None,
                        "acceptance_rate": agg("acceptance_rate"), "mean_accepted_per_round": agg("mean_accepted_per_round"),
                        "tokens_per_target_forward": agg("tokens_per_target_forward"),
                        "peak_mem_mib": (agg("peak_mem_bytes") or 0) / 2**20})
    eq_keys = sorted({k for e in equality for k in e} - {"setting", "prompt", "input_len"})
    eq_keys = [k for k in eq_keys if k not in ("setting", "prompt", "input_len") and not k.endswith(("_first_mismatch", "_error"))]
    eq_rate = {k: sum(bool(e[k]) for e in equality if k in e) / max(1, sum(k in e for e in equality)) for k in eq_keys}
    res = {"target": describe_model(target), "drafts": {n: describe_model(m) for n, (m, _) in drafts.items()},
           "tokenizer_compat": compat, "attention": a.attn, "equality": equality, "equality_rate": eq_rate,
           "summary": summary, "runs": runs,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    (out / "speculative_results.json").write_text(json.dumps(res, indent=2))
    lines = ["| setting | method | draft | k | tok/s | speedup | accept rate | acc/round | tok/target fwd | peak MiB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, p=2: "-" if x is None else f"{x:.{p}f}"
    for r in summary:
        lines.append(f"| {r['setting']} | {r['method']} | {r['draft'] or '-'} | {r['k'] or '-'} | {f(r['tokens_per_s'],1)} | "
                     f"{f(r['speedup_vs_greedy'])} | {f(r['acceptance_rate'])} | {f(r['mean_accepted_per_round'])} | "
                     f"{f(r['tokens_per_target_forward'])} | {f(r['peak_mem_mib'],0)} |")
    lines += ["", "Token-equality rate vs greedy target:", "```", json.dumps(eq_rate, indent=2), "```"]
    (out / "speculative_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for s, ls in ((settings[0][0], "-"), (settings[1][0], "--")):
        for d, c in (("draft_base", "#9aa5b1"), ("draft_finetuned", "#3b6ea5")):
            rs = sorted([r for r in summary if r["setting"] == s and r["draft"] == d and r["method"] == "speculative"],
                        key=lambda r: r["k"])
            if rs:
                axs[0].plot([r["k"] for r in rs], [r["acceptance_rate"] for r in rs], "o" + ls, color=c, label=f"{d} / {s}")
                axs[1].plot([r["k"] for r in rs], [r["speedup_vs_greedy"] for r in rs], "o" + ls, color=c, label=f"{d} / {s}")
    axs[1].axhline(1.0, color="#d1495b", lw=1)
    axs[0].set_ylabel("draft acceptance rate"); axs[1].set_ylabel("speedup vs greedy target")
    for ax in axs:
        ax.set_xlabel("draft length k"); ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out / "speculative_benchmark.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
