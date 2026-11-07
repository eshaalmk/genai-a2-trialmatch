"""End-to-end screening used by the browser app: parse note -> per-criterion prediction with
a selectable decoding engine -> grounded explanation -> rule-based aggregation -> report."""
from __future__ import annotations

import datetime as dt
import html
import json
import re
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import torch

from . import config as C
from .data import Sentence, split_sentences
from .decoding import hf_cached, hf_uncached, manual_cached, manual_uncached
from .manual_decoder import ManualDecoder
from .modeling import describe_model, load_model, stop_token_ids
from .prompts import build_explanation, build_prompt, parse_output
from .speculative import speculative_greedy
from .aggregation import aggregate, parse_criteria  # noqa: F401  (re-exported for the app)

MODES = {
    "kv_cached (own decoder + own KV cache)": "manual_cached",
    "kv_cached + tiled online-softmax attention": "manual_cached_tiled",
    "speculative (0.5B draft, k=4)": "speculative",
    "hf_cached (explicit DynamicCache)": "hf_cached",
    "uncached reference (slow)": "hf_uncached",
}

DISCLAIMER = ("Educational screening prototype for a university assignment. It is NOT a clinical "
              "enrollment decision, has not been validated for clinical use, and can be wrong. "
              "All outputs must be reviewed by qualified clinical research staff.")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=C.REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


class Screener:
    def __init__(self, base: str = C.TARGET_MODEL, adapter: Optional[str] = None,
                 draft_base: Optional[str] = C.DRAFT_MODEL, draft_adapter: Optional[str] = None):
        self.model, self.tok = load_model(base, adapter)
        self.dev = next(self.model.parameters()).device
        self.stops = stop_token_ids(self.tok)
        self.dec = ManualDecoder(self.model, attn_impl="torch_sdpa")
        self.dec_tiled = ManualDecoder(self.model, attn_impl="tiled")
        self.draft = None
        self.draft_info = None
        if draft_base:
            try:
                dm, _ = load_model(draft_base, draft_adapter)
                self.draft = ManualDecoder(dm, attn_impl="torch_sdpa")
                self.draft_info = describe_model(dm)
            except Exception as e:     # app still works without speculative mode
                self.draft_info = {"error": str(e)}
        self.info = describe_model(self.model)

    # ------------------------------------------------------------------ decode
    def _generate(self, ids: torch.Tensor, mode: str, max_new: int, k: int = 4) -> Dict:
        if mode == "manual_cached":
            return manual_cached(self.dec, ids, max_new, self.stops)
        if mode == "manual_cached_tiled":
            return manual_cached(self.dec_tiled, ids, max_new, self.stops)
        if mode == "hf_cached":
            return hf_cached(self.model, ids, max_new, self.stops)
        if mode == "hf_uncached":
            return hf_uncached(self.model, ids, max_new, self.stops)
        if mode == "manual_uncached":
            return manual_uncached(self.dec, ids, max_new, self.stops)
        if mode == "speculative":
            if self.draft is None:
                raise RuntimeError("Draft model not loaded; speculative mode unavailable.")
            return speculative_greedy(self.dec, self.draft, ids, max_new, k, self.stops)
        raise ValueError(mode)

    def screen_one(self, sents: List[Sentence], ctype: str, ctext: str, mode: str,
                   max_new: int = C.MAX_NEW_TOKENS) -> Dict:
        ids, info = build_prompt(self.tok, sents, ctype, ctext)
        t0 = time.time()
        g = self._generate(torch.tensor([ids], device=self.dev), mode, max_new)
        text = self.tok.decode(g["tokens"], skip_special_tokens=True)
        parsed = parse_output(text, ctype, info["kept_sentence_ids"])
        smap = dict(sents)
        label = parsed["label"]
        ev = parsed["evidence"]
        return {
            "criterion_type": ctype, "criterion_text": ctext, "label": label,
            "evidence_ids": ev, "evidence_text": [{"id": i, "text": smap[i]} for i in ev if i in smap],
            "explanation": build_explanation(label, ctype, ctext, ev, smap),
            "unsupported": label not in C.UNDECIDED_LABELS and not ev,
            "parse_ok": parsed["parse_ok"], "raw_output": text,
            "truncated_note": info["truncated"], "dropped_sentence_ids": info["dropped_sentence_ids"],
            "prompt_tokens": info["prompt_tokens"], "new_tokens": g["n_new_tokens"],
            "latency_s": time.time() - t0, "acceptance_rate": g.get("acceptance_rate"),
        }

    def screen(self, note_text: str, criteria: List[Tuple[str, str]], mode: str, progress=None) -> Dict:
        sents = split_sentences(note_text)
        results = []
        for i, (t, c) in enumerate(criteria):
            if progress is not None:
                progress((i, len(criteria)), desc=f"criterion {i + 1}/{len(criteria)}")
            results.append(self.screen_one(sents, t, c, mode))
        return {"sentences": sents, "results": results, "aggregate": aggregate(results),
                "provenance": self.provenance(mode)}

    def provenance(self, mode: str) -> Dict:
        return {"timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "base_model": self.info.get("base_model"), "adapter": self.info.get("adapter"),
                "dtype": self.info.get("dtype"), "device": self.info.get("device"),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "decoding_mode": mode, "decoding": "greedy (deterministic)", "git_commit": git_commit(),
                "draft_model": (self.draft_info or {}).get("base_model"),
                "draft_adapter": (self.draft_info or {}).get("adapter")}

    def _run_mode(self, ids, mode, n, force):
        return {"manual_cached": lambda: manual_cached(self.dec, ids, n, self.stops, force),
                "manual_cached_tiled": lambda: manual_cached(self.dec_tiled, ids, n, self.stops, force),
                "hf_cached": lambda: hf_cached(self.model, ids, n, self.stops, force),
                "hf_uncached": lambda: hf_uncached(self.model, ids, n, self.stops, force),
                "manual_uncached": lambda: manual_uncached(self.dec, ids, n, self.stops, force)}[mode]()

    # ------------------------------------------------------------------ live benchmark
    def benchmark(self, note_text: str, ctype: str, ctext: str, modes: List[str], max_new: int,
                  force_len: bool, ks=(2, 4, 6)) -> List[Dict]:
        sents = split_sentences(note_text)
        ids, _ = build_prompt(self.tok, sents, ctype, ctext)
        ids = torch.tensor([ids], device=self.dev)
        rows, ref = [], None
        for mode in modes:
            variants = [(mode, None)] if mode != "speculative" else [(mode, k) for k in ks]
            for m, k in variants:
                if m == "speculative" and self.draft is None:
                    continue
                run = (lambda n, f: speculative_greedy(self.dec, self.draft, ids, n, k, self.stops, f)) \
                    if m == "speculative" else (lambda n, f: self._run_mode(ids, m, n, f))
                run(4, False)                         # warm-up (kernel selection, allocator)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                g = run(max_new, force_len)
                peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
                if ref is None:
                    ref = g["tokens"]
                n = min(len(ref), len(g["tokens"]))
                same = ref[:n] == g["tokens"][:n] and len(ref) == len(g["tokens"])
                rows.append({"mode": m + (f" k={k}" if k else ""), "new_tokens": g["n_new_tokens"],
                             "prefill_ms": round(g["prefill_s"] * 1e3, 1), "total_ms": round(g["total_s"] * 1e3, 1),
                             "tokens_per_s": round(g["n_new_tokens"] / g["total_s"], 1) if g["total_s"] else None,
                             "acceptance_rate": round(g["acceptance_rate"], 3) if g.get("acceptance_rate") is not None else None,
                             "same_tokens_as_first_mode": same, "peak_mem_MiB": round(peak) if peak else None,
                             "output": self.tok.decode(g["tokens"], skip_special_tokens=True)[:120]})
        return rows


# ---------------------------------------------------------------------- report export
def report_html(res: Dict, patient_label: str = "patient") -> str:
    e = html.escape
    agg = res["aggregate"]
    rows = []
    for r in res["results"]:
        ev = "<br>".join(f"[{x['id']}] {e(x['text'])}" for x in r["evidence_text"]) or "<i>none cited</i>"
        flag = " ⚠ unsupported" if r["unsupported"] else ""
        rows.append(f"<tr><td>{e(r['criterion_type'])}</td><td>{e(r['criterion_text'])}</td>"
                    f"<td><b>{e(r['label'])}</b>{flag}</td><td>{ev}</td><td>{e(r['explanation'])}</td></tr>")
    note = "".join(f"<li value='{i}'>{e(t)}</li>" for i, t in res["sentences"])
    prov = "".join(f"<tr><td>{e(str(k))}</td><td>{e(str(v))}</td></tr>" for k, v in res["provenance"].items())
    missing = "".join(f"<li>{e(c)}</li>" for c in agg["missing_information_criteria"]) or "<li>none</li>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Eligibility screening report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ccc;padding:6px;vertical-align:top}}
th{{background:#f2f4f7;text-align:left}}.warn{{background:#fff4e5;border:1px solid #f0b46c;padding:10px;border-radius:6px}}
ol li{{margin:2px 0}}</style></head><body>
<h1>Eligibility screening report</h1>
<p class="warn"><b>Notice:</b> {e(DISCLAIMER)}</p>
<h2>Preliminary outcome (rule-based aggregation): {e(agg['verdict'])}</h2>
<p>{agg['n_criteria']} criteria · {agg['n_failing']} failing · {agg['n_missing_information']} missing information ·
{agg['n_unsupported']} unsupported decisions.<br><small>Rule: {e(agg['rule'])}</small></p>
<h3>Missing information</h3><ul>{missing}</ul>
<h2>Criterion-level results</h2>
<table><tr><th>Type</th><th>Criterion</th><th>Prediction</th><th>Evidence (sentence IDs)</th><th>Explanation</th></tr>
{''.join(rows)}</table>
<h2>Patient note ({e(patient_label)}) as numbered for the model</h2><ol start="0">{note}</ol>
<h2>Provenance</h2><table>{prov}</table></body></html>"""


def report_json(res: Dict) -> str:
    return json.dumps({"disclaimer": DISCLAIMER, **res}, indent=2, default=str)
