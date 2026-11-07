"""Browser application (Gradio): criterion-level clinical-trial eligibility screening with
evidence, exportable report, and a live inference benchmark of the decoding mechanisms.

Start:  python app/app.py --adapter <path-or-hub-id> [--draft_adapter ...] [--share]
Then open the printed URL (local: http://127.0.0.1:7860, Colab: the *.gradio.live link)."""
import argparse
import datetime as dt
import html
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr
import pandas as pd

from trialmatch import config as C
from trialmatch.data import split_sentences
from trialmatch.screen import DISCLAIMER, MODES, Screener, parse_criteria, report_html, report_json
from trialmatch.trial_fetch import fetch_criteria

SCREENER = None
MAX_CRITERIA = 40
MAX_NOTE_CHARS = 12000
REPORT_DIR = C.RESULTS_DIR / "app_reports"

EXAMPLE_NOTE = (
    "A 62-year-old man presents with three months of exertional chest pain relieved by rest. "
    "He has type 2 diabetes treated with metformin and a 30 pack-year smoking history. "
    "Blood pressure is 148/92 mmHg. "
    "Serum creatinine is 1.1 mg/dL. "
    "An exercise stress test showed 2 mm ST depression at 6 METs. "
    "He has no history of stroke or bleeding disorders. "
    "He is not taking any anticoagulants.")
EXAMPLE_CRITERIA = """Inclusion Criteria:
- Age 40 to 80 years
- Stable angina with a positive stress test
- Female patients of childbearing potential with a negative pregnancy test
Exclusion Criteria:
- Severe renal impairment (eGFR < 30 mL/min/1.73m2)
- Current treatment with oral anticoagulants
- Known left ventricular ejection fraction below 35%"""


def val_examples():
    p = C.DATA_DIR / "val.jsonl"
    if not p.exists():
        return {}
    ex = {}
    for line in open(p):
        r = json.loads(line)
        key = f"{r['patient_id']} · {r['trial_id']}"
        e = ex.setdefault(key, {"note": "\n".join(f"{i}. {t}" for i, t in r["sentences"]), "crit": []})
        e["crit"].append(f"{'INC' if r['criterion_type'] == 'inclusion' else 'EXC'}: {r['criterion_text']}")
    return dict(list(ex.items())[:25])


VAL_EX = val_examples()


# ------------------------------------------------------------------------------- helpers
def validate(note, criteria_text):
    if not note or not note.strip():
        raise gr.Error("Please enter a patient note.")
    if len(note) > MAX_NOTE_CHARS:
        raise gr.Error(f"Patient note is too long ({len(note)} characters, limit {MAX_NOTE_CHARS}).")
    sents = split_sentences(note)
    if len(sents) < 1:
        raise gr.Error("Could not find any sentences in the patient note.")
    crits = parse_criteria(criteria_text)
    if not crits:
        raise gr.Error("Please enter at least one criterion (one per line, or under "
                       "'Inclusion Criteria:' / 'Exclusion Criteria:' headings, or prefixed INC:/EXC:).")
    if len(crits) > MAX_CRITERIA:
        gr.Warning(f"{len(crits)} criteria found; only the first {MAX_CRITERIA} are screened.")
        crits = crits[:MAX_CRITERIA]
    long = [c for _, c in crits if len(c) > 1200]
    if long:
        raise gr.Error("A criterion is longer than 1200 characters; please split it into separate criteria.")
    return sents, crits


def numbered_preview(note):
    sents = split_sentences(note or "")
    if not sents:
        return "<i>No sentences yet.</i>"
    return "<ol start='0' style='margin:0'>" + "".join(
        f"<li value='{i}'>{html.escape(t)}</li>" for i, t in sents) + "</ol>"


LABEL_COLORS = {"included": "#2e7d32", "not excluded": "#2e7d32", "not included": "#c62828",
                "excluded": "#c62828", "not enough information": "#b26a00", "not applicable": "#546e7a"}


def verdict_html(res):
    a = res["aggregate"]
    color = {"Potentially eligible": "#2e7d32", "Likely ineligible": "#c62828"}.get(a["verdict"], "#b26a00")
    miss = "".join(f"<li>{html.escape(c)}</li>" for c in a["missing_information_criteria"]) or "<li>none</li>"
    return (f"<div style='border-left:6px solid {color};padding:10px 14px;background:rgba(127,127,127,.06)'>"
            f"<div style='font-size:1.25em;font-weight:600;color:{color}'>{html.escape(a['verdict'])}</div>"
            f"<div>Preliminary outcome from an explicit aggregation rule over criterion-level predictions "
            f"(not a ground-truth label). {a['n_criteria']} criteria · {a['n_failing']} failing · "
            f"{a['n_missing_information']} missing information · {a['n_unsupported']} unsupported.</div>"
            f"<details><summary>Missing information</summary><ul>{miss}</ul></details>"
            f"<small>{html.escape(a['rule'])}</small></div>")


def evidence_view(res, idx):
    if not res:
        return ""
    idx = int(idx or 0)
    r = res["results"][idx]
    cited = set(r["evidence_ids"])
    color = LABEL_COLORS.get(r["label"], "#333")
    items = []
    for i, t in res["sentences"]:
        style = f"background:{color}22;border-left:4px solid {color};padding-left:6px;font-weight:500" if i in cited else ""
        items.append(f"<li value='{i}' style='{style}'>{html.escape(t)}</li>")
    warn = ("<p style='color:#c62828'><b>Unsupported decision:</b> no evidence sentence was cited.</p>"
            if r["unsupported"] else "")
    trunc = (f"<p style='color:#b26a00'>Note was truncated for the model; sentences {r['dropped_sentence_ids']} "
             f"were not seen.</p>" if r["truncated_note"] else "")
    return (f"<p><b>{html.escape(r['criterion_type'])}:</b> {html.escape(r['criterion_text'])}</p>"
            f"<p>Prediction: <b style='color:{color}'>{html.escape(r['label'])}</b> · cited sentences "
            f"{sorted(cited) or '[]'}</p>{warn}{trunc}<p>{html.escape(r['explanation'])}</p>"
            f"<ol start='0'>{''.join(items)}</ol>"
            f"<details><summary>Raw model output</summary><code>{html.escape(r['raw_output'])}</code></details>")


# ------------------------------------------------------------------------------- actions
def do_fetch(nct):
    try:
        title, crit = fetch_criteria(nct)
        gr.Info(f"Loaded criteria for {nct.strip().upper()}: {title}")
        return crit
    except Exception as e:
        raise gr.Error(str(e))


def do_screen(note, criteria_text, mode_label, progress=gr.Progress()):
    sents, crits = validate(note, criteria_text)
    mode = MODES[mode_label]
    if mode == "hf_uncached" and len(crits) > 8:
        gr.Warning("Uncached reference mode is slow; consider a cached mode for many criteria.")
    try:
        res = SCREENER.screen(note, crits, mode, progress=progress)
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Inference failed: {e}")
    if any(r["truncated_note"] for r in res["results"]):
        gr.Warning("The note exceeded the model's prompt budget; trailing sentences were dropped (see evidence view).")
    df = pd.DataFrame([{
        "#": i, "type": r["criterion_type"], "criterion": r["criterion_text"][:140], "prediction": r["label"],
        "evidence IDs": ", ".join(map(str, r["evidence_ids"])) or "—",
        "flag": "unsupported" if r["unsupported"] else ("parse fallback" if not r["parse_ok"] else ""),
        "latency s": round(r["latency_s"], 2)} for i, r in enumerate(res["results"])])
    # persistent copy + downloads
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    h = REPORT_DIR / f"screening_report_{stamp}.html"
    j = REPORT_DIR / f"screening_report_{stamp}.json"
    h.write_text(report_html(res)); j.write_text(report_json(res))
    choices = [(f"{i}: [{r['criterion_type'][:3]}] {r['criterion_text'][:70]}", i) for i, r in enumerate(res["results"])]
    return (verdict_html(res), df, gr.update(choices=choices, value=0), evidence_view(res, 0),
            [str(h), str(j)], res)


def do_bench(note, criteria_text, crit_idx, modes, max_new, force):
    sents, crits = validate(note, criteria_text)
    if not modes:
        raise gr.Error("Select at least one decoding mode.")
    i = int(crit_idx or 0)
    if i >= len(crits):
        i = 0
    t, c = crits[i]
    keys = [MODES[m] for m in modes]
    try:
        rows = SCREENER.benchmark(note, t, c, keys, int(max_new), bool(force))
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Benchmark failed: {e}")
    df = pd.DataFrame(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].barh(df["mode"], df["tokens_per_s"].fillna(0), color="#3b6ea5")
    ax[0].set_xlabel("tokens / s (end-to-end)")
    ax[1].barh(df["mode"], df["total_ms"], color="#d1495b")
    ax[1].set_xlabel("total latency (ms)")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.invert_yaxis()
    fig.tight_layout()
    eq = "all modes produced identical tokens" if df["same_tokens_as_first_mode"].all() else \
        "some modes differ from the first mode — see table (possible floating-point near-tie)"
    return df, fig, f"Criterion benchmarked: **[{t}] {c[:150]}** — {eq}."


def saved_results_md():
    parts = []
    files = [("Split manifest", "data/split_manifest.json"), ("Before / after fine-tuning", "eval/before_after.md"),
             ("KV cache", "kvcache/kvcache_summary.md"), ("Speculative decoding", "speculative/speculative_summary.md"),
             ("Model candidates", "research/model_candidates.md")]
    for title, rel in files:
        p = C.RESULTS_DIR / rel
        if p.exists():
            txt = p.read_text()
            if rel.endswith(".json"):
                d = json.loads(txt)
                txt = "\n".join(f"- **{s}**: {v['n_patients']} patients, {v['n_annotations']} annotations"
                                for s, v in d["splits"].items())
            parts.append(f"### {title}\n\n{txt}")
    ap = C.RESULTS_DIR / "attention" / "attention_results.json"
    if ap.exists():
        d = json.loads(ap.read_text())
        parts.append(f"### Tiled attention\n\nAll correctness checks pass: **{d['all_pass']}** · {d['property_checks']}")
    return "\n\n".join(parts) or "_No saved results found in `results/` yet._"


def build_ui():
    info = SCREENER.info
    with gr.Blocks(title="TrialMatch — eligibility screening prototype") as demo:
        gr.Markdown("# TrialMatch: criterion-level clinical-trial eligibility screening")
        gr.HTML(f"<div style='background:#fff4e5;border:1px solid #f0b46c;color:#5a3b00;padding:10px;"
                f"border-radius:6px'><b>⚠ Notice:</b> {html.escape(DISCLAIMER)}</div>")
        gr.Markdown(f"Model: `{info.get('base_model')}` + adapter `{info.get('adapter') or 'none (base model)'}` · "
                    f"{info.get('dtype')} on {info.get('device')} · draft: "
                    f"`{(SCREENER.draft_info or {}).get('base_model', 'unavailable')}`")
        state = gr.State(None)
        with gr.Tab("Screen a patient"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        ex_dd = gr.Dropdown(list(VAL_EX.keys()), label="Load a validation-set example (optional)",
                                            value=None, scale=3)
                        ex_btn = gr.Button("Load demo example", scale=1)
                    note = gr.Textbox(label="Patient note (free text, or numbered '0. ...' sentences)", lines=10)
                    with gr.Accordion("How the model sees the note (numbered sentences)", open=False):
                        preview = gr.HTML()
                with gr.Column():
                    with gr.Row():
                        nct = gr.Textbox(label="ClinicalTrials.gov ID (optional)", placeholder="NCT01234567", scale=3)
                        fetch_btn = gr.Button("Fetch criteria", scale=1)
                    crit = gr.Textbox(label="Eligibility criteria (one per line; 'Inclusion Criteria:' / "
                                            "'Exclusion Criteria:' headings or INC:/EXC: prefixes)", lines=10)
                    mode = gr.Dropdown(list(MODES.keys()), value=list(MODES.keys())[0], label="Decoding engine")
                    run = gr.Button("Screen", variant="primary")
            verdict = gr.HTML()
            table = gr.Dataframe(label="Criterion-level results", wrap=True, interactive=False)
            sel = gr.Dropdown(label="Inspect criterion", choices=[], interactive=True)
            evid = gr.HTML()
            files = gr.File(label="Download report (HTML / JSON, with provenance)", file_count="multiple")

        with gr.Tab("Inference benchmark (live)"):
            gr.Markdown("Runs the selected decoding mechanisms on the current note and one criterion, "
                        "measured live on this machine (greedy, warm-up run first). Token equality is "
                        "checked against the first selected mode.")
            with gr.Row():
                bcrit = gr.Number(value=0, precision=0, label="Criterion # (from the criteria box, 0-based)")
                bmodes = gr.CheckboxGroup(list(MODES.keys()), value=list(MODES.keys())[:3], label="Modes")
            with gr.Row():
                bmax = gr.Slider(8, 256, value=64, step=8, label="max new tokens")
                bforce = gr.Checkbox(value=True, label="force exactly max new tokens (ignore EOS) for fair timing")
            brun = gr.Button("Run benchmark", variant="primary")
            bmsg = gr.Markdown()
            btab = gr.Dataframe(wrap=True, interactive=False)
            bplot = gr.Plot()

        with gr.Tab("Saved experiment results"):
            refresh = gr.Button("Refresh")
            saved = gr.Markdown(saved_results_md())

        # wiring
        def load_ex(key):
            if not key:
                return gr.update(), gr.update()
            return VAL_EX[key]["note"], "\n".join(VAL_EX[key]["crit"])
        ex_dd.change(load_ex, ex_dd, [note, crit])
        ex_btn.click(lambda: (EXAMPLE_NOTE, EXAMPLE_CRITERIA), None, [note, crit])
        note.change(numbered_preview, note, preview)
        fetch_btn.click(do_fetch, nct, crit)
        run.click(do_screen, [note, crit, mode], [verdict, table, sel, evid, files, state])
        sel.change(lambda res, i: evidence_view(res, i), [state, sel], evid)
        brun.click(do_bench, [note, crit, bcrit, bmodes, bmax, bforce], [btab, bplot, bmsg])
        refresh.click(saved_results_md, None, saved)
    return demo


def main():
    global SCREENER
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("TM_BASE_MODEL", C.TARGET_MODEL))
    ap.add_argument("--adapter", default=os.environ.get("TM_ADAPTER"))
    ap.add_argument("--draft_base", default=os.environ.get("TM_DRAFT_MODEL", C.DRAFT_MODEL))
    ap.add_argument("--draft_adapter", default=os.environ.get("TM_DRAFT_ADAPTER"))
    ap.add_argument("--no_draft", action="store_true")
    ap.add_argument("--share", action="store_true", help="public *.gradio.live link (use in Colab)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    if not a.adapter:
        print("WARNING: no --adapter given; running the BASE model (before fine-tuning).")
    print("Loading models ...")
    SCREENER = Screener(a.base, a.adapter, None if a.no_draft else a.draft_base, a.draft_adapter)
    print("Models loaded:", SCREENER.info)
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(server_name=a.host, server_port=a.port, share=a.share,
                                                   allowed_paths=[str(REPORT_DIR)])


if __name__ == "__main__":
    main()
