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
import datetime
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


def _status_color(label):
    return LABEL_COLORS.get(str(label).lower(), "#64748b")


def verdict_html(res):
    a = res["aggregate"]
    verdict = str(a.get("verdict", "Undetermined"))
    color = {"Potentially eligible": "#16835d", "Likely ineligible": "#c43d4b"}.get(verdict, "#b7791f")
    stats = [
        ("Criteria assessed", a.get("n_criteria", len(res.get("results", [])))),
        ("Failing criteria", a.get("n_failing", 0)),
        ("Missing information", a.get("n_missing_information", 0)),
        ("Unsupported", a.get("n_unsupported", 0)),
    ]
    stat_html = "".join(
        f'<div class="tm-stat"><div class="tm-stat-value">{html.escape(str(v))}</div>'
        f'<div class="tm-stat-label">{html.escape(k)}</div></div>' for k, v in stats
    )
    missing = "".join(f"<li>{html.escape(str(c))}</li>" for c in a.get("missing_information_criteria", [])) or "<li>None reported</li>"
    label_counts = {}
    for item in res.get("results", []):
        key = str(item.get("label", "unknown"))
        label_counts[key] = label_counts.get(key, 0) + 1
    breakdown = "".join(
        f'<span class="tm-breakdown-item"><i style="background:{_status_color(k)}"></i>'
        f'{html.escape(k)} <b>{v}</b></span>'
        for k, v in sorted(label_counts.items(), key=lambda pair: (-pair[1], pair[0]))
    )
    return f"""
    <section class="tm-verdict" style="--verdict:{color}">
      <div class="tm-verdict-top"><div><div class="tm-eyebrow">PRELIMINARY SCREENING OUTCOME</div>
      <div class="tm-verdict-title">{html.escape(verdict)}</div></div>
      <span class="tm-outcome-pill">{len(res.get('results', []))} criteria reviewed</span></div>
      <div class="tm-stat-grid">{stat_html}</div>
      <div class="tm-eyebrow">PREDICTION BREAKDOWN</div><div class="tm-breakdown">{breakdown}</div>
      <p class="tm-caveat">Preliminary aggregation of model predictions, not a ground-truth eligibility label or clinical decision. A qualified reviewer must verify the record and criteria.</p>
      <details class="tm-details"><summary>Missing information and aggregation rule</summary>
      <ul>{missing}</ul><p>{html.escape(str(a.get("rule", "")))}</p></details>
    </section>"""


def criterion_cards_html(res):
    """Expandable, self-contained criterion review cards."""
    sentence_map = {i: text for i, text in res.get("sentences", [])}
    cards = []
    for i, r in enumerate(res.get("results", [])):
        label = str(r.get("label", "unknown"))
        flags = []
        if r.get("unsupported"):
            flags.append("Unsupported · no evidence cited")
        if not r.get("parse_ok", True):
            flags.append("Output parse fallback")
        if r.get("truncated_note"):
            flags.append("Patient note truncated")
        evidence = r.get("evidence_ids", []) or []
        evidence_html = "".join(
            f'<li class="tm-inline-sentence cited"><b>[{int(eid)}]</b> '
            f'{html.escape(str(sentence_map.get(int(eid), "Sentence not available in displayed record.")))}</li>'
            for eid in evidence
        ) or '<div class="tm-info">No evidence sentence was cited. This prediction requires careful human review.</div>'
        flag_html = "".join(f'<span class="tm-flag">{html.escape(flag)}</span>' for flag in flags)
        explanation = html.escape(str(r.get("explanation") or "No explanation was returned by the model."))
        cards.append(f"""
        <details class="tm-criterion" style="--status:{_status_color(label)}">
          <summary class="tm-criterion-summary">
            <div class="tm-criterion-head"><span class="tm-index">{i + 1:02d}</span>
            <span class="tm-kind">{html.escape(str(r.get("criterion_type", "criterion")).title())}</span>
            <span class="tm-status">{html.escape(label)}</span></div>
            <div class="tm-criterion-text">{html.escape(str(r.get("criterion_text", "")))}</div>
            <div class="tm-criterion-meta">
              <span>Evidence IDs: <b>{html.escape(", ".join(map(str, evidence)) if evidence else "None")}</b></span>
              <span>Inference: <b>{float(r.get("latency_s", 0)):.2f}s</b></span>
              <span class="tm-expand-hint">Expand for explanation and evidence ▾</span>
            </div>
            <div class="tm-flags">{flag_html}</div>
          </summary>
          <div class="tm-card-expanded">
            <div class="tm-eyebrow">MODEL EXPLANATION</div>
            <div class="tm-explanation">{explanation}</div>
            <div class="tm-eyebrow">CITED PATIENT-RECORD EVIDENCE</div>
            <ol class="tm-inline-evidence">{evidence_html}</ol>
            {('<div class="tm-warning">One or more reliability flags are present. Do not treat this prediction as verified.</div>' if flags else '')}
            <details class="tm-details"><summary>View raw model output</summary>
              <pre class="tm-raw">{html.escape(str(r.get("raw_output", "")))}</pre>
            </details>
          </div>
        </details>""")
    return '<div class="tm-criteria-list">' + "".join(cards) + "</div>" if cards else ""
def evidence_view(res, idx):
    if not res:
        return '<div class="tm-empty">Run screening and select a criterion to inspect its evidence.</div>'
    idx = max(0, min(int(idx or 0), len(res["results"]) - 1))
    r = res["results"][idx]
    cited = set(r.get("evidence_ids", []))
    color = _status_color(r.get("label", ""))
    items = []
    for i, sentence in res.get("sentences", []):
        is_cited = i in cited
        cls = "tm-sentence cited" if is_cited else "tm-sentence"
        tag = '<span class="tm-cited-tag">CITED EVIDENCE</span>' if is_cited else ""
        items.append(
            f'<li class="{cls}"><span class="tm-sentence-id">[{i}]</span> '
            f'{html.escape(str(sentence))}{tag}</li>'
        )
    warnings = []
    if r.get("unsupported"):
        warnings.append(
            '<div class="tm-warning"><b>Unsupported prediction:</b> '
            'no evidence sentence was cited. Treat cautiously.</div>'
        )
    if r.get("truncated_note"):
        warnings.append(
            f'<div class="tm-warning"><b>Input truncation:</b> sentences '
            f'{html.escape(str(r.get("dropped_sentence_ids", [])))} '
            'were not provided to the model.</div>'
        )
    if not cited:
        warnings.append(
            '<div class="tm-info">No evidence IDs were returned for this criterion.</div>'
        )
    return f"""
    <section class="tm-evidence-card">
      <div class="tm-evidence-top"><div>
      <div class="tm-eyebrow">{html.escape(str(r.get("criterion_type", "")).upper())} CRITERION</div>
      <h3>{html.escape(str(r.get("criterion_text", "")))}</h3></div>
      <span class="tm-status" style="--status:{color}">{html.escape(str(r.get("label", "")))}</span></div>
      <div class="tm-explanation"><div class="tm-eyebrow">MODEL EXPLANATION</div>
      {html.escape(str(r.get("explanation", "No explanation returned.")))}</div>
      <div class="tm-evidence-meta"><b>Evidence sentence IDs:</b>
      {html.escape(", ".join(map(str, sorted(cited))) if cited else "None")}
      · <b>Inference time:</b> {float(r.get("latency_s", 0)):.2f}s</div>
      {''.join(warnings)}
      <div class="tm-eyebrow tm-record-label">PATIENT RECORD · CITED SENTENCES HIGHLIGHTED</div>
      <ol class="tm-sentence-list" start="0">{''.join(items)}</ol>
      <details class="tm-details"><summary>View raw model output</summary>
      <pre class="tm-raw">{html.escape(str(r.get("raw_output", "")))}</pre></details>
    </section>"""


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
    cards = criterion_cards_html(res)
    # persistent copy + downloads
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    h = REPORT_DIR / f"screening_report_{stamp}.html"
    j = REPORT_DIR / f"screening_report_{stamp}.json"
    h.write_text(report_html(res), encoding="utf-8")
    j.write_text(report_json(res), encoding="utf-8")
    choices = [
        (f"{i}: [{r['criterion_type'][:3]}] {r['criterion_text'][:70]}", i)
        for i, r in enumerate(res["results"])
    ]
    return (
        verdict_html(res),
        cards,
        gr.update(choices=choices, value=0),
        evidence_view(res, 0),
        [str(h), str(j)],
        res
    )


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
    ax[1].barh(df["mode"], df["total_ms"].fillna(0) / 1000.0, color="#b7791f")
    ax[1].set_xlabel("total latency (s)")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.invert_yaxis()
    fig.tight_layout()
    eq = ("all selected modes produced **identical tokens**" if df["same_tokens_as_first_mode"].all()
          else "some modes differ from the first mode (see table); on bf16/fp16 hardware this can be a "
               "floating-point near-tie")
    return df, fig, f"Benchmarked criterion: **[{t}] {html.escape(c[:150])}** — {eq}."


def saved_results_md():
    """Saved experiment summaries (from results/) plus the app's saved screening reports."""
    parts = []
    for title, rel in [("Before / after fine-tuning", "eval/before_after.md"),
                       ("KV-cache benchmark", "kvcache/kvcache_summary.md"),
                       ("Speculative decoding", "speculative/speculative_summary.md"),
                       ("Speculative decoding (FP32 equality check)", "speculative_fp32/speculative_summary.md"),
                       ("Model candidates", "research/model_candidates.md")]:
        p = C.RESULTS_DIR / rel
        if p.exists():
            parts.append(f"### {title}\n\n{p.read_text(encoding='utf-8')}")
    ap = C.RESULTS_DIR / "attention" / "attention_results.json"
    if ap.exists():
        d = json.loads(ap.read_text(encoding="utf-8"))
        parts.append(f"### Tiled attention\n\nAll correctness checks pass: **{d.get('all_pass')}** · "
                     f"{d.get('property_checks')}")
    files = sorted(REPORT_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True) if REPORT_DIR.exists() else []
    lines = ["### Saved screening reports", "", "| File | Type | Last modified |", "|---|---|---|"]
    for path in files[:30]:
        modified = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        lines.append(f"| `{path.name}` | {path.suffix.lstrip('.').upper()} | {modified} |")
    parts.append("\n".join(lines) if files else "### Saved screening reports\n\nNone yet.")
    return "\n\n".join(parts)


def build_ui():
    """Build the TrialMatch Gradio interface."""

    CSS = """
    :root {
        --tm-border: #dce5ec;
    }

    .gradio-container {
        max-width: 1450px !important;
    }

    .tm-header {
    background:linear-gradient(120deg,#12334b,#176b70);
    color:#ffffff !important;
    padding:30px 34px;
    border-radius:18px;
    margin-bottom:18px;
}

.tm-header h1,
.tm-header p,
.tm-header .tm-meta {
    color:#ffffff !important;
}

.tm-header h1 {
    font-size:34px;
    margin:0 0 5px;
    font-weight:850;
}

.tm-header p {
    font-size:15px;
    margin:0;
    opacity:.95;
}

.tm-header .tm-meta {
    font-size:10px;
    letter-spacing:.13em;
    margin-top:18px;
    opacity:.85;
    font-weight:800;
}
    .tm-meta {
        font-size:10px;
        letter-spacing:.13em;
        margin-top:18px;
        opacity:.75;
        font-weight:800;
    }
    .tm-notice {
        background:#fff8e9;
        border:1px solid #f0d9a7;
        border-left:5px solid #d59a2a;
        color:#76531b;
        border-radius:10px;
        padding:13px 16px;
        margin-bottom:18px;
        font-size:12px;
        line-height:1.6;
    }
    .tm-card {
        background:#fff;
        border:1px solid var(--tm-border);
        border-radius:15px;
        padding:18px;
        box-shadow:0 3px 12px rgba(20,43,69,.035);
    }
    .tm-section-title {
        color:#18354e;
        font-size:16px;
        font-weight:850;
        margin-bottom:4px;
    }
    .tm-subtitle {
        color:#64748b;
        font-size:12px;
        margin-bottom:15px;
        line-height:1.5;
    }
    .tm-verdict {
        background:#fff;
        border:1px solid var(--tm-border);
        border-left:6px solid var(--verdict);
        border-radius:16px;
        padding:22px;
        margin:18px 0;
        box-shadow:0 5px 18px rgba(20,43,69,.05);
    }
    .tm-verdict-top {
        display:flex;
        justify-content:space-between;
        align-items:center;
        gap:14px;
        flex-wrap:wrap;
    }
    .tm-eyebrow {
        color:#64748b;
        font-size:10px;
        letter-spacing:.12em;
        font-weight:800;
        margin-bottom:7px;
    }
    .tm-verdict-title {
        color:var(--verdict);
        font-size:26px;
        font-weight:800;
    }
    .tm-outcome-pill {
        border-radius:999px;
        padding:7px 12px;
        background:#edf4f7;
        color:#24536a;
        font-size:12px;
        font-weight:700;
    }
    .tm-stat-grid {
        display:grid;
        grid-template-columns:repeat(4,minmax(0,1fr));
        gap:10px;
        margin:20px 0 12px;
    }
    .tm-stat {
        background:#f6f9fb;
        border:1px solid #e7edf2;
        border-radius:12px;
        padding:13px;
    }
    .tm-stat-value {
        color:#18354e;
        font-size:25px;
        font-weight:800;
    }
    .tm-stat-label {
        color:#64748b;
        font-size:12px;
        margin-top:3px;
    }
    .tm-caveat {
        color:#64748b;
        font-size:12px;
        line-height:1.55;
        margin:14px 0 6px;
    }
    .tm-details {
        margin-top:12px;
        color:#475569;
        font-size:13px;
    }
    .tm-details summary {
        cursor:pointer;
        font-weight:700;
    }

    .tm-breakdown {
        display:flex;
        flex-wrap:wrap;
        gap:9px;
        margin:8px 0 14px;
    }
    .tm-breakdown-item {
        display:inline-flex;
        align-items:center;
        gap:7px;
        padding:7px 10px;
        background:#f6f9fb;
        border:1px solid #e7edf2;
        border-radius:999px;
        color:#475569;
        font-size:11px;
    }
    .tm-breakdown-item i {
        width:8px;
        height:8px;
        border-radius:50%;
        display:inline-block;
    }
    .tm-criterion {
        display:block;
        padding:0;
        overflow:hidden;
    }
    .tm-criterion-summary {
        display:block;
        list-style:none;
        cursor:pointer;
        padding:16px;
    }
    .tm-criterion-summary::-webkit-details-marker {
        display:none;
    }
    .tm-criterion[open] .tm-criterion-summary {
        background:#fbfdfe;
    }
    .tm-expand-hint {
        color:#167c80;
        font-weight:700;
    }
    .tm-flags {
        display:flex;
        flex-wrap:wrap;
        gap:6px;
        margin-top:9px;
    }
    .tm-flags .tm-flag {
        margin:0;
    }
    .tm-card-expanded {
        border-top:1px solid #e8eef3;
        padding:16px;
        background:#fff;
    }
    .tm-inline-evidence {
        padding-left:22px;
        margin:8px 0;
    }
    .tm-inline-sentence {
        padding:10px 12px;
        margin:6px 0;
        border-radius:8px;
        background:#e8f4ef;
        border-left:4px solid #16835d;
        color:#183f32;
        line-height:1.55;
    }
    .tm-inline-sentence b {
        color:#16835d;
        margin-right:5px;
    }

    .tm-criteria-list {
        display:grid;
        grid-template-columns:repeat(2,minmax(0,1fr));
        gap:12px;
        margin:12px 0 22px;
    }
    .tm-criterion {
        background:#fff;
        border:1px solid var(--tm-border);
        border-top:4px solid var(--status);
        border-radius:13px;
        padding:16px;
        box-shadow:0 3px 12px rgba(20,43,69,.035);
    }
    .tm-criterion-head {
        display:flex;
        align-items:center;
        gap:9px;
        flex-wrap:wrap;
        margin-bottom:12px;
    }
    .tm-index {
        color:#64748b;
        font-size:12px;
        font-weight:800;
    }
    .tm-kind {
        color:#64748b;
        font-size:11px;
        text-transform:uppercase;
        letter-spacing:.08em;
        font-weight:800;
    }
    .tm-status {
        display:inline-block;
        margin-left:auto;
        background:color-mix(in srgb,var(--status) 11%,white);
        color:var(--status);
        border:1px solid color-mix(in srgb,var(--status) 25%,white);
        padding:5px 9px;
        border-radius:999px;
        font-size:11px;
        font-weight:800;
    }
    .tm-criterion-text {
        color:#18354e;
        font-size:15px;
        line-height:1.5;
        font-weight:650;
    }
    .tm-criterion-meta {
        display:flex;
        justify-content:space-between;
        gap:10px;
        flex-wrap:wrap;
        border-top:1px solid #edf1f5;
        padding-top:11px;
        margin-top:13px;
        color:#64748b;
        font-size:11px;
    }
    .tm-criterion-meta b {
        color:#334155;
    }
    .tm-flag {
        color:#a45b12;
        background:#fff7e9;
        border-radius:7px;
        padding:7px 9px;
        margin-top:10px;
        font-size:11px;
    }
    .tm-evidence-card {
        background:#fff;
        border:1px solid var(--tm-border);
        border-radius:15px;
        padding:20px;
        margin:12px 0;
    }
    .tm-evidence-top {
        display:flex;
        justify-content:space-between;
        align-items:flex-start;
        gap:14px;
        flex-wrap:wrap;
    }
    .tm-evidence-top h3 {
        color:#18354e;
        font-size:17px;
        margin:0 0 14px;
        line-height:1.5;
    }
    .tm-explanation {
        background:#f4f8fa;
        border-radius:10px;
        padding:14px;
        color:#334155;
        line-height:1.65;
        margin:8px 0 12px;
    }
    .tm-evidence-meta {
        color:#64748b;
        font-size:12px;
        margin:10px 0 16px;
    }
    .tm-warning {
        background:#fff5f3;
        border-left:4px solid #c43d4b;
        color:#92313b;
        padding:10px 12px;
        margin:8px 0;
        border-radius:7px;
        font-size:12px;
    }
    .tm-info {
        background:#eff6ff;
        border-left:4px solid #4784b8;
        color:#315d7c;
        padding:10px 12px;
        margin:8px 0;
        border-radius:7px;
        font-size:12px;
    }
    .tm-record-label {
        margin-top:20px;
    }
    .tm-sentence-list {
        padding-left:28px;
        margin:10px 0;
    }
    .tm-sentence {
        padding:10px 12px;
        margin:6px 0;
        border-radius:8px;
        background:#f7f9fb;
        color:#475569;
        line-height:1.6;
    }
    .tm-sentence.cited {
        background:#e8f4ef;
        border-left:4px solid #16835d;
        color:#183f32;
    }
    .tm-sentence-id {
        color:#64748b;
        font-size:11px;
        font-weight:800;
        margin-right:5px;
    }
    .tm-cited-tag {
        display:inline-block;
        margin-left:8px;
        color:#16835d;
        font-size:9px;
        font-weight:900;
        letter-spacing:.08em;
    }
    .tm-raw {
        white-space:pre-wrap;
        overflow-wrap:anywhere;
        background:#f6f8fa;
        padding:12px;
        border-radius:8px;
        font-size:11px;
    }
    .tm-empty {
        color:#64748b;
        padding:18px;
        background:#fff;
        border:1px dashed var(--tm-border);
        border-radius:12px;
    }
    @media(max-width:800px) {
        .tm-stat-grid {grid-template-columns:repeat(2,minmax(0,1fr))}
        .tm-criteria-list {grid-template-columns:1fr}
    }
    """

    with gr.Blocks(
        title="TrialMatch | Clinical Trial Screening",
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="slate",
            neutral_hue="slate"
        ),
        css=CSS
    ) as demo:

        state = gr.State(None)

        # HEADER
        gr.HTML("""
        <div class="tm-header">
            <h1>TrialMatch</h1>
            <p>AI-assisted clinical trial eligibility screening</p>
            <div class="tm-meta">
                CRITERION-LEVEL ANALYSIS &nbsp; · &nbsp;
                EVIDENCE EXTRACTION &nbsp; · &nbsp;
                RESEARCH PROTOTYPE
            </div>
        </div>
        """)

        gr.HTML(
            f"""
            <div class="tm-notice">
                <b>Important research notice</b><br>
                {html.escape(DISCLAIMER)}
            </div>
            """
        )

        info = SCREENER.info
        gr.HTML(
            f"""<div class="tm-subtitle">Loaded model: <b>{html.escape(str(info.get('base_model')))}</b>
            + adapter <b>{html.escape(str(info.get('adapter') or 'none (BASE model)'))}</b> ·
            {html.escape(str(info.get('dtype')))} on {html.escape(str(info.get('device')))} · draft:
            <b>{html.escape(str((SCREENER.draft_info or {}).get('base_model', 'unavailable')))}</b>
            {'+ fine-tuned adapter' if (SCREENER.draft_info or {}).get('adapter') else ''}</div>"""
        )

        # NAVIGATION
        with gr.Tabs():

            # ---------------------------------------------------------
            # SCREENING TAB
            # ---------------------------------------------------------
            with gr.Tab("Patient screening"):

                gr.Markdown("## New screening")
                gr.Markdown(
                    "Enter a patient record and the trial's eligibility criteria "
                    "to evaluate each criterion individually."
                )

                with gr.Row(equal_height=False):

                    # PATIENT PANEL
                    with gr.Column(scale=1, elem_classes=["tm-card"]):

                        gr.HTML("""
                        <div class="tm-section-title">01 · Patient information</div>
                        <div class="tm-subtitle">
                            Provide the patient record for analysis.
                        </div>
                        """)

                        with gr.Row():
                            ex_dd = gr.Dropdown(
                                choices=list(VAL_EX.keys()),
                                label="Validation example",
                                value=None,
                                scale=3
                            )

                            ex_btn = gr.Button(
                                "Load demo",
                                scale=1
                            )

                        note = gr.Textbox(
                            label="Patient clinical note",
                            placeholder=(
                                "Enter the patient's clinical history, "
                                "symptoms, laboratory values and relevant findings..."
                            ),
                            lines=12,
                            max_lines=20
                        )

                        with gr.Accordion(
                            "Preview numbered patient sentences",
                            open=False
                        ):
                            preview = gr.HTML()

                    # CRITERIA PANEL
                    with gr.Column(scale=1, elem_classes=["tm-card"]):

                        gr.HTML("""
                        <div class="tm-section-title">02 · Trial information</div>
                        <div class="tm-subtitle">
                            Enter criteria manually or fetch them from ClinicalTrials.gov.
                        </div>
                        """)

                        with gr.Row():
                            nct = gr.Textbox(
                                label="ClinicalTrials.gov identifier",
                                placeholder="e.g. NCT01234567",
                                scale=3
                            )

                            fetch_btn = gr.Button(
                                "Fetch criteria",
                                scale=1
                            )

                        crit = gr.Textbox(
                            label="Eligibility criteria",
                            placeholder=(
                                "Inclusion Criteria:\n"
                                "- Age 40 to 80 years\n"
                                "- Stable angina with a positive stress test\n\n"
                                "Exclusion Criteria:\n"
                                "- Severe renal impairment"
                            ),
                            lines=12,
                            max_lines=20
                        )

                        mode = gr.Dropdown(
                            choices=list(MODES.keys()),
                            value=list(MODES.keys())[0],
                            label="Inference / decoding method"
                        )

                # RUN BUTTON
                with gr.Row():
                    run = gr.Button(
                        "Run eligibility screening",
                        variant="primary",
                        size="lg"
                    )
                # RESULTS
                gr.Markdown("---")
                gr.Markdown("## Screening results")
                gr.Markdown(
                    "Review the preliminary outcome, individual criterion predictions "
                    "and the supporting patient evidence."
                )

                verdict = gr.HTML()

                gr.Markdown("### Criterion-by-criterion assessment")
                gr.Markdown(
                    "Expand any criterion to review the model explanation, "
                    "cited patient-record sentences, raw output and reliability warnings."
                )

                table = gr.HTML(
                    value='<div class="tm-empty">Criterion cards will appear here after screening.</div>',
                    label="Criterion-by-criterion results"
                )

                gr.Markdown("### Evidence and explanation")
                gr.Markdown(
                    "The expandable criterion cards above include the explanation "
                    "and cited evidence. Use this focused inspector for a full-record "
                    "view of any selected criterion."
                )

                sel = gr.Dropdown(
                    label="Optional full-record evidence inspector",
                    choices=[],
                    interactive=True
                )

                evid = gr.HTML()

                gr.Markdown("### Export screening report")
                gr.Markdown(
                    "Download a self-contained HTML report or structured JSON output."
                )

                files = gr.File(
                    label="Download screening report",
                    file_count="multiple"
                )

            # ---------------------------------------------------------
            # BENCHMARK TAB
            # ---------------------------------------------------------
            with gr.Tab("Inference benchmark"):

                gr.Markdown("## Live inference benchmark")
                gr.Markdown(
                    "Compare decoding methods using the current patient note "
                    "and one selected eligibility criterion. Measurements are "
                    "performed on the machine running TrialMatch."
                )

                with gr.Group():
                    bcrit = gr.Number(
                        value=0,
                        precision=0,
                        label="Criterion index (0-based)"
                    )

                    bmodes = gr.CheckboxGroup(
                        choices=list(MODES.keys()),
                        value=list(MODES.keys())[:3],
                        label="Decoding methods to compare"
                    )

                    with gr.Row():
                        bmax = gr.Slider(
                            minimum=8,
                            maximum=256,
                            value=64,
                            step=8,
                            label="Maximum generated tokens"
                        )

                        bforce = gr.Checkbox(
                            value=True,
                            label="Force exact output length for timing"
                        )

                    brun = gr.Button(
                        "Run benchmark",
                        variant="primary"
                    )

                bmsg = gr.Markdown()

                gr.Markdown("### Benchmark measurements")

                btab = gr.Dataframe(
                    wrap=True,
                    interactive=False
                )

                bplot = gr.Plot()

            # ---------------------------------------------------------
            # SAVED RESULTS TAB
            # ---------------------------------------------------------
            with gr.Tab("Research results"):

                gr.Markdown("## Saved experiment results")
                gr.Markdown(
                    "Review the experimental results generated by the "
                    "training, evaluation and optimization workflows."
                )

                refresh = gr.Button("Refresh experiment results")

                saved = gr.Markdown(
                    value=saved_results_md()
                )

        # FOOTER
        gr.HTML("""
        <div class="tm-footer">
            TrialMatch · Educational research prototype ·
            Not intended for clinical decision-making
        </div>
        """)

        # -------------------------------------------------------------
        # EVENT WIRING — existing application logic retained
        # -------------------------------------------------------------

        def load_ex(key):
            if not key:
                return gr.update(), gr.update()

            return (
                VAL_EX[key]["note"],
                "\n".join(VAL_EX[key]["crit"])
            )

        ex_dd.change(
            load_ex,
            ex_dd,
            [note, crit]
        )

        ex_btn.click(
            lambda: (EXAMPLE_NOTE, EXAMPLE_CRITERIA),
            None,
            [note, crit]
        )

        note.change(
            numbered_preview,
            note,
            preview
        )

        fetch_btn.click(
            do_fetch,
            nct,
            crit
        )

        run.click(
            do_screen,
            [note, crit, mode],
            [verdict, table, sel, evid, files, state]
        )

        sel.change(
            lambda res, i: evidence_view(res, i),
            [state, sel],
            evid
        )

        brun.click(
            do_bench,
            [note, crit, bcrit, bmodes, bmax, bforce],
            [btab, bplot, bmsg]
        )

        refresh.click(
            saved_results_md,
            None,
            saved
        )

    return demo


DEFAULT_ADAPTER = ROOT / "weights" / "qwen15_lora"
DEFAULT_DRAFT_ADAPTER = ROOT / "weights" / "qwen05_lora"


def _default(env, path):
    """Env var wins; otherwise use the standard weights/ folder if it holds an adapter."""
    if os.environ.get(env):
        return os.environ[env]
    return str(path) if (path / "adapter_config.json").exists() else None


def main():
    global SCREENER
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("TM_BASE_MODEL", C.TARGET_MODEL))
    ap.add_argument("--adapter", default=_default("TM_ADAPTER", DEFAULT_ADAPTER))
    ap.add_argument("--draft_base", default=os.environ.get("TM_DRAFT_MODEL", C.DRAFT_MODEL))
    ap.add_argument("--draft_adapter", default=_default("TM_DRAFT_ADAPTER", DEFAULT_DRAFT_ADAPTER))
    ap.add_argument("--no_draft", action="store_true")
    ap.add_argument("--share", action="store_true", help="public *.gradio.live link (use in Colab)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to allow other devices on the network")
    ap.add_argument("--no_browser", action="store_true", help="do not open a browser tab automatically")
    a = ap.parse_args()

    if not a.adapter:
        print(
            "WARNING: no adapter found in weights/qwen15_lora "
            "and none given; running the BASE model."
        )

    print(
        "Loading models (first start downloads ~4 GB of base weights "
        "from Hugging Face) ..."
    )

    SCREENER = Screener(
        a.base,
        a.adapter,
        None if a.no_draft else a.draft_base,
        a.draft_adapter
    )

    print("Models loaded:", SCREENER.info)

    demo = build_ui()

    print(
        f"\n>>> Open http://127.0.0.1:{a.port} in your browser "
        "(keep this window open) <<<\n"
    )

    demo.queue(default_concurrency_limit=1).launch(
        server_name=a.host,
        server_port=a.port,
        share=a.share,
        inbrowser=not (a.no_browser or a.share),
        allowed_paths=[str(REPORT_DIR)]
    )


if __name__ == "__main__":
    main()