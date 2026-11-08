"""Batched prediction for evaluation, classification + evidence metrics, error analysis."""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import config as C
from .prompts import build_explanation, build_prompt, parse_output


# ------------------------------------------------------------------------------------
# Prediction (HF generate, greedy; the custom decoders are validated against this path)
# ------------------------------------------------------------------------------------
def predict_records(model, tok, recs: List[Dict], batch_size: int = 8,
                    max_new_tokens: int = C.MAX_NEW_TOKENS,
                    max_prompt_tokens: int = C.MAX_PROMPT_TOKENS) -> List[Dict]:
    import torch
    from .modeling import stop_token_ids
    dev = next(model.parameters()).device
    stops = sorted(stop_token_ids(tok))
    prompts = [build_prompt(tok, r["sentences"], r["criterion_type"], r["criterion_text"],
                            max_prompt_tokens, r.get("gold_evidence")) for r in recs]
    order = np.argsort([-len(p[0]) for p in prompts])   # length-sorted batches = less padding
    out = [None] * len(recs)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        L = max(len(prompts[i][0]) for i in idx)
        ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):       # left padding for decoder-only generation
            p = prompts[i][0]
            ids[j, L - len(p):] = torch.tensor(p)
            att[j, L - len(p):] = 1
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev),
                                 max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                 temperature=None, top_p=None, top_k=None,
                                 repetition_penalty=1.0, eos_token_id=stops, pad_token_id=tok.pad_token_id)
        dt = (time.time() - t0) / len(idx)
        for j, i in enumerate(idx):
            new = gen[j, L:].tolist()
            new = [t for t in new if t != tok.pad_token_id]
            text = tok.decode(new, skip_special_tokens=True)
            r = recs[i]
            pinfo = prompts[i][1]
            parsed = parse_output(text, r["criterion_type"], pinfo["kept_sentence_ids"])
            smap = dict(r["sentences"])
            out[i] = {
                "annotation_id": r["annotation_id"], "patient_id": r["patient_id"],
                "trial_id": r["trial_id"], "criterion_type": r["criterion_type"],
                "criterion_text": r["criterion_text"],
                "gold_label": r["gold_label"], "gold_evidence": r["gold_evidence"],
                "pred_label": parsed["label"], "pred_evidence": parsed["evidence"],
                "evidence_text": {i_: smap[i_] for i_ in parsed["evidence"] if i_ in smap},
                "explanation": build_explanation(parsed["label"], r["criterion_type"],
                                                 r["criterion_text"], parsed["evidence"], smap),
                "parse_ok": parsed["parse_ok"], "json_ok": parsed["json_ok"],
                "raw_output": text, "n_new_tokens": len(new), "sec_per_example": dt,
                "truncated": pinfo["truncated"], "gold_evidence_lost": pinfo["gold_evidence_lost"],
                "prompt_tokens": pinfo["prompt_tokens"],
            }
    return out


# ------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------
def _cls_metrics(gold: List[str], pred: List[str], labels: List[str]) -> Dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    if not gold:
        return {"n": 0}
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    present = [l for l in labels if l in set(gold) | set(pred)]
    _, _, f_present, _ = precision_recall_fscore_support(gold, pred, labels=present, zero_division=0)
    return {
        "n": len(gold),
        "accuracy": float(accuracy_score(gold, pred)),
        # macro-F1 over labels occurring in gold or predictions of this subset
        "macro_f1": float(np.mean(f_present)) if len(present) else 0.0,
        # macro-F1 over labels with gold support only (stricter view for small subsets)
        "macro_f1_gold_labels": float(np.mean([f[i] for i, l in enumerate(labels) if s[i] > 0]))
        if any(s) else 0.0,
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i])} for i, l in enumerate(labels)},
        "labels": labels,
        "confusion_matrix": confusion_matrix(gold, pred, labels=labels).tolist(),
    }


def _evidence_metrics(rows: List[Dict]) -> Dict:
    """Micro: pool (record, sentence) pairs over all records (empty sets contribute 0).
    Macro: average per-record P/R/F1 over records with non-empty gold evidence; a record
    with empty prediction there scores P=R=F1=0. Records with empty gold evidence are
    reported separately as 'correct abstention' (predicted set also empty)."""
    tp = fp = fn = 0
    per = []
    empty_gold, empty_gold_abstain = 0, 0
    for r in rows:
        g, p = set(r["gold_evidence"]), set(r["pred_evidence"])
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        if not g:
            empty_gold += 1
            empty_gold_abstain += int(not p)
            continue
        pr = len(g & p) / len(p) if p else 0.0
        rc = len(g & p) / len(g)
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per.append((pr, rc, f1))
    mp = tp / (tp + fp) if tp + fp else 0.0
    mr = tp / (tp + fn) if tp + fn else 0.0
    mf = 2 * mp * mr / (mp + mr) if mp + mr else 0.0
    arr = np.array(per) if per else np.zeros((1, 3))
    return {
        "micro": {"precision": mp, "recall": mr, "f1": mf, "tp": tp, "fp": fp, "fn": fn},
        "macro_nonempty_gold": {"precision": float(arr[:, 0].mean()), "recall": float(arr[:, 1].mean()),
                                "f1": float(arr[:, 2].mean()), "n_records": len(per)},
        "empty_gold": {"n_records": empty_gold, "correct_empty_prediction": empty_gold_abstain,
                       "rate": empty_gold_abstain / empty_gold if empty_gold else None},
    }


def compute_metrics(preds: List[Dict]) -> Dict:
    res = {"n": len(preds),
           "invalid_output_rate": float(np.mean([not p["parse_ok"] for p in preds])) if preds else 0,
           "json_valid_rate": float(np.mean([p["json_ok"] for p in preds])) if preds else 0,
           "truncated_examples": int(sum(p["truncated"] for p in preds)),
           "examples_with_gold_evidence_lost": int(sum(bool(p["gold_evidence_lost"]) for p in preds))}
    res["overall"] = _cls_metrics([p["gold_label"] for p in preds], [p["pred_label"] for p in preds],
                                  C.ALL_LABELS)
    for t, labs in (("inclusion", C.INCLUSION_LABELS), ("exclusion", C.EXCLUSION_LABELS)):
        sub = [p for p in preds if p["criterion_type"] == t]
        res[t] = _cls_metrics([p["gold_label"] for p in sub], [p["pred_label"] for p in sub], labs)
        res[t]["evidence"] = _evidence_metrics(sub)
    res["evidence"] = _evidence_metrics(preds)
    res["mean_new_tokens"] = float(np.mean([p["n_new_tokens"] for p in preds])) if preds else 0
    return res


def plot_confusions(metrics: Dict, out_dir: Path, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    short = {"included": "incl", "not included": "not incl", "excluded": "excl",
             "not excluded": "not excl", "not enough information": "NEI", "not applicable": "N/A"}
    paths = []
    for key in ("overall", "inclusion", "exclusion"):
        m = metrics[key]
        if not m.get("n"):
            continue
        cm = np.array(m["confusion_matrix"])
        labs = [short[l] for l in m["labels"]]
        fig, ax = plt.subplots(figsize=(1.1 * len(labs) + 2, 1.0 * len(labs) + 1.6))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labs)), labs, rotation=35, ha="right")
        ax.set_yticks(range(len(labs)), labs)
        ax.set_xlabel("predicted"); ax.set_ylabel("expert (gold)")
        thr = cm.max() / 2 if cm.max() else 1
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > thr else "black", fontsize=9)
        ax.set_title(f"{tag} — {key} (n={m['n']}, acc={m['accuracy']:.2f}, macro-F1={m['macro_f1']:.2f})",
                     fontsize=9)
        fig.tight_layout()
        p = out_dir / f"confusion_{tag}_{key}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(str(p))
    return paths


# ------------------------------------------------------------------------------------
# Error analysis
# ------------------------------------------------------------------------------------
def categorize(p: Dict) -> List[str]:
    cats = []
    g, y = p["gold_label"], p["pred_label"]
    ge, pe = set(p["gold_evidence"]), set(p["pred_evidence"])
    dec = lambda l: l not in C.UNDECIDED_LABELS
    if not p["parse_ok"]:
        cats.append("invalid_output")
    if dec(y) and not pe:
        cats.append("unsupported_decision")          # decisive label with no cited evidence
    if g != y:
        if dec(g) and y == "not enough information":
            cats.append("missed_information")        # info was in the note, model said NEI
        elif g == "not enough information" and dec(y):
            cats.append("decision_without_information")  # model decided where experts saw none
        elif dec(g) and dec(y):
            cats.append("polarity_error")            # included<->not included etc.
        elif "not applicable" in (g, y):
            cats.append("applicability_error")
        if ge and ge == pe:
            cats.append("contradiction_same_evidence_wrong_label")
    else:
        if ge != pe:
            f1 = (2 * len(ge & pe) / (len(ge) + len(pe))) if (ge or pe) else 1.0
            if f1 < 0.5:
                cats.append("right_label_wrong_evidence")
    if p.get("gold_evidence_lost"):
        cats.append("truncation_lost_evidence")
    return cats


def error_analysis(preds: List[Dict], sentences_by_ann: Dict[int, List], out_dir: Path,
                   per_category: int = 2, tag: str = "finetuned_test") -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_cat = defaultdict(list)
    for p in preds:
        for c in categorize(p):
            by_cat[c].append(p)
    counts = {c: len(v) for c, v in by_cat.items()}
    picked, seen = [], set()
    for c in sorted(by_cat, key=lambda c: -len(by_cat[c])):
        for p in by_cat[c][:per_category]:
            if p["annotation_id"] not in seen:
                picked.append((c, p)); seen.add(p["annotation_id"])
    # top up to at least 8 distinct errors
    rest = [p for p in preds if p["gold_label"] != p["pred_label"] and p["annotation_id"] not in seen]
    for p in rest[:max(0, 8 - len(picked))]:
        picked.append(("other_label_error", p)); seen.add(p["annotation_id"])

    lines = [f"# Error analysis ({tag})", "", "Category counts:", ""]
    lines += [f"- **{c}**: {n}" for c, n in sorted(counts.items(), key=lambda x: -x[1])]
    lines.append("")
    for k, (c, p) in enumerate(picked, 1):
        smap = dict(sentences_by_ann.get(p["annotation_id"], []))
        lines += [f"## Case {k}: {c}  (annotation {p['annotation_id']}, {p['patient_id']}, {p['trial_id']})",
                  f"- Criterion ({p['criterion_type']}): {p['criterion_text']}",
                  f"- Expert: **{p['gold_label']}**, evidence {p['gold_evidence']}",
                  f"- Model: **{p['pred_label']}**, evidence {p['pred_evidence']}",
                  f"- Raw output: `{p['raw_output'].strip()[:300]}`",
                  "- Gold evidence text: " + ("; ".join(f"[{i}] {smap.get(i, '?')}" for i in p['gold_evidence']) or "(none)"),
                  "- Predicted evidence text: " + ("; ".join(f"[{i}] {smap.get(i, '?')}" for i in p['pred_evidence']) or "(none)"),
                  "- Analysis: _write your interpretation here_", ""]
    (out_dir / f"error_analysis_{tag}.md").write_text("\n".join(lines))
    with open(out_dir / f"errors_{tag}.jsonl", "w") as f:
        for p in preds:
            cats = categorize(p)
            if cats:
                f.write(json.dumps({**p, "categories": cats}) + "\n")
    return {"category_counts": counts, "n_selected": len(picked)}
