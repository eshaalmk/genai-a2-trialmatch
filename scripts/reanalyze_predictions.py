"""CPU-only re-analysis of saved predictions (no model needed).

1. Separates LABEL-ERROR categories (mutually exclusive, sum to the number of wrong labels)
   from EVIDENCE FLAGS (can apply to correct predictions too).
2. Compares the model's "decisive label with no cited evidence" rate against the SAME rate
   in the expert annotations, so 'unsupported_decision' is judged against the gold convention.
3. Patient-level bootstrap confidence intervals (resampling the 8 patients, since
   annotations of one patient are not independent) for accuracy, macro-F1 and the
   fine-tuned-minus-base difference.

Usage: python scripts/reanalyze_predictions.py --split test
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trialmatch import config as C

UND = C.UNDECIDED_LABELS


def load(p):
    return [json.loads(l) for l in open(p)]


def label_error_type(p):
    g, y = p["gold_label"], p["pred_label"]
    if g == y:
        return None
    dec = lambda l: l not in UND
    if dec(g) and y == "not enough information":
        return "missed_information"
    if g == "not enough information" and dec(y):
        return "decision_without_information"
    if dec(g) and dec(y):
        return "polarity_error"
    return "applicability_error"


def macro_f1(gold, pred):
    from sklearn.metrics import f1_score
    labs = sorted(set(gold) | set(pred))
    return f1_score(gold, pred, labels=labs, average="macro", zero_division=0)


def bootstrap(base, ft, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pats = sorted({p["patient_id"] for p in ft})
    bb = defaultdict(list); fb = defaultdict(list)
    for p in base:
        bb[p["patient_id"]].append(p)
    for p in ft:
        fb[p["patient_id"]].append(p)
    out = defaultdict(list)
    for _ in range(n):
        s = rng.choice(pats, size=len(pats), replace=True)
        B = [p for x in s for p in bb[x]]
        F = [p for x in s for p in fb[x]]
        for tag, R in (("base", B), ("ft", F)):
            g = [p["gold_label"] for p in R]; y = [p["pred_label"] for p in R]
            out[f"{tag}_acc"].append(np.mean([a == b for a, b in zip(g, y)]))
            out[f"{tag}_macro_f1"].append(macro_f1(g, y))
        out["diff_acc"].append(out["ft_acc"][-1] - out["base_acc"][-1])
        out["diff_macro_f1"].append(out["ft_macro_f1"][-1] - out["base_macro_f1"][-1])
    return {k: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
            for k, v in out.items()}


def summarize(preds):
    n = len(preds)
    wrong = [p for p in preds if p["gold_label"] != p["pred_label"]]
    err = Counter(label_error_type(p) for p in wrong)
    gold_dec = [p for p in preds if p["gold_label"] not in UND]
    pred_dec = [p for p in preds if p["pred_label"] not in UND]
    pair = Counter((p["gold_label"], p["pred_label"]) for p in wrong)
    return {
        "n": n, "n_label_errors": len(wrong),
        "label_error_types (sum = n_label_errors)": dict(err),
        "top_confusions (gold -> pred)": {f"{g} -> {y}": c for (g, y), c in pair.most_common(8)},
        "gold_decisive_with_empty_gold_evidence": f"{sum(not p['gold_evidence'] for p in gold_dec)}/{len(gold_dec)}",
        "pred_decisive_with_empty_pred_evidence": f"{sum(not p['pred_evidence'] for p in pred_dec)}/{len(pred_dec)}",
        "pred_unsupported_among_CORRECT_labels": sum(1 for p in pred_dec if not p["pred_evidence"] and p["gold_label"] == p["pred_label"]),
        "pred_unsupported_among_WRONG_labels": sum(1 for p in pred_dec if not p["pred_evidence"] and p["gold_label"] != p["pred_label"]),
        "records_with_empty_gold_evidence": sum(not p["gold_evidence"] for p in preds),
        "empty_gold_and_empty_pred": sum(not p["gold_evidence"] and not p["pred_evidence"] for p in preds),
        "nonempty_gold_but_empty_pred": sum(bool(p["gold_evidence"]) and not p["pred_evidence"] for p in preds),
        "gold_label_counts": dict(Counter(p["gold_label"] for p in preds)),
        "pred_label_counts": dict(Counter(p["pred_label"] for p in preds)),
        "patients": len({p["patient_id"] for p in preds}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--eval_dir", default=str(C.RESULTS_DIR / "eval"))
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    d = Path(a.eval_dir)
    base = load(d / f"predictions_base_{a.split}.jsonl")
    ft = load(d / f"predictions_finetuned_{a.split}.jsonl")
    assert sorted(p["annotation_id"] for p in base) == sorted(p["annotation_id"] for p in ft)
    res = {"base": summarize(base), "finetuned": summarize(ft), "bootstrap_over_patients": bootstrap(base, ft, a.n_boot)}
    # trivial baseline: majority TRAIN label per criterion type (no model)
    tr = C.DATA_DIR / "train.jsonl"
    if tr.exists():
        from sklearn.metrics import f1_score
        trl = [json.loads(l) for l in open(tr)]
        maj = {t: Counter(r["gold_label"] for r in trl if r["criterion_type"] == t).most_common(1)[0][0]
               for t in ("inclusion", "exclusion")}
        g = [p["gold_label"] for p in ft]; y = [maj[p["criterion_type"]] for p in ft]
        res["majority_baseline"] = {"rule": maj, "accuracy": float(np.mean([x == z for x, z in zip(g, y)])),
                                    "macro_f1": float(macro_f1(g, y)),
                                    "train_label_counts": dict(Counter(r["gold_label"] for r in trl))}
    from sklearn.metrics import precision_recall_fscore_support
    labs = C.ALL_LABELS
    pr, rc, f1, sup = precision_recall_fscore_support([p["gold_label"] for p in ft], [p["pred_label"] for p in ft],
                                                      labels=labs, zero_division=0)
    res["finetuned_per_class"] = {l: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
                                      "support": int(sup[i])} for i, l in enumerate(labs)}
    (d / f"reanalysis_{a.split}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
