"""Before/after fine-tuning comparison on identical examples (same IDs, same decoding)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trialmatch import config as C


def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def preds(p):
    if not Path(p).exists():
        return {}
    return {json.loads(l)["annotation_id"]: json.loads(l) for l in open(p)}


def row(tag, split, m):
    o, i, e = m["overall"], m["inclusion"], m["exclusion"]
    return {"model": tag, "split": split, "n": o["n"], "accuracy": o["accuracy"], "macro_f1": o["macro_f1"],
            "incl_acc": i.get("accuracy"), "incl_macro_f1": i.get("macro_f1"),
            "excl_acc": e.get("accuracy"), "excl_macro_f1": e.get("macro_f1"),
            "evid_micro_p": m["evidence"]["micro"]["precision"], "evid_micro_r": m["evidence"]["micro"]["recall"],
            "evid_micro_f1": m["evidence"]["micro"]["f1"],
            "evid_macro_f1": m["evidence"]["macro_nonempty_gold"]["f1"],
            "invalid_rate": m["invalid_output_rate"], "json_valid_rate": m["json_valid_rate"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default=str(C.RESULTS_DIR / "eval"))
    ap.add_argument("--tags", nargs="+", default=["base", "finetuned"])
    a = ap.parse_args()
    d = Path(a.eval_dir)
    rows, flips = [], {}
    for split in ("val", "test"):
        for t in a.tags:
            m = load(d / f"metrics_{t}_{split}.json")
            if m:
                rows.append(row(t, split, m))
        if len(a.tags) >= 2:
            pb, pf = preds(d / f"predictions_{a.tags[0]}_{split}.jsonl"), preds(d / f"predictions_{a.tags[-1]}_{split}.jsonl")
            common = sorted(set(pb) & set(pf))
            if common:
                assert set(pb) == set(pf), "before/after must be evaluated on identical examples"
                fixed = sum(pb[i]["pred_label"] != pb[i]["gold_label"] and pf[i]["pred_label"] == pf[i]["gold_label"] for i in common)
                broke = sum(pb[i]["pred_label"] == pb[i]["gold_label"] and pf[i]["pred_label"] != pf[i]["gold_label"] for i in common)
                flips[split] = {"n_common": len(common), "fixed_by_finetuning": fixed, "broken_by_finetuning": broke}
    if not rows:
        sys.exit("no metrics found")
    cols = list(rows[0].keys())
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    md += ["", "Per-example label changes (same annotation IDs):", "```", json.dumps(flips, indent=2), "```"]
    (d / "before_after.md").write_text("\n".join(md))
    (d / "before_after.json").write_text(json.dumps({"rows": rows, "flips": flips}, indent=2))
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = ["accuracy", "macro_f1", "evid_micro_f1", "json_valid_rate"]
    for split in ("val", "test"):
        rs = [r for r in rows if r["split"] == split]
        if len(rs) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        w = 0.8 / len(rs)
        colors = ["#9aa5b1", "#3b6ea5", "#d1495b"]
        for j, r in enumerate(rs):
            ax.bar([i + j * w for i in range(len(metrics))], [r[m] or 0 for m in metrics], w,
                   label=r["model"], color=colors[j % 3])
        ax.set_xticks([i + w * (len(rs) - 1) / 2 for i in range(len(metrics))], metrics)
        ax.set_ylim(0, 1); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"Before vs after fine-tuning ({split})", fontsize=10)
        fig.tight_layout(); fig.savefig(d / f"before_after_{split}.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
