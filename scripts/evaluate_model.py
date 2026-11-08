"""Generate predictions for a split and compute all metrics.

Test-set protection: --split test requires --final, so the held-out set is only touched
deliberately, after all model/prompt/threshold choices were made on validation."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trialmatch import config as C
from trialmatch.data import read_jsonl
from trialmatch.evaluate import compute_metrics, error_analysis, plot_confusions, predict_records
from trialmatch.modeling import describe_model, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=C.TARGET_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final", action="store_true", help="required to evaluate on the test split")
    ap.add_argument("--tag", required=True, help="e.g. base / finetuned")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=C.MAX_NEW_TOKENS)
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "eval"))
    ap.add_argument("--limit", type=int, default=None, help="debug only")
    ap.add_argument("--skip_existing", action="store_true")
    a = ap.parse_args()
    if a.split == "test" and not a.final:
        sys.exit("Refusing to touch the test split without --final.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.tag}_{a.split}"
    if a.skip_existing and (out / f"metrics_{name}.json").exists():
        print("exists, skipping", name)
        return

    recs = read_jsonl(C.DATA_DIR / f"{a.split}.jsonl")
    if a.limit:
        recs = recs[:a.limit]
    model, tok = load_model(a.base, a.adapter)
    t0 = time.time()
    preds = predict_records(model, tok, recs, batch_size=a.batch_size, max_new_tokens=a.max_new_tokens)
    elapsed = time.time() - t0
    with open(out / f"predictions_{name}.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    m = compute_metrics(preds)
    m["run"] = {"model": describe_model(model), "split": a.split, "tag": a.tag,
                "decoding": {"strategy": "greedy", "do_sample": False, "num_beams": 1, "repetition_penalty": 1.0,
                             "model_default_generation_config": getattr(model, "_tm_original_generation_config", None),
                             "max_new_tokens": a.max_new_tokens, "batch_size": a.batch_size,
                             "padding": "left", "max_prompt_tokens": C.MAX_PROMPT_TOKENS},
                "elapsed_s": elapsed, "n": len(preds)}
    (out / f"metrics_{name}.json").write_text(json.dumps(m, indent=2))
    plot_confusions(m, out / "figures", name)
    if a.split == "test" or a.tag != "base":
        sents = {r["annotation_id"]: r["sentences"] for r in recs}
        ea = error_analysis(preds, sents, out / "errors", tag=name)
        print("error categories:", ea)
    o = m["overall"]
    print(f"[{name}] acc={o['accuracy']:.3f} macroF1={o['macro_f1']:.3f} "
          f"incl acc={m['inclusion'].get('accuracy', 0):.3f} excl acc={m['exclusion'].get('accuracy', 0):.3f} "
          f"evidence microF1={m['evidence']['micro']['f1']:.3f} invalid={m['invalid_output_rate']:.3f}")


if __name__ == "__main__":
    main()
