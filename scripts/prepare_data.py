"""Load the TrialGPT criterion annotations, drop GPT-4 columns, build the fixed
37/8/8 patient-grouped split (seed 42), run leakage checks, write the manifest and
model-facing JSONL files, and report prompt lengths / truncation."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trialmatch import config as C
from trialmatch.data import (assign_split, build_manifest, clean, leakage_checks, load_raw,
                             make_split, save_manifest, to_records, write_jsonl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_data", default=str(C.DATA_DIR))
    ap.add_argument("--out_results", default=str(C.RESULTS_DIR / "data"))
    ap.add_argument("--tokenizer", default=C.TARGET_MODEL)
    ap.add_argument("--no_tokenizer", action="store_true")
    a = ap.parse_args()
    out_res = Path(a.out_results)
    out_res.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    print("raw columns:", list(raw.columns))
    df, report = clean(raw)
    split = make_split(df)
    df = assign_split(df, split)
    checks = leakage_checks(df, split)
    man = build_manifest(df, split)
    man["leakage_checks"] = checks
    man["cleaning"] = report
    save_manifest(man, df, out_res)

    for s in ("train", "val", "test"):
        recs = to_records(df[df["split"] == s])
        write_jsonl(recs, Path(a.out_data) / f"{s}.jsonl")
        # guard: no GPT-4 field may appear in model-facing records
        assert not any(k.startswith("gpt4") or k == "explanation_correctness" for k in recs[0]), recs[0].keys()

    if not a.no_tokenizer:
        from transformers import AutoTokenizer
        from trialmatch.data import read_jsonl
        from trialmatch.prompts import build_prompt
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        stats = {}
        for s in ("train", "val", "test"):
            lens, trunc, lost = [], 0, 0
            for r in read_jsonl(Path(a.out_data) / f"{s}.jsonl"):
                ids, info = build_prompt(tok, r["sentences"], r["criterion_type"], r["criterion_text"],
                                         C.MAX_PROMPT_TOKENS, r["gold_evidence"])
                lens.append(info["prompt_tokens_full"]); trunc += info["truncated"]; lost += bool(info["gold_evidence_lost"])
            stats[s] = {"prompt_tokens_min": min(lens), "prompt_tokens_max": max(lens),
                        "prompt_tokens_mean": sum(lens) / len(lens), "budget": C.MAX_PROMPT_TOKENS,
                        "truncated_examples": trunc, "examples_losing_gold_evidence": lost}
        (out_res / "prompt_length_report.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats, indent=2))

    print(json.dumps({s: {k: v for k, v in i.items() if k != "annotations_per_patient"}
                      for s, i in man["splits"].items()}, indent=2))
    print("cleaning:", report)
    print("leakage checks:", checks)


if __name__ == "__main__":
    main()
