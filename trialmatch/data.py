"""Dataset loading, note sentence parsing, the fixed patient-grouped split, manifest and
leakage checks."""
from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config as C

Sentence = Tuple[int, str]


# ------------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Load the full TrialGPT criterion-annotation table (single 'train' split on HF)."""
    from datasets import load_dataset
    ds = load_dataset(C.DATASET_ID, split="train")
    return ds.to_pandas()


def clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Drop GPT-4 columns immediately and normalise fields. Returns (clean_df, report)."""
    report = {"rows_raw": int(len(df))}
    present_forbidden = [c for c in C.FORBIDDEN_COLUMNS if c in df.columns]
    df = df.drop(columns=present_forbidden + [c for c in C.IGNORED_COLUMNS if c in df.columns])
    report["dropped_gpt4_columns"] = present_forbidden

    df = df.copy()
    df["criterion_text"] = df["criterion_text"].fillna("").astype(str).str.strip()
    empty = df["criterion_text"] == ""
    report["rows_dropped_empty_criterion"] = int(empty.sum())
    df = df[~empty]

    df["criterion_type"] = df["criterion_type"].str.strip().str.lower()
    df["expert_eligibility"] = df["expert_eligibility"].str.strip().str.lower()
    df["expert_sentences"] = df["expert_sentences"].apply(parse_id_list)

    bad_type = ~df.apply(lambda r: r["expert_eligibility"] in C.labels_for(r["criterion_type"]), axis=1)
    report["rows_label_inconsistent_with_type"] = int(bad_type.sum())
    # keep them (expert ground truth), but they are counted and reported
    df["n_sentences"] = df["note"].apply(lambda n: len(parse_note(n)))
    out_of_range = df.apply(lambda r: any(i >= r["n_sentences"] for i in r["expert_sentences"]), axis=1)
    report["rows_evidence_id_out_of_range"] = int(out_of_range.sum())
    df["expert_sentences"] = df.apply(
        lambda r: [i for i in r["expert_sentences"] if 0 <= i < r["n_sentences"]], axis=1)
    report["rows_clean"] = int(len(df))
    return df.reset_index(drop=True), report


def parse_id_list(x) -> List[int]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return sorted({int(i) for i in x})
    s = str(x).strip()
    if not s:
        return []
    try:
        v = json.loads(s)
    except Exception:
        try:
            v = ast.literal_eval(s)
        except Exception:
            v = [int(t) for t in re.findall(r"\d+", s)]
    if isinstance(v, int):
        v = [v]
    return sorted({int(i) for i in v})


# ------------------------------------------------------------------------------------
# Sentence parsing. TrialGPT notes are already split and numbered "0. ...", "1. ...".
# The IDs in expert_sentences refer to those numbers, so we must keep them exactly.
# ------------------------------------------------------------------------------------
_LINE = re.compile(r"^\s*(\d+)\.\s+(.*\S)\s*$")


def parse_note(note: str) -> List[Sentence]:
    """Parse a numbered note into [(id, text)], preserving the original IDs."""
    note = note or ""
    lines = [l for l in note.splitlines() if l.strip()]
    out: List[Sentence] = []
    ok = bool(lines)
    for l in lines:
        m = _LINE.match(l)
        if m and int(m.group(1)) == len(out):
            out.append((len(out), m.group(2).strip()))
        elif out:  # continuation line of the previous sentence
            out[-1] = (out[-1][0], out[-1][1] + " " + l.strip())
        else:
            ok = False
            break
    if ok and len(out) > 1:
        return out
    line_result = out if ok else []
    # Fallback: numbers inline ("0. text 1. text"); search sequentially for "i. "
    marks, pos, i = [], 0, 0
    while True:
        m = re.compile(rf"(?:^|(?<=\s)){i}\.\s").search(note, pos)
        if not m:
            break
        marks.append((i, m.start(), m.end()))
        pos, i = m.end(), i + 1
    if marks:
        res = []
        for j, (sid, _, end) in enumerate(marks):
            stop = marks[j + 1][1] if j + 1 < len(marks) else len(note)
            res.append((sid, note[end:stop].strip()))
        if len(res) > len(line_result):
            return res
    return line_result


def split_sentences(text: str) -> List[Sentence]:
    """For user-supplied notes in the app: keep numbering if present, else split."""
    parsed = parse_note(text)
    if parsed:
        return parsed
    text = re.sub(r"[ \t]+", " ", text or "").strip()
    if not text:
        return []
    chunks: List[str] = []
    for para in re.split(r"\n+", text):
        # split after . ! ? followed by whitespace + capital/digit, but not after common abbreviations
        parts = re.split(r"(?<=[.!?])(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bvs\.)(?<!\be\.g\.)(?<!\bi\.e\.)"
                         r"\s+(?=[A-Z0-9\"(])", para.strip())
        chunks.extend(p.strip() for p in parts if p.strip())
    return list(enumerate(chunks))


def format_note(sents: List[Sentence]) -> str:
    return "\n".join(f"{i}. {t}" for i, t in sents)


# ------------------------------------------------------------------------------------
# Split
# ------------------------------------------------------------------------------------
def make_split(df: pd.DataFrame, seed: int = C.SPLIT_SEED) -> Dict[str, List[str]]:
    """Sort unique patient IDs, shuffle with numpy default_rng(seed), take 37/8/8."""
    pids = np.array(sorted(df["patient_id"].unique().tolist()))
    expected = C.N_TRAIN + C.N_VAL + C.N_TEST
    if len(pids) != expected:
        raise ValueError(f"Expected {expected} unique patients, found {len(pids)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    pids = pids.tolist()
    return {"train": pids[:C.N_TRAIN],
            "val": pids[C.N_TRAIN:C.N_TRAIN + C.N_VAL],
            "test": pids[C.N_TRAIN + C.N_VAL:]}


def assign_split(df: pd.DataFrame, split: Dict[str, List[str]]) -> pd.DataFrame:
    m = {p: s for s, ps in split.items() for p in ps}
    df = df.copy()
    df["split"] = df["patient_id"].map(m)
    return df


def leakage_checks(df: pd.DataFrame, split: Dict[str, List[str]]) -> Dict:
    """Assertions that the split is patient-disjoint and complete. Raises on failure."""
    s = {k: set(v) for k, v in split.items()}
    checks = {
        "train_val_overlap": sorted(s["train"] & s["val"]),
        "train_test_overlap": sorted(s["train"] & s["test"]),
        "val_test_overlap": sorted(s["val"] & s["test"]),
        "unassigned_rows": int(df["split"].isna().sum()),
    }
    # each patient's annotations land in exactly one split
    per_patient = df.groupby("patient_id")["split"].nunique()
    checks["patients_in_multiple_splits"] = per_patient[per_patient > 1].index.tolist()
    # identical note text under different patient IDs across splits would also be leakage
    note_split = df.groupby("note")["split"].nunique()
    checks["identical_notes_across_splits"] = int((note_split > 1).sum())
    # identical (note, criterion) pairs across splits
    pair_split = df.groupby(["note", "criterion_text"])["split"].nunique()
    checks["identical_note_criterion_pairs_across_splits"] = int((pair_split > 1).sum())
    ok = (not checks["train_val_overlap"] and not checks["train_test_overlap"]
          and not checks["val_test_overlap"] and checks["unassigned_rows"] == 0
          and not checks["patients_in_multiple_splits"]
          and checks["identical_notes_across_splits"] == 0)
    checks["passed"] = bool(ok)
    if not ok:
        raise AssertionError(f"Leakage check failed: {checks}")
    return checks


def build_manifest(df: pd.DataFrame, split: Dict[str, List[str]]) -> Dict:
    man = {"dataset": C.DATASET_ID, "seed": C.SPLIT_SEED,
           "procedure": "sorted unique patient_id -> numpy.random.default_rng(42).shuffle -> "
                        "first 37 train, next 8 val, last 8 test",
           "splits": {}}
    for name, pids in split.items():
        sub = df[df["split"] == name]
        man["splits"][name] = {
            "patient_ids": pids,
            "n_patients": len(pids),
            "n_annotations": int(len(sub)),
            "n_trials": int(sub["trial_id"].nunique()),
            "annotations_per_patient": {p: int((sub["patient_id"] == p).sum()) for p in pids},
            "by_criterion_type": {k: int(v) for k, v in Counter(sub["criterion_type"]).items()},
            "label_counts": {k: int(v) for k, v in Counter(sub["expert_eligibility"]).items()},
            "label_counts_by_type": {
                t: {k: int(v) for k, v in Counter(sub[sub.criterion_type == t]["expert_eligibility"]).items()}
                for t in ["inclusion", "exclusion"]},
            "empty_gold_evidence": int((sub["expert_sentences"].apply(len) == 0).sum()),
        }
    return man


def save_manifest(man: Dict, df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split_manifest.json").write_text(json.dumps(man, indent=2))
    rows = []
    for s, info in man["splits"].items():
        for p in info["patient_ids"]:
            rows.append({"split": s, "patient_id": p,
                         "n_annotations": info["annotations_per_patient"][p]})
    pd.DataFrame(rows).to_csv(out_dir / "split_manifest.csv", index=False)
    df[["annotation_id", "patient_id", "trial_id", "split"]].to_csv(
        out_dir / "annotation_split_assignment.csv", index=False)


def to_records(df: pd.DataFrame) -> List[Dict]:
    """Model-facing records. Only allowed fields go in."""
    recs = []
    for r in df.itertuples(index=False):
        recs.append({
            "annotation_id": int(r.annotation_id),
            "patient_id": r.patient_id,
            "trial_id": r.trial_id,
            "split": r.split,
            "sentences": parse_note(r.note),
            "criterion_type": r.criterion_type,
            "criterion_text": r.criterion_text,
            "gold_label": r.expert_eligibility,
            "gold_evidence": list(r.expert_sentences),
        })
    return recs


def write_jsonl(recs: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path) -> List[Dict]:
    with open(path) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    for r in recs:
        if "sentences" in r:
            r["sentences"] = [tuple(s) for s in r["sentences"]]
    return recs
