"""CPU-only tests (no model download). Run: python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trialmatch import config as C
from trialmatch.data import (assign_split, leakage_checks, make_split, parse_id_list, parse_note,
                             split_sentences)
from trialmatch.evaluate import _evidence_metrics, categorize, compute_metrics
from trialmatch.prompts import build_explanation, parse_output


def test_parse_note_numbered_lines():
    note = "0. A 58-year-old woman.\n1. She has 2.5 mg/dL creatinine.\n2. No history of stroke."
    s = parse_note(note)
    assert s == [(0, "A 58-year-old woman."), (1, "She has 2.5 mg/dL creatinine."), (2, "No history of stroke.")]


def test_parse_note_inline_and_continuation():
    s = parse_note("0. First sentence with 3. decimals? no 1. Second one. 2. Third.")
    assert [i for i, _ in s] == [0, 1, 2]
    s2 = parse_note("0. First line\ncontinues here.\n1. Second.")
    assert s2 == [(0, "First line continues here."), (1, "Second.")]


def test_split_sentences_free_text():
    s = split_sentences("Dr. Smith saw the patient. BP was 120/80. She is 45 years old.")
    assert [t for _, t in s] == ["Dr. Smith saw the patient.", "BP was 120/80.", "She is 45 years old."]


def test_parse_id_list():
    assert parse_id_list("[0, 2]") == [0, 2]
    assert parse_id_list("[]") == []
    assert parse_id_list([3, 1, 3]) == [1, 3]
    assert parse_id_list(None) == []


def test_split_is_deterministic_and_disjoint():
    pids = [f"sigir-{20000 + i}" for i in range(53)]
    rows = [{"patient_id": p, "note": f"note {p}", "criterion_text": f"c{j}", "annotation_id": k}
            for k, (p, j) in enumerate([(p, j) for p in pids for j in range(3)])]
    df = pd.DataFrame(rows)
    s1, s2 = make_split(df), make_split(df)
    assert s1 == s2
    assert [len(s1[k]) for k in ("train", "val", "test")] == [37, 8, 8]
    # exactly the prescribed procedure
    arr = np.array(sorted(pids)); np.random.default_rng(42).shuffle(arr)
    assert s1["train"] == arr[:37].tolist() and s1["test"] == arr[45:].tolist()
    df = assign_split(df, s1)
    assert leakage_checks(df, s1)["passed"]


def test_parse_output_variants():
    r = parse_output('{"evidence": [0, 3], "label": "not excluded"}', "exclusion", [0, 1, 2, 3])
    assert r["parse_ok"] and r["label"] == "not excluded" and r["evidence"] == [0, 3]
    r = parse_output('Answer: {"evidence": [9], "label": "included"}', "inclusion", [0, 1])
    assert r["label"] == "included" and r["evidence"] == [] and r["evidence_out_of_range"] == [9]
    r = parse_output("The patient is not included.", "inclusion")
    assert r["label"] == "not included" and not r["json_ok"]
    r = parse_output('{"evidence": [], "label": "excluded"}', "inclusion")  # wrong label set
    assert not r["parse_ok"] and r["label"] == "not enough information"
    r = parse_output("gibberish", "exclusion")
    assert not r["parse_ok"]


def test_evidence_metrics_empty_handling():
    rows = [{"gold_evidence": [1, 2], "pred_evidence": [2, 3]},
            {"gold_evidence": [], "pred_evidence": []},
            {"gold_evidence": [], "pred_evidence": [4]},
            {"gold_evidence": [5], "pred_evidence": []}]
    m = _evidence_metrics(rows)
    assert m["micro"]["tp"] == 1 and m["micro"]["fp"] == 2 and m["micro"]["fn"] == 2
    assert m["macro_nonempty_gold"]["n_records"] == 2
    assert abs(m["macro_nonempty_gold"]["f1"] - 0.25) < 1e-9
    assert m["empty_gold"]["n_records"] == 2 and m["empty_gold"]["correct_empty_prediction"] == 1


def test_compute_metrics_and_categories():
    preds = []
    for gl, pl, t in [("included", "included", "inclusion"), ("not included", "not enough information", "inclusion"),
                      ("excluded", "not excluded", "exclusion"), ("not enough information", "excluded", "exclusion")]:
        preds.append({"gold_label": gl, "pred_label": pl, "criterion_type": t, "gold_evidence": [0],
                      "pred_evidence": [0] if pl != "excluded" else [], "parse_ok": True, "json_ok": True,
                      "n_new_tokens": 10, "truncated": False, "gold_evidence_lost": []})
    m = compute_metrics(preds)
    assert m["overall"]["accuracy"] == 0.25
    assert m["inclusion"]["n"] == 2 and m["exclusion"]["n"] == 2
    assert "missed_information" in categorize(preds[1])
    assert "polarity_error" in categorize(preds[2])
    assert "unsupported_decision" in categorize(preds[3])


def test_explanation_never_claims_without_evidence():
    e = build_explanation("excluded", "exclusion", "x", [], {0: "a"})
    assert "UNSUPPORTED" in e
    e = build_explanation("not enough information", "inclusion", "x", [], {})
    assert "missing information" in e


def test_criteria_parser():
    from trialmatch.aggregation import aggregate, parse_criteria
    txt = "Inclusion Criteria:\n- Age > 18\n* Diabetes\nExclusion Criteria:\n1. Pregnancy\nINC: BMI < 30"
    c = parse_criteria(txt)
    assert c == [("inclusion", "Age > 18"), ("inclusion", "Diabetes"), ("exclusion", "Pregnancy"), ("inclusion", "BMI < 30")]
    res = [{"criterion_type": "inclusion", "criterion_text": "a", "label": "included"},
           {"criterion_type": "exclusion", "criterion_text": "b", "label": "not enough information"}]
    assert aggregate(res)["verdict"].startswith("Undetermined")
    res[1]["label"] = "excluded"
    assert aggregate(res)["verdict"] == "Likely ineligible"
