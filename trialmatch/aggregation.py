"""Torch-free helpers: criteria parsing and the explicit patient-trial aggregation rule."""
import re
from typing import Dict, List, Tuple


def parse_criteria(text: str, default_type: str = "inclusion") -> List[Tuple[str, str]]:
    """Accepts lines like 'INC: ...' / 'EXC: ...', or blocks under 'Inclusion Criteria:' and
    'Exclusion Criteria:' headings (ClinicalTrials.gov style). Bullets/numbers are stripped."""
    out, cur = [], default_type
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower().rstrip(":")
        if re.match(r"^(key )?inclusion criteria", low):
            cur = "inclusion"; continue
        if re.match(r"^(key )?exclusion criteria", low):
            cur = "exclusion"; continue
        m = re.match(r"^(inc|incl|inclusion|exc|excl|exclusion)\s*[:\-]\s*(.+)$", line, flags=re.I)
        if m:
            t = "inclusion" if m.group(1).lower().startswith("inc") else "exclusion"
            out.append((t, m.group(2).strip()))
            continue
        line = re.sub(r"^([\-\*••]|\d+[\.\)])\s*", "", line).strip()
        if len(line) >= 3:
            out.append((cur, line))
    return out


def aggregate(results: List[Dict]) -> Dict:
    """Explicit rule (NOT a ground-truth label; the dataset is criterion-level only):
      - 'Likely ineligible'  if any inclusion is 'not included' or any exclusion is 'excluded'
      - 'Potentially eligible' if every inclusion is 'included'/'not applicable' and every
        exclusion is 'not excluded'/'not applicable'
      - otherwise 'Undetermined — needs review' (some criteria lack information)"""
    inc = [r for r in results if r["criterion_type"] == "inclusion"]
    exc = [r for r in results if r["criterion_type"] == "exclusion"]
    failing = [r for r in inc if r["label"] == "not included"] + [r for r in exc if r["label"] == "excluded"]
    missing = [r for r in results if r["label"] == "not enough information"]
    unsupported = [r for r in results if r.get("unsupported")]
    if failing:
        verdict = "Likely ineligible"
    elif all(r["label"] in ("included", "not applicable") for r in inc) and \
            all(r["label"] in ("not excluded", "not applicable") for r in exc) and results:
        verdict = "Potentially eligible"
    else:
        verdict = "Undetermined — needs review"
    return {"verdict": verdict, "n_criteria": len(results), "n_failing": len(failing),
            "n_missing_information": len(missing), "n_unsupported": len(unsupported),
            "failing_criteria": [r["criterion_text"] for r in failing],
            "missing_information_criteria": [r["criterion_text"] for r in missing],
            "rule": "ineligible if any inclusion='not included' or exclusion='excluded'; eligible if all "
                    "inclusions met/NA and all exclusions not met/NA; otherwise needs review."}
