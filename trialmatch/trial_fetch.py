"""Optional: fetch public eligibility criteria from ClinicalTrials.gov (API v2) for the app.
Public metadata only supports the application; it is never used for training/evaluation."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request


def fetch_criteria(nct_id: str, timeout: int = 15):
    nct_id = (nct_id or "").strip().upper()
    if not re.fullmatch(r"NCT\d{8}", nct_id):
        raise ValueError("Trial ID must look like NCT01234567.")
    url = (f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
           "?fields=protocolSection.identificationModule.briefTitle,protocolSection.eligibilityModule")
    req = urllib.request.Request(url, headers={"User-Agent": "trialmatch-edu-prototype/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise ValueError(f"ClinicalTrials.gov returned HTTP {e.code} for {nct_id}.") from e
    except Exception as e:
        raise ValueError(f"Could not reach ClinicalTrials.gov ({e}). Paste the criteria manually.") from e
    ps = data.get("protocolSection", {})
    title = ps.get("identificationModule", {}).get("briefTitle", "")
    crit = ps.get("eligibilityModule", {}).get("eligibilityCriteria", "")
    if not crit:
        raise ValueError(f"No eligibility criteria text found for {nct_id}.")
    return title, crit
