"""Prompt construction, truncation policy, target serialisation and robust output parsing."""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from . import config as C
from .data import Sentence, format_note

SYSTEM_PROMPT = (
    "You are a careful clinical-trial eligibility screener. You read a patient note whose "
    "sentences are numbered and judge ONE eligibility criterion. Use only information stated "
    "in the note. If the note does not contain enough information to decide, say so: missing "
    "information is never proof of eligibility or ineligibility. Answer with a single JSON "
    "object and nothing else."
)

LABEL_GUIDE = {
    "inclusion": (
        '"included" = the note shows the patient meets this inclusion criterion; '
        '"not included" = the note shows the patient does not meet it; '
        '"not enough information" = the note does not say enough to decide; '
        '"not applicable" = the criterion does not apply to this patient.'),
    "exclusion": (
        '"excluded" = the note shows the patient meets this exclusion criterion; '
        '"not excluded" = the note shows the patient does not meet it; '
        '"not enough information" = the note does not say enough to decide; '
        '"not applicable" = the criterion does not apply to this patient.'),
}


def user_prompt(sents: List[Sentence], ctype: str, ctext: str) -> str:
    return (
        f"Patient note:\n{format_note(sents)}\n\n"
        f"Criterion type: {ctype}\n"
        f"Criterion: {ctext}\n\n"
        f"Allowed labels: {LABEL_GUIDE[ctype]}\n"
        'Respond with JSON: {"evidence": [IDs of the note sentences that support the judgment, '
        'or [] if none], "label": "<one allowed label>"}'
    )


def target_text(label: str, evidence: List[int]) -> str:
    return json.dumps({"evidence": sorted(int(i) for i in evidence), "label": label})


def chat_prompt_ids(tokenizer, sents, ctype, ctext) -> List[int]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(sents, ctype, ctext)}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):   # newer HF versions may return BatchEncoding
        ids = ids["input_ids"]
    return list(ids)


def build_prompt(tokenizer, sents: List[Sentence], ctype: str, ctext: str,
                 max_prompt_tokens: int = C.MAX_PROMPT_TOKENS,
                 gold_evidence: Optional[List[int]] = None) -> Tuple[List[int], Dict]:
    """Tokenise the chat prompt. If it exceeds the budget, drop whole trailing sentences
    (original IDs are preserved for those kept) and report exactly what was removed."""
    info = {"n_sentences": len(sents), "dropped_sentence_ids": [], "truncated": False,
            "gold_evidence_lost": []}
    kept = list(sents)
    ids = chat_prompt_ids(tokenizer, kept, ctype, ctext)
    info["prompt_tokens_full"] = len(ids)
    while len(ids) > max_prompt_tokens and len(kept) > 1:
        info["dropped_sentence_ids"].append(kept.pop()[0])
        ids = chat_prompt_ids(tokenizer, kept, ctype, ctext)
    if len(ids) > max_prompt_tokens:   # a single huge sentence / criterion: hard cut (rare)
        ids = ids[:max_prompt_tokens]
        info["hard_cut"] = True
    info["truncated"] = bool(info["dropped_sentence_ids"]) or info.get("hard_cut", False)
    info["prompt_tokens"] = len(ids)
    if gold_evidence:
        info["gold_evidence_lost"] = sorted(set(gold_evidence) & set(info["dropped_sentence_ids"]))
    info["kept_sentence_ids"] = [i for i, _ in kept]
    return ids, info


# ------------------------------------------------------------------------------------
# Output parsing
# ------------------------------------------------------------------------------------
_LABELS_LONGEST_FIRST = sorted(C.ALL_LABELS, key=len, reverse=True)


def parse_output(text: str, ctype: str, valid_ids: Optional[List[int]] = None) -> Dict:
    """Parse model text into {label, evidence}. Never silently invents a decision:
    if no allowed label can be recovered, label = 'not enough information' and
    parse_ok=False (counted and reported as invalid output)."""
    res = {"raw": text, "json_ok": False, "label_ok": False, "parse_ok": False,
           "label": None, "evidence": []}
    obj = None
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            res["json_ok"] = isinstance(obj, dict)
        except Exception:
            obj = None
    label, ev = None, []
    if isinstance(obj, dict):
        lab = obj.get("label", obj.get("eligibility"))
        if isinstance(lab, str):
            label = lab.strip().lower()
        e = obj.get("evidence", obj.get("sentence_ids", []))
        if isinstance(e, (list, tuple)):
            for x in e:
                try:
                    ev.append(int(x))
                except Exception:
                    pass
        elif isinstance(e, (int, str)):
            ev = [int(t) for t in re.findall(r"\d+", str(e))]
    if label is None:  # regex fallback over raw text
        low = text.lower()
        for cand in _LABELS_LONGEST_FIRST:
            if re.search(rf'"?{re.escape(cand)}"?', low):
                label = cand
                break
        mm = re.search(r'"evidence"\s*:\s*\[([^\]]*)\]', text)
        if mm:
            ev = [int(t) for t in re.findall(r"\d+", mm.group(1))]
    allowed = C.labels_for(ctype)
    if label in allowed:
        res["label_ok"] = True
    else:
        res["label_wrong_type"] = label
        label = "not enough information"
    if valid_ids is not None:
        vs = set(valid_ids)
        res["evidence_out_of_range"] = sorted(set(ev) - vs)
        ev = [i for i in ev if i in vs]
    res["label"] = label
    res["evidence"] = sorted(set(ev))
    res["parse_ok"] = res["label_ok"]
    return res


def build_explanation(label: str, ctype: str, ctext: str, evidence: List[int],
                      sent_map: Dict[int, str]) -> str:
    """Grounded explanation built only from the cited sentences (no free text the
    model could hallucinate). Expert explanations are not in the dataset and GPT-4
    explanations are forbidden, so the explanation is deterministic and traceable."""
    cited = "; ".join(f'[{i}] "{sent_map[i]}"' for i in evidence if i in sent_map)
    if label == "not enough information":
        s = ("The note does not establish whether this criterion is met. This is reported as "
             "missing information, not as evidence for or against eligibility.")
        if cited:
            s += f" Related sentences considered: {cited}."
        return s
    if label == "not applicable":
        s = "The criterion was judged not to apply to this patient."
        if cited:
            s += f" Basis: {cited}."
        return s
    meaning = {"included": "meets this inclusion criterion",
               "not included": "does not meet this inclusion criterion",
               "excluded": "meets this exclusion criterion (would be excluded)",
               "not excluded": "does not meet this exclusion criterion"}[label]
    if cited:
        return f"The patient {meaning}, based on {cited}."
    return (f"The model judged that the patient {meaning}, but cited no supporting sentence. "
            "UNSUPPORTED decision: treat as low confidence and review manually.")
