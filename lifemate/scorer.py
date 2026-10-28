"""Local re-implementation of the LifeMate metric (see evaluation.md).

Assumptions where the official description is not explicit:
- non-fuzzy arguments must be exactly equal (no case/whitespace normalisation);
- a call matches only if submitted and gold calls have the same argument keys;
- fuzzy token-F1 uses whitespace tokens after lower-casing and Unicode NFC.
"""
from __future__ import annotations

import unicodedata
from collections import Counter


def fuzzy_keys(catalog: dict[str, dict]) -> dict[str, set[str]]:
    return {
        name: {k for k, p in spec.get("parameters", {}).get("properties", {}).items() if p.get("x-match") == "fuzzy"}
        for name, spec in catalog.items()
    }


def token_f1(a, b) -> float:
    ta = Counter(unicodedata.normalize("NFC", str(a).lower()).split())
    tb = Counter(unicodedata.normalize("NFC", str(b).lower()).split())
    if not ta and not tb:
        return 1.0
    common = sum((ta & tb).values())
    if common == 0:
        return 0.0
    p, r = common / sum(ta.values()), common / sum(tb.values())
    return 2 * p * r / (p + r)


def call_matches(pred: dict, gold: dict, fuzzy: dict[str, set[str]]) -> bool:
    if pred.get("name") != gold.get("name"):
        return False
    pa, ga = pred.get("arguments") or {}, gold.get("arguments") or {}
    if set(pa) != set(ga):
        return False
    fz = fuzzy.get(gold["name"], set())
    for k, gv in ga.items():
        if k in fz:
            if token_f1(pa[k], gv) <= 0.6:
                return False
        elif pa[k] != gv:
            return False
    return True


def score_sample(pred_calls: list[dict], pred_hash: str, gold_calls: list[dict], gold_hash: str,
                 fuzzy: dict[str, set[str]]) -> dict:
    if not gold_calls:
        s = 1.0 if not pred_calls else 0.0
        return {"score": s, "matched": 0, "precision": s, "recall": s, "state_ok": pred_hash == gold_hash}
    used = [False] * len(gold_calls)
    matched = 0
    for pc in pred_calls:
        for j, gc in enumerate(gold_calls):
            if not used[j] and call_matches(pc, gc, fuzzy):
                used[j] = True
                matched += 1
                break
    unmatched = len(pred_calls) - matched
    precision = matched / (matched + unmatched) if pred_calls else 0.0
    recall = matched / len(gold_calls)
    state_ok = pred_hash == gold_hash
    score = 0.0 if (not state_ok or matched == 0) else 2 * precision * recall / (precision + recall)
    return {"score": score, "matched": matched, "precision": precision, "recall": recall, "state_ok": state_ok}


def category(gold_calls: list[dict]) -> str:
    if not gold_calls:
        return "out_of_scope"
    if any(c["name"] == "sys.ask_missing_info" for c in gold_calls):
        return "ask_missing_info"
    return "action"


def summarize(results: list[dict]) -> dict:
    """results: [{"category": ..., "score": ..., "state_ok": ..., "precision": ..., "recall": ...}]"""
    def agg(rs):
        n = len(rs)
        return {
            "n": n,
            "score": round(100 * sum(r["score"] for r in rs) / max(n, 1), 2),
            "state_acc": round(100 * sum(r["state_ok"] for r in rs) / max(n, 1), 2),
            "precision": round(100 * sum(r["precision"] for r in rs) / max(n, 1), 2),
            "recall": round(100 * sum(r["recall"] for r in rs) / max(n, 1), 2),
        }
    out = {"all": agg(results)}
    for cat in sorted({r["category"] for r in results}):
        out[cat] = agg([r for r in results if r["category"] == cat])
    return out
