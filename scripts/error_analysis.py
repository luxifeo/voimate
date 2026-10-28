"""Error analysis of an agent run on train: where the points are lost, by error type, tool and argument.

    uv run python scripts/error_analysis.py llm_qwen35_4b_gold
    uv run python scripts/error_analysis.py llm_qwen35_4b_gold --examples 3

Each sample with score < 1 gets one primary error type (first rule that applies, see `classify`).
"Points lost" = sum of (1 - score) over the samples of a type, as a share of the run's total loss.
Writes outputs/runs/train/<run>/error_analysis.jsonl (one record per imperfect sample).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.data import OUTPUT_ROOT, load_catalog, load_split  # noqa: E402
from lifemate.scorer import call_matches, category, fuzzy_keys, token_f1  # noqa: E402

ASK = "sys.ask_missing_info"
LABELS = {
    "oos_called": "OOS: gold không gọi gì, model lại gọi tool",
    "no_call": "Không gọi tool nào (gold có call)",
    "over_ask": "Hỏi thừa: gold hành động, model gọi ask_missing_info",
    "under_ask": "Thiếu hỏi: gold ask_missing_info, model hành động/không hỏi",
    "ask_wrong": "Ask sai nội dung (tool_name/params)",
    "write_missing_tool": "Thiếu write call (không gọi đúng tool)",
    "write_wrong_args": "Write call đúng tool nhưng sai đối số",
    "write_extra": "Write call thừa",
    "read_only": "Chỉ lệch read call (writes đúng, mất điểm precision/recall)",
    "state_only": "Calls khớp nhưng state lệch",
    "other": "Khác",
}


def unmatched(pred: list[dict], gold: list[dict], fz) -> tuple[list[dict], list[dict]]:
    """Greedy matching as in score_sample; returns (unmatched pred calls, unmatched gold calls)."""
    used, extra = [False] * len(gold), []
    for pc in pred:
        for j, gc in enumerate(gold):
            if not used[j] and call_matches(pc, gc, fz):
                used[j] = True
                break
        else:
            extra.append(pc)
    return extra, [g for g, u in zip(gold, used) if not u]


def arg_diff(pred: dict, gold: dict, fz) -> list[tuple[str, str, object, object]]:
    """[(kind, key, gold value, pred value)] with kind in missing_key / extra_key / value / fuzzy_low."""
    pa, ga, out = pred.get("arguments") or {}, gold.get("arguments") or {}, []
    for k in ga.keys() - pa.keys():
        out.append(("missing_key", k, ga[k], None))
    for k in pa.keys() - ga.keys():
        out.append(("extra_key", k, None, pa[k]))
    for k in ga.keys() & pa.keys():
        if k in fz.get(gold["name"], set()):
            if token_f1(pa[k], ga[k]) <= 0.6:
                out.append(("fuzzy_low", k, ga[k], pa[k]))
        elif pa[k] != ga[k]:
            out.append(("value", k, ga[k], pa[k]))
    return out


def classify(pred, gold, s, kinds, fz) -> tuple[str, dict]:
    extra, missing = unmatched(pred, gold, fz)
    info = {"extra": extra, "missing": missing}
    pred_ask, gold_ask = any(c["name"] == ASK for c in pred), any(c["name"] == ASK for c in gold)
    if not gold:
        return "oos_called", info
    if not pred:
        return "no_call", info
    if gold_ask and not pred_ask:
        return "under_ask", info
    if pred_ask and not gold_ask:
        return "over_ask", info
    if gold_ask and any(c["name"] == ASK for c in missing):
        return "ask_wrong", info
    miss_w = [c for c in missing if kinds.get(c["name"]) == "write"]
    if miss_w:
        diffs = []
        for g in miss_w:
            same = [p for p in extra if p["name"] == g["name"]]
            if same:
                diffs.append((g["name"], min((arg_diff(p, g, fz) for p in same), key=len)))
        if diffs:
            info["diffs"] = diffs
            return "write_wrong_args", info
        return "write_missing_tool", info
    if any(kinds.get(c["name"]) == "write" for c in extra):
        return "write_extra", info
    if extra or missing:
        return "read_only", info
    if not s["state_ok"]:
        return "state_only", info
    return "other", info


def short(calls: list[dict]) -> str:
    return "; ".join(f"{c['name']}({json.dumps(c.get('arguments') or {}, ensure_ascii=False)})" for c in calls) or "∅"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--examples", type=int, default=2)
    args = ap.parse_args()

    run_dir = OUTPUT_ROOT / "runs" / "train" / args.run
    rows = {r["id"]: r for r in load_split("train")}
    catalog = load_catalog("train")
    kinds = {n: s.get("kind") for n, s in catalog.items()}
    fz = fuzzy_keys(catalog)
    sub = {d["id"]: d for d in map(json.loads, open(run_dir / "submission.jsonl", encoding="utf-8"))}
    scores = {d["id"]: d for d in map(json.loads, open(run_dir / "scores.jsonl", encoding="utf-8"))}
    traces = {d["id"]: d for d in map(json.loads, open(run_dir / "traces.jsonl", encoding="utf-8"))}

    n = len(scores)
    total_loss = sum(1 - s["score"] for s in scores.values())
    by_type, records = defaultdict(list), []
    key_errors, extra_calls, missing_calls = Counter(), Counter(), Counter()
    app_score = defaultdict(list)
    for sid, s in scores.items():
        gold, pred = rows[sid]["label"]["tool_calls"], sub[sid]["tool_calls"]
        for app in {c["name"].split(".")[0] for c in gold if c["name"] != ASK} or {"(none)"}:
            app_score[app].append(s["score"])
        if s["score"] >= 1:
            continue
        etype, info = classify(pred, gold, s, kinds, fz)
        by_type[etype].append(sid)
        for c in info["extra"]:
            extra_calls[(c["name"], kinds.get(c["name"]))] += 1
        for c in info["missing"]:
            missing_calls[(c["name"], kinds.get(c["name"]))] += 1
        for g, p in [(g, p) for g in info["missing"] for p in info["extra"] if p["name"] == g["name"]]:
            for kind, k, _, _ in arg_diff(p, g, fz):
                key_errors[(g["name"], k, kind)] += 1
        records.append({"id": sid, "category": s["category"], "type": etype, "score": s["score"],
                        "state_ok": s["state_ok"], "transcript": rows[sid]["audio_text"], "gold": gold,
                        "pred": pred, "diffs": info.get("diffs"), "stop": traces[sid].get("stop")})
    with open(run_dir / "error_analysis.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    perfect = sum(s["score"] >= 1 for s in scores.values())
    print(f"run {args.run}: {n} samples, score {100 * sum(s['score'] for s in scores.values()) / n:.2f}, "
          f"perfect {perfect}, imperfect {n - perfect}, points lost {100 * total_loss / n:.2f}\n")
    print("| Loại lỗi | Mẫu | Điểm mất (/100) | % tổng mất | theo category gold |\n|---|---|---|---|---|")
    for t, ids in sorted(by_type.items(), key=lambda kv: -sum(1 - scores[i]["score"] for i in kv[1])):
        loss = sum(1 - scores[i]["score"] for i in ids)
        cats = Counter(scores[i]["category"] for i in ids)
        print(f"| {LABELS[t]} | {len(ids)} | {100 * loss / n:.2f} | {loss / total_loss:.0%} | "
              f"{', '.join(f'{c} {k}' for c, k in cats.most_common())} |")

    print("\nĐối số sai hay gặp nhất (tool, key, kiểu lỗi):")
    for (tool, k, kind), c in key_errors.most_common(15):
        print(f"  {c:>3}  {tool}.{k}  {kind}")
    print("\nCall thừa hay gặp nhất:  " + ", ".join(f"{t} [{kd}] ({c})" for (t, kd), c in extra_calls.most_common(10)))
    print("Call thiếu hay gặp nhất: " + ", ".join(f"{t} [{kd}] ({c})" for (t, kd), c in missing_calls.most_common(10)))

    print("\nĐiểm theo app (mẫu có gold call vào app đó; app có >= 8 mẫu):")
    for app, sc in sorted(app_score.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        if len(sc) >= 8:
            print(f"  {app:<18} n={len(sc):<3} score {100 * sum(sc) / len(sc):5.1f}")
    print("\nStop reasons (mẫu lỗi): " + ", ".join(f"{k} {v}" for k, v in Counter(r["stop"] for r in records).items()))

    for t, ids in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        print(f"\n### {LABELS[t]}")
        for r in [r for r in records if r["type"] == t][: args.examples]:
            print(f"[{r['id']}] score {r['score']:.2f} state_ok {r['state_ok']}\n  user: {r['transcript']}\n"
                  f"  gold: {short(r['gold'])}\n  pred: {short(r['pred'])}")
            if r["diffs"]:
                print(f"  diff: {r['diffs']}")


if __name__ == "__main__":
    main()
