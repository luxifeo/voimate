"""Compare agent runs on the same train subset: score, paired difference to a baseline, error types and the targeted
counters of the experiments (extra arguments, wrong weekday/date claims in reasoning, LLM calls, tokens, latency).

    uv run python scripts/compare_runs.py h0_base_gold h1_oracle_gold h5_user_gold --base h0_base_gold
    uv run python scripts/compare_runs.py h0_base_gold llm_qwen35_4b_gold llm_qwen35_4b_gold_log --base h0_base_gold

Paired difference: per-sample score of the run minus the baseline on the samples both runs have, with the number of
samples that went up / down. Two runs of the same config differ by about 1.1 points of standard error on 300 samples
(vLLM is not batch-invariant), so a single run's gain under ~2-3 points is not evidence on its own.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]

from error_analysis import arg_diff, classify, unmatched  # noqa: E402
from reasoning_date_audit import claims, date_arg_diffs  # noqa: E402
from lifemate.data import OUTPUT_ROOT, load_catalog, load_split  # noqa: E402
from lifemate.scorer import fuzzy_keys  # noqa: E402

ERROR_TYPES = ["oos_called", "no_call", "over_ask", "under_ask", "ask_wrong", "write_missing_tool", "write_wrong_args",
               "write_extra", "read_only", "state_only", "other"]


def jsonl(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {d["id"]: d for d in map(json.loads, f)}


def run_stats(name: str, rows: dict, kinds: dict, fz) -> dict:
    d = OUTPUT_ROOT / "runs" / "train" / name
    scores, sub = jsonl(d / "scores.jsonl"), jsonl(d / "submission.jsonl")
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    loss, extra_keys, n_pred, date_vals = Counter(), Counter(), 0, 0
    for sid, s in scores.items():
        gold, pred = rows[sid]["label"]["tool_calls"], sub[sid]["tool_calls"]
        n_pred += len(pred)
        if s["score"] < 1:
            loss[classify(pred, gold, s, kinds, fz)[0]] += 1 - s["score"]
        extra, missing = unmatched(pred, gold, fz)
        for g in missing:
            same = [p for p in extra if p["name"] == g["name"]]
            if same:
                for kind, k, _, _ in min((arg_diff(p, g, fz) for p in same), key=len):
                    extra_keys[kind] += 1
                date_vals += len(date_arg_diffs(same[0], g))
    calls, lat, ptok, ctok, wd_bad, wd_all, calc = defaultdict(int), defaultdict(float), [], [], 0, 0, 0
    log = d / "llm_calls.jsonl"
    if log.exists():
        with open(log, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                sid = r["sample_id"]
                calls[sid] += 1
                lat[sid] += r["latency_s"]
                resp = r.get("response") or {}
                usage = resp.get("usage") or {}
                ptok.append(usage.get("prompt_tokens", 0))
                ctok.append(usage.get("completion_tokens", 0))
                msg = (resp.get("choices") or [{}])[0].get("message") or {}
                calc += sum(1 for c in msg.get("tool_calls") or [] if c["function"]["name"] == "scratch__calc")
                text = msg.get("reasoning_content") or msg.get("reasoning") or ""
                today = datetime.fromisoformat(rows[sid]["context"]["current_time"]).date()
                for c in claims(text, today):
                    if c["kind"] == "weekday":
                        wd_all += 1
                        wd_bad += not c["ok"]
    n = len(scores)
    return {"name": name, "n": n, "scores": {k: v["score"] for k, v in scores.items()},
            "score": 100 * sum(s["score"] for s in scores.values()) / n,
            "cat": {c: metrics[c]["score"] for c in ("action", "ask_missing_info", "out_of_scope") if c in metrics},
            "loss": {k: 100 * v / n for k, v in loss.items()}, "extra_key": extra_keys["extra_key"],
            "missing_key": extra_keys["missing_key"], "value": extra_keys["value"] + extra_keys["fuzzy_low"],
            "pred_calls": n_pred / n, "llm_calls": sum(calls.values()) / max(len(calls), 1),
            "latency": st.mean(lat.values()) if lat else None, "ptok": st.mean(ptok) if ptok else None,
            "ctok": st.mean(ctok) if ctok else None, "wd": (wd_bad, wd_all), "calc": calc, "date_vals": date_vals,
            "sec_per_sample": metrics.get("config", {}).get("sec_per_sample")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--base", default=None, help="baseline run for the paired difference (default: first run)")
    args = ap.parse_args()
    rows = {r["id"]: r for r in load_split("train")}
    catalog = load_catalog("train")
    kinds, fz = {n: s.get("kind") for n, s in catalog.items()}, fuzzy_keys(catalog)
    stats = [run_stats(r, rows, kinds, fz) for r in args.runs]
    base = next(s for s in stats if s["name"] == (args.base or args.runs[0]))

    print("| run | n | score | Δ vs base (paired) | ↑/↓ | action | ask | oos | calls/sample | extra keys | missing keys "
          "| wrong values | wrong date/time values | weekday claims wrong | LLM calls | prompt tok | completion tok | wall s/sample | calc |")
    print("|---" * 19 + "|")
    for s in stats:
        common = s["scores"].keys() & base["scores"].keys()
        diffs = [s["scores"][i] - base["scores"][i] for i in common]
        up, down = sum(x > 1e-9 for x in diffs), sum(x < -1e-9 for x in diffs)
        se = 100 * st.pstdev(diffs) / len(diffs) ** 0.5 if diffs else 0
        c = s["cat"]
        print(f"| {s['name']} | {s['n']} | {s['score']:.2f} | {100 * st.mean(diffs):+.2f} ± {se:.2f} | {up}/{down} "
              f"| {c.get('action', 0):.1f} | {c.get('ask_missing_info', 0):.1f} | {c.get('out_of_scope', 0):.1f} "
              f"| {s['pred_calls']:.2f} | {s['extra_key']} | {s['missing_key']} | {s['value']} | {s['date_vals']} "
              f"| {s['wd'][0]}/{s['wd'][1]} | {s['llm_calls']:.2f} | {s['ptok'] or 0:.0f} | {s['ctok'] or 0:.0f} "
              f"| {s['sec_per_sample'] or 0:.2f} | {s['calc']} |")
    print("\nĐiểm mất theo loại lỗi (/100):")
    print("| loại | " + " | ".join(s["name"] for s in stats) + " |")
    print("|---" * (len(stats) + 1) + "|")
    for t in ERROR_TYPES:
        if any(t in s["loss"] for s in stats):
            print(f"| {t} | " + " | ".join(f"{s['loss'].get(t, 0):.2f}" for s in stats) + " |")


if __name__ == "__main__":
    main()
