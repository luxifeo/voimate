"""How often does the agent fail because it does not use the conversation history?

    uv run python scripts/history_usage.py llm_qwen35_4b_gold

Two heuristics over the samples of a train run that have a history:
1. History-dependent gold calls: a gold write / ask call with an argument value that appears in the history
   (user turns, assistant calls, tool results) but neither in the final request nor in any tool result of the
   current turn (gold reads or the model's own reads). The value could only come from the history.
   For each such call: did the model make it correctly, use another value, ask instead, or skip it?
2. Redundant re-reads: the model repeats a read call (same name and arguments) already made in the history,
   and the gold does not repeat it.
Writes outputs/runs/train/<run>/history_usage.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.data import OUTPUT_ROOT, load_catalog, load_split  # noqa: E402
from lifemate.scorer import call_matches, fuzzy_keys  # noqa: E402

ASK = "sys.ask_missing_info"


def norm(x) -> str:
    return unicodedata.normalize("NFC", str(x).lower())


def scalars(args: dict) -> list[tuple[str, object]]:
    out = []
    for k, v in (args or {}).items():
        for x in (v if isinstance(v, list) else [v]):
            if isinstance(x, bool) or x is None:
                continue
            if isinstance(x, (int, float)) or (isinstance(x, str) and len(x) >= 3):
                out.append((k, x))
    return out


def present(v, text: str) -> bool:
    return norm(v) in text if isinstance(v, str) else json.dumps(v) in text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--hist-chars", type=int, default=1500, help="history tool-result truncation used by the agent")
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()

    run_dir = OUTPUT_ROOT / "runs" / "train" / args.run
    rows = {r["id"]: r for r in load_split("train")}
    cat = load_catalog("train")
    kinds = {n: s.get("kind") for n, s in cat.items()}
    fz = fuzzy_keys(cat)
    load = lambda f: {d["id"]: d for d in map(json.loads, open(run_dir / f, encoding="utf-8"))}  # noqa: E731
    sub, scores, traces = load("submission.jsonl"), load("scores.jsonl"), load("traces.jsonl")

    n_hist = 0
    outcome, records = Counter(), []
    reread_samples, reread_loss = [], 0.0
    for sid, s in scores.items():
        r = rows[sid]
        if not r["history"]:
            continue
        n_hist += 1
        hist_parts, hist_seen, hist_reads = [], [], set()
        for t in r["history"]:
            if t["role"] == "user":
                hist_parts.append(t.get("audio_text", ""))
                continue
            hist_parts.append(t.get("text", ""))
            for c, res in zip(t.get("tool_calls", []), t.get("tool_results", []) + [None] * 99):
                hist_parts.append(json.dumps(c.get("arguments") or {}, ensure_ascii=False))
                full = json.dumps(res, ensure_ascii=False)
                hist_parts.append(full)
                hist_seen.append(full[: args.hist_chars])
                if kinds.get(c["name"]) == "read":
                    hist_reads.add((c["name"], json.dumps(c.get("arguments") or {}, sort_keys=True)))
        H = norm(" ".join(hist_parts))
        H_seen = norm(" ".join(p for p in hist_parts if p not in hist_seen) + " " + " ".join(hist_seen))
        F = norm(r["audio_text"])
        gold, pred = r["label"]["tool_calls"], sub[sid]["tool_calls"]
        cur = norm(json.dumps(r["label"]["tool_results"], ensure_ascii=False) + " " +
                   " ".join(x for st in traces[sid].get("trace", []) for x in st.get("results", [])))

        # 1. history-dependent gold calls
        for g in gold:
            if kinds.get(g["name"]) == "read" or g["name"] == ASK:
                continue
            dep = [(k, v) for k, v in scalars(g["arguments"]) if present(v, H) and not present(v, F) and not present(v, cur)]
            if not dep:
                continue
            same = [p for p in pred if p["name"] == g["name"]]
            if any(call_matches(p, g, fz) for p in pred):
                res = "đúng"
            elif same:
                res = "gọi đúng tool nhưng sai đối số"
            elif any(p["name"] == ASK for p in pred):
                res = "hỏi lại thay vì dùng history"
            else:
                res = "bỏ qua / làm việc khác"
            visible = all(present(v, H_seen) for _, v in dep)
            outcome[res] += 1
            records.append({"id": sid, "kind": "dependent", "result": res, "score": s["score"], "call": g["name"],
                            "dep_args": dep, "visible_in_context": visible, "text": r["audio_text"],
                            "pred_same": [p["arguments"] for p in same]})

        # 2. redundant re-reads
        gold_keys = {(c["name"], json.dumps(c["arguments"], sort_keys=True)) for c in gold}
        rr = [p for p in pred if (p["name"], json.dumps(p["arguments"], sort_keys=True)) in hist_reads - gold_keys]
        if rr:
            reread_samples.append(sid)
            reread_loss += 1 - s["score"]
            records.append({"id": sid, "kind": "reread", "score": s["score"], "calls": [p["name"] for p in rr],
                            "text": r["audio_text"]})

    with open(run_dir / "history_usage.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    n = len(scores)
    dep = [x for x in records if x["kind"] == "dependent"]
    dep_samples = {x["id"] for x in dep}
    fail = [x for x in dep if x["result"] != "đúng"]
    fail_samples = {x["id"] for x in fail}
    lost = sum(1 - scores[i]["score"] for i in fail_samples)
    print(f"run {args.run}: {n} samples, {n_hist} with history\n")
    print(f"1) Call gold phụ thuộc history: {len(dep)} call trong {len(dep_samples)} mẫu")
    for k, v in outcome.most_common():
        print(f"   {v:>3}  {k}")
    print(f"   -> {len(fail_samples)} mẫu có ít nhất một call như vậy sai; các mẫu này mất {100 * lost / n:.2f} điểm/100 "
          f"(gồm cả lỗi khác trong cùng mẫu)")
    print(f"   giá trị cần dùng bị cắt khỏi context (history result > {args.hist_chars} ký tự): "
          f"{sum(not x['visible_in_context'] for x in fail)} call sai")
    print(f"\n2) Đọc lại thừa read call đã có trong history: {len(reread_samples)} mẫu, "
          f"các mẫu này mất {100 * reread_loss / n:.2f} điểm/100")
    print(f"\nTổng: {len(fail_samples | set(reread_samples))} mẫu dính ít nhất một dấu hiệu")
    for res in [k for k, _ in outcome.most_common() if k != "đúng"]:
        print(f"\n### {res}")
        for x in [x for x in fail if x["result"] == res][: args.examples]:
            print(f"[{x['id']}] score {x['score']:.2f}  {x['call']}  cần từ history: {x['dep_args']}\n"
                  f"  user: {x['text']}\n  model: {x['pred_same'] or '-'}")
    print("\n### đọc lại thừa")
    for x in [x for x in records if x["kind"] == "reread"][: args.examples]:
        print(f"[{x['id']}] score {x['score']:.2f}  {x['calls']}  user: {x['text']}")


if __name__ == "__main__":
    main()
