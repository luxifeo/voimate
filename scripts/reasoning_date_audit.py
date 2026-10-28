"""Check the date claims the model makes in its reasoning against the sample's real calendar, and the date/time
arguments it sends against the gold calls.

    uv run python scripts/reasoning_date_audit.py llm_qwen35_4b_gold_log
    uv run python scripts/reasoning_date_audit.py llm_qwen35_4b_asr_log --examples 20

Reasoning claims checked (from outputs/runs/train/<run>/llm_calls.jsonl):
- weekday <-> date: "thứ Sáu 2026-11-27", "2026-11-14T12:30 (thứ Năm)", "Friday (2026-11-27)"
- relative day <-> date: "mai (2026-11-12)", "hôm nay là 2026-11-02", "chiều nay (2026-10-14)", "ngày kia (2026-10-28)"
Argument check: for each unmatched model call with a same-name unmatched gold call, every date/datetime argument
that differs is classified by the day and time offset to the gold value.
Writes outputs/runs/train/<run>/date_audit.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]

from error_analysis import unmatched  # noqa: E402
from lifemate.data import OUTPUT_ROOT, load_catalog, load_split  # noqa: E402
from lifemate.scorer import fuzzy_keys  # noqa: E402

WEEKDAY = {"hai": 0, "2": 0, "ba": 1, "3": 1, "tư": 2, "4": 2, "năm": 3, "5": 3, "sáu": 4, "6": 4, "bảy": 5, "7": 5,
           "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
WD_RE = r"(?:thứ\s*(hai|ba|tư|năm|sáu|bảy|[2-7])\b|(chủ\s*nhật)|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b)"
DATE_RE = r"(20\d\d)-(\d\d)-(\d\d)|\b(\d{1,2})/(\d{1,2})/(20\d\d)\b"
REL = [(r"hôm nay|sáng nay|trưa nay|chiều nay|tối nay|đêm nay|\btoday\b|this (?:morning|afternoon|evening)|tonight", 0),
       (r"ngày mai|sáng mai|trưa mai|chiều mai|tối mai|\bmai\b|\btomorrow\b", 1),
       (r"ngày kia|\bmốt\b|day after tomorrow", 2),
       (r"hôm qua|\byesterday\b", -1)]
GAP = r"[\s:,(=là\-–]{0,12}"  # "mai (", "hôm nay là ", "today: "
ISO_DT = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:T(\d{2}):(\d{2}))?")


def parse_date(m: re.Match) -> date | None:
    g = m.groups()
    try:
        return date(int(g[0]), int(g[1]), int(g[2])) if g[0] else date(int(g[5]), int(g[4]), int(g[3]))
    except ValueError:
        return None


def weekday_of(m: re.Match) -> int:
    return 6 if m.group(2) else WEEKDAY[(m.group(1) or m.group(3)).lower()]


def claims(text: str, today: date) -> list[dict]:
    """Every checkable date claim in a reasoning text, with ok = consistent with the real calendar."""
    out, low = [], text.lower()
    wd, dt = re.compile(WD_RE), re.compile(DATE_RE)
    for w in wd.finditer(low):  # weekday then date, or date then weekday, within a short window
        best = None
        for d in dt.finditer(low, max(0, w.start() - 40), min(len(low), w.end() + 25)):
            between = low[d.end():w.start()] if d.end() <= w.start() else low[w.end():d.start()]
            if len(between) <= 22 and not re.search(r"[.;\n]|đến|tới|->|→| - ", between):
                best = d if best is None or abs(d.start() - w.start()) < abs(best.start() - w.start()) else best
        if best and (d0 := parse_date(best)):
            out.append({"kind": "weekday", "said": text[w.start():w.end()], "date": d0.isoformat(),
                        "ok": d0.weekday() == weekday_of(w), "real": d0.strftime("%A"),
                        "ctx": text[max(0, min(w.start(), best.start()) - 30): max(w.end(), best.end()) + 30]})
    for pat, off in REL:
        for r in re.finditer(pat, low):
            d = re.compile(GAP + "(?:" + DATE_RE + ")").match(low, r.end())
            if not d:
                continue
            d0 = parse_date(re.compile(DATE_RE).search(low, r.end()))
            if d0:
                out.append({"kind": f"relative{off:+d}", "said": text[r.start():r.end()], "date": d0.isoformat(),
                            "ok": (d0 - today).days == off, "real": f"{(d0 - today).days:+d} ngày",
                            "ctx": text[max(0, r.start() - 30): d.end() + 30]})
    return out


def date_arg_diffs(pred: dict, gold: dict) -> list[dict]:
    out = []
    pa, ga = pred.get("arguments") or {}, gold.get("arguments") or {}
    for k in pa.keys() & ga.keys():
        p, g = pa[k], ga[k]
        if not (isinstance(p, str) and isinstance(g, str) and p != g):
            continue
        mp, mg = ISO_DT.match(p), ISO_DT.match(g)
        if not (mp and mg):
            continue
        days = (date.fromisoformat(mp.group(1)) - date.fromisoformat(mg.group(1))).days
        kind = f"lệch {days:+d} ngày" if days else "cùng ngày"
        if mp.group(2) and mg.group(2):
            mins = (int(mp.group(2)) * 60 + int(mp.group(3))) - (int(mg.group(2)) * 60 + int(mg.group(3)))
            if mins:
                kind += f", lệch giờ {mins / 60:+g}h"
            elif not days:
                kind = "cùng ngày giờ, khác định dạng/giây/offset"
        out.append({"tool": gold["name"], "key": k, "gold": g, "pred": p, "kind": kind})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()
    run_dir = OUTPUT_ROOT / "runs" / "train" / args.run
    rows = {r["id"]: r for r in load_split("train")}
    fz = fuzzy_keys(load_catalog("train"))
    scores = {d["id"]: d for d in map(json.loads, open(run_dir / "scores.jsonl", encoding="utf-8"))}
    sub = {d["id"]: d for d in map(json.loads, open(run_dir / "submission.jsonl", encoding="utf-8"))}

    records, n_claims, kinds = [], 0, Counter()
    with open(run_dir / "llm_calls.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            msg = ((r.get("response") or {}).get("choices") or [{}])[0].get("message") or {}
            text = msg.get("reasoning_content") or msg.get("reasoning") or ""
            today = datetime.fromisoformat(rows[r["sample_id"]]["context"]["current_time"]).date()
            cs = claims(text, today)
            n_claims += len(cs)
            for c in cs:
                kinds[(c["kind"], c["ok"])] += 1
                if not c["ok"]:
                    records.append({"id": r["sample_id"], "step": r["step"], "type": "reasoning", **c,
                                    "today": f"{today.isoformat()} {today.strftime('%A')}",
                                    "score": scores[r["sample_id"]]["score"]})
    arg_kinds = Counter()
    for sid, s in scores.items():
        extra, missing = unmatched(sub[sid]["tool_calls"], rows[sid]["label"]["tool_calls"], fz)
        for g in missing:
            for p in [p for p in extra if p["name"] == g["name"]][:1]:
                for d in date_arg_diffs(p, g):
                    arg_kinds[d["kind"]] += 1
                    records.append({"id": sid, "type": "argument", "score": s["score"],
                                    "text": rows[sid]["audio_text"], "now": rows[sid]["context"]["current_time"], **d})
    with open(run_dir / "date_audit.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    bad = [x for x in records if x["type"] == "reasoning"]
    print(f"run {args.run}: {n_claims} date claims checked in reasoning, {len(bad)} wrong "
          f"in {len({x['id'] for x in bad})} samples")
    for (k, ok), c in sorted(kinds.items()):
        print(f"  {k:<12} {'đúng' if ok else 'SAI '} {c}")
    args_bad = [x for x in records if x["type"] == "argument"]
    print(f"\nDate/datetime arguments differing from gold: {len(args_bad)} in {len({x['id'] for x in args_bad})} samples")
    for k, c in arg_kinds.most_common():
        print(f"  {c:>3}  {k}")
    print("\n### Reasoning sai lịch")
    for x in bad[: args.examples]:
        print(f"[{x['id']} b{x['step']} score {x['score']:.2f}] hôm nay {x['today']} | nói '{x['said']}' cho {x['date']}"
              f" (thực tế {x['real']})\n    …{' '.join(x['ctx'].split())}…")
    print("\n### Đối số ngày/giờ lệch gold")
    for x in args_bad[: args.examples]:
        print(f"[{x['id']} score {x['score']:.2f}] {x['tool']}.{x['key']}: gold {x['gold']} | model {x['pred']} ({x['kind']})"
              f"\n    now {x['now']} | {x['text']}")


if __name__ == "__main__":
    main()
