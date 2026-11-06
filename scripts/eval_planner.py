"""Tool accuracy of the planner sub-agent (lifemate/planner.py): the tools of its plan against the gold tools on train.

    uv run python scripts/eval_planner.py --dev 300 --name v6_nothink
    uv run python scripts/eval_planner.py --dev 300 --transcripts gold --name v6_nothink_gold

Plan tools = the tool of every step line ("S1: app.tool | ...") plus every ask line ("ask: app.tool | ..."), kept when
installed. Gold tools and metrics are those of scripts/eval_tool_selector.py. Output: outputs/planner/<split>/<name>/
(llm_calls.jsonl with every request/response, predictions.jsonl with the plans, metrics.json).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai import OpenAI  # noqa: E402
from tqdm import tqdm  # noqa: E402

from eval_tool_selector import metrics  # noqa: E402
from lifemate.data import DATA_ROOT, OUTPUT_ROOT, load_catalog, load_split  # noqa: E402
from lifemate.llm_log import LLMCallLog  # noqa: E402
from lifemate.planner import PlannerConfig, plan_request  # noqa: E402
from lifemate.scorer import category  # noqa: E402
from lifemate.tool_selector import gold_tools  # noqa: E402
from run_agent import load_transcripts  # noqa: E402

OUT = OUTPUT_ROOT / "planner"


def plan_tools(plan: str, installed: set[str]) -> list[str]:
    plan = re.sub(r"\bapp\.(?=[a-z_]+\.[a-z_]+)", "", plan)  # "S1: app.alarm.create" -> "alarm.create"
    found = re.findall(r"^\W*S\d+\s*:\s*\W*([a-z_]+\.[a-z_]+)", plan, re.M)
    found += re.findall(r"^\W*ask\s*:\s*\W*([a-z_]+\.[a-z_]+)", plan, re.M)
    return [t for t in dict.fromkeys(found) if t in installed]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--dev", type=int, default=300, help="random subset size (seed 0); 0 = all")
    ap.add_argument("--transcripts", choices=["asr", "gold"], default="asr")
    ap.add_argument("--asr-cache", default="_gip15")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--base-url", default="http://localhost:8011/v1")
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--name", default="plan")
    args = ap.parse_args()

    rows = load_split(args.split)
    if args.dev:
        rows = random.Random(0).sample(rows, args.dev)
    catalog = load_catalog(args.split)
    raw = json.loads((DATA_ROOT / args.split / f"{args.split}_tools_catalog.json").read_text(encoding="utf-8"))
    raw_apps = {a["id"]: a for a in raw["apps"]}
    transcripts = load_transcripts(args.split, rows, args.transcripts, args.asr_cache)
    out_dir = OUT / args.split / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PlannerConfig(model=args.model, thinking=args.thinking, max_tokens=args.max_tokens)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    llm_log = LLMCallLog(out_dir / "llm_calls.jsonl")

    def one(r):
        try:
            return plan_request(client, cfg, r, transcripts, raw_apps, catalog, llm_log)
        except Exception as e:
            return {"plan": "", "reasoning": "", "finish_reason": "error", "usage": {}, "error": f"{type(e).__name__}: {e}"}

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(tqdm(pool.map(one, rows), total=len(rows), desc="plan"))
    elapsed = time.time() - t0

    preds = []
    for r, res in zip(rows, results):
        inst = r["context"]["installed_apps"]
        g = gold_tools(r["label"]["tool_calls"])
        tools = plan_tools(res["plan"], {t for a in inst.values() for t in a["tools"]})
        apps = list(dict.fromkeys(t.split(".")[0] for t in tools))
        preds.append({"id": r["id"], "category": category(r["label"]["tool_calls"]),
                      "request": transcripts.get(r["audio"], ""), "gold_text": r["audio_text"],
                      "gold_tools": sorted(g), "gold_apps": sorted({t.split(".")[0] for t in g}),
                      "pred_tools": tools, "pred_apps": apps,
                      "tools_of_pred_apps": sum(len(inst[a]["tools"]) for a in apps if a in inst),
                      "installed_tools": sum(len(a["tools"]) for a in inst.values()), **res})
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    m = metrics(preds)
    m["config"] = {**vars(args), "sec_per_sample": round(elapsed / len(rows), 3),
                   "errors": sum(p["finish_reason"] == "error" for p in preds),
                   "length_cutoffs": sum(p["finish_reason"] == "length" for p in preds),
                   "avg_completion_tokens": round(sum(p["usage"].get("completion_tokens", 0) for p in preds) / len(preds), 1)}
    (out_dir / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    a = m["all"]["tools"]
    print(f"{args.name}: covered {a['covered']} exact {a['exact']} P {a['precision']} R {a['recall']} "
          f"avg_pred {a['avg_pred']} | OOS -> empty {m['out_of_scope']['pred_empty']} | "
          f"cutoffs {m['config']['length_cutoffs']} | {m['config']['sec_per_sample']} s/sample")


if __name__ == "__main__":
    main()
