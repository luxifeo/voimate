"""Run the agent on a split from cached transcripts, write submission.jsonl + traces, and score it on train.

    uv run python scripts/run_agent.py --split train --dev 300 --thinking --name v3
    uv run python scripts/run_agent.py --split train --dev 300 --thinking --transcripts gold --name v3_gold
    uv run python scripts/run_agent.py --split public_test --thinking --name v3
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from lifemate.agent import AgentConfig  # noqa: E402
from lifemate.data import DATA_ROOT, OUTPUT_ROOT, load_catalog, load_split  # noqa: E402
from lifemate.llm_log import LLMCallLog  # noqa: E402
from lifemate.pipeline import (build_fewshots, build_tool_filters, param_usage_notes, run_split,  # noqa: E402
                               score_split, write_submission)
from lifemate.retriever import FewShotRetriever  # noqa: E402


def load_transcripts(split: str, rows: list[dict], mode: str, asr_cache: str) -> dict[str, str]:
    if mode == "gold":
        out = {}
        for r in rows:
            out[r["audio"]] = r["audio_text"]
            for t in r["history"]:
                if t.get("role") == "user":
                    out[t["audio"]] = t["audio_text"]
        return out
    return json.loads((OUTPUT_ROOT / "asr" / f"{split}{asr_cache}.json").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--dev", type=int, default=0, help="train only: random subset size (seed 0); 0 = all")
    ap.add_argument("--ids", default="", help="comma-separated sample ids")
    ap.add_argument("--transcripts", choices=["asr", "gold"], default="asr")
    ap.add_argument("--asr-cache", default="_ctx", help="suffix of outputs/asr/<split><suffix>.json ('' = base prompt)")
    ap.add_argument("--fewshot", type=int, default=4)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--no-nudge", action="store_true")
    ap.add_argument("--no-keep-reasoning", action="store_true")
    ap.add_argument("--input-mode", choices=["text", "audio", "audio+text"], default="text",
                    help="how the final request reaches the agent")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--base-url", default="http://localhost:8011/v1")
    ap.add_argument("--model", default="gemma-4-e2b", help="served model name")
    ap.add_argument("--max-tokens", type=int, default=1024, help="completion budget per LLM call (incl. reasoning)")
    ap.add_argument("--embed-device", default="cpu")
    ap.add_argument("--name", default="run")
    # experiment switches (defaults = submitted behaviour)
    ap.add_argument("--fewshot-style", choices=["full", "calls"], default="full")
    ap.add_argument("--tool-filter", choices=["none", "oracle", "retrieval"], default="none")
    ap.add_argument("--filter-k", type=int, default=16, help="retrieval filter: nearest train examples whose apps are kept")
    ap.add_argument("--filter-top-apps", type=int, default=3, help="retrieval filter: apps kept by description similarity")
    ap.add_argument("--time-position", choices=["system", "end", "user"], default="system")
    ap.add_argument("--date-table", choices=["v1", "weeks"], default="v1")
    ap.add_argument("--arg-guide", action="store_true")
    ap.add_argument("--history-mode", choices=["messages", "summary", "llm_summary"], default="messages")
    ap.add_argument("--calculator", action="store_true")
    args = ap.parse_args()

    rows = load_split(args.split)
    if args.ids:
        keep = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in keep]
    elif args.dev:
        rows = random.Random(0).sample(rows, args.dev)
    catalog = load_catalog(args.split)
    transcripts = load_transcripts(args.split, rows, args.transcripts, args.asr_cache)
    cfg = AgentConfig(model=args.model, max_tokens=args.max_tokens, thinking=args.thinking, input_mode=args.input_mode,
                      nudge=not args.no_nudge, keep_reasoning=not args.no_keep_reasoning,
                      time_position=args.time_position, date_table=args.date_table, arg_guide=args.arg_guide,
                      history_mode=args.history_mode, calculator=args.calculator)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300)
    train_rows = load_split("train")
    need_retriever = args.fewshot or args.tool_filter == "retrieval"
    retriever = FewShotRetriever(train_rows, device=args.embed_device) if need_retriever else None
    fewshots = build_fewshots(rows, transcripts, retriever, args.fewshot, exclude_self=args.split == "train",
                              style=args.fewshot_style)
    app_descs = {a["id"]: f"{a['name_vi']}: {a.get('description', '')}. "
                          + "; ".join(t.get("description", "") for t in a["tools"])
                 for a in json.loads((DATA_ROOT / args.split / f"{args.split}_tools_catalog.json").read_text(encoding="utf-8"))["apps"]}
    tool_apps = build_tool_filters(rows, transcripts, retriever, args.tool_filter, exclude_self=args.split == "train",
                                   k=args.filter_k, app_descs=app_descs, top_apps=args.filter_top_apps)
    # usage notes from labelled train rows outside the evaluated subset
    evaluated = {r["id"] for r in rows} if args.split == "train" else set()
    param_notes = param_usage_notes([r for r in train_rows if r["id"] not in evaluated], catalog) if args.arg_guide else None

    out_dir = OUTPUT_ROOT / "runs" / args.split / args.name
    llm_log = LLMCallLog(out_dir / "llm_calls.jsonl")  # every request/response, browsable in lab/app.py
    t0 = time.time()
    results = run_split(args.split, DATA_ROOT, rows, catalog, transcripts, fewshots, cfg, client, args.workers, llm_log,
                        tool_apps=tool_apps, param_notes=param_notes)
    elapsed = time.time() - t0

    write_submission(out_dir / "submission.jsonl", rows, results)
    with open(out_dir / "traces.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            apps = tool_apps.get(r["id"])
            f.write(json.dumps({"id": r["id"], "transcript": transcripts.get(r["audio"]), **results[r["id"]][2],
                                **({"tool_apps": sorted(apps)} if apps is not None else {})},
                               ensure_ascii=False) + "\n")
    n_err = sum(1 for v in results.values() if "error" in v[2])
    print(f"{len(rows)} samples in {elapsed:.0f}s ({elapsed / max(len(rows), 1):.2f}s/sample), errors: {n_err}")

    if "label" not in rows[0]:
        return
    summary, per_sample = score_split(args.split, DATA_ROOT, rows, catalog, results)
    summary["config"] = {**vars(args), "agent": vars(cfg), "sec_per_sample": round(elapsed / max(len(rows), 1), 2)}
    (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "scores.jsonl", "w", encoding="utf-8") as f:
        for s in per_sample:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
