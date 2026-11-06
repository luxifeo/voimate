"""Accuracy of the tool selector sub-agent (lifemate/tool_selector.py) against the tools of the gold calls on train.

    uv run python scripts/eval_tool_selector.py --dev 300 --name zs_nothink
    uv run python scripts/eval_tool_selector.py --dev 300 --thinking --name zs_think
    uv run python scripts/eval_tool_selector.py --dev 300 --fewshot 8 --name fs8_nothink
    uv run python scripts/eval_tool_selector.py --dev 300 --method retrieval --name retrieval   # current bge-m3 app filter
    uv run python scripts/eval_tool_selector.py --report zs_nothink zs_think fs8_nothink retrieval

Gold tools per sample = every tool of label.tool_calls except sys.ask_missing_info, plus the tool an ask is about
(its tool_name argument); out-of-scope samples have none. Output: outputs/tool_selector/<split>/<name>/
(llm_calls.jsonl with every request/response, predictions.jsonl, metrics.json).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai import OpenAI  # noqa: E402
from tqdm import tqdm  # noqa: E402

from lifemate.data import DATA_ROOT, OUTPUT_ROOT, load_catalog, load_split  # noqa: E402
from lifemate.llm_log import LLMCallLog  # noqa: E402
from lifemate.pipeline import build_tool_filters  # noqa: E402
from lifemate.retriever import FewShotRetriever, example_query_text  # noqa: E402
from lifemate.scorer import category  # noqa: E402
from lifemate.tool_selector import SelectorConfig, format_selection_example, gold_tools, select_tools  # noqa: E402
from run_agent import load_transcripts  # noqa: E402

OUT = OUTPUT_ROOT / "tool_selector"


def query_texts(rows, transcripts):
    return [example_query_text(r, transcripts.get(r["audio"], ""),
                               [transcripts.get(t["audio"], "") for t in r["history"] if t.get("role") == "user"])
            for r in rows]


def metrics(preds: list[dict]) -> dict:
    """Micro precision / recall over tools and apps, share of samples whose gold set is fully covered, exact matches,
    and the number of tools the agent would see (selected tools, or every tool of the selected apps)."""
    def block(ps):
        n = max(len(ps), 1)
        out = {"n": len(ps)}
        for unit in ("tools", "apps"):
            tp = sum(len(set(p[f"pred_{unit}"]) & set(p[f"gold_{unit}"])) for p in ps)
            npred = sum(len(p[f"pred_{unit}"]) for p in ps)
            ngold = sum(len(p[f"gold_{unit}"]) for p in ps)
            out[unit] = {"precision": round(100 * tp / max(npred, 1), 2), "recall": round(100 * tp / max(ngold, 1), 2),
                         "covered": round(100 * sum(set(p[f"gold_{unit}"]) <= set(p[f"pred_{unit}"]) for p in ps) / n, 2),
                         "exact": round(100 * sum(set(p[f"gold_{unit}"]) == set(p[f"pred_{unit}"]) for p in ps) / n, 2),
                         "avg_pred": round(npred / n, 2), "avg_gold": round(ngold / n, 2)}
        out["avg_tools_of_pred_apps"] = round(sum(p["tools_of_pred_apps"] for p in ps) / n, 2)
        out["avg_installed_tools"] = round(sum(p["installed_tools"] for p in ps) / n, 2)
        return out

    res = {"all": block(preds)}
    for cat in ("action", "ask_missing_info", "out_of_scope"):
        res[cat] = block([p for p in preds if p["category"] == cat])
    oos = [p for p in preds if p["category"] == "out_of_scope"]
    res["out_of_scope"]["pred_empty"] = round(100 * sum(not p["pred_tools"] for p in oos) / max(len(oos), 1), 2)
    return res


def run(args) -> None:
    rows = load_split(args.split)
    if args.dev:
        rows = random.Random(0).sample(rows, args.dev)
    catalog = load_catalog(args.split)
    raw = json.loads((DATA_ROOT / args.split / f"{args.split}_tools_catalog.json").read_text(encoding="utf-8"))
    raw_apps = {a["id"]: a for a in raw["apps"]}
    transcripts = load_transcripts(args.split, rows, args.transcripts, args.asr_cache)
    out_dir = OUT / args.split / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.method == "retrieval":  # reference: the bge-m3 app filter of run_agent.py --tool-filter retrieval
        retriever = FewShotRetriever(load_split("train"), device="cpu")
        app_descs = {a["id"]: f"{a['name_vi']}: {a.get('description', '')}. "
                              + "; ".join(t.get("description", "") for t in a["tools"]) for a in raw["apps"]}
        sel = build_tool_filters(rows, transcripts, retriever, "retrieval", exclude_self=args.split == "train",
                                 k=args.filter_k, app_descs=app_descs, top_apps=args.filter_top_apps)
        results = {r["id"]: {"apps": sorted(sel[r["id"]]), "tools": [t for a in sorted(sel[r["id"]])
                                                                     for t in r["context"]["installed_apps"][a]["tools"]]}
                   for r in rows}
    else:
        fewshots = {r["id"]: [] for r in rows}
        if args.fewshot:
            retriever = FewShotRetriever(load_split("train"), device="cpu")
            for r, q in zip(rows, retriever.encode(query_texts(rows, transcripts))):
                hits = retriever.search(q, args.fewshot, exclude_id=r["id"] if args.split == "train" else None,
                                        installed_apps=set(r["context"]["installed_apps"]))
                fewshots[r["id"]] = [format_selection_example(h) for h in hits]
        cfg = SelectorConfig(model=args.model, thinking=args.thinking, max_tokens=args.max_tokens)
        client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
        llm_log = LLMCallLog(out_dir / "llm_calls.jsonl")

        def one(r):
            try:
                return r["id"], select_tools(client, cfg, r, transcripts, raw_apps, catalog, fewshots[r["id"]], llm_log)
            except Exception as e:
                return r["id"], {"apps": [], "tools": [], "error": f"{type(e).__name__}: {e}"}

        with ThreadPoolExecutor(args.workers) as pool:
            results = dict(tqdm(pool.map(one, rows), total=len(rows), desc="select"))
    elapsed = time.time() - t0

    preds = []
    for r in rows:
        g = gold_tools(r["label"]["tool_calls"])
        res = results[r["id"]]
        inst = r["context"]["installed_apps"]
        preds.append({"id": r["id"], "category": category(r["label"]["tool_calls"]),
                      "request": transcripts.get(r["audio"], ""),
                      "gold_tools": sorted(g), "gold_apps": sorted({t.split(".")[0] for t in g}),
                      "pred_tools": res["tools"], "pred_apps": res["apps"],
                      "tools_of_pred_apps": sum(len(inst[a]["tools"]) for a in res["apps"] if a in inst),
                      "installed_tools": sum(len(a["tools"]) for a in inst.values()),
                      **{k: res[k] for k in ("raw", "usage", "error") if k in res}})
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    m = metrics(preds)
    m["config"] = {**vars(args), "sec_per_sample": round(elapsed / len(rows), 3),
                   "errors": sum("error" in p for p in preds),
                   "avg_completion_tokens": round(sum(p.get("usage", {}).get("completion_tokens", 0) for p in preds) / len(preds), 1),
                   "avg_prompt_tokens": round(sum(p.get("usage", {}).get("prompt_tokens", 0) for p in preds) / len(preds), 1)}
    (out_dir / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in m.items() if k != "config"}["all"], indent=1), m["config"]["sec_per_sample"], "s/sample")


def report(names: list[str], split: str) -> None:
    ms = {n: json.loads((OUT / split / n / "metrics.json").read_text(encoding="utf-8")) for n in names}
    print("| selector | tool P | tool R | samples all gold tools covered | exact tool set | app R | samples all gold apps covered "
          "| tools selected | tools of selected apps | OOS → empty | s/sample | errors |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, m in ms.items():
        a = m["all"]
        print(f"| {n} | {a['tools']['precision']} | {a['tools']['recall']} | {a['tools']['covered']} | {a['tools']['exact']} "
              f"| {a['apps']['recall']} | {a['apps']['covered']} | {a['tools']['avg_pred']} | {a['avg_tools_of_pred_apps']} "
              f"| {m['out_of_scope']['pred_empty']} | {m['config']['sec_per_sample']} | {m['config']['errors']} |")
    first = next(iter(ms.values()))
    print(f"\nInstalled tools per sample: {first['all']['avg_installed_tools']}; gold tools per sample: "
          f"{first['all']['tools']['avg_gold']}\n")
    print("Samples with all gold tools covered, by category:\n")
    print("| selector | action | ask_missing_info | out_of_scope |\n|---|---|---|---|")
    for n, m in ms.items():
        print(f"| {n} | " + " | ".join(f"{m[c]['tools']['covered']} (n={m[c]['n']})" for c in ("action", "ask_missing_info", "out_of_scope")) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--dev", type=int, default=300, help="random subset size (seed 0); 0 = all")
    ap.add_argument("--transcripts", choices=["asr", "gold"], default="asr")
    ap.add_argument("--asr-cache", default="_gip15")
    ap.add_argument("--method", choices=["llm", "retrieval"], default="llm")
    ap.add_argument("--fewshot", type=int, default=0)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--filter-k", type=int, default=16)
    ap.add_argument("--filter-top-apps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--base-url", default="http://localhost:8011/v1")
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--name", default="sel")
    ap.add_argument("--report", nargs="*", default=None, help="print a comparison table of finished runs")
    args = ap.parse_args()
    if args.report is not None:
        report(args.report, args.split)
    else:
        run(args)


if __name__ == "__main__":
    main()
