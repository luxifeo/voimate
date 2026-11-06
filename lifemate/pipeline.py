"""Split-level inference shared by the evaluation script (scripts/run_agent.py) and the submission script
(scripts/predict.py): few-shot retrieval, the agent over every sample, submission writing and train scoring."""
from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from .agent import AgentConfig, run_agent
from .data import new_env
from .llm_log import LLMCallLog
from .retriever import FewShotRetriever, example_query_text, format_example
from .scorer import category, fuzzy_keys, score_sample, summarize


def build_fewshots(rows: list[dict], transcripts: dict[str, str], retriever: FewShotRetriever | None, k: int,
                   exclude_self: bool, style: str = "full") -> dict[str, list[str]]:
    if not k or retriever is None:
        return {r["id"]: [] for r in rows}
    queries = [
        example_query_text(r, transcripts.get(r["audio"], ""),
                           [transcripts.get(t["audio"], "") for t in r["history"] if t.get("role") == "user"])
        for r in rows
    ]
    out = {}
    for r, q in zip(rows, retriever.encode(queries)):
        hits = retriever.search(q, k, exclude_id=r["id"] if exclude_self else None,
                                installed_apps=set(r["context"]["installed_apps"]))
        out[r["id"]] = [format_example(h, style=style) for h in hits]
    return out


def _apps(calls: list[dict]) -> set[str]:
    return {c["name"].split(".")[0] for c in calls} - {"sys"}


def history_apps(item: dict) -> set[str]:
    return _apps([c for t in item.get("history", []) for c in (t.get("tool_calls") or [])])


def build_tool_filters(rows: list[dict], transcripts: dict[str, str], retriever: FewShotRetriever | None, mode: str,
                       exclude_self: bool, k: int = 16, app_descs: dict[str, str] | None = None,
                       top_apps: int = 3) -> dict[str, set[str] | None]:
    """Installed apps whose tools are offered to the agent (None = all).
    oracle: apps of the gold calls; all apps when there is none
    (out-of-scope requests), so the filter itself never reveals the answer "no call". Analysis only (needs labels).
    retrieval: apps of the gold calls of the k nearest train examples, plus the top_apps apps whose description is
    closest to the request, plus apps used earlier in the conversation."""
    if mode == "none":
        return {r["id"]: None for r in rows}
    out = {}
    if mode == "oracle":
        for r in rows:
            sel = _apps(r["label"]["tool_calls"]) & set(r["context"]["installed_apps"])
            out[r["id"]] = sel or None
        return out
    queries = retriever.encode([
        example_query_text(r, transcripts.get(r["audio"], ""),
                           [transcripts.get(t["audio"], "") for t in r["history"] if t.get("role") == "user"])
        for r in rows])
    ids = list(app_descs)
    app_emb = retriever.encode([app_descs[a] for a in ids])
    for r, q in zip(rows, queries):
        installed = set(r["context"]["installed_apps"])
        hits = retriever.search(q, k, exclude_id=r["id"] if exclude_self else None, installed_apps=installed)
        by_desc = [ids[i] for i in (-(app_emb @ q)).argsort() if ids[i] in installed][:top_apps]
        out[r["id"]] = (set().union(*[_apps(h["label"]["tool_calls"]) for h in hits]) | set(by_desc)
                        | history_apps(r)) & installed
    return out


def param_usage_notes(train_rows: list[dict], catalog: dict, min_uses: int = 5) -> dict[str, dict[str, str]]:
    """{tool: {optional parameter: note}} from how often the labelled train calls fill each optional parameter:
    rarely (<10%) -> only when the user says it; usually (>=80%) -> normally filled, with the usual value when one
    value covers >=80% of the uses."""
    uses, filled = Counter(), Counter()
    values: dict[tuple, Counter] = {}
    for r in train_rows:
        for c in r["label"]["tool_calls"]:
            uses[c["name"]] += 1
            for k, v in (c.get("arguments") or {}).items():
                filled[(c["name"], k)] += 1
                values.setdefault((c["name"], k), Counter())[json.dumps(v, ensure_ascii=False)] += 1
    notes: dict[str, dict[str, str]] = {}
    for name, spec in catalog.items():
        params = spec.get("parameters") or {}
        required = set(params.get("required", []))
        n = uses[name]
        if n < min_uses:
            continue
        for k in params.get("properties", {}):
            if k in required:
                continue
            rate = filled[(name, k)] / n
            if rate < 0.1:
                notes.setdefault(name, {})[k] = "hầu như không dùng: chỉ điền khi người dùng nói rõ giá trị này"
            elif rate >= 0.8:
                note = "thường được điền"
                top, cnt = values[(name, k)].most_common(1)[0]
                if cnt / filled[(name, k)] >= 0.8:
                    note += f", thường là {top}"
                notes.setdefault(name, {})[k] = note
    return notes


def run_split(split: str, root: Path, rows: list[dict], catalog: dict, transcripts: dict[str, str],
              fewshots: dict[str, list[str]], cfg: AgentConfig, client: OpenAI, workers: int = 32,
              llm_log: LLMCallLog | None = None, tool_apps: dict[str, set[str] | None] | None = None,
              param_notes: dict[str, dict[str, str]] | None = None,
              param_notes_by_id: dict[str, dict[str, dict[str, str]]] | None = None) -> dict[str, tuple]:
    """{sample_id: (tool_calls, state_hash, info)} with exactly one ToolEnv per sample.
    param_notes_by_id overrides param_notes per sample (k-fold notes when evaluating on the whole train split)."""
    def infer(item):
        env = new_env(split, item["id"], root)
        notes = param_notes_by_id[item["id"]] if param_notes_by_id is not None else param_notes
        try:
            res = run_agent(client, cfg, item, env, catalog, transcripts, fewshots.get(item["id"], []),
                            audio_path=root / split / item["audio"], llm_log=llm_log,
                            apps=(tool_apps or {}).get(item["id"]), param_notes=notes)
            return item["id"], res.tool_calls, res.state_hash, {"trace": res.trace, "stop": res.stop_reason}
        except Exception as e:  # keep the calls already made: they are part of the submission
            return item["id"], env.get_tool_calls(), env.get_state_hash(), {"error": f"{type(e).__name__}: {e}"}

    results = {}
    with ThreadPoolExecutor(workers) as pool:
        for sid, calls, h, info in tqdm(pool.map(infer, rows), total=len(rows), desc="agent"):
            results[sid] = (calls, h, info)
    return results


def write_submission(path: Path, rows: list[dict], results: dict[str, tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            calls, h, _ = results[r["id"]]
            f.write(json.dumps({"id": r["id"], "tool_calls": calls, "state_hash": h}, ensure_ascii=False) + "\n")


def score_split(split: str, root: Path, rows: list[dict], catalog: dict, results: dict[str, tuple]) -> tuple[dict, list[dict]]:
    """Score labelled rows: the gold state hash comes from replaying label.tool_calls on a fresh ToolEnv."""
    fz = fuzzy_keys(catalog)
    per_sample = []
    for r in rows:
        gold_calls = r["label"]["tool_calls"]
        genv = new_env(split, r["id"], root)
        for c in gold_calls:
            genv.call(c["name"], c["arguments"])
        calls, h, _ = results[r["id"]]
        s = score_sample(calls, h, gold_calls, genv.get_state_hash(), fz)
        per_sample.append({"id": r["id"], "category": category(gold_calls), **s})
    return summarize(per_sample), per_sample
