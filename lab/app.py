"""LLM lab: browse the logged LLM calls of agent runs next to the gold labels, and replay any call with edited
messages, tools or sampling parameters against a vLLM OpenAI-compatible server.

    uv run --group lab python lab/app.py --port 8765    # then open http://localhost:8765

Runs come from outputs/runs/<split>/<name>/ (submission.jsonl, traces.jsonl, llm_calls.jsonl, scores.jsonl on train).
Replays only send the edited request to the LLM: no ToolEnv is created and no tool is executed.
Every replay is appended to outputs/lab/replays.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from compare_asr import align  # noqa: E402
from error_analysis import LABELS, arg_diff, classify  # noqa: E402
from lifemate.asr import _words  # noqa: E402
from lifemate.data import OUTPUT_ROOT, load_catalog, load_split  # noqa: E402
from lifemate.scorer import call_matches, fuzzy_keys  # noqa: E402

RUNS = OUTPUT_ROOT / "runs"
REPLAY_LOG = OUTPUT_ROOT / "lab" / "replays.jsonl"
_replay_lock = threading.Lock()

app = FastAPI(title="LLM lab")


@lru_cache(maxsize=4)
def split_data(split: str) -> tuple[dict, dict, dict, dict]:
    """(rows by id, catalog, tool kinds, fuzzy keys) of a split."""
    catalog = load_catalog(split)
    return ({r["id"]: r for r in load_split(split)}, catalog, {n: s.get("kind") for n, s in catalog.items()},
            fuzzy_keys(catalog))


def run_dir(split: str, name: str) -> Path:
    d = (RUNS / split / name).resolve()
    if d.parent.parent != RUNS.resolve() or not (d / "submission.jsonl").exists():
        raise HTTPException(404, f"run {split}/{name} not found")
    return d


def read_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {d["id"]: d for d in map(json.loads, f)}


_run_cache: dict[tuple, dict] = {}


def run_files(d: Path) -> dict:
    """submission / scores / traces of a run and the byte offsets of each sample's lines in llm_calls.jsonl,
    cached until one of the files changes."""
    files = ["submission.jsonl", "scores.jsonl", "traces.jsonl", "llm_calls.jsonl"]
    key = (str(d), *((d / f).stat().st_mtime if (d / f).exists() else 0 for f in files))
    if key not in _run_cache:
        offsets: dict[str, list[int]] = {}
        log = d / "llm_calls.jsonl"
        if log.exists():
            with open(log, "rb") as f:
                pos = 0
                for line in f:
                    sid = json.loads(line).get("sample_id")
                    offsets.setdefault(sid, []).append(pos)
                    pos += len(line)
        _run_cache[key] = {"sub": read_jsonl(d / "submission.jsonl"), "scores": read_jsonl(d / "scores.jsonl"),
                           "traces": read_jsonl(d / "traces.jsonl"), "offsets": offsets}
    return _run_cache[key]


def llm_calls(d: Path, offsets: list[int]) -> list[dict]:
    out = []
    with open(d / "llm_calls.jsonl", "rb") as f:
        for off in offsets:
            f.seek(off)
            out.append(json.loads(f.readline()))
    return sorted(out, key=lambda r: r.get("step", 0))


def match_calls(pred: list[dict], gold: list[dict], fz) -> tuple[list, list]:
    """Greedy matching as in the scorer: index of the matched gold call for each pred call (or None) and the
    matched flags of the gold calls; unmatched pred calls with a same-name unmatched gold call get an argument diff."""
    used, pm = [False] * len(gold), []
    for pc in pred:
        for j, gc in enumerate(gold):
            if not used[j] and call_matches(pc, gc, fz):
                used[j] = True
                pm.append(j)
                break
        else:
            pm.append(None)
    pred_out = []
    for pc, j in zip(pred, pm):
        diff = None
        if j is None:
            cands = [g for g, u in zip(gold, used) if not u and g["name"] == pc["name"]]
            if cands:
                diff = min((arg_diff(pc, g, fz) for g in cands), key=len)
        pred_out.append({**pc, "match": j, "diff": diff})
    return pred_out, [{**g, "matched": u} for g, u in zip(gold, used)]


def word_diff(ref: str, hyp: str) -> list[list[str]]:
    """Word alignment after the WER normalization (lower case, NFC, punctuation removed)."""
    return [list(p) for p in align(_words(ref), _words(hyp))]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/runs")
def list_runs():
    out = []
    for d in sorted(RUNS.glob("*/*/submission.jsonl"), key=lambda p: -p.stat().st_mtime):
        d = d.parent
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8")) if (d / "metrics.json").exists() else {}
        out.append({"split": d.parent.name, "name": d.name, "score": m.get("all", {}).get("score"),
                    "has_log": (d / "llm_calls.jsonl").exists() and (d / "llm_calls.jsonl").stat().st_size > 0,
                    "model": m.get("config", {}).get("model")})
    return out


@app.get("/api/runs/{split}/{name}/samples")
def list_samples(split: str, name: str):
    d = run_dir(split, name)
    rows, _, kinds, fz = split_data(split)
    rf = run_files(d)
    out = []
    for sid, sub in rf["sub"].items():
        r, s, t = rows.get(sid, {}), rf["scores"].get(sid), rf["traces"].get(sid, {})
        etype = None
        if s and s["score"] < 1:
            etype = classify(sub["tool_calls"], r["label"]["tool_calls"], s, kinds, fz)[0]
        out.append({"id": sid, "score": s and s["score"], "state_ok": s and s["state_ok"],
                    "category": s and s.get("category"), "etype": etype,
                    "transcript": t.get("transcript") or r.get("audio_text", ""), "n_calls": len(sub["tool_calls"]),
                    "n_llm": len(rf["offsets"].get(sid, [])), "stop": t.get("stop"), "error": t.get("error"),
                    "history": len(r.get("history", []))})
    return {"samples": out, "labels": LABELS}


@app.get("/api/runs/{split}/{name}/samples/{sid}")
def sample_detail(split: str, name: str, sid: str):
    d = run_dir(split, name)
    rows, _, kinds, fz = split_data(split)
    rf = run_files(d)
    if sid not in rf["sub"]:
        raise HTTPException(404, f"{sid} not in run")
    r, s, t = rows[sid], rf["scores"].get(sid), rf["traces"].get(sid, {})
    pred = rf["sub"][sid]["tool_calls"]
    label = r.get("label")
    out = {"id": sid, "context": r["context"], "history": r.get("history", []), "score": s,
           "transcript": t.get("transcript"), "gold_text": r.get("audio_text"), "stop": t.get("stop"),
           "error": t.get("error"), "trace": t.get("trace", []),
           "llm_calls": llm_calls(d, rf["offsets"][sid]) if sid in rf["offsets"] else [],
           "kinds": {c["name"]: kinds.get(c["name"]) for c in pred + (label["tool_calls"] if label else [])}}
    if label:
        out["pred"], out["gold"] = match_calls(pred, label["tool_calls"], fz)
        out["gold_results"] = label.get("tool_results", [])
        if s and s["score"] < 1:
            out["etype"] = classify(pred, label["tool_calls"], s, kinds, fz)[0]
        if t.get("transcript") and r.get("audio_text"):
            out["word_diff"] = word_diff(r["audio_text"], t["transcript"])
    else:
        out["pred"], out["gold"] = [{**c, "match": None, "diff": None} for c in pred], None
    return out


def _http(method: str, url: str, body: dict | None = None, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, method=method, headers={"Content-Type": "application/json"},
                                 data=json.dumps(body, ensure_ascii=False).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise HTTPException(e.code, e.read().decode(errors="replace")[:4000])
    except (urllib.error.URLError, TimeoutError) as e:
        raise HTTPException(502, f"{url}: {e}")


@app.get("/api/models")
def models(base_url: str):
    return [m["id"] for m in _http("GET", base_url.rstrip("/") + "/models", timeout=10).get("data", [])]


class Replay(BaseModel):
    base_url: str
    body: dict
    source: dict | None = None  # {run, sample_id, step} of the logged call being edited


@app.post("/api/chat")
def replay(req: Replay):
    """Send a wire-format chat.completions body as-is (vLLM extras such as chat_template_kwargs, top_k at top level)."""
    t0 = time.time()
    resp = _http("POST", req.base_url.rstrip("/") + "/chat/completions", req.body)
    latency = round(time.time() - t0, 3)
    REPLAY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _replay_lock, open(REPLAY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "source": req.source,
                            "base_url": req.base_url, "latency_s": latency, "request": req.body, "response": resp},
                           ensure_ascii=False) + "\n")
    return {"response": resp, "latency_s": latency}


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
