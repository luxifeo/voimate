"""End-to-end inference on one split: ASR (gipformer 1.5, CPU) -> few-shot retrieval (bge-m3) -> agent (Qwen3.5-4B)
-> submission.jsonl. Needs a running vLLM server for qwen3.5-4b (scripts/serve_vllm.sh locally, docker/entrypoint.sh
in the image).

    uv run python scripts/predict.py --root data/train+public_test --split public_test --out outputs/submission.jsonl

Agent config = c1_combo in claudedocs/llm-hypotheses-2026-10-28.md: 8 few-shot examples, week-tagged date table,
optional-parameter guide with usage notes from the labelled train calls.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from lifemate.agent import AgentConfig  # noqa: E402
from lifemate.asr import TransducerASR  # noqa: E402
from lifemate.data import OUTPUT_ROOT, load_catalog, load_split, user_audio_paths  # noqa: E402
from lifemate.pipeline import build_fewshots, param_usage_notes, run_split, write_submission  # noqa: E402
from lifemate.retriever import FewShotRetriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="folder holding the split folders")
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--out", type=Path, default=Path("submission.jsonl"))
    ap.add_argument("--train-jsonl", type=Path, default=None, help="labelled train rows for few-shot (default <root>/train/train.jsonl)")
    ap.add_argument("--index-dir", type=Path, default=OUTPUT_ROOT / "index", help="cache of the train embeddings")
    ap.add_argument("--base-url", default="http://localhost:8011/v1")
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--fewshot", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096, help="completion budget per LLM call (incl. reasoning)")
    ap.add_argument("--asr-threads", type=int, default=8)
    ap.add_argument("--embed-device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="only the first N samples (smoke test)")
    args = ap.parse_args()

    t0 = time.time()
    rows = load_split(args.split, args.root)
    if args.limit:
        rows = rows[: args.limit]
    catalog = load_catalog(args.split, args.root)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300)

    # 1. ASR of every user audio (history turns + final request)
    paths = list(dict.fromkeys(p for r in rows for p in user_audio_paths(r)))
    asr_cache = args.out.parent / f"asr_gip15_{args.split}.json"
    transcripts = json.loads(asr_cache.read_text(encoding="utf-8")) if asr_cache.exists() else {}
    todo = [p for p in paths if p not in transcripts]
    if todo:
        asr = TransducerASR(num_threads=args.asr_threads)
        transcripts.update(zip(todo, asr.transcribe([args.root / args.split / p for p in todo])))
        asr_cache.parent.mkdir(parents=True, exist_ok=True)
        asr_cache.write_text(json.dumps(transcripts, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"asr: {len(paths)} audios in {time.time() - t0:.0f}s", flush=True)

    # 2. few-shot demonstrations and optional-parameter usage notes from the labelled train split
    train_jsonl = args.train_jsonl or args.root / "train" / "train.jsonl"
    with open(train_jsonl, encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f if line.strip()]
    retriever = FewShotRetriever(train_rows, device=args.embed_device, cache_dir=args.index_dir) if args.fewshot else None
    fewshots = build_fewshots(rows, transcripts, retriever, args.fewshot, exclude_self=False)
    param_notes = param_usage_notes(train_rows, catalog)

    # 3. agent, one ToolEnv per sample
    cfg = AgentConfig(model=args.model, max_tokens=args.max_tokens, thinking=True, date_table="weeks", arg_guide=True)
    results = run_split(args.split, args.root, rows, catalog, transcripts, fewshots, cfg, client, args.workers,
                        param_notes=param_notes)
    write_submission(args.out, rows, results)
    n_err = sum(1 for v in results.values() if "error" in v[2])
    print(f"{len(rows)} lines -> {args.out} in {time.time() - t0:.0f}s, agent errors: {n_err}", flush=True)


if __name__ == "__main__":
    main()
