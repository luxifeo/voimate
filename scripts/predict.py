"""End-to-end inference on one split: ASR (Gemma-4-E2B audio) -> few-shot retrieval (bge-m3) -> agent -> submission.jsonl.
Needs a running vLLM server for gemma-4-e2b (scripts/serve_vllm.sh locally, docker/entrypoint.sh in the image).

    uv run python scripts/predict.py --root data/train+public_test --split public_test --out outputs/submission.jsonl
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
from lifemate.asr import context_prompt, transcribe_all  # noqa: E402
from lifemate.data import OUTPUT_ROOT, load_catalog, load_split, user_audio_paths  # noqa: E402
from lifemate.pipeline import build_fewshots, run_split, write_submission  # noqa: E402
from lifemate.retriever import FewShotRetriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="folder holding the split folders")
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--out", type=Path, default=Path("submission.jsonl"))
    ap.add_argument("--train-jsonl", type=Path, default=None, help="labelled train rows for few-shot (default <root>/train/train.jsonl)")
    ap.add_argument("--index-dir", type=Path, default=OUTPUT_ROOT / "index", help="cache of the train embeddings")
    ap.add_argument("--base-url", default="http://localhost:8011/v1")
    ap.add_argument("--model", default="gemma-4-e2b")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--fewshot", type=int, default=4)
    ap.add_argument("--embed-device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="only the first N samples (smoke test)")
    args = ap.parse_args()

    t0 = time.time()
    rows = load_split(args.split, args.root)
    if args.limit:
        rows = rows[: args.limit]
    catalog = load_catalog(args.split, args.root)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300)

    # 1. ASR of every user audio (history turns + final request), with the installed apps as context
    raw_catalog = json.loads((args.root / args.split / f"{args.split}_tools_catalog.json").read_text(encoding="utf-8"))
    app_names = {a["id"]: a["name_vi"] for a in raw_catalog["apps"]}
    prompts = {}
    for r in rows:
        prompt = context_prompt([app_names.get(a, a) for a in r["context"]["installed_apps"]])
        prompts.update({p: prompt for p in user_audio_paths(r)})
    asr_cache = args.out.parent / f"asr_{args.split}.json"
    transcripts = transcribe_all(client, args.model, args.root / args.split, prompts, asr_cache, workers=args.workers)
    print(f"asr: {len(prompts)} audios in {time.time() - t0:.0f}s", flush=True)

    # 2. few-shot demonstrations retrieved from the labelled train split
    retriever = None
    if args.fewshot:
        train_jsonl = args.train_jsonl or args.root / "train" / "train.jsonl"
        with open(train_jsonl, encoding="utf-8") as f:
            train_rows = [json.loads(line) for line in f if line.strip()]
        retriever = FewShotRetriever(train_rows, device=args.embed_device, cache_dir=args.index_dir)
    fewshots = build_fewshots(rows, transcripts, retriever, args.fewshot, exclude_self=False)

    # 3. agent, one ToolEnv per sample
    cfg = AgentConfig(model=args.model, thinking=True)
    results = run_split(args.split, args.root, rows, catalog, transcripts, fewshots, cfg, client, args.workers)
    write_submission(args.out, rows, results)
    n_err = sum(1 for v in results.values() if "error" in v[2])
    print(f"{len(rows)} lines -> {args.out} in {time.time() - t0:.0f}s, agent errors: {n_err}", flush=True)


if __name__ == "__main__":
    main()
