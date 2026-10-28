"""Transcribe every user audio of a split (final request + history turns) and cache the transcripts.

    uv run python scripts/run_asr.py --split train --engine gipformer      # -> outputs/asr/train_gip15.json
    uv run python scripts/run_asr.py --split public_test --engine gemma    # Gemma via vLLM -> public_test_ctx.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from lifemate.asr import ASR_PROMPT, TransducerASR, context_prompt, error_rates, transcribe_all  # noqa: E402
from lifemate.data import DATA_ROOT, OUTPUT_ROOT, load_split, user_audio_paths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="only the first N samples (0 = all)")
    ap.add_argument("--base-url", default="http://localhost:8011/v1")
    ap.add_argument("--model", default="gemma-4-e2b")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--prompt", choices=["base", "ctx"], default="ctx",
                    help="gemma only: ctx adds the installed apps as context")
    ap.add_argument("--engine", choices=["gemma", "gipformer"], default="gipformer")
    args = ap.parse_args()

    rows = load_split(args.split)
    if args.limit:
        rows = rows[: args.limit]
    app_names = {a["id"]: a["name_vi"] for a in
                 json.loads((DATA_ROOT / args.split / f"{args.split}_tools_catalog.json").read_text(encoding="utf-8"))["apps"]}
    prompts = {}
    for r in rows:
        prompt = ASR_PROMPT if args.prompt == "base" else context_prompt(
            [app_names.get(a, a) for a in r["context"]["installed_apps"]])
        prompts.update({p: prompt for p in user_audio_paths(r)})
    paths = list(prompts)
    if args.engine == "gipformer":
        cache_file = OUTPUT_ROOT / "asr" / f"{args.split}_gip15.json"
        cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
        todo = [p for p in paths if p not in cache]
        if todo:
            cache.update(zip(todo, TransducerASR().transcribe([DATA_ROOT / args.split / p for p in todo])))
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    else:
        client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=120)
        suffix = "" if args.prompt == "base" else f"_{args.prompt}"
        cache = transcribe_all(client, args.model, DATA_ROOT / args.split, prompts,
                               OUTPUT_ROOT / "asr" / f"{args.split}{suffix}.json", workers=args.workers)
    print(f"{len(paths)} audio paths, cache has {len(cache)} transcripts")

    if "audio_text" in rows[0]:  # train: compare with the reference transcripts
        final = [(r["audio_text"], cache[r["audio"]]) for r in rows]
        hist = [(t["audio_text"], cache[t["audio"]]) for r in rows for t in r["history"]
                if t.get("role") == "user" and t.get("audio_text")]
        report = {"final_turn": error_rates(final), "history_turns": error_rates(hist)}
        print(json.dumps(report, indent=2))
        for ref, hyp in final[:15]:
            print(f"REF: {ref}\nHYP: {hyp}\n")


if __name__ == "__main__":
    main()
