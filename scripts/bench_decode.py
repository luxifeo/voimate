"""gipformer decoding settings on the dev clip set: greedy (production) vs the modified beam search that the official
gipformer scripts default to. Experiment only; the production pipeline is untouched.

    uv run python scripts/bench_decode.py run
    uv run python scripts/bench_decode.py report

Clip set = bench_enhance "dev" (every user audio of random.Random(0).sample(train, 300)). Audio is read exactly as in
production (soundfile float32, 16 kHz mono). Transcripts go to outputs/asr/train_gip15_dec_<variant>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_enhance import clip_set, references  # noqa: E402
from bench_tempo import errors, wer  # noqa: E402
from lifemate.asr import TransducerASR  # noqa: E402
from lifemate.data import DATA_ROOT, OUTPUT_ROOT  # noqa: E402

VARIANTS = {  # name: (OfflineRecognizer.from_transducer kwargs, batch size)
    "greedy_b32": ({"decoding_method": "greedy_search"}, 32),  # production settings, re-run for timing
    "greedy_b1": ({"decoding_method": "greedy_search"}, 1),
    "beam4": ({"decoding_method": "modified_beam_search", "max_active_paths": 4}, 32),
    "beam8": ({"decoding_method": "modified_beam_search", "max_active_paths": 8}, 32),
}


def cache_file(name: str) -> Path:
    return OUTPUT_ROOT / "asr" / f"train_gip15_dec_{name}.json"


def recognizer(kwargs: dict, threads: int):
    import sherpa_onnx
    from huggingface_hub import snapshot_download

    from lifemate.asr import GIPFORMER

    d = Path(snapshot_download(GIPFORMER, allow_patterns=["*.onnx", "tokens.txt", "config.json"]))
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))["onnx"]
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(d / cfg["encoder"]), decoder=str(d / cfg["decoder"]), joiner=str(d / cfg["joiner"]),
        tokens=str(d / "tokens.txt"), num_threads=threads, **kwargs)


def run(names: list[str], threads: int) -> None:
    clips = clip_set("dev")
    audio = [sf.read(DATA_ROOT / "train" / p, dtype="float32") for p in clips]
    timing = {}
    for name in names:
        kwargs, batch = VARIANTS[name]
        rec = recognizer(kwargs, threads)
        t0 = time.time()
        texts = []
        for i in range(0, len(clips), batch):
            streams = []
            for x, sr in audio[i:i + batch]:
                s = rec.create_stream()
                s.accept_waveform(sr, x)
                streams.append(s)
            rec.decode_streams(streams)
            texts += [TransducerASR.tidy(s.result.text) for s in streams]
        timing[name] = (time.time() - t0) / len(clips)
        cache_file(name).write_text(json.dumps(dict(zip(clips, texts)), ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"{name}: {len(clips)} clips, {timing[name]:.3f} s/clip ({threads} threads)", flush=True)
    timing_file = OUTPUT_ROOT / "asr" / "dec_timing.json"
    old = json.loads(timing_file.read_text(encoding="utf-8")) if timing_file.exists() else {}
    timing_file.write_text(json.dumps({**old, **timing}, indent=1), encoding="utf-8")


def report() -> None:
    clips = clip_set("dev")
    refs, finals = references()
    hyps = {"greedy_b32 (production)": json.loads((OUTPUT_ROOT / "asr" / "train_gip15.json").read_text(encoding="utf-8"))}
    hyps |= {n: json.loads(cache_file(n).read_text(encoding="utf-8")) for n in VARIANTS if cache_file(n).exists()}
    timing = json.loads((OUTPUT_ROOT / "asr" / "dec_timing.json").read_text(encoding="utf-8"))
    base = "greedy_b32 (production)"
    err = {n: {p: errors(refs[p], h[p]) for p in clips} for n, h in hyps.items()}
    print(f"{len(clips)} clips\n")
    print("| variant | WER all | WER final | WER history | clips better / worse vs production | identical text | s/clip |")
    print("|---|---|---|---|---|---|---|")
    for n, e in err.items():
        up = sum(e[p][0] < err[base][p][0] for p in clips)
        down = sum(e[p][0] > err[base][p][0] for p in clips)
        same = sum(hyps[n][p] == hyps[base][p] for p in clips)
        print(f"| {n} | {wer(e.values()):.2f} | {wer([e[p] for p in clips if p in finals]):.2f} | "
              f"{wer([e[p] for p in clips if p not in finals]):.2f} | {up} / {down} | {same} | {timing.get(n, float('nan')):.3f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    run(args.variants, args.threads) if args.cmd == "run" else report()


if __name__ == "__main__":
    main()
