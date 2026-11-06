"""Does slowing the audio down before gipformer ASR lower WER? Experiment only; the production pipeline is untouched.

    uv run python scripts/bench_tempo.py run                  # transcribe the dev clip set under every METHODS entry
    uv run python scripts/bench_tempo.py report               # WER tables (overall, by speaking rate, rate-gated)

Clip set = bench_enhance "dev": every user audio of random.Random(0).sample(train, 300).
Audio is decoded and transformed by ffmpeg to 16 kHz mono float32, then transcribed with the production TransducerASR
recognizer (batch 32). Transcripts go to outputs/asr/train_gip15_tempo_<method>.json ({rel_path: text}).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_enhance import clip_set, references  # noqa: E402
from lifemate.asr import TransducerASR, _edit_distance, _words  # noqa: E402
from lifemate.data import DATA_ROOT, OUTPUT_ROOT  # noqa: E402

SR = 16000
METHODS = {  # name: ffmpeg -af chain applied after resampling to 16 kHz
    "raw": "",
    "atempo90": "atempo=0.9",  # WSOLA time stretch, pitch kept
    "atempo80": "atempo=0.8",
    "atempo70": "atempo=0.7",
    "rubberband80": "rubberband=tempo=0.8",  # phase-vocoder time stretch, pitch kept
    "speed90": f"asetrate={int(SR * 0.9)},aresample={SR}",  # resample: 0.9x speed, pitch lowered (Kaldi speed perturb)
}


def cache_file(method: str) -> Path:
    return OUTPUT_ROOT / "asr" / f"train_gip15_tempo_{method}.json"


def load_audio(path: Path, chain: str) -> np.ndarray:
    af = f"aresample={SR}" + (f",{chain}" if chain else "")
    out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-af", af, "-ac", "1",
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def run(methods: list[str], threads: int) -> None:
    clips = clip_set("dev")
    asr = TransducerASR(num_threads=threads)
    for m in methods:
        if cache_file(m).exists():
            print(m, "cached")
            continue
        with ThreadPoolExecutor(8) as pool:
            audio = list(pool.map(lambda p: load_audio(DATA_ROOT / "train" / p, METHODS[m]), clips))
        texts = []
        for i in range(0, len(clips), 32):
            streams = []
            for a in audio[i:i + 32]:
                s = asr.recognizer.create_stream()
                s.accept_waveform(SR, a)
                streams.append(s)
            asr.recognizer.decode_streams(streams)
            texts += [asr.tidy(s.result.text) for s in streams]
        cache_file(m).write_text(json.dumps(dict(zip(clips, texts)), ensure_ascii=False, indent=0), encoding="utf-8")
        print(m, len(clips), "clips", flush=True)
    durations = OUTPUT_ROOT / "asr" / "tempo_durations.json"
    if not durations.exists():
        with ThreadPoolExecutor(8) as pool:
            lens = list(pool.map(lambda p: len(load_audio(DATA_ROOT / "train" / p, "")) / SR, clips))
        durations.write_text(json.dumps(dict(zip(clips, lens))), encoding="utf-8")


def errors(ref: str, hyp: str) -> tuple[int, int]:
    rw = _words(ref)
    return _edit_distance(rw, _words(hyp)), len(rw)


def wer(pairs: list[tuple[int, int]]) -> float:
    return 100 * sum(e for e, _ in pairs) / max(sum(n for _, n in pairs), 1)


def report() -> None:
    clips = clip_set("dev")
    refs, finals = references()
    dur = json.loads((OUTPUT_ROOT / "asr" / "tempo_durations.json").read_text(encoding="utf-8"))
    hyps = {m: json.loads(cache_file(m).read_text(encoding="utf-8")) for m in METHODS if cache_file(m).exists()}
    err = {m: {p: errors(refs[p], h[p]) for p in clips} for m, h in hyps.items()}
    prod = json.loads((OUTPUT_ROOT / "asr" / "train_gip15.json").read_text(encoding="utf-8"))
    print(f"{len(clips)} clips; raw transcripts identical to production cache: "
          f"{sum(hyps['raw'][p] == prod[p] for p in clips)}/{len(clips)}\n")

    print("| method | WER all | WER final | WER history | clips better / worse vs raw |\n|---|---|---|---|---|")
    for m in hyps:
        e = err[m]
        up = sum(e[p][0] < err["raw"][p][0] for p in clips)
        down = sum(e[p][0] > err["raw"][p][0] for p in clips)
        print(f"| {m} | {wer(e.values()):.2f} | {wer([e[p] for p in clips if p in finals]):.2f} | "
              f"{wer([e[p] for p in clips if p not in finals]):.2f} | {up} / {down} |")

    # speaking rate from the reference (analysis) and from the raw hypothesis (usable at test time)
    ref_rate = {p: len(_words(refs[p])) / max(dur[p], 0.1) for p in clips}
    hyp_rate = {p: len(_words(hyps["raw"][p])) / max(dur[p], 0.1) for p in clips}
    cuts = np.quantile(list(ref_rate.values()), [1 / 3, 2 / 3])
    bins = [("slow", lambda r: r < cuts[0]), ("mid", lambda r: cuts[0] <= r < cuts[1]), ("fast", lambda r: r >= cuts[1])]
    print(f"\nWER by reference speaking rate (words/s tertiles {cuts[0]:.2f} / {cuts[1]:.2f})\n")
    print("| method | " + " | ".join(b for b, _ in bins) + " |\n|---|" + "---|" * len(bins))
    for m in hyps:
        print(f"| {m} | " + " | ".join(f"{wer([err[m][p] for p in clips if f(ref_rate[p])]):.2f}" for _, f in bins) + " |")

    print("\nRate-gated: slowed transcript only where the raw-hypothesis rate >= the q-quantile\n")
    qs = [0.5, 0.67, 0.8, 0.9]
    print("| method | " + " | ".join(f"q={q}" for q in qs) + " |\n|---|" + "---|" * len(qs))
    for m in hyps:
        if m == "raw":
            continue
        row = []
        for q in qs:
            t = np.quantile(list(hyp_rate.values()), q)
            row.append(f"{wer([err[m][p] if hyp_rate[p] >= t else err['raw'][p] for p in clips]):.2f}")
        print(f"| {m} | " + " | ".join(row) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    run(args.methods, args.threads) if args.cmd == "run" else report()


if __name__ == "__main__":
    main()
