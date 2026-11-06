"""Does band-limiting the audio (e.g. down-sampling to 8 kHz) before gipformer ASR lower WER? Experiment only; the
production pipeline is untouched.

    uv run python scripts/bench_bandwidth.py run                       # dev clip set, every METHODS entry
    uv run python scripts/bench_bandwidth.py run --set rest --methods sr8k   # confirm on the non-dev train clips
    uv run python scripts/bench_bandwidth.py report [--set rest]

Clip sets: dev = bench_enhance "dev" (every user audio of random.Random(0).sample(train, 300)); rest = every user audio
of the other train rows. Audio is decoded and transformed by ffmpeg, then transcribed with the production TransducerASR
recognizer (batch 32). Transcripts go to outputs/asr/train_gip15_bw_<method>[_rest].json ({rel_path: text}); the
production cache train_gip15.json serves as "raw" for the rest set.
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

from bench_enhance import clip_set, dev_rows, references, snr_table  # noqa: E402
from bench_tempo import errors, wer  # noqa: E402
from lifemate.asr import GIPFORMER, TransducerASR  # noqa: E402
from lifemate.data import DATA_ROOT, OUTPUT_ROOT, load_split, user_audio_paths  # noqa: E402

SR = 16000
METHODS = {  # name: (ffmpeg -af chain applied after resampling to 16 kHz, sample rate handed to sherpa-onnx)
    "raw": ("", SR),
    "sr6k": ("aresample=6000,aresample=16000", SR),  # band limit 3 kHz
    "sr8k": ("aresample=8000,aresample=16000", SR),  # band limit 4 kHz, ffmpeg swr
    "sr8k_soxr": ("aresample=8000:resampler=soxr,aresample=16000:resampler=soxr", SR),
    "feed8k": ("aresample=8000", 8000),  # 8 kHz samples, sherpa-onnx resamples to 16 kHz itself
    "tel": ("aresample=8000,highpass=f=300,lowpass=f=3400,aresample=16000", SR),  # telephone band 300-3400 Hz
    "sr11k": ("aresample=11025,aresample=16000", SR),  # band limit 5.5 kHz
    "sr12k": ("aresample=12000,aresample=16000", SR),  # band limit 6 kHz
}


def clips_for(name: str) -> list[str]:
    if name == "rest":
        dev = {r["id"] for r in dev_rows()}
        return [p for r in load_split("train") if r["id"] not in dev for p in user_audio_paths(r)]
    return clip_set(name)


def cache_file(method: str, set_name: str, provider: str = "cpu") -> Path:
    if method == "raw" and set_name == "rest" and provider == "cpu":
        return OUTPUT_ROOT / "asr" / "train_gip15.json"
    tag = ("" if set_name == "dev" else "_" + set_name) + ("" if provider == "cpu" else "_" + provider)
    return OUTPUT_ROOT / "asr" / f"train_gip15_bw_{method}{tag}.json"


def make_asr(provider: str, threads: int) -> TransducerASR:
    """Production recognizer; provider="cuda" rebuilds it on the GPU (needs a CUDA build of sherpa-onnx)."""
    asr = TransducerASR(num_threads=threads)
    if provider != "cpu":
        import sherpa_onnx
        from huggingface_hub import snapshot_download

        d = Path(snapshot_download(GIPFORMER, allow_patterns=["*.onnx", "tokens.txt", "config.json"]))
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))["onnx"]
        asr.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / cfg["encoder"]), decoder=str(d / cfg["decoder"]), joiner=str(d / cfg["joiner"]),
            tokens=str(d / "tokens.txt"), num_threads=threads, decoding_method="greedy_search", provider=provider)
    return asr


def load_audio(path: Path, chain: str) -> np.ndarray:
    af = f"aresample={SR}" + (f",{chain}" if chain else "")
    out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-af", af, "-ac", "1",
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def run(methods: list[str], set_name: str, threads: int, provider: str) -> None:
    clips = clips_for(set_name)
    asr = make_asr(provider, threads)
    for m in methods:
        if cache_file(m, set_name, provider).exists():
            print(m, "cached")
            continue
        chain, sr = METHODS[m]
        texts = []
        for i in range(0, len(clips), 32):
            with ThreadPoolExecutor(8) as pool:
                audio = list(pool.map(lambda p: load_audio(DATA_ROOT / "train" / p, chain), clips[i:i + 32]))
            streams = []
            for a in audio:
                s = asr.recognizer.create_stream()
                s.accept_waveform(sr, a)
                streams.append(s)
            asr.recognizer.decode_streams(streams)
            texts += [asr.tidy(s.result.text) for s in streams]
        cache_file(m, set_name, provider).write_text(json.dumps(dict(zip(clips, texts)), ensure_ascii=False,
                                                                indent=0), encoding="utf-8")
        print(m, len(clips), "clips", flush=True)


def delta_ci(e: dict, base: dict, clips: list[str], n_boot: int = 2000) -> tuple[float, float]:
    """95% paired bootstrap CI (over clips) of WER(method) - WER(raw), in WER points."""
    d = np.array([e[p][0] - base[p][0] for p in clips], dtype=np.float64)
    n = np.array([base[p][1] for p in clips], dtype=np.float64)
    idx = np.random.default_rng(0).integers(0, len(clips), (n_boot, len(clips)))
    boot = 100 * d[idx].sum(1) / n[idx].sum(1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def report(set_name: str, provider: str) -> None:
    clips = clips_for(set_name)
    refs, finals = references()
    snr = snr_table()
    hyps = {m: json.loads(cache_file(m, set_name, provider).read_text(encoding="utf-8"))
            for m in METHODS if cache_file(m, set_name, provider).exists()}
    err = {m: {p: errors(refs[p], h[p]) for p in clips} for m, h in hyps.items()}
    if set_name == "dev" or provider != "cpu":
        prod = json.loads((OUTPUT_ROOT / "asr" / "train_gip15.json").read_text(encoding="utf-8"))
        print(f"raw transcripts identical to production cache: {sum(hyps['raw'][p] == prod[p] for p in clips)}"
              f"/{len(clips)}\n")
    print(f"{set_name}: {len(clips)} clips\n")
    print("| method | WER all | final | history | delta vs raw [95% CI] | clips better / worse |\n|---|---|---|---|---|---|")
    for m in hyps:
        e, b = err[m], err["raw"]
        lo, hi = delta_ci(e, b, clips)
        up = sum(e[p][0] < b[p][0] for p in clips)
        down = sum(e[p][0] > b[p][0] for p in clips)
        print(f"| {m} | {wer(e.values()):.2f} | {wer([e[p] for p in clips if p in finals]):.2f} | "
              f"{wer([e[p] for p in clips if p not in finals]):.2f} | "
              f"{wer(e.values()) - wer(b.values()):+.2f} [{lo:+.2f}, {hi:+.2f}] | {up} / {down} |")

    bins = [("<15 dB", lambda s: s < 15), ("15-30 dB", lambda s: 15 <= s < 30), (">=30 dB", lambda s: s >= 30)]
    print("\nWER by estimated SNR\n")
    print("| method | " + " | ".join(f"{b} (n={sum(f(snr[p]) for p in clips)})" for b, f in bins)
          + " |\n|---|" + "---|" * len(bins))
    for m in hyps:
        print(f"| {m} | " + " | ".join(f"{wer([err[m][p] for p in clips if f(snr[p])]):.2f}" for _, f in bins) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--set", default="dev", choices=["dev", "rest"])
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.methods, args.set, args.threads, args.provider)
    else:
        report(args.set, args.provider)


if __name__ == "__main__":
    main()
