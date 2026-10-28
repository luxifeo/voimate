"""Speech enhancement / vocal separation before gipformer ASR: does it lower WER? Experiment only; the production
pipeline (lifemate/*.py, scripts/predict.py, Docker) is untouched.

    # 1. enhance every user audio of a clip set -> float32 wavs under outputs/asr/enhanced/<method>/
    uv run python scripts/bench_enhance.py enhance --method gtcrn --set dev
    uv run python scripts/bench_enhance.py enhance --method dpdfnet8 --set dev
    uv run --with demucs==4.1.0 python scripts/bench_enhance.py enhance --method htdemucs --set dev --workers 4
    #    (or project torch + only demucs/einops/julius/lameenc/sphn on PYTHONPATH, for the cu126 GPU build:
    #     CUDA_VISIBLE_DEVICES=1 PYTHONPATH=<dir> uv run python ... --method htdemucs --device cuda:0 --workers 1)
    # 2. transcribe alpha*enhanced + (1-alpha)*original with the production TransducerASR (gipformer 1.5)
    #    -> outputs/asr/train_gip15_<method>[_mix<alpha%>].json  ({rel_path: text}, usable as --asr-cache)
    uv run python scripts/bench_enhance.py asr --method gtcrn --alpha 1 0.7 0.5 --set dev
    # 3. SNR-gated cache: enhanced text where the clip's estimated SNR < T dB, raw text elsewhere (no new ASR)
    uv run python scripts/bench_enhance.py gate --method gtcrn_mix50 --snr 20
    # 4. single-clip runtime, tables
    uv run python scripts/bench_enhance.py timing --method gtcrn --clips 40 --threads 8
    uv run python scripts/bench_enhance.py report > /tmp/report.md

Clip sets: dev = every user audio (history + final) of random.Random(0).sample(load_split("train"), 300);
final = the final-request audio of every train row. SNR is the same frame-energy estimate as
outputs/asr/audio_stats.json (95th minus 10th percentile of 25 ms frame energies).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lifemate.asr import _edit_distance, _words  # noqa: E402
from lifemate.data import DATA_ROOT, OUTPUT_ROOT, load_split, user_audio_paths  # noqa: E402

SPLIT = "train"
SR = 16000
ASR_DIR = OUTPUT_ROOT / "asr"
ENH_DIR = ASR_DIR / "enhanced"
MODEL_DIR = ENH_DIR / "models"
RAW_CACHE = ASR_DIR / "train_gip15.json"
SNR_FILE = ENH_DIR / "snr_user_audio.json"


# ---------------------------------------------------------------- clip sets, references, SNR
def dev_rows() -> list[dict]:
    return random.Random(0).sample(load_split(SPLIT), 300)


def clip_set(name: str) -> list[str]:
    if name == "dev":
        return [p for r in dev_rows() for p in user_audio_paths(r)]
    if name == "final":
        return [r["audio"] for r in load_split(SPLIT)]
    if name == "final_rest":  # final turns of the train rows outside dev
        dev = {r["id"] for r in dev_rows()}
        return [r["audio"] for r in load_split(SPLIT) if r["id"] not in dev]
    if name == "lowsnr_rest":  # every user audio (history + final) of non-dev rows with estimated SNR < 20 dB
        dev, snr = {r["id"] for r in dev_rows()}, snr_table()
        return [p for r in load_split(SPLIT) if r["id"] not in dev for p in user_audio_paths(r) if snr[p] < 20]
    raise ValueError(name)


def references() -> tuple[dict[str, str], set[str]]:
    """{rel_path: reference text} for every user audio of the split, and the set of final-turn paths."""
    refs, finals = {}, set()
    for r in load_split(SPLIT):
        for t in r.get("history", []):
            if t.get("role") == "user" and t.get("audio"):
                refs[t["audio"]] = t["audio_text"]
        refs[r["audio"]] = r["audio_text"]
        finals.add(r["audio"])
    return refs, finals


def estimate_snr(x: np.ndarray) -> float:
    """Same estimate as audio_stats.json: 95th - 10th percentile of 25 ms / 10 ms-hop frame energies (dB)."""
    frames = np.lib.stride_tricks.sliding_window_view(x, 400)[::160]
    db = 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-10)
    return float(np.percentile(db, 95) - np.percentile(db, 10))


def snr_table() -> dict[str, float]:
    if SNR_FILE.exists():
        return json.loads(SNR_FILE.read_text())
    refs, _ = references()
    out = {p: estimate_snr(sf.read(DATA_ROOT / SPLIT / p, dtype="float32")[0]) for p in refs}
    SNR_FILE.parent.mkdir(parents=True, exist_ok=True)
    SNR_FILE.write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------- enhancers: 16 kHz mono float32 -> same length
def _fit(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


class SherpaDenoiser:
    """GTCRN / DPDFNet through sherpa-onnx. Both are streaming models run from a cold state, which mutes the
    speech onset; '<model>_warm' first feeds the time-reversed first 2 s of the clip, then drops that part."""

    WARM = 2 * SR

    def __init__(self, method: str, threads: int):
        import sherpa_onnx as so
        base = method.removesuffix("_warm")
        self.warm = method.endswith("_warm")
        if base == "gtcrn":
            m = so.OfflineSpeechDenoiserModelConfig(
                gtcrn=so.OfflineSpeechDenoiserGtcrnModelConfig(model=str(MODEL_DIR / "gtcrn_simple.onnx")),
                num_threads=threads)
        else:  # dpdfnet8, dpdfnet2, dpdfnet_baseline
            m = so.OfflineSpeechDenoiserModelConfig(
                dpdfnet=so.OfflineSpeechDenoiserDpdfNetModelConfig(model=str(MODEL_DIR / f"{base}.onnx")),
                num_threads=threads)
        self.den = so.OfflineSpeechDenoiser(so.OfflineSpeechDenoiserConfig(model=m))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        pad = np.ascontiguousarray(x[: min(len(x), self.WARM)][::-1]) if self.warm else x[:0]
        y = self.den(np.concatenate([pad, x]), SR)
        assert y.sample_rate == SR, y.sample_rate
        return _fit(np.asarray(y.samples, dtype=np.float32)[len(pad):], len(x))


class Demucs:
    """htdemucs (Demucs v4) 'vocals' stem; deterministic (shifts=0), 16 kHz mono <-> 44.1 kHz stereo."""

    def __init__(self, method: str, threads: int, device: str = "cpu"):
        import torch
        from demucs.pretrained import get_model
        torch.set_num_threads(threads)
        self.torch, self.device = torch, device
        self.model = get_model(method).eval().to(device)
        self.vocals = self.model.sources.index("vocals")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        import julius
        from demucs.apply import apply_model
        t = self.torch
        with t.inference_mode():
            wav = julius.resample_frac(t.from_numpy(x).to(self.device), SR, self.model.samplerate)
            wav = wav[None].expand(self.model.audio_channels, -1)
            mean, std = wav.mean(), wav.std() + 1e-8
            src = apply_model(self.model, ((wav - mean) / std)[None], shifts=0, split=True, overlap=0.25,
                              progress=False, num_workers=0, device=self.device)[0]
            voc = src[self.vocals] * std + mean
            y = julius.resample_frac(voc.mean(0), self.model.samplerate, SR)
        return _fit(y.cpu().numpy(), len(x))


def make_enhancer(method: str, threads: int, device: str = "cpu"):
    if method.removesuffix("_warm") in ("gtcrn", "dpdfnet8", "dpdfnet2", "dpdfnet_baseline"):
        return SherpaDenoiser(method, threads)
    if method.startswith("htdemucs"):
        return Demucs(method, threads, device)
    raise ValueError(method)


def enhanced_path(method: str, rel: str) -> Path:
    return ENH_DIR / method / rel


_ENH = None


def _worker_init(method: str, threads: int, device: str) -> None:
    global _ENH
    _ENH = make_enhancer(method, threads, device)


def _worker_run(job: tuple[str, str]) -> tuple[str, float, float]:
    method, rel = job
    x, sr = sf.read(DATA_ROOT / SPLIT / rel, dtype="float32")
    assert sr == SR and x.ndim == 1, (rel, sr, x.shape)
    t0 = time.perf_counter()
    y = _ENH(x)
    dt = time.perf_counter() - t0
    out = enhanced_path(method, rel)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, y, SR, subtype="FLOAT")
    return rel, dt, len(x) / SR


def cmd_enhance(args) -> None:
    rels = clip_set(args.set)
    todo = [r for r in rels if not enhanced_path(args.method, r).exists()]
    timing_file = ENH_DIR / args.method / "_timing.json"
    timing = json.loads(timing_file.read_text()) if timing_file.exists() else {}
    print(f"{args.method}: {len(rels)} clips in set {args.set}, {len(todo)} to enhance", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(args.workers, _worker_init, (args.method, args.threads, args.device)) as pool:
        for i, (rel, dt, dur) in enumerate(pool.imap_unordered(_worker_run, [(args.method, r) for r in todo])):
            timing[rel] = [round(dt, 4), round(dur, 3)]
            if i % 100 == 99:
                print(f"  {i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
                timing_file.write_text(json.dumps(timing))
    timing_file.parent.mkdir(parents=True, exist_ok=True)
    timing_file.write_text(json.dumps(timing))
    print(f"done in {time.time() - t0:.0f}s ({args.workers} workers x {args.threads} threads)")


def cmd_timing(args) -> None:
    """Sequential single-clip latency in one process (what the pipeline would pay per request)."""
    rels = clip_set("dev")[: args.clips]
    enh = make_enhancer(args.method, args.threads, args.device)
    x0, _ = sf.read(DATA_ROOT / SPLIT / rels[0], dtype="float32")
    enh(x0)  # warm-up
    secs, durs = [], []
    for r in rels:
        x, _ = sf.read(DATA_ROOT / SPLIT / r, dtype="float32")
        t0 = time.perf_counter()
        enh(x)
        secs.append(time.perf_counter() - t0)
        durs.append(len(x) / SR)
    secs, durs = np.array(secs), np.array(durs)
    res = {"method": args.method, "threads": args.threads, "clips": len(rels), "mean_s": float(secs.mean()),
           "p50_s": float(np.median(secs)), "p95_s": float(np.percentile(secs, 95)), "max_s": float(secs.max()),
           "mean_audio_s": float(durs.mean()), "rtf": float(secs.sum() / durs.sum())}
    out = ENH_DIR / "timing.json"
    allres = json.loads(out.read_text()) if out.exists() else {}
    res["device"] = args.device
    if args.device.startswith("cuda"):
        import torch
        res["gpu_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    allres[f"{args.method}@{args.device}/{args.threads}t"] = res
    out.write_text(json.dumps(allres, indent=1))
    print(json.dumps(res))


# ---------------------------------------------------------------- ASR on (mixed) enhanced audio
def cache_name(method: str, alpha: float) -> str:
    return method if alpha >= 1 else f"{method}_mix{round(alpha * 100)}"


def cache_path(name: str) -> Path:
    return ASR_DIR / f"train_gip15_{name}.json"


def transcribe_arrays(asr, arrays: list[np.ndarray], batch: int = 32) -> list[str]:
    """Same decoding as lifemate.asr.TransducerASR.transcribe, from in-memory 16 kHz float32 arrays."""
    out = []
    for i in range(0, len(arrays), batch):
        streams = []
        for a in arrays[i:i + batch]:
            s = asr.recognizer.create_stream()
            s.accept_waveform(SR, a)
            streams.append(s)
        asr.recognizer.decode_streams(streams)
        out += [asr.tidy(s.result.text) for s in streams]
    return out


def cmd_asr(args) -> None:
    from lifemate.asr import TransducerASR

    asr = TransducerASR(num_threads=args.threads)
    rels = clip_set(args.set)
    raw = {r: sf.read(DATA_ROOT / SPLIT / r, dtype="float32")[0] for r in rels}
    if args.method == "raw":
        t0 = time.time()
        texts = transcribe_arrays(asr, [raw[r] for r in rels])
        ref = json.loads(RAW_CACHE.read_text(encoding="utf-8"))
        diff = [r for r, t in zip(rels, texts) if ref.get(r) != t]
        print(f"raw re-run: {len(rels)} clips in {time.time() - t0:.0f}s, {len(diff)} differ from {RAW_CACHE.name}")
        for r in diff[:5]:
            print("  ", r, "|", ref.get(r), "|", texts[rels.index(r)])
        return
    enh = {r: sf.read(enhanced_path(args.method, r), dtype="float32")[0] for r in rels}
    for alpha in args.alpha:
        name = cache_name(args.method, alpha)
        out = cache_path(name)
        cache = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        todo = [r for r in rels if r not in cache]
        t0 = time.time()
        texts = transcribe_arrays(asr, [(alpha * enh[r] + (1 - alpha) * raw[r]).astype(np.float32) for r in todo])
        cache.update(zip(todo, texts))
        out.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"{name}: {len(todo)} clips transcribed in {time.time() - t0:.0f}s -> {out} ({len(cache)} total)",
              flush=True)


def cmd_gate(args) -> None:
    """Enhanced transcript where SNR < threshold, raw transcript elsewhere (decoding is per clip, so this equals
    running the gate inside the pipeline). Covers every train user audio when the enhanced cache has every clip
    below the threshold, otherwise only the clips of the enhanced cache."""
    snr = snr_table()
    refs, _ = references()
    raw = json.loads(RAW_CACHE.read_text(encoding="utf-8"))
    enh = json.loads(cache_path(args.method).read_text(encoding="utf-8"))
    for th in args.snr:
        keys = refs if all(r in enh for r in refs if snr[r] < th) else enh
        gated = {r: (enh[r] if snr[r] < th else raw[r]) for r in keys}
        out = cache_path(f"{args.method}_gate{th:g}")
        out.write_text(json.dumps(gated, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"{out.name}: {sum(snr[r] < th for r in gated)}/{len(gated)} clips use the enhanced transcript")


# ---------------------------------------------------------------- report
def clip_errors(refs: dict[str, str], hyps: dict[str, str], rels: list[str], norm=_words) -> np.ndarray:
    """(n, 4) array of [word errors, ref words, char errors, ref chars] per clip."""
    out = np.zeros((len(rels), 4))
    for i, r in enumerate(rels):
        rw, hw = norm(refs[r]), norm(hyps[r])
        rc, hc = list(" ".join(rw)), list(" ".join(hw))
        out[i] = [_edit_distance(rw, hw), len(rw), _edit_distance(rc, hc), len(rc)]
    return out


def wer(e: np.ndarray) -> float:
    return e[:, 0].sum() / max(e[:, 1].sum(), 1)


def cer(e: np.ndarray) -> float:
    return e[:, 2].sum() / max(e[:, 3].sum(), 1)


def boot_ci(e_m: np.ndarray, e_b: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% paired-bootstrap CI (over clips) of WER(method) - WER(raw), in WER points."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e_m), (n, len(e_m)))
    d = (e_m[idx, 0].sum(1) - e_b[idx, 0].sum(1)) / e_b[idx, 1].sum(1)
    return float(np.percentile(d, 2.5) * 100), float(np.percentile(d, 97.5) * 100)


def cmd_report(args) -> None:
    from bench_asr import normalize_vi

    refs, finals = references()
    snr = snr_table()
    raw = json.loads(RAW_CACHE.read_text(encoding="utf-8"))
    names = args.methods or sorted(p.name[len("train_gip15_"):-5] for p in ASR_DIR.glob("train_gip15_*.json"))
    caches = {n: json.loads(cache_path(n).read_text(encoding="utf-8")) for n in names}
    dev = clip_set("dev")
    dev_final = [r for r in dev if r in finals]
    dev_hist = [r for r in dev if r not in finals]
    pr = print

    pr(f"## Dev 300: {len(dev_final)} final + {len(dev_hist)} history clips\n")
    pr("| method | final WER | final CER | hist WER | hist CER | all WER | all WER num-norm | "
       "ΔWER all vs raw [95% CI] |\n|---|---|---|---|---|---|---|---|")
    base = {k: clip_errors(refs, raw, v) for k, v in (("f", dev_final), ("h", dev_hist), ("a", dev))}
    base_n = clip_errors(refs, raw, dev, normalize_vi)
    rows = [("raw", raw)] + [(n, c) for n, c in caches.items() if all(r in c for r in dev)]
    errs = {}
    for name, hyp in rows:
        e = {k: clip_errors(refs, hyp, v) for k, v in (("f", dev_final), ("h", dev_hist), ("a", dev))}
        errs[name] = e
        en = clip_errors(refs, hyp, dev, normalize_vi)
        lo, hi = boot_ci(e["a"], base["a"]) if name != "raw" else (0, 0)
        delta = (wer(e["a"]) - wer(base["a"])) * 100
        pr(f"| {name} | {wer(e['f']):.2%} | {cer(e['f']):.2%} | {wer(e['h']):.2%} | {cer(e['h']):.2%} | "
           f"{wer(e['a']):.2%} | {wer(en):.2%} | "
           + ("—" if name == "raw" else f"{delta:+.2f} [{lo:+.2f}, {hi:+.2f}]") + " |")
    del base_n

    buckets = [(-99, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 999)]
    pr("\n## Dev 300 WER by estimated SNR bucket (final + history clips)\n")
    pr("| method | " + " | ".join(f"{'<' + str(hi) if lo < 0 else (str(lo) + '+' if hi > 900 else f'{lo}–{hi}')} dB"
                                   for lo, hi in buckets) + " |")
    pr("|---|" + "---|" * len(buckets))
    s = np.array([snr[r] for r in dev])
    masks = [(s >= lo) & (s < hi) for lo, hi in buckets]
    pr("| clips | " + " | ".join(str(int(m.sum())) for m in masks) + " |")
    for name in errs:
        e = errs[name]["a"]
        pr(f"| {name} | " + " | ".join(f"{wer(e[m]):.1%}" for m in masks) + " |")

    gates = (10, 15, 20, 25, 30)
    pr("\n## Dev 300 SNR gate sweep: enhanced transcript if SNR < T dB, raw otherwise (all WER, final + history)\n")
    pr("| method | " + " | ".join(f"T={t} (n={int((s < t).sum())})" for t in gates) + " |")
    pr("|---|" + "---|" * len(gates))
    pr(f"| raw | " + " | ".join(f"{wer(errs['raw']['a']):.2%}" for _ in gates) + " |")
    for name in errs:
        if name == "raw" or "_gate" in name:
            continue
        cells = []
        for t in gates:
            g = np.where((s < t)[:, None], errs[name]["a"], errs["raw"]["a"])
            cells.append(f"{wer(g):.2%}")
        pr(f"| {name} | " + " | ".join(cells) + " |")

    # held-out check on the final turns of the 1245 non-dev rows, for caches that cover them
    rest = clip_set("final_rest")
    full = [(n, c) for n, c in caches.items() if all(r in c for r in rest)]
    if full:
        pr(f"\n## Held-out: final turns of the {len(rest)} train rows outside dev 300\n")
        pr("| method | WER | CER | ΔWER vs raw [95% CI] |\n|---|---|---|---|")
        b = clip_errors(refs, raw, rest)
        pr(f"| raw | {wer(b):.2%} | {cer(b):.2%} | — |")
        for name, hyp in full:
            e = clip_errors(refs, hyp, rest)
            lo, hi = boot_ci(e, b)
            pr(f"| {name} | {wer(e):.2%} | {cer(e):.2%} | {(wer(e) - wer(b)) * 100:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
        allf = [r["audio"] for r in load_split(SPLIT)]
        pr(f"\n## All {len(allf)} train final turns\n")
        pr("| method | WER | CER |\n|---|---|---|")
        b = clip_errors(refs, raw, allf)
        pr(f"| raw | {wer(b):.2%} | {cer(b):.2%} |")
        for name, hyp in full:
            e = clip_errors(refs, hyp, allf)
            pr(f"| {name} | {wer(e):.2%} | {cer(e):.2%} |")

    low = clip_set("lowsnr_rest")
    held = [(n, c) for n, c in caches.items() if all(r in c for r in low)]
    if held:
        sl = np.array([snr[r] for r in low])
        b = clip_errors(refs, raw, low)
        pr(f"\n## Held-out low-SNR check: user audios (final + history) of the non-dev rows with SNR < 20 dB\n")
        pr("| method | WER SNR<15 | ΔWER SNR<15 [95% CI] | WER SNR<20 | ΔWER SNR<20 [95% CI] |\n|---|---|---|---|---|")
        pr(f"| raw (n={int((sl < 15).sum())} / {len(low)}) | {wer(b[sl < 15]):.2%} | — | {wer(b):.2%} | — |")
        for name, hyp in held:
            e = clip_errors(refs, hyp, low)
            m = sl < 15
            lo15, hi15 = boot_ci(e[m], b[m])
            lo20, hi20 = boot_ci(e, b)
            pr(f"| {name} | {wer(e[m]):.2%} | {(wer(e[m]) - wer(b[m])) * 100:+.2f} [{lo15:+.2f}, {hi15:+.2f}] | "
               f"{wer(e):.2%} | {(wer(e) - wer(b)) * 100:+.2f} [{lo20:+.2f}, {hi20:+.2f}] |")
        pooled = low + [r for r in dev if snr[r] < 20]  # every train user audio with SNR < 20 dB
        sp = np.array([snr[r] for r in pooled])
        bp = clip_errors(refs, raw, pooled)
        pr(f"\n### Pooled dev + held-out: every train user audio with SNR < 20 dB\n")
        pr("| method | WER SNR<15 | ΔWER SNR<15 [95% CI] | WER SNR<20 | ΔWER SNR<20 [95% CI] |\n|---|---|---|---|---|")
        pr(f"| raw (n={int((sp < 15).sum())} / {len(pooled)}) | {wer(bp[sp < 15]):.2%} | — | {wer(bp):.2%} | — |")
        for name, hyp in held:
            if not all(r in hyp for r in pooled):
                continue
            e = clip_errors(refs, hyp, pooled)
            m = sp < 15
            lo15, hi15 = boot_ci(e[m], bp[m])
            lo20, hi20 = boot_ci(e, bp)
            pr(f"| {name} | {wer(e[m]):.2%} | {(wer(e[m]) - wer(bp[m])) * 100:+.2f} [{lo15:+.2f}, {hi15:+.2f}] | "
               f"{wer(e):.2%} | {(wer(e) - wer(bp)) * 100:+.2f} [{lo20:+.2f}, {hi20:+.2f}] |")

    every = list(refs)
    full = [(n, c) for n, c in caches.items() if all(r in c for r in every)]
    if full:
        nf = sum(r in finals for r in every)
        pr(f"\n## Every train user audio ({nf} final + {len(every) - nf} history; dev 300 rows included)\n")
        pr("| method | final WER | hist WER | all WER | all CER | ΔWER all vs raw [95% CI] |\n|---|---|---|---|---|---|")
        isf = np.array([r in finals for r in every])
        ba = clip_errors(refs, raw, every)
        bf, bh = ba[isf], ba[~isf]
        pr(f"| raw | {wer(bf):.2%} | {wer(bh):.2%} | {wer(ba):.2%} | {cer(ba):.2%} | — |")
        for name, hyp in full:
            ea = clip_errors(refs, hyp, every)
            ef, eh = ea[isf], ea[~isf]
            lo, hi = boot_ci(ea, ba)
            pr(f"| {name} | {wer(ef):.2%} | {wer(eh):.2%} | {wer(ea):.2%} | {cer(ea):.2%} | "
               f"{(wer(ea) - wer(ba)) * 100:+.3f} [{lo:+.3f}, {hi:+.3f}] |")

    if args.examples:
        hyp = caches[args.examples]
        e_m, e_b = errs[args.examples]["a"], errs["raw"]["a"]
        gain = e_b[:, 0] - e_m[:, 0]
        order = np.argsort(-gain, kind="stable")
        pr(f"\n## Examples for {args.examples} (word errors raw -> enhanced)\n")
        for title, idx in (("helped", order[:5]), ("hurt", order[::-1][:5])):
            pr(f"### {title}\n")
            for i in idx:
                r = dev[i]
                pr(f"- `{r}` SNR {snr[r]:.1f} dB, errors {int(e_b[i, 0])} -> {int(e_m[i, 0])}\n"
                   f"  - ref: {refs[r]}\n  - raw: {raw[r]}\n  - enh: {hyp[r]}")
            pr("")
        pr(f"clips helped {int((gain > 0).sum())}, hurt {int((gain < 0).sum())}, unchanged {int((gain == 0).sum())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enhance")
    e.add_argument("--method", required=True)
    e.add_argument("--set", default="dev", choices=["dev", "final", "final_rest", "lowsnr_rest"])
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--threads", type=int, default=1)
    e.add_argument("--device", default="cpu", help="htdemucs only, e.g. cuda:0 (with CUDA_VISIBLE_DEVICES)")
    t = sub.add_parser("timing")
    t.add_argument("--method", required=True)
    t.add_argument("--clips", type=int, default=40)
    t.add_argument("--threads", type=int, default=8)
    t.add_argument("--device", default="cpu")
    a = sub.add_parser("asr")
    a.add_argument("--method", required=True)
    a.add_argument("--alpha", type=float, nargs="+", default=[1.0])
    a.add_argument("--set", default="dev", choices=["dev", "final", "final_rest", "lowsnr_rest"])
    a.add_argument("--threads", type=int, default=8)
    g = sub.add_parser("gate")
    g.add_argument("--method", required=True)
    g.add_argument("--snr", type=float, nargs="+", required=True)
    r = sub.add_parser("report")
    r.add_argument("--methods", nargs="*")
    r.add_argument("--examples", default="")
    args = ap.parse_args()
    {"enhance": cmd_enhance, "timing": cmd_timing, "asr": cmd_asr, "gate": cmd_gate, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
