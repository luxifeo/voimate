"""Benchmark Vietnamese ASR models on the final-turn audios of a labelled split.

    uv run --with sherpa-onnx python scripts/bench_asr.py run --backend sherpa --model g-group-ai-lab/gipformer1.5-68M-rnnt --name gipformer15
    uv run python scripts/bench_asr.py run --backend whisper --model vinai/PhoWhisper-medium --name phowhisper-medium
    uv run python scripts/bench_asr.py table      # every cache under outputs/asr/bench + the Gemma caches

Transcripts go to outputs/asr/bench/<name>_<split>.json ({audio rel path: text, "_meta": {...}}).
`table` scores them against audio_text on the clips every model has, with two normalisations:
  basic    lower-case, punctuation removed (same as lifemate.asr.error_rates)
  num      basic + tone-mark style unified + numbers written as Vietnamese words on both sides
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.asr import _edit_distance, _words  # noqa: E402
from lifemate.data import DATA_ROOT, OUTPUT_ROOT, load_split  # noqa: E402

BENCH_DIR = OUTPUT_ROOT / "asr" / "bench"

# ---------------------------------------------------------------- number normalisation (scoring only)
DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
TONE_STYLE = {"oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa", "oè": "òe", "oé": "óe", "oẻ": "ỏe",
              "oẽ": "õe", "oẹ": "ọe", "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy"}
# spoken variants mapped to one form on both sides (symmetric, so WER stays fair)
VARIANTS = {"ngàn": "nghìn", "tư": "bốn", "lẻ": "linh", "lăm": "năm", "mốt": "một", "rưởi": "rưỡi"}


def _read_under_1000(n: int, full: bool) -> list[str]:
    h, t, u = n // 100, n // 10 % 10, n % 10
    out = []
    if full or h:
        out += [DIGITS[h], "trăm"]
    if t == 0:
        if u:
            out += (["linh"] if out else []) + [DIGITS[u]]
    elif t == 1:
        out += ["mười"] + ([DIGITS[u]] if u else [])
    else:
        out += [DIGITS[t], "mươi"] + ([DIGITS[u]] if u else [])
    return out


def read_number(n: int) -> list[str]:
    if n == 0:
        return ["không"]
    out, scales = [], [(10**9, "tỷ"), (10**6, "triệu"), (10**3, "nghìn")]
    started = False
    for size, word in scales:
        if n >= size:
            out += read_number(n // size) if n // size >= 1000 else _read_under_1000(n // size, started)
            out.append(word)
            n %= size
            started = True
    if n:
        out += _read_under_1000(n, started)
    return out


def _spell(match: re.Match) -> str:
    num, unit = match.group(1).rstrip(".,:"), (match.group(3) or "").lower()
    words: list[str] = []
    if not re.fullmatch(r"\d+(:\d+)?|\d{1,3}(\.\d{3})*(,\d+)?|\d+[.,]\d+", num):  # dates, versions, ...: each digit run
        return " " + " ".join(w for part in re.findall(r"\d+", num) for w in read_number(int(part))) + " "
    if re.fullmatch(r"0\d+|\d{7,}", num):  # phone numbers, codes: digit by digit
        words = [DIGITS[int(c)] for c in num]
    elif ":" in num:  # 7:30
        hh, mm = num.split(":")[:2]
        words = read_number(int(hh)) + ["giờ"] + (read_number(int(mm)) if int(mm) else [])
    else:
        intpart, _, dec = re.sub(r"\.(?=\d{3}\b)", "", num).partition(",")
        if not dec and "." in intpart:
            intpart, _, dec = intpart.partition(".")
        words = read_number(int(intpart)) if intpart else []
        if dec:
            words += ["phẩy"] + read_number(int(dec))
    words += {"h": ["giờ"], "%": ["phần", "trăm"], "đ": ["đồng"], "k": ["nghìn"], "tr": ["triệu"]}.get(unit, [])
    return " " + " ".join(words) + " "


def normalize_vi(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"(\d)h(\d)", r"\1:\2", text)
    text = re.sub(r"(\d[\d.,:]*)(\s*)(h|%|đ|k|tr)?(?![^\W\d_])", _spell, text)
    for a, b in TONE_STYLE.items():
        text = re.sub(a + r"(?![^\W\d_])", b, text)
    return [VARIANTS.get(w, w) for w in _words(text)]


def rates(pairs: list[tuple[str, str]], norm) -> tuple[float, float]:
    we = wt = ce = ct = 0
    for ref, hyp in pairs:
        r, h = norm(ref), norm(hyp)
        we += _edit_distance(r, h)
        wt += len(r)
        rc, hc = list(" ".join(r)), list(" ".join(h))
        ce += _edit_distance(rc, hc)
        ct += len(rc)
    return we / max(wt, 1), ce / max(ct, 1)


# ---------------------------------------------------------------- backends: list of wav paths -> list of texts
def run_sherpa(model: str, wavs: list[Path], device: str) -> list[str]:
    import sherpa_onnx
    import soundfile as sf
    from huggingface_hub import snapshot_download

    d = Path(snapshot_download(model))

    def pick(part: str) -> str:
        return str(next(p for p in sorted(d.glob(f"{part}*.onnx")) if ".int8." not in p.name))

    tokens = d / "tokens.txt"
    if not tokens.exists():  # hynt/Zipformer ships bpe.model + config only: rebuild tokens.txt from the bpe model
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(d / "bpe.model"))
        tokens = BENCH_DIR / f"{model.replace('/', '__')}_tokens.txt"
        tokens.write_text("".join(f"{sp.id_to_piece(i)} {i}\n" for i in range(sp.get_piece_size())), encoding="utf-8")
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=pick("encoder"), decoder=pick("decoder"), joiner=pick("joiner"), tokens=str(tokens),
        num_threads=8, decoding_method="greedy_search", provider="cuda" if device.startswith("cuda") else "cpu")
    out = []
    for i in range(0, len(wavs), 32):
        streams = []
        for w in wavs[i:i + 32]:
            audio, sr = sf.read(w, dtype="float32")
            s = rec.create_stream()
            s.accept_waveform(sr, audio)
            streams.append(s)
        rec.decode_streams(streams)
        out += [s.result.text for s in streams]
    return out


def run_whisper(model: str, wavs: list[Path], device: str) -> list[str]:
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(model)
    m = WhisperForConditionalGeneration.from_pretrained(model, torch_dtype=torch.float16).to(device).eval()
    out = []
    for i in range(0, len(wavs), 16):
        audios = [sf.read(w, dtype="float32")[0] for w in wavs[i:i + 16]]
        feats = proc(audios, sampling_rate=16000, return_tensors="pt").input_features.to(device, torch.float16)
        with torch.no_grad():
            ids = m.generate(feats, language="vi", task="transcribe", num_beams=1, do_sample=False, max_new_tokens=220)
        out += [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
    return out


def run_qwen3asr(model: str, wavs: list[Path], device: str) -> list[str]:
    import torch
    from qwen_asr import Qwen3ASRModel

    m = Qwen3ASRModel.from_pretrained(model, dtype=torch.bfloat16, device_map=device, max_new_tokens=256,
                                      max_inference_batch_size=32)
    out = []
    for i in range(0, len(wavs), 32):
        res = m.transcribe(audio=[str(w) for w in wavs[i:i + 32]], language="Vietnamese")
        out += [r.text for r in res]
    return out


def run_nemo(model: str, wavs: list[Path], device: str) -> list[str]:
    import nemo.collections.asr as nemo_asr
    from huggingface_hub import snapshot_download

    # the repo's .nemo file name differs from the repo name, so from_pretrained misses it: restore the file directly
    nemo_file = next(Path(snapshot_download(model, allow_patterns=["*.nemo"])).glob("*.nemo"))
    m = nemo_asr.models.ASRModel.restore_from(str(nemo_file), map_location=device).eval()
    res = m.transcribe([str(w) for w in wavs], batch_size=32)
    return [r.text if hasattr(r, "text") else str(r) for r in res]


BACKENDS = {"sherpa": run_sherpa, "whisper": run_whisper, "qwen3asr": run_qwen3asr, "nemo": run_nemo}


def cmd_run(args) -> None:
    rows = load_split(args.split)
    if args.limit:
        rows = rows[: args.limit]
    rels = [r["audio"] for r in rows]
    wavs = [DATA_ROOT / args.split / p for p in rels]
    t0 = time.time()
    texts = BACKENDS[args.backend](args.model, wavs, args.device)
    elapsed = time.time() - t0
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    cache = {p: " ".join(t.split()) for p, t in zip(rels, texts)}
    cache["_meta"] = {"backend": args.backend, "model": args.model, "device": args.device, "clips": len(rels),
                      "seconds": round(elapsed, 1)}
    out = BENCH_DIR / f"{args.name}_{args.split}.json"
    out.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    pairs = [(r["audio_text"], cache[r["audio"]]) for r in rows]
    print(f"{args.name}: {len(rels)} clips in {elapsed:.0f}s; WER basic {rates(pairs, _words)[0]:.1%}, "
          f"num {rates(pairs, normalize_vi)[0]:.1%} -> {out}")


def cmd_table(args) -> None:
    rows = {r["audio"]: r["audio_text"] for r in load_split(args.split)}
    caches = {"gemma-4-e2b (prompt gốc)": OUTPUT_ROOT / "asr" / f"{args.split}.json",
              "gemma-4-e2b (prompt ngữ cảnh)": OUTPUT_ROOT / "asr" / f"{args.split}_ctx.json"}
    caches.update({p.name.removesuffix(f"_{args.split}.json"): p for p in sorted(BENCH_DIR.glob(f"*_{args.split}.json"))})
    loaded = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in caches.items() if p.exists()}
    common = sorted(set(rows).intersection(*[set(c) for c in loaded.values()]))
    print(f"{len(common)} final-turn clips scored by every model\n")
    print(f"| model | WER basic | CER basic | WER num-norm | CER num-norm | sec/clip |\n|---|---|---|---|---|---|")
    for name, cache in sorted(loaded.items(), key=lambda kv: rates([(rows[p], kv[1][p]) for p in common], normalize_vi)[0]):
        pairs = [(rows[p], cache[p]) for p in common]
        (wb, cb), (wn, cn) = rates(pairs, _words), rates(pairs, normalize_vi)
        meta = cache.get("_meta", {})
        spc = f"{meta['seconds'] / meta['clips']:.3f}" if meta else "n/a"
        print(f"| {name} | {wb:.1%} | {cb:.1%} | {wn:.1%} | {cn:.1%} | {spc} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend", choices=list(BACKENDS), required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--split", default="train")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--device", default="cuda:0")
    t = sub.add_parser("table")
    t.add_argument("--split", default="train")
    args = ap.parse_args()
    {"run": cmd_run, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    main()
