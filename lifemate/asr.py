"""Speech-to-text with Gemma-4-E2B-it through the vLLM OpenAI-compatible API."""
from __future__ import annotations

import base64
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# Prompt recommended by the Gemma 4 model card for ASR; the audio goes after the text.
ASR_PROMPT = (
    "Transcribe the following speech segment in Vietnamese into Vietnamese text.\n\n"
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."
)


def context_prompt(app_names: list[str]) -> str:
    """ASR prompt with the sample's installed apps as domain context (about -1.8 WER points on a 300-clip train subset)."""
    return (ASR_PROMPT + "\n\nContext: the speaker is giving a spoken command to a Vietnamese voice assistant "
            f"(trợ lý ảo) that controls these apps: {', '.join(app_names)}. "
            "The speech may contain common English words such as book, AC, app.")


def transcribe(client: OpenAI, model: str, wav_path: Path, prompt: str = ASR_PROMPT, max_tokens: int = 256) -> str:
    audio_b64 = base64.b64encode(wav_path.read_bytes()).decode()
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
            ],
        }],
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return " ".join((resp.choices[0].message.content or "").split())


def transcribe_all(client: OpenAI, model: str, split_dir: Path, prompts: dict[str, str], cache_file: Path,
                   workers: int = 32) -> dict[str, str]:
    """Transcribe every relative audio path of {rel_path: prompt}, reusing and extending a JSON cache {rel_path: text}."""
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    todo = [p for p in prompts if p not in cache]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if todo:
        with ThreadPoolExecutor(workers) as pool:
            futures = {p: pool.submit(transcribe, client, model, split_dir / p, prompts[p]) for p in todo}
            for i, (p, fut) in enumerate(tqdm(futures.items(), total=len(futures), desc="asr")):
                cache[p] = fut.result()
                if i % 500 == 499:
                    cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
        cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    return cache


GIPFORMER = "g-group-ai-lab/gipformer1.5-68M-rnnt"


class TransducerASR:
    """Vietnamese Zipformer-transducer ASR (gipformer 1.5, 68M params, MIT) run on CPU with sherpa-onnx.
    Raw output is upper-case words without punctuation, numbers spelled out; `tidy` turns it into sentence case."""

    def __init__(self, repo: str = GIPFORMER, num_threads: int = 8):
        import sherpa_onnx
        from huggingface_hub import snapshot_download

        d = Path(snapshot_download(repo, allow_patterns=["*.onnx", "tokens.txt", "config.json"]))
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))["onnx"]
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / cfg["encoder"]), decoder=str(d / cfg["decoder"]), joiner=str(d / cfg["joiner"]),
            tokens=str(d / "tokens.txt"), num_threads=num_threads, decoding_method="greedy_search")

    @staticmethod
    def tidy(text: str) -> str:
        text = " ".join(text.lower().split())
        return text[:1].upper() + text[1:]

    def transcribe(self, wav_paths: list[Path], batch: int = 32) -> list[str]:
        import soundfile as sf

        out = []
        for i in range(0, len(wav_paths), batch):
            streams = []
            for p in wav_paths[i:i + batch]:
                audio, sr = sf.read(p, dtype="float32")
                s = self.recognizer.create_stream()
                s.accept_waveform(sr, audio)
                streams.append(s)
            self.recognizer.decode_streams(streams)
            out += [self.tidy(s.result.text) for s in streams]
        return out


def _words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    return re.findall(r"[^\W_]+", text)


def _edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def error_rates(pairs: list[tuple[str, str]]) -> dict[str, float]:
    """Corpus WER and CER (punctuation removed, lower-cased) over (reference, hypothesis) pairs."""
    wer_err = wer_tot = cer_err = cer_tot = 0
    for ref, hyp in pairs:
        rw, hw = _words(ref), _words(hyp)
        wer_err += _edit_distance(rw, hw)
        wer_tot += len(rw)
        rc, hc = list(" ".join(rw)), list(" ".join(hw))
        cer_err += _edit_distance(rc, hc)
        cer_tot += len(rc)
    return {"wer": wer_err / max(wer_tot, 1), "cer": cer_err / max(cer_tot, 1), "n": len(pairs)}
