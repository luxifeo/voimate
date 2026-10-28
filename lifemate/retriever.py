"""Few-shot example retrieval over the labelled train split with a multilingual embedding model (bge-m3)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import OUTPUT_ROOT

EMBED_MODEL = "BAAI/bge-m3"


def example_query_text(item: dict, final_text: str, history_texts: list[str]) -> str:
    """Text embedded for a sample: the previous user turn (for context-dependent follow-ups) and the final request."""
    prev = history_texts[-1] if history_texts else ""
    return f"{prev}\n{final_text}".strip()


class FewShotRetriever:
    def __init__(self, train_rows: list[dict], device: str = "cuda", model_name: str = EMBED_MODEL,
                 cache_dir: Path = OUTPUT_ROOT / "index"):
        from sentence_transformers import SentenceTransformer

        self.rows = train_rows
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 256
        texts = [
            example_query_text(r, r["audio_text"],
                               [t["audio_text"] for t in r["history"] if t.get("role") == "user"])
            for r in train_rows
        ]
        cache = cache_dir / f"train_{model_name.split('/')[-1]}_{len(texts)}.npy"
        if cache.exists():
            self.emb = np.load(cache)
        else:
            self.emb = self.encode(texts)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.emb)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=len(texts) > 256)

    def search(self, query_vec: np.ndarray, k: int, exclude_id: str | None = None,
               installed_apps: set[str] | None = None) -> list[dict]:
        """Top-k train rows. With installed_apps, rows whose gold calls need an app the sample lacks are skipped."""
        order = np.argsort(-(self.emb @ query_vec))
        out = []
        for i in order:
            r = self.rows[i]
            if r["id"] == exclude_id:
                continue
            if installed_apps is not None:
                apps = {c["name"].split(".")[0] for c in r["label"]["tool_calls"]} - {"sys"}
                if not apps <= installed_apps:
                    continue
            out.append(r)
            if len(out) == k:
                break
        return out


def _short(obj, limit: int) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "…"


def format_example(row: dict, result_chars: int = 220, style: str = "full") -> str:
    """Render a train row as a compact demonstration: last history exchange, request, gold calls with results.
    style "calls": only the request and the gold calls (no earlier turn, no results)."""
    lines = []
    hist = row.get("history", [])
    if len(hist) >= 2 and style == "full":
        u, a = hist[-2], hist[-1]
        lines.append(f"(Lượt trước) Người dùng: {u.get('audio_text', '')}")
        prev_calls = a.get("tool_calls") or []
        if prev_calls:
            lines.append("(Lượt trước) Đã gọi: " + "; ".join(f"{c['name']} {_short(c['arguments'], 120)}" for c in prev_calls))
        lines.append(f"(Lượt trước) Trợ lý: {a.get('text', '')}")
    lines.append(f"Người dùng: {row['audio_text']}")
    calls = row["label"]["tool_calls"]
    if not calls:
        lines.append("Lời gọi: [] (không gọi tool nào)")
    for c, res in zip(calls, row["label"]["tool_results"]):
        lines.append(f"→ {c['name']} {_short(c['arguments'], 300)}"
                     + (f"  ⇒ {_short(res, result_chars)}" if style == "full" else ""))
    return "\n".join(lines)
