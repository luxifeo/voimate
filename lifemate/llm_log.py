"""Append-only JSONL log of the agent's LLM calls: one line per chat.completions request with the exact wire-format
body sent (messages, tools, sampling params), the raw response (content, reasoning, tool calls, usage) and the latency.
Read by the LLM lab app (lab/app.py), which can replay any logged request with edited parameters."""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

_DATA_URL = re.compile(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=]+")


class LLMCallLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        """Serialize now (the caller keeps mutating its message list); base64 audio is replaced by a placeholder."""
        line = json.dumps(record, ensure_ascii=False, default=str)
        line = _DATA_URL.sub(lambda m: f"<base64 omitted, {len(m.group())} chars>", line)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
