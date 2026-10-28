"""Split loading, tool catalog access and the per-split zaic_tools import."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "train+public_test"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def load_split(split: str, root: Path = DATA_ROOT) -> list[dict]:
    with open(root / split / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_catalog(split: str, root: Path = DATA_ROOT) -> dict[str, dict]:
    """{tool_name: tool spec} for every tool of the split (sys.ask_missing_info included)."""
    with open(root / split / f"{split}_tools_catalog.json", encoding="utf-8") as f:
        catalog = json.load(f)
    return {tool["name"]: tool for app in catalog["apps"] for tool in app["tools"]}


def sample_tools(item: dict) -> list[str]:
    """Tools the sample may call: its installed apps plus sys.ask_missing_info."""
    names = [t for app in item["context"]["installed_apps"].values() for t in app["tools"]]
    return names + ["sys.ask_missing_info"]


def user_audio_paths(item: dict) -> list[str]:
    """Relative paths of every user audio of a sample (history turns first, final request last)."""
    paths = [turn["audio"] for turn in item.get("history", []) if turn.get("role") == "user" and turn.get("audio")]
    return paths + [item["audio"]]


_ZAIC = {}


def zaic_tools(split: str, root: Path = DATA_ROOT):
    """Import the split's own zaic_tools package. Each split ships its own version under the same module
    name, so one process can only work with one split."""
    if split in _ZAIC:
        return _ZAIC[split]
    if _ZAIC:
        raise RuntimeError(f"zaic_tools already imported for {list(_ZAIC)}; use one split per process")
    sys.path.insert(0, str(root / split))
    module = importlib.import_module("zaic_tools")
    _ZAIC[split] = module
    return module


def new_env(split: str, sample_id: str, root: Path = DATA_ROOT):
    return zaic_tools(split, root).ToolEnv(split, sample_id, root=root)
