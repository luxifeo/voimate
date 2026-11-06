"""Tool selector sub-agent: one LLM call that reads the last user request, the earlier turns and the installed apps
with their tools, and returns the apps and tools needed to handle the request. Experiment only (scripts/eval_tool_selector.py);
the agent and the submission path do not use it."""
from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from .agent import _dump, chat
from .llm_log import LLMCallLog

SELECT_RULES = """You select tools for LifeMate, a Vietnamese voice assistant that acts on the user's apps by calling tools.
The user's requests were transcribed from Vietnamese speech and may contain small recognition errors.

Given the installed apps and their tools, the earlier turns of the conversation and the LAST user request, list every
tool needed to fully handle the LAST request:
1. Earlier turns are context only (what "it", "that one", "the other one" refer to); their tool calls were already executed.
2. Include the read tools needed before a write to look up IDs or facts, e.g. contacts.search before sms.send to a
   person, calendar.list_events before calendar.update_event, bill_payment.list_bills before bill_payment.pay.
3. A request with several parts needs the tools of every part, including conditional parts
   (e.g. "if it rains tomorrow, move my run" needs the weather forecast and the calendar tools).
4. If a required detail is missing and the assistant would have to ask for it, still list the tool that would perform the action.
5. If no installed app can handle the request, or it is small talk, return empty lists.
6. Use only the exact app ids and tool names listed below. "apps" are the app ids of the selected tools.

Answer with JSON only: {"apps": [...], "tools": [...]}"""


@dataclass
class SelectorConfig:
    model: str = "qwen3.5-4b"
    thinking: bool = False
    max_tokens: int = 2048
    result_chars: int = 300  # earlier tool results are cut to this many characters


def catalog_block(item: dict, raw_apps: dict[str, dict], catalog: dict[str, dict]) -> str:
    """Installed apps with their tools: kind, description and parameter names (required ones marked with *)."""
    lines = []
    for app_id, app in item["context"]["installed_apps"].items():
        meta = raw_apps.get(app_id, {})
        lines.append(f"- {app_id} ({meta.get('name_vi', app_id)}): {meta.get('description', '')}")
        for name in app["tools"]:
            spec = catalog[name]
            params = spec.get("parameters") or {}
            req = set(params.get("required", []))
            args = ", ".join(k + ("*" if k in req else "") for k in params.get("properties", {}))
            lines.append(f"    - {name} [{spec.get('kind', '')}]: {spec.get('description', '')} ({args})")
    return "\n".join(lines)


def history_block(item: dict, transcripts: dict[str, str], result_chars: int) -> str:
    lines = []
    for turn in item.get("history", []):
        if turn["role"] == "user":
            lines.append(f"User: {transcripts.get(turn.get('audio'), turn.get('audio_text', ''))}")
            continue
        results = turn.get("tool_results") or []
        for i, c in enumerate(turn.get("tool_calls") or []):
            res = results[i] if i < len(results) else {"ok": True}
            lines.append(f"  -> {c['name']} {json.dumps(c.get('arguments') or {}, ensure_ascii=False)} "
                         f"=> {_dump(res, result_chars)}")
        if turn.get("text"):
            lines.append(f"Assistant: {turn['text']}")
    return "\n".join(lines) or "(none)"


def format_selection_example(row: dict) -> str:
    """A train row as a demonstration: the previous user turn, the request and the tools its gold calls use."""
    hist = [t for t in row.get("history", []) if t["role"] == "user"]
    prev = f"Previous user turn: {hist[-1]['audio_text']}\n" if hist else ""
    tools = gold_tools(row["label"]["tool_calls"])
    return f"{prev}Request: {row['audio_text']}\n=> {json.dumps({'apps': sorted({t.split('.')[0] for t in tools}), 'tools': sorted(tools)}, ensure_ascii=False)}"


def gold_tools(calls: list[dict]) -> set[str]:
    """Tools a gold call list needs: every called tool except sys.ask_missing_info, plus the tool an ask is about."""
    out = set()
    for c in calls:
        if c["name"] == "sys.ask_missing_info":
            if (c.get("arguments") or {}).get("tool_name"):
                out.add(c["arguments"]["tool_name"])
        else:
            out.add(c["name"])
    return out


def select_tools(client: OpenAI, cfg: SelectorConfig, item: dict, transcripts: dict[str, str], raw_apps: dict[str, dict],
                 catalog: dict[str, dict], fewshot: list[str] | None = None,
                 llm_log: LLMCallLog | None = None) -> dict:
    """{"apps": [...], "tools": [...], "raw": model text, "usage": {...}}; names outside the installed tools are dropped,
    and apps always include the apps of the selected tools."""
    installed_apps = list(item["context"]["installed_apps"])
    installed_tools = [t for a in item["context"]["installed_apps"].values() for t in a["tools"]]
    system = SELECT_RULES + "\n\n# Installed apps and tools\n" + catalog_block(item, raw_apps, catalog)
    if fewshot:
        system += "\n\n# Examples (other users, other apps may be installed)\n" + "\n\n".join(fewshot)
    user = (f"# Earlier turns\n{history_block(item, transcripts, cfg.result_chars)}\n\n"
            f"# LAST user request\n{transcripts.get(item['audio'], '')}")
    schema = {"type": "object", "additionalProperties": False, "required": ["apps", "tools"], "properties": {
        "apps": {"type": "array", "items": {"type": "string", "enum": installed_apps}},
        "tools": {"type": "array", "items": {"type": "string", "enum": installed_tools}}}}
    body = {"model": cfg.model, "temperature": 0.0, "max_tokens": cfg.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "selection", "schema": schema}},
            "chat_template_kwargs": {"enable_thinking": cfg.thinking}}
    resp = chat(client, body, llm_log, sample_id=item["id"], step=0)
    text = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {}
    tools = [t for t in dict.fromkeys(parsed.get("tools") or []) if t in installed_tools]
    apps = [a for a in dict.fromkeys((parsed.get("apps") or []) + [t.split(".")[0] for t in tools]) if a in installed_apps]
    return {"apps": apps, "tools": tools, "raw": text, "usage": resp.usage.model_dump() if resp.usage else {}}
