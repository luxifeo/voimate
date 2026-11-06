"""Planner sub-agent: one LLM call that reads the earlier turns, the LAST user request and the installed tools, and
writes a short analysis plan (meaning, context, parts, ordered tool steps, missing information) for the agent to follow.
Experiment only; the agent and the submission path do not use it."""
from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from .agent import chat
from .llm_log import LLMCallLog
from .tool_selector import history_block

PLAN_RULES = """You are the planner of LifeMate, a Vietnamese voice assistant that acts on the user's apps by calling tools.
You do not call tools. You write a short analysis plan that the executor will follow to handle the LAST user request.
The requests were transcribed from Vietnamese speech and may contain recognition errors.

Write the plan in exactly these 6 sections, in this order, each section once. One short line per item, at most 30 lines
in total. Write decisions only: no "wait", "maybe", "let's assume", no alternatives. Never repeat a line or a phrase.
Stop after section 6.

## 1. Request
- Heard: the LAST request, copied word for word.
- Changes: what the user takes back or replaces inside this message. "à thôi", "thôi khỏi", "khoan", "đừng ... nữa",
  "không cần nữa" cancel the part before them; "à không, X" and "nhầm, X" replace the value before them with X.
  "(nếu) ... không thì thôi" is the else-branch of a condition ("otherwise do nothing"), not a change.
  "none" if there are none.
- Meaning: what the user wants after the changes, in one sentence, or "unclear". The transcript may contain
  sound-alike recognition errors: read an odd word as the closest-sounding action, device, contact or item of the
  installed apps or the earlier turns (e.g. "bớt thức" = "báo thức", "bút xe" = "book xe"), without discussing it.

## 2. Context from earlier turns
- Done: actions already executed in earlier turns. The executor must not redo them.
- Known: IDs and facts from earlier tool results or assistant replies that this request needs, as name = value.
  Copy IDs exactly; never invent one.
- Refers to: what "nó", "cái đó", "cái còn lại", "người đó", "lúc nãy" point to.
Each item is "none" when nothing applies.

## 3. Parts
One line per part of the request: "P1: <action in a few words> | do", "| cancelled" or "| if <condition>".
- cancelled: taken back under Changes, or the user says they will do it themselves later ("để mình tự xem",
  "về nhà mình check lại"). A cancelled part gets no steps.
- if <condition>: when the earlier turns already give the values the condition needs (forecast, prices, balance,
  distance), decide it now, show the arithmetic, and keep only the branch that applies. Otherwise the executor checks
  the condition with a read tool first and acts only if it holds.
Last line: "Scope: in", or "Scope: out (<reason>)" when no installed tool can do any part, the meaning is unclear, or
the message is only chit-chat, thanks or a statement. With "Scope: out", write "none" in sections 4 to 6.

## 4. Steps
Numbered, in execution order: "S1: <tool name> | <purpose> | <param> = <value> (<source>); ...".
- source: user (the user's words), earlier (earlier turns), S<n> (the result of step n), ask (the user must say it).
- An ID parameter is marked "<- tool" in the tool list. Get it from exactly that read tool (chat_id comes from
  chat_ott.list_chats, not contacts.search), unless the ID is already Known.
- Add a read step only when a later step needs its result or the user asked for that information. No read "to be
  sure" and no read of what is already Known.
- Choose the tool whose description matches the exact action and object (light vs socket switch, alarm vs reminder,
  phone top-up vs money transfer, update vs cancel vs create).
- Keep times, dates and amounts as the user said them ("7 giờ tối mai", "500 nghìn"); the executor converts them.
- A write step whose required value only the user can give still gets its step, with source "ask".

## 5. Missing information
- "ask: <tool name> | <param>, ..." for each step with a source "ask": a REQUIRED value only the user can give
  (amount, time, recipient, content, choice between several matching items) that the user did not say.
- Never name a read tool (search, list, get) or an ID here: the executor calls the read tools itself. The executor
  does the complete parts first, then asks.
- "none" if nothing is missing.

## 6. Tools
One line: the tool names of section 4 in order, joined by " -> ", or "none".

# Example
Earlier turns:
User: Tìm số chị Lan giúp anh
  -> contacts.search {"query": "Lan"} => {"ok": true, "contacts": [{"id": "c_12", "name": "Chị Lan", "phone": "0901234567"}]}
Assistant: Dạ số chị Lan là 0901234567 ạ.
LAST user request: Nhắn Zalo cho chị ấy tối nay họp lúc tám giờ, gọi luôn cho chị ấy, à mà thôi khỏi gọi, rồi đặt bớt thức bảy rưỡi tối

## 1. Request
- Heard: Nhắn Zalo cho chị ấy tối nay họp lúc tám giờ, gọi luôn cho chị ấy, à mà thôi khỏi gọi, rồi đặt bớt thức bảy rưỡi tối
- Changes: "à mà thôi khỏi gọi" cancels the call
- Meaning: Send chị Lan a Zalo message about the 8 pm meeting tonight and set an alarm ("báo thức") at 7:30 pm.
## 2. Context from earlier turns
- Done: contacts.search found chị Lan
- Known: chị Lan = c_12 (contacts.search)
- Refers to: "chị ấy" = chị Lan
## 3. Parts
- P1: Zalo message to chị Lan about the meeting | do
- P2: call chị Lan | cancelled
- P3: alarm at 7:30 pm | do
Scope: in
## 4. Steps
S1: chat_ott.list_chats | find chị Lan's chat_id (c_12 is a contact_id, not a chat_id) | no params
S2: chat_ott.send_message | tell chị Lan about the meeting | chat_id = chị Lan's chat (S1); content = "tối nay họp lúc tám giờ" (user)
S3: alarm.create | alarm before the meeting | time = "bảy rưỡi tối" (user)
## 5. Missing information
none
## 6. Tools
chat_ott.list_chats -> chat_ott.send_message -> alarm.create"""


@dataclass
class PlannerConfig:
    model: str = "qwen3.5-4b"
    thinking: bool = False
    max_tokens: int = 1024
    result_chars: int = 1500  # earlier tool results are cut to this many characters (the agent's history_result_chars)


def id_sources(catalog: dict[str, dict], installed: list[str]) -> dict[tuple[str, str], str]:
    """{(tool, ID param): read tools that return it}: an x-ref "app.collection" maps to the installed read tools of that
    app whose returns or description name the collection and that do not themselves need the ID."""
    out = {}
    for name in installed:
        for k, p in ((catalog[name].get("parameters") or {}).get("properties") or {}).items():
            if "x-ref" not in p:
                continue
            app, coll = p["x-ref"].split(".", 1)
            src = [r for r in installed
                   if r.split(".")[0] == app and catalog[r].get("kind") == "read" and r != name
                   and coll in f"{catalog[r].get('returns', '')} {catalog[r].get('description', '')}"
                   and k not in ((catalog[r].get("parameters") or {}).get("properties") or {})]
            out[name, k] = " or ".join(src) or f"an ID from {p['x-ref']}"
    return out


def tools_block(item: dict, raw_apps: dict[str, dict], catalog: dict[str, dict]) -> str:
    """Installed apps with their tools: kind, description, parameters (required ones marked with *, ID parameters with
    the read tool that returns the ID) and what read tools return."""
    installed = [t for a in item["context"]["installed_apps"].values() for t in a["tools"]]
    refs = id_sources(catalog, installed)
    lines = []
    for app_id, app in item["context"]["installed_apps"].items():
        meta = raw_apps.get(app_id, {})
        lines.append(f"- {app_id} ({meta.get('name_vi', app_id)}): {meta.get('description', '')}")
        for name in app["tools"]:
            spec = catalog[name]
            params = spec.get("parameters") or {}
            req = set(params.get("required", []))
            args = ", ".join(k + ("*" if k in req else "") + (f" <- {refs[name, k]}" if (name, k) in refs else "")
                             for k in params.get("properties", {}))
            ret = f" returns: {spec['returns']}" if spec.get("kind") == "read" and spec.get("returns") else ""
            lines.append(f"    - {name} [{spec.get('kind', '')}]: {spec.get('description', '')} ({args}){ret}")
    return "\n".join(lines)


def plan_request(client: OpenAI, cfg: PlannerConfig, item: dict, transcripts: dict[str, str],
                 raw_apps: dict[str, dict], catalog: dict[str, dict], llm_log: LLMCallLog | None = None) -> dict:
    """{"plan": plan text, "reasoning": thinking text or "", "finish_reason": ..., "usage": {...}}."""
    system = PLAN_RULES + "\n\n# Installed apps and tools\n" + tools_block(item, raw_apps, catalog)
    user = (f"# Earlier turns\n{history_block(item, transcripts, cfg.result_chars)}\n\n"
            f"# LAST user request\n{transcripts.get(item['audio'], '')}")
    body = {"model": cfg.model, "temperature": 0.0, "max_tokens": cfg.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "chat_template_kwargs": {"enable_thinking": cfg.thinking}}
    resp = chat(client, body, llm_log, sample_id=item["id"], step=0)
    choice = resp.choices[0]
    reasoning = getattr(choice.message, "reasoning", None) or getattr(choice.message, "reasoning_content", None)
    return {"plan": choice.message.content or "", "reasoning": reasoning or "", "finish_reason": choice.finish_reason,
            "usage": resp.usage.model_dump() if resp.usage else {}}
