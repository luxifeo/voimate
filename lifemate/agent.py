"""Tool-calling agent on Gemma-4-E2B-it (native function calling through vLLM's gemma4 tool parser).

Each step the model sees the system prompt (rules, user context, date table, few-shot demos), the earlier
conversation (with the earlier tool calls/results) and the results of its own calls so far. Every proposed call is
schema-checked with env.validate() first: a rejected call is returned to the model as an error without being executed,
so it never reaches the submission log (rejected calls would count as unmatched calls).
"""
from __future__ import annotations

import ast
import base64
import copy
import json
import operator
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from openai import OpenAI

from .llm_log import LLMCallLog

WEEKDAYS_VI = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
STANDARD_KEYS = {"type", "description", "enum", "items", "properties", "required", "nullable"}

SYSTEM_RULES = """You are LifeMate, a Vietnamese voice assistant that ACTS on the user's apps by calling tools.
The user's requests were transcribed from Vietnamese speech and may contain small recognition errors.

Rules:
1. Handle only the LAST user request. Earlier turns are context; their tool calls were already executed, never repeat them.
2. Never invent IDs (device_id, contact_id, event_id, bill_id, chat_id, ...). Take them from a read tool result or from earlier tool results in this conversation. If the needed ID is already in an earlier result, reuse it without reading again.
3. When the next call needs the result of a read tool, call the read tool alone first and wait for its result.
4. Call list/status/search read tools with no arguments ({}) unless a filter is clearly required (contacts.search takes the spoken name or relation as query, e.g. {"query": "mẹ"}). Never add optional arguments the user did not ask for. Do not call read tools the request does not need.
5. Datetimes: ISO-8601 with seconds and offset, e.g. 2026-11-28T17:00:00+07:00. Dates: YYYY-MM-DD. Resolve "hôm nay", "mai", "tối nay", "thứ sáu này", "tuần sau" with the date table below. Money is an integer in VND ("500 nghìn" = 500000, "1 triệu 2" = 1200000). When moving an event, shift both start and end.
6. Never guess. Every required value (amount, time, address, mode, recipient, which item, ...) must come from the user's words, the context or tool results. If a required value is missing or ambiguous (the user did not say it, a search returned several matches and the request does not say which one, or nothing matched), call sys__ask_missing_info with tool_name (format app.tool, e.g. "sms.send") and params (the missing parameter names), then stop. Do the complete parts of the request first.
7. You can never ask the user in plain text: the only way to ask is sys__ask_missing_info. Never say you will do something: do it now by calling the tool.
8. Out of scope: call NO tool when no installed app can do the request, when the message is only chit-chat/thanks/a statement, or when you cannot tell what the user wants. If the user takes back an action in the same message ("... à thôi khỏi", "thôi không cần", "đừng ... nữa"), do not do that action. Use sys__ask_missing_info only when you know exactly which tool to call and only one of its required parameters is missing.
9. Conditional requests ("nếu mai mưa thì ..."): check the condition with a read tool first, then act only if it holds.
10. Free-text arguments (content, title, note, ...): short Vietnamese text taken from the user's words.
11. When the request is fully handled, answer with one short Vietnamese sentence and no tool call."""

NUDGE = ("(Hệ thống) Không được hỏi người dùng hay hứa hẹn bằng lời. Nếu thiếu thông tin bắt buộc, hãy gọi "
         "sys__ask_missing_info. Nếu yêu cầu còn phần chưa thực hiện, hãy gọi tool ngay. Nếu đã xong, trả lời ngắn gọn.")

# experiment switch arg_guide: extra rule next to the per-parameter usage notes from the train labels
ARG_RULE = """12. Optional parameters (not in `required`): follow the usage note in each parameter's description. Fill an optional parameter only when the user's words call for it or its note says it is normally filled; never add one just to restate a default (repeat "once", limit, unread_only, type/room/floor filters, attendees, note, label) or because an example used it."""

# experiment switch history_mode "llm_summary": one extra LLM call condenses the earlier turns
HISTORY_SUMMARY_PROMPT = """Tóm tắt ngắn gọn đoạn hội thoại giữa người dùng và trợ lý ảo dưới đây thành các gạch đầu dòng tiếng Việt.
Giữ nguyên chính xác mọi ID (contact_id, event_id, device_id, bill_id, ...), tên người, địa điểm, thời gian, số tiền và kết quả quan trọng của các tool.
Ghi rõ việc nào đã làm xong (không được làm lại) và điều gì người dùng còn đang nói dở hoặc trợ lý đang hỏi. Không thêm thông tin mới.

Hội thoại:
"""

# experiment switch calculator: a private scratch tool handled in-process (never sent to the ToolEnv)
CALC_NAME = "scratch.calc"
CALC_RULE = """13. Arithmetic: before using any computed value (sums or differences of amounts, comparing two numbers, a time shifted by N minutes/hours, the weekday of a date, a date N days away), compute it with scratch__calc and use its result. scratch__calc is your private calculator, not one of the user's apps; it is not shown to the user."""
CALC_TOOL = {"type": "function", "function": {
    "name": "scratch__calc",
    "description": ("Máy tính nháp riêng của trợ lý (không phải app của người dùng, không tính là hành động). "
                    "Biểu thức kiểu Python: + - * / // % ** ( ), so sánh < <= > >= == !=, and/or/not, abs round min max sum; "
                    "ngày giờ: add_minutes('2026-11-14T15:00:00+07:00', -45), add_days('2026-11-14', 3), "
                    "weekday('2026-11-14'), days_between('2026-11-14', '2026-11-20'). Trả về: {ok, result}"),
    "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "biểu thức cần tính"}},
                   "required": ["expression"]}}}


@dataclass
class AgentConfig:
    model: str = "gemma-4-e2b"
    temperature: float = 0.0
    max_tokens: int = 1024
    thinking: bool = False
    max_steps: int = 10
    max_calls: int = 10
    max_rejections: int = 3
    result_chars: int = 6000
    history_result_chars: int = 1500
    # final user turn given to the model as: "text" (ASR transcript), "audio" (raw audio) or "audio+text" (both)
    input_mode: str = "text"
    # one extra turn when the model ends with a question or a promise instead of a tool call
    nudge: bool = True
    # send the model's reasoning back with its tool calls (the Gemma 4 template keeps it for tool-call turns)
    keep_reasoning: bool = True
    # experiment switches; the defaults are the submitted behaviour
    date_table: str = "v1"  # "v1": yesterday .. +13 days; "weeks": last week .. the week after next, tagged by week
    time_position: str = "system"  # date table in the system prompt context, "end" of the system prompt, or "user" message
    arg_guide: bool = False  # ARG_RULE + per-parameter usage notes (param_notes) in the tool schemas
    history_mode: str = "messages"  # chat messages; "summary": one compact block in the system prompt; "llm_summary": LLM-condensed block
    history_summary_chars: int = 800
    calculator: bool = False  # scratch__calc tool + CALC_RULE


def to_model_name(name: str) -> str:
    """vLLM's gemma4 parser only accepts \\w+ function names, so 'app.tool' is exposed as 'app__tool'."""
    return name.replace(".", "__")


def from_model_name(name: str) -> str:
    return name.replace("__", ".", 1)


def _clean_schema(p: dict) -> dict:
    """Keep the keys the Gemma chat template renders; fold the others (x-ref, minimum, format, ...) into the description."""
    p = copy.deepcopy(p)
    notes = []
    if "x-ref" in p:
        notes.append(f"phải là ID có thật lấy từ {p['x-ref']}")
    if p.get("x-match") == "fuzzy":
        notes.append("văn bản tự do, ngắn gọn theo lời người dùng")
    for k in list(p):
        if k.startswith("x-"):
            p.pop(k)
        elif k not in STANDARD_KEYS:
            notes.append(f"{k}: {json.dumps(p.pop(k), ensure_ascii=False)}")
    if isinstance(p.get("items"), dict):
        p["items"] = _clean_schema(p["items"])
    if isinstance(p.get("properties"), dict):
        p["properties"] = {k: _clean_schema(v) for k, v in p["properties"].items()}
    if notes:
        p["description"] = "; ".join(x for x in [p.get("description", ""), *notes] if x)
    return p


def to_openai_tool(spec: dict, notes: dict[str, str] | None = None) -> dict:
    """`notes`: {parameter: usage note} appended to the parameter descriptions (arg_guide)."""
    params = spec.get("parameters") or {"type": "object", "properties": {}}
    if notes:
        params = copy.deepcopy(params)
        for k, note in notes.items():
            p = params.get("properties", {}).get(k)
            if p is not None:
                p["description"] = "; ".join(x for x in [p.get("description", ""), note] if x)
    desc = spec.get("description", "")
    one_of = params.get("x-one-of")
    if one_of and "một trong" not in desc:
        desc += ". Bắt buộc một trong: " + " hoặc ".join("+".join(g) for g in one_of)
    if spec["name"] == "sys.ask_missing_info":
        desc += ". Gọi khi thiếu tham số bắt buộc; tool_name dạng app.tool (ví dụ sms.send)."
    elif spec.get("kind") == "read":
        desc += " (chỉ đọc)"
    if spec.get("returns"):
        desc += f". Trả về: {spec['returns']}"
    clean = _clean_schema({k: v for k, v in params.items() if k != "x-one-of"})
    clean.setdefault("type", "object")
    clean.setdefault("properties", {})
    return {"type": "function", "function": {"name": to_model_name(spec["name"]), "description": desc, "parameters": clean}}


WEEK_TAGS = {-1: "tuần trước", 0: "tuần này", 1: "tuần sau", 2: "tuần sau nữa"}


def date_table_weeks(current_time: str) -> str:
    """date_table "weeks": Monday of last week to Sunday of the week after next, every day tagged with its week."""
    today = datetime.fromisoformat(current_time).date()
    monday = today - timedelta(days=today.weekday())
    lines = []
    for d in range(-7, 21):
        day = monday + timedelta(days=d)
        rel = {-1: "hôm qua", 0: "hôm nay", 1: "ngày mai", 2: "ngày kia"}.get((day - today).days)
        tags = ", ".join(x for x in [rel, WEEK_TAGS[d // 7]] if x)
        lines.append(f"- {WEEKDAYS_VI[day.weekday()]} {day.isoformat()} ({tags})")
    return "\n".join(lines)


def date_table(current_time: str, days: int = 14, style: str = "v1") -> str:
    if style == "weeks":
        return date_table_weeks(current_time)
    now = datetime.fromisoformat(current_time)
    lines = []
    for d in range(-1, days):
        day = now + timedelta(days=d)
        tag = {-1: " (hôm qua)", 0: " (hôm nay)", 1: " (ngày mai)", 2: " (ngày kia)"}.get(d, "")
        lines.append(f"- {WEEKDAYS_VI[day.weekday()]} {day.date().isoformat()}{tag}")
    return "\n".join(lines)


def time_context(current_time: str, style: str = "v1") -> str:
    """Current time and date table as one block (time_position "end" / "user")."""
    now = datetime.fromisoformat(current_time)
    return (f"# Time Context\n- Now: {current_time} ({WEEKDAYS_VI[now.weekday()]})\n"
            f"Date table (weeks start on Monday):\n{date_table(current_time, style=style)}")


def build_system_prompt(item: dict, catalog: dict, fewshot: list[str], cfg: AgentConfig | None = None,
                        history_block: str = "") -> str:
    cfg = cfg or AgentConfig()
    ctx = item["context"]
    now = datetime.fromisoformat(ctx["current_time"])
    apps = ", ".join(ctx["installed_apps"])
    profile = json.dumps(ctx.get("user_profile", {}), ensure_ascii=False)
    rules = SYSTEM_RULES + ("\n" + ARG_RULE if cfg.arg_guide else "") + ("\n" + CALC_RULE if cfg.calculator else "")
    parts = [
        rules,
        "\nUser context:",
        f"- Now: {ctx['current_time']} ({WEEKDAYS_VI[now.weekday()]}), timezone {ctx.get('timezone', 'Asia/Ho_Chi_Minh')}",
        f"- Location: {json.dumps(ctx.get('location', {}))}",
        f"- Profile: {profile}",
        f"- Installed apps: {apps}",
    ]
    if cfg.time_position == "system":
        parts += ["\nDate table (weeks start on Monday):", date_table(ctx["current_time"], style=cfg.date_table)]
    if history_block:
        parts += ["\nEarlier turns of this conversation (already executed: never repeat their calls; reuse their IDs and "
                  "results):", history_block]
    if fewshot:
        parts.append("\nSolved examples from other users (their IDs and data differ from this user's; tool names written app.tool):")
        parts.extend(f"\n[Example {i + 1}]\n{ex}" for i, ex in enumerate(fewshot))
    if cfg.time_position == "end":
        parts.append("\n" + time_context(ctx["current_time"], cfg.date_table))
    return "\n".join(parts)


def _dump(obj, limit: int) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "…(cắt bớt)"


def _compact(obj):
    """Drop null / empty values recursively (history_mode "summary")."""
    if isinstance(obj, dict):
        return {k: _compact(v) for k, v in obj.items() if v not in (None, "", [], {})}
    if isinstance(obj, list):
        return [_compact(v) for v in obj]
    return obj


def history_summary(item: dict, transcripts: dict[str, str], cfg: AgentConfig) -> str:
    """Earlier turns as one compact text block: user words, calls with arguments, compacted results, assistant replies."""
    lines = []
    for turn in item.get("history", []):
        if turn["role"] == "user":
            lines.append(f"Người dùng: {transcripts.get(turn.get('audio'), turn.get('audio_text', ''))}")
            continue
        results = turn.get("tool_results") or []
        for i, c in enumerate(turn.get("tool_calls") or []):
            res = results[i] if i < len(results) else {"ok": True}
            lines.append(f"  → {c['name']} {json.dumps(c.get('arguments') or {}, ensure_ascii=False)} "
                         f"⇒ {_dump(_compact(res), cfg.history_summary_chars)}")
        if turn.get("text"):
            lines.append(f"Trợ lý: {turn['text']}")
    return "\n".join(lines)


def history_messages(item: dict, transcripts: dict[str, str], cfg: AgentConfig) -> list[dict]:
    msgs = []
    for t, turn in enumerate(item.get("history", [])):
        if turn["role"] == "user":
            msgs.append({"role": "user", "content": transcripts.get(turn.get("audio"), turn.get("audio_text", ""))})
            continue
        calls = turn.get("tool_calls") or []
        results = turn.get("tool_results") or []
        if calls:
            ids = [f"h{t}_{i}" for i in range(len(calls))]
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": cid, "type": "function",
                 "function": {"name": to_model_name(c["name"]), "arguments": json.dumps(c.get("arguments") or {}, ensure_ascii=False)}}
                for cid, c in zip(ids, calls)]})
            for i, cid in enumerate(ids):
                res = results[i] if i < len(results) else {"ok": True}
                msgs.append({"role": "tool", "tool_call_id": cid, "content": _dump(res, cfg.history_result_chars)})
        msgs.append({"role": "assistant", "content": turn.get("text", "")})
    return msgs


@dataclass
class AgentResult:
    tool_calls: list
    state_hash: str
    trace: list = field(default_factory=list)
    stop_reason: str = ""


def _iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


CALC_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "add_minutes": lambda t, n: (_iso(t) + timedelta(minutes=n)).isoformat(),
    "add_days": lambda d, n: ((_iso(d) + timedelta(days=n)).isoformat() if "T" in d
                              else (_iso(d) + timedelta(days=n)).date().isoformat()),
    "weekday": lambda d: WEEKDAYS_VI[_iso(d).weekday()],
    "days_between": lambda a, b: (_iso(b).date() - _iso(a).date()).days,
}
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Eq: operator.eq,
        ast.NotEq: operator.ne}


def calc(expression: str):
    """Evaluate a small arithmetic / comparison / date expression without eval()."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, str)):
            return n.value
        if isinstance(n, (ast.List, ast.Tuple)):
            return [ev(x) for x in n.elts]
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = ev(n.left), ev(n.right)
            if not all(isinstance(x, (int, float)) for x in (a, b)):
                raise ValueError("phép tính chỉ dùng cho số")
            if isinstance(n.op, ast.Pow) and abs(b) > 64:
                raise ValueError("số mũ quá lớn")
            return _BIN[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd, ast.Not)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else (v if isinstance(n.op, ast.UAdd) else not v)
        if isinstance(n, ast.Compare):
            left = ev(n.left)
            for op, right in zip(n.ops, n.comparators):
                if type(op) not in _CMP:
                    raise ValueError("phép so sánh không hỗ trợ")
                r = ev(right)
                if not _CMP[type(op)](left, r):
                    return False
                left = r
            return True
        if isinstance(n, ast.BoolOp):
            vals = [ev(x) for x in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in CALC_FUNCS and not n.keywords:
            return CALC_FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"không hỗ trợ: {ast.dump(n)[:80]}")

    try:
        v = ev(ast.parse(str(expression), mode="eval"))
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        return {"ok": True, "result": v}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def final_user_message(final_text: str, audio_path: Path | None, mode: str, time_block: str = "") -> dict:
    if time_block and mode == "text":  # time_position "user": the date table follows the request
        return {"role": "user", "content": f"# User Question\n{final_text}\n\n{time_block}"}
    if mode == "text" or audio_path is None:
        return {"role": "user", "content": final_text}
    audio = {"type": "audio_url",
             "audio_url": {"url": "data:audio/wav;base64," + base64.b64encode(audio_path.read_bytes()).decode()}}
    if mode == "audio":
        text = "Yêu cầu của người dùng nằm trong đoạn âm thanh sau."
    else:
        text = f"Yêu cầu của người dùng nằm trong đoạn âm thanh sau. Bản chép lời tự động (có thể sai): \"{final_text}\""
    return {"role": "user", "content": [{"type": "text", "text": text}, audio]}  # audio after text (model card)


VLLM_EXTRA_KEYS = {"chat_template_kwargs"}  # vLLM-only body fields, sent through the OpenAI client's extra_body


def chat(client: OpenAI, body: dict, log: LLMCallLog | None = None, **meta):
    """One chat.completions call from a wire-format request body; with `log`, the request, the raw response (or the
    error) and the latency are appended to the LLM call log together with `meta` (sample id, step)."""
    t0 = time.time()
    try:
        resp = client.chat.completions.create(**{k: v for k, v in body.items() if k not in VLLM_EXTRA_KEYS},
                                              extra_body={k: v for k, v in body.items() if k in VLLM_EXTRA_KEYS})
    except Exception as e:
        if log:
            log.write({**meta, "latency_s": round(time.time() - t0, 3), "request": body, "response": None,
                       "error": f"{type(e).__name__}: {e}"})
        raise
    if log:
        log.write({**meta, "latency_s": round(time.time() - t0, 3), "request": body, "response": resp.model_dump(),
                   "error": None})
    return resp


def run_agent(client: OpenAI, cfg: AgentConfig, item: dict, env, catalog: dict, transcripts: dict[str, str],
              fewshot: list[str], audio_path: Path | None = None, llm_log: LLMCallLog | None = None,
              apps: set[str] | None = None, param_notes: dict[str, dict[str, str]] | None = None) -> AgentResult:
    """`apps`: only the tools of these installed apps are offered (tool filter experiment); None = every installed app.
    `param_notes`: {tool: {parameter: usage note}} added to the tool schemas when cfg.arg_guide."""
    installed = item["context"]["installed_apps"]
    allowed = [t for a, app in installed.items() if apps is None or a in apps for t in app["tools"]]
    allowed.append("sys.ask_missing_info")
    notes = (param_notes or {}) if cfg.arg_guide else {}
    tools = [to_openai_tool(catalog[t], notes.get(t)) for t in allowed if t in catalog]
    if cfg.calculator:
        tools.append(CALC_TOOL)
    final_text = transcripts.get(item["audio"], item.get("audio_text", ""))
    summary = cfg.history_mode in ("summary", "llm_summary") and history_summary(item, transcripts, cfg)
    if summary and cfg.history_mode == "llm_summary":
        resp = chat(client, {
            "model": cfg.model, "temperature": cfg.temperature, "max_tokens": 600,
            "messages": [{"role": "user", "content": HISTORY_SUMMARY_PROMPT + summary}],
            "chat_template_kwargs": {"enable_thinking": False},
        }, llm_log, sample_id=item["id"], step=-1)
        summary = (resp.choices[0].message.content or "").strip() or summary
    messages = [{"role": "system", "content": build_system_prompt(item, catalog, fewshot, cfg, summary or "")}]
    if not summary:
        messages += history_messages(item, transcripts, cfg)
    time_block = time_context(item["context"]["current_time"], cfg.date_table) if cfg.time_position == "user" else ""
    messages.append(final_user_message(final_text, audio_path, cfg.input_mode, time_block))

    trace, executed = [], set()
    n_calls = rejections = 0
    nudged = False
    stop_reason = "max_steps"
    for step in range(cfg.max_steps):
        resp = chat(client, {
            "model": cfg.model, "messages": messages, "tools": tools, "tool_choice": "auto",
            "temperature": cfg.temperature, "max_tokens": cfg.max_tokens,
            "chat_template_kwargs": {"enable_thinking": cfg.thinking},
        }, llm_log, sample_id=item["id"], step=step)
        msg = resp.choices[0].message
        calls = msg.tool_calls or []
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
        trace.append({"content": msg.content, "reasoning": reasoning,
                      "calls": [(c.function.name, c.function.arguments) for c in calls]})
        if not calls:
            text = msg.content or ""
            if cfg.nudge and not nudged and ("?" in text or " sẽ " in f" {text.lower()} "):
                nudged = True
                messages += [{"role": "assistant", "content": text}, {"role": "user", "content": NUDGE}]
                trace[-1]["nudged"] = True
                continue
            stop_reason = "answered"
            break
        assistant = {"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
            for c in calls]}
        if cfg.keep_reasoning and reasoning:  # Gemma 4 templates read `reasoning`, Qwen-style ones `reasoning_content`
            assistant["reasoning"] = assistant["reasoning_content"] = reasoning
        messages.append(assistant)
        stop = False
        for c in calls:
            name = from_model_name(c.function.name)
            if stop:  # calls emitted after ask_missing_info in the same turn are dropped
                result = {"ok": False, "error": "bỏ qua: đang chờ người dùng bổ sung thông tin"}
            elif cfg.calculator and name == CALC_NAME:  # private scratch tool: never reaches the ToolEnv
                try:
                    result = calc(json.loads(c.function.arguments or "{}").get("expression", ""))
                except (json.JSONDecodeError, AttributeError):
                    result = {"ok": False, "error": "arguments phải là JSON object có expression"}
            else:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = None
                if not isinstance(args, dict):
                    result = {"ok": False, "error": "arguments phải là JSON object"}
                    rejections += 1
                else:
                    if name == "sys.ask_missing_info" and isinstance(args.get("tool_name"), str):
                        args["tool_name"] = from_model_name(args["tool_name"])
                    key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                    check = env.validate(name, args)
                    if key in executed:
                        result = {"ok": False, "error": "lời gọi này đã thực hiện rồi, không gọi lại"}
                        rejections += 1
                    elif not check["ok"]:
                        result = {"ok": False, "error": check["error"]}
                        rejections += 1
                    else:
                        result = env.call(name, args)
                        executed.add(key)
                        n_calls += 1
                        stop = name == "sys.ask_missing_info"
            trace[-1].setdefault("results", []).append(_dump(result, 400))
            messages.append({"role": "tool", "tool_call_id": c.id, "content": _dump(result, cfg.result_chars)})
        if stop:
            stop_reason = "ask_missing_info"
            break
        if n_calls >= cfg.max_calls:
            stop_reason = "max_calls"
            break
        if rejections >= cfg.max_rejections:
            stop_reason = "max_rejections"
            break
    return AgentResult(env.get_tool_calls(), env.get_state_hash(), trace, stop_reason)
