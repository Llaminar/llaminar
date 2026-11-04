#!/usr/bin/env python3
"""Generate DeepSeek-V4 chat-encoding golden prompts from the production vLLM encoder.

The production comparison target is vLLM's DeepSeek-V4 tokenizer mode: the
standalone ``vllm/tokenizers/deepseek_v4_encoding.py`` plus the request mapping
of ``vllm/tokenizers/deepseek_v4.py`` (thinking defaults, reasoning-effort
aliases, request tools attached to the first system message) and the OpenAI
request normalization that renames ``reasoning_content`` to ``reasoning``.
This script imports only the encoding module (it has no vLLM dependencies),
re-states the wrapper mapping below, and writes every case with its exact
prompt to a JSON fixture consumed by ``Test__DeepSeekV4ChatEncoding``.

usage: make_chat_encoding_golden.py --encoding PATH/deepseek_v4_encoding.py OUT.json
"""
import argparse
import copy
import importlib.util
import json
import sys


def load_encoding(path):
    spec = importlib.util.spec_from_file_location("deepseek_v4_encoding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(enc, request):
    """Mirror vLLM's DeepSeek-V4 tokenizer wrapper for one OpenAI request."""
    kwargs = dict(request.get("chat_template_kwargs") or {})
    if "reasoning_effort" in request:
        kwargs.setdefault("reasoning_effort", request["reasoning_effort"])
    messages = copy.deepcopy(request["messages"])
    for msg in messages:  # OpenAI normalization: reasoning_content -> reasoning
        rc = msg.pop("reasoning_content", None)
        if rc is not None and msg.get("reasoning") is None:
            msg["reasoning"] = rc
        # vLLM's chat_message parser flattens all-text tool content with "\n";
        # other roles keep OpenAI parts, which the encoder joins with "\n\n".
        if msg.get("role") == "tool" and isinstance(msg.get("content"), list):
            msg["content"] = "\n".join(p.get("text", "") for p in msg["content"])

    thinking = kwargs.get("thinking")
    enable_thinking = kwargs.get("enable_thinking")
    thinking_enabled = bool(thinking) or bool(enable_thinking)
    if "thinking" not in kwargs and "enable_thinking" not in kwargs:
        thinking_enabled = True
    thinking_mode = "thinking" if thinking_enabled else "chat"

    tools = request.get("tools")
    if tools:
        idx = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
        if idx is None:
            messages.insert(0, {"role": "system"})
            idx = 0
        messages[idx]["tools"] = tools

    effort = kwargs.get("reasoning_effort")
    if not isinstance(effort, str):
        effort = "high" if thinking_enabled else None
    elif effort == "none":
        thinking_mode, effort = "chat", None
    elif effort == "max":
        effort = "max"
    elif effort in ("low", "minimal", "medium"):
        effort = "low"
    else:
        effort = "high"
    return enc.encode_messages(messages, thinking_mode=thinking_mode, drop_thinking=True,
                               reasoning_effort=effort, add_generation_prompt=True)


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city, \"quoted\" and ünïcode.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "days": {"type": "integer", "minimum": 1, "maximum": 14},
        "threshold": {"type": "number", "default": 0.5, "multipleOf": 1e-07, "exclusiveMaximum": 100.0},
        "tags": {"type": "array", "items": {"type": "string"}}},
        "required": ["city"]}}}
SEARCH = {"type": "function", "function": {
    "name": "search", "description": "Search the web.\nReturns JSON.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
CALL_A = {"id": "call_a", "type": "function", "function": {
    "name": "get_weather", "arguments": json.dumps({"city": "Berlin", "days": 3, "tags": ["a", "ü"]})}}
CALL_B = {"id": "call_b", "type": "function", "function": {
    "name": "search", "arguments": "{\"query\": \"line1\\nline2 \\\"q\\\"\"}"}}

CASES = [
    {"name": "user_default", "messages": [{"role": "user", "content": "Hello"}]},
    {"name": "thinking_off", "chat_template_kwargs": {"thinking": False},
     "messages": [{"role": "user", "content": "Hello"}]},
    {"name": "enable_thinking_off", "chat_template_kwargs": {"enable_thinking": False},
     "messages": [{"role": "user", "content": "Hello"}]},
    {"name": "effort_none", "reasoning_effort": "none", "messages": [{"role": "user", "content": "Hi"}]},
    {"name": "effort_low", "reasoning_effort": "low", "messages": [{"role": "user", "content": "Hi"}]},
    {"name": "effort_medium", "reasoning_effort": "medium", "messages": [{"role": "user", "content": "Hi"}]},
    {"name": "effort_high", "reasoning_effort": "high", "messages": [{"role": "user", "content": "Hi"}]},
    {"name": "effort_xhigh", "reasoning_effort": "xhigh", "messages": [{"role": "user", "content": "Hi"}]},
    {"name": "effort_max", "reasoning_effort": "max", "messages": [{"role": "user", "content": "Hi"}]},
    {"name": "kwargs_effort_max", "chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"},
     "messages": [{"role": "user", "content": "Hi"}]},
    {"name": "effort_with_thinking_off", "reasoning_effort": "max", "chat_template_kwargs": {"thinking": False},
     "messages": [{"role": "user", "content": "Hi"}]},
    {"name": "system_user", "messages": [
        {"role": "system", "content": "You are terse."}, {"role": "user", "content": "2+2?"}]},
    {"name": "multi_turn_reasoning_dropped", "messages": [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer one", "reasoning_content": "thought one"},
        {"role": "user", "content": "second"}]},
    {"name": "multi_turn_chat", "chat_template_kwargs": {"thinking": False}, "messages": [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer one", "reasoning_content": "thought one"},
        {"role": "user", "content": "second"}]},
    {"name": "tools_no_system", "tools": [WEATHER, SEARCH], "messages": [
        {"role": "user", "content": "Weather in Berlin?"}]},
    {"name": "tools_with_system", "tools": [WEATHER], "messages": [
        {"role": "system", "content": "Use tools."}, {"role": "user", "content": "Weather?"}]},
    {"name": "tools_second_system", "tools": [WEATHER], "messages": [
        {"role": "user", "content": "x"}, {"role": "system", "content": "late system"},
        {"role": "user", "content": "y"}]},
    {"name": "tool_round_trip", "tools": [WEATHER, SEARCH], "messages": [
        {"role": "user", "content": "Weather and search"},
        {"role": "assistant", "content": "", "reasoning_content": "I need both.", "tool_calls": [CALL_A, CALL_B]},
        {"role": "tool", "tool_call_id": "call_b", "content": "search result"},
        {"role": "tool", "tool_call_id": "call_a", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny.", "reasoning": "Done."},
        {"role": "user", "content": "Thanks"}]},
    {"name": "tool_round_trip_chat", "tools": [WEATHER], "chat_template_kwargs": {"thinking": False}, "messages": [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": None, "tool_calls": [CALL_A]},
        {"role": "tool", "tool_call_id": "call_a", "content": [{"type": "text", "text": "sunny"}, {"type": "text", "text": "dry"}]}]},
    {"name": "consecutive_assistant_merge", "messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "part one", "reasoning_content": "r1"},
        {"role": "assistant", "content": "part two", "reasoning_content": "r2"},
        {"role": "user", "content": "q2"}]},
    {"name": "content_parts", "messages": [
        {"role": "system", "content": [{"type": "text", "text": "sys a"}, {"type": "text", "text": "sys b"}]},
        {"role": "user", "content": [{"type": "text", "text": "line A"}, {"type": "text", "text": "line B"}]}]},
    {"name": "unicode_and_specials", "messages": [
        {"role": "user", "content": "Grüße 你好 \"quotes\" \\ back\nnewline\ttab"}]},
    {"name": "trailing_assistant_prefill", "messages": [
        {"role": "user", "content": "Count"}, {"role": "assistant", "content": "1, 2,"}]},
    {"name": "developer_role", "messages": [
        {"role": "developer", "content": "dev rules"}, {"role": "user", "content": "go"}]},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", required=True, help="production vllm/tokenizers/deepseek_v4_encoding.py")
    ap.add_argument("out")
    a = ap.parse_args()
    enc = load_encoding(a.encoding)
    out = {"source": "vLLM v0.33 deepseek_v4_encoding.py + deepseek_v4.py request mapping", "cases": []}
    for case in CASES:
        request = {k: v for k, v in case.items() if k != "name"}
        out["cases"].append({"name": case["name"], "request": request, "prompt": render(enc, request)})
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"wrote {len(out['cases'])} cases to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
