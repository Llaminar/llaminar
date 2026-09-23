#!/usr/bin/env python3
"""Exercise and authenticate real OpenAI-compatible tool round trips over HTTP.

The server must generate a schema-valid call, preserve its identity through a
tool-result message, and consume an answer absent from the original prompt.
Both ordinary JSON and SSE use the same semantic oracle. Saved raw exchanges
are revalidated by the outer E2E driver; a shell success or a passed flag alone
cannot manufacture tool coverage. This observer never executes arbitrary model
commands: its only tool is the fixed, local receipt oracle below.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request

TOOL_NAME = "lookup_inventory"
ARGUMENTS = {"sku": "cobalt-widget"}
RECEIPT = "CEDAR-7429"
TOOL = {"type": "function", "function": {
    "name": TOOL_NAME, "description": "Look up the receipt code for an inventory item.",
    "parameters": {"type": "object", "properties": {"sku": {"type": "string"}},
                   "required": ["sku"], "additionalProperties": False}}}
MESSAGES = [
    {"role": "system", "content": "Use lookup_inventory to look up the requested item. "
     "Never invent its receipt code. After receiving the tool result, reply with only its receipt code."},
    {"role": "user", "content": "Look up the inventory receipt for SKU cobalt-widget using the tool."},
]
PROBES = (("required_json", False, "required"), ("required_sse", True, "required"),
          ("named_json", False, "named"), ("auto_sse", True, "auto"))


def request_body(stream: bool, choice: str, messages: list | None = None) -> dict:
    """Create the immutable public request, with no parser- or backend-specific controls."""
    tool_choice = {"type": "function", "function": {"name": TOOL_NAME}} if choice == "named" else choice
    return {"messages": MESSAGES if messages is None else messages, "tools": [TOOL],
            "tool_choice": tool_choice, "stream": stream, "enable_thinking": False,
            "temperature": 0, "seed": 42, "max_tokens": 256}


def response_message(exchange: dict, stream: bool) -> tuple[dict, str]:
    """Validate JSON or assemble indexed SSE deltas without hiding malformed terminal state."""
    if not isinstance(exchange, dict) or exchange.get("status") != 200:
        raise ValueError("tool request did not return HTTP 200")
    raw = exchange.get("body")
    if not isinstance(raw, str):
        raise ValueError("missing raw tool response")
    if not stream:
        response = json.loads(raw)
        if not isinstance(response, dict):
            raise ValueError("tool response must be a JSON object")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("tool response requires one choice")
        choice = choices[0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ValueError("tool response omitted the assistant message")
        return choice["message"], choice.get("finish_reason")
    if not exchange.get("content_type", "").startswith("text/event-stream"):
        raise ValueError("streaming tool request did not return SSE")
    message, calls, finish, done = {"role": "assistant", "content": ""}, {}, None, False
    for line in raw.splitlines():
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:") or done:
            raise ValueError("malformed SSE or data after DONE")
        payload = line[5:].strip()
        if payload == "[DONE]":
            if finish is None:
                raise ValueError("SSE DONE preceded its finish reason")
            done = True
            continue
        event = json.loads(payload)
        if not isinstance(event, dict) or "error" in event or not isinstance(event.get("choices"), list):
            raise ValueError("streaming tool error or missing choices")
        if not event["choices"]:
            continue  # Optional usage-only event is not a second terminal choice.
        if len(event["choices"]) != 1 or finish is not None:
            raise ValueError("streaming tool response has extra choices after completion")
        choice = event["choices"][0]
        if not isinstance(choice, dict):
            raise ValueError("streaming tool choice must be a JSON object")
        delta = choice.get("delta")
        if not isinstance(delta, dict) or delta.get("role", "assistant") != "assistant":
            raise ValueError("invalid assistant tool delta")
        for field in ("content", "reasoning_content"):
            value = delta.get(field)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError("non-text assistant delta")
                message[field] = message.get(field, "") + value
        fragments = delta.get("tool_calls", [])
        if not isinstance(fragments, list):
            raise ValueError("tool deltas must be an indexed array")
        for fragment in fragments:
            if not isinstance(fragment, dict) or not isinstance(fragment.get("function", {}), dict):
                raise ValueError("tool function delta must be a JSON object")
            index = fragment.get("index")
            if type(index) is not int or index < 0:
                raise ValueError("tool delta omitted its nonnegative index")
            call = calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            if fragment.get("type", "function") != "function":
                raise ValueError("unknown streaming tool type")
            if fragment.get("id"):
                if call["id"] and call["id"] != fragment["id"]:
                    raise ValueError("streaming tool identity changed")
                call["id"] = fragment["id"]
            for field in ("name", "arguments"):
                value = fragment.get("function", {}).get(field, "")
                if not isinstance(value, str):
                    raise ValueError("non-text streaming function delta")
                call["function"][field] += value
        finish = choice.get("finish_reason")
    if not done:
        raise ValueError("streaming tool response omitted DONE")
    if calls:
        if set(calls) != set(range(len(calls))):
            raise ValueError("streaming tool indices are not contiguous")
        message["tool_calls"] = [calls[index] for index in range(len(calls))]
    return message, finish


def required_call(exchange: dict, stream: bool) -> dict:
    """Require the actual generated call and exact schema arguments, not leaked markup."""
    message, finish = response_message(exchange, stream)
    calls = message.get("tool_calls")
    if message.get("role") != "assistant" or finish != "tool_calls" or not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("model did not complete exactly one tool call")
    call = calls[0]
    if (not isinstance(call, dict) or not isinstance(call.get("function"), dict)
            or not isinstance(call.get("id"), str) or not call["id"] or call.get("type") != "function"
            or call.get("function", {}).get("name") != TOOL_NAME
            or not isinstance(call["function"].get("arguments"), str)
            or json.loads(call["function"]["arguments"]) != ARGUMENTS):
        raise ValueError("tool name, identity or arguments do not match the requested schema")
    if any(marker in (message.get("content") or "") for marker in ("<tool_call>", "<function=", "<parameter=")):
        raise ValueError("native tool protocol leaked into assistant content")
    return {"id": call["id"], "type": "function", "function": dict(call["function"])}


def followup_body(call: dict, stream: bool) -> dict:
    """Join the tool response to its actual generated call ID, keeping the oracle out of the prompt."""
    messages = [*MESSAGES, {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"],
                 "content": json.dumps({"receipt": RECEIPT, "quantity": 17})}]
    return request_body(stream, "none", messages)


def validate_probe(row: dict, probe: tuple) -> None:
    """Recompute the complete two-turn oracle from retained HTTP bytes and request bodies."""
    name, stream, choice = probe
    if row.get("name") != name or row.get("request") != request_body(stream, choice):
        raise ValueError("tool probe changed its canonical request")
    call = required_call(row["response"], stream)
    if row.get("followup_request") != followup_body(call, stream):
        raise ValueError("tool result did not join the generated call identity")
    message, finish = response_message(row["followup_response"], stream)
    if (finish != "stop" or message.get("role") != "assistant" or message.get("tool_calls")
            or (message.get("content") or "").strip() != RECEIPT):
        raise ValueError("assistant did not consume the tool receipt and finish naturally")


def validate_evidence(document: dict) -> None:
    """Reject missing/partial coverage and forged passed flags at outer certification admission."""
    if not isinstance(document, dict):
        raise ValueError("HTTP tool-calling evidence must be a JSON object")
    rows = document.get("results")
    if (document.get("schema") != 1 or document.get("complete") is not True
            or not isinstance(rows, list) or len(rows) != len(PROBES)):
        raise ValueError("missing or incomplete HTTP tool-calling evidence")
    for row, probe in zip(rows, PROBES):
        if not isinstance(row, dict) or row.get("passed") is not True:
            raise ValueError("failed HTTP tool-calling probe")
        try:
            validate_probe(row, probe)
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("malformed HTTP tool-calling evidence") from error


def exchange(base_url: str, body: dict, timeout: int) -> dict:
    """Perform one bounded real HTTP request; retain raw SSE rather than a synthesized answer."""
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return {"status": response.status, "content_type": response.headers.get("Content-Type", ""),
                "body": response.read().decode("utf-8")}


def publish(path: Path, document: dict) -> None:
    """Atomically preserve completed probes if a later request stalls or is cancelled."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as temporary:
        pending = Path(temporary.name)
        try:
            json.dump(document, temporary, indent=2)
            temporary.flush()
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)


def run_checks(base_url: str, output: Path, timeout: int) -> dict:
    """Run independent probes once; never retry a bad answer or weaken a selected protocol."""
    document = {"schema": 1, "complete": False, "results": []}
    publish(output, document)
    for probe in PROBES:
        name, stream, choice = probe
        row = {"name": name, "passed": False, "request": request_body(stream, choice)}
        document["results"].append(row)
        publish(output, document)
        started = time.monotonic()
        try:
            row["response"] = exchange(base_url, row["request"], timeout)
            call = required_call(row["response"], stream)
            row["followup_request"] = followup_body(call, stream)
            row["followup_response"] = exchange(base_url, row["followup_request"], timeout)
            validate_probe(row, probe)
            row["passed"] = True
        except (ValueError, KeyError, TypeError, OSError) as error:
            row["error"] = str(error)
        row["elapsed_seconds"] = time.monotonic() - started
        publish(output, document)
        print(f"[tool-calling] {'PASS' if row['passed'] else 'FAIL'} {name}: {row.get('error', 'round trip verified')}", flush=True)
    document["complete"] = True
    publish(output, document)
    return document


def main() -> int:
    """Drive a live server only; its lifecycle and image identity remain harness-owned."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-timeout", type=int, default=120)
    args = parser.parse_args()
    if args.request_timeout <= 0:
        parser.error("request timeout must be positive")
    result = run_checks(args.base_url, args.output, args.request_timeout)
    return 0 if all(row["passed"] for row in result["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
