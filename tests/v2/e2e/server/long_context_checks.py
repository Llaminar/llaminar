#!/usr/bin/env python3
"""Objective long-context checks for a live Llaminar server endpoint.

This helper is intended to be invoked by the server E2E shell harness after
`llaminar2 serve` is already running. It posts OpenAI-compatible chat
completion requests to `/v1/chat/completions`, keeps thinking disabled for
deterministic checks, and verifies long-context behavior using exact recall,
strict JSON parsing, loose long-generation degeneration metrics, cache-reset
probing, and context-boundary requests.

Example:
    python3 tests/v2/e2e/server/long_context_checks.py \
      --base-url http://127.0.0.1:19080 \
      --tag qwen2.5/cpu \
      --tier lite \
      --min-prompt-tokens 900 \
      --long-max-tokens 512 \
      --context-length 4096 \
      --request-timeout 180

The helper uses only the Python standard library and prints concise PASS/FAIL
lines. It continues after independent failures when practical and exits nonzero
if any check fails.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import pathlib
import re
import socket
import sys
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable


CHAT_PATH = "/v1/chat/completions"
EXPECTED_JSON_KEYS = {"alpha", "middle", "omega"}
MIN_RECALL_RECORDS = 32
ESTIMATED_TOKENS_PER_RECALL_RECORD = 42
CODE_WORDS_A = (
    "amber",
    "basil",
    "cedar",
    "delta",
    "ember",
    "fable",
    "garnet",
    "harbor",
    "iris",
    "juniper",
    "kelp",
    "laurel",
)
CODE_WORDS_B = (
    "atlas",
    "beacon",
    "cobalt",
    "dawn",
    "elm",
    "fjord",
    "grove",
    "haven",
    "ion",
    "jasmine",
    "keystone",
    "lagoon",
)
CODE_WORDS_C = (
    "north",
    "south",
    "east",
    "west",
    "upper",
    "lower",
    "inner",
    "outer",
    "prime",
    "quiet",
    "rapid",
    "steady",
)
ERROR_STRINGS = (
    "invalid_request_error",
    "traceback",
    "segmentation fault",
    "cuda error",
    "hip error",
    "runtime error",
    "server error",
)


def content_looks_like_server_error(content: str) -> str | None:
    """Return the matching server-error marker when the whole content looks like an error.

    The structured-generation prompt asks for sentences about reliable inference, so a
    healthy model may legitimately write phrases such as "runtime error" inside a
    numbered report. Treat these phrases as harness failures only when the response
    does not contain any numbered report lines and therefore looks like an error body
    accidentally surfaced as assistant content.
    """

    if numbered_lines(content):
        return None

    lower_content = content.strip().lower()
    if not lower_content:
        return None

    for error_string in ERROR_STRINGS:
        if error_string in lower_content:
            return error_string
    return None


class CheckError(Exception):
    """Raised when a single objective check fails."""


@dataclasses.dataclass(frozen=True)
class HttpResponse:
    """HTTP response wrapper that preserves status and JSON parse results."""

    status: int
    body: str
    data: Any
    error: str = ""


@dataclasses.dataclass(frozen=True)
class ChatResult:
    """Validated chat completion fields used by the long-context checks."""

    content: str
    finish_reason: str
    usage: dict[str, int]
    response: HttpResponse


@dataclasses.dataclass(frozen=True)
class TierSettings:
    """Tier-specific thresholds that keep lite cheaper than full."""

    min_numbered_lines: int
    requested_numbered_lines: int
    min_completion_tokens: int
    boundary_close_ratio: float


class CheckRunner:
    """Collects failures while printing concise harness-friendly status lines."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.failures: list[str] = []

    def pass_(self, name: str, detail: str) -> None:
        print(f"PASS [{self.tag}] {name}: {detail}", flush=True)

    def fail(self, name: str, detail: str) -> None:
        self.failures.append(f"{name}: {detail}")
        print(f"FAIL [{self.tag}] {name}: {detail}", flush=True)

    def run(self, name: str, check: Callable[[], str]) -> None:
        try:
            detail = check()
        except CheckError as exc:
            self.fail(name, str(exc))
        except (RuntimeError, ValueError, TypeError, KeyError, IndexError) as exc:
            self.fail(name, f"unexpected exception: {type(exc).__name__}: {exc}")
        else:
            self.pass_(name, detail)


def parse_boolish(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="Base server URL, for example http://127.0.0.1:19080")
    parser.add_argument("--tag", default="long-context", help="Tag shown in PASS/FAIL output")
    parser.add_argument("--tier", choices=("lite", "full"), default="lite")
    parser.add_argument("--min-prompt-tokens", type=int, default=900)
    parser.add_argument("--long-max-tokens", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument(
        "--thinking-model",
        nargs="?",
        const="true",
        default="false",
        type=parse_boolish,
        help="Accepted for harness compatibility; deterministic checks keep enable_thinking=false.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run small internal tests for pure parsing and metric helpers without contacting a server.",
    )
    args = parser.parse_args(argv)

    if not args.self_test and not args.base_url:
        parser.error("--base-url is required unless --self-test is used")
    if args.min_prompt_tokens <= 0:
        parser.error("--min-prompt-tokens must be positive")
    if args.long_max_tokens <= 0:
        parser.error("--long-max-tokens must be positive")
    if args.context_length <= 0:
        parser.error("--context-length must be positive")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")

    return args


def tier_settings(tier: str, long_max_tokens: int) -> TierSettings:
    usable_max = max(1, long_max_tokens - 1)
    if tier == "lite":
        min_lines = max(8, min(32, long_max_tokens // 16))
        requested_lines = max(50, min(90, long_max_tokens // 6))
        min_completion = max(64, int(long_max_tokens * 0.45))
        return TierSettings(
            min_numbered_lines=min_lines,
            requested_numbered_lines=requested_lines,
            min_completion_tokens=min(min_completion, usable_max),
            boundary_close_ratio=0.70,
        )

    # Full-tier structured generation is primarily an anti-degeneration and
    # long-completion check. Some large MoE models spend more tokens per line,
    # so leave headroom while still requiring sustained numbered output.
    min_lines = max(40, min(120, long_max_tokens // 18))
    requested_lines = max(140, min(220, long_max_tokens // 6))
    min_completion = max(256, int(long_max_tokens * 0.65))
    return TierSettings(
        min_numbered_lines=min_lines,
        requested_numbered_lines=requested_lines,
        min_completion_tokens=min(min_completion, usable_max),
        boundary_close_ratio=0.80,
    )


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: float) -> HttpResponse:
    url = base_url.rstrip("/") + path
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            response_body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        response_body = exc.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        return HttpResponse(status=0, body="", data=None, error=str(exc))

    try:
        data = json.loads(response_body)
    except json.JSONDecodeError:
        data = None

    return HttpResponse(status=status, body=response_body, data=data)


def make_chat_payload(messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    return {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "enable_thinking": False,
    }


def preview(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def artifact_safe_name(text: str) -> str:
    """Return a filesystem-safe identifier for failed-response artifacts."""

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return safe.strip("_")[:96] or "long_context"


def write_failed_content_artifact(args: argparse.Namespace, check_name: str, content: str) -> str:
    """Persist failed assistant content when the shell harness provides a directory."""

    artifact_dir = os.environ.get("LLAMINAR_E2E_LONG_CONTEXT_ARTIFACT_DIR")
    if not artifact_dir:
        return ""

    path = pathlib.Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    artifact_path = path / (
        f"{artifact_safe_name(args.tag)}_{artifact_safe_name(check_name)}_failed_content.txt"
    )
    artifact_path.write_text(content, encoding="utf-8")
    return str(artifact_path)


def raise_with_content_artifact(
    args: argparse.Namespace,
    check_name: str,
    message: str,
    content: str,
) -> None:
    """Raise CheckError and include a durable artifact path when available."""

    artifact = write_failed_content_artifact(args, check_name, content)
    if artifact:
        raise CheckError(f"{message}; artifact={artifact}")
    raise CheckError(message)


def require_int_field(mapping: dict[str, Any], field: str, minimum: int = 0) -> int:
    value = mapping.get(field)
    if not isinstance(value, int):
        raise CheckError(f"usage.{field} missing or non-integer")
    if value < minimum:
        raise CheckError(f"usage.{field}={value} below {minimum}")
    return value


def validate_chat_completion(
    response: HttpResponse,
    expected_context_length: int,
    allowed_finish_reasons: set[str],
) -> ChatResult:
    if response.status != 200:
        detail = response.error or preview(response.body)
        raise CheckError(f"expected HTTP 200, got {response.status}: {detail}")
    if not isinstance(response.data, dict):
        raise CheckError("response body is not valid JSON object")

    choices = response.data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise CheckError("response.choices missing or empty")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise CheckError("response.choices[0] is not an object")

    message = choice.get("message")
    if not isinstance(message, dict):
        raise CheckError("choices[0].message missing or non-object")
    content = message.get("content")
    if not isinstance(content, str):
        raise CheckError("choices[0].message.content missing or non-string")
    if not content.strip():
        raise CheckError("assistant content is empty")

    finish_reason = choice.get("finish_reason")
    if finish_reason not in allowed_finish_reasons:
        expected = ",".join(sorted(allowed_finish_reasons))
        raise CheckError(f"finish_reason={finish_reason!r}, expected one of {{{expected}}}")

    usage_obj = response.data.get("usage")
    if not isinstance(usage_obj, dict):
        raise CheckError("response.usage missing or non-object")
    usage = {
        "prompt_tokens": require_int_field(usage_obj, "prompt_tokens", 1),
        "completion_tokens": require_int_field(usage_obj, "completion_tokens", 1),
        "total_tokens": require_int_field(usage_obj, "total_tokens", 1),
        "context_window": require_int_field(usage_obj, "context_window", 1),
        "context_used": require_int_field(usage_obj, "context_used", 1),
    }
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise CheckError(
            "usage.total_tokens does not equal prompt_tokens + completion_tokens "
            f"({usage['total_tokens']} != {usage['prompt_tokens']} + {usage['completion_tokens']})"
        )
    if usage["context_window"] != expected_context_length:
        raise CheckError(
            f"usage.context_window={usage['context_window']}, expected {expected_context_length}"
        )
    if usage["context_used"] > usage["context_window"]:
        raise CheckError(
            f"usage.context_used={usage['context_used']} exceeds context_window={usage['context_window']}"
        )

    return ChatResult(content=content, finish_reason=str(finish_reason), usage=usage, response=response)


def deterministic_code(namespace: str, index: int) -> str:
    seed = sum(ord(char) for char in namespace)
    first = CODE_WORDS_A[(index + seed) % len(CODE_WORDS_A)]
    second = CODE_WORDS_B[((index // len(CODE_WORDS_A)) + seed) % len(CODE_WORDS_B)]
    third = CODE_WORDS_C[((index // (len(CODE_WORDS_A) * len(CODE_WORDS_B))) + seed) % len(CODE_WORDS_C)]
    return f"{first} {second} {third}"


def record_count_for_context(min_prompt_tokens: int, context_length: int, max_tokens: int, tier: str) -> int:
    safety_margin = max(512, context_length // 8)
    prompt_budget = max(0, context_length - max_tokens - safety_margin)

    tier_extra_tokens = 900 if tier == "full" else 360
    context_bonus_rate = 0.18 if tier == "full" else 0.10
    context_bonus_tokens = int(min(max(0, context_length - 4096), 2048) * context_bonus_rate)
    target_prompt_tokens = min(prompt_budget, min_prompt_tokens + tier_extra_tokens + context_bonus_tokens)

    by_prompt_target = int(target_prompt_tokens / ESTIMATED_TOKENS_PER_RECALL_RECORD)
    by_context_cap = int(prompt_budget / ESTIMATED_TOKENS_PER_RECALL_RECORD)
    if by_context_cap < MIN_RECALL_RECORDS:
        return MIN_RECALL_RECORDS
    return max(MIN_RECALL_RECORDS, min(by_prompt_target, by_context_cap))


def make_audit_record(index: int, code: str, namespace: str) -> str:
    checksum = 1000 + ((index * 37 + len(namespace) * 97) % 9000)
    lane = chr(ord("A") + (index % 6))
    return (
        f"Ledger filler {index:04d}: distractor audit value \"{code}\". "
        f"Lane {lane}; checksum {checksum}; status normal; this is not the requested value."
    )


def build_needle_prompt(
    placement: str,
    min_prompt_tokens: int,
    context_length: int,
    max_tokens: int,
    tier: str,
) -> tuple[list[dict[str, str]], str, list[str], int]:
    count = record_count_for_context(min_prompt_tokens, context_length, max_tokens, tier)
    target_index_by_placement = {
        "beginning": min(3, count - 1),
        "middle": count // 2,
        "end": max(0, count - 5),
    }
    target_index = target_index_by_placement[placement]
    namespace = f"LCN-{placement[:3].upper()}"
    target_key_by_placement = {
        "beginning": "alpha",
        "middle": "middle",
        "end": "omega",
    }
    target_value_by_placement = {
        "beginning": "LCJSON-ALPHA-314159",
        "middle": "LCJSON-MIDDLE-271828",
        "end": "LCJSON-OMEGA-161803",
    }
    target_key = target_key_by_placement[placement]
    target_value = target_value_by_placement[placement]

    records: list[str] = []
    codes: list[str] = []
    for index in range(count):
        code = target_value if index == target_index else deterministic_code(namespace, index)
        codes.append(code)
        if index == target_index:
            records.append(
                f"Ledger item {index:04d}: REQUIRED_JSON_FIELD {target_key} has exact value {code}. "
                "This is the only requested field."
            )
        else:
            records.append(make_audit_record(index, code, namespace))

    neighbor_indices = sorted(
        idx for idx in (target_index - 2, target_index - 1, target_index + 1, target_index + 2)
        if 0 <= idx < count
    )
    distractors = [codes[idx] for idx in neighbor_indices]

    user_prompt = "\n".join(
        [
            "Task: read the ledger and return one minified JSON object.",
            "The only allowed key is answer.",
            f"Only the REQUIRED_JSON_FIELD named {target_key} matters for the final answer.",
            *records,
            "Return exactly one minified JSON object and no prose.",
            'The object shape is {"answer":"VALUE_FROM_LEDGER"}.',
            f"Use the exact REQUIRED_JSON_FIELD value for {target_key} from the ledger.",
        ]
    )
    messages = [
        {"role": "system", "content": "You are a strict JSON renderer. Output JSON only."},
        {"role": "user", "content": user_prompt},
    ]
    return messages, codes[target_index], distractors, count


def needle_max_tokens(long_max_tokens: int) -> int:
    return min(128, max(64, long_max_tokens // 2))


def run_needle_check(args: argparse.Namespace, placement: str) -> str:
    max_tokens = needle_max_tokens(args.long_max_tokens)
    messages, target_code, distractors, count = build_needle_prompt(
        placement,
        args.min_prompt_tokens,
        args.context_length,
        max_tokens,
        args.tier,
    )
    response = post_json(args.base_url, CHAT_PATH, make_chat_payload(messages, max_tokens), args.request_timeout)
    result = validate_chat_completion(response, args.context_length, {"stop", "length"})

    if result.usage["prompt_tokens"] < args.min_prompt_tokens:
        raise CheckError(
            f"prompt_tokens={result.usage['prompt_tokens']} below target {args.min_prompt_tokens}; "
            f"records={count}"
        )

    check_name = f"long_needle_recall_{placement}"
    if target_code not in result.content:
        raise_with_content_artifact(
            args,
            check_name,
            f"target code {target_code} missing from content: {preview(result.content)}",
            result.content,
        )
    leaked = [code for code in distractors if code in result.content]
    if leaked:
        raise_with_content_artifact(
            args,
            check_name,
            f"nearby distractor code(s) appeared: {','.join(leaked)}",
            result.content,
        )

    return (
        f"target={target_code} prompt_tokens={result.usage['prompt_tokens']} "
        f"finish={result.finish_reason}"
    )


def build_multi_needle_prompt(
    min_prompt_tokens: int,
    context_length: int,
    max_tokens: int,
    tier: str,
) -> tuple[list[dict[str, str]], dict[str, str], int]:
    count = record_count_for_context(min_prompt_tokens, context_length, max_tokens, tier)
    sentinels = {
        "alpha": "LCJSON-ALPHA-314159",
        "middle": "LCJSON-MIDDLE-271828",
        "omega": "LCJSON-OMEGA-161803",
    }
    positions = {
        "alpha": min(3, count - 1),
        "middle": count // 2,
        "omega": max(0, count - 4),
    }

    lines: list[str] = [
        "Task: read the ledger and return one minified JSON object.",
        "The only allowed keys are alpha, middle, and omega.",
        "Every allowed key must be present exactly once. Never use an empty key.",
        "The ledger contains many filler facts plus three named sentinel values.",
    ]
    for index in range(count):
        inserted = False
        for key, position in positions.items():
            if index == position:
                lines.append(
                    f"Ledger item {index:04d}: REQUIRED_JSON_FIELD {key} has exact value {sentinels[key]}."
                )
                inserted = True
                break
        if not inserted:
            code = deterministic_code("LCJSON-FILL", index)
            lines.append(
                f"Ledger item {index:04d}: filler code {code}; phase stable; checksum {7000 + index}."
            )
    lines.extend(
        [
            "Return exactly one minified JSON object and no prose.",
            'The object shape is {"alpha":"VALUE_FROM_LEDGER","middle":"VALUE_FROM_LEDGER","omega":"VALUE_FROM_LEDGER"}.',
            "Use the exact REQUIRED_JSON_FIELD values from the ledger.",
            "Do not omit omega. Do not add keys. Do not use markdown fences.",
        ]
    )
    messages = [
        {"role": "system", "content": "You are a strict JSON renderer. Output JSON only."},
        {"role": "user", "content": "\n".join(lines)},
    ]
    return messages, sentinels, count


def strip_optional_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def run_multi_needle_json(args: argparse.Namespace) -> str:
    max_tokens = min(128, max(80, args.long_max_tokens // 2))
    messages, sentinels, count = build_multi_needle_prompt(
        args.min_prompt_tokens,
        args.context_length,
        max_tokens,
        args.tier,
    )
    response = post_json(args.base_url, CHAT_PATH, make_chat_payload(messages, max_tokens), args.request_timeout)
    result = validate_chat_completion(response, args.context_length, {"stop", "length"})
    if result.usage["prompt_tokens"] < args.min_prompt_tokens:
        raise CheckError(
            f"prompt_tokens={result.usage['prompt_tokens']} below target {args.min_prompt_tokens}; "
            f"records={count}"
        )

    json_text = strip_optional_code_fence(result.content)
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise_with_content_artifact(
            args,
            "multi_needle_strict_json_recall",
            f"assistant content is not strict JSON: {exc.msg}; content={preview(result.content)}",
            result.content,
        )
    if not isinstance(parsed, dict):
        raise_with_content_artifact(
            args,
            "multi_needle_strict_json_recall",
            f"JSON root is {type(parsed).__name__}, expected object",
            result.content,
        )
    if set(parsed.keys()) != EXPECTED_JSON_KEYS:
        raise_with_content_artifact(
            args,
            "multi_needle_strict_json_recall",
            f"JSON keys {sorted(parsed.keys())}, expected {sorted(EXPECTED_JSON_KEYS)}; "
            f"content={preview(result.content)}",
            result.content,
        )
    mismatches = [key for key, value in sentinels.items() if parsed.get(key) != value]
    if mismatches:
        detail = ", ".join(f"{key}={parsed.get(key)!r}" for key in mismatches)
        raise_with_content_artifact(
            args,
            "multi_needle_strict_json_recall",
            f"sentinel mismatch: {detail}",
            result.content,
        )

    return f"prompt_tokens={result.usage['prompt_tokens']} finish={result.finish_reason}"


def structured_prompt(settings: TierSettings) -> tuple[list[dict[str, str]], str]:
    cache_sentinel = "LCACHE-RESET-SENTINEL-593821"
    lines = [
        f"Control marker for cache isolation: {cache_sentinel}.",
        "Do not copy the control marker into the report.",
        f"Write {settings.requested_numbered_lines} numbered lines.",
        "Each line must use this exact format:",
        "001 | one short distinct sentence about reliable inference",
        "002 | one short distinct sentence about reliable inference",
        "Keep each sentence concise and vary the wording.",
        "Never restart numbering; count upward from 001.",
        "After the final requested line, write END_OF_REPORT on its own line.",
        "If the token limit interrupts the report, stop wherever the limit occurs.",
    ]
    messages = [
        {"role": "system", "content": "You produce deterministic machine-checkable reports."},
        {"role": "user", "content": "\n".join(lines)},
    ]
    return messages, cache_sentinel


def numbered_lines(content: str) -> list[int]:
    numbers: list[int] = []
    for match in re.finditer(r"(?m)^\s*(\d{1,4})\s*\|", content):
        try:
            numbers.append(int(match.group(1)))
        except ValueError:
            continue
    return numbers


def word_tokens(content: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9_'-]+", content)]


def repeated_word_run(tokens: list[str]) -> int:
    longest = 0
    current = 0
    previous = None
    for token in tokens:
        if token == previous:
            current += 1
        else:
            current = 1
            previous = token
        longest = max(longest, current)
    return longest


def duplicate_line_ratio(content: str) -> float:
    lines = [" ".join(line.split()).lower() for line in content.splitlines() if line.strip()]
    if not lines:
        return 0.0
    return (len(lines) - len(set(lines))) / len(lines)


def unique_ngram_ratio(tokens: list[str], size: int) -> float | None:
    if len(tokens) <= size:
        return None
    ngrams = [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]
    return len(set(ngrams)) / len(ngrams)


def common_phrase_coverage(tokens: list[str], size: int) -> float | None:
    if len(tokens) < size * 3:
        return None
    counter = collections.Counter(
        tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)
    )
    if not counter:
        return None
    return (counter.most_common(1)[0][1] * size) / len(tokens)


def validate_number_progression(numbers: list[int]) -> tuple[bool, str]:
    if len(numbers) < 4:
        return True, "short sequence"

    increases = sum(1 for left, right in zip(numbers, numbers[1:]) if right > left)
    resets = sum(1 for left, right in zip(numbers, numbers[1:]) if right <= left or (left > 10 and right <= 3))
    increase_ratio = increases / (len(numbers) - 1)
    reset_cap = max(2, len(numbers) // 10)
    if increase_ratio < 0.75:
        return False, f"line numbers increase only {increase_ratio:.2f} of the time"
    if resets > reset_cap:
        return False, f"line numbers reset {resets} times; cap {reset_cap}"
    return True, f"increase_ratio={increase_ratio:.2f} resets={resets}"


def degeneration_failures(content: str) -> tuple[list[str], dict[str, float | int | None]]:
    failures: list[str] = []
    metrics: dict[str, float | int | None] = {}

    char_match = re.search(r"([^\s])\1{32,}", content)
    if char_match:
        failures.append("repeated character run exceeds 32")

    tokens = word_tokens(content)
    max_word_run = repeated_word_run(tokens)
    metrics["max_word_run"] = max_word_run
    if max_word_run > 12:
        failures.append(f"repeated word run {max_word_run} exceeds 12")

    fourgram_ratio = unique_ngram_ratio(tokens, 4) if len(tokens) > 200 else None
    metrics["unique_4gram_ratio"] = fourgram_ratio
    if fourgram_ratio is not None and fourgram_ratio < 0.20:
        failures.append(f"unique 4-gram ratio {fourgram_ratio:.2f} below 0.20")

    dup_ratio = duplicate_line_ratio(content)
    metrics["duplicate_line_ratio"] = dup_ratio
    if dup_ratio > 0.30:
        failures.append(f"duplicate-line ratio {dup_ratio:.2f} above 0.30")

    phrase_coverage = common_phrase_coverage(tokens, 8)
    metrics["common_8token_phrase_coverage"] = phrase_coverage
    if phrase_coverage is not None and phrase_coverage > 0.20:
        failures.append(f"common 8-token phrase coverage {phrase_coverage:.2f} above 0.20")

    return failures, metrics


def run_structured_generation(args: argparse.Namespace, settings: TierSettings) -> tuple[str, str]:
    messages, cache_sentinel = structured_prompt(settings)
    response = post_json(
        args.base_url,
        CHAT_PATH,
        make_chat_payload(messages, args.long_max_tokens),
        args.request_timeout,
    )
    result = validate_chat_completion(response, args.context_length, {"stop", "length"})
    content = result.content

    if "\ufffd" in content:
        raise CheckError("replacement character U+FFFD appeared in content")

    line_numbers = numbered_lines(content)
    error_string = content_looks_like_server_error(content)
    if error_string is not None:
        raise_with_content_artifact(
            args,
            "structured_long_generation",
            f"server error-like string appeared in content: {error_string}",
            content,
        )
    if len(line_numbers) < settings.min_numbered_lines:
        raise_with_content_artifact(
            args,
            "structured_long_generation",
            f"only {len(line_numbers)} numbered lines; expected at least {settings.min_numbered_lines}",
            content,
        )
    progress_ok, progress_detail = validate_number_progression(line_numbers)
    if not progress_ok:
        raise_with_content_artifact(args, "structured_long_generation", progress_detail, content)

    full_structure_completed = "END_OF_REPORT" in content and len(line_numbers) >= settings.min_numbered_lines
    if result.usage["completion_tokens"] < settings.min_completion_tokens and not full_structure_completed:
        raise_with_content_artifact(
            args,
            "structured_long_generation",
            f"completion_tokens={result.usage['completion_tokens']} below target "
            f"{settings.min_completion_tokens}; finish={result.finish_reason}",
            content,
        )

    degeneration_errors, metrics = degeneration_failures(content)
    if degeneration_errors:
        raise_with_content_artifact(
            args,
            "structured_long_generation",
            "; ".join(degeneration_errors),
            content,
        )

    metric_bits = [
        f"lines={len(line_numbers)}",
        f"completion_tokens={result.usage['completion_tokens']}",
        f"finish={result.finish_reason}",
        progress_detail,
        f"dup_lines={metrics['duplicate_line_ratio']:.2f}",
    ]
    if metrics["unique_4gram_ratio"] is not None:
        metric_bits.append(f"u4={metrics['unique_4gram_ratio']:.2f}")
    return " ".join(metric_bits), cache_sentinel


def run_cache_reset_probe(args: argparse.Namespace, prior_sentinels: Iterable[str]) -> str:
    messages = [
        {"role": "system", "content": "You are a calculator. Reply only with the numeric answer."},
        {"role": "user", "content": "What is 6+7? Reply with only 13."},
    ]
    response = post_json(args.base_url, CHAT_PATH, make_chat_payload(messages, 8), args.request_timeout)
    result = validate_chat_completion(response, args.context_length, {"stop", "length"})

    numbers = re.findall(r"-?\d+", result.content)
    if not numbers or numbers[-1] != "13":
        raise CheckError(f"expected numeric answer 13, got {preview(result.content)}")
    leaked = [sentinel for sentinel in prior_sentinels if sentinel and sentinel in result.content]
    if leaked:
        raise CheckError(f"prior sentinel leaked into short response: {','.join(leaked[:3])}")

    prompt_cap = max(128, args.context_length // 4)
    if result.usage["prompt_tokens"] > prompt_cap:
        raise CheckError(
            f"short prompt usage too large after long request: {result.usage['prompt_tokens']} > {prompt_cap}"
        )
    return f"answer=13 prompt_tokens={result.usage['prompt_tokens']} finish={result.finish_reason}"


def boundary_messages(word_count: int, answer: str) -> list[dict[str, str]]:
    padding = " ".join("blue" for _ in range(word_count))
    user = (
        "Boundary probe. The repeated padding words are inert and should be ignored.\n"
        f"Padding begins: {padding}\n"
        f"Padding ends. Reply with only {answer}."
    )
    return [
        {"role": "system", "content": "You answer boundary probes exactly and briefly."},
        {"role": "user", "content": user},
    ]


def run_valid_boundary(args: argparse.Namespace, settings: TierSettings) -> str:
    max_tokens = 8
    starting_words = max(32, int(args.context_length * 0.92))
    word_counts = [max(32, int(starting_words * factor)) for factor in (1.0, 0.88, 0.76, 0.64, 0.52)]
    last_detail = "no request attempted"

    for word_count in word_counts:
        response = post_json(
            args.base_url,
            CHAT_PATH,
            make_chat_payload(boundary_messages(word_count, "BOUNDARY_OK"), max_tokens),
            args.request_timeout,
        )
        if response.status == 400:
            last_detail = f"word_count={word_count} returned HTTP 400"
            continue
        result = validate_chat_completion(response, args.context_length, {"stop", "length"})
        if "BOUNDARY_OK" not in result.content:
            raise CheckError(f"valid boundary response missing BOUNDARY_OK: {preview(result.content)}")
        context_window = result.usage["context_window"]
        context_used = result.usage["context_used"]
        if context_used >= context_window:
            raise CheckError(f"valid boundary context_used={context_used} is not below {context_window}")
        ratio = context_used / context_window
        if ratio < settings.boundary_close_ratio:
            raise CheckError(
                f"valid boundary was not close enough: context_used/window={ratio:.2f}; "
                f"required {settings.boundary_close_ratio:.2f}"
            )
        return f"context_used={context_used}/{context_window} ratio={ratio:.2f} word_count={word_count}"

    raise CheckError(f"could not find a valid near-boundary request; last={last_detail}")


def run_oversized_boundary(args: argparse.Namespace) -> str:
    word_count = max(args.context_length + 512, int(args.context_length * 1.35))
    response = post_json(
        args.base_url,
        CHAT_PATH,
        make_chat_payload(boundary_messages(word_count, "TOO_LARGE_SHOULD_FAIL"), 4),
        args.request_timeout,
    )
    if response.status != 400:
        detail = response.error or preview(response.body)
        raise CheckError(f"expected HTTP 400 for oversized prompt, got {response.status}: {detail}")
    if not isinstance(response.data, dict):
        raise CheckError("oversized error body is not valid JSON object")

    error = response.data.get("error")
    if not isinstance(error, dict):
        raise CheckError("oversized error body missing error object")
    error_type = error.get("type")
    if error_type != "invalid_request_error":
        raise CheckError(f"oversized error type={error_type!r}, expected invalid_request_error")
    error_param = error.get("param")
    if error_param is not None and error_param != "messages":
        raise CheckError(f"oversized error param={error_param!r}, expected messages when present")
    return f"HTTP 400 invalid_request_error word_count={word_count}"


def expected_sentinels(args: argparse.Namespace, settings: TierSettings) -> list[str]:
    sentinels: list[str] = []
    for placement in ("beginning", "middle", "end"):
        max_tokens = needle_max_tokens(args.long_max_tokens)
        _, target, distractors, _ = build_needle_prompt(
            placement,
            args.min_prompt_tokens,
            args.context_length,
            max_tokens,
            args.tier,
        )
        sentinels.append(target)
        sentinels.extend(distractors)
    max_tokens = min(128, max(80, args.long_max_tokens // 2))
    _, json_sentinels, _ = build_multi_needle_prompt(
        args.min_prompt_tokens,
        args.context_length,
        max_tokens,
        args.tier,
    )
    sentinels.extend(json_sentinels.values())
    _, cache_sentinel = structured_prompt(settings)
    sentinels.append(cache_sentinel)
    return sentinels


def run_self_test() -> int:
    assert strip_optional_code_fence('```json\n{"alpha":"a"}\n```') == '{"alpha":"a"}'
    assert strip_optional_code_fence('{"alpha":"a"}') == '{"alpha":"a"}'

    lite_4k_count = record_count_for_context(900, 4096, 128, "lite")
    full_4k_count = record_count_for_context(900, 4096, 128, "full")
    lite_8k_count = record_count_for_context(900, 8192, 128, "lite")
    full_8k_count = record_count_for_context(900, 8192, 128, "full")
    assert 32 <= lite_4k_count <= 48, lite_4k_count
    assert lite_4k_count < full_4k_count <= 64, full_4k_count
    assert lite_4k_count < lite_8k_count <= 64, lite_8k_count
    assert full_4k_count < full_8k_count <= 80, full_8k_count

    failures, _ = degeneration_failures("A" * 40)
    assert any("repeated character" in failure for failure in failures)

    repeated_text = "\n".join("001 | same line" for _ in range(10))
    failures, metrics = degeneration_failures(repeated_text)
    assert metrics["duplicate_line_ratio"] > 0.30
    assert any("duplicate-line" in failure for failure in failures)

    assert content_looks_like_server_error("runtime error: graph replay failed") == "runtime error"
    assert content_looks_like_server_error("001 | Reliable inference handles runtime errors safely.") is None

    ok, _ = validate_number_progression([1, 2, 3, 4, 5, 6])
    assert ok
    ok, detail = validate_number_progression([1, 2, 1, 2, 1, 2, 1])
    assert not ok, detail

    print("PASS [self-test] helper pure-function checks", flush=True)
    return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()

    runner = CheckRunner(args.tag)
    settings = tier_settings(args.tier, args.long_max_tokens)
    sentinels = expected_sentinels(args, settings)

    for placement in ("beginning", "middle", "end"):
        runner.run(
            f"long needle recall {placement}",
            lambda placement=placement: run_needle_check(args, placement),
        )

    runner.run("multi-needle strict JSON recall", lambda: run_multi_needle_json(args))

    structured_sentinel_holder = {"detail": ""}

    def structured_check() -> str:
        detail, cache_sentinel = run_structured_generation(args, settings)
        structured_sentinel_holder["detail"] = cache_sentinel
        return detail

    runner.run("structured long generation", structured_check)
    if structured_sentinel_holder["detail"] and structured_sentinel_holder["detail"] not in sentinels:
        sentinels.append(structured_sentinel_holder["detail"])

    runner.run("cache-reset probe", lambda: run_cache_reset_probe(args, sentinels))
    runner.run("valid near-boundary context", lambda: run_valid_boundary(args, settings))
    runner.run("oversized context rejection", lambda: run_oversized_boundary(args))

    if runner.failures:
        print(f"FAIL [{args.tag}] summary: {len(runner.failures)} check(s) failed", flush=True)
        return 1

    print(f"PASS [{args.tag}] summary: all long-context checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
