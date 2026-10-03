#!/usr/bin/env python3
"""Replay a preserved public HTTP sequence until its first failure.

The existing server harness owns placement, readiness, driver observation and
retirement. This diagnostic changes only the HTTP workload. It never retries a
failed request, edits model policy, or produces an E2E certificate. Every request
is published before submission, including SSE requests, so a native stall leaves
an unambiguous replay point and its partial response bytes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid
from typing import Callable

from stochastic_mtp_scenarios import decode_response
from long_context_checks import build_needle_prompt

# Reuse the routine generation consumer: the hammer must not invent weaker
# definitions of full/partial KV, GDN or shifted MTP-state restoration.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts/ci"))
from generation_regression_http import (MTPPolicy, PrefixProbe, PrefixState,
                                       validate_partial_prompt, validate_prefix_outcome)
from generation_tokens import TokenTrace, compare_tokens


class ReplayState(str, Enum):
    """Explicit diagnostic lifecycle; none of these states certifies a model."""

    READY = "ready"
    REQUEST = "request_in_flight"
    FAILED = "failed"
    STOPPED = "bounded_diagnostic_complete"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class ReplayRequest:
    """One immutable captured request and an independently specified oracle."""

    name: str
    body: dict
    expected_numeric_answer: str | None = None
    expected_json_answer: str | None = None
    prefix_probe: PrefixProbe | None = None
    prefix_source: str | None = None


@dataclass(frozen=True)
class PrefixPressure:
    """Diagnostic workload geometry projected from one canonical model/cell.

    This changes input requests, not cache capacity or inference policy. New
    cycle identities create admission pressure under the existing RAM/disk
    budgets; only completed terminal outcomes count as restoration evidence.
    """

    context_length: int
    minimum_prompt_tokens: int
    max_tokens: int
    prefix_state: PrefixState
    mtp_policy: MTPPolicy

    @classmethod
    def from_document(cls, document: dict) -> PrefixPressure:
        """Reject missing model-state contracts before any model admission."""
        if not isinstance(document, dict):
            raise ValueError("prefix pressure requires canonical geometry and state policy")
        context, minimum, maximum = (document.get(key) for key in (
            "context_length", "minimum_prompt_tokens", "max_tokens"))
        if (any(type(value) is not int or value <= 0 for value in (context, minimum, maximum))
                or minimum + maximum >= context):
            raise ValueError("invalid prefix-pressure context/prompt geometry")
        return cls(context, minimum, maximum, PrefixState(document.get("prefix_state")),
                   MTPPolicy(document.get("mtp_policy")))

    def document(self) -> dict:
        """Retain the immutable projection with the diagnostic configuration."""
        return {"context_length": self.context_length,
                "minimum_prompt_tokens": self.minimum_prompt_tokens,
                "max_tokens": self.max_tokens, "prefix_state": self.prefix_state.value,
                "mtp_policy": self.mtp_policy.value}

    def requests(self, cycle: int, nonce: str) -> tuple[ReplayRequest, ...]:
        """Rotate mature long-needle geometry and explicit restore boundaries.

        The unique early system prefix prevents yesterday's hot entry from
        turning a cold admission into an accidental restore. Extending the
        complete conversation preserves a token boundary; merely changing the
        last user sentence would not prove recurrent-state partial restoration.
        """
        placement = ("beginning", "middle", "end")[(cycle - 1) % 3]
        messages, answer, _, _ = build_needle_prompt(placement, self.minimum_prompt_tokens,
                                                    self.context_length, self.max_tokens, "full")
        messages[0] = {**messages[0], "content":
            f"Prefix diagnostic {nonce}:{cycle}.\n" + messages[0]["content"]}
        body = {"messages": messages, "temperature": 0.0, "seed": 4242,
                "enable_thinking": False, "max_tokens": self.max_tokens, "stream": False,
                "return_token_ids": True, "return_runtime_summary": True}
        # Match the canonical generation partial probe: append the assistant
        # turn, not another user turn. Qwen's template strips historic reasoning
        # once a newer user turn exists; that would rewrite the empty <think>
        # boundary already encoded in the producer's prompt. The token oracle
        # below still verifies this eligibility using actual returned IDs.
        suffix = {**body, "messages": [*messages,
            {"role": "assistant", "content": json.dumps({"answer": answer}, separators=(",", ":"))}]}
        return (ReplayRequest("prefix_cold_long", body, expected_json_answer=answer,
                              prefix_probe=PrefixProbe.FRESH),
                ReplayRequest("prefix_full_long", body, expected_json_answer=answer,
                              prefix_probe=PrefixProbe.FULL, prefix_source="prefix_cold_long"),
                ReplayRequest("prefix_partial_suffix", suffix, expected_json_answer=answer,
                              prefix_probe=PrefixProbe.PARTIAL, prefix_source="prefix_cold_long"),
                ReplayRequest("prefix_full_suffix", suffix, expected_json_answer=answer,
                              prefix_probe=PrefixProbe.FULL, prefix_source="prefix_partial_suffix"),
                ReplayRequest("prefix_full_after_short", body, expected_json_answer=answer,
                              prefix_probe=PrefixProbe.FULL, prefix_source="prefix_cold_long"),
                ReplayRequest("prefix_suffix_after_short", suffix, expected_json_answer=answer,
                              prefix_probe=PrefixProbe.FULL, prefix_source="prefix_partial_suffix"))


def read_sequence(document: dict) -> tuple[ReplayRequest, ...]:
    """Reject incomplete/ambiguous replays before starting a server or sending I/O."""
    if (not isinstance(document, dict) or type(document.get("schema")) is not int
            or document["schema"] != 1):
        raise ValueError("HTTP replay sequence requires schema 1")
    cases = document.get("requests")
    if not isinstance(cases, list) or not cases:
        raise ValueError("HTTP replay sequence is empty")
    requests, names = [], set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("HTTP replay request is malformed")
        name, body, oracle = (case.get("name"), case.get("body"),
                              case.get("expected_numeric_answer"))
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
                or name in names or not isinstance(body, dict)
                or not isinstance(oracle, str) or not re.fullmatch(r"-?\d+", oracle)
                or type(body.get("max_tokens")) is not int or body["max_tokens"] <= 0
                or type(body.get("stream", False)) is not bool
                or not isinstance(body.get("messages"), list) or not body["messages"]):
            raise ValueError("HTTP replay needs unique names, messages, a token budget and a numeric oracle")
        if any(not isinstance(message, dict) or not isinstance(message.get("content"), str)
               or message.get("role") not in ("system", "user", "assistant")
               for message in body["messages"]):
            raise ValueError("HTTP replay message is malformed")
        # NaN/Infinity must fail here, not turn into a different sampler law.
        json.dumps(body, allow_nan=False)
        names.add(name)
        requests.append(ReplayRequest(name, body, oracle))
    return tuple(requests)


def validate_response(request: ReplayRequest, raw: str) -> dict | None:
    """Require a clean terminal, correct arithmetic and the complete SSE envelope."""
    streaming = request.body.get("stream", False)
    content, _ = decode_response(raw, streaming=streaming)
    if streaming:
        lines = [line[5:].strip() for line in raw.splitlines() if line.startswith("data:")]
        if not lines or lines[-1] != "[DONE]" or "[DONE]" in lines[:-1]:
            raise ValueError("SSE response lacks one terminal [DONE]")
        packets = [json.loads(line) for line in lines[:-1]]
        if (not packets or packets[0].get("object") != "chat.completion.chunk"
                or packets[0].get("choices", [{}])[0].get("delta", {}).get("role") != "assistant"
                or any(packet.get("system_fingerprint") != "llaminar-v2" for packet in packets)
                or len({packet.get("id") for packet in packets}) != 1
                or not str(packets[0].get("id", "")).startswith("chatcmpl-")):
            raise ValueError("SSE response has an invalid role/identity envelope")
    else:
        packets = [json.loads(raw)]
    terminal = [choice.get("finish_reason") for packet in packets
                for choice in packet.get("choices", []) if choice.get("finish_reason")]
    if terminal != ["stop"]:
        raise ValueError(f"arithmetic did not finish naturally: {terminal}")
    if request.expected_json_answer is not None:
        if json.loads(content) != {"answer": request.expected_json_answer}:
            raise ValueError(f"incorrect long-needle JSON: {content!r}")
    else:
        numbers = re.findall(r"-?\d+", content)
        if not numbers or numbers[-1] != request.expected_numeric_answer or "</think>" in content:
            raise ValueError(f"incorrect arithmetic/control-tag leakage: {content!r}")
    return None if streaming else packets[0]


def validate_prefix_response(request: ReplayRequest, response: dict, policy: PrefixPressure,
                             previous: dict[str, tuple[TokenTrace, int]], ordinal: int) -> dict:
    """Prove actual restoration and repeat equality from production terminal receipts.

    Returned token IDs authenticate geometry. The first response is an exact
    repeatability baseline, not an independent math oracle: needle recall is
    checked separately against the mature prompt's known answer.
    """
    trace = TokenTrace.from_response(response, requested_max_tokens=request.body["max_tokens"])
    if len(trace.prompt) < policy.minimum_prompt_tokens:
        raise ValueError("prefix pressure prompt is below the canonical long-context minimum")
    source = previous.get(request.prefix_source) if request.prefix_source else None
    if request.prefix_probe is not PrefixProbe.FRESH and source is None:
        raise ValueError("prefix restore omitted its completed source request")
    probe = {"id": request.name, "prefix": request.prefix_probe.value}
    priors = [source[0]] if source else []
    validate_partial_prompt(probe, trace, priors)
    validate_prefix_outcome(policy.document(), probe, response, trace, priors)
    if request.prefix_probe is PrefixProbe.FULL:
        mismatch = compare_tokens(source[0], trace)
        if mismatch:
            raise ValueError(f"prefix exact repeat drift: {mismatch}")
    previous[request.name] = (trace, ordinal)
    return {"probe": request.prefix_probe.value,
            "source_ordinal": source[1] if source else None,
            "prompt_tokens": len(trace.prompt), "completion_tokens": len(trace.completion),
            "prefix_cache": response["runtime_summary"]["prefix_cache"],
            "mtp": response["runtime_summary"].get("mtp")}


class ReplayJournal:
    """One writer owns compact progress; append-only per-request artifacts retain history."""

    def __init__(self, directory: Path, sequence: dict) -> None:
        """Create a fresh diagnostic root without overwriting an earlier failure."""
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.started = time.monotonic()
        self.snapshot = {"schema": 1, "kind": "http_lifecycle_hammer_diagnostic",
                         "certified": False, "state": ReplayState.READY.value,
                         "run_nonce": uuid.uuid4().hex,
                         "completed_cycles": 0, "completed_requests": 0,
                         "active_request": None, "first_failure": None}
        self.write_json(directory / "sequence.json", sequence)
        self.publish()

    @staticmethod
    def write_json(path: Path, value: object) -> None:
        """Atomically replace complete JSON and clean only this writer's temporary file."""
        temporary = path.with_name(path.name + ".tmp")
        try:
            temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def publish(self) -> None:
        """Keep unfinished work visible even if the client is killed mid-request."""
        self.snapshot["elapsed_seconds"] = time.monotonic() - self.started
        self.write_json(self.directory / "progress.json", self.snapshot)

    def begin(self, request: ReplayRequest, cycle: int, ordinal: int) -> Path:
        """Publish exact request bytes before HTTP; allocation extent is never a request count."""
        if self.snapshot["state"] != ReplayState.READY.value:
            raise ValueError("HTTP replay cannot begin outside its ready state")
        stem = self.directory / f"http_{ordinal:09d}"
        self.write_json(stem.with_suffix(".request.json"), request.body)
        self.snapshot.update(state=ReplayState.REQUEST.value, active_request={
            "name": request.name, "cycle": cycle, "ordinal": ordinal,
            "request": str(stem.with_suffix(".request.json")),
            "response": str(stem.with_suffix(".response.txt")),
            "oracle": {"numeric_answer": request.expected_numeric_answer,
                       "json_answer": request.expected_json_answer},
            "prefix_probe": request.prefix_probe.value if request.prefix_probe else None,
            "prefix_source": request.prefix_source,
            "started_unix_seconds": time.time()})
        self.publish()
        return stem

    def finish(self, record: dict, *, failed: bool) -> None:
        """Freeze the first failure; never clear it or submit another request afterward."""
        if (self.snapshot["state"] != ReplayState.REQUEST.value
                or self.snapshot["active_request"]["ordinal"] != record.get("ordinal")):
            raise ValueError("HTTP replay completion does not own the active request")
        self.write_json(self.directory / f"http_{record['ordinal']:09d}.result.json", record)
        if failed:
            self.snapshot.update(state=ReplayState.FAILED.value, first_failure=record)
        else:
            self.snapshot.update(state=ReplayState.READY.value, active_request=None)
            self.snapshot["completed_requests"] += 1
        self.publish()


def capture_stall(command: list[str], directory: Path, stem: Path) -> dict:
    """Run one explicit passive diagnostic only after an HTTP wait has exceeded its threshold.

    The command receives the already-published request and progress paths; it
    must authenticate any process it inspects. Native inference is not restarted
    or synchronized. A separate process group bounds the diagnostic itself.
    """
    env = {**os.environ, "LLAMINAR_HTTP_STALL_REQUEST": str(stem.with_suffix(".request.json")),
           "LLAMINAR_HTTP_STALL_PROGRESS": str(directory / "progress.json")}
    with (stem.with_suffix(".diagnostic.log")).open("wb") as log:
        with subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = 124
    return {"return_code": code, "log": str(stem.with_suffix(".diagnostic.log"))}


def send_request(base_url: str, stem: Path, timeout: float, stall_seconds: float,
                 on_stall: list[str]) -> dict:
    """Submit once with a total HTTP deadline, retaining partial SSE bytes on failure.

    curl's total deadline prevents a trickle of SSE data from resetting the
    watchdog. The optional diagnostic is dormant during healthy requests and
    does not determine whether the HTTP request succeeded.
    """
    command = ["curl", "--silent", "--show-error", "--no-buffer", "--fail-with-body",
               "--max-time", str(timeout), "--connect-timeout", str(min(5, timeout)),
               "--header", "Content-Type: application/json", "--data-binary",
               "@" + str(stem.with_suffix(".request.json")), "--output",
               str(stem.with_suffix(".response.txt")), "--dump-header",
               str(stem.with_suffix(".headers.txt")), "--write-out", "%{http_code}",
               base_url.rstrip("/") + "/v1/chat/completions"]
    started, diagnostics = time.monotonic(), None
    with stem.with_suffix(".curl.log").open("wb") as error_log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error_log) as process:
            try:
                if on_stall:
                    try:
                        process.wait(timeout=stall_seconds)
                    except subprocess.TimeoutExpired:
                        diagnostics = capture_stall(on_stall, stem.parent, stem)
                remaining = max(0.001, timeout + 1 - (time.monotonic() - started))
                status, _ = process.communicate(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()
                status, _ = process.communicate()
                return {"return_code": 124, "http_status": status.decode(),
                        "diagnostics": diagnostics}
            except BaseException:
                process.kill()
                process.wait()
                raise
    return {"return_code": process.returncode, "http_status": status.decode(),
            "diagnostics": diagnostics}


def run(base_url: str, sequence: dict, directory: Path, *, request_timeout: float = 180,
        cycle_limit: int = 0, stall_seconds: float = 10, on_stall: list[str] | None = None,
        send: Callable = send_request, prefix_pressure: dict | None = None) -> int:
    """Repeat exact request order until failure; a finite limit is only a tooling self-test."""
    requests = read_sequence(sequence)
    pressure = PrefixPressure.from_document(prefix_pressure) if prefix_pressure is not None else None
    if (type(cycle_limit) is not int or cycle_limit < 0
            or type(request_timeout) not in (int, float) or not math.isfinite(request_timeout)
            or request_timeout <= 0 or type(stall_seconds) not in (int, float)
            or not math.isfinite(stall_seconds) or not 0 < stall_seconds < request_timeout
            or (on_stall is not None and (not isinstance(on_stall, list) or not on_stall
                or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in on_stall)))):
        raise ValueError("invalid HTTP hammer budget/diagnostic argv")
    journal = ReplayJournal(directory, sequence)
    if pressure:
        journal.write_json(directory / "prefix-pressure.json", pressure.document())
        journal.snapshot["prefix_probes"] = {probe.value: 0 for probe in PrefixProbe}
        journal.snapshot["prefix_storage_tiers"] = {}
    cycle, ordinal = 0, 0
    try:
        while cycle_limit == 0 or cycle < cycle_limit:
            cycle += 1
            # Keep the original sequence unchanged and exercise large-to-small
            # reset/replay between two independently authenticated long restores.
            prefix = pressure.requests(cycle, journal.snapshot["run_nonce"]) if pressure else ()
            cycle_requests = (*prefix[:4], *requests, *prefix[4:])
            previous: dict[str, tuple[TokenTrace, int]] = {}
            for request in cycle_requests:
                ordinal += 1
                stem = journal.begin(request, cycle, ordinal)
                started = time.monotonic()
                record = {"name": request.name, "cycle": cycle, "ordinal": ordinal}
                try:
                    # Long cold prefills are legitimate work, not a ten-second
                    # stall. Do not attach a debugger to them: the ordinary HTTP
                    # and native collective watchdogs still fail and retain the
                    # exact request. Restore/suffix calls retain passive probes.
                    diagnostic = [] if request.prefix_probe is PrefixProbe.FRESH else on_stall or []
                    record.update(send(base_url, stem, request_timeout, stall_seconds, diagnostic))
                    if record["return_code"] != 0 or record["http_status"] != "200":
                        raise ValueError(f"HTTP failed: curl={record['return_code']} status={record['http_status']}")
                    response = validate_response(request, stem.with_suffix(".response.txt").read_text())
                    if request.prefix_probe is not None:
                        record["prefix_evidence"] = validate_prefix_response(
                            request, response, pressure, previous, ordinal)
                        journal.snapshot["prefix_probes"][request.prefix_probe.value] += 1
                        tier = record["prefix_evidence"]["prefix_cache"]["storage_tier"]
                        tiers = journal.snapshot["prefix_storage_tiers"]
                        tiers[tier] = tiers.get(tier, 0) + 1
                    record["outcome"] = "passed"
                except Exception as error:
                    record.update(outcome="failed", error=str(error))
                record["elapsed_seconds"] = time.monotonic() - started
                journal.finish(record, failed=record["outcome"] == "failed")
                if record["outcome"] == "failed":
                    print(f"[http-hammer] FAIL cycle={cycle} request={request.name}: {record['error']}", flush=True)
                    return 1
            journal.snapshot["completed_cycles"] = cycle
            journal.publish()
            print(f"[http-hammer] PASS cycle={cycle} completed_requests={ordinal}", flush=True)
        journal.snapshot["state"] = ReplayState.STOPPED.value
        journal.publish()
        return 0
    except BaseException:
        journal.snapshot["state"] = ReplayState.INTERRUPTED.value
        journal.publish()
        raise


def main(argv: list[str] | None = None) -> int:
    """Consume explicit diagnostic configuration after the owning harness publishes readiness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.configuration.read_text())
    if (not isinstance(config, dict) or type(config.get("schema")) is not int
            or config["schema"] != 1 or config.get("kind") != "http_lifecycle_hammer_diagnostic"):
        raise ValueError("invalid HTTP hammer configuration")
    return run(args.base_url, config["sequence"], args.output,
               request_timeout=config["request_timeout_seconds"], cycle_limit=config["cycle_limit"],
               stall_seconds=config["stall_seconds"], on_stall=config.get("on_stall"),
               prefix_pressure=config.get("prefix_pressure"))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f"[http-hammer] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
