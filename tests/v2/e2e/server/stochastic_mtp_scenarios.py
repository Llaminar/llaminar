#!/usr/bin/env python3
"""HTTP regression matrix for stochastic MTP request/graph identity (issue #15).

The server owns model placement and fixed/dynamic MTP policy. These scenarios
only exercise the public chat surface, including omitted seeds, SSE errors, and
sampler changes between requests. Seeded controls must survive intervening
requests and prefix restoration byte-exactly. No inference retry is permitted.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Scenario:
    """One public sampling law; reuse_key names a byte-identical seeded control."""

    name: str
    settings: dict[str, Any]
    reuse_key: str = ""
    allowed_rejection_code: str = ""


SCENARIOS = (
    Scenario("unseeded_reported", dict(temperature=1.0, top_p=0.95, top_k=20,
                                     presence_penalty=1.5, stream=False)),
    Scenario("explicit_zero_seed_stream", dict(temperature=1.0, top_p=0.95,
             top_k=20, presence_penalty=1.5, seed=0, stream=True)),
    Scenario("seeded_control", dict(temperature=0.7, top_p=0.9, top_k=40,
             seed=12345, stream=False), "seeded_control"),
    Scenario("narrow_cold", dict(temperature=0.2, top_p=0.2, top_k=1,
             frequency_penalty=0.25, stream=True)),
    Scenario("wide_hot", dict(temperature=1.3, top_p=1.0, top_k=256,
             presence_penalty=-0.25, frequency_penalty=0.5, stream=False)),
    Scenario("different_seed_and_penalties", dict(temperature=0.8, top_p=0.75,
             top_k=2, presence_penalty=0.75, frequency_penalty=-0.1,
             seed=23456, stream=True)),
    Scenario("greedy_interleave", dict(temperature=0.0, top_p=1.0,
             top_k=40, stream=False)),
    Scenario("seeded_control_restored_stream", dict(temperature=0.7, top_p=0.9,
             top_k=40, seed=12345, stream=True), "seeded_control"),
    Scenario("unseeded_restored", dict(temperature=1.0, top_p=0.95, top_k=20,
                                     presence_penalty=1.5, stream=False)),
    Scenario("dry_penalty", dict(temperature=0.8, top_p=0.9, top_k=40,
             dry_multiplier=0.8, dry_base=1.75, dry_allowed_length=2,
             dry_penalty_last_n=128, dry_sequence_breakers=["\n", ":"], stream=False),
             allowed_rejection_code="unsupported_mtp_sampling_policy"),
    Scenario("one_token_budget", dict(temperature=1.0, top_p=0.95, top_k=20,
                                     max_tokens=1, stream=True)),
)


def payload(model: str, scenario: Scenario) -> dict[str, Any]:
    """Use a stable prompt so changed sampling must not consume stale captures."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: PASS."}],
        "max_tokens": 16 if scenario.name in ("unseeded_reported", "unseeded_restored") else 32,
        **({"return_runtime_summary": True} if not scenario.settings["stream"] else {}),
        **scenario.settings,
    }


def decode_response(body: str, *, streaming: bool) -> tuple[str, str]:
    """Require text and a clean terminal result, including HTTP-200 SSE errors."""
    content, reasoning = [], []
    finished = False
    packets = []
    if streaming:
        for line in body.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            packets.append(json.loads(data))
    else:
        packets.append(json.loads(body))
    for packet in packets:
        if packet.get("error"):
            raise ValueError(f"HTTP inference error: {packet['error']}")
        for choice in packet.get("choices", []):
            message = choice.get("delta" if streaming else "message", {})
            if message.get("error") or choice.get("finish_reason") == "error":
                raise ValueError(f"streamed inference error: {choice}")
            content.append(message.get("content") or "")
            reasoning.append(message.get("reasoning_content") or "")
            finished |= choice.get("finish_reason") in ("stop", "length")
    result = ("".join(content), "".join(reasoning))
    if not finished or not any(result):
        raise ValueError("response lacks generated text or a clean stop/length terminal")
    return result


def post(base_url: str, body: dict[str, Any], timeout: float) -> str:
    """Perform exactly one public request; HTTP errors are failures, not retries."""
    request = Request(base_url.rstrip("/") + "/v1/chat/completions",
                      data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode()


def run(base_url: str, model: str, timeout: float, *,
        scenarios: tuple[Scenario, ...] = SCENARIOS,
        send: Callable[[str, dict[str, Any], float], str] = post) -> list[dict[str, Any]]:
    """Collect all failures in order; one bad policy must not hide later cells."""
    records = []
    controls: dict[str, tuple[str, str]] = {}
    for scenario in scenarios:
        body = payload(model, scenario)
        record: dict[str, Any] = {"scenario": scenario.name, "request": body}
        started = time.monotonic()
        try:
            raw = send(base_url, body, timeout)
            record["response"] = raw
            observed = decode_response(raw, streaming=body["stream"])
            if not body["stream"]:
                # The terminal public summary is authority-owned evidence, not
                # a request flag or a profiling-dependent claim. SSE currently
                # has no summary extension; its response/lifecycle is tested
                # here and the owning harness checks captured-path PerfStats.
                mtp = json.loads(raw).get("runtime_summary", {}).get("mtp", {})
                if (mtp.get("enabled") is not True or mtp.get("bypassed") is not False
                        or mtp.get("verifier_runs", 0) <= 0
                        or (body["temperature"] > 0 and mtp.get("stochastic_verify") is not True)):
                    raise ValueError("request did not execute the admitted MTP verifier")
            if scenario.reuse_key:
                if scenario.reuse_key in controls and controls[scenario.reuse_key] != observed:
                    raise ValueError("seeded output changed after sampling-policy changes/prefix restore")
                controls.setdefault(scenario.reuse_key, observed)
            record.update(status="passed", outcome="generation")
        except HTTPError as error:
            # CPU supports DRY; GPUs currently reject it at request admission.
            # A successful feature implementation still has to prove generation
            # above. Only the exact typed 400 is a valid negative certificate:
            # arbitrary 4xx, late 500s, and SSE errors must stay red.
            record["http_status"] = error.code
            try:
                # Decoding belongs inside this failure boundary too. A proxy
                # returning a malformed body must not abort the remaining
                # scenarios or masquerade as the precise admission rejection.
                raw = error.read().decode()
                record["response"] = raw
                detail = json.loads(raw).get("error", {})
                if (error.code != 400 or not scenario.allowed_rejection_code
                        or detail.get("type") != "invalid_request_error"
                        or detail.get("code") != scenario.allowed_rejection_code
                        or detail.get("param") != "dry_multiplier"):
                    raise ValueError(f"unexpected HTTP rejection: {error.code} {raw}")
                record.update(status="passed", outcome="unsupported_policy_rejected")
            except Exception as rejection_error:
                record.update(status="failed", error=str(rejection_error))
        except Exception as error:
            record.update(status="failed", error=str(error))
        record["seconds"] = time.monotonic() - started
        records.append(record)
        print(f"{scenario.name}: {record['status']} ({record['seconds']:.2f}s)", flush=True)
    return records


def main() -> int:
    """Run against an already-ready server; the E2E harness owns its lifecycle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--scenario", choices=[case.name for case in SCENARIOS])
    args = parser.parse_args()
    model = args.model
    if not model:
        with urlopen(args.base_url.rstrip("/") + "/v1/models", timeout=args.timeout) as response:
            model = json.load(response)["data"][0]["id"]
    cases = tuple(case for case in SCENARIOS if not args.scenario or case.name == args.scenario)
    records = run(args.base_url, model, args.timeout, scenarios=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"model": model, "cases": records}, indent=2) + "\n")
    return int(any(record["status"] != "passed" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
