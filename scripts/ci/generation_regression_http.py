#!/usr/bin/env python3
"""Drive canonical token probes through an already-ready production server.

The mature server harness owns launch, readiness, graph/path evidence and
shutdown. This module owns request admission, exact terminal-token comparison,
completed prefix/MTP-state restoration and adaptive-controller observations.
Completed movement is checked from the placement authority's immutable journal,
never synthesized from optional profiling or inferred from configuration names.
Main-model state is a mandatory typed declaration: hybrid recurrent state is
required even with MTP off; attention-only cells cannot report a hybrid restore.
These obligations use the declared policy, never a response flag that could
self-select a weaker check. Collection writes unapproved
observations, never a certificate or
an automatically updated expected stream. MTP may compare only with the typed
MTP-off control supplied by the canonical matrix.
"""
from __future__ import annotations

import argparse
import dataclasses
from enum import Enum
import json
import math
from pathlib import Path
import time
import urllib.request
from typing import Iterable

from generation_tokens import GenerationWorkload, TokenTrace, compare_tokens
from generation_movement_ledger import MovementLedgerObserver, MovementRequirement
from production_artifacts import write_json


class MTPPolicy(str, Enum):
    """Exact producer-owned policy; names and CLI strings are never parsed."""
    OFF = "off"
    DEPTH1 = "depth_1"
    DEPTH2 = "depth_2"
    DEPTH3 = "depth_3"
    DEPTH15 = "depth_15"
    DYNAMIC = "dynamic"


class PrefixProbe(str, Enum):
    """Requested cache lifecycle; terminal path evidence must prove it occurred."""
    FRESH = "fresh"
    FULL = "full"
    PARTIAL = "partial"


class PrefixState(str, Enum):
    """Main-model restore contract, independent of speculative execution policy."""
    ATTENTION_KV = "attention_kv"
    HYBRID_RECURRENT = "hybrid_recurrent"


def generation_profile(record: dict) -> dict:
    """Validate complete canonical requests before any server/model admission."""
    workload = GenerationWorkload.from_record(record)
    profile = record["runtime"]["generation"]
    MovementRequirement(record["runtime"].get("movement_evidence"))
    policy = MTPPolicy(profile.get("mtp_policy"))
    PrefixState(profile.get("prefix_state"))
    if profile.get("mtp_verify_mode") != "speculative-sampling":
        raise ValueError("generation requires the canonical stochastic-capable MTP verifier policy")
    identity = record.get("id")
    control = profile.get("serial_control_id")
    if not isinstance(identity, str) or not identity or not isinstance(control, str) or not control:
        raise ValueError("generation cell requires explicit cell and serial-control identities")
    if (policy is MTPPolicy.OFF) != (identity == control):
        raise ValueError("only the MTP-off cell may name itself as its serial control")
    readiness = profile.get("readiness_timeout_seconds")
    if type(readiness) is not int or not 0 < readiness <= 600:
        raise ValueError("generation readiness must fit the exact-cell watchdog")
    context = record["runtime"].get("context_length")
    if type(context) is not int or context <= workload.max_tokens:
        raise ValueError("generation context must leave room for prompt and completion")
    requests = profile.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("canonical generation omitted its request sequence")
    seen, roles, prior_bodies = set(), set(), []
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("malformed canonical generation request")
        name, body = request.get("id"), request.get("body")
        role = PrefixProbe(request.get("prefix"))
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("empty or duplicate generation request identity")
        seen.add(name)
        roles.add(role)
        if not isinstance(body, dict) or body.get("return_token_ids") is not True or body.get("stream", False) is not False:
            raise ValueError("generation requires non-streaming, original token-ID responses")
        if body.get("return_runtime_summary") is not True:
            raise ValueError("generation requires authoritative terminal runtime summaries")
        if body.get("stop") or body.get("tools") or body.get("tool_choice"):
            raise ValueError("generation cannot substitute caller stop strings or tools for natural EOS")
        if type(body.get("max_tokens")) is not int or body["max_tokens"] != workload.max_tokens:
            raise ValueError("request budget differs from the canonical generation workload")
        # The public sampler consumes uint32. In particular, 2**32 must not
        # narrow to the zero/unseeded policy while the record claims a seed.
        if type(body.get("seed")) is not int or not 0 < body["seed"] < 2**32:
            raise ValueError("generation requires a positive uint32 position-keyed sampling seed; zero is unseeded")
        if (type(body.get("temperature")) not in (int, float) or not math.isfinite(body["temperature"])
                or body["temperature"] <= 0 or type(body.get("top_p")) not in (int, float)
                or not 0 < body["top_p"] <= 1 or type(body.get("top_k")) is not int or body["top_k"] <= 0):
            raise ValueError("generation requires explicit finite stochastic sampling parameters")
        messages = body.get("messages")
        if (not isinstance(messages, list) or len(messages) < 2
                or any(not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant")
                       or not isinstance(message.get("content"), str) or not message["content"] for message in messages)):
            raise ValueError("generation requires exact nonempty canonical message bytes")
        if not prior_bodies and role is not PrefixProbe.FRESH:
            raise ValueError("generation sequence must begin with a fresh request")
        if prior_bodies and role is PrefixProbe.FRESH:
            raise ValueError("only the initial request can claim a fresh server cache")
        if role is PrefixProbe.FULL and body not in prior_bodies:
            raise ValueError("full restore must repeat an earlier exact request")
        if role is PrefixProbe.PARTIAL and not any(
                len(messages) > len(prior["messages"]) and messages[:len(prior["messages"])] == prior["messages"]
                for prior in prior_bodies):
            raise ValueError("partial restore must extend an earlier complete conversation")
        prior_bodies.append(body)
    if roles != set(PrefixProbe):
        raise ValueError("canonical generation must cover fresh, full and partial prefix requests")
    return profile


def serial_workload_identity(record: dict) -> dict:
    """Return workload identity shared by the exact serial/MTP configuration.

    The ordinary server arguments differ for MTP execution, so they cannot be
    used as a serial equality key. The canonical producer supplies the control
    identity; prompt bytes, sampler settings, horizon and model must still agree.
    """
    profile = generation_profile(record)
    return {"control_id": profile["serial_control_id"], "model": record["model"],
            "context_length": record["runtime"]["context_length"],
            "max_tokens": profile["max_tokens"],
            "minimum_completion_tokens": profile["minimum_completion_tokens"],
            "prefix_state": profile["prefix_state"],
            "requests": profile["requests"]}


def observation_traces(record: dict, expected: dict) -> dict[str, TokenTrace]:
    """Independently revalidate all persisted responses, not merely a pass flag."""
    if (not isinstance(expected, dict) or expected.get("schema") != 1
            or expected.get("complete") is not True or expected.get("repeatability_passed") is not True):
        raise ValueError("serial control is incomplete or failed repeatability")
    if expected.get("configuration") != record:
        raise ValueError("observation configuration differs from the canonical cell")
    source_profile = generation_profile(record)
    rows = expected.get("requests")
    required = {request["id"]: request for request in source_profile["requests"]}
    if (not isinstance(rows, list) or len(rows) != len(required)
            or any(not isinstance(row, dict) for row in rows)
            or [row.get("id") for row in rows] != list(required)):
        raise ValueError("serial control omitted, reordered or duplicated request evidence")
    workload = GenerationWorkload.from_record(record)
    movement = MovementLedgerObserver(MovementRequirement(record["runtime"]["movement_evidence"]))
    traces = {}
    by_body = {}
    for row in rows:
        request = required[row["id"]]
        if row.get("body") != request["body"]:
            raise ValueError("serial observation request differs from the canonical request")
        trace = workload.observe(row.get("response"))
        validate_partial_prompt(request, trace, traces.values())
        validate_prefix_outcome(source_profile, request, row["response"], trace, traces.values())
        validate_mtp_outcome(source_profile, row["response"])
        movement.observe(row["response"])
        key = json.dumps(request["body"], sort_keys=True)
        if key in by_body and compare_tokens(by_body[key], trace):
            raise ValueError("serial control's repeated requests are not token-exact")
        by_body[key] = trace
        traces[row["id"]] = trace
    movement.finish()
    return traces


def validate_partial_prompt(request: dict, trace: TokenTrace, previous: Iterable[TokenTrace]) -> None:
    """Require a real cached-prompt token boundary, not just matching text.

    Hybrid recurrent state is saved at the earlier complete prompt. Templates
    can rewrite history, so message-level prefix equality alone is insufficient.
    This proves eligibility only; the terminal lifecycle must also prove restore.
    """
    if PrefixProbe(request["prefix"]) is PrefixProbe.PARTIAL and not any(
            len(prior.prompt) < len(trace.prompt) and trace.prompt[:len(prior.prompt)] == prior.prompt
            for prior in previous):
        raise ValueError("partial restore prompt does not retain an earlier complete token prefix")


def admit_control(record: dict, expected: dict) -> dict[str, TokenTrace]:
    """Reject incomplete, mismatched or speculative observations as an oracle."""
    source = expected.get("configuration") if isinstance(expected, dict) else None
    source_profile = generation_profile(source)
    if MTPPolicy(source_profile["mtp_policy"]) is not MTPPolicy.OFF:
        raise ValueError("MTP output cannot serve as its own expected serial stream")
    if serial_workload_identity(source) != serial_workload_identity(record):
        raise ValueError("serial control does not match the exact canonical workload/model")
    return observation_traces(source, expected)


def validate_prefix_outcome(profile: dict, request: dict, response: dict, trace: TokenTrace,
                            previous: Iterable[TokenTrace]) -> None:
    """Prove the requested restore from the runner's completed outcome.

    Matching input tokens establish eligibility, not an actual cache hit.
    Optional profiling cannot stand in for this request-local authority. A
    movement-invalidated miss still does not certify a restore; its published
    epoch span is retained for diagnosis, not used to waive the requirement.
    """
    summary = response.get("runtime_summary")
    if not isinstance(summary, dict) or type(summary.get("schema")) is not int or summary["schema"] != 1:
        raise ValueError("missing or unsupported terminal runtime summary")
    prefix = summary.get("prefix_cache")
    if not isinstance(prefix, dict):
        raise ValueError("missing terminal prefix outcome")
    for name in ("enabled", "bypassed", "hit", "partial_hit", "terminal_logits_restored",
                 "terminal_hidden_restored", "mtp_state_restored", "hybrid_state_restored"):
        if type(prefix.get(name)) is not bool:
            raise ValueError("malformed terminal prefix flag: " + name)
    for name in ("requested_tokens", "matched_tokens", "matched_blocks", "admission_epoch_earliest",
                 "admission_epoch_latest", "completion_movement_epoch"):
        if type(prefix.get(name)) is not int or prefix[name] < 0:
            raise ValueError("malformed terminal prefix counter: " + name)
    if not prefix["admission_epoch_earliest"] <= prefix["admission_epoch_latest"] <= prefix["completion_movement_epoch"]:
        raise ValueError("terminal prefix epoch span is reversed")
    requested, matched = prefix["requested_tokens"], prefix["matched_tokens"]
    if (prefix["enabled"] is not True or prefix["bypassed"] is not False
            or prefix.get("bypass_reason") != "" or requested != len(trace.prompt) or matched > requested):
        raise ValueError("prefix request was disabled, bypassed or has inconsistent token geometry")
    role = PrefixProbe(request["prefix"])
    if role is PrefixProbe.FRESH:
        if (prefix["hit"] or prefix["partial_hit"] or matched != 0 or prefix["matched_blocks"] != 0
                or prefix.get("storage_tier") != "none" or any(prefix[name] for name in (
                    "terminal_logits_restored", "terminal_hidden_restored", "mtp_state_restored", "hybrid_state_restored"))):
            raise ValueError("fresh request unexpectedly restored cached state")
        return
    # The runner's hit flag means full hit, not the union of both outcomes.
    # Partial restore is independently true and must not be mistaken for a miss.
    if not (prefix["hit"] or prefix["partial_hit"]) or prefix.get("storage_tier") not in ("ram", "disk-hydrated", "device-hot", "mixed"):
        raise ValueError("requested prefix restore did not occur: " + request["id"])
    if role is PrefixProbe.FULL:
        if not prefix["hit"] or prefix["partial_hit"] or matched != requested or not prefix["terminal_logits_restored"]:
            raise ValueError("full prefix request did not restore the complete terminal boundary")
    elif (prefix["hit"] or not prefix["partial_hit"] or not 0 < matched < requested or not any(
            len(prior.prompt) <= matched and trace.prompt[:len(prior.prompt)] == prior.prompt
            for prior in previous)):
        raise ValueError("partial prefix request did not restore an earlier complete prompt boundary")
    # Main recurrent state and the optional predictor are separate obligations.
    # Never let MTP-off silently turn a hybrid model into a KV-only proof. The
    # inverse check catches a stale summary inherited by an attention-only cell.
    state = PrefixState(profile["prefix_state"])
    if state is PrefixState.HYBRID_RECURRENT and not prefix["hybrid_state_restored"]:
        raise ValueError("hybrid prefix request did not restore main-model recurrent state")
    if state is PrefixState.ATTENTION_KV and prefix["hybrid_state_restored"]:
        raise ValueError("attention-only prefix request unexpectedly restored hybrid state")
    # MTP must restore shifted sidecar state at either cache boundary. A full
    # hit additionally needs terminal hidden because there is no suffix forward
    # to produce the predictor's input. Consult the immutable requested policy,
    # not response.mtp.enabled (which might itself be the broken observation).
    if MTPPolicy(profile["mtp_policy"]) is not MTPPolicy.OFF:
        if not prefix["mtp_state_restored"]:
            raise ValueError("MTP prefix request did not restore shifted sidecar state")
        if role is PrefixProbe.FULL and not prefix["terminal_hidden_restored"]:
            raise ValueError("MTP full prefix request did not restore terminal hidden state")


def post_completion(base_url: str, body: dict, timeout: float) -> dict:
    """Send exactly one public request; never retry or extend a short completion."""
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def validate_mtp_outcome(profile: dict, response: dict) -> None:
    """Require actual terminal MTP execution, not a flag or profiling assertion.

    A positive seed selects the production serial-sample-equivalent stochastic
    verifier. These request-local observations prove that drafting, verification
    and acceptance occurred under the named policy without downloading state.
    """
    summary = response.get("runtime_summary")
    mtp = summary.get("mtp") if isinstance(summary, dict) else None
    if not isinstance(mtp, dict):
        raise ValueError("missing terminal MTP outcome")
    for name in ("enabled", "bypassed", "stochastic_verify", "adaptive_depth_enabled"):
        if type(mtp.get(name)) is not bool:
            raise ValueError("malformed terminal MTP flag: " + name)
    for name in ("draft_steps", "verifier_runs", "verifier_token_count", "accepted_tokens",
                 "stochastic_accept_tests", "current_depth", "min_depth", "max_depth", "depth_policy_updates"):
        if type(mtp.get(name)) is not int or mtp[name] < 0:
            raise ValueError("malformed terminal MTP counter: " + name)
    policy = MTPPolicy(profile["mtp_policy"])
    if policy is MTPPolicy.OFF:
        if mtp["enabled"] or any(mtp[name] for name in (
                "draft_steps", "verifier_runs", "verifier_token_count", "accepted_tokens", "stochastic_accept_tests")):
            raise ValueError("serial control executed speculative work")
        return
    if (not mtp["enabled"] or mtp["bypassed"] or mtp.get("bypass_reason") != ""
            or not mtp["stochastic_verify"] or mtp.get("verify_mode") != profile["mtp_verify_mode"]
            or any(mtp[name] == 0 for name in ("draft_steps", "verifier_runs", "verifier_token_count",
                                            "accepted_tokens", "stochastic_accept_tests"))):
        raise ValueError("MTP request omitted active stochastic drafting, verification or accepted drafts")
    if not 0 < mtp["min_depth"] <= mtp["current_depth"] <= mtp["max_depth"]:
        raise ValueError("terminal MTP depth is outside its admitted bounds")
    dynamic = policy is MTPPolicy.DYNAMIC
    if mtp["adaptive_depth_enabled"] != dynamic or mtp.get("depth_policy_mode") != ("dynamic" if dynamic else "fixed"):
        raise ValueError("terminal MTP depth policy differs from the canonical cell")
    if dynamic and mtp["depth_policy_updates"] == 0:
        raise ValueError("dynamic MTP request did not observe a completed depth-policy update")
    fixed_depths = {MTPPolicy.DEPTH1: 1, MTPPolicy.DEPTH2: 2, MTPPolicy.DEPTH3: 3, MTPPolicy.DEPTH15: 15}
    if not dynamic and any(mtp[name] != fixed_depths[policy] for name in ("min_depth", "current_depth", "max_depth")):
        raise ValueError("fixed MTP request did not execute its named draft depth")


def run_probes(record: dict, base_url: str, output: Path, expected: dict | None = None) -> dict:
    """Collect or compare an entire ordered workload within one bounded session.

    All observations remain diagnostic until the outer campaign authenticates
    independent numerical provenance and full path/ledger evidence. Even a
    successful serial comparison cannot issue an image certificate here.
    """
    profile = generation_profile(record)
    if expected is None and MTPPolicy(profile["mtp_policy"]) is not MTPPolicy.OFF:
        raise ValueError("MTP generation requires its serial control before inference")
    controls = admit_control(record, expected) if expected is not None else None
    output.mkdir(parents=True, exist_ok=False)
    workload = GenerationWorkload.from_record(record)
    movement = MovementLedgerObserver(MovementRequirement(record["runtime"]["movement_evidence"]))
    report = {"schema": 1, "configuration": record, "complete": False,
              "certification_eligible": False, "repeatability_passed": False,
              "serial_comparison_passed": False if controls else None, "requests": []}
    started = time.monotonic()
    deadline = started + 600
    repeated = {}
    traces = []
    try:
        for request in profile["requests"]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("generation request sequence exceeded the exact-cell watchdog")
            before = time.monotonic()
            response = post_completion(base_url, request["body"], remaining)
            row = {"id": request["id"], "body": request["body"], "response": response,
                   "elapsed_seconds": time.monotonic() - before}
            report["requests"].append(row)
            # Persist the offending response too: the first failure is the
            # useful evidence, not just a boolean in the campaign report.
            trace = workload.observe(response)
            validate_partial_prompt(request, trace, traces)
            validate_prefix_outcome(profile, request, response, trace, traces)
            validate_mtp_outcome(profile, response)
            movement.observe(response)
            key = json.dumps(request["body"], sort_keys=True)
            reference = controls[request["id"]] if controls else repeated.get(key)
            mismatch = compare_tokens(reference, trace) if reference else None
            if mismatch:
                row["mismatch"] = dataclasses.asdict(mismatch)
                raise ValueError(f"{request['id']}: token drift in {mismatch.phase.value} at {mismatch.position}")
            repeated[key] = trace
            traces.append(trace)
            print(f"[model-parity-generation] request={request['id']} tokens={len(trace.completion)} "
                  f"elapsed={row['elapsed_seconds']:.3f}s", flush=True)
        movement.finish()
        report.update(complete=True, repeatability_passed=True,
                      serial_comparison_passed=True if controls else None)
    except BaseException as error:
        report["error"] = str(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "observations.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    """HTTP-only entry point called after the shared harness publishes readiness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    args = parser.parse_args(argv)
    run_probes(json.loads(args.configuration.read_text()), args.base_url, args.output,
               json.loads(args.control.read_text()) if args.control else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
