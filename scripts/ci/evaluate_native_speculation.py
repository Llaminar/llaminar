#!/usr/bin/env python3
## @file evaluate_native_speculation.py
## @brief Compare frozen native AR/MTP experiments without launching inference.
##
## The independent evaluator owns the plan and its externally retained digest.
## Native BenchmarkRunner JSON owns token counts, wall times and MTP counters;
## this consumer never replays rows, changes runtime policy or promotes code.
## Evidence hashes detect substitutions, not dishonesty by a collector. Build,
## model, placement and hardware provenance must come from a trusted collector.
"""Fail-closed, device-free analysis of native speculation experiments.

Use ``inspect`` before collection to validate a frozen plan and print the
identity digests the collector must retain. Use ``evaluate`` after collecting
all planned AR/stock/candidate pairs. Reports are diagnostic, never production
certificates. See docs/native-speculation-evaluation.md for the trust boundary.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tempfile
from typing import Any

from production_artifacts import CPUISA, digest, positive, write_json
from run_model_parity_benchmarks import validate_measurement, validate_workload

PLAN_SCHEMA = "llaminar.native-speculation-plan.v1"
EVIDENCE_SCHEMA = "llaminar.native-speculation-evidence.v1"
REPORT_SCHEMA = "llaminar.native-speculation-evaluation.v1"
COUNTERS = ("draft_steps", "accepted_tokens", "rejected_tokens",
            "verifier_runs", "verifier_token_count")
CONFIG_IO_FIELDS = {"model_path", "benchmark_json_output_path", "sampling", "n_predict"}
INT32_MAX = 2**31 - 1
UINT64_MAX = 2**64 - 1


class Role(str, Enum):
    """Distinct baselines prevent an AR speedup being presented as an MTP gain."""

    AR = "ar"
    STOCK = "stock"
    CANDIDATE = "candidate"


class Split(str, Enum):
    """Development and previously unseen requests never share an aggregate."""

    DEVELOPMENT = "development"
    HELD_OUT = "held_out"


@dataclass(frozen=True, slots=True)
class FrozenPlan:
    """Retain validated intent in immutable storage, not aliased nested dicts.

    Each public view is a fresh consumer-owned projection. Mutating a returned
    config, request or build map cannot change the intent named by the retained
    digest. Evaluation materializes these views once, not once per observation.
    """

    _document_json: str
    sha256: str
    deployment_sha256: str
    _build_items: tuple[tuple[str, str], ...]

    @property
    def document(self) -> dict:
        """Return an independently owned copy of the validated plan document."""
        return json.loads(self._document_json)

    @property
    def builds(self) -> dict[str, str]:
        """Project immutable build identities without exposing writable storage."""
        return dict(self._build_items)

    @property
    def requests(self) -> dict[str, dict]:
        """Index a new request view; nested workloads remain caller-owned."""
        return {row["id"]: row for row in self.document["requests"]}

    def identity(self) -> dict:
        """Return collector bindings without exposing held-out prompt contents."""
        return {"plan_sha256": self.sha256,
                "deployment_sha256": self.deployment_sha256,
                "build_sha256": self.builds}


def require(condition: bool, message: str) -> None:
    """Reject invalid evidence instead of guessing an omitted observation."""
    if not condition:
        raise ValueError(message)


def integer(value: Any, name: str, minimum: int = 0, *, maximum: int = UINT64_MAX) -> int:
    """Require a representable native integer, excluding booleans and fractions."""
    require(type(value) is int and minimum <= value <= maximum,
            f"invalid integer {name}: expected {minimum}..{maximum}")
    return value


def sha256(value: Any) -> str:
    """Accept only complete lowercase SHA-256 identities."""
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "invalid SHA-256 identity")
    return value


def object_pairs(pairs: list[tuple[str, Any]]) -> dict:
    """Reject duplicate JSON keys, including nested identity or timing fields."""
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_document(path: Path, expected: str | None = None) -> tuple[dict, str]:
    """Hash and decode the same bytes, avoiding a hash/read substitution window."""
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if expected is not None:
        require(actual == sha256(expected), f"artifact digest mismatch: {path}")
    document = json.loads(payload, object_pairs_hook=object_pairs)
    require(isinstance(document, dict) and bool(document), f"expected nonempty JSON object: {path}")
    # json.loads accepts non-standard NaN/Infinity; canonical metadata may not.
    digest(document)
    return document, actual


def artifact(reference: dict, directory: Path) -> tuple[dict, str]:
    """Resolve an evaluator-recorded JSON artifact, never a process to execute."""
    require(isinstance(reference, dict) and set(reference) == {"path", "sha256"},
            "artifact needs exactly path and sha256")
    require(isinstance(reference["path"], str) and bool(reference["path"]), "empty artifact path")
    return read_document(directory / reference["path"], sha256(reference["sha256"]))


def load_plan(path: Path, expected: str) -> FrozenPlan:
    """Validate a precommitted deployment, implementation set and request split.

    One common deployment binds all roles. AR and stock must use the same
    Release build; candidate source/binary changes are allowed, but compiler,
    ISA and runtime-library identities must remain matched. Resolved native
    configuration is pinned separately for each role, not reconstructed from
    command-line names. Only explicitly pinned MTP settings may differ.
    """
    plan, actual = read_document(path, expected)
    require(set(plan) == {"schema", "deployment", "implementations", "requests", "pairs"}
            and plan["schema"] == PLAN_SCHEMA, "unsupported or incomplete evaluation plan")
    deployment = plan["deployment"]
    require(isinstance(deployment, dict) and set(deployment) == {"model", "hardware", "execution"},
            "deployment needs model, hardware and execution artifacts")
    deployment_ids = {key: artifact(value, path.parent)[1] for key, value in deployment.items()}
    roles = plan["implementations"]
    require(isinstance(roles, dict) and set(roles) == {role.value for role in Role},
            "plan must name ar, stock and candidate implementations")
    builds, controls, configurations = {}, [], []
    for role in Role:
        entry = roles[role.value]
        require(isinstance(entry, dict) and set(entry) == {"build", "config"},
                f"{role.value}: implementation needs build and config")
        build, builds[role.value] = artifact(entry["build"], path.parent)
        require(build.get("build_type") == "Release", "only Release evidence is comparable")
        require(isinstance(build.get("source_revision"), str)
                and re.fullmatch(r"[0-9a-f]{40}", build["source_revision"]) is not None,
                "build needs a complete source revision")
        sha256(build.get("binary_sha256"))
        for key in ("compiler", "runtime_libraries"):
            require(isinstance(build.get(key), dict) and bool(build[key]), f"build lacks {key}")
        CPUISA(build.get("cpu_isa"))
        controls.append({key: build[key] for key in ("compiler", "runtime_libraries", "cpu_isa")})
        config = entry["config"]
        required = {"benchmark_mode", "prefix_cache_enabled", "mtp_enabled", "mtp_verify_mode",
                    "mtp_depth_policy", "mtp_draft_tokens", "mtp_graph_capacity_draft_tokens",
                    "mtp_max_request_batch"}
        require(isinstance(config, dict) and required <= set(config)
                and not CONFIG_IO_FIELDS.intersection(config), "incomplete semantic benchmark config")
        require(config["benchmark_mode"] is True and type(config["prefix_cache_enabled"]) is bool,
                "benchmark and prefix-cache policies must be explicit")
        require(config["mtp_enabled"] is (role is not Role.AR), "AR/MTP role and config disagree")
        configurations.append({key: value for key, value in config.items() if not key.startswith("mtp_")})
    require(builds["ar"] == builds["stock"], "AR and stock must use the identical build artifact")
    # Python's container equality aliases bool/int and int/float values. The
    # canonical identity encoding retains JSON types, including nested fields.
    require(len({digest(item) for item in controls}) == 1, "compiler/ISA/runtime library identity differs")
    require(len({digest(item) for item in configurations}) == 1, "non-MTP execution policy differs")
    pairs = plan["pairs"]
    require(isinstance(pairs, list) and len(pairs) >= 3
            and all(isinstance(item, str) and item for item in pairs)
            and len(set(pairs)) == len(pairs), "need at least three unique paired blocks")
    requests, prompts, workloads = {}, {}, set()
    require(isinstance(plan["requests"], list) and plan["requests"], "empty request inventory")
    for request in plan["requests"]:
        require(isinstance(request, dict) and set(request) == {"id", "split", "workload"},
                "request needs id, split and workload")
        identifier = request["id"]
        require(isinstance(identifier, str) and identifier and identifier not in requests, "duplicate/empty request id")
        split = Split(request["split"])
        workload = request["workload"]
        require(isinstance(workload, dict) and set(workload) == {
            "schema", "prompt_unit", "prompt_repetitions", "decode_tokens", "warmup_iterations",
            "measurement_iterations", "seed", "temperature", "top_k", "top_p"}, "unsupported workload fields")
        require(type(workload["schema"]) is int and isinstance(workload["prompt_unit"], str),
                "invalid workload schema or prompt")
        prompt = validate_workload(workload)
        integer(workload["decode_tokens"], "decode_tokens", 2, maximum=INT32_MAX)
        integer(workload["measurement_iterations"], "measurement_iterations", 3, maximum=INT32_MAX)
        integer(workload["warmup_iterations"], "warmup_iterations", 1, maximum=INT32_MAX)
        integer(workload["seed"], "seed", maximum=INT32_MAX)
        integer(workload["top_k"], "top_k", maximum=INT32_MAX)
        require(type(workload["temperature"]) in (int, float) and workload["temperature"] >= 0,
                "invalid sampling temperature")
        require(positive(workload["top_p"]) <= 1, "invalid sampling top_p")
        # Changing the seed or dividing a prompt into different units does not
        # make a development prompt unseen. Multiple seeds within a split are OK.
        prompt_id = hashlib.sha256(prompt.encode()).hexdigest()
        require(prompt_id not in prompts or prompts[prompt_id] == split,
                "development/held-out prompt leakage")
        prompts[prompt_id] = split
        workload_id = digest({"prompt_sha256": prompt_id,
                              **{key: workload[key] for key in ("decode_tokens", "seed", "top_k")},
                              "temperature": float(workload["temperature"]), "top_p": float(workload["top_p"])})
        require(workload_id not in workloads, "duplicate request workload would reweight the comparison")
        workloads.add(workload_id)
        requests[identifier] = request
    require({row["split"] for row in requests.values()} == {split.value for split in Split},
            "plan requires disjoint development and held-out requests")
    return FrozenPlan(json.dumps(plan, sort_keys=True, allow_nan=False), actual,
                      digest(deployment_ids), tuple(builds.items()))


def native_measurement(data: dict, workload: dict, config: dict, role: Role) -> dict:
    """Validate native samples and derive full-decode costs, not acceptance proxies.

    The terminal-prefill token is excluded from decode throughput. Verifier
    amortization includes the *entire* decode wall interval, including any
    ordinary decode work; it is not a sampled transaction latency. Physical
    verifier rows are reported as work, never as useful emitted tokens.
    """
    validate_measurement(data, workload)
    actual_config = data["config"]
    require(digest({key: value for key, value in actual_config.items() if key not in CONFIG_IO_FIELDS}) == digest(config),
            "native configuration differs from frozen role")
    require(type(actual_config["n_predict"]) is int and actual_config["n_predict"] == workload["decode_tokens"],
            "native output budget differs")
    sampler = actual_config["sampling"]
    integer(sampler["seed"], "sampling seed", maximum=INT32_MAX)
    integer(sampler["top_k"], "sampling top_k", maximum=INT32_MAX)
    require(type(sampler["temperature"]) in (int, float), "invalid sampling temperature")
    positive(sampler["top_p"])
    require(sampler == {key: workload[key] for key in ("seed", "temperature", "top_k", "top_p")},
            "native sampler differs from frozen workload")
    integer(data["prompt"]["bytes"], "prompt bytes", 1)
    integer(data["measurement_iterations"], "measurement_iterations", 1, maximum=INT32_MAX)
    integer(data["warmup_iterations"], "warmup_iterations", 1, maximum=INT32_MAX)
    totals = {key: 0 for key in COUNTERS}
    decode_times, prefill_times, outputs = [], [], []
    for index, row in enumerate(data["iterations"], 1):
        require(integer(row["iteration"], "iteration", 1, maximum=INT32_MAX) == index, "native iteration order/gaps")
        counts = row["tokens"]
        prefill = integer(counts["prefill"], "prefill", 1, maximum=INT32_MAX)
        decode = integer(counts["decode"], "decode", 2, maximum=INT32_MAX)
        useful = integer(counts["decode_after_prefill"], "decode_after_prefill", 1, maximum=INT32_MAX)
        require(useful == decode - 1, "decode numerator includes terminal-prefill token")
        tokens = row["generated_token_ids"]
        require(isinstance(tokens, list) and len(tokens) == decode, "missing/truncated generated token ids")
        for token in tokens:
            integer(token, "generated_token_id", maximum=INT32_MAX)
        outputs.append(tokens)
        for phase, count, rate in (("prefill", prefill, "prefill"), ("decode", useful, "decode_after_prefill")):
            elapsed = positive(row["timing_ms"][phase])
            require(math.isclose(positive(row["throughput_tokens_per_sec"][rate]),
                                 count * 1000 / elapsed, rel_tol=1e-8, abs_tol=0.0),
                    f"native {phase} time/throughput arithmetic disagrees")
            (prefill_times if phase == "prefill" else decode_times).append(elapsed)
        for key in COUNTERS:
            totals[key] += integer(row["mtp"][key], key)
    require(all(tokens == outputs[0] for tokens in outputs), "seeded output changed between native repeats")
    aggregate = data["mtp"]
    require(aggregate["config_enabled"] is (role is not Role.AR), "native MTP activation differs")
    require(aggregate["bypassed"] is False, "native execution bypassed the selected policy")
    for key in ("transaction_validation_failures", "unsafe_verifier_state_rejections"):
        require(integer(aggregate[key], key) == 0, f"native {key}")
    for key in COUNTERS:
        require(integer(aggregate[key], key) == totals[key], "native MTP aggregate includes omitted/warmup work")
    require((not any(totals.values())) if role is Role.AR else totals["verifier_runs"] > 0,
            "native counters do not prove the selected AR/MTP path")
    runs = totals["verifier_runs"]
    emitted = (workload["decode_tokens"] - 1) * len(outputs)
    attempted = totals["accepted_tokens"] + totals["rejected_tokens"]
    return {"output": outputs[0], "prefill_tokens": data["iterations"][0]["tokens"]["prefill"],
            "decode_ms": statistics.median(decode_times), "prefill_ms": statistics.median(prefill_times),
            "prefill_plus_decode_ms": statistics.median([p + d for p, d in zip(prefill_times, decode_times)]),
            "decode_ms_per_emitted_token": sum(decode_times) / emitted,
            "decode_tokens_per_second": emitted * 1000 / sum(decode_times),
            "mtp": {**totals, "accepted_fraction": totals["accepted_tokens"] / attempted if attempted else None,
                    "emitted_tokens_per_verifier_invocation": emitted / runs if runs else None,
                    "whole_decode_ms_per_verifier_invocation": sum(decode_times) / runs if runs else None}}


def timing_observations(data: dict) -> list[dict]:
    """Retain largest native timers without adding overlapping spans or devices.

    These are attribution hints, not a wall-time partition or a roofline. The
    original hashed benchmark remains the authority for the complete inventory.
    Missing/disabled PerfStats means unobserved, never zero-cost work.
    """
    stats = data["perf_stats"]
    require(stats["schema"] == "llaminar.perf_stats.v1" and type(stats["enabled"]) is bool
            and isinstance(stats["records"], list), "invalid native PerfStats envelope")
    require(stats["enabled"] or not stats["records"], "disabled PerfStats contains observations")
    timers = []
    for record in stats["records"]:
        require(isinstance(record, dict), "invalid PerfStats record")
        if record["kind"] != "timer":
            continue
        elapsed = integer(record["total_ns"], "timer total_ns")
        count = integer(record["count"], "timer count", 1)
        require(all(isinstance(record[key], str) for key in ("domain", "name", "phase", "device"))
                and isinstance(record["tags"], dict), "incomplete timer identity")
        timers.append({**{key: record[key] for key in ("domain", "name", "phase", "device", "tags")},
                       "count": count, "total_ns": elapsed})
    return sorted(timers, key=lambda row: row["total_ns"], reverse=True)[:10]


def distribution(values: list[float]) -> dict:
    """Describe paired blocks without pretending their range is a confidence interval."""
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "count": len(values)}


def evaluate(plan: FrozenPlan, evidence_path: Path) -> dict:
    """Compare every frozen request/block/role or fail without a partial green.

    Exact emitted tokens are checked against the same-build AR control across
    all blocks and roles. This cannot prove serial-row logits, cache bytes or
    publication state, so even a faster, token-exact candidate stays diagnostic.
    """
    evidence, evidence_id = read_document(evidence_path)
    require(set(evidence) == {"schema", "plan_sha256", "observations"}
            and evidence["schema"] == EVIDENCE_SCHEMA and evidence["plan_sha256"] == plan.sha256,
            "evidence does not belong to the frozen plan")
    document = plan.document
    request_inventory = {row["id"]: row for row in document["requests"]}
    builds = plan.builds
    pairs = document["pairs"]
    expected = {(request, pair, role.value) for request in request_inventory for pair in pairs for role in Role}
    rows, artifacts_seen = {}, set()
    require(isinstance(evidence["observations"], list), "invalid observations list")
    for observation in evidence["observations"]:
        require(isinstance(observation, dict) and set(observation) == {
            "request", "pair", "role", "deployment_sha256", "build_sha256", "measurement", "benchmark"},
            "incomplete observation identity")
        key = (observation["request"], observation["pair"], observation["role"])
        require(key in expected and key not in rows, "unexpected/duplicate observation")
        request, pair, name = key
        require(observation["measurement"] == "unprofiled", "profiled timings cannot establish economy")
        require(observation["deployment_sha256"] == plan.deployment_sha256, "deployment identity mismatch")
        require(observation["build_sha256"] == builds[name], "implementation identity mismatch")
        data, benchmark_id = artifact(observation["benchmark"], evidence_path.parent)
        require(benchmark_id not in artifacts_seen, "reused benchmark artifact is not an independent block")
        artifacts_seen.add(benchmark_id)
        sample = native_measurement(data, request_inventory[request]["workload"],
                                    document["implementations"][name]["config"], Role(name))
        rows[key] = {**sample, "benchmark_sha256": benchmark_id,
                     "timers_nonadditive": timing_observations(data)}
    require(set(rows) == expected, f"incomplete evidence: missing {len(expected - set(rows))} observations")
    requests = []
    comparisons = (("stock_vs_ar", "ar", "stock"), ("candidate_vs_ar", "ar", "candidate"),
                   ("candidate_vs_stock", "stock", "candidate"))
    for name, request in request_inventory.items():
        control = rows[name, pairs[0], "ar"]
        for pair in pairs:
            for role in Role:
                sample = rows[name, pair, role.value]
                require(sample["output"] == control["output"], f"exact AR token mismatch: {name}/{pair}/{role.value}")
                require(sample["prefill_tokens"] == control["prefill_tokens"], "cross-role tokenizer mismatch")
        gains = {}
        for label, baseline, candidate in comparisons:
            gains[label] = {metric: distribution([rows[name, pair, baseline][metric] /
                                                  rows[name, pair, candidate][metric] for pair in pairs])
                            for metric in ("decode_ms", "prefill_ms", "prefill_plus_decode_ms")}
        requests.append({"id": name, "split": request["split"], "speedup": gains,
                         "blocks": [{"pair": pair, "roles": {
                             role.value: {key: value for key, value in rows[name, pair, role.value].items() if key != "output"}
                             for role in Role}} for pair in pairs]})
    summaries = {}
    for split in Split:
        selected = [row for row in requests if row["split"] == split.value]
        summaries[split.value] = {
            label: math.exp(statistics.mean(math.log(row["speedup"][label]["decode_ms"]["median"]) for row in selected))
            for label, _, _ in comparisons}
    report = {"schema": REPORT_SCHEMA, "status": "compared", "certification_eligible": False,
            **plan.identity(), "evidence_sha256": evidence_id, "requests": requests,
            "decode_speedup_geomean_by_split": summaries,
            "correctness": {"emitted_token_ids": "exact_ar_match", "serial_row_and_state_bytes": "not_observed",
                            "production_path_and_prerequisite_gates": "not_qualified_by_this_tool"},
            "unobserved_headroom": {
                "acceptance": "No complete per-transaction live proposal/commit trace; padded verifier rows are not capacity.",
                "kernel_or_overhead": "PerfStats spans may overlap; no disjoint critical-path attribution or measured byte/bandwidth bound."},
            "statistical_scope": "Paired-block medians/ranges and equal-request geometric means; no significance or promotion claim."}
    # Finite inputs can still overflow during ratios or accumulation. Reject
    # non-finite derived evidence at the public interface, before publication.
    digest(report)
    return report


def main(argv: list[str] | None = None) -> int:
    """Inspect frozen intent or publish one new diagnostic report, never run models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "evaluate"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True, help="digest retained outside candidate write authority")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.command == "evaluate" and (args.evidence is None or args.report is None):
        parser.error("evaluate requires --evidence and --report")
    if args.command == "inspect" and (args.evidence is not None or args.report is not None):
        parser.error("inspect does not consume evidence or publish a report")
    try:
        plan = load_plan(args.plan, args.plan_sha256)
        if args.command == "inspect":
            print(json.dumps(plan.identity(), indent=2, sort_keys=True))
            return 0
        report = evaluate(plan, args.evidence)
        # Reserve a fresh name before atomic publication. Refusing replacement
        # prevents a failed rerun being mistaken for a surviving prior success,
        # and makes every existing input path ineligible as the report target.
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x"):
            pass
        # The shared atomic writer uses a predictable .tmp sibling. Isolate
        # it so that even a pre-existing input with that name remains intact.
        with tempfile.TemporaryDirectory(prefix=".native-evaluation-", dir=args.report.parent) as directory:
            staged = Path(directory) / "report.json"
            write_json(staged, report)
            staged.replace(args.report)
        print(f"Diagnostic comparison written to {args.report}; not a production qualification.")
        return 0
    except (ValueError, KeyError, TypeError, AttributeError, OSError, OverflowError) as error:
        print(f"[native-speculation-evaluation] ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
