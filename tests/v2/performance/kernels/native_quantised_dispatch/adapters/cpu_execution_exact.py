"""Authenticate fused GPUAlignedExpert exact rows through the CPU strong adapter.

The existing adapter remains the byte, effective-route and raw-timing authority.
This extension binds its evidence to the full production key and complete live
bundle witness. Whole-bundle latency is retained as whole-bundle latency; no
per-leaf division, inferred producer or generic training label is introduced.
"""
from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Iterable, Mapping

from .cpu_verifier import (
    CPUVerifierAdapterContext, REQUIRED_RAW_COLUMNS, REQUIRED_TIMING_COLUMNS,
    _timing_key, adapt_cpu_verifier_row,
)
from ..corpus import ObservationCorpus
from ..cpu_exact_overlay import CPUExecutionExactKey, canonical_digest, validate_filled_bundle
from ..format_registry import format_spec
from ..schema import Backend, NativeQuantisedObservation, classify_aspect

EXACT_IDENTITY_COLUMNS = frozenset({
    "exact_key", "descriptor_keys", "exact_key_verified", "numerical_policy",
    "purpose", "partial_route", "team_lifetime", "output_layout", "holdout",
    "descriptor_policies", "baseline_descriptor_policies",
})


def _json_array(raw: str, name: str) -> list:
    """Reject malformed/foreign metadata before it can acquire exact identity."""
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _exact_timing_key(raw: Mapping[str, str]) -> tuple:
    """Extend existing timing identity with all producer and bundle authority."""
    return (_timing_key(raw), *(raw[name] for name in sorted(EXACT_IDENTITY_COLUMNS)))


def adapt_cpu_execution_exact_row(
    raw: Mapping[str, str], context: CPUVerifierAdapterContext,
    timing_samples: tuple[float, ...] | None = None,
    *, native_timing=None,
) -> NativeQuantisedObservation:
    """Bind a strong native row to actual execution identity and live class."""
    missing = EXACT_IDENTITY_COLUMNS.difference(raw)
    if missing:
        raise ValueError(f"CPU execution exact evidence missing fields: {sorted(missing)}")
    if raw["exact_key_verified"] != "1" or raw["numerical_policy"] != "GPUAlignedExpert":
        raise ValueError("CPU exact producer must authenticate actual GPUAlignedExpert plan")
    key = CPUExecutionExactKey.from_scalars(_json_array(raw["exact_key"], "exact_key"))
    if key.rows < 2:
        raise ValueError("CPU row exact overlay requires grouped target rows")
    members = tuple(CPUExecutionExactKey.from_scalars(v) for v in
                    _json_array(raw["descriptor_keys"], "descriptor_keys"))
    validate_filled_bundle(key, members)
    if raw["purpose"] != ("decode", "mtp_verifier", "prefill")[key.operation]:
        raise ValueError("CPU exact purpose disagrees with the original runtime key")
    if raw["partial_route"] != "OutputTileLocal" or raw["output_layout"] != "aligned_expert_major_v1":
        raise ValueError("CPU exact witness changed partial route or output layout")
    if raw["team_lifetime"] not in {"standalone", "retained"} or raw["holdout"] not in {"0", "1"}:
        raise ValueError("CPU exact evidence requires explicit team/holdout provenance")
    if raw["team_lifetime"] == "retained" and key.team_entry != 1:
        raise ValueError("retained fixture must observe its actual existing-region entry")
    spec = format_spec(raw["source_format"])
    if (key.source_codebook, key.source_superblock) != (spec.source_codebook_id, int(spec.source_is_superblock)):
        raise ValueError("CPU exact source tuple disagrees with its concrete source format")
    expected = {"m": key.rows, "n": key.physical_n, "k": key.k,
                "source_codebook": key.source_codebook, "threads": key.workers,
                "k_tiles": key.k_tiles, "n_block_chunks": key.ambient_n_block_chunks}
    if any(int(raw[name]) != value for name, value in expected.items()):
        raise ValueError("CPU exact scalar telemetry disagrees with its actual runtime key")
    if raw["build_isa"] != ("AVX2", "AVX512")[key.build_isa] or raw["runtime_isa_effective"] != ("AVX2", "AVX512")[key.runtime_isa]:
        raise ValueError("CPU exact effective ISA disagrees with its actual runtime key")
    effective = _json_array(raw["descriptor_policies"], "descriptor_policies")
    baseline = _json_array(raw["baseline_descriptor_policies"], "baseline_descriptor_policies")
    requested = {"Pairwise": 0, "cpu.native_quantised.verifier.pairwise": 0,
                 "WideRows": 1, "cpu.native_quantised.verifier.wide_rows": 1}.get(raw["candidate_id"])
    if requested is None or len(effective) != len(members) or len(baseline) != len(members):
        raise ValueError("CPU exact witness requires a complete effective/baseline policy vector")
    if any(type(v) is not int or not 0 <= v <= 0xFFFFFFFF for v in (*effective, *baseline)):
        raise ValueError("CPU exact witness policy vector contains an invalid original policy ID")
    for member, actual, old in zip(members, effective, baseline):
        if actual != (requested if member == key else old):
            raise ValueError("CPU exact replay changed an unrelated descriptor policy")
    base = adapt_cpu_verifier_row(raw, context, timing_samples, native_timing=native_timing)
    if base.backend != Backend.CPU:
        raise ValueError("CPU exact producer cannot be a foreign backend")
    if base.candidate_id not in {"cpu.native_quantised.verifier.pairwise", "cpu.native_quantised.verifier.wide_rows"}:
        raise ValueError("CPU exact K-part class only admits Pairwise/WideRows candidates")
    witness = {**base.physical_launch_evidence,
               "schema": "cpu-execution-exact-witness-v1", "key": key.scalars(),
               "descriptors": [member.scalars() for member in members],
               "timing_basis": "whole_fused_bundle", "team_lifetime": raw["team_lifetime"],
               "holdout": raw["holdout"] == "1",
               "descriptor_policies": effective, "baseline_descriptor_policies": baseline}
    # Existing common operation/bundle fields provide exact separation. The
    # original full vector remains provenance; policy never divides its cost.
    # The common exact oracle sees the actual whole operation and one canonical
    # vector. Only this emitter projects its certified execution-class result
    # onto the smaller runtime key; no latency or work is divided per leaf.
    observation = replace(
        base, operation_kind="CPUExecutionExactFusedBundle",
        bundle_signature=key.signature(),
        projection_n_vector=tuple(member.physical_n for member in members),
        aggregate_n=sum(member.physical_n for member in members),
        work_items=sum(member.physical_n for member in members) * key.k,
        aspect_ratio=sum(member.physical_n for member in members) / key.k,
        aspect_bucket=classify_aspect(sum(member.physical_n for member in members), key.k),
        generic_eligible=False,
        trial_set_hash=canonical_digest({"original": base.trial_set_hash, "witness": witness}),
        physical_launch_evidence=witness,
    )
    observation.validate()
    return observation


def adapt_cpu_execution_exact_csv(
    paths: Iterable[Path], context: CPUVerifierAdapterContext,
    timing_sidecars: Iterable[Path],
) -> ObservationCorpus:
    """Reuse strict strong-row validation, rejecting all unclaimed timing rows."""
    timings: dict[tuple, list[float]] = {}
    for path in timing_sidecars:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = REQUIRED_TIMING_COLUMNS | EXACT_IDENTITY_COLUMNS
            if required.difference(reader.fieldnames or ()):
                raise ValueError(f"{path}: incomplete CPU exact timing identity")
            for raw in reader:
                key = _exact_timing_key(raw)
                values = timings.setdefault(key, [])
                if int(raw["sample_index"]) != len(values):
                    raise ValueError("CPU exact timing indices are not contiguous")
                value = float.fromhex(raw["latency_us_hex"])
                if abs(float(raw["latency_us"]) - value) > 5.1e-10:
                    raise ValueError("CPU exact readable/native timing values disagree")
                values.append(value)
    observations = []
    seen = set()
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if (REQUIRED_RAW_COLUMNS | EXACT_IDENTITY_COLUMNS).difference(reader.fieldnames or ()):
                raise ValueError(f"{path}: incomplete CPU exact aggregate identity")
            for raw in reader:
                key = _exact_timing_key(raw)
                if key in seen:
                    raise ValueError("duplicate CPU exact aggregate/timing identity")
                seen.add(key)
                observations.append(adapt_cpu_execution_exact_row(raw, context, tuple(timings.pop(key, ()))))
    if not observations or timings:
        raise ValueError("CPU exact evidence is empty or has unclaimed timing rows")
    return ObservationCorpus(observations)


def adapt_cpu_execution_exact_native(
    path: Path, context: CPUVerifierAdapterContext, *, holdout: bool,
    installed_choices: Mapping[CPUExecutionExactKey, int] | None = None,
) -> NativeQuantisedObservation:
    """Import retained evidence from the existing fused verifier owner.

    Native JSON is the sample sidecar and retains original double observations.
    Reconstructing the existing strong row reuses its correctness, source alias,
    ISA, route and timing admission; no private CSV producer is introduced.
    """
    from .evidence import native_double_digest
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema=payload.get("schema")
    if schema not in {"cpu_execution_exact_native_v1","cpu_execution_exact_native_v2"}:
        raise ValueError("CPU exact native evidence has an unsupported producer schema")
    tournament=None
    if schema=="cpu_execution_exact_native_v2":
        from ..adaptive_tournament import validate_fused_tournament
        tournament=validate_fused_tournament(payload,minimum_samples=context.minimum_promotion_samples)
        if payload.get("holdout") is not holdout:
            raise ValueError("adaptive native result changed its declared mixed holdout")
    key = CPUExecutionExactKey.from_scalars(payload["exact_key"])
    timing = payload["timing"]; correct = payload["correctness"]; counters = payload["counters"]
    target = payload["target_descriptor"]
    members = payload["descriptor_keys"]
    if type(target) is not int or not 0 <= target < len(members) or tuple(members[target]) != key.scalars():
        raise ValueError("CPU exact native target descriptor changed")
    roots = 2 if payload["output_role"] == "gate_up" else 1
    expected_rows = [rows for rows in payload["expert_rows"] for _ in range(roots)]
    if payload["member_rows"] != expected_rows or [values[9] for values in members] != expected_rows:
        raise ValueError("CPU exact native live row vector changed or included capacity tails")
    raw_samples = payload["raw_samples_us"]
    samples = tuple(payload["sorted_samples_us"])
    if tuple(sorted(raw_samples)) != samples or len(samples) != timing["sample_count"]:
        raise ValueError("CPU exact native raw/sorted sample inventory changed")
    if native_double_digest(samples) != timing["sorted_native_digest"]:
        raise ValueError("CPU exact native timing bytes do not match their original digest")
    if (not correct["bitwise_equal"] or correct["mismatch_count"] != 0 or
        correct["nonfinite_count"] != 0 or correct["guard_replays"] != (4 if tournament is not None else 2) or
        correct["actual_digest"] != correct["expected_digest"]):
        raise ValueError("CPU exact native production bytes or repeated guards failed")
    if (not counters["enabled"] or counters["projection_observations"] != counters["expected_projection_observations"] or
        counters["authentication_calls"] != 1 or not counters["authentication_outside_timing"] or
        counters["authenticated_projection_observations"] != len(payload["descriptor_keys"]) or not counters["route_records"]):
        raise ValueError("CPU exact native evidence is missing its original route counters")
    if tournament is not None:
        expected=(3+(timing["sample_count"]*timing["repeat_count"] if tournament.promotion_grade else 0))*len(members)
        if (not correct.get("repeated_replay") or counters["expected_projection_observations"]!=expected or
                counters.get("sample_observation_basis")!="screen_and_final_timed_calls_only"):
            raise ValueError("adaptive native counters do not separate screening/final/correctness work")
    placement = payload["placement"]
    if placement["workers"] != key.workers or placement["admitted_physical_cores"] < key.workers:
        raise ValueError("CPU exact native placement does not admit its actual physical team")
    if context.profile.installable and not placement["disjoint_single_core_masks"]:
        raise ValueError("CPU exact installation requires authenticated disjoint physical workers")
    requested_candidate = payload["candidate"]
    candidate = requested_candidate
    if installed_choices is not None:
        if requested_candidate != "Auto" or key not in installed_choices:
            raise ValueError("installed validation requires Auto and a measured full-key hit")
        candidate = ("Pairwise", "WideRows")[installed_choices[key]]
        expected_hits = sum(CPUExecutionExactKey.from_scalars(values) in installed_choices
                            for values in payload["descriptor_keys"])
        if expected_hits == 0 or counters["execution_exact_hits"] != expected_hits:
            raise ValueError("installed Auto did not execute the exact generated full-key choices")
    elif candidate not in {"Pairwise", "WideRows"}:
        raise ValueError("Auto/disabled evidence is a control, not a measured exact candidate")
    source_format = payload["source_format"]
    spec = format_spec(source_format)
    for name, expected in (("source_superblock", bool(key.source_superblock)),
                           ("prepared_codebook", key.source_codebook),
                           ("prepared_encoding", key.prepared_encoding)):
        if payload[name] != expected:
            raise ValueError(f"CPU exact native {name} disagrees with its actual key")
    raw = {
        "backend": "cpu", "phase": "verifier_rows", "execution_mode": "eager",
        "source_format": source_format, "source_codebook": str(spec.source_codebook_id),
        "execution_codebook": str(spec.runtime_codebook("cpu")),
        "shape": payload.get("shape", f"CPUExpertExact_N{key.physical_n}_K{key.k}"),
        "m": str(key.rows), "n": str(key.physical_n), "k": str(key.k),
        "candidate_id": candidate, "observed_candidate_id": candidate,
        "build_isa": ("AVX2", "AVX512")[key.build_isa],
        "runtime_isa_requested": ("AVX2", "AVX512")[key.runtime_isa],
        "runtime_isa_effective": ("AVX2", "AVX512")[key.runtime_isa],
        "threads": str(key.workers), "weight_bytes": str(payload["weight_bytes"]),
        "warmup_count": str(timing["warmup_count"]), "sample_count": str(timing["sample_count"]),
        **{name: str(timing[name]) for name in ("min_us", "median_us", "p95_us", "mad_us", "cv")},
        "bit_mismatches": "0", "first_bit_mismatch": "0", "repeat_byte_mismatches": "0",
        **{name: str(correct[name]) for name in ("max_abs", "relative_l2", "cosine", "symmetric_kld")},
        "grouped_output_digest": correct["actual_digest"], "serial_output_digest": correct["expected_digest"],
        "timing_sample_digest": timing["sorted_native_digest"], "route_counter_ok": "1",
        "k_tiles": str(key.k_tiles), "n_block_chunks": str(key.ambient_n_block_chunks),
        "numerical_correctness": "1", "correctness_pass": "1", "is_winner": "0",
        "exact_key": json.dumps(payload["exact_key"], separators=(",", ":")),
        "descriptor_keys": json.dumps(payload["descriptor_keys"], separators=(",", ":")),
        "descriptor_policies": json.dumps(payload["descriptor_policies"], separators=(",", ":")),
        "baseline_descriptor_policies": json.dumps(payload["baseline_descriptor_policies"], separators=(",", ":")),
        "exact_key_verified": "1" if payload["exact_key_verified"] else "0",
        "numerical_policy": payload["numerical_policy"], "purpose": payload["purpose"],
        "partial_route": payload["partial_route"], "team_lifetime": payload["team_lifetime"],
        "output_layout": payload["output_layout"], "holdout": str(int(holdout)),
    }
    result = adapt_cpu_execution_exact_row(raw, context, samples, native_timing=tournament)
    return replace(result, physical_launch_evidence={**result.physical_launch_evidence,
        "native_evidence_path": str(Path(path).resolve()),
        "native_evidence_mapping_digest": canonical_digest(payload),
        "requested_candidate": requested_candidate,
        "installed_generated_lookup_verified": installed_choices is not None,
        "target_descriptor": payload["target_descriptor"], "placement": placement,
        "expert_rows": payload["expert_rows"], "output_role": payload["output_role"]})
