#!/usr/bin/env python3
"""Validate the server E2E GPU graph-capture topology contract.

The server harness emits one PerfStats document after every matrix cell. This
module turns the graph-plan records in that document into a strict architectural
gate:

* Homogeneous CUDA-only and ROCm-only cells must use one fully captured graph.
* Segmented plans and segmented replay are forbidden for homogeneous cells.
* Segmentation requires a mixed device-type topology and either in-graph
  collectives or a completely materialized/replayed pipeline or sparse-overlay
  transaction boundary. Child TP collectives do not describe an overlay's
  cross-rank sparse-return protocol. Rank-local overlays instead authenticate
  the sealed retained parent's concurrent CPU ticket service and its completed
  submissions; they do not manufacture a cross-rank coordinator.

Keeping topology parsing and record classification here makes the policy
unit-testable. The shell harness remains responsible for feature-specific
counters such as MTP acceptance and prefix-cache movement.

Physical proof follows the installed executable family: one full graph,
retained parent, or independently instantiated heterogeneous segments. Segment
proof joins each unit's capture and launch by rank/device/context and stage
identity; a neighboring executable or a graph-only child cannot satisfy it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


_DEVICE_KIND_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(cuda|rocm|cpu):[0-9]+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GraphCaptureValidation:
    """Result of validating one E2E cell's graph-capture evidence."""

    error: str | None
    device_kinds: frozenset[str]
    heterogeneous_device_mix: bool
    has_collective_evidence: bool
    has_pipeline_boundary_evidence: bool
    has_overlay_boundary_evidence: bool
    has_full_graph_plan: bool
    has_nonempty_full_graph_executable: bool
    has_segmented_execution: bool
    prefill_lifecycle_complete: bool
    missing_prefill_phases: tuple[str, ...]
    incomplete_contexts: tuple[str, ...]


class DecodeGraphRequirement(Enum):
    """Evidence fragment, short captured probe, or repeated production decode."""

    OBSERVE = "observe"
    CAPTURE = "capture"
    REPLAY = "replay"


def device_kinds_for_cell(backend: str, extra_flags: str) -> frozenset[str]:
    """Return every device kind explicitly present in one matrix cell.

    The regular expression intentionally scans complete option values instead
    of maintaining parsers for each domain grammar. It therefore recognizes:

    * ``--tp-devices cuda:0,cuda:1``
    * ``--define-domain stage=cuda:0,rocm:0;...``
    * ``--device-map 0=cuda:0,1=rocm:0``
    * ``--moe-routed-expert-domain name=0:cpu:0,1:cpu:0;...``

    Unknown or auto-selected ``tp``/``pp`` cells remain conservatively
    non-heterogeneous. They cannot use the segmentation exception without an
    explicit, auditable mixed topology.
    """

    haystack = f"{backend} {extra_flags}"
    return frozenset(
        match.group(1).lower()
        for match in _DEVICE_KIND_PATTERN.finditer(haystack)
    )


def _numeric(value: Any) -> float:
    """Convert a PerfStats value to a comparison-safe floating-point number."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _record_value(record: Mapping[str, Any]) -> float:
    """Return the value/count field used by counter-style PerfStats records."""

    return _numeric(record.get("value", record.get("count", 0.0)))


def _device_owner(record: Mapping[str, Any]) -> str:
    """Qualify process-local graph identity in an all-rank aggregate.

    Unqualified records still support validation of one raw diagnostic file.
    The E2E aggregate always supplies rank, verified against server membership.
    """
    device = str(record.get("device", ""))
    return f"rank={record['rank']}/{device}" if "rank" in record else device


def _has_collective_evidence(records: Iterable[Mapping[str, Any]]) -> bool:
    """Return true when runtime policy records prove collective graph nodes."""

    for record in records:
        tags = record.get("tags") or {}
        if tags.get("has_collectives") == "true":
            return True
    return False


def _pipeline_boundary_evidence(
    records: Sequence[Mapping[str, Any]],
) -> tuple[bool, tuple[str, ...]]:
    """Authenticate rank-local PP boundaries separately from child collectives.

    A fully captured GPU child followed by a CPU child has no collective node
    inside either child graph. The frozen pipeline plan owns this boundary.
    Join its exact geometry, materialization and completed transactions by rank;
    child native executable checks remain independently mandatory below.
    """
    owners: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        if (record.get("domain") == "forward_graph"
                and record.get("device") == "pipeline_coordinator"
                and (record.get("tags") or {}).get("boundary_authority") == "rank_pipeline_graph_plan"):
            owners.setdefault(_device_owner(record), []).append(record)
    errors = []
    for owner, rows in owners.items():
        plans = [r for r in rows if r.get("name") == "segmented_plan_segments"]
        captures = [r for r in rows if r.get("name") == "segmented_graph_capture_segments"]
        replays = [r for r in rows if r.get("name") == "segmented_replay_segments"]
        try:
            if len(plans) != 1 or len(captures) != 1 or not replays:
                raise ValueError("missing/ambiguous plan, materialization or replay")
            tags = plans[0].get("tags") or {}
            geometry = tuple(int(tags[key]) for key in ("total_segments", "native_segments", "host_segments"))
            total, native, host = geometry
            if total < 2 or native < 1 or host < 0 or native + host != total:
                raise ValueError("invalid native/host segment geometry")
            for record, expected in ((plans[0], total), (captures[0], native)):
                evidence = record.get("tags") or {}
                if (tuple(int(evidence[key]) for key in ("total_segments", "native_segments", "host_segments")) != geometry
                        or evidence.get("heterogeneous_segmented") != "true"
                        or record.get("phase") != "setup"
                        or _record_value(record) != expected * _numeric(record.get("count", 1))
                        or _numeric(record.get("count", 1)) <= 0):
                    raise ValueError("materialized inventory differs from pipeline plan")
            phases = set()
            for record in replays:
                evidence = record.get("tags") or {}
                transactions = int(evidence["transactions"])
                count = _numeric(record.get("count", 1))
                if (int(evidence["segments_per_transaction"]) != total or transactions <= 0 or count <= 0
                        or evidence.get("heterogeneous_segmented") != "true"
                        or _record_value(record) != total * transactions * count):
                    raise ValueError("incomplete pipeline transaction")
                phases.add(record.get("phase"))
            if not {"prefill", "decode"} <= phases:
                raise ValueError("pipeline has no completed prefill/decode pair")
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{owner}: {error}")
    return bool(owners) and not errors, tuple(errors)


def _overlay_boundary_evidence(
    records: Sequence[Mapping[str, Any]],
) -> tuple[bool, tuple[str, ...]]:
    """Join the sealed cross-rank overlay plan to completed sparse returns.

    The transaction coordinator publishes this plan after every serving family
    is prepared, and publishes replay only after the exact sparse-return fence
    and follower-slot retirement. A child's ``has_collectives`` describes TP
    nodes, so it can truthfully be false for this different boundary authority.
    One immutable plan per continuation rank must agree with its materialized
    inventory and every retired sequence. A command may contain many prefill
    chunks or device-selected generation rounds; the coordinator's existing
    sequence identity distinguishes them without inferring another lifecycle
    from counter multiplicities. Physical GPU executable/launch proof
    remains independently mandatory in ``_incomplete_graph_contexts``.
    """
    owners: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        if (record.get("domain") == "forward_graph"
                and record.get("device") == "continuation_rank"
                and (record.get("tags") or {}).get("authority") == "typed_overlay_transaction_plan"):
            owners.setdefault(_device_owner(record), []).append(record)
    errors = []
    geometry_keys = ("continuation_rank", "graph_family_generation", "follower_segments",
                     "native_segments", "eager_host_segments", "native_participants")
    for owner, rows in owners.items():
        plans = [r for r in rows if r.get("name") == "segmented_plan_segments"]
        captures = [r for r in rows if r.get("name") == "segmented_graph_capture_segments"]
        replays = [r for r in rows if r.get("name") == "segmented_replay_segments"]
        try:
            if len(plans) != 1 or len(captures) != 1 or not replays:
                raise ValueError("missing/ambiguous plan, materialization or retirement")
            tags = plans[0].get("tags") or {}
            geometry = tuple(int(tags[key]) for key in geometry_keys)
            rank, generation, followers, native, host, participants = geometry
            total = followers + 1
            if (rank < 0 or generation <= 0 or followers < 1 or native < 1 or host < 0
                    or native + host != total or participants < native
                    or int(plans[0]["rank"]) != rank):
                raise ValueError("invalid rank/generation or native/host segment geometry")
            for record, expected in ((plans[0], total), (captures[0], native)):
                evidence = record.get("tags") or {}
                # Setup is one sealed plan, not a sum of unrelated generations.
                if (tuple(int(evidence[key]) for key in geometry_keys) != geometry
                        or record.get("phase") != "setup"
                        or _numeric(record.get("count", 1)) != 1
                        or _record_value(record) != expected):
                    raise ValueError("materialized inventory differs from overlay plan")
            phases, sequences = set(), set()
            for record in replays:
                evidence = record.get("tags") or {}
                command, sequence, groups, depth = (int(evidence[key]) for key in
                    ("command", "sequence", "graph_groups", "draft_depth"))
                phase = record.get("phase")
                if (int(evidence["plan_segments"]) != total or groups <= 0 or depth < 0
                        or command <= 0 or sequence <= 0 or sequence in sequences
                        or evidence.get("terminal") != "sparse_return_retired"
                        or phase not in {"prefill", "decode", "mtp"}
                        or (phase != "mtp" and depth != 0)
                        or _numeric(record.get("count", 1)) != 1
                        or _record_value(record) != total * groups):
                    raise ValueError("incomplete or duplicate overlay transaction")
                phases.add(phase)
                sequences.add(sequence)
            if any((r.get("tags") or {}).get("scope") != "cross_rank_expert_overlay" for r in rows):
                raise ValueError("wrong overlay boundary scope")
            if "prefill" not in phases or not {"decode", "mtp"} & phases:
                raise ValueError("overlay has no completed prefill/generation pair")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            errors.append(f"{owner}: {error}")
    return bool(owners) and not errors, tuple(errors)


def _retained_ticket_boundary_evidence(
    records: Sequence[Mapping[str, Any]],
) -> tuple[bool, tuple[str, ...]]:
    """Join a retained CPU-service boundary to its own native parent lifecycle.

    The executor seals the physical service-program inventory with its parent.
    It emits initial/repeat submission only after both parent launch and the
    concurrent CPU worker succeed; GPU completion remains event ordered. Use
    those existing records, not a new execution state or a cross-rank counter.
    Logical layer cutpoints may collapse into one physical service program.
    Unused setup families are allowed, but cannot prove a completed boundary.
    """
    names = {"retained_parent_executable_nodes", "retained_parent_transaction_zero_launches",
             "retained_parent_replays"}
    owners: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        if record.get("domain") == "forward_graph" and record.get("name") in names:
            key = (_device_owner(record), str((record.get("tags") or {}).get("context", "")))
            owners.setdefault(key, []).append(record)
    errors, completed = [], False
    for (owner, context), rows in owners.items():
        if not any((r.get("tags") or {}).get("boundary_authority") == "concurrent_ticket_service"
                   or _numeric((r.get("tags") or {}).get("ticket_service_units")) > 0 for r in rows):
            continue
        try:
            geometry, observed = None, set()
            for record in rows:
                tags = record.get("tags") or {}
                units = (int(tags["child_units"]), int(tags["ticket_service_units"]))
                count, value = _numeric(record.get("count", 1)), _record_value(record)
                if (not context or not re.fullmatch(r"(?:cuda|rocm):[0-9]+", str(record.get("device", "")), re.I)
                        or min(units) <= 0 or (geometry is not None and units != geometry)
                        or tags.get("boundary_authority") != "concurrent_ticket_service"
                        or not math.isfinite(count) or count <= 0 or not count.is_integer()
                        or not math.isfinite(value) or value <= 0
                        or (record["name"] != "retained_parent_executable_nodes" and value != count)):
                    raise ValueError("inconsistent native parent/service inventory or completion count")
                geometry = units
                observed.add(record["name"])
            if "retained_parent_executable_nodes" not in observed:
                raise ValueError("ticket service has no matching native parent")
            if "retained_parent_replays" in observed and "retained_parent_transaction_zero_launches" not in observed:
                raise ValueError("ticket replay has no completed initial submission")
            completed |= "retained_parent_transaction_zero_launches" in observed
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            errors.append(f"{owner}:{context}: {error}")
    return completed and not errors, tuple(errors)


def _has_full_graph_plan(records: Iterable[Mapping[str, Any]]) -> bool:
    """Return true when at least one non-empty full-graph plan was published."""

    for record in records:
        name = str(record.get("name", ""))
        tags = record.get("tags") or {}
        if (
            name == "full_graph_plan_graphs"
            and tags.get("type") == "capturable"
            and _record_value(record) > 0.0
        ):
            return True
        if (
            name == "graph_replay_plan_graphs"
            and tags.get("source") == "full_graph_capture"
            and tags.get("type") == "capturable"
            and _record_value(record) > 0.0
        ):
            return True
    return False


def _is_nonempty_full_graph_executable(record: Mapping[str, Any]) -> bool:
    """Recognize physical instantiation independently of initial submission.

    Setup may instantiate without launching; the per-context lifecycle below
    separately requires a launch for runtime admissions. Both callers use this
    one predicate so an optimized setup cannot pass one check and fail another.
    A retained graph-only child is not an executable and never qualifies.
    """
    tags = record.get("tags") or {}
    return (
        record.get("domain") == "forward_graph"
        and record.get("name") == "full_graph_capture_executable_nodes"
        and tags.get("source") == "full_graph_capture"
        and tags.get("type") in {"captured_executable", "materialized_unlaunched_executable"}
        and _record_value(record) > 0.0
    )


def _has_segmented_execution(records: Iterable[Mapping[str, Any]]) -> bool:
    """Return true for any segmented policy, plan, replay, or attribution.

    Capture-policy records are emitted before execution selects or builds a
    replay plan. Treating an admitted segmented policy as evidence keeps the
    E2E contract fail-closed even if a later instrumentation defect omits the
    segmented plan/replay counters. Homogeneous cells must publish
    ``require_full_graph`` at every main and MTP sidecar capture boundary.
    """

    for record in records:
        name = str(record.get("name", ""))
        tags = record.get("tags") or {}
        if name.startswith("retained_parent_"):
            return True
        if tags.get("heterogeneous_segmented") == "true":
            return True
        if (
            tags.get("replay_plan_policy")
            == "allow_heterogeneous_boundary_segmentation"
        ):
            return True
        if tags.get("heterogeneous_segmentation_admitted") == "true":
            return True
        if name.startswith(("segmented_plan_", "segmented_replay_")):
            return True
        if name == "graph_replay_plan_segments":
            return True
        if tags.get("source") == "segmented_graph_capture":
            return True
        if tags.get("graph_capture_scope") in {
            "segmented_capture_plan",
            "segmented_replay_events",
            "segmented_replay_host",
        }:
            return True
    return False


def _missing_prefill_phases(
    records: Iterable[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Require capture/materialization then replay for each used prefill owner.

    Setup captures without launching a synthetic request. Requiring an eager
    warmup would reject the installed production lifecycle. Unused materialized
    buckets are allowed; every replay must have matching capture evidence.
    """
    captured: set[tuple[str, ...]] = set()
    replayed: set[tuple[str, ...]] = set()
    eager = False
    for record in records:
        if record.get("name") != "prefill_graph_phase" or _record_value(record) <= 0:
            continue
        tags = record.get("tags") or {}
        phase = str(tags.get("capture_phase", ""))
        key = (_device_owner(record), *(str(tags.get(field, "")) for field in
            ("bucket_seq_len", "domain_id", "participant_id", "placement_epoch", "topology_signature")))
        if phase == "warmup":
            eager = True
        elif phase == "capture" or (phase == "materialized_without_launch" and tags.get("cache_phase") == "ready"):
            captured.add(key)
        elif phase == "replay":
            replayed.add(key)
    missing = []
    if eager:
        missing.append("retired eager warmup executed")
    if not captured or replayed - captured:
        missing.append("capture")
    if not replayed:
        missing.append("replay")
    return tuple(missing)


def _incomplete_graph_contexts(
    records: Iterable[Mapping[str, Any]],
    *,
    allow_heterogeneous_execution: bool = False,
) -> tuple[str, ...]:
    """Return GPU graph contexts that did not advance through their lifecycle.

    ``decode_graph_phase`` aggregates both setup captures and live submissions
    by device, context, and phase. Count explicitly unlaunched materializations
    separately: building two setup shapes is not two inference invocations.
    Live policy admissions and capture-with-launch/replay phases establish the
    execution obligation. The retired eager ``warmup`` phase is always an
    ownership defect, not acceptable graph-planning evidence.

    A capture phase needs the physical executable family selected by topology.
    Full graphs, retained parents and heterogeneous segments keep separate
    proofs; none can borrow another rank, device or context's executable.
    """

    # Multiple independent family observers consume the same immutable rows.
    # Materialize references once so an iterator cannot lose later evidence.
    records = tuple(records)
    phase_counts: dict[tuple[str, str], dict[str, float]] = {}
    sidecar_path_counts: dict[tuple[str, str, str], dict[str, float]] = {}
    executable_contexts: set[tuple[str, str]] = set()
    materialized_contexts: set[tuple[str, str]] = set()
    materialized_capture_counts: dict[tuple[str, str], float] = {}
    parent_counts: dict[tuple[str, str], dict[str, float]] = {}
    runtime_invocations: dict[tuple[str, str], float] = {}
    parent_metrics = {
        "retained_parent_executable_nodes": "nodes",
        "retained_parent_materialized_without_launch": "materialized",
        "retained_parent_transaction_zero_launches": "initial",
        "retained_parent_replays": "replay",
    }

    for record in records:
        name = str(record.get("name", ""))
        tags = record.get("tags") or {}
        device = _device_owner(record)
        context = str(tags.get("context", ""))
        if not context:
            continue

        key = (device, context)
        if name in {"decode_capture_policy", "sidecar_decode_capture_policy"}:
            runtime_invocations[key] = runtime_invocations.get(key, 0.0) + _record_value(record)
        if name in parent_metrics and record.get("domain") == "forward_graph":
            counts = parent_counts.setdefault(key, {})
            # Graph-only compilation children are not executable evidence.
            # Every physical parent also declares its nonempty child inventory.
            value = _record_value(record)
            if _numeric(tags.get("child_units")) > 0.0 and value > 0.0:
                metric = parent_metrics[name]
                counts[metric] = counts.get(metric, 0.0) + value
        if name == "decode_graph_phase":
            phase = str(tags.get("phase", ""))
            if phase in {"warmup", "capture", "replay"}:
                counts = phase_counts.setdefault(key, {})
                counts[phase] = counts.get(phase, 0.0) + _record_value(record)
        elif name == "sidecar_graph_capture_path":
            seq_len = str(tags.get("seq_len", ""))
            path = str(tags.get("path", ""))
            if seq_len and path:
                # A sidecar executable can serve multiple logical invocation
                # roles. Its immutable graph_context is the lifecycle owner;
                # grouping by the caller-facing context would conceal repeated
                # rebuilds and split capture/replay evidence across aliases.
                graph_context = str(tags.get("graph_context", context))
                sidecar_key = (device, graph_context, seq_len)
                counts = sidecar_path_counts.setdefault(sidecar_key, {})
                counts[path] = counts.get(path, 0.0) + _record_value(record)
        elif _is_nonempty_full_graph_executable(record):
            executable_contexts.add(key)
            if tags.get("type") == "materialized_unlaunched_executable":
                materialized_contexts.add(key)
                # value is the number of device nodes, not captures. Only the
                # physical record's sample count can discharge setup phases.
                materialized_capture_counts[key] = (
                    materialized_capture_counts.get(key, 0.0)
                    + max(0.0, _numeric(record.get("count", 0.0))))

    segmented_contexts, segment_errors = (
        _segmented_executable_contexts(records) if allow_heterogeneous_execution else (set(), ())
    )
    incomplete: list[str] = list(segment_errors)
    proven_parents: set[tuple[str, str]] = set()
    if allow_heterogeneous_execution:
        for key, counts in sorted(parent_counts.items()):
            reasons = []
            if counts.get("nodes", 0.0) <= 0.0:
                reasons.append("retained parent has no context-matched executable nodes")
            if counts.get("materialized", 0.0) + counts.get("initial", 0.0) <= 0.0:
                reasons.append("retained parent has no materialization or transaction-zero launch")
            if counts.get("replay", 0.0) > 0.0 and counts.get("initial", 0.0) <= 0.0:
                reasons.append("retained parent replay has no transaction-zero launch")
            # Capture counters aggregate several setup shapes. They are not
            # inference calls: an unused materialized family needs no replay.
            if (max(counts.get("initial", 0.0), runtime_invocations.get(key, 0.0)) > 1.0
                    and counts.get("replay", 0.0) <= 0.0):
                reasons.append("missing retained-parent replay after repeated execution")
            if reasons:
                incomplete.append(f"{key[0]}:{key[1]} ({'; '.join(reasons)})")
            else:
                proven_parents.add(key)
    for key, counts in sorted(phase_counts.items()):
        device, context = key
        total = sum(counts.values())
        capture_count = counts.get("capture", 0.0)
        replay_count = counts.get("replay", 0.0)
        warmup_count = counts.get("warmup", 0.0)
        reasons: list[str] = []
        if warmup_count > 0.0:
            reasons.append("retired eager warmup phase was executed")
        if allow_heterogeneous_execution and (key in parent_counts or key in segmented_contexts):
            # Each heterogeneous family was validated independently against its
            # actual physical executables. Eager execution is still forbidden.
            if warmup_count > 0.0:
                incomplete.append(f"{device}:{context} ({'; '.join(reasons)})")
            continue
        if total >= 1.0 and capture_count <= 0.0 and key not in materialized_contexts:
            reasons.append("missing transaction-zero capture")
        if capture_count > 0.0 and key not in executable_contexts:
            reasons.append("capture has no context-matched executable nodes")
        initial_launches = max(0.0, capture_count - materialized_capture_counts.get(key, 0.0))
        admitted = runtime_invocations.get(key, 0.0)
        if admitted > 0.0 and initial_launches + replay_count <= 0.0:
            reasons.append("missing launch after runtime admission")
        if max(initial_launches + replay_count + warmup_count, admitted) >= 2.0 and replay_count <= 0.0:
            reasons.append("missing replay after repeated execution")
        if reasons:
            label = f"{device or 'unknown'}:{context}"
            incomplete.append(f"{label} ({'; '.join(reasons)})")

    for key, counts in sorted(sidecar_path_counts.items()):
        device, context, seq_len = key
        total = sum(counts.values())
        rebuild_count = counts.get("plain_after_build", 0.0)
        replay_count = counts.get("full_graph", 0.0)
        reasons = []
        if counts.get("retained_parent", 0.0) > 0.0:
            parent_key = (device, context)
            if not allow_heterogeneous_execution or parent_key not in proven_parents:
                reasons.append("sidecar has no matching certified retained parent")
            else:
                replay_count += parent_counts[parent_key].get("replay", 0.0)
        if rebuild_count > 1.0:
            reasons.append(
                f"rebuilt graph {rebuild_count:g} times for one stable shape"
            )
        if total >= 3.0 and replay_count <= 0.0:
            reasons.append("missing full-graph replay after repeated execution")
        if reasons:
            label = (
                f"{device or 'unknown'}:{context}"
                f"[seq_len={seq_len}]"
            )
            incomplete.append(f"{label} ({'; '.join(reasons)})")

    return tuple(incomplete)


def _segmented_executable_contexts(
    records: Sequence[Mapping[str, Any]],
) -> tuple[set[tuple[str, str]], tuple[str, ...]]:
    """Authenticate every heterogeneous segment's physical capture/launch pair.

    The publisher names units by their first/last stage and stage count within
    one rank/device/context. Setup may materialize many unused bucket shapes;
    their node counts are not launch counts. A used context must launch every
    unit, each launch must have its own nonempty executable, and a materialized
    family must publish transaction zero before it can claim replay. Runtime
    graph-cache identity separately authenticates embedded pointers/geometry.
    """
    nodes: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
    launches: dict[tuple[str, str], dict[tuple[str, str, str], float]] = {}
    materialized: set[tuple[str, str]] = set()
    initial: set[tuple[str, str]] = set()
    invocations: dict[tuple[str, str], float] = {}
    errors: list[str] = []
    for record in records:
        if record.get("domain") != "forward_graph":
            continue
        tags = record.get("tags") or {}
        context = str(tags.get("context", ""))
        if not context:
            continue
        key = (_device_owner(record), context)
        name, value = record.get("name"), _record_value(record)
        if name == "materialized_graph_transaction_zero_launches" and value > 0:
            initial.add(key)
        if name == "decode_capture_policy" or (
                name == "decode_graph_phase" and tags.get("phase") == "replay"):
            invocations[key] = max(invocations.get(key, 0.0), value)
        is_node = name == "segmented_graph_capture_executable_nodes"
        is_launch = name == "segmented_replay_segments" and tags.get("type") == "capturable"
        if not (is_node or is_launch):
            continue
        # Keep even an invalid node's context in the comparison. Otherwise a
        # malformed inventory could disappear rather than fail closed.
        nodes.setdefault(key, set())
        stage = tuple(str(tags.get(field, "")) for field in
                      ("first_stage", "last_stage", "stage_count"))
        if (not stage[0] or not stage[1] or not stage[2].isdigit()
                or int(stage[2]) <= 0 or not 0 < value < float("inf")):
            errors.append(f"{key[0]}:{context} (invalid physical segment identity/count)")
            continue
        if is_node:
            if (tags.get("source") != "segmented_graph_capture" or tags.get("type") not in
                    {"captured_executable", "materialized_unlaunched_executable"}):
                errors.append(f"{key[0]}:{context} (segment is not an instantiated executable)")
                continue
            nodes[key].add(stage)
            if tags.get("type") == "materialized_unlaunched_executable":
                materialized.add(key)
        else:
            counts = launches.setdefault(key, {})
            counts[stage] = counts.get(stage, 0.0) + value

    for key, captured in sorted(nodes.items()):
        replay = launches.get(key, {})
        reasons = []
        if not captured:
            reasons.append("no nonempty instantiated segments")
        if replay or invocations.get(key, 0.0) > 0:
            if set(replay) - captured:
                reasons.append("segment replay has no matching executable")
            if captured - set(replay):
                reasons.append("instantiated segment has no matching launch")
            if key in materialized and key not in initial:
                reasons.append("materialized segments have no transaction-zero launch")
            if invocations.get(key, 0.0) > 1 and any(count < 2 for count in replay.values()):
                reasons.append("segment missing replay after repeated execution")
        if reasons:
            errors.append(f"{key[0]}:{key[1]} ({'; '.join(reasons)})")
    return set(nodes), tuple(errors)


def validate_graph_capture_policy(
    records: Sequence[Mapping[str, Any]],
    backend: str,
    extra_flags: str,
    *,
    require_prefill_lifecycle: bool = False,
    decode_requirement: DecodeGraphRequirement = DecodeGraphRequirement.OBSERVE,
) -> GraphCaptureValidation:
    """Validate full-graph/segmented execution against the cell topology.

    Returns a structured result instead of raising so the shell harness can
    report one concise failure without a Python traceback.
    """

    device_kinds = device_kinds_for_cell(backend, extra_flags)
    heterogeneous_device_mix = len(device_kinds) > 1
    has_collective_evidence = _has_collective_evidence(records)
    has_pipeline_boundary_evidence, pipeline_errors = _pipeline_boundary_evidence(records)
    has_overlay_boundary_evidence, overlay_errors = _overlay_boundary_evidence(records)
    has_local_ticket_boundary, local_ticket_errors = _retained_ticket_boundary_evidence(records)
    has_overlay_boundary_evidence |= has_local_ticket_boundary
    overlay_errors += local_ticket_errors
    has_boundary_evidence = (has_collective_evidence or has_pipeline_boundary_evidence
                             or has_overlay_boundary_evidence)
    has_full_graph_plan = _has_full_graph_plan(records)
    has_nonempty_full_graph_executable = any(
        _is_nonempty_full_graph_executable(record) for record in records)
    has_segmented_execution = _has_segmented_execution(records)
    missing_prefill_phases = (
        _missing_prefill_phases(records)
        if require_prefill_lifecycle
        else ()
    )
    prefill_lifecycle_complete = not missing_prefill_phases
    incomplete_contexts = _incomplete_graph_contexts(
        records,
        allow_heterogeneous_execution=heterogeneous_device_mix and has_boundary_evidence,
    )
    # Retained parents publish their own replay counter instead of the legacy
    # decode_graph_phase replay. Validate both physical families here, after
    # joining captures/launches above; the shell must not apply a second,
    # full-graph-only interpretation. Prefill/sidecar work cannot stand in for
    # replay of the model's serial decode or grouped verification executable.
    has_decode_replay = any(
        r.get("domain") == "forward_graph"
        and (r.get("tags") or {}).get("context") in {"main_decode", "main_verifier"}
        and _record_value(r) > 0
        and ((r.get("name") == "decode_graph_phase"
              and (r.get("tags") or {}).get("phase") == "replay")
             or (r.get("name") == "retained_parent_replays"
                 and heterogeneous_device_mix and has_boundary_evidence))
        for r in records
    )
    has_decode_capture = any(
        r.get("domain") == "forward_graph" and r.get("name") == "decode_graph_phase"
        and (r.get("tags") or {}).get("context") in {"main_decode", "main_verifier"}
        and (r.get("tags") or {}).get("phase") == "capture"
        and _record_value(r) > 0 for r in records
    )

    error: str | None = None
    if has_segmented_execution and not heterogeneous_device_mix:
        error = (
            "segmented GPU graph execution is forbidden for a homogeneous "
            f"device domain (device kinds: {sorted(device_kinds) or ['unknown']})"
        )
    elif pipeline_errors:
        error = "Incomplete pipeline graph boundary: " + "; ".join(pipeline_errors)
    elif overlay_errors:
        error = "Incomplete ExpertOverlay graph boundary: " + "; ".join(overlay_errors)
    elif has_segmented_execution and not has_boundary_evidence:
        error = (
            "segmented GPU graph execution is admissible only for a "
            "heterogeneous graph with runtime-proven collective nodes or a complete "
            "pipeline/ExpertOverlay boundary"
        )
    elif not has_full_graph_plan and not has_segmented_execution:
        error = (
            "GPU case emitted neither a full-graph plan nor an admissible "
            "heterogeneous collective segmented plan"
        )
    elif has_full_graph_plan and not has_nonempty_full_graph_executable:
        error = (
            "full-graph planning succeeded but PerfStats did not prove a "
            "non-empty instantiated GPU graph executable"
        )
    elif missing_prefill_phases:
        error = (
            "full-tier GPU prefill graph did not complete "
            "capture/materialization and replay lifecycle; missing phases: "
            + ", ".join(missing_prefill_phases)
        )
    elif incomplete_contexts:
        error = (
            "GPU graph contexts did not complete capture/replay "
            "lifecycle: " + ", ".join(incomplete_contexts)
        )
    elif decode_requirement is not DecodeGraphRequirement.OBSERVE and not has_decode_capture:
        error = "GPU case emitted no context-matched decode graph capture evidence"
    elif decode_requirement is DecodeGraphRequirement.REPLAY and not has_decode_replay:
        error = "GPU case expected context-matched decode graph replay but emitted no replay evidence"

    return GraphCaptureValidation(
        error=error,
        device_kinds=device_kinds,
        heterogeneous_device_mix=heterogeneous_device_mix,
        has_collective_evidence=has_collective_evidence,
        has_pipeline_boundary_evidence=has_pipeline_boundary_evidence,
        has_overlay_boundary_evidence=has_overlay_boundary_evidence,
        has_full_graph_plan=has_full_graph_plan,
        has_nonempty_full_graph_executable=(
            has_nonempty_full_graph_executable
        ),
        has_segmented_execution=has_segmented_execution,
        prefill_lifecycle_complete=prefill_lifecycle_complete,
        missing_prefill_phases=missing_prefill_phases,
        incomplete_contexts=incomplete_contexts,
    )
