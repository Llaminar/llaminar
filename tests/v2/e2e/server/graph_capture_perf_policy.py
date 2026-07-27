#!/usr/bin/env python3
"""Validate the server E2E GPU graph-capture topology contract.

The server harness emits one PerfStats document after every matrix cell. This
module turns the graph-plan records in that document into a strict architectural
gate:

* Homogeneous CUDA-only and ROCm-only cells must use one fully captured graph.
* Segmented plans and segmented replay are forbidden for homogeneous cells.
* Segmentation is admissible only when command-line topology proves a mixed
  device-type domain and PerfStats proves that the graph contains collectives.

Keeping topology parsing and record classification here makes the policy
unit-testable. The shell harness remains responsible for feature-specific
counters such as MTP acceptance and prefix-cache movement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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
    has_full_graph_plan: bool
    has_nonempty_full_graph_executable: bool
    has_segmented_execution: bool
    prefill_lifecycle_complete: bool
    missing_prefill_phases: tuple[str, ...]
    incomplete_contexts: tuple[str, ...]


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


def _has_collective_evidence(records: Iterable[Mapping[str, Any]]) -> bool:
    """Return true when runtime policy records prove collective graph nodes."""

    for record in records:
        tags = record.get("tags") or {}
        if tags.get("has_collectives") == "true":
            return True
    return False


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


def _has_nonempty_full_graph_executable(
    records: Iterable[Mapping[str, Any]],
) -> bool:
    """Return true when capture instantiated a full graph with device nodes."""

    for record in records:
        name = str(record.get("name", ""))
        tags = record.get("tags") or {}
        if (
            name == "full_graph_capture_executable_nodes"
            and tags.get("source") == "full_graph_capture"
            and tags.get("type") == "captured_executable"
            and _record_value(record) > 0.0
        ):
            return True
    return False


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
        if tags.get("collective_segmented") == "true":
            return True
        if (
            tags.get("replay_plan_policy")
            == "allow_heterogeneous_collective_segmentation"
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
    """Return required prefill graph lifecycle phases absent from PerfStats.

    A full-tier GPU request matrix sends enough bucket-compatible prompts to
    move a persistent prefill executable through warmup, capture, and replay.
    Merely proving a decode graph exists would leave model admission outside
    the graph-capture architecture and conceal repeated prefill warmup caused
    by stale cache ownership.
    """

    observed: set[str] = set()
    for record in records:
        if record.get("name") != "prefill_graph_phase":
            continue
        tags = record.get("tags") or {}
        phase = str(tags.get("capture_phase", ""))
        if phase in {"warmup", "capture", "replay"} and _record_value(record) > 0.0:
            observed.add(phase)

    return tuple(
        phase
        for phase in ("warmup", "capture", "replay")
        if phase not in observed
    )


def _incomplete_graph_contexts(
    records: Iterable[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return GPU graph contexts that did not advance through their lifecycle.

    ``decode_graph_phase`` is emitted once for each invocation and aggregated
    by device, context, and phase. A context may legitimately execute only its
    warmup once in a tiny cell. Once it is exercised twice, however, the second
    invocation must capture. Once it is exercised three times, the third must
    replay. Repeated warmup is therefore an ownership/reset defect, not
    acceptable evidence of graph planning.

    A capture phase counts only when PerfStats also reports a non-empty
    instantiated full-graph executable for the same device and context. This
    prevents an unrelated MTP helper graph from satisfying the main verifier's
    requirement.
    """

    phase_counts: dict[tuple[str, str], dict[str, float]] = {}
    executable_contexts: set[tuple[str, str]] = set()

    for record in records:
        name = str(record.get("name", ""))
        tags = record.get("tags") or {}
        device = str(record.get("device", ""))
        context = str(tags.get("context", ""))
        if not context:
            continue

        key = (device, context)
        if name == "decode_graph_phase":
            phase = str(tags.get("phase", ""))
            if phase in {"warmup", "capture", "replay"}:
                counts = phase_counts.setdefault(key, {})
                counts[phase] = counts.get(phase, 0.0) + _record_value(record)
        elif (
            name == "full_graph_capture_executable_nodes"
            and tags.get("source") == "full_graph_capture"
            and tags.get("type") == "captured_executable"
            and _record_value(record) > 0.0
        ):
            executable_contexts.add(key)

    incomplete: list[str] = []
    for key, counts in sorted(phase_counts.items()):
        device, context = key
        total = sum(counts.values())
        capture_count = counts.get("capture", 0.0)
        replay_count = counts.get("replay", 0.0)
        reasons: list[str] = []
        if total >= 2.0 and capture_count <= 0.0:
            reasons.append("missing capture after repeated execution")
        if capture_count > 0.0 and key not in executable_contexts:
            reasons.append("capture has no context-matched executable nodes")
        if total >= 3.0 and replay_count <= 0.0:
            reasons.append("missing replay after repeated execution")
        if reasons:
            label = f"{device or 'unknown'}:{context}"
            incomplete.append(f"{label} ({'; '.join(reasons)})")

    return tuple(incomplete)


def validate_graph_capture_policy(
    records: Sequence[Mapping[str, Any]],
    backend: str,
    extra_flags: str,
    *,
    require_prefill_lifecycle: bool = False,
) -> GraphCaptureValidation:
    """Validate full-graph/segmented execution against the cell topology.

    Returns a structured result instead of raising so the shell harness can
    report one concise failure without a Python traceback.
    """

    device_kinds = device_kinds_for_cell(backend, extra_flags)
    heterogeneous_device_mix = len(device_kinds) > 1
    has_collective_evidence = _has_collective_evidence(records)
    has_full_graph_plan = _has_full_graph_plan(records)
    has_nonempty_full_graph_executable = (
        _has_nonempty_full_graph_executable(records)
    )
    has_segmented_execution = _has_segmented_execution(records)
    missing_prefill_phases = (
        _missing_prefill_phases(records)
        if require_prefill_lifecycle
        else ()
    )
    prefill_lifecycle_complete = not missing_prefill_phases
    incomplete_contexts = _incomplete_graph_contexts(records)

    error: str | None = None
    if has_segmented_execution and not heterogeneous_device_mix:
        error = (
            "segmented GPU graph execution is forbidden for a homogeneous "
            f"device domain (device kinds: {sorted(device_kinds) or ['unknown']})"
        )
    elif has_segmented_execution and not has_collective_evidence:
        error = (
            "segmented GPU graph execution is admissible only for a "
            "heterogeneous graph with runtime-proven collective nodes"
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
            "warmup/capture/replay lifecycle; missing phases: "
            + ", ".join(missing_prefill_phases)
        )
    elif incomplete_contexts:
        error = (
            "GPU graph contexts did not complete warmup/capture/replay "
            "lifecycle: " + ", ".join(incomplete_contexts)
        )

    return GraphCaptureValidation(
        error=error,
        device_kinds=device_kinds,
        heterogeneous_device_mix=heterogeneous_device_mix,
        has_collective_evidence=has_collective_evidence,
        has_full_graph_plan=has_full_graph_plan,
        has_nonempty_full_graph_executable=(
            has_nonempty_full_graph_executable
        ),
        has_segmented_execution=has_segmented_execution,
        prefill_lifecycle_complete=prefill_lifecycle_complete,
        missing_prefill_phases=missing_prefill_phases,
        incomplete_contexts=incomplete_contexts,
    )
