#!/usr/bin/env python3
"""Validate production FlashAttention capture-plan evidence.

Qwen model graphs declare a backend-neutral, geometry-selected prefill policy.
CUDA and ROCm then choose one immutable physical transaction while the graph is
captured. This validator proves that a live server cell actually reached that
backend policy and that its published geometry is internally coherent.

The check intentionally consumes only capture-time ``gpu_graph_inventory``
records. It does not inspect live sequence lengths, synchronize a stream, or
introduce host-owned inference state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from graph_capture_perf_policy import device_kinds_for_cell


_PLAN_RECORD_BY_BACKEND = {
    "cuda": "cuda_fa2_parallel_plan_selections",
    "rocm": "rocm_fa2_parallel_plan_selections",
}

_SELECTED_MODES_BY_BACKEND = {
    "cuda": frozenset(
        {"query_sequence", "key_value_context", "device_adaptive"}
    ),
    "rocm": frozenset({"query_sequence", "key_value_context"}),
}

_COMMON_POSITIVE_TAGS = (
    "batch_size",
    "query_rows",
    "local_query_heads",
    "head_dim",
    "kv_capacity",
    "query_grid_blocks",
)


@dataclass(frozen=True)
class FlashAttentionPlanValidation:
    """Result of authenticating one server cell's FA2 capture inventory."""

    error: str | None
    expected_backends: frozenset[str]
    observed_backends: frozenset[str]
    plan_count: int
    selected_modes: frozenset[str]


def _numeric(value: Any) -> float:
    """Convert one PerfStats scalar without accepting malformed text."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _integer_tag(tags: Mapping[str, Any], key: str) -> int | None:
    """Return an integer tag, or ``None`` when its spelling is not exact."""

    try:
        value = int(str(tags.get(key, "")))
    except ValueError:
        return None
    return value


def _validate_plan_record(
    record: Mapping[str, Any],
    backend_kind: str,
) -> str | None:
    """Return a precise error for one malformed backend plan record."""

    expected_name = _PLAN_RECORD_BY_BACKEND[backend_kind]
    if record.get("name") != expected_name:
        return f"unexpected {backend_kind} FA2 record name"
    if record.get("domain") != "gpu_graph_inventory":
        return f"{expected_name} is outside gpu_graph_inventory"
    if record.get("phase") != "capture_setup":
        return f"{expected_name} was not published during capture_setup"
    if not str(record.get("device", "")).lower().startswith(
        backend_kind + ":"
    ):
        return f"{expected_name} has a mismatched device identity"
    if _numeric(record.get("value", record.get("count", 0.0))) <= 0.0:
        return f"{expected_name} did not record an exercised capture plan"

    tags = record.get("tags") or {}
    if tags.get("requested_axis") != "geometry_selected":
        return (
            f"{expected_name} did not enter the production geometry-selected "
            f"policy (requested_axis={tags.get('requested_axis')!r})"
        )

    selected_mode = str(tags.get("selected_axis", ""))
    if selected_mode not in _SELECTED_MODES_BY_BACKEND[backend_kind]:
        return f"{expected_name} published invalid selected_axis={selected_mode!r}"

    for key in _COMMON_POSITIVE_TAGS:
        value = _integer_tag(tags, key)
        if value is None or value <= 0:
            return f"{expected_name} requires positive integer tag {key}"
    if str(tags.get("kv_storage", "")).lower() in {"", "unknown"}:
        return f"{expected_name} did not identify its native KV storage"

    context_keys = (
        "context_partitions",
        "context_partition_slots",
        "device_direct_partition_limit",
    )
    context_values: dict[str, int] = {}
    for key in context_keys:
        value = _integer_tag(tags, key)
        if value is None or value < 0:
            return f"{expected_name} requires non-negative integer tag {key}"
        context_values[key] = value

    if selected_mode == "query_sequence":
        if any(context_values.values()):
            return (
                f"{expected_name} query_sequence plan retained context "
                "workspace geometry"
            )
    else:
        partitions = context_values["context_partitions"]
        slots = context_values["context_partition_slots"]
        direct_limit = context_values["device_direct_partition_limit"]
        if partitions <= 0 or slots <= 0 or slots > partitions:
            return f"{expected_name} published an invalid context partition envelope"
        if direct_limit > partitions:
            return f"{expected_name} direct partition prefix exceeds its envelope"

    if backend_kind == "rocm":
        tile_q = _integer_tag(tags, "tile_q")
        tile_kv = _integer_tag(tags, "tile_kv")
        if tile_q is None or tile_q <= 0 or tile_kv is None or tile_kv <= 0:
            return f"{expected_name} requires positive HIP tile geometry"

        rocm_context_keys = (
            "context_phase_block_slots",
            "reducer_dimension_wavefronts",
            "reducer_block_slots",
        )
        rocm_context_values: list[int] = []
        for key in rocm_context_keys:
            value = _integer_tag(tags, key)
            if value is None or value < 0:
                return f"{expected_name} requires non-negative integer tag {key}"
            rocm_context_values.append(value)
        if selected_mode == "query_sequence" and any(rocm_context_values):
            return f"{expected_name} query plan retained HIP reducer geometry"
        if selected_mode == "key_value_context" and (
            context_values["device_direct_partition_limit"] <= 0
            or any(value <= 0 for value in rocm_context_values)
        ):
            return f"{expected_name} context plan is missing its three-node envelope"

    if backend_kind == "cuda":
        query_warp_groups = _integer_tag(tags, "query_warp_groups")
        reducer_warps = _integer_tag(tags, "reducer_dimension_warps")
        if query_warp_groups is None or query_warp_groups <= 0:
            return f"{expected_name} requires positive CUDA query warp groups"
        if reducer_warps is None or reducer_warps < 0:
            return f"{expected_name} requires a non-negative CUDA reducer geometry"
        if selected_mode == "query_sequence" and reducer_warps != 0:
            return f"{expected_name} query plan retained CUDA reducer geometry"
        if selected_mode != "query_sequence" and reducer_warps <= 0:
            return f"{expected_name} context plan omitted CUDA reducer geometry"

    return None


def validate_flash_attention_plan_policy(
    records: Iterable[Mapping[str, Any]],
    backend: str,
    extra_flags: str,
) -> FlashAttentionPlanValidation:
    """Require one coherent FA2 capture plan for every explicit GPU backend."""

    expected_backends = frozenset(
        kind
        for kind in device_kinds_for_cell(backend, extra_flags)
        if kind in _PLAN_RECORD_BY_BACKEND
    )
    plan_records: list[tuple[str, Mapping[str, Any]]] = []
    for record in records:
        name = record.get("name")
        for backend_kind, expected_name in _PLAN_RECORD_BY_BACKEND.items():
            if name == expected_name:
                plan_records.append((backend_kind, record))
                break

    observed_backends = frozenset(kind for kind, _ in plan_records)
    selected_modes = frozenset(
        str((record.get("tags") or {}).get("selected_axis", ""))
        for _, record in plan_records
    )

    error: str | None = None
    if not expected_backends:
        error = "GPU cell has no explicit CUDA/ROCm device kind for FA2 validation"
    else:
        missing = expected_backends - observed_backends
        unexpected = observed_backends - expected_backends
        if missing:
            error = (
                "GPU cell emitted no FlashAttention capture plan for: "
                + ", ".join(sorted(missing))
            )
        elif unexpected:
            error = (
                "GPU cell emitted FlashAttention plans for undeclared backends: "
                + ", ".join(sorted(unexpected))
            )
        else:
            for backend_kind, record in plan_records:
                error = _validate_plan_record(record, backend_kind)
                if error:
                    break

    return FlashAttentionPlanValidation(
        error=error,
        expected_backends=expected_backends,
        observed_backends=observed_backends,
        plan_count=len(plan_records),
        selected_modes=selected_modes,
    )
