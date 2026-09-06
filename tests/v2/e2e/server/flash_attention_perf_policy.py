#!/usr/bin/env python3
"""Validate production FlashAttention plan and execution evidence.

Qwen model graphs declare a backend-neutral, geometry-selected prefill policy.
CUDA and ROCm then choose one immutable physical transaction while the graph is
captured. This validator proves that a live server cell actually reached that
backend policy and that its published geometry is internally coherent.

CPU has no graph-capture boundary, so its corresponding evidence is emitted at
the common format-generic execution planner. A targeted long-context lane can
therefore prove that both complete-query ownership and K/V-context ownership
ran in production, rather than inferring mode selection from unit-level policy
tests or benchmark-only controls.

Expert-only overlay domains own no attention: the exported public CLI's base
model domain identifies the participants that must supply attention evidence.
Ordinary TP/PP retains its all-participant requirement. No model/backend name
or absence of evidence can grant the expert-only exception.

The GPU check consumes only capture-time ``gpu_graph_inventory`` records. The
CPU check consumes aggregate execution counters and never inspects tensor data
or introduces an additional synchronization boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import shlex
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


@dataclass(frozen=True)
class CPUFlashAttentionExecutionValidation:
    """Result of authenticating CPU FA2 physical execution records."""

    error: str | None
    plan_count: int
    execution_count: float
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


def attention_device_kinds_for_cell(backend: str, extra_flags: str) -> frozenset[str]:
    """Resolve attention ownership from the canonical exported CLI intent.

    The engine's ``effectiveBaseModelDomain()`` uses the explicit base domain,
    or the continuation domain when omitted. Only domain membership is needed
    here; allocation, placement, and graph planning remain engine-owned.
    General orchestration and routed-expert placement are separate CLI domain
    namespaces. Their matching declarations corroborate the ordered physical
    membership; duplicates within either namespace or conflicting membership
    fail closed instead of exempting a GPU. Scope/compute policy attributes
    remain engine-owned and do not change which backend owns attention.
    """
    options: dict[str, list[str]] = {}
    arguments = shlex.split(extra_flags)
    relevant = {"--moe-routed-expert-placement", "--moe-routed-expert-continuation-domain",
                "--moe-routed-expert-base-model-domain", "--moe-routed-expert-domain",
                "--define-domain"}
    for index, argument in enumerate(arguments):
        flag, separator, value = argument.partition("=")
        if flag not in relevant:
            continue
        if not separator:
            value = arguments[index + 1] if index + 1 < len(arguments) else ""
        if not value or value.startswith("--"):
            raise ValueError(f"attention ownership requires a value for {flag}")
        options.setdefault(flag, []).append(value)
    if "--moe-routed-expert-placement" not in options:
        return device_kinds_for_cell(backend, extra_flags)

    def single(flag: str) -> str:
        values = options.get(flag, [])
        if len(values) > 1:
            raise ValueError(f"ambiguous attention ownership: repeated {flag}")
        return values[0] if values else ""

    single("--moe-routed-expert-placement")
    continuation = single("--moe-routed-expert-continuation-domain")
    base = single("--moe-routed-expert-base-model-domain") or continuation
    if not base:
        raise ValueError("overlay attention ownership has no base/continuation domain")
    memberships: list[tuple[str, ...]] = []
    for namespace in ("--moe-routed-expert-domain", "--define-domain"):
        matches = [value.partition("=")[2].split(";", 1)[0]
                   for value in options.get(namespace, [])
                   if value.partition("=")[0] == base]
        if len(matches) > 1:
            raise ValueError(f"overlay attention domain {base!r} has repeated {namespace} declarations")
        if matches:
            members = tuple(member.strip() for member in matches[0].split(","))
            if not all(members):
                raise ValueError(f"overlay attention domain {base!r} has an empty participant")
            memberships.append(members)
    if not memberships:
        raise ValueError(f"overlay attention domain {base!r} has no declaration")
    # Participant identity and order, not just a matching GPU kind, must agree.
    # Ignoring one namespace could otherwise hide a missing attention backend.
    if any(members != memberships[0] for members in memberships[1:]):
        raise ValueError(f"overlay attention domain {base!r} has conflicting domain membership")
    kinds = device_kinds_for_cell("", ",".join(memberships[0]))
    if not kinds:
        raise ValueError(f"overlay attention domain {base!r} has no explicit devices")
    return kinds


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
    """Require coherent FA2 capture plans on every declared attention backend."""

    ownership_error = None
    try:
        attention_kinds = attention_device_kinds_for_cell(backend, extra_flags)
    except ValueError as error:
        attention_kinds = frozenset()
        ownership_error = str(error)
    expected_backends = attention_kinds.intersection(_PLAN_RECORD_BY_BACKEND)
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

    error: str | None = ownership_error
    if error:
        pass
    elif not attention_kinds:
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


_CPU_EXECUTION_RECORD = "cpu_fa2_parallel_plan_executions"
_CPU_SELECTED_MODES = frozenset({"query_sequence", "key_value_context"})
_CPU_POSITIVE_TAGS = (
    "query_rows",
    "local_query_heads",
    "head_dim",
    "physical_workers",
    "arithmetic_partitions",
    "context_partitions",
    "context_partition_rows",
    "physical_kv_tile",
)


def _validate_cpu_execution_record(record: Mapping[str, Any]) -> str | None:
    """Return a precise error for one malformed CPU FA2 execution record."""

    if record.get("domain") != "kernel":
        return f"{_CPU_EXECUTION_RECORD} is outside the kernel domain"
    if record.get("phase") != "execute":
        return f"{_CPU_EXECUTION_RECORD} was not published during execute"
    if str(record.get("device", "")).lower() != "cpu":
        return f"{_CPU_EXECUTION_RECORD} has a mismatched device identity"
    if _numeric(record.get("value", record.get("count", 0.0))) <= 0.0:
        return f"{_CPU_EXECUTION_RECORD} did not record an execution"

    tags = record.get("tags") or {}
    if tags.get("requested_axis") != "geometry_selected":
        return (
            f"{_CPU_EXECUTION_RECORD} did not enter the production "
            "geometry-selected policy "
            f"(requested_axis={tags.get('requested_axis')!r})"
        )

    selected_mode = str(tags.get("selected_mode", ""))
    if selected_mode not in _CPU_SELECTED_MODES:
        return (
            f"{_CPU_EXECUTION_RECORD} published invalid "
            f"selected_mode={selected_mode!r}"
        )

    values: dict[str, int] = {}
    for key in _CPU_POSITIVE_TAGS:
        value = _integer_tag(tags, key)
        if value is None or value <= 0:
            return f"{_CPU_EXECUTION_RECORD} requires positive integer tag {key}"
        values[key] = value

    context_partitions = values["context_partitions"]
    arithmetic_partitions = values["arithmetic_partitions"]
    if arithmetic_partitions < context_partitions:
        return (
            f"{_CPU_EXECUTION_RECORD} scheduled more physical context "
            "producers than canonical arithmetic summaries"
        )
    if selected_mode == "query_sequence" and context_partitions != 1:
        return (
            f"{_CPU_EXECUTION_RECORD} query_sequence execution retained "
            "multiple context producers"
        )
    if selected_mode == "key_value_context" and context_partitions <= 1:
        return (
            f"{_CPU_EXECUTION_RECORD} context execution did not expose "
            "multiple context producers"
        )

    partition_rows = values["context_partition_rows"]
    physical_tile = values["physical_kv_tile"]
    if physical_tile > partition_rows or partition_rows % physical_tile != 0:
        return (
            f"{_CPU_EXECUTION_RECORD} physical K/V tile does not preserve "
            "canonical summary boundaries"
        )

    return None


def validate_cpu_flash_attention_execution_policy(
    records: Iterable[Mapping[str, Any]],
    *,
    require_query_sequence: bool = True,
    require_key_value_context: bool = True,
) -> CPUFlashAttentionExecutionValidation:
    """Require coherent production CPU FA2 evidence for selected modes.

    The targeted long-context gate requires both branches: prefill should have
    enough independent query rows for query-sequence ownership, while decode or
    compact grouped verification should expose long K/V spans across physical
    workers. Callers may relax either presence requirement for narrower probes,
    but every observed record remains subject to the same fail-closed checks.
    """

    execution_records = [
        record
        for record in records
        if record.get("name") == _CPU_EXECUTION_RECORD
    ]
    selected_modes = frozenset(
        str((record.get("tags") or {}).get("selected_mode", ""))
        for record in execution_records
    )
    execution_count = sum(
        _numeric(record.get("value", record.get("count", 0.0)))
        for record in execution_records
    )

    error: str | None = None
    if not execution_records:
        error = "CPU cell emitted no FlashAttention physical execution evidence"
    else:
        for record in execution_records:
            error = _validate_cpu_execution_record(record)
            if error:
                break

    if error is None and require_query_sequence:
        if "query_sequence" not in selected_modes:
            error = "CPU cell never exercised query-sequence FlashAttention"
    if error is None and require_key_value_context:
        if "key_value_context" not in selected_modes:
            error = "CPU cell never exercised K/V-context FlashAttention"

    return CPUFlashAttentionExecutionValidation(
        error=error,
        plan_count=len(execution_records),
        execution_count=execution_count,
        selected_modes=selected_modes,
    )
