#!/usr/bin/env python3
"""Authenticate each selected participant's declared attention implementations.

The startup plan declares FA and QSA ownership independently, including dense
predictors beside a sparse trunk. Native QSA capture observations bind the exact
immutable arithmetic/geometry signature to stage submission; constructor setup
alone cannot prove capture. CPU observations are emitted after execution. The
ordinary graph validator remains responsible for complete executable lifecycle.
No tensor reads, backend initialization or synchronization occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping

from flash_attention_perf_policy import validate_flash_attention_plan_policy
from server_execution_contract import ServerExecutionContract


@dataclass(frozen=True)
class AttentionExecutionValidation:
    """Bounded observer result; family ownership is never inferred from absence."""

    error: str | None
    flash_participants: int
    sparse_participants: int
    sparse_programs: int


_QSA_IDENTITY = ("policy", "program", "parts", "phase", "rows", "query_heads",
                 "kv_heads", "query_head_start", "kv_head_start", "capture_signature")
_QSA_CAPTURE = "qsa_attention_program_captures"
_QSA_EXECUTION = "attention_program_executions"
_FA_NAMES = {"cuda_fa2_parallel_plan_selections", "rocm_fa2_parallel_plan_selections"}


def _owner(record: Mapping[str, Any]) -> tuple[int, str]:
    """Require exact rank-local identity, never a backend-wide match."""
    rank, device = record.get("rank"), record.get("device")
    if type(rank) is not int or rank < 0 or not isinstance(device, str):
        raise ValueError("attention evidence requires exact rank/device identity")
    return rank, device


def _positive(record: Mapping[str, Any]) -> bool:
    """Reject nonfinite, textual and boolean counter values."""
    value = record.get("value")
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _integer(tags: Mapping[str, Any], key: str, minimum: int) -> int:
    """Require canonical nonnegative decimal tags and checked geometry."""
    value = tags.get(key)
    if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]*", value) or int(value) < minimum:
        raise ValueError(f"QSA requires integer tag {key} >= {minimum}")
    return int(value)


def _qsa_identity(record: Mapping[str, Any], *, domain: str, phase: str) -> tuple:
    """Authenticate the sealed native arithmetic program and immutable binding."""
    if record.get("domain") != domain or record.get("phase") != phase or not _positive(record):
        raise ValueError("QSA evidence has invalid domain, phase or counter value")
    tags = record.get("tags")
    if not isinstance(tags, dict):
        raise ValueError("QSA evidence requires structured tags")
    if tags.get("phase") not in {"prefill", "decode"}:
        raise ValueError("QSA requires an explicit execution phase")
    policy = tags.get("policy")
    parts = {"fp32": 8, "fp16_tile": 8, "fp16_tile_33": 33}.get(policy)
    if parts is None or tags.get("program") != policy or _integer(tags, "parts", 1) != parts:
        raise ValueError("QSA arithmetic policy, program and partial count disagree")
    _, device = _owner(record)
    if policy != "fp32" and not device.startswith("ROCm:"):
        raise ValueError("QSA FP16 arithmetic requires its ROCm participant")
    for key in ("rows", "query_heads", "kv_heads", "capture_signature"):
        _integer(tags, key, 1)
    for key in ("query_head_start", "kv_head_start"):
        _integer(tags, key, 0)
    return tuple(tags[key] for key in _QSA_IDENTITY)


def validate_attention_execution_policy(
    records: Iterable[Mapping[str, Any]], contract: ServerExecutionContract,
    *, require_prefill: bool = True,
) -> AttentionExecutionValidation:
    """Require every declared family on each rank/device without family fallback.

    FA keeps its existing geometry policy and is checked separately per owner.
    QSA's execution/capture witness must match constructor metadata exactly.
    Every replayed prefill bucket requires its own captured QSA row capacity;
    unused setup variants remain distinct and cannot certify runtime buckets.
    """
    rows = tuple(records)
    flash, sparse = contract.flash_attention_participants, contract.sparse_attention_participants
    programs = 0
    error = None
    try:
        for record in rows:
            if record.get("name") in _FA_NAMES:
                if _owner(record) not in flash:
                    raise ValueError("FlashAttention plan belongs to an undeclared participant")
                if not isinstance(record.get("tags"), dict) or not _positive(record):
                    raise ValueError("FlashAttention plan has malformed tags or counter value")
        for owner in sorted(flash):
            backend = owner[1].partition(":")[0].lower()
            if backend == "cpu":
                continue  # The existing explicit CPU context-parallel gate owns FA execution modes.
            selected = [r for r in rows if r.get("name") in _FA_NAMES and _owner(r) == owner]
            result = validate_flash_attention_plan_policy(selected, frozenset({backend}))
            if result.error:
                raise ValueError(f"rank={owner[0]}/{owner[1]}: {result.error}")
        setup: dict[tuple[int, str], set[tuple]] = {}
        observed: dict[tuple[int, str], set[tuple]] = {}
        for record in rows:
            name = record.get("name")
            if name not in {"attention_arithmetic_policy", _QSA_CAPTURE, _QSA_EXECUTION}:
                continue
            owner = _owner(record)
            if owner not in sparse:
                raise ValueError("QSA evidence belongs to an undeclared participant")
            if name == "attention_arithmetic_policy":
                identity = _qsa_identity(record, domain="qsa", phase="setup")
                setup.setdefault(owner, set()).add(identity)
            else:
                cpu = owner[1] == "CPU"
                if (cpu and name != _QSA_EXECUTION) or (not cpu and name != _QSA_CAPTURE):
                    raise ValueError("QSA execution evidence used the wrong backend boundary")
                identity = _qsa_identity(record, domain="qsa" if cpu else "gpu_graph_inventory",
                                         phase="execute" if cpu else "capture_setup")
                observed.setdefault(owner, set()).add(identity)
                programs += 1
        for owner in sorted(sparse):
            actual = observed.get(owner, set())
            if not actual:
                raise ValueError(f"rank={owner[0]}/{owner[1]} emitted no QSA capture/execution evidence")
            if not actual <= setup.get(owner, set()):
                raise ValueError("QSA capture/execution has missing or stale arithmetic setup")
            phases = {identity[3] for identity in actual}
            expected = {"prefill", "decode"} if require_prefill else {"decode"}
            if not expected <= phases:
                raise ValueError(f"rank={owner[0]}/{owner[1]} omitted QSA phase evidence: {sorted(expected-phases)}")
            if owner[1] == "CPU":
                continue
            captured_decode = [r for r in rows if r.get("domain") == "forward_graph"
                and r.get("name") in {"full_graph_capture_stage_types", "segmented_graph_capture_stage_types"}
                and r.get("rank") == owner[0] and r.get("device") == owner[1]
                and (r.get("tags") or {}).get("stage_type") == "QWEN4EXP_QSA_ATTENTION"]
            if not captured_decode:
                raise ValueError("QSA lacks its native captured stage inventory")
            for record in captured_decode:
                tags = record.get("tags") or {}
                if (record.get("phase") != "decode" or not _positive(record)
                        or tags.get("attribution") != "graph_replay_metadata"
                        or tags.get("source") != ("full_graph_capture" if record["name"].startswith("full_") else "segmented_graph_capture")
                        or tags.get("graph_capture_scope") != ("full_graph_capture_plan" if record["name"].startswith("full_") else "segmented_capture_plan")
                        or not tags.get("context")
                        or tags.get("type") not in {"captured_executable", "captured_child_template"}):
                    raise ValueError("QSA captured stage inventory has invalid provenance")
            captured_rows = {identity[4] for identity in actual if identity[3] == "prefill"}
            for record in rows:
                if (record.get("domain") == "forward_graph" and record.get("name") == "prefill_graph_phase"
                        and record.get("rank") == owner[0] and record.get("device") == owner[1]
                        and (record.get("tags") or {}).get("capture_phase") == "replay"):
                    if (record.get("phase") != "prefill" or not _positive(record)
                            or record["tags"].get("bucket_seq_len") not in captured_rows):
                        raise ValueError("QSA replayed prefill bucket has no matching captured program")
    except (ValueError, TypeError) as problem:
        error = str(problem)
    return AttentionExecutionValidation(error, len(flash), len(sparse), programs)
