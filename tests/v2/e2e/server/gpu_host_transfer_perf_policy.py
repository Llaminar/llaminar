#!/usr/bin/env python3
"""Validate the server E2E GPU-to-host transfer contract.

GPU inference owns intermediate execution state on the device.  The host may
observe only a compact serial-visible token result or the terminal response
ledger produced after a complete device-owned generation request. Prefix-cache
movement is the sole data-plane exception because RAM and disk are intentional
cache tiers rather than execution-state mirrors. ROCm additionally permits one
strictly authenticated scheduler ticket per hosted transaction: 48 bytes for
dynamic MTP and 60 bytes for homogeneous device-side MoE rebalancing. HIP
graphs do not provide conditional nodes, so the host submits the
already-captured branch named by this immutable control-plane snapshot without
receiving mutable model state, verifier data, KV data, or draft tokens.

PerfStats includes both semantic operation records and legacy aggregate
``transfer/d2h`` byte counters.  The aggregates do not identify the caller, so
they cannot prove or disprove the hot-path contract.  This validator therefore
fails closed over every *semantic* D2H/host-materialization record while
ignoring only those two exact aggregate names.  Adding a new host read to the
production path consequently breaks the canonical GPU E2E matrix until its
boundary is explicitly reviewed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


_UNATTRIBUTED_TRANSFER_AGGREGATES = frozenset({"d2h", "d2h_bytes"})

# These operations materialize the compact, authoritative response of one
# decode/MTP transaction.  They never expose draft proposals, verifier rows,
# recurrent state, or KV data to host execution.
_FINAL_RESPONSE_OPERATIONS = frozenset(
    {
        "first_token_greedy_device_target_slot_d2h_sync",
        "grouped_outcome_greedy_device_outcome_host_bridge",
        "grouped_outcome_stochastic_device_outcome_host_bridge",
        "device_generation_terminal_d2h_enqueue",
        "device_generation_terminal_d2h_wait",
        "rank_mirrored_localtp_greedy_outcome_host_materializations",
        "rank_mirrored_localtp_stochastic_outcome_host_materializations",
        "sample_stochastic_distribution_d2h_sync",
        "stochastic_batch_summary_d2h_sync",
        "stochastic_request_batch_summary_d2h_enqueue",
        "stochastic_request_batch_summary_d2h_sync",
        "stochastic_request_batch_summary_d2h_wait",
    }
)

_ROCM_HOST_DISPATCH_OPERATIONS = {
    "device_generation_dispatch_ticket_d2h_submissions": ("mtp", "48"),
    "device_moe_rebalance_dispatch_ticket_d2h_submissions": (
        "moe_rebalance",
        "60",
    ),
}


@dataclass(frozen=True)
class GPUHostTransferValidation:
    """Result of checking semantic host-transfer records for one GPU cell."""

    error: str | None
    final_response_operations: tuple[str, ...]
    scheduler_dispatch_operations: tuple[str, ...]
    prefix_cache_operations: tuple[str, ...]
    forbidden_operations: tuple[str, ...]


def _numeric(value: Any) -> float:
    """Convert a PerfStats field to a comparison-safe number."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _record_exercised(record: Mapping[str, Any]) -> bool:
    """Return whether an aggregated counter/timer observed real work.

    Timers carry ``value == 0`` and expose activity through ``count`` and
    ``total_ns``.  Counters normally use ``value``.  Inspecting all three keeps
    the policy correct for either PerfStats record kind.
    """

    return any(
        _numeric(record.get(field)) > 0.0
        for field in ("value", "count", "total_ns")
    )


def _is_semantic_host_transfer(name: str) -> bool:
    """Recognize operation names that publish GPU data to host code."""

    lowered = name.lower()
    if lowered == "host_logits_access":
        return True
    return any(
        marker in lowered
        for marker in ("d2h", "host_bridge", "host_materialization")
    )


def _is_explicit_prefix_cache_tier_transfer(
    record: Mapping[str, Any],
) -> bool:
    """Accept only transfers explicitly owned by the RAM/disk cache tiers."""

    name = str(record.get("name", "")).lower()
    domain = str(record.get("domain", "")).lower()
    tags = {
        str(key).lower(): str(value).lower()
        for key, value in (record.get("tags") or {}).items()
    }
    cache_owned = "prefix_cache" in name or domain == "prefix_cache"
    tier = tags.get("tier", tags.get("destination_tier", ""))
    direction = tags.get("direction", "")
    return cache_owned and (
        tier in {"ram", "disk", "host", "ssd"}
        or direction in {"device_to_ram", "device_to_disk", "gpu_to_ram"}
    )


def _is_authenticated_rocm_scheduler_ticket(
    record: Mapping[str, Any],
) -> bool:
    """Accept only the reviewed HIP conditional-graph control snapshot.

    The exact byte count deliberately belongs to the gate. A future ABI change
    must be reviewed here instead of silently expanding this narrow scheduler
    boundary into a model-state readback. CUDA has native conditional graph
    nodes and therefore may never emit this operation.
    """

    operation = _ROCM_HOST_DISPATCH_OPERATIONS.get(str(record.get("name", "")))
    if operation is None:
        return False
    expected_domain, expected_bytes = operation
    if str(record.get("domain", "")).lower() != expected_domain:
        return False
    if not str(record.get("device", "")).lower().startswith("rocm:"):
        return False

    tags = {
        str(key): str(value)
        for key, value in (record.get("tags") or {}).items()
    }
    return (
        tags.get("bytes") == expected_bytes
        and tags.get("authority") == "immutable_scheduler_snapshot"
        and tags.get("state_payload") == "false"
    )


def validate_gpu_host_transfer_policy(
    records: Iterable[Mapping[str, Any]],
) -> GPUHostTransferValidation:
    """Reject every exercised intermediate GPU-to-host operation.

    The allowlist is based on semantic operation identity, never substring
    guesses such as ``summary`` or ``result``.  This is deliberate: an unknown
    D2H operation represents a new architecture boundary and must receive an
    explicit review before canonical GPU serving tests can pass.
    """

    final_response: set[str] = set()
    scheduler_dispatch: set[str] = set()
    prefix_cache: set[str] = set()
    forbidden: set[str] = set()

    for record in records:
        name = str(record.get("name", ""))
        if not name or not _record_exercised(record):
            continue
        if name in _UNATTRIBUTED_TRANSFER_AGGREGATES:
            continue
        if not _is_semantic_host_transfer(name):
            continue
        if name in _FINAL_RESPONSE_OPERATIONS:
            final_response.add(name)
            continue
        if _is_authenticated_rocm_scheduler_ticket(record):
            scheduler_dispatch.add(name)
            continue
        if _is_explicit_prefix_cache_tier_transfer(record):
            prefix_cache.add(name)
            continue
        forbidden.add(name)

    forbidden_operations = tuple(sorted(forbidden))
    error = None
    if forbidden_operations:
        error = (
            "GPU inference performed intermediate device-to-host transfers; "
            "only compact final-response materialization and explicit "
            "RAM/disk prefix-cache tier movement are permitted, plus ROCm's "
            "exact authenticated immutable graph-dispatch ticket: "
            + ", ".join(forbidden_operations)
        )

    return GPUHostTransferValidation(
        error=error,
        final_response_operations=tuple(sorted(final_response)),
        scheduler_dispatch_operations=tuple(sorted(scheduler_dispatch)),
        prefix_cache_operations=tuple(sorted(prefix_cache)),
        forbidden_operations=forbidden_operations,
    )
