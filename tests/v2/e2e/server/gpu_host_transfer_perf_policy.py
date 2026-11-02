#!/usr/bin/env python3
"""Validate the server E2E GPU-to-host transfer contract.

GPU inference owns intermediate execution state on the device.  The host may
observe only a compact serial-visible token result or the terminal response
ledger produced after a complete device-owned generation request. Prefix-cache
movement and explicitly declared heterogeneous activation collectives are
data-plane boundaries, not execution-state mirrors. The shared activation
channel additionally requires same-rank node-local mapped-region evidence;
a CPU expert participant necessarily receives the activations it computes.
Likewise a CPU pipeline tail owns host logits: observing that CPU-owned value
is not a GPU download merely because an earlier pipeline stage runs on a GPU.
ROCm additionally permits one
strictly authenticated scheduler ticket per hosted transaction: 52 bytes for
dynamic MTP and 60 bytes for homogeneous device-side MoE rebalancing. HIP
graphs do not provide conditional nodes, so the host submits the
already-captured branch named by this immutable control-plane snapshot without
receiving mutable model state, verifier data, KV data, or draft tokens.
CUDA permits the same generation ticket only at an explicitly heterogeneous
boundary with same-rank/device retained-transaction evidence. Homogeneous CUDA
still requires its native conditional parent; its rebalance ticket is forbidden.

PerfStats includes both semantic operation records and legacy aggregate
``transfer/d2h`` byte counters.  The aggregates do not identify the caller, so
they cannot prove or disprove the hot-path contract.  This validator therefore
fails closed over every *semantic* D2H/host-materialization record while
ignoring only those two exact aggregate names.  Adding a new host read to the
production path consequently breaks the canonical GPU E2E matrix until its
boundary is explicitly reviewed here.
"""

from __future__ import annotations

from graph_capture_perf_policy import DecodeGraphRequirement, validate_graph_capture_policy

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


_UNATTRIBUTED_TRANSFER_AGGREGATES = frozenset({"d2h", "d2h_bytes"})

# This is the single Python-side review point for the C++ wire ABI declared by
# DeviceGenerationDispatchTicket. A unit architecture test compares these
# values to SamplingMath.h so the two languages cannot drift silently again.
DEVICE_GENERATION_DISPATCH_TICKET_ABI_VERSION = 2
DEVICE_GENERATION_DISPATCH_TICKET_WORD_COUNT = 13
DEVICE_GENERATION_DISPATCH_TICKET_BYTES = (
    DEVICE_GENERATION_DISPATCH_TICKET_WORD_COUNT * 4
)

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

_CAPTURED_HOST_DISPATCH_OPERATIONS = {
    "device_generation_dispatch_ticket_d2h_submissions": (
        "mtp",
        str(DEVICE_GENERATION_DISPATCH_TICKET_BYTES),
    ),
    "device_moe_rebalance_dispatch_ticket_d2h_submissions": (
        "moe_rebalance",
        "60",
    ),
}


def device_generation_dispatch_ticket_abi_is_canonical(
    tags: Mapping[str, Any],
) -> bool:
    """Authenticate the exact reviewed device-generation ticket wire ABI."""

    normalized = {str(key): str(value) for key, value in tags.items()}
    return (
        normalized.get("bytes")
        == str(DEVICE_GENERATION_DISPATCH_TICKET_BYTES)
        and normalized.get("abi_version")
        == str(DEVICE_GENERATION_DISPATCH_TICKET_ABI_VERSION)
        and normalized.get("word_count")
        == str(DEVICE_GENERATION_DISPATCH_TICKET_WORD_COUNT)
    )


@dataclass(frozen=True)
class GPUHostTransferValidation:
    """Result of checking semantic host-transfer records for one GPU cell."""

    error: str | None
    final_response_operations: tuple[str, ...]
    scheduler_dispatch_operations: tuple[str, ...]
    prefix_cache_operations: tuple[str, ...]
    activation_collective_operations: tuple[str, ...]
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


def _is_authenticated_captured_scheduler_ticket(
    record: Mapping[str, Any],
    *,
    heterogeneous_ticket_owners: frozenset[tuple[object, str]] = frozenset(),
) -> bool:
    """Accept only the reviewed captured-transaction control snapshot.

    The exact byte count deliberately belongs to the gate. A future ABI change
    must be reviewed here instead of silently expanding this narrow scheduler
    boundary into a model-state readback. CUDA's sole exception is a proven
    heterogeneous captured transaction, never a homogeneous parent or a
    host-authoritative rebalance controller.
    """

    operation = _CAPTURED_HOST_DISPATCH_OPERATIONS.get(str(record.get("name", "")))
    if operation is None:
        return False
    expected_domain, expected_bytes = operation
    if str(record.get("domain", "")).lower() != expected_domain:
        return False
    device = str(record.get("device", "")).lower()
    heterogeneous_generation = (
        record.get("name") == "device_generation_dispatch_ticket_d2h_submissions"
        and (record.get("rank"), device) in heterogeneous_ticket_owners
    )
    if not device.startswith("rocm:") and not heterogeneous_generation:
        return False

    tags = {
        str(key): str(value)
        for key, value in (record.get("tags") or {}).items()
    }
    abi_valid = tags.get("bytes") == expected_bytes
    if str(record.get("name", "")) == (
        "device_generation_dispatch_ticket_d2h_submissions"
    ):
        abi_valid = device_generation_dispatch_ticket_abi_is_canonical(tags)

    return (
        abi_valid
        and tags.get("authority") == "immutable_scheduler_snapshot"
        and tags.get("state_payload") == "false"
    )


def _live_heterogeneous_ticket_owners(records, device_kinds):
    """Authenticate current-window hosted launches without requiring cold setup telemetry.

    The exact same rank/device must own completed heterogeneous native replay,
    typed hosted geometry and balanced immutable-ticket/branch/terminal records.
    Materializing an unused sibling, or merely naming a mixed CLI, proves none
    of those. This reads existing observations only; it changes no runtime path.
    """
    owners = set()
    if len(device_kinds) < 2 or "cuda" not in device_kinds:
        return owners
    integer = lambda value: str(value).isdigit()
    for launch in records:
        device = str(launch.get("device", "")).lower()
        if (launch.get("domain") != "mtp" or launch.get("name") != "device_generation_loop_graph_launches"
                or not device.startswith("cuda:") or not _record_exercised(launch)):
            continue
        tags = launch.get("tags") or {}
        fields = ("requests", "draft_depth", "minimum_draft_depth", "maximum_draft_depth", "fragments")
        if (tags.get("execution") != "hosted_ticket_selected_captured_transactions"
                or tags.get("backend") != "CUDA" or not all(integer(tags.get(k)) for k in fields)
                or int(tags["requests"]) != 1 or int(tags["fragments"]) <= 0
                or not 0 <= int(tags["minimum_draft_depth"]) <= int(tags["draft_depth"]) <= int(tags["maximum_draft_depth"]) <= 15
                or tags.get("conditional_fragments") != "0" or tags.get("selector_gated_fragments") != "0"
                or type(launch.get("rank")) is not int or launch["rank"] < 0):
            continue
        owner = (launch.get("rank"), device)
        local = tuple(row for row in records if row.get("rank") == owner[0])
        graph_records = tuple(row for row in local if row.get("domain") != "forward_graph"
            or str(row.get("device", "")).lower() == device
            or row.get("device") in {"pipeline_coordinator", "continuation_rank"})
        graph = validate_graph_capture_policy(graph_records, device_kinds,
            decode_requirement=DecodeGraphRequirement.REPLAY)
        if graph.error or not (graph.has_overlay_boundary_evidence or graph.has_pipeline_boundary_evidence):
            continue
        controller = tuple(row for row in local if row.get("domain") == "mtp"
            and str(row.get("device", "")).lower() == device and _record_exercised(row))
        named = lambda name: tuple(row for row in controller if row.get("name") == name)
        total = lambda name: sum(_numeric(row.get("value")) for row in named(name))
        tickets = named("device_generation_dispatch_ticket_d2h_submissions")
        observed = named("device_generation_dispatch_tickets_observed")
        branches = named("hosted_device_generation_transaction_submissions")
        terminal = total("hosted_device_generation_terminal_submissions")
        fragments = named("hosted_device_generation_fragment_submissions")
        if (not tickets or not observed or not branches or terminal <= 0
                or terminal != _numeric(launch.get("value"))
                or any(not device_generation_dispatch_ticket_abi_is_canonical(row.get("tags") or {})
                    or (row.get("tags") or {}).get("authority") != "immutable_scheduler_snapshot"
                    or (row.get("tags") or {}).get("state_payload") != "false" for row in tickets)
                or any((row.get("tags") or {}).get("entry_source") != "committed_transaction_ticket"
                    or (row.get("tags") or {}).get("complete") not in {"true", "false"} for row in observed)
                or any((row.get("tags") or {}).get("entry_source") != "committed_transaction_ticket"
                    or (row.get("tags") or {}).get("dynamic_depth_source") != "device_controller_ticket"
                    or not integer((row.get("tags") or {}).get("fragments"))
                    or not 0 < int((row.get("tags") or {})["fragments"]) <= int(tags["fragments"]) for row in branches)
                or any(not integer((row.get("tags") or {}).get("fragment_index"))
                    or not 0 <= int((row.get("tags") or {})["fragment_index"]) < int(tags["fragments"]) for row in fragments)
                or sum(_numeric(row.get("value")) for row in observed if (row.get("tags") or {}).get("complete") == "true") != terminal
                or sum(_numeric(row.get("value")) for row in observed if (row.get("tags") or {}).get("complete") == "false") != total("hosted_device_generation_transaction_submissions")
                or sum(_numeric(row.get("value")) * int((row.get("tags") or {})["fragments"]) for row in branches) != total("hosted_device_generation_fragment_submissions")
                or total("device_generation_dispatch_ticket_d2h_submissions") != total("device_generation_dispatch_tickets_observed")
                or total("device_generation_dispatch_ticket_d2h_submissions") != total("hosted_device_generation_transaction_submissions") + terminal):
            continue
        owners.add(owner)
    return owners


def validate_gpu_host_transfer_policy(
    records: Iterable[Mapping[str, Any]],
    *,
    device_kinds: frozenset[str] = frozenset(),
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
    activation_collective: set[str] = set()
    forbidden: set[str] = set()
    records = tuple(records)
    # A heterogeneous CLI alone is insufficient: the exact emitting owner must
    # have materialized the retained ticket-selected transaction family. Another
    # rank/device's graph cannot authorize a CUDA host readback.
    heterogeneous_ticket_owners = frozenset(
        (record.get("rank"), str(record.get("device", "")).lower())
        for record in records
        if len(device_kinds) > 1 and "cuda" in device_kinds
        and record.get("domain") == "mtp"
        and record.get("name") == "device_generation_loop_graph_materializations"
        and str(record.get("device", "")).lower().startswith("cuda:")
        and _record_exercised(record)
        and (record.get("tags") or {}).get("execution") ==
            "hosted_captured_transactions_with_ticket_only_dispatch"
        and _numeric((record.get("tags") or {}).get("fragments")) > 0
    )
    heterogeneous_ticket_owners |= frozenset(_live_heterogeneous_ticket_owners(records, device_kinds))
    # Registration is runtime proof that the shared wire actually belongs to
    # this process and node. A different rank's mapping cannot authorize DMA.
    mapped_ranks = {
        record.get("rank") for record in records
        if record.get("domain") == "moe_overlay_activation_epoch"
        and record.get("name") in {"mapped_regions_registered", "mapped_regions_allocated"}
        and _record_exercised(record)
        and (record.get("tags") or {}).get("scope") == "node_local"
        and (record.get("tags") or {}).get("blocking") == "false"
        and (record.get("tags") or {}).get("mapping") in {
            "typed_external_host_pages", "backend_owned_mapped_host_pages"}
    }

    for record in records:
        name = str(record.get("name", ""))
        if not name or not _record_exercised(record):
            continue
        if name in _UNATTRIBUTED_TRANSFER_AGGREGATES:
            continue
        if not _is_semantic_host_transfer(name):
            continue
        device = str(record.get("device", "")).lower()
        if (name == "host_logits_access" and record.get("domain") == "sampling"
                and "cpu" in device_kinds
                and (device == "cpu" or (device.startswith("cpu:") and device[4:].isdigit()))
                and (record.get("tags") or {}).get("source") in {
                    "rank_orchestrator_logits", "device_graph_logits"}):
            # The emitting runtime owner, not the cell's broad backend label,
            # determines whether this is a D2H boundary. Missing/GPU ownership
            # remains forbidden, even in a topology containing a CPU tier.
            continue
        if name in _FINAL_RESPONSE_OPERATIONS:
            final_response.add(name)
            continue
        if _is_authenticated_captured_scheduler_ticket(
            record, heterogeneous_ticket_owners=heterogeneous_ticket_owners
        ):
            scheduler_dispatch.add(name)
            continue
        if _is_explicit_prefix_cache_tier_transfer(record):
            prefix_cache.add(name)
            continue
        tags = record.get("tags") or {}
        source_kind = str(record.get("device", "")).split(":", 1)[0].lower()
        if (len(device_kinds) > 1 and source_kind in device_kinds
                and source_kind in {"cuda", "rocm"}
                and record.get("rank") in mapped_ranks
                and record.get("domain") == "moe_overlay_activation_epoch"
                and name == "shared_physical_dispatch_d2h_bytes"
                and tags.get("host_blocking") == "false"
                and tags.get("payload_layout") == "shared_physical_rows"
                and tags.get("payload_path") == "shared_physical_mapped"
                and tags.get("role") == "shared dispatch lane batch"):
            activation_collective.add(name)
            continue
        forbidden.add(name)

    forbidden_operations = tuple(sorted(forbidden))
    error = None
    if forbidden_operations:
        error = (
            "GPU inference performed intermediate device-to-host transfers; "
            "only compact final-response materialization and explicit "
            "RAM/disk prefix-cache tier movement, certified node-local "
            "heterogeneous activation collectives, and exact authenticated "
            "immutable graph-dispatch tickets at a certified HIP or "
            "heterogeneous captured boundary are permitted: "
            + ", ".join(forbidden_operations)
        )

    return GPUHostTransferValidation(
        error=error,
        final_response_operations=tuple(sorted(final_response)),
        scheduler_dispatch_operations=tuple(sorted(scheduler_dispatch)),
        prefix_cache_operations=tuple(sorted(prefix_cache)),
        activation_collective_operations=tuple(sorted(activation_collective)),
        forbidden_operations=forbidden_operations,
    )
