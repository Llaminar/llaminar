#!/usr/bin/env python3
"""Validate fully device-owned GPU dynamic MTP generation evidence.

Homogeneous CUDA composes one maximum-capacity graph family and selects an exact
draft-depth prefix inside a native selector-gated WHILE. HIP lacks conditional
nodes; a heterogeneous CUDA controller also cannot compose its remote boundary
inside one native parent. Those policies publish one authenticated immutable
scheduler ticket and submit the named captured transaction. Mutable generation
state and outcome reduction remain device-owned. Evidence joins by rank and
device, so another participant cannot certify a missing boundary or ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from gpu_host_transfer_perf_policy import (
    device_generation_dispatch_ticket_abi_is_canonical,
)
from graph_capture_perf_policy import DecodeGraphRequirement, validate_graph_capture_policy


@dataclass(frozen=True)
class MTPDeviceGenerationValidation:
    """Result of validating one backend's dynamic generation graph family."""

    error: str | None
    devices: tuple[str, ...]


def _numeric(value: Any) -> float:
    """Convert one PerfStats field without allowing malformed text to pass."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _value(record: Mapping[str, Any]) -> float:
    """Return the exercised value of a counter or timer record."""

    return max(
        _numeric(record.get("value")),
        _numeric(record.get("count")),
        _numeric(record.get("total_ns")),
    )


def _integer(value: Any) -> int:
    """Convert one integral tag, returning an invalid sentinel on bad input."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _positive_transaction_sequence(record: Mapping[str, Any]) -> bool:
    """Prove every observed ticket carried a positive native transaction ID.

    Lifecycle IDs belong to a fixed-width ordered witness, never to a map key.
    Exact integer extrema retain the per-observation positivity check without
    retaining one counter per transaction. Missing or malformed evidence fails.
    """

    def unsigned(value: Any) -> bool:
        return type(value) is int and 0 <= value < 1 << 64

    count = record.get("count")
    words = record.get("sequence_word_count")
    minimum = record.get("sequence_minimum_words")
    maximum = record.get("sequence_maximum_words")
    return (
        record.get("kind") == "counter"
        and "transaction" not in (record.get("tags") or {})
        and unsigned(count) and count > 0
        and type(record.get("value")) in (int, float) and record["value"] == count
        and unsigned(words) and words == count
        and isinstance(minimum, list) and isinstance(maximum, list)
        and len(minimum) == len(maximum) == 1
        and unsigned(minimum[0]) and unsigned(maximum[0])
        # ABI-v2 publishes a signed 32-bit transaction count. A negative
        # native value converted to an unsigned observation must still fail.
        and 0 < minimum[0] <= maximum[0] <= (1 << 31) - 1
        and unsigned(record.get("sequence_digest_lo"))
        and unsigned(record.get("sequence_digest_hi"))
    )


def _records_by_device(
    records: Iterable[Mapping[str, Any]],
    name: str,
) -> dict[str, list[Mapping[str, Any]]]:
    """Group exercised records of one exact semantic operation by device."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        if record.get("name") != name or _value(record) <= 0.0:
            continue
        device = str(record.get("device", ""))
        if not device:
            continue
        grouped.setdefault(device, []).append(record)
    return grouped


def _sum_by_device(
    records: Iterable[Mapping[str, Any]],
    name: str,
) -> dict[str, float]:
    """Sum one exact counter independently for every participant device."""

    return {
        device: sum(_numeric(record.get("value")) for record in device_records)
        for device, device_records in _records_by_device(records, name).items()
    }


def _terminal_vocabulary_family_error(
    records: Iterable[Mapping[str, Any]],
    device: str,
    tags: Mapping[str, Any],
    verifier_rows: int,
) -> str | None:
    """Require declared head ownership and the native serial/verifier captures.

    Capture counters prove graph construction, not an execution multiplier.
    The generation launch and terminal ledger checks separately prove replay.
    """

    head = tags.get("terminal_head_policy")
    native = tags.get("native_local_vocabulary")
    if head not in {"mirrored-full-vocabulary", "vocabulary-sharded"} or native not in {"true", "false"}:
        return f"MTP terminal vocabulary ownership is missing or invalid on {device}"
    if native == "false":
        return None
    if head != "vocabulary-sharded":
        return f"MTP native vocabulary publication disagrees with head ownership on {device}"
    captures = _records_by_device(records, "native_vocabulary_allgather_graph_nodes")
    device_captures = captures.get(device, ())
    for rows in {1, verifier_rows}:
        if not any(
            _integer((record.get("tags") or {}).get("rows_capacity")) == rows
            and (rows == 1 or (record.get("tags") or {}).get("live_rows") == "device")
            for record in device_captures
        ):
            return f"MTP native vocabulary family lacks its {rows}-row capture on {device}"
    return None


def validate_cuda_dynamic_mtp_device_generation_policy(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_minimum_depth: int,
    expected_maximum_depth: int,
) -> MTPDeviceGenerationValidation:
    """Require one exact native dynamic generation family on every CUDA device.

    Args:
        records: PerfStats records from one canonical server cell.
        expected_minimum_depth: Inclusive CLI-configured dynamic selector floor.
        expected_maximum_depth: Inclusive CLI-configured dynamic selector cap.

    Returns:
        A fail-closed validation result naming every proven CUDA participant.
    """

    records = tuple(records)
    if (
        expected_minimum_depth <= 0
        or expected_maximum_depth <= expected_minimum_depth
    ):
        return MTPDeviceGenerationValidation(
            error="invalid expected CUDA dynamic MTP selector range",
            devices=(),
        )

    materializations = _records_by_device(
        records, "device_generation_loop_graph_materializations"
    )
    cuda_devices = tuple(
        sorted(device for device in materializations if device.startswith("CUDA:"))
    )
    if not cuda_devices:
        return MTPDeviceGenerationValidation(
            error=(
                "CUDA dynamic MTP emitted no native device-generation graph "
                "materialization"
            ),
            devices=(),
        )

    for device in cuda_devices:
        observed_sampling_modes: set[str] = set()
        for record in materializations[device]:
            tags = record.get("tags") or {}
            head_error = _terminal_vocabulary_family_error(
                records, device, tags, expected_maximum_depth + 1
            )
            if head_error:
                return MTPDeviceGenerationValidation(error=head_error, devices=cuda_devices)
            sampling_mode = str(tags.get("sampling_mode", ""))
            observed_sampling_modes.add(sampling_mode)
            if (
                tags.get("backend") != "CUDA"
                or tags.get("depth_policy") != "dynamic"
                or tags.get("execution")
                != "native_device_controlled_selector_while"
                or _integer(tags.get("minimum_draft_depth"))
                != expected_minimum_depth
                or _integer(tags.get("maximum_draft_depth"))
                != expected_maximum_depth
                or _integer(tags.get("draft_depth"))
                != expected_maximum_depth
                or _integer(tags.get("verifier_rows"))
                != expected_maximum_depth + 1
                or sampling_mode not in {"greedy", "stochastic"}
            ):
                return MTPDeviceGenerationValidation(
                    error=(
                        "CUDA dynamic MTP materialized an incomplete or "
                        f"mismatched graph family on {device}: {record}"
                    ),
                    devices=cuda_devices,
                )
        if "stochastic" not in observed_sampling_modes:
            return MTPDeviceGenerationValidation(
                error=(
                    "CUDA dynamic MTP never materialized the native stochastic "
                    f"sampling parent on {device}"
                ),
                devices=cuda_devices,
            )

    launches = _records_by_device(records, "device_generation_loop_graph_launches")
    for device in cuda_devices:
        if not any(
            (record.get("tags") or {}).get("execution")
            == "single_async_native_selector_while_launch"
            and _integer((record.get("tags") or {}).get("minimum_draft_depth"))
            == expected_minimum_depth
            and _integer((record.get("tags") or {}).get("maximum_draft_depth"))
            == expected_maximum_depth
            for record in launches.get(device, ())
        ):
            return MTPDeviceGenerationValidation(
                error=f"CUDA dynamic MTP never launched its native selector/WHILE on {device}",
                devices=cuda_devices,
            )

    # Capacity warmup is a request-level orchestration edge and therefore has no
    # participant device label. The per-device materialization records above prove
    # that every child consumed that same maximum-capacity family.
    capture_transactions = [
        record
        for record in records
        if record.get("name")
        == "dynamic_device_generation_capacity_capture_transactions"
        and _value(record) > 0.0
    ]
    if not any(
        _integer((record.get("tags") or {}).get("capture_depth"))
        == expected_maximum_depth
        and expected_minimum_depth
        <= _integer((record.get("tags") or {}).get("selected_depth"))
        <= expected_maximum_depth
        and (record.get("tags") or {}).get("authority")
        == "device_generation_controller"
        for record in capture_transactions
    ):
        return MTPDeviceGenerationValidation(
            error=(
                "CUDA dynamic MTP did not warm its maximum-capacity child family "
                "under device selector ownership"
            ),
            devices=cuda_devices,
        )

    transactions = _sum_by_device(records, "device_generation_terminal_transactions")
    compact_reductions = _sum_by_device(
        records, "device_generation_terminal_compact_outcome_reductions"
    )
    drafts = _sum_by_device(
        records, "device_generation_terminal_attempted_draft_tokens"
    )
    verifier_tokens = _sum_by_device(
        records, "device_generation_terminal_verifier_tokens"
    )
    updates = _sum_by_device(records, "device_generation_terminal_depth_updates")
    promotions = _sum_by_device(
        records, "device_generation_terminal_depth_promotions"
    )
    demotions = _sum_by_device(
        records, "device_generation_terminal_depth_demotions"
    )
    evaluated_windows = _sum_by_device(
        records, "device_generation_terminal_depth_evaluated_windows"
    )
    accepted = _sum_by_device(
        records, "device_generation_terminal_accepted_speculative_tokens"
    )
    rejected = _sum_by_device(
        records, "device_generation_terminal_rejected_transactions"
    )
    consumed_rows = _sum_by_device(
        records, "device_generation_terminal_consumed_verifier_rows"
    )
    terminal_bridges = _sum_by_device(
        records, "device_generation_terminal_response_bridges"
    )
    for device in cuda_devices:
        transaction_count = transactions.get(device, 0.0)
        attempted_drafts = drafts.get(device, 0.0)
        logical_verifier_tokens = verifier_tokens.get(device, 0.0)
        if (
            transaction_count <= 0.0
            or compact_reductions.get(device, 0.0) != transaction_count
            or attempted_drafts
            < transaction_count * expected_minimum_depth
            or attempted_drafts
            > transaction_count * expected_maximum_depth
            or logical_verifier_tokens
            != attempted_drafts + transaction_count
            or updates.get(device, 0.0)
            != promotions.get(device, 0.0) + demotions.get(device, 0.0)
            or evaluated_windows.get(device, 0.0) <= 0.0
            or updates.get(device, 0.0) > evaluated_windows.get(device, 0.0)
            or consumed_rows.get(device, 0.0) <= 0.0
            or accepted.get(device, 0.0) < 0.0
            or accepted.get(device, 0.0) > consumed_rows.get(device, 0.0)
            or rejected.get(device, 0.0) < 0.0
            or rejected.get(device, 0.0) > transaction_count
            or terminal_bridges.get(device, 0.0) <= 0.0
        ):
            return MTPDeviceGenerationValidation(
                error=(
                    "CUDA dynamic MTP terminal depth ledger is incomplete or "
                    f"inconsistent on {device}: transactions={transaction_count} "
                    f"drafts={attempted_drafts} verifier={logical_verifier_tokens} "
                    f"compact_reductions={compact_reductions.get(device, 0.0)} "
                    f"consumed_rows={consumed_rows.get(device, 0.0)} "
                    f"accepted={accepted.get(device, 0.0)} "
                    f"rejected={rejected.get(device, 0.0)} "
                    f"evaluated_windows={evaluated_windows.get(device, 0.0)} "
                    f"updates={updates.get(device, 0.0)} "
                    f"promotions={promotions.get(device, 0.0)} "
                    f"demotions={demotions.get(device, 0.0)}"
                ),
                devices=cuda_devices,
            )

        compact_records = _records_by_device(
            records,
            "device_generation_terminal_compact_outcome_reductions",
        ).get(device, ())
        if not compact_records or any(
            (record.get("tags") or {}).get("authority")
            != "device_generation_controller"
            or (record.get("tags") or {}).get("accounting_role")
            != "captured_graph_replay_multiplier"
            or (record.get("tags") or {}).get("source")
            not in {
                "captured_greedy_compact_outcome",
                "captured_stochastic_compact_outcome",
            }
            or (record.get("tags") or {}).get("execution")
            != "native_conditional_graph"
            for record in compact_records
        ) or not any(
            (record.get("tags") or {}).get("source")
            == "captured_stochastic_compact_outcome"
            for record in compact_records
        ):
            return MTPDeviceGenerationValidation(
                error=(
                    "CUDA dynamic MTP terminal compact-outcome provenance is "
                    f"missing or malformed on {device}"
                ),
                devices=cuda_devices,
            )

    mirrored_totals = (
        transactions,
        compact_reductions,
        drafts,
        verifier_tokens,
        evaluated_windows,
        accepted,
        rejected,
        consumed_rows,
    )
    for totals in mirrored_totals:
        values = {totals.get(device, -1.0) for device in cuda_devices}
        if len(values) != 1:
            return MTPDeviceGenerationValidation(
                error=(
                    "CUDA dynamic MTP terminal device ledgers disagree across "
                    f"mirrored participants: {totals}"
                ),
                devices=cuda_devices,
            )

    return MTPDeviceGenerationValidation(error=None, devices=cuda_devices)


def _cuda_hosted_boundary_error(
    records: tuple[Mapping[str, Any], ...],
    device: str,
    device_kinds: frozenset[str],
) -> str | None:
    """Require a real heterogeneous boundary and this controller's native replay.

    The caller has already isolated the rank. First authenticate its complete
    graph plan, then restrict native graph evidence to the controller owner.
    Coordinator records stay available because they own the external boundary;
    neighboring devices' executable and replay counters cannot fill a hole.
    """
    if len(device_kinds) < 2:
        return f"CUDA hosted MTP requires a heterogeneous execution boundary on {device}"
    graph = validate_graph_capture_policy(records, device_kinds)
    if graph.error or not (graph.has_pipeline_boundary_evidence or graph.has_overlay_boundary_evidence):
        return (f"CUDA hosted MTP has no authenticated heterogeneous boundary on {device}: "
                + (graph.error or "no completed pipeline/overlay boundary"))
    owner_records = tuple(row for row in records if row.get("domain") != "forward_graph"
                          or str(row.get("device", "")).lower() == device.lower()
                          or row.get("device") in {"pipeline_coordinator", "continuation_rank"})
    owner = validate_graph_capture_policy(
        owner_records, device_kinds, decode_requirement=DecodeGraphRequirement.REPLAY)
    if owner.error:
        return f"CUDA hosted MTP lacks its own captured replay on {device}: {owner.error}"
    return None


def validate_host_scheduled_mtp_device_generation_policy(
    records: Iterable[Mapping[str, Any]],
    *,
    device_kinds: frozenset[str],
    expected_minimum_depth: int,
    expected_maximum_depth: int,
) -> MTPDeviceGenerationValidation:
    """Require ticket-selected captured MTP on HIP or heterogeneous CUDA.

    Args:
        records: PerfStats from one server cell, optionally qualified by rank.
        device_kinds: Device kinds authenticated by the resolved service plan.
        expected_minimum_depth: Inclusive configured dynamic selector floor.
        expected_maximum_depth: Inclusive configured selector cap; a pinned
            range remains valid for fixed-geometry dynamic controller probes.

    Returns:
        A fail-closed result naming every proven controller participant. Rank
        ledgers are validated separately before combining participant names.
        The ABI-v2 52-byte ticket contains only an immutable branch decision.
    """
    if (not isinstance(device_kinds, frozenset) or not device_kinds
            or not device_kinds <= {"cpu", "cuda", "rocm"}):
        raise ValueError("hosted MTP certification requires resolved device kinds")
    records = tuple(records)
    # Capacity warmup belongs to the service orchestrator. Unlike device
    # ledgers it has no participant owner and need not be repeated on every rank.
    capture_transactions = [
        record
        for record in records
        if record.get("name")
        == "dynamic_device_generation_capacity_capture_transactions"
        and _value(record) > 0.0
    ]
    if not any(
        _integer((record.get("tags") or {}).get("capture_depth"))
        == expected_maximum_depth
        and expected_minimum_depth
        <= _integer((record.get("tags") or {}).get("selected_depth"))
        <= expected_maximum_depth
        and (record.get("tags") or {}).get("authority")
        == "device_generation_controller"
        for record in capture_transactions
    ):
        return MTPDeviceGenerationValidation(
            error=(
                "Hosted dynamic MTP did not warm its maximum-capacity captured "
                "transaction family under device selector ownership"
            ),
            devices=(),
        )

    ranks = {row.get("rank") for row in records
             if row.get("domain") == "mtp"
             and row.get("name") == "device_generation_loop_graph_materializations"
             and _value(row) > 0
             and (str(row.get("device", "")).startswith("ROCm:")
                  or (row.get("tags") or {}).get("execution")
                  == "hosted_captured_transactions_with_ticket_only_dispatch")}
    if not ranks:
        return MTPDeviceGenerationValidation("Hosted MTP emitted no controller materialization", ())
    devices: list[str] = []
    for rank in sorted(ranks, key=str):
        rows = tuple(row for row in records if row.get("rank") == rank)
        result = _validate_host_scheduled_rank(
            rows, device_kinds=device_kinds, expected_minimum_depth=expected_minimum_depth,
            expected_maximum_depth=expected_maximum_depth)
        if result.error:
            prefix = f"rank={rank}: " if rank is not None else ""
            return MTPDeviceGenerationValidation(prefix + result.error, tuple(devices))
        devices.extend(f"rank={rank}/{device}" if len(ranks) > 1 else device
                       for device in result.devices)
    return MTPDeviceGenerationValidation(None, tuple(devices))


def _validate_host_scheduled_rank(
    records: Iterable[Mapping[str, Any]],
    *,
    device_kinds: frozenset[str],
    expected_minimum_depth: int,
    expected_maximum_depth: int,
) -> MTPDeviceGenerationValidation:
    """Join one rank's controller, native graph, scheduler ticket and ledgers.

    CUDA ticket dispatch requires a captured heterogeneous execution boundary;
    HIP also supports this policy in homogeneous domains. All other arithmetic,
    depth, stochastic sampling and per-participant evidence is backend symmetric.
    """

    records = tuple(records)
    if (
        expected_minimum_depth <= 0
        or expected_maximum_depth < expected_minimum_depth
    ):
        return MTPDeviceGenerationValidation(
            error="invalid expected hosted dynamic MTP selector range",
            devices=(),
        )

    materializations = _records_by_device(
        records, "device_generation_loop_graph_materializations"
    )
    devices = tuple(
        sorted(device for device, rows in materializations.items()
               if device.startswith("ROCm:") or (
                   device.startswith("CUDA:") and any(
                       (row.get("tags") or {}).get("execution")
                       == "hosted_captured_transactions_with_ticket_only_dispatch"
                       for row in rows)))
    )
    if not devices:
        return MTPDeviceGenerationValidation(
            error=(
                "Hosted dynamic MTP emitted no ticket-selected captured graph "
                "materialization"
            ),
            devices=(),
        )

    policy_selections = _records_by_device(
        records, "device_generation_execution_policy_selections"
    )
    launches = _records_by_device(records, "device_generation_loop_graph_launches")
    ticket_submissions = _records_by_device(
        records, "device_generation_dispatch_ticket_d2h_submissions"
    )
    ticket_observations = _records_by_device(
        records, "device_generation_dispatch_tickets_observed"
    )
    transaction_submissions = _records_by_device(
        records, "hosted_device_generation_transaction_submissions"
    )

    for device in devices:
        kind = device.partition(":")[0].lower()
        if kind not in device_kinds:
            return MTPDeviceGenerationValidation(
                error=f"Hosted MTP controller {device} is outside the resolved device kinds",
                devices=devices)
        if kind == "cuda":
            boundary_error = _cuda_hosted_boundary_error(records, device, device_kinds)
            if boundary_error:
                return MTPDeviceGenerationValidation(error=boundary_error, devices=devices)
        expected_backend = "CUDA" if kind == "cuda" else "HIP"
        observed_sampling_modes: set[str] = set()
        for record in materializations[device]:
            tags = record.get("tags") or {}
            head_error = _terminal_vocabulary_family_error(
                records, device, tags, expected_maximum_depth + 1
            )
            if head_error:
                return MTPDeviceGenerationValidation(error=head_error, devices=devices)
            sampling_mode = str(tags.get("sampling_mode", ""))
            observed_sampling_modes.add(sampling_mode)
            if (
                tags.get("backend") != expected_backend
                or tags.get("depth_policy") != "dynamic"
                or tags.get("execution")
                != "hosted_captured_transactions_with_ticket_only_dispatch"
                or tags.get("conditional_fragments") != "0"
                or _integer(tags.get("fragments")) <= 0
                or _integer(tags.get("minimum_draft_depth"))
                != expected_minimum_depth
                or _integer(tags.get("maximum_draft_depth"))
                != expected_maximum_depth
                or _integer(tags.get("draft_depth"))
                != expected_maximum_depth
                or _integer(tags.get("verifier_rows"))
                != expected_maximum_depth + 1
                or _integer(tags.get("physical_verifier_rows"))
                != expected_maximum_depth + 1
                or sampling_mode not in {"greedy", "stochastic"}
            ):
                return MTPDeviceGenerationValidation(
                    error=(
                        "Hosted dynamic MTP materialized an incomplete or "
                        f"mismatched hosted graph family on {device}: {record}"
                    ),
                    devices=devices,
                )
        if "stochastic" not in observed_sampling_modes:
            return MTPDeviceGenerationValidation(
                error=(
                    "Hosted dynamic MTP never materialized its stochastic "
                    f"ticket-selected graph family on {device}"
                ),
                devices=devices,
            )

        if not any(
            (record.get("tags") or {}).get("policy")
            == "host_scheduled_captured_transactions"
            and (record.get("tags") or {}).get("selection_boundary")
            == "pre_first_draft"
            and (record.get("tags") or {}).get("topology") == "dynamic_depth"
            for record in policy_selections.get(device, ())
        ):
            return MTPDeviceGenerationValidation(
                error=(
                    "Hosted dynamic MTP did not select its hosted captured-graph "
                    f"policy before the first draft on {device}"
                ),
                devices=devices,
            )

        if not any(
            (record.get("tags") or {}).get("backend") == expected_backend
            and (record.get("tags") or {}).get("execution")
            == "hosted_ticket_selected_captured_transactions"
            and (record.get("tags") or {}).get("conditional_fragments") == "0"
            and _integer((record.get("tags") or {}).get("fragments")) > 0
            and _integer((record.get("tags") or {}).get("minimum_draft_depth"))
            == expected_minimum_depth
            and _integer((record.get("tags") or {}).get("maximum_draft_depth"))
            == expected_maximum_depth
            for record in launches.get(device, ())
        ):
            return MTPDeviceGenerationValidation(
                error=(
                    "Hosted dynamic MTP never launched a ticket-selected captured "
                    f"transaction on {device}"
                ),
                devices=devices,
            )

        device_ticket_submissions = ticket_submissions.get(device, ())
        if not device_ticket_submissions or any(
            not device_generation_dispatch_ticket_abi_is_canonical(
                record.get("tags") or {}
            )
            or (record.get("tags") or {}).get("authority")
            != "immutable_scheduler_snapshot"
            or (record.get("tags") or {}).get("state_payload") != "false"
            for record in device_ticket_submissions
        ):
            return MTPDeviceGenerationValidation(
                error=(
                    "Hosted dynamic MTP scheduler ticket boundary is missing or "
                    f"malformed on {device}"
                ),
                devices=devices,
            )

        if not ticket_observations.get(device) or any(
            not _positive_transaction_sequence(record)
            or not (
                expected_minimum_depth
                <= _integer((record.get("tags") or {}).get("next_depth"))
                <= expected_maximum_depth
            )
            or (record.get("tags") or {}).get("complete")
            not in {"true", "false"}
            or (record.get("tags") or {}).get("maintenance_due")
            not in {"true", "false"}
            for record in ticket_observations[device]
        ):
            return MTPDeviceGenerationValidation(
                error=(
                    "Hosted dynamic MTP observed an unauthenticated scheduler "
                    f"decision on {device}"
                ),
                devices=devices,
            )

        if not transaction_submissions.get(device) or any(
            not (
                expected_minimum_depth
                <= _integer((record.get("tags") or {}).get("depth"))
                <= expected_maximum_depth
            )
            or (record.get("tags") or {}).get("dynamic_depth_source")
            != "device_controller_ticket"
            or _integer((record.get("tags") or {}).get("fragments")) <= 0
            for record in transaction_submissions[device]
        ):
            return MTPDeviceGenerationValidation(
                error=(
                    "Hosted dynamic MTP transaction submission did not consume "
                    f"the authenticated device-controller ticket on {device}"
                ),
                devices=devices,
            )

    transactions = _sum_by_device(records, "device_generation_terminal_transactions")
    compact_reductions = _sum_by_device(
        records, "device_generation_terminal_compact_outcome_reductions"
    )
    drafts = _sum_by_device(
        records, "device_generation_terminal_attempted_draft_tokens"
    )
    verifier_tokens = _sum_by_device(
        records, "device_generation_terminal_verifier_tokens"
    )
    updates = _sum_by_device(records, "device_generation_terminal_depth_updates")
    promotions = _sum_by_device(
        records, "device_generation_terminal_depth_promotions"
    )
    demotions = _sum_by_device(
        records, "device_generation_terminal_depth_demotions"
    )
    evaluated_windows = _sum_by_device(
        records, "device_generation_terminal_depth_evaluated_windows"
    )
    accepted = _sum_by_device(
        records, "device_generation_terminal_accepted_speculative_tokens"
    )
    rejected = _sum_by_device(
        records, "device_generation_terminal_rejected_transactions"
    )
    consumed_rows = _sum_by_device(
        records, "device_generation_terminal_consumed_verifier_rows"
    )
    terminal_bridges = _sum_by_device(
        records, "device_generation_terminal_response_bridges"
    )
    ticket_submission_totals = _sum_by_device(
        records, "device_generation_dispatch_ticket_d2h_submissions"
    )
    ticket_observation_totals = _sum_by_device(
        records, "device_generation_dispatch_tickets_observed"
    )
    hosted_transaction_totals = _sum_by_device(
        records, "hosted_device_generation_transaction_submissions"
    )
    hosted_terminal_totals = _sum_by_device(
        records, "hosted_device_generation_terminal_submissions"
    )

    for device in devices:
        transaction_count = transactions.get(device, 0.0)
        attempted_drafts = drafts.get(device, 0.0)
        logical_verifier_tokens = verifier_tokens.get(device, 0.0)
        if (
            transaction_count <= 0.0
            or compact_reductions.get(device, 0.0) != transaction_count
            or ticket_submission_totals.get(device, 0.0) != transaction_count
            or ticket_observation_totals.get(device, 0.0) != transaction_count
            or hosted_transaction_totals.get(device, 0.0)
            + hosted_terminal_totals.get(device, 0.0)
            != transaction_count
            or hosted_terminal_totals.get(device, 0.0) <= 0.0
            or attempted_drafts
            < transaction_count * expected_minimum_depth
            or attempted_drafts
            > transaction_count * expected_maximum_depth
            or logical_verifier_tokens != attempted_drafts + transaction_count
            or updates.get(device, 0.0)
            != promotions.get(device, 0.0) + demotions.get(device, 0.0)
            or evaluated_windows.get(device, 0.0) <= 0.0
            or updates.get(device, 0.0) > evaluated_windows.get(device, 0.0)
            or consumed_rows.get(device, 0.0) <= 0.0
            or accepted.get(device, 0.0) < 0.0
            or accepted.get(device, 0.0) > consumed_rows.get(device, 0.0)
            or rejected.get(device, 0.0) < 0.0
            or rejected.get(device, 0.0) > transaction_count
            or terminal_bridges.get(device, 0.0) <= 0.0
        ):
            return MTPDeviceGenerationValidation(
                error=(
                    "Hosted dynamic MTP ledger is incomplete or "
                    f"inconsistent on {device}: transactions={transaction_count} "
                    f"tickets={ticket_submission_totals.get(device, 0.0)}/"
                    f"{ticket_observation_totals.get(device, 0.0)} "
                    f"submissions={hosted_transaction_totals.get(device, 0.0)}+"
                    f"{hosted_terminal_totals.get(device, 0.0)} "
                    f"drafts={attempted_drafts} verifier={logical_verifier_tokens} "
                    f"compact_reductions={compact_reductions.get(device, 0.0)} "
                    f"consumed_rows={consumed_rows.get(device, 0.0)} "
                    f"accepted={accepted.get(device, 0.0)} "
                    f"rejected={rejected.get(device, 0.0)} "
                    f"evaluated_windows={evaluated_windows.get(device, 0.0)} "
                    f"updates={updates.get(device, 0.0)}"
                ),
                devices=devices,
            )

        compact_records = _records_by_device(
            records,
            "device_generation_terminal_compact_outcome_reductions",
        ).get(device, ())
        if not compact_records or any(
            (record.get("tags") or {}).get("authority")
            != "device_generation_controller"
            or (record.get("tags") or {}).get("accounting_role")
            != "captured_graph_replay_multiplier"
            or (record.get("tags") or {}).get("source")
            not in {
                "captured_greedy_compact_outcome",
                "captured_stochastic_compact_outcome",
            }
            or (record.get("tags") or {}).get("execution")
            != "host_scheduled_captured_transactions"
            for record in compact_records
        ) or not any(
            (record.get("tags") or {}).get("source")
            == "captured_stochastic_compact_outcome"
            for record in compact_records
        ):
            return MTPDeviceGenerationValidation(
                error=(
                    "Hosted dynamic MTP terminal compact-outcome provenance is "
                    f"missing or malformed on {device}"
                ),
                devices=devices,
            )

    mirrored_totals = (
        transactions,
        compact_reductions,
        drafts,
        verifier_tokens,
        ticket_submission_totals,
        ticket_observation_totals,
        hosted_transaction_totals,
        hosted_terminal_totals,
        evaluated_windows,
        accepted,
        rejected,
        consumed_rows,
    )
    for totals in mirrored_totals:
        values = {totals.get(device, -1.0) for device in devices}
        if len(values) != 1:
            return MTPDeviceGenerationValidation(
                error=(
                    "Hosted dynamic MTP device ledgers disagree across mirrored "
                    f"participants: {totals}"
                ),
                devices=devices,
            )

    return MTPDeviceGenerationValidation(error=None, devices=devices)
