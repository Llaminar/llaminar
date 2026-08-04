#!/usr/bin/env python3
"""Validate fully device-owned CUDA dynamic MTP generation evidence.

The dynamic stochastic production lane must compose one maximum-capacity CUDA
graph family, select one exact draft-depth branch inside a native SWITCH/WHILE,
and expose only a terminal controller ledger.  This validator correlates the
per-device PerfStats records and fails closed if the runtime returns to a host
transaction loop or reconstructs variable-depth work from one scalar depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class MTPDeviceGenerationValidation:
    """Result of validating one CUDA dynamic generation graph family."""

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
            sampling_mode = str(tags.get("sampling_mode", ""))
            observed_sampling_modes.add(sampling_mode)
            if (
                tags.get("backend") != "CUDA"
                or tags.get("depth_policy") != "dynamic"
                or tags.get("execution")
                != "native_device_controlled_switch_while"
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
            == "single_async_native_switch_while_launch"
            and _integer((record.get("tags") or {}).get("minimum_draft_depth"))
            == expected_minimum_depth
            and _integer((record.get("tags") or {}).get("maximum_draft_depth"))
            == expected_maximum_depth
            for record in launches.get(device, ())
        ):
            return MTPDeviceGenerationValidation(
                error=f"CUDA dynamic MTP never launched its native SWITCH/WHILE on {device}",
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
            != "captured_stochastic_compact_outcome"
            or (record.get("tags") or {}).get("execution")
            != "native_device_generation_parent"
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
