#!/usr/bin/env python3
"""Validate the reusable GPU request-input bank lifetime from PerfStats.

GPU request tokens, positions, and sequence lengths are uploaded into one
persistent arena-owned input bank. Reusing that storage is legal only after
the complete main/MTP reader chain publishes a completion event. The next
request then waits on that event before overwriting the bank.

This validator keeps that three-edge protocol explicit in the server E2E gate:

1. ``device_input_event_waits`` proves graph readers waited for admission.
2. ``device_input_reuse_publications`` proves every consumed admission was
   released by the transitive final reader.
3. ``device_input_reuse_waits`` proves a later request ordered its overwrite
   after the preceding release.

The validator intentionally consumes only PerfStats records. It therefore
proves the production server path exercised the protocol without adding host
synchronization or test-only hooks to inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_DOMAIN = "request_admission"
_OWNED_ROWS = "device_owned_input_rows"
_ADMISSION_WAITS = "device_input_event_waits"
_REUSE_PUBLICATIONS = "device_input_reuse_publications"
_REUSE_WAITS = "device_input_reuse_waits"


@dataclass(frozen=True)
class RequestInputDeviceEvidence:
    """Aggregated reusable-bank evidence for one physical GPU."""

    device: str
    owned_rows: float
    admission_waits: float
    reuse_publications: float
    reuse_waits: float


@dataclass(frozen=True)
class RequestInputLifetimeValidation:
    """Result of validating all GPU request-input producers in one E2E cell."""

    error: str | None
    devices: tuple[RequestInputDeviceEvidence, ...]


def _numeric(value: Any) -> float:
    """Convert one PerfStats value into a comparison-safe number."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def validate_request_input_lifetime_policy(
    records: Sequence[Mapping[str, Any]],
    *,
    require_reuse: bool = True,
) -> RequestInputLifetimeValidation:
    """Require a complete admission/read/release/reuse event chain per GPU.

    ``require_reuse`` should be true for the server matrix because every cell
    sends multiple requests. A focused one-request probe may set it false while
    still proving that its sole admission was consumed and released.
    """

    values: dict[str, dict[str, float]] = {}
    interesting = {
        _OWNED_ROWS,
        _ADMISSION_WAITS,
        _REUSE_PUBLICATIONS,
        _REUSE_WAITS,
    }

    for record in records:
        if record.get("domain") != _DOMAIN:
            continue
        name = str(record.get("name", ""))
        if name not in interesting:
            continue
        device = str(record.get("device", "")) or "unknown"
        bucket = values.setdefault(
            device,
            {counter_name: 0.0 for counter_name in interesting},
        )
        bucket[name] += _numeric(
            record.get("value", record.get("count", 0.0))
        )

    active_devices = tuple(
        RequestInputDeviceEvidence(
            device=device,
            owned_rows=counters[_OWNED_ROWS],
            admission_waits=counters[_ADMISSION_WAITS],
            reuse_publications=counters[_REUSE_PUBLICATIONS],
            reuse_waits=counters[_REUSE_WAITS],
        )
        for device, counters in sorted(values.items())
        if counters[_OWNED_ROWS] > 0.0
    )
    if not active_devices:
        return RequestInputLifetimeValidation(
            error=(
                "GPU cell emitted no device-owned request-input admission "
                "evidence"
            ),
            devices=(),
        )

    errors: list[str] = []
    for evidence in active_devices:
        prefix = f"{evidence.device}:"
        if evidence.admission_waits <= 0.0:
            errors.append(f"{prefix} no admission-to-reader event waits")
        if evidence.reuse_publications <= 0.0:
            errors.append(f"{prefix} no final-reader reuse publications")
        if require_reuse and evidence.reuse_waits <= 0.0:
            errors.append(f"{prefix} no release-to-next-writer event waits")

        # Every consumed admission must have exactly one final-reader release.
        # A mismatch means either a request was read without releasing the bank
        # or a release was published without a corresponding admitted request.
        if evidence.admission_waits != evidence.reuse_publications:
            errors.append(
                f"{prefix} consumed admissions "
                f"({evidence.admission_waits:g}) != releases "
                f"({evidence.reuse_publications:g})"
            )

        # The first admission has no predecessor. Every later reuse wait must
        # therefore be backed by a publication, but publications may exceed
        # waits by one for each still-live orchestrator at artifact export.
        if evidence.reuse_waits > evidence.reuse_publications:
            errors.append(
                f"{prefix} overwrite waits ({evidence.reuse_waits:g}) exceed "
                f"published releases ({evidence.reuse_publications:g})"
            )

    return RequestInputLifetimeValidation(
        error=(
            "GPU request-input lifetime protocol incomplete: "
            + "; ".join(errors)
            if errors
            else None
        ),
        devices=active_devices,
    )
