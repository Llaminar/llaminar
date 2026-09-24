#!/usr/bin/env python3
"""Validate the canonical current-batch-LLEP verifier policy in PerfStats.

LLEP applies only to ordinary prefill. The canonical grouped verifier remains
ordinary expert parallel: whole experts stay on static owners, grouped rows run
through the runtime-table kernel, and no current-batch expert transfer occurs.
The MTP terminal head may be mirrored independently; that does not replicate
routed experts or alter verifier row assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional


@dataclass(frozen=True)
class LLEPVerifierPolicyValidation:
    """Result of validating the canonical grouped-verifier policy."""

    error: Optional[str]
    policy: Optional[str] = None


def _positive(record: Mapping[str, object]) -> bool:
    """Return whether a counter record proves at least one execution."""

    value = record.get("value", record.get("count", 0))
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def validate_llep_verifier_policy(
    records: Iterable[Mapping[str, object]],
) -> LLEPVerifierPolicyValidation:
    """Require grouped static-owner verification and reject prefill policy bleed."""

    static_owner = []
    forbidden = []
    for record in records:
        if record.get("phase") != "verifier" or not _positive(record):
            continue
        name = record.get("name")
        if (
            record.get("domain") == "moe_routed_execution"
            and name == "static_owner_grouped_verifier_calls"
        ):
            static_owner.append(record)
        elif name in {
            "device_rebalance_llep_resident_assignment_calls",
            "fully_replicated_local_verifier_execution_calls",
        }:
            forbidden.append(record)

    if forbidden:
        return LLEPVerifierPolicyValidation(
            "GPU current-batch LLEP leaked prefill assignment or replicated-"
            "expert policy into grouped verification"
        )
    if not static_owner:
        return LLEPVerifierPolicyValidation(
            "GPU current-batch LLEP emitted no grouped static-owner verifier evidence"
        )

    expected = {
        "execution_policy": "static_owner_grouped",
        "assignment": "static_owner",
        "runtime_grouping": "runtime_table",
        "row_execution_policy": "participant_assigned",
        "current_batch_transport": "none",
    }
    policy = "static_owner_grouped"
    for record in static_owner:
        tags = record.get("tags") or {}
        if not isinstance(tags, Mapping):
            return LLEPVerifierPolicyValidation(
                f"GPU LLEP+MTP {policy} verifier record has malformed tags"
            )
        mismatches = {
            key: (tags.get(key), expected_value)
            for key, expected_value in expected.items()
            if tags.get(key) != expected_value
        }
        if mismatches:
            return LLEPVerifierPolicyValidation(
                f"GPU LLEP+MTP {policy} verifier policy is incomplete: "
                f"{mismatches}"
            )

    return LLEPVerifierPolicyValidation(None, policy)
