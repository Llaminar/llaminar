#!/usr/bin/env python3
"""Validate the typed LLEP verifier execution policy in PerfStats.

LLEP has two economical grouped-verifier policies. Participant-assigned rows
may select among already-resident experts using the device logical position.
A mirrored verifier instead executes every row locally and therefore performs
no participant assignment at all. The latter is not missing work: assignment,
current-batch transport, and routed-result collectives would all be redundant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional


@dataclass(frozen=True)
class LLEPVerifierPolicyValidation:
    """Result of validating one homogeneous LLEP verifier policy."""

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
    """Require exactly one complete, no-transport LLEP verifier policy.

    Mixing policies inside one homogeneous graph is rejected because it makes
    routed-result ownership ambiguous. Every matching record must carry the
    complete policy tags; a counter name alone is not sufficient evidence.
    """

    resident = []
    mirrored = []
    for record in records:
        if (
            record.get("domain") != "moe_rebalance"
            or record.get("phase") != "verifier"
            or not _positive(record)
        ):
            continue
        name = record.get("name")
        if name == "device_rebalance_llep_resident_assignment_calls":
            resident.append(record)
        elif name == "fully_replicated_local_verifier_execution_calls":
            mirrored.append(record)

    if resident and mirrored:
        return LLEPVerifierPolicyValidation(
            "GPU LLEP+MTP emitted both participant-assigned and fully "
            "replicated verifier policies"
        )
    if not resident and not mirrored:
        return LLEPVerifierPolicyValidation(
            "GPU LLEP+MTP emitted no typed verifier execution-policy evidence"
        )

    if mirrored:
        expected = {
            "execution_policy": "fully_replicated_local",
            "participant_assignment": "none",
            "current_batch_transport": "none",
            "routed_result_collective": "none",
        }
        policy = "fully_replicated_local"
        selected = mirrored
    else:
        expected = {
            "assignment": "logical_position_resident",
            "current_batch_transport": "none",
        }
        policy = "logical_position_resident"
        selected = resident

    for record in selected:
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
