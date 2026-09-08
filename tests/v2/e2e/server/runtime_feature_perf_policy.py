#!/usr/bin/env python3
"""Certify configured runtime features from completed production operations.

HTTP answers alone cannot prove that a prefix was restored, a speculative draft
was accepted, or an expert changed placement. This observer validates those
facts after server shutdown; it never owns or mutates inference state. Capture,
transport legality and memory ownership remain the other canonical validators'
responsibilities. Both host-owned and device-owned placement implementations
publish completion evidence, with backend-independent counters for MTP/cache.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
import math
import shlex
from typing import Any, Iterable, Mapping


class MovementEvidence(str, Enum):
    """Cross-language projection of ModelParityCase::movementEvidence()."""
    NOT_APPLICABLE = "not_applicable"
    FORBIDDEN = "forbidden"
    REQUIRED = "required"


def _positive(value: Any) -> float:
    """Return a finite positive observation, never a malformed success value."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def validate_runtime_feature_policy(
    records: Iterable[Mapping[str, Any]], extra_flags: str,
    movement_evidence: MovementEvidence,
) -> str | None:
    """Require actual execution of the public CLI's selected feature contract.

    Completed migration edges are emitted by the placement authority only after
    commit. The all-GPU overlay authority exports completed transactions, edges,
    and bytes after publication and reader retirement; homogeneous device
    placement also exports applied arrivals plus useful physical payload bytes.
    Proposals, cache lookup hits, and an MTP domain's mere presence cannot
    substitute for these completed operations.
    """
    if not isinstance(movement_evidence, MovementEvidence):
        return "runtime certification requires typed movement evidence"
    arguments = shlex.split(extra_flags)
    options: dict[str, str] = {}
    for index, argument in enumerate(arguments):
        flag, separator, value = argument.partition("=")
        if not flag.startswith("--"):
            continue
        if not separator and index + 1 < len(arguments):
            value = arguments[index + 1]
        options[flag] = value

    totals: dict[tuple[str, str], float] = defaultdict(float)
    mtp_restores = 0.0
    for record in records:
        domain, name = record.get("domain"), record.get("name")
        if domain not in {"mtp", "prefix_cache", "moe_rebalance",
                          "moe_overlay_controller", "moe_overlay_residency"}:
            continue
        value = _positive(record.get("value", record.get("count", 0)))
        totals[(domain, name)] += value
        if (domain == "prefix_cache" and name == "populate_restores"
                and (record.get("tags") or {}).get("includes_mtp_state") == "true"):
            mtp_restores += value

    if "--prefix-cache" in options:
        if totals[("prefix_cache", "harvest_inserts")] <= 0:
            return "prefix-cache certification observed no harvested prefix"
        if totals[("prefix_cache", "populate_restores")] <= 0:
            return "prefix-cache certification observed no actual restore (lookup hits are insufficient)"
        if "--mtp" in options and mtp_restores <= 0:
            return "prefix-cache certification restored no MTP-bearing state"

    if "--mtp" in options:
        attempts = (totals[("mtp", "draft_steps")]
                    + totals[("mtp", "device_generation_terminal_attempted_draft_tokens")])
        if attempts <= 0 or totals[("mtp", "accepted_tokens")] <= 0:
            return "MTP certification requires attempted and accepted draft tokens"
        if (options.get("--mtp-depth-policy") == "dynamic"
                and totals[("mtp", "depth_policy_windows")] <= 0):
            return "dynamic MTP certification observed no depth-controller window"

    committed_edges = totals[("moe_overlay_residency", "expert_migration_edges")]
    applied = (totals[("moe_rebalance", "device_rebalance_wave_applied_arrivals_total")]
               + totals[("moe_rebalance", "device_rebalance_transfer_current_applied_arrivals")])
    physical_bytes = totals[("moe_rebalance", "device_rebalance_transfer_useful_payload_bytes")]
    overlay_edges = totals[("moe_overlay_controller", "dynamic_migration_edges")]
    overlay_transactions = totals[("moe_overlay_controller", "dynamic_movement_transactions")]
    overlay_bytes = totals[("moe_overlay_controller", "dynamic_physical_bytes")]
    # These counters follow transactionComplete() and physical source retirement.
    # Do not substitute the command count or prepared-arrival diagnostics.
    completed_overlay = overlay_edges > 0 and overlay_transactions > 0 and overlay_bytes > 0
    completed_movement = (committed_edges > 0 or (applied > 0 and physical_bytes > 0)
                          or completed_overlay)
    if movement_evidence is MovementEvidence.REQUIRED and not completed_movement:
        return "dynamic movement certification observed no committed physical expert movement"
    if movement_evidence is MovementEvidence.FORBIDDEN and (
            committed_edges > 0 or applied > 0 or physical_bytes > 0
            or overlay_edges > 0 or overlay_transactions > 0 or overlay_bytes > 0):
        return "static movement certification observed expert movement"
    return None
