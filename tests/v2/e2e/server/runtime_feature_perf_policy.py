#!/usr/bin/env python3
"""Certify configured runtime features from completed production operations.

HTTP answers alone cannot prove that a prefix was restored, a speculative draft
was accepted, or an expert changed placement. This observer validates those
facts after server shutdown; it never owns or mutates inference state. Capture,
transport legality and memory ownership remain the other canonical validators'
responsibilities. Both host-owned and device-owned placement implementations
publish completion evidence, with backend-independent counters for MTP/cache.
The server harness discards its model at process shutdown and exports no
prepared-context reuse contract. Undoing expert placement there is forbidden
cleanup work, not useful runtime movement or an acceptable teardown cost.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
import math
import json
import shlex
from typing import Any, Iterable, Mapping


class MovementEvidence(str, Enum):
    """Cross-language projection of ModelParityCase::movementEvidence()."""
    NOT_APPLICABLE = "not_applicable"
    FORBIDDEN = "forbidden"
    REQUIRED = "required"


def _positive(value: Any) -> float:
    """Return a finite positive observation, never a malformed success value."""
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def _publication_sequence(record: Mapping[str, Any] | None) -> tuple[int, ...] | None:
    """Read a bounded two-word-per-wave identity witness, never a counter total.

    The transport and placement owner independently fold candidate epoch and
    migration cardinality. Equality certifies the same ordered publications;
    no per-epoch telemetry key or reconstructed placement state is needed.
    """
    if record is None or record.get("kind") != "ordered_sequence":
        return None
    fields = ("count", "sequence_word_count", "sequence_digest_lo", "sequence_digest_hi")
    values = tuple(record.get(name) for name in fields)
    if (any(type(value) is not int or not 0 <= value < 2**64 for value in values)
            or values[0] == 0 or values[1] != 2 * values[0]):
        return None
    return values


def _overlay_transaction(record: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Keep the existing device completion trio within one exact publication."""
    tags = record.get("tags") or {}
    if tags.get("policy_owner") != "device":
        return None
    identity = []
    for name in ("transaction", "candidate_epoch"):
        value = tags.get(name)
        if not isinstance(value, str) or not value.isascii() or not value.isdecimal() or not 0 < int(value) < 2**64:
            return None
        identity.append(int(value))
    return (record.get("rank"), record.get("device"), record.get("phase"), *identity)


def validate_runtime_feature_policy(
    records: Iterable[Mapping[str, Any]], extra_flags: str,
    movement_evidence: MovementEvidence,
) -> str | None:
    """Require actual execution of the public CLI's selected feature contract.

    Completed migration edges are emitted by the placement authority only after
    commit, but must be corroborated by measured, published payload bytes and
    matching bounded transport/owner publication sequences. Calibration cannot
    enter those placement-only witnesses. The all-GPU overlay authority exports completed transactions, edges,
    and bytes after publication and reader retirement; homogeneous device
    placement exports qualified retained-wave copy/apply/byte lower bounds.
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
    native_observations: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    host_publications: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    host_edge_scopes: set[tuple[Any, ...]] = set()
    overlay_observations: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    invalid_publication = False
    placement_payload = 0.0
    host_names = {"placement_transport_publications", "placement_owner_publications",
                  "placement_published_payload_bytes"}
    overlay_names = {"dynamic_movement_transactions", "dynamic_migration_edges", "dynamic_physical_bytes"}
    mtp_restores = 0.0
    for record in records:
        domain, name = record.get("domain"), record.get("name")
        if domain not in {"mtp", "prefix_cache", "moe_rebalance",
                          "moe_overlay_controller", "moe_overlay_residency"}:
            continue
        value = _positive(record.get("value", record.get("count", 0)))
        if (domain == "moe_overlay_controller" and value > 0
                and name in {"prepared_context_restore_movement_waves",
                             "prepared_context_restore_certifications"}):
            return "server disposal restored expert placement without a retained model context"
        if domain == "moe_overlay_residency" and name == "prepared_context_restoration_edges" and value > 0:
            return "server disposal restored expert placement without a retained model context"
        totals[(domain, name)] += value
        if domain == "moe_overlay_residency":
            scope = (record.get("rank"), record.get("device"), record.get("phase"))
            if name == "expert_migration_edges" and value > 0:
                host_edge_scopes.add(scope)
            tags = record.get("tags") or {}
            if tags.get("purpose") == "placement_change":
                if name in {"placement_transfer_operations_completed", "placement_transfer_payload_bytes_completed",
                            "placement_published_payload_bytes"}:
                    placement_payload += value
                if name in host_names:
                    if name in host_publications[scope]:
                        invalid_publication = True
                    host_publications[scope][name] = record
        if domain == "moe_overlay_controller" and name in overlay_names and value > 0:
            identity = _overlay_transaction(record)
            if identity is None:
                invalid_publication = True
            else:
                # Edge rows are distinct experts in one completed wave, so
                # aggregate only after retaining rank/transaction identity.
                observation = overlay_observations[identity]
                observation[name] = observation.get(name, 0.0) + value
        if domain == "moe_rebalance":
            # These are snapshots, not additive movement events. Keep each
            # publication intact; do not pair copy/apply evidence across ranks,
            # devices, phases, or request-reset observations.
            identity = (record.get("rank"), record.get("device"), record.get("phase"),
                        json.dumps(record.get("tags") or {}, sort_keys=True))
            native_observations[identity][name] = value
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
    native_names = ("device_rebalance_request_copied_payload_lower_bound",
                    "device_rebalance_request_applied_payload_lower_bound",
                    "device_rebalance_request_useful_payload_bytes_lower_bound")
    completed_native = any(all(observation.get(name, 0) > 0 for name in native_names)
                           for observation in native_observations.values())
    overlay_edges = totals[("moe_overlay_controller", "dynamic_migration_edges")]
    overlay_transactions = totals[("moe_overlay_controller", "dynamic_movement_transactions")]
    overlay_bytes = totals[("moe_overlay_controller", "dynamic_physical_bytes")]
    # These counters follow transactionComplete() and physical source retirement.
    # Do not substitute the command count or prepared-arrival diagnostics.
    completed_overlay = bool(overlay_observations) and all(
        all(observation.get(name, 0) > 0 for name in overlay_names)
        for observation in overlay_observations.values())
    completed_host = bool(host_edge_scopes) and host_edge_scopes == set(host_publications)
    published_bytes = 0.0
    for scope, publication in host_publications.items():
        transport = _publication_sequence(publication.get("placement_transport_publications"))
        owner = _publication_sequence(publication.get("placement_owner_publications"))
        completed_host &= transport is not None and transport == owner
        payload = publication.get("placement_published_payload_bytes", {}).get("value")
        valid_payload = type(payload) in (int, float) and 0 <= payload < 2**64 and math.isfinite(payload)
        completed_host &= valid_payload
        published_bytes += payload if valid_payload else 0
    # A rank can own only no-op projections for a particular wave. All rank
    # publications must match, and the completed cohort must carry real bytes;
    # bytes copied before an aborted publication cannot substitute for them.
    completed_host &= published_bytes > 0
    completed_movement = completed_host or completed_native or completed_overlay
    if movement_evidence is MovementEvidence.REQUIRED and (
            invalid_publication or (committed_edges > 0 and not completed_host)
            or (overlay_observations and not completed_overlay)):
        return "dynamic movement certification has inconsistent committed physical publication evidence"
    if movement_evidence is MovementEvidence.REQUIRED and not completed_movement:
        return "dynamic movement certification observed no committed physical expert movement"
    if movement_evidence is MovementEvidence.FORBIDDEN and (
            committed_edges > 0 or applied > 0 or physical_bytes > 0
            or any(totals[("moe_rebalance", name)] > 0 for name in native_names)
            or overlay_edges > 0 or overlay_transactions > 0 or overlay_bytes > 0
            or placement_payload > 0 or host_publications):
        return "static movement certification observed expert movement"
    return None
