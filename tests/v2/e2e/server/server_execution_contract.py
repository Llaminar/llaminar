#!/usr/bin/env python3
"""Authenticate the runtime-owned topology and service policy after shutdown.

Every frontend converges on the same admitted execution plan. Its startup
projection supplies selected devices and attention ownership per rank, plus
one service-authority policy. This observer never plans placement or interprets
CLI strings. Missing evidence is fatal: discovering a GPU or observing some
graph work cannot silently exempt a configured feature or participant.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class RuntimeFeaturePolicy:
    """Immutable service selection; actual execution needs separate witnesses."""

    prefix_cache: bool = False
    mtp: bool = False
    mtp_verify_mode: str = "greedy"
    mtp_depth_policy: str = "fixed"
    mtp_min_depth: int = 1
    mtp_max_depth: int = 1
    expert_overlay: bool = False
    current_batch_llep: bool = False
    residency_maintenance: str = "off"


@dataclass(frozen=True)
class ServerExecutionContract:
    """Complete selected communicator projection, independent of CLI spelling."""

    device_kinds: frozenset[str]
    attention_device_kinds: frozenset[str]
    features: RuntimeFeaturePolicy

    @property
    def uses_gpu(self) -> bool:
        """Whether any selected participant requires native GPU graph proof."""
        return bool(self.device_kinds & {"cuda", "rocm"})


def _devices(value: Any) -> frozenset[str]:
    """Decode exact DeviceId spellings without scanning arbitrary strings."""
    if not isinstance(value, str):
        raise ValueError("execution topology requires a device list")
    values = value.split(",") if value else []
    if (len(set(values)) != len(values)
            or any(not re.fullmatch(r"CPU|(?:CUDA|ROCm):(?:0|[1-9][0-9]*)", v) for v in values)):
        raise ValueError("execution topology has invalid or duplicate devices")
    return frozenset(values)


def _boolean(tags: Mapping[str, Any], key: str) -> bool:
    """Reject missing flags rather than silently treating them as disabled."""
    if tags.get(key) not in ("true", "false"):
        raise ValueError(f"execution policy requires explicit {key}")
    return tags[key] == "true"


def _choice(tags: Mapping[str, Any], key: str, choices: set[str]) -> str:
    """Accept only the policy enum's current public spellings."""
    value = tags.get(key)
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"execution policy has invalid {key}")
    return value


def _positive(tags: Mapping[str, Any], key: str) -> int:
    """Read a positive decimal policy bound; graph capacity is not this bound."""
    value = tags.get(key)
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError(f"execution policy requires positive {key}")
    return int(value)


def validate_server_execution_contract(data: Mapping[str, Any]) -> ServerExecutionContract:
    """Require one topology per selected rank and one policy from the authority.

    The ranked artifact collector owns communicator authentication. This check
    additionally binds the new projection to that exact membership and rejects
    partial, contradictory or future-schema evidence. Relays may have empty
    device sets; a complete service must have selected model/attention devices.
    """
    size, authority = data.get("world_size"), data.get("authority_rank")
    if (type(size) is not int or size <= 0 or type(authority) is not int
            or not 0 <= authority < size):
        raise ValueError("execution contract requires authenticated communicator membership")
    topologies: dict[int, tuple[frozenset[str], frozenset[str]]] = {}
    policies = []
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError("execution contract requires records")
    for record in records:
        if record.get("domain") != "server" or record.get("name") not in {
                "execution_topology", "execution_policy"}:
            continue
        rank, tags = record.get("rank"), record.get("tags")
        if (type(rank) is not int or not 0 <= rank < size or record.get("value") != 1
                or record.get("phase") != "startup" or not isinstance(tags, dict)
                or tags.get("schema") != "1" or tags.get("source") != "resolved_execution_plan"):
            raise ValueError("invalid startup execution contract provenance")
        if record["name"] == "execution_policy":
            if rank != authority:
                raise ValueError("execution policy was not published by the service authority")
            policies.append(tags)
            continue
        if rank in topologies:
            raise ValueError("duplicate execution topology for rank")
        devices, attention = _devices(tags.get("devices")), _devices(tags.get("attention_devices"))
        if not attention <= devices:
            raise ValueError("attention ownership includes unselected devices")
        topologies[rank] = (devices, attention)
    if set(topologies) != set(range(size)):
        raise ValueError("missing execution topology for selected ranks")
    if len(policies) != 1:
        raise ValueError("expected exactly one service execution policy")
    tags = policies[0]
    features = RuntimeFeaturePolicy(
        prefix_cache=_boolean(tags, "prefix_cache"), mtp=_boolean(tags, "mtp"),
        mtp_verify_mode=_choice(tags, "mtp_verify_mode", {"greedy", "speculative-sampling"}),
        mtp_depth_policy=_choice(tags, "mtp_depth_policy", {"fixed", "observe", "dynamic"}),
        mtp_min_depth=_positive(tags, "mtp_min_depth"), mtp_max_depth=_positive(tags, "mtp_max_depth"),
        expert_overlay=_boolean(tags, "expert_overlay"),
        current_batch_llep=_boolean(tags, "current_batch_llep"),
        residency_maintenance=_choice(tags, "residency_maintenance", {"off", "observe", "dynamic"}))
    if features.mtp_min_depth > features.mtp_max_depth:
        raise ValueError("execution policy has reversed MTP bounds")
    kinds = frozenset(d.partition(":")[0].lower() for devices, _ in topologies.values() for d in devices)
    attention_kinds = frozenset(d.partition(":")[0].lower() for _, devices in topologies.values() for d in devices)
    if not kinds or not attention_kinds:
        raise ValueError("execution topology has no model participants")
    return ServerExecutionContract(kinds, attention_kinds, features)
