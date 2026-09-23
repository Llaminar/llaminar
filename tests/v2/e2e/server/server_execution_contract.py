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
from collections import Counter
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


def automatic_selection_policy(profile: Mapping[str, Any]) -> dict:
    """Read the canonical cell's constraints without expanding or choosing topology."""
    selection = profile.get("planning")
    if (not isinstance(selection, dict) or selection.get("mode") != "auto"
            or selection.get("strategy") not in {"single", "tp", "pp", "expert-overlay"}):
        raise ValueError("E2E requires an explicit automatic planning contract; rebuild/export stale manifests")
    counts = selection.get("device_counts")
    if (not isinstance(counts, dict) or not counts or not set(counts) <= {"cpu", "cuda", "rocm"}
            or any(type(count) is not int or count <= 0 for count in counts.values())):
        raise ValueError("E2E automatic planning requires exact positive physical device counts")
    ranks = selection.get("mpi_ranks")
    if type(ranks) is not int or ranks <= 0:
        raise ValueError("E2E automatic planning requires an exact positive MPI rank count")
    result = {"mode": "auto", "strategy": selection["strategy"],
              "device_counts": dict(counts), "mpi_ranks": ranks}
    if "pipeline_domains" in selection or "pipeline_layers" in selection:
        domains, layers = selection.get("pipeline_domains"), selection.get("pipeline_layers")
        if (selection["strategy"] != "pp" or type(layers) is not int or layers <= 0
                or not isinstance(domains, list) or len(domains) < 2):
            raise ValueError("automatic pipeline contract requires domains and model layer count")
        totals = Counter()
        for domain in domains:
            if (not isinstance(domain, dict) or set(domain) != {"backend", "devices"}
                    or domain["backend"] not in counts or type(domain["devices"]) is not int
                    or domain["devices"] <= 0):
                raise ValueError("automatic pipeline contract has invalid domain membership")
            totals[domain["backend"]] += domain["devices"]
        if dict(totals) != counts or layers < len(domains):
            raise ValueError("automatic pipeline domains disagree with devices/layers")
        result.update(pipeline_domains=[dict(domain) for domain in domains], pipeline_layers=layers)
    return result


def _validate_pipeline_domains(rows: list, participants: dict, expected: dict) -> list[dict]:
    """Prove the actual TP groups and contiguous layer cover without choosing their order.

    Only rank-local pipeline cells currently declare this obligation. Physical
    device identity is authenticated by the caller before joining these rows.
    Singleton stages cannot masquerade as a TP domain, and aliases cannot make
    a device belong to two stages. Auto still chooses which vendor comes first
    and how many layers to place there.
    """
    domains = {}
    owners = set()
    ranks = set()
    for row in rows:
        if row.get("name") != "execution_pipeline_domain":
            continue
        rank, tags = row.get("rank"), row.get("tags", {})
        if (type(rank) is not int or row.get("value") != 1 or row.get("phase") != "startup"
                or tags.get("schema") != "1" or tags.get("source") != "resolved_execution_plan"
                or tags.get("scope") != "rank_local"):
            raise ValueError("automatic pipeline has invalid domain provenance")
        integers = {}
        for key in ("stage", "stages", "first_layer", "end_layer"):
            value = tags.get(key)
            if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]*", value):
                raise ValueError("automatic pipeline has invalid domain/layer coordinates")
            integers[key] = int(value)
        devices = _devices(tags.get("devices"))
        identities = {participants.get((rank, device)) for device in devices}
        if (not devices or None in identities or owners & identities
                or len(identities) != len(devices) or len({identity[1] for identity in identities}) != 1
                or integers["stages"] != len(expected["pipeline_domains"])
                or integers["stage"] in domains):
            raise ValueError("automatic pipeline has incomplete, duplicated or nonhomogeneous domain ownership")
        ranks.add(rank)
        owners.update(identities)
        domains[integers["stage"]] = (integers, {"backend": next(iter(identities))[1], "devices": len(identities)})
    if (len(ranks) != 1 or owners != set(participants.values())
            or set(domains) != set(range(len(expected["pipeline_domains"])))):
        raise ValueError("automatic pipeline did not prove every declared domain/participant")
    frontier = 0
    observed = []
    for index in range(len(domains)):
        bounds, domain = domains[index]
        if bounds["first_layer"] != frontier or bounds["end_layer"] <= frontier:
            raise ValueError("automatic pipeline layer coverage has a gap, overlap or empty stage")
        frontier = bounds["end_layer"]
        observed.append(domain)
    signature = lambda domains: Counter((domain["backend"], domain["devices"]) for domain in domains)
    if frontier != expected["pipeline_layers"] or signature(observed) != signature(expected["pipeline_domains"]):
        raise ValueError("automatic pipeline domain shape or model layer coverage mismatch")
    return observed


def validate_automatic_selection(data: Mapping[str, Any], selection: Mapping[str, Any]) -> dict:
    """Join resolved execution to physical inventory, then check the cell's constraints.

    Device ordinals are rank-local visibility, not physical identity. Deduplicate
    by inventory node, backend and UUID/NUMA endpoint; neither idle GPUs nor two
    MPI views of one GPU can satisfy a multi-device case. This is observation
    after shutdown, never a second planner or a live execution authority.
    """
    expected = automatic_selection_policy({"planning": selection})
    validate_server_execution_contract(data)
    if data["world_size"] != expected["mpi_ranks"]:
        raise ValueError(f"automatic selection MPI rank mismatch: observed {data['world_size']}, "
                         f"expected {expected['mpi_ranks']}")
    rows = [row for row in data["records"] if row.get("domain") == "server"]
    memberships, participants, visible = {}, {}, set()
    strategy = None
    for row in rows:
        name, rank, tags = row.get("name"), row.get("rank"), row.get("tags", {})
        if name == "execution_policy":
            strategy = tags.get("execution_strategy")
        elif name == "execution_topology":
            visible.update((rank, device) for device in _devices(tags["devices"]))
        elif name == "rank_membership":
            node = tags.get("node_id")
            if (rank in memberships or type(rank) is not int or not 0 <= rank < data["world_size"]
                    or row.get("value") != 1 or row.get("phase") != "startup"
                    or tags.get("identity_source") != "communicator_cluster_inventory"
                    or tags.get("rank") != str(rank) or tags.get("world_size") != str(data["world_size"])
                    or tags.get("authority_rank") != str(data["authority_rank"])
                    or not isinstance(node, str) or not re.fullmatch(r"0|[1-9][0-9]*", node)):
                raise ValueError("automatic selection has incomplete physical rank identity")
            memberships[rank] = node
        elif name == "execution_participant":
            device = row.get("device")
            if (type(rank) is not int or not 0 <= rank < data["world_size"]
                    or row.get("value") != 1 or row.get("phase") != "startup"
                    or tags.get("schema") != "1" or tags.get("source") != "resolved_execution_plan"
                    or tags.get("identity_source") != "communicator_cluster_inventory"
                    or len(_devices(device)) != 1 or (rank, device) in participants):
                raise ValueError("automatic selection has invalid/duplicate execution participant")
            backend, physical_id = tags.get("backend"), tags.get("physical_id")
            if (backend != device.partition(":")[0] or not isinstance(physical_id, str) or not physical_id
                    or (backend == "CPU" and not re.fullmatch(r"numa:(?:0|[1-9][0-9]*)", physical_id))):
                raise ValueError("automatic selection participant lacks physical UUID/NUMA identity")
            participants[rank, device] = (tags.get("node_id"), backend.lower(), physical_id)
    if set(memberships) != set(range(data["world_size"])) or set(participants) != visible:
        raise ValueError("automatic selection omitted ranks/participants or included idle devices")
    if any(identity[0] != memberships[rank] for (rank, _), identity in participants.items()):
        raise ValueError("automatic selection participant belongs to another physical node")
    counts = dict(Counter(backend for _, backend, _ in set(participants.values())))
    if strategy != expected["strategy"] or counts != expected["device_counts"]:
        raise ValueError(f"automatic selection mismatch: observed strategy={strategy}, devices={counts}; expected {expected}")
    result = {"strategy": strategy, "device_counts": counts,
              "mpi_ranks": data["world_size"]}
    if "pipeline_domains" in expected:
        result["pipeline_domains"] = _validate_pipeline_domains(rows, participants, expected)
        result["pipeline_layers"] = expected["pipeline_layers"]
    return result
