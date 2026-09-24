#!/usr/bin/env python3
"""Prove remote expert work from the existing rank-qualified server observations.

This is an additive HTTP E2E observer, not an MPI launcher, inventory collector,
placement controller or replacement for graph/MTP/prefix/movement assertions.
The typed matrix supplies the expected remote-host count. Runtime membership
supplies physical node identity; paired transport witnesses supply actual MPI
traffic; completed local-expert rows supply computation. None alone suffices.

The returned compact summary is useful evidence, but cannot certify an image
without the outer phase's exact image/ISA, frontend, HTTP and cloud-retirement
proofs. In particular a hostname is a label, never a physical-node authority.
"""
from __future__ import annotations

from collections import defaultdict
import math
import re


def _integer(value: object, label: str, *, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    """Validate JSON integer identities without accepting bools or numeric strings."""
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"cross-host evidence has invalid {label}")
    return value


def _tag(tags: dict, name: str) -> int:
    """Read the collector's canonical unsigned decimal topology tags."""
    value = tags.get(name)
    if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError(f"cross-host evidence has invalid {name} tag")
    return _integer(int(value), name)


def _depth_tag(tags: dict) -> int:
    """Main graphs use -1; speculative ordinals are canonical nonnegative int32s."""
    value = tags.get("mtp_depth")
    if not isinstance(value, str) or not re.fullmatch(r"-1|0|[1-9][0-9]*", value):
        raise ValueError("cross-host evidence has invalid mtp_depth tag")
    return _integer(int(value), "mtp_depth", minimum=-1, maximum=2**31 - 1)


def _counter(row: dict) -> int:
    """Counters are exact nonnegative integers within double's exact range."""
    value = row.get("value")
    if (row.get("kind") != "counter" or type(value) not in (int, float)
            or value < 0 or value > 2**53 - 1 or not math.isfinite(value) or int(value) != value):
        raise ValueError("cross-host evidence has invalid integer counter")
    _integer(row.get("count"), "counter observation count", minimum=1)
    return int(value)


def _membership(data: dict, remote_hosts: int) -> tuple[int, dict[int, dict]]:
    """Require the exact execution communicator and distinct MPI physical groups."""
    size = _integer(data.get("world_size"), "world size", minimum=1)
    authority = _integer(data.get("authority_rank"), "authority rank", maximum=size - 1)
    if size != remote_hosts + 1:
        raise ValueError("cross-host execution rank count differs from the canonical topology")
    members = {}
    for row in data["records"]:
        if row.get("domain") != "server" or row.get("name") != "rank_membership":
            continue
        rank = _integer(row.get("rank"), "membership rank", maximum=size - 1)
        tags = row.get("tags")
        if (not isinstance(tags, dict) or rank in members or _counter(row) != 1
                or row["count"] != 1 or row.get("phase") != "startup"
                or tags.get("identity_source") != "communicator_cluster_inventory"
                or _tag(tags, "rank") != rank or _tag(tags, "world_size") != size
                or _tag(tags, "authority_rank") != authority):
            raise ValueError("missing, conflicting or repeated physical server membership")
        hostname = tags.get("hostname")
        if not isinstance(hostname, str) or not hostname or any(c in hostname for c in "\x00\r\n"):
            raise ValueError("cross-host membership omitted its diagnostic hostname")
        members[rank] = {"node_id": _tag(tags, "node_id"), "hostname": hostname}
        if _tag(tags, "local_rank") != 0:
            raise ValueError("cross-host topology requires one execution rank per physical host")
    if set(members) != set(range(size)):
        raise ValueError("cross-host evidence omitted a server participant")
    if len({row["node_id"] for row in members.values()}) != size:
        raise ValueError("cross-host ranks share a physical host; remote inference was not proved")
    return authority, members


_DIRECTIONS = {"Dispatch": "moe_overlay_rank_batch_dispatch_transactions",
               "ReturnReduce": "moe_overlay_rank_batch_return_transactions"}
_EMPTY_COUNTER = "moe_overlay_rank_batch_empty_return_transactions"
_EMPTY_SEQUENCE = "moe_overlay_rank_batch_empty_return_sequence"
_COMPLETION_SEQUENCE = "moe_overlay_graph_completion_sequence"
_PHASES = {"prefill", "decode", "grouped_verifier"}
_GEOMETRY = ("source_world_rank", "target_world_rank", "domain_ordinal", "layer", "mtp_depth", "tier")


def validate_cross_host_expert_execution(data: dict, topology: dict) -> dict:
    """Require completed prefill/decode expert work and matched MPI on every remote rank.

    Transport source/target mean immutable continuation/expert endpoints even
    for ReturnReduce; they are not reversed for the returning payload. Endpoint
    observations stay separate until their counters and ordered digests agree.
    Counts/bytes are credited once per wire exchange, not once per observer.
    Empty route envelopes cannot pass because every remote endpoint also needs
    positive completed CPU routes in both prefill and decode/verifier phases.
    """
    if not isinstance(topology, dict) or topology.get("kind") != "cross-host-expert-overlay":
        raise ValueError("cross-host proof requires its canonical typed topology")
    remote_count = _integer(topology.get("remote_cpu_hosts"), "remote CPU hosts", minimum=1,
                            maximum=2**31 - 2)
    if (type(topology.get("execution_ranks")) is not int or topology["execution_ranks"] != remote_count + 1
            or type(topology.get("continuation_devices")) is not int or topology["continuation_devices"] != 1
            or type(topology.get("cpu_ranks_per_host")) is not int or topology["cpu_ranks_per_host"] != 1
            or topology.get("continuation_backend") not in ("cuda", "rocm")):
        raise ValueError("cross-host proof has inconsistent canonical participant geometry")
    if (not isinstance(data, dict) or data.get("schema") != "llaminar.perf_stats.v1"
            or not isinstance(data.get("records"), list)
            or not all(isinstance(row, dict) for row in data["records"])):
        raise ValueError("cross-host proof requires complete rank-qualified PerfStats")
    authority, members = _membership(data, remote_count)
    remote = set(members) - {authority}
    # Keyed by exact graph/transport geometry, not by rank totals: mismatched
    # layers, domains, MTP depths and directions cannot cancel each other out.
    sequences, counters, empty_sequences, empty_counters, completions = {}, {}, {}, {}, {}
    routes = defaultdict(int)
    transport_names = {*_DIRECTIONS.values(), "moe_overlay_rank_batch_payload_bytes",
                       "moe_overlay_rank_batch_sequence", _EMPTY_COUNTER, _EMPTY_SEQUENCE}
    for row in data["records"]:
        if row.get("domain") != "forward_graph":
            continue
        name = row.get("name")
        if name not in transport_names and name not in ("moe_overlay_local_expert_active_routes", _COMPLETION_SEQUENCE):
            continue
        rank = _integer(row.get("rank"), "observing rank", maximum=len(members) - 1)
        tags = row.get("tags")
        if not isinstance(tags, dict):
            raise ValueError("cross-host work/transport omitted typed tags")
        if name == _COMPLETION_SEQUENCE:
            source, target = _tag(tags, "source_world_rank"), _tag(tags, "target_world_rank")
            endpoint = tags.get("endpoint_role")
            if (source != authority or target not in remote or endpoint not in ("source", "target")
                    or rank != (source if endpoint == "source" else target)
                    or row.get("device") != "mpi" or row.get("phase") != "inference"
                    or row.get("kind") != "ordered_sequence"):
                raise ValueError("cross-host graph completion has invalid endpoint ownership")
            count = _integer(row.get("count"), "graph completion count", minimum=1)
            if (type(row.get("value")) not in (int, float) or row["value"] != count
                    or type(row.get("sequence_word_count")) is not int or row["sequence_word_count"] != 14 * count):
                raise ValueError("cross-host graph completion has an invalid receipt ABI")
            indexed = (target, endpoint)
            if indexed in completions:
                raise ValueError("duplicate cross-host graph completion")
            completions[indexed] = (count, *(_integer(row.get(field), field, maximum=2**64 - 1)
                for field in ("sequence_digest_lo", "sequence_digest_hi")))
            continue
        if name == "moe_overlay_local_expert_active_routes":
            if rank not in remote:
                continue  # Continuation's local experts are not remote CPU proof.
            value = _counter(row)
            phase = tags.get("service_source")
            if (tags.get("completion") != "local_expert_packet_complete"
                    or tags.get("identity_source") != "sparse_collective_key"
                    or tags.get("device_kind") != "CPU" or phase not in _PHASES):
                raise ValueError("remote CPU routes lack completed production expert computation")
            if value and (not _tag(tags, "input_rows") or not _tag(tags, "output_rows")):
                raise ValueError("remote expert computation has no live input/return rows")
            routes[rank, _tag(tags, "tier"), phase] += value
            continue
        geometry = tuple(_depth_tag(tags) if key == "mtp_depth" else _tag(tags, key) for key in _GEOMETRY)
        source, target = geometry[:2]
        direction, endpoint = tags.get("direction"), tags.get("endpoint_role")
        if source != authority or target not in remote or direction not in _DIRECTIONS:
            raise ValueError("cross-host traffic has a foreign continuation/expert endpoint")
        if endpoint not in ("source", "target") or rank != (source if endpoint == "source" else target):
            raise ValueError("cross-host transport observer does not own its endpoint")
        if tags.get("transport") != "mpi" or row.get("device") != "mpi":
            raise ValueError("cross-host inference used a node-local activation channel")
        if _tag(tags, "participant_count") <= 0:
            raise ValueError("cross-host exchange has no admitted expert participants")
        key = (*geometry, direction, _tag(tags, "participant_count"))
        empty_outcome = name in (_EMPTY_COUNTER, _EMPTY_SEQUENCE)
        if empty_outcome and direction != "ReturnReduce":
            raise ValueError("empty numerical outcome must name the return direction")
        if name in ("moe_overlay_rank_batch_sequence", _EMPTY_SEQUENCE):
            phase = row.get("phase")
            if row.get("kind") != "ordered_sequence" or phase not in _PHASES:
                raise ValueError("cross-host exchange omitted production-phase ordering evidence")
            if (phase == "grouped_verifier") != (geometry[4] >= 0):
                raise ValueError("cross-host exchange depth marker disagrees with its inference phase")
            count = _integer(row.get("count"), "ordered transaction count", minimum=1)
            words = _integer(row.get("sequence_word_count"), "ordered word count", minimum=1)
            if words != 4 * count or row.get("value") != count or type(row.get("value")) not in (int, float):
                raise ValueError("cross-host ordered witness has the wrong transaction ABI")
            signature = (count, words, *(_integer(row.get(field), field, maximum=2**64 - 1)
                for field in ("sequence_digest_lo", "sequence_digest_hi")))
            indexed, value = (key, phase, endpoint), signature
            destination = empty_sequences if empty_outcome else sequences
        else:
            if name not in ("moe_overlay_rank_batch_payload_bytes", _EMPTY_COUNTER, _DIRECTIONS[direction]):
                raise ValueError("cross-host transaction counter disagrees with its direction")
            if row.get("phase") != "moe_overlay":
                raise ValueError("cross-host transaction counter has a foreign phase")
            indexed, value = (key, name, endpoint), (_counter(row), row["count"])
            destination = empty_counters if empty_outcome else counters
        if indexed in destination:
            raise ValueError("duplicate cross-host transport observation")
        destination[indexed] = value

    if not sequences:
        raise ValueError("cross-host proof has no completed MPI exchange")
    for target in remote:
        if ((target, "source") not in completions or
                completions.get((target, "source")) != completions.get((target, "target"))):
            raise ValueError("cross-host graph completion receipts are missing or mismatched")
    totals = defaultdict(int)
    paired = set()
    for key, phase, endpoint in sequences:
        signature = sequences[key, phase, endpoint]
        peer = "target" if endpoint == "source" else "source"
        if sequences.get((key, phase, peer)) != signature:
            raise ValueError("cross-host sender/receiver transaction sequences differ")
        if endpoint == "source":
            totals[key] += signature[0]
            paired.add((key, phase))
    expected_counters = set()
    result = {rank: {**members[rank], "rank": rank, "prefill_routes": 0, "decode_routes": 0,
                     "dispatch_transactions": 0, "return_transactions": 0,
                     "empty_return_outcomes": 0,
                     "completed_graph_transactions": completions[rank, "source"][0],
                     "dispatch_bytes": 0, "return_bytes": 0} for rank in sorted(remote)}
    for key, count in totals.items():
        direction = key[-2]
        for name in (_DIRECTIONS[direction], "moe_overlay_rank_batch_payload_bytes"):
            expected_counters.update((key, name, endpoint) for endpoint in ("source", "target"))
            source_value = counters.get((key, name, "source"))
            if source_value is None or counters.get((key, name, "target")) != source_value:
                raise ValueError("cross-host sender/receiver counters or bytes differ")
            value, observations = source_value
            if observations != count or (value != count if name == _DIRECTIONS[direction] else value <= 0):
                raise ValueError("cross-host counters do not cover the ordered exchange sequence")
            field = ("dispatch" if direction == "Dispatch" else "return") + (
                "_transactions" if name == _DIRECTIONS[direction] else "_bytes")
            result[key[1]][field] += value
    if counters.keys() != expected_counters:
        raise ValueError("cross-host traffic counters have no matched ordered witness")
    # Empty outcomes are independently paired and never enter the physical
    # transaction/byte totals above. Each geometry still has a complete outcome.
    empty_totals = defaultdict(int)
    for (key, phase, endpoint), signature in empty_sequences.items():
        peer = "target" if endpoint == "source" else "source"
        if empty_sequences.get((key, phase, peer)) != signature:
            raise ValueError("cross-host empty outcome sequences differ")
        if endpoint == "source":
            empty_totals[key] += signature[0]
    expected_empty_counters = set()
    for key, count in empty_totals.items():
        for endpoint in ("source", "target"):
            indexed = (key, _EMPTY_COUNTER, endpoint)
            expected_empty_counters.add(indexed)
            if empty_counters.get(indexed) != (count, count):
                raise ValueError("cross-host empty outcome counter has no matching sequence")
        result[key[1]]["empty_return_outcomes"] += count
    if empty_counters.keys() != expected_empty_counters:
        raise ValueError("cross-host empty counter has no matched ordered witness")
    # Reconcile the union, including orphan returns and empty-only geometries.
    outcomes = {(key[:-2] + (key[-1],), phase) for key, phase, _ in (*sequences, *empty_sequences)}
    for geometry, phase in outcomes:
        dispatch_key = (*geometry[:-1], "Dispatch", geometry[-1])
        return_key = (*geometry[:-1], "ReturnReduce", geometry[-1])
        dispatch_count = sequences.get((dispatch_key, phase, "source"), (0,))[0]
        return_count = sequences.get((return_key, phase, "source"), (0,))[0]
        empty_count = empty_sequences.get((return_key, phase, "source"), (0,))[0]
        if not dispatch_count or dispatch_count != return_count + empty_count:
            raise ValueError("cross-host dispatch has no completed matching return or empty outcome")
    observed_work_edges = {(key[1], key[5], phase) for key, phase in paired if key[-2] == "ReturnReduce"}
    for (rank, tier, phase), count in routes.items():
        if count:
            if (rank, tier, phase) not in observed_work_edges:
                raise ValueError("completed CPU routes have no matching MPI phase/tier")
            result[rank]["prefill_routes" if phase == "prefill" else "decode_routes"] += count
    for rank, summary in result.items():
        if any(summary[field] <= 0 for field in ("prefill_routes", "decode_routes", "dispatch_bytes", "return_bytes")):
            raise ValueError(f"remote CPU rank {rank} did not complete both prefill and decode expert work")
    return {"schema": 1, "authority_rank": authority, "world_size": len(members),
            "remote_cpu_hosts": [result[rank] for rank in sorted(result)]}
