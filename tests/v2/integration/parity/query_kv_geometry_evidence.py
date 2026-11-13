#!/usr/bin/env python3
"""Validate existing cold QueryInterval/PMA/source facts in a native PerfStats artifact.

Reads JSON metadata only. Expected source names/formats and actual participant Q
intervals must come from the caller's authenticated original source/plan; this
module does not load models, discover devices, fit geometry or change execution.
Constructed-family evidence is explicitly distinct from per-instance replay.
"""
import argparse
import json
import math
from pathlib import Path
import re


def require(ok, message):
    """Reject incomplete or conflicting observations without producing a pass."""
    if not ok:
        raise ValueError(message)


def integer(value, name, minimum=0):
    """Read a nonnegative authored integer or canonical decimal PerfStats tag."""
    if isinstance(value, str):
        require(re.fullmatch(r"0|[1-9][0-9]*", value) is not None, f"invalid {name}")
        value = int(value)
    require(type(value) is int and value >= minimum, f"invalid {name}")
    return value


def validate_geometry(artifact, expected):
    """Join complete source/assignment/cache/PMA metadata and real parent-path evidence.

    This proves constructed source-slice metadata and installed admission inputs.
    It deliberately cannot certify a particular parent replay's individual leaf.
    No tensor, executable, model or runtime payload is opened or hashed.
    """
    require(type(expected) is dict and type(artifact) is dict, "expected metadata objects")
    require(expected.get("schema") == "query_kv_geometry_expectation_v1", "unknown expectation schema")
    q = integer(expected.get("original_q"), "original_q", 1)
    kv = integer(expected.get("original_kv"), "original_kv", 1)
    width = integer(expected.get("head_dim"), "head_dim", 1)
    k = integer(expected.get("k"), "k", 1)
    rank = integer(expected.get("world_rank"), "world_rank")
    require(q % kv == 0, "original GQA ratio is not integral")
    participants = expected.get("participants")
    sources = expected.get("sources")
    require(type(participants) is list and 2 <= len(participants) <= 8, "expected complete TP2..8 participants")
    require(type(sources) is list and sources, "missing original K/V source inventory")
    degree, ratio = len(participants), q // kv
    owners, all_q = {}, []
    for index, participant in enumerate(participants):
        require(type(participant) is dict, "invalid participant metadata")
        device = participant.get("device")
        require(type(device) is str and re.fullmatch(r"(?:ROCm|CUDA):[0-9]+", device), "invalid actual device")
        require(device not in owners, "duplicate actual device")
        first = integer(participant.get("q_first"), "q_first")
        count = integer(participant.get("q_count"), "q_count", 1)
        require(first + count <= q, "foreign query interval")
        start, end = first // ratio, (first + count - 1) // ratio + 1
        owners[device] = dict(participant=index, degree=degree, original_q=q, original_kv=kv,
                              head_dim=width, k=k, q_first=first, q_count=count,
                              kv_first=start, kv_count=end-start)
        all_q.extend(range(first, first+count))
    require(sorted(all_q) == list(range(q)), "query ownership is incomplete or overlapping")
    require(len({device.split(':')[0] for device in owners}) == 1, "mixed backend membership")
    main_layers = integer(expected.get("main_layer_count"), "original main layer count", 1)
    attention_layers = expected.get("attention_layers")
    require(type(attention_layers) is list and attention_layers, "missing original sparse layer declaration")
    declared_layers = [integer(layer, "original sparse layer") for layer in attention_layers]
    require(len(set(declared_layers)) == len(declared_layers) and max(declared_layers) < main_layers,
            "foreign or duplicate original sparse layer")
    source_map = {}
    for source in sources:
        require(type(source) is dict and source.get("role") in ("K", "V"), "invalid original role")
        layer = integer(source.get("layer"), "source layer")
        role = source["role"]
        require(source.get("name") == f"blk.{layer}.attn_{role.lower()}.weight", "foreign original source name")
        key = layer, role
        require(key not in source_map, "duplicate source identity")
        integer(source.get("codebook"), "original codebook")
        require(type(source.get("superblock")) is bool, "missing original source family")
        source_map[key] = source
    require(set(source_map) == {(layer, role) for layer in declared_layers for role in ("K", "V")}, "incomplete original K/V roles")
    require(artifact.get("schema") == "llaminar.perf_stats.v1", "unknown actual PerfStats schema")
    records = artifact.get("records")
    require(type(records) is list, "missing actual PerfStats records")
    seen = {device: dict(admission=[], cache=[], projections={}) for device in owners}
    replay_devices = set()
    for record in records:
        require(type(record) is dict, "invalid PerfStats record")
        geometry = record.get("domain") == "gpu_graph_inventory" and record.get("name") == "query_kv_geometry"
        replay = record.get("domain") == "forward_graph" and record.get("name") in (
            "full_graph_replay_calls", "retained_parent_replays")
        if not geometry and not replay:
            continue
        require(record.get("kind") == "counter", "relevant evidence is not an actual counter")
        count, value = record.get("count"), record.get("value")
        require(type(count) is int and count > 0 and type(value) in (int, float) and
                math.isfinite(value) and value == count, "malformed positive construction/replay counter")
        device = record.get("device")
        if replay:
            require(record.get("phase") == "decode", "unexpected actual parent replay phase")
            if device in owners:
                replay_devices.add(device)
            continue
        require(device in owners, "geometry observation names a foreign device")
        tags = record.get("tags")
        require(type(tags) is dict and tags.get("schema") == "query_kv_geometry_v1", "missing geometry schema")
        require(tags.get("construction_only") == "true" and tags.get("per_instance_replay_proven") == "false",
                "construction evidence was relabeled as per-instance replay")
        integer(tags.get("admission_identity"), "actual admission identity", 1)
        kind = tags.get("kind")
        phases = {"selected_admission_input":"admission_installed", "graph_cache_pma":"cache_constructed",
                  "prepared_projection":"cold_binding"}
        require(kind in phases and record.get("phase") == phases[kind], "wrong successful setup publication phase")
        if kind in ("selected_admission_input", "graph_cache_pma"):
            require(tags.get("coverage") == "QueryInterval", "Legacy coverage cannot qualify")
            for name, value in owners[device].items():
                require(integer(tags.get(name), name) == value, f"{device} mismatched assignment {name}")
            require(integer(tags.get("world_rank"), "world rank") == rank, "foreign PMA rank")
            planned = integer(tags.get("kv_planned_bytes"), "KV owner planned bytes", 1)
            require(integer(tags.get("kv_resident_bytes"), "KV resident bytes") <= planned, "invalid retained owner bytes")
            if kind == "graph_cache_pma":
                for key in ("cache", "payload"):
                    require(integer(tags.get(key+"_first"), key+" first") == owners[device]["kv_first"] and
                            integer(tags.get(key+"_count"), key+" count", 1) == owners[device]["kv_count"],
                            f"{device} actual {key} origin/count mismatch")
                require(0 < integer(tags.get("kv_materialized_bytes"), "live KV owner bytes") <= planned,
                        "invalid actual materialized KV owner")
            seen[device]["admission" if kind == "selected_admission_input" else "cache"].append(tags)
        elif kind == "prepared_projection":
            layer = integer(tags.get("source_layer"), "actual source layer")
            key = layer, tags.get("role")
            require(key in source_map, "foreign/changed Q or K/V source identity")
            source = source_map[key]
            require(tags.get("source_name") == source["name"], "source name differs")
            require(integer(tags.get("source_codebook"), "source codebook") == source["codebook"] and
                    tags.get("source_superblock") == str(source["superblock"]).lower(), "source family differs")
            for name in ("binding_id", "stage_variant_identity"):
                integer(tags.get(name), name, 1)
            integer(tags.get("model_id"), "model_id")
            require(integer(tags.get("physical_n"), "physical N", 1) == owners[device]["kv_count"]*width and
                    integer(tags.get("first_column"), "first column") == owners[device]["kv_first"]*width and
                    integer(tags.get("policy_n"), "original N", 1) == kv*width and
                    integer(tags.get("k"), "K", 1) == k, "physical/original projection interval differs")
            seen[device]["projections"].setdefault(key, []).append(tags)
        else:
            raise ValueError("unknown geometry observation kind")
    for device, group in seen.items():
        require(group["admission"] and group["cache"], f"{device} missing selected admission/cache evidence")
        require(set(group["projections"]) == set(source_map), f"{device} missing original source binding")
        joined = group["admission"] + group["cache"] + [tag for values in group["projections"].values() for tag in values]
        require(len({tag["admission_identity"] for tag in joined}) == 1, f"{device} torn PMA owner identity")
        owner_lines = group["admission"] + group["cache"]
        require(len({(tag["kv_planned_bytes"], tag["kv_resident_bytes"]) for tag in owner_lines}) == 1,
                f"{device} admission/cache owner bytes disagree")
    require(replay_devices == set(owners), "missing actual parent replay evidence")
    return dict(schema="query_kv_geometry_evidence_v1", passed=True, participants=degree,
                original_source_roles=len(source_map), constructed_source_slice_geometry_proven=True,
                selected_admission_inputs_proven=True, actual_cache_geometry_proven=True,
                parent_replay_evidence_present=True, normalized_execution_family_proven=False,
                workspace_extent_proven=False, per_instance_replay_proven=False)


def main():
    """Validate two metadata artifacts; never open paths named inside them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perf-stats", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_geometry(json.loads(args.perf_stats.read_text()),
                                       json.loads(args.expected.read_text())), indent=2))


if __name__ == "__main__":
    main()
