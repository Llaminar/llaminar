#!/usr/bin/env python3
"""Validate actual-program diagnostic SDK receipts without granting native limits.

This is an offline inspector, not a second process runner or admission ledger.
The canonical runner owns child/ELF/external-device/reset-receipt authentication.
Host samples include observer overhead and cannot become a General certificate
without the separate complete-program review and cold/warm/prefix campaign.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter, deque

PREFIX = "POOLED_NATIVE_API "
INVENTORY_PREFIX = "POOLED_NATIVE_GRAPH_INVENTORY "
FIELDS = ("rss_bytes", "peak_rss_bytes", "glibc_live_bytes", "glibc_arena_bytes", "glibc_mapped_bytes", "monotonic_ns")
CREATORS = {"EventCreate": "event", "GraphCreate": "graph", "GraphClone": "graph", "StreamCreateWithFlags": "stream", "EventCreateWithFlags": "event", "StreamEndCapture": "graph", "GraphInstantiateWithFlags": "executable", "GraphInstantiate": "executable"}
DESTROYERS = {"StreamDestroy": "stream", "EventDestroy": "event", "GraphDestroy": "graph", "GraphExecDestroy": "executable"}
OPERATIONS = set(CREATORS) | set(DESTROYERS) | {"StreamBeginCapture", "GraphLaunch", "EventRecord", "StreamWaitEvent", "DeviceReset"}


def parse_records(lines):
    """Retain only exact labeled receipts; malformed labeled JSON is fatal."""
    rows = []
    for line in lines:
        if line.startswith(PREFIX):
            rows.append(json.loads(line[len(PREFIX):]))
    return rows


def validate_records(rows, expected_instantiation_failure=0):
    """Validate observed resource ancestry; nested SDK forwarding is one handle, not two."""
    if not rows:
        raise ValueError("Missing actual-program SDK observations")
    pending, ordinals, live, completed = {}, {}, {}, []
    parents, stacks, destroyed, captures = {}, {}, {}, {}
    inputs, births, injected, lineage = {}, {}, [], []
    fields = set(FIELDS) | {"schema", "pid", "thread", "parent_backend", "parent_operation", "parent_ordinal",
                            "qualified", "backend", "operation", "phase", "ordinal", "handle", "caller", "result", "stream", "flags"}

    def descendant(child, ancestor):
        while child is not None:
            child = parents.get(child)
            if child == ancestor:
                return True
        return False

    for row in rows:
        if not isinstance(row, dict) or set(row) != fields or row.get("schema") != 2 or row.get("qualified") is not False:
            raise ValueError("Wrong diagnostic SDK receipt role/schema")
        if type(row.get("pid")) is not int or row["pid"] <= 0 or row["pid"] != rows[0].get("pid"):
            raise ValueError("Changed original native process identity")
        if type(row.get("thread")) is not int or row["thread"] <= 0:
            raise ValueError("Missing original API thread identity")
        backend, operation, phase = (row.get(k) for k in ("backend", "operation", "phase"))
        if backend not in ("CUDA", "ROCm") or operation not in OPERATIONS or phase not in ("before", "after"):
            raise ValueError("Unsupported actual-program API receipt")
        if any(type(row.get(k)) is not int or not 0 <= row[k] <= (1 << 64) - 1 for k in FIELDS):
            raise ValueError("Missing host-only sample field")
        if row["monotonic_ns"] <= 0 or type(row.get("result")) is not int or type(row.get("ordinal")) is not int or row["ordinal"] <= 0:
            raise ValueError("Invalid original API ordinal/result")
        if not isinstance(row.get("caller"), str) or not row["caller"].startswith("0x"):
            raise ValueError("Missing original API caller identity")
        if not isinstance(row.get("stream"), str) or type(row.get("flags")) is not int or not 0 <= row["flags"] <= (1 << 64) - 1:
            raise ValueError("Missing original API stream/policy observation")
        if not isinstance(row.get("handle"), str):
            raise ValueError("Missing original API handle observation")
        if type(row.get("parent_ordinal")) is not int or row["parent_ordinal"] < 0:
            raise ValueError("Invalid original API parent ordinal")
        parent = None
        if row["parent_ordinal"]:
            parent = row["parent_backend"], row["parent_operation"], row["parent_ordinal"]
        elif row["parent_backend"] != "" or row["parent_operation"] != "":
            raise ValueError("Partial original API parent identity")
        ticket = backend, operation, row["ordinal"]
        stack = stacks.setdefault(row["thread"], [])
        if phase == "before":
            seen = ordinals.setdefault((backend, operation), set())
            if row["ordinal"] in seen:
                raise ValueError("Duplicated API observation")
            seen.add(row["ordinal"])
            if parent != (stack[-1] if stack else None) or (parent and parent not in pending):
                raise ValueError("Foreign or missing original API parent/thread")
            parents[ticket] = parent
            stack.append(ticket)
            pending[ticket] = row
            if operation in ("GraphInstantiate", "GraphInstantiateWithFlags", "GraphClone"):
                original = live.get((backend, "graph", row["handle"]))
                if original is None:
                    raise ValueError("Native graph consumer lacks its original live graph")
                inputs[ticket] = original
            if operation == "GraphLaunch" and (backend, "executable", row["handle"]) not in live:
                raise ValueError("Graph launch lacks its original live executable")
            if operation in DESTROYERS and (backend, DESTROYERS[operation], row["handle"]) not in live:
                raise ValueError("Native destroy lacks its original live handle")
            if operation == "DeviceReset" and any(k[0] == backend for k in live):
                raise ValueError("Native reset precedes actual program handle teardown")
        else:
            before = pending.get(ticket)
            if before is None or not stack or stack[-1] != ticket or parent != parents[ticket]:
                raise ValueError("Unpaired original API stack completion")
            if any(row[k] != before[k] for k in ("thread", "caller", "stream", "flags")) or row["monotonic_ns"] < before["monotonic_ns"]:
                raise ValueError("Unpaired/reordered native API completion")
            stack.pop()
            del pending[ticket]
            if row["result"]:
                if operation not in ("GraphInstantiate", "GraphInstantiateWithFlags") or row["ordinal"] != expected_instantiation_failure:
                    raise ValueError("Unexpected native API failure")
                injected.append(ticket)
            elif operation in CREATORS:
                if row["handle"] in ("(nil)", "0x0", "0", ""):
                    raise ValueError("Native success returned no original handle")
                handle = backend, CREATORS[operation], row["handle"]
                if handle in live:
                    # A real nested SDK call can publish the same result through
                    # its outer API. Only actual same-thread ancestry permits it.
                    if not descendant(live[handle], ticket):
                        raise ValueError("Native creator aliased an unrelated live handle")
                else:
                    births[CREATORS[operation]] = births.get(CREATORS[operation], 0) + 1
                live[handle] = ticket
                if operation == "StreamEndCapture":
                    capture = captures.pop((backend, row["stream"]), None)
                    if capture is None:
                        raise ValueError("Captured graph lacks its original successful capture begin")
                    lineage.append({"creator": ticket, "kind": "graph", "handle": row["handle"], "source_capture": capture, "stream": row["stream"]})
                elif operation in ("GraphClone", "GraphInstantiate", "GraphInstantiateWithFlags"):
                    lineage.append({"creator": ticket, "kind": CREATORS[operation], "handle": row["handle"], "source_graph": inputs[ticket]})
                elif operation == "GraphCreate":
                    lineage.append({"creator": ticket, "kind": "graph", "handle": row["handle"]})
            elif operation in DESTROYERS:
                if row["handle"] != before["handle"]:
                    raise ValueError("Native teardown changed its original handle")
                handle = backend, DESTROYERS[operation], row["handle"]
                if handle in live:
                    del live[handle]
                elif not any(value == handle and descendant(child, ticket) for child, value in destroyed.items()):
                    raise ValueError("Native teardown is not its original nested completion")
                destroyed[ticket] = handle
            elif row["handle"] != before["handle"]:
                raise ValueError("Native API changed its original non-creator handle")
            elif operation == "StreamBeginCapture":
                key = backend, row["stream"]
                if key in captures:
                    raise ValueError("Original stream capture began twice")
                captures[key] = ticket
            completed.append(row)
    if pending or live or captures or any(stacks.values()):
        raise ValueError("Incomplete original API pair or outstanding native program handle")
    for seen in ordinals.values():
        # Atomic assignment precedes recording; distinct threads may publish
        # ordinals in either order. Gaps remain invalid after the full receipt.
        if seen != set(range(1, len(seen) + 1)):
            raise ValueError("Skipped API observation")
    if bool(expected_instantiation_failure) != bool(injected) or len(injected) > 1:
        raise ValueError("Missing/duplicate original instantiation failure receipt")
    operations = {row["operation"] for row in completed}
    required = {"StreamBeginCapture", "StreamEndCapture", "GraphDestroy", "DeviceReset"}
    if not expected_instantiation_failure:
        required |= {"GraphLaunch", "GraphExecDestroy"}
    if not required <= operations or not operations.intersection({"GraphInstantiate", "GraphInstantiateWithFlags"}):
        raise ValueError("Actual captured program/reset coverage is incomplete")
    return {"qualified": False, "resource_limits_installed": False, "receipt_schema": 2,
            "observer_overhead_included": True, "creator_lineage_validated": True,
            "native_handle_creations": births, "creator_lineage": lineage, "api_calls": len(completed),
            "native_handle_teardown_observed": True, "validated_original_reset_receipt_required": True,
            "cold_warm_prefix_campaign_required": True,
            "host_sample_peaks": {k: max(row[k] for row in rows) for k in FIELDS if k != "monotonic_ns"}}


def validate_graph_inventory(lines):
    """Require a complete flat DAG snapshot inside every actual instantiate call.

    Native kinds are selected-SDK values recorded by the diagnostic. This closed
    lookup scope accepts only kernel/copy/set/empty/event nodes and cannot attest
    child, conditional, host callback or allocation graph families.
    """
    fields = {"schema", "pid", "thread", "backend", "operation", "ordinal", "graph", "observation",
              "nodes", "edges", "index", "node", "other", "native_kind", "error", "qualified"}
    pending, done = {}, []
    for line in lines:
        if line.startswith(PREFIX):
            row = json.loads(line[len(PREFIX):])
            if row["operation"] not in ("GraphInstantiate", "GraphInstantiateWithFlags"):
                continue
            ticket = row["backend"], row["operation"], row["ordinal"]
            if row["phase"] == "before":
                if ticket in pending:
                    raise ValueError("Duplicated original instantiate inventory")
                pending[ticket] = {"original": row, "state": "waiting", "nodes": {}, "edges": []}
            else:
                value = pending.pop(ticket, None)
                if value is None or value["state"] != "complete":
                    raise ValueError("Original instantiation lacks a complete graph inventory")
                done.append({"instantiate": ticket, "graph": value["original"]["handle"],
                             "nodes": len(value["nodes"]), "edges": len(value["edges"]),
                             "native_kind_counts": dict(Counter(value["nodes"].values())),
                             "maximum_nesting": 0, "child_graphs": 0, "default_edges_only": True,
                             "instantiate_result": row["result"]})
            continue
        if not line.startswith(INVENTORY_PREFIX):
            continue
        row = json.loads(line[len(INVENTORY_PREFIX):])
        if not isinstance(row, dict) or set(row) != fields or type(row["schema"]) is not int or row["schema"] != 1 or row["qualified"] is not False:
            raise ValueError("Invalid graph inventory schema/role")
        for name in ("pid", "thread", "ordinal", "nodes", "edges", "index"):
            if type(row[name]) is not int or row[name] < 0:
                raise ValueError("Invalid graph inventory numeric identity")
        if type(row["native_kind"]) is not int or any(not isinstance(row[name], str) for name in
                ("backend", "operation", "graph", "observation", "node", "other", "error")):
            raise ValueError("Invalid graph inventory native identity")
        ticket = row["backend"], row["operation"], row["ordinal"]
        value = pending.get(ticket)
        if value is None or any(row[name] != value["original"][name] for name in ("pid", "thread")) or row["graph"] != value["original"]["handle"]:
            raise ValueError("Graph inventory is outside its original instantiate/graph/thread")
        event = row["observation"]
        if event == "failed" or row["error"]:
            raise ValueError("Native flat graph inspection failed: " + row["error"])
        if event == "begin":
            if value["state"] != "waiting" or not 0 < row["nodes"] <= 65536 or not 0 <= row["edges"] <= 1048576:
                raise ValueError("Invalid or duplicate graph inventory beginning")
            if row["index"] != 0 or row["node"] != "(nil)" or row["other"] != "(nil)" or row["native_kind"] != -1:
                raise ValueError("Graph inventory boundary has unexpected node data")
            value.update(state="nodes", expected_nodes=row["nodes"], expected_edges=row["edges"])
            continue
        if value["state"] not in ("nodes", "edges") or (row["nodes"], row["edges"]) != (value["expected_nodes"], value["expected_edges"]):
            raise ValueError("Graph inventory extent or state changed")
        if event == "node":
            if value["state"] != "nodes" or row["index"] != len(value["nodes"]) or row["node"] in value["nodes"] or not row["node"].startswith("0x") or int(row["node"], 16) == 0:
                raise ValueError("Missing, duplicated or reordered graph node")
            if row["native_kind"] not in (0, 1, 2, 5, 6, 7):
                raise ValueError("Node kind is outside the closed flat lookup graph")
            if row["other"] != "(nil)":
                raise ValueError("Graph node has unexpected edge data")
            value["nodes"][row["node"]] = row["native_kind"]
        elif event == "edge":
            if len(value["nodes"]) != value["expected_nodes"] or row["index"] != len(value["edges"]):
                raise ValueError("Graph edges precede complete nodes or skip an index")
            edge = row["node"], row["other"]
            if any(node not in value["nodes"] for node in edge) or edge[0] == edge[1] or edge in value["edges"]:
                raise ValueError("Foreign, self or duplicate graph edge")
            if row["native_kind"] != -1:
                raise ValueError("Graph edge has unexpected node-kind data")
            value["state"] = "edges"
            value["edges"].append(edge)
        elif event == "complete":
            if row["index"] != 0 or row["node"] != "(nil)" or row["other"] != "(nil)" or row["native_kind"] != -1:
                raise ValueError("Graph inventory boundary has unexpected node data")
            if (len(value["nodes"]), len(value["edges"])) != (value["expected_nodes"], value["expected_edges"]):
                raise ValueError("Incomplete actual graph node/edge inventory")
            outgoing = {node: [] for node in value["nodes"]}
            degree = {node: 0 for node in value["nodes"]}
            for source, target in value["edges"]:
                outgoing[source].append(target)
                degree[target] += 1
            ready = deque(node for node, count in degree.items() if not count)
            visited = 0
            while ready:
                source = ready.popleft()
                visited += 1
                for target in outgoing[source]:
                    degree[target] -= 1
                    if not degree[target]:
                        ready.append(target)
            if visited != len(value["nodes"]):
                raise ValueError("Actual graph inventory contains a dependency cycle")
            value["state"] = "complete"
        else:
            raise ValueError("Unknown graph inventory observation")
    if pending or not done:
        raise ValueError("Missing or incomplete original instantiation inventory")
    return {"qualified": False, "resource_limits_installed": False, "flat_lookup_scope": True,
            "cold_sdk_queries_and_host_storage_included": True, "inventories": done}


def main():
    """Inspect preserved receipts; this command never opens a device or starts a child."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-log", type=Path, required=True)
    parser.add_argument("--expected-instantiation-failure", type=int, default=0)
    parser.add_argument("--require-graph-inventory", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_instantiation_failure < 0:
        parser.error("Invalid original instantiate failure ordinal")
    lines = args.api_log.read_text().splitlines()
    result = validate_records(parse_records(lines), args.expected_instantiation_failure)
    if args.require_graph_inventory:
        result["graph_inventory"] = validate_graph_inventory(lines)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
