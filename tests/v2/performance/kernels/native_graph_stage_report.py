#!/usr/bin/env python3
"""Render captured stage GPU-event intervals without double-counting overlap.

The input is a standalone native observer receipt, never CPU launch timing.
Stage spans include scheduling and diagnostic event overhead. Inclusive scopes
may contain nested stages; physical-node ownership remains exclusive. Missing
events and opaque conditional bodies stay explicit rather than acquiring an
invented zero-duration measurement. Each graph shows its last completed replay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def interval_union_ms(intervals: list[tuple[float, float]]) -> float:
    """Measure covered device-clock time, counting overlapping intervals once."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0.0
    for next_start, next_end in ordered[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + end - start


def summarize_graph(trace: dict[str, Any]) -> dict[str, Any]:
    """Validate exact node ownership and retain both elapsed and overlapping cost."""
    if (trace.get("source") != "native_cuda_graph_events" or
            trace.get("stage_attribution") != "canonical_capture_scopes" or
            trace.get("snapshot") != "last_completed_replay" or
            not trace.get("diagnostic_only") or int(trace.get("launches", 0)) < 1):
        raise ValueError("A completed native CUDA trace with canonical stage attribution is required")
    duration = float(trace["duration_ms"])
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Invalid graph GPU duration")
    nodes = {int(node["id"]): node for node in trace["nodes"]}
    if len(nodes) != len(trace["nodes"]):
        raise ValueError("Duplicate native node identity")
    stages = {int(stage["id"]): stage for stage in trace["stages"]}
    if len(stages) != len(trace["stages"]) or not stages:
        raise ValueError("Missing or duplicate captured stage identity")
    intervals: dict[int, tuple[float, float]] = {}
    for node_id, node in nodes.items():
        if node.get("instrumented"):
            if "start_ms" not in node or "end_ms" not in node:
                raise ValueError("Instrumented native node has no completed GPU interval")
            start, end = float(node["start_ms"]), float(node["end_ms"])
            if not all(map(math.isfinite, (start, end))) or start < 0 or end < start or end > duration + 0.002:
                raise ValueError("Native node GPU interval lies outside its completed parent")
            intervals[node_id] = start, end
        if "stage" in node:
            owner = node["stage"]
            definition = stages.get(int(owner["id"]))
            if definition is None or any(owner[key] != definition[key] for key in ("name", "type")):
                raise ValueError("Native node has stale stage ownership")
            if node_id not in definition["native_node_ids"]:
                raise ValueError("Native node is absent from its owner's capture scope")
    rows = []
    for stage_id, stage in stages.items():
        ids = [int(node_id) for node_id in stage["native_node_ids"]]
        if len(set(ids)) != len(ids) or any(node_id not in nodes for node_id in ids):
            raise ValueError("Capture scope contains duplicate or missing native nodes")
        parent = int(stage["parent"])
        visited = {stage_id}
        while parent != -1:
            if parent in visited or parent not in stages:
                raise ValueError("Capture scope hierarchy is cyclic or incomplete")
            visited.add(parent)
            parent = int(stages[parent]["parent"])
        measured = [intervals[node_id] for node_id in ids if node_id in intervals]
        owned = [node_id for node_id in ids if nodes[node_id].get("stage", {}).get("id") == stage_id]
        begin = min((value[0] for value in measured), default=None)
        end = max((value[1] for value in measured), default=None)
        complete = len(measured) == len(ids)
        rows.append({
            "id": stage_id, "name": stage["name"], "type": stage["type"],
            "parent": stage["parent"], "native_nodes": len(ids),
            "owned_nodes": len(owned), "measured_nodes": len(measured),
            "measurement": "empty_capture_scope" if not ids else "complete" if complete else "partial",
            "start_ms": begin, "end_ms": end,
            "gpu_span_ms": (end - begin if measured else 0.0 if not ids else None),
            "gpu_covered_ms": interval_union_ms(measured) if measured or not ids else None,
            "inclusive_node_ms": sum(stop - start for start, stop in measured),
            "owned_node_ms": sum(intervals[node_id][1] - intervals[node_id][0]
                                 for node_id in owned if node_id in intervals),
            "opaque_conditional_nodes": sum(nodes[node_id]["name"] == "conditional_body_opaque" for node_id in ids),
        })
    unknown = [node_id for node_id, node in nodes.items() if "stage" not in node]
    return {
        "device": trace["device"], "graph": trace["graph"], "launches": trace["launches"],
        "snapshot": trace["snapshot"], "duration_ms": duration,
        "timed_native_nodes": len(intervals), "native_nodes": len(nodes),
        "attributed_nodes": len(nodes) - len(unknown),
        "unattributed_nodes": len(unknown),
        "unattributed_node_ms": sum(intervals[node_id][1] - intervals[node_id][0]
                                    for node_id in unknown if node_id in intervals),
        "native_node_union_ms": interval_union_ms(list(intervals.values())),
        "native_node_sum_ms": sum(end - start for start, end in intervals.values()),
        "stages": sorted(rows, key=lambda row: (row["start_ms"] is None, row["start_ms"] or 0, row["id"])),
    }


HTML = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Captured CUDA stage timing</title><style>
body{font:14px system-ui,sans-serif;margin:24px;color:#202636;background:#f7f9fc}
h1{font-size:24px}p{max-width:1000px;line-height:1.5}select,input{padding:7px;margin:6px}
table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;padding:7px;border-bottom:1px solid #e4e8ee}
th{position:sticky;top:0;background:#eaf0f7;cursor:pointer}.num{text-align:right;font-variant-numeric:tabular-nums}
.track{height:12px;background:#eef1f6;position:relative;min-width:170px}.bar{height:100%;position:absolute;background:#3476cc}
.partial{color:#a43b00}.muted{color:#566175}#summary{padding:12px;background:white;margin:12px 0}
</style><h1>Captured CUDA stage timing</h1>
<p>GPU events measure the last completed replay of each retained graph. Stage spans include scheduling and event overhead.
Covered time counts overlap once within a stage. Node sums can overlap across branches and nested stages and must not be added to obtain graph elapsed time.
Conditional bodies are opaque. Use an uninstrumented run for throughput.</p>
<label>Graph <select id="graph"></select></label><label>Stage <input id="filter" placeholder="Name or type"></label>
<label><input id="empty" type="checkbox">Show empty scopes</label><div id="summary"></div>
<table><thead><tr><th data-key="name">Stage</th><th data-key="type">Type</th><th>GPU interval</th>
<th data-key="gpu_span_ms">Span ms</th><th data-key="gpu_covered_ms">Covered ms</th>
<th data-key="owned_node_ms">Owned node ms</th><th>Nodes</th><th>Evidence</th></tr></thead><tbody id="rows"></tbody></table>
<script id="data" type="application/json">PAYLOAD</script><script>
const report=JSON.parse(document.getElementById('data').textContent),selector=document.getElementById('graph');
for(let i=0;i<report.graphs.length;i++){const g=report.graphs[i],o=document.createElement('option');o.value=i;o.textContent=`CUDA ${g.device}, graph ${g.graph}, ${g.duration_ms.toFixed(3)} ms`;selector.append(o)}
let sortKey='start_ms',descending=false;
const fmt=x=>x===null?'unmeasured':x.toFixed(3);
function cell(row,text,cls=''){const td=document.createElement('td');td.textContent=text;td.className=cls;row.append(td);return td}
function render(){const g=report.graphs[+selector.value],q=document.getElementById('filter').value.toLowerCase(),showEmpty=document.getElementById('empty').checked;
document.getElementById('summary').textContent=`Graph elapsed ${fmt(g.duration_ms)} ms · ${g.stages.length} captured stages · ${g.attributed_nodes}/${g.native_nodes} attributed nodes · ${g.unattributed_nodes} graph plumbing/unattributed nodes · ${g.launches} launches (last replay shown)`;
const body=document.getElementById('rows');body.replaceChildren();let rows=g.stages.filter(s=>(showEmpty||s.native_nodes)&&(`${s.name} ${s.type}`.toLowerCase().includes(q)));
rows.sort((a,b)=>{let x=a[sortKey],y=b[sortKey];if(x===null)return 1;if(y===null)return -1;return (typeof x==='string'?x.localeCompare(y):x-y)*(descending?-1:1)});
for(const s of rows){const tr=document.createElement('tr');cell(tr,s.name);cell(tr,s.type);const td=cell(tr,'');const track=document.createElement('div');track.className='track';
if(s.start_ms!==null&&g.duration_ms){const bar=document.createElement('div');bar.className='bar';bar.style.left=`${100*s.start_ms/g.duration_ms}%`;bar.style.width=`${100*s.gpu_span_ms/g.duration_ms}%`;bar.title=`${fmt(s.start_ms)}–${fmt(s.end_ms)} ms`;track.append(bar)}td.append(track);
cell(tr,fmt(s.gpu_span_ms),'num');cell(tr,fmt(s.gpu_covered_ms),'num');cell(tr,fmt(s.owned_node_ms),'num');cell(tr,`${s.measured_nodes}/${s.native_nodes}`,'num');
cell(tr,s.measurement+(s.opaque_conditional_nodes?` · ${s.opaque_conditional_nodes} opaque conditional`:'')+(s.parent!==-1?' · nested':''),s.measurement==='partial'?'partial':'muted');body.append(tr)}}
selector.onchange=render;document.getElementById('filter').oninput=render;document.getElementById('empty').onchange=render;
document.querySelectorAll('th[data-key]').forEach(th=>th.onclick=()=>{descending=sortKey===th.dataset.key?!descending:true;sortKey=th.dataset.key;render()});render();
</script></html>"""


def write_report(paths: list[Path], output: Path) -> dict[str, Any]:
    """Write an auditable JSON/CSV report and a self-contained interactive view."""
    graphs = []
    for path in paths:
        graph = summarize_graph(json.loads(path.read_text()))
        graph["trace_path"] = str(path.resolve())
        graph["trace_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        graphs.append(graph)
    if not graphs:
        raise ValueError("No completed native stage traces found")
    report = {"schema_version": 1, "diagnostic_only": True,
              "intervals_include_scheduling_and_event_overhead": True,
              "conditional_bodies_timed_as_opaque": True, "graphs": graphs}
    output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2)
    (output / "stage-timing.json").write_text(payload + "\n")
    (output / "stage-timing.html").write_text(HTML.replace("PAYLOAD", payload.replace("<", "\\u003c")))
    with (output / "stage-timing.csv").open("w", newline="") as handle:
        fields = ["device", "graph", *graphs[0]["stages"][0].keys()]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for graph in graphs:
            for stage in graph["stages"]:
                writer.writerow({"device": graph["device"], "graph": graph["graph"], **stage})
    return report


def main() -> None:
    """Resolve an exact trace cohort; missing evidence fails the diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = write_report(sorted(args.trace_dir.glob("cuda-event-*.json")), args.output)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({"graphs": len(report["graphs"]), "report": str(args.output / "stage-timing.html")}))


if __name__ == "__main__":
    main()
