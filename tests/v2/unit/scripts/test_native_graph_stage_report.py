#!/usr/bin/env python3
"""Device-free proofs for GPU stage attribution, overlap and missing evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "performance" / "kernels"))
from native_graph_stage_report import interval_union_ms, summarize_graph, write_report


def trace_fixture() -> dict:
    """Retain overlapping branches, an inclusive scope and unowned plumbing."""
    stages = [
        {"id": 100, "parent": -1, "name": "block", "type": "SUBGRAPH", "native_node_ids": [0, 1]},
        {"id": 101, "parent": 100, "name": "norm", "type": "RMSNORM", "native_node_ids": [0]},
        {"id": 102, "parent": 100, "name": "projection", "type": "MATMUL", "native_node_ids": [1]},
        {"id": 103, "parent": -1, "name": "no_op", "type": "COPY", "native_node_ids": []},
    ]
    nodes = []
    for node_id, stage, begin, end in [(0, stages[1], 0.0, 4.0), (1, stages[2], 3.0, 7.0)]:
        nodes.append({"id": node_id, "name": "kernel", "instrumented": True,
                      "start_ms": begin, "end_ms": end,
                      "stage": {key: stage[key] for key in ("id", "name", "type")}})
    nodes.append({"id": 2, "name": "memset", "instrumented": True, "start_ms": 7.0, "end_ms": 8.0})
    return {"source": "native_cuda_graph_events", "diagnostic_only": True,
            "stage_attribution": "canonical_capture_scopes", "snapshot": "last_completed_replay",
            "launches": 2, "device": 1, "graph": 7, "duration_ms": 8.0, "nodes": nodes, "stages": stages}


class NativeGraphStageReportTest(unittest.TestCase):
    """Protect the interpretation of genuine GPU event evidence."""

    def test_overlap_and_inclusive_scopes_do_not_invent_elapsed_time(self) -> None:
        self.assertEqual(interval_union_ms([(3.0, 7.0), (0.0, 4.0), (7.0, 8.0)]), 8.0)
        graph = summarize_graph(trace_fixture())
        rows = {row["id"]: row for row in graph["stages"]}
        self.assertEqual(graph["native_node_sum_ms"], 9.0)
        self.assertEqual(graph["native_node_union_ms"], 8.0)
        self.assertEqual(rows[100]["gpu_span_ms"], 7.0)
        self.assertEqual(rows[100]["gpu_covered_ms"], 7.0)
        self.assertEqual(rows[100]["inclusive_node_ms"], 8.0)
        self.assertEqual(rows[100]["owned_node_ms"], 0.0)
        self.assertEqual(sum(row["owned_node_ms"] for row in rows.values()), 8.0)
        self.assertEqual(graph["unattributed_nodes"], 1)
        self.assertEqual(graph["unattributed_node_ms"], 1.0)
        self.assertEqual(rows[103]["measurement"], "empty_capture_scope")

    def test_unmeasured_nodes_are_partial_and_conditionals_stay_opaque(self) -> None:
        trace = trace_fixture()
        trace["nodes"][0]["instrumented"] = False
        del trace["nodes"][0]["start_ms"], trace["nodes"][0]["end_ms"]
        trace["nodes"][1]["name"] = "conditional_body_opaque"
        rows = {row["id"]: row for row in summarize_graph(trace)["stages"]}
        self.assertEqual(rows[101]["measurement"], "partial")
        self.assertIsNone(rows[101]["gpu_span_ms"])
        self.assertIsNone(rows[101]["gpu_covered_ms"])
        self.assertEqual(rows[100]["measurement"], "partial")
        self.assertEqual(rows[102]["opaque_conditional_nodes"], 1)

    def test_missing_intervals_and_stale_ownership_fail(self) -> None:
        mutations = [
            lambda t: t["nodes"][0].pop("end_ms"),
            lambda t: t["nodes"][0]["stage"].update(name="stale"),
            lambda t: t["stages"][1].update(native_node_ids=[]),
            lambda t: t["stages"][0].update(native_node_ids=[0, 0]),
            lambda t: t["stages"][0].update(parent=101),
            lambda t: t["stages"][0].update(parent=999),
            lambda t: t["nodes"][0].update(end_ms=10.0),
            lambda t: t.update(stage_attribution="unavailable"),
            lambda t: t.update(launches=0),
        ]
        for mutation in mutations:
            trace = trace_fixture()
            mutation(trace)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summarize_graph(trace)

    def test_standalone_artifacts_bind_source_and_escape_stage_names(self) -> None:
        trace = trace_fixture()
        trace["stages"][1]["name"] = "</script><script>unexpected()</script>"
        trace["nodes"][0]["stage"]["name"] = trace["stages"][1]["name"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cuda-event-1-7.json"
            source.write_text(json.dumps(trace))
            output = root / "report"
            report = write_report([source], output)
            self.assertEqual(len(report["graphs"][0]["trace_sha256"]), 64)
            self.assertEqual(json.loads((output / "stage-timing.json").read_text()), report)
            self.assertNotIn(trace["stages"][1]["name"], (output / "stage-timing.html").read_text())
            self.assertIn("gpu_span_ms", (output / "stage-timing.csv").read_text())
            with self.assertRaises(ValueError):
                write_report([], output)


if __name__ == "__main__":
    unittest.main()
