"""Device-free regressions for overlap accounting and trace admission."""

import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import attribute_nsight as attribution


class AttributionTests(unittest.TestCase):
    """Small exact intervals catch inflated sums and discarded boundary work."""

    def setUp(self):
        self.rules = attribution.compile_rules([
            {"category": "Projection", "pattern": "mat"},
            {"category": "Attention", "pattern": "attn"}])
        self.geometry = (2, 1, 1, 32, 1, 1)

    def kernel(self, start, end, name="mat", geometry=None, device=0):
        return attribution.Kernel(start, end, name, geometry or self.geometry, device)

    def test_same_and_cross_category_overlap_conserve_time(self):
        rows = [self.kernel(0, 10), self.kernel(2, 8), self.kernel(5, 12, "attn")]
        sums, unions, exclusive, calls, _, _ = attribution.measure(
            rows, 0, 15, {"mat": "Projection", "attn": "Attention"})
        self.assertEqual(sums["Projection"], 16)
        self.assertEqual(unions["Projection"], 10)
        self.assertEqual(calls["Projection"], 2)
        self.assertEqual(exclusive["Projection"], 5)
        self.assertEqual(exclusive[attribution.OVERLAP], 5)
        self.assertEqual(exclusive["Attention"], 2)
        self.assertEqual(exclusive[attribution.NO_KERNEL], 3)
        self.assertEqual(sum(exclusive.values()), 15)

    def test_boundary_clipping_and_ties(self):
        rows = [self.kernel(0, 10), self.kernel(10, 20, "attn")]
        _, unions, exclusive, _, _, _ = attribution.measure(
            rows, 5, 15, {"mat": "Projection", "attn": "Attention"})
        self.assertEqual(dict(unions), {"Projection": 5, "Attention": 5})
        self.assertEqual(exclusive[attribution.OVERLAP], 0)
        self.assertEqual(sum(exclusive.values()), 10)

    def test_empty_window_is_no_kernel_not_host_tax(self):
        result = attribution.measure([], 10, 30, {})
        self.assertEqual(result[2][attribution.NO_KERNEL], 20)

    def test_long_crossing_kernel_is_not_lost_and_samples_are_averaged(self):
        rows = [self.kernel(0, 1000), self.kernel(10, 20, "attn")]
        summaries, categories, _ = attribution.analyze(
            rows, {"decode": [(500, 600), (700, 900)]}, self.rules)
        self.assertEqual(summaries["decode"]["mean_window_ms"], 150 / 1e6)
        projection = next(row for row in categories if row["category"] == "Projection")
        self.assertEqual(projection["exclusive_ms"], 150 / 1e6)
        self.assertEqual(projection["calls_per_sample"], 1)

    def test_geometry_is_not_merged(self):
        rows = [self.kernel(0, 10), self.kernel(10, 20, geometry=(4, 1, 1, 64, 1, 1))]
        _, _, physical = attribution.analyze(rows, {"decode": [(0, 20)]}, self.rules)
        self.assertEqual(len(physical), 2)
        self.assertEqual({r["blockX"] for r in physical}, {32, 64})

    def test_ordered_classification(self):
        rules = attribution.compile_rules([
            {"category": "Head", "pattern": "MAT.*head"},
            {"category": "Projection", "pattern": "mat"}])
        self.assertEqual(attribution.classify("mat_head", rules), "Head")
        self.assertEqual(attribution.classify("unseen", rules), attribution.UNKNOWN)

    def test_invalid_bounds_and_duplicate_samples(self):
        for windows in ([], [{"phase": "x", "start_ns": 2, "end_ns": 1}],
                        [{"phase": "x", "start_ns": 0.5, "end_ns": 1}],
                        [{"phase": "x", "start_ns": 0, "end_ns": 10}] * 2):
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                attribution.validate_windows(windows)

    def test_multiple_devices_require_selection(self):
        rows = [self.kernel(0, 10), self.kernel(0, 10, device=1)]
        with self.assertRaisesRegex(ValueError, "multiple GPUs"):
            attribution.analyze(rows, {"decode": [(0, 10)]}, self.rules)
        summaries, _, physical = attribution.analyze(rows, {"decode": [(0, 10)]}, self.rules, 1)
        self.assertEqual(summaries["decode"]["devices"], [1])
        self.assertEqual(len(physical), 1)

    def test_sqlite_cli_preserves_unknown_evidence_before_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.sqlite"
            with sqlite3.connect(trace) as db:
                db.execute("CREATE TABLE StringIds(id INTEGER, value TEXT)")
                db.execute("INSERT INTO StringIds VALUES(1, 'unknown_kernel')")
                db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,"
                           "demangledName INTEGER,gridX INTEGER,gridY INTEGER,gridZ INTEGER,"
                           "blockX INTEGER,blockY INTEGER,blockZ INTEGER,deviceId INTEGER)")
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(0,10,1,2,1,1,32,1,1,0)")
            self.assertEqual(attribution.read_kernels(trace)[0].name, "unknown_kernel")
            windows, rules = root / "windows.json", root / "rules.json"
            windows.write_text(json.dumps([{"phase": "decode", "start_ns": 0, "end_ns": 20}]))
            rules.write_text(json.dumps([{"category": "Projection", "pattern": "mat"}]))
            with contextlib.redirect_stdout(io.StringIO()):
                result = attribution.main([
                    "--sqlite", str(trace), "--windows", str(windows), "--rules", str(rules),
                    "--output-prefix", str(root / "result")])
            self.assertEqual(result, 1)
            self.assertIn("unknown_kernel", (root / "result-kernels.csv").read_text())
            self.assertEqual(json.loads((root / "result.json").read_text())["unclassified"],
                             ["unknown_kernel"])


if __name__ == "__main__":
    unittest.main()
