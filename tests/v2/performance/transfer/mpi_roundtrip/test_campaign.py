#!/usr/bin/env python3
"""Device-free functional tests of evidence admission and comparison cells."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from run_campaign import order_for_round, result_from_log, summarize


def observation(candidate="mapped", size=16384, root=0, mean=20.0):
    """Construct one explicit validated native observation for parser tests."""
    return dict(candidate=candidate, bytes=size, root=root, samples=100,
                mean_us=mean, p50_us=mean, p95_us=mean + 1, p99_us=mean + 2, correct=True)


class EvidenceContracts(unittest.TestCase):
    """Incomplete, failed, nonfinite and differently scoped evidence cannot win."""

    def test_order_rotates_and_reverses(self):
        candidates = ["mapped", "registered", "device"]
        orders = [order_for_round(candidates, r) for r in range(6)]
        self.assertEqual(len({tuple(o) for o in orders}), 6)
        self.assertTrue(all(set(o) == set(candidates) for o in orders))
        self.assertEqual(candidates, ["mapped", "registered", "device"])

    def test_accepts_exact_terminal_row(self):
        row = observation()
        self.assertEqual(result_from_log("native log\n" + json.dumps(row), "mapped", 16384, 0), row)

    def test_rejects_partial_or_wrong_results(self):
        row = observation()
        for log in ("", json.dumps(row) + "\n" + json.dumps(row)):
            with self.assertRaises(ValueError):
                result_from_log(log, "mapped", 16384, 0)
        for key, value in (("candidate", "device"), ("bytes", 8192), ("root", 1),
                           ("correct", False), ("mean_us", float("nan")), ("p99_us", float("inf")),
                           ("samples", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                result_from_log(json.dumps(dict(row, **{key: value})), "mapped", 16384, 0)

    def test_keeps_directions_and_shapes_separate(self):
        rows = [observation(candidate=c, size=size, root=root, mean=base)
                for size in (16384, 1048576) for root in (0, 1)
                for c, base in (("mapped", 20.0), ("device", 30.0)) for _ in range(3)]
        cells = summarize(rows)
        self.assertEqual(len(cells), 4)
        self.assertTrue(all(c["lowest_median"] == "mapped" and c["trial_ranges_separated"] for c in cells))
        with self.assertRaises(ValueError):
            summarize(rows[:-1])
        with self.assertRaises(ValueError):
            summarize(rows[:-3])

    def test_missing_transport_admission_fails_before_launch(self):
        for candidate, diagnostic in (("device", "no silently staged candidate"),
                                      ("mapped-dmabuf", "require explicit --verbs-rails"),
                                      ("write-combined-verbs", "require explicit --verbs-rails")):
            with self.subTest(candidate=candidate), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "must-not-exist"
                command = [sys.executable, str(Path(__file__).with_name("run_campaign.py")),
                           "--mpi", "/not-executed", "--binary", "/not-read", "--hosts", "a:1,b:1",
                           "--rsh-agent", "/not-executed", "--session-owner", "/not-executed", "--daemon", "/not-executed",
                           "--net-devices", "test:1", "--library-path", "/not-used",
                           "--candidates", candidate, "--output", str(output)]
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)
                self.assertFalse(output.exists())

    def test_overlapping_ranges_are_not_a_clear_win(self):
        rows = [observation("mapped", mean=19), observation("mapped", mean=23),
                observation("device", mean=20), observation("device", mean=24)]
        self.assertFalse(summarize(rows)[0]["trial_ranges_separated"])


if __name__ == "__main__":
    unittest.main()
