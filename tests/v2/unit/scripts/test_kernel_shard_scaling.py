"""Device-free cumulative target and actual-participant reduction regressions.

These synthetic timing records test report semantics only. They neither execute
kernels nor certify source/precision/complete-operation provenance, which remains
owned by the collector. Actual TP cohorts must retain every rank/paired round.
"""

import sys
import csv
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "performance" / "kernels"))
from plot_kernel_shard_scaling import (
    Disposition, MeasurementScope, TargetStatus, main, prioritize, render_svg, summarize,
)


def samples(times=(400, 200, 100), disposition="partitioned"):
    """Build local-shape distributions with a slow outlier, not an actual TP run."""
    return [dict(operation="projection <Q/K>", workload="main & tail", degree=str(degree),
                 sample=str(index), latency_us=str(time * factor), disposition=disposition)
            for degree, time in zip((1, 2, 4), times)
            for index, factor in enumerate((1, 1, 10))]


def actual_samples(times=(400, 200, 100), disposition="partitioned"):
    """Retain all ranks at each degree and identical sample rounds, no fake zeros."""
    return [dict(row, scope="actual_tp", rank=str(rank)) for row in samples(times, disposition)
            for rank in range(int(row["degree"]))]


class KernelShardScalingTest(unittest.TestCase):
    """Targets are independent cumulative1→2/1→4;2→4 remains descriptive."""

    def test_median_ratios_absolute_latency_and_svg(self):
        row, = summarize(samples())
        self.assertEqual((row.one_us, row.two_us, row.four_us), (400, 200, 100))
        self.assertEqual((row.one_to_two, row.one_to_four, row.two_to_four), (2, 4, 2))
        self.assertEqual((row.tune_two, row.tune_four), (False, False))
        self.assertEqual(row.sample_counts, (3, 3, 3))
        self.assertEqual(row.scope, MeasurementScope.LOCAL_SHAPE)
        ET.fromstring(render_svg([row]))
        self.assertIn("&lt;Q/K&gt;", render_svg([row]))
        self.assertIn("descriptive", render_svg([row]))
        self.assertIn("local_shape", render_svg([row]))

    def test_tp2_passes_tp4_fails_independently(self):
        row, = summarize(samples((360, 180, 101)))
        self.assertEqual(row.tp2_status, TargetStatus.MET)
        self.assertEqual(row.tp4_status, TargetStatus.MISSED)
        self.assertFalse(row.tune_two)
        self.assertTrue(row.tune_four)

    def test_tp2_fails_tp4_passes_independently(self):
        row, = summarize(samples((360, 200, 100)))
        self.assertTrue(row.tune_two)
        self.assertFalse(row.tune_four)
        self.assertEqual(row.one_to_four, 3.6)
        self.assertEqual(row.two_to_four, 2)

    def test_targets_are_inclusive(self):
        row, = summarize(samples((666, 360, 185)))
        self.assertEqual(row.one_to_two, 1.85)
        self.assertEqual(row.one_to_four, 3.6)
        self.assertEqual((row.tp2_status, row.tp4_status), (TargetStatus.MET, TargetStatus.MET))
        row, = summarize(samples((400, 200, 100)), 2, 4)
        self.assertEqual((row.tp2_status, row.tp4_status), (TargetStatus.MET, TargetStatus.MET))

    def test_two_to_four_not_a_gate(self):
        row, = summarize(samples((400, 100, 100)))
        self.assertEqual(row.two_to_four, 1)
        self.assertFalse(row.tune_two)
        self.assertFalse(row.tune_four)

    def test_local_replicated_collective_are_unassessed_never_green(self):
        for disposition in ("replicated", "collective"):
            row, = summarize(samples((100, 150, 200), disposition))
            self.assertEqual(row.disposition, Disposition(disposition))
            self.assertEqual((row.tp2_status, row.tp4_status),
                             (TargetStatus.NOT_APPLICABLE, TargetStatus.NOT_APPLICABLE))
            self.assertEqual((row.one_us, row.two_us, row.four_us), (100, 150, 200))
            self.assertNotEqual(row.tp2_status, TargetStatus.MET)
            self.assertNotEqual(row.tp4_status, TargetStatus.MET)

    def test_actual_replicated_collective_keep_target_misses(self):
        for disposition in ("replicated", "collective"):
            row, = summarize(actual_samples((100, 150, 200), disposition))
            self.assertEqual(row.scope, MeasurementScope.ACTUAL_TP)
            self.assertEqual(row.disposition, Disposition(disposition))
            self.assertEqual((row.tp2_status, row.tp4_status), (TargetStatus.MISSED, TargetStatus.MISSED))
            self.assertEqual((row.tune_two, row.tune_four), (True, True))

    def test_actual_slowest_rank_each_round_precedes_median(self):
        data = []
        # TP2 per-rank medians are both1, but each complete round's max is100.
        values = {1: [[400], [400], [400]], 2: [[100, 1], [1, 100], [1, 1]],
                  4: [[80, 1, 1, 1], [1, 80, 1, 1], [1, 1, 80, 1]]}
        for degree, rounds in values.items():
            for sample, ranks in enumerate(rounds):
                for rank, latency in enumerate(ranks):
                    data.append(dict(operation="op", workload="retained", scope="actual_tp",
                        rank=str(rank), degree=str(degree), sample=str(sample), latency_us=str(latency),
                        disposition="partitioned"))
        row, = summarize(data)
        self.assertEqual((row.one_us, row.two_us, row.four_us), (400, 100, 80))
        self.assertEqual(row.reduced_round_samples[1], ((0, 100), (1, 100), (2, 1)))
        self.assertEqual((row.one_to_two, row.one_to_four, row.two_to_four), (4, 5, 1.25))
        self.assertEqual(row.sample_counts, (3, 3, 3))

    def test_actual_missing_duplicate_foreign_ranks_fail(self):
        data = actual_samples()
        for mutate in (lambda x: x[:-1], lambda x: x + [x[-1]],
                       lambda x: [dict(x[0], rank="1"), *x[1:]],
                       lambda x: [dict(x[0], rank="-1"), *x[1:]],
                       lambda x: [dict(x[0], rank=""), *x[1:]]):
            with self.assertRaises(ValueError):
                summarize(mutate(data))

    def test_actual_missing_round_even_when_remaining_ranks_complete_fails(self):
        data = [r for r in actual_samples() if not (r["degree"] == "4" and r["sample"] == "2")]
        with self.assertRaisesRegex(ValueError, "paired round"):
            summarize(data)

    def test_local_rank_or_mixed_scope_rejected(self):
        data = samples()
        for extra in ({"rank": "0"}, {"scope": "actual_tp", "rank": "0"}, {"scope": "unknown"}):
            with self.assertRaises(ValueError):
                summarize([*data[1:], dict(data[0], **extra)])

    def test_missing_degree_and_duplicate_sample_cannot_pass(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            summarize(samples()[:6])
        data = samples()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize(data + [data[0]])

    def test_bad_values_and_targets_fail_closed(self):
        for factory in (samples, actual_samples):
            for bad in ("0", "-1", "nan", "inf", "-inf"):
                data = factory(); data[0]["latency_us"] = bad
                with self.assertRaises(ValueError):
                    summarize(data)
        for bad in (0, 1, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                summarize(samples(), bad, 3.6)
            with self.assertRaises(ValueError):
                summarize(samples(), 1.85, bad)
        with self.assertRaises(ValueError):
            summarize([])

    def test_semantics_cannot_change_across_degrees(self):
        data = samples(); data[3]["disposition"] = "replicated"
        with self.assertRaisesRegex(ValueError, "conflicting"):
            summarize(data)

    def test_rank_uses_cumulative_absolute_time_and_occurrences_not_ratio(self):
        tiny = samples((2, 2, 2)); large = samples((400, 300, 200))
        for row in tiny: row["operation"] = "tiny"
        for row in large: row.update(operation="large", calls="30")
        rows = prioritize(summarize(tiny + large), 2)
        self.assertEqual(rows[0].operation, "large")
        self.assertEqual(rows[0].excess_us(2), 3000)
        self.assertEqual(rows[0].excess_us(4), 3000)
        for disposition in ("replicated", "collective"):
            row, = summarize(actual_samples((100, 1000, 2000), disposition))
            self.assertEqual(row.excess_us(2), 0)

    def test_cli_defaults_and_raw_samples_and_csv_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "samples.csv"; output = root / "report"
            data = actual_samples((360, 180, 101), "collective")
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
            with patch.object(sys, "argv", ["scaling", "--samples", str(source), "--output-prefix", str(output)]):
                main()
            result = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(result["targets_relative_to_tp1"], {"2": 1.85, "4": 3.6})
            self.assertEqual(result["raw_samples"], data)
            self.assertEqual(result["rows"][0]["scope"], "actual_tp")
            with output.with_suffix(".csv").open(newline="") as handle:
                row, = list(csv.DictReader(handle))
            self.assertEqual((row["scope"], row["disposition"], row["tp2_status"], row["tp4_status"]),
                             ("actual_tp", "collective", "met", "missed"))
            ET.fromstring(output.with_suffix(".svg").read_text())

    def test_retired_per_doubling_cli_is_not_silently_reinterpreted(self):
        with patch.object(sys, "argv", ["scaling", "--samples", "unused", "--output-prefix", "unused", "--threshold", "1.9"]):
            with self.assertRaises(SystemExit) as failure:
                main()
        self.assertEqual(failure.exception.code, 2)

    def test_occurrences_consistent_positive_and_ranking_degree_valid(self):
        for count in ("0", "-1", "2"):
            data = samples(); data[0]["calls"] = count
            with self.assertRaises(ValueError): summarize(data)
        with self.assertRaises(ValueError): prioritize(summarize(samples()), 3)


if __name__ == "__main__":
    unittest.main()
