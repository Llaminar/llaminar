"""Device-free checks for the kernel scaling report's measurement semantics."""

import sys
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "performance" / "kernels"))
from plot_kernel_shard_scaling import Disposition, prioritize, render_svg, summarize


def samples(times=(400, 200, 100), disposition="partitioned"):
    """Build three complete timing distributions with one slow outlier each."""
    return [dict(operation="projection <Q/K>", workload="main & tail", degree=str(degree),
                 sample=str(index), latency_us=str(time * factor), disposition=disposition)
            for degree, time in zip((1, 2, 4), times)
            for index, factor in enumerate((1, 1, 10))]


class KernelShardScalingTest(unittest.TestCase):
    """Do not confuse per-doubling speedup, replication or missing measurements."""

    def test_median_ratios_and_svg(self):
        row, = summarize(samples())
        self.assertEqual((row.one_us, row.two_us, row.four_us), (400, 200, 100))
        self.assertEqual((row.one_to_two, row.two_to_four), (2, 2))
        self.assertEqual((row.tune_two, row.tune_four), (False, False))
        self.assertEqual(row.sample_counts, (3, 3, 3))
        ET.fromstring(render_svg([row]))
        self.assertIn("&lt;Q/K&gt;", render_svg([row]))

    def test_second_doubling_is_independent(self):
        row, = summarize(samples((400, 200, 160)))
        self.assertFalse(row.tune_two)
        self.assertTrue(row.tune_four)
        self.assertEqual(row.two_to_four, 1.25)

    def test_target_is_inclusive(self):
        row, = summarize(samples((361, 190, 100)))
        self.assertFalse(row.tune_two)
        self.assertFalse(row.tune_four)

    def test_replicated_and_communication_are_not_half_work(self):
        for disposition in ("replicated", "collective"):
            row, = summarize(samples((100, 150, 200), disposition))
            self.assertEqual(row.disposition, Disposition(disposition))
            self.assertFalse(row.tune_two)
            self.assertFalse(row.tune_four)

    def test_missing_degree_cannot_pass(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            summarize(samples()[:6])

    def test_duplicate_identity_cannot_bias_median(self):
        rows = samples()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize(rows + [rows[0]])

    def test_bad_values_fail_closed(self):
        for bad in ("0", "-1", "nan", "inf"):
            rows = samples()
            rows[0]["latency_us"] = bad
            with self.assertRaises(ValueError):
                summarize(rows)
        for bad in (0, 1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                summarize(samples(), bad)

    def test_disposition_cannot_change_across_degrees(self):
        rows = samples()
        rows[3]["disposition"] = "replicated"
        with self.assertRaisesRegex(ValueError, "conflicting"):
            summarize(rows)

    def test_rank_uses_absolute_time_and_occurrences_not_ratio(self):
        tiny = samples((2, 2, 2))
        large = samples((400, 300, 200))
        for row in tiny:
            row["operation"] = "tiny"
        for row in large:
            row["operation"] = "large"
            row["calls"] = "30"
        rows = prioritize(summarize(tiny + large), 2)
        self.assertEqual(rows[0].operation, "large")
        self.assertEqual(rows[0].excess_us(2), 3000)
        self.assertEqual(rows[0].excess_us(4), 1500)
        for disposition in ("replicated", "collective"):
            row, = summarize(samples((100, 1000, 2000), disposition))
            self.assertEqual(row.excess_us(2), 0)

    def test_occurrence_counts_are_consistent_and_positive(self):
        for count in ("0", "-1", "2"):
            rows = samples()
            rows[0]["calls"] = count
            with self.assertRaises(ValueError):
                summarize(rows)
        with self.assertRaises(ValueError):
            prioritize(summarize(samples()), 3)


if __name__ == "__main__":
    unittest.main()
