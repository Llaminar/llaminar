"""Device-free regression for comparing verifier work and calibrated service rates."""

import unittest

from verifier_roofline import analyze


def evidence():
    device = dict(device="test-card", isa="gfx906", clock_policy="locked")
    read = dict(resource="memory", unit="KiB", counter="FetchSize", regime="streaming")
    dot = dict(resource="dot", unit="operations", counter="dot4", regime="registers")
    references = [
        dict(**device, **read, id="memory", samples=[{"count": 60, "us": 10}]),
        dict(**device, **dot, id="dot", samples=[{"count": 30, "us": 2}]),
    ]
    leaf = dict(**device, case="Q5", role="gate", symbol="kernel", variant="control",
                timing_us=[8, 10, 12], work=[dict(**read, reference="memory", count=36),
                                              dict(**dot, reference="dot", count=75)])
    return dict(references=references, leaves=[leaf])


class VerifierRooflineTest(unittest.TestCase):
    def test_overlapping_resources_and_dispersion(self):
        rows, (leaf,) = analyze(evidence())
        self.assertEqual([r["reference_us"] for r in rows], [6, 5])
        self.assertEqual(leaf["limiting_measured_resource"], "memory")
        self.assertEqual(leaf["fraction_of_reference"], 0.6)
        self.assertEqual((leaf["measured_us_p25"], leaf["measured_us_p75"]), (9, 11))

    def test_rate_exceeded_remains_visible(self):
        document = evidence()
        document["leaves"][0]["timing_us"] = [4]
        self.assertEqual(analyze(document)[1][0]["fraction_of_reference"], 1.5)

    def test_zero_measured_fetch_retains_the_other_bottleneck(self):
        document = evidence()
        document["leaves"][0]["work"][0]["count"] = 0
        rows, (leaf,) = analyze(document)
        self.assertEqual(rows[0]["reference_us"], 0)
        self.assertEqual(leaf["limiting_measured_resource"], "dot")

    def test_calibration_must_match_the_work(self):
        for field in ("device", "isa", "clock_policy", "unit", "counter", "regime"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Mismatched"):
                document = evidence()
                document["references"][0][field] = "other"
                analyze(document)

    def test_missing_or_duplicated_measurements_fail(self):
        for change in ("missing_timing", "empty_calibration", "duplicate_resource", "bad_count"):
            with self.subTest(change=change), self.assertRaises(ValueError):
                document = evidence()
                leaf = document["leaves"][0]
                if change == "missing_timing":
                    leaf["timing_us"].clear()
                elif change == "empty_calibration":
                    document["references"][0]["samples"].clear()
                elif change == "duplicate_resource":
                    leaf["work"].append(leaf["work"][0])
                else:
                    leaf["work"][0]["count"] = float("nan")
                analyze(document)


if __name__ == "__main__":
    unittest.main()
