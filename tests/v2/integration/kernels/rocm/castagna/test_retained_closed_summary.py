"""Device-free tests for the MI50 three-session Release checkpoint decision."""
import unittest

from summarize_retained_closed import validate_checkpoint, validate_route_work, validate_source_window


def session(owner=70.0, delta=-1.0, pci="0000:97:00.0"):
    """Return the measurements consumed by the checkpoint after native log validation."""
    return {"retained_median_us": owner, "paired_retained_minus_castagna_median_us": delta, "pci": pci}


class RetainedClosedCheckpoint(unittest.TestCase):
    def test_uneven_original_shards_and_source_format(self):
        for degree, member, routed, shared in (
            (2, 0, (0, 256), (0, 320)), (2, 1, (256, 384), (320, 320)),
            (4, 3, (384, 256), (448, 192)),
        ):
            with self.subTest(degree=degree, member=member):
                device = dict(H="2560", F=str(routed[1]), S=str(shared[1]), E="512", topK="10")
                source = dict(real_gguf="1", generated="0", layer="2", tp=str(degree), participant=str(member),
                              routed_first=str(routed[0]), routed_count=str(routed[1]), shared_first=str(shared[0]),
                              shared_count=str(shared[1]), gate="Q5_K", down="Q8_0")
                encodings = ("Q5_K", "Q8_0")
                validate_source_window(device, source, 2, degree, member, encodings)
                for replacement in ({"gate": "Q4_K"}, {"down": "Q5_1"}, {"routed_first": "1"}, {"shared_count": "160"}):
                    with self.assertRaisesRegex(ValueError, "GGUF weights or slice"):
                        validate_source_window(device, source | replacement, 2, degree, member, encodings)

    def test_replay_only_drift_is_visible_and_cannot_excuse_unmatched_timed_work(self):
        replay = [dict(history=str(i), selected_set_intersection="10", top_k="10") for i in range(20)]
        timing = [dict(record=str((i + 5) % 4), intersection="10", top_k="10") for i in range(37)]
        replay[16]["selected_set_intersection"] = "9"
        self.assertEqual(validate_route_work(replay, timing, False), [16])
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            validate_route_work(replay, timing, True)
        for field, value in (("record", "3"), ("intersection", "9")):
            changed = [timing[0] | {field: value}, *timing[1:]]
            with self.assertRaisesRegex(ValueError, "scheduled"):
                validate_route_work(replay, changed, False)
        replay[2]["selected_set_intersection"] = "9"
        with self.assertRaisesRegex(ValueError, "scheduled"):
            validate_route_work(replay, timing, False)

    def test_inclusive_cutoff(self):
        validate_checkpoint([session(69.8), session(), session(69.9)])

    def test_one_slow_session_fails_even_if_pooled_median_passes(self):
        with self.assertRaisesRegex(ValueError, "session 3"):
            validate_checkpoint([session(69.8), session(69.8), session(70.001)])

    def test_paired_tie_fails(self):
        with self.assertRaisesRegex(ValueError, "beat paired Castagna"):
            validate_checkpoint([session(), session(delta=0.0), session()])

    def test_same_device_and_three_fresh_sessions_required(self):
        for records in ([session()] * 2, [session(), session(), session(pci="0000:49:00.0")]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                validate_checkpoint(records)


if __name__ == "__main__":
    unittest.main()
