"""Device-free tests for the MI50 three-session Release checkpoint decision."""
import unittest

from summarize_retained_closed import validate_checkpoint


def session(owner=70.0, delta=-1.0, pci="0000:97:00.0"):
    """Return the measurements consumed by the checkpoint after native log validation."""
    return {"retained_median_us": owner, "paired_retained_minus_castagna_median_us": delta, "pci": pci}


class RetainedClosedCheckpoint(unittest.TestCase):
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
