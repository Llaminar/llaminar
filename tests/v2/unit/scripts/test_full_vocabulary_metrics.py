#!/usr/bin/env python3
"""Device-free full-vocabulary observations, tie policy and legacy API proofs.

Fixtures are tiny authored logits, never model files or regenerated references.
Tests exercise the actual shared module and corpus compatibility wrapper; none
apply model acceptance limits or certify an inference backend.
"""
from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts/ci'))
import numpy as np
from full_vocabulary_metrics import full_vocabulary_metrics


class FullVocabularyMetricsTests(unittest.TestCase):
    """Verify one authority for finite full-tail FP64 observations and extents."""

    def metrics(self, reference, candidate, vocabulary=6):
        """Call the actual module with the fixture's independent complete extent."""
        return full_vocabulary_metrics(reference, candidate, vocabulary=vocabulary)

    def test_equal_distributions_and_reference_entropy(self):
        """Exact matching rows have zero errors and known uniform entropy."""
        result = self.metrics(np.zeros(6), np.zeros(6))
        self.assertEqual(result['kl_nats'], 0.)
        self.assertEqual(result['total_variation'], 0.)
        self.assertTrue(result['top1_same'])
        self.assertEqual(result['top5_overlap'], 1.)
        self.assertEqual(result['reference_top5_ids'], [0, 1, 2, 3, 4])
        self.assertEqual(result['candidate_top5_ids'], [0, 1, 2, 3, 4])
        self.assertAlmostEqual(result['reference_entropy_nats'], math.log(6))
        for key in ['reference_top1_logit_margin', 'reference_top1_probability_margin',
                    'reference_top1_probability_error', 'max_probability_error',
                    'mean_absolute_probability_error']:
            self.assertEqual(result[key], 0., key)

    def test_constant_shift_retains_distribution_metrics(self):
        """Independent row offsets do not create probability or ranking error."""
        p = np.array([4., 3., 2., 1., 0., -1.])
        q = np.array([3., 4., 2., 0., 1., -1.])
        a = self.metrics(p, q)
        b = self.metrics(p + 1024., q - 2048.)
        self.assertEqual(a, b)

    def test_same_argmax_can_hide_large_probability_error(self):
        """Correct top1 alone cannot stand in for a complete distribution check."""
        p = [math.log(9.), 0., -100., -100., -100., -100.]
        q = [math.log(1.01), 0., -100., -100., -100., -100.]
        result = self.metrics(p, q)
        self.assertTrue(result['top1_same'])
        self.assertGreater(result['kl_nats'], .3)
        self.assertGreater(result['total_variation'], .39)
        self.assertLess(result['reference_top1_probability_error'], -.39)
        self.assertGreater(result['max_probability_error'], .39)

    def test_near_tie_argmax_flip_has_tiny_distribution_drift(self):
        """A discontinuous rank flip can accompany negligible full-tail error."""
        eps = 1e-7
        result = self.metrics([eps, 0., -50., -50., -50., -50.],
                              [0., eps, -50., -50., -50., -50.])
        self.assertFalse(result['top1_same'])
        self.assertLess(result['kl_nats'], 1e-12)
        self.assertLess(result['total_variation'], 1e-6)
        self.assertEqual(result['reference_top1_logit_margin'], eps)
        self.assertGreater(result['reference_top1_probability_margin'], 0.)

    def test_exact_top5_intersection_and_cutoff_tie(self):
        """Lowest original token ID wins equal logits at the selected boundary."""
        result = self.metrics([7., 6., 5., 4., 3., 3.],
                              [7., 6., 5., 4., 2., 3.])
        self.assertEqual(result['reference_top5_ids'], [0, 1, 2, 3, 4])
        self.assertEqual(result['candidate_top5_ids'], [0, 1, 2, 3, 5])
        self.assertEqual(result['top5_overlap'], .8)
        self.assertEqual(result['top5_width'], 5)
        # Same set under a different order retains full intersection evidence.
        reordered = self.metrics([7., 6., 5., 4., 3., 0.],
                                 [3., 4., 5., 6., 7., 0.])
        self.assertFalse(reordered['top1_same'])
        self.assertEqual(reordered['top5_overlap'], 1.)

    def test_extreme_finite_logits_and_representational_underflow(self):
        """No probability floor or tail cut is needed for large finite ranges."""
        large = float(np.finfo(np.float32).max)
        p = np.array([large, 0., -large, -1., -2., -3.])
        q = np.array([0., large, -large, -1., -2., -3.])
        same = self.metrics(p, p)
        self.assertEqual(same['kl_nats'], 0.)
        self.assertEqual(same['reference_entropy_nats'], 0.)
        different = self.metrics(p, q)
        self.assertEqual(different['kl_nats'], large)
        self.assertEqual(different['total_variation'], 1.)
        self.assertFalse(different['top1_same'])
        self.assertTrue(all(math.isfinite(different[k]) for k in
                            ['kl_nats', 'total_variation', 'reference_entropy_nats',
                             'reference_top1_logit_margin', 'max_probability_error']))

    def test_positive_subthreshold_reference_tail_is_not_trimmed(self):
        """Retain a real contribution below the old short-helper1e-10 cutoff."""
        p = np.array([0., -25., -50., -60., -70., -80.])
        q = np.array([0., -50., -50., -60., -70., -80.])
        result = self.metrics(p, q)
        def log_probabilities(x):
            """Closed fixture equation, not a second production scorer."""
            return x - np.log(np.exp(x).sum())
        lp, lq = log_probabilities(p), log_probabilities(q)
        expected = float(np.dot(np.exp(lp), lp - lq))
        self.assertGreater(result['kl_nats'], 1e-10)
        self.assertAlmostEqual(result['kl_nats'], expected, places=20)

    def test_declared_extent_rejects_even_equally_truncated_rows(self):
        """Matching shortened payloads cannot manufacture a full-vocab result."""
        for p, q in [(np.zeros(5), np.zeros(5)), (np.zeros(6), np.zeros(5)),
                     (np.zeros((1, 6)), np.zeros(6)), (np.zeros(6), np.zeros((6, 1))),
                     (np.array(0.), np.array(0.)), ([], [])]:
            with self.subTest(reference_shape=np.shape(p), candidate_shape=np.shape(q)):
                with self.assertRaises(ValueError):
                    self.metrics(p, q)
        for vocabulary in [True, False, 1, 0, -1, 6., '6', None]:
            with self.subTest(vocabulary=vocabulary):
                with self.assertRaises(ValueError):
                    self.metrics(np.zeros(6), np.zeros(6), vocabulary=vocabulary)

    def test_invalid_type_nonfinite_and_masked_rows_fail_on_either_side(self):
        """No coercion from invalid evidence or repair of either original row."""
        malformed = [np.zeros(6, dtype=bool), np.zeros(6, dtype=complex),
                     ['0'] * 6, np.array([0.] * 6, dtype=object),
                     np.ma.array(np.zeros(6), mask=[0, 0, 0, 0, 0, 1])]
        for value in [math.nan, math.inf, -math.inf]:
            row = np.zeros(6)
            row[5] = value
            malformed.append(row)
        for bad in malformed:
            for p, q in [(bad, np.zeros(6)), (np.zeros(6), bad)]:
                with self.subTest(dtype=str(getattr(bad, 'dtype', type(bad)))):
                    with self.assertRaises(ValueError):
                        self.metrics(p, q)

    def test_unrepresentable_fp64_range_fails_explicitly(self):
        """Finite inputs cannot legitimize an infinite/undefined metric result."""
        large = float(np.finfo(np.float64).max)
        with self.assertRaisesRegex(ValueError, 'FP64'):
            self.metrics([large, -large, 0., 1., 2., 3.], np.zeros(6))

    def test_legacy_canary_api_values_and_optional_diagnostics_compatible(self):
        """The long-canary's original three keys retain the original equation."""
        import model_kld_corpus as corpus
        p = np.array([0., 1., 2.])
        q = np.array([2., 0., 1.])
        result = corpus.row_metrics(p, q)
        lp = (p - p.max()) - np.log(np.exp(p - p.max()).sum())
        lq = (q - q.max()) - np.log(np.exp(q - q.max()).sum())
        self.assertEqual(result['kl_nats'], float(np.dot(np.exp(lp), lp - lq)))
        self.assertEqual(result['total_variation'], float(np.abs(np.exp(lp) - np.exp(lq)).sum() / 2))
        self.assertEqual(result['top1_same'], bool(np.argmax(p) == np.argmax(q)))
        self.assertEqual(result['top5_width'], 3)
        self.assertIsNone(result['top5_overlap'])
        self.assertEqual(result, corpus.row_metrics(p, q, vocabulary=3))
        with self.assertRaises(ValueError):
            corpus.row_metrics(p, q, vocabulary=6)


if __name__ == '__main__':
    unittest.main()
