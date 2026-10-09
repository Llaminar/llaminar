#!/usr/bin/env python3
"""Device-free proofs that context selection consumes PMA outcomes faithfully.

Sweep every capacity boundary in small synthetic geometries, including model
maxima which are not cache-block aligned. Unknown errors and changed workload
budgets may never be converted into successful smaller-context evidence.
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests/v2/e2e/server'))
from opencode_context_admission import AdmissionOutcome, ContextPolicy, select_context


class ContextAdmissionTests(unittest.TestCase):
    """The runtime owns every capacity decision; the search only schedules probes."""

    def test_complete_capacity_and_alignment_sweep(self):
        for maximum in range(17, 65):
            for alignment in (1, 2, 4, 8):
                for capacity in range(9, maximum + 1):
                    candidates = []
                    def probe(value):
                        candidates.append(value)
                        return AdmissionOutcome.ACCEPTED if value <= capacity else AdmissionOutcome.CAPACITY_REJECTED
                    policy = ContextPolicy(maximum, alignment, 8)
                    minimum = (9 + alignment - 1) // alignment * alignment
                    with self.subTest(maximum=maximum, alignment=alignment, capacity=capacity):
                        if capacity < minimum:
                            with self.assertRaisesRegex(ValueError, 'No admitted context'):
                                select_context(policy, probe)
                        else:
                            result = select_context(policy, probe)
                            expected = maximum if capacity == maximum else capacity // alignment * alignment
                            self.assertEqual(result['context_tokens'], expected)
                            self.assertEqual(result['output_tokens'], 8)
                            self.assertEqual(result['model_maximum_admitted'], capacity == maximum)
                        self.assertEqual(candidates[0], maximum)
                        self.assertEqual(len(candidates), len(set(candidates)))
                        self.assertTrue(all(8 < value <= maximum for value in candidates))

    def test_full_model_context_needs_only_one_probe(self):
        calls = []
        def probe(value):
            calls.append(value)
            return AdmissionOutcome.ACCEPTED
        result = select_context(ContextPolicy(262144, 64), probe)
        self.assertEqual(calls, [262144])
        self.assertEqual(result['context_tokens'], 262144)
        self.assertIsNone(result['first_rejected_context'])

    def test_unknown_failures_do_not_select_a_smaller_context(self):
        calls = []
        def probe(value):
            calls.append(value)
            raise RuntimeError('native planning failure')
        with self.assertRaisesRegex(RuntimeError, 'native planning failure'):
            select_context(ContextPolicy(262144, 64), probe)
        self.assertEqual(calls, [262144])
        for invalid in (True, False, 'accepted', None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'typed admission'):
                select_context(ContextPolicy(262144, 64), lambda _: invalid)

    def test_geometry_and_output_policy_are_explicit(self):
        for values in ((True, 64, 32), (262144, 0, 32), (262144, 64, 0),
                       (32768, 64, 32768), (2**31, 64, 32768), (64.0, 8, 16)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ContextPolicy(*values)


if __name__ == '__main__':
    unittest.main()
