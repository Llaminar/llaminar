#!/usr/bin/env python3
"""Device-free algebra, accepted-history ownership, and byte-gate regressions.

Passing these tests means that the research harness detects the known numeric
incompatibility and preserves its exact-arithmetic invariant. It does not mean
that the proposed verifier is byte equivalent or qualified for production.
"""

from dataclasses import replace
import unittest

import numpy as np

from harness import (Arithmetic, Comparison, History, LCG, Precision, Tree,
                     Node, Work, dense_step, dense_tree, leaf_evaluate, operands, run_case,
                     scalar_witness, topology)


class LeafEliminationReference(unittest.TestCase):
    """Exercise independent algebra, rejection, pending ownership, and flushing."""

    def test_exact_tree_system_and_every_endpoint_continuation(self):
        """Both algebraic verifiers agree exactly with independently built branch states."""
        for shape in ("chain", "star", "ternary", "irregular", "early_leaves"):
            for regime in ("ordinary", "zero", "endpoint", "parallel", "tiny"):
                with self.subTest(shape=shape, regime=regime):
                    result = run_case(("exact", shape, regime, 3, 2, 20261024, False, 4))
                    self.assertTrue(all(item["mismatches"] == 0 for item in result["comparisons"].values()))

    def test_adjust_every_query_variant(self):
        """The original exploratory all-output variant also has exact algebraic coverage."""
        result = run_case(("exact", "early_leaves", "zero", 3, 2, 20261025, True, 4))
        self.assertEqual(result["comparisons"]["continuation_outputs"]["mismatches"], 0)

    def test_projection_counts_exclude_leaf_corrections(self):
        """Count actual actions, including the old pending key and zero-child forest roots."""
        arithmetic, rng = Arithmetic(Precision.EXACT), LCG(123)
        state = rng.vector(arithmetic, 6).reshape(3, 2)
        first = operands(arithmetic, rng, (-1,), 3, 2, "ordinary")
        history = leaf_evaluate(arithmetic, History(state), first).commit(arithmetic, 0)
        for name, expected, rows in (("chain", 18, 9), ("star", 20, 1), ("ternary", 52, 13)):
            tree = operands(arithmetic, rng, topology(name, rng), 3, 2, "ordinary")
            evaluation = leaf_evaluate(arithmetic, history, tree)
            with self.subTest(shape=name):
                self.assertEqual(evaluation.work.projection_vectors, expected)
                self.assertEqual(len(evaluation.corrections) + len(evaluation.pending_records), rows)
                self.assertEqual(set(evaluation.corrections), tree.internal)

    def test_capacity_zero_and_small_buffers_survive_many_rounds(self):
        """Pending updates, long-to-short trees and prefix flushes preserve the accepted state."""
        arithmetic = Arithmetic(Precision.EXACT)
        for capacity in (0, 1, 3):
            rng = LCG(456)
            state = rng.vector(arithmetic, 6).reshape(3, 2)
            history = History(state.copy(), capacity=capacity)
            for round_index in range(12):
                parents = (-1, 0, 0, 2, 3) if round_index % 2 == 0 else (-1,)
                tree = operands(arithmetic, rng, parents, 3, 2, "zero")
                states, _ = dense_tree(arithmetic, state, tree)
                evaluated = leaf_evaluate(arithmetic, history, tree)
                endpoint = None if round_index % 3 == 1 else len(parents) - 1
                history = evaluated.commit(arithmetic, endpoint)
                if endpoint is not None:
                    state = states[endpoint]
                self.assertLessEqual(len(history.resolved), capacity)
                np.testing.assert_array_equal(state, history.materialize(arithmetic))
                if round_index % 3 == 2:
                    history = history.flush(arithmetic, Work())
                    self.assertIsNone(history.pending)
                    self.assertEqual(history.resolved, ())
                    np.testing.assert_array_equal(state, history.checkpoint)

    def test_rejection_and_empty_verification_preserve_pending_identity(self):
        """No accepted draft may replace state or silently discard a pending leaf."""
        arithmetic, rng = Arithmetic(Precision.FP32), LCG(789)
        state = rng.vector(arithmetic, 35).reshape(7, 5)
        tree = operands(arithmetic, rng, (-1,), 7, 5, "ordinary")
        history = leaf_evaluate(arithmetic, History(state), tree).commit(arithmetic, 0)
        for candidate in (Tree(()), tree):
            evaluation = leaf_evaluate(arithmetic, history, candidate)
            self.assertIs(evaluation.commit(arithmetic, None), history)
        empty = leaf_evaluate(arithmetic, history, Tree(()))
        self.assertEqual(empty.work.projection_vectors, 0)

    def test_accepted_raw_operands_own_their_storage(self):
        """Reusing candidate input storage cannot corrupt an accepted pending update."""
        arithmetic, rng = Arithmetic(Precision.EXACT), LCG(321)
        state = rng.vector(arithmetic, 6).reshape(3, 2)
        tree = operands(arithmetic, rng, (-1, -1), 3, 2, "ordinary")
        history = leaf_evaluate(arithmetic, History(state), tree).commit(arithmetic, 0)
        expected = history.materialize(arithmetic)
        for node in tree.nodes:
            node.k.fill(arithmetic.scalar(7))
            node.v.fill(arithmetic.scalar(9))
        np.testing.assert_array_equal(history.materialize(arithmetic), expected)
        work = Work()
        flushed = history.flush(arithmetic, work)
        self.assertEqual(work.flush_pending_resolutions, 1)
        np.testing.assert_array_equal(flushed.checkpoint, expected)

    def test_invalid_topology_and_acceptance_fail_closed(self):
        """Reject cycles, future parents, mixed geometry and out-of-range acceptance."""
        arithmetic, rng = Arithmetic(Precision.EXACT), LCG(1)
        node = operands(arithmetic, rng, (-1,), 3, 2, "ordinary").nodes[0]
        for parent in (-2, 0, 1):
            with self.assertRaises(ValueError):
                Tree((replace(node, parent=parent),))
        with self.assertRaises(ValueError):
            Tree((node, replace(node, v=node.v[:1])))
        with self.assertRaises(ValueError):
            History(arithmetic.zeros((3, 2)), capacity=-1)
        evaluation = leaf_evaluate(arithmetic, History(arithmetic.zeros((3, 2))), Tree((node,)))
        for endpoint in (-1, 1):
            with self.assertRaises(ValueError):
                evaluation.commit(arithmetic, endpoint)

    def test_minimal_float32_counterexample_is_not_a_tolerance_pass(self):
        """A real-arithmetic identity does not preserve the native FP32 program."""
        witness = scalar_witness()
        self.assertEqual(witness["exact"]["mismatches"], 0)
        self.assertEqual(witness["fp32"]["first"]["expected_bits"], "0x0")
        self.assertEqual(witness["fp32"]["first"]["actual_bits"], "0x33000000")
        for precision in ("fp32", "fp64"):
            result = run_case((precision, "early_leaves", "ordinary", 7, 5, 20261024, False, 4))
            for name in ("leaf_outputs", "accepted_states", "continuation_outputs"):
                self.assertGreater(result["comparisons"][name]["mismatches"], 0)

    def test_comparison_detects_signed_zero_and_rejects_nonfinite(self):
        """Byte accounting cannot quietly erase sign bits or pass a NaN comparison."""
        comparison = Comparison()
        comparison.observe(np.array([0], np.float32), np.array([-0.0], np.float32), "zero")
        self.assertEqual(comparison.mismatches, 1)
        self.assertEqual(comparison.max_abs, 0)
        with self.assertRaises(AssertionError):
            comparison.observe(np.array([np.nan], np.float32), np.array([np.nan], np.float32), "nan")

    def test_adjusted_query_alone_loses_serial_rounding_information(self):
        """Equal adjusted projections can correspond to different native serial outputs."""
        arithmetic = Arithmetic(Precision.FP32)
        one = np.ones(1, dtype=np.float32)
        node = Node(-1, one, one, np.array([2 ** -25], np.float32), np.float32(1), np.float32(1))
        adjusted = node.decay * (node.q - node.beta * arithmetic.dot(node.k, node.q) * node.k)
        zero, unit = np.zeros((1, 1), np.float32), np.ones((1, 1), np.float32)
        self.assertEqual(arithmetic.action(zero, adjusted).tobytes(), arithmetic.action(unit, adjusted).tobytes())
        self.assertNotEqual(dense_step(arithmetic, zero, node)[1].tobytes(), dense_step(arithmetic, unit, node)[1].tobytes())


if __name__ == "__main__":
    unittest.main()
