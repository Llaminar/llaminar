#!/usr/bin/env python3
"""Exact-rational checks for full and prefix-bounded backward-query algebra.

These independent dense recurrence comparisons authenticate the proposed
identity and every prefix stopping point. Native FP32 qualification is a
separate check and deliberately records rounding counterexamples.
"""
import unittest
import numpy as np
from backward import backward_reference
from harness import Arithmetic, LCG, Precision, dense_tree, operands, topology


class BackwardQueryReference(unittest.TestCase):
    """Prove algebra for branching, zero/endpoint gates and every prefix cut."""
    def test_exact_backward_and_combined(self):
        """All contractions equal dense recurrence in exact arithmetic."""
        arithmetic=Arithmetic(Precision.EXACT)
        for shape in ('empty','singleton','chain','star','ternary','irregular','early_leaves'):
            for regime in ('ordinary','zero','endpoint','parallel','tiny'):
                rng=LCG(20261024)
                root=rng.vector(arithmetic,6).reshape(3,2)
                tree=operands(arithmetic,rng,topology(shape,rng),3,2,regime)
                states,expected=dense_tree(arithmetic,root,tree)
                for cutoff in range(max((len(p) for p in tree.paths),default=0)+1):
                    retained={i:states[i] for i in tree.internal if len(tree.paths[i])<=cutoff}
                    with self.subTest(shape=shape,regime=regime,cutoff=cutoff):
                        actual=backward_reference(arithmetic,root,tree,retained)
                        for lhs,rhs in zip(expected,actual,strict=True):
                            np.testing.assert_array_equal(lhs,rhs)


if __name__=='__main__':
    unittest.main()
