#!/usr/bin/env python3
"""Negative controls for the prepared NativeQuantised publication writer policy."""
from __future__ import annotations

import unittest

from check_native_quantised_publication_writers import ALLOWED_WRITERS, scan_source

OUTSIDE = "src/v2/execution/moe/Example.cpp"
PRODUCER = "src/v2/kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h"


class NativeQuantisedPublicationWriterPolicyTests(unittest.TestCase):
    """Every bypass of the generation-advancing writer must be reported."""

    def test_reads_are_allowed_everywhere(self):
        source = "void f(const P& p){auto*d=p.native_interleaved.data();auto n=p.native_interleaved.size();(void)d;(void)n;}"
        self.assertEqual(scan_source(OUTSIDE, source), [])

    def test_mutations_outside_named_producers_fail(self):
        for statement in ("auto s=p.native_interleaved.beginRewrite();",
                          "auto r=p.native_interleaved.retainedSlotRoot();",
                          "p.native_interleaved.replace(std::move(v));",
                          "p.native_interleaved.resize(n);",
                          "p.native_interleaved.resize_uninitialized(n);",
                          "p.native_interleaved.resize_uninitialized_aligned(n,64);",
                          "p.native_interleaved.reserve(n);",
                          "p.native_interleaved.release();"):
            with self.subTest(statement=statement):
                self.assertTrue(scan_source(OUTSIDE, "void f(P& p){" + statement + "}"))
                self.assertEqual(scan_source(PRODUCER, "void f(P& p){" + statement + "}"), [])

    def test_const_cast_of_published_bytes_fails_even_in_producers(self):
        for expression in ("const_cast<std::uint8_t*>(p.native_interleaved.data())",
                           "const_cast<uint8_t *>( view.interleavedBase() + 64 )"):
            with self.subTest(expression=expression):
                source = "void f(P& p){auto*w=" + expression + ";(void)w;}"
                self.assertTrue(scan_source(OUTSIDE, source))
                self.assertTrue(scan_source(PRODUCER, source))

    def test_comments_and_strings_are_not_code(self):
        source = '// p.native_interleaved.beginRewrite();\nconst char* s="p.native_interleaved.resize(1)";'
        self.assertEqual(scan_source(OUTSIDE, source), [])

    def test_allowlist_is_narrow(self):
        self.assertLessEqual(len(ALLOWED_WRITERS), 9)
        self.assertTrue(all(path.startswith("src/v2/") for path in ALLOWED_WRITERS))


if __name__ == "__main__":
    unittest.main()
