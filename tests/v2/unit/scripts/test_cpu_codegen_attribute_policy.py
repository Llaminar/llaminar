#!/usr/bin/env python3
"""Negative and positive controls for check_cpu_codegen_attribute_policy.py."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_cpu_codegen_attribute_policy as policy  # noqa: E402


def messages(text: str) -> list[str]:
    return [violation.message for violation in policy.scan_text("src/v2/x.h", text)]


class CodegenAttributePolicy(unittest.TestCase):
    def test_admits_exact_and_register_pressure_attributes(self) -> None:
        for text in (
            '#define E __attribute__((optimize("fp-contract=off", "no-associative-math")))',
            '#define C __attribute__((optimize("fp-contract=off", "no-associative-math", "no-thread-jumps")))',
            '#define U \\\n    __attribute__((optimize("fp-contract=off", "no-associative-math", "no-schedule-insns")))',
            '__attribute__((always_inline, optimize("fp-contract=off", "no-associative-math"))) void f();',
            '[[gnu::noinline, gnu::optimize("no-tree-slp-vectorize")]] void g();',
            '[[gnu::optimize("fp-contract=off,no-associative-math,no-schedule-insns")]] void h();',
            '__attribute__((__optimize__("-ffp-contract=off", "-fno-associative-math"))) void i();',
            '// __attribute__((optimize("fast-math"))) is a comment\nint x;',
            '/* #pragma GCC optimize("O3") */ const char *s = "target(\\"avx2\\")";',
            '_Pragma("clang fp contract(off)") _Pragma("clang fp reassociate(off)")',
            'auto t = binding.target().resolve(); const Config target(main);',
            '[[nodiscard]] [[maybe_unused]] int j();',
        ):
            with self.subTest(text=text):
                self.assertEqual(messages(text), [])

    def test_rejects_fp_semantic_options(self) -> None:
        for option in ("fast-math", "-ffast-math", "fp-contract=fast", "fp-contract=on", "associative-math",
                       "reciprocal-math", "finite-math-only", "unsafe-math-optimizations", "no-signed-zeros",
                       "no-trapping-math", "Ofast", "O3"):
            with self.subTest(option=option):
                found = messages(f'__attribute__((optimize("fp-contract=off", "no-associative-math", "{option}"))) void f();')
                self.assertEqual(len(found), 1, found)
                self.assertIn("FP-semantic", found[0])

    def test_rejects_unadmitted_options(self) -> None:
        for text in ('__attribute__((optimize("unroll-loops"))) void f();',
                     '[[gnu::optimize("no-tree-vectorize")]] void f();',
                     '__attribute__((optimize(2))) void f();',
                     '#define OPTS "fp-contract=off"\n__attribute__((optimize(OPTS))) void f();'):
            with self.subTest(text=text):
                self.assertEqual(len(messages(text)), 1, messages(text))

    def test_register_pressure_tokens_require_the_exact_pair(self) -> None:
        for text in ('__attribute__((optimize("no-schedule-insns"))) void f();',
                     '__attribute__((optimize("fp-contract=off", "no-thread-jumps"))) void f();',
                     '[[gnu::optimize("no-associative-math", "no-schedule-insns")]] void f();'):
            with self.subTest(text=text):
                found = messages(text)
                self.assertEqual(len(found), 1, found)
                self.assertIn("require fp-contract=off and no-associative-math", found[0])

    def test_rejects_target_attributes_and_pragmas(self) -> None:
        for text in ('__attribute__((target("avx2"))) void f();',
                     '__attribute__((target_clones("default", "avx2"))) void f();',
                     '[[gnu::target("arch=armv8.2-a+dotprod")]] void f();',
                     '__attribute__((__target__("sse4.2"))) void f();',
                     '#pragma GCC optimize("O3")',
                     '#pragma GCC target("avx2")',
                     '#  pragma GCC push_options',
                     '_Pragma("GCC optimize(\\"fast-math\\")")',
                     '#pragma clang optimize off',
                     '#pragma clang fp contract(fast)',
                     '#pragma STDC FP_CONTRACT ON',
                     '_Pragma("clang fp reassociate(on)")'):
            with self.subTest(text=text):
                self.assertGreaterEqual(len(messages(text)), 1, text)

    def test_tree_scan_reports_paths_lines_and_requires_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "src/v2/kernels"
            source.mkdir(parents=True)
            (source / "ok.h").write_text('#define E __attribute__((optimize("fp-contract=off", "no-associative-math")))\n')
            self.assertEqual(policy.main(["--repo-root", root]), 0)
            (source / "bad.cpp").write_text('int a;\n\n__attribute__((optimize("fast-math"))) void f();\n')
            violations, files, attributes = policy.scan(Path(root))
            self.assertEqual((files, attributes), (2, 2))
            self.assertEqual([(v.path, v.line) for v in violations], [("src/v2/kernels/bad.cpp", 3)])
            self.assertEqual(policy.main(["--repo-root", root]), 1)
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "src/v2").mkdir(parents=True)
            (Path(root) / "src/v2/a.h").write_text("int a;\n")
            self.assertEqual(policy.main(["--repo-root", root]), 2)

    def test_repository_tree_is_clean(self) -> None:
        root = Path(__file__).resolve().parents[4]
        violations, files, attributes = policy.scan(root)
        self.assertGreater(files, 0)
        self.assertGreater(attributes, 0)
        self.assertEqual([str(v) for v in violations], [])


if __name__ == "__main__":
    unittest.main()
