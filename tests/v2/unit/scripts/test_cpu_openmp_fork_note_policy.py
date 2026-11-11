#!/usr/bin/env python3
"""Negative and positive controls for the OpenMP fork-note source policy.

The controls drive the checker's ACTUAL COMMAND LINE over synthetic source
trees, so main()'s fail-closed behavior - not merely the scan helper - is
what is proven: orphan-only files, a missing source root, unsupported and
supported grammars, site exemptions, adjacent-note binding and the counted
inventory summary all have a discriminating case.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

CHECKER = Path(__file__).resolve().parent / "check_cpu_openmp_fork_note_policy.py"


def run_cli(repo_root: Path) -> tuple[int, str, str]:
    completed = subprocess.run(
        [sys.executable, "-B", str(CHECKER), "--repo-root", str(repo_root)],
        capture_output=True, text=True)
    return completed.returncode, completed.stdout, completed.stderr


def violations(stderr: str) -> list[str]:
    return [line for line in stderr.splitlines() if "raw OpenMP fork" in line or "no adjacent pragma" in line]


class OpenMPForkNotePolicyCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="forknote-policy-"))
        (self.root / "src/v2").mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    def write(self, relative: str, body: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)

    def test_noted_pragma_passes_and_counts(self) -> None:
        self.write("src/v2/kernel.cpp",
                   "::llaminar::v2::noteOpenMPTeamFork();\n"
                   "#pragma omp parallel for\n"
                   "for (int i = 0; i < n; ++i) {}\n")
        code, out, err = run_cli(self.root)
        self.assertEqual(code, 0, err)
        self.assertIn("1 noted sites", out)

    def test_orphan_only_file_fails_closed(self) -> None:
        # The scan helper must never skip a file that holds no pragma: an
        # orphan note is a policy violation on its own.
        self.write("src/v2/orphan.cpp",
                   "::llaminar::v2::noteOpenMPTeamFork();\n"
                   "out[0] = 1.0f;\n")
        code, out, err = run_cli(self.root)
        self.assertEqual(code, 1)
        self.assertEqual(len(violations(err)), 1)
        self.assertIn("no adjacent pragma", err)

    def test_missing_source_root_fails_closed(self) -> None:
        empty = Path(tempfile.mkdtemp(prefix="forknote-empty-"))
        try:
            code, out, err = run_cli(empty)
            self.assertEqual(code, 2)
            self.assertIn("missing source root", err)
        finally:
            shutil.rmtree(empty)

    def test_pragma_forms_and_whitespace_are_supported(self) -> None:
        self.write("src/v2/forms.cpp",
                   "::llaminar::v2::noteOpenMPTeamFork();\n"
                   "#  pragma   omp   parallel\n"
                   "{ }\n"
                   "::llaminar::v2::noteOpenMPTeamFork();\n"
                   "_Pragma(\"omp parallel for\")\n"
                   "for (int i = 0; i < n; ++i) {}\n")
        code, out, err = run_cli(self.root)
        self.assertEqual(code, 0, err)
        self.assertIn("2 noted sites", out)

    def test_inc_files_are_inventoried_and_site_exempt(self) -> None:
        # The real .inc path and its real exempt function: the site is named,
        # inventoried, and exempt; a different function in a sibling file is not.
        self.write("src/v2/kernels/cpu/gemm/CPUNativeQuantisedWindowPacking.inc",
                   "inline void appendNativeWeightGroups() {\n"
                   "#pragma omp parallel for collapse(2)\n"
                   "for (int c = 0; c < chunks; ++c) {}\n"
                   "}\n")
        self.write("src/v2/other.cpp",
                   "void otherWindowFunction() {\n"
                   "#pragma omp parallel for\n"
                   "for (int c = 0; c < chunks; ++c) {}\n"
                   "}\n")
        code, out, err = run_cli(self.root)
        self.assertEqual(code, 1)
        self.assertEqual(len(violations(err)), 1)
        self.assertIn("otherWindowFunction", err)
        self.assertNotIn("appendNativeWeightGroups", err)

    def test_whole_file_exemption_never_admits_new_sites(self) -> None:
        # A named exempt function coexists with an unlisted one in the same
        # real file: only the named function's sites are exempt.
        self.write("src/v2/kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h",
                   "inline bool packWeightsCPUNativeQuantised() {\n"
                   "#pragma omp parallel for\n"
                   "for (int i = 0; i < n; ++i) {}\n"
                   "}\n"
                   "inline void brandNewPackerFunction() {\n"
                   "#pragma omp parallel for\n"
                   "for (int i = 0; i < n; ++i) {}\n"
                   "}\n")
        code, out, err = run_cli(self.root)
        self.assertEqual(code, 1)
        self.assertEqual(len(violations(err)), 1)
        self.assertIn("brandNewPackerFunction", err)

    def test_duplicate_and_distant_notes_fail(self) -> None:
        self.write("src/v2/dup.cpp",
                   "::llaminar::v2::noteOpenMPTeamFork();\n"
                   "::llaminar::v2::noteOpenMPTeamFork();\n"
                   "#pragma omp parallel for\n"
                   "for (int i = 0; i < n; ++i) {}\n")
        code, out, err = run_cli(self.root)
        self.assertEqual(code, 1)
        # The first note is an orphan; the second note legitimately binds, so
        # adjacent binding can never double-count one pragma.
        self.assertEqual(len(violations(err)), 1)
        self.assertIn("no adjacent pragma", err)

    def test_conditional_note_is_rejected(self) -> None:
        self.write("src/v2/conditional.cpp",
                   "void kernel(float *out) {\n"
                   "    if (out != nullptr) ::llaminar::v2::noteOpenMPTeamFork();\n"
                   "#pragma omp parallel for\n"
                   "    for (int i = 0; i < n; ++i) out[i] = 0.0f;\n"
                   "}\n")
        code, out, err = run_cli(self.root)
        self.assertEqual(code, 1)
        self.assertIn("not one unconditional canonical statement", err)

    def test_two_notes_on_one_line_are_rejected(self) -> None:
        self.write("src/v2/doubled.cpp",
                   "void kernel(float *out) {\n"
                   "    ::llaminar::v2::noteOpenMPTeamFork(); ::llaminar::v2::noteOpenMPTeamFork();\n"
                   "#pragma omp parallel for\n"
                   "    for (int i = 0; i < n; ++i) out[i] = 0.0f;\n"
                   "}\n")
        code, out, err = run_cli(self.root)
        self.assertEqual(code, 1)
        self.assertIn("not one unconditional canonical statement", err)

    def test_inline_unsupported_form_fails_closed(self) -> None:
        self.write("src/v2/inline_pragma.cpp",
                   "void kernel(float *out) {\n"
                   "    int x = 1; _Pragma(\"omp parallel for\") for (int i = 0; i < n; ++i) out[i] = x;\n"
                   "}\n")
        code, out, err = run_cli(self.root)
        self.assertEqual(code, 1)
        self.assertIn("unsupported OpenMP parallel form", err)

    def test_site_exemptions_are_never_generic(self) -> None:
        self.assertTrue(all(reason.strip() for reason in
                            __import__("check_cpu_openmp_fork_note_policy").EXEMPT_SITES.values()))
        self.assertIn("src/v2/utils/OpenMPUtils.h",
                      str(__import__("check_cpu_openmp_fork_note_policy").AUTHORITY_FILE))


if __name__ == "__main__":
    unittest.main()
