#!/usr/bin/env python3
"""Negative controls for the OpenMP fork-note source policy."""
from __future__ import annotations

import unittest

import check_cpu_openmp_fork_note_policy as policy


def violations(source: str) -> list[str]:
    return [violation.message for violation in policy.scan("probe.cpp", source)]


class OpenMPForkNotePolicyTest(unittest.TestCase):
    def test_noted_pragma_is_accepted(self) -> None:
        noted = """
        void kernel(float *out) {
            ::llaminar::v2::noteOpenMPTeamFork();
        #pragma omp parallel for
            for (int i = 0; i < n; ++i) out[i] = 0.0f;
        }"""
        self.assertEqual(violations(noted), [])

    def test_unnoted_pragma_is_rejected_with_inventory(self) -> None:
        unnoted = """
        void kernel(float *out) {
        #pragma omp parallel for
            for (int i = 0; i < n; ++i) out[i] = 0.0f;
        }"""
        messages = violations(unnoted)
        self.assertEqual(len(messages), 1)
        self.assertIn("raw OpenMP fork outside the note authority", messages[0])

    def test_note_separated_from_pragma_by_blank_lines_still_binds(self) -> None:
        separated = """
        void kernel(float *out) {
            ::llaminar::v2::noteOpenMPTeamFork();  // one real fork below

        #pragma omp parallel num_threads(4)
            { work(); }
        }"""
        self.assertEqual(violations(separated), [])

    def test_note_with_no_following_pragma_is_rejected(self) -> None:
        orphan = """
        void prepare(float *out) {
            ::llaminar::v2::noteOpenMPTeamFork();
            out[0] = 1.0f;
        }"""
        messages = violations(orphan)
        self.assertEqual(len(messages), 1)
        self.assertIn("no pragma after it", messages[0])

    def test_stale_note_beyond_the_window_does_not_bind(self) -> None:
        stale = """
        void kernel(float *out) {
            ::llaminar::v2::noteOpenMPTeamFork();
            for (int i = 0; i < 7; ++i) { sink(i); }
            for (int i = 0; i < 7; ++i) { sink(i); }
            for (int i = 0; i < 7; ++i) { sink(i); }
            for (int i = 0; i < 7; ++i) { sink(i); }
            for (int i = 0; i < 7; ++i) { sink(i); }
            for (int i = 0; i < 7; ++i) { sink(i); }
            for (int i = 0; i < 7; ++i) { sink(i); }
        #pragma omp parallel for
            for (int i = 0; i < n; ++i) out[i] = 0.0f;
        }"""
        messages = violations(stale)
        self.assertEqual(len(messages), 2)
        self.assertTrue(any("no pragma after it" in message for message in messages))
        self.assertTrue(any("raw OpenMP fork outside the note authority" in message
                            for message in messages))

    def test_commented_pragma_and_note_neither_bind_nor_reject(self) -> None:
        commented = """
        void kernel(float *out) {
            // ::llaminar::v2::noteOpenMPTeamFork();
            // #pragma omp parallel for
            for (int i = 0; i < n; ++i) out[i] = 0.0f;
        }"""
        self.assertEqual(violations(commented), [])

    def test_exempt_files_are_skipped_entirely(self) -> None:
        exempt = """
        void load(void *region) {
        #pragma omp parallel for
            for (int i = 0; i < n; ++i) touch(region, i);
        }"""
        self.assertEqual(policy.scan("src/v2/loaders/MmapRegion.h", exempt), [])
        self.assertIn("src/v2/loaders/MmapRegion.h", policy.EXEMPT_FILES)

    def test_exempt_files_are_never_generic(self) -> None:
        self.assertTrue(all(reason.strip() for reason in policy.EXEMPT_FILES.values()))
        self.assertIn("src/v2/utils/OpenMPUtils.h", policy.EXEMPT_FILES)


if __name__ == "__main__":
    unittest.main()
