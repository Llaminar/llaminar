#!/usr/bin/env python3
"""Negative controls for the CPU RoPE angle storage source policy."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_cpu_rope_angle_storage_policy as policy  # noqa: E402


def violations(source: str) -> list[str]:
    return [violation.message for violation in policy.scan("probe.cpp", source)]


class CPURoPEAngleStoragePolicyTest(unittest.TestCase):
    def test_named_inv_freq_memo_is_the_only_thread_local(self) -> None:
        admitted = """
        const std::vector<float> &get_inv_freq_cached(int d, float b) {
            thread_local std::array<Memo, 8> memo{};
            return *memo[0].values;
        }"""
        self.assertEqual(violations(admitted), [])
        rejected = """
        void apply_rope_vectorized(float *q) {
            thread_local RoPEPersistentState state;
        }"""
        self.assertEqual(len(violations(rejected)), 1)

    def test_former_verifier_cache_and_per_call_tables_are_rejected(self) -> None:
        former = """
        namespace {
            thread_local VerifierAngleRows g_verifier_angle_rows;
        }
        static void apply_rope_pair_positions(int rows, int half) {
            std::vector<float> cos_table(static_cast<size_t>(rows) * half);
            std::vector<float> sin_table(cos_table.size());
        }
        void update_rope_cache(RoPEPersistentState &state) {
            state.cos_curr.resize(4);
        }"""
        self.assertEqual(len(violations(former)), 4)

    def test_references_and_allowlisted_functions_are_admitted(self) -> None:
        source = """
        void fill_rope_angle_row(const std::vector<float> &inv_freq, int p, float *c, float *s) {
            c[0] = inv_freq[0];
        }
        void apply_rope_q16_1_integer(int seq_len, int half) {
            std::vector<int16_t> cos_table(seq_len * half);
        }"""
        self.assertEqual(violations(source), [])

    def test_allocation_forms_are_rejected_in_unlisted_functions(self) -> None:
        for statement in ("auto *p = new float[4];", "auto p = std::make_unique<float[]>(4);",
                          "void *p = malloc(16);", "angles.reserve(4);", "angles.assign(4, 0);",
                          "auto copy = std::vector<float>(4);"):
            with self.subTest(statement=statement):
                self.assertEqual(len(violations(f"void apply_rope_partial() {{ {statement} }}")), 1)

    def test_the_global_table_is_the_only_namespace_scope_storage(self) -> None:
        self.assertEqual(violations("static std::unordered_map<uint64_t, std::vector<float>> g_inv_freq_cache;"), [])
        self.assertEqual(len(violations("static std::vector<float> g_shared_angles;")), 1)


if __name__ == "__main__":
    unittest.main()
