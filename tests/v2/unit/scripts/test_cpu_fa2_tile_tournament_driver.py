#!/usr/bin/env python3
"""! @file test_cpu_fa2_tile_tournament_driver.py
@brief Fast unit tests for CPU FA2 aggregate tournament authentication.

These tests are device-free and execute no performance binary. They lock down
nearest-rank p95 semantics, Linux CPU-list parsing, all-regime totality, all
seven candidate rows per domain, and disagreement rejection in the parent
driver that certifies process-isolated worker results.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


DRIVER_PATH = (
    Path(__file__).resolve().parents[2]
    / "performance/kernels/cpu/attention/run_cpu_fa2_tile_tournament.py"
)
SPEC = importlib.util.spec_from_file_location("cpu_fa2_tournament_driver", DRIVER_PATH)
assert SPEC is not None and SPEC.loader is not None
DRIVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DRIVER
SPEC.loader.exec_module(DRIVER)


def family_csv(regrets: dict[str, float] | None = None) -> str:
    """Build one authenticated three-regime, seven-candidate worker payload."""

    regrets = regrets or {
        "decode": 1.0,
        "grouped_verifier": 2.0,
        "prefill": 3.0,
    }
    points = {
        "decode": (1, 8192),
        "grouped_verifier": (15, 8207),
        "prefill": (128, 8320),
    }
    rows: list[str] = []
    for regime, (query_rows, kv_rows) in points.items():
        for tile in (4, 8, 16, 32, 64, 128, 256):
            columns = [
                "Qwen@TP1",
                "fp32",
                "avx512",
                "avx512",
                regime,
                str(query_rows),
                str(kv_rows),
                "8",
                "2",
                "64",
                "4",
                "28",
                "256",
                "256",
                "256",
                "256",
                "10.0",
                "9.0",
                str(regrets[regime]),
                str(tile),
                "8.0",
                "10.0",
                "1.0",
            ]
            rows.append(",".join(columns))
    return "\n".join(rows)


class CPUFA2TournamentDriverTests(unittest.TestCase):
    """Authenticate every failure-sensitive pure function in the driver."""

    def test_nearest_rank_matches_cpp_gate(self) -> None:
        values = list(range(1, 82))
        self.assertEqual(DRIVER.nearest_rank_percentile(values, 0.95), 77)

    def test_cpu_list_is_total_and_deduplicated(self) -> None:
        self.assertEqual(DRIVER.parse_cpu_list("0-2,2,7"), [0, 1, 2, 7])
        with self.assertRaises(ValueError):
            DRIVER.parse_cpu_list("4-2")

    def test_family_parser_requires_all_regimes_and_candidates(self) -> None:
        domains = DRIVER.parse_family_output(family_csv(), "fp32", 64)
        self.assertEqual(len(domains), 3)
        self.assertEqual(
            {domain.regime for domain in domains},
            DRIVER.EXPECTED_REGIMES,
        )

        missing_candidate = family_csv().splitlines()
        with self.assertRaises(RuntimeError):
            DRIVER.parse_family_output(
                "\n".join(missing_candidate[:-1]), "fp32", 64
            )

    def test_family_parser_rejects_candidate_disagreement(self) -> None:
        rows = family_csv().splitlines()
        columns = rows[1].split(",")
        columns[18] = "99.0"
        rows[1] = ",".join(columns)
        with self.assertRaises(RuntimeError):
            DRIVER.parse_family_output("\n".join(rows), "fp32", 64)

    def test_family_repeats_use_domain_median_and_require_totality(self) -> None:
        repeats = [
            DRIVER.parse_family_output(
                family_csv(
                    {
                        "decode": value,
                        "grouped_verifier": 2.0,
                        "prefill": 3.0,
                    }
                ),
                "fp32",
                64,
            )
            for value in (1.0, 99.0, 2.0)
        ]
        aggregated = DRIVER.aggregate_family_repeats(repeats)
        decode = next(item for item in aggregated if item.regime == "decode")
        self.assertEqual(decode.regret_percent, 2.0)

        with self.assertRaises(RuntimeError):
            DRIVER.aggregate_family_repeats([repeats[0], repeats[1][:-1]])


if __name__ == "__main__":
    unittest.main()
