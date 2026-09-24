#!/usr/bin/env python3
"""Focused regressions for strong CPU grouped-verifier adaptation."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

# CTest executes direct ``add_test`` Python regressions from the build tree.
# Resolve imports from this file instead of depending on the caller's current
# working directory so the test exercises the same adapter in local and CI
# invocations.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.v2.performance.kernels.native_vnni_dispatch.adapters.cpu_verifier import (
    CPUVerifierAdapterContext,
    adapt_cpu_verifier_csv,
    adapt_cpu_verifier_row,
    raw_corpus_id,
)
from tests.v2.performance.kernels.native_vnni_dispatch.adapters.evidence import (
    summarize_sorted_timing,
)
from tests.v2.performance.kernels.native_vnni_dispatch.exact_oracle import (
    candidate_is_eligible,
)


class CPUNativeVNNIVerifierAdapterTest(unittest.TestCase):
    """Prove exact-route admission, byte gates, Q8_K, and timing provenance."""

    @staticmethod
    def context() -> CPUVerifierAdapterContext:
        return CPUVerifierAdapterContext.workflow_smoke(
            corpus_id="sha256:" + "2" * 64
        )

    @staticmethod
    def row(candidate: str = "Pairwise") -> dict[str, str]:
        observed = "Pairwise" if candidate == "WideRows" else candidate
        route_matches = observed == candidate
        return {
            "backend": "cpu",
            "phase": "verifier_rows",
            "source_format": "Q8_K",
            "source_codebook": "21",
            "execution_codebook": "21",
            "shape": "StrongSmoke",
            "execution_mode": "eager",
            "m": "2",
            "n": "128",
            "k": "256",
            "candidate_id": candidate,
            "build_isa": "AVX2",
            "runtime_isa_requested": "AVX2",
            "runtime_isa_effective": "AVX2",
            "threads": "4",
            "weight_bytes": "36864",
            "warmup_count": "1",
            "sample_count": "3",
            "min_us": "10",
            "median_us": "11",
            "p95_us": "13",
            "mad_us": "1",
            "cv": "0.1",
            "serial_median_us": "9",
            "speedup": "0.818181818",
            "bit_mismatches": "0",
            "first_bit_mismatch": "0",
            "repeat_byte_mismatches": "0",
            "max_abs": "0",
            "relative_l2": "0",
            "cosine": "1",
            "symmetric_kld": "0",
            "grouped_output_digest": "fnv1a64:equal00000000000",
            "serial_output_digest": "fnv1a64:equal00000000000",
            "timing_sample_digest": "fnv1a64:timing0000000000",
            "route_counter_ok": "1",
            "observed_candidate_id": observed,
            "k_tiles": "0",
            "n_block_chunks": "1",
            "numerical_correctness": "1",
            "correctness_pass": "1" if route_matches else "0",
            "is_winner": "1" if route_matches else "0",
        }

    def test_q8k_pairwise_row_is_byte_exact_and_eligible(self) -> None:
        observation = adapt_cpu_verifier_row(self.row(), self.context())

        self.assertEqual(observation.source_format, "Q8_K")
        self.assertEqual(observation.runtime_codebook_id, 21)
        self.assertTrue(candidate_is_eligible(
            observation, self.context().serial_m1_policy_hash
        ))

    def test_extended_runtime_m_is_not_rejected_by_the_adapter(self) -> None:
        row = self.row()
        row["m"] = "31"

        observation = adapt_cpu_verifier_row(row, self.context())

        self.assertEqual(observation.m, 31)
        self.assertTrue(observation.bitwise_equal)

    def test_wide_request_normalized_to_pairwise_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact requested production route"):
            adapt_cpu_verifier_row(self.row("WideRows"), self.context())

    def test_full_k_candidate_in_k_partition_domain_is_rejected(self) -> None:
        row = self.row("FullKRowChunkGrid")
        row["k_tiles"] = "4"

        with self.assertRaisesRegex(ValueError, "full-K verifier route"):
            adapt_cpu_verifier_row(row, self.context())

    def test_impossible_build_and_runtime_isa_pair_is_rejected(self) -> None:
        row = self.row()
        row["runtime_isa_requested"] = "AVX512"
        row["runtime_isa_effective"] = "AVX512"

        with self.assertRaisesRegex(ValueError, "AVX2-only build"):
            adapt_cpu_verifier_row(row, self.context())

    def test_raw_sidecar_reconstructs_every_robust_timing_field(self) -> None:
        samples = (10.0, 11.0, 13.0)
        summary = summarize_sorted_timing(samples)
        row = self.row()
        row.update({
            "min_us": str(summary.minimum),
            "median_us": str(summary.median),
            "p95_us": str(summary.p95),
            "mad_us": str(summary.mad),
            "cv": repr(summary.cv),
            "timing_sample_digest": summary.digest,
        })
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aggregate = root / "cpu.csv"
            sidecar = root / "cpu-timing.csv"
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(row))
                writer.writeheader()
                writer.writerow(row)
            timing_rows = [{
                "backend": "cpu",
                "phase": "verifier_rows",
                "source_format": "Q8_K",
                "source_codebook": "21",
                "execution_codebook": "21",
                "shape": "StrongSmoke",
                "execution_mode": "eager",
                "m": "2",
                "n": "128",
                "k": "256",
                "candidate_id": "Pairwise",
                "build_isa": "AVX2",
                "runtime_isa_requested": "AVX2",
                "runtime_isa_effective": "AVX2",
                "sample_index": str(index),
                "latency_us": f"{value:.9f}",
                "latency_us_hex": value.hex(),
            } for index, value in enumerate(samples)]
            with sidecar.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(timing_rows[0]))
                writer.writeheader()
                writer.writerows(timing_rows)

            context = CPUVerifierAdapterContext.workflow_smoke(
                corpus_id=raw_corpus_id((aggregate, sidecar))
            )
            corpus = adapt_cpu_verifier_csv(
                (aggregate,), context, timing_sidecars=(sidecar,)
            )
            self.assertEqual(len(corpus), 1)

            row["timing_sample_digest"] = "fnv1a64:0000000000000000"
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(row))
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "digest"):
                adapt_cpu_verifier_csv(
                    (aggregate,), context, timing_sidecars=(sidecar,)
                )

    def test_relaxed_legacy_row_is_rejected_at_the_header(self) -> None:
        row = self.row()
        del row["repeat_byte_mismatches"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weak.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(row))
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "missing strong CPU"):
                adapt_cpu_verifier_csv((path,), self.context())


if __name__ == "__main__":
    unittest.main()
