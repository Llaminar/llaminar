#!/usr/bin/env python3
"""Focused regressions for strong ROCm NativeVNNI decode adaptation."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.v2.performance.kernels.native_vnni_dispatch.adapters.evidence import (
    summarize_sorted_timing,
)
from tests.v2.performance.kernels.native_vnni_dispatch.adapters.rocm_decode import (
    ROCmDecodeAdapterContext,
    adapt_rocm_decode_csv,
    adapt_rocm_decode_row,
    raw_corpus_id,
    read_rocm_decode_timing_sidecars,
)
from tests.v2.performance.kernels.native_vnni_dispatch.exact_oracle import (
    candidate_is_eligible,
)
from tests.v2.performance.kernels.native_vnni_dispatch.schema import (
    SemanticContract,
)


class ROCmNativeVNNIDecodeAdapterTest(unittest.TestCase):
    """Prove contract normalization, exactness, and sidecar verification."""

    @staticmethod
    def context() -> ROCmDecodeAdapterContext:
        return ROCmDecodeAdapterContext.workflow_smoke(
            corpus_id="sha256:" + "1" * 64
        )

    @staticmethod
    def row(*, m: int = 2, candidate: str = "INHERIT_SERIAL_M1") -> dict[str, str]:
        """Build one valid strong-evidence Q8 grouped-verifier row."""

        return {
            "backend": "rocm",
            "phase": "decode",
            "source_format": "Q8_0",
            "source_codebook": "19",
            "execution_codebook": "19",
            "shape": "35BMoE_Expert_GateUp",
            "execution_mode": "eager",
            "m": str(m),
            "n": "512",
            "k": "2048",
            "candidate_id": candidate,
            "kb": "32",
            "target_waves": "4",
            "weight_bytes": "1114112",
            "warmup_count": "2",
            "sample_count": "3",
            "min_us": "10",
            "median_us": "11",
            "p95_us": "13",
            "mad_us": "1",
            "cv": "0.1",
            "effective_bandwidth_gbs": "30",
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
            "observed_candidate_id": candidate,
            "observed_path": "split_reduce",
            "serial_m1_kb": "32",
            "serial_m1_target_waves": "4",
            "serial_route_counter_ok": "1",
            "numerical_correctness": "1",
            "correctness_pass": "1",
            "is_winner": "1",
        }

    def test_verifier_inherits_serial_policy_and_is_exactly_eligible(self) -> None:
        observation = adapt_rocm_decode_row(self.row(), self.context())

        self.assertEqual(
            observation.semantic_contract,
            SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
        )
        self.assertEqual(
            observation.candidate_id,
            "rocm.nvnni.decode.verifier.inherit_serial_m1",
        )
        self.assertTrue(candidate_is_eligible(
            observation, self.context().serial_m1_policy_hash
        ))

    def test_extended_runtime_m_keeps_the_grouped_verifier_contract(self) -> None:
        observation = adapt_rocm_decode_row(self.row(m=31), self.context())

        self.assertEqual(observation.m, 31)
        self.assertEqual(
            observation.semantic_contract,
            SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
        )

    def test_explicit_kb_cannot_masquerade_as_grouped_verifier_candidate(self) -> None:
        row = self.row(candidate="KB32")
        row["observed_candidate_id"] = "KB32"

        with self.assertRaisesRegex(ValueError, "does not support Verifier"):
            adapt_rocm_decode_row(row, self.context())

    def test_atomic_publication_evidence_is_rejected(self) -> None:
        """A retired completion-order reduction cannot enter a corpus."""

        row = self.row()
        row["observed_path"] = "atomic_reduce"

        with self.assertRaisesRegex(ValueError, "unsupported publication path"):
            adapt_rocm_decode_row(row, self.context())

    def test_fast_m1_candidate_may_be_numerically_correct_without_byte_identity(self) -> None:
        row = self.row(m=1, candidate="KB8/TW24")
        row.update({
            "kb": "8",
            "observed_candidate_id": "KB8",
            "bit_mismatches": "3",
            "first_bit_mismatch": "2",
            "max_abs": "0.000001",
            "relative_l2": "0.0000001",
            "grouped_output_digest": "fnv1a64:different000000",
        })

        observation = adapt_rocm_decode_row(row, self.context())

        self.assertEqual(observation.semantic_contract, SemanticContract.FAST)
        self.assertEqual(
            observation.candidate_id,
            "rocm.nvnni.decode.fast.kb8",
        )
        self.assertFalse(observation.bitwise_equal)
        self.assertTrue(candidate_is_eligible(observation))

    def test_exact_timing_sidecar_reconstructs_digest_and_robust_statistics(self) -> None:
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
            aggregate = root / "decode.csv"
            sidecar = root / "decode-timing.csv"
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(row))
                writer.writeheader()
                writer.writerow(row)
            timing_rows = []
            for index, value in enumerate(samples):
                timing_rows.append({
                    "backend": "rocm",
                    "phase": "decode",
                    "source_format": "Q8_0",
                    "source_codebook": "19",
                    "execution_codebook": "19",
                    "shape": "35BMoE_Expert_GateUp",
                    "execution_mode": "eager",
                    "m": "2",
                    "n": "512",
                    "k": "2048",
                    "candidate_id": "INHERIT_SERIAL_M1",
                    "kb": "32",
                    "target_waves": "4",
                    "sample_index": str(index),
                    "timed_replays": "1",
                    "latency_us": f"{value:.9f}",
                    "latency_us_hex": value.hex(),
                })
            with sidecar.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(timing_rows[0]))
                writer.writeheader()
                writer.writerows(timing_rows)

            context = ROCmDecodeAdapterContext.workflow_smoke(
                corpus_id=raw_corpus_id((aggregate, sidecar))
            )
            corpus = adapt_rocm_decode_csv(
                (aggregate,), context, timing_sidecars=(sidecar,)
            )
            self.assertEqual(len(corpus), 1)
            self.assertEqual(corpus.observations[0].timing_sample_hash, summary.digest)

            row["mad_us"] = "99"
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(row))
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "aggregate mad_us"):
                adapt_rocm_decode_csv(
                    (aggregate,), context, timing_sidecars=(sidecar,)
                )

    def test_parallel_timing_reader_preserves_complete_trial_sequences(self) -> None:
        """A byte split inside one trial must advance to the next trial."""

        rows = []
        for shape, samples in (
            ("LargeFirstTrial", tuple(float(index + 1) for index in range(20))),
            ("SmallSecondTrial", (101.0, 102.0, 103.0)),
        ):
            for sample_index, latency in enumerate(samples):
                rows.append({
                    "backend": "rocm",
                    "phase": "decode",
                    "source_format": "Q8_0",
                    "source_codebook": "19",
                    "execution_codebook": "19",
                    "shape": shape,
                    "execution_mode": "eager",
                    "m": "1",
                    "n": "512",
                    "k": "2048",
                    "candidate_id": "KB32/TW4",
                    "kb": "32",
                    "target_waves": "4",
                    "sample_index": str(sample_index),
                    "timed_replays": "1",
                    "latency_us": f"{latency:.9f}",
                    "latency_us_hex": latency.hex(),
                })

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rocm-timing.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            serial = read_rocm_decode_timing_sidecars((path,), workers=1)
            parallel = read_rocm_decode_timing_sidecars((path,), workers=2)

        self.assertEqual(parallel, serial)
        self.assertEqual(len(parallel), 2)
        self.assertEqual(sorted(len(samples) for samples in parallel.values()), [3, 20])

    def test_timing_reader_rejects_nonpositive_worker_count(self) -> None:
        """Explicit analyzer parallelism must remain a positive contract."""

        with self.assertRaisesRegex(ValueError, "worker count must be positive"):
            read_rocm_decode_timing_sidecars((Path("unused.csv"),), workers=0)


if __name__ == "__main__":
    unittest.main()
