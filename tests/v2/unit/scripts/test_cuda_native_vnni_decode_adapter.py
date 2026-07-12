#!/usr/bin/env python3
"""Focused regressions for strong CUDA NativeVNNI decode adaptation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.v2.performance.kernels.native_vnni_dispatch.adapters.cuda_decode import (  # noqa: E402
    CUDADecodeAdapterContext,
    adapt_cuda_decode_row,
)
from tests.v2.performance.kernels.native_vnni_dispatch.exact_oracle import (  # noqa: E402
    candidate_is_eligible,
)


class CUDANativeVNNIDecodeAdapterTest(unittest.TestCase):
    """Prove candidate normalization, route inheritance, and byte gates."""

    @staticmethod
    def context() -> CUDADecodeAdapterContext:
        return CUDADecodeAdapterContext.workflow_smoke(
            corpus_id="sha256:" + "7" * 64
        )

    @staticmethod
    def fast_row() -> dict[str, str]:
        candidate = "cuda.nvnni.decode.fast_m1.kpar.tn256.cpt4.kb8"
        return {
            "backend": "cuda",
            "phase": "decode",
            "source_format": "Q8_K",
            "source_codebook": "21",
            "execution_codebook": "19",
            "shape": "StrongSmoke",
            "execution_mode": "eager",
            "m": "1",
            "n": "128",
            "k": "256",
            "candidate_id": candidate,
            "family": "kpar",
            "tile_n": "256",
            "cpt": "4",
            "target_waves": "0",
            "mkg": "0",
            "max_kb": "0",
            "exact_kb": "8",
            "force_two_phase": "1",
            "weight_bytes": "36864",
            "warmup_count": "2",
            "sample_count": "5",
            "min_us": "10",
            "median_us": "11",
            "p95_us": "13",
            "mad_us": "1",
            "cv": "0.1",
            "effective_bandwidth_gbs": "3.351272727",
            "bit_mismatches": "0",
            "first_bit_mismatch": "0",
            "repeat_byte_mismatches": "0",
            "max_abs": "0",
            "relative_l2": "0",
            "cosine": "1",
            "symmetric_kld": "0",
            "grouped_output_digest": "sha256:same",
            "serial_output_digest": "sha256:same",
            "timing_sample_digest": "sha256:timing",
            "supported": "1",
            "graph_capture_ok": "1",
            "workspace_ok": "1",
            "explicit_stream_ok": "1",
            "route_counter_ok": "1",
            "observed_candidate_id": candidate,
            "observed_path": "kpar",
            "observed_tile_n": "256",
            "observed_cpt": "4",
            "observed_effective_kb": "8",
            "serial_m1_candidate_id": candidate,
            "serial_route_counter_ok": "1",
            "numerical_correctness": "1",
            "correctness_pass": "1",
            "is_winner": "1",
        }

    @classmethod
    def verifier_row(cls) -> dict[str, str]:
        row = cls.fast_row()
        row.update({
            "candidate_id": "INHERIT_SERIAL_M1",
            "family": "inherit_serial_m1",
            "tile_n": "0",
            "cpt": "0",
            "target_waves": "0",
            "mkg": "0",
            "max_kb": "0",
            "exact_kb": "0",
            "force_two_phase": "0",
            "m": "3",
            "observed_candidate_id": "INHERIT_SERIAL_M1",
        })
        return row

    def test_fast_m1_preserves_exact_kpart_identity(self) -> None:
        observation = adapt_cuda_decode_row(self.fast_row(), self.context())
        self.assertEqual(observation.runtime_codebook_id, 19)
        self.assertEqual(
            observation.effective_candidate_id,
            "cuda.nvnni.decode.fast_m1.kpar.tn256.cpt4.kb8",
        )
        self.assertTrue(candidate_is_eligible(observation))

    def test_verifier_inherits_public_m1_and_is_byte_eligible(self) -> None:
        observation = adapt_cuda_decode_row(self.verifier_row(), self.context())
        self.assertEqual(observation.m, 3)
        self.assertTrue(observation.bitwise_equal)
        self.assertTrue(candidate_is_eligible(
            observation,
            self.context().serial_m1_policy_hash,
        ))

    def test_sixteen_row_verifier_uses_same_bitwise_contract(self) -> None:
        row = self.verifier_row()
        row["m"] = "16"
        observation = adapt_cuda_decode_row(row, self.context())
        self.assertEqual(observation.m, 16)
        self.assertTrue(observation.bitwise_equal)
        self.assertTrue(candidate_is_eligible(
            observation,
            self.context().serial_m1_policy_hash,
        ))

    def test_verifier_byte_mismatch_is_ineligible(self) -> None:
        row = self.verifier_row()
        row.update({
            "bit_mismatches": "1",
            "first_bit_mismatch": "17",
            "grouped_output_digest": "sha256:grouped",
            "serial_output_digest": "sha256:serial",
            "max_abs": "0.0001",
            "correctness_pass": "0",
        })
        observation = adapt_cuda_decode_row(row, self.context())
        self.assertFalse(candidate_is_eligible(
            observation,
            self.context().serial_m1_policy_hash,
        ))

    def test_inherited_route_mismatch_cannot_claim_forced_route(self) -> None:
        row = self.verifier_row()
        row["observed_tile_n"] = "128"
        row["correctness_pass"] = "0"
        observation = adapt_cuda_decode_row(row, self.context())
        self.assertFalse(observation.forced_route_ok)
        self.assertFalse(candidate_is_eligible(
            observation,
            self.context().serial_m1_policy_hash,
        ))

    def test_inherited_kpart_mismatch_cannot_claim_batch_invariance(self) -> None:
        row = self.verifier_row()
        row["observed_effective_kb"] = "4"
        row["correctness_pass"] = "0"
        observation = adapt_cuda_decode_row(row, self.context())
        self.assertFalse(observation.forced_route_ok)
        self.assertFalse(candidate_is_eligible(
            observation,
            self.context().serial_m1_policy_hash,
        ))

    def test_removed_atomic_candidate_is_rejected(self) -> None:
        row = self.fast_row()
        row["candidate_id"] = (
            "cuda.nvnni.kpar.tn256.cpt4.tw8.mkg4.kb4.phase2"
        )
        with self.assertRaisesRegex(ValueError, "unknown forceable candidate"):
            adapt_cuda_decode_row(row, self.context())


if __name__ == "__main__":
    unittest.main()
