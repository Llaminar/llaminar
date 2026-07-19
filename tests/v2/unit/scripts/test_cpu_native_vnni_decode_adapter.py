"""Unit regressions for strict CPU NativeVNNI M=1 evidence adaptation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.adapters.cpu_decode import (  # noqa: E402
    CPUDecodeAdapterContext,
    adapt_cpu_decode_row,
)
from native_vnni_dispatch.format_registry import format_spec  # noqa: E402
from native_vnni_dispatch.profiles import MeasurementProfile  # noqa: E402


def context() -> CPUDecodeAdapterContext:
    """Return non-installable provenance suitable for pure adapter tests."""

    return CPUDecodeAdapterContext(
        profile=MeasurementProfile.QUICK,
        run_id="unit-run",
        corpus_id="sha256:" + "1" * 64,
        git_revision="unit-revision",
        build_id="unit-build",
        compiler_id="unit-compiler",
        architecture_class="unit-cpu",
        device_name="unit-device",
        driver_runtime="unit-runtime",
        frozen_serial_policy_hash="sha256:" + "2" * 64,
        raw_timing_sidecar_retained=False,
    )


def raw_row() -> dict[str, str]:
    """Return one byte-exact, route-authenticated NBC4 observation."""

    spec = format_spec("Q4_K")
    return {
        "backend": "cpu",
        "phase": "decode_m1",
        "source_format": "Q4_K",
        "source_codebook": str(spec.source_codebook_id),
        "execution_codebook": str(spec.runtime_codebook("cpu")),
        "shape": "UnitDecode",
        "execution_mode": "eager",
        "m": "1",
        "n": "5120",
        "k": "5120",
        "candidate_id": "cpu.nvnni.decode.n_chunk_grid.nbc4",
        "build_isa": "AVX512",
        "runtime_isa_requested": "AVX512",
        "runtime_isa_effective": "AVX512",
        "threads": "28",
        "weight_bytes": "4194304",
        "warmup_count": "1",
        "sample_count": "1",
        "min_us": "10.0",
        "median_us": "10.0",
        "p95_us": "10.0",
        "mad_us": "0.0",
        "cv": "0.0",
        "bit_mismatches": "0",
        "first_bit_mismatch": "0",
        "repeat_byte_mismatches": "0",
        "max_abs": "0.0",
        "relative_l2": "0.0",
        "cosine": "1.0",
        "symmetric_kld": "0.0",
        "output_digest": "sha256:" + "3" * 64,
        "oracle_output_digest": "sha256:" + "3" * 64,
        "timing_sample_digest": "sha256:" + "4" * 64,
        "route_counter_ok": "1",
        "observed_candidate_id": "Nbc4",
        "k_tiles": "0",
        "n_block_chunks": "4",
        "serial_kpart": "0",
        "numerical_correctness": "1",
        "correctness_pass": "1",
        "is_winner": "1",
    }


class CPUDecodeAdapterTest(unittest.TestCase):
    """Protect byte, route, ISA, and arithmetic-bundle admission gates."""

    def test_adapts_forceable_byte_exact_candidate(self) -> None:
        observation = adapt_cpu_decode_row(raw_row(), context())

        self.assertTrue(observation.supported)
        self.assertTrue(observation.bitwise_equal)
        self.assertEqual(observation.m, 1)
        self.assertEqual(
            observation.operation_kind, "NativeVNNIFastM1Projection"
        )
        self.assertIn("serial-full-k", observation.bundle_signature)

    def test_normalized_nominal_candidate_is_not_supported(self) -> None:
        raw = raw_row()
        raw["candidate_id"] = "cpu.nvnni.decode.n_chunk_grid.nbc8"
        raw["correctness_pass"] = "0"

        observation = adapt_cpu_decode_row(raw, context())

        self.assertFalse(observation.supported)
        self.assertFalse(observation.forced_route_ok)

    def test_rejects_digest_mismatch_hidden_behind_zero_count(self) -> None:
        raw = raw_row()
        raw["oracle_output_digest"] = "sha256:" + "5" * 64

        with self.assertRaisesRegex(ValueError, "mismatch count"):
            adapt_cpu_decode_row(raw, context())

    def test_rejects_non_m1_evidence(self) -> None:
        raw = raw_row()
        raw["m"] = "2"

        with self.assertRaisesRegex(ValueError, "requires M=1"):
            adapt_cpu_decode_row(raw, context())

    def test_kpart_bundle_is_bound_to_route_counter(self) -> None:
        raw = raw_row()
        raw["k_tiles"] = "4"
        raw["serial_kpart"] = "1"

        observation = adapt_cpu_decode_row(raw, context())

        self.assertIn("serial-kpart", observation.bundle_signature)

    def test_rejects_inconsistent_kpart_marker(self) -> None:
        raw = raw_row()
        raw["k_tiles"] = "4"

        with self.assertRaisesRegex(ValueError, "serial_kpart"):
            adapt_cpu_decode_row(raw, context())


if __name__ == "__main__":
    unittest.main()
