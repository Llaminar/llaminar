#!/usr/bin/env python3
"""Regression tests for model-tiered ordinary-prefill measurements."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.prefill_matrix import (  # noqa: E402
    CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS,
    PREFILL_M_BUCKETS,
    QWEN36_35B_MOE_SHAPES,
    cpu_prefill_maximum_weight_elements_for_m,
    cpu_prefill_measurements,
    gpu_prefill_measurements,
)
from native_vnni_dispatch.qwen_release_geometry import (  # noqa: E402
    qwen_release_geometries,
)


class NativeVNNIPrefillMatrixTest(unittest.TestCase):
    """Lock projection applicability and economical M ceilings."""

    def test_model_tiers_have_expected_depths(self) -> None:
        matrix = {row.shape.name: row.m_values for row in cpu_prefill_measurements()}

        self.assertEqual(matrix["7B_FFN_Up"], PREFILL_M_BUCKETS)
        self.assertEqual(matrix["14B_FFN_Up"][-1], 4096)
        self.assertEqual(matrix["32B_FFN_Up"], (64, 256, 1024))

    def test_shared_attention_geometry_uses_mid_tier_economy(self) -> None:
        matrix = {row.shape.name: row for row in cpu_prefill_measurements()}

        self.assertEqual(matrix["32B_AttnOut"].maximum_m, 4096)
        self.assertEqual(matrix["32B_QKV"].maximum_m, 4096)
        self.assertEqual(matrix["32B_AttnOut"].shape.n, 5120)

    def test_matrix_covers_dense_and_moe_production_without_lm_heads(self) -> None:
        measurements = cpu_prefill_measurements()
        names = {row.shape.name for row in measurements}

        self.assertEqual(len(measurements), 75)
        self.assertFalse(any("LM_Head" in name for name in names))
        for prefix in ("0.5B", "1.5B", "3B", "7B", "14B", "32B"):
            self.assertTrue(any(name.startswith(prefix) for name in names), prefix)
        self.assertTrue(set(QWEN36_35B_MOE_SHAPES).issubset(names))
        measured_dimensions = {
            (row.shape.n, row.shape.k) for row in measurements
        }
        required_prefill_dimensions = {
            (geometry.n, geometry.k)
            for geometry in qwen_release_geometries()
            if any(use.projection != "lm_head" for use in geometry.uses)
        }
        self.assertTrue(
            required_prefill_dimensions.issubset(measured_dimensions)
        )

    def test_matrix_has_166_shape_depth_cells(self) -> None:
        self.assertEqual(
            sum(len(row.m_values) for row in cpu_prefill_measurements()),
            425,
        )

    def test_cpu_matrix_measures_qwen36_35b_moe_at_every_depth(self) -> None:
        """CPU receives exact overlays for every active MoE projection."""

        matrix = {row.shape.name: row.m_values for row in cpu_prefill_measurements()}

        for name in QWEN36_35B_MOE_SHAPES:
            self.assertEqual(matrix[name], PREFILL_M_BUCKETS, name)

    def test_cpu_adaptive_work_envelope_tracks_model_tiers(self) -> None:
        """Fresh synthetic evidence cannot exceed each M bucket's economy cap."""

        self.assertEqual(
            cpu_prefill_maximum_weight_elements_for_m(16384),
            18944 * 3584,
        )
        self.assertEqual(
            cpu_prefill_maximum_weight_elements_for_m(4096),
            13824 * 5120,
        )
        self.assertEqual(
            cpu_prefill_maximum_weight_elements_for_m(1024),
            CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS,
        )
        with self.assertRaisesRegex(ValueError, "not a canonical"):
            cpu_prefill_maximum_weight_elements_for_m(128)

    def test_immutable_measurement_matrices_are_memoized(self) -> None:
        """Repeated planning calls must not rebuild release geometry manifests."""

        self.assertIs(cpu_prefill_measurements(), cpu_prefill_measurements())
        self.assertIs(gpu_prefill_measurements(), gpu_prefill_measurements())
        before = cpu_prefill_maximum_weight_elements_for_m.cache_info()
        cpu_prefill_maximum_weight_elements_for_m(16384)
        cpu_prefill_maximum_weight_elements_for_m(16384)
        after = cpu_prefill_maximum_weight_elements_for_m.cache_info()
        self.assertGreaterEqual(after.hits - before.hits, 2)

    def test_gpu_matrix_measures_32b_directly_through_m8192(self) -> None:
        matrix = {row.shape.name: row.m_values for row in gpu_prefill_measurements()}

        expected_large = (64, 256, 1024, 2048, 4096, 8192)
        self.assertEqual(matrix["32B_AttnOut"], expected_large)
        self.assertEqual(matrix["32B_QKV"], expected_large)
        self.assertEqual(matrix["32B_FFN_Up"], expected_large)
        self.assertEqual(matrix["32B_FFN_Dn"], expected_large)

    def test_gpu_matrix_keeps_m16384_for_smaller_models(self) -> None:
        matrix = {row.shape.name: row.m_values for row in gpu_prefill_measurements()}

        self.assertEqual(matrix["7B_FFN_Up"], PREFILL_M_BUCKETS)
        self.assertEqual(
            sum(len(row.m_values) for row in gpu_prefill_measurements()),
            498,
        )

    def test_cuda_and_rocm_matrix_measure_qwen36_35b_moe_at_every_depth(self) -> None:
        """The shared GPU inventory is mandatory for both backend collectors."""

        matrix = {row.shape.name: row.m_values for row in gpu_prefill_measurements()}

        for backend in ("cuda", "rocm"):
            with self.subTest(backend=backend):
                for name in QWEN36_35B_MOE_SHAPES:
                    self.assertEqual(matrix[name], PREFILL_M_BUCKETS, name)


if __name__ == "__main__":
    unittest.main()
