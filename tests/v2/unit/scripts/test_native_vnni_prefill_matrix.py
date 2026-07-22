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
    CPU_LARGE_MODEL_PREFILL_M_BUCKETS,
    CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS,
    CPU_PREFILL_M_BUCKETS,
    CPU_SMALL_MODEL_PREFILL_M_BUCKETS,
    GPU_PREFILL_M_BUCKETS,
    QWEN36_35B_MOE_SHAPES,
    cpu_prefill_maximum_weight_elements_for_m,
    cpu_prefill_measurements,
    gpu_prefill_measurements,
)
from native_vnni_dispatch.qwen_release_geometry import (  # noqa: E402
    qwen_release_geometries,
)
from native_vnni_dispatch.shape_manifest import (  # noqa: E402
    ShapeRole,
    load_shape_manifest,
)


class NativeVNNIPrefillMatrixTest(unittest.TestCase):
    """Lock projection applicability and comprehensive exact-overlay M rows."""

    def test_cpu_exact_projection_depths_follow_model_size_tier(self) -> None:
        matrix = {row.shape.name: row.m_values for row in cpu_prefill_measurements()}

        self.assertEqual(
            matrix["7B_FFN_Up"], CPU_SMALL_MODEL_PREFILL_M_BUCKETS
        )
        self.assertEqual(
            matrix["14B_FFN_Up"], CPU_LARGE_MODEL_PREFILL_M_BUCKETS
        )
        self.assertEqual(
            matrix["32B_FFN_Up"], CPU_LARGE_MODEL_PREFILL_M_BUCKETS
        )

    def test_large_attention_geometry_stops_at_m128(self) -> None:
        matrix = {row.shape.name: row for row in cpu_prefill_measurements()}

        self.assertEqual(
            matrix["32B_AttnOut"].m_values,
            CPU_LARGE_MODEL_PREFILL_M_BUCKETS,
        )
        self.assertEqual(
            matrix["32B_QKV"].m_values,
            CPU_LARGE_MODEL_PREFILL_M_BUCKETS,
        )
        self.assertEqual(matrix["32B_AttnOut"].shape.n, 5120)

    def test_shared_release_geometry_uses_largest_owning_model_tier(self) -> None:
        """A geometry used by a large release must obey the large CPU cap."""

        matrix = {row.shape.name: row.m_values for row in cpu_prefill_measurements()}

        self.assertEqual(
            matrix["Qwen35Release_32x2048"],
            CPU_LARGE_MODEL_PREFILL_M_BUCKETS,
        )
        self.assertEqual(
            matrix["Qwen35Release_16x2048"],
            CPU_SMALL_MODEL_PREFILL_M_BUCKETS,
        )

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

    def test_all_backends_share_every_applicable_exact_overlay(self) -> None:
        """CPU, CUDA, and ROCm must sweep one identical production matrix."""

        manifest = load_shape_manifest()
        release_prefill_dimensions = {
            (geometry.n, geometry.k)
            for geometry in qwen_release_geometries()
            if any(use.projection != "lm_head" for use in geometry.uses)
        }
        expected = {
            (shape.n, shape.k)
            for shape in manifest.shapes
            if shape.role == ShapeRole.PRODUCTION
            and shape.exact_overlay
            and "LM_Head" not in shape.name
            and (
                shape.model_family != "qwen35-qwen36-release-geometries"
                or (shape.n, shape.k) in release_prefill_dimensions
            )
        }
        for backend, measurements in (
            ("cpu", cpu_prefill_measurements()),
            ("cuda", gpu_prefill_measurements()),
            ("rocm", gpu_prefill_measurements()),
        ):
            with self.subTest(backend=backend):
                self.assertEqual(
                    {(row.shape.n, row.shape.k) for row in measurements},
                    expected,
                )
                expected_m_values = (
                    GPU_PREFILL_M_BUCKETS
                    if backend in {"cuda", "rocm"}
                    else None
                )
                if expected_m_values is not None:
                    self.assertTrue(all(
                        row.m_values == expected_m_values
                        for row in measurements
                    ))

    def test_matrix_has_every_shape_depth_cell(self) -> None:
        matrix = cpu_prefill_measurements()

        self.assertEqual(
            sum(len(row.m_values) for row in matrix),
            224,
        )
        self.assertEqual(
            sum(
                row.m_values == CPU_LARGE_MODEL_PREFILL_M_BUCKETS
                for row in matrix
            ),
            38,
        )

    def test_cpu_matrix_measures_qwen36_35b_moe_at_every_depth(self) -> None:
        """CPU receives exact overlays for every active MoE projection."""

        matrix = {row.shape.name: row.m_values for row in cpu_prefill_measurements()}

        for name in QWEN36_35B_MOE_SHAPES:
            self.assertEqual(
                matrix[name], CPU_LARGE_MODEL_PREFILL_M_BUCKETS, name
            )

    def test_cpu_adaptive_work_envelope_matches_every_m(self) -> None:
        """Large projections stop at 128; smaller shapes stop at 512."""

        self.assertEqual(
            cpu_prefill_maximum_weight_elements_for_m(128),
            CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS,
        )
        self.assertEqual(
            cpu_prefill_maximum_weight_elements_for_m(512),
            18_944 * 3_584,
        )
        for unsupported_m in (1024, 2048, 4096):
            with self.subTest(m=unsupported_m):
                with self.assertRaisesRegex(ValueError, "not a canonical"):
                    cpu_prefill_maximum_weight_elements_for_m(unsupported_m)

    def test_immutable_measurement_matrices_are_memoized(self) -> None:
        """Repeated planning calls must not rebuild release geometry manifests."""

        self.assertIs(cpu_prefill_measurements(), cpu_prefill_measurements())
        self.assertIs(gpu_prefill_measurements(), gpu_prefill_measurements())
        before = cpu_prefill_maximum_weight_elements_for_m.cache_info()
        cpu_prefill_maximum_weight_elements_for_m(512)
        cpu_prefill_maximum_weight_elements_for_m(512)
        after = cpu_prefill_maximum_weight_elements_for_m.cache_info()
        self.assertGreaterEqual(after.hits - before.hits, 2)

    def test_gpu_matrix_measures_32b_at_every_canonical_m(self) -> None:
        matrix = {row.shape.name: row.m_values for row in gpu_prefill_measurements()}

        self.assertEqual(matrix["32B_AttnOut"], GPU_PREFILL_M_BUCKETS)
        self.assertEqual(matrix["32B_QKV"], GPU_PREFILL_M_BUCKETS)
        self.assertEqual(matrix["32B_FFN_Up"], GPU_PREFILL_M_BUCKETS)
        self.assertEqual(matrix["32B_FFN_Dn"], GPU_PREFILL_M_BUCKETS)

    def test_gpu_matrix_keeps_m16384_for_smaller_models(self) -> None:
        matrix = {row.shape.name: row.m_values for row in gpu_prefill_measurements()}

        self.assertEqual(matrix["7B_FFN_Up"], GPU_PREFILL_M_BUCKETS)
        self.assertEqual(
            sum(len(row.m_values) for row in gpu_prefill_measurements()),
            len(gpu_prefill_measurements()) * len(GPU_PREFILL_M_BUCKETS),
        )

    def test_cuda_and_rocm_matrix_measure_qwen36_35b_moe_at_every_depth(self) -> None:
        """The shared GPU inventory is mandatory for both backend collectors."""

        matrix = {row.shape.name: row.m_values for row in gpu_prefill_measurements()}

        for backend in ("cuda", "rocm"):
            with self.subTest(backend=backend):
                for name in QWEN36_35B_MOE_SHAPES:
                    self.assertEqual(
                        matrix[name], GPU_PREFILL_M_BUCKETS, name
                    )


if __name__ == "__main__":
    unittest.main()
