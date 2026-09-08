#!/usr/bin/env python3
"""Regression tests for the bounded NativeVNNI GPU measurement plan."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
sys.path.insert(0, str(KERNEL_ROOT))

from native_vnni_dispatch.measurement_plan import load_gpu_measurement_plan
from native_vnni_dispatch.format_registry import FORMAT_SPECS
from native_vnni_dispatch.qwen_release_geometry import qwen_release_geometries
from native_vnni_dispatch.schema import (
    AspectBucket,
    Backend,
    ExecutionMode,
    SemanticContract,
)
from native_vnni_dispatch.shape_manifest import (
    ShapePartition,
    ShapeRole,
    load_shape_manifest,
)


class NativeVNNIGPUMeasurementPlanTest(unittest.TestCase):
    def test_common_plan_is_bounded_and_contains_every_production_shape(self) -> None:
        manifest = load_shape_manifest()
        plan = load_gpu_measurement_plan(manifest=manifest)
        names = plan.common_development_shapes

        self.assertEqual(len(names), len(set(names)))
        self.assertTrue({
            shape.name
            for shape in manifest.shapes
            if shape.role == ShapeRole.PRODUCTION
        }.issubset(names))
        self.assertEqual(
            {manifest.by_name(name).aspect_bucket for name in names},
            set(AspectBucket),
        )
        self.assertTrue(all(
            manifest.by_name(name).fast_partition == ShapePartition.DEVELOPMENT
            and manifest.by_name(name).verifier_partition
            == ShapePartition.DEVELOPMENT
            for name in names
        ))

    def test_q4_refinement_is_cuda_fast_only(self) -> None:
        manifest = load_shape_manifest()
        plan = load_gpu_measurement_plan(manifest=manifest)
        cuda_q4 = plan.scoped_fast_shapes(
            backend=Backend.CUDA,
            source_format="Q4_0",
        )

        self.assertEqual(len(cuda_q4), 141)
        self.assertTrue(all(
            manifest.by_name(name).model_family == "fast-m1-cv-refinement-v5"
            and manifest.by_name(name).verifier_partition is None
            for name in cuda_q4
        ))
        self.assertEqual(
            plan.scoped_fast_shapes(
                backend=Backend.CUDA,
                source_format="Q5_0",
            ),
            (),
        )
        scoped = cuda_q4[0]
        self.assertTrue(plan.fast_shape_applies(
            manifest,
            backend=Backend.CUDA,
            source_format="Q4_0",
            shape_name=scoped,
        ))
        self.assertFalse(plan.fast_shape_applies(
            manifest,
            backend=Backend.CUDA,
            source_format="Q5_0",
            shape_name=scoped,
        ))
        self.assertFalse(plan.fast_shape_applies(
            manifest,
            backend=Backend.ROCM,
            source_format="Q4_0",
            shape_name=scoped,
        ))
        self.assertFalse(plan.verifier_shape_applies(
            manifest,
            shape_name=scoped,
        ))
        self.assertEqual(
            plan.scoped_fast_shapes(
                backend=Backend.ROCM,
                source_format="Q4_0",
            ),
            (),
        )

    def test_plan_digest_binds_resolved_shapes_and_manifest(self) -> None:
        manifest = load_shape_manifest()
        plan = load_gpu_measurement_plan(manifest=manifest)

        self.assertRegex(plan.digest(manifest), r"^sha256:[0-9a-f]{64}$")
        self.assertLess(
            len(plan.common_development_shapes),
            len(manifest.partition_names(
                verifier=False,
                partition=ShapePartition.DEVELOPMENT,
            )),
        )

    def test_physical_surface_inventory_scopes_cuda_q4_extension_exactly(self) -> None:
        """Only CUDA Q4 Fast M1 receives the 141-shape refinement delta."""

        manifest = load_shape_manifest()
        plan = load_gpu_measurement_plan(manifest=manifest)
        common_args = {
            "source_formats": tuple(spec.label for spec in FORMAT_SPECS),
            "execution_modes": (
                ExecutionMode.EAGER,
                ExecutionMode.GRAPH_CAPTURED,
            ),
            "verifier_m_values": frozenset((*range(2, 17), 31)),
        }
        cuda = plan.expected_decode_surfaces(
            manifest,
            backend=Backend.CUDA,
            **common_args,
        )
        rocm = plan.expected_decode_surfaces(
            manifest,
            backend=Backend.ROCM,
            **common_args,
        )
        scoped = plan.scoped_fast_shapes(
            backend=Backend.CUDA,
            source_format="Q4_0",
        )[0]

        self.assertEqual(len(cuda) - len(rocm), 141 * 2)
        self.assertIn((
            SemanticContract.FAST,
            1,
            scoped,
            "Q4_0",
            ExecutionMode.EAGER,
        ), cuda)
        self.assertNotIn((
            SemanticContract.FAST,
            1,
            scoped,
            "Q5_0",
            ExecutionMode.EAGER,
        ), cuda)
        self.assertFalse(any(surface[2] == scoped for surface in rocm))
        self.assertFalse(any(
            surface[2] == scoped
            and surface[0] == SemanticContract.VERIFIER_SERIAL_M1_BITWISE
            for surface in cuda
        ))

    def test_exact_overlays_cover_every_format_mode_and_decode_m(self) -> None:
        """All known geometries own all-format evidence on both GPU backends."""

        manifest = load_shape_manifest()
        plan = load_gpu_measurement_plan(manifest=manifest)
        exact_shapes = {
            shape.name
            for shape in manifest.shapes
            if shape.role == ShapeRole.PRODUCTION and shape.exact_overlay
        }
        source_formats = tuple(spec.label for spec in FORMAT_SPECS)
        modes = (ExecutionMode.EAGER, ExecutionMode.GRAPH_CAPTURED)
        verifier_ms = frozenset((*range(2, 17), 31))
        for backend in (Backend.CUDA, Backend.ROCM):
            surfaces = plan.expected_decode_surfaces(
                manifest,
                backend=backend,
                source_formats=source_formats,
                execution_modes=modes,
                verifier_m_values=verifier_ms,
            )
            with self.subTest(backend=backend.value):
                for shape_name in exact_shapes:
                    for source_format in source_formats:
                        for mode in modes:
                            self.assertIn((
                                SemanticContract.FAST,
                                1,
                                shape_name,
                                source_format,
                                mode,
                            ), surfaces)
                            for m in verifier_ms:
                                self.assertIn((
                                    SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
                                    m,
                                    shape_name,
                                    source_format,
                                    mode,
                                ), surfaces)

    def test_qwen36_release_geometries_resolve_to_exact_overlay_shapes(self) -> None:
        """Every reviewed Qwen 3.6 sidecar dimension reaches the same plan."""

        manifest = load_shape_manifest()
        production_dimensions = {
            (shape.n, shape.k)
            for shape in manifest.shapes
            if shape.role == ShapeRole.PRODUCTION and shape.exact_overlay
        }
        qwen36_dimensions = {
            (geometry.n, geometry.k)
            for geometry in qwen_release_geometries()
            if any(
                use.release_id.startswith("Qwen3.6-")
                for use in geometry.uses
            )
        }
        self.assertTrue(qwen36_dimensions.issubset(production_dimensions))


if __name__ == "__main__":
    unittest.main()
