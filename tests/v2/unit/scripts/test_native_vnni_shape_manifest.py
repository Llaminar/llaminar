#!/usr/bin/env python3
"""Regression tests for the shared NativeVNNI shape/split manifest."""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.schema import AspectBucket  # noqa: E402
from native_vnni_dispatch.shape_manifest import (  # noqa: E402
    ShapePartition,
    ShapeRole,
    load_shape_manifest,
    partition_assignments,
)


class NativeVNNIShapeManifestTest(unittest.TestCase):
    """Lock inventory breadth and the MTP regression geometries."""

    def test_shape_name_lookup_builds_one_reusable_index(self) -> None:
        """Per-observation validation must not rescan the whole manifest."""

        manifest = load_shape_manifest()
        shape_name = manifest.shapes[-1].name
        self.assertNotIn("_shapes_by_name", manifest.__dict__)
        first = manifest.by_name(shape_name)
        self.assertIn("_shapes_by_name", manifest.__dict__)
        self.assertIs(first, manifest.by_name(shape_name))
        with self.assertRaisesRegex(KeyError, "unknown NativeVNNI shape"):
            manifest.by_name("NotAProductionShape")

    def test_every_aspect_has_disjoint_development_and_sealed_breadth(self) -> None:
        """Both semantic surfaces retain independent grouped holdouts."""

        manifest = load_shape_manifest()
        for bucket in AspectBucket:
            shapes = [
                shape for shape in manifest.shapes if shape.aspect_bucket == bucket
            ]
            for field in ("fast_partition", "verifier_partition"):
                development = [
                    shape
                    for shape in shapes
                    if getattr(shape, field) == ShapePartition.DEVELOPMENT
                ]
                sealed = [
                    shape
                    for shape in shapes
                    if getattr(shape, field) == ShapePartition.SEALED
                ]
                self.assertGreaterEqual(len(development), 16, (bucket, field))
                self.assertGreaterEqual(len(sealed), 5, (bucket, field))
            self.assertFalse(
                {
                    shape.name
                    for shape in shapes
                    if shape.fast_partition == ShapePartition.SEALED
                }
                & {
                    shape.name
                    for shape in shapes
                    if shape.verifier_partition == ShapePartition.SEALED
                }
            )

    def test_opened_prefill_shapes_rotate_into_development(self) -> None:
        """Every spent prefill witness leaves a disjoint fresh holdout pool."""

        manifest = load_shape_manifest()
        self.assertEqual(len(manifest.shapes), 594)
        fast_sealed = [
            shape
            for shape in manifest.shapes
            if shape.fast_partition == ShapePartition.SEALED
        ]
        verifier_sealed = [
            shape
            for shape in manifest.shapes
            if shape.verifier_partition == ShapePartition.SEALED
        ]
        self.assertEqual(len(fast_sealed), 24)
        self.assertEqual(len(verifier_sealed), 23)
        fast_kpart_sealed = [
            shape
            for shape in fast_sealed
            if shape.name == "CPUDecodeKPart_Balanced_Sealed_3200x4096"
        ]
        self.assertEqual(len(fast_kpart_sealed), 1)
        fast_grid_sealed = [
            shape for shape in fast_sealed if shape not in fast_kpart_sealed
        ]
        self.assertTrue(
            all(shape.model_family == "sealed-grid-v4" for shape in fast_grid_sealed)
        )
        self.assertTrue(
            all(shape.model_family == "sealed-grid-v4" for shape in verifier_sealed)
        )
        self.assertTrue(
            all(shape.name.startswith("V4FastSealed_") for shape in fast_grid_sealed)
        )
        self.assertTrue(
            all(
                shape.name.startswith("V4VerifierSealed_")
                for shape in verifier_sealed
            )
        )
        self.assertTrue(
            all(
                shape.fast_partition == ShapePartition.DEVELOPMENT
                for shape in manifest.shapes
                if shape.model_family in {"sealed-grid-v2", "sealed-grid-v3"}
            )
        )
        self.assertTrue(
            all(
                shape.verifier_partition == ShapePartition.DEVELOPMENT
                for shape in manifest.shapes
                if shape.model_family in {"sealed-grid-v2", "sealed-grid-v3"}
            )
        )
        prefill_sealed = [
            shape
            for shape in manifest.shapes
            if shape.prefill_partition == ShapePartition.SEALED
        ]
        self.assertEqual(len(prefill_sealed), 8)
        self.assertEqual(
            Counter(shape.model_family for shape in prefill_sealed),
            Counter({
                "cpu-prefill-sealed-v5": 4,
                "cpu-prefill-sealed-v6": 1,
                "cpu-prefill-sealed-v8": 3,
            }),
        )
        self.assertTrue(
            all(
                shape.fast_partition is None
                and shape.verifier_partition is None
                for shape in prefill_sealed
            )
        )
        opened_prefill_v5 = [
            shape
            for shape in manifest.shapes
            if shape.model_family == "cpu-prefill-sealed-v5"
            and shape.prefill_partition == ShapePartition.DEVELOPMENT
        ]
        self.assertEqual(len(opened_prefill_v5), 4)
        opened_prefill_v6 = [
            shape
            for shape in manifest.shapes
            if shape.model_family == "cpu-prefill-sealed-v6"
            and shape.prefill_partition == ShapePartition.DEVELOPMENT
        ]
        self.assertEqual(len(opened_prefill_v6), 3)
        opened_prefill_v7 = [
            shape
            for shape in manifest.shapes
            if shape.model_family == "cpu-prefill-sealed-v7"
            and shape.prefill_partition == ShapePartition.DEVELOPMENT
        ]
        self.assertEqual(len(opened_prefill_v7), 3)
        opened_prefill = [
            shape
            for shape in manifest.shapes
            if shape.model_family == "cpu-prefill-sealed-v3"
        ]
        self.assertEqual(len(opened_prefill), 8)
        self.assertTrue(
            all(
                shape.prefill_partition == ShapePartition.DEVELOPMENT
                for shape in opened_prefill
            )
        )
        prefill_refinement = [
            shape
            for shape in manifest.shapes
            if shape.model_family == "cpu-prefill-cv-refinement-v4"
        ]
        self.assertEqual(len(prefill_refinement), 4)
        self.assertTrue(
            all(
                shape.prefill_partition == ShapePartition.DEVELOPMENT
                for shape in prefill_refinement
            )
        )
        development_grid = [
            shape
            for shape in manifest.shapes
            if shape.model_family == "development-grid-v3"
        ]
        self.assertEqual(len(development_grid), 46)
        self.assertTrue(
            all(
                shape.fast_partition == ShapePartition.DEVELOPMENT
                and shape.verifier_partition == ShapePartition.DEVELOPMENT
                for shape in development_grid
            )
        )

    def test_shapes_do_not_exceed_supported_production_work_envelope(self) -> None:
        """Synthetic grids must not exceed the largest supported projection."""

        manifest = load_shape_manifest()
        self.assertEqual(manifest.maximum_supported_weight_elements, 1271398400)
        self.assertEqual(
            max(shape.work_items for shape in manifest.shapes),
            manifest.maximum_supported_weight_elements,
        )
        self.assertTrue(
            all(
                shape.work_items <= manifest.maximum_supported_weight_elements
                for shape in manifest.shapes
            )
        )

    def test_cpu_measurement_envelope_includes_every_exact_overlay(self) -> None:
        """CPU collection must not omit a known production exact geometry."""

        manifest = load_shape_manifest()
        self.assertEqual(
            manifest.maximum_cpu_measurement_weight_elements,
            manifest.maximum_supported_weight_elements,
        )
        measured_names = set(manifest.cpu_measurement_names(verifier=True))
        self.assertTrue({
            "32B_AttnOut",
            "32B_QKV",
            "32B_FFN_Up",
            "32B_FFN_Dn",
            "32B_LM_Head",
        }.issubset(measured_names))
        self.assertIn("Qwen36_LM_Head", measured_names)
        self.assertTrue({
            shape.name
            for shape in manifest.shapes
            if shape.role == ShapeRole.PRODUCTION and shape.exact_overlay
        }.issubset(measured_names))
        self.assertEqual(
            max(manifest.by_name(name).work_items for name in measured_names),
            manifest.maximum_cpu_measurement_weight_elements,
        )

    def test_qwen25_gqa_projection_geometry_is_not_three_hidden_widths(self) -> None:
        """QKV overlays must include the smaller grouped K/V head widths."""

        manifest = load_shape_manifest()
        expected = {
            "0.5B_QKV": (896 + 2 * 2 * 64, 896),
            "1.5B_QKV": (1536 + 2 * 2 * 128, 1536),
            "3B_QKV": (2048 + 2 * 2 * 128, 2048),
            "7B_QKV": (3584 + 2 * 4 * 128, 3584),
            "32B_QKV": (5120 + 2 * 8 * 128, 5120),
        }
        self.assertEqual(
            {
                name: (manifest.by_name(name).n, manifest.by_name(name).k)
                for name in expected
            },
            expected,
        )

    def test_transition_grid_is_development_only_for_both_contracts(self) -> None:
        """Crossover refinement must never consume a fresh sealed holdout."""

        manifest = load_shape_manifest()
        transition_grid = [
            shape
            for shape in manifest.shapes
            if shape.model_family == "transition-grid-v3"
        ]
        self.assertEqual(len(transition_grid), 20)
        self.assertEqual(
            {shape.aspect_bucket for shape in transition_grid},
            set(AspectBucket),
        )
        for bucket in AspectBucket:
            self.assertEqual(
                sum(
                    shape.aspect_bucket == bucket
                    for shape in transition_grid
                ),
                5,
            )
        self.assertTrue(
            all(
                shape.role == ShapeRole.CERTIFICATION
                and not shape.exact_overlay
                and shape.fast_partition == ShapePartition.DEVELOPMENT
                and shape.verifier_partition == ShapePartition.DEVELOPMENT
                for shape in transition_grid
            )
        )

    def test_transition_refinement_is_development_only(self) -> None:
        """A measured crossover regression must not consume sealed evidence."""

        refinement = [
            shape
            for shape in load_shape_manifest().shapes
            if shape.model_family == "transition-refinement-v3"
        ]
        self.assertEqual(
            [(shape.n, shape.k) for shape in refinement],
            [(840, 2624), (848, 2656), (856, 2688)],
        )
        self.assertTrue(
            all(
                shape.role == ShapeRole.CERTIFICATION
                and not shape.exact_overlay
                and shape.aspect_bucket == AspectBucket.TALL
                and shape.fast_partition == ShapePartition.DEVELOPMENT
                and shape.verifier_partition == ShapePartition.DEVELOPMENT
                for shape in refinement
            )
        )

    def test_v4_transition_neighborhoods_cover_every_measured_failure_region(self) -> None:
        """The new lattice is cross-format development evidence, not Q4 overlays."""

        neighborhoods = [
            shape
            for shape in load_shape_manifest().shapes
            if shape.model_family == "transition-neighborhood-v4"
        ]
        self.assertEqual(len(neighborhoods), 32)
        self.assertEqual(
            {
                bucket: sum(shape.aspect_bucket == bucket for shape in neighborhoods)
                for bucket in AspectBucket
            },
            {
                AspectBucket.TALL: 11,
                AspectBucket.BALANCED: 13,
                AspectBucket.WIDE: 8,
                AspectBucket.VERY_WIDE: 0,
            },
        )
        self.assertTrue(
            all(
                shape.role == ShapeRole.CERTIFICATION
                and not shape.exact_overlay
                and shape.fast_partition == ShapePartition.DEVELOPMENT
                and shape.verifier_partition == ShapePartition.DEVELOPMENT
                for shape in neighborhoods
            )
        )

    def test_fast_m1_refinement_is_systematic_and_excluded_from_verifier(self) -> None:
        """Confirmed Fast misses gain local evidence without verifier expansion."""

        manifest = load_shape_manifest()
        refinement = [
            shape
            for shape in manifest.shapes
            if shape.model_family == "fast-m1-cv-refinement-v5"
        ]
        self.assertEqual(len(refinement), 141)
        self.assertEqual(
            {
                bucket: sum(shape.aspect_bucket == bucket for shape in refinement)
                for bucket in AspectBucket
            },
            {
                AspectBucket.TALL: 52,
                AspectBucket.BALANCED: 33,
                AspectBucket.WIDE: 56,
                AspectBucket.VERY_WIDE: 0,
            },
        )
        self.assertTrue(
            all(
                shape.role == ShapeRole.CERTIFICATION
                and not shape.exact_overlay
                and shape.fast_partition == ShapePartition.DEVELOPMENT
                and shape.verifier_partition is None
                for shape in refinement
            )
        )

        anchors = {
            (2048, 11008),
            (11008, 2048),
            (1024, 5472),
            (576, 2304),
            (2304, 2304),
            (1152, 5760),
            (6144, 1024),
            (1792, 2048),
            (2304, 2560),
            (736, 992),
            (9984, 1280),
            (928, 2944),
            (57344, 7168),
            (8064, 768),
            (672, 1856),
            (5632, 1408),
            (6656, 1664),
            (1728, 1856),
        }
        preexisting_dimensions = {
            (shape.n, shape.k)
            for shape in manifest.shapes
            if shape.model_family != "fast-m1-cv-refinement-v5"
        }
        expected_dimensions: set[tuple[int, int]] = set()
        for anchor_n, anchor_k in anchors:
            for delta in (-128, -64, 64, 128):
                expected_dimensions.add((anchor_n + delta, anchor_k))
                expected_dimensions.add((anchor_n, anchor_k + delta))
        expected_dimensions -= preexisting_dimensions
        self.assertEqual(
            {(shape.n, shape.k) for shape in refinement},
            expected_dimensions,
        )

    def test_fast_m1_kpart_refinement_fills_cartesian_neighborhood(self) -> None:
        """K-part CV must see N/K interactions instead of two isolated axes."""

        manifest = load_shape_manifest()
        refinement = [
            shape
            for shape in manifest.shapes
            if shape.model_family == "fast-m1-kpart-cartesian-v6"
        ]
        n_neighbors = {1920, 1984, 2112, 2176}
        k_neighbors = {10880, 10944, 11072, 11136}
        self.assertEqual(len(refinement), 16)
        self.assertEqual(
            {(shape.n, shape.k) for shape in refinement},
            {
                (n, k)
                for n in n_neighbors
                for k in k_neighbors
            },
        )
        self.assertTrue(
            all(
                shape.role == ShapeRole.CERTIFICATION
                and not shape.exact_overlay
                and shape.aspect_bucket == AspectBucket.TALL
                and shape.fast_partition == ShapePartition.DEVELOPMENT
                and shape.verifier_partition is None
                and shape.prefill_partition is None
                for shape in refinement
            )
        )

        refinement_names = {shape.name for shape in refinement}
        self.assertTrue(
            refinement_names.issubset(
                manifest.partition_names(
                    verifier=False,
                    partition=ShapePartition.DEVELOPMENT,
                )
            )
        )
        verifier_names = set(
            manifest.partition_names(
                verifier=True,
                partition=ShapePartition.DEVELOPMENT,
            )
        ) | set(
            manifest.partition_names(
                verifier=True,
                partition=ShapePartition.SEALED,
            )
        )
        self.assertTrue(refinement_names.isdisjoint(verifier_names))

        shape = refinement[0]
        with self.assertRaisesRegex(
            ValueError,
            "not applicable to the verifier semantic-contract surface",
        ):
            partition_assignments(
                (("refinement-group", shape.name),),
                verifier=True,
                manifest=manifest,
            )

    def test_fast_m1_kpart_dense_refinement_resolves_narrow_boundaries(self) -> None:
        """All-format M=1 CV owns 32-value and isolated-miss neighborhoods."""

        manifest = load_shape_manifest()
        refinement = [
            shape
            for shape in manifest.shapes
            if shape.model_family == "fast-m1-kpart-dense-v7"
        ]
        inner_n = {1984, 2016, 2048, 2080, 2112}
        inner_k = {10944, 10976, 11008, 11040, 11072}
        prior_grid = {
            (n, k)
            for n in {1984, 2048, 2112}
            for k in {10944, 11008, 11072}
        }
        dense_3b = {
            (n, k) for n in inner_n for k in inner_k
        } - prior_grid
        balanced = {
            (n, k)
            for n in {3008, 3072, 3136}
            for k in {4032, 4096, 4160}
        } - {(3072, 4096)}

        self.assertEqual(len(refinement), 24)
        self.assertEqual(
            {(shape.n, shape.k) for shape in refinement},
            dense_3b | balanced,
        )
        self.assertEqual(
            {
                bucket: sum(
                    shape.aspect_bucket == bucket for shape in refinement
                )
                for bucket in AspectBucket
            },
            {
                AspectBucket.TALL: 20,
                AspectBucket.BALANCED: 4,
                AspectBucket.WIDE: 0,
                AspectBucket.VERY_WIDE: 0,
            },
        )
        self.assertTrue(
            all(
                shape.role == ShapeRole.CERTIFICATION
                and not shape.exact_overlay
                and shape.fast_partition == ShapePartition.DEVELOPMENT
                and shape.verifier_partition is None
                and shape.prefill_partition is None
                for shape in refinement
            )
        )

    def test_fast_m1_kpart_transition_lattice_is_cartesian_complete(self) -> None:
        """Coherent CV folds must not inherit holes in the 3B transition grid."""

        manifest = load_shape_manifest()
        expected_n = {1920, 1984, 2016, 2048, 2080, 2112, 2176}
        expected_k = {10880, 10944, 10976, 11008, 11040, 11072, 11136}
        lattice_shapes = [
            shape
            for shape in manifest.shapes
            if shape.n in expected_n
            and shape.k in expected_k
        ]
        lattice = {(shape.n, shape.k) for shape in lattice_shapes}
        completion = [
            shape
            for shape in manifest.shapes
            if shape.model_family == "fast-m1-kpart-grid-completion-v8"
        ]

        self.assertEqual(
            lattice,
            {(n, k) for n in expected_n for k in expected_k},
        )
        self.assertTrue(
            all(
                shape.fast_partition == ShapePartition.DEVELOPMENT
                for shape in lattice_shapes
            )
        )
        self.assertEqual(len(completion), 8)
        self.assertTrue(
            all(
                shape.role == ShapeRole.CERTIFICATION
                and not shape.exact_overlay
                and shape.aspect_bucket == AspectBucket.TALL
                and shape.fast_partition == ShapePartition.DEVELOPMENT
                and shape.verifier_partition is None
                and shape.prefill_partition is None
                for shape in completion
            )
        )

    def test_fast_m1_kpart_v9_midk_support_is_all_format_development(self) -> None:
        """The 2048x8192 miss owns a compact two-dimensional support lattice."""

        manifest = load_shape_manifest()
        refinement = [
            shape
            for shape in manifest.shapes
            if shape.model_family == "fast-m1-kpart-midk-support-v9"
        ]
        expected = {
            (n, k)
            for n in (2016, 2032, 2048)
            for k in (8128, 8192, 8256)
        } - {(2048, 8192)}

        self.assertEqual({(shape.n, shape.k) for shape in refinement}, expected)
        self.assertTrue(all(
            shape.role == ShapeRole.CERTIFICATION
            and not shape.exact_overlay
            and shape.aspect_bucket == AspectBucket.TALL
            and shape.fast_partition == ShapePartition.DEVELOPMENT
            and shape.verifier_partition is None
            and shape.prefill_partition is None
            for shape in refinement
        ))

    def test_fast_m1_kpart_v9_terminal_support_crosses_both_grids(self) -> None:
        """Terminal K-part support spans N-chunk and K-partition boundaries."""

        manifest = load_shape_manifest()
        refinement = [
            shape
            for shape in manifest.shapes
            if shape.model_family == "fast-m1-kpart-terminal-support-v9"
        ]
        expected = (
            {
                (n, k)
                for n in (2000, 2032, 2048)
                for k in (11168, 11200, 11232, 11264)
            }
            | {(2016, 11200), (2016, 11264)}
        )

        self.assertEqual({(shape.n, shape.k) for shape in refinement}, expected)
        self.assertTrue(all(
            shape.role == ShapeRole.CERTIFICATION
            and not shape.exact_overlay
            and shape.aspect_bucket == AspectBucket.TALL
            and shape.fast_partition == ShapePartition.DEVELOPMENT
            and shape.verifier_partition is None
            and shape.prefill_partition is None
            for shape in refinement
        ))

    def test_shape_without_any_semantic_surface_is_rejected(self) -> None:
        """Explicit null may scope a shape, but may not orphan it entirely."""

        source = Path(
            REPO_ROOT
            / "tests/v2/performance/kernels/native_vnni_dispatch/manifests"
            / "native_vnni_decode_shapes_v5.json"
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["shapes"][0]["fast_partition"] = None
        payload["shapes"][0]["verifier_partition"] = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "orphaned.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "at least one semantic-contract surface must apply",
            ):
                load_shape_manifest(path)

    def test_duplicate_dimensions_are_rejected_before_sweep(self) -> None:
        """Two names cannot collapse onto one exact runtime dispatch key."""

        manifest = load_shape_manifest()
        source = Path(
            REPO_ROOT
            / "tests/v2/performance/kernels/native_vnni_dispatch/manifests"
            / "native_vnni_decode_shapes_v5.json"
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        duplicate = dict(payload["shapes"][0])
        duplicate["name"] = "DuplicateDimensionsRegression"
        payload["shapes"].append(duplicate)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate.*dimensions"):
                load_shape_manifest(path)
        self.assertTrue(manifest.shapes)

    def test_failed_cuda_sweep_geometries_are_generic_certification_shapes(self) -> None:
        """The canonical suite's small shapes must not be exact-only overlays."""

        manifest = load_shape_manifest()
        expected = {
            (192, 256),
            (128, 256),
            (384, 512),
        }
        represented = {
            (shape.n, shape.k)
            for shape in manifest.shapes
            if shape.role == ShapeRole.CERTIFICATION and not shape.exact_overlay
        }
        self.assertTrue(expected.issubset(represented))

    def test_all_model_shapes_remain_exact_overlay_obligations(self) -> None:
        """Synthetic generic training cannot replace known model decisions."""

        manifest = load_shape_manifest()
        production = [
            shape for shape in manifest.shapes if shape.role == ShapeRole.PRODUCTION
        ]
        self.assertEqual(len(production), 86)
        self.assertTrue(all(shape.exact_overlay for shape in production))

    def test_manifest_digest_is_stable_and_content_addressed(self) -> None:
        """Generated policy provenance must bind the complete reviewed split."""

        manifest = load_shape_manifest()
        self.assertRegex(manifest.digest(), r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(manifest.digest(), load_shape_manifest().digest())

    def test_all_backend_trainers_share_one_overlay_inventory(self) -> None:
        """CPU, CUDA, and ROCm may not name private shape catalog paths."""

        cmake = (REPO_ROOT / "tests/v2/CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        self.assertEqual(cmake.count("native_vnni_decode_shapes_v5.json"), 1)
        self.assertEqual(
            cmake.count("qwen35_qwen36_release_models_v1.json"),
            1,
        )
        definition_blocks = {
            match.group("target"): match.group("body")
            for match in re.finditer(
                r"target_compile_definitions\(\s*(?P<target>\S+)\s+"
                r"(?P<body>.*?)\)",
                cmake,
                re.DOTALL,
            )
        }
        expected_shared_targets = {
            "v2_perf_cpu_native_vnni_gemv",
            "v2_perf_cuda_native_vnni_decode_trainer",
            "v2_perf_cuda_native_vnni_gemm",
            "v2_perf_native_vnni_sweep",
            "v2_perf_native_vnni_throughput",
        }
        shared_targets = {
            target
            for target, body in definition_blocks.items()
            if "LLAMINAR_NATIVE_VNNI_SHAPE_MANIFEST_PATH" in body
        }
        self.assertSetEqual(shared_targets, expected_shared_targets)
        for target in expected_shared_targets:
            self.assertIn(
                "LLAMINAR_NATIVE_VNNI_QWEN_RELEASE_CATALOG_PATH",
                definition_blocks[target],
                target,
            )

        # The production MoE prefill harness consumes the same release-model
        # catalog plus its centralized real-GGUF codebook-pattern manifest. It
        # is intentionally not a generic shape-manifest trainer.
        catalog_only_targets = {
            target
            for target, body in definition_blocks.items()
            if "LLAMINAR_NATIVE_VNNI_QWEN_RELEASE_CATALOG_PATH" in body
        } - shared_targets
        self.assertSetEqual(
            catalog_only_targets,
            {"v2_perf_moe_verifier_prefill"},
        )
        self.assertIn(
            "LLAMINAR_NATIVE_VNNI_MOE_GGUF_PATTERN_MANIFEST_PATH",
            definition_blocks["v2_perf_moe_verifier_prefill"],
        )


if __name__ == "__main__":
    unittest.main()
