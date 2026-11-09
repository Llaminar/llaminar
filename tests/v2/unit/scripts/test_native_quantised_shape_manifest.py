#!/usr/bin/env python3
"""Regression tests for the shared NativeQuantised shape/split manifest.

Production trainers and device-free diagnostic parsers share the CMake-owned
catalog bindings. Injected test documents do not authorize a private catalog
path or change the release inventory used by a compiled consumer.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_quantised_dispatch.schema import AspectBucket  # noqa: E402
from native_quantised_dispatch.shape_manifest import (  # noqa: E402
    ExpertProjectionLayout,
    ExpertProjectionRole,
    ShapePartition,
    ShapeRole,
    load_shape_manifest,
    expert_projection_geometries,
    partition_assignments,
)


class NativeQuantisedShapeManifestTest(unittest.TestCase):
    """Lock inventory breadth and the MTP regression geometries."""

    def test_shape_name_lookup_builds_one_reusable_index(self) -> None:
        """Per-observation validation must not rescan the whole manifest."""

        manifest = load_shape_manifest()
        shape_name = manifest.shapes[-1].name
        self.assertNotIn("_shapes_by_name", manifest.__dict__)
        first = manifest.by_name(shape_name)
        self.assertIn("_shapes_by_name", manifest.__dict__)
        self.assertIs(first, manifest.by_name(shape_name))
        with self.assertRaisesRegex(KeyError, "unknown NativeQuantised shape"):
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
        self.assertEqual(len(manifest.shapes), 639)
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
            / "tests/v2/performance/kernels/native_quantised_dispatch/manifests"
            / "native_quantised_decode_shapes_v5.json"
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
            / "tests/v2/performance/kernels/native_quantised_dispatch/manifests"
            / "native_quantised_decode_shapes_v5.json"
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
        self.assertEqual(len(production), 131)
        self.assertTrue(all(shape.exact_overlay for shape in production))
        tp_local = {
            shape.name: (shape.n, shape.k)
            for shape in production
            if shape.model_family == "qwen35-moe-tp-local"
        }
        self.assertEqual(
            tp_local,
            {
                "Qwen35MoE_122B_TP2_AttnKVProjection": (256, 3072),
                "Qwen35MoE_122B_TP2_GDNZProjection": (4096, 3072),
                "Qwen35MoE_122B_TP2_AttnGDNOutputProjection": (3072, 4096),
                "Qwen35MoE_122B_TP4_AttnKVProjection": (128, 3072),
                "Qwen35MoE_122B_TP4_GDNZProjection": (2048, 3072),
                "Qwen35MoE_122B_TP4_AttnGDNOutputProjection": (3072, 2048),
            },
        )
        exact_dimensions = {
            (shape.n, shape.k)
            for shape in production
            if shape.exact_overlay
        }
        expected_by_degree = {
            2: {(256, 3072), (4096, 3072), (3072, 4096)},
            4: {(128, 3072), (2048, 3072), (3072, 2048)},
            8: {(64, 3072), (1024, 3072), (3072, 1024)},
        }
        for degree, dimensions in expected_by_degree.items():
            with self.subTest(tp_degree=degree):
                self.assertTrue(dimensions.issubset(exact_dimensions))

    def test_qfn_native_geometry_obligations_reuse_existing_exact_keys(self) -> None:
        """Own physical P1/TP2/TP4 dimensions without installing a runtime policy."""
        manifest = load_shape_manifest()
        production = {(s.n, s.k): s for s in manifest.shapes
                      if s.role == ShapeRole.PRODUCTION}
        required = {
            (4, 10240),
            (160, 2560),
            (256, 2560),
            (320, 2560),
            (320, 10240),
            (512, 2560),
            (640, 640),
            (640, 2560),
            (1280, 640),
            (1536, 2560),
            (2560, 160),
            (2560, 320),
            (2560, 640),
            (2560, 1536),
            (2560, 2560),
            (2560, 3072),
            (2560, 5120),
            (2560, 6144),
            (3072, 2560),
            (5120, 2560),
            (6144, 2560),
            (10240, 2560),
            (12288, 2560),
            (248320, 2560),
        }
        self.assertTrue(required.issubset(production))
        self.assertEqual(len(production), len([s for s in manifest.shapes
                         if s.role == ShapeRole.PRODUCTION]))
        self.assertEqual(production[(248320, 2560)].model_family,
                         "qwen35-qwen36-release-geometries")
        self.assertEqual(production[(2560, 5120)].model_family,
                         "qwen35-qwen36-release-geometries")
        for shape in manifest.shapes:
            if shape.model_family == "qwen4exp":
                self.assertTrue(shape.exact_overlay)
                self.assertEqual(shape.fast_partition, ShapePartition.DEVELOPMENT)
                self.assertEqual(shape.verifier_partition, ShapePartition.DEVELOPMENT)
                self.assertIsNone(shape.prefill_partition)
                self.assertEqual(shape.k % 32, 0)
        qfn = {(s.n, s.k) for s in manifest.shapes if s.model_family == "qwen4exp"}
        # Current HC-up TwoTerm and floating-only alpha/beta/indexer do
        # not become mislabeled SingleQ8 obligations. KV512 is independently
        # applicable even though a floating indexer also uses that geometry.
        self.assertIn((4, 10240), qfn)  # Supported predictor HC injection uses Q8_0.
        self.assertFalse(qfn & {(10240, 320), (48, 2560),
                                (24, 2560), (12, 2560), (128, 2560)})

    def test_qfn_tp8_and_aligned_expert_tails_are_exact_overlay_obligations(self) -> None:
        """Cover equal TP1/2/4/8 and their actual native-block remainder cuts.

        The immutable QFN geometry is H2560/F640, Q24/KV2/D256 and
        GDN K16/V48/D128. Qwen4ExpSchema uses query-interval KV coverage;
        TensorParallelConfig and MoEExpertShardCohort cut F in blocks of 32.
        This is a finite source-geometry witness, not arbitrary proportional
        topology coverage or a second production/backend shape inventory.
        """

        manifest = load_shape_manifest()
        production = {(s.n, s.k): s for s in manifest.shapes
                      if s.role == ShapeRole.PRODUCTION and s.exact_overlay}
        hidden, intermediate, vocab = 2560, 640, 248320
        query_heads, kv_heads, head_dim = 24, 2, 256
        gdn_key_heads, gdn_value_heads, gdn_dim = 16, 48, 128
        query_heads_per_kv = query_heads // kv_heads
        for degree in (1, 2, 4, 8):
            local_query_heads = query_heads // degree
            local_gdn_keys = gdn_key_heads * local_query_heads // query_heads
            local_gdn_values = gdn_value_heads * local_query_heads // query_heads
            blocks, remainder = divmod(intermediate // 32, degree)
            intermediate_widths = tuple(
                32 * (blocks + (participant < remainder))
                for participant in range(degree)
            )
            local_kv_widths = {
                (((participant + 1) * local_query_heads - 1)
                 // query_heads_per_kv - participant * local_query_heads
                 // query_heads_per_kv + 1) * head_dim
                for participant in range(degree)
            }
            required = {
                (2 * local_query_heads * head_dim, hidden),  # fused Q/gate
                ((2 * local_gdn_keys + local_gdn_values) * gdn_dim, hidden),
                (local_gdn_values * gdn_dim, hidden),  # GDN Z
                (hidden, local_query_heads * head_dim),  # QSA output
                (hidden, local_gdn_values * gdn_dim),  # GDN output
                (hidden // degree, intermediate),  # owned down columns
                (vocab // degree, hidden),
                *((width, hidden) for width in local_kv_widths),
                *((width, hidden) for width in intermediate_widths),
                *((hidden, width) for width in intermediate_widths),
            }
            with self.subTest(tp_degree=degree):
                self.assertFalse(required - production.keys(),
                                 f"missing QFN TP{degree}: {sorted(required - production.keys())}")
                names = manifest.exact_overlay_names_for_dimensions(sorted(required))
                self.assertEqual(len(names), len(required))
                self.assertEqual(local_kv_widths, {512} if degree == 1 else {256})
                self.assertTrue(all(width % 32 == 0 for width in intermediate_widths))
                self.assertEqual(sum(intermediate_widths), intermediate)
                if degree == 8:
                    self.assertEqual(intermediate_widths, (96,) * 4 + (64,) * 4)
                    self.assertNotIn(80, intermediate_widths)
        # The current MTP sidecar is replicated and the terminal policy can
        # mirror the full vocabulary. Neither loses its unsharded obligation.
        self.assertIn((2560, 5120), production)
        self.assertIn((248320, 2560), production)

    def test_expert_catalog_preserves_operation_and_original_slice_identity(self) -> None:
        """Typed expert ownership survives physical-key deduplication."""

        manifest = load_shape_manifest()
        projections = expert_projection_geometries(manifest)
        self.assertEqual(len(projections), 58)
        self.assertEqual(projections, expert_projection_geometries(manifest))
        self.assertEqual({p.model_family for p in projections}, {"qwen4exp"})
        self.assertEqual({p.role for p in projections}, set(ExpertProjectionRole))
        self.assertEqual({p.layout for p in projections}, set(ExpertProjectionLayout))
        self.assertEqual(len({(p.physical_n, p.physical_k) for p in projections}), 13)
        for projection in projections:
            with self.subTest(projection=projection):
                shape = manifest.by_name(projection.shape_name)
                self.assertEqual((shape.n, shape.k),
                                 (projection.physical_n, projection.physical_k))
                self.assertEqual(projection.policy_n, projection.physical_n)
                self.assertEqual((projection.source_n, projection.source_k),
                                 (640, 2560) if projection.role == ExpertProjectionRole.GATE_UP
                                 else (2560, 640))
        for degree in (2, 4, 8):
            for role in ExpertProjectionRole:
                shards = [p for p in projections if p.tp_degree == degree
                          and p.layout == ExpertProjectionLayout.INTERMEDIATE_SHARD
                          and p.role == role]
                self.assertEqual([p.participant for p in shards], list(range(degree)))
                frontier = 0
                for shard in shards:
                    self.assertEqual(shard.intermediate_begin, frontier)
                    self.assertEqual(shard.intermediate_width % 32, 0)
                    if role == ExpertProjectionRole.GATE_UP:
                        self.assertEqual((shard.first_n, shard.first_k), (frontier, 0))
                    else:
                        self.assertEqual((shard.first_n, shard.first_k), (0, frontier))
                    frontier += shard.intermediate_width
                self.assertEqual(frontier, 640)
                if degree == 8:
                    self.assertEqual([p.intermediate_width for p in shards],
                                     [96] * 4 + [64] * 4)
            columns = [p for p in projections if p.tp_degree == degree
                       and p.layout == ExpertProjectionLayout.DOWN_COLUMNS
                       and p.role == ExpertProjectionRole.DOWN]
            self.assertEqual([p.first_n for p in columns],
                             [p * (2560 // degree) for p in range(degree)])
            self.assertEqual(sum(p.physical_n for p in columns), 2560)
            self.assertTrue(all(p.physical_k == 640 for p in columns))

        # Canonical physical aliases may belong to another release. The source
        # operation still comes from typed ownership, not that alias's family.
        original = manifest.by_name("Qwen4Exp_320x2560")
        alias = replace(original, name="SharedReleaseGateKey",
                        model_family="qwen35-qwen36-release-geometries")
        aliased = replace(manifest, shapes=tuple(alias if s == original else s
                                                 for s in manifest.shapes))
        resolved = [p for p in expert_projection_geometries(aliased)
                    if p.physical_n == 320 and p.physical_k == 2560]
        self.assertTrue(resolved)
        self.assertTrue(all(p.shape_name == alias.name and p.model_family == "qwen4exp"
                            and p.role == ExpertProjectionRole.GATE_UP for p in resolved))
        missing = replace(manifest, shapes=tuple(s for s in manifest.shapes if s != original))
        with self.assertRaisesRegex(ValueError, "geometry 320x2560 resolves to 0"):
            expert_projection_geometries(missing)
        duplicate = replace(manifest, shapes=(*manifest.shapes, alias))
        with self.assertRaisesRegex(ValueError, "geometry 320x2560 resolves to 2"):
            expert_projection_geometries(duplicate)
        with self.assertRaisesRegex(ValueError, "source-owned expert projection"):
            replace(projections[0], first_n=641)
        with self.assertRaisesRegex(ValueError, "source-owned expert projection"):
            replace(projections[0], policy_n=projections[0].source_n + 32)

    def test_manifest_digest_is_stable_and_content_addressed(self) -> None:
        """Generated policy provenance must bind the complete reviewed split."""

        manifest = load_shape_manifest()
        self.assertRegex(manifest.digest(), r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(manifest.digest(), load_shape_manifest().digest())

    def test_dense_27b_tp2_projection_geometry_is_complete(self) -> None:
        """Column and reduction shards must share one all-backend exact inventory."""

        manifest = load_shape_manifest()
        expected = {
            "QwenDense27B_TP2_FFN_GateUp": (8704, 5120),
            "QwenDense27B_TP2_FFN_DownProjection": (5120, 8704),
            "QwenDense27B_TP2_GDN_ZProjection": (3072, 5120),
            "QwenDense27B_TP2_GDN_OutputProjection": (5120, 3072),
            "QwenDense27B_TP2_AttnKVProjection": (512, 5120),
            "QwenDense27B_TP2_LM_Head": (124160, 5120),
        }
        for name, dimensions in expected.items():
            with self.subTest(shape=name):
                shape = manifest.by_name(name)
                self.assertEqual((shape.n, shape.k), dimensions)
                self.assertEqual(shape.role, ShapeRole.PRODUCTION)
                self.assertTrue(shape.exact_overlay)
                self.assertEqual(shape.fast_partition, ShapePartition.DEVELOPMENT)
                self.assertEqual(shape.verifier_partition, ShapePartition.DEVELOPMENT)
        # QKV and fused query/gate shards already have these physical keys.
        # Reuse those entries so aliases cannot choose conflicting dispatches.
        production_dimensions = {
            (shape.n, shape.k) for shape in manifest.shapes
            if shape.role == ShapeRole.PRODUCTION and shape.exact_overlay
        }
        self.assertTrue({(5120, 5120), (6144, 5120)}.issubset(production_dimensions))

    def test_all_backend_trainers_share_one_overlay_inventory(self) -> None:
        """CPU, CUDA, and ROCm may not name private shape catalog paths."""

        cmake = (REPO_ROOT / "tests/v2/CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        self.assertEqual(cmake.count("native_quantised_decode_shapes_v5.json"), 1)
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
            "v2_perf_cpu_native_quantised_gemv",
            "v2_test_native_quantised_shape_manifest_projection",
            "v2_perf_cuda_native_quantised_decode_trainer",
            "v2_perf_cuda_native_quantised_gemm",
            "v2_perf_native_quantised_sweep",
            "v2_perf_native_quantised_throughput",
        }
        shared_targets = {
            target
            for target, body in definition_blocks.items()
            if "LLAMINAR_NATIVE_QUANTISED_SHAPE_MANIFEST_PATH" in body
        }
        self.assertSetEqual(shared_targets, expected_shared_targets)
        for target in expected_shared_targets:
            self.assertIn(
                "LLAMINAR_NATIVE_QUANTISED_QWEN_RELEASE_CATALOG_PATH",
                definition_blocks[target],
                target,
            )

        # The MoE trainer and its device-free diagnostic parser consume the
        # centralized release/codebook declarations rather than the generic
        # shape inventory. Fixtures inject documents without replacing those
        # compile-time bindings with a private or dummy catalog path.
        catalog_only_targets = {
            target
            for target, body in definition_blocks.items()
            if "LLAMINAR_NATIVE_QUANTISED_QWEN_RELEASE_CATALOG_PATH" in body
        } - shared_targets
        expected_catalog_only_targets = {
            "v2_perf_moe_verifier_prefill",
            "v2_unit_moe_projection_diagnostic",
        }
        self.assertSetEqual(catalog_only_targets, expected_catalog_only_targets)
        for target in expected_catalog_only_targets:
            with self.subTest(target=target):
                self.assertIn(
                    'LLAMINAR_NATIVE_QUANTISED_MOE_GGUF_PATTERN_MANIFEST_PATH="${LLAMINAR_NATIVE_QUANTISED_MOE_GGUF_PATTERN_MANIFEST}"',
                    definition_blocks[target],
                )
                self.assertIn(
                    'LLAMINAR_NATIVE_QUANTISED_QWEN_RELEASE_CATALOG_PATH="${LLAMINAR_NATIVE_QUANTISED_QWEN_RELEASE_CATALOG}"',
                    definition_blocks[target],
                )


if __name__ == "__main__":
    unittest.main()
