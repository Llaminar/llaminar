#!/usr/bin/env python3
"""Regression tests for released Qwen NativeVNNI geometry coverage."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
sys.path.insert(0, str(KERNEL_ROOT))

from native_vnni_dispatch.qwen_release_geometry import (  # noqa: E402
    QWEN_RELEASE_MODELS,
    model_projection_geometries,
    qwen_release_geometries,
)
from native_vnni_dispatch.shape_manifest import (  # noqa: E402
    ShapeRole,
    load_shape_manifest,
)


class NativeVNNIQwenReleaseGeometryTest(unittest.TestCase):
    """Prove release completeness without introducing model-aware dispatch."""

    def test_official_dense_and_moe_release_sizes_are_complete(self) -> None:
        """Lock the reviewed Qwen 3.5/3.6 release inventory."""

        self.assertEqual(
            {model.release_id for model in QWEN_RELEASE_MODELS},
            {
                "Qwen3.5-0.8B",
                "Qwen3.5-2B",
                "Qwen3.5-4B",
                "Qwen3.5-9B",
                "Qwen3.5-27B",
                "Qwen3.5-35B-A3B",
                "Qwen3.5-122B-A10B",
                "Qwen3.5-397B-A17B",
                "Qwen3.6-27B",
                "Qwen3.6-35B-A3B",
            },
        )

    def test_large_moe_and_mtp_geometries_match_official_configs(self) -> None:
        """Exercise the easy-to-miss 122B, 397B, and concatenated MTP shapes."""

        by_release = {model.release_id: model for model in QWEN_RELEASE_MODELS}
        projection_maps = {
            release: {
                projection: (n, k)
                for projection, n, k in model_projection_geometries(model)
            }
            for release, model in by_release.items()
        }
        self.assertEqual(
            projection_maps["Qwen3.5-122B-A10B"]["gdn_qkv"],
            (12_288, 3_072),
        )
        self.assertEqual(
            projection_maps["Qwen3.5-122B-A10B"]["lm_head"],
            (248_320, 3_072),
        )
        self.assertEqual(
            projection_maps["Qwen3.5-397B-A17B"]["attention_q_gate"],
            (16_384, 4_096),
        )
        self.assertEqual(
            projection_maps["Qwen3.5-397B-A17B"]["lm_head"],
            (248_320, 4_096),
        )
        self.assertEqual(
            projection_maps["Qwen3.6-27B"]["mtp_hidden_embedding"],
            (5_120, 10_240),
        )

    def test_release_aliases_collapse_to_geometry_only(self) -> None:
        """Qwen 3.5/3.6 aliases must not duplicate generated dispatch keys."""

        geometries = qwen_release_geometries()
        self.assertEqual(len(geometries), 60)
        self.assertEqual(
            len({(geometry.n, geometry.k) for geometry in geometries}),
            len(geometries),
        )
        shared = next(
            geometry
            for geometry in geometries
            if (geometry.n, geometry.k) == (12_288, 5_120)
        )
        self.assertEqual(
            {use.release_id for use in shared.uses},
            {"Qwen3.5-27B", "Qwen3.6-27B"},
        )

    def test_every_release_geometry_is_a_production_exact_overlay(self) -> None:
        """No generic rule may substitute for a known released-model overlay."""

        manifest = load_shape_manifest()
        production = {
            (shape.n, shape.k): shape
            for shape in manifest.shapes
            if shape.role == ShapeRole.PRODUCTION
        }
        for geometry in qwen_release_geometries():
            with self.subTest(n=geometry.n, k=geometry.k):
                shape = production[(geometry.n, geometry.k)]
                self.assertTrue(shape.exact_overlay)


if __name__ == "__main__":
    unittest.main()
