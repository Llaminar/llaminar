"""Pure model-scope projection tests; no timing, compiler, native or GPU work."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "performance/kernels"))
from native_quantised_dispatch.modern_sweep_scope import (
    is_modern_qwen_family, modern_sweep_shape_names, modern_expert_projection_geometries,
)
from native_quantised_dispatch.qwen_release_geometry import QwenProjectionUse, QwenReleaseGeometry
from native_quantised_dispatch.shape_manifest import (
    AspectBucket, NativeQuantisedShape, NativeQuantisedShapeManifest,
    ShapePartition, ShapeRole, expert_projection_geometries, load_shape_manifest,
)


class ModernSweepScopeTest(unittest.TestCase):
    def shape(self, name, n, k, family, role=ShapeRole.PRODUCTION):
        return NativeQuantisedShape(name, n, k, family, role, role == ShapeRole.PRODUCTION,
            AspectBucket.BALANCED, ShapePartition.DEVELOPMENT, ShapePartition.DEVELOPMENT, None)

    def fixture(self):
        shapes = (
            self.shape("old_shared", 256, 512, "qwen25-7b"),
            self.shape("old_only", 768, 512, "qwen25-0.5b"),
            self.shape("qwen3_only", 640, 640, "qwen3-32b"),
            self.shape("declared35", 1024, 2048, "qwen35-moe-tp-local"),
            self.shape("declared38", 2048, 4096, "qwen38-moe"),
            self.shape("qfn", 640, 2560, "qwen4exp"),
            self.shape("cert_shared", 256, 512, "certification-grid", ShapeRole.CERTIFICATION),
        )
        manifest = NativeQuantisedShapeManifest("unit-scope", "inert selection fixture", 1 << 30, 1 << 30, shapes)
        releases = (
            QwenReleaseGeometry(256, 512, (
                QwenProjectionUse("Qwen3-32B", "expert_gate_up"),
                QwenProjectionUse("Qwen3.6-35B-A3B", "expert_gate_up"))),
            QwenReleaseGeometry(768, 512, (QwenProjectionUse("Qwen2.5-0.5B", "ffn_gate_up"),)),
        )
        return manifest, releases

    def test_explicit_version_boundaries_and_qfn_are_retained(self):
        for identity in ("qwen35-moe", "qwen36-dense", "Qwen3.5-35B-A3B", "Qwen3.6-27B",
                         "qwen38", "Qwen3.8-80B", "qwen4exp", "qwen35-qwen36-release-geometries"):
            with self.subTest(identity=identity): self.assertTrue(is_modern_qwen_family(identity))
        for identity in ("qwen25-7b", "Qwen2.5-32B", "Qwen3-32B", "qwen3", "Qwen3.80-80B",
                         "qwen350", "qwen36other", "other-qwen38", "certification-grid", ""):
            with self.subTest(identity=identity): self.assertFalse(is_modern_qwen_family(identity))

    def test_shared_legacy_name_survives_and_legacy_only_jobs_do_not(self):
        manifest, releases = self.fixture()
        with patch("native_quantised_dispatch.modern_sweep_scope.qwen_release_geometries", return_value=releases):
            self.assertEqual(modern_sweep_shape_names(manifest),
                ("old_shared", "declared35", "declared38", "qfn"))

    def test_requested_order_and_physical_deduplication_are_stable(self):
        manifest, releases = self.fixture()
        with patch("native_quantised_dispatch.modern_sweep_scope.qwen_release_geometries", return_value=releases):
            names = ("qfn", "old_shared", "old_shared", "old_only", "declared38", "qfn")
            self.assertEqual(modern_sweep_shape_names(manifest, candidate_names=names),
                             ("qfn", "old_shared", "declared38"))
            self.assertEqual(modern_sweep_shape_names(manifest, candidate_names=()), ())

    def test_model_filter_never_changes_runtime_manifest_or_key_names(self):
        manifest, releases = self.fixture()
        original, digest = manifest.shapes, manifest.digest()
        with patch("native_quantised_dispatch.modern_sweep_scope.qwen_release_geometries", return_value=releases):
            modern_sweep_shape_names(manifest)
        self.assertIs(manifest.shapes, original)
        self.assertEqual(manifest.digest(), digest)
        self.assertEqual(manifest.by_name("old_only").name, "old_only")

    def test_missing_name_is_not_silently_treated_as_excluded_model(self):
        manifest, releases = self.fixture()
        with patch("native_quantised_dispatch.modern_sweep_scope.qwen_release_geometries", return_value=releases):
            with self.assertRaises(KeyError): modern_sweep_shape_names(manifest, candidate_names=("missing",))

    def test_expert_shared_keys_preserve_all_original_provenance_records(self):
        base = expert_projection_geometries()[0]
        old_shared = replace(base, model_family="qwen3-old", shape_name="old_shared")
        modern = replace(base, model_family="qwen38-moe", shape_name="explicit_modern")
        old_only = replace(base, model_family="qwen25-old", shape_name="old_only",
                           physical_n=base.physical_n - 1, policy_n=base.policy_n - 1)
        sources = (old_only, old_shared, modern, old_shared)
        selected = modern_expert_projection_geometries(sources)
        self.assertEqual(selected, (old_shared, modern, old_shared))
        self.assertIs(selected[0], old_shared)
        self.assertIs(selected[1], modern)
        self.assertEqual(sources, (old_only, old_shared, modern, old_shared))

    def test_current_expert_catalog_contains_no_excluded_legacy_only_owner(self):
        rows = expert_projection_geometries()
        self.assertEqual(modern_expert_projection_geometries(), rows)
        self.assertTrue(all(is_modern_qwen_family(row.model_family) for row in rows))

    def test_current_source_retains_every_qfn_and_modern_release_geometry(self):
        manifest = load_shape_manifest()
        digest = manifest.digest()
        selected = modern_sweep_shape_names(manifest)
        physical = {(manifest.by_name(name).n, manifest.by_name(name).k) for name in selected}
        from native_quantised_dispatch.qwen_release_geometry import qwen_release_geometries
        self.assertTrue({(row.n, row.k) for row in qwen_release_geometries()}.issubset(physical))
        self.assertTrue({row.name for row in manifest.shapes if row.role == ShapeRole.PRODUCTION
                         and row.model_family == "qwen4exp"}.issubset(selected))
        self.assertEqual(len(physical), len(selected))
        self.assertEqual(manifest.digest(), digest)


if __name__ == "__main__":
    unittest.main()
