#!/usr/bin/env python3
"""Replay dependency identity and OCI admission without compilers or devices.

The fixture repository separates source edits from actual dependency changes.
Real OCI descriptors exercise swapped ISAs, stale inputs, missing/corrupt blobs
and symlink substitution through the same admission used by cloud builds.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import toolchain_cache as cache


def oci(layout: Path, admitted: dict) -> tuple[dict, Path]:
    """Write an actual descriptor graph with small, authenticated layer bytes."""
    (layout / "blobs/sha256").mkdir(parents=True)
    def store(value, kind):
        data = value if isinstance(value, bytes) else json.dumps(value).encode()
        digest = hashlib.sha256(data).hexdigest()
        path = layout / "blobs/sha256" / digest
        path.write_bytes(data)
        return {"mediaType": kind, "digest": "sha256:" + digest, "size": len(data)}, path
    layer, layer_path = store(b"completed-sdk-and-toolchain", "application/vnd.oci.image.layer.v1.tar")
    config, _ = store({"os": "linux", "architecture": "amd64", "config": {"Labels": {
        cache.INPUT_LABEL: admitted["fingerprint"], cache.ISA_LABEL: admitted["cpu_isa"]}}},
        "application/vnd.oci.image.config.v1+json")
    manifest, _ = store({"schemaVersion": 2, "config": config, "layers": [layer]}, cache.MANIFEST_TYPE)
    (layout / "index.json").write_text(json.dumps({"schemaVersion": 2, "manifests": [manifest]}))
    (layout / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}')
    return manifest, layer_path


class ToolchainCacheTests(unittest.TestCase):
    """A declared toolchain authority either admits exact bytes or fails hard."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        for name, data in {"Dockerfile": "FROM scratch AS toolchain\n", ".dockerignore": ".git\n",
                           "requirements.txt": "dependency==1\n", "scripts/docker/seed": "SDK1\n",
                           "scripts/build/guard.py": "policy = 1\n", "src/source.cpp": "version1\n",
                           "scripts/ci/toolchain_cache.py": "schema = 1\n",
                           ".github/actions/cache-develop-toolchain/action.yml": "name: policy\n"}.items():
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data)
        self.tree = self.commit()
        self.policy = patch.object(cache, "ROOT", self.repo)
        self.policy.start()
        self.addCleanup(self.policy.stop)

    def git(self, *arguments):
        """Keep temporary Git state isolated from developer hooks/configuration."""
        return subprocess.check_output(["git", "-C", str(self.repo), *arguments], stderr=subprocess.DEVNULL)

    def commit(self):
        """Publish fixture inputs into the same object database read by admission."""
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "-c", "core.hooksPath=/dev/null", "commit", "-qm", "fixture")
        return self.git("rev-parse", "HEAD^{tree}").decode().strip()

    def test_dependency_key_survives_source_edits_and_rejects_each_dependency_change(self):
        before = cache.identity(self.tree, "AVX512")
        (self.repo / "src/source.cpp").write_text("version2\n")
        tree = self.commit()
        self.assertEqual(cache.identity(tree, "AVX512"), before)
        self.assertNotEqual(cache.identity(tree, "AVX2")["fingerprint"], before["fingerprint"])
        for name in ("Dockerfile", ".dockerignore", "requirements.txt", "scripts/docker/seed", "scripts/build/guard.py",
                     "scripts/ci/toolchain_cache.py", ".github/actions/cache-develop-toolchain/action.yml"):
            previous = cache.identity(tree, "AVX512")
            with (self.repo / name).open("a") as stream:
                stream.write("changed\n")
            tree = self.commit()
            self.assertNotEqual(cache.identity(tree, "AVX512")["fingerprint"], previous["fingerprint"])

    def test_restricted_context_contains_all_keyed_dependencies_and_no_application_source(self):
        output = self.root / "output"
        cache.prepare(self.root / "prepared", "both", output)
        self.assertTrue((self.root / "prepared/inputs/scripts/docker/seed").is_file())
        self.assertFalse((self.root / "prepared/inputs/src").exists())
        values = dict(line.split("=", 1) for line in output.read_text().splitlines())
        for isa in cache.ISAS:
            self.assertIn(cache.identity(self.tree, isa)["fingerprint"], values["key_" + isa.lower()])

    def test_artifact_survives_source_failure_and_binds_a_new_source_tree_by_digest(self):
        expected = cache.identity(self.tree, "AVX512")
        root = self.root / "artifacts"
        descriptor, _ = oci(root / "avx512", expected)
        # A failed/changed application does not change the completed dependency image.
        (self.repo / "src/source.cpp").write_text("downstream source failed\n")
        source_tree = self.commit()
        with patch.dict(os.environ, {cache.TOOLCHAIN_ROOT_ENV: str(root)}):
            arguments = cache.build_context_arguments("AVX512", source_tree)
        self.assertEqual(arguments, ["--build-context",
            f"toolchain=oci-layout://{root}/avx512@{descriptor['digest']}"])
        (self.repo / "scripts/docker/seed").write_text("SDK2\n")
        source_tree = self.commit()
        with patch.dict(os.environ, {cache.TOOLCHAIN_ROOT_ENV: str(root)}), \
             self.assertRaisesRegex(ValueError, "dependency inputs/ISA"):
            cache.build_context_arguments("AVX512", source_tree)

    def test_artifact_rejects_cross_isa_missing_or_corrupt_layer(self):
        expected = cache.identity(self.tree, "AVX512")
        layout = self.root / "layout"
        _, layer = oci(layout, expected)
        cache.admit(layout, expected, verify_layers=True)
        with self.assertRaisesRegex(ValueError, "dependency inputs/ISA"):
            cache.admit(layout, cache.identity(self.tree, "AVX2"))
        original = layer.read_bytes()
        layer.write_bytes(b"x" * len(original))
        with self.assertRaisesRegex(ValueError, "layer digest mismatch"):
            cache.admit(layout, expected, verify_layers=True)
        layer.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            cache.admit(layout, expected)
        other = self.root / "other"
        other.write_bytes(original)
        layer.symlink_to(other)
        with self.assertRaisesRegex(ValueError, "symlink"):
            cache.admit(layout, expected)

    def test_declared_authority_cannot_silently_build_an_absent_toolchain(self):
        for root in ("", "relative", str(self.root / "absent")):
            with self.subTest(root=root), patch.dict(os.environ, {cache.TOOLCHAIN_ROOT_ENV: root}), \
                 self.assertRaisesRegex(ValueError, "prepared absolute directory"):
                cache.build_context_arguments("AVX512", self.tree)
        with patch.dict(os.environ, {cache.TOOLCHAIN_ROOT_ENV: str(self.root)}), \
             self.assertRaisesRegex(ValueError, "missing"):
            cache.build_context_arguments("AVX512", self.tree)

    def test_invalid_metadata_cannot_substitute_or_redirect_the_cached_image(self):
        expected = cache.identity(self.tree, "AVX512")
        layout = self.root / "layout"
        descriptor, _ = oci(layout, expected)
        index = {"schemaVersion": 2, "manifests": [descriptor]}
        for manifests in ([], [descriptor, descriptor], [{**descriptor, "digest": "../../other"}]):
            with self.subTest(manifests=manifests):
                (layout / "index.json").write_text(json.dumps({**index, "manifests": manifests}))
                with self.assertRaises(ValueError):
                    cache.admit(layout, expected)
        (layout / "index.json").write_text(json.dumps(index))
        manifest = layout / "blobs/sha256" / descriptor["digest"].split(":")[1]
        original = manifest.read_bytes()
        manifest.write_bytes(b"x" * len(original))
        with self.assertRaisesRegex(ValueError, "blob digest mismatch"):
            cache.admit(layout, expected)

    def test_cloud_driver_rejects_bad_toolchain_before_snapshot_or_docker(self):
        """The image pipeline must enforce admission before paying for a build."""
        import run_production_pipeline as pipeline
        source = {"tree": self.tree, "revision": "fixture", "dirty": False}
        with patch.dict(os.environ, {cache.TOOLCHAIN_ROOT_ENV: str(self.root / "absent")}), \
             patch.object(pipeline, "snapshot") as snapshot, \
             patch.object(pipeline, "run") as run, \
             patch.object(pipeline, "image_identity"), \
             self.assertRaisesRegex(ValueError, "prepared absolute directory"):
            pipeline.build(argparse.Namespace(cpu_isa="AVX512"), source, self.root)
        snapshot.assert_not_called()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
