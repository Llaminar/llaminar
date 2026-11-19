#!/usr/bin/env python3
"""@file test_toolchain_cache.py
@brief Replay dependency identity and OCI admission without compilers or devices.

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

import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import toolchain_cache as cache


def oci(layout: Path, admitted: dict, *, architecture: str = "amd64") -> tuple[dict, Path]:
    """Write an actual descriptor graph with small, authenticated layer bytes."""
    (layout / "blobs/sha256").mkdir(parents=True)
    def store(value, kind):
        data = value if isinstance(value, bytes) else json.dumps(value).encode()
        digest = hashlib.sha256(data).hexdigest()
        path = layout / "blobs/sha256" / digest
        path.write_bytes(data)
        return {"mediaType": kind, "digest": "sha256:" + digest, "size": len(data)}, path
    layer, layer_path = store(b"completed-sdk-and-toolchain", "application/vnd.oci.image.layer.v1.tar")
    config, _ = store({"os": "linux", "architecture": architecture, "config": {"Labels": {
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

    def test_sdk_cache_upload_failure_is_independent_of_toolchain_build_status(self):
        """Only publishing intermediate layers can fail independently of the build."""
        action = yaml.safe_load((ROOT / ".github/actions/cache-develop-toolchain/action.yml").read_text())
        steps = action["runs"]["steps"]
        checkpoint = next(step for step in steps if step.get("with", {}).get("target") == "gpu-sdk")
        cache_export = dict(field.split("=", 1) for field in checkpoint["with"]["cache-to"].split(","))
        self.assertEqual(cache_export["type"], "gha")
        self.assertEqual(cache_export.get("ignore-error"), "true")
        for step in steps:
            self.assertFalse(step.get("continue-on-error", False))

    def test_dedicated_avx512_cache_miss_uses_the_persistent_builder(self):
        """A warm AVX512 worker can produce a verified OCI image without a remote SDK round-trip."""
        action = yaml.safe_load((ROOT / ".github/actions/cache-develop-toolchain/action.yml").read_text())
        checkpoint, avx512, avx2 = [step for step in action["runs"]["steps"]
                                    if step.get("with", {}).get("target") in {"gpu-sdk", "toolchain"}
                                    and "NEON" not in step["if"]]
        self.assertNotIn("inputs.cpu-isa == 'AVX512'", checkpoint["if"])
        self.assertIn("inputs.cpu-isa == 'both'", checkpoint["if"])
        self.assertIn("inputs.cpu-isa == 'AVX2'", checkpoint["if"])
        self.assertEqual(avx512["with"]["cache-from"], "type=registry,ref=${{ steps.cache.outputs.image }}-avx512")
        self.assertIn("type=oci,", avx512["with"]["outputs"])
        self.assertIn("type=gha,", avx2["with"]["cache-from"])

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
        for isa in ("AVX512", "AVX2"):
            self.assertIn(cache.identity(self.tree, isa)["fingerprint"], values["key_" + isa.lower()])

    def test_neon_cache_admits_only_arm64_and_emits_the_pinned_context(self):
        """Restore a real OCI descriptor graph and pass its exact identity to BuildKit."""
        output = self.root / "output"
        prepared = self.root / "prepared"
        cache.prepare(prepared, "NEON", output)
        values = dict(line.split("=", 1) for line in output.read_text().splitlines())
        expected = cache.identity(self.tree, "NEON")
        self.assertIn(expected["fingerprint"], values["key_neon"])
        self.assertNotEqual(expected["fingerprint"], cache.identity(self.tree, "AVX2")["fingerprint"])
        foreign = self.root / "foreign"
        oci(foreign, expected)
        with self.assertRaisesRegex(ValueError, "must target linux/arm64"):
            cache.admit(foreign, expected, verify_layers=True)
        layout = Path(values["path_neon"])
        manifest, _ = oci(layout, expected, architecture="arm64")
        with patch.object(sys, "argv", ["toolchain_cache.py", "verify", "--root", str(prepared),
                                        "--cpu-isa", "NEON", "--cache-hit", "true",
                                        "--github-output", str(output)]):
            cache.main()
        receipt = json.loads((prepared / "neon-toolchain.json").read_text())
        self.assertTrue(receipt["cache_restored"])
        self.assertEqual(receipt["manifest"], manifest["digest"])
        context = f"toolchain=oci-layout://{layout}@{manifest['digest']}"
        self.assertIn(f"build_context={context}", output.read_text().splitlines())

    def test_arm64_pr_and_publication_share_the_neon_cache_contract(self):
        """Require authenticated inputs and keep registry writes on trusted develop jobs."""
        action = yaml.safe_load((ROOT / ".github/actions/cache-develop-toolchain/action.yml").read_text())
        neon_steps = [step for step in action["runs"]["steps"] if "NEON" in step.get("if", "")]
        restore, checkpoint, toolchain, verify, save = neon_steps
        self.assertEqual(restore["with"], save["with"])
        self.assertEqual(restore["uses"], "actions/cache/restore@v5")
        self.assertEqual(save["uses"], "actions/cache/save@v5")
        self.assertEqual(checkpoint["with"]["target"], "gpu-sdk")
        self.assertIn("scope=llaminar-gpu-sdk-neon", checkpoint["with"]["cache-to"])
        self.assertEqual(toolchain["with"]["target"], "toolchain")
        for step in (checkpoint, toolchain):
            self.assertEqual(step["with"]["platforms"], "linux/arm64")
            self.assertIn("LLAMINAR_CUDA_ARCHS=121", step["with"]["build-args"])
            self.assertIn("LLAMINAR_ENABLE_ROCM=OFF", step["with"]["build-args"])
        self.assertIn("--github-output", verify["run"])
        workflows = ROOT / ".github/workflows"
        pr = yaml.load((workflows / "develop-pr-arm64.yml").read_text(), Loader=yaml.BaseLoader)
        publication = yaml.load((workflows / "ci.yml").read_text(), Loader=yaml.BaseLoader)
        pr_steps = pr["jobs"]["arm64"]["steps"]
        publication_steps = publication["jobs"]["develop-arm64"]["steps"]
        for steps in (pr_steps, publication_steps):
            cache_step = next(step for step in steps if step.get("id") == "toolchain")
            self.assertEqual(cache_step["uses"], "./.github/actions/cache-develop-toolchain")
            self.assertEqual(cache_step["with"]["cpu-isa"], "NEON")
        build = next(step["with"] for step in pr_steps if step.get("with", {}).get("target") == "test-runner")
        self.assertEqual(build["build-contexts"], "${{ steps.toolchain.outputs.neon-context }}")
        self.assertIn("type=registry,ref=${{ steps.toolchain.outputs.neon-registry-cache }}", build["cache-from"])
        self.assertTrue(build["cache-to"].startswith("type=gha,scope=llaminar-build-neon-pr,"))
        self.assertIn("scope=llaminar-build-neon-pr", build["cache-from"])
        publish_build = next(step for step in publication_steps if "--cache-to" in step.get("run", ""))
        login = next(step for step in publication_steps if step.get("uses") == "docker/login-action@v4")
        self.assertLess(publication_steps.index(login), publication_steps.index(publish_build))
        self.assertEqual(publish_build["env"]["REGISTRY_CACHE"],
                         "${{ steps.toolchain.outputs.neon-registry-cache }}")
        self.assertIn("--build-context", publish_build["run"])
        self.assertIn("--cache-from", publish_build["run"])
        self.assertIn('--cache-to "type=registry,ref=$REGISTRY_CACHE', publish_build["run"])

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
