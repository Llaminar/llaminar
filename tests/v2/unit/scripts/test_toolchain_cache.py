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
        for name, data in {"Dockerfile": "FROM scratch AS toolchain\n"
                           "COPY scripts/docker/seed requirements.txt /sdk/\n"
                           "COPY scripts/build/ /build/\nFROM toolchain AS builder\nCOPY src/ /src/\n",
                           ".dockerignore": ".git\n", "requirements.txt": "dependency==1\n",
                           "scripts/docker/seed": "SDK1\n", "scripts/docker/packager": "runtime only\n",
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
        dockerfile = self.repo / "Dockerfile"
        with dockerfile.open("a") as stream:
            stream.write("COPY scripts/docker/packager /packager\n")
        (self.repo / "scripts/docker/packager").write_text("new downstream package\n")
        tree = self.commit()
        self.assertEqual(cache.identity(tree, "AVX512"), before)
        dockerfile.write_text(dockerfile.read_text().replace("FROM scratch", "ARG SDK_VERSION=2\nFROM scratch"))
        tree = self.commit()
        self.assertNotEqual(cache.identity(tree, "AVX512"), before)
        for name in (".dockerignore", "requirements.txt", "scripts/docker/seed", "scripts/build/guard.py",
                     "scripts/ci/toolchain_cache.py", ".github/actions/cache-develop-toolchain/action.yml"):
            previous = cache.identity(tree, "AVX512")
            with (self.repo / name).open("a") as stream:
                stream.write("changed\n")
            tree = self.commit()
            self.assertNotEqual(cache.identity(tree, "AVX512")["fingerprint"], previous["fingerprint"])

    def test_restricted_context_contains_all_keyed_dependencies_and_no_application_source(self):
        inputs = cache.DependencyContext.from_tree(self.tree)
        context = self.root / "inputs"
        (self.repo / "scripts/docker/seed").write_text("dirty, not authorized\n")
        inputs.stage(context)
        self.assertEqual((context / "scripts/docker/seed").read_text(), "SDK1\n")
        files = {str(path.relative_to(context)) for path in context.rglob("*") if path.is_file()}
        self.assertEqual(files, {"Dockerfile", ".dockerignore", "requirements.txt", "scripts/docker/seed",
                                 "scripts/build/guard.py", *cache.POLICY})
        self.assertNotIn(b"AS builder", (context / "Dockerfile").read_bytes())
        self.assertEqual(hashlib.sha256((context / "Dockerfile").read_bytes()).hexdigest(),
                         inputs.identity("AVX512")["recipe_sha256"])

    def test_new_copy_directory_tracks_nested_files_and_modes_without_an_input_list(self):
        dockerfile = self.repo / "Dockerfile"
        dockerfile.write_text(dockerfile.read_text().replace("FROM toolchain AS builder",
            "COPY extra/tools/ /new-sdk/\nFROM toolchain AS builder"))
        helper = self.repo / "extra/tools/nested/generate"
        helper.parent.mkdir(parents=True)
        helper.write_text("generator v1")
        first = cache.DependencyContext.from_tree(self.commit())
        helper.chmod(0o755)
        second = cache.DependencyContext.from_tree(self.commit())
        self.assertNotEqual(first.identity("AVX2"), second.identity("AVX2"))
        ctx = self.root / "second"
        second.stage(ctx)
        self.assertEqual((ctx / "extra/tools/nested/generate").stat().st_mode & 0o777, 0o755)
        (helper.parent / "new-input").write_text("more data")
        self.assertNotEqual(second.identity("AVX2"), cache.identity(self.commit(), "AVX2"))

    def test_missing_ambiguous_or_unsupported_input_fails_before_issuing_identity(self):
        dockerfile = self.repo / "Dockerfile"
        original = dockerfile.read_text()
        boundary = "FROM toolchain AS builder"
        for instruction in ("COPY missing /tmp/", "COPY --from=other x /tmp/", "COPY scripts/* /tmp/",
                            "ADD remote /tmp/", "RUN --mount=type=bind,target=/src build", boundary):
            with self.subTest(instruction=instruction):
                dockerfile.write_text(original.replace(boundary, instruction + "\n" + boundary))
                with self.assertRaises(ValueError):
                    cache.identity(self.commit(), "AVX2")
        dockerfile.write_text(original.replace(boundary, "FROM scratch AS app"))
        with self.assertRaisesRegex(ValueError, "boundary"):
            cache.identity(self.commit(), "AVX2")
        dockerfile.write_text(original)
        seed = self.repo / "scripts/docker/seed"
        seed.unlink()
        seed.symlink_to("../../requirements.txt")
        with self.assertRaisesRegex(ValueError, "regular committed file"):
            cache.identity(self.commit(), "AVX512")

    def test_artifact_survives_source_failure_and_binds_a_new_source_tree_by_digest(self):
        expected = cache.identity(self.tree, "AVX512")
        root = self.root / "artifacts"
        descriptor, _ = oci(root / "avx512", expected)
        cache.verify(root, expected)
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

    def test_cold_build_and_layer_verification_are_mandatory_even_without_a_cache_service(self):
        # Replace only the external solver. The real staging, command, verifier,
        # receipts and source-consumption path run for single and dual-ISA callers.
        real_run = subprocess.run
        for request in (*cache.ISAS, "both"):
            with self.subTest(request=request):
                calls = []
                def solve(command, **kwargs):
                    if command[0] != "docker":
                        return real_run(command, **kwargs)
                    self.assertTrue(kwargs["check"])
                    self.assertEqual(command[:3], ["docker", "buildx", "build"])
                    self.assertIn("--target=toolchain", command)
                    self.assertIn("--platform=linux/amd64", command)
                    self.assertFalse(any(arg.startswith(("--cache-from", "--cache-to")) or
                                         "type=gha" in arg for arg in command))
                    selected = command[command.index("--build-arg") + 1].split("=")[1]
                    expected = cache.identity(self.tree, selected)
                    self.assertIn(f"{cache.INPUT_LABEL}={expected['fingerprint']}", command)
                    self.assertIn(f"{cache.ISA_LABEL}={selected}", command)
                    self.assertTrue((Path(command[-1]) / "scripts/build/guard.py").is_file())
                    self.assertFalse((Path(command[-1]) / "src").exists())
                    self.assertEqual(command[command.index("--output") + 1],
                        f"type=oci,dest={root / selected.lower()},tar=false")
                    oci(root / selected.lower(), expected)
                    calls.append(selected)
                    return subprocess.CompletedProcess(command, 0)
                root = self.root / request
                with patch.object(subprocess, "run", side_effect=solve), patch.dict(os.environ, {
                        "ACTIONS_CACHE_URL": "http://127.0.0.1:1/unavailable", cache.TOOLCHAIN_ROOT_ENV: str(root)}):
                    cache.build_toolchains(root, request)
                    self.assertEqual(calls, list(cache.ISAS) if request == "both" else [request])
                    for selected in calls:
                        arguments = cache.build_context_arguments(selected, self.tree)
                        receipt = json.loads((root / f"{selected.lower()}-toolchain.json").read_bytes())
                        self.assertTrue(arguments[-1].endswith("@" + receipt["manifest"]))

    def test_failed_solve_or_corrupt_export_cannot_issue_a_success_receipt(self):
        real_run = subprocess.run
        for reason in ("build-error", "corrupt-layer"):
            with self.subTest(reason=reason):
                root = self.root / reason
                def solve(command, **kwargs):
                    if command[0] != "docker":
                        return real_run(command, **kwargs)
                    if reason == "build-error":
                        raise subprocess.CalledProcessError(19, command)
                    _, layer = oci(root / "avx2", cache.identity(self.tree, "AVX2"))
                    layer.write_bytes(b"x" * layer.stat().st_size)
                    return subprocess.CompletedProcess(command, 0)
                with patch.object(subprocess, "run", side_effect=solve), self.assertRaises(
                        subprocess.CalledProcessError if reason == "build-error" else ValueError):
                    cache.build_toolchains(root, "AVX2")
                self.assertFalse((root / "avx2-toolchain.json").exists())

    def test_missing_or_stale_receipt_rejected_before_image_consumption(self):
        expected = cache.identity(self.tree, "AVX2")
        root = self.root / "artifacts"
        oci(root / "avx2", expected)
        with patch.dict(os.environ, {cache.TOOLCHAIN_ROOT_ENV: str(root)}):
            with self.assertRaisesRegex(ValueError, "missing"):
                cache.build_context_arguments("AVX2", self.tree)
            cache.verify(root, expected)
            receipt = root / "avx2-toolchain.json"
            values = json.loads(receipt.read_bytes())
            values["manifest"] = "sha256:" + "a" * 64
            receipt.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, "receipt does not match"):
                cache.build_context_arguments("AVX2", self.tree)

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
