"""Replay canonical RCCL/NCCL admission with a real Git fixture and a tiny image archive."""
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location("canonical_collectives", ROOT / "scripts/build/verify_collective_dependency.py")
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)
IMAGE = "ghcr.io/llaminar/llaminar@sha256:" + "a" * 64


class CanonicalCollectiveTest(unittest.TestCase):
    """Qualify relocated image materialization and reject stale or foreign artifacts."""

    def setUp(self):
        storage = tempfile.TemporaryDirectory()
        self.addCleanup(storage.cleanup)
        self.root = Path(storage.name)
        self.docker = self.root / "scripts/docker"
        (self.docker / "patches").mkdir(parents=True)
        for vendor in ("rccl", "nccl"):
            owner = 'rocm' if vendor == 'rccl' else vendor
            (self.docker / f"{owner}-release.env").write_text(
                f"LLAMINAR_{vendor.upper()}_GIT_REF=v1\nLLAMINAR_{vendor.upper()}_GIT_COMMIT={'1' * 40}\n")
            (self.docker / "patches" / f"{vendor}-one.patch").write_text("published patch\n")
        (self.docker / "rccl-functions.txt").write_text("AllReduce Ring\n")
        def git(*args):
            return subprocess.check_output(["git", "-c", "user.name=test", "-c", "user.email=test@localhost",
                                            "-C", str(self.root), *args], text=True).strip()
        git("init", "-q")
        git("add", "scripts")
        git("commit", "-qm", "pinned dependencies")
        self.revision = git("rev-parse", "HEAD")
        self.library = self.root / "relocated dso"
        self.library.write_bytes(b"published file")
        with tarfile.open(self.root / "image.tar", "w") as archive:
            member = tarfile.TarInfo("lib.so")
            member.size = self.library.stat().st_size
            archive.addfile(member, io.BytesIO(self.library.read_bytes()))
        # The executable records the actual argv and returns the same streams as
        # Docker's immutable image inspect, stopped create, archive and remove.
        docker = self.root / "docker"
        docker.write_text("""#!/bin/sh
set -eu
echo "$1" >> "$TEST_WORK/actions"
case "$1" in
image) echo "$TEST_REV";;
create) echo "$TEST_ID";;
cp) cat "$TEST_WORK/image.tar";;
rm) :;;
*) exit 2;;
esac
""")
        docker.chmod(0o755)
        self.enterContext(mock.patch.object(GUARD, "ROOT", self.root))
        self.enterContext(mock.patch.object(GUARD, "DOCKER", self.docker))
        self.enterContext(mock.patch.dict(os.environ, TEST_WORK=str(self.root), TEST_REV=self.revision,
                                         TEST_ID="4" * 64, PATH=str(self.root) + ":" + os.environ["PATH"]))

    def test_identical_relocated_bytes_for_both_vendors(self):
        for backend in ("rccl", "nccl"):
            with self.subTest(backend=backend):
                GUARD.confirm_published(backend, self.library, IMAGE)
        self.assertEqual((self.root / "actions").read_text().splitlines(), ["image", "create", "cp", "rm"] * 2)

    def test_different_bytes_under_same_library_name_are_rejected_and_released(self):
        self.library.write_bytes(b"experimental small all-gather")
        with self.assertRaisesRegex(ValueError, "differs"):
            GUARD.confirm_published("rccl", self.library, IMAGE)
        self.assertEqual((self.root / "actions").read_text().splitlines()[-1], "rm")

    def test_modified_upstream_source_pin_patches_and_function_set_are_rejected(self):
        for path in (self.docker / "rocm-release.env", self.docker / "patches/rccl-one.patch",
                     self.docker / "rccl-functions.txt"):
            with self.subTest(path=path):
                original = path.read_text()
                path.write_text(original.replace("1" * 40, "2" * 40) if path.suffix == ".env" else "changed")
                with self.assertRaisesRegex(ValueError, "revision|policy"):
                    GUARD.check_image_source("rccl", self.revision)
                path.write_text(original)

    def test_new_untracked_patch_and_mutable_image_tag_are_rejected(self):
        (self.docker / "patches/nccl-additional.patch").write_text("new")
        with self.assertRaisesRegex(ValueError, "different NCCL build policy"):
            GUARD.check_image_source("nccl", self.revision)
        with self.assertRaisesRegex(ValueError, "manifest digest"):
            GUARD.confirm_published("nccl", self.library, "ghcr.io/llaminar/llaminar:develop")
        self.assertFalse((self.root / "actions").exists())

    def test_unproven_source_library_fails_before_probing_a_device(self):
        for vendor in ("rccl", "nccl"):
            with self.subTest(vendor=vendor), self.assertRaisesRegex(ValueError, "canonical"):
                GUARD.confirm_source(vendor, self.library, self.root / "missing", self.root / "missing", "")


if __name__ == "__main__":
    unittest.main()
