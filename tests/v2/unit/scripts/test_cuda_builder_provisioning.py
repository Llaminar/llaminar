#!/usr/bin/env python3
## @file test_cuda_builder_provisioning.py
# @brief Replay real CUDA installer profiles, repository selection and the resource-proof tool dependency.
"""Execute the actual installer with package and child-process effects isolated.

The NCCL child checks that binary-inspection tooling was provisioned before its
resource proof. The original missing-package control must fail. The image's
Ubuntu release and Debian architecture are supplied by the replay, never read
from the host, so repository selection is exercised identically everywhere.
No packages, SDKs, network access, root privileges or GPU devices are needed.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
INSTALLER = ROOT / "scripts/docker/install-cuda.sh"
PACKAGE = "cuda-cuobjdump-13-0"


class CUDABuilderProvisioningTests(unittest.TestCase):
    """Native resource proofs require tooling even when profiling is disabled."""

    def run_installer(self, mode: str, *, original: bool = False, architecture: str = "amd64",
                      release: str = "24.04"):
        """Replay package ordering while intercepting every external mutation.

        @param architecture Value the image's `dpkg --print-architecture` reports.
        @param release VERSION_ID of the image's os-release.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            commands = root / "commands"
            commands.mkdir()
            log = root / "commands.jsonl"
            shim = commands / "shim"
            shim.write_text(f"#!{sys.executable}\n" + r'''import json, os, sys
from pathlib import Path
log = Path(os.environ["CUDA_PROVISIONING_LOG"])
name = Path(sys.argv[0]).name
if name == "dpkg" and sys.argv[1:] == ["--print-architecture"]:
    print(os.environ["CUDA_PROVISIONING_ARCHITECTURE"])
if name == "bash":
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    packages = {arg for command, args in calls if command == "apt-get" and "install" in args for arg in args}
    if "cuda-cuobjdump-13-0" not in packages:
        raise SystemExit("NCCL resource inspection requires cuda-cuobjdump-13-0")
with log.open("a") as output:
    output.write(json.dumps((name, sys.argv[1:])) + "\n")
''')
            shim.chmod(0o755)
            for command in ("curl", "dpkg", "apt-get", "rm", "bash"):
                (commands / command).symlink_to(shim)
            os_release = root / "os-release"
            os_release.write_text(f'ID=ubuntu\nVERSION_ID="{release}"\n')
            script = root / "install-cuda.sh"
            text = INSTALLER.read_text()
            # The installer must read exactly one os-release; the replay supplies the image's.
            self.assertEqual(text.count("/etc/os-release"), 1)
            text = text.replace("/etc/os-release", str(os_release))
            if original:
                text = text.replace("        cuda-cuobjdump-13-0 \\\n", "")
            script.write_text(text)
            result = subprocess.run(["/usr/bin/bash", str(script)], text=True, capture_output=True,
                                    env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}",
                                         "MODE": mode, "INSTALL_CUDA_PROFILERS": "0", "CUDAARCHS": "80",
                                         "CUDA_PROVISIONING_LOG": str(log),
                                         "CUDA_PROVISIONING_ARCHITECTURE": architecture})
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            return result, calls

    def test_builder_installs_inspector_before_nccl_and_runtime_omits_it(self):
        """The inspection dependency belongs to the lean builder profile."""
        for mode in ("full", "runtime"):
            with self.subTest(mode=mode):
                result, calls = self.run_installer(mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                packages = {arg for command, args in calls if command == "apt-get" and "install" in args for arg in args}
                self.assertEqual(PACKAGE in packages, mode == "full")
                self.assertEqual(any(command == "bash" for command, _ in calls), mode == "full")

    def test_repository_follows_image_release_and_architecture(self):
        """amd64 uses x86_64 and arm64 uses sbsa for each supported Ubuntu release."""
        for architecture, release, repository in (
                ("amd64", "24.04", "ubuntu2404/x86_64"), ("arm64", "24.04", "ubuntu2404/sbsa"),
                ("amd64", "22.04", "ubuntu2204/x86_64"), ("arm64", "22.04", "ubuntu2204/sbsa")):
            with self.subTest(architecture=architecture, release=release):
                result, calls = self.run_installer("runtime", architecture=architecture, release=release)
                self.assertEqual(result.returncode, 0, result.stderr)
                downloads = [arg for command, args in calls if command == "curl" for arg in args
                             if arg.startswith("https://")]
                self.assertEqual(downloads, [
                    f"https://developer.download.nvidia.com/compute/cuda/repos/{repository}/cuda-keyring_1.1-1_all.deb"])

    def test_unsupported_image_fails_before_any_download_or_package_change(self):
        """An unknown architecture or release fails closed before curl, dpkg -i or apt-get."""
        for architecture, release in (("ppc64el", "24.04"), ("", "24.04"), ("arm64", "20.04")):
            with self.subTest(architecture=architecture, release=release):
                result, calls = self.run_installer("full", architecture=architecture, release=release)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("for the CUDA repository", result.stderr)
                self.assertFalse([command for command, args in calls
                                  if command in ("curl", "apt-get") or args[:1] == ["-i"]])

    def test_original_missing_inspector_fails_native_resource_admission(self):
        """Replay the former minimal installer instead of merely scanning text."""
        result, _ = self.run_installer("full", original=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NCCL resource inspection requires", result.stderr)


if __name__ == "__main__":
    unittest.main()
