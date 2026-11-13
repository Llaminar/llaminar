#!/usr/bin/env python3
## @file test_cuda_builder_provisioning.py
# @brief Replay real CUDA installer profiles and their resource-proof tool dependency.
"""Execute the actual installer with package and child-process effects isolated.

The NCCL child checks that binary-inspection tooling was provisioned before its
resource proof. The original missing-package control must fail. No packages,
SDKs, network access, root privileges or GPU devices are needed for this replay.
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

    def run_installer(self, mode: str, *, original: bool = False):
        """Replay package ordering while intercepting every external mutation."""
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
            script = root / "install-cuda.sh"
            text = INSTALLER.read_text()
            if original:
                text = text.replace("        cuda-cuobjdump-13-0 \\\n", "")
            script.write_text(text)
            result = subprocess.run(["/usr/bin/bash", str(script)], text=True, capture_output=True,
                                    env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}",
                                         "MODE": mode, "INSTALL_CUDA_PROFILERS": "0", "CUDAARCHS": "80",
                                         "CUDA_PROVISIONING_LOG": str(log)})
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

    def test_original_missing_inspector_fails_native_resource_admission(self):
        """Replay the former minimal installer instead of merely scanning text."""
        result, _ = self.run_installer("full", original=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NCCL resource inspection requires", result.stderr)


if __name__ == "__main__":
    unittest.main()
