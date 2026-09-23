#!/usr/bin/env python3
"""Device-free contract tests for HTTP E2E container accelerator visibility.

Automatic planning must discover every permitted backend inside the actual
runtime container. In particular, a CUDA constraint is not a literal device
address, and excluding CUDA must not require an NVIDIA driver on ROCm hosts.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


INTENT_SCRIPT = Path(__file__).resolve().parents[2] / "e2e/server/docker_device_intent.sh"


def needs_cuda(*arguments: str) -> bool:
    """Invoke the exact shell decision used by the production HTTP runner."""
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; shift; docker_args_need_cuda "$@"',
         "device-intent-test", str(INTENT_SCRIPT), *arguments],
        capture_output=True, text=True, check=False,
    )
    if result.returncode not in (0, 1):
        raise AssertionError(f"device intent crashed: {result.stderr}")
    return result.returncode == 0


class TestServerDockerDeviceIntent(unittest.TestCase):
    """Lock down auto, constrained-auto and authored placement decisions."""

    def test_auto_cuda_backend_constraint_injects_driver(self) -> None:
        self.assertTrue(needs_cuda(
            "--auto", "--only-backends", "cuda",
            "--auto-device-counts", "cuda=1", "--only-strategies", "single"))
        self.assertTrue(needs_cuda(
            "--auto", "--only-backends", "cpu,cuda",
            "--auto-device-counts", "cpu=2,cuda=2",
            "--only-strategies", "expert-overlay"))
        self.assertTrue(needs_cuda("--only-backends=cpu,cuda"))

    def test_unconstrained_auto_discovers_cuda(self) -> None:
        self.assertTrue(needs_cuda("--auto"))
        self.assertTrue(needs_cuda("--plan-workload", "512,384"))
        self.assertTrue(needs_cuda("--config", "/tmp/saved-plan.json"))

    def test_auto_excluding_cuda_needs_no_nvidia_driver(self) -> None:
        self.assertFalse(needs_cuda("--auto", "--only-backends", "cpu,rocm"))
        self.assertFalse(needs_cuda("--only-backends=cpu"))

    def test_authored_placement_uses_named_devices(self) -> None:
        self.assertTrue(needs_cuda("--device", "cuda:0"))
        self.assertTrue(needs_cuda("--expert-tier", "accelerator=cuda:0,cuda:1;priority=0"))
        self.assertFalse(needs_cuda("--device", "rocm:0"))
        self.assertFalse(needs_cuda("--expert-tier", "accelerator=rocm:0;priority=0",
                                    "--expert-tier", "capacity=cpu:0;priority=10"))


if __name__ == "__main__":
    unittest.main()
