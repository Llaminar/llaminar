#!/usr/bin/env python3
"""CPU-only contract tests for NativeVNNI policy accelerator ownership."""

from __future__ import annotations

import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.policy_accelerator import (  # noqa: E402
    ACCELERATOR_ENVIRONMENT,
    ACCELERATOR_LANES_ENVIRONMENT,
    LIBRARY_ENVIRONMENT_BY_BACKEND,
    arm_policy_worker_parent_death_signal,
    policy_accelerator_specs_from_environment,
    policy_accelerator_worker_specs,
)


class NativeVNNIPolicyAcceleratorTest(unittest.TestCase):
    """Validate strict parsing and lane expansion without loading a GPU DSO."""

    def test_empty_inventory_explicitly_selects_cpu_fitting(self) -> None:
        self.assertEqual(policy_accelerator_specs_from_environment({}), ())

    @mock.patch("native_vnni_dispatch.policy_accelerator.os.kill")
    @mock.patch("native_vnni_dispatch.policy_accelerator.os.getpid")
    @mock.patch("native_vnni_dispatch.policy_accelerator.os.getppid")
    @mock.patch("native_vnni_dispatch.policy_accelerator.ctypes.CDLL")
    def test_worker_arms_parent_death_before_vendor_runtime_ownership(
        self,
        load_libc: mock.Mock,
        get_parent_pid: mock.Mock,
        get_pid: mock.Mock,
        kill: mock.Mock,
    ) -> None:
        """The Linux kernel must own cleanup after coordinator death."""

        prctl = mock.Mock(return_value=0)
        load_libc.return_value.prctl = prctl
        get_parent_pid.return_value = 321
        get_pid.return_value = 654

        arm_policy_worker_parent_death_signal(321)

        load_libc.assert_called_once_with(None, use_errno=True)
        self.assertEqual(prctl.call_args.args[:2], (1, 9))
        kill.assert_not_called()

        get_parent_pid.return_value = 1
        arm_policy_worker_parent_death_signal(321)
        kill.assert_called_once_with(654, 9)

    def test_physical_inventory_preserves_order_and_backend_library(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cuda = Path(temporary) / "cuda.so"
            rocm = Path(temporary) / "rocm.so"
            cuda.touch()
            rocm.touch()
            specs = policy_accelerator_specs_from_environment({
                ACCELERATOR_ENVIRONMENT: "cuda:1,rocm:3,cuda:0",
                LIBRARY_ENVIRONMENT_BY_BACKEND["cuda"]: str(cuda),
                LIBRARY_ENVIRONMENT_BY_BACKEND["rocm"]: str(rocm),
            })

        self.assertEqual(
            tuple(spec.label for spec in specs),
            ("cuda:1", "rocm:3", "cuda:0"),
        )
        self.assertEqual(specs[0].library_path, cuda.resolve())
        self.assertEqual(specs[1].library_path, rocm.resolve())

    def test_requested_backend_requires_an_existing_library(self) -> None:
        with self.assertRaisesRegex(ValueError, "is unset"):
            policy_accelerator_specs_from_environment({
                ACCELERATOR_ENVIRONMENT: "cuda:0",
            })
        with self.assertRaises(FileNotFoundError):
            policy_accelerator_specs_from_environment({
                ACCELERATOR_ENVIRONMENT: "rocm:0",
                LIBRARY_ENVIRONMENT_BY_BACKEND["rocm"]: "/absent/rocm.so",
            })

    def test_malformed_or_duplicate_device_identity_is_rejected(self) -> None:
        for inventory in ("cuda", "cuda:-1", "cuda:01", "vulkan:0"):
            with self.subTest(inventory=inventory):
                with self.assertRaises(ValueError):
                    policy_accelerator_specs_from_environment({
                        ACCELERATOR_ENVIRONMENT: inventory,
                    })
        with tempfile.NamedTemporaryFile() as library:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                policy_accelerator_specs_from_environment({
                    ACCELERATOR_ENVIRONMENT: "cuda:0,cuda:0",
                    LIBRARY_ENVIRONMENT_BY_BACKEND["cuda"]: library.name,
                })

    def test_lane_assignment_is_deterministic_and_strict(self) -> None:
        with tempfile.NamedTemporaryFile() as library:
            specs = policy_accelerator_specs_from_environment({
                ACCELERATOR_ENVIRONMENT: "cuda:0,cuda:1",
                LIBRARY_ENVIRONMENT_BY_BACKEND["cuda"]: library.name,
            })
        workers = policy_accelerator_worker_specs(
            specs,
            {ACCELERATOR_LANES_ENVIRONMENT: "3"},
        )
        self.assertEqual(
            tuple(spec.label for spec in workers),
            ("cuda:0", "cuda:1"),
        )
        self.assertEqual(tuple(spec.lane_count for spec in workers), (3, 3))
        for value in ("0", "-1", "01", "many"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    policy_accelerator_worker_specs(
                        specs,
                        {ACCELERATOR_LANES_ENVIRONMENT: value},
                    )

    def test_default_owns_one_persistent_workspace_per_physical_gpu(self) -> None:
        """Do not replicate multi-GiB complete-tree scratch implicitly."""

        with tempfile.NamedTemporaryFile() as library:
            specs = policy_accelerator_specs_from_environment({
                ACCELERATOR_ENVIRONMENT: "cuda:0,rocm:0,rocm:1",
                LIBRARY_ENVIRONMENT_BY_BACKEND["cuda"]: library.name,
                LIBRARY_ENVIRONMENT_BY_BACKEND["rocm"]: library.name,
            })
        workers = policy_accelerator_worker_specs(specs, {})
        self.assertEqual(
            tuple(spec.label for spec in workers),
            ("cuda:0", "rocm:0", "rocm:1"),
        )


if __name__ == "__main__":
    unittest.main()
