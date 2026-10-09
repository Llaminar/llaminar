#!/usr/bin/env python3
"""Prove container IPC storage and recursive-mutex owner identities agree.

ROCm SMI stores robust recursive process-shared mutexes in /dev/shm. Different
PID namespaces can assign the same TID to unrelated owners of that storage.
These device-free regressions execute the launch policy and its CLI, including
the explicit shared modes, without requiring Docker or occupying accelerators.
The native two-container negative control is retained with the investigation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import docker_paths
import cross_host_containers


def server_function(name: str) -> str:
    """Load the production shell function for execution with fake external I/O."""
    shell = (ROOT / "tests/v2/e2e/server/test_server_e2e.sh").read_text()
    return name + "() {" + shell.split(name + "() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"


class DockerIPCIdentityTests(unittest.TestCase):
    """Exercise private storage, explicit sharing and invalid override rejection."""

    def test_default_launches_isolate_ipc_for_every_backend(self):
        """Unrelated jobs must not share SMI locks while reusing private TIDs."""
        for backend in ("CPU", "CUDA", "ROCm", "CPU+CUDA+ROCm"):
            with self.subTest(backend=backend), \
                 patch.object(docker_paths, "rocm_device_group_ids", return_value=(992,)), \
                 patch.object(docker_paths.subprocess, "check_output", return_value="--gpus\nall\n"):
                args = docker_paths.device_args("image", backend)
            self.assertEqual(args[args.index("--ipc") + 1], "private")
            self.assertEqual(args[args.index("--shm-size") + 1], "16g")
            self.assertNotIn("--pid", args)

    def test_shared_ipc_requires_the_same_pid_namespace(self):
        """Host and peer storage must use the corresponding owner-ID domain."""
        for mode in ("host", "container:peer-12", "container:abc123"):
            with self.subTest(mode=mode):
                self.assertEqual(docker_paths.container_namespace_args(mode, "16g"),
                                 ["--ipc", mode, "--pid", mode])

    def test_private_and_shareable_storage_receive_real_capacity(self):
        """Private storage has a capacity; joined storage cannot be resized."""
        for mode, expected in (("", "private"), ("none", "private"),
                               ("private", "private"), ("shareable", "shareable")):
            with self.subTest(mode=mode):
                self.assertEqual(docker_paths.container_namespace_args(mode, "32g"),
                                 ["--ipc", expected, "--shm-size", "32g"])

    def test_invalid_modes_fail_before_docker_launch(self):
        """An incomplete peer or misspelled namespace never selects a default."""
        for mode in ("container:", "container:a/b", "container:bad peer", "hosst"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "IPC"):
                docker_paths.container_namespace_args(mode, "16g")

    def test_cli_preserves_arguments_and_rejects_namespace_overrides(self):
        """The shell consumer receives one argument per line or a fatal error."""
        command = [sys.executable, str(ROOT / "scripts/ci/docker_paths.py"),
                   "--container-namespace-args", "host", "16g"]
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["--ipc", "host", "--pid", "host"])
        for override in ("--ipc=private", "--ipc", "--pid=private", "--pid", "--shm-size=8g"):
            with self.subTest(override=override):
                result = subprocess.run([*command, override], text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("namespace policy", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_devcontainer_owns_its_shared_memory(self):
        """Development kernels must not inherit host locks with private TIDs."""
        args = json.loads((ROOT / ".devcontainer/devcontainer.json").read_text())["runArgs"]
        self.assertIn("--ipc=private", args)
        self.assertIn("--shm-size=32g", args)
        self.assertNotIn("--pid=host", args)

    def test_cross_host_fleet_keeps_each_nodes_ipc_private(self):
        """Execute acquisition/retirement while observing both Docker creates."""
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(cross_host_containers, "execute", return_value="[]") as execute, \
             patch.object(cross_host_containers, "ssh_argv", side_effect=lambda peer, key, argv: ["peer", *argv]), \
             patch.object(cross_host_containers, "mpi_parameters", return_value="fixture"), \
             patch.object(docker_paths, "containing_container", return_value=None), \
             patch.object(docker_paths, "mounts", return_value=[]), \
             patch.object(docker_paths, "rocm_device_group_ids", return_value=(992,)):
            root = Path(directory)
            with cross_host_containers.container_fleet(
                    image="image", hosts=[{"public_ip": "192.0.2.2", "private_ip": "192.0.2.2",
                                           "runtime_image": "peer-image"}], key=root / "key",
                    workspace=root / "workspace", artifact=root / "artifact", model_dir=root,
                    model_name="model.gguf", frontend_mode="serve", plan_args=[], backend="rocm",
                    controller_address="192.0.2.1", continuation_devices=2):
                pass
            creates = [call.args[0] for call in execute.call_args_list
                       if "create" in call.args[0] and "docker" in call.args[0]]
            self.assertEqual(len(creates), 2)
            for args in creates:
                self.assertEqual(args[args.index("--ipc") + 1], "private")
                self.assertEqual(args[args.index("--shm-size") + 1], "16g")
                self.assertNotIn("--pid", args)

    def test_http_shell_launch_uses_coupled_policy_before_start(self):
        """Run the real shell launcher and policy CLI, mocking only Docker I/O."""
        for mode, extra in (("", []), ("host", []), ("container:peer", []),
                            ("private", ["--pid=host"]), ("invalid", [])):
            with self.subTest(mode=mode, extra=extra), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                script = server_function("start_server_process") + r'''
set -eu
is_docker_mode() { return 0; }
container_model_path() { printf '%s\n' "$1"; }
resolve_docker_network() { printf 'host\n'; }
docker_args_need_cuda() { return 1; }
append_rocm_device_nodes() { return 1; }
python3() {
    if [[ "$2" == --container-namespace-args ]]; then command python3 "$@";
    else printf '%s\n' "$2"; fi
}
docker() {
    if [[ "$1" == run ]]; then
        printf '%s\0' "$@" > "$LOG_DIR/arguments"
        printf 'container-id\n'
    fi
}
DOCKER_NAME_PREFIX=fixture DOCKER_NETWORK=host DOCKER_GPUS=none DOCKER_SHM_SIZE=16g
DOCKER_NUMA_SECCOMP=0 DOCKER_USER=0:0 CONTAINER_IMAGE=image BLUE= NC=
DOCKER_CAP_ARGS=() ACTIVE_DOCKER_CONTAINERS=() ACTIVE_LOG_FOLLOW_PIDS=()
fixture_env=() fixture_args=(serve)
''' + "DOCKER_EXTRA_ARGS=(" + shlex.join(extra) + ")\n" + r'''
start_server_process fixture 18080 "$LOG_DIR/model.gguf" "$LOG_DIR/server.log" fixture fixture_env fixture_args
wait
'''
                result = subprocess.run(["bash", "-c", script], text=True, capture_output=True,
                    env={**os.environ, "REPO_ROOT": str(ROOT), "LOG_DIR": str(root), "DOCKER_IPC": mode})
                if extra or mode == "invalid":
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse((root / "arguments").exists())
                    continue
                self.assertEqual(result.returncode, 0, result.stderr)
                args = (root / "arguments").read_bytes().decode().split("\0")[:-1]
                self.assertEqual(args[args.index("--ipc") + 1], mode or "private")
                if mode:
                    self.assertEqual(args[args.index("--pid") + 1], mode)
                    self.assertNotIn("--shm-size", args)
                else:
                    self.assertNotIn("--pid", args)
                    self.assertEqual(args[args.index("--shm-size") + 1], "16g")

    def test_shared_pid_retirement_signals_only_the_owned_container(self):
        """Execute the retirement payload against separate proc namespace fixtures."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / "proc"
            for identity, namespace in (("self", "mnt:[11]"), ("101", "mnt:[11]"),
                                        ("202", "mnt:[22]")):
                process = proc / identity
                (process / "ns").mkdir(parents=True)
                (process / "ns/mnt").symlink_to(namespace)
                (process / "comm").write_text("llaminar2\n")
                (process / "environ").write_bytes(b"OMPI_COMM_WORLD_RANK=0\0")
            script = server_function("signal_container_llaminar_processes") + r'''
docker() {
    local payload="${5//\/proc/$PROC_FIXTURE}"
    /bin/sh -c 'kill() { printf "%s\n" "$*" >> "$SIGNAL_FIXTURE"; }; '"$payload" _ "$7"
}
signal_container_llaminar_processes own-container TERM
'''
            result = subprocess.run(["bash", "-c", script], text=True, capture_output=True,
                env={**os.environ, "PROC_FIXTURE": str(proc), "SIGNAL_FIXTURE": str(root / "signals")})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / "signals").read_text().splitlines(), ["-TERM 101"])


if __name__ == "__main__":
    unittest.main()
