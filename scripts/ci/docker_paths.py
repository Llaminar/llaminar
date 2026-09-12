#!/usr/bin/env python3
"""Resolve node-local Docker bind sources, including private devcontainer tmpfs.

The Docker daemon resolves sources in its own mount namespace. Ordinary bind
and volume paths map through Docker's mount inventory. A private model tmpfs
must first be explicitly published as a host bind mount; runc rejects direct
/proc/PID/root magic-link sources. Publication pins the existing pages without
copying or clearing them, and remains until manual unmount/reboot. This is
intentionally not a remote-Docker transport.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib
import tempfile
import uuid

from production_artifacts import image_identity


def containing_container() -> dict | None:
    """Identify our containing Docker namespace, or ordinary daemon-local host."""
    endpoint = os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock")
    if not endpoint.startswith("unix://"):
        raise ValueError("device certification requires a node-local Docker daemon")
    if not Path("/.dockerenv").exists():
        return None
    container = Path("/etc/hostname").read_text().strip()
    return json.loads(subprocess.check_output(["docker", "inspect", container], text=True))[0]


def private_model_mount(path: Path) -> Path:
    """Only the explicitly owned parity tmpfs may be published by this tool."""
    mount = json.loads(subprocess.check_output(["findmnt", "-J", "-T", str(path)], text=True))["filesystems"][0]
    if mount["fstype"] != "tmpfs" or mount["source"] != "llaminar-production-parity":
        raise ValueError(f"{path} has no daemon-visible bind mapping or owned model tmpfs")
    return Path(mount["target"])


def export_path(container: dict, root: Path) -> str:
    """Use a collision-free, narrowly owned host mount destination."""
    identity = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    return f"/mnt/llaminar-docker-tmpfs/{container['Id']}/{identity}"


def publish_model_cache(path: Path) -> None:
    """Idempotently pin the private tmpfs in the daemon's host namespace.

    This is a setup operation, not an inference fallback. It requires a
    privileged node-local helper. Existing exports must refer to the same
    device/inode; nothing is overwritten or automatically unmounted.
    """
    container = containing_container()
    if container is None:
        return
    root = private_model_mount(path)
    pid = container["State"]["Pid"]
    if type(pid) is not int or pid <= 0:
        raise ValueError("invalid containing container host PID")
    destination = export_path(container, root)
    source = f"/proc/{pid}/root{root}"
    helper = Path(__file__).with_name("publish_model_tmpfs.py").read_text()
    subprocess.run(["docker", "run", "--rm", "-i", "--privileged", "--pid=host", "ubuntu:24.04",
                    "nsenter", "-t", "1", "-m", "-r", "--", "/usr/bin/python3", "-",
                    source, destination], input=helper, text=True, check=True, timeout=30)
    print(f"[production-ci] model tmpfs published at {destination}; retained until manual unmount/reboot", file=sys.stderr)


def host_path(path: Path) -> str:
    """Return the daemon-visible path for an existing local file or directory."""
    resolved = path.resolve(strict=True)
    container = containing_container()
    if container is None:
        return str(resolved)
    # Prefer the most specific mount when a workspace contains another volume.
    for mount in sorted(container["Mounts"], key=lambda item: len(item["Destination"]), reverse=True):
        destination = Path(mount["Destination"])
        if resolved.is_relative_to(destination) and mount["Type"] in ("bind", "volume"):
            return str(Path(mount["Source"]) / resolved.relative_to(destination))
    root = private_model_mount(resolved)
    return str(Path(export_path(container, root)) / resolved.relative_to(root))


def mounts(pairs: list[tuple[Path, str, bool]]) -> list[str]:
    """Build explicit bind mounts; --mount rejects absent daemon-side sources."""
    result = []
    for source, destination, readonly in pairs:
        origin = host_path(source)
        if "," in origin + destination:
            raise ValueError("Docker bind paths may not contain commas")
        result += ["--mount", f"type=bind,src={origin},dst={destination}" +
                   (",readonly" if readonly else "")]
    return result


def cuda_driver_required(image: str) -> bool:
    """Honor image linkage independently of the selected inference topology.

    A full-backend shared core links the NVIDIA driver even when the request
    selects only CPU or ROCm. Host driver injection is a deployment dependency;
    it does not authorize the engine to execute on an unselected device.
    """
    return image_identity(image)["labels"].get("org.llaminar.cuda") == "ON"


def daemon_device_metadata(image: str, script: str) -> str:
    """Read one completed device probe without depending on Docker attach.

    Short-lived containers can finish with complete logs while the attached
    client receives no stdout. Metadata is a control-plane input, so publish
    it in the owned container's filesystem and copy it only after successful
    completion. There is one execution, no retry or permission workaround.
    The read-only daemon /dev mount opens no accelerator. Every created probe
    is retired on success, script failure, timeout, or failed result transfer.
    """
    name = "llaminar-device-probe-" + uuid.uuid4().hex
    result_path = "/tmp/llaminar-device-metadata"
    subprocess.run([
        "docker", "create", "--name", name, "--entrypoint", "/bin/sh",
        "--mount", "type=bind,src=/dev,dst=/host-dev,readonly", image,
        "-c", 'exec /bin/sh -eu -c "$1" > "$2"',
        "llaminar-device-probe", script, result_path],
        check=True, stdout=subprocess.DEVNULL, timeout=30)
    try:
        subprocess.run(["docker", "start", name], check=True,
                       stdout=subprocess.DEVNULL, timeout=30)
        status = subprocess.check_output(["docker", "wait", name], text=True, timeout=30).strip()
        if status != "0":
            raise RuntimeError(f"Docker device metadata probe {name} exited with status {status!r}")
        with tempfile.TemporaryDirectory(prefix="llaminar-device-metadata-") as directory:
            output = Path(directory) / "result"
            subprocess.run(["docker", "cp", name + ":" + result_path, str(output)],
                           check=True, stdout=subprocess.DEVNULL, timeout=30)
            return output.read_text(encoding="utf-8")
    finally:
        subprocess.run(["docker", "rm", "--force", "--volumes", name],
                       check=True, stdout=subprocess.DEVNULL, timeout=30)


def rocm_device_group_ids(image: str) -> tuple[int, ...]:
    """Read numeric device owners in the daemon's namespace, not our container.

    Docker recreates --device nodes with the host's mode and ownership. A
    devcontainer may have different GIDs or chmods on its own nodes, so its
    supplementary groups cannot prove access in a sibling test container.
    This metadata-only probe opens no accelerator and changes no permissions.
    """
    script = ('set -eu; test -c /host-dev/kfd; '
              'for node in /host-dev/kfd /host-dev/dri/card* /host-dev/dri/renderD*; do '
              'if [ -c "$node" ]; then stat -c %g "$node"; fi; done')
    output = daemon_device_metadata(image, script)
    groups = output.splitlines()
    if not groups or any(not value.isdecimal() for value in groups):
        raise ValueError("Docker daemon did not report valid ROCm device group IDs")
    return tuple(sorted({int(value) for value in groups}))


def nvidia_device_nodes(image: str) -> tuple[str, ...]:
    """Resolve NVIDIA passthrough nodes through the same completed probe.

    Missing nodes remain an empty inventory for the caller's explicit required
    hardware check. A failed probe is an error, not an alternate device map.
    """
    script = ('for node in /host-dev/nvidiactl /host-dev/nvidia-uvm '
              '/host-dev/nvidia-uvm-tools /host-dev/nvidia-modeset '
              '/host-dev/nvidia[0-9]* /host-dev/nvidia-caps/*; do '
              'if [ -c "$node" ]; then printf "/dev%s\\n" "${node#/host-dev}"; fi; done')
    nodes = daemon_device_metadata(image, script).splitlines()
    if any(not node.startswith("/dev/nvidia") or ".." in Path(node).parts for node in nodes):
        raise ValueError("Docker daemon reported an invalid NVIDIA device path")
    return tuple(sorted(set(nodes)))


def device_args(image: str, backends: str, *, user: str = "0:0") -> list[str]:
    """Supply image driver dependencies and the selected ROCm device nodes."""
    result = ["--user", user, "--network", "host", "--ipc", "host",
              "--security-opt", "seccomp=unconfined", "--cap-add", "SYS_NICE",
              "--cap-add", "SYS_PTRACE"]
    if "CUDA" in backends or cuda_driver_required(image):
        helper = Path(__file__).with_name("docker_gpu_run_args.sh")
        result += subprocess.check_output(["bash", str(helper), "--probe-image", image,
                                           "--required"], text=True).splitlines()
    if "ROCm" in backends:
        result += ["--device=/dev/kfd", "--device=/dev/dri"]
        for group in rocm_device_group_ids(image):
            result += ["--group-add", str(group)]
    return result


if __name__ == "__main__":
    if sys.argv[1] == "--publish-model-cache":
        publish_model_cache(Path(sys.argv[2]))
    elif sys.argv[1] == "--cuda-driver-required":
        print("yes" if cuda_driver_required(sys.argv[2]) else "no")
    elif sys.argv[1] == "--nvidia-device-nodes":
        for node in nvidia_device_nodes(sys.argv[2]):
            print(node)
    else:
        print(host_path(Path(sys.argv[1])))
