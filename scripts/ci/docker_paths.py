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
import re
import subprocess
import sys
import hashlib
import tempfile
import uuid


# ARC runner pods use a node-local Docker daemon through a Unix socket.
# Kubernetes mounts selected host paths into both the runner and that daemon
# namespace at the same absolute location, so Docker can bind them without
# translating through a container ID. The deployment must declare those roots
# explicitly: accepting an arbitrary caller path here would conceal an
# incomplete mount contract.
SHARED_DAEMON_ROOTS_ENV = "LLAMINAR_DOCKER_SHARED_ROOTS"

# Docker accepts a full object ID or an unambiguous hexadecimal prefix.  A
# runner/pod hostname is otherwise just a hostname: treating a friendly label
# such as "xeon" as a Docker object silently turns a topology error into an
# empty bind source at a much later launch boundary.
_DOCKER_CONTAINER_ID = re.compile(r"^[0-9a-f]{12,64}$", re.IGNORECASE)
_DOCKER_CGROUP_ID = re.compile(
    r"(?:^|/)(?:docker-)?([0-9a-f]{12,64})(?:\.scope)?(?:/|$)|"
    r"(?:^|/)(?:docker|cri-containerd)-([0-9a-f]{12,64})\.scope(?:/|$)",
    re.IGNORECASE,
)


def local_docker_endpoint() -> None:
    """Reject remote daemons before resolving any local bind source.

    Device certification and its model/cache mounts are node-local protocols.
    ARC's host-socket daemon is admitted through separately declared shared
    roots; a TCP endpoint is never such a daemon.
    """
    endpoint = os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock")
    if not endpoint.startswith("unix://"):
        raise ValueError("device certification requires a node-local Docker daemon")


def shared_daemon_roots() -> tuple[Path, ...]:
    """Return exact runner roots mounted identically into its ARC Docker daemon.

    ``LLAMINAR_DOCKER_SHARED_ROOTS`` is an infrastructure-owned, path-separator
    delimited list.  Each entry must already exist, be an absolute normalized
    directory, and be narrower than filesystem root.  This makes a missing
    Kubernetes hostPath/volumeMount a fatal configuration error instead of
    letting a future bind silently target the wrong namespace.
    """
    raw = os.environ.get(SHARED_DAEMON_ROOTS_ENV)
    if raw is None:
        return ()
    if not raw:
        raise ValueError(f"{SHARED_DAEMON_ROOTS_ENV} must not be empty when declared")
    roots: list[Path] = []
    for entry in raw.split(os.pathsep):
        if not entry:
            raise ValueError(f"{SHARED_DAEMON_ROOTS_ENV} must not contain an empty root")
        candidate = Path(entry)
        if not candidate.is_absolute():
            raise ValueError(f"{SHARED_DAEMON_ROOTS_ENV} root must be absolute: {entry!r}")
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(f"{SHARED_DAEMON_ROOTS_ENV} root does not exist: {entry!r}") from error
        if resolved != candidate:
            raise ValueError(f"{SHARED_DAEMON_ROOTS_ENV} root must be normalized: {entry!r}")
        if not resolved.is_dir() or resolved == Path("/"):
            raise ValueError(f"{SHARED_DAEMON_ROOTS_ENV} root is not a scoped directory: {entry!r}")
        if resolved in roots:
            raise ValueError(f"{SHARED_DAEMON_ROOTS_ENV} contains duplicate root: {entry!r}")
        roots.append(resolved)
    return tuple(roots)


def shared_daemon_path(path: Path) -> str | None:
    """Return an explicitly shared ARC source path, or ``None`` if not declared."""
    for root in shared_daemon_roots():
        if path.is_relative_to(root):
            return str(path)
    return None


def containing_container_candidates(hostname: str, cgroup: str) -> tuple[str, ...]:
    """Return only structurally valid Docker IDs exposed by this namespace.

    A Kubernetes/ARC runner may use a descriptive hostname that has no Docker
    object on the host daemon.  Docker cgroup syntax has a small, explicit
    grammar; scan only those forms and never promote an arbitrary cgroup
    component to an object ID.  Callers still authenticate every candidate
    with ``docker inspect`` because a container-runtime ID need not belong to
    the Docker daemon serving this process.
    """
    candidates: list[str] = []
    if _DOCKER_CONTAINER_ID.fullmatch(hostname):
        candidates.append(hostname.lower())
    for match in _DOCKER_CGROUP_ID.finditer(cgroup):
        candidate = next(value for value in match.groups() if value is not None).lower()
        if candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def containing_container() -> dict | None:
    """Identify our containing Docker namespace, or fail with a typed contract.

    Namespace translation is valid only when this process can prove that its
    Docker container is visible to the same node-local daemon.  ARC's
    same-path shared-root contract is deliberately preferred and bypasses this
    function entirely; an unrecognised pod/container must not be guessed.
    """
    local_docker_endpoint()
    if not Path("/.dockerenv").exists():
        return None
    hostname = Path("/etc/hostname").read_text(encoding="utf-8").strip()
    try:
        cgroup = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    except FileNotFoundError:
        # A compliant Docker container always has cgroups, but retaining an
        # empty value gives the hostname path a precise, portable diagnostic.
        cgroup = ""
    for candidate in containing_container_candidates(hostname, cgroup):
        try:
            inspected = json.loads(subprocess.check_output(
                ["docker", "inspect", candidate], text=True, stderr=subprocess.DEVNULL))
        except subprocess.CalledProcessError:
            # A CRI/containerd ID is not necessarily a Docker object.  It is
            # evidence to try, never permission to invent another mapping.
            continue
        if (isinstance(inspected, list) and len(inspected) == 1
                and isinstance(inspected[0], dict)
                and isinstance(inspected[0].get("Id"), str)
                and inspected[0]["Id"].lower().startswith(candidate)):
            return inspected[0]
    raise ValueError(
        "cannot identify this Docker container through the node-local daemon; "
        f"declare {SHARED_DAEMON_ROOTS_ENV} for an exact same-path ARC mount "
        "instead of relying on namespace translation")


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
    resolved = path.resolve(strict=True)
    # ARC roots are already daemon-visible at the exact same spelling.  Do not
    # enter the private-tmpfs publication path merely because the runner itself
    # happens to be containerized.
    local_docker_endpoint()
    if shared_daemon_path(resolved) is not None:
        return
    container = containing_container()
    if container is None:
        return
    root = private_model_mount(resolved)
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
    # ARC's declared DIND roots are first-class infrastructure topology, not a
    # best-effort alternative to Docker mount inspection.  Validate locality
    # even when the path itself needs no translation.
    local_docker_endpoint()
    if shared := shared_daemon_path(resolved):
        return shared
    container = containing_container()
    if container is None:
        return str(resolved)
    # Compare both sides in this container's namespace. Docker retains the
    # launch spelling (for example /var/run/docker.sock), while resolve()
    # canonicalizes the caller to /run/docker.sock. The daemon-side source is
    # deliberately NOT resolved here: its symlinks belong to another namespace.
    # Resolve before sorting so nested mounts still win through directory aliases.
    candidates = [(Path(mount["Destination"]).resolve(), mount)
                  for mount in container["Mounts"] if mount["Type"] in ("bind", "volume")]
    for destination, mount in sorted(candidates, key=lambda item: len(item[0].parts), reverse=True):
        if resolved.is_relative_to(destination):
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


def validate_attached_execution(image: str) -> None:
    """Prove Docker waits for a child and preserves delayed output and failure.

    A socket proxy can accept ordinary Docker API requests but truncate the
    half-closed, hijacked connection used by exec/attach. The CLI then reports
    success while its child still runs. Check that transport before spending
    cloud capacity or accepting frontend evidence, without retries or a second
    process-launch implementation. The caller chooses its Docker endpoint;
    this function never silently changes it.
    """
    name = "llaminar-attach-probe-" + uuid.uuid4().hex
    marker = "llaminar-completed-" + uuid.uuid4().hex
    # Match inference's host network; this local process/pipe proof must not
    # wait for unrelated bridge or firewall provisioning.
    subprocess.run(["docker", "create", "--name", name, "--network", "host", "--entrypoint", "/bin/sleep",
                    image, "infinity"], check=True, stdout=subprocess.DEVNULL, timeout=30)
    try:
        subprocess.run(["docker", "start", name], check=True,
                       stdout=subprocess.DEVNULL, timeout=30)
        # The delay exceeds the proxy's half-close window; the sentinel and
        # intentional nonzero exit must both arrive from this same child.
        observed = subprocess.run(["docker", "exec", name, "/bin/sh", "-c",
            'sleep 1; printf "%s\\n" "$1"; exit 23', "llaminar-attach-probe", marker],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
        if observed.returncode != 23 or observed.stdout != marker + "\n":
            raise RuntimeError(
                "Docker exec did not preserve child completion, output and exit status; "
                "use a directly accessible daemon socket, not a half-close-breaking proxy. "
                "The devcontainer exposes it as unix:///var/run/docker-host.sock.")
    finally:
        subprocess.run(["docker", "rm", "--force", "--volumes", name],
                       check=True, stdout=subprocess.DEVNULL, timeout=30)


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
        # Metadata never needs a bridge. Match the host-network inference
        # container so an unrelated network lifecycle cannot block admission.
        "docker", "create", "--name", name, "--network", "host", "--entrypoint", "/bin/sh",
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
    """Expose drivers/devices only for the explicitly selected backend set.

    A full image defers CUDA driver binding until CUDA preparation. CPU-only
    remote ranks and ROCm lanes therefore need no NVIDIA runtime installation.
    """
    result = ["--user", user, "--network", "host", "--ipc", "host",
              "--security-opt", "seccomp=unconfined", "--cap-add", "SYS_NICE",
              "--cap-add", "SYS_PTRACE"]
    if "CUDA" in backends:
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
    elif sys.argv[1] == "--nvidia-device-nodes":
        for node in nvidia_device_nodes(sys.argv[2]):
            print(node)
    else:
        print(host_path(Path(sys.argv[1])))
