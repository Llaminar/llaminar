#!/usr/bin/env python3
"""Own the containers enclosing one public cross-host MPI frontend invocation.

MPI bootstrap runs inside the controller image. Its SSH agent starts each
remote MPI daemon inside that same image, so the daemon and inference child
share libraries, process environment, and local rendezvous files. This module
does not create an MPI job, choose a topology, or replace Llaminar's bootstrap.
It also runs as the small copied launcher inside these temporary containers;
those roles use the standard library only and never import repository code.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import uuid

CASE_ROOT = Path("/run/llaminar-cross-host")
CONTROL = str(CASE_ROOT / "control.py")
CONFIG = str(CASE_ROOT / "fleet.json")
KEY = "/run/llaminar-cross-host-key"


def execute(argv: list[str], *, timeout: int = 60) -> str:
    """Complete one bounded infrastructure command, retaining failure output."""
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=timeout)


def ssh_argv(peer: dict, key: str, command: list[str]) -> list[str]:
    """Keep OpenMPI's shell payload inside one quoted remote container command."""
    return ["ssh", "-i", key, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
            f"llaminar@{peer['public_ip']}", shlex.join(command)]


def daemon(configuration: dict, argv: list[str]) -> None:
    """Enter only the declared peer container for MPI's ordinary daemon launch.

    OpenMPI supplies the destination then a shell command (possibly split over
    arguments). Its shell syntax is evaluated inside the remote image, where
    MPI is installed. No application MPI variables cross a Docker-run boundary.
    """
    if len(argv) < 2:
        raise ValueError("MPI SSH agent requires a destination and daemon command")
    matches = [peer for peer in configuration["peers"] if peer["mpi_ip"] == argv[0]]
    if len(matches) != 1:
        raise ValueError("MPI attempted to launch a daemon outside the admitted container fleet")
    peer = matches[0]
    command = ["docker", "exec", peer["container"], "/bin/sh", "-c", " ".join(argv[1:])]
    os.execvp("ssh", ssh_argv(peer, KEY, command))


def run_inside(configuration: dict, argv: list[str]) -> int:
    """Publish the exact child group before waiting; stop addresses this owner.

    This is an OS process owner only. The real CLI receives the full hostfile
    and owns discovery, planning, selection and inference. The marker remains
    inside the temporary controller container and is never a model-state flag.
    """
    marker = CASE_ROOT / "frontend-process.json"
    env = {**os.environ,
           "OMPI_MCA_plm_rsh_agent": f"/usr/bin/python3 {CONTROL} daemon {CONFIG}",
           "OMPI_MCA_plm_rsh_no_tree_spawn": "1"}
    with subprocess.Popen(["/usr/local/bin/llaminar2", *argv], env=env,
                          start_new_session=True) as process:
        marker.write_text(json.dumps({"pid": process.pid}))
        try:
            return process.wait()
        finally:
            marker.unlink(missing_ok=True)


def stop_inside() -> None:
    """Signal the live frontend group while allowing ordinary MPI shutdown."""
    marker = CASE_ROOT / "frontend-process.json"
    if not marker.exists():
        return
    pid = json.loads(marker.read_text())["pid"]
    if type(pid) is not int or pid <= 1:
        raise ValueError("invalid owned frontend process identity")
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # The sole child owner can finish between the read and signal.


def collect_peer_evidence(configuration: dict) -> None:
    """Collect every rank's files before the HTTP observer validates membership.

    Independent peers download concurrently. SSH compression matters here:
    complete diagnostic JSON is large and highly repetitive, unlike model
    weights. The received files themselves remain unchanged plain JSON.
    Peer directories remain separate until every download and validation
    completes. A collision then fails before publishing any peer's files.
    Only the case output directory is copied; controller keys are never there.
    """
    artifact = Path(configuration["artifact"])
    peers = configuration["peers"]
    if not peers:
        return

    def download(index_peer: tuple[int, dict]) -> Path:
        """Collect one exact owned peer with the existing bounded subprocess calls."""
        index, peer = index_peer
        target = artifact / f"peer-{index}"
        target.mkdir()
        remote = peer["directory"] + "/collected"
        execute(ssh_argv(peer, configuration["key"], ["mkdir", "-p", remote]))
        execute(ssh_argv(peer, configuration["key"],
            ["docker", "cp", f"{peer['container']}:{CASE_ROOT}/output/.", remote]))
        execute(["scp", "-C", "-q", "-r", "-i", configuration["key"], "-o", "BatchMode=yes",
                 "-o", "StrictHostKeyChecking=accept-new",
                 f"llaminar@{peer['public_ip']}:{remote}/.", str(target)], timeout=60)
        return target

    # The executor joins every owned operation even when a peer fails. Each
    # child retains the frontend's process group and the existing cell-wide
    # watchdog still bounds cancellation; there is no detached transfer owner.
    with ThreadPoolExecutor(max_workers=len(peers)) as workers:
        targets = list(workers.map(download, enumerate(peers, 1)))
    files: list[tuple[Path, Path]] = []
    destinations: set[Path] = set()
    for target in targets:
        for path in target.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError("remote evidence must contain only regular rank artifacts")
            destination = artifact / path.name
            if destination.exists() or destination.is_symlink() or destination in destinations:
                raise ValueError(f"remote ranks duplicated an artifact: {path.name}")
            destinations.add(destination)
            files.append((path, destination))
    for path, destination in files:
        shutil.copyfile(path, destination)


def runtime_environment(environment: dict[str, str]) -> dict[str, str]:
    """Forward engine observation policy without the Azure controller settings.

    CI may supply cloud configuration through LLAMINAR_AZURE_* variables.
    Those belong exclusively to the outer lease owner, never an MPI child.
    """
    return {name: value for name, value in environment.items()
            if name.startswith("LLAMINAR_") and not name.startswith("LLAMINAR_AZURE_")}


def mpi_parameters(address: str, interfaces: list[dict]) -> str:
    """Select one actual host interface for both MPI control and payload traffic.

    Resolve the admitted IPv4 address against that host's live interface list.
    Every container shares its host's network namespace and reads a different
    file at the same path. Never broadcast a controller interface to peers.

    Use the exact interface name, not a /32 CIDR: OpenMPI 4.1's mask conversion
    shifts a 32-bit integer by 32, which can yield a zero mask and admit every
    interface, including loopback. A missing or ambiguous address is fatal;
    neither a guessed device name nor an expanded subnet preserves admission.
    """
    import ipaddress
    import re
    selected = ipaddress.IPv4Address(address)
    if selected.is_loopback or selected.is_unspecified or selected.is_multicast:
        raise ValueError("MPI requires an assigned non-loopback unicast IPv4 address")
    matches = {item["ifname"] for item in interfaces
               if "UP" in item.get("flags", []) and
               any(entry.get("family") == "inet" and entry.get("local") == str(selected)
                   for entry in item.get("addr_info", []))}
    if len(matches) != 1:
        raise ValueError(f"MPI address {selected} must belong to exactly one active interface")
    interface = matches.pop()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]*", interface):
        raise ValueError("MPI interface name is not an unambiguous MCA device selector")
    return f"oob_tcp_if_include={interface}\nbtl_tcp_if_include={interface}\n"


def container_cli_command(configuration: dict, argv: list[str]) -> list[str]:
    """Translate mounted inputs and observation variables for one public CLI job."""
    container = configuration["controller"]
    base = ["docker", "exec"]
    for name, value in runtime_environment(os.environ).items():
        if name == "LLAMINAR_PERF_STATS_JSON":
            value = str(CASE_ROOT / "output" / Path(value).name)
        base += ["-e", f"{name}={value}"]
    base += [container, "python3", CONTROL, "run", CONFIG]
    args = [configuration["container_model"] if value == configuration["host_model"] else value
            for value in argv]
    return base + args


def prepare_plan(configuration: dict) -> int:
    """Run and distribute the public plan before serve's readiness clock starts.

    The outer owner charges this process to the same cell budget as HTTP work.
    Cancellation stops its real MPI job; no partial plan can enter serving.
    Direct auto-serve has no separate planning phase and cannot call this role.
    """
    if configuration["frontend"] != "plan-apply":
        raise ValueError("only plan/apply has a separate plan preparation phase")

    def interrupted(signum, _frame):
        """Unwind into the exact in-container process owner on cancellation."""
        raise InterruptedError(f"Plan preparation stopped with signal {signum}")

    old = signal.signal(signal.SIGTERM, interrupted)
    try:
        subprocess.run(container_cli_command(configuration, configuration["plan_args"]), check=True)
        for peer in configuration["peers"]:
            execute(["scp", "-q", "-i", configuration["key"], "-o", "BatchMode=yes",
                     "-o", "StrictHostKeyChecking=accept-new",
                     str(Path(configuration["workspace"]) / "plan.json"),
                     f"llaminar@{peer['public_ip']}:{peer['directory']}/plan.json"])
        return 0
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            execute(["docker", "exec", configuration["controller"], "python3", CONTROL, "stop", CONFIG])
        finally:
            signal.signal(signal.SIGTERM, old)


def frontend(configuration: dict, argv: list[str]) -> int:
    """Run only serve inside the HTTP harness's readiness window.

    Plan/apply consumes the already prepared document; auto-serve performs its
    own normal in-process planning. On shutdown the proxy retires the exact
    CLI/MPI owner and collects rank files before the HTTP observer proceeds.
    """
    container = configuration["controller"]
    process = None

    def interrupted(signum, _frame):
        """Leave the normal wait so the single finally owner stops the child."""
        raise InterruptedError(f"HTTP harness stopped frontend with signal {signum}")

    old = signal.signal(signal.SIGTERM, interrupted)
    try:
        process = subprocess.Popen(container_cli_command(configuration, argv))
        return process.wait()
    except InterruptedError:
        return 0  # The HTTP harness deliberately ends a healthy persistent server.
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            execute(["docker", "exec", container, "python3", CONTROL, "stop", CONFIG])
            if process is not None:
                process.wait(timeout=30)
            collect_peer_evidence(configuration)
        finally:
            signal.signal(signal.SIGTERM, old)


@contextmanager
def container_fleet(*, image: str, hosts: list[dict], key: Path, workspace: Path,
                    artifact: Path, model_dir: Path, model_name: str,
                    frontend_mode: str, plan_args: list[str], backend: str,
                    controller_address: str, continuation_devices: int):
    """Acquire exact named containers and retire all acquired owners on every exit.

    Mount paths are resolved through the repository's Docker namespace mapper.
    All children use the supplied immutable image, including remote MPI daemons;
    no host MPI ABI or Docker entrypoint assumption enters inference execution.
    """
    import docker_paths

    if type(continuation_devices) is not int or continuation_devices <= 0:
        raise ValueError("container fleet requires a positive declared GPU inventory")
    identity = "llaminar-remote-" + uuid.uuid4().hex
    workspace.mkdir(parents=True)
    artifact.mkdir(parents=True, exist_ok=True)
    peers = [{**peer, "container": f"{identity}-{index}",
              "mpi_ip": peer.get("mpi_ip", peer["private_ip"]),
              "directory": f"/home/llaminar/{identity}-{index}"}
             for index, peer in enumerate(hosts, 1)]
    configuration = {"controller": identity + "-0", "peers": peers, "key": str(key),
        "workspace": str(workspace), "artifact": str(artifact), "frontend": frontend_mode,
        "host_model": str(model_dir / model_name), "container_model": f"/opt/llaminar-models/{model_name}",
        "plan_args": plan_args}
    (workspace / "fleet.json").write_text(json.dumps(configuration))
    (workspace / "hosts").write_text("\n".join(
        f"{address} slots=1" for address in [controller_address, *(p["mpi_ip"] for p in peers)]) + "\n")
    (workspace / "plan.json").touch()
    (workspace / "mpi.conf").write_text(mpi_parameters(controller_address,
        json.loads(execute(["ip", "-j", "-4", "address", "show", "up"]))))
    for index, peer in enumerate(peers, 1):
        inventory = json.loads(execute(ssh_argv(peer, str(key),
            ["ip", "-j", "-4", "address", "show", "up"])))
        (workspace / f"mpi-peer-{index}.conf").write_text(mpi_parameters(peer["mpi_ip"], inventory))
    shutil.copyfile(__file__, workspace / "control.py")
    launcher = workspace / "launch"
    launcher.write_text("#!/bin/sh\nexec " + shlex.join([
        sys.executable, str(Path(__file__).resolve()), "frontend", str(workspace / "fleet.json")]) + ' "$@"\n')
    launcher.chmod(0o700)
    acquired = []
    try:
        # The manifest uses CLI spellings; Docker's backend inventory retains
        # the public backend names. Preserve ROCm's case so its devices are
        # actually admitted instead of silently producing a CPU-only image.
        local_options = docker_paths.device_args(image, {"cuda": "CUDA", "rocm": "ROCm"}[backend])
        # The fixture supplies a physical inventory, not a hand-built execution
        # plan. Expose exactly its declared accelerator count; otherwise auto
        # discovery on a larger host silently exercises a different topology.
        visibility = "CUDA_VISIBLE_DEVICES" if backend == "cuda" else "ROCR_VISIBLE_DEVICES"
        local_options += ["-e", visibility + "=" + ",".join(map(str, range(continuation_devices)))]
        containing = docker_paths.containing_container()
        if containing is not None:
            # The authenticated tunnel and HTTP client live in the controller
            # namespace; --network host would instead select the daemon host.
            local_options[local_options.index("--network") + 1] = "container:" + containing["Id"]
        for index, peer in enumerate([None, *peers]):
            name = configuration["controller"] if peer is None else peer["container"]
            runtime_image = image if peer is None else peer["runtime_image"]
            command = execute if peer is None else lambda cmd, peer=peer: execute(ssh_argv(peer, str(key), cmd))
            if peer is None:
                mounts = docker_paths.mounts([(workspace, str(CASE_ROOT), False),
                    (artifact, str(CASE_ROOT / "output"), False),
                    (model_dir, "/opt/llaminar-models", True)])
                options = local_options
            else:
                command(["mkdir", "-p", peer["directory"] + "/output"])
                # Docker may create the nested output mountpoint in the local
                # workspace. Transfer only declared launch files, never a
                # directory scan that could include outputs or credentials.
                for filename in ("fleet.json", "hosts", "plan.json", "control.py"):
                    path = workspace / filename
                    execute(["scp", "-q", "-i", str(key), "-o", "BatchMode=yes",
                             "-o", "StrictHostKeyChecking=accept-new", str(path),
                             f"llaminar@{peer['public_ip']}:{peer['directory']}/{path.name}"])
                execute(["scp", "-q", "-i", str(key), "-o", "BatchMode=yes",
                         "-o", "StrictHostKeyChecking=accept-new", str(workspace / f"mpi-peer-{index}.conf"),
                         f"llaminar@{peer['public_ip']}:{peer['directory']}/mpi.conf"])
                mounts = ["--mount", f"type=bind,src={peer['directory']},dst={CASE_ROOT}",
                          "--mount", "type=bind,src=/opt/llaminar-models,dst=/opt/llaminar-models,readonly"]
                options = ["--user", "0:0", "--network", "host", "--ipc", "host",
                           "--security-opt", "seccomp=unconfined", "--cap-add", "SYS_NICE"]
            command(["docker", "create", "--name", name, "--label", f"llaminar.remote-owner={identity}",
                     *options, *mounts, "-e", f"OMPI_MCA_mca_base_param_files={CASE_ROOT}/mpi.conf",
                     "--entrypoint", "/bin/sleep", runtime_image, "infinity"])
            acquired.append((command, name))
            command(["docker", "start", name])
        # Only the controller receives this ephemeral credential; no image
        # commit or artifact collection ever includes the container's /run key.
        execute(["docker", "cp", str(key), f"{configuration['controller']}:{KEY}"])
        execute(["docker", "exec", configuration["controller"], "chmod", "600", KEY])
        yield launcher
    finally:
        errors = []
        for command, name in reversed(acquired):
            try:
                command(["docker", "rm", "--force", "--volumes", name])
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("cross-host container retirement failed", errors)


def main() -> int:
    """Dispatch only the narrow process/container infrastructure roles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("daemon", "run", "stop", "prepare", "frontend"))
    parser.add_argument("configuration", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    configuration = json.loads(args.configuration.read_text())
    if args.role == "daemon":
        daemon(configuration, args.arguments)
    elif args.role == "run":
        return run_inside(configuration, args.arguments)
    elif args.role == "stop":
        stop_inside()
    elif args.role == "prepare":
        return prepare_plan(configuration)
    else:
        return frontend(configuration, args.arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
