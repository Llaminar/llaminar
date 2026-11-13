## @file test_mpi_native_daemon.py
#  @brief Exercise native OpenMPI daemon launch and host-local backing ownership.
"""Actual native daemon path over a declared same-host rsh adapter.

Only the rsh transport hop is local. OpenMPI constructs its actual launch-agent
command, the production owner starts actual orted, and actual ranks communicate.
This does not certify SSH authentication or a cross-host filesystem namespace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import shlex
import shutil
import subprocess
import sys
import tempfile
import time


def _atomic(path: Path, value: dict) -> None:
    """Publish complete observations outside native-owned runtime roots."""
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def _read(path: Path) -> dict | None:
    """Consume only the native probe's atomic publication."""
    return json.loads(path.read_text()) if path.is_file() else None


def _hash(path: Path) -> str:
    """Bind observations to the exact helper, module and native probe."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset(path: Path) -> dict:
    """Capture exact bytes and physical file identity, rejecting concurrent replacement."""
    before = path.stat()
    digest = _hash(path)
    after = path.stat()
    fields = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if fields(before) != fields(after):
        raise AssertionError(f"Native asset changed while hashing: {path}")
    return {"sha256": digest, "device": after.st_dev, "inode": after.st_ino,
            "bytes": after.st_size, "mtime_ns": after.st_mtime_ns}


def _rsh(directory: Path, arguments: list[str]) -> int:
    """Preserve native rsh shell grammar while making the transport same-host."""
    with (directory / "transport.jsonl").open("a") as log:
        log.write(json.dumps({"pid": os.getpid(), "argv": arguments,
                              "temporary": os.environ.get("TMPDIR"),
                              "shared": os.environ.get("OMPI_MCA_btl_vader_backing_directory")}) + "\n")
    if len(arguments) < 2 or arguments[0] != "127.0.0.2":
        raise RuntimeError("Unexpected native same-host transport arguments")
    os.execv("/bin/sh", ["sh", "-c", " ".join(arguments[1:])])
    raise AssertionError("execv returned")


def _failure_receipt(result: dict, directory: Path) -> None:
    """Preserve bounded failure facts before an outer CTest owner deletes backing."""
    def bounded(value):
        if isinstance(value, str):
            return value[:2048]
        if isinstance(value, dict):
            return {key: bounded(item) for key, item in value.items()}
        if isinstance(value, list):
            return [bounded(item) for item in value[:16]]
        return value

    fields = ("case", "error", "final_returncode", "helper_sha256", "probe_sha256",
              "selected_module_sha256", "original_daemon_bindings", "original_daemon_roots",
              "daemon_observation", "observations", "changed_assets", "remaining_original_processes")
    receipt = {key: result[key] for key in fields if key in result}
    receipt["process_tail"] = result.get("process_events", [])[-16:]
    output = directory / "output.log"
    if output.is_file():
        receipt["native_output_tail"] = output.read_text(errors="replace")[-8192:]
    receipt = bounded(receipt)
    encoded = json.dumps(receipt)
    if len(encoded.encode()) > 65536:
        receipt.pop("process_tail", None)
        receipt["truncated"] = True
        encoded = json.dumps(receipt)
    if len(encoded.encode()) > 65536:
        receipt = {"case": result.get("case"), "error": str(result.get("error"))[:2048],
                   "truncated": True, "observations": [
                       {key: bounded(rank.get(key)) for key in ("pid", "rank", "temporary", "shared")}
                       for rank in result.get("observations", [])[:2] if rank],
                   "daemon_observation": bounded(result.get("daemon_observation"))}
        encoded = json.dumps(receipt)
    print("MPI_DAEMON_FAILURE " + encoded, flush=True)


class ObservedTree:
    """Retain exact kernel exit witnesses for the test-owned process subtree.

    This monitor is diagnostic only. The native guardian remains the physical
    file owner and is solely responsible for signaling, reaping and deleting.
    """

    def __init__(self) -> None:
        self.handles: dict[int, int] = {}
        self.events: list[dict] = []
        self.last: dict[int, tuple] = {}
        self.required: list[tuple[int, int]] = []

    def sample(self, parent: int) -> None:
        """Observe owned descendants without signaling or opening unrelated roots."""
        try:
            children = [int(value) for value in Path(
                f"/proc/{parent}/task/{parent}/children").read_text().split()]
        except FileNotFoundError:
            return
        for pid in children:
            try:
                if pid not in self.handles:
                    self.handles[pid] = os.pidfd_open(pid)
                command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                command = [part.decode(errors="replace") for part in command if part]
                environment = dict(part.split(b"=", 1) for part in Path(
                    f"/proc/{pid}/environ").read_bytes().split(b"\0") if b"=" in part)
                temporary = environment.get(b"TMPDIR", b"").decode()
                shared = environment.get(b"OMPI_MCA_btl_vader_backing_directory", b"").decode()
                state = (tuple(command), temporary, shared)
                if self.last.get(pid) != state:
                    self.events.append({"pid": pid, "argv": command,
                                        "temporary": temporary, "shared": shared})
                    self.last[pid] = state
            except (ProcessLookupError, FileNotFoundError, PermissionError):
                # exec can temporarily revoke proc inspection; required live
                # daemon facts are checked separately after rank publication.
                pass
            self.sample(pid)

    def retain_live(self, pid: int) -> int:
        """Pin a required rank/daemon identity independently of discovery samples."""
        descriptor = os.pidfd_open(pid)
        self.required.append((pid, descriptor))
        self.require_live(descriptor)
        return descriptor

    @staticmethod
    def require_live(descriptor: int) -> None:
        """Reject required process retirement while attribution is being observed."""
        poll = select.poll()
        poll.register(descriptor, select.POLLIN)
        if poll.poll(0):
            raise AssertionError("Required native rank or daemon retired before attribution")

    def remaining(self) -> list[int]:
        """Report only retained original pidfds that have no terminal observation."""
        result = []
        for pid, descriptor in [*self.handles.items(), *self.required]:
            poll = select.poll()
            poll.register(descriptor, select.POLLIN)
            if not poll.poll(0):
                result.append(pid)
        return result

    def close(self) -> None:
        """Release diagnostic descriptors after native retirement is checked."""
        for descriptor in self.handles.values():
            os.close(descriptor)
        self.handles.clear()
        for _, descriptor in self.required:
            os.close(descriptor)
        self.required.clear()


def _environment(base: Path) -> dict[str, str]:
    """Choose CPU-only fixture policy while preserving authored native arguments."""
    result = dict(os.environ, TMPDIR=str(base), HWLOC_COMPONENTS="-gl",
                  OMP_NUM_THREADS="1", OMP_DYNAMIC="false",
                  OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_NAMESPACE", "MPI_LOCALRANKID", "OMPI_MCA_schizo"):
        result.pop(name, None)
    return result


def _identity(path: str) -> tuple[int, int]:
    """Observe actual backing identity while the declared native owner is live."""
    if not path:
        raise AssertionError("Native owner published an empty physical root path")
    info = Path(path).stat()
    return info.st_dev, info.st_ino


def _process_lifetime(pid: int) -> tuple[int, int]:
    """Read PPID and kernel start time without confusing parenthesized comm text."""
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[1]), int(fields[19])


def _rank_daemon(observations: list[dict], probe: Path, tree: ObservedTree) -> tuple[dict, dict, str]:
    """Bind one live native daemon through both actual ranks' stable kernel parents.

    Fork children initially retain orted's command line before exec. Discovery
    events therefore cannot establish daemon identity. Published probe reports
    keep ranks live; pin their pidfds, verify the real probe executable, and use
    their current PPIDs before taking one consistent native-parent snapshot.
    """
    parents = []
    ranks = []
    for rank in observations:
        pid = rank["pid"]
        if not isinstance(pid, int) or pid <= 0 or pid in [item["pid"] for item in ranks]:
            raise AssertionError("Invalid or duplicate actual rank identity")
        descriptor = tree.retain_live(pid)
        before = _process_lifetime(pid)
        if _identity(f"/proc/{pid}/exe") != _identity(str(probe)):
            raise AssertionError("Published rank is not the selected actual native probe")
        if before != _process_lifetime(pid):
            raise AssertionError("Actual rank changed parent or lifetime during attribution")
        tree.require_live(descriptor)
        parents.append(before[0])
        ranks.append({"pid": pid, "parent": before[0], "start_time": before[1]})
    if len(set(parents)) != 1 or parents[0] <= 0:
        raise AssertionError("Actual ranks do not share their declared single native daemon")
    pid = parents[0]
    descriptor = tree.retain_live(pid)
    before = _process_lifetime(pid)
    executable = Path(f"/proc/{pid}/exe").resolve(strict=True)
    executable_identity = _identity(f"/proc/{pid}/exe")
    if executable.name != "orted":
        raise AssertionError("Actual rank parent is not the selected native orted")
    command = Path(f"/proc/{pid}/cmdline").read_bytes()
    environment = dict(part.split(b"=", 1) for part in Path(
        f"/proc/{pid}/environ").read_bytes().split(b"\0") if b"=" in part)
    maps = Path(f"/proc/{pid}/maps").read_text()
    if (_process_lifetime(pid) != before or
            _identity(f"/proc/{pid}/exe") != executable_identity or
            Path(f"/proc/{pid}/cmdline").read_bytes() != command):
        raise AssertionError("Actual native daemon changed image/lifetime during attribution")
    tree.require_live(descriptor)
    for rank in ranks:
        if _process_lifetime(rank["pid"]) != (pid, rank["start_time"]):
            raise AssertionError("Actual rank lost its pinned native daemon during attribution")
    owner = {"pid": pid, "parent": before[0], "start_time": before[1],
             "executable": str(executable), "executable_identity": executable_identity,
             "argv": [part.decode(errors="replace") for part in command.split(b"\0") if part],
             "temporary": environment.get(b"TMPDIR", b"").decode(),
             "shared": environment.get(b"OMPI_MCA_btl_vader_backing_directory", b"").decode(),
             "rank_parents": ranks}
    return owner, environment, maps


class DaemonReplay:
    """One exact native candidate and explicit same-host transport contract."""

    def __init__(self, args: argparse.Namespace, root: Path) -> None:
        self.args, self.root = args, root
        self.results: list[dict] = []

    def run(self, mode: str) -> None:
        """Complete a real collective, then prove native retirement and path attribution."""
        directory = self.root / mode
        directory.mkdir()
        launcher = directory / "launcher"
        daemon = directory / "daemon"
        reports = directory / "reports"
        for path in (launcher, daemon, reports):
            path.mkdir()
        # Keep the selected Python interpreter explicit. This transport adapter
        # is test-private and never enters production launch or ownership code.
        adapter = directory / "local-rsh"
        adapter.write_text(f"#!{sys.executable}\nimport runpy,sys\n"
                           f"sys.argv[1:1]=['--local-rsh',{str(directory)!r}]\n"
                           f"runpy.run_path({str(Path(__file__).resolve())!r},run_name='__main__')\n")
        adapter.chmod(0o700)
        helper = self.args.helper.resolve()
        negative = mode in ("missing", "incompatible", "incompatible-version", "unselected-inclusion", "unselected-exclusion")
        if mode in ("missing", "incompatible", "incompatible-version"):
            prefix = directory / "isolated-prefix"
            prefix.mkdir()
            shutil.copy2(helper, prefix / helper.name)
            helper = prefix / helper.name
            if mode in ("incompatible", "incompatible-version"):
                components = prefix / "llaminar_mpi_components"
                components.mkdir()
                shutil.copy2(self.args.incompatible_version_module if mode == "incompatible-version" else self.args.incompatible_module,
                             components / "mca_schizo_llaminar.so")
        command = [str(helper), "--session-base", str(launcher), "--",
                   str(self.args.mpirun), "-np", "2", "--host", "127.0.0.2:2",
                   "--bind-to", "none", "--mca", "plm", "rsh",
                   "--mca", "plm_rsh_force_rsh", "1", "--mca", "plm_rsh_no_tree_spawn", "1",
                   "--mca", "plm_rsh_agent", str(adapter),
                   "--mca", "btl_vader_single_copy_mechanism", "none",
                   "--launch-agent", shlex.quote(str(helper)) + " --mpi-daemon --session-base "
                   + shlex.quote(str(daemon)) + " -- orted", "--", str(self.args.probe),
                   str(reports), "wait" if mode == "wait" else "inspect"]
        native_receipt = directory / "native-status.json"
        if mode == "wait":
            command = command[:4] + [sys.executable, str(self.args.status_helper),
                                     "--mpi-receipt", str(native_receipt), "--"] + command[4:]
        result = {"scope": "same-host native remote-daemon branch; no cross-host namespace proof",
                  "case": mode, "command": command, "helper_sha256": _hash(helper),
                  "probe_sha256": _hash(self.args.probe),
                  "selected_module": str(self.args.module.resolve()),
                  "selected_module_sha256": _hash(self.args.module)}
        assets = {str(path.resolve()): _asset(path) for path in
                  (helper, self.args.module, self.args.probe)}
        if mode in ("incompatible", "incompatible-version"):
            foreign = helper.parent / "llaminar_mpi_components" / "mca_schizo_llaminar.so"
            assets[str(foreign.resolve())] = _asset(foreign)
        result["assets"] = assets
        _atomic(directory / "command.json", result)
        tree = ObservedTree()
        process = None
        failure = None
        try:
            observations = None
            original_roots: list[dict] = []
            sent = False
            started = time.monotonic()
            environment = _environment(launcher)
            if mode == "unselected-inclusion":
                environment["OMPI_MCA_schizo"] = "ompi,orte"
            elif mode == "unselected-exclusion":
                environment["OMPI_MCA_schizo"] = "^llaminar"
            with (directory / "output.log").open("w") as output:
                process = subprocess.Popen(command, env=environment, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    tree.sample(process.pid)
                    snapshots = [_read(reports / f"rank-{rank}.json") for rank in range(2)]
                    if all(snapshots) and observations is None:
                        observations = snapshots
                        result["observations"] = observations
                        owner, env, maps = _rank_daemon(observations, self.args.probe, tree)
                        result["daemon_observation"] = owner
                        for path in (owner["temporary"], owner["shared"]):
                            original_roots.append({"path": path, "identity": _identity(path)})
                        keys = observations[0].get("owned_bindings")
                        if not keys:
                            raise AssertionError("Probe omitted shared source-declared owner bindings")
                        original_bindings = {key: env[key.encode()].decode() if key.encode() in env else None
                                             for key in keys}
                        result["original_daemon_bindings"] = original_bindings
                        result["original_daemon_roots"] = original_roots
                        component = helper.parent / "llaminar_mpi_components" / "mca_schizo_llaminar.so"
                        if str(component) not in maps:
                            raise AssertionError("Required native owner component was not mapped in actual daemon")
                        result["module_sha256"] = _hash(component)
                        # Check while live, before a passing collective or
                        # shared filesystem can conceal foreign physical roots.
                        for rank in observations:
                            if rank.get("owned_bindings") != original_bindings:
                                raise AssertionError("Actual rank bindings differ from original daemon declaration")
                            for key in ("temporary", "shared"):
                                if rank[key] != owner[key] or _identity(rank[key]) != _identity(owner[key]):
                                    raise AssertionError(f"Rank {rank['rank']} {key} differs from actual daemon owner")
                    if mode == "success" and observations:
                        (reports / "release").touch()
                    if mode == "wait" and observations and not sent:
                        process.terminate()
                        sent = True
                    if time.monotonic() - started > 30:
                        raise AssertionError("Actual native daemon launch did not complete within 30 seconds")
                    time.sleep(0.01)
                result["returncode"] = process.wait(timeout=40)
            if negative:
                output = (directory / "output.log").read_text()
                expected_diagnostic = {
                    "missing": "Owned MPI native environment component is missing",
                    "incompatible": "Owned MPI native environment component ABI is incompatible",
                    "incompatible-version": "Owned MPI native environment component ABI is incompatible",
                    "unselected-inclusion": "Owned MPI native environment component is not selected",
                    "unselected-exclusion": "Owned MPI native environment component is not selected",
                }[mode]
                if result["returncode"] != 125 or expected_diagnostic not in output:
                    raise AssertionError("Required native module failed for an unrelated reason")
                if list(reports.iterdir()) or (directory / "transport.jsonl").exists():
                    raise AssertionError("Invalid native module entered daemon/rank execution")
                if list(launcher.iterdir()) or list(daemon.iterdir()) or tree.remaining():
                    raise AssertionError("Rejected native module did not retire exact owned state")
                result.update(passed=True, required_module_rejected=True,
                              exact_diagnostic=expected_diagnostic)
                return
            observations = observations or [_read(reports / f"rank-{rank}.json") for rank in range(2)]
            if not all(observations):
                raise AssertionError("Missing actual native rank observations")
            result["observations"] = observations
            expected = 0
            if mode == "wait":
                native = _read(native_receipt)
                if native is None or native["signals"] != [15] or native["returncode"] == 0:
                    raise AssertionError("Missing actual native MPI interruption status")
                result["native_status"] = native
                expected = native["returncode"] if native["returncode"] >= 0 else 128 - native["returncode"]
            if result["returncode"] != expected:
                raise AssertionError("Owner did not preserve actual native launcher terminal status")
            for rank in observations:
                if rank["size"] != 2 or rank["sum"] != 3:
                    raise AssertionError("Actual native rank collective failed")
                if Path(rank["temporary"]).parent != daemon:
                    raise AssertionError("Rank temporary root belongs to launcher instead of daemon")
                for mapping in rank["mapped_backing"]:
                    if not any(mapping.startswith(rank[key] + "/") for key in ("temporary", "shared")):
                        raise AssertionError("Actual native mapping escaped original daemon roots")
            if not original_roots:
                raise AssertionError("No live original daemon-root identities captured")
            if list(launcher.iterdir()) or list(daemon.iterdir()) or tree.remaining():
                raise AssertionError("Native process/backing retirement is incomplete")
            if any(Path(root["path"]).exists() for root in original_roots):
                raise AssertionError("Original daemon backing survived terminal retirement")
            transport = [json.loads(line) for line in (directory / "transport.jsonl").read_text().splitlines()]
            if not transport or not any("--mpi-daemon" in " ".join(item["argv"]) for item in transport):
                raise AssertionError("Native OpenMPI did not invoke actual daemon launch-agent")
            result.update(passed=True, original_daemon_roots=original_roots)
        except BaseException as error:
            failure = error
            result.update(passed=False, error=str(error))
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            changed_assets = []
            for path, original in assets.items():
                try:
                    if _asset(Path(path)) != original:
                        changed_assets.append(path)
                except (OSError, AssertionError):
                    changed_assets.append(path)
            if changed_assets:
                result.update(passed=False, changed_assets=changed_assets)
            result["process_events"] = tree.events
            result["remaining_original_processes"] = tree.remaining()
            if process is not None:
                result["final_returncode"] = process.returncode
            _atomic(directory / "result.json", result)
            self.results.append(result)
            tree.close()
            if not result.get("passed", False):
                _failure_receipt(result, directory)
            if changed_assets:
                raise AssertionError("Native assets changed during the exact observed transaction")
        if failure is not None:
            raise failure


def main() -> None:
    """Run selected cases; retire passing scratch and retain explicit/failing evidence."""
    if len(sys.argv) >= 4 and sys.argv[1] == "--local-rsh":
        raise SystemExit(_rsh(Path(sys.argv[2]), sys.argv[3:]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--mpirun", required=True, type=Path)
    parser.add_argument("--case", choices=("success", "wait", "missing", "incompatible", "incompatible-version", "unselected-inclusion", "unselected-exclusion", "all"), default="all")
    parser.add_argument("--incompatible-module", type=Path)
    parser.add_argument("--incompatible-version-module", type=Path)
    parser.add_argument("--status-helper", type=Path,
                        default=Path(__file__).with_name("test_mpi_session_lifecycle.py"))
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    selected_module = args.helper.resolve().parent / "llaminar_mpi_components" / "mca_schizo_llaminar.so"
    if not selected_module.is_file() or not args.module.is_file() or not selected_module.samefile(args.module):
        parser.error("Selected module asset must be the exact helper-sibling native component")
    if args.case in ("incompatible", "all") and args.incompatible_module is None:
        parser.error("Incompatible-module case requires its built CPU-only fixture")
    if args.case in ("incompatible-version", "all") and args.incompatible_version_module is None:
        parser.error("Incompatible-version case requires its built CPU-only fixture")
    root = args.evidence or Path(tempfile.mkdtemp(prefix="llaminar-mpi-daemon-"))
    root.mkdir(parents=True, exist_ok=True)
    replay = DaemonReplay(args, root)
    passed = False
    try:
        for mode in ("success", "wait", "missing", "incompatible", "incompatible-version", "unselected-inclusion", "unselected-exclusion"):
            if args.case in (mode, "all"):
                replay.run(mode)
        print(json.dumps({"passed": True, "scope": "same-host only", "cases": [
            {"case": result["case"], "passed": result["passed"],
             "returncode": result["final_returncode"],
             "required_module_rejected": result.get("required_module_rejected", False)}
            for result in replay.results]}))
        passed = True
    finally:
        # Every successful native case proves its original processes and roots
        # retired before returning. Failed or incomplete transactions cannot
        # authorize recursive deletion of potentially live runtime backing.
        if passed and args.evidence is None:
            shutil.rmtree(root)
        else:
            print(f"Native daemon ownership evidence retained: {root}", file=sys.stderr)


if __name__ == "__main__":
    main()
