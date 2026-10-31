## @file test_mpi_session_lifecycle.py
#  @brief Replay the production MPI process/file owner with real CPU MPI jobs.
"""Prove process retirement before file cleanup without touching another job."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import pty
import shutil
import signal
import subprocess
import sys
import tempfile
import termios
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts/ci"))
from mpi_session_retirement import require_retired


def _wait(predicate, description: str, seconds: float = 15.0):
    """Bound a test observation; inference has no artificial response deadline."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for {description}")


def _read(path: Path) -> dict | None:
    """Read only atomically published child observations."""
    return json.loads(path.read_text()) if path.is_file() else None


def _publish(path: Path, value: dict) -> None:
    """Publish a complete observation outside the runtime-owned directory."""
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value) + "\n")
    pending.replace(path)


def _mpi_receipt(report: Path, command: list[str]) -> int:
    """Record the real native launcher's status rather than guessing its signal convention."""
    child = None
    pending = []

    def forward(number, _frame):
        """Keep the actual MPI launcher responsible for its own rank shutdown."""
        pending.append(number)
        if child is not None and child.poll() is None:
            child.send_signal(number)

    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
        signal.signal(number, forward)
    child = subprocess.Popen(command, close_fds=True)
    for number in pending:
        if child.poll() is None:
            child.send_signal(number)
    status = child.wait()
    _publish(report, {"pid": child.pid, "returncode": status, "signals": pending})
    return status if status >= 0 else 128 - status


def _child(mode: str, report: Path, outside: Path, singleton: Path | None = None) -> int:
    """Exercise exact owner boundaries with a disposable child, never a real user job."""
    temporary = Path(os.environ["TMPDIR"])
    shared = Path(os.environ["OMPI_MCA_btl_vader_backing_directory"])
    (temporary / "leftover").write_text("owned temporary data")
    (shared / "leftover").write_text("owned shared memory data")
    (temporary / f"external-link-{os.getpid()}").symlink_to(outside, target_is_directory=True)
    observation = {"pid": os.getpid(), "temporary": str(temporary), "shared": str(shared)}
    if mode in {"orphan", "stubborn-parent"}:
        child_mode = "stubborn" if mode == "stubborn-parent" else "wait"
        child_report = report.with_name(report.stem + "-descendant.json")
        child = subprocess.Popen([sys.executable, __file__, "--child", child_mode,
                                  str(child_report), str(outside)], start_new_session=True)
        observation["descendant"] = child.pid
        _wait(lambda: _read(child_report), "actual descendant startup")
    if mode in {"stubborn", "stubborn-parent"}:
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
            signal.signal(number, signal.SIG_IGN)
    if mode in {"graceful", "delayed-stop"}:
        def graceful_stop(_number, _frame):
            """Expose a clean application exit after its real shutdown work."""
            if mode == "delayed-stop":
                time.sleep(0.25)
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, graceful_stop)
    if mode == "replace-root":
        moved = temporary.with_name(temporary.name + "-moved")
        temporary.rename(moved)
        temporary.mkdir(mode=0o700)
        (temporary / "foreign").write_text("replacement must survive")
        observation["moved"] = str(moved)
    _publish(report, observation)
    if mode.startswith("frontend-"):
        if singleton is None:
            raise AssertionError("Frontend ownership replay requires the actual singleton probe")
        frontend_report = report.with_name(report.stem + "-frontend.json")
        command = [str(singleton), "--observation", str(frontend_report), "success"]
        if mode == "frontend-fresh":
            return subprocess.run(command, close_fds=True, check=False).returncode
        if mode == "frontend-missing-location":
            os.environ.pop("TMPDIR")
        elif mode == "frontend-foreign-location":
            os.environ["TMPDIR"] = str(outside)
        elif mode.startswith("frontend-close-"):
            os.close(int(mode.removeprefix("frontend-close-")))
        elif mode == "frontend-replace-pidfd":
            with open("/dev/null", "rb") as replacement:
                os.dup2(replacement.fileno(), 5)
        elif mode in {"frontend-replace-temporary", "frontend-replace-shared"}:
            is_temporary = mode == "frontend-replace-temporary"
            original = temporary if is_temporary else shared
            replacement = outside / mode
            replacement.mkdir(mode=0o700)
            shutil.copyfile(original / ".owner", replacement / ".owner")
            descriptor = os.open(replacement, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.dup2(descriptor, 6 if is_temporary else 7)
            finally:
                os.close(descriptor)
            for name, value in list(os.environ.items()):
                if value == str(original):
                    os.environ[name] = str(replacement)
        elif mode == "frontend-external-broken":
            os.environ["OMPI_COMM_WORLD_SIZE"] = "1"
            os.close(6)
        elif mode != "frontend-reuse":
            raise AssertionError(f"Unknown exact ownership replay {mode}")
        os.execv(command[0], command)
    if mode in {"wait", "stubborn", "stubborn-parent", "graceful", "delayed-stop"}:
        while True:
            signal.pause()
    return 7 if mode == "failure" else 0


class Replay:
    """One disposable test tree containing receipts, never another owner's state."""

    def __init__(self, args: argparse.Namespace, root: Path) -> None:
        self.args, self.root = args, root
        self.outside = root / "unrelated"
        self.outside.mkdir()
        (self.outside / "sentinel").write_text("untouched")
        self.results: list[dict] = []
        self.children: list[subprocess.Popen] = []

    def start(self, name: str, command: list[str], *, owned: bool = True,
              environment: dict[str, str] | None = None,
              executable: Path | None = None) -> tuple[subprocess.Popen, Path]:
        """Retain exact child ownership and external logs for every tested invocation."""
        directory = self.root / name
        directory.mkdir()
        base = directory / "sessions"
        base.mkdir()
        argv = ([str(self.args.helper), "--session-base", str(base), "--"] + command
                if owned else command)
        child_environment = dict(os.environ, TMPDIR=str(base))
        child_environment.update(environment or {})
        if owned and executable is not None:
            raise AssertionError("Alternate executable identity belongs to a direct frontend replay")
        with (directory / "output.log").open("w") as output:
            process = subprocess.Popen(argv, env=child_environment, executable=executable,
                                       stdout=output, stderr=subprocess.STDOUT)
        self.children.append(process)
        return process, directory

    def finish(self, process: subprocess.Popen, directory: Path, expected: int,
               observations: list[dict], *, seconds: float = 15.0) -> None:
        """Require original status, actual exit witnesses and both roots gone."""
        actual = process.wait(timeout=seconds)
        if actual != expected:
            raise AssertionError(f"{directory.name}: exit {actual}, expected {expected}:\n"
                                 f"{(directory / 'output.log').read_text()}")
        require_retired([directory / "sessions"])
        roots = {Path(item[key]) for item in observations for key in ("temporary", "shared")}
        _wait(lambda: all(not path.exists() for path in roots), f"{directory.name} file retirement")
        if list((directory / "sessions").iterdir()):
            raise AssertionError(f"{directory.name}: session files survived launcher exit")
        for item in observations:
            # pidfds authenticate the exact process when still present. Already
            # absent PIDs are terminal too; this check never sends a signal.
            for pid in (item["pid"], item.get("descendant"), *item.get("native_children", [])):
                if pid is None:
                    continue
                try:
                    descriptor = os.pidfd_open(pid)
                except ProcessLookupError:
                    continue
                try:
                    poll = select.poll()
                    poll.register(descriptor, select.POLLIN)
                    if not poll.poll(0):
                        raise AssertionError(f"{directory.name}: process {pid} survived cleanup")
                finally:
                    os.close(descriptor)
        if (self.outside / "sentinel").read_text() != "untouched":
            raise AssertionError("Cleanup changed an unrelated directory")
        self.results.append({"case": directory.name, "returncode": actual, "retired": True})

    def child(self, name: str, mode: str) -> tuple[subprocess.Popen, Path, dict]:
        """Start one child that reports both actual runtime paths before its outcome."""
        report = self.root / f"{name}.json"
        process, directory = self.start(name, [sys.executable, __file__, "--child", mode,
                                               str(report), str(self.outside)] +
                                               ([str(self.args.singleton)] if mode.startswith("frontend-") else []))
        observation = _wait(lambda: _read(report), f"{name} ready")
        if Path(observation["temporary"]).parent != directory / "sessions":
            raise AssertionError("Owner did not use the requested base")
        if Path(observation["shared"]).parent != Path("/dev/shm"):
            raise AssertionError("MPI shared backing left its real shared-memory filesystem")
        return process, directory, observation

    def launcher(self) -> None:
        """Normal/error/orphan exits and concurrent interrupted owners share one implementation."""
        for name, mode, status in (("normal", "success", 0), ("failure", "failure", 7),
                                   ("successful-orphan", "orphan", 0)):
            process, directory, observation = self.child(name, mode)
            self.finish(process, directory, status, [observation])
        missing, directory = self.start("missing-executable", [str(self.root / "absent")])
        self.finish(missing, directory, 125, [])
        # Keep one owner live while repeatedly retiring neighbors. This proves
        # isolation, including a symlink into an unrelated directory per child.
        neighbor, neighbor_dir, neighbor_state = self.child("live-neighbor", "wait")
        for index, number in enumerate((signal.SIGTERM, signal.SIGINT, signal.SIGKILL) * 3):
            process, directory, observation = self.child(f"interrupt-{index}", "wait")
            process.send_signal(number)
            expected = -signal.SIGKILL if number == signal.SIGKILL else 128 + number
            self.finish(process, directory, expected, [observation])
            if neighbor.poll() is not None or not all(Path(neighbor_state[key]).is_dir()
                                                      for key in ("temporary", "shared")):
                raise AssertionError("Retiring a neighbor removed a live owner's state")
        neighbor.terminate()
        self.finish(neighbor, neighbor_dir, 143, [neighbor_state])
        process, directory, observation = self.child("graceful-stop", "graceful")
        process.terminate()
        self.finish(process, directory, 0, [observation])
        self.terminal()
        self.retirement()

    def retirement(self) -> None:
        """Keep the first failed-root diagnostic after an earlier root retired."""
        # Renaming/replacing an owned root invalidates the deletion authority.
        process, directory, observation = self.child("changed-root", "replace-root")
        status = process.wait(timeout=15)
        if status != 125 or not Path(observation["moved"]).is_dir() or not (
                Path(observation["temporary"]) / "foreign").is_file():
            raise AssertionError("Changed root identity did not fail closed")
        log = (directory / "output.log").read_text()
        primary = "MPI session root identity changed; backing retained: " + observation["temporary"]
        hidden = "MPI session root identity changed; backing retained: " + observation["shared"]
        if primary not in log or hidden in log:
            raise AssertionError("Cleanup hid the original failed root behind an already retired root: " + log)
        # Both are test-created roots; their child has exited and no runtime
        # remains. The production owner intentionally retained the ambiguous one.
        for path in (Path(observation["moved"]), Path(observation["temporary"]),
                     Path(observation["shared"])):
            if path.exists():
                shutil.rmtree(path)
        self.results.append({"case": "changed-root", "returncode": status, "rejected": True})

    def terminal(self) -> None:
        """Keep real interactive stdin/stdout usable through the detached owner."""
        directory = self.root / "terminal"
        directory.mkdir()
        base = directory / "sessions"
        base.mkdir()
        master, slave = pty.openpty()
        try:
            attributes = termios.tcgetattr(slave)
            attributes[3] &= ~termios.ECHO
            termios.tcsetattr(slave, termios.TCSANOW, attributes)
            process = subprocess.Popen([str(self.args.helper), "--session-base", str(base), "--", "/bin/cat"],
                                       stdin=slave, stdout=slave, stderr=slave)
            self.children.append(process)
            os.close(slave)
            slave = -1
            message = b"original interactive MPI frontend input\n"
            os.write(master, message)
            readable, _, _ = select.select([master], [], [], 10)
            if not readable or os.read(master, 4096).replace(b"\r\n", b"\n") != message:
                raise AssertionError("Owner changed interactive stdin/stdout behavior")
            os.write(master, b"\x04")
            if process.wait(timeout=10) != 0 or list(base.iterdir()):
                raise AssertionError("Interactive frontend did not retire cleanly")
            self.results.append({"case": "terminal", "returncode": 0, "retired": True})
        finally:
            os.close(master)
            if slave >= 0:
                os.close(slave)

    def mpi(self) -> None:
        """Exercise real two-rank communication and actual PMIx/shared backing."""
        for mode in ("success", "failure", "wait"):
            reports = self.root / f"mpi-{mode}-reports"
            reports.mkdir()
            command = [str(self.args.mpirun), "-np", "2",
                "--bind-to", "none", "--mca", "btl_vader_single_copy_mechanism", "none", "--",
                str(self.args.probe), str(reports), mode]
            receipt = self.root / f"mpi-{mode}-status.json"
            if mode == "wait":
                command = [sys.executable, __file__, "--mpi-receipt", str(receipt), "--", *command]
            process, directory = self.start(f"mpi-{mode}", command)
            observations = [_wait(lambda rank=rank: _read(reports / f"rank-{rank}.json"),
                                  f"MPI rank {rank} startup") for rank in range(2)]
            for item in observations:
                if item["sum"] != 3 or item["size"] != 2:
                    raise AssertionError("Real MPI collective failed")
                for mapping in item["mapped_backing"]:
                    if not any(mapping.startswith(item[key] + "/") for key in ("temporary", "shared")):
                        raise AssertionError(f"MPI backing escaped its owner: {mapping}")
            if mode == "wait":
                process.terminate()
                native = _wait(lambda: _read(receipt), "actual interrupted MPI exit status")
                if native["signals"] != [signal.SIGTERM] or native["returncode"] == 0:
                    raise AssertionError(f"Native MPI signal observation is incomplete: {native}")
                expected = native["returncode"] if native["returncode"] >= 0 else 128 - native["returncode"]
                observations[0]["native_children"] = [native["pid"]]
            else:
                expected = {"success": 0, "failure": 7}[mode]
            self.finish(process, directory, expected, observations)

    def native_options(self) -> None:
        """Exercise canonical launcher names and inherited/opaque native file options."""
        launchers = sorted({self.args.mpirun, self.args.mpirun.resolve()}, key=str)
        for index, launcher in enumerate(launchers):
            reports = self.root / f"native-alias-{index}-reports"
            reports.mkdir()
            foreign = str(self.outside)
            environment = {name: foreign for name in (
                "OMPI_MCA_orte_tmpdir_base", "OMPI_MCA_orte_local_tmpdir_base",
                "OMPI_MCA_orte_remote_tmpdir_base", "PRTE_MCA_prte_local_tmpdir_base",
                "PRTE_MCA_prte_remote_tmpdir_base", "OMPI_MCA_btl_vader_backing_directory",
                "OMPI_MCA_btl_sm_backing_directory", "OMPI_MCA_shmem_mmap_backing_file_base_dir")}
            environment["OMPI_MCA_shmem_mmap_relocate_backing_file"] = "1"
            process, directory = self.start(f"native-alias-{index}", [str(launcher), "-np", "2",
                "--bind-to", "none", "--", str(self.args.probe), str(reports), "success"],
                environment=environment)
            observations = [_wait(lambda rank=rank: _read(reports / f"rank-{rank}.json"),
                                  "canonical native launcher startup") for rank in range(2)]
            for item in observations:
                if item["sum"] != 3 or item["size"] != 2:
                    raise AssertionError("Canonical native alias did not execute real MPI")
                for mapping in item["mapped_backing"]:
                    if not any(mapping.startswith(item[key] + "/") for key in ("temporary", "shared")):
                        raise AssertionError(f"Inherited native location escaped owner: {mapping}")
            self.finish(process, directory, 0, observations)
            if sorted(path.name for path in self.outside.iterdir()) != ["sentinel"]:
                raise AssertionError("Native inherited location wrote outside the owned roots")
        invalid = (
            ("location", ["--mca", "orte_tmpdir_base", str(self.outside)], "location MCA overrides"),
            ("opaque-delimiter", ["--mca", "unused_non_location", "--", "--mca",
                                  "orte_tmpdir_base", str(self.outside)], "location MCA overrides"),
            ("location-export", ["-x", "TMPDIR=" + str(self.outside)], "cannot export"),
            ("env-list", ["--mca", "mca_base_env_list", "SOME_VALUE=1"], "mca_base_env_list"),
        )
        for name, options, diagnostic in invalid:
            process, directory = self.start("native-reject-" + name,
                [str(self.args.mpirun.resolve()), *options, "--", str(self.root / "must-not-execute")])
            self.finish(process, directory, 125, [])
            if diagnostic not in (directory / "output.log").read_text():
                raise AssertionError(f"Native {name} failed for an unrelated reason")
        report = self.root / "native-application-arguments.json"
        arguments = ["--mca", "orte_tmpdir_base", str(self.outside), "--tmpdir", "opaque-value"]
        code = ("import json,os,pathlib,sys; "
                "pathlib.Path(sys.argv[1]).write_text(json.dumps({"
                "'pid':os.getpid(),'temporary':os.environ['TMPDIR'],"
                "'shared':os.environ['OMPI_MCA_btl_vader_backing_directory'],"
                "'mmap':os.environ['OMPI_MCA_shmem_mmap_backing_file_base_dir'],"
                "'arguments':sys.argv[2:]}))")
        process, directory = self.start("native-application-arguments",
            [str(self.args.mpirun.resolve()), "-np", "1", "--bind-to", "none", "--",
             sys.executable, "-c", code, str(report), *arguments])
        observation = _wait(lambda: _read(report), "opaque native application arguments")
        if observation["arguments"] != arguments or observation["mmap"] != observation["shared"]:
            raise AssertionError("Native option ownership changed app arguments or lost mmap backing")
        self.finish(process, directory, 0, [observation])

    def stubborn(self) -> None:
        """Exercise the standard 30-second grace and newly adopted SIGKILL descendants."""
        process, directory, observation = self.child("noncooperative-tree", "stubborn-parent")
        process.terminate()
        started = time.monotonic()
        self.finish(process, directory, 137, [observation], seconds=40)
        elapsed = time.monotonic() - started
        if not 29 <= elapsed < 40:
            raise AssertionError(f"Unexpected native retirement grace: {elapsed:.3f}s")

    def ctest_timeout(self) -> None:
        """A real CTest timeout must wait for owner cleanup, even after a clean child exit."""
        registration = self.root / "ctest-registration"
        registration.mkdir()
        report = self.root / "ctest-timeout.json"
        base = self.root / "ctest-timeout" / "sessions"
        command = [str(self.args.helper), "--session-base", str(base), "--", sys.executable,
                   __file__, "--child", "delayed-stop", str(report), str(self.outside)]
        words = " ".join("[=[" + word + "]=]" for word in command)
        (registration / "CTestTestfile.cmake").write_text(
            f"add_test(owned_timeout {words})\n"
            "set_tests_properties(owned_timeout PROPERTIES TIMEOUT 1 "
            "TIMEOUT_SIGNAL_NAME SIGTERM TIMEOUT_SIGNAL_GRACE_PERIOD 36)\n")
        process, directory = self.start("ctest-timeout", ["ctest", "--test-dir", str(registration),
                                                        "--output-on-failure"], owned=False)
        observation = _wait(lambda: _read(report), "CTest timeout child startup")
        self.finish(process, directory, 8, [observation])
        if "Timeout" not in (directory / "output.log").read_text():
            raise AssertionError("CTest did not retain the original timeout failure")

    def singleton(self) -> None:
        """Run the actual application session with its real singleton MPI daemon."""
        for mode, number, status in (("success", None, 0), ("throw", None, 7),
                                     ("inspect", None, 0), ("wait", signal.SIGTERM, 143),
                                     ("wait", signal.SIGKILL, -9)):
            name = f"singleton-{mode}-{number or 0}"
            report = self.root / (name + ".json")
            process, directory = self.start(name, [str(self.args.singleton), "--observation",
                                                   str(report), mode], owned=False)
            observation = _wait(lambda: _read(report), f"{name} real MPI startup")
            if mode == "inspect":
                children = Path(f"/proc/{observation['pid']}/task/{observation['pid']}/children")
                descendant_pids = [int(value) for value in children.read_text().split()]
                # This selected native OpenMPI singleton creates an orted daemon.
                # Record actual children, not an invented process-group receipt.
                observation["native_children"] = descendant_pids
                report.with_suffix(report.suffix + ".release").touch()
            if number is not None:
                process.send_signal(number)
            self.finish(process, directory, status, [observation])
        for index, argv0 in enumerate(("authored argv zero", "mpirun")):
            name = f"singleton-argv0-{index}"
            report = self.root / (name + ".json")
            process, directory = self.start(name, [argv0, "--observation", str(report), "success"],
                                             owned=False, executable=self.args.singleton)
            observation = _wait(lambda: _read(report), "original argv0 native singleton startup")
            if observation["argv0"] != argv0:
                raise AssertionError("Process ownership changed the original authored argv0")
            self.finish(process, directory, 0, [observation])
        report = self.root / "external-mpi.json"
        process, directory = self.start("external-mpi", [str(self.args.mpirun), "-np", "1",
            "--bind-to", "none", "--", str(self.args.singleton), "--observation", str(report), "external"])
        observation = _wait(lambda: _read(report), "external rank startup")
        self.finish(process, directory, 0, [observation])

    def identity(self) -> None:
        """Distinguish fresh frontend entry from a broken original descriptor loan."""
        for mode in ("frontend-reuse", "frontend-fresh", "frontend-missing-location",
                     "frontend-foreign-location", "frontend-close-5", "frontend-close-6",
                     "frontend-close-7", "frontend-close-8", "frontend-replace-pidfd",
                     "frontend-replace-temporary", "frontend-replace-shared",
                     "frontend-external-broken"):
            process, directory, observation = self.child(mode, mode)
            valid = mode in {"frontend-reuse", "frontend-fresh"}
            self.finish(process, directory, 0 if valid else 125, [observation])
            report = self.root / (mode + "-frontend.json")
            if not valid:
                if report.exists():
                    raise AssertionError(f"{mode} entered MPI despite broken original ownership")
                if "Broken MPI session borrow" not in (directory / "output.log").read_text():
                    raise AssertionError(f"{mode} failed for an unrelated reason")
            else:
                frontend = _read(report)
                if frontend is None:
                    raise AssertionError("Actual frontend did not publish MPI startup")
                if mode == "frontend-reuse" and frontend["temporary"] != observation["temporary"]:
                    raise AssertionError("A valid original borrower created a second owner")
                if mode == "frontend-fresh" and Path(frontend["temporary"]).parent != Path(observation["temporary"]):
                    raise AssertionError("Fresh frontend did not own its distinct startup scope")

    def cleanup(self) -> None:
        """Retire only direct test-owned children if a regression assertion fails."""
        for process in self.children:
            if process.poll() is None:
                process.terminate()
        for process in self.children:
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main() -> None:
    """Run selected real ownership cases; preserve evidence if any contract fails."""
    if len(sys.argv) in (5, 6) and sys.argv[1] == "--child":
        raise SystemExit(_child(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]),
                               Path(sys.argv[5]) if len(sys.argv) == 6 else None))
    if len(sys.argv) > 5 and sys.argv[1] == "--mpi-receipt" and sys.argv[3] == "--":
        raise SystemExit(_mpi_receipt(Path(sys.argv[2]), sys.argv[4:]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--singleton", type=Path)
    parser.add_argument("--mpirun", type=Path, required=True)
    parser.add_argument("--case", choices=("launcher", "mpi", "native_options", "stubborn", "ctest_timeout", "singleton", "identity", "retirement", "all"), default="all")
    args = parser.parse_args()
    if args.case in {"singleton", "identity", "all"} and args.singleton is None:
        parser.error("Actual frontend ownership cases require --singleton")
    root = Path(tempfile.mkdtemp(prefix="llaminar-mpi-replay-"))
    replay = Replay(args, root)
    passed = False
    failures = []
    try:
        selected = (("retirement",) if args.case == "retirement" else
                    ("launcher", "mpi", "native_options", "stubborn", "ctest_timeout", "singleton", "identity"))
        for case in selected:
            if args.case in {case, "all"}:
                try:
                    getattr(replay, case)()
                except Exception as error:
                    failures.append({"case": case, "error": str(error)})
                    replay.cleanup()
        passed = not failures
        print(json.dumps({"passed": passed, "cases": replay.results, "failures": failures}))
        if not passed:
            raise SystemExit(1)
    finally:
        replay.cleanup()
        if passed:
            shutil.rmtree(root)
        else:
            print(f"MPI lifetime evidence retained at {root}", file=sys.stderr)


if __name__ == "__main__":
    main()
