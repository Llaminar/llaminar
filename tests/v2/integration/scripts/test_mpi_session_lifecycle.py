## @file test_mpi_session_lifecycle.py
#  @brief Replay the production MPI process/file owner with real CPU MPI jobs.
"""Prove process retirement before file cleanup without touching another job."""
from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import ExitStack
import hashlib
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

# Test witness for the two MPISessionOwner startup directory loans.
_DIRECTORY_LOAN_FDS = {"temporary": 6, "shared": 7}


def _sha256(path: Path) -> str:
    """Hash one exact native asset, not its metadata or its location."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if mode.startswith("root-"):
        # Only this child's exact guardian-issued root is a mutation target.
        operation, side = mode.removeprefix("root-").rsplit("-", 1)
        original, loan = (shared, 7) if side == "shared" else (temporary, 6)
        held, named = os.fstat(loan), original.stat()
        if (held.st_dev, held.st_ino, held.st_uid) != (named.st_dev, named.st_ino, named.st_uid):
            raise AssertionError("Fixture root differs from its original guardian loan")
        observation.update(root=str(original), operation=operation,
                           original_inode=held.st_ino, original_device=held.st_dev)
        if operation == "unlink":
            shutil.rmtree(original)
            observation["held_nlink"] = os.fstat(loan).st_nlink
            if observation["held_nlink"] != 0:
                raise AssertionError("Actual original directory did not become unlinked")
            observation["release"] = str(report.with_suffix(".release"))
        else:
            if operation.startswith("unlinked-"):
                shutil.rmtree(original)
                replacement = operation.removeprefix("unlinked-")
            else:
                moved = original.with_name(original.name + "-moved")
                original.rename(moved)
                observation["moved"] = str(moved)
                replacement = operation
            observation["held_nlink"] = os.fstat(loan).st_nlink
            if replacement == "replace-dir":
                original.mkdir(mode=0o700)
                (original / "foreign").write_text("replacement must survive")
            elif replacement == "replace-file":
                original.write_text("replacement must survive")
            elif replacement == "replace-symlink":
                original.symlink_to(outside, target_is_directory=True)
            elif replacement != "rename":
                raise AssertionError("Unknown root mutation")
    _publish(report, observation)
    if mode.startswith("root-unlink-"):
        # Keep a real consumer alive until the parent proves no cleanup occurred.
        _wait(lambda: Path(observation["release"]).exists(), "original consumer release")
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
                os.dup2(descriptor, _DIRECTORY_LOAN_FDS["temporary" if is_temporary else "shared"])
            finally:
                os.close(descriptor)
            for name, value in list(os.environ.items()):
                if value == str(original):
                    os.environ[name] = str(replacement)
        elif mode == "frontend-external-broken":
            os.environ["OMPI_COMM_WORLD_SIZE"] = "1"
            os.close(_DIRECTORY_LOAN_FDS["temporary"])
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
               observations: list[dict], *, seconds: float = 15.0,
               check_retirement: Callable[[], None] | None = None) -> None:
        """Prove the original status and all retirement checks before recording success.

        A specialized case can pass a check over retained pre-release witnesses
        instead of asking this method to inspect its observations' numeric PIDs.
        """
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
        if check_retirement is not None:
            check_retirement()
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

    def root_unlink(self) -> None:
        """Accept only actual zero-link originals after the last real child exits."""
        self.root_mutations(("unlink",))

    def root_identity(self) -> None:
        """Retain still-linked originals and every foreign replacement unchanged."""
        self.root_mutations(("rename", "replace-dir", "replace-file", "replace-symlink",
                             "unlinked-replace-dir", "unlinked-replace-file", "unlinked-replace-symlink"))

    def root_retirement(self) -> None:
        """Run both physical-unlink and adversarial original-root contracts."""
        self.root_unlink()
        self.root_identity()

    def root_mutations(self, operations: tuple[str, ...]) -> None:
        """Use only exact test-owned helpers/roots; a neighboring owner stays live."""
        group = "-".join(operations)
        neighbor, neighbor_dir, neighbor_state = self.child("neighbor-" + group, "wait")
        sides = ("shared", "temporary") if self.args.root_side == "both" else (self.args.root_side,)
        try:
            for repeat in range(self.args.repeat):
                for operation in operations:
                    for side in sides:
                        name = f"{operation}-{side}-{repeat}"
                        process, directory, observation = self.child(name, f"root-{operation}-{side}")
                        target = Path(observation["root"])
                        if operation == "unlink":
                            # The untouched opposite root proves the guardian has
                            # not begun cleanup while its actual child remains.
                            opposite = Path(observation["temporary" if side == "shared" else "shared"])
                            if process.poll() is not None or not (opposite / ".owner").is_file():
                                raise AssertionError("Guardian retired backing before its live consumer")
                            Path(observation["release"]).write_text("original consumer may exit")
                            self.finish(process, directory, 0, [observation])
                        else:
                            status = process.wait(timeout=15)
                            log = (directory / "output.log").read_text()
                            if status != 125 or "MPI session root identity changed; backing retained: " + str(target) not in log:
                                raise AssertionError(f"{name}: original identity rejection missing: {status}: {log}")
                            if "moved" in observation and not Path(observation["moved"]).is_dir():
                                raise AssertionError("Renamed still-linked original was removed")
                            if operation.endswith("replace-dir") and (target / "foreign").read_text() != "replacement must survive":
                                raise AssertionError("Foreign directory was changed")
                            if operation.endswith("replace-file") and target.read_text() != "replacement must survive":
                                raise AssertionError("Foreign file was changed")
                            if operation.endswith("replace-symlink") and (not target.is_symlink() or target.resolve() != self.outside):
                                raise AssertionError("Foreign symlink was changed")
                            # These exact roots/replacements were created by this
                            # disposable test. All original children have exited.
                            retained = [target, Path(observation["temporary"]), Path(observation["shared"])]
                            if "moved" in observation:
                                retained.append(Path(observation["moved"]))
                            for path in retained:
                                if path.is_symlink() or path.is_file():
                                    path.unlink()
                                elif path.is_dir():
                                    shutil.rmtree(path)
                            require_retired([directory / "sessions"], timeout_seconds=0)
                            self.results.append({"case": name, "returncode": status, "rejected": True,
                                                 "held_nlink": observation["held_nlink"]})
                        self.results[-1].update(original_device=observation["original_device"],
                                                original_inode=observation["original_inode"],
                                                held_nlink=observation["held_nlink"],
                                                original_root=observation["root"])
                        if neighbor.poll() is not None or not all((Path(neighbor_state[key]) / ".owner").is_file()
                                                                  for key in ("temporary", "shared")):
                            raise AssertionError("Root retirement changed a live neighboring owner")
                        if (self.outside / "sentinel").read_text() != "untouched":
                            raise AssertionError("Root retirement followed a foreign symlink")
        finally:
            if neighbor.poll() is None:
                neighbor.terminate()
            self.finish(neighbor, neighbor_dir, 143, [neighbor_state])

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

    def staged_singleton(self) -> None:
        """Require actual MPI initialization through the relocated frontend's exact sibling."""
        original_helper = self.args.helper.resolve(strict=True)
        component_path = Path("llaminar_mpi_components/mca_schizo_llaminar.so")
        component = self.args.module.resolve(strict=True)
        if not (original_helper.parent / component_path).samefile(component):
            raise AssertionError("Staged native module is not the exact build-helper sibling")
        # The negative case is sensitive to a PATH lookup: a usable copy of
        # the same helper is still on PATH, just not beside the staged binary.
        environment = {"LC_ALL": "C", "PATH": os.pathsep.join(
            (str(original_helper.parent), os.environ.get("PATH", os.defpath)))}
        helper_name = "llaminar_mpi_session"
        path_helper = shutil.which(helper_name, path=environment["PATH"])
        if path_helper is None or not original_helper.samefile(path_helper):
            raise AssertionError("Staged fixture lacks the original PATH helper control")
        fixture_sources = (self.args.singleton.resolve(strict=True), component)
        fixture_digests = tuple(_sha256(source) for source in fixture_sources)
        helper_digest = _sha256(original_helper)

        # Omit exactly one asset per negative. The complete build layout stays
        # available outside each stage but cannot stand in for a missing sibling.
        for name, missing in (("stage-ok", None), ("stage-no-helper", "helper"),
                              ("stage-no-module", "module")):
            stage = self.root / (name + "-bin")
            stage.mkdir(mode=0o700)
            frontend = stage / self.args.singleton.name
            helper = stage / helper_name
            native_module = stage / component_path
            native_module.parent.mkdir(mode=0o700)
            install = (("frontend", fixture_sources[0], frontend, fixture_digests[0]),
                       ("module", component, native_module, fixture_digests[1]),
                       ("helper", original_helper, helper, helper_digest))
            for role, source, target, expected in install:
                if role != missing:
                    shutil.copy2(source, target)
                    if _sha256(target) != expected:
                        raise AssertionError(f"Staged native asset changed during copy: {target}")
            assets = {"frontend_sha256": fixture_digests[0], "module_sha256": fixture_digests[1],
                      "module": str(native_module), "staged_module_present": missing != "module"}
            report = self.root / (name + ".json")
            process, directory = self.start(name, [str(frontend), "--observation", str(report), "inspect"],
                                            owned=False, environment=environment)
            if missing is not None:
                absent = helper if missing == "helper" else native_module
                diagnostic = ("MPI session ownership: cannot prepare exact child spawn: No such file or directory"
                              if missing == "helper" else "Owned MPI native environment component is missing")

                def require_rejection() -> None:
                    """Require the exact native refusal with no rank observation and no replacement asset."""
                    if ((directory / "output.log").read_text().splitlines() != [diagnostic]
                            or list(self.root.glob(report.name + "*")) or os.path.lexists(absent)):
                        raise AssertionError("A missing exact sibling did not reject before MPI rank entry")
                    for role, source, target, expected in install:
                        if _sha256(source) != expected or (role != missing and _sha256(target) != expected):
                            raise AssertionError(f"Staged rejection lost the exact {role} control")

                self.finish(process, directory, 125, [], check_retirement=require_rejection)
                self.results[-1].update(exact_diagnostic=diagnostic, path_helper=path_helper,
                                        alternate_module=str(component), missing_asset=missing,
                                        frontend=str(frontend), **assets)
                continue

            def initialized_rank() -> dict | None:
                if process.poll() is not None:
                    raise AssertionError(f"Staged frontend exited before live rank observation: {process.returncode}")
                return _read(report)

            observation = _wait(initialized_rank, "staged MPI rank initialization")
            if (observation["rank"], observation["world"], observation["mode"], observation["argv0"]) != (
                    0, 1, "inspect", str(frontend)):
                raise AssertionError("Staged singleton did not enter its actual rank with the original argv")

            def parent(pid: int) -> int:
                """Read the kernel parent from proc stat after the parenthesized comm field."""
                return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])

            def pin(stack: ExitStack, pid: int, image: Path) -> select.poll:
                """Retain the live process identity and verify the actual executable inode."""
                if type(pid) is not int or pid <= 1:
                    raise AssertionError("Invalid staged process identity")
                descriptor = os.pidfd_open(pid)
                stack.callback(os.close, descriptor)
                poll = select.poll()
                poll.register(descriptor, select.POLLIN)
                if poll.poll(0) or not image.samefile(f"/proc/{pid}/exe") or poll.poll(0):
                    raise AssertionError(f"Staged process is not live in its expected image: {image}")
                return poll

            # Acquire witnesses before releasing the rank. Do not reopen a
            # numeric PID after reaping or treat a reused pathname as backing.
            with ExitStack() as witnesses:
                rank_pid = observation["pid"]
                rank = pin(witnesses, rank_pid, frontend)
                owner_pid = parent(rank_pid)
                owner = pin(witnesses, owner_pid, helper)
                if rank_pid == process.pid or owner_pid == process.pid or parent(owner_pid) != process.pid:
                    raise AssertionError("Staged rank did not descend through its exact sibling guardian")
                roots = []
                for key, base in (("temporary", directory / "sessions"), ("shared", Path("/dev/shm"))):
                    original = Path(observation[key])
                    if original.parent != base or not original.name.startswith("llaminar-mpi-"):
                        raise AssertionError("Staged rank escaped its original runtime root")
                    fd = os.open(original, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
                    witnesses.callback(os.close, fd)
                    info = os.fstat(fd)
                    slot = _DIRECTORY_LOAN_FDS[key]
                    # Follow the parked rank's procfd link, never a second pathname lookup.
                    loan = os.open(f"/proc/{rank_pid}/fd/{slot}", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
                    witnesses.callback(os.close, loan)
                    loan_info = os.fstat(loan)
                    if (info.st_nlink == 0 or info.st_uid != os.geteuid() or
                            (info.st_dev, info.st_ino, info.st_uid) !=
                            (loan_info.st_dev, loan_info.st_ino, loan_info.st_uid)):
                        raise AssertionError(f"Staged {key} root did not match the rank's original directory loan")
                    roots.append((original, fd, loan, {"role": key, "path": str(original),
                        "device": info.st_dev, "inode": info.st_ino, "uid": info.st_uid, "rank_fd": slot,
                        "rank_device": loan_info.st_dev, "rank_inode": loan_info.st_ino, "rank_uid": loan_info.st_uid}))
                if (rank.poll(0) or owner.poll(0) or parent(rank_pid) != owner_pid
                        or parent(owner_pid) != process.pid or process.poll() is not None):
                    raise AssertionError("Staged ancestry changed before releasing the exact rank")
                def check_originals_retired() -> None:
                    """Require both held pidfds and both original directory inodes to retire."""
                    for poll in (rank, owner):
                        if not any(mask & select.POLLIN for _, mask in poll.poll(0)):
                            raise AssertionError("Staged rank or sibling guardian survived frontend exit")
                    for path, fd, loan, receipt in roots:
                        links = os.fstat(fd).st_nlink
                        loan_links = os.fstat(loan).st_nlink
                        name_present = os.path.lexists(path)
                        if links != 0 or loan_links != 0 or name_present:
                            raise AssertionError(f"Staged backing did not physically retire: {path}")
                        receipt.update(held_nlink=links, rank_loan_nlink=loan_links, name_present=name_present)

                report.with_suffix(report.suffix + ".release").touch(exist_ok=False)
                self.finish(process, directory, 0, [], check_retirement=check_originals_retired)
                self.results[-1].update(rank=observation, frontend_pid=process.pid,
                                        guardian_pid=owner_pid, guardian=str(helper), helper_sha256=helper_digest,
                                        original_roots=[receipt for _, _, _, receipt in roots], **assets)

    def legacy_placement(self) -> None:
        """Require the real frontend to reject old placement input before MPI entry."""
        marker = "LLAMINAR_MANAGED_PLACEMENT"
        receipt = "LLAMINAR_PLACEMENT_RECEIPT_V1"
        if marker in os.environ or receipt in os.environ:
            raise AssertionError("A clean parent is required for independent presence controls")
        diagnostic = "Unexpected managed placement environment in native MPI bootstrap"
        for name, environment in (("marker", {marker: "1"}),
                                  ("receipt", {receipt: "stale"}),
                                  ("empty-marker", {marker: ""}),
                                  ("empty-receipt", {receipt: ""}),
                                  ("pair", {marker: "1", receipt: "stale"})):
            report = self.root / ("legacy-" + name + ".json")
            process, directory = self.start("legacy-" + name,
                [str(self.args.singleton), "--observation", str(report), "success"],
                owned=False, environment=environment)

            def require_refusal() -> None:
                if ((directory / "output.log").read_text().splitlines() != [diagnostic]
                        or list(self.root.glob(report.name + "*"))):
                    raise AssertionError("Legacy placement did not reject before rank entry")

            self.finish(process, directory, 125, [], check_retirement=require_refusal)
            self.results[-1]["diagnostic"] = diagnostic

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
    parser.add_argument("--module", type=Path, help="Exact built native module for the staged singleton")
    parser.add_argument("--mpirun", type=Path, required=True)
    parser.add_argument("--case", choices=("launcher", "mpi", "native_options", "stubborn", "ctest_timeout",
                                          "singleton", "staged_singleton", "legacy_placement", "identity", "retirement",
                                          "root_retirement", "root_unlink", "root_identity", "all"), default="all")
    parser.add_argument("--root-side", choices=("both", "shared", "temporary"), default="both")
    parser.add_argument("--repeat", type=int, choices=range(1, 21), default=1)
    args = parser.parse_args()
    if args.case in {"singleton", "staged_singleton", "legacy_placement", "identity", "all"} and args.singleton is None:
        parser.error("Actual frontend ownership cases require --singleton")
    if args.case in {"staged_singleton", "all"} and args.module is None:
        parser.error("Staged singleton requires --module")
    # A staged test nests another native singleton under the CTest owner.
    # Use a short, exclusive Linux scratch tree outside its TMPDIR, both to
    # leave room for native socket suffixes and to preserve failure evidence.
    staging = args.case in {"staged_singleton", "all"}
    root = Path(tempfile.mkdtemp(prefix="lm-stage-" if staging else "llaminar-mpi-replay-",
                                 dir="/tmp" if staging else None))
    replay = Replay(args, root)
    passed = False
    failures = []
    try:
        selected = ((args.case,) if args.case != "all" else
                    ("launcher", "mpi", "native_options", "stubborn", "ctest_timeout", "singleton",
                     "staged_singleton", "legacy_placement", "identity"))
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
