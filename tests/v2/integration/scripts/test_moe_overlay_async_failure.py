#!/usr/bin/env python3
"""Prove actual MPI service teardown fails before releasing a failed native owner.

Each observation runs in a fresh canonical process-owner session. The failed
probe must first demonstrate an unacknowledged original transaction, then reach
production follower teardown's fatal boundary. A generic nonzero exit or timeout
cannot pass; exact durable witnesses distinguish that boundary from a crash.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--mpirun", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="llaminar-async-failure-") as temporary:
        root = Path(temporary)
        for mode in ("ready", "failed"):
            directory = root / mode
            directory.mkdir()
            sessions = directory / "sessions"
            sessions.mkdir()
            command = [str(args.helper), "--session-base", str(sessions), "--",
                       args.mpirun, "-np", "2", "--bind-to", "none", "--map-by", "slot",
                       "--mca", "mpi_leave_pinned", "1",
                       "--mca", "orte_allowed_exit_without_sync", "1", "--",
                       str(args.probe), str(directory), mode]
            environment = dict(os.environ, TMPDIR=str(sessions))
            process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                # Signal the canonical ancestor so it retires every actual MPI
                # child before the temporary session roots can be removed.
                process.terminate()
                try:
                    process.communicate(timeout=36)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=5)
                raise AssertionError(f"{mode} native replay exceeded its standard progress deadline")
            result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            if result.returncode == 0 and mode == "failed":
                raise AssertionError("Failed native fence returned success")
            if result.returncode != 0 and mode == "ready":
                raise AssertionError("Positive native retirement failed: " + result.stdout + result.stderr)
            required = {"submitted-original-ticket"}
            forbidden = set()
            if mode == "ready":
                required |= {"completion-retired", "owner-released"}
                forbidden |= {"fatal-before-owner-release", "unexpected-owner-unwind"}
            else:
                required |= {"service-retained-failed-fence", "no-false-completion", "fatal-before-owner-release"}
                forbidden |= {"owner-released", "completion-retired", "unexpected-owner-unwind"}
            for marker in required:
                if not (directory / marker).is_file():
                    raise AssertionError(f"Missing {mode} witness {marker}: " + result.stdout + result.stderr)
            for marker in forbidden:
                if (directory / marker).exists():
                    raise AssertionError(f"Forbidden {mode} lifetime transition: {marker}")
            print(f"Original MPI async service {mode}: exact terminal ownership proven", flush=True)


if __name__ == "__main__":
    main()
