## @file test_native_cpu_entry_lifecycle.py
#  @brief Exercise actual native startup exec, loan continuity, rejection and cancellation.
"""Focused CPU-only process lifecycle controls; no model, MPI or GPU execution."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import tempfile
import time


def observation(path: Path) -> dict[str, str]:
    """Read one atomically published, test-owned native exec observation."""
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


def wait_observation(path: Path, process: subprocess.Popen) -> dict[str, str]:
    """Require actual entry at the exec seam rather than sleeping past a failed child."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.exists():
            return observation(path)
        if process.poll() is not None:
            raise AssertionError(f"Child exited before native exec observation: {process.returncode}")
        time.sleep(0.01)
    raise AssertionError("Native startup did not reach its exec seam")


def run_case(args, root: Path, action: str, expected: int, *, initial_worker: bool = False) -> None:
    """Cross the real owner/exec seam and require exact retirement and status evidence."""
    case = root / action
    case.mkdir()
    base = case / "sessions"
    base.mkdir()
    audit = case / "exec.txt"
    environment = dict(os.environ)
    environment.pop("LLAMINAR_NATIVE_CORE_PLACES", None)
    environment.update(
        OMP_PLACES="cores", OMP_PROC_BIND="spread", OMP_DYNAMIC="false",
        LLAMINAR_FORCE_CPU_ONLY_STARTUP="1", LLAMINAR_SKIP_CUDA_STARTUP="1",
        LLAMINAR_SKIP_ROCM_STARTUP="1", LD_PRELOAD=str(args.fault),
        LLAMINAR_TEST_NATIVE_EXEC_ACTION=action,
        LLAMINAR_TEST_NATIVE_EXEC_AUDIT=str(audit),
    )
    if initial_worker:
        environment["LLAMINAR_TEST_NATIVE_ENTRY_WORKER"] = "1"
    else:
        environment.pop("LLAMINAR_TEST_NATIVE_ENTRY_WORKER", None)
    command = [str(args.helper), "--session-base", str(base), "--", str(args.probe),
               "--native-startup-ownership-contract"]
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    try:
        seen = None
        if action == "pause":
            seen = wait_observation(audit, process)
            process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=30)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=30)
        raise
    (case / "output.log").write_text(output)
    if process.returncode != expected:
        raise AssertionError(f"{action}: exit {process.returncode}, expected {expected}\n{output}")
    if initial_worker:
        if audit.exists() or "process with existing workers" not in output:
            raise AssertionError(f"Pre-entry workers did not prevent publication/exec:\n{output}")
    else:
        seen = seen or observation(audit)
        if action == "pass":
            match = re.search(r"native_entry_pid=(\d+) shared_device=(\d+) shared_inode=(\d+)", output)
            if not match or tuple(match.groups()) != (seen["pid"], seen["shared_device"], seen["shared_inode"]):
                raise AssertionError(f"Native exec changed PID or original shared-directory loan:\n{output}")
        elif action == "fail" and "Cannot enter native CPU places" not in output:
            raise AssertionError(f"Exec failure was misclassified:\n{output}")
        for name in ("temporary", "shared"):
            if Path(seen[name]).exists():
                raise AssertionError(f"{action}: original {name} root survived owner retirement")
    if list(base.iterdir()):
        raise AssertionError(f"{action}: original session directory was not retired")
    print(f"PASS {action}: actual original-entry lifecycle")


def main() -> None:
    """Run only the explicit focused lifecycle controls on production-linked executables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--fault", type=Path, required=True)
    args = parser.parse_args()
    for name in ("probe", "helper", "fault"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    root = Path(tempfile.mkdtemp(prefix="llaminar-native-entry-"))
    try:
        run_case(args, root, "pass", 0)
        run_case(args, root, "fail", 125)
        run_case(args, root, "pause", 143)
        run_case(args, root, "worker", 125, initial_worker=True)
    except BaseException:
        print(f"Native entry failure evidence retained at {root}")
        raise
    else:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
