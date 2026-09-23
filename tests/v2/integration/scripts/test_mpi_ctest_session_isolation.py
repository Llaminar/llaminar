## @file test_mpi_ctest_session_isolation.py
#  @brief Verify that concurrently scheduled CTest MPI jobs own distinct
#  Open MPI session roots and that those registrations execute successfully.

"""Protect the Unit/preflight gates from Open MPI session-directory races."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def _run(command: list[str]) -> str:
    """Run a gate command and retain its output in a focused failure."""
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise AssertionError(
            f"{' '.join(command)} failed ({completed.returncode}):\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return completed.stdout


def _assert_session_ownership(build_dir: Path) -> int:
    """Check every MPI Unit/preflight registration for a unique build-owned root."""
    inventory = json.loads(
        _run(["ctest", "--test-dir", str(build_dir), "--show-only=json-v1"])
    )
    owners: dict[Path, str] = {}
    checked = 0
    for test in inventory["tests"]:
        command = test.get("command", [])
        if not command or Path(command[0]).name not in {"mpirun", "mpiexec"}:
            continue
        labels = next(
            (item["value"] for item in test.get("properties", [])
             if item["name"] == "LABELS"),
            [],
        )
        if "Unit" not in labels and "ProductionTestPreflight" not in labels:
            continue

        name = test["name"]
        try:
            option = command.index("orte_tmpdir_base")
            assert command[option - 1] == "--mca"
            session_dir = Path(command[option + 1]).resolve()
        except (ValueError, IndexError, AssertionError) as error:
            raise AssertionError(f"{name} has no explicit Open MPI session root") from error
        if not session_dir.is_relative_to(build_dir.resolve()) or not session_dir.is_dir():
            raise AssertionError(f"{name} has an unavailable session root: {session_dir}")
        if previous := owners.get(session_dir):
            raise AssertionError(f"{name} shares its MPI session root with {previous}")
        owners[session_dir] = name
        checked += 1

    if not checked:
        raise AssertionError("No MPI Unit/preflight registrations were checked")
    return checked


def main() -> None:
    """Inspect registrations, then repeat parallel one-rank transcode jobs."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_mpi_ctest_session_isolation.py BUILD_DIR")
    build_dir = Path(sys.argv[1]).resolve()
    checked = _assert_session_ownership(build_dir)

    # These four small tests previously shared /tmp/ompi.<host>.<uid>. Run
    # their real CTest registrations concurrently and repeatedly so an MPI
    # launcher teardown cannot delete another launcher's session parent.
    _run([
        "ctest", "--test-dir", str(build_dir), "--output-on-failure",
        "--parallel", "4", "--repeat", "until-fail:5", "-R",
        r"^V2_Unit_(Q2_K|Q3_K|Q4_K|Q6_K)_Transcode$",
    ])
    print(f"Open MPI session isolation passed for {checked} Unit/preflight jobs")


if __name__ == "__main__":
    main()
