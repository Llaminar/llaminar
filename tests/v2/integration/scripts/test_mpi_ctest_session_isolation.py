## @file test_mpi_ctest_session_isolation.py
#  @brief Verify that concurrently scheduled CTest MPI jobs own distinct
#  launcher owners and that those registrations execute and retire successfully.

"""Protect the Unit/preflight gates from Open MPI session-directory races."""

from __future__ import annotations

import json
import copy
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


def _validate_session_ownership(inventory: dict, build_dir: Path) -> dict[str, Path]:
    """Require the shared production owner around every actual MPI gate command."""
    owners: dict[Path, str] = {}
    checked: dict[str, Path] = {}
    for test in inventory["tests"]:
        command = test.get("command", [])
        native_names = {"mpirun", "mpiexec", "mpirun.openmpi", "mpiexec.openmpi", "orterun", "prterun"}
        # A script's --mpirun operand or an application's prompt is not the
        # process being launched. Only the original command slot owns grammar.
        raw_native = bool(command) and Path(command[0]).name in native_names
        owned_native = len(command) > 4 and Path(command[4]).name in native_names
        labels = next(
            (item["value"] for item in test.get("properties", [])
             if item["name"] == "LABELS"),
            [],
        )
        if "Unit" not in labels and "ProductionTestPreflight" not in labels:
            continue
        if not raw_native and not owned_native and "ProcessOwned" not in labels:
            continue

        name = test["name"]
        if (len(command) < 5 or Path(command[0]).name != "llaminar_mpi_session"
                or command[1] != "--session-base" or command[3] != "--"):
            raise AssertionError(f"{name} has no original MPI process/file owner")
        properties = {item["name"]: item["value"] for item in test.get("properties", [])}
        if (properties.get("TIMEOUT_SIGNAL_NAME") != "SIGTERM"
                or properties.get("TIMEOUT_SIGNAL_GRACE_PERIOD") != 36):
            raise AssertionError(f"{name} does not retain its owner during timeout cleanup")
        helper = Path(command[0]).resolve()
        if not helper.is_relative_to(build_dir.resolve()) or not helper.is_file():
            raise AssertionError(f"{name} has an unavailable MPI session helper: {helper}")
        session_dir = Path(command[2]).resolve()
        if not session_dir.is_relative_to(build_dir.resolve()) or not session_dir.is_dir():
            raise AssertionError(f"{name} has an unavailable session root: {session_dir}")
        location_options = {
            "orte_tmpdir_base", "prte_tmpdir_base", "btl_vader_backing_directory",
            "btl_sm_backing_directory", "shmem_mmap_backing_file_base_dir",
            "orte_local_tmpdir_base", "orte_remote_tmpdir_base",
            "prte_local_tmpdir_base", "prte_remote_tmpdir_base",
        }
        if owned_native:
            index = 5
            while index < len(command) and command[index] != "--":
                if command[index] in {"--mca", "-mca", "--gmca", "-gmca"}:
                    if index + 2 >= len(command):
                        raise AssertionError(f"{name} has incomplete native MCA options")
                    if command[index + 1] in location_options:
                        raise AssertionError(f"{name} overrides its MPI owner's file locations")
                    index += 3
                elif command[index] == "-x":
                    index += 2
                else:
                    index += 1
            if index + 1 >= len(command):
                raise AssertionError(f"{name} has no explicit MPI/application boundary")
        if previous := owners.get(session_dir):
            raise AssertionError(f"{name} shares its MPI session root with {previous}")
        owners[session_dir] = name
        checked[name] = session_dir

    if not checked:
        raise AssertionError("No MPI/process-owned Unit/preflight registrations were checked")
    return checked


def _assert_session_ownership(build_dir: Path) -> dict[str, Path]:
    """Validate the actual inventory and reject its original unowned launch form."""
    inventory = json.loads(
        _run(["ctest", "--test-dir", str(build_dir), "--show-only=json-v1"])
    )
    checked = _validate_session_ownership(inventory, build_dir)
    original = copy.deepcopy(inventory)
    first = next(test for test in original["tests"] if test["name"] in checked)
    first["command"] = first["command"][4:]
    try:
        _validate_session_ownership(original, build_dir)
    except AssertionError as error:
        if "no original MPI process/file owner" not in str(error):
            raise
    else:
        raise AssertionError("Unowned MPI registration negative control unexpectedly passed")
    return checked


def main() -> None:
    """Inspect registrations, then repeat parallel one-rank transcode jobs."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_mpi_ctest_session_isolation.py BUILD_DIR")
    build_dir = Path(sys.argv[1]).resolve()
    checked = _assert_session_ownership(build_dir)
    repeated = [f"V2_Unit_{quant}_Transcode" for quant in ("Q2_K", "Q3_K", "Q4_K", "Q6_K")]
    before = {name: set(checked[name].iterdir()) for name in repeated}

    # These four small tests previously shared /tmp/ompi.<host>.<uid>. Run
    # their real CTest registrations concurrently and repeatedly so an MPI
    # launcher teardown cannot delete another launcher's session parent.
    _run([
        "ctest", "--test-dir", str(build_dir), "--output-on-failure",
        "--parallel", "4", "--repeat", "until-fail:5", "-R",
        r"^V2_Unit_(Q2_K|Q3_K|Q4_K|Q6_K)_Transcode$",
    ])
    for name in repeated:
        if leaked := set(checked[name].iterdir()) - before[name]:
            raise AssertionError(f"{name} retained runtime files after exit: {sorted(leaked)}")
    print(f"MPI session ownership and retirement passed for {len(checked)} Unit/preflight jobs")


if __name__ == "__main__":
    main()
