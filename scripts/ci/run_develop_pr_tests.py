#!/usr/bin/env python3
"""Run one explicitly isolated PR test scope in the installed image.

CTest owns test selection, resource weights and timeouts. The canonical
production scheduler supplies the host lane. A separate GPU container executes
the CMake-owned single-device PR inventory, serially on CUDA then ROCm.
Installed-image admission proves executable presence,
while separate JUnit evidence proves every selected test ran without skips.
This scoped PR receipt cannot certify complete production prerequisites.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from enum import Enum
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from prebuilt_test_image import InstalledTestScope, validate
from production_artifacts import write_json
from run_production_parity_campaigns import (
    PreflightExecutionLane, PreflightRegistration, discover_production_parity_unit_tests,
    discover_production_test_preflight_registrations, preflight_execution_plan,
    _preflight_index_selection,
)


class PRTestScope(str, Enum):
    """Disjoint container ownership: device-free checks or required GPU checks."""

    HOST = "unit-and-host"
    GPU = "single-gpu"

    @property
    def lanes(self) -> tuple[str, ...]:
        """Return the exact receipt lanes required for this container scope."""
        return ("unit", "host") if self is PRTestScope.HOST else ("cuda", "rocm")

    @property
    def backends(self) -> str:
        """Expose accelerators only in the explicitly selected GPU container."""
        return "CPU" if self is PRTestScope.HOST else "CUDA+ROCm"


def single_gpu_plan(registrations: tuple[PreflightRegistration, ...]) -> dict[str, tuple[str, ...]]:
    """Admit every CMake PR GPU member and reject incomplete hardware contracts.

    This is a declared subset of production preflight, never a filter that
    silently drops tests after hardware discovery or a failed execution.
    Missing devices must fail the native tests; neither vendor is optional.
    """
    selected = {"cuda": [], "rocm": []}
    for row in registrations:
        if "DevelopGPUPreflight" not in row.labels:
            continue
        vendors = row.labels & {"CUDA", "ROCm"}
        if (len(vendors) != 1 or "SingleGPU" not in row.labels
                or "HardwareOptional" in row.labels or row.mpi_ranks != 1
                or row.lane not in (PreflightExecutionLane.CUDA_SOCKET,
                                    PreflightExecutionLane.ROCM_SOCKET)):
            raise ValueError(f"PR GPU test lacks strict single-device ownership: {row.name}")
        selected[next(iter(vendors)).lower()].append(row.name)
    if not all(selected.values()):
        raise ValueError("PR GPU gate requires nonempty CUDA and ROCm inventories")
    return {lane: tuple(names) for lane, names in selected.items()}


def validate_results(path: Path, expected: tuple[str, ...]) -> None:
    """Reject omitted, duplicate, skipped or unsuccessful PR observations."""
    cases = list(ET.parse(path).getroot().iter("testcase"))
    if sorted(case.get("name") for case in cases) != sorted(expected):
        raise ValueError("PR JUnit evidence does not cover the exact CTest inventory")
    for case in cases:
        if (case.get("status") != "run"
                or any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))):
            raise ValueError(f"PR test did not pass without skipping: {case.get('name')}")


def main(argv: list[str] | None = None) -> int:
    """Authenticate installed tests and retain exact, non-skipping scope evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--installed-build-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", type=PRTestScope, choices=[scope.value for scope in PRTestScope],
                        default=PRTestScope.HOST)
    args = parser.parse_args(argv)
    build = args.build_dir.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # The model-free companion contains Unit and all preflight executables.
    # Admission authenticates installation; each scope selects its own lanes.
    validate(args.installed_build_receipt, build, InstalledTestScope.PRODUCTION)
    registrations = discover_production_test_preflight_registrations(build)
    if args.scope is PRTestScope.HOST:
        units = discover_production_parity_unit_tests(build)
        hosts = preflight_execution_plan(registrations)[PreflightExecutionLane.HOST]
        if not units or not hosts:
            raise ValueError("PR gate requires nonempty Unit and host preflight inventories")
        lanes = (("unit", units, len(os.sched_getaffinity(0))), ("host", hosts, 1))
    else:
        lanes = tuple((lane, tests, 1) for lane, tests in single_gpu_plan(registrations).items())
    report = {"schema": 1, "scope": args.scope.value, "complete": False,
              "return_code": None, "lanes": {}}
    path = output / "pr-tests.json"
    write_json(path, report)
    # Unit keeps unrestricted parallelism. Host follows the production lane's
    # serial resource ownership; each test retains its own CMake worker budget.
    # The AVX2 runner has one CPU package and one GPU of each vendor. Keep GPU
    # lanes sequential so their MPI/OpenMP placement never competes with itself.
    for lane, tests, parallel in lanes:
        with ExitStack() as selectors:
            selection = (["-L", "^Unit$"] if lane == "unit" else
                         ["-I", str(selectors.enter_context(
                             _preflight_index_selection(build, registrations, tests)))])
            result = subprocess.run([
                "ctest", "--test-dir", str(build), "--output-on-failure", "--parallel",
                str(parallel), "--no-tests=error", *selection,
                "--output-log", str(output / f"{lane}.log"),
                "--output-junit", str(output / f"{lane}.xml"),
            ], check=False)
        evidence = {"complete": False, "return_code": result.returncode,
                    "tests": tests, "test_count": len(tests)}
        report["lanes"][lane] = evidence
        report["return_code"] = result.returncode
        write_json(path, report)
        if result.returncode:
            return result.returncode
        validate_results(output / f"{lane}.xml", tests)
        evidence["complete"] = True
        write_json(path, report)
    report["complete"] = True
    write_json(path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
