#!/usr/bin/env python3
"""Run one explicitly isolated PR test scope in the installed image.

CTest owns test selection, resource weights and timeouts. The canonical
production scheduler supplies all lanes and hardware qualification. A separate
GPU container executes every eligible non-host production preflight test,
including exclusive, mixed-backend and multi-rank work.
Installed-image admission proves executable presence,
while separate JUnit evidence proves every selected test ran without skips.
This scoped PR receipt cannot certify complete production prerequisites.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from enum import Enum
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from prebuilt_test_image import InstalledTestScope, validate
from production_artifacts import write_json
from mpi_session_retirement import owned_session_bases, retirement_scope
from run_production_parity_campaigns import (
    PreflightExecutionLane, PreflightRegistration, discover_production_parity_unit_tests,
    discover_production_test_preflight_registrations, preflight_execution_plan,
    preflight_optional_hardware_requirements, preflight_hardware_qualification,
    observe_preflight_hardware, source_hardware_devices, _preflight_index_selection,
)


class PRTestScope(str, Enum):
    """Disjoint container ownership: device-free checks or required GPU checks."""

    HOST = "unit-and-host"
    GPU = "hardware-preflight"

    @property
    def lanes(self) -> tuple[str, ...]:
        """Return the exact receipt lanes required for this container scope."""
        return ("unit", "host") if self is PRTestScope.HOST else ("cuda", "rocm", "exclusive")

    @property
    def backends(self) -> str:
        """Expose accelerators only in the explicitly selected GPU container."""
        return "CPU" if self is PRTestScope.HOST else "CUDA+ROCm"


def hardware_preflight_plan(registrations: tuple[PreflightRegistration, ...],
                            hardware: dict) -> tuple[dict[str, tuple[str, ...]], dict]:
    """Run the complete hardware-qualified complement of the CPU host lane.

    CMake and the production qualifier own eligibility. MPI width, exclusive
    scheduling and mixed vendors never exclude a supported test. Only a declared
    absent hardware geometry may remain unqualified, with its evidence retained.
    """
    observed = source_hardware_devices(hardware)
    if not {"cuda", "rocm"} <= {address[0] for address in observed}:
        raise ValueError("PR hardware preflight requires both CUDA and ROCm devices")
    requirements = preflight_optional_hardware_requirements(registrations, shared_capacity=True)
    qualification = preflight_hardware_qualification(
        tuple(row.name for row in registrations), requirements, hardware if requirements else None)
    selected = set(qualification["selected"])
    plan = preflight_execution_plan(row for row in registrations if row.name in selected)
    return {"cuda": plan[PreflightExecutionLane.CUDA_SOCKET],
            "rocm": plan[PreflightExecutionLane.ROCM_SOCKET],
            "exclusive": plan[PreflightExecutionLane.EXCLUSIVE]}, qualification


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
    inventory = json.loads(subprocess.run(
        ["ctest", "--test-dir", str(build), "--show-only=json-v1"],
        check=True, capture_output=True, text=True).stdout)
    session_bases = owned_session_bases(inventory)
    qualification = None
    if args.scope is PRTestScope.HOST:
        units = discover_production_parity_unit_tests(build)
        hosts = preflight_execution_plan(registrations)[PreflightExecutionLane.HOST]
        if not units or not hosts:
            raise ValueError("PR gate requires nonempty Unit and host preflight inventories")
        lanes = (("unit", units, len(os.sched_getaffinity(0))), ("host", hosts, 1))
    else:
        plan, qualification = hardware_preflight_plan(
            registrations, observe_preflight_hardware(build, None, shared_capacity=True))
        lanes = tuple((lane, tests, 1) for lane, tests in plan.items())
    report = {"schema": 1, "scope": args.scope.value, "complete": False,
              "return_code": None, "lanes": {}}
    if qualification is not None:
        report["qualification"] = qualification
    path = output / "pr-tests.json"
    write_json(path, report)
    # Unit keeps unrestricted parallelism. Host follows the production lane's
    # serial resource ownership; each test retains its own CMake worker budget.
    # The AVX2 runner has one CPU package and one GPU of each vendor. Keep GPU
    # lanes sequential so their MPI/OpenMP placement never competes with itself.
    overall = 0
    for lane, tests, parallel in lanes:
        print(f"[production-preflight] {lane}: selected {len(tests)} tests", flush=True)
        if not tests:
            report["lanes"][lane] = {"complete": True, "return_code": 0,
                                      "tests": [], "test_count": 0}
            continue
        with ExitStack() as selectors:
            selectors.enter_context(retirement_scope(
                session_bases[name] for name in tests if name in session_bases))
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
        overall = overall or result.returncode
        report["return_code"] = overall
        write_json(path, report)
        if result.returncode not in (0, 8):
            # Infrastructure/termination errors cannot admit another native lane.
            return result.returncode
        if not result.returncode:
            try:
                validate_results(output / f"{lane}.xml", tests)
            except ValueError as error:
                evidence["validation_error"] = str(error)
                print(f"[production-preflight] {lane}: {error}", flush=True)
                overall = overall or 8
            else:
                evidence["complete"] = True
        # CTest exit 8 means completed tests failed. After native retirement,
        # retain their red result and still exercise the independent next lane.
        print(f"[production-preflight] {lane}: {len(tests)} selected, "
              f"result={'passed' if evidence['complete'] else 'failed'}", flush=True)
        report["return_code"] = overall
        write_json(path, report)
    report["complete"] = overall == 0
    report["return_code"] = overall
    write_json(path, report)
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
