#!/usr/bin/env python3
"""Run complete Unit and host-preflight inventories in the installed PR image.

CTest owns test selection, resource weights and timeouts. The canonical
production scheduler supplies the host lane; CUDA, ROCm and exclusive lanes
are never executed here. Installed-image admission proves executable presence,
while separate JUnit evidence proves every selected test ran without skips.
This scoped PR receipt cannot certify complete production prerequisites.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from prebuilt_test_image import InstalledTestScope, validate
from production_artifacts import write_json
from run_production_parity_campaigns import (
    PreflightExecutionLane, discover_production_parity_unit_tests,
    discover_production_test_preflight_registrations, preflight_execution_plan,
    _preflight_index_selection,
)


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
    """Authenticate installed tests and retain exact Unit and host-lane evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--installed-build-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    build = args.build_dir.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # The model-free companion contains Unit and all preflight executables.
    # Admission authenticates their installation; only the two lanes below run.
    validate(args.installed_build_receipt, build, InstalledTestScope.PRODUCTION)
    units = discover_production_parity_unit_tests(build)
    registrations = discover_production_test_preflight_registrations(build)
    plan = preflight_execution_plan(registrations)
    hosts = plan[PreflightExecutionLane.HOST]
    if not units or not hosts:
        raise ValueError("PR gate requires nonempty Unit and host preflight inventories")
    report = {"schema": 1, "scope": "unit-and-host", "complete": False,
              "return_code": None, "lanes": {}}
    path = output / "pr-tests.json"
    write_json(path, report)
    # Unit keeps unrestricted parallelism. Host follows the production lane's
    # serial resource ownership; each test retains its own CMake worker budget.
    for lane, tests, parallel in (("unit", units, len(os.sched_getaffinity(0))),
                                   ("host", hosts, 1)):
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
