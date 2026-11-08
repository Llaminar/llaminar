#!/usr/bin/env python3
"""Run the complete device-free Unit inventory in an installed PR test image.

CTest owns test selection, resource weights and timeouts. The installed receipt
proves that every Unit executable exists; JUnit evidence proves every discovered
test actually ran without skips. This entrypoint never runs integration tests
and its receipt cannot certify production prerequisites.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from prebuilt_test_image import InstalledTestScope, validate
from production_artifacts import write_json
from run_production_parity_campaigns import discover_production_parity_unit_tests


def validate_results(path: Path, expected: tuple[str, ...]) -> None:
    """Reject omitted, duplicate, skipped or unsuccessful Unit observations."""
    cases = list(ET.parse(path).getroot().iter("testcase"))
    if sorted(case.get("name") for case in cases) != sorted(expected):
        raise ValueError("Unit JUnit evidence does not cover the exact CTest inventory")
    for case in cases:
        if (case.get("status") != "run"
                or any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))):
            raise ValueError(f"Unit test did not pass without skipping: {case.get('name')}")


def main(argv: list[str] | None = None) -> int:
    """Authenticate the Unit installation, execute it and retain exact evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--installed-build-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    build = args.build_dir.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    validate(args.installed_build_receipt, build, InstalledTestScope.UNIT)
    tests = discover_production_parity_unit_tests(build)
    result = subprocess.run([
        "ctest", "--test-dir", str(build), "--output-on-failure", "--parallel",
        str(len(os.sched_getaffinity(0))), "--no-tests=error", "-L", "^Unit$",
        "--output-log", str(output / "unit.log"),
        "--output-junit", str(output / "unit.xml"),
    ], check=False)
    report = {"schema": 1, "scope": "unit", "complete": False,
              "return_code": result.returncode, "tests": tests, "test_count": len(tests)}
    write_json(output / "unit.json", report)
    if result.returncode:
        return result.returncode
    validate_results(output / "unit.xml", tests)
    report["complete"] = True
    write_json(output / "unit.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
