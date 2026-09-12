#!/usr/bin/env python3
"""Seal the build inventory before stripping a Docker builder's intermediates.

An installed test image cannot incrementally rebuild: its object files have
deliberately been discarded. This receipt changes only build preparation, not
test execution. Every Unit and ProductionParityPreflight test still runs. The
outer CI driver pins the immutable image ID and never overlays build/source
files, while this receipt checks its installed CTest and executable inventory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from production_artifacts import digest, write_json
from run_production_parity_campaigns import (
    PRODUCTION_CAMPAIGN_NAME, PRODUCTION_PARITY_PREFLIGHT_LABEL,
    PRODUCTION_PARITY_UNIT_LABEL, PRODUCTION_PARITY_UNIT_PREFIX,
    discover_production_parity_preflight_tests,
    discover_production_parity_unit_tests,
    _as_string_list, _property_map,
)


def inventory(build: Path) -> dict:
    """Describe exact registrations and installed executables/shared libraries."""
    build = build.resolve(strict=True)
    registrations = subprocess.check_output(
        ["ctest", "--test-dir", str(build), "--show-only=json-v1"], text=True)
    tests = json.loads(registrations)["tests"]
    files = [build / "CMakeCache.txt", build / "compile_commands.json",
             build / "build.ninja", build / "CMakeFiles/rules.ninja",
             *build.rglob("CTestTestfile.cmake")]
    # The builder installs the canonical gates, not unrelated integration/perf
    # targets. Still seal the complete registration inventory, but require
    # executable materialization only for tests that the pipeline will execute.
    for test in tests:
        name = test["name"]
        labels = _as_string_list(_property_map(test).get("LABELS", []))
        required = (name.startswith(PRODUCTION_PARITY_UNIT_PREFIX)
                    or PRODUCTION_PARITY_UNIT_LABEL in labels
                    or PRODUCTION_PARITY_PREFLIGHT_LABEL in labels
                    or PRODUCTION_CAMPAIGN_NAME.search(name))
        if not required:
            continue
        # CTest can omit the command entirely when it cannot resolve the test
        # binary. Checking only paths in an existing command misses that case.
        if not test.get("command"):
            raise ValueError(f"installed test executable is missing: {name}: CTest omitted command")
        for argument in test.get("command", []):
            path = Path(argument)
            if path.is_relative_to(build) and path.name.startswith("v2_") and not path.is_file():
                raise ValueError(f"installed test executable is missing: {name}: {path}")
    for path in build.rglob("*"):
        if path.is_file() and (path.name.startswith("v2_") or path.name.startswith("libllaminar")):
            files.append(path)
    return {"build_dir": str(build), "tests": tests,
            "files": {str(path.relative_to(build)): {"bytes": path.stat().st_size,
                       "mtime_seconds": int(path.stat().st_mtime)} for path in sorted(set(files))},
            "registration_digest": digest([path.read_text() for path in sorted(build.rglob("CTestTestfile.cmake"))])}


def validate(path: Path, build: Path) -> None:
    """Reject stale, partial or foreign installed test trees before any gate."""
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != 1 or receipt.get("inventory") != inventory(build):
        raise ValueError("installed test image receipt does not match this build/inventory")


def seal(path: Path, build: Path) -> None:
    """Reject invalid gate contracts before installing a test-image receipt.

    Reuse the campaign's complete registration audit, including timeout,
    fixture and label contracts. Executable presence alone is insufficient:
    an image whose tests run manually may still be inadmissible to production.
    """
    discover_production_parity_unit_tests(build)
    discover_production_parity_preflight_tests(build)
    write_json(path, {"schema": 1, "inventory": inventory(build)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    if args.seal:
        seal(args.receipt, args.build_dir)
    else:
        validate(args.receipt, args.build_dir)
