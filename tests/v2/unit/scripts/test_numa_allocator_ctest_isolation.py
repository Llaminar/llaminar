#!/usr/bin/env python3
"""Verify the CTest isolation contract for exact physical NUMA placement."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


TARGET_TEST = "V2_Unit_NUMAAllocator"


def test_properties(ctest: str, build_dir: Path) -> dict[str, object]:
    """Return the registered CTest properties for the one physical NUMA test."""
    result = subprocess.run(
        [
            ctest,
            "--test-dir",
            str(build_dir),
            "--show-only=json-v1",
            "-R",
            f"^{TARGET_TEST}$",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(result.stdout)
    tests = document.get("tests", [])
    if len(tests) != 1 or tests[0].get("name") != TARGET_TEST:
        raise AssertionError(
            f"expected exactly {TARGET_TEST!r} in CTest inventory, got {tests!r}"
        )
    return {
        property_["name"]: property_.get("value")
        for property_ in tests[0].get("properties", [])
    }


def main() -> int:
    """Fail when the physical placement proof can race the Unit fan-out."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctest", required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()

    properties = test_properties(args.ctest, args.build_dir)
    if properties.get("RUN_SERIAL") not in (True, "TRUE", "True", "1"):
        raise AssertionError(
            f"{TARGET_TEST} must retain RUN_SERIAL for exact first-touch certification; "
            f"registered properties were {properties!r}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"NUMA CTest isolation contract failed: {error}", file=sys.stderr)
        raise SystemExit(1)
