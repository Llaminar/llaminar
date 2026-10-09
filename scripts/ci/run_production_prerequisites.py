#!/usr/bin/env python3
"""Run the canonical Unit/preflight transaction without admitting any model.

Run this command in the installed test runner before the outer CI driver starts
Release HTTP cells in the sibling runtime image. Local callers use the same
authority with an ordinary build tree. CTest owns both complete inventories;
this entrypoint has no selectors, skip switches, receipt synthesis or model
staging. Image/source/ISA binding belongs to the outer pipeline that launches
the immutable test-runner ID and retains this command's canonical receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from run_production_parity_campaigns import (
    run_production_test_preflight,
    reuse_unchanged_production_test_preflight,
)


def validate_source_freshness(build: Path, report: Path) -> None:
    """Reject edits to interpreted tests and policies that Ninja cannot rebuild.

    Local receipt reuse belongs to this checkout's configured build. Git owns
    the source inventory; file and containing-directory metadata conservatively
    detects edits, additions and removals after the actual gate. No payload is
    read or hashed, and ignored models, caches and build products are excluded.
    The canonical validator separately authenticates success and test inventory.
    """
    root = Path(__file__).resolve().parents[2]
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    homes = [line.split("=", 1)[1] for line in cache.splitlines()
             if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL=")]
    if len(homes) != 1 or Path(homes[0]).resolve() != root / "src/v2":
        raise ValueError("prerequisite reuse requires this checkout's configured build")
    document = json.loads(report.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    completed = document.get("preflight_completed_ns")
    if type(completed) is not int or completed <= 0:
        raise ValueError("local prerequisite reuse requires an explicit gate completion time")
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root,
    ).decode("utf-8").split("\0")
    paths: set[Path] = set()
    for name in filter(None, names):
        path = root / name
        paths.add(path)
        # A removed tracked file still has a live containing directory. Its
        # timestamp records deletion; do not pretend that missing bytes passed.
        paths.update(parent for parent in path.parents if parent.is_relative_to(root))
    for path in sorted(paths):
        try:
            value = path.lstat()
        except FileNotFoundError:
            continue  # The containing-directory boundary remains mandatory.
        if max(value.st_mtime_ns, value.st_ctime_ns) > completed:
            raise ValueError(f"prerequisite source changed after the completed gate: {path}")


def main(argv: list[str] | None = None) -> int:
    """Execute complete gates or authenticate an explicitly selected prior receipt.

    An installed receipt authenticates the prebuilt test inventory instead of
    attempting to rebuild stripped objects. It never replaces test execution.
    An existing result directory is rejected before any build or device work,
    so a failed rerun cannot overwrite earlier prerequisite evidence.
    Local reuse rebuilds first, then rejects changed build or test inventory.
    Reuse never writes a replacement receipt or silently selects a new run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--installed-build-receipt", type=Path,
                        help="Authenticate an immutable test runner's installed test files; still run both gates")
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--output", type=Path,
                        help="New directory for the canonical receipt, CTest logs and JUnit evidence")
    evidence.add_argument("--reuse-report", type=Path,
                         help="Rebuild prerequisites, then authenticate an unchanged complete prior receipt")
    args = parser.parse_args(argv)
    build = args.build_dir.expanduser().resolve(strict=True)
    installed = (args.installed_build_receipt.expanduser().resolve(strict=True)
                 if args.installed_build_receipt is not None else None)
    if args.reuse_report is not None:
        if installed is not None:
            parser.error("Receipt reuse requires a local build, not an installed runner receipt")
        # Source edits must cross Ninja's rebuild boundary before evidence can
        # be reused. A rebuild or inventory change then makes the old receipt
        # fail closed; this mode never silently runs a different gate.
        code = subprocess.run(["cmake", "--build", str(build), "--parallel", "--target",
                               "v2_unit_gate", "v2_production_test_preflight_gate"]).returncode
        if code:
            return code
        try:
            validate_source_freshness(build, args.reuse_report)
            return reuse_unchanged_production_test_preflight(build, args.reuse_report)[0]
        except ValueError as error:
            parser.error(str(error))
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    code, _, _ = run_production_test_preflight(
        build, None, installed_build_receipt=installed, artifact_directory=output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
