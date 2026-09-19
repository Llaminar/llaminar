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
from pathlib import Path

from run_production_parity_campaigns import run_production_parity_preflight


def main(argv: list[str] | None = None) -> int:
    """Execute both complete gates once and preserve their own evidence files.

    An installed receipt authenticates the prebuilt test inventory instead of
    attempting to rebuild stripped objects. It never replaces test execution.
    An existing result directory is rejected before any build or device work,
    so a failed rerun cannot overwrite earlier prerequisite evidence.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--installed-build-receipt", type=Path,
                        help="Authenticate an immutable test runner's installed test files; still run both gates")
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for the canonical receipt, CTest logs and JUnit evidence")
    args = parser.parse_args(argv)
    build = args.build_dir.expanduser().resolve(strict=True)
    installed = (args.installed_build_receipt.expanduser().resolve(strict=True)
                 if args.installed_build_receipt is not None else None)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    code, _, _ = run_production_parity_preflight(
        build, None, installed_build_receipt=installed, artifact_directory=output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
