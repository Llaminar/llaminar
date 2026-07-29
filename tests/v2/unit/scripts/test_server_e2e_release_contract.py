#!/usr/bin/env python3
"""Prove canonical server E2E entry points use the deployable Release binary."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


def parse_args() -> argparse.Namespace:
    """Parse the repository root supplied by CTest."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    return parser.parse_args()


def require(text: str, needle: str, owner: pathlib.Path) -> None:
    """Require one literal contract marker in an active build/test file."""

    if needle not in text:
        raise AssertionError(f"{owner} is missing required contract: {needle}")


def main() -> int:
    """Validate harness, precommit, and CTest Release ownership."""

    args = parse_args()
    root = args.repo_root.resolve()
    build_dir = args.build_dir.resolve()
    harness_path = root / "tests/v2/e2e/server/test_server_e2e.sh"
    precommit_path = root / ".githooks/pre-commit"
    tests_cmake_path = root / "tests/v2/CMakeLists.txt"
    source_cmake_path = root / "src/v2/CMakeLists.txt"
    cache_path = build_dir / "CMakeCache.txt"
    ctest_manifest_path = build_dir / "tests/v2/CTestTestfile.cmake"

    harness = harness_path.read_text(encoding="utf-8")
    precommit = precommit_path.read_text(encoding="utf-8")
    tests_cmake = tests_cmake_path.read_text(encoding="utf-8")
    source_cmake = source_cmake_path.read_text(encoding="utf-8")

    require(
        harness,
        'BINARY="${LLAMINAR_BINARY:-${REPO_ROOT}/build_v2_release/llaminar2}"',
        harness_path,
    )
    if "build_v2_integration/llaminar2" in harness:
        raise AssertionError(
            f"{harness_path} still names the Integration runtime as a server binary"
        )
    require(
        harness,
        'STARTUP_TIMEOUT="${LLAMINAR_E2E_STARTUP_TIMEOUT_SECONDS:-60}"',
        harness_path,
    )

    require(
        precommit,
        'E2E_SERVER_ARGS=(--binary "$BUILD_V2_RELEASE/llaminar2"',
        precommit_path,
    )
    require(
        precommit,
        'LLAMINAR_E2E_STARTUP_TIMEOUT_SECONDS="${LLAMINAR_E2E_STARTUP_TIMEOUT_SECONDS:-60}"',
        precommit_path,
    )
    require(
        tests_cmake,
        'if(CMAKE_BUILD_TYPE STREQUAL "Release")',
        tests_cmake_path,
    )
    require(
        tests_cmake,
        "NAME V2_E2E_Server_MultiTurn",
        tests_cmake_path,
    )

    cache = cache_path.read_text(encoding="utf-8")
    build_type_match = re.search(
        r"^CMAKE_BUILD_TYPE:STRING=(.+)$",
        cache,
        flags=re.MULTILINE,
    )
    if build_type_match is None:
        raise AssertionError(f"{cache_path} does not declare CMAKE_BUILD_TYPE")

    ctest_manifest = ctest_manifest_path.read_text(encoding="utf-8")
    build_type = build_type_match.group(1)
    server_e2e_registered = "V2_E2E_Server_MultiTurn" in ctest_manifest
    if build_type == "Release" and not server_e2e_registered:
        raise AssertionError(
            f"{ctest_manifest_path} omits the canonical server E2E in Release"
        )
    if build_type != "Release" and server_e2e_registered:
        raise AssertionError(
            f"{ctest_manifest_path} registers the canonical server E2E in {build_type}"
        )

    if "E2ERelease" in source_cmake or "E2ERELEASE" in source_cmake:
        raise AssertionError(
            f"{source_cmake_path} still defines the retired E2ERelease build type"
        )
    require(
        source_cmake,
        "set(LLAMINAR_SUPPORTED_BUILD_TYPES Debug Release Integration)",
        source_cmake_path,
    )
    require(
        source_cmake,
        "if(CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE IN_LIST LLAMINAR_SUPPORTED_BUILD_TYPES)",
        source_cmake_path,
    )

    print("server E2E Release contract: PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"server E2E Release contract: FAIL: {error}", file=sys.stderr)
        sys.exit(1)
