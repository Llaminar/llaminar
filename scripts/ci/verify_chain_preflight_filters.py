#!/usr/bin/env python3
"""Resolve chain-added preflight filter names to their TEST() declarations.

Standing guard for the arm-idea08 chain: every focused preflight registration a
lane adds must select REAL tests. GTest exits 0 on an empty filter, so a typo'd
filter (or the Test__ prefix mistake) certifies nothing silently. This script
enumerates the chain's registrations, parses their --gtest_filter= names and
resolves each against the test sources: a name that does not resolve is a
vacuous filter and must be fixed before handback.

Usage: python3 scripts/ci/verify_chain_preflight_filters.py [--repo ROOT]
Exit code 1 if any chain filter name fails to resolve.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Chain registrations: entry-name -> (cmake file, expected test-source owner).
CHAIN_ENTRIES = {
    "V2_Integration_MPITopologyCPUPreparedLayoutObservation": ("tests/v2/CMakeLists.txt", "tests/v2/unit/utils/Test__MPITopologyCapabilityExchange.cpp"),
    "V2_Integration_ResolvedCpuPreparedLayoutDocument": ("tests/v2/CMakeLists.txt", "tests/v2/unit/config/Test__OrchestrationConfigDocument.cpp"),
    "V2_Integration_ExpertTierRectangularCPUWire": ("tests/v2/cmake/V2ExpertTierRectangularWeightTests.cmake", "tests/v2/unit/moe/Test__ExpertTierWeightStream.cpp"),
    "V2_Integration_MoELocalTPReplicatedDecodeDefault": ("tests/v2/CMakeLists.txt", "tests/v2/unit/execution/moe/Test__MoEExpertOverlayRuntimePlan.cpp"),
    "V2_Integration_MoEPipelineStageCapacity": ("tests/v2/CMakeLists.txt", "tests/v2/unit/execution/moe/Test__MoEOverlayCapacityResolver.cpp"),
}

def filter_names(cmake_file: Path, entry: str):
    text = cmake_file.read_text()
    # find the add_v2_test(V2_Integration_X ... ) block
    start = text.find(f"add_v2_test({entry}")
    if start < 0:
        sys.exit(f"registration {entry} not found in {cmake_file}")
    end = text.find("\n)", start)
    block = text[start:end + 2]
    m = re.search(r"--gtest_filter=([^\s]+)", block)
    if not m:
        return []
    return [n for n in m.group(1).split(":") if n]

def resolve(test_source: Path, name: str):
    suite, _, case = name.partition(".")
    if not case:
        return f"NO (no case part in {name})"
    text = test_source.read_text(errors="ignore")
    if case.endswith("*"):
        # URL-style wildcard: at least one declaration must match the prefix.
        prefix = re.escape(case.rstrip("*"))
        pattern = re.compile(
            r"\bTEST(?:_F|_P)?\(\s*" + re.escape(suite) + r"\s*,\s*" + prefix)
        return "OK" if pattern.search(text) else "NO"
    # TEST(Suite, Case) / TEST_F / TEST_P declarations
    pattern = re.compile(r"\bTEST(?:_F|_P)?\(\s*" + re.escape(suite) + r"\s*,\s*" + re.escape(case) + r"\s*\)")
    return "OK" if pattern.search(text) else "NO"

def main():
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO
    failures = []
    total = 0
    for entry, (cmake_rel, source_rel) in CHAIN_ENTRIES.items():
        cmake_file = repo / cmake_rel
        source = repo / source_rel
        for name in filter_names(cmake_file, entry):
            total += 1
            result = resolve(source, name)
            print(f"{result:4}  {entry}: {name}")
            if result != "OK":
                failures.append(f"{entry}: {name}")
    print(f"\nresolved {total - len(failures)}/{total} chain filter names")
    if failures:
        print("MISSING:")
        for f in failures:
            print(" -", f)
        sys.exit(1)

if __name__ == "__main__":
    main()
