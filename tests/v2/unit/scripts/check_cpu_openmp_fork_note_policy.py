#!/usr/bin/env python3
"""Hold the runtime OpenMP fork inventory exhaustively against the note authority.

Every real fork in src/v2 - whether spelled as a raw `#pragma omp parallel`
or as a `_Pragma("omp parallel ...")` - must either observe itself through
`noteOpenMPTeamFork()` on the immediately preceding line, or belong to a
NAMED SITE exemption (exact file and enclosing function, each with its own
reason: the one-time weight-preparation, loading, planning and bootstrap
sites that run outside execution). utils/OpenMPUtils.h is the single
file-level authority: its `_Pragma` lines are the macro definitions whose
expansions are the noted forks by construction.

The checker is fail-closed: it scans every supported file whether or not a
pragma is present (an orphan note fails), fails on a missing source root,
rejects duplicate or non-adjacent notes, and never admits a whole file as
exempt. Supported grammar, explicitly: a line whose first token is `#` or
`_Pragma`, followed by `omp parallel` with arbitrary horizontal whitespace.
It is not C++ semantic analysis; unknown provenance fails closed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys

from check_gpu_blocking_sync_policy import (
    enclosing_caller,
    function_intervals,
    strip_comments_and_literals,
)

SUPPORTED_SUFFIXES = (".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh", ".inc", ".inl")

# Exact (file, enclosing function) -> reason. Never a generic exemption.
EXEMPT_SITES: dict[tuple[str, str], str] = {
    ("src/v2/app/MPIBootstrapPhase.cpp", "bindOpenMPWorkersFastestFirst"):
        "one-time rank bootstrap affinity binding at process start",
    ("src/v2/app/MPIBootstrapPhase.cpp", "verifyStartupThreadAffinity"):
        "one-time rank bootstrap affinity verification at process start",
    ("src/v2/kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h", "repackNativeBlocksToInterleaved"):
        "one-time weight packing for prepared publication",
    ("src/v2/kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h", "packExpandedInt8Direct"):
        "one-time weight packing for prepared publication",
    ("src/v2/kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h", "packWeightsCPUNativeQuantised"):
        "one-time weight packing for prepared publication",
    ("src/v2/kernels/cpu/gemm/CPUPreparedWeightWriter.cpp", "append"):
        "one-time prepared-weight publication",
    ("src/v2/kernels/cuda/gemm/CUDAWeightPacker.cpp", "packNativeQuantisedCUDA"):
        "one-time host-side weight packing for GPU preparation",
    ("src/v2/kernels/cuda/gemm/CUDAWeightPacker.cpp", "packMoEExpertsCUDA"):
        "one-time host-side weight packing for GPU preparation",
    ("src/v2/kernels/rocm/gemm/ROCmWeightPacker.cpp", "packNativeQuantised"):
        "one-time host-side weight packing for GPU preparation",
    ("src/v2/kernels/rocm/gemm/ROCmWeightPacker.cpp", "packWeightsToROCm"):
        "one-time host-side weight packing for GPU preparation",
    ("src/v2/kernels/rocm/gemm/ROCmWeightPacker.cpp", "packMoEExpertsROCm"):
        "one-time host-side weight packing for GPU preparation",
    ("src/v2/loaders/MmapRegion.h", "create"):
        "one-time model load and mmap population",
    ("src/v2/loaders/PreparedHalfWeights.h", "materializePreparedHalfWeights"):
        "one-time prepared-halfweight materialization at load",
    ("src/v2/loaders/WeightManager.cpp", "sliceRows"):
        "one-time model load work",
    ("src/v2/loaders/WeightManager.cpp", "sliceTailColumns"):
        "one-time model load work",
    ("src/v2/planning/PlanningMemoryBandwidthMeasurement.cpp", "measure"):
        "planning-time bandwidth measurement outside execution",
    ("src/v2/execution/moe/MoEExpertWeightService.cpp", "prepareGemmEngines"):
        "expert GEMM engine preparation at graph build",
    ("src/v2/kernels/cpu/gemm/CPUNativeQuantisedWindowPacking.inc", "appendNativeWeightGroups"):
        "one-time prepared-weight window publication inside the packer",
}

# The single file-level authority: its _Pragma lines are the macro definitions
# whose expansions call the note themselves.
AUTHORITY_FILE = "src/v2/utils/OpenMPUtils.h"

PRAGMA = re.compile(r"^[ \t]*(?:#[ \t]*pragma[ \t]+omp[ \t]+parallel\b|"
                    r"_Pragma[ \t]*\([ \t]*\"omp[ \t]+parallel)", re.MULTILINE)
NOTE = "noteOpenMPTeamFork"


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def scan(relative: str, raw: str) -> tuple[list[Violation], int, int]:
    """Return violations plus (noted, exempted) site counts for one unit."""
    if relative == AUTHORITY_FILE:
        return [], 0, len(PRAGMA.findall(raw))
    lines = [re.sub(r"//.*$", "", line) for line in
             re.sub(r"/\*.*?\*/", lambda match: re.sub(r"[^\n]", " ", match.group(0)),
                    raw, flags=re.DOTALL).splitlines()]
    stripped = strip_comments_and_literals(raw)
    intervals = function_intervals(stripped)
    # The policy helpers resolve positions inside the comment-and-literal-
    # stripped text, so probe offsets come from that same text's line starts.
    stripped_lines = stripped.splitlines()
    line_offsets = [0]
    for line in stripped_lines:
        line_offsets.append(line_offsets[-1] + len(line) + 1)

    def enclosing(line_index: int) -> str:
        offset = line_offsets[line_index] if line_index < len(line_offsets) else len(stripped)
        return enclosing_caller(intervals, offset).rsplit("::", 1)[-1]

    violations: list[Violation] = []
    noted, exempted = 0, 0
    index = 0
    while index < len(lines):
        if NOTE in lines[index]:
            following = index + 1
            while following < len(lines) and not lines[following].strip():
                following += 1
            if following < len(lines) and PRAGMA.match(lines[following]):
                noted += 1
                index = following + 1  # The adjacent pair is consumed as one unit.
            else:
                violations.append(Violation(relative, index + 1,
                                            "noteOpenMPTeamFork with no adjacent pragma"))
                index += 1
            continue
        match = PRAGMA.match(lines[index])
        if match is None:
            index += 1
            continue
        function = enclosing(index)
        if (relative, function) in EXEMPT_SITES:
            exempted += 1
            index += 1
            continue
        violations.append(Violation(relative, index + 1,
                                    f"raw OpenMP fork in {function or '<global>'} outside the note "
                                    "authority and outside the named site exemptions"))
        index += 1
    return violations, noted, exempted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    source_root = args.repo_root / "src/v2"
    if not source_root.is_dir():
        print(f"missing source root: {source_root}", file=sys.stderr)
        return 2
    violations: list[Violation] = []
    noted, exempted, files = 0, 0, 0
    for path in sorted(source_root.rglob("*")):
        if path.suffix not in SUPPORTED_SUFFIXES or not path.is_file():
            continue
        relative = str(path.relative_to(args.repo_root))
        raw = path.read_text(encoding="utf-8", errors="replace")
        if not PRAGMA.search(raw) and NOTE not in raw:
            continue
        files += 1
        file_violations, file_noted, file_exempted = scan(relative, raw)
        violations.extend(file_violations)
        noted += file_noted
        exempted += file_exempted
    for violation in violations:
        print(violation, file=sys.stderr)
    if violations:
        return 1
    print(f"OpenMP fork-note policy: {noted} noted sites, {exempted} named exempt sites, "
          f"{files} inventoried files, inventory exhaustive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
