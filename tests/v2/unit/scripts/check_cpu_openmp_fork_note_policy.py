#!/usr/bin/env python3
"""Hold the runtime OpenMP fork inventory exhaustively against the note authority.

Every raw `#pragma omp parallel` site in src/v2 is a real fork, and the team
fork counter (utils/OpenMPUtils.h) may only claim "every real fork" when each
site either observes itself through `noteOpenMPTeamFork()` immediately before
the pragma or belongs to the NAMED one-time exemptions (weight preparation,
model loading, planning measurement, rank bootstrap, expert engine
preparation at graph build). The checker also rejects an orphan note - a
noteOpenMPTeamFork call with no pragma after it - so the inventory cannot
drift in either direction. It is not C++ semantic analysis; unknown
provenance fails closed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]

# Exact file -> reason. Never a generic exemption.
EXEMPT_FILES = {
    "src/v2/utils/OpenMPUtils.h":
        "the noting macro authority itself; its _Pragma expansions are the counted forks",
    "src/v2/app/MPIBootstrapPhase.cpp":
        "one-time rank bootstrap affinity binding at process start",
    "src/v2/kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h":
        "one-time weight packing for prepared publication",
    "src/v2/kernels/cpu/gemm/CPUPreparedWeightWriter.cpp":
        "one-time prepared-weight publication",
    "src/v2/kernels/cuda/gemm/CUDAWeightPacker.cpp":
        "one-time host-side weight packing for GPU preparation",
    "src/v2/kernels/rocm/gemm/ROCmWeightPacker.cpp":
        "one-time host-side weight packing for GPU preparation",
    "src/v2/loaders/MmapRegion.h":
        "one-time model load and mmap population",
    "src/v2/loaders/PreparedHalfWeights.h":
        "one-time prepared-halfweight load work",
    "src/v2/loaders/WeightManager.cpp":
        "one-time model load work",
    "src/v2/planning/PlanningMemoryBandwidthMeasurement.cpp":
        "planning-time bandwidth measurement outside execution",
    "src/v2/execution/moe/MoEExpertWeightService.cpp":
        "expert GEMM engine preparation at graph build",
}

PRAGMA = re.compile(r"^(\s*)#pragma omp parallel\b", re.MULTILINE)
NOTE = "noteOpenMPTeamFork()"


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def strip_comment_lines(raw: str) -> list[str]:
    """Blank out // and /* */ comment text so probes never match inside comments."""
    text = re.sub(r"/\*.*?\*/", lambda match: re.sub(r"[^\n]", " ", match.group(0)),
                  raw, flags=re.DOTALL)
    return [re.sub(r"//.*$", "", line) for line in text.splitlines()]


def scan(relative: str, raw: str) -> list[Violation]:
    """Return every policy violation in one translation unit."""
    if relative in EXEMPT_FILES:
        return []
    lines = strip_comment_lines(raw)
    violations: list[Violation] = []
    noted_lines: set[int] = set()
    for index, line in enumerate(lines):
        if NOTE in line:
            # An orphan note fails unless a pragma follows within six lines.
            window = "\n".join(lines[index + 1:index + 7])
            if PRAGMA.search(window):
                for offset in range(1, 7):
                    if index + offset < len(lines) and PRAGMA.match(lines[index + offset]):
                        noted_lines.add(index + offset)
                        break
            else:
                violations.append(Violation(relative, index + 1,
                                            "noteOpenMPTeamFork with no pragma after it"))
    for index, line in enumerate(lines):
        match = PRAGMA.match(line)
        if match is None or index in noted_lines:
            continue
        violations.append(Violation(relative, index + 1,
                                    "raw OpenMP fork outside the note authority "
                                    "and outside the named exemptions"))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    violations: list[Violation] = []
    sites, exempted = 0, 0
    for path in sorted((args.repo_root / "src/v2").rglob("*")):
        if path.suffix not in (".cpp", ".h", ".hpp", ".cc"):
            continue
        relative = str(path.relative_to(args.repo_root))
        raw = path.read_text(encoding="utf-8", errors="replace")
        if PRAGMA.search(raw) is None:
            continue
        if relative in EXEMPT_FILES:
            exempted += 1
            continue
        sites += len(PRAGMA.findall(raw))
        violations.extend(scan(relative, raw))
    for violation in violations:
        print(violation, file=sys.stderr)
    if violations:
        return 1
    print(f"OpenMP fork-note policy: {sites} noted runtime sites, "
          f"{exempted} named exempt files, inventory exhaustive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
