#!/usr/bin/env python3
"""Reject CPU RoPE angle storage that bypasses the admitted workspace.

CPU RoPE borrows its absolute-position angle planes from the `cpu_rope_angles`
workspace that RoPEStage binds. This lexical gate keeps the RoPE translation
units from regrowing thread-local caches or per-call allocations:

* `thread_local` is admitted only in `get_inv_freq_cached`. That memo stores
  pointers to immutable, process-lifetime inverse-frequency tables, never
  angles or growable storage. Its key is the complete input identity of the
  function: the tables depend only on (rotary width, frequency base), and the
  function has no scaling or rope-type input.
* Object declarations of `std::vector`, `resize`/`assign`/`reserve`, `new`,
  `malloc` and `make_unique`/`make_shared` are rejected in every function
  except the named exceptions below. Each exception names its tracked
  follow-up; the Q16/hybrid family is test-only today (see the follow-up) and
  keeps its own Q15 angle authority.

It is not C++ semantic analysis; unknown provenance fails closed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys

from check_gpu_blocking_sync_policy import (enclosing_caller, function_intervals,
                                            strip_comments_and_literals)

FILES = (
    "src/v2/kernels/cpu/primitives/RoPEPrimitives.cpp",
    "src/v2/kernels/cpu/ops/CPURoPEKernelT.cpp",
)

THREAD_LOCAL_OWNER = "get_inv_freq_cached"
# The process-lifetime inverse-frequency table itself (namespace scope).
GLOBAL_TABLE = "g_inv_freq_cache"

FOLLOW_UP_A = ("arm-perf-review 07-hot-path-allocation-cleanup.md, follow-up A: "
               "convert the test-only Q16/hybrid RoPE family to the admitted angle workspace")

# Exact function -> reason. Never a generic exemption.
ALLOWED_STORAGE = {
    "publish_inv_freq": "process-lifetime immutable inverse-frequency table, published once per key",
    "apply_rope_q8_1_to_fp32": FOLLOW_UP_A,
    "apply_rope_q16_1_integer": FOLLOW_UP_A,
    "apply_rope_q16_integer": FOLLOW_UP_A,
    "apply_rope_q8_1_to_q16_head_scalar": FOLLOW_UP_A,
    "apply_rope_q8_1_to_q16_head_fixed_scale": FOLLOW_UP_A,
    "apply_rope_q8_1_to_q16": FOLLOW_UP_A,
    "apply_rope_q8_1_to_q16_1": FOLLOW_UP_A,
    "apply_rope_q8_1_to_q16_fixed_scale": FOLLOW_UP_A,
    "apply_rope_q16_to_q16_fixed_scale": FOLLOW_UP_A,
    "apply_rope_q16_to_q16_dynamic_scale": FOLLOW_UP_A,
    "compute_rope_sincos_q15": FOLLOW_UP_A,
}

STORAGE = re.compile(
    r"\bstd::vector\s*<[^;{}()]*>\s+[A-Za-z_]\w*\s*[({=;]|"   # vector object declaration
    r"\bstd::vector\s*<[^;{}()]*>\s*\(|"                      # vector temporary
    r"\.\s*(?:resize|assign|reserve)\s*\(|"
    r"\bnew\s+[A-Za-z_:]|\bmalloc\s*\(|\bmake_unique\b|\bmake_shared\b")
THREAD_LOCAL = re.compile(r"\bthread_local\b")


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    caller: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.caller}: {self.message}"


def scan(path: str, raw: str) -> list[Violation]:
    """Return every policy violation in one translation unit."""
    source = strip_comments_and_literals(raw)
    intervals = function_intervals(source)
    violations: list[Violation] = []

    def line_of(offset: int) -> int:
        return source.count("\n", 0, offset) + 1

    for match in THREAD_LOCAL.finditer(source):
        caller = enclosing_caller(intervals, match.start())
        if caller != THREAD_LOCAL_OWNER:
            violations.append(Violation(path, line_of(match.start()), caller,
                                        "thread_local storage outside the named inv_freq pointer memo"))
    for match in STORAGE.finditer(source):
        caller = enclosing_caller(intervals, match.start())
        if caller == "<global>" and GLOBAL_TABLE in match.group(0):
            continue
        if caller not in ALLOWED_STORAGE:
            violations.append(Violation(path, line_of(match.start()), caller,
                                        f"angle/scratch storage outside the admitted workspace: {match.group(0).strip()}"))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    violations: list[Violation] = []
    for relative in FILES:
        path = args.repo_root / relative
        if not path.is_file():
            print(f"missing policy subject: {relative}", file=sys.stderr)
            return 2
        violations.extend(scan(relative, path.read_text(encoding="utf-8")))
    for violation in violations:
        print(violation, file=sys.stderr)
    if violations:
        return 1
    print(f"CPU RoPE angle storage policy: {len(FILES)} files clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
