#!/usr/bin/env python3
"""Confine writes to prepared CPU NativeQuantised bytes to named publication producers.

CPUNativeQuantisedPublishedBytes gives readers const access only and advances a
publication generation on every mutation, so the type already rejects silent
writes. This lexical gate additionally keeps the mutation entry points
(beginRewrite, retainedSlotRoot, storage replacement/resizing/release) inside
the producers that either seal the result themselves or hand it to a named
publication point that does. It also forbids casting const away from the
published bytes, which would bypass the generation. It is not C++ semantic
analysis; it complements the runtime seal lifecycle tests.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from check_gpu_blocking_sync_policy import strip_comments_and_literals

SUFFIXES = {".cpp", ".h", ".hpp", ".inc"}

# File -> why it may mutate published bytes. Keep this list narrow and named.
ALLOWED_WRITERS = {
    "src/v2/kernels/cpu/gemm/CPUNativeQuantisedPublishedBytes.h": "defines the generation-advancing mutations",
    "src/v2/kernels/cpu/gemm/CPUNativeQuantisedPackedWeights.h": "owner release for deferred packing",
    "src/v2/kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h": "prepare; seals after the last write",
    "src/v2/kernels/cpu/gemm/CPUNativeQuantisedWindowPacking.inc": "windowed prepare; the caller seals after the last window",
    "src/v2/kernels/cpu/gemm/CPUPackedWeights.h": "explicit-placement copy; reseals a sealed source",
    "src/v2/kernels/PackedWeightsSerialization.h": "wire/disk receive; KernelFactory admits before engine publication",
    "src/v2/execution/moe/ExpertTierWeightStream.cpp": "host reference transcoding; seals before returning",
    "src/v2/execution/moe/CpuExpertSlotPool.cpp": "slot storage; acquisition opens a generation, arrival admits",
    "src/v2/kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h": "overlay slot rewrite/admission API",
}

MUTATION = re.compile(
    r"\b(?:beginRewrite|retainedSlotRoot|unsealedGenerationBytes)\s*\(|"
    r"\bnative_interleaved\s*\.\s*(?:replace|resize|resize_uninitialized|resize_uninitialized_aligned|reserve|release)\s*\(")
CONST_CAST = re.compile(
    r"\bconst_cast\s*<[^>]*>\s*\(\s*[^;]*?(?:native_interleaved\s*\.\s*data\s*\(|interleavedBase\s*\()")


def scan_source(path: str, source: str) -> list[str]:
    """Return policy violations for one source file's executable text."""
    code = strip_comments_and_literals(source)
    violations = []
    for match in CONST_CAST.finditer(code):
        line = code.count("\n", 0, match.start()) + 1
        violations.append(f"{path}:{line}: casting const away from published NativeQuantised bytes bypasses their generation")
    if path not in ALLOWED_WRITERS:
        for match in MUTATION.finditer(code):
            line = code.count("\n", 0, match.start()) + 1
            violations.append(f"{path}:{line}: published NativeQuantised bytes may only be mutated by a named producer")
    return violations


def scan(root: Path) -> list[str]:
    """Scan every production source below src/v2."""
    violations = []
    for file in sorted((root / "src" / "v2").rglob("*")):
        if file.suffix in SUFFIXES and file.is_file():
            relative = file.relative_to(root).as_posix()
            violations.extend(scan_source(relative, file.read_text(encoding="utf-8", errors="replace")))
    for allowed in ALLOWED_WRITERS:
        if not (root / allowed).is_file():
            violations.append(f"{allowed}: allowlisted NativeQuantised writer no longer exists; remove its entry")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    violations = scan(args.repo_root.resolve())
    for violation in violations:
        print(violation, file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
