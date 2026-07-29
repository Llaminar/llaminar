#!/usr/bin/env python3
"""Enforce one visible ownership boundary for backend GEMM/GEMV code.

Production matrix-multiplication kernels, dispatch tables, packed-weight
representations, and weight packers belong below:

    src/v2/kernels/<backend>/gemm/

Keeping CPU, CUDA, and ROCm symmetric makes the complete implementation surface
discoverable without knowing historical backend-specific directory names.  In
particular, this policy forbids restoring the old CPU ``native_vnni`` sibling
tree or placing a backend weight packer at the backend root.

The check is intentionally filename based.  Adjacent facilities such as
activation rotation, generic tensor repacking, MoE routing, and KV-cache code
retain their own ownership directories even when they feed a GEMM.

The same ownership boundary also rejects retired launcher controls.  These
symbols used to remain as no-op ABI stubs after their alternate implementations
were deleted; retaining them made tests and future callers believe a runtime
choice still existed.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import tempfile


BACKENDS = ("cpu", "cuda", "rocm")
PRODUCTION_SUFFIXES = {
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hip",
    ".hpp",
    ".inc",
    ".json",
}
GEMM_OWNED_NAME = re.compile(
    r"(?:gemm|gemv|packedweights|weightpacker)",
    re.IGNORECASE,
)
RETIRED_ENTRYPOINTS = (
    "setNativeVNNIEnabled",
    "isNativeVNNIEnabled",
    "setForceCutlassFallback",
    "isForceCutlassFallback",
    "cudaNativeVNNIPrefill_freeStreamKFixup",
)


def find_violations(repo_root: pathlib.Path) -> list[str]:
    """Return stable diagnostics for backend GEMM ownership violations."""

    kernels_root = repo_root / "src" / "v2" / "kernels"
    violations: list[str] = []

    for backend in BACKENDS:
        backend_root = kernels_root / backend
        if not backend_root.is_dir():
            continue

        legacy_root = backend_root / "native_vnni"
        if legacy_root.exists():
            violations.append(
                f"{legacy_root.relative_to(repo_root)}: legacy NativeVNNI "
                "directory is forbidden; move its production files to gemm/"
            )

        canonical_root = backend_root / "gemm"
        for path in backend_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in PRODUCTION_SUFFIXES:
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
            for entrypoint in RETIRED_ENTRYPOINTS:
                if entrypoint in source:
                    violations.append(
                        f"{path.relative_to(repo_root)}: retired GEMM "
                        f"entrypoint {entrypoint} is forbidden"
                    )
            if not GEMM_OWNED_NAME.search(path.name):
                continue
            if canonical_root not in path.parents:
                violations.append(
                    f"{path.relative_to(repo_root)}: backend GEMM/GEMV and "
                    "packed-weight files must live below "
                    f"src/v2/kernels/{backend}/gemm/"
                )

    return sorted(violations)


def run_self_test() -> int:
    """Prove that each forbidden layout category is detected."""

    with tempfile.TemporaryDirectory() as temp_dir:
        root = pathlib.Path(temp_dir)
        good = root / "src/v2/kernels/cuda/gemm/CUDAGemmKernel.cu"
        good.parent.mkdir(parents=True)
        good.write_text("// canonical fixture\n", encoding="utf-8")
        if find_violations(root):
            print("canonical GEMM fixture was rejected", file=sys.stderr)
            return 1

        bad_files = (
            root / "src/v2/kernels/cpu/CPUGemvKernel.h",
            root / "src/v2/kernels/cuda/CUDAPackedWeights.h",
            root / "src/v2/kernels/rocm/ROCmWeightPacker.cpp",
        )
        for bad_file in bad_files:
            bad_file.parent.mkdir(parents=True, exist_ok=True)
            bad_file.write_text("// misplaced fixture\n", encoding="utf-8")

        legacy_file = (
            root
            / "src/v2/kernels/cpu/native_vnni/CPUNativeVNNIDecode.h"
        )
        legacy_file.parent.mkdir(parents=True)
        legacy_file.write_text("// legacy fixture\n", encoding="utf-8")

        retired_entrypoint = (
            root / "src/v2/kernels/cuda/gemm/CUDAObsoleteGemmControl.cpp"
        )
        retired_entrypoint.write_text(
            "void setForceCutlassFallback(bool) {}\n",
            encoding="utf-8",
        )

        violations = find_violations(root)
        if len(violations) != 5:
            print(
                f"expected five ownership violations, found {len(violations)}",
                file=sys.stderr,
            )
            for violation in violations:
                print(violation, file=sys.stderr)
            return 1

    return 0


def main() -> int:
    """Parse CLI options and validate either fixtures or the real source tree."""

    parser = argparse.ArgumentParser(
        description="Enforce canonical backend GEMM/GEMV source ownership."
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[4],
        help="Repository root containing src/v2/kernels.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run isolated positive and negative policy fixtures.",
    )
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    violations = find_violations(args.repo_root.resolve())
    if violations:
        print("Backend GEMM/GEMV layout policy violations:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1

    print("Backend GEMM/GEMV layout policy: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
