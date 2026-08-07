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
    "cudaNativeVNNIPrefill_setStreamKMode",
    "cudaNativeVNNIPrefill_getStreamKMode",
    "cudaNativeVNNIPrefill_setDeterministicMode",
    "cudaNativeVNNIPrefill_getDeterministicMode",
    "CUDA_NATIVE_VNNI_PREFILL_SPLITK_PARTIALS",
    "CUDA_NATIVE_VNNI_PREFILL_STREAMK_FIXUP",
    "LLAMINAR_FORCE_PREFILL_SPLIT_K",
    "LLAMINAR_STREAM_K",
    "LLAMINAR_ROCM_NVNNI_ATOMIC_REDUCE",
    "nvnni_atomic_reduce",
)
NATIVE_VNNI_PREFILL_FORBIDDEN_REDUCTIONS = (
    "atomicAdd(",
    "split_k",
    "split-K",
    "SPLIT_K",
    "Stream-K",
    "stream-K",
    "STREAM_K",
)
ROCM_NATIVE_VNNI_M1_FORBIDDEN_REDUCTIONS = (
    "use_atomic_reduce",
    "atomicAdd(&d_C_fp32[n], result)",
)
CUDA_NATIVE_VNNI_GEMV_FORBIDDEN_REDUCTIONS = (
    "atomicAdd(",
    "nativeVnniGemv_epilogue",
    "kTwoPhaseMaxBytes",
    "debugEnv().gemm.deterministic",
)
CPU_NATIVE_VNNI_GROUPED_FALLBACK = re.compile(
    r"\bverifier_policy\s*=\s*VerifierRowsPolicy::Pairwise\s*;"
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
            if path.name == "CUDANativeVNNIPrefillKernels.cu":
                for token in NATIVE_VNNI_PREFILL_FORBIDDEN_REDUCTIONS:
                    if token in source:
                        violations.append(
                            f"{path.relative_to(repo_root)}: CUDA NativeVNNI "
                            f"prefill reduction token {token!r} is forbidden; "
                            "retain the public M=1 arithmetic tree"
                        )
            if path.name == "ROCmGemvKernel_native_VNNI.hip":
                for token in ROCM_NATIVE_VNNI_M1_FORBIDDEN_REDUCTIONS:
                    if token in source:
                        violations.append(
                            f"{path.relative_to(repo_root)}: ROCm NativeVNNI "
                            f"serial-decode reduction token {token!r} is "
                            "forbidden; retain the ordered K-partition fold"
                        )
            if path.name == "CUDANativeVNNIGemvShardImpl.cu.inc":
                for token in CUDA_NATIVE_VNNI_GEMV_FORBIDDEN_REDUCTIONS:
                    if token in source:
                        violations.append(
                            f"{path.relative_to(repo_root)}: CUDA NativeVNNI "
                            f"GEMV reduction token {token!r} is forbidden; "
                            "serial and grouped decode must retain one ordered "
                            "K-partition publication contract"
                        )
            if (
                path.name == "CPUNativeVNNIGemv.h"
                and CPU_NATIVE_VNNI_GROUPED_FALLBACK.search(source)
            ):
                violations.append(
                    f"{path.relative_to(repo_root)}: CPU grouped verifier "
                    "policy substitution is forbidden; reject an invalid "
                    "physical policy before launch"
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

        unsafe_prefill = (
            root
            / "src/v2/kernels/cuda/gemm/CUDANativeVNNIPrefillKernels.cu"
        )
        unsafe_prefill.write_text(
            "void bad(float *p) { atomicAdd(p, 1.0f); }\n",
            encoding="utf-8",
        )

        unsafe_rocm_decode = (
            root
            / "src/v2/kernels/rocm/gemm/ROCmGemvKernel_native_VNNI.hip"
        )
        unsafe_rocm_decode.parent.mkdir(parents=True, exist_ok=True)
        unsafe_rocm_decode.write_text(
            "bool use_atomic_reduce = true;\n",
            encoding="utf-8",
        )

        unsafe_cuda_decode = (
            root
            / "src/v2/kernels/cuda/gemm/"
            "CUDANativeVNNIGemvShardImpl.cu.inc"
        )
        unsafe_cuda_decode.parent.mkdir(parents=True, exist_ok=True)
        unsafe_cuda_decode.write_text(
            "void bad(float *p) { atomicAdd(p, 1.0f); }\n",
            encoding="utf-8",
        )

        unsafe_cpu_grouped = (
            root / "src/v2/kernels/cpu/gemm/CPUNativeVNNIGemv.h"
        )
        unsafe_cpu_grouped.parent.mkdir(parents=True, exist_ok=True)
        unsafe_cpu_grouped.write_text(
            "verifier_policy = VerifierRowsPolicy::Pairwise;\n",
            encoding="utf-8",
        )

        violations = find_violations(root)
        if len(violations) != 9:
            print(
                f"expected nine ownership violations, found {len(violations)}",
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
