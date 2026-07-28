#!/usr/bin/env python3
"""Enforce the explicit, stream-ordered KV-state lifecycle contract.

KV payload allocation is model-lifetime state. Request, sequence, and layer
resets only make old rows unreachable; they must not free storage, scrub whole
payloads, select an implicit GPU stream, synchronize the host, or transfer
canonical GPU metadata through host memory.

This source sanitizer keeps that architecture structural. It also verifies
that DeviceGraphOrchestrator expresses request reset as one ordered
join -> reset -> event-publication transaction.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

from check_gpu_blocking_sync_policy import strip_comments_and_literals


RESET_METHODS = (
    "resetRequestState",
    "resetSequenceState",
    "resetLayerSequenceState",
    "resetLayerState",
)

RESET_SCOPE_GUARDS = {
    "resetRequestState": "context.permitsRequestReset()",
    "resetSequenceState": "context.permitsSequenceReset()",
    "resetLayerSequenceState": "context.permitsLayerSequenceReset()",
    "resetLayerState": "context.permitsLayerReset()",
}

GPU_RESET_SOURCES = (
    (
        "CUDA",
        "src/v2/kernels/cuda/kvcache/CUDARingKVCacheBase.cpp",
        "CUDARingKVCacheBase",
    ),
    (
        "ROCm",
        "src/v2/kernels/rocm/kvcache/ROCmRingKVCacheBase.cpp",
        "ROCmRingKVCacheBase",
    ),
)

FORBIDDEN_RESET_OPERATIONS = (
    "StreamSynchronize(",
    "DeviceSynchronize(",
    "EventSynchronize(",
    "getEffectiveStream(",
    "getDefaultStream(",
    "cudaMalloc(",
    "cudaMallocAsync(",
    "hipMalloc(",
    "hipMallocAsync(",
    "cudaFree(",
    "cudaFreeAsync(",
    "hipFree(",
    "hipFreeAsync(",
    "deviceToHost(",
    "hostToDevice(",
)


def extract_function(source: str, qualified_name: str) -> str:
    """Return one C++ function body, including braces, or an empty string."""

    executable = strip_comments_and_literals(source)
    name_pos = executable.find(qualified_name)
    if name_pos < 0:
        return ""
    open_brace = executable.find("{", name_pos + len(qualified_name))
    if open_brace < 0:
        return ""

    depth = 0
    for index in range(open_brace, len(executable)):
        char = executable[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return executable[open_brace : index + 1]
    return ""


def validate_interface(source: str) -> list[str]:
    """Validate the IKVCache ownership and reset vocabulary."""

    executable = strip_comments_and_literals(source)
    failures: list[str] = []
    for required in (
        "enum class StateRole",
        "struct StateOwnership",
        "enum class StateResetBoundary",
        "struct StateResetContext",
        "void *execution_stream",
        "const char *reason",
        "bindStateOwnership",
    ):
        if required not in executable:
            failures.append(f"IKVCache is missing required lifecycle API: {required}")

    for method in RESET_METHODS:
        pattern = re.compile(
            rf"\bvirtual\s+bool\s+{method}\s*\([^;]*\)\s*=\s*0\s*;",
            re.DOTALL,
        )
        if not pattern.search(executable):
            failures.append(f"IKVCache reset method is not a pure contract: {method}")

    for obsolete in (
        r"\bclear_sequence\s*\(",
        r"\bclear_layer\s*\(",
        r"\bvirtual\s+(?:bool|void)\s+clear\s*\(",
    ):
        if re.search(obsolete, executable):
            failures.append(f"IKVCache retains obsolete reset API matching {obsolete}")
    return failures


def validate_gpu_reset_source(
    backend: str,
    class_name: str,
    source: str,
) -> list[str]:
    """Validate every canonical GPU reset implementation."""

    failures: list[str] = []
    for method in RESET_METHODS:
        body = extract_function(source, f"{class_name}::{method}")
        if not body:
            failures.append(f"{backend} is missing {class_name}::{method}")
            continue
        if "context.execution_stream" not in body:
            failures.append(f"{backend} {method} does not use the caller's stream")
        if "context.hasReason()" not in body:
            failures.append(f"{backend} {method} does not reject an unnamed reset")
        if RESET_SCOPE_GUARDS[method] not in body:
            failures.append(
                f"{backend} {method} does not enforce its semantic reset scope"
            )
        invalidates_derived_views = (
            "onResetLayerSequenceState(" in body or
            (method == "resetSequenceState" and "truncateSequence(" in body)
        )
        if not invalidates_derived_views:
            failures.append(
                f"{backend} {method} does not invalidate cache-owned derived views"
            )
        for forbidden in FORBIDDEN_RESET_OPERATIONS:
            if forbidden in body:
                failures.append(
                    f"{backend} {method} contains forbidden hot-path operation "
                    f"{forbidden}"
                )
    return failures


def validate_orchestrator(header: str, source: str) -> list[str]:
    """Validate the one-stream request reset transaction and retired alias."""

    executable_header = strip_comments_and_literals(header)
    executable_source = strip_comments_and_literals(source)
    failures: list[str] = []

    if re.search(r"\bclearInferenceState\s*\(", executable_header + executable_source):
        failures.append("DeviceGraphOrchestrator retains obsolete clearInferenceState alias")

    body = extract_function(
        executable_header,
        "resetInferenceState(const InferenceStateResetRequest &request)",
    )
    if not body:
        return failures + ["DeviceGraphOrchestrator resetInferenceState body is missing"]

    ordered_needles = (
        "RequestStateResetTransaction reset_transaction",
        "joinPriorDeviceWorkForRequestStateReset(",
        "state_.resetCommittedKVAndRecurrentState(",
        "state_.resetMTPShiftedSidecarState(",
        "publishRequestStateResetReady(",
        "reset_transaction.markPublished()",
    )
    positions: list[int] = []
    for needle in ordered_needles:
        position = body.find(needle)
        if position < 0:
            failures.append(f"request reset transaction is missing: {needle}")
        positions.append(position)
    if all(position >= 0 for position in positions) and positions != sorted(positions):
        failures.append("request reset transaction is not ordered join -> reset -> publish")

    if body.count("reset_transaction.executionStream()") < 2:
        failures.append(
            "request reset must use the same typed stream for producer join and publication"
        )
    if body.count("reset_transaction.cacheContext()") < 2:
        failures.append(
            "committed and shifted-MTP state must consume the same reset context"
        )
    return failures


def validate_obsolete_symbols(repo_root: pathlib.Path) -> list[str]:
    """Reject retired cache lifecycle names anywhere in production source."""

    failures: list[str] = []
    patterns = (
        re.compile(r"\bclear_sequence\s*\("),
        re.compile(r"\bclear_layer\s*\("),
    )
    for path in (repo_root / "src" / "v2").rglob("*"):
        if path.suffix not in {".cpp", ".cu", ".cuh", ".h", ".hip", ".hpp"}:
            continue
        executable = strip_comments_and_literals(path.read_text(encoding="utf-8"))
        for pattern in patterns:
            match = pattern.search(executable)
            if match:
                line = executable.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{path.relative_to(repo_root)}:{line} retains obsolete "
                    f"KV lifecycle symbol {match.group(0).rstrip('(')}"
                )
    return failures


def validate(repo_root: pathlib.Path) -> list[str]:
    """Run the repository KV-state lifecycle policy."""

    interface_path = repo_root / "src/v2/kernels/IKVCache.h"
    orchestrator_header_path = (
        repo_root
        / "src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
    )
    orchestrator_source_path = orchestrator_header_path.with_suffix(".cpp")

    failures = validate_interface(interface_path.read_text(encoding="utf-8"))
    for backend, relative_path, class_name in GPU_RESET_SOURCES:
        source = (repo_root / relative_path).read_text(encoding="utf-8")
        failures.extend(validate_gpu_reset_source(backend, class_name, source))
    failures.extend(
        validate_orchestrator(
            orchestrator_header_path.read_text(encoding="utf-8"),
            orchestrator_source_path.read_text(encoding="utf-8"),
        )
    )
    failures.extend(validate_obsolete_symbols(repo_root))
    return failures


def parse_args() -> argparse.Namespace:
    """Parse the repository root used by CTest and local runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[4],
    )
    return parser.parse_args()


def main() -> int:
    """Run the sanitizer and report every structural violation."""

    failures = validate(parse_args().repo_root.resolve())
    if failures:
        print("KV-state lifecycle policy violations:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("KV-state lifecycle policy: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
