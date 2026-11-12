#!/usr/bin/env python3
"""Keep native preparation/callback authority inside exact PMA and SDK owners.

Immutable geometry/BOM values may enter planning. Opening native preparation or
issuing physical root claims is restricted to the typed native infrastructure;
ordinary engine allocation must continue to consume its admitted source owner.
Test-only router bootstrap may declare/define its isolated infrastructure entry,
but no production caller may manufacture synthetic rank/device membership.
"""
from pathlib import Path
import argparse
import re
import sys
from check_cpu_prepared_allocation_policy import executable_source
CORE = {"src/v2/planning/PhysicalMemoryAuthority.h", "src/v2/planning/PhysicalMemoryAuthority.cpp"}
BINDINGS = {"src/v2/collective/NativeStoragePmaBinding.h", "src/v2/collective/NativeStoragePreparationPmaBinding.h"}
OWNERS = {"src/v2/collective/LocalTPContext.h", "src/v2/collective/LocalTPContext.cpp",
          "src/v2/collective/backends/NCCLBackend.h", "src/v2/collective/backends/NCCLBackend.cpp",
          "src/v2/collective/backends/RCCLBackend.h", "src/v2/collective/backends/RCCLBackend.cpp",
          "src/v2/collective/coordinators/NCCLCoordinator.cu", "src/v2/collective/coordinators/RCCLCoordinator.cpp"}
# These two original worker owners consume the same Preparing PMA callbacks.
# They may retain bindings, but cannot open raw epochs or mint root permits.
OWNERS |= {"src/v2/backends/NativeContextPreparation.h", "src/v2/backends/NativeContextPreparation.cpp",
           "src/v2/backends/PreparedGPUWorkerThread.h", "src/v2/backends/PreparedGPUWorkerThread.cpp"}
# These physical-lifetime owners retain the SAME original source binding only.
# They cannot construct a binding, carry another binding kind, or open an epoch.
PROCESS_RETENTION_OWNERS = {
    "src/v2/backends/GPUDeviceProcessLifetime.h", "src/v2/backends/GPUDeviceProcessLifetime.cpp",
    "src/v2/backends/NativeContextRetirement.h", "src/v2/backends/NativeContextRetirement.cpp",
}
RETAINED_SOURCE = re.compile(
    r"\b(?:class\s+|std\s*::\s*shared_ptr\s*<\s*)(NativeStoragePreparationPmaBinding)(?=\s*[;>])")
REPLAY_DECLARATIONS = {
    "RetainedHostServiceReplayAccess": {
        "src/v2/execution/local_execution/graph/RetainedHostServiceProgram.h",
        "src/v2/execution/local_execution/graph/RetainedHostServiceDeclarations.h"},
    "RetainedHostServiceCalibrationAccess": {
        "src/v2/execution/local_execution/graph/RetainedHostServiceProgram.h"},
    "NativeOrderingEventGroupReplayAccess": {
        "src/v2/backends/NativeOrderingEventGroup.h", "src/v2/backends/NativeOrderingEventBudgetProjection.h", "src/v2/backends/NativeAuxiliaryResourceCatalog.h"},
    "NativeContextSourceLifecycleDiagnosticAccess": {"src/v2/backends/cuda/NvidiaDeviceContext.h", "src/v2/backends/rocm/AMDDeviceContext.h"},
    "NativeContextPreparationReplayAccess": {
        "src/v2/planning/PhysicalMemoryAuthority.h", "src/v2/backends/NativeContextOpaqueContract.h",
        "src/v2/backends/NativeContextPreparation.h", "src/v2/backends/PreparedGPUWorkerThread.h",
        "src/v2/backends/GPUDeviceContextPool.h",
        "src/v2/backends/RetainedParticipantGraphExecutable.h"},
    "NativeContextBirthDiagnosticAccess": {
        "src/v2/collective/NativeStoragePreparationPmaBinding.h",
        "src/v2/backends/NativeContextPreparation.h", "src/v2/backends/PreparedGPUWorkerThread.h"},
    "PreparedGPUWorkerThreadReplayAccess": {"src/v2/backends/PreparedGPUWorkerThread.h"},
    "NativeContextRetirementReplayAccess": {"src/v2/backends/GPUDeviceContextPool.h"},
    "ReadAheadNativeResourceReplayAccess": {
        "src/v2/backends/ReadAheadNativeResourceFamily.h",
        "src/v2/backends/NativeContextRetirement.h",
        "src/v2/backends/ReadAheadNativeResourceObligation.h"},
    "ReadAheadNativeParentTerminalReplayAccess": {"src/v2/backends/ReadAheadNativeParentTerminal.h"},
}
REPLAY = re.compile(r"\b(?:" + "|".join(REPLAY_DECLARATIONS) + r")\b")
ENTRY = re.compile(r"\b(?:beginPreparationEpoch|claimPreparation|prepareNativeAdmission|publishNativeAdmission|claimPreparedNative|PhysicalMemoryNativeSetupEpoch|PreparingNativeStorageTag|NativeStoragePreparationReplayAccess)\b")
BIND = re.compile(r"\b(?:NativeStoragePmaBinding|NativeStoragePreparationPmaBinding|NativeStorageSessionPreparation)\b")
SUFFIXES = {".h", ".hpp", ".cpp", ".cc", ".cu", ".cuh", ".hip", ".inc"}
# Match the method token itself so aliases, using declarations and function
# pointers cannot evade the production-call prohibition. Only these exact
# infrastructure signatures are legal; their surrounding files are not exempt.
TEST_ROUTER_INITIALIZATION = re.compile(r"\binitForTests\b")
TEST_ROUTER_INFRASTRUCTURE = {
    "src/v2/collective/BackendRouter.h": re.compile(
        r"\bstatic\s+bool\s+(initForTests)\s*\(\s*\)\s*;"),
    "src/v2/collective/BackendRouter.cpp": re.compile(
        r"\bbool\s+GlobalBackendRouter\s*::\s*(initForTests)\s*\(\s*\)\s*\{"),
}

def scan_source(path: str, source: str) -> list[str]:
    """Reject source-level bypasses without confusing literals/comments for code."""
    if not ENTRY.search(source) and not BIND.search(source) and not REPLAY.search(source) and not TEST_ROUTER_INITIALIZATION.search(source): return []
    code = executable_source(source)
    failures = []
    infrastructure = TEST_ROUTER_INFRASTRUCTURE.get(path)
    declarations = {match.span(1) for match in infrastructure.finditer(code)} if infrastructure else set()
    for match in TEST_ROUTER_INITIALIZATION.finditer(code):
        if match.span() not in declarations:
            failures.append(f"{path}:{code.count(chr(10),0,match.start())+1}: test-only router initialization cannot construct production rank/device membership")
    if path not in CORE | BINDINGS:
        failures += [f"{path}:{code.count(chr(10),0,m.start())+1}: native root/epoch authority is PMA-infrastructure-only" for m in ENTRY.finditer(code)]
    if path not in CORE | BINDINGS | OWNERS:
        retained = {item.span(1) for item in RETAINED_SOURCE.finditer(code)} if path in PROCESS_RETENTION_OWNERS else set()
        failures += [f"{path}:{code.count(chr(10),0,m.start())+1}: native binding belongs to its exact SDK/session owner"
                     for m in BIND.finditer(code) if m.span() not in retained]
    # Only named forward/friend declarations are production code. Definitions,
    # aliases and calls through these testing privileges must stay in tests,
    # including inside otherwise trusted authority headers.
    for match in REPLAY.finditer(code):
        name = match.group()
        declaration = re.compile(r"\b(?:friend\s+)?struct\s+(?:test\s*::\s*)?(" + name + r")\s*;")
        declared = path in REPLAY_DECLARATIONS[name] and any(
            item.span(1) == match.span() for item in declaration.finditer(code))
        if not declared:
            failures.append(f"{path}:{code.count(chr(10),0,match.start())+1}: native replay/diagnostic access is test-only")
    return failures

def main() -> int:
    """Scan production source only; tests exercise the private replay friend explicitly."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    root=parser.parse_args().root.resolve()
    if not (root/"src/v2").is_dir():raise ValueError("Missing production source authority")
    failures=[]
    for path in (root/"src/v2").rglob("*"):
        if path.is_file() and path.suffix in SUFFIXES:failures+=scan_source(path.relative_to(root).as_posix(),path.read_text())
    for failure in failures:print(failure,file=sys.stderr)
    if not failures:print("PASS exact native preparation authority source policy")
    return bool(failures)
if __name__=="__main__":raise SystemExit(main())
