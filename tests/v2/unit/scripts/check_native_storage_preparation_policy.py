#!/usr/bin/env python3
"""Keep native preparation/callback authority inside exact PMA and SDK owners.

Immutable geometry/BOM values may enter planning. Opening native preparation or
issuing physical root claims is restricted to the typed native infrastructure;
ordinary engine allocation must continue to consume its admitted source owner.
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
ENTRY = re.compile(r"\b(?:beginPreparationEpoch|claimPreparation|prepareNativeAdmission|publishNativeAdmission|claimPreparedNative|PhysicalMemoryNativeSetupEpoch|PreparingNativeStorageTag|NativeStoragePreparationReplayAccess)\b")
BIND = re.compile(r"\b(?:NativeStoragePmaBinding|NativeStoragePreparationPmaBinding|NativeStorageSessionPreparation)\b")
SUFFIXES = {".h", ".hpp", ".cpp", ".cc", ".cu", ".cuh", ".hip", ".inc"}

def scan_source(path: str, source: str) -> list[str]:
    """Reject source-level bypasses without confusing literals/comments for code."""
    if not ENTRY.search(source) and not BIND.search(source): return []
    code = executable_source(source)
    failures = []
    if path not in CORE | BINDINGS:
        failures += [f"{path}:{code.count(chr(10),0,m.start())+1}: native root/epoch authority is PMA-infrastructure-only" for m in ENTRY.finditer(code)]
    if path not in CORE | BINDINGS | OWNERS:
        failures += [f"{path}:{code.count(chr(10),0,m.start())+1}: native binding belongs to its exact SDK/session owner" for m in BIND.finditer(code)]
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
