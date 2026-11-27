#!/usr/bin/env python3
"""Compare the compiled constexpr row routing with the register checker's exact leaf inventory.

This contract rejects parser drift and proves each replacement, and no retired
wide key, belongs to the dispatcher's admitted symbol union. Native numerical
and production-TU stack gates remain separate mandatory qualifications.
"""
import importlib.util
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location(
    "tile_evidence", root / "tests/v2/cmake/CPUNativeTileRegisterEvidence.py")
evidence = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evidence
spec.loader.exec_module(evidence)
routing = evidence.row_decomposition_keys(root)
output = subprocess.run(
    [sys.argv[1], "--gtest_filter=CPUNativeNEONPreparedProjection.DumpRowDecompositionKeys"],
    check=True, capture_output=True, text=True).stdout
compiled = {}
for source, pair, tail in re.findall(r"NEON_ROW_ROUTE ([0-9:]+) ([0-9]+) ([0-9]+)", output):
    key = tuple(map(int, source.split(":"))) + (0,)
    suffix = key[2:]
    assert key not in compiled, f"duplicate compiled route {key}"
    compiled[key] = ((2, int(pair), *suffix), (key[0] - 2, int(tail), *suffix))
assert compiled == routing, "compiled routing differs from checker source-derived routing"
for source, leaves in routing.items():
    admitted = evidence.dispatched_tile_keys(*source[:2])
    assert source not in admitted, f"retired wide program still admitted: {source}"
    assert set(leaves) <= admitted, f"replacement absent from production union: {source}"
print(f"PASS: {len(routing)} exact compiled routes match actual-dispatch leaf unions")
