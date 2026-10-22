"""Shared balanced-order authority for captured geometry diagnostic timings.

The C++ trainer and this device-free validator consume the same small inventory.
Operations per sample describe measurement, never a production dispatch key.
"""
from __future__ import annotations

import math
import re
import statistics
from functools import lru_cache
from pathlib import Path

PAIRED_GEOMETRY_GENERATION = "rocm-native-q8-fixed-arithmetic-geometry-v2"
GEOMETRY_ORDER_PROTOCOL = "balanced-rotation-paired-v1"


@lru_cache(maxsize=1)
def geometry_timing_orders() -> tuple[tuple[int, ...], ...]:
    """Read the common inventory and reject incomplete or biased schedules."""
    source = Path(__file__).with_name("NativeVNNIGeometryTimingOrder.def").read_text()
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    rows = []
    for line in source.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"\s*LLAMINAR_GEOMETRY_TIMING_ORDER\(([0-4]),\s*([0-4]),\s*([0-4]),\s*([0-4]),\s*([0-4])\)\s*", line)
        if not match:
            raise ValueError("malformed common geometry timing inventory")
        rows.append(tuple(map(int, match.groups())))
    if len(rows) != 10 or any(sorted(row) != list(range(5)) for row in rows):
        raise ValueError("geometry timing inventory is not ten complete permutations")
    if any(sum(row[position] == candidate for row in rows) != 2
           for candidate in range(5) for position in range(5)):
        raise ValueError("geometry timing inventory biases candidate positions")
    if any(sum(row.index(a) < row.index(b) for row in rows) != 5
           for a in range(5) for b in range(a + 1, 5)):
        raise ValueError("geometry timing inventory biases candidate pairs")
    return tuple(rows)


def geometry_timing_position(round_index: int, candidate_index: int) -> int:
    """Resolve one physical position through the shared ordering authority."""
    if type(round_index) is not int or round_index < 0 or type(candidate_index) is not int or not 0 <= candidate_index < 5:
        raise ValueError("invalid geometry timing coordinate")
    rows = geometry_timing_orders()
    return rows[round_index % len(rows)].index(candidate_index)


def paired_geometry_summary(candidate: dict, baseline: dict) -> dict:
    """Use every matched round; preserve both total and normalized intervals."""
    if any(candidate[name] != baseline[name] for name in ("protocol", "operations_per_sample")):
        raise ValueError("paired timings require one measurement regime")
    a, b = candidate["rounds"], baseline["rounds"]
    if len(a) != len(b) or not a or any(x["round_index"] != y["round_index"] for x, y in zip(a, b)):
        raise ValueError("paired timings require complete matching rounds")
    ratios = [math.log(y["latency_us"] / x["latency_us"]) for x, y in zip(a, b)]
    return {"paired_rounds": len(ratios), "paired_median_speedup": math.exp(statistics.median(ratios)),
            "paired_geomean_speedup": math.exp(statistics.fmean(ratios)),
            "operations_per_sample": candidate["operations_per_sample"],
            "median_total_interval_us": statistics.median(x["interval_us"] for x in a)}
