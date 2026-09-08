#!/usr/bin/env python3
"""Attribute explicit Nsight windows without double-counting concurrent kernels.

The caller owns phase boundaries and semantic classification. This helper reads
an existing SQLite export, clips physical kernel intervals to those windows,
and preserves geometry in the CSV. It does not infer a model, request boundary,
critical path, or time spent in DMA from an absence of kernel activity.
"""

import argparse
import bisect
from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import json
from pathlib import Path
import re
import sqlite3


UNKNOWN = "UNCLASSIFIED"
OVERLAP = "Mixed-category overlap"
NO_KERNEL = "No kernel active"
GEOMETRY = ("gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ")


@dataclass(frozen=True)
class Kernel:
    """One physical activity; timestamps are Nsight nanoseconds, not CPU time."""

    start: int
    end: int
    name: str
    geometry: tuple
    device: int


def compile_rules(records):
    """Validate ordered semantic rules; first matching expression wins."""
    if not isinstance(records, list) or not records:
        raise ValueError("rules must be a nonempty JSON list")
    rules = []
    for record in records:
        category, pattern = record["category"], record["pattern"]
        if not isinstance(category, str) or not category.strip():
            raise ValueError("every rule requires a nonempty category")
        if category in (UNKNOWN, OVERLAP, NO_KERNEL):
            raise ValueError("rule category collides with an accounting row")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("every rule requires a nonempty pattern")
        rules.append((category, re.compile(pattern, re.IGNORECASE)))
    return rules


def validate_windows(records):
    """Reject empty, inverted or overlapping samples of the same phase.

    Different phase views may intentionally describe the same work (for example
    the whole decode cycle and its captured-graph subset). They are never summed
    together by this script.
    """
    if not isinstance(records, list) or not records:
        raise ValueError("windows must be a nonempty JSON list")
    phases = defaultdict(list)
    for row in records:
        phase, start, end = row["phase"], row["start_ns"], row["end_ns"]
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError("every window requires a nonempty phase")
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            raise ValueError("window bounds must be increasing integer nanoseconds")
        phases[phase].append((start, end))
    for bounds in phases.values():
        bounds.sort()
        if any(a[1] > b[0] for a, b in zip(bounds, bounds[1:])):
            raise ValueError("samples within one phase must not overlap")
    return phases


def read_kernels(path):
    """Load real kernel activities, failing on metadata-only or unknown schemas."""
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as db:
        fields = ",".join("k." + field for field in GEOMETRY)
        rows = db.execute(
            "SELECT k.start,k.end,s.value," + fields + ",k.deviceId "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL AS k "
            "JOIN StringIds AS s ON s.id=k.demangledName ORDER BY k.start"
        ).fetchall()
    if not rows:
        raise ValueError("trace has no executed kernel activities")
    kernels = [Kernel(r[0], r[1], r[2], tuple(r[3:9]), r[9]) for r in rows]
    if any(k.end <= k.start for k in kernels):
        raise ValueError("trace contains a nonpositive kernel interval")
    return kernels


def classify(name, rules):
    """Keep unmatched physical work visible instead of dropping its duration."""
    return next((label for label, pattern in rules if pattern.search(name)), UNKNOWN)


def measure(kernels, start, end, labels):
    """Partition a window into exclusive categories, overlap, and no-kernel time.

    Per-category active counts handle simultaneous kernels from the same family.
    All events at one timestamp take effect together, so an end/start tie does
    not invent an overlap or a gap. Raw sums remain a separate work metric.
    """
    events = defaultdict(Counter)
    summed, union, exclusive, calls = (Counter() for _ in range(4))
    geometry_time, geometry_calls = Counter(), Counter()
    for kernel in kernels:
        a, b = max(start, kernel.start), min(end, kernel.end)
        if a >= b:
            continue
        label = labels[kernel.name]
        events[a][label] += 1
        events[b][label] -= 1
        summed[label] += b - a
        calls[label] += 1
        key = (kernel.name, kernel.geometry, kernel.device)
        geometry_time[key] += b - a
        geometry_calls[key] += 1
    active = Counter()
    previous = start
    for point in sorted(set(events) | {start, end}):
        duration = point - previous
        live = [label for label, count in active.items() if count > 0]
        for label in live:
            union[label] += duration
        exclusive[live[0] if len(live) == 1 else OVERLAP if live else NO_KERNEL] += duration
        active.update(events[point])
        if any(count < 0 for count in active.values()):
            raise ValueError("unbalanced kernel start/end events")
        previous = point
    if any(active.values()) or sum(exclusive.values()) != end - start:
        raise ValueError("interval partition does not conserve elapsed time")
    return summed, union, exclusive, calls, geometry_time, geometry_calls


def analyze(kernels, phases, rules, device_id=None):
    """Aggregate sample means while retaining every selected kernel's geometry."""
    if device_id is not None:
        kernels = [k for k in kernels if k.device == device_id]
    if not kernels:
        raise ValueError("no activities for the selected device")
    kernels.sort(key=lambda k: k.start)
    starts = [k.start for k in kernels]
    prefix_ends = []
    maximum_end = 0
    for k in kernels:
        maximum_end = max(maximum_end, k.end)
        prefix_ends.append(maximum_end)
    # A graph replay repeats the same few symbols thousands of times. Classify
    # unique names once; regex work must not scale with the generation horizon.
    labels = {name: classify(name, rules) for name in {k.name for k in kernels}}
    categories, physical, summaries = [], [], {}
    for phase, bounds in phases.items():
        totals = [Counter() for _ in range(6)]
        selected_devices = set()
        for a, b in bounds:
            # Prefix maxima include long-running kernels crossing the left edge;
            # no guessed maximum duration or correlation-ID filter is needed.
            lo, hi = bisect.bisect_right(prefix_ends, a), bisect.bisect_left(starts, b)
            work = [k for k in kernels[lo:hi] if k.end > a]
            selected_devices.update(k.device for k in work)
            for total, sample in zip(totals, measure(work, a, b, labels)):
                total.update(sample)
        if len(selected_devices) > 1:
            raise ValueError("multiple GPUs in selected windows: specify --device-id")
        count, denom = len(bounds), len(bounds) * 1e6
        for label in sorted(set().union(*(set(t) for t in totals[:4]))):
            categories.append(dict(
                phase=phase, category=label, samples=count,
                summed_ms=totals[0][label] / denom, union_ms=totals[1][label] / denom,
                exclusive_ms=totals[2][label] / denom, calls_per_sample=totals[3][label] / count))
        for key, duration in sorted(totals[4].items(), key=lambda row: -row[1]):
            name, geometry, device = key
            physical.append(dict(phase=phase, category=labels[name], kernel=name,
                                 device=device, **dict(zip(GEOMETRY, geometry)),
                                 summed_ms=duration / denom,
                                 calls_per_sample=totals[5][key] / count))
        summaries[phase] = dict(samples=count, mean_window_ms=sum(b-a for a, b in bounds)/denom,
                                devices=sorted(selected_devices))
    return summaries, categories, physical


def main(argv=None):
    """Write diagnostic artifacts even when classification is incomplete."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--rules", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--device-id", type=int)
    args = parser.parse_args(argv)
    try:
        windows = json.loads(args.windows.read_text())
        rule_records = json.loads(args.rules.read_text())
        summaries, categories, physical = analyze(
            read_kernels(args.sqlite), validate_windows(windows),
            compile_rules(rule_records), args.device_id)
    except (OSError, ValueError, KeyError, TypeError, re.error, sqlite3.Error) as error:
        parser.error(str(error))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, rows in (("categories", categories), ("kernels", physical)):
        with Path(str(args.output_prefix) + "-" + suffix + ".csv").open("w", newline="") as output:
            if rows:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    unknown = sorted({r["kernel"] for r in physical if r["category"] == UNKNOWN})
    report = dict(trace=str(args.sqlite.resolve()), windows=windows, rules=rule_records,
                  device_id=args.device_id, phases=summaries, unclassified=unknown,
                  timing_scope="profiled kernel intervals; not benchmark latency or critical path")
    Path(str(args.output_prefix) + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["phases"], indent=2))
    if unknown:
        print(f"Unclassified kernel names: {len(unknown)}; inspect the kernels CSV")
    return 1 if unknown else 0


if __name__ == "__main__":
    raise SystemExit(main())
