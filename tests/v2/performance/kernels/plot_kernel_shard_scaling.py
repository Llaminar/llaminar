#!/usr/bin/env python3
"""Plot captured kernel scaling without conflating it with multi-device throughput.

Input is a small, backend-neutral CSV: operation, workload, degree, sample,
latency_us, disposition, and optional positive ``calls`` per workload.
Disposition is ``partitioned``, ``replicated`` or
``collective``. Each curve must contain degrees 1, 2 and 4 on the same device
and binary, with unchanged precision. The collector owns that provenance.
This tool owns validation, median reduction, per-doubling tuning decisions,
and a dependency-free SVG report. Optional occurrence-weighted excess ranking
prioritizes investigations but does not estimate an additive critical path.
It never trains or installs dispatch policy.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from enum import Enum
from html import escape
import json
import math
from pathlib import Path
import statistics
from typing import Iterable


class Disposition(str, Enum):
    """Whether adding participants actually halves this kernel's local work."""

    PARTITIONED = "partitioned"
    REPLICATED = "replicated"
    COLLECTIVE = "collective"


@dataclass(frozen=True)
class ScalingRow:
    """One complete workload curve and its independent per-doubling gates."""

    operation: str
    workload: str
    disposition: Disposition
    one_us: float
    two_us: float
    four_us: float
    one_to_two: float
    two_to_four: float
    tune_two: bool
    tune_four: bool
    sample_counts: tuple[int, int, int]
    calls: int = 1

    def excess_us(self, degree: int) -> float:
        """Lost local time relative to ideal halving, weighted by caller counts.

        This is a prioritization estimate, never an additive critical path:
        independently timed operations may overlap in production. Replication
        and communication have no half-work scaling promise.
        """
        if degree not in (2, 4):
            raise ValueError("ranking degree must be two or four")
        if self.disposition != Disposition.PARTITIONED:
            return 0.0
        before, after = (self.one_us, self.two_us) if degree == 2 else (self.two_us, self.four_us)
        return max(0.0, after - before / 2) * self.calls


def summarize(records: Iterable[dict[str, str]], threshold: float = 1.9) -> list[ScalingRow]:
    """Reject incomplete/ambiguous curves and assess only partitioned compute.

    Medians, not minima, define ratios. A fast individual sample cannot hide a
    slow typical kernel. Replicated work and collectives retain measurements
    but never masquerade as half-work compute scaling failures.
    """
    if not math.isfinite(threshold) or threshold <= 1:
        raise ValueError("scaling threshold must be finite and greater than one")
    groups: dict[tuple[str, str], dict] = {}
    for record in records:
        key = record["operation"], record["workload"]
        if not all(key):
            raise ValueError("every sample requires an operation and workload")
        degree, sample = int(record["degree"]), int(record["sample"])
        latency = float(record["latency_us"])
        disposition = Disposition(record["disposition"])
        calls = int(record.get("calls", "1"))
        if calls <= 0:
            raise ValueError("operation occurrence count must be positive")
        if degree not in (1, 2, 4) or sample < 0 or not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"invalid timing sample for {key}")
        group = groups.setdefault(key, {"disposition": disposition, "calls": calls, "samples": {1: {}, 2: {}, 4: {}}})
        if group["calls"] != calls:
            raise ValueError(f"conflicting occurrence counts for {key}")
        if group["disposition"] != disposition:
            raise ValueError(f"conflicting sharding semantics for {key}")
        values = group["samples"][degree]
        if sample in values:
            raise ValueError(f"duplicate sample identity for {key}, degree {degree}, sample {sample}")
        values[sample] = latency
    if not groups:
        raise ValueError("no kernel samples")
    result = []
    for (operation, workload), group in sorted(groups.items()):
        counts = tuple(len(group["samples"][degree]) for degree in (1, 2, 4))
        if not all(counts):
            raise ValueError(f"incomplete 1/2/4 curve for {(operation, workload)}")
        one, two, four = (statistics.median(group["samples"][degree].values()) for degree in (1, 2, 4))
        first, second = one / two, two / four
        assess = group["disposition"] == Disposition.PARTITIONED
        result.append(ScalingRow(operation, workload, group["disposition"],
            one, two, four, first, second, assess and first < threshold,
            assess and second < threshold, counts, group["calls"]))
    return result


def prioritize(rows: list[ScalingRow], degree: int) -> list[ScalingRow]:
    """Rank by absolute recoverable time rather than the worst ratio alone."""
    if degree not in (2, 4):
        raise ValueError("ranking degree must be two or four")
    return sorted(rows, key=lambda row: (-row.excess_us(degree), row.operation, row.workload))


def render_svg(rows: list[ScalingRow], threshold: float = 1.9) -> str:
    """Render per-doubling bars and local latency; bars never sum overlapping work."""
    width, top, row_height = 1240, 130, 68
    height = top + row_height * len(rows) + 75
    start, span = 390, 360
    limit = max(2.15, *(max(row.one_to_two, row.two_to_four) * 1.06 for row in rows))
    scale = span / limit
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
           '<title id="title">Captured kernel shard scaling</title>',
           '<desc id="desc">Local kernel time at one, two and four card shapes. Red bars miss the per-doubling target. These are compute measurements, not multi-card throughput.</desc>',
           '<style>text{font-family:system-ui,sans-serif;fill:#e7edf5}.small{font-size:12px}.label{font-size:14px}.number{font:13px monospace}</style>',
           f'<rect width="{width}" height="{height}" fill="#101827"/>',
           '<text x="24" y="32" font-size="22">Kernel scaling · 1 → 2 → 4 card shapes</text>',
           f'<text x="24" y="59" class="label">Median local latency · target ≥{threshold:.2f}× for each doubling · red = needs tuning</text>',
           '<text x="24" y="82" class="small">Same GPU and precision. No interconnect in isolated probes. Kernel intervals must not be added across overlapping branches.</text>',
           '<text x="24" y="112" class="label">Operation / workload</text>',
           '<text x="390" y="112" class="label">Speedup: upper 1→2, lower 2→4</text>',
           '<text x="862" y="112" class="label">1 card µs</text>',
           '<text x="982" y="112" class="label">2 cards µs</text>',
           '<text x="1102" y="112" class="label">4 cards µs</text>']
    gate_x = start + threshold * scale
    svg.append(f'<line x1="{gate_x:.2f}" x2="{gate_x:.2f}" y1="120" y2="{height - 62}" stroke="#f0c75e" stroke-dasharray="5 4"/>')
    for index, row in enumerate(rows):
        y = top + index * row_height
        svg.extend([f'<text x="24" y="{y + 17}" class="label">{escape(row.operation)}</text>',
                    f'<text x="24" y="{y + 36}" class="small">{escape(row.workload)} · {row.disposition.value}</text>'])
        for offset, value, tune in ((0, row.one_to_two, row.tune_two), (24, row.two_to_four, row.tune_four)):
            color = "#ed7777" if tune else "#55c6b3" if row.disposition == Disposition.PARTITIONED else "#95a2b8"
            length = value * scale
            svg.extend([f'<rect x="{start}" y="{y + offset}" width="{length:.2f}" height="19" rx="3" fill="{color}"/>',
                        f'<text x="{start + length + 7:.2f}" y="{y + offset + 15}" class="number">{value:.2f}×</text>'])
        for x, value in ((862, row.one_us), (982, row.two_us), (1102, row.four_us)):
            svg.append(f'<text x="{x}" y="{y + 26}" class="number">{value:,.2f}</text>')
    svg.extend([f'<text x="24" y="{height - 32}" class="small">A passing compute curve is not a correctness or end-to-end performance certificate. Replicated/collective rows have no half-work gate.</text>', '</svg>'])
    return "\n".join(svg) + "\n"


def main() -> None:
    """Write reviewable CSV/JSON and SVG from one explicit measurement file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=1.9)
    parser.add_argument("--rank-degree", type=int, choices=(2, 4),
                        help="Rank by occurrence-weighted excess over ideal halving, not end-to-end savings")
    args = parser.parse_args()
    with args.samples.open(newline="") as handle:
        rows = summarize(csv.DictReader(handle), args.threshold)
    if args.rank_degree:
        rows = prioritize(rows, args.rank_degree)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps({
        "threshold_per_doubling": args.threshold,
        "ranking_degree": args.rank_degree,
        "ranking_note": "Isolated local-time excess, not additive critical-path savings",
        "priority": [{"operation": row.operation, "workload": row.workload,
                      "two_way_excess_us": row.excess_us(2),
                      "four_way_excess_us": row.excess_us(4)} for row in rows],
        "samples": str(args.samples), "rows": [asdict(row) for row in rows]}, indent=2) + "\n")
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows({**asdict(row), "disposition": row.disposition.value} for row in rows)
    args.output_prefix.with_suffix(".svg").write_text(render_svg(rows, args.threshold))
    print(f"{len(rows)} curves; {sum(row.tune_two for row in rows)} fail 1→2, "
          f"{sum(row.tune_four for row in rows)} fail 2→4")


if __name__ == "__main__":
    main()
