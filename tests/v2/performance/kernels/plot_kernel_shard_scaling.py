#!/usr/bin/env python3
"""Report TP1-relative targets without conflating local shapes and actual TP.

CSV: operation,workload,degree,sample,latency_us,disposition, optional calls,
scope and rank. Missing scope means local_shape: independent local kernel
shapes, not measured multi-device throughput. actual_tp requires every rank in
paired rounds at degrees1/2/4. Its latency is the slowest participant per round,
then the median across those complete rounds. The collector owns source,
precision, stream/event timing and matched operation boundaries/provenance.

Targets independently compare T1/T2>=1.85 and T1/T4>=3.6 by default. T2/T4 is
descriptive only. Replicated/collective local shapes have no scaling verdict;
actual TP measurements retain their costs and face the same numeric
targets for the declared measured operation (which may be one kernel). Reports
do not certify correctness, end-to-end performance or dispatch installation. Optional excess ranking is not an additive critical-path claim.
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
    """Source-owned placement semantics, never inferred from measured speedup."""

    PARTITIONED = "partitioned"
    REPLICATED = "replicated"
    COLLECTIVE = "collective"


class MeasurementScope(str, Enum):
    """Whether samples are isolated local shapes or an actual participant cohort."""

    LOCAL_SHAPE = "local_shape"
    ACTUAL_TP = "actual_tp"


class TargetStatus(str, Enum):
    """Explicit unmet, met or unassessed target; unassessed never means green."""

    MET = "met"
    MISSED = "missed"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class ScalingRow:
    """One complete curve, cumulative targets and retained reduced round samples."""

    operation: str
    workload: str
    disposition: Disposition
    scope: MeasurementScope
    one_us: float
    two_us: float
    four_us: float
    one_to_two: float
    one_to_four: float
    two_to_four: float
    tp2_target: float
    tp4_target: float
    tp2_status: TargetStatus
    tp4_status: TargetStatus
    tune_two: bool
    tune_four: bool
    sample_counts: tuple[int, int, int]
    reduced_round_samples: tuple[tuple[tuple[int, float], ...], ...]
    calls: int = 1

    def excess_us(self, degree: int) -> float:
        """Rank partitioned time over TP1/degree, not achievable additive savings.

        Actual TP latency may include arrival/communication; local shapes omit
        those costs. Replicated/collective work has no ideal work-division model.
        """
        if degree not in (2, 4):
            raise ValueError("ranking degree must be two or four")
        if self.disposition != Disposition.PARTITIONED:
            return 0.0
        after = self.two_us if degree == 2 else self.four_us
        return max(0.0, after - self.one_us / degree) * self.calls


def summarize(records: Iterable[dict[str, str]], tp2_target: float = 1.85,
              tp4_target: float = 3.6) -> list[ScalingRow]:
    """Validate complete inputs and assess independent cumulative TP1 targets.

    Actual TP rows require one value for every rank0..degree-1 in each paired
    sample round, with identical round sets across all degrees. Max-rank per
    round precedes median reduction; median-of-rank medians is not substituted.
    Local shape sample identities remain independent across degrees and cannot
    carry participant ranks. No-op/raw collector meaning is outside this tool.
    """
    for value in (tp2_target, tp4_target):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 1:
            raise ValueError("TP1-relative targets must be finite and greater than one")
    groups: dict[tuple[str, str], dict] = {}
    for record in records:
        key = record["operation"], record["workload"]
        if not all(key):
            raise ValueError("every sample requires an operation and workload")
        degree, sample = int(record["degree"]), int(record["sample"])
        latency = float(record["latency_us"])
        disposition = Disposition(record["disposition"])
        scope = MeasurementScope(record.get("scope") or MeasurementScope.LOCAL_SHAPE.value)
        calls = int(record.get("calls") or "1")
        if calls <= 0:
            raise ValueError("operation occurrence count must be positive")
        if degree not in (1, 2, 4) or sample < 0 or not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"invalid timing sample for {key}")
        raw_rank = record.get("rank", "")
        if scope == MeasurementScope.LOCAL_SHAPE:
            if raw_rank not in (None, ""):
                raise ValueError("local_shape samples cannot claim participant ranks")
            rank = 0
        else:
            if raw_rank in (None, ""):
                raise ValueError("actual_tp sample requires a participant rank")
            rank = int(raw_rank)
            if rank < 0 or rank >= degree:
                raise ValueError("participant rank outside degree")
        group = groups.setdefault(key, {"disposition": disposition, "scope": scope,
            "calls": calls, "samples": {1: {}, 2: {}, 4: {}}})
        if group["calls"] != calls:
            raise ValueError(f"conflicting occurrence counts for {key}")
        if group["disposition"] != disposition or group["scope"] != scope:
            raise ValueError(f"conflicting measurement scope/sharding semantics for {key}")
        participants = group["samples"][degree].setdefault(sample, {})
        if rank in participants:
            raise ValueError(f"duplicate sample/rank identity for {key}, degree {degree}, sample {sample}, rank {rank}")
        participants[rank] = latency
    if not groups:
        raise ValueError("no kernel samples")
    result = []
    for (operation, workload), group in sorted(groups.items()):
        samples = group["samples"]
        counts = tuple(len(samples[degree]) for degree in (1, 2, 4))
        if not all(counts):
            raise ValueError(f"incomplete 1/2/4 curve for {(operation, workload)}")
        actual = group["scope"] == MeasurementScope.ACTUAL_TP
        if actual and not (set(samples[1]) == set(samples[2]) == set(samples[4])):
            raise ValueError("incomplete paired round sets for actual_tp")
        reduced = []
        for degree in (1, 2, 4):
            rounds = []
            for sample, participants in sorted(samples[degree].items()):
                expected = set(range(degree)) if actual else {0}
                if set(participants) != expected:
                    raise ValueError(f"incomplete participant ranks at degree {degree}, round {sample}")
                rounds.append((sample, max(participants.values())))
            reduced.append(tuple(rounds))
        one, two, four = (statistics.median(v for _, v in rounds) for rounds in reduced)
        first, cumulative, second = one / two, one / four, two / four
        if not all(math.isfinite(v) and v > 0 for v in (first, cumulative, second)):
            raise ValueError("nonfinite derived scaling ratio")
        assess = actual or group["disposition"] == Disposition.PARTITIONED
        status_two = (TargetStatus.MET if first >= tp2_target else TargetStatus.MISSED) if assess else TargetStatus.NOT_APPLICABLE
        status_four = (TargetStatus.MET if cumulative >= tp4_target else TargetStatus.MISSED) if assess else TargetStatus.NOT_APPLICABLE
        result.append(ScalingRow(operation, workload, group["disposition"], group["scope"],
            one, two, four, first, cumulative, second, tp2_target, tp4_target,
            status_two, status_four, status_two == TargetStatus.MISSED,
            status_four == TargetStatus.MISSED, counts, tuple(reduced), group["calls"]))
    return result


def prioritize(rows: list[ScalingRow], degree: int) -> list[ScalingRow]:
    """Rank by absolute TP1-relative excess within one explicit measurement scope."""
    if degree not in (2, 4):
        raise ValueError("ranking degree must be two or four")
    return sorted(rows, key=lambda row: (-row.excess_us(degree), row.operation, row.workload))


def render_svg(rows: list[ScalingRow]) -> str:
    """Render cumulative targets, descriptive2→4 and absolute latency with scope."""
    if not rows:
        raise ValueError("no curves to render")
    width, top, row_height = 1320, 132, 88
    height = top + row_height * len(rows) + 75
    start, span = 390, 395
    limit = max(4.15, *(max(row.one_to_two, row.one_to_four, row.tp2_target, row.tp4_target) * 1.12 for row in rows))
    scale = span / limit
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
           '<title id="title">TP1-relative kernel scaling</title>',
           '<desc id="desc">Cumulative1→2 and1→4 targets; local shapes and actual participant cohorts are explicitly distinguished. Gray means not assessed.</desc>',
           '<style>text{font-family:system-ui,sans-serif;fill:#e7edf5}.small{font-size:12px}.label{font-size:14px}.number{font:13px monospace}</style>',
           f'<rect width="{width}" height="{height}" fill="#101827"/>',
           '<text x="24" y="32" font-size="22">Kernel scaling · independent TP1-relative targets</text>',
           '<text x="24" y="59" class="label">Upper1→2 / lower1→4 · red misses target · gray unassessed ·2→4 descriptive only</text>',
           '<text x="24" y="82" class="small">local_shape is not actual TP throughput. actual_tp = median of per-round slowest participant. Intervals may overlap.</text>',
           '<text x="24" y="112" class="label">Operation / workload / scope</text>',
           '<text x="390" y="112" class="label">Cumulative speedup and target tick</text>',
           '<text x="914" y="112" class="label">TP1/shape1 µs</text>',
           '<text x="1040" y="112" class="label">TP2/shape2 µs</text>',
           '<text x="1170" y="112" class="label">TP4/shape4 µs</text>']
    for index, row in enumerate(rows):
        y = top + index * row_height
        svg.extend([f'<text x="24" y="{y + 17}" class="label">{escape(row.operation)}</text>',
                    f'<text x="24" y="{y + 36}" class="small">{escape(row.workload)} · {row.disposition.value}</text>',
                    f'<text x="24" y="{y + 55}" class="small">{row.scope.value}</text>'])
        for offset, value, target, status in ((0, row.one_to_two, row.tp2_target, row.tp2_status),
                                            (24, row.one_to_four, row.tp4_target, row.tp4_status)):
            color = {TargetStatus.MISSED: "#ed7777", TargetStatus.MET: "#55c6b3", TargetStatus.NOT_APPLICABLE: "#95a2b8"}[status]
            length = value * scale
            svg.extend([f'<rect x="{start}" y="{y + offset}" width="{length:.2f}" height="19" rx="3" fill="{color}"/>',
                        f'<text x="{start + length + 7:.2f}" y="{y + offset + 15}" class="number">{value:.2f}×</text>',
                        f'<line x1="{start + target * scale:.2f}" x2="{start + target * scale:.2f}" y1="{y + offset}" y2="{y + offset + 19}" stroke="#f0c75e"/>'])
        svg.append(f'<text x="390" y="{y + 66}" class="small">2→4 {row.two_to_four:.3f}× (descriptive) · targets {row.tp2_target:g}/{row.tp4_target:g}×</text>')
        for x, value in ((914, row.one_us), (1040, row.two_us), (1170, row.four_us)):
            svg.append(f'<text x="{x}" y="{y + 26}" class="number">{value:,.3f}</text>')
    svg.extend([f'<text x="24" y="{height - 32}" class="small">Target status is scoped evidence, not correctness/installation/end-to-end certification. Unassessed rows are never target passes.</text>', '</svg>'])
    return "\n".join(svg) + "\n"


def main() -> None:
    """Write reviewable raw/reduced CSV/JSON and SVG; target misses are reported."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--tp2-target", type=float, default=1.85,
                        help="Inclusive cumulative TP1/TP2 target (default1.85)")
    parser.add_argument("--tp4-target", type=float, default=3.6,
                        help="Inclusive cumulative TP1/TP4 target (default3.6)")
    parser.add_argument("--rank-degree", type=int, choices=(2, 4),
                        help="Rank partitioned time over TP1/degree, not additive end-to-end savings")
    args = parser.parse_args()
    with args.samples.open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows = summarize(raw, args.tp2_target, args.tp4_target)
    if args.rank_degree:
        rows = prioritize(rows, args.rank_degree)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps({
        "targets_relative_to_tp1": {"2": args.tp2_target, "4": args.tp4_target},
        "ranking_degree": args.rank_degree,
        "ranking_note": "Partitioned time over TP1/degree; not additive critical-path savings",
        "priority": [{"operation": row.operation, "workload": row.workload, "scope": row.scope.value,
                      "two_way_excess_us": row.excess_us(2), "four_way_excess_us": row.excess_us(4)} for row in rows],
        "samples": str(args.samples), "raw_samples": raw, "rows": [asdict(row) for row in rows]}, indent=2) + "\n")
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows({**asdict(row), "disposition": row.disposition.value, "scope": row.scope.value,
                          "tp2_status": row.tp2_status.value, "tp4_status": row.tp4_status.value} for row in rows)
    args.output_prefix.with_suffix(".svg").write_text(render_svg(rows))
    print(f"{len(rows)} curves; {sum(row.tune_two for row in rows)} miss TP1→TP2, "
          f"{sum(row.tune_four for row in rows)} miss TP1→TP4; 2→4 descriptive only")


if __name__ == "__main__":
    main()
