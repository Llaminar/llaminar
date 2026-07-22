"""Diagnose the profiler feature surface consumed by NativeVNNI fitting.

This command intentionally builds the same :class:`ProfilerFeatureCatalog`
used by the learner. It therefore reports normalized, provenance-checked
features rather than inspecting raw ``perf``/Nsight/rocprof counters through a
second set of formulas. The report answers four pre-fit questions:

* Did every required physical candidate publish profiler evidence?
* Is each auxiliary metric complete across whole candidate contests?
* Does that metric vary between candidates at the same work point?
* Did CPU duration/concurrency evidence pass its reliability gate?

Canonical timing remains the optimization label. This diagnostic only decides
whether a profiler transaction contains meaningful explanatory signal before a
costly cross-validated tree search begins.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from .candidate_observation import read_observation_csv
from .profiler_model import (
    PhysicalCandidateKey,
    ProfilerCandidateDescriptor,
    ProfilerFeatureCatalog,
    _auxiliary_metric_reliability_weight,
    _is_anchor_dynamic_profiler_feature,
    load_profiler_feature_catalog,
)


def _contest_key(descriptor: ProfilerCandidateDescriptor) -> tuple[str, ...]:
    """Return one work-point identity with only candidate choice removed."""

    fields = list(descriptor.key.canonical_tuple())
    # PhysicalCandidateKey canonical field 7 is effective_candidate_id. Every
    # other field, including exact M/N/K and execution mode, remains fixed so
    # no metric is compared across different physical launches.
    fields[7] = "<candidate>"
    return tuple(fields)


def _metric_names(
    descriptors: Iterable[ProfilerCandidateDescriptor],
) -> tuple[str, ...]:
    """Return normalized dynamic learner features, excluding raw counters."""

    return tuple(sorted({
        name
        for descriptor in descriptors
        for name, value in descriptor.features.items()
        if isinstance(value, (int, float))
        and name.startswith("metric.")
    }))


def _relative_span(values: Iterable[float]) -> float:
    """Measure candidate separation without exploding near a zero center."""

    finite = tuple(value for value in values if math.isfinite(value))
    if len(finite) < 2:
        return 0.0
    scale = max(abs(statistics.median(finite)), 1.0e-12)
    return (max(finite) - min(finite)) / scale


def _metric_report(
    name: str,
    descriptors: tuple[ProfilerCandidateDescriptor, ...],
    contests: Mapping[tuple[str, ...], tuple[ProfilerCandidateDescriptor, ...]],
) -> dict[str, object]:
    """Summarize availability and within-contest candidate discrimination."""

    available = sum(name in descriptor.features for descriptor in descriptors)
    multi_candidate_contests = tuple(
        rows for rows in contests.values() if len(rows) > 1
    )
    complete_spans = tuple(
        _relative_span(float(row.features[name]) for row in rows)
        for rows in multi_candidate_contests
        if all(name in row.features for row in rows)
    )
    varying = tuple(span for span in complete_spans if span > 1.0e-12)
    return {
        "available_descriptors": available,
        "availability_rate": available / len(descriptors),
        "complete_candidate_contests": len(complete_spans),
        "complete_candidate_contest_rate": (
            len(complete_spans) / len(multi_candidate_contests)
            if multi_candidate_contests
            else 0.0
        ),
        "varying_complete_contests": len(varying),
        "varying_complete_contest_rate": (
            len(varying) / len(complete_spans) if complete_spans else 0.0
        ),
        "median_candidate_relative_span": (
            statistics.median(complete_spans) if complete_spans else 0.0
        ),
        "p95_candidate_relative_span": (
            sorted(complete_spans)[
                min(len(complete_spans) - 1, int(0.95 * len(complete_spans)))
            ]
            if complete_spans
            else 0.0
        ),
    }


def _metric_expected_direction(name: str) -> str:
    """Describe the reviewed economy direction for one normalized feature."""

    lower_is_better = (
        "registers_per_thread",
        "shared_memory_bytes",
        "local_memory_bytes",
        "scratch_bytes",
        "spill",
        "warp_cycles_per_issued_instruction",
        "duration_ns_per_",
        "fetch_to_expected_byte_ratio",
        "write_to_output_byte_ratio",
    )
    higher_is_better = (
        "occupancy_pct",
        "throughput_pct_of_peak",
        "pipe_utilization_pct",
        "executed_ipc_active",
        "effective_gbytes_per_second",
        "effective_gops",
        "observed_fetch_gbytes_per_second",
        "cache_hit_pct",
    )
    if any(token in name for token in lower_is_better):
        return "lower_is_better"
    if any(token in name for token in higher_is_better):
        return "higher_is_better"
    return "unreviewed"


def _metric_signal_class(name: str) -> str:
    """Separate causal counters from direct profiler-duration proxies."""

    if any(token in name for token in (
        "effective_gbytes_per_second",
        "effective_gops",
        "duration_ns_per_",
        "observed_fetch_gbytes_per_second",
    )):
        return "profiler_duration_proxy"
    if any(token in name for token in (
        "registers_per_thread",
        "shared_memory_bytes",
        "local_memory_bytes",
        "vgpr_count",
        "sgpr_count",
        "lds_bytes",
        "scratch_bytes",
        "theoretical_occupancy_pct",
    )):
        return "static_resource"
    return "dynamic_hardware_counter"


def _performance_signal_report(
    descriptors: tuple[ProfilerCandidateDescriptor, ...],
    contests: Mapping[
        tuple[str, ...], tuple[ProfilerCandidateDescriptor, ...]
    ],
    timing_us_by_key: Mapping[PhysicalCandidateKey, float],
    *,
    minimum_slowdown_ratio: float,
) -> dict[str, object]:
    """Compare profiler features between canonical fast and slow candidates.

    Every comparison holds the complete physical work point fixed. Canonical
    repeated timing chooses the endpoints; the discrete profiler launch only
    supplies explanatory metrics and never becomes the performance label.
    """

    if minimum_slowdown_ratio <= 1.0:
        raise ValueError("minimum slowdown ratio must be greater than one")
    metric_names = _metric_names(descriptors)
    comparisons: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    eligible_contests = 0
    clear_contests = 0
    slowdown_ratios: list[float] = []
    for rows in contests.values():
        timed = tuple(
            row for row in rows if row.key in timing_us_by_key
        )
        if len(timed) < 2:
            continue
        eligible_contests += 1
        fastest = min(timed, key=lambda row: timing_us_by_key[row.key])
        slowest = max(timed, key=lambda row: timing_us_by_key[row.key])
        fastest_us = timing_us_by_key[fastest.key]
        slowest_us = timing_us_by_key[slowest.key]
        if fastest_us <= 0.0 or slowest_us / fastest_us < minimum_slowdown_ratio:
            continue
        clear_contests += 1
        slowdown_ratios.append(slowest_us / fastest_us)
        for name in metric_names:
            if name not in fastest.features or name not in slowest.features:
                continue
            fast_value = float(fastest.features[name])
            slow_value = float(slowest.features[name])
            scale = max(abs(fast_value), abs(slow_value), 1.0e-12)
            comparisons[name].append((
                fast_value,
                slow_value,
                (fast_value - slow_value) / scale,
            ))

    metric_reports = {}
    for name in metric_names:
        pairs = comparisons.get(name, ())
        direction = _metric_expected_direction(name)
        dynamic = _is_anchor_dynamic_profiler_feature(name)
        auxiliary_weight = (
            _auxiliary_metric_reliability_weight(name) if dynamic else 0.0
        )
        learner_role = (
            "auxiliary_target"
            if auxiliary_weight > 0.0
            else "diagnostic_only"
            if dynamic
            else "runtime_static_input"
        )
        matches = 0
        ties = 0
        for fast_value, slow_value, _relative_delta in pairs:
            if math.isclose(fast_value, slow_value, rel_tol=1.0e-12, abs_tol=1.0e-12):
                ties += 1
            elif (
                direction == "higher_is_better" and fast_value > slow_value
            ) or (
                direction == "lower_is_better" and fast_value < slow_value
            ):
                matches += 1
        reviewed = direction != "unreviewed"
        non_ties = len(pairs) - ties
        relative_deltas = [pair[2] for pair in pairs]
        metric_reports[name] = {
            "signal_class": _metric_signal_class(name),
            "learner_role": learner_role,
            "auxiliary_reliability_weight": auxiliary_weight,
            "expected_direction": direction,
            "paired_fast_slow_contests": len(pairs),
            "paired_fast_slow_contest_rate": (
                len(pairs) / clear_contests if clear_contests else 0.0
            ),
            "directional_matches": matches if reviewed else None,
            "directional_non_ties": non_ties if reviewed else None,
            "directional_match_rate": (
                matches / non_ties if reviewed and non_ties else None
            ),
            "tie_rate": ties / len(pairs) if pairs else 0.0,
            "median_fast_minus_slow_relative": (
                statistics.median(relative_deltas)
                if relative_deltas else 0.0
            ),
            "median_absolute_fast_slow_relative_gap": (
                statistics.median(abs(value) for value in relative_deltas)
                if relative_deltas else 0.0
            ),
        }
    return {
        "minimum_slowdown_ratio": minimum_slowdown_ratio,
        "timed_multi_candidate_contests": eligible_contests,
        "clear_fast_slow_contests": clear_contests,
        "median_slowest_over_fastest": (
            statistics.median(slowdown_ratios) if slowdown_ratios else 1.0
        ),
        "metrics": metric_reports,
    }


def diagnose_profiler_catalog(
    catalog: ProfilerFeatureCatalog,
    *,
    source_formats: Iterable[str],
    timing_us_by_key: Mapping[PhysicalCandidateKey, float] | None = None,
    minimum_slowdown_ratio: float = 1.05,
) -> dict[str, object]:
    """Return a deterministic, JSON-serializable pre-fit diagnostic report."""

    descriptors = tuple(catalog.descriptors.values())
    source_format_set = frozenset(source_formats)
    contests_mutable: dict[
        tuple[str, ...], list[ProfilerCandidateDescriptor]
    ] = defaultdict(list)
    for descriptor in descriptors:
        contests_mutable[_contest_key(descriptor)].append(descriptor)
    contests = {
        key: tuple(rows) for key, rows in contests_mutable.items()
    }
    architecture_counts = Counter(
        descriptor.key.architecture_class for descriptor in descriptors
    )
    candidate_counts = Counter(
        descriptor.key.effective_candidate_id for descriptor in descriptors
    )
    runtime_codebook_counts = Counter(
        descriptor.key.runtime_codebook_id for descriptor in descriptors
    )
    prepared_family_counts = Counter(
        descriptor.key.prepared_family_id for descriptor in descriptors
    )
    duration_reliability = Counter(
        int(descriptor.features.get(
            "profile.cpu.duration_features_reliable", 0.0
        ))
        for descriptor in descriptors
        if descriptor.key.backend.value == "cpu"
    )
    metrics = {
        name: _metric_report(name, descriptors, contests)
        for name in _metric_names(descriptors)
    }
    report = {
        "catalog_digest": catalog.digest,
        "catalog_model_digest": catalog.model_digest,
        "descriptor_count": len(descriptors),
        "contest_count": len(contests),
        "multi_candidate_contest_count": sum(
            len(rows) > 1 for rows in contests.values()
        ),
        "source_formats": sorted(source_format_set),
        "source_format_count": len(source_format_set),
        "architecture_counts": dict(sorted(architecture_counts.items())),
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "runtime_codebook_counts": {
            str(codebook): count
            for codebook, count in sorted(runtime_codebook_counts.items())
        },
        "runtime_codebook_count": len(runtime_codebook_counts),
        "prepared_family_counts": dict(sorted(prepared_family_counts.items())),
        "cpu_duration_reliability": {
            "reliable": duration_reliability[1],
            "unreliable": duration_reliability[0],
            "reliable_rate": (
                duration_reliability[1] / sum(duration_reliability.values())
                if duration_reliability
                else 0.0
            ),
        },
        "metrics": metrics,
    }
    if timing_us_by_key is not None:
        report["performance_signal"] = _performance_signal_report(
            descriptors,
            contests,
            timing_us_by_key,
            minimum_slowdown_ratio=minimum_slowdown_ratio,
        )
    return report


def _parse_args() -> argparse.Namespace:
    """Parse the standalone exact-profiler diagnostic command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--minimum-slowdown-ratio",
        type=float,
        default=1.05,
        help=(
            "minimum canonical slow/fast timing ratio for profiler feature "
            "calibration (default: 1.05)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Authenticate evidence, build learner features, and publish diagnostics."""

    args = _parse_args()
    observations = read_observation_csv((args.observation,))
    catalog = load_profiler_feature_catalog(
        observations,
        args.requests,
        args.evidence,
    )
    timing_values: dict[PhysicalCandidateKey, list[float]] = defaultdict(list)
    for observation in observations:
        timing_values[catalog.descriptor_for(observation).key].append(
            observation.median_us
        )
    timing_us_by_key = {
        key: statistics.median(values)
        for key, values in timing_values.items()
    }
    report = diagnose_profiler_catalog(
        catalog,
        source_formats=(row.source_format for row in observations),
        timing_us_by_key=timing_us_by_key,
        minimum_slowdown_ratio=args.minimum_slowdown_ratio,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
