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
    ProfilerCandidateDescriptor,
    ProfilerFeatureCatalog,
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


def diagnose_profiler_catalog(
    catalog: ProfilerFeatureCatalog,
    *,
    source_formats: Iterable[str],
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
    return {
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


def _parse_args() -> argparse.Namespace:
    """Parse the standalone exact-profiler diagnostic command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path)
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
    report = diagnose_profiler_catalog(
        catalog,
        source_formats=(row.source_format for row in observations),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
