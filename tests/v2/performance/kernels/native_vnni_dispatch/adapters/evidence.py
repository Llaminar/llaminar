"""Shared exact-evidence helpers for backend NativeVNNI adapters.

The C++ trainers intentionally emit both readable aggregates and exact native
``double`` samples.  Adapters must reconstruct the aggregates from those exact
samples before a row can become policy evidence; trusting the aggregate CSV
alone would allow truncated, reordered, or stale sidecars to pass unnoticed.
"""

from __future__ import annotations

import math
import hashlib
import statistics
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class RobustTimingEvidence:
    """Python reproduction of ``trainer::TimingEvidence`` in caller units."""

    minimum: float
    median: float
    p95: float
    mad: float
    cv: float
    digest: str


def raw_corpus_id(paths: Iterable[Path]) -> str:
    """Stream-hash canonical evidence paths and bytes without a whole-file copy.

    The canonical path is part of the corpus identity, so moving a shard remains
    a deliberate provenance change while relative and absolute aliases of the
    same file do not create two identities. Reading fixed-size chunks avoids a
    second in-memory copy of timing sidecars that can contain millions of rows.
    """

    digest = hashlib.sha256()
    canonical_paths = sorted(
        Path(item).resolve(strict=True) for item in paths
    )
    for path in canonical_paths:
        digest.update(str(path).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def native_double_digest(values: Iterable[float]) -> str:
    """Reproduce the trainers' native-byte FNV-1a timing fingerprint."""

    result = 1469598103934665603
    prime = 1099511628211
    for value in values:
        for byte in struct.pack("=d", value):
            result ^= byte
            result = (result * prime) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{result:016x}"


def summarize_sorted_timing(values: Iterable[float]) -> RobustTimingEvidence:
    """Recompute the exact C++ robust statistics for sorted timing samples."""

    samples = tuple(values)
    if not samples:
        raise ValueError("timing evidence must contain at least one sample")
    if any(not math.isfinite(value) or value <= 0.0 for value in samples):
        raise ValueError("timing evidence contains a non-positive/non-finite sample")
    if tuple(sorted(samples)) != samples:
        raise ValueError("timing evidence is not in trainer-sorted order")

    median = samples[len(samples) // 2]
    p95_rank = math.ceil(0.95 * len(samples))
    p95 = samples[min(len(samples) - 1, max(1, p95_rank) - 1)]
    deviations = sorted(abs(value - median) for value in samples)
    mad = deviations[len(deviations) // 2]
    mean = statistics.fmean(samples)
    cv = math.sqrt(
        statistics.fmean((value - mean) ** 2 for value in samples)
    ) / mean
    return RobustTimingEvidence(
        minimum=samples[0],
        median=median,
        p95=p95,
        mad=mad,
        cv=cv,
        digest=native_double_digest(samples),
    )


def verify_aggregate_timing(
    raw: Mapping[str, str],
    samples: Iterable[float],
    *,
    median_field: str,
    sample_to_aggregate_scale: float = 1.0,
    time_tolerance: float = 5.1e-7,
    cv_tolerance: float = 5.1e-7,
) -> RobustTimingEvidence:
    """Verify one aggregate row against its exact sorted timing sidecar.

    ``sample_to_aggregate_scale`` supports trainers that preserve event values
    in milliseconds while reporting aggregate microseconds.  The digest always
    covers the unscaled native values exactly as the C++ trainer observed them.
    """

    summary = summarize_sorted_timing(samples)
    expected = {
        "min_us": summary.minimum * sample_to_aggregate_scale,
        median_field: summary.median * sample_to_aggregate_scale,
        "p95_us": summary.p95 * sample_to_aggregate_scale,
        "mad_us": summary.mad * sample_to_aggregate_scale,
    }
    for name, recomputed in expected.items():
        aggregate = float(raw[name])
        if not math.isclose(
            aggregate, recomputed, rel_tol=0.0, abs_tol=time_tolerance
        ):
            raise ValueError(
                f"aggregate {name}={aggregate} disagrees with raw timing {recomputed}"
            )
    aggregate_cv = float(raw["cv"])
    if not math.isclose(
        aggregate_cv, summary.cv, rel_tol=0.0, abs_tol=cv_tolerance
    ):
        raise ValueError(
            f"aggregate cv={aggregate_cv} disagrees with raw timing {summary.cv}"
        )
    timing_digest = raw["timing_sample_digest"].strip()
    if summary.digest != timing_digest:
        raise ValueError(
            "raw timing sidecar digest does not match aggregate timing_sample_digest"
        )
    return summary
