"""Contract eligibility and mode-specific, alias-robust exact winners."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, Mapping

from .corpus import (
    ObservationCorpus,
    RuntimeKey,
    SurfaceKey,
    observed_surfaces,
    runtime_key,
)
from .schema import ExecutionMode, NativeVNNIObservation, SemanticContract


@dataclass(frozen=True)
class CandidateSurfaceTiming:
    """Robust latency for one candidate on one source-alias/execution surface."""

    candidate_id: str
    surface: SurfaceKey
    median_us: float
    p95_us: float
    cv: float
    sample_count: int


@dataclass(frozen=True)
class ExactWinner:
    """Runtime-representable candidate minimizing worst-surface regret."""

    key: RuntimeKey
    candidate_id: str
    candidate_family: str
    config_json: dict
    arithmetic_fingerprint: str
    max_surface_regret: float
    p95_surface_regret: float
    mean_normalized_latency: float
    max_cv: float
    surfaces: tuple[CandidateSurfaceTiming, ...]


def candidate_is_eligible(
    observation: NativeVNNIObservation,
    current_serial_m1_hash: str | None = None,
) -> bool:
    """Apply the semantic contract before any performance optimization."""

    common = (
        observation.supported
        and observation.forced_route_ok
        and observation.route_counter_ok
        and observation.workspace_ok
        and observation.explicit_stream_ok
        and observation.observed_candidate_id == observation.effective_candidate_id
        and (
            observation.execution_mode != ExecutionMode.GRAPH_CAPTURED
            or observation.graph_capture_ok
        )
    )
    if not common:
        return False

    if observation.semantic_contract == SemanticContract.FAST:
        return observation.numerical_correctness

    if observation.semantic_contract != SemanticContract.VERIFIER_SERIAL_M1_BITWISE:
        return False
    if current_serial_m1_hash is not None and (
        observation.serial_m1_policy_hash != current_serial_m1_hash
    ):
        return False
    return (
        observation.bitwise_equal
        and observation.repeat_equal
        and observation.mismatch_count == 0
        and observation.numerical_correctness
        and observation.ordered_reduction
        and not observation.uses_atomic_reduction
    )


def _aggregate_surface(rows: list[NativeVNNIObservation]) -> CandidateSurfaceTiming:
    """Merge repeated aggregate runs without pretending they are raw samples."""

    first = rows[0]
    return CandidateSurfaceTiming(
        candidate_id=first.effective_candidate_id,
        surface=SurfaceKey(first.source_format, first.execution_mode),
        median_us=statistics.median(row.median_us for row in rows),
        p95_us=max(row.p95_us for row in rows),
        cv=max(row.cv for row in rows),
        sample_count=sum(row.sample_count for row in rows),
    )


def build_exact_winner(
    rows: Iterable[NativeVNNIObservation],
    *,
    required_surfaces: frozenset[SurfaceKey] | None = None,
    current_serial_m1_hash: str | None = None,
    runtime_key_override: RuntimeKey | None = None,
) -> ExactWinner:
    """Select one exact candidate using worst source-alias normalized latency.

    Policy ABI v2 discriminates eager from graph-captured execution because
    multi-kernel launch gaps can change the economical schedule. It still cannot
    discriminate source aliases that normalize to the same prepared codebook.
    A candidate missing or ineligible on any required alias surface is removed
    before timing comparison.
    """

    row_list = list(rows)
    if not row_list:
        raise ValueError("cannot construct an exact winner from an empty row set")
    keys = {runtime_key(row) for row in row_list}
    if runtime_key_override is not None:
        keys = {
            replace(key, execution_mode=runtime_key_override.execution_mode)
            for key in keys
        }
    if len(keys) != 1:
        raise ValueError("exact winner rows span multiple runtime keys")
    key = next(iter(keys))
    if runtime_key_override is not None and key != runtime_key_override:
        raise ValueError("exact winner override disagrees with normalized rows")
    surfaces = required_surfaces or observed_surfaces(row_list)
    if not surfaces:
        raise ValueError("an exact runtime key has no alias/mode surfaces")

    grouped: dict[tuple[str, SurfaceKey], list[NativeVNNIObservation]] = defaultdict(list)
    exemplars: dict[str, NativeVNNIObservation] = {}
    for row in row_list:
        if not candidate_is_eligible(row, current_serial_m1_hash):
            continue
        # Exact overlays publish one concrete launch identity. Shape-resolved
        # candidates are generic policy formulas whose effective launch varies
        # with N/K; their directly measured concrete source row remains the
        # eligible exact candidate for this key.
        if row.candidate_id != row.effective_candidate_id:
            continue
        candidate = row.effective_candidate_id
        surface = SurfaceKey(row.source_format, row.execution_mode)
        grouped[(candidate, surface)].append(row)
        exemplars.setdefault(candidate, row)

    candidate_ids = sorted({candidate for candidate, _ in grouped})
    complete_candidates = [
        candidate
        for candidate in candidate_ids
        if all((candidate, surface) in grouped for surface in surfaces)
    ]
    if not complete_candidates:
        raise ValueError(
            f"no eligible candidate covers every required surface for {key}"
        )

    timing: dict[tuple[str, SurfaceKey], CandidateSurfaceTiming] = {
        group_key: _aggregate_surface(group_rows)
        for group_key, group_rows in grouped.items()
        if group_key[0] in complete_candidates and group_key[1] in surfaces
    }
    best_latency = {
        surface: min(timing[(candidate, surface)].median_us
                     for candidate in complete_candidates)
        for surface in surfaces
    }

    scored = []
    for candidate in complete_candidates:
        candidate_timings = tuple(
            timing[(candidate, surface)] for surface in sorted(surfaces)
        )
        regrets = [
            item.median_us / best_latency[item.surface] - 1.0
            for item in candidate_timings
        ]
        p95_regrets = [
            item.p95_us / best_latency[item.surface] - 1.0
            for item in candidate_timings
        ]
        normalized = [
            item.median_us / best_latency[item.surface]
            for item in candidate_timings
        ]
        scored.append((
            max(regrets),
            max(p95_regrets),
            statistics.fmean(normalized),
            max(item.cv for item in candidate_timings),
            candidate,
            candidate_timings,
        ))

    score = min(scored, key=lambda item: item[:5])
    exemplar = exemplars[score[4]]
    return ExactWinner(
        key=key,
        candidate_id=score[4],
        candidate_family=exemplar.candidate_family,
        config_json=exemplar.config_json,
        arithmetic_fingerprint=exemplar.arithmetic_fingerprint,
        max_surface_regret=score[0],
        p95_surface_regret=score[1],
        mean_normalized_latency=score[2],
        max_cv=score[3],
        surfaces=score[5],
    )


def build_exact_winners(
    corpus: ObservationCorpus,
    *,
    required_surfaces: Mapping[RuntimeKey, frozenset[SurfaceKey]] | None = None,
    serial_m1_hashes: Mapping[RuntimeKey, str] | None = None,
) -> dict[RuntimeKey, ExactWinner]:
    """Construct exact winners for every runtime key in a validated corpus."""

    result: dict[RuntimeKey, ExactWinner] = {}
    for key in corpus.runtime_keys():
        result[key] = build_exact_winner(
            corpus.rows_for_runtime_key(key),
            required_surfaces=(required_surfaces or {}).get(key),
            current_serial_m1_hash=(serial_m1_hashes or {}).get(key),
            runtime_key_override=key,
        )
    return result
