"""Contract eligibility and mode-specific, alias-robust exact winners."""

from __future__ import annotations

import math
import multiprocessing
import os
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Iterable, Mapping
from itertools import repeat
from types import MappingProxyType

from .corpus import (
    ObservationCorpus,
    RuntimeKey,
    SurfaceKey,
    _physical_core_count,
    observed_surfaces,
    runtime_key,
)
from .schema import Backend, ExecutionMode, NativeQuantisedObservation, SemanticContract
from .format_registry import format_spec, runtime_aliases


# Exact runtime keys are independent optimization problems. Fork workers read
# the immutable corpus through copy-on-write mappings, while their task payload
# contains only a small key. This avoids pickling trainer-scale row inventories
# into every worker.
_PARALLEL_EXACT_CORPUS: ObservationCorpus | None = None
_PARALLEL_EXACT_REQUIRED_SURFACES: Mapping[
    RuntimeKey, frozenset[SurfaceKey]
] = {}
_PARALLEL_EXACT_SERIAL_HASHES: Mapping[RuntimeKey, str] = {}


def _build_exact_winner_for_key(key: RuntimeKey,
        projection: ArithmeticPolicyKeyProjection | None = None) -> tuple[RuntimeKey, "ExactWinner"]:
    """Build one exact winner from the fork-inherited immutable corpus."""

    if _PARALLEL_EXACT_CORPUS is None:
        raise RuntimeError("parallel exact-winner corpus was not initialized")
    winner = build_exact_winner(
        _PARALLEL_EXACT_CORPUS.rows_for_runtime_key(key),
        required_surfaces=_PARALLEL_EXACT_REQUIRED_SURFACES.get(key),
        current_serial_m1_hash=_PARALLEL_EXACT_SERIAL_HASHES.get(key),
        runtime_key_override=key,
        arithmetic_projection=projection,
    )
    return key, winner


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


@dataclass(frozen=True)
class ArithmeticPolicyKeyProjection:
    """Project raw physical identities onto a proven source-arithmetic ABI.

    Only ROCm's reviewed source-aware decode selector is admitted here. Raw
    observations, preparation IDs and physical launch evidence stay unchanged.
    This grouping key names a policy authority, not a pretend packed payload.
    Geometry and arithmetic partition identity cannot be projected away.
    """

    backend: Backend

    def project(self, key: RuntimeKey) -> RuntimeKey:
        """Resolve exactly one canonical policy ID for a physical format key."""

        if self.backend != Backend.ROCM or key.backend != self.backend:
            raise ValueError("unproved arithmetic policy key projection")
        aliases = runtime_aliases(self.backend.value, key.runtime_codebook_id)
        policies = {format_spec(alias).arithmetic_policy_codebook(self.backend.value)
                    for alias in aliases}
        if len(policies) != 1:
            raise ValueError("physical aliases do not identify one arithmetic policy")
        policy, = policies
        return replace(key, runtime_codebook_id=policy,
                       prepared_family_id=f"{self.backend.value}.arithmetic-policy-cb{policy}",
                       packing_abi="source-arithmetic-policy-v1")


@dataclass(frozen=True)
class ArithmeticPolicyGroups:
    """Immutable source-policy index retaining original physical observations."""

    groups: Mapping[RuntimeKey, tuple[NativeQuantisedObservation, ...]]

    def __post_init__(self) -> None:
        """Freeze index ownership without duplicating observation payloads."""
        object.__setattr__(self, "groups", MappingProxyType(dict(self.groups)))

    def runtime_keys(self) -> tuple[RuntimeKey, ...]:
        """Return deterministic policy-key order for the existing worker pool."""
        return tuple(sorted(self.groups))

    def rows_for_runtime_key(self, key: RuntimeKey) -> tuple[NativeQuantisedObservation, ...]:
        """Read raw physical evidence for exactly one policy authority."""
        return self.groups.get(key, ())


def candidate_is_eligible(
    observation: NativeQuantisedObservation,
    current_serial_m1_hash: str | None = None,
) -> bool:
    """Apply the semantic contract before any performance optimization."""

    from .adaptive_tournament import is_promotion_grade
    if not is_promotion_grade(observation):return False
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


def _aggregate_surface(rows: list[NativeQuantisedObservation]) -> CandidateSurfaceTiming:
    """Merge repeated aggregate runs without pretending they are raw samples."""

    first = rows[0]
    return CandidateSurfaceTiming(
        candidate_id=first.effective_candidate_id,
        surface=SurfaceKey.from_observation(first),
        median_us=statistics.median(row.median_us for row in rows),
        p95_us=max(row.p95_us for row in rows),
        cv=max(row.cv for row in rows),
        sample_count=sum(row.sample_count for row in rows),
    )


def build_exact_winner(
    rows: Iterable[NativeQuantisedObservation],
    *,
    required_surfaces: frozenset[SurfaceKey] | None = None,
    current_serial_m1_hash: str | None = None,
    runtime_key_override: RuntimeKey | None = None,
    arithmetic_projection: ArithmeticPolicyKeyProjection | None = None,
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
    if arithmetic_projection is not None:
        keys = {arithmetic_projection.project(key) for key in keys}
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

    grouped: dict[tuple[str, SurfaceKey], list[NativeQuantisedObservation]] = defaultdict(list)
    exemplars: dict[str, NativeQuantisedObservation] = {}
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
        surface = SurfaceKey.from_observation(row)
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
    corpus: ObservationCorpus | ArithmeticPolicyGroups,
    *,
    required_surfaces: Mapping[RuntimeKey, frozenset[SurfaceKey]] | None = None,
    serial_m1_hashes: Mapping[RuntimeKey, str] | None = None,
    arithmetic_projection: ArithmeticPolicyKeyProjection | None = None,
) -> dict[RuntimeKey, ExactWinner]:
    """Construct exact winners across affinity-visible physical CPU cores.

    Winner selection has no cross-key mutable state. Large production corpora
    therefore distribute sorted runtime keys across fork workers and collect
    results in input order, preserving the byte-stable policy order. Small
    corpora remain serial to avoid process startup overhead. The optional
    ``LLAMINAR_NATIVE_QUANTISED_EXACT_WINNER_WORKERS`` override is still capped at
    the physical core count.
    """

    keys = corpus.runtime_keys()
    requested_workers = int(os.environ.get(
        "LLAMINAR_NATIVE_QUANTISED_EXACT_WINNER_WORKERS",
        str(_physical_core_count()),
    ))
    if requested_workers < 1:
        raise ValueError("exact-winner worker count must be positive")
    worker_count = min(
        requested_workers,
        _physical_core_count(),
        max(1, len(keys) // 16),
    )
    if worker_count == 1:
        return {
            key: build_exact_winner(
                corpus.rows_for_runtime_key(key),
                required_surfaces=(required_surfaces or {}).get(key),
                current_serial_m1_hash=(serial_m1_hashes or {}).get(key),
                runtime_key_override=key,
                arithmetic_projection=arithmetic_projection,
            )
            for key in keys
        }

    global _PARALLEL_EXACT_CORPUS
    global _PARALLEL_EXACT_REQUIRED_SURFACES
    global _PARALLEL_EXACT_SERIAL_HASHES
    _PARALLEL_EXACT_CORPUS = corpus
    _PARALLEL_EXACT_REQUIRED_SURFACES = required_surfaces or {}
    _PARALLEL_EXACT_SERIAL_HASHES = serial_m1_hashes or {}
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            pairs = tuple(executor.map(_build_exact_winner_for_key, keys,
                                       repeat(arithmetic_projection)))
    finally:
        _PARALLEL_EXACT_CORPUS = None
        _PARALLEL_EXACT_REQUIRED_SURFACES = {}
        _PARALLEL_EXACT_SERIAL_HASHES = {}
    return dict(pairs)
