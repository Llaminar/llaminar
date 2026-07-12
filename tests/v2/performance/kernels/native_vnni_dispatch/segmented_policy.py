"""Bounded aspect/work learner minimizing measured candidate latency regret."""

from __future__ import annotations

import itertools
import hashlib
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from .corpus import GenericDomain, ObservationCorpus, RuntimeKey, SurfaceKey, generic_domain, runtime_key
from .exact_oracle import candidate_is_eligible
from .schema import NativeVNNIObservation


MAX_WORK_ITEMS = (1 << 63) - 1


@dataclass(frozen=True)
class CandidatePointCost:
    """Worst alias/mode regret for one candidate at one logical shape point."""

    runtime_key: RuntimeKey
    shape_group_id: str
    work_items: int
    candidate_id: str
    max_surface_regret: float
    p95_surface_regret: float
    mean_surface_regret: float


@dataclass(frozen=True)
class GenericDispatchRule:
    """One contiguous work-size segment selected for a generic domain."""

    domain: GenericDomain
    min_work_items: int
    max_work_items: int
    candidate_id: str
    arithmetic_fingerprint: str
    development_shape_groups: tuple[str, ...]
    development_max_regret: float
    development_p95_regret: float
    development_mean_regret: float


@dataclass(frozen=True)
class DomainCrossValidation:
    """Grouped development-CV result used to freeze one domain's complexity."""

    domain: GenericDomain
    selected_max_segments: int
    fold_count: int
    shape_group_count: int
    required_point_count: int
    covered_point_count: int
    max_regret: float
    p95_regret: float
    mean_regret: float


@dataclass(frozen=True)
class GenericPolicy:
    """Frozen generic rules plus domains lacking enough safe evidence."""

    rules: tuple[GenericDispatchRule, ...]
    unpromoted_domains: tuple[GenericDomain, ...]
    cross_validation: tuple[DomainCrossValidation, ...] = ()

    def resolve(self, domain: GenericDomain, work_items: int) -> GenericDispatchRule | None:
        """Resolve one generic key without consulting any exact overlay."""

        for rule in self.rules:
            if (
                rule.domain == domain
                and rule.min_work_items <= work_items <= rule.max_work_items
            ):
                return rule
        return None


def _surface_candidate_timings(
    rows: Iterable[NativeVNNIObservation],
    current_serial_m1_hash: str | None,
) -> tuple[dict[tuple[str, SurfaceKey], float], frozenset[SurfaceKey]]:
    grouped: dict[tuple[str, SurfaceKey], list[float]] = defaultdict(list)
    all_surfaces: set[SurfaceKey] = set()
    for row in rows:
        surface = SurfaceKey(row.source_format, row.execution_mode)
        all_surfaces.add(surface)
        if not row.generic_eligible:
            continue
        if not candidate_is_eligible(row, current_serial_m1_hash):
            continue
        grouped[(row.effective_candidate_id, surface)].append(row.median_us)
    return (
        {key: statistics.median(values) for key, values in grouped.items()},
        frozenset(all_surfaces),
    )


def build_candidate_point_costs(
    corpus: ObservationCorpus,
    *,
    serial_m1_hashes: Mapping[RuntimeKey, str] | None = None,
) -> dict[GenericDomain, list[CandidatePointCost]]:
    """Build the complete eligible candidate regret matrix for development.

    Rows are grouped by runtime key and shape group. A candidate must cover every
    observed alias/mode surface at that point. Missing candidate measurements are
    therefore unavailable rather than being treated as zero latency or an exact
    winner label miss.
    """

    point_rows: dict[tuple[RuntimeKey, str], list[NativeVNNIObservation]] = defaultdict(list)
    for row in corpus:
        point_rows[(runtime_key(row), row.shape_group_id)].append(row)

    result: dict[GenericDomain, list[CandidatePointCost]] = defaultdict(list)
    for (key, shape_group_id), rows in point_rows.items():
        current_hash = (serial_m1_hashes or {}).get(key)
        timings, surfaces = _surface_candidate_timings(rows, current_hash)
        candidates = sorted({candidate for candidate, _ in timings})
        complete = [
            candidate
            for candidate in candidates
            if all((candidate, surface) in timings for surface in surfaces)
        ]
        if not complete:
            continue
        best = {
            surface: min(timings[(candidate, surface)] for candidate in complete)
            for surface in surfaces
        }
        domain = generic_domain(rows[0])
        for candidate in complete:
            regrets = [
                timings[(candidate, surface)] / best[surface] - 1.0
                for surface in surfaces
            ]
            result[domain].append(CandidatePointCost(
                runtime_key=key,
                shape_group_id=shape_group_id,
                work_items=key.aggregate_n * key.k,
                candidate_id=candidate,
                max_surface_regret=max(regrets),
                p95_surface_regret=_percentile(regrets, 0.95),
                mean_surface_regret=statistics.fmean(regrets),
            ))
    return dict(result)


def _percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.inf
    index = (len(ordered) - 1) * quantile
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    fraction = index - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _point_matrix(costs: list[CandidatePointCost]):
    matrix: dict[tuple[RuntimeKey, str], dict[str, CandidatePointCost]] = defaultdict(dict)
    for cost in costs:
        matrix[(cost.runtime_key, cost.shape_group_id)][cost.candidate_id] = cost
    return matrix


def _best_candidate_for_segment(
    points: list[tuple[RuntimeKey, str]],
    matrix: Mapping[tuple[RuntimeKey, str], Mapping[str, CandidatePointCost]],
) -> tuple[str, tuple[float, int, float, float, str]] | None:
    common_candidates = set(matrix[points[0]])
    for point in points[1:]:
        common_candidates.intersection_update(matrix[point])
    if not common_candidates:
        return None

    scored = []
    for candidate in sorted(common_candidates):
        rows = [matrix[point][candidate] for point in points]
        max_regret = max(row.max_surface_regret for row in rows)
        over_limit = sum(row.max_surface_regret > 0.03 for row in rows)
        p95 = _percentile((row.max_surface_regret for row in rows), 0.95)
        mean = statistics.fmean(row.mean_surface_regret for row in rows)
        scored.append((max_regret, over_limit, p95, mean, candidate))
    score = min(scored)
    return score[4], score


def fit_domain_rules(
    domain: GenericDomain,
    costs: list[CandidatePointCost],
    observations: Iterable[NativeVNNIObservation],
    *,
    max_segments: int = 3,
    min_shape_groups_per_leaf: int = 2,
) -> tuple[GenericDispatchRule, ...]:
    """Exhaustively fit up to three contiguous work segments for one domain."""

    if max_segments < 1 or max_segments > 3:
        raise ValueError("policy ABI v1 permits one to three work segments")
    matrix = _point_matrix(costs)
    points = sorted(
        matrix,
        key=lambda point: (
            point[0].aggregate_n * point[0].k,
            point[0].aggregate_n,
            point[0].k,
            point[1],
        ),
    )
    if len({point[1] for point in points}) < min_shape_groups_per_leaf:
        return ()

    # Cuts occur only between distinct work values so equal-work shapes cannot
    # be assigned different policies through incidental ordering.
    cut_indices = [
        index
        for index in range(1, len(points))
        if points[index - 1][0].aggregate_n * points[index - 1][0].k
        < points[index][0].aggregate_n * points[index][0].k
    ]
    candidates = []
    for segment_count in range(1, max_segments + 1):
        for cuts in itertools.combinations(cut_indices, segment_count - 1):
            boundaries = (0, *cuts, len(points))
            segments = [points[boundaries[i]:boundaries[i + 1]]
                        for i in range(segment_count)]
            if any(
                len({point[1] for point in segment}) < min_shape_groups_per_leaf
                for segment in segments
            ):
                continue
            selected = [_best_candidate_for_segment(segment, matrix) for segment in segments]
            if any(item is None for item in selected):
                continue
            scores = [item[1] for item in selected if item is not None]
            all_regrets = [
                matrix[point][selected[index][0]].max_surface_regret
                for index, segment in enumerate(segments)
                for point in segment
            ]
            objective = (
                max(score[0] for score in scores),
                sum(score[1] for score in scores),
                _percentile(all_regrets, 0.95),
                statistics.fmean(all_regrets),
                segment_count,
                tuple(item[0] for item in selected if item is not None),
                cuts,
            )
            candidates.append((objective, segments, selected))

    if not candidates:
        return ()
    _, segments, selected = min(candidates, key=lambda item: item[0])
    exemplar_by_candidate = {
        row.effective_candidate_id: row
        for row in observations
        if generic_domain(row) == domain
    }
    rules = []
    previous_max = 0
    for index, (segment, selection) in enumerate(zip(segments, selected)):
        assert selection is not None
        candidate, score = selection
        if index == len(segments) - 1:
            maximum = MAX_WORK_ITEMS
        else:
            lower_work = segment[-1][0].aggregate_n * segment[-1][0].k
            upper_runtime_key = segments[index + 1][0][0]
            upper_work = upper_runtime_key.aggregate_n * upper_runtime_key.k
            # Place the runtime boundary between measured work values. This is
            # stable for unseen integer work sizes and does not assign the whole
            # gap to whichever segment happened to own the lower observation.
            maximum = lower_work + (upper_work - lower_work) // 2
        exemplar = exemplar_by_candidate[candidate]
        rules.append(GenericDispatchRule(
            domain=domain,
            min_work_items=previous_max + 1,
            max_work_items=maximum,
            candidate_id=candidate,
            arithmetic_fingerprint=exemplar.arithmetic_fingerprint,
            development_shape_groups=tuple(sorted({point[1] for point in segment})),
            development_max_regret=score[0],
            development_p95_regret=score[2],
            development_mean_regret=score[3],
        ))
        previous_max = maximum
    return tuple(rules)


def _domain_folds(
    domain: GenericDomain,
    shape_groups: Iterable[str],
    *,
    seed: str,
) -> tuple[frozenset[str], ...]:
    """Build deterministic grouped 5-fold or leave-one-group-out partitions."""

    groups = tuple(sorted(set(shape_groups)))
    if len(groups) < 3:
        return ()
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"{seed}\0{domain!r}\0{group}".encode()
        ).digest(),
    )
    if len(groups) < 10:
        return tuple(frozenset((group,)) for group in ordered)

    buckets: list[set[str]] = [set() for _ in range(5)]
    for index, group in enumerate(ordered):
        buckets[index % len(buckets)].add(group)
    return tuple(frozenset(bucket) for bucket in buckets if bucket)


def _evaluate_rules(
    rules: tuple[GenericDispatchRule, ...],
    held_out_costs: list[CandidatePointCost],
) -> tuple[list[float], int, int]:
    """Evaluate frozen candidate/cut decisions on held-out shape points."""

    matrix = _point_matrix(held_out_costs)
    regrets: list[float] = []
    uncovered = 0
    for (runtime_key, _shape_group), candidates in matrix.items():
        work_items = runtime_key.aggregate_n * runtime_key.k
        rule = next((
            item for item in rules
            if item.min_work_items <= work_items <= item.max_work_items
        ), None)
        if rule is None or rule.candidate_id not in candidates:
            uncovered += 1
            continue
        regrets.append(candidates[rule.candidate_id].max_surface_regret)
    return regrets, uncovered, len(matrix)


def cross_validate_domain_complexity(
    domain: GenericDomain,
    costs: list[CandidatePointCost],
    observations: Iterable[NativeVNNIObservation],
    *,
    max_segments: int,
    min_shape_groups_per_leaf: int,
    seed: str,
) -> DomainCrossValidation | None:
    """Select permitted segment complexity without row or shape-group leakage."""

    shape_groups = {cost.shape_group_id for cost in costs}
    folds = _domain_folds(domain, shape_groups, seed=seed)
    if not folds:
        return None
    domain_observations = tuple(
        row for row in observations if generic_domain(row) == domain
    )

    scored = []
    for complexity in range(1, max_segments + 1):
        all_regrets: list[float] = []
        uncovered = 0
        required = 0
        for held_out_groups in folds:
            training_costs = [
                cost for cost in costs
                if cost.shape_group_id not in held_out_groups
            ]
            held_out_costs = [
                cost for cost in costs
                if cost.shape_group_id in held_out_groups
            ]
            training_observations = [
                row for row in domain_observations
                if row.shape_group_id not in held_out_groups
            ]
            rules = fit_domain_rules(
                domain,
                training_costs,
                training_observations,
                max_segments=complexity,
                min_shape_groups_per_leaf=min_shape_groups_per_leaf,
            )
            fold_regrets, fold_uncovered, fold_required = _evaluate_rules(
                rules, held_out_costs
            )
            all_regrets.extend(fold_regrets)
            uncovered += fold_uncovered
            required += fold_required

        if required == 0:
            continue
        max_regret = max(all_regrets, default=math.inf)
        p95_regret = _percentile(all_regrets, 0.95)
        mean_regret = statistics.fmean(all_regrets) if all_regrets else math.inf
        objective = (
            uncovered,
            max_regret,
            sum(regret > 0.03 for regret in all_regrets),
            p95_regret,
            mean_regret,
            complexity,
        )
        scored.append((
            objective,
            DomainCrossValidation(
                domain=domain,
                selected_max_segments=complexity,
                fold_count=len(folds),
                shape_group_count=len(shape_groups),
                required_point_count=required,
                covered_point_count=required - uncovered,
                max_regret=max_regret,
                p95_regret=p95_regret,
                mean_regret=mean_regret,
            ),
        ))

    if not scored:
        return None
    _, result = min(scored, key=lambda item: item[0])
    if result.covered_point_count != result.required_point_count:
        return None
    return result


def fit_generic_policy(
    development: ObservationCorpus,
    *,
    serial_m1_hashes: Mapping[RuntimeKey, str] | None = None,
    max_segments: int = 3,
    min_shape_groups_per_leaf: int = 2,
    cross_validation_seed: str = "native-vnni-development-cv-v1",
) -> GenericPolicy:
    """Cross-validate complexity, then fit each final domain exactly once."""

    costs = build_candidate_point_costs(
        development, serial_m1_hashes=serial_m1_hashes
    )
    rules = []
    unpromoted = []
    cross_validation = []
    observations = development.observations
    for domain in development.generic_domains():
        validation = cross_validate_domain_complexity(
            domain,
            costs.get(domain, []),
            observations,
            max_segments=max_segments,
            min_shape_groups_per_leaf=min_shape_groups_per_leaf,
            seed=cross_validation_seed,
        )
        if validation is None:
            unpromoted.append(domain)
            continue
        domain_rules = fit_domain_rules(
            domain,
            costs.get(domain, []),
            observations,
            max_segments=validation.selected_max_segments,
            min_shape_groups_per_leaf=min_shape_groups_per_leaf,
        )
        if domain_rules:
            # CV metrics live in the sibling DomainCrossValidation record so
            # every leaf shares one unambiguous domain score.
            rules.extend(domain_rules)
            cross_validation.append(validation)
        else:
            unpromoted.append(domain)
    return GenericPolicy(
        rules=tuple(sorted(rules, key=lambda rule: (
            rule.domain, rule.min_work_items, rule.max_work_items
        ))),
        unpromoted_domains=tuple(sorted(unpromoted)),
        cross_validation=tuple(sorted(
            cross_validation, key=lambda result: result.domain
        )),
    )
