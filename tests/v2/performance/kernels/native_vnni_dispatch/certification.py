"""Generic-only sealed evaluation and p95-regret certificate."""

from __future__ import annotations

import math
import multiprocessing
import os
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from statistics import NormalDist
from typing import Mapping

from .corpus import (
    GenericDomain,
    ObservationCorpus,
    RuntimeKey,
    SurfaceKey,
    _physical_core_count,
)
from .exact_oracle import build_exact_winner, candidate_is_eligible
from .policy_ir import PolicyIR
from .schema import (
    MINIMUM_PASSING_DOMAIN_FRACTION,
    NativeVNNIObservation,
    P95_REGRET_BUDGET,
    SemanticContract,
)
from .segmented_policy import (
    GenericDispatchRule,
    domain_promotion_quota_is_satisfied,
)


def _nearest_rank_p95(values: tuple[float, ...]) -> float:
    """Return the deterministic nearest-rank p95 used by every regret gate."""

    if not values:
        return math.inf
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


@dataclass(frozen=True)
class CertificationCell:
    """One sealed shape evaluated through generic resolution only."""

    runtime_key: RuntimeKey
    shape_group_id: str
    domain: GenericDomain
    selected_candidate_id: str
    exact_candidate_id: str
    observed_worst_surface_regret: float
    simultaneous_95pct_upper_regret: float
    alias_count: int
    execution_mode_count: int


@dataclass(frozen=True)
class CertificationRuleCoverage:
    """One frozen generic leaf and its independent sealed exercise count.

    Keeping the complete rule in the diagnostic report makes an unreachable
    leaf actionable: the report identifies its backend/domain, candidate, and
    predicates instead of reducing a split-design failure to one integer.
    """

    rule: GenericDispatchRule
    sealed_hit_count: int


@dataclass(frozen=True)
class CertificationDomainResult:
    """Sealed p95 evidence for one complete generic dispatch domain."""

    domain: GenericDomain
    sealed_cell_count: int
    p95_observed_regret: float
    p95_simultaneous_95pct_upper_regret: float

    def passes(self, p95_regret: float = P95_REGRET_BUDGET) -> bool:
        """Return whether both strict per-domain performance gates pass."""

        return (
            self.p95_observed_regret < p95_regret
            and self.p95_simultaneous_95pct_upper_regret < p95_regret
        )


@dataclass(frozen=True)
class CertificationReport:
    """Immutable sealed certificate bound to a frozen generic policy digest."""

    frozen_generic_policy_digest: str
    cells: tuple[CertificationCell, ...]
    sealed_cell_count: int
    out_of_scope_cell_count: int
    required_cell_count: int
    covered_cell_count: int
    verifier_bitwise_failures: int
    unexercised_rule_count: int
    unpromoted_domain_count: int
    rule_coverage: tuple[CertificationRuleCoverage, ...] = ()
    p95_regret_budget: float = P95_REGRET_BUDGET
    minimum_passing_domain_fraction: float = MINIMUM_PASSING_DOMAIN_FRACTION

    @property
    def coverage(self) -> float:
        return (
            float(self.covered_cell_count) / float(self.required_cell_count)
            if self.required_cell_count
            else 0.0
        )

    @property
    def max_observed_regret(self) -> float:
        return max((cell.observed_worst_surface_regret for cell in self.cells), default=math.inf)

    @property
    def p95_observed_regret(self) -> float:
        return _nearest_rank_p95(tuple(
            cell.observed_worst_surface_regret for cell in self.cells
        ))

    @property
    def max_simultaneous_95pct_upper_regret(self) -> float:
        return max((cell.simultaneous_95pct_upper_regret for cell in self.cells), default=math.inf)

    @property
    def p95_simultaneous_95pct_upper_regret(self) -> float:
        """Return nearest-rank p95 of conservative per-cell regret bounds."""

        return _nearest_rank_p95(tuple(
            cell.simultaneous_95pct_upper_regret for cell in self.cells
        ))

    @property
    def domain_results(self) -> tuple[CertificationDomainResult, ...]:
        """Aggregate sealed cells into the domains that own promotion."""

        grouped: dict[GenericDomain, list[CertificationCell]] = defaultdict(list)
        for cell in self.cells:
            grouped[cell.domain].append(cell)
        return tuple(
            CertificationDomainResult(
                domain=domain,
                sealed_cell_count=len(cells),
                p95_observed_regret=_nearest_rank_p95(tuple(
                    cell.observed_worst_surface_regret for cell in cells
                )),
                p95_simultaneous_95pct_upper_regret=_nearest_rank_p95(tuple(
                    cell.simultaneous_95pct_upper_regret for cell in cells
                )),
            )
            for domain, cells in sorted(grouped.items())
        )

    def passing_domain_count(
        self,
        p95_regret: float | None = None,
    ) -> int:
        """Count domains whose observed and conservative p95 are in budget."""

        budget = self.p95_regret_budget if p95_regret is None else p95_regret
        return sum(result.passes(budget) for result in self.domain_results)

    @property
    def required_domain_count(self) -> int:
        """Return the number of independently certified generic domains."""

        return len(self.domain_results)

    @property
    def passing_domain_fraction(self) -> float:
        """Return the default-budget fraction used by artifact publication."""

        return (
            float(self.passing_domain_count()) / float(self.required_domain_count)
            if self.required_domain_count
            else 0.0
        )

    @property
    def over_budget_domains(self) -> tuple[CertificationDomainResult, ...]:
        """Retain every certificate-budget exception as diagnostics."""

        return tuple(
            result
            for result in self.domain_results
            if not result.passes(self.p95_regret_budget)
        )

    def require_promotable(
        self,
        *,
        p95_regret: float | None = None,
        minimum_passing_fraction: float | None = None,
    ) -> None:
        """Fail unless hard gates and the frozen promotion criteria pass."""

        budget = self.p95_regret_budget if p95_regret is None else p95_regret
        minimum_fraction = (
            self.minimum_passing_domain_fraction
            if minimum_passing_fraction is None
            else minimum_passing_fraction
        )
        failures = []
        if self.coverage != 1.0:
            failures.append(
                f"sealed coverage {self.covered_cell_count}/{self.required_cell_count}"
            )
        if self.out_of_scope_cell_count:
            failures.append(
                f"{self.out_of_scope_cell_count} sealed cell(s) lack generic scope"
            )
        if self.unpromoted_domain_count:
            failures.append(
                f"{self.unpromoted_domain_count} generic domain(s) are unpromoted"
            )
        if self.unexercised_rule_count:
            failures.append(f"{self.unexercised_rule_count} generic rule(s) lack sealed exercise")
        if self.verifier_bitwise_failures:
            failures.append(f"{self.verifier_bitwise_failures} verifier byte failure(s)")
        passing_domains = self.passing_domain_count(budget)
        required_domains = self.required_domain_count
        if not domain_promotion_quota_is_satisfied(
            passing_domains,
            required_domains,
            minimum_passing_fraction=minimum_fraction,
        ):
            over_budget = tuple(
                result
                for result in self.domain_results
                if not result.passes(budget)
            )
            worst = max(
                over_budget,
                key=lambda result: (
                    result.p95_observed_regret,
                    result.p95_simultaneous_95pct_upper_regret,
                    result.domain,
                ),
                default=None,
            )
            detail = (
                "; worst p95 observed regret "
                f"{worst.p95_observed_regret:.4%}, p95 simultaneous regret UCB "
                f"{worst.p95_simultaneous_95pct_upper_regret:.4%}"
                if worst is not None
                else ""
            )
            failures.append(
                "sealed domain promotion quota "
                f"{passing_domains}/{required_domains} "
                f"does not reach {100.0 * minimum_fraction:g}%"
                f"{detail}"
            )
        if failures:
            raise ValueError("generic policy is not promotable: " + "; ".join(failures))


def _aggregate_candidate_surface(
    rows,
    candidate_id,
    serial_hash,
    *,
    require_generic_eligible=True,
):
    """Aggregate one nominal candidate without losing physical provenance.

    Generic policy candidates may be shape-resolved formulas whose nominal
    identity differs from the concrete launch recorded by the benchmark.  The
    selected side of certification must remain generic-eligible, while the
    exact oracle's reference side must read the concrete measured candidate
    that formulas intentionally alias.
    """

    grouped = defaultdict(list)
    for row in rows:
        if row.candidate_id != candidate_id:
            continue
        if (
            (require_generic_eligible and not row.generic_eligible)
            or not candidate_is_eligible(row, serial_hash)
        ):
            continue
        grouped[(row.source_format, row.execution_mode)].append(row)
    return {
        surface: {
            "median": statistics.median(item.median_us for item in values),
            "cv": max(item.cv for item in values),
            "samples": sum(item.sample_count for item in values),
            "effective_candidates": {
                item.effective_candidate_id for item in values
            },
            "timing_hashes": {item.timing_sample_hash for item in values},
        }
        for surface, values in grouped.items()
    }


_PARALLEL_CERTIFICATION_POLICY: PolicyIR | None = None
_PARALLEL_CERTIFICATION_POINTS: tuple[
    tuple[
        RuntimeKey,
        str,
        GenericDomain,
        tuple[NativeVNNIObservation, ...],
    ],
    ...,
] = ()
_PARALLEL_CERTIFICATION_SERIAL_HASHES: Mapping[RuntimeKey, str] = {}
_PARALLEL_CERTIFICATION_PROMOTED_DOMAINS: frozenset[GenericDomain] = (
    frozenset()
)


def _evaluate_certification_point(
    policy: PolicyIR,
    point: tuple[
        RuntimeKey,
        str,
        GenericDomain,
        tuple[NativeVNNIObservation, ...],
    ],
    serial_m1_hashes: Mapping[RuntimeKey, str],
    promoted_domains: frozenset[GenericDomain],
):
    """Evaluate one sealed runtime-key/shape cell independently.

    The returned counters preserve the original serial accounting even when a
    selected rule lacks complete candidate evidence. The provisional payload
    deliberately omits the simultaneous confidence bound; the parent computes
    one deterministic Bonferroni correction after it knows the global surface
    comparison count.
    """

    key, shape_group_id, domain, rows = point
    if domain not in promoted_domains:
        return 0, 1, 0, None, None

    rule = policy.resolve_generic(
        domain,
        key.aggregate_n,
        key.k,
        key.launch_k_tiles,
    )
    if rule is None:
        return 1, 0, 0, None, None

    serial_hash = serial_m1_hashes.get(key)
    verifier_failures = 0
    if key.semantic_contract == SemanticContract.VERIFIER_SERIAL_M1_BITWISE:
        verifier_failures = sum(
            1
            for row in rows
            if row.candidate_id == rule.candidate_id
            and (not row.bitwise_equal or not row.repeat_equal)
        )
    eligible_candidates = sorted({
        row.candidate_id
        for row in rows
        if row.generic_eligible and candidate_is_eligible(row, serial_hash)
    })
    surfaces = {(row.source_format, row.execution_mode) for row in rows}
    candidate_surfaces = {
        candidate: _aggregate_candidate_surface(rows, candidate, serial_hash)
        for candidate in eligible_candidates
    }
    complete = [
        candidate
        for candidate in eligible_candidates
        if set(candidate_surfaces[candidate]) == surfaces
    ]
    if rule.candidate_id not in complete or not complete:
        return 1, 0, verifier_failures, rule.identity(), None

    exact_winner = build_exact_winner(
        rows,
        required_surfaces=frozenset(
            SurfaceKey(source_format, execution_mode)
            for source_format, execution_mode in surfaces
        ),
        current_serial_m1_hash=serial_hash,
        runtime_key_override=key,
    )
    exact_candidate = exact_winner.candidate_id
    exact_surface_stats = _aggregate_candidate_surface(
        rows,
        exact_candidate,
        serial_hash,
        require_generic_eligible=False,
    )
    if set(exact_surface_stats) != surfaces:
        return 1, 0, verifier_failures, rule.identity(), None

    selected = candidate_surfaces[rule.candidate_id]
    surface_rows = []
    for surface in sorted(surfaces, key=lambda item: (item[0], item[1].value)):
        exact_stats = exact_surface_stats[surface]
        exact_latency = exact_stats["median"]
        selected_stats = selected[surface]
        ratio = selected_stats["median"] / exact_latency
        same_timing_evidence = bool(
            selected_stats["timing_hashes"] & exact_stats["timing_hashes"]
        )
        variance = 0.0 if same_timing_evidence else (
            selected_stats["cv"] ** 2 / max(1, selected_stats["samples"])
            + exact_stats["cv"] ** 2 / max(1, exact_stats["samples"])
        )
        surface_rows.append((
            ratio - 1.0,
            ratio,
            math.sqrt(variance),
            exact_candidate,
        ))
    return (
        1,
        0,
        verifier_failures,
        rule.identity(),
        (
            key,
            shape_group_id,
            domain,
            rule,
            surfaces,
            surface_rows,
        ),
    )


def _evaluate_parallel_certification_point(index: int):
    """Evaluate one fork-inherited sealed point by stable ordinal."""

    if _PARALLEL_CERTIFICATION_POLICY is None:
        raise RuntimeError("parallel certification policy is unavailable")
    return _evaluate_certification_point(
        _PARALLEL_CERTIFICATION_POLICY,
        _PARALLEL_CERTIFICATION_POINTS[index],
        _PARALLEL_CERTIFICATION_SERIAL_HASHES,
        _PARALLEL_CERTIFICATION_PROMOTED_DOMAINS,
    )


def certify_generic_policy(
    policy: PolicyIR,
    sealed: ObservationCorpus,
    *,
    serial_m1_hashes: Mapping[RuntimeKey, str] | None = None,
    familywise_alpha: float = 0.05,
) -> CertificationReport:
    """Evaluate the frozen generic IR with exact overlays deliberately bypassed.

    A Bonferroni simultaneous normal bound is applied to log latency ratios
    using each aggregate row's coefficient of variation and sample count. This
    is conservative and deterministic. Production adapters may replace it with
    paired bootstrap confirmation, but may not use independent per-cell bounds.
    """

    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must be strictly between zero and one")
    frozen_digest = policy.digest(generic_only=True)
    point_rows = defaultdict(list)
    for row in sealed:
        point_rows[(sealed.runtime_key_for(row), row.shape_group_id)].append(row)

    sealed_count = len(point_rows)
    required_count = 0
    out_of_scope_count = 0
    provisional = []
    verifier_failures = 0
    rule_hits = Counter()
    promoted_domains = {rule.domain for rule in policy.generic_rules}
    unpromoted_domains = set(policy.unpromoted_domains)
    if promoted_domains & unpromoted_domains:
        raise ValueError("policy domain is both promoted and unpromoted")

    points = tuple(
        (
            point_key[0],
            point_key[1],
            sealed.generic_domain_for(rows[0]),
            tuple(rows),
        )
        for point_key, rows in sorted(
            point_rows.items(),
            key=lambda item: item[0],
        )
    )
    requested_workers = int(os.environ.get(
        "LLAMINAR_NATIVE_VNNI_CERTIFICATION_WORKERS",
        str(_physical_core_count()),
    ))
    if requested_workers < 1:
        raise ValueError("certification worker count must be positive")
    worker_count = min(
        requested_workers,
        _physical_core_count(),
        len(points),
    )
    if worker_count > 1 and len(points) >= 8:
        global _PARALLEL_CERTIFICATION_POLICY
        global _PARALLEL_CERTIFICATION_POINTS
        global _PARALLEL_CERTIFICATION_SERIAL_HASHES
        global _PARALLEL_CERTIFICATION_PROMOTED_DOMAINS
        _PARALLEL_CERTIFICATION_POLICY = policy
        _PARALLEL_CERTIFICATION_POINTS = points
        _PARALLEL_CERTIFICATION_SERIAL_HASHES = serial_m1_hashes or {}
        _PARALLEL_CERTIFICATION_PROMOTED_DOMAINS = frozenset(promoted_domains)
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                point_results = tuple(executor.map(
                    _evaluate_parallel_certification_point,
                    range(len(points)),
                ))
        finally:
            _PARALLEL_CERTIFICATION_POLICY = None
            _PARALLEL_CERTIFICATION_POINTS = ()
            _PARALLEL_CERTIFICATION_SERIAL_HASHES = {}
            _PARALLEL_CERTIFICATION_PROMOTED_DOMAINS = frozenset()
    else:
        point_results = tuple(
            _evaluate_certification_point(
                policy,
                point,
                serial_m1_hashes or {},
                frozenset(promoted_domains),
            )
            for point in points
        )

    for (
        required_delta,
        out_of_scope_delta,
        point_verifier_failures,
        rule_identity,
        provisional_cell,
    ) in point_results:
        required_count += required_delta
        out_of_scope_count += out_of_scope_delta
        verifier_failures += point_verifier_failures
        if rule_identity is not None:
            rule_hits[rule_identity] += 1
        if provisional_cell is not None:
            provisional.append(provisional_cell)

    comparison_count = max(1, sum(len(item[4]) for item in provisional))
    z_score = NormalDist().inv_cdf(1.0 - familywise_alpha / comparison_count)
    cells = []
    for key, shape_group_id, domain, rule, surfaces, surface_rows in provisional:
        worst = max(surface_rows, key=lambda item: item[0])
        simultaneous_ucb = max(
            math.exp(math.log(ratio) + z_score * standard_error) - 1.0
            for _, ratio, standard_error, _ in surface_rows
        )
        cells.append(CertificationCell(
            runtime_key=key,
            shape_group_id=shape_group_id,
            domain=domain,
            selected_candidate_id=rule.candidate_id,
            exact_candidate_id=worst[3],
            observed_worst_surface_regret=worst[0],
            simultaneous_95pct_upper_regret=simultaneous_ucb,
            alias_count=len({surface[0] for surface in surfaces}),
            execution_mode_count=len({surface[1] for surface in surfaces}),
        ))

    expected_rules = {
        rule.identity()
        for rule in policy.generic_rules
    }
    unexercised = len(expected_rules - set(rule_hits))
    rule_coverage = tuple(
        CertificationRuleCoverage(
            rule=rule,
            sealed_hit_count=rule_hits[rule.identity()],
        )
        for rule in policy.generic_rules
    )
    return CertificationReport(
        frozen_generic_policy_digest=frozen_digest,
        cells=tuple(cells),
        sealed_cell_count=sealed_count,
        out_of_scope_cell_count=out_of_scope_count,
        required_cell_count=required_count,
        covered_cell_count=len(cells),
        verifier_bitwise_failures=verifier_failures,
        unexercised_rule_count=unexercised,
        unpromoted_domain_count=len(unpromoted_domains),
        rule_coverage=rule_coverage,
    )
