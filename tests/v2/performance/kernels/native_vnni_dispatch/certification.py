"""Generic-only sealed evaluation and simultaneous maximum-regret certificate."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import NormalDist
from typing import Mapping

from .corpus import (
    GenericDomain,
    ObservationCorpus,
    RuntimeKey,
    SurfaceKey,
    generic_domain,
    runtime_key,
)
from .exact_oracle import build_exact_winner, candidate_is_eligible
from .policy_ir import PolicyIR
from .schema import SemanticContract


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
class CertificationReport:
    """Immutable sealed certificate bound to a frozen generic policy digest."""

    frozen_generic_policy_digest: str
    cells: tuple[CertificationCell, ...]
    required_cell_count: int
    covered_cell_count: int
    verifier_bitwise_failures: int
    unexercised_rule_count: int

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
        values = sorted(cell.observed_worst_surface_regret for cell in self.cells)
        if not values:
            return math.inf
        return values[math.ceil(0.95 * len(values)) - 1]

    @property
    def max_simultaneous_95pct_upper_regret(self) -> float:
        return max((cell.simultaneous_95pct_upper_regret for cell in self.cells), default=math.inf)

    def require_promotable(self, *, max_regret: float = 0.03) -> None:
        """Fail unless coverage, byte correctness, and both max gates pass."""

        failures = []
        if self.coverage != 1.0:
            failures.append(
                f"sealed coverage {self.covered_cell_count}/{self.required_cell_count}"
            )
        if self.unexercised_rule_count:
            failures.append(f"{self.unexercised_rule_count} generic rule(s) lack sealed exercise")
        if self.verifier_bitwise_failures:
            failures.append(f"{self.verifier_bitwise_failures} verifier byte failure(s)")
        if self.max_observed_regret > max_regret:
            failures.append(
                f"max observed regret {self.max_observed_regret:.4%} > {max_regret:.2%}"
            )
        if self.max_simultaneous_95pct_upper_regret > max_regret:
            failures.append(
                "simultaneous max-regret UCB "
                f"{self.max_simultaneous_95pct_upper_regret:.4%} > {max_regret:.2%}"
            )
        if failures:
            raise ValueError("generic policy is not promotable: " + "; ".join(failures))


def _aggregate_candidate_surface(rows, candidate_id, serial_hash):
    grouped = defaultdict(list)
    for row in rows:
        if row.effective_candidate_id != candidate_id:
            continue
        if not row.generic_eligible or not candidate_is_eligible(row, serial_hash):
            continue
        grouped[(row.source_format, row.execution_mode)].append(row)
    return {
        surface: {
            "median": statistics.median(item.median_us for item in values),
            "cv": max(item.cv for item in values),
            "samples": sum(item.sample_count for item in values),
        }
        for surface, values in grouped.items()
    }


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
        point_rows[(runtime_key(row), row.shape_group_id)].append(row)

    required_count = len(point_rows)
    provisional = []
    verifier_failures = 0
    rule_hits = Counter()

    for (key, shape_group_id), rows in sorted(point_rows.items(), key=lambda item: item[0]):
        domain = generic_domain(rows[0])
        rule = policy.resolve_generic(domain, key.aggregate_n * key.k)
        if rule is None:
            continue
        rule_hits[(rule.domain, rule.min_work_items, rule.max_work_items, rule.candidate_id)] += 1
        serial_hash = (serial_m1_hashes or {}).get(key)

        eligible_candidates = sorted({
            row.effective_candidate_id
            for row in rows
            if row.generic_eligible and candidate_is_eligible(row, serial_hash)
        })
        if key.semantic_contract == SemanticContract.VERIFIER_SERIAL_M1_BITWISE:
            verifier_failures += sum(
                1
                for row in rows
                if row.effective_candidate_id == rule.candidate_id
                and (not row.bitwise_equal or not row.repeat_equal)
            )
        surfaces = {(row.source_format, row.execution_mode) for row in rows}
        candidate_surfaces = {
            candidate: _aggregate_candidate_surface(rows, candidate, serial_hash)
            for candidate in eligible_candidates
        }
        complete = [
            candidate for candidate in eligible_candidates
            if set(candidate_surfaces[candidate]) == surfaces
        ]
        if rule.candidate_id not in complete or not complete:
            continue

        exact_winner = build_exact_winner(
            rows,
            required_surfaces=frozenset(
                SurfaceKey(source_format, execution_mode)
                for source_format, execution_mode in surfaces
            ),
            current_serial_m1_hash=serial_hash,
        )
        exact_candidate = exact_winner.candidate_id
        if exact_candidate not in candidate_surfaces:
            continue
        selected = candidate_surfaces[rule.candidate_id]
        exact_surface_stats = candidate_surfaces[exact_candidate]
        surface_rows = []
        for surface in sorted(surfaces, key=lambda item: (item[0], item[1].value)):
            exact_stats = exact_surface_stats[surface]
            exact_latency = exact_stats["median"]
            selected_stats = selected[surface]
            ratio = selected_stats["median"] / exact_latency
            variance = 0.0 if rule.candidate_id == exact_candidate else (
                selected_stats["cv"] ** 2 / max(1, selected_stats["samples"])
                + exact_stats["cv"] ** 2 / max(1, exact_stats["samples"])
            )
            surface_rows.append((
                ratio - 1.0,
                ratio,
                math.sqrt(variance),
                exact_candidate,
            ))
        provisional.append((key, shape_group_id, domain, rule, surfaces, surface_rows))

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
        (rule.domain, rule.min_work_items, rule.max_work_items, rule.candidate_id)
        for rule in policy.generic_rules
    }
    unexercised = len(expected_rules - set(rule_hits))
    return CertificationReport(
        frozen_generic_policy_digest=frozen_digest,
        cells=tuple(cells),
        required_cell_count=required_count,
        covered_cell_count=len(cells),
        verifier_bitwise_failures=verifier_failures,
        unexercised_rule_count=unexercised,
    )
