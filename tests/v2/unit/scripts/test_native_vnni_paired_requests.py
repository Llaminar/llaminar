#!/usr/bin/env python3
"""Regression tests for automated NativeVNNI paired request planning."""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.corpus import (  # noqa: E402
    ObservationCorpus,
    generic_domain,
    runtime_key,
)
from native_vnni_dispatch.candidate_registry import (  # noqa: E402
    cuda_native_vnni_gemv_registry,
)
from native_vnni_dispatch.cuda_shape_resolved import (  # noqa: E402
    project_cuda_shape_resolved_candidates,
    resolve_cuda_concrete_candidate_id,
)
from native_vnni_dispatch.paired_confirmation import (  # noqa: E402
    PairedCellKey,
    PairedTimingComparison,
)
from native_vnni_dispatch.paired_requests import (  # noqa: E402
    PAIRED_REQUEST_SCHEMA_VERSION,
    _effective_candidate,
    _policy_from_cached_validations,
    build_paired_request_plan,
    projected_domain_cache_key,
    projected_domain_cache_keys,
    projected_cuda_domain_corpora,
    write_json,
    write_request_shards,
)
from native_vnni_dispatch import paired_requests as paired_requests_module  # noqa: E402
from native_vnni_dispatch.schema import (  # noqa: E402
    Backend,
    ExecutionMode,
    P95_REGRET_BUDGET,
    SemanticContract,
)
from native_vnni_dispatch.segmented_policy import (  # noqa: E402
    BoundaryPlacement,
    CrossValidationCell,
    DomainCrossValidation,
    DomainPromotionDiagnostic,
    FeaturePolicy,
    GenericPolicy,
    PolicyFitCache,
    ProfilerInfluence,
    load_cached_cross_validations,
)
from test_native_vnni_common_dispatch_policy import observation  # noqa: E402


class NativeVNNIPairedRequestsTest(unittest.TestCase):
    """Prove formula resolution, tournament closure, and terminal states."""

    def test_parallel_projected_domain_keys_match_serial_order(self) -> None:
        """Parallel precomputation must preserve every projected cache key."""

        rows = ObservationCorpus(tuple(
            observation(
                candidate="cuda.test.projected",
                family="projected",
                shape_group=f"shape-{index}",
                backend=Backend.CUDA,
                mode=mode,
                n=n,
                k=k,
                m=1,
                contract=SemanticContract.FAST,
            )
            for index, (mode, n, k) in enumerate((
                (ExecutionMode.EAGER, 4096, 1024),
                (ExecutionMode.GRAPH_CAPTURED, 4096, 1024),
                (ExecutionMode.EAGER, 1024, 4096),
                (ExecutionMode.GRAPH_CAPTURED, 1024, 4096),
            ))
        ))
        with mock.patch.object(
            paired_requests_module,
            "_physical_core_count",
            return_value=1,
        ):
            serial = projected_domain_cache_keys(rows)
        with mock.patch.object(
            paired_requests_module,
            "_physical_core_count",
            return_value=2,
        ):
            parallel = projected_domain_cache_keys(rows)

        self.assertEqual(tuple(parallel), tuple(serial))
        self.assertEqual(parallel, serial)

    def test_batched_cuda_projection_matches_every_domain_projection(self) -> None:
        """Known-domain projection skips duplicate indexing without drift."""

        formula = next(
            candidate
            for candidate in cuda_native_vnni_gemv_registry().entries
            if candidate.config_json.get("family") == "kpar_formula"
        )
        rows = []
        for index, (mode, n, k) in enumerate((
            (ExecutionMode.EAGER, 1152, 3584),
            (ExecutionMode.GRAPH_CAPTURED, 1536, 4096),
            (ExecutionMode.EAGER, 2048, 5120),
            (ExecutionMode.GRAPH_CAPTURED, 2560, 6144),
        )):
            rows.append(observation(
                candidate=resolve_cuda_concrete_candidate_id(formula, n, k),
                family="kpar",
                shape_group=f"batched-projection-{index}",
                backend=Backend.CUDA,
                mode=mode,
                n=n,
                k=k,
                m=1,
                contract=SemanticContract.FAST,
            ))
        development = ObservationCorpus(tuple(rows))
        with mock.patch.object(
            ObservationCorpus,
            "generic_domain_for",
            side_effect=AssertionError("projected rows rebuilt domain keys"),
        ):
            batched = projected_cuda_domain_corpora(development)

        self.assertEqual(tuple(batched), development.generic_domains())
        for domain in development.generic_domains():
            serial = project_cuda_shape_resolved_candidates(
                development.rows_for_generic_domain(domain)
            )
            self.assertEqual(
                batched[domain].observations,
                serial.observations,
            )

    @staticmethod
    def _fixture(
        *,
        same_effective: bool = False,
        direct_ratio: float | None = None,
    ):
        """Build one failed held-out CUDA cell and optional direct evidence."""

        selected_nominal = "cuda.test.formula.selected"
        exact_nominal = "cuda.test.formula.exact"
        selected_effective = "cuda.nvnni.decode.fast_m1.kpar.tn64.cpt2.kb7"
        exact_effective = (
            selected_effective
            if same_effective
            else "cuda.nvnni.decode.fast_m1.direct.tn64.cpt1"
        )
        common = {
            "source_format": "Q4_0",
            "shape_group": "cuda-decode:PlannerShape:n1152:k3584",
            "shape_name": "PlannerShape",
            "n": 1152,
            "k": 3584,
            "m": 1,
            "mode": ExecutionMode.EAGER,
            "contract": SemanticContract.FAST,
            "backend": Backend.CUDA,
        }
        selected = dataclasses.replace(
            observation(candidate=selected_nominal, latency_us=10.5, **common),
            effective_candidate_id=selected_effective,
            observed_candidate_id=selected_effective,
        )
        exact = dataclasses.replace(
            observation(candidate=exact_nominal, latency_us=10.0, **common),
            effective_candidate_id=exact_effective,
            observed_candidate_id=exact_effective,
        )
        corpus = ObservationCorpus((selected, exact))
        key = runtime_key(selected)
        domain = generic_domain(selected)
        cell = CrossValidationCell(
            runtime_key=key,
            shape_group_id=selected.shape_group_id,
            selected_candidate_id=selected_nominal,
            exact_candidate_id=exact_nominal,
            observed_broad_regret=0.05,
        )
        validation = DomainCrossValidation(
            domain=domain,
            selected_feature_policy=FeaturePolicy.CONTINUOUS,
            selected_max_leaves=1,
            selected_boundary_placement=BoundaryPlacement.MIDPOINT,
            selected_profiler_influence=ProfilerInfluence.MEASURED_ONLY,
            fold_count=5,
            shape_group_count=5,
            required_point_count=1,
            covered_point_count=1,
            max_regret=0.05,
            p95_regret=0.05,
            mean_regret=0.05,
            worst_shape_group_id=selected.shape_group_id,
            worst_aggregate_n=selected.aggregate_n,
            worst_k=selected.k,
            worst_selected_candidate_id=selected_nominal,
            worst_exact_candidate_id=exact_nominal,
            cells=(cell,),
        )
        policy = GenericPolicy(
            rules=(),
            unpromoted_domains=(domain,),
            cross_validation=(validation,),
        )
        pair_key = PairedCellKey(
            backend="cuda",
            source_format="Q4_0",
            source_codebook=0,
            execution_codebook=0,
            shape="PlannerShape",
            execution_mode="eager",
            m=1,
            n=1152,
            k=3584,
        )
        comparisons = {}
        if direct_ratio is not None:
            comparisons[pair_key] = (PairedTimingComparison(
                key=pair_key,
                selected_effective_candidate_id=selected_effective,
                exact_effective_candidate_id=exact_effective,
                selected_to_exact_median_ratio=direct_ratio,
                pair_count=30,
            ),)
        return corpus, policy, comparisons

    def test_missing_direct_edge_emits_forceable_concrete_request(self) -> None:
        """Nominal formula IDs never cross the C++ trainer boundary."""

        corpus, policy, comparisons = self._fixture()
        plan = build_paired_request_plan(corpus, policy, comparisons)
        self.assertEqual(plan.status, "pending")
        self.assertEqual(len(plan.requests), 1)
        request = plan.requests[0]
        self.assertEqual(
            request.selected_candidate_id,
            "cuda.nvnni.decode.fast_m1.kpar.tn64.cpt2.kb7",
        )
        self.assertNotIn("formula", request.selected_candidate_id)
        self.assertEqual(request.reason, "missing_direct_tournament_edge")

    def test_selected_candidate_star_covers_every_forceable_launch(self) -> None:
        """One refinement batch directly compares every candidate to its anchor."""

        corpus, policy, comparisons = self._fixture()
        exemplar = corpus.observations[0]
        frontier_nominal = "cuda.test.formula.frontier"
        frontier_effective = "cuda.nvnni.decode.fast_m1.kpar.tn128.cpt1.kb5"
        frontier_row = dataclasses.replace(
            exemplar,
            candidate_id=frontier_nominal,
            effective_candidate_id=frontier_effective,
            observed_candidate_id=frontier_effective,
        )
        corpus = ObservationCorpus((*corpus.observations, frontier_row))
        selected = policy.cross_validation[0].cells[0]
        frontier_cell = dataclasses.replace(
            selected,
            selected_candidate_id=frontier_nominal,
            observed_broad_regret=0.06,
        )
        validation = dataclasses.replace(
            policy.cross_validation[0],
            competitive_cells=(selected, frontier_cell),
        )
        policy = dataclasses.replace(policy, cross_validation=(validation,))

        plan = build_paired_request_plan(corpus, policy, comparisons)

        self.assertEqual(len(plan.requests), 2)
        self.assertEqual(plan.tournament_completion_request_count, 1)
        completion_requests = [
            request
            for request in plan.requests
            if request.reason == "complete_tournament_star_edge"
        ]
        self.assertEqual(len(completion_requests), 1)
        self.assertEqual(
            completion_requests[0].exact_candidate_id,
            frontier_effective,
        )

    def test_indirect_graph_path_does_not_replace_direct_star_edge(self) -> None:
        """A directionally mixed graph cannot certify a selected launch."""

        corpus, policy, _ = self._fixture()
        selected_row, exact_row = corpus.observations
        frontier_nominal = "cuda.test.formula.connected-frontier"
        frontier_effective = "cuda.nvnni.decode.fast_m1.kpar.tn128.cpt1.kb5"
        frontier_row = dataclasses.replace(
            selected_row,
            candidate_id=frontier_nominal,
            effective_candidate_id=frontier_effective,
            observed_candidate_id=frontier_effective,
        )
        corpus = ObservationCorpus((*corpus.observations, frontier_row))
        selected_cell = policy.cross_validation[0].cells[0]
        frontier_cell = dataclasses.replace(
            selected_cell,
            selected_candidate_id=frontier_nominal,
            observed_broad_regret=0.04,
        )
        validation = dataclasses.replace(
            policy.cross_validation[0],
            competitive_cells=(selected_cell, frontier_cell),
        )
        policy = dataclasses.replace(policy, cross_validation=(validation,))
        key = PairedCellKey(
            backend="cuda",
            source_format="Q4_0",
            source_codebook=0,
            execution_codebook=0,
            shape="PlannerShape",
            execution_mode="eager",
            m=1,
            n=1152,
            k=3584,
        )
        comparisons = {key: (
            PairedTimingComparison(
                key=key,
                selected_effective_candidate_id=(
                    selected_row.effective_candidate_id
                ),
                exact_effective_candidate_id=exact_row.effective_candidate_id,
                selected_to_exact_median_ratio=1.05,
                pair_count=30,
            ),
            PairedTimingComparison(
                key=key,
                selected_effective_candidate_id=frontier_effective,
                exact_effective_candidate_id=(
                    exact_row.effective_candidate_id
                ),
                selected_to_exact_median_ratio=1.01,
                pair_count=30,
            ),
        )}

        plan = build_paired_request_plan(corpus, policy, comparisons)

        self.assertEqual(plan.status, "pending")
        self.assertEqual(len(plan.requests), 1)
        self.assertEqual(
            plan.requests[0].exact_candidate_id,
            frontier_effective,
        )
        self.assertEqual(plan.tournament_completion_request_count, 1)

    def test_complete_star_is_reused_when_later_fit_changes_selection(self) -> None:
        """A later fitted winner cannot mint a second tournament star."""

        corpus, policy, comparisons = self._fixture(direct_ratio=0.90)
        original = policy.cross_validation[0].cells[0]
        later_selection = dataclasses.replace(
            original,
            selected_candidate_id=original.exact_candidate_id,
            exact_candidate_id=original.selected_candidate_id,
            observed_broad_regret=0.10,
        )
        validation = dataclasses.replace(
            policy.cross_validation[0],
            cells=(later_selection,),
            p95_regret=0.10,
            max_regret=0.10,
            mean_regret=0.10,
        )
        policy = dataclasses.replace(policy, cross_validation=(validation,))

        plan = build_paired_request_plan(corpus, policy, comparisons)

        self.assertEqual(plan.status, "confirmed_failure")
        self.assertFalse(plan.requests)
        self.assertEqual(len(plan.confirmed_cv_misses), 1)
        issue = plan.confirmed_cv_misses[0]
        self.assertEqual(
            issue.reason,
            "complete_tournament_star_confirms_over_budget",
        )
        self.assertAlmostEqual(issue.direct_paired_regret, 1.0 / 0.90 - 1.0)

    def test_formula_request_resolves_from_compact_direct_corpus(self) -> None:
        """Paired planning does not require materialized formula-alias rows."""

        formula = next(
            candidate
            for candidate in cuda_native_vnni_gemv_registry().entries
            if candidate.config_json.get("family") == "kpar_formula"
        )
        n = 1152
        k = 3584
        concrete = resolve_cuda_concrete_candidate_id(formula, n, k)
        measured = observation(
            backend=Backend.CUDA,
            contract=SemanticContract.FAST,
            mode=ExecutionMode.EAGER,
            m=1,
            n=n,
            k=k,
            candidate=concrete,
            shape_group="compact-formula-source",
        )

        self.assertEqual(
            _effective_candidate((measured,), formula.candidate_id),
            concrete,
        )

    def test_green_selected_model_does_not_measure_failing_alternates(self) -> None:
        """Frontier closure ends as soon as the selected domain is economical."""

        corpus, policy, comparisons = self._fixture()
        failed = policy.cross_validation[0].cells[0]
        passing = dataclasses.replace(failed, observed_broad_regret=0.01)
        alternate = dataclasses.replace(
            failed,
            selected_candidate_id=failed.exact_candidate_id,
            exact_candidate_id=failed.selected_candidate_id,
            observed_broad_regret=0.05,
        )
        validation = dataclasses.replace(
            policy.cross_validation[0],
            max_regret=0.01,
            p95_regret=0.01,
            cells=(passing,),
            competitive_cells=(alternate,),
        )
        policy = dataclasses.replace(policy, cross_validation=(validation,))

        plan = build_paired_request_plan(corpus, policy, comparisons)

        self.assertEqual(plan.status, "green")
        self.assertFalse(plan.requests)
        self.assertEqual(plan.tournament_completion_request_count, 0)

    def test_final_publication_failure_cannot_report_green(self) -> None:
        """A CV-covered domain still blocks planning when publication failed."""

        corpus, policy, comparisons = self._fixture()
        validation = policy.cross_validation[0]
        passing_cell = dataclasses.replace(
            validation.cells[0],
            observed_broad_regret=0.01,
        )
        passing = dataclasses.replace(
            validation,
            max_regret=0.01,
            p95_regret=0.01,
            mean_regret=0.01,
            cells=(passing_cell,),
        )
        policy = dataclasses.replace(
            policy,
            cross_validation=(passing,),
            unpromoted_domains=(passing.domain,),
            promotion_diagnostics=(
                DomainPromotionDiagnostic(
                    domain=passing.domain,
                    rejection_stage="final_fit_p95",
                    cv_required_point_count=1,
                    cv_covered_point_count=1,
                    cv_p95_regret=0.01,
                    cv_max_regret=0.01,
                    final_cross_fitted_p95_regret=0.08,
                    final_rule_count=2,
                    final_worst_p95_regret=0.08,
                    final_worst_max_regret=0.08,
                    final_worst_shape_group_id=passing_cell.shape_group_id,
                    final_worst_aggregate_n=(
                        passing_cell.runtime_key.aggregate_n
                    ),
                    final_worst_k=passing_cell.runtime_key.k,
                    final_worst_candidate_id=(
                        passing_cell.selected_candidate_id
                    ),
                ),
            ),
        )

        plan = build_paired_request_plan(corpus, policy, comparisons)

        self.assertEqual(plan.status, "insufficient_cross_validation")
        self.assertTrue(plan.unvalidated_domains)
        self.assertFalse(plan.requests)
        self.assertEqual(len(plan.promotion_diagnostics), 1)
        diagnostic = plan.promotion_diagnostics[0]
        self.assertTrue(diagnostic.blocking)
        self.assertEqual(diagnostic.rejection_stage, "final_fit_p95")
        self.assertEqual(diagnostic.selected_max_leaves, 1)
        self.assertEqual(diagnostic.final_rule_count, 2)
        self.assertEqual(diagnostic.final_cross_fitted_p95_regret, 0.08)
        self.assertEqual(
            diagnostic.final_worst_aggregate_n,
            passing_cell.runtime_key.aggregate_n,
        )
        report = plan.report_mapping()
        self.assertEqual(
            report["promotion_diagnostics"][0]["final_worst_candidate_id"],
            passing_cell.selected_candidate_id,
        )

    def test_p95_passing_domain_ignores_isolated_over_budget_tail(self) -> None:
        """Worst-cell regret is diagnostic after the domain p95 gate passes."""

        corpus, policy, comparisons = self._fixture()
        validation = dataclasses.replace(
            policy.cross_validation[0],
            max_regret=0.50,
            p95_regret=0.049,
        )
        policy = dataclasses.replace(policy, cross_validation=(validation,))

        plan = build_paired_request_plan(corpus, policy, comparisons)

        self.assertEqual(plan.status, "green")
        self.assertFalse(plan.requests)
        self.assertFalse(plan.confirmed_cv_misses)
        self.assertEqual(plan.failed_cv_cell_count, 0)
        self.assertEqual(plan.passing_cv_cell_count, 1)

    def test_p95_equal_to_budget_still_opens_refinement(self) -> None:
        """Installation requires p95 to be strictly lower than five percent."""

        corpus, policy, comparisons = self._fixture()
        boundary_cell = dataclasses.replace(
            policy.cross_validation[0].cells[0],
            observed_broad_regret=P95_REGRET_BUDGET,
        )
        validation = dataclasses.replace(
            policy.cross_validation[0],
            max_regret=P95_REGRET_BUDGET,
            p95_regret=P95_REGRET_BUDGET,
            cells=(boundary_cell,),
        )
        policy = dataclasses.replace(policy, cross_validation=(validation,))

        plan = build_paired_request_plan(corpus, policy, comparisons)

        self.assertEqual(plan.status, "pending")
        self.assertEqual(len(plan.requests), 1)

    def test_projection_identity_binds_candidate_registry(self) -> None:
        """A formula-inventory edit invalidates every derived domain cache key."""

        corpus, _, _ = self._fixture()
        domain = corpus.generic_domains()[0]
        original = projected_domain_cache_key(corpus, domain)
        registry = mock.Mock()
        registry.digest.return_value = "sha256:different-registry"
        with mock.patch(
            "native_vnni_dispatch.paired_requests.cuda_native_vnni_gemv_registry",
            return_value=registry,
        ):
            changed = projected_domain_cache_key(corpus, domain)
        self.assertNotEqual(changed, original)

    def test_direct_over_budget_edge_blocks_before_opening_sealed_data(self) -> None:
        """A measured CV miss cannot be mislabeled green or spend holdout data."""

        corpus, policy, comparisons = self._fixture(direct_ratio=1.05)
        plan = build_paired_request_plan(corpus, policy, comparisons)
        self.assertEqual(plan.status, "confirmed_failure")
        self.assertFalse(plan.requests)
        self.assertEqual(len(plan.confirmed_cv_misses), 1)

    def test_pending_edges_take_precedence_over_current_confirmed_cv_miss(self) -> None:
        """The loop must collect evidence that can still change the tree."""

        corpus, policy, comparisons = self._fixture(direct_ratio=1.05)
        failed = build_paired_request_plan(corpus, policy, comparisons)
        corpus2, policy2, _ = self._fixture()
        pending = build_paired_request_plan(corpus2, policy2, {})
        combined = dataclasses.replace(
            failed,
            requests=pending.requests,
        )
        self.assertEqual(combined.status, "pending")

    def test_direct_pass_conflicting_with_tournament_stops_without_retiming(self) -> None:
        """A measured graph conflict cannot mint unbounded duplicate requests."""

        corpus, policy, comparisons = self._fixture(direct_ratio=1.01)
        plan = build_paired_request_plan(corpus, policy, comparisons)
        self.assertEqual(plan.status, "evidence_conflict")
        self.assertEqual(len(plan.conflicts), 1)
        self.assertFalse(plan.requests)
        self.assertEqual(
            plan.conflicts[0].reason,
            "tournament_fit_conflicts_with_complete_star",
        )

    def test_true_source_aliases_share_one_runtime_tournament_edge(self) -> None:
        """One packed CPU launch gets one pooled verdict and one request."""

        rows = []
        for source_format in ("IQ4_NL", "IQ4_XS"):
            rows.extend((
                observation(
                    backend=Backend.CPU,
                    contract=SemanticContract.FAST,
                    mode=ExecutionMode.EAGER,
                    m=1,
                    source_format=source_format,
                    candidate="cpu.selected",
                    family="selected",
                    shape_group="cpu-alias-shape",
                    latency_us=11.0,
                ),
                observation(
                    backend=Backend.CPU,
                    contract=SemanticContract.FAST,
                    mode=ExecutionMode.EAGER,
                    m=1,
                    source_format=source_format,
                    candidate="cpu.exact",
                    family="exact",
                    shape_group="cpu-alias-shape",
                    latency_us=10.0,
                ),
            ))
        corpus = ObservationCorpus(rows)
        exemplar = rows[0]
        cell = CrossValidationCell(
            runtime_key=runtime_key(exemplar),
            shape_group_id=exemplar.shape_group_id,
            selected_candidate_id="cpu.selected",
            exact_candidate_id="cpu.exact",
            observed_broad_regret=0.10,
        )
        domain = generic_domain(exemplar)
        validation = DomainCrossValidation(
            domain=domain,
            selected_feature_policy=FeaturePolicy.CONTINUOUS,
            selected_max_leaves=1,
            selected_boundary_placement=BoundaryPlacement.MIDPOINT,
            selected_profiler_influence=ProfilerInfluence.MEASURED_ONLY,
            fold_count=5,
            shape_group_count=5,
            required_point_count=1,
            covered_point_count=1,
            max_regret=0.10,
            p95_regret=0.10,
            mean_regret=0.10,
            worst_shape_group_id=exemplar.shape_group_id,
            worst_aggregate_n=exemplar.aggregate_n,
            worst_k=exemplar.k,
            worst_selected_candidate_id="cpu.selected",
            worst_exact_candidate_id="cpu.exact",
            cells=(cell,),
        )
        policy = GenericPolicy(
            rules=(),
            unpromoted_domains=(domain,),
            cross_validation=(validation,),
        )
        comparisons = {}
        for source_format, ratio in (("IQ4_NL", 1.10), ("IQ4_XS", 0.90)):
            row = next(item for item in rows if item.source_format == source_format)
            key = PairedCellKey(
                backend="cpu",
                source_format=source_format,
                source_codebook=row.source_codebook_id,
                execution_codebook=row.runtime_codebook_id,
                shape=row.shape_name,
                execution_mode=row.execution_mode.value,
                m=row.m,
                n=row.aggregate_n,
                k=row.k,
                architecture_class=row.architecture_class,
            )
            comparisons[key] = (PairedTimingComparison(
                key=key,
                selected_effective_candidate_id="cpu.selected",
                exact_effective_candidate_id="cpu.exact",
                selected_to_exact_median_ratio=ratio,
                pair_count=30,
            ),)

        plan = build_paired_request_plan(corpus, policy, comparisons)

        self.assertEqual(plan.status, "green")
        self.assertFalse(plan.confirmed_cv_misses)
        self.assertFalse(plan.conflicts)
        self.assertFalse(plan.requests)
        self.assertEqual(plan.equivalent_effective_surface_count, 1)

    def test_paired_refinement_stops_at_the_95_percent_domain_quota(self) -> None:
        """One explicit performance exception cannot keep refinement open."""

        corpus, policy, _ = self._fixture()
        failing = policy.cross_validation[0]
        passing_cell = dataclasses.replace(
            failing.cells[0],
            observed_broad_regret=0.01,
        )
        passing = dataclasses.replace(
            failing,
            max_regret=0.01,
            p95_regret=0.01,
            mean_regret=0.01,
            cells=(passing_cell,),
        )
        quota_policy = dataclasses.replace(
            policy,
            cross_validation=(failing,) + (passing,) * 19,
        )

        plan = build_paired_request_plan(corpus, quota_policy, {})

        self.assertEqual(plan.status, "green")
        self.assertFalse(plan.requests)
        self.assertFalse(plan.confirmed_cv_misses)
        self.assertEqual(plan.failed_cv_cell_count, 1)

    def test_nominal_choices_with_same_effective_launch_need_no_timing(self) -> None:
        """Two formulas resolving to one launch have exactly zero schedule regret."""

        corpus, policy, comparisons = self._fixture(same_effective=True)
        plan = build_paired_request_plan(corpus, policy, comparisons)
        self.assertEqual(plan.status, "green")
        self.assertFalse(plan.requests)
        self.assertEqual(plan.equivalent_effective_surface_count, 1)

    def test_over_budget_compromise_requests_its_broad_alias_competitor(self) -> None:
        """Selected==exact cannot hide a source-alias performance conflict."""

        corpus, policy, comparisons = self._fixture(same_effective=True)
        competitor = dataclasses.replace(
            corpus.observations[0],
            candidate_id="cuda.test.physical-competitor",
            effective_candidate_id=(
                "cuda.nvnni.decode.fast_m1.direct.tn128.cpt1"
            ),
            observed_candidate_id=(
                "cuda.nvnni.decode.fast_m1.direct.tn128.cpt1"
            ),
            min_us=9.0,
            median_us=9.0,
            p95_us=9.0,
        )
        corpus = ObservationCorpus((*corpus.observations, competitor))

        plan = build_paired_request_plan(corpus, policy, comparisons)

        self.assertEqual(plan.status, "pending")
        self.assertEqual(len(plan.requests), 1)
        self.assertEqual(
            plan.requests[0].exact_candidate_id,
            competitor.effective_candidate_id,
        )

    def test_minimax_exact_does_not_hide_worst_alias_regret_witness(self) -> None:
        """Pair the selected route with the candidate causing its max regret."""

        corpus, policy, comparisons = self._fixture()
        selected, compromise = corpus.observations
        witness = dataclasses.replace(
            selected,
            candidate_id="cuda.test.worst-alias-witness",
            effective_candidate_id=(
                "cuda.nvnni.decode.fast_m1.direct.tn128.cpt1"
            ),
            observed_candidate_id=(
                "cuda.nvnni.decode.fast_m1.direct.tn128.cpt1"
            ),
            min_us=9.0,
            median_us=9.0,
            p95_us=9.0,
        )
        # The CV oracle remains the minimax compromise, but its selected
        # candidate's reported max regret is actually witnessed by the third
        # physical launch above. The planner must measure that causal edge.
        self.assertNotEqual(
            policy.cross_validation[0].cells[0].exact_candidate_id,
            witness.candidate_id,
        )
        corpus = ObservationCorpus((selected, compromise, witness))

        plan = build_paired_request_plan(corpus, policy, comparisons)

        self.assertEqual(plan.status, "pending")
        self.assertEqual(len(plan.requests), 2)
        self.assertEqual(
            {request.exact_candidate_id for request in plan.requests},
            {
                compromise.effective_candidate_id,
                witness.effective_candidate_id,
            },
        )

    def test_written_manifest_has_exact_trainer_schema_and_count(self) -> None:
        """The JSON handoff preserves provenance and concrete request identity."""

        corpus, policy, comparisons = self._fixture()
        plan = build_paired_request_plan(corpus, policy, comparisons)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.json"
            write_json(path, plan.request_manifest_mapping())
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["schema_version"],
            PAIRED_REQUEST_SCHEMA_VERSION,
        )
        self.assertEqual(payload["backend"], "cuda")
        self.assertEqual(payload["request_count"], 1)
        self.assertEqual(len(payload["requests"]), 1)
        self.assertEqual(
            payload["requests"][0]["architecture_class"],
            "unit-cuda-native-vnni-v1",
        )

    def test_request_shards_are_bounded_and_architecture_homogeneous(self) -> None:
        """Producer processes receive only one physical backend architecture."""

        corpus, policy, comparisons = self._fixture()
        plan = build_paired_request_plan(corpus, policy, comparisons)
        repeated = dataclasses.replace(
            plan,
            requests=(
                plan.requests[0],
                dataclasses.replace(
                    plan.requests[0],
                    request_id="second-request",
                    shape="SecondShape",
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_request_shards(
                Path(directory), repeated, max_requests_per_shard=1
            )
            payloads = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in paths
            ]
            index = json.loads(
                (Path(directory) / "index.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(paths), 2)
        self.assertTrue(all(item["request_count"] == 1 for item in payloads))
        self.assertEqual(index["request_count"], 2)
        self.assertEqual(index["shard_count"], 2)
        self.assertEqual(
            {item["architecture_class"] for item in index["shards"]},
            {"unit-cuda-native-vnni-v1"},
        )
        self.assertEqual(
            index["schema_version"],
            "native-vnni-paired-request-shards-v2-content-addressed",
        )
        self.assertTrue(all(
            item["request_digest"] in item["path"]
            for item in index["shards"]
        ))

    def test_request_shards_keep_one_runtime_tournament_atomic(self) -> None:
        """A packing target never splits direct-star edges across processes."""

        corpus, policy, comparisons = self._fixture()
        plan = build_paired_request_plan(corpus, policy, comparisons)
        second_edge = dataclasses.replace(
            plan.requests[0],
            request_id="same-cell-second-edge",
            exact_candidate_id=(
                "cuda.nvnni.decode.fast_m1.direct.tn128.cpt1"
            ),
            reason="complete_tournament_star_edge",
        )
        atomic = dataclasses.replace(
            plan,
            requests=(plan.requests[0], second_edge),
        )

        with tempfile.TemporaryDirectory() as directory:
            paths = write_request_shards(
                Path(directory), atomic, max_requests_per_shard=1
            )
            payload = json.loads(paths[0].read_text(encoding="utf-8"))

        self.assertEqual(len(paths), 1)
        self.assertEqual(payload["request_count"], 2)
        self.assertEqual(
            {item["shape_group_id"] for item in payload["requests"]},
            {plan.requests[0].shape_group_id},
        )

    def test_request_shard_name_changes_with_request_content(self) -> None:
        """A stale CSV cannot impersonate evidence for a replanned shard."""

        corpus, policy, comparisons = self._fixture()
        plan = build_paired_request_plan(corpus, policy, comparisons)
        changed = dataclasses.replace(
            plan,
            requests=(dataclasses.replace(
                plan.requests[0],
                request_id="content-addressed-replan",
            ),),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_path = write_request_shards(
                root, plan, max_requests_per_shard=16
            )[0]
            stale_evidence = original_path.with_name(
                original_path.name.removesuffix(".requests.json") + ".csv"
            )
            stale_evidence.write_text("retained evidence\n", encoding="utf-8")
            changed_path = write_request_shards(
                root, changed, max_requests_per_shard=16
            )[0]
            changed_evidence = changed_path.with_name(
                changed_path.name.removesuffix(".requests.json") + ".csv"
            )

        self.assertNotEqual(original_path.name, changed_path.name)
        self.assertNotEqual(stale_evidence.name, changed_evidence.name)

    def test_cpu_request_retains_complete_isa_architecture_class(self) -> None:
        """CPU paired edges cannot cross build/runtime/thread fit domains."""

        architecture = (
            "unit-x86|build=AVX512|runtime=AVX2|threads=28"
        )
        common = {
            "source_format": "Q4_0",
            "shape_group": "cpu-decode:UnitShape:n896:k896",
            "shape_name": "UnitShape",
            "n": 896,
            "k": 896,
            "m": 1,
            "mode": ExecutionMode.EAGER,
            "contract": SemanticContract.FAST,
            "backend": Backend.CPU,
        }
        selected = dataclasses.replace(
            observation(
                candidate="cpu.nvnni.decode.n_chunk_grid.nbc1",
                latency_us=10.6,
                **common,
            ),
            architecture_class=architecture,
        )
        exact = dataclasses.replace(
            observation(
                candidate="cpu.nvnni.decode.n_chunk_grid.nbc2",
                latency_us=10.0,
                **common,
            ),
            architecture_class=architecture,
        )
        corpus = ObservationCorpus((selected, exact)).with_collapsed_aspect_domains()
        key = runtime_key(selected)
        domain = corpus.generic_domains()[0]
        cell = CrossValidationCell(
            runtime_key=key,
            shape_group_id=selected.shape_group_id,
            selected_candidate_id=selected.candidate_id,
            exact_candidate_id=exact.candidate_id,
            observed_broad_regret=0.06,
        )
        validation = DomainCrossValidation(
            domain=domain,
            selected_feature_policy=FeaturePolicy.CONTINUOUS,
            selected_max_leaves=1,
            selected_boundary_placement=BoundaryPlacement.MIDPOINT,
            selected_profiler_influence=ProfilerInfluence.MEASURED_ONLY,
            fold_count=5,
            shape_group_count=5,
            required_point_count=1,
            covered_point_count=1,
            max_regret=0.06,
            p95_regret=0.06,
            mean_regret=0.06,
            worst_shape_group_id=selected.shape_group_id,
            worst_aggregate_n=selected.aggregate_n,
            worst_k=selected.k,
            worst_selected_candidate_id=selected.candidate_id,
            worst_exact_candidate_id=exact.candidate_id,
            cells=(cell,),
        )
        policy = GenericPolicy(
            rules=(),
            unpromoted_domains=(domain,),
            cross_validation=(validation,),
        )

        plan = build_paired_request_plan(corpus, policy, {})

        self.assertEqual(plan.backend, "cpu")
        self.assertEqual(len(plan.requests), 1)
        self.assertEqual(plan.requests[0].architecture_class, architecture)
        self.assertEqual(
            plan.request_manifest_mapping()["backend"], "cpu"
        )

    def test_grouped_cpu_verifier_cell_emits_a_paired_request(self) -> None:
        """Grouped M>1 refinement uses the real verifier producer surface."""

        architecture = "unit-x86|build=AVX512|runtime=AVX512|threads=28"
        common = {
            "source_format": "Q4_0",
            "shape_group": "cpu-verifier:UnitShape:n896:k896",
            "shape_name": "UnitShape",
            "n": 896,
            "k": 896,
            "m": 4,
            "mode": ExecutionMode.EAGER,
            "contract": SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
            "backend": Backend.CPU,
        }
        selected = dataclasses.replace(
            observation(
                candidate="cpu.nvnni.verifier.pairwise",
                latency_us=10.6,
                **common,
            ),
            architecture_class=architecture,
        )
        exact = dataclasses.replace(
            observation(
                candidate="cpu.nvnni.verifier.wide_rows",
                latency_us=10.0,
                **common,
            ),
            architecture_class=architecture,
        )
        corpus = ObservationCorpus(
            (selected, exact)
        ).with_collapsed_aspect_domains()
        domain = corpus.generic_domains()[0]
        cell = CrossValidationCell(
            runtime_key=runtime_key(selected),
            shape_group_id=selected.shape_group_id,
            selected_candidate_id=selected.candidate_id,
            exact_candidate_id=exact.candidate_id,
            observed_broad_regret=0.06,
        )
        validation = DomainCrossValidation(
            domain=domain,
            selected_feature_policy=FeaturePolicy.CONTINUOUS,
            selected_max_leaves=1,
            selected_boundary_placement=BoundaryPlacement.MIDPOINT,
            selected_profiler_influence=ProfilerInfluence.MEASURED_ONLY,
            fold_count=5,
            shape_group_count=5,
            required_point_count=1,
            covered_point_count=1,
            max_regret=0.06,
            p95_regret=0.06,
            mean_regret=0.06,
            worst_shape_group_id=selected.shape_group_id,
            worst_aggregate_n=selected.aggregate_n,
            worst_k=selected.k,
            worst_selected_candidate_id=selected.candidate_id,
            worst_exact_candidate_id=exact.candidate_id,
            cells=(cell,),
        )
        policy = GenericPolicy(
            rules=(),
            unpromoted_domains=(domain,),
            cross_validation=(validation,),
        )

        plan = build_paired_request_plan(corpus, policy, {})

        self.assertEqual(plan.status, "pending")
        self.assertEqual(len(plan.requests), 1)
        self.assertEqual(plan.requests[0].m, 4)
        self.assertEqual(plan.requests[0].architecture_class, architecture)

    def test_burned_cell_is_grounded_but_never_remeasured(self) -> None:
        """An exhaustive inspected-seal point blocks green without a new edge."""

        corpus, policy, comparisons = self._fixture()
        shape_group = policy.cross_validation[0].cells[0].shape_group_id

        plan = build_paired_request_plan(
            corpus,
            policy,
            comparisons,
            supplemental_shape_group_ids=frozenset({shape_group}),
            supplemental_evidence_digests=("sha256:" + "a" * 64,),
        )

        self.assertEqual(plan.requests, ())
        self.assertEqual(plan.status, "insufficient_cross_validation")
        self.assertTrue(any(
            item.startswith("supplemental-development:")
            for item in plan.unvalidated_domains
        ))

    def test_cached_cv_bootstrap_requires_exact_current_domain_inventory(self) -> None:
        """Enumeration cannot bless a partial or cross-corpus cache generation."""

        corpus, policy, _ = self._fixture()
        validation = policy.cross_validation[0]
        with tempfile.TemporaryDirectory() as directory:
            cache = PolicyFitCache(Path(directory))
            cache.store_validation("a" * 64, validation.domain, validation)
            loaded = load_cached_cross_validations(Path(directory))

        self.assertEqual(loaded, (validation,))
        rebuilt = _policy_from_cached_validations(corpus, loaded)
        self.assertEqual(rebuilt.cross_validation, (validation,))
        self.assertEqual(
            rebuilt.unpromoted_domains,
            tuple(sorted(corpus.generic_domains())),
        )

        foreign = dataclasses.replace(
            validation,
            domain=dataclasses.replace(
                validation.domain,
                architecture_class="foreign-architecture",
            ),
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            _policy_from_cached_validations(corpus, (foreign,))


if __name__ == "__main__":
    unittest.main()
