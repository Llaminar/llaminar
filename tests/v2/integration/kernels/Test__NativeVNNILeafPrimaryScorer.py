#!/usr/bin/env python3
"""GPU integration proof for exact NativeVNNI leaf scoring and tree fitting."""

from __future__ import annotations

import argparse
import math
import random
import struct
import sys
import unittest
from array import array
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.policy_accelerator import (  # noqa: E402
    TREE_MAXIMUM_HELDOUT_POINTS,
    UINT32_MAX,
    LeafPrimaryScore,
    NativeVNNILeafPrimaryScorer,
)
from native_vnni_dispatch.corpus import RuntimeKey  # noqa: E402
from native_vnni_dispatch.schema import (  # noqa: E402
    AspectBucket,
    Backend,
    ExecutionMode,
    P95_REGRET_BUDGET,
    SemanticContract,
)
from native_vnni_dispatch import segmented_policy  # noqa: E402


def _serial_primary_scores(
    fitting_regrets: tuple[float, ...],
    measured_p95_regrets: tuple[float, ...],
    *,
    point_count: int,
    candidate_count: int,
    subsets: tuple[int, ...],
) -> tuple[LeafPrimaryScore, ...]:
    """Compute the exact Python oracle with the production key predicates."""

    results = []
    for subset in subsets:
        keys = []
        for candidate in range(candidate_count):
            selected_fitting = tuple(
                fitting_regrets[point * candidate_count + candidate]
                for point in range(point_count)
                if subset & (1 << point)
            )
            selected_measured = tuple(
                measured_p95_regrets[point * candidate_count + candidate]
                for point in range(point_count)
                if subset & (1 << point)
            )
            if not selected_fitting or not all(
                math.isfinite(value)
                for value in (*selected_fitting, *selected_measured)
            ):
                continue
            keys.append((
                int(
                    _nearest_rank_p95(selected_measured)
                    >= P95_REGRET_BUDGET
                ),
                _nearest_rank_p95(selected_fitting),
                candidate,
            ))
        if not keys:
            results.append(LeafPrimaryScore(
                fitting_p95_regret=math.inf,
                failed_leaf_count=UINT32_MAX,
                survivor_indices=(),
            ))
            continue
        best_primary = min(key[:2] for key in keys)
        results.append(LeafPrimaryScore(
            fitting_p95_regret=best_primary[1],
            failed_leaf_count=best_primary[0],
            survivor_indices=tuple(
                candidate for failures, maximum, candidate in keys
                if (failures, maximum) == best_primary
            ),
        ))
    return tuple(results)


def _nearest_rank_p95(values: tuple[float, ...]) -> float:
    """Return the exact conservative order statistic used by the scorer ABI."""

    ordered = sorted(values)
    rank = (95 * len(ordered) + 99) // 100
    return ordered[rank - 1]


class TestNativeVNNILeafPrimaryScorer(unittest.TestCase):
    """Compare one vendor DSO with exact serial scoring over adversarial shapes."""

    scorer: NativeVNNILeafPrimaryScorer

    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = NativeVNNILeafPrimaryScorer(
            TEST_ARGUMENTS.library,
            backend=TEST_ARGUMENTS.backend,
            device_ordinal=TEST_ARGUMENTS.device,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Release the backend session before the test process exits."""

        cls.scorer.close()

    def assert_scores_equal(
        self,
        actual: tuple[LeafPrimaryScore, ...],
        expected: tuple[LeafPrimaryScore, ...],
    ) -> None:
        """Require byte-identical maxima and exact counts/survivor inventories."""

        self.assertEqual(len(actual), len(expected))
        for actual_score, expected_score in zip(actual, expected):
            self.assertEqual(
                struct.pack("=d", actual_score.fitting_p95_regret),
                struct.pack("=d", expected_score.fitting_p95_regret),
            )
            self.assertEqual(
                actual_score.failed_leaf_count,
                expected_score.failed_leaf_count,
            )
            self.assertEqual(
                actual_score.survivor_indices,
                expected_score.survivor_indices,
            )

    def assert_warm_graph_replay_is_allocation_and_transfer_free(
        self,
        before,
        after,
    ) -> None:
        """Require persistent storage and a transfer-free captured hot path."""

        self.assertEqual(
            after.device_allocation_count,
            before.device_allocation_count,
        )
        self.assertEqual(after.device_free_count, before.device_free_count)
        self.assertEqual(after.device_sync_count, before.device_sync_count)
        self.assertEqual(after.captured_graph_transfer_count, 0)
        self.assertEqual(before.captured_graph_transfer_count, 0)

    def run_case(
        self,
        fitting_regrets: tuple[float, ...],
        *,
        point_count: int,
        candidate_count: int,
        subsets: tuple[int, ...],
        measured_p95_regrets: tuple[float, ...] | None = None,
    ) -> None:
        """Run one matrix through the serial oracle and selected GPU backend."""

        measured = (
            fitting_regrets
            if measured_p95_regrets is None
            else measured_p95_regrets
        )
        expected = _serial_primary_scores(
            fitting_regrets,
            measured,
            point_count=point_count,
            candidate_count=candidate_count,
            subsets=subsets,
        )
        fitting_matrix = array("d", fitting_regrets)
        measured_matrix = array("d", measured)
        actual = self.scorer.score(
            fitting_matrix,
            measured_matrix,
            point_count=point_count,
            candidate_count=candidate_count,
            subsets=subsets,
        )
        self.assert_scores_equal(actual, expected)
        # A second call with the identical array object must reuse the uploaded
        # matrix and scratch session while preserving exact output bytes.
        repeated = self.scorer.score(
            fitting_matrix,
            measured_matrix,
            point_count=point_count,
            candidate_count=candidate_count,
            subsets=reversed(subsets),
        )
        self.assert_scores_equal(repeated, tuple(reversed(expected)))

    def test_reports_requested_backend_and_visible_device(self) -> None:
        """The worker must bind the requested vendor runtime and ordinal."""

        self.assertEqual(self.scorer.backend, TEST_ARGUMENTS.backend)
        self.assertEqual(self.scorer.device_ordinal, TEST_ARGUMENTS.device)
        self.assertGreater(self.scorer.device_count, TEST_ARGUMENTS.device)

    def test_threshold_ties_and_no_common_candidate_are_exact(self) -> None:
        """Exercise budget neighbors, duplicate columns, and invalid intersections."""

        below = math.nextafter(P95_REGRET_BUDGET, 0.0)
        above = math.nextafter(P95_REGRET_BUDGET, math.inf)
        regrets = (
            below, below, 0.01, math.inf,
            P95_REGRET_BUDGET,
            P95_REGRET_BUDGET,
            math.inf,
            0.02,
            above, above, 0.01, math.inf,
        )
        subsets = (
            0b001,
            0b010,
            0b100,
            0b011,
            0b111,
            0b101,
        )
        self.run_case(
            regrets,
            point_count=3,
            candidate_count=4,
            subsets=subsets,
        )

        self.run_case(
            (math.inf, 0.01, 0.02, math.inf),
            point_count=2,
            candidate_count=2,
            subsets=(0b11,),
        )

    def test_segment_p95_replaces_point_failure_count_and_maximum(self) -> None:
        """The accelerator must use the same nearest-rank p95 as installation."""

        self.run_case(
            (
                0.40, 0.04,
                0.00, 0.04,
                0.00, 0.04,
                0.00, 0.04,
                0.00, 0.00,
            ),
            point_count=5,
            candidate_count=2,
            subsets=(0b11111,),
        )

    def test_nearest_rank_p95_is_exact_for_every_tree_population(self) -> None:
        """Prove every population rank admitted by complete device search."""

        point_count = 64
        fitting_regrets = tuple(
            point_index / 1000.0 for point_index in range(point_count)
        )
        self.run_case(
            fitting_regrets,
            point_count=point_count,
            candidate_count=1,
            subsets=tuple(
                (1 << population_size) - 1
                for population_size in range(1, point_count + 1)
            ),
        )

    def test_measured_p95_gate_precedes_bounded_fitting_p95(self) -> None:
        """Profiler-informed fitting cannot manufacture installation evidence."""

        self.run_case(
            (
                0.10, 0.04,
                0.10, 0.04,
                0.10, 0.04,
                0.10, 0.04,
                0.10, 0.04,
            ),
            measured_p95_regrets=(
                0.01, 0.04,
                0.01, 0.04,
                0.01, 0.04,
                0.01, 0.04,
                0.01, 0.04,
            ),
            point_count=5,
            candidate_count=2,
            subsets=(0b11111,),
        )

    def test_multiword_subsets_and_candidate_masks_match_serial(self) -> None:
        """Cross four mask words and the 256-thread launch boundary."""

        point_count = 194
        candidate_count = 257
        random_source = random.Random(0x4E564E4E49)
        values = []
        boundary_values = (
            0.0,
            math.nextafter(P95_REGRET_BUDGET, 0.0),
            P95_REGRET_BUDGET,
            math.nextafter(P95_REGRET_BUDGET, math.inf),
            0.08,
        )
        for point in range(point_count):
            row = []
            for candidate in range(candidate_count):
                value = (
                    boundary_values[(point * 11 + candidate * 7) % len(boundary_values)]
                    + random_source.randrange(0, 1000) * 1.0e-8
                )
                if (point * 17 + candidate * 13) % 101 == 0:
                    value = math.inf
                row.append(value)
            # Candidate 256 is always valid, which distinguishes a genuinely
            # empty intersection from an accidental all-invalid random matrix.
            row[256] = 0.2 + point * 1.0e-8
            # Duplicate columns force the GPU to return every exact primary tie
            # whenever either member happens to win a requested subset.
            row[1] = row[0]
            values.extend(row)

        subsets = [
            (1 << point_count) - 1,
            sum(1 << point for point in range(0, point_count, 2)),
            sum(1 << point for point in range(1, point_count, 2)),
            (1 << 63) | (1 << 64) | (1 << 65),
            (1 << 0) | (1 << 64) | (1 << 129) | (1 << 193),
        ]
        for _ in range(19):
            selected = random_source.sample(range(point_count), 1 + random_source.randrange(40))
            subsets.append(sum(1 << point for point in selected))
        self.run_case(
            tuple(values),
            point_count=point_count,
            candidate_count=candidate_count,
            subsets=tuple(subsets),
        )

    def test_complete_device_tree_matches_python_oracle(self) -> None:
        """Prove frontier expansion, dedup, top-64, and reconstruction exactly."""

        costs = []
        for point_index in range(40):
            n = 128 + point_index * 32
            key = RuntimeKey(
                backend=Backend.CPU,
                architecture_class="unit-avx2",
                semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
                operation_kind="NativeVNNIPrefillProjection",
                bundle_signature="unit-single-projection",
                projection_n_vector=(n,),
                prepared_family_id="unit-prepared",
                packing_abi="unit-packing",
                runtime_codebook_id=6,
                execution_mode=ExecutionMode.EAGER,
                m=64,
                aggregate_n=n,
                k=512,
            )
            candidate_regrets = {
                "candidate.small-left": (
                    0.04 if point_index == 0
                    else 0.0 if point_index < 3
                    else 0.50
                ),
                "candidate.large-right": (
                    0.04 if point_index < 3 else 0.0
                ),
                "candidate.balanced-left": (
                    0.10 if point_index == 0
                    else 0.0 if point_index < 20
                    else 0.50
                ),
            }
            for candidate, regret in candidate_regrets.items():
                costs.append(segmented_policy.CandidatePointCost(
                    runtime_key=key,
                    shape_group_id=f"tree-oracle-{point_index:02d}",
                    candidate_id=candidate,
                    max_surface_regret=regret,
                    p95_surface_regret=regret,
                    mean_surface_regret=regret,
                ))

        expected = segmented_policy._fit_tree_budgets(
            costs,
            max_leaves=3,
            min_shape_groups_per_leaf=2,
        )
        actual = segmented_policy._fit_tree_budgets(
            costs,
            max_leaves=3,
            min_shape_groups_per_leaf=2,
            primary_scorer=self.scorer,
        )
        after_first = self.scorer.runtime_stats()
        repeated = segmented_policy._fit_tree_budgets(
            costs,
            max_leaves=3,
            min_shape_groups_per_leaf=2,
            primary_scorer=self.scorer,
        )
        after_replay = self.scorer.runtime_stats()

        self.assertEqual(actual, expected)
        self.assertEqual(repeated, expected)
        self.assertEqual(actual[-1].root.failed_leaf_count, 0)
        self.assertLess(
            actual[-1].root.worst_leaf_p95_regret,
            P95_REGRET_BUDGET,
        )
        # A changed Python matrix object republishes bytes into the same backend
        # context. Stable dimensions must neither grow device storage nor recapture
        # the complete search; they replay the graph and synchronize exactly once
        # after the final result/status download.
        self.assertEqual(
            after_replay.matrix_growth_count,
            after_first.matrix_growth_count,
        )
        self.assertEqual(
            after_replay.tree_scratch_growth_count,
            after_first.tree_scratch_growth_count,
        )
        self.assertEqual(
            after_replay.graph_capture_count,
            after_first.graph_capture_count,
        )
        self.assertEqual(
            after_replay.graph_replay_count,
            after_first.graph_replay_count + 1,
        )
        self.assertEqual(
            after_replay.tree_search_count,
            after_first.tree_search_count + 1,
        )
        self.assertEqual(
            after_replay.tree_search_stream_sync_count,
            after_first.tree_search_stream_sync_count + 1,
        )
        self.assertEqual(after_replay.tree_search_intermediate_sync_count, 0)
        self.assert_warm_graph_replay_is_allocation_and_transfer_free(
            after_first,
            after_replay,
        )

    def test_fused_grouped_cv_matches_python_without_tree_download(self) -> None:
        """Prove captured heldout routing, exact winners, and compact D2H."""

        heldout_groups = frozenset(
            f"fused-cv-{index:02d}" for index in (2, 7, 12, 17, 22)
        )
        costs = []
        for point_index in range(25):
            n = 97 + point_index * 43 + (point_index % 4) * 17
            k = 320 + (point_index * 7 % 11) * 64
            key = RuntimeKey(
                backend=Backend.CPU,
                architecture_class="unit|threads=28",
                semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
                operation_kind="NativeVNNIPrefillProjection",
                bundle_signature="unit-fused-cv-projection",
                projection_n_vector=(n,),
                prepared_family_id="unit-prepared",
                packing_abi="unit-packing",
                runtime_codebook_id=9,
                execution_mode=ExecutionMode.EAGER,
                m=64,
                aggregate_n=n,
                k=k,
            )
            shape_group = f"fused-cv-{point_index:02d}"
            preferred = (point_index // 5) % 3
            for candidate_index, candidate in enumerate((
                "candidate.alpha",
                "candidate.beta",
                "candidate.gamma",
            )):
                distance = abs(candidate_index - preferred)
                maximum = 0.004 + distance * 0.027
                costs.append(segmented_policy.CandidatePointCost(
                    runtime_key=key,
                    shape_group_id=shape_group,
                    candidate_id=candidate,
                    max_surface_regret=maximum,
                    p95_surface_regret=maximum * 0.8,
                    mean_surface_regret=maximum * 0.6,
                ))
            if shape_group in heldout_groups:
                # This candidate has no finite training column. The fused path
                # must still retain it in the heldout oracle inventory.
                costs.append(segmented_policy.CandidatePointCost(
                    runtime_key=key,
                    shape_group_id=shape_group,
                    candidate_id="candidate.heldout-only",
                    max_surface_regret=0.001,
                    p95_surface_regret=0.0008,
                    mean_surface_regret=0.0006,
                ))

        domain = segmented_policy.GenericDomain(
            backend=Backend.CPU,
            architecture_class="unit|threads=28",
            semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
            operation_kind="NativeVNNIPrefillProjection",
            bundle_signature="unit-fused-cv-projection",
            prepared_family_id="unit-prepared",
            packing_abi="unit-packing",
            runtime_codebook_id=9,
            execution_mode=ExecutionMode.EAGER,
            m=64,
            aspect_bucket=AspectBucket.BALANCED,
            all_aspects=True,
        )
        arguments = (
            domain,
            0,
            heldout_groups,
            segmented_policy.FeaturePolicy.FULL_ROW_GRID_LAUNCH_GEOMETRY,
            segmented_policy.BoundaryPlacement.MIDPOINT,
            segmented_policy.ProfilerInfluence.MEASURED_ONLY,
            costs,
            4,
            2,
        )
        expected = segmented_policy._evaluate_placement_fold(arguments)
        actual = segmented_policy._evaluate_placement_fold(
            arguments, self.scorer
        )
        after_first = self.scorer.runtime_stats()
        repeated = segmented_policy._evaluate_placement_fold(
            arguments, self.scorer
        )
        after_replay = self.scorer.runtime_stats()

        self.assertEqual(actual, expected)
        self.assertEqual(repeated, expected)
        for complexity in actual.complexities:
            self.assertEqual(complexity.uncovered, 0)
            self.assertTrue(all(
                exact.candidate_id == "candidate.heldout-only"
                for _selected, exact in complexity.decisions
            ))
        self.assertEqual(
            after_replay.fused_evaluation_count,
            after_first.fused_evaluation_count + 1,
        )
        self.assertEqual(
            after_replay.tree_search_stream_sync_count,
            after_first.tree_search_stream_sync_count + 1,
        )
        self.assertEqual(after_replay.tree_search_intermediate_sync_count, 0)
        self.assertEqual(
            after_replay.graph_capture_count,
            after_first.graph_capture_count,
        )
        self.assertEqual(
            after_replay.graph_replay_count,
            after_first.graph_replay_count + 1,
        )
        expected_compact_bytes = (
            4 * (8 + TREE_MAXIMUM_HELDOUT_POINTS * 4)
            + len(heldout_groups) * 4
            + 4
        )
        self.assertEqual(
            after_replay.final_result_d2h_bytes
            - after_first.final_result_d2h_bytes,
            expected_compact_bytes,
        )
        self.assert_warm_graph_replay_is_allocation_and_transfer_free(
            after_first,
            after_replay,
        )

    def test_fused_grouped_cv_covers_444_point_cpu_decode_domain(self) -> None:
        """Regress the production collapsed-aspect domain beyond ABI v7.

        The current CPU M=1 corpus contains 444 points per collapsed codebook
        domain. A five-way grouped fold leaves 399 training points, which used
        to fail before graph capture because ABI v7 carried only four 64-bit
        point-mask words. A second deliberately inverted fold retains 300
        held-out points, proving that the tiled evaluator also crosses its
        256-thread block boundary. Keep the exact production fused evaluator
        involved: tree fitting, heldout threshold publication, routing, and
        measured winner selection must all agree with serial Python without
        downloading an intermediate tree.
        """

        point_count = 444
        heldout_groups = frozenset(
            f"cpu-decode-444-{point_index:03d}"
            for point_index in range(0, point_count, 10)
        )
        costs = []
        for point_index in range(point_count):
            n = 320 + point_index * 16
            k = 768 + (point_index % 13) * 64
            key = RuntimeKey(
                backend=Backend.CPU,
                architecture_class="unit-avx2|threads=28",
                semantic_contract=SemanticContract.FAST,
                operation_kind="NativeVNNIFastM1Projection",
                bundle_signature="unit-cpu-decode-444",
                projection_n_vector=(n,),
                prepared_family_id="unit-prepared",
                packing_abi="unit-packing",
                runtime_codebook_id=6,
                execution_mode=ExecutionMode.EAGER,
                m=1,
                aggregate_n=n,
                k=k,
            )
            shape_group = f"cpu-decode-444-{point_index:03d}"
            preferred = 0 if point_index < point_count // 2 else 1
            for candidate_index, candidate in enumerate((
                "candidate.low-n",
                "candidate.high-n",
                "candidate.steady",
            )):
                if candidate_index == 2:
                    regret = 0.032
                else:
                    regret = 0.004 if candidate_index == preferred else 0.090
                costs.append(segmented_policy.CandidatePointCost(
                    runtime_key=key,
                    shape_group_id=shape_group,
                    candidate_id=candidate,
                    max_surface_regret=regret,
                    p95_surface_regret=regret,
                    mean_surface_regret=regret,
                ))

        domain = segmented_policy.GenericDomain(
            backend=Backend.CPU,
            architecture_class="unit-avx2|threads=28",
            semantic_contract=SemanticContract.FAST,
            operation_kind="NativeVNNIFastM1Projection",
            bundle_signature="unit-cpu-decode-444",
            prepared_family_id="unit-prepared",
            packing_abi="unit-packing",
            runtime_codebook_id=6,
            execution_mode=ExecutionMode.EAGER,
            m=1,
            aspect_bucket=AspectBucket.BALANCED,
            all_aspects=True,
        )
        arguments = (
            domain,
            0,
            heldout_groups,
            segmented_policy.FeaturePolicy.CONTINUOUS,
            segmented_policy.BoundaryPlacement.MIDPOINT,
            segmented_policy.ProfilerInfluence.MEASURED_ONLY,
            costs,
            2,
            2,
        )
        expected = segmented_policy._evaluate_placement_fold(arguments)
        actual = segmented_policy._evaluate_placement_fold(
            arguments, self.scorer
        )
        after_first = self.scorer.runtime_stats()
        repeated = segmented_policy._evaluate_placement_fold(
            arguments, self.scorer
        )
        after_replay = self.scorer.runtime_stats()

        self.assertEqual(actual, expected)
        self.assertEqual(repeated, expected)
        self.assertEqual(len(heldout_groups), 45)
        self.assertEqual(
            after_replay.fused_evaluation_count,
            after_first.fused_evaluation_count + 1,
        )
        self.assertEqual(
            after_replay.tree_search_stream_sync_count,
            after_first.tree_search_stream_sync_count + 1,
        )
        self.assertEqual(after_replay.tree_search_intermediate_sync_count, 0)

        # The ordinary five-way fold above exercises a >256-point training
        # mask but has only 45 held-out rows.  Exercise the complementary
        # boundary explicitly: the evaluation kernel uses one 256-thread tile
        # per budget, so 300 held-out rows require two independently reduced
        # blocks whose coverage cardinalities must add exactly.
        large_heldout_groups = frozenset(
            f"cpu-decode-444-{point_index:03d}"
            for point_index in range(300)
        )
        large_heldout_arguments = (
            domain,
            1,
            large_heldout_groups,
            segmented_policy.FeaturePolicy.CONTINUOUS,
            segmented_policy.BoundaryPlacement.MIDPOINT,
            segmented_policy.ProfilerInfluence.MEASURED_ONLY,
            costs,
            2,
            2,
        )
        expected_large = segmented_policy._evaluate_placement_fold(
            large_heldout_arguments
        )
        actual_large = segmented_policy._evaluate_placement_fold(
            large_heldout_arguments, self.scorer
        )
        after_large_first = self.scorer.runtime_stats()
        repeated_large = segmented_policy._evaluate_placement_fold(
            large_heldout_arguments, self.scorer
        )
        after_large_replay = self.scorer.runtime_stats()

        self.assertEqual(len(large_heldout_groups), 300)
        self.assertEqual(actual_large, expected_large)
        self.assertEqual(repeated_large, expected_large)
        self.assertEqual(
            after_large_replay.fused_evaluation_count,
            after_large_first.fused_evaluation_count + 1,
        )
        self.assertEqual(
            after_large_replay.graph_capture_count,
            after_large_first.graph_capture_count,
        )
        self.assertEqual(
            after_large_replay.graph_replay_count,
            after_large_first.graph_replay_count + 1,
        )
        self.assertEqual(
            after_large_replay.matrix_growth_count,
            after_large_first.matrix_growth_count,
        )
        self.assertEqual(
            after_large_replay.tree_scratch_growth_count,
            after_large_first.tree_scratch_growth_count,
        )
        self.assertEqual(
            after_large_replay.tree_scratch_high_water_bytes,
            after_large_first.tree_scratch_high_water_bytes,
        )
        self.assertEqual(
            after_large_replay.tree_search_stream_sync_count,
            after_large_first.tree_search_stream_sync_count + 1,
        )
        self.assertEqual(
            after_large_replay.tree_search_intermediate_sync_count,
            0,
        )

    def test_complete_device_tree_crosses_point_mask_word_boundary(self) -> None:
        """Prove ABI v5 fits more than 64 shape groups without host replay."""

        costs = []
        for point_index in range(81):
            n = 256 + point_index * 32
            key = RuntimeKey(
                backend=Backend.CPU,
                architecture_class="unit-avx512",
                semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
                operation_kind="NativeVNNIPrefillProjection",
                bundle_signature="unit-multiword-projection",
                projection_n_vector=(n,),
                prepared_family_id="unit-prepared",
                packing_abi="unit-packing",
                runtime_codebook_id=0,
                execution_mode=ExecutionMode.EAGER,
                m=64,
                aggregate_n=n,
                k=1280,
            )
            for candidate, regret in (
                (
                    "candidate.left",
                    0.004 if point_index < 41 else 0.080,
                ),
                (
                    "candidate.right",
                    0.080 if point_index < 41 else 0.005,
                ),
            ):
                costs.append(segmented_policy.CandidatePointCost(
                    runtime_key=key,
                    shape_group_id=f"multiword-tree-{point_index:03d}",
                    candidate_id=candidate,
                    max_surface_regret=regret,
                    p95_surface_regret=regret,
                    mean_surface_regret=regret,
                ))

        expected = segmented_policy._fit_tree_budgets(
            costs,
            max_leaves=3,
            min_shape_groups_per_leaf=2,
        )
        actual = segmented_policy._fit_tree_budgets(
            costs,
            max_leaves=3,
            min_shape_groups_per_leaf=2,
            primary_scorer=self.scorer,
        )

        self.assertEqual(actual, expected)
        self.assertEqual(actual[-1].root.failed_leaf_count, 0)

        def point_masks(node) -> tuple[int, ...]:
            """Return every private tree leaf mask in canonical preorder."""

            if isinstance(node, segmented_policy._LeafNode):
                return (node.point_mask,)
            return point_masks(node.left) + point_masks(node.right)

        self.assertTrue(any(
            point_mask >> 64
            for point_mask in point_masks(actual[-1].root)
        ))

    def test_complete_tree_matches_every_candidate_group_width(self) -> None:
        """Prove every CUDA specialization and ROCm wave64 padding width."""

        for candidate_count in range(1, 10):
            with self.subTest(candidate_count=candidate_count):
                costs = []
                for point_index in range(12):
                    n = 192 + point_index * 64
                    k = 384 + (point_index % 4) * 128
                    key = RuntimeKey(
                        backend=Backend.CPU,
                        architecture_class="unit|threads=28",
                        semantic_contract=(
                            SemanticContract.VERIFIER_SERIAL_M1_BITWISE
                        ),
                        operation_kind="NativeVNNIPrefillProjection",
                        bundle_signature="unit-single-projection",
                        projection_n_vector=(n,),
                        prepared_family_id="unit-prepared",
                        packing_abi="unit-packing",
                        runtime_codebook_id=11,
                        execution_mode=ExecutionMode.EAGER,
                        m=256,
                        aggregate_n=n,
                        k=k,
                    )
                    preferred = (point_index // 3) % candidate_count
                    for candidate_index in range(candidate_count):
                        distance = abs(candidate_index - preferred)
                        regret = 0.004 + distance * 0.019
                        costs.append(segmented_policy.CandidatePointCost(
                            runtime_key=key,
                            shape_group_id=(
                                f"candidate-width-{candidate_count}-"
                                f"{point_index:02d}"
                            ),
                            candidate_id=f"candidate.{candidate_index:02d}",
                            max_surface_regret=regret,
                            p95_surface_regret=regret,
                            mean_surface_regret=regret,
                        ))

                expected = segmented_policy._fit_tree_budgets(
                    costs,
                    max_leaves=3,
                    min_shape_groups_per_leaf=2,
                )
                actual = segmented_policy._fit_tree_budgets(
                    costs,
                    max_leaves=3,
                    min_shape_groups_per_leaf=2,
                    primary_scorer=self.scorer,
                )
                self.assertEqual(actual, expected)

    def test_complete_tree_matches_across_feature_and_placement_families(
        self,
    ) -> None:
        """Exercise general geometry, bounded priors, and sparse availability."""

        random_source = random.Random(0x54524545)
        costs = []
        candidates = (
            "candidate.alpha",
            "candidate.beta",
            "candidate.delta",
            "candidate.gamma",
        )
        for point_index in range(14):
            n = 160 + point_index * 47 + (point_index % 3) * 11
            k = 320 + (point_index * 5 % 9) * 64
            key = RuntimeKey(
                backend=Backend.CPU,
                architecture_class="unit|threads=28",
                semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
                operation_kind="NativeVNNIPrefillProjection",
                bundle_signature="unit-single-projection",
                projection_n_vector=(n,),
                prepared_family_id="unit-prepared",
                packing_abi="unit-packing",
                runtime_codebook_id=9,
                execution_mode=ExecutionMode.EAGER,
                m=256,
                aggregate_n=n,
                k=k,
            )
            for candidate_index, candidate in enumerate(candidates):
                if candidate == "candidate.delta" and point_index % 4 == 0:
                    continue
                base = (
                    ((point_index + 3) * (candidate_index + 5) * 17) % 113
                ) / 1000.0
                maximum = base + random_source.randrange(0, 7) / 10000.0
                p95 = maximum * (0.75 + candidate_index * 0.05)
                mean = p95 * (0.65 + (point_index % 3) * 0.07)
                prediction = max(
                    0.0,
                    maximum
                    + ((point_index + candidate_index) % 5 - 2) * 0.004,
                )
                costs.append(segmented_policy.CandidatePointCost(
                    runtime_key=key,
                    shape_group_id=f"general-tree-{point_index:02d}",
                    candidate_id=candidate,
                    max_surface_regret=maximum,
                    p95_surface_regret=p95,
                    mean_surface_regret=mean,
                    profiler_predicted_regret=prediction,
                ))

        configurations = (
            (
                segmented_policy.FeaturePolicy.CONTINUOUS,
                segmented_policy.BoundaryPlacement.MIDPOINT,
            ),
            (
                segmented_policy.FeaturePolicy.TILE_64_128,
                segmented_policy.BoundaryPlacement.LOWER_EDGE,
            ),
            (
                segmented_policy.FeaturePolicy.TILE_512,
                segmented_policy.BoundaryPlacement.MIDPOINT,
            ),
            (
                segmented_policy.FeaturePolicy.PARALLEL_WAVE_SCHEDULES,
                segmented_policy.BoundaryPlacement.MIDPOINT,
            ),
        )
        for feature_policy, placement in configurations:
            with self.subTest(
                feature_policy=feature_policy.value,
                placement=placement.value,
            ):
                expected = segmented_policy._fit_tree_budgets(
                    costs,
                    max_leaves=4,
                    min_shape_groups_per_leaf=2,
                    boundary_placement=placement,
                    feature_policy=feature_policy,
                )
                actual = segmented_policy._fit_tree_budgets(
                    costs,
                    max_leaves=4,
                    min_shape_groups_per_leaf=2,
                    boundary_placement=placement,
                    feature_policy=feature_policy,
                    primary_scorer=self.scorer,
                )
                self.assertEqual(actual, expected)

    def test_identical_geometry_returns_one_leaf_for_every_budget(self) -> None:
        """No legal threshold is a terminal incumbent, not an empty policy."""

        costs = []
        key = RuntimeKey(
            backend=Backend.CPU,
            architecture_class="unit|threads=28",
            semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
            operation_kind="NativeVNNIPrefillProjection",
            bundle_signature="unit-single-projection",
            projection_n_vector=(512,),
            prepared_family_id="unit-prepared",
            packing_abi="unit-packing",
            runtime_codebook_id=9,
            execution_mode=ExecutionMode.EAGER,
            m=256,
            aggregate_n=512,
            k=1024,
        )
        for group_index in range(4):
            for candidate, regret in (
                ("candidate.best", 0.01),
                ("candidate.other", 0.04),
            ):
                costs.append(segmented_policy.CandidatePointCost(
                    runtime_key=key,
                    shape_group_id=f"same-geometry-{group_index}",
                    candidate_id=candidate,
                    max_surface_regret=regret,
                    p95_surface_regret=regret,
                    mean_surface_regret=regret,
                ))

        expected = segmented_policy._fit_tree_budgets(
            costs,
            max_leaves=4,
            min_shape_groups_per_leaf=2,
        )
        actual = segmented_policy._fit_tree_budgets(
            costs,
            max_leaves=4,
            min_shape_groups_per_leaf=2,
            primary_scorer=self.scorer,
        )
        self.assertEqual(actual, expected)
        self.assertEqual(tuple(fit.leaf_count for fit in actual), (1, 1, 1, 1))


def _parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    """Separate scorer configuration from unittest's own command-line flags."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", required=True, choices=("cuda", "rocm"))
    parser.add_argument("--library", required=True)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_known_args()


TEST_ARGUMENTS, UNITTEST_ARGUMENTS = _parse_arguments()


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *UNITTEST_ARGUMENTS])
