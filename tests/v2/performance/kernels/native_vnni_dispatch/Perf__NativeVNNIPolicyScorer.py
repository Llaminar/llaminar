#!/usr/bin/env python3
"""Measure production-shaped CUDA/ROCm NativeVNNI policy-tree scoring.

The dispatch-policy fitter does not time model kernels.  It repeatedly fits a
bounded generic dispatch tree over CPU, CUDA, or ROCm timing evidence.  This
performance harness isolates the GPU tree-search transaction used by grouped
cross validation so scorer changes can be measured without launching an entire
multi-hour corpus fit.

The default geometry mirrors one collapsed CPU M=1 domain: 444 measured shape
groups, five launch candidates, a five-way held-out fold, every production
runtime feature axis, and all 32 publication leaf budgets.  The first measured
transaction includes graph capture; subsequent transactions must reuse the
same persistent device buffers and executable graph.  No serial Python tree
oracle is run because that would dominate the timing this harness is intended
to expose.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.corpus import RuntimeKey  # noqa: E402
from native_vnni_dispatch.policy_accelerator import (  # noqa: E402
    NativeVNNILeafPrimaryScorer,
    TreeRuntimeStats,
)
from native_vnni_dispatch.schema import (  # noqa: E402
    AspectBucket,
    Backend,
    ExecutionMode,
    SemanticContract,
)
from native_vnni_dispatch import segmented_policy  # noqa: E402


def build_production_shaped_domain(
    *,
    point_count: int,
    candidate_count: int,
    domain_index: int = 0,
) -> tuple[segmented_policy.GenericDomain, tuple]:
    """Construct one deterministic cost domain resembling CPU decode CV.

    Geometry deliberately crosses N/K tile, aspect-ratio, task-grid, and final
    parallel-wave boundaries.  Candidate winners rotate through broad regions
    so retained leaf budgets remain meaningful. The public helper is shared by
    the single-transaction and end-to-end CV performance harnesses; keeping one
    fixture prevents their projections from quietly measuring different work.
    """

    domain = segmented_policy.GenericDomain(
        backend=Backend.CPU,
        architecture_class=(
            f"perf-avx512-{domain_index}|runtime=AVX512|threads=28"
        ),
        semantic_contract=SemanticContract.FAST,
        operation_kind="NativeVNNIFastM1Projection",
        bundle_signature=f"perf-production-shaped-cpu-decode-{domain_index}",
        prepared_family_id="perf-native-vnni",
        packing_abi="perf-native-vnni-v1",
        runtime_codebook_id=6,
        execution_mode=ExecutionMode.EAGER,
        m=1,
        aspect_bucket=AspectBucket.BALANCED,
        all_aspects=True,
    )
    costs = []
    for point_index in range(point_count):
        # Coprime progressions prevent accidental collapse to a tiny threshold
        # inventory while retaining realistic multiples and tails.
        n = 192 + ((point_index * 977) % 196608)
        k = 256 + ((point_index * 353) % 49152)
        if point_index % 3 == 0:
            n = ((n + 63) // 64) * 64
        if point_index % 5 == 0:
            k = ((k + 31) // 32) * 32
        shape_group = (
            f"policy-scorer-perf-{domain_index:03d}-{point_index:03d}"
        )
        key = RuntimeKey(
            backend=Backend.CPU,
            architecture_class=domain.architecture_class,
            semantic_contract=domain.semantic_contract,
            operation_kind=domain.operation_kind,
            bundle_signature=domain.bundle_signature,
            projection_n_vector=(n,),
            prepared_family_id=domain.prepared_family_id,
            packing_abi=domain.packing_abi,
            runtime_codebook_id=domain.runtime_codebook_id,
            execution_mode=domain.execution_mode,
            m=domain.m,
            aggregate_n=n,
            k=k,
        )
        preferred = (
            (point_index // 23)
            + domain_index
            + (1 if n > k else 0)
            + (n // 4096)
            + (k // 8192)
        ) % candidate_count
        for candidate_index in range(candidate_count):
            circular_distance = min(
                (candidate_index - preferred) % candidate_count,
                (preferred - candidate_index) % candidate_count,
            )
            regret = 0.004 + circular_distance * 0.041
            costs.append(segmented_policy.CandidatePointCost(
                runtime_key=key,
                shape_group_id=shape_group,
                candidate_id=f"cpu.nvnni.decode.n_chunk_grid.nbc{1 << candidate_index}",
                max_surface_regret=regret,
                p95_surface_regret=regret,
                mean_surface_regret=regret * 0.75,
                profiler_predicted_regret=max(
                    0.0,
                    regret + (((point_index + candidate_index) % 5) - 2) * 0.003,
                ),
            ))

    return domain, tuple(costs)


def _build_production_shaped_fold(
    *,
    point_count: int,
    candidate_count: int,
    max_leaves: int,
) -> tuple:
    """Construct one fixed heldout fold for isolated graph replay timing."""

    domain, costs = build_production_shaped_domain(
        point_count=point_count,
        candidate_count=candidate_count,
    )
    heldout_groups = frozenset(
        cost.shape_group_id
        for cost in costs
        if int(cost.shape_group_id.rsplit("-", 1)[1]) % 5 == 0
    )

    return (
        domain,
        0,
        heldout_groups,
        segmented_policy.FeaturePolicy.FULL_ROW_GRID_LAUNCH_GEOMETRY,
        segmented_policy.BoundaryPlacement.MIDPOINT,
        segmented_policy.ProfilerInfluence.MEASURED_ONLY,
        costs,
        max_leaves,
        2,
    )


def _stats_delta(before, after) -> dict[str, int]:
    """Return monotonic scorer counters changed by one benchmark window."""

    return {
        field: getattr(after, field) - getattr(before, field)
        for field in before.__dataclass_fields__
        if isinstance(getattr(before, field), int)
    }


def main() -> int:
    """Run cold capture and warm graph-replay measurements."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("cuda", "rocm"))
    parser.add_argument("--library", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--points", type=int, default=444)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--max-leaves", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.points <= 5 or arguments.candidates <= 0:
        parser.error("points must exceed five and candidates must be positive")
    if arguments.max_leaves <= 0 or arguments.max_leaves > 32:
        parser.error("max-leaves must be in [1, 32]")
    if arguments.repeats <= 0:
        parser.error("repeats must be positive")

    fold = _build_production_shaped_fold(
        point_count=arguments.points,
        candidate_count=arguments.candidates,
        max_leaves=arguments.max_leaves,
    )
    scorer = NativeVNNILeafPrimaryScorer(
        arguments.library,
        backend=arguments.backend,
        device_ordinal=arguments.device,
    )
    try:
        before = TreeRuntimeStats(*([0] * len(TreeRuntimeStats.__dataclass_fields__)))
        started = time.perf_counter()
        first = segmented_policy._evaluate_placement_fold(
            fold, primary_scorer=scorer
        )
        cold_seconds = time.perf_counter() - started
        after_cold = scorer.runtime_stats()

        warm_seconds = []
        for _ in range(arguments.repeats):
            started = time.perf_counter()
            repeated = segmented_policy._evaluate_placement_fold(
                fold, primary_scorer=scorer
            )
            warm_seconds.append(time.perf_counter() - started)
            if repeated != first:
                raise RuntimeError("warm scorer replay changed the fitted result")
        after_warm = scorer.runtime_stats()
    finally:
        scorer.close()

    warm_stats = _stats_delta(after_cold, after_warm)
    forbidden_warm_activity = {
        field: warm_stats[field]
        for field in (
            "device_allocation_count",
            "device_free_count",
            "device_sync_count",
            "captured_graph_transfer_count",
            "tree_search_intermediate_sync_count",
            "graph_capture_count",
        )
        if warm_stats[field] != 0
    }
    if forbidden_warm_activity:
        raise RuntimeError(
            "warm captured replay performed forbidden runtime work: "
            f"{forbidden_warm_activity}"
        )
    if warm_stats["graph_replay_count"] != arguments.repeats:
        raise RuntimeError(
            "warm transactions did not all reuse the captured graph: "
            f"{warm_stats['graph_replay_count']} replays for "
            f"{arguments.repeats} transactions"
        )
    if warm_stats["tree_search_stream_sync_count"] != arguments.repeats:
        raise RuntimeError(
            "warm transactions must synchronize only once at final result "
            "publication"
        )

    print(json.dumps({
        "backend": arguments.backend,
        "device": arguments.device,
        "points": arguments.points,
        "training_points": arguments.points - (arguments.points + 4) // 5,
        "heldout_points": (arguments.points + 4) // 5,
        "candidates": arguments.candidates,
        "max_leaves": arguments.max_leaves,
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "warm_median_seconds": statistics.median(warm_seconds),
        "cold_stats": _stats_delta(before, after_cold),
        "warm_stats": warm_stats,
        "scratch_high_water_bytes": after_warm.tree_scratch_high_water_bytes,
        "last_expansion_capacity": after_warm.last_expansion_capacity,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
