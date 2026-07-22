#!/usr/bin/env python3
"""Measure the real multi-accelerator NativeVNNI grouped-CV pipeline.

This harness executes production task construction, geometry grouping, dynamic
CUDA/ROCm worker scheduling, all profiler-influence tree fits, result assembly,
deterministic cross-validation ranking, and the complete 30-way publication
tree tournament for each selected OOF winner. It deliberately excludes corpus
measurement and profiler-model training: its target is the exact-tree fit phase
that previously consumed hours despite six available accelerators.

The measured sample is projected to a configurable canonical domain count. A
projection is useful only when the sample includes at least one worst-shaped
domain, so the default runs the 444-point, five-candidate CPU-decode geometry.
Additional comma-separated point counts can model the smaller domains in a real
corpus while preserving the production 5-fold x 12-policy x 2-placement x
4-profiler-influence tournament.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.policy_accelerator import (  # noqa: E402
    PolicyAcceleratorSpec,
)
from native_vnni_dispatch import segmented_policy  # noqa: E402
from Perf__NativeVNNIPolicyScorer import (  # noqa: E402
    build_production_shaped_domain,
)


def _parse_point_counts(raw: str) -> tuple[int, ...]:
    """Parse a nonempty canonical comma-separated domain-size sample."""

    values = tuple(int(token) for token in raw.split(",") if token)
    if not values or any(value < 10 or value > 512 for value in values):
        raise argparse.ArgumentTypeError(
            "point counts must be comma-separated integers in [10, 512]"
        )
    return values


def _parse_accelerators(
    raw: str,
    *,
    cuda_library: Path,
    rocm_library: Path,
) -> tuple[PolicyAcceleratorSpec, ...]:
    """Resolve one strict persistent worker specification per device token."""

    libraries = {"cuda": cuda_library, "rocm": rocm_library}
    specs = []
    observed = set()
    for token in raw.split(","):
        backend, separator, raw_ordinal = token.partition(":")
        if separator != ":" or backend not in libraries:
            raise ValueError(f"invalid accelerator token {token!r}")
        ordinal = int(raw_ordinal)
        identity = (backend, ordinal)
        if ordinal < 0 or identity in observed:
            raise ValueError(f"invalid or duplicate accelerator {token!r}")
        observed.add(identity)
        library = libraries[backend]
        if not library.is_file():
            raise FileNotFoundError(
                f"{backend} scorer library does not exist: {library}"
            )
        specs.append(PolicyAcceleratorSpec(
            backend=backend,
            device_ordinal=ordinal,
            library_path=library,
        ))
    if not specs:
        raise ValueError("at least one CUDA or ROCm accelerator is required")
    return tuple(specs)


def _prediction_cache(domain, costs, *, seed: str):
    """Build deterministic profiler predictions without fitting a surrogate.

    CV consumes already-sealed prediction surfaces. Generating deterministic
    values here isolates tree-CV economics while retaining all four production
    profiler-influence variants and the exact task cardinality they create.
    """

    pool_key = segmented_policy._profiler_transfer_key(domain)
    predictions = {}
    for heldout_groups in segmented_policy._domain_folds(
        domain, costs, seed=seed
    ):
        heldout_geometries = {
            (cost.runtime_key.aggregate_n, cost.runtime_key.k)
            for cost in costs
            if cost.shape_group_id in heldout_groups
        }
        predictions[(pool_key, tuple(sorted(heldout_geometries)))] = {
            (cost.runtime_key, cost.shape_group_id, cost.candidate_id):
                cost.profiler_predicted_regret
            for cost in costs
        }
    return pool_key, predictions


def main() -> int:
    """Execute sampled production CV and report a canonical-corpus projection."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-library", required=True, type=Path)
    parser.add_argument("--rocm-library", required=True, type=Path)
    parser.add_argument(
        "--accelerators",
        default="cuda:0,cuda:1,rocm:0,rocm:1,rocm:2,rocm:3",
    )
    parser.add_argument("--point-counts", type=_parse_point_counts, default=(444,))
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--max-leaves", type=int, default=32)
    parser.add_argument("--min-groups-per-leaf", type=int, default=2)
    parser.add_argument("--lanes-per-accelerator", type=int, default=1)
    parser.add_argument("--project-domains", type=int, default=108)
    parser.add_argument("--target-minutes", type=float, default=10.0)
    parser.add_argument(
        "--skip-publication",
        action="store_true",
        help="measure grouped CV only (diagnostic; not an end-to-end projection)",
    )
    arguments = parser.parse_args()
    if arguments.candidates <= 0:
        parser.error("candidates must be positive")
    if arguments.lanes_per_accelerator <= 0:
        parser.error("lanes-per-accelerator must be positive")
    if arguments.max_leaves <= 0 or arguments.max_leaves > 32:
        parser.error("max-leaves must be in [1, 32]")
    if arguments.project_domains <= 0 or arguments.target_minutes <= 0.0:
        parser.error("projection and target must be positive")

    physical_specs = _parse_accelerators(
        arguments.accelerators,
        cuda_library=arguments.cuda_library.resolve(),
        rocm_library=arguments.rocm_library.resolve(),
    )
    specs = tuple(
        PolicyAcceleratorSpec(
            backend=spec.backend,
            device_ordinal=spec.device_ordinal,
            library_path=spec.library_path,
            lane_count=arguments.lanes_per_accelerator,
        )
        for spec in physical_specs
    )
    seed = "native-vnni-policy-cv-perf-v1"

    started = time.perf_counter()
    domains = []
    costs_by_domain = {}
    fold_counts = {}
    tasks = []
    for domain_index, point_count in enumerate(arguments.point_counts):
        domain, costs = build_production_shaped_domain(
            point_count=point_count,
            candidate_count=arguments.candidates,
            domain_index=domain_index,
        )
        pool_key, prediction_cache = _prediction_cache(
            domain, costs, seed=seed
        )
        domain_tasks, fold_count = segmented_policy._domain_fold_tasks(
            domain,
            list(costs),
            max_leaves=arguments.max_leaves,
            min_shape_groups_per_leaf=arguments.min_groups_per_leaf,
            seed=seed,
            profiler_training_pool_key=pool_key,
            profiler_prediction_cache=prediction_cache,
            compact=True,
        )
        domains.append(domain)
        costs_by_domain[domain] = list(costs)
        fold_counts[domain] = fold_count
        tasks.extend(domain_tasks)
    task_build_complete = time.perf_counter()

    groups = segmented_policy._group_accelerated_cv_tasks(tasks)
    weights = tuple(
        sum(
            segmented_policy._fold_task_weight(task)
            for _index, task in group
        )
        for group in groups
    )
    segmented_policy._PARALLEL_ACCELERATED_CV_TASK_GROUPS = groups
    try:
        task_timings = []
        grouped_results = segmented_policy._run_accelerated_tasks(
            specs, weights, "cv", task_timings
        )
    finally:
        segmented_policy._PARALLEL_ACCELERATED_CV_TASK_GROUPS = ()
    scheduler_complete = time.perf_counter()

    results_by_domain = defaultdict(list)
    for group_results in grouped_results:
        for _task_index, result in group_results:
            results_by_domain[result.domain].append(result)
    validations = []
    for domain in domains:
        ranked = segmented_policy._rank_domain_cross_validations(
            domain,
            costs_by_domain[domain],
            fold_counts[domain],
            results_by_domain[domain],
            max_leaves=arguments.max_leaves,
            retain_model_cells=True,
        )
        if not ranked:
            raise RuntimeError(f"CV produced no ranked model for {domain!r}")
        validations.append(ranked[0])
    reduction_complete = time.perf_counter()

    publication_tasks = []
    for context_index, (domain, validation) in enumerate(
        zip(domains, validations)
    ):
        publication_costs, target_by_point = (
            segmented_policy._cross_fitted_publication_costs(
                costs_by_domain[domain], validation
            )
        )
        for feature_policy in segmented_policy._leaf_budget_feature_policies(
            arguments.max_leaves
        ):
            for placement in (
                segmented_policy._leaf_budget_boundary_placements(
                    arguments.max_leaves
                )
            ):
                publication_tasks.append((
                    context_index,
                    (
                        publication_costs,
                        target_by_point,
                        arguments.max_leaves,
                        arguments.min_groups_per_leaf,
                        feature_policy,
                        placement,
                    ),
                ))
    publication_build_complete = time.perf_counter()

    publication_timings = []
    publication_results = []
    if publication_tasks and not arguments.skip_publication:
        segmented_policy._PARALLEL_FINAL_CANDIDATE_TASKS = tuple(
            publication_tasks
        )
        try:
            publication_results = segmented_policy._run_accelerated_tasks(
                specs,
                tuple(
                    max(
                        1,
                        len(candidate_args[0])
                        * candidate_args[2]
                        * len(segmented_policy.FEATURE_AXES_BY_POLICY[
                            candidate_args[4]
                        ]),
                    )
                    for _context_index, candidate_args in publication_tasks
                ),
                "final-candidate",
                publication_timings,
            )
        finally:
            segmented_policy._PARALLEL_FINAL_CANDIDATE_TASKS = ()
        candidates_by_context = defaultdict(list)
        for context_index, candidate in publication_results:
            if candidate is not None:
                candidates_by_context[context_index].append(candidate)
        missing_contexts = sorted(
            set(range(len(domains))).difference(candidates_by_context)
        )
        if missing_contexts:
            raise RuntimeError(
                "publication tournament produced no tree for contexts "
                f"{missing_contexts}"
            )
    publication_complete = time.perf_counter()

    sample_domains = len(domains)
    if len(task_timings) != len(groups):
        raise RuntimeError(
            "accelerator timing evidence did not cover every geometry group"
        )
    if {timing.task_index for timing in task_timings} != set(range(len(groups))):
        raise RuntimeError(
            "accelerator timing evidence changed geometry-group identity"
        )
    if any(timing.elapsed_seconds <= 0.0 for timing in task_timings):
        raise RuntimeError("accelerator task service time must be positive")

    # Per-task intervals overlap across explicit stream lanes. Dividing their
    # sum by the active lane inventory reconstructs the ideal sample makespan
    # under the production refill scheduler, while the observed excess retains
    # process startup, pipe coordination, and the finite-sample scheduling tail
    # as a one-time full-fit cost. Scaling the raw sample wall time would charge
    # that fixed cost once per sample rather than once per persistent worker pool.
    active_lane_count = sum(spec.lane_count for spec in specs)
    task_service_seconds = sum(
        timing.elapsed_seconds for timing in task_timings
    )
    ideal_sample_accelerator_seconds = (
        task_service_seconds / active_lane_count
    )
    observed_accelerator_seconds = scheduler_complete - task_build_complete
    fixed_scheduler_seconds = max(
        0.0,
        observed_accelerator_seconds - ideal_sample_accelerator_seconds,
    )
    projection_scale = arguments.project_domains / sample_domains
    projected_cv_accelerator_seconds = (
        fixed_scheduler_seconds
        + ideal_sample_accelerator_seconds * projection_scale
    )
    publication_service_seconds = sum(
        timing.elapsed_seconds for timing in publication_timings
    )
    ideal_sample_publication_seconds = (
        publication_service_seconds / active_lane_count
    )
    observed_publication_seconds = (
        publication_complete - publication_build_complete
    )
    fixed_publication_seconds = max(
        0.0,
        observed_publication_seconds - ideal_sample_publication_seconds,
    )
    projected_publication_seconds = (
        fixed_publication_seconds
        + ideal_sample_publication_seconds * projection_scale
    )
    projected_accelerator_seconds = (
        projected_cv_accelerator_seconds + projected_publication_seconds
    )
    projected_host_seconds = (
        (task_build_complete - started)
        + (reduction_complete - scheduler_complete)
        + (publication_build_complete - reduction_complete)
    ) * projection_scale
    total_seconds = publication_complete - started
    projected_seconds = projected_host_seconds + projected_accelerator_seconds
    projection_target_seconds = arguments.target_minutes * 60.0
    service_times_by_backend = defaultdict(list)
    for timing in task_timings:
        backend = timing.accelerator_label.partition(":")[0]
        service_times_by_backend[backend].append(timing.elapsed_seconds)
    publication_service_times_by_backend = defaultdict(list)
    for timing in publication_timings:
        backend = timing.accelerator_label.partition(":")[0]
        publication_service_times_by_backend[backend].append(
            timing.elapsed_seconds
        )
    print(json.dumps({
        "accelerators": [spec.label for spec in specs],
        "accelerator_lane_counts": [spec.lane_count for spec in specs],
        "candidate_count": arguments.candidates,
        "domain_point_counts": list(arguments.point_counts),
        "fit_transaction_count": len(tasks),
        "fixed_scheduler_and_tail_seconds": fixed_scheduler_seconds,
        "fixed_publication_scheduler_and_tail_seconds": (
            fixed_publication_seconds
        ),
        "fold_count_by_domain": [fold_counts[domain] for domain in domains],
        "geometry_group_count": len(groups),
        "ideal_sample_accelerator_seconds": ideal_sample_accelerator_seconds,
        "ideal_sample_publication_seconds": (
            ideal_sample_publication_seconds
        ),
        "lanes_per_accelerator": arguments.lanes_per_accelerator,
        "max_leaves": arguments.max_leaves,
        "profiler_influence_count": len(segmented_policy.TREE_PROFILER_INFLUENCES),
        "projected_domain_count": arguments.project_domains,
        "projected_accelerator_seconds": projected_accelerator_seconds,
        "projected_cv_accelerator_seconds": (
            projected_cv_accelerator_seconds
        ),
        "projected_publication_seconds": projected_publication_seconds,
        "projected_host_seconds": projected_host_seconds,
        "projected_seconds": projected_seconds,
        "projected_minutes": projected_seconds / 60.0,
        "projection_target_minutes": arguments.target_minutes,
        "projection_meets_target": projected_seconds <= projection_target_seconds,
        "sample_domain_count": sample_domains,
        "publication_candidate_count": (
            0 if arguments.skip_publication else len(publication_tasks)
        ),
        "publication_candidate_count_per_domain": (
            len(segmented_policy._leaf_budget_feature_policies(
                arguments.max_leaves
            ))
            * len(segmented_policy._leaf_budget_boundary_placements(
                arguments.max_leaves
            ))
        ),
        "publication_build_seconds": (
            publication_build_complete - reduction_complete
        ),
        "publication_seconds": observed_publication_seconds,
        "publication_service_count_by_backend": {
            backend: len(values)
            for backend, values
            in publication_service_times_by_backend.items()
        },
        "publication_service_median_seconds_by_backend": {
            backend: statistics.median(values)
            for backend, values
            in publication_service_times_by_backend.items()
        },
        "publication_service_seconds": publication_service_seconds,
        "task_service_count_by_backend": {
            backend: len(values)
            for backend, values in service_times_by_backend.items()
        },
        "task_service_median_seconds_by_backend": {
            backend: statistics.median(values)
            for backend, values in service_times_by_backend.items()
        },
        "task_service_seconds": task_service_seconds,
        "task_build_seconds": task_build_complete - started,
        "accelerated_cv_seconds": scheduler_complete - task_build_complete,
        "cv_reduction_seconds": reduction_complete - scheduler_complete,
        "total_seconds": total_seconds,
        "validation_p95_regrets": [value.p95_regret for value in validations],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
