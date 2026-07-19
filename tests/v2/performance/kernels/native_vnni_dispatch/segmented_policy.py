"""Bounded multi-feature learner minimizing measured candidate latency regret.

The policy is represented as non-overlapping decision-tree leaves. Each leaf
contains an explicit conjunction over continuous shape features and discrete
launch-geometry features, then names one forceable candidate. Trees are fitted
with a bounded, deterministic beam over the complete candidate cost matrix,
selected by grouped cross-validation, and flattened into backend-neutral rules
for emitters.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import re
import statistics
import sys
import time
import traceback
from array import array
from collections import defaultdict, deque
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait as wait_for_futures,
)
from dataclasses import dataclass, replace
from enum import Enum
from fractions import Fraction
from functools import cached_property, lru_cache
from multiprocessing.connection import wait as wait_for_connections
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .corpus import (
    GenericDomain,
    ObservationCorpus,
    RuntimeKey,
    SurfaceKey,
)
from .exact_oracle import candidate_is_eligible
from .paired_confirmation import PairedCellKey, PairedTimingComparison
from .policy_accelerator import (
    UINT32_MAX,
    AcceleratedFoldEvaluation,
    AcceleratedTreeFitResult,
    NativeVNNILeafPrimaryScorer,
    PolicyAcceleratorSpec,
    TreeBoundaryPlacement,
    TreeFeatureAxisDescriptor,
    TreeThresholdDescriptor,
    TreeThresholdOperation,
    arm_policy_worker_parent_death_signal,
    policy_accelerator_specs_from_environment,
    policy_accelerator_worker_specs,
)
from .profiles import MIN_GENERIC_CROSS_VALIDATION_SHAPE_GROUPS
from .profiler_model import (
    ProfilerFeatureCatalog,
    ProfilerObservationIndex,
    apply_profiler_regret_predictions,
    build_profiler_model_record_index,
    build_profiler_observation_index,
)
from .schema import (
    AspectBucket,
    Backend,
    ExecutionMode,
    NativeVNNIObservation,
    P95_REGRET_BUDGET,
    SemanticContract,
)


TREE_BEAM_WIDTH = 64
DEFAULT_TREE_LEAVES = 16
MAX_TREE_LEAVES = 32
GENERIC_REGRET_BUDGET = P95_REGRET_BUDGET
GENERIC_DOMAIN_PROMOTION_NUMERATOR = 99
GENERIC_DOMAIN_PROMOTION_DENOMINATOR = 100
POLICY_FIT_CACHE_SCHEMA_VERSION = "native-vnni-policy-fit-cache-v10"
POLICY_COST_STATISTIC_VERSION = "conservative-nearest-rank-p95-v1"
POLICY_PRIMARY_OBJECTIVE_VERSION = (
    "cross-fitted-publication-measured-leaf-p95-v13"
)
CROSS_VALIDATION_FOLD_SCHEMA_VERSION = "geometry-atomic-kd-regions-v1"
PROFILER_REGRET_BLEND_WEIGHT = 0.25
PROFILER_MAX_REGRET_ADJUSTMENT = GENERIC_REGRET_BUDGET


def _physical_core_worker_count(
    *,
    affinity: Iterable[int] | None = None,
    topology_root: Path = Path("/sys/devices/system/cpu"),
) -> int:
    """Return affinity-visible physical cores for process-pool defaults.

    The policy fitter runs independent CPU-bound Python/sklearn processes.
    Scheduling one process per SMT sibling doubles resident model state and
    run-queue pressure without doubling tree throughput. Linux exposes stable
    package/core identities for every logical CPU; count unique identities only
    within the process affinity mask so container and NUMA pinning remain
    authoritative. A platform without complete topology metadata falls back to
    its visible logical count rather than guessing an SMT ratio.
    """

    if affinity is None:
        try:
            visible_cpus = tuple(sorted(os.sched_getaffinity(0)))
        except AttributeError:
            visible_cpus = tuple(range(os.cpu_count() or 1))
    else:
        visible_cpus = tuple(sorted(set(affinity)))
    if not visible_cpus:
        return 1

    physical_cores = set()
    for cpu in visible_cpus:
        topology = topology_root / f"cpu{cpu}" / "topology"
        try:
            package_id = int(
                (topology / "physical_package_id").read_text(
                    encoding="utf-8"
                ).strip()
            )
            core_id = int(
                (topology / "core_id").read_text(encoding="utf-8").strip()
            )
        except (OSError, ValueError):
            return len(visible_cpus)
        physical_cores.add((package_id, core_id))
    return max(1, len(physical_cores))


class FeatureAxis(str, Enum):
    """Runtime-visible scalar axes permitted in a generic policy tree.

    The original N/K/work feature set treated launch geometry as continuous.
    NativeVNNI kernels do not: changing N across a tile boundary changes the
    CTA count, while K partition economics depend on K groups per N tile.  The
    tiled axes expose those deterministic discontinuities without admitting a
    backend-specific model or a shape-name lookup disguised as generic policy.
    Every width is used by at least one reviewed CUDA NativeVNNI candidate and
    remains meaningful to CPU and ROCm as a portable work-granularity feature.
    """

    AGGREGATE_N = "aggregate_n"
    K = "k"
    WORK_ITEMS = "work_items"
    ASPECT_RATIO = "aspect_ratio"
    N_TILES_32 = "n_tiles_32"
    N_TILES_64 = "n_tiles_64"
    N_TILES_128 = "n_tiles_128"
    N_TILES_256 = "n_tiles_256"
    N_TILES_512 = "n_tiles_512"
    N_TILES_1024 = "n_tiles_1024"
    K_GROUPS_PER_N_TILE_32 = "k_groups_per_n_tile_32"
    K_GROUPS_PER_N_TILE_64 = "k_groups_per_n_tile_64"
    K_GROUPS_PER_N_TILE_128 = "k_groups_per_n_tile_128"
    K_GROUPS_PER_N_TILE_256 = "k_groups_per_n_tile_256"
    K_GROUPS_PER_N_TILE_512 = "k_groups_per_n_tile_512"
    K_GROUPS_PER_N_TILE_1024 = "k_groups_per_n_tile_1024"
    N_FINAL_TILE_VALUES_32 = "n_final_tile_values_32"
    N_FINAL_TILE_VALUES_64 = "n_final_tile_values_64"
    N_FINAL_TILE_VALUES_128 = "n_final_tile_values_128"
    N_FINAL_TILE_VALUES_256 = "n_final_tile_values_256"
    N_FINAL_TILE_VALUES_512 = "n_final_tile_values_512"
    N_FINAL_TILE_VALUES_1024 = "n_final_tile_values_1024"
    K_FINAL_TILE_VALUES_32 = "k_final_tile_values_32"
    K_FINAL_TILE_VALUES_64 = "k_final_tile_values_64"
    K_FINAL_TILE_VALUES_128 = "k_final_tile_values_128"
    K_FINAL_TILE_VALUES_256 = "k_final_tile_values_256"
    K_FINAL_TILE_VALUES_512 = "k_final_tile_values_512"
    K_FINAL_TILE_VALUES_1024 = "k_final_tile_values_1024"
    N_TILE_UTILIZATION_32 = "n_tile_utilization_32"
    N_TILE_UTILIZATION_64 = "n_tile_utilization_64"
    N_TILE_UTILIZATION_128 = "n_tile_utilization_128"
    N_TILE_UTILIZATION_256 = "n_tile_utilization_256"
    N_TILE_UTILIZATION_512 = "n_tile_utilization_512"
    N_TILE_UTILIZATION_1024 = "n_tile_utilization_1024"
    N_PARALLEL_WAVES_64 = "n_parallel_waves_64"
    N_PARALLEL_WAVES_128 = "n_parallel_waves_128"
    N_PARALLEL_WAVES_256 = "n_parallel_waves_256"
    N_PARALLEL_WAVES_512 = "n_parallel_waves_512"
    N_PARALLEL_WAVES_1024 = "n_parallel_waves_1024"
    N_FINAL_PARALLEL_WAVE_UTILIZATION_64 = (
        "n_final_parallel_wave_utilization_64"
    )
    N_FINAL_PARALLEL_WAVE_UTILIZATION_128 = (
        "n_final_parallel_wave_utilization_128"
    )
    N_FINAL_PARALLEL_WAVE_UTILIZATION_256 = (
        "n_final_parallel_wave_utilization_256"
    )
    N_FINAL_PARALLEL_WAVE_UTILIZATION_512 = (
        "n_final_parallel_wave_utilization_512"
    )
    N_FINAL_PARALLEL_WAVE_UTILIZATION_1024 = (
        "n_final_parallel_wave_utilization_1024"
    )
    MN_PARALLEL_WAVES_64 = "mn_parallel_waves_64"
    MN_FINAL_PARALLEL_WAVE_UTILIZATION_64 = (
        "mn_final_parallel_wave_utilization_64"
    )
    N_TILE_ALIGNED_32 = "n_tile_aligned_32"
    N_TILE_ALIGNED_64 = "n_tile_aligned_64"
    N_TILE_ALIGNED_128 = "n_tile_aligned_128"
    N_TILE_ALIGNED_256 = "n_tile_aligned_256"
    N_TILE_ALIGNED_512 = "n_tile_aligned_512"
    N_TILE_ALIGNED_1024 = "n_tile_aligned_1024"


class BoundaryPlacement(str, Enum):
    """Unseen-value placement permitted between adjacent training features."""

    MIDPOINT = "midpoint"
    LOWER_EDGE = "lower_edge"


class ProfilerInfluence(str, Enum):
    """How authenticated profiler economics may influence tree fitting.

    Every strength is scored against raw held-out canonical timings. Including
    ``MEASURED_ONLY`` in every profiler-informed CV makes profiler evidence
    strictly additive: counters may select a better generic tree, but cannot
    force a domain away from a better timing-only model. The bounded variants
    let grouped CV choose how strongly to shrink fitting cost toward the
    leakage-controlled profiler prediction without crossing the measured
    five-percent pass/fail boundary. ``CROSS_FITTED_TEACHER`` instead lets the
    surrogate choose the held candidate directly. That choice is still scored
    only by held canonical timing and is distilled into a normal generic tree;
    the Python surrogate is never part of runtime dispatch.
    """

    MEASURED_ONLY = "measured_only"
    BOUNDED_PRIOR = "bounded_prior"
    BOUNDED_PRIOR_HALF = "bounded_prior_half"
    BOUNDED_PRIOR_FULL = "bounded_prior_full"
    CROSS_FITTED_TEACHER = "cross_fitted_teacher"


class FeaturePolicy(str, Enum):
    """Reviewed feature families selected only by grouped validation.

    Narrow policies keep each fitted tree compact and let grouped CV decide
    whether one launch discontinuity generalizes for a domain. The additive
    ``FULL_LAUNCH_GEOMETRY`` preserves the original N-only tile and worker-wave
    model. The row-grid variants add the CPU ``M * ceil(N / 64)`` workshare
    without replacing that proven tournament member. Feature expansion must be
    monotonic: a new family may win grouped CV, but it may never make an older
    certifiable family unavailable. Narrow policies remain in the tournament,
    so a larger search cannot displace them unless held-out timings prove it.
    """

    CONTINUOUS = "continuous"
    TILE_32 = "tile_32"
    TILE_64 = "tile_64"
    TILE_128 = "tile_128"
    TILE_256 = "tile_256"
    TILE_512 = "tile_512"
    TILE_1024 = "tile_1024"
    TILE_64_128 = "tile_64_128"
    PARALLEL_WAVE_SCHEDULES = "parallel_wave_schedules"
    ROW_GRID_PARALLEL_WAVE_SCHEDULES = "row_grid_parallel_wave_schedules"
    FULL_LAUNCH_GEOMETRY = "full_launch_geometry"
    FULL_ROW_GRID_LAUNCH_GEOMETRY = "full_row_grid_launch_geometry"


BOUNDARY_PLACEMENTS = (
    BoundaryPlacement.MIDPOINT,
    BoundaryPlacement.LOWER_EDGE,
)

PROFILER_INFLUENCES = (
    ProfilerInfluence.MEASURED_ONLY,
    ProfilerInfluence.BOUNDED_PRIOR,
    ProfilerInfluence.BOUNDED_PRIOR_HALF,
    ProfilerInfluence.BOUNDED_PRIOR_FULL,
    ProfilerInfluence.CROSS_FITTED_TEACHER,
)

TREE_PROFILER_INFLUENCES = tuple(
    influence
    for influence in PROFILER_INFLUENCES
    if influence != ProfilerInfluence.CROSS_FITTED_TEACHER
)

PROFILER_BLEND_WEIGHT_BY_INFLUENCE = {
    ProfilerInfluence.MEASURED_ONLY: 0.0,
    ProfilerInfluence.BOUNDED_PRIOR: PROFILER_REGRET_BLEND_WEIGHT,
    ProfilerInfluence.BOUNDED_PRIOR_HALF: 0.5,
    ProfilerInfluence.BOUNDED_PRIOR_FULL: 1.0,
}

FEATURE_POLICIES = tuple(FeaturePolicy)


N_TILE_WIDTH_BY_AXIS = {
    FeatureAxis.N_TILES_32: 32,
    FeatureAxis.N_TILES_64: 64,
    FeatureAxis.N_TILES_128: 128,
    FeatureAxis.N_TILES_256: 256,
    FeatureAxis.N_TILES_512: 512,
    FeatureAxis.N_TILES_1024: 1024,
}

K_GROUPS_PER_N_TILE_WIDTH_BY_AXIS = {
    FeatureAxis.K_GROUPS_PER_N_TILE_32: 32,
    FeatureAxis.K_GROUPS_PER_N_TILE_64: 64,
    FeatureAxis.K_GROUPS_PER_N_TILE_128: 128,
    FeatureAxis.K_GROUPS_PER_N_TILE_256: 256,
    FeatureAxis.K_GROUPS_PER_N_TILE_512: 512,
    FeatureAxis.K_GROUPS_PER_N_TILE_1024: 1024,
}

N_FINAL_TILE_WIDTH_BY_AXIS = {
    FeatureAxis.N_FINAL_TILE_VALUES_32: 32,
    FeatureAxis.N_FINAL_TILE_VALUES_64: 64,
    FeatureAxis.N_FINAL_TILE_VALUES_128: 128,
    FeatureAxis.N_FINAL_TILE_VALUES_256: 256,
    FeatureAxis.N_FINAL_TILE_VALUES_512: 512,
    FeatureAxis.N_FINAL_TILE_VALUES_1024: 1024,
}

K_FINAL_TILE_WIDTH_BY_AXIS = {
    FeatureAxis.K_FINAL_TILE_VALUES_32: 32,
    FeatureAxis.K_FINAL_TILE_VALUES_64: 64,
    FeatureAxis.K_FINAL_TILE_VALUES_128: 128,
    FeatureAxis.K_FINAL_TILE_VALUES_256: 256,
    FeatureAxis.K_FINAL_TILE_VALUES_512: 512,
    FeatureAxis.K_FINAL_TILE_VALUES_1024: 1024,
}

N_TILE_UTILIZATION_WIDTH_BY_AXIS = {
    FeatureAxis.N_TILE_UTILIZATION_32: 32,
    FeatureAxis.N_TILE_UTILIZATION_64: 64,
    FeatureAxis.N_TILE_UTILIZATION_128: 128,
    FeatureAxis.N_TILE_UTILIZATION_256: 256,
    FeatureAxis.N_TILE_UTILIZATION_512: 512,
    FeatureAxis.N_TILE_UTILIZATION_1024: 1024,
}

N_TILE_ALIGNED_WIDTH_BY_AXIS = {
    FeatureAxis.N_TILE_ALIGNED_32: 32,
    FeatureAxis.N_TILE_ALIGNED_64: 64,
    FeatureAxis.N_TILE_ALIGNED_128: 128,
    FeatureAxis.N_TILE_ALIGNED_256: 256,
    FeatureAxis.N_TILE_ALIGNED_512: 512,
    FeatureAxis.N_TILE_ALIGNED_1024: 1024,
}

N_PARALLEL_WAVE_WIDTH_BY_AXIS = {
    FeatureAxis.N_PARALLEL_WAVES_64: 64,
    FeatureAxis.N_PARALLEL_WAVES_128: 128,
    FeatureAxis.N_PARALLEL_WAVES_256: 256,
    FeatureAxis.N_PARALLEL_WAVES_512: 512,
    FeatureAxis.N_PARALLEL_WAVES_1024: 1024,
}

N_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS = {
    FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_64: 64,
    FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_128: 128,
    FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_256: 256,
    FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_512: 512,
    FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_1024: 1024,
}

# The CPU row-chunk prefill kernel publishes one independent task for every
# `(row, 64-column chunk)` pair.  That launch is fundamentally different from
# the two-row kernels above, whose OpenMP workshare contains only N blocks and
# iterates rows inside each worker.  Keep the physical row-grid geometry as an
# explicit feature instead of asking continuous N/K thresholds to approximate
# a periodic `M * ceil(N / 64) mod threads` occupancy boundary.
MN_PARALLEL_WAVE_WIDTH_BY_AXIS = {
    FeatureAxis.MN_PARALLEL_WAVES_64: 64,
}

MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS = {
    FeatureAxis.MN_FINAL_PARALLEL_WAVE_UTILIZATION_64: 64,
}

# These immutable memberships participate in every threshold construction.
# Building fresh sets in `FeatureThreshold.__post_init__` used to turn the
# quadratic all-pairs metadata pass into millions of short-lived Python
# allocations. Keep one canonical inventory shared by ordinary tree fitting,
# accelerator metadata, and emitted-threshold validation.
N_PARALLEL_WAVE_AXES = frozenset((
    *N_PARALLEL_WAVE_WIDTH_BY_AXIS,
    *N_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS,
))
MN_PARALLEL_WAVE_AXES = frozenset((
    *MN_PARALLEL_WAVE_WIDTH_BY_AXIS,
    *MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS,
))
PARALLEL_WAVE_AXES = N_PARALLEL_WAVE_AXES | MN_PARALLEL_WAVE_AXES


BASE_FEATURE_AXES = (
    FeatureAxis.AGGREGATE_N,
    FeatureAxis.K,
    FeatureAxis.WORK_ITEMS,
    FeatureAxis.ASPECT_RATIO,
)

FEATURE_AXES_BY_POLICY = {
    FeaturePolicy.CONTINUOUS: BASE_FEATURE_AXES,
    FeaturePolicy.TILE_32: (
        *BASE_FEATURE_AXES,
        FeatureAxis.N_TILES_32,
        FeatureAxis.K_GROUPS_PER_N_TILE_32,
        FeatureAxis.N_FINAL_TILE_VALUES_32,
        FeatureAxis.K_FINAL_TILE_VALUES_32,
        FeatureAxis.N_TILE_UTILIZATION_32,
        FeatureAxis.N_TILE_ALIGNED_32,
    ),
    FeaturePolicy.TILE_64: (
        *BASE_FEATURE_AXES,
        FeatureAxis.N_TILES_64,
        FeatureAxis.K_GROUPS_PER_N_TILE_64,
        FeatureAxis.N_FINAL_TILE_VALUES_64,
        FeatureAxis.K_FINAL_TILE_VALUES_64,
        FeatureAxis.N_TILE_UTILIZATION_64,
        FeatureAxis.N_TILE_ALIGNED_64,
        FeatureAxis.N_PARALLEL_WAVES_64,
        FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_64,
    ),
    FeaturePolicy.TILE_128: (
        *BASE_FEATURE_AXES,
        FeatureAxis.N_TILES_128,
        FeatureAxis.K_GROUPS_PER_N_TILE_128,
        FeatureAxis.N_FINAL_TILE_VALUES_128,
        FeatureAxis.K_FINAL_TILE_VALUES_128,
        FeatureAxis.N_TILE_UTILIZATION_128,
        FeatureAxis.N_TILE_ALIGNED_128,
        FeatureAxis.N_PARALLEL_WAVES_128,
        FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_128,
    ),
    FeaturePolicy.TILE_256: (
        *BASE_FEATURE_AXES,
        FeatureAxis.N_TILES_256,
        FeatureAxis.K_GROUPS_PER_N_TILE_256,
        FeatureAxis.N_FINAL_TILE_VALUES_256,
        FeatureAxis.K_FINAL_TILE_VALUES_256,
        FeatureAxis.N_TILE_UTILIZATION_256,
        FeatureAxis.N_TILE_ALIGNED_256,
        FeatureAxis.N_PARALLEL_WAVES_256,
        FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_256,
    ),
    FeaturePolicy.TILE_512: (
        *BASE_FEATURE_AXES,
        FeatureAxis.N_TILES_512,
        FeatureAxis.K_GROUPS_PER_N_TILE_512,
        FeatureAxis.N_FINAL_TILE_VALUES_512,
        FeatureAxis.K_FINAL_TILE_VALUES_512,
        FeatureAxis.N_TILE_UTILIZATION_512,
        FeatureAxis.N_TILE_ALIGNED_512,
        FeatureAxis.N_PARALLEL_WAVES_512,
        FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_512,
    ),
    FeaturePolicy.TILE_1024: (
        *BASE_FEATURE_AXES,
        FeatureAxis.N_TILES_1024,
        FeatureAxis.K_GROUPS_PER_N_TILE_1024,
        FeatureAxis.N_FINAL_TILE_VALUES_1024,
        FeatureAxis.K_FINAL_TILE_VALUES_1024,
        FeatureAxis.N_TILE_UTILIZATION_1024,
        FeatureAxis.N_TILE_ALIGNED_1024,
        FeatureAxis.N_PARALLEL_WAVES_1024,
        FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_1024,
    ),
    FeaturePolicy.TILE_64_128: (
        *BASE_FEATURE_AXES,
        FeatureAxis.N_TILES_64,
        FeatureAxis.K_GROUPS_PER_N_TILE_64,
        FeatureAxis.N_FINAL_TILE_VALUES_64,
        FeatureAxis.K_FINAL_TILE_VALUES_64,
        FeatureAxis.N_TILE_UTILIZATION_64,
        FeatureAxis.N_TILE_ALIGNED_64,
        FeatureAxis.N_PARALLEL_WAVES_64,
        FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_64,
        FeatureAxis.N_TILES_128,
        FeatureAxis.K_GROUPS_PER_N_TILE_128,
        FeatureAxis.N_FINAL_TILE_VALUES_128,
        FeatureAxis.K_FINAL_TILE_VALUES_128,
        FeatureAxis.N_TILE_UTILIZATION_128,
        FeatureAxis.N_TILE_ALIGNED_128,
        FeatureAxis.N_PARALLEL_WAVES_128,
        FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_128,
    ),
    FeaturePolicy.PARALLEL_WAVE_SCHEDULES: (
        *BASE_FEATURE_AXES,
        *N_PARALLEL_WAVE_WIDTH_BY_AXIS,
        *N_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS,
    ),
    FeaturePolicy.ROW_GRID_PARALLEL_WAVE_SCHEDULES: (
        *BASE_FEATURE_AXES,
        *N_PARALLEL_WAVE_WIDTH_BY_AXIS,
        *N_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS,
        *MN_PARALLEL_WAVE_WIDTH_BY_AXIS,
        *MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS,
    ),
    FeaturePolicy.FULL_LAUNCH_GEOMETRY: tuple(
        axis
        for axis in FeatureAxis
        if axis not in {
            *MN_PARALLEL_WAVE_WIDTH_BY_AXIS,
            *MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS,
        }
    ),
    FeaturePolicy.FULL_ROW_GRID_LAUNCH_GEOMETRY: tuple(FeatureAxis),
}


FEATURE_AXIS_PRIORITY = {
    FeatureAxis.K: 0,
    FeatureAxis.AGGREGATE_N: 1,
    FeatureAxis.WORK_ITEMS: 2,
    FeatureAxis.ASPECT_RATIO: 3,
    FeatureAxis.N_TILES_32: 4,
    FeatureAxis.N_TILES_64: 5,
    FeatureAxis.N_TILES_128: 6,
    FeatureAxis.N_TILES_256: 7,
    FeatureAxis.N_TILES_512: 8,
    FeatureAxis.N_TILES_1024: 9,
    FeatureAxis.K_GROUPS_PER_N_TILE_32: 10,
    FeatureAxis.K_GROUPS_PER_N_TILE_64: 11,
    FeatureAxis.K_GROUPS_PER_N_TILE_128: 12,
    FeatureAxis.K_GROUPS_PER_N_TILE_256: 13,
    FeatureAxis.K_GROUPS_PER_N_TILE_512: 14,
    FeatureAxis.K_GROUPS_PER_N_TILE_1024: 15,
    FeatureAxis.N_FINAL_TILE_VALUES_32: 16,
    FeatureAxis.N_FINAL_TILE_VALUES_64: 17,
    FeatureAxis.N_FINAL_TILE_VALUES_128: 18,
    FeatureAxis.N_FINAL_TILE_VALUES_256: 19,
    FeatureAxis.N_FINAL_TILE_VALUES_512: 20,
    FeatureAxis.N_FINAL_TILE_VALUES_1024: 21,
    FeatureAxis.N_TILE_UTILIZATION_32: 22,
    FeatureAxis.N_TILE_UTILIZATION_64: 23,
    FeatureAxis.N_TILE_UTILIZATION_128: 24,
    FeatureAxis.N_TILE_UTILIZATION_256: 25,
    FeatureAxis.N_TILE_UTILIZATION_512: 26,
    FeatureAxis.N_TILE_UTILIZATION_1024: 27,
    FeatureAxis.N_PARALLEL_WAVES_64: 28,
    FeatureAxis.N_PARALLEL_WAVES_128: 29,
    FeatureAxis.N_PARALLEL_WAVES_256: 30,
    FeatureAxis.N_PARALLEL_WAVES_512: 31,
    FeatureAxis.N_PARALLEL_WAVES_1024: 32,
    FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_64: 33,
    FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_128: 34,
    FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_256: 35,
    FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_512: 36,
    FeatureAxis.N_FINAL_PARALLEL_WAVE_UTILIZATION_1024: 37,
    FeatureAxis.MN_PARALLEL_WAVES_64: 38,
    FeatureAxis.MN_FINAL_PARALLEL_WAVE_UTILIZATION_64: 39,
    FeatureAxis.N_TILE_ALIGNED_32: 40,
    FeatureAxis.N_TILE_ALIGNED_64: 41,
    FeatureAxis.N_TILE_ALIGNED_128: 42,
    FeatureAxis.N_TILE_ALIGNED_256: 43,
    FeatureAxis.N_TILE_ALIGNED_512: 44,
    FeatureAxis.N_TILE_ALIGNED_1024: 45,
    FeatureAxis.K_FINAL_TILE_VALUES_32: 46,
    FeatureAxis.K_FINAL_TILE_VALUES_64: 47,
    FeatureAxis.K_FINAL_TILE_VALUES_128: 48,
    FeatureAxis.K_FINAL_TILE_VALUES_256: 49,
    FeatureAxis.K_FINAL_TILE_VALUES_512: 50,
    FeatureAxis.K_FINAL_TILE_VALUES_1024: 51,
}


def _n_tile_count(aggregate_n: int, tile_width: int) -> int:
    """Return the exact positive ceil-div N tile count used by launchers."""

    return (aggregate_n + tile_width - 1) // tile_width


@dataclass(frozen=True, order=True)
class FeatureThreshold:
    """Exact rational threshold used by one binary decision-tree node."""

    axis: FeatureAxis
    numerator: int
    denominator: int = 1
    parallelism_width: int = 1
    task_multiplier: int = 1

    def __post_init__(self) -> None:
        if self.denominator <= 0:
            raise ValueError("feature-threshold denominator must be positive")
        if self.numerator < 0:
            raise ValueError("feature-threshold numerator must be non-negative")
        if self.parallelism_width <= 0:
            raise ValueError("feature-threshold parallelism width must be positive")
        if self.task_multiplier <= 0:
            raise ValueError("feature-threshold task multiplier must be positive")
        if self.axis not in PARALLEL_WAVE_AXES and (
            self.parallelism_width != 1 or self.task_multiplier != 1
        ):
            raise ValueError(
                "parallelism metadata is valid only for parallel-wave features"
            )
        if self.axis in N_PARALLEL_WAVE_AXES and self.task_multiplier != 1:
            raise ValueError(
                "N-only parallel-wave features cannot multiply task count"
            )
        # M=1 is a legitimate trained domain (decode/GEMV), not an omitted
        # multiplier. In that regime the MN axes reduce to their N-only
        # counterparts; retaining the exact multiplier keeps the common
        # feature tournament total across CPU, CUDA, and ROCm domains.

    def matches_less_equal(self, aggregate_n: int, k: int) -> bool:
        """Evaluate the threshold without floating-point boundary drift."""

        if aggregate_n <= 0 or k <= 0:
            return False
        if self.axis == FeatureAxis.AGGREGATE_N:
            return aggregate_n * self.denominator <= self.numerator
        if self.axis == FeatureAxis.K:
            return k * self.denominator <= self.numerator
        if self.axis == FeatureAxis.WORK_ITEMS:
            return aggregate_n * k * self.denominator <= self.numerator
        if self.axis == FeatureAxis.ASPECT_RATIO:
            return aggregate_n * self.denominator <= k * self.numerator
        if self.axis in N_TILE_WIDTH_BY_AXIS:
            tiles = _n_tile_count(
                aggregate_n, N_TILE_WIDTH_BY_AXIS[self.axis]
            )
            return tiles * self.denominator <= self.numerator
        if self.axis in K_GROUPS_PER_N_TILE_WIDTH_BY_AXIS:
            tiles = _n_tile_count(
                aggregate_n,
                K_GROUPS_PER_N_TILE_WIDTH_BY_AXIS[self.axis],
            )
            # K is measured in 32-value NativeVNNI groups. Keep K itself on
            # the left so the comparison remains exact even before schema
            # validation has established K % 32 == 0.
            return (
                k * self.denominator
                <= 32 * tiles * self.numerator
            )
        if self.axis in N_FINAL_TILE_WIDTH_BY_AXIS:
            width = N_FINAL_TILE_WIDTH_BY_AXIS[self.axis]
            final_tile_values = (aggregate_n - 1) % width + 1
            return final_tile_values * self.denominator <= self.numerator
        if self.axis in K_FINAL_TILE_WIDTH_BY_AXIS:
            width = K_FINAL_TILE_WIDTH_BY_AXIS[self.axis]
            final_tile_values = (k - 1) % width + 1
            return final_tile_values * self.denominator <= self.numerator
        if self.axis in N_TILE_UTILIZATION_WIDTH_BY_AXIS:
            width = N_TILE_UTILIZATION_WIDTH_BY_AXIS[self.axis]
            tiles = _n_tile_count(aggregate_n, width)
            return (
                aggregate_n * self.denominator
                <= tiles * width * self.numerator
            )
        if self.axis in N_TILE_ALIGNED_WIDTH_BY_AXIS:
            width = N_TILE_ALIGNED_WIDTH_BY_AXIS[self.axis]
            aligned = int(aggregate_n % width == 0)
            return aligned * self.denominator <= self.numerator
        if self.axis in N_PARALLEL_WAVE_WIDTH_BY_AXIS:
            tasks = _n_tile_count(
                aggregate_n, N_PARALLEL_WAVE_WIDTH_BY_AXIS[self.axis]
            )
            waves = _n_tile_count(tasks, self.parallelism_width)
            return waves * self.denominator <= self.numerator
        if self.axis in N_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS:
            tasks = _n_tile_count(
                aggregate_n,
                N_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS[self.axis],
            )
            final_wave_tasks = (tasks - 1) % self.parallelism_width + 1
            return (
                final_wave_tasks * self.denominator
                <= self.parallelism_width * self.numerator
            )
        if self.axis in MN_PARALLEL_WAVE_WIDTH_BY_AXIS:
            tasks = self.task_multiplier * _n_tile_count(
                aggregate_n, MN_PARALLEL_WAVE_WIDTH_BY_AXIS[self.axis]
            )
            waves = _n_tile_count(tasks, self.parallelism_width)
            return waves * self.denominator <= self.numerator
        if self.axis in MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS:
            tasks = self.task_multiplier * _n_tile_count(
                aggregate_n,
                MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS[self.axis],
            )
            final_wave_tasks = (tasks - 1) % self.parallelism_width + 1
            return (
                final_wave_tasks * self.denominator
                <= self.parallelism_width * self.numerator
            )
        raise ValueError(f"unknown NativeVNNI feature axis {self.axis!r}")


def _accelerated_threshold_descriptor(
    threshold: FeatureThreshold,
) -> TreeThresholdDescriptor:
    """Translate one policy threshold into the exact shared GPU operation ABI."""

    operation = None
    tile_width = 0
    if threshold.axis == FeatureAxis.AGGREGATE_N:
        operation = TreeThresholdOperation.AGGREGATE_N
    elif threshold.axis == FeatureAxis.K:
        operation = TreeThresholdOperation.K
    elif threshold.axis == FeatureAxis.WORK_ITEMS:
        operation = TreeThresholdOperation.WORK_ITEMS
    elif threshold.axis == FeatureAxis.ASPECT_RATIO:
        operation = TreeThresholdOperation.ASPECT_RATIO
    elif threshold.axis in N_TILE_WIDTH_BY_AXIS:
        operation = TreeThresholdOperation.N_TILES
        tile_width = N_TILE_WIDTH_BY_AXIS[threshold.axis]
    elif threshold.axis in K_GROUPS_PER_N_TILE_WIDTH_BY_AXIS:
        operation = TreeThresholdOperation.K_GROUPS_PER_N_TILE
        tile_width = K_GROUPS_PER_N_TILE_WIDTH_BY_AXIS[threshold.axis]
    elif threshold.axis in N_FINAL_TILE_WIDTH_BY_AXIS:
        operation = TreeThresholdOperation.N_FINAL_TILE
        tile_width = N_FINAL_TILE_WIDTH_BY_AXIS[threshold.axis]
    elif threshold.axis in K_FINAL_TILE_WIDTH_BY_AXIS:
        operation = TreeThresholdOperation.K_FINAL_TILE
        tile_width = K_FINAL_TILE_WIDTH_BY_AXIS[threshold.axis]
    elif threshold.axis in N_TILE_UTILIZATION_WIDTH_BY_AXIS:
        operation = TreeThresholdOperation.N_TILE_UTILIZATION
        tile_width = N_TILE_UTILIZATION_WIDTH_BY_AXIS[threshold.axis]
    elif threshold.axis in N_TILE_ALIGNED_WIDTH_BY_AXIS:
        operation = TreeThresholdOperation.N_TILE_ALIGNED
        tile_width = N_TILE_ALIGNED_WIDTH_BY_AXIS[threshold.axis]
    elif threshold.axis in N_PARALLEL_WAVE_WIDTH_BY_AXIS:
        operation = TreeThresholdOperation.N_PARALLEL_WAVES
        tile_width = N_PARALLEL_WAVE_WIDTH_BY_AXIS[threshold.axis]
    elif threshold.axis in N_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS:
        operation = TreeThresholdOperation.N_FINAL_PARALLEL_WAVE_UTILIZATION
        tile_width = N_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS[threshold.axis]
    elif threshold.axis in MN_PARALLEL_WAVE_WIDTH_BY_AXIS:
        operation = TreeThresholdOperation.MN_PARALLEL_WAVES
        tile_width = MN_PARALLEL_WAVE_WIDTH_BY_AXIS[threshold.axis]
    elif threshold.axis in MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS:
        operation = TreeThresholdOperation.MN_FINAL_PARALLEL_WAVE_UTILIZATION
        tile_width = MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS[threshold.axis]
    if operation is None:
        raise ValueError(
            f"feature axis {threshold.axis.value!r} has no GPU evaluator"
        )
    return TreeThresholdDescriptor(
        axis_priority=FEATURE_AXIS_PRIORITY[threshold.axis],
        operation=operation,
        tile_width=tile_width,
        numerator=threshold.numerator,
        denominator=threshold.denominator,
        parallelism_width=threshold.parallelism_width,
        task_multiplier=threshold.task_multiplier,
    )


def _accelerated_feature_axis_descriptor(
    axis: FeatureAxis,
) -> TreeFeatureAxisDescriptor:
    """Translate one policy feature into compact device-generation metadata."""

    exemplar = FeatureThreshold(axis, 0, 1, 1, 1)
    if axis in N_PARALLEL_WAVE_AXES:
        exemplar = FeatureThreshold(axis, 0, 1, 1, 1)
    descriptor = _accelerated_threshold_descriptor(exemplar)
    return TreeFeatureAxisDescriptor(
        axis_priority=descriptor.axis_priority,
        operation=descriptor.operation,
        tile_width=descriptor.tile_width,
    )


def _accelerated_boundary_placement(
    placement: BoundaryPlacement,
) -> TreeBoundaryPlacement:
    """Map the reviewed policy enum to the stable device ABI value."""

    return (
        TreeBoundaryPlacement.MIDPOINT
        if placement == BoundaryPlacement.MIDPOINT
        else TreeBoundaryPlacement.LOWER_EDGE
    )


FEATURE_AXIS_BY_PRIORITY = {
    priority: axis for axis, priority in FEATURE_AXIS_PRIORITY.items()
}


def _feature_threshold_from_accelerated_descriptor(
    descriptor: TreeThresholdDescriptor,
) -> FeatureThreshold:
    """Reconstruct one emitted predicate and verify its axis operation."""

    try:
        axis = FEATURE_AXIS_BY_PRIORITY[descriptor.axis_priority]
    except KeyError as error:
        raise RuntimeError(
            "GPU tree returned an unknown feature-axis priority"
        ) from error
    threshold = FeatureThreshold(
        axis,
        descriptor.numerator,
        descriptor.denominator,
        descriptor.parallelism_width,
        descriptor.task_multiplier,
    )
    expected = _accelerated_threshold_descriptor(threshold)
    if (
        expected.operation != descriptor.operation
        or expected.tile_width != descriptor.tile_width
    ):
        raise RuntimeError("GPU tree threshold operation changed its feature axis")
    return threshold


@dataclass(frozen=True, order=True)
class FeaturePredicate:
    """One path predicate selecting either side of a feature threshold."""

    threshold: FeatureThreshold
    require_less_equal: bool

    def matches(self, aggregate_n: int, k: int) -> bool:
        """Return whether dimensions follow this recorded tree edge."""

        comparison = self.threshold.matches_less_equal(aggregate_n, k)
        return comparison if self.require_less_equal else not comparison


@dataclass(frozen=True)
class CandidatePointCost:
    """Worst alias regret for one candidate at one logical shape point."""

    runtime_key: RuntimeKey
    shape_group_id: str
    candidate_id: str
    max_surface_regret: float
    p95_surface_regret: float
    mean_surface_regret: float
    profiler_predicted_regret: float | None = None
    profiler_blend_weight: float = PROFILER_REGRET_BLEND_WEIGHT
    cross_fitted_selection_regret: float | None = None

    @property
    def selection_regret(self) -> float:
        """Return a bounded profiler-informed development fitting cost.

        Canonical timing always owns the hard five-percent pass/fail topology.
        Within that topology, a separately profiled launch may move the fitting
        cost one quarter of the way toward its Extra-Trees prediction. The
        absolute adjustment is capped at one complete regret budget so a poorly
        calibrated extrapolation cannot dominate a measured candidate edge.

        This symmetric shrinkage is intentional. It can rank two measured
        passing candidates and can smooth a noisy failing outlier, unlike the
        former one-sided penalty. Clamping to the original side of the budget
        ensures that neither operation manufactures certification evidence.

        Final publication fitting is a distinct second stage.  Its target is
        the candidate selected for this point while the point was held out of
        grouped CV, represented by ``cross_fitted_selection_regret``.  That
        target takes precedence because profiler evidence has already affected
        the fold model that produced it.  Measured p95/max/mean fields remain
        untouched and continue to own every promotion and diagnostic gate.
        """

        if self.cross_fitted_selection_regret is not None:
            if (
                not math.isfinite(self.cross_fitted_selection_regret)
                or self.cross_fitted_selection_regret < 0.0
            ):
                raise ValueError(
                    "cross-fitted selection regret must be finite and non-negative"
                )
            return self.cross_fitted_selection_regret

        prediction = self.profiler_predicted_regret
        if prediction is None:
            return self.max_surface_regret
        if not 0.0 <= self.profiler_blend_weight <= 1.0:
            raise ValueError("profiler blend weight must be in [0, 1]")
        correction = self.profiler_blend_weight * (
            prediction - self.max_surface_regret
        )
        correction = max(
            -PROFILER_MAX_REGRET_ADJUSTMENT,
            min(PROFILER_MAX_REGRET_ADJUSTMENT, correction),
        )
        informed = max(0.0, self.max_surface_regret + correction)
        if self.max_surface_regret < GENERIC_REGRET_BUDGET:
            return min(
                math.nextafter(GENERIC_REGRET_BUDGET, 0.0),
                informed,
            )
        return max(
            math.nextafter(GENERIC_REGRET_BUDGET, math.inf),
            informed,
        )


@dataclass(frozen=True)
class GenericDispatchRule:
    """One bounded decision-tree leaf selected for a generic domain."""

    domain: GenericDomain
    predicates: tuple[FeaturePredicate, ...]
    candidate_id: str
    arithmetic_fingerprint: str
    development_shape_groups: tuple[str, ...]
    development_max_regret: float
    development_p95_regret: float
    development_mean_regret: float

    def matches(self, aggregate_n: int, k: int) -> bool:
        """Evaluate every root-to-leaf predicate for one runtime shape."""

        return all(
            predicate.matches(aggregate_n, k)
            for predicate in self.predicates
        )

    def identity(self) -> tuple:
        """Return a stable hashable identity for coverage accounting."""

        return (
            self.domain,
            self.predicates,
            self.candidate_id,
        )


@dataclass(frozen=True)
class CrossValidationCell:
    """One held-out shape decision retained for paired confirmation."""

    runtime_key: RuntimeKey
    shape_group_id: str
    selected_candidate_id: str
    exact_candidate_id: str
    observed_broad_regret: float


@dataclass(frozen=True)
class DomainCrossValidation:
    """Grouped development-CV result freezing one domain's leaf budget."""

    domain: GenericDomain
    selected_feature_policy: FeaturePolicy
    selected_max_leaves: int
    selected_boundary_placement: BoundaryPlacement
    selected_profiler_influence: ProfilerInfluence
    fold_count: int
    shape_group_count: int
    required_point_count: int
    covered_point_count: int
    max_regret: float
    p95_regret: float
    mean_regret: float
    worst_shape_group_id: str
    worst_aggregate_n: int
    worst_k: int
    worst_selected_candidate_id: str
    worst_exact_candidate_id: str
    cells: tuple[CrossValidationCell, ...]
    competitive_cells: tuple[CrossValidationCell, ...] = ()
    publication_feature_policy: FeaturePolicy | None = None
    publication_max_leaves: int | None = None
    publication_boundary_placement: BoundaryPlacement | None = None
    publication_oof_p95_regret: float | None = None
    publication_oof_mean_regret: float | None = None
    publication_oof_mismatch_count: int | None = None


@dataclass(frozen=True)
class DomainPromotionDiagnostic:
    """Why one generic domain failed before sealed certification.

    Development CV and the final all-development tree are separate promotion
    gates.  Retaining both surfaces prevents refinement planning from treating
    a final-tree leaf failure as another held-out-CV evidence gap.
    """

    domain: GenericDomain
    rejection_stage: str
    cv_required_point_count: int
    cv_covered_point_count: int
    cv_p95_regret: float | None
    cv_max_regret: float | None
    final_cross_fitted_p95_regret: float | None
    final_rule_count: int
    final_worst_p95_regret: float | None
    final_worst_max_regret: float | None
    final_worst_shape_group_id: str
    final_worst_aggregate_n: int | None
    final_worst_k: int | None
    final_worst_candidate_id: str


def domain_cross_validation_is_promotable(
    validation: DomainCrossValidation,
    *,
    p95_regret: float = GENERIC_REGRET_BUDGET,
) -> bool:
    """Return whether one domain earned a production generic rule.

    Failed development CV remains valuable refinement evidence, but it cannot
    be converted into a generic dispatch claim and deferred to the sealed gate.
    Exact overlays are additive and never repair a missing generic domain, so
    every production compiler must reject ``unpromoted_domains`` before seal.
    """

    return (
        domain_cross_validation_has_complete_coverage(validation)
        and validation.p95_regret < p95_regret
        and (
            validation.publication_oof_p95_regret is None
            or validation.publication_oof_p95_regret < p95_regret
        )
    )


def domain_cross_validation_has_complete_coverage(
    validation: DomainCrossValidation,
) -> bool:
    """Return whether CV made a real decision over every required point.

    Coverage is a structural requirement, unlike the per-domain performance
    budget. A corpus-level promotion quota may admit a small number of fully
    covered domains whose best learned rule remains over budget, but it must
    never turn missing evidence into a dispatch rule.
    """

    return (
        validation.required_point_count > 0
        and validation.covered_point_count == validation.required_point_count
    )


def domain_promotion_quota_is_satisfied(
    passing_domain_count: int,
    required_domain_count: int,
) -> bool:
    """Return whether at least 99% of required domains pass strict p95 CV.

    Integer cross multiplication makes the inclusive corpus-level boundary
    exact: 99 of 100 domains passes, while 98 of 99 does not. Individual
    domains still pass only when their measured p95 regret is strictly below
    :data:`GENERIC_REGRET_BUDGET`.
    """

    if required_domain_count <= 0:
        return False
    if passing_domain_count < 0 or passing_domain_count > required_domain_count:
        raise ValueError("passing domain count is outside the required domain set")
    return (
        passing_domain_count * GENERIC_DOMAIN_PROMOTION_DENOMINATOR
        >= required_domain_count * GENERIC_DOMAIN_PROMOTION_NUMERATOR
    )


@dataclass(frozen=True)
class GenericPolicy:
    """Generic leaves plus explicit per-domain performance diagnostics.

    ``unpromoted_domains`` contains only corpus-blocking obligations. A policy
    that satisfies the 99% performance quota has total generic rules and an
    empty blocking set while retaining diagnostics for its sub-1% over-budget
    domains.
    """

    rules: tuple[GenericDispatchRule, ...]
    unpromoted_domains: tuple[GenericDomain, ...]
    cross_validation: tuple[DomainCrossValidation, ...] = ()
    promotion_diagnostics: tuple[DomainPromotionDiagnostic, ...] = ()

    def resolve(
        self,
        domain: GenericDomain,
        aggregate_n: int,
        k: int,
    ) -> GenericDispatchRule | None:
        """Resolve one generic key without consulting any exact overlay."""

        matches = [
            rule
            for rule in self.rules
            if rule.domain == domain and rule.matches(aggregate_n, k)
        ]
        if len(matches) > 1:
            raise ValueError(f"generic policy leaves overlap for {domain}")
        return matches[0] if matches else None


@dataclass
class _GenericPartitionNode:
    """One validation-only node reconstructed from flattened generic leaves."""

    threshold: FeatureThreshold | None = None
    leaf: GenericDispatchRule | None = None
    less_equal: "_GenericPartitionNode | None" = None
    greater: "_GenericPartitionNode | None" = None


def validate_generic_rule_partition(
    rules: Iterable[GenericDispatchRule],
) -> None:
    """Prove flattened rules are total for every unseen positive geometry.

    A few sampled shapes cannot establish generic dispatch. The stronger proof
    reconstructs the original binary decision tree from each root-to-leaf
    predicate sequence. Every internal node must name one exact threshold and
    own both complementary branches. Every terminal node must own exactly one
    leaf and no children. Since :meth:`FeatureThreshold.matches_less_equal`
    returns one Boolean for every positive ``(N, K)``, this recursive invariant
    proves that geometries below, above, and between all fitted observations
    resolve to exactly one rule.

    The proof is deliberately independent of threshold axis. It therefore
    covers direct geometry, aspect ratio, work size, tile count, and launch-wave
    features without reducing them to a finite test grid.
    """

    materialized = tuple(rules)
    if not materialized:
        raise ValueError("generic rule partition is empty")
    domains = {rule.domain for rule in materialized}
    if len(domains) != 1:
        raise ValueError("generic rule partition mixes policy domains")

    root = _GenericPartitionNode()
    for rule in materialized:
        node = root
        for predicate in rule.predicates:
            if node.leaf is not None:
                raise ValueError(
                    "generic rule partition extends an existing terminal leaf"
                )
            if node.threshold is None:
                node.threshold = predicate.threshold
            elif node.threshold != predicate.threshold:
                raise ValueError(
                    "generic rule partition has inconsistent sibling thresholds"
                )
            branch_name = (
                "less_equal" if predicate.require_less_equal else "greater"
            )
            child = getattr(node, branch_name)
            if child is None:
                child = _GenericPartitionNode()
                setattr(node, branch_name, child)
            node = child
        if node.leaf is not None:
            raise ValueError("generic rule partition repeats one terminal path")
        if node.threshold is not None:
            raise ValueError("generic rule partition terminates at an internal node")
        node.leaf = rule

    def validate_node(node: _GenericPartitionNode) -> None:
        """Recursively require a terminal leaf or a complete binary split."""

        if node.leaf is not None:
            if (
                node.threshold is not None
                or node.less_equal is not None
                or node.greater is not None
            ):
                raise ValueError(
                    "generic rule terminal also contains decision children"
                )
            return
        if (
            node.threshold is None
            or node.less_equal is None
            or node.greater is None
        ):
            raise ValueError(
                "generic rule partition leaves unseen geometry uncovered"
            )
        validate_node(node.less_equal)
        validate_node(node.greater)

    validate_node(root)


def _runtime_key_mapping(key: RuntimeKey) -> dict[str, object]:
    """Serialize one exact runtime key without relying on enum repr strings."""

    return {
        "backend": key.backend.value,
        "architecture_class": key.architecture_class,
        "semantic_contract": key.semantic_contract.value,
        "operation_kind": key.operation_kind,
        "bundle_signature": key.bundle_signature,
        "projection_n_vector": list(key.projection_n_vector),
        "prepared_family_id": key.prepared_family_id,
        "packing_abi": key.packing_abi,
        "runtime_codebook_id": key.runtime_codebook_id,
        "execution_mode": key.execution_mode.value,
        "m": key.m,
        "aggregate_n": key.aggregate_n,
        "k": key.k,
    }


def _runtime_key_from_mapping(raw: Mapping[str, object]) -> RuntimeKey:
    """Reconstruct one validated runtime key from a fit-cache record."""

    return RuntimeKey(
        backend=Backend(str(raw["backend"])),
        architecture_class=str(raw["architecture_class"]),
        semantic_contract=SemanticContract(str(raw["semantic_contract"])),
        operation_kind=str(raw["operation_kind"]),
        bundle_signature=str(raw["bundle_signature"]),
        projection_n_vector=tuple(
            int(value) for value in raw["projection_n_vector"]
        ),
        prepared_family_id=str(raw["prepared_family_id"]),
        packing_abi=str(raw["packing_abi"]),
        runtime_codebook_id=int(raw["runtime_codebook_id"]),
        execution_mode=ExecutionMode(str(raw["execution_mode"])),
        m=int(raw["m"]),
        aggregate_n=int(raw["aggregate_n"]),
        k=int(raw["k"]),
    )


def _generic_domain_mapping(domain: GenericDomain) -> dict[str, object]:
    """Serialize every field that can alter one independent policy domain."""

    return {
        "backend": domain.backend.value,
        "architecture_class": domain.architecture_class,
        "semantic_contract": domain.semantic_contract.value,
        "operation_kind": domain.operation_kind,
        "bundle_signature": domain.bundle_signature,
        "prepared_family_id": domain.prepared_family_id,
        "packing_abi": domain.packing_abi,
        "runtime_codebook_id": domain.runtime_codebook_id,
        "execution_mode": domain.execution_mode.value,
        "m": domain.m,
        "aspect_bucket": domain.aspect_bucket.value,
        "all_aspects": domain.all_aspects,
    }


def _generic_domain_from_mapping(raw: Mapping[str, object]) -> GenericDomain:
    """Reconstruct one policy domain from a fit-cache record."""

    return GenericDomain(
        backend=Backend(str(raw["backend"])),
        architecture_class=str(raw["architecture_class"]),
        semantic_contract=SemanticContract(str(raw["semantic_contract"])),
        operation_kind=str(raw["operation_kind"]),
        bundle_signature=str(raw["bundle_signature"]),
        prepared_family_id=str(raw["prepared_family_id"]),
        packing_abi=str(raw["packing_abi"]),
        runtime_codebook_id=int(raw["runtime_codebook_id"]),
        execution_mode=ExecutionMode(str(raw["execution_mode"])),
        m=int(raw["m"]),
        aspect_bucket=AspectBucket(str(raw["aspect_bucket"])),
        all_aspects=bool(raw["all_aspects"]),
    )


def _cross_validation_cell_mapping(
    cell: CrossValidationCell,
) -> dict[str, object]:
    """Serialize one held-out decision with exact hexadecimal float identity."""

    return {
        "runtime_key": _runtime_key_mapping(cell.runtime_key),
        "shape_group_id": cell.shape_group_id,
        "selected_candidate_id": cell.selected_candidate_id,
        "exact_candidate_id": cell.exact_candidate_id,
        "observed_broad_regret_hex": cell.observed_broad_regret.hex(),
    }


def _cross_validation_cell_from_mapping(
    raw: Mapping[str, object],
) -> CrossValidationCell:
    """Reconstruct one held-out decision from a fit-cache record."""

    return CrossValidationCell(
        runtime_key=_runtime_key_from_mapping(raw["runtime_key"]),
        shape_group_id=str(raw["shape_group_id"]),
        selected_candidate_id=str(raw["selected_candidate_id"]),
        exact_candidate_id=str(raw["exact_candidate_id"]),
        observed_broad_regret=float.fromhex(
            str(raw["observed_broad_regret_hex"])
        ),
    )


def _cross_validation_mapping(
    validation: DomainCrossValidation,
) -> dict[str, object]:
    """Serialize a complete reusable domain-CV result."""

    return {
        "domain": _generic_domain_mapping(validation.domain),
        "selected_feature_policy": validation.selected_feature_policy.value,
        "selected_max_leaves": validation.selected_max_leaves,
        "selected_boundary_placement": (
            validation.selected_boundary_placement.value
        ),
        "selected_profiler_influence": (
            validation.selected_profiler_influence.value
        ),
        "fold_count": validation.fold_count,
        "shape_group_count": validation.shape_group_count,
        "required_point_count": validation.required_point_count,
        "covered_point_count": validation.covered_point_count,
        "max_regret_hex": validation.max_regret.hex(),
        "p95_regret_hex": validation.p95_regret.hex(),
        "mean_regret_hex": validation.mean_regret.hex(),
        "worst_shape_group_id": validation.worst_shape_group_id,
        "worst_aggregate_n": validation.worst_aggregate_n,
        "worst_k": validation.worst_k,
        "worst_selected_candidate_id": (
            validation.worst_selected_candidate_id
        ),
        "worst_exact_candidate_id": validation.worst_exact_candidate_id,
        "cells": [
            _cross_validation_cell_mapping(cell) for cell in validation.cells
        ],
        "competitive_cells": [
            _cross_validation_cell_mapping(cell)
            for cell in validation.competitive_cells
        ],
        "publication_feature_policy": (
            validation.publication_feature_policy.value
            if validation.publication_feature_policy is not None
            else None
        ),
        "publication_max_leaves": validation.publication_max_leaves,
        "publication_boundary_placement": (
            validation.publication_boundary_placement.value
            if validation.publication_boundary_placement is not None
            else None
        ),
        "publication_oof_p95_regret_hex": (
            validation.publication_oof_p95_regret.hex()
            if validation.publication_oof_p95_regret is not None
            else None
        ),
        "publication_oof_mean_regret_hex": (
            validation.publication_oof_mean_regret.hex()
            if validation.publication_oof_mean_regret is not None
            else None
        ),
        "publication_oof_mismatch_count": (
            validation.publication_oof_mismatch_count
        ),
    }


def _cross_validation_from_mapping(
    raw: Mapping[str, object],
) -> DomainCrossValidation:
    """Reconstruct and type-check a complete cached domain-CV result."""

    return DomainCrossValidation(
        domain=_generic_domain_from_mapping(raw["domain"]),
        selected_feature_policy=FeaturePolicy(
            str(raw["selected_feature_policy"])
        ),
        selected_max_leaves=int(raw["selected_max_leaves"]),
        selected_boundary_placement=BoundaryPlacement(
            str(raw["selected_boundary_placement"])
        ),
        selected_profiler_influence=ProfilerInfluence(str(
            raw.get(
                "selected_profiler_influence",
                ProfilerInfluence.MEASURED_ONLY.value,
            )
        )),
        fold_count=int(raw["fold_count"]),
        shape_group_count=int(raw["shape_group_count"]),
        required_point_count=int(raw["required_point_count"]),
        covered_point_count=int(raw["covered_point_count"]),
        max_regret=float.fromhex(str(raw["max_regret_hex"])),
        p95_regret=float.fromhex(str(raw["p95_regret_hex"])),
        mean_regret=float.fromhex(str(raw["mean_regret_hex"])),
        worst_shape_group_id=str(raw["worst_shape_group_id"]),
        worst_aggregate_n=int(raw["worst_aggregate_n"]),
        worst_k=int(raw["worst_k"]),
        worst_selected_candidate_id=str(
            raw["worst_selected_candidate_id"]
        ),
        worst_exact_candidate_id=str(raw["worst_exact_candidate_id"]),
        cells=tuple(
            _cross_validation_cell_from_mapping(cell)
            for cell in raw["cells"]
        ),
        competitive_cells=tuple(
            _cross_validation_cell_from_mapping(cell)
            for cell in raw.get("competitive_cells", ())
        ),
        publication_feature_policy=(
            FeaturePolicy(str(raw["publication_feature_policy"]))
            if raw.get("publication_feature_policy") is not None
            else None
        ),
        publication_max_leaves=(
            int(raw["publication_max_leaves"])
            if raw.get("publication_max_leaves") is not None
            else None
        ),
        publication_boundary_placement=(
            BoundaryPlacement(str(raw["publication_boundary_placement"]))
            if raw.get("publication_boundary_placement") is not None
            else None
        ),
        publication_oof_p95_regret=(
            float.fromhex(str(raw["publication_oof_p95_regret_hex"]))
            if raw.get("publication_oof_p95_regret_hex") is not None
            else None
        ),
        publication_oof_mean_regret=(
            float.fromhex(str(raw["publication_oof_mean_regret_hex"]))
            if raw.get("publication_oof_mean_regret_hex") is not None
            else None
        ),
        publication_oof_mismatch_count=(
            int(raw["publication_oof_mismatch_count"])
            if raw.get("publication_oof_mismatch_count") is not None
            else None
        ),
    )


def _generic_dispatch_rule_mapping(
    rule: GenericDispatchRule,
) -> dict[str, object]:
    """Serialize one final publication leaf without float precision loss."""

    return {
        "domain": _generic_domain_mapping(rule.domain),
        "predicates": [
            {
                "axis": predicate.threshold.axis.value,
                "numerator": predicate.threshold.numerator,
                "denominator": predicate.threshold.denominator,
                "parallelism_width": predicate.threshold.parallelism_width,
                "task_multiplier": predicate.threshold.task_multiplier,
                "require_less_equal": predicate.require_less_equal,
            }
            for predicate in rule.predicates
        ],
        "candidate_id": rule.candidate_id,
        "arithmetic_fingerprint": rule.arithmetic_fingerprint,
        "development_shape_groups": list(rule.development_shape_groups),
        "development_max_regret_hex": rule.development_max_regret.hex(),
        "development_p95_regret_hex": rule.development_p95_regret.hex(),
        "development_mean_regret_hex": rule.development_mean_regret.hex(),
    }


def _generic_dispatch_rule_from_mapping(
    raw: Mapping[str, object],
) -> GenericDispatchRule:
    """Reconstruct one type-checked final publication leaf from cache."""

    return GenericDispatchRule(
        domain=_generic_domain_from_mapping(raw["domain"]),
        predicates=tuple(
            FeaturePredicate(
                threshold=FeatureThreshold(
                    axis=FeatureAxis(str(predicate["axis"])),
                    numerator=int(predicate["numerator"]),
                    denominator=int(predicate["denominator"]),
                    parallelism_width=int(predicate["parallelism_width"]),
                    task_multiplier=int(predicate["task_multiplier"]),
                ),
                require_less_equal=bool(predicate["require_less_equal"]),
            )
            for predicate in raw["predicates"]
        ),
        candidate_id=str(raw["candidate_id"]),
        arithmetic_fingerprint=str(raw["arithmetic_fingerprint"]),
        development_shape_groups=tuple(
            str(item) for item in raw["development_shape_groups"]
        ),
        development_max_regret=float.fromhex(
            str(raw["development_max_regret_hex"])
        ),
        development_p95_regret=float.fromhex(
            str(raw["development_p95_regret_hex"])
        ),
        development_mean_regret=float.fromhex(
            str(raw["development_mean_regret_hex"])
        ),
    )


def _content_key(payload: Mapping[str, object]) -> str:
    """Return a deterministic SHA-256 key for one immutable cache input."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generic_domain_corpus_digest(
    corpus: ObservationCorpus,
    domain: GenericDomain,
) -> str:
    """Hash semantic observations owned by one policy domain.

    Targeted refinement appends rows to a small subset of generic domains. A
    whole-corpus cache identity would therefore invalidate unrelated regret
    matrices and CV fits. ``corpus_id`` is deliberately excluded because the
    adapter derives it from the complete aggregate and writes that same global
    identity into every row. All measured values, eligibility evidence, build
    provenance, and runtime geometry remain in the digest, so only the
    aggregate-membership label is ignored.
    """

    rows = corpus.rows_for_generic_domain(domain)
    if not rows:
        raise ValueError("cannot hash an empty generic policy domain")
    serialized_rows = []
    for row in rows:
        mapping = row.canonical_mapping()
        mapping.pop("corpus_id")
        serialized_rows.append(json.dumps(
            mapping, sort_keys=True, separators=(",", ":")
        ))
    serialized = "[" + ",".join(sorted(serialized_rows)) + "]"
    if not corpus.distinguishes_execution_mode:
        serialized = "mode-collapsed:" + serialized
    if not corpus.distinguishes_aspect_bucket:
        serialized = "aspect-collapsed:" + serialized
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _legacy_generic_domain_corpus_digest(
    corpus: ObservationCorpus,
    domain: GenericDomain,
) -> str:
    """Reproduce the v15 aggregate-coupled domain digest for cache migration."""

    rows = corpus.rows_for_generic_domain(domain)
    if not rows:
        raise ValueError("cannot hash an empty generic policy domain")
    return ObservationCorpus._from_validated(
        rows,
        distinguish_execution_mode=corpus.distinguishes_execution_mode,
        distinguish_aspect_bucket=corpus.distinguishes_aspect_bucket,
    ).digest()


def _serial_m1_hash_digest(
    serial_m1_hashes: Mapping[RuntimeKey, str] | None,
    runtime_keys: Iterable[RuntimeKey] | None = None,
) -> str:
    """Hash the serial-oracle identities that can affect one cost matrix.

    A domain cost is filtered against the current serial-M1 policy hash at
    each runtime point.  Targeted refinement may add runtime keys to another
    domain, so including the process-wide hash inventory here would invalidate
    every otherwise independent domain cache entry.  ``runtime_keys=None``
    intentionally reproduces the legacy global identity so existing v6 cache
    records can be authenticated and promoted to their domain-local keys.
    """

    hashes = serial_m1_hashes or {}
    selected_keys = (
        tuple(sorted(hashes))
        if runtime_keys is None
        else tuple(sorted(set(runtime_keys)))
    )
    return "sha256:" + _content_key({
        "serial_m1_hashes": [
            {
                "runtime_key": _runtime_key_mapping(key),
                "policy_hash": hashes.get(key),
            }
            for key in selected_keys
        ],
    })


@dataclass(frozen=True)
class PolicyFitCache:
    """Content-addressed costs, CV, and final trees for incremental fitting.

    Broad observations never change during a paired tournament. Only the small
    set of domains touched by newly measured edges should be recomputed. The
    cache therefore binds every entry to one domain's immutable observation
    digest and paired-evidence digest. A production entry also retains the
    stable all-development publication leaves, avoiding a second GPU tree fit
    for every unchanged domain. Missing entries are ordinary first-use misses;
    malformed or contradictory entries fail loudly instead of silently
    changing the learned policy.
    """

    directory: Path

    def _path(self, kind: str, content_key: str) -> Path:
        """Return the immutable file location for one content-addressed entry."""

        return self.directory / kind / f"{content_key}.json"

    def _read(self, kind: str, content_key: str) -> Mapping[str, object] | None:
        """Read and validate one immutable cache payload if it exists."""

        path = self._path(kind, content_key)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        if raw.get("schema_version") != POLICY_FIT_CACHE_SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported policy fit cache schema")
        if raw.get("kind") != kind or raw.get("content_key") != content_key:
            raise ValueError(f"{path}: policy fit cache identity mismatch")
        return raw

    def _write(
        self,
        kind: str,
        content_key: str,
        payload: Mapping[str, object],
    ) -> None:
        """Atomically publish one deterministic cache payload."""

        path = self._path(kind, content_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        complete = {
            "schema_version": POLICY_FIT_CACHE_SCHEMA_VERSION,
            "kind": kind,
            "content_key": content_key,
            **payload,
        }
        encoded = (
            json.dumps(complete, sort_keys=True, separators=(",", ":")) + "\n"
        )
        if path.exists():
            if path.read_text(encoding="utf-8") != encoded:
                raise ValueError(f"{path}: content-addressed cache collision")
            return
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)

    def cost_key(
        self,
        domain: GenericDomain,
        domain_corpus_digest: str,
        paired_domain_digest: str,
        serial_m1_hash_digest: str,
    ) -> str:
        """Hash every input that can change one domain's regret matrix."""

        if not domain_corpus_digest.startswith("sha256:"):
            raise ValueError("domain corpus digest must be a sha256 identity")

        return _content_key({
            "schema_version": POLICY_FIT_CACHE_SCHEMA_VERSION,
            "kind": "candidate-costs",
            "domain": _generic_domain_mapping(domain),
            "domain_corpus_digest": domain_corpus_digest,
            "paired_domain_digest": paired_domain_digest,
            "serial_m1_hash_digest": serial_m1_hash_digest,
            "cost_statistic": POLICY_COST_STATISTIC_VERSION,
        })

    def load_costs(
        self,
        content_key: str,
        domain: GenericDomain,
    ) -> list[CandidatePointCost] | None:
        """Load one exact regret matrix and reject cross-domain contamination."""

        raw = self._read("candidate-costs", content_key)
        if raw is None:
            return None
        if _generic_domain_from_mapping(raw["domain"]) != domain:
            raise ValueError("candidate-cost cache changed policy domain")
        candidate_ids = tuple(str(item) for item in raw["candidate_ids"])
        points_raw = raw["points"]
        if (
            any(not item for item in candidate_ids)
            or len(set(candidate_ids)) != len(candidate_ids)
            or (not candidate_ids and points_raw)
        ):
            raise ValueError("candidate-cost cache has an invalid ID inventory")
        costs = []
        for point in points_raw:
            runtime_key = _runtime_key_from_mapping(point["runtime_key"])
            shape_group_id = str(point["shape_group_id"])
            point_candidates = set()
            for values in point["candidate_costs"]:
                if len(values) != 4:
                    raise ValueError(
                        "candidate-cost cache tuple must contain four fields"
                    )
                candidate_index = int(values[0])
                if not 0 <= candidate_index < len(candidate_ids):
                    raise ValueError(
                        "candidate-cost cache candidate index is out of range"
                    )
                if candidate_index in point_candidates:
                    raise ValueError(
                        "candidate-cost cache repeats a candidate at one point"
                    )
                point_candidates.add(candidate_index)
                costs.append(CandidatePointCost(
                    runtime_key=runtime_key,
                    shape_group_id=shape_group_id,
                    candidate_id=candidate_ids[candidate_index],
                    max_surface_regret=float.fromhex(str(values[1])),
                    p95_surface_regret=float.fromhex(str(values[2])),
                    mean_surface_regret=float.fromhex(str(values[3])),
                ))
        if any(cost.runtime_key.backend != domain.backend for cost in costs):
            raise ValueError("candidate-cost cache contains another backend")
        return costs

    def store_costs(
        self,
        content_key: str,
        domain: GenericDomain,
        costs: Iterable[CandidatePointCost],
    ) -> None:
        """Publish one domain matrix with interned point and candidate identity.

        A dense NativeVNNI domain can contain hundreds of thousands of candidate
        costs. Repeating a full runtime key and long candidate ID on every row
        turns a useful cache into a multi-gigabyte artifact. The compact layout
        retains source order exactly while spelling each point and candidate ID
        once; hexadecimal floats preserve policy bytes across reloads.
        """

        costs = tuple(costs)
        candidate_ids = tuple(sorted({cost.candidate_id for cost in costs}))
        candidate_index = {
            candidate_id: index
            for index, candidate_id in enumerate(candidate_ids)
        }
        points: dict[
            tuple[RuntimeKey, str], list[list[object]]
        ] = {}
        point_candidate_indices: dict[tuple[RuntimeKey, str], set[int]] = {}
        closed_points = set()
        previous_point = None
        for cost in costs:
            point = (cost.runtime_key, cost.shape_group_id)
            if point != previous_point:
                if point in closed_points:
                    raise ValueError(
                        "candidate costs for one point must remain contiguous"
                    )
                if previous_point is not None:
                    closed_points.add(previous_point)
                previous_point = point
            index = candidate_index[cost.candidate_id]
            indices = point_candidate_indices.setdefault(point, set())
            if index in indices:
                raise ValueError(
                    "candidate costs repeat a candidate at one runtime point"
                )
            indices.add(index)
            points.setdefault(point, []).append([
                index,
                cost.max_surface_regret.hex(),
                cost.p95_surface_regret.hex(),
                cost.mean_surface_regret.hex(),
            ])
        self._write("candidate-costs", content_key, {
            "domain": _generic_domain_mapping(domain),
            "candidate_ids": list(candidate_ids),
            "points": [
                {
                    "runtime_key": _runtime_key_mapping(runtime_key),
                    "shape_group_id": shape_group_id,
                    "candidate_costs": candidate_costs,
                }
                for (
                    runtime_key,
                    shape_group_id,
                ), candidate_costs in points.items()
            ],
        })

    def profiler_prediction_key(
        self,
        *,
        training_pool_digest: str,
        profiler_model_digest: str,
        held_out_geometries: Iterable[tuple[int, int]],
        prediction_points: Iterable[ProfilerPredictionPoint],
    ) -> str:
        """Hash one leakage-controlled profiler prediction surface.

        A surface is reusable only while its complete timing-label pool,
        model-visible profiler descriptors, held geometry boundary, and output
        point inventory are unchanged.  Naming the requested points in the key
        keeps the payload compact: CV does not need predictions for every row
        in a large cross-M transfer pool, and a later caller cannot mistake a
        partial prediction map for a complete one.
        """

        for label, digest in (
            ("training pool", training_pool_digest),
            ("profiler model", profiler_model_digest),
        ):
            if not digest.startswith("sha256:"):
                raise ValueError(
                    f"profiler prediction {label} digest must be a sha256 identity"
                )
        geometries = tuple(sorted(set(held_out_geometries)))
        points = tuple(sorted(set(prediction_points)))
        return _content_key({
            "schema_version": POLICY_FIT_CACHE_SCHEMA_VERSION,
            "kind": "profiler-prediction-surface",
            "training_pool_digest": training_pool_digest,
            "profiler_model_digest": profiler_model_digest,
            "held_out_geometries": [list(item) for item in geometries],
            "prediction_points": [
                {
                    "runtime_key": _runtime_key_mapping(runtime_key),
                    "shape_group_id": shape_group_id,
                    "candidate_id": candidate_id,
                }
                for runtime_key, shape_group_id, candidate_id in points
            ],
        })

    def load_profiler_predictions(
        self,
        content_key: str,
        *,
        training_pool_digest: str,
        profiler_model_digest: str,
        held_out_geometries: Iterable[tuple[int, int]],
        prediction_points: Iterable[ProfilerPredictionPoint],
    ) -> dict[ProfilerPredictionPoint, float | None] | None:
        """Load one exact profiler surface and reject partial point maps."""

        raw = self._read("profiler-prediction-surface", content_key)
        if raw is None:
            return None
        expected_geometries = tuple(sorted(set(held_out_geometries)))
        expected_points = frozenset(prediction_points)
        if raw.get("training_pool_digest") != training_pool_digest:
            raise ValueError("profiler prediction cache changed training pool")
        if raw.get("profiler_model_digest") != profiler_model_digest:
            raise ValueError("profiler prediction cache changed model descriptors")
        observed_geometries = tuple(
            (int(item[0]), int(item[1]))
            for item in raw.get("held_out_geometries", [])
        )
        if observed_geometries != expected_geometries:
            raise ValueError("profiler prediction cache changed held geometries")

        predictions = {}
        for item in raw.get("predictions", []):
            point = (
                _runtime_key_from_mapping(item["runtime_key"]),
                str(item["shape_group_id"]),
                str(item["candidate_id"]),
            )
            if point in predictions:
                raise ValueError("profiler prediction cache repeats one point")
            encoded_prediction = item["prediction"]
            predictions[point] = (
                None
                if encoded_prediction is None
                else float.fromhex(str(encoded_prediction))
            )
        if frozenset(predictions) != expected_points:
            missing = len(expected_points - predictions.keys())
            foreign = len(predictions.keys() - expected_points)
            raise ValueError(
                "profiler prediction cache changed requested point inventory: "
                f"missing={missing}, foreign={foreign}"
            )
        return predictions

    def store_profiler_predictions(
        self,
        content_key: str,
        *,
        training_pool_digest: str,
        profiler_model_digest: str,
        held_out_geometries: Iterable[tuple[int, int]],
        predictions: Mapping[ProfilerPredictionPoint, float | None],
    ) -> None:
        """Publish the compact predictions consumed by one CV surface."""

        self._write("profiler-prediction-surface", content_key, {
            "training_pool_digest": training_pool_digest,
            "profiler_model_digest": profiler_model_digest,
            "held_out_geometries": [
                list(item) for item in sorted(set(held_out_geometries))
            ],
            "predictions": [
                {
                    "runtime_key": _runtime_key_mapping(runtime_key),
                    "shape_group_id": shape_group_id,
                    "candidate_id": candidate_id,
                    "prediction": (
                        None
                        if prediction is None
                        else float(prediction).hex()
                    ),
                }
                for (
                    runtime_key,
                    shape_group_id,
                    candidate_id,
                ), prediction in sorted(predictions.items())
            ],
        })

    def validation_key(
        self,
        domain: GenericDomain,
        cost_key: str,
        *,
        max_leaves: int,
        min_shape_groups_per_leaf: int,
        cross_validation_seed: str,
        profiler_feature_catalog_digest: str | None,
        fit_final_rules: bool,
        profiler_training_pool_digest: str | None = None,
    ) -> str:
        """Hash the exact model search applied to one cached regret matrix.

        Profiler-informed predictions are trained across every format in one
        cross-M prefill transfer pool.  The pool digest therefore participates in new
        CV keys even though candidate costs themselves remain domain-local.
        Omitting it intentionally reproduces the legacy v6 key for migration.
        """

        payload = {
            "schema_version": POLICY_FIT_CACHE_SCHEMA_VERSION,
            "kind": "domain-cross-validation",
            "domain": _generic_domain_mapping(domain),
            "cost_key": cost_key,
            "max_leaves": max_leaves,
            "min_shape_groups_per_leaf": min_shape_groups_per_leaf,
            "cross_validation_seed": cross_validation_seed,
            "cross_validation_fold_schema": (
                CROSS_VALIDATION_FOLD_SCHEMA_VERSION
            ),
            "profiler_feature_catalog_digest": profiler_feature_catalog_digest,
            "selection_mode": (
                "production-final-frontier"
                if fit_final_rules
                else "paired-request-frontier"
            ),
            "tree_beam_width": TREE_BEAM_WIDTH,
            "primary_objective": POLICY_PRIMARY_OBJECTIVE_VERSION,
            "paired_model_frontier": "complete",
            "feature_policies": [item.value for item in FEATURE_POLICIES],
            "boundary_placements": [
                item.value for item in BOUNDARY_PLACEMENTS
            ],
            "profiler_influences": [
                item.value for item in PROFILER_INFLUENCES
            ],
        }
        if profiler_training_pool_digest is not None:
            payload["profiler_training_pool_digest"] = (
                profiler_training_pool_digest
            )
        return _content_key(payload)

    def load_validation(
        self,
        content_key: str,
        domain: GenericDomain,
    ) -> tuple[bool, DomainCrossValidation | None]:
        """Return a hit flag because ``None`` is itself a cached CV result."""

        hit, validation, _rules = self.load_validation_entry(
            content_key,
            domain,
        )
        return hit, validation

    def load_validation_entry(
        self,
        content_key: str,
        domain: GenericDomain,
    ) -> tuple[
        bool,
        DomainCrossValidation | None,
        tuple[GenericDispatchRule, ...] | None,
    ]:
        """Load CV plus an optional content-identical final publication tree."""

        raw = self._read("domain-cross-validation", content_key)
        if raw is None:
            return False, None, None
        if _generic_domain_from_mapping(raw["domain"]) != domain:
            raise ValueError("cross-validation cache changed policy domain")
        validation_raw = raw["validation"]
        validation = (
            _cross_validation_from_mapping(validation_raw)
            if validation_raw is not None
            else None
        )
        if validation is not None and validation.domain != domain:
            raise ValueError("cross-validation cache changed policy domain")
        rules_raw = raw.get("final_rules")
        rules = (
            tuple(
                _generic_dispatch_rule_from_mapping(rule)
                for rule in rules_raw
            )
            if rules_raw is not None
            else None
        )
        if rules is not None:
            if any(rule.domain != domain for rule in rules):
                raise ValueError(
                    "cross-validation cache changed final-rule domain"
                )
            if rules:
                validate_generic_rule_partition(rules)
        return True, validation, rules

    def store_validation(
        self,
        content_key: str,
        domain: GenericDomain,
        validation: DomainCrossValidation | None,
        final_rules: Iterable[GenericDispatchRule] | None = None,
    ) -> None:
        """Publish complete CV and optional stable final publication leaves."""

        rules = tuple(final_rules) if final_rules is not None else None
        if rules is not None:
            if any(rule.domain != domain for rule in rules):
                raise ValueError("cannot cache final rules for another domain")
            if rules:
                validate_generic_rule_partition(rules)

        self._write("domain-cross-validation", content_key, {
            "domain": _generic_domain_mapping(domain),
            "validation": (
                _cross_validation_mapping(validation)
                if validation is not None
                else None
            ),
            "final_rules": (
                [
                    _generic_dispatch_rule_mapping(rule)
                    for rule in rules
                ]
                if rules is not None
                else None
            ),
        })


def load_cached_cross_validations(
    directory: Path,
) -> tuple[DomainCrossValidation, ...]:
    """Load one complete read-only CV generation for request bootstrapping.

    The ordinary fitter addresses entries by content key because it knows every
    model, corpus, profiler, and paired-evidence input. A refinement bootstrap
    starts from the opposite direction: the expensive fit has already finished
    and its 108 held-out decisions are the artifact that identifies which
    candidate edges need contemporary timing. This reader therefore enumerates
    the immutable cache directory while retaining the same strict schema,
    filename/content-key, root-domain, and typed-validation checks as
    :class:`PolicyFitCache`.

    Callers must additionally compare the returned domain inventory with their
    current development corpus. This function deliberately cannot bless a
    partial cache, choose among duplicate generations, or reinterpret a cached
    ``None`` result as a validation.
    """

    root = Path(directory)
    validation_directory = (
        root
        if root.name == "domain-cross-validation"
        else root / "domain-cross-validation"
    )
    if not validation_directory.is_dir():
        raise ValueError(
            f"{validation_directory}: cached CV directory does not exist"
        )

    validations: dict[GenericDomain, DomainCrossValidation] = {}
    paths = tuple(sorted(validation_directory.glob("*.json")))
    if not paths:
        raise ValueError(f"{validation_directory}: cached CV directory is empty")
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        content_key = path.stem
        if raw.get("schema_version") != POLICY_FIT_CACHE_SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported policy fit cache schema")
        if (
            raw.get("kind") != "domain-cross-validation"
            or raw.get("content_key") != content_key
        ):
            raise ValueError(f"{path}: policy fit cache identity mismatch")
        domain = _generic_domain_from_mapping(raw["domain"])
        validation_raw = raw.get("validation")
        if validation_raw is None:
            raise ValueError(f"{path}: cached CV result is empty")
        validation = _cross_validation_from_mapping(validation_raw)
        if validation.domain != domain:
            raise ValueError(f"{path}: cached CV changed policy domain")
        if domain in validations:
            raise ValueError(
                f"{validation_directory}: duplicate cached CV domain {domain!r}"
            )
        validations[domain] = validation
    return tuple(validations[domain] for domain in sorted(validations))


def _solve_dense_system(
    matrix: list[list[float]],
    right_hand_side: list[float],
) -> list[float]:
    """Solve one small positive-definite normal system deterministically."""

    size = len(right_hand_side)
    augmented = [
        [*matrix[row], right_hand_side[row]] for row in range(size)
    ]
    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot][column]) <= 1.0e-12:
            raise ValueError("paired timing graph is underdetermined")
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row],
                    augmented[column],
                    strict=True,
                )
            ]
    return [augmented[row][-1] for row in range(size)]


def _paired_effective_latencies(
    broad_latencies: Mapping[str, float],
    comparisons: Iterable[PairedTimingComparison],
) -> dict[str, float]:
    """Fit a weighted log-latency tournament and preserve one broad anchor.

    An edge records ``log(selected / exact)``. Multiple provisional choices
    can form a tree or cycle within one runtime cell. Solving all edges jointly
    prevents the active correction from oscillating when the learner changes
    candidates, while anchoring the broad-fastest node in each connected
    component keeps separate confirmation sessions off the absolute clock axis.
    """

    edges = tuple(comparisons)
    adjacency: dict[str, set[str]] = defaultdict(set)
    for comparison in edges:
        selected = comparison.selected_effective_candidate_id
        exact = comparison.exact_effective_candidate_id
        if selected not in broad_latencies or exact not in broad_latencies:
            raise ValueError(
                "paired timing candidate is absent from broad evidence"
            )
        adjacency[selected].add(exact)
        adjacency[exact].add(selected)

    corrected = {}
    remaining = set(adjacency)
    while remaining:
        seed = min(remaining)
        stack = [seed]
        component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(sorted(adjacency[node] - component))
        remaining.difference_update(component)

        anchor = min(
            component,
            key=lambda node: (broad_latencies[node], node),
        )
        anchor_log = math.log(broad_latencies[anchor])
        unknown = tuple(sorted(component - {anchor}))
        index = {node: position for position, node in enumerate(unknown)}
        normal = [[0.0 for _ in unknown] for _ in unknown]
        rhs = [0.0 for _ in unknown]
        for comparison in edges:
            selected = comparison.selected_effective_candidate_id
            exact = comparison.exact_effective_candidate_id
            if selected not in component or exact not in component:
                continue
            target = math.log(comparison.selected_to_exact_median_ratio)
            coefficients = {}
            if selected == anchor:
                target -= anchor_log
            else:
                coefficients[index[selected]] = 1.0
            if exact == anchor:
                target += anchor_log
            else:
                coefficients[index[exact]] = -1.0
            weight = float(comparison.pair_count)
            for row, row_coefficient in coefficients.items():
                rhs[row] += weight * row_coefficient * target
                for column, column_coefficient in coefficients.items():
                    normal[row][column] += (
                        weight * row_coefficient * column_coefficient
                    )
        solved = _solve_dense_system(normal, rhs) if unknown else []
        corrected[anchor] = broad_latencies[anchor]
        corrected.update({
            node: math.exp(solved[index[node]]) for node in unknown
        })
    return corrected


def _paired_cell_key_for_observation(
    row: NativeVNNIObservation,
) -> PairedCellKey:
    """Project one observation onto the complete paired-evidence identity."""

    return PairedCellKey(
        backend=row.backend.value,
        source_format=row.source_format,
        source_codebook=row.source_codebook_id,
        execution_codebook=row.runtime_codebook_id,
        shape=row.shape_name,
        execution_mode=row.execution_mode.value,
        m=row.m,
        n=row.aggregate_n,
        k=row.k,
        architecture_class=(
            row.architecture_class if row.backend == Backend.CPU else ""
        ),
    )


def _paired_domain_digest(
    rows: Iterable[NativeVNNIObservation],
    comparisons: Mapping[
        PairedCellKey, tuple[PairedTimingComparison, ...]
    ] | None,
) -> str:
    """Hash only tournament edges capable of changing this domain's costs."""

    relevant_keys = {
        _paired_cell_key_for_observation(row) for row in rows
    }
    payload = []
    for key in sorted(relevant_keys):
        for comparison in sorted(
            (comparisons or {}).get(key, ()),
            key=lambda item: (
                item.selected_effective_candidate_id,
                item.exact_effective_candidate_id,
                item.selected_to_exact_median_ratio,
                item.pair_count,
            ),
        ):
            payload.append({
                "key": {
                    "backend": key.backend,
                    "architecture_class": key.architecture_class,
                    "source_format": key.source_format,
                    "source_codebook": key.source_codebook,
                    "execution_codebook": key.execution_codebook,
                    "shape": key.shape,
                    "execution_mode": key.execution_mode,
                    "m": key.m,
                    "n": key.n,
                    "k": key.k,
                },
                "selected_effective_candidate_id": (
                    comparison.selected_effective_candidate_id
                ),
                "exact_effective_candidate_id": (
                    comparison.exact_effective_candidate_id
                ),
                "selected_to_exact_median_ratio_hex": (
                    comparison.selected_to_exact_median_ratio.hex()
                ),
                "pair_count": comparison.pair_count,
            })
    return "sha256:" + _content_key({"comparisons": payload})


_PARALLEL_CACHE_IDENTITY_CORPUS: ObservationCorpus | None = None
_PARALLEL_CACHE_IDENTITY_DOMAINS: tuple[GenericDomain, ...] = ()
_PARALLEL_CACHE_IDENTITY_COMPARISONS: Mapping[
    PairedCellKey, tuple[PairedTimingComparison, ...]
] | None = None
_PARALLEL_CACHE_IDENTITY_SERIAL_HASHES: Mapping[RuntimeKey, str] | None = None


def _domain_cache_identity_at(index: int) -> tuple[
    GenericDomain,
    str,
    str,
    str,
    str,
]:
    """Build every cache-key input owned by one independent policy domain."""

    if _PARALLEL_CACHE_IDENTITY_CORPUS is None:
        raise RuntimeError("parallel domain cache identity corpus is unavailable")
    corpus = _PARALLEL_CACHE_IDENTITY_CORPUS
    domain = _PARALLEL_CACHE_IDENTITY_DOMAINS[index]
    rows = corpus.rows_for_generic_domain(domain)
    runtime_keys = {corpus.runtime_key_for(row) for row in rows}
    return (
        domain,
        _paired_domain_digest(
            rows,
            _PARALLEL_CACHE_IDENTITY_COMPARISONS,
        ),
        generic_domain_corpus_digest(corpus, domain),
        _legacy_generic_domain_corpus_digest(corpus, domain),
        _serial_m1_hash_digest(
            _PARALLEL_CACHE_IDENTITY_SERIAL_HASHES,
            runtime_keys,
        ),
    )


def _domain_cache_identity_inputs(
    corpus: ObservationCorpus,
    domains: tuple[GenericDomain, ...],
    paired_comparisons: Mapping[
        PairedCellKey, tuple[PairedTimingComparison, ...]
    ] | None,
    serial_m1_hashes: Mapping[RuntimeKey, str] | None,
) -> tuple[tuple[GenericDomain, str, str, str, str], ...]:
    """Compute deterministic domain-local cache identities across host cores.

    Domain corpus hashes, paired-edge hashes, legacy migration hashes, and
    serial-oracle inventories share no mutable state. A trainer-sized corpus
    therefore forks one task per domain and returns results in canonical domain
    order. Small corpora remain in-process to avoid process startup overhead.
    """

    global _PARALLEL_CACHE_IDENTITY_CORPUS
    global _PARALLEL_CACHE_IDENTITY_DOMAINS
    global _PARALLEL_CACHE_IDENTITY_COMPARISONS
    global _PARALLEL_CACHE_IDENTITY_SERIAL_HASHES

    requested_workers = int(os.environ.get(
        "LLAMINAR_NATIVE_VNNI_POLICY_WORKERS",
        str(_physical_core_worker_count()),
    ))
    if requested_workers < 1:
        raise ValueError("policy worker count must be positive")
    worker_count = min(
        requested_workers,
        len(domains),
        max(1, len(corpus) // 4096),
    )
    _PARALLEL_CACHE_IDENTITY_CORPUS = corpus
    _PARALLEL_CACHE_IDENTITY_DOMAINS = domains
    _PARALLEL_CACHE_IDENTITY_COMPARISONS = paired_comparisons
    _PARALLEL_CACHE_IDENTITY_SERIAL_HASHES = serial_m1_hashes
    try:
        if worker_count > 1 and len(domains) >= 4:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                return tuple(executor.map(
                    _domain_cache_identity_at,
                    range(len(domains)),
                ))
        return tuple(
            _domain_cache_identity_at(index) for index in range(len(domains))
        )
    finally:
        _PARALLEL_CACHE_IDENTITY_CORPUS = None
        _PARALLEL_CACHE_IDENTITY_DOMAINS = ()
        _PARALLEL_CACHE_IDENTITY_COMPARISONS = None
        _PARALLEL_CACHE_IDENTITY_SERIAL_HASHES = None


def _surface_candidate_timings(
    rows: Iterable[NativeVNNIObservation],
    current_serial_m1_hash: str | None,
    paired_comparisons: Mapping[
        PairedCellKey, tuple[PairedTimingComparison, ...]
    ] | None = None,
) -> tuple[dict[tuple[str, SurfaceKey], float], frozenset[SurfaceKey]]:
    """Aggregate broad latency and apply validated paired ratio corrections."""

    grouped: dict[tuple[str, SurfaceKey], list[float]] = defaultdict(list)
    effective_ids: dict[tuple[str, SurfaceKey], set[str]] = defaultdict(set)
    all_surfaces: set[SurfaceKey] = set()
    identity_rows: dict[SurfaceKey, NativeVNNIObservation] = {}
    for row in rows:
        surface = SurfaceKey(row.source_format, row.execution_mode)
        all_surfaces.add(surface)
        identity_rows.setdefault(surface, row)
        if not row.generic_eligible:
            continue
        if not candidate_is_eligible(row, current_serial_m1_hash):
            continue
        grouped[(row.candidate_id, surface)].append(row.median_us)
        effective_ids[(row.candidate_id, surface)].add(
            row.effective_candidate_id
        )

    timings = {
        key: statistics.median(values) for key, values in grouped.items()
    }
    broad_timings = dict(timings)
    for surface, exemplar in identity_rows.items():
        pair_key = _paired_cell_key_for_observation(exemplar)
        comparisons = (paired_comparisons or {}).get(pair_key, ())
        if not comparisons:
            continue
        effective_broad: dict[str, list[float]] = defaultdict(list)
        for (candidate, candidate_surface), candidate_timings in (
            effective_ids.items()
        ):
            if candidate_surface != surface:
                continue
            for effective_id in candidate_timings:
                effective_broad[effective_id].append(
                    broad_timings[(candidate, candidate_surface)]
                )
        corrected_effective = _paired_effective_latencies(
            {
                effective_id: statistics.median(values)
                for effective_id, values in effective_broad.items()
            },
            comparisons,
        )
        for (candidate, candidate_surface), candidate_effective_ids in (
            effective_ids.items()
        ):
            if candidate_surface != surface:
                continue
            corrected_values = [
                corrected_effective[effective_id]
                for effective_id in candidate_effective_ids
                if effective_id in corrected_effective
            ]
            if corrected_values:
                timings[(candidate, surface)] = statistics.median(
                    corrected_values
                )
    return timings, frozenset(all_surfaces)


def _percentile(values: Iterable[float], quantile: float) -> float:
    """Return the deterministic conservative nearest-rank percentile.

    Certification already uses nearest-rank p95.  Development fitting uses the
    same statistic so the learner, generated policy, and sealed gate reason
    about one population contract.  Returning an observed FP64 value also lets
    CUDA and ROCm compare percentile keys byte-exactly without backend-specific
    interpolation rounding.
    """

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile quantile must be in [0, 1]")
    ordered = sorted(values)
    if not ordered:
        return math.inf
    if quantile == 0.95:
        # P95 owns the installation gate, so spell its rank as the exact
        # rational contract shared by Python, CUDA, and ROCm.  Multiplying by
        # binary floating-point 0.95 can straddle an integer rank for some
        # population sizes and silently select a different observed sample.
        rank = (95 * len(ordered) + 99) // 100
    else:
        rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _ordered_mean(values: Iterable[float]) -> float:
    """Return one backend-reproducible serial-order FP64 mean.

    Tree search compares means only after exact p95 keys. A fixed left-to-right
    addition order is deliberately used instead of a host-library compensated
    sum so CUDA, ROCm, and the Python oracle can reproduce every tie-break bit.
    """

    total = 0.0
    count = 0
    for value in values:
        total += value
        count += 1
    return total / count if count else math.inf


def build_domain_candidate_point_costs(
    corpus: ObservationCorpus,
    domain: GenericDomain,
    *,
    serial_m1_hashes: Mapping[RuntimeKey, str] | None = None,
    paired_comparisons: Mapping[
        PairedCellKey, tuple[PairedTimingComparison, ...]
    ] | None = None,
) -> list[CandidatePointCost]:
    """Build one independent domain's eligible candidate regret matrix.

    Rows are grouped by runtime key and shape group. A nominal candidate must
    cover every source alias represented at that point. Shape-resolved
    candidates may have different concrete effective IDs at different runtime
    keys, but retain one stable policy-selectable candidate ID. Keeping this
    operation domain-local is what allows a new paired edge to invalidate only
    the policy surface it can actually affect.
    """

    point_rows: dict[
        tuple[RuntimeKey, str], list[NativeVNNIObservation]
    ] = defaultdict(list)
    for row in corpus.rows_for_generic_domain(domain):
        point_rows[(corpus.runtime_key_for(row), row.shape_group_id)].append(row)

    result = []
    for (key, shape_group_id), rows in point_rows.items():
        current_hash = (serial_m1_hashes or {}).get(key)
        timings, surfaces = _surface_candidate_timings(
            rows,
            current_hash,
            paired_comparisons,
        )
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
        if corpus.generic_domain_for(rows[0]) != domain:
            raise ValueError("domain-local cost construction changed domain")
        for candidate in complete:
            regrets = [
                timings[(candidate, surface)] / best[surface] - 1.0
                for surface in surfaces
            ]
            result.append(CandidatePointCost(
                runtime_key=key,
                shape_group_id=shape_group_id,
                candidate_id=candidate,
                max_surface_regret=max(regrets),
                p95_surface_regret=_percentile(regrets, 0.95),
                mean_surface_regret=statistics.fmean(regrets),
            ))
    return result


def build_candidate_point_costs(
    corpus: ObservationCorpus,
    *,
    serial_m1_hashes: Mapping[RuntimeKey, str] | None = None,
    paired_comparisons: Mapping[
        PairedCellKey, tuple[PairedTimingComparison, ...]
    ] | None = None,
) -> dict[GenericDomain, list[CandidatePointCost]]:
    """Build every domain matrix through the incremental domain primitive."""

    return {
        domain: build_domain_candidate_point_costs(
            corpus,
            domain,
            serial_m1_hashes=serial_m1_hashes,
            paired_comparisons=paired_comparisons,
        )
        for domain in corpus.generic_domains()
    }


def _point_matrix(costs: list[CandidatePointCost]):
    """Index candidate costs by indivisible runtime/shape point."""

    matrix: dict[
        tuple[RuntimeKey, str], dict[str, CandidatePointCost]
    ] = defaultdict(dict)
    for cost in costs:
        matrix[(cost.runtime_key, cost.shape_group_id)][cost.candidate_id] = cost
    return matrix


def _architecture_parallelism_width(architecture_class: str) -> int:
    """Return authenticated CPU worker width, or one for non-CPU domains.

    CPU policy architecture classes end in ``|threads=N`` and generated rules
    independently guard that same architecture/thread surface. Embedding this
    width in a wave predicate is therefore no less portable than embedding a
    CUDA tile width in a tile-count predicate. GPU domains currently return one,
    making the CPU-specific axes constant and unable to create a split.
    """

    match = re.search(r"(?:^|\|)threads=([1-9][0-9]*)(?:\||$)", architecture_class)
    return int(match.group(1)) if match is not None else 1


def _feature_value(
    axis: FeatureAxis,
    point: tuple[RuntimeKey, str],
    parallelism_width: int,
) -> Fraction:
    """Return an exact feature value used only to enumerate legal cuts."""

    key = point[0]
    if axis == FeatureAxis.AGGREGATE_N:
        return Fraction(key.aggregate_n, 1)
    if axis == FeatureAxis.K:
        return Fraction(key.k, 1)
    if axis == FeatureAxis.WORK_ITEMS:
        return Fraction(key.aggregate_n * key.k, 1)
    if axis == FeatureAxis.ASPECT_RATIO:
        return Fraction(key.aggregate_n, key.k)
    if axis in N_TILE_WIDTH_BY_AXIS:
        return Fraction(
            _n_tile_count(key.aggregate_n, N_TILE_WIDTH_BY_AXIS[axis]),
            1,
        )
    if axis in K_GROUPS_PER_N_TILE_WIDTH_BY_AXIS:
        tiles = _n_tile_count(
            key.aggregate_n,
            K_GROUPS_PER_N_TILE_WIDTH_BY_AXIS[axis],
        )
        return Fraction(key.k, 32 * tiles)
    if axis in N_FINAL_TILE_WIDTH_BY_AXIS:
        width = N_FINAL_TILE_WIDTH_BY_AXIS[axis]
        return Fraction((key.aggregate_n - 1) % width + 1, 1)
    if axis in K_FINAL_TILE_WIDTH_BY_AXIS:
        width = K_FINAL_TILE_WIDTH_BY_AXIS[axis]
        return Fraction((key.k - 1) % width + 1, 1)
    if axis in N_TILE_UTILIZATION_WIDTH_BY_AXIS:
        width = N_TILE_UTILIZATION_WIDTH_BY_AXIS[axis]
        tiles = _n_tile_count(key.aggregate_n, width)
        return Fraction(key.aggregate_n, tiles * width)
    if axis in N_TILE_ALIGNED_WIDTH_BY_AXIS:
        width = N_TILE_ALIGNED_WIDTH_BY_AXIS[axis]
        return Fraction(int(key.aggregate_n % width == 0), 1)
    if axis in N_PARALLEL_WAVE_WIDTH_BY_AXIS:
        tasks = _n_tile_count(
            key.aggregate_n, N_PARALLEL_WAVE_WIDTH_BY_AXIS[axis]
        )
        return Fraction(_n_tile_count(tasks, parallelism_width), 1)
    if axis in N_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS:
        tasks = _n_tile_count(
            key.aggregate_n, N_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS[axis]
        )
        final_wave_tasks = (tasks - 1) % parallelism_width + 1
        return Fraction(final_wave_tasks, parallelism_width)
    if axis in MN_PARALLEL_WAVE_WIDTH_BY_AXIS:
        tasks = key.m * _n_tile_count(
            key.aggregate_n, MN_PARALLEL_WAVE_WIDTH_BY_AXIS[axis]
        )
        return Fraction(_n_tile_count(tasks, parallelism_width), 1)
    if axis in MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS:
        tasks = key.m * _n_tile_count(
            key.aggregate_n, MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS[axis]
        )
        final_wave_tasks = (tasks - 1) % parallelism_width + 1
        return Fraction(final_wave_tasks, parallelism_width)
    raise ValueError(f"unknown NativeVNNI feature axis {axis!r}")


def _threshold_between(
    axis: FeatureAxis,
    lower: Fraction,
    upper: Fraction,
    placement: BoundaryPlacement,
    parallelism_width: int,
    task_multiplier: int,
) -> FeatureThreshold:
    """Place one reviewed threshold between adjacent observations.

    Every rational value in ``[lower, upper)`` induces the same training
    partition but can route unseen values differently. Grouped development CV
    chooses midpoint interpolation or the more aggressive lower edge per
    domain before the final policy is fitted.
    """

    if not lower < upper:
        raise ValueError("feature split requires strictly increasing values")
    selected = (
        (lower + upper) / 2
        if placement == BoundaryPlacement.MIDPOINT
        else lower
    )
    return FeatureThreshold(
        axis,
        selected.numerator,
        selected.denominator,
        parallelism_width if axis in PARALLEL_WAVE_AXES else 1,
        task_multiplier if axis in MN_PARALLEL_WAVE_AXES else 1,
    )


def _accelerated_threshold_components_between(
    axis: FeatureAxis,
    lower: Fraction,
    upper: Fraction,
    placement: BoundaryPlacement,
    parallelism_width: int,
    task_multiplier: int,
) -> tuple[int, int, int, int]:
    """Return one normalized threshold without allocating a Fraction/dataclass.

    Accelerator metadata enumerates every ordered pair of observed axis values,
    not merely adjacent values. A 335-point full-geometry fold can therefore
    evaluate millions of pairs. Constructing a temporary ``Fraction`` and
    ``FeatureThreshold`` for each pair dominated the complete GPU fit even
    though most objects were immediately discarded by a set.

    This helper performs the identical rational midpoint normalization with
    integer arithmetic and returns the four scalar fields used for deduplication.
    Only unique tuples become ``FeatureThreshold`` objects later.
    """

    if not lower < upper:
        raise ValueError("feature split requires strictly increasing values")
    if placement == BoundaryPlacement.MIDPOINT:
        numerator = (
            lower.numerator * upper.denominator
            + upper.numerator * lower.denominator
        )
        denominator = 2 * lower.denominator * upper.denominator
        divisor = math.gcd(numerator, denominator)
        numerator //= divisor
        denominator //= divisor
    else:
        numerator = lower.numerator
        denominator = lower.denominator
    return (
        numerator,
        denominator,
        parallelism_width if axis in PARALLEL_WAVE_AXES else 1,
        task_multiplier if axis in MN_PARALLEL_WAVE_AXES else 1,
    )


@dataclass(frozen=True)
class _LeafNode:
    """Internal fitted tree leaf before path predicates are materialized."""

    candidate_id: str
    points: tuple[tuple[RuntimeKey, str], ...]
    point_mask: int
    regrets: tuple[float, ...]
    measured_max_regret: float
    p95_regret: float
    mean_regret: float
    measured_failure_count: int

    @cached_property
    def signature(self) -> tuple:
        """Return the immutable leaf identity once for beam deduplication."""

        return ("leaf", self.candidate_id, tuple(point[1] for point in self.points))

    @cached_property
    def max_regret(self) -> float:
        """Return the exact worst point regret owned by this leaf."""

        return max(self.regrets, default=math.inf)

    @cached_property
    def failure_count(self) -> int:
        """Count measured point maxima at or above the strict budget."""

        return self.measured_failure_count

    @cached_property
    def failed_leaf_count(self) -> int:
        """Return one exactly when this segment fails its measured p95 gate."""

        return int(self.p95_regret >= GENERIC_REGRET_BUDGET)

    @cached_property
    def worst_leaf_p95_regret(self) -> float:
        """Expose this segment's measured p95 for recursive tree scoring."""

        return self.p95_regret


@dataclass(frozen=True)
class _SplitNode:
    """Internal fitted binary split with deterministic child ownership."""

    threshold: FeatureThreshold
    left: "_TreeNode"
    right: "_TreeNode"

    @cached_property
    def signature(self) -> tuple:
        """Return the recursively stable tree identity once per node."""

        return (
            "split",
            FEATURE_AXIS_PRIORITY[self.threshold.axis],
            self.threshold.axis.value,
            self.threshold.numerator,
            self.threshold.denominator,
            self.threshold.parallelism_width,
            self.threshold.task_multiplier,
            self.left.signature,
            self.right.signature,
        )

    @cached_property
    def regrets(self) -> tuple[float, ...]:
        """Materialize child regrets only when a later objective key needs them."""

        return (*self.left.regrets, *self.right.regrets)

    @cached_property
    def max_regret(self) -> float:
        """Combine the child maxima without rescanning every owned point."""

        return max(self.left.max_regret, self.right.max_regret)

    @cached_property
    def failure_count(self) -> int:
        """Combine measured point-failure counts from disjoint children."""

        return self.left.failure_count + self.right.failure_count

    @cached_property
    def failed_leaf_count(self) -> int:
        """Count child segments whose measured p95 is not promotable."""

        return self.left.failed_leaf_count + self.right.failed_leaf_count

    @cached_property
    def worst_leaf_p95_regret(self) -> float:
        """Return the exact worst measured segment p95 in this subtree."""

        return max(
            self.left.worst_leaf_p95_regret,
            self.right.worst_leaf_p95_regret,
        )

    @cached_property
    def measured_max_regret(self) -> float:
        """Return the diagnostic maximum measured surface regret."""

        return max(
            self.left.measured_max_regret,
            self.right.measured_max_regret,
        )


_TreeNode = _LeafNode | _SplitNode


def _tree_leaves(root: _TreeNode) -> tuple[_LeafNode, ...]:
    """Return fitted leaves in stable preorder for reports and publication."""

    if isinstance(root, _LeafNode):
        return (root,)
    return (*_tree_leaves(root.left), *_tree_leaves(root.right))


@dataclass(frozen=True)
class _TreeFit:
    """One fitted tree and the complete development regret distribution."""

    root: _TreeNode
    leaf_count: int

    @property
    def regrets(self) -> tuple[float, ...]:
        """Expose the immutable root distribution for reports and later keys."""

        return self.root.regrets

    @property
    def primary_objective(self) -> tuple[int, float]:
        """Return the exact installation-aligned beam keys.

        A production tree is promotable only when every generic segment has
        measured p95 regret strictly below five percent.  The first key makes
        any all-promotable tree categorically better than a tree that would be
        rejected after fitting.  The second key is the bounded fitting p95;
        it is measured-only unless grouped CV explicitly selects profiler
        influence, and it cannot change the first key.
        """

        return (
            self.root.failed_leaf_count,
            _percentile(self.regrets, 0.95),
        )

    @cached_property
    def objective(self) -> tuple:
        """Return the immutable lexicographic beam objective once per fit."""

        return (
            *self.primary_objective,
            _ordered_mean(self.regrets),
            self.root.worst_leaf_p95_regret,
            self.root.failure_count,
            self.root.measured_max_regret,
            self.leaf_count,
            self.root.signature,
        )


def _finish_best_leaf(
    points: tuple[tuple[RuntimeKey, str], ...],
    matrix: Mapping[
        tuple[RuntimeKey, str], Mapping[str, CandidatePointCost]
    ],
    primary_candidates: Iterable[str],
    primary_best: tuple[int, float],
    point_mask: int = 0,
) -> _TreeFit | None:
    """Resolve exact later objective keys for primary-key survivors only."""

    scored = []
    for candidate in sorted(primary_candidates):
        if any(candidate not in matrix[point] for point in points):
            raise RuntimeError(
                "primary scorer returned a candidate unavailable at a leaf point"
            )
        rows = [matrix[point][candidate] for point in points]
        regrets = tuple(row.selection_regret for row in rows)
        measured_p95_regret = _percentile(
            (row.p95_surface_regret for row in rows), 0.95
        )
        primary = (
            int(measured_p95_regret >= GENERIC_REGRET_BUDGET),
            _percentile(regrets, 0.95),
        )
        if primary != primary_best:
            raise RuntimeError(
                "primary scorer survivor does not match its published exact key"
            )
        scored.append((
            primary_best[0],
            primary_best[1],
            _ordered_mean(regrets),
            measured_p95_regret,
            _ordered_mean(row.mean_surface_regret for row in rows),
            sum(
                row.max_surface_regret >= GENERIC_REGRET_BUDGET
                for row in rows
            ),
            max(row.max_surface_regret for row in rows),
            candidate,
            regrets,
        ))
    if not scored:
        return None
    score = min(scored, key=lambda item: item[:8])
    leaf = _LeafNode(
        candidate_id=score[7],
        points=points,
        point_mask=point_mask,
        regrets=score[8],
        measured_max_regret=score[6],
        p95_regret=score[3],
        mean_regret=score[4],
        measured_failure_count=score[5],
    )
    return _TreeFit(root=leaf, leaf_count=1)


def _best_leaf(
    points: tuple[tuple[RuntimeKey, str], ...],
    matrix: Mapping[
        tuple[RuntimeKey, str], Mapping[str, CandidatePointCost]
    ],
    point_mask: int = 0,
) -> _TreeFit | None:
    """Choose the worst-regret-optimal candidate shared by every point."""

    common_candidates = set(matrix[points[0]])
    for point in points[1:]:
        common_candidates.intersection_update(matrix[point])
    if not common_candidates:
        return None

    # The leaf objective is lexicographic.  Canonical measured p95 owns the hard
    # pass/fail key; bounded fitting p95 may then distinguish candidates without
    # manufacturing installation evidence.  Retain exact ties on those first
    # two keys and evaluate measured diagnostics and stable names on the usually
    # tiny survivor set.
    primary_by_candidate = {}
    for candidate in sorted(common_candidates):
        rows = tuple(matrix[point][candidate] for point in points)
        regrets = tuple(row.selection_regret for row in rows)
        measured_p95_regret = _percentile(
            (row.p95_surface_regret for row in rows), 0.95
        )
        primary_by_candidate[candidate] = (
            int(measured_p95_regret >= GENERIC_REGRET_BUDGET),
            _percentile(regrets, 0.95),
        )
    primary_best = min(primary_by_candidate.values())
    return _finish_best_leaf(
        points,
        matrix,
        (
            candidate
            for candidate, primary in primary_by_candidate.items()
            if primary == primary_best
        ),
        primary_best,
        point_mask,
    )


def _materialize_accelerated_tree_results(
    results: tuple[AcceleratedTreeFitResult, ...],
    *,
    points: tuple[tuple[RuntimeKey, str], ...],
    matrix: Mapping[
        tuple[RuntimeKey, str], Mapping[str, CandidatePointCost]
    ],
    candidates: tuple[str, ...],
) -> tuple[_TreeFit | None, ...]:
    """Reconstruct immutable Python nodes from compact device tree results."""

    fitted = []
    for result in results:
        if not result.has_tree:
            fitted.append(None)
            continue
        leaves = []
        for point_mask, candidate_index in zip(
            result.leaf_masks, result.candidate_indices
        ):
            if candidate_index >= len(candidates):
                raise RuntimeError(
                    "GPU tree search returned an out-of-range candidate"
                )
            subset = tuple(
                point
                for point_index, point in enumerate(points)
                if point_mask & (1 << point_index)
            )
            candidate = candidates[candidate_index]
            rows = tuple(matrix[point][candidate] for point in subset)
            primary = (
                int(_percentile(
                    (row.p95_surface_regret for row in rows), 0.95
                ) >= GENERIC_REGRET_BUDGET),
                _percentile((row.selection_regret for row in rows), 0.95),
            )
            leaf_fit = _finish_best_leaf(
                subset,
                matrix,
                (candidate,),
                primary,
                point_mask,
            )
            if leaf_fit is None:
                raise RuntimeError("GPU tree search returned an empty leaf")
            leaves.append(leaf_fit.root)

        token_index = 0
        leaf_index = 0

        def parse() -> _TreeNode:
            """Consume one preorder structure node and matching leaf."""

            nonlocal token_index, leaf_index
            if token_index >= len(result.structure_tokens):
                raise RuntimeError("GPU tree structure ended inside a node")
            position = token_index
            token = result.structure_tokens[position]
            token_index += 1
            if token == 0:
                if leaf_index >= len(leaves):
                    raise RuntimeError("GPU tree structure has excess leaves")
                leaf = leaves[leaf_index]
                leaf_index += 1
                return leaf
            if token != 1 or position >= len(result.split_thresholds):
                raise RuntimeError(
                    "GPU tree structure returned an invalid split marker"
                )
            descriptor = result.split_thresholds[position]
            if descriptor is None:
                raise RuntimeError("GPU tree split omitted its exact threshold")
            return _SplitNode(
                _feature_threshold_from_accelerated_descriptor(descriptor),
                parse(),
                parse(),
            )

        root = parse()
        if token_index != len(result.structure_tokens) or leaf_index != len(leaves):
            raise RuntimeError("GPU tree structure did not consume exact output")
        fitted.append(_TreeFit(root=root, leaf_count=len(leaves)))
    return tuple(fitted)


def _fit_tree_budgets_accelerated(
    points: tuple[tuple[RuntimeKey, str], ...],
    matrix: Mapping[
        tuple[RuntimeKey, str], Mapping[str, CandidatePointCost]
    ],
    *,
    max_leaves: int,
    min_shape_groups_per_leaf: int,
    boundary_placement: BoundaryPlacement,
    feature_policy: FeaturePolicy,
    primary_scorer: NativeVNNILeafPrimaryScorer,
) -> tuple[_TreeFit | None, ...]:
    """Fit one complete beam on the explicitly selected CUDA/ROCm device."""

    feature_axes = FEATURE_AXES_BY_POLICY[feature_policy]
    parallelism_widths = {
        _architecture_parallelism_width(point[0].architecture_class)
        for point in points
    }
    if len(parallelism_widths) != 1:
        raise ValueError("one generic policy domain changed parallelism width")
    parallelism_width = next(iter(parallelism_widths))
    task_multipliers = {point[0].m for point in points}
    if len(task_multipliers) != 1:
        raise ValueError("one generic policy domain changed its M multiplier")
    task_multiplier = next(iter(task_multipliers))
    candidates = tuple(sorted({
        candidate
        for point in points
        for candidate in matrix[point]
    }))
    fitting_regrets = array("d")
    measured_p95_regrets = array("d")
    measured_mean_regrets = array("d")
    measured_max_regrets = array("d")
    for point in points:
        for candidate in candidates:
            cost = matrix[point].get(candidate)
            fitting_regrets.append(
                cost.selection_regret if cost is not None else math.inf
            )
            measured_p95_regrets.append(
                cost.p95_surface_regret if cost is not None else math.inf
            )
            measured_mean_regrets.append(
                cost.mean_surface_regret if cost is not None else math.inf
            )
            measured_max_regrets.append(
                cost.max_surface_regret if cost is not None else math.inf
            )

    shape_groups = tuple(sorted({point[1] for point in points}))
    shape_group_rank = {
        shape_group: rank for rank, shape_group in enumerate(shape_groups)
    }
    device_results = primary_scorer.fit_tree_budgets(
        fitting_regrets,
        measured_p95_regrets,
        measured_mean_regrets,
        measured_max_regrets,
        point_count=len(points),
        candidate_count=len(candidates),
        point_group_ranks=tuple(
            shape_group_rank[point[1]] for point in points
        ),
        training_aggregate_n=tuple(
            point[0].aggregate_n for point in points
        ),
        training_k=tuple(point[0].k for point in points),
        feature_axes=tuple(
            _accelerated_feature_axis_descriptor(axis)
            for axis in feature_axes
        ),
        boundary_placement=_accelerated_boundary_placement(
            boundary_placement
        ),
        parallelism_width=parallelism_width,
        task_multiplier=task_multiplier,
        min_shape_groups_per_leaf=min_shape_groups_per_leaf,
        max_leaves=max_leaves,
    )
    return _materialize_accelerated_tree_results(
        device_results,
        points=points,
        matrix=matrix,
        candidates=candidates,
    )


def _fit_and_evaluate_tree_budgets_accelerated(
    training_costs: list[CandidatePointCost],
    heldout_costs: list[CandidatePointCost],
    *,
    max_leaves: int,
    min_shape_groups_per_leaf: int,
    boundary_placement: BoundaryPlacement,
    feature_policy: FeaturePolicy,
    primary_scorer: NativeVNNILeafPrimaryScorer,
) -> tuple[_ComplexityFoldResult, ...]:
    """Run grouped-CV search and heldout scoring in one accelerator graph.

    The union candidate inventory is deliberate. A candidate may be absent
    from every training group yet present in a heldout group; its all-infinite
    training column cannot enter a fitted leaf, while its measured heldout
    column must remain eligible to become the exact oracle winner.
    """

    training_matrix = _point_matrix(training_costs)
    training_points = tuple(sorted(
        training_matrix,
        key=lambda point: (
            point[0].aggregate_n,
            point[0].k,
            point[1],
        ),
    ))
    heldout_matrix = _point_matrix(heldout_costs)
    heldout_points = tuple(sorted(
        heldout_matrix,
        key=lambda point: (
            point[0].aggregate_n,
            point[0].k,
            point[1],
        ),
    ))
    required = len(heldout_points)
    if len({point[1] for point in training_points}) < min_shape_groups_per_leaf:
        return tuple(
            _ComplexityFoldResult(
                complexity=complexity,
                decisions=(),
                uncovered=required,
                required=required,
            )
            for complexity in range(1, max_leaves + 1)
        )

    feature_axes = FEATURE_AXES_BY_POLICY[feature_policy]
    parallelism_widths = {
        _architecture_parallelism_width(point[0].architecture_class)
        for point in training_points
    }
    if len(parallelism_widths) != 1:
        raise ValueError("one generic policy domain changed parallelism width")
    parallelism_width = next(iter(parallelism_widths))
    task_multipliers = {point[0].m for point in training_points}
    if len(task_multipliers) != 1:
        raise ValueError("one generic policy domain changed its M multiplier")
    task_multiplier = next(iter(task_multipliers))
    candidates = tuple(sorted({
        candidate
        for matrix in (training_matrix, heldout_matrix)
        for point_candidates in matrix.values()
        for candidate in point_candidates
    }))
    fitting_regrets = array("d")
    measured_p95_regrets = array("d")
    measured_mean_regrets = array("d")
    measured_max_regrets = array("d")
    for point in training_points:
        for candidate in candidates:
            cost = training_matrix[point].get(candidate)
            fitting_regrets.append(
                cost.selection_regret if cost is not None else math.inf
            )
            measured_p95_regrets.append(
                cost.p95_surface_regret if cost is not None else math.inf
            )
            measured_mean_regrets.append(
                cost.mean_surface_regret if cost is not None else math.inf
            )
            measured_max_regrets.append(
                cost.max_surface_regret if cost is not None else math.inf
            )

    heldout_measured_p95 = array("d")
    heldout_measured_means = array("d")
    heldout_measured_maxima = array("d")
    for point in heldout_points:
        for candidate in candidates:
            cost = heldout_matrix[point].get(candidate)
            heldout_measured_p95.append(
                cost.p95_surface_regret if cost is not None else math.inf
            )
            heldout_measured_means.append(
                cost.mean_surface_regret if cost is not None else math.inf
            )
            heldout_measured_maxima.append(
                cost.max_surface_regret if cost is not None else math.inf
            )

    shape_groups = tuple(sorted({point[1] for point in training_points}))
    shape_group_rank = {
        shape_group: rank for rank, shape_group in enumerate(shape_groups)
    }
    evaluations = primary_scorer.fit_tree_budgets_and_evaluate(
        fitting_regrets,
        measured_p95_regrets,
        measured_mean_regrets,
        measured_max_regrets,
        point_count=len(training_points),
        candidate_count=len(candidates),
        point_group_ranks=tuple(
            shape_group_rank[point[1]] for point in training_points
        ),
        training_aggregate_n=tuple(
            point[0].aggregate_n for point in training_points
        ),
        training_k=tuple(point[0].k for point in training_points),
        feature_axes=tuple(
            _accelerated_feature_axis_descriptor(axis)
            for axis in feature_axes
        ),
        boundary_placement=_accelerated_boundary_placement(
            boundary_placement
        ),
        parallelism_width=parallelism_width,
        task_multiplier=task_multiplier,
        heldout_aggregate_n=tuple(
            point[0].aggregate_n for point in heldout_points
        ),
        heldout_k=tuple(point[0].k for point in heldout_points),
        heldout_measured_p95_regrets=heldout_measured_p95,
        heldout_measured_mean_regrets=heldout_measured_means,
        heldout_measured_max_regrets=heldout_measured_maxima,
        min_shape_groups_per_leaf=min_shape_groups_per_leaf,
        max_leaves=max_leaves,
    )

    results = []
    for complexity, evaluation in enumerate(evaluations, start=1):
        decisions = []
        for point, selected_index, exact_index in zip(
            heldout_points,
            evaluation.selected_candidate_indices,
            evaluation.exact_candidate_indices,
        ):
            if selected_index == UINT32_MAX:
                continue
            selected = heldout_matrix[point].get(candidates[selected_index])
            exact = heldout_matrix[point].get(candidates[exact_index])
            if selected is None or exact is None:
                raise RuntimeError(
                    "GPU heldout evaluation selected an unavailable candidate"
                )
            decisions.append((selected, exact))
        results.append(_ComplexityFoldResult(
            complexity=complexity,
            decisions=tuple(decisions),
            uncovered=required - len(decisions),
            required=required,
        ))
    return tuple(results)


def _fit_tree_budgets(
    costs: list[CandidatePointCost],
    *,
    max_leaves: int,
    min_shape_groups_per_leaf: int,
    boundary_placement: BoundaryPlacement = BoundaryPlacement.MIDPOINT,
    feature_policy: FeaturePolicy = FeaturePolicy.CONTINUOUS,
    primary_scorer: NativeVNNILeafPrimaryScorer | None = None,
) -> tuple[_TreeFit | None, ...]:
    """Fit every leaf budget with a deterministic bounded tree beam.

    Exhaustively enumerating every axis-aligned tree is exponential in the
    number of reviewed shapes and became impractical as soon as the production
    manifest grew beyond the initial sparse corpus. The beam retains the best
    ``TREE_BEAM_WIDTH`` distinct exact-leaf-count trees at each depth. Every
    retained tree is still scored against every point and every common
    candidate, and every legal feature threshold remains available when any
    retained leaf is expanded.
    """

    if max_leaves < 1 or max_leaves > MAX_TREE_LEAVES:
        raise ValueError(
            "policy feature schema v3 permits one to "
            f"{MAX_TREE_LEAVES} leaves"
        )
    matrix = _point_matrix(costs)
    points = tuple(sorted(
        matrix,
        key=lambda point: (
            point[0].aggregate_n,
            point[0].k,
            point[1],
        ),
    ))
    if len({point[1] for point in points}) < min_shape_groups_per_leaf:
        return tuple(None for _ in range(max_leaves))

    if primary_scorer is not None:
        return _fit_tree_budgets_accelerated(
            points,
            matrix,
            max_leaves=max_leaves,
            min_shape_groups_per_leaf=min_shape_groups_per_leaf,
            boundary_placement=boundary_placement,
            feature_policy=feature_policy,
            primary_scorer=primary_scorer,
        )

    feature_axes = FEATURE_AXES_BY_POLICY[feature_policy]
    parallelism_widths = {
        _architecture_parallelism_width(point[0].architecture_class)
        for point in points
    }
    if len(parallelism_widths) != 1:
        raise ValueError("one generic policy domain changed parallelism width")
    parallelism_width = next(iter(parallelism_widths))
    task_multipliers = {point[0].m for point in points}
    if len(task_multipliers) != 1:
        raise ValueError("one generic policy domain changed its M multiplier")
    task_multiplier = next(iter(task_multipliers))
    feature_values = {
        (axis, point): _feature_value(axis, point, parallelism_width)
        for axis in feature_axes
        for point in points
    }
    full_point_mask = (1 << len(points)) - 1
    shape_groups = tuple(sorted({point[1] for point in points}))
    shape_group_index = {
        shape_group: index
        for index, shape_group in enumerate(shape_groups)
    }
    point_shape_group_bits = tuple(
        1 << shape_group_index[point[1]]
        for point in points
    )

    @lru_cache(maxsize=None)
    def points_from_mask(
        subset_mask: int,
    ) -> tuple[tuple[RuntimeKey, str], ...]:
        """Return stable matrix-order points for one canonical subset mask."""

        if subset_mask <= 0 or subset_mask & ~full_point_mask:
            raise ValueError("tree subset mask is outside the fitted point matrix")
        return tuple(
            point
            for index, point in enumerate(points)
            if subset_mask & (1 << index)
        )

    @lru_cache(maxsize=None)
    def shape_group_count(subset_mask: int) -> int:
        """Count distinct split groups without constructing a Python set."""

        group_mask = 0
        remaining = subset_mask
        while remaining:
            bit = remaining & -remaining
            point_index = bit.bit_length() - 1
            group_mask |= point_shape_group_bits[point_index]
            remaining ^= bit
        return group_mask.bit_count()

    @lru_cache(maxsize=None)
    def threshold_point_mask(threshold: FeatureThreshold) -> int:
        """Cache global measured-point membership for one exact threshold."""

        mask = 0
        for point_index, point in enumerate(points):
            if threshold.matches_less_equal(
                point[0].aggregate_n, point[0].k
            ):
                mask |= 1 << point_index
        return mask

    @lru_cache(maxsize=None)
    def best_leaf(
        subset_mask: int,
    ) -> _TreeFit | None:
        """Score one point subset once, independently of tree budget."""

        return _best_leaf(
            points_from_mask(subset_mask),
            matrix,
            subset_mask,
        )

    @lru_cache(maxsize=None)
    def candidate_splits(
        subset_mask: int,
    ) -> tuple[
        tuple[
            FeatureThreshold,
            int,
            int,
        ],
        ...,
    ]:
        """Enumerate legal splits with bitwise child-subset construction."""

        result = []
        subset = points_from_mask(subset_mask)
        for axis in feature_axes:
            values = sorted({feature_values[(axis, point)] for point in subset})
            for lower, upper in zip(values, values[1:]):
                threshold = _threshold_between(
                    axis,
                    lower,
                    upper,
                    boundary_placement,
                    parallelism_width,
                    task_multiplier,
                )
                left_mask = subset_mask & threshold_point_mask(threshold)
                right_mask = subset_mask ^ left_mask
                if (
                    shape_group_count(left_mask) < min_shape_groups_per_leaf
                    or shape_group_count(right_mask) < min_shape_groups_per_leaf
                ):
                    continue
                # Two feature axes can induce the same partition on measured
                # points while routing an unseen point differently. Retaining
                # both thresholds lets grouped CV judge that distinction.
                result.append((threshold, left_mask, right_mask))
        return tuple(result)

    @lru_cache(maxsize=None)
    def leaves(root: _TreeNode) -> tuple[_LeafNode, ...]:
        """Return fitted leaves in stable preorder for deterministic expansion."""

        if isinstance(root, _LeafNode):
            return (root,)
        return (*leaves(root.left), *leaves(root.right))

    def replace_leaf(
        root: _TreeNode,
        target: _LeafNode,
        replacement: _SplitNode,
    ) -> _TreeNode:
        """Return an immutable tree with exactly one leaf replaced."""

        if isinstance(root, _LeafNode):
            return replacement if root == target else root
        return _SplitNode(
            root.threshold,
            replace_leaf(root.left, target, replacement),
            replace_leaf(root.right, target, replacement),
        )

    initial = best_leaf(full_point_mask)
    if initial is None:
        return tuple(None for _ in range(max_leaves))

    frontier = (initial,)
    incumbent = initial
    results: list[_TreeFit | None] = [initial]
    for exact_leaf_count in range(2, max_leaves + 1):
        expanded: dict[tuple, _TreeFit] = {}

        def expansion_inputs():
            """Yield every legal one-leaf expansion in deterministic order."""

            for fit in frontier:
                for target in leaves(fit.root):
                    for threshold, left_mask, right_mask in candidate_splits(
                        target.point_mask
                    ):
                        yield fit, target, threshold, left_mask, right_mask

        depth_expansions = expansion_inputs()

        for fit, target, threshold, left_mask, right_mask in depth_expansions:
            left = best_leaf(left_mask)
            right = best_leaf(right_mask)
            if left is None or right is None:
                continue
            root = replace_leaf(
                fit.root,
                target,
                _SplitNode(threshold, left.root, right.root),
            )
            candidate = _TreeFit(
                root=root,
                leaf_count=exact_leaf_count,
            )
            previous = expanded.get(root.signature)
            if (
                previous is None
                or candidate.primary_objective < previous.primary_objective
                or (
                    candidate.primary_objective == previous.primary_objective
                    and candidate.objective < previous.objective
                )
            ):
                expanded[root.signature] = candidate

        if not expanded:
            results.extend([incumbent] * (max_leaves - exact_leaf_count + 1))
            break
        # The complete objective is lexicographic.  Primary groups with worse
        # max-regret/failure-count keys cannot enter the beam while an earlier
        # group still has capacity.  Materialize p95/mean and recursive
        # signatures only for groups that can contribute to the retained beam.
        primary_groups: dict[tuple[float, int], list[_TreeFit]] = defaultdict(list)
        for candidate in expanded.values():
            primary_groups[candidate.primary_objective].append(candidate)
        selected = []
        for primary in sorted(primary_groups):
            remaining = TREE_BEAM_WIDTH - len(selected)
            if remaining == 0:
                break
            selected.extend(sorted(
                primary_groups[primary], key=lambda fit: fit.objective
            )[:remaining])
        frontier = tuple(selected)
        best_at_depth = frontier[0]
        if best_at_depth.objective < incumbent.objective:
            incumbent = best_at_depth
        results.append(incumbent)

    return tuple(results)


def _fit_tree(
    costs: list[CandidatePointCost],
    *,
    max_leaves: int,
    min_shape_groups_per_leaf: int,
    boundary_placement: BoundaryPlacement = BoundaryPlacement.MIDPOINT,
    feature_policy: FeaturePolicy = FeaturePolicy.CONTINUOUS,
    primary_scorer: NativeVNNILeafPrimaryScorer | None = None,
) -> _TreeFit | None:
    """Return the best tree permitted by one maximum leaf budget."""

    return _fit_tree_budgets(
        costs,
        max_leaves=max_leaves,
        min_shape_groups_per_leaf=min_shape_groups_per_leaf,
        boundary_placement=boundary_placement,
        feature_policy=feature_policy,
        primary_scorer=primary_scorer,
    )[-1]


def _flatten_rules(
    domain: GenericDomain,
    root: _TreeNode,
    exemplar_by_candidate: Mapping[str, NativeVNNIObservation],
    predicates: tuple[FeaturePredicate, ...] = (),
) -> tuple[GenericDispatchRule, ...]:
    """Materialize non-overlapping root-to-leaf predicates in preorder."""

    if isinstance(root, _LeafNode):
        exemplar = exemplar_by_candidate[root.candidate_id]
        return (GenericDispatchRule(
            domain=domain,
            predicates=predicates,
            candidate_id=root.candidate_id,
            arithmetic_fingerprint=exemplar.arithmetic_fingerprint,
            development_shape_groups=tuple(sorted({
                point[1] for point in root.points
            })),
            development_max_regret=root.measured_max_regret,
            development_p95_regret=root.p95_regret,
            development_mean_regret=root.mean_regret,
        ),)
    left_predicate = FeaturePredicate(root.threshold, True)
    right_predicate = FeaturePredicate(root.threshold, False)
    return (
        *_flatten_rules(
            domain,
            root.left,
            exemplar_by_candidate,
            (*predicates, left_predicate),
        ),
        *_flatten_rules(
            domain,
            root.right,
            exemplar_by_candidate,
            (*predicates, right_predicate),
        ),
    )


def fit_domain_rules(
    domain: GenericDomain,
    costs: list[CandidatePointCost],
    observations: Iterable[NativeVNNIObservation],
    *,
    max_leaves: int = DEFAULT_TREE_LEAVES,
    min_shape_groups_per_leaf: int = 2,
    boundary_placement: BoundaryPlacement = BoundaryPlacement.MIDPOINT,
    feature_policy: FeaturePolicy = FeaturePolicy.CONTINUOUS,
    primary_scorer: NativeVNNILeafPrimaryScorer | None = None,
) -> tuple[GenericDispatchRule, ...]:
    """Fit and flatten one bounded multi-feature decision tree."""

    fit = _fit_tree(
        costs,
        max_leaves=max_leaves,
        min_shape_groups_per_leaf=min_shape_groups_per_leaf,
        boundary_placement=boundary_placement,
        feature_policy=feature_policy,
        primary_scorer=primary_scorer,
    )
    if fit is None:
        return ()
    exemplar_by_candidate = {
        row.candidate_id: row
        for row in observations
    }
    return _flatten_rules(domain, fit.root, exemplar_by_candidate)


def _domain_folds(
    domain: GenericDomain,
    costs: Iterable[CandidatePointCost],
    *,
    seed: str,
) -> tuple[frozenset[str], ...]:
    """Build deterministic geometry-atomic coherent validation regions.

    A generic dispatch claim concerns unseen geometry, not an unseen shape
    label.  All shape groups sharing one exact ``(N,K)`` point therefore leave
    development together; otherwise another alias of the held-out geometry
    leaks its timing surface into training.  Small domains use leave-one-
    geometry-out validation.  Larger domains recursively bisect the widest
    exact N/K region, producing five balanced spatial folds that deliberately
    withhold coherent neighborhoods and both support boundaries.

    The previous feature-sorted round-robin partition placed adjacent geometry
    in every training fold.  It measured interpolation only and consequently
    certified policies that failed badly below the development K envelope.
    The seed remains in geometric tie-breaking so the partition identity is
    explicit and reproducible without weakening spatial locality.
    """

    point_by_group: dict[str, RuntimeKey] = {}
    for cost in costs:
        point_by_group.setdefault(cost.shape_group_id, cost.runtime_key)
    groups_by_geometry: dict[tuple[int, int], set[str]] = defaultdict(set)
    for group, point in point_by_group.items():
        groups_by_geometry[(point.aggregate_n, point.k)].add(group)
    geometries = tuple(sorted(groups_by_geometry))
    if len(geometries) < MIN_GENERIC_CROSS_VALIDATION_SHAPE_GROUPS:
        return ()
    if len(geometries) < 10:
        return tuple(
            frozenset(groups_by_geometry[geometry])
            for geometry in geometries
        )

    regions: list[tuple[tuple[int, int], ...]] = [geometries]
    while len(regions) < 5:
        splittable = [
            (index, region)
            for index, region in enumerate(regions)
            if len(region) >= 2
        ]
        if not splittable:
            break

        def split_priority(item: tuple[int, tuple[tuple[int, int], ...]]):
            """Prefer the largest region, then its exact multiplicative span."""

            index, region = item
            n_values = [geometry[0] for geometry in region]
            k_values = [geometry[1] for geometry in region]
            n_span = Fraction(max(n_values), min(n_values))
            k_span = Fraction(max(k_values), min(k_values))
            return (
                len(region),
                max(n_span, k_span),
                min(n_span, k_span),
                -index,
            )

        region_index, region = max(splittable, key=split_priority)
        n_values = [geometry[0] for geometry in region]
        k_values = [geometry[1] for geometry in region]
        n_span = Fraction(max(n_values), min(n_values))
        k_span = Fraction(max(k_values), min(k_values))
        if n_span == k_span:
            tie_digest = hashlib.sha256(
                f"{seed}\0{domain!r}\0{region_index}\0{len(region)}".encode()
            ).digest()
            split_on_n = bool(tie_digest[0] & 1)
        else:
            split_on_n = n_span > k_span
        ordered = tuple(sorted(
            region,
            key=(
                (lambda geometry: (geometry[0], geometry[1]))
                if split_on_n
                else (lambda geometry: (geometry[1], geometry[0]))
            ),
        ))
        midpoint = len(ordered) // 2
        regions[region_index:region_index + 1] = [
            ordered[:midpoint],
            ordered[midpoint:],
        ]

    return tuple(
        frozenset(
            group
            for geometry in region
            for group in groups_by_geometry[geometry]
        )
        for region in regions
        if region
    )


def _evaluate_rules(
    rules: tuple[GenericDispatchRule, ...],
    held_out_costs: list[CandidatePointCost],
) -> tuple[list[tuple[CandidatePointCost, CandidatePointCost]], int, int]:
    """Evaluate frozen tree leaves on held-out shape points."""

    matrix = _point_matrix(held_out_costs)
    decisions: list[tuple[CandidatePointCost, CandidatePointCost]] = []
    uncovered = 0
    for (key, _shape_group), candidates in matrix.items():
        matches = [rule for rule in rules if rule.matches(key.aggregate_n, key.k)]
        if len(matches) != 1 or matches[0].candidate_id not in candidates:
            uncovered += 1
            continue
        selected = candidates[matches[0].candidate_id]
        exact = min(
            candidates.values(),
            key=lambda cost: (
                cost.max_surface_regret,
                cost.p95_surface_regret,
                cost.mean_surface_regret,
                cost.candidate_id,
            ),
        )
        decisions.append((selected, exact))
    return decisions, uncovered, len(matrix)


def _evaluate_tree(
    root: _TreeNode | None,
    held_out_costs: list[CandidatePointCost],
) -> tuple[list[tuple[CandidatePointCost, CandidatePointCost]], int, int]:
    """Evaluate a fitted CV tree without materializing production rule metadata.

    Cross-validation needs only the selected candidate ID at each held-out
    point. Arithmetic fingerprints come from observations when the final policy
    is frozen; projecting and retaining millions of formula observation aliases
    merely to flatten temporary CV trees is redundant.
    """

    matrix = _point_matrix(held_out_costs)
    if root is None:
        return [], len(matrix), len(matrix)
    decisions = []
    uncovered = 0
    for (key, _shape_group), candidates in matrix.items():
        node = root
        while isinstance(node, _SplitNode):
            node = (
                node.left
                if node.threshold.matches_less_equal(key.aggregate_n, key.k)
                else node.right
            )
        selected = candidates.get(node.candidate_id)
        if selected is None:
            uncovered += 1
            continue
        exact = min(
            candidates.values(),
            key=lambda cost: (
                cost.max_surface_regret,
                cost.p95_surface_regret,
                cost.mean_surface_regret,
                cost.candidate_id,
            ),
        )
        decisions.append((selected, exact))
    return decisions, uncovered, len(matrix)


@dataclass(frozen=True)
class _ComplexityFoldResult:
    """Held-out decisions for one placement, fold, and leaf budget."""

    complexity: int
    decisions: tuple[tuple[CandidatePointCost, CandidatePointCost], ...]
    uncovered: int
    required: int


@dataclass(frozen=True)
class _PlacementFoldResult:
    """All leaf budgets evaluated for one independent CV work item."""

    domain: GenericDomain
    fold_index: int
    feature_policy: FeaturePolicy
    placement: BoundaryPlacement
    profiler_influence: ProfilerInfluence
    complexities: tuple[_ComplexityFoldResult, ...]


def _cross_validation_cell(
    selected: CandidatePointCost,
    exact: CandidatePointCost,
) -> CrossValidationCell:
    """Materialize one held-out decision without an intermediate collection."""

    return CrossValidationCell(
        runtime_key=selected.runtime_key,
        shape_group_id=selected.shape_group_id,
        selected_candidate_id=selected.candidate_id,
        exact_candidate_id=exact.candidate_id,
        observed_broad_regret=selected.max_surface_regret,
    )


def _cross_validation_cells(
    decisions: Iterable[tuple[CandidatePointCost, CandidatePointCost]],
) -> tuple[CrossValidationCell, ...]:
    """Materialize the selected model's held-out decisions in stable order."""

    return tuple(sorted(
        (
            _cross_validation_cell(selected, exact)
            for selected, exact in decisions
        ),
        key=lambda cell: (
            cell.runtime_key,
            cell.shape_group_id,
        ),
    ))


def _evaluate_placement_fold(
    args,
    primary_scorer: NativeVNNILeafPrimaryScorer | None = None,
) -> _PlacementFoldResult:
    """Fit one fold/placement tree sequence in an isolated worker process."""

    (
        domain,
        fold_index,
        held_out_groups,
        feature_policy,
        placement,
        profiler_influence,
        costs,
        max_leaves,
        min_shape_groups_per_leaf,
    ) = args
    training_costs = [
        cost for cost in costs if cost.shape_group_id not in held_out_groups
    ]
    held_out_costs = [
        cost for cost in costs if cost.shape_group_id in held_out_groups
    ]
    if primary_scorer is not None:
        try:
            results = list(_fit_and_evaluate_tree_budgets_accelerated(
                training_costs,
                held_out_costs,
                max_leaves=max_leaves,
                min_shape_groups_per_leaf=min_shape_groups_per_leaf,
                boundary_placement=placement,
                feature_policy=feature_policy,
                primary_scorer=primary_scorer,
            ))
        except Exception as error:
            training_group_count = len({
                cost.shape_group_id for cost in training_costs
            })
            heldout_group_count = len({
                cost.shape_group_id for cost in held_out_costs
            })
            raise RuntimeError(
                "accelerated grouped CV failed for "
                f"domain={domain!r}, fold={fold_index}, "
                f"feature_policy={feature_policy.value}, "
                f"placement={placement.value}, "
                f"profiler_influence={profiler_influence.value}, "
                f"training_groups={training_group_count}, "
                f"heldout_groups={heldout_group_count}, "
                f"max_leaves={max_leaves}: {error}"
            ) from error
    else:
        fits = _fit_tree_budgets(
            training_costs,
            max_leaves=max_leaves,
            min_shape_groups_per_leaf=min_shape_groups_per_leaf,
            boundary_placement=placement,
            feature_policy=feature_policy,
        )
        results = []
        for complexity, fit in enumerate(fits, start=1):
            decisions, uncovered, required = _evaluate_tree(
                fit.root if fit is not None else None,
                held_out_costs,
            )
            results.append(_ComplexityFoldResult(
                complexity=complexity,
                decisions=tuple(decisions),
                uncovered=uncovered,
                required=required,
            ))
    return _PlacementFoldResult(
        domain=domain,
        fold_index=fold_index,
        feature_policy=feature_policy,
        placement=placement,
        profiler_influence=profiler_influence,
        complexities=tuple(results),
    )


def _rank_domain_cross_validations(
    domain: GenericDomain,
    costs: list[CandidatePointCost],
    fold_count: int,
    fold_results: Iterable[_PlacementFoldResult],
    *,
    max_leaves: int,
    retain_model_cells: bool = True,
) -> tuple[DomainCrossValidation, ...]:
    """Return every evaluated model in normative deterministic CV order.

    Full-development fitting is a second model-selection surface: two models
    can both satisfy grouped held-out CV while only one produces p95-safe leaves
    after all development groups are fitted together. Retaining this ranked
    frontier lets that second gate choose among models already certified by CV
    instead of treating the first CV winner as an irreversible one-shot choice.
    """

    fold_results = tuple(fold_results)
    available_influences = tuple(
        influence
        for influence in PROFILER_INFLUENCES
        if any(
            result.profiler_influence == influence
            for result in fold_results
        )
    )
    model_keys = tuple(
        (feature_policy, placement, influence, complexity)
        for feature_policy in FEATURE_POLICIES
        for placement in BOUNDARY_PLACEMENTS
        for influence in available_influences
        for complexity in range(1, max_leaves + 1)
    )
    decisions_by_model = {key: [] for key in model_keys}
    uncovered_by_model = {key: 0 for key in model_keys}
    required_by_model = {key: 0 for key in model_keys}
    for result in fold_results:
        if result.domain != domain:
            raise ValueError("cross-validation fold result changed domain")
        for complexity in result.complexities:
            model_key = (
                result.feature_policy,
                result.placement,
                result.profiler_influence,
                complexity.complexity,
            )
            decisions_by_model[model_key].extend(complexity.decisions)
            uncovered_by_model[model_key] += complexity.uncovered
            required_by_model[model_key] += complexity.required

    scored = []
    competitive = {}
    shape_groups = {cost.shape_group_id for cost in costs}
    for feature_policy, placement, influence, complexity in model_keys:
        model_key = (feature_policy, placement, influence, complexity)
        decisions = decisions_by_model[model_key]
        all_regrets = [
            selected.max_surface_regret for selected, _exact in decisions
        ]
        uncovered = uncovered_by_model[model_key]
        required = required_by_model[model_key]
        if required == 0 or not decisions:
            continue
        worst_selected, worst_exact = max(
            decisions,
            key=lambda pair: (
                pair[0].max_surface_regret,
                pair[0].shape_group_id,
                pair[0].candidate_id,
            ),
        )
        max_regret = max(all_regrets, default=math.inf)
        p95_regret = _percentile(all_regrets, 0.95)
        mean_regret = statistics.fmean(all_regrets) if all_regrets else math.inf
        failed_cell_count = sum(
            regret >= GENERIC_REGRET_BUDGET for regret in all_regrets
        )
        if retain_model_cells:
            # Paired refinement needs the union of over-budget alternate model
            # edges, but it never needs a complete cell tuple for every model
            # in the tournament. A broad CPU decode domain evaluates roughly
            # two thousand models over hundreds of held-out points; eagerly
            # constructing and sorting every repeated cell made two large
            # domains dominate the otherwise parallel CV reduction. Intern
            # only distinct competitive edges here and hydrate the selected
            # model once after ranking.
            for selected, exact in decisions:
                if (
                    selected.max_surface_regret < GENERIC_REGRET_BUDGET
                    or selected.candidate_id == exact.candidate_id
                ):
                    continue
                identity = (
                    selected.runtime_key,
                    selected.shape_group_id,
                    selected.candidate_id,
                    exact.candidate_id,
                )
                if identity not in competitive:
                    competitive[identity] = _cross_validation_cell(
                        selected, exact
                    )
        objective = (
            uncovered,
            # Installation is governed by population p95 regret. Prefer a
            # model on the passing side of that boundary, then optimize p95
            # directly. Worst-cell regret remains a later diagnostic
            # tie-breaker so isolated outliers stay visible without allowing
            # them to veto an otherwise economical generic policy.
            p95_regret >= GENERIC_REGRET_BUDGET,
            p95_regret,
            mean_regret,
            failed_cell_count,
            max_regret,
            complexity,
            list(FEATURE_POLICIES).index(feature_policy),
            0 if placement == BoundaryPlacement.MIDPOINT else 1,
            PROFILER_INFLUENCES.index(influence),
        )
        scored.append((
            objective,
            DomainCrossValidation(
                domain=domain,
                selected_feature_policy=feature_policy,
                selected_max_leaves=complexity,
                selected_boundary_placement=placement,
                selected_profiler_influence=influence,
                fold_count=fold_count,
                shape_group_count=len(shape_groups),
                required_point_count=required,
                covered_point_count=required - uncovered,
                max_regret=max_regret,
                p95_regret=p95_regret,
                mean_regret=mean_regret,
                worst_shape_group_id=worst_selected.shape_group_id,
                worst_aggregate_n=worst_selected.runtime_key.aggregate_n,
                worst_k=worst_selected.runtime_key.k,
                worst_selected_candidate_id=worst_selected.candidate_id,
                worst_exact_candidate_id=worst_exact.candidate_id,
                cells=(),
            ),
        ))
    if not scored:
        return ()
    ranked = sorted(scored, key=lambda item: item[0])
    _, result = ranked[0]
    if result.covered_point_count != result.required_point_count:
        return ()

    if not retain_model_cells:
        return tuple(validation for _objective, validation in ranked)

    # Paired timing changes an individual candidate edge, and that correction
    # can promote the next CV model even when the currently selected tree is
    # fully measured. Retain the complete evaluated model frontier as compact
    # summaries and attach the deduplicated alternate-edge union to each one.
    # Only the selected model is hydrated eagerly; final publication can
    # hydrate another frontier member from the immutable fold results if the
    # first model fails its second-stage leaf gate.
    competitive_cells = tuple(sorted(
        competitive.values(),
        key=lambda cell: (
            cell.runtime_key,
            cell.shape_group_id,
            cell.selected_candidate_id,
            cell.exact_candidate_id,
        ),
    ))
    selected_validation = _hydrate_cross_validation_cells(
        result, fold_results
    )
    return tuple(
        replace(
            validation,
            cells=(selected_validation.cells if index == 0 else ()),
            competitive_cells=competitive_cells,
        )
        for index, (_objective, validation) in enumerate(ranked)
    )


def _hydrate_cross_validation_cells(
    validation: DomainCrossValidation,
    fold_results: Iterable[_PlacementFoldResult],
) -> DomainCrossValidation:
    """Attach held-out cells only to the model retained by production fitting.

    A production fit may evaluate thousands of alternate models per domain but
    emits only one selected CV report. Materializing every alternate model's
    repeated cell objects made host-side reduction single-threaded and
    allocation-bound. The immutable fold results already contain the exact
    decisions, so reconstruct the selected report once after CV or final-frontier
    selection. Paired-request planning keeps the eager detailed path because it
    intentionally consumes competitive edges from every model.
    """

    if validation.cells:
        return validation
    decisions = []
    uncovered = 0
    required = 0
    matching_folds = 0
    for result in fold_results:
        if (
            result.domain != validation.domain
            or result.feature_policy != validation.selected_feature_policy
            or result.placement != validation.selected_boundary_placement
            or result.profiler_influence
            != validation.selected_profiler_influence
        ):
            continue
        complexity_index = validation.selected_max_leaves - 1
        if not 0 <= complexity_index < len(result.complexities):
            raise ValueError(
                "selected CV complexity is absent from retained fold results"
            )
        complexity = result.complexities[complexity_index]
        decisions.extend(complexity.decisions)
        uncovered += complexity.uncovered
        required += complexity.required
        matching_folds += 1
    if matching_folds != validation.fold_count:
        raise ValueError(
            "selected CV model is incomplete in retained fold results: "
            f"expected {validation.fold_count}, observed {matching_folds}"
        )
    if (
        required != validation.required_point_count
        or required - uncovered != validation.covered_point_count
    ):
        raise ValueError("selected CV cell coverage changed during hydration")
    cells = _cross_validation_cells(decisions)
    if len(cells) != validation.covered_point_count:
        raise ValueError("selected CV cell cardinality changed during hydration")
    return replace(validation, cells=cells)


def _select_domain_cross_validation(
    domain: GenericDomain,
    costs: list[CandidatePointCost],
    fold_count: int,
    fold_results: Iterable[_PlacementFoldResult],
    *,
    max_leaves: int,
) -> DomainCrossValidation | None:
    """Return the first model from the complete deterministic CV frontier."""

    ranked = _rank_domain_cross_validations(
        domain,
        costs,
        fold_count,
        fold_results,
        max_leaves=max_leaves,
    )
    return ranked[0] if ranked else None


def _domain_fold_tasks(
    domain: GenericDomain,
    costs: list[CandidatePointCost],
    *,
    max_leaves: int,
    min_shape_groups_per_leaf: int,
    seed: str,
    profiler_training_pool_key: tuple[object, ...] | None = None,
    profiler_prediction_cache: dict[
        tuple[tuple[object, ...], tuple[tuple[int, int], ...]],
        dict[tuple[RuntimeKey, str, str], float | None],
    ] | None = None,
) -> tuple[tuple[tuple, ...], int]:
    """Build immutable fold/placement work from the sufficient cost matrix.

    Profiler prediction surfaces are precomputed independently for each fold
    from a cross-M, cross-format transfer pool. Every occurrence of a held-out
    N/K geometry is removed from that complete pool before fitting, so neither
    another codebook nor another prefill row count can leak the fold's timing
    answer. This routine deliberately consumes a read-only completed cache:
    task assembly must never hide a serial model fit.
    """

    folds = _domain_folds(domain, costs, seed=seed)
    if (profiler_training_pool_key is None) != (
        profiler_prediction_cache is None
    ):
        raise ValueError(
            "profiler-informed CV requires explicit prediction-cache ownership"
        )
    tasks = []
    for fold_index, held_out_groups in enumerate(folds):
        fold_cost_variants = [(
            ProfilerInfluence.MEASURED_ONLY,
            tuple(costs),
        )]
        if profiler_prediction_cache is not None:
            training_costs = [
                cost
                for cost in costs
                if cost.shape_group_id not in held_out_groups
            ]
            held_out_costs = [
                cost
                for cost in costs
                if cost.shape_group_id in held_out_groups
            ]
            held_out_geometries = {
                (cost.runtime_key.aggregate_n, cost.runtime_key.k)
                for cost in held_out_costs
            }
            prediction_cache_key = (
                profiler_training_pool_key,
                tuple(sorted(held_out_geometries)),
            )
            if prediction_cache_key not in profiler_prediction_cache:
                raise RuntimeError(
                    "profiler prediction cache was not completed before CV "
                    f"task assembly: {prediction_cache_key}"
                )
            predicted_regret_by_point = profiler_prediction_cache[
                prediction_cache_key
            ]
            informed_fold_costs = tuple((
                *(
                    replace(
                        cost,
                        profiler_predicted_regret=predicted_regret_by_point[(
                            cost.runtime_key,
                            cost.shape_group_id,
                            cost.candidate_id,
                        )],
                    )
                    for cost in training_costs
                ),
                *held_out_costs,
            ))
            fold_cost_variants.extend(
                (
                    influence,
                    tuple(
                        replace(
                            cost,
                            profiler_blend_weight=(
                                PROFILER_BLEND_WEIGHT_BY_INFLUENCE[influence]
                            ),
                        )
                        for cost in informed_fold_costs
                    ),
                )
                for influence in TREE_PROFILER_INFLUENCES
                if influence != ProfilerInfluence.MEASURED_ONLY
            )
        tasks.extend(
            (
                domain,
                fold_index,
                held_out_groups,
                feature_policy,
                placement,
                influence,
                fold_costs,
                max_leaves,
                min_shape_groups_per_leaf,
            )
            for influence, fold_costs in fold_cost_variants
            for feature_policy in FEATURE_POLICIES
            for placement in BOUNDARY_PLACEMENTS
        )
    return tuple(tasks), len(folds)


def _group_accelerated_cv_tasks(
    tasks: Iterable[tuple],
) -> tuple[tuple[tuple[int, tuple], ...], ...]:
    """Batch cost variants that share one immutable accelerator geometry.

    Measured-only and profiler-informed variants change only the fitting-regret
    matrix. Their training/held-out points, feature axes, threshold placement,
    and graph dimensions are identical. Keeping those variants in one worker
    transaction lets the first variant construct exact pair metadata and capture
    the graph; later variants reuse both while publishing independent CV results.

    Original task indices travel with every item so dynamic device scheduling
    cannot change the deterministic result order consumed by model ranking.
    """

    groups: dict[tuple[object, ...], list[tuple[int, tuple]]] = {}
    influences_by_group: dict[tuple[object, ...], set[ProfilerInfluence]] = {}
    for task_index, task in enumerate(tasks):
        if len(task) != 9:
            raise ValueError("accelerated CV task changed its tuple ABI")
        group_key = (
            task[0],  # GenericDomain
            task[1],  # fold index
            task[2],  # exact held-out group inventory
            task[3],  # feature policy
            task[4],  # threshold placement
            task[7],  # maximum leaves
            task[8],  # minimum groups per leaf
        )
        influence = task[5]
        observed_influences = influences_by_group.setdefault(group_key, set())
        if influence in observed_influences:
            raise ValueError(
                "accelerated CV geometry repeats one profiler influence"
            )
        observed_influences.add(influence)
        groups.setdefault(group_key, []).append((task_index, task))
    return tuple(tuple(group) for group in groups.values())


def _domain_profiler_teacher_fold_results(
    domain: GenericDomain,
    costs: list[CandidatePointCost],
    *,
    max_leaves: int,
    seed: str,
    profiler_training_pool_key: tuple[object, ...],
    profiler_prediction_cache: ProfilerPredictionCache,
) -> tuple[_PlacementFoldResult, ...]:
    """Score leakage-controlled teacher choices on held canonical timing.

    The ExtraTrees surrogate for each fold was trained after removing every
    occurrence of the held N/K geometry across all M values and codebooks. Its
    dynamic profiler counters were masked under that same boundary. It may
    therefore infer candidate economics at held runtime features without seeing
    their timing answer. Candidate selection uses only those predictions;
    measured regret appears for the first time in the returned decisions and
    remains the sole CV acceptance score.

    A successful teacher is not emitted at runtime. Its out-of-fold candidate
    choices flow through the existing cross-fitted publication stage, which
    distills them into the same bounded, total generic C++ tree used by every
    other tournament member.
    """

    results = []
    for fold_index, held_out_groups in enumerate(
        _domain_folds(domain, costs, seed=seed)
    ):
        held_out_costs = [
            cost
            for cost in costs
            if cost.shape_group_id in held_out_groups
        ]
        held_out_geometries = {
            (cost.runtime_key.aggregate_n, cost.runtime_key.k)
            for cost in held_out_costs
        }
        request_key = (
            profiler_training_pool_key,
            tuple(sorted(held_out_geometries)),
        )
        predictions = profiler_prediction_cache.get(request_key)
        if predictions is None:
            raise RuntimeError(
                "cross-fitted teacher prediction surface was not completed "
                f"before CV scoring: {request_key}"
            )

        matrix = _point_matrix(held_out_costs)
        decisions = []
        uncovered = 0
        for point, candidates in sorted(matrix.items()):
            predicted_candidates = []
            for candidate in candidates.values():
                prediction = predictions[(
                    candidate.runtime_key,
                    candidate.shape_group_id,
                    candidate.candidate_id,
                )]
                if prediction is None:
                    predicted_candidates = []
                    break
                predicted_candidates.append((prediction, candidate.candidate_id))
            if not predicted_candidates:
                uncovered += 1
                continue
            selected_id = min(predicted_candidates)[1]
            selected = candidates[selected_id]
            exact = min(
                candidates.values(),
                key=lambda candidate: (
                    candidate.max_surface_regret,
                    candidate.p95_surface_regret,
                    candidate.mean_surface_regret,
                    candidate.candidate_id,
                ),
            )
            decisions.append((selected, exact))

        decisions = tuple(decisions)
        complexities = tuple(
            _ComplexityFoldResult(
                complexity=complexity,
                decisions=decisions,
                uncovered=uncovered,
                required=len(matrix),
            )
            for complexity in range(1, max_leaves + 1)
        )
        results.append(_PlacementFoldResult(
            domain=domain,
            fold_index=fold_index,
            feature_policy=FeaturePolicy.CONTINUOUS,
            placement=BoundaryPlacement.MIDPOINT,
            profiler_influence=ProfilerInfluence.CROSS_FITTED_TEACHER,
            complexities=complexities,
        ))
    return tuple(results)


def cross_validate_domain_complexity(
    domain: GenericDomain,
    costs: list[CandidatePointCost],
    observations: Iterable[NativeVNNIObservation],
    *,
    max_leaves: int,
    min_shape_groups_per_leaf: int,
    seed: str,
) -> DomainCrossValidation | None:
    """Select a leaf budget without candidate-row or shape-group leakage."""

    # Preserve the public API used by callers that own both evidence views. CV
    # itself intentionally consumes only the sufficient candidate-cost matrix.
    _ = observations
    tasks, fold_count = _domain_fold_tasks(
        domain,
        costs,
        max_leaves=max_leaves,
        min_shape_groups_per_leaf=min_shape_groups_per_leaf,
        seed=seed,
    )
    if not tasks:
        return None
    results = tuple(_evaluate_placement_fold(task) for task in tasks)
    return _select_domain_cross_validation(
        domain,
        costs,
        fold_count,
        results,
        max_leaves=max_leaves,
    )


# Forked workers inherit these immutable task tuples copy-on-write. Passing the
# complete candidate matrix through every ProcessPool queue item made a single
# all-format CV pass serialization-bound and left most CPU cores idle.
_PARALLEL_CV_TASKS: tuple[tuple, ...] = ()
_PARALLEL_ACCELERATED_CV_TASK_GROUPS: tuple[
    tuple[tuple[int, tuple], ...], ...
] = ()
_PARALLEL_CV_REDUCTION_TASKS: tuple[tuple, ...] = ()
_PARALLEL_FINAL_TASKS: tuple[tuple, ...] = ()


def _evaluate_parallel_cv_index(index: int) -> _PlacementFoldResult:
    """Resolve one inherited CV task by index inside a forked worker."""

    return _evaluate_placement_fold(_PARALLEL_CV_TASKS[index])


def _reduce_parallel_cv_index(index: int) -> tuple[
    GenericDomain,
    tuple[DomainCrossValidation, ...],
    DomainCrossValidation | None,
]:
    """Reduce one inherited domain's immutable fold results.

    The accelerator produces independent fold-result populations for every
    policy domain. Ranking models, calculating regret statistics, collecting
    paired-planning edges, and hydrating the selected model therefore share no
    mutable state between domains. Workers inherit the large cost and fold
    matrices copy-on-write and receive only an integer index through the pool;
    this avoids serializing the complete corpus into every queue item.
    """

    (
        domain,
        costs,
        fold_count,
        fold_results,
        max_leaves,
        fit_final_rules,
    ) = _PARALLEL_CV_REDUCTION_TASKS[index]
    ranked = _rank_domain_cross_validations(
        domain,
        costs,
        fold_count,
        fold_results,
        max_leaves=max_leaves,
        retain_model_cells=not fit_final_rules,
    )
    selected = ranked[0] if ranked else None
    if fit_final_rules and selected is not None:
        selected = _hydrate_cross_validation_cells(selected, fold_results)
    return domain, ranked, selected


def _cv_reduction_worker_count(
    task_count: int,
    *,
    requested_workers: int | None = None,
) -> int:
    """Return the physical-core-capped domain-reduction worker count."""

    if task_count < 0:
        raise ValueError("CV reduction task count cannot be negative")
    if task_count == 0:
        return 0
    physical_cores = _physical_core_worker_count()
    if requested_workers is None:
        requested_workers = int(os.environ.get(
            "LLAMINAR_NATIVE_VNNI_POLICY_WORKERS",
            str(physical_cores),
        ))
    if requested_workers < 1:
        raise ValueError("policy worker count must be positive")
    return min(requested_workers, physical_cores, task_count)


def _reduce_domain_cross_validations(
    tasks: tuple[tuple, ...],
    *,
    requested_workers: int | None = None,
) -> tuple[
    tuple[
        tuple[
            GenericDomain,
            tuple[DomainCrossValidation, ...],
            DomainCrossValidation | None,
        ],
        ...,
    ],
    int,
]:
    """Reduce independent domains across physical cores deterministically.

    ``ProcessPoolExecutor.map`` returns results in task order even though the
    domains finish out of order. The caller constructs tasks in canonical
    domain order, so parallel execution cannot perturb generated policy bytes,
    paired-request ordering, or cache contents. Explicit worker requests are
    still capped at the affinity-visible physical-core count: SMT siblings add
    duplicate Python processes and resident model state without adding useful
    throughput to this allocation-heavy reduction.
    """

    global _PARALLEL_CV_REDUCTION_TASKS

    if not tasks:
        return (), 0
    worker_count = _cv_reduction_worker_count(
        len(tasks), requested_workers=requested_workers
    )

    _PARALLEL_CV_REDUCTION_TASKS = tasks
    try:
        if worker_count > 1 and len(tasks) >= 4:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                reduced = tuple(executor.map(
                    _reduce_parallel_cv_index,
                    range(len(tasks)),
                ))
        else:
            reduced = tuple(
                _reduce_parallel_cv_index(index)
                for index in range(len(tasks))
            )
        return reduced, worker_count
    finally:
        _PARALLEL_CV_REDUCTION_TASKS = ()


def _cross_fitted_publication_costs(
    costs: Iterable[CandidatePointCost],
    validation: DomainCrossValidation,
) -> tuple[tuple[CandidatePointCost, ...], dict[tuple[RuntimeKey, str], str]]:
    """Bind every development point to its independently selected OOF target.

    Grouped CV evaluates each shape exactly once while that shape is absent from
    the training fold.  Those decisions are the only non-leaking target labels
    available for a publication tree.  The former final fit discarded them and
    refitted directly against the all-development oracle, allowing a different
    in-sample partition to fail the leaf-p95 gate even when CV had selected a
    stable candidate surface.

    Target candidates retain their measured broad regret.  Every non-target
    candidate receives one domain-local penalty above the complete measured
    range, so tree search first reproduces OOF behavior while the untouched
    p95/max/mean fields continue to enforce measured economics.  This is model
    distillation, not synthetic timing evidence: the override exists only in
    memory and is deliberately absent from candidate-cost cache serialization.
    """

    costs = tuple(costs)
    target_by_point: dict[tuple[RuntimeKey, str], str] = {}
    observed_regret_by_point: dict[tuple[RuntimeKey, str], float] = {}
    for cell in validation.cells:
        point = (cell.runtime_key, cell.shape_group_id)
        if point in target_by_point:
            raise ValueError("cross-fitted publication target repeats one point")
        target_by_point[point] = cell.selected_candidate_id
        observed_regret_by_point[point] = cell.observed_broad_regret

    matrix = _point_matrix(list(costs))
    if set(target_by_point) != set(matrix):
        missing = set(matrix).difference(target_by_point)
        unexpected = set(target_by_point).difference(matrix)
        raise ValueError(
            "cross-fitted publication targets do not cover the measured matrix: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    if not costs:
        return (), target_by_point

    measured_maximum = max(cost.max_surface_regret for cost in costs)
    if not math.isfinite(measured_maximum):
        raise ValueError("cross-fitted publication costs must be finite")
    mismatch_penalty = max(1.0, measured_maximum + 1.0)
    publication_costs = []
    for cost in costs:
        point = (cost.runtime_key, cost.shape_group_id)
        target_candidate = target_by_point[point]
        if target_candidate not in matrix[point]:
            raise ValueError(
                "cross-fitted publication target is unavailable at its point"
            )
        is_target = cost.candidate_id == target_candidate
        if is_target and (
            cost.max_surface_regret != observed_regret_by_point[point]
        ):
            raise ValueError(
                "cross-fitted publication target changed measured broad regret"
            )
        publication_costs.append(replace(
            cost,
            cross_fitted_selection_regret=(
                cost.max_surface_regret if is_target else mismatch_penalty
            ),
        ))
    return tuple(publication_costs), target_by_point


def _fit_cross_fitted_publication_rules(
    domain: GenericDomain,
    costs: Iterable[CandidatePointCost],
    observations: Iterable[NativeVNNIObservation],
    validation: DomainCrossValidation,
    *,
    max_leaves: int,
    min_shape_groups_per_leaf: int,
    primary_scorer: NativeVNNILeafPrimaryScorer | None = None,
) -> tuple[tuple[GenericDispatchRule, ...], DomainCrossValidation]:
    """Distill one held-out decision surface into the final generic tree.

    The CV model's feature policy and leaf count describe how OOF labels were
    obtained; they need not be the most compact representation of those labels
    once all points are available.  Publication therefore runs the complete
    reviewed feature/placement tournament against the fixed OOF targets.  Tree
    search still gives measured leaf-p95 failures absolute priority, and the
    configured maximum leaf budget remains unchanged.
    """

    observations = tuple(observations)
    publication_costs, target_by_point = _cross_fitted_publication_costs(
        costs, validation
    )
    measured_matrix = _point_matrix(list(publication_costs))
    fitted = []
    for feature_policy in FEATURE_POLICIES:
        for placement in BOUNDARY_PLACEMENTS:
            fit = _fit_tree(
                list(publication_costs),
                max_leaves=max_leaves,
                min_shape_groups_per_leaf=min_shape_groups_per_leaf,
                boundary_placement=placement,
                feature_policy=feature_policy,
                primary_scorer=primary_scorer,
            )
            if fit is None:
                continue
            selected_by_point = {
                point: leaf.candidate_id
                for leaf in _tree_leaves(fit.root)
                for point in leaf.points
            }
            selected_costs = tuple(
                measured_matrix[point][selected_by_point[point]]
                for point in sorted(measured_matrix)
            )
            observed_p95 = _percentile(
                (cost.max_surface_regret for cost in selected_costs), 0.95
            )
            observed_mean = _ordered_mean(
                cost.max_surface_regret for cost in selected_costs
            )
            mismatch_count = sum(
                selected_by_point.get(point) != target_candidate
                for point, target_candidate in target_by_point.items()
            )
            fitted.append((
                fit.root.failed_leaf_count,
                observed_p95 >= GENERIC_REGRET_BUDGET,
                observed_p95,
                observed_mean,
                mismatch_count,
                *fit.objective[1:-1],
                FEATURE_POLICIES.index(feature_policy),
                BOUNDARY_PLACEMENTS.index(placement),
                fit.root.signature,
                feature_policy,
                placement,
                fit,
                selected_by_point,
            ))
    if not fitted:
        return (), replace(
            validation,
            publication_oof_p95_regret=math.inf,
            publication_oof_mean_regret=math.inf,
            publication_oof_mismatch_count=len(target_by_point),
        )

    selected = min(fitted)
    feature_policy, placement, fit, selected_by_point = selected[-4:]
    exemplar_by_candidate = {
        row.candidate_id: row for row in observations
    }
    rules = _flatten_rules(domain, fit.root, exemplar_by_candidate)
    mismatch_count = sum(
        selected_by_point.get(point) != target_candidate
        for point, target_candidate in target_by_point.items()
    )
    selected_costs = tuple(
        measured_matrix[point][selected_by_point[point]]
        for point in sorted(measured_matrix)
    )
    return rules, replace(
        validation,
        publication_feature_policy=feature_policy,
        publication_max_leaves=fit.leaf_count,
        publication_boundary_placement=placement,
        publication_oof_p95_regret=_percentile(
            (cost.max_surface_regret for cost in selected_costs), 0.95
        ),
        publication_oof_mean_regret=_ordered_mean(
            cost.max_surface_regret for cost in selected_costs
        ),
        publication_oof_mismatch_count=mismatch_count,
    )


def _fit_final_domain(
    args,
    primary_scorer: NativeVNNILeafPrimaryScorer | None = None,
):
    """Fit cross-fitted publication trees until one passes both p95 gates.

    Every frontier model already passed grouped held-out CV.  Its hydrated OOF
    decisions are distilled into a fresh final tree whose labels never came
    from an in-sample decision at the same shape.  A result is promotable only
    when the distilled decision population remains below five-percent p95 and
    every emitted measured leaf independently remains below the same bound.
    """

    (
        domain,
        costs,
        observations,
        validation_frontier,
        fold_results,
        min_shape_groups_per_leaf,
        max_leaves,
    ) = args
    if not validation_frontier:
        raise ValueError("final fit requires a non-empty promotable CV frontier")

    best_failure = None
    fitted_by_targets = {}
    for frontier_rank, validation in enumerate(validation_frontier):
        validation = _hydrate_cross_validation_cells(validation, fold_results)
        target_signature = tuple(
            (cell.runtime_key, cell.shape_group_id, cell.selected_candidate_id)
            for cell in validation.cells
        )
        cached = fitted_by_targets.get(target_signature)
        if cached is None:
            cached = _fit_cross_fitted_publication_rules(
                domain,
                costs,
                observations,
                validation,
                max_leaves=max_leaves,
                min_shape_groups_per_leaf=min_shape_groups_per_leaf,
                primary_scorer=primary_scorer,
            )
            fitted_by_targets[target_signature] = cached
        rules, fitted_validation = cached
        if fitted_validation is not validation:
            fitted_validation = replace(
                validation,
                publication_feature_policy=(
                    fitted_validation.publication_feature_policy
                ),
                publication_max_leaves=(
                    fitted_validation.publication_max_leaves
                ),
                publication_boundary_placement=(
                    fitted_validation.publication_boundary_placement
                ),
                publication_oof_p95_regret=(
                    fitted_validation.publication_oof_p95_regret
                ),
                publication_oof_mean_regret=(
                    fitted_validation.publication_oof_mean_regret
                ),
                publication_oof_mismatch_count=(
                    fitted_validation.publication_oof_mismatch_count
                ),
            )
        cross_fitted_p95 = (
            fitted_validation.publication_oof_p95_regret
            if fitted_validation.publication_oof_p95_regret is not None
            else math.inf
        )
        rules_are_promotable = bool(rules) and all(
            rule.development_p95_regret < GENERIC_REGRET_BUDGET
            for rule in rules
        )
        if cross_fitted_p95 < GENERIC_REGRET_BUDGET and rules_are_promotable:
            return domain, rules, fitted_validation

        worst_p95 = max(
            (rule.development_p95_regret for rule in rules),
            default=math.inf,
        )
        worst_max = max(
            (rule.development_max_regret for rule in rules),
            default=math.inf,
        )
        failure = (
            (
                cross_fitted_p95 >= GENERIC_REGRET_BUDGET,
                cross_fitted_p95,
                not bool(rules),
                worst_p95,
                worst_max,
                frontier_rank,
            ),
            rules,
            fitted_validation,
        )
        if best_failure is None or failure[0] < best_failure[0]:
            best_failure = failure

    if best_failure is None:
        raise RuntimeError("final fit exhausted its CV frontier without a result")
    return domain, best_failure[1], best_failure[2]


def _fit_parallel_final_index(index: int):
    """Resolve one inherited final-fit task by index inside a forked worker."""

    return _fit_final_domain(_PARALLEL_FINAL_TASKS[index])


@dataclass(frozen=True)
class AcceleratedTaskTiming:
    """One production accelerator task's isolated lane service interval.

    The interval begins immediately before Python materializes a geometry
    group's matrices and ends after all grouped influences have returned their
    exact heldout results. It therefore includes metadata preparation, graph
    capture or replay, kernel execution, the terminal stream wait, final D2H,
    and result materialization, but excludes scheduler queueing and pipe
    transfer. ``lane_sequence`` identifies first-use effects without treating
    them as steady-state throughput.
    """

    task_index: int
    accelerator_label: str
    lane_index: int
    lane_sequence: int
    elapsed_seconds: float


def _accelerated_worker(
    spec: PolicyAcceleratorSpec,
    task_kind: str,
    connection,
    expected_parent_pid: int,
) -> None:
    """Own one vendor context and dynamically fill its explicit stream lanes.

    CUDA and HIP remain isolated in backend-local processes. Inside one process,
    every lane owns a distinct scorer session, stream, captured graph cache, and
    persistent workspace. ``ctypes`` releases the GIL during DSO calls, so a
    thread pool can keep independent graph replays in flight while this control
    thread publishes completions and immediately refills the freed lane.
    """

    arm_policy_worker_parent_death_signal(expected_parent_pid)
    scorers = []
    executor = None
    try:
        lane_count = getattr(spec, "lane_count", 1)
        if lane_count <= 0:
            raise ValueError("policy accelerator lane count must be positive")
        scorers = [spec.create_scorer() for _lane in range(lane_count)]
        executor = ThreadPoolExecutor(
            max_workers=lane_count,
            thread_name_prefix=f"native-vnni-{spec.label}",
        )

        def evaluate_task(task_index, scorer):
            """Execute one scheduler item on its lane-owned scorer session."""

            started = time.perf_counter()
            if task_kind == "cv":
                result = tuple(
                    (
                        original_index,
                        _evaluate_placement_fold(
                            task,
                            primary_scorer=scorer,
                        ),
                    )
                    for original_index, task
                    in _PARALLEL_ACCELERATED_CV_TASK_GROUPS[task_index]
                )
            elif task_kind == "final":
                result = _fit_final_domain(
                    _PARALLEL_FINAL_TASKS[task_index],
                    primary_scorer=scorer,
                )
            else:
                raise RuntimeError(
                    f"unknown accelerated policy task kind {task_kind!r}"
                )
            return result, time.perf_counter() - started

        futures = {}
        lane_sequences = [0 for _lane in range(lane_count)]

        def receive_lane_work(lane_index):
            """Receive exactly one replacement command for a free lane."""

            message = connection.recv()
            if message[0] == "stop":
                return
            if message[0] != "task" or len(message) != 2:
                raise RuntimeError(
                    f"policy worker received invalid command {message!r}"
                )
            task_index = message[1]
            future = executor.submit(
                evaluate_task, task_index, scorers[lane_index]
            )
            futures[future] = (lane_index, task_index)

        connection.send(("ready", lane_count))
        for lane_index in range(lane_count):
            receive_lane_work(lane_index)
        while futures:
            completed, _pending = wait_for_futures(
                tuple(futures), return_when=FIRST_COMPLETED
            )
            for future in completed:
                lane_index, task_index = futures.pop(future)
                result, elapsed_seconds = future.result()
                connection.send((
                    "result",
                    task_index,
                    result,
                    lane_index,
                    lane_sequences[lane_index],
                    elapsed_seconds,
                ))
                lane_sequences[lane_index] += 1
                receive_lane_work(lane_index)
        connection.send(("done",))
    except BaseException:
        try:
            connection.send(("error", spec.label, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for scorer in scorers:
            close = getattr(scorer, "close", None)
            if close is not None:
                close()
        connection.close()


def _run_accelerated_tasks(
    specs: tuple[PolicyAcceleratorSpec, ...],
    task_weights: tuple[int, ...],
    task_kind: str,
    task_timings: list[AcceleratedTaskTiming] | None = None,
) -> list:
    """Dynamically schedule weighted tasks and restore exact task-index order.

    A worker process receives only one accelerator specification for its whole
    lifetime.  This is stricter than a generic process pool: no scheduling
    accident can load CUDA and HIP DSOs into the same process.  Parent/child
    pipes stream each completed task independently. Longest estimated tasks are
    offered first, and every free lane pulls the next task, eliminating the
    severe static-partition tail observed in 32-leaf capacity fitting.
    """

    if not specs:
        raise ValueError("accelerated policy scheduling requires a device spec")
    if task_kind not in ("cv", "final"):
        raise ValueError("accelerated policy task kind must be cv or final")
    if not task_weights:
        return []
    timing_enabled = os.environ.get(
        "LLAMINAR_NATIVE_VNNI_POLICY_TIMING", "0"
    ) == "1"
    scheduling_started = time.perf_counter()
    last_progress_report = scheduling_started
    completed_weight = 0
    total_weight = sum(task_weights)
    active_worker_count = min(len(specs), len(task_weights))
    pending = deque(sorted(
        range(len(task_weights)),
        key=lambda index: (-task_weights[index], index),
    ))
    context = multiprocessing.get_context("fork")
    coordinator_pid = os.getpid()
    workers = []
    connections = {}
    for worker_index in range(active_worker_count):
        parent_connection, worker_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=_accelerated_worker,
            args=(
                specs[worker_index],
                task_kind,
                worker_connection,
                coordinator_pid,
            ),
            name=(
                f"native-vnni-policy-{specs[worker_index].label}"
                f"-lane-{worker_index}"
            ),
        )
        process.start()
        worker_connection.close()
        workers.append(process)
        connections[parent_connection] = {
            "process": process,
            "spec": specs[worker_index],
            "task_indices": set(),
            "lane_count": 0,
        }

    results = {}
    try:
        while connections:
            ready = wait_for_connections(tuple(connections), timeout=1.0)
            if not ready:
                failed = [
                    state
                    for state in connections.values()
                    if state["process"].exitcode is not None
                ]
                if failed:
                    state = failed[0]
                    raise RuntimeError(
                        f"policy accelerator worker {state['spec'].label} "
                        f"exited with status {state['process'].exitcode} before "
                        "publishing completion"
                    )
                continue
            for connection in ready:
                state = connections[connection]
                try:
                    message = connection.recv()
                except EOFError as error:
                    raise RuntimeError(
                        f"policy accelerator worker {state['spec'].label} closed its "
                        "result pipe before completion"
                    ) from error
                if message[0] == "ready":
                    if state["task_indices"]:
                        raise RuntimeError("busy policy worker published ready")
                    lane_count = message[1] if len(message) == 2 else 1
                    if lane_count <= 0:
                        raise RuntimeError(
                            "policy worker published a non-positive lane count"
                        )
                    state["lane_count"] = lane_count
                    for _lane in range(lane_count):
                        if pending:
                            task_index = pending.popleft()
                            state["task_indices"].add(task_index)
                            connection.send(("task", task_index))
                        else:
                            connection.send(("stop",))
                elif message[0] == "result":
                    if len(message) == 6:
                        (
                            _kind,
                            task_index,
                            result,
                            lane_index,
                            lane_sequence,
                            elapsed_seconds,
                        ) = message
                    elif len(message) == 3:
                        # The compact form is retained for scheduler-only unit
                        # probes that intentionally do not own a GPU runtime.
                        _kind, task_index, result = message
                        lane_index = 0
                        lane_sequence = 0
                        elapsed_seconds = 0.0
                    else:
                        raise RuntimeError(
                            "policy worker published a malformed result"
                        )
                    if task_index not in state["task_indices"]:
                        raise RuntimeError(
                            "policy worker published a result for another task"
                        )
                    if task_index in results:
                        raise RuntimeError(
                            f"policy task {task_index} was published twice"
                        )
                    results[task_index] = result
                    if task_timings is not None and len(message) == 6:
                        task_timings.append(AcceleratedTaskTiming(
                            task_index=task_index,
                            accelerator_label=state["spec"].label,
                            lane_index=lane_index,
                            lane_sequence=lane_sequence,
                            elapsed_seconds=elapsed_seconds,
                        ))
                    completed_weight += task_weights[task_index]
                    now = time.perf_counter()
                    if timing_enabled and (
                        len(results) == len(task_weights)
                        or now - last_progress_report >= 30.0
                    ):
                        elapsed = now - scheduling_started
                        weighted_fraction = (
                            completed_weight / total_weight
                            if total_weight > 0 else 1.0
                        )
                        weighted_rate = (
                            completed_weight / elapsed if elapsed > 0.0 else 0.0
                        )
                        remaining_weight = total_weight - completed_weight
                        eta = (
                            remaining_weight / weighted_rate
                            if weighted_rate > 0.0 else math.inf
                        )
                        print(
                            "NativeVNNI accelerator progress "
                            f"phase={task_kind} "
                            f"tasks={len(results)}/{len(task_weights)} "
                            f"weighted={weighted_fraction:.2%} "
                            f"elapsed={elapsed:.1f}s eta={eta:.1f}s "
                            f"workers={active_worker_count}",
                            flush=True,
                        )
                        last_progress_report = now
                    state["task_indices"].remove(task_index)
                    if pending:
                        next_index = pending.popleft()
                        state["task_indices"].add(next_index)
                        connection.send(("task", next_index))
                    else:
                        connection.send(("stop",))
                elif message[0] == "done":
                    if state["task_indices"]:
                        raise RuntimeError("busy policy worker stopped early")
                    connection.close()
                    del connections[connection]
                elif message[0] == "error":
                    _kind, label, detail = message
                    raise RuntimeError(
                        f"policy accelerator worker {label} failed:\n{detail}"
                    )
                else:
                    raise RuntimeError(
                        f"policy accelerator worker {state['spec'].label} published an "
                        f"unknown message {message[0]!r}"
                    )
    except BaseException:
        for process in workers:
            if process.is_alive():
                process.terminate()
        raise
    finally:
        for connection in tuple(connections):
            connection.close()
        for process in workers:
            process.join()

    missing = sorted(set(range(len(task_weights))) - set(results))
    if missing:
        raise RuntimeError(
            f"policy accelerator workers omitted task indices {missing}"
        )
    return [results[index] for index in range(len(task_weights))]


def _profiler_transfer_key(domain: GenericDomain) -> tuple[object, ...]:
    """Return the cross-M, cross-format economy-learning domain.

    Tensor format identity is intentionally excluded: codebook, prepared
    family, and packing ABI remain explicit model-record features so the
    surrogate can learn their effect while sharing evidence. Prefill M is also
    intentionally excluded because it is a first-class numeric model feature:
    the surrogate must learn how one physical candidate scales across work
    sizes instead of fitting an unrelated forest for each sampled row count.
    Architecture, arithmetic contract, operation, bundle, and execution mode
    stay fixed because crossing those boundaries would mix production regimes.
    """

    return (
        domain.backend,
        domain.architecture_class,
        domain.semantic_contract,
        domain.operation_kind,
        domain.bundle_signature,
        domain.execution_mode,
    )


ProfilerPredictionCacheKey = tuple[
    tuple[object, ...],
    tuple[tuple[int, int], ...],
]
ProfilerPredictionPoint = tuple[RuntimeKey, str, str]
ProfilerPredictionCache = dict[
    ProfilerPredictionCacheKey,
    dict[ProfilerPredictionPoint, float | None],
]
ProfilerPredictionPoolResult = tuple[
    tuple[
        ProfilerPredictionCacheKey,
        dict[ProfilerPredictionPoint, float | None],
    ], ...
]
ProfilerPredictionPointRequests = Mapping[
    ProfilerPredictionCacheKey,
    frozenset[ProfilerPredictionPoint],
]

_PARALLEL_PROFILER_POOL_TASKS: tuple[
    tuple[tuple[object, ...], tuple[ProfilerPredictionCacheKey, ...]], ...
] = ()
_PARALLEL_PROFILER_TRAINING_POOLS: Mapping[
    tuple[object, ...], tuple[CandidatePointCost, ...]
] = {}
_PARALLEL_PROFILER_CORPUS: ObservationCorpus | None = None
_PARALLEL_PROFILER_CATALOG: ProfilerFeatureCatalog | None = None
_PARALLEL_PROFILER_OBSERVATION_INDEX: ProfilerObservationIndex | None = None
_PARALLEL_PROFILER_MODEL_RECORD_INDICES: Mapping[
    tuple[object, ...], Mapping[
        tuple[RuntimeKey, str, str], dict[str, float | str]
    ]
] = {}
_PARALLEL_PROFILER_PREDICTION_POINTS: ProfilerPredictionPointRequests = {}


def _domain_profiler_prediction_request_keys(
    domain: GenericDomain,
    costs: list[CandidatePointCost],
    *,
    seed: str,
) -> tuple[ProfilerPredictionCacheKey, ...]:
    """Enumerate the leakage-controlled profiler surfaces needed by one CV.

    The key carries the cross-M transfer identity and complete held-out N/K
    geometry set. Consequently, M and format domains that use the same fold
    reuse one model prediction surface without allowing another row count or
    format's copy of held-out geometry to enter model training.
    """

    pool_key = _profiler_transfer_key(domain)
    requests = []
    for held_out_groups in _domain_folds(domain, costs, seed=seed):
        held_out_geometries = {
            (cost.runtime_key.aggregate_n, cost.runtime_key.k)
            for cost in costs
            if cost.shape_group_id in held_out_groups
        }
        requests.append((pool_key, tuple(sorted(held_out_geometries))))
    return tuple(requests)


def _domain_profiler_prediction_requests(
    domain: GenericDomain,
    costs: list[CandidatePointCost],
    *,
    seed: str,
) -> dict[ProfilerPredictionCacheKey, frozenset[ProfilerPredictionPoint]]:
    """Return each fold surface and the exact predictions CV will consume.

    A fold needs predictions for its domain-local training rows and for the
    held rows scored by the cross-fitted teacher. The surrogate still trains
    only on the leakage-controlled cross-M pool with every held N/K removed;
    asking it to infer held features does not expose held timing labels. Keeping
    the output inventory domain-local avoids predicting and persisting every
    unrelated format and M row in the shared transfer pool. Multiple domains
    may request the same held-geometry surface, in which case the caller unions
    their point inventories before fitting it once.
    """

    pool_key = _profiler_transfer_key(domain)
    requests: dict[
        ProfilerPredictionCacheKey, set[ProfilerPredictionPoint]
    ] = defaultdict(set)
    for held_out_groups in _domain_folds(domain, costs, seed=seed):
        held_out_geometries = {
            (cost.runtime_key.aggregate_n, cost.runtime_key.k)
            for cost in costs
            if cost.shape_group_id in held_out_groups
        }
        request_key = (pool_key, tuple(sorted(held_out_geometries)))
        requests[request_key].update(
            (
                cost.runtime_key,
                cost.shape_group_id,
                cost.candidate_id,
            )
            for cost in costs
        )
    return {
        key: frozenset(points)
        for key, points in requests.items()
    }


def _fit_profiler_prediction_pool(
    pool_key: tuple[object, ...],
    request_keys: tuple[ProfilerPredictionCacheKey, ...],
    training_pool: tuple[CandidatePointCost, ...],
    corpus: ObservationCorpus,
    catalog: ProfilerFeatureCatalog,
    observation_index: ProfilerObservationIndex,
    model_record_index: Mapping[
        tuple[RuntimeKey, str, str], dict[str, float | str]
    ] | None = None,
    prediction_points_by_request: ProfilerPredictionPointRequests | None = None,
) -> ProfilerPredictionPoolResult:
    """Fit requested forests from one immutable pool feature index."""

    if model_record_index is None:
        model_record_index = build_profiler_model_record_index(
            training_pool,
            corpus,
            catalog,
            observation_index=observation_index,
        )
    fitted = []
    for request_key in request_keys:
        request_pool_key, held_out_geometries = request_key
        if request_pool_key != pool_key:
            raise ValueError(
                "profiler prediction request belongs to a different transfer "
                f"pool: expected {pool_key}, received {request_pool_key}"
            )
        held_out_geometry_set = set(held_out_geometries)
        model_training_costs = [
            cost
            for cost in training_pool
            if (
                cost.runtime_key.aggregate_n,
                cost.runtime_key.k,
            ) not in held_out_geometry_set
        ]
        requested_points = (
            frozenset(
                (
                    cost.runtime_key,
                    cost.shape_group_id,
                    cost.candidate_id,
                )
                for cost in training_pool
            )
            if prediction_points_by_request is None
            else prediction_points_by_request[request_key]
        )
        prediction_costs = [
            cost
            for cost in training_pool
            if (
                cost.runtime_key,
                cost.shape_group_id,
                cost.candidate_id,
            ) in requested_points
        ]
        observed_points = {
            (
                cost.runtime_key,
                cost.shape_group_id,
                cost.candidate_id,
            )
            for cost in prediction_costs
        }
        if observed_points != requested_points:
            raise ValueError(
                "profiler prediction request contains points outside its "
                f"training pool: missing={len(requested_points - observed_points)}"
            )
        informed_pool = apply_profiler_regret_predictions(
            prediction_costs,
            corpus,
            catalog,
            observation_index=observation_index,
            model_training_costs=model_training_costs,
            model_record_index=model_record_index,
            excluded_profiler_geometries=frozenset(
                held_out_geometry_set
            ),
        )
        fitted.append((
            request_key,
            {
                (
                    cost.runtime_key,
                    cost.shape_group_id,
                    cost.candidate_id,
                ): cost.profiler_predicted_regret
                for cost in informed_pool
            },
        ))
    return tuple(fitted)


def _fit_parallel_profiler_pool_index(
    index: int,
) -> ProfilerPredictionPoolResult:
    """Fit one global pool task inside a forked copy-on-write worker."""

    if (
        _PARALLEL_PROFILER_CORPUS is None
        or _PARALLEL_PROFILER_CATALOG is None
        or _PARALLEL_PROFILER_OBSERVATION_INDEX is None
    ):
        raise RuntimeError("parallel profiler model context is not initialized")
    pool_key, request_keys = _PARALLEL_PROFILER_POOL_TASKS[index]
    return _fit_profiler_prediction_pool(
        pool_key,
        request_keys,
        _PARALLEL_PROFILER_TRAINING_POOLS[pool_key],
        _PARALLEL_PROFILER_CORPUS,
        _PARALLEL_PROFILER_CATALOG,
        _PARALLEL_PROFILER_OBSERVATION_INDEX,
        _PARALLEL_PROFILER_MODEL_RECORD_INDICES[pool_key],
        _PARALLEL_PROFILER_PREDICTION_POINTS,
    )


def _populate_profiler_prediction_cache(
    prediction_cache: ProfilerPredictionCache,
    request_keys: Iterable[ProfilerPredictionCacheKey],
    training_pools: Mapping[
        tuple[object, ...], tuple[CandidatePointCost, ...]
    ],
    corpus: ObservationCorpus,
    catalog: ProfilerFeatureCatalog,
    observation_index: ProfilerObservationIndex,
    *,
    requested_workers: int | None = None,
    prediction_points_by_request: ProfilerPredictionPointRequests | None = None,
    fit_cache: PolicyFitCache | None = None,
    training_pool_digests: Mapping[tuple[object, ...], str] | None = None,
    profiler_model_digests: Mapping[tuple[object, ...], str] | None = None,
) -> int:
    """Fit every missing profiler surface concurrently across host cores.

    The parent materializes each cross-M pool's candidate descriptors and
    normalized profiler interactions once. Forked workers inherit those
    immutable indices copy-on-write and independently fit held-out surfaces,
    allowing even one large pool to occupy the complete host. Every individual
    sklearn model is single-threaded, preventing nested oversubscription while
    surface-level work occupies all host cores. Fixed random seeds make
    completion order irrelevant to output.

    Returns the number of model workers used, for fit diagnostics.
    """

    global _PARALLEL_PROFILER_POOL_TASKS
    global _PARALLEL_PROFILER_TRAINING_POOLS
    global _PARALLEL_PROFILER_CORPUS
    global _PARALLEL_PROFILER_CATALOG
    global _PARALLEL_PROFILER_OBSERVATION_INDEX
    global _PARALLEL_PROFILER_MODEL_RECORD_INDICES
    global _PARALLEL_PROFILER_PREDICTION_POINTS

    request_keys = tuple(dict.fromkeys(request_keys))
    if prediction_points_by_request is None:
        prediction_points_by_request = {
            request_key: frozenset(
                (
                    cost.runtime_key,
                    cost.shape_group_id,
                    cost.candidate_id,
                )
                for cost in training_pools[request_key[0]]
            )
            for request_key in request_keys
        }
    else:
        missing_point_inventories = (
            set(request_keys) - set(prediction_points_by_request)
        )
        if missing_point_inventories:
            raise ValueError(
                "profiler prediction requests omit point inventories for "
                f"{len(missing_point_inventories)} surfaces"
            )

    persistent_keys: dict[ProfilerPredictionCacheKey, str] = {}
    if fit_cache is not None:
        if training_pool_digests is None or profiler_model_digests is None:
            raise ValueError(
                "persistent profiler predictions require pool and model digests"
            )
        for request_key in request_keys:
            pool_key, held_out_geometries = request_key
            persistent_key = fit_cache.profiler_prediction_key(
                training_pool_digest=training_pool_digests[pool_key],
                profiler_model_digest=profiler_model_digests[pool_key],
                held_out_geometries=held_out_geometries,
                prediction_points=prediction_points_by_request[request_key],
            )
            persistent_keys[request_key] = persistent_key
            if request_key in prediction_cache:
                continue
            cached_predictions = fit_cache.load_profiler_predictions(
                persistent_key,
                training_pool_digest=training_pool_digests[pool_key],
                profiler_model_digest=profiler_model_digests[pool_key],
                held_out_geometries=held_out_geometries,
                prediction_points=prediction_points_by_request[request_key],
            )
            if cached_predictions is not None:
                prediction_cache[request_key] = cached_predictions

    missing_requests = tuple(dict.fromkeys(
        key for key in request_keys if key not in prediction_cache
    ))
    if not missing_requests:
        return 0
    requests_by_pool: dict[
        tuple[object, ...], list[ProfilerPredictionCacheKey]
    ] = defaultdict(list)
    for request_key in missing_requests:
        requests_by_pool[request_key[0]].append(request_key)
    model_record_indices = {
        pool_key: build_profiler_model_record_index(
            training_pools[pool_key],
            corpus,
            catalog,
            observation_index=observation_index,
        )
        for pool_key in requests_by_pool
    }
    # A forest surface is independently deterministic once the shared feature
    # index and held-out geometry set are fixed. Keep one request per task so a
    # cross-M pool cannot collapse a two-socket fit to one active process.
    pool_tasks = tuple(
        (pool_key, (request_key,))
        for pool_key, pool_requests in requests_by_pool.items()
        for request_key in pool_requests
    )

    if requested_workers is None:
        requested_workers = int(os.environ.get(
            "LLAMINAR_NATIVE_VNNI_PROFILER_MODEL_WORKERS",
            str(_physical_core_worker_count()),
        ))
    if requested_workers < 1:
        raise ValueError("profiler model worker count must be positive")
    worker_count = min(requested_workers, len(pool_tasks))
    executor = None
    try:
        if worker_count == 1 or len(pool_tasks) < 4:
            fitted_pools = (
                _fit_profiler_prediction_pool(
                    pool_key,
                    pool_requests,
                    training_pools[pool_key],
                    corpus,
                    catalog,
                    observation_index,
                    model_record_indices[pool_key],
                    prediction_points_by_request,
                )
                for pool_key, pool_requests in pool_tasks
            )
        else:
            _PARALLEL_PROFILER_POOL_TASKS = pool_tasks
            _PARALLEL_PROFILER_TRAINING_POOLS = training_pools
            _PARALLEL_PROFILER_CORPUS = corpus
            _PARALLEL_PROFILER_CATALOG = catalog
            _PARALLEL_PROFILER_OBSERVATION_INDEX = observation_index
            _PARALLEL_PROFILER_MODEL_RECORD_INDICES = model_record_indices
            _PARALLEL_PROFILER_PREDICTION_POINTS = (
                prediction_points_by_request
            )
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("fork"),
            )
            futures = tuple(
                executor.submit(_fit_parallel_profiler_pool_index, index)
                for index in range(len(pool_tasks))
            )
            # Publication order has no semantic meaning because every result
            # carries its content-addressed request key. Consume completion
            # order so a slow early forest cannot strand hundreds of finished
            # surfaces in executor buffers if the fit is interrupted.
            fitted_pools = (
                future.result() for future in as_completed(futures)
            )
        for fitted_pool in fitted_pools:
            for request_key, predictions in fitted_pool:
                prediction_cache[request_key] = predictions
                if fit_cache is not None:
                    pool_key, held_out_geometries = request_key
                    fit_cache.store_profiler_predictions(
                        persistent_keys[request_key],
                        training_pool_digest=training_pool_digests[pool_key],
                        profiler_model_digest=profiler_model_digests[pool_key],
                        held_out_geometries=held_out_geometries,
                        predictions=predictions,
                    )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        _PARALLEL_PROFILER_POOL_TASKS = ()
        _PARALLEL_PROFILER_TRAINING_POOLS = {}
        _PARALLEL_PROFILER_CORPUS = None
        _PARALLEL_PROFILER_CATALOG = None
        _PARALLEL_PROFILER_OBSERVATION_INDEX = None
        _PARALLEL_PROFILER_MODEL_RECORD_INDICES = {}
        _PARALLEL_PROFILER_PREDICTION_POINTS = {}
    return worker_count


def fit_generic_policy(
    development: ObservationCorpus,
    *,
    serial_m1_hashes: Mapping[RuntimeKey, str] | None = None,
    paired_comparisons: Mapping[
        PairedCellKey, tuple[PairedTimingComparison, ...]
    ] | None = None,
    max_leaves: int = DEFAULT_TREE_LEAVES,
    min_shape_groups_per_leaf: int = 2,
    cross_validation_seed: str = (
        "native-vnni-development-cv-v9-cross-fitted-publication"
    ),
    fit_final_rules: bool = True,
    policy_accelerators: tuple[PolicyAcceleratorSpec, ...] | None = None,
    fit_cache: PolicyFitCache | None = None,
    domain_corpus_provider: (
        Callable[[GenericDomain], ObservationCorpus] | None
    ) = None,
    domain_corpus_digest_provider: (
        Callable[[GenericDomain], str] | None
    ) = None,
    profiler_feature_catalog: ProfilerFeatureCatalog | None = None,
) -> GenericPolicy:
    """Cross-validate leaf complexity, then optionally fit each final tree once.

    Paired-request planning needs every held-out decision but does not consume a
    final dispatch tree. Skipping the redundant full-development fit keeps each
    refinement iteration focused on the CV edges that can still change. Policy
    compilation leaves this option enabled and therefore preserves the normal
    frozen-rule contract.  ``policy_accelerators=None`` reads the canonical
    accelerator environment; an explicit empty tuple selects CPU fitting.  A
    non-empty accelerator inventory is strict and never falls back to CPU.
    When ``fit_cache`` is supplied, candidate costs and CV results are reused
    independently per policy domain; a new paired edge therefore retrains only
    the domain whose tournament changed. ``domain_corpus_provider`` may lazily
    add shape-resolved candidate rows for only a cache-miss domain; it must
    preserve that domain's complete identity. A cached lazy projection must
    also provide ``domain_corpus_digest_provider`` so cache lookup never trusts
    a projection whose inputs cannot be identified without materializing it.
    """

    global _PARALLEL_CV_TASKS
    global _PARALLEL_ACCELERATED_CV_TASK_GROUPS
    global _PARALLEL_FINAL_TASKS

    accelerator_specs = (
        policy_accelerator_specs_from_environment()
        if policy_accelerators is None
        else policy_accelerators
    )
    accelerator_workers = policy_accelerator_worker_specs(accelerator_specs)
    if (
        fit_cache is not None
        and domain_corpus_provider is not None
        and domain_corpus_digest_provider is None
    ):
        raise ValueError(
            "cached domain corpus projection requires a domain digest provider"
        )

    timing_enabled = os.environ.get(
        "LLAMINAR_NATIVE_VNNI_POLICY_TIMING", "0"
    ) == "1"
    fit_started = time.perf_counter()
    domains = development.generic_domains()
    domain_corpora: dict[GenericDomain, ObservationCorpus] = {}
    shared_profiler_observation_index = (
        build_profiler_observation_index(development)
        if profiler_feature_catalog is not None
        and domain_corpus_provider is None
        else None
    )

    def corpus_for_domain(domain: GenericDomain) -> ObservationCorpus:
        """Materialize one optional domain projection at most once per fit."""

        if domain_corpus_provider is None:
            return development
        domain_corpus = domain_corpora.get(domain)
        if domain_corpus is None:
            domain_corpus = domain_corpus_provider(domain)
            if domain_corpus.generic_domains() != (domain,):
                raise ValueError(
                    "domain corpus provider changed or mixed policy domains"
                )
            domain_corpora[domain] = domain_corpus
        return domain_corpus

    # Four cache generations can coexist while a long-running corpus is
    # refined. New keys use a semantic domain digest and domain-local serial
    # hashes. Read-through candidates retain both the old aggregate-coupled
    # corpus digest and the older process-global serial inventory, allowing an
    # already-paid fit to migrate without another search.
    global_serial_m1_hash_digest = _serial_m1_hash_digest(serial_m1_hashes)
    cost_keys = {}
    aggregate_local_cost_keys = {}
    semantic_global_cost_keys = {}
    aggregate_global_cost_keys = {}
    legacy_cost_key_candidates = {}
    validation_keys = {}
    validations: dict[GenericDomain, DomainCrossValidation | None] = {}
    validation_frontiers: dict[
        GenericDomain, tuple[DomainCrossValidation, ...]
    ] = {}
    cached_final_rules: dict[
        GenericDomain, tuple[GenericDispatchRule, ...]
    ] = {}
    validation_cache_hits = 0
    cache_identity_inputs = {
        domain: (
            paired_domain_digest,
            semantic_domain_corpus_digest,
            aggregate_domain_corpus_digest,
            local_serial_m1_hash_digest,
        )
        for (
            domain,
            paired_domain_digest,
            semantic_domain_corpus_digest,
            aggregate_domain_corpus_digest,
            local_serial_m1_hash_digest,
        ) in (
            _domain_cache_identity_inputs(
                development,
                domains,
                paired_comparisons,
                serial_m1_hashes,
            )
            if fit_cache is not None and domain_corpus_provider is None
            else ()
        )
    }
    cache_identity_complete = time.perf_counter()
    for domain in domains:
        if fit_cache is not None:
            if domain_corpus_provider is None:
                (
                    paired_domain_digest,
                    semantic_domain_corpus_digest,
                    aggregate_domain_corpus_digest,
                    local_serial_m1_hash_digest,
                ) = cache_identity_inputs[domain]
            else:
                domain_rows = development.rows_for_generic_domain(domain)
                paired_domain_digest = _paired_domain_digest(
                    domain_rows,
                    paired_comparisons,
                )
                if domain_corpus_digest_provider is None:
                    raise RuntimeError(
                        "domain corpus provider lost its digest provider"
                    )
                semantic_domain_corpus_digest = (
                    domain_corpus_digest_provider(domain)
                )
                domain_runtime_keys = {
                    development.runtime_key_for(row) for row in domain_rows
                }
                local_serial_m1_hash_digest = _serial_m1_hash_digest(
                    serial_m1_hashes,
                    domain_runtime_keys,
                )
            cost_key = fit_cache.cost_key(
                domain,
                semantic_domain_corpus_digest,
                paired_domain_digest,
                local_serial_m1_hash_digest,
            )
            cost_keys[domain] = cost_key
            semantic_global_cost_keys[domain] = fit_cache.cost_key(
                domain,
                semantic_domain_corpus_digest,
                paired_domain_digest,
                global_serial_m1_hash_digest,
            )
            if domain_corpus_digest_provider is None:
                aggregate_local_cost_keys[domain] = fit_cache.cost_key(
                    domain,
                    aggregate_domain_corpus_digest,
                    paired_domain_digest,
                    local_serial_m1_hash_digest,
                )
                aggregate_global_cost_keys[domain] = fit_cache.cost_key(
                    domain,
                    aggregate_domain_corpus_digest,
                    paired_domain_digest,
                    global_serial_m1_hash_digest,
                )
            legacy_cost_key_candidates[domain] = tuple(dict.fromkeys(
                key
                for key in (
                    aggregate_local_cost_keys.get(domain),
                    semantic_global_cost_keys[domain],
                    aggregate_global_cost_keys.get(domain),
                )
                if key is not None and key != cost_key
            ))
    cost_keys_complete = time.perf_counter()

    def profiler_pool_digests_for(
        keys: Mapping[GenericDomain, str],
    ) -> dict[GenericDomain, str]:
        """Hash cross-M transfer pools for one candidate-cost key generation."""

        if fit_cache is None or profiler_feature_catalog is None:
            return {}
        pool_domains: dict[tuple[object, ...], list[GenericDomain]] = (
            defaultdict(list)
        )
        for pool_domain in domains:
            if pool_domain in keys:
                pool_domains[_profiler_transfer_key(pool_domain)].append(
                    pool_domain
                )
        result = {}
        for grouped_domains in pool_domains.values():
            digest = "sha256:" + _content_key({
                "domain_cost_keys": [
                    {
                        "domain": _generic_domain_mapping(pool_domain),
                        "cost_key": keys[pool_domain],
                    }
                    for pool_domain in sorted(grouped_domains)
                ],
            })
            for pool_domain in grouped_domains:
                result[pool_domain] = digest
        return result

    profiler_pool_digests = profiler_pool_digests_for(cost_keys)
    aggregate_local_pool_digests = profiler_pool_digests_for(
        aggregate_local_cost_keys
    )
    profiler_model_digests = {}
    if profiler_feature_catalog is not None:
        if domain_corpus_provider is None:
            profiler_pool_rows = defaultdict(list)
            for pool_domain in domains:
                for row in development.rows_for_generic_domain(pool_domain):
                    runtime_key = development.runtime_key_for(row)
                    if (
                        row.generic_eligible
                        and candidate_is_eligible(
                            row,
                            (serial_m1_hashes or {}).get(runtime_key),
                        )
                    ):
                        profiler_pool_rows[
                            _profiler_transfer_key(pool_domain)
                        ].append(row)
            profiler_model_digests = {
                pool_key: profiler_feature_catalog.model_digest_for(rows)
                for pool_key, rows in profiler_pool_rows.items()
            }
        else:
            # Lazy formula projection deliberately avoids materializing a
            # cache-hit domain. Its exact projected corpus digest still guards
            # costs; use the complete model-visible descriptor table until the
            # projection layer can provide a descriptor-subset digest directly.
            profiler_model_digests = {
                _profiler_transfer_key(pool_domain): (
                    profiler_feature_catalog.model_digest
                )
                for pool_domain in domains
            }
    profiler_identities_complete = time.perf_counter()

    if fit_cache is not None:
        # Full profiler provenance is immutable for the fit.  Legacy cache-key
        # migration consults it for every domain, but canonical serialization
        # belongs outside that loop even though ProfilerFeatureCatalog also
        # memoizes the value defensively.
        profiler_catalog_digest = (
            profiler_feature_catalog.digest
            if profiler_feature_catalog is not None
            else None
        )
        for domain in domains:
            cost_key = cost_keys[domain]
            profiler_model_digest = (
                profiler_model_digests[_profiler_transfer_key(domain)]
                if profiler_feature_catalog is not None
                else None
            )
            validation_key = fit_cache.validation_key(
                domain,
                cost_key,
                max_leaves=max_leaves,
                min_shape_groups_per_leaf=min_shape_groups_per_leaf,
                cross_validation_seed=cross_validation_seed,
                profiler_feature_catalog_digest=(
                    profiler_model_digest
                ),
                fit_final_rules=fit_final_rules,
                profiler_training_pool_digest=profiler_pool_digests.get(
                    domain
                ),
            )
            validation_keys[domain] = validation_key
            hit, cached_validation, cached_rules = (
                fit_cache.load_validation_entry(
                    validation_key,
                    domain,
                )
            )
            legacy_profiler_catalog_digest = (
                profiler_catalog_digest
                if profiler_catalog_digest is not None
                and profiler_catalog_digest != profiler_model_digest
                else None
            )
            catalog_digest_candidates = (
                tuple(dict.fromkeys((
                    profiler_model_digest,
                    profiler_catalog_digest,
                )))
                if profiler_catalog_digest is not None
                else (None,)
            )
            legacy_validation_inputs = []
            if legacy_profiler_catalog_digest is not None:
                # Read through both profiler-cache generations that predate
                # pool-local model identities.  The older production ABI used
                # the complete provenance digest and had no transfer-pool
                # field.  A short-lived intermediate ABI added the pool field
                # while retaining that complete digest.  Keeping the two
                # candidates explicit lets an already-paid fit migrate without
                # rebuilding profiler surrogates or cross-validation folds.
                legacy_validation_inputs.extend((
                    (
                        cost_key,
                        profiler_pool_digests.get(domain),
                        legacy_profiler_catalog_digest,
                    ),
                    (
                        cost_key,
                        None,
                        legacy_profiler_catalog_digest,
                    ),
                ))
            for legacy_cost_key, legacy_pool_digest in (
                (
                    aggregate_local_cost_keys.get(domain),
                    aggregate_local_pool_digests.get(domain),
                ),
                (semantic_global_cost_keys[domain], None),
                (aggregate_global_cost_keys.get(domain), None),
            ):
                legacy_validation_inputs.extend(
                    (
                        legacy_cost_key,
                        legacy_pool_digest,
                        catalog_digest,
                    )
                    for catalog_digest in catalog_digest_candidates
                )
            if not hit:
                for (
                    legacy_cost_key,
                    legacy_pool_digest,
                    legacy_catalog_digest,
                ) in dict.fromkeys(
                    legacy_validation_inputs
                ):
                    if legacy_cost_key is None:
                        continue
                    legacy_validation_key = fit_cache.validation_key(
                        domain,
                        legacy_cost_key,
                        max_leaves=max_leaves,
                        min_shape_groups_per_leaf=min_shape_groups_per_leaf,
                        cross_validation_seed=cross_validation_seed,
                        profiler_feature_catalog_digest=legacy_catalog_digest,
                        fit_final_rules=fit_final_rules,
                        profiler_training_pool_digest=legacy_pool_digest,
                    )
                    if legacy_validation_key == validation_key:
                        continue
                    hit, cached_validation, cached_rules = (
                        fit_cache.load_validation_entry(
                            legacy_validation_key,
                            domain,
                        )
                    )
                    if hit:
                        if (
                            cached_rules is not None
                            or not fit_final_rules
                            or cached_validation is None
                            or not domain_cross_validation_is_promotable(
                                cached_validation
                            )
                        ):
                            fit_cache.store_validation(
                                validation_key,
                                domain,
                                cached_validation,
                                cached_rules,
                            )
                        break
            if hit:
                validations[domain] = cached_validation
                validation_frontiers[domain] = (
                    (cached_validation,)
                    if cached_validation is not None
                    and domain_cross_validation_has_complete_coverage(
                        cached_validation
                    )
                    else ()
                )
                if cached_rules is not None:
                    cached_final_rules[domain] = cached_rules
                validation_cache_hits += 1
    validation_cache_complete = time.perf_counter()

    # Paired-request planning consumes held-out decisions, not final rules. A
    # cached validation is therefore sufficient by itself. Production fitting
    # also bypasses a domain whose stable final publication leaves are cached;
    # only a missing final tree or an affected profiler-transfer pool needs its
    # candidate matrix deserialized.
    costs = {}
    cost_cache_hits = 0
    cost_cache_bypasses = 0
    missing_profiler_pools = {
        _profiler_transfer_key(domain)
        for domain in domains
        if domain not in validations
    }
    for domain in domains:
        requires_costs = (
            domain not in validations
            or (
                fit_final_rules
                and domain not in cached_final_rules
                and validations[domain] is not None
                and domain_cross_validation_has_complete_coverage(
                    validations[domain]
                )
            )
            or (
                profiler_feature_catalog is not None
                and _profiler_transfer_key(domain) in missing_profiler_pools
            )
        )
        if not requires_costs:
            cost_cache_bypasses += 1
            continue
        cached_costs = (
            fit_cache.load_costs(cost_keys[domain], domain)
            if fit_cache is not None
            else None
        )
        if cached_costs is None and fit_cache is not None:
            for legacy_cost_key in legacy_cost_key_candidates[domain]:
                cached_costs = fit_cache.load_costs(legacy_cost_key, domain)
                if cached_costs is not None:
                    fit_cache.store_costs(
                        cost_keys[domain], domain, cached_costs
                    )
                    break
        if cached_costs is not None:
            costs[domain] = cached_costs
            cost_cache_hits += 1
            continue
        domain_corpus = corpus_for_domain(domain)
        domain_costs = build_domain_candidate_point_costs(
            domain_corpus,
            domain,
            serial_m1_hashes=serial_m1_hashes,
            paired_comparisons=paired_comparisons,
        )
        costs[domain] = domain_costs
        if fit_cache is not None:
            fit_cache.store_costs(cost_keys[domain], domain, domain_costs)
    costs_complete = time.perf_counter()
    if (
        profiler_feature_catalog is not None
        and domain_corpus_provider is not None
    ):
        # Cross-format profiler transfer needs authenticated observations for
        # every cost in the cross-M pool. Lazy CUDA projection therefore
        # materializes each domain exactly once before model construction; a
        # domain-local index would silently make pooling backend-asymmetric.
        for domain in domains:
            corpus_for_domain(domain)
        merged_profiler_index = {}
        for domain_corpus in domain_corpora.values():
            for point, observation in build_profiler_observation_index(
                domain_corpus
            ).items():
                previous = merged_profiler_index.setdefault(point, observation)
                if previous != observation:
                    raise ValueError(
                        "pooled profiler observation index changed one point"
                    )
        shared_profiler_observation_index = merged_profiler_index
    profiler_training_pool_lists: dict[
        tuple[object, ...], list[CandidatePointCost]
    ] = (
        defaultdict(list)
    )
    if profiler_feature_catalog is not None:
        for pool_domain, domain_costs in costs.items():
            profiler_training_pool_lists[
                _profiler_transfer_key(pool_domain)
            ].extend(domain_costs)
    profiler_training_pools = {
        key: tuple(pool)
        for key, pool in profiler_training_pool_lists.items()
    }
    profiler_training_pool_digests = {}
    if fit_cache is not None and profiler_feature_catalog is not None:
        for pool_domain in domains:
            pool_key = _profiler_transfer_key(pool_domain)
            digest = profiler_pool_digests[pool_domain]
            previous = profiler_training_pool_digests.setdefault(
                pool_key,
                digest,
            )
            if previous != digest:
                raise ValueError(
                    "profiler transfer pool domains resolved to different "
                    "candidate-cost identities"
                )
    tasks = []
    profiler_teacher_fold_results = []
    fold_counts = {}
    profiler_prediction_cache: ProfilerPredictionCache = {}
    profiler_prediction_request_keys = []
    profiler_prediction_point_sets: dict[
        ProfilerPredictionCacheKey, set[ProfilerPredictionPoint]
    ] = defaultdict(set)
    if profiler_feature_catalog is not None:
        for domain in domains:
            if domain in validations:
                continue
            domain_requests = _domain_profiler_prediction_requests(
                domain,
                costs.get(domain, []),
                seed=cross_validation_seed,
            )
            profiler_prediction_request_keys.extend(domain_requests)
            for request_key, points in domain_requests.items():
                profiler_prediction_point_sets[request_key].update(points)
        if shared_profiler_observation_index is None:
            raise RuntimeError(
                "profiler-informed fit did not construct a shared observation index"
            )
        profiler_models_started = time.perf_counter()
        if timing_enabled:
            print(
                "NativeVNNI profiler model plan "
                f"transfer_pools={len(profiler_training_pools)} "
                f"unique_surfaces={len(set(profiler_prediction_request_keys))}",
                file=sys.stderr,
                flush=True,
            )
        profiler_model_workers = _populate_profiler_prediction_cache(
            profiler_prediction_cache,
            profiler_prediction_request_keys,
            profiler_training_pools,
            development,
            profiler_feature_catalog,
            shared_profiler_observation_index,
            prediction_points_by_request={
                key: frozenset(points)
                for key, points in profiler_prediction_point_sets.items()
            },
            fit_cache=fit_cache,
            training_pool_digests=(
                profiler_training_pool_digests
                if fit_cache is not None
                else None
            ),
            profiler_model_digests=(
                profiler_model_digests
                if fit_cache is not None
                else None
            ),
        )
    else:
        profiler_models_started = time.perf_counter()
        profiler_model_workers = 0
    profiler_models_complete = time.perf_counter()
    if timing_enabled and profiler_feature_catalog is not None:
        print(
            "NativeVNNI profiler models complete "
            f"elapsed={profiler_models_complete - profiler_models_started:.3f}s "
            f"workers={profiler_model_workers}",
            file=sys.stderr,
            flush=True,
        )
    for domain in domains:
        if domain in validations:
            continue
        domain_tasks, fold_count = _domain_fold_tasks(
            domain,
            costs.get(domain, []),
            max_leaves=max_leaves,
            min_shape_groups_per_leaf=min_shape_groups_per_leaf,
            seed=cross_validation_seed,
            profiler_training_pool_key=(
                _profiler_transfer_key(domain)
                if profiler_feature_catalog is not None
                else None
            ),
            profiler_prediction_cache=(
                profiler_prediction_cache
                if profiler_feature_catalog is not None
                else None
            ),
        )
        tasks.extend(domain_tasks)
        fold_counts[domain] = fold_count
        if profiler_feature_catalog is not None:
            teacher_results = _domain_profiler_teacher_fold_results(
                domain,
                costs.get(domain, []),
                max_leaves=max_leaves,
                seed=cross_validation_seed,
                profiler_training_pool_key=_profiler_transfer_key(domain),
                profiler_prediction_cache=profiler_prediction_cache,
            )
            if len(teacher_results) != fold_count:
                raise RuntimeError(
                    "cross-fitted profiler teacher changed the domain fold count"
                )
            profiler_teacher_fold_results.extend(teacher_results)
    tasks_complete = time.perf_counter()
    requested_workers = None
    if accelerator_workers and tasks:
        accelerated_task_groups = _group_accelerated_cv_tasks(tasks)
        _PARALLEL_ACCELERATED_CV_TASK_GROUPS = accelerated_task_groups
        try:
            grouped_fold_results = _run_accelerated_tasks(
                accelerator_workers,
                tuple(
                    sum(max(1, len(task[6]) * task[7]) for _index, task in group)
                    for group in accelerated_task_groups
                ),
                "cv",
            )
            fold_results_by_index = {
                original_index: result
                for group_results in grouped_fold_results
                for original_index, result in group_results
            }
            if len(fold_results_by_index) != len(tasks):
                raise RuntimeError(
                    "accelerated CV groups changed the result cardinality: "
                    f"expected={len(tasks)}, "
                    f"observed={len(fold_results_by_index)}"
                )
            fold_results = [
                fold_results_by_index[index] for index in range(len(tasks))
            ]
        finally:
            _PARALLEL_ACCELERATED_CV_TASK_GROUPS = ()
    else:
        requested_workers = int(os.environ.get(
            "LLAMINAR_NATIVE_VNNI_POLICY_WORKERS",
            str(_physical_core_worker_count()),
        ))
        worker_count = max(1, min(requested_workers, len(tasks)))
        if worker_count > 1 and len(tasks) >= 4:
            _PARALLEL_CV_TASKS = tuple(tasks)
            try:
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=multiprocessing.get_context("fork"),
                ) as executor:
                    fold_results = list(executor.map(
                        _evaluate_parallel_cv_index,
                        range(len(_PARALLEL_CV_TASKS)),
                    ))
            finally:
                _PARALLEL_CV_TASKS = ()
        else:
            fold_results = [_evaluate_placement_fold(task) for task in tasks]
    # The teacher is evaluated in the parent from already-persisted prediction
    # surfaces. It performs no model fit and no tree search, so appending these
    # compact decisions after the accelerated work preserves both process-pool
    # ownership and deterministic tournament ordering.
    fold_results = [*fold_results, *profiler_teacher_fold_results]
    folds_complete = time.perf_counter()

    results_by_domain: dict[GenericDomain, list[_PlacementFoldResult]] = (
        defaultdict(list)
    )
    for result in fold_results:
        results_by_domain[result.domain].append(result)
    reduction_tasks = tuple(
        (
            domain,
            costs.get(domain, []),
            fold_counts.get(domain, 0),
            tuple(results_by_domain.get(domain, ())),
            max_leaves,
            fit_final_rules,
        )
        for domain in domains
        if domain not in validations
    )
    reduced_domains, reduction_worker_count = (
        _reduce_domain_cross_validations(
            reduction_tasks,
            requested_workers=requested_workers,
        )
    )
    for domain, ranked_validations, validation in reduced_domains:
        validations[domain] = validation
        promotable_frontier = tuple(
            candidate
            for candidate in ranked_validations
            if domain_cross_validation_is_promotable(candidate)
        )
        validation_frontiers[domain] = (
            promotable_frontier
            if promotable_frontier
            else tuple(ranked_validations)
        )
        if fit_cache is not None and (
            not fit_final_rules
            or validation is None
            or not domain_cross_validation_is_promotable(validation)
        ):
            # A promotable model is still provisional until the complete
            # development tree passes its own leaf-p95 gate. Content-addressed
            # cache records are immutable, so publish those domains exactly
            # once after final-frontier selection instead of attempting to
            # mutate a preliminary CV winner in place.
            fit_cache.store_validation(
                validation_keys[domain], domain, validation
            )
    selection_complete = time.perf_counter()

    structurally_complete_validations = {
        domain: validation
        for domain, validation in validations.items()
        if validation is not None
        and domain_cross_validation_has_complete_coverage(validation)
    }
    cv_passing_domain_count = sum(
        domain_cross_validation_is_promotable(validation)
        for validation in structurally_complete_validations.values()
    )
    cv_allows_performance_exceptions = (
        len(structurally_complete_validations) == len(domains)
        and domain_promotion_quota_is_satisfied(
            cv_passing_domain_count,
            len(domains),
        )
    )

    final_tasks = []
    if fit_final_rules:
        for domain, validation in validations.items():
            if domain in cached_final_rules:
                continue
            if (
                validation is None
                or not domain_cross_validation_has_complete_coverage(validation)
                or (
                    not domain_cross_validation_is_promotable(validation)
                    and not cv_allows_performance_exceptions
                )
            ):
                continue
            domain_corpus = corpus_for_domain(domain)
            domain_costs = costs.get(domain, [])
            final_tasks.append((
                domain,
                domain_costs,
                domain_corpus.rows_for_generic_domain(domain),
                validation_frontiers.get(domain, (validation,)),
                tuple(results_by_domain.get(domain, ())),
                min_shape_groups_per_leaf,
                max_leaves,
            ))
    final_tasks = tuple(final_tasks)
    if accelerator_workers and final_tasks:
        _PARALLEL_FINAL_TASKS = final_tasks
        try:
            final_fits = _run_accelerated_tasks(
                accelerator_workers,
                tuple(
                    max(
                        1,
                        len(task[1])
                        * task[6]
                        * len(FEATURE_POLICIES)
                        * len(BOUNDARY_PLACEMENTS),
                    )
                    for task in _PARALLEL_FINAL_TASKS
                ),
                "final",
            )
        finally:
            _PARALLEL_FINAL_TASKS = ()
    else:
        if requested_workers is None:
            requested_workers = int(os.environ.get(
                "LLAMINAR_NATIVE_VNNI_POLICY_WORKERS",
                str(_physical_core_worker_count()),
            ))
        final_worker_count = max(1, min(requested_workers, len(final_tasks)))
        if final_worker_count > 1 and len(final_tasks) >= 4:
            _PARALLEL_FINAL_TASKS = final_tasks
            try:
                with ProcessPoolExecutor(
                    max_workers=final_worker_count,
                    mp_context=multiprocessing.get_context("fork"),
                ) as executor:
                    final_fits = list(executor.map(
                        _fit_parallel_final_index,
                        range(len(_PARALLEL_FINAL_TASKS)),
                    ))
            finally:
                _PARALLEL_FINAL_TASKS = ()
        else:
            final_fits = [_fit_final_domain(task) for task in final_tasks]
    final_complete = time.perf_counter()
    if timing_enabled:
        print(
            "NativeVNNI policy timing "
            "cache_identity="
            f"{cache_identity_complete - fit_started:.3f}s "
            "cost_keys="
            f"{cost_keys_complete - cache_identity_complete:.3f}s "
            "profiler_identity="
            f"{profiler_identities_complete - cost_keys_complete:.3f}s "
            "cache_lookup="
            f"{validation_cache_complete - profiler_identities_complete:.3f}s "
            "candidate_costs="
            f"{costs_complete - validation_cache_complete:.3f}s "
            f"profiler_setup={profiler_models_started - costs_complete:.3f}s "
            f"profiler_models={profiler_models_complete - profiler_models_started:.3f}s "
            f"task_construction={tasks_complete - profiler_models_complete:.3f}s "
            f"fold_fit={folds_complete - tasks_complete:.3f}s "
            f"cv_reduction={selection_complete - folds_complete:.3f}s "
            f"final_fit={final_complete - selection_complete:.3f}s "
            f"cost_cache_hits={cost_cache_hits}/{len(domains)} "
            f"cost_cache_bypasses={cost_cache_bypasses}/{len(domains)} "
            f"cv_cache_hits={validation_cache_hits}/{len(domains)} "
            f"final_rule_cache_hits={len(cached_final_rules)}/{len(domains)} "
            f"profiler_model_workers={profiler_model_workers} "
            f"cv_reduction_workers={reduction_worker_count} "
            f"accelerator_processes={min(len(accelerator_workers), len(tasks))} ",
            file=sys.stderr,
            flush=True,
        )
    rules_by_domain = dict(cached_final_rules)
    for domain, domain_rules, selected_validation in final_fits:
        selected_validation = _hydrate_cross_validation_cells(
            selected_validation,
            results_by_domain.get(domain, ()),
        )
        rules_by_domain[domain] = domain_rules
        validations[domain] = selected_validation
        if (
            fit_cache is not None
            and domain_cross_validation_is_promotable(selected_validation)
        ):
            # Publish the stable full-development winner under the same search
            # key. A later immutable-corpus refit can reuse it directly instead
            # of reconstructing an already-proven model frontier.
            fit_cache.store_validation(
                validation_keys[domain],
                domain,
                selected_validation,
                domain_rules,
            )

    rules = []
    unpromoted = []
    structurally_unpromoted = []
    performance_exception_rules = []
    promotion_diagnostics = []
    cross_validation = []
    for domain in domains:
        validation = validations[domain]
        if validation is not None:
            cross_validation.append(validation)
        domain_rules = rules_by_domain.get(domain, ())
        worst_final_rule = (
            max(
                domain_rules,
                key=lambda rule: (
                    rule.development_p95_regret,
                    rule.development_max_regret,
                    rule.candidate_id,
                ),
            )
            if domain_rules
            else None
        )
        worst_final_cost = (
            max(
                (
                    cost
                    for cost in costs.get(domain, ())
                    if worst_final_rule is not None
                    and cost.candidate_id == worst_final_rule.candidate_id
                    and worst_final_rule.matches(
                        cost.runtime_key.aggregate_n,
                        cost.runtime_key.k,
                    )
                ),
                key=lambda cost: (
                    cost.p95_surface_regret,
                    cost.max_surface_regret,
                    cost.shape_group_id,
                ),
                default=None,
            )
            if worst_final_rule is not None
            else None
        )
        rules_are_promotable = bool(domain_rules) and all(
            rule.development_p95_regret < GENERIC_REGRET_BUDGET
            for rule in domain_rules
        )
        rejection_stage = ""
        if validation is None:
            rejection_stage = "cross_validation_missing"
        elif (
            validation.required_point_count <= 0
            or validation.covered_point_count
            != validation.required_point_count
        ):
            rejection_stage = "cross_validation_coverage"
        elif (
            fit_final_rules
            and not domain_rules
            and (
                domain_cross_validation_is_promotable(validation)
                or cv_allows_performance_exceptions
            )
        ):
            rejection_stage = "final_fit_empty"
        elif validation.p95_regret >= GENERIC_REGRET_BUDGET:
            rejection_stage = "cross_validation_p95"
        elif (
            fit_final_rules
            and validation.publication_oof_p95_regret is not None
            and validation.publication_oof_p95_regret
            >= GENERIC_REGRET_BUDGET
        ):
            rejection_stage = "final_fit_cross_fitted_p95"
        elif fit_final_rules and not rules_are_promotable:
            rejection_stage = "final_fit_p95"
        if rejection_stage:
            unpromoted.append(domain)
            if rejection_stage in {
                "cross_validation_missing",
                "cross_validation_coverage",
                "final_fit_empty",
            }:
                structurally_unpromoted.append(domain)
            else:
                performance_exception_rules.extend(domain_rules)
            promotion_diagnostics.append(DomainPromotionDiagnostic(
                domain=domain,
                rejection_stage=rejection_stage,
                cv_required_point_count=(
                    validation.required_point_count
                    if validation is not None
                    else 0
                ),
                cv_covered_point_count=(
                    validation.covered_point_count
                    if validation is not None
                    else 0
                ),
                cv_p95_regret=(
                    validation.p95_regret
                    if validation is not None
                    else None
                ),
                cv_max_regret=(
                    validation.max_regret
                    if validation is not None
                    else None
                ),
                final_cross_fitted_p95_regret=(
                    validation.publication_oof_p95_regret
                    if validation is not None
                    else None
                ),
                final_rule_count=len(domain_rules),
                final_worst_p95_regret=(
                    worst_final_rule.development_p95_regret
                    if worst_final_rule is not None
                    else None
                ),
                final_worst_max_regret=(
                    worst_final_rule.development_max_regret
                    if worst_final_rule is not None
                    else None
                ),
                final_worst_shape_group_id=(
                    worst_final_cost.shape_group_id
                    if worst_final_cost is not None
                    else ""
                ),
                final_worst_aggregate_n=(
                    worst_final_cost.runtime_key.aggregate_n
                    if worst_final_cost is not None
                    else None
                ),
                final_worst_k=(
                    worst_final_cost.runtime_key.k
                    if worst_final_cost is not None
                    else None
                ),
                final_worst_candidate_id=(
                    worst_final_rule.candidate_id
                    if worst_final_rule is not None
                    else ""
                ),
            ))
            continue
        rules.extend(domain_rules)

    passing_domain_count = len(domains) - len(unpromoted)
    corpus_is_promotable = (
        not structurally_unpromoted
        and domain_promotion_quota_is_satisfied(
            passing_domain_count,
            len(domains),
        )
    )
    if corpus_is_promotable:
        # The exception trees are still the best measured generic dispatch for
        # their complete domains. Publish them to preserve totality, retain the
        # diagnostics above, and clear only the corpus-blocking obligation set.
        rules.extend(performance_exception_rules)
        unpromoted.clear()

    return GenericPolicy(
        rules=tuple(sorted(rules, key=lambda rule: (
            rule.domain,
            tuple(
                (
                    predicate.threshold.axis.value,
                    predicate.threshold.numerator,
                    predicate.threshold.denominator,
                    predicate.threshold.parallelism_width,
                    predicate.threshold.task_multiplier,
                    predicate.require_less_equal,
                )
                for predicate in rule.predicates
            ),
            rule.candidate_id,
        ))),
        unpromoted_domains=tuple(sorted(unpromoted)),
        cross_validation=tuple(sorted(
            cross_validation, key=lambda result: result.domain
        )),
        promotion_diagnostics=tuple(sorted(
            promotion_diagnostics,
            key=lambda result: result.domain,
        )),
    )
