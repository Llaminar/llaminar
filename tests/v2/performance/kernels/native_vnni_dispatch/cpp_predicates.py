"""Render backend-neutral NativeVNNI tree predicates as exact C++ expressions.

The common learner owns the semantics of every feature axis. Backend policy
generators must therefore emit those semantics mechanically rather than
reconstructing a smaller local model. Keeping the integer/rational rendering in
one module prevents CUDA, ROCm, and CPU from disagreeing at aspect or launch-tile
boundaries after they have certified the same :class:`GenericDispatchRule`.
"""

from __future__ import annotations

from .schema import AspectBucket
from .segmented_policy import (
    FEATURE_AXIS_PRIORITY,
    K_FINAL_TILE_WIDTH_BY_AXIS,
    K_GROUPS_PER_N_TILE_WIDTH_BY_AXIS,
    MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS,
    MN_PARALLEL_WAVE_WIDTH_BY_AXIS,
    N_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS,
    N_FINAL_TILE_WIDTH_BY_AXIS,
    N_PARALLEL_WAVE_WIDTH_BY_AXIS,
    N_TILE_ALIGNED_WIDTH_BY_AXIS,
    N_TILE_WIDTH_BY_AXIS,
    N_TILE_UTILIZATION_WIDTH_BY_AXIS,
    FeatureAxis,
    FeaturePredicate,
    GenericDispatchRule,
)


def _i64(expression: str) -> str:
    """Cast one runtime dimension before any potentially widening arithmetic."""

    return f"static_cast<long long>({expression})"


def aspect_condition(
    bucket: AspectBucket,
    *,
    n_expression: str = "n",
    k_expression: str = "k",
) -> str:
    """Return the exact, non-overlapping condition for one aspect bucket.

    The Python classifier uses the ordered boundaries ``N/K >= 16``, ``>= 2``,
    and ``>= 3/4``. Generated C++ spells each interval independently so a rule
    remains correct even when another bucket has no emitted leaves. No floating
    division is permitted at a dispatch boundary.
    """

    n64 = _i64(n_expression)
    k64 = _i64(k_expression)
    return {
        AspectBucket.VERY_WIDE: f"{n64} >= 16LL * {k64}",
        AspectBucket.WIDE: (
            f"{n64} >= 2LL * {k64} && {n64} < 16LL * {k64}"
        ),
        AspectBucket.BALANCED: (
            f"4LL * {n64} >= 3LL * {k64} && {n64} < 2LL * {k64}"
        ),
        AspectBucket.TALL: f"4LL * {n64} < 3LL * {k64}",
    }[bucket]


def predicate_condition(
    predicate: FeaturePredicate,
    *,
    n_expression: str = "n",
    k_expression: str = "k",
    work_expression: str = "work_items",
) -> str:
    """Render one learner edge with the same exact rational comparison as Python.

    Tile-count features use positive ceil division, and K-groups-per-N-tile uses
    the learner's 32-value NativeVNNI group unit. Every intermediate is widened
    before addition or multiplication so large LM-head dimensions cannot change
    a branch through signed 32-bit overflow.
    """

    threshold = predicate.threshold
    numerator = threshold.numerator
    denominator = threshold.denominator
    n64 = _i64(n_expression)
    k64 = _i64(k_expression)

    if threshold.axis == FeatureAxis.AGGREGATE_N:
        comparison = f"{n64} * {denominator}LL <= {numerator}LL"
    elif threshold.axis == FeatureAxis.K:
        comparison = f"{k64} * {denominator}LL <= {numerator}LL"
    elif threshold.axis == FeatureAxis.WORK_ITEMS:
        comparison = (
            f"{work_expression} * {denominator}LL <= {numerator}LL"
        )
    elif threshold.axis == FeatureAxis.ASPECT_RATIO:
        comparison = (
            f"{n64} * {denominator}LL <= {k64} * {numerator}LL"
        )
    elif threshold.axis in N_TILE_WIDTH_BY_AXIS:
        width = N_TILE_WIDTH_BY_AXIS[threshold.axis]
        tile_count = f"(({n64} + {width - 1}LL) / {width}LL)"
        comparison = (
            f"{tile_count} * {denominator}LL <= {numerator}LL"
        )
    elif threshold.axis in K_GROUPS_PER_N_TILE_WIDTH_BY_AXIS:
        width = K_GROUPS_PER_N_TILE_WIDTH_BY_AXIS[threshold.axis]
        tile_count = f"(({n64} + {width - 1}LL) / {width}LL)"
        comparison = (
            f"{k64} * {denominator}LL <= "
            f"32LL * {tile_count} * {numerator}LL"
        )
    elif threshold.axis in N_FINAL_TILE_WIDTH_BY_AXIS:
        width = N_FINAL_TILE_WIDTH_BY_AXIS[threshold.axis]
        final_tile_values = f"((({n64} - 1LL) % {width}LL) + 1LL)"
        comparison = (
            f"{final_tile_values} * {denominator}LL <= {numerator}LL"
        )
    elif threshold.axis in K_FINAL_TILE_WIDTH_BY_AXIS:
        width = K_FINAL_TILE_WIDTH_BY_AXIS[threshold.axis]
        final_tile_values = f"((({k64} - 1LL) % {width}LL) + 1LL)"
        comparison = (
            f"{final_tile_values} * {denominator}LL <= {numerator}LL"
        )
    elif threshold.axis in N_TILE_UTILIZATION_WIDTH_BY_AXIS:
        width = N_TILE_UTILIZATION_WIDTH_BY_AXIS[threshold.axis]
        tile_count = f"(({n64} + {width - 1}LL) / {width}LL)"
        comparison = (
            f"{n64} * {denominator}LL <= "
            f"{tile_count} * {width}LL * {numerator}LL"
        )
    elif threshold.axis in N_TILE_ALIGNED_WIDTH_BY_AXIS:
        width = N_TILE_ALIGNED_WIDTH_BY_AXIS[threshold.axis]
        aligned = f"(({n64} % {width}LL) == 0LL ? 1LL : 0LL)"
        comparison = (
            f"{aligned} * {denominator}LL <= {numerator}LL"
        )
    elif threshold.axis in N_PARALLEL_WAVE_WIDTH_BY_AXIS:
        width = N_PARALLEL_WAVE_WIDTH_BY_AXIS[threshold.axis]
        tasks = f"(({n64} + {width - 1}LL) / {width}LL)"
        waves = (
            f"(({tasks} + {threshold.parallelism_width - 1}LL) / "
            f"{threshold.parallelism_width}LL)"
        )
        comparison = f"{waves} * {denominator}LL <= {numerator}LL"
    elif threshold.axis in N_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS:
        width = N_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS[threshold.axis]
        tasks = f"(({n64} + {width - 1}LL) / {width}LL)"
        final_wave_tasks = (
            f"((({tasks} - 1LL) % {threshold.parallelism_width}LL) + 1LL)"
        )
        comparison = (
            f"{final_wave_tasks} * {denominator}LL <= "
            f"{threshold.parallelism_width}LL * {numerator}LL"
        )
    elif threshold.axis in MN_PARALLEL_WAVE_WIDTH_BY_AXIS:
        width = MN_PARALLEL_WAVE_WIDTH_BY_AXIS[threshold.axis]
        n_tiles = f"(({n64} + {width - 1}LL) / {width}LL)"
        tasks = f"({threshold.task_multiplier}LL * {n_tiles})"
        waves = (
            f"(({tasks} + {threshold.parallelism_width - 1}LL) / "
            f"{threshold.parallelism_width}LL)"
        )
        comparison = f"{waves} * {denominator}LL <= {numerator}LL"
    elif threshold.axis in MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS:
        width = MN_FINAL_PARALLEL_WAVE_WIDTH_BY_AXIS[threshold.axis]
        n_tiles = f"(({n64} + {width - 1}LL) / {width}LL)"
        tasks = f"({threshold.task_multiplier}LL * {n_tiles})"
        final_wave_tasks = (
            f"((({tasks} - 1LL) % {threshold.parallelism_width}LL) + 1LL)"
        )
        comparison = (
            f"{final_wave_tasks} * {denominator}LL <= "
            f"{threshold.parallelism_width}LL * {numerator}LL"
        )
    else:
        raise ValueError(
            f"unsupported NativeVNNI policy feature {threshold.axis!r}"
        )
    return comparison if predicate.require_less_equal else f"!({comparison})"


def generic_rule_sort_key(rule: GenericDispatchRule) -> tuple:
    """Return one deterministic backend-independent order for emitted leaves."""

    predicates = tuple(
        (
            FEATURE_AXIS_PRIORITY[predicate.threshold.axis],
            predicate.threshold.numerator,
            predicate.threshold.denominator,
            predicate.threshold.parallelism_width,
            predicate.threshold.task_multiplier,
            predicate.require_less_equal,
        )
        for predicate in rule.predicates
    )
    return (
        not rule.domain.all_aspects,
        list(AspectBucket).index(rule.domain.aspect_bucket),
        predicates,
        rule.candidate_id,
    )


def render_if_header(
    conditions: tuple[str, ...] | list[str],
    *,
    indent: str,
) -> list[str]:
    """Render a readable multi-line C++ ``if`` header for a conjunction."""

    if not conditions:
        raise ValueError("a generated dispatch branch needs at least one condition")
    if len(conditions) == 1:
        return [f"{indent}if ({conditions[0]})"]
    continuation = indent + "    "
    lines = [f"{indent}if ({conditions[0]} &&"]
    lines.extend(
        f"{continuation}{condition} &&"
        for condition in conditions[1:-1]
    )
    lines.append(f"{continuation}{conditions[-1]})")
    return lines
