"""Auditable CUDA shape-resolved K-partition evidence projection.

Parameterized generic candidates do not invent measurements. For each runtime
key this module resolves the formula to one concrete exact-KB candidate and
projects that candidate's directly measured correctness, output, route, and
timing evidence under the nominal formula ID. Missing exact evidence leaves the
formula unavailable for that point.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import defaultdict
from functools import lru_cache
from typing import Iterable

from .candidate_registry import CandidateSpec, cuda_native_vnni_gemv_registry
from .corpus import GenericDomain, ObservationCorpus, RuntimeKey, runtime_key
from .schema import NativeVNNIObservation, SemanticContract


CUDA_FORMULA_MAX_KB = 256
CUDA_SHAPE_RESOLVED_PROJECTION_VERSION = (
    "native-vnni-cuda-shape-resolved-projection-v1"
)


def _clone_validated_observation(
    source: NativeVNNIObservation,
    **updates,
) -> NativeVNNIObservation:
    """Clone one validated frozen row without replaying its 70-field parser.

    Shape-resolved projection changes only registry-owned policy metadata on an
    immutable source row that already passed complete schema validation.  A
    normal ``dataclasses.replace`` reconstructs and rebinds every field for
    hundreds of thousands of formula aliases.  Copying the instance dictionary
    preserves every proven source value verbatim and makes the reviewed updates
    explicit.  The projection regression validates every generated row, and the
    final ``ObservationCorpus`` still checks candidate identity consistency.
    """

    clone = object.__new__(NativeVNNIObservation)
    object.__setattr__(clone, "__dict__", source.__dict__.copy())
    clone.__dict__.update(updates)
    return clone


@lru_cache(maxsize=None)
def _sorted_positive_divisors(value: int) -> tuple[int, ...]:
    """Return every positive divisor once for exact nearest-factor searches."""

    if value <= 0:
        raise ValueError("divisor inventory requires a positive integer")
    lower = []
    upper = []
    for divisor in range(1, math.isqrt(value) + 1):
        if value % divisor:
            continue
        lower.append(divisor)
        paired = value // divisor
        if paired != divisor:
            upper.append(paired)
    return tuple((*lower, *reversed(upper)))


def _nearest_factor_kb(
    *,
    grid_n: int,
    k_groups: int,
    target_blocks: int,
    min_kgroups_per_cta: int,
    max_kb: int,
) -> int:
    """Mirror CUDA's deterministic occupancy partition resolver exactly."""

    if min_kgroups_per_cta <= 0:
        raise ValueError("min_kgroups_per_cta must be positive")
    kb = max(2, (target_blocks + grid_n - 1) // grid_n)
    kb_max = max(2, k_groups // min_kgroups_per_cta)
    kb = min(kb, kb_max)
    if k_groups % kb:
        divisors = _sorted_positive_divisors(k_groups)
        insertion = bisect_left(divisors, kb)
        lower = (
            divisors[insertion - 1]
            if insertion > 0 and divisors[insertion - 1] >= 2
            else -1
        )
        upper = (
            divisors[insertion]
            if insertion < len(divisors) and divisors[insertion] <= kb_max
            else -1
        )
        if lower > 0 and upper > 0:
            lower_distance = abs(grid_n * lower - target_blocks)
            upper_distance = abs(grid_n * upper - target_blocks)
            kb = upper if upper_distance < lower_distance else lower
        elif lower > 0:
            kb = lower
        elif upper > 0:
            kb = upper
    return min(max_kb, k_groups, max(1, kb))


def _canonical_partition_kb(
    *,
    grid_n: int,
    k_groups: int,
    target_blocks: int,
    min_kgroups_per_cta: int,
    max_kb: int,
) -> int:
    """Resolve target occupancy without a divisor-induced KB discontinuity.

    Exact KPAR kernels already support a shorter final K partition. Requiring
    ``KB`` to divide ``k_groups`` is therefore not a correctness condition and
    can turn a desired 16-way split into 83 partitions when ``k_groups`` is
    prime. This resolver first computes the desired partition count, converts
    it to a groups-per-partition width, then returns the smallest KB spelling
    that realizes that width. The result is one of the economical exact-KB
    schedules emitted by the trainer and retains deterministic ascending
    partial publication for every row.
    """

    if min_kgroups_per_cta <= 0:
        raise ValueError("min_kgroups_per_cta must be positive")
    desired_kb = max(2, (target_blocks + grid_n - 1) // grid_n)
    kb_max = max(1, k_groups // min_kgroups_per_cta)
    desired_kb = min(desired_kb, kb_max, max_kb, k_groups)
    groups_per_partition = (
        k_groups + desired_kb - 1
    ) // desired_kb
    canonical_kb = (
        k_groups + groups_per_partition - 1
    ) // groups_per_partition
    return min(max_kb, k_groups, max(1, canonical_kb))


def resolve_cuda_formula_kb(candidate: CandidateSpec, n: int, k: int) -> int:
    """Resolve one reviewed formula to its concrete exact partition count."""

    config = candidate.config_json
    if config.get("family") != "kpar_formula":
        raise ValueError(f"{candidate.candidate_id} is not a CUDA KB formula")
    if n <= 0 or k <= 0 or k % 32:
        raise ValueError(f"formula dimensions must satisfy N>0 and K%32==0: {n}x{k}")
    tile_n = int(config["tile_n"])
    max_kb = int(config["max_kb"])
    k_groups = k // 32
    formula_kind = str(config["formula_kind"])
    if formula_kind == "target_blocks":
        return _nearest_factor_kb(
            grid_n=(n + tile_n - 1) // tile_n,
            k_groups=k_groups,
            target_blocks=int(config["target_blocks"]),
            min_kgroups_per_cta=int(config["min_kgroups_per_cta"]),
            max_kb=max_kb,
        )
    if formula_kind == "canonical_target_blocks":
        return _canonical_partition_kb(
            grid_n=(n + tile_n - 1) // tile_n,
            k_groups=k_groups,
            target_blocks=int(config["target_blocks"]),
            min_kgroups_per_cta=int(config["min_kgroups_per_cta"]),
            max_kb=max_kb,
        )
    if formula_kind == "blocks_per_partition":
        groups_per_partition = int(config["blocks_per_partition"])
        if groups_per_partition <= 0:
            raise ValueError("blocks_per_partition must be positive")
        return min(
            max_kb,
            k_groups,
            (k_groups + groups_per_partition - 1) // groups_per_partition,
        )
    raise ValueError(f"unknown CUDA KB formula kind {formula_kind!r}")


@lru_cache(maxsize=None)
def resolve_cuda_concrete_candidate_id(
    candidate: CandidateSpec,
    n: int,
    k: int,
) -> str:
    """Return the forceable candidate ID represented at one CUDA shape.

    Static WIDE, DIRECT, and exact-KB KPAR candidates already name the launch
    that the C++ trainer can force. A generic KPAR formula instead names a
    policy rule, so confirmation tooling must resolve its exact KB before it
    can ask the production launcher to measure the selected/reference pair.
    Keeping this conversion beside the formula arithmetic prevents the Python
    planner and C++ trainer command line from developing subtly different
    spellings for the same launch.
    """

    config = candidate.config_json
    if config.get("family") != "kpar_formula":
        return candidate.candidate_id
    kb = resolve_cuda_formula_kb(candidate, n, k)
    return (
        "cuda.nvnni.decode.fast_m1.kpar."
        f"tn{int(config['tile_n'])}.cpt{int(config['cpt'])}.kb{kb}"
    )


def project_cuda_shape_resolved_candidates(
    corpus: Iterable,
    *,
    known_generic_domain: GenericDomain | None = None,
) -> ObservationCorpus:
    """Add formula observations backed one-for-one by direct exact evidence.

    The returned corpus retains every original row. A projected row keeps the
    source row's effective/observed candidate, output digests, and timing-sample
    hash, while replacing only the nominal policy candidate and its immutable
    formula identity. This lets exact overlays remain concrete and lets generic
    policy fitting compare deterministic formulas without timing equivalent
    kernels twice.
    """

    rows_in_corpus = tuple(corpus)
    registry = cuda_native_vnni_gemv_registry()
    formulas = tuple(
        (
            candidate,
            candidate.candidate_policy_hash(),
            candidate.config_json,
        )
        for candidate in registry.entries
        if candidate.config_json.get("family") == "kpar_formula"
    )
    grouped: dict[RuntimeKey, list] = defaultdict(list)
    for row in rows_in_corpus:
        grouped[runtime_key(row)].append(row)

    # A concrete KPAR ``kbN`` schedule is an exact-shape decision: N fixes the
    # partition count and therefore the ordered reduction identity. It is not a
    # total generic candidate because another K may not expose that KB in its
    # economical measured matrix. Keep the row for exact-overlay selection, but
    # prevent the generic learner from extrapolating the literal KB. The
    # formula rows added below are the only policy-selectable generic KPAR
    # candidates and resolve to directly measured concrete evidence per shape.
    projected = [
        _clone_validated_observation(row, generic_eligible=False)
        if (
            row.semantic_contract == SemanticContract.FAST
            and row.m == 1
            and row.config_json.get("family") == "kpar"
        )
        else row
        for row in rows_in_corpus
    ]
    resolved_candidate_ids = {}
    for key, rows in grouped.items():
        if key.semantic_contract != SemanticContract.FAST or key.m != 1:
            continue
        concrete_rows: dict[str, list] = defaultdict(list)
        for row in rows:
            if row.candidate_id == row.effective_candidate_id:
                concrete_rows[row.effective_candidate_id].append(row)
        for formula, candidate_policy_hash, formula_config in formulas:
            resolution_key = (
                formula.candidate_id,
                key.aggregate_n,
                key.k,
            )
            concrete_id = resolved_candidate_ids.get(resolution_key)
            if concrete_id is None:
                concrete_id = resolve_cuda_concrete_candidate_id(
                    formula, key.aggregate_n, key.k
                )
                resolved_candidate_ids[resolution_key] = concrete_id
            for source in concrete_rows.get(concrete_id, ()):
                projection = _clone_validated_observation(
                    source,
                    candidate_id=formula.candidate_id,
                    candidate_family=formula.candidate_family,
                    config_json=formula_config,
                    generic_eligible=True,
                    arithmetic_fingerprint=formula.arithmetic_fingerprint,
                    candidate_policy_hash=candidate_policy_hash,
                )
                projected.append(projection)
    # Source identities were already proven by their owning corpus. Every new
    # formula row takes all nominal identity fields from one immutable registry
    # entry, so a second config-dictionary comparison over millions of aliases
    # cannot discover new information. Runtime/domain indices are still built
    # normally and the projection regression validates generated row schemas.
    return ObservationCorpus._from_validated(
        projected,
        revalidate_candidate_identities=False,
        known_generic_domain=known_generic_domain,
    )
