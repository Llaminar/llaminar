"""Auditable ROCm shape-clamped K-partition evidence projection.

ROCm's production M=1 launcher treats a generated KB as a requested partition
count and clamps it to the number of 32-value K groups before launch. Concrete
``KBx`` measurements consequently describe exact shapes, while generic policy
rules describe ``min(requested_kb, K / 32)``. This module makes that distinction
explicit: every formula row is backed one-for-one by a measured concrete row
and retains its correctness, route, output digest, and timing sample identity.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Iterable

from .candidate_registry import (
    CandidateSpec,
    rocm_native_vnni_decode_formula_registry,
)
from .corpus import GenericDomain, ObservationCorpus, RuntimeKey, runtime_key
from .schema import NativeVNNIObservation, SemanticContract


ROCM_SHAPE_RESOLVED_PROJECTION_VERSION = (
    "native-vnni-rocm-shape-clamped-projection-v1"
)


def _clone_validated_observation(
    source: NativeVNNIObservation,
    **updates,
) -> NativeVNNIObservation:
    """Clone one validated row while changing only registry-owned identity."""

    clone = object.__new__(NativeVNNIObservation)
    object.__setattr__(clone, "__dict__", source.__dict__.copy())
    clone.__dict__.update(updates)
    return clone


def resolve_rocm_formula_kb(candidate: CandidateSpec, k: int) -> int:
    """Resolve one requested-KB formula exactly as the production launcher."""

    config = candidate.config_json
    if config.get("family") != "clamped_kb_formula":
        raise ValueError(f"{candidate.candidate_id} is not a ROCm KB formula")
    group_width = int(config["k_group_width"])
    if k <= 0 or group_width <= 0 or k % group_width:
        raise ValueError(
            f"ROCm formula K must be a positive multiple of {group_width}: {k}"
        )
    requested_kb = int(config["kb"])
    if requested_kb <= 0:
        raise ValueError("ROCm formula requested KB must be positive")
    return min(requested_kb, k // group_width)


@lru_cache(maxsize=None)
def resolve_rocm_concrete_candidate_id(
    candidate: CandidateSpec,
    k: int,
) -> str:
    """Return the measured concrete launch represented at one K geometry."""

    return (
        "rocm.nvnni.decode.fast."
        f"kb{resolve_rocm_formula_kb(candidate, k)}"
    )


def project_rocm_shape_resolved_candidates(
    corpus: Iterable[NativeVNNIObservation],
    *,
    known_generic_domain: GenericDomain | None = None,
    distinguish_execution_mode: bool = True,
) -> ObservationCorpus:
    """Add total requested-KB formulas backed by concrete exact evidence.

    Original concrete rows remain available to exact-overlay selection but are
    marked ineligible for generic fitting. Formula aliases keep the concrete
    effective/observed launch identity and all measured evidence, replacing
    only the nominal policy identity with the registry-owned clamped formula.
    """

    source_rows = tuple(corpus)
    formulas = tuple(
        (
            candidate,
            candidate.candidate_policy_hash(),
            candidate.config_json,
        )
        for candidate in rocm_native_vnni_decode_formula_registry().entries
    )
    grouped: dict[RuntimeKey, list[NativeVNNIObservation]] = defaultdict(list)
    for row in source_rows:
        grouped[runtime_key(row)].append(row)

    projected = [
        _clone_validated_observation(row, generic_eligible=False)
        if row.semantic_contract == SemanticContract.FAST and row.m == 1
        else row
        for row in source_rows
    ]
    resolved_ids: dict[tuple[str, int], str] = {}
    for key, rows in grouped.items():
        if key.semantic_contract != SemanticContract.FAST or key.m != 1:
            continue
        concrete_rows: dict[str, list[NativeVNNIObservation]] = defaultdict(list)
        for row in rows:
            if row.candidate_id == row.effective_candidate_id:
                concrete_rows[row.effective_candidate_id].append(row)
        for formula, candidate_policy_hash, formula_config in formulas:
            resolution_key = (formula.candidate_id, key.k)
            concrete_id = resolved_ids.get(resolution_key)
            if concrete_id is None:
                concrete_id = resolve_rocm_concrete_candidate_id(formula, key.k)
                resolved_ids[resolution_key] = concrete_id
            for source in concrete_rows.get(concrete_id, ()):
                projected.append(_clone_validated_observation(
                    source,
                    candidate_id=formula.candidate_id,
                    candidate_family=formula.candidate_family,
                    config_json=formula_config,
                    generic_eligible=True,
                    arithmetic_fingerprint=formula.arithmetic_fingerprint,
                    candidate_policy_hash=candidate_policy_hash,
                ))

    return ObservationCorpus._from_validated(
        projected,
        distinguish_execution_mode=distinguish_execution_mode,
        revalidate_candidate_identities=False,
        known_generic_domain=known_generic_domain,
    )
