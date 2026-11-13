"""Select requested modern-model sweep work without changing runtime support.

Canonical production names can belong to an older model even when their matrix
is also used by a modern release. Selection therefore follows declared release
uses and physical dimensions, then keeps the original canonical name. It never
rewrites producer identity, certifies evidence, or removes installed policy keys.
QFN's explicit qwen4exp declarations remain in scope independently of versioned
release IDs; this module does not rename that family or invent release geometry.
"""
from __future__ import annotations

from typing import Iterable

from .qwen_release_geometry import qwen_release_geometries
from .shape_manifest import (
    ExpertProjectionGeometry, NativeQuantisedShapeManifest, ShapeRole,
    expert_projection_geometries, load_shape_manifest,
)


_MODERN_FAMILY_PREFIXES = (
    "qwen35", "qwen36", "qwen38", "qwen3.5", "qwen3.6", "qwen3.8", "qwen4exp",
)


def is_modern_qwen_family(identity: str) -> bool:
    """Recognize declared 3.5/3.6/3.8 and QFN identities at exact boundaries.

    This accepts existing normalized model-family tags and release IDs, rather
    than interpreting arbitrary repository paths or guessing a version from
    tensor dimensions. Qwen2.5 and original Qwen3 are outside this sweep scope.
    """
    if not isinstance(identity, str):
        raise TypeError("model-family identity must be a string")
    normalized = identity.strip().lower()
    return any(normalized == prefix or normalized.startswith((prefix + "-", prefix + "_"))
               for prefix in _MODERN_FAMILY_PREFIXES)


def modern_sweep_shape_names(
    manifest: NativeQuantisedShapeManifest | None = None, *,
    candidate_names: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Project new model-only work onto original canonical production shapes.

    A physical (N,K) remains selected when any modern release owns it, even if
    deduplication retained a Qwen2.5/Qwen3 name. Explicit modern declarations
    also own their participant/shard geometries, including all QFN rows. The
    result preserves caller/manifest order and executes each physical shape
    once. Certification grids and legacy-only shapes are not new model jobs.

    The source manifest and its digest remain unchanged. Unknown requested
    names still raise through the original manifest resolver. This projection
    makes no statement about reuse eligibility: original actual architecture,
    ABI, numerical policy, producer and full-grade evidence stay with existing
    collection/admission owners.
    """
    source = manifest if manifest is not None else load_shape_manifest()
    modern_dimensions = {
        (shape.n, shape.k) for shape in source.shapes
        if shape.role == ShapeRole.PRODUCTION and is_modern_qwen_family(shape.model_family)
    }
    modern_dimensions.update(
        (geometry.n, geometry.k) for geometry in qwen_release_geometries()
        if any(is_modern_qwen_family(use.release_id) for use in geometry.uses)
    )
    names = source.names(production_only=True) if candidate_names is None else candidate_names
    selected: list[str] = []
    seen: set[tuple[int, int]] = set()
    for name in names:
        shape = source.by_name(name)
        physical = shape.n, shape.k
        if shape.role == ShapeRole.PRODUCTION and physical in modern_dimensions and physical not in seen:
            selected.append(shape.name)
            seen.add(physical)
    return tuple(selected)


def modern_expert_projection_geometries(
    projections: Iterable[ExpertProjectionGeometry] | None = None,
) -> tuple[ExpertProjectionGeometry, ...]:
    """Keep modern/shared physical-arithmetic expert owners with provenance.

    Role, physical dimensions and arithmetic policy width match the existing
    CPU expert grouping. This is an inclusion projection, not a deduplication
    of source/layout/participant records: every original owner of a retained
    key stays intact, including an old-named owner shared with a modern model.
    Collection and evidence admission still own all other job discriminators.
    """
    source = tuple(expert_projection_geometries() if projections is None else projections)
    def key(projection: ExpertProjectionGeometry) -> tuple[object, ...]:
        return projection.role, projection.physical_n, projection.physical_k, projection.policy_n
    retained = {key(projection) for projection in source if is_modern_qwen_family(projection.model_family)}
    return tuple(projection for projection in source if key(projection) in retained)
