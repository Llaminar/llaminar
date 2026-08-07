"""Canonical model-tiered NativeVNNI ordinary-prefill measurement matrices.

Ordinary prefill does not project every prompt row through the LM head, so LM
head geometries belong to decode/GEMV training but are intentionally absent
here. Every non-LM-head production exact geometry is an explicit anchor on
every backend. CPU uses model-tiered depth ceilings to keep expensive evidence
collection finite, while GPU retains the broader launch inventory. Generic
geometry rules continue to own every positive M that is not an exact measured
cell, including values beyond the largest measurement bucket.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

from .qwen_release_geometry import QWEN_RELEASE_MODELS, qwen_release_geometries
from .qwen_moe_gguf_patterns import (
    QwenMoEPrefillMixture,
    qwen_moe_prefill_mixtures,
)
from .shape_manifest import (
    NativeVNNIShape,
    NativeVNNIShapeManifest,
    ShapeRole,
    load_shape_manifest,
)


# The generic CPU policy owns the union of all three model-tier inventories.
# Exact overlays deliberately spend less measurement time as model size grows:
# large CPU projections reach steady state quickly, so deeper rows add little
# evidence while consuming most of the collection budget. Keeping the union
# explicit is important: runtime bucketing, generic-tree totality, and sealed
# certification must still cover every positive M even when one exact overlay
# is measured at only one canonical anchor.
CPU_BELOW_7B_PREFILL_M_BUCKETS = (32, 128)
CPU_7B_TO_BELOW_14B_PREFILL_M_BUCKETS = (32, 64)
CPU_14B_PLUS_PREFILL_M_BUCKETS = (32,)
CPU_PREFILL_M_BUCKETS = tuple(sorted(set(
    CPU_BELOW_7B_PREFILL_M_BUCKETS
    + CPU_7B_TO_BELOW_14B_PREFILL_M_BUCKETS
    + CPU_14B_PLUS_PREFILL_M_BUCKETS
)))
_REPO_ROOT = Path(__file__).resolve().parents[5]
_PREFILL_BUCKET_DEFINITION = (
    _REPO_ROOT / "src/v2/utils/PrefillGraphBuckets.def"
)
_PREFILL_BUCKET_PATTERN = re.compile(
    r"^LLAMINAR_PREFILL_GRAPH_BUCKET\(([1-9][0-9]*)\)$"
)


def _load_gpu_prefill_m_buckets() -> tuple[int, ...]:
    """Read the exact physical graph buckets shared with the C++ runtime."""

    buckets = tuple(
        int(match.group(1))
        for line in _PREFILL_BUCKET_DEFINITION.read_text(
            encoding="utf-8"
        ).splitlines()
        if (match := _PREFILL_BUCKET_PATTERN.fullmatch(line.strip()))
    )
    if not buckets:
        raise ValueError(
            f"{_PREFILL_BUCKET_DEFINITION}: no prefill graph buckets"
        )
    if tuple(sorted(set(buckets))) != buckets:
        raise ValueError(
            f"{_PREFILL_BUCKET_DEFINITION}: buckets must be unique and ordered"
        )
    return buckets


# Long contexts replay these physical captures in chunks. Evidence for virtual
# M=8192 or M=16384 would train launches that production can never select.
GPU_PREFILL_M_BUCKETS = _load_gpu_prefill_m_buckets()

# Versions 8 through 10 of the immutable CPU split used this older, virtual-M
# GPU schedule. Keep its identity explicit for historical artifact readers;
# changing the live graph buckets must never reinterpret a signed corpus.
LEGACY_GPU_PREFILL_M_BUCKETS = (
    64, 256, 1024, 2048, 4096, 8192, 16384
)

# Historical CPU certificates legitimately contain these older buckets.  They
# remain readable as immutable evidence, but new CPU collection and generated
# runtime bucketing use ``CPU_PREFILL_M_BUCKETS`` exclusively.
HISTORICAL_CPU_PREFILL_M_BUCKETS = tuple(sorted(set(
    LEGACY_GPU_PREFILL_M_BUCKETS + (128, 512)
)))
SUPPORTED_CPU_PREFILL_M_BUCKETS = tuple(sorted(set(
    CPU_PREFILL_M_BUCKETS + HISTORICAL_CPU_PREFILL_M_BUCKETS
)))
CPU_MIDDLE_MODEL_THRESHOLD_BILLIONS = 7.0
CPU_LARGE_MODEL_THRESHOLD_BILLIONS = 14.0

_FIXED_MODEL_FAMILY_SIZE_BILLIONS = {
    "qwen36-dense": 27.0,
    "qwen36-moe": 35.0,
}

# Ordinary prefill never projects through the vocabulary-sized LM head.  This
# separate ceiling therefore ends at Qwen2.5 32B's largest FFN projection while
# the shared CPU shape manifest remains broad enough to train decode/GEMV.
CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS = 27648 * 5120

# These are the exact NativeVNNI projections exercised by the actively used
# Qwen 3.6 35B-A3B MoE model. Expert matrices are small because each routed
# expert has a narrow intermediate width; the recurrent GDN projections carry
# the model-width work. All remain below the small-model economy envelope, so
# these projections belong to a 35B release and therefore use the deliberately
# short large-model inventory despite their individually modest dimensions.
QWEN36_35B_MOE_SHAPES = (
    "35BMoE_Expert_GateUp",
    "35BMoE_Expert_Down",
    "Qwen36MoE_GDN_QKVProjection",
    "Qwen36MoE_GDN_ZProjection",
)


@lru_cache(maxsize=1)
def _largest_release_owner_by_dimensions() -> dict[tuple[int, int], float]:
    """Map each release geometry to its largest owning model.

    Several releases may share one ``(N, K)`` pair. The largest owner controls
    the evidence budget so a geometry used by both a small and a very large
    checkpoint cannot accidentally inherit the small-model schedule.
    """

    model_sizes = {
        model.release_id: model.parameter_count_billions
        for model in QWEN_RELEASE_MODELS
    }
    return {
        (geometry.n, geometry.k): max(
            model_sizes[use.release_id] for use in geometry.uses
        )
        for geometry in qwen_release_geometries()
    }


def _shape_owner_size_billions(shape: NativeVNNIShape) -> float:
    """Return the largest model size that owns one exact-overlay geometry."""

    fixed_size = _FIXED_MODEL_FAMILY_SIZE_BILLIONS.get(shape.model_family)
    if fixed_size is not None:
        return fixed_size
    if shape.model_family.startswith("qwen25-"):
        size_text = shape.model_family.removeprefix("qwen25-").removesuffix("b")
        try:
            return float(size_text)
        except ValueError as error:
            raise ValueError(
                f"{shape.name}: malformed Qwen2.5 model-family size"
            ) from error
    if shape.model_family == "qwen35-qwen36-release-geometries":
        try:
            return _largest_release_owner_by_dimensions()[(shape.n, shape.k)]
        except KeyError as error:
            raise ValueError(
                f"{shape.name}: release geometry has no owning model"
            ) from error
    raise ValueError(
        f"{shape.name}: production prefill model family has no CPU size tier"
    )


def _cpu_prefill_m_values_for_shape(
    shape: NativeVNNIShape,
) -> tuple[int, ...]:
    """Select the reviewed exact-overlay M inventory for one model owner."""

    owner_size = _shape_owner_size_billions(shape)
    if owner_size >= CPU_LARGE_MODEL_THRESHOLD_BILLIONS:
        return CPU_14B_PLUS_PREFILL_M_BUCKETS
    if owner_size >= CPU_MIDDLE_MODEL_THRESHOLD_BILLIONS:
        return CPU_7B_TO_BELOW_14B_PREFILL_M_BUCKETS
    return CPU_BELOW_7B_PREFILL_M_BUCKETS


@dataclass(frozen=True)
class CPUPrefillMeasurement:
    """One real projection geometry and one backend's prefill depths."""

    shape: NativeVNNIShape
    m_values: tuple[int, ...]

    @property
    def maximum_m(self) -> int:
        """Return the largest measured prefill row count."""

        return self.m_values[-1]


@dataclass(frozen=True)
class MoEPrefillMixtureMeasurement:
    """One concrete production MoE codebook mixture and measured M buckets."""

    mixture: QwenMoEPrefillMixture
    m_values: tuple[int, ...]

    @property
    def maximum_m(self) -> int:
        """Return the largest measured grouped-prefill row count."""

        return self.m_values[-1]


def _ordinary_prefill_exact_shapes(
    manifest: NativeVNNIShapeManifest,
) -> tuple[NativeVNNIShape, ...]:
    """Return every exact production geometry used by ordinary prefill.

    A release-catalog geometry may have several model/projection aliases. It
    remains prefill-applicable when at least one alias is not an LM head. The
    manually declared Qwen2.5/Qwen3.6 LM-head rows are excluded by name for the
    same semantic reason. Backend-specific planning applies the CPU model-size
    tier only after this shared geometry inventory has been resolved.
    """

    release_prefill_dimensions = {
        (geometry.n, geometry.k)
        for geometry in qwen_release_geometries()
        if any(use.projection != "lm_head" for use in geometry.uses)
    }
    result = []
    for shape in manifest.shapes:
        if shape.role != ShapeRole.PRODUCTION or not shape.exact_overlay:
            continue
        if "LM_Head" in shape.name:
            continue
        if (
            shape.model_family == "qwen35-qwen36-release-geometries"
            and (shape.n, shape.k) not in release_prefill_dimensions
        ):
            continue
        result.append(shape)
    return tuple(result)


@lru_cache(maxsize=1)
def cpu_prefill_measurements() -> tuple[CPUPrefillMeasurement, ...]:
    """Resolve and validate the complete checked-in CPU prefill matrix."""

    manifest = load_shape_manifest()
    requested = _ordinary_prefill_exact_shapes(manifest)
    names = [shape.name for shape in requested]
    if len(names) != len(set(names)):
        raise ValueError("CPU prefill matrix contains duplicate shape names")

    measurements = []
    for shape in requested:
        name = shape.name
        if shape.role != ShapeRole.PRODUCTION or not shape.exact_overlay:
            raise ValueError(f"{name}: prefill shape must be a production overlay")
        if "LM_Head" in name:
            raise ValueError(f"{name}: LM heads are not ordinary prefill GEMMs")
        if shape.work_items > CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS:
            raise ValueError(f"{name}: exceeds the CPU measurement envelope")
        m_values = _cpu_prefill_m_values_for_shape(shape)
        measurements.append(CPUPrefillMeasurement(shape, m_values))
    return tuple(measurements)


@lru_cache(maxsize=len(CPU_PREFILL_M_BUCKETS))
def cpu_prefill_maximum_weight_elements_for_m(m: int) -> int:
    """Return the largest CPU projection that may be freshly timed at ``m``.

    Adaptive refinement obeys the same model-tiered exact-overlay envelope as
    the reviewed production matrix. M=32 may use every production geometry,
    M=64 is bounded by 7B-to-below-14B owners, and M=128 is bounded by owners
    below 7B.

    Args:
        m: One exact canonical prefill row-count bucket.

    Returns:
        The greatest ``N * K`` represented by a production measurement that
        explicitly includes ``m``.

    Raises:
        ValueError: If ``m`` is not one of the canonical prefill buckets.
    """

    if m not in CPU_PREFILL_M_BUCKETS:
        raise ValueError(f"M={m} is not a canonical CPU prefill bucket")
    eligible = tuple(
        measurement.shape.work_items
        for measurement in cpu_prefill_measurements()
        if m in measurement.m_values
    )
    if not eligible:
        raise ValueError(f"M={m} has no canonical CPU prefill measurement")
    return max(eligible)


@lru_cache(maxsize=1)
def gpu_prefill_measurements() -> tuple[CPUPrefillMeasurement, ...]:
    """Resolve the shared CUDA/ROCm prefill matrix.

    CUDA and ROCm consume the same comprehensive exact-overlay matrix as CPU.
    """

    manifest = load_shape_manifest()
    requested = _ordinary_prefill_exact_shapes(manifest)
    names = [shape.name for shape in requested]
    if len(names) != len(set(names)):
        raise ValueError("GPU prefill matrix contains duplicate shape names")

    measurements = []
    for shape in requested:
        name = shape.name
        if shape.role != ShapeRole.PRODUCTION or not shape.exact_overlay:
            raise ValueError(f"{name}: prefill shape must be a production overlay")
        if "LM_Head" in name:
            raise ValueError(f"{name}: LM heads are not ordinary prefill GEMMs")
        if shape.work_items > manifest.maximum_supported_weight_elements:
            raise ValueError(f"{name}: exceeds the supported runtime envelope")
        measurements.append(CPUPrefillMeasurement(shape, GPU_PREFILL_M_BUCKETS))
    return tuple(measurements)


@lru_cache(maxsize=3)
def moe_prefill_mixture_measurements(
    backend: str,
) -> tuple[MoEPrefillMixtureMeasurement, ...]:
    """Return the all-format production MoE mixture matrix for one backend.

    This matrix is additive to ordinary per-projection measurements. It times
    the complete routed/shared role tuple so a locally optimal gate tile cannot
    hide an expensive directory rebuild or interaction with the down/shared
    path. CPU retains the reviewed large-model M=32 budget; CUDA and ROCm own
    every canonical GPU prefill bucket.
    """

    normalized = backend.lower()
    if normalized == "cpu":
        m_values = CPU_14B_PLUS_PREFILL_M_BUCKETS
    elif normalized in {"cuda", "rocm"}:
        m_values = GPU_PREFILL_M_BUCKETS
    else:
        raise ValueError(f"unknown NativeVNNI backend {backend!r}")
    return tuple(
        MoEPrefillMixtureMeasurement(mixture, m_values)
        for mixture in qwen_moe_prefill_mixtures()
    )


def main() -> int:
    """Expose stable tab-separated records to the shell refresh transaction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        action="store_true",
        help="Print name, N, K, and comma-separated M buckets",
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Select the backend-specific economy envelope (default: cpu)",
    )
    args = parser.parse_args()
    if not args.records:
        parser.error("use --records")
    measurements = (
        cpu_prefill_measurements()
        if args.backend == "cpu"
        else gpu_prefill_measurements()
    )
    for measurement in measurements:
        print(
            f"{measurement.shape.name}\t{measurement.shape.n}\t"
            f"{measurement.shape.k}\t"
            f"{','.join(str(m) for m in measurement.m_values)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
