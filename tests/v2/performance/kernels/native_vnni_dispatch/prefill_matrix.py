"""Canonical model-tiered NativeVNNI ordinary-prefill measurement matrices.

Ordinary prefill does not project every prompt row through the LM head, so LM
head geometries belong to decode/GEMV training but are intentionally absent
here. Every released Qwen 3.5/3.6 dense and MoE text-backbone projection is an
explicit production anchor on every backend. CPU timing keeps dense 27B and
the very large 122B/397B MoE releases at M=1024, while smaller dense releases
and the actively used 35B-A3B MoE retain deeper coverage. CUDA and ROCm extend
large-release evidence through M=8192 because those launches remain economical.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache

from .qwen_release_geometry import qwen_release_geometries
from .shape_manifest import (
    NativeVNNIShape,
    NativeVNNIShapeManifest,
    ShapeRole,
    load_shape_manifest,
)


PREFILL_M_BUCKETS = (64, 256, 1024, 2048, 4096, 8192, 16384)

# Ordinary prefill never projects through the vocabulary-sized LM head.  This
# separate ceiling therefore ends at Qwen2.5 32B's largest FFN projection while
# the shared CPU shape manifest remains broad enough to train decode/GEMV.
CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS = 27648 * 5120

SMALL_MODEL_SHAPES = (
    "0.5B_AttnOut", "0.5B_QKV", "0.5B_FFN_Up", "0.5B_FFN_Dn",
    "1.5B_AttnOut", "1.5B_QKV", "1.5B_FFN_Up", "1.5B_FFN_Dn",
    "3B_AttnOut", "3B_QKV", "3B_FFN_Up", "3B_FFN_Dn",
    "7B_AttnOut", "7B_QKV", "7B_FFN_Up", "7B_FFN_Dn",
)

MID_MODEL_SHAPES = (
    "14B_FFN_Up",
    "14B_FFN_Dn",
    # Qwen2.5 14B and 32B share d_model/head geometry.
    "32B_AttnOut",
    "32B_QKV",
)

LARGE_MODEL_SHAPES = (
    "32B_FFN_Up",
    "32B_FFN_Dn",
)

# These are the exact NativeVNNI projections exercised by the actively used
# Qwen 3.6 35B-A3B MoE model. Expert matrices are small because each routed
# expert has a narrow intermediate width; the recurrent GDN projections carry
# the model-width work. All remain below the small-model economy envelope, so
# measuring the complete M inventory is both useful and affordable.
QWEN36_35B_MOE_SHAPES = (
    "35BMoE_Expert_GateUp",
    "35BMoE_Expert_Down",
    "Qwen36MoE_GDN_QKVProjection",
    "Qwen36MoE_GDN_ZProjection",
)


CPU_QWEN_RELEASE_MAXIMUM_M = {
    "Qwen3.5-0.8B": 16384,
    "Qwen3.5-2B": 16384,
    "Qwen3.5-4B": 16384,
    "Qwen3.5-9B": 16384,
    "Qwen3.5-27B": 1024,
    "Qwen3.5-35B-A3B": 16384,
    "Qwen3.5-122B-A10B": 1024,
    "Qwen3.5-397B-A17B": 1024,
    "Qwen3.6-27B": 1024,
    "Qwen3.6-35B-A3B": 16384,
}

GPU_QWEN_RELEASE_MAXIMUM_M = {
    release_id: (16384 if maximum_m == 16384 else 8192)
    for release_id, maximum_m in CPU_QWEN_RELEASE_MAXIMUM_M.items()
}


@dataclass(frozen=True)
class CPUPrefillMeasurement:
    """One real projection geometry and one backend's prefill depths."""

    shape: NativeVNNIShape
    m_values: tuple[int, ...]

    @property
    def maximum_m(self) -> int:
        """Return the largest measured prefill row count."""

        return self.m_values[-1]


def _through(maximum_m: int) -> tuple[int, ...]:
    values = tuple(m for m in PREFILL_M_BUCKETS if m <= maximum_m)
    if not values or values[-1] != maximum_m:
        raise ValueError(f"prefill maximum M={maximum_m} is not a bucket")
    return values


def _qwen_release_prefill_requests(
    manifest: NativeVNNIShapeManifest,
    maximum_m_by_release: dict[str, int],
) -> tuple[tuple[str, int], ...]:
    """Resolve deduplicated released-model GEMM geometries to shape names.

    A geometry can be shared by several releases and projection roles. The
    broadest approved M range wins because runtime dispatch is geometry-only.
    LM-head-only geometries are omitted: ordinary prefill does not materialize
    logits for every prompt row.
    """

    production_by_dimensions = {
        (shape.n, shape.k): shape
        for shape in manifest.shapes
        if shape.role == ShapeRole.PRODUCTION and shape.exact_overlay
    }
    requested = []
    for geometry in qwen_release_geometries():
        prefill_uses = tuple(
            use for use in geometry.uses if use.projection != "lm_head"
        )
        if not prefill_uses:
            continue
        shape = production_by_dimensions[(geometry.n, geometry.k)]
        maximum_m = max(
            maximum_m_by_release[use.release_id] for use in prefill_uses
        )
        requested.append((shape.name, maximum_m))
    return tuple(requested)


def _merge_requests(
    *groups: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, int], ...]:
    """Merge stable shape requests while retaining the greatest M ceiling."""

    maximum_by_name: dict[str, int] = {}
    for group in groups:
        for name, maximum_m in group:
            maximum_by_name[name] = max(
                maximum_m,
                maximum_by_name.get(name, 0),
            )
    return tuple(maximum_by_name.items())


@lru_cache(maxsize=1)
def cpu_prefill_measurements() -> tuple[CPUPrefillMeasurement, ...]:
    """Resolve and validate the complete checked-in CPU prefill matrix."""

    manifest = load_shape_manifest()
    base_requested = (
        *((name, 16384) for name in SMALL_MODEL_SHAPES),
        *((name, 16384) for name in QWEN36_35B_MOE_SHAPES),
        *((name, 4096) for name in MID_MODEL_SHAPES),
        *((name, 1024) for name in LARGE_MODEL_SHAPES),
    )
    requested = _merge_requests(
        base_requested,
        _qwen_release_prefill_requests(
            manifest,
            CPU_QWEN_RELEASE_MAXIMUM_M,
        ),
    )
    names = [name for name, _ in requested]
    if len(names) != len(set(names)):
        raise ValueError("CPU prefill matrix contains duplicate shape names")

    measurements = []
    for name, maximum_m in requested:
        shape = manifest.by_name(name)
        if shape.role != ShapeRole.PRODUCTION or not shape.exact_overlay:
            raise ValueError(f"{name}: prefill shape must be a production overlay")
        if "LM_Head" in name:
            raise ValueError(f"{name}: LM heads are not ordinary prefill GEMMs")
        if shape.work_items > CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS:
            raise ValueError(f"{name}: exceeds the CPU measurement envelope")
        measurements.append(CPUPrefillMeasurement(shape, _through(maximum_m)))
    return tuple(measurements)


@lru_cache(maxsize=len(PREFILL_M_BUCKETS))
def cpu_prefill_maximum_weight_elements_for_m(m: int) -> int:
    """Return the largest CPU projection that may be freshly timed at ``m``.

    Adaptive refinement must obey the same economy contract as the reviewed
    production matrix.  In particular, a synthetic geometry below the global
    Qwen2.5 32B weight-size ceiling is not permission to launch a 32B-class
    GEMM at M=16384.  Deriving this limit from the checked-in matrix keeps the
    policy in one place: small-model depths extend through M=16384, 14B and
    shared 32B attention depths extend through M=4096, and the largest 32B FFN
    projections stop at M=1024.

    Args:
        m: One exact canonical prefill row-count bucket.

    Returns:
        The greatest ``N * K`` represented by a production measurement that
        explicitly includes ``m``.

    Raises:
        ValueError: If ``m`` is not one of the canonical prefill buckets.
    """

    if m not in PREFILL_M_BUCKETS:
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

    GPU collection measures every 14B and 32B production projection through
    M=8192. Smaller models retain M=16384 so dispatch sees the large-batch
    transition where arithmetic intensity and occupancy can change.
    """

    manifest = load_shape_manifest()
    base_requested = (
        *((name, 16384) for name in SMALL_MODEL_SHAPES),
        *((name, 16384) for name in QWEN36_35B_MOE_SHAPES),
        *((name, 8192) for name in MID_MODEL_SHAPES),
        *((name, 8192) for name in LARGE_MODEL_SHAPES),
    )
    requested = _merge_requests(
        base_requested,
        _qwen_release_prefill_requests(
            manifest,
            GPU_QWEN_RELEASE_MAXIMUM_M,
        ),
    )
    names = [name for name, _ in requested]
    if len(names) != len(set(names)):
        raise ValueError("GPU prefill matrix contains duplicate shape names")

    measurements = []
    for name, maximum_m in requested:
        shape = manifest.by_name(name)
        if shape.role != ShapeRole.PRODUCTION or not shape.exact_overlay:
            raise ValueError(f"{name}: prefill shape must be a production overlay")
        if "LM_Head" in name:
            raise ValueError(f"{name}: LM heads are not ordinary prefill GEMMs")
        if shape.work_items > manifest.maximum_supported_weight_elements:
            raise ValueError(f"{name}: exceeds the supported runtime envelope")
        measurements.append(CPUPrefillMeasurement(shape, _through(maximum_m)))
    return tuple(measurements)


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
