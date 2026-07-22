"""Profiler-informed candidate regret modeling for NativeVNNI policy fitting.

Hardware counters are development-only explanatory features. They are joined
to immutable canonical timing observations through authenticated profiler
request/evidence manifests, summarized per physical launch candidate, and used
to train a deterministic nonlinear surrogate inside each grouped CV fold.
Measured latency remains authoritative for exact winners, the hard
five-percent topology, held-out CV, and sealed promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from .candidate_observation import read_observation_csv
from .corpus import ObservationCorpus, RuntimeKey
from .format_registry import format_spec, runtime_aliases
from .profiler_evidence import (
    ProfilerFeatureRow,
    _equal_ranges,
    _offline_worker_count,
    _physical_core_count,
    profiler_feature_rows,
    read_profiler_evidence_manifest,
    read_profiler_request_manifest,
)
from .schema import Backend, ExecutionMode, NativeVNNIObservation


LEGACY_PROFILER_FEATURE_CATALOG_CACHE_VERSION = (
    "native-vnni-profiler-catalog-cache-v1"
)
PROFILER_FEATURE_CATALOG_CACHE_VERSION = (
    "native-vnni-profiler-catalog-cache-v2-jsonl"
)


def _file_sha256(path: Path) -> str:
    """Hash one canonical source artifact without loading it into Python memory."""

    with path.open("rb") as handle:
        return "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()


def _cache_payload_digest(payload: Mapping[str, object]) -> str:
    """Return the canonical identity of one derived catalog-cache payload."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


PROFILER_CATALOG_NORMALIZATION_VERSION = (
    "native-vnni-profiler-extra-trees-v16-calibrated-hardware-targets"
)
PROFILER_SURROGATE_VERSION = (
    "native-vnni-profiler-xgboost-cuda-hist-v21-vectorized-targets"
)
PROFILER_XGBOOST_VERSION = "3.1.3"
PROFILER_GPU_HIST_TARGET_PROJECTIONS = 4
PROFILER_GPU_HIST_BOOST_ROUNDS = 8
PROFILER_GPU_HIST_MAX_DEPTH = 8
PROFILER_GPU_HIST_MIN_CHILD_WEIGHT = 3.0
PROFILER_GPU_HIST_MAX_BIN = 256
PROFILER_GPU_HIST_LEARNING_RATE = 0.25
PROFILER_GPU_HIST_COLSAMPLE_BYNODE = 0.25
PROFILER_GPU_HIST_RANDOM_STATE = 19348
PROFILER_AUXILIARY_TARGET_TOTAL_WEIGHT = 0.5


def _profiler_gpu_quantile_threads() -> int:
    """Divide affinity-visible physical cores among active CUDA fit workers."""

    explicit = os.environ.get(
        "LLAMINAR_NATIVE_VNNI_PROFILER_GPU_PREPROCESS_THREADS"
    )
    if explicit is not None:
        threads = int(explicit)
        if threads < 1:
            raise ValueError(
                "profiler GPU preprocessing thread count must be positive"
            )
        return min(threads, _physical_core_count())
    raw_devices = os.environ.get(
        "LLAMINAR_NATIVE_VNNI_PROFILER_MODEL_CUDA_DEVICES"
    )
    if raw_devices:
        device_count = len({
            item.strip() for item in raw_devices.split(",") if item.strip()
        })
    else:
        device_count = len(tuple(Path("/dev").glob("nvidia[0-9]*")))
    return max(1, _physical_core_count() // max(1, device_count))


def _profiler_surrogate_identity() -> tuple[str, ...]:
    """Return every hyperparameter that can change surrogate predictions.

    Catalog normalization and surrogate fitting deliberately have independent
    identities. Changing the forest must invalidate cached prediction surfaces,
    while leaving the expensive authenticated profiler catalog reusable.
    """

    return (
        PROFILER_SURROGATE_VERSION,
        f"xgboost={PROFILER_XGBOOST_VERSION}",
        "tree_method=hist",
        "multi_strategy=one_output_per_tree",
        f"target_projections={PROFILER_GPU_HIST_TARGET_PROJECTIONS}",
        f"boost_rounds={PROFILER_GPU_HIST_BOOST_ROUNDS}",
        f"max_depth={PROFILER_GPU_HIST_MAX_DEPTH}",
        f"min_child_weight={PROFILER_GPU_HIST_MIN_CHILD_WEIGHT:.17g}",
        f"max_bin={PROFILER_GPU_HIST_MAX_BIN}",
        f"learning_rate={PROFILER_GPU_HIST_LEARNING_RATE:.17g}",
        f"colsample_bynode={PROFILER_GPU_HIST_COLSAMPLE_BYNODE:.17g}",
        f"random_state={PROFILER_GPU_HIST_RANDOM_STATE}",
        f"quantile_threads={_profiler_gpu_quantile_threads()}",
        (
            "auxiliary_target_total_weight="
            f"{PROFILER_AUXILIARY_TARGET_TOTAL_WEIGHT:.17g}"
        ),
    )


def _profiler_surrogate_digest(model_input_digest: str) -> str:
    """Bind normalized model inputs to the exact surrogate implementation."""

    encoded = "\0".join((model_input_digest, *_profiler_surrogate_identity()))
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def _mixed_profiler_targets(targets: np.ndarray) -> np.ndarray:
    """Project profiler targets into bounded regret-coupled objectives.

    Stable XGBoost supports CUDA histogram training for one scalar tree family
    per output, but not CUDA vector leaves. The deterministic transform keeps
    every scalar objective coupled to standardized regret and a balanced
    projection of the profiler targets. Each auxiliary column is assigned one
    of three zero-sum sign patterns. The output mean is therefore exactly the
    original regret target, while the fixed projection count prevents training
    cost from growing with every newly collected hardware counter.
    """

    values = np.asarray(targets, dtype=np.float32, order="C")
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("profiler surrogate requires regret and auxiliary targets")
    projection_count = PROFILER_GPU_HIST_TARGET_PROJECTIONS
    if projection_count != 4:
        raise ValueError("profiler target projection currently requires four outputs")
    patterns = np.asarray((
        (1.0, 1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0, -1.0),
        (1.0, -1.0, -1.0, 1.0),
    ), dtype=np.float32)
    auxiliary = values[:, 1:]
    signs = patterns[np.arange(auxiliary.shape[1]) % len(patterns)]
    deviations = auxiliary @ signs
    deviations *= np.float32(1.0 / math.sqrt(auxiliary.shape[1]))
    mixed = values[:, :1] + deviations
    # Derive the final column from the first three in float32 so the arithmetic
    # mean reconstructs regret independently of projection rounding.
    mixed[:, -1] = (
        np.float32(projection_count) * values[:, 0]
        - np.sum(mixed[:, :-1], axis=1, dtype=np.float32)
    )
    return mixed


def _fit_profiler_surrogate(
    training_matrix: np.ndarray,
    targets: np.ndarray,
    prediction_matrix: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    """Fit one deterministic GPU histogram surrogate and predict regret.

    Production callers pass an explicit CUDA device. Unit tests may pass
    ``cpu`` to exercise identical XGBoost model semantics without GPU work;
    there is no runtime device fallback and an unavailable requested GPU is a
    hard fit failure.
    """

    try:
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError(
            "profiler fitting requires xgboost=="
            f"{PROFILER_XGBOOST_VERSION} with CUDA support"
        ) from error
    if xgb.__version__ != PROFILER_XGBOOST_VERSION:
        raise RuntimeError(
            "profiler fitting requires xgboost=="
            f"{PROFILER_XGBOOST_VERSION}, found {xgb.__version__}"
        )

    mixed_targets = _mixed_profiler_targets(targets)
    quantile_threads = _profiler_gpu_quantile_threads()
    training_data = xgb.QuantileDMatrix(
        np.asarray(training_matrix, dtype=np.float32, order="C"),
        mixed_targets,
        max_bin=PROFILER_GPU_HIST_MAX_BIN,
        nthread=quantile_threads,
    )
    booster = xgb.train(
        {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": device,
            "fail_on_invalid_gpu_id": True,
            "multi_strategy": "one_output_per_tree",
            "max_depth": PROFILER_GPU_HIST_MAX_DEPTH,
            "min_child_weight": PROFILER_GPU_HIST_MIN_CHILD_WEIGHT,
            "max_bin": PROFILER_GPU_HIST_MAX_BIN,
            "eta": PROFILER_GPU_HIST_LEARNING_RATE,
            "subsample": 1.0,
            "colsample_bynode": PROFILER_GPU_HIST_COLSAMPLE_BYNODE,
            "seed": PROFILER_GPU_HIST_RANDOM_STATE,
            "nthread": 1,
            "verbosity": 0,
        },
        training_data,
        num_boost_round=PROFILER_GPU_HIST_BOOST_ROUNDS,
    )
    configured_device = json.loads(booster.save_config())[
        "learner"
    ]["generic_param"]["device"]
    if device.startswith("cuda") and configured_device != device:
        raise RuntimeError(
            "profiler surrogate changed requested GPU device: "
            f"requested={device}, configured={configured_device}"
        )
    prediction_data = xgb.QuantileDMatrix(
        np.asarray(prediction_matrix, dtype=np.float32, order="C"),
        ref=training_data,
        max_bin=PROFILER_GPU_HIST_MAX_BIN,
        nthread=quantile_threads,
    )
    mixed_predictions = np.asarray(
        booster.predict(prediction_data),
        dtype=np.float32,
    )
    if mixed_predictions.ndim == 1:
        mixed_predictions = mixed_predictions[:, np.newaxis]
    if mixed_predictions.shape != (
        len(prediction_matrix),
        mixed_targets.shape[1],
    ):
        raise RuntimeError("profiler surrogate returned an invalid prediction shape")
    return np.mean(mixed_predictions, axis=1, dtype=np.float64)

_GPU_STATIC_RESOURCE_METRICS = frozenset((
    "gpu.registers_per_thread",
    "gpu.static_shared_memory_bytes",
    "gpu.dynamic_shared_memory_bytes",
    "gpu.local_memory_bytes_per_thread",
    "gpu.vgpr_count",
    "gpu.sgpr_count",
    "gpu.lds_bytes",
    "gpu.scratch_bytes",
))
_GPU_PERCENT_METRICS = frozenset((
    "gpu.theoretical_occupancy_pct",
    "gpu.achieved_occupancy_pct",
    "gpu.compute_throughput_pct_of_peak",
    "gpu.alu_pipe_utilization_pct",
    "gpu.fma_pipe_utilization_pct",
    "gpu.tensor_pipe_utilization_pct",
    "gpu.dram_throughput_pct_of_peak",
    "gpu.l1_throughput_pct_of_peak",
    "gpu.l2_throughput_pct_of_peak",
    "gpu.gpu_busy_pct",
    "gpu.valu_busy_pct",
    "gpu.valu_utilization_pct",
    "gpu.memory_unit_busy_pct",
    "gpu.memory_unit_stalled_pct",
    "gpu.l2_cache_hit_pct",
    "gpu.lds_bank_conflict_pct",
))
_GPU_RATE_METRICS = frozenset((
    "gpu.executed_ipc_active",
    "gpu.warp_cycles_per_issued_instruction",
))


class CandidateCostLike(Protocol):
    """Structural interface consumed without importing the policy module."""

    runtime_key: RuntimeKey
    shape_group_id: str
    candidate_id: str
    max_surface_regret: float
    profiler_predicted_regret: float


_PARALLEL_MODEL_RECORD_COSTS: tuple[CandidateCostLike, ...] = ()
_PARALLEL_MODEL_RECORD_OBSERVATIONS: ProfilerObservationIndex | None = None
_PARALLEL_MODEL_RECORD_EXEMPLARS: Mapping[
    tuple[object, ...], NativeVNNIObservation
] | None = None
_PARALLEL_MODEL_RECORD_CATALOG: ProfilerFeatureCatalog | None = None
_PARALLEL_PROFILER_DESCRIPTOR_GROUPS: tuple[
    tuple[str, tuple[ProfilerFeatureRow, ...]], ...
] = ()


class ProfilerModelRecordIndex(
    Mapping[tuple[RuntimeKey, str, str], dict[str, float | str]]
):
    """Disk-backed numeric profiler records shared by every CV surface.

    The old index stored one Python dictionary for every candidate point.  A
    production CUDA transfer pool contains more than three million points, so
    repeating roughly two hundred string-keyed fields per point consumed about
    100 GiB before the first forest was fitted.  It also forced workers to
    publish tens of GiB of pickle shards and made each later ``fork()`` spend
    seconds copying page tables.

    This index stores the exact same model inputs once in a C-contiguous
    ``float32`` memmap.  Scikit-learn's tree implementation converts dense
    inputs to ``float32`` internally, so this removes Python-object expansion
    without reducing the precision seen by the estimator.  Raw auxiliary
    profiler targets remain ``float64`` and retain ``NaN`` missingness until a
    particular held-out surface applies its completeness and centering rules.
    Anchor N/K pairs are separate integer columns so coherent-CV masking does
    not require reconstructing a record.

    ``Mapping`` remains implemented for diagnostics and focused tests.  A
    lookup reconstructs one record lazily; production fitting uses aligned row
    indices and never enters that deliberately slower interface.
    """

    def __init__(
        self,
        *,
        costs: tuple[CandidateCostLike, ...],
        observations: ProfilerObservationIndex,
        exemplars: Mapping[tuple[object, ...], NativeVNNIObservation],
        catalog: ProfilerFeatureCatalog,
        directory: Path,
        input_feature_names: tuple[str, ...],
        auxiliary_feature_names: tuple[str, ...],
        unique_point_count: int | None,
        point_rows: dict[tuple[RuntimeKey, str, str], int] | None = None,
    ) -> None:
        self.costs = costs
        self.observations = observations
        self.exemplars = exemplars
        self.catalog = catalog
        self.directory = directory
        self.input_feature_names = input_feature_names
        self.auxiliary_feature_names = auxiliary_feature_names
        self.unique_point_count = unique_point_count
        self.creator_pid = os.getpid()
        self._point_rows = point_rows
        self._input_matrix: np.memmap | None = None
        self._auxiliary_matrix: np.memmap | None = None
        self._anchor_matrix: np.memmap | None = None
        self._contest_ids: np.memmap | None = None
        self._regret_targets: np.memmap | None = None
        self._closed = False

    @property
    def row_count(self) -> int:
        """Return the aligned cost/matrix row count, including repetitions."""

        return len(self.costs)

    @property
    def input_matrix(self) -> np.ndarray:
        """Open the immutable runtime/static model matrix lazily per process."""

        if self.row_count == 0:
            return np.empty(
                (0, len(self.input_feature_names)), dtype=np.float32
            )
        if self._input_matrix is None:
            self._input_matrix = np.memmap(
                self.directory / "model-inputs.f32",
                dtype=np.float32,
                mode="r",
                shape=(self.row_count, len(self.input_feature_names)),
            )
        return self._input_matrix

    @property
    def auxiliary_matrix(self) -> np.ndarray:
        """Open raw calibrated auxiliary targets with ``NaN`` missingness."""

        if not self.auxiliary_feature_names:
            return np.empty((self.row_count, 0), dtype=np.float64)
        if self._auxiliary_matrix is None:
            self._auxiliary_matrix = np.memmap(
                self.directory / "auxiliary-targets.f64",
                dtype=np.float64,
                mode="r",
                shape=(self.row_count, len(self.auxiliary_feature_names)),
            )
        return self._auxiliary_matrix

    @property
    def anchor_matrix(self) -> np.ndarray:
        """Open exact profiler anchor N/K pairs for coherent holdout masking."""

        if self.row_count == 0:
            return np.empty((0, 2), dtype=np.int64)
        if self._anchor_matrix is None:
            self._anchor_matrix = np.memmap(
                self.directory / "anchor-nk.i64",
                dtype=np.int64,
                mode="r",
                shape=(self.row_count, 2),
            )
        return self._anchor_matrix

    @property
    def contest_ids(self) -> np.ndarray:
        """Open stable runtime/shape contest identifiers for vector centering."""

        if self.row_count == 0:
            return np.empty((0,), dtype=np.int64)
        if self._contest_ids is None:
            self._contest_ids = np.memmap(
                self.directory / "contest-ids.i64",
                dtype=np.int64,
                mode="r",
                shape=(self.row_count,),
            )
        return self._contest_ids

    @property
    def regret_targets(self) -> np.ndarray:
        """Open aligned ``log1p(max_surface_regret)`` training labels.

        These labels depend only on the immutable candidate-cost rows, not on
        a fold's held geometry. Matrix workers calculate them once while the
        corresponding records are already hot. Reusing the aligned array
        avoids re-entering millions of Python cost objects for every GPU fit.
        """

        if self.row_count == 0:
            return np.empty((0,), dtype=np.float64)
        if self._regret_targets is None:
            self._regret_targets = np.memmap(
                self.directory / "regret-targets.f64",
                dtype=np.float64,
                mode="r",
                shape=(self.row_count,),
            )
        return self._regret_targets

    def _ensure_point_rows(self) -> dict[tuple[RuntimeKey, str, str], int]:
        """Build the diagnostic point lookup only when Mapping access is used."""

        if self._point_rows is None:
            rows = {}
            for index, cost in enumerate(self.costs):
                rows.setdefault(
                    (cost.runtime_key, cost.shape_group_id, cost.candidate_id),
                    index,
                )
            if (
                self.unique_point_count is not None
                and len(rows) != self.unique_point_count
            ):
                raise RuntimeError(
                    "compact profiler record point count changed after publication"
                )
            self.unique_point_count = len(rows)
            self._point_rows = rows
        return self._point_rows

    def row_indices_for(
        self,
        costs: Iterable[CandidateCostLike],
    ) -> np.ndarray:
        """Resolve arbitrary cost objects through the lazy diagnostic lookup."""

        point_rows = self._ensure_point_rows()
        result = []
        for cost in costs:
            point = (cost.runtime_key, cost.shape_group_id, cost.candidate_id)
            try:
                result.append(point_rows[point])
            except KeyError as error:
                raise ValueError(
                    f"profiler record index cannot resolve candidate point {point}"
                ) from error
        return np.asarray(result, dtype=np.int64)

    def row_indices_for_points(
        self,
        points: Iterable[tuple[RuntimeKey, str, str]],
    ) -> np.ndarray:
        """Resolve a prediction inventory through the shared pool row index."""

        point_rows = self._ensure_point_rows()
        result = []
        for point in points:
            try:
                result.append(point_rows[point])
            except KeyError as error:
                raise ValueError(
                    f"profiler record index cannot resolve candidate point {point}"
                ) from error
        return np.asarray(result, dtype=np.int64)

    def __getitem__(
        self,
        point: tuple[RuntimeKey, str, str],
    ) -> dict[str, float | str]:
        row = self._ensure_point_rows()[point]
        return _profiler_record_for_cost(
            self.costs[row],
            self.observations,
            self.exemplars,
            self.catalog,
        )

    def __iter__(self) -> Iterator[tuple[RuntimeKey, str, str]]:
        return iter(self._ensure_point_rows())

    def __len__(self) -> int:
        if self.unique_point_count is None:
            self._ensure_point_rows()
        if self.unique_point_count is None:
            raise RuntimeError("profiler record point count was not initialized")
        return self.unique_point_count

    def close(self) -> None:
        """Release mappings and remove parent-owned temporary matrix files."""

        if self._closed:
            return
        self._closed = True
        for attribute in (
            "_input_matrix",
            "_auxiliary_matrix",
            "_anchor_matrix",
            "_contest_ids",
            "_regret_targets",
        ):
            matrix = getattr(self, attribute)
            if matrix is not None:
                mmap = getattr(matrix, "_mmap", None)
                if mmap is not None:
                    mmap.close()
                setattr(self, attribute, None)
        if os.getpid() == self.creator_pid:
            shutil.rmtree(self.directory, ignore_errors=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Interpreter shutdown may already have torn down imported modules.
            pass
_PARALLEL_PROFILER_CACHE_DESCRIPTORS: tuple[
    tuple[PhysicalCandidateKey, ProfilerCandidateDescriptor], ...
] = ()


@dataclass(frozen=True, order=True)
class PhysicalCandidateKey:
    """Complete identity of one profiled physical kernel invocation.

    Scheduling candidate IDs are intentionally not sufficient. NativeVNNI
    candidates with the same row tiling can dispatch LUT-decoded, direct INT8,
    or another prepared codebook microkernel. Reusing counters across those
    paths would inject another format's instruction and cache behavior into the
    fit. Dynamic counters also vary with execution mode and every dimension of
    the submitted work, so ``M``, the projection vector, aggregate ``N``, and
    ``K`` are part of the key rather than metadata on a reusable descriptor.
    """

    backend: Backend
    architecture_class: str
    operation_kind: str
    bundle_signature: str
    prepared_family_id: str
    packing_abi: str
    runtime_codebook_id: int
    effective_candidate_id: str
    execution_mode: ExecutionMode
    m: int
    projection_n_vector: tuple[int, ...]
    aggregate_n: int
    k: int

    def canonical_tuple(self) -> tuple[str, ...]:
        """Return the stable JSON/sort representation of this identity."""

        return (
            self.backend.value,
            self.architecture_class,
            self.operation_kind,
            self.bundle_signature,
            self.prepared_family_id,
            self.packing_abi,
            str(self.runtime_codebook_id),
            self.effective_candidate_id,
            self.execution_mode.value,
            str(self.m),
            "[" + ",".join(str(value) for value in self.projection_n_vector) + "]",
            str(self.aggregate_n),
            str(self.k),
        )
ProfilerObservationKey = tuple[RuntimeKey, str, str]
ProfilerObservationIndex = Mapping[
    ProfilerObservationKey, NativeVNNIObservation
]


def _physical_key(row: NativeVNNIObservation) -> PhysicalCandidateKey:
    """Return the request layer's exact physical launch identity."""

    return PhysicalCandidateKey(
        backend=row.backend,
        architecture_class=row.architecture_class,
        operation_kind=row.operation_kind,
        bundle_signature=row.bundle_signature,
        prepared_family_id=row.prepared_family_id,
        packing_abi=row.packing_abi,
        runtime_codebook_id=row.runtime_codebook_id,
        effective_candidate_id=row.effective_candidate_id,
        execution_mode=row.execution_mode,
        m=row.m,
        projection_n_vector=row.projection_n_vector,
        aggregate_n=row.aggregate_n,
        k=row.k,
    )


def _signed_log1p(value: float) -> float:
    """Compress counter magnitudes while preserving signed diagnostics."""

    return math.copysign(math.log1p(abs(value)), value)


def _flatten_config(
    prefix: str,
    value: object,
    output: dict[str, float | str],
) -> None:
    """Flatten typed candidate configuration into deterministic model fields."""

    if isinstance(value, bool):
        output[prefix] = float(value)
    elif isinstance(value, (int, float)):
        output[prefix] = _signed_log1p(float(value))
    elif isinstance(value, str):
        output[prefix] = value
    elif isinstance(value, Mapping):
        for name, item in sorted(value.items()):
            _flatten_config(f"{prefix}.{name}", item, output)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _flatten_config(f"{prefix}.{index}", item, output)
    elif value is not None:
        raise ValueError(f"unsupported profiler-model config value at {prefix}")


def _execution_regime(operation_kind: str, m: int) -> str:
    """Classify one launch by the performance question its counters answer.

    Operation identity takes precedence over row count because grouped MoE
    prefill and grouped MTP decode can both use small matrices. The bounded-M
    fallback covers older evidence whose operation name predates the explicit
    naming convention.
    """

    normalized = operation_kind.strip().lower()
    if "prefill" in normalized or "gemm" in normalized:
        return "prefill"
    if (
        "decode" in normalized
        or "gemv" in normalized
        or "verifier" in normalized
    ):
        return "decode"
    return "decode" if m <= 15 else "prefill"


def _sum_metric(
    metric_values: Mapping[str, list[float]],
    metric_id: str,
) -> float | None:
    """Return one finite dispatch sum or ``None`` when unsupported."""

    values = metric_values.get(metric_id, ())
    return sum(values) if values else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Return one non-negative finite ratio only when both inputs are usable."""

    if (
        numerator is None
        or denominator is None
        or denominator <= 0.0
    ):
        return None
    result = numerator / denominator
    return result if math.isfinite(result) and result >= 0.0 else None


def _estimated_work(observation: NativeVNNIObservation) -> tuple[float, float, float]:
    """Return logical MACs, payload bytes, and output bytes for normalization.

    NativeVNNI payload sizes are specified per 32 weights. Activations and
    outputs use a conservative four-byte estimate; this is an explanatory
    geometry feature, not a byte-exact runtime accounting claim.
    """

    spec = format_spec(observation.source_format)
    logical_macs = float(
        observation.m * observation.aggregate_n * observation.k
    )
    weight_bytes = (
        float(observation.aggregate_n * observation.k)
        * float(spec.payload_bytes)
        / 32.0
    )
    activation_bytes = float(observation.m * observation.k * 4)
    output_bytes = float(observation.m * observation.aggregate_n * 4)
    return logical_macs, weight_bytes + activation_bytes + output_bytes, output_bytes


def _normalized_metric_features(
    observation: NativeVNNIObservation,
    metric_values: Mapping[str, list[float]],
) -> dict[str, float]:
    """Derive regime-aware rates without exposing raw absolute counters.

    Decode/GEMV retains both bytes-moved and logical-compute views. M=1 is
    predominantly bandwidth-bound, while grouped verifier rows reuse a weight
    payload and can become compute-bound as M rises. Prefill/GEMM is normalized
    by logical MACs. Static resource usage and already-normalized profiler
    percentages remain valid candidate attributes in either regime.
    """

    features: dict[str, float] = {}
    regime = _execution_regime(observation.operation_kind, observation.m)
    logical_macs, expected_bytes, output_bytes = _estimated_work(observation)

    cycles = _sum_metric(metric_values, "cpu.cycles")
    reference_cycles = _sum_metric(metric_values, "cpu.ref_cycles")
    instructions = _sum_metric(metric_values, "cpu.instructions")
    task_clock_ns = _sum_metric(metric_values, "cpu.task_clock_ns")
    wall_clock_ns = _sum_metric(metric_values, "cpu.wall_clock_ns")
    l1d_loads = _sum_metric(metric_values, "cpu.l1d_loads")
    l1d_load_misses = _sum_metric(metric_values, "cpu.l1d_load_misses")
    llc_loads = _sum_metric(metric_values, "cpu.llc_loads")
    llc_load_misses = _sum_metric(metric_values, "cpu.llc_load_misses")
    active_thread_concurrency = _ratio(task_clock_ns, wall_clock_ns)
    cpu_threads = _cpu_parallelism_width(observation.architecture_class)
    duration_features_reliable = (
        wall_clock_ns is not None
        and wall_clock_ns >= 500_000.0
        and active_thread_concurrency is not None
        and cpu_threads is not None
        and active_thread_concurrency <= float(cpu_threads) * 1.05
    )
    features["profile.cpu.duration_features_reliable"] = float(
        duration_features_reliable
    )
    cpu_ratios = {
        "branch_miss_fraction": _ratio(
            _sum_metric(metric_values, "cpu.branch_misses"),
            _sum_metric(metric_values, "cpu.branches"),
        ),
        "cache_miss_fraction": _ratio(
            _sum_metric(metric_values, "cpu.cache_misses"),
            _sum_metric(metric_values, "cpu.cache_references"),
        ),
        "l1d_load_miss_fraction": _ratio(
            l1d_load_misses,
            l1d_loads,
        ),
        "llc_load_miss_fraction": _ratio(
            llc_load_misses,
            llc_loads,
        ),
    }
    if duration_features_reliable:
        cpu_ratios.update({
            "ipc": _ratio(instructions, cycles),
            "core_to_reference_cycle_ratio": _ratio(
                cycles, reference_cycles
            ),
            "active_thread_concurrency": active_thread_concurrency,
        })
    for name, value in cpu_ratios.items():
        if value is not None:
            features[f"metric.cpu.{name}"] = value

    # Counter rates retain useful traffic information even when the matching
    # denominator event is unavailable in the simultaneous event group. In
    # particular, this keeps the already-paid LLC-miss evidence model-visible
    # on machines whose generic LLC-load event is not schedulable.
    normalized_counts = [
        ("instructions", instructions),
        ("l1d_loads", l1d_loads),
        ("l1d_load_misses", l1d_load_misses),
        ("llc_load_misses", llc_load_misses),
    ]
    if duration_features_reliable:
        normalized_counts.extend((
            ("cycles", cycles),
            ("reference_cycles", reference_cycles),
            ("task_clock_ns", task_clock_ns),
            ("wall_clock_ns", wall_clock_ns),
        ))
    normalization_units = (
        (
            ("per_expected_mib", expected_bytes / float(1 << 20)),
            ("per_million_macs", logical_macs / 1.0e6),
        )
        if regime == "decode"
        else (("per_million_macs", logical_macs / 1.0e6),)
    )
    for name, value in normalized_counts:
        for unit_name, work_units in normalization_units:
            normalized = _ratio(value, work_units)
            if normalized is not None:
                features[
                    f"metric.cpu.{regime}.{name}_{unit_name}_log1p"
                ] = _signed_log1p(normalized)

    for metric_id in sorted(_GPU_STATIC_RESOURCE_METRICS):
        values = metric_values.get(metric_id, ())
        if values:
            features[f"metric.{metric_id}.maximum_log1p"] = _signed_log1p(
                max(values)
            )
    for metric_id in sorted(_GPU_PERCENT_METRICS):
        values = metric_values.get(metric_id, ())
        if values:
            fractions = [value / 100.0 for value in values]
            features[f"metric.{metric_id}.fraction_mean"] = (
                sum(fractions) / len(fractions)
            )
            features[f"metric.{metric_id}.fraction_maximum"] = max(fractions)
    for metric_id in sorted(_GPU_RATE_METRICS):
        values = metric_values.get(metric_id, ())
        if values:
            transformed = [_signed_log1p(value) for value in values]
            features[f"metric.{metric_id}.mean_log1p"] = (
                sum(transformed) / len(transformed)
            )
            features[f"metric.{metric_id}.maximum_log1p"] = max(transformed)

    duration_ns = _sum_metric(metric_values, "gpu.duration_ns")
    if duration_ns is not None and duration_ns > 0.0:
        if regime == "decode":
            # One byte/ns is one decimal GB/s.
            features[
                "metric.gpu.decode.effective_gbytes_per_second_log1p"
            ] = (
                _signed_log1p(expected_bytes / duration_ns)
            )
            features[
                "metric.gpu.decode.duration_ns_per_expected_mib_log1p"
            ] = (
                _signed_log1p(duration_ns / (expected_bytes / float(1 << 20)))
            )
            # Grouped verifier rows reuse one weight payload while their MAC
            # count grows with M. Retaining the compute view beside bandwidth
            # lets one model describe the observed memory-to-compute transition
            # without misclassifying grouped decode as ordinary prefill.
            features["metric.gpu.decode.effective_gops_log1p"] = _signed_log1p(
                (2.0 * logical_macs) / duration_ns
            )
            features[
                "metric.gpu.decode.duration_ns_per_gmac_log1p"
            ] = _signed_log1p(duration_ns / (logical_macs / 1.0e9))
        else:
            # One operation/ns is one decimal GOP/s; each MAC contributes two
            # arithmetic operations to the conventional GEMM throughput claim.
            features["metric.gpu.prefill.effective_gops_log1p"] = _signed_log1p(
                (2.0 * logical_macs) / duration_ns
            )
            features[
                "metric.gpu.prefill.duration_ns_per_gmac_log1p"
            ] = _signed_log1p(duration_ns / (logical_macs / 1.0e9))

    spill_requests = _sum_metric(
        metric_values, "gpu.local_memory_spill_requests"
    )
    spills_per_million_outputs = _ratio(
        spill_requests,
        output_bytes / 4.0 / 1.0e6,
    )
    if spills_per_million_outputs is not None:
        features[
            "metric.gpu.spills_per_million_outputs_log1p"
        ] = _signed_log1p(spills_per_million_outputs)

    fetch_bytes = _sum_metric(metric_values, "gpu.fetch_kib")
    if fetch_bytes is not None:
        fetch_bytes *= 1024.0
        amplification = _ratio(fetch_bytes, expected_bytes)
        if amplification is not None:
            features["metric.gpu.fetch_to_expected_byte_ratio"] = amplification
        if duration_ns is not None and duration_ns > 0.0:
            features[
                "metric.gpu.observed_fetch_gbytes_per_second_log1p"
            ] = (
                _signed_log1p(fetch_bytes / duration_ns)
            )
    write_bytes = _sum_metric(metric_values, "gpu.write_kib")
    if write_bytes is not None:
        write_amplification = _ratio(write_bytes * 1024.0, output_bytes)
        if write_amplification is not None:
            features["metric.gpu.write_to_output_byte_ratio"] = (
                write_amplification
            )
    wavefronts = _sum_metric(metric_values, "gpu.wavefront_count")
    waves_per_million_outputs = _ratio(
        wavefronts,
        output_bytes / 4.0 / 1.0e6,
    )
    if waves_per_million_outputs is not None:
        features[
            "metric.gpu.wavefronts_per_million_outputs_log1p"
        ] = _signed_log1p(waves_per_million_outputs)
    return features


@dataclass(frozen=True)
class ProfilerCandidateDescriptor:
    """One authenticated exact physical invocation's profiler features.

    Candidate configuration and static resource use describe the physical
    kernel. Dynamic hardware counters describe that kernel *at the request
    geometry that produced them*. Retaining the anchor is therefore part of
    the evidence value: a coherent geometry holdout must not consume IPC,
    cache, throughput, or duration evidence measured at the geometry whose
    timing answer it is trying to predict.
    """

    key: PhysicalCandidateKey
    anchor_m: int
    anchor_n: int
    anchor_k: int
    features: Mapping[str, float | str]

    @property
    def anchor_geometry(self) -> tuple[int, int]:
        """Return the N/K geometry used by the isolated profiler launch."""

        return (self.anchor_n, self.anchor_k)


@dataclass(frozen=True)
class ProfilerFeatureCatalog:
    """Complete exact-launch descriptors bound to one timing corpus."""

    corpus_digest: str
    request_manifest_digest: str
    evidence_manifest_digest: str
    descriptors: Mapping[PhysicalCandidateKey, ProfilerCandidateDescriptor]

    def __post_init__(self) -> None:
        """Reject descriptors whose redundant launch geometry is inconsistent."""

        for key, descriptor in self.descriptors.items():
            descriptor_geometry = (
                descriptor.anchor_m,
                descriptor.anchor_n,
                descriptor.anchor_k,
            )
            key_geometry = (key.m, key.aggregate_n, key.k)
            if descriptor.key != key or descriptor_geometry != key_geometry:
                raise ValueError(
                    "profiler descriptor disagrees with its exact physical "
                    f"launch key {key.canonical_tuple()}"
                )

    @cached_property
    def digest(self) -> str:
        """Hash every model-visible descriptor and its source identities.

        A catalog is an immutable evidence value.  Policy cache migration asks
        for this identity once per generic domain, so recomputing the complete
        canonical JSON document at every access would turn a constant identity
        into a serial O(domain-count * descriptor-count) operation.
        """

        payload = {
            "version": PROFILER_CATALOG_NORMALIZATION_VERSION,
            "corpus_digest": self.corpus_digest,
            "request_manifest_digest": self.request_manifest_digest,
            "evidence_manifest_digest": self.evidence_manifest_digest,
            "descriptors": [
                {
                    "key": list(key.canonical_tuple()),
                    "anchor": {
                        "m": descriptor.anchor_m,
                        "n": descriptor.anchor_n,
                        "k": descriptor.anchor_k,
                    },
                    "features": dict(sorted(descriptor.features.items())),
                }
                for key, descriptor in sorted(
                    self.descriptors.items(),
                    key=lambda item: item[0].canonical_tuple(),
                )
            ],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    @cached_property
    def model_input_digest(self) -> str:
        """Hash normalized profiler inputs independently of the estimator.

        ``digest`` remains the full evidence provenance identity recorded in a
        frozen policy.  It intentionally changes when the authenticated timing
        corpus or profiler transaction changes.  A fit-cache key has a narrower
        purpose: it must change only when an input visible to the profiler
        surrogate changes.  Including the whole timing-corpus digest there made
        an additive refinement in one domain invalidate cross-validation for
        every unrelated ISA and M domain even when all normalized profiler
        descriptors were byte-for-byte identical.

        Candidate-cost and cross-M transfer-pool digests independently bind the
        timing rows used by each fit. This digest covers normalized descriptor
        semantics while omitting both source transaction identity and mutable
        surrogate hyperparameters. That separation lets a fit tune its forest
        without reparsing or republishing profiler evidence.
        """

        return self.model_input_digest_for_descriptors(self.descriptors)

    @cached_property
    def model_digest(self) -> str:
        """Hash normalized inputs and every prediction-changing fit setting."""

        return _profiler_surrogate_digest(self.model_input_digest)

    def model_digest_for(
        self,
        observations: Iterable[NativeVNNIObservation],
    ) -> str:
        """Hash the exact descriptor subset reachable by one transfer pool."""

        descriptors = {}
        for observation in observations:
            descriptor = self.descriptor_for(observation)
            descriptors[descriptor.key] = descriptor
        return self.model_digest_for_descriptors(descriptors)

    @staticmethod
    def model_input_digest_for_descriptors(
        descriptors: Mapping[
            PhysicalCandidateKey,
            ProfilerCandidateDescriptor,
        ],
    ) -> str:
        """Hash one normalized descriptor subset for durable catalog reuse."""

        payload = {
            "version": PROFILER_CATALOG_NORMALIZATION_VERSION,
            "descriptors": [
                {
                    "key": list(key.canonical_tuple()),
                    "anchor": {
                        "m": descriptor.anchor_m,
                        "n": descriptor.anchor_n,
                        "k": descriptor.anchor_k,
                    },
                    "features": dict(sorted(descriptor.features.items())),
                }
                for key, descriptor in sorted(
                    descriptors.items(),
                    key=lambda item: item[0].canonical_tuple(),
                )
            ],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def model_digest_for_descriptors(
        descriptors: Mapping[
            PhysicalCandidateKey,
            ProfilerCandidateDescriptor,
        ],
    ) -> str:
        """Hash normalized descriptors plus the current surrogate identity."""

        return _profiler_surrogate_digest(
            ProfilerFeatureCatalog.model_input_digest_for_descriptors(
                descriptors
            )
        )

    def descriptor_for(
        self,
        observation: NativeVNNIObservation,
    ) -> ProfilerCandidateDescriptor:
        """Resolve one timing row to its separately profiled exact launch."""

        key = _physical_key(observation)
        descriptor = self.descriptors.get(key)
        if descriptor is None:
            raise ValueError(
                "profiler feature catalog omits physical candidate "
                f"{observation.effective_candidate_id} for "
                f"{observation.architecture_class}/{observation.operation_kind}/"
                f"{observation.prepared_family_id}/cb{observation.runtime_codebook_id}/"
                f"M{observation.m}/N{observation.aggregate_n}/K{observation.k}/"
                f"{observation.execution_mode.value}"
            )
        return descriptor

    def rebind(self, corpus: ObservationCorpus) -> "ProfilerFeatureCatalog":
        """Reuse immutable physical-candidate evidence in a newer fit corpus.

        Adapter/schema evolution may change timing-row digests without changing
        the physical invocation. Rebinding is permitted only when every exact
        eligible target point resolves to existing evidence; it never
        synthesizes counters or weakens source-manifest validation.
        """

        rebound = ProfilerFeatureCatalog(
            corpus_digest=corpus.digest(),
            request_manifest_digest=self.request_manifest_digest,
            evidence_manifest_digest=self.evidence_manifest_digest,
            descriptors=self.descriptors,
        )
        for observation in corpus:
            if observation.supported and observation.generic_eligible:
                rebound.descriptor_for(observation)
        return rebound


def _physical_key_from_cache(values: list[str]) -> PhysicalCandidateKey:
    """Restore one exact physical key from its stable canonical tuple."""

    if len(values) != 13:
        raise ValueError("profiler catalog cache key changed tuple arity")
    projection_text = values[10]
    if not projection_text.startswith("[") or not projection_text.endswith("]"):
        raise ValueError("profiler catalog cache projection vector is malformed")
    projection_body = projection_text[1:-1]
    projection = tuple(
        int(value) for value in projection_body.split(",") if value
    )
    if not projection:
        raise ValueError("profiler catalog cache projection vector is empty")
    return PhysicalCandidateKey(
        backend=Backend(values[0]),
        architecture_class=values[1],
        operation_kind=values[2],
        bundle_signature=values[3],
        prepared_family_id=values[4],
        packing_abi=values[5],
        runtime_codebook_id=int(values[6]),
        effective_candidate_id=values[7],
        execution_mode=ExecutionMode(values[8]),
        m=int(values[9]),
        projection_n_vector=projection,
        aggregate_n=int(values[11]),
        k=int(values[12]),
    )


def _cache_descriptor_mapping(
    key: PhysicalCandidateKey,
    descriptor: ProfilerCandidateDescriptor,
) -> dict[str, object]:
    """Return one compact cache record in stable key order."""

    return {
        "anchor_k": descriptor.anchor_k,
        "anchor_m": descriptor.anchor_m,
        "anchor_n": descriptor.anchor_n,
        "features": dict(sorted(descriptor.features.items())),
        "key": list(key.canonical_tuple()),
        "record_type": "descriptor",
    }


def _digest_descriptor_mapping(
    key: PhysicalCandidateKey,
    descriptor: ProfilerCandidateDescriptor,
) -> dict[str, object]:
    """Return the descriptor representation used by public catalog digests."""

    return {
        "anchor": {
            "k": descriptor.anchor_k,
            "m": descriptor.anchor_m,
            "n": descriptor.anchor_n,
        },
        "features": dict(sorted(descriptor.features.items())),
        "key": list(key.canonical_tuple()),
    }


def _write_profiler_cache_descriptor_range(
    task: tuple[int, int, Path, Path],
) -> tuple[Path, Path]:
    """Serialize one inherited descriptor range into two ordered shards."""

    begin, end, cache_path, digest_path = task
    with (
        cache_path.open("wb") as cache_output,
        digest_path.open("wb") as digest_output,
    ):
        for index in range(begin, end):
            key, descriptor = _PARALLEL_PROFILER_CACHE_DESCRIPTORS[index]
            cache_output.write(json.dumps(
                _cache_descriptor_mapping(key, descriptor),
                sort_keys=True,
                separators=(",", ":"),
            ).encode() + b"\n")
            digest_output.write(json.dumps(
                _digest_descriptor_mapping(key, descriptor),
                sort_keys=True,
                separators=(",", ":"),
            ).encode() + b"\n")
    return cache_path, digest_path


def _json_bytes(value: object) -> bytes:
    """Encode one scalar or small mapping with canonical JSON settings."""

    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _descriptor_array_digest(
    prefix: bytes,
    suffix: bytes,
    descriptor_paths: Iterable[Path],
) -> str:
    """Hash a canonical JSON document around ordered JSONL descriptors."""

    digest = hashlib.sha256()
    digest.update(prefix)
    first = True
    for path in descriptor_paths:
        with path.open("rb") as handle:
            for line in handle:
                record = line.rstrip(b"\n")
                if not first:
                    digest.update(b",")
                digest.update(record)
                first = False
    digest.update(suffix)
    return "sha256:" + digest.hexdigest()


def _write_profiler_feature_catalog_cache(
    path: Path,
    catalog: ProfilerFeatureCatalog,
    *,
    request_file_digest: str,
    evidence_file_digest: str,
) -> None:
    """Publish a streamed, authenticated normalized catalog for repeat fitting.

    A six-figure catalog previously encoded the complete descriptor array four
    times in one process: two public identities, the cache identity, and final
    JSON output.  JSONL v2 serializes each descriptor once per required public
    representation in fork workers.  The parent streams those ordered shards
    through SHA-256 and the atomic output file without materializing another
    corpus-sized Python string.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".inprogress")
    descriptors = tuple(sorted(
        catalog.descriptors.items(), key=lambda item: item[0].canonical_tuple()
    ))
    worker_count = _offline_worker_count(
        len(descriptors),
        environment_name="LLAMINAR_NATIVE_VNNI_IO_WORKERS",
    )
    global _PARALLEL_PROFILER_CACHE_DESCRIPTORS
    with tempfile.TemporaryDirectory(
        prefix=f".{path.name}.parts-", dir=path.parent
    ) as directory:
        root = Path(directory)
        tasks = tuple(
            (
                begin,
                end,
                root / f"cache-{index:04d}.jsonl",
                root / f"digest-{index:04d}.jsonl",
            )
            for index, (begin, end) in enumerate(
                _equal_ranges(len(descriptors), worker_count)
            )
        )
        _PARALLEL_PROFILER_CACHE_DESCRIPTORS = descriptors
        try:
            if worker_count == 1:
                shards = tuple(
                    _write_profiler_cache_descriptor_range(task) for task in tasks
                )
            else:
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=multiprocessing.get_context("fork"),
                ) as executor:
                    shards = tuple(executor.map(
                        _write_profiler_cache_descriptor_range, tasks
                    ))
        finally:
            _PARALLEL_PROFILER_CACHE_DESCRIPTORS = ()

        digest_paths = tuple(item[1] for item in shards)
        catalog_prefix = (
            b'{"corpus_digest":' + _json_bytes(catalog.corpus_digest)
            + b',"descriptors":['
        )
        catalog_suffix = (
            b'],"evidence_manifest_digest":'
            + _json_bytes(catalog.evidence_manifest_digest)
            + b',"request_manifest_digest":'
            + _json_bytes(catalog.request_manifest_digest)
            + b',"version":'
            + _json_bytes(PROFILER_CATALOG_NORMALIZATION_VERSION)
            + b"}"
        )
        model_prefix = b'{"descriptors":['
        model_suffix = (
            b'],"version":'
            + _json_bytes(PROFILER_CATALOG_NORMALIZATION_VERSION)
            + b"}"
        )
        catalog_digest = _descriptor_array_digest(
            catalog_prefix, catalog_suffix, digest_paths
        )
        model_input_digest = _descriptor_array_digest(
            model_prefix, model_suffix, digest_paths
        )
        cached_catalog_digest = catalog.__dict__.get("digest")
        cached_model_input_digest = catalog.__dict__.get("model_input_digest")
        if cached_catalog_digest not in (None, catalog_digest):
            raise ValueError("streamed profiler catalog digest changed semantics")
        if cached_model_input_digest not in (None, model_input_digest):
            raise ValueError(
                "streamed profiler model-input digest changed semantics"
            )
        catalog.__dict__["digest"] = catalog_digest
        catalog.__dict__["model_input_digest"] = model_input_digest

        header = {
            "catalog_digest": catalog_digest,
            "corpus_digest": catalog.corpus_digest,
            "descriptor_count": len(descriptors),
            "evidence_file_digest": evidence_file_digest,
            "evidence_manifest_digest": catalog.evidence_manifest_digest,
            # These legacy field names are retained so existing v2 JSONL
            # caches remain reusable. They identify normalized catalog inputs,
            # not the independently versioned GPU histogram surrogate.
            "model_digest": model_input_digest,
            "profiler_model_version": PROFILER_CATALOG_NORMALIZATION_VERSION,
            "record_type": "header",
            "request_file_digest": request_file_digest,
            "request_manifest_digest": catalog.request_manifest_digest,
            "schema_version": PROFILER_FEATURE_CATALOG_CACHE_VERSION,
        }
        cache_digest = hashlib.sha256()
        with temporary.open("wb") as output:
            header_line = _json_bytes(header) + b"\n"
            output.write(header_line)
            cache_digest.update(header_line)
            for cache_path, _ in shards:
                with cache_path.open("rb") as shard:
                    while chunk := shard.read(1024 * 1024):
                        output.write(chunk)
                        cache_digest.update(chunk)
            footer = _json_bytes({
                "cache_digest": "sha256:" + cache_digest.hexdigest(),
                "record_type": "footer",
            }) + b"\n"
            output.write(footer)
        temporary.replace(path)


def _read_profiler_feature_catalog_cache(
    path: Path,
    *,
    request_file_digest: str,
    evidence_file_digest: str,
) -> ProfilerFeatureCatalog | None:
    """Load a cache only when both canonical source files are unchanged."""

    if not path.is_file():
        return None
    with path.open("rb") as handle:
        first_line = handle.readline()
        try:
            header = json.loads(first_line)
        except json.JSONDecodeError:
            return None
        if (
            isinstance(header, dict)
            and header.get("schema_version")
            == PROFILER_FEATURE_CATALOG_CACHE_VERSION
        ):
            expected_header_fields = {
                "catalog_digest", "corpus_digest", "descriptor_count",
                "evidence_file_digest", "evidence_manifest_digest",
                "model_digest", "profiler_model_version", "record_type",
                "request_file_digest", "request_manifest_digest",
                "schema_version",
            }
            if (
                set(header) != expected_header_fields
                or header["record_type"] != "header"
                or header["profiler_model_version"]
                != PROFILER_CATALOG_NORMALIZATION_VERSION
                or header["request_file_digest"] != request_file_digest
                or header["evidence_file_digest"] != evidence_file_digest
            ):
                return None
            digest = hashlib.sha256(first_line)
            descriptors = {}
            footer = None
            for line in handle:
                raw_record = json.loads(line)
                if raw_record.get("record_type") == "footer":
                    footer = raw_record
                    break
                digest.update(line)
                expected_record_fields = {
                    "anchor_k", "anchor_m", "anchor_n", "features", "key",
                    "record_type",
                }
                if (
                    set(raw_record) != expected_record_fields
                    or raw_record["record_type"] != "descriptor"
                ):
                    return None
                key = _physical_key_from_cache(list(raw_record["key"]))
                descriptor = ProfilerCandidateDescriptor(
                    key=key,
                    anchor_m=int(raw_record["anchor_m"]),
                    anchor_n=int(raw_record["anchor_n"]),
                    anchor_k=int(raw_record["anchor_k"]),
                    features=dict(raw_record["features"]),
                )
                if key in descriptors:
                    raise ValueError("profiler catalog cache repeats a physical key")
                descriptors[key] = descriptor
            if (
                footer is None
                or set(footer) != {"cache_digest", "record_type"}
                or footer["record_type"] != "footer"
                or footer["cache_digest"] != "sha256:" + digest.hexdigest()
                or handle.read(1)
                or len(descriptors) != int(header["descriptor_count"])
            ):
                return None
            catalog = ProfilerFeatureCatalog(
                corpus_digest=str(header["corpus_digest"]),
                request_manifest_digest=str(header["request_manifest_digest"]),
                evidence_manifest_digest=str(header["evidence_manifest_digest"]),
                descriptors=descriptors,
            )
            catalog.__dict__["digest"] = str(header["catalog_digest"])
            catalog.__dict__["model_input_digest"] = str(
                header["model_digest"]
            )
            return catalog

    raw = (
        header
        if isinstance(header, dict)
        else json.loads(path.read_text(encoding="utf-8"))
    )
    if not isinstance(raw, dict) or "cache_digest" not in raw:
        return None
    if (
        raw.get("schema_version")
        != LEGACY_PROFILER_FEATURE_CATALOG_CACHE_VERSION
        or raw.get("profiler_model_version")
        != PROFILER_CATALOG_NORMALIZATION_VERSION
        or raw.get("request_file_digest") != request_file_digest
        or raw.get("evidence_file_digest") != evidence_file_digest
    ):
        return None
    cache_digest = str(raw.pop("cache_digest"))
    if _cache_payload_digest(raw) != cache_digest:
        return None
    descriptors = {}
    for item in raw.get("descriptors", ()):
        key = _physical_key_from_cache(list(item["key"]))
        descriptor = ProfilerCandidateDescriptor(
            key=key,
            anchor_m=int(item["anchor_m"]),
            anchor_n=int(item["anchor_n"]),
            anchor_k=int(item["anchor_k"]),
            features=dict(item["features"]),
        )
        if key in descriptors:
            raise ValueError("profiler catalog cache repeats a physical key")
        descriptors[key] = descriptor
    if len(descriptors) != int(raw.get("descriptor_count", -1)):
        return None
    catalog = ProfilerFeatureCatalog(
        corpus_digest=str(raw["corpus_digest"]),
        request_manifest_digest=str(raw["request_manifest_digest"]),
        evidence_manifest_digest=str(raw["evidence_manifest_digest"]),
        descriptors=descriptors,
    )
    # The enclosing cache digest authenticates these precomputed identities.
    # Priming cached_property slots prevents another descriptor-wide JSON pass.
    catalog.__dict__["digest"] = str(raw["catalog_digest"])
    catalog.__dict__["model_input_digest"] = str(raw["model_digest"])
    return catalog


def _profiler_candidate_descriptor(
    request_id: str,
    rows: tuple[ProfilerFeatureRow, ...],
) -> ProfilerCandidateDescriptor:
    """Normalize every dispatch belonging to one exact physical invocation."""

    observation = rows[0].observation
    if any(
        _physical_key(row.observation) != _physical_key(observation)
        for row in rows
    ):
        raise ValueError(f"{request_id}: profiler dispatches changed candidate")
    if any(
        row.observation.launch_n_block_chunks
        != observation.launch_n_block_chunks
        for row in rows
    ):
        raise ValueError(
            f"{request_id}: profiler dispatches changed CPU N-block geometry"
        )
    features: dict[str, float | str] = {
        "profile.dispatch_count": _signed_log1p(float(len(rows))),
        "profile.execution_regime": _execution_regime(
            observation.operation_kind, observation.m
        ),
        "profile.candidate_family": observation.candidate_family,
    }
    _flatten_config("config", observation.config_json, features)
    if observation.launch_n_block_chunks > 0:
        features["profile.launch_n_block_chunks"] = _signed_log1p(
            float(observation.launch_n_block_chunks)
        )
    metric_values: dict[str, list[float]] = defaultdict(list)
    for dispatch in rows:
        features[
            f"profile.dispatch_kind.{dispatch.dispatch_index}"
        ] = dispatch.dispatch_kind.value
        if dispatch.block is not None:
            features[f"profile.block_threads.{dispatch.dispatch_index}"] = (
                _signed_log1p(float(math.prod(dispatch.block)))
            )
        for metric_id, value in dispatch.metric_values.items():
            if value is not None and math.isfinite(value):
                metric_values[metric_id].append(float(value))
    if not metric_values:
        raise ValueError(f"{request_id}: candidate has no available profiler metrics")
    features.update(_normalized_metric_features(observation, metric_values))
    key = _physical_key(observation)
    return ProfilerCandidateDescriptor(
        key=key,
        anchor_m=observation.m,
        anchor_n=observation.aggregate_n,
        anchor_k=observation.k,
        features=features,
    )


def _build_profiler_descriptor_range(
    bounds: tuple[int, int],
) -> tuple[ProfilerCandidateDescriptor, ...]:
    """Normalize one inherited range of independent exact-point groups."""

    begin, end = bounds
    return tuple(
        _profiler_candidate_descriptor(*_PARALLEL_PROFILER_DESCRIPTOR_GROUPS[index])
        for index in range(begin, end)
    )


def build_profiler_feature_catalog(
    corpus: ObservationCorpus,
    feature_rows: Iterable[ProfilerFeatureRow],
    *,
    request_manifest_digest: str,
    evidence_manifest_digest: str,
    workers: int | None = None,
    parallel_threshold: int = 4096,
) -> ProfilerFeatureCatalog:
    """Aggregate dispatch evidence into normalized exact-point descriptors.

    The authenticated evidence retains raw counters. The learned catalog does
    not: absolute duration, cycle, cache, grid, and traffic counts describe the
    representative request's size as much as they describe the kernel. Only
    static launch resources, normalized rates, and regime-aware work costs are
    admitted to model records.
    """

    grouped: dict[str, list[ProfilerFeatureRow]] = defaultdict(list)
    for row in feature_rows:
        grouped[row.request_id].append(row)
    groups = tuple(
        (request_id, tuple(rows)) for request_id, rows in sorted(grouped.items())
    )
    if workers is None:
        worker_count = _offline_worker_count(
            len(groups),
            environment_name="LLAMINAR_NATIVE_VNNI_POLICY_WORKERS",
        )
    else:
        if workers < 1:
            raise ValueError("profiler descriptor worker count must be positive")
        worker_count = min(workers, _physical_core_count(), len(groups))
    if worker_count <= 1 or len(groups) < parallel_threshold:
        normalized = tuple(
            _profiler_candidate_descriptor(request_id, rows)
            for request_id, rows in groups
        )
    else:
        global _PARALLEL_PROFILER_DESCRIPTOR_GROUPS
        _PARALLEL_PROFILER_DESCRIPTOR_GROUPS = groups
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                partitions = tuple(executor.map(
                    _build_profiler_descriptor_range,
                    _equal_ranges(len(groups), worker_count),
                ))
        finally:
            _PARALLEL_PROFILER_DESCRIPTOR_GROUPS = ()
        normalized = tuple(
            descriptor for partition in partitions for descriptor in partition
        )

    descriptors = {}
    for descriptor in normalized:
        key = descriptor.key
        previous = descriptors.setdefault(key, descriptor)
        if previous != descriptor:
            raise ValueError(
                "duplicate exact physical launch descriptors differ for "
                f"{key.canonical_tuple()}"
            )

    catalog = ProfilerFeatureCatalog(
        corpus_digest=corpus.digest(),
        request_manifest_digest=request_manifest_digest,
        evidence_manifest_digest=evidence_manifest_digest,
        descriptors=descriptors,
    )
    # Every eligible physical invocation in development must have evidence. A
    # sparse profiler sidecar cannot silently become a partially informed fit.
    for observation in corpus:
        if observation.supported and observation.generic_eligible:
            catalog.descriptor_for(observation)
    return catalog


def load_profiler_feature_catalog(
    corpus: ObservationCorpus,
    request_manifest_path: Path,
    evidence_manifest_path: Path,
    *,
    source_corpus: ObservationCorpus | None = None,
    source_corpus_path: Path | None = None,
    cache_path: Path | None = None,
) -> ProfilerFeatureCatalog:
    """Authenticate profiler sidecars and build the development feature set.

    ``source_corpus`` or ``source_corpus_path`` identifies the immutable common
    observations from which profiler requests were emitted. The pathname form
    is lazy: an authenticated normalized-cache hit is already bound to the raw
    request/evidence file digests, so repeat fits can rebind it without parsing
    the redundant observation witness. A cache miss reads and authenticates the
    source normally before publishing a replacement cache.
    """

    if source_corpus is not None and source_corpus_path is not None:
        raise ValueError(
            "profiler source_corpus and source_corpus_path are mutually exclusive"
        )
    request_file_digest = _file_sha256(request_manifest_path)
    evidence_file_digest = _file_sha256(evidence_manifest_path)
    cached = (
        _read_profiler_feature_catalog_cache(
            cache_path,
            request_file_digest=request_file_digest,
            evidence_file_digest=evidence_file_digest,
        )
        if cache_path is not None
        else None
    )
    if cached is not None:
        if source_corpus is None:
            return (
                cached
                if cached.corpus_digest == corpus.digest()
                else cached.rebind(corpus)
            )
        if cached.corpus_digest == source_corpus.digest():
            return (
                cached
                if cached.corpus_digest == corpus.digest()
                else cached.rebind(corpus)
            )

    evidence_corpus = (
        source_corpus
        or (
            read_observation_csv((source_corpus_path,))
            if source_corpus_path is not None
            else corpus
        )
    )

    requests = read_profiler_request_manifest(request_manifest_path)
    evidence = read_profiler_evidence_manifest(evidence_manifest_path)
    rows = profiler_feature_rows(
        evidence_corpus,
        requests,
        evidence,
        require_complete=True,
    )
    catalog = build_profiler_feature_catalog(
        evidence_corpus,
        rows,
        request_manifest_digest=requests.digest(),
        evidence_manifest_digest=evidence.digest(),
    )
    if cache_path is not None:
        _write_profiler_feature_catalog_cache(
            cache_path,
            catalog,
            request_file_digest=request_file_digest,
            evidence_file_digest=evidence_file_digest,
        )
    return catalog if evidence_corpus.digest() == corpus.digest() else catalog.rebind(corpus)


def merge_profiler_feature_catalogs(
    corpus: ObservationCorpus,
    catalogs: Iterable[ProfilerFeatureCatalog],
) -> ProfilerFeatureCatalog:
    """Merge independently authenticated additive profiler transactions.

    A newly implemented physical candidate needs a new isolated counter launch,
    but unchanged candidates do not. Each input catalog has already proved its
    own timing-corpus, request-manifest, and evidence-manifest identities. This
    function unions only their exact physical descriptors, rejects
    conflicting duplicate evidence, content-addresses the complete provenance
    set, and finally re-establishes complete coverage against ``corpus``.

    The coverage check happens after the union. Consequently, neither an old
    catalog that lacks a new candidate nor a new incremental catalog that lacks
    old candidates is usable alone; together they must describe every supported
    generic launch in the expanded fit corpus.
    """

    sources = tuple(catalogs)
    if not sources:
        raise ValueError("profiler catalog merge requires at least one source")

    descriptors: dict[PhysicalCandidateKey, ProfilerCandidateDescriptor] = {}
    for catalog in sources:
        for key, descriptor in catalog.descriptors.items():
            previous = descriptors.setdefault(key, descriptor)
            if previous != descriptor:
                raise ValueError(
                    "additive profiler catalogs disagree for exact physical launch "
                    f"{key.canonical_tuple()}"
                )

    provenance = sorted({
        (
            catalog.corpus_digest,
            catalog.request_manifest_digest,
            catalog.evidence_manifest_digest,
            catalog.digest,
        )
        for catalog in sources
    })
    def additive_digest(kind: str, identities: object) -> str:
        encoded = json.dumps(
            {
                "schema_version": "native-vnni-additive-profiler-catalog-v1",
                "kind": kind,
                "identities": identities,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    merged = ProfilerFeatureCatalog(
        corpus_digest=corpus.digest(),
        request_manifest_digest=additive_digest(
            "request-manifests",
            sorted({
                (catalog.corpus_digest, catalog.request_manifest_digest)
                for catalog in sources
            }),
        ),
        evidence_manifest_digest=additive_digest(
            "evidence-manifests",
            provenance,
        ),
        descriptors=descriptors,
    )
    for observation in corpus:
        if observation.supported and observation.generic_eligible:
            merged.descriptor_for(observation)
    return merged


def _runtime_features(
    key: RuntimeKey,
    payload_bytes_per_32_weights: float,
) -> dict[str, float]:
    """Return runtime-visible geometry in decode/prefill economic units."""

    if payload_bytes_per_32_weights <= 0.0:
        raise ValueError("runtime format payload width must be positive")
    logical_macs = float(key.m * key.aggregate_n * key.k)
    weight_bytes = (
        float(key.aggregate_n * key.k)
        * payload_bytes_per_32_weights
        / 32.0
    )
    expected_bytes = (
        weight_bytes
        + float(key.m * key.k * 4)
        + float(key.m * key.aggregate_n * 4)
    )
    regime = _execution_regime(key.operation_kind, key.m)
    result = {
        "runtime.log2_m": math.log2(key.m),
        "runtime.log2_n": math.log2(key.aggregate_n),
        "runtime.log2_k": math.log2(key.k),
        "runtime.log2_work": math.log2(key.aggregate_n * key.k),
        "runtime.log2_macs": math.log2(logical_macs),
        "runtime.aspect": key.aggregate_n / key.k,
        "runtime.n_tiles_64": math.ceil(key.aggregate_n / 64),
        "runtime.n_tiles_128": math.ceil(key.aggregate_n / 128),
        "runtime.n_tiles_256": math.ceil(key.aggregate_n / 256),
    }
    if regime == "decode":
        result.update({
            "runtime.decode.log2_expected_bytes": math.log2(expected_bytes),
            "runtime.decode.weight_byte_fraction": weight_bytes / expected_bytes,
            "runtime.decode.arithmetic_intensity": (
                2.0 * logical_macs / expected_bytes
            ),
        })
    else:
        result.update({
            "runtime.prefill.log2_expected_bytes": math.log2(expected_bytes),
            "runtime.prefill.arithmetic_intensity": (
                2.0 * logical_macs / expected_bytes
            ),
        })
    return result


def _flattened_positive_integer(
    features: Mapping[str, float | str],
    name: str,
) -> int | None:
    """Recover one positive integer from the catalog's signed-log encoding."""

    encoded = features.get(name)
    if not isinstance(encoded, float):
        return None
    decoded = math.expm1(encoded)
    rounded = round(decoded)
    if rounded <= 0 or not math.isclose(decoded, rounded, abs_tol=1.0e-9):
        raise ValueError(
            f"profiler candidate feature {name!r} is not a positive integer"
        )
    return int(rounded)


def _cpu_parallelism_width(architecture_class: str) -> int | None:
    """Extract the production OpenMP width from one CPU architecture class."""

    for component in architecture_class.split("|"):
        if not component.startswith("threads="):
            continue
        value = component.removeprefix("threads=")
        if not value.isdigit() or int(value) <= 0:
            raise ValueError(
                "CPU profiler architecture class has invalid thread width: "
                f"{architecture_class!r}"
            )
        return int(value)
    return None


def _cpu_decode_schedule_features(
    key: RuntimeKey,
    candidate_features: Mapping[str, float | str],
) -> dict[str, float]:
    """Model the production CPU M=1 N-chunk ownership geometry.

    CPU decode candidates preserve the serial-M1 arithmetic and differ only in
    how many adjacent 64-column chunks one task owns.  The raw
    ``n_block_chunks`` value does not expose the economically important launch
    transitions:

    * full-K decode skips OpenMP when the N-block count is below the physical
      thread count, making the number of serial blocks the dominant overhead;
    * K-part decode always publishes an N-block-by-K-tile producer grid and a
      second N-chunk reduction grid; and
    * both routes cross worker-wave and N-tail boundaries as ``N`` changes.

    The exact K-tile count belongs to the frozen arithmetic policy rather than
    the schedule candidate.  Strong raw launch telemetry records it once for
    every measured point, and production computes the same value before
    dispatch.  Multiplying it by the candidate's N-block count exposes the real
    producer-grid waves without guessing or leaking timing labels.
    """

    if (
        key.backend != Backend.CPU
        or key.operation_kind != "NativeVNNIFastM1Projection"
        or key.m != 1
        or candidate_features.get("config.route") != "n_chunk_grid"
    ):
        return {}
    threads = _cpu_parallelism_width(key.architecture_class)
    if threads is None:
        raise ValueError(
            "CPU decode profiler model requires threads in architecture class"
        )
    n_block_chunks = _flattened_positive_integer(
        candidate_features,
        "config.n_block_chunks",
    )
    if n_block_chunks is None:
        raise ValueError("CPU decode n_chunk_grid omits n_block_chunks")

    n_chunks = (key.aggregate_n + 63) // 64
    n_blocks = (n_chunks + n_block_chunks - 1) // n_block_chunks
    n_capacity = n_blocks * n_block_chunks * 64
    serial_kpart = ":serial-kpart:" in key.bundle_signature
    serial_full_k = ":serial-full-k:" in key.bundle_signature
    if serial_kpart == serial_full_k:
        raise ValueError(
            "CPU decode bundle must name exactly one serial arithmetic regime"
        )
    if serial_kpart != (key.launch_k_tiles > 1):
        raise ValueError(
            "CPU decode bundle disagrees with authenticated K-tile geometry"
        )

    # This condition is byte-for-byte the production full-K cutoff in
    # gemv_native_vnni_preq(). K-part never takes the serial fast path because
    # its producer and ordered reduction are two OpenMP workshares.
    serial_fast_path = serial_full_k and n_blocks < threads
    n_block_waves = (n_blocks + threads - 1) // threads
    final_n_block_wave = (n_blocks - 1) % threads + 1
    reduction_waves = (n_chunks + threads - 1) // threads
    final_reduction_wave = (n_chunks - 1) % threads + 1
    producer_tasks = n_blocks * max(1, key.launch_k_tiles)
    producer_waves = (producer_tasks + threads - 1) // threads
    final_producer_wave = (producer_tasks - 1) % threads + 1
    k_blocks = (key.k + 31) // 32
    k_tiles = max(1, key.launch_k_tiles)
    k_blocks_per_tile = (k_blocks + k_tiles - 1) // k_tiles
    final_k_tile_blocks = max(
        0,
        min(
            k_blocks_per_tile,
            k_blocks - (k_tiles - 1) * k_blocks_per_tile,
        ),
    )
    logical_macs = key.aggregate_n * key.k

    return {
        "schedule.cpu_decode.n_block_chunks": float(n_block_chunks),
        "schedule.cpu_decode.n_chunks": float(n_chunks),
        "schedule.cpu_decode.n_blocks": float(n_blocks),
        "schedule.cpu_decode.log2_n_blocks": math.log2(n_blocks),
        "schedule.cpu_decode.n_block_waves": float(n_block_waves),
        "schedule.cpu_decode.log2_n_block_waves": math.log2(n_block_waves),
        "schedule.cpu_decode.final_n_block_wave_utilization": (
            final_n_block_wave / float(threads)
        ),
        "schedule.cpu_decode.n_block_thread_coverage": min(
            1.0,
            n_blocks / float(threads),
        ),
        "schedule.cpu_decode.n_grid_utilization": (
            key.aggregate_n / float(n_capacity)
        ),
        "schedule.cpu_decode.log2_output_values_per_n_block": math.log2(
            key.aggregate_n / float(n_blocks)
        ),
        "schedule.cpu_decode.log2_macs_per_n_block": math.log2(
            logical_macs / float(n_blocks)
        ),
        "schedule.cpu_decode.serial_full_k_fast_path": float(serial_fast_path),
        "schedule.cpu_decode.openmp_region": float(not serial_fast_path),
        "schedule.cpu_decode.serial_full_k_block_count": (
            float(n_blocks) if serial_fast_path else 0.0
        ),
        "schedule.cpu_decode.kpart_producer_n_blocks": (
            float(n_blocks) if serial_kpart else 0.0
        ),
        "schedule.cpu_decode.kpart_k_tiles": (
            float(key.launch_k_tiles) if serial_kpart else 0.0
        ),
        "schedule.cpu_decode.kpart_producer_tasks": (
            float(producer_tasks) if serial_kpart else 0.0
        ),
        "schedule.cpu_decode.kpart_producer_waves": (
            float(producer_waves) if serial_kpart else 0.0
        ),
        "schedule.cpu_decode.kpart_final_producer_wave_utilization": (
            final_producer_wave / float(threads) if serial_kpart else 0.0
        ),
        "schedule.cpu_decode.kpart_k_blocks": (
            float(k_blocks) if serial_kpart else 0.0
        ),
        "schedule.cpu_decode.kpart_k_blocks_per_tile": (
            float(k_blocks_per_tile) if serial_kpart else 0.0
        ),
        "schedule.cpu_decode.kpart_log2_k_blocks_per_tile": (
            math.log2(k_blocks_per_tile) if serial_kpart else 0.0
        ),
        "schedule.cpu_decode.kpart_final_k_tile_blocks": (
            float(final_k_tile_blocks) if serial_kpart else 0.0
        ),
        "schedule.cpu_decode.kpart_final_k_tile_utilization": (
            final_k_tile_blocks / float(k_blocks_per_tile)
            if serial_kpart else 0.0
        ),
        "schedule.cpu_decode.kpart_final_k_tile_empty": (
            float(serial_kpart and final_k_tile_blocks == 0)
        ),
        "schedule.cpu_decode.kpart_reduction_tasks": (
            float(n_chunks) if serial_kpart else 0.0
        ),
        "schedule.cpu_decode.kpart_reduction_waves": (
            float(reduction_waves) if serial_kpart else 0.0
        ),
        "schedule.cpu_decode.kpart_final_reduction_wave_utilization": (
            final_reduction_wave / float(threads) if serial_kpart else 0.0
        ),
    }


def _cpu_prefill_schedule_features(
    key: RuntimeKey,
    candidate_features: Mapping[str, float | str],
) -> dict[str, float]:
    """Model the exact CPU ordinary-prefill task geometry for one candidate.

    The three full-K CPU schedules perform identical arithmetic but expose very
    different units of work to OpenMP:

    * row-chunk-grid publishes one `(row, 64-column chunk)` task;
    * two-row N-major publishes one N-block task that visits every row pair;
    * two-row pair grids publish one `(row pair, N block)` task.

    Raw `(M,N,K)` and a categorical route name do not tell a learner where
    those schedules cross worker-wave boundaries. These features reproduce the
    production formulas exactly and expose only runtime-visible geometry. They
    contain no timing label, shape identity, exact winner, or sealed evidence.
    Other backends and CPU K-part candidates return an empty map until their
    own reviewed launch formulas are represented symmetrically.
    """

    if key.backend != Backend.CPU or "prefill" not in key.operation_kind.lower():
        return {}
    route = candidate_features.get("config.route")
    if route not in {
        "row_chunk_grid",
        "two_row_full_output_tiles",
        "two_row_pair_grid",
    }:
        return {}
    threads = _cpu_parallelism_width(key.architecture_class)
    if threads is None:
        raise ValueError(
            "CPU prefill profiler model requires threads in architecture class"
        )

    n_chunks = (key.aggregate_n + 63) // 64
    if route == "row_chunk_grid":
        row_pairs = key.m
        n_block_chunks = 1
        n_blocks = n_chunks
        parallel_tasks = key.m * n_chunks
        rows_per_task = 1
        m_pair_utilization = 1.0
    else:
        row_pairs = (key.m + 1) // 2
        n_block_chunks = _flattened_positive_integer(
            candidate_features,
            "config.n_block_chunks",
        )
        if n_block_chunks is None:
            raise ValueError(
                f"CPU prefill route {route!r} omits n_block_chunks"
            )
        n_blocks = (n_chunks + n_block_chunks - 1) // n_block_chunks
        m_pair_utilization = key.m / float(row_pairs * 2)
        if route == "two_row_pair_grid":
            parallel_tasks = row_pairs * n_blocks
            rows_per_task = min(2, key.m)
        else:
            parallel_tasks = n_blocks
            rows_per_task = key.m

    parallel_waves = (parallel_tasks + threads - 1) // threads
    final_wave_tasks = (parallel_tasks - 1) % threads + 1
    n_capacity = n_blocks * n_block_chunks * 64
    output_values = key.m * key.aggregate_n
    logical_macs = output_values * key.k
    return {
        "schedule.cpu_prefill.parallel_tasks": float(parallel_tasks),
        "schedule.cpu_prefill.log2_parallel_tasks": math.log2(parallel_tasks),
        "schedule.cpu_prefill.parallel_waves": float(parallel_waves),
        "schedule.cpu_prefill.log2_parallel_waves": math.log2(parallel_waves),
        "schedule.cpu_prefill.final_wave_utilization": (
            final_wave_tasks / float(threads)
        ),
        "schedule.cpu_prefill.thread_coverage": min(
            1.0,
            parallel_tasks / float(threads),
        ),
        "schedule.cpu_prefill.n_chunks": float(n_chunks),
        "schedule.cpu_prefill.n_blocks": float(n_blocks),
        "schedule.cpu_prefill.n_block_chunks": float(n_block_chunks),
        "schedule.cpu_prefill.n_grid_utilization": (
            key.aggregate_n / float(n_capacity)
        ),
        "schedule.cpu_prefill.row_pairs": float(row_pairs),
        "schedule.cpu_prefill.rows_per_task": float(rows_per_task),
        "schedule.cpu_prefill.m_pair_utilization": m_pair_utilization,
        "schedule.cpu_prefill.log2_output_values_per_task": math.log2(
            output_values / float(parallel_tasks)
        ),
        "schedule.cpu_prefill.log2_macs_per_task": math.log2(
            logical_macs / float(parallel_tasks)
        ),
    }


def _cpu_verifier_schedule_features(
    key: RuntimeKey,
    candidate_features: Mapping[str, float | str],
) -> dict[str, float]:
    """Model one production CPU grouped-verifier launch exactly.

    Grouped verifier candidates share the serial-M1 arithmetic tree, but their
    OpenMP work differs substantially. K-part Pairwise/WideRows publish a row
    tile by N-chunk by K-tile producer grid followed by an ordered per-row
    reduction. Full-K candidates instead choose row-chunk, N-major, or
    row-pair/N-block ownership. The inherited serial-decode N-block width is
    authenticated launch telemetry because it is geometry and cache dependent;
    explicit full-K candidates additionally prove that telemetry agrees with
    their registered width.

    These are analytical, runtime-visible schedule features. They contain no
    timing, winner, model-name, or sealed-evidence information.
    """

    if (
        key.backend != Backend.CPU
        or key.operation_kind != "NativeVNNIDecodeProjection"
        or key.m < 2
    ):
        return {}
    threads = _cpu_parallelism_width(key.architecture_class)
    if threads is None:
        raise ValueError(
            "CPU verifier profiler model requires threads in architecture class"
        )
    route = candidate_features.get("config.route")
    if route not in {
        "decode_equivalent_kpart_rows",
        "row_chunk_grid",
        "two_row_n_major",
        "two_row_pair_grid",
    }:
        raise ValueError(f"unknown CPU verifier route {route!r}")
    row_tile = _flattened_positive_integer(candidate_features, "config.row_tile")
    if row_tile is None or row_tile not in {1, 2, 4}:
        raise ValueError("CPU verifier candidate omits a legal physical row tile")
    n_block_chunks = _flattened_positive_integer(
        candidate_features,
        "profile.launch_n_block_chunks",
    )
    if n_block_chunks is None:
        raise ValueError(
            "CPU verifier descriptor omits authenticated N-block telemetry"
        )
    registered_n_block_chunks = _flattened_positive_integer(
        candidate_features,
        "config.n_block_chunks",
    )
    if (
        registered_n_block_chunks is not None
        and registered_n_block_chunks != n_block_chunks
    ):
        raise ValueError(
            "CPU verifier launch N-block width disagrees with candidate registry"
        )

    k_tile_policy = candidate_features.get("config.k_tile_policy")
    kpart = key.launch_k_tiles > 1
    if k_tile_policy == "full_k" and kpart:
        raise ValueError("CPU full-K verifier candidate reached a K-partition key")
    if k_tile_policy == "inherit_serial_m1" and route != (
        "decode_equivalent_kpart_rows"
    ):
        raise ValueError("CPU inherited-K verifier candidate has the wrong route")
    if k_tile_policy not in {"full_k", "inherit_serial_m1"}:
        raise ValueError("CPU verifier candidate omits its K-tile policy")

    n_chunks = (key.aggregate_n + 63) // 64
    n_blocks = (n_chunks + n_block_chunks - 1) // n_block_chunks
    n_capacity = n_blocks * n_block_chunks * 64
    row_tiles = (key.m + row_tile - 1) // row_tile
    row_tile_capacity = row_tiles * row_tile
    k_tiles = max(1, key.launch_k_tiles)
    k_blocks = (key.k + 31) // 32
    k_blocks_per_tile = (k_blocks + k_tiles - 1) // k_tiles
    final_k_tile_blocks = max(
        0,
        min(
            k_blocks_per_tile,
            k_blocks - (k_tiles - 1) * k_blocks_per_tile,
        ),
    )

    if kpart:
        if route != "decode_equivalent_kpart_rows":
            raise ValueError("CPU K-part verifier key reached a full-K route")
        producer_tasks = row_tiles * n_chunks * k_tiles
        reduction_tasks = key.m * n_chunks
        rows_per_task = min(row_tile, key.m)
    elif route == "row_chunk_grid":
        producer_tasks = key.m * n_chunks
        reduction_tasks = 0
        rows_per_task = 1
    elif route == "two_row_n_major":
        producer_tasks = n_blocks
        reduction_tasks = 0
        rows_per_task = key.m
    else:
        producer_tasks = row_tiles * n_blocks
        reduction_tasks = 0
        rows_per_task = min(row_tile, key.m)

    producer_waves = (producer_tasks + threads - 1) // threads
    final_producer_wave = (producer_tasks - 1) % threads + 1
    reduction_waves = (
        (reduction_tasks + threads - 1) // threads
        if reduction_tasks > 0
        else 0
    )
    final_reduction_wave = (
        (reduction_tasks - 1) % threads + 1
        if reduction_tasks > 0
        else 0
    )
    direct_small_m2 = (
        not kpart
        and route in {
            "decode_equivalent_kpart_rows",
            "two_row_pair_grid",
        }
        and row_tile == 2
        and key.m == 2
        and producer_tasks <= max(1, threads * 2)
        and k_blocks <= 64
    )
    output_values = key.m * key.aggregate_n
    logical_macs = output_values * key.k

    return {
        "schedule.cpu_verifier.n_chunks": float(n_chunks),
        "schedule.cpu_verifier.n_blocks": float(n_blocks),
        "schedule.cpu_verifier.n_block_chunks": float(n_block_chunks),
        "schedule.cpu_verifier.n_grid_utilization": (
            key.aggregate_n / float(n_capacity)
        ),
        "schedule.cpu_verifier.row_tile": float(row_tile),
        "schedule.cpu_verifier.row_tiles": float(row_tiles),
        "schedule.cpu_verifier.row_tile_utilization": (
            key.m / float(row_tile_capacity)
        ),
        "schedule.cpu_verifier.rows_per_task": float(rows_per_task),
        "schedule.cpu_verifier.producer_tasks": float(producer_tasks),
        "schedule.cpu_verifier.log2_producer_tasks": math.log2(producer_tasks),
        "schedule.cpu_verifier.producer_waves": float(producer_waves),
        "schedule.cpu_verifier.log2_producer_waves": math.log2(producer_waves),
        "schedule.cpu_verifier.final_producer_wave_utilization": (
            final_producer_wave / float(threads)
        ),
        "schedule.cpu_verifier.producer_thread_coverage": min(
            1.0,
            producer_tasks / float(threads),
        ),
        "schedule.cpu_verifier.log2_output_values_per_producer_task": (
            math.log2(output_values / float(producer_tasks))
        ),
        "schedule.cpu_verifier.log2_macs_per_producer_task": (
            math.log2(logical_macs / float(producer_tasks))
        ),
        "schedule.cpu_verifier.openmp_region": float(not direct_small_m2),
        "schedule.cpu_verifier.direct_small_m2": float(direct_small_m2),
        "schedule.cpu_verifier.kpart": float(kpart),
        "schedule.cpu_verifier.kpart_k_tiles": (
            float(k_tiles) if kpart else 0.0
        ),
        "schedule.cpu_verifier.kpart_k_blocks": (
            float(k_blocks) if kpart else 0.0
        ),
        "schedule.cpu_verifier.kpart_k_blocks_per_tile": (
            float(k_blocks_per_tile) if kpart else 0.0
        ),
        "schedule.cpu_verifier.kpart_final_k_tile_blocks": (
            float(final_k_tile_blocks) if kpart else 0.0
        ),
        "schedule.cpu_verifier.kpart_final_k_tile_utilization": (
            final_k_tile_blocks / float(k_blocks_per_tile) if kpart else 0.0
        ),
        "schedule.cpu_verifier.kpart_reduction_tasks": float(reduction_tasks),
        "schedule.cpu_verifier.kpart_reduction_waves": float(reduction_waves),
        "schedule.cpu_verifier.kpart_final_reduction_wave_utilization": (
            final_reduction_wave / float(threads)
            if reduction_tasks > 0
            else 0.0
        ),
    }


def build_profiler_observation_index(
    corpus: ObservationCorpus,
) -> dict[ProfilerObservationKey, NativeVNNIObservation]:
    """Select one deterministic alias row for every candidate point.

    Observation digests serialize the complete correctness and timing proof, so
    computing them is deliberately more expensive than projecting a runtime
    key.  A grouped CV domain reuses the same candidate points in every fold.
    Building this index once per domain therefore avoids repeatedly hashing the
    same immutable rows without changing which alias supplies profiler data.
    """

    result = {}
    for runtime in corpus.runtime_keys():
        winners: dict[
            tuple[str, str],
            tuple[tuple[str, str, str], NativeVNNIObservation],
        ] = {}
        for row in corpus.rows_for_runtime_key(runtime):
            point = (row.shape_group_id, row.candidate_id)
            prefix = (
                row.source_format,
                row.execution_mode.value,
                row.effective_candidate_id,
            )
            previous = winners.get(point)
            if previous is None or prefix < previous[0]:
                winners[point] = (prefix, row)
            elif prefix == previous[0] and row is not previous[1]:
                if row.digest() < previous[1].digest():
                    winners[point] = (prefix, row)
        for (shape_group_id, candidate_id), (_prefix, row) in winners.items():
            result[(runtime, shape_group_id, candidate_id)] = row
    return result


def _model_record(
    key: RuntimeKey,
    descriptor: ProfilerCandidateDescriptor,
    source_format: str,
) -> dict[str, float | str]:
    """Combine runtime geometry with a compact reviewed interaction set.

    Interacting every counter with every runtime dimension produced hundreds
    of weak columns from one representative profile. Decode uses byte volume,
    shape, and row count; prefill uses MAC volume, arithmetic intensity, shape,
    and row count. Only normalized metrics and launch/config geometry receive
    interactions.
    """

    payload_bytes = float(format_spec(source_format).payload_bytes)
    runtime = _runtime_features(key, payload_bytes)
    regime = _execution_regime(key.operation_kind, key.m)
    interaction_names = (
        (
            "runtime.decode.log2_expected_bytes",
            "runtime.aspect",
            "runtime.log2_m",
            "runtime.n_tiles_128",
        )
        if regime == "decode"
        else (
            "runtime.prefill.log2_expected_bytes",
            "runtime.prefill.arithmetic_intensity",
            "runtime.aspect",
            "runtime.log2_m",
        )
    )
    anchor_log2_m = math.log2(descriptor.anchor_m)
    anchor_log2_n = math.log2(descriptor.anchor_n)
    anchor_log2_k = math.log2(descriptor.anchor_k)
    record: dict[str, float | str] = {
        **runtime,
        **_cpu_decode_schedule_features(key, descriptor.features),
        **_cpu_verifier_schedule_features(key, descriptor.features),
        **_cpu_prefill_schedule_features(key, descriptor.features),
        # These dimensions are constant in the old per-domain Ridge model but
        # become essential when one cross-M surrogate transfers evidence across
        # tensor formats and work sizes. They are runtime-visible identities,
        # never shape-name overlays or sealed labels.
        "runtime.backend": key.backend.value,
        "runtime.architecture_class": key.architecture_class,
        "runtime.semantic_contract": key.semantic_contract.value,
        "runtime.operation_kind": key.operation_kind,
        "runtime.bundle_signature": key.bundle_signature,
        "runtime.prepared_family_id": key.prepared_family_id,
        "runtime.packing_abi": key.packing_abi,
        "runtime.codebook": str(key.runtime_codebook_id),
        "runtime.execution_mode": key.execution_mode.value,
        # Dynamic counters are measurements at this exact profiler request,
        # not timeless properties of the candidate. The anchor and distance
        # features let the development learner reason about transfer while the
        # coherent-CV layer can remove counters measured inside a held region.
        "profile.anchor.m": float(descriptor.anchor_m),
        "profile.anchor.n": float(descriptor.anchor_n),
        "profile.anchor.k": float(descriptor.anchor_k),
        "profile.anchor.log2_m": anchor_log2_m,
        "profile.anchor.log2_n": anchor_log2_n,
        "profile.anchor.log2_k": anchor_log2_k,
        "profile.anchor.abs_delta_log2_m": abs(
            runtime["runtime.log2_m"] - anchor_log2_m
        ),
        "profile.anchor.abs_delta_log2_n": abs(
            runtime["runtime.log2_n"] - anchor_log2_n
        ),
        "profile.anchor.abs_delta_log2_k": abs(
            runtime["runtime.log2_k"] - anchor_log2_k
        ),
        "profile.anchor.dynamic_metrics_available": 1.0,
    }
    for name, value in descriptor.features.items():
        record[name] = value
        if isinstance(value, float) and name.startswith((
            "metric.",
            "config.",
            "profile.block_threads.",
            "profile.dispatch_count",
        )):
            for runtime_name in interaction_names:
                runtime_value = runtime[runtime_name]
                record[f"interaction.{runtime_name}.{name}"] = (
                    runtime_value * value
                )
    return record


def _is_anchor_dynamic_profiler_feature(name: str) -> bool:
    """Return whether a model column contains request-local counter evidence.

    Candidate configuration, dispatch kind, block size, and static GPU
    resource requirements survive geometry holdout. CPU performance counters
    and GPU achieved-throughput/rate measurements do not. Interactions inherit
    the locality of the counter they contain.
    """

    if name.startswith("interaction."):
        metric_offset = name.find("metric.")
        if metric_offset >= 0:
            return _is_anchor_dynamic_profiler_feature(name[metric_offset:])
        return False
    if name == "profile.cpu.duration_features_reliable":
        return True
    if name.startswith("metric.cpu."):
        return True
    if not name.startswith("metric.gpu."):
        return False
    static_metric_ids = (
        *_GPU_STATIC_RESOURCE_METRICS,
        "gpu.theoretical_occupancy_pct",
    )
    return not any(
        name.startswith(f"metric.{metric_id}.")
        for metric_id in static_metric_ids
    )


def _profiler_model_input_record(
    record: Mapping[str, float | str],
) -> dict[str, float | str]:
    """Return runtime/static inputs available for every unseen work point.

    Exact dynamic counters are training evidence, not runtime-visible inputs.
    Feeding them into training while masking them for a held point creates a
    missing-feature distribution shift and lets the forest key directly on a
    paid measurement. The multi-output target below carries their explanatory
    signal without requiring a profiler at dispatch time.
    """

    return {
        name: value
        for name, value in record.items()
        if not name.startswith("profile.anchor.")
        and not _is_anchor_dynamic_profiler_feature(name)
    }


def _profiler_auxiliary_target_record(
    record: Mapping[str, float | str],
) -> dict[str, float]:
    """Return calibrated exact hardware metrics as auxiliary fit targets.

    Profiler duration and any rate derived from it remain useful diagnostics,
    but canonical repeated timing already supplies the optimization label.
    Supervising on the one-shot duration again would duplicate that label and
    disguise timing leakage as profiler insight. Likewise, aggregate compute
    throughput owns the CUDA/ROCm pipe signal; its ALU/FMA/tensor constituents
    are intentionally not counted as three additional copies of one target.
    """

    return {
        name: value
        for name, value in record.items()
        if isinstance(value, float)
        and _is_anchor_dynamic_profiler_feature(name)
        and not name.startswith("interaction.")
        and name != "profile.anchor.dynamic_metrics_available"
        # This bit says whether duration-derived counters passed the control
        # overhead gate; it is not itself an economic property. Likewise, the
        # one-shot wall/task clocks are diagnostics beside the canonical
        # repeated timing corpus, not auxiliary timing labels.
        and name != "profile.cpu.duration_features_reliable"
        and "task_clock_ns_" not in name
        and "wall_clock_ns_" not in name
        and _auxiliary_metric_reliability_weight(name) > 0.0
    }


def _center_complete_candidate_relative_auxiliary(
    auxiliary_records: list[dict[str, float]],
    point_members: Mapping[tuple[RuntimeKey, str], list[int]],
) -> list[dict[str, float]]:
    """Center only metrics available for every candidate in one contest.

    A missing optional counter must mean "no evidence at this point", not the
    numeric value zero. A metric is therefore admitted for a runtime/shape
    contest only when every competing candidate published it. Unavailable
    contests remain all-zero after matrix assembly, which is the neutral value
    for candidate-relative supervision.
    """

    centered = [dict() for _ in auxiliary_records]
    for members in point_members.values():
        names = {
            name
            for index in members
            for name in auxiliary_records[index]
        }
        for name in names:
            if not all(name in auxiliary_records[index] for index in members):
                continue
            mean = sum(
                auxiliary_records[index][name] for index in members
            ) / float(len(members))
            for index in members:
                centered[index][name] = auxiliary_records[index][name] - mean
    return centered


def _auxiliary_metric_reliability_weight(name: str) -> float:
    """Return an audited relative trust weight for one profiler target.

    The CPU control probes showed instruction counts to be the most repeatable
    signal, L1 traffic to be useful but somewhat noisier, and sparse LLC misses
    to be diagnostic rather than authoritative. Duration-derived cycle/rate
    features are admitted only after the reliability gate and remain below the
    deterministic work counters. GPU metrics retain equal provisional weight
    until their backend-specific repeatability probes calibrate this table.
    """

    if any(token in name for token in (
        "effective_gbytes_per_second",
        "effective_gops",
        "duration_ns_per_",
        "observed_fetch_gbytes_per_second",
    )):
        return 0.0
    if "instructions_" in name:
        return 1.0
    if "l1d_load" in name:
        return 0.75
    if "llc_load" in name:
        return 0.25
    if name.startswith("metric.cpu."):
        return 0.5
    if name == "metric.gpu.compute_throughput_pct_of_peak.fraction_mean":
        return 1.0
    if name == "metric.gpu.dram_throughput_pct_of_peak.fraction_mean":
        return 1.0
    if name == "metric.gpu.achieved_occupancy_pct.fraction_mean":
        return 0.25
    if any(token in name for token in (
        "alu_pipe_utilization_pct",
        "fma_pipe_utilization_pct",
        "tensor_pipe_utilization_pct",
        ".fraction_maximum",
    )):
        return 0.0
    return 0.5


def _mask_held_out_profiler_anchor(
    record: dict[str, float | str],
    held_out_geometries: frozenset[tuple[int, int]],
) -> dict[str, float | str]:
    """Remove dynamic counters measured at a coherently held N/K geometry."""

    anchor = (
        int(record["profile.anchor.n"]),
        int(record["profile.anchor.k"]),
    )
    if anchor not in held_out_geometries:
        return record
    masked = {
        name: value
        for name, value in record.items()
        if not _is_anchor_dynamic_profiler_feature(name)
    }
    masked["profile.anchor.dynamic_metrics_available"] = 0.0
    return masked


def _profiler_exemplar_index(
    observations: ProfilerObservationIndex,
) -> dict[tuple[object, ...], NativeVNNIObservation]:
    """Index static candidate descriptors usable at an unseen work point."""

    result = {}
    for (runtime, _shape_group, candidate_id), observation in observations.items():
        identity = (
            runtime.backend,
            runtime.architecture_class,
            runtime.semantic_contract,
            runtime.operation_kind,
            runtime.bundle_signature,
            runtime.prepared_family_id,
            runtime.packing_abi,
            runtime.runtime_codebook_id,
            runtime.execution_mode,
            runtime.m,
            candidate_id,
        )
        previous = result.get(identity)
        if previous is None:
            result[identity] = observation
            continue
        observation_prefix = (
            observation.source_format,
            observation.shape_group_id,
        )
        previous_prefix = (
            previous.source_format,
            previous.shape_group_id,
        )
        if (
            observation_prefix < previous_prefix
            or (
                observation_prefix == previous_prefix
                and observation.digest() < previous.digest()
            )
        ):
            result[identity] = observation
    return result


def _profiler_record_for_cost(
    cost: CandidateCostLike,
    observations: ProfilerObservationIndex,
    exemplars: Mapping[tuple[object, ...], NativeVNNIObservation],
    catalog: ProfilerFeatureCatalog,
) -> dict[str, float | str]:
    """Resolve exact profiler data or static-only features for a burned point."""

    point = (cost.runtime_key, cost.shape_group_id, cost.candidate_id)
    observation = observations.get(point)
    extrapolated = observation is None
    if observation is None:
        runtime = cost.runtime_key
        effective_candidate_id = cost.candidate_id
        if runtime.backend == Backend.CUDA:
            # CUDA generic KPAR formula IDs are policy decisions, not physical
            # launches. Resolve the exact measured KB candidate directly from
            # runtime geometry instead of materializing millions of cloned
            # observation aliases solely to recover this catalog key.
            from .candidate_registry import cuda_native_vnni_gemv_registry
            from .cuda_shape_resolved import resolve_cuda_concrete_candidate_id

            candidate = cuda_native_vnni_gemv_registry().resolve(
                cost.candidate_id
            )
            effective_candidate_id = resolve_cuda_concrete_candidate_id(
                candidate,
                runtime.aggregate_n,
                runtime.k,
            )
        physical_key = PhysicalCandidateKey(
            backend=runtime.backend,
            architecture_class=runtime.architecture_class,
            operation_kind=runtime.operation_kind,
            bundle_signature=runtime.bundle_signature,
            prepared_family_id=runtime.prepared_family_id,
            packing_abi=runtime.packing_abi,
            runtime_codebook_id=runtime.runtime_codebook_id,
            effective_candidate_id=effective_candidate_id,
            execution_mode=runtime.execution_mode,
            m=runtime.m,
            projection_n_vector=runtime.projection_n_vector,
            aggregate_n=runtime.aggregate_n,
            k=runtime.k,
        )
        descriptor = catalog.descriptors.get(physical_key)
        if descriptor is not None:
            aliases = runtime_aliases(
                runtime.backend.value,
                runtime.runtime_codebook_id,
            )
            if not aliases:
                raise ValueError(
                    "profiler model cannot resolve a source format for "
                    f"runtime codebook {runtime.runtime_codebook_id}"
                )
            payload_bytes = {format_spec(alias).payload_bytes for alias in aliases}
            if len(payload_bytes) != 1:
                raise ValueError(
                    "runtime codebook aliases disagree on profiler payload size: "
                    f"backend={runtime.backend.value}, "
                    f"codebook={runtime.runtime_codebook_id}, aliases={aliases}"
                )
            return _model_record(runtime, descriptor, min(aliases))
        identity = (
            runtime.backend,
            runtime.architecture_class,
            runtime.semantic_contract,
            runtime.operation_kind,
            runtime.bundle_signature,
            runtime.prepared_family_id,
            runtime.packing_abi,
            runtime.runtime_codebook_id,
            runtime.execution_mode,
            runtime.m,
            cost.candidate_id,
        )
        observation = exemplars.get(identity)
        if observation is None:
            raise ValueError(
                f"profiler model cannot resolve candidate point {point}"
            )
    descriptor = catalog.descriptor_for(observation)
    record = _model_record(
        cost.runtime_key,
        descriptor,
        observation.source_format,
    )
    if extrapolated:
        # A burned seal has timing but no isolated profiler launch. Preserve
        # candidate configuration and static resource features while making
        # every request-local counter unavailable to the surrogate.
        record = _mask_held_out_profiler_anchor(
            record,
            frozenset({(descriptor.anchor_n, descriptor.anchor_k)}),
        )
    return record


def _profiler_model_record_worker_count(
    record_count: int,
    *,
    requested_workers: int | None,
    parallel_threshold: int,
) -> int:
    """Choose useful index workers without scheduling SMT siblings.

    Model-record construction is Python object work rather than a trainer
    kernel. One process per affinity-visible physical core is therefore the
    useful upper bound. The records-per-worker floor keeps process startup and
    shard serialization from overwhelming small fit pools.
    """

    if record_count < 0:
        raise ValueError("profiler model record count cannot be negative")
    if parallel_threshold < 1:
        raise ValueError("profiler model parallel threshold must be positive")
    if record_count == 0:
        return 0
    if requested_workers is None:
        worker_count = _offline_worker_count(
            record_count,
            environment_name=(
                "LLAMINAR_NATIVE_VNNI_PROFILER_RECORD_WORKERS"
            ),
            records_per_worker=2048,
        )
    else:
        if requested_workers < 1:
            raise ValueError(
                "profiler model record worker count must be positive"
            )
        worker_count = min(
            requested_workers,
            _physical_core_count(),
            record_count,
        )
    return 1 if record_count < parallel_threshold else worker_count


def _parallel_model_record_context() -> tuple[
    tuple[CandidateCostLike, ...],
    ProfilerObservationIndex,
    Mapping[tuple[object, ...], NativeVNNIObservation],
    ProfilerFeatureCatalog,
]:
    """Return the inherited immutable matrix-construction context."""

    if (
        _PARALLEL_MODEL_RECORD_OBSERVATIONS is None
        or _PARALLEL_MODEL_RECORD_EXEMPLARS is None
        or _PARALLEL_MODEL_RECORD_CATALOG is None
    ):
        raise RuntimeError(
            "parallel profiler model record context is not initialized"
        )
    return (
        _PARALLEL_MODEL_RECORD_COSTS,
        _PARALLEL_MODEL_RECORD_OBSERVATIONS,
        _PARALLEL_MODEL_RECORD_EXEMPLARS,
        _PARALLEL_MODEL_RECORD_CATALOG,
    )


def _dict_vector_feature_name(name: str, value: float | str) -> str:
    """Return the exact dense ``DictVectorizer`` column identity."""

    return f"{name}={value}" if isinstance(value, str) else name


def _discover_parallel_profiler_model_schema(
    task: tuple[int, int],
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    """Discover one range's compact input and auxiliary column vocabulary."""

    costs, observations, exemplars, catalog = _parallel_model_record_context()
    begin, end = task
    input_names: set[str] = set()
    auxiliary_names: set[str] = set()
    for cost in costs[begin:end]:
        record = _profiler_record_for_cost(
            cost,
            observations,
            exemplars,
            catalog,
        )
        input_names.update(
            _dict_vector_feature_name(name, value)
            for name, value in _profiler_model_input_record(record).items()
        )
        auxiliary_names.update(_profiler_auxiliary_target_record(record))
    return tuple(sorted(input_names)), tuple(sorted(auxiliary_names)), os.getpid()


def _fill_parallel_profiler_model_matrix(
    task: tuple[
        int,
        int,
        str,
        tuple[str, ...],
        tuple[str, ...],
    ],
) -> tuple[int, int]:
    """Write one disjoint range directly into shared numeric memmaps.

    No candidate records cross a process pipe and no worker owns a private
    result shard.  Each process creates only a bounded block of temporary rows,
    then writes it into its disjoint matrix range.  The files are inherited by
    later forest workers through ordinary read-only page-cache mappings.
    """

    costs, observations, exemplars, catalog = _parallel_model_record_context()
    begin, end, raw_directory, input_names, auxiliary_names = task
    directory = Path(raw_directory)
    row_count = len(costs)
    input_columns = {name: index for index, name in enumerate(input_names)}
    auxiliary_columns = {
        name: index for index, name in enumerate(auxiliary_names)
    }
    inputs = np.memmap(
        directory / "model-inputs.f32",
        dtype=np.float32,
        mode="r+",
        shape=(row_count, len(input_names)),
    )
    auxiliary = (
        np.memmap(
            directory / "auxiliary-targets.f64",
            dtype=np.float64,
            mode="r+",
            shape=(row_count, len(auxiliary_names)),
        )
        if auxiliary_names
        else None
    )
    anchors = np.memmap(
        directory / "anchor-nk.i64",
        dtype=np.int64,
        mode="r+",
        shape=(row_count, 2),
    )
    regret_targets = np.memmap(
        directory / "regret-targets.f64",
        dtype=np.float64,
        mode="r+",
        shape=(row_count,),
    )

    block_rows = 1024
    for block_begin in range(begin, end, block_rows):
        block_end = min(end, block_begin + block_rows)
        input_block = np.zeros(
            (block_end - block_begin, len(input_names)),
            dtype=np.float32,
        )
        auxiliary_block = (
            np.full(
                (block_end - block_begin, len(auxiliary_names)),
                np.nan,
                dtype=np.float64,
            )
            if auxiliary_names
            else None
        )
        anchor_block = np.empty(
            (block_end - block_begin, 2),
            dtype=np.int64,
        )
        regret_block = np.empty(block_end - block_begin, dtype=np.float64)
        for local_index, cost in enumerate(costs[block_begin:block_end]):
            record = _profiler_record_for_cost(
                cost,
                observations,
                exemplars,
                catalog,
            )
            for name, value in _profiler_model_input_record(record).items():
                feature_name = _dict_vector_feature_name(name, value)
                try:
                    column = input_columns[feature_name]
                except KeyError as error:
                    raise RuntimeError(
                        "profiler model schema changed between discovery and "
                        f"matrix publication: {feature_name}"
                    ) from error
                input_block[local_index, column] = (
                    1.0 if isinstance(value, str) else float(value)
                )
            if auxiliary_block is not None:
                for name, value in _profiler_auxiliary_target_record(
                    record
                ).items():
                    auxiliary_block[
                        local_index, auxiliary_columns[name]
                    ] = value
            anchor_block[local_index] = (
                int(record["profile.anchor.n"]),
                int(record["profile.anchor.k"]),
            )
            regret_block[local_index] = math.log1p(
                max(0.0, cost.max_surface_regret)
            )
        if not np.isfinite(input_block).all():
            raise ValueError("profiler model input matrix contains non-finite data")
        inputs[block_begin:block_end] = input_block
        if auxiliary is not None and auxiliary_block is not None:
            auxiliary[block_begin:block_end] = auxiliary_block
        anchors[block_begin:block_end] = anchor_block
        regret_targets[block_begin:block_end] = regret_block

    inputs.flush()
    anchors.flush()
    regret_targets.flush()
    if auxiliary is not None:
        auxiliary.flush()
    return end - begin, os.getpid()


def _compact_profiler_regret_prediction_values(
    costs: list[CandidateCostLike],
    training_costs: list[CandidateCostLike] | None,
    index: ProfilerModelRecordIndex,
    *,
    training_row_indices: Iterable[int] | None,
    prediction_row_indices: Iterable[int] | None,
    excluded_profiler_geometries: frozenset[tuple[int, int]],
    surrogate_device: str,
) -> np.ndarray | None:
    """Fit one surface and return aligned regret values without row objects.

    Row selection preserves the exact source order consumed by the former
    ``DictVectorizer`` path.  Auxiliary target centering is expressed as
    vector reductions over stable contest IDs; a metric contributes only when
    every candidate in that runtime/shape contest owns an unmasked value.
    """

    if training_row_indices is None:
        if training_costs is None:
            raise ValueError(
                "profiler training costs are required without aligned rows"
            )
        training_rows = index.row_indices_for(training_costs)
    else:
        training_rows = np.fromiter(
            training_row_indices,
            dtype=np.int64,
        )
    if prediction_row_indices is None:
        prediction_rows = index.row_indices_for(costs)
    else:
        prediction_rows = np.fromiter(
            prediction_row_indices,
            dtype=np.int64,
            count=len(costs),
        )
    training_count = (
        len(training_rows) if training_costs is None else len(training_costs)
    )
    if len(training_rows) != training_count:
        raise ValueError("profiler training row inventory changed")
    if len(prediction_rows) != len(costs):
        raise ValueError("profiler prediction row inventory changed")
    if (
        np.any(training_rows < 0)
        or np.any(training_rows >= index.row_count)
        or np.any(prediction_rows < 0)
        or np.any(prediction_rows >= index.row_count)
    ):
        raise ValueError("profiler matrix row index is out of range")

    # Advanced indexing intentionally produces a private C-contiguous block.
    # It keeps scikit-learn's fit input in exact filtered source order without
    # exposing held timing rows through zero-weight implementation details.
    training_matrix = np.asarray(
        index.input_matrix[training_rows],
        dtype=np.float32,
        order="C",
    )
    contest_ids = np.asarray(index.contest_ids[training_rows], dtype=np.int64)
    anchors = np.asarray(index.anchor_matrix[training_rows], dtype=np.int64)
    held_anchor = np.zeros(len(training_rows), dtype=np.bool_)
    for n, k in excluded_profiler_geometries:
        held_anchor |= (anchors[:, 0] == n) & (anchors[:, 1] == k)

    group_count = (
        int(contest_ids.max()) + 1 if len(contest_ids) else 0
    )
    contest_sizes = np.bincount(contest_ids, minlength=group_count)
    varying_auxiliary: list[tuple[str, float, float, np.ndarray]] = []
    for column, name in enumerate(index.auxiliary_feature_names):
        values = np.asarray(
            index.auxiliary_matrix[training_rows, column],
            dtype=np.float64,
        )
        available = np.isfinite(values) & ~held_anchor
        available_counts = np.bincount(
            contest_ids,
            weights=available.astype(np.float64),
            minlength=group_count,
        )
        complete = available_counts == contest_sizes
        sums = np.bincount(
            contest_ids,
            weights=np.where(available, values, 0.0),
            minlength=group_count,
        )
        means = np.zeros(group_count, dtype=np.float64)
        populated = complete & (contest_sizes > 0)
        means[populated] = sums[populated] / contest_sizes[populated]
        admitted = available & complete[contest_ids]
        centered = np.zeros(len(values), dtype=np.float64)
        centered[admitted] = values[admitted] - means[contest_ids[admitted]]
        mean = float(np.mean(centered, dtype=np.float64))
        variance = float(np.var(centered, dtype=np.float64))
        if variance > 1.0e-18:
            varying_auxiliary.append(
                (name, mean, math.sqrt(variance), centered)
            )
    if not varying_auxiliary:
        return None

    regret_targets = np.asarray(
        index.regret_targets[training_rows],
        dtype=np.float64,
    )
    regret_mean = float(np.mean(regret_targets, dtype=np.float64))
    regret_variance = float(np.var(regret_targets, dtype=np.float64))
    if regret_variance <= 1.0e-18:
        return None
    regret_scale = math.sqrt(regret_variance)
    auxiliary_weight_sum = sum(
        _auxiliary_metric_reliability_weight(name)
        for name, _mean, _scale, _values in varying_auxiliary
    )
    auxiliary_scales = {
        name: math.sqrt(
            PROFILER_AUXILIARY_TARGET_TOTAL_WEIGHT
            * _auxiliary_metric_reliability_weight(name)
            / auxiliary_weight_sum
        )
        for name, _mean, _scale, _values in varying_auxiliary
    }
    targets = np.empty(
        (training_count, 1 + len(varying_auxiliary)),
        dtype=np.float64,
    )
    targets[:, 0] = (regret_targets - regret_mean) / regret_scale
    for output_column, (name, mean, scale, values) in enumerate(
        varying_auxiliary,
        start=1,
    ):
        targets[:, output_column] = (
            auxiliary_scales[name] * (values - mean) / scale
        )

    prediction_matrix = np.asarray(
        index.input_matrix[prediction_rows],
        dtype=np.float32,
        order="C",
    )
    predictions = _fit_profiler_surrogate(
        training_matrix,
        targets,
        prediction_matrix,
        device=surrogate_device,
    )
    return np.maximum(
        0.0,
        np.expm1(predictions * regret_scale + regret_mean),
    ).astype(np.float64, copy=False)


def _apply_compact_profiler_regret_predictions(
    costs: list[CandidateCostLike],
    training_costs: list[CandidateCostLike],
    index: ProfilerModelRecordIndex,
    *,
    training_row_indices: Iterable[int] | None,
    prediction_row_indices: Iterable[int] | None,
    excluded_profiler_geometries: frozenset[tuple[int, int]],
    surrogate_device: str,
) -> list[CandidateCostLike]:
    """Attach compact predictions for callers that require cost objects."""

    predictions = _compact_profiler_regret_prediction_values(
        costs,
        training_costs,
        index,
        training_row_indices=training_row_indices,
        prediction_row_indices=prediction_row_indices,
        excluded_profiler_geometries=excluded_profiler_geometries,
        surrogate_device=surrogate_device,
    )
    if predictions is None:
        return costs
    return [
        replace(cost, profiler_predicted_regret=float(prediction))
        for cost, prediction in zip(costs, predictions, strict=True)
    ]


def apply_profiler_regret_predictions(
    costs: Iterable[CandidateCostLike],
    corpus: ObservationCorpus,
    catalog: ProfilerFeatureCatalog,
    *,
    observation_index: ProfilerObservationIndex | None = None,
    model_training_costs: Iterable[CandidateCostLike] | None = None,
    model_record_index: (
        Mapping[tuple[RuntimeKey, str, str], dict[str, float | str]]
        | ProfilerModelRecordIndex
        | None
    ) = None,
    model_training_row_indices: Iterable[int] | None = None,
    prediction_row_indices: Iterable[int] | None = None,
    excluded_profiler_geometries: frozenset[tuple[int, int]] = frozenset(),
    _surrogate_device: str = "cuda:0",
) -> list[CandidateCostLike]:
    """Train one deterministic GPU histogram surrogate and attach predictions.

    ``costs`` are the rows whose fitting costs receive predictions.
    ``model_training_costs`` may be a wider, leakage-controlled cross-M pool
    spanning tensor formats and prefill work sizes. The policy learner excludes
    every held-out geometry across that entire pool before supplying it. This
    lets the nonlinear model learn candidate/shape/work-size crossovers without
    seeing a held-out timing label under another codebook or M. Exact dynamic
    counters are candidate-relative auxiliary targets, never model inputs;
    ``excluded_profiler_geometries`` applies the same boundary to those targets.
    Runtime geometry, static candidate configuration, schedule formulas, and
    static resource use remain available inputs.

    Predictions are clamped to non-negative regret. The policy learner owns the
    deliberately separate question of how much a surrogate may influence a
    measured candidate cost; keeping the raw prediction here makes that budget
    explicit and independently testable instead of hiding it in model output.
    """

    costs = list(costs)
    if not costs:
        return costs
    training_costs = list(
        costs if model_training_costs is None else model_training_costs
    )
    if not training_costs:
        raise ValueError("profiler model training pool is empty")
    candidate_ids = {cost.candidate_id for cost in training_costs}
    measured_targets = {
        float(cost.max_surface_regret) for cost in training_costs
    }
    if len(candidate_ids) < 2 or len(measured_targets) < 2:
        # A one-candidate domain and an exact all-candidate tie have no dispatch
        # choice for profiler features to explain. Preserve measured costs; the
        # catalog completeness gate still proves that every launch was profiled.
        return costs
    if isinstance(model_record_index, ProfilerModelRecordIndex):
        return _apply_compact_profiler_regret_predictions(
            costs,
            training_costs,
            model_record_index,
            training_row_indices=model_training_row_indices,
            prediction_row_indices=prediction_row_indices,
            excluded_profiler_geometries=excluded_profiler_geometries,
            surrogate_device=_surrogate_device,
        )
    if model_training_row_indices is not None or prediction_row_indices is not None:
        raise ValueError(
            "explicit profiler matrix rows require a compact record index"
        )
    observations = (
        build_profiler_observation_index(corpus)
        if observation_index is None
        else observation_index
    )
    exemplars = (
        None
        if model_record_index is not None
        else _profiler_exemplar_index(observations)
    )

    def record_for(cost: CandidateCostLike) -> dict[str, float | str]:
        """Resolve one candidate point to its authenticated model record."""

        point = (cost.runtime_key, cost.shape_group_id, cost.candidate_id)
        if model_record_index is not None:
            record = model_record_index.get(point)
            if record is None:
                raise ValueError(
                    f"profiler record index cannot resolve candidate point {point}"
                )
            return _mask_held_out_profiler_anchor(
                record,
                excluded_profiler_geometries,
            )
        if exemplars is None:
            raise RuntimeError("profiler exemplar index was not initialized")
        return _mask_held_out_profiler_anchor(
            _profiler_record_for_cost(
                cost,
                observations,
                exemplars,
                catalog,
            ),
            excluded_profiler_geometries,
        )

    training_records = []
    regret_targets = []
    for cost in training_costs:
        training_records.append(record_for(cost))
        regret_targets.append(math.log1p(max(0.0, cost.max_surface_regret)))

    input_vectorizer = DictVectorizer(sparse=False)
    training_matrix = input_vectorizer.fit_transform([
        _profiler_model_input_record(record) for record in training_records
    ])
    auxiliary_records = [
        _profiler_auxiliary_target_record(record)
        for record in training_records
    ]
    point_members: dict[tuple[RuntimeKey, str], list[int]] = defaultdict(list)
    for index, cost in enumerate(training_costs):
        point_members[(cost.runtime_key, cost.shape_group_id)].append(index)
    auxiliary_records = _center_complete_candidate_relative_auxiliary(
        auxiliary_records,
        point_members,
    )
    auxiliary_names = sorted({
        name for record in auxiliary_records for name in record
    })
    varying_auxiliary: list[tuple[str, float, float]] = []
    for name in auxiliary_names:
        values = [record.get(name, 0.0) for record in auxiliary_records]
        mean = sum(values) / float(len(values))
        variance = sum((value - mean) ** 2 for value in values) / float(
            len(values)
        )
        if variance > 1.0e-18:
            varying_auxiliary.append((name, mean, math.sqrt(variance)))
    if not varying_auxiliary:
        # A fold with no varying exact dynamic evidence is measured-only.
        # Candidate configuration and static resources cannot masquerade as
        # profiler influence by themselves.
        return costs

    regret_mean = sum(regret_targets) / float(len(regret_targets))
    regret_variance = sum(
        (value - regret_mean) ** 2 for value in regret_targets
    ) / float(len(regret_targets))
    if regret_variance <= 1.0e-18:
        return costs
    regret_scale = math.sqrt(regret_variance)
    auxiliary_weight_sum = sum(
        _auxiliary_metric_reliability_weight(name)
        for name, _mean, _scale in varying_auxiliary
    )
    auxiliary_scales = {
        name: math.sqrt(
            PROFILER_AUXILIARY_TARGET_TOTAL_WEIGHT
            * _auxiliary_metric_reliability_weight(name)
            / auxiliary_weight_sum
        )
        for name, _mean, _scale in varying_auxiliary
    }
    targets = []
    for regret, record in zip(
        regret_targets, auxiliary_records, strict=True
    ):
        targets.append([
            (regret - regret_mean) / regret_scale,
            *(
                auxiliary_scales[name]
                * (record.get(name, 0.0) - mean)
                / scale
                for name, mean, scale in varying_auxiliary
            ),
        ])
    prediction_matrix = input_vectorizer.transform([
        _profiler_model_input_record(record_for(cost)) for cost in costs
    ])
    predictions = _fit_profiler_surrogate(
        training_matrix,
        np.asarray(targets, dtype=np.float64),
        prediction_matrix,
        device=_surrogate_device,
    )
    return [
        replace(
            cost,
            profiler_predicted_regret=max(
                0.0,
                math.expm1(
                    float(prediction) * regret_scale + regret_mean
                ),
            ),
        )
        for cost, prediction in zip(costs, predictions, strict=True)
    ]


def build_profiler_model_record_index(
    costs: Iterable[CandidateCostLike],
    corpus: ObservationCorpus,
    catalog: ProfilerFeatureCatalog,
    *,
    observation_index: ProfilerObservationIndex | None = None,
    exemplar_index: Mapping[
        tuple[object, ...], NativeVNNIObservation
    ] | None = None,
    workers: int | None = None,
    parallel_threshold: int = 4096,
) -> ProfilerModelRecordIndex:
    """Materialize immutable surrogate features into shared numeric matrices.

    Cross-validation fits many leakage-controlled forests over different row
    subsets of the same cross-M transfer pool. Rebuilding runtime/profiler
    interactions for every fold is pure duplicate work and used to serialize
    policy fitting for several minutes. This index is independent of timing
    targets and held-out labels, so every forest can safely reuse it while
    retaining its own strictly filtered training-row set.

    Large pools are partitioned into deterministic contiguous ranges and
    evaluated by affinity-visible physical-core workers. One persistent pool
    first discovers the exact ``DictVectorizer`` vocabulary, then writes
    disjoint ranges directly into shared memmaps. No expanded records or matrix
    blocks cross worker pipes. ``workers`` and ``parallel_threshold`` remain
    exposed for deterministic serial/parallel regressions.
    """

    global _PARALLEL_MODEL_RECORD_COSTS
    global _PARALLEL_MODEL_RECORD_OBSERVATIONS
    global _PARALLEL_MODEL_RECORD_EXEMPLARS
    global _PARALLEL_MODEL_RECORD_CATALOG

    started = time.perf_counter()
    observations = (
        build_profiler_observation_index(corpus)
        if observation_index is None
        else observation_index
    )
    exemplars = (
        _profiler_exemplar_index(observations)
        if exemplar_index is None
        else exemplar_index
    )
    costs = tuple(costs)
    worker_count = _profiler_model_record_worker_count(
        len(costs),
        requested_workers=workers,
        parallel_threshold=parallel_threshold,
    )
    worker_pids: set[int] = set()
    directory = Path(tempfile.mkdtemp(
        prefix="native-vnni-profiler-model-matrix-",
    ))
    input_names: tuple[str, ...] = ()
    auxiliary_names: tuple[str, ...] = ()
    point_rows: dict[tuple[RuntimeKey, str, str], int] = {}
    _PARALLEL_MODEL_RECORD_COSTS = costs
    _PARALLEL_MODEL_RECORD_OBSERVATIONS = observations
    _PARALLEL_MODEL_RECORD_EXEMPLARS = exemplars
    _PARALLEL_MODEL_RECORD_CATALOG = catalog
    executor: ProcessPoolExecutor | None = None
    try:
        if costs:
            ranges = _equal_ranges(len(costs), max(1, worker_count))
            if worker_count <= 1:
                schema_results = tuple(
                    _discover_parallel_profiler_model_schema(item)
                    for item in ranges
                )
                worker_pids.add(os.getpid())
            else:
                executor = ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=multiprocessing.get_context("fork"),
                )
                schema_results = tuple(executor.map(
                    _discover_parallel_profiler_model_schema,
                    ranges,
                    chunksize=1,
                ))
            input_names = tuple(sorted({
                name
                for names, _auxiliary, worker_pid in schema_results
                for name in names
            }))
            auxiliary_names = tuple(sorted({
                name
                for _inputs, names, worker_pid in schema_results
                for name in names
            }))
            worker_pids.update(
                worker_pid for _inputs, _auxiliary, worker_pid in schema_results
            )
            if not input_names:
                raise RuntimeError("profiler model input schema is empty")

            input_matrix = np.memmap(
                directory / "model-inputs.f32",
                dtype=np.float32,
                mode="w+",
                shape=(len(costs), len(input_names)),
            )
            del input_matrix
            if auxiliary_names:
                auxiliary_matrix = np.memmap(
                    directory / "auxiliary-targets.f64",
                    dtype=np.float64,
                    mode="w+",
                    shape=(len(costs), len(auxiliary_names)),
                )
                del auxiliary_matrix
            anchor_matrix = np.memmap(
                directory / "anchor-nk.i64",
                dtype=np.int64,
                mode="w+",
                shape=(len(costs), 2),
            )
            del anchor_matrix
            regret_targets = np.memmap(
                directory / "regret-targets.f64",
                dtype=np.float64,
                mode="w+",
                shape=(len(costs),),
            )
            del regret_targets
            matrix_tasks = tuple(
                (
                    begin,
                    end,
                    str(directory),
                    input_names,
                    auxiliary_names,
                )
                for begin, end in ranges
            )
            if executor is None:
                matrix_results = tuple(
                    _fill_parallel_profiler_model_matrix(item)
                    for item in matrix_tasks
                )
            else:
                matrix_results = tuple(executor.map(
                    _fill_parallel_profiler_model_matrix,
                    matrix_tasks,
                    chunksize=1,
                ))
            if sum(count for count, _pid in matrix_results) != len(costs):
                raise RuntimeError(
                    "parallel profiler model matrix lost or duplicated rows"
                )
            worker_pids.update(pid for _count, pid in matrix_results)

            contest_matrix = np.memmap(
                directory / "contest-ids.i64",
                dtype=np.int64,
                mode="w+",
                shape=(len(costs),),
            )
            contest_numbers: dict[tuple[RuntimeKey, str], int] = {}
            for row, cost in enumerate(costs):
                point_rows.setdefault(
                    (cost.runtime_key, cost.shape_group_id, cost.candidate_id),
                    row,
                )
                contest = (cost.runtime_key, cost.shape_group_id)
                contest_id = contest_numbers.setdefault(
                    contest,
                    len(contest_numbers),
                )
                contest_matrix[row] = contest_id
            contest_matrix.flush()
            del contest_matrix
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        _PARALLEL_MODEL_RECORD_COSTS = ()
        _PARALLEL_MODEL_RECORD_OBSERVATIONS = None
        _PARALLEL_MODEL_RECORD_EXEMPLARS = None
        _PARALLEL_MODEL_RECORD_CATALOG = None

    records = ProfilerModelRecordIndex(
        costs=costs,
        observations=observations,
        exemplars=exemplars,
        catalog=catalog,
        directory=directory,
        input_feature_names=input_names,
        auxiliary_feature_names=auxiliary_names,
        # Prediction folds resolve domain-local point inventories through this
        # one parent-owned table. Forked GPU workers share it copy-on-write,
        # replacing a complete Python scan of the transfer pool per surface.
        unique_point_count=len(point_rows),
        point_rows=point_rows,
    )
    if os.environ.get("LLAMINAR_NATIVE_VNNI_POLICY_TIMING", "0") == "1":
        matrix_bytes = sum(
            path.stat().st_size for path in directory.iterdir() if path.is_file()
        )
        print(
            "NativeVNNI profiler model record index complete "
            f"rows={records.row_count} "
            f"features={len(input_names)} auxiliary={len(auxiliary_names)} "
            f"matrix_gib={matrix_bytes / float(1 << 30):.3f} "
            f"workers={worker_count} "
            f"processes={len(worker_pids)} "
            f"elapsed={time.perf_counter() - started:.3f}s",
            file=sys.stderr,
            flush=True,
        )
    return records
