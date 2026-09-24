"""Strict Python binding for exact GPU NativeVNNI policy-tree fitting.

ABI v12 executes the complete bounded beam search on CUDA or ROCm as one captured
stream transaction. It retains grow-only matrix storage, metadata, ping-pong
frontiers, exact-signature hash storage, compact selection indices, and graph
executables across fits. Device counters connect every leaf depth without host
cardinality reads or intermediate synchronization. The older batched
leaf-primary entry point remains an actively tested diagnostic primitive for
proving the measured-p95 gate and bounded-fitting-p95 tie set independently.

Every percentile is a conservative nearest-rank order statistic over
host-authenticated FP64 values. Exact compile options, fixed-order sums, and
canonical structural signatures make the returned tree sequence identical to
the Python oracle rather than merely numerically close.

This module does not contain a CPU fallback.  Constructing an accelerator is
an explicit request to use one backend DSO and one physical device; a missing
library, backend mismatch, invalid ordinal, or runtime error is fatal.  Policy
workers live in separate processes so CUDA and HIP runtimes are never loaded
into the same address space.
"""

from __future__ import annotations

import ctypes
import math
import os
import signal
import sys
from array import array
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ERROR_CAPACITY = 2048
SCORER_ABI_VERSION = 15
TREE_MAXIMUM_FEATURE_AXES = 72
UINT32_MAX = (1 << 32) - 1
TREE_MAXIMUM_POINTS = 512
TREE_POINT_MASK_WORDS = TREE_MAXIMUM_POINTS // 64
TREE_MAXIMUM_LEAVES = 32
TREE_MAXIMUM_STRUCTURE_TOKENS = 63
TREE_MAXIMUM_HELDOUT_POINTS = 512
SUPPORTED_BACKENDS = frozenset(("cuda", "rocm"))
ACCELERATOR_ENVIRONMENT = "LLAMINAR_NATIVE_VNNI_POLICY_ACCELERATORS"
ACCELERATOR_LANES_ENVIRONMENT = (
    "LLAMINAR_NATIVE_VNNI_POLICY_LANES_PER_ACCELERATOR"
)
LIBRARY_ENVIRONMENT_BY_BACKEND = {
    "cuda": "LLAMINAR_NATIVE_VNNI_POLICY_CUDA_SCORER_LIBRARY",
    "rocm": "LLAMINAR_NATIVE_VNNI_POLICY_ROCM_SCORER_LIBRARY",
}


def arm_policy_worker_parent_death_signal(expected_parent_pid: int) -> None:
    """Kill this accelerator worker if its owning coordinator disappears.

    Accelerator workers own vendor runtime contexts and persistent device
    allocations. Normal scheduler exceptions terminate and join them from a
    ``finally`` block, but an externally killed coordinator cannot execute
    Python cleanup. Linux ``PR_SET_PDEATHSIG`` closes that lifecycle hole by
    asking the kernel to deliver ``SIGKILL`` when the current parent exits.

    Parent death can race the ``prctl`` call. The post-arm parent-PID check
    catches that interval before the worker loads either vendor runtime. A
    non-Linux platform or failed syscall is a hard error because silently
    running without the ownership guarantee would reintroduce orphaned GPU
    contexts during multi-hour policy fits.
    """

    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "accelerated policy workers require Linux parent-death signaling"
        )
    if expected_parent_pid <= 1:
        raise ValueError("policy worker expected parent PID must exceed one")

    pr_set_pdeathsig = 1
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    prctl.restype = ctypes.c_int
    if prctl(pr_set_pdeathsig, signal.SIGKILL, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            "failed to arm policy worker parent-death signal",
        )
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


@dataclass(frozen=True)
class LeafPrimaryScore:
    """Exact first-two-key result for one requested point subset.

    ``survivor_indices`` contains every candidate tied on
    ``failed_leaf_count`` and ``fitting_p95_regret``. An empty tuple is the
    deliberate representation of a
    subset with no candidate valid at every selected point; in that case the
    scalar fields carry positive infinity and ``UINT32_MAX`` respectively.
    """

    fitting_p95_regret: float
    failed_leaf_count: int
    survivor_indices: tuple[int, ...]

    @property
    def has_candidate(self) -> bool:
        """Return whether at least one common candidate survived."""

        return bool(self.survivor_indices)


class _TreePointMaskC(ctypes.Structure):
    """ctypes mirror of one fixed-width little-word-first point mask."""

    _fields_ = [
        ("words", ctypes.c_uint64 * TREE_POINT_MASK_WORDS),
    ]


class TreeThresholdOperation(IntEnum):
    """Exact device operations corresponding to policy feature families."""

    AGGREGATE_N = 0
    K = 1
    WORK_ITEMS = 2
    ASPECT_RATIO = 3
    N_TILES = 4
    K_GROUPS_PER_N_TILE = 5
    N_FINAL_TILE = 6
    N_TILE_UTILIZATION = 7
    N_PARALLEL_WAVES = 8
    N_FINAL_PARALLEL_WAVE_UTILIZATION = 9
    MN_PARALLEL_WAVES = 10
    MN_FINAL_PARALLEL_WAVE_UTILIZATION = 11
    N_TILE_ALIGNED = 12
    K_FINAL_TILE = 13
    KPART_PRODUCER_WAVES = 14
    KPART_FINAL_PRODUCER_WAVE_UTILIZATION = 15
    KPART_K_BLOCKS_PER_TILE = 16
    KPART_FINAL_K_TILE_BLOCKS = 17
    KPART_FINAL_K_TILE_UTILIZATION = 18
    KPART_K_TILE_COUNT = 19


@dataclass(frozen=True)
class TreeThresholdDescriptor:
    """One exact rational threshold consumed by fused heldout evaluation."""

    axis_priority: int
    operation: TreeThresholdOperation
    tile_width: int
    numerator: int
    denominator: int
    parallelism_width: int
    task_multiplier: int


class _TreeThresholdC(ctypes.Structure):
    """ctypes mirror of one exact integer threshold descriptor."""

    _fields_ = [
        ("axis_priority", ctypes.c_uint32),
        ("operation", ctypes.c_uint32),
        ("tile_width", ctypes.c_uint32),
        ("numerator", ctypes.c_uint64),
        ("denominator", ctypes.c_uint64),
        ("parallelism_width", ctypes.c_uint32),
        ("task_multiplier", ctypes.c_uint32),
    ]


@dataclass(frozen=True)
class TreeFeatureAxisDescriptor:
    """One compact feature operation expanded into values on the device."""

    axis_priority: int
    operation: TreeThresholdOperation
    tile_width: int


class _TreeFeatureAxisC(ctypes.Structure):
    """ctypes mirror of one device-generated feature axis."""

    _fields_ = [
        ("axis_priority", ctypes.c_uint32),
        ("operation", ctypes.c_uint32),
        ("tile_width", ctypes.c_uint32),
    ]


class TreeBoundaryPlacement(IntEnum):
    """Exact unseen-value placement understood by the device graph."""

    MIDPOINT = 0
    LOWER_EDGE = 1


class _TreeFitResultC(ctypes.Structure):
    """ctypes mirror of the fixed-width v12 tree-search result ABI."""

    _fields_ = [
        ("leaf_count", ctypes.c_uint32),
        ("leaf_masks", _TreePointMaskC * TREE_MAXIMUM_LEAVES),
        ("candidate_indices", ctypes.c_uint32 * TREE_MAXIMUM_LEAVES),
        ("structure_token_count", ctypes.c_uint32),
        (
            "structure_tokens",
            ctypes.c_uint32 * TREE_MAXIMUM_STRUCTURE_TOKENS,
        ),
        (
            "split_thresholds",
            _TreeThresholdC * TREE_MAXIMUM_STRUCTURE_TOKENS,
        ),
    ]


class _TreeFoldEvaluationC(ctypes.Structure):
    """ctypes mirror of one compact heldout decision set."""

    _fields_ = [
        ("required_point_count", ctypes.c_uint32),
        ("covered_point_count", ctypes.c_uint32),
        (
            "selected_candidate_indices",
            ctypes.c_uint32 * TREE_MAXIMUM_HELDOUT_POINTS,
        ),
    ]


class _TreeRuntimeStatsC(ctypes.Structure):
    """ctypes mirror of persistent-storage and graph diagnostics."""

    _fields_ = [
        ("prepare_count", ctypes.c_uint64),
        ("matrix_growth_count", ctypes.c_uint64),
        ("tree_scratch_growth_count", ctypes.c_uint64),
        ("graph_capture_count", ctypes.c_uint64),
        ("graph_replay_count", ctypes.c_uint64),
        ("tree_search_count", ctypes.c_uint64),
        ("tree_search_retry_count", ctypes.c_uint64),
        ("tree_search_stream_sync_count", ctypes.c_uint64),
        ("tree_search_intermediate_sync_count", ctypes.c_uint64),
        ("fused_evaluation_count", ctypes.c_uint64),
        ("final_result_d2h_bytes", ctypes.c_uint64),
        ("tree_scratch_high_water_bytes", ctypes.c_uint64),
        ("device_allocation_count", ctypes.c_uint64),
        ("device_free_count", ctypes.c_uint64),
        ("h2d_copy_count", ctypes.c_uint64),
        ("h2d_bytes", ctypes.c_uint64),
        ("d2h_copy_count", ctypes.c_uint64),
        ("d2h_bytes", ctypes.c_uint64),
        ("stream_sync_count", ctypes.c_uint64),
        ("device_sync_count", ctypes.c_uint64),
        ("captured_graph_transfer_count", ctypes.c_uint64),
        ("expanded_candidate_count", ctypes.c_uint64),
        ("structurally_unique_candidate_count", ctypes.c_uint64),
        ("last_expansion_capacity", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


@dataclass(frozen=True)
class AcceleratedTreeFitResult:
    """One canonical best tree returned for a maximum leaf budget."""

    leaf_masks: tuple[int, ...]
    candidate_indices: tuple[int, ...]
    structure_tokens: tuple[int, ...]
    split_thresholds: tuple[TreeThresholdDescriptor | None, ...]

    @property
    def has_tree(self) -> bool:
        """Return whether the fitted budget has a common-candidate tree."""

        return bool(self.leaf_masks)


@dataclass(frozen=True)
class AcceleratedFoldEvaluation:
    """Selected and exact candidate indices for one heldout leaf budget."""

    selected_candidate_indices: tuple[int, ...]
    exact_candidate_indices: tuple[int, ...]
    covered_point_count: int
    required_point_count: int


@dataclass(frozen=True)
class TreeRuntimeStats:
    """Monotonic proof of session reuse and one-transaction tree searches."""

    prepare_count: int
    matrix_growth_count: int
    tree_scratch_growth_count: int
    graph_capture_count: int
    graph_replay_count: int
    tree_search_count: int
    tree_search_retry_count: int
    tree_search_stream_sync_count: int
    tree_search_intermediate_sync_count: int
    fused_evaluation_count: int
    final_result_d2h_bytes: int
    tree_scratch_high_water_bytes: int
    device_allocation_count: int
    device_free_count: int
    h2d_copy_count: int
    h2d_bytes: int
    d2h_copy_count: int
    d2h_bytes: int
    stream_sync_count: int
    device_sync_count: int
    captured_graph_transfer_count: int
    expanded_candidate_count: int
    structurally_unique_candidate_count: int
    last_expansion_capacity: int


@dataclass(frozen=True)
class PolicyAcceleratorSpec:
    """Immutable worker ownership for one backend-local physical device."""

    backend: str
    device_ordinal: int
    library_path: Path
    lane_count: int = 1

    @property
    def label(self) -> str:
        """Return the stable user-facing accelerator identity."""

        return f"{self.backend}:{self.device_ordinal}"

    def create_scorer(self) -> "NativeVNNILeafPrimaryScorer":
        """Load exactly this spec's vendor DSO in the current worker process."""

        return NativeVNNILeafPrimaryScorer(
            self.library_path,
            backend=self.backend,
            device_ordinal=self.device_ordinal,
        )


def policy_accelerator_specs_from_environment(
    environment: Mapping[str, str] | None = None,
) -> tuple[PolicyAcceleratorSpec, ...]:
    """Parse strict per-device policy workers from the canonical environment.

    ``LLAMINAR_NATIVE_VNNI_POLICY_ACCELERATORS`` is a comma-separated list such
    as ``cuda:0,cuda:1,rocm:0``.  Every referenced backend also requires its
    backend-specific scorer-library variable.  Empty configuration means the
    caller deliberately selected the canonical CPU fitter; malformed or
    incomplete non-empty configuration is always an error.
    """

    source = os.environ if environment is None else environment
    raw = source.get(ACCELERATOR_ENVIRONMENT, "").strip()
    if not raw:
        return ()

    identities = []
    for raw_token in raw.split(","):
        token = raw_token.strip()
        pieces = token.split(":")
        if len(pieces) != 2 or pieces[0].lower() not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"invalid policy accelerator {token!r}; expected cuda:N or rocm:N"
            )
        backend = pieces[0].lower()
        try:
            ordinal = int(pieces[1])
        except ValueError as error:
            raise ValueError(
                f"invalid policy accelerator ordinal in {token!r}"
            ) from error
        if ordinal < 0 or str(ordinal) != pieces[1]:
            raise ValueError(
                f"policy accelerator ordinal must be canonical and non-negative: "
                f"{token!r}"
            )
        identity = (backend, ordinal)
        if identity in identities:
            raise ValueError(f"duplicate policy accelerator {token!r}")
        identities.append(identity)

    libraries = {}
    for backend in {backend for backend, _ordinal in identities}:
        variable = LIBRARY_ENVIRONMENT_BY_BACKEND[backend]
        raw_path = source.get(variable, "").strip()
        if not raw_path:
            raise ValueError(
                f"{ACCELERATOR_ENVIRONMENT} requests {backend}, but {variable} "
                "is unset"
            )
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"{variable} does not name a scorer DSO: {path}"
            )
        libraries[backend] = path

    return tuple(
        PolicyAcceleratorSpec(
            backend=backend,
            device_ordinal=ordinal,
            library_path=libraries[backend],
        )
        for backend, ordinal in identities
    )


def policy_accelerator_worker_specs(
    physical_specs: tuple[PolicyAcceleratorSpec, ...],
    environment: Mapping[str, str] | None = None,
) -> tuple[PolicyAcceleratorSpec, ...]:
    """Attach explicitly requested stream lanes to each physical accelerator.

    One backend-local process owns the vendor context for one physical device.
    Each requested lane owns a distinct stream, graph cache, and grow-only tree
    workspace inside that process, allowing shrinking hierarchy levels from one
    fit to overlap the broad kernels of another without vendor-context switching.
    The production default remains one lane because a maximum tree workspace is
    large; explicit higher counts are validated and remain visible in the spec.

    An explicit environment override remains useful for controlled small-domain
    experiments.  It is deliberately strict and never changes devices or moves
    a failed task onto CPU.
    """

    if not physical_specs:
        return ()
    source = os.environ if environment is None else environment
    raw_lanes = source.get(ACCELERATOR_LANES_ENVIRONMENT, "1").strip()
    try:
        lanes = int(raw_lanes)
    except ValueError as error:
        raise ValueError(
            f"{ACCELERATOR_LANES_ENVIRONMENT} must be a positive integer"
        ) from error
    if lanes <= 0 or str(lanes) != raw_lanes:
        raise ValueError(
            f"{ACCELERATOR_LANES_ENVIRONMENT} must be a canonical positive integer"
        )
    return tuple(
        PolicyAcceleratorSpec(
            backend=spec.backend,
            device_ordinal=spec.device_ordinal,
            library_path=spec.library_path,
            lane_count=lanes,
        )
        for spec in physical_specs
    )


def _as_pointer(values: array, ctype: type[ctypes._SimpleCData]):
    """Return a typed pointer into one non-empty mutable ``array`` buffer."""

    return ctypes.cast(
        ctypes.addressof(ctype.from_buffer(values)),
        ctypes.POINTER(ctype),
    )


def _pack_subset_masks(
    subsets: Iterable[int],
    *,
    point_count: int,
) -> tuple[tuple[int, ...], array, int]:
    """Validate Python integer masks and pack little-word-first uint64 rows.

    Python integers make the call site independent of the number of measured
    points.  Bit ``p`` selects regret-matrix row ``p``.  The C ABI consumes the
    same logical representation split into consecutive 64-bit words.
    """

    subset_values = tuple(subsets)
    if not subset_values:
        raise ValueError("leaf scorer requires at least one point subset")
    word_count = (point_count + 63) // 64
    packed = array("Q")
    if packed.itemsize != ctypes.sizeof(ctypes.c_uint64):
        raise RuntimeError("Python array('Q') is not a native uint64 buffer")
    for subset in subset_values:
        if isinstance(subset, bool) or not isinstance(subset, int):
            raise TypeError("point subsets must be Python integer bitmasks")
        if subset <= 0:
            raise ValueError("every leaf-scorer subset must select a point")
        if subset >> point_count:
            raise ValueError("leaf-scorer subset selects a point outside the matrix")
        packed.extend(
            (subset >> (word * 64)) & ((1 << 64) - 1)
            for word in range(word_count)
        )
    return subset_values, packed, word_count


def _pack_tree_point_masks(
    masks: Sequence[int],
    *,
    label: str,
) -> array:
    """Pack v12 fixed-width masks while rejecting bits outside the tree ABI."""

    packed = array("Q")
    if packed.itemsize != ctypes.sizeof(ctypes.c_uint64):
        raise RuntimeError("Python array('Q') is not a native uint64 buffer")
    for mask in masks:
        if isinstance(mask, bool) or not isinstance(mask, int):
            raise TypeError(f"{label} must contain Python integer bitmasks")
        if mask < 0 or mask >> TREE_MAXIMUM_POINTS:
            raise ValueError(
                f"{label} contains a bit outside the {TREE_MAXIMUM_POINTS}-point "
                "tree ABI"
            )
        packed.extend(
            (mask >> (word * 64)) & ((1 << 64) - 1)
            for word in range(TREE_POINT_MASK_WORDS)
        )
    return packed


def _native_array(typecode: str, values: Sequence[int | float]) -> array:
    """Reuse an already native-width array instead of copying its contents.

    Grouped CV constructs regret matrices as ``array('d')`` before entering the
    backend binding.  Re-copying those arrays adds host bandwidth and also
    defeats the session's object-identity fast path.  Structural metadata may
    likewise be retained by the feature-metadata cache as a native uint32
    array.  All other sequence types still receive the same validated copy as
    before.
    """

    if isinstance(values, array) and values.typecode == typecode:
        return values
    return array(typecode, values)


def _threshold_descriptor_from_c(
    threshold: _TreeThresholdC,
) -> TreeThresholdDescriptor:
    """Validate and materialize one selected device threshold."""

    try:
        operation = TreeThresholdOperation(threshold.operation)
    except ValueError as error:
        raise RuntimeError(
            "device tree returned an unknown threshold operation"
        ) from error
    if (
        threshold.axis_priority >= TREE_MAXIMUM_FEATURE_AXES
        or threshold.denominator == 0
        or threshold.parallelism_width == 0
        or threshold.task_multiplier == 0
    ):
        raise RuntimeError("device tree returned a malformed threshold")
    return TreeThresholdDescriptor(
        axis_priority=int(threshold.axis_priority),
        operation=operation,
        tile_width=int(threshold.tile_width),
        numerator=int(threshold.numerator),
        denominator=int(threshold.denominator),
        parallelism_width=int(threshold.parallelism_width),
        task_multiplier=int(threshold.task_multiplier),
    )


def _pack_feature_axes(
    descriptors: Sequence[TreeFeatureAxisDescriptor],
) -> ctypes.Array:
    """Pack one compact, uniquely ranked feature-axis inventory."""

    if not descriptors:
        raise ValueError("device tree search requires at least one feature axis")
    priorities = set()
    values = []
    for descriptor in descriptors:
        if not isinstance(descriptor.operation, TreeThresholdOperation):
            raise TypeError("feature-axis operation must be TreeThresholdOperation")
        if not 0 <= descriptor.axis_priority < TREE_MAXIMUM_FEATURE_AXES:
            raise ValueError(
                "feature-axis priority must be in "
                f"[0, {TREE_MAXIMUM_FEATURE_AXES})"
            )
        if descriptor.axis_priority in priorities:
            raise ValueError("feature-axis priorities must be unique")
        priorities.add(descriptor.axis_priority)
        if not 0 <= descriptor.tile_width <= UINT32_MAX:
            raise ValueError("feature-axis tile width exceeds uint32")
        if (
            descriptor.operation <= TreeThresholdOperation.ASPECT_RATIO
            and descriptor.tile_width != 0
        ) or (
            descriptor.operation > TreeThresholdOperation.ASPECT_RATIO
            and descriptor.tile_width == 0
        ):
            raise ValueError("feature-axis tile width does not match its operation")
        values.append(_TreeFeatureAxisC(
            descriptor.axis_priority,
            int(descriptor.operation),
            descriptor.tile_width,
        ))
    return (_TreeFeatureAxisC * len(values))(*values)


class NativeVNNILeafPrimaryScorer:
    """One strict binding to a CUDA or ROCm exact policy-search DSO.

    The object validates the DSO's compiled backend and device inventory at
    construction. It prepares one opaque device session for the current regret
    matrix and retains that session across leaf diagnostics and complete tree
    searches. Moving to a new matrix explicitly destroys the old session before
    uploading the replacement. The DSO owns all backend runtime calls and device
    allocations; this Python layer owns only canonical host buffers and the
    opaque handle.
    """

    def __init__(
        self,
        library_path: str | Path,
        *,
        backend: str,
        device_ordinal: int,
    ) -> None:
        normalized_backend = backend.strip().lower()
        if normalized_backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                "leaf scorer backend must be one of: "
                + ", ".join(sorted(SUPPORTED_BACKENDS))
            )
        if isinstance(device_ordinal, bool) or not isinstance(device_ordinal, int):
            raise TypeError("leaf scorer device ordinal must be an integer")
        if device_ordinal < 0:
            raise ValueError("leaf scorer device ordinal must be non-negative")

        path = Path(library_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"leaf scorer DSO does not exist: {path}")
        self._library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
        self._configure_abi()

        actual_abi = self._library.llaminarNativeVNNILeafPrimaryScorerAbiVersion()
        if actual_abi != SCORER_ABI_VERSION:
            raise RuntimeError(
                f"leaf scorer ABI mismatch: Python requires {SCORER_ABI_VERSION}, "
                f"DSO reports {actual_abi}"
            )

        actual_backend = self._library.llaminarNativeVNNILeafPrimaryScorerBackend()
        if actual_backend is None:
            raise RuntimeError("leaf scorer DSO returned a null backend name")
        actual_backend_name = actual_backend.decode("ascii")
        if actual_backend_name != normalized_backend:
            raise RuntimeError(
                f"leaf scorer backend mismatch: requested {normalized_backend!r}, "
                f"DSO reports {actual_backend_name!r}"
            )

        error = ctypes.create_string_buffer(ERROR_CAPACITY)
        device_count = self._library.llaminarNativeVNNILeafPrimaryScorerDeviceCount(
            error, len(error)
        )
        if device_count < 0:
            detail = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(f"{normalized_backend} device query failed: {detail}")
        if device_ordinal >= device_count:
            raise ValueError(
                f"{normalized_backend} device ordinal {device_ordinal} is outside "
                f"the visible inventory of {device_count} device(s)"
            )

        self.backend = normalized_backend
        self.device_ordinal = device_ordinal
        self.device_count = device_count
        self.library_path = path
        self._session: ctypes.c_void_p | None = None
        self._session_fitting_regrets: array | None = None
        self._session_measured_p95_regrets: array | None = None
        self._session_dimensions: tuple[int, int] | None = None

    def _configure_abi(self) -> None:
        """Declare every C function signature before making a runtime call."""

        library = self._library
        library.llaminarNativeVNNILeafPrimaryScorerAbiVersion.argtypes = []
        library.llaminarNativeVNNILeafPrimaryScorerAbiVersion.restype = (
            ctypes.c_uint32
        )
        library.llaminarNativeVNNILeafPrimaryScorerBackend.argtypes = []
        library.llaminarNativeVNNILeafPrimaryScorerBackend.restype = ctypes.c_char_p
        library.llaminarNativeVNNILeafPrimaryScorerDeviceCount.argtypes = [
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.llaminarNativeVNNILeafPrimaryScorerDeviceCount.restype = ctypes.c_int
        library.llaminarNativeVNNILeafPrimaryScorerCreate.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.llaminarNativeVNNILeafPrimaryScorerCreate.restype = ctypes.c_void_p
        library.llaminarNativeVNNILeafPrimaryScorerPrepare.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.llaminarNativeVNNILeafPrimaryScorerPrepare.restype = ctypes.c_int
        library.llaminarNativeVNNILeafPrimaryScorerScore.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.llaminarNativeVNNILeafPrimaryScorerScore.restype = ctypes.c_int
        library.llaminarNativeVNNITreeSearch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(_TreeFeatureAxisC),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(_TreeFitResultC),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.llaminarNativeVNNITreeSearch.restype = ctypes.c_int
        library.llaminarNativeVNNITreeSearchAndEvaluate.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(_TreeFeatureAxisC),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(_TreeFoldEvaluationC),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.llaminarNativeVNNITreeSearchAndEvaluate.restype = ctypes.c_int
        library.llaminarNativeVNNITreeRuntimeStats.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_TreeRuntimeStatsC),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.llaminarNativeVNNITreeRuntimeStats.restype = ctypes.c_int
        library.llaminarNativeVNNILeafPrimaryScorerDestroy.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.llaminarNativeVNNILeafPrimaryScorerDestroy.restype = ctypes.c_int

    def _prepare_session(
        self,
        fitting_regret_buffer: array,
        measured_p95_regret_buffer: array,
        *,
        point_count: int,
        candidate_count: int,
    ) -> None:
        """Upload a changed matrix once and retain its backend-owned session."""

        dimensions = (point_count, candidate_count)
        if (
            self._session is not None
            and self._session_fitting_regrets is fitting_regret_buffer
            and self._session_measured_p95_regrets is measured_p95_regret_buffer
            and self._session_dimensions == dimensions
        ):
            return
        error = ctypes.create_string_buffer(ERROR_CAPACITY)
        if self._session is None:
            address = self._library.llaminarNativeVNNILeafPrimaryScorerCreate(
                self.device_ordinal,
                _as_pointer(fitting_regret_buffer, ctypes.c_double),
                _as_pointer(measured_p95_regret_buffer, ctypes.c_double),
                point_count,
                candidate_count,
                error,
                len(error),
            )
            if not address:
                detail = error.value.decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"{self.backend}:{self.device_ordinal} scorer preparation "
                    f"failed: {detail}"
                )
            self._session = ctypes.c_void_p(address)
        else:
            result = self._library.llaminarNativeVNNILeafPrimaryScorerPrepare(
                self._session,
                _as_pointer(fitting_regret_buffer, ctypes.c_double),
                _as_pointer(measured_p95_regret_buffer, ctypes.c_double),
                point_count,
                candidate_count,
                error,
                len(error),
            )
            if result != 0:
                detail = error.value.decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"{self.backend}:{self.device_ordinal} scorer matrix "
                    f"publication failed with ABI status {result}: {detail}"
                )
        # Retain the exact array object as both the session identity and a clear
        # ownership signal: callers must not mutate it while the session lives.
        self._session_fitting_regrets = fitting_regret_buffer
        self._session_measured_p95_regrets = measured_p95_regret_buffer
        self._session_dimensions = dimensions

    def runtime_stats(self) -> TreeRuntimeStats:
        """Read graph, allocation, retry, and synchronization diagnostics."""

        if self._session is None:
            raise RuntimeError("tree runtime stats require a prepared session")
        raw = _TreeRuntimeStatsC()
        error = ctypes.create_string_buffer(ERROR_CAPACITY)
        result = self._library.llaminarNativeVNNITreeRuntimeStats(
            self._session, ctypes.byref(raw), error, len(error)
        )
        if result != 0:
            detail = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{self.backend}:{self.device_ordinal} runtime stats failed "
                f"with ABI status {result}: {detail}"
            )
        return TreeRuntimeStats(
            prepare_count=raw.prepare_count,
            matrix_growth_count=raw.matrix_growth_count,
            tree_scratch_growth_count=raw.tree_scratch_growth_count,
            graph_capture_count=raw.graph_capture_count,
            graph_replay_count=raw.graph_replay_count,
            tree_search_count=raw.tree_search_count,
            tree_search_retry_count=raw.tree_search_retry_count,
            tree_search_stream_sync_count=raw.tree_search_stream_sync_count,
            tree_search_intermediate_sync_count=
                raw.tree_search_intermediate_sync_count,
            fused_evaluation_count=raw.fused_evaluation_count,
            final_result_d2h_bytes=raw.final_result_d2h_bytes,
            tree_scratch_high_water_bytes=raw.tree_scratch_high_water_bytes,
            device_allocation_count=raw.device_allocation_count,
            device_free_count=raw.device_free_count,
            h2d_copy_count=raw.h2d_copy_count,
            h2d_bytes=raw.h2d_bytes,
            d2h_copy_count=raw.d2h_copy_count,
            d2h_bytes=raw.d2h_bytes,
            stream_sync_count=raw.stream_sync_count,
            device_sync_count=raw.device_sync_count,
            captured_graph_transfer_count=raw.captured_graph_transfer_count,
            expanded_candidate_count=raw.expanded_candidate_count,
            structurally_unique_candidate_count=
                raw.structurally_unique_candidate_count,
            last_expansion_capacity=raw.last_expansion_capacity,
        )

    def close(self) -> None:
        """Release the prepared device matrix, stream, and reusable scratch."""

        if self._session is None:
            return
        session = self._session
        self._session = None
        self._session_fitting_regrets = None
        self._session_measured_p95_regrets = None
        self._session_dimensions = None
        error = ctypes.create_string_buffer(ERROR_CAPACITY)
        result = self._library.llaminarNativeVNNILeafPrimaryScorerDestroy(
            session, error, len(error)
        )
        if result != 0:
            detail = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{self.backend}:{self.device_ordinal} scorer destruction failed "
                f"with ABI status {result}: {detail}"
            )

    def score(
        self,
        fitting_regrets: Sequence[float],
        measured_p95_regrets: Sequence[float],
        *,
        point_count: int,
        candidate_count: int,
        subsets: Iterable[int],
    ) -> tuple[LeafPrimaryScore, ...]:
        """Score every subset and return all exact primary-key survivors.

        Both inputs are row-major ``[point_count, candidate_count]`` FP64
        matrices. Finite values are valid regrets and positive infinity marks
        an unavailable candidate. The matrices must carry identical
        availability. NaN and negative infinity are rejected so corrupt learner
        state cannot silently turn into candidate invalidity.
        """

        for name, value in (
            ("point_count", point_count),
            ("candidate_count", candidate_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0 or value > UINT32_MAX:
                raise ValueError(f"{name} must be in [1, UINT32_MAX]")

        expected_regrets = point_count * candidate_count
        if len(fitting_regrets) != expected_regrets:
            raise ValueError(
                f"fitting regret matrix has {len(fitting_regrets)} values; "
                f"expected {expected_regrets}"
            )
        if len(measured_p95_regrets) != expected_regrets:
            raise ValueError(
                "measured p95 regret matrix has "
                f"{len(measured_p95_regrets)} values; expected {expected_regrets}"
            )
        fitting_regret_buffer = (
            fitting_regrets
            if isinstance(fitting_regrets, array)
            and fitting_regrets.typecode == "d"
            else array("d", fitting_regrets)
        )
        measured_p95_regret_buffer = (
            measured_p95_regrets
            if isinstance(measured_p95_regrets, array)
            and measured_p95_regrets.typecode == "d"
            else array("d", measured_p95_regrets)
        )
        if (
            fitting_regret_buffer.itemsize != ctypes.sizeof(ctypes.c_double)
            or measured_p95_regret_buffer.itemsize
            != ctypes.sizeof(ctypes.c_double)
        ):
            raise RuntimeError("Python array('d') is not a native double buffer")
        prepared = (
            self._session is not None
            and self._session_fitting_regrets is fitting_regret_buffer
            and self._session_measured_p95_regrets is measured_p95_regret_buffer
            and self._session_dimensions == (point_count, candidate_count)
        )
        if not prepared:
            for fitting, measured in zip(
                fitting_regret_buffer, measured_p95_regret_buffer
            ):
                if any(
                    math.isnan(value) or (math.isinf(value) and value < 0.0)
                    for value in (fitting, measured)
                ):
                    raise ValueError(
                        "regret matrix values must be finite or positive infinity"
                    )
                if math.isfinite(fitting) != math.isfinite(measured):
                    raise ValueError(
                        "fitting and measured p95 matrices changed candidate "
                        "availability"
                    )
            self._prepare_session(
                fitting_regret_buffer,
                measured_p95_regret_buffer,
                point_count=point_count,
                candidate_count=candidate_count,
            )

        subset_values, subset_buffer, subset_word_count = _pack_subset_masks(
            subsets, point_count=point_count
        )
        subset_count = len(subset_values)
        if subset_count > UINT32_MAX:
            raise ValueError("subset count exceeds the uint32 ABI")
        candidate_word_count = (candidate_count + 63) // 64

        best_fitting_p95 = array("d", (0.0 for _ in range(subset_count)))
        best_failed_leaves = array("I", (0 for _ in range(subset_count)))
        survivors = array(
            "Q", (0 for _ in range(subset_count * candidate_word_count))
        )
        if best_failed_leaves.itemsize != ctypes.sizeof(ctypes.c_uint32):
            raise RuntimeError("Python array('I') is not a native uint32 buffer")
        error = ctypes.create_string_buffer(ERROR_CAPACITY)
        result = self._library.llaminarNativeVNNILeafPrimaryScorerScore(
            self._session,
            _as_pointer(subset_buffer, ctypes.c_uint64),
            subset_count,
            subset_word_count,
            _as_pointer(best_fitting_p95, ctypes.c_double),
            _as_pointer(best_failed_leaves, ctypes.c_uint32),
            _as_pointer(survivors, ctypes.c_uint64),
            error,
            len(error),
        )
        if result != 0:
            detail = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{self.backend}:{self.device_ordinal} leaf scoring failed "
                f"with ABI status {result}: {detail}"
            )

        scores = []
        for subset_index in range(subset_count):
            survivor_indices = []
            for word in range(candidate_word_count):
                mask = survivors[subset_index * candidate_word_count + word]
                while mask:
                    bit = (mask & -mask).bit_length() - 1
                    candidate = word * 64 + bit
                    if candidate >= candidate_count:
                        raise RuntimeError(
                            "leaf scorer returned an out-of-range survivor bit"
                        )
                    survivor_indices.append(candidate)
                    mask &= mask - 1
            fitting_p95 = best_fitting_p95[subset_index]
            failed_leaf_count = best_failed_leaves[subset_index]
            if survivor_indices:
                if (
                    not math.isfinite(fitting_p95)
                    or failed_leaf_count not in (0, 1)
                ):
                    raise RuntimeError(
                        "leaf scorer returned survivors with an invalid primary key"
                    )
            elif not (
                math.isinf(fitting_p95)
                and fitting_p95 > 0.0
                and failed_leaf_count == UINT32_MAX
            ):
                raise RuntimeError(
                    "leaf scorer returned no survivors without the invalid sentinel"
                )
            scores.append(LeafPrimaryScore(
                fitting_p95_regret=fitting_p95,
                failed_leaf_count=failed_leaf_count,
                survivor_indices=tuple(survivor_indices),
            ))
        return tuple(scores)

    def fit_tree_budgets(
        self,
        fitting_regrets: Sequence[float],
        measured_p95_regrets: Sequence[float],
        measured_mean_regrets: Sequence[float],
        measured_max_regrets: Sequence[float],
        *,
        point_count: int,
        candidate_count: int,
        point_group_ranks: Sequence[int],
        training_aggregate_n: Sequence[int],
        training_k: Sequence[int],
        training_launch_k_tiles: Sequence[int],
        feature_axes: Sequence[TreeFeatureAxisDescriptor],
        boundary_placement: TreeBoundaryPlacement,
        parallelism_width: int,
        task_multiplier: int,
        min_shape_groups_per_leaf: int,
        max_leaves: int,
    ) -> tuple[AcceleratedTreeFitResult, ...]:
        """Run one complete bounded beam search on the selected accelerator.

        This is a strict device route. Unsupported dimensions, malformed
        feature metadata, an ABI error, or a backend failure is reported to the
        caller; the method never substitutes the Python search.
        """

        if not 1 <= point_count <= TREE_MAXIMUM_POINTS:
            raise ValueError(
                f"device tree search point_count must be in [1, "
                f"{TREE_MAXIMUM_POINTS}]"
            )
        if not 1 <= max_leaves <= TREE_MAXIMUM_LEAVES:
            raise ValueError(
                f"device tree search max_leaves must be in [1, "
                f"{TREE_MAXIMUM_LEAVES}]"
            )
        if candidate_count <= 0 or candidate_count > UINT32_MAX:
            raise ValueError("candidate_count must be in [1, UINT32_MAX]")
        matrix_count = point_count * candidate_count
        matrices = (
            ("fitting", fitting_regrets),
            ("measured p95", measured_p95_regrets),
            ("measured mean", measured_mean_regrets),
            ("measured maximum", measured_max_regrets),
        )
        for name, values in matrices:
            if len(values) != matrix_count:
                raise ValueError(
                    f"{name} matrix has {len(values)} values; "
                    f"expected {matrix_count}"
                )
        if len(point_group_ranks) != point_count:
            raise ValueError("point_group_ranks must contain one value per point")
        if len(training_aggregate_n) != point_count or len(training_k) != point_count:
            raise ValueError("training N/K must contain one value per point")
        if len(training_launch_k_tiles) != point_count:
            raise ValueError(
                "training K-tile geometry must contain one value per point"
            )
        if any(
            value <= 0 or value > (1 << 64) - 1
            for value in (*training_aggregate_n, *training_k)
        ):
            raise ValueError("training N/K must fit positive uint64")
        if any(
            isinstance(value, bool) or not 0 <= value <= UINT32_MAX
            for value in training_launch_k_tiles
        ):
            raise ValueError("training K-tile geometry must fit uint32")
        if any(
            isinstance(rank, bool) or not 0 <= rank < TREE_MAXIMUM_POINTS
            for rank in point_group_ranks
        ):
            raise ValueError("point-group ranks must fit the tree point ABI")
        axis_count = len(feature_axes)
        axis_buffer = _pack_feature_axes(feature_axes)
        if not isinstance(boundary_placement, TreeBoundaryPlacement):
            raise TypeError("boundary_placement must be TreeBoundaryPlacement")
        if not 1 <= parallelism_width <= UINT32_MAX:
            raise ValueError("parallelism_width must fit positive uint32")
        if not 1 <= task_multiplier <= UINT32_MAX:
            raise ValueError("task_multiplier must fit positive uint32")
        if min_shape_groups_per_leaf <= 0:
            raise ValueError("min_shape_groups_per_leaf must be positive")

        double_buffers = tuple(
            _native_array("d", values) for _name, values in matrices
        )
        for values in zip(*double_buffers):
            if any(
                math.isnan(value) or (math.isinf(value) and value < 0.0)
                for value in values
            ):
                raise ValueError(
                    "tree-search regret matrices must contain finite values or "
                    "positive infinity"
                )
            availability = tuple(math.isfinite(value) for value in values)
            if len(set(availability)) != 1:
                raise ValueError(
                    "tree-search regret matrices changed candidate availability"
                )
        point_rank_buffer = _native_array("I", point_group_ranks)
        training_n_buffer = _native_array("Q", training_aggregate_n)
        training_k_buffer = _native_array("Q", training_k)
        training_k_tiles_buffer = _native_array(
            "I", training_launch_k_tiles
        )
        if point_rank_buffer.itemsize != 4:
            raise RuntimeError("Python array('I') is not a native uint32 buffer")
        if training_n_buffer.itemsize != 8 or training_k_buffer.itemsize != 8:
            raise RuntimeError("Python array('Q') is not a native uint64 buffer")
        if training_k_tiles_buffer.itemsize != 4:
            raise RuntimeError("Python array('I') is not a native uint32 buffer")

        self._prepare_session(
            double_buffers[0],
            double_buffers[1],
            point_count=point_count,
            candidate_count=candidate_count,
        )
        output_type = _TreeFitResultC * max_leaves
        output = output_type()
        error = ctypes.create_string_buffer(ERROR_CAPACITY)
        result = self._library.llaminarNativeVNNITreeSearch(
            self._session,
            _as_pointer(double_buffers[2], ctypes.c_double),
            _as_pointer(double_buffers[3], ctypes.c_double),
            _as_pointer(point_rank_buffer, ctypes.c_uint32),
            _as_pointer(training_n_buffer, ctypes.c_uint64),
            _as_pointer(training_k_buffer, ctypes.c_uint64),
            _as_pointer(training_k_tiles_buffer, ctypes.c_uint32),
            axis_buffer,
            axis_count,
            int(boundary_placement),
            parallelism_width,
            task_multiplier,
            min_shape_groups_per_leaf,
            max_leaves,
            output,
            error,
            len(error),
        )
        if result != 0:
            detail = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{self.backend}:{self.device_ordinal} tree search failed "
                f"with ABI status {result}: {detail}"
            )

        fitted = []
        for budget, raw in enumerate(output, start=1):
            if raw.leaf_count > min(budget, TREE_MAXIMUM_LEAVES):
                raise RuntimeError("tree search exceeded its published leaf budget")
            if raw.structure_token_count > TREE_MAXIMUM_STRUCTURE_TOKENS:
                raise RuntimeError("tree search returned too many structure tokens")
            if raw.leaf_count == 0:
                if raw.structure_token_count != 0:
                    raise RuntimeError("empty tree result retained structure tokens")
                fitted.append(AcceleratedTreeFitResult((), (), (), ()))
                continue
            if raw.structure_token_count != 2 * raw.leaf_count - 1:
                raise RuntimeError("tree search structure is not a full binary tree")
            structure_tokens = tuple(
                raw.structure_tokens[:raw.structure_token_count]
            )
            if any(token not in (0, 1) for token in structure_tokens):
                raise RuntimeError("tree search returned an invalid node marker")
            fitted.append(AcceleratedTreeFitResult(
                leaf_masks=tuple(
                    sum(
                        int(raw.leaf_masks[leaf].words[word]) << (word * 64)
                        for word in range(TREE_POINT_MASK_WORDS)
                    )
                    for leaf in range(raw.leaf_count)
                ),
                candidate_indices=tuple(
                    raw.candidate_indices[:raw.leaf_count]
                ),
                structure_tokens=structure_tokens,
                split_thresholds=tuple(
                    _threshold_descriptor_from_c(raw.split_thresholds[position])
                    if token == 1 else None
                    for position, token in enumerate(structure_tokens)
                ),
            ))
        return tuple(fitted)

    def fit_tree_budgets_and_evaluate(
        self,
        fitting_regrets: Sequence[float],
        measured_p95_regrets: Sequence[float],
        measured_mean_regrets: Sequence[float],
        measured_max_regrets: Sequence[float],
        *,
        point_count: int,
        candidate_count: int,
        point_group_ranks: Sequence[int],
        training_aggregate_n: Sequence[int],
        training_k: Sequence[int],
        training_launch_k_tiles: Sequence[int],
        feature_axes: Sequence[TreeFeatureAxisDescriptor],
        boundary_placement: TreeBoundaryPlacement,
        parallelism_width: int,
        task_multiplier: int,
        heldout_aggregate_n: Sequence[int],
        heldout_k: Sequence[int],
        heldout_launch_k_tiles: Sequence[int],
        heldout_measured_p95_regrets: Sequence[float],
        heldout_measured_mean_regrets: Sequence[float],
        heldout_measured_max_regrets: Sequence[float],
        min_shape_groups_per_leaf: int,
        max_leaves: int,
    ) -> tuple[AcceleratedFoldEvaluation, ...]:
        """Fit and evaluate grouped-CV budgets in one captured transaction.

        Raw heldout geometry and measured candidate surfaces are uploaded to
        persistent session buffers. CUDA/ROCm evaluates exact integer tree
        predicates, selected-candidate availability, and canonical measured
        winners after the search nodes in the same graph. No tree structure is
        copied to Python on this route.
        """

        if not 1 <= point_count <= TREE_MAXIMUM_POINTS:
            raise ValueError(
                f"device tree search point_count must be in [1, "
                f"{TREE_MAXIMUM_POINTS}]"
            )
        heldout_point_count = len(heldout_aggregate_n)
        if not 1 <= heldout_point_count <= TREE_MAXIMUM_HELDOUT_POINTS:
            raise ValueError(
                "fused heldout point count must be in [1, "
                f"{TREE_MAXIMUM_HELDOUT_POINTS}]"
            )
        if len(heldout_k) != heldout_point_count:
            raise ValueError("heldout K must contain one value per point")
        if len(heldout_launch_k_tiles) != heldout_point_count:
            raise ValueError(
                "heldout K-tile geometry must contain one value per point"
            )
        if not 1 <= max_leaves <= TREE_MAXIMUM_LEAVES:
            raise ValueError(
                f"device tree search max_leaves must be in [1, "
                f"{TREE_MAXIMUM_LEAVES}]"
            )
        if candidate_count <= 0 or candidate_count > UINT32_MAX:
            raise ValueError("candidate_count must be in [1, UINT32_MAX]")
        if min_shape_groups_per_leaf <= 0:
            raise ValueError("min_shape_groups_per_leaf must be positive")

        matrix_count = point_count * candidate_count
        training_matrices = (
            ("fitting", fitting_regrets),
            ("measured p95", measured_p95_regrets),
            ("measured mean", measured_mean_regrets),
            ("measured maximum", measured_max_regrets),
        )
        for name, values in training_matrices:
            if len(values) != matrix_count:
                raise ValueError(
                    f"{name} matrix has {len(values)} values; "
                    f"expected {matrix_count}"
                )
        heldout_matrix_count = heldout_point_count * candidate_count
        heldout_matrices = (
            ("heldout measured p95", heldout_measured_p95_regrets),
            ("heldout measured mean", heldout_measured_mean_regrets),
            ("heldout measured maximum", heldout_measured_max_regrets),
        )
        for name, values in heldout_matrices:
            if len(values) != heldout_matrix_count:
                raise ValueError(
                    f"{name} matrix has {len(values)} values; "
                    f"expected {heldout_matrix_count}"
                )
        if len(point_group_ranks) != point_count:
            raise ValueError("point_group_ranks must contain one value per point")
        if len(training_aggregate_n) != point_count or len(training_k) != point_count:
            raise ValueError("training N/K must contain one value per point")
        if len(training_launch_k_tiles) != point_count:
            raise ValueError(
                "training K-tile geometry must contain one value per point"
            )
        if any(
            value <= 0 or value > (1 << 64) - 1
            for value in (*training_aggregate_n, *training_k)
        ):
            raise ValueError("training N/K must fit positive uint64")
        if any(
            isinstance(value, bool) or not 0 <= value <= UINT32_MAX
            for value in training_launch_k_tiles
        ):
            raise ValueError("training K-tile geometry must fit uint32")
        if any(
            isinstance(rank, bool) or not 0 <= rank < TREE_MAXIMUM_POINTS
            for rank in point_group_ranks
        ):
            raise ValueError("point-group ranks must fit the tree point ABI")
        axis_count = len(feature_axes)
        axis_buffer = _pack_feature_axes(feature_axes)
        if not isinstance(boundary_placement, TreeBoundaryPlacement):
            raise TypeError("boundary_placement must be TreeBoundaryPlacement")
        if not 1 <= parallelism_width <= UINT32_MAX:
            raise ValueError("parallelism_width must fit positive uint32")
        if not 1 <= task_multiplier <= UINT32_MAX:
            raise ValueError("task_multiplier must fit positive uint32")

        training_double_buffers = tuple(
            _native_array("d", values) for _name, values in training_matrices
        )
        for values in zip(*training_double_buffers):
            if any(
                math.isnan(value) or (math.isinf(value) and value < 0.0)
                for value in values
            ):
                raise ValueError(
                    "tree-search regret matrices must contain finite values or "
                    "positive infinity"
                )
            if len({math.isfinite(value) for value in values}) != 1:
                raise ValueError(
                    "tree-search regret matrices changed candidate availability"
                )
        heldout_double_buffers = tuple(
            _native_array("d", values) for _name, values in heldout_matrices
        )
        for values in zip(*heldout_double_buffers):
            if any(
                math.isnan(value) or (math.isinf(value) and value < 0.0)
                for value in values
            ):
                raise ValueError(
                    "heldout matrices must contain finite values or positive "
                    "infinity"
                )
            if len({math.isfinite(value) for value in values}) != 1:
                raise ValueError(
                    "heldout matrices changed candidate availability"
                )
        for point in range(heldout_point_count):
            row = slice(
                point * candidate_count,
                (point + 1) * candidate_count,
            )
            if not any(math.isfinite(value) for value in heldout_double_buffers[0][row]):
                raise ValueError("every heldout point requires one candidate")

        point_rank_buffer = _native_array("I", point_group_ranks)
        uint64_buffers = (
            _native_array("Q", training_aggregate_n),
            _native_array("Q", training_k),
            _native_array("Q", heldout_aggregate_n),
            _native_array("Q", heldout_k),
        )
        uint32_buffers = (
            _native_array("I", training_launch_k_tiles),
            _native_array("I", heldout_launch_k_tiles),
        )
        if point_rank_buffer.itemsize != 4:
            raise RuntimeError("Python array('I') is not a native uint32 buffer")
        if any(buffer.itemsize != 8 for buffer in uint64_buffers):
            raise RuntimeError("Python array('Q') is not a native uint64 buffer")
        if any(buffer.itemsize != 4 for buffer in uint32_buffers):
            raise RuntimeError("Python array('I') is not a native uint32 buffer")
        if any(value <= 0 or value > (1 << 64) - 1 for value in heldout_aggregate_n):
            raise ValueError("heldout aggregate N must fit positive uint64")
        if any(value <= 0 or value > (1 << 64) - 1 for value in heldout_k):
            raise ValueError("heldout K must fit positive uint64")
        if any(
            isinstance(value, bool) or not 0 <= value <= UINT32_MAX
            for value in heldout_launch_k_tiles
        ):
            raise ValueError("heldout K-tile geometry must fit uint32")

        self._prepare_session(
            training_double_buffers[0],
            training_double_buffers[1],
            point_count=point_count,
            candidate_count=candidate_count,
        )
        evaluation_type = _TreeFoldEvaluationC * max_leaves
        raw_evaluations = evaluation_type()
        exact_type = ctypes.c_uint32 * heldout_point_count
        raw_exact = exact_type()
        error = ctypes.create_string_buffer(ERROR_CAPACITY)
        result = self._library.llaminarNativeVNNITreeSearchAndEvaluate(
            self._session,
            _as_pointer(training_double_buffers[2], ctypes.c_double),
            _as_pointer(training_double_buffers[3], ctypes.c_double),
            _as_pointer(point_rank_buffer, ctypes.c_uint32),
            _as_pointer(uint64_buffers[0], ctypes.c_uint64),
            _as_pointer(uint64_buffers[1], ctypes.c_uint64),
            _as_pointer(uint32_buffers[0], ctypes.c_uint32),
            axis_buffer,
            _as_pointer(uint64_buffers[2], ctypes.c_uint64),
            _as_pointer(uint64_buffers[3], ctypes.c_uint64),
            _as_pointer(uint32_buffers[1], ctypes.c_uint32),
            _as_pointer(heldout_double_buffers[0], ctypes.c_double),
            _as_pointer(heldout_double_buffers[1], ctypes.c_double),
            _as_pointer(heldout_double_buffers[2], ctypes.c_double),
            heldout_point_count,
            axis_count,
            int(boundary_placement),
            parallelism_width,
            task_multiplier,
            min_shape_groups_per_leaf,
            max_leaves,
            raw_evaluations,
            raw_exact,
            error,
            len(error),
        )
        if result != 0:
            detail = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{self.backend}:{self.device_ordinal} fused tree evaluation "
                f"failed with ABI status {result}: {detail}"
            )

        exact = tuple(int(value) for value in raw_exact)
        if any(candidate >= candidate_count for candidate in exact):
            raise RuntimeError("fused tree evaluation returned an invalid exact winner")
        evaluated = []
        for budget_index, raw in enumerate(raw_evaluations, start=1):
            if raw.required_point_count != heldout_point_count:
                raise RuntimeError(
                    f"{self.backend}:{self.device_ordinal} fused evaluation "
                    f"budget={budget_index} returned required_point_count="
                    f"{raw.required_point_count}, expected={heldout_point_count}; "
                    f"training_points={point_count}, candidates={candidate_count}"
                )
            selected = tuple(
                int(raw.selected_candidate_indices[point])
                for point in range(heldout_point_count)
            )
            if any(
                candidate != UINT32_MAX and candidate >= candidate_count
                for candidate in selected
            ):
                raise RuntimeError(
                    "fused evaluation returned an out-of-range selected candidate"
                )
            covered = sum(candidate != UINT32_MAX for candidate in selected)
            if raw.covered_point_count != covered:
                raise RuntimeError(
                    f"{self.backend}:{self.device_ordinal} fused evaluation "
                    f"budget={budget_index} returned covered_point_count="
                    f"{raw.covered_point_count}, reconstructed={covered}; "
                    f"heldout_points={heldout_point_count}, "
                    f"training_points={point_count}, candidates={candidate_count}"
                )
            evaluated.append(AcceleratedFoldEvaluation(
                selected_candidate_indices=selected,
                exact_candidate_indices=exact,
                covered_point_count=covered,
                required_point_count=heldout_point_count,
            ))
        return tuple(evaluated)
