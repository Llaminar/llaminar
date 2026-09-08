"""Isolated Linux perf, Nsight Compute, and rocprofiler collectors.

The collector consumes :mod:`profiler_evidence` requests only after canonical
timing has finished. GPU and CPU requests amortize process/runtime setup across
strict batches. Every CPU member still owns a separately reset/armed/disarmed
``perf_event_open`` interval. Every CUDA member owns a separate profiler range
and explicit stream whose stable ID partitions its physical Nsight dispatches.
Each request therefore retains exactly one production operation and
independently attributed counters even though setup and the raw tool report are
shared.
Raw tool reports are retained and hashed before normalized evidence is emitted.

The command builders and parsers are intentionally ordinary Python with no
backend runtime dependency.  Unit tests can therefore prove launch isolation,
metric normalization, and failure behavior without touching a GPU; actual
counter collection remains an integration/performance operation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .profiler_evidence import (
    EXPECTED_PROFILER_TOOL,
    PROFILER_COLLECTOR_VERSION,
    PROFILER_METRIC_SET_VERSION,
    MetricAvailability,
    MetricDefinition,
    ProfiledDispatch,
    ProfiledDispatchKind,
    ProfilerEvidence,
    ProfilerEvidenceManifest,
    ProfilerEvidenceStatus,
    ProfilerMetric,
    ProfilerRequest,
    ProfilerRequestManifest,
    metric_definitions,
    read_profiler_evidence_manifest,
    read_profiler_request_manifest,
    validate_profiler_evidence_coverage,
    write_profiler_evidence_manifest,
    _offline_worker_count,
    _physical_core_count,
)
from .schema import Backend, ExecutionMode


DEFAULT_BINARIES = {
    Backend.CPU: Path(
        "build_v2_release/tests/v2/v2_perf_cpu_native_vnni_gemv"
    ),
    Backend.CUDA: Path(
        "build_v2_release/tests/v2/v2_perf_cuda_native_vnni_decode_trainer"
    ),
    Backend.ROCM: Path(
        "build_v2_release/tests/v2/v2_perf_native_vnni_throughput"
    ),
}

DEFAULT_CPU_AVX512_BINARY = Path(
    "build_v2_release_avx512/tests/v2/v2_perf_cpu_native_vnni_gemv"
)

DEFAULT_TOOL_CANDIDATES = {
    Backend.CPU: (Path("/usr/bin/perf"),),
    Backend.CUDA: (
        Path("/usr/local/cuda/bin/ncu"),
        Path("/opt/nvidia/nsight-compute/ncu"),
    ),
    Backend.ROCM: (
        Path("/opt/rocm/bin/rocprofv3"),
        Path("/opt/rocm/bin/rocprof"),
    ),
}

# ROCm 7.1 selected-region interception retains process-global HIP graph
# submission state. A 512-member mixed eager/graph counter process repeatedly
# failed on selected request 304. The captured core placed the fault inside
# librocprofiler-sdk while it dereferenced the first byte beyond a profiler-
# owned /dev/zero AQL mapping during hipGraphLaunch. The identical GPUBusy plan
# truncated to 256 requests completed. A fresh process resets that state.
# Keep the independent graph limit because a resumed corpus may contain graph
# requests only, even when the original transaction was mode-balanced.
ROCM_MAX_GPU_PROCESS_BATCH_SIZE = 256
ROCM_MAX_GRAPH_REQUESTS_PER_PROCESS = 256

PERF_EVENT_TO_METRIC = {
    "cycles": "cpu.cycles",
    "ref-cycles": "cpu.ref_cycles",
    "instructions": "cpu.instructions",
    "task-clock": "cpu.task_clock_ns",
    "wall-clock": "cpu.wall_clock_ns",
    "branches": "cpu.branches",
    "branch-misses": "cpu.branch_misses",
    "cache-references": "cpu.cache_references",
    "cache-misses": "cpu.cache_misses",
    "l1-dcache-loads": "cpu.l1d_loads",
    "l1-dcache-load-misses": "cpu.l1d_load_misses",
    "llc-loads": "cpu.llc_loads",
    "llc-load-misses": "cpu.llc_load_misses",
}

# Linux perf's hardware-cache aliases are case-sensitive even though its CSV
# output is normalized by the parser below. Keep the portable core events in
# their canonical lowercase spelling and the cache aliases exactly as published
# by `perf list cache`.
PERF_EVENTS = (
    "cycles",
    "ref-cycles",
    "instructions",
    "task-clock",
    "branches",
    "branch-misses",
    "cache-references",
    "cache-misses",
    "L1-dcache-loads",
    "L1-dcache-load-misses",
    "LLC-loads",
    "LLC-load-misses",
)

ROCM_COUNTER_GROUPS = (
    ("GPUBusy", "VALUBusy", "SALUBusy"),
    ("MemUnitBusy", "MemUnitStalled", "L2CacheHit"),
    # gfx906 cannot schedule FetchSize and WriteSize in one hardware profile;
    # rocprofiler-sdk reports capability error 38 and can hang while handling
    # the resulting abort. Keep traffic and remaining SQ-derived metrics as
    # reviewed singleton passes instead of paying that failure on every cell.
    ("FetchSize",),
    ("WriteSize",),
    ("Wavefronts",),
    ("VALUUtilization",),
    ("LDSBankConflict",),
    # These instruction-per-work-item counters share one schedulable gfx906
    # pass. Unlike volatile busy percentages, they directly expose excess ALU
    # and flat-memory work in an otherwise identical candidate contest.
    ("VALUInsts", "FlatVMemInsts"),
)

# Request only the reviewed feature schema. Section-based Nsight collection
# exports hundreds of unrelated metrics and needs roughly 17 replay passes on
# Ampere. These raw names cover the same required occupancy, resource, speed-
# of-light, memory, IPC, stall, and spilling signals in a bounded profile.
NCU_METRICS = (
    "gpu__time_duration.sum",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "launch__local_mem_per_thread",
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed",
)

CUDA_PRODUCTION_KERNEL_FILTER = "regex:nativeVnni"


_PARALLEL_JOURNAL_REQUESTS: Mapping[str, ProfilerRequest] = {}


@dataclass(frozen=True)
class CollectorOptions:
    """Resolved process/tool paths and backend-specific launch placement."""

    backend: Backend
    binary: Path
    tool: Path
    raw_directory: Path
    cpu_list: str | None = None
    timeout_seconds: int = 300
    cpu_avx2_binary: Path | None = None
    cpu_avx512_binary: Path | None = None
    tool_command_prefix: tuple[str, ...] = ()
    device_ordinal: int | None = None


def _parse_cpu_set(cpu_list: str) -> set[int]:
    """Expand one Linux CPU-list expression into its exact logical CPUs."""

    selected_cpus: set[int] = set()
    for token in cpu_list.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            begin_text, end_text = token.split("-", 1)
            begin = int(begin_text)
            end = int(end_text)
            if begin < 0 or end < begin:
                raise ValueError(f"invalid CPU range {token!r}")
            selected_cpus.update(range(begin, end + 1))
        else:
            cpu = int(token)
            if cpu < 0:
                raise ValueError(f"invalid CPU ordinal {token!r}")
            selected_cpus.add(cpu)
    if not selected_cpus:
        raise ValueError("CPU list must select at least one logical CPU")
    return selected_cpus


def _parse_cpu_lane_lists(cpu_list: str | None, lane_count: int) -> tuple[str, ...]:
    """Validate disjoint semicolon-delimited CPU masks for profiler lanes."""

    if cpu_list is None:
        raise ValueError("CPU profiling requires --cpu-list")
    lanes = tuple(item.strip() for item in cpu_list.split(";") if item.strip())
    if len(lanes) != lane_count:
        raise ValueError(
            f"CPU profiling requested {lane_count} lanes but --cpu-list defines "
            f"{len(lanes)} semicolon-delimited masks"
        )
    occupied: set[int] = set()
    for lane, lane_list in enumerate(lanes):
        selected = _parse_cpu_set(lane_list)
        overlap = occupied.intersection(selected)
        if overlap:
            raise ValueError(
                f"CPU profiler lane {lane} overlaps an earlier lane on CPUs "
                f"{sorted(overlap)}"
            )
        occupied.update(selected)
    return lanes


def _binary_for_request(
    request: ProfilerRequest, options: CollectorOptions
) -> Path:
    """Select the CPU build that produced a request's canonical observation."""

    if request.backend != Backend.CPU:
        return options.binary
    match = re.search(r"(?:^|:)build=(AVX2|AVX512)(?::|$)", request.threading_or_stream_mode)
    if not match:
        raise ValueError(
            f"{request.request_id}: CPU request does not identify its build ISA"
        )
    build_isa = match.group(1)
    selected = (
        options.cpu_avx2_binary
        if build_isa == "AVX2"
        else options.cpu_avx512_binary
    )
    if selected is None:
        raise ValueError(
            f"{request.request_id}: collector has no {build_isa} trainer binary"
        )
    return selected


def _sha256_json(value: Any) -> str:
    """Hash one JSON-compatible command/provenance object."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_files(root: Path) -> str:
    """Hash every retained raw artifact by relative path and exact bytes."""

    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"profiler produced no raw artifacts in {root}")
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _sha256_file(path: Path) -> str:
    """Return the exact content identity of one profiler trainer binary."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_binary_provenance(
    path: Path,
    request: ProfilerRequest,
    binary: Path,
    binary_digest: str | None = None,
) -> None:
    """Retain the profiling executable identity beside the raw counters.

    The request's ``build_id`` belongs to canonical timing. A later isolated
    profiler launch may intentionally use a rebuilt harness when the physical
    candidate fingerprint is unchanged, for example after installing a new
    generated dispatch table. Recording both identities makes that distinction
    explicit while the enclosing ``raw_artifact_digest`` authenticates this
    record with every profiler output file.
    """

    path.write_text(
        json.dumps(
            {
                "candidate_policy_hash": request.candidate_policy_hash,
                "profiler_binary": str(binary.resolve()),
                "profiler_binary_digest": binary_digest or _sha256_file(binary),
                "timing_build_id": request.build_id,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _tool_command(
    options: CollectorOptions,
    *arguments: str,
) -> list[str]:
    """Build one authenticated profiler command with optional privilege."""

    return [*options.tool_command_prefix, str(options.tool), *arguments]


def _tool_version(options: CollectorOptions) -> str:
    """Return one stable first-line tool version without invoking a workload."""

    completed = subprocess.run(
        _tool_command(options, "--version"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    text = (completed.stdout or "").strip()
    if completed.returncode != 0 or not text:
        raise RuntimeError(
            f"unable to query profiler version from {options.tool}: {text}"
        )
    return text.splitlines()[0].strip()


def _resolve_tool(backend: Backend, explicit: Path | None) -> Path | None:
    """Find an executable profiler without quietly substituting another tool."""

    if explicit is not None:
        return explicit if explicit.is_file() and os.access(explicit, os.X_OK) else None
    for candidate in DEFAULT_TOOL_CANDIDATES[backend]:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    executable = {
        Backend.CPU: "perf",
        Backend.CUDA: "ncu",
        Backend.ROCM: "rocprofv3",
    }[backend]
    resolved = shutil.which(executable)
    return None if resolved is None else Path(resolved)


def _rocm_profile_variant(request: ProfilerRequest) -> str:
    """Return the ROCm trainer spelling for one registry candidate."""

    config = request.config_json
    if "kb" in config:
        return f"KB{int(config['kb'])}"
    if config.get("split_policy") == "inherit_serial_m1":
        return "INHERIT_SERIAL_M1"
    raise ValueError(
        f"{request.request_id}: unsupported ROCm profiler candidate config"
    )


def _profile_environment(request: ProfilerRequest, raw_dir: Path) -> dict[str, str]:
    """Build exact one-cell trainer filters from one authenticated request."""

    environment = os.environ.copy()
    environment["LLAMINAR_NATIVE_VNNI_PROFILE_REQUEST_ID"] = request.request_id
    environment["LLAMINAR_PERF_STATS_JSON"] = "1"
    # A stale shell setting must not turn an ordinary full-row request into a
    # different physical launch. Counted occupancy belongs to the request.
    environment.pop("LLAMINAR_CUDA_NVNNI_DECODE_ACTIVE_ROWS", None)
    if request.active_rows is not None:
        if request.backend != Backend.CUDA:
            raise ValueError("this backend profiler trainer has no device-counted row surface")
        environment["LLAMINAR_CUDA_NVNNI_DECODE_ACTIVE_ROWS"] = str(request.active_rows)
    if request.backend == Backend.CUDA:
        environment.update({
            "LLAMINAR_CUDA_NVNNI_DECODE_FORMATS": request.source_format,
            "LLAMINAR_CUDA_NVNNI_DECODE_SHAPES": request.shape_name,
            "LLAMINAR_CUDA_NVNNI_DECODE_CANDIDATES": request.effective_candidate_id,
            "LLAMINAR_CUDA_NVNNI_DECODE_M": str(request.m),
            "LLAMINAR_CUDA_NVNNI_DECODE_EXECUTION_MODES": request.execution_mode.value,
            "LLAMINAR_CUDA_NVNNI_DECODE_MAX_CASES": "1",
            "LLAMINAR_CUDA_NVNNI_DECODE_WARMUPS": "1",
            "LLAMINAR_CUDA_NVNNI_DECODE_SAMPLES": "1",
            "LLAMINAR_CUDA_NVNNI_DECODE_TIMED_REPLAYS": "1",
            "LLAMINAR_CUDA_NVNNI_DECODE_CSV": str(raw_dir / "trainer.csv"),
            "LLAMINAR_CUDA_NVNNI_DECODE_TIMING_CSV": str(
                raw_dir / "trainer.timing.csv"
            ),
        })
    elif request.backend == Backend.ROCM:
        variant = _rocm_profile_variant(request)
        environment.update({
            "LLAMINAR_ROCM_NVNNI_DECODE_FORMATS": request.source_format,
            "LLAMINAR_ROCM_NVNNI_DECODE_SHAPES": request.shape_name,
            "LLAMINAR_ROCM_NVNNI_DECODE_VARIANTS": variant,
            "LLAMINAR_ROCM_NVNNI_DECODE_M": str(request.m),
            "LLAMINAR_ROCM_NVNNI_DECODE_EXECUTION_MODES": request.execution_mode.value,
            "LLAMINAR_ROCM_NVNNI_DECODE_MAX_CASES": "1",
            "LLAMINAR_ROCM_NVNNI_DECODE_WARMUPS": "1",
            "LLAMINAR_ROCM_NVNNI_DECODE_SAMPLES": "1",
            "LLAMINAR_ROCM_NVNNI_DECODE_CSV": str(raw_dir / "trainer.csv"),
            "LLAMINAR_ROCM_NVNNI_DECODE_TIMING_CSV": str(
                raw_dir / "trainer.timing.csv"
            ),
        })
    elif request.backend == Backend.CPU:
        match = re.search(
            r"requested=(AUTO|AVX2|AVX512):effective=(AVX2|AVX512):threads=(\d+)",
            request.threading_or_stream_mode,
        )
        if not match:
            raise ValueError(
                f"{request.request_id}: CPU threading/ISA provenance is not parseable"
            )
        requested_isa, _, threads = match.groups()
        try:
            prefix = _cpu_profiler_environment_prefix(request.operation_kind)
        except ValueError as exc:
            raise ValueError(f"{request.request_id}: {exc}") from exc
        environment.update({
            f"{prefix}_FORMATS": request.source_format,
            f"{prefix}_SHAPE_NAME": request.shape_name,
            f"{prefix}_N": str(request.aggregate_n),
            f"{prefix}_K": str(request.k),
            f"{prefix}_M": str(request.m),
            f"{prefix}_CANDIDATES": request.effective_candidate_id,
            f"{prefix}_MAX_CASES": "1",
            f"{prefix}_WARMUP": "1",
            f"{prefix}_ITERS": "1",
            f"{prefix}_THREADS": threads,
            f"{prefix}_STRONG_CSV": str(raw_dir / "trainer.csv"),
            f"{prefix}_TIMING_CSV": str(raw_dir / "trainer.timing.csv"),
            "OMP_NUM_THREADS": threads,
            "OMP_PLACES": "cores",
            "OMP_PROC_BIND": "close",
        })
        if requested_isa == "AUTO":
            environment.pop("LLAMINAR_ISA_LEVEL", None)
        else:
            environment["LLAMINAR_ISA_LEVEL"] = requested_isa
    else:
        raise ValueError(f"unsupported profiler backend {request.backend}")
    return environment


@dataclass(frozen=True)
class CPUProfilerProcessBatch:
    """Exact CPU requests allowed to share fixture and OpenMP-team setup."""

    requests: tuple[ProfilerRequest, ...]

    def __post_init__(self) -> None:
        if not self.requests:
            raise ValueError("CPU profiler process batch must not be empty")


@dataclass(frozen=True)
class GPUProfilerProcessBatch:
    """Exact same-backend GPU requests sharing one profiler process."""

    requests: tuple[ProfilerRequest, ...]

    def __post_init__(self) -> None:
        if not self.requests:
            raise ValueError("GPU profiler process batch must not be empty")
        backends = {request.backend for request in self.requests}
        if len(backends) != 1 or not backends.issubset({Backend.CUDA, Backend.ROCM}):
            raise ValueError(
                "GPU profiler batch must contain one CUDA or ROCm backend"
            )


def _gpu_process_batch_key(request: ProfilerRequest) -> tuple[object, ...]:
    """Return immutable context fields that one GPU process may share.

    Source format remains part of the key. Adjacent shape/M cells therefore
    reuse one CUDA context and Nsight injection session without forcing the
    trainer to keep several independently packed codebooks resident at once.
    Candidate, shape, mode, and work geometry deliberately remain outside this
    key and are selected exactly by the authenticated TSV plan.
    """

    if request.backend not in {Backend.CUDA, Backend.ROCM}:
        raise ValueError("GPU profiler batch requires CUDA or ROCm requests")
    return (
        request.backend,
        request.architecture_class,
        request.build_id,
        request.compiler_id,
        request.device_name,
        request.driver_runtime,
        request.threading_or_stream_mode,
        request.semantic_contract,
        request.operation_kind,
        request.bundle_signature,
        request.source_format,
        request.source_codebook_id,
        request.prepared_family_id,
        request.packing_abi,
        request.runtime_codebook_id,
        request.active_rows,
    )


def _gpu_cell_key(request: ProfilerRequest) -> tuple[object, ...]:
    """Return fields that must remain together when chunking GPU batches."""

    return (
        request.shape_group_id,
        request.shape_name,
        request.execution_mode,
        request.m,
        request.projection_n_vector,
        request.aggregate_n,
        request.k,
    )


def _build_gpu_process_batches(
    requests: Sequence[ProfilerRequest],
    maximum_size: int,
    maximum_graph_captured: int | None = None,
) -> tuple[GPUProfilerProcessBatch, ...]:
    """Pack complete same-backend GPU work cells into bounded batches.

    A cell contains every measured candidate for one exact physical geometry.
    Keeping it intact lets the trainer prepare serial/correctness state once
    and profile all candidates while that state is resident. Only a cell that
    individually exceeds the configured process bound is split, preserving
    totality without creating an unbounded report.
    """

    if maximum_size <= 0:
        raise ValueError("GPU profiler batch size must be positive")
    if maximum_graph_captured is not None and maximum_graph_captured <= 0:
        raise ValueError("GPU graph-captured profiler batch size must be positive")
    grouped: dict[tuple[object, ...], list[ProfilerRequest]] = {}
    for request in requests:
        grouped.setdefault(_gpu_process_batch_key(request), []).append(request)

    batches: list[GPUProfilerProcessBatch] = []
    for process_key in sorted(grouped, key=repr):
        cells: dict[tuple[object, ...], list[ProfilerRequest]] = {}
        for request in grouped[process_key]:
            cells.setdefault(_gpu_cell_key(request), []).append(request)
        current: list[ProfilerRequest] = []
        for cell_key in sorted(cells, key=repr):
            members = sorted(
                cells[cell_key],
                key=lambda item: (
                    item.effective_candidate_id,
                    item.request_id,
                ),
            )
            graph_members = sum(
                request.execution_mode == ExecutionMode.GRAPH_CAPTURED
                for request in members
            )
            current_graph_members = sum(
                request.execution_mode == ExecutionMode.GRAPH_CAPTURED
                for request in current
            )
            exceeds_graph_limit = (
                maximum_graph_captured is not None
                and current_graph_members + graph_members
                > maximum_graph_captured
            )
            if current and (
                len(current) + len(members) > maximum_size
                or exceeds_graph_limit
            ):
                batches.append(GPUProfilerProcessBatch(tuple(current)))
                current = []
            cell_limit = maximum_size
            if (
                maximum_graph_captured is not None
                and graph_members == len(members)
            ):
                cell_limit = min(cell_limit, maximum_graph_captured)
            while len(members) > cell_limit:
                batches.append(GPUProfilerProcessBatch(
                    tuple(members[:cell_limit])
                ))
                members = members[cell_limit:]
            current.extend(members)
        if current:
            batches.append(GPUProfilerProcessBatch(tuple(current)))
    return tuple(batches)


def _cpu_profiler_environment_prefix(operation_kind: str) -> str:
    """Return the trainer environment namespace for one CPU operation."""

    if operation_kind == "NativeVNNIFastM1Projection":
        return "LLAMINAR_CPU_NVNNI_DECODE"
    if operation_kind == "NativeVNNIDecodeProjection":
        return "LLAMINAR_CPU_NVNNI_VERIFIER"
    if operation_kind == "NativeVNNIPrefillProjection":
        return "LLAMINAR_CPU_NVNNI_PREFILL"
    raise ValueError(f"unsupported CPU profiler operation {operation_kind!r}")


def _cpu_process_batch_key(request: ProfilerRequest) -> tuple[object, ...]:
    """Return launch context that must remain fixed for one CPU process."""

    if request.backend != Backend.CPU:
        raise ValueError("CPU profiler batch cannot contain another backend")
    return (
        request.architecture_class,
        request.build_id,
        request.compiler_id,
        request.device_name,
        request.driver_runtime,
        request.threading_or_stream_mode,
        request.semantic_contract,
        request.operation_kind,
        request.bundle_signature,
        request.execution_mode,
        request.aggregate_n,
        request.k,
    )


def _build_cpu_process_batches(
    requests: Sequence[ProfilerRequest],
    maximum_size: int,
) -> tuple[CPUProfilerProcessBatch, ...]:
    """Partition exact requests without creating Cartesian launch aliases."""

    if maximum_size <= 0:
        raise ValueError("CPU profiler batch size must be positive")
    grouped: dict[tuple[object, ...], list[ProfilerRequest]] = {}
    for request in requests:
        grouped.setdefault(_cpu_process_batch_key(request), []).append(request)
    batches: list[CPUProfilerProcessBatch] = []
    for key in sorted(grouped, key=repr):
        cells: dict[tuple[object, ...], list[ProfilerRequest]] = {}
        for request in grouped[key]:
            cell = (
                request.source_format,
                request.m,
                request.projection_n_vector,
            )
            cells.setdefault(cell, []).append(request)
        current: list[ProfilerRequest] = []
        for cell in sorted(cells):
            members = sorted(
                cells[cell],
                key=lambda item: (
                    item.effective_candidate_id,
                    item.request_id,
                ),
            )
            if current and len(current) + len(members) > maximum_size:
                batches.append(CPUProfilerProcessBatch(tuple(current)))
                current = []
            while len(members) > maximum_size:
                batches.append(CPUProfilerProcessBatch(
                    tuple(members[:maximum_size])
                ))
                members = members[maximum_size:]
            current.extend(members)
        if current:
            batches.append(CPUProfilerProcessBatch(tuple(current)))
    return tuple(batches)


def _write_cpu_batch_plan(
    path: Path,
    batch: CPUProfilerProcessBatch,
    raw_directories: Mapping[str, Path],
) -> str:
    """Write the trainer's exact TSV plan and return its content digest."""

    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow((
        "request_id",
        "operation_kind",
        "source_format",
        "execution_mode",
        "m",
        "projection_n_vector",
        "aggregate_n",
        "k",
        "effective_candidate_id",
        "output_path",
    ))
    for request in batch.requests:
        fields = (
            request.request_id,
            request.operation_kind,
            request.source_format,
            request.execution_mode.value,
            str(request.m),
            ",".join(str(value) for value in request.projection_n_vector),
            str(request.aggregate_n),
            str(request.k),
            request.effective_candidate_id,
            str((raw_directories[request.request_id] / "perf-stat.csv").resolve()),
        )
        if any("\t" in field or "\n" in field or "\r" in field for field in fields):
            raise ValueError("CPU profiler batch identity contains a TSV delimiter")
        writer.writerow(fields)
    payload = output.getvalue()
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cpu_batch_environment(
    batch: CPUProfilerProcessBatch,
    plan_path: Path,
    plan_digest: str,
    batch_raw_dir: Path,
) -> dict[str, str]:
    """Build one process environment without weakening exact launch ownership."""

    first = batch.requests[0]
    environment = _profile_environment(first, batch_raw_dir)
    environment.pop("LLAMINAR_NATIVE_VNNI_PROFILE_REQUEST_ID", None)
    environment.pop("LLAMINAR_NATIVE_VNNI_PERF_STATS_PATH", None)
    environment["LLAMINAR_NATIVE_VNNI_PROFILE_BATCH_PATH"] = str(
        plan_path.resolve()
    )
    environment["LLAMINAR_NATIVE_VNNI_PROFILE_BATCH_DIGEST"] = plan_digest
    prefix = _cpu_profiler_environment_prefix(first.operation_kind)
    environment[f"{prefix}_FORMATS"] = ",".join(sorted({
        request.source_format for request in batch.requests
    }))
    environment[f"{prefix}_M"] = ",".join(str(value) for value in sorted({
        request.m for request in batch.requests
    }))
    environment[f"{prefix}_CANDIDATES"] = ",".join(sorted({
        request.effective_candidate_id for request in batch.requests
    }))
    environment[f"{prefix}_MAX_CASES"] = str(len({
        (request.source_format, request.m) for request in batch.requests
    }))
    environment[f"{prefix}_STRONG_CSV"] = str(batch_raw_dir / "trainer.csv")
    environment[f"{prefix}_TIMING_CSV"] = str(
        batch_raw_dir / "trainer.timing.csv"
    )
    return environment


def _write_gpu_batch_plan(
    path: Path,
    batch: GPUProfilerProcessBatch,
    raw_directories: Mapping[str, Path],
) -> str:
    """Write one exact GPU profiler plan and return its content digest."""

    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow((
        "request_id",
        "operation_kind",
        "source_format",
        "execution_mode",
        "shape_name",
        "m",
        "projection_n_vector",
        "aggregate_n",
        "k",
        "effective_candidate_id",
        "output_path",
    ))
    for request in batch.requests:
        fields = (
            request.request_id,
            request.operation_kind,
            request.source_format,
            request.execution_mode.value,
            request.shape_name,
            str(request.m),
            ",".join(str(value) for value in request.projection_n_vector),
            str(request.aggregate_n),
            str(request.k),
            request.effective_candidate_id,
            str(raw_directories[request.request_id].resolve()),
        )
        if any("\t" in field or "\n" in field or "\r" in field for field in fields):
            raise ValueError("GPU profiler batch identity contains a TSV delimiter")
        writer.writerow(fields)
    payload = output.getvalue()
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cuda_batch_environment(
    batch: GPUProfilerProcessBatch,
    plan_path: Path,
    plan_digest: str,
    batch_raw_dir: Path,
) -> dict[str, str]:
    """Build one exact-plan CUDA environment without Cartesian aliases."""

    first = batch.requests[0]
    if any(request.active_rows != first.active_rows for request in batch.requests):
        raise ValueError("one CUDA profiler process must retain one active-row occupancy")
    environment = _profile_environment(first, batch_raw_dir)
    environment.pop("LLAMINAR_NATIVE_VNNI_PROFILE_REQUEST_ID", None)
    environment["LLAMINAR_NATIVE_VNNI_PROFILE_BATCH_PATH"] = str(
        plan_path.resolve()
    )
    environment["LLAMINAR_NATIVE_VNNI_PROFILE_BATCH_DIGEST"] = plan_digest
    environment["LLAMINAR_CUDA_NVNNI_DECODE_FORMATS"] = ",".join(sorted({
        request.source_format for request in batch.requests
    }))
    environment["LLAMINAR_CUDA_NVNNI_DECODE_SHAPES"] = ",".join(sorted({
        request.shape_name for request in batch.requests
    }))
    environment["LLAMINAR_CUDA_NVNNI_DECODE_CANDIDATES"] = ",".join(sorted({
        request.effective_candidate_id for request in batch.requests
    }))
    environment["LLAMINAR_CUDA_NVNNI_DECODE_M"] = ",".join(str(value) for value in sorted({
        request.m for request in batch.requests
    }))
    environment["LLAMINAR_CUDA_NVNNI_DECODE_EXECUTION_MODES"] = ",".join(sorted({
        request.execution_mode.value for request in batch.requests
    }))
    environment["LLAMINAR_CUDA_NVNNI_DECODE_MAX_CASES"] = str(len({
        (request.source_format, request.shape_name, request.m)
        for request in batch.requests
    }))
    environment["LLAMINAR_CUDA_NVNNI_DECODE_CSV"] = str(
        batch_raw_dir / "trainer.csv"
    )
    environment["LLAMINAR_CUDA_NVNNI_DECODE_TIMING_CSV"] = str(
        batch_raw_dir / "trainer.timing.csv"
    )
    return environment


def _rocm_batch_environment(
    batch: GPUProfilerProcessBatch,
    plan_path: Path,
    plan_digest: str,
    batch_raw_dir: Path,
) -> dict[str, str]:
    """Build one exact-plan ROCm environment without Cartesian aliases."""

    first = batch.requests[0]
    if first.backend != Backend.ROCM:
        raise ValueError("ROCm batch environment requires ROCm requests")
    environment = _profile_environment(first, batch_raw_dir)
    environment.pop("LLAMINAR_NATIVE_VNNI_PROFILE_REQUEST_ID", None)
    environment["LLAMINAR_NATIVE_VNNI_PROFILE_BATCH_PATH"] = str(
        plan_path.resolve()
    )
    environment["LLAMINAR_NATIVE_VNNI_PROFILE_BATCH_DIGEST"] = plan_digest
    environment["LLAMINAR_ROCM_NVNNI_DECODE_FORMATS"] = ",".join(sorted({
        request.source_format for request in batch.requests
    }))
    environment["LLAMINAR_ROCM_NVNNI_DECODE_SHAPES"] = ",".join(sorted({
        request.shape_name for request in batch.requests
    }))
    environment["LLAMINAR_ROCM_NVNNI_DECODE_VARIANTS"] = ",".join(sorted({
        _rocm_profile_variant(request) for request in batch.requests
    }))
    environment["LLAMINAR_ROCM_NVNNI_DECODE_M"] = ",".join(str(value) for value in sorted({
        request.m for request in batch.requests
    }))
    environment["LLAMINAR_ROCM_NVNNI_DECODE_EXECUTION_MODES"] = ",".join(sorted({
        request.execution_mode.value for request in batch.requests
    }))
    environment["LLAMINAR_ROCM_NVNNI_DECODE_MAX_CASES"] = str(len({
        (request.source_format, request.shape_name, request.m)
        for request in batch.requests
    }))
    environment["LLAMINAR_ROCM_NVNNI_DECODE_CSV"] = str(
        batch_raw_dir / "trainer.csv"
    )
    environment["LLAMINAR_ROCM_NVNNI_DECODE_TIMING_CSV"] = str(
        batch_raw_dir / "trainer.timing.csv"
    )
    return environment


def _apply_device_placement(
    environment: dict[str, str],
    options: CollectorOptions,
) -> None:
    """Bind one isolated profiler worker to its assigned physical GPU lane."""

    if options.device_ordinal is None:
        return
    if options.backend == Backend.CUDA:
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = str(options.device_ordinal)
    elif options.backend == Backend.ROCM:
        environment["ROCR_VISIBLE_DEVICES"] = str(options.device_ordinal)


def _trainer_arguments(request: ProfilerRequest) -> list[str]:
    """Return the only gtest route allowed in an isolated profile process."""

    if request.backend == Backend.CPU:
        if request.operation_kind == "NativeVNNIFastM1Projection":
            return ["--gtest_filter=*TrainerCsv_StrongDecode_AllFormats"]
        if request.operation_kind == "NativeVNNIDecodeProjection":
            return ["--gtest_filter=*TrainerCsv_StrongVerifierRows_AllFormats"]
        if request.operation_kind == "NativeVNNIPrefillProjection":
            return ["--gtest_filter=*TrainerCsv_StrongPrefill_AllFormats"]
        raise ValueError(
            f"unsupported CPU profiler operation {request.operation_kind!r}"
        )
    if request.backend == Backend.CUDA:
        return ["--gtest_filter=*TrainerCsv_StrongDecode_AllFormats"]
    if request.backend == Backend.ROCM:
        return ["--gtest_filter=*TrainerCsv_CodebookTagged"]
    raise ValueError(f"unsupported profiler backend {request.backend}")


def _command_digest(
    request: ProfilerRequest,
    command: Sequence[str],
    environment: Mapping[str, str],
) -> str:
    """Bind evidence to argv and every environment discriminator we inject."""

    controlled_environment = {
        name: value
        for name, value in environment.items()
        if name.startswith("LLAMINAR_") or name.startswith("OMP_")
    }
    return _sha256_json({
        "request_id": request.request_id,
        "command": list(command),
        "environment": controlled_environment,
    })


def _parse_number(raw: str) -> float:
    """Parse profiler numbers with grouping separators, percent, and blanks."""

    value = raw.strip().replace(",", "").replace("%", "")
    if not value or value.startswith("<not") or value.lower() in {"n/a", "nan"}:
        raise ValueError(f"profiler value is unavailable: {raw!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"profiler value is non-finite: {raw!r}")
    return result


def _metric(
    definition: MetricDefinition,
    value: float | None,
    source_name: str,
    reason: str | None = None,
) -> ProfilerMetric:
    """Construct one measured or explicitly unsupported canonical metric."""

    return ProfilerMetric(
        metric_id=definition.metric_id,
        category=definition.category,
        unit=definition.unit,
        availability=(
            MetricAvailability.MEASURED
            if value is not None
            else MetricAvailability.UNSUPPORTED_BY_TOOL
        ),
        value=value,
        source_name=source_name,
        reason=None if value is not None else (reason or "metric unavailable"),
    )


def _complete_metric_inventory(
    backend: Backend,
    values: Mapping[str, tuple[float, str]],
    *,
    unavailable_reason: str,
) -> tuple[ProfilerMetric, ...]:
    """Fill every canonical metric explicitly and reject no field by omission."""

    return tuple(
        _metric(
            definition,
            values.get(definition.metric_id, (None, definition.metric_id))[0],
            values.get(definition.metric_id, (None, definition.metric_id))[1],
            unavailable_reason,
        )
        for definition in metric_definitions(backend)
    )


def _kernel_fingerprint(
    kernel_name: str,
    grid: tuple[int, int, int] | None,
    block: tuple[int, int, int] | None,
) -> str:
    """Hash a physical dispatch identity independently of profiler row order."""

    return _sha256_json({"kernel": kernel_name, "grid": grid, "block": block})


def parse_perf_stat(path: Path, request: ProfilerRequest) -> tuple[ProfiledDispatch, ...]:
    """Normalize one semicolon-separated controlled Linux perf report."""

    values: dict[str, tuple[float, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter=";"):
            if len(row) < 3 or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            event = row[2].strip().lower().split(":", 1)[0]
            metric_id = PERF_EVENT_TO_METRIC.get(event)
            if metric_id is None:
                continue
            try:
                value = _parse_number(row[0])
            except ValueError:
                continue
            unit = row[1].strip().lower()
            if metric_id == "cpu.task_clock_ns":
                if unit in {"msec", "ms"}:
                    value *= 1.0e6
                elif unit in {"usec", "us"}:
                    value *= 1.0e3
                elif unit not in {"nsec", "ns"}:
                    raise ValueError(f"unrecognized perf task-clock unit {unit!r}")
            values[metric_id] = (value, event)

    metrics = _complete_metric_inventory(
        Backend.CPU,
        values,
        unavailable_reason="Linux perf did not publish this optional event",
    )
    dispatch = ProfiledDispatch(
        dispatch_index=0,
        dispatch_kind=ProfiledDispatchKind.CPU_PARALLEL_REGION,
        kernel_name=request.effective_candidate_id,
        kernel_fingerprint=_kernel_fingerprint(
            request.effective_candidate_id, None, None
        ),
        grid=None,
        block=None,
        metrics=metrics,
    )
    dispatch.validate(Backend.CPU)
    return (dispatch,)


def _find_ncu_header(lines: Sequence[str]) -> int:
    """Locate the Nsight CSV header after diagnostics/banner text."""

    for index, line in enumerate(lines):
        if "Kernel Name" in line and (
            "Metric Name" in line or "gpu__time_duration" in line
        ):
            return index
    raise ValueError("Nsight Compute CSV does not contain a metric header")


def _parse_geometry(raw: str) -> tuple[int, int, int]:
    """Normalize Nsight scalar or parenthesized launch dimensions."""

    numbers = [int(value) for value in re.findall(r"\d+", raw)]
    if not numbers:
        raise ValueError(f"missing launch geometry in {raw!r}")
    numbers.extend([1] * (3 - len(numbers)))
    return tuple(numbers[:3])  # type: ignore[return-value]


NCU_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "gpu.duration_ns": ("gpu__time_duration.sum", "Duration"),
    "gpu.registers_per_thread": (
        "launch__registers_per_thread",
        "Registers Per Thread",
    ),
    "gpu.static_shared_memory_bytes": (
        "launch__shared_mem_per_block_static",
        "Static Shared Memory Per Block",
    ),
    "gpu.dynamic_shared_memory_bytes": (
        "launch__shared_mem_per_block_dynamic",
        "Dynamic Shared Memory Per Block",
    ),
    "gpu.local_memory_bytes_per_thread": (
        "launch__local_mem_per_thread",
        "Local Memory Per Thread",
    ),
    "gpu.theoretical_occupancy_pct": (
        "sm__maximum_warps_per_active_cycle_pct",
        "Theoretical Occupancy",
    ),
    "gpu.achieved_occupancy_pct": (
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "Achieved Occupancy",
    ),
    "gpu.compute_throughput_pct_of_peak": (
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "Compute (SM) Throughput",
    ),
    "gpu.dram_throughput_pct_of_peak": (
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "DRAM Throughput",
    ),
    "gpu.l1_throughput_pct_of_peak": (
        "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__throughput.avg.pct_of_peak_sustained_active",
        "L1/TEX Cache Throughput",
    ),
    "gpu.l2_throughput_pct_of_peak": (
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "L2 Cache Throughput",
    ),
    "gpu.executed_ipc_active": (
        "smsp__inst_executed.avg.per_cycle_active",
        "Executed IPC Active",
    ),
    "gpu.warp_cycles_per_issued_instruction": (
        "smsp__average_warp_latency_per_inst_issued.ratio",
        "Warp Cycles Per Issued Instruction",
    ),
    "gpu.local_memory_spill_requests": (
        "derived__local_spilling_requests",
        "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum",
        "Local Memory Spilling Requests",
    ),
    "gpu.alu_pipe_utilization_pct": (
        "smsp__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed",
    ),
    "gpu.fma_pipe_utilization_pct": (
        "smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
    ),
    "gpu.tensor_pipe_utilization_pct": (
        "smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed",
    ),
}

NCU_COMPUTE_PIPE_METRIC_IDS = (
    "gpu.alu_pipe_utilization_pct",
    "gpu.fma_pipe_utilization_pct",
    "gpu.tensor_pipe_utilization_pct",
)


def _convert_ncu_value(metric_id: str, value: float, unit: str) -> float:
    """Convert Nsight SI display units without confusing Kbyte with KiB.

    Nsight uses decimal prefixes even for memory resources. Preserve explicit
    IEC spellings for imported diagnostic CSVs, and strip per-block/per-thread
    suffixes before interpreting the unit. Missing metrics remain missing; a
    unit conversion cannot manufacture a zero-spill certificate.
    """

    normalized = unit.strip().lower()
    if metric_id == "gpu.duration_ns":
        if normalized in {"us", "usecond", "useconds"}:
            return value * 1.0e3
        if normalized in {"ms", "msecond", "mseconds"}:
            return value * 1.0e6
        if normalized in {"s", "second", "seconds"}:
            return value * 1.0e9
        return value
    if metric_id.endswith("_bytes") or "memory_bytes" in metric_id:
        base = normalized.split("/", 1)[0].strip()
        scales = {
            "kbyte": 1.0e3, "kbytes": 1.0e3, "kb": 1.0e3,
            "mbyte": 1.0e6, "mbytes": 1.0e6, "mb": 1.0e6,
            "gbyte": 1.0e9, "gbytes": 1.0e9, "gb": 1.0e9,
            "kib": 1024.0, "kibyte": 1024.0, "kibytes": 1024.0,
            "mib": 1024.0 ** 2, "mibyte": 1024.0 ** 2,
            "gib": 1024.0 ** 3, "gibyte": 1024.0 ** 3,
        }
        return value * scales.get(base, 1.0)
    return value


@dataclass(frozen=True)
class _NCUDispatchRecord:
    """One parsed Nsight dispatch plus its stable CUDA stream identity."""

    stream_id: int
    dispatch: ProfiledDispatch


def _parse_ncu_csv_with_stream_ids(
    path: Path,
    request: ProfilerRequest,
) -> tuple[_NCUDispatchRecord, ...]:
    """Normalize raw Nsight CSV while retaining request ownership metadata."""

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = _find_ncu_header(lines)
    parsed_rows = list(csv.reader(lines[header:]))
    if not parsed_rows:
        raise ValueError("Nsight Compute CSV contains an empty metric table")
    columns = parsed_rows[0]
    wide_format = "Metric Name" not in columns
    units_by_column: dict[str, str] = {}
    if wide_format:
        if len(parsed_rows) < 2:
            raise ValueError("wide Nsight Compute CSV omits its unit row")
        units_by_column = dict(zip(columns, parsed_rows[1], strict=False))
        data_rows = parsed_rows[2:]
    else:
        data_rows = parsed_rows[1:]
    reader = (dict(zip(columns, values, strict=False)) for values in data_rows)
    grouped: dict[
        tuple[int, int, str, tuple[int, int, int], tuple[int, int, int]],
        dict[str, tuple[float, str]],
    ] = {}
    for fallback_index, row in enumerate(reader):
        kernel = (row.get("Kernel Name") or "").strip()
        if not kernel:
            continue
        identifier_raw = (row.get("ID") or str(fallback_index)).strip()
        try:
            identifier = int(identifier_raw)
        except ValueError:
            identifier = fallback_index
        stream_raw = (row.get("Stream") or "0").strip()
        try:
            stream_id = int(stream_raw)
        except ValueError as error:
            raise ValueError(
                f"invalid Nsight CUDA stream identity {stream_raw!r}"
            ) from error
        grid = _parse_geometry((row.get("Grid Size") or "1").strip())
        block = _parse_geometry((row.get("Block Size") or "1").strip())
        key = (identifier, stream_id, kernel, grid, block)
        metrics = grouped.setdefault(key, {})
        if wide_format:
            for metric_id, aliases in NCU_METRIC_ALIASES.items():
                for metric_name in aliases:
                    raw_value = row.get(metric_name)
                    if raw_value is None:
                        continue
                    try:
                        value = _parse_number(raw_value)
                    except ValueError:
                        continue
                    value = _convert_ncu_value(
                        metric_id,
                        value,
                        units_by_column.get(metric_name, ""),
                    )
                    metrics[metric_id] = (value, metric_name)
                    break
        else:
            metric_name = (row.get("Metric Name") or "").strip()
            if not metric_name:
                continue
            for metric_id, aliases in NCU_METRIC_ALIASES.items():
                if metric_name not in aliases:
                    continue
                value = _parse_number(row.get("Metric Value") or "")
                value = _convert_ncu_value(
                    metric_id, value, row.get("Metric Unit") or ""
                )
                metrics[metric_id] = (value, metric_name)
                break

    records = []
    for dispatch_index, ((_, stream_id, kernel, grid, block), values) in enumerate(
        sorted(grouped.items(), key=lambda item: item[0])
    ):
        pipe_utilizations = [
            values[metric_id][0]
            for metric_id in NCU_COMPUTE_PIPE_METRIC_IDS
            if metric_id in values
        ]
        if pipe_utilizations:
            values["gpu.compute_throughput_pct_of_peak"] = (
                max(pipe_utilizations),
                "derived:max_one_pass_compute_pipe_utilization",
            )
        metrics = _complete_metric_inventory(
            Backend.CUDA,
            values,
            unavailable_reason="Nsight Compute section did not expose this optional metric",
        )
        dispatch = ProfiledDispatch(
            dispatch_index=dispatch_index,
            dispatch_kind=ProfiledDispatchKind.GPU_KERNEL,
            kernel_name=kernel,
            kernel_fingerprint=_kernel_fingerprint(kernel, grid, block),
            grid=grid,
            block=block,
            metrics=metrics,
        )
        dispatch.validate(Backend.CUDA)
        records.append(_NCUDispatchRecord(stream_id, dispatch))
    if not records:
        raise ValueError("Nsight Compute reported no kernels in the target region")
    return tuple(records)


def parse_ncu_csv(
    path: Path,
    request: ProfilerRequest,
) -> tuple[ProfiledDispatch, ...]:
    """Normalize raw-page Nsight Compute CSV into ordered kernel dispatches."""

    return tuple(
        record.dispatch
        for record in _parse_ncu_csv_with_stream_ids(path, request)
    )


def _partition_cuda_batch_dispatches(
    dispatches: Sequence[_NCUDispatchRecord],
    request_streams: Sequence[tuple[str, int]],
) -> dict[str, tuple[ProfiledDispatch, ...]]:
    """Partition one Nsight report by its exact per-request CUDA streams.

    The trainer publishes CUDA's stable stream ID beside each exact request.
    Nsight exports that same ID for every selected production kernel, avoiding
    synthetic marker launches and any reconstruction from candidate families.
    Each member's physical operation dispatches are renumbered from zero so the
    ordinary single-request evidence invariant remains unchanged.
    """

    if not request_streams:
        raise ValueError("CUDA profiler batch reported no request streams")
    request_ids = [request_id for request_id, _ in request_streams]
    stream_ids = [stream_id for _, stream_id in request_streams]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("CUDA profiler batch request IDs contain duplicates")
    if len(set(stream_ids)) != len(stream_ids):
        raise ValueError("CUDA profiler batch stream IDs contain duplicates")

    segments: dict[int, list[ProfiledDispatch]] = {
        stream_id: [] for stream_id in stream_ids
    }
    for record in dispatches:
        if record.stream_id not in segments:
            raise ValueError(
                "Nsight reported a production dispatch on an unclaimed CUDA "
                f"stream: {record.stream_id}"
            )
        segments[record.stream_id].append(record.dispatch)

    result: dict[str, tuple[ProfiledDispatch, ...]] = {}
    for request_id, stream_id in request_streams:
        segment = sorted(
            segments[stream_id], key=lambda item: item.dispatch_index
        )
        if not segment:
            raise ValueError(
                f"CUDA profiler request {request_id} has no physical dispatch"
            )
        result[request_id] = tuple(
            replace(dispatch, dispatch_index=index)
            for index, dispatch in enumerate(segment)
        )
    return result


ROCM_METRIC_ALIASES = {
    "gpu.duration_ns": ("DurationNs", "Duration", "End_Timestamp-Start_Timestamp"),
    "gpu.vgpr_count": ("arch_vgpr", "VGPR_Count", "VGPR Count"),
    "gpu.sgpr_count": ("sgpr", "SGPR_Count", "SGPR Count"),
    "gpu.lds_bytes": (
        "lds",
        "Group_Segment_Size",
        "LDS_Block_Size",
        "LDS Block Size",
    ),
    "gpu.scratch_bytes": (
        "scr",
        "Private_Segment_Size",
        "Scratch_Size",
        "Scratch Size",
    ),
    "gpu.achieved_occupancy_pct": ("Occupancy", "AchievedOccupancy"),
    "gpu.gpu_busy_pct": ("GPUBusy",),
    "gpu.valu_busy_pct": ("VALUBusy",),
    "gpu.valu_utilization_pct": ("VALUUtilization",),
    "gpu.memory_unit_busy_pct": ("MemUnitBusy",),
    "gpu.memory_unit_stalled_pct": ("MemUnitStalled",),
    "gpu.l2_cache_hit_pct": ("L2CacheHit",),
    "gpu.fetch_kib": ("FetchSize", "FETCH_SIZE"),
    "gpu.write_kib": ("WriteSize", "WRITE_SIZE"),
    "gpu.wavefront_count": ("Wavefronts", "SQ_WAVES_sum"),
    "gpu.lds_bank_conflict_pct": ("LDSBankConflict",),
    "gpu.valu_instructions_per_workitem": ("VALUInsts",),
    "gpu.flat_vmem_instructions_per_workitem": ("FlatVMemInsts",),
}


def _read_rocm_csv(path: Path) -> list[dict[str, str]]:
    """Read one rocprofiler CSV only when it carries a kernel identity."""

    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            if not fields.intersection({"KernelName", "Kernel_Name", "Kernel Name"}):
                return []
            return [dict(row) for row in reader]
    except (csv.Error, OSError):
        return []


def _first(row: Mapping[str, str], names: Iterable[str]) -> str:
    """Return the first nonempty spelling used by one profiler version."""

    for name in names:
        value = (row.get(name) or "").strip().strip('"')
        if value:
            return value
    return ""


def parse_rocprof_csvs(
    paths: Iterable[Path], request: ProfilerRequest
) -> tuple[ProfiledDispatch, ...]:
    """Merge selected-region rocprofiler passes by target launch ordinal.

    Counter collection can rename kernels to a request-specific ROCTx range so
    `--kernel-include-regex` excludes setup launches. The independent trace pass
    deliberately preserves physical symbols. Joining by dispatch ID would fail
    because IDs restart in every process, and joining by name would fail after
    range renaming. Every pass contains the same single candidate pipeline, so
    its contiguous dispatch ordinal is the stable cross-process identity.
    """

    all_paths = tuple(sorted(Path(path) for path in paths))
    trace_paths = tuple(
        path
        for path in all_paths
        if "kernel_trace" in path.name and path.parent.name == "trace"
    )
    if not trace_paths:
        trace_paths = tuple(path for path in all_paths if "kernel_trace" in path.name)

    @dataclass
    class MutableDispatch:
        kernel: str
        grid: tuple[int, int, int]
        block: tuple[int, int, int]
        values: dict[str, tuple[float, str]]

    physical: list[MutableDispatch] = []
    if trace_paths:
        trace_rows = []
        for path in trace_paths[:1]:
            trace_rows.extend(_read_rocm_csv(path))
        trace_rows.sort(
            key=lambda row: int(_first(row, ("Dispatch_Id", "Index")) or 0)
        )
        for row in trace_rows:
            kernel = _first(row, ("KernelName", "Kernel_Name", "Kernel Name"))
            if not kernel:
                continue
            grid = (
                int(_first(row, ("Grid_Size_X", "Grid Size X")) or 1),
                int(_first(row, ("Grid_Size_Y", "Grid Size Y")) or 1),
                int(_first(row, ("Grid_Size_Z", "Grid Size Z")) or 1),
            )
            block = (
                int(_first(row, ("Workgroup_Size_X", "Workgroup Size X", "wgr")) or 1),
                int(_first(row, ("Workgroup_Size_Y", "Workgroup Size Y")) or 1),
                int(_first(row, ("Workgroup_Size_Z", "Workgroup Size Z")) or 1),
            )
            values: dict[str, tuple[float, str]] = {}
            start = _first(row, ("Start_Timestamp", "BeginNs"))
            end = _first(row, ("End_Timestamp", "EndNs"))
            if start and end:
                values["gpu.duration_ns"] = (
                    _parse_number(end) - _parse_number(start),
                    "End_Timestamp-Start_Timestamp",
                )
            for metric_id, aliases in ROCM_METRIC_ALIASES.items():
                for alias in aliases:
                    raw = (row.get(alias) or "").strip()
                    if raw:
                        values[metric_id] = (_parse_number(raw), alias)
                        break
            physical.append(MutableDispatch(kernel, grid, block, values))

    counter_paths = tuple(path for path in all_paths if "counter_collection" in path.name)
    expected_counter_kernel = f"NativeVNNIProfile::{request.request_id}"
    for path in counter_paths:
        rows = [
            row
            for row in _read_rocm_csv(path)
            if _first(row, ("KernelName", "Kernel_Name", "Kernel Name"))
            == expected_counter_kernel
        ]
        grouped_rows: dict[int, list[dict[str, str]]] = {}
        for row in rows:
            dispatch_id = int(_first(row, ("Dispatch_Id", "Index")) or 0)
            grouped_rows.setdefault(dispatch_id, []).append(row)
        ordered_groups = [grouped_rows[key] for key in sorted(grouped_rows)]
        if not physical:
            for rows_for_dispatch in ordered_groups:
                row = rows_for_dispatch[0]
                kernel = _first(row, ("KernelName", "Kernel_Name", "Kernel Name"))
                block_size = int(_first(row, ("Workgroup_Size", "wgr")) or 1)
                physical.append(MutableDispatch(
                    kernel=kernel,
                    grid=(int(_first(row, ("Grid_Size",)) or 1), 1, 1),
                    block=(block_size, 1, 1),
                    values={},
                ))
        if len(ordered_groups) != len(physical):
            raise ValueError(
                f"rocprofiler counter pass {path} reported {len(ordered_groups)} "
                f"dispatches; isolated trace reported {len(physical)}"
            )
        for ordinal, rows_for_dispatch in enumerate(ordered_groups):
            values = physical[ordinal].values
            first_row = rows_for_dispatch[0]
            start = _first(first_row, ("Start_Timestamp", "BeginNs"))
            end = _first(first_row, ("End_Timestamp", "EndNs"))
            if "gpu.duration_ns" not in values and start and end:
                values["gpu.duration_ns"] = (
                    _parse_number(end) - _parse_number(start),
                    "End_Timestamp-Start_Timestamp",
                )
            for row in rows_for_dispatch:
                counter_name = _first(row, ("Counter_Name", "Counter Name"))
                counter_value = _first(row, ("Counter_Value", "Counter Value"))
                if counter_name and counter_value:
                    for metric_id, aliases in ROCM_METRIC_ALIASES.items():
                        if counter_name in aliases:
                            values[metric_id] = (
                                _parse_number(counter_value),
                                counter_name,
                            )
                            break
                for metric_id, aliases in ROCM_METRIC_ALIASES.items():
                    for alias in aliases:
                        raw = (row.get(alias) or "").strip()
                        if raw:
                            # The uninstrumented selected-region trace is the
                            # authority for physical resource metadata and
                            # duration. Counter passes rename kernels and may
                            # report profiler-specific launch metadata; use it
                            # only to fill fields absent from the trace.
                            values.setdefault(
                                metric_id, (_parse_number(raw), alias)
                            )
                            break

    # The compact unit fixture and legacy rocprof CSV place trace metadata and
    # wide counters in one file without the production trace/ directory.
    if not physical:
        generic_rows = []
        for path in all_paths:
            generic_rows.extend(_read_rocm_csv(path))
        for row in generic_rows:
            kernel = _first(row, ("KernelName", "Kernel_Name", "Kernel Name"))
            if not kernel:
                continue
            block_size = int(_first(row, ("Workgroup_Size", "wgr")) or 1)
            values = {}
            for metric_id, aliases in ROCM_METRIC_ALIASES.items():
                for alias in aliases:
                    raw = (row.get(alias) or "").strip()
                    if raw:
                        values[metric_id] = (_parse_number(raw), alias)
                        break
            physical.append(MutableDispatch(
                kernel=kernel,
                grid=(1, 1, 1),
                block=(block_size, 1, 1),
                values=values,
            ))

    dispatches = []
    for dispatch_index, item in enumerate(physical):
        kernel = item.kernel
        grid = item.grid
        block = item.block
        values = item.values
        metrics = _complete_metric_inventory(
            Backend.ROCM,
            values,
            unavailable_reason="rocprofiler did not expose this optional metric",
        )
        dispatch = ProfiledDispatch(
            dispatch_index=dispatch_index,
            dispatch_kind=ProfiledDispatchKind.GPU_KERNEL,
            kernel_name=kernel,
            kernel_fingerprint=_kernel_fingerprint(kernel, grid, block),
            grid=grid,
            block=block,
            metrics=metrics,
        )
        dispatch.validate(Backend.ROCM)
        dispatches.append(dispatch)
    if not dispatches:
        raise ValueError("rocprofiler reported no kernels in the selected region")
    return tuple(dispatches)


def parse_rocprof_batch_csvs(
    paths: Iterable[Path],
    requests: Sequence[ProfilerRequest],
) -> dict[str, tuple[ProfiledDispatch, ...]]:
    """Partition one process-amortized rocprof report by exact request range.

    The uninstrumented trace preserves physical kernel names, while every
    counter pass renames kernels to ``NativeVNNIProfile::<request-id>``. Dispatch
    IDs restart between profiler processes, so the stable join is the ordered
    sequence of request ranges and the launch ordinal within each range. All
    files are parsed once regardless of batch size; no request can inherit a
    neighboring range's counters.
    """

    if not requests:
        raise ValueError("ROCm profiler batch parser requires requests")
    if any(request.backend != Backend.ROCM for request in requests):
        raise ValueError("ROCm profiler batch parser received another backend")
    request_by_kernel = {
        f"NativeVNNIProfile::{request.request_id}": request
        for request in requests
    }
    if len(request_by_kernel) != len(requests):
        raise ValueError("ROCm profiler batch repeats a request ID")

    @dataclass
    class MutableDispatch:
        kernel: str
        grid: tuple[int, int, int]
        block: tuple[int, int, int]
        values: dict[str, tuple[float, str]]

    all_paths = tuple(sorted(Path(path) for path in paths))
    trace_paths = tuple(
        path
        for path in all_paths
        if "kernel_trace" in path.name and path.parent.name == "trace"
    )
    if not trace_paths:
        raise ValueError("batched rocprofiler evidence has no physical trace")
    trace_rows = _read_rocm_csv(trace_paths[0])
    trace_rows.sort(
        key=lambda row: int(_first(row, ("Dispatch_Id", "Index")) or 0)
    )
    physical: list[MutableDispatch] = []
    for row in trace_rows:
        kernel = _first(row, ("KernelName", "Kernel_Name", "Kernel Name"))
        if not kernel:
            continue
        grid = (
            int(_first(row, ("Grid_Size_X", "Grid Size X")) or 1),
            int(_first(row, ("Grid_Size_Y", "Grid Size Y")) or 1),
            int(_first(row, ("Grid_Size_Z", "Grid Size Z")) or 1),
        )
        block = (
            int(_first(row, ("Workgroup_Size_X", "Workgroup Size X", "wgr")) or 1),
            int(_first(row, ("Workgroup_Size_Y", "Workgroup Size Y")) or 1),
            int(_first(row, ("Workgroup_Size_Z", "Workgroup Size Z")) or 1),
        )
        values: dict[str, tuple[float, str]] = {}
        start = _first(row, ("Start_Timestamp", "BeginNs"))
        end = _first(row, ("End_Timestamp", "EndNs"))
        if start and end:
            values["gpu.duration_ns"] = (
                _parse_number(end) - _parse_number(start),
                "End_Timestamp-Start_Timestamp",
            )
        for metric_id, aliases in ROCM_METRIC_ALIASES.items():
            for alias in aliases:
                raw = (row.get(alias) or "").strip()
                if raw:
                    values[metric_id] = (_parse_number(raw), alias)
                    break
        physical.append(MutableDispatch(kernel, grid, block, values))
    if not physical:
        raise ValueError("batched rocprofiler trace reported no kernels")

    counter_paths = tuple(
        path for path in all_paths if "counter_collection" in path.name
    )
    if not counter_paths:
        raise ValueError("batched rocprofiler evidence has no counter pass")
    spans: dict[str, tuple[int, int]] = {}
    for pass_index, path in enumerate(counter_paths):
        rows_by_request: dict[str, dict[int, list[dict[str, str]]]] = {}
        first_dispatch_by_request: dict[str, int] = {}
        request_by_dispatch: dict[int, str] = {}
        for row in _read_rocm_csv(path):
            kernel = _first(
                row, ("KernelName", "Kernel_Name", "Kernel Name")
            )
            request = request_by_kernel.get(kernel)
            if request is None:
                continue
            dispatch_id = int(_first(row, ("Dispatch_Id", "Index")) or 0)
            previous_request = request_by_dispatch.setdefault(
                dispatch_id, request.request_id
            )
            if previous_request != request.request_id:
                raise ValueError(
                    f"rocprofiler batch counter pass {path} attributed "
                    f"dispatch {dispatch_id} to multiple requests"
                )
            rows_by_request.setdefault(request.request_id, {}).setdefault(
                dispatch_id, []
            ).append(row)
            first_dispatch_by_request.setdefault(request.request_id, dispatch_id)
            first_dispatch_by_request[request.request_id] = min(
                first_dispatch_by_request[request.request_id], dispatch_id
            )
        expected_ids = {request.request_id for request in requests}
        if set(rows_by_request) != expected_ids:
            missing = sorted(expected_ids.difference(rows_by_request))
            extra = sorted(set(rows_by_request).difference(expected_ids))
            raise ValueError(
                f"rocprofiler batch counter pass {path} changed request "
                f"coverage: missing={missing} extra={extra}"
            )
        ordered_ids = sorted(
            expected_ids, key=lambda request_id: first_dispatch_by_request[request_id]
        )
        range_order: list[str] = []
        for dispatch_id in sorted(request_by_dispatch):
            request_id = request_by_dispatch[dispatch_id]
            if not range_order or range_order[-1] != request_id:
                range_order.append(request_id)
        if range_order != ordered_ids or len(range_order) != len(expected_ids):
            raise ValueError(
                f"rocprofiler batch counter pass {path} contains a "
                "non-contiguous or repeated request range"
            )
        if pass_index == 0:
            offset = 0
            for request_id in ordered_ids:
                count = len(rows_by_request[request_id])
                spans[request_id] = (offset, offset + count)
                offset += count
            if offset != len(physical):
                raise ValueError(
                    "rocprofiler batch counter/trace launch totals differ: "
                    f"counter={offset} trace={len(physical)}"
                )
        elif ordered_ids != sorted(
            spans, key=lambda request_id: spans[request_id][0]
        ):
            raise ValueError("rocprofiler batch changed request order between passes")

        for request_id in ordered_ids:
            begin, end = spans[request_id]
            ordered_groups = [
                rows_by_request[request_id][dispatch_id]
                for dispatch_id in sorted(rows_by_request[request_id])
            ]
            if len(ordered_groups) != end - begin:
                raise ValueError(
                    f"rocprofiler batch pass {path} changed dispatch count for "
                    f"{request_id}"
                )
            for ordinal, rows_for_dispatch in enumerate(ordered_groups):
                values = physical[begin + ordinal].values
                for row in rows_for_dispatch:
                    counter_name = _first(row, ("Counter_Name", "Counter Name"))
                    counter_value = _first(row, ("Counter_Value", "Counter Value"))
                    if counter_name and counter_value:
                        for metric_id, aliases in ROCM_METRIC_ALIASES.items():
                            if counter_name in aliases:
                                values[metric_id] = (
                                    _parse_number(counter_value), counter_name
                                )
                                break
                    for metric_id, aliases in ROCM_METRIC_ALIASES.items():
                        for alias in aliases:
                            raw = (row.get(alias) or "").strip()
                            if raw:
                                values.setdefault(
                                    metric_id, (_parse_number(raw), alias)
                                )
                                break

    result: dict[str, tuple[ProfiledDispatch, ...]] = {}
    for request in requests:
        begin, end = spans[request.request_id]
        dispatches = []
        for dispatch_index, item in enumerate(physical[begin:end]):
            metrics = _complete_metric_inventory(
                Backend.ROCM,
                item.values,
                unavailable_reason=(
                    "rocprofiler did not expose this optional metric"
                ),
            )
            dispatch = ProfiledDispatch(
                dispatch_index=dispatch_index,
                dispatch_kind=ProfiledDispatchKind.GPU_KERNEL,
                kernel_name=item.kernel,
                kernel_fingerprint=_kernel_fingerprint(
                    item.kernel, item.grid, item.block
                ),
                grid=item.grid,
                block=item.block,
                metrics=metrics,
            )
            dispatch.validate(Backend.ROCM)
            dispatches.append(dispatch)
        if not dispatches:
            raise ValueError(
                f"rocprofiler batch request {request.request_id} has no kernels"
            )
        result[request.request_id] = tuple(dispatches)
    return result


def _failed_evidence(
    request: ProfilerRequest,
    status: ProfilerEvidenceStatus,
    reason: str,
    tool_name: str,
) -> ProfilerEvidence:
    """Create an explicit non-success record without partial metric payloads."""

    return ProfilerEvidence(
        request_id=request.request_id,
        observation_digest=request.observation_digest,
        backend=request.backend,
        status=status,
        status_reason=reason,
        profiler_tool=tool_name,
        profiler_tool_version="",
        metric_set_version=PROFILER_METRIC_SET_VERSION,
        collector_version=PROFILER_COLLECTOR_VERSION,
        command_digest="",
        raw_artifact_digest="",
        profiler_pass_count=0,
        target_launches_per_profiler_pass=1,
        dispatches=(),
    )


def _unsupported_evidence(request: ProfilerRequest) -> ProfilerEvidence:
    """Record an unreachable timing-matrix cell without invoking a profiler."""

    return _failed_evidence(
        request,
        ProfilerEvidenceStatus.CANDIDATE_UNSUPPORTED,
        "canonical timing observation proved this candidate route unreachable",
        EXPECTED_PROFILER_TOOL[request.backend],
    )


def _write_process_log(
    path: Path, command: Sequence[str], completed: subprocess.CompletedProcess[str]
) -> None:
    """Retain argv, exit code, stdout, and stderr for failed or successful runs."""

    path.write_text(
        json.dumps({"command": list(command), "returncode": completed.returncode}, sort_keys=True)
        + "\n\n[stdout]\n"
        + (completed.stdout or "")
        + "\n[stderr]\n"
        + (completed.stderr or ""),
        encoding="utf-8",
    )


def _run_command(
    command: Sequence[str],
    environment: Mapping[str, str],
    log_path: Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    """Execute one profiler pass and retain complete process diagnostics."""

    completed = subprocess.run(
        list(command),
        env=dict(environment),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_seconds,
    )
    _write_process_log(log_path, command, completed)
    return completed


def _collect_cpu(
    request: ProfilerRequest,
    options: CollectorOptions,
    raw_dir: Path,
) -> ProfilerEvidence:
    """Collect one process-scoped ``perf_event_open`` candidate region.

    The trainer owns disabled inherited counters before it creates the OpenMP
    worker pool. It resets/enables them immediately around the one requested
    production launch and writes the raw semicolon-separated counts itself.
    Running under an external system-wide ``perf stat`` process would re-open
    the FIFO polling tail that this collector is specifically designed to
    exclude from per-invocation evidence.
    """

    if not options.cpu_list:
        raise ValueError("CPU profiling requires --cpu-list")
    selected_cpus = _parse_cpu_set(options.cpu_list)
    threads_match = re.search(
        r"(?:^|:)threads=(\d+)(?::|$)", request.threading_or_stream_mode
    )
    if not threads_match:
        raise ValueError("CPU profiler request does not identify its thread count")
    expected_threads = int(threads_match.group(1))
    if len(selected_cpus) != expected_threads:
        raise ValueError(
            f"CPU profiler selected {len(selected_cpus)} cores for a "
            f"{expected_threads}-thread timing observation"
        )
    environment = _profile_environment(request, raw_dir)
    environment["LLAMINAR_NATIVE_VNNI_PERF_STATS_PATH"] = str(
        raw_dir / "perf-stat.csv"
    )
    command = [
        "taskset",
        "-c",
        options.cpu_list,
        str(options.binary),
        *_trainer_arguments(request),
    ]
    completed = _run_command(
        command,
        environment,
        raw_dir / "perf-process.log",
        options.timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Linux perf-event trainer exited {completed.returncode}")
    dispatches = parse_perf_stat(raw_dir / "perf-stat.csv", request)
    return ProfilerEvidence(
        request_id=request.request_id,
        observation_digest=request.observation_digest,
        backend=request.backend,
        status=ProfilerEvidenceStatus.COMPLETE,
        status_reason=None,
        profiler_tool=EXPECTED_PROFILER_TOOL[request.backend],
        profiler_tool_version=_tool_version(options),
        metric_set_version=PROFILER_METRIC_SET_VERSION,
        collector_version=PROFILER_COLLECTOR_VERSION,
        command_digest=_command_digest(request, command, environment),
        raw_artifact_digest=_sha256_files(raw_dir),
        profiler_pass_count=1,
        target_launches_per_profiler_pass=1,
        dispatches=dispatches,
    )


def _next_attempt_directory(parent: Path) -> Path:
    """Create and return the next immutable raw-attempt directory."""

    parent.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (parent / f"attempt-{attempt:04d}").exists():
        attempt += 1
    result = parent / f"attempt-{attempt:04d}"
    result.mkdir()
    return result


def _collect_cpu_batch(
    batch: CPUProfilerProcessBatch,
    options: CollectorOptions,
) -> dict[str, ProfilerEvidence]:
    """Collect several exact CPU requests in one setup-amortized process."""

    if not options.cpu_list:
        raise ValueError("CPU profiling requires --cpu-list")
    selected_cpus = _parse_cpu_set(options.cpu_list)
    selected_binaries = {
        _binary_for_request(request, options) for request in batch.requests
    }
    if len(selected_binaries) != 1:
        raise ValueError("one CPU profiler batch selected multiple trainer binaries")
    selected_binary = next(iter(selected_binaries))
    if not selected_binary.is_file() or not os.access(selected_binary, os.X_OK):
        raise RuntimeError(f"trainer binary is unavailable: {selected_binary}")

    first = batch.requests[0]
    threads_match = re.search(
        r"(?:^|:)threads=(\d+)(?::|$)", first.threading_or_stream_mode
    )
    if not threads_match:
        raise ValueError("CPU profiler request does not identify its thread count")
    expected_threads = int(threads_match.group(1))
    if len(selected_cpus) != expected_threads:
        raise ValueError(
            f"CPU profiler selected {len(selected_cpus)} cores for a "
            f"{expected_threads}-thread timing observation"
        )

    batch_identity = hashlib.sha256("\n".join(
        request.request_id for request in batch.requests
    ).encode("utf-8")).hexdigest()[:24]
    batch_raw_dir = _next_attempt_directory(
        options.raw_directory / "_cpu_process_batches" / batch_identity
    )
    request_raw_dirs: dict[str, Path] = {}
    for request in batch.requests:
        raw_dir = _next_attempt_directory(
            options.raw_directory / request.request_id
        )
        request_raw_dirs[request.request_id] = raw_dir
        _write_binary_provenance(
            raw_dir / "profiler-binary.json",
            request,
            selected_binary,
        )

    plan_path = batch_raw_dir / "requests.tsv"
    plan_digest = _write_cpu_batch_plan(
        plan_path, batch, request_raw_dirs
    )
    environment = _cpu_batch_environment(
        batch, plan_path, plan_digest, batch_raw_dir
    )
    command = [
        "taskset",
        "-c",
        options.cpu_list,
        str(selected_binary),
        *_trainer_arguments(first),
    ]
    completed = _run_command(
        command,
        environment,
        batch_raw_dir / "perf-process.log",
        options.timeout_seconds,
    )
    process_lines = (completed.stdout + "\n" + completed.stderr).splitlines()
    tool_version = _tool_version(options)
    results: dict[str, ProfilerEvidence] = {}
    for request in batch.requests:
        matching_launches = [
            line for line in process_lines
            if f"request={request.request_id} " in line and "launches=1" in line
        ]
        if len(matching_launches) != 1:
            results[request.request_id] = _failed_evidence(
                request,
                ProfilerEvidenceStatus.LAUNCH_FAILED,
                (
                    f"batch trainer exited {completed.returncode} and reported "
                    f"{len(matching_launches)} exact target launches"
                ),
                EXPECTED_PROFILER_TOOL[request.backend],
            )
            continue
        raw_dir = request_raw_dirs[request.request_id]
        try:
            dispatches = parse_perf_stat(raw_dir / "perf-stat.csv", request)
            (raw_dir / "batch-member.json").write_text(
                json.dumps({
                    "batch_plan_digest": plan_digest,
                    "request_id": request.request_id,
                    "target_launches": 1,
                }, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            evidence = ProfilerEvidence(
                request_id=request.request_id,
                observation_digest=request.observation_digest,
                backend=request.backend,
                status=ProfilerEvidenceStatus.COMPLETE,
                status_reason=None,
                profiler_tool=EXPECTED_PROFILER_TOOL[request.backend],
                profiler_tool_version=tool_version,
                metric_set_version=PROFILER_METRIC_SET_VERSION,
                collector_version=PROFILER_COLLECTOR_VERSION,
                command_digest=_command_digest(request, command, environment),
                raw_artifact_digest=_sha256_files(raw_dir),
                profiler_pass_count=1,
                target_launches_per_profiler_pass=1,
                dispatches=dispatches,
            )
            evidence.validate(request)
            results[request.request_id] = evidence
        except (OSError, KeyError, TypeError, ValueError) as error:
            results[request.request_id] = _failed_evidence(
                request,
                ProfilerEvidenceStatus.PARSE_FAILED,
                str(error),
                EXPECTED_PROFILER_TOOL[request.backend],
            )
    return results


def collect_cpu_batch(
    batch: CPUProfilerProcessBatch,
    options: CollectorOptions,
) -> dict[str, ProfilerEvidence]:
    """Collect one CPU process batch and convert shared failures per request."""

    try:
        return _collect_cpu_batch(batch, options)
    except subprocess.TimeoutExpired as error:
        status = ProfilerEvidenceStatus.LAUNCH_FAILED
        reason = f"profiler transaction timed out after {error.timeout} seconds"
    except (OSError, RuntimeError) as error:
        status = ProfilerEvidenceStatus.LAUNCH_FAILED
        reason = str(error)
    except (KeyError, TypeError, ValueError) as error:
        status = ProfilerEvidenceStatus.PARSE_FAILED
        reason = str(error)
    return {
        request.request_id: _failed_evidence(
            request,
            status,
            reason,
            EXPECTED_PROFILER_TOOL[request.backend],
        )
        for request in batch.requests
    }


def _collect_cuda(
    request: ProfilerRequest,
    options: CollectorOptions,
    raw_dir: Path,
) -> ProfilerEvidence:
    """Collect one Nsight Compute report with app-controlled start/stop."""

    environment = _profile_environment(request, raw_dir)
    _apply_device_placement(environment, options)
    report = raw_dir / "profile"
    command = _tool_command(
        options,
        "--profile-from-start",
        "off",
        "--target-processes",
        "all",
        "--kernel-name",
        CUDA_PRODUCTION_KERNEL_FILTER,
        "--metrics",
        ",".join(NCU_METRICS),
        "-o",
        str(report),
        "-f",
        str(options.binary),
        *_trainer_arguments(request),
    )
    completed = _run_command(
        command, environment, raw_dir / "ncu-process.log", options.timeout_seconds
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Nsight Compute/trainer exited {completed.returncode}")
    report_path = report.with_suffix(".ncu-rep")
    export_command = _tool_command(
        options,
        "-i",
        str(report_path),
        "--page",
        "raw",
        "--csv",
        "--print-units",
        "base",
    )
    exported = _run_command(
        export_command,
        environment,
        raw_dir / "ncu-export.log",
        options.timeout_seconds,
    )
    if exported.returncode != 0:
        raise RuntimeError(f"Nsight Compute report export exited {exported.returncode}")
    csv_path = raw_dir / "ncu-raw.csv"
    csv_path.write_text(exported.stdout or "", encoding="utf-8")
    dispatches = parse_ncu_csv(csv_path, request)
    return ProfilerEvidence(
        request_id=request.request_id,
        observation_digest=request.observation_digest,
        backend=request.backend,
        status=ProfilerEvidenceStatus.COMPLETE,
        status_reason=None,
        profiler_tool=EXPECTED_PROFILER_TOOL[request.backend],
        profiler_tool_version=_tool_version(options),
        metric_set_version=PROFILER_METRIC_SET_VERSION,
        collector_version=PROFILER_COLLECTOR_VERSION,
        command_digest=_command_digest(request, command, environment),
        raw_artifact_digest=_sha256_files(raw_dir),
        profiler_pass_count=1,
        target_launches_per_profiler_pass=1,
        dispatches=dispatches,
    )


def _cuda_request_streams_from_process_log(
    completed: subprocess.CompletedProcess[str],
    batch: GPUProfilerProcessBatch,
) -> tuple[tuple[str, int], ...]:
    """Authenticate the trainer's exact request-to-CUDA-stream ownership."""

    pattern = re.compile(
        r"\[NativeVNNIProfiler\]\[CUDA\] request=([^\s]+) .* "
        r"stream_id=([0-9]+) launches=1$"
    )
    request_streams = tuple(
        (match.group(1), int(match.group(2)))
        for line in ((completed.stdout or "") + "\n" + (completed.stderr or "")).splitlines()
        if (match := pattern.search(line))
    )
    expected = {request.request_id for request in batch.requests}
    observed = {request_id for request_id, _ in request_streams}
    stream_ids = {stream_id for _, stream_id in request_streams}
    if (
        len(request_streams) != len(batch.requests)
        or observed != expected
        or len(stream_ids) != len(request_streams)
    ):
        raise RuntimeError(
            "CUDA batch trainer did not report one unique stream per exact "
            f"request: reported={len(request_streams)} "
            f"unique_streams={len(stream_ids)} expected={len(batch.requests)}"
        )
    return request_streams


def _collect_cuda_batch(
    batch: GPUProfilerProcessBatch,
    options: CollectorOptions,
) -> dict[str, ProfilerEvidence]:
    """Profile many exact CUDA operations in one setup-amortized process.

    Nsight still records every selected physical kernel independently. The
    trainer launches each request on a distinct explicit stream and reports
    its stable CUDA stream ID. This function partitions the raw report by those
    IDs before publishing member evidence, so one member never inherits
    counters from its neighbors.
    """

    selected_binaries = {
        _binary_for_request(request, options) for request in batch.requests
    }
    if len(selected_binaries) != 1:
        raise ValueError("one CUDA profiler batch selected multiple binaries")
    selected_binary = next(iter(selected_binaries))
    if not selected_binary.is_file() or not os.access(selected_binary, os.X_OK):
        raise RuntimeError(f"trainer binary is unavailable: {selected_binary}")

    batch_identity = hashlib.sha256("\n".join(
        request.request_id for request in batch.requests
    ).encode("utf-8")).hexdigest()[:24]
    batch_raw_dir = _next_attempt_directory(
        options.raw_directory / "_cuda_process_batches" / batch_identity
    )
    request_raw_dirs: dict[str, Path] = {}
    binary_digest = _sha256_file(selected_binary)
    for request in batch.requests:
        raw_dir = _next_attempt_directory(
            options.raw_directory / request.request_id
        )
        request_raw_dirs[request.request_id] = raw_dir
        _write_binary_provenance(
            raw_dir / "profiler-binary.json",
            request,
            selected_binary,
            binary_digest,
        )

    plan_path = batch_raw_dir / "requests.tsv"
    plan_digest = _write_gpu_batch_plan(
        plan_path, batch, request_raw_dirs
    )
    environment = _cuda_batch_environment(
        batch, plan_path, plan_digest, batch_raw_dir
    )
    _apply_device_placement(environment, options)
    report = batch_raw_dir / "profile"
    first = batch.requests[0]
    command = _tool_command(
        replace(options, binary=selected_binary),
        "--profile-from-start",
        "off",
        "--target-processes",
        "all",
        "--kernel-name",
        CUDA_PRODUCTION_KERNEL_FILTER,
        "--metrics",
        ",".join(NCU_METRICS),
        "-o",
        str(report),
        "-f",
        str(selected_binary),
        *_trainer_arguments(first),
    )
    completed = _run_command(
        command,
        environment,
        batch_raw_dir / "ncu-process.log",
        options.timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Nsight Compute/CUDA batch trainer exited {completed.returncode}"
        )
    request_streams = _cuda_request_streams_from_process_log(completed, batch)

    report_path = report.with_suffix(".ncu-rep")
    export_command = _tool_command(
        options,
        "-i",
        str(report_path),
        "--page",
        "raw",
        "--csv",
        "--print-units",
        "base",
    )
    exported = _run_command(
        export_command,
        environment,
        batch_raw_dir / "ncu-export.log",
        options.timeout_seconds,
    )
    if exported.returncode != 0:
        raise RuntimeError(
            f"Nsight Compute CUDA batch export exited {exported.returncode}"
        )
    csv_path = batch_raw_dir / "ncu-raw.csv"
    csv_path.write_text(exported.stdout or "", encoding="utf-8")
    all_dispatches = _parse_ncu_csv_with_stream_ids(csv_path, first)
    partitioned = _partition_cuda_batch_dispatches(
        all_dispatches, request_streams
    )
    batch_raw_digest = _sha256_files(batch_raw_dir)
    tool_version = _tool_version(options)

    results: dict[str, ProfilerEvidence] = {}
    order_index = {
        request_id: index
        for index, (request_id, _) in enumerate(request_streams)
    }
    stream_by_request = dict(request_streams)
    for request in batch.requests:
        raw_dir = request_raw_dirs[request.request_id]
        dispatches = partitioned[request.request_id]
        (raw_dir / "batch-member.json").write_text(
            json.dumps({
                "batch_plan_digest": plan_digest,
                "batch_raw_artifact_digest": batch_raw_digest,
                "physical_dispatch_count": len(dispatches),
                "request_id": request.request_id,
                "request_launch_order": order_index[request.request_id],
                "stream_id": stream_by_request[request.request_id],
                "target_launches": 1,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        evidence = ProfilerEvidence(
            request_id=request.request_id,
            observation_digest=request.observation_digest,
            backend=request.backend,
            status=ProfilerEvidenceStatus.COMPLETE,
            status_reason=None,
            profiler_tool=EXPECTED_PROFILER_TOOL[request.backend],
            profiler_tool_version=tool_version,
            metric_set_version=PROFILER_METRIC_SET_VERSION,
            collector_version=PROFILER_COLLECTOR_VERSION,
            command_digest=_command_digest(request, command, environment),
            raw_artifact_digest=_sha256_files(raw_dir),
            profiler_pass_count=1,
            target_launches_per_profiler_pass=1,
            dispatches=dispatches,
        )
        evidence.validate(request)
        results[request.request_id] = evidence
    return results


def collect_cuda_batch(
    batch: GPUProfilerProcessBatch,
    options: CollectorOptions,
) -> dict[str, ProfilerEvidence]:
    """Collect one CUDA batch and convert shared failures per request."""

    try:
        return _collect_cuda_batch(batch, options)
    except subprocess.TimeoutExpired as error:
        status = ProfilerEvidenceStatus.LAUNCH_FAILED
        reason = f"profiler transaction timed out after {error.timeout} seconds"
    except (OSError, RuntimeError) as error:
        status = ProfilerEvidenceStatus.LAUNCH_FAILED
        reason = str(error)
    except (KeyError, TypeError, ValueError) as error:
        status = ProfilerEvidenceStatus.PARSE_FAILED
        reason = str(error)
    return {
        request.request_id: _failed_evidence(
            request,
            status,
            reason,
            EXPECTED_PROFILER_TOOL[request.backend],
        )
        for request in batch.requests
    }


def _collect_rocm(
    request: ProfilerRequest,
    options: CollectorOptions,
    raw_dir: Path,
) -> ProfilerEvidence:
    """Collect selected-region rocprofiler trace and bounded counter groups."""

    if options.tool.name != "rocprofv3":
        raise ValueError(
            "isolated NativeVNNI profiling requires rocprofv3 selected regions"
        )
    environment = _profile_environment(request, raw_dir)
    _apply_device_placement(environment, options)
    commands = []
    trace_dir = raw_dir / "trace"
    trace_dir.mkdir()
    trace_command = _tool_command(
        options,
        "--selected-regions",
        "--kernel-trace",
        "--marker-trace",
        "--minimum-output-data",
        "0",
        "--output-file",
        "profile",
        "--output-format",
        "csv",
        "--output-directory",
        str(trace_dir),
        "--",
        str(options.binary),
        *_trainer_arguments(request),
    )
    commands.append(trace_command)
    completed = _run_command(
        trace_command,
        environment,
        raw_dir / "rocprof-trace.log",
        options.timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"rocprofiler trace/trainer exited {completed.returncode}")

    successful_counter_passes = 0

    def run_counter_pass(counters: Sequence[str], label: str) -> bool:
        """Run one counter-compatible process and retain failed attempts too."""

        nonlocal successful_counter_passes
        pass_dir = raw_dir / label
        pass_dir.mkdir()
        command = _tool_command(
            options,
            "--selected-regions",
            "--kernel-trace",
            "--marker-trace",
            "--kernel-rename",
            "--pmc",
            *counters,
            "--minimum-output-data",
            "0",
            "--output-file",
            "profile",
            "--output-format",
            "csv",
            "--output-directory",
            str(pass_dir),
            "--",
            str(options.binary),
            *_trainer_arguments(request),
        )
        commands.append(command)
        log_path = raw_dir / f"rocprof-{label}.log"
        try:
            completed = _run_command(
                command,
                environment,
                log_path,
                min(30, options.timeout_seconds),
            )
        except subprocess.TimeoutExpired as error:
            timeout_stdout = error.stdout or ""
            timeout_stderr = error.stderr or ""
            if isinstance(timeout_stdout, bytes):
                timeout_stdout = timeout_stdout.decode(errors="replace")
            if isinstance(timeout_stderr, bytes):
                timeout_stderr = timeout_stderr.decode(errors="replace")
            log_path.write_text(
                json.dumps({
                    "command": command,
                    "timeout_seconds": error.timeout,
                }, sort_keys=True)
                + "\n\n[stdout]\n"
                + timeout_stdout
                + "\n[stderr]\n"
                + timeout_stderr,
                encoding="utf-8",
            )
            return False
        if completed.returncode == 0:
            successful_counter_passes += 1
            return True
        return False

    for pass_index, counters in enumerate(ROCM_COUNTER_GROUPS):
        if run_counter_pass(counters, f"counters-{pass_index}"):
            continue
        # rocprofiler requires a counter set that fits one hardware pass. An
        # architecture may reject a reviewed group, so retry each metric in an
        # independent process rather than dropping the signal or contaminating
        # canonical timing with an in-process multiplexed fallback.
        for counter_index, counter in enumerate(counters):
            if not run_counter_pass(
                (counter,), f"counters-{pass_index}-single-{counter_index}"
            ):
                raise RuntimeError(
                    f"rocprofiler rejected required counter {counter}"
                )

    dispatches = parse_rocprof_csvs(raw_dir.rglob("*.csv"), request)
    return ProfilerEvidence(
        request_id=request.request_id,
        observation_digest=request.observation_digest,
        backend=request.backend,
        status=ProfilerEvidenceStatus.COMPLETE,
        status_reason=None,
        profiler_tool=EXPECTED_PROFILER_TOOL[request.backend],
        profiler_tool_version=_tool_version(options),
        metric_set_version=PROFILER_METRIC_SET_VERSION,
        collector_version=PROFILER_COLLECTOR_VERSION,
        command_digest=_sha256_json({
            "request_id": request.request_id,
            "commands": commands,
            "environment": {
                name: value
                for name, value in environment.items()
                if name.startswith("LLAMINAR_") or name.startswith("OMP_")
            },
        }),
        raw_artifact_digest=_sha256_files(raw_dir),
        profiler_pass_count=1 + successful_counter_passes,
        target_launches_per_profiler_pass=1,
        dispatches=dispatches,
    )


def _rocm_request_order_from_process_log(
    completed: subprocess.CompletedProcess[str],
    batch: GPUProfilerProcessBatch,
) -> tuple[str, ...]:
    """Authenticate one reported production launch for every batch member."""

    pattern = re.compile(
        r"\[NativeVNNIProfiler\]\[ROCm\] request=([^\s]+) .* launches=1$"
    )
    request_order = tuple(
        match.group(1)
        for line in (
            (completed.stdout or "") + "\n" + (completed.stderr or "")
        ).splitlines()
        if (match := pattern.search(line))
    )
    expected = {request.request_id for request in batch.requests}
    if len(request_order) != len(batch.requests) or set(request_order) != expected:
        raise RuntimeError(
            "ROCm batch trainer did not report exactly one production launch "
            f"per request: reported={len(request_order)} "
            f"unique={len(set(request_order))} expected={len(batch.requests)}"
        )
    return request_order


def _collect_rocm_batch(
    batch: GPUProfilerProcessBatch,
    options: CollectorOptions,
) -> dict[str, ProfilerEvidence]:
    """Collect one setup-amortized ROCm process batch.

    Every profiler pass executes the same authenticated TSV plan. The trainer
    opens one named selected region around each exact request's extra production
    launch, so rocprofiler process startup and weight preparation are amortized
    while counter ownership remains request-local.
    """

    if options.tool.name != "rocprofv3":
        raise ValueError(
            "batched NativeVNNI profiling requires rocprofv3 selected regions"
        )
    if any(request.backend != Backend.ROCM for request in batch.requests):
        raise ValueError("ROCm profiler batch contains another backend")
    selected_binaries = {
        _binary_for_request(request, options) for request in batch.requests
    }
    if len(selected_binaries) != 1:
        raise ValueError("one ROCm profiler batch selected multiple binaries")
    selected_binary = next(iter(selected_binaries))
    if not selected_binary.is_file() or not os.access(selected_binary, os.X_OK):
        raise RuntimeError(f"trainer binary is unavailable: {selected_binary}")

    batch_identity = hashlib.sha256("\n".join(
        request.request_id for request in batch.requests
    ).encode("utf-8")).hexdigest()[:24]
    batch_raw_dir = _next_attempt_directory(
        options.raw_directory / "_rocm_process_batches" / batch_identity
    )
    request_raw_dirs: dict[str, Path] = {}
    binary_digest = _sha256_file(selected_binary)
    for request in batch.requests:
        raw_dir = _next_attempt_directory(
            options.raw_directory / request.request_id
        )
        request_raw_dirs[request.request_id] = raw_dir
        _write_binary_provenance(
            raw_dir / "profiler-binary.json",
            request,
            selected_binary,
            binary_digest,
        )

    plan_path = batch_raw_dir / "requests.tsv"
    plan_digest = _write_gpu_batch_plan(
        plan_path, batch, request_raw_dirs
    )
    environment = _rocm_batch_environment(
        batch, plan_path, plan_digest, batch_raw_dir
    )
    _apply_device_placement(environment, options)
    batch_options = replace(options, binary=selected_binary)
    first = batch.requests[0]
    commands: list[list[str]] = []

    trace_dir = batch_raw_dir / "trace"
    trace_dir.mkdir()
    trace_command = _tool_command(
        batch_options,
        "--selected-regions",
        "--kernel-trace",
        "--marker-trace",
        "--minimum-output-data",
        "0",
        "--output-file",
        "profile",
        "--output-format",
        "csv",
        "--output-directory",
        str(trace_dir),
        "--",
        str(selected_binary),
        *_trainer_arguments(first),
    )
    commands.append(trace_command)
    trace_process = _run_command(
        trace_command,
        environment,
        batch_raw_dir / "rocprof-trace.log",
        options.timeout_seconds,
    )
    if trace_process.returncode != 0:
        raise RuntimeError(
            f"rocprofiler trace/ROCm batch trainer exited "
            f"{trace_process.returncode}"
        )
    request_order = _rocm_request_order_from_process_log(trace_process, batch)

    successful_counter_passes = 0
    successful_counter_directories: list[Path] = []

    def run_counter_pass(counters: Sequence[str], label: str) -> bool:
        """Run one compatible counter pass over the unchanged exact plan."""

        nonlocal successful_counter_passes
        pass_dir = batch_raw_dir / label
        pass_dir.mkdir()
        command = _tool_command(
            batch_options,
            "--selected-regions",
            "--kernel-trace",
            "--marker-trace",
            "--kernel-rename",
            "--pmc",
            *counters,
            "--minimum-output-data",
            "0",
            "--output-file",
            "profile",
            "--output-format",
            "csv",
            "--output-directory",
            str(pass_dir),
            "--",
            str(selected_binary),
            *_trainer_arguments(first),
        )
        commands.append(command)
        log_path = batch_raw_dir / f"rocprof-{label}.log"
        try:
            completed = _run_command(
                command,
                environment,
                log_path,
                options.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            timeout_stdout = error.stdout or ""
            timeout_stderr = error.stderr or ""
            if isinstance(timeout_stdout, bytes):
                timeout_stdout = timeout_stdout.decode(errors="replace")
            if isinstance(timeout_stderr, bytes):
                timeout_stderr = timeout_stderr.decode(errors="replace")
            log_path.write_text(
                json.dumps({
                    "command": command,
                    "timeout_seconds": error.timeout,
                }, sort_keys=True)
                + "\n\n[stdout]\n"
                + timeout_stdout
                + "\n[stderr]\n"
                + timeout_stderr,
                encoding="utf-8",
            )
            return False
        if completed.returncode != 0:
            return False
        counter_order = _rocm_request_order_from_process_log(completed, batch)
        if counter_order != request_order:
            raise RuntimeError(
                "ROCm batch trainer changed request launch order between "
                f"profiler passes: trace={request_order} counter={counter_order}"
            )
        successful_counter_passes += 1
        successful_counter_directories.append(pass_dir)
        return True

    for pass_index, counters in enumerate(ROCM_COUNTER_GROUPS):
        if run_counter_pass(counters, f"counters-{pass_index}"):
            continue
        for counter_index, counter in enumerate(counters):
            if not run_counter_pass(
                (counter,), f"counters-{pass_index}-single-{counter_index}"
            ):
                raise RuntimeError(
                    f"rocprofiler rejected required counter {counter}"
                )

    parse_paths = [*trace_dir.rglob("*.csv")]
    for counter_directory in successful_counter_directories:
        parse_paths.extend(counter_directory.rglob("*.csv"))
    partitioned = parse_rocprof_batch_csvs(parse_paths, batch.requests)
    batch_raw_digest = _sha256_files(batch_raw_dir)
    tool_version = _tool_version(options)
    order_index = {
        request_id: index for index, request_id in enumerate(request_order)
    }
    controlled_environment = {
        name: value
        for name, value in environment.items()
        if name.startswith("LLAMINAR_") or name.startswith("OMP_")
    }

    results: dict[str, ProfilerEvidence] = {}
    for request in batch.requests:
        raw_dir = request_raw_dirs[request.request_id]
        dispatches = partitioned[request.request_id]
        (raw_dir / "batch-member.json").write_text(
            json.dumps({
                "batch_plan_digest": plan_digest,
                "batch_raw_artifact_digest": batch_raw_digest,
                "physical_dispatch_count": len(dispatches),
                "request_id": request.request_id,
                "request_launch_order": order_index[request.request_id],
                "target_launches": 1,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        evidence = ProfilerEvidence(
            request_id=request.request_id,
            observation_digest=request.observation_digest,
            backend=request.backend,
            status=ProfilerEvidenceStatus.COMPLETE,
            status_reason=None,
            profiler_tool=EXPECTED_PROFILER_TOOL[request.backend],
            profiler_tool_version=tool_version,
            metric_set_version=PROFILER_METRIC_SET_VERSION,
            collector_version=PROFILER_COLLECTOR_VERSION,
            command_digest=_sha256_json({
                "request_id": request.request_id,
                "commands": commands,
                "environment": controlled_environment,
            }),
            raw_artifact_digest=_sha256_files(raw_dir),
            profiler_pass_count=1 + successful_counter_passes,
            target_launches_per_profiler_pass=1,
            dispatches=dispatches,
        )
        evidence.validate(request)
        results[request.request_id] = evidence
    return results


def collect_rocm_batch(
    batch: GPUProfilerProcessBatch,
    options: CollectorOptions,
) -> dict[str, ProfilerEvidence]:
    """Collect one ROCm batch and convert a shared failure per request."""

    try:
        return _collect_rocm_batch(batch, options)
    except subprocess.TimeoutExpired as error:
        status = ProfilerEvidenceStatus.LAUNCH_FAILED
        reason = f"profiler transaction timed out after {error.timeout} seconds"
    except (OSError, RuntimeError) as error:
        status = ProfilerEvidenceStatus.LAUNCH_FAILED
        reason = str(error)
    except (KeyError, TypeError, ValueError) as error:
        status = ProfilerEvidenceStatus.PARSE_FAILED
        reason = str(error)
    return {
        request.request_id: _failed_evidence(
            request,
            status,
            reason,
            EXPECTED_PROFILER_TOOL[request.backend],
        )
        for request in batch.requests
    }


def collect_request(
    request: ProfilerRequest,
    options: CollectorOptions,
) -> ProfilerEvidence:
    """Collect and validate one request, returning a typed failure on errors."""

    if request.backend != options.backend:
        raise ValueError("collector backend does not match profiler request")
    if not request.profile_required:
        return _unsupported_evidence(request)
    try:
        selected_binary = _binary_for_request(request, options)
    except ValueError as error:
        return _failed_evidence(
            request,
            ProfilerEvidenceStatus.LAUNCH_FAILED,
            str(error),
            EXPECTED_PROFILER_TOOL[request.backend],
        )
    if not selected_binary.is_file() or not os.access(selected_binary, os.X_OK):
        return _failed_evidence(
            request,
            ProfilerEvidenceStatus.LAUNCH_FAILED,
            f"trainer binary is unavailable: {selected_binary}",
            EXPECTED_PROFILER_TOOL[request.backend],
        )
    if not options.tool.is_file() or not os.access(options.tool, os.X_OK):
        return _failed_evidence(
            request,
            ProfilerEvidenceStatus.TOOL_UNAVAILABLE,
            f"profiler tool is unavailable: {options.tool}",
            EXPECTED_PROFILER_TOOL[request.backend],
        )

    request_directory = options.raw_directory / request.request_id
    request_directory.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (request_directory / f"attempt-{attempt:04d}").exists():
        attempt += 1
    raw_dir = request_directory / f"attempt-{attempt:04d}"
    raw_dir.mkdir()
    request_options = replace(options, binary=selected_binary)
    _write_binary_provenance(
        raw_dir / "profiler-binary.json",
        request,
        selected_binary,
    )
    try:
        if request.backend == Backend.CPU:
            result = _collect_cpu(request, request_options, raw_dir)
        elif request.backend == Backend.CUDA:
            result = _collect_cuda(request, request_options, raw_dir)
        elif request.backend == Backend.ROCM:
            result = _collect_rocm(request, request_options, raw_dir)
        else:
            raise ValueError(f"unsupported profiler backend {request.backend}")
        result.validate(request)
        return result
    except subprocess.TimeoutExpired as error:
        return _failed_evidence(
            request,
            ProfilerEvidenceStatus.LAUNCH_FAILED,
            f"profiler transaction timed out after {error.timeout} seconds",
            EXPECTED_PROFILER_TOOL[request.backend],
        )
    except (OSError, RuntimeError) as error:
        return _failed_evidence(
            request,
            ProfilerEvidenceStatus.LAUNCH_FAILED,
            str(error),
            EXPECTED_PROFILER_TOOL[request.backend],
        )
    except (KeyError, TypeError, ValueError) as error:
        return _failed_evidence(
            request,
            ProfilerEvidenceStatus.PARSE_FAILED,
            str(error),
            EXPECTED_PROFILER_TOOL[request.backend],
        )


def _truncate_torn_journal_tail(journal: Path) -> tuple[bytes, int, int]:
    """Return the complete JSONL extent after discarding one torn tail.

    A killed append can damage only the final line because every completed
    batch is flushed and fsynced. Scanning backward in bounded blocks avoids
    materializing a multi-gigabyte journal merely to find its last newline.
    """

    with journal.open("r+b") as handle:
        header = handle.readline()
        if not header.endswith(b"\n"):
            raise ValueError("profiler checkpoint journal has no complete header")
        content_begin = handle.tell()
        handle.seek(0, os.SEEK_END)
        content_end = handle.tell()
        if content_end == content_begin:
            return header, content_begin, content_end
        handle.seek(content_end - 1)
        if handle.read(1) == b"\n":
            return header, content_begin, content_end

        scan_end = content_end
        complete_end = content_begin
        while scan_end > content_begin:
            scan_begin = max(content_begin, scan_end - 1024 * 1024)
            handle.seek(scan_begin)
            block = handle.read(scan_end - scan_begin)
            newline = block.rfind(b"\n")
            if newline >= 0:
                complete_end = scan_begin + newline + 1
                break
            scan_end = scan_begin
        handle.truncate(complete_end)
        handle.flush()
        os.fsync(handle.fileno())
        return header, content_begin, complete_end


def _journal_byte_ranges(
    journal: Path,
    content_begin: int,
    content_end: int,
    worker_count: int,
) -> tuple[tuple[Path, int, int], ...]:
    """Partition complete JSONL records into deterministic byte ranges."""

    if worker_count < 1:
        raise ValueError("profiler journal worker count must be positive")
    if content_end <= content_begin:
        return ()
    boundaries = [content_begin]
    with journal.open("rb") as handle:
        for worker_index in range(1, worker_count):
            target = content_begin + (
                (content_end - content_begin) * worker_index // worker_count
            )
            handle.seek(target)
            handle.readline()
            boundary = handle.tell()
            if content_begin < boundary < content_end:
                boundaries.append(boundary)
    boundaries.append(content_end)
    boundaries = sorted(set(boundaries))
    return tuple(
        (journal, begin, end)
        for begin, end in zip(boundaries, boundaries[1:])
        if begin < end
    )


def _decode_validate_journal_range(
    task: tuple[Path, int, int],
) -> tuple[ProfilerEvidence, ...]:
    """Decode and validate one inherited range of journal evidence."""

    journal, begin, end = task
    with journal.open("rb") as handle:
        handle.seek(begin)
        encoded_records = handle.read(end - begin).splitlines()
    evidence_records = []
    for encoded in encoded_records:
        mapping = json.loads(encoded)
        if mapping.get("record_type") != "evidence":
            raise ValueError("profiler checkpoint journal record is invalid")
        evidence = ProfilerEvidence.from_mapping(mapping["evidence"])
        request = _PARALLEL_JOURNAL_REQUESTS.get(evidence.request_id)
        if request is None:
            raise ValueError(
                "profiler checkpoint journal contains a foreign request"
            )
        evidence.validate(request)
        evidence_records.append(evidence)
    return tuple(evidence_records)


def _load_incremental_evidence(
    output: Path,
    requests: ProfilerRequestManifest,
    resume: bool,
    *,
    workers: int | None = None,
    parallel_threshold_bytes: int = 16 * 1024 * 1024,
) -> dict[str, ProfilerEvidence]:
    """Load a manifest and journal on affinity-visible physical cores.

    The journal can exceed a gigabyte during an all-format GPU profile. Each
    line is independently canonical and fsynced, so workers decode disjoint
    newline-aligned ranges and validate records against the inherited immutable
    request map. The parent alone applies ordered duplicate/rewrite semantics.
    """

    journal = Path(str(output) + ".inprogress.jsonl")
    if not output.exists() and not journal.exists():
        return {}
    if not resume:
        existing = output if output.exists() else journal
        raise ValueError(f"evidence output already exists; pass --resume: {existing}")
    records: dict[str, ProfilerEvidence] = {}
    if output.exists():
        manifest = read_profiler_evidence_manifest(output)
        if manifest.collector_version != PROFILER_COLLECTOR_VERSION:
            raise ValueError(
                "cannot resume a profiler checkpoint from another collector "
                "generation; retain it as immutable fitting evidence"
            )
        validate_profiler_evidence_coverage(
            requests, manifest, require_complete=False
        )
        records.update({item.request_id: item for item in manifest.evidence})
    if journal.exists():
        if parallel_threshold_bytes < 1:
            raise ValueError("profiler journal parallel threshold must be positive")
        header_line, content_begin, content_end = _truncate_torn_journal_tail(
            journal
        )
        header = json.loads(header_line)
        expected_header = {
            "collector_version": PROFILER_COLLECTOR_VERSION,
            "record_type": "native-vnni-profiler-journal-v1",
            "request_manifest_digest": requests.digest(),
        }
        if header != expected_header:
            raise ValueError("profiler checkpoint journal header is incompatible")
        request_by_id = {
            request.request_id: request for request in requests.requests
        }
        payload_bytes = content_end - content_begin
        if workers is None:
            worker_count = _offline_worker_count(
                max(1, payload_bytes // 4096),
                environment_name="LLAMINAR_NATIVE_VNNI_IO_WORKERS",
            )
        else:
            if workers < 1:
                raise ValueError("profiler journal worker count must be positive")
            worker_count = min(workers, _physical_core_count())
        if payload_bytes < parallel_threshold_bytes:
            worker_count = 1
        tasks = _journal_byte_ranges(
            journal,
            content_begin,
            content_end,
            max(1, worker_count),
        )

        def merge_partition(partition: Iterable[ProfilerEvidence]) -> None:
            """Apply historical ordered terminal-rewrite semantics."""

            for evidence in partition:
                previous = records.get(evidence.request_id)
                if previous is not None and previous.status in {
                    ProfilerEvidenceStatus.COMPLETE,
                    ProfilerEvidenceStatus.CANDIDATE_UNSUPPORTED,
                } and previous.canonical_mapping() != evidence.canonical_mapping():
                    raise ValueError(
                        "profiler checkpoint journal rewrites terminal evidence"
                    )
                records[evidence.request_id] = evidence

        global _PARALLEL_JOURNAL_REQUESTS
        _PARALLEL_JOURNAL_REQUESTS = request_by_id
        try:
            if len(tasks) <= 1:
                for task in tasks:
                    merge_partition(_decode_validate_journal_range(task))
            else:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=len(tasks),
                    mp_context=multiprocessing.get_context("fork"),
                ) as executor:
                    for partition in executor.map(
                        _decode_validate_journal_range,
                        tasks,
                    ):
                        merge_partition(partition)
        finally:
            _PARALLEL_JOURNAL_REQUESTS = {}
        recovered = ProfilerEvidenceManifest(
            request_manifest_digest=requests.digest(),
            corpus_digest=requests.corpus_digest,
            candidate_registry_digest=requests.candidate_registry_digest,
            evidence=tuple(records[key] for key in sorted(records)),
        )
        validate_profiler_evidence_coverage(
            requests, recovered, require_complete=False
        )
    return records


def _append_checkpoint_journal(
    output: Path,
    requests: ProfilerRequestManifest,
    evidence: Iterable[ProfilerEvidence],
) -> None:
    """Durably append completed exact records without rewriting the manifest."""

    records = tuple(evidence)
    if not records:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    journal = Path(str(output) + ".inprogress.jsonl")
    new_file = not journal.exists()
    with journal.open("a", encoding="utf-8") as handle:
        if new_file:
            handle.write(json.dumps({
                "collector_version": PROFILER_COLLECTOR_VERSION,
                "record_type": "native-vnni-profiler-journal-v1",
                "request_manifest_digest": requests.digest(),
            }, sort_keys=True, separators=(",", ":")) + "\n")
        for item in records:
            handle.write(json.dumps({
                "evidence": item.canonical_mapping(),
                "record_type": "evidence",
            }, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_checkpoint(
    output: Path,
    requests: ProfilerRequestManifest,
    records: Mapping[str, ProfilerEvidence],
) -> ProfilerEvidenceManifest:
    """Atomically replace the incremental JSON after every candidate."""

    manifest = ProfilerEvidenceManifest(
        request_manifest_digest=requests.digest(),
        corpus_digest=requests.corpus_digest,
        candidate_registry_digest=requests.candidate_registry_digest,
        evidence=tuple(records[key] for key in sorted(records)),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_profiler_evidence_manifest(output, manifest)
    journal = Path(str(output) + ".inprogress.jsonl")
    if journal.exists():
        journal.unlink()
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the unattended per-backend collector command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-directory", required=True)
    parser.add_argument("--backend", required=True, choices=[item.value for item in Backend])
    parser.add_argument("--binary")
    parser.add_argument("--tool")
    parser.add_argument(
        "--sudo-tool",
        action="store_true",
        help="launch the profiler itself through non-interactive sudo -E",
    )
    parser.add_argument("--cpu-list")
    parser.add_argument("--cpu-avx2-binary")
    parser.add_argument("--cpu-avx512-binary")
    parser.add_argument(
        "--cpu-process-batch-size",
        type=int,
        default=1024,
        help=(
            "maximum exact requests sharing CPU fixture/OpenMP setup; every "
            "member still owns one isolated counter transaction (default: 1024)"
        ),
    )
    parser.add_argument(
        "--disable-cpu-process-batching",
        action="store_true",
        help="diagnostic only: restore one fresh CPU trainer process per request",
    )
    parser.add_argument(
        "--gpu-process-batch-size",
        type=int,
        default=None,
        help=(
            "maximum exact GPU requests sharing one profiler process/report; "
            "each member retains a distinct controlled range. Defaults to "
            "512 for CUDA and 256 for ROCm; ROCm is capped at 256 total and "
            "256 graph-captured requests per process"
        ),
    )
    parser.add_argument(
        "--disable-gpu-process-batching",
        action="store_true",
        help="diagnostic only: restore one profiler process per GPU request",
    )
    parser.add_argument("--request-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--device-lanes",
        type=int,
        default=1,
        help=(
            "independent physical profiler workers; CPU lanes use "
            "semicolon-delimited --cpu-list masks (default: 1)"
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    return parser


def _validate_gpu_process_batch_size(backend: Backend, size: int) -> None:
    """Reject an unsafe ROCm profiler lifetime before launching a process."""

    if size <= 0:
        raise ValueError("--gpu-process-batch-size must be positive")
    if backend == Backend.ROCM and size > ROCM_MAX_GPU_PROCESS_BATCH_SIZE:
        raise ValueError(
            "ROCm profiler process batches are capped at "
            f"{ROCM_MAX_GPU_PROCESS_BATCH_SIZE} requests: ROCm 7.1 "
            "selected-region graph interception corrupts HSA packet state "
            "beyond that validated lifetime"
        )


def _resolve_gpu_process_batch_size(
    backend: Backend,
    requested: int | None,
) -> int:
    """Resolve the backend default and enforce the validated ROCm lifetime."""

    size = (
        ROCM_MAX_GPU_PROCESS_BATCH_SIZE
        if requested is None and backend == Backend.ROCM
        else 512 if requested is None else requested
    )
    _validate_gpu_process_batch_size(backend, size)
    return size


def main(argv: Sequence[str] | None = None) -> int:
    """Collect selected requests, checkpoint each result, and report coverage."""

    args = build_argument_parser().parse_args(argv)
    backend = Backend(args.backend)
    requests = read_profiler_request_manifest(Path(args.requests))
    output = Path(args.output)
    records = _load_incremental_evidence(output, requests, args.resume)
    selected_ids = set(args.request_id)
    pending = [
        request
        for request in requests.requests
        if request.backend == backend
        and (not selected_ids or request.request_id in selected_ids)
        and (
            request.request_id not in records
            or (
                args.retry_failures
                and records[request.request_id].status
                not in {
                    ProfilerEvidenceStatus.COMPLETE,
                    ProfilerEvidenceStatus.CANDIDATE_UNSUPPORTED,
                }
            )
        )
    ]
    if selected_ids - {request.request_id for request in requests.requests}:
        raise ValueError("--request-id contains an ID absent from the manifest")
    selected_backends = {
        request.backend
        for request in requests.requests
        if request.request_id in selected_ids
    }
    if selected_ids and selected_backends != {backend}:
        raise ValueError("--request-id contains a request for another backend")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        pending = pending[: args.limit]
    if args.device_lanes <= 0:
        raise ValueError("--device-lanes must be positive")
    if args.cpu_process_batch_size <= 0:
        raise ValueError("--cpu-process-batch-size must be positive")
    gpu_process_batch_size = _resolve_gpu_process_batch_size(
        backend,
        args.gpu_process_batch_size,
    )
    cpu_lane_lists = (
        _parse_cpu_lane_lists(args.cpu_list, args.device_lanes)
        if backend == Backend.CPU
        else ()
    )

    tool = _resolve_tool(backend, None if args.tool is None else Path(args.tool))
    tool_path = tool or Path(args.tool or DEFAULT_TOOL_CANDIDATES[backend][0])
    selected_binary = (
        Path(args.binary) if args.binary else DEFAULT_BINARIES[backend]
    )
    tool_command_prefix: tuple[str, ...] = ()
    if args.sudo_tool and os.geteuid() != 0:
        if shutil.which("sudo") is None:
            raise ValueError("--sudo-tool requires an executable sudo")
        tool_command_prefix = ("sudo", "-n", "-E")
    options = CollectorOptions(
        backend=backend,
        binary=selected_binary,
        tool=tool_path,
        raw_directory=Path(args.raw_directory),
        cpu_list=cpu_lane_lists[0] if cpu_lane_lists else args.cpu_list,
        timeout_seconds=args.timeout_seconds,
        cpu_avx2_binary=(
            selected_binary
            if args.cpu_avx2_binary is None
            else Path(args.cpu_avx2_binary)
        ),
        cpu_avx512_binary=(
            DEFAULT_CPU_AVX512_BINARY
            if args.cpu_avx512_binary is None
            else Path(args.cpu_avx512_binary)
        ),
        tool_command_prefix=tool_command_prefix,
    )
    if tool is None and pending:
        for request in pending:
            records[request.request_id] = _failed_evidence(
                request,
                ProfilerEvidenceStatus.TOOL_UNAVAILABLE,
                f"profiler tool is unavailable: {tool_path}",
                EXPECTED_PROFILER_TOOL[request.backend],
            )
        manifest = _write_checkpoint(output, requests, records)
        report = validate_profiler_evidence_coverage(
            requests, manifest, require_complete=False
        )
        print(json.dumps(asdict(report), sort_keys=True))
        return 1
    failures = 0

    def announce(index: int, request: ProfilerRequest, lane: int) -> None:
        """Print one stable launch record before handing work to a GPU lane."""

        print(
            f"[{backend.value} {index}/{len(pending)}] {request.request_id} "
            f"lane={lane} "
            f"{request.source_format} M={request.m} "
            f"N={request.aggregate_n} K={request.k} "
            f"{request.effective_candidate_id}",
            flush=True,
        )

    def track(request: ProfilerRequest, result: ProfilerEvidence) -> None:
        """Add one completed exact result to coordinator-owned state."""

        nonlocal failures
        records[request.request_id] = result
        if result.status not in {
            ProfilerEvidenceStatus.COMPLETE,
            ProfilerEvidenceStatus.CANDIDATE_UNSUPPORTED,
        }:
            failures += 1
            print(f"  failed: {result.status.value}: {result.status_reason}", file=sys.stderr)

    def publish(request: ProfilerRequest, result: ProfilerEvidence) -> None:
        """Checkpoint one completed lane result in the coordinator process."""

        track(request, result)
        _append_checkpoint_journal(output, requests, (result,))

    if (
        backend in {Backend.CUDA, Backend.ROCM}
        and not args.disable_gpu_process_batching
        and pending
    ):
        required = [request for request in pending if request.profile_required]
        terminal = [request for request in pending if not request.profile_required]
        for index, request in enumerate(terminal, start=1):
            announce(index, request, 0)
            track(request, _unsupported_evidence(request))
        if terminal:
            _append_checkpoint_journal(
                output,
                requests,
                (records[request.request_id] for request in terminal),
            )

        batches = _build_gpu_process_batches(
            required,
            gpu_process_batch_size,
            maximum_graph_captured=(
                ROCM_MAX_GRAPH_REQUESTS_PER_PROCESS
                if backend == Backend.ROCM
                else None
            ),
        )
        batch_collector = (
            collect_cuda_batch if backend == Backend.CUDA else collect_rocm_batch
        )

        def announce_gpu_batch(
            index: int,
            batch: GPUProfilerProcessBatch,
            lane: int,
        ) -> None:
            """Report one GPU process and its independently ranged members."""

            first = batch.requests[0]
            print(
                f"[{backend.value} batch {index}/{len(batches)}] lane={lane} "
                f"requests={len(batch.requests)} "
                f"format={first.source_format} operation={first.operation_kind}",
                flush=True,
            )

        def publish_gpu_batch(
            batch: GPUProfilerProcessBatch,
            results: Mapping[str, ProfilerEvidence],
        ) -> None:
            """Atomically journal every exact member of one GPU process."""

            expected = {request.request_id for request in batch.requests}
            if set(results) != expected:
                raise RuntimeError(
                    f"{backend.value} profiler batch returned a partial ID set"
                )
            for request in batch.requests:
                track(request, results[request.request_id])
            _append_checkpoint_journal(
                output,
                requests,
                (results[request.request_id] for request in batch.requests),
            )

        if args.device_lanes == 1:
            for index, batch in enumerate(batches, start=1):
                announce_gpu_batch(index, batch, 0)
                publish_gpu_batch(
                    batch,
                    batch_collector(
                        batch,
                        replace(options, device_ordinal=0),
                    ),
                )
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.device_lanes,
            ) as executor:
                indexed_batches = iter(enumerate(batches, start=1))
                active_batches: dict[
                    concurrent.futures.Future[dict[str, ProfilerEvidence]],
                    tuple[GPUProfilerProcessBatch, int],
                ] = {}

                def submit_gpu_batch(lane: int) -> bool:
                    try:
                        index, batch = next(indexed_batches)
                    except StopIteration:
                        return False
                    announce_gpu_batch(index, batch, lane)
                    future = executor.submit(
                        batch_collector,
                        batch,
                        replace(options, device_ordinal=lane),
                    )
                    active_batches[future] = (batch, lane)
                    return True

                for lane in range(args.device_lanes):
                    if not submit_gpu_batch(lane):
                        break
                while active_batches:
                    done, _ = concurrent.futures.wait(
                        active_batches,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        batch, lane = active_batches.pop(future)
                        publish_gpu_batch(batch, future.result())
                        submit_gpu_batch(lane)

        manifest = _write_checkpoint(output, requests, records)
        report = validate_profiler_evidence_coverage(
            requests, manifest, require_complete=args.finalize
        )
        print(json.dumps(asdict(report), sort_keys=True))
        return 1 if failures else 0

    if (
        backend == Backend.CPU
        and not args.disable_cpu_process_batching
        and pending
    ):
        required = [request for request in pending if request.profile_required]
        terminal = [request for request in pending if not request.profile_required]
        for index, request in enumerate(terminal, start=1):
            announce(index, request, 0)
            track(request, _unsupported_evidence(request))
        if terminal:
            _append_checkpoint_journal(
                output,
                requests,
                (records[request.request_id] for request in terminal),
            )

        batches = _build_cpu_process_batches(
            required, args.cpu_process_batch_size
        )

        def announce_batch(
            index: int,
            batch: CPUProfilerProcessBatch,
            lane: int,
        ) -> None:
            """Report one process and its non-merged exact launch count."""

            first = batch.requests[0]
            print(
                f"[cpu batch {index}/{len(batches)}] lane={lane} "
                f"requests={len(batch.requests)} N={first.aggregate_n} "
                f"K={first.k} operation={first.operation_kind}",
                flush=True,
            )

        def publish_batch(
            batch: CPUProfilerProcessBatch,
            results: Mapping[str, ProfilerEvidence],
        ) -> None:
            """Publish every exact member after one batch process exits."""

            expected = {request.request_id for request in batch.requests}
            if set(results) != expected:
                raise RuntimeError("CPU profiler batch returned a partial ID set")
            for request in batch.requests:
                track(request, results[request.request_id])
            _append_checkpoint_journal(
                output,
                requests,
                (results[request.request_id] for request in batch.requests),
            )

        if args.device_lanes == 1:
            for index, batch in enumerate(batches, start=1):
                announce_batch(index, batch, 0)
                publish_batch(
                    batch,
                    collect_cpu_batch(
                        batch,
                        replace(options, cpu_list=cpu_lane_lists[0]),
                    ),
                )
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.device_lanes,
            ) as executor:
                indexed_batches = iter(enumerate(batches, start=1))
                active_batches: dict[
                    concurrent.futures.Future[dict[str, ProfilerEvidence]],
                    tuple[CPUProfilerProcessBatch, int],
                ] = {}

                def submit_batch(lane: int) -> bool:
                    try:
                        index, batch = next(indexed_batches)
                    except StopIteration:
                        return False
                    announce_batch(index, batch, lane)
                    future = executor.submit(
                        collect_cpu_batch,
                        batch,
                        replace(options, cpu_list=cpu_lane_lists[lane]),
                    )
                    active_batches[future] = (batch, lane)
                    return True

                for lane in range(args.device_lanes):
                    if not submit_batch(lane):
                        break
                while active_batches:
                    done, _ = concurrent.futures.wait(
                        active_batches,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        batch, lane = active_batches.pop(future)
                        publish_batch(batch, future.result())
                        submit_batch(lane)

        manifest = _write_checkpoint(output, requests, records)
        report = validate_profiler_evidence_coverage(
            requests, manifest, require_complete=args.finalize
        )
        print(json.dumps(asdict(report), sort_keys=True))
        return 1 if failures else 0

    if args.device_lanes == 1:
        for index, request in enumerate(pending, start=1):
            announce(index, request, 0)
            publish(request, collect_request(
                request,
                replace(
                    options,
                    cpu_list=(cpu_lane_lists[0] if cpu_lane_lists else options.cpu_list),
                    device_ordinal=0 if backend != Backend.CPU else None,
                ),
            ))
    else:
        # Keep at most one profiler process on each physical device. The
        # coordinator alone mutates the evidence checkpoint, so concurrent
        # workers cannot race JSON publication or lose a completed request.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.device_lanes,
        ) as executor:
            indexed = iter(enumerate(pending, start=1))
            active: dict[
                concurrent.futures.Future[ProfilerEvidence],
                tuple[ProfilerRequest, int],
            ] = {}

            def submit_next(lane: int) -> bool:
                """Submit the next pending request to one newly free lane."""

                try:
                    index, request = next(indexed)
                except StopIteration:
                    return False
                announce(index, request, lane)
                future = executor.submit(
                    collect_request,
                    request,
                    replace(
                        options,
                        cpu_list=(
                            cpu_lane_lists[lane]
                            if cpu_lane_lists
                            else options.cpu_list
                        ),
                        device_ordinal=(lane if backend != Backend.CPU else None),
                    ),
                )
                active[future] = (request, lane)
                return True

            for lane in range(args.device_lanes):
                if not submit_next(lane):
                    break
            while active:
                done, _ = concurrent.futures.wait(
                    active,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    request, lane = active.pop(future)
                    publish(request, future.result())
                    submit_next(lane)

    manifest = _write_checkpoint(output, requests, records)
    report = validate_profiler_evidence_coverage(
        requests, manifest, require_complete=args.finalize
    )
    print(json.dumps(asdict(report), sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
