"""Isolated Linux perf, Nsight Compute, and rocprofiler collectors.

The collector consumes :mod:`profiler_evidence` requests only after canonical
timing has finished. GPU requests use fresh profiler processes. CPU requests
may share one trainer process when their build, ISA, operation, and N/K geometry
match; each batch member still owns a separately reset/armed/disarmed
``perf_event_open`` interval, exactly one production launch, and one raw report.
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
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed.avg.per_cycle_active",
    "smsp__average_warp_latency_per_inst_issued.ratio",
    "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum",
)


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
                "profiler_binary_digest": _sha256_file(binary),
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


def _profile_environment(request: ProfilerRequest, raw_dir: Path) -> dict[str, str]:
    """Build exact one-cell trainer filters from one authenticated request."""

    environment = os.environ.copy()
    environment["LLAMINAR_NATIVE_VNNI_PROFILE_REQUEST_ID"] = request.request_id
    environment["LLAMINAR_PERF_STATS_JSON"] = "1"
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
        config = request.config_json
        if "kb" in config:
            variant = f"KB{int(config['kb'])}"
        elif config.get("split_policy") == "inherit_serial_m1":
            variant = "INHERIT_SERIAL_M1"
        else:
            raise ValueError(
                f"{request.request_id}: unsupported ROCm profiler candidate config"
            )
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
}


def _convert_ncu_value(metric_id: str, value: float, unit: str) -> float:
    """Convert Nsight display units to the canonical metric schema."""

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
        if "kbyte" in normalized or normalized in {"kb", "kib"}:
            return value * 1024.0
        if "mbyte" in normalized or normalized in {"mb", "mib"}:
            return value * 1024.0 * 1024.0
    return value


def parse_ncu_csv(path: Path, request: ProfilerRequest) -> tuple[ProfiledDispatch, ...]:
    """Normalize raw-page Nsight Compute CSV into ordered kernel dispatches."""

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
    grouped: dict[tuple[int, str, tuple[int, int, int], tuple[int, int, int]], dict[str, tuple[float, str]]] = {}
    for fallback_index, row in enumerate(reader):
        kernel = (row.get("Kernel Name") or "").strip()
        if not kernel:
            continue
        identifier_raw = (row.get("ID") or str(fallback_index)).strip()
        try:
            identifier = int(identifier_raw)
        except ValueError:
            identifier = fallback_index
        grid = _parse_geometry((row.get("Grid Size") or "1").strip())
        block = _parse_geometry((row.get("Block Size") or "1").strip())
        key = (identifier, kernel, grid, block)
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

    dispatches = []
    for dispatch_index, ((_, kernel, grid, block), values) in enumerate(
        sorted(grouped.items(), key=lambda item: item[0])
    ):
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
        dispatches.append(dispatch)
    if not dispatches:
        raise ValueError("Nsight Compute reported no kernels in the target region")
    return tuple(dispatches)


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


def _load_incremental_evidence(
    output: Path,
    requests: ProfilerRequestManifest,
    resume: bool,
) -> dict[str, ProfilerEvidence]:
    """Load a materialized manifest plus any durable in-progress journal."""

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
        payload = journal.read_bytes()
        lines = payload.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b"\n"):
            # A killed append may leave only its final JSON object torn. Every
            # earlier newline-terminated member is independently recoverable.
            lines.pop()
            with journal.open("r+b") as handle:
                handle.truncate(sum(len(line) for line in lines))
                handle.flush()
                os.fsync(handle.fileno())
        if not lines:
            raise ValueError("profiler checkpoint journal has no complete header")
        header = json.loads(lines[0])
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
        for encoded in lines[1:]:
            mapping = json.loads(encoded)
            if mapping.get("record_type") != "evidence":
                raise ValueError("profiler checkpoint journal record is invalid")
            evidence = ProfilerEvidence.from_mapping(mapping["evidence"])
            request = request_by_id.get(evidence.request_id)
            if request is None:
                raise ValueError(
                    "profiler checkpoint journal contains a foreign request"
                )
            evidence.validate(request)
            previous = records.get(evidence.request_id)
            if previous is not None and previous.status in {
                ProfilerEvidenceStatus.COMPLETE,
                ProfilerEvidenceStatus.CANDIDATE_UNSUPPORTED,
            } and previous.canonical_mapping() != evidence.canonical_mapping():
                raise ValueError(
                    "profiler checkpoint journal rewrites terminal evidence"
                )
            records[evidence.request_id] = evidence
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
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=output.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(manifest.canonical_mapping(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output)
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
        default=256,
        help=(
            "maximum exact requests sharing CPU fixture/OpenMP setup; every "
            "member still owns one isolated counter transaction (default: 256)"
        ),
    )
    parser.add_argument(
        "--disable-cpu-process-batching",
        action="store_true",
        help="diagnostic only: restore one fresh CPU trainer process per request",
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
