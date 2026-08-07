"""Resumable production-MoE prefill candidate-pair sweep transaction.

The C++ performance binary owns GPU execution, graph capture, correctness, and
canonical HIP/CUDA event timing. This module owns the durable transaction
around those launches: it expands the pinned GGUF inventory into immutable
``(backend, routed source tuple, route profile, M)`` cells, assigns cells to distinct devices,
authenticates every aggregate and raw timing row, and atomically promotes only
complete cells. A rerun therefore preserves every validated cell and launches
only missing or invalid work.

One cell is intentionally the unit of process isolation. The C++ executable
prepares all real experts once, screens the complete gate/up-by-down candidate
pair space, and robustly retimes finalists. Distinct cells can run concurrently
on distinct physical accelerators without duplicating a cell or sharing a GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .adapters.evidence import verify_aggregate_timing
from .prefill_matrix import GPU_PREFILL_M_BUCKETS
from .moe_routing_profiles import (
    MOE_ROUTING_PROFILES,
    summarize_moe_routing_profile,
)
from .qwen_moe_gguf_patterns import (
    QwenMoERoutedPrefillCase,
    load_qwen_moe_gguf_pattern_inventory,
    qwen_moe_routed_prefill_cases,
)


AGGREGATE_COLUMNS = (
    "backend", "phase", "case_id", "route_profile", "source_gate", "source_up",
    "source_down", "gate_execution_codebook", "up_execution_codebook",
    "down_execution_codebook", "hidden_size", "expert_width",
    "expert_count", "top_k", "route_active_experts",
    "route_max_assignments", "route_assignment_cv", "m", "candidate_id", "gate_tile_m",
    "gate_tile_n", "down_tile_m", "down_tile_n", "gate_local_bytes",
    "down_local_bytes", "gate_registers", "down_registers",
    "gate_shared_bytes", "down_shared_bytes", "gate_active_blocks_per_sm",
    "down_active_blocks_per_sm", "screening_sample_count",
    "screening_replays", "robust_sample_count", "robust_replays",
    "selected_sample_count", "selected_replays", "min_us", "median_us",
    "p95_us", "mad_us", "cv", "timing_sample_digest", "bit_mismatches",
    "first_bit_mismatch", "route_counter_ok", "finalist", "is_winner",
)

TIMING_COLUMNS = (
    "backend", "phase", "case_id", "route_profile", "m", "candidate_id", "timing_phase",
    "sample_index", "timed_replays", "latency_us", "latency_ms_hex",
)

FIXED_TILE_CANDIDATES = tuple(
    (tile_m, tile_n)
    for tile_m in (4, 8, 12, 16)
    for tile_n in (64, 128, 256)
)

# Marker 20 is the fully device-owned TM12/TM16 adaptive candidate. It uses a
# single measured output width until another adaptive width proves it deserves
# the additional compile and corpus cost.
TILE_CANDIDATES = FIXED_TILE_CANDIDATES + ((20, 128),)

PAIR_CANDIDATE_IDS = tuple(
    f"g_tm{gate_m}_tn{gate_n}__d_tm{down_m}_tn{down_n}"
    for gate_m, gate_n in TILE_CANDIDATES
    for down_m, down_n in TILE_CANDIDATES
)

CUDA_GATEUP_TILE_N_CANDIDATES = (64, 96, 128, 160, 192, 224, 256)
CUDA_CANDIDATE_IDS = tuple(
    f"g_tn{tile_n}__d_fixed"
    for tile_n in CUDA_GATEUP_TILE_N_CANDIDATES
)

_UINT64_MAX = (1 << 64) - 1


@dataclass(frozen=True, order=True)
class ProductionMoEPrefillCell:
    """One immutable, independently resumable accelerator measurement cell."""

    backend: str
    case: QwenMoERoutedPrefillCase
    m: int
    route_profile: str

    def __post_init__(self) -> None:
        """Reject incomplete corpus identities before paths are derived."""

        if self.backend not in {"cuda", "rocm"}:
            raise ValueError(f"unsupported production GPU backend {self.backend!r}")
        if self.m <= 0:
            raise ValueError("production MoE sweep M must be positive")
        if self.route_profile not in MOE_ROUTING_PROFILES:
            raise ValueError(
                f"unknown MoE routing profile {self.route_profile!r}"
            )

    @property
    def identity(self) -> tuple[object, ...]:
        """Return the complete timing-corpus identity for this cell."""

        return self.backend, self.case.source_key, self.route_profile, self.m

    @property
    def cell_id(self) -> str:
        """Return a bounded content-addressed filename stem."""

        encoded = json.dumps(
            self.identity, sort_keys=True, separators=(",", ":")
        ).encode()
        suffix = hashlib.sha256(encoded).hexdigest()[:20]
        return f"{self.backend}-{self.route_profile}-m{self.m}-{suffix}"

    def canonical_mapping(self) -> dict[str, object]:
        """Serialize this cell for a reviewable deterministic plan."""

        return {
            "backend": self.backend,
            "cell_id": self.cell_id,
            "case_id": self.case.evidence_id,
            "route_profile": self.route_profile,
            "m": self.m,
            "hidden_size": self.case.hidden_size,
            "routed_expert_width": self.case.routed_expert_width,
            "expert_count": self.case.expert_count,
            "experts_per_token": self.case.experts_per_token,
            "source_gate": self.case.routed.gate,
            "source_up": self.case.routed.up,
            "source_down": self.case.routed.down,
            "mixture_overlay_keys": [
                list(key) for key in self.case.mixture_overlay_keys
            ],
        }


@dataclass(frozen=True)
class CellValidation:
    """Authenticated completion summary for one cell."""

    candidate_count: int
    finalist_count: int
    winner_id: str
    winner_median_us: float


@dataclass(frozen=True)
class RawTimingPhase:
    """Sorted native event samples and their exact replay cardinality."""

    samples_ms: tuple[float, ...]
    replays_per_sample: int


@dataclass(frozen=True)
class CellPaths:
    """Final and staging paths for one atomic cell transaction."""

    aggregate: Path
    timing: Path
    log: Path
    aggregate_staging: Path
    timing_staging: Path
    log_staging: Path


def production_moe_prefill_cells(
    backend: str,
) -> tuple[ProductionMoEPrefillCell, ...]:
    """Expand every routed GGUF source tuple across every GPU M bucket."""

    normalized = backend.strip().lower()
    if normalized not in {"cuda", "rocm"}:
        raise ValueError(f"production MoE GPU sweep does not support {backend!r}")
    return tuple(
        ProductionMoEPrefillCell(normalized, case, m, route_profile)
        for case in qwen_moe_routed_prefill_cases()
        for route_profile in MOE_ROUTING_PROFILES
        for m in GPU_PREFILL_M_BUCKETS
    )


def production_moe_prefill_candidate_ids(backend: str) -> tuple[str, ...]:
    """Return the complete byte-eligible candidate inventory per backend."""

    normalized = backend.strip().lower()
    if normalized == "rocm":
        return PAIR_CANDIDATE_IDS
    if normalized == "cuda":
        return CUDA_CANDIDATE_IDS
    raise ValueError(f"production MoE GPU sweep does not support {backend!r}")


def _cell_paths(root: Path, cell: ProductionMoEPrefillCell) -> CellPaths:
    cells = root / "cells"
    stem = cells / cell.cell_id
    return CellPaths(
        aggregate=stem.with_suffix(".csv"),
        timing=stem.with_suffix(".timing.csv"),
        log=stem.with_suffix(".log"),
        aggregate_staging=stem.with_suffix(".csv.inprogress"),
        timing_staging=stem.with_suffix(".timing.csv.inprogress"),
        log_staging=stem.with_suffix(".log.inprogress"),
    )


def production_moe_prefill_cell_paths(
    root: Path,
    cell: ProductionMoEPrefillCell,
) -> CellPaths:
    """Return the stable on-disk locations owned by one planned cell."""

    return _cell_paths(Path(root), cell)


def _require_exact_header(
    path: Path,
    actual: Sequence[str] | None,
    expected: Sequence[str],
) -> None:
    """Reject reordered, duplicated, missing, or future-extended CSV schemas."""

    if tuple(actual or ()) != tuple(expected):
        raise ValueError(
            f"{path}: CSV header mismatch; expected={tuple(expected)!r}, "
            f"actual={tuple(actual or ())!r}"
        )


def _explicit_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"{name} must be an explicit boolean, got {value!r}")


def _expected_runtime_codebooks(
    cell: ProductionMoEPrefillCell,
) -> tuple[int, int, int]:
    return cell.case.routed.runtime_codebooks(cell.backend)


def _read_aggregate_rows(
    path: Path,
    cell: ProductionMoEPrefillCell,
) -> dict[str, dict[str, str]]:
    """Authenticate the complete backend-specific candidate matrix."""

    expected_codebooks = _expected_runtime_codebooks(cell)
    expected_candidate_ids = set(
        production_moe_prefill_candidate_ids(cell.backend)
    )
    indexed: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_exact_header(path, reader.fieldnames, AGGREGATE_COLUMNS)
        for line, row in enumerate(reader, start=2):
            context = f"{path}:{line}"
            candidate_id = row["candidate_id"].strip()
            if candidate_id in indexed:
                raise ValueError(f"{context}: duplicate candidate {candidate_id}")
            if row["backend"] != cell.backend or (
                row["phase"] != "moe_production_prefill"
            ):
                raise ValueError(f"{context}: wrong backend or phase")
            if row["case_id"] != cell.case.evidence_id or int(row["m"]) != cell.m:
                raise ValueError(f"{context}: row belongs to another sweep cell")
            if row["route_profile"] != cell.route_profile:
                raise ValueError(f"{context}: routing profile disagrees with cell identity")
            if (
                row["source_gate"] != cell.case.routed.gate
                or row["source_up"] != cell.case.routed.up
                or row["source_down"] != cell.case.routed.down
            ):
                raise ValueError(f"{context}: source formats disagree with plan")
            observed_codebooks = tuple(
                int(row[name]) for name in (
                    "gate_execution_codebook",
                    "up_execution_codebook",
                    "down_execution_codebook",
                )
            )
            if observed_codebooks != expected_codebooks:
                raise ValueError(f"{context}: runtime codebooks disagree with registry")
            observed_geometry = tuple(
                int(row[name]) for name in (
                    "hidden_size", "expert_width", "expert_count", "top_k"
                )
            )
            expected_geometry = (
                cell.case.hidden_size,
                cell.case.routed_expert_width,
                cell.case.expert_count,
                cell.case.experts_per_token,
            )
            if observed_geometry != expected_geometry:
                raise ValueError(f"{context}: geometry disagrees with plan")
            expected_route_stats = summarize_moe_routing_profile(
                cell.route_profile,
                cell.m,
                cell.case.experts_per_token,
                cell.case.expert_count,
            )
            if int(row["route_active_experts"]) != (
                expected_route_stats.active_experts
            ) or int(row["route_max_assignments"]) != (
                expected_route_stats.maximum_assignments
            ) or not math.isclose(
                float(row["route_assignment_cv"]),
                expected_route_stats.assignment_cv,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"{context}: routing load statistics disagree with profile"
                )
            if candidate_id not in expected_candidate_ids:
                raise ValueError(f"{context}: unknown candidate {candidate_id}")

            if cell.backend == "rocm":
                expected_id = (
                    f"g_tm{int(row['gate_tile_m'])}_tn{int(row['gate_tile_n'])}"
                    f"__d_tm{int(row['down_tile_m'])}_tn{int(row['down_tile_n'])}"
                )
            else:
                expected_id = f"g_tn{int(row['gate_tile_n'])}__d_fixed"
                expected_fixed_down_threads = cell.case.experts_per_token * 32
                if (
                    int(row["gate_tile_m"]) != 0
                    or int(row["down_tile_m"]) != 0
                    or int(row["down_tile_n"]) != expected_fixed_down_threads
                ):
                    raise ValueError(
                        f"{context}: CUDA fixed arithmetic geometry changed"
                    )
            if candidate_id != expected_id:
                raise ValueError(
                    f"{context}: candidate ID disagrees with geometry fields"
                )
            if int(row["gate_local_bytes"]) != 0 or (
                int(row["down_local_bytes"]) != 0
            ):
                raise ValueError(f"{context}: spilling candidate reached timing evidence")
            for name in (
                "gate_registers", "down_registers",
                "gate_active_blocks_per_sm", "down_active_blocks_per_sm",
            ):
                if int(row[name]) <= 0:
                    raise ValueError(f"{context}: invalid resource field {name}")
            if int(row["gate_shared_bytes"]) < 0 or int(row["down_shared_bytes"]) < 0:
                raise ValueError(f"{context}: negative shared-memory evidence")
            if int(row["bit_mismatches"]) != 0 or (
                int(row["first_bit_mismatch"]) != _UINT64_MAX
            ):
                raise ValueError(f"{context}: candidate failed byte equality")
            if not _explicit_bool("route_counter_ok", row["route_counter_ok"]):
                raise ValueError(f"{context}: forced production route was not observed")

            finalist = _explicit_bool("finalist", row["finalist"])
            winner = _explicit_bool("is_winner", row["is_winner"])
            if winner and not finalist:
                raise ValueError(f"{context}: winner was not robustly retimed")
            screening_count = int(row["screening_sample_count"])
            screening_replays = int(row["screening_replays"])
            robust_count = int(row["robust_sample_count"])
            robust_replays = int(row["robust_replays"])
            selected_count = int(row["selected_sample_count"])
            selected_replays = int(row["selected_replays"])
            if screening_count <= 0 or screening_replays <= 0:
                raise ValueError(f"{context}: missing screening timing")
            if finalist != (robust_count > 0):
                raise ValueError(f"{context}: finalist/robust timing disagree")
            if finalist and robust_replays <= 0:
                raise ValueError(f"{context}: finalist has no replay count")
            expected_selected = robust_count if finalist else screening_count
            expected_replays = robust_replays if finalist else screening_replays
            if selected_count != expected_selected or selected_replays != expected_replays:
                raise ValueError(f"{context}: selected timing phase is inconsistent")
            indexed[candidate_id] = dict(row)

    if set(indexed) != expected_candidate_ids:
        missing = sorted(expected_candidate_ids - set(indexed))
        unexpected = sorted(set(indexed) - expected_candidate_ids)
        raise ValueError(
            f"{path}: candidate matrix mismatch; missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    return indexed


def _read_timing_rows(
    path: Path,
    cell: ProductionMoEPrefillCell,
) -> dict[tuple[str, str], RawTimingPhase]:
    """Authenticate contiguous, sorted native event samples per timing phase."""

    expected_candidate_ids = set(
        production_moe_prefill_candidate_ids(cell.backend)
    )
    indexed: dict[tuple[str, str], list[float]] = {}
    replay_counts: dict[tuple[str, str], int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_exact_header(path, reader.fieldnames, TIMING_COLUMNS)
        for line, row in enumerate(reader, start=2):
            context = f"{path}:{line}"
            if row["backend"] != cell.backend or (
                row["phase"] != "moe_production_prefill"
            ):
                raise ValueError(f"{context}: wrong backend or phase")
            if row["case_id"] != cell.case.evidence_id or int(row["m"]) != cell.m:
                raise ValueError(f"{context}: row belongs to another sweep cell")
            if row["route_profile"] != cell.route_profile:
                raise ValueError(f"{context}: routing profile disagrees with cell identity")
            candidate_id = row["candidate_id"]
            timing_phase = row["timing_phase"]
            if candidate_id not in expected_candidate_ids or timing_phase not in {
                "screening", "robust"
            }:
                raise ValueError(f"{context}: invalid candidate or timing phase")
            key = candidate_id, timing_phase
            samples = indexed.setdefault(key, [])
            sample_index = int(row["sample_index"])
            if sample_index != len(samples):
                raise ValueError(f"{context}: non-contiguous timing sample index")
            replays = int(row["timed_replays"])
            if replays <= 0:
                raise ValueError(f"{context}: timed_replays must be positive")
            if key in replay_counts and replay_counts[key] != replays:
                raise ValueError(f"{context}: replay count changed within phase")
            replay_counts[key] = replays
            latency_ms = float.fromhex(row["latency_ms_hex"])
            latency_us = float(row["latency_us"])
            if not math.isfinite(latency_ms) or latency_ms <= 0.0:
                raise ValueError(f"{context}: invalid event latency")
            if not math.isclose(
                latency_us, latency_ms * 1000.0, rel_tol=0.0, abs_tol=5.1e-7
            ):
                raise ValueError(f"{context}: readable/exact timing disagree")
            samples.append(latency_ms)

    result: dict[tuple[str, str], RawTimingPhase] = {}
    for key, values in indexed.items():
        samples = tuple(values)
        if tuple(sorted(samples)) != samples:
            raise ValueError(f"{path}: timing samples are not sorted for {key}")
        result[key] = RawTimingPhase(samples, replay_counts[key])
    return result


def validate_production_moe_prefill_cell(
    aggregate_path: Path,
    timing_path: Path,
    cell: ProductionMoEPrefillCell,
) -> CellValidation:
    """Prove one cell complete before it may be reused or combined."""

    aggregate = _read_aggregate_rows(Path(aggregate_path), cell)
    timing = _read_timing_rows(Path(timing_path), cell)
    finalists = []
    winners = []
    for candidate_id, row in aggregate.items():
        finalist = _explicit_bool("finalist", row["finalist"])
        if finalist:
            finalists.append(candidate_id)
        if _explicit_bool("is_winner", row["is_winner"]):
            winners.append(candidate_id)
        expected_phases = {"screening"} | ({"robust"} if finalist else set())
        actual_phases = {
            phase for candidate, phase in timing if candidate == candidate_id
        }
        if actual_phases != expected_phases:
            raise ValueError(
                f"{candidate_id}: timing phases mismatch; "
                f"expected={expected_phases}, actual={actual_phases}"
            )
        for phase in expected_phases:
            phase_evidence = timing[candidate_id, phase]
            samples = phase_evidence.samples_ms
            count_field = (
                "robust_sample_count" if phase == "robust"
                else "screening_sample_count"
            )
            replay_field = (
                "robust_replays" if phase == "robust" else "screening_replays"
            )
            if len(samples) != int(row[count_field]):
                raise ValueError(f"{candidate_id}: {phase} sample count mismatch")
            if phase_evidence.replays_per_sample != int(row[replay_field]):
                raise ValueError(f"{candidate_id}: {phase} replay count mismatch")
        selected_phase = "robust" if finalist else "screening"
        verify_aggregate_timing(
            row,
            timing[candidate_id, selected_phase].samples_ms,
            median_field="median_us",
            sample_to_aggregate_scale=1000.0,
        )

    if not finalists:
        raise ValueError("cell retained no robust finalist")
    if len(winners) != 1:
        raise ValueError(f"cell must contain exactly one winner, got {winners}")
    winner = aggregate[winners[0]]
    return CellValidation(
        candidate_count=len(aggregate),
        finalist_count=len(finalists),
        winner_id=winners[0],
        winner_median_us=float(winner["median_us"]),
    )


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Publish one deterministic JSON document through an adjacent staging file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".inprogress")
    with staging.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, path)


def write_sweep_plan(
    path: Path,
    backend: str,
    cells: Sequence[ProductionMoEPrefillCell],
) -> None:
    """Write the complete GGUF/source/M plan and immutable manifest digest."""

    inventory = load_qwen_moe_gguf_pattern_inventory()
    records = [cell.canonical_mapping() for cell in cells]
    body = {
        "schema_version": "native-vnni-production-moe-prefill-sweep-v3",
        "backend": backend,
        "gguf_manifest_digest": inventory.source_digest,
        "gpu_m_buckets": list(GPU_PREFILL_M_BUCKETS),
        "route_profiles": list(MOE_ROUTING_PROFILES),
        "candidate_count": len(production_moe_prefill_candidate_ids(backend)),
        "cells": records,
    }
    body["plan_digest"] = "sha256:" + hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_json_atomic(Path(path), body)


def _cell_complete(root: Path, cell: ProductionMoEPrefillCell) -> bool:
    paths = _cell_paths(root, cell)
    try:
        validate_production_moe_prefill_cell(
            paths.aggregate, paths.timing, cell
        )
    except (OSError, ValueError):
        return False
    return True


_BACKEND_TESTS = {
    "cuda": "Perf__MoEVerifierPrefill.CUDA_ProductionGGUFMixtureCandidateTrainer",
    "rocm": "Perf__MoEVerifierPrefill.ROCm_ProductionGGUFMixtureCandidatePairTrainer",
}


def _run_accelerator_cell(
    binary: Path,
    root: Path,
    cell: ProductionMoEPrefillCell,
    device: int,
) -> CellValidation:
    """Launch, validate, and atomically promote one isolated GPU cell."""

    if cell.backend not in _BACKEND_TESTS:
        raise ValueError(f"unsupported production GPU backend {cell.backend!r}")
    paths = _cell_paths(root, cell)
    paths.aggregate.parent.mkdir(parents=True, exist_ok=True)
    for staging in (
        paths.aggregate_staging, paths.timing_staging, paths.log_staging
    ):
        staging.unlink(missing_ok=True)

    environment = os.environ.copy()
    prefix = f"LLAMINAR_{cell.backend.upper()}_MOE_PRODUCTION_SWEEP"
    environment.update({
        prefix: "1",
        f"{prefix}_CASES": cell.case.evidence_id,
        f"{prefix}_M": str(cell.m),
        f"{prefix}_ROUTE_PROFILE": cell.route_profile,
        f"{prefix}_MAX_CELLS": "1",
        f"{prefix}_DEVICE": str(device),
        f"{prefix}_CSV": str(
            paths.aggregate_staging
        ),
        f"{prefix}_TIMING_CSV": str(
            paths.timing_staging
        ),
    })
    command = (
        str(binary),
        f"--gtest_filter={_BACKEND_TESTS[cell.backend]}",
    )
    with paths.log_staging.open("wb") as log:
        completed = subprocess.run(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.flush()
        os.fsync(log.fileno())
    if completed.returncode != 0:
        raise RuntimeError(
            f"{cell.cell_id} failed on {cell.backend} device {device}; "
            f"inspect {paths.log_staging}"
        )
    validation = validate_production_moe_prefill_cell(
        paths.aggregate_staging,
        paths.timing_staging,
        cell,
    )
    os.replace(paths.aggregate_staging, paths.aggregate)
    os.replace(paths.timing_staging, paths.timing)
    os.replace(paths.log_staging, paths.log)
    return validation


def run_missing_accelerator_cells(
    binary: Path,
    root: Path,
    cells: Sequence[ProductionMoEPrefillCell],
    devices: Sequence[int],
    *,
    maximum_new_cells: int | None = None,
) -> tuple[ProductionMoEPrefillCell, ...]:
    """Run distinct missing cells concurrently, one worker per physical GPU."""

    backends = {cell.backend for cell in cells}
    if len(backends) != 1 or not backends or not backends <= set(_BACKEND_TESTS):
        raise ValueError("one run must contain cells for exactly one GPU backend")
    backend = next(iter(backends))
    if not devices or len(set(devices)) != len(devices) or any(
        device < 0 for device in devices
    ):
        raise ValueError(
            f"{backend} device list must contain distinct non-negative ordinals"
        )
    binary = Path(binary)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"{backend} trainer binary is not executable: {binary}")
    root = Path(root)
    missing = [cell for cell in cells if not _cell_complete(root, cell)]
    if maximum_new_cells is not None:
        if maximum_new_cells <= 0:
            raise ValueError("maximum_new_cells must be positive")
        missing = missing[:maximum_new_cells]
    work: queue.Queue[ProductionMoEPrefillCell] = queue.Queue()
    for cell in missing:
        work.put(cell)

    failures: list[BaseException] = []
    completed_cells: list[ProductionMoEPrefillCell] = []
    lock = threading.Lock()
    stop = threading.Event()

    def worker(device: int) -> None:
        while not stop.is_set():
            try:
                cell = work.get_nowait()
            except queue.Empty:
                return
            try:
                _run_accelerator_cell(binary, root, cell, device)
                with lock:
                    completed_cells.append(cell)
            except BaseException as error:  # preserve worker traceback object
                with lock:
                    failures.append(error)
                stop.set()
            finally:
                work.task_done()

    threads = [
        threading.Thread(target=worker, args=(device,), daemon=False)
        for device in devices
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(
            f"{backend} production MoE sweep worker failed"
        ) from failures[0]
    return tuple(sorted(completed_cells))


def run_missing_rocm_cells(
    binary: Path,
    root: Path,
    cells: Sequence[ProductionMoEPrefillCell],
    devices: Sequence[int],
    *,
    maximum_new_cells: int | None = None,
) -> tuple[ProductionMoEPrefillCell, ...]:
    """Run missing ROCm cells through the backend-neutral transaction."""

    if any(cell.backend != "rocm" for cell in cells):
        raise ValueError("ROCm runner received a non-ROCm cell")
    return run_missing_accelerator_cells(
        binary,
        root,
        cells,
        devices,
        maximum_new_cells=maximum_new_cells,
    )


def run_missing_cuda_cells(
    binary: Path,
    root: Path,
    cells: Sequence[ProductionMoEPrefillCell],
    devices: Sequence[int],
    *,
    maximum_new_cells: int | None = None,
) -> tuple[ProductionMoEPrefillCell, ...]:
    """Run missing CUDA cells through the backend-neutral transaction."""

    if any(cell.backend != "cuda" for cell in cells):
        raise ValueError("CUDA runner received a non-CUDA cell")
    return run_missing_accelerator_cells(
        binary,
        root,
        cells,
        devices,
        maximum_new_cells=maximum_new_cells,
    )


def _combine_csvs_atomic(
    output: Path,
    inputs: Iterable[Path],
    columns: Sequence[str],
) -> None:
    """Combine validated one-cell CSVs without retaining duplicate headers."""

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".inprogress")
    with staging.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for path in inputs:
            with Path(path).open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                _require_exact_header(Path(path), reader.fieldnames, columns)
                writer.writerows(reader)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(staging, output)


def combine_production_moe_prefill_cells(
    root: Path,
    cells: Sequence[ProductionMoEPrefillCell],
) -> tuple[Path, Path]:
    """Require every planned cell and publish deterministic aggregate corpora."""

    root = Path(root)
    paths = []
    for cell in cells:
        cell_paths = _cell_paths(root, cell)
        validate_production_moe_prefill_cell(
            cell_paths.aggregate, cell_paths.timing, cell
        )
        paths.append(cell_paths)
    aggregate = root / "production_moe_prefill.csv"
    timing = root / "production_moe_prefill.timing.csv"
    _combine_csvs_atomic(
        aggregate,
        (path.aggregate for path in paths),
        AGGREGATE_COLUMNS,
    )
    _combine_csvs_atomic(
        timing,
        (path.timing for path in paths),
        TIMING_COLUMNS,
    )
    return aggregate, timing


def _parse_positive_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return values


def _parse_devices(raw: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("devices must be non-negative ordinals")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("devices must not contain duplicates")
    return values


def _filtered_cells(args: argparse.Namespace) -> tuple[ProductionMoEPrefillCell, ...]:
    cells = production_moe_prefill_cells(args.backend)
    if args.case:
        selected = set(args.case)
        cells = tuple(cell for cell in cells if cell.case.evidence_id in selected)
    if args.m:
        selected_m = set(args.m)
        cells = tuple(cell for cell in cells if cell.m in selected_m)
    if args.route_profile:
        selected_profiles = set(args.route_profile)
        cells = tuple(
            cell for cell in cells
            if cell.route_profile in selected_profiles
        )
    if not cells:
        raise ValueError("sweep filters selected no production MoE cells")
    return cells


def main() -> int:
    """CLI entry point for plan, resumable execution, and atomic combination."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "combine"):
        command = subparsers.add_parser(name)
        command.add_argument("--backend", choices=("cuda", "rocm"), required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--case", action="append", default=[])
        command.add_argument("--m", type=_parse_positive_ints)
        command.add_argument(
            "--route-profile",
            action="append",
            choices=MOE_ROUTING_PROFILES,
            default=[],
        )
    run = subparsers.choices["run"]
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--devices", type=_parse_devices, required=True)
    run.add_argument("--maximum-new-cells", type=int)

    args = parser.parse_args()
    cells = _filtered_cells(args)
    output_dir = Path(args.output_dir)
    write_sweep_plan(output_dir / "plan.json", args.backend, cells)
    if args.command == "plan":
        print(f"planned {len(cells)} cells in {output_dir / 'plan.json'}")
        return 0
    if args.command == "run":
        completed = run_missing_accelerator_cells(
            args.binary,
            output_dir,
            cells,
            args.devices,
            maximum_new_cells=args.maximum_new_cells,
        )
        remaining = sum(not _cell_complete(output_dir, cell) for cell in cells)
        print(f"completed {len(completed)} new cells; {remaining} cells remain")
        return 0
    aggregate, timing = combine_production_moe_prefill_cells(output_dir, cells)
    print(f"combined {len(cells)} cells into {aggregate} and {timing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
