"""Resumable all-format dense NativeVNNI prefill sweep transaction.

The CUDA and ROCm C++ trainers own production-kernel execution, byte-exact
certification, and native event timing. This module owns the durable corpus
transaction around those trainers. It expands every production Qwen prefill
geometry across every source format and canonical GPU prefill bucket, runs one
isolated ``(format, geometry, M)`` tournament per accelerator process, validates
the aggregate against every raw event sample, and atomically promotes only a
complete cell.

Source-format aliases remain distinct measurement cells even when preparation
normalizes them to one runtime codebook. The generated runtime overlay may
collapse such aliases only after their independently measured winners agree.
This distinction makes the corpus comprehensive without pretending that an
execution-codebook key can observe source metadata that production discarded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import statistics
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .format_registry import CODEBOOK_PAYLOAD_BYTES, FORMAT_SPECS, NativeVNNIFormatSpec, registry_digest
from .corpus_provenance import (
    TRAINER_PROVENANCE_SCHEMA,
    mapping_digest as _mapping_digest,
    require_release_trainer,
    sha256_file as _sha256_file,
    trainer_provenance,
)
from .prefill_matrix import GPU_PREFILL_M_BUCKETS, gpu_prefill_measurements
from .shape_manifest import NativeVNNIShape, load_shape_manifest


CUDA_AGGREGATE_COLUMNS = (
    "format", "codebook", "shape", "m", "n", "k", "tile", "tile_id",
    "strategy", "requested_k_partitions", "tiles", "min_us", "mean_us", "tops",
    "pct_peak", "gpu", "byte_mismatches", "first_byte_mismatch",
    "correctness_pass", "observed_tile_id", "observed_k_partitions",
    "observed_bk256", "observed_canonical_kpart",
    "canonical_kpart_available",
    "primary_registers_per_thread",
    "primary_local_memory_bytes_per_thread",
    "primary_static_shared_memory_bytes",
    "primary_dynamic_shared_memory_bytes",
    "primary_threads_per_block",
    "primary_max_threads_per_block",
    "primary_max_active_blocks_per_sm",
    "auxiliary_registers_per_thread",
    "auxiliary_local_memory_bytes_per_thread",
    "auxiliary_static_shared_memory_bytes",
    "auxiliary_dynamic_shared_memory_bytes",
    "auxiliary_threads_per_block",
    "auxiliary_max_threads_per_block",
    "auxiliary_max_active_blocks_per_sm",
    "staging_schedule", "observed_staging_schedule",
)

ROCM_AGGREGATE_COLUMNS = (
    "backend", "phase", "format", "codebook", "shape", "category", "m",
    "n", "k", "variant", "min_us", "mean_us", "stddev_us", "gflops",
    "cosine", "correctness_pass", "byte_mismatches_vs_auto",
    "first_byte_mismatch_vs_auto", "is_best", "observed_n_tile",
    "observed_m_tile", "observed_min_blocks", "observed_unroll",
    "observed_full_tiles", "registers_per_thread",
    "local_memory_bytes_per_thread", "static_shared_memory_bytes",
    "max_threads_per_block", "max_active_blocks_per_sm",
)

CUDA_TIMING_COLUMNS = (
    "backend", "phase", "format", "codebook", "shape", "m", "n", "k",
    "tile", "tile_id", "strategy", "requested_k_partitions", "sample_index",
    "timed_replays", "latency_us", "latency_us_hex",
    "staging_schedule",
)

ROCM_TIMING_COLUMNS = (
    "backend", "phase", "format", "codebook", "shape", "m", "n", "k",
    "variant", "sample_index", "timed_replays", "latency_us",
    "latency_us_hex",
)

CUDA_TILE_NAMES = (
    "T64x64_w2x2",
    "T64x128_w2x2",
    "T64x128_w4x2",
    "T64x128_w2x4",
    "T128x128_w4x2",
    "T128x128_w4x4",
)

CUDA_CANDIDATE_IDS = (
    "AUTO",
    *(f"STD:t{tile}:full" for tile in range(len(CUDA_TILE_NAMES))),
    *(f"KPART:t{tile}" for tile in range(len(CUDA_TILE_NAMES))),
    *(f"STD:t{tile}:full:stage{staging}"
      for staging in (1, 2, 3) for tile in (0, 2, 3, 4, 5)),
    *(f"KPART:t{tile}:stage{staging}"
      for staging in (1, 2, 3) for tile in (0, 2, 3, 4, 5)),
    "BK256:full",
)

CUDA_GENERIC_CANDIDATE_IDS = CUDA_CANDIDATE_IDS[:-1]


def _cuda_candidate_is_spill_free(
    execution_codebook: int,
    candidate: str,
) -> bool:
    """Mirror the structurally implemented, compiler-proven no-spill inventory.

    The native trainer re-queries these exact symbols before timing. Any drift
    changes the complete candidate set and fails the immutable plan, rather than
    silently changing a candidate's staging or accepting a partial tournament.
    """

    if not candidate.startswith(("STD:", "KPART:")):
        return True
    tile_id = int(candidate.split(":")[1].removeprefix("t"))
    if ":stage" in candidate:
        staging = int(candidate.rsplit(":stage", 1)[1])
        if staging not in (1, 2, 3):
            return False
        if CODEBOOK_PAYLOAD_BYTES.get(execution_codebook) not in (16, 20) or tile_id == 1:
            return False
        # Exact production-module resource inventory: these asymmetric direct
        # symbols spill eight bytes; canonical publication and other tiles do not.
        if execution_codebook in (5, 7) and tile_id == 2 and staging in (2, 3) and candidate.startswith("STD:"):
            return False
    if execution_codebook in {10, 17}:
        return tile_id not in {1, 4, 5}
    if execution_codebook in {8, 9, 13, 14} and candidate.startswith("STD:"):
        return tile_id not in {1, 4}
    return True


ROCM_CANDIDATE_IDS = (
    "N64/MT16/MB1",
    "N64/MT16/MB2",
    "N64/MT32/MB1",
    "N64/MT32/MB2",
    "N64/MT64/MB1",
    "N64/MT64/MB2",
    "N128/MT16/MB1",
    "N128/MT16/MB2",
    "N128/MT32/MB1",
    "N128/MT32/MB2",
    "N64/MT64/MB1/U0",
    "N64/MT64/MB1/U1",
    "N64/MT64/MB1/U2",
    "N64/MT64/MB1/U4",
    "N64/MT64/MB2/U0",
    "N64/MT64/MB2/U1",
    "N64/MT64/MB2/U2",
    "N64/MT64/MB2/U4",
    "N128/MT32/MB1/U0",
    "N128/MT32/MB1/U1",
    "N128/MT32/MB1/U2",
    "N128/MT32/MB1/U4",
    "N128/MT32/MB2/U0",
    "N128/MT32/MB2/U1",
    "N128/MT32/MB2/U2",
    "N128/MT32/MB2/U4",
    "Auto",
)

ROCM_Q6_FULL_TILE_CANDIDATE_IDS = tuple(
    f"N{n_tile}/MT{m_tile}/MB{min_blocks}/U{unroll}/FULL"
    for n_tile in (64, 128)
    for m_tile in (16, 32)
    for min_blocks in (1, 2)
    for unroll in (0, 1, 2, 4)
)

# gfx906 resource queries prove that these checked-edge Q6_K instantiations
# allocate 20 bytes of scratch per thread. Full-tile versions of the same
# geometry are spill-free, while the checked fallback uses MB1. Candidate
# identity includes the source codebook, so formats whose compiled kernels do
# not spill retain their complete launch surface.
ROCM_Q6_SPILLING_CHECKED_CANDIDATE_IDS = frozenset((
    "N64/MT64/MB2",
    "N128/MT32/MB2",
    *(f"N64/MT64/MB2/U{unroll}" for unroll in (0, 1, 2, 4)),
    *(f"N128/MT32/MB2/U{unroll}" for unroll in (0, 1, 2, 4)),
))

DEFAULT_WARMUP_RUNS = 3
DEFAULT_BENCH_RUNS = 10
SWEEP_PLAN_FILENAME = "production_dense_prefill.plan.json"
SWEEP_PLAN_SCHEMA = "native-vnni-production-dense-prefill-sweep-v2"
CELL_MANIFEST_SCHEMA = "native-vnni-production-dense-prefill-cell-v1"
_UINT64_MAX = (1 << 64) - 1


@dataclass(frozen=True)
class DensePrefillCell:
    """One independently resumable source-format/geometry/M tournament."""

    backend: str
    source_format: NativeVNNIFormatSpec
    shape: NativeVNNIShape
    m: int

    @property
    def identity(self) -> tuple[object, ...]:
        """Return every field that changes prepared data or kernel dispatch."""

        return (
            self.backend,
            self.source_format.label,
            self.source_format.source_codebook_id,
            self.source_format.runtime_codebook(self.backend),
            self.shape.name,
            self.shape.n,
            self.shape.k,
            self.m,
        )

    @property
    def cell_id(self) -> str:
        """Return a bounded content-addressed filename stem."""

        encoded = json.dumps(
            self.identity, separators=(",", ":"), sort_keys=False
        ).encode()
        suffix = hashlib.sha256(encoded).hexdigest()[:20]
        return (
            f"{self.backend}-{self.source_format.label.lower()}-m{self.m}-"
            f"{suffix}"
        )

    def canonical_mapping(self) -> dict[str, object]:
        """Serialize this complete cell into the checked sweep plan."""

        return {
            "backend": self.backend,
            "cell_id": self.cell_id,
            "source_format": self.source_format.label,
            "source_codebook": self.source_format.source_codebook_id,
            "execution_codebook": self.source_format.runtime_codebook(
                self.backend
            ),
            "shape": self.shape.name,
            "model_family": self.shape.model_family,
            "aspect_bucket": self.shape.aspect_bucket.value,
            "m": self.m,
            "n": self.shape.n,
            "k": self.shape.k,
        }


@dataclass(frozen=True)
class DenseCellValidation:
    """Authenticated completion summary for one dense prefill cell."""

    candidate_count: int
    sample_count: int
    winner_id: str
    winner_min_us: float


@dataclass(frozen=True)
class CellPaths:
    """Final and adjacent staging paths for one atomic cell transaction."""

    aggregate: Path
    timing: Path
    log: Path
    manifest: Path
    aggregate_staging: Path
    timing_staging: Path
    log_staging: Path
    manifest_staging: Path


@dataclass(frozen=True)
class BatchPaths:
    """Ephemeral aggregate, timing, and log paths for one native process."""

    aggregate: Path
    timing: Path
    log: Path


def dense_prefill_candidate_ids(
    backend: str,
    source_format: NativeVNNIFormatSpec | None = None,
    *,
    canonical_kpart_available: bool = True,
) -> tuple[str, ...]:
    """Return the complete byte-eligible inventory for one prepared family."""

    normalized = backend.strip().lower()
    if normalized == "cuda":
        candidates = (
            CUDA_CANDIDATE_IDS if source_format is None else
            CUDA_CANDIDATE_IDS
            if source_format.gpu_execution_codebook_id == 0
            else CUDA_GENERIC_CANDIDATE_IDS
        )
        if source_format is not None:
            execution_codebook = source_format.runtime_codebook(normalized)
            candidates = tuple(
                candidate for candidate in candidates
                if _cuda_candidate_is_spill_free(
                    execution_codebook, candidate
                )
            )
        if not canonical_kpart_available:
            candidates = tuple(
                candidate for candidate in candidates
                if not candidate.startswith("KPART:")
            )
        return candidates
    if normalized == "rocm":
        if (
            source_format is not None
            and source_format.runtime_codebook(normalized) == 8
        ):
            checked = tuple(
                candidate for candidate in ROCM_CANDIDATE_IDS[:-1]
                if candidate not in ROCM_Q6_SPILLING_CHECKED_CANDIDATE_IDS
            )
            return (
                checked
                + ROCM_Q6_FULL_TILE_CANDIDATE_IDS
                + ROCM_CANDIDATE_IDS[-1:]
            )
        return ROCM_CANDIDATE_IDS
    raise ValueError(f"dense GPU prefill sweep does not support {backend!r}")


def production_dense_prefill_cells(
    backend: str,
) -> tuple[DensePrefillCell, ...]:
    """Expand all formats, exact Qwen prefill geometries, and GPU M buckets."""

    normalized = backend.strip().lower()
    dense_prefill_candidate_ids(normalized)
    return tuple(
        DensePrefillCell(normalized, source_format, measurement.shape, m)
        for source_format in FORMAT_SPECS
        for measurement in gpu_prefill_measurements()
        for m in measurement.m_values
    )


def _cell_sort_key(cell: DensePrefillCell) -> tuple[object, ...]:
    return cell.identity


def _cell_paths(root: Path, cell: DensePrefillCell) -> CellPaths:
    stem = Path(root) / "cells" / cell.cell_id
    return CellPaths(
        aggregate=stem.with_suffix(".csv"),
        timing=stem.with_suffix(".timing.csv"),
        log=stem.with_suffix(".log"),
        manifest=stem.with_suffix(".manifest.json"),
        aggregate_staging=stem.with_suffix(".csv.inprogress"),
        timing_staging=stem.with_suffix(".timing.csv.inprogress"),
        log_staging=stem.with_suffix(".log.inprogress"),
        manifest_staging=stem.with_suffix(".manifest.json.inprogress"),
    )


def _batch_group_key(cell: DensePrefillCell) -> tuple[object, ...]:
    """Return the state that can share prepared weights in one process.

    The native trainers accept several M buckets while retaining one source
    format and one matrix geometry. Keeping those fields fixed lets the CUDA
    and ROCm harnesses prepare/upload the weight once without changing any
    candidate identity or timing semantics.
    """

    return (
        cell.backend,
        cell.source_format.label,
        cell.source_format.source_codebook_id,
        cell.source_format.runtime_codebook(cell.backend),
        cell.shape.name,
        cell.shape.n,
        cell.shape.k,
    )


def _validated_batch_cells(
    cells: Sequence[DensePrefillCell],
) -> tuple[DensePrefillCell, ...]:
    """Validate and order one same-format/same-geometry M batch."""

    ordered = tuple(sorted(cells, key=lambda cell: cell.m))
    if not ordered:
        raise ValueError("dense prefill process batch cannot be empty")
    if any(
        _batch_group_key(cell) != _batch_group_key(ordered[0])
        for cell in ordered
    ):
        raise ValueError(
            "dense prefill process batch must retain one format and geometry"
        )
    if len({cell.m for cell in ordered}) != len(ordered):
        raise ValueError("dense prefill process batch contains duplicate M")
    return ordered


def _batch_paths(
    root: Path,
    cells: Sequence[DensePrefillCell],
) -> BatchPaths:
    """Return collision-resistant staging paths for one process batch."""

    ordered = _validated_batch_cells(cells)
    digest = hashlib.sha256(json.dumps(
        [cell.identity for cell in ordered],
        separators=(",", ":"),
    ).encode()).hexdigest()[:20]
    first = ordered[0]
    stem = (
        Path(root) / "batches" /
        f"{first.backend}-{first.source_format.label.lower()}-{digest}"
    )
    return BatchPaths(
        aggregate=stem.with_suffix(".csv.inprogress"),
        timing=stem.with_suffix(".timing.csv.inprogress"),
        log=stem.with_suffix(".log.inprogress"),
    )


def production_dense_prefill_cell_paths(
    root: Path,
    cell: DensePrefillCell,
) -> CellPaths:
    """Expose stable paths for diagnostics and turnkey status reporting."""

    return _cell_paths(Path(root), cell)


def _require_exact_header(
    path: Path,
    actual: Sequence[str] | None,
    expected: Sequence[str],
) -> None:
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


def _cuda_candidate_id(row: Mapping[str, str], context: str) -> str:
    """Reconstruct one CUDA candidate ID from launchable policy fields."""

    strategy = row["strategy"].strip().upper()
    staging = int(row["staging_schedule"])
    if staging not in (0, 1, 2, 3):
        raise ValueError(f"{context}: invalid CUDA staging schedule")
    suffix = f":stage{staging}" if staging else ""
    tile_id = int(row["tile_id"])
    requested_k_partitions = int(row["requested_k_partitions"])
    tile = row["tile"].strip()
    if strategy == "AUTO":
        if tile_id != -1 or requested_k_partitions != 0 or tile != "AUTO" or staging:
            raise ValueError(f"{context}: malformed AUTO candidate")
        return "AUTO"
    if strategy == "STD":
        if tile_id not in range(len(CUDA_TILE_NAMES)):
            raise ValueError(f"{context}: invalid CUDA tile ID")
        if tile != CUDA_TILE_NAMES[tile_id] or requested_k_partitions != 1:
            raise ValueError(f"{context}: CUDA tile metadata disagrees")
        return f"{strategy}:t{tile_id}:full{suffix}"
    if strategy == "KPART":
        if tile_id not in range(len(CUDA_TILE_NAMES)):
            raise ValueError(f"{context}: invalid CUDA KPART tile ID")
        if tile != CUDA_TILE_NAMES[tile_id] or requested_k_partitions != 0:
            raise ValueError(f"{context}: malformed canonical KPART candidate")
        return f"KPART:t{tile_id}{suffix}"
    if strategy == "BK256":
        if tile_id != -2 or tile != "BK256_128x128" or staging:
            raise ValueError(f"{context}: malformed BK256 candidate")
        if requested_k_partitions != 1:
            raise ValueError(f"{context}: BK256 must own the complete K walk")
        return f"{strategy}:full"
    raise ValueError(f"{context}: unsupported CUDA strategy {strategy!r}")


def _candidate_id(
    backend: str,
    row: Mapping[str, str],
    context: str,
) -> str:
    if backend == "cuda":
        return _cuda_candidate_id(row, context)
    return row["variant"].strip()


def _aggregate_columns(backend: str) -> tuple[str, ...]:
    return (
        CUDA_AGGREGATE_COLUMNS if backend == "cuda"
        else ROCM_AGGREGATE_COLUMNS
    )


def _timing_columns(backend: str) -> tuple[str, ...]:
    return CUDA_TIMING_COLUMNS if backend == "cuda" else ROCM_TIMING_COLUMNS


def _validate_common_identity(
    row: Mapping[str, str],
    cell: DensePrefillCell,
    context: str,
) -> None:
    if row["format"] != cell.source_format.label:
        raise ValueError(f"{context}: source format disagrees with plan")
    if int(row["codebook"]) != cell.source_format.source_codebook_id:
        raise ValueError(f"{context}: source codebook disagrees with registry")
    if row["shape"] != cell.shape.name:
        raise ValueError(f"{context}: shape disagrees with plan")
    if (int(row["m"]), int(row["n"]), int(row["k"])) != (
        cell.m,
        cell.shape.n,
        cell.shape.k,
    ):
        raise ValueError(f"{context}: dimensions disagree with plan")


def _read_aggregate_rows(
    path: Path,
    cell: DensePrefillCell,
) -> dict[str, dict[str, str]]:
    """Authenticate every candidate result and its byte certificate."""

    allowed = set(dense_prefill_candidate_ids(
        cell.backend, cell.source_format,
        canonical_kpart_available=True,
    ))
    indexed: dict[str, dict[str, str]] = {}
    canonical_kpart_availability: set[bool] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_exact_header(Path(path), reader.fieldnames, _aggregate_columns(
            cell.backend
        ))
        for line, row in enumerate(reader, start=2):
            context = f"{path}:{line}"
            _validate_common_identity(row, cell, context)
            if cell.backend == "rocm" and (
                row["backend"] != "rocm" or row["phase"] != "prefill"
            ):
                raise ValueError(f"{context}: wrong ROCm backend or phase")
            candidate = _candidate_id(cell.backend, row, context)
            if candidate not in allowed:
                raise ValueError(f"{context}: unknown candidate {candidate!r}")
            if candidate in indexed:
                raise ValueError(f"{context}: duplicate candidate {candidate!r}")

            mismatch_field = (
                "byte_mismatches" if cell.backend == "cuda"
                else "byte_mismatches_vs_auto"
            )
            first_field = (
                "first_byte_mismatch" if cell.backend == "cuda"
                else "first_byte_mismatch_vs_auto"
            )
            if int(row[mismatch_field]) != 0 or int(row[first_field]) != _UINT64_MAX:
                raise ValueError(f"{context}: candidate failed byte equality")
            if not _explicit_bool("correctness_pass", row["correctness_pass"]):
                raise ValueError(f"{context}: candidate failed correctness gate")
            for field in ("min_us", "mean_us"):
                value = float(row[field])
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"{context}: invalid {field}")
            if cell.backend == "cuda":
                if int(row["observed_staging_schedule"]) != int(row["staging_schedule"]):
                    raise ValueError(f"{context}: forced CUDA staging did not execute")
                canonical_kpart_availability.add(_explicit_bool(
                    "canonical_kpart_available",
                    row["canonical_kpart_available"],
                ))
                for field in ("tops", "pct_peak"):
                    if not math.isfinite(float(row[field])) or float(row[field]) <= 0:
                        raise ValueError(f"{context}: invalid {field}")
                if int(row["gpu"]) != 0:
                    raise ValueError(f"{context}: masked CUDA ordinal must be zero")
                observed_tile = int(row["observed_tile_id"])
                observed_k_partitions = int(row["observed_k_partitions"])
                observed_bk256 = int(row["observed_bk256"])
                observed_canonical_kpart = int(
                    row["observed_canonical_kpart"]
                )
                if observed_canonical_kpart not in {0, 1}:
                    raise ValueError(f"{context}: invalid CUDA K publication marker")
                if observed_bk256 not in {0, 1}:
                    raise ValueError(f"{context}: invalid BK256 route marker")
                if observed_bk256:
                    if cell.source_format.gpu_execution_codebook_id != 0 or (
                        observed_tile not in {-3, -2}
                    ):
                        raise ValueError(f"{context}: impossible BK256 observation")
                elif observed_tile not in range(len(CUDA_TILE_NAMES)):
                    raise ValueError(f"{context}: invalid observed CUDA tile")
                if candidate.startswith("KPART:"):
                    if (
                        observed_canonical_kpart != 1
                        or observed_k_partitions <= 1
                        or observed_bk256 != 0
                        or observed_tile != int(candidate.split(":")[1][1:])
                    ):
                        raise ValueError(
                            f"{context}: canonical CUDA KPART did not execute"
                        )
                elif (
                    observed_canonical_kpart != 0
                    or observed_k_partitions != 1
                ):
                    raise ValueError(
                        f"{context}: noncanonical CUDA reduction entered evidence"
                    )
                if candidate.startswith("STD:") and (
                    observed_tile != int(candidate.split(":")[1][1:])
                    or observed_bk256 != 0
                ):
                    raise ValueError(
                        f"{context}: forced CUDA tile did not execute"
                    )
                if candidate == "BK256:full" and observed_bk256 != 1:
                    raise ValueError(f"{context}: forced BK256 route did not execute")
                primary_registers = int(row["primary_registers_per_thread"])
                primary_local = int(
                    row["primary_local_memory_bytes_per_thread"]
                )
                primary_static_shared = int(
                    row["primary_static_shared_memory_bytes"]
                )
                primary_dynamic_shared = int(
                    row["primary_dynamic_shared_memory_bytes"]
                )
                primary_threads = int(row["primary_threads_per_block"])
                primary_max_threads = int(
                    row["primary_max_threads_per_block"]
                )
                primary_active_blocks = int(
                    row["primary_max_active_blocks_per_sm"]
                )
                if (
                    primary_registers <= 0
                    or primary_local != 0
                    or primary_static_shared < 0
                    or primary_dynamic_shared < 0
                    or primary_threads <= 0
                    or primary_max_threads < primary_threads
                    or primary_active_blocks <= 0
                ):
                    raise ValueError(
                        f"{context}: ineligible CUDA primary resources"
                    )

                auxiliary_registers = int(
                    row["auxiliary_registers_per_thread"]
                )
                auxiliary_local = int(
                    row["auxiliary_local_memory_bytes_per_thread"]
                )
                auxiliary_static_shared = int(
                    row["auxiliary_static_shared_memory_bytes"]
                )
                auxiliary_dynamic_shared = int(
                    row["auxiliary_dynamic_shared_memory_bytes"]
                )
                auxiliary_threads = int(row["auxiliary_threads_per_block"])
                auxiliary_max_threads = int(
                    row["auxiliary_max_threads_per_block"]
                )
                auxiliary_active_blocks = int(
                    row["auxiliary_max_active_blocks_per_sm"]
                )
                auxiliary_values = (
                    auxiliary_registers,
                    auxiliary_local,
                    auxiliary_static_shared,
                    auxiliary_dynamic_shared,
                    auxiliary_threads,
                    auxiliary_max_threads,
                    auxiliary_active_blocks,
                )
                if auxiliary_local != 0 or (
                    auxiliary_threads == 0 and any(auxiliary_values)
                ) or (
                    auxiliary_threads > 0
                    and (
                        auxiliary_registers <= 0
                        or auxiliary_static_shared < 0
                        or auxiliary_dynamic_shared < 0
                        or auxiliary_max_threads < auxiliary_threads
                        or auxiliary_active_blocks <= 0
                    )
                ):
                    raise ValueError(
                        f"{context}: ineligible CUDA auxiliary resources"
                    )
                if candidate.startswith("KPART:") != (auxiliary_threads > 0):
                    raise ValueError(
                        f"{context}: CUDA reducer resource identity disagrees"
                    )
            else:
                if float(row["stddev_us"]) < 0.0 or float(row["gflops"]) <= 0.0:
                    raise ValueError(f"{context}: invalid ROCm timing statistics")
                if not math.isclose(float(row["cosine"]), 1.0):
                    raise ValueError(f"{context}: byte-exact ROCm row lacks unity cosine")
                observed = (
                    int(row["observed_n_tile"]),
                    int(row["observed_m_tile"]),
                    int(row["observed_min_blocks"]),
                    int(row["observed_unroll"]),
                    int(row["observed_full_tiles"]),
                )
                if observed[0] not in {64, 128} or observed[1] not in {
                    16, 32, 64
                } or observed[2] not in {1, 2, 3} or observed[3] not in {
                    0, 1, 2, 4
                } or observed[4] not in {0, 1}:
                    raise ValueError(
                        f"{context}: invalid observed ROCm launch tuple {observed}"
                    )
                if (
                    int(row["registers_per_thread"]) <= 0
                    or int(row["local_memory_bytes_per_thread"]) != 0
                    or int(row["static_shared_memory_bytes"]) < 0
                    or int(row["max_threads_per_block"]) < 256
                    or int(row["max_active_blocks_per_sm"]) <= 0
                ):
                    raise ValueError(
                        f"{context}: ineligible ROCm compiler resources"
                    )
                if candidate != "Auto":
                    components = candidate.split("/")
                    expected_observed = (
                        int(components[0].removeprefix("N")),
                        int(components[1].removeprefix("MT")),
                        int(components[2].removeprefix("MB")),
                        int(components[3].removeprefix("U"))
                        if len(components) >= 4 and components[3].startswith("U")
                        else 4,
                        int(components[-1] == "FULL"),
                    )
                    if observed != expected_observed:
                        raise ValueError(
                            f"{context}: forced ROCm launch did not execute; "
                            f"expected={expected_observed}, observed={observed}"
                        )
            indexed[candidate] = dict(row)

    if cell.backend == "cuda":
        if len(canonical_kpart_availability) != 1:
            raise ValueError(
                f"{path}: inconsistent canonical KPART availability marker"
            )
        canonical_kpart_available = next(iter(
            canonical_kpart_availability
        ))
    else:
        canonical_kpart_available = False
    expected = set(dense_prefill_candidate_ids(
        cell.backend,
        cell.source_format,
        canonical_kpart_available=canonical_kpart_available,
    ))
    if set(indexed) != expected:
        raise ValueError(
            f"{path}: candidate matrix mismatch; "
            f"missing={sorted(expected - set(indexed))}, "
            f"unexpected={sorted(set(indexed) - expected)}"
        )
    return indexed


def _read_timing_rows(
    path: Path,
    cell: DensePrefillCell,
    bench_runs: int,
    expected_candidates: Iterable[str] | None = None,
) -> dict[str, tuple[float, ...]]:
    """Authenticate contiguous native-event samples for every candidate."""

    expected = set(
        dense_prefill_candidate_ids(cell.backend, cell.source_format)
        if expected_candidates is None else expected_candidates
    )
    samples: dict[str, list[float]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_exact_header(Path(path), reader.fieldnames, _timing_columns(
            cell.backend
        ))
        for line, row in enumerate(reader, start=2):
            context = f"{path}:{line}"
            if row["backend"] != cell.backend or row["phase"] != "prefill":
                raise ValueError(f"{context}: wrong backend or phase")
            _validate_common_identity(row, cell, context)
            candidate = _candidate_id(cell.backend, row, context)
            if candidate not in expected:
                raise ValueError(f"{context}: unknown candidate {candidate!r}")
            candidate_samples = samples.setdefault(candidate, [])
            if int(row["sample_index"]) != len(candidate_samples):
                raise ValueError(f"{context}: non-contiguous sample index")
            if int(row["timed_replays"]) != 1:
                raise ValueError(f"{context}: dense event sample must time one replay")
            readable = float(row["latency_us"])
            exact = float.fromhex(row["latency_us_hex"])
            if not math.isfinite(exact) or exact <= 0.0:
                raise ValueError(f"{context}: invalid event latency")
            if not math.isclose(readable, exact, rel_tol=0.0, abs_tol=5.1e-10):
                raise ValueError(f"{context}: readable/exact timing disagree")
            candidate_samples.append(exact)

    if set(samples) != expected:
        raise ValueError(f"{path}: raw timing candidate matrix is incomplete")
    result = {candidate: tuple(values) for candidate, values in samples.items()}
    for candidate, values in result.items():
        if len(values) != bench_runs:
            raise ValueError(
                f"{path}: {candidate} has {len(values)} samples, expected "
                f"{bench_runs}"
            )
    return result


def validate_production_dense_prefill_cell(
    aggregate_path: Path,
    timing_path: Path,
    cell: DensePrefillCell,
    *,
    bench_runs: int = DEFAULT_BENCH_RUNS,
) -> DenseCellValidation:
    """Prove one aggregate/sidecar pair complete before reuse or promotion."""

    if bench_runs <= 0:
        raise ValueError("bench_runs must be positive")
    aggregate = _read_aggregate_rows(Path(aggregate_path), cell)
    timing = _read_timing_rows(
        Path(timing_path), cell, bench_runs, aggregate.keys()
    )
    for candidate, row in aggregate.items():
        values = timing[candidate]
        expected_min = min(values)
        expected_mean = statistics.fmean(values)
        if not math.isclose(
            float(row["min_us"]), expected_min, rel_tol=0.0, abs_tol=5.1e-4
        ):
            raise ValueError(f"{candidate}: aggregate minimum disagrees with sidecar")
        if not math.isclose(
            float(row["mean_us"]), expected_mean, rel_tol=0.0, abs_tol=5.1e-4
        ):
            raise ValueError(f"{candidate}: aggregate mean disagrees with sidecar")
        if cell.backend == "rocm":
            expected_stddev = math.sqrt(statistics.fmean(
                (value - expected_mean) ** 2 for value in values
            ))
            if not math.isclose(
                float(row["stddev_us"]), expected_stddev,
                rel_tol=0.0, abs_tol=5.1e-4,
            ):
                raise ValueError(
                    f"{candidate}: aggregate stddev disagrees with sidecar"
                )

    winners = sorted(
        aggregate,
        key=lambda candidate: (float(aggregate[candidate]["min_us"]), candidate),
    )
    if cell.backend == "rocm":
        marked = [
            candidate for candidate, row in aggregate.items()
            if _explicit_bool("is_best", row["is_best"])
        ]
        if len(marked) != 1 or marked[0] != winners[0]:
            raise ValueError(
                f"ROCm winner marker disagrees with measured minimum: {marked}"
            )
    winner = winners[0]
    return DenseCellValidation(
        candidate_count=len(aggregate),
        sample_count=sum(len(values) for values in timing.values()),
        winner_id=winner,
        winner_min_us=float(aggregate[winner]["min_us"]),
    )


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".inprogress")
    with staging.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, path)


def _sweep_plan_payload(
    backend: str,
    cells: Sequence[DensePrefillCell],
    trainer: Mapping[str, object],
    warmup_runs: int,
    bench_runs: int,
) -> dict[str, object]:
    """Construct the complete content-addressed sweep-plan payload."""

    normalized = backend.strip().lower()
    if warmup_runs <= 0 or bench_runs <= 0:
        raise ValueError("timing cardinalities must be positive")
    if any(cell.backend != normalized for cell in cells):
        raise ValueError("sweep plan contains a cell for another backend")
    if trainer.get("backend") != normalized:
        raise ValueError("trainer provenance belongs to another backend")
    body: dict[str, object] = {
        "schema_version": SWEEP_PLAN_SCHEMA,
        "backend": normalized,
        "format_registry_digest": registry_digest(),
        "shape_manifest_digest": load_shape_manifest().digest(),
        "gpu_m_buckets": list(GPU_PREFILL_M_BUCKETS),
        "candidate_ids_by_source_format": {
            source_format.label: list(dense_prefill_candidate_ids(
                normalized, source_format
            ))
            for source_format in FORMAT_SPECS
        },
        "warmup_runs": warmup_runs,
        "bench_runs": bench_runs,
        "trainer_provenance": dict(trainer),
        "trainer_provenance_digest": _mapping_digest(trainer),
        "cells": [cell.canonical_mapping() for cell in cells],
    }
    body["plan_digest"] = _mapping_digest(body)
    return body


def _read_sweep_plan(path: Path) -> dict[str, object]:
    """Read and authenticate an existing immutable sweep plan."""

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: sweep plan must be a JSON object")
    if payload.get("schema_version") != SWEEP_PLAN_SCHEMA:
        raise ValueError(f"{path}: unsupported sweep-plan schema")
    recorded_digest = payload.get("plan_digest")
    unsigned = dict(payload)
    unsigned.pop("plan_digest", None)
    if recorded_digest != _mapping_digest(unsigned):
        raise ValueError(f"{path}: sweep-plan digest mismatch")
    trainer = payload.get("trainer_provenance")
    if not isinstance(trainer, dict) or (
        payload.get("trainer_provenance_digest") != _mapping_digest(trainer)
    ):
        raise ValueError(f"{path}: trainer provenance digest mismatch")
    return payload


def write_sweep_plan(
    path: Path,
    backend: str,
    cells: Sequence[DensePrefillCell],
    trainer: Mapping[str, object],
    *,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    bench_runs: int = DEFAULT_BENCH_RUNS,
) -> dict[str, object]:
    """Create one immutable plan, or prove an existing plan is identical."""

    path = Path(path)
    expected = _sweep_plan_payload(
        backend, cells, trainer, warmup_runs, bench_runs
    )
    if path.exists():
        existing = _read_sweep_plan(path)
        if existing != expected:
            raise ValueError(
                f"{path}: immutable sweep plan disagrees with this run; "
                "use a new corpus directory"
            )
        return existing
    _write_json_atomic(path, expected)
    return expected


def load_sweep_plan(
    path: Path,
    backend: str,
    cells: Sequence[DensePrefillCell],
    *,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    bench_runs: int = DEFAULT_BENCH_RUNS,
) -> dict[str, object]:
    """Authenticate a plan without mutating it for status or installation."""

    path = Path(path)
    existing = _read_sweep_plan(path)
    trainer = existing["trainer_provenance"]
    assert isinstance(trainer, dict)
    expected = _sweep_plan_payload(
        backend, cells, trainer, warmup_runs, bench_runs
    )
    if existing != expected:
        raise ValueError(
            f"{path}: requested matrix or timing cardinality disagrees with "
            "the immutable sweep plan"
        )
    return existing


def load_planned_dense_prefill_cells(
    path: Path,
    backend: str,
) -> tuple[tuple[DensePrefillCell, ...], dict[str, object]]:
    """Reconstruct the exact matrix named by an authenticated sweep plan."""

    normalized = backend.strip().lower()
    existing = _read_sweep_plan(Path(path))
    if existing.get("backend") != normalized:
        raise ValueError(f"{path}: sweep plan belongs to another backend")
    planned_rows = existing.get("cells")
    if not isinstance(planned_rows, list) or not planned_rows:
        raise ValueError(f"{path}: sweep plan has no cells")

    available = {
        cell.cell_id: cell
        for cell in production_dense_prefill_cells(normalized)
    }
    selected = []
    seen: set[str] = set()
    for index, row in enumerate(planned_rows):
        if not isinstance(row, dict) or not isinstance(row.get("cell_id"), str):
            raise ValueError(f"{path}: malformed cell row {index}")
        cell_id = row["cell_id"]
        if cell_id in seen:
            raise ValueError(f"{path}: duplicate planned cell {cell_id}")
        try:
            cell = available[cell_id]
        except KeyError as error:
            raise ValueError(
                f"{path}: planned cell {cell_id} is absent from current matrix"
            ) from error
        if row != cell.canonical_mapping():
            raise ValueError(f"{path}: planned cell {cell_id} changed identity")
        seen.add(cell_id)
        selected.append(cell)

    cells = tuple(selected)
    validated = load_sweep_plan(
        path,
        normalized,
        cells,
        warmup_runs=int(existing["warmup_runs"]),
        bench_runs=int(existing["bench_runs"]),
    )
    return cells, validated


def _cell_manifest_payload(
    paths: CellPaths,
    cell: DensePrefillCell,
    plan: Mapping[str, object],
    native_process: Mapping[str, object],
) -> dict[str, object]:
    """Bind one promoted cell to its producer and exact artifact bytes."""

    body: dict[str, object] = {
        "schema_version": CELL_MANIFEST_SCHEMA,
        "cell": cell.canonical_mapping(),
        "plan_digest": plan["plan_digest"],
        "trainer_provenance_digest": plan["trainer_provenance_digest"],
        "warmup_runs": plan["warmup_runs"],
        "bench_runs": plan["bench_runs"],
        "native_process": dict(native_process),
        "artifacts": {
            "aggregate": {
                "filename": paths.aggregate.name,
                "sha256": _sha256_file(paths.aggregate_staging),
            },
            "timing": {
                "filename": paths.timing.name,
                "sha256": _sha256_file(paths.timing_staging),
            },
            "log": {
                "filename": paths.log.name,
                "sha256": _sha256_file(paths.log_staging),
            },
        },
    }
    body["manifest_digest"] = _mapping_digest(body)
    return body


def _stage_cell_manifest(
    paths: CellPaths,
    cell: DensePrefillCell,
    plan: Mapping[str, object],
    native_process: Mapping[str, object],
) -> None:
    """Durably stage the commit marker after all cell artifacts validate."""

    payload = _cell_manifest_payload(paths, cell, plan, native_process)
    paths.manifest_staging.parent.mkdir(parents=True, exist_ok=True)
    with paths.manifest_staging.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_cell_manifest(path: Path) -> dict[str, object]:
    """Read and authenticate one cell commit marker."""

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: cell manifest must be a JSON object")
    if payload.get("schema_version") != CELL_MANIFEST_SCHEMA:
        raise ValueError(f"{path}: unsupported cell-manifest schema")
    recorded_digest = payload.get("manifest_digest")
    unsigned = dict(payload)
    unsigned.pop("manifest_digest", None)
    if recorded_digest != _mapping_digest(unsigned):
        raise ValueError(f"{path}: cell-manifest digest mismatch")
    return payload


def validate_promoted_production_dense_prefill_cell(
    root: Path,
    cell: DensePrefillCell,
    plan: Mapping[str, object],
) -> DenseCellValidation:
    """Prove a committed cell belongs to one exact trainer/plan closure."""

    paths = _cell_paths(Path(root), cell)
    manifest = _read_cell_manifest(paths.manifest)
    if manifest.get("cell") != cell.canonical_mapping():
        raise ValueError(f"{paths.manifest}: cell identity mismatch")
    for field in (
        "plan_digest",
        "trainer_provenance_digest",
        "warmup_runs",
        "bench_runs",
    ):
        if manifest.get(field) != plan.get(field):
            raise ValueError(f"{paths.manifest}: {field} mismatch")

    native_process = manifest.get("native_process")
    if not isinstance(native_process, dict):
        raise ValueError(f"{paths.manifest}: missing native-process identity")
    cell_ids = native_process.get("cell_ids")
    if (
        native_process.get("backend") != cell.backend
        or not isinstance(native_process.get("invocation_id"), str)
        or not native_process["invocation_id"]
        or not isinstance(native_process.get("device_ordinal"), int)
        or native_process["device_ordinal"] < 0
        or not isinstance(cell_ids, list)
        or cell.cell_id not in cell_ids
    ):
        raise ValueError(f"{paths.manifest}: invalid native-process identity")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "aggregate", "timing", "log"
    }:
        raise ValueError(f"{paths.manifest}: incomplete artifact inventory")
    for role, artifact_path in (
        ("aggregate", paths.aggregate),
        ("timing", paths.timing),
        ("log", paths.log),
    ):
        record = artifacts[role]
        if not isinstance(record, dict) or (
            record.get("filename") != artifact_path.name
            or record.get("sha256") != _sha256_file(artifact_path)
        ):
            raise ValueError(
                f"{paths.manifest}: {role} artifact digest mismatch"
            )
    return validate_production_dense_prefill_cell(
        paths.aggregate,
        paths.timing,
        cell,
        bench_runs=int(plan["bench_runs"]),
    )


def _cell_complete(
    root: Path,
    cell: DensePrefillCell,
    plan: Mapping[str, object],
) -> bool:
    try:
        validate_promoted_production_dense_prefill_cell(root, cell, plan)
    except (OSError, ValueError):
        return False
    return True


_BACKEND_TESTS = {
    "cuda": "CUDANativeVNNIGemmPerf.TileSweep_AllStrategies",
    "rocm": "NativeVNNISweepTest.TrainerCsv_CodebookTagged",
}


def _cell_environment(
    cell: DensePrefillCell,
    paths: CellPaths,
    device: int,
    warmup_runs: int,
    bench_runs: int,
) -> dict[str, str]:
    """Build one process-isolated trainer environment for a physical GPU."""

    environment = os.environ.copy()
    environment["GTEST_COLOR"] = "no"
    if cell.backend == "cuda":
        environment.pop("ROCR_VISIBLE_DEVICES", None)
        environment.pop("HIP_VISIBLE_DEVICES", None)
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(device),
            "LLAMINAR_TILE_SWEEP_FORMAT": cell.source_format.label,
            "LLAMINAR_TILE_SWEEP_SHAPES": cell.shape.name,
            "LLAMINAR_TILE_SWEEP_PREFILL_M": str(cell.m),
            "LLAMINAR_TILE_SWEEP_STRATEGIES": (
                "auto,std,kpart,bk256"
            ),
            "LLAMINAR_TILE_SWEEP_WARMUP": str(warmup_runs),
            "LLAMINAR_TILE_SWEEP_BENCH": str(bench_runs),
            "LLAMINAR_TILE_SWEEP_CSV": str(paths.aggregate_staging),
            "LLAMINAR_TILE_SWEEP_TIMING_CSV": str(paths.timing_staging),
        })
    else:
        # HIP visibility is the sole ordinal authority. Setting both HIP and
        # ROCR masks renumbers the selected physical card through ROCR and then
        # applies the original ordinal again through HIP, hiding cards 1..N.
        environment.pop("ROCR_VISIBLE_DEVICES", None)
        environment.update({
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": str(device),
            "LLAMINAR_ROCM_NVNNI_SWEEP_FORMATS": cell.source_format.label,
            "LLAMINAR_ROCM_NVNNI_SWEEP_SHAPES": cell.shape.name,
            "LLAMINAR_ROCM_NVNNI_SWEEP_M": str(cell.m),
            "LLAMINAR_ROCM_NVNNI_SWEEP_MAX_CASES": "1",
            "LLAMINAR_ROCM_NVNNI_SWEEP_WARMUP": str(warmup_runs),
            "LLAMINAR_ROCM_NVNNI_SWEEP_BENCH": str(bench_runs),
            "LLAMINAR_ROCM_NVNNI_SWEEP_CSV": str(paths.aggregate_staging),
            "LLAMINAR_ROCM_NVNNI_SWEEP_TIMING_CSV": str(
                paths.timing_staging
            ),
        })
    return environment


def _batch_environment(
    cells: Sequence[DensePrefillCell],
    paths: BatchPaths,
    device: int,
    warmup_runs: int,
    bench_runs: int,
) -> dict[str, str]:
    """Build one native-process environment for several exact M cells.

    Only M varies inside a batch. The C++ harness therefore retains the same
    prepared weight while constructing a fresh persistent execution object for
    each bucket. Candidate timing remains one native event sample per launch;
    batching removes setup work but does not combine or amortize timed kernels.
    """

    ordered = _validated_batch_cells(cells)
    first = ordered[0]
    surrogate_paths = CellPaths(
        aggregate=paths.aggregate,
        timing=paths.timing,
        log=paths.log,
        manifest=paths.log.with_suffix(".manifest.json"),
        aggregate_staging=paths.aggregate,
        timing_staging=paths.timing,
        log_staging=paths.log,
        manifest_staging=paths.log.with_suffix(".manifest.json.inprogress"),
    )
    environment = _cell_environment(
        first,
        surrogate_paths,
        device,
        warmup_runs,
        bench_runs,
    )
    m_values = ",".join(str(cell.m) for cell in ordered)
    if first.backend == "cuda":
        environment["LLAMINAR_TILE_SWEEP_PREFILL_M"] = m_values
    else:
        environment["LLAMINAR_ROCM_NVNNI_SWEEP_M"] = m_values
        environment["LLAMINAR_ROCM_NVNNI_SWEEP_MAX_CASES"] = str(len(ordered))
    return environment


def _write_csv_rows(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    """Durably write one already-staged cell CSV before authentication."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _split_batch_csv(
    source: Path,
    root: Path,
    cells: Sequence[DensePrefillCell],
    *,
    timing: bool,
) -> None:
    """Split native multi-M output back into independently resumable cells.

    Every row is matched against the immutable cell identity before it reaches
    a per-cell staging file. Full candidate/sample validation happens after
    both CSVs have been split, so no final corpus path can be published from a
    malformed or incomplete batch.
    """

    ordered = _validated_batch_cells(cells)
    columns = (
        _timing_columns(ordered[0].backend)
        if timing else _aggregate_columns(ordered[0].backend)
    )
    cells_by_m = {cell.m: cell for cell in ordered}
    rows_by_m: dict[int, list[dict[str, str]]] = {
        cell.m: [] for cell in ordered
    }
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_exact_header(source, reader.fieldnames, columns)
        for line, row in enumerate(reader, start=2):
            try:
                cell = cells_by_m[int(row["m"])]
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"{source}:{line}: row does not belong to the process batch"
                ) from error
            _validate_common_identity(row, cell, f"{source}:{line}")
            rows_by_m[cell.m].append(dict(row))

    for cell in ordered:
        if not rows_by_m[cell.m]:
            raise ValueError(f"{source}: M={cell.m} produced no rows")
        paths = _cell_paths(root, cell)
        destination = paths.timing_staging if timing else paths.aggregate_staging
        _write_csv_rows(destination, columns, rows_by_m[cell.m])


def _write_batch_logs(
    source: Path,
    root: Path,
    cells: Sequence[DensePrefillCell],
) -> None:
    """Attach the exact native-process log to every promoted cell."""

    payload = source.read_bytes()
    for cell in _validated_batch_cells(cells):
        destination = _cell_paths(root, cell).log_staging
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def _run_accelerator_cell(
    binary: Path,
    root: Path,
    cell: DensePrefillCell,
    device: int,
    warmup_runs: int,
    bench_runs: int,
    plan: Mapping[str, object],
) -> DenseCellValidation:
    """Launch, authenticate, and atomically promote one isolated cell."""

    paths = _cell_paths(root, cell)
    paths.aggregate.parent.mkdir(parents=True, exist_ok=True)
    for staging in (
        paths.aggregate_staging,
        paths.timing_staging,
        paths.log_staging,
        paths.manifest_staging,
    ):
        staging.unlink(missing_ok=True)
    native_process = {
        "invocation_id": str(uuid.uuid4()),
        "backend": cell.backend,
        "device_ordinal": device,
        "cell_ids": [cell.cell_id],
    }
    command = (
        str(binary),
        f"--gtest_filter={_BACKEND_TESTS[cell.backend]}",
    )
    environment = _cell_environment(
        cell, paths, device, warmup_runs, bench_runs
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
    validation = validate_production_dense_prefill_cell(
        paths.aggregate_staging,
        paths.timing_staging,
        cell,
        bench_runs=bench_runs,
    )
    _stage_cell_manifest(paths, cell, plan, native_process)
    os.replace(paths.aggregate_staging, paths.aggregate)
    os.replace(paths.timing_staging, paths.timing)
    os.replace(paths.log_staging, paths.log)
    os.replace(paths.manifest_staging, paths.manifest)
    validate_promoted_production_dense_prefill_cell(root, cell, plan)
    return validation


def _run_accelerator_batch(
    binary: Path,
    root: Path,
    cells: Sequence[DensePrefillCell],
    device: int,
    warmup_runs: int,
    bench_runs: int,
    plan: Mapping[str, object],
) -> tuple[DenseCellValidation, ...]:
    """Measure all missing M buckets for one prepared weight transaction.

    The native process emits one combined aggregate and timing sidecar. This
    function then reconstructs the original per-cell artifacts, authenticates
    every candidate and every timing sample, and only afterward promotes each
    cell atomically. A crash can therefore cost at most one format/geometry M
    batch; previously completed cells remain untouched and immediately
    resumable.
    """

    ordered = _validated_batch_cells(cells)
    paths = _batch_paths(root, ordered)
    paths.aggregate.parent.mkdir(parents=True, exist_ok=True)
    for path in (paths.aggregate, paths.timing, paths.log):
        path.unlink(missing_ok=True)
    for cell in ordered:
        cell_paths = _cell_paths(root, cell)
        for staging in (
            cell_paths.aggregate_staging,
            cell_paths.timing_staging,
            cell_paths.log_staging,
            cell_paths.manifest_staging,
        ):
            staging.unlink(missing_ok=True)

    native_process = {
        "invocation_id": str(uuid.uuid4()),
        "backend": ordered[0].backend,
        "device_ordinal": device,
        "cell_ids": [cell.cell_id for cell in ordered],
    }

    command = (
        str(binary),
        f"--gtest_filter={_BACKEND_TESTS[ordered[0].backend]}",
    )
    environment = _batch_environment(
        ordered, paths, device, warmup_runs, bench_runs
    )
    with paths.log.open("wb") as log:
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
            f"{ordered[0].backend} dense prefill batch failed on device "
            f"{device}; inspect {paths.log}"
        )

    _split_batch_csv(paths.aggregate, root, ordered, timing=False)
    _split_batch_csv(paths.timing, root, ordered, timing=True)
    _write_batch_logs(paths.log, root, ordered)

    validations = []
    for cell in ordered:
        cell_paths = _cell_paths(root, cell)
        validations.append(validate_production_dense_prefill_cell(
            cell_paths.aggregate_staging,
            cell_paths.timing_staging,
            cell,
            bench_runs=bench_runs,
        ))
        _stage_cell_manifest(cell_paths, cell, plan, native_process)

    for cell in ordered:
        cell_paths = _cell_paths(root, cell)
        os.replace(cell_paths.aggregate_staging, cell_paths.aggregate)
        os.replace(cell_paths.timing_staging, cell_paths.timing)
        os.replace(cell_paths.log_staging, cell_paths.log)
        os.replace(cell_paths.manifest_staging, cell_paths.manifest)
        validate_promoted_production_dense_prefill_cell(root, cell, plan)
    for path in (paths.aggregate, paths.timing, paths.log):
        path.unlink(missing_ok=True)
    return tuple(validations)


def run_missing_accelerator_cells(
    binary: Path,
    root: Path,
    cells: Sequence[DensePrefillCell],
    devices: Sequence[int],
    *,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    bench_runs: int = DEFAULT_BENCH_RUNS,
    maximum_new_cells: int | None = None,
    producer: Mapping[str, object] | None = None,
) -> tuple[DensePrefillCell, ...]:
    """Run distinct missing cells concurrently, one worker per physical GPU."""

    backends = {cell.backend for cell in cells}
    if len(backends) != 1 or not backends or not backends <= set(_BACKEND_TESTS):
        raise ValueError("one run must contain cells for exactly one GPU backend")
    backend = next(iter(backends))
    if warmup_runs <= 0 or bench_runs <= 0:
        raise ValueError("timing cardinalities must be positive")
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
    producer = (
        trainer_provenance(binary, backend)
        if producer is None else dict(producer)
    )
    plan = write_sweep_plan(
        root / SWEEP_PLAN_FILENAME,
        backend,
        cells,
        producer,
        warmup_runs=warmup_runs,
        bench_runs=bench_runs,
    )
    missing = [
        cell for cell in cells
        if not _cell_complete(root, cell, plan)
    ]
    if maximum_new_cells is not None:
        if maximum_new_cells <= 0:
            raise ValueError("maximum_new_cells must be positive")
        missing = missing[:maximum_new_cells]

    grouped: dict[tuple[object, ...], list[DensePrefillCell]] = {}
    for cell in missing:
        grouped.setdefault(_batch_group_key(cell), []).append(cell)
    batches = tuple(
        _validated_batch_cells(group)
        for _, group in sorted(grouped.items())
    )

    work: queue.Queue[tuple[DensePrefillCell, ...]] = queue.Queue()
    for batch in batches:
        work.put(batch)
    failures: list[BaseException] = []
    completed_cells: list[DensePrefillCell] = []
    lock = threading.Lock()
    stop = threading.Event()

    def worker(device: int) -> None:
        while not stop.is_set():
            try:
                batch = work.get_nowait()
            except queue.Empty:
                return
            try:
                if len(batch) == 1:
                    _run_accelerator_cell(
                        binary,
                        root,
                        batch[0],
                        device,
                        warmup_runs,
                        bench_runs,
                        plan,
                    )
                else:
                    _run_accelerator_batch(
                        binary,
                        root,
                        batch,
                        device,
                        warmup_runs,
                        bench_runs,
                        plan,
                    )
                with lock:
                    completed_cells.extend(batch)
                    print(
                        f"[{backend} device {device}] completed "
                        f"{len(batch)} cells for "
                        f"{batch[0].source_format.label}/"
                        f"{batch[0].shape.name}",
                        flush=True,
                    )
            except BaseException as error:  # retain worker traceback object
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
            f"{backend} production dense prefill worker failed"
        ) from failures[0]
    return tuple(sorted(completed_cells, key=_cell_sort_key))


def _combine_csvs_atomic(
    output: Path,
    inputs: Iterable[Path],
    columns: Sequence[str],
) -> None:
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


def combine_production_dense_prefill_cells(
    root: Path,
    cells: Sequence[DensePrefillCell],
    *,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    bench_runs: int = DEFAULT_BENCH_RUNS,
) -> tuple[Path, Path]:
    """Require every planned cell and publish deterministic combined corpora."""

    if not cells:
        raise ValueError("cannot combine an empty dense prefill plan")
    backend = cells[0].backend
    if any(cell.backend != backend for cell in cells):
        raise ValueError("cannot combine dense prefill backends")
    root = Path(root)
    plan = load_sweep_plan(
        root / SWEEP_PLAN_FILENAME,
        backend,
        cells,
        warmup_runs=warmup_runs,
        bench_runs=bench_runs,
    )
    paths = []
    for cell in cells:
        cell_paths = _cell_paths(root, cell)
        validate_promoted_production_dense_prefill_cell(root, cell, plan)
        paths.append(cell_paths)
    aggregate = root / "production_dense_prefill.csv"
    timing = root / "production_dense_prefill.timing.csv"
    _combine_csvs_atomic(
        aggregate,
        (path.aggregate for path in paths),
        _aggregate_columns(backend),
    )
    _combine_csvs_atomic(
        timing,
        (path.timing for path in paths),
        _timing_columns(backend),
    )
    return aggregate, timing


def _parse_devices(raw: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("devices must be non-negative ordinals")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("devices must not contain duplicates")
    return values


def _parse_positive_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return values


def _filtered_cells(args: argparse.Namespace) -> tuple[DensePrefillCell, ...]:
    cells = production_dense_prefill_cells(args.backend)
    if args.format:
        selected = {value.upper() for value in args.format}
        cells = tuple(
            cell for cell in cells if cell.source_format.label in selected
        )
    if args.shape:
        selected = set(args.shape)
        cells = tuple(cell for cell in cells if cell.shape.name in selected)
    if args.m:
        selected_m = set(args.m)
        cells = tuple(cell for cell in cells if cell.m in selected_m)
    if not cells:
        raise ValueError("sweep filters selected no dense prefill cells")
    return cells


def main() -> int:
    """CLI entry point for planning, execution, status, and combination."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "run", "status", "combine"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--backend", choices=("cuda", "rocm"), required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--format", action="append")
        subparser.add_argument("--shape", action="append")
        subparser.add_argument("--m", type=_parse_positive_ints)
        subparser.add_argument(
            "--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS
        )
        subparser.add_argument(
            "--bench-runs", type=int, default=DEFAULT_BENCH_RUNS
        )
    plan_command = subparsers.choices["plan"]
    plan_command.add_argument("--binary", type=Path, required=True)
    run = subparsers.choices["run"]
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--devices", type=_parse_devices, required=True)
    run.add_argument("--maximum-new-cells", type=int)

    args = parser.parse_args()
    if args.warmup_runs <= 0 or args.bench_runs <= 0:
        parser.error("timing cardinalities must be positive")
    cells = _filtered_cells(args)
    plan_path = args.output_dir / SWEEP_PLAN_FILENAME
    if args.command == "plan":
        producer = trainer_provenance(args.binary, args.backend)
        require_release_trainer(producer)
        write_sweep_plan(
            plan_path,
            args.backend,
            cells,
            producer,
            warmup_runs=args.warmup_runs,
            bench_runs=args.bench_runs,
        )
        print(f"planned {len(cells)} cells in {plan_path}")
    elif args.command == "run":
        producer = trainer_provenance(args.binary, args.backend)
        require_release_trainer(producer)
        completed = run_missing_accelerator_cells(
            args.binary,
            args.output_dir,
            cells,
            args.devices,
            warmup_runs=args.warmup_runs,
            bench_runs=args.bench_runs,
            maximum_new_cells=args.maximum_new_cells,
            producer=producer,
        )
        print(f"completed {len(completed)} new cells of {len(cells)} planned")
    elif args.command == "status":
        plan = load_sweep_plan(
            plan_path,
            args.backend,
            cells,
            warmup_runs=args.warmup_runs,
            bench_runs=args.bench_runs,
        )
        complete = sum(
            _cell_complete(args.output_dir, cell, plan)
            for cell in cells
        )
        print(f"{complete}/{len(cells)} cells complete")
    else:
        aggregate, timing = combine_production_dense_prefill_cells(
            args.output_dir,
            cells,
            warmup_runs=args.warmup_runs,
            bench_runs=args.bench_runs,
        )
        print(f"combined {len(cells)} cells into {aggregate} and {timing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
