"""Resumable production-MoE prefill candidate-pair sweep transaction.

The C++ performance binary owns GPU execution, graph capture, correctness, and
canonical HIP/CUDA event timing. This module owns the durable transaction
around those launches: it expands the pinned GGUF inventory into immutable
``(backend, routed source tuple, route profile, M)`` cells, assigns distinct
same-weight M batches to physical devices, authenticates every aggregate and
raw timing row, and atomically promotes only complete cells. A rerun therefore
preserves every validated cell and launches only missing or invalid work.

The C++ executable prepares one real routed-expert set, then measures every M
in that same source/geometry/routing batch. Python splits the native evidence
back into independently committed cells. Content-addressed per-invocation plans
make supplemental shapes or M values additive without weakening the immutable
Release-trainer identity or the selected source identity in each plan.
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
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .adapters.evidence import verify_aggregate_timing
from .corpus_provenance import (
    TRAINER_PROVENANCE_SCHEMA,
    mapping_digest,
    require_release_trainer,
    sha256_file,
    trainer_provenance,
)
from .prefill_matrix import GPU_PREFILL_M_BUCKETS
from .moe_routing_profiles import (
    MOE_ROUTING_PROFILES,
    summarize_moe_routing_profile,
)
from .qwen_moe_gguf_patterns import (
    QwenMoERoutedPrefillCase,
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
CORPUS_MANIFEST_SCHEMA = "native-vnni-production-moe-prefill-corpus-v2"
SWEEP_PLAN_SCHEMA = "native-vnni-production-moe-prefill-plan-v2"
CELL_MANIFEST_SCHEMA = "native-vnni-production-moe-prefill-cell-v2"
COMBINED_MANIFEST_SCHEMA = "native-vnni-production-moe-prefill-combined-v1"
CORPUS_MANIFEST_FILENAME = "corpus.json"


def _is_sha256_digest(value: object) -> bool:
    """Return whether ``value`` is one canonical prefixed SHA-256 digest."""

    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


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

        gate_codebook, up_codebook, down_codebook = (
            self.case.routed.runtime_codebooks(self.backend)
        )
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
            "gate_execution_codebook": gate_codebook,
            "up_execution_codebook": up_codebook,
            "down_execution_codebook": down_codebook,
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
    manifest: Path
    aggregate_staging: Path
    timing_staging: Path
    log_staging: Path
    manifest_staging: Path


@dataclass(frozen=True)
class BatchPaths:
    """Ephemeral native-process outputs for one same-weight M batch."""

    aggregate: Path
    timing: Path
    log: Path


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
        manifest=stem.with_suffix(".manifest.json"),
        aggregate_staging=stem.with_suffix(".csv.inprogress"),
        timing_staging=stem.with_suffix(".timing.csv.inprogress"),
        log_staging=stem.with_suffix(".log.inprogress"),
        manifest_staging=stem.with_suffix(".manifest.json.inprogress"),
    )


def production_moe_prefill_cell_paths(
    root: Path,
    cell: ProductionMoEPrefillCell,
) -> CellPaths:
    """Return the stable on-disk locations owned by one planned cell."""

    return _cell_paths(Path(root), cell)


def _batch_group_key(
    cell: ProductionMoEPrefillCell,
) -> tuple[object, ...]:
    """Return every prepared-weight and routing field held fixed in a batch."""

    return (
        cell.backend,
        cell.case.source_key,
        cell.route_profile,
    )


def _validated_batch_cells(
    cells: Sequence[ProductionMoEPrefillCell],
) -> tuple[ProductionMoEPrefillCell, ...]:
    """Validate one process batch in which only the runtime row count varies."""

    ordered = tuple(sorted(cells, key=lambda cell: cell.m))
    if not ordered:
        raise ValueError("production MoE process batch cannot be empty")
    group = _batch_group_key(ordered[0])
    if any(_batch_group_key(cell) != group for cell in ordered):
        raise ValueError(
            "production MoE process batch changed format, geometry, or route"
        )
    if len({cell.m for cell in ordered}) != len(ordered):
        raise ValueError("production MoE process batch contains duplicate M")
    return ordered


def _batch_paths(
    root: Path,
    cells: Sequence[ProductionMoEPrefillCell],
) -> BatchPaths:
    """Return collision-resistant staging paths for one native process."""

    ordered = _validated_batch_cells(cells)
    identity = [cell.identity for cell in ordered]
    digest = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()[:20]
    first = ordered[0]
    stem = (
        Path(root) / "batches" /
        f"{first.backend}-{first.route_profile}-{digest}"
    )
    return BatchPaths(
        aggregate=stem.with_suffix(".csv.inprogress"),
        timing=stem.with_suffix(".timing.csv.inprogress"),
        log=stem.with_suffix(".log.inprogress"),
    )


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


def _corpus_manifest_payload(
    backend: str,
    trainer: Mapping[str, object],
) -> dict[str, object]:
    """Construct the immutable producer and inventory contract for one root."""

    normalized = backend.strip().lower()
    if normalized not in {"cuda", "rocm"}:
        raise ValueError(f"unsupported production GPU backend {backend!r}")
    require_release_trainer(trainer)
    if trainer.get("backend") != normalized:
        raise ValueError("trainer provenance belongs to another backend")
    body: dict[str, object] = {
        "schema_version": CORPUS_MANIFEST_SCHEMA,
        "backend": normalized,
        "route_profiles": list(MOE_ROUTING_PROFILES),
        "candidate_ids": list(production_moe_prefill_candidate_ids(normalized)),
        "trainer_provenance": dict(trainer),
        "trainer_provenance_digest": mapping_digest(trainer),
    }
    body["corpus_digest"] = mapping_digest(body)
    return body


def _read_corpus_manifest(path: Path) -> dict[str, object]:
    """Read and authenticate one immutable corpus-root contract."""

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: corpus manifest must be a JSON object")
    if payload.get("schema_version") != CORPUS_MANIFEST_SCHEMA:
        raise ValueError(f"{path}: unsupported corpus-manifest schema")
    recorded = payload.get("corpus_digest")
    if not _is_sha256_digest(recorded):
        raise ValueError(f"{path}: malformed corpus-manifest digest")
    unsigned = dict(payload)
    unsigned.pop("corpus_digest", None)
    if recorded != mapping_digest(unsigned):
        raise ValueError(f"{path}: corpus-manifest digest mismatch")
    trainer = payload.get("trainer_provenance")
    if not isinstance(trainer, dict) or (
        payload.get("trainer_provenance_digest") != mapping_digest(trainer)
    ):
        raise ValueError(f"{path}: trainer provenance digest mismatch")
    if payload.get("backend") not in {"cuda", "rocm"}:
        raise ValueError(f"{path}: unsupported corpus backend")
    if not _is_sha256_digest(payload.get("trainer_provenance_digest")):
        raise ValueError(f"{path}: malformed trainer provenance digest")
    for name in ("route_profiles", "candidate_ids"):
        values = payload.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{path}: malformed {name}")
    return payload


def write_corpus_manifest(
    root: Path,
    backend: str,
    trainer: Mapping[str, object],
) -> dict[str, object]:
    """Create one corpus root, or prove its producer contract is unchanged."""

    root = Path(root)
    path = root / CORPUS_MANIFEST_FILENAME
    expected = _corpus_manifest_payload(backend, trainer)
    if path.exists():
        existing = _read_corpus_manifest(path)
        if existing != expected:
            raise ValueError(
                f"{path}: immutable corpus manifest disagrees with this run; "
                "use a new corpus directory"
            )
        return existing
    _write_json_atomic(path, expected)
    return expected


def load_corpus_manifest(root: Path, backend: str) -> dict[str, object]:
    """Authenticate the corpus root against today's immutable registries."""

    path = Path(root) / CORPUS_MANIFEST_FILENAME
    existing = _read_corpus_manifest(path)
    trainer = existing["trainer_provenance"]
    assert isinstance(trainer, dict)
    expected = _corpus_manifest_payload(backend, trainer)
    if existing != expected:
        raise ValueError(
            f"{path}: corpus inventory or candidate contract changed"
        )
    return existing


def _sweep_plan_payload(
    cells: Sequence[ProductionMoEPrefillCell],
    corpus: Mapping[str, object],
) -> dict[str, object]:
    """Construct one exact additive invocation plan within a corpus root."""

    ordered = tuple(sorted(cells))
    if not ordered:
        raise ValueError("production MoE sweep plan cannot be empty")
    backend = str(corpus["backend"])
    if any(cell.backend != backend for cell in ordered):
        raise ValueError("sweep plan contains a cell for another backend")
    if len({cell.cell_id for cell in ordered}) != len(ordered):
        raise ValueError("sweep plan contains duplicate cells")
    body: dict[str, object] = {
        "schema_version": SWEEP_PLAN_SCHEMA,
        "backend": backend,
        "corpus_digest": corpus["corpus_digest"],
        "cells": [cell.canonical_mapping() for cell in ordered],
    }
    body["plan_digest"] = mapping_digest(body)
    return body


def _plan_path(root: Path, plan_digest: str) -> Path:
    """Map one authenticated digest onto its immutable plan filename."""

    if not plan_digest.startswith("sha256:") or len(plan_digest) != 71:
        raise ValueError("invalid production MoE sweep-plan digest")
    return Path(root) / "plans" / f"{plan_digest.removeprefix('sha256:')}.json"


def _read_sweep_plan(path: Path) -> dict[str, object]:
    """Read and authenticate one content-addressed invocation plan."""

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: sweep plan must be a JSON object")
    if payload.get("schema_version") != SWEEP_PLAN_SCHEMA:
        raise ValueError(f"{path}: unsupported sweep-plan schema")
    recorded = payload.get("plan_digest")
    if not _is_sha256_digest(recorded):
        raise ValueError(f"{path}: malformed sweep-plan digest")
    unsigned = dict(payload)
    unsigned.pop("plan_digest", None)
    if recorded != mapping_digest(unsigned):
        raise ValueError(f"{path}: sweep-plan digest mismatch")
    if path.name != f"{recorded.removeprefix('sha256:')}.json":
        raise ValueError(f"{path}: filename disagrees with sweep-plan digest")
    if payload.get("backend") not in {"cuda", "rocm"}:
        raise ValueError(f"{path}: unsupported sweep-plan backend")
    if not _is_sha256_digest(payload.get("corpus_digest")):
        raise ValueError(f"{path}: malformed sweep-plan corpus digest")
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells or any(
        not isinstance(cell, dict)
        or not isinstance(cell.get("cell_id"), str)
        or not cell["cell_id"]
        for cell in cells
    ):
        raise ValueError(f"{path}: malformed sweep-plan cell inventory")
    cell_ids = [str(cell["cell_id"]) for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError(f"{path}: duplicate sweep-plan cells")
    return payload


def write_sweep_plan(
    root: Path,
    backend: str,
    cells: Sequence[ProductionMoEPrefillCell],
    trainer: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Create/reuse an additive plan under one immutable corpus contract."""

    corpus = write_corpus_manifest(root, backend, trainer)
    expected = _sweep_plan_payload(cells, corpus)
    path = _plan_path(root, str(expected["plan_digest"]))
    if path.exists():
        existing = _read_sweep_plan(path)
        if existing != expected:
            raise ValueError(f"{path}: content-addressed sweep plan changed")
        return corpus, existing
    _write_json_atomic(path, expected)
    return corpus, expected


def load_sweep_plan(
    root: Path,
    backend: str,
    cells: Sequence[ProductionMoEPrefillCell],
) -> tuple[dict[str, object], dict[str, object]]:
    """Authenticate one requested matrix without mutating corpus state."""

    corpus = load_corpus_manifest(root, backend)
    expected = _sweep_plan_payload(cells, corpus)
    existing = _read_sweep_plan(
        _plan_path(root, str(expected["plan_digest"]))
    )
    if existing != expected:
        raise ValueError("requested production MoE sweep plan changed")
    return corpus, existing


def _cell_manifest_payload(
    paths: CellPaths,
    cell: ProductionMoEPrefillCell,
    corpus: Mapping[str, object],
    plan: Mapping[str, object],
    native_process: Mapping[str, object],
) -> dict[str, object]:
    """Bind one promoted cell to its plan, producer, and exact bytes."""

    mapping = cell.canonical_mapping()
    if (
        corpus.get("backend") != cell.backend
        or plan.get("backend") != cell.backend
        or plan.get("corpus_digest") != corpus.get("corpus_digest")
        or mapping not in plan.get("cells", [])
    ):
        raise ValueError("cell manifest producer closure does not own cell")
    body: dict[str, object] = {
        "schema_version": CELL_MANIFEST_SCHEMA,
        "cell": mapping,
        "corpus_digest": corpus["corpus_digest"],
        "plan_digest": plan["plan_digest"],
        "trainer_provenance_digest": corpus["trainer_provenance_digest"],
        "native_process": dict(native_process),
        "artifacts": {
            "aggregate": {
                "filename": paths.aggregate.name,
                "sha256": sha256_file(paths.aggregate_staging),
            },
            "timing": {
                "filename": paths.timing.name,
                "sha256": sha256_file(paths.timing_staging),
            },
            "log": {
                "filename": paths.log.name,
                "sha256": sha256_file(paths.log_staging),
            },
        },
    }
    body["manifest_digest"] = mapping_digest(body)
    return body


def _stage_cell_manifest(
    paths: CellPaths,
    cell: ProductionMoEPrefillCell,
    corpus: Mapping[str, object],
    plan: Mapping[str, object],
    native_process: Mapping[str, object],
) -> None:
    """Durably stage a cell commit marker after evidence validates."""

    payload = _cell_manifest_payload(
        paths, cell, corpus, plan, native_process
    )
    paths.manifest_staging.parent.mkdir(parents=True, exist_ok=True)
    with paths.manifest_staging.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_cell_manifest(path: Path) -> dict[str, object]:
    """Read and authenticate one per-cell commit marker."""

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: cell manifest must be a JSON object")
    if payload.get("schema_version") != CELL_MANIFEST_SCHEMA:
        raise ValueError(f"{path}: unsupported cell-manifest schema")
    recorded = payload.get("manifest_digest")
    if not _is_sha256_digest(recorded):
        raise ValueError(f"{path}: malformed cell-manifest digest")
    unsigned = dict(payload)
    unsigned.pop("manifest_digest", None)
    if recorded != mapping_digest(unsigned):
        raise ValueError(f"{path}: cell-manifest digest mismatch")
    return payload


def validate_promoted_production_moe_prefill_cell(
    root: Path,
    cell: ProductionMoEPrefillCell,
    corpus: Mapping[str, object] | None = None,
) -> CellValidation:
    """Prove one committed cell belongs to the exact corpus closure."""

    root = Path(root)
    corpus = (
        load_corpus_manifest(root, cell.backend)
        if corpus is None else dict(corpus)
    )
    paths = _cell_paths(root, cell)
    manifest = _read_cell_manifest(paths.manifest)
    if manifest.get("cell") != cell.canonical_mapping():
        raise ValueError(f"{paths.manifest}: cell identity mismatch")
    for field in ("corpus_digest", "trainer_provenance_digest"):
        if manifest.get(field) != corpus.get(field):
            raise ValueError(f"{paths.manifest}: {field} mismatch")

    plan_digest = str(manifest.get("plan_digest"))
    plan = _read_sweep_plan(_plan_path(root, plan_digest))
    if plan.get("corpus_digest") != corpus.get("corpus_digest") or (
        cell.canonical_mapping() not in plan.get("cells", [])
    ):
        raise ValueError(f"{paths.manifest}: producing plan does not own cell")

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
        or any(not isinstance(cell_id, str) for cell_id in cell_ids)
        or len(cell_ids) != len(set(cell_ids))
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
            set(record) != {"filename", "sha256"}
            or not _is_sha256_digest(record.get("sha256"))
            or record.get("filename") != artifact_path.name
            or record.get("sha256") != sha256_file(artifact_path)
        ):
            raise ValueError(
                f"{paths.manifest}: {role} artifact digest mismatch"
            )
    return validate_production_moe_prefill_cell(
        paths.aggregate, paths.timing, cell
    )


def _cell_complete(
    root: Path,
    cell: ProductionMoEPrefillCell,
    corpus: Mapping[str, object],
) -> bool:
    try:
        validate_promoted_production_moe_prefill_cell(root, cell, corpus)
    except (OSError, ValueError):
        return False
    return True


_BACKEND_TESTS = {
    "cuda": "Perf__MoEVerifierPrefill.CUDA_ProductionGGUFMixtureCandidateTrainer",
    "rocm": "Perf__MoEVerifierPrefill.ROCm_ProductionGGUFMixtureCandidatePairTrainer",
}


def _batch_environment(
    cells: Sequence[ProductionMoEPrefillCell],
    paths: BatchPaths,
    device: int,
) -> dict[str, str]:
    """Build the exact native environment for a same-weight multi-M batch."""

    ordered = _validated_batch_cells(cells)
    first = ordered[0]
    environment = os.environ.copy()
    prefix = f"LLAMINAR_{first.backend.upper()}_MOE_PRODUCTION_SWEEP"
    environment.update({
        prefix: "1",
        f"{prefix}_CASES": first.case.evidence_id,
        f"{prefix}_M": ",".join(str(cell.m) for cell in ordered),
        f"{prefix}_ROUTE_PROFILE": first.route_profile,
        f"{prefix}_MAX_CELLS": str(len(ordered)),
        f"{prefix}_DEVICE": str(device),
        f"{prefix}_CSV": str(paths.aggregate),
        f"{prefix}_TIMING_CSV": str(paths.timing),
    })
    return environment


def _write_csv_rows(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    """Durably publish one already-separated cell CSV into staging."""

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
    cells: Sequence[ProductionMoEPrefillCell],
    *,
    timing: bool,
) -> None:
    """Split multi-M native evidence into independently resumable cells."""

    ordered = _validated_batch_cells(cells)
    columns = TIMING_COLUMNS if timing else AGGREGATE_COLUMNS
    cells_by_m = {cell.m: cell for cell in ordered}
    rows_by_m: dict[int, list[dict[str, str]]] = {
        cell.m: [] for cell in ordered
    }
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_exact_header(source, reader.fieldnames, columns)
        for line, row in enumerate(reader, start=2):
            context = f"{source}:{line}"
            try:
                cell = cells_by_m[int(row["m"])]
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"{context}: row does not belong to the native M batch"
                ) from error
            if (
                row["backend"] != cell.backend
                or row["phase"] != "moe_production_prefill"
                or row["case_id"] != cell.case.evidence_id
                or row["route_profile"] != cell.route_profile
            ):
                raise ValueError(
                    f"{context}: row identity changed inside native M batch"
                )
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
    cells: Sequence[ProductionMoEPrefillCell],
) -> None:
    """Attach one native process log to every cell emitted by that process."""

    payload = source.read_bytes()
    for cell in _validated_batch_cells(cells):
        destination = _cell_paths(root, cell).log_staging
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def _run_accelerator_batch(
    binary: Path,
    root: Path,
    cells: Sequence[ProductionMoEPrefillCell],
    device: int,
    corpus: Mapping[str, object],
    plan: Mapping[str, object],
) -> tuple[CellValidation, ...]:
    """Measure all missing M buckets while retaining one prepared expert set.

    The C++ trainer emits independent event samples for every M and candidate;
    batching removes only process startup and immutable expert preparation. The
    combined files are split, fully authenticated, and fsynced before any cell
    reaches its final path, preserving per-M resume semantics.
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
    environment = _batch_environment(ordered, paths, device)
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
            f"{ordered[0].backend} production MoE batch failed on device "
            f"{device}; inspect {paths.log}"
        )

    _split_batch_csv(paths.aggregate, root, ordered, timing=False)
    _split_batch_csv(paths.timing, root, ordered, timing=True)
    _write_batch_logs(paths.log, root, ordered)
    validations = []
    for cell in ordered:
        cell_paths = _cell_paths(root, cell)
        validations.append(validate_production_moe_prefill_cell(
            cell_paths.aggregate_staging,
            cell_paths.timing_staging,
            cell,
        ))
        _stage_cell_manifest(
            cell_paths,
            cell,
            corpus,
            plan,
            native_process,
        )

    for cell in ordered:
        cell_paths = _cell_paths(root, cell)
        os.replace(cell_paths.aggregate_staging, cell_paths.aggregate)
        os.replace(cell_paths.timing_staging, cell_paths.timing)
        os.replace(cell_paths.log_staging, cell_paths.log)
        # The manifest is the transaction commit marker. Promoting it last
        # prevents a crash during any preceding rename from creating a cell
        # that a future resume can mistake for complete evidence.
        os.replace(cell_paths.manifest_staging, cell_paths.manifest)
        validate_promoted_production_moe_prefill_cell(root, cell, corpus)
    for path in (paths.aggregate, paths.timing, paths.log):
        path.unlink(missing_ok=True)
    return tuple(validations)


def run_missing_accelerator_cells(
    binary: Path,
    root: Path,
    cells: Sequence[ProductionMoEPrefillCell],
    devices: Sequence[int],
    *,
    maximum_new_cells: int | None = None,
    producer: Mapping[str, object] | None = None,
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
    producer = (
        trainer_provenance(binary, backend)
        if producer is None else dict(producer)
    )
    corpus, plan = write_sweep_plan(root, backend, cells, producer)
    missing = [
        cell for cell in cells if not _cell_complete(root, cell, corpus)
    ]
    if maximum_new_cells is not None:
        if maximum_new_cells <= 0:
            raise ValueError("maximum_new_cells must be positive")
        missing = missing[:maximum_new_cells]
    grouped: dict[tuple[object, ...], list[ProductionMoEPrefillCell]] = {}
    for cell in missing:
        grouped.setdefault(_batch_group_key(cell), []).append(cell)
    work: queue.Queue[tuple[ProductionMoEPrefillCell, ...]] = queue.Queue()
    for _key, batch in sorted(grouped.items()):
        work.put(_validated_batch_cells(batch))

    failures: list[BaseException] = []
    completed_cells: list[ProductionMoEPrefillCell] = []
    lock = threading.Lock()
    stop = threading.Event()

    def worker(device: int) -> None:
        while not stop.is_set():
            try:
                batch = work.get_nowait()
            except queue.Empty:
                return
            try:
                _run_accelerator_batch(
                    binary,
                    root,
                    batch,
                    device,
                    corpus,
                    plan,
                )
                with lock:
                    completed_cells.extend(batch)
                    print(
                        f"[{backend} device {device}] completed "
                        f"{len(batch)} cells for "
                        f"{batch[0].case.evidence_id}/"
                        f"{batch[0].route_profile}",
                        flush=True,
                    )
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
    producer: Mapping[str, object] | None = None,
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
        producer=producer,
    )


def run_missing_cuda_cells(
    binary: Path,
    root: Path,
    cells: Sequence[ProductionMoEPrefillCell],
    devices: Sequence[int],
    *,
    maximum_new_cells: int | None = None,
    producer: Mapping[str, object] | None = None,
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
        producer=producer,
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

    ordered = tuple(sorted(cells))
    if not ordered:
        raise ValueError("cannot combine an empty production MoE plan")
    backend = ordered[0].backend
    if any(cell.backend != backend for cell in ordered):
        raise ValueError("cannot combine production MoE backends")
    root = Path(root)
    corpus, plan = load_sweep_plan(root, backend, ordered)
    paths = []
    for cell in ordered:
        cell_paths = _cell_paths(root, cell)
        validate_promoted_production_moe_prefill_cell(
            root, cell, corpus
        )
        paths.append(cell_paths)
    plan_digest = str(plan["plan_digest"])
    combined_root = (
        root / "combined" / plan_digest.removeprefix("sha256:")
    )
    aggregate = combined_root / "production_moe_prefill.csv"
    timing = combined_root / "production_moe_prefill.timing.csv"
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
    combined_manifest: dict[str, object] = {
        "schema_version": COMBINED_MANIFEST_SCHEMA,
        "backend": backend,
        "corpus_digest": corpus["corpus_digest"],
        "plan_digest": plan_digest,
        "cells": [
            {
                "cell_id": cell.cell_id,
                "manifest_sha256": sha256_file(path.manifest),
            }
            for cell, path in zip(ordered, paths, strict=True)
        ],
        "artifacts": {
            "aggregate": {
                "filename": aggregate.name,
                "sha256": sha256_file(aggregate),
            },
            "timing": {
                "filename": timing.name,
                "sha256": sha256_file(timing),
            },
        },
    }
    combined_manifest["manifest_digest"] = mapping_digest(combined_manifest)
    _write_json_atomic(combined_root / "manifest.json", combined_manifest)
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
    cases = qwen_moe_routed_prefill_cases()
    if args.case:
        selected = set(args.case)
        known = {case.evidence_id for case in cases}
        if unknown := selected - known:
            raise ValueError(f"unknown production MoE cases: {sorted(unknown)}")
        cases = tuple(case for case in cases if case.evidence_id in selected)
    m_values = tuple(args.m) if args.m else GPU_PREFILL_M_BUCKETS
    route_profiles = (
        tuple(dict.fromkeys(args.route_profile))
        if args.route_profile else MOE_ROUTING_PROFILES
    )
    cells = tuple(
        ProductionMoEPrefillCell(
            args.backend,
            case,
            m,
            route_profile,
        )
        for case in cases
        for route_profile in route_profiles
        for m in m_values
    )
    if not cells:
        raise ValueError("sweep filters selected no production MoE cells")
    return cells


def main() -> int:
    """CLI entry point for planning, execution, status, and combination."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "status", "combine"):
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
    plan_command = subparsers.choices["plan"]
    plan_command.add_argument("--binary", type=Path, required=True)
    run = subparsers.choices["run"]
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--devices", type=_parse_devices, required=True)
    run.add_argument("--maximum-new-cells", type=int)

    args = parser.parse_args()
    cells = _filtered_cells(args)
    output_dir = Path(args.output_dir)
    if args.command == "plan":
        producer = trainer_provenance(args.binary, args.backend)
        corpus, plan = write_sweep_plan(
            output_dir, args.backend, cells, producer
        )
        plan_path = _plan_path(output_dir, str(plan["plan_digest"]))
        print(
            f"planned {len(cells)} cells in {plan_path}; "
            f"corpus={corpus['corpus_digest']}"
        )
        return 0
    if args.command == "run":
        producer = trainer_provenance(args.binary, args.backend)
        completed = run_missing_accelerator_cells(
            args.binary,
            output_dir,
            cells,
            args.devices,
            maximum_new_cells=args.maximum_new_cells,
            producer=producer,
        )
        corpus, _plan = load_sweep_plan(output_dir, args.backend, cells)
        remaining = sum(
            not _cell_complete(output_dir, cell, corpus) for cell in cells
        )
        print(f"completed {len(completed)} new cells; {remaining} cells remain")
        return 0
    if args.command == "status":
        corpus, plan = load_sweep_plan(output_dir, args.backend, cells)
        complete = sum(
            _cell_complete(output_dir, cell, corpus) for cell in cells
        )
        print(
            f"{complete}/{len(cells)} cells complete; "
            f"plan={plan['plan_digest']}"
        )
        return 0
    aggregate, timing = combine_production_moe_prefill_cells(output_dir, cells)
    print(f"combined {len(cells)} cells into {aggregate} and {timing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
