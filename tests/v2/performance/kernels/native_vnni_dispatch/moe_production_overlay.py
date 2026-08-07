"""Generate production-MoE exact overlays from authenticated GPU cells.

Generic role-specific rules remain responsible for every unseen geometry and
positive M. This module emits the additive exact layer above them. A runtime
key contains only facts available at graph capture: prepared execution
codebooks, routed geometry, expert count, top-k, and exact M. GGUF source labels
remain evidence surfaces and never leak into production dispatch.

Several source formats can collapse onto one prepared execution codebook. A
candidate is installable for that runtime key only when the C++ tournament
robustly retimed it on every contributing source surface. Selection then
minimizes maximum per-surface regret, followed by p95 and mean regret. This is
the exact-overlay analogue of alias-robust generic training and prevents one
source alias from hiding another alias's performance loss.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .adapters.evidence import raw_corpus_id
from .production_moe_prefill_sweep import (
    AGGREGATE_COLUMNS,
    ProductionMoEPrefillCell,
    production_moe_prefill_cell_paths,
    production_moe_prefill_candidate_ids,
    production_moe_prefill_cells,
    validate_production_moe_prefill_cell,
)
from .moe_routing_profiles import MOE_ROUTING_PROFILES


MAXIMUM_OVERLAY_SURFACE_REGRET = 0.05
from .qwen_moe_gguf_patterns import qwen_moe_routed_prefill_cases


_PAIR_PATTERN = re.compile(
    r"g_tm(?P<gate_m>\d+)_tn(?P<gate_n>\d+)"
    r"__d_tm(?P<down_m>\d+)_tn(?P<down_n>\d+)"
)
_CUDA_PATTERN = re.compile(r"g_tn(?P<gate_n>\d+)__d_fixed")


def _candidate_dimensions(
    backend: str,
    candidate_id: str,
) -> tuple[int, int, int, int]:
    """Decode one authenticated backend candidate into generated geometry."""

    if backend == "rocm":
        match = _PAIR_PATTERN.fullmatch(candidate_id)
        if match is not None:
            return (
                int(match["gate_m"]),
                int(match["gate_n"]),
                int(match["down_m"]),
                int(match["down_n"]),
            )
    elif backend == "cuda":
        match = _CUDA_PATTERN.fullmatch(candidate_id)
        if match is not None:
            return 0, int(match["gate_n"]), 0, 0
    else:
        raise ValueError(f"unsupported production MoE backend {backend!r}")
    raise ValueError(f"selected malformed {backend} candidate {candidate_id}")


@dataclass(frozen=True, order=True)
class MoEProductionOverlayKey:
    """Complete model-agnostic key for one exact production pair."""

    gateup_codebook: int
    down_codebook: int
    hidden_size: int
    expert_width: int
    expert_count: int
    top_k: int
    m: int


@dataclass(frozen=True)
class MoECandidateSurface:
    """One source-alias and route-profile surface for one candidate."""

    surface_id: str
    candidate_id: str
    median_us: float
    p95_us: float
    cv: float
    finalist: bool


@dataclass(frozen=True)
class MoEProductionOverlayWinner:
    """Alias-robust exact pair selected for one runtime key."""

    key: MoEProductionOverlayKey
    candidate_id: str
    gate_tile_m: int
    gate_tile_n: int
    down_tile_m: int
    down_tile_n: int
    surface_count: int
    maximum_surface_regret: float
    p95_surface_regret: float
    mean_surface_regret: float
    maximum_cv: float


def _explicit_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"{name} must be an explicit boolean, got {value!r}")


def _p95(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("p95 requires at least one value")
    ordered = sorted(values)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return ordered[min(len(ordered) - 1, rank - 1)]


def _row_key(row: Mapping[str, str]) -> MoEProductionOverlayKey:
    gate = int(row["gate_execution_codebook"])
    up = int(row["up_execution_codebook"])
    if gate != up:
        raise ValueError("production fused gate/up row has distinct codebooks")
    return MoEProductionOverlayKey(
        gateup_codebook=gate,
        down_codebook=int(row["down_execution_codebook"]),
        hidden_size=int(row["hidden_size"]),
        expert_width=int(row["expert_width"]),
        expert_count=int(row["expert_count"]),
        top_k=int(row["top_k"]),
        m=int(row["m"]),
    )


def _cell_runtime_key(
    cell: ProductionMoEPrefillCell,
) -> MoEProductionOverlayKey:
    gate, up, down = cell.case.routed.runtime_codebooks(cell.backend)
    if gate != up:
        raise ValueError(f"{cell.case.evidence_id}: gate/up codebooks differ")
    return MoEProductionOverlayKey(
        gateup_codebook=gate,
        down_codebook=down,
        hidden_size=cell.case.hidden_size,
        expert_width=cell.case.routed_expert_width,
        expert_count=cell.case.expert_count,
        top_k=cell.case.experts_per_token,
        m=cell.m,
    )


def require_moe_overlay_surface_totality(
    cells: Sequence[ProductionMoEPrefillCell],
    *,
    backend: str,
) -> None:
    """Require all source aliases and routing profiles for every runtime key.

    Targeted additive installation is useful, but an exact runtime key does not
    encode either GGUF source alias or route distribution. Supplying one profile
    or one alias would therefore recreate the overfit that this corpus is meant
    to prevent. This gate computes the complete expected surface set from the
    pinned manifest and rejects a partial selection before reading or rendering
    any winner.
    """

    normalized_backend = backend.strip().lower()
    if normalized_backend not in {"cuda", "rocm"}:
        raise ValueError(f"unsupported production MoE backend {backend!r}")
    grouped: dict[MoEProductionOverlayKey, list[ProductionMoEPrefillCell]] = {}
    for cell in cells:
        if cell.backend != normalized_backend:
            raise ValueError(
                f"{normalized_backend} surface gate received {cell.backend} cell"
            )
        grouped.setdefault(_cell_runtime_key(cell), []).append(cell)

    known_cases = qwen_moe_routed_prefill_cases()
    for key, key_cells in grouped.items():
        expected = {
            (case.evidence_id, profile)
            for case in known_cases
            for profile in MOE_ROUTING_PROFILES
            if _cell_runtime_key(ProductionMoEPrefillCell(
                normalized_backend, case, key.m, profile
            )) == key
        }
        actual = {
            (cell.case.evidence_id, cell.route_profile) for cell in key_cells
        }
        if actual != expected:
            raise ValueError(
                f"{key}: runtime key has incomplete route/source surfaces; "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )


def discover_additive_moe_production_cells(
    work_directory: Path,
    *,
    backend: str,
    baseline_cells: Sequence[ProductionMoEPrefillCell],
) -> tuple[ProductionMoEPrefillCell, ...]:
    """Discover complete content-addressed cells beyond the baseline matrix.

    Exact overlays are deliberately additive: measuring a new production M
    must not invalidate or rerun every previously authenticated cell. The cell
    filename is derived from its complete identity, so this reader can recover
    a supplemental cell from the first aggregate row, reconstruct its expected
    path, and then apply the ordinary aggregate/raw-timing validator.

    A runtime codebook key may have several GGUF source aliases. Installing
    only one of those surfaces would overfit the overlay, so discovery requires
    every known alias at the supplemental M before returning that key. Staging
    debris, orphaned aggregate/timing files, unknown source cases, and filenames
    that disagree with the reconstructed identity are fatal.
    """

    normalized_backend = backend.strip().lower()
    if normalized_backend not in {"cuda", "rocm"}:
        raise ValueError(f"unsupported production MoE backend {backend!r}")

    root = Path(work_directory)
    cell_directory = root / "cells"
    if not cell_directory.is_dir():
        return ()
    in_progress = tuple(sorted(cell_directory.glob("*.inprogress")))
    if in_progress:
        raise ValueError(
            "production MoE overlay corpus contains incomplete staging files: "
            f"{in_progress[:4]}"
        )

    cases = {
        case.evidence_id: case for case in qwen_moe_routed_prefill_cases()
    }
    baseline_ids = {cell.cell_id for cell in baseline_cells}
    discovered: dict[str, ProductionMoEPrefillCell] = {}

    aggregate_paths = tuple(sorted(
        path
        for path in cell_directory.glob(f"{normalized_backend}-*.csv")
        if not path.name.endswith(".timing.csv")
    ))
    for aggregate_path in aggregate_paths:
        timing_path = aggregate_path.with_name(
            aggregate_path.name[:-4] + ".timing.csv"
        )
        if not timing_path.is_file():
            raise ValueError(
                f"{aggregate_path}: supplemental aggregate has no raw timing sidecar"
            )
        with aggregate_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != AGGREGATE_COLUMNS:
                raise ValueError(
                    f"{aggregate_path}: supplemental aggregate schema changed"
                )
            first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"{aggregate_path}: supplemental aggregate is empty")
        case_id = first_row["case_id"]
        if case_id not in cases:
            raise ValueError(
                f"{aggregate_path}: unknown production MoE source case {case_id!r}"
            )
        m = int(first_row["m"])
        if m <= 0:
            raise ValueError(f"{aggregate_path}: supplemental M must be positive")
        route_profile = first_row["route_profile"]
        if route_profile not in MOE_ROUTING_PROFILES:
            raise ValueError(
                f"{aggregate_path}: unknown routing profile {route_profile!r}"
            )
        cell = ProductionMoEPrefillCell(
            normalized_backend,
            cases[case_id],
            m,
            route_profile,
        )
        expected_paths = production_moe_prefill_cell_paths(root, cell)
        if aggregate_path != expected_paths.aggregate or timing_path != expected_paths.timing:
            raise ValueError(
                f"{aggregate_path}: supplemental filename disagrees with cell identity"
            )
        if cell.cell_id in baseline_ids:
            continue
        validate_production_moe_prefill_cell(aggregate_path, timing_path, cell)
        if cell.cell_id in discovered:
            raise ValueError(f"duplicate supplemental cell {cell.cell_id}")
        discovered[cell.cell_id] = cell

    for timing_path in sorted(
        cell_directory.glob(f"{normalized_backend}-*.timing.csv")
    ):
        aggregate_path = timing_path.with_name(
            timing_path.name[:-len(".timing.csv")] + ".csv"
        )
        if not aggregate_path.is_file():
            raise ValueError(
                f"{timing_path}: supplemental timing sidecar has no aggregate"
            )

    cells_by_key: dict[
        MoEProductionOverlayKey,
        list[ProductionMoEPrefillCell],
    ] = {}
    for cell in discovered.values():
        cells_by_key.setdefault(_cell_runtime_key(cell), []).append(cell)
    known_cases = tuple(cases.values())
    for key, actual_cells in cells_by_key.items():
        expected_surfaces = {
            (case.evidence_id, route_profile)
            for case in known_cases
            for route_profile in MOE_ROUTING_PROFILES
            if _cell_runtime_key(ProductionMoEPrefillCell(
                normalized_backend, case, key.m, route_profile
            )) == key
        }
        actual_surfaces = {
            (cell.case.evidence_id, cell.route_profile) for cell in actual_cells
        }
        if actual_surfaces != expected_surfaces:
            raise ValueError(
                f"{key}: supplemental runtime key has incomplete source aliases; "
                f"missing={sorted(expected_surfaces - actual_surfaces)}, "
                f"unexpected={sorted(actual_surfaces - expected_surfaces)}"
            )

    return tuple(sorted(discovered.values(), key=lambda cell: cell.cell_id))


def read_authenticated_moe_surfaces(
    work_directory: Path,
    cells: Sequence[ProductionMoEPrefillCell],
    *,
    backend: str,
) -> tuple[tuple[MoECandidateSurface, MoEProductionOverlayKey], ...]:
    """Validate every cell and return its aggregate candidate surfaces."""

    normalized_backend = backend.strip().lower()
    if normalized_backend not in {"cuda", "rocm"}:
        raise ValueError(f"unsupported production MoE backend {backend!r}")
    records = []
    require_moe_overlay_surface_totality(cells, backend=normalized_backend)
    for cell in cells:
        if cell.backend != normalized_backend:
            raise ValueError(
                f"{normalized_backend} overlay reader received a "
                f"{cell.backend} cell"
            )
        paths = production_moe_prefill_cell_paths(work_directory, cell)
        validate_production_moe_prefill_cell(paths.aggregate, paths.timing, cell)
        with paths.aggregate.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != AGGREGATE_COLUMNS:
                raise ValueError(f"{paths.aggregate}: aggregate schema changed")
            for line, row in enumerate(reader, start=2):
                key = _row_key(row)
                expected_key = _cell_runtime_key(cell)
                if key != expected_key:
                    raise ValueError(
                        f"{paths.aggregate}:{line}: runtime key changed after "
                        "cell authentication"
                    )
                records.append((
                    MoECandidateSurface(
                        surface_id=(
                            f"{cell.case.evidence_id}"
                            f"/route={cell.route_profile}"
                        ),
                        candidate_id=row["candidate_id"],
                        median_us=float(row["median_us"]),
                        p95_us=float(row["p95_us"]),
                        cv=float(row["cv"]),
                        finalist=_explicit_bool("finalist", row["finalist"]),
                    ),
                    key,
                ))
    return tuple(records)


def select_moe_production_overlay_winners(
    records: Iterable[
        tuple[MoECandidateSurface, MoEProductionOverlayKey]
    ],
    *,
    backend: str = "rocm",
    expected_keys: Iterable[MoEProductionOverlayKey] | None = None,
    maximum_surface_regret: float = MAXIMUM_OVERLAY_SURFACE_REGRET,
) -> dict[MoEProductionOverlayKey, MoEProductionOverlayWinner]:
    """Select one robust candidate across every source alias and route profile.

    The exact key cannot encode a prompt's route distribution, so installation
    minimizes worst-surface regret and rejects the key outright when even the
    best common finalist is more than ``maximum_surface_regret`` behind a
    profile-local winner. The generic production policy remains the complete
    dispatch rule for keys that lack certifiable exact evidence.
    """

    normalized_backend = backend.strip().lower()
    expected_candidate_ids = production_moe_prefill_candidate_ids(
        normalized_backend
    )
    expected_candidates = set(expected_candidate_ids)
    if not math.isfinite(maximum_surface_regret) or maximum_surface_regret < 0.0:
        raise ValueError("maximum surface regret must be finite and non-negative")
    grouped: dict[
        MoEProductionOverlayKey,
        dict[str, dict[str, MoECandidateSurface]],
    ] = {}
    for surface, key in records:
        surfaces = grouped.setdefault(key, {})
        candidates = surfaces.setdefault(surface.surface_id, {})
        if surface.candidate_id in candidates:
            raise ValueError(
                f"duplicate candidate {surface.candidate_id} on {surface.surface_id}"
            )
        if surface.candidate_id not in expected_candidates:
            raise ValueError(
                f"unknown {normalized_backend} candidate {surface.candidate_id}"
            )
        candidates[surface.candidate_id] = surface

    if expected_keys is not None:
        expected = set(expected_keys)
        actual = set(grouped)
        if expected != actual:
            raise ValueError(
                "runtime overlay key matrix is incomplete; "
                f"missing={sorted(expected - actual)[:8]}, "
                f"unexpected={sorted(actual - expected)[:8]}"
            )

    winners = {}
    for key, surfaces in sorted(grouped.items()):
        if not surfaces:
            raise ValueError(f"{key}: runtime key has no source surfaces")
        for surface_id, candidates in surfaces.items():
            if set(candidates) != expected_candidates:
                raise ValueError(
                    f"{key}/{surface_id}: incomplete candidate matrix"
                )
        robust_intersection = set.intersection(*(
            {
                candidate_id
                for candidate_id, surface in candidates.items()
                if surface.finalist
            }
            for candidates in surfaces.values()
        ))
        if not robust_intersection:
            raise ValueError(
                f"{key}: no candidate was robustly retimed on every alias surface"
            )

        best_by_surface = {
            surface_id: min(
                surface.median_us
                for surface in candidates.values()
                if surface.finalist
            )
            for surface_id, candidates in surfaces.items()
        }
        scored = []
        for candidate_id in sorted(robust_intersection):
            candidate_surfaces = [
                candidates[candidate_id] for candidates in surfaces.values()
            ]
            regrets = [
                surface.median_us / best_by_surface[surface.surface_id] - 1.0
                for surface in candidate_surfaces
            ]
            scored.append((
                max(regrets),
                _p95(regrets),
                statistics.fmean(regrets),
                statistics.fmean(
                    surface.median_us for surface in candidate_surfaces
                ),
                candidate_id,
                candidate_surfaces,
            ))
        maximum, p95, mean, _, candidate_id, selected_surfaces = min(scored)
        if maximum > maximum_surface_regret:
            raise ValueError(
                f"{key}: best common finalist maximum surface regret "
                f"{maximum:.6f} exceeds {maximum_surface_regret:.6f}"
            )
        gate_tile_m, gate_tile_n, down_tile_m, down_tile_n = (
            _candidate_dimensions(normalized_backend, candidate_id)
        )
        winners[key] = MoEProductionOverlayWinner(
            key=key,
            candidate_id=candidate_id,
            gate_tile_m=gate_tile_m,
            gate_tile_n=gate_tile_n,
            down_tile_m=down_tile_m,
            down_tile_n=down_tile_n,
            surface_count=len(surfaces),
            maximum_surface_regret=maximum,
            p95_surface_regret=p95,
            mean_surface_regret=mean,
            maximum_cv=max(surface.cv for surface in selected_surfaces),
        )
    return winners


def _generate_rocm_moe_production_overlay_include(
    winners: Mapping[
        MoEProductionOverlayKey,
        MoEProductionOverlayWinner,
    ],
    *,
    corpus_digest: str,
) -> str:
    """Render a sorted binary-search exact table for capture-time dispatch."""

    if not winners:
        raise ValueError("cannot generate an empty production MoE overlay")
    lines = [
        "// Auto-generated by moe_production_overlay.py. DO NOT EDIT.",
        "// Exact production pair overlays precede complementary generic rules.",
        f"// Authenticated corpus digest: {corpus_digest}",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace llaminar2::rocm::generated",
        "{",
        "    struct ROCmMoEProductionPrefillOverlayConfig",
        "    {",
        "        uint8_t gateup_tile_m = 0;",
        "        uint16_t gateup_tile_n = 0;",
        "        uint8_t down_tile_m = 0;",
        "        uint16_t down_tile_n = 0;",
        "    };",
        "",
        "    struct ROCmMoEProductionPrefillOverlayEntry",
        "    {",
        "        uint8_t gateup_codebook;",
        "        uint8_t down_codebook;",
        "        int32_t hidden_size;",
        "        int32_t expert_width;",
        "        int32_t expert_count;",
        "        int32_t top_k;",
        "        int32_t m;",
        "        ROCmMoEProductionPrefillOverlayConfig config;",
        "    };",
        "",
        "    inline constexpr ROCmMoEProductionPrefillOverlayEntry",
        "        kROCmMoEProductionPrefillOverlayEntries[] = {",
    ]
    for key in sorted(winners):
        winner = winners[key]
        lines.append(
            "        {"
            f"{key.gateup_codebook}, {key.down_codebook}, "
            f"{key.hidden_size}, {key.expert_width}, {key.expert_count}, "
            f"{key.top_k}, {key.m}, "
            "{"
            f"{winner.gate_tile_m}, {winner.gate_tile_n}, "
            f"{winner.down_tile_m}, {winner.down_tile_n}"
            "}},"
        )
    lines.extend([
        "    };",
        "",
        "    inline int compareROCmMoEProductionPrefillOverlayKey(",
        "        const ROCmMoEProductionPrefillOverlayEntry &entry,",
        "        uint8_t gateup_codebook,",
        "        uint8_t down_codebook,",
        "        int hidden_size,",
        "        int expert_width,",
        "        int expert_count,",
        "        int top_k,",
        "        int m)",
        "    {",
        "        if (entry.gateup_codebook != gateup_codebook)",
        "            return entry.gateup_codebook < gateup_codebook ? -1 : 1;",
        "        if (entry.down_codebook != down_codebook)",
        "            return entry.down_codebook < down_codebook ? -1 : 1;",
        "        if (entry.hidden_size != hidden_size)",
        "            return entry.hidden_size < hidden_size ? -1 : 1;",
        "        if (entry.expert_width != expert_width)",
        "            return entry.expert_width < expert_width ? -1 : 1;",
        "        if (entry.expert_count != expert_count)",
        "            return entry.expert_count < expert_count ? -1 : 1;",
        "        if (entry.top_k != top_k)",
        "            return entry.top_k < top_k ? -1 : 1;",
        "        if (entry.m != m)",
        "            return entry.m < m ? -1 : 1;",
        "        return 0;",
        "    }",
        "",
        "    inline bool selectROCmMoEProductionPrefillOverlay(",
        "        uint8_t gateup_codebook,",
        "        uint8_t down_codebook,",
        "        int hidden_size,",
        "        int expert_width,",
        "        int expert_count,",
        "        int top_k,",
        "        int m,",
        "        ROCmMoEProductionPrefillOverlayConfig &config)",
        "    {",
        "        if (hidden_size <= 0 || expert_width <= 0 ||",
        "            expert_count <= 0 || top_k <= 0 || top_k > expert_count ||",
        "            m <= 0)",
        "        {",
        "            return false;",
        "        }",
        "        size_t first = 0;",
        "        size_t last = sizeof(kROCmMoEProductionPrefillOverlayEntries) /",
        "                      sizeof(kROCmMoEProductionPrefillOverlayEntries[0]);",
        "        while (first < last)",
        "        {",
        "            const size_t middle = first + (last - first) / 2;",
        "            const auto &entry = kROCmMoEProductionPrefillOverlayEntries[middle];",
        "            const int comparison = compareROCmMoEProductionPrefillOverlayKey(",
        "                entry, gateup_codebook, down_codebook, hidden_size,",
        "                expert_width, expert_count, top_k, m);",
        "            if (comparison < 0)",
        "                first = middle + 1;",
        "            else",
        "                last = middle;",
        "        }",
        "        if (first == sizeof(kROCmMoEProductionPrefillOverlayEntries) /",
        "                         sizeof(kROCmMoEProductionPrefillOverlayEntries[0]))",
        "        {",
        "            return false;",
        "        }",
        "        const auto &entry = kROCmMoEProductionPrefillOverlayEntries[first];",
        "        if (compareROCmMoEProductionPrefillOverlayKey(",
        "                entry, gateup_codebook, down_codebook, hidden_size,",
        "                expert_width, expert_count, top_k, m) != 0)",
        "        {",
        "            return false;",
        "        }",
        "        config = entry.config;",
        "        return true;",
        "    }",
        "} // namespace llaminar2::rocm::generated",
        "",
    ])
    return "\n".join(lines)


def _generate_cuda_moe_production_overlay_include(
    winners: Mapping[
        MoEProductionOverlayKey,
        MoEProductionOverlayWinner,
    ],
    *,
    corpus_digest: str,
) -> str:
    """Render CUDA's arithmetic-neutral gate/up block-width overlay."""

    if not winners:
        raise ValueError("cannot generate an empty production MoE overlay")
    lines = [
        "// Auto-generated by moe_production_overlay.py. DO NOT EDIT.",
        "// Exact production geometry precedes the total generic capture policy.",
        f"// Authenticated corpus digest: {corpus_digest}",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace llaminar2::cuda::generated",
        "{",
        "    struct CUDAMoEProductionPrefillOverlayEntry",
        "    {",
        "        uint8_t gateup_codebook;",
        "        uint8_t down_codebook;",
        "        int32_t hidden_size;",
        "        int32_t expert_width;",
        "        int32_t expert_count;",
        "        int32_t top_k;",
        "        int32_t m;",
        "        uint16_t gateup_tile_n;",
        "    };",
        "",
        "    inline constexpr CUDAMoEProductionPrefillOverlayEntry",
        "        kCUDAMoEProductionPrefillOverlayEntries[] = {",
    ]
    for key in sorted(winners):
        winner = winners[key]
        lines.append(
            "        {"
            f"{key.gateup_codebook}, {key.down_codebook}, "
            f"{key.hidden_size}, {key.expert_width}, {key.expert_count}, "
            f"{key.top_k}, {key.m}, {winner.gate_tile_n}"
            "},"
        )
    lines.extend([
        "    };",
        "",
        "    inline int compareCUDAMoEProductionPrefillOverlayKey(",
        "        const CUDAMoEProductionPrefillOverlayEntry &entry,",
        "        uint8_t gateup_codebook,",
        "        uint8_t down_codebook,",
        "        int hidden_size,",
        "        int expert_width,",
        "        int expert_count,",
        "        int top_k,",
        "        int m)",
        "    {",
        "        if (entry.gateup_codebook != gateup_codebook)",
        "            return entry.gateup_codebook < gateup_codebook ? -1 : 1;",
        "        if (entry.down_codebook != down_codebook)",
        "            return entry.down_codebook < down_codebook ? -1 : 1;",
        "        if (entry.hidden_size != hidden_size)",
        "            return entry.hidden_size < hidden_size ? -1 : 1;",
        "        if (entry.expert_width != expert_width)",
        "            return entry.expert_width < expert_width ? -1 : 1;",
        "        if (entry.expert_count != expert_count)",
        "            return entry.expert_count < expert_count ? -1 : 1;",
        "        if (entry.top_k != top_k)",
        "            return entry.top_k < top_k ? -1 : 1;",
        "        if (entry.m != m)",
        "            return entry.m < m ? -1 : 1;",
        "        return 0;",
        "    }",
        "",
        "    inline bool selectCUDAMoEProductionPrefillOverlay(",
        "        uint8_t gateup_codebook,",
        "        uint8_t down_codebook,",
        "        int hidden_size,",
        "        int expert_width,",
        "        int expert_count,",
        "        int top_k,",
        "        int m,",
        "        int &gateup_tile_n)",
        "    {",
        "        if (hidden_size <= 0 || expert_width <= 0 ||",
        "            expert_count <= 0 || top_k <= 0 || top_k > expert_count ||",
        "            m <= 0)",
        "        {",
        "            return false;",
        "        }",
        "        size_t first = 0;",
        "        size_t last = sizeof(kCUDAMoEProductionPrefillOverlayEntries) /",
        "                      sizeof(kCUDAMoEProductionPrefillOverlayEntries[0]);",
        "        while (first < last)",
        "        {",
        "            const size_t middle = first + (last - first) / 2;",
        "            const auto &entry = kCUDAMoEProductionPrefillOverlayEntries[middle];",
        "            const int comparison = compareCUDAMoEProductionPrefillOverlayKey(",
        "                entry, gateup_codebook, down_codebook, hidden_size,",
        "                expert_width, expert_count, top_k, m);",
        "            if (comparison < 0)",
        "                first = middle + 1;",
        "            else",
        "                last = middle;",
        "        }",
        "        if (first == sizeof(kCUDAMoEProductionPrefillOverlayEntries) /",
        "                         sizeof(kCUDAMoEProductionPrefillOverlayEntries[0]))",
        "        {",
        "            return false;",
        "        }",
        "        const auto &entry = kCUDAMoEProductionPrefillOverlayEntries[first];",
        "        if (compareCUDAMoEProductionPrefillOverlayKey(",
        "                entry, gateup_codebook, down_codebook, hidden_size,",
        "                expert_width, expert_count, top_k, m) != 0)",
        "        {",
        "            return false;",
        "        }",
        "        gateup_tile_n = entry.gateup_tile_n;",
        "        return true;",
        "    }",
        "} // namespace llaminar2::cuda::generated",
        "",
    ])
    return "\n".join(lines)


def generate_moe_production_overlay_include(
    winners: Mapping[
        MoEProductionOverlayKey,
        MoEProductionOverlayWinner,
    ],
    *,
    backend: str = "rocm",
    corpus_digest: str,
) -> str:
    """Render one backend's sorted capture-time exact-overlay table."""

    if not winners:
        raise ValueError("cannot generate an empty production MoE overlay")
    normalized_backend = backend.strip().lower()
    if normalized_backend == "rocm":
        return _generate_rocm_moe_production_overlay_include(
            winners, corpus_digest=corpus_digest
        )
    if normalized_backend == "cuda":
        return _generate_cuda_moe_production_overlay_include(
            winners, corpus_digest=corpus_digest
        )
    raise ValueError(f"unsupported production MoE backend {backend!r}")


def write_moe_production_overlay_summary(
    path: Path,
    winners: Mapping[
        MoEProductionOverlayKey,
        MoEProductionOverlayWinner,
    ],
) -> None:
    """Publish a reviewable winner/regret summary beside generated C++."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "gateup_codebook", "down_codebook", "hidden_size",
            "expert_width", "expert_count", "top_k", "m", "candidate_id",
            "gate_tile_m", "gate_tile_n", "down_tile_m", "down_tile_n",
            "surface_count", "maximum_surface_regret",
            "p95_surface_regret", "mean_surface_regret", "maximum_cv",
        ])
        for key in sorted(winners):
            winner = winners[key]
            writer.writerow([
                key.gateup_codebook,
                key.down_codebook,
                key.hidden_size,
                key.expert_width,
                key.expert_count,
                key.top_k,
                key.m,
                winner.candidate_id,
                winner.gate_tile_m,
                winner.gate_tile_n,
                winner.down_tile_m,
                winner.down_tile_n,
                winner.surface_count,
                f"{winner.maximum_surface_regret:.9f}",
                f"{winner.p95_surface_regret:.9f}",
                f"{winner.mean_surface_regret:.9f}",
                f"{winner.maximum_cv:.9f}",
            ])


def main() -> int:
    """Validate one backend's complete selected surfaces and emit its overlay."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "rocm"), required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--m", type=int, action="append", default=[])
    args = parser.parse_args()

    baseline_cells = production_moe_prefill_cells(args.backend)
    if args.case:
        selected_cases = set(args.case)
        baseline_cells = tuple(
            cell for cell in baseline_cells
            if cell.case.evidence_id in selected_cases
        )
    if args.m:
        if any(m <= 0 for m in args.m):
            raise ValueError("selected overlay M values must be positive")
        selected_m = set(args.m)
        baseline_cells = tuple(
            cell for cell in baseline_cells if cell.m in selected_m
        )
    if not baseline_cells:
        raise ValueError("overlay filters selected no baseline cells")
    require_moe_overlay_surface_totality(
        baseline_cells, backend=args.backend
    )
    additive_cells = discover_additive_moe_production_cells(
        args.work_dir,
        backend=args.backend,
        baseline_cells=baseline_cells,
    )
    cells = baseline_cells + additive_cells
    records = read_authenticated_moe_surfaces(
        args.work_dir, cells, backend=args.backend
    )
    expected_keys = {_cell_runtime_key(cell) for cell in cells}
    winners = select_moe_production_overlay_winners(
        records, backend=args.backend, expected_keys=expected_keys
    )
    evidence_paths = []
    for cell in cells:
        paths = production_moe_prefill_cell_paths(args.work_dir, cell)
        evidence_paths.extend((paths.aggregate, paths.timing))
    corpus_digest = raw_corpus_id(evidence_paths)
    generated = generate_moe_production_overlay_include(
        winners, backend=args.backend, corpus_digest=corpus_digest
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_name(args.output.name + ".inprogress")
    staging.write_text(generated, encoding="utf-8")
    staging.replace(args.output)
    write_moe_production_overlay_summary(args.summary_csv, winners)
    print(
        f"generated {len(winners)} exact {args.backend.upper()} "
        "production-MoE overlays from "
        f"{len(baseline_cells)} mandatory and "
        f"{len(additive_cells)} additive authenticated cells"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
