"""Generate dense GPU prefill exact overlays from authenticated sweep cells.

Every source format is measured independently, while production dispatch sees
the prepared execution codebook. This generator therefore pools source aliases
only after validating each cell and chooses the candidate with the lowest
geometric-mean normalized regret across those aliases. The selected candidate
must resolve to one concrete physical launch tuple for the whole runtime key;
an AUTO label is never emitted into production policy.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .production_dense_prefill_sweep import (
    DEFAULT_BENCH_RUNS,
    DEFAULT_WARMUP_RUNS,
    SWEEP_PLAN_FILENAME,
    DensePrefillCell,
    _cuda_candidate_is_spill_free,
    _read_aggregate_rows,
    _read_timing_rows,
    dense_prefill_candidate_ids,
    load_planned_dense_prefill_cells,
    load_sweep_plan,
    production_dense_prefill_cell_paths,
    validate_promoted_production_dense_prefill_cell,
)


@dataclass(frozen=True, order=True)
class DenseOverlayKey:
    """One runtime-visible exact dense prefill dispatch key."""

    execution_codebook: int
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class DenseOverlayEntry:
    """One pooled exact winner and its concrete backend launch tuple."""

    backend: str
    key: DenseOverlayKey
    source_formats: tuple[str, ...]
    shape_names: tuple[str, ...]
    candidate_id: str
    launch: tuple[int, ...]
    geometric_mean_regret: float
    maximum_alias_regret: float
    geometric_mean_speedup_vs_auto: float


def _expected_overlay_inventory(
    cells: Sequence[DensePrefillCell],
    backend: str,
) -> dict[DenseOverlayKey, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Collapse source cells into the exact runtime-visible key inventory."""

    grouped: dict[DenseOverlayKey, tuple[set[str], set[str]]] = {}
    for cell in cells:
        if cell.backend != backend:
            raise ValueError(
                "dense overlay totality inventory contains a foreign backend"
            )
        key = DenseOverlayKey(
            execution_codebook=cell.source_format.runtime_codebook(backend),
            m=cell.m,
            n=cell.shape.n,
            k=cell.shape.k,
        )
        source_formats, shape_names = grouped.setdefault(
            key, (set(), set())
        )
        source_formats.add(cell.source_format.label)
        shape_names.add(cell.shape.name)
    return {
        key: (tuple(sorted(source_formats)), tuple(sorted(shape_names)))
        for key, (source_formats, shape_names) in grouped.items()
    }


def validate_dense_overlay_totality(
    entries: Sequence[DenseOverlayEntry],
    cells: Sequence[DensePrefillCell],
    backend: str,
) -> None:
    """Prove exact-key totality, alias coverage, and launch eligibility.

    Exact overlays are additive to generic dispatch, but every cell named by an
    immutable sweep plan must install exactly one runtime key.  This gate also
    checks that CUDA winners remain inside the compiler-proven no-spill launch
    inventory.  It runs before publication, making a partial or stale include
    structurally impossible to install through the turnkey generator.
    """

    normalized = backend.strip().lower()
    expected = _expected_overlay_inventory(cells, normalized)
    observed: dict[DenseOverlayKey, DenseOverlayEntry] = {}
    for entry in entries:
        if entry.backend != normalized:
            raise ValueError(
                "dense overlay totality received a foreign backend entry"
            )
        if entry.key in observed:
            raise ValueError(f"duplicate dense overlay key: {entry.key}")
        observed[entry.key] = entry

    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "dense overlay is not total for its immutable sweep plan: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    if tuple(entry.key for entry in entries) != tuple(sorted(expected)):
        raise ValueError("dense overlay keys are not strictly sorted")

    for key, entry in observed.items():
        expected_formats, expected_shapes = expected[key]
        if entry.source_formats != expected_formats:
            raise ValueError(
                f"{key}: source alias coverage mismatch: "
                f"expected={expected_formats} observed={entry.source_formats}"
            )
        if entry.shape_names != expected_shapes:
            raise ValueError(
                f"{key}: exact geometry-name coverage mismatch: "
                f"expected={expected_shapes} observed={entry.shape_names}"
            )

        if normalized != "cuda":
            continue
        tile_id, k_partitions, bk256, canonical_kpart = entry.launch
        if bk256:
            if (
                key.execution_codebook != 0
                or tile_id not in {-3, -2}
                or k_partitions != 1
                or canonical_kpart
            ):
                raise ValueError(f"{key}: invalid CUDA BK256 exact winner")
            continue
        if tile_id not in range(6):
            raise ValueError(f"{key}: invalid CUDA BK64 tile {tile_id}")
        if canonical_kpart:
            if k_partitions <= 1:
                raise ValueError(
                    f"{key}: canonical CUDA winner has no K partitioning"
                )
            candidate = f"KPART:t{tile_id}"
        else:
            if k_partitions != 1:
                raise ValueError(
                    f"{key}: direct CUDA winner has an invalid partition count"
                )
            candidate = f"STD:t{tile_id}:full"
        if not _cuda_candidate_is_spill_free(
            key.execution_codebook, candidate
        ):
            raise ValueError(
                f"{key}: CUDA exact winner {candidate} is compiler-proven "
                "to spill and is ineligible for production"
            )


def _launch_tuple(
    backend: str,
    row: Mapping[str, str],
) -> tuple[int, ...]:
    if backend == "cuda":
        return (
            int(row["observed_tile_id"]),
            int(row["observed_k_partitions"]),
            int(row["observed_bk256"]),
            int(row["observed_canonical_kpart"]),
        )
    return (
        int(row["observed_n_tile"]),
        int(row["observed_m_tile"]),
        int(row["observed_min_blocks"]),
        int(row["observed_unroll"]),
        int(row["observed_full_tiles"]),
    )


def _geometric_mean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("geometric mean requires finite positive observations")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def collect_dense_overlay_entries(
    root: Path,
    backend: str,
    *,
    cells: Sequence[DensePrefillCell] | None = None,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    bench_runs: int = DEFAULT_BENCH_RUNS,
) -> tuple[DenseOverlayEntry, ...]:
    """Validate cells and pool source aliases into runtime dispatch entries."""

    normalized = backend.strip().lower()
    if cells is None:
        selected_cells, plan = load_planned_dense_prefill_cells(
            Path(root) / SWEEP_PLAN_FILENAME,
            normalized,
        )
        if (
            int(plan["warmup_runs"]) != warmup_runs
            or int(plan["bench_runs"]) != bench_runs
        ):
            raise ValueError(
                "overlay timing cardinality disagrees with the sweep plan"
            )
    else:
        selected_cells = tuple(cells)
        plan = load_sweep_plan(
            Path(root) / SWEEP_PLAN_FILENAME,
            normalized,
            selected_cells,
            warmup_runs=warmup_runs,
            bench_runs=bench_runs,
        )
    if not selected_cells or any(
        cell.backend != normalized for cell in selected_cells
    ):
        raise ValueError("overlay input must contain one non-empty backend")
    if len({cell.identity for cell in selected_cells}) != len(selected_cells):
        raise ValueError("overlay input contains duplicate dense prefill cells")

    grouped: dict[
        DenseOverlayKey,
        list[tuple[
            DensePrefillCell,
            dict[str, dict[str, str]],
            dict[str, tuple[float, ...]],
        ]],
    ] = {}
    for cell in selected_cells:
        paths = production_dense_prefill_cell_paths(root, cell)
        validate_promoted_production_dense_prefill_cell(
            root,
            cell,
            plan,
        )
        aggregate = _read_aggregate_rows(paths.aggregate, cell)
        timing = _read_timing_rows(
            paths.timing, cell, bench_runs, aggregate.keys()
        )
        key = DenseOverlayKey(
            execution_codebook=cell.source_format.runtime_codebook(normalized),
            m=cell.m,
            n=cell.shape.n,
            k=cell.shape.k,
        )
        grouped.setdefault(key, []).append((cell, aggregate, timing))

    result = []
    for key in sorted(grouped):
        observations = grouped[key]
        source_formats = tuple(sorted(
            cell.source_format.label for cell, _, _ in observations
        ))
        if len(source_formats) != len(set(source_formats)):
            raise ValueError(
                f"{key}: duplicate source alias in runtime overlay group"
            )
        shape_names = tuple(sorted({
            cell.shape.name for cell, _, _ in observations
        }))
        # The aggregate parser has already authenticated the exact launchable
        # candidate set, including the geometry-dependent canonical-KPART
        # availability marker. Reconstructing a format-only inventory here
        # would reintroduce KPART candidates for shapes whose K geometry yields
        # only one partition and has no corresponding timing evidence.
        candidate_sets = [
            set(aggregate)
            for _, aggregate, _ in observations
        ]
        common_candidates = set.intersection(*candidate_sets)
        if not common_candidates:
            raise ValueError(f"{key}: source aliases share no launch candidate")

        median_by_source: list[dict[str, float]] = []
        aggregate_by_source: list[dict[str, dict[str, str]]] = []
        for _, aggregate, timing in observations:
            medians = {
                candidate: statistics.median(timing[candidate])
                for candidate in common_candidates
            }
            median_by_source.append(medians)
            aggregate_by_source.append(aggregate)

        scores: dict[str, tuple[float, float]] = {}
        for candidate in common_candidates:
            regrets = []
            for medians in median_by_source:
                best = min(medians.values())
                regrets.append(medians[candidate] / best)
            scores[candidate] = (
                _geometric_mean(regrets),
                max(regrets),
            )
        winner = min(
            common_candidates,
            key=lambda candidate: (
                scores[candidate][0],
                scores[candidate][1],
                candidate,
            ),
        )

        launch_tuples = {
            _launch_tuple(normalized, aggregate[winner])
            for aggregate in aggregate_by_source
        }
        if len(launch_tuples) != 1:
            raise ValueError(
                f"{key}: winner {winner} resolved to conflicting physical "
                f"launches across {source_formats}: {sorted(launch_tuples)}"
            )
        launch = next(iter(launch_tuples))
        auto_speedups = []
        for medians in median_by_source:
            if "AUTO" in medians:
                auto = medians["AUTO"]
            elif "Auto" in medians:
                auto = medians["Auto"]
            else:
                raise ValueError(f"{key}: candidate inventory has no AUTO baseline")
            auto_speedups.append(auto / medians[winner])

        result.append(DenseOverlayEntry(
            backend=normalized,
            key=key,
            source_formats=source_formats,
            shape_names=shape_names,
            candidate_id=winner,
            launch=launch,
            geometric_mean_regret=scores[winner][0] - 1.0,
            maximum_alias_regret=scores[winner][1] - 1.0,
            geometric_mean_speedup_vs_auto=_geometric_mean(auto_speedups),
        ))
    entries = tuple(result)
    validate_dense_overlay_totality(
        entries,
        selected_cells,
        normalized,
    )
    return entries


def _render_cuda(entries: Sequence[DenseOverlayEntry]) -> str:
    rows = []
    for entry in entries:
        tile_id, k_partitions, bk256, canonical_kpart = entry.launch
        if (
            bk256 not in {0, 1}
            or canonical_kpart not in {0, 1}
            or (canonical_kpart == 0 and k_partitions != 1)
            or (
                canonical_kpart == 1
                and (k_partitions <= 1 or bk256 != 0)
            )
        ):
            raise ValueError(
                f"{entry.key}: CUDA overlay is not serial-row-equivalent"
            )
        rows.append(
            "        {"
            f"{entry.key.execution_codebook}, {entry.key.m}, {entry.key.n}, "
            f"{entry.key.k}, {{{tile_id}, {str(bool(bk256)).lower()}, "
            f"{str(bool(canonical_kpart)).lower()}}}"
            "},"
        )
    return "\n".join([
        "// Auto-generated by dense_production_overlay.py. DO NOT EDIT.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace llaminar2::cuda::generated",
        "{",
        "    struct CUDADensePrefillOverlayConfig",
        "    {",
        "        int tile_id;",
        "        bool bk256;",
        "        bool canonical_kpart;",
        "    };",
        "",
        "    struct CUDADensePrefillOverlayEntry",
        "    {",
        "        uint8_t codebook;",
        "        int m;",
        "        int n;",
        "        int k;",
        "        CUDADensePrefillOverlayConfig config;",
        "    };",
        "",
        "    inline constexpr CUDADensePrefillOverlayEntry",
        "        kCUDADensePrefillOverlayEntries[] = {",
        *rows,
        "    };",
        "",
        "    inline int compareCUDADensePrefillOverlayKey(",
        "        const CUDADensePrefillOverlayEntry &entry,",
        "        uint8_t codebook, int m, int n, int k)",
        "    {",
        "        if (entry.codebook != codebook)",
        "            return entry.codebook < codebook ? -1 : 1;",
        "        if (entry.m != m)",
        "            return entry.m < m ? -1 : 1;",
        "        if (entry.n != n)",
        "            return entry.n < n ? -1 : 1;",
        "        if (entry.k != k)",
        "            return entry.k < k ? -1 : 1;",
        "        return 0;",
        "    }",
        "",
        "    inline bool selectCUDADensePrefillOverlay(",
        "        uint8_t codebook, int m, int n, int k,",
        "        CUDADensePrefillOverlayConfig &config)",
        "    {",
        "        size_t first = 0;",
        "        size_t last = sizeof(kCUDADensePrefillOverlayEntries) /",
        "                      sizeof(kCUDADensePrefillOverlayEntries[0]);",
        "        while (first < last)",
        "        {",
        "            const size_t middle = first + (last - first) / 2;",
        "            const auto &entry = kCUDADensePrefillOverlayEntries[middle];",
        "            const int comparison = compareCUDADensePrefillOverlayKey(",
        "                entry, codebook, m, n, k);",
        "            if (comparison < 0)",
        "                first = middle + 1;",
        "            else",
        "                last = middle;",
        "        }",
        "        constexpr size_t count =",
        "            sizeof(kCUDADensePrefillOverlayEntries) /",
        "            sizeof(kCUDADensePrefillOverlayEntries[0]);",
        "        if (first == count)",
        "            return false;",
        "        const auto &entry = kCUDADensePrefillOverlayEntries[first];",
        "        if (compareCUDADensePrefillOverlayKey(",
        "                entry, codebook, m, n, k) != 0)",
        "            return false;",
        "        config = entry.config;",
        "        return true;",
        "    }",
        "}",
        "",
    ])


def _render_rocm(entries: Sequence[DenseOverlayEntry]) -> str:
    rows = [
        "        {"
        f"{entry.key.execution_codebook}, {entry.key.m}, {entry.key.n}, "
        f"{entry.key.k}, {{{entry.launch[0]}, {entry.launch[1]}, "
        f"{entry.launch[2]}, {entry.launch[3]}, "
        f"{str(bool(entry.launch[4])).lower()}}}"
        "},"
        for entry in entries
    ]
    return "\n".join([
        "// Auto-generated by dense_production_overlay.py. DO NOT EDIT.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace llaminar2::rocm::generated",
        "{",
        "    struct ROCmDensePrefillOverlayConfig",
        "    {",
        "        int n_tile;",
        "        int m_tile;",
        "        int min_blocks;",
        "        int unroll;",
        "        bool full_tiles;",
        "    };",
        "",
        "    struct ROCmDensePrefillOverlayEntry",
        "    {",
        "        uint8_t codebook;",
        "        int m;",
        "        int n;",
        "        int k;",
        "        ROCmDensePrefillOverlayConfig config;",
        "    };",
        "",
        "    inline constexpr ROCmDensePrefillOverlayEntry",
        "        kROCmDensePrefillOverlayEntries[] = {",
        *rows,
        "    };",
        "",
        "    inline int compareROCmDensePrefillOverlayKey(",
        "        const ROCmDensePrefillOverlayEntry &entry,",
        "        uint8_t codebook, int m, int n, int k)",
        "    {",
        "        if (entry.codebook != codebook)",
        "            return entry.codebook < codebook ? -1 : 1;",
        "        if (entry.m != m)",
        "            return entry.m < m ? -1 : 1;",
        "        if (entry.n != n)",
        "            return entry.n < n ? -1 : 1;",
        "        if (entry.k != k)",
        "            return entry.k < k ? -1 : 1;",
        "        return 0;",
        "    }",
        "",
        "    inline bool selectROCmDensePrefillOverlay(",
        "        uint8_t codebook, int m, int n, int k,",
        "        ROCmDensePrefillOverlayConfig &config)",
        "    {",
        "        size_t first = 0;",
        "        size_t last = sizeof(kROCmDensePrefillOverlayEntries) /",
        "                      sizeof(kROCmDensePrefillOverlayEntries[0]);",
        "        while (first < last)",
        "        {",
        "            const size_t middle = first + (last - first) / 2;",
        "            const auto &entry = kROCmDensePrefillOverlayEntries[middle];",
        "            const int comparison = compareROCmDensePrefillOverlayKey(",
        "                entry, codebook, m, n, k);",
        "            if (comparison < 0)",
        "                first = middle + 1;",
        "            else",
        "                last = middle;",
        "        }",
        "        constexpr size_t count =",
        "            sizeof(kROCmDensePrefillOverlayEntries) /",
        "            sizeof(kROCmDensePrefillOverlayEntries[0]);",
        "        if (first == count)",
        "            return false;",
        "        const auto &entry = kROCmDensePrefillOverlayEntries[first];",
        "        if (compareROCmDensePrefillOverlayKey(",
        "                entry, codebook, m, n, k) != 0)",
        "            return false;",
        "        config = entry.config;",
        "        return true;",
        "    }",
        "}",
        "",
    ])


def render_dense_overlay(entries: Sequence[DenseOverlayEntry]) -> str:
    """Render one backend's sorted, concrete C++ exact-overlay table."""

    if not entries:
        raise ValueError("cannot render an empty dense prefill overlay")
    backend = entries[0].backend
    if any(entry.backend != backend for entry in entries):
        raise ValueError("cannot render mixed-backend overlay entries")
    if tuple(sorted(entries, key=lambda entry: entry.key)) != tuple(entries):
        raise ValueError("dense prefill overlay entries must be sorted")
    return _render_cuda(entries) if backend == "cuda" else _render_rocm(entries)


def write_dense_overlay(
    output: Path,
    entries: Sequence[DenseOverlayEntry],
) -> None:
    """Atomically publish one generated include after complete validation."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".inprogress")
    encoded = render_dense_overlay(entries)
    with staging.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, output)


def write_overlay_summary(
    output: Path,
    entries: Sequence[DenseOverlayEntry],
) -> None:
    """Publish reviewable winner/regret evidence alongside generated policy."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".inprogress")
    columns = (
        "backend", "execution_codebook", "source_formats", "shape_names",
        "m", "n", "k", "candidate_id", "launch",
        "geometric_mean_regret", "maximum_alias_regret",
        "geometric_mean_speedup_vs_auto",
    )
    with staging.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for entry in entries:
            writer.writerow({
                "backend": entry.backend,
                "execution_codebook": entry.key.execution_codebook,
                "source_formats": "/".join(entry.source_formats),
                "shape_names": "/".join(entry.shape_names),
                "m": entry.key.m,
                "n": entry.key.n,
                "k": entry.key.k,
                "candidate_id": entry.candidate_id,
                "launch": "/".join(str(value) for value in entry.launch),
                "geometric_mean_regret": f"{entry.geometric_mean_regret:.9f}",
                "maximum_alias_regret": f"{entry.maximum_alias_regret:.9f}",
                "geometric_mean_speedup_vs_auto": (
                    f"{entry.geometric_mean_speedup_vs_auto:.9f}"
                ),
            })
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, output)


def main() -> int:
    """Generate an installable include only from a complete corpus."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "rocm"), required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--bench-runs", type=int, default=DEFAULT_BENCH_RUNS)
    args = parser.parse_args()
    entries = collect_dense_overlay_entries(
        args.work_dir,
        args.backend,
        warmup_runs=args.warmup_runs,
        bench_runs=args.bench_runs,
    )
    write_dense_overlay(args.output, entries)
    write_overlay_summary(args.summary_csv, entries)
    print(f"installed {len(entries)} {args.backend} dense prefill overlays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
