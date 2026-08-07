#!/usr/bin/env python3
"""Regressions for the resumable all-format dense GPU prefill sweep."""

from __future__ import annotations

import csv
import math
import os
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_TOOLS = REPO_ROOT / "tests/v2/performance/kernels"
if str(KERNEL_TOOLS) not in os.sys.path:
    os.sys.path.insert(0, str(KERNEL_TOOLS))

from native_vnni_dispatch.format_registry import FORMAT_SPECS  # noqa: E402
from native_vnni_dispatch.dense_production_overlay import (  # noqa: E402
    DenseOverlayEntry,
    DenseOverlayKey,
    collect_dense_overlay_entries,
    render_dense_overlay,
)
from native_vnni_dispatch.prefill_matrix import (  # noqa: E402
    GPU_PREFILL_M_BUCKETS,
    gpu_prefill_measurements,
)
from native_vnni_dispatch.production_dense_prefill_sweep import (  # noqa: E402
    CUDA_AGGREGATE_COLUMNS,
    CUDA_CANDIDATE_IDS,
    CUDA_GENERIC_CANDIDATE_IDS,
    CUDA_TILE_NAMES,
    CUDA_TIMING_COLUMNS,
    ROCM_AGGREGATE_COLUMNS,
    ROCM_CANDIDATE_IDS,
    ROCM_Q6_FULL_TILE_CANDIDATE_IDS,
    ROCM_Q6_SPILLING_CHECKED_CANDIDATE_IDS,
    ROCM_TIMING_COLUMNS,
    _cell_environment,
    combine_production_dense_prefill_cells,
    dense_prefill_candidate_ids,
    production_dense_prefill_cell_paths,
    production_dense_prefill_cells,
    run_missing_accelerator_cells,
    validate_production_dense_prefill_cell,
    write_sweep_plan,
)


_UINT64_MAX = (1 << 64) - 1


def _cuda_launch(candidate: str) -> dict[str, object]:
    if candidate == "AUTO":
        return {
            "tile": "AUTO",
            "tile_id": -1,
            "strategy": "AUTO",
            "requested_k_partitions": 0,
        }
    strategy, remainder = candidate.split(":", maxsplit=1)
    if strategy == "STD":
        tile_text, full_text = remainder.split(":")
        tile_id = int(tile_text.removeprefix("t"))
        if full_text != "full":
            raise ValueError(f"unexpected STD policy {candidate}")
        return {
            "tile": CUDA_TILE_NAMES[tile_id],
            "tile_id": tile_id,
            "strategy": strategy,
            "requested_k_partitions": 1,
        }
    if strategy == "KPART":
        tile_id = int(remainder.removeprefix("t"))
        return {
            "tile": CUDA_TILE_NAMES[tile_id],
            "tile_id": tile_id,
            "strategy": strategy,
            "requested_k_partitions": 0,
        }
    return {
        "tile": "BK256_128x128",
        "tile_id": -2,
        "strategy": strategy,
        "requested_k_partitions": 1,
    }


def _write_cell(
    root: Path,
    backend: str,
    *,
    bench_runs: int = 3,
    cell=None,
    canonical_kpart_available: bool = True,
):
    cell = cell or production_dense_prefill_cells(backend)[0]
    paths = production_dense_prefill_cell_paths(root, cell)
    paths.aggregate.parent.mkdir(parents=True, exist_ok=True)
    candidates = dense_prefill_candidate_ids(
        backend,
        cell.source_format,
        canonical_kpart_available=canonical_kpart_available,
    )
    samples_by_candidate = {
        candidate: tuple(
            10.0 + candidate_index + sample_index * 0.125
            for sample_index in range(bench_runs)
        )
        for candidate_index, candidate in enumerate(candidates)
    }

    if backend == "cuda":
        aggregate_columns = CUDA_AGGREGATE_COLUMNS
        timing_columns = CUDA_TIMING_COLUMNS
    else:
        aggregate_columns = ROCM_AGGREGATE_COLUMNS
        timing_columns = ROCM_TIMING_COLUMNS

    with paths.aggregate.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_columns)
        writer.writeheader()
        for candidate_index, candidate in enumerate(candidates):
            samples = samples_by_candidate[candidate]
            mean = statistics.fmean(samples)
            if backend == "cuda":
                launch = _cuda_launch(candidate)
                writer.writerow({
                    "format": cell.source_format.label,
                    "codebook": cell.source_format.source_codebook_id,
                    "shape": cell.shape.name,
                    "m": cell.m,
                    "n": cell.shape.n,
                    "k": cell.shape.k,
                    **launch,
                    "tiles": 1,
                    "min_us": f"{min(samples):.3f}",
                    "mean_us": f"{mean:.3f}",
                    "tops": "1.0000",
                    "pct_peak": "1.00",
                    "gpu": 0,
                    "byte_mismatches": 0,
                    "first_byte_mismatch": _UINT64_MAX,
                    "correctness_pass": 1,
                    "observed_tile_id": (
                        -3 if candidate == "BK256:full"
                        else 1 if candidate == "AUTO"
                        else int(candidate.split(":")[1][1:])
                    ),
                    "observed_k_partitions": (
                        4 if candidate.startswith("KPART:") else 1
                    ),
                    "observed_bk256": int(candidate == "BK256:full"),
                    "observed_canonical_kpart": int(
                        candidate.startswith("KPART:")
                    ),
                    "canonical_kpart_available": int(
                        canonical_kpart_available
                    ),
                })
            else:
                stddev = math.sqrt(statistics.fmean(
                    (sample - mean) ** 2 for sample in samples
                ))
                components = candidate.split("/")
                explicit_unroll = (
                    int(components[3].removeprefix("U"))
                    if len(components) >= 4 and components[3].startswith("U")
                    else 4
                )
                writer.writerow({
                    "backend": "rocm",
                    "phase": "prefill",
                    "format": cell.source_format.label,
                    "codebook": cell.source_format.source_codebook_id,
                    "shape": cell.shape.name,
                    "category": cell.shape.aspect_bucket.value,
                    "m": cell.m,
                    "n": cell.shape.n,
                    "k": cell.shape.k,
                    "variant": candidate,
                    "min_us": f"{min(samples):.3f}",
                    "mean_us": f"{mean:.3f}",
                    "stddev_us": f"{stddev:.3f}",
                    "gflops": "1.000",
                    "cosine": "1.000000",
                    "correctness_pass": 1,
                    "byte_mismatches_vs_auto": 0,
                    "first_byte_mismatch_vs_auto": _UINT64_MAX,
                    "is_best": int(candidate_index == 0),
                    "observed_n_tile": (
                        128 if candidate == "Auto"
                        else int(candidate.split("/")[0].removeprefix("N"))
                    ),
                    "observed_m_tile": (
                        32 if candidate == "Auto"
                        else int(candidate.split("/")[1].removeprefix("MT"))
                    ),
                    "observed_min_blocks": (
                        1 if candidate == "Auto"
                        else int(candidate.split("/")[2].removeprefix("MB"))
                    ),
                    "observed_unroll": (
                        2 if candidate == "Auto"
                        else explicit_unroll
                    ),
                    "observed_full_tiles": int(
                        candidate != "Auto" and components[-1] == "FULL"
                    ),
                    "registers_per_thread": 96,
                    "local_memory_bytes_per_thread": 0,
                    "static_shared_memory_bytes": 16384,
                    "max_threads_per_block": 256,
                    "max_active_blocks_per_sm": 2,
                })

    with paths.timing.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=timing_columns)
        writer.writeheader()
        for candidate in candidates:
            for sample_index, latency_us in enumerate(
                samples_by_candidate[candidate]
            ):
                common = {
                    "backend": backend,
                    "phase": "prefill",
                    "format": cell.source_format.label,
                    "codebook": cell.source_format.source_codebook_id,
                    "shape": cell.shape.name,
                    "m": cell.m,
                    "n": cell.shape.n,
                    "k": cell.shape.k,
                    "sample_index": sample_index,
                    "timed_replays": 1,
                    "latency_us": f"{latency_us:.9f}",
                    "latency_us_hex": latency_us.hex(),
                }
                if backend == "cuda":
                    writer.writerow({**common, **_cuda_launch(candidate)})
                else:
                    writer.writerow({**common, "variant": candidate})
    paths.log.write_text("complete\n", encoding="utf-8")
    return cell, paths


class ProductionDensePrefillSweepTest(unittest.TestCase):
    """Keep the dense corpus comprehensive, exact, and safely resumable."""

    def test_matrix_is_all_format_shape_and_m_complete_on_both_backends(self) -> None:
        expected_count = (
            len(FORMAT_SPECS)
            * len(gpu_prefill_measurements())
            * len(GPU_PREFILL_M_BUCKETS)
        )
        self.assertEqual(expected_count, 33075)
        for backend in ("cuda", "rocm"):
            with self.subTest(backend=backend):
                cells = production_dense_prefill_cells(backend)
                self.assertEqual(len(cells), expected_count)
                self.assertEqual(len({cell.cell_id for cell in cells}), len(cells))
                self.assertEqual(
                    {cell.source_format.label for cell in cells},
                    {spec.label for spec in FORMAT_SPECS},
                )
                self.assertEqual(
                    {cell.shape.name for cell in cells},
                    {item.shape.name for item in gpu_prefill_measurements()},
                )
                self.assertEqual({cell.m for cell in cells}, set(GPU_PREFILL_M_BUCKETS))
                self.assertFalse(any("LM_Head" in cell.shape.name for cell in cells))

    def test_candidate_inventories_keep_only_byte_eligible_cuda_routes(self) -> None:
        self.assertEqual(dense_prefill_candidate_ids("cuda"), CUDA_CANDIDATE_IDS)
        self.assertEqual(len(CUDA_CANDIDATE_IDS), 14)
        self.assertEqual(len(CUDA_GENERIC_CANDIDATE_IDS), 13)
        self.assertFalse(any(candidate.startswith("SK1") for candidate in CUDA_CANDIDATE_IDS))
        self.assertFalse(any(candidate.startswith("SK2") for candidate in CUDA_CANDIDATE_IDS))
        self.assertEqual(
            sum(candidate.startswith("KPART:") for candidate in CUDA_CANDIDATE_IDS),
            len(CUDA_TILE_NAMES),
        )
        self.assertEqual(dense_prefill_candidate_ids("rocm"), ROCM_CANDIDATE_IDS)
        self.assertEqual(len(ROCM_CANDIDATE_IDS), 27)
        q6 = next(spec for spec in FORMAT_SPECS if spec.label == "Q6_K")
        q6_checked = tuple(
            candidate for candidate in ROCM_CANDIDATE_IDS[:-1]
            if candidate not in ROCM_Q6_SPILLING_CHECKED_CANDIDATE_IDS
        )
        self.assertEqual(
            dense_prefill_candidate_ids("rocm", q6),
            q6_checked
            + ROCM_Q6_FULL_TILE_CANDIDATE_IDS
            + ROCM_CANDIDATE_IDS[-1:],
        )
        self.assertEqual(len(ROCM_Q6_SPILLING_CHECKED_CANDIDATE_IDS), 10)
        self.assertTrue(
            ROCM_Q6_SPILLING_CHECKED_CANDIDATE_IDS.isdisjoint(
                dense_prefill_candidate_ids("rocm", q6)
            )
        )

    def test_both_backends_authenticate_aggregate_against_every_sample(self) -> None:
        for backend in ("cuda", "rocm"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                cell, paths = _write_cell(Path(tmp), backend, bench_runs=3)
                result = validate_production_dense_prefill_cell(
                    paths.aggregate, paths.timing, cell, bench_runs=3
                )
                self.assertEqual(
                    result.candidate_count,
                    len(dense_prefill_candidate_ids(
                        backend, cell.source_format
                    )),
                )
                self.assertEqual(result.sample_count, result.candidate_count * 3)
                self.assertEqual(result.winner_min_us, 10.0)

    def test_one_non_winning_byte_mismatch_invalidates_the_entire_cell(self) -> None:
        """Every timed regime must be decode-equivalent, not only the winner."""

        for backend in ("cuda", "rocm"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                cell, paths = _write_cell(Path(tmp), backend, bench_runs=3)
                columns = (
                    CUDA_AGGREGATE_COLUMNS
                    if backend == "cuda"
                    else ROCM_AGGREGATE_COLUMNS
                )
                with paths.aggregate.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))

                self.assertGreater(len(rows), 1)
                mismatch_field = (
                    "byte_mismatches"
                    if backend == "cuda"
                    else "byte_mismatches_vs_auto"
                )
                first_field = (
                    "first_byte_mismatch"
                    if backend == "cuda"
                    else "first_byte_mismatch_vs_auto"
                )
                rows[-1][mismatch_field] = "1"
                rows[-1][first_field] = "0"
                with paths.aggregate.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(rows)

                with self.assertRaisesRegex(ValueError, "failed byte equality"):
                    validate_production_dense_prefill_cell(
                        paths.aggregate,
                        paths.timing,
                        cell,
                        bench_runs=3,
                    )

    def test_cuda_noncanonical_observed_reduction_invalidates_the_cell(self) -> None:
        """A byte certificate cannot disguise execution of an unsafe route."""

        with tempfile.TemporaryDirectory() as tmp:
            cell, paths = _write_cell(Path(tmp), "cuda", bench_runs=3)
            with paths.aggregate.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[-1]["observed_k_partitions"] = "2"
            with paths.aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CUDA_AGGREGATE_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "noncanonical CUDA reduction"):
                validate_production_dense_prefill_cell(
                    paths.aggregate,
                    paths.timing,
                    cell,
                    bench_runs=3,
                )

    def test_cuda_non_kpar_shape_excludes_every_canonical_candidate(self) -> None:
        """M=1 routes without partitions cannot advertise KPART evidence."""

        with tempfile.TemporaryDirectory() as tmp:
            cell, paths = _write_cell(
                Path(tmp),
                "cuda",
                bench_runs=3,
                canonical_kpart_available=False,
            )
            result = validate_production_dense_prefill_cell(
                paths.aggregate,
                paths.timing,
                cell,
                bench_runs=3,
            )
            self.assertEqual(
                result.candidate_count,
                len(dense_prefill_candidate_ids(
                    "cuda",
                    cell.source_format,
                    canonical_kpart_available=False,
                )),
            )

            with paths.aggregate.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["canonical_kpart_available"] = "1"
            with paths.aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CUDA_AGGREGATE_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(
                ValueError,
                "inconsistent canonical KPART availability",
            ):
                validate_production_dense_prefill_cell(
                    paths.aggregate,
                    paths.timing,
                    cell,
                    bench_runs=3,
                )

    def test_truncated_timing_sidecar_is_not_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cell, paths = _write_cell(Path(tmp), "cuda", bench_runs=3)
            lines = paths.timing.read_text(encoding="utf-8").splitlines()
            paths.timing.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "samples"):
                validate_production_dense_prefill_cell(
                    paths.aggregate, paths.timing, cell, bench_runs=3
                )

    def test_complete_cell_is_reused_without_relaunch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell, _ = _write_cell(root, "rocm", bench_runs=3)
            binary = root / "trainer"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary.chmod(0o755)
            with mock.patch(
                "native_vnni_dispatch.production_dense_prefill_sweep._run_accelerator_cell"
            ) as launch:
                completed = run_missing_accelerator_cells(
                    binary,
                    root,
                    (cell,),
                    (0,),
                    warmup_runs=2,
                    bench_runs=3,
                )
            self.assertEqual(completed, ())
            launch.assert_not_called()

    def test_device_masks_and_exact_cell_filters_are_backend_specific(self) -> None:
        for backend, device in (("cuda", 1), ("rocm", 3)):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                cell = production_dense_prefill_cells(backend)[0]
                paths = production_dense_prefill_cell_paths(Path(tmp), cell)
                environment = _cell_environment(cell, paths, device, 2, 3)
                if backend == "cuda":
                    self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "1")
                    self.assertNotIn("ROCR_VISIBLE_DEVICES", environment)
                    self.assertNotIn("HIP_VISIBLE_DEVICES", environment)
                    self.assertEqual(
                        environment["LLAMINAR_TILE_SWEEP_FORMAT"],
                        cell.source_format.label,
                    )
                else:
                    self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
                    self.assertNotIn("ROCR_VISIBLE_DEVICES", environment)
                    self.assertEqual(environment["HIP_VISIBLE_DEVICES"], "3")
                    self.assertEqual(
                        environment["LLAMINAR_ROCM_NVNNI_SWEEP_FORMATS"],
                        cell.source_format.label,
                    )

    def test_plan_and_combined_outputs_are_published_only_from_valid_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell, _ = _write_cell(root, "cuda", bench_runs=3)
            plan = root / "plan.json"
            write_sweep_plan(plan, "cuda", (cell,), warmup_runs=2, bench_runs=3)
            payload = __import__("json").loads(plan.read_text(encoding="utf-8"))
            self.assertEqual(payload["bench_runs"], 3)
            self.assertEqual(payload["cells"][0]["cell_id"], cell.cell_id)
            aggregate, timing = combine_production_dense_prefill_cells(
                root, (cell,), bench_runs=3
            )
            self.assertTrue(aggregate.is_file())
            self.assertTrue(timing.is_file())

    def test_overlay_pools_source_aliases_into_one_concrete_runtime_key(self) -> None:
        cells = production_dense_prefill_cells("cuda")
        anchor = next(cell for cell in cells if cell.source_format.label == "IQ4_NL")
        aliases = tuple(
            cell for cell in cells
            if cell.source_format.label in {"IQ4_NL", "IQ4_XS"}
            and cell.shape.name == anchor.shape.name
            and cell.m == anchor.m
        )
        self.assertEqual(len(aliases), 2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cell in aliases:
                _write_cell(root, "cuda", bench_runs=3, cell=cell)
            entries = collect_dense_overlay_entries(
                root, "cuda", cells=aliases, bench_runs=3
            )
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry.key.execution_codebook, 4)
            self.assertEqual(entry.source_formats, ("IQ4_NL", "IQ4_XS"))
            self.assertEqual(entry.candidate_id, "AUTO")
            self.assertEqual(entry.launch, (1, 1, 0, 0))
            rendered = render_dense_overlay(entries)
            self.assertIn("selectCUDADensePrefillOverlay", rendered)
            self.assertNotIn("AUTO", rendered)

    def test_rocm_overlay_preserves_full_tile_specialization(self) -> None:
        """Checked and proof-carrying kernels are distinct runtime launches."""

        entry = DenseOverlayEntry(
            backend="rocm",
            key=DenseOverlayKey(
                execution_codebook=8,
                m=512,
                n=512,
                k=2048,
            ),
            source_formats=("Q6_K",),
            shape_names=("35BMoE_Expert_GateUp",),
            candidate_id="N128/MT32/MB2/U4/FULL",
            launch=(128, 32, 2, 4, 1),
            geometric_mean_regret=0.0,
            maximum_alias_regret=0.0,
            geometric_mean_speedup_vs_auto=1.1,
        )

        rendered = render_dense_overlay((entry,))

        self.assertIn("selectROCmDensePrefillOverlay", rendered)
        self.assertIn("{128, 32, 2, 4, true}", rendered)

    def test_overlay_rejects_aliases_with_conflicting_physical_auto_launches(self) -> None:
        cells = production_dense_prefill_cells("cuda")
        anchor = next(cell for cell in cells if cell.source_format.label == "IQ4_NL")
        aliases = tuple(
            cell for cell in cells
            if cell.source_format.label in {"IQ4_NL", "IQ4_XS"}
            and cell.shape.name == anchor.shape.name
            and cell.m == anchor.m
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cell in aliases:
                _write_cell(root, "cuda", bench_runs=3, cell=cell)
            second = production_dense_prefill_cell_paths(root, aliases[1])
            with second.aggregate.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            rows[0]["observed_tile_id"] = "2"
            with second.aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CUDA_AGGREGATE_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "conflicting physical launches"):
                collect_dense_overlay_entries(
                    root, "cuda", cells=aliases, bench_runs=3
                )


if __name__ == "__main__":
    unittest.main()
