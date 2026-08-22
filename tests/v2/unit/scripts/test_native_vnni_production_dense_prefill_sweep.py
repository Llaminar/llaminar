#!/usr/bin/env python3
"""Regressions for the resumable all-format dense GPU prefill sweep."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
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
    validate_dense_overlay_totality,
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
    SWEEP_PLAN_FILENAME,
    TRAINER_PROVENANCE_SCHEMA,
    ROCM_AGGREGATE_COLUMNS,
    ROCM_CANDIDATE_IDS,
    ROCM_Q6_FULL_TILE_CANDIDATE_IDS,
    ROCM_Q6_SPILLING_CHECKED_CANDIDATE_IDS,
    ROCM_TIMING_COLUMNS,
    _batch_environment,
    _batch_paths,
    _cell_environment,
    _stage_cell_manifest,
    combine_production_dense_prefill_cells,
    dense_prefill_candidate_ids,
    load_planned_dense_prefill_cells,
    load_sweep_plan,
    production_dense_prefill_cell_paths,
    production_dense_prefill_cells,
    run_missing_accelerator_cells,
    validate_production_dense_prefill_cell,
    validate_promoted_production_dense_prefill_cell,
    write_sweep_plan,
)


_UINT64_MAX = (1 << 64) - 1


def _test_provenance(
    backend: str,
    *,
    binary_digest: str = "sha256:test-binary",
    core_digest: str = "sha256:test-core",
) -> dict[str, object]:
    return {
        "schema_version": TRAINER_PROVENANCE_SCHEMA,
        "backend": backend,
        "binary_name": f"{backend}-trainer",
        "binary_sha256": binary_digest,
        "core_library_name": "libllaminar2_core.so",
        "core_library_sha256": core_digest,
        "cmake_build_type": "Release",
    }


def _cuda_launch(candidate: str) -> dict[str, object]:
    if candidate == "AUTO":
        return {
            "tile": "AUTO",
            # Tile 2 is resource-eligible for every compiled runtime codebook.
            # Synthetic AUTO evidence must obey the same compiler-resource
            # contract as the native scorer's resolved physical launch.
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
                        else 2 if candidate == "AUTO"
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
                    "primary_registers_per_thread": 96,
                    "primary_local_memory_bytes_per_thread": 0,
                    "primary_static_shared_memory_bytes": 16384,
                    "primary_dynamic_shared_memory_bytes": (
                        65536 if candidate == "BK256:full" else 0
                    ),
                    "primary_threads_per_block": (
                        512 if candidate == "BK256:full" else 128
                    ),
                    "primary_max_threads_per_block": (
                        512 if candidate == "BK256:full" else 128
                    ),
                    "primary_max_active_blocks_per_sm": 2,
                    "auxiliary_registers_per_thread": (
                        16 if candidate.startswith("KPART:") else 0
                    ),
                    "auxiliary_local_memory_bytes_per_thread": 0,
                    "auxiliary_static_shared_memory_bytes": 0,
                    "auxiliary_dynamic_shared_memory_bytes": 0,
                    "auxiliary_threads_per_block": (
                        256 if candidate.startswith("KPART:") else 0
                    ),
                    "auxiliary_max_threads_per_block": (
                        1024 if candidate.startswith("KPART:") else 0
                    ),
                    "auxiliary_max_active_blocks_per_sm": (
                        4 if candidate.startswith("KPART:") else 0
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


def _commit_test_cells(
    root: Path,
    backend: str,
    cells,
    *,
    warmup_runs: int = 2,
    bench_runs: int = 3,
    provenance: dict[str, object] | None = None,
):
    """Publish real per-cell commit markers for already-written fixtures."""

    cells = tuple(cells)
    producer = provenance or _test_provenance(backend)
    plan = write_sweep_plan(
        root / SWEEP_PLAN_FILENAME,
        backend,
        cells,
        producer,
        warmup_runs=warmup_runs,
        bench_runs=bench_runs,
    )
    process = {
        "invocation_id": "test-native-process",
        "backend": backend,
        "device_ordinal": 0,
        "cell_ids": [cell.cell_id for cell in cells],
    }
    for cell in cells:
        paths = production_dense_prefill_cell_paths(root, cell)
        shutil.copyfile(paths.aggregate, paths.aggregate_staging)
        shutil.copyfile(paths.timing, paths.timing_staging)
        shutil.copyfile(paths.log, paths.log_staging)
        _stage_cell_manifest(paths, cell, plan, process)
        paths.aggregate_staging.unlink()
        paths.timing_staging.unlink()
        paths.log_staging.unlink()
        os.replace(paths.manifest_staging, paths.manifest)
        validate_promoted_production_dense_prefill_cell(root, cell, plan)
    return plan


class ProductionDensePrefillSweepTest(unittest.TestCase):
    """Keep the dense corpus comprehensive, exact, and safely resumable."""

    def test_matrix_is_all_format_shape_and_m_complete_on_both_backends(self) -> None:
        expected_count = (
            len(FORMAT_SPECS)
            * len(gpu_prefill_measurements())
            * len(GPU_PREFILL_M_BUCKETS)
        )
        self.assertEqual(expected_count, 35721)
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
        q6_cuda = next(spec for spec in FORMAT_SPECS if spec.label == "Q6_K")
        self.assertEqual(
            dense_prefill_candidate_ids("cuda", q6_cuda),
            (
                "AUTO",
                "STD:t0:full",
                "STD:t2:full",
                "STD:t3:full",
                "STD:t5:full",
                *(f"KPART:t{tile}" for tile in range(6)),
            ),
        )
        for label in ("Q2_K", "IQ1_M"):
            source_format = next(
                spec for spec in FORMAT_SPECS if spec.label == label
            )
            self.assertEqual(
                dense_prefill_candidate_ids("cuda", source_format),
                (
                    "AUTO",
                    "STD:t0:full",
                    "STD:t2:full",
                    "STD:t3:full",
                    "KPART:t0",
                    "KPART:t2",
                    "KPART:t3",
                ),
            )
        q8_inventories = {
            dense_prefill_candidate_ids(
                "cuda",
                next(spec for spec in FORMAT_SPECS if spec.label == label),
            )
            for label in ("Q8_0", "Q8_1", "Q8_K")
        }
        self.assertEqual(len(q8_inventories), 1)
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

    def test_cuda_spilling_primary_or_reducer_never_enters_evidence(self) -> None:
        """Static local memory invalidates even a byte-correct fast row."""

        for candidate_prefix, field in (
            ("STD:", "primary_local_memory_bytes_per_thread"),
            ("KPART:", "auxiliary_local_memory_bytes_per_thread"),
        ):
            with self.subTest(candidate=candidate_prefix), tempfile.TemporaryDirectory() as tmp:
                cell, paths = _write_cell(Path(tmp), "cuda", bench_runs=3)
                with paths.aggregate.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                target = next(
                    row for row in rows
                    if f"{row['strategy']}:".startswith(candidate_prefix)
                )
                target[field] = "16"
                with paths.aggregate.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=CUDA_AGGREGATE_COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)

                with self.assertRaisesRegex(ValueError, "ineligible CUDA"):
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
            producer = _test_provenance("rocm")
            _commit_test_cells(
                root, "rocm", (cell,), provenance=producer
            )
            binary = root / "trainer"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary.chmod(0o755)
            with mock.patch(
                "native_vnni_dispatch.production_dense_prefill_sweep."
                "trainer_provenance",
                return_value=producer,
            ), mock.patch(
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

    def test_changed_binary_or_core_digest_rejects_resume_before_launch(self) -> None:
        """A rebuilt trainer closure requires a fresh immutable corpus root."""

        for changed in ("binary", "core"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cell, _ = _write_cell(root, "cuda", bench_runs=3)
                _commit_test_cells(root, "cuda", (cell,))
                binary = root / "trainer"
                binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                binary.chmod(0o755)
                observed = _test_provenance(
                    "cuda",
                    binary_digest=(
                        "sha256:changed" if changed == "binary"
                        else "sha256:test-binary"
                    ),
                    core_digest=(
                        "sha256:changed" if changed == "core"
                        else "sha256:test-core"
                    ),
                )
                with mock.patch(
                    "native_vnni_dispatch.production_dense_prefill_sweep."
                    "trainer_provenance",
                    return_value=observed,
                ), mock.patch(
                    "native_vnni_dispatch.production_dense_prefill_sweep."
                    "_run_accelerator_cell"
                ) as launch, self.assertRaisesRegex(
                    ValueError, "immutable sweep plan"
                ):
                    run_missing_accelerator_cells(
                        binary,
                        root,
                        (cell,),
                        (0,),
                        warmup_runs=2,
                        bench_runs=3,
                    )
                launch.assert_not_called()

    def test_status_validation_never_rewrites_plan_or_timing_contract(self) -> None:
        """Read-only inspection cannot adapt a corpus to new CLI arguments."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell, _ = _write_cell(root, "cuda", bench_runs=3)
            _commit_test_cells(root, "cuda", (cell,))
            plan_path = root / SWEEP_PLAN_FILENAME
            original = plan_path.read_bytes()

            load_sweep_plan(
                plan_path,
                "cuda",
                (cell,),
                warmup_runs=2,
                bench_runs=3,
            )
            self.assertEqual(plan_path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "timing cardinality"):
                load_sweep_plan(
                    plan_path,
                    "cuda",
                    (cell,),
                    warmup_runs=2,
                    bench_runs=4,
                )
            self.assertEqual(plan_path.read_bytes(), original)

    def test_batch_cell_manifests_share_one_native_process_identity(self) -> None:
        """All M cells emitted by one process retain that shared provenance."""

        cells = production_dense_prefill_cells("cuda")
        anchor = cells[0]
        same_weight = tuple(
            cell for cell in cells
            if cell.source_format == anchor.source_format
            and cell.shape == anchor.shape
        )[:3]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cell in same_weight:
                _write_cell(root, "cuda", bench_runs=3, cell=cell)
            _commit_test_cells(root, "cuda", same_weight)
            manifests = [
                json.loads(production_dense_prefill_cell_paths(
                    root, cell
                ).manifest.read_text(encoding="utf-8"))
                for cell in same_weight
            ]

        process_records = [manifest["native_process"] for manifest in manifests]
        self.assertTrue(all(record == process_records[0] for record in process_records))
        self.assertEqual(
            process_records[0]["cell_ids"],
            [cell.cell_id for cell in same_weight],
        )

    def test_artifact_mutation_invalidates_promoted_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell, paths = _write_cell(root, "rocm", bench_runs=3)
            plan = _commit_test_cells(root, "rocm", (cell,))
            paths.log.write_text("mutated\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact digest mismatch"):
                validate_promoted_production_dense_prefill_cell(
                    root, cell, plan
                )

    def test_missing_m_buckets_share_one_native_weight_process(self) -> None:
        """Batching removes setup work without weakening cell-level resume."""

        cells = production_dense_prefill_cells("cuda")
        anchor = cells[0]
        same_weight = tuple(
            cell for cell in cells
            if cell.source_format == anchor.source_format
            and cell.shape == anchor.shape
        )[:3]
        self.assertEqual(len(same_weight), 3)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "trainer"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary.chmod(0o755)
            with mock.patch(
                "native_vnni_dispatch.production_dense_prefill_sweep."
                "trainer_provenance",
                return_value=_test_provenance("cuda"),
            ), mock.patch(
                "native_vnni_dispatch.production_dense_prefill_sweep."
                "_run_accelerator_batch"
            ) as batch_launch, mock.patch(
                "native_vnni_dispatch.production_dense_prefill_sweep."
                "_run_accelerator_cell"
            ) as cell_launch:
                completed = run_missing_accelerator_cells(
                    binary,
                    root,
                    same_weight,
                    (0,),
                    warmup_runs=2,
                    bench_runs=3,
                )

            self.assertEqual(completed, same_weight)
            batch_launch.assert_called_once()
            self.assertEqual(
                batch_launch.call_args.args[2],
                same_weight,
            )
            cell_launch.assert_not_called()

    def test_batch_environment_varies_only_m_and_preserves_output_identity(self) -> None:
        """The native process must never mix formats or matrix geometries."""

        cells = production_dense_prefill_cells("rocm")
        anchor = cells[0]
        same_weight = tuple(
            cell for cell in cells
            if cell.source_format == anchor.source_format
            and cell.shape == anchor.shape
        )[:3]
        with tempfile.TemporaryDirectory() as tmp:
            paths = _batch_paths(Path(tmp), same_weight)
            environment = _batch_environment(
                same_weight,
                paths,
                device=2,
                warmup_runs=3,
                bench_runs=7,
            )

        self.assertEqual(environment["HIP_VISIBLE_DEVICES"], "2")
        self.assertEqual(
            environment["LLAMINAR_ROCM_NVNNI_SWEEP_FORMATS"],
            anchor.source_format.label,
        )
        self.assertEqual(
            environment["LLAMINAR_ROCM_NVNNI_SWEEP_SHAPES"],
            anchor.shape.name,
        )
        self.assertEqual(
            environment["LLAMINAR_ROCM_NVNNI_SWEEP_M"],
            ",".join(str(cell.m) for cell in same_weight),
        )
        self.assertEqual(
            environment["LLAMINAR_ROCM_NVNNI_SWEEP_MAX_CASES"],
            str(len(same_weight)),
        )

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
            payload = write_sweep_plan(
                plan,
                "cuda",
                (cell,),
                _test_provenance("cuda"),
                warmup_runs=2,
                bench_runs=3,
            )
            self.assertEqual(payload["bench_runs"], 3)
            self.assertEqual(payload["cells"][0]["cell_id"], cell.cell_id)
            plan.replace(root / SWEEP_PLAN_FILENAME)
            _commit_test_cells(root, "cuda", (cell,))
            aggregate, timing = combine_production_dense_prefill_cells(
                root, (cell,), warmup_runs=2, bench_runs=3
            )
            self.assertTrue(aggregate.is_file())
            self.assertTrue(timing.is_file())

    def test_filtered_overlay_consumes_the_exact_immutable_plan_matrix(self) -> None:
        cells = production_dense_prefill_cells("cuda")
        selected = cells[:2]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cell in selected:
                _write_cell(root, "cuda", bench_runs=3, cell=cell)
            _commit_test_cells(root, "cuda", selected)
            reconstructed, plan = load_planned_dense_prefill_cells(
                root / SWEEP_PLAN_FILENAME,
                "cuda",
            )
            self.assertEqual(reconstructed, selected)
            self.assertEqual(len(plan["cells"]), 2)
            entries = collect_dense_overlay_entries(
                root,
                "cuda",
                warmup_runs=2,
                bench_runs=3,
            )
            self.assertTrue(entries)

    def test_cuda_overlay_honors_geometry_without_canonical_kpart(self) -> None:
        """One-partition shapes must not invent absent KPART timing rows."""

        cell = production_dense_prefill_cells("cuda")[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_cell(
                root,
                "cuda",
                bench_runs=3,
                cell=cell,
                canonical_kpart_available=False,
            )
            _commit_test_cells(root, "cuda", (cell,))
            entries = collect_dense_overlay_entries(
                root,
                "cuda",
                cells=(cell,),
                warmup_runs=2,
                bench_runs=3,
            )

            self.assertEqual(len(entries), 1)
            self.assertFalse(entries[0].candidate_id.startswith("KPART:"))

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
            _commit_test_cells(root, "cuda", aliases)
            entries = collect_dense_overlay_entries(
                root,
                "cuda",
                cells=aliases,
                warmup_runs=2,
                bench_runs=3,
            )
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry.key.execution_codebook, 4)
            self.assertEqual(entry.source_formats, ("IQ4_NL", "IQ4_XS"))
            self.assertEqual(entry.candidate_id, "AUTO")
            self.assertEqual(entry.launch, (2, 1, 0, 0))
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
            rows[0]["observed_tile_id"] = "3"
            with second.aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CUDA_AGGREGATE_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            _commit_test_cells(root, "cuda", aliases)
            with self.assertRaisesRegex(ValueError, "conflicting physical launches"):
                collect_dense_overlay_entries(
                    root,
                    "cuda",
                    cells=aliases,
                    warmup_runs=2,
                    bench_runs=3,
                )

    def test_overlay_totality_rejects_missing_and_spilling_runtime_keys(self) -> None:
        """Installation cannot omit a planned key or promote a spilling tile."""

        cells = production_dense_prefill_cells("cuda")
        anchor = next(
            cell for cell in cells
            if cell.source_format.runtime_codebook("cuda") == 8
        )
        selected = tuple(
            cell for cell in cells
            if cell.source_format == anchor.source_format
            and cell.shape == anchor.shape
        )[:2]
        self.assertEqual(len(selected), 2)

        def entry_for(cell, *, tile_id: int = 2) -> DenseOverlayEntry:
            return DenseOverlayEntry(
                backend="cuda",
                key=DenseOverlayKey(
                    execution_codebook=cell.source_format.runtime_codebook(
                        "cuda"
                    ),
                    m=cell.m,
                    n=cell.shape.n,
                    k=cell.shape.k,
                ),
                source_formats=(cell.source_format.label,),
                shape_names=(cell.shape.name,),
                candidate_id=f"STD:t{tile_id}:full",
                launch=(tile_id, 1, 0, 0),
                geometric_mean_regret=0.0,
                maximum_alias_regret=0.0,
                geometric_mean_speedup_vs_auto=1.0,
            )

        with self.assertRaisesRegex(ValueError, "not total"):
            validate_dense_overlay_totality(
                (entry_for(selected[0]),),
                selected,
                "cuda",
            )

        spilling = tuple(entry_for(cell, tile_id=1) for cell in selected)
        with self.assertRaisesRegex(ValueError, "compiler-proven to spill"):
            validate_dense_overlay_totality(spilling, selected, "cuda")


if __name__ == "__main__":
    unittest.main()
