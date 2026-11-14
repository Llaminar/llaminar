#!/usr/bin/env python3
"""Device-free regressions for the resumable all-format dense GPU prefill sweep.

Synthetic trainer records exercise exact family/geometry identity, resource and
byte-certificate rejection, and durable publication without launching devices.
Auto may select a different physical family than a forced tournament tile;
that distinction must survive validation without weakening either contract.
Matrix cardinality follows the canonical capture, format and shape inventories,
so adding a shared CUDA/ROCm bucket extends coverage without a second row list.
ROCm synthetic measurements retain source and physical execution IDs separately;
physical24 cannot be inferred from a historical source4-only packet.
Narrowed candidate protocols retain exhaustive historical IDs and cannot share
a batch, resume plan or profiler witness with a different roster.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import unittest
from dataclasses import replace
from functools import cache
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_TOOLS = REPO_ROOT / "tests/v2/performance/kernels"
if str(KERNEL_TOOLS) not in os.sys.path:
    os.sys.path.insert(0, str(KERNEL_TOOLS))

from native_quantised_dispatch.gpu_architecture import GPUArchitecture
from native_quantised_dispatch.format_registry import FORMAT_SPECS  # noqa: E402
from native_quantised_dispatch.dense_production_overlay import (  # noqa: E402
    DenseOverlayEntry,
    DensePreparedWeightFamily,
    DenseOverlayKey,
    collect_dense_overlay_entries,
    render_dense_overlay,
    retain_dense_overlay_base,
    validate_dense_overlay_totality,
    write_dense_overlay,
)
from native_quantised_dispatch.prefill_matrix import (  # noqa: E402
    GPU_PREFILL_M_BUCKETS,
    gpu_prefill_measurements,
)
from native_quantised_dispatch.production_dense_prefill_sweep import (  # noqa: E402
    DenseCandidateScope,
    CUDA_AGGREGATE_COLUMNS,
    ROCM_NATIVE_Q8_CANDIDATE_IDS,
    ROCM_WMMA_CANDIDATE_IDS,
    CUDA_CANDIDATE_IDS,
    CUDA_GENERIC_CANDIDATE_IDS,
    CUDA_TILE_NAMES,
    CUDA_TIMING_COLUMNS,
    DENSE_PREFILL_CAPTURED_OPERATIONS,
    SWEEP_PLAN_FILENAME,
    TRAINER_PROVENANCE_SCHEMA,
    ROCM_AGGREGATE_COLUMNS,
    ROCM_CANDIDATE_IDS,
    ROCM_Q6_FULL_TILE_CANDIDATE_IDS,
    ROCM_Q6_SPILLING_CHECKED_CANDIDATE_IDS,
    ROCM_TIMING_COLUMNS,
    _cuda_candidate_id,
    _cuda_candidate_is_spill_free,
    _read_aggregate_rows,
    _read_timing_rows,
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


@cache
def _fixture_cells(backend: str):
    """Reuse immutable canonical cells when selecting synthetic row fixtures.

    Each backend's inventory contains tens of thousands of frozen declarations.
    Selecting one fixture must not rebuild that whole matrix for every malformed
    CSV or source alias. The exhaustive inventory test still calls the production
    expansion directly, and plan readers keep their real authentication path.
    Tests derive changed cells with dataclasses.replace; no shared cell is edited.
    """
    return production_dense_prefill_cells(backend)


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
    """Encode synthetic evidence with the same independent physical axes."""
    base, separator, stage_text = candidate.partition(":stage")
    staging = int(stage_text) if separator else 0
    if staging not in (0, 1, 2, 3):
        raise ValueError(f"unexpected staging policy {candidate}")
    candidate = base
    if candidate == "AUTO":
        return {
            "tile": "AUTO",
            # Tile 2 is resource-eligible for every compiled runtime codebook.
            # Synthetic AUTO evidence must obey the same compiler-resource
            # contract as the native scorer's resolved physical launch.
            "tile_id": -1,
            "strategy": "AUTO",
            "requested_k_partitions": 0,
            "staging_schedule": staging,
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
            "staging_schedule": staging,
        }
    if strategy == "KPART":
        tile_id = int(remainder.removeprefix("t"))
        return {
            "tile": CUDA_TILE_NAMES[tile_id],
            "tile_id": tile_id,
            "strategy": strategy,
            "requested_k_partitions": 0,
            "staging_schedule": staging,
        }
    return {
        "tile": "BK256_128x128",
        "tile_id": -2,
        "strategy": strategy,
        "requested_k_partitions": 1,
        "staging_schedule": staging,
    }


def _write_cell(
    root: Path,
    backend: str,
    *,
    bench_runs: int = 3,
    cell=None,
    canonical_kpart_available: bool = True,
    winning_candidate: str | None = None,
):
    cell = cell or _fixture_cells(backend)[0]
    paths = production_dense_prefill_cell_paths(root, cell)
    paths.aggregate.parent.mkdir(parents=True, exist_ok=True)
    candidates = dense_prefill_candidate_ids(
        backend,
        cell.source_format,
        canonical_kpart_available=canonical_kpart_available,
        prepared_family=cell.prepared_family, gpu_architecture=cell.gpu_architecture, candidate_scope=cell.candidate_scope,
    )
    samples_by_candidate = {
        candidate: tuple(
            (1.0 if candidate == winning_candidate else 10.0 + candidate_index) + sample_index * 0.125
            for sample_index in range(bench_runs)
        )
        for candidate_index, candidate in enumerate(candidates)
    }

    if backend == "cuda":
        aggregate_columns = CUDA_AGGREGATE_COLUMNS
        timing_columns = CUDA_TIMING_COLUMNS
    else:
        aggregate_columns = ROCM_AGGREGATE_COLUMNS + (("prepared_family", "gpu_architecture", "gpu_uuid", "gpu_pci", "native_ordinal") if cell.prepared_family is not None else ())
        timing_columns = ROCM_TIMING_COLUMNS + (("prepared_family", "gpu_architecture", "gpu_uuid", "gpu_pci", "native_ordinal") if cell.prepared_family is not None else ())

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
                    "observed_staging_schedule": launch["staging_schedule"],
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
                int8 = cell.effective_prepared_family == DensePreparedWeightFamily.INT8_BLOCKWISE
                int8_v3 = candidate.startswith("INT8V3/")
                wmma = candidate.startswith("WMMA/")
                explicit_unroll = (
                    int(components[3].removeprefix("U"))
                    if len(components) >= 4 and components[3].startswith("U")
                    else 4
                )
                writer.writerow({
                    "backend": "rocm",
                    "phase": "prefill",
                    **({"prepared_family": "native_quantised", "gpu_architecture": cell.gpu_architecture,
                        "gpu_uuid": "0123456789abcdef0123456789abcdef", "gpu_pci": "0000:8c:00.0", "native_ordinal": "0"}
                       if cell.prepared_family is not None else {}),
                    "format": cell.source_format.label,
                    "codebook": cell.source_format.source_codebook_id,
                    "execution_codebook": cell.source_format.runtime_codebook("rocm"),
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
                    "is_best": int(candidate == winning_candidate if winning_candidate else candidate_index == 0),
                    "observed_n_tile": (
                        int(components[1].removeprefix("N")) if wmma else
                        (64 if int8_v3 else 128) if int8
                        else 128 if candidate == "Auto"
                        else int(candidate.split("/")[0].removeprefix("N"))
                    ),
                    "observed_m_tile": (
                        int(components[2].removeprefix("MT")) if wmma else
                        32 if candidate == "Auto"
                        else int(candidate.split("/")[1].removeprefix("MT"))
                    ),
                    "observed_min_blocks": (
                        1 if wmma or int8 or candidate == "Auto"
                        else int(candidate.split("/")[2].removeprefix("MB"))
                    ),
                    "observed_unroll": (
                        0 if wmma else 2 if candidate == "Auto"
                        else int(components[2].removeprefix("U")) if int8
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
                    "observed_producer": (
                        "native_wmma" if wmma else
                        ("int8_blockwise_v3" if int8_v3 else "int8_blockwise_v7")
                        if int8 else "native_cooperative"
                    ),
                    "dynamic_shared_memory_bytes": 0,
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
                    **({"prepared_family": "native_quantised", "gpu_architecture": cell.gpu_architecture,
                        "gpu_uuid": "0123456789abcdef0123456789abcdef", "gpu_pci": "0000:8c:00.0", "native_ordinal": "0"}
                       if cell.prepared_family is not None else {}),
                    "format": cell.source_format.label,
                    "codebook": cell.source_format.source_codebook_id,
                    "shape": cell.shape.name,
                    "m": cell.m,
                    "n": cell.shape.n,
                    "k": cell.shape.k,
                    "sample_index": sample_index,
                    "timed_replays": 1,
                    "captured_operations": DENSE_PREFILL_CAPTURED_OPERATIONS,
                    "latency_us": f"{latency_us:.9f}",
                    "latency_us_hex": latency_us.hex(),
                }
                if backend == "cuda":
                    writer.writerow({**common, **_cuda_launch(candidate)})
                else:
                    writer.writerow({**common,
                        "execution_codebook": cell.source_format.runtime_codebook("rocm"),
                        "variant": candidate})
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
    installed_policy: dict[str, object] | None = None,
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
        installed_policy=installed_policy,
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

    @staticmethod
    def _native_q8_source(label="Q8_0"):
        """Bind the existing HC production shape; synthetic clocks remain test-only."""
        return next(cell for cell in _fixture_cells("rocm")
                    if cell.source_format.label == label
                    and cell.shape.name == "Qwen4Exp_320x10240" and cell.m == 64)

    def _scoped_wmma_cell(self, label="Q4_0"):
        """Use the canonical native shape with the explicit five-candidate protocol."""
        return replace(self._native_q8_source(label),
            prepared_family=DensePreparedWeightFamily.NATIVE_QUANTISED,gpu_architecture="gfx1100",
            candidate_scope=DenseCandidateScope.INSTALLED_VS_NATIVE_WMMA)

    def test_candidate_scope_preserves_exhaustive_identity(self) -> None:
        """Default protocols retain their historical field set and immutable plan bytes."""
        for backend in ("cuda","rocm"):
            legacy=next(cell for cell in _fixture_cells(backend) if cell.source_format.label=="Q4_0"
                        and cell.shape.name=="Qwen4Exp_320x10240" and cell.m==64)
            # Frozen from the original v4 owner before adding candidate scope.
            self.assertEqual(legacy.cell_id,{"cuda":"cuda-q4_0-m64-e71f1f041dff0f26a111",
                "rocm":"rocm-q4_0-m64-dfebf34f4acfa3615952"}[backend])
            explicit=replace(legacy,candidate_scope=DenseCandidateScope.EXHAUSTIVE)
            self.assertEqual(legacy.identity,explicit.identity)
            self.assertEqual(legacy.cell_id,explicit.cell_id)
            self.assertNotIn("candidate_scope",explicit.canonical_mapping())
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/SWEEP_PLAN_FILENAME
                first=write_sweep_plan(path,backend,(legacy,),_test_provenance(backend))
                before=path.read_bytes()
                self.assertEqual(first,write_sweep_plan(path,backend,(explicit,),_test_provenance(backend)))
                self.assertEqual(before,path.read_bytes())
                self.assertTrue(first["schema_version"].endswith("-v4"))
                self.assertNotIn("candidate_scope",first)

    def test_candidate_scope_all_formats_require_exact_five_roster(self) -> None:
        """All21 formats require Auto and every native candidate, with no extra rows."""
        expected=("Auto",*ROCM_WMMA_CANDIDATE_IDS)
        for source in FORMAT_SPECS:
            cell=self._scoped_wmma_cell(source.label)
            with self.subTest(source=source.label),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);_,paths=_write_cell(root,"rocm",cell=cell)
                self.assertEqual(tuple(_read_aggregate_rows(paths.aggregate,cell)),expected)
                self.assertEqual(validate_production_dense_prefill_cell(paths.aggregate,paths.timing,cell,bench_runs=3).candidate_count,5)
                env=_cell_environment(cell,paths,5,2,3)
                self.assertEqual(tuple(env["LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_VARIANTS"].split(",")),expected)
                with paths.aggregate.open() as handle:rows=list(csv.DictReader(handle))
                for malformed in (rows[1:],rows+[dict(rows[0],variant=ROCM_CANDIDATE_IDS[0])]):
                    with paths.aggregate.open("w",newline="") as handle:
                        writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(malformed)
                    with self.assertRaises(ValueError):
                        validate_production_dense_prefill_cell(paths.aggregate,paths.timing,cell,bench_runs=3)

    def test_candidate_scope_rejects_incompatible_protocols(self) -> None:
        """No foreign architecture, preparation, backend or untyped string can narrow."""
        cell=self._scoped_wmma_cell()
        for change in ({"gpu_architecture":"gfx906"},{"gpu_architecture":None},
                       {"backend":"cuda"},{"prepared_family":None},
                       {"prepared_family":DensePreparedWeightFamily.INT8_BLOCKWISE},
                       {"candidate_scope":"installed-vs-native-wmma"}):
            with self.subTest(change=change),self.assertRaises(ValueError):replace(cell,**change)
        for backend,family,arch in (("rocm",None,"gfx1100"),("rocm",DensePreparedWeightFamily.NATIVE_QUANTISED,"gfx906"),
                                   ("cuda",DensePreparedWeightFamily.NATIVE_QUANTISED,"gfx1100")):
            with self.assertRaises(ValueError):
                dense_prefill_candidate_ids(backend,cell.source_format,prepared_family=family,gpu_architecture=arch,
                    candidate_scope=DenseCandidateScope.INSTALLED_VS_NATIVE_WMMA)

    def test_candidate_scope_plan_resume_and_batch_are_separate(self) -> None:
        """A narrowed plan round-trips exactly and rejects exhaustive or mixed resumes."""
        cell=self._scoped_wmma_cell();wide=replace(cell,candidate_scope=DenseCandidateScope.EXHAUSTIVE)
        self.assertNotEqual(cell.cell_id,wide.cell_id)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cells=(cell,replace(cell,m=128))
            plan=write_sweep_plan(root/SWEEP_PLAN_FILENAME,"rocm",cells,_test_provenance("rocm"))
            self.assertTrue(plan["schema_version"].endswith("-v5"))
            loaded,roundtrip=load_planned_dense_prefill_cells(root/SWEEP_PLAN_FILENAME,"rocm")
            self.assertEqual(loaded,cells);self.assertEqual(roundtrip,plan)
            env=_batch_environment(cells,_batch_paths(root,cells),5,2,3)
            self.assertEqual(env["LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_VARIANTS"],",".join(("Auto",*ROCM_WMMA_CANDIDATE_IDS)))
            with self.assertRaises(ValueError):
                write_sweep_plan(root/SWEEP_PLAN_FILENAME,"rocm",(wide,),_test_provenance("rocm"))
            with self.assertRaisesRegex(ValueError,"candidate-scope"):
                write_sweep_plan(root/"mixed.json","rocm",(wide,cell),_test_provenance("rocm"))
            with self.assertRaises(ValueError):_batch_paths(root,(wide,replace(cell,m=128)))

    def test_candidate_scope_keeps_native_auto_first_tie_order(self) -> None:
        """Equal clocks retain native filtered generator order: Auto precedes WMMA."""
        cell=self._scoped_wmma_cell()
        with tempfile.TemporaryDirectory() as tmp:
            _,paths=_write_cell(Path(tmp),"rocm",cell=cell)
            for path in (paths.aggregate,paths.timing):
                with path.open() as handle:rows=list(csv.DictReader(handle))
                if path==paths.aggregate:
                    original=rows[0]
                    for row in rows:
                        for field in ("min_us","mean_us","stddev_us"):row[field]=original[field]
                else:
                    for row in rows:
                        value=10.+int(row["sample_index"])*.125
                        row.update(latency_us=f"{value:.9f}",latency_us_hex=value.hex())
                with path.open("w",newline="") as handle:
                    writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
            result=validate_production_dense_prefill_cell(paths.aggregate,paths.timing,cell,bench_runs=3)
            self.assertEqual(result.winner_id,"Auto")

    def test_candidate_scope_profiles_retain_actual_auto_and_aliases(self) -> None:
        """Reduced rosters preserve every source alias and an unforced physical Auto witness."""
        from native_quantised_dispatch.native_prefill_profiler import build_manifest,validate_request
        from native_quantised_dispatch.corpus_provenance import mapping_digest
        cells=tuple(self._scoped_wmma_cell(source.label) for source in FORMAT_SPECS)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for cell in cells:_write_cell(root,"rocm",cell=cell,winning_candidate="WMMA/N64/MT32")
            _commit_test_cells(root,"rocm",cells)
            manifest=build_manifest(root,warmups=2,samples=3)
            self.assertEqual(len(manifest.requests),2*len(FORMAT_SPECS))
            for request in manifest.requests:
                validate_request(request)
                self.assertEqual(request.config_json["cell"]["candidate_scope"],DenseCandidateScope.INSTALLED_VS_NATIVE_WMMA.value)
                if request.config_json["purpose"]=="baseline":
                    self.assertEqual(request.config_json["requested_candidate"],"Auto")
                    self.assertTrue(request.effective_candidate_id.startswith("Auto:"))
            request=manifest.requests[0]
            config={**request.config_json,"cell":{**request.config_json["cell"],"candidate_scope":"unknown"}}
            with self.assertRaises(ValueError):
                validate_request(replace(request,config_json=config,candidate_policy_hash=mapping_digest(config)))

    def test_candidate_scope_profiler_rejects_relabelled_forced_dot4(self) -> None:
        """Standalone profiler admission cannot narrow a forced exhaustive Dot4 witness."""
        from native_quantised_dispatch.native_prefill_profiler import build_manifest,validate_request
        from native_quantised_dispatch.corpus_provenance import mapping_digest
        cell=replace(self._scoped_wmma_cell(),candidate_scope=DenseCandidateScope.EXHAUSTIVE)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            _write_cell(root,"rocm",cell=cell,winning_candidate=ROCM_CANDIDATE_IDS[0])
            _commit_test_cells(root,"rocm",(cell,))
            request=next(r for r in build_manifest(root,warmups=2,samples=3).requests if r.config_json["purpose"]=="winner")
            validate_request(request)
            config={**request.config_json,"cell":{**request.config_json["cell"],
                "candidate_scope":DenseCandidateScope.INSTALLED_VS_NATIVE_WMMA.value}}
            with self.assertRaisesRegex(ValueError,"outside the declared"):
                validate_request(replace(request,config_json=config,candidate_policy_hash=mapping_digest(config)))

    def test_candidate_scope_cli_preserves_complete_canonical_inventory(self) -> None:
        """CLI narrowing changes only the roster, never source aliases, shapes or M."""
        from argparse import Namespace
        from native_quantised_dispatch.production_dense_prefill_sweep import _filtered_cells
        args=Namespace(backend="rocm",prepared_family="native-quantised",gpu_architecture="gfx1100",
            candidate_scope="installed-vs-native-wmma",format=None,shape=None,m=None)
        scoped=_filtered_cells(args)
        exhaustive=production_dense_prefill_cells("rocm",prepared_family=DensePreparedWeightFamily.NATIVE_QUANTISED,
            gpu_architecture="gfx1100",all_native_formats=True)
        key=lambda cell:(cell.source_format.label,cell.shape.name,cell.shape.n,cell.shape.k,cell.m)
        self.assertEqual(tuple(map(key,scoped)),tuple(map(key,exhaustive)))
        self.assertEqual({cell.source_format.label for cell in scoped},{source.label for source in FORMAT_SPECS})
        self.assertTrue(all(cell.candidate_scope is DenseCandidateScope.INSTALLED_VS_NATIVE_WMMA for cell in scoped))

    def test_candidate_scope_plan_rejects_rehashed_schema_disagreement(self) -> None:
        """A valid digest cannot authenticate conflicting schema/cell/top-level scope."""
        from native_quantised_dispatch.corpus_provenance import mapping_digest
        cell=self._scoped_wmma_cell()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/SWEEP_PLAN_FILENAME
            original=write_sweep_plan(path,"rocm",(cell,),_test_provenance("rocm"))
            mutations=[]
            legacy_schema=dict(original,schema_version="native-quantised-production-dense-prefill-sweep-v4")
            mutations.append(legacy_schema)
            legacy_no_top=dict(legacy_schema);legacy_no_top.pop("candidate_scope");mutations.append(legacy_no_top)
            missing_scope=dict(original);missing_scope.pop("candidate_scope");mutations.append(missing_scope)
            wrong_cell=dict(original,cells=[dict(original["cells"][0],candidate_scope="exhaustive")]);mutations.append(wrong_cell)
            unknown_scope=dict(original,candidate_scope="unknown");mutations.append(unknown_scope)
            for payload in mutations:
                payload.pop("plan_digest",None);payload["plan_digest"]=mapping_digest(payload)
                path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):load_planned_dense_prefill_cells(path,"rocm")

    def test_wmma_all_formats_keep_architecture_and_native_family(self) -> None:
        """Every format has a gfx1100 matrix candidate; gfx906 never inherits it."""
        base = production_dense_prefill_cells("rocm")[0]
        for source in FORMAT_SPECS:
            native = replace(base, source_format=source,
                prepared_family=DensePreparedWeightFamily.NATIVE_QUANTISED, gpu_architecture="gfx1100")
            candidates = dense_prefill_candidate_ids("rocm", source,
                prepared_family=native.prepared_family, gpu_architecture=native.gpu_architecture)
            self.assertTrue(set(ROCM_WMMA_CANDIDATE_IDS).issubset(candidates))
            self.assertFalse(set(ROCM_WMMA_CANDIDATE_IDS).intersection(dense_prefill_candidate_ids(
                "rocm", source, prepared_family=native.prepared_family, gpu_architecture="gfx906")))
            with tempfile.TemporaryDirectory() as tmp:
                _, paths = _write_cell(Path(tmp), "rocm", cell=native)
                result = validate_production_dense_prefill_cell(paths.aggregate, paths.timing, native, bench_runs=3)
                self.assertEqual(result.candidate_count, len(candidates))
                with paths.aggregate.open() as handle:
                    rows = list(csv.DictReader(handle))
                row = next(row for row in rows if row["variant"].startswith("WMMA/"))
                row["observed_producer"] = "native_cooperative"
                with paths.aggregate.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                    writer.writeheader(); writer.writerows(rows)
                with self.assertRaises(ValueError):
                    validate_production_dense_prefill_cell(paths.aggregate, paths.timing, native, bench_runs=3)

    def test_wmma_resource_admission_keeps_compiled_thread_extent(self) -> None:
        """WMMA and cooperative kernels each require their compiled 256-thread launch."""
        base = production_dense_prefill_cells("rocm")[0]
        native = replace(base, prepared_family=DensePreparedWeightFamily.NATIVE_QUANTISED,
            gpu_architecture="gfx1100")
        with tempfile.TemporaryDirectory() as tmp:
            _, paths = _write_cell(Path(tmp), "rocm", cell=native)
            with paths.aggregate.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                if row["observed_producer"] == "native_wmma":
                    row["max_threads_per_block"] = "256"
            def write_rows():
                """Keep the full exact candidate matrix while changing one resource witness."""
                with paths.aggregate.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                    writer.writeheader(); writer.writerows(rows)
            write_rows()
            validate_production_dense_prefill_cell(paths.aggregate, paths.timing, native, bench_runs=3)
            for producer, rejected_threads, admitted_threads in (
                    ("native_wmma", "255", "256"), ("native_cooperative", "255", "256")):
                row = next(row for row in rows if row["observed_producer"] == producer)
                row["max_threads_per_block"] = rejected_threads
                write_rows()
                with self.assertRaisesRegex(ValueError, "ineligible ROCm compiler resources"):
                    validate_production_dense_prefill_cell(paths.aggregate, paths.timing, native, bench_runs=3)
                row["max_threads_per_block"] = admitted_threads
                write_rows()
                validate_production_dense_prefill_cell(paths.aggregate, paths.timing, native, bench_runs=3)

    def test_wmma_all_format_profile_witnesses_retain_actual_baseline(self) -> None:
        """Profiling consumes sealed winners for every original source format."""
        from native_quantised_dispatch.native_prefill_profiler import build_manifest, validate_request
        base = production_dense_prefill_cells("rocm")[0]
        cells = tuple(replace(base, source_format=source,
            prepared_family=DensePreparedWeightFamily.NATIVE_QUANTISED, gpu_architecture="gfx1100")
            for source in FORMAT_SPECS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cell in cells:
                _write_cell(root, "rocm", cell=cell, winning_candidate="WMMA/N64/MT32")
            _commit_test_cells(root, "rocm", cells)
            manifest = build_manifest(root, warmups=2, samples=3)
            self.assertEqual(len(manifest.requests), len(FORMAT_SPECS) * 2)
            for request in manifest.requests:
                validate_request(request)
                source = next(source for source in FORMAT_SPECS if source.label == request.source_format)
                self.assertEqual(request.runtime_codebook_id, source.runtime_codebook("rocm"))
                if request.config_json["purpose"] == "winner":
                    self.assertEqual(request.candidate_family, "NativeWmma")
                    self.assertEqual(request.config_json["requested_candidate"], "WMMA/N64/MT32")
                else:
                    self.assertEqual(request.config_json["requested_candidate"], "Auto")

    def test_wmma_emitter_requires_native_gfx1100_evidence(self) -> None:
        """A synthetic exact row tests emission only, never native qualification."""
        entry = DenseOverlayEntry("rocm", DenseOverlayKey(0, 64, 128, 256,
            DensePreparedWeightFamily.NATIVE_QUANTISED, GPUArchitecture.parse("rocm", "gfx1100")),
            ("Q4_0",), ("test",), "WMMA/N64/MT32", (64, 32, 1, 0, 0, 4), 0.0, 0.0, 0.0)
        self.assertIn("VNNIPrefillProducer::NativeWmma", render_dense_overlay((entry,)))
        legacy = replace(entry, key=replace(entry.key, architecture=None),
                         candidate_id="N64/MT32/MB1/U2", launch=(64, 32, 1, 2, 0, 0))
        base = render_dense_overlay((legacy, entry))
        retained, receipt = retain_dense_overlay_base(base, (entry,))
        self.assertEqual(receipt["retained_keys"], 1)
        self.assertEqual(receipt["replaced_keys"], 1)
        self.assertEqual(retained, base)
        for architecture in (None, GPUArchitecture.parse("rocm", "gfx906")):
            with self.assertRaisesRegex(ValueError, "gfx1100"):
                render_dense_overlay((replace(entry, key=replace(entry.key, architecture=architecture)),))

    def test_native_low_bit_publication_requires_every_shared_codebook_alias(self) -> None:
        """An installed physical row applies to both block32 and superblock sources."""
        for codebook, aliases in ((5, ("Q4_1", "Q4_K")), (7, ("Q5_1", "Q5_K"))):
            entry = DenseOverlayEntry("rocm", DenseOverlayKey(codebook, 64, 128, 256,
                DensePreparedWeightFamily.NATIVE_QUANTISED, GPUArchitecture.parse("rocm", "gfx1100")),
                aliases, ("test",), "WMMA/N64/MT32", (64, 32, 1, 0, 0, 4), 0.0, 0.0, 0.0)
            self.assertIn("VNNIPrefillProducer::NativeWmma", render_dense_overlay((entry,)))
            for alias in aliases:
                with self.assertRaisesRegex(ValueError, "complete source-alias evidence"):
                    render_dense_overlay((replace(entry, source_formats=(alias,)),))

    def test_native_q8_family_identity_preserves_old_keys(self) -> None:
        old = self._native_q8_source()
        explicit_old = replace(old, prepared_family=DensePreparedWeightFamily.INT8_BLOCKWISE)
        native = replace(old, prepared_family=DensePreparedWeightFamily.NATIVE_QUANTISED, gpu_architecture="gfx906")
        self.assertEqual(old, explicit_old)
        self.assertEqual(old.cell_id, explicit_old.cell_id)
        self.assertNotIn("prepared_family", old.canonical_mapping())
        self.assertEqual(old.identity, (old.backend, old.source_format.label,
            old.source_format.source_codebook_id, old.source_format.runtime_codebook("rocm"),
            old.shape.name, old.shape.n, old.shape.k, old.m))
        self.assertNotEqual(old.cell_id, native.cell_id)
        self.assertEqual(native.canonical_mapping()["prepared_family"], "native_quantised")
        self.assertEqual(dense_prefill_candidate_ids("rocm", native.source_format,
            prepared_family=native.prepared_family), ROCM_NATIVE_Q8_CANDIDATE_IDS)
        self.assertTrue(all(candidate.startswith("INT8") or candidate == "Auto"
                            for candidate in dense_prefill_candidate_ids("rocm", old.source_format)))
        with self.assertRaisesRegex(ValueError, "typed prepared family"):
            replace(old, prepared_family=0)
        with self.assertRaisesRegex(ValueError, "original ROCm Q8"):
            dense_prefill_candidate_ids("cuda", old.source_format,
                prepared_family=DensePreparedWeightFamily.INT8_BLOCKWISE)

    def test_native_q8_complete_alias_records_use_strong_reader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cells = tuple(replace(self._native_q8_source(label),
                prepared_family=DensePreparedWeightFamily.NATIVE_QUANTISED, gpu_architecture="gfx906")
                for label in ("Q8_0", "Q8_1", "Q8_K"))
            for cell in cells:
                _, paths = _write_cell(root, "rocm", cell=cell)
                result = validate_production_dense_prefill_cell(paths.aggregate, paths.timing, cell, bench_runs=3)
                self.assertEqual(result.candidate_count, 4)
                self.assertEqual(result.sample_count, 12)
            _commit_test_cells(root, "rocm", cells)
            loaded, plan = load_planned_dense_prefill_cells(root / SWEEP_PLAN_FILENAME, "rocm")
            self.assertEqual(loaded, cells)
            entries = collect_dense_overlay_entries(root, "rocm", cells=cells, warmup_runs=2, bench_runs=3)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].key.prepared_family, DensePreparedWeightFamily.NATIVE_QUANTISED)
            self.assertEqual(entries[0].source_formats, ("Q8_0", "Q8_1", "Q8_K"))
            self.assertEqual(entries[0].launch, (64, 16, 1, 2, 0, 0))
            validate_dense_overlay_totality(entries, cells, "rocm")
            self.assertEqual(plan["candidate_ids_by_source_format"]["Q8_K"], list(ROCM_NATIVE_Q8_CANDIDATE_IDS))

    def test_native_q8_cannot_relabel_expanded_clocks(self) -> None:
        old = self._native_q8_source()
        native = replace(old, prepared_family=DensePreparedWeightFamily.NATIVE_QUANTISED, gpu_architecture="gfx906")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, old_paths = _write_cell(root, "rocm", cell=old)
            with self.assertRaisesRegex(ValueError, "header"):
                validate_production_dense_prefill_cell(old_paths.aggregate, old_paths.timing, native, bench_runs=3)
            _, paths = _write_cell(root, "rocm", cell=native)
            with paths.aggregate.open(newline="") as handle:
                reader = csv.DictReader(handle); columns = reader.fieldnames; rows = list(reader)
            rows[-1]["observed_producer"] = "int8_blockwise_v7"
            with paths.aggregate.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns);writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "invalid observed ROCm launch tuple"):
                validate_production_dense_prefill_cell(paths.aggregate, paths.timing, native, bench_runs=3)

    def test_native_q8_old_complete_records_reuse_original_producer(self) -> None:
        for native_scope in (False, True):
            with self.subTest(native=native_scope), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp);cell = self._native_q8_source()
                if native_scope:
                    cell = replace(cell, prepared_family=DensePreparedWeightFamily.NATIVE_QUANTISED, gpu_architecture="gfx906")
                _, paths = _write_cell(root, "rocm", cell=cell)
                producer = _test_provenance("rocm")
                _commit_test_cells(root, "rocm", (cell,), provenance=producer)
                before = {path.name: path.read_bytes() for path in (paths.aggregate, paths.timing, paths.manifest)}
                binary = root / "trainer";binary.write_text("#!/bin/sh\nexit 0\n");binary.chmod(0o755)
                with mock.patch("native_quantised_dispatch.production_dense_prefill_sweep._run_accelerator_cell") as launch:
                    completed = run_missing_accelerator_cells(binary, root, (cell,), (0,),
                        warmup_runs=2, bench_runs=3, producer=producer)
                self.assertEqual(completed, ());launch.assert_not_called()
                self.assertEqual(before, {path.name: path.read_bytes() for path in (paths.aggregate, paths.timing, paths.manifest)})
                other = replace(cell, prepared_family=(DensePreparedWeightFamily.INT8_BLOCKWISE
                    if native_scope else DensePreparedWeightFamily.NATIVE_QUANTISED),
                    gpu_architecture=None if native_scope else "gfx906")
                with self.assertRaises(ValueError):
                    write_sweep_plan(root / SWEEP_PLAN_FILENAME, "rocm", (other,), producer, warmup_runs=2, bench_runs=3)
                self.assertEqual(before, {path.name: path.read_bytes() for path in (paths.aggregate, paths.timing, paths.manifest)})

    def test_native_q8_emitter_retains_expanded_family_rows(self) -> None:
        old = self._native_q8_source()
        native = replace(old, prepared_family=DensePreparedWeightFamily.NATIVE_QUANTISED, gpu_architecture="gfx906")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);expanded_root = root / "expanded";expanded_root.mkdir()
            native_root = root / "native";native_root.mkdir()
            _write_cell(expanded_root, "rocm", cell=old);_commit_test_cells(expanded_root, "rocm", (old,))
            aliases = tuple(replace(native, source_format=self._native_q8_source(label).source_format) for label in ("Q8_0", "Q8_1", "Q8_K"))
            for alias in aliases:
                _write_cell(native_root, "rocm", cell=alias)
            _commit_test_cells(native_root, "rocm", aliases)
            expanded = collect_dense_overlay_entries(expanded_root, "rocm", cells=(old,), warmup_runs=2, bench_runs=3)
            measured = collect_dense_overlay_entries(native_root, "rocm", cells=aliases, warmup_runs=2, bench_runs=3)
            self.assertEqual(expanded[0].key.prepared_family, DensePreparedWeightFamily.INT8_BLOCKWISE)
            base = render_dense_overlay(expanded)
            merged, receipt = retain_dense_overlay_base(base, measured)
            self.assertEqual(receipt["retained_keys"], 1)
            self.assertEqual(receipt["replaced_keys"], 0)
            self.assertIn("DensePrefillWeightFamily::Int8Blockwise", merged)
            self.assertIn("DensePrefillWeightFamily::NativeQuantised", merged)
            self.assertEqual(retain_dense_overlay_base(merged, measured)[0], merged)

    def test_native_q8_environment_and_batch_keep_family_bound(self) -> None:
        old = self._native_q8_source();native = replace(old, prepared_family=DensePreparedWeightFamily.NATIVE_QUANTISED, gpu_architecture="gfx906")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ,
            {"LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_PREPARED_FAMILY": "native_quantised"}):
            root = Path(tmp)
            old_environment = _cell_environment(old, production_dense_prefill_cell_paths(root, old), 2, 2, 3)
            environment = _cell_environment(native, production_dense_prefill_cell_paths(root, native), 2, 2, 3)
            self.assertNotIn("LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_PREPARED_FAMILY", old_environment)
            self.assertEqual(environment["LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_PREPARED_FAMILY"], "native_quantised")
            self.assertEqual(environment["LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_VARIANTS"].split(","), list(ROCM_NATIVE_Q8_CANDIDATE_IDS))
            self.assertEqual(environment["HIP_VISIBLE_DEVICES"], "2")
            with self.assertRaisesRegex(ValueError, "one format and geometry"):
                _batch_environment((old, replace(native, m=448)), _batch_paths(root, (old, replace(native, m=448))), 2, 2, 3)
            with self.assertRaisesRegex(ValueError, "one prepared-family protocol"):
                write_sweep_plan(root / SWEEP_PLAN_FILENAME, "rocm", (old, native), _test_provenance("rocm"))

    def test_native_all_format_runner_executes_the_planned_candidate_inventory(self) -> None:
        """Single and batched processes cannot silently prune planned WMMA candidates."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for architecture in ("gfx906", "gfx1100"):
                for source_format in FORMAT_SPECS:
                    with self.subTest(architecture=architecture, source=source_format.label):
                        cell = replace(self._native_q8_source(), source_format=source_format,
                            prepared_family=DensePreparedWeightFamily.NATIVE_QUANTISED,
                            gpu_architecture=architecture)
                        expected = dense_prefill_candidate_ids("rocm", source_format,
                            prepared_family=cell.prepared_family, gpu_architecture=architecture)
                        environment = _cell_environment(cell,
                            production_dense_prefill_cell_paths(root, cell), 5, 2, 3)
                        selected = tuple(environment["LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_VARIANTS"].split(","))
                        self.assertEqual(selected, expected)
                        self.assertEqual(set(selected) & set(ROCM_WMMA_CANDIDATE_IDS),
                            set(ROCM_WMMA_CANDIDATE_IDS) if architecture == "gfx1100" else set())
                        cells = (cell, replace(cell, m=cell.m + 64))
                        batched = _batch_environment(cells, _batch_paths(root, cells), 5, 2, 3)
                        self.assertEqual(batched["LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_VARIANTS"],
                            environment["LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_VARIANTS"])

    def test_gfx1100_counter_policy_requires_measured_dispatch_activity(self) -> None:
        """Missing VALUBusy stays explicit; gated-off perfmon clocks cannot certify a kernel."""
        from native_quantised_dispatch.profiler_collectors import (
            _complete_metric_inventory, _kernel_fingerprint, _rocm_counter_groups)
        from native_quantised_dispatch.profiler_evidence import (
            Backend, ProfiledDispatch, ProfiledDispatchKind)
        counters = {name for group in _rocm_counter_groups("gfx1100") for name in group}
        self.assertEqual(counters, {"GPUBusy", "MemUnitBusy", "L2CacheHit", "FETCH_SIZE",
                                   "Wavefronts", "LDSBankConflict", "VALUInsts"})
        self.assertIn("VALUBusy", {name for group in _rocm_counter_groups("gfx906") for name in group})
        with self.assertRaisesRegex(ValueError, "no reviewed counter policy"):
            _rocm_counter_groups("gfx1101")
        values = {name: (value, name) for name, value in {
            "gpu.duration_ns": 1000, "gpu.vgpr_count": 100, "gpu.sgpr_count": 128,
            "gpu.lds_bytes": 17920, "gpu.scratch_bytes": 0, "gpu.gpu_busy_pct": 100,
            "gpu.memory_unit_busy_pct": 59, "gpu.wavefront_count": 8192}.items()}
        def dispatch(measured):
            """Construct the physical record through the same strict metric completer."""
            return ProfiledDispatch(0, ProfiledDispatchKind.GPU_KERNEL, "native_wmma",
                _kernel_fingerprint("native_wmma", (256, 1, 1), (256, 1, 1)),
                (256, 1, 1), (256, 1, 1), _complete_metric_inventory(Backend.ROCM, measured,
                    unavailable_reason="SDK has no counter", architecture_class="gfx1100"))
        dispatch(values).validate(Backend.ROCM, "gfx1100")
        with self.assertRaisesRegex(ValueError, "valu_busy_pct"):
            dispatch(values).validate(Backend.ROCM, "gfx906")
        for name in ("gpu.gpu_busy_pct", "gpu.memory_unit_busy_pct", "gpu.wavefront_count"):
            with self.assertRaisesRegex(ValueError, "required profiler metric"):
                dispatch({key: value for key, value in values.items() if key != name}).validate(Backend.ROCM, "gfx1100")
        for name in ("gpu.gpu_busy_pct", "gpu.wavefront_count"):
            with self.assertRaisesRegex(ValueError, "positive dispatch activity"):
                dispatch({**values, name: (0, name)}).validate(Backend.ROCM, "gfx1100")

    def test_gfx1100_counter_policy_retains_legacy_collector_requirements(self) -> None:
        """Retained v8 records keep optional waves; new native evidence requires v9."""
        from types import SimpleNamespace
        from native_quantised_dispatch.profiler_collectors import _complete_metric_inventory
        from native_quantised_dispatch.profiler_evidence import (
            Backend, EXPECTED_PROFILER_TOOL, PROFILER_COLLECTOR_VERSION,
            PROFILER_METRIC_SET_VERSION, ProfiledDispatch, ProfiledDispatchKind,
            ProfilerEvidence, ProfilerEvidenceStatus, metric_definitions)
        values = {definition.metric_id: (0 if definition.metric_id == "gpu.scratch_bytes" else 1,
                  definition.metric_id)
                  for definition in metric_definitions(Backend.ROCM) if definition.required}
        dispatch = ProfiledDispatch(0, ProfiledDispatchKind.GPU_KERNEL, "retained_kernel",
            "sha256:" + "1" * 64, (1, 1, 1), (256, 1, 1),
            _complete_metric_inventory(Backend.ROCM, values,
                unavailable_reason="optional in the original v8 collection"))
        request = SimpleNamespace(request_id="retained-v8", observation_digest="sha256:" + "2" * 64,
            backend=Backend.ROCM, profile_required=True, architecture_class="gfx1100",
            operation_kind="NativeQuantisedDecodeProjection")
        evidence = ProfilerEvidence(request.request_id, request.observation_digest, Backend.ROCM,
            ProfilerEvidenceStatus.COMPLETE, None, EXPECTED_PROFILER_TOOL[Backend.ROCM], "fixture-v8",
            PROFILER_METRIC_SET_VERSION, "native-quantised-isolated-profiler-v8-ncu-si-byte-units",
            "sha256:" + "3" * 64, "sha256:" + "4" * 64, 1, 1, (dispatch,))
        evidence.validate(request)
        # A new collector cannot inherit the historical absence of wave counts.
        with self.assertRaisesRegex(ValueError, "wavefront_count.*required"):
            replace(evidence, collector_version=PROFILER_COLLECTOR_VERSION).validate(request)
        native_request = SimpleNamespace(**{**vars(request),
            "operation_kind": "ROCmNativeQuantisedDensePrefill"})
        with self.assertRaisesRegex(ValueError, "authenticated native counter policy"):
            evidence.validate(native_request)

    def test_gfx1100_profile_batch_rejects_before_launch(self) -> None:
        """A batch cannot bypass the isolated native power-state authority."""
        from types import SimpleNamespace
        from native_quantised_dispatch.profiler_collectors import (
            CollectorOptions, GPUProfilerProcessBatch, _collect_rocm_batch)
        from native_quantised_dispatch.profiler_evidence import Backend
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            options = CollectorOptions(Backend.ROCM, root / "trainer", root / "rocprofv3", root)
            batch = GPUProfilerProcessBatch((SimpleNamespace(
                backend=Backend.ROCM, architecture_class="gfx1100"),))
            with mock.patch("native_quantised_dispatch.profiler_collectors._run_command") as launch, \
                 mock.patch("native_quantised_dispatch.profiler_collectors._binary_for_request") as binary:
                with self.assertRaisesRegex(ValueError, "power-state ownership.*batching is not admitted"):
                    _collect_rocm_batch(batch, options)
                launch.assert_not_called()
                binary.assert_not_called()
            self.assertEqual(tuple(root.iterdir()), ())

    def test_native_prefill_profile_rejects_rebuilt_executable_or_core_before_power_changes(self) -> None:
        """Identical tile dimensions cannot relabel another wave/code-generation policy."""
        from types import SimpleNamespace
        from native_quantised_dispatch import native_prefill_profiler as owner
        from native_quantised_dispatch.corpus_provenance import mapping_digest
        from native_quantised_dispatch.profiler_collectors import CollectorOptions, _collect_rocm
        from native_quantised_dispatch.profiler_evidence import Backend
        original = _test_provenance("rocm")
        request = SimpleNamespace(operation_kind=owner.OPERATION, architecture_class="gfx906",
            build_id=mapping_digest(original), validate=lambda: None)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            options = CollectorOptions(Backend.ROCM, root / "trainer", root / "rocprofv3", root)
            for changed in ("binary_sha256", "core_library_sha256"):
                with self.subTest(changed=changed), \
                        mock.patch.object(owner, "trainer_provenance", return_value={**original, changed: "sha256:" + "9" * 64}), \
                        mock.patch("native_quantised_dispatch.profiler_collectors._rocm_native_power_scope") as power, \
                        mock.patch("native_quantised_dispatch.profiler_collectors._collect_rocm_admitted") as collect:
                    with self.assertRaisesRegex(ValueError, "exact timing executable/core closure"):
                        _collect_rocm(request, options, root)
                    power.assert_not_called()
                    collect.assert_not_called()
            self.assertEqual(tuple(root.iterdir()), ())
            with mock.patch.object(owner, "trainer_provenance", return_value=original):
                self.assertEqual(owner.validate_binary(request, options.binary), original)

    def test_native_prefill_profile_rechecks_closure_before_restoring_power_and_sealing(self) -> None:
        """A mutation during counter collection fails after guaranteed power cleanup."""
        from contextlib import contextmanager
        from types import SimpleNamespace
        from native_quantised_dispatch import native_prefill_profiler as owner
        from native_quantised_dispatch.corpus_provenance import mapping_digest
        from native_quantised_dispatch.profiler_collectors import CollectorOptions, _collect_rocm
        from native_quantised_dispatch.profiler_evidence import Backend
        original = _test_provenance("rocm")
        request = SimpleNamespace(operation_kind=owner.OPERATION, architecture_class="gfx906",
            build_id=mapping_digest(original), validate=lambda: None)
        restored = []
        @contextmanager
        def power(*args):
            try:
                yield
            finally:
                restored.append(True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            options = CollectorOptions(Backend.ROCM, root / "trainer", root / "rocprofv3", root)
            with mock.patch.object(owner, "trainer_provenance", side_effect=[original,
                    {**original, "core_library_sha256": "sha256:" + "9" * 64}]), \
                    mock.patch("native_quantised_dispatch.profiler_collectors._rocm_native_power_scope", side_effect=power), \
                    mock.patch("native_quantised_dispatch.profiler_collectors._collect_rocm_admitted") as collect:
                with self.assertRaisesRegex(ValueError, "exact timing executable/core closure"):
                    _collect_rocm(request, options, root)
                collect.assert_called_once()
            self.assertEqual(restored, [True])
            receipt = json.loads((root / "native-executable-closure.json").read_text())
            self.assertEqual(receipt["profiler_closure"], original)

    def test_gfx1100_profile_power_scope_preserves_existing_stable_policy(self) -> None:
        """An already stable participant retains its caller-owned policy without writes."""
        from types import SimpleNamespace
        from native_quantised_dispatch.profiler_collectors import CollectorOptions, _rocm_native_power_scope
        from native_quantised_dispatch.profiler_evidence import Backend
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tool = root / "rocprofv3"; tool.write_text("profiler fixture")
            (root / "amd-smi").write_text("same-SDK management fixture")
            options = CollectorOptions(Backend.ROCM, root / "trainer", tool, root)
            request = SimpleNamespace(architecture_class="gfx1100",
                config_json={"native_participant": {"gpu_pci": "0000:b5:00.0"}})
            stable = "AMDSMI_DEV_PERF_LEVEL_STABLE_STD"
            def query(command, environment, log_path, timeout_seconds):
                """Reject every management write while answering the three state queries."""
                self.assertIn("metric", command)
                self.assertNotIn("set", command)
                return subprocess.CompletedProcess(command, 0,
                    json.dumps({"gpu_data": [{"perf_level": stable}]}), "")
            with mock.patch("native_quantised_dispatch.profiler_collectors._run_command", side_effect=query) as run, \
                 mock.patch("native_quantised_dispatch.profiler_collectors.tempfile.gettempdir", return_value=tmp):
                with _rocm_native_power_scope(request, options, root):
                    pass
            self.assertEqual(run.call_count, 3)
            receipt = json.loads((root / "native-power-state.json").read_text())
            self.assertEqual((receipt["before"], receipt["during"], receipt["after"]), (stable,) * 3)
            self.assertEqual(receipt["restoration_errors"], [])

    def test_gfx1100_profile_power_scope_restores_after_failed_collection(self) -> None:
        """A failed counter pass still restores and authenticates the original policy."""
        from types import SimpleNamespace
        from native_quantised_dispatch.profiler_collectors import (
            CollectorOptions, ROCmProfilerPowerRestorationError, _rocm_native_power_scope,
            _tool_command, collect_request)
        from native_quantised_dispatch.profiler_evidence import Backend
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tool = root / "rocprofv3"; tool.write_text("profiler fixture")
            (root / "amd-smi").write_text("same-SDK management fixture")
            options = CollectorOptions(Backend.ROCM, root / "trainer", tool, root,
                rocm_power_command_prefix=("sudo", "-n"))
            self.assertEqual(_tool_command(options, "--version"), [str(tool), "--version"])
            request = SimpleNamespace(architecture_class="gfx1100",
                config_json={"native_participant": {"gpu_pci": "0000:b5:00.0"}})
            mode = "AUTO"
            failure = None
            changes = []
            def run(command, environment, log_path, timeout_seconds):
                """Model the native CLI state transition without touching a device."""
                nonlocal mode
                if "set" in command:
                    self.assertEqual(command[:2], ["sudo", "-n"])
                    if command[-1] == "AUTO" and failure == "command":
                        return subprocess.CompletedProcess(command, 1, "", "restore rejected")
                    if not (command[-1] == "AUTO" and failure == "wrong-state"):
                        mode = command[-1]
                    changes.append(command[-1])
                if log_path.name == "power-after.log" and failure == "query":
                    return subprocess.CompletedProcess(command, 1, "", "state unavailable")
                payload = {"gpu_data": [{"perf_level": "AMDSMI_DEV_PERF_LEVEL_" + mode}]}
                return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
            with mock.patch("native_quantised_dispatch.profiler_collectors._run_command", side_effect=run), \
                 mock.patch("native_quantised_dispatch.profiler_collectors.tempfile.gettempdir", return_value=tmp):
                with self.assertRaisesRegex(RuntimeError, "counter failure"):
                    with _rocm_native_power_scope(request, options, root):
                        self.assertEqual(mode, "STABLE_STD")
                        raise RuntimeError("counter failure")
                self.assertEqual(mode, "AUTO")
                self.assertEqual(changes, ["STABLE_STD", "AUTO"])
                receipt = json.loads((root / "native-power-state.json").read_text())
                self.assertEqual(receipt["before"], receipt["after"])
                self.assertEqual(receipt["during"], "AMDSMI_DEV_PERF_LEVEL_STABLE_STD")
                mode = "MANUAL"; changes.clear()
                with self.assertRaisesRegex(ValueError, "state left unchanged"):
                    with _rocm_native_power_scope(request, options, root):
                        self.fail("unsupported power policy entered collection")
                self.assertEqual(mode, "MANUAL")
                self.assertEqual(changes, [])
                for failure in ("command", "wrong-state", "query"):
                    with self.subTest(failure=failure):
                        mode = "AUTO"
                        with self.assertRaises(ROCmProfilerPowerRestorationError):
                            with _rocm_native_power_scope(request, options, root):
                                pass
                        receipt = json.loads((root / "native-power-state.json").read_text())
                        self.assertTrue(receipt["restoration_errors"])
                        self.assertNotEqual(receipt["after"], receipt["before"])
            # Restoration failure escapes per-request failure conversion, so a
            # collection loop cannot launch its next request in a leaked mode.
            options.binary.write_text("test executable"); options.binary.chmod(0o755); tool.chmod(0o755)
            routed = SimpleNamespace(backend=Backend.ROCM, profile_required=True, request_id="fatal-policy")
            with mock.patch("native_quantised_dispatch.profiler_collectors._write_binary_provenance"), \
                 mock.patch("native_quantised_dispatch.profiler_collectors._collect_rocm",
                    side_effect=ROCmProfilerPowerRestorationError("unrestored")) as collect:
                with self.assertRaises(ROCmProfilerPowerRestorationError):
                    for _ in range(2):
                        collect_request(routed, options)
                self.assertEqual(collect.call_count, 1)

    def test_native_q8_architecture_identity_and_receipt_are_strict(self) -> None:
        """Authored metadata cannot relabel a different architecture or participant."""
        cell = replace(self._native_q8_source(), prepared_family=DensePreparedWeightFamily.NATIVE_QUANTISED, gpu_architecture="gfx906")
        other = replace(cell, gpu_architecture="gfx1100")
        self.assertNotEqual(cell.cell_id, other.cell_id)
        with tempfile.TemporaryDirectory() as tmp:
            _, paths = _write_cell(Path(tmp), "rocm", cell=cell)
            with self.assertRaisesRegex(ValueError, "architecture"):
                validate_production_dense_prefill_cell(paths.aggregate, paths.timing, other, bench_runs=3)
            for field, value in (("gpu_uuid", "0" * 32), ("gpu_pci", "missing"), ("native_ordinal", "1")):
                _, paths = _write_cell(Path(tmp), "rocm", cell=cell)
                with paths.aggregate.open(newline="") as handle:
                    reader = csv.DictReader(handle); columns = reader.fieldnames; rows = list(reader)
                rows[0][field] = value
                with paths.aggregate.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
                with self.assertRaisesRegex(ValueError, "participant"):
                    validate_production_dense_prefill_cell(paths.aggregate, paths.timing, cell, bench_runs=3)

    def test_native_q8_two_architecture_retention_keeps_legacy_rows(self) -> None:
        """A new architecture delta replaces only its own complete runtime key."""
        first = DenseOverlayEntry("rocm", DenseOverlayKey(19, 64, 320, 2560,
            architecture=GPUArchitecture.parse("rocm", "gfx906")),
            ("Q8_0", "Q8_1", "Q8_K"), ("authored",), "N64/MT16/MB1/U2", (64, 16, 1, 2, 0, 0), 1., 1., 1.)
        second = replace(first, key=replace(first.key, architecture=GPUArchitecture.parse("rocm", "gfx1100")))
        base = render_dense_overlay((first, second))
        changed = replace(first, launch=(64, 32, 1, 2, 0, 0))
        output, receipt = retain_dense_overlay_base(base, (changed,))
        self.assertEqual(receipt["replaced_keys"], 1)
        self.assertEqual(receipt["retained_keys"], 1)
        self.assertIn(second.key.architecture.cpp_literal(), output)
        with self.assertRaisesRegex(ValueError, "source-alias"):
            render_dense_overlay((replace(first, source_formats=("Q8_0",)),))
        with self.assertRaisesRegex(ValueError, "architecture"):
            render_dense_overlay((replace(first, key=replace(first.key, architecture=None)),))

    def test_low_bit_legacy_and_qualified_rows_keep_generator_order(self) -> None:
        """Additive WMMA publication preserves legacy and other-target rows at one key."""
        legacy = DenseOverlayEntry("rocm", DenseOverlayKey(0, 64, 128, 256),
            ("Q4_0",), ("authored",), "N64/MT16/MB1/U2", (64, 16, 1, 2, 0, 0), 0., 0., 1.)
        older = replace(legacy, key=replace(legacy.key, architecture=GPUArchitecture.parse("rocm", "gfx906")),
                        launch=(64, 32, 1, 2, 0, 0))
        wmma = replace(legacy, key=replace(legacy.key, architecture=GPUArchitecture.parse("rocm", "gfx1100")),
                       candidate_id="WMMA/N64/MT64", launch=(64, 64, 1, 0, 0, 4))
        base = render_dense_overlay((legacy, older))
        output, receipt = retain_dense_overlay_base(base, (wmma,))
        self.assertEqual((receipt["retained_keys"], receipt["replaced_keys"], receipt["output_keys"]), (2, 0, 3))
        rows = [line for line in output.splitlines() if line.startswith("        {0,")]
        self.assertEqual(len(rows), 3)
        base_rows = [line for line in base.splitlines() if line.startswith("        {0,")]
        self.assertEqual(rows[:2], base_rows)
        self.assertIn(wmma.key.architecture.cpp_literal(), rows[2])
        self.assertEqual(retain_dense_overlay_base(output, (wmma,))[0], output)
        changed, receipt = retain_dense_overlay_base(output, (replace(wmma, launch=(128, 64, 1, 0, 0, 4)),))
        self.assertEqual(receipt["replaced_keys"], 1)
        for row in base_rows:
            self.assertIn(row, changed)

    def test_native_q8_profile_requests_preserve_unforced_auto(self) -> None:
        """Original common profiling lowers the physical baseline to actual Auto."""
        from native_quantised_dispatch.native_prefill_profiler import build_manifest, environment, validate_request
        from native_quantised_dispatch.profiler_collectors import _profile_environment, _apply_device_placement, CollectorOptions
        cells = tuple(replace(self._native_q8_source(label), prepared_family=DensePreparedWeightFamily.NATIVE_QUANTISED,
                              gpu_architecture="gfx906") for label in ("Q8_0", "Q8_1", "Q8_K"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cell in cells:
                _write_cell(root, "rocm", cell=cell)
            _commit_test_cells(root, "rocm", cells)
            manifest = build_manifest(root, warmups=2, samples=3)
            self.assertEqual(len(manifest.requests), 6)
            for request in manifest.requests:
                validate_request(request)
                env = _profile_environment(request, root)
                self.assertNotIn("LLAMINAR_ROCM_NATIVE_QUANTISED_DECODE_VARIANTS", env)
                self.assertEqual(env["LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_PREPARED_FAMILY"], "native_quantised")
                self.assertEqual(env["LLAMINAR_ROCM_DENSE_PRODUCTION_PROFILE_CANDIDATE"], request.config_json["requested_candidate"])
                if request.config_json["purpose"] == "baseline":
                    self.assertEqual(request.config_json["requested_candidate"], "Auto")
                    self.assertNotEqual(request.candidate_id, "Auto")
            wrong = replace(manifest.requests[0], config_json={**manifest.requests[0].config_json, "launch": [128, 32, 1, 2, 0, 0]})
            with self.assertRaisesRegex(ValueError, "witness"):
                validate_request(wrong)

    def test_native_q8_publication_authenticates_profiler_schema(self) -> None:
        """Publish typed evidence through the real readers and reject stale versions."""
        from native_quantised_dispatch.native_prefill_profiler import build_manifest, require_profile
        from native_quantised_dispatch.profiler_evidence import (
            EXPECTED_PROFILER_TOOL, PROFILER_COLLECTOR_VERSION,
            PROFILER_METRIC_SET_VERSION, MetricAvailability, ProfiledDispatch,
            ProfiledDispatchKind, ProfilerEvidence, ProfilerEvidenceManifest,
            ProfilerEvidenceStatus, ProfilerMetric, metric_definitions,
            write_profiler_evidence_manifest, write_profiler_request_manifest,
        )

        cells = tuple(replace(self._native_q8_source(label),
            prepared_family=DensePreparedWeightFamily.NATIVE_QUANTISED,
            gpu_architecture="gfx906") for label in ("Q8_0", "Q8_1", "Q8_K"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cell in cells:
                _write_cell(root, "rocm", cell=cell)
            _commit_test_cells(root, "rocm", cells)
            requests = build_manifest(root, warmups=2, samples=3)
            request_path = root / "profile-requests.json"
            evidence_path = root / "profile-evidence.json"
            write_profiler_request_manifest(request_path, requests)
            records = []
            for request in requests.requests:
                metrics = tuple(ProfilerMetric(
                    metric_id=definition.metric_id, category=definition.category,
                    unit=definition.unit, availability=MetricAvailability.MEASURED,
                    value=0.0 if definition.metric_id == "gpu.scratch_bytes" else 1.0,
                    source_name="unit::" + definition.metric_id, reason=None,
                ) for definition in metric_definitions(request.backend))
                dispatch = ProfiledDispatch(
                    dispatch_index=0, dispatch_kind=ProfiledDispatchKind.GPU_KERNEL,
                    kernel_name="unitNativeQ8Kernel", kernel_fingerprint="sha256:" + "1" * 64,
                    grid=(1, 1, 1), block=(64, 1, 1), metrics=metrics,
                )
                records.append(ProfilerEvidence(
                    request_id=request.request_id, observation_digest=request.observation_digest,
                    backend=request.backend, status=ProfilerEvidenceStatus.COMPLETE,
                    status_reason=None, profiler_tool=EXPECTED_PROFILER_TOOL[request.backend],
                    profiler_tool_version="unit-profiler-1", metric_set_version=PROFILER_METRIC_SET_VERSION,
                    collector_version=PROFILER_COLLECTOR_VERSION,
                    command_digest="sha256:" + "2" * 64, raw_artifact_digest="sha256:" + "3" * 64,
                    profiler_pass_count=1, target_launches_per_profiler_pass=1, dispatches=(dispatch,),
                ))
            evidence = ProfilerEvidenceManifest(
                request_manifest_digest=requests.digest(), corpus_digest=requests.corpus_digest,
                candidate_registry_digest=requests.candidate_registry_digest, evidence=tuple(records),
            )
            write_profiler_evidence_manifest(evidence_path, evidence)
            # No reader or typed field is mocked: this is the formerly failing
            # publication boundary, including every source alias and Auto control.
            require_profile(root, request_path, evidence_path, warmups=2, samples=3)

            stale_record = replace(records[0], metric_set_version="stale-metric-set")
            scratch_dispatch = replace(records[0].dispatches[0], metrics=tuple(
                replace(metric, value=1.0) if metric.metric_id == "gpu.scratch_bytes" else metric
                for metric in records[0].dispatches[0].metrics))
            scratch_record = replace(records[0], dispatches=(scratch_dispatch,))
            for label, changed, message in (
                ("record version", replace(evidence, evidence=(stale_record, *records[1:])),
                 "unsupported profiler metric set"),
                ("missing control", replace(evidence, evidence=tuple(records[1:])),
                 "coverage is incomplete"),
                ("scratch", replace(evidence, evidence=(scratch_record, *records[1:])),
                 "nonzero local/scratch traffic"),
            ):
                with self.subTest(case=label):
                    write_profiler_evidence_manifest(evidence_path, changed)
                    with self.assertRaisesRegex(ValueError, message):
                        require_profile(root, request_path, evidence_path, warmups=2, samples=3)

            raw = evidence.canonical_mapping()
            raw["metric_set_version"] = "stale-metric-set"
            evidence_path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "unsupported profiler evidence metric set"):
                require_profile(root, request_path, evidence_path, warmups=2, samples=3)

    def test_native_q8_missing_raw_samples_prevents_completion(self) -> None:
        cell = replace(self._native_q8_source(), prepared_family=DensePreparedWeightFamily.NATIVE_QUANTISED, gpu_architecture="gfx906")
        with tempfile.TemporaryDirectory() as tmp:
            _, paths = _write_cell(Path(tmp), "rocm", cell=cell)
            lines = paths.timing.read_text().splitlines(keepends=True)
            paths.timing.write_text("".join(lines[:-1]))
            with self.assertRaisesRegex(ValueError, "samples, expected"):
                validate_production_dense_prefill_cell(paths.aggregate, paths.timing, cell, bench_runs=3)

    def test_matrix_is_all_format_shape_and_m_complete_on_both_backends(self) -> None:
        expected_count = (
            len(FORMAT_SPECS)
            * len(gpu_prefill_measurements())
            * len(GPU_PREFILL_M_BUCKETS)
        )
        self.assertEqual(len(FORMAT_SPECS), 21)
        self.assertIn(448, GPU_PREFILL_M_BUCKETS)
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
        self.assertEqual(len(CUDA_CANDIDATE_IDS), 44)
        self.assertEqual(len(CUDA_GENERIC_CANDIDATE_IDS), 43)
        self.assertFalse(any(candidate.startswith("SK1") for candidate in CUDA_CANDIDATE_IDS))
        self.assertFalse(any(candidate.startswith("SK2") for candidate in CUDA_CANDIDATE_IDS))
        self.assertEqual(
            sum(candidate.startswith("KPART:") for candidate in CUDA_CANDIDATE_IDS),
            len(CUDA_TILE_NAMES) + 3 * 5,
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

    def test_rocm_iq4xs_packets_require_actual_physical24(self) -> None:
        """Source4 stays original; neither packet can silently claim old4."""
        cell = next(cell for cell in production_dense_prefill_cells("rocm")
                    if cell.source_format.label == "IQ4_XS")
        for field in ("aggregate", "timing"):
            with self.subTest(packet=field), tempfile.TemporaryDirectory() as tmp:
                _, paths = _write_cell(Path(tmp), "rocm", cell=cell, bench_runs=3)
                validate_production_dense_prefill_cell(paths.aggregate, paths.timing, cell, bench_runs=3)
                packet = getattr(paths, field)
                with packet.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    columns = reader.fieldnames
                    rows = list(reader)
                self.assertTrue(all(row["codebook"] == "4" and row["execution_codebook"] == "24"
                                    for row in rows))
                rows[0]["execution_codebook"] = "4"
                with packet.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaisesRegex(ValueError, "physical execution codebook disagrees"):
                    validate_production_dense_prefill_cell(paths.aggregate, paths.timing, cell, bench_runs=3)

    def test_both_backends_reject_direct_or_partial_captured_timing(self) -> None:
        """A valid byte oracle cannot certify the wrong execution/timing mode."""
        for backend in ("cuda", "rocm"):
            for count in (0, 1, DENSE_PREFILL_CAPTURED_OPERATIONS - 1):
                with self.subTest(backend=backend, count=count), tempfile.TemporaryDirectory() as tmp:
                    cell, paths = _write_cell(Path(tmp), backend, bench_runs=3)
                    with paths.timing.open(newline="", encoding="utf-8") as handle:
                        rows = list(csv.DictReader(handle))
                    rows[0]["captured_operations"] = str(count)
                    with paths.timing.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=(
                            CUDA_TIMING_COLUMNS if backend == "cuda" else ROCM_TIMING_COLUMNS))
                        writer.writeheader()
                        writer.writerows(rows)
                    with self.assertRaisesRegex(ValueError, "complete captured operation batch"):
                        validate_production_dense_prefill_cell(
                            paths.aggregate, paths.timing, cell, bench_runs=3)

    def test_rocm_auto_streaming_observation_is_distinct_from_forced_tiles(self) -> None:
        """Auto can select its existing barrier-free family outside the tournament tiles."""
        cells = production_dense_prefill_cells("rocm")
        for source_format, m_tile in (("IQ1_M", 8), ("IQ1_S", 16)):
            with self.subTest(m_tile=m_tile), tempfile.TemporaryDirectory() as tmp:
                cell = next(cell for cell in cells
                            if cell.source_format.label == source_format
                            and cell.shape.name == "Qwen36MoE_GDN_QKVProjection"
                            and cell.m == 64)
                cell, paths = _write_cell(Path(tmp), "rocm", cell=cell, bench_runs=3)
                with paths.aggregate.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                auto = next(row for row in rows if row["variant"] == "Auto")
                auto.update(observed_n_tile="256", observed_m_tile=str(m_tile),
                            observed_min_blocks="3", observed_unroll="0",
                            observed_full_tiles="0", observed_producer="native_streaming")
                with paths.aggregate.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=ROCM_AGGREGATE_COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)
                result = validate_production_dense_prefill_cell(
                    paths.aggregate, paths.timing, cell, bench_runs=3)
                self.assertEqual(result.sample_count, result.candidate_count * 3)

    def test_rocm_streaming_observation_rejects_invented_or_forced_launch(self) -> None:
        """A valid Auto family is not permission to loosen any candidate's identity."""
        fields = ("observed_n_tile", "observed_m_tile", "observed_min_blocks",
                  "observed_unroll", "observed_full_tiles")
        invalid = ((128, 8, 3, 0, 0), (256, 32, 3, 0, 0), (256, 8, 2, 0, 0),
                   (256, 8, 3, 4, 0), (256, 8, 3, 0, 1))
        cases = [("Auto", launch) for launch in invalid]
        cases.append(("N64/MT16/MB1", (256, 8, 3, 0, 0)))
        for candidate, launch in cases:
            with self.subTest(candidate=candidate, launch=launch), tempfile.TemporaryDirectory() as tmp:
                cell, paths = _write_cell(Path(tmp), "rocm", bench_runs=3)
                with paths.aggregate.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                target = next(row for row in rows if row["variant"] == candidate)
                target.update(zip(fields, map(str, launch)))
                target["observed_producer"] = "native_streaming"
                with paths.aggregate.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=ROCM_AGGREGATE_COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaisesRegex(ValueError, "invalid observed ROCm launch"):
                    validate_production_dense_prefill_cell(
                        paths.aggregate, paths.timing, cell, bench_runs=3)

    def test_rocm_q8_aliases_authenticate_their_int8_producers(self) -> None:
        """Each Q8 source measures actual V3/V7 controls, never ignored low-bit knobs."""
        cells = production_dense_prefill_cells("rocm")
        for source in ("Q8_0", "Q8_1", "Q8_K"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as tmp:
                cell = next(cell for cell in cells if cell.source_format.label == source)
                cell, paths = _write_cell(Path(tmp), "rocm", cell=cell)
                candidates = dense_prefill_candidate_ids("rocm", cell.source_format)
                self.assertEqual(len(candidates), 25)
                self.assertTrue(all(c == "Auto" or c.startswith(("INT8V3/", "INT8V7/"))
                                    for c in candidates))
                result = validate_production_dense_prefill_cell(
                    paths.aggregate, paths.timing, cell, bench_runs=3)
                self.assertEqual(result.sample_count, 75)

    def test_rocm_winner_uses_raw_ticks_and_native_tie_order(self) -> None:
        """Rounded display values and lexical names never override native timing."""
        first, later = "N64/MT16/MB1", "N128/MT16/MB1"
        for exact_tie in (True, False):
            with self.subTest(exact_tie=exact_tie), tempfile.TemporaryDirectory() as tmp:
                cell, paths = _write_cell(Path(tmp), "rocm")
                samples = {
                    first: (10.0002, 10.5002, 10.8002),
                    later: (10.0002 if exact_tie else 10.0001, 10.5002, 10.8002),
                }
                winner = first if exact_tie else later
                with paths.aggregate.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    row["is_best"] = str(int(row["variant"] == winner))
                    if row["variant"] in samples:
                        values = samples[row["variant"]]
                        mean = statistics.fmean(values)
                        row.update(min_us=f"{min(values):.3f}", mean_us=f"{mean:.3f}",
                                   stddev_us=f"{math.sqrt(statistics.fmean((v-mean)**2 for v in values)):.3f}")
                with paths.aggregate.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=ROCM_AGGREGATE_COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)
                with paths.timing.open(newline="", encoding="utf-8") as handle:
                    timing_rows = list(csv.DictReader(handle))
                for row in timing_rows:
                    if row["variant"] in samples:
                        value = samples[row["variant"]][int(row["sample_index"])]
                        row.update(latency_us=f"{value:.9f}", latency_us_hex=value.hex())
                with paths.timing.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=ROCM_TIMING_COLUMNS)
                    writer.writeheader()
                    writer.writerows(timing_rows)
                result = validate_production_dense_prefill_cell(
                    paths.aggregate, paths.timing, cell, bench_runs=3)
                self.assertEqual(result.winner_id, winner)
                self.assertEqual(result.winner_min_us, min(samples[winner]))
                for row in rows:
                    row["is_best"] = str(int(row["variant"] == (later if exact_tie else first)))
                with paths.aggregate.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=ROCM_AGGREGATE_COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaisesRegex(ValueError, "winner marker disagrees"):
                    validate_production_dense_prefill_cell(
                        paths.aggregate, paths.timing, cell, bench_runs=3)

    def test_rocm_producer_identity_rejects_cross_family_and_stale_evidence(self) -> None:
        """A plausible tile/resource tuple cannot authenticate another producer."""
        cells = production_dense_prefill_cells("rocm")
        cases = (
            ("Q8_0", "Auto", {"observed_producer": "native_cooperative"}),
            ("Q8_1", "INT8V3/MT16/U0", {"observed_producer": "int8_blockwise_v7"}),
            ("Q8_K", "INT8V7/MT32/U1", {"observed_unroll": "2"}),
            ("Q8_0", "Auto", {"dynamic_shared_memory_bytes": "-1"}),
            ("Q6_K", "Auto", {"observed_producer": "int8_blockwise_v7"}),
            ("Q4_0", "Auto", {"observed_producer": ""}),
        )
        for source, candidate, changed in cases:
            with self.subTest(source=source, candidate=candidate), tempfile.TemporaryDirectory() as tmp:
                cell = next(cell for cell in cells if cell.source_format.label == source)
                cell, paths = _write_cell(Path(tmp), "rocm", cell=cell)
                with paths.aggregate.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                next(row for row in rows if row["variant"] == candidate).update(changed)
                with paths.aggregate.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=ROCM_AGGREGATE_COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    validate_production_dense_prefill_cell(
                        paths.aggregate, paths.timing, cell, bench_runs=3)

    def test_cuda_staging_is_distinct_in_aggregate_and_every_timing_sample(self) -> None:
        """A faster copy schedule cannot inherit the register schedule's receipt."""
        with tempfile.TemporaryDirectory() as tmp:
            cell, paths = _write_cell(Path(tmp), "cuda")
            aggregate = _read_aggregate_rows(paths.aggregate, cell)
            samples = _read_timing_rows(paths.timing, cell, 3)
            candidates = ("STD:t0:full", *(f"STD:t0:full:stage{s}" for s in (1, 2, 3)))
            self.assertEqual(len({samples[c] for c in candidates}), 4)
            for staging, candidate in enumerate(candidates):
                self.assertEqual(int(aggregate[candidate]["observed_staging_schedule"]), staging)
                self.assertEqual(_cuda_candidate_id(aggregate[candidate], "test"), candidate)

    def test_cuda_staging_parser_rejects_unknown_and_non_bk64_schedules(self) -> None:
        for strategy, invalid, message in (
            ("STD:t0:full", -1, "invalid CUDA staging"),
            ("STD:t0:full", 4, "invalid CUDA staging"),
            ("KPART:t0", 255, "invalid CUDA staging"),
            ("AUTO", 1, "malformed AUTO"),
            ("BK256:full", 3, "malformed BK256"),
        ):
            with self.subTest(strategy=strategy, staging=invalid):
                row = {key: str(value) for key, value in _cuda_launch(strategy).items()}
                row["staging_schedule"] = str(invalid)
                with self.assertRaisesRegex(ValueError, message):
                    _cuda_candidate_id(row, "test")

    def test_cuda_staging_inventory_rejects_unsupported_and_spilling_symbols(self) -> None:
        """All formats get their physical copy-width and exact-resource gate."""
        from native_quantised_dispatch.format_registry import GPU_EXECUTION_CODEBOOK_PAYLOAD_BYTES

        for codebook, payload_bytes in GPU_EXECUTION_CODEBOOK_PAYLOAD_BYTES.items():
            for tile in range(6):
                for staging in (1, 2, 3):
                    for canonical in (False, True):
                        candidate = f"KPART:t{tile}" if canonical else f"STD:t{tile}:full"
                        candidate += f":stage{staging}"
                        expected = payload_bytes in (16, 20) and tile != 1
                        if codebook in (5, 7) and tile == 2 and staging in (2, 3) and not canonical:
                            expected = False
                        with self.subTest(codebook=codebook, candidate=candidate):
                            self.assertEqual(_cuda_candidate_is_spill_free(codebook, candidate), expected)

    def test_cuda_iq4_xs_staging_uses_physical_payload_identity(self) -> None:
        """Physical24 retains all staged launches even though its source ID is4."""
        cell = next(cell for cell in production_dense_prefill_cells("cuda")
                    if cell.source_format.label == "IQ4_XS")
        candidates = dense_prefill_candidate_ids("cuda", cell.source_format)
        self.assertEqual(len(candidates), 43)
        self.assertEqual(sum(":stage" in candidate for candidate in candidates), 30)
        with tempfile.TemporaryDirectory() as tmp:
            cell, paths = _write_cell(Path(tmp), "cuda", cell=cell,
                                      winning_candidate="KPART:t2:stage3")
            validate_production_dense_prefill_cell(paths.aggregate, paths.timing, cell, bench_runs=3)

    def test_cuda_collection_clears_inherited_tile_and_peak_overrides(self) -> None:
        """A user's diagnostic environment cannot shrink or relabel a tournament."""
        cell = production_dense_prefill_cells("cuda")[0]
        paths = production_dense_prefill_cell_paths(Path("/tmp/unused-dense-cell"), cell)
        with mock.patch.dict(os.environ, {"LLAMINAR_TILE_SWEEP_TILES": "2",
                                        "LLAMINAR_CUDA_NATIVE_GEMM_PEAK_TC_TOPS": "1"}):
            environment = _cell_environment(cell, paths, 0, 10, 31)
        self.assertNotIn("LLAMINAR_TILE_SWEEP_TILES", environment)
        self.assertNotIn("LLAMINAR_CUDA_NATIVE_GEMM_PEAK_TC_TOPS", environment)

    def test_cuda_observed_staging_must_equal_the_forced_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cell, paths = _write_cell(Path(tmp), "cuda")
            with paths.aggregate.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            target = next(row for row in rows if row["staging_schedule"] == "2")
            target["observed_staging_schedule"] = "1"
            with paths.aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CUDA_AGGREGATE_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "forced CUDA staging did not execute"):
                validate_production_dense_prefill_cell(paths.aggregate, paths.timing, cell, bench_runs=3)

    def test_old_cuda_csv_without_staging_identity_is_not_resumable(self) -> None:
        for kind in ("aggregate", "timing"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                cell, paths = _write_cell(Path(tmp), "cuda")
                path = getattr(paths, kind)
                with path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    columns = tuple(name for name in reader.fieldnames if "staging" not in name)
                    rows = list(reader)
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaisesRegex(ValueError, "CSV header mismatch"):
                    validate_production_dense_prefill_cell(paths.aggregate, paths.timing, cell, bench_runs=3)

    def test_cuda_overlay_emits_typed_staging_without_changing_other_axes(self) -> None:
        for staging, name in enumerate(("RegisterDecode", "AsyncPayload", "AsyncWeightOperands", "AsyncAllOperands")):
            entry = DenseOverlayEntry(
                backend="cuda", key=DenseOverlayKey(execution_codebook=4, m=512, n=512, k=2048),
                source_formats=("IQ4_NL", "IQ4_XS"), shape_names=("35BMoE_Expert_GateUp",),
                candidate_id="KPART:t2" + (f":stage{staging}" if staging else ""),
                launch=(2, 4, 0, 1, staging), geometric_mean_regret=0.0,
                maximum_alias_regret=0.0, geometric_mean_speedup_vs_auto=1.1,
            )
            with self.subTest(staging=staging):
                rendered = render_dense_overlay((entry,))
                self.assertIn(f"{{2, false, true, prefill::PrefillStagingSchedule::{name}}}", rendered)
                self.assertIn('#include "CUDANativeQuantisedPrefillSchedule.h"', rendered)

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
                "native_quantised_dispatch.production_dense_prefill_sweep."
                "trainer_provenance",
                return_value=producer,
            ), mock.patch(
                "native_quantised_dispatch.production_dense_prefill_sweep._run_accelerator_cell"
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
                    "native_quantised_dispatch.production_dense_prefill_sweep."
                    "trainer_provenance",
                    return_value=observed,
                ), mock.patch(
                    "native_quantised_dispatch.production_dense_prefill_sweep."
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
                "native_quantised_dispatch.production_dense_prefill_sweep."
                "trainer_provenance",
                return_value=_test_provenance("cuda"),
            ), mock.patch(
                "native_quantised_dispatch.production_dense_prefill_sweep."
                "_run_accelerator_batch"
            ) as batch_launch, mock.patch(
                "native_quantised_dispatch.production_dense_prefill_sweep."
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
            environment["LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_FORMATS"],
            anchor.source_format.label,
        )
        self.assertEqual(
            environment["LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_SHAPES"],
            anchor.shape.name,
        )
        self.assertEqual(
            environment["LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_M"],
            ",".join(str(cell.m) for cell in same_weight),
        )
        self.assertEqual(
            environment["LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_MAX_CASES"],
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
                        environment["LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_FORMATS"],
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

    def test_additive_overlay_preserves_unmeasured_rows_and_selector_on_both_backends(self) -> None:
        """A focused refresh must neither erase old coverage nor rewrite its ABI."""

        for backend in ("cuda", "rocm"):
            with self.subTest(backend=backend):
                first = DenseOverlayEntry(
                    backend, DenseOverlayKey(4, 512, 5120, 17408),
                    ("IQ4_NL", "IQ4_XS"), ("Qwen36_FFN_DownProjection",),
                    "STD:t5:full" if backend == "cuda" else "N64/MT16/MB1",
                    (5, 1, 0, 0, 0) if backend == "cuda" else (64, 16, 1, 0, 0, 0),
                    0.0, 0.0, 1.0,
                )
                last = replace(first, key=replace(first.key, m=1024))
                base = render_dense_overlay((first, last))
                if backend == "cuda":
                    # Preserve the installed representation with an omitted,
                    # explicitly defaulted staging field; do not normalize it.
                    base = base.replace(", prefill::PrefillStagingSchedule::RegisterDecode", "")
                updated = replace(first, launch=(4, 1, 0, 0, 3) if backend == "cuda"
                                  else (128, 32, 1, 2, 1, 0))
                added = replace(updated, key=replace(first.key, m=4096))
                output, receipt = retain_dense_overlay_base(base, (updated, added))
                base_rows = [row for row in base.splitlines() if row.startswith("        {4,")]
                new_rows = [row for row in output.splitlines() if row.startswith("        {4,")]
                self.assertEqual(len(new_rows), 3)
                self.assertEqual(new_rows[1], base_rows[1])
                self.assertNotEqual(new_rows[0], base_rows[0])
                self.assertEqual(receipt["replaced_keys"], 1)
                self.assertEqual(receipt["retained_keys"], 1)
                self.assertEqual(receipt["refreshed_keys"], 2)
                self.assertEqual(receipt["output_keys"], 3)
                self.assertEqual(output.split("    inline int compare", 1)[1],
                                 base.split("    inline int compare", 1)[1])
                self.assertIn("not remeasured", receipt["base_evidence"])
                self.assertEqual(retain_dense_overlay_base(base, (updated, added)),
                                 (output, receipt))

    def test_additive_overlay_rejects_changed_abi_and_malformed_base(self) -> None:
        """Retention accepts exact generated syntax, never arbitrary C++ text."""

        first = DenseOverlayEntry(
            "cuda", DenseOverlayKey(4, 512, 5120, 17408), (), (),
            "STD:t5:full", (5, 1, 0, 0, 0), 0.0, 0.0, 1.0,
        )
        second = replace(first, key=replace(first.key, m=1024))
        base = render_dense_overlay((first, second))
        rows = [row for row in base.splitlines(keepends=True) if row.startswith("        {4,")]
        malformed = (
            base.replace("return true;", "return false;"),
            base.replace(rows[0], rows[0] + rows[0]),
            base.replace("".join(rows), "".join(reversed(rows))),
            base.replace("{4, 512,", "{4, 0,"),
            base.replace("{4, 512,", "{256, 512,"),
            base.replace("{4, 512,", "{4, 2147483648,"),
            base.replace("PrefillStagingSchedule::RegisterDecode}", "PrefillStagingSchedule::Unknown}"),
            base.replace(rows[0], "        /* extra executable text */\n" + rows[0]),
            base.replace("".join(rows), ""),
        )
        for candidate in malformed:
            with self.subTest(candidate=candidate[:80]), self.assertRaises(ValueError):
                retain_dense_overlay_base(candidate, (first,))
        with self.assertRaises(ValueError):
            retain_dense_overlay_base(base, (first, first))
        rocm = replace(first, backend="rocm", launch=(64, 16, 1, 0, 0))
        with self.assertRaises(ValueError):
            retain_dense_overlay_base(base, (rocm,))

    def test_additive_overlay_publication_requires_a_distinct_receipt(self) -> None:
        """A failed retention request cannot overwrite the installed include."""

        entry = DenseOverlayEntry(
            "cuda", DenseOverlayKey(4, 512, 5120, 17408), (), (),
            "STD:t5:full", (5, 1, 0, 0, 0), 0.0, 0.0, 1.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.inc"
            base.write_text(render_dense_overlay((entry,)))
            before = base.read_bytes()
            receipt = root / "retention.json"
            for args in ({"base_include": base}, {"retention_receipt": receipt},
                         {"base_include": base, "retention_receipt": base}):
                with self.assertRaises(ValueError):
                    write_dense_overlay(base, (entry,), **args)
                self.assertEqual(base.read_bytes(), before)
            output = root / "candidate.inc"
            write_dense_overlay(output, (entry,), base_include=base,
                                retention_receipt=receipt)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(base.read_bytes(), before)
            self.assertEqual(json.loads(receipt.read_text())["replaced_keys"], 1)

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
        anchor = next(cell for cell in cells if cell.source_format.label == "Q4_1")
        aliases = tuple(
            cell for cell in cells
            if cell.source_format.label in {"Q4_1", "Q4_K"}
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
            self.assertEqual(entry.key.execution_codebook, 5)
            self.assertEqual(entry.source_formats, ("Q4_1", "Q4_K"))
            self.assertEqual(entry.candidate_id, "AUTO")
            self.assertEqual(entry.launch, (2, 1, 0, 0, 0))
            rendered = render_dense_overlay(entries)
            self.assertIn("selectCUDADensePrefillOverlay", rendered)
            self.assertNotIn("AUTO", rendered)

    def test_iq4nl_and_iq4xs_retain_distinct_physical_overlay_keys(self) -> None:
        """Shared source4 policy does not pool distinct factor programs4/24."""
        cells = production_dense_prefill_cells("cuda")
        anchor = next(cell for cell in cells if cell.source_format.label == "IQ4_NL")
        selected = tuple(
            cell for cell in cells
            if cell.source_format.label in {"IQ4_NL", "IQ4_XS"}
            and cell.shape.name == anchor.shape.name and cell.m == anchor.m
        )
        self.assertEqual(len(selected), 2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cell in selected:
                _write_cell(root, "cuda", bench_runs=3, cell=cell)
            # Different physical programs may independently own different tile
            # observations. True aliases above still reject this disagreement.
            xs = next(cell for cell in selected if cell.source_format.label == "IQ4_XS")
            paths = production_dense_prefill_cell_paths(root, xs)
            with paths.aggregate.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["observed_tile_id"] = "3"
            with paths.aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CUDA_AGGREGATE_COLUMNS)
                writer.writeheader(); writer.writerows(rows)
            _commit_test_cells(root, "cuda", selected)
            entries = collect_dense_overlay_entries(
                root, "cuda", cells=selected, warmup_runs=2, bench_runs=3)
            self.assertEqual(len(entries), 2)
            by_codebook = {entry.key.execution_codebook: entry for entry in entries}
            self.assertEqual(set(by_codebook), {4, 24})
            self.assertEqual(by_codebook[4].source_formats, ("IQ4_NL",))
            self.assertEqual(by_codebook[24].source_formats, ("IQ4_XS",))
            self.assertEqual(by_codebook[4].launch, (2, 1, 0, 0, 0))
            self.assertEqual(by_codebook[24].launch, (3, 1, 0, 0, 0))

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
            launch=(128, 32, 2, 4, 1, 0),
            geometric_mean_regret=0.0,
            maximum_alias_regret=0.0,
            geometric_mean_speedup_vs_auto=1.1,
        )

        rendered = render_dense_overlay((entry,))

        self.assertIn("selectROCmDensePrefillOverlay", rendered)
        self.assertIn("{128, 32, 2, 4, true, VNNIPrefillProducer::NativeCooperative}", rendered)

    def test_rocm_overlay_retains_distinct_int8_and_streaming_producers(self) -> None:
        """Installation cannot reinterpret identical tile fields as another kernel."""
        for codebook, producer, family, launch in (
            (19, 2, "Int8BlockwiseV3", (64, 32, 1, 2, 0)),
            (19, 3, "Int8BlockwiseV7", (128, 64, 1, 2, 0)),
            (17, 1, "NativeStreaming", (256, 8, 3, 0, 0)),
        ):
            with self.subTest(family=family):
                entry = DenseOverlayEntry("rocm", DenseOverlayKey(codebook, 512, 4096, 2048,
                    DensePreparedWeightFamily.INT8_BLOCKWISE if producer in (2, 3)
                    else DensePreparedWeightFamily.NATIVE_QUANTISED),
                    (), (), "Auto", (*launch, producer), 0.0, 0.0, 1.0)
                rendered = render_dense_overlay((entry,))
                self.assertIn(f"VNNIPrefillProducer::{family}", rendered)
                self.assertEqual(retain_dense_overlay_base(rendered, (entry,))[0], rendered)
                with self.assertRaisesRegex(ValueError, "producer identity"):
                    render_dense_overlay((replace(entry, launch=launch),))

    def test_rocm_same_geometry_retention_separates_prepared_families(self) -> None:
        """One family's refresh cannot erase or consume the other's exact row."""
        native = DenseOverlayEntry(
            "rocm", DenseOverlayKey(19, 64, 512, 5120, architecture=GPUArchitecture.parse("rocm", "gfx906")), ("Q8_0", "Q8_1", "Q8_K"), (),
            "native_fixture", (64, 16, 1, 4, 0, 0), 0.0, 0.0, 1.0)
        expanded = replace(native, key=replace(native.key,
            prepared_family=DensePreparedWeightFamily.INT8_BLOCKWISE, architecture=None),
            candidate_id="int8_fixture", launch=(64, 16, 1, 4, 0, 2))
        base = render_dense_overlay((native, expanded))
        self.assertIn("5120, DensePrefillWeightFamily::NativeQuantised", base)
        self.assertIn("5120, DensePrefillWeightFamily::Int8Blockwise", base)
        changed = replace(native, launch=(128, 32, 1, 2, 0, 0))
        output, receipt = retain_dense_overlay_base(base, (changed,))
        int8_row = next(row for row in base.splitlines()
                        if row.startswith("        {19,") and "Int8Blockwise," in row)
        self.assertIn(int8_row, output)
        self.assertEqual(receipt["replaced_keys"], 1)
        self.assertEqual(receipt["retained_keys"], 1)
        self.assertEqual(receipt["output_keys"], 2)
        self.assertEqual(receipt["key_schema"], "codebook-m-n-k-prepared-family-native-architecture")
        self.assertEqual(retain_dense_overlay_base(output, (changed,))[0], output)
        for malformed in (replace(native, launch=expanded.launch),
                          replace(expanded, launch=native.launch)):
            with self.assertRaisesRegex(ValueError, "prepared family"):
                render_dense_overlay((malformed,))
        with self.assertRaisesRegex(ValueError, "typed prepared family"):
            replace(native.key, prepared_family=255)
        with self.assertRaisesRegex(ValueError, "duplicate dense overlay key"):
            render_dense_overlay((native, native))
        with self.assertRaisesRegex(ValueError, "retained producer"):
            retain_dense_overlay_base(base.replace(
                "VNNIPrefillProducer::NativeCooperative", "VNNIPrefillProducer::Int8BlockwiseV3"),
                (changed,))
        # The old untyped program is not an admissible additive base. Migration
        # must bind its original producer values explicitly before retention.
        with self.assertRaisesRegex(ValueError, "selector ABI mismatch"):
            retain_dense_overlay_base(base.replace(
                '#include "kernels/rocm/gemm/ROCmDenseProductionPrefillOverlayTypes.h"\n', ""), (changed,))

    def test_installed_rocm_family_key_abi_is_retained_without_reinterpretation(self) -> None:
        """The installed program must match the generator's complete typed ABI."""
        path = REPO_ROOT / "src/v2/kernels/rocm/gemm/ROCmDenseProductionPrefillOverlayGenerated.inc"
        base = path.read_text(encoding="utf-8")
        expanded = DenseOverlayEntry("rocm", DenseOverlayKey(19, 64, 512, 5120,
            DensePreparedWeightFamily.INT8_BLOCKWISE), (), (), "installed_identity",
            (64, 16, 1, 4, 0, 2), 0.0, 0.0, 1.0)
        output, receipt = retain_dense_overlay_base(base, (expanded,))
        for line in base.splitlines(keepends=True):
            if line.startswith("        {"):
                self.assertIn(line, output)
        self.assertEqual(receipt["replaced_keys"], 1)
        # Adding an uninstalled native row is a metadata fixture only. The base
        # INT8 row keeps exactly its original bytes and producer contract.
        native = replace(expanded, key=replace(expanded.key,
            prepared_family=DensePreparedWeightFamily.NATIVE_QUANTISED, architecture=GPUArchitecture.parse("rocm", "gfx906")),
            source_formats=("Q8_0", "Q8_1", "Q8_K"), launch=(64, 16, 1, 4, 0, 0))
        output, receipt = retain_dense_overlay_base(base, (native,))
        self.assertEqual(receipt["replaced_keys"], 0)
        self.assertEqual(receipt["output_keys"], receipt["base_keys"] + 1)
        for row in base.splitlines():
            if row.startswith("        {"):
                self.assertIn(row + "\n", output)

    def test_rocm_totality_requires_canonical_prepared_family(self) -> None:
        """A same-geometry native row cannot certify the current INT8 trainer."""
        cell = next(cell for cell in production_dense_prefill_cells("rocm")
                    if cell.source_format.runtime_codebook("rocm") == 19)
        native = DenseOverlayEntry("rocm", DenseOverlayKey(19, cell.m,
            cell.shape.n, cell.shape.k, architecture=GPUArchitecture.parse("rocm", "gfx906")), ("Q8_0", "Q8_1", "Q8_K"),
            (cell.shape.name,), "native_fixture", (64, 16, 1, 0, 0, 0),
            0.0, 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "not total"):
            validate_dense_overlay_totality((native,), (cell,), "rocm")
        expanded = replace(native, key=replace(native.key,
            prepared_family=DensePreparedWeightFamily.INT8_BLOCKWISE, architecture=None),
            source_formats=(cell.source_format.label,), launch=(64, 16, 1, 0, 0, 2))
        validate_dense_overlay_totality((expanded,), (cell,), "rocm")

    def test_overlay_rejects_aliases_with_conflicting_physical_auto_launches(self) -> None:
        cells = production_dense_prefill_cells("cuda")
        anchor = next(cell for cell in cells if cell.source_format.label == "Q4_1")
        aliases = tuple(
            cell for cell in cells
            if cell.source_format.label in {"Q4_1", "Q4_K"}
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
                launch=(tile_id, 1, 0, 0, 0),
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


def load_tests(loader, tests, pattern):
    """Partition CTest's complete module deterministically; direct runs stay whole.

    Match the existing profiler-evidence shard protocol. Sorting complete test
    IDs gives disjoint, exhaustive membership independent of suite nesting.
    Explicit named integration selectors do not invoke this module discovery.
    """
    del loader, pattern
    index_text = os.environ.get("LLAMINAR_UNIT_SHARD_INDEX")
    count_text = os.environ.get("LLAMINAR_UNIT_SHARD_COUNT")
    if index_text is None and count_text is None:
        return tests
    if index_text is None or count_text is None:
        raise ValueError("both unit-test shard values must be set together")
    index, count = int(index_text), int(count_text)
    if count <= 0 or not 0 <= index < count:
        raise ValueError("unit-test shard index must be in [0, positive count)")

    def leaves(suite):
        """Yield each original test exactly once, without running it."""
        for test in suite:
            if isinstance(test, unittest.TestSuite):
                yield from leaves(test)
            else:
                yield test

    ordered = sorted(leaves(tests), key=lambda test: test.id())
    selected = ordered[index::count]
    if not selected:
        raise ValueError("unit-test shard must not be empty")
    return unittest.TestSuite(selected)


if __name__ == "__main__":
    unittest.main()
