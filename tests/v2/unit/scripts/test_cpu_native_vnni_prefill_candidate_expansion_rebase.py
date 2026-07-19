#!/usr/bin/env python3
"""Regression tests for durable CPU candidate-expansion checkpoint rebasing."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.cpu_prefill_candidate_expansion import (  # noqa: E402
    build_cpu_prefill_candidate_expansion_plan,
    cpu_prefill_candidate_expansion_plan_token,
    write_cpu_prefill_candidate_expansion_plan,
)
from native_vnni_dispatch.rebase_cpu_prefill_candidate_expansion_checkpoint import (  # noqa: E402
    REBASE_WORKERS_ENVIRONMENT,
    cpu_prefill_candidate_expansion_rebase_worker_count,
    rebase_cpu_prefill_candidate_expansion_checkpoint,
)


SOURCE_CANDIDATES = (
    "cpu.nvnni.prefill.row_chunk_grid.full_k",
    "cpu.nvnni.prefill.two_row_tiles.nbc1.full_k",
    "cpu.nvnni.prefill.two_row_tiles.nbc2.full_k",
    "cpu.nvnni.prefill.two_row_tiles.nbc4.full_k",
    "cpu.nvnni.prefill.two_row_tiles.nbc8.full_k",
    "cpu.nvnni.prefill.two_row_tiles.nbc16.full_k",
    "cpu.nvnni.prefill.decode_equivalent_kpart.pairwise",
    "cpu.nvnni.prefill.decode_equivalent_kpart.wide_rows",
)
UNIT_BUILD_DIGEST = "sha256:" + "b" * 64


class CPUNativeVNNIPrefillCheckpointRebaseTest(unittest.TestCase):
    """Prove migration preserves bytes and refuses changed experiments."""

    @staticmethod
    def _write_source(path: Path) -> None:
        """Write the smallest complete three-regime source plan fixture."""

        fieldnames = (
            "backend",
            "phase",
            "source_format",
            "source_codebook",
            "execution_codebook",
            "shape",
            "execution_mode",
            "m",
            "n",
            "k",
            "candidate_id",
            "build_isa",
            "runtime_isa_requested",
            "runtime_isa_effective",
            "threads",
        )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for build_isa, runtime_isa in (
                ("AVX2", "AVX2"),
                ("AVX512", "AVX2"),
                ("AVX512", "AVX512"),
            ):
                for candidate in SOURCE_CANDIDATES:
                    writer.writerow({
                        "backend": "cpu",
                        "phase": "prefill_gemm",
                        "source_format": "Q4_0",
                        "source_codebook": 2,
                        "execution_codebook": 2,
                        "shape": "unit-shape",
                        "execution_mode": "eager",
                        "m": 2,
                        "n": 1024,
                        "k": 2048,
                        "candidate_id": candidate,
                        "build_isa": build_isa,
                        "runtime_isa_requested": runtime_isa,
                        "runtime_isa_effective": runtime_isa,
                        "threads": 28,
                    })

    @staticmethod
    def _partial_paths(
        directory: Path,
        token: str,
        source_format: str,
        shape: str,
        isa_regime: str,
    ) -> tuple[Path, Path]:
        stem = (
            f"cpu_prefill.candidate-expansion.{token}."
            f"{source_format}.{shape}.{isa_regime}"
        )
        aggregate = directory / f"{stem}.csv"
        return aggregate, directory / f"{stem}.timing.csv"

    @staticmethod
    def _write_partial(path: Path, plan, record) -> None:
        """Write one exact candidate-by-M matrix for transaction testing."""

        fieldnames = (
            "backend",
            "phase",
            "source_format",
            "shape",
            "m",
            "n",
            "k",
            "candidate_id",
            "build_isa",
            "runtime_isa_requested",
            "runtime_isa_effective",
            "threads",
        )
        build_isa, runtime_isa = {
            "avx2-build.avx2-runtime": ("AVX2", "AVX2"),
            "avx512-build.avx2-runtime": ("AVX512", "AVX2"),
            "avx512-build.avx512-runtime": ("AVX512", "AVX512"),
        }[record.isa_regime]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for m in record.m_values:
                for candidate in plan.collection_candidate_ids:
                    writer.writerow({
                        "backend": "cpu",
                        "phase": "prefill_gemm",
                        "source_format": record.source_format,
                        "shape": record.shape_name,
                        "m": m,
                        "n": record.n,
                        "k": record.k,
                        "candidate_id": candidate,
                        "build_isa": build_isa,
                        "runtime_isa_requested": runtime_isa,
                        "runtime_isa_effective": runtime_isa,
                        "threads": record.threads,
                    })

    def _plans(self, root: Path):
        source_csv = root / "immutable.csv"
        source_timing = root / "immutable.timing.csv"
        self._write_source(source_csv)
        source_timing.write_text("retained timing source\n", encoding="utf-8")
        current = build_cpu_prefill_candidate_expansion_plan(
            source_csv,
            source_timing,
            collection_build_digest=UNIT_BUILD_DIGEST,
        )
        historical = replace(
            current,
            implementation_digest="sha256:" + "e" * 64,
        )
        return historical, current

    def test_production_rebase_uses_a_bounded_process_width(self) -> None:
        self.assertGreater(
            cpu_prefill_candidate_expansion_rebase_worker_count(1855, {}),
            1,
        )
        self.assertLessEqual(
            cpu_prefill_candidate_expansion_rebase_worker_count(1855, {}),
            32,
        )
        self.assertEqual(
            cpu_prefill_candidate_expansion_rebase_worker_count(
                1855,
                {REBASE_WORKERS_ENVIRONMENT: "7"},
            ),
            7,
        )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            cpu_prefill_candidate_expansion_rebase_worker_count(
                1855,
                {REBASE_WORKERS_ENVIRONMENT: "0"},
            )

    def test_rebase_copies_only_complete_validated_pairs_without_rewriting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_directory = root / "old"
            target_directory = root / "new"
            source_directory.mkdir()
            target_directory.mkdir()
            historical, current = self._plans(root)
            source_plan = source_directory / "cpu_prefill_candidate_expansion.v4.json"
            target_plan = target_directory / "cpu_prefill_candidate_expansion.v4.json"
            write_cpu_prefill_candidate_expansion_plan(source_plan, historical)
            write_cpu_prefill_candidate_expansion_plan(target_plan, current)

            record = historical.records[0]
            source_token = cpu_prefill_candidate_expansion_plan_token(source_plan)
            source_aggregate, source_timing = self._partial_paths(
                source_directory,
                source_token,
                record.source_format,
                record.shape_name,
                record.isa_regime,
            )
            self._write_partial(source_aggregate, historical, record)
            source_timing.write_bytes(b"raw timing sidecar remains immutable\n")
            source_aggregate_bytes = source_aggregate.read_bytes()
            source_timing_bytes = source_timing.read_bytes()
            validated: list[tuple[Path, Path, bool]] = []

            def accept(aggregate: Path, timing: Path, *, candidate_expansion: bool) -> None:
                validated.append((aggregate, timing, candidate_expansion))

            manifest_path = rebase_cpu_prefill_candidate_expansion_checkpoint(
                source_directory,
                target_directory,
                target_plan,
                validator=accept,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            target_token = cpu_prefill_candidate_expansion_plan_token(target_plan)
            target_aggregate, target_timing = self._partial_paths(
                target_directory,
                target_token,
                record.source_format,
                record.shape_name,
                record.isa_regime,
            )

            self.assertEqual(manifest["published_count"], 1)
            self.assertEqual(manifest["recollect_count"], 2)
            self.assertEqual(target_aggregate.read_bytes(), source_aggregate_bytes)
            self.assertEqual(target_timing.read_bytes(), source_timing_bytes)
            self.assertEqual(source_aggregate.read_bytes(), source_aggregate_bytes)
            self.assertEqual(source_timing.read_bytes(), source_timing_bytes)
            self.assertEqual(len(validated), 2)
            self.assertTrue(all(call[2] for call in validated))

            second_manifest = rebase_cpu_prefill_candidate_expansion_checkpoint(
                source_directory,
                target_directory,
                target_plan,
                validator=accept,
            )
            second = json.loads(second_manifest.read_text(encoding="utf-8"))
            self.assertEqual(second["published"][0]["status"], "already_published")

    def test_rebase_rejects_changed_measurement_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_directory = root / "old"
            target_directory = root / "new"
            source_directory.mkdir()
            target_directory.mkdir()
            historical, current = self._plans(root)
            changed_record = replace(current.records[0], threads=56)
            incompatible = replace(
                current,
                records=(changed_record, *current.records[1:]),
            )
            source_plan = source_directory / "cpu_prefill_candidate_expansion.v4.json"
            target_plan = target_directory / "cpu_prefill_candidate_expansion.v4.json"
            write_cpu_prefill_candidate_expansion_plan(source_plan, historical)
            write_cpu_prefill_candidate_expansion_plan(target_plan, incompatible)

            with self.assertRaisesRegex(
                ValueError,
                "incompatible selected-cell inventories",
            ):
                rebase_cpu_prefill_candidate_expansion_checkpoint(
                    source_directory,
                    target_directory,
                    target_plan,
                    validator=lambda *args, **kwargs: None,
                )

    def test_harness_build_change_requires_an_explicit_recorded_audit(
        self,
    ) -> None:
        """Timing-harness rebuilds are reusable only through a visible review."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_directory = root / "old"
            target_directory = root / "new"
            source_directory.mkdir()
            target_directory.mkdir()
            historical, current = self._plans(root)
            rebuilt = replace(
                current,
                collection_build_digest="sha256:" + "c" * 64,
            )
            source_plan = (
                source_directory / "cpu_prefill_candidate_expansion.v4.json"
            )
            target_plan = (
                target_directory / "cpu_prefill_candidate_expansion.v4.json"
            )
            write_cpu_prefill_candidate_expansion_plan(
                source_plan, historical
            )
            write_cpu_prefill_candidate_expansion_plan(target_plan, rebuilt)

            with self.assertRaisesRegex(
                ValueError,
                "different measurements: collection_build_digest",
            ):
                rebase_cpu_prefill_candidate_expansion_checkpoint(
                    source_directory,
                    target_directory,
                    target_plan,
                    validator=lambda *args, **kwargs: None,
                )

            audit = (
                "unit timing-harness policy changed; production kernels and "
                "candidate inventory were reviewed unchanged"
            )
            manifest_path = (
                rebase_cpu_prefill_candidate_expansion_checkpoint(
                    source_directory,
                    target_directory,
                    target_plan,
                    validator=lambda *args, **kwargs: None,
                    harness_only_build_change_audit=audit,
                )
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["build_compatibility"],
                "audited-harness-only-change-v1",
            )
            self.assertEqual(
                manifest["source_collection_build_digest"],
                UNIT_BUILD_DIGEST,
            )
            self.assertEqual(
                manifest["target_collection_build_digest"],
                rebuilt.collection_build_digest,
            )
            self.assertEqual(
                manifest["harness_only_build_change_audit"], audit
            )

    def test_rebase_extends_production_subset_into_reviewed_development_plan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_directory = root / "old"
            target_directory = root / "new"
            source_directory.mkdir()
            target_directory.mkdir()
            _, complete = self._plans(root)
            inherited_record = complete.records[0]
            historical = replace(
                complete,
                implementation_digest="sha256:" + "e" * 64,
                selection_policy="manifest-production-cells-v1",
                selected_cell_count=len(inherited_record.m_values),
                selected_cell_digest="sha256:" + "1" * 64,
                records=(inherited_record,),
            )
            reviewed = replace(
                complete,
                selection_policy="split-development-cells-v1",
            )
            source_plan = source_directory / "cpu_prefill_candidate_expansion.v4.json"
            target_plan = target_directory / "cpu_prefill_candidate_expansion.v4.json"
            write_cpu_prefill_candidate_expansion_plan(source_plan, historical)
            write_cpu_prefill_candidate_expansion_plan(target_plan, reviewed)
            source_token = cpu_prefill_candidate_expansion_plan_token(source_plan)
            source_aggregate, source_timing = self._partial_paths(
                source_directory,
                source_token,
                inherited_record.source_format,
                inherited_record.shape_name,
                inherited_record.isa_regime,
            )
            self._write_partial(source_aggregate, historical, inherited_record)
            source_timing.write_bytes(b"retained subset timing\n")

            manifest_path = rebase_cpu_prefill_candidate_expansion_checkpoint(
                source_directory,
                target_directory,
                target_plan,
                validator=lambda *args, **kwargs: None,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["published_count"], 1)
            self.assertEqual(manifest["recollect_count"], 2)
            self.assertEqual(
                manifest["selection_compatibility"],
                "manifest-production-cells-v1-to-"
                "split-development-cells-v1",
            )
            self.assertEqual(
                {item["reason"] for item in manifest["recollect"]},
                {"target_record_not_in_source_plan"},
            )

    def test_rebase_restores_dynamic_refinement_cells_to_full_source_plan(
        self,
    ) -> None:
        """A static split projection must not erase measured refinement cells."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_directory = root / "reviewed"
            target_directory = root / "full"
            source_directory.mkdir()
            target_directory.mkdir()
            _, complete = self._plans(root)
            inherited_record = complete.records[0]
            reviewed = replace(
                complete,
                implementation_digest="sha256:" + "e" * 64,
                selection_policy="split-development-cells-v1",
                selected_cell_count=len(inherited_record.m_values),
                selected_cell_digest="sha256:" + "1" * 64,
                records=(inherited_record,),
            )
            full_source = replace(
                complete,
                selection_policy="all-immutable-source-cells-v1",
            )
            source_plan = (
                source_directory / "cpu_prefill_candidate_expansion.v4.json"
            )
            target_plan = (
                target_directory / "cpu_prefill_candidate_expansion.v4.json"
            )
            write_cpu_prefill_candidate_expansion_plan(source_plan, reviewed)
            write_cpu_prefill_candidate_expansion_plan(target_plan, full_source)
            source_token = cpu_prefill_candidate_expansion_plan_token(source_plan)
            source_aggregate, source_timing = self._partial_paths(
                source_directory,
                source_token,
                inherited_record.source_format,
                inherited_record.shape_name,
                inherited_record.isa_regime,
            )
            self._write_partial(source_aggregate, reviewed, inherited_record)
            source_timing.write_bytes(b"retained reviewed timing\n")

            manifest_path = rebase_cpu_prefill_candidate_expansion_checkpoint(
                source_directory,
                target_directory,
                target_plan,
                validator=lambda *args, **kwargs: None,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["published_count"], 1)
            self.assertEqual(manifest["recollect_count"], 2)
            self.assertEqual(
                manifest["selection_compatibility"],
                "split-development-cells-v1-to-"
                "all-immutable-source-cells-v1",
            )

    def test_rebase_preserves_completed_shard_beside_failed_inprogress_pair(
        self,
    ) -> None:
        """A noisy launch cannot invalidate evidence published before it failed."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_directory = root / "old"
            target_directory = root / "new"
            source_directory.mkdir()
            target_directory.mkdir()
            historical, current = self._plans(root)
            source_plan = source_directory / "cpu_prefill_candidate_expansion.v4.json"
            target_plan = target_directory / "cpu_prefill_candidate_expansion.v4.json"
            write_cpu_prefill_candidate_expansion_plan(source_plan, historical)
            write_cpu_prefill_candidate_expansion_plan(target_plan, current)

            completed = historical.records[0]
            failed = historical.records[1]
            source_token = cpu_prefill_candidate_expansion_plan_token(source_plan)
            completed_aggregate, completed_timing = self._partial_paths(
                source_directory,
                source_token,
                completed.source_format,
                completed.shape_name,
                completed.isa_regime,
            )
            self._write_partial(completed_aggregate, historical, completed)
            completed_timing.write_bytes(b"completed timing evidence\n")
            completed_bytes = (
                completed_aggregate.read_bytes(),
                completed_timing.read_bytes(),
            )

            failed_aggregate, failed_timing = self._partial_paths(
                source_directory,
                source_token,
                failed.source_format,
                failed.shape_name,
                failed.isa_regime,
            )
            failed_aggregate_inprogress = failed_aggregate.with_suffix(
                f"{failed_aggregate.suffix}.inprogress"
            )
            failed_timing_inprogress = failed_timing.with_suffix(
                f"{failed_timing.suffix}.inprogress"
            )
            failed_aggregate_inprogress.write_bytes(b"unstable aggregate\n")
            failed_timing_inprogress.write_bytes(b"unstable timing evidence\n")

            manifest_path = rebase_cpu_prefill_candidate_expansion_checkpoint(
                source_directory,
                target_directory,
                target_plan,
                validator=lambda *args, **kwargs: None,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            target_token = cpu_prefill_candidate_expansion_plan_token(target_plan)
            target_aggregate, target_timing = self._partial_paths(
                target_directory,
                target_token,
                completed.source_format,
                completed.shape_name,
                completed.isa_regime,
            )

            self.assertEqual(manifest["published_count"], 1)
            self.assertEqual(manifest["recollect_count"], 2)
            self.assertEqual(target_aggregate.read_bytes(), completed_bytes[0])
            self.assertEqual(target_timing.read_bytes(), completed_bytes[1])
            self.assertEqual(
                failed_aggregate_inprogress.read_bytes(), b"unstable aggregate\n"
            )
            self.assertEqual(
                failed_timing_inprogress.read_bytes(), b"unstable timing evidence\n"
            )
            self.assertEqual(
                {item["reason"] for item in manifest["recollect"]},
                {"source_pair_incomplete"},
            )


if __name__ == "__main__":
    unittest.main()
