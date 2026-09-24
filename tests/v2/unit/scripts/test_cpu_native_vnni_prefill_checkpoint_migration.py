#!/usr/bin/env python3
"""Regression tests for atomic CPU prefill checkpoint protocol upgrades."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.adapters.evidence import (  # noqa: E402
    summarize_sorted_timing,
)
from native_vnni_dispatch.migrate_cpu_prefill_checkpoint import (  # noqa: E402
    MIGRATION_MANIFEST_FILE,
    TARGET_CONTRACT_FILE,
    migrate_cpu_prefill_checkpoint,
)
from native_vnni_dispatch.profiles import (  # noqa: E402
    LEGACY_ADAPTIVE_TIMING_CEILING_POLICY,
    LEGACY_ADAPTIVE_TIMING_PROTOCOL,
)


class CPUPrefillCheckpointMigrationTest(unittest.TestCase):
    """Prove that only reusable finalized v12 evidence can cross contracts."""

    @staticmethod
    def diagnostic_row() -> dict[str, str]:
        """Build one valid v12 one-sample diagnostic aggregate row."""

        timing = summarize_sorted_timing([10.0])
        return {
            "backend": "cpu",
            "phase": "prefill_gemm",
            "source_format": "Q4_K",
            "source_codebook": "5",
            "execution_codebook": "5",
            "shape": "0.5B_AttnOut",
            "execution_mode": "eager",
            "m": "64",
            "n": "896",
            "k": "896",
            "candidate_id": "cpu.nvnni.prefill.row_chunk_grid.full_k",
            "build_isa": "AVX512",
            "runtime_isa_requested": "AVX512",
            "runtime_isa_effective": "AVX512",
            "threads": "28",
            "measurement_coordination": "process-local-complete-round-v1",
            "mpi_world_size": "1",
            "mpi_rank": "0",
            "weight_bytes": "501760",
            "serial_oracle_policy": "boundary-quartile-sentinel-v1",
            "serial_oracle_rows": "16",
            "preconditioning_policy": "disabled",
            "preconditioning_m": "0",
            "preconditioning_budget_us": "0",
            "preconditioning_duration_us": "0",
            "warmup_count": "0",
            "warmup_round_count": "0",
            "warmup_budget_policy": "disabled",
            "warmup_probe_latency_us": "10",
            "warmup_budget_floor_us": "0",
            "warmup_latency_multiplier": "0",
            "warmup_budget_ceiling_us": "0",
            "warmup_budget_us": "0",
            "warmup_duration_us": "0",
            "global_warmup_round_count": "0",
            "warmup_wall_duration_us": "0",
            "warmup_max_round_duration_us": "0",
            "warmup_round_timeout_us": "30000000",
            "timing_order_seed": "12345",
            "global_timing_round_count": "1",
            "sample_count": "1",
            "timing_protocol": "elapsed-stability-interleaved-v12",
            "min_sample_count": "5",
            "stable_sample_count": "30",
            "max_sample_count": "180",
            "timing_budget_us": "2000000",
            "timed_duration_us": "10",
            "median_stability_limit": "0.02",
            "median_relative_drift": "inf",
            "timing_converged": "0",
            "timing_stop_reason": "hard_max_samples",
            "min_us": "10",
            "median_us": "10",
            "p95_us": "10",
            "mad_us": "0",
            "cv": "0",
            "serial_median_us": "12",
            "speedup": "1.2",
            "bit_mismatches": "0",
            "first_bit_mismatch": "0",
            "repeat_byte_mismatches": "0",
            "max_abs": "0",
            "relative_l2": "0",
            "cosine": "1",
            "symmetric_kld": "0",
            "grouped_output_digest": "fnv1a64:equal",
            "serial_output_digest": "fnv1a64:equal",
            "timing_sample_digest": timing.digest,
            "route_counter_ok": "0",
            "observed_candidate_id": (
                "cpu.nvnni.prefill.decode_equivalent_kpart.pairwise"
            ),
            "k_tiles": "1",
            "k_tile_blocks": "28",
            "n_block_chunks": "4",
            "numerical_correctness": "1",
            "correctness_pass": "0",
            "is_winner": "0",
        }

    @classmethod
    def write_source(
        cls,
        root: Path,
        *,
        row: dict[str, str] | None = None,
    ) -> tuple[Path, Path]:
        """Write one paired aggregate/sidecar and an ignored failed shard."""

        root.mkdir()
        aggregate = root / "cpu_prefill.Q4_K.0.5B_AttnOut.avx512.csv"
        timing = root / "cpu_prefill.Q4_K.0.5B_AttnOut.avx512.timing.csv"
        source_row = row or cls.diagnostic_row()
        with aggregate.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(source_row))
            writer.writeheader()
            writer.writerow(source_row)
        timing_row = {
            key: source_row[key]
            for key in (
                "backend", "phase", "source_format", "source_codebook",
                "execution_codebook", "shape", "execution_mode", "m", "n", "k",
                "candidate_id", "build_isa", "runtime_isa_requested",
                "runtime_isa_effective",
            )
        }
        timing_row.update({
            "sample_index": "0",
            "latency_us": "10.000000000",
            "latency_us_hex": float(10.0).hex(),
        })
        with timing.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(timing_row))
            writer.writeheader()
            writer.writerow(timing_row)
        (root / TARGET_CONTRACT_FILE).write_text("a" * 64 + "\n")
        (root / f"{aggregate.name}.inprogress").write_text("failed\n")
        return aggregate, timing

    def test_migrates_reusable_rows_and_preserves_timing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            source = temp / "source"
            output = temp / "output"
            source_aggregate, source_timing = self.write_source(source)
            source_aggregate_bytes = source_aggregate.read_bytes()
            source_timing_bytes = source_timing.read_bytes()

            manifest = migrate_cpu_prefill_checkpoint(
                source,
                output,
                "b" * 64,
                target_inventory={source_aggregate.name: frozenset({64})},
            )

            with (output / source_aggregate.name).open(
                newline="", encoding="utf-8"
            ) as handle:
                migrated = next(csv.DictReader(handle))
            self.assertEqual(
                migrated["timing_protocol"], LEGACY_ADAPTIVE_TIMING_PROTOCOL
            )
            self.assertEqual(
                migrated["timing_ceiling_policy"],
                LEGACY_ADAPTIVE_TIMING_CEILING_POLICY,
            )
            self.assertEqual(migrated["timing_stop_reason"], "fixed_samples")
            self.assertNotIn(b"\r", (output / source_aggregate.name).read_bytes())
            self.assertEqual((output / source_timing.name).read_bytes(), source_timing_bytes)
            self.assertEqual(source_aggregate.read_bytes(), source_aggregate_bytes)
            self.assertEqual((output / TARGET_CONTRACT_FILE).read_text(), "b" * 64 + "\n")
            self.assertFalse((output / f"{source_aggregate.name}.inprogress").exists())
            self.assertEqual(manifest["rows"], 1)
            self.assertEqual(manifest["forceable_rows"], 0)
            self.assertEqual(manifest["diagnostic_rows"], 1)
            persisted = json.loads((output / MIGRATION_MANIFEST_FILE).read_text())
            self.assertEqual(persisted, manifest)

    def test_rejects_unstable_forceable_evidence_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            source = temp / "source"
            output = temp / "output"
            row = self.diagnostic_row()
            row.update({
                "route_counter_ok": "1",
                "observed_candidate_id": row["candidate_id"],
                "timing_stop_reason": "hard_max_samples",
            })
            aggregate, _ = self.write_source(source, row=row)

            with self.assertRaisesRegex(ValueError, "did not finalize stable"):
                migrate_cpu_prefill_checkpoint(
                    source,
                    output,
                    "b" * 64,
                    target_inventory={aggregate.name: frozenset({64})},
                )

            self.assertFalse(output.exists())

    def test_rejects_nonempty_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            source = temp / "source"
            output = temp / "output"
            self.write_source(source)
            output.mkdir()
            (output / "unrelated.txt").write_text("keep\n")

            with self.assertRaisesRegex(ValueError, "must be empty"):
                migrate_cpu_prefill_checkpoint(source, output, "b" * 64)

            self.assertEqual((output / "unrelated.txt").read_text(), "keep\n")

    def test_current_protocol_shard_is_rebased_without_semantic_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            source = temp / "source"
            output = temp / "output"
            row = self.diagnostic_row()
            row.update({
                "timing_protocol": LEGACY_ADAPTIVE_TIMING_PROTOCOL,
                "timing_ceiling_policy": (
                    LEGACY_ADAPTIVE_TIMING_CEILING_POLICY
                ),
                "timing_stop_reason": "fixed_samples",
            })
            aggregate, _ = self.write_source(source, row=row)

            manifest = migrate_cpu_prefill_checkpoint(
                source,
                output,
                "b" * 64,
                target_inventory={aggregate.name: frozenset({64})},
            )

            self.assertEqual(manifest["aggregate_shards"], 1)
            self.assertTrue((output / aggregate.name).is_file())

    def test_changed_m_inventory_is_omitted_for_recollection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            source = temp / "source"
            output = temp / "output"
            aggregate, _ = self.write_source(source)

            manifest = migrate_cpu_prefill_checkpoint(
                source,
                output,
                "b" * 64,
                target_inventory={aggregate.name: frozenset({64, 256})},
            )

            self.assertEqual(manifest["aggregate_shards"], 0)
            self.assertFalse((output / aggregate.name).exists())
            omitted = manifest["omitted_incompatible_finalized_shards"]
            self.assertEqual(len(omitted), 1)
            self.assertEqual(omitted[0]["reason"], "complete-m-inventory-changed")
            self.assertEqual(omitted[0]["source_m_values"], [64])
            self.assertEqual(omitted[0]["target_m_values"], [64, 256])


if __name__ == "__main__":
    unittest.main()
