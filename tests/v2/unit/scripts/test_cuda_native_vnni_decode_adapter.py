#!/usr/bin/env python3
"""Focused regressions for strong CUDA NativeVNNI decode adaptation."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.v2.performance.kernels.native_vnni_dispatch.adapters.cuda_decode import (  # noqa: E402
    CUDADecodeAdapterContext,
    adapt_cuda_decode_csv,
    adapt_cuda_decode_row,
    read_cuda_decode_timing_sidecars,
)
from tests.v2.performance.kernels.native_vnni_dispatch.exact_oracle import (  # noqa: E402
    candidate_is_eligible,
)
from tests.v2.performance.kernels.native_vnni_dispatch.cuda_resume import (  # noqa: E402
    extract_cuda_development_generation,
)


class CUDANativeVNNIDecodeAdapterTest(unittest.TestCase):
    """Prove candidate normalization, route inheritance, and byte gates."""

    @staticmethod
    def context() -> CUDADecodeAdapterContext:
        return CUDADecodeAdapterContext.workflow_smoke(
            corpus_id="sha256:" + "7" * 64
        )

    @staticmethod
    def fast_row() -> dict[str, str]:
        candidate = "cuda.nvnni.decode.fast_m1.kpar.tn256.cpt4.kb8"
        return {
            "backend": "cuda",
            "phase": "decode",
            "source_format": "Q8_K",
            "source_codebook": "21",
            "execution_codebook": "19",
            "shape": "StrongSmoke",
            "execution_mode": "eager",
            "m": "1",
            "n": "128",
            "k": "256",
            "candidate_id": candidate,
            "measurement_order": "0",
            "measurement_order_seed": "123456789",
            "measurement_protocol": "sample_interleaved_v1",
            "family": "kpar",
            "tile_n": "256",
            "cpt": "4",
            "target_waves": "0",
            "mkg": "0",
            "max_kb": "0",
            "exact_kb": "8",
            "force_two_phase": "1",
            "weight_bytes": "36864",
            "warmup_count": "2",
            "sample_count": "5",
            "min_us": "10",
            "median_us": "11",
            "p95_us": "13",
            "mad_us": "1",
            "cv": "0.1",
            "effective_bandwidth_gbs": "3.351272727",
            "bit_mismatches": "0",
            "first_bit_mismatch": "0",
            "repeat_byte_mismatches": "0",
            "max_abs": "0",
            "relative_l2": "0",
            "cosine": "1",
            "symmetric_kld": "0",
            "grouped_output_digest": "sha256:same",
            "serial_output_digest": "sha256:same",
            "timing_sample_digest": "sha256:timing",
            "supported": "1",
            "graph_capture_ok": "1",
            "workspace_ok": "1",
            "explicit_stream_ok": "1",
            "route_counter_ok": "1",
            "observed_candidate_id": candidate,
            "observed_path": "kpar",
            "observed_tile_n": "256",
            "observed_cpt": "4",
            "observed_effective_kb": "8",
            "serial_m1_candidate_id": candidate,
            "serial_route_counter_ok": "1",
            "numerical_correctness": "1",
            "correctness_pass": "1",
            "is_winner": "1",
        }

    @classmethod
    def verifier_row(cls) -> dict[str, str]:
        row = cls.fast_row()
        grouped_candidate = (
            "cuda.nvnni.decode.verifier.inherit_serial_m1.r4"
        )
        row.update({
            "candidate_id": grouped_candidate,
            "family": "inherit_serial_m1",
            "grouped_rows": "4",
            "tile_n": "0",
            "cpt": "0",
            "target_waves": "0",
            "mkg": "0",
            "max_kb": "0",
            "exact_kb": "0",
            "force_two_phase": "0",
            "m": "3",
            "observed_candidate_id": grouped_candidate,
        })
        return row

    @classmethod
    def tensor_core_verifier_row(cls) -> dict[str, str]:
        """Model the fixed-W4 production tensor-core evidence ABI."""

        row = cls.verifier_row()
        candidate = "cuda.nvnni.decode.verifier.tensor_core_mma16"
        row.update({
            "candidate_id": candidate,
            "family": "tensor_core_mma16",
            "grouped_rows": "0",
            "observed_candidate_id": candidate,
            "observed_path": "tensor_core",
            "observed_tile_n": "8",
            "observed_cpt": "0",
        })
        return row

    @classmethod
    def fused_row(cls) -> dict[str, str]:
        """Describe one exact compiled CTA-local publication and its resources."""

        row = cls.fast_row()
        candidate = "cuda.nvnni.decode.fast_m1.fused_kpar.tn16.cpt1.kb8"
        row.update({
            "candidate_id": candidate, "family": "fused_kpar",
            "tile_n": "16", "cpt": "1", "observed_candidate_id": candidate,
            "observed_path": "fused_kpar", "observed_tile_n": "16", "observed_cpt": "1",
            "serial_m1_candidate_id": candidate, "compiler_registers": "38",
            "compiler_local_bytes": "0", "compiler_static_shared_bytes": "0",
            "launch_dynamic_shared_bytes": "512", "compiler_max_threads": "1024",
            "launch_active_blocks_per_sm": "8", "launch_threads": "128",
        })
        return row

    def test_fused_kpar_resource_and_identity_admission(self) -> None:
        """Neither spilled kernels nor a substituted reduction tree are evidence."""

        row = self.fused_row()
        observation = adapt_cuda_decode_row(row, self.context())
        self.assertTrue(candidate_is_eligible(observation))
        for field, value in (
            ("compiler_local_bytes", "4"), ("compiler_registers", "0"),
            ("launch_dynamic_shared_bytes", "256"), ("compiler_max_threads", "64"),
            ("launch_active_blocks_per_sm", "0"), ("launch_threads", "64"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                adapt_cuda_decode_row({**row, field: value}, self.context())
        changed = {**row, "observed_effective_kb": "4", "correctness_pass": "0"}
        self.assertFalse(candidate_is_eligible(adapt_cuda_decode_row(changed, self.context())))

    def test_grouped_inherits_exact_fused_kpar_partition(self) -> None:
        """A grouped physical producer may change, but its serial KB may not."""

        row = self.verifier_row()
        row.update({
            "serial_m1_candidate_id": "cuda.nvnni.decode.fast_m1.fused_kpar.tn16.cpt1.kb8",
            "observed_path": "fused_kpar", "observed_tile_n": "16", "observed_cpt": "1",
        })
        self.assertTrue(candidate_is_eligible(adapt_cuda_decode_row(row, self.context())))
        row.update({"observed_effective_kb": "4", "correctness_pass": "0"})
        self.assertFalse(candidate_is_eligible(adapt_cuda_decode_row(row, self.context())))

    def test_fast_m1_preserves_exact_kpart_identity(self) -> None:
        observation = adapt_cuda_decode_row(self.fast_row(), self.context())
        self.assertEqual(observation.runtime_codebook_id, 19)
        self.assertEqual(
            observation.effective_candidate_id,
            "cuda.nvnni.decode.fast_m1.kpar.tn256.cpt4.kb8",
        )
        self.assertTrue(candidate_is_eligible(observation))

    def test_verifier_inherits_public_m1_and_is_byte_eligible(self) -> None:
        observation = adapt_cuda_decode_row(self.verifier_row(), self.context())
        self.assertEqual(observation.m, 3)
        self.assertTrue(observation.bitwise_equal)
        self.assertTrue(candidate_is_eligible(
            observation,
            self.context().serial_m1_policy_hash,
        ))

    def test_sixteen_row_verifier_uses_same_bitwise_contract(self) -> None:
        row = self.verifier_row()
        row["m"] = "16"
        row["grouped_rows"] = "16"
        row["candidate_id"] = (
            "cuda.nvnni.decode.verifier.inherit_serial_m1.r16"
        )
        row["observed_candidate_id"] = row["candidate_id"]
        observation = adapt_cuda_decode_row(row, self.context())
        self.assertEqual(observation.m, 16)
        self.assertTrue(observation.bitwise_equal)
        self.assertTrue(candidate_is_eligible(
            observation,
            self.context().serial_m1_policy_hash,
        ))

    def test_tensor_core_row_materializes_fixed_w4_registry_geometry(self) -> None:
        """Retained v1 rows authenticate the sole W4 tensor-core candidate."""

        observation = adapt_cuda_decode_row(
            self.tensor_core_verifier_row(),
            self.context(),
        )
        self.assertEqual(
            observation.effective_candidate_id,
            "cuda.nvnni.decode.verifier.tensor_core_mma16",
        )
        self.assertEqual(observation.config_json, {
            "family": "tensor_core_mma16",
            "warps_per_block": 4,
        })
        self.assertTrue(observation.forced_route_ok)
        self.assertTrue(candidate_is_eligible(
            observation,
            self.context().serial_m1_policy_hash,
        ))

    def test_tensor_core_row_rejects_inherited_route_telemetry(self) -> None:
        row = self.tensor_core_verifier_row()
        row.update({
            "observed_path": "kpar",
            "observed_tile_n": "256",
            "observed_cpt": "4",
            "correctness_pass": "0",
        })
        observation = adapt_cuda_decode_row(row, self.context())
        self.assertFalse(observation.forced_route_ok)
        self.assertFalse(candidate_is_eligible(
            observation,
            self.context().serial_m1_policy_hash,
        ))

    def test_resume_extracts_development_bytes_by_sealed_run_identity(self) -> None:
        """Combined certification output recovers the exact profiler lineage."""

        fields = (
            "schema_version",
            "run_id",
            "backend",
            "semantic_contract",
            "git_revision",
            "build_id",
            "compiler_id",
            "architecture_class",
            "device_name",
            "driver_runtime",
            "serial_m1_policy_hash",
            "config_json",
        )
        development = {
            "schema_version": "fixture-v1",
            "run_id": "development-generation",
            "backend": "cuda",
            "semantic_contract": "Fast",
            "git_revision": "deadbeef",
            "build_id": "sha256:development-build",
            "compiler_id": "nvcc-fixture",
            "architecture_class": "sm_86",
            "device_name": "RTX 3090",
            "driver_runtime": "driver-fixture",
            "serial_m1_policy_hash": "sha256:development-policy",
            "config_json": '{"family":"kpar","tile_n":128}',
        }
        sealed = {
            **development,
            "run_id": "sealed-generation",
            "build_id": "sha256:sealed-build",
            "serial_m1_policy_hash": "sha256:sealed-policy",
        }
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "combined.csv"
            destination = Path(root) / "development.csv"
            context = Path(root) / "development.context"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(development)
                writer.writerow(sealed)
                writer.writerow(development)
            original_lines = source.read_bytes().splitlines(keepends=True)

            counts = extract_cuda_development_generation(
                source,
                destination,
                context,
                excluded_run_id="sealed-generation",
            )

            self.assertEqual(counts, (2, 1))
            self.assertEqual(
                destination.read_bytes(),
                original_lines[0] + original_lines[1] + original_lines[3],
            )
            self.assertEqual(
                context.read_text(encoding="utf-8").splitlines(),
                [
                    "development-generation",
                    "deadbeef",
                    "sha256:development-build",
                    "nvcc-fixture",
                    "sm_86",
                    "RTX 3090",
                    "driver-fixture",
                    "sha256:development-policy",
                ],
            )

    def test_parallel_aggregate_adaptation_matches_serial_order(self) -> None:
        """Byte-range workers must preserve every adapted row and its order."""

        rows = []
        for index in range(40):
            row = self.fast_row()
            row["shape"] = f"StrongSmoke{index:03d}"
            row["n"] = str(128 + index)
            rows.append(row)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "aggregate.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            serial = adapt_cuda_decode_csv(
                (path,),
                self.context(),
                workers=1,
                parallel_threshold_bytes=1,
            )
            parallel = adapt_cuda_decode_csv(
                (path,),
                self.context(),
                workers=2,
                parallel_threshold_bytes=1,
            )

        self.assertEqual(parallel.observations, serial.observations)

    def test_verifier_byte_mismatch_is_ineligible(self) -> None:
        row = self.verifier_row()
        row.update({
            "bit_mismatches": "1",
            "first_bit_mismatch": "17",
            "grouped_output_digest": "sha256:grouped",
            "serial_output_digest": "sha256:serial",
            "max_abs": "0.0001",
            "correctness_pass": "0",
        })
        observation = adapt_cuda_decode_row(row, self.context())
        self.assertFalse(candidate_is_eligible(
            observation,
            self.context().serial_m1_policy_hash,
        ))

    def test_inherited_route_mismatch_cannot_claim_forced_route(self) -> None:
        row = self.verifier_row()
        row["observed_tile_n"] = "128"
        row["correctness_pass"] = "0"
        observation = adapt_cuda_decode_row(row, self.context())
        self.assertFalse(observation.forced_route_ok)
        self.assertFalse(candidate_is_eligible(
            observation,
            self.context().serial_m1_policy_hash,
        ))

    def test_inherited_kpart_mismatch_cannot_claim_batch_invariance(self) -> None:
        row = self.verifier_row()
        row["observed_effective_kb"] = "4"
        row["correctness_pass"] = "0"
        observation = adapt_cuda_decode_row(row, self.context())
        self.assertFalse(observation.forced_route_ok)
        self.assertFalse(candidate_is_eligible(
            observation,
            self.context().serial_m1_policy_hash,
        ))

    def test_removed_atomic_candidate_is_rejected(self) -> None:
        row = self.fast_row()
        row["candidate_id"] = (
            "cuda.nvnni.kpar.tn256.cpt4.tw8.mkg4.kb4.phase2"
        )
        with self.assertRaisesRegex(ValueError, "unknown forceable candidate"):
            adapt_cuda_decode_row(row, self.context())

    def test_batched_cuda_event_samples_preserve_per_launch_evidence(self) -> None:
        """Accept stable replay batches and reject a changed replay divisor."""

        fieldnames = (
            "backend", "phase", "source_format", "source_codebook",
            "execution_codebook", "shape", "execution_mode", "m", "n",
            "k", "candidate_id", "measurement_order",
            "measurement_order_seed", "measurement_protocol",
            "sample_measurement_order", "sample_order_seed", "sample_index",
            "timed_replays", "latency_us", "latency_us_hex",
        )
        base = {
            "backend": "cuda",
            "phase": "decode",
            "source_format": "Q8_K",
            "source_codebook": "21",
            "execution_codebook": "19",
            "shape": "StrongSmoke",
            "execution_mode": "eager",
            "m": "1",
            "n": "128",
            "k": "256",
            "candidate_id": (
                "cuda.nvnni.decode.fast_m1.kpar.tn256.cpt4.kb8"
            ),
            "measurement_order": "0",
            "measurement_order_seed": "123456789",
            "measurement_protocol": "sample_interleaved_v1",
            "sample_measurement_order": "0",
            "timed_replays": "16",
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "timing.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for index, latency in enumerate((1.25, 1.5)):
                    writer.writerow({
                        **base,
                        "sample_order_seed": str(9000 + index),
                        "sample_index": str(index),
                        "latency_us": f"{latency:.9f}",
                        "latency_us_hex": latency.hex(),
                    })
            evidence = read_cuda_decode_timing_sidecars((path,))
            self.assertEqual(next(iter(evidence.values())), (1.25, 1.5))

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["timed_replays"] = "8"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "timed_replays changed"):
                read_cuda_decode_timing_sidecars((path,))

    def test_every_timing_sample_is_a_complete_candidate_permutation(self) -> None:
        """Missing or repeated candidates cannot masquerade as interleaving."""

        fieldnames = (
            "backend", "phase", "source_format", "source_codebook",
            "execution_codebook", "shape", "execution_mode", "m", "n",
            "k", "candidate_id", "measurement_order",
            "measurement_order_seed", "measurement_protocol",
            "sample_measurement_order", "sample_order_seed", "sample_index",
            "timed_replays", "latency_us", "latency_us_hex",
        )
        candidates = (
            "cuda.nvnni.decode.fast_m1.kpar.tn64.cpt2.kb8",
            "cuda.nvnni.decode.fast_m1.kpar.tn32.cpt1.kb4",
        )
        rows = []
        for sample_index in range(3):
            order = candidates if sample_index % 2 == 0 else candidates[::-1]
            for sample_order, candidate in enumerate(order):
                latency = 10.0 + sample_index + sample_order * 0.1
                rows.append({
                    "backend": "cuda",
                    "phase": "decode",
                    "source_format": "Q8_K",
                    "source_codebook": "21",
                    "execution_codebook": "19",
                    "shape": "StrongSmoke",
                    "execution_mode": "eager",
                    "m": "1",
                    "n": "128",
                    "k": "256",
                    "candidate_id": candidate,
                    "measurement_order": str(candidates.index(candidate)),
                    "measurement_order_seed": "123456789",
                    "measurement_protocol": "sample_interleaved_v1",
                    "sample_measurement_order": str(sample_order),
                    "sample_order_seed": str(7000 + sample_index),
                    "sample_index": str(sample_index),
                    "timed_replays": "16",
                    "latency_us": f"{latency:.9f}",
                    "latency_us_hex": latency.hex(),
                })

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "interleaved.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            evidence = read_cuda_decode_timing_sidecars((path,))
            self.assertEqual(len(evidence), 2)

            # Keep the first group larger so the approximate half-file split
            # lands inside it and must advance to the second group's boundary.
            second_group = [
                {
                    **row,
                    "shape": "StrongSmokeSecond",
                    "sample_measurement_order": "0",
                }
                for row in rows
                if row["candidate_id"] == candidates[0]
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows((*rows, *second_group))
            serial = read_cuda_decode_timing_sidecars((path,), workers=1)
            parallel = read_cuda_decode_timing_sidecars((path,), workers=2)
            self.assertEqual(parallel, serial)
            self.assertEqual(len(parallel), 3)

            rows[1]["sample_measurement_order"] = "0"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "not one contiguous"):
                read_cuda_decode_timing_sidecars((path,))


if __name__ == "__main__":
    unittest.main()
