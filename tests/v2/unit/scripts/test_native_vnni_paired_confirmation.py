#!/usr/bin/env python3
"""Regression tests for strict NativeVNNI paired timing certification."""

from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.paired_confirmation import (  # noqa: E402
    CUDA_PAIRED_PROTOCOL_VERSION,
    LEGACY_PAIRED_PROTOCOL_VERSION,
    PAIRED_PROTOCOL_VERSION,
    REQUIRED_COLUMNS,
    REQUIRED_COLUMNS_V1,
    certify_paired_confirmation,
    paired_timing_comparisons,
    read_paired_confirmation_csv,
)
from native_vnni_dispatch.schema import P95_REGRET_BUDGET  # noqa: E402


class NativeVNNIPairedConfirmationTest(unittest.TestCase):
    """Prove pair completeness, route evidence, and simultaneous bounds."""

    @staticmethod
    def _rows(
        *,
        pair_count: int = 30,
        selected_ratio: float = 1.01,
        request_id: str = "cuda-pair-unit-first",
    ) -> list[dict[str, str]]:
        """Build deterministic promotion-strength selected/exact pairs."""

        rows = []
        for pair_index in range(pair_count):
            exact_latency = 10.0 + (pair_index % 5) * 0.01
            latency_by_role = {
                "selected": exact_latency * selected_ratio,
                "exact": exact_latency,
            }
            order = (
                ("selected", "exact")
                if pair_index % 2 == 0
                else ("exact", "selected")
            )
            for within_pair_order, role in enumerate(order):
                candidate = f"cuda.test.{role}"
                latency = latency_by_role[role]
                rows.append({
                    "protocol_version": PAIRED_PROTOCOL_VERSION,
                    "request_id": request_id,
                    "backend": "cuda",
                    "architecture_class": "sm86",
                    "phase": "decode",
                    "source_format": "Q4_0",
                    "source_codebook": "0",
                    "execution_codebook": "0",
                    "shape": "PairedRegression",
                    "execution_mode": "eager",
                    "m": "1",
                    "n": "1152",
                    "k": "5760",
                    "pair_index": str(pair_index),
                    "configured_pair_count": str(pair_count),
                    "within_pair_order": str(within_pair_order),
                    "cell_order_seed": "12345",
                    "pair_order_seed": str(50_000 + pair_index),
                    "candidate_role": role,
                    "candidate_id": candidate,
                    "timed_replays": "16",
                    "latency_us": f"{latency:.9f}",
                    "latency_us_hex": latency.hex(),
                    "warmup_count": "5",
                    "bit_mismatches": "0",
                    "first_bit_mismatch": "0",
                    "repeat_byte_mismatches": "0",
                    "max_abs": "0",
                    "relative_l2": "0",
                    "cosine": "1",
                    "symmetric_kld": "0",
                    "grouped_output_digest": "fnv1a64:1111111111111111",
                    "serial_output_digest": "fnv1a64:2222222222222222",
                    "graph_capture_ok": "1",
                    "workspace_ok": "1",
                    "explicit_stream_ok": "1",
                    "route_counter_ok": "1",
                    "observed_candidate_id": candidate,
                    "observed_path": "kpar",
                    "observed_tile_n": "64",
                    "observed_cpt": "2",
                    "observed_effective_kb": "60",
                    "serial_m1_candidate_id": "cuda.test.serial",
                    "numerical_correctness": "1",
                    "correctness_pass": "1",
                })
        return rows

    @staticmethod
    def _write(path: Path, rows: list[dict[str, str]]) -> None:
        """Write one exact-schema temporary paired shard."""

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)

    def test_valid_pairs_produce_a_promotable_simultaneous_certificate(self) -> None:
        """Low-regret interleaved evidence passes both hard gates."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired.csv"
            self._write(path, self._rows())
            cells = read_paired_confirmation_csv((path,))
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0].pair_count, 30)
        self.assertEqual(cells[0].selected_first_count, 15)
        self.assertEqual(cells[0].exact_first_count, 15)
        report = certify_paired_confirmation(
            cells,
            bootstrap_replicates=1_000,
        )
        self.assertTrue(math.isclose(
            report.observed_max_regret,
            0.01,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ))
        report.require_promotable()

    def test_vectorized_bootstrap_is_invariant_to_worker_count(self) -> None:
        """Parallel partitioning cannot alter a retained certificate."""

        first_rows = self._rows(request_id="cuda-pair-workers-first")
        second_rows = self._rows(
            request_id="cuda-pair-workers-second",
            selected_ratio=1.02,
        )
        for row in second_rows:
            row["shape"] = "PairedRegressionSecond"
            row["n"] = "1280"
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.csv"
            second_path = Path(directory) / "second.csv"
            self._write(first_path, first_rows)
            self._write(second_path, second_rows)
            cells = read_paired_confirmation_csv((first_path, second_path))

        serial = certify_paired_confirmation(
            cells,
            bootstrap_replicates=1_000,
            workers=1,
        )
        parallel = certify_paired_confirmation(
            cells,
            bootstrap_replicates=1_000,
            workers=2,
        )
        self.assertEqual(serial, parallel)

    def test_missing_pair_blocks_confirmation(self) -> None:
        """A partial final pair cannot silently reduce the sample count."""

        rows = self._rows()
        rows.pop()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.csv"
            self._write(path, rows)
            with self.assertRaisesRegex(ValueError, "does not have two rows"):
                read_paired_confirmation_csv((path,))

    def test_non_interleaved_order_blocks_confirmation(self) -> None:
        """Both candidates must occupy distinct positions in every pair."""

        rows = self._rows()
        rows[1]["within_pair_order"] = "0"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "order.csv"
            self._write(path, rows)
            with self.assertRaisesRegex(ValueError, "order is incomplete"):
                read_paired_confirmation_csv((path,))

    def test_unobserved_forced_route_blocks_confirmation(self) -> None:
        """Timing a dispatcher substitute cannot certify the named schedule."""

        rows = self._rows()
        rows[0]["observed_candidate_id"] = "cuda.test.wrong"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.csv"
            self._write(path, rows)
            with self.assertRaisesRegex(ValueError, "did not execute"):
                read_paired_confirmation_csv((path,))

    def test_high_regret_fails_the_simultaneous_gate(self) -> None:
        """High-confidence evidence cannot waive the five-percent ceiling."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slow.csv"
            self._write(path, self._rows(selected_ratio=1.06))
            cells = read_paired_confirmation_csv((path,))
        report = certify_paired_confirmation(
            cells,
            bootstrap_replicates=1_000,
        )
        with self.assertRaisesRegex(ValueError, "max-regret UCB"):
            report.require_promotable()

    def test_exact_five_percent_boundary_is_not_promotable(self) -> None:
        """The shared five-percent threshold is deliberately strict."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boundary.csv"
            self._write(
                path,
                self._rows(selected_ratio=1.0 + P95_REGRET_BUDGET),
            )
            cells = read_paired_confirmation_csv((path,))
        report = certify_paired_confirmation(
            cells,
            bootstrap_replicates=1_000,
        )
        with self.assertRaisesRegex(ValueError, "max-regret UCB"):
            report.require_promotable()

    def test_one_cell_retains_multiple_selected_candidate_corrections(self) -> None:
        """Successive provisional choices form a tournament, not an overwrite."""

        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.csv"
            second_path = Path(directory) / "second.csv"
            first_rows = self._rows()
            second_rows = self._rows(
                selected_ratio=1.02,
                request_id="cuda-pair-unit-second",
            )
            for row in second_rows:
                if row["candidate_role"] == "selected":
                    row["candidate_id"] = "cuda.test.selected.second"
                    row["observed_candidate_id"] = row["candidate_id"]
            self._write(first_path, first_rows)
            self._write(second_path, second_rows)
            cells = (
                *read_paired_confirmation_csv((first_path,)),
                *read_paired_confirmation_csv((second_path,)),
            )
        comparisons = paired_timing_comparisons(cells)
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(len(next(iter(comparisons.values()))), 2)

    def test_one_v2_file_can_retain_multiple_edges_for_the_same_cell(self) -> None:
        """Request IDs make one batched trainer transaction unambiguous."""

        first_rows = self._rows(request_id="cuda-pair-unit-a")
        second_rows = self._rows(
            selected_ratio=1.02,
            request_id="cuda-pair-unit-b",
        )
        for row in second_rows:
            if row["candidate_role"] == "selected":
                row["candidate_id"] = "cuda.test.selected.second"
                row["observed_candidate_id"] = row["candidate_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tournament.csv"
            self._write(path, [*first_rows, *second_rows])
            cells = read_paired_confirmation_csv((path,))
        self.assertEqual(len(cells), 2)
        comparisons = paired_timing_comparisons(cells)
        self.assertEqual(len(next(iter(comparisons.values()))), 2)

    def test_legacy_v1_file_remains_readable_with_content_bound_identity(self) -> None:
        """Retained pre-manifest evidence is portable across filesystem paths."""

        rows = self._rows()
        for row in rows:
            row["protocol_version"] = LEGACY_PAIRED_PROTOCOL_VERSION
            del row["request_id"]
            del row["architecture_class"]
        with tempfile.TemporaryDirectory() as directory:
            request_ids = []
            for name in ("first.csv", "renamed.csv"):
                path = Path(directory) / name
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=sorted(REQUIRED_COLUMNS_V1),
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                request_ids.append(
                    read_paired_confirmation_csv((path,))[0].request_id
                )
        self.assertEqual(request_ids[0], request_ids[1])
        self.assertTrue(request_ids[0].startswith("legacy-v1:"))

    def test_retained_cuda_v2_file_remains_readable(self) -> None:
        """The generic v3 reader preserves already-collected CUDA evidence."""

        rows = self._rows()
        for row in rows:
            row["protocol_version"] = CUDA_PAIRED_PROTOCOL_VERSION
            del row["architecture_class"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cuda-v2.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=sorted(REQUIRED_COLUMNS - {"architecture_class"}),
                )
                writer.writeheader()
                writer.writerows(rows)
            cells = read_paired_confirmation_csv((path,))
        self.assertEqual(cells[0].key.backend, "cuda")
        self.assertEqual(cells[0].key.architecture_class, "")

    def test_cpu_v3_retains_isa_domain_without_graph_capture(self) -> None:
        """CPU pairs are eager and retain their build/runtime policy surface."""

        rows = self._rows(request_id="cpu-pair-unit")
        for row in rows:
            row.update({
                "backend": "cpu",
                "phase": "decode_m1",
                "architecture_class": (
                    "x86_64|build=AVX512|runtime=AVX2|threads=28"
                ),
                "graph_capture_ok": "0",
            })
            row["candidate_id"] = row["candidate_id"].replace(
                "cuda.test", "cpu.nvnni.decode"
            )
            row["observed_candidate_id"] = row["candidate_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cpu-v3.csv"
            self._write(path, rows)
            cells = read_paired_confirmation_csv((path,))
        self.assertEqual(cells[0].key.backend, "cpu")
        self.assertIn("build=AVX512", cells[0].key.architecture_class)


if __name__ == "__main__":
    unittest.main()
