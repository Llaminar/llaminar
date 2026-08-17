#!/usr/bin/env python3
"""Regression tests for strict NativeVNNI paired timing certification."""

from __future__ import annotations

import csv
import math
import statistics
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.paired_confirmation import (  # noqa: E402
    CPU_ISOLATED_PAIRED_PROTOCOL_VERSION,
    CPU_PROCESS_ISOLATED_TIMING_SCOPE,
    CUDA_PAIRED_PROTOCOL_VERSION,
    LEGACY_PAIRED_PROTOCOL_VERSION,
    PAIRED_PROTOCOL_VERSION,
    REQUIRED_COLUMNS,
    REQUIRED_COLUMNS_V1,
    REQUIRED_COLUMNS_V4,
    PairedCellEvidence,
    PairedCellKey,
    PairedTimingComparison,
    certify_paired_confirmation,
    paired_timing_comparisons,
    prefer_promotion_eligible_comparisons,
    promotion_eligible_comparisons,
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

    def test_typed_evidence_rejects_incomplete_order_identity(self) -> None:
        """No fitting caller can construct a partially identified crossover."""

        key = PairedCellKey(
            backend="cpu",
            source_format="Q4_0",
            source_codebook=0,
            execution_codebook=0,
            shape="TypedBoundaryRegression",
            execution_mode="eager",
            m=1,
            n=1152,
            k=5760,
            architecture_class="avx512",
        )
        with self.assertRaisesRegex(ValueError, "invocation-order identity"):
            PairedCellEvidence(
                key=key,
                selected_candidate_id="cpu.selected",
                exact_candidate_id="cpu.exact",
                selected_latency_us=(10.0, 10.0),
                exact_latency_us=(10.0, 10.0),
                selected_first_count=1,
                exact_first_count=1,
                selected_ran_first=(True,),
            )

    def test_crossover_estimator_cancels_large_invocation_position_bias(self) -> None:
        """AB/BA strata remove period bias before declaring candidate regret.

        This pattern is a reduced regression for the CPU M=1 evidence that
        originally produced a false 12.8% miss. Most selected-first ratios are
        below one and most exact-first ratios are above one because invocation
        position dominates the candidate difference. The two populations have
        overlapping tails, so taking one median over their union lands between
        those tails and reports a false failure. Their order-stratified
        crossover estimate recovers the intended 3.7% candidate ratio.
        """

        rows = self._rows(request_id="cpu-pair-position-bias")
        selected_first_ratios = [0.89] * 14 + [1.13]
        exact_first_ratios = [1.208] * 14 + [1.126]
        selected_first_index = 0
        exact_first_index = 0
        for pair_index in range(30):
            pair = [
                row for row in rows if int(row["pair_index"]) == pair_index
            ]
            selected = next(
                row for row in pair if row["candidate_role"] == "selected"
            )
            exact = next(
                row for row in pair if row["candidate_role"] == "exact"
            )
            if selected["within_pair_order"] == "0":
                ratio = selected_first_ratios[selected_first_index]
                selected_first_index += 1
            else:
                ratio = exact_first_ratios[exact_first_index]
                exact_first_index += 1
            exact_latency = 100.0
            selected_latency = exact_latency * ratio
            exact["latency_us"] = f"{exact_latency:.9f}"
            exact["latency_us_hex"] = exact_latency.hex()
            selected["latency_us"] = f"{selected_latency:.9f}"
            selected["latency_us_hex"] = selected_latency.hex()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "position-bias.csv"
            self._write(path, rows)
            cells = read_paired_confirmation_csv((path,))

        pooled_regret = math.expm1(
            statistics.median(cells[0].log_latency_ratios)
        )
        expected_crossover_regret = math.sqrt(0.89 * 1.208) - 1.0
        self.assertGreater(pooled_regret, P95_REGRET_BUDGET)
        self.assertAlmostEqual(
            cells[0].observed_regret,
            expected_crossover_regret,
            places=12,
        )
        comparison = next(iter(paired_timing_comparisons(cells).values()))[0]
        self.assertAlmostEqual(
            comparison.selected_to_exact_median_ratio,
            1.0 + expected_crossover_regret,
            places=12,
        )

    def test_candidates_may_use_distinct_stable_arithmetic_paths(self) -> None:
        """A tournament compares routes; only per-role route drift is invalid."""

        rows = self._rows(request_id="cuda-pair-distinct-routes")
        for row in rows:
            if row["candidate_role"] == "exact":
                row.update({
                    "observed_path": "direct",
                    "observed_tile_n": "64",
                    "observed_cpt": "1",
                    "observed_effective_kb": "0",
                })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "distinct-routes.csv"
            self._write(path, rows)
            cells = read_paired_confirmation_csv((path,))

        self.assertEqual(cells[0].observed_path, "kpar")
        self.assertEqual(cells[0].exact_observed_path, "direct")

    def test_arithmetic_path_drift_within_one_role_fails_closed(self) -> None:
        """Repeated pairs must execute one stable route for each candidate."""

        rows = self._rows(request_id="cuda-pair-route-drift")
        exact_rows = [row for row in rows if row["candidate_role"] == "exact"]
        exact_rows[0]["observed_path"] = "direct"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route-drift.csv"
            self._write(path, rows)
            with self.assertRaisesRegex(ValueError, "changed within a role"):
                read_paired_confirmation_csv((path,))

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

    def test_identical_candidates_require_an_explicit_sealed_witness_id(
        self,
    ) -> None:
        """Ordinary tournaments cannot silently compare one route to itself."""

        request_id = "cpu-single-forceable-witness"
        rows = self._rows(request_id=request_id)
        for row in rows:
            row["candidate_id"] = "cpu.nvnni.verifier.pairwise"
            row["observed_candidate_id"] = row["candidate_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identical.csv"
            self._write(path, rows)
            with self.assertRaisesRegex(ValueError, "are identical"):
                read_paired_confirmation_csv((path,))
            cells = read_paired_confirmation_csv(
                (path,),
                allow_identical_request_ids=frozenset({request_id}),
            )

        self.assertEqual(len(cells), 1)
        self.assertEqual(
            cells[0].selected_candidate_id,
            cells[0].exact_candidate_id,
        )

    def test_unobserved_forced_route_blocks_confirmation(self) -> None:
        """Timing a dispatcher substitute cannot certify the named schedule."""

        rows = self._rows()
        rows[0]["observed_candidate_id"] = "cuda.test.wrong"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.csv"
            self._write(path, rows)
            with self.assertRaisesRegex(ValueError, "did not execute"):
                read_paired_confirmation_csv((path,))

    def test_grouped_cpu_pairs_require_serial_byte_identity(self) -> None:
        """M>1 CPU evidence must independently prove complete-row equality."""

        rows = self._rows(request_id="cpu-grouped-pair-unit")
        for row in rows:
            role = row["candidate_role"]
            candidate = f"cpu.nvnni.verifier.{role}"
            row.update({
                "backend": "cpu",
                "architecture_class": (
                    "test|build=AVX512|runtime=AVX512|threads=28"
                ),
                "phase": "verifier_rows",
                "m": "15",
                "candidate_id": candidate,
                "observed_candidate_id": candidate,
                "observed_path": "grouped-verifier-rows",
                "grouped_output_digest": "fnv1a64:1111111111111111",
                "serial_output_digest": "fnv1a64:1111111111111111",
            })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grouped.csv"
            self._write(path, rows)
            cells = read_paired_confirmation_csv((path,))

        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0].key.m, 15)
        self.assertEqual(cells[0].observed_path, "grouped-verifier-rows")

    def test_grouped_cpu_digest_mismatch_blocks_confirmation(self) -> None:
        """A producer pass flag cannot conceal non-identical grouped bytes."""

        rows = self._rows(request_id="cpu-grouped-byte-failure")
        for row in rows:
            role = row["candidate_role"]
            candidate = f"cpu.nvnni.verifier.{role}"
            row.update({
                "backend": "cpu",
                "architecture_class": (
                    "test|build=AVX512|runtime=AVX512|threads=28"
                ),
                "phase": "verifier_rows",
                "m": "15",
                "candidate_id": candidate,
                "observed_candidate_id": candidate,
                "observed_path": "grouped-verifier-rows",
            })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grouped-mismatch.csv"
            self._write(path, rows)
            with self.assertRaisesRegex(ValueError, "not byte-identical"):
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
        """Historical CPU pairs remain readable only as development priors."""

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
        self.assertFalse(cells[0].promotion_eligible_timing)

    def test_cpu_v4_requires_and_retains_process_isolation(self) -> None:
        """Promotion-grade CPU timing proves it had no socket-local co-run peer."""

        rows = self._rows(request_id="cpu-pair-isolated")
        for row in rows:
            row.update({
                "protocol_version": CPU_ISOLATED_PAIRED_PROTOCOL_VERSION,
                "backend": "cpu",
                "phase": "decode_m1",
                "architecture_class": (
                    "x86_64|build=AVX512|runtime=AVX2|threads=28"
                ),
                "timing_scope": CPU_PROCESS_ISOLATED_TIMING_SCOPE,
                "mpi_world_size": "1",
                "graph_capture_ok": "0",
            })
            row["candidate_id"] = row["candidate_id"].replace(
                "cuda.test", "cpu.nvnni.decode"
            )
            row["observed_candidate_id"] = row["candidate_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cpu-v4.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=sorted(REQUIRED_COLUMNS_V4),
                )
                writer.writeheader()
                writer.writerows(rows)
            cells = read_paired_confirmation_csv((path,))
        self.assertTrue(cells[0].promotion_eligible_timing)
        self.assertEqual(cells[0].mpi_world_size, 1)

        for row in rows:
            row["mpi_world_size"] = "2"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cpu-v4-concurrent.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=sorted(REQUIRED_COLUMNS_V4),
                )
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "used an MPI peer"):
                read_paired_confirmation_csv((path,))

    def test_isolated_cpu_edge_supersedes_legacy_co_run_evidence(self) -> None:
        """A clean edge is never averaged with an unrecorded co-run context."""

        key = PairedCellKey(
            backend="cpu",
            source_format="IQ4_NL",
            source_codebook=4,
            execution_codebook=4,
            shape="IsolationPreference",
            execution_mode="eager",
            m=1,
            n=1984,
            k=10624,
            architecture_class="avx2",
        )
        legacy = PairedTimingComparison(
            key=key,
            selected_effective_candidate_id="cpu.nbc1",
            exact_effective_candidate_id="cpu.nbc4",
            selected_to_exact_median_ratio=1.68,
            pair_count=30,
        )
        isolated = PairedTimingComparison(
            key=key,
            selected_effective_candidate_id="cpu.nbc1",
            exact_effective_candidate_id="cpu.nbc4",
            selected_to_exact_median_ratio=0.82,
            pair_count=30,
            timing_scope=CPU_PROCESS_ISOLATED_TIMING_SCOPE,
            mpi_world_size=1,
        )
        self.assertEqual(
            prefer_promotion_eligible_comparisons((legacy, isolated)),
            (isolated,),
        )
        self.assertEqual(
            promotion_eligible_comparisons((legacy, isolated)),
            (isolated,),
        )


if __name__ == "__main__":
    unittest.main()
