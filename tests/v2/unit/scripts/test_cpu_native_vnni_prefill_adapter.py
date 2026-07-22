#!/usr/bin/env python3
"""Regression tests for strong CPU NativeVNNI prefill evidence."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.adapters.cpu_prefill import (  # noqa: E402
    CPUPrefillAdapterContext,
    adapt_cpu_prefill_csv,
    adapt_cpu_prefill_row,
)
from native_vnni_dispatch.adapters.evidence import (  # noqa: E402
    summarize_sorted_timing,
)
from native_vnni_dispatch.candidate_observation import (  # noqa: E402
    read_observation_csv,
    write_observation_csv,
)
from native_vnni_dispatch.profiles import (  # noqa: E402
    ADAPTIVE_TIMING_CEILING_POLICY,
    ADAPTIVE_TIMING_PROTOCOL,
    LEGACY_ADAPTIVE_TIMING_CEILING_POLICY,
    CANDIDATE_EXPANSION_EVIDENCE_KEY,
    CANDIDATE_EXPANSION_NORMALIZATION_SCHEMA,
    CANDIDATE_EXPANSION_TIMING_STOP_REASON,
    LEGACY_ADAPTIVE_TIMING_PROTOCOL,
    MeasurementProfile,
    observation_has_promotion_timing,
)
from native_vnni_dispatch.schema import SemanticContract  # noqa: E402


class CPUNativeVNNIPrefillAdapterTest(unittest.TestCase):
    """Prove full-K route identity and arbitrary-M admission."""

    @staticmethod
    def context() -> CPUPrefillAdapterContext:
        return CPUPrefillAdapterContext.workflow_smoke(
            corpus_id="sha256:" + "1" * 64
        )

    @staticmethod
    def row() -> dict[str, str]:
        candidate = "cpu.nvnni.prefill.two_row_tiles.nbc1.full_k"
        return {
            "backend": "cpu",
            "phase": "prefill_gemm",
            "source_format": "Q4_K",
            "source_codebook": "5",
            "execution_codebook": "5",
            "shape": "32B_FFN_Up",
            "execution_mode": "eager",
            "m": "64",
            "n": "27648",
            "k": "5120",
            "candidate_id": candidate,
            "build_isa": "AVX512",
            "runtime_isa_requested": "AVX512",
            "runtime_isa_effective": "AVX512",
            "threads": "28",
            "measurement_coordination": "process-local-complete-round-v1",
            "mpi_world_size": "1",
            "mpi_rank": "0",
            "weight_bytes": "1000000",
            "serial_oracle_policy": "boundary-quartile-sentinel-v1",
            "serial_oracle_rows": "16",
            "preconditioning_policy": "disabled",
            "preconditioning_m": "0",
            "preconditioning_budget_us": "0",
            "preconditioning_duration_us": "0",
            "warmup_count": "2",
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
            "global_timing_round_count": "3",
            "sample_count": "3",
            "timing_protocol": ADAPTIVE_TIMING_PROTOCOL,
            "timing_ceiling_policy": ADAPTIVE_TIMING_CEILING_POLICY,
            "min_sample_count": "3",
            "stable_sample_count": "3",
            "max_sample_count": "3",
            "timing_budget_us": "0",
            "timed_duration_us": "30",
            "stationary_sample_begin": "0",
            "stationary_sample_count": "3",
            "stationary_duration_us": "30",
            "median_stability_limit": "0.02",
            "median_relative_drift": "0",
            "timing_converged": "1",
            "timing_stop_reason": "stationary_window",
            "min_us": "9",
            "median_us": "10",
            "p95_us": "11",
            "mad_us": "1",
            "cv": "0.1",
            "serial_median_us": "20",
            "speedup": "2",
            "bit_mismatches": "0",
            "first_bit_mismatch": "0",
            "repeat_byte_mismatches": "0",
            "max_abs": "0",
            "relative_l2": "0",
            "cosine": "1",
            "symmetric_kld": "0",
            "grouped_output_digest": "fnv1a64:equal",
            "serial_output_digest": "fnv1a64:equal",
            "timing_sample_digest": "fnv1a64:timing",
            "route_counter_ok": "1",
            "observed_candidate_id": candidate,
            "k_tiles": "1",
            "k_tile_blocks": "160",
            "n_block_chunks": "1",
            "numerical_correctness": "1",
            "correctness_pass": "1",
            "is_winner": "1",
        }

    @classmethod
    def production_row_and_timings(cls) -> tuple[dict[str, str], tuple[float, ...]]:
        """Build authenticated elapsed/stable evidence for an expensive GEMM."""

        timings = (500000.0, 501000.0, 499000.0, 500500.0, 499500.0)
        summary = summarize_sorted_timing(sorted(timings))
        row = cls.row()
        row.update({
            "preconditioning_policy": "source-complete-round-v1",
            "preconditioning_m": "64",
            "preconditioning_budget_us": "1000000",
            "preconditioning_duration_us": "1100000",
            "warmup_count": "5",
            "warmup_round_count": "3",
            "warmup_budget_policy": "source-fixed-v1",
            "warmup_probe_latency_us": "500000",
            "warmup_budget_floor_us": "1000000",
            "warmup_latency_multiplier": "0",
            "warmup_budget_ceiling_us": "0",
            "warmup_budget_us": "1000000",
            "warmup_duration_us": "1100000",
            "global_warmup_round_count": "3",
            "warmup_wall_duration_us": "1200000",
            "warmup_max_round_duration_us": "400000",
            "warmup_round_timeout_us": "30000000",
            "measurement_coordination": "mpi-complete-round-v1",
            "mpi_world_size": "2",
            "mpi_rank": "0",
            "global_timing_round_count": str(len(timings)),
            "sample_count": str(len(timings)),
            "min_sample_count": "5",
            "stable_sample_count": "5",
            "max_sample_count": "180",
            "timing_budget_us": "2000000",
            "timed_duration_us": str(sum(timings)),
            "stationary_sample_begin": "0",
            "stationary_sample_count": str(len(timings)),
            "stationary_duration_us": str(sum(timings)),
            "median_stability_limit": "0.02",
            "median_relative_drift": str(1500.0 / 501000.0),
            "timing_converged": "1",
            "timing_stop_reason": "stationary_window",
            "min_us": str(summary.minimum),
            "median_us": str(summary.median),
            "p95_us": str(summary.p95),
            "mad_us": str(summary.mad),
            "cv": str(summary.cv),
            "timing_sample_digest": summary.digest,
        })
        return row, timings

    @classmethod
    def candidate_expansion_row_and_timings(
        cls,
    ) -> tuple[dict[str, str], tuple[float, ...]]:
        """Build the bounded evidence permitted only for anchored expansion."""

        timings = (20000.0,) * 5
        summary = summarize_sorted_timing(sorted(timings))
        row = cls.row()
        row.update({
            "m": "256",
            "preconditioning_policy": "source-complete-round-v1",
            "preconditioning_m": "64",
            "preconditioning_budget_us": "500000",
            "preconditioning_duration_us": "504000",
            "warmup_count": "127",
            "warmup_round_count": "125",
            "warmup_budget_policy": "transition-fixed-v1",
            "warmup_probe_latency_us": "4000",
            "warmup_budget_floor_us": "25000",
            "warmup_latency_multiplier": "0",
            "warmup_budget_ceiling_us": "0",
            "warmup_budget_us": "25000",
            "warmup_duration_us": "28000",
            "global_warmup_round_count": "125",
            "warmup_wall_duration_us": "520000",
            "warmup_max_round_duration_us": "4100",
            "warmup_round_timeout_us": "30000000",
            "measurement_coordination": "mpi-complete-round-v1",
            "mpi_world_size": "2",
            "mpi_rank": "0",
            "global_timing_round_count": str(len(timings)),
            "sample_count": str(len(timings)),
            "min_sample_count": "5",
            "stable_sample_count": "5",
            "max_sample_count": "60",
            "timing_budget_us": "100000",
            "timed_duration_us": str(sum(timings)),
            "stationary_sample_begin": "0",
            "stationary_sample_count": "5",
            "stationary_duration_us": "100000",
            "median_stability_limit": "0.02",
            "median_relative_drift": "0",
            "timing_converged": "1",
            "timing_stop_reason": "stationary_window",
            "min_us": str(summary.minimum),
            "median_us": str(summary.median),
            "p95_us": str(summary.p95),
            "mad_us": str(summary.mad),
            "cv": str(summary.cv),
            "timing_sample_digest": summary.digest,
        })
        return row, timings

    def test_forceable_full_k_candidate_becomes_common_bitwise_evidence(self) -> None:
        observation = adapt_cpu_prefill_row(self.row(), self.context())

        self.assertTrue(observation.supported)
        self.assertTrue(observation.bitwise_equal)
        self.assertEqual(
            observation.semantic_contract,
            SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
        )
        self.assertEqual(observation.operation_kind, "NativeVNNIPrefillProjection")

    def test_prefill_depth_is_not_artificially_bounded_to_mtp_rows(self) -> None:
        row = self.row()
        row["m"] = "16384"

        observation = adapt_cpu_prefill_row(row, self.context())

        self.assertEqual(observation.m, 16384)
        self.assertTrue(observation.supported)

    def test_stationary_duration_allows_csv_decimal_rounding(self) -> None:
        row = self.row()
        row["stationary_duration_us"] = "30.000000001"

        observation = adapt_cpu_prefill_row(row, self.context())

        self.assertTrue(observation.supported)

    def test_normalized_n_block_request_is_retained_but_not_supported(self) -> None:
        row = self.row()
        row["candidate_id"] = (
            "cpu.nvnni.prefill.two_row_tiles.nbc16.full_k"
        )
        row["observed_candidate_id"] = (
            "cpu.nvnni.prefill.row_chunk_grid.full_k"
        )
        row["n_block_chunks"] = "16"
        row["correctness_pass"] = "0"

        observation = adapt_cpu_prefill_row(row, self.context())

        self.assertFalse(observation.supported)
        self.assertFalse(observation.forced_route_ok)

    def test_k_tiled_route_cannot_claim_full_k_correctness(self) -> None:
        row = self.row()
        row["k_tiles"] = "2"

        with self.assertRaisesRegex(ValueError, "correctness_pass disagrees"):
            adapt_cpu_prefill_row(row, self.context())

    def test_serial_kpart_candidate_is_independently_forceable(self) -> None:
        row = self.row()
        candidate = (
            "cpu.nvnni.prefill.decode_equivalent_kpart.pairwise"
        )
        row.update({
            "candidate_id": candidate,
            "observed_candidate_id": candidate,
            "k_tiles": "7",
            "k_tile_blocks": "23",
            "n_block_chunks": "1",
        })

        observation = adapt_cpu_prefill_row(row, self.context())

        self.assertTrue(observation.supported)
        self.assertTrue(observation.bitwise_equal)

    def test_wrong_surface_is_rejected(self) -> None:
        row = self.row()
        row["phase"] = "verifier_rows"

        with self.assertRaisesRegex(ValueError, "wrong surface"):
            adapt_cpu_prefill_row(row, self.context())

    def test_elapsed_stable_evidence_is_installable_without_thirty_replays(self) -> None:
        row, timings = self.production_row_and_timings()
        context = replace(
            self.context(),
            profile=MeasurementProfile.PRODUCTION,
            raw_timing_sidecar_retained=True,
        )

        observation = adapt_cpu_prefill_row(row, context, timings)

        self.assertEqual(observation.sample_count, 5)
        self.assertTrue(observation.bitwise_equal)
        self.assertTrue(observation_has_promotion_timing(observation))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "common.csv"
            write_observation_csv(path, (observation,))
            restored = read_observation_csv((path,)).observations[0]
        self.assertEqual(
            restored.adaptive_timing_evidence,
            observation.adaptive_timing_evidence,
        )
        self.assertTrue(observation_has_promotion_timing(restored))

    def test_current_production_elapsed_floors_are_installable_and_hard(self) -> None:
        """The two-hour corpus budgets remain authenticated promotion gates."""

        timings = (20000.0,) * 5
        summary = summarize_sorted_timing(sorted(timings))
        row = self.row()
        row.update({
            "preconditioning_policy": "source-complete-round-v1",
            "preconditioning_m": "64",
            "preconditioning_budget_us": "50000",
            "preconditioning_duration_us": "50000",
            "warmup_count": "5",
            "warmup_round_count": "3",
            "warmup_budget_policy": "source-fixed-v1",
            "warmup_probe_latency_us": "20000",
            "warmup_budget_floor_us": "50000",
            "warmup_latency_multiplier": "0",
            "warmup_budget_ceiling_us": "0",
            "warmup_budget_us": "50000",
            "warmup_duration_us": "50000",
            "global_warmup_round_count": "3",
            "warmup_wall_duration_us": "60000",
            "warmup_max_round_duration_us": "21000",
            "warmup_round_timeout_us": "30000000",
            "measurement_coordination": "mpi-complete-round-v1",
            "mpi_world_size": "2",
            "mpi_rank": "0",
            "global_timing_round_count": "5",
            "sample_count": "5",
            "min_sample_count": "5",
            "stable_sample_count": "5",
            "max_sample_count": "180",
            "timing_budget_us": "100000",
            "timed_duration_us": "100000",
            "stationary_sample_begin": "0",
            "stationary_sample_count": "5",
            "stationary_duration_us": "100000",
            "median_stability_limit": "0.02",
            "median_relative_drift": "0",
            "timing_converged": "1",
            "timing_stop_reason": "stationary_window",
            "min_us": str(summary.minimum),
            "median_us": str(summary.median),
            "p95_us": str(summary.p95),
            "mad_us": str(summary.mad),
            "cv": str(summary.cv),
            "timing_sample_digest": summary.digest,
        })
        context = replace(
            self.context(),
            profile=MeasurementProfile.PRODUCTION,
            raw_timing_sidecar_retained=True,
        )

        observation = adapt_cpu_prefill_row(row, context, timings)

        self.assertTrue(observation_has_promotion_timing(observation))
        below_floor = dict(row)
        below_floor["timing_budget_us"] = "99999"
        with self.assertRaisesRegex(
            ValueError,
            "installable CPU prefill adaptive timing evidence is incomplete",
        ):
            adapt_cpu_prefill_row(below_floor, context, timings)

    def test_legacy_sample_floor_evidence_remains_valid_as_source(self) -> None:
        """A content-addressed v13 corpus remains usable as expansion source."""

        timings = (1000.0,) * 30
        summary = summarize_sorted_timing(sorted(timings))
        row = self.row()
        row.update({
            "preconditioning_policy": "source-complete-round-v1",
            "preconditioning_m": "64",
            "preconditioning_budget_us": "250000",
            "preconditioning_duration_us": "260000",
            "warmup_count": "252",
            "warmup_round_count": "250",
            "warmup_budget_policy": "source-fixed-v1",
            "warmup_probe_latency_us": "1000",
            "warmup_budget_floor_us": "250000",
            "warmup_latency_multiplier": "0",
            "warmup_budget_ceiling_us": "0",
            "warmup_budget_us": "250000",
            "warmup_duration_us": "260000",
            "global_warmup_round_count": "250",
            "warmup_wall_duration_us": "270000",
            "warmup_max_round_duration_us": "1100",
            "warmup_round_timeout_us": "30000000",
            "measurement_coordination": "mpi-complete-round-v1",
            "mpi_world_size": "2",
            "mpi_rank": "0",
            "global_timing_round_count": str(len(timings)),
            "sample_count": str(len(timings)),
            "timing_protocol": LEGACY_ADAPTIVE_TIMING_PROTOCOL,
            "timing_ceiling_policy": LEGACY_ADAPTIVE_TIMING_CEILING_POLICY,
            "min_sample_count": "5",
            "stable_sample_count": "30",
            "max_sample_count": "180",
            "timing_budget_us": "250000",
            "timed_duration_us": str(sum(timings)),
            "median_stability_limit": "0.02",
            "median_relative_drift": "0",
            "timing_converged": "1",
            "timing_stop_reason": "stable_sample_floor",
            "min_us": str(summary.minimum),
            "median_us": str(summary.median),
            "p95_us": str(summary.p95),
            "mad_us": str(summary.mad),
            "cv": str(summary.cv),
            "timing_sample_digest": summary.digest,
        })
        context = replace(
            self.context(),
            profile=MeasurementProfile.PRODUCTION,
            raw_timing_sidecar_retained=True,
        )

        observation = adapt_cpu_prefill_row(row, context, timings)

        self.assertEqual(observation.sample_count, 30)
        self.assertTrue(observation.bitwise_equal)
        self.assertTrue(observation_has_promotion_timing(observation))

    def test_candidate_expansion_short_budget_requires_anchor_normalization(
        self,
    ) -> None:
        row, timings = self.candidate_expansion_row_and_timings()
        ordinary_context = replace(
            self.context(),
            profile=MeasurementProfile.PRODUCTION,
            raw_timing_sidecar_retained=True,
        )
        with self.assertRaisesRegex(
            ValueError,
            "warmup budget policy disagrees",
        ):
            adapt_cpu_prefill_row(row, ordinary_context, timings)

        expansion_context = replace(
            ordinary_context,
            anchored_candidate_expansion=True,
        )
        raw_observation = adapt_cpu_prefill_row(
            row,
            expansion_context,
            timings,
        )
        self.assertFalse(observation_has_promotion_timing(raw_observation))

        normalized = replace(
            raw_observation,
            adaptive_timing_evidence={
                **raw_observation.adaptive_timing_evidence,
                CANDIDATE_EXPANSION_EVIDENCE_KEY: {
                    "schema_version": CANDIDATE_EXPANSION_NORMALIZATION_SCHEMA,
                    "plan_digest": "sha256:" + "2" * 64,
                    "status": "scaled_to_immutable_source_anchor",
                    "anchor_candidate_id": (
                        "cpu.nvnni.prefill.row_chunk_grid.full_k"
                    ),
                    "source_anchor_median_us_hex": (25000.0).hex(),
                    "expansion_anchor_median_us_hex": (50000.0).hex(),
                    "scale_hex": (0.5).hex(),
                },
            },
        )
        normalized.validate()
        self.assertTrue(observation_has_promotion_timing(normalized))

        malformed = replace(
            normalized,
            adaptive_timing_evidence={
                **normalized.adaptive_timing_evidence,
                CANDIDATE_EXPANSION_EVIDENCE_KEY: {
                    **normalized.adaptive_timing_evidence[
                        CANDIDATE_EXPANSION_EVIDENCE_KEY
                    ],
                    "scale_hex": (0.75).hex(),
                },
            },
        )
        malformed.validate()
        self.assertFalse(observation_has_promotion_timing(malformed))

    def test_anchored_expansion_uses_parity_launches_before_stationary_timing(
        self,
    ) -> None:
        """A normalized cohort need not discard a separate warmup phase."""

        row, timings = self.candidate_expansion_row_and_timings()
        row.update({
            "preconditioning_policy": "disabled",
            "preconditioning_m": "0",
            "preconditioning_budget_us": "0",
            "preconditioning_duration_us": "0",
            "warmup_count": "2",
            "warmup_round_count": "0",
            "warmup_budget_policy": "disabled",
            "warmup_budget_floor_us": "0",
            "warmup_latency_multiplier": "0",
            "warmup_budget_ceiling_us": "0",
            "warmup_budget_us": "0",
            "warmup_duration_us": "0",
            "global_warmup_round_count": "0",
            "warmup_wall_duration_us": "0",
            "warmup_max_round_duration_us": "0",
        })
        context = replace(
            self.context(),
            profile=MeasurementProfile.PRODUCTION,
            raw_timing_sidecar_retained=True,
            anchored_candidate_expansion=True,
        )

        raw_observation = adapt_cpu_prefill_row(row, context, timings)
        normalized = replace(
            raw_observation,
            adaptive_timing_evidence={
                **raw_observation.adaptive_timing_evidence,
                CANDIDATE_EXPANSION_EVIDENCE_KEY: {
                    "schema_version": CANDIDATE_EXPANSION_NORMALIZATION_SCHEMA,
                    "plan_digest": "sha256:" + "3" * 64,
                    "status": "scaled_to_immutable_source_anchor",
                    "anchor_candidate_id": (
                        "cpu.nvnni.prefill.row_chunk_grid.full_k"
                    ),
                    "source_anchor_median_us_hex": (25000.0).hex(),
                    "expansion_anchor_median_us_hex": (50000.0).hex(),
                    "scale_hex": (0.5).hex(),
                },
            },
        )

        normalized.validate()
        self.assertEqual(normalized.warmup_count, 2)
        self.assertTrue(observation_has_promotion_timing(normalized))

    def test_anchored_complete_rounds_defer_raw_drift_to_normalization(
        self,
    ) -> None:
        """Common shuffled epochs need one anchor proof, not lucky raw tails."""

        row, _ = self.candidate_expansion_row_and_timings()
        timings = (20000.0, 20000.0, 30000.0, 30000.0, 30000.0)
        summary = summarize_sorted_timing(sorted(timings))
        row.update({
            "timed_duration_us": str(sum(timings)),
            "stationary_duration_us": str(sum(timings)),
            "median_relative_drift": str(1.0 / 3.0),
            "timing_converged": "0",
            "timing_stop_reason": CANDIDATE_EXPANSION_TIMING_STOP_REASON,
            "min_us": str(summary.minimum),
            "median_us": str(summary.median),
            "p95_us": str(summary.p95),
            "mad_us": str(summary.mad),
            "cv": str(summary.cv),
            "timing_sample_digest": summary.digest,
        })
        production = replace(
            self.context(),
            profile=MeasurementProfile.PRODUCTION,
            raw_timing_sidecar_retained=True,
        )
        with self.assertRaisesRegex(
            ValueError,
            "anchored timing stop requires candidate expansion",
        ):
            adapt_cpu_prefill_row(row, production, timings)

        raw_observation = adapt_cpu_prefill_row(
            row,
            replace(production, anchored_candidate_expansion=True),
            timings,
        )
        self.assertFalse(observation_has_promotion_timing(raw_observation))
        normalized = replace(
            raw_observation,
            adaptive_timing_evidence={
                **raw_observation.adaptive_timing_evidence,
                CANDIDATE_EXPANSION_EVIDENCE_KEY: {
                    "schema_version": CANDIDATE_EXPANSION_NORMALIZATION_SCHEMA,
                    "plan_digest": "sha256:" + "4" * 64,
                    "status": "scaled_to_immutable_source_anchor",
                    "anchor_candidate_id": (
                        "cpu.nvnni.prefill.row_chunk_grid.full_k"
                    ),
                    "source_anchor_median_us_hex": (15000.0).hex(),
                    "expansion_anchor_median_us_hex": (30000.0).hex(),
                    "scale_hex": (0.5).hex(),
                },
            },
        )
        normalized.validate()
        self.assertTrue(observation_has_promotion_timing(normalized))

    def test_anchored_expansion_retains_bounded_rejection_sample_guard(
        self,
    ) -> None:
        """The shorter anchored ceiling remains explicit authenticated policy."""

        row, timings = self.candidate_expansion_row_and_timings()
        row["max_sample_count"] = "59"
        context = replace(
            self.context(),
            profile=MeasurementProfile.PRODUCTION,
            raw_timing_sidecar_retained=True,
            anchored_candidate_expansion=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "adaptive timing evidence is incomplete",
        ):
            adapt_cpu_prefill_row(row, context, timings)

    def test_transition_warmup_budget_is_derived_from_the_candidate_probe(
        self,
    ) -> None:
        row, timings = self.production_row_and_timings()
        row.update({
            "m": "256",
            "warmup_budget_policy": "transition-probe-scaled-v1",
            "warmup_probe_latency_us": "30000",
            "warmup_budget_floor_us": "100000",
            "warmup_latency_multiplier": "60",
            "warmup_budget_ceiling_us": "2000000",
            "warmup_budget_us": "1800000",
            "warmup_duration_us": "1900000",
            "warmup_wall_duration_us": "2000000",
            "warmup_max_round_duration_us": "700000",
        })
        context = replace(
            self.context(),
            profile=MeasurementProfile.PRODUCTION,
            raw_timing_sidecar_retained=True,
        )

        observation = adapt_cpu_prefill_row(row, context, timings)

        self.assertEqual(observation.m, 256)
        self.assertTrue(observation.supported)

    def test_transition_warmup_budget_cannot_disagree_with_probe_formula(
        self,
    ) -> None:
        row, timings = self.production_row_and_timings()
        row.update({
            "m": "256",
            "warmup_budget_policy": "transition-probe-scaled-v1",
            "warmup_probe_latency_us": "30000",
            "warmup_budget_floor_us": "100000",
            "warmup_latency_multiplier": "60",
            "warmup_budget_ceiling_us": "2000000",
            "warmup_budget_us": "1700000",
            "warmup_duration_us": "1900000",
        })
        context = replace(
            self.context(),
            profile=MeasurementProfile.PRODUCTION,
            raw_timing_sidecar_retained=True,
        )

        with self.assertRaisesRegex(ValueError, "transition warmup formula"):
            adapt_cpu_prefill_row(row, context, timings)

    def test_elapsed_timing_claim_must_match_acquisition_order_sidecar(self) -> None:
        row, timings = self.production_row_and_timings()
        row["median_relative_drift"] = "0"
        context = replace(
            self.context(),
            profile=MeasurementProfile.PRODUCTION,
            raw_timing_sidecar_retained=True,
        )

        with self.assertRaisesRegex(ValueError, "median drift disagrees"):
            adapt_cpu_prefill_row(row, context, timings)

    def test_installable_multi_rank_evidence_requires_complete_round_coordination(
        self,
    ) -> None:
        row, timings = self.production_row_and_timings()
        row["measurement_coordination"] = "process-local-complete-round-v1"
        context = replace(
            self.context(),
            profile=MeasurementProfile.PRODUCTION,
            raw_timing_sidecar_retained=True,
        )

        with self.assertRaisesRegex(ValueError, "lacks MPI coordination"):
            adapt_cpu_prefill_row(row, context, timings)

    def test_global_round_count_must_cover_every_candidate_sample(self) -> None:
        row = self.row()
        row["global_timing_round_count"] = "2"

        with self.assertRaisesRegex(ValueError, "global round provenance"):
            adapt_cpu_prefill_row(row, self.context())

    def test_warmup_round_watchdog_is_authenticated(self) -> None:
        row = self.row()
        row["warmup_max_round_duration_us"] = row["warmup_round_timeout_us"]

        with self.assertRaisesRegex(ValueError, "round watchdog"):
            adapt_cpu_prefill_row(row, self.context())

    def test_long_complete_round_uses_probe_derived_watchdog(self) -> None:
        row = self.row()
        row.update({
            "complete_round_probe_duration_us": "10000000",
            "warmup_round_timeout_us": "41000000",
        })

        observation = adapt_cpu_prefill_row(row, self.context())
        self.assertTrue(observation.supported)

        row["warmup_round_timeout_us"] = "41000001"
        with self.assertRaisesRegex(ValueError, "disagrees with probes"):
            adapt_cpu_prefill_row(row, self.context())

    def test_unstable_cheap_row_cannot_stop_at_the_sample_guard_alone(self) -> None:
        row = self.row()
        row.update({
            "sample_count": "180",
            "global_timing_round_count": "180",
            "max_sample_count": "180",
            "timing_budget_us": "250000",
            "timed_duration_us": "200000",
            "stationary_sample_begin": "0",
            "stationary_sample_count": "180",
            "stationary_duration_us": "200000",
            "median_relative_drift": "0.03",
            "timing_converged": "0",
            "timing_stop_reason": "hard_samples_and_four_elapsed",
        })

        with self.assertRaisesRegex(ValueError, "recovery ceiling lacks both guards"):
            adapt_cpu_prefill_row(row, self.context())

    def test_installable_hard_ceiling_row_must_still_be_stable(self) -> None:
        """Exhausting both ceilings cannot promote a statistically unstable row."""

        timings = (2000.0,) * 220 + (2060.0,) * 40
        stationary_begin = len(timings)
        stationary_duration_us = 0.0
        while (
            stationary_begin > 0
            and (
                len(timings) - stationary_begin < 30
                or stationary_duration_us < 110000.0
            )
        ):
            stationary_begin -= 1
            stationary_duration_us += timings[stationary_begin]
        stationary_timings = timings[stationary_begin:]
        summary = summarize_sorted_timing(sorted(stationary_timings))
        relative_drift = 60.0 / 2060.0
        row = self.row()
        row.update({
            "sample_count": str(len(timings)),
            "global_timing_round_count": str(len(timings)),
            "max_sample_count": str(len(timings)),
            "timing_budget_us": "110000",
            "timed_duration_us": str(sum(timings)),
            "stationary_sample_begin": str(stationary_begin),
            "stationary_sample_count": str(len(stationary_timings)),
            "stationary_duration_us": str(stationary_duration_us),
            "median_relative_drift": str(relative_drift),
            "timing_converged": "0",
            "timing_stop_reason": "hard_samples_and_four_elapsed",
            "min_us": str(summary.minimum),
            "median_us": str(summary.median),
            "p95_us": str(summary.p95),
            "mad_us": str(summary.mad),
            "cv": str(summary.cv),
            "timing_sample_digest": summary.digest,
        })
        context = replace(
            self.context(),
            profile=MeasurementProfile.PARTIAL_PRODUCTION,
            raw_timing_sidecar_retained=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "installable CPU prefill adaptive timing evidence is incomplete",
        ):
            adapt_cpu_prefill_row(row, context, timings)

    def test_stationary_row_may_exceed_sample_floor_to_reach_elapsed_floor(self) -> None:
        row = self.row()
        row.update({
            "sample_count": "240",
            "global_timing_round_count": "240",
            "max_sample_count": "180",
            "timed_duration_us": "300000",
            "stationary_sample_begin": "0",
            "stationary_sample_count": "240",
            "stationary_duration_us": "300000",
            "median_relative_drift": "0.01",
            "timing_converged": "1",
            "timing_stop_reason": "stationary_window",
        })

        observation = adapt_cpu_prefill_row(row, self.context())

        self.assertEqual(observation.sample_count, 240)
        self.assertTrue(observation.supported)

    def test_corpus_rejects_independent_warmup_candidate_retirement(self) -> None:
        first = self.row()
        second = dict(first)
        candidate = "cpu.nvnni.prefill.two_row_tiles.nbc2.full_k"
        second.update({
            "candidate_id": candidate,
            "observed_candidate_id": candidate,
            "n_block_chunks": "2",
            "warmup_count": "3",
            "warmup_round_count": "1",
            "global_warmup_round_count": "1",
        })

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cpu-prefill.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(first))
                writer.writeheader()
                writer.writerows((first, second))

            with self.assertRaisesRegex(
                ValueError, "complete warmup-round provenance disagrees"
            ):
                adapt_cpu_prefill_csv((path,), self.context())

    def test_corpus_accepts_independent_stationary_candidate_retirement(
        self,
    ) -> None:
        """Ordinary candidates freeze independently within shared global rounds."""

        first = self.row()
        first.update({
            "global_timing_round_count": "5",
            "max_sample_count": "180",
        })
        second = dict(first)
        candidate = "cpu.nvnni.prefill.two_row_tiles.nbc2.full_k"
        second.update({
            "candidate_id": candidate,
            "observed_candidate_id": candidate,
            "n_block_chunks": "2",
            "sample_count": "5",
            "timed_duration_us": "50",
            "stationary_sample_count": "5",
            "stationary_duration_us": "50",
        })

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cpu-prefill-independent-timing.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(first))
                writer.writeheader()
                writer.writerows((first, second))

            corpus = adapt_cpu_prefill_csv((path,), self.context())

        self.assertEqual(len(corpus), 2)
        self.assertEqual({row.sample_count for row in corpus}, {3, 5})

    def test_anchored_corpus_still_requires_matching_complete_rounds(self) -> None:
        first = self.row()
        first.update({
            "min_sample_count": "2",
            "stable_sample_count": "2",
        })
        second = dict(first)
        candidate = "cpu.nvnni.prefill.two_row_tiles.nbc2.full_k"
        second.update({
            "candidate_id": candidate,
            "observed_candidate_id": candidate,
            "n_block_chunks": "2",
            "sample_count": "2",
            "timed_duration_us": "20",
            "stationary_sample_count": "2",
            "stationary_duration_us": "20",
        })
        context = replace(
            self.context(),
            anchored_candidate_expansion=True,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cpu-prefill-anchored-rounds.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(first))
                writer.writeheader()
                writer.writerows((first, second))

            with self.assertRaisesRegex(
                ValueError, "did not share the final round"
            ):
                adapt_cpu_prefill_csv((path,), context)

    def test_corpus_accepts_append_only_m_extension_preconditioning(self) -> None:
        """Historical and extension jobs may precondition the same shape anew."""

        first, _ = self.production_row_and_timings()
        second = dict(first)
        second.update({
            "m": "256",
            "preconditioning_m": "256",
            "preconditioning_budget_us": "250000",
            "preconditioning_duration_us": "260000",
            "warmup_probe_latency_us": "100000",
            "warmup_budget_floor_us": "250000",
            "warmup_budget_us": "250000",
            "warmup_duration_us": "260000",
            "warmup_wall_duration_us": "270000",
            "warmup_max_round_duration_us": "100000",
            "timing_order_seed": "54321",
        })

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cpu-prefill-extension.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(first))
                writer.writeheader()
                writer.writerows((first, second))

            corpus = adapt_cpu_prefill_csv((path,), self.context())

        self.assertEqual(len(corpus), 2)
        self.assertEqual({row.m for row in corpus}, {64, 256})


if __name__ == "__main__":
    unittest.main()
