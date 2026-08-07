#!/usr/bin/env python3
"""Regressions for resumable production-MoE prefill sweep cells."""

from __future__ import annotations

import csv
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
sys.path.insert(0, str(KERNEL_ROOT))

from native_vnni_dispatch.adapters.evidence import (  # noqa: E402
    summarize_sorted_timing,
)
from native_vnni_dispatch.prefill_matrix import (  # noqa: E402
    GPU_PREFILL_M_BUCKETS,
)
from native_vnni_dispatch.moe_routing_profiles import (  # noqa: E402
    MOE_ROUTING_PROFILES,
    make_moe_routing_indices,
    summarize_moe_routing_profile,
)
from native_vnni_dispatch.production_moe_prefill_sweep import (  # noqa: E402
    AGGREGATE_COLUMNS,
    CUDA_CANDIDATE_IDS,
    PAIR_CANDIDATE_IDS,
    TIMING_COLUMNS,
    ProductionMoEPrefillCell,
    production_moe_prefill_candidate_ids,
    production_moe_prefill_cell_paths,
    production_moe_prefill_cells,
    run_missing_accelerator_cells,
    validate_production_moe_prefill_cell,
)
from native_vnni_dispatch.moe_production_overlay import (  # noqa: E402
    MoECandidateSurface,
    MoEProductionOverlayKey,
    discover_additive_moe_production_cells,
    generate_moe_production_overlay_include,
    require_moe_overlay_surface_totality,
    select_moe_production_overlay_winners,
)


_PAIR_PATTERN = re.compile(
    r"g_tm(?P<gate_m>\d+)_tn(?P<gate_n>\d+)"
    r"__d_tm(?P<down_m>\d+)_tn(?P<down_n>\d+)"
)
_CUDA_PATTERN = re.compile(r"g_tn(?P<gate_n>\d+)__d_fixed")


class NativeVNNIProductionMoEPrefillSweepTest(unittest.TestCase):
    """Lock total planning and strict atomic-cell authentication."""

    def _write_valid_cell(
        self,
        root: Path,
        cell: ProductionMoEPrefillCell,
    ) -> tuple[Path, Path]:
        aggregate_path = root / "cell.csv"
        timing_path = root / "cell.timing.csv"
        codebooks = cell.case.routed.runtime_codebooks(cell.backend)
        route_stats = summarize_moe_routing_profile(
            cell.route_profile,
            cell.m,
            cell.case.experts_per_token,
            cell.case.expert_count,
        )
        with (
            aggregate_path.open("w", newline="", encoding="utf-8") as aggregate,
            timing_path.open("w", newline="", encoding="utf-8") as timing,
        ):
            aggregate_writer = csv.DictWriter(
                aggregate, fieldnames=AGGREGATE_COLUMNS
            )
            timing_writer = csv.DictWriter(timing, fieldnames=TIMING_COLUMNS)
            aggregate_writer.writeheader()
            timing_writer.writeheader()
            candidate_ids = production_moe_prefill_candidate_ids(cell.backend)
            for candidate_index, candidate_id in enumerate(candidate_ids):
                pair_match = _PAIR_PATTERN.fullmatch(candidate_id)
                cuda_match = _CUDA_PATTERN.fullmatch(candidate_id)
                assert pair_match is not None or cuda_match is not None
                finalist = (
                    candidate_index < 12
                    if cell.backend == "rocm"
                    else True
                )
                screening = tuple(
                    1.0 + candidate_index * 0.001 + offset * 0.01
                    for offset in range(3)
                )
                robust = tuple(
                    0.95 + candidate_index * 0.001 + offset * 0.005
                    for offset in range(5)
                ) if finalist else ()
                selected = robust or screening
                summary = summarize_sorted_timing(selected)
                aggregate_writer.writerow({
                    "backend": cell.backend,
                    "phase": "moe_production_prefill",
                    "case_id": cell.case.evidence_id,
                    "route_profile": cell.route_profile,
                    "source_gate": cell.case.routed.gate,
                    "source_up": cell.case.routed.up,
                    "source_down": cell.case.routed.down,
                    "gate_execution_codebook": codebooks[0],
                    "up_execution_codebook": codebooks[1],
                    "down_execution_codebook": codebooks[2],
                    "hidden_size": cell.case.hidden_size,
                    "expert_width": cell.case.routed_expert_width,
                    "expert_count": cell.case.expert_count,
                    "top_k": cell.case.experts_per_token,
                    "route_active_experts": route_stats.active_experts,
                    "route_max_assignments": route_stats.maximum_assignments,
                    "route_assignment_cv": route_stats.assignment_cv,
                    "m": cell.m,
                    "candidate_id": candidate_id,
                    "gate_tile_m": pair_match["gate_m"] if pair_match else 0,
                    "gate_tile_n": (
                        pair_match["gate_n"]
                        if pair_match else cuda_match["gate_n"]
                    ),
                    "down_tile_m": pair_match["down_m"] if pair_match else 0,
                    "down_tile_n": (
                        pair_match["down_n"]
                        if pair_match else cell.case.experts_per_token * 32
                    ),
                    "gate_local_bytes": 0,
                    "down_local_bytes": 0,
                    "gate_registers": 48,
                    "down_registers": 56,
                    "gate_shared_bytes": 432,
                    "down_shared_bytes": 144,
                    "gate_active_blocks_per_sm": 5,
                    "down_active_blocks_per_sm": 16,
                    "screening_sample_count": len(screening),
                    "screening_replays": 2,
                    "robust_sample_count": len(robust),
                    "robust_replays": 4 if robust else 0,
                    "selected_sample_count": len(selected),
                    "selected_replays": 4 if robust else 2,
                    "min_us": summary.minimum * 1000.0,
                    "median_us": summary.median * 1000.0,
                    "p95_us": summary.p95 * 1000.0,
                    "mad_us": summary.mad * 1000.0,
                    "cv": summary.cv,
                    "timing_sample_digest": summary.digest,
                    "bit_mismatches": 0,
                    "first_bit_mismatch": (1 << 64) - 1,
                    "route_counter_ok": 1,
                    "finalist": int(finalist),
                    "is_winner": int(candidate_index == 0),
                })
                for phase, samples, replays in (
                    ("screening", screening, 2),
                    ("robust", robust, 4),
                ):
                    for sample_index, latency_ms in enumerate(samples):
                        timing_writer.writerow({
                            "backend": cell.backend,
                            "phase": "moe_production_prefill",
                            "case_id": cell.case.evidence_id,
                            "route_profile": cell.route_profile,
                            "m": cell.m,
                            "candidate_id": candidate_id,
                            "timing_phase": phase,
                            "sample_index": sample_index,
                            "timed_replays": replays,
                            "latency_us": f"{latency_ms * 1000.0:.9f}",
                            "latency_ms_hex": latency_ms.hex(),
                        })
        return aggregate_path, timing_path

    def test_every_backend_plan_is_routed_case_by_m_total(self) -> None:
        for backend in ("cuda", "rocm"):
            cells = production_moe_prefill_cells(backend)
            self.assertEqual(
                len(cells),
                49 * len(MOE_ROUTING_PROFILES) * len(GPU_PREFILL_M_BUCKETS),
            )
            self.assertEqual(len({cell.identity for cell in cells}), len(cells))
            self.assertEqual(len({cell.cell_id for cell in cells}), len(cells))
            self.assertEqual(
                {cell.m for cell in cells}, set(GPU_PREFILL_M_BUCKETS)
            )
            self.assertEqual(
                {cell.route_profile for cell in cells},
                set(MOE_ROUTING_PROFILES),
            )
            self.assertTrue(all(cell.case.evidence_id for cell in cells))

    def test_route_profiles_are_unique_per_row_and_change_load_shape(self) -> None:
        summaries = {}
        for profile in MOE_ROUTING_PROFILES:
            routes = make_moe_routing_indices(profile, 512, 8, 256)
            for row in range(512):
                row_routes = routes[row * 8:(row + 1) * 8]
                self.assertEqual(len(set(row_routes)), 8)
            summaries[profile] = summarize_moe_routing_profile(
                profile, 512, 8, 256
            )

        self.assertEqual(summaries["uniform"].assignment_cv, 0.0)
        self.assertEqual(summaries["uniform"].active_experts, 256)
        self.assertEqual(summaries["hotset"].active_experts, 32)
        self.assertGreater(
            summaries["hotset"].assignment_cv,
            summaries["power_law"].assignment_cv,
        )
        self.assertGreater(summaries["power_law"].assignment_cv, 0.0)

    def test_rocm_tournament_never_skips_serial_row_proof(self) -> None:
        source = (
            REPO_ROOT
            / "tests/v2/performance/kernels/moe/Perf__ROCmMoEVerifierPrefill.cpp"
        ).read_text(encoding="utf-8")
        self.assertNotIn("serial_proof_max_rows", source)
        self.assertNotIn("MOE_PRODUCTION_SWEEP_SERIAL_MAX_M", source)
        self.assertIn(
            "const std::vector<float> serial = runRowwiseDecode(", source
        )

    def test_cuda_tournament_never_skips_serial_row_proof(self) -> None:
        source = (
            REPO_ROOT
            / "tests/v2/performance/kernels/moe/Perf__MoEVerifierPrefill.cpp"
        ).read_text(encoding="utf-8")
        self.assertNotIn("serial_proof_max_rows", source)
        self.assertNotIn("MOE_PRODUCTION_SWEEP_SERIAL_MAX_M", source)
        self.assertIn(
            "const std::vector<float> serial = runCudaRowwiseDecode(", source
        )

    def test_validator_accepts_one_complete_candidate_pair_cell(self) -> None:
        cell = production_moe_prefill_cells("rocm")[0]
        with tempfile.TemporaryDirectory() as directory:
            aggregate, timing = self._write_valid_cell(Path(directory), cell)
            validation = validate_production_moe_prefill_cell(
                aggregate, timing, cell
            )

        self.assertEqual(validation.candidate_count, 169)
        self.assertEqual(validation.finalist_count, 12)
        self.assertEqual(validation.winner_id, PAIR_CANDIDATE_IDS[0])

    def test_validator_accepts_complete_cuda_fixed_arithmetic_cell(self) -> None:
        cell = production_moe_prefill_cells("cuda")[0]
        with tempfile.TemporaryDirectory() as directory:
            aggregate, timing = self._write_valid_cell(Path(directory), cell)
            validation = validate_production_moe_prefill_cell(
                aggregate, timing, cell
            )

        self.assertEqual(validation.candidate_count, 7)
        self.assertEqual(validation.finalist_count, 7)
        self.assertEqual(validation.winner_id, CUDA_CANDIDATE_IDS[0])

    def test_resume_skips_valid_cells_and_partitions_missing_cells_once(self) -> None:
        cells = production_moe_prefill_cells("cuda")[:5]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed = root / "seed"
            seed.mkdir()
            aggregate, timing = self._write_valid_cell(seed, cells[0])
            completed_paths = production_moe_prefill_cell_paths(root, cells[0])
            completed_paths.aggregate.parent.mkdir(parents=True)
            aggregate.replace(completed_paths.aggregate)
            timing.replace(completed_paths.timing)

            calls: list[tuple[str, int]] = []
            calls_lock = threading.Lock()
            first_wave = threading.Barrier(2)

            def record_cell(
                _binary: Path,
                _root: Path,
                cell: ProductionMoEPrefillCell,
                device: int,
            ) -> None:
                with calls_lock:
                    calls.append((cell.cell_id, device))
                    call_index = len(calls)
                if call_index <= 2:
                    first_wave.wait(timeout=5.0)

            with patch(
                "native_vnni_dispatch.production_moe_prefill_sweep."
                "_run_accelerator_cell",
                side_effect=record_cell,
            ):
                completed = run_missing_accelerator_cells(
                    Path(sys.executable), root, cells, devices=(0, 1)
                )

        self.assertCountEqual(completed, cells[1:])
        self.assertCountEqual(
            (cell_id for cell_id, _device in calls),
            (cell.cell_id for cell in cells[1:]),
        )
        self.assertEqual(len(calls), len({cell_id for cell_id, _ in calls}))
        self.assertEqual({device for _cell_id, device in calls}, {0, 1})

    def test_overlay_discovers_one_complete_additive_runtime_key(self) -> None:
        baseline = production_moe_prefill_cells("rocm")
        source = next(
            cell.case
            for cell in baseline
            if cell.case.evidence_id
            == "h2048_r512_e256_k8_gIQ2_S_uIQ2_S_dIQ3_S"
        )
        additive = tuple(
            ProductionMoEPrefillCell("rocm", source, 513, profile)
            for profile in MOE_ROUTING_PROFILES
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, cell in enumerate(additive):
                seed = root / f"seed-{index}"
                seed.mkdir()
                aggregate, timing = self._write_valid_cell(seed, cell)
                paths = production_moe_prefill_cell_paths(root, cell)
                paths.aggregate.parent.mkdir(parents=True, exist_ok=True)
                aggregate.replace(paths.aggregate)
                timing.replace(paths.timing)

            discovered = discover_additive_moe_production_cells(
                root,
                backend="rocm",
                baseline_cells=baseline,
            )

        self.assertEqual(set(discovered), set(additive))

    def test_overlay_rejects_half_published_additive_cell(self) -> None:
        baseline = production_moe_prefill_cells("rocm")
        source = next(
            cell.case
            for cell in baseline
            if cell.case.evidence_id
            == "h2048_r512_e256_k8_gIQ2_S_uIQ2_S_dIQ3_S"
        )
        additive = ProductionMoEPrefillCell("rocm", source, 513, "uniform")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed = root / "seed"
            seed.mkdir()
            aggregate, _timing = self._write_valid_cell(seed, additive)
            paths = production_moe_prefill_cell_paths(root, additive)
            paths.aggregate.parent.mkdir(parents=True)
            aggregate.replace(paths.aggregate)

            with self.assertRaisesRegex(
                ValueError, "has no raw timing sidecar"
            ):
                discover_additive_moe_production_cells(
                    root,
                    backend="rocm",
                    baseline_cells=baseline,
                )

    def test_overlay_rejects_partial_route_profile_selection(self) -> None:
        cells = production_moe_prefill_cells("rocm")
        selected = tuple(
            cell for cell in cells
            if cell.m == 512
            and cell.case.evidence_id
            == "h2048_r512_e256_k8_gIQ2_S_uIQ2_S_dIQ3_S"
            and cell.route_profile != "power_law"
        )
        with self.assertRaisesRegex(ValueError, "incomplete route/source surfaces"):
            require_moe_overlay_surface_totality(selected, backend="rocm")

    def test_validator_rejects_cuda_arithmetic_geometry_as_dispatch(self) -> None:
        cell = production_moe_prefill_cells("cuda")[0]
        with tempfile.TemporaryDirectory() as directory:
            aggregate, timing = self._write_valid_cell(Path(directory), cell)
            with aggregate.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["down_tile_n"] = "288"
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=AGGREGATE_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "fixed arithmetic geometry"):
                validate_production_moe_prefill_cell(aggregate, timing, cell)

    def test_validator_rejects_any_spilling_candidate(self) -> None:
        cell = production_moe_prefill_cells("rocm")[0]
        with tempfile.TemporaryDirectory() as directory:
            aggregate, timing = self._write_valid_cell(Path(directory), cell)
            with aggregate.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[17]["gate_local_bytes"] = "8"
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=AGGREGATE_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "spilling candidate"):
                validate_production_moe_prefill_cell(aggregate, timing, cell)

    def test_validator_rejects_truncated_raw_timing(self) -> None:
        cell = production_moe_prefill_cells("rocm")[0]
        with tempfile.TemporaryDirectory() as directory:
            aggregate, timing = self._write_valid_cell(Path(directory), cell)
            lines = timing.read_text(encoding="utf-8").splitlines()
            timing.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sample count mismatch"):
                validate_production_moe_prefill_cell(aggregate, timing, cell)

    def test_exact_overlay_selection_is_alias_robust(self) -> None:
        key = MoEProductionOverlayKey(13, 11, 2048, 512, 256, 8, 64)
        records = []
        for surface_index, surface_id in enumerate(("IQ2_S/IQ3_S", "alias")):
            for candidate_index, candidate_id in enumerate(PAIR_CANDIDATE_IDS):
                records.append((
                    MoECandidateSurface(
                        surface_id=surface_id,
                        candidate_id=candidate_id,
                        median_us=(
                            1000.0 + candidate_index * 4.0
                            + surface_index * candidate_index
                        ),
                        p95_us=1010.0 + candidate_index * 4.0,
                        cv=0.01,
                        finalist=candidate_index < 24,
                    ),
                    key,
                ))

        winners = select_moe_production_overlay_winners(
            records, expected_keys={key}
        )
        self.assertEqual(winners[key].candidate_id, PAIR_CANDIDATE_IDS[0])
        self.assertEqual(winners[key].surface_count, 2)
        generated = generate_moe_production_overlay_include(
            winners, corpus_digest="sha256:" + "1" * 64
        )
        self.assertIn("selectROCmMoEProductionPrefillOverlay", generated)
        self.assertIn("13, 11, 2048, 512, 256, 8, 64", generated)

    def test_exact_overlay_selection_minimizes_worst_route_profile_regret(
        self,
    ) -> None:
        key = MoEProductionOverlayKey(13, 11, 2048, 512, 256, 8, 512)
        records = []
        for profile in MOE_ROUTING_PROFILES:
            for candidate_index, candidate_id in enumerate(PAIR_CANDIDATE_IDS):
                median_us = 200.0 + candidate_index
                if candidate_index == 0:
                    median_us = 100.0 if profile == "uniform" else 125.0
                elif candidate_index == 1:
                    median_us = 125.0 if profile == "uniform" else 100.0
                elif candidate_index == 2:
                    median_us = 102.0
                records.append((
                    MoECandidateSurface(
                        surface_id=f"source/route={profile}",
                        candidate_id=candidate_id,
                        median_us=median_us,
                        p95_us=median_us + 1.0,
                        cv=0.01,
                        finalist=candidate_index < 24,
                    ),
                    key,
                ))

        winners = select_moe_production_overlay_winners(
            records, expected_keys={key}
        )
        self.assertEqual(winners[key].candidate_id, PAIR_CANDIDATE_IDS[2])
        self.assertEqual(winners[key].surface_count, len(MOE_ROUTING_PROFILES))
        self.assertLess(winners[key].maximum_surface_regret, 0.021)

    def test_exact_overlay_preserves_device_adaptive_rocm_candidate(self) -> None:
        """Keep the route-robust TM12/TM16 device planner installable."""

        key = MoEProductionOverlayKey(13, 11, 2048, 512, 256, 8, 512)
        adaptive = "g_tm20_tn128__d_tm20_tn128"
        self.assertIn(adaptive, PAIR_CANDIDATE_IDS)

        records = []
        for profile_index, profile in enumerate(MOE_ROUTING_PROFILES):
            local_specialist = PAIR_CANDIDATE_IDS[profile_index]
            for candidate_id in PAIR_CANDIDATE_IDS:
                median_us = 120.0
                if candidate_id == local_specialist:
                    median_us = 100.0
                if candidate_id == adaptive:
                    median_us = 101.0
                records.append((
                    MoECandidateSurface(
                        surface_id=f"source/route={profile}",
                        candidate_id=candidate_id,
                        median_us=median_us,
                        p95_us=median_us + 1.0,
                        cv=0.01,
                        finalist=True,
                    ),
                    key,
                ))

        winners = select_moe_production_overlay_winners(
            records, expected_keys={key}
        )
        winner = winners[key]
        self.assertEqual(winner.candidate_id, adaptive)
        self.assertEqual(
            (winner.gate_tile_m, winner.gate_tile_n),
            (20, 128),
        )
        self.assertEqual(
            (winner.down_tile_m, winner.down_tile_n),
            (20, 128),
        )
        generated = generate_moe_production_overlay_include(
            winners, corpus_digest="sha256:" + "3" * 64
        )
        self.assertIn(
            "{13, 11, 2048, 512, 256, 8, 512, {20, 128, 20, 128}}",
            generated,
        )

    def test_exact_overlay_rejects_unavoidable_cross_profile_regret(self) -> None:
        key = MoEProductionOverlayKey(13, 11, 2048, 512, 256, 8, 512)
        records = []
        for profile_index, profile in enumerate(MOE_ROUTING_PROFILES):
            for candidate_index, candidate_id in enumerate(PAIR_CANDIDATE_IDS):
                median_us = 200.0 + candidate_index
                if candidate_index == profile_index:
                    median_us = 100.0
                elif candidate_index < len(MOE_ROUTING_PROFILES):
                    median_us = 120.0
                records.append((
                    MoECandidateSurface(
                        surface_id=f"source/route={profile}",
                        candidate_id=candidate_id,
                        median_us=median_us,
                        p95_us=median_us + 1.0,
                        cv=0.01,
                        finalist=candidate_index < 24,
                    ),
                    key,
                ))

        with self.assertRaisesRegex(ValueError, "maximum surface regret"):
            select_moe_production_overlay_winners(
                records, expected_keys={key}
            )

    def test_every_backend_rejects_empty_overlay_generation(self) -> None:
        for backend in ("cuda", "rocm"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(
                    ValueError, "cannot generate an empty production MoE overlay"
                ):
                    generate_moe_production_overlay_include(
                        {},
                        backend=backend,
                        corpus_digest="sha256:" + "0" * 64,
                    )

    def test_cuda_exact_overlay_selects_only_arithmetic_neutral_geometry(
        self,
    ) -> None:
        key = MoEProductionOverlayKey(13, 11, 2048, 512, 256, 8, 16)
        records = []
        winning_index = 3
        for surface_index, surface_id in enumerate(("IQ2_S/IQ3_S", "alias")):
            for candidate_index, candidate_id in enumerate(CUDA_CANDIDATE_IDS):
                distance = abs(candidate_index - winning_index)
                records.append((
                    MoECandidateSurface(
                        surface_id=surface_id,
                        candidate_id=candidate_id,
                        median_us=1000.0 + distance * 10.0 + surface_index,
                        p95_us=1010.0 + distance * 10.0 + surface_index,
                        cv=0.01,
                        finalist=True,
                    ),
                    key,
                ))

        winners = select_moe_production_overlay_winners(
            records, backend="cuda", expected_keys={key}
        )
        winner = winners[key]
        self.assertEqual(winner.candidate_id, CUDA_CANDIDATE_IDS[winning_index])
        self.assertEqual(winner.gate_tile_n, 160)
        self.assertEqual(winner.gate_tile_m, 0)
        self.assertEqual(winner.down_tile_m, 0)
        self.assertEqual(winner.down_tile_n, 0)
        generated = generate_moe_production_overlay_include(
            winners,
            backend="cuda",
            corpus_digest="sha256:" + "2" * 64,
        )
        self.assertIn("selectCUDAMoEProductionPrefillOverlay", generated)
        self.assertIn("13, 11, 2048, 512, 256, 8, 16, 160", generated)

    def test_exact_overlay_rejects_no_cross_alias_finalist(self) -> None:
        key = MoEProductionOverlayKey(13, 11, 2048, 512, 256, 8, 64)
        records = []
        for surface_index, surface_id in enumerate(("one", "two")):
            for candidate_index, candidate_id in enumerate(PAIR_CANDIDATE_IDS):
                records.append((
                    MoECandidateSurface(
                        surface_id=surface_id,
                        candidate_id=candidate_id,
                        median_us=1000.0 + candidate_index,
                        p95_us=1010.0 + candidate_index,
                        cv=0.01,
                        finalist=(
                            candidate_index < 12 if surface_index == 0
                            else 12 <= candidate_index < 24
                        ),
                    ),
                    key,
                ))

        with self.assertRaisesRegex(ValueError, "no candidate was robustly retimed"):
            select_moe_production_overlay_winners(
                records, expected_keys={key}
            )


if __name__ == "__main__":
    unittest.main()
