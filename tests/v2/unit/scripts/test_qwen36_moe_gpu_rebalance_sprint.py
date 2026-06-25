#!/usr/bin/env python3
"""Regression tests for the Qwen3.6 MoE GPU rebalance sprint wrapper."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "run_qwen36_moe_gpu_rebalance_sprint.sh"


class Qwen36MoEGPURebalanceSprintTest(unittest.TestCase):
    def run_script(
        self,
        *,
        placement: str = "twocard",
        cases: str = "static",
        rebalance_window: int | None = None,
        perfstats: bool = False,
        stage_gpu_stats: bool = False,
        capture_collectives: bool = False,
        no_capture_collectives: bool = False,
        defer_captured_collective_sync: bool = False,
        dense_tp: bool = False,
        dense_decode_replicated: bool = False,
        dense_policy: str | None = None,
        allreduce_precision: str | None = None,
        allreduce_fp16_min_elements: str | None = None,
        small_gpu_allreduce: bool = False,
        small_gpu_allreduce_max_elements: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env.pop("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", None)
            env.pop("LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES", None)
            env.pop("LLAMINAR_GPU_GRAPH_DEFER_CAPTURED_COLLECTIVE_FINAL_SYNC", None)
            if extra_env:
                env.update(extra_env)

            args = [
                str(SCRIPT),
                "--dry-run",
                "--backend",
                "cuda",
                "--placement",
                placement,
                "--cases",
                cases,
                "--bin",
                "/bin/true",
                "--model",
                str(Path(tmp) / "model.gguf"),
                "--out",
                str(Path(tmp) / "out"),
                "--reps",
                "1",
                "--context-length",
                "1024",
                "--n-predict",
                "16",
            ]
            if rebalance_window is not None:
                args.extend(["--rebalance-window", str(rebalance_window)])
            if perfstats:
                args.append("--perfstats")
            if stage_gpu_stats:
                args.append("--stage-gpu-stats")
            if capture_collectives:
                args.append("--capture-collectives")
            if no_capture_collectives:
                args.append("--no-capture-collectives")
            if defer_captured_collective_sync:
                args.append("--defer-captured-collective-sync")
            if dense_tp:
                args.append("--dense-tp")
            if dense_decode_replicated:
                args.append("--dense-decode-replicated")
            if dense_policy is not None:
                args.extend(["--dense-policy", dense_policy])
            if allreduce_precision is not None:
                args.extend(["--allreduce-precision", allreduce_precision])
            if allreduce_fp16_min_elements is not None:
                args.extend([
                    "--allreduce-fp16-min-elements",
                    allreduce_fp16_min_elements,
                ])
            if small_gpu_allreduce:
                args.append("--small-gpu-allreduce")
            if small_gpu_allreduce_max_elements is not None:
                args.extend([
                    "--small-gpu-allreduce-max-elements",
                    small_gpu_allreduce_max_elements,
                ])

            return subprocess.run(
                args,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

    def test_twocard_dry_run_captures_collective_graphs_by_default(self) -> None:
        result = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES=1", result.stdout)
        self.assertIn("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=0", result.stdout)
        self.assertNotIn("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=1", result.stdout)
        self.assertNotIn("LLAMINAR_PERF_STATS_JSON=", result.stdout)
        self.assertNotIn("tp_allreduce_bom", result.stdout)
        self.assertIn("--moe-expert-overlay", result.stdout)

    def test_no_capture_collectives_dry_run_forces_segmented_collective_graphs(self) -> None:
        result = self.run_script(no_capture_collectives=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=1", result.stdout)
        self.assertNotIn("LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES=1", result.stdout)

    def test_perfstats_dry_run_includes_allreduce_bom_without_stage_gpu_timing(self) -> None:
        result = self.run_script(perfstats=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LLAMINAR_PERF_STATS_JSON=", result.stdout)
        self.assertIn("tp_allreduce_bom", result.stdout)
        self.assertIn("forward_graph", result.stdout)
        self.assertNotIn("stage_gpu", result.stdout)

    def test_stage_gpu_stats_dry_run_adds_heavy_timing_filter(self) -> None:
        result = self.run_script(stage_gpu_stats=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LLAMINAR_PERF_STATS_JSON=", result.stdout)
        self.assertIn("stage_gpu", result.stdout)

    def test_single_card_dry_run_does_not_force_segmented_collective_graphs(self) -> None:
        result = self.run_script(placement="single")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=1", result.stdout)
        self.assertIn("-d cuda:0", result.stdout)

    def test_capture_collectives_dry_run_uses_captured_collective_graph_policy(self) -> None:
        result = self.run_script(capture_collectives=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES=1", result.stdout)
        self.assertIn("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=0", result.stdout)
        self.assertNotIn("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=1", result.stdout)

    def test_defer_captured_collective_sync_dry_run_sets_decode_sync_env(self) -> None:
        result = self.run_script(
            capture_collectives=True,
            defer_captured_collective_sync=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES=1", result.stdout)
        self.assertIn(
            "LLAMINAR_GPU_GRAPH_DEFER_CAPTURED_COLLECTIVE_FINAL_SYNC=1",
            result.stdout,
        )

    def test_defer_captured_collective_sync_dry_run_does_not_affect_single_card_baseline(self) -> None:
        result = self.run_script(
            placement="single",
            capture_collectives=True,
            defer_captured_collective_sync=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(
            "LLAMINAR_GPU_GRAPH_DEFER_CAPTURED_COLLECTIVE_FINAL_SYNC=1",
            result.stdout,
        )

    def test_dynamic_cases_use_sprint_rebalance_window(self) -> None:
        result = self.run_script(cases="dynamic,dynamic_hot10")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("--moe-rebalance-window 64"), 2)

    def test_rebalance_window_override_is_used(self) -> None:
        result = self.run_script(cases="observe", rebalance_window=8)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--moe-rebalance-window 8", result.stdout)
        self.assertNotIn("--moe-rebalance-window 64", result.stdout)

    def test_existing_segmented_collective_setting_is_preserved(self) -> None:
        result = self.run_script(
            extra_env={"LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED": "0"}
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=1", result.stdout)

    def test_dense_tp_dry_run_opts_two_card_overlay_into_dense_sharding(self) -> None:
        result = self.run_script(dense_tp=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "--moe-expert-overlay-dense-policy tensor-parallel",
            result.stdout,
        )

    def test_dense_tp_dry_run_does_not_affect_single_card_baseline(self) -> None:
        result = self.run_script(placement="single", dense_tp=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--moe-expert-overlay-dense-policy", result.stdout)

    def test_dense_decode_replicated_dry_run_opts_two_card_overlay_into_decode_full_dense(self) -> None:
        result = self.run_script(dense_tp=True, dense_decode_replicated=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "--moe-expert-overlay-dense-policy phase-split-hybrid-tp-ae",
            result.stdout,
        )

    def test_dense_decode_replicated_dry_run_does_not_affect_single_card_baseline(self) -> None:
        result = self.run_script(
            placement="single",
            dense_tp=True,
            dense_decode_replicated=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--moe-expert-overlay-dense-policy", result.stdout)

    def test_dense_policy_dry_run_overrides_legacy_dense_flags_for_twocard_overlay(self) -> None:
        result = self.run_script(
            dense_tp=True,
            dense_decode_replicated=True,
            dense_policy="tensor-parallel-decode-mirrored-embedding",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "--moe-expert-overlay-dense-policy tensor-parallel-decode-mirrored-embedding",
            result.stdout,
        )
        self.assertNotIn(
            "--moe-expert-overlay-dense-policy phase-split-hybrid-tp-ae",
            result.stdout,
        )

    def test_dense_policy_dry_run_does_not_affect_single_card_baseline(self) -> None:
        result = self.run_script(
            placement="single",
            dense_policy="tensor-parallel-decode-mirrored-embedding",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--moe-expert-overlay-dense-policy", result.stdout)

    def test_allreduce_precision_dry_run_sets_collective_precision_flag(self) -> None:
        result = self.run_script(allreduce_precision="fp16")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--tp-allreduce-precision fp16", result.stdout)
        self.assertNotIn("LLAMINAR_ALLREDUCE_PRECISION=fp16", result.stdout)

    def test_allreduce_fp16_min_elements_dry_run_sets_threshold_env(self) -> None:
        result = self.run_script(
            allreduce_precision="fp16",
            allreduce_fp16_min_elements="8192",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--tp-allreduce-precision fp16", result.stdout)
        self.assertIn("LLAMINAR_ALLREDUCE_FP16_MIN_ELEMENTS=8192", result.stdout)

    def test_invalid_allreduce_precision_is_rejected(self) -> None:
        result = self.run_script(allreduce_precision="int4")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--allreduce-precision must be fp16, fp32, or bf16", result.stderr)

    def test_invalid_allreduce_fp16_min_elements_is_rejected(self) -> None:
        result = self.run_script(allreduce_fp16_min_elements="-1")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--allreduce-fp16-min-elements must be a non-negative integer", result.stderr)

    def test_small_gpu_allreduce_dry_run_sets_twocard_env(self) -> None:
        result = self.run_script(
            small_gpu_allreduce=True,
            small_gpu_allreduce_max_elements="4096",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LLAMINAR_LOCALTP_SMALL_GPU_ALLREDUCE=1", result.stdout)
        self.assertIn(
            "LLAMINAR_LOCALTP_SMALL_GPU_ALLREDUCE_MAX_ELEMENTS=4096",
            result.stdout,
        )

    def test_small_gpu_allreduce_dry_run_does_not_affect_single_card_baseline(self) -> None:
        result = self.run_script(
            placement="single",
            small_gpu_allreduce=True,
            small_gpu_allreduce_max_elements="4096",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("LLAMINAR_LOCALTP_SMALL_GPU_ALLREDUCE=1", result.stdout)
        self.assertNotIn(
            "LLAMINAR_LOCALTP_SMALL_GPU_ALLREDUCE_MAX_ELEMENTS=4096",
            result.stdout,
        )

    def test_invalid_small_gpu_allreduce_max_elements_is_rejected(self) -> None:
        result = self.run_script(small_gpu_allreduce_max_elements="-1")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "--small-gpu-allreduce-max-elements must be a non-negative integer",
            result.stderr,
        )


if __name__ == "__main__":
    unittest.main()
