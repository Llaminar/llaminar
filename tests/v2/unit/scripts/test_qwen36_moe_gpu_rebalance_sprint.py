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
        rebalance_trace: bool = False,
        capture_collectives: bool = False,
        no_capture_collectives: bool = False,
        defer_captured_collective_sync: bool = False,
        dense_tp: bool = False,
        dense_decode_replicated: bool = False,
        dense_policy: str | None = None,
        assignment_policy: str | None = None,
        allreduce_precision: str | None = None,
        allreduce_fp16_min_elements: str | None = None,
        small_gpu_allreduce: bool = False,
        small_gpu_allreduce_max_elements: str | None = None,
        no_require_prefill_graph: bool = False,
        n_predict_list: str | None = None,
        seeds: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env.pop("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", None)
            env.pop("LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES", None)
            env.pop("LLAMINAR_GPU_GRAPH_DEFER_CAPTURED_COLLECTIVE_FINAL_SYNC", None)
            env.pop("LLAMINAR_PREFILL_GRAPH_REQUIRED", None)
            env.pop("LLAMINAR_MOE_REBALANCE_TRACE_JSONL", None)
            env.pop("LLAMINAR_MOE_DEVICE_REBALANCE_LOAD_STATS", None)
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
                n_predict_list if n_predict_list is not None else "16",
            ]
            if n_predict_list is not None and "," in n_predict_list:
                args[-2] = "--n-predict-list"
            if seeds is not None:
                args.extend(["--seeds", seeds])
            if rebalance_window is not None:
                args.extend(["--rebalance-window", str(rebalance_window)])
            if perfstats:
                args.append("--perfstats")
            if stage_gpu_stats:
                args.append("--stage-gpu-stats")
            if rebalance_trace:
                args.append("--rebalance-trace")
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
            if assignment_policy is not None:
                args.extend(["--assignment-policy", assignment_policy])
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
            if no_require_prefill_graph:
                args.append("--no-require-prefill-graph")

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
        self.assertIn("LLAMINAR_PREFILL_GRAPH_REQUIRED=1", result.stdout)
        self.assertNotIn("assignment=least_loaded_ep", result.stdout)

    def test_twocard_dry_run_can_request_llep_assignment_policy(self) -> None:
        result = self.run_script(assignment_policy="least_loaded_ep")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("compute=apportioned_experts", result.stdout)
        self.assertIn("assignment=least_loaded_ep", result.stdout)
        self.assertIn("owner=0", result.stdout)

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

    def test_rebalance_trace_dry_run_writes_trace_sidecar_without_perfstats(self) -> None:
        result = self.run_script(rebalance_trace=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LLAMINAR_MOE_REBALANCE_TRACE_JSONL=", result.stdout)
        self.assertIn("LLAMINAR_MOE_DEVICE_REBALANCE_LOAD_STATS=1", result.stdout)
        self.assertIn("rebalance_trace.jsonl", result.stdout)
        self.assertNotIn("LLAMINAR_PERF_STATS_JSON=", result.stdout)

    def test_rebalance_trace_respects_explicit_load_stats_override(self) -> None:
        result = self.run_script(
            rebalance_trace=True,
            extra_env={"LLAMINAR_MOE_DEVICE_REBALANCE_LOAD_STATS": "0"},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LLAMINAR_MOE_REBALANCE_TRACE_JSONL=", result.stdout)
        self.assertIn("LLAMINAR_MOE_DEVICE_REBALANCE_LOAD_STATS=0", result.stdout)
        self.assertNotIn("LLAMINAR_MOE_DEVICE_REBALANCE_LOAD_STATS=1", result.stdout)

    def test_decode_length_seed_matrix_expands_trace_runs(self) -> None:
        result = self.run_script(
            rebalance_trace=True,
            n_predict_list="512,1024",
            seeds="101,202",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("LLAMINAR_MOE_REBALANCE_TRACE_JSONL="), 4)
        self.assertEqual(result.stdout.count("--seed 101"), 2)
        self.assertEqual(result.stdout.count("--seed 202"), 2)
        self.assertIn("/n_512/seed_101/", result.stdout)
        self.assertIn("/n_1024/seed_202/", result.stdout)

    def test_reusing_output_dir_appends_summary_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            common_args = [
                str(SCRIPT),
                "--dry-run",
                "--backend",
                "cuda",
                "--placement",
                "twocard",
                "--cases",
                "dynamic_hot10",
                "--bin",
                "/bin/true",
                "--model",
                str(Path(tmp) / "model.gguf"),
                "--out",
                str(out_dir),
                "--reps",
                "1",
                "--context-length",
                "1024",
                "--seeds",
                "101",
            ]
            first = subprocess.run(
                common_args + ["--n-predict", "512"],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            second = subprocess.run(
                common_args + ["--n-predict", "1024"],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            summary = (out_dir / "summary.tsv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(summary), 3)
            self.assertTrue(summary[0].startswith("backend\tplacement\tcase"))
            self.assertIn("\t512\t101\t", summary[1])
            self.assertIn("\t1024\t101\t", summary[2])

    def test_single_card_dry_run_does_not_force_segmented_collective_graphs(self) -> None:
        result = self.run_script(placement="single")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=1", result.stdout)
        self.assertNotIn("LLAMINAR_PREFILL_GRAPH_REQUIRED=1", result.stdout)
        self.assertIn("-d cuda:0", result.stdout)

    def test_twocard_dry_run_can_opt_out_of_prefill_graph_requirement_for_diagnostics(self) -> None:
        result = self.run_script(no_require_prefill_graph=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LLAMINAR_PREFILL_GRAPH_REQUIRED=0", result.stdout)
        self.assertNotIn("LLAMINAR_PREFILL_GRAPH_REQUIRED=1", result.stdout)

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

    def test_device_rebalance_env_knobs_are_forwarded_to_benchmark(self) -> None:
        result = self.run_script(
            extra_env={
                "LLAMINAR_MOE_DEVICE_REBALANCE_COMPACT_PAYLOAD_SLOTS": "2",
                "LLAMINAR_MOE_DEVICE_REBALANCE_MAINTENANCE_SLACK_TOKENS": "4",
                "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_MAINTENANCE_PERIOD_TOKENS": "128",
                "LLAMINAR_MOE_DEVICE_REBALANCE_INITIAL_MAINTENANCE_PERIOD_TOKENS": "65",
                "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_WAVE_SPREAD_IMPROVEMENT_PER_PAYLOAD_SLOT": "4096",
                "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_ROUTER_SPREAD_IMPROVEMENT_PER_PAYLOAD_SLOT": "2048",
                "LLAMINAR_MOE_DEVICE_REBALANCE_MAX_POST_WAVE_LOAD_SPREAD_PERMILLE": "75",
            }
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "LLAMINAR_MOE_DEVICE_REBALANCE_COMPACT_PAYLOAD_SLOTS=2",
            result.stdout,
        )
        self.assertIn(
            "LLAMINAR_MOE_DEVICE_REBALANCE_MAINTENANCE_SLACK_TOKENS=4",
            result.stdout,
        )
        self.assertIn(
            "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_MAINTENANCE_PERIOD_TOKENS=128",
            result.stdout,
        )
        self.assertIn(
            "LLAMINAR_MOE_DEVICE_REBALANCE_INITIAL_MAINTENANCE_PERIOD_TOKENS=65",
            result.stdout,
        )
        self.assertIn(
            "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_WAVE_SPREAD_IMPROVEMENT_PER_PAYLOAD_SLOT=4096",
            result.stdout,
        )
        self.assertIn(
            "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_ROUTER_SPREAD_IMPROVEMENT_PER_PAYLOAD_SLOT=2048",
            result.stdout,
        )
        self.assertIn(
            "LLAMINAR_MOE_DEVICE_REBALANCE_MAX_POST_WAVE_LOAD_SPREAD_PERMILLE=75",
            result.stdout,
        )

    def test_dynamic_policy_env_knobs_are_forwarded_to_benchmark(self) -> None:
        result = self.run_script(
            cases="dynamic",
            extra_env={
                "LLAMINAR_MOE_DYNAMIC_IMBALANCE_THRESHOLD_PERMILLE": "1100",
                "LLAMINAR_MOE_DYNAMIC_MIN_IMPROVEMENT_PERMILLE": "0",
                "LLAMINAR_MOE_DYNAMIC_MAX_SWAPS_PER_LAYER": "8",
                "LLAMINAR_MOE_DYNAMIC_MAX_PLAN_ENTRIES_PER_WAVE": "32",
                "LLAMINAR_MOE_DYNAMIC_MIN_WINDOW_ACTIVATIONS": "16",
            },
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "LLAMINAR_MOE_DYNAMIC_IMBALANCE_THRESHOLD_PERMILLE=1100",
            result.stdout,
        )
        self.assertIn(
            "LLAMINAR_MOE_DYNAMIC_MIN_IMPROVEMENT_PERMILLE=0",
            result.stdout,
        )
        self.assertIn(
            "LLAMINAR_MOE_DYNAMIC_MAX_SWAPS_PER_LAYER=8",
            result.stdout,
        )
        self.assertIn(
            "LLAMINAR_MOE_DYNAMIC_MAX_PLAN_ENTRIES_PER_WAVE=32",
            result.stdout,
        )
        self.assertIn(
            "LLAMINAR_MOE_DYNAMIC_MIN_WINDOW_ACTIVATIONS=16",
            result.stdout,
        )

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
