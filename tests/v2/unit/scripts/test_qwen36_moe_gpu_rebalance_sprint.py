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
        decode_assignment_policy: str | None = None,
        prefill_assignment_policy: str | None = None,
        allreduce_precision: str | None = None,
        allreduce_fp16_min_elements: str | None = None,
        no_require_prefill_graph: bool = False,
        n_predict_list: str | None = None,
        seeds: str | None = None,
        prompt: str | None = None,
        prompt_file: str | None = None,
        maintenance_slack_tokens: int | None = None,
        minimum_maintenance_period_tokens: int | None = None,
        initial_maintenance_period_tokens: int | None = None,
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
            env.pop("LLAMINAR_GPU_MOE_REBALANCE_PROMPT", None)
            env.pop("LLAMINAR_GPU_MOE_REBALANCE_PROMPT_FILE", None)
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
            if prompt is not None:
                args.extend(["--prompt", prompt])
            if prompt_file is not None:
                args.extend(["--prompt-file", prompt_file])
            if rebalance_window is not None:
                args.extend(["--rebalance-window", str(rebalance_window)])
            if maintenance_slack_tokens is not None:
                args.extend([
                    "--maintenance-slack-tokens",
                    str(maintenance_slack_tokens),
                ])
            if minimum_maintenance_period_tokens is not None:
                args.extend([
                    "--minimum-maintenance-period-tokens",
                    str(minimum_maintenance_period_tokens),
                ])
            if initial_maintenance_period_tokens is not None:
                args.extend([
                    "--initial-maintenance-period-tokens",
                    str(initial_maintenance_period_tokens),
                ])
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
            if decode_assignment_policy is not None:
                args.extend([
                    "--routed-decode-assignment-policy",
                    decode_assignment_policy,
                ])
            if prefill_assignment_policy is not None:
                args.extend([
                    "--routed-prefill-assignment-policy",
                    prefill_assignment_policy,
                ])
            if allreduce_precision is not None:
                args.extend(["--allreduce-precision", allreduce_precision])
            if allreduce_fp16_min_elements is not None:
                args.extend([
                    "--allreduce-fp16-min-elements",
                    allreduce_fp16_min_elements,
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
        self.assertIn("--moe-routed-expert-placement", result.stdout)
        self.assertIn("LLAMINAR_PREFILL_GRAPH_REQUIRED=1", result.stdout)
        self.assertIn("routed_compute=apportioned", result.stdout)
        self.assertIn("routed_phase=uniform", result.stdout)
        self.assertIn("routed_decode_assignment=static-owner", result.stdout)
        self.assertIn("routed_prefill_assignment=static-owner", result.stdout)
        self.assertIn("priority=0", result.stdout)
        self.assertNotIn("fallback=", result.stdout)

    def test_twocard_dry_run_can_request_llep_assignment_policy(self) -> None:
        result = self.run_script(
            prefill_assignment_policy="least-loaded-resident"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("routed_compute=apportioned", result.stdout)
        self.assertIn("routed_phase=uniform", result.stdout)
        self.assertIn("routed_decode_assignment=static-owner", result.stdout)
        self.assertIn(
            "routed_prefill_assignment=least-loaded-resident",
            result.stdout,
        )
        self.assertIn("owner=0", result.stdout)

    def test_llep_declares_apportioned_least_loaded_policy(self) -> None:
        result = self.run_script(cases="llep")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--moe-residency-maintenance off", result.stdout)
        self.assertIn("routed_compute=apportioned", result.stdout)
        self.assertIn("routed_phase=uniform", result.stdout)
        self.assertIn("routed_decode_assignment=static-owner", result.stdout)
        self.assertIn(
            "routed_prefill_assignment=least-loaded-resident",
            result.stdout,
        )
        self.assertNotIn("routed_compute=replicated", result.stdout)
        self.assertIn("--moe-continuation-dense-policy tensor-parallel", result.stdout)
        self.assertIn("--mtp-terminal-head-policy mirrored-full-vocabulary", result.stdout)
        self.assertNotIn("--moe-device-rebalance-maintenance-slack-tokens", result.stdout)

    def test_replicated_experts_decode_llep_is_an_explicit_distinct_case(self) -> None:
        result = self.run_script(cases="llep_replicated_experts_decode")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--moe-residency-maintenance off", result.stdout)
        self.assertIn("routed_compute=replicated", result.stdout)
        self.assertIn(
            "routed_phase=prefill-apportioned-decode-replicated",
            result.stdout,
        )
        self.assertIn("routed_decode_assignment=static-owner", result.stdout)
        self.assertIn("routed_prefill_assignment=least-loaded-resident", result.stdout)

    def test_obsolete_ambiguous_replicated_decode_case_is_rejected(self) -> None:
        result = self.run_script(cases="llep_replicated_decode")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown case 'llep_replicated_decode'", result.stderr)

    def test_dynamic_policy_can_override_every_maintenance_cadence_axis(self) -> None:
        result = self.run_script(
            cases="dynamic",
            rebalance_window=32,
            maintenance_slack_tokens=2,
            minimum_maintenance_period_tokens=96,
            initial_maintenance_period_tokens=48,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "--moe-device-rebalance-maintenance-slack-tokens 2",
            result.stdout,
        )
        self.assertIn(
            "--moe-device-rebalance-min-maintenance-period-tokens 96",
            result.stdout,
        )
        self.assertIn(
            "--moe-device-rebalance-initial-maintenance-period-tokens 48",
            result.stdout,
        )

    def test_llep_rejects_conflicting_static_owner_assignment(self) -> None:
        result = self.run_script(
            cases="llep",
            prefill_assignment_policy="static-owner",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "LLEP requires routed assignment policy least-loaded-resident",
            result.stderr,
        )

    def test_no_capture_collectives_is_rejected_for_homogeneous_twocard(self) -> None:
        result = self.run_script(no_capture_collectives=True)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "homogeneous two-card runs require graph-captured collectives",
            result.stderr,
        )
        self.assertNotIn("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=1", result.stdout)

    def test_perfstats_dry_run_includes_allreduce_bom_without_stage_gpu_timing(self) -> None:
        result = self.run_script(perfstats=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LLAMINAR_PERF_STATS_JSON=", result.stdout)
        self.assertIn("tp_allreduce_bom", result.stdout)
        self.assertIn("forward_graph", result.stdout)
        self.assertIn("mtp", result.stdout)
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

    def test_inline_prompt_is_forwarded_as_one_argument(self) -> None:
        result = self.run_script(prompt="Explain why fixed prompts aid comparison.")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "--prompt Explain\\ why\\ fixed\\ prompts\\ aid\\ comparison.",
            result.stdout,
        )

    def test_prompt_file_is_forwarded_as_one_argument(self) -> None:
        result = self.run_script(prompt_file="/tmp/llep benchmark prompt.txt")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "--prompt-file /tmp/llep\\ benchmark\\ prompt.txt",
            result.stdout,
        )

    def test_inline_and_file_prompts_are_mutually_exclusive(self) -> None:
        result = self.run_script(
            prompt="inline",
            prompt_file="/tmp/prompt.txt",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "--prompt and --prompt-file are mutually exclusive",
            result.stderr,
        )

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
        self.assertEqual(
            result.stdout.count("--moe-residency-maintenance-window 64"),
            2,
        )

    def test_rebalance_window_override_is_used(self) -> None:
        result = self.run_script(cases="observe", rebalance_window=8)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--moe-residency-maintenance-window 8", result.stdout)
        self.assertNotIn("--moe-residency-maintenance-window 64", result.stdout)

    def test_device_rebalance_env_knobs_are_forwarded_to_benchmark(self) -> None:
        result = self.run_script(
            extra_env={
                "LLAMINAR_MOE_DEVICE_REBALANCE_COMPACT_PAYLOAD_SLOTS": "2",
                "LLAMINAR_MOE_DEVICE_REBALANCE_MAINTENANCE_SLACK_TOKENS": "4",
                "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_MAINTENANCE_PERIOD_TOKENS": "128",
                "LLAMINAR_MOE_DEVICE_REBALANCE_INITIAL_MAINTENANCE_PERIOD_TOKENS": "65",
                "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_WAVE_SPREAD_IMPROVEMENT_PER_PAYLOAD_SLOT": "4096",
                "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_FOREIGN_ROWS_PER_CRITICAL_PATH_PAYLOAD_SLOT": "512",
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
            "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_FOREIGN_ROWS_PER_CRITICAL_PATH_PAYLOAD_SLOT=512",
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
            "--moe-continuation-dense-policy tensor-parallel",
            result.stdout,
        )

    def test_dense_tp_dry_run_does_not_affect_single_card_baseline(self) -> None:
        result = self.run_script(placement="single", dense_tp=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--moe-continuation-dense-policy", result.stdout)

    def test_dense_decode_replicated_dry_run_opts_two_card_overlay_into_decode_full_dense(self) -> None:
        result = self.run_script(dense_tp=True, dense_decode_replicated=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "--moe-continuation-dense-policy prefill-tensor-parallel-decode-replicated",
            result.stdout,
        )

    def test_dense_decode_replicated_dry_run_does_not_affect_single_card_baseline(self) -> None:
        result = self.run_script(
            placement="single",
            dense_tp=True,
            dense_decode_replicated=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--moe-continuation-dense-policy", result.stdout)

    def test_dense_policy_dry_run_overrides_legacy_dense_flags_for_twocard_overlay(self) -> None:
        result = self.run_script(
            dense_tp=True,
            dense_decode_replicated=True,
            dense_policy="tensor-parallel-decode-mirrored-embedding",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "--moe-continuation-dense-policy tensor-parallel-decode-mirrored-embedding",
            result.stdout,
        )
        self.assertNotIn(
            "--moe-continuation-dense-policy prefill-tensor-parallel-decode-replicated",
            result.stdout,
        )

    def test_dense_policy_dry_run_does_not_affect_single_card_baseline(self) -> None:
        result = self.run_script(
            placement="single",
            dense_policy="tensor-parallel-decode-mirrored-embedding",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--moe-continuation-dense-policy", result.stdout)

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

if __name__ == "__main__":
    unittest.main()
