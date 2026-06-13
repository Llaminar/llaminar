#!/usr/bin/env python3
"""Regression tests for the MTP iteration benchmark matrix wrapper."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "run_mtp_iteration_benchmark_matrix.sh"


class MTPIterationBenchmarkMatrixTest(unittest.TestCase):
    def run_matrix(
        self,
        variants: str,
        *,
        allow_partial: bool = False,
        topologies: str = "single",
        models: str = "dense",
        perfstats: bool = False,
        gpu_stage_timing: bool = False,
        mtp_request_batch: int | str = 1,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dense = tmp_path / "dense.gguf"
            moe = tmp_path / "moe.gguf"
            dense.write_text("dense fixture\n", encoding="utf-8")
            moe.write_text("moe fixture\n", encoding="utf-8")
            cmd = [
                str(SCRIPT),
                "--dry-run",
                "--binary",
                "/bin/true",
                "--dense-model",
                str(dense),
                "--moe-model",
                str(moe),
                "--topologies",
                topologies,
                "--devices",
                "cpu:0",
                "--models",
                models,
                "--modes",
                "greedy",
                "--variants",
                variants,
                "--mtp-request-batch",
                str(mtp_request_batch),
                "--output-dir",
                str(tmp_path / "out"),
            ]
            if allow_partial:
                cmd.append("--allow-partial-variants")
            if perfstats:
                cmd.append("--perfstats")
            if gpu_stage_timing:
                cmd.append("--gpu-stage-timing")
            return subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

    def test_dynamic_requires_fixed_depth_neighbors(self) -> None:
        result = self.run_matrix("baseline,fixed_d1,dynamic")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("dynamic matrix rows require", result.stderr)
        self.assertIn("fixed_d2", result.stderr)
        self.assertIn("fixed_d3", result.stderr)

    def test_dynamic_accepts_full_iteration_variant_shape(self) -> None:
        result = self.run_matrix("baseline,fixed_d1,fixed_d2,fixed_d3,dynamic")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dry-run:", result.stdout)

    def test_partial_variant_escape_is_explicit(self) -> None:
        result = self.run_matrix("baseline,dynamic", allow_partial=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dry-run:", result.stdout)
        self.assertIn("--mtp-depth-policy dynamic", result.stdout)
        self.assertIn("--mtp-min-draft-tokens 1", result.stdout)
        self.assertNotIn("--mtp-initial-draft-tokens", result.stdout)

    def test_localtp_rocm_topology_uses_tp_devices_without_single_device_flag(self) -> None:
        result = self.run_matrix(
            "baseline",
            topologies="localtp_rocm2",
            models="dense",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--tp-devices rocm:0\\,rocm:1", result.stdout)
        self.assertIn("--tp-scope local", result.stdout)
        self.assertIn("--backend rccl", result.stdout)
        self.assertNotIn(" -d ", result.stdout)

    def test_localpp_rocm_topology_uses_named_domains(self) -> None:
        result = self.run_matrix(
            "baseline",
            topologies="localpp_rocm2",
            models="dense",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--pipeline-parallelism-degree 2", result.stdout)
        self.assertIn("--define-domain stage0=rocm:0\\;scope=local\\;owner=0", result.stdout)
        self.assertIn("--pp-stage 0=stage0:0-31", result.stdout)
        self.assertIn("--pp-stage 1=stage1:32-63", result.stdout)
        self.assertNotIn(" -d ", result.stdout)

    def test_nodelocaltp_topology_uses_mpi_device_map(self) -> None:
        result = self.run_matrix(
            "baseline",
            topologies="nodelocaltp_cpu2",
            models="dense",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--mpi-procs 2", result.stdout)
        self.assertIn("--device-map 0=cpu:0\\,1=cpu:1", result.stdout)
        self.assertIn("--tp-scope node_local", result.stdout)
        self.assertNotIn(" -d ", result.stdout)

    def test_expert_overlay_topology_uses_moe_overlay_flags(self) -> None:
        result = self.run_matrix(
            "baseline",
            topologies="expert_overlay_rocm2_cpu2",
            models="moe",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--moe-expert-overlay tiered", result.stdout)
        self.assertIn("--moe-expert-overlay-continuation qwen36_moe_rocm_hot", result.stdout)
        self.assertIn("qwen36_moe_rocm_hot=rocm:0\\,rocm:1", result.stdout)
        self.assertIn("qwen36_moe_cpu_cold=cpu:0\\,cpu:1", result.stdout)
        self.assertIn("cold@qwen36_moe_cpu_cold\\;priority=1", result.stdout)
        self.assertNotIn(" -d ", result.stdout)

    def test_topology_model_mismatch_fails_fast(self) -> None:
        result = self.run_matrix(
            "baseline",
            topologies="localpp_rocm2",
            models="moe",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("topology 'localpp_rocm2' is not supported for model 'moe'", result.stderr)

    def test_gpu_stage_timing_requires_perfstats_output(self) -> None:
        result = self.run_matrix(
            "fixed_d1",
            gpu_stage_timing=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--gpu-stage-timing requires --perfstats", result.stderr)

    def test_gpu_stage_timing_sets_perfstats_env_for_mtp_rows(self) -> None:
        result = self.run_matrix(
            "fixed_d1",
            perfstats=True,
            gpu_stage_timing=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LLAMINAR_PERF_STATS_JSON=", result.stdout)
        self.assertIn("LLAMINAR_PERF_STATS_GPU_STAGE_TIMING=1", result.stdout)

    def test_mtp_request_batch_is_forwarded_to_mtp_variants(self) -> None:
        result = self.run_matrix(
            "fixed_d1",
            mtp_request_batch=4,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--mtp-max-request-batch 4", result.stdout)

    def test_mtp_request_batch_rejects_non_positive_values(self) -> None:
        result = self.run_matrix(
            "fixed_d1",
            mtp_request_batch=0,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--mtp-request-batch must be a positive integer", result.stderr)

    def test_baseline_summary_row_matches_header_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dense = tmp_path / "dense.gguf"
            dense.write_text("dense fixture\n", encoding="utf-8")
            fake_binary = tmp_path / "fake_llaminar2"
            fake_binary.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    out=""
                    while [[ $# -gt 0 ]]; do
                      if [[ "$1" == "--benchmark-json-output" ]]; then
                        out="$2"
                        shift 2
                      else
                        shift
                      fi
                    done
                    cat > "${out}" <<'JSON'
                    {"success":true,"throughput_tokens_per_sec":{"decode":10.0,"overall":20.0},"tokens":{"prefill":1,"decode":1},"config":{"mtp_depth_policy":"fixed","mtp_draft_tokens":1},"mtp":{}}
                    JSON
                    """
                ),
                encoding="utf-8",
            )
            fake_binary.chmod(0o755)
            output_dir = tmp_path / "out"
            result = subprocess.run(
                [
                    str(SCRIPT),
                    "--binary",
                    str(fake_binary),
                    "--dense-model",
                    str(dense),
                    "--devices",
                    "cpu:0",
                    "--models",
                    "dense",
                    "--modes",
                    "greedy",
                    "--variants",
                    "baseline",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            lines = (output_dir / "summary.tsv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            header = lines[0].split("\t")
            row = lines[1].split("\t")
            self.assertEqual(len(header), len(row))
            self.assertEqual(len(header), 75)
            self.assertIn("topology", header)
            self.assertEqual(row[header.index("topology")], "single")
            self.assertEqual(row[header.index("device")], "cpu:0")
            for required_column in (
                "generated_policy",
                "min_depth",
                "max_depth",
                "request_batch",
                "depth_updates",
                "depth_promotions",
                "depth_demotions",
                "depth_windows",
                "last_depth_reason",
            ):
                self.assertIn(required_column, header)
            self.assertEqual(row[header.index("depth_updates")], "0")
            self.assertEqual(row[header.index("last_depth_reason")], "")
            self.assertEqual(row[header.index("generated_policy")], "false")
            self.assertEqual(row[header.index("request_batch")], "1")

    def test_perfstats_rows_emit_ranked_stage_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dense = tmp_path / "dense.gguf"
            dense.write_text("dense fixture\n", encoding="utf-8")
            fake_binary = tmp_path / "fake_llaminar2"
            fake_binary.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    out=""
                    while [[ $# -gt 0 ]]; do
                      if [[ "$1" == "--benchmark-json-output" ]]; then
                        out="$2"
                        shift 2
                      else
                        shift
                      fi
                    done
                    cat > "${out}" <<'JSON'
                    {"success":true,"throughput_tokens_per_sec":{"decode":10.0,"overall":20.0},"tokens":{"prefill":1,"decode":1},"config":{"mtp_depth_policy":"fixed","mtp_draft_tokens":1},"mtp":{}}
                    JSON
                    if [[ -n "${LLAMINAR_PERF_STATS_JSON:-}" ]]; then
                      cat > "${LLAMINAR_PERF_STATS_JSON}" <<'JSON'
                    {
                      "schema": "llaminar.perf_stats.v1",
                      "records": [
                        {
                          "kind": "timer",
                          "domain": "stage_gpu",
                          "name": "type.MOE_EXPERT_FFN",
                          "phase": "decode",
                          "device": "CUDA:0",
                          "tags": {
                            "context": "main_verifier",
                            "source": "stage_timeline",
                            "stage_count": "40"
                          },
                          "count": 3,
                          "total_ms": 12.5,
                          "avg_us": 4166.6
                        },
                        {
                          "kind": "timer",
                          "domain": "mtp",
                          "name": "condition_forward",
                          "phase": "decode",
                          "count": 2,
                          "total_ms": 7.25,
                          "avg_us": 3625.0
                        },
                        {
                          "kind": "timer",
                          "domain": "stage_gpu",
                          "name": "type.MOE_ROUTER",
                          "phase": "prefill",
                          "count": 99,
                          "total_ms": 99.0
                        }
                      ]
                    }
                    JSON
                    fi
                    """
                ),
                encoding="utf-8",
            )
            fake_binary.chmod(0o755)
            output_dir = tmp_path / "out"
            result = subprocess.run(
                [
                    str(SCRIPT),
                    "--binary",
                    str(fake_binary),
                    "--dense-model",
                    str(dense),
                    "--devices",
                    "cpu:0",
                    "--models",
                    "dense",
                    "--modes",
                    "greedy",
                    "--variants",
                    "fixed_d1",
                    "--perfstats",
                    "--gpu-stage-timing",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            lines = (output_dir / "stage_summary.tsv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            header = lines[0].split("\t")
            first = lines[1].split("\t")
            second = lines[2].split("\t")
            self.assertEqual(header[:5], ["topology", "device", "model", "mode", "variant"])
            self.assertEqual(first[header.index("name")], "type.MOE_EXPERT_FFN")
            self.assertEqual(first[header.index("context")], "main_verifier")
            self.assertEqual(first[header.index("total_ms")], "12.5")
            self.assertEqual(first[header.index("stage_count")], "40")
            self.assertEqual(first[header.index("source")], "stage_timeline")
            self.assertEqual(second[header.index("name")], "condition_forward")
            self.assertEqual(second[header.index("total_ms")], "7.25")
            self.assertNotIn("type.MOE_ROUTER", "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
