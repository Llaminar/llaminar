#!/usr/bin/env python3
"""Regression tests for the NativeVNNI dispatch refresh wrapper."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "refresh_native_vnni_dispatch_tables.sh"
VALIDATOR = (
    REPO_ROOT
    / "tests"
    / "v2"
    / "performance"
    / "kernels"
    / "validate_native_vnni_generated_dispatch_ids.py"
)


def _current_cpu_serial_policy_hash() -> str:
    """Read the versioned arithmetic identity used by a fresh CPU seal."""

    path = (
        REPO_ROOT
        / "tests/v2/performance/kernels/native_vnni_dispatch/manifests"
        / "cpu_native_vnni_serial_m1_arithmetic_v1.json"
    )
    return str(json.loads(path.read_text(encoding="utf-8"))["contract_hash"])

CPU_PREFILL_CHECKPOINT = (
    "run_id,git_revision,build_id,compiler_id,architecture_class,device_name,"
    "driver_runtime,serial_m1_policy_hash,marker\n"
    "stable-cpu-prefill-development,deadbeef,sha256:fixture|cpu_isa=AVX2,"
    "fixture-cxx,x86_64-fixture|build=AVX2,fixture-cpu,fixture-linux,"
    f"{_current_cpu_serial_policy_hash()},fixture\n"
)

CPU_DECODE_CHECKPOINT = (
    "run_id,git_revision,build_id,compiler_id,architecture_class,device_name,"
    "driver_runtime,serial_m1_policy_hash,marker\n"
    "stable-cpu-decode-development,deadbeef,sha256:fixture|cpu_isa=AVX2,"
    "fixture-cxx,x86_64-fixture|build=AVX2,fixture-cpu,fixture-linux,"
    f"{_current_cpu_serial_policy_hash()},fixture\n"
)


class NativeVNNIDispatchRefreshTest(unittest.TestCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        return self.run_script_with_output_files({}, *args)

    def run_script_with_output_files(
        self,
        output_files: dict[str, str],
        *args: str,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run one dry transaction after publishing resumable output fixtures."""

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            output_dir.mkdir(parents=True)
            for relative_path, contents in output_files.items():
                path = output_dir / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
            return subprocess.run(
                [
                    str(SCRIPT),
                    "--dry-run",
                    "--cuda-sweep-bin",
                    "/bin/true",
                    "--rocm-decode-bin",
                    "/bin/true",
                    "--cpu-avx2-sweep-bin",
                    "/bin/true",
                    "--cpu-avx512-sweep-bin",
                    "/bin/true",
                    "--cpu-threads",
                    "28",
                    "--output-dir",
                    str(output_dir),
                    *args,
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                env=(
                    {**os.environ, **environment}
                    if environment is not None
                    else None
                ),
            )

    def test_cpu_fixed_timing_floor_preflights_before_collection(self) -> None:
        """Production cannot collect rows its own adapter will later reject."""

        result = self.run_script_with_output_files(
            {},
            "--backend",
            "cpu",
            "--profile",
            "all",
            "--install",
            environment={
                "LLAMINAR_NATIVE_VNNI_REFRESH_CPU_WARMUP": "1",
                "LLAMINAR_NATIVE_VNNI_REFRESH_CPU_ITERS": "3",
            },
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "collector is configured for 1/3 timing but the installable "
            "evidence floor is 5/30",
            result.stderr,
        )

    def test_cpu_fixed_timing_floor_reaches_every_analyzer(self) -> None:
        """The shell transaction exposes one consistent analyzer contract."""

        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "quick",
            "--cpu-minimum-promotion-warmups",
            "1",
            "--cpu-minimum-promotion-samples",
            "3",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--minimum-promotion-warmups 1", result.stdout)
        self.assertIn("--minimum-promotion-samples 3", result.stdout)

    def test_cpu_paired_resume_budget_starts_after_completed_iterations(
        self,
    ) -> None:
        """Historical refinement rounds cannot exhaust a new invocation."""

        wrapper = SCRIPT.read_text(encoding="utf-8")
        cpu_refinement = wrapper.split(
            "run_cpu_decode_paired_refinement() {", 1
        )[1].split("refresh_cpu_decode() {", 1)[0]

        self.assertIn(
            "for ((candidate_iteration = 0; ; ++candidate_iteration))",
            cpu_refinement,
        )
        self.assertIn(
            "iteration_limit=$((first_iteration + paired_max_iterations))",
            cpu_refinement,
        )
        self.assertIn("iteration < iteration_limit", cpu_refinement)
        self.assertNotIn(
            "candidate_iteration < paired_max_iterations",
            cpu_refinement,
        )

    def test_both_backends_emit_m_aware_sweep_contract(self) -> None:
        result = self.run_script("--backend", "both", "--profile", "quick")

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        runtime_m = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31"
        self.assertIn(f"LLAMINAR_CUDA_NVNNI_DECODE_M={runtime_m}", stdout)
        self.assertIn(f"LLAMINAR_ROCM_NVNNI_DECODE_M={runtime_m}", stdout)
        self.assertIn(
            "LLAMINAR_ROCM_NVNNI_DECODE_EXECUTION_MODES=eager,graph_captured",
            stdout,
        )
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_TIMING_CSV=", stdout)
        self.assertIn(
            "LLAMINAR_CUDA_NVNNI_DECODE_EXECUTION_MODES=eager,graph_captured",
            stdout,
        )
        for grouped_rows in (2, 4, 8, 16, 32):
            self.assertIn(
                "cuda.nvnni.decode.verifier.inherit_serial_m1."
                f"r{grouped_rows}",
                stdout,
            )
        self.assertIn(
            "cuda.nvnni.decode.verifier.tensor_core_mma16",
            stdout,
        )
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_TIMING_CSV=", stdout)
        self.assertIn("analyze_cuda_native_vnni_decode_trainer.py", result.stdout)
        self.assertNotIn("infer_gemv_dispatch_heuristic.py", result.stdout)
        self.assertNotIn("analyze_cuda_tc_gemv_dispatch.py", result.stdout)
        self.assertIn("analyze_rocm_native_vnni_decode_trainer.py", result.stdout)
        self.assertIn("validate_native_vnni_generated_dispatch_ids.py", result.stdout)
        self.assertNotIn("native_vnni_dispatch.profiler_collectors", result.stdout)

    def test_gpu_measurement_lanes_partition_work_without_duplication(self) -> None:
        """CUDA shards formats while ROCm shards finer-grained shapes."""

        formats = "Q4_0,Q5_0,Q6_K,Q8_0"
        result = self.run_script(
            "--backend",
            "both",
            "--profile",
            "quick",
            "--cuda-formats",
            formats,
            "--rocm-formats",
            formats,
            "--cuda-measurement-lanes",
            "2",
            "--rocm-measurement-lanes",
            "4",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn("CUDA measurement lanes: 2 disjoint format shard(s)", stdout)
        self.assertIn("ROCm measurement lanes: 4 parallel sweep lane(s)", stdout)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", stdout)
        self.assertIn("CUDA_VISIBLE_DEVICES=1", stdout)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_FORMATS=Q4_0,Q6_K", stdout)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_FORMATS=Q5_0,Q8_0", stdout)
        self.assertIn("ROCR_VISIBLE_DEVICES=0", stdout)
        self.assertIn("ROCR_VISIBLE_DEVICES=1", stdout)
        self.assertNotIn("ROCR_VISIBLE_DEVICES=2", stdout)
        self.assertEqual(
            stdout.count(f"LLAMINAR_ROCM_NVNNI_DECODE_FORMATS={formats}"),
            2,
        )
        self.assertIn(
            "LLAMINAR_ROCM_NVNNI_DECODE_SHAPES=Qwen36_FFN_DownProjection",
            stdout,
        )
        self.assertIn(
            "LLAMINAR_ROCM_NVNNI_DECODE_SHAPES=Qwen36_GDN_OutputProjection",
            stdout,
        )
        self.assertIn("cuda_decode_sweep.lane0.csv", stdout)
        self.assertIn("cuda_decode_sweep.lane1.csv", stdout)
        self.assertIn("rocm_decode_sweep.lane1.csv", stdout)
        self.assertEqual(stdout.count("LLAMINAR_CUDA_NVNNI_DECODE_FORMATS="), 2)
        self.assertEqual(stdout.count("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS="), 2)

    def test_gpu_measurement_lane_count_must_be_positive(self) -> None:
        result = self.run_script(
            "--backend",
            "cuda",
            "--cuda-measurement-lanes",
            "0",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "--cuda-measurement-lanes must be a positive integer or auto",
            result.stderr,
        )

    def test_single_gpu_backend_does_not_validate_inactive_lane_inventory(self) -> None:
        result = self.run_script(
            "--backend",
            "cuda",
            "--profile",
            "quick",
            "--cuda-formats",
            "Q4_0,Q5_0",
            "--cuda-measurement-lanes",
            "2",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CUDA measurement lanes: 2", result.stdout)
        self.assertNotIn("ROCm measurement lanes:", result.stdout)

    def test_metadata_readers_do_not_sigpipe_verbose_tool_producers(self) -> None:
        """Strict pipefail must not turn a version-banner read into exit 141."""

        wrapper = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("first_output_line()", wrapper)
        self.assertIn("hipcc --version | first_output_line", wrapper)
        self.assertIn("driver_version --format=csv,noheader -i 0 | first_output_line", wrapper)
        self.assertNotRegex(wrapper, r"--version\s*\|\s*head\s+-n\s+1")

    def test_cpu_prefill_refinement_probes_generated_boundary_rings(self) -> None:
        """Each failed fit obtains C++ routes before timing dynamic shapes."""

        wrapper = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--probe-development-fit", wrapper)
        self.assertIn("--refinement-probe-plan", wrapper)
        self.assertIn("--refinement-probe-route-manifest", wrapper)
        self.assertIn(
            "LLAMINAR_CPU_NVNNI_PREFILL_ROUTE_REFINEMENT_PROBE_JSON=",
            wrapper,
        )

    def test_cpu_prefill_fit_only_retains_measured_build_provenance(self) -> None:
        """A fitter replay must not claim that a rebuilt harness timed rows."""

        wrapper = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("csv_unique_field_prefix()", wrapper)
        self.assertIn(
            'if (( skip_sweep )) && [[ -s "${cpu_prefill_common_csv}" ]]',
            wrapper,
        )
        self.assertIn(
            '"${cpu_prefill_common_csv}" build_id \'|cpu_isa=\'',
            wrapper,
        )
        self.assertIn(
            '"${cpu_prefill_common_csv}" architecture_class \'|build=\'',
            wrapper,
        )
        self.assertIn(
            "Fit-only CPU prefill replay retains measured build provenance",
            wrapper,
        )

    def test_fresh_sealed_rows_use_current_measurement_provenance(self) -> None:
        """A reused development fit must not relabel newly timed holdout rows."""

        wrapper = SCRIPT.read_text(encoding="utf-8")
        prefill = wrapper[wrapper.index("refresh_cpu_prefill()") :]

        self.assertIn("cpu_measurement_build_id", prefill)
        self.assertIn(
            '"collection_build_id=${cpu_measurement_build_id}"',
            prefill,
        )
        self.assertIn(
            'cpu_build_id="${cpu_measurement_build_id}"',
            prefill,
        )
        self.assertIn(
            'cpu_development_build_id="${cpu_build_id}"',
            prefill,
        )
        self.assertIn(
            '--build-id "${cpu_development_build_id}"',
            prefill,
        )
        self.assertIn(
            '--sealed-build-id "${cpu_measurement_build_id}"',
            prefill,
        )
        self.assertIn(
            '--sealed-serial-m1-policy-hash',
            prefill,
        )
        self.assertIn(
            '[[ "${cpu_serial_policy_hash}" != "${cpu_measurement_serial_policy_hash}" ]]',
            prefill,
        )
        restore = prefill.index(
            'cpu_build_id="${cpu_measurement_build_id}"',
            prefill.index("The certification holdout is launched"),
        )
        sealed_launch = prefill.index(
            '--sealed-witness-plan "${cpu_prefill_sealed_witness_plan_json}"',
            restore,
        )
        certify = prefill.index("local -a cpu_prefill_certify=(", restore)
        self.assertLess(restore, sealed_launch)
        self.assertLess(restore, certify)

    def test_cpu_decode_fit_replay_separates_development_and_seal_builds(self) -> None:
        """M=1 paired evidence must be burned by a trainer rebuild."""

        wrapper = SCRIPT.read_text(encoding="utf-8")
        decode = wrapper[
            wrapper.index("refresh_cpu_decode()") :
            wrapper.index("refresh_cpu()")
        ]
        self.assertIn("cpu_measurement_build_id", decode)
        self.assertIn(
            'if (( skip_sweep )) && [[ -s "${cpu_decode_common_csv}" ]]',
            decode,
        )
        self.assertIn(
            '"${cpu_decode_common_csv}" build_id \'|cpu_isa=\'',
            decode,
        )
        self.assertIn(
            "Fit-only CPU decode replay retains measured build provenance",
            decode,
        )
        self.assertIn(
            'csv_has_field "${cpu_decode_common_csv}" launch_k_tiles',
            decode,
        )
        self.assertIn(
            "re-adapting retained raw timing evidence",
            decode,
        )
        self.assertGreaterEqual(
            decode.count('--sealed-build-id "${cpu_measurement_build_id}"'),
            3,
        )
        self.assertIn("cpu_decode_sealed_build_token", decode)
        self.assertIn("cpu_measurement_serial_policy_hash", decode)
        self.assertIn(
            "fresh CPU decode seal cannot certify a changed serial arithmetic policy",
            decode,
        )

    def test_cpu_grouped_resume_reuses_authenticated_common_corpus(self) -> None:
        """A downstream failure cannot force another gigabyte adaptation."""

        wrapper = SCRIPT.read_text(encoding="utf-8")
        grouped = wrapper[
            wrapper.index("refresh_cpu()") :
            wrapper.index("refresh_cpu_prefill()")
        ]

        self.assertIn("cpu_grouped_development_run_id_file", wrapper)
        self.assertIn("reuse_cpu_grouped_common", grouped)
        self.assertIn(
            '(( resume_cpu_partials || skip_sweep ))',
            grouped,
        )
        self.assertIn(
            "Reusing authenticated CPU grouped common corpus",
            grouped,
        )
        self.assertIn("csv_first_cpu_provenance", grouped)
        self.assertIn(
            "Fit-only CPU grouped replay retains measured build provenance",
            grouped,
        )
        self.assertEqual(
            grouped.count('--sealed-build-id "${cpu_measurement_build_id}"'),
            2,
        )
        self.assertEqual(
            grouped.count(
                '--development-profiler-observations "${cpu_common_csv}"'
            ),
            2,
        )
        freeze = grouped.index("local -a cpu_frozen_args=(")
        self.assertLess(grouped.index("--reuse-development-common"), freeze + 800)

    def test_cpu_prefill_sealed_replay_always_reuses_final_shards(self) -> None:
        """Installation replay must not require a flag to preserve sealed data."""

        wrapper = SCRIPT.read_text(encoding="utf-8")
        prefill = wrapper[wrapper.index("refresh_cpu_prefill()") :]
        sealed = prefill[
            prefill.index("local sealed_partials=()") :
            prefill.index("combine_csvs \"${cpu_prefill_sealed_csv}\"")
        ]

        self.assertIn(
            'if [[ -s "${partial}" && -s "${timing_partial}" ]]; then',
            sealed,
        )
        self.assertNotIn(
            'if (( resume_cpu_partials )) &&',
            sealed,
        )

    def test_rocm_device_identity_cannot_select_a_cpu_agent(self) -> None:
        """ROCm policy provenance must come from the GPU-only product table."""

        wrapper = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("detect_rocm_device_name()", wrapper)
        self.assertIn("rocm-smi --showproductname --csv", wrapper)
        self.assertIn('row.get("device", "").startswith("card")', wrapper)
        self.assertIn('row.get("Card Series", "").strip()', wrapper)
        self.assertNotRegex(
            wrapper,
            r'rocm_device_name=.*rocminfo.*Marketing Name',
        )

    def test_cuda_production_scopes_q4_refinement_across_shape_lanes(self) -> None:
        """CUDA-only Q4 evidence is shape-sharded, never format-multiplied."""

        result = self.run_script(
            "--backend",
            "cuda",
            "--profile",
            "all",
            "--cuda-formats",
            "Q4_0,Q5_0",
            "--cuda-measurement-lanes",
            "2",
            "--skip-profiler-evidence",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        common_lines = [
            line for line in stdout.splitlines()
            if "cuda_decode_m1-development-common.lane" in line
            and "LLAMINAR_CUDA_NVNNI_DECODE_SHAPES=" in line
        ]
        refinement_lines = [
            line for line in stdout.splitlines()
            if "cuda_decode_m1-development-cuda-q4-refinement.lane" in line
            and "LLAMINAR_CUDA_NVNNI_DECODE_SHAPES=" in line
        ]
        self.assertEqual(len(common_lines), 2)
        self.assertEqual(len(refinement_lines), 2)
        scoped_lane0 = "FastM1RefineV5_3B_FFN_Dn_Nm128_1920x11008"
        scoped_lane1 = "FastM1RefineV5_3B_FFN_Dn_Nm64_1984x11008"
        self.assertTrue(all(scoped_lane0 not in line for line in common_lines))
        self.assertIn("CUDA_VISIBLE_DEVICES=0", refinement_lines[0])
        self.assertIn("CUDA_VISIBLE_DEVICES=1", refinement_lines[1])
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_FORMATS=Q4_0", refinement_lines[0])
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_FORMATS=Q4_0", refinement_lines[1])
        self.assertIn(scoped_lane0, refinement_lines[0])
        self.assertNotIn(scoped_lane0, refinement_lines[1])
        self.assertIn(scoped_lane1, refinement_lines[1])
        self.assertNotIn(scoped_lane1, refinement_lines[0])

    def test_cuda_fit_only_reuses_corpus_without_kernel_or_profiler_launch(self) -> None:
        """A sealed corpus may refit, but cannot silently recollect evidence."""

        required = {
            "cuda_decode_m1.development.csv": "header\nrow\n",
            "cuda_decode_m1.development.timing.csv": "header\nrow\n",
            "cuda_decode_m1.sealed.csv": "header\nrow\n",
            "cuda_decode_m1.sealed.timing.csv": "header\nrow\n",
            "cuda_decode_verifier.csv": "header\nrow\n",
            "cuda_decode_verifier.timing.csv": "header\nrow\n",
            "cuda_decode_m1_common_observations.csv": "header\nrow\n",
            "cuda_profiler_requests.json": "{}\n",
            "cuda_profiler_evidence.json": "{}\n",
            "cuda_decode_common_observations.csv": "header\nrow\n",
            "cuda_final_profiler_requests.json": "{}\n",
            "cuda_final_profiler_evidence.json": "{}\n",
            "cuda_paired_refinement/iteration-000.csv": "header\nrow\n",
        }
        result = self.run_script_with_output_files(
            required,
            "--backend",
            "cuda",
            "--profile",
            "all",
            "--skip-sweep",
            "--reuse-profiler-evidence",
            "--install",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("compact-witnesses", result.stdout)
        self.assertIn("export-features", result.stdout)
        self.assertIn("--paired-csv", result.stdout)
        self.assertIn("analyze_cuda_native_vnni_decode_trainer.py", result.stdout)
        self.assertNotIn(
            "LLAMINAR_CUDA_NVNNI_DECODE_PAIRED_REQUEST_MANIFEST=",
            result.stdout,
        )
        self.assertNotIn("native_vnni_dispatch.profiler_collectors", result.stdout)
        self.assertNotIn("LLAMINAR_CUDA_NVNNI_DECODE_CSV=", result.stdout)

    def test_cuda_audited_development_reuse_opens_only_fresh_phases(self) -> None:
        """Retained M1 development is authenticated before seal and grouped work."""

        result = self.run_script(
            "--backend",
            "cuda",
            "--profile",
            "all",
            "--reuse-cuda-development",
            "--cuda-development-build-change-audit",
            "selector-cache-only rebuild; candidate arithmetic unchanged",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertNotIn("cuda_decode_m1-development-common.lane", stdout)
        self.assertNotIn(
            "cuda_decode_m1-development-cuda-q4-refinement.lane",
            stdout,
        )
        self.assertIn("cuda_decode_m1.sealed.csv", stdout)
        self.assertIn("cuda_decode_verifier.development.csv", stdout)
        self.assertIn("cuda_decode_verifier.sealed.csv", stdout)
        self.assertIn("--development-build-change-audit", stdout)
        self.assertIn("--sealed-build-id", stdout)
        self.assertIn("load_cuda_development_context", stdout)
        self.assertIn("write_cuda_measurement_context", stdout)
        self.assertIn("cuda_decode_m1_common_observations.reuse-source.csv", stdout)
        self.assertIn("cp --reflink=auto", stdout)
        self.assertIn(
            "cuda_decode_m1_common_observations.reuse-source.csv.inprogress",
            stdout,
        )
        self.assertNotRegex(
            stdout,
            r"mv [^\n]*cuda_decode_m1_common_observations\.csv "
            r"[^\n]*cuda_decode_m1_common_observations\.reuse-source\.csv(?:\n|$)",
        )

        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            'cuda_reuse_context_source="${cuda_reuse_common_source}"',
            script,
        )
        self.assertNotIn(
            "retained CUDA common-observation snapshot conflicts",
            script,
        )
        identity_check = (
            'cmp --silent \\\n'
            '          "${cuda_reuse_common_source}" "${cuda_m1_common_csv}"'
        )
        self.assertIn(identity_check, script)
        self.assertLess(
            script.index(identity_check, script.index("collect_backend_profiler_evidence")),
            script.index(
                "native_vnni_dispatch.common_observation_migration",
                script.index(identity_check, script.index("collect_backend_profiler_evidence")),
            ),
        )
        self.assertIn(
            "Authenticated byte-identical already-current CUDA "
            "common-observation replay",
            script,
        )
        self.assertIn("--finalize-retained-identity", script)
        self.assertIn(
            'cuda_profiler_observation_source="${cuda_reuse_common_source}"',
            script,
        )
        profiler_call = (
            'cuda "${cuda_profiler_observation_source}" "${cuda_sweep_bin}"'
        )
        self.assertIn(profiler_call, script)
        self.assertLess(
            script.index(profiler_call),
            script.index(
                "native_vnni_dispatch.common_observation_migration",
                script.index(profiler_call),
            ),
        )

    def test_rocm_audited_development_reuse_opens_only_fresh_phases(self) -> None:
        """Retained ROCm M1 timing keeps its identity while the seal is fresh."""

        provenance_header = (
            "run_id,git_revision,build_id,compiler_id,architecture_class,"
            "device_name,driver_runtime,serial_m1_policy_hash\n"
        )
        provenance_row = (
            "retained-rocm-development,deadbeef,sha256:retained-build,"
            "HIP fixture,gfx906,MI50,ROCm fixture,sha256:retained-policy\n"
        )
        required = {
            "rocm_decode_fast.development.csv": "header\nrow\n",
            "rocm_decode_fast.development.timing.csv": "header\nrow\n",
            "rocm_decode_fast.development-common.csv": (
                provenance_header + provenance_row
            ),
            "rocm_profiler_requests.json": "{}\n",
            "rocm_profiler_evidence.json": "{}\n",
        }
        result = self.run_script_with_output_files(
            required,
            "--backend",
            "rocm",
            "--profile",
            "all",
            "--reuse-rocm-development",
            "--rocm-development-build-change-audit",
            "profiler lifetime guard only; candidate arithmetic unchanged",
            "--rocm-measurement-lanes",
            "2",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn(
            "Reusing authenticated ROCm development generation "
            "retained-rocm-development",
            stdout,
        )
        self.assertNotIn("rocm_decode_fast-development.lane", stdout)
        self.assertIn("rocm_decode_fast-sealed.lane", stdout)
        self.assertIn("rocm_decode_verifier.lane", stdout)
        self.assertIn(
            "--development-build-change-audit "
            "profiler\\ lifetime\\ guard\\ only",
            stdout,
        )
        self.assertIn("--build-id sha256:retained-build", stdout)
        self.assertIn("--sealed-build-id sha256:dry-run-rocm-build", stdout)
        self.assertEqual(
            stdout.count("--development-profiler-observations"),
            2,
        )
        self.assertEqual(stdout.count("--generic-max-leaves 1"), 2)
        self.assertIn("rocm_profiler_observation_witnesses.csv", stdout)
        self.assertIn("rocm_decode_fast.development-common.csv", stdout)
        self.assertIn("rocm_decode_fast_common_observations.csv", stdout)
        self.assertEqual(
            stdout.count("native_vnni_dispatch.profiler_collectors"),
            1,
        )
        self.assertIn("rocm_final_profiler_requests.json.seed.inprogress", stdout)
        self.assertIn("rocm_final_profiler_requests.json", stdout)

        script = SCRIPT.read_text(encoding="utf-8")
        self.assertEqual(
            script.count("seed_backend_profiler_evidence_for_extension \\\n"),
            2,
        )

    def test_rocm_audited_resume_reuses_complete_sealed_aggregate(self) -> None:
        """A post-seal fit failure must not repay four GPU timing lanes."""

        provenance_header = (
            "run_id,git_revision,build_id,compiler_id,architecture_class,"
            "device_name,driver_runtime,serial_m1_policy_hash\n"
        )
        provenance_row = (
            "retained-rocm-development,deadbeef,sha256:retained-build,"
            "HIP fixture,gfx906,MI50,ROCm fixture,sha256:retained-policy\n"
        )
        required = {
            "rocm_decode_fast.development.csv": "header\nrow\n",
            "rocm_decode_fast.development.timing.csv": "header\nrow\n",
            "rocm_decode_fast.sealed.csv": "header\nrow\n",
            "rocm_decode_fast.sealed.timing.csv": "header\nrow\n",
            "rocm_decode_fast.development-common.csv": (
                provenance_header + provenance_row
            ),
            "rocm_profiler_requests.json": "{}\n",
            "rocm_profiler_evidence.json": "{}\n",
        }
        result = self.run_script_with_output_files(
            required,
            "--backend",
            "rocm",
            "--profile",
            "all",
            "--reuse-rocm-development",
            "--rocm-development-build-change-audit",
            "profiler lifetime guard only; candidate arithmetic unchanged",
            "--rocm-measurement-lanes",
            "4",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "Reusing existing ROCm Fast sealed aggregate",
            result.stdout,
        )
        self.assertNotIn("rocm_decode_fast-sealed.lane", result.stdout)
        self.assertIn("rocm_decode_verifier.lane", result.stdout)

    def test_rocm_complete_resume_preserves_all_measurement_provenance(self) -> None:
        """Fit-only continuation reuses development, sealed, and verifier IDs."""

        provenance_header = (
            "run_id,git_revision,build_id,compiler_id,architecture_class,"
            "device_name,driver_runtime,serial_m1_policy_hash\n"
        )
        development = (
            "retained-rocm-development,dev-git,sha256:dev-build,HIP fixture,"
            "gfx906,MI50,ROCm fixture,sha256:dev-policy\n"
        )
        sealed = (
            "retained-rocm-sealed,seal-git,sha256:seal-build,HIP fixture,"
            "gfx906,MI50,ROCm fixture,sha256:seal-policy\n"
        )
        verifier = (
            "retained-rocm-verifier,verifier-git,sha256:verifier-build,"
            "HIP fixture,gfx906,MI50,ROCm fixture,sha256:verifier-policy\n"
        )
        required = {
            "rocm_decode_fast.development.csv": "header\nrow\n",
            "rocm_decode_fast.development.timing.csv": "header\nrow\n",
            "rocm_decode_fast.sealed.csv": "header\nrow\n",
            "rocm_decode_fast.sealed.timing.csv": "header\nrow\n",
            "rocm_decode_verifier.csv": "header\nrow\n",
            "rocm_decode_verifier.timing.csv": "header\nrow\n",
            "rocm_decode_fast.development-common.csv": (
                provenance_header + development
            ),
            "rocm_decode_fast_common_observations.csv": (
                provenance_header + development + sealed
            ),
            "rocm_decode_verifier_common_observations.csv": (
                provenance_header + verifier
            ),
            "rocm_profiler_requests.json": "{}\n",
            "rocm_profiler_evidence.json": "{}\n",
        }
        result = self.run_script_with_output_files(
            required,
            "--backend",
            "rocm",
            "--profile",
            "all",
            "--skip-sweep",
            "--reuse-rocm-development",
            "--rocm-development-build-change-audit",
            "profiler process lifetime only",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn(
            "Reusing authenticated ROCm sealed and verifier generations "
            "retained-rocm-sealed / retained-rocm-verifier",
            stdout,
        )
        self.assertNotIn("rocm_decode_fast-development.lane", stdout)
        self.assertNotIn("rocm_decode_fast-sealed.lane", stdout)
        self.assertNotIn("rocm_decode_verifier.lane", stdout)
        self.assertIn("--sealed-build-id sha256:seal-build", stdout)
        self.assertIn("--run-id retained-rocm-verifier", stdout)
        self.assertIn("--build-id sha256:verifier-build", stdout)
        self.assertIn(
            "--serial-m1-policy-hash sha256:verifier-policy",
            stdout,
        )

    def test_rocm_development_reuse_requires_an_audit(self) -> None:
        """A harness rebuild cannot silently relabel retained ROCm timing."""

        result = self.run_script(
            "--backend",
            "rocm",
            "--profile",
            "all",
            "--reuse-rocm-development",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "--reuse-rocm-development requires a non-empty "
            "--rocm-development-build-change-audit",
            result.stderr,
        )

    def test_rocm_generic_leaf_budget_is_validated_before_collection(self) -> None:
        """An impossible generic-tree budget must fail before GPU work."""

        result = self.run_script(
            "--backend",
            "rocm",
            "--profile",
            "all",
            "--rocm-generic-max-leaves",
            "0",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "--rocm-generic-max-leaves must be in [1, 32]",
            result.stderr,
        )

    def test_cuda_development_reuse_requires_explicit_audit(self) -> None:
        """Cross-build evidence reuse cannot silently relabel an old corpus."""

        result = self.run_script(
            "--backend",
            "cuda",
            "--profile",
            "all",
            "--reuse-cuda-development",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "requires a non-empty --cuda-development-build-change-audit",
            result.stderr,
        )

    def test_cuda_generic_leaf_budget_is_validated_before_collection(self) -> None:
        """An impossible CUDA generic-tree budget must fail before GPU work."""

        result = self.run_script(
            "--backend",
            "cuda",
            "--profile",
            "all",
            "--cuda-generic-max-leaves",
            "0",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "--cuda-generic-max-leaves must be in [1, 32]",
            result.stderr,
        )

    def test_cuda_best_effort_leaf_budget_is_transaction_wide(self) -> None:
        """Planning, M1 certification, and grouped compile share one budget."""

        result = self.run_script(
            "--backend",
            "cuda",
            "--profile",
            "all",
            "--install",
            "--cuda-generic-max-leaves",
            "1",
            "--maximum-p95-regret-percent",
            "100",
            "--minimum-passing-domain-percent",
            "0",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("--max-leaves 1"), 1)
        self.assertEqual(result.stdout.count("--max-regret 1"), 1)
        self.assertEqual(result.stdout.count("--generic-max-leaves 1"), 3)
        self.assertNotIn("policy_fit_cache/cuda_decode", result.stdout)
        self.assertEqual(
            result.stdout.count("cuda_paired_refinement/fit-cache"),
            3,
        )

    def test_cuda_paired_resume_never_overwrites_retained_generation(self) -> None:
        """A resumed tournament allocates its next durable iteration additively."""

        result = self.run_script_with_output_files(
            {"cuda_paired_refinement/iteration-000.csv": "header\nrow\n"},
            "--backend",
            "cuda",
            "--profile",
            "all",
            "--reuse-cuda-development",
            "--cuda-development-build-change-audit",
            "reviewed selector-only rebuild",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "--paired-csv ",
            result.stdout,
        )
        self.assertIn("iteration-000.csv", result.stdout)
        self.assertIn("iteration-001.requests.json", result.stdout)
        self.assertIn("iteration-001.csv", result.stdout)

    def test_cuda_m1_seal_resume_collects_only_grouped_phases(self) -> None:
        """A completed M1 seal must survive a certification-code interruption."""

        result = self.run_script(
            "--backend",
            "cuda",
            "--profile",
            "all",
            "--resume-cuda-after-m1-seal",
            "--cuda-development-build-change-audit",
            "parallel certification only; CUDA arithmetic unchanged",
            "--cuda-generic-max-leaves",
            "2",
            "--install",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cuda_decode_m1.sealed.csv", result.stdout)
        self.assertNotIn("cuda_decode_m1-sealed.lane", result.stdout)
        self.assertIn(
            "load_cuda_measurement_context", result.stdout
        )
        self.assertIn("cuda_decode_verifier.development.csv", result.stdout)
        self.assertIn("cuda_decode_verifier.sealed.csv", result.stdout)

    def test_cuda_complete_fit_resume_never_relaunches_timing_sweeps(self) -> None:
        """Post-collection analyzer repairs must reuse every durable CUDA row."""

        retained_corpus = {
            "cuda_decode_m1.development.csv": "header\nrow\n",
            "cuda_decode_m1.development.timing.csv": "header\nrow\n",
            "cuda_decode_m1.sealed.csv": "header\nrow\n",
            "cuda_decode_m1.sealed.timing.csv": "header\nrow\n",
            "cuda_decode_verifier.csv": "header\nrow\n",
            "cuda_decode_verifier.timing.csv": "header\nrow\n",
        }
        result = self.run_script_with_output_files(
            retained_corpus,
            "--backend",
            "cuda",
            "--profile",
            "all",
            "--resume-cuda-after-m1-seal",
            "--cuda-development-build-change-audit",
            "adapter-only metadata normalization; CUDA arithmetic unchanged",
            "--skip-sweep",
            "--cuda-generic-max-leaves",
            "2",
            "--install",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("load_cuda_measurement_context", result.stdout)
        self.assertIn("--verifier-input", result.stdout)
        self.assertIn("cuda_decode_verifier.csv", result.stdout)
        self.assertNotIn("LLAMINAR_CUDA_NVNNI_DECODE_CSV=", result.stdout)

    def test_rocm_fit_only_reuses_corpus_without_kernel_or_profiler_launch(self) -> None:
        """ROCm fit replay authenticates sidecars and performs no HIP sweep."""

        required = {
            "rocm_decode_fast.development.csv": "header\nrow\n",
            "rocm_decode_fast.development.timing.csv": "header\nrow\n",
            "rocm_decode_fast.sealed.csv": "header\nrow\n",
            "rocm_decode_fast.sealed.timing.csv": "header\nrow\n",
            "rocm_decode_verifier.csv": "header\nrow\n",
            "rocm_decode_verifier.timing.csv": "header\nrow\n",
            "rocm_decode_fast_common_observations.csv": "header\nrow\n",
            "rocm_profiler_requests.json": "{}\n",
            "rocm_profiler_evidence.json": "{}\n",
            "rocm_decode_common_observations.csv": "header\nrow\n",
            "rocm_final_profiler_requests.json": "{}\n",
            "rocm_final_profiler_evidence.json": "{}\n",
        }
        result = self.run_script_with_output_files(
            required,
            "--backend",
            "rocm",
            "--profile",
            "all",
            "--skip-sweep",
            "--reuse-profiler-evidence",
            "--install",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("compact-witnesses", result.stdout)
        self.assertIn("export-features", result.stdout)
        self.assertIn("analyze_rocm_native_vnni_decode_trainer.py", result.stdout)
        self.assertNotIn("native_vnni_dispatch.profiler_collectors", result.stdout)
        self.assertNotIn("LLAMINAR_ROCM_NVNNI_DECODE_CSV=", result.stdout)

    def test_cpu_verifier_fit_only_reuses_all_isa_corpus_evidence(self) -> None:
        """CPU policy replay retains ISA evidence and launches no timing test."""

        required = {
            "cpu_decode_development_run_id.txt": (
                "stable-cpu-decode-development\n"
            ),
            "cpu_decode_m1.development.csv": "header\nrow\n",
            "cpu_decode_m1.development.timing.csv": "header\nrow\n",
            "cpu_decode_m1.sealed.csv": "header\nrow\n",
            "cpu_decode_m1.sealed.timing.csv": "header\nrow\n",
            "cpu_decode_m1_common_observations.csv": CPU_DECODE_CHECKPOINT,
            "cpu_decode_profiler_requests.json": "{}\n",
            "cpu_decode_profiler_evidence.json": "{}\n",
            "cpu_verifier_rows.development.csv": "header\nrow\n",
            "cpu_verifier_rows.development.timing.csv": "header\nrow\n",
            "cpu_verifier_rows.sealed.csv": "header\nrow\n",
            "cpu_verifier_rows.sealed.timing.csv": "header\nrow\n",
            "cpu_verifier_rows_common_observations.csv": CPU_DECODE_CHECKPOINT,
            "cpu_profiler_requests.json": "{}\n",
            "cpu_profiler_evidence.json": "{}\n",
            "cpu_final_profiler_requests.json": "{}\n",
            "cpu_final_profiler_evidence.json": "{}\n",
        }
        result = self.run_script_with_output_files(
            required,
            "--backend",
            "cpu",
            "--profile",
            "all",
            "--skip-sweep",
            "--reuse-profiler-evidence",
            "--install",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("compact-witnesses", result.stdout)
        self.assertIn("export-features", result.stdout)
        self.assertIn("analyze_cpu_native_vnni_verifier_trainer.py", result.stdout)
        self.assertNotIn("native_vnni_dispatch.profiler_collectors", result.stdout)
        self.assertRegex(
            result.stdout,
            r"ln .*cpu_profiler_requests\.json "
            r".*cpu_final_profiler_requests\.json\.inprogress",
        )
        self.assertRegex(
            result.stdout,
            r"ln .*cpu_profiler_evidence\.json "
            r".*cpu_final_profiler_evidence\.json\.inprogress",
        )
        self.assertNotIn("LLAMINAR_CPU_NVNNI_VERIFIER_STRONG_CSV=", result.stdout)

    def test_cpu_grouped_certification_never_reprofiles_identical_evidence(
        self,
    ) -> None:
        """The post-certification aliases reuse the authenticated transaction."""

        wrapper = SCRIPT.read_text(encoding="utf-8")
        grouped = wrapper[
            wrapper.index("refresh_cpu()") :
            wrapper.index("refresh_cpu_prefill()")
        ]
        certification = grouped[grouped.index("local -a cpu_certify_args=(") :]

        self.assertIn(
            "reuse_identical_backend_profiler_evidence",
            certification,
        )
        self.assertIn('"${cpu_profiler_requests}"', certification)
        self.assertIn('"${cpu_profiler_evidence}"', certification)
        certification_call = certification[
            certification.index("reuse_identical_backend_profiler_evidence") :
        ]
        self.assertNotIn(
            "collect_backend_profiler_evidence",
            certification_call.split("finish_backend_collection_target", 1)[0],
        )

    def test_cpu_prefill_fit_only_cannot_install_generated_policy(self) -> None:
        """Offline prefill evidence must never become a production artifact."""

        result = self.run_script(
            "--backend",
            "cpu-prefill",
            "--profile",
            "all",
            "--skip-sweep",
            "--reuse-profiler-evidence",
            "--install",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "ordinary prefill is heuristic-only; cpu-prefill cannot install",
            result.stderr,
        )
        self.assertNotIn("analyze_cpu_native_vnni_prefill_trainer.py", result.stdout)

    def test_dispatch_validator_decodes_cpu_prefill_packed_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            include = Path(tmp) / "prefill.inc"
            # CB=5, M=1024, K=5120, N=27648 in the 8/16/20/20 ABI.
            key = (5 << 56) | (1024 << 40) | (5120 << 20) | 27648
            include.write_text(
                "packCPUNativeVNNIPrefillPolicyKey\n"
                f"switch (key) {{ case 0x{key:016x}ULL: break; }}\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(VALIDATOR), str(include)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("validated 1", result.stdout)

    def test_dispatch_validator_decodes_compact_cpu_verifier_tables(self) -> None:
        """Packed exact keys remain visible to canonical-codebook validation."""

        with tempfile.TemporaryDirectory() as tmp:
            include = Path(tmp) / "verifier.inc"
            keys = [
                (5 << 56) | (2 << 48) | (2048 << 24) | 4096,
                (5 << 56) | (4 << 48) | (2048 << 24) | 4096,
            ]
            include.write_text(
                "packCPUNativeVNNIVerifierRowsPolicyKey\n"
                "inline constexpr uint64_t "
                "kCPUNativeVNNIVerifierExactAVX2AVX2T28Keys[] =\n"
                "{\n"
                f"    0x{keys[0]:016x}ULL, 0x{keys[1]:016x}ULL,\n"
                "};\n"
                "inline constexpr uint64_t "
                "kCPUNativeVNNIVerifierExactAVX2AVX2T28WideRowsMask[] =\n"
                "{\n"
                "    0x0000000000000002ULL,\n"
                "};\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(VALIDATOR), str(include)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("validated 1", result.stdout)

    def test_dispatch_validator_rejects_cpu_verifier_abi3_cardinality_mismatch(
        self,
    ) -> None:
        """ABI3 exact overlays require one policy ordinal for every key."""

        with tempfile.TemporaryDirectory() as tmp:
            include = Path(tmp) / "verifier-abi3-cardinality.inc"
            keys = [
                (5 << 56) | (2 << 48) | (2048 << 24) | 4096,
                (5 << 56) | (4 << 48) | (2048 << 24) | 4096,
            ]
            include.write_text(
                "#define LLAMINAR_CPU_NVNNI_VERIFIER_POLICY_ABI 3\n"
                "enum class CPUNativeVNNIVerifierRowsPolicy : uint8_t\n"
                "{\n"
                "    Pairwise = 0,\n"
                "    WideRows = 1,\n"
                "};\n"
                "packCPUNativeVNNIVerifierRowsPolicyKey\n"
                "inline constexpr uint64_t "
                "kCPUNativeVNNIVerifierExactAVX2AVX2T28Keys[] =\n"
                "{\n"
                f"    0x{keys[0]:016x}ULL, 0x{keys[1]:016x}ULL,\n"
                "};\n"
                "inline constexpr uint8_t "
                "kCPUNativeVNNIVerifierExactAVX2AVX2T28Policies[] =\n"
                "{\n"
                "    0,\n"
                "};\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(VALIDATOR), str(include)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("1 policies for 2 keys", result.stderr)

    def test_dispatch_validator_rejects_unknown_cpu_verifier_abi3_policy(
        self,
    ) -> None:
        """ABI3 policy bytes must name a registered generated-policy ordinal."""

        with tempfile.TemporaryDirectory() as tmp:
            include = Path(tmp) / "verifier-abi3-unknown-policy.inc"
            key = (5 << 56) | (2 << 48) | (2048 << 24) | 4096
            include.write_text(
                "#define LLAMINAR_CPU_NVNNI_VERIFIER_POLICY_ABI 3\n"
                "enum class CPUNativeVNNIVerifierRowsPolicy : uint8_t\n"
                "{\n"
                "    Pairwise = 0,\n"
                "    WideRows = 1,\n"
                "};\n"
                "packCPUNativeVNNIVerifierRowsPolicyKey\n"
                "inline constexpr uint64_t "
                "kCPUNativeVNNIVerifierExactAVX2AVX2T28Keys[] =\n"
                "{\n"
                f"    0x{key:016x}ULL,\n"
                "};\n"
                "inline constexpr uint8_t "
                "kCPUNativeVNNIVerifierExactAVX2AVX2T28Policies[] =\n"
                "{\n"
                "    9,\n"
                "};\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(VALIDATOR), str(include)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown policy ordinals [9]", result.stderr)

    def test_dispatch_validator_rejects_unsorted_cpu_verifier_table(self) -> None:
        """Binary-search exact arrays must be strictly sorted and duplicate-free."""

        with tempfile.TemporaryDirectory() as tmp:
            include = Path(tmp) / "verifier-unsorted.inc"
            first = (5 << 56) | (4 << 48) | (2048 << 24) | 4096
            second = (5 << 56) | (2 << 48) | (2048 << 24) | 4096
            include.write_text(
                "packCPUNativeVNNIVerifierRowsPolicyKey\n"
                "inline constexpr uint64_t "
                "kCPUNativeVNNIVerifierExactAVX2AVX2T28Keys[] =\n"
                "{\n"
                f"    0x{first:016x}ULL, 0x{second:016x}ULL,\n"
                "};\n"
                "inline constexpr uint64_t "
                "kCPUNativeVNNIVerifierExactAVX2AVX2T28WideRowsMask[] =\n"
                "{\n"
                "    0x0000000000000000ULL,\n"
                "};\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(VALIDATOR), str(include)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not strictly sorted", result.stderr)

    def test_dispatch_validator_rejects_cpu_verifier_mask_overflow(self) -> None:
        """A policy bit may not address a nonexistent exact-overlay key."""

        with tempfile.TemporaryDirectory() as tmp:
            include = Path(tmp) / "verifier-mask.inc"
            key = (5 << 56) | (2 << 48) | (2048 << 24) | 4096
            include.write_text(
                "packCPUNativeVNNIVerifierRowsPolicyKey\n"
                "inline constexpr uint64_t "
                "kCPUNativeVNNIVerifierExactAVX2AVX2T28Keys[] =\n"
                "{\n"
                f"    0x{key:016x}ULL,\n"
                "};\n"
                "inline constexpr uint64_t "
                "kCPUNativeVNNIVerifierExactAVX2AVX2T28WideRowsMask[] =\n"
                "{\n"
                "    0x0000000000000002ULL,\n"
                "};\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(VALIDATOR), str(include)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sets unused policy bits", result.stderr)

    def test_onednn_build_cache_is_partitioned_by_compiled_cpu_isa(self) -> None:
        cmake = (REPO_ROOT / "src" / "v2" / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn(
            'set(ONEDNN_BUILD_DIR "${ONEDNN_EXTERNAL_DIR}/build-${ONEDNN_ISA_SUFFIX}")',
            cmake,
        )
        self.assertIn("unset(ONEDNN_LIB CACHE)", cmake)
        self.assertIn("build-${ONEDNN_ISA_SUFFIX}", dockerfile)
        self.assertIn(
            "build-$(printf '%s' \"${LLAMINAR_CPU_ISA}\"",
            dockerfile,
        )

    def test_profiler_delta_resume_recovers_completed_and_journaled_work(self) -> None:
        """The transaction driver must never repay a timestamped delta."""

        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("count-uncovered-requests", script)
        self.assertIn(
            "python3 -m native_vnni_dispatch.profiler_checkpoint",
            script,
        )
        self.assertIn('local partial_journal="${partial_evidence}.inprogress.jsonl"', script)
        self.assertIn('--covered-requests "${covered_request}"', script)
        self.assertIn('delta_request_candidates=("${request_prefix}".delta-*', script)
        self.assertIn(
            '-s "${active_evidence_manifest}.inprogress.jsonl"',
            script,
        )
        self.assertIn("cpu_decode_development_run_id_file", script)
        self.assertIn("retained_delta_witnesses", script)

    def test_initial_profiler_transaction_resumes_after_request_publication(
        self,
    ) -> None:
        """An interrupt after request emission must not require final evidence."""

        script = SCRIPT.read_text(encoding="utf-8")
        request_only = (
            'elif [[ -s "${request_manifest}" && '
            '! -s "${evidence_manifest}" ]]; then'
        )
        completed_or_partial = (
            'elif [[ -s "${request_manifest}" || '
            '-s "${evidence_manifest}" ]]; then'
        )
        self.assertIn(request_only, script)
        self.assertLess(script.index(request_only), script.index(completed_or_partial))
        self.assertIn(
            '-s "${active_evidence_manifest}.inprogress.jsonl"', script
        )
        self.assertIn(
            "native-vnni-profiler-request-v5-stratified-exact-point", script
        )
        self.assertIn("cuda_exact_profiler_transaction=1", script)
        self.assertIn(
            "Resuming current CUDA exact-point profiler transaction", script
        )

    def test_custom_m_values_are_forwarded_to_cuda_and_rocm(self) -> None:
        result = self.run_script("--backend", "all", "--m-values", "2,4")

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_M=2,4", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_M=2,4", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_M=2,4", stdout)

    def test_install_copies_generated_backend_artifacts(self) -> None:
        result = self.run_script(
            "--backend", "all", "--profile", "all", "--install",
            "--cuda-measurement-lanes", "2",
            "--rocm-measurement-lanes", "4",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CUDANativeVNNIGemvDispatchHeuristicGenerated.inc", result.stdout)
        self.assertIn("ROCmNativeVNNIDecodeDispatchGenerated.inc", result.stdout)
        self.assertIn("CPUNativeVNNIVerifierRowsPolicyGenerated.inc", result.stdout)
        self.assertNotIn("CPUNativeVNNIPrefillPolicyGenerated.inc", result.stdout)
        self.assertIn("src/v2/kernels/cuda/gemm", result.stdout)
        self.assertIn("src/v2/kernels/rocm/gemm", result.stdout)
        self.assertIn("src/v2/kernels/cpu/gemm", result.stdout)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn(
            "LLAMINAR_CPU_NVNNI_VERIFIER_SHAPE_NAME=32B_LM_Head",
            stdout,
        )
        self.assertIn(
            "LLAMINAR_CPU_NVNNI_VERIFIER_SHAPE_NAME=Qwen36_LM_Head",
            stdout,
        )
        self.assertIn(
            "native_vnni_dispatch.profiler_evidence emit-requests",
            result.stdout,
        )
        self.assertEqual(
            result.stdout.count("native_vnni_dispatch.profiler_collectors"),
            6,
        )
        self.assertEqual(
            result.stdout.count(
                "native_vnni_dispatch.profiler_evidence export-features"
            ),
            6,
        )
        self.assertEqual(
            result.stdout.count(
                "native_vnni_dispatch.profiler_evidence emit-requests"
            ),
            6,
        )
        self.assertEqual(
            result.stdout.count(
                "native_vnni_dispatch.profiler_evidence compact-witnesses"
            ),
            6,
        )
        self.assertIn("--backend cuda", result.stdout)
        self.assertIn("--backend rocm", result.stdout)
        self.assertIn("--device-lanes 2", result.stdout)
        self.assertIn("--device-lanes 4", result.stdout)
        self.assertIn("--backend cpu", result.stdout)
        self.assertIn("--cpu-avx2-binary /bin/true", result.stdout)
        self.assertIn("--cpu-avx512-binary /bin/true", result.stdout)
        self.assertIn("--finalize", result.stdout)
        self.assertIn("cuda_profiler_features.csv", result.stdout)
        self.assertIn("rocm_profiler_features.csv", result.stdout)
        self.assertIn("cpu_profiler_features.csv", result.stdout)
        self.assertIn("--observation", result.stdout)

    def test_production_refresh_exports_explicit_gpu_policy_workers(self) -> None:
        result = self.run_script(
            "--backend",
            "cuda",
            "--profile",
            "all",
            "--policy-accelerators",
            "cuda:0,rocm:0",
            "--policy-lanes",
            "3",
            "--policy-cuda-scorer",
            "/bin/true",
            "--policy-rocm-scorer",
            "/bin/true",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "NativeVNNI policy fitter: rebuilding scorer targets="
            "v2_native_vnni_leaf_primary_scorer_cuda,"
            "v2_native_vnni_leaf_primary_scorer_rocm",
            result.stdout,
        )
        self.assertIn(
            "--target v2_native_vnni_leaf_primary_scorer_cuda "
            "v2_native_vnni_leaf_primary_scorer_rocm",
            result.stdout,
        )
        self.assertIn(
            "NativeVNNI policy fitter: accelerators=cuda:0,rocm:0 "
            "lanes-per-device=3",
            result.stdout,
        )
        self.assertIn("native_vnni_dispatch.paired_requests", result.stdout)
        self.assertIn("--fit-cache-dir", result.stdout)
        self.assertIn("cuda_paired_refinement/fit-cache", result.stdout)

    def test_production_refresh_can_explicitly_select_cpu_policy_fit(self) -> None:
        result = self.run_script(
            "--backend",
            "cuda",
            "--profile",
            "all",
            "--policy-accelerators",
            "cpu",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "NativeVNNI policy fitter: canonical CPU process pool",
            result.stdout,
        )

    def test_install_cannot_skip_per_candidate_profiler_evidence(self) -> None:
        result = self.run_script(
            "--backend",
            "cuda",
            "--profile",
            "all",
            "--install",
            "--skip-profiler-evidence",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires complete isolated profiler evidence", result.stderr)

    def test_smoke_profile_cannot_install_any_production_policy(self) -> None:
        result = self.run_script("--backend", "cpu", "--install")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("installing NativeVNNI dispatch requires", result.stderr)

    def test_install_requires_complete_runtime_m_envelope(self) -> None:
        result = self.run_script(
            "--backend",
            "all",
            "--profile",
            "all",
            "--m-values",
            "1,2,3,4,16",
            "--install",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("complete M=1,2..16,31 matrix", result.stderr)

    def test_family_smoke_is_stratified_by_format(self) -> None:
        result = self.run_script(
            "--backend",
            "all",
            "--profile",
            "family-smoke",
            "--cuda-formats",
            "Q4_0,IQ4_XS",
            "--rocm-formats",
            "Q4_0,IQ4_XS",
            "--cpu-formats",
            "Q4_0,IQ4_XS",
            "--m-values",
            "1,2",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_FORMATS=Q4_0", stdout)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_FORMATS=IQ4_XS", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=Q4_0", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=IQ4_XS", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS=Q4_0", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS=IQ4_XS", stdout)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_M=1,2", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_M=1,2", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_M=2", stdout)
        self.assertNotIn("LLAMINAR_CPU_NVNNI_PREFILL_M=", stdout)
        self.assertNotIn("PREFILL_SHAPE_NAME=", stdout)
        self.assertIn("combine-csv", stdout)
        self.assertIn("cuda_decode_sweep.Q4_0.csv", stdout)
        self.assertIn("cuda_decode_sweep.IQ4_XS.csv", stdout)
        self.assertIn("rocm_decode_sweep.Q4_0.csv", stdout)
        self.assertIn("rocm_decode_sweep.IQ4_XS.csv", stdout)
        self.assertIn("cpu_verifier_rows.Q4_0.", stdout)
        self.assertIn("cpu_verifier_rows.IQ4_XS.", stdout)

    def test_default_family_smoke_covers_full_format_inventory(self) -> None:
        result = self.run_script("--backend", "all", "--profile", "family-smoke")

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_FORMATS=Q4_0", stdout)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_FORMATS=IQ1_M", stdout)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_FORMATS=Q8_0", stdout)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_FORMATS=Q8_1", stdout)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_FORMATS=Q8_K", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=Q4_0", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=IQ1_M", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=Q8_0", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=Q8_1", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=Q8_K", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS=Q4_0", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS=IQ1_M", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS=Q8_0", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS=Q8_1", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS=Q8_K", stdout)
        self.assertIn("cuda_decode_sweep.Q8_0.csv", stdout)
        self.assertIn("rocm_decode_sweep.IQ1_M.csv", stdout)
        self.assertIn("cpu_verifier_rows.Q8_1.", stdout)

    def test_cuda_production_uses_strong_samples_and_complete_matrix(self) -> None:
        smoke = self.run_script("--backend", "cuda", "--profile", "family-smoke")
        strict = self.run_script("--backend", "cuda", "--profile", "all")

        self.assertEqual(smoke.returncode, 0, smoke.stderr)
        self.assertEqual(strict.returncode, 0, strict.stderr)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_WARMUPS=2", smoke.stdout)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_SAMPLES=5", smoke.stdout)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_TIMED_REPLAYS=4", smoke.stdout)
        self.assertNotIn("--require-complete", smoke.stdout)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_WARMUPS=5", strict.stdout)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_SAMPLES=30", strict.stdout)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_TIMED_REPLAYS=16", strict.stdout)
        self.assertIn("--profile production", strict.stdout)
        self.assertIn("--require-complete", strict.stdout)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_M=1", strict.stdout)
        self.assertIn(
            "LLAMINAR_CUDA_NVNNI_DECODE_M=2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31",
            strict.stdout.replace("\\,", ","),
        )
        self.assertIn("--require-fast-m1-complete", strict.stdout)
        self.assertIn("--verifier-input", strict.stdout)
        self.assertNotIn("--m1-input", strict.stdout)
        self.assertNotIn("--m1-build-id", strict.stdout)
        self.assertNotIn("--m1-baseline-policy-hash", strict.stdout)
        self.assertIn("--freeze-generic", strict.stdout)
        self.assertIn("--certify-generic", strict.stdout)
        self.assertIn("--development-input", strict.stdout)
        self.assertIn("--sealed-input", strict.stdout)
        self.assertIn("--frozen-policy-json", strict.stdout)
        self.assertIn("--certified-m1-include", strict.stdout)
        self.assertIn("--certified-m1-policy-json", strict.stdout)
        self.assertNotIn("--frozen-m1-include", strict.stdout)
        self.assertIn("--shape-manifest", strict.stdout)
        self.assertIn("--policy-json", strict.stdout)
        self.assertIn("--exact-only", strict.stdout)
        self.assertIn("native_vnni_dispatch.paired_requests", strict.stdout)
        self.assertIn(
            "LLAMINAR_CUDA_NVNNI_DECODE_PAIRED_REQUEST_MANIFEST=",
            strict.stdout,
        )
        self.assertIn("--paired-development-csv", strict.stdout)
        self.assertNotIn(
            "cmp --silent ",
            "\n".join(
                line
                for line in strict.stdout.splitlines()
                if "CUDANativeVNNIGemvDispatchHeuristicGenerated" in line
            ),
        )
        self.assertIn("Integration_Balanced_192x256", strict.stdout)
        self.assertIn("Integration_Tall_128x256", strict.stdout)

        development = strict.stdout.index("cuda_decode_m1.development.csv")
        frozen = strict.stdout.index("--freeze-generic")
        sealed = strict.stdout.index("cuda_decode_m1.sealed.csv")
        certified = strict.stdout.index("--certify-generic")
        verifier_development = strict.stdout.index(
            "cuda_decode_verifier.development.csv"
        )
        verifier_sealed = strict.stdout.index("cuda_decode_verifier.sealed.csv")
        self.assertLess(development, frozen)
        self.assertLess(frozen, sealed)
        self.assertLess(sealed, certified)
        self.assertLess(certified, verifier_development)
        self.assertLess(verifier_development, verifier_sealed)

    def test_cuda_production_skip_sweep_requires_complete_fit_corpus(self) -> None:
        """Fit-only replay cannot synthesize a development corpus from nothing."""

        result = self.run_script(
            "--backend", "cuda", "--profile", "all", "--skip-sweep"
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CUDA fit-only corpus is missing", result.stderr)
        self.assertIn("cuda_decode_m1.development.csv", result.stderr)

    def test_cpu_production_freezes_before_paired_seal_and_installing(self) -> None:
        """CPU publication must use fresh post-freeze paired certificates."""

        result = self.run_script(
            "--backend", "cpu", "--profile", "all", "--install"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout
        self.assertIn("--freeze-generic", stdout)
        self.assertIn("--certify-generic", stdout)
        self.assertEqual(stdout.count("--reuse-development-common"), 5)
        self.assertIn("--development-input", stdout)
        self.assertNotIn("--sealed-input", stdout)
        self.assertIn("--sealed-plan-json", stdout)
        self.assertIn("--sealed-paired-dir", stdout)
        self.assertIn("--require-inventory-formats", stdout)
        self.assertIn("--require-inventory-shapes", stdout)
        self.assertIn("--require-inventory-m-values", stdout)
        self.assertNotIn("--require-key ", stdout)
        self.assertIn("--sealed-route-probe-json", stdout)
        self.assertIn(".dry-run-probe.csv.inprogress", stdout)
        self.assertIn(
            "dry-run: collect CPU grouped-decode sealed paired shards",
            stdout,
        )
        self.assertIn("--frozen-policy-json", stdout)
        self.assertIn("--certification-diagnostic", stdout)
        self.assertIn("cpu_decode_m1_certification_diagnostic.json", stdout)
        self.assertIn("native_vnni_dispatch.policy_artifact", stdout)
        self.assertIn("cpu_verifier_rows_policy.json", stdout)
        self.assertIn("CPUNativeVNNIVerifierRowsPolicyGenerated.inc.inprogress", stdout)

        verifier = stdout.index("analyze_cpu_native_vnni_verifier_trainer.py")
        frozen = stdout.index("--freeze-generic", verifier)
        sealed = stdout.index(
            "dry-run: collect CPU grouped-decode sealed paired shards",
            frozen,
        )
        certified = stdout.index("--certify-generic", sealed)
        install_gate = stdout.index(
            "native_vnni_dispatch.policy_artifact", certified
        )
        install_copy = stdout.index(
            "CPUNativeVNNIVerifierRowsPolicyGenerated.inc.inprogress"
        )
        self.assertLess(frozen, sealed)
        self.assertLess(sealed, certified)
        self.assertLess(certified, install_gate)
        self.assertLess(install_gate, install_copy)

    def test_cpu_grouped_best_effort_leaf_budget_is_transaction_wide(self) -> None:
        """Planning, freeze, and certification must use one explicit budget."""

        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "all",
            "--install",
            "--cpu-grouped-max-leaves",
            "1",
            "--minimum-passing-domain-percent",
            "0",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-leaves 1", result.stdout)
        self.assertIn("--max-regret 0.05", result.stdout)
        self.assertEqual(result.stdout.count("--generic-max-leaves 1"), 2)

        invalid = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "all",
            "--cpu-grouped-max-leaves",
            "0",
        )
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn(
            "--cpu-grouped-max-leaves must be in [1, 32]",
            invalid.stderr,
        )

    def test_cpu_burned_seal_options_require_positional_plan_pairs(self) -> None:
        """A burned generation is meaningless without both authenticated halves."""

        decode = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "all",
            "--cpu-decode-burned-sealed-plan",
            "/fixture/decode-plan.json",
        )
        grouped = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "all",
            "--cpu-grouped-burned-sealed-paired-dir",
            "/fixture/grouped-pairs",
        )

        self.assertNotEqual(decode.returncode, 0)
        self.assertIn(
            "burned CPU decode plans and paired directories must pair",
            decode.stderr,
        )
        self.assertNotEqual(grouped.returncode, 0)
        self.assertIn(
            "burned CPU grouped plans and paired directories must pair",
            grouped.stderr,
        )

    def test_cpu_burned_seals_reach_every_generic_generation_step(self) -> None:
        """Freeze, fresh-seal planning, and certification share burned costs."""

        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "all",
            "--install",
            "--cpu-decode-burned-sealed-plan",
            "/fixture/decode-plan.json",
            "--cpu-decode-burned-sealed-paired-dir",
            "/fixture/decode-pairs",
            "--cpu-grouped-burned-sealed-plan",
            "/fixture/grouped-plan.json",
            "--cpu-grouped-burned-sealed-paired-dir",
            "/fixture/grouped-pairs",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("/fixture/decode-plan.json"), 4)
        self.assertEqual(result.stdout.count("/fixture/decode-pairs"), 4)
        self.assertEqual(result.stdout.count("/fixture/grouped-plan.json"), 3)
        self.assertEqual(result.stdout.count("/fixture/grouped-pairs"), 3)
        self.assertIn("--surface grouped-verifier", result.stdout)
        self.assertIn("policy_fit_cache/cpu_grouped", result.stdout)
        self.assertIn(
            "cpu_verifier_rows_certification_diagnostic.json",
            result.stdout,
        )

    def test_partitioned_cuda_fast_sweep_is_m1_measurement_only(self) -> None:
        """Development evidence cannot accidentally open the Fast holdout."""

        result = self.run_script(
            "--backend",
            "cuda",
            "--profile",
            "all",
            "--shape-partition",
            "fast-development",
            "--cuda-formats",
            "Q4_0",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_M=1", stdout)
        self.assertIn("V4VerifierSealed_Tall_192x416", stdout)
        self.assertNotIn("V4FastSealed_Tall_160x288", stdout)
        self.assertIn("cuda_decode_fast-development.csv", stdout)
        self.assertNotIn("analyze_cuda_native_vnni_decode_trainer.py", stdout)

    def test_partitioned_cuda_holdouts_publish_canonical_resume_artifacts(
        self,
    ) -> None:
        """Independent CUDA phases flow directly into fit-only resume paths."""

        expectations = {
            "fast-sealed": "cuda_decode_m1.sealed.csv",
            "verifier-development": "cuda_decode_verifier.development.csv",
            "verifier-sealed": "cuda_decode_verifier.sealed.csv",
        }
        for partition, expected in expectations.items():
            with self.subTest(partition=partition):
                result = self.run_script(
                    "--backend",
                    "cuda",
                    "--profile",
                    "all",
                    "--shape-partition",
                    partition,
                    "--cuda-formats",
                    "Q4_0",
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected, result.stdout)
                self.assertNotIn(
                    f"cuda_decode_{partition}.csv",
                    result.stdout,
                )

    def test_partitioned_cuda_verifier_seal_composes_fit_input(self) -> None:
        """Completing the verifier holdout publishes the canonical aggregate."""

        result = self.run_script(
            "--backend",
            "cuda",
            "--profile",
            "all",
            "--shape-partition",
            "verifier-sealed",
            "--cuda-formats",
            "Q4_0",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("combine-csv", result.stdout)
        self.assertIn("cuda_decode_verifier.csv", result.stdout)
        self.assertIn("cuda_decode_verifier.development.csv", result.stdout)
        self.assertIn("cuda_decode_verifier.sealed.csv", result.stdout)

    def test_partitioned_rocm_verifier_sweep_uses_complete_runtime_m(self) -> None:
        """Verifier holdout collection excludes M1 and remains measurement-only."""

        result = self.run_script(
            "--backend",
            "rocm",
            "--profile",
            "all",
            "--shape-partition",
            "verifier-sealed",
            "--rocm-formats",
            "Q4_0",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn(
            "LLAMINAR_ROCM_NVNNI_DECODE_M="
            "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31",
            stdout,
        )
        self.assertIn("V4VerifierSealed_Tall_192x416", stdout)
        self.assertNotIn("V4FastSealed_Tall_160x288", stdout)
        self.assertNotIn("analyze_rocm_native_vnni_decode_trainer.py", stdout)

    def test_full_rocm_refresh_excludes_cuda_q4_fast_refinement(self) -> None:
        """CUDA/Q4-only refinement geometry must never leak into ROCm."""

        result = self.run_script(
            "--backend",
            "rocm",
            "--profile",
            "all",
            "--rocm-formats",
            "Q4_0",
            "--skip-profiler-evidence",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        fast_line = next(
            line for line in stdout.splitlines()
            if "LLAMINAR_ROCM_NVNNI_DECODE_CSV=" in line
            and "rocm_decode_fast.development.csv" in line
        )
        verifier_line = next(
            line for line in stdout.splitlines()
            if "LLAMINAR_ROCM_NVNNI_DECODE_CSV=" in line
            and "rocm_decode_verifier.csv" in line
        )
        fast_only = "FastM1RefineV5_3B_FFN_Dn_Nm128_1920x11008"
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_M=1", fast_line)
        self.assertNotIn(fast_only, fast_line)
        self.assertIn(
            "LLAMINAR_ROCM_NVNNI_DECODE_M="
            "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31",
            verifier_line,
        )
        self.assertNotIn(fast_only, verifier_line)
        self.assertIn("combine-csv", stdout)

    def test_full_cpu_refresh_collects_decode_before_grouped_verifier(self) -> None:
        """CPU owns independent Fast-M1 and grouped-verifier transactions."""

        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "all",
            "--cpu-formats",
            "Q4_0",
            "--install",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        trainer_lines = [
            line for line in stdout.splitlines()
            if "LLAMINAR_CPU_NVNNI_VERIFIER_M=" in line
        ]
        self.assertTrue(trainer_lines)
        expected_m = (
            "LLAMINAR_CPU_NVNNI_VERIFIER_M="
            "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31"
        )
        self.assertTrue(all(expected_m in line for line in trainer_lines))
        self.assertIn("LLAMINAR_CPU_NVNNI_DECODE_M=1", stdout)
        self.assertIn(
            "FastM1RefineV5_3B_FFN_Dn_Nm128_1920x11008",
            stdout,
        )
        self.assertIn("V4VerifierSealed_Tall_192x416", stdout)
        self.assertLess(
            stdout.index("TrainerCsv_StrongDecode_AllFormats"),
            stdout.index("TrainerCsv_StrongVerifierRows_AllFormats"),
        )

    def test_cpu_fast_partition_runs_only_cpu_decode_trainer(self) -> None:
        """A Fast partition is isolated from grouped-verifier collection."""

        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "all",
            "--shape-partition",
            "fast-development",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LLAMINAR_CPU_NVNNI_DECODE_M=1", result.stdout)
        self.assertIn("TrainerCsv_StrongDecode_AllFormats", result.stdout)
        self.assertNotIn("LLAMINAR_CPU_NVNNI_VERIFIER_M=", result.stdout)

    def test_cpu_decode_checkpoint_installs_and_stops_before_grouped(self) -> None:
        """The certified M=1 prerequisite can be completed independently."""

        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "all",
            "--cpu-formats",
            "Q4_0",
            "--install",
            "--stop-after-cpu-decode",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn("LLAMINAR_CPU_NVNNI_DECODE_M=1", stdout)
        self.assertIn("--certify-generic", stdout)
        self.assertIn("CPUNativeVNNIDecodePolicyGenerated.inc.inprogress", stdout)
        self.assertIn("--target v2_perf_cpu_native_vnni_gemv", stdout)
        self.assertNotIn("LLAMINAR_CPU_NVNNI_VERIFIER_M=", stdout)

    def test_cpu_decode_checkpoint_requires_certified_install_transaction(self) -> None:
        """Stopping after decode cannot leave an uninstalled prerequisite."""

        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "all",
            "--stop-after-cpu-decode",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--stop-after-cpu-decode requires --install", result.stderr)

    def test_cpu_grouped_resume_authenticates_and_bypasses_decode(self) -> None:
        """A completed M=1 transaction resumes without mutating its evidence."""

        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "all",
            "--cpu-formats",
            "Q4_0",
            "--install",
            "--resume-after-cpu-decode",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertNotIn("LLAMINAR_CPU_NVNNI_DECODE_M=1", stdout)
        self.assertNotIn("TrainerCsv_StrongDecode_AllFormats", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_M=2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31", stdout)
        self.assertIn("native_vnni_dispatch.policy_artifact", stdout)
        self.assertIn("cmp --silent", stdout)
        self.assertIn("Authenticated installed CPU decode prerequisite", stdout)

    def test_cpu_grouped_resume_requires_installed_production_contract(self) -> None:
        """The grouped continuation cannot bless an uninstalled M=1 table."""

        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "all",
            "--resume-after-cpu-decode",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--resume-after-cpu-decode requires --install", result.stderr)

    def test_cpu_decode_batch_limit_checkpoints_before_fitting(self) -> None:
        """A bounded M=1 run publishes partials and remains resumable."""

        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "all",
            "--shape-partition",
            "fast-development",
            "--cpu-formats",
            "Q4_0",
            "--cpu-batch-limit",
            "1",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "CPU M=1 collection checkpointed after 1 pending batch(es)",
            result.stdout,
        )
        self.assertEqual(
            result.stdout.count("TrainerCsv_StrongDecode_AllFormats"),
            2,
        )
        self.assertNotIn("analyze_cpu_native_vnni_decode_trainer.py", result.stdout)
        self.assertNotIn("LLAMINAR_CPU_NVNNI_VERIFIER_M=", result.stdout)

    def test_obsolete_cuda_decode_generators_are_retired(self) -> None:
        """The common analyzer is the sole CUDA decode policy authority."""

        gemm = (
            REPO_ROOT
            / "tests"
            / "v2"
            / "performance"
            / "kernels"
            / "cuda"
            / "gemm"
        )
        for name in (
            "infer_gemv_dispatch_heuristic.py",
            "analyze_cuda_tc_gemv_dispatch.py",
            "validate_cuda_gemv_dispatch_generator.py",
            "validate_cuda_gemv_dispatch_base_merge.py",
        ):
            self.assertFalse((gemm / name).exists(), name)

    def test_qwen36_profiles_split_lm_head_without_changing_full_profile(self) -> None:
        core = self.run_script("--backend", "rocm", "--profile", "qwen36-core")
        lm_head = self.run_script("--backend", "rocm", "--profile", "qwen36-lm-head")
        moe = self.run_script("--backend", "rocm", "--profile", "qwen36-moe")
        full = self.run_script("--backend", "rocm", "--profile", "qwen36")
        cuda_core = self.run_script("--backend", "cuda", "--profile", "qwen36-core")

        self.assertEqual(core.returncode, 0, core.stderr)
        self.assertEqual(lm_head.returncode, 0, lm_head.stderr)
        self.assertEqual(moe.returncode, 0, moe.stderr)
        self.assertEqual(full.returncode, 0, full.stderr)
        self.assertEqual(cuda_core.returncode, 0, cuda_core.stderr)

        core_stdout = core.stdout.replace("\\,", ",")
        lm_stdout = lm_head.stdout.replace("\\,", ",")
        moe_stdout = moe.stdout.replace("\\,", ",")
        full_stdout = full.stdout.replace("\\,", ",")

        self.assertIn("Qwen36_FFN_GateUp", core_stdout)
        self.assertIn("Qwen36_GDN_OutputProjection", core_stdout)
        self.assertNotIn("Qwen36_LM_Head", core_stdout)

        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_SHAPES=Qwen36_LM_Head", lm_stdout)
        self.assertIn(
            "LLAMINAR_ROCM_NVNNI_DECODE_EXECUTION_MODES=eager,graph_captured",
            lm_stdout,
        )
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_TIMING_CSV=", lm_stdout)
        self.assertNotIn("Qwen36_FFN_GateUp", lm_stdout)

        self.assertIn(
            "LLAMINAR_ROCM_NVNNI_DECODE_SHAPES=35BMoE_Expert_GateUp,35BMoE_Expert_Down,"
            "Qwen36MoE_GDN_QKVProjection,Qwen36MoE_GDN_ZProjection",
            moe_stdout,
        )
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=", moe_stdout)
        self.assertNotIn("LLAMINAR_ROCM_NVNNI_DECODE_REFERENCE", moe_stdout)
        self.assertIn("--base-include", moe_stdout)
        self.assertIn("ROCmNativeVNNIDecodeDispatchGenerated.inc", moe_stdout)
        self.assertNotIn("Qwen36_LM_Head", moe_stdout)

        self.assertIn("Qwen36_FFN_GateUp", full_stdout)
        self.assertIn("Qwen36_LM_Head", full_stdout)
        self.assertIn("35BMoE_Expert_GateUp", full_stdout)
        self.assertNotIn("LLAMINAR_ROCM_NVNNI_DECODE_REFERENCE", core_stdout)
        self.assertIn("--profile partial-production", cuda_core.stdout)
        self.assertIn("--require-complete", cuda_core.stdout)

    def test_cuda_qwen36_moe_profile_uses_real_expert_buckets(self) -> None:
        result = self.run_script(
            "--backend",
            "cuda",
            "--profile",
            "qwen36-moe",
            "--cuda-formats",
            "Q4_K,Q6_K",
            "--m-values",
            "2,3,4",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn(
            "LLAMINAR_CUDA_NVNNI_DECODE_SHAPES=35BMoE_Expert_GateUp,35BMoE_Expert_Down,"
            "Qwen36MoE_GDN_QKVProjection,Qwen36MoE_GDN_ZProjection",
            stdout,
        )
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_M=2,3,4", stdout)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_FORMATS=Q4_K,Q6_K", stdout)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_CANDIDATES=", stdout)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_WARMUPS=5", stdout)
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_SAMPLES=30", stdout)
        self.assertIn("cuda_decode_sweep.csv", stdout)
        self.assertIn("CUDANativeVNNIGemvDispatchHeuristicGenerated.inc", stdout)

    def test_cpu_backend_emits_verifier_policy_refresh_contract(self) -> None:
        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "quick",
            "--cpu-formats",
            "Q4_K,Q6_K",
            "--m-values",
            "2,3,4",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn("LLAMINAR_CPU_NVNNI_DECODE_FORMATS=Q4_K,Q6_K", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_DECODE_M=1", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_DECODE_STRONG_CSV=", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_DECODE_TIMING_CSV=", stdout)
        self.assertIn("TrainerCsv_StrongDecode_AllFormats", stdout)
        self.assertIn("analyze_cpu_native_vnni_decode_trainer.py", stdout)
        self.assertIn("CPUNativeVNNIDecodePolicyGenerated.inc", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS=Q4_K,Q6_K", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_M=2,3,4", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_SHAPE_NAME=Qwen36_FFN_DownProjection", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_SHAPE_NAME=Qwen36_GDN_OutputProjection", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_N=5120", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_K=17408", stdout)
        self.assertIn("LLAMINAR_ISA_LEVEL=avx2", stdout)
        self.assertIn("LLAMINAR_ISA_LEVEL=avx512", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_STRONG_CSV=", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_TIMING_CSV=", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_THREADS=28", stdout)
        self.assertIn("OMP_NUM_THREADS=28", stdout)
        self.assertIn("OMP_PLACES=cores", stdout)
        self.assertIn("OMP_PROC_BIND=close", stdout)
        self.assertIn("OMP_DYNAMIC=false", stdout)
        self.assertIn("mpirun --bind-to socket --map-by socket", stdout)
        self.assertIn("TrainerCsv_StrongVerifierRows_AllFormats", stdout)
        self.assertIn("avx2-build.avx2-runtime", stdout)
        self.assertIn("avx512-build.avx2-runtime", stdout)
        self.assertIn("avx512-build.avx512-runtime", stdout)
        self.assertIn("analyze_cpu_native_vnni_verifier_trainer.py", stdout)
        self.assertIn("--require-isa-matrix", stdout)
        self.assertIn("CPUNativeVNNIVerifierRowsPolicyGenerated.inc", stdout)
        self.assertIn("validate_native_vnni_generated_dispatch_ids.py", stdout)

    def test_cpu_grouped_verifier_rejects_an_m1_only_inventory(self) -> None:
        """M=1 belongs to serial decode and cannot become grouped evidence."""

        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "quick",
            "--m-values",
            "1",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "CPU grouped-verifier collection requires at least one "
            "--m-values entry >= 2",
            result.stderr,
        )

    def test_cpu_two_socket_mpi_batch_assigns_distinct_jobs(self) -> None:
        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "quick",
            "--cpu-formats",
            "Q4_K",
            "--m-values",
            "2,3,4",
            "--cpu-measurement-lanes",
            "2",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        mpi_lines = [
            line.replace("\\,", ",")
            for line in result.stdout.splitlines()
            if "mpirun --bind-to socket --map-by socket" in line
            and "LLAMINAR_CPU_NVNNI_VERIFIER_STRONG_CSV=" in line
        ]
        self.assertTrue(mpi_lines)
        first = mpi_lines[0]
        self.assertEqual(first.count(" -np 1 env "), 2)
        self.assertIn("Qwen36_FFN_DownProjection", first)
        self.assertIn("Qwen36_GDN_OutputProjection", first)
        self.assertIn("avx2-build.avx2-runtime.csv", first)
        self.assertTrue(
            any("avx512-build.avx2-runtime.csv" in line for line in mpi_lines)
        )
        self.assertTrue(
            any("avx512-build.avx512-runtime.csv" in line for line in mpi_lines)
        )
        csv_paths = re.findall(
            r"LLAMINAR_CPU_NVNNI_VERIFIER_STRONG_CSV=([^ ]+)", first
        )
        self.assertEqual(len(csv_paths), 2)
        self.assertEqual(len(set(csv_paths)), 2)

    def test_cpu_measurement_lanes_default_to_detected_sockets(self) -> None:
        """Turnkey CPU collection must not silently leave sockets idle."""

        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "quick",
            "--cpu-formats",
            "Q4_K",
            "--m-values",
            "2,3,4",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        socket_rows = subprocess.run(
            ["lscpu", "-p=socket"],
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout.splitlines()
        socket_count = len({
            row for row in socket_rows if row and not row.startswith("#")
        })
        first = next(
            line.replace("\\,", ",")
            for line in result.stdout.splitlines()
            if "mpirun --bind-to socket --map-by socket" in line
            and "LLAMINAR_CPU_NVNNI_VERIFIER_STRONG_CSV=" in line
        )
        self.assertEqual(first.count(" -np 1 env "), socket_count)
        csv_paths = re.findall(
            r"LLAMINAR_CPU_NVNNI_VERIFIER_STRONG_CSV=([^ ]+)", first
        )
        self.assertEqual(len(csv_paths), socket_count)
        self.assertEqual(len(set(csv_paths)), socket_count)

    def test_cpu_bounded_collection_checkpoints_atomic_format_shards(self) -> None:
        """A bounded invocation leaves resumable finals, never partial CSVs."""

        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "quick",
            "--cpu-formats",
            "Q4_K,Q6_K",
            "--m-values",
            "2,3,4",
            "--cpu-measurement-lanes",
            "2",
            "--cpu-format-shards",
            "--cpu-batch-limit",
            "1",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn(".csv.inprogress", stdout)
        self.assertIn("dry-run: mv", stdout)
        self.assertIn("CPU M=1 collection checkpointed after 1", stdout)
        self.assertIn("resume from", stdout)
        self.assertNotIn("analyze_cpu_native_vnni_verifier_trainer.py", stdout)
        mpi_lines = [
            line for line in stdout.splitlines()
            if "mpirun --bind-to socket --map-by socket" in line
        ]
        self.assertEqual(len(mpi_lines), 1)
        self.assertIn("DECODE_FORMATS=Q4_K", mpi_lines[0])
        self.assertNotIn("DECODE_FORMATS=Q4_K,Q6_K", mpi_lines[0])

    def test_cpu_batch_limit_rejects_negative_values(self) -> None:
        result = self.run_script(
            "--backend",
            "cpu",
            "--cpu-batch-limit",
            "-1",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("non-negative integer", result.stderr)

    def test_cpu_resume_fails_closed_on_collection_contract_change(self) -> None:
        """Completed shard names cannot be reused with different M or formats."""

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            output.mkdir()
            (output / "cpu_collection_contract.sha256").write_text(
                "stale-contract\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    str(SCRIPT),
                    "--backend",
                    "cpu",
                    "--profile",
                    "all",
                    "--output-dir",
                    str(output),
                    "--cpu-avx2-sweep-bin",
                    "/bin/true",
                    "--cpu-avx512-sweep-bin",
                    "/bin/true",
                    "--cpu-threads",
                    "28",
                    "--cpu-formats",
                    "Q4_K",
                    "--m-values",
                    "2",
                    "--shape-partition",
                    "verifier-development",
                    "--resume-cpu-partials",
                    "--skip-profiler-evidence",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("collection contract changed", result.stderr)

    def test_cpu_prefill_fails_before_collection_when_isa_binary_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    str(SCRIPT),
                    "--backend",
                    "cpu-prefill",
                    "--profile",
                    "quick",
                    "--output-dir",
                    str(Path(tmp) / "out"),
                    "--cpu-avx2-sweep-bin",
                    "/bin/true",
                    "--cpu-avx512-sweep-bin",
                    "/bin/true",
                    "--cpu-threads",
                    "28",
                    "--skip-profiler-evidence",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("does not contain required gtest", result.stderr)
        self.assertIn("rebuild the matching ISA trainer", result.stderr)

    def test_cpu_prefill_backend_uses_tiered_matrix_and_generator(self) -> None:
        result = self.run_script(
            "--backend",
            "cpu-prefill",
            "--profile",
            "quick",
            "--cpu-formats",
            "Q4_K",
            "--cpu-measurement-lanes",
            "2",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn("LLAMINAR_CPU_NVNNI_PREFILL_M=32", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_PREFILL_SHAPE_NAME=7B_FFN_Up", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_PREFILL_SHAPE_NAME=0.5B_AttnOut", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_PREFILL_MIN_ITERS=5", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_PREFILL_MAX_ITERS=180", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_PREFILL_WARMUP_BUDGET_US=0", stdout)
        self.assertIn(
            "LLAMINAR_CPU_NVNNI_PREFILL_TRANSITION_WARMUP_BUDGET_US=0",
            stdout,
        )
        self.assertIn(
            "LLAMINAR_CPU_NVNNI_PREFILL_TRANSITION_WARMUP_LATENCY_MULTIPLIER=60",
            stdout,
        )
        self.assertIn(
            "LLAMINAR_CPU_NVNNI_PREFILL_WARMUP_ROUND_TIMEOUT_US=30000000",
            stdout,
        )
        self.assertIn("LLAMINAR_CPU_NVNNI_PREFILL_TIMING_BUDGET_US=100000", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_PREFILL_MEDIAN_STABILITY=0.02", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_PREFILL_MPI_ROUND_SYNC=0", stdout)
        self.assertIn("TrainerCsv_StrongPrefill_AllFormats", stdout)
        self.assertIn("analyze_cpu_native_vnni_prefill_trainer.py", stdout)
        self.assertIn("CPUNativeVNNIPrefillPolicyGenerated.inc", stdout)
        self.assertIn("validate_native_vnni_generated_dispatch_ids.py", stdout)

    def test_cpu_prefill_production_checkpoint_uses_exhaustive_plan(self) -> None:
        result = self.run_script(
            "--backend",
            "cpu-prefill",
            "--profile",
            "all",
            "--cpu-formats",
            "Q4_K",
            "--cpu-measurement-lanes",
            "2",
            "--cpu-batch-limit",
            "1",
            "--skip-profiler-evidence",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn(
            "LLAMINAR_CPU_NVNNI_PREFILL_M=32,64",
            stdout,
        )
        self.assertNotIn("LLAMINAR_CPU_NVNNI_PREFILL_M=64,128,256,512", stdout)
        self.assertNotIn(",256", stdout)
        self.assertIn("avx2-build.avx2-runtime", stdout)
        self.assertIn("avx512-build.avx2-runtime", stdout)
        self.assertIn("TrainerCsv_PrefillSerialRouteManifest", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_PREFILL_WARMUP_BUDGET_US=50000", stdout)
        self.assertIn(
            "LLAMINAR_CPU_NVNNI_PREFILL_TRANSITION_WARMUP_BUDGET_US=50000",
            stdout,
        )
        self.assertIn(
            "LLAMINAR_CPU_NVNNI_PREFILL_TRANSITION_WARMUP_BUDGET_CEILING_US=500000",
            stdout,
        )
        self.assertIn("LLAMINAR_CPU_NVNNI_PREFILL_MPI_ROUND_SYNC=1", stdout)
        self.assertIn("CPU NativeVNNI prefill checkpoint", stdout)
        self.assertIn(
            "require every CPU frequency policy governor == performance",
            stdout,
        )
        self.assertIn("CPU prefill corpus transaction completed", stdout)
        self.assertNotIn("analyze_cpu_native_vnni_prefill_trainer.py", stdout)

    def test_cpu_prefill_candidate_expansion_collects_only_anchor_and_new_family(self) -> None:
        result = self.run_script(
            "--backend",
            "cpu-prefill",
            "--profile",
            "all",
            "--cpu-measurement-lanes",
            "2",
            "--collect-cpu-prefill-candidate-expansion-source-aggregate",
            "/evidence/immutable-source.csv",
            "--collect-cpu-prefill-candidate-expansion-source-timing",
            "/evidence/immutable-source.timing.csv",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn("authenticate candidate-expansion source", stdout)
        self.assertIn("cpu_prefill_candidate_expansion.v4.json", stdout)
        self.assertIn(
            '--collection-build-digest "${cpu_build_id}"',
            SCRIPT.read_text(encoding="utf-8"),
        )
        candidate_environment = next(
            item for item in stdout.split()
            if item.startswith("LLAMINAR_CPU_NVNNI_PREFILL_CANDIDATES=")
        )
        self.assertIn("row_chunk_grid", candidate_environment)
        self.assertEqual(candidate_environment.count("two_row_pair_grid"), 5)
        self.assertNotIn("two_row_tiles", candidate_environment)
        self.assertEqual(
            candidate_environment.count("decode_equivalent_kpart"),
            1,
        )
        self.assertIn("LLAMINAR_CPU_NVNNI_PREFILL_MPI_ROUND_SYNC=1", stdout)
        self.assertIn(
            "LLAMINAR_CPU_NVNNI_PREFILL_ANCHORED_EXPANSION=1",
            stdout,
        )
        self.assertIn(
            "LLAMINAR_CPU_NVNNI_PREFILL_WARMUP_BUDGET_US=0",
            stdout,
        )
        self.assertIn(
            "LLAMINAR_CPU_NVNNI_PREFILL_TRANSITION_WARMUP_BUDGET_US=0",
            stdout,
        )
        self.assertIn("LLAMINAR_CPU_NVNNI_PREFILL_WARMUP=2", stdout)
        self.assertIn(
            "LLAMINAR_CPU_NVNNI_PREFILL_TRANSITION_WARMUP_LATENCY_MULTIPLIER=0",
            stdout,
        )
        self.assertIn(
            "LLAMINAR_CPU_NVNNI_PREFILL_MIN_ITERS=5",
            stdout,
        )
        self.assertIn(
            "LLAMINAR_CPU_NVNNI_PREFILL_STATIONARY_MIN_ITERS=5",
            stdout,
        )
        self.assertIn(
            "LLAMINAR_CPU_NVNNI_PREFILL_MAX_ITERS=180",
            stdout,
        )
        self.assertIn(
            "LLAMINAR_CPU_NVNNI_PREFILL_TIMING_BUDGET_US=100000",
            stdout,
        )
        self.assertIn("--candidate-expansion-plan-json", stdout)
        self.assertIn("--candidate-expansion-input", stdout)
        self.assertIn("--candidate-expansion-timing-sidecar", stdout)
        self.assertNotIn(
            "--development-only",
            SCRIPT.read_text(encoding="utf-8"),
        )
        wrapper = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            '"${cpu_prefill_candidate_expansion_max_iters}" 5 2 3',
            wrapper,
        )
        self.assertIn(
            '"${cpu_prefill_candidate_expansion_timing_budget_us}"',
            wrapper,
        )
        self.assertIn(
            "Validate the complete MPMD batch before publishing either rank",
            wrapper,
        )
        self.assertLess(
            wrapper.index("for ((launch_attempt = 1;"),
            wrapper.index(
                'run_cmd mv "${partials_ref[job_index]}.inprogress"'
            ),
        )

    def test_cpu_prefill_candidate_expansion_rebases_an_older_checkpoint(self) -> None:
        result = self.run_script(
            "--backend",
            "cpu-prefill",
            "--profile",
            "all",
            "--collect-cpu-prefill-candidate-expansion-source-aggregate",
            "/evidence/immutable-source.csv",
            "--collect-cpu-prefill-candidate-expansion-source-timing",
            "/evidence/immutable-source.timing.csv",
            "--resume-cpu-partials",
            "--resume-cpu-candidate-expansion-from",
            "/evidence/older-checkpoint",
            "--cpu-prefill-harness-build-change-audit",
            "reviewed timing harness only",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "rebase CPU prefill candidate-expansion checkpoint",
            result.stdout,
        )
        self.assertIn("/evidence/older-checkpoint", result.stdout)
        self.assertIn(
            "audit timing-harness-only build change",
            result.stdout,
        )
        self.assertIn(
            "rebased_candidate_expansion_records",
            SCRIPT.read_text(encoding="utf-8"),
        )
        self.assertIn(
            '("\\n" if records else "")',
            SCRIPT.read_text(encoding="utf-8"),
        )

    def test_candidate_expansion_rebase_requires_resume_mode(self) -> None:
        result = self.run_script(
            "--backend",
            "cpu-prefill",
            "--profile",
            "all",
            "--collect-cpu-prefill-candidate-expansion-source-aggregate",
            "/evidence/immutable-source.csv",
            "--collect-cpu-prefill-candidate-expansion-source-timing",
            "/evidence/immutable-source.timing.csv",
            "--resume-cpu-candidate-expansion-from",
            "/evidence/older-checkpoint",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --resume-cpu-partials", result.stderr)

    def test_cpu_prefill_sealed_holdout_launches_only_after_freeze(self) -> None:
        """Each evidence lifetime must remain explicit and correctly ordered."""

        wrapper = SCRIPT.read_text(encoding="utf-8")
        production = wrapper[wrapper.index('if [[ "${measurement_profile}" == "production" ]]') :]
        refinement = production.index("--development-update-records-from")
        development_fit = production.index("--development-fit-diagnostic")
        freeze = production.index('run_cmd "${cpu_prefill_freeze[@]}"')
        freeze_checkpoint = production.index(
            'if (( stop_after_cpu_prefill_freeze )); then'
        )
        sealed_launch = production.index(
            '--sealed-witness-plan "${cpu_prefill_sealed_witness_plan_json}"'
        )
        certify = production.index('local -a cpu_prefill_certify=(')

        self.assertLess(refinement, freeze)
        self.assertLess(refinement, development_fit)
        self.assertLess(development_fit, freeze)
        self.assertLess(freeze, freeze_checkpoint)
        self.assertLess(freeze_checkpoint, sealed_launch)
        self.assertLess(freeze, sealed_launch)
        self.assertLess(sealed_launch, certify)
        self.assertNotIn(
            "native_vnni_dispatch.cpu_prefill_split_manifest",
            production,
        )
        self.assertIn(
            '"${cpu_prefill_active_development_csvs[${certify_development_index}]}"',
            production,
        )
        self.assertIn("cpu_prefill_generic_refinement_module", wrapper)
        self.assertIn("--generic-refinement-plan-json", production)
        self.assertIn(
            'generic-refinement-r${round}.${plan_token}.${format_spec}',
            production,
        )
        self.assertIn("validate_cpu_prefill_partial", production)
        self.assertIn(
            "Discarding non-installable CPU prefill refinement partial",
            production,
        )
        self.assertIn(
            "Ignoring non-installable completed CPU prefill refinement round",
            production,
        )
        self.assertLess(
            production.index("validate_cpu_prefill_partial"),
            production.index(
                'run_cmd mv "${partials_ref[job_index]}.inprogress"'
            ),
        )
        self.assertIn('sha256sum "${plan_path}"', production)
        self.assertIn("cpu_prefill_sweep.development-refinement-v4.csv", wrapper)
        self.assertIn("cpu_prefill_sweep.development-v4.csv", wrapper)
        self.assertIn("cpu_prefill_sweep.sealed-v5.csv", wrapper)
        self.assertIn("cpu_prefill_sweep.development-v6.csv", wrapper)
        self.assertIn("cpu_prefill_sweep.sealed-v6.csv", wrapper)
        self.assertIn("cpu_prefill_sweep.development-v7.csv", wrapper)
        self.assertIn("cpu_prefill_sweep.sealed-v7.csv", wrapper)
        self.assertIn("cpu_prefill_sweep.development-v8.csv", wrapper)
        self.assertIn("cpu_prefill_sweep.sealed-v8.csv", wrapper)
        self.assertIn("cpu_prefill_sweep.sealed.csv", wrapper)
        self.assertNotIn("--sealed-records", production)
        self.assertIn("sealed_pending_phase_inventory", production)
        self.assertIn(
            '[[ "${sealed_phase_inventory}" != "${sealed_pending_phase_inventory}" ]]',
            production,
        )
        self.assertIn("--coalesce-format-fixtures", production)
        self.assertIn('--source-formats "${cpu_formats}"', production)
        self.assertIn(
            'sealed_group_ends+=("${#sealed_job_shapes[@]}")',
            production,
        )
        self.assertIn("reuse_cpu_prefill_profiler_evidence", production)
        self.assertIn(
            "Reusing per-variant CPU prefill profiler evidence",
            production,
        )

    def test_cpu_prefill_resume_restores_completed_generic_round_chain(self) -> None:
        """A restart must fit the latest aggregate without repeating old rounds."""

        fixtures = {
            "cpu_prefill_common_observations.v8.csv": CPU_PREFILL_CHECKPOINT,
            "cpu_prefill_generic_refinement.round1.json": "round-1-plan\n",
            "cpu_prefill_sweep.development-generic-r1.csv": "round-1-csv\n",
            "cpu_prefill_sweep.development-generic-r1.timing.csv": (
                "round-1-timing\n"
            ),
            "cpu_prefill_generic_refinement.round2.json": "round-2-plan\n",
            "cpu_prefill_sweep.development-generic-r2.csv": "round-2-csv\n",
            "cpu_prefill_sweep.development-generic-r2.timing.csv": (
                "round-2-timing\n"
            ),
            # A plan without its completed aggregate is an interrupted next
            # round. The current fit must regenerate/authenticate it.
            "cpu_prefill_generic_refinement.round3.json": "stale-round-3-plan\n",
        }
        result = self.run_script_with_output_files(
            fixtures,
            "--backend",
            "cpu-prefill",
            "--profile",
            "all",
            "--skip-sweep",
            "--resume-cpu-partials",
            "--skip-profiler-evidence",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout
        self.assertIn(
            "Resuming CPU prefill from completed generic refinement round 2",
            stdout,
        )
        self.assertIn(
            "Resuming CPU prefill common observation run identity: "
            "stable-cpu-prefill-development",
            stdout,
        )
        self.assertRegex(
            stdout,
            r"--run-id(?:\\ | )stable-cpu-prefill-development",
        )
        self.assertIn("cpu_prefill_sweep.development-generic-r2.csv", stdout)
        self.assertIn(
            "cpu_prefill_generic_refinement.round1.json",
            stdout,
        )
        self.assertIn(
            "cpu_prefill_generic_refinement.round2.json",
            stdout,
        )
        self.assertNotRegex(
            stdout,
            r"--generic-refinement-plan-json[^\n]*round3\.json",
        )

    def test_existing_cpu_prefill_refinement_requires_complete_provenance(self) -> None:
        """Direct collection cannot accept only a convenient plan pathname."""

        result = self.run_script(
            "--backend",
            "cpu-prefill",
            "--profile",
            "all",
            "--collect-cpu-prefill-refinement-plan",
            "/tmp/round7.json",
            "--cpu-prefill-refinement-round",
            "7",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "requires its plan, source fit, source observations, and round",
            result.stderr,
        )

    def test_explicit_refinement_predecessor_requires_aggregate_and_timing(self) -> None:
        """A post-lineage refinement cannot authenticate half a predecessor."""

        result = self.run_script(
            "--backend",
            "cpu-prefill",
            "--profile",
            "all",
            "--collect-cpu-prefill-refinement-plan",
            "/tmp/round8.json",
            "--cpu-prefill-refinement-source-fit",
            "/tmp/fit.json",
            "--cpu-prefill-refinement-source-observations",
            "/tmp/observations.csv",
            "--cpu-prefill-refinement-source-aggregate",
            "/tmp/development-v9.csv",
            "--cpu-prefill-refinement-round",
            "8",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "requires both source aggregate and timing sidecar",
            result.stderr,
        )

    def test_existing_cpu_prefill_refinement_uses_canonical_collector(self) -> None:
        """The resume entry point must not grow a second measurement path."""

        wrapper = SCRIPT.read_text(encoding="utf-8")
        direct = wrapper[wrapper.index(
            'if [[ -n "${collect_cpu_prefill_refinement_plan}" ]]; then'
        ):]

        self.assertIn(
            "collect_cpu_prefill_generic_refinement_round",
            direct,
        )
        self.assertIn(
            "cpu_prefill_refinement_launch_attempts",
            wrapper,
        )
        self.assertIn("|| return $?", wrapper)
        self.assertIn("cpu_prefill_refinement_source_fit", direct)
        self.assertIn("cpu_prefill_refinement_source_observations", direct)
        self.assertIn("cpu_prefill_refinement_source_aggregate", direct)
        self.assertIn("cpu_prefill_refinement_source_timing", direct)
        self.assertIn(
            '--split-manifest "${cpu_prefill_split_manifest_path}"',
            direct,
        )
        self.assertIn("validate_cpu_prefill_partial", direct)
        self.assertIn("combine_compatible_csvs", direct)
        self.assertIn("--thread-count", direct)
        self.assertIn(
            'lineage_thread_count}" != "${cpu_threads}',
            direct,
        )
        self.assertLess(
            direct.index("validate_cpu_prefill_partial"),
            direct.index("collect_cpu_prefill_generic_refinement_round"),
        )

    def test_cpu_prefill_lineage_requires_every_immutable_source(self) -> None:
        """An additive split migration cannot be named by its plan alone."""

        result = self.run_script(
            "--backend",
            "cpu-prefill",
            "--profile",
            "all",
            "--collect-cpu-prefill-development-lineage-plan",
            "/tmp/v8-to-v9.json",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "requires its plan, source aggregate, timing sidecar, source split manifest, and source route manifests",
            result.stderr,
        )

    def test_cpu_prefill_replay_recipe_replaces_manual_lineage_flags(self) -> None:
        """Production replay cannot combine typed and hand-assembled provenance."""

        result = self.run_script(
            "--backend",
            "cpu-prefill",
            "--profile",
            "all",
            "--skip-sweep",
            "--cpu-prefill-fit-replay-recipe",
            "/tmp/replay.json",
            "--cpu-prefill-fit-candidate-expansion-plan",
            "/tmp/manual-expansion.json",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "replaces every manual CPU prefill fit lineage option",
            result.stderr,
        )

    def test_cpu_prefill_primary_profiler_requires_complete_triplet(self) -> None:
        """A replacement base cannot infer evidence or witnesses by filename."""

        result = self.run_script(
            "--backend",
            "cpu-prefill",
            "--profile",
            "all",
            "--skip-sweep",
            "--reuse-profiler-evidence",
            "--cpu-prefill-fit-candidate-expansion-plan",
            "/tmp/candidate-expansion.json",
            "--cpu-prefill-fit-candidate-expansion-source-aggregate",
            "/tmp/base.csv",
            "--cpu-prefill-fit-candidate-expansion-source-timing",
            "/tmp/base.timing.csv",
            "--cpu-prefill-fit-candidate-expansion-input",
            "/tmp/expansion.csv",
            "--cpu-prefill-fit-candidate-expansion-timing",
            "/tmp/expansion.timing.csv",
            "--cpu-prefill-fit-primary-profiler-requests",
            "/tmp/aligned-requests.json",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "primary profiler requests, evidence, and observations must form "
            "one complete transaction",
            result.stderr,
        )

    def test_cpu_prefill_primary_profiler_is_fit_only_evidence(self) -> None:
        """A paid replacement transaction cannot silently relaunch profilers."""

        result = self.run_script(
            "--backend",
            "cpu-prefill",
            "--profile",
            "all",
            "--skip-sweep",
            "--cpu-prefill-fit-candidate-expansion-plan",
            "/tmp/candidate-expansion.json",
            "--cpu-prefill-fit-candidate-expansion-source-aggregate",
            "/tmp/base.csv",
            "--cpu-prefill-fit-candidate-expansion-source-timing",
            "/tmp/base.timing.csv",
            "--cpu-prefill-fit-candidate-expansion-input",
            "/tmp/expansion.csv",
            "--cpu-prefill-fit-candidate-expansion-timing",
            "/tmp/expansion.timing.csv",
            "--cpu-prefill-fit-primary-profiler-requests",
            "/tmp/aligned-requests.json",
            "--cpu-prefill-fit-primary-profiler-evidence",
            "/tmp/aligned-evidence.json",
            "--cpu-prefill-fit-primary-profiler-observations",
            "/tmp/aligned-observations.csv",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "primary profiler replacement requires --reuse-profiler-evidence",
            result.stderr,
        )

    def test_cpu_prefill_recipe_preflight_precedes_route_probes(self) -> None:
        """Invalid replay lineage must fail before invoking any CPU binary."""

        wrapper = SCRIPT.read_text(encoding="utf-8")
        preflight = wrapper.index(
            "native_vnni_dispatch.cpu_prefill_replay_recipe"
        )
        route_probe = wrapper.index(
            "Ask the production C++ tile planner which serial arithmetic family"
        )
        self.assertLess(preflight, route_probe)
        self.assertIn(
            "preflight failed before expensive work",
            wrapper[preflight:route_probe],
        )

    def test_cpu_prefill_lineage_fit_forwards_complete_mixed_split_provenance(
        self,
    ) -> None:
        """Fit-only replay must preserve source routes, rounds, and splits."""

        result = self.run_script_with_output_files(
            {"cpu_prefill_common_observations.v8.csv": CPU_PREFILL_CHECKPOINT},
            "--backend",
            "cpu-prefill",
            "--profile",
            "all",
            "--skip-sweep",
            "--skip-profiler-evidence",
            "--cpu-prefill-fit-development-lineage-plan",
            "/tmp/v9-to-v10.json",
            "--cpu-prefill-lineage-source-aggregate",
            "/tmp/development-v9.csv",
            "--cpu-prefill-lineage-source-timing",
            "/tmp/development-v9.timing.csv",
            "--cpu-prefill-lineage-source-split-manifest",
            "/tmp/split-v9.json",
            "--cpu-prefill-lineage-source-refinement-split-manifest",
            "/tmp/split-v8.json",
            "--cpu-prefill-lineage-source-route-manifest",
            "/tmp/route-avx2.csv",
            "--cpu-prefill-lineage-source-route-manifest",
            "/tmp/route-avx512-avx2.csv",
            "--cpu-prefill-lineage-source-route-manifest",
            "/tmp/route-avx512.csv",
            "--cpu-prefill-lineage-source-refinement-plan",
            "/tmp/round-v8.json",
            "--cpu-prefill-lineage-source-refinement-plan",
            "/tmp/round-v9.json",
            "--cpu-prefill-fit-candidate-expansion-plan",
            "/tmp/candidate-expansion.json",
            "--cpu-prefill-fit-candidate-expansion-source-aggregate",
            "/tmp/base-r16.csv",
            "--cpu-prefill-fit-candidate-expansion-source-timing",
            "/tmp/base-r16.timing.csv",
            "--cpu-prefill-fit-candidate-expansion-input",
            "/tmp/candidate-expansion.csv",
            "--cpu-prefill-fit-candidate-expansion-timing",
            "/tmp/candidate-expansion.timing.csv",
            "--cpu-prefill-fit-additive-aggregate",
            "/tmp/refinement-r17.csv",
            "--cpu-prefill-fit-additive-timing",
            "/tmp/refinement-r17.timing.csv",
            "--cpu-prefill-fit-additive-aggregate",
            "/tmp/lineage-v10.csv",
            "--cpu-prefill-fit-additive-timing",
            "/tmp/lineage-v10.timing.csv",
            "--cpu-prefill-fit-additive-aggregate",
            "/tmp/generic-refinement-r1.csv",
            "--cpu-prefill-fit-additive-timing",
            "/tmp/generic-refinement-r1.timing.csv",
            "--cpu-prefill-fit-generic-refinement-plan",
            "/tmp/generic-refinement-r1.json",
            "--cpu-prefill-fit-additive-profiler-requests",
            "/tmp/pair-grid-profiler-requests.json",
            "--cpu-prefill-fit-additive-profiler-evidence",
            "/tmp/pair-grid-profiler-evidence.json",
            "--cpu-prefill-fit-additive-profiler-observations",
            "/tmp/pair-grid-profiler-witnesses.csv",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout
        self.assertIn(
            "require complete CPU prefill candidate-expansion fit transaction",
            stdout,
        )
        self.assertIn(
            "--development-lineage-plan-json /tmp/v9-to-v10.json",
            stdout,
        )
        self.assertIn(
            "--development-lineage-source-route-manifest /tmp/route-avx2.csv",
            stdout,
        )
        self.assertIn(
            "--development-lineage-source-refinement-split-manifest /tmp/split-v8.json",
            stdout,
        )
        self.assertIn(
            "--development-lineage-source-refinement-plan-json /tmp/round-v9.json",
            stdout,
        )
        self.assertIn(
            "--input /tmp/base-r16.csv /tmp/refinement-r17.csv /tmp/lineage-v10.csv /tmp/generic-refinement-r1.csv",
            stdout,
        )
        self.assertIn(
            "--generic-refinement-plan-json /tmp/generic-refinement-r1.json",
            stdout,
        )
        self.assertIn(
            "--adapted-development-prefix-input",
            stdout,
        )
        self.assertNotIn("--adapted-development-prefix-count", stdout)
        self.assertIn(
            "--candidate-expansion-plan-json /tmp/candidate-expansion.json",
            stdout,
        )
        self.assertIn(
            "--candidate-expansion-input /tmp/candidate-expansion.csv",
            stdout,
        )
        self.assertIn(
            "--candidate-expansion-timing-sidecar /tmp/candidate-expansion.timing.csv",
            stdout,
        )
        self.assertIn(
            "--development-profiler-requests /tmp/pair-grid-profiler-requests.json",
            stdout,
        )
        self.assertIn(
            "--development-profiler-evidence /tmp/pair-grid-profiler-evidence.json",
            stdout,
        )
        self.assertIn(
            "--development-profiler-observations /tmp/pair-grid-profiler-witnesses.csv",
            stdout,
        )
        self.assertGreaterEqual(
            stdout.count("--development-lineage-plan-json"),
            4,
        )

    def test_cpu_prefill_lineage_is_additive_and_plan_authenticated(self) -> None:
        """The migration collector must preserve source bytes and add one plan."""

        wrapper = SCRIPT.read_text(encoding="utf-8")
        direct = wrapper[wrapper.index(
            'if [[ -n "${collect_cpu_prefill_development_lineage_plan}" ]]; then'
        ):]

        self.assertIn("cpu_prefill_development_lineage_module", direct)
        self.assertIn(
            "collect_cpu_prefill_development_lineage_increment",
            direct,
        )
        self.assertIn("cpu_prefill_lineage_source_aggregate", direct)
        self.assertIn("cpu_prefill_lineage_source_timing", direct)
        self.assertIn("cpu_prefill_lineage_source_route_manifests", direct)
        self.assertIn(
            "cpu_prefill_lineage_source_refinement_split_manifests",
            direct,
        )
        self.assertIn("validate_cpu_prefill_partial", direct)
        self.assertIn(
            'combine_compatible_csvs "${lineage_development_csv}"',
            direct,
        )
        self.assertIn(
            'combine_compatible_csvs "${lineage_development_timing_csv}"',
            direct,
        )
        self.assertIn('--build', direct)
        self.assertIn(
            '[[ ! -s "${collect_cpu_prefill_development_lineage_plan}" ]]',
            direct,
        )
        self.assertIn("development-lineage.increment.csv", direct)
        self.assertIn("development-lineage.combined.csv", direct)
        self.assertNotRegex(direct, r"development-lineage-v[0-9]+")
        self.assertLess(
            direct.index("cpu_prefill_development_lineage_module"),
            direct.index("collect_cpu_prefill_development_lineage_increment"),
        )

        collector = wrapper[
            wrapper.index("collect_cpu_prefill_development_lineage_increment()"):
            wrapper.index("collect_cpu_prefill_candidate_expansion()")
        ]
        self.assertIn("--launch-records", collector)
        self.assertNotIn("--thread-count", collector)
        self.assertNotIn("--candidate-expansion", collector)
        self.assertIn(
            "job_timing_partials || return $?",
            collector,
        )
        self.assertIn(
            'combine_csvs "${aggregate}" "${partials[@]}"',
            collector,
        )

    def test_cpu_prefill_baseline_reuse_is_an_explicit_production_mode(self) -> None:
        result = self.run_script(
            "--backend",
            "cpu-prefill",
            "--profile",
            "all",
            "--skip-cpu-prefill-baseline",
            "--cpu-batch-limit",
            "1",
            "--skip-profiler-evidence",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn("require complete CPU prefill baseline", stdout)
        self.assertIn("cpu_prefill.refinement-v4.", stdout)
        self.assertIn("CPU NativeVNNI prefill refinement-v4 checkpoint", stdout)
        self.assertNotIn("publish CPU prefill contract", stdout)

    def test_cpu_prefill_freeze_checkpoint_rejects_installation(self) -> None:
        result = self.run_script(
            "--backend",
            "cpu-prefill",
            "--profile",
            "all",
            "--stop-after-cpu-prefill-freeze",
            "--install",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "ordinary prefill is heuristic-only; cpu-prefill cannot install",
            result.stderr,
        )

    def test_cpu_prefill_profiler_ablation_is_diagnostic_only(self) -> None:
        """The timing-only A/B half cannot enter a production freeze."""

        result = self.run_script(
            "--backend",
            "cpu-prefill",
            "--profile",
            "all",
            "--cpu-prefill-ablate-profiler-features",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "requires --cpu-prefill-diagnostic-max-leaves",
            result.stderr,
        )
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--ablate-profiler-features", script)
        self.assertIn(".profiler-ablated", script)
        self.assertIn(".profiler-informed", script)
        self.assertLess(
            script.index("development diagnostic complete"),
            script.index("if (( cpu_prefill_unpromoted_domains == 0 ))"),
        )

    def test_cpu_prefill_reused_development_can_collect_fresh_sealed_holdout(
        self,
    ) -> None:
        """A lineage replay may freeze, measure sealed evidence, and certify."""

        result = self.run_script_with_output_files(
            {
                "cpu_prefill_common_observations.v8.csv": (
                    CPU_PREFILL_CHECKPOINT
                ),
                "cpu_prefill_profiler_source_observations.csv": "header\nrow\n",
                "cpu_prefill_profiler_observation_witnesses.csv": "header\nrow\n",
                "cpu_prefill_profiler_requests.json": "{}\n",
                "cpu_prefill_profiler_evidence.json": "{}\n",
            },
            "--backend",
            "cpu-prefill",
            "--profile",
            "all",
            "--skip-sweep",
            "--collect-cpu-prefill-sealed-after-freeze",
            "--reuse-profiler-evidence",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("publish CPU prefill contract", result.stdout)
        self.assertIn(
            "collect fresh CPU prefill sealed-v8 witness holdout",
            result.stdout,
        )
        self.assertIn("--freeze-generic", result.stdout)
        self.assertIn("--certify-generic", result.stdout)
        self.assertIn("cpu_prefill_final_profiler_requests.json", result.stdout)
        self.assertIn("compose-evidence", result.stdout)
        self.assertIn(
            "cpu_prefill_final_profiler_observation_witnesses.csv",
            result.stdout,
        )
        self.assertNotIn("native_vnni_dispatch.profiler_collectors", result.stdout)

    def test_cpu_prefill_fresh_sealed_collection_requires_reused_development(
        self,
    ) -> None:
        result = self.run_script(
            "--backend",
            "cpu-prefill",
            "--profile",
            "all",
            "--collect-cpu-prefill-sealed-after-freeze",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --skip-sweep", result.stderr)

    def test_cpu_qwen36_profile_trains_forced_policy_variants(self) -> None:
        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "qwen36-core",
            "--cpu-formats",
            "Q4_K",
            "--m-values",
            "2,3,4",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_SHAPE_NAME=Qwen36_FFN_GateUp", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_SHAPE_NAME=Qwen36_FFN_DownProjection", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_SHAPE_NAME=Qwen36_GDN_OutputProjection", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_WARMUP=5", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_ITERS=30", stdout)
        self.assertIn("--require-inventory-formats Q4_K", stdout)
        self.assertIn(
            "--require-inventory-shapes "
            "Qwen36_Attn_QKVProjection,Qwen36_FFN_GateUp,"
            "Qwen36_FFN_DownProjection,Qwen36_GDN_InnerProjection,"
            "Qwen36_GDN_ZProjection,Qwen36_GDN_TimeProjection,"
            "Qwen36_GDN_OutputProjection",
            stdout,
        )
        self.assertIn("--require-inventory-m-values 2,3,4", stdout)

    def test_cpu_qwen36_policy_requirements_skip_decode_m1(self) -> None:
        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "qwen36-core",
            "--cpu-formats",
            "Q4_K",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn(
            "--require-inventory-m-values "
            "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31",
            stdout,
        )
        self.assertNotIn("--require-inventory-m-values 1,", stdout)

    def test_cpu_qwen36_lm_head_uses_stable_required_key_training_budget(self) -> None:
        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "qwen36-lm-head",
            "--cpu-formats",
            "Q4_K",
            "--m-values",
            "2,3,4",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_SHAPE_NAME=Qwen36_LM_Head", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_WARMUP=5", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_ITERS=30", stdout)
        self.assertIn("--require-inventory-formats Q4_K", stdout)
        self.assertIn("--require-inventory-shapes Qwen36_LM_Head", stdout)
        self.assertIn("--require-inventory-m-values 2,3,4", stdout)

    def test_cpu_qwen36_moe_profile_uses_real_expert_buckets_and_stable_budget(self) -> None:
        result = self.run_script(
            "--backend",
            "cpu",
            "--profile",
            "qwen36-moe",
            "--cpu-formats",
            "Q4_K",
            "--m-values",
            "2,3,4",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_SHAPE_NAME=35BMoE_Expert_GateUp", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_SHAPE_NAME=35BMoE_Expert_Down", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_SHAPE_NAME=Qwen36MoE_GDN_QKVProjection", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_SHAPE_NAME=Qwen36MoE_GDN_ZProjection", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_N=512", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_K=2048", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_N=2048", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_K=512", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_N=8192", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_N=4096", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_WARMUP=5", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_ITERS=30", stdout)
        self.assertIn("--require-inventory-formats Q4_K", stdout)
        self.assertIn(
            "--require-inventory-shapes "
            "35BMoE_Expert_GateUp,35BMoE_Expert_Down,"
            "Qwen36MoE_GDN_QKVProjection,Qwen36MoE_GDN_ZProjection",
            stdout,
        )
        self.assertIn("--require-inventory-m-values 2,3,4", stdout)

    def test_cpu_generated_verifier_policy_is_checked_in_and_consumed(self) -> None:
        source_path = (
            REPO_ROOT
            / "src"
            / "v2"
            / "kernels"
            / "cpu"
            / "gemm"
            / "CPUNativeVNNIGemv.h"
        )
        generated_path = source_path.parent / "CPUNativeVNNIVerifierRowsPolicyGenerated.inc"

        self.assertTrue(generated_path.is_file(), "CPU verifier generated policy include is missing")
        source = source_path.read_text(encoding="utf-8")
        generated_source = generated_path.read_text(encoding="utf-8")
        self.assertIn("CPUNativeVNNIVerifierRowsPolicyGenerated.inc", source)
        self.assertIn("selectCPUNativeVNNIVerifierRowsGeneratedPolicy", source)
        self.assertRegex(
            source,
            r"selectVerifierRowsPolicy\(\s*"
            r"packed,\s*M,\s*N,\s*K,\s*effective_isa,\s*num_threads,\s*"
            r"cfg\.k_tiles\)",
        )
        self.assertIn("struct VerifierRowsPolicyCacheEntry", source)
        self.assertIn("static thread_local VerifierRowsPolicyCacheEntry most_recent", source)
        self.assertIn("static thread_local std::array<", source)
        self.assertRegex(
            source,
            r"most_recent\.matches\(\s*geometry_key,\s*threads,\s*"
            r"serial_k_tiles,\s*runtime_isa\)",
        )
        self.assertRegex(
            source,
            r"cache_entry\.matches\(\s*geometry_key,\s*threads,\s*"
            r"serial_k_tiles,\s*runtime_isa\)",
        )
        self.assertIn("omp_get_max_threads()", source)
        self.assertIn("No certified CPU NativeVNNI verifier-row policy", source)
        self.assertRegex(
            source,
            r"(?s)if \(selected_generated_policy\).*?return policy;.*?"
            r"throw std::runtime_error\(\s*std::string\("
            r'"No certified CPU NativeVNNI verifier-row policy',
        )
        self.assertIn("use_avx512 && M >= 3 && use_wide_rows", source)
        self.assertIn("const int row_tile_count = (M + 3) / 4", source)
        self.assertNotIn("policy_tile_rows", source)
        self.assertNotIn("M > 4 && resolve_generated_policy", source)
        self.assertIn(
            "selectCPUNativeVNNIVerifierRowsGeneratedPolicy(",
            source,
        )
        self.assertIn("reduceNativeVNNIKTilePartialsExact", source)
        self.assertEqual(
            len(
                re.findall(
                    r"^\s+reduceNativeVNNIKTilePartialsExact\($",
                    source,
                    flags=re.MULTILINE,
                )
            ),
            4,
        )
        self.assertEqual(source.count("_mm512_loadu_ps(base)"), 1)
        self.assertIn("grouped_k_parallel_row_tiles", source)
        self.assertIn("const int row_tile_width =", source)
        self.assertIn("plan.effective_verifier_schedule", source)
        self.assertNotIn("if (use_avx512 && M >= 2 && M <= 4)", source)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_POLICY_ABI 3", generated_source)
        self.assertIn("enum class CPUNativeVNNIBuildISA", generated_source)
        self.assertIn("enum class CPUNativeVNNIRuntimeISA", generated_source)
        self.assertIn("threads == 28", generated_source)

        validated = subprocess.run(
            ["python3", str(VALIDATOR), str(generated_path)],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(validated.returncode, 0, validated.stderr)

    def test_production_prefill_is_heuristic_only_on_every_backend(self) -> None:
        source_path = (
            REPO_ROOT
            / "src"
            / "v2"
            / "kernels"
            / "cpu"
            / "gemm"
            / "CPUNativeVNNIGemv.h"
        )
        source = source_path.read_text(encoding="utf-8")
        cuda_kernel_path = (
            REPO_ROOT
            / "src"
            / "v2"
            / "kernels"
            / "cuda"
            / "gemm"
            / "CUDANativeVNNIPrefillKernels.cu"
        )
        cuda_workspace_path = cuda_kernel_path.parent / "CUDAQuantisedGemmKernel.cpp"
        rocm_kernel_path = (
            REPO_ROOT
            / "src"
            / "v2"
            / "kernels"
            / "rocm"
            / "gemm"
            / "ROCmNativeVNNIGemmShardImpl.hip.inc"
        )
        cuda_source = cuda_kernel_path.read_text(encoding="utf-8")
        cuda_workspace_source = cuda_workspace_path.read_text(encoding="utf-8")
        rocm_source = rocm_kernel_path.read_text(encoding="utf-8")

        retired_paths = (
            source_path.parent / "CPUNativeVNNIPrefillPolicyGenerated.inc",
            cuda_kernel_path.parent / "CUDANativeVNNIPrefillDispatchGenerated.inc",
            rocm_kernel_path.parent / "ROCmNativeVNNIPrefillDispatchGenerated.inc",
        )
        for retired_path in retired_paths:
            self.assertFalse(
                retired_path.exists(),
                f"ordinary prefill must not consume generated policy {retired_path}",
            )

        self.assertNotIn("CPUNativeVNNIPrefillPolicyGenerated.inc", source)
        self.assertNotIn("selectCPUNativeVNNIPrefillGeneratedPolicy", source)
        self.assertNotIn("selectPrefillPolicy(", source)
        self.assertNotIn("No certified CPU NativeVNNI prefill policy", source)
        self.assertIn("switch (schedule_override)", source)
        self.assertIn("serial_cfg.k_tiles > 1", source)
        self.assertIn("VerifierRowsPolicy::Pairwise", source)

        self.assertNotIn("CUDANativeVNNIPrefillDispatchGenerated.inc", cuda_source)
        self.assertNotIn("selectPrefillTileGenerated", cuda_source)
        self.assertNotIn("CUDANativeVNNIPrefillDispatchGenerated.inc", cuda_workspace_source)
        self.assertIn(
            "chooseQ40PrefillRoute(M, N, K, prefill_ctx)",
            cuda_source,
        )
        self.assertIn(
            "M, N, K, prefill_ctx, complexity, CB",
            cuda_source,
        )

        self.assertNotIn("ROCmNativeVNNIPrefillDispatchGenerated.inc", rocm_source)
        self.assertNotIn("selectROCmNativeVNNIPrefillGenerated", rocm_source)
        self.assertIn("PATTERN S OPTIMIZED DISPATCH", rocm_source)

    def test_turnkey_install_preserves_rocm_native_vnni_physical_shards(
        self,
    ) -> None:
        """A policy refresh may replace an include, never the kernel topology."""

        cmake_source = (
            REPO_ROOT / "src" / "v2" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        refresh_source = SCRIPT.read_text(encoding="utf-8")
        gemm_directory = (
            REPO_ROOT / "src" / "v2" / "kernels" / "rocm" / "gemm"
        )

        self.assertFalse(
            (gemm_directory / "ROCmQuantisedGemmKernel_native_VNNI.hip").exists()
        )
        self.assertTrue(
            (gemm_directory / "ROCmNativeVNNIGemmShardImpl.hip.inc").is_file()
        )
        self.assertTrue(
            (gemm_directory / "ROCmNativeVNNIGemmDispatch.cpp").is_file()
        )
        for shard in range(8):
            source_name = f"ROCmNativeVNNIGemmShard{shard}.hip"
            self.assertTrue((gemm_directory / source_name).is_file())
            self.assertEqual(cmake_source.count(source_name), 1)

        self.assertNotIn(
            "ROCmQuantisedGemmKernel_native_VNNI.hip",
            refresh_source,
        )
        self.assertIn(
            'rocm_source_include="${repo_root}/src/v2/kernels/rocm/gemm/'
            'ROCmNativeVNNIDecodeDispatchGenerated.inc"',
            refresh_source,
        )
        self.assertNotIn("ROCmNativeVNNIGemmShard", refresh_source)

    def test_turnkey_install_preserves_cuda_native_vnni_physical_shards(
        self,
    ) -> None:
        """A CUDA policy refresh authenticates, but never rewrites, topology."""

        cmake_source = (
            REPO_ROOT / "src" / "v2" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        refresh_source = SCRIPT.read_text(encoding="utf-8")
        gemm_directory = (
            REPO_ROOT / "src" / "v2" / "kernels" / "cuda" / "gemm"
        )

        retired_source = "CUDANativeVNNIGemvTuned.cu"
        self.assertFalse((gemm_directory / retired_source).exists())
        self.assertNotIn(retired_source, cmake_source)

        structural_sources = (
            "CUDANativeVNNIGemvDispatch.cu",
            "CUDANativeVNNIGemvShard.h",
            "CUDANativeVNNIGemvShardImpl.cu.inc",
        )
        for source_name in structural_sources:
            self.assertTrue((gemm_directory / source_name).is_file())

        expected_codebooks = (
            (0, 4),
            (5, 6),
            (7, 8),
            (9, 10),
            (11, 12),
            (13, 14),
            (15, 16),
            (17, 19),
        )
        for shard, (first_codebook, second_codebook) in enumerate(
            expected_codebooks
        ):
            source_name = f"CUDANativeVNNIGemvShard{shard}.cu"
            source_path = gemm_directory / source_name
            self.assertTrue(source_path.is_file())
            self.assertEqual(cmake_source.count(source_name), 1)
            source = source_path.read_text(encoding="utf-8")
            self.assertIn(
                f"MACRO({first_codebook}); MACRO({second_codebook})",
                source,
            )
            self.assertIn(
                '#include "CUDANativeVNNIGemvShardImpl.cu.inc"',
                source,
            )

        self.assertEqual(
            cmake_source.count("CUDANativeVNNIGemvDispatch.cu"),
            1,
        )
        self.assertIn(
            "validate_cuda_native_vnni_gemv_physical_shards",
            refresh_source,
        )
        self.assertIn(
            "cuda_native_vnni_serial_policy_hash",
            refresh_source,
        )
        self.assertIn(
            'local cuda_source_include="${repo_root}/src/v2/kernels/cuda/gemm/'
            'CUDANativeVNNIGemvDispatchHeuristicGenerated.inc"',
            refresh_source,
        )
        self.assertEqual(refresh_source.count(retired_source), 1)
        self.assertIn(
            f'grep -F -q "{retired_source}" "${{cmake_path}}"',
            refresh_source,
        )

    def test_strong_gpu_trainers_own_the_complete_runtime_m_envelope(self) -> None:
        cuda_trainer = (
            REPO_ROOT
            / "tests/v2/performance/kernels/cuda/gemm"
            / "Perf__CUDANativeVNNIDecodeTrainer.cpp"
        ).read_text(encoding="utf-8")
        rocm_trainer = (
            REPO_ROOT
            / "tests/v2/performance/kernels/rocm"
            / "Perf__NativeVNNI_Throughput.cpp"
        ).read_text(encoding="utf-8")
        canonical = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31"

        self.assertIn("9, 10, 11, 12, 13, 14, 15, 16, 31", cuda_trainer)
        self.assertIn(canonical, rocm_trainer)
        self.assertNotIn("m > 4", rocm_trainer)


if __name__ == "__main__":
    unittest.main()
