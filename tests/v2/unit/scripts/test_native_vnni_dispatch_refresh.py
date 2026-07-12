#!/usr/bin/env python3
"""Regression tests for the NativeVNNI dispatch refresh wrapper."""

from __future__ import annotations

import re
import subprocess
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


class NativeVNNIDispatchRefreshTest(unittest.TestCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmp:
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
                    str(Path(tmp) / "out"),
                    *args,
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

    def test_both_backends_emit_m_aware_sweep_contract(self) -> None:
        result = self.run_script("--backend", "both", "--profile", "quick")

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        runtime_m = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31"
        self.assertIn(f"LLAMINAR_CUDA_TC_SWEEP_M={runtime_m}", stdout)
        self.assertIn(f"LLAMINAR_ROCM_NVNNI_DECODE_M={runtime_m}", stdout)
        self.assertIn(
            "LLAMINAR_ROCM_NVNNI_DECODE_EXECUTION_MODES=eager,graph_captured",
            stdout,
        )
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_TIMING_CSV=", stdout)
        self.assertIn("LLAMINAR_CUDA_TC_SWEEP_FAMILIES=wide,kpar,direct", stdout)
        self.assertIn("infer_gemv_dispatch_heuristic.py", result.stdout)
        self.assertIn("analyze_cuda_tc_gemv_dispatch.py", result.stdout)
        self.assertIn("analyze_rocm_native_vnni_decode_trainer.py", result.stdout)
        self.assertIn("validate_native_vnni_generated_dispatch_ids.py", result.stdout)

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

    def test_custom_m_values_are_forwarded_to_cuda_and_rocm(self) -> None:
        result = self.run_script("--backend", "all", "--m-values", "2,4")

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn("LLAMINAR_CUDA_TC_SWEEP_M=2,4", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_M=2,4", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_M=2,4", stdout)

    def test_install_copies_generated_backend_artifacts(self) -> None:
        result = self.run_script(
            "--backend", "all", "--profile", "qwen36", "--install"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CUDANativeVNNIGemvDispatchHeuristicGenerated.inc", result.stdout)
        self.assertIn("ROCmNativeVNNIDecodeDispatchGenerated.inc", result.stdout)
        self.assertIn("CPUNativeVNNIVerifierRowsPolicyGenerated.inc", result.stdout)
        self.assertIn("src/v2/kernels/cuda/gemm", result.stdout)
        self.assertIn("src/v2/kernels/rocm/gemm", result.stdout)
        self.assertIn("src/v2/kernels/cpu/native_vnni", result.stdout)

    def test_cpu_smoke_profile_cannot_install_a_production_policy(self) -> None:
        result = self.run_script("--backend", "cpu", "--install")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("installing CPU dispatch requires", result.stderr)

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
        self.assertIn("LLAMINAR_CUDA_TC_FORMATS=Q4_0", stdout)
        self.assertIn("LLAMINAR_CUDA_TC_FORMATS=IQ4_XS", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=Q4_0", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=IQ4_XS", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS=Q4_0", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS=IQ4_XS", stdout)
        self.assertIn("LLAMINAR_CUDA_TC_SWEEP_M=1,2", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_M=1,2", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_M=1,2", stdout)
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
        self.assertIn("LLAMINAR_CUDA_TC_FORMATS=Q4_0", stdout)
        self.assertIn("LLAMINAR_CUDA_TC_FORMATS=IQ1_M", stdout)
        self.assertIn("LLAMINAR_CUDA_TC_FORMATS=Q8_0", stdout)
        self.assertIn("LLAMINAR_CUDA_TC_FORMATS=Q8_1", stdout)
        self.assertIn("LLAMINAR_CUDA_TC_FORMATS=Q8_K", stdout)
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

    def test_cuda_family_smoke_uses_proxy_thresholds(self) -> None:
        smoke = self.run_script("--backend", "cuda", "--profile", "family-smoke")
        strict = self.run_script("--backend", "cuda", "--profile", "qwen36")

        self.assertEqual(smoke.returncode, 0, smoke.stderr)
        self.assertEqual(strict.returncode, 0, strict.stderr)
        self.assertIn("--min-overall-family-pct 0.0", smoke.stdout)
        self.assertIn("--min-overall-exact-pct 0.0", smoke.stdout)
        self.assertIn("--min-fallback-family-pct 0.0", smoke.stdout)
        self.assertIn("--min-fallback-exact-pct 0.0", smoke.stdout)
        self.assertIn("--min-overall-family-pct 99.0", strict.stdout)
        self.assertIn("--min-overall-exact-pct 99.0", strict.stdout)
        self.assertIn("--min-fallback-family-pct 97.0", strict.stdout)
        self.assertIn("--min-fallback-exact-pct 30.0", strict.stdout)

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
        self.assertIn("--min-overall-family-pct 99.0", cuda_core.stdout)
        self.assertIn("--min-fallback-family-pct 97.0", cuda_core.stdout)

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
            "LLAMINAR_CUDA_TC_SHAPES=35BMoE_Expert_GateUp,35BMoE_Expert_Down,"
            "Qwen36MoE_GDN_QKVProjection,Qwen36MoE_GDN_ZProjection",
            stdout,
        )
        self.assertIn("LLAMINAR_CUDA_TC_SWEEP_M=2,3,4", stdout)
        self.assertIn("LLAMINAR_CUDA_TC_FORMATS=Q4_K,Q6_K", stdout)
        self.assertIn("LLAMINAR_CUDA_TC_SWEEP_FAMILIES=wide,kpar,direct", stdout)
        self.assertIn("--min-overall-family-pct 99.0", stdout)
        self.assertIn("--min-overall-exact-pct 99.0", stdout)
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
        self.assertIn("TrainerCsv_StrongVerifierRows_AllFormats", stdout)
        self.assertIn("avx2-build.avx2-runtime", stdout)
        self.assertIn("avx512-build.avx2-runtime", stdout)
        self.assertIn("avx512-build.avx512-runtime", stdout)
        self.assertIn("analyze_cpu_native_vnni_verifier_trainer.py", stdout)
        self.assertIn("--require-isa-matrix", stdout)
        self.assertIn("CPUNativeVNNIVerifierRowsPolicyGenerated.inc", stdout)
        self.assertIn("validate_native_vnni_generated_dispatch_ids.py", stdout)

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
        self.assertIn("--require-key Q4_K:2:17408:5120", stdout)
        self.assertIn("--require-key Q4_K:4:5120:17408", stdout)
        self.assertIn("--require-key Q4_K:3:5120:6144", stdout)

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
        self.assertIn("--require-key Q4_K:2:17408:5120", stdout)
        self.assertNotIn("--require-key Q4_K:1:", stdout)

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
        self.assertIn("--require-key Q4_K:2:248320:5120", stdout)
        self.assertIn("--require-key Q4_K:4:248320:5120", stdout)

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
        self.assertIn("--require-key Q4_K:2:512:2048", stdout)
        self.assertIn("--require-key Q4_K:4:512:2048", stdout)
        self.assertIn("--require-key Q4_K:2:2048:512", stdout)
        self.assertIn("--require-key Q4_K:4:2048:512", stdout)

    def test_cpu_generated_verifier_policy_is_checked_in_and_consumed(self) -> None:
        source_path = (
            REPO_ROOT
            / "src"
            / "v2"
            / "kernels"
            / "cpu"
            / "native_vnni"
            / "CPUNativeVNNIGemv.h"
        )
        generated_path = source_path.parent / "CPUNativeVNNIVerifierRowsPolicyGenerated.inc"

        self.assertTrue(generated_path.is_file(), "CPU verifier generated policy include is missing")
        source = source_path.read_text(encoding="utf-8")
        generated_source = generated_path.read_text(encoding="utf-8")
        self.assertIn("CPUNativeVNNIVerifierRowsPolicyGenerated.inc", source)
        self.assertIn("selectCPUNativeVNNIVerifierRowsGeneratedPolicy", source)
        self.assertIn("selectVerifierRowsPolicy(packed, M, N, K)", source)
        self.assertIn("omp_get_max_threads()", source)
        self.assertIn("No certified CPU NativeVNNI verifier-row policy", source)
        self.assertNotIn("return VerifierRowsPolicy::Pairwise;", source)
        self.assertIn("use_avx512 && M >= 3 && use_wide_rows", source)
        self.assertIn("const int row_tile_count = (M + 3) / 4", source)
        self.assertIn("const int policy_tile_rows = std::min(M, 4)", source)
        self.assertIn("resolve_generated_policy(M)", source)
        self.assertIn("M > 4 && resolve_generated_policy(policy_tile_rows)", source)
        self.assertIn("reduceNativeVNNIKTilePartialsExact", source)
        self.assertEqual(
            len(
                re.findall(
                    r"^\s+reduceNativeVNNIKTilePartialsExact\($",
                    source,
                    flags=re.MULTILINE,
                )
            ),
            3,
        )
        self.assertEqual(source.count("_mm512_loadu_ps(base)"), 1)
        self.assertIn("grouped_k_parallel_row_tiles", source)
        self.assertIn("const int row_tile_width = use_avx512 ? 4 : 2", source)
        self.assertNotIn("if (use_avx512 && M >= 2 && M <= 4)", source)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_POLICY_ABI 2", generated_source)
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


if __name__ == "__main__":
    unittest.main()
