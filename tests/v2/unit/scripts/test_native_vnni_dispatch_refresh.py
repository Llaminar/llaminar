#!/usr/bin/env python3
"""Regression tests for the NativeVNNI dispatch refresh wrapper."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "refresh_native_vnni_dispatch_tables.sh"
CPU_ANALYZER = (
    REPO_ROOT
    / "tests"
    / "v2"
    / "performance"
    / "kernels"
    / "cpu"
    / "analyze_cpu_native_vnni_verifier_trainer.py"
)
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
                    "--cpu-sweep-bin",
                    "/bin/true",
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
        self.assertIn("LLAMINAR_CUDA_TC_SWEEP_M=1,2,3,4", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_M=1,2,3,4", stdout)
        self.assertIn("LLAMINAR_CUDA_TC_SWEEP_FAMILIES=wide,kpar,direct", stdout)
        self.assertIn("infer_gemv_dispatch_heuristic.py", result.stdout)
        self.assertIn("analyze_cuda_tc_gemv_dispatch.py", result.stdout)
        self.assertIn("analyze_rocm_native_vnni_decode_trainer.py", result.stdout)
        self.assertIn("validate_native_vnni_generated_dispatch_ids.py", result.stdout)

    def test_custom_m_values_are_forwarded_to_cuda_and_rocm(self) -> None:
        result = self.run_script("--backend", "all", "--m-values", "2,4")

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn("LLAMINAR_CUDA_TC_SWEEP_M=2,4", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_M=2,4", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_M=2,4", stdout)

    def test_install_copies_generated_backend_artifacts(self) -> None:
        result = self.run_script("--backend", "all", "--install")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CUDANativeVNNIGemvDispatchHeuristicGenerated.inc", result.stdout)
        self.assertIn("ROCmNativeVNNIDecodeDispatchGenerated.inc", result.stdout)
        self.assertIn("CPUNativeVNNIVerifierRowsPolicyGenerated.inc", result.stdout)
        self.assertIn("src/v2/kernels/cuda/gemm", result.stdout)
        self.assertIn("src/v2/kernels/rocm/gemm", result.stdout)
        self.assertIn("src/v2/kernels/cpu/native_vnni", result.stdout)

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
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=Q4_0", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=IQ1_M", stdout)
        self.assertNotIn("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=Q8_0", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS=Q4_0", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS=IQ1_M", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS=Q8_0", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS=Q8_1", stdout)
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
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_REFERENCE=native-auto", lm_stdout)
        self.assertNotIn("Qwen36_FFN_GateUp", lm_stdout)

        self.assertIn(
            "LLAMINAR_ROCM_NVNNI_DECODE_SHAPES=35BMoE_Expert_GateUp,35BMoE_Expert_Down,"
            "Qwen36MoE_GDN_QKVProjection,Qwen36MoE_GDN_ZProjection",
            moe_stdout,
        )
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=", moe_stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_REFERENCE=fp32", moe_stdout)
        self.assertIn("--base-include", moe_stdout)
        self.assertIn("ROCmNativeVNNIDecodeDispatchGenerated.inc", moe_stdout)
        self.assertNotIn("Qwen36_LM_Head", moe_stdout)

        self.assertIn("Qwen36_FFN_GateUp", full_stdout)
        self.assertIn("Qwen36_LM_Head", full_stdout)
        self.assertIn("35BMoE_Expert_GateUp", full_stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_REFERENCE=fp32", core_stdout)
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
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_VARIANTS=0", stdout)
        self.assertIn("MTP_VerifierRows_GroupedVsSerial_Synthetic", stdout)
        self.assertIn("analyze_cpu_native_vnni_verifier_trainer.py", stdout)
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
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_VARIANTS=1", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_WARMUP=5", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_ITERS=10", stdout)
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
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_VARIANTS=1", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_WARMUP=5", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_ITERS=10", stdout)
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
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_VARIANTS=1", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_WARMUP=5", stdout)
        self.assertIn("LLAMINAR_CPU_NVNNI_VERIFIER_ITERS=10", stdout)
        self.assertIn("--require-key Q4_K:2:512:2048", stdout)
        self.assertIn("--require-key Q4_K:4:512:2048", stdout)
        self.assertIn("--require-key Q4_K:2:2048:512", stdout)
        self.assertIn("--require-key Q4_K:4:2048:512", stdout)

    def test_cpu_verifier_analyzer_generates_valid_policy_include(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "cpu_verifier.csv"
            inc_path = root / "CPUNativeVNNIVerifierRowsPolicyGenerated.inc"
            summary_path = root / "summary.txt"
            csv_path.write_text(
                "backend,phase,format,codebook,is_nibble_lut,payload_bytes,"
                "is_asymmetric,is_superblock,shape,n,k,m,isa,grouped_min_us,"
                "serial_min_us,speedup,grouped_mean_us,serial_mean_us,cosine,"
                "relative_l2,symmetric_kl,max_abs,correctness_pass\n"
                "cpu,verifier_rows,Q4_K,5,1,16,1,1,Qwen36Verifier,5120,5120,3,"
                "AVX512,10.0,20.0,2.0,11.0,21.0,1.0,0.0,0.0,0.0,1\n"
                "cpu,verifier_rows_pairwise_policy,Q4_K,5,1,16,1,1,"
                "Qwen36_FFN_DownProjection,5120,5120,3,AVX512,9.0,20.0,"
                "2.222222,10.0,21.0,1.0,0.0,0.0,0.0,1\n"
                "cpu,verifier_rows_wide_policy,Q6_K,8,0,24,0,1,"
                "Qwen36_GDN_OutputProjection,5120,5120,4,AVX512,10.0,36.0,"
                "3.6,11.0,37.0,1.0,0.0,0.0,0.0,1\n"
                "cpu,verifier_rows,Q8_0,19,0,32,0,0,Qwen36Verifier,"
                "5120,5120,2,AVX512,40.0,20.0,0.5,41.0,21.0,"
                "1.0,0.0,0.0,0.0,1\n",
                encoding="utf-8",
            )

            generated = subprocess.run(
                [
                    "python3",
                    str(CPU_ANALYZER),
                    "--input",
                    str(csv_path),
                    "--output",
                    str(inc_path),
                    "--summary",
                    str(summary_path),
                    "--require-key",
                    "Q4_K:3:5120:5120",
                    "--require-key",
                    "Q6_K:4:5120:5120",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            self.assertTrue(inc_path.exists())
            text = inc_path.read_text(encoding="utf-8")
            self.assertIn("CB=5 (Q4_1)", text)
            self.assertIn("CPUNativeVNNIVerifierRowsPolicy::WideRows", text)
            self.assertIn("CPUNativeVNNIVerifierRowsPolicy::Pairwise", text)
            self.assertIn("verifier_rows_pairwise_policy", text)
            self.assertIn("verifier_rows_wide_policy", text)
            self.assertNotIn("Q8_0", text)
            self.assertNotIn("0.500000f", text)

            validated = subprocess.run(
                ["python3", str(VALIDATOR), str(inc_path)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)
            self.assertIn("validated", validated.stdout)

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
        self.assertIn("CPUNativeVNNIVerifierRowsPolicyGenerated.inc", source)
        self.assertIn("selectCPUNativeVNNIVerifierRowsGeneratedPolicy", source)
        self.assertIn("selectVerifierRowsPolicy(packed, M, N, K)", source)
        self.assertIn("M == 3 && use_wide_rows", source)
        self.assertIn("M == 4 && use_wide_rows", source)

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
