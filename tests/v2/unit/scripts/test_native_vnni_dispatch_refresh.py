#!/usr/bin/env python3
"""Regression tests for the NativeVNNI dispatch refresh wrapper."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "refresh_native_vnni_dispatch_tables.sh"


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
        result = self.run_script("--backend", "both", "--m-values", "2,4")

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn("LLAMINAR_CUDA_TC_SWEEP_M=2,4", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_M=2,4", stdout)

    def test_install_copies_generated_backend_artifacts(self) -> None:
        result = self.run_script("--backend", "both", "--install")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CUDANativeVNNIGemvDispatchHeuristicGenerated.inc", result.stdout)
        self.assertIn("ROCmNativeVNNIDecodeDispatchGenerated.inc", result.stdout)
        self.assertIn("src/v2/kernels/cuda/gemm", result.stdout)
        self.assertIn("src/v2/kernels/rocm/gemm", result.stdout)

    def test_family_smoke_is_stratified_by_format(self) -> None:
        result = self.run_script(
            "--backend",
            "both",
            "--profile",
            "family-smoke",
            "--cuda-formats",
            "Q4_0,IQ4_XS",
            "--rocm-formats",
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
        self.assertIn("LLAMINAR_CUDA_TC_SWEEP_M=1,2", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_M=1,2", stdout)
        self.assertIn("combine-csv", stdout)
        self.assertIn("cuda_decode_sweep.Q4_0.csv", stdout)
        self.assertIn("cuda_decode_sweep.IQ4_XS.csv", stdout)
        self.assertIn("rocm_decode_sweep.Q4_0.csv", stdout)
        self.assertIn("rocm_decode_sweep.IQ4_XS.csv", stdout)

    def test_default_family_smoke_covers_full_format_inventory(self) -> None:
        result = self.run_script("--backend", "both", "--profile", "family-smoke")

        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout.replace("\\,", ",")
        self.assertIn("LLAMINAR_CUDA_TC_FORMATS=Q4_0", stdout)
        self.assertIn("LLAMINAR_CUDA_TC_FORMATS=IQ1_M", stdout)
        self.assertIn("LLAMINAR_CUDA_TC_FORMATS=Q8_0", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=Q4_0", stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=IQ1_M", stdout)
        self.assertNotIn("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=Q8_0", stdout)
        self.assertIn("cuda_decode_sweep.Q8_0.csv", stdout)
        self.assertIn("rocm_decode_sweep.IQ1_M.csv", stdout)

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
        full = self.run_script("--backend", "rocm", "--profile", "qwen36")
        cuda_core = self.run_script("--backend", "cuda", "--profile", "qwen36-core")

        self.assertEqual(core.returncode, 0, core.stderr)
        self.assertEqual(lm_head.returncode, 0, lm_head.stderr)
        self.assertEqual(full.returncode, 0, full.stderr)
        self.assertEqual(cuda_core.returncode, 0, cuda_core.stderr)

        core_stdout = core.stdout.replace("\\,", ",")
        lm_stdout = lm_head.stdout.replace("\\,", ",")
        full_stdout = full.stdout.replace("\\,", ",")

        self.assertIn("Qwen36_FFN_GateUp", core_stdout)
        self.assertIn("Qwen36_GDN_OutputProjection", core_stdout)
        self.assertNotIn("Qwen36_LM_Head", core_stdout)

        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_SHAPES=Qwen36_LM_Head", lm_stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_REFERENCE=native-auto", lm_stdout)
        self.assertNotIn("Qwen36_FFN_GateUp", lm_stdout)

        self.assertIn("Qwen36_FFN_GateUp", full_stdout)
        self.assertIn("Qwen36_LM_Head", full_stdout)
        self.assertIn("LLAMINAR_ROCM_NVNNI_DECODE_REFERENCE=fp32", core_stdout)
        self.assertIn("--min-overall-family-pct 99.0", cuda_core.stdout)
        self.assertIn("--min-fallback-family-pct 97.0", cuda_core.stdout)


if __name__ == "__main__":
    unittest.main()
