#!/usr/bin/env python3
"""Regression tests for the generated MTP depth-policy trainer."""

from __future__ import annotations

import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
TRAINER = REPO_ROOT / "scripts" / "train_mtp_depth_policy.py"


class MTPDepthPolicyTrainerTest(unittest.TestCase):
    def write_summary(self, path: Path) -> None:
        path.write_text(
            textwrap.dedent(
                """\
                device\tmodel\tmode\tvariant\tsuccess\tdecode_tps\tacceptance_pct
                cuda:0\tdense\tgreedy\tfixed_d1\ttrue\t10.0\t90.0
                cuda:0\tdense\tgreedy\tfixed_d2\ttrue\t14.0\t82.0
                cuda:0\tdense\tgreedy\tfixed_d3\ttrue\t18.0\t75.0
                rocm:0\tdense\tgreedy\tfixed_d1\ttrue\t10.0\t80.0
                rocm:0\tdense\tgreedy\tfixed_d2\ttrue\t13.0\t70.0
                rocm:0\tdense\tgreedy\tfixed_d3\ttrue\t12.0\t40.0
                cpu:0\tdense\tstochastic\tfixed_d1\ttrue\t10.0\t55.0
                cpu:0\tdense\tstochastic\tfixed_d2\ttrue\t9.0\t30.0
                cpu:0\tdense\tstochastic\tfixed_d3\ttrue\t8.0\t20.0
                cuda:0\tdense\tstochastic\tfixed_d1\ttrue\t10.0\t65.0
                cuda:0\tdense\tstochastic\tfixed_d2\ttrue\t11.0\t76.0
                cuda:0\tdense\tstochastic\tfixed_d3\ttrue\t12.0\t81.0
                rocm:0\tdense\tstochastic\tfixed_d1\ttrue\t10.0\t84.0
                rocm:0\tdense\tstochastic\tfixed_d2\ttrue\t9.0\t67.0
                rocm:0\tdense\tstochastic\tfixed_d3\ttrue\t8.0\t65.0
                """
            ),
            encoding="utf-8",
        )

    def test_trainer_generates_policy_include_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary = tmp / "summary.tsv"
            output = tmp / "MTPDepthPolicyGenerated.inc"
            report = tmp / "report.txt"
            self.write_summary(summary)

            result = subprocess.run(
                [
                    "python3",
                    str(TRAINER),
                    "--input",
                    str(summary),
                    "--output",
                    str(output),
                    "--summary",
                    str(report),
                    "--holdout-modulus",
                    "999",
                    "--holdout-bucket",
                    "998",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            include_text = output.read_text(encoding="utf-8")
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("kMTPGeneratedDepthPolicyRules", include_text)
            self.assertIn("MTPVerifyMode::Greedy", include_text)
            self.assertIn("MTPVerifyMode::SpeculativeSampling", include_text)
            self.assertIn("MTPDepthPolicyBackend::CUDA", include_text)
            self.assertIn("MTPDepthPolicyBackend::ROCm", include_text)
            self.assertIn("MTPDepthPolicyModelClass::Dense", include_text)
            self.assertIn("trained_cuda_dense_greedy_promote_d1_to_d3", include_text)
            self.assertIn("trained_cuda_dense_greedy_hold_d3_to_d3", include_text)
            self.assertIn("trained_rocm_dense_greedy_demote_d3_to_d2", include_text)
            self.assertIn("trained_cuda_dense_stochastic_promote_d1_to_d3", include_text)
            self.assertIn("trained_cuda_dense_stochastic_promote_d2_to_d3", include_text)
            self.assertIn("trained_rocm_dense_stochastic_demote_d2_to_d1", include_text)
            self.assertIn("trained_rocm_dense_stochastic_demote_d3_to_d1", include_text)
            self.assertIn(", +2, \"trained_cuda_dense_greedy_promote_d1_to_d3\"", include_text)
            self.assertIn("examples=15", report_text)

    def test_holdout_accuracy_gate_can_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary = tmp / "summary.tsv"
            output = tmp / "MTPDepthPolicyGenerated.inc"
            report = tmp / "report.txt"
            self.write_summary(summary)

            result = subprocess.run(
                [
                    "python3",
                    str(TRAINER),
                    "--input",
                    str(summary),
                    "--output",
                    str(output),
                    "--summary",
                    str(report),
                    "--holdout-modulus",
                    "1",
                    "--holdout-bucket",
                    "0",
                    "--min-holdout-accuracy",
                    "1.1",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("holdout", result.stderr)

    def test_cross_backend_stochastic_policy_keeps_backend_specific_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary = tmp / "summary.tsv"
            output = tmp / "MTPDepthPolicyGenerated.inc"
            report = tmp / "report.txt"
            summary.write_text(
                textwrap.dedent(
                    """\
                    device\tmodel\tmode\tcase\tvariant\tsuccess\tdecode_tps\tacceptance_pct
                    cuda:0\tdense\tstochastic\tdefault\tfixed_d1\ttrue\t10.0\t65.0
                    cuda:0\tdense\tstochastic\tdefault\tfixed_d2\ttrue\t11.0\t76.0
                    cuda:0\tdense\tstochastic\tdefault\tfixed_d3\ttrue\t12.0\t81.0
                    rocm:0\tdense\tstochastic\tdefault\tfixed_d1\ttrue\t10.0\t84.0
                    rocm:0\tdense\tstochastic\tdefault\tfixed_d2\ttrue\t9.0\t67.0
                    rocm:0\tdense\tstochastic\tdefault\tfixed_d3\ttrue\t8.0\t65.0
                    """
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "python3",
                    str(TRAINER),
                    "--input",
                    str(summary),
                    "--output",
                    str(output),
                    "--summary",
                    str(report),
                    "--holdout-modulus",
                    "999",
                    "--holdout-bucket",
                    "998",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            include_text = output.read_text(encoding="utf-8")
            self.assertIn("trained_cuda_dense_stochastic_promote_d1_to_d3", include_text)
            self.assertIn("trained_cuda_dense_stochastic_promote_d2_to_d3", include_text)
            self.assertIn("trained_rocm_dense_stochastic_demote_d2_to_d1", include_text)
            self.assertIn("trained_rocm_dense_stochastic_demote_d3_to_d1", include_text)
            self.assertNotIn("trained_rocm_dense_stochastic_promote_d1", include_text)
            self.assertNotIn("trained_cuda_dense_stochastic_demote_d2", include_text)

    def test_model_class_policy_keeps_dense_and_moe_rows_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary = tmp / "summary.tsv"
            output = tmp / "MTPDepthPolicyGenerated.inc"
            report = tmp / "report.txt"
            summary.write_text(
                textwrap.dedent(
                    """\
                    device\tmodel\tmode\tcase\tvariant\tsuccess\tdecode_tps\tacceptance_pct
                    rocm:0\tdense\tgreedy\tdefault\tfixed_d1\ttrue\t10.0\t90.0
                    rocm:0\tdense\tgreedy\tdefault\tfixed_d2\ttrue\t12.0\t92.0
                    rocm:0\tdense\tgreedy\tdefault\tfixed_d3\ttrue\t14.0\t98.0
                    rocm:0\tmoe\tgreedy\tdefault\tfixed_d1\ttrue\t10.0\t82.0
                    rocm:0\tmoe\tgreedy\tdefault\tfixed_d2\ttrue\t12.0\t76.0
                    rocm:0\tmoe\tgreedy\tdefault\tfixed_d3\ttrue\t16.0\t81.0
                    """
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "python3",
                    str(TRAINER),
                    "--input",
                    str(summary),
                    "--output",
                    str(output),
                    "--summary",
                    str(report),
                    "--holdout-modulus",
                    "999",
                    "--holdout-bucket",
                    "998",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            include_text = output.read_text(encoding="utf-8")
            self.assertIn("trained_rocm_dense_greedy_promote_d1_to_d3", include_text)
            self.assertIn("trained_rocm_dense_greedy_promote_d2_to_d3", include_text)
            self.assertIn("trained_rocm_dense_greedy_hold_d3_to_d3", include_text)
            self.assertIn("trained_rocm_moe_greedy_promote_d1_to_d3", include_text)
            self.assertIn("trained_rocm_moe_greedy_promote_d2_to_d3", include_text)
            self.assertIn("trained_rocm_moe_greedy_hold_d3_to_d3", include_text)
            self.assertIn("MTPDepthPolicyModelClass::Dense, 1, 0.900000", include_text)
            self.assertIn("MTPDepthPolicyModelClass::MoE, 1, 0.820000", include_text)
            self.assertIn("MTPDepthPolicyModelClass::Dense, 2, 0.920000", include_text)
            self.assertIn("MTPDepthPolicyModelClass::MoE, 2, 0.760000", include_text)
            self.assertIn("MTPDepthPolicyModelClass::MoE, 3, 0.760000", include_text)
            self.assertIn(", +2, \"trained_rocm_moe_greedy_promote_d1_to_d3\"", include_text)


if __name__ == "__main__":
    unittest.main()
