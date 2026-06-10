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
    def run_matrix(self, variants: str, *, allow_partial: bool = False) -> subprocess.CompletedProcess[str]:
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
                "--devices",
                "cpu:0",
                "--models",
                "dense",
                "--modes",
                "greedy",
                "--variants",
                variants,
                "--output-dir",
                str(tmp_path / "out"),
            ]
            if allow_partial:
                cmd.append("--allow-partial-variants")
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
            self.assertEqual(len(lines[0].split("\t")), len(lines[1].split("\t")))
            self.assertEqual(len(lines[0].split("\t")), 59)


if __name__ == "__main__":
    unittest.main()
