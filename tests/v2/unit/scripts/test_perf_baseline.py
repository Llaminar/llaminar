#!/usr/bin/env python3
"""Contract tests for scripts/perf_baseline.py.

Every case drives the real command line against a temporary git repository and
a synthetic CMakeCache.txt, so identity capture, refusal rules and the
regression verdict are proven through the same entry point users run.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "perf_baseline.py"


def benchmark_document(prefill: float = 400.0, decode: float = 20.0, prompt: str = "a" * 64) -> dict:
    """Return the subset of llaminar.benchmark.v1 the recorder consumes, plus unrelated fields."""

    return {
        "schema": "llaminar.benchmark.v1",
        "prompt": {"source": "inline", "sha256": prompt},
        "tokens": {"prefill": 512, "decode": 256},
        "timing_ms": {"prefill": 1280.0, "decode": 12800.0},
        "throughput_tokens_per_sec": {"prefill": prefill, "decode": decode},
        "decode_latency_ms": {"mean": 1000.0 / decode, "p50": 1000.0 / decode, "samples": 256},
        "runtime_state": {"architecture": "qwen35moe"},
    }


class PerfBaselineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "src"
        self.source.mkdir()
        for command in (["init", "-q"], ["config", "user.email", "t@example.invalid"],
                        ["config", "user.name", "t"]):
            subprocess.run(["git", "-C", str(self.source), *command], check=True)
        (self.source / "file.txt").write_text("one\n")
        subprocess.run(["git", "-C", str(self.source), "add", "file.txt"], check=True)
        subprocess.run(["git", "-C", str(self.source), "commit", "-q", "-m", "init"], check=True)
        self.commit = subprocess.run(["git", "-C", str(self.source), "rev-parse", "HEAD"], check=True,
                                     capture_output=True, text=True).stdout.strip()
        self.release = self.make_build("release", "Release", "NEON_DOTPROD")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_build(self, name: str, build_type: str, isa: str) -> Path:
        build = self.root / name
        build.mkdir()
        (build / "CMakeCache.txt").write_text(
            "# comment\n//doc line\n"
            f"CMAKE_BUILD_TYPE:STRING={build_type}\nLLAMINAR_CPU_ISA:STRING={isa}\n"
            "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++-14\nHAVE_CUDA:BOOL=OFF\nHAVE_ROCM:BOOL=OFF\n")
        compiler = build / "CMakeFiles" / "4.0.3"
        compiler.mkdir(parents=True)
        (compiler / "CMakeCXXCompiler.cmake").write_text('set(CMAKE_CXX_COMPILER_VERSION "14.2.0")\n')
        return build

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)

    def record(self, document: dict, name: str, *extra: str, kind: str = "llaminar-benchmark",
               build: Path | None = None) -> tuple[subprocess.CompletedProcess[str], Path]:
        input_path = self.root / f"{name}.input.json"
        input_path.write_text(json.dumps(document))
        output = self.root / f"{name}.baseline.json"
        result = self.run_cli("record", "--kind", kind, "--input", str(input_path), "--build-dir",
                              str(build or self.release), "--source-dir", str(self.source), "--output",
                              str(output), *extra)
        return result, output

    def test_record_stamps_identity_and_normalizes_benchmark_metrics(self) -> None:
        result, output = self.record(benchmark_document(), "a", "--run-command", "llaminar2 benchmark -n 256")
        self.assertEqual(result.returncode, 0, result.stderr)
        baseline = json.loads(output.read_text())
        self.assertEqual(baseline["schema"], "llaminar.perf-baseline.v1")
        self.assertEqual(baseline["source"], {"commit": self.commit, "dirty": False})
        self.assertEqual(baseline["build"]["build_type"], "Release")
        self.assertEqual(baseline["build"]["cpu_isa"], "NEON_DOTPROD")
        self.assertEqual(baseline["build"]["cxx_compiler_version"], "14.2.0")
        self.assertTrue(baseline["certifying"])
        self.assertEqual(baseline["workload"], {"prompt_sha256": "a" * 64, "prefill_tokens": 512,
                                                "decode_tokens": 256})
        self.assertEqual(baseline["metrics"]["decode_tokens_per_sec"],
                         {"value": 20.0, "unit": "tok/s", "better": "higher"})
        self.assertEqual(baseline["metrics"]["decode_latency_mean_ms"]["better"], "lower")
        self.assertEqual(baseline["raw"]["runtime_state"]["architecture"], "qwen35moe")
        self.assertEqual(baseline["command"], "llaminar2 benchmark -n 256")
        self.assertGreater(baseline["host"]["page_size"], 0)

    def test_dirty_tree_is_refused_unless_allowed_and_then_digested(self) -> None:
        (self.source / "file.txt").write_text("two\n")
        refused, _ = self.record(benchmark_document(), "dirty")
        self.assertEqual(refused.returncode, 2)
        self.assertIn("uncommitted", refused.stderr)
        allowed, output = self.record(benchmark_document(), "dirty-ok", "--allow-dirty")
        self.assertEqual(allowed.returncode, 0, allowed.stderr)
        source = json.loads(output.read_text())["source"]
        self.assertTrue(source["dirty"])
        self.assertEqual(len(source["diff_sha256"]), 64)

    def test_non_release_requires_diagnostic_and_never_certifies(self) -> None:
        integration = self.make_build("integration", "Integration", "NEON_DOTPROD")
        refused, _ = self.record(benchmark_document(), "int", build=integration)
        self.assertEqual(refused.returncode, 2)
        self.assertIn("not Release", refused.stderr)
        allowed, output = self.record(benchmark_document(), "int-diag", "--diagnostic", build=integration)
        self.assertEqual(allowed.returncode, 0, allowed.stderr)
        self.assertFalse(json.loads(output.read_text())["certifying"])

    def test_invalid_inputs_fail_closed(self) -> None:
        wrong = benchmark_document()
        wrong["schema"] = "llaminar.benchmark.v0"
        self.assertEqual(self.record(wrong, "schema")[0].returncode, 2)
        missing = benchmark_document()
        del missing["decode_latency_ms"]
        self.assertEqual(self.record(missing, "missing")[0].returncode, 2)
        nan = benchmark_document(decode=float("nan"))
        self.assertEqual(self.record(nan, "nan")[0].returncode, 2)
        no_direction = {"harness": "Perf__X", "metrics": {"gbps": {"value": 1.0, "unit": "GB/s"}}}
        self.assertEqual(self.record(no_direction, "dir", kind="kernel-metrics")[0].returncode, 2)
        empty = {"harness": "Perf__X", "metrics": {}}
        self.assertEqual(self.record(empty, "empty", kind="kernel-metrics")[0].returncode, 2)

    def diff(self, a: Path, b: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return self.run_cli("diff", "--baseline", str(a), "--candidate", str(b), "--json", *extra)

    def test_diff_reports_regression_improvement_and_tolerance(self) -> None:
        _, base = self.record(benchmark_document(), "base")
        _, same = self.record(benchmark_document(prefill=401.0), "same")
        _, slower = self.record(benchmark_document(decode=19.0), "slower")
        _, faster = self.record(benchmark_document(decode=21.0), "faster")
        unchanged = self.diff(base, same)
        self.assertEqual(unchanged.returncode, 0, unchanged.stderr)
        self.assertEqual(json.loads(unchanged.stdout)["regressions"], [])
        regressed = self.diff(base, slower)
        self.assertEqual(regressed.returncode, 1, regressed.stderr)
        report = json.loads(regressed.stdout)
        # Lower throughput and higher latency are both regressions in their own direction.
        self.assertEqual(report["regressions"],
                         ["decode_latency_mean_ms", "decode_latency_p50_ms", "decode_tokens_per_sec"])
        improved = self.diff(base, faster)
        self.assertEqual(improved.returncode, 0, improved.stderr)
        verdicts = {row["metric"]: row["verdict"] for row in json.loads(improved.stdout)["metrics"]}
        self.assertEqual(verdicts["decode_tokens_per_sec"], "improved")
        self.assertEqual(verdicts["prefill_tokens_per_sec"], "within-tolerance")
        self.assertEqual(self.diff(base, slower, "--threshold-pct", "6").returncode, 0)

    def test_sample_noise_widens_tolerance(self) -> None:
        def kernel(value: float, samples: list[float]) -> dict:
            return {"harness": "Perf__CPUNativeNEONDecodeBandwidth", "parameters": {"format": "Q4_K", "m": 1},
                    "metrics": {"gbps": {"value": value, "unit": "GB/s", "better": "higher", "samples": samples}}}
        _, quiet_a = self.record(kernel(100.0, [100.0, 100.1, 99.9]), "qa", kind="kernel-metrics")
        _, quiet_b = self.record(kernel(96.0, [96.0, 96.1, 95.9]), "qb", kind="kernel-metrics")
        _, noisy_a = self.record(kernel(100.0, [90.0, 100.0, 110.0]), "na", kind="kernel-metrics")
        _, noisy_b = self.record(kernel(96.0, [86.0, 96.0, 106.0]), "nb", kind="kernel-metrics")
        self.assertEqual(self.diff(quiet_a, quiet_b).returncode, 1)
        noisy = self.diff(noisy_a, noisy_b)
        self.assertEqual(noisy.returncode, 0, noisy.stderr)
        self.assertGreater(json.loads(noisy.stdout)["metrics"][0]["tolerance_pct"], 4.0)
        # One scheduler stall must not hide a real regression of stable samples.
        _, stall_a = self.record(kernel(100.0, [100.0, 100.1, 99.9, 100.0, 1.0]), "sa", kind="kernel-metrics")
        _, stall_b = self.record(kernel(96.0, [96.0, 96.1, 95.9, 96.0, 1.0]), "sb", kind="kernel-metrics")
        self.assertEqual(self.diff(stall_a, stall_b).returncode, 1)

    def test_incomparable_identity_is_refused_unless_named(self) -> None:
        _, base = self.record(benchmark_document(), "base")
        _, other_prompt = self.record(benchmark_document(prompt="b" * 64), "prompt")
        _, other_isa = self.record(benchmark_document(), "isa", build=self.make_build("i8mm", "Release", "NEON_I8MM"))
        refused = self.diff(base, other_prompt)
        self.assertEqual(refused.returncode, 2)
        self.assertIn("workload", refused.stderr)
        isa = self.diff(base, other_isa)
        self.assertEqual(isa.returncode, 2)
        self.assertIn("cpu_isa", isa.stderr)
        allowed = self.diff(base, other_isa, "--allow-identity-change", "cpu_isa")
        self.assertEqual(allowed.returncode, 0, allowed.stderr)
        self.assertEqual(json.loads(allowed.stdout)["allowed_identity_changes"], ["cpu_isa"])
        _, kernel = self.record({"harness": "Perf__X", "metrics": {"gbps": {"value": 1.0, "unit": "GB/s",
                                 "better": "higher"}}}, "kernel", kind="kernel-metrics")
        self.assertEqual(self.diff(base, kernel).returncode, 2)

    def test_diagnostic_record_makes_the_comparison_non_certifying(self) -> None:
        _, base = self.record(benchmark_document(), "base")
        _, diagnostic = self.record(benchmark_document(), "diag", "--diagnostic")
        report = self.diff(base, diagnostic)
        self.assertEqual(report.returncode, 0, report.stderr)
        self.assertFalse(json.loads(report.stdout)["certifying"])


if __name__ == "__main__":
    unittest.main()
