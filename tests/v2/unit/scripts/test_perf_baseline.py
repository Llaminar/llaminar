#!/usr/bin/env python3
"""Contract tests for scripts/perf_baseline.py.

Every case drives the real command line. ``run`` executes a measuring stand-in
that lives inside a synthetic build tree, so provenance binding (executable
location and digest, fresh result, exit status, clean Release tree) is proven
through the same path users run. ``diff`` negative controls feed hostile
documents to the diff CLI itself, not only to the recorder.
"""

from __future__ import annotations

import copy
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "perf_baseline.py"


def benchmark_document(prefill: float = 400.0, decode: float = 20.0, *, temperature: float = 0.0,
                       tokens: list[int] | None = None, success: bool = True) -> dict:
    """A llaminar.benchmark.v1 document with the fields the recorder binds."""

    return {
        "schema": "llaminar.benchmark.v1", "success": success, "failure_reason": "",
        "prefill_success": True, "decode_success": True,
        "measurement_iterations": 3, "warmup_iterations": 1,
        "prompt": {"source": "inline", "bytes": 64, "sha256": "a" * 64},
        "tokens": {"prefill": 512, "decode": 256, "decode_after_prefill": 255, "total": 768},
        "timing_ms": {"prefill": 1280.0, "decode": 12800.0, "total": 14080.0},
        "throughput_tokens_per_sec": {"prefill": prefill, "decode": decode, "decode_after_prefill": decode,
                                      "overall": 54.5},
        "decode_latency_ms": {"mean": 1000.0 / decode, "p50": 1000.0 / decode, "p90": 60.0, "samples": 256},
        "generated_token_ids": tokens if tokens is not None else [1, 2, 3],
        "runtime_state": {"architecture": "qwen35moe", "execution_path": "single", "primary_device": "cpu:0",
                          "current_position": 768},
        "runtime_configuration": {"sampling": {"temperature": temperature, "seed": 42}, "mtp_enabled": False},
    }


def kernel_document(value: float, samples: list[float] | None = None) -> dict:
    metric = {"value": value, "unit": "GB/s", "better": "higher"}
    if samples is not None:
        metric["samples"] = samples
    return {"harness": "Perf__X", "parameters": {"format": "Q4_K"}, "metrics": {"gbps": metric}}


class PerfBaselineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "src"
        self.source.mkdir()
        for command in (["init", "-q"], ["config", "user.email", "t@example.invalid"], ["config", "user.name", "t"]):
            subprocess.run(["git", "-C", str(self.source), *command], check=True)
        (self.source / "file.txt").write_text("one\n")
        subprocess.run(["git", "-C", str(self.source), "add", "file.txt"], check=True)
        subprocess.run(["git", "-C", str(self.source), "commit", "-q", "-m", "init"], check=True)
        self.release = self.make_build("release", "Release")
        self.counter = 0

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_build(self, name: str, build_type: str, compiler_version: str = "14.2.0") -> Path:
        build = self.root / name
        (build / "CMakeFiles" / "4.0.3").mkdir(parents=True)
        (build / "CMakeCache.txt").write_text(
            "# comment\n//doc\n"
            f"CMAKE_BUILD_TYPE:STRING={build_type}\nLLAMINAR_CPU_ISA:STRING=NEON_DOTPROD\n"
            "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++-14\nCMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG\n"
            "HAVE_CUDA:BOOL=OFF\nHAVE_ROCM:BOOL=OFF\n")
        (build / "CMakeFiles" / "4.0.3" / "CMakeCXXCompiler.cmake").write_text(
            f'set(CMAKE_CXX_COMPILER_VERSION "{compiler_version}")\n')
        return build

    def stand_in(self, build: Path, document: dict | None, *, exit_code: int = 0, mutate: bool = False) -> Path:
        """A measuring executable inside the build tree that writes its result to argv[1]."""

        self.counter += 1
        payload = self.root / f"payload{self.counter}.json"
        if document is not None:
            payload.write_text(json.dumps(document))
        script = build / f"measure{self.counter}"
        body = [f"#!{sys.executable}", "import shutil, sys"]
        if document is not None:
            body.append(f"shutil.copyfile({str(payload)!r}, sys.argv[1])")
        if mutate:
            body.append("open(__file__, 'a').write('# mutated\\n')")
        body.append(f"sys.exit({exit_code})")
        script.write_text("\n".join(body) + "\n")
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        return script

    def cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)

    def run_record(self, document: dict | None, *extra: str, kind: str = "llaminar-benchmark",
                   build: Path | None = None, exit_code: int = 0, mutate: bool = False,
                   executable: Path | None = None) -> tuple[subprocess.CompletedProcess[str], Path]:
        build = build or self.release
        script = executable or self.stand_in(build, document, exit_code=exit_code, mutate=mutate)
        self.counter += 1
        result, output = self.root / f"result{self.counter}.json", self.root / f"baseline{self.counter}.json"
        proc = self.cli("run", "--kind", kind, "--build-dir", str(build), "--source-dir", str(self.source),
                        "--output", str(output), "--result-json", str(result), *extra, "--", str(script), str(result))
        return proc, output

    def import_record(self, document: dict, *extra: str, kind: str = "llaminar-benchmark") -> tuple[subprocess.CompletedProcess[str], Path]:
        self.counter += 1
        source, output = self.root / f"import{self.counter}.json", self.root / f"baseline{self.counter}.json"
        source.write_text(json.dumps(document))
        proc = self.cli("import", "--kind", kind, "--input", str(source), "--build-dir", str(self.release),
                        "--source-dir", str(self.source), "--output", str(output), *extra)
        return proc, output

    def diff(self, a: Path, b: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return self.cli("diff", "--baseline", str(a), "--candidate", str(b), "--json", *extra)

    # --- provenance -------------------------------------------------------------------------
    def test_run_binds_executable_launch_and_inputs(self) -> None:
        model = self.root / "model.gguf"
        model.write_bytes(b"GGUF")
        proc, output = self.run_record(benchmark_document(), "--input-file", f"model={model}")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        document = json.loads(output.read_text())
        self.assertEqual(document["grade"], "bound")
        self.assertEqual(len(document["workload"]["declared_inputs"]["model"]), 64)
        self.assertTrue(all(len(d) == 64 for d in document["launch"]["binaries"].values()))
        self.assertEqual(len(document["launch"]["result_sha256"]), 64)
        self.assertEqual(document["build"]["cxx_compiler_version"], "14.2.0")
        self.assertIn("NEON_DOTPROD", document["build"]["cpu_isa"])

    def test_import_is_always_unbound_even_for_a_clean_release_tree(self) -> None:
        proc, output = self.import_record(benchmark_document())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(output.read_text())["grade"], "unbound")

    def test_run_refusals(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        foreign = self.stand_in(outside, benchmark_document())
        cases = {
            "outside build dir": self.run_record(None, executable=foreign)[0],
            "nonzero exit": self.run_record(benchmark_document(), exit_code=3)[0],
            "no result written": self.run_record(None)[0],
            "binary changed during run": self.run_record(benchmark_document(), mutate=True)[0],
            "failed benchmark": self.run_record(benchmark_document(success=False))[0],
        }
        for name, proc in cases.items():
            with self.subTest(name):
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        script = self.stand_in(self.release, benchmark_document())
        existing = self.root / "existing.json"
        existing.write_text("{}")
        proc = self.cli("run", "--kind", "llaminar-benchmark", "--build-dir", str(self.release), "--source-dir",
                        str(self.source), "--output", str(self.root / "x.json"), "--result-json", str(existing),
                        "--", str(script), str(existing))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("already exists", proc.stderr)

    def test_dirty_or_non_release_runs_are_unbound(self) -> None:
        (self.source / "file.txt").write_text("two\n")
        self.assertEqual(self.run_record(benchmark_document())[0].returncode, 2)
        proc, output = self.run_record(benchmark_document(), "--allow-dirty")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(output.read_text())["grade"], "unbound")
        subprocess.run(["git", "-C", str(self.source), "checkout", "-q", "--", "file.txt"], check=True)
        proc, output = self.run_record(benchmark_document(), build=self.make_build("integration", "Integration"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(output.read_text())["grade"], "unbound")

    # --- comparability ----------------------------------------------------------------------
    def test_diff_verdicts_and_bound_grade(self) -> None:
        _, base = self.run_record(benchmark_document())
        _, same = self.run_record(benchmark_document(prefill=401.0))
        _, slower = self.run_record(benchmark_document(decode=19.0))
        _, faster = self.run_record(benchmark_document(decode=21.0))
        report = self.diff(base, same)
        self.assertEqual(report.returncode, 0, report.stderr)
        self.assertEqual(json.loads(report.stdout)["grade"], "bound")
        regressed = self.diff(base, slower)
        self.assertEqual(regressed.returncode, 1, regressed.stderr)
        self.assertEqual(json.loads(regressed.stdout)["regressions"],
                         ["decode_latency_mean_ms", "decode_latency_p50_ms", "decode_tokens_per_sec"])
        verdicts = {r["metric"]: r["verdict"] for r in json.loads(self.diff(base, faster).stdout)["metrics"]}
        self.assertEqual(verdicts["decode_tokens_per_sec"], "improved")
        self.assertEqual(verdicts["prefill_tokens_per_sec"], "within-tolerance")
        self.assertEqual(self.diff(base, slower, "--threshold-pct", "6").returncode, 0)
        _, imported = self.import_record(benchmark_document())
        self.assertIn("launch_env", self.diff(base, imported).stderr)
        mixed = self.diff(base, imported, "--allow-identity-change", "launch_env")
        self.assertEqual(json.loads(mixed.stdout)["grade"], "unbound")

    def test_incomparable_identity_is_refused_unless_declared(self) -> None:
        model_a, model_b = self.root / "a.gguf", self.root / "b.gguf"
        model_a.write_bytes(b"A")
        model_b.write_bytes(b"B")
        _, base = self.run_record(benchmark_document(), "--input-file", f"model={model_a}")
        cases = {
            "declared_inputs": self.run_record(benchmark_document(), "--input-file", f"model={model_b}")[1],
            "workload": self.run_record(benchmark_document(temperature=0.7), "--input-file", f"model={model_a}")[1],
            "output": self.run_record(benchmark_document(tokens=[9]), "--input-file", f"model={model_a}")[1],
            "compiler": self.run_record(benchmark_document(), "--input-file", f"model={model_a}",
                                        build=self.make_build("gcc15", "Release", "15.1.0"))[1],
        }
        for field, candidate in cases.items():
            with self.subTest(field):
                refused = self.diff(base, candidate)
                self.assertEqual(refused.returncode, 2)
                self.assertIn(field, refused.stderr)
                self.assertEqual(self.diff(base, candidate, "--allow-identity-change", field).returncode, 0)
        env = dict(os.environ, OMP_NUM_THREADS="3")
        script = self.stand_in(self.release, benchmark_document())
        result, output = self.root / "env-result.json", self.root / "env-baseline.json"
        subprocess.run([sys.executable, str(SCRIPT), "run", "--kind", "llaminar-benchmark", "--build-dir",
                        str(self.release), "--source-dir", str(self.source), "--output", str(output), "--result-json",
                        str(result), "--input-file", f"model={model_a}", "--", str(script), str(result)],
                       env=env, check=True, capture_output=True)
        self.assertIn("launch_env", self.diff(base, output).stderr)

    def test_robust_spread(self) -> None:
        _, quiet_a = self.import_record(kernel_document(100.0, [100.0, 100.1, 99.9]), kind="kernel-metrics")
        _, quiet_b = self.import_record(kernel_document(96.0, [96.0, 96.1, 95.9]), kind="kernel-metrics")
        _, noisy_a = self.import_record(kernel_document(100.0, [90.0, 100.0, 110.0]), kind="kernel-metrics")
        _, noisy_b = self.import_record(kernel_document(96.0, [86.0, 96.0, 106.0]), kind="kernel-metrics")
        _, stall_a = self.import_record(kernel_document(100.0, [100.0, 100.1, 99.9, 100.0, 1.0]), kind="kernel-metrics")
        _, stall_b = self.import_record(kernel_document(96.0, [96.0, 96.1, 95.9, 96.0, 1.0]), kind="kernel-metrics")
        self.assertEqual(self.diff(quiet_a, quiet_b).returncode, 1)
        self.assertEqual(self.diff(noisy_a, noisy_b).returncode, 0)
        self.assertEqual(self.diff(stall_a, stall_b).returncode, 1)

    # --- validation at every entry point --------------------------------------------------
    def test_import_rejects_invalid_inputs(self) -> None:
        bad_kernel = [kernel_document(0.0), kernel_document(1.0, []), {"harness": "X", "parameters": {}, "metrics": {}},
                      {"harness": "X", "parameters": {}, "metrics": {"m": {"value": 1.0, "unit": "u"}}},
                      {"harness": "X", "metrics": {"m": {"value": 1.0, "unit": "u", "better": "higher"}}}]
        for index, document in enumerate(bad_kernel):
            with self.subTest(index=index):
                self.assertEqual(self.import_record(document, kind="kernel-metrics")[0].returncode, 2)
        missing = benchmark_document()
        del missing["runtime_configuration"]
        self.assertEqual(self.import_record(missing)[0].returncode, 2)
        nan_text = self.root / "nan.json"
        nan_text.write_text(json.dumps(benchmark_document()).replace('"decode": 20.0', '"decode": NaN', 1))
        proc = self.cli("import", "--kind", "llaminar-benchmark", "--input", str(nan_text), "--build-dir",
                        str(self.release), "--source-dir", str(self.source), "--output", str(self.root / "n.json"))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("NaN", proc.stderr)

    def test_diff_cli_rejects_hostile_documents_and_thresholds(self) -> None:
        _, good = self.import_record(kernel_document(10.0, [10.0, 10.1]), kind="kernel-metrics")
        valid = json.loads(good.read_text())

        def hostile(mutator, raw_text: str | None = None) -> Path:
            self.counter += 1
            path = self.root / f"hostile{self.counter}.json"
            document = copy.deepcopy(valid)
            mutator(document)
            path.write_text(raw_text if raw_text is not None else json.dumps(document))
            return path

        text = json.dumps(valid)
        cases = {
            "NaN literal": hostile(lambda d: None, text.replace('"value": 10.0', '"value": NaN', 1)),
            "Infinity sample": hostile(lambda d: None, text.replace("10.1", "Infinity", 1)),
            "empty metrics": hostile(lambda d: d.update(metrics={})),
            "zero value": hostile(lambda d: d["metrics"]["gbps"].update(value=0.0)),
            "bad direction": hostile(lambda d: d["metrics"]["gbps"].update(better="up")),
            "missing workload": hostile(lambda d: d.pop("workload")),
            "forged bound grade": hostile(lambda d: d.update(grade="bound")),
            "wrong schema": hostile(lambda d: d.update(schema="llaminar.perf-baseline.v1")),
        }
        for name, path in cases.items():
            with self.subTest(name):
                for order in ((good, path), (path, good)):
                    self.assertEqual(self.diff(*order).returncode, 2, name)
        for threshold in ("inf", "nan", "-1"):
            with self.subTest(threshold=threshold):
                self.assertEqual(self.diff(good, good, "--threshold-pct", threshold).returncode, 2)


if __name__ == "__main__":
    unittest.main()
