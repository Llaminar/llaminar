#!/usr/bin/env python3
"""Contract tests for scripts/perf_baseline.py.

``run`` is exercised with a real dynamic executable: a copy of this Python
interpreter placed inside a synthetic build tree, so ldd observation, executable
admission and digest checks operate on an actual ELF. The measuring script is
passed as an argument and takes its behaviour from PERF_* variables, which are
not part of the recorded launch environment, so repeated runs share argv.
On non-Linux hosts the suite asserts that ``run`` refuses to operate.

``diff`` negative controls feed hostile documents to the diff CLI in both
argument orders. Fixtures under fixtures/perf_baseline/ are two real outputs of
Perf__CPUNativeQuantised_DecodeBandwidth (Q4_K, 1 thread, 128 KiB, 5 iterations)
that differ only in measured values.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "perf_baseline.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "perf_baseline"
LINUX = sys.platform.startswith("linux")

MEASURE = """import os, shutil, sys
action = os.environ.get("PERF_ACTION", "")
if action == "replace-exe":
    exe = os.environ["PERF_EXE"]
    shutil.copy2(exe, exe + ".new"); open(exe + ".new", "ab").write(b"x"); os.replace(exe + ".new", exe)
if action == "check-argv0" and sys.orig_argv[0] != os.environ["PERF_EXE"]:
    sys.exit(4)
if action == "touch-input":
    open(os.environ["PERF_INPUT"], "ab").write(b"x")
if action != "nowrite":
    shutil.copyfile(os.environ["PERF_PAYLOAD"], sys.argv[1])
sys.exit(3 if action == "exit3" else 0)
"""


def benchmark_document(prefill: float = 400.0, decode: float = 20.0, *, temperature: float = 0.0,
                       tokens: list[int] | None = None, success: bool = True) -> dict:
    return {
        "schema": "llaminar.benchmark.v1", "success": success, "failure_reason": "",
        "prefill_success": True, "decode_success": True, "measurement_iterations": 3, "warmup_iterations": 1,
        "prompt": {"source": "inline", "bytes": 64, "sha256": "a" * 64},
        "tokens": {"prefill": 512, "decode": 256, "decode_after_prefill": 255, "total": 768},
        "throughput_tokens_per_sec": {"prefill": prefill, "decode": decode},
        "decode_latency_ms": {"mean": 1000.0 / decode, "p50": 1000.0 / decode, "p90": 60.0, "samples": 256},
        "generated_token_ids": tokens if tokens is not None else [1, 2, 3],
        "runtime_state": {"architecture": "qwen35moe", "execution_path": "single", "primary_device": "cpu:0"},
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
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"GGUF-A")
        self.measure = self.root / "measure.py"
        self.measure.write_text(MEASURE)
        self.counter = 0

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_build(self, name: str, build_type: str, compiler_version: str = "14.2.0") -> Path:
        build = self.root / name
        (build / "CMakeFiles" / "4.0.3").mkdir(parents=True)
        (build / "CMakeCache.txt").write_text(
            f"# c\n//d\nCMAKE_BUILD_TYPE:STRING={build_type}\nLLAMINAR_CPU_ISA:STRING=NEON_DOTPROD\n"
            "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++-14\nCMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG\n"
            "HAVE_CUDA:BOOL=OFF\nHAVE_ROCM:BOOL=OFF\n")
        (build / "CMakeFiles" / "4.0.3" / "CMakeCXXCompiler.cmake").write_text(
            f'set(CMAKE_CXX_COMPILER_VERSION "{compiler_version}")\n')
        if LINUX:
            shutil.copyfile(Path(sys.executable).resolve(), build / "py")
            (build / "py").chmod(0o755)
        return build

    def cli(self, *args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, env=env)

    def run_record(self, payload: dict | Path | None, *extra: str, kind: str = "llaminar-benchmark",
                   build: Path | None = None, executable: Path | None = None, action: str = "",
                   model: Path | None = None, env_extra: dict | None = None) -> tuple[subprocess.CompletedProcess[str], Path]:
        build = build or self.release
        self.counter += 1
        payload_path = self.root / f"payload{self.counter}.json"
        if isinstance(payload, Path):
            payload_path = payload
        elif payload is not None:
            payload_path.write_text(json.dumps(payload))
        result, output = self.root / f"result{self.counter}.json", self.root / f"baseline{self.counter}.json"
        env = dict(os.environ, PYTHONHOME=sys.base_prefix, PERF_PAYLOAD=str(payload_path), PERF_ACTION=action,
                   PERF_EXE=str(build / "py"), PERF_INPUT=str(model or self.model), **(env_extra or {}))
        inputs = ["--input-file", f"model={model or self.model}"] if kind == "llaminar-benchmark" else []
        proc = self.cli("run", "--kind", kind, "--build-dir", str(build), "--source-dir", str(self.source),
                        "--output", str(output), "--result-json", str(result), *inputs, *extra, "--",
                        str(executable or build / "py"), str(self.measure), str(result), env=env)
        return proc, output

    def import_record(self, document: dict | Path, *extra: str, kind: str = "llaminar-benchmark",
                      with_model: bool = True) -> tuple[subprocess.CompletedProcess[str], Path]:
        self.counter += 1
        output = self.root / f"baseline{self.counter}.json"
        if isinstance(document, Path):
            source = document
        else:
            source = self.root / f"import{self.counter}.json"
            source.write_text(json.dumps(document))
        inputs = ["--input-file", f"model={self.model}"] if kind == "llaminar-benchmark" and with_model else []
        proc = self.cli("import", "--kind", kind, "--input", str(source), "--build-dir", str(self.release),
                        "--source-dir", str(self.source), "--output", str(output), *inputs, *extra)
        return proc, output

    def diff(self, a: Path, b: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return self.cli("diff", "--baseline", str(a), "--candidate", str(b), "--json", *extra)

    def require_linux_run(self) -> bool:
        """On non-Linux hosts prove run refuses, and tell the caller to stop."""

        if LINUX:
            return True
        proc, _ = self.run_record(benchmark_document(), executable=Path(sys.executable))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("Linux", proc.stderr)
        return False

    # --- provenance ------------------------------------------------------------------------
    def test_run_binds_one_admitted_executable_and_launch(self) -> None:
        if not self.require_linux_run():
            return
        link = self.release / "measure-link"
        link.symlink_to(self.release / "py")
        # The measure exits 4 unless it was executed through the resolved admitted path.
        proc, output = self.run_record(benchmark_document(), executable=link, action="check-argv0")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        document = json.loads(output.read_text())
        launch = document["launch"]
        self.assertEqual(document["grade"], "bound")
        self.assertEqual(launch["argv"][0], "py", "the resolved admitted executable is what runs and is recorded")
        self.assertEqual(launch["argv"][-1], "<result-json>")
        self.assertIn("py", launch["binaries"])
        self.assertTrue(launch["cpu_affinity"])
        self.assertIn("ldd", launch["loader_observation"])
        self.assertEqual(len(document["workload"]["declared_inputs"]["model"]), 64)
        outside = self.root / "outside-link"
        outside.symlink_to(Path(sys.executable).resolve())
        moved = self.release / "escape"
        moved.symlink_to(outside)
        self.assertEqual(self.run_record(benchmark_document(), executable=moved)[0].returncode, 2)

    def test_run_refusals(self) -> None:
        if not self.require_linux_run():
            return
        def fresh(name: str) -> Path:
            return self.make_build(f"refusal-{name}", "Release")

        cases = {
            "interpreter outside build dir": self.run_record(benchmark_document(), executable=Path(sys.executable)),
            "nonzero exit": self.run_record(benchmark_document(), action="exit3"),
            "no result written": self.run_record(benchmark_document(), action="nowrite"),
            "executable replaced during run": self.run_record(benchmark_document(), action="replace-exe",
                                                              build=fresh("replace")),
            "declared input changed during run": self.run_record(benchmark_document(), action="touch-input"),
            "failed benchmark": self.run_record(benchmark_document(success=False)),
            "script is not a dynamic executable": self.run_record(benchmark_document(), executable=self.copy_script()),
        }
        for name, (proc, _) in cases.items():
            with self.subTest(name):
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        payload = self.root / "p.json"
        payload.write_text(json.dumps(benchmark_document()))
        no_model = self.cli("run", "--kind", "llaminar-benchmark", "--build-dir", str(self.release), "--source-dir",
                            str(self.source), "--output", str(self.root / "o.json"), "--result-json",
                            str(self.root / "r.json"), "--", str(self.release / "py"), str(self.measure),
                            str(self.root / "r.json"),
                            env=dict(os.environ, PYTHONHOME=sys.base_prefix, PERF_PAYLOAD=str(payload)))
        self.assertEqual(no_model.returncode, 2)
        self.assertIn("model", no_model.stderr)
        existing = self.root / "existing.json"
        existing.write_text("{}")
        proc = self.cli("run", "--kind", "llaminar-benchmark", "--build-dir", str(self.release), "--source-dir",
                        str(self.source), "--output", str(self.root / "x.json"), "--result-json", str(existing),
                        "--input-file", f"model={self.model}", "--", str(self.release / "py"), str(self.measure),
                        str(existing), env=dict(os.environ, PYTHONHOME=sys.base_prefix, PERF_PAYLOAD=str(payload)))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("already exists", proc.stderr)

    def test_run_refuses_without_ldd(self) -> None:
        if not self.require_linux_run():
            return
        tools = self.root / "tools"
        tools.mkdir()
        (tools / "git").symlink_to(shutil.which("git"))
        self.assertIsNone(shutil.which("ldd", path=str(tools)))
        proc, _ = self.run_record(benchmark_document(), env_extra={"PATH": str(tools)})
        self.assertEqual(proc.returncode, 2)
        self.assertIn("ldd", proc.stderr)

    def copy_script(self) -> Path:
        script = self.release / "script-exe"
        # Writes a valid result, so only the ldd observation can reject it.
        script.write_text(f"#!{sys.executable}\nimport os, shutil, sys\n"
                          "shutil.copyfile(os.environ['PERF_PAYLOAD'], sys.argv[2])\n")
        script.chmod(0o755)
        return script

    def test_grades(self) -> None:
        if not self.require_linux_run():
            return
        proc, output = self.run_record(benchmark_document(), "--allow-dirty")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(output.read_text())["grade"], "unbound", "--allow-dirty always yields unbound")
        (self.source / "file.txt").write_text("two\n")
        self.assertEqual(self.run_record(benchmark_document())[0].returncode, 2)
        subprocess.run(["git", "-C", str(self.source), "checkout", "-q", "--", "file.txt"], check=True)
        proc, output = self.run_record(benchmark_document(), build=self.make_build("integration", "Integration"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(output.read_text())["grade"], "unbound")
        proc, output = self.import_record(benchmark_document())
        self.assertEqual(json.loads(output.read_text())["grade"], "unbound")

    # --- comparability ---------------------------------------------------------------------
    def test_real_harness_outputs_differing_only_in_measurements_are_comparable(self) -> None:
        if not self.require_linux_run():
            return
        first, second = FIXTURES / "decode_bandwidth_run_a.json", FIXTURES / "decode_bandwidth_run_b.json"
        a, b = json.loads(first.read_text()), json.loads(second.read_text())
        self.assertEqual((a["harness"], a["parameters"]), (b["harness"], b["parameters"]))
        self.assertNotEqual(a["metrics"], b["metrics"], "fixtures must differ in measured values")
        env = {"LLAMINAR_CPU_DECODE_BW_JSON": "set-per-run"}
        _, run_a = self.run_record(first, kind="kernel-metrics", env_extra=env)
        _, run_b = self.run_record(second, kind="kernel-metrics", env_extra=env)
        report = self.diff(run_a, run_b)
        self.assertIn(report.returncode, (0, 1), report.stderr)
        self.assertEqual(json.loads(report.stdout)["waived_identity"], [])

    def test_fresh_result_destinations_do_not_split_identity(self) -> None:
        if not self.require_linux_run():
            return
        # Each run writes a different result path; the harness variable names it explicitly.
        results = []
        for _ in range(2):
            self.counter += 1
            result = self.root / f"fresh{self.counter}.json"
            payload = self.root / f"fp{self.counter}.json"
            payload.write_text(json.dumps(benchmark_document()))
            output = self.root / f"fo{self.counter}.json"
            env = dict(os.environ, PYTHONHOME=sys.base_prefix, PERF_PAYLOAD=str(payload),
                       LLAMINAR_CPU_DECODE_BW_JSON=str(result))
            proc = self.cli("run", "--kind", "llaminar-benchmark", "--build-dir", str(self.release), "--source-dir",
                            str(self.source), "--output", str(output), "--result-json", str(result), "--input-file",
                            f"model={self.model}", "--", str(self.release / "py"), str(self.measure), str(result), env=env)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            results.append(output)
            self.assertEqual(json.loads(output.read_text())["launch"]["env"]["LLAMINAR_CPU_DECODE_BW_JSON"], "<result-json>")
        self.assertEqual(self.diff(*results).returncode, 0, self.diff(*results).stderr)

    def test_diff_verdicts(self) -> None:
        if not self.require_linux_run():
            return
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

    def test_each_identity_field_refuses_until_exactly_that_field_is_waived(self) -> None:
        if not self.require_linux_run():
            return
        other_model = self.root / "model-b.gguf"
        other_model.write_bytes(b"GGUF-B")
        _, base = self.run_record(benchmark_document())
        cases = {
            "declared_inputs": self.run_record(benchmark_document(), model=other_model)[1],
            "workload": self.run_record(benchmark_document(temperature=0.7))[1],
            "output": self.run_record(benchmark_document(tokens=[9]))[1],
            "compiler": self.run_record(benchmark_document(), build=self.make_build("gcc15", "Release", "15.1.0"))[1],
            "launch_env": self.run_record(benchmark_document(), env_extra={"OMP_NUM_THREADS": "3"})[1],
            "argv": self.run_record(benchmark_document(), env_extra={})[1],
        }
        argv_doc = json.loads(cases["argv"].read_text())
        argv_doc["launch"]["argv"].append("--extra")
        cases["argv"].write_text(json.dumps(argv_doc))
        for field, candidate in cases.items():
            with self.subTest(field):
                refused = self.diff(base, candidate)
                self.assertEqual(refused.returncode, 2, field)
                self.assertIn(field, refused.stderr)
                accepted = self.diff(base, candidate, "--allow-identity-change", field)
                self.assertEqual(accepted.returncode, 0, accepted.stderr)
                self.assertEqual(json.loads(accepted.stdout)["waived_identity"], [field])

    def test_unknown_identity_never_matches(self) -> None:
        _, first = self.import_record(benchmark_document())
        _, second = self.import_record(benchmark_document())
        refused = self.diff(first, second)
        self.assertEqual(refused.returncode, 2)
        self.assertIn("unknown identity", refused.stderr)
        waivers = [arg for field in ("argv", "launch_env", "cwd", "cpu_affinity")
                   for arg in ("--allow-identity-change", field)]
        accepted = self.diff(first, second, *waivers)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertEqual(json.loads(accepted.stdout)["grade"], "unbound")

    def test_robust_spread(self) -> None:
        waivers = [arg for field in ("argv", "launch_env", "cwd", "cpu_affinity")
                   for arg in ("--allow-identity-change", field)]
        records = {name: self.import_record(kernel_document(value, samples), kind="kernel-metrics")[1]
                   for name, value, samples in [
                       ("quiet_a", 100.0, [100.0, 100.1, 99.9]), ("quiet_b", 96.0, [96.0, 96.1, 95.9]),
                       ("noisy_a", 100.0, [90.0, 100.0, 110.0]), ("noisy_b", 96.0, [86.0, 96.0, 106.0]),
                       ("stall_a", 100.0, [100.0, 100.1, 99.9, 100.0, 1.0]),
                       ("stall_b", 96.0, [96.0, 96.1, 95.9, 96.0, 1.0])]}
        self.assertEqual(self.diff(records["quiet_a"], records["quiet_b"], *waivers).returncode, 1)
        self.assertEqual(self.diff(records["noisy_a"], records["noisy_b"], *waivers).returncode, 0)
        self.assertEqual(self.diff(records["stall_a"], records["stall_b"], *waivers).returncode, 1)

    # --- validation at every entry point ---------------------------------------------------
    def test_import_rejects_invalid_inputs(self) -> None:
        bad_kernel = {
            "zero value": kernel_document(0.0), "empty samples": kernel_document(1.0, []),
            "zero sample": kernel_document(1.0, [1.0, 0.0]), "negative sample": kernel_document(1.0, [1.0, -5.0]),
            "empty metrics": {"harness": "X", "parameters": {"a": 1}, "metrics": {}},
            "no direction": {"harness": "X", "parameters": {"a": 1}, "metrics": {"m": {"value": 1.0, "unit": "u"}}},
            "no parameters": {"harness": "X", "metrics": {"m": {"value": 1.0, "unit": "u", "better": "higher"}}},
            "empty parameters": {"harness": "X", "parameters": {}, "metrics": {"m": {"value": 1.0, "unit": "u", "better": "higher"}}},
        }
        for name, document in bad_kernel.items():
            with self.subTest(name):
                self.assertEqual(self.import_record(document, kind="kernel-metrics")[0].returncode, 2)
        missing = benchmark_document()
        del missing["runtime_configuration"]
        null_config = benchmark_document()
        null_config["runtime_configuration"] = None
        for name, document in {"missing configuration": missing, "null configuration": null_config}.items():
            with self.subTest(name):
                self.assertEqual(self.import_record(document)[0].returncode, 2)
        self.assertEqual(self.import_record(benchmark_document(), with_model=False)[0].returncode, 2)
        observed = kernel_document(1.0)
        observed["observations"] = {"timings": [{"upper_median_us": 1.0}]}
        nan_observation = self.root / "nan-observation.json"
        nan_observation.write_text(json.dumps(observed).replace('"upper_median_us": 1.0', '"upper_median_us": NaN'))
        proc = self.import_record(nan_observation, kind="kernel-metrics")[0]
        self.assertEqual(proc.returncode, 2)
        self.assertIn("NaN", proc.stderr)
        nan_text = self.root / "nan.json"
        nan_text.write_text(json.dumps(benchmark_document()).replace('"decode": 20.0', '"decode": NaN', 1))
        self.assertEqual(self.import_record(nan_text)[0].returncode, 2)

    def test_diff_cli_rejects_hostile_documents_in_both_orders(self) -> None:
        _, good_kernel = self.import_record(kernel_document(10.0, [10.0, 10.1]), kind="kernel-metrics")
        _, good_bench = self.import_record(benchmark_document())
        kernel, bench = json.loads(good_kernel.read_text()), json.loads(good_bench.read_text())

        def hostile(base: dict, mutator, raw_text: str | None = None) -> Path:
            self.counter += 1
            path = self.root / f"hostile{self.counter}.json"
            document = copy.deepcopy(base)
            mutator(document)
            path.write_text(raw_text if raw_text is not None else json.dumps(document))
            return path

        def set_path(keys, value):
            def mutate(d):
                for key in keys[:-1]:
                    d = d[key]
                d[keys[-1]] = value
            return mutate

        text = json.dumps(kernel)
        cases = {
            "NaN literal": (good_kernel, hostile(kernel, lambda d: None, text.replace('"value": 10.0', '"value": NaN', 1))),
            "Infinity sample": (good_kernel, hostile(kernel, lambda d: None, text.replace("10.1", "Infinity", 1))),
            "zero sample": (good_kernel, hostile(kernel, set_path(["metrics", "gbps", "samples"], [10.0, 0.0]))),
            "negative sample": (good_kernel, hostile(kernel, set_path(["metrics", "gbps", "samples"], [10.0, -50.0]))),
            "empty metrics": (good_kernel, hostile(kernel, set_path(["metrics"], {}))),
            "zero value": (good_kernel, hostile(kernel, set_path(["metrics", "gbps", "value"], 0.0))),
            "bad direction": (good_kernel, hostile(kernel, set_path(["metrics", "gbps", "better"], "up"))),
            "empty build": (good_kernel, hostile(kernel, set_path(["build"], {}))),
            "empty host": (good_kernel, hostile(kernel, set_path(["host"], {}))),
            "null cpu isa": (good_kernel, hostile(kernel, set_path(["build", "cpu_isa"], None))),
            "missing workload": (good_kernel, hostile(kernel, lambda d: d.pop("workload"))),
            "bound with empty launch": (good_kernel, hostile(kernel, lambda d: d.update(grade="bound", launch={}))),
            "bound without launch": (good_kernel, hostile(kernel, lambda d: d.update(grade="bound"))),
            "null runtime configuration": (good_bench, hostile(bench, set_path(["workload", "runtime_configuration"], None))),
            "missing model input": (good_bench, hostile(bench, set_path(["workload", "declared_inputs"], {}))),
            "wrong schema": (good_kernel, hostile(kernel, set_path(["schema"], "llaminar.perf-baseline.v2"))),
        }
        # Imports have no launch record; waive exactly those unknown fields so that each
        # hostile document is rejected by validation itself, not by the identity refusal.
        launch_waivers = [arg for field in ("argv", "launch_env", "cwd", "cpu_affinity")
                          for arg in ("--allow-identity-change", field)]
        for good in (good_kernel, good_bench):
            self.assertEqual(self.diff(good, good, *launch_waivers).returncode, 0, "valid pair must compare")
        for name, (good, path) in cases.items():
            with self.subTest(name):
                for order in ((good, path), (path, good)):
                    self.assertEqual(self.diff(*order, *launch_waivers).returncode, 2, name)
        for threshold in ("inf", "nan", "-1"):
            with self.subTest(threshold=threshold):
                self.assertEqual(self.diff(good_kernel, good_kernel, *launch_waivers, "--threshold-pct=" + threshold).returncode, 2)


if __name__ == "__main__":
    unittest.main()
