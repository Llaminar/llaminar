#!/usr/bin/env python3
"""Device-free tests for finite exact collection and generic-workflow retirement.

Command planning never discovers devices, starts compilers or launches kernels.
Mock native lanes test ownership/resume plumbing only; arithmetic correctness,
actual profiling, and source-bound Release qualification remain native gates.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts/refresh_native_quantised_dispatch_tables.sh"
RETIRED = ROOT / "scripts/train_native_quantised_dispatch.sh"
DRIVER = ROOT / "scripts/native_quantised/exact_refresh.py"
spec = importlib.util.spec_from_file_location("exact_refresh_test_driver", DRIVER)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class NativeQuantisedDispatchRefreshTest(unittest.TestCase):
    """Prove the public CLI routes only finite exact-overlay transactions."""

    def invoke(self, *arguments):
        """Run only the public argument planner, never a native fixture."""
        return subprocess.run(["bash", str(SCRIPT), *map(str, arguments)],
                              capture_output=True, text=True, cwd=ROOT)

    def test_cpu_new_choice_is_explicit_and_forwarded_only_in_expert_scope(self):
        args=driver.parser().parse_args(['--backend','cpu','--stage','expert-only',
            '--output-dir','/unused','--expert-model-family','qwen4exp','--expert-tp-degree','2',
            '--expert-layout','intermediate_shard','--expert-isa','AVX512:AVX512',
            '--expert-purpose','prefill','--expert-ordered-three'])
        self.assertIn('--expert-ordered-three',driver.cpu_command(args))
        args.expert_ordered_three=False
        self.assertNotIn('--expert-ordered-three',driver.cpu_command(args))

    def test_retired_training_command_has_no_executable_generic_path(self):
        result = subprocess.run(["bash", str(RETIRED), "--backend", "cpu", "--install"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("training/refitting is retired", result.stderr)
        self.assertNotIn("cmake --build", result.stdout)
        help_result = subprocess.run(["bash", str(RETIRED), "--help"],
                                     capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("exact overlays", help_result.stdout)

    def test_dry_run_policy_discovery_is_fixture_owned(self):
        """The exact planner has no host-discovery or native side effects."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not-created"
            with patch.object(driver.subprocess, "run", side_effect=AssertionError("native discovery")), \
                 patch.object(driver, "trainer_provenance", side_effect=AssertionError("binary discovery")), \
                 patch.object(driver, "gpu_decode", side_effect=AssertionError("GPU launch")), \
                 patch("sys.stdout") as stdout:
                self.assertEqual(driver.main(["--backend", "all", "--output-dir", str(output), "--dry-run"]), 0)
            self.assertFalse(output.exists())
            self.assertTrue(stdout.write.called)

    def test_cpu_exact_plan_preserves_all_phase_and_isa_owners(self):
        result = self.invoke("--backend", "cpu", "--surface", "decode", "--output-dir", "/unused", "--dry-run", "--model-scope", "all-retained")
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertFalse(plan["generic_fit"])
        self.assertEqual(plan["production_shapes"], {"cpu": 131})
        self.assertEqual(plan["formats"], 21)
        command, = plan["commands"]
        self.assertIn("native_quantised_dispatch.cpu_exact_sweep", command)
        self.assertIn("--plan-only", command)
        self.assertIn("--stage", command)
        self.assertIn("all", command)
        self.assertIn("--avx2-bin", command)
        self.assertIn("--avx512-bin", command)
        self.assertIn("--fused-avx2-bin", command)
        self.assertIn("--fused-avx512-bin", command)
        self.assertFalse(any("generic" in token or "fit-cache" in token for token in command))

    def test_retired_generic_flags_fail_before_discovery(self):
        for flag in ("--freeze-generic", "--certify-generic", "--generic-max-leaves",
                     "--maximum-p95-regret-percent", "--minimum-passing-domain-percent",
                     "--corpus-root", "--policy-fitter"):
            with self.subTest(flag=flag):
                result = self.invoke("--backend", "cpu", "--output-dir", "/unused", flag, "1", "--dry-run")
                self.assertEqual(result.returncode, 2)
                self.assertIn("generic fitting/refitting controls are retired", result.stderr)

    def test_private_shape_profiles_and_unknown_flags_fail(self):
        for arguments in (("--profile", "quick"), ("--shapes", "QFN-local"),
                          ("--shape-partition", "fast-development"), ("--m-values", "4")):
            with self.subTest(arguments=arguments):
                result = self.invoke("--backend", "cpu", "--output-dir", "/unused", "--dry-run", *arguments)
                self.assertEqual(result.returncode, 2)

    def test_exact_plan_routes_existing_gpu_prefill_modules(self):
        result = self.invoke("--backend", "rocm", "--output-dir", "/unused", "--dry-run", "--model-scope", "all-retained")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = json.loads(result.stdout)["commands"]
        self.assertEqual(commands[0], ["GPU_EXACT_TRANSACTION", "rocm", "M1", "owning_rebuild", "grouped_M2..16,31"])
        native = [command for command in commands if "run" in command]
        for name, command in zip(("dense", "moe"), native):
            self.assertIn(f"native_quantised_dispatch.production_{name}_prefill_sweep", command)
            self.assertIn("--backend", command)
            self.assertIn("rocm", command)
            self.assertNotIn("--shape", command)
            self.assertNotIn("--format", command)
            self.assertNotIn("--m", command)
        emitters = [command for command in commands if "--retain-base-include" in command]
        self.assertEqual(len(emitters), 2)
        self.assertTrue(all("--retention-receipt" in command for command in emitters))
        self.assertEqual(len([command for command in commands if "combine" in command]), 2)

    def test_missing_later_backend_arguments_fail_before_any_launch(self):
        with patch.object(driver, "run", side_effect=AssertionError("early CPU launch")):
            with self.assertRaises(SystemExit) as caught:
                driver.main(["--backend", "all", "--output-dir", "/unused", "--install"])
        self.assertEqual(caught.exception.code, 2)

    def test_exact_cpu_execution_routes_without_learner(self):
        with patch.object(driver, "run") as execute:
            self.assertEqual(driver.main(["--backend", "cpu", "--surface", "decode",
                "--output-dir", "/unused", "--threads", "28", "--install", "--resume", "--sudo-tool"]), 0)
        command = execute.call_args.args[0]
        self.assertIn("native_quantised_dispatch.cpu_exact_sweep", command)
        self.assertIn("--install", command)
        self.assertIn("--resume", command)
        self.assertIn("--sudo-tool", command)
        self.assertNotIn("--freeze-generic", command)

    def test_sudo_is_forwarded_only_to_isolated_profiler_owner(self):
        args = driver.parser().parse_args(["--backend", "rocm", "--output-dir", "/unused",
            "--rocm-profiler", "/selected/rocprofv3", "--sudo-tool"])
        with patch.object(driver, "run") as execute:
            driver.profile(args, "rocm", Path("/native"), Path("/evidence"), Path("/common.csv"))
        commands = [call.args[0] for call in execute.call_args_list]
        self.assertNotIn("--sudo-tool", commands[0])
        self.assertIn("native_quantised_dispatch.profiler_collectors", commands[1])
        self.assertIn("--sudo-tool", commands[1])
        self.assertNotIn("--sudo-tool", commands[2])

    def test_dense_cpu_prefill_is_an_explicit_gap(self):
        with patch.object(driver, "run", side_effect=AssertionError("fallback")):
            with self.assertRaises(SystemExit):
                driver.main(["--backend", "cpu", "--surface", "dense-prefill", "--output-dir", "/unused"])

    def test_positive_timing_and_thread_controls_are_required(self):
        for flag in ("--threads", "--warmups", "--samples", "--repeats"):
            with self.subTest(flag=flag):
                result = self.invoke("--backend", "cpu", "--output-dir", "/unused", "--dry-run", flag, "0")
                self.assertEqual(result.returncode, 2)
                self.assertIn("must be positive", result.stderr)

    def test_native_lanes_cover_only_disjoint_production_shapes_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = driver.parser().parse_args(["--backend", "rocm", "--output-dir", temporary, "--model-scope", "all-retained"])
            names = driver.load_shape_manifest().names(production_only=True)
            observed = []
            prefix = "LLAMINAR_ROCM_NATIVE_QUANTISED_DECODE_"
            def native(command, *, env, log):
                owned = tuple(env[prefix + "SHAPES"].split(","))
                observed.append(owned)
                self.assertEqual(len(env[prefix + "FORMATS"].split(",")), 21)
                self.assertEqual(env[prefix + "M"], "1")
                self.assertIn("TrainerCsv_CodebookTagged", command[1])
                for field in ("CSV", "TIMING_CSV"):
                    with Path(env[prefix + field]).open("w", newline="") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(("shape", "evidence"))
                        writer.writerows((name, "fixture") for name in owned)
            with patch.object(driver, "run", side_effect=native):
                aggregate, timing = driver.collect(args, "rocm", Path("/not-a-native-binary"), directory,
                    "1", names, (0, 1), {"build_id": "fixture"})
            self.assertEqual(set(observed[0]) & set(observed[1]), set())
            self.assertEqual(set(observed[0]) | set(observed[1]), set(names))
            with aggregate.open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), len(names))
            args.resume = True
            with patch.object(driver, "run", side_effect=AssertionError("repeat paid native work")):
                self.assertEqual(driver.collect(args, "rocm", Path("/unused"), directory,
                    "1", names, (0, 1), {"build_id": "fixture"}), (aggregate, timing))
                with self.assertRaisesRegex(ValueError, "identical source/build contract"):
                    driver.collect(args, "rocm", Path("/unused"), directory,
                        "1", names, (0, 1), {"build_id": "changed"})

    def test_csv_lane_schema_mismatch_never_publishes_complete_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second, output = (root / name for name in ("one.csv", "two.csv", "all.csv"))
            first.write_text("a,b\n1,2\n")
            second.write_text("b,c\n2,3\n")
            with self.assertRaisesRegex(ValueError, "CSV schema"):
                driver.merge_csv(output, [first, second])
            self.assertFalse(output.exists())

    def test_exact_install_cannot_rebuild_a_foreign_source_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "CMakeCache.txt"
            cache.write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/foreign/source/src/v2\n")
            with patch.object(driver, "_find_cmake_cache", return_value=cache):
                with self.assertRaisesRegex(ValueError, "owning build of this source"):
                    driver.owning_build_root(Path("/snapshot/native-trainer"))
            cache.write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={ROOT}/src/v2\n")
            with patch.object(driver, "_find_cmake_cache", return_value=cache):
                self.assertEqual(driver.owning_build_root(Path("/owning/native-trainer")), cache.parent)


if __name__ == "__main__":
    unittest.main()
