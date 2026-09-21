#!/usr/bin/env python3
"""Device-free regressions for typed E2E discovery and full-check admission."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
sys.path.insert(0, str(ROOT / "tests/v2/e2e/server"))
import run_model_parity_e2e as e2e
import model_parity_inventory as cell_inventory
import long_context_checks as long_context


class LongContextProgressTests(unittest.TestCase):
    """A killed HTTP check must preserve progress without manufacturing a pass."""

    PROFILE = {"tier": "full", "context_length": 8192,
               "minimum_prompt_tokens": 4096, "generation_tokens": 2048}

    def test_active_check_survives_cancellation_with_previous_results(self):
        """The START record must reach disk before a cancellable HTTP call."""
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "long_context_results.json"
            runner = long_context.CheckRunner("probe", artifact, self.PROFILE)
            for index in range(4):
                runner.run(f"recall-{index}", lambda: "correct")

            def cancelled_request():
                snapshot = json.loads(artifact.read_text())
                self.assertFalse(snapshot["complete"])
                self.assertEqual(len(snapshot["results"]), 4)
                self.assertEqual(snapshot["active_check"]["name"], "structured generation")
                self.assertGreater(snapshot["active_check"]["started_unix_seconds"], 0)
                raise KeyboardInterrupt("watchdog")

            with self.assertRaises(KeyboardInterrupt):
                runner.run("structured generation", cancelled_request)
            with self.assertRaisesRegex(ValueError, "missing or failed"):
                e2e.validate_long_context_evidence(Path(directory), self.PROFILE)
            self.assertEqual(len(json.loads(artifact.read_text())["results"]), 4)

    def test_completion_requires_all_eight_checks_and_normal_finish(self):
        """Neither eight interim records nor a short finished run certify."""
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "long_context_results.json"
            runner = long_context.CheckRunner("probe", artifact, self.PROFILE)
            runner.run("first", lambda: "correct")
            runner.finish()
            self.assertFalse(json.loads(artifact.read_text())["complete"])
            for index in range(7):
                runner.run(f"remaining-{index}", lambda: "correct")
            self.assertFalse(json.loads(artifact.read_text())["complete"])
            runner.finish()
            e2e.validate_long_context_evidence(Path(directory), self.PROFILE)
            self.assertIsNone(json.loads(artifact.read_text())["active_check"])
            snapshot = json.loads(artifact.read_text())
            snapshot["active_check"] = {"name": "unfinished"}
            artifact.write_text(json.dumps(snapshot))
            with self.assertRaisesRegex(ValueError, "missing or failed"):
                e2e.validate_long_context_evidence(Path(directory), self.PROFILE)

    def test_failed_checks_remain_failed_while_later_checks_continue(self):
        """Completeness and correctness are independent certificate obligations."""
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "long_context_results.json"
            runner = long_context.CheckRunner("probe", artifact, self.PROFILE)
            for index, error in enumerate((long_context.CheckError("bad recall"), ValueError("bad JSON"))):
                def fail(error=error):
                    raise error
                runner.run(f"failure-{index}", fail)
            for index in range(6):
                runner.run(f"success-{index}", lambda: "correct")
            runner.finish()
            self.assertEqual(len(runner.failures), 2)
            self.assertTrue(json.loads(artifact.read_text())["complete"])
            with self.assertRaisesRegex(ValueError, "missing or failed"):
                e2e.validate_long_context_evidence(Path(directory), self.PROFILE)

    def test_check_duration_is_monotonic_request_wall_time(self):
        """Artifact mtimes are not timing evidence; measure around the call."""
        runner = long_context.CheckRunner("probe")
        with patch.object(long_context.time, "monotonic", side_effect=(10.0, 12.75)):
            runner.run("timed", lambda: "correct")
        self.assertEqual(runner.results[0]["elapsed_seconds"], 2.75)

    def test_failed_atomic_publication_preserves_previous_json_and_cleans_temp(self):
        """Publication failure is fatal and cannot erase the last complete JSON."""
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "long_context_results.json"
            runner = long_context.CheckRunner("probe", artifact, self.PROFILE)
            original = artifact.read_bytes()
            with patch.object(long_context.os, "replace", side_effect=OSError("disk failure")):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    runner.run("not entered", lambda: self.fail("request ran before publication"))
            self.assertEqual(artifact.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [artifact])


def inventory(records):
    """Produce Google's JSON discovery shape, including full parameter values."""
    return json.dumps({"testsuites": [{"name": "Suite", "testsuite": [
        {"name": name, "value_param": json.dumps(record)} for name, record in records]}]})


class E2EDiscoveryTests(unittest.TestCase):
    def test_cell_timeout_is_explicit_positive_metadata_not_name_inference(self):
        """Only the typed profile grants an extended allowance."""
        profile = {"cell_timeout_seconds": {"AVX512": 900, "AVX2": 1200}}
        self.assertEqual(e2e.cell_timeout_seconds(profile, e2e.CPUISA.AVX512), 900)
        self.assertEqual(e2e.cell_timeout_seconds(profile, e2e.CPUISA.AVX2), 1200)
        for budgets in (None, 900, {}, {"AVX2": 1200},
                        {"AVX512": 900, "AVX2": 1200, "SSE2": 1800}):
            with self.subTest(budgets=budgets), self.assertRaises(ValueError):
                e2e.cell_timeout_seconds({"cell_timeout_seconds": budgets}, e2e.CPUISA.AVX2)
        for isa in e2e.CPUISA:
            for seconds in (None, 0, -1, True, "1200", 1200.5):
                invalid = {**profile["cell_timeout_seconds"], isa.value: seconds}
                with self.subTest(isa=isa, seconds=seconds), self.assertRaises(ValueError):
                    e2e.cell_timeout_seconds({"cell_timeout_seconds": invalid}, e2e.CPUISA.AVX2)
        with self.assertRaises(ValueError):
            e2e.cell_timeout_seconds({"id": "AVX2_CPU_only"}, e2e.CPUISA.AVX2)
        with self.assertRaises(ValueError):
            e2e.cell_timeout_seconds(profile, "AVX2")

    def test_runtime_isa_comes_from_the_tested_build_or_image(self):
        """One inventory companion cannot impose its own ISA on either runtime."""
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "llaminar2"
            cache = binary.parent / "CMakeCache.txt"
            for isa in e2e.CPUISA:
                cache.write_text(f"CMAKE_BUILD_TYPE:STRING=Release\nLLAMINAR_CPU_ISA:STRING={isa.value}\n")
                self.assertEqual(e2e.release_binary_cpu_isa(binary), isa)
                cache.write_text(f"CMAKE_BUILD_TYPE:STRING=Release\nLLAMINAR_CPU_ISA:STRING={isa.value.lower()}\n")
                self.assertEqual(e2e.release_binary_cpu_isa(binary), isa)
                identity = {"labels": {"org.llaminar.cpu_isa": isa.value}}
                self.assertEqual(e2e.image_cpu_isa(identity), isa)
            for contents in ("", "CMAKE_BUILD_TYPE:STRING=Integration\nLLAMINAR_CPU_ISA:STRING=AVX2\n",
                             "CMAKE_BUILD_TYPE:STRING=Release\n",
                             "CMAKE_BUILD_TYPE:STRING=Release\nLLAMINAR_CPU_ISA:STRING=native\n"):
                cache.write_text(contents)
                with self.assertRaises(ValueError):
                    e2e.release_binary_cpu_isa(binary)
        for identity in ({}, {"labels": None}, {"labels": {}},
                         {"labels": {"org.llaminar.cpu_isa": "SSE2"}}):
            with self.assertRaises(ValueError):
                e2e.image_cpu_isa(identity)

    def test_extended_cell_still_has_one_process_group_deadline(self):
        """Plan, process creation and HTTP all spend the approved twenty minutes."""
        process = MagicMock()
        process.__enter__.return_value = process
        process.wait.side_effect = [0, e2e.subprocess.TimeoutExpired("http", 1155)]
        with patch.object(e2e.time, "monotonic", side_effect=[100, 100, 100, 140, 145]), \
             patch.object(e2e.subprocess, "Popen", return_value=process), \
             patch.object(e2e.parity, "_terminate_process_group") as retire:
            budget = e2e.E2ECellBudget(1200)
            self.assertEqual(e2e.run_e2e_process(["plan"], {}, None, budget=budget), 0)
            self.assertEqual(e2e.run_e2e_process(["http"], {}, None, budget=budget), 124)
        self.assertEqual([call.kwargs["timeout"] for call in process.wait.call_args_list], [1200, 1155])
        retire.assert_called_once_with(process)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                e2e.E2ECellBudget(invalid)

    def test_cpu_node_tp_uses_cpu_evidence_without_adding_device_placement(self):
        self.assertEqual(e2e.harness_backend("CPU"), "cpu:0")
        for signature in ("CUDA", "ROCm", "CPU+CUDA", "CPU+ROCm", "CUDA+ROCm"):
            self.assertEqual(e2e.harness_backend(signature), "tp")
        shell = (ROOT / "tests/v2/e2e/server/test_server_e2e.sh").read_text()
        self.assertIn('if [[ -z "$SERVER_ARGS_FILE" && "$backend" != "tp" && "$backend" != "pp" ]]', shell)

    def test_exact_cell_timeout_retires_the_whole_server_group(self):
        process = MagicMock()
        process.__enter__.return_value = process
        process.wait.side_effect = e2e.subprocess.TimeoutExpired("harness", 900)
        with patch.object(e2e.subprocess, "Popen", return_value=process) as launch, \
             patch.object(e2e.parity, "_terminate_process_group") as retire, \
             patch.object(e2e.time, "monotonic", return_value=100):
            self.assertEqual(e2e.run_e2e_process(["harness"], {}, None), 124)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        process.wait.assert_called_once_with(timeout=900)
        retire.assert_called_once_with(process)

    def test_successful_cell_does_not_retire_an_already_completed_server(self):
        process = MagicMock()
        process.__enter__.return_value = process
        process.wait.return_value = 0
        with patch.object(e2e.subprocess, "Popen", return_value=process), \
             patch.object(e2e.parity, "_terminate_process_group") as retire:
            self.assertEqual(e2e.run_e2e_process(["harness"], {}, None), 0)
        retire.assert_not_called()

    def test_plan_and_http_share_one_immutable_cell_deadline(self):
        """Separate readiness does not grant a second fifteen-minute cell budget."""
        process = MagicMock()
        process.__enter__.return_value = process
        process.wait.return_value = 0
        with patch.object(e2e.time, "monotonic", side_effect=[100, 100, 100, 140, 145]), \
             patch.object(e2e.subprocess, "Popen", return_value=process):
            budget = e2e.E2ECellBudget()
            self.assertEqual(e2e.run_e2e_process(["plan"], {}, None, budget=budget), 0)
            self.assertEqual(e2e.run_e2e_process(["http"], {}, None, budget=budget), 0)
        self.assertEqual([call.kwargs["timeout"] for call in process.wait.call_args_list], [900, 855])

    def test_expired_cell_never_launches_another_phase(self):
        with patch.object(e2e.time, "monotonic", side_effect=[100, 1000]), \
             patch.object(e2e.subprocess, "Popen") as launch:
            budget = e2e.E2ECellBudget()
            self.assertEqual(e2e.run_e2e_process(["http"], {}, None, budget=budget), 124)
        launch.assert_not_called()

    def test_process_creation_cannot_extend_the_cell_deadline(self):
        process = MagicMock()
        process.__enter__.return_value = process
        with patch.object(e2e.time, "monotonic", side_effect=[100, 999, 1000]), \
             patch.object(e2e.subprocess, "Popen", return_value=process), \
             patch.object(e2e.parity, "_terminate_process_group") as retire:
            budget = e2e.E2ECellBudget()
            self.assertEqual(e2e.run_e2e_process(["http"], {}, None, budget=budget), 124)
        retire.assert_called_once_with(process)
        process.wait.assert_not_called()

    def test_full_parameter_survives_large_domain_arguments_and_escaping(self):
        record = {"model_parity_schema": 1, "id": "cell", "model": "/models/a model.gguf",
                  "e2e": {"server_args": ["--define-domain", "name=" + "x" * 1000, "a\n\"b"]}}
        actual = cell_inventory.parse_parameters(inventory([("ProductionParity/cell", record)]))
        self.assertEqual(actual["Suite.ProductionParity/cell"], record)

    def test_untagged_cells_and_focused_diagnostics_do_not_become_certificates(self):
        record = {"model_parity_schema": 1, "e2e": None}
        actual = cell_inventory.parse_parameters(inventory([
            ("ProductionParity/off", record), ("FocusedDiagnostic", {"ignored": True})]))
        self.assertEqual(len(actual), 1)
        self.assertIsNone(actual["Suite.ProductionParity/off"]["e2e"])

    def test_duplicate_stale_or_truncated_metadata_fails_closed(self):
        record = {"model_parity_schema": 1, "e2e": None}
        with self.assertRaises(ValueError):
            cell_inventory.parse_parameters(inventory([("ProductionParity/a", record)] * 2))
        with self.assertRaises(ValueError):
            cell_inventory.parse_parameters(inventory([("ProductionParity/a", {"model_parity_schema": 0})]))
        with self.assertRaises(ValueError):
            cell_inventory.parse_parameters('{"testsuites":')

    def test_inherited_lite_or_small_context_cannot_weaken_profile(self):
        profile = {"context_length": 8192, "minimum_prompt_tokens": 4096,
                   "generation_tokens": 2048, "request_timeout_seconds": 600,
                   "readiness_timeout_seconds": 180, "cell_timeout_seconds": {"AVX512": 900, "AVX2": 900},
                   "thinking_modes": "both", "movement_evidence": "required"}
        with patch.dict(e2e.os.environ, {"LLAMINAR_E2E_LONG_CONTEXT_TIER": "lite",
                                        "LLAMINAR_E2E_CONTEXT_LENGTH": "128",
                                        "LLAMINAR_E2E_STARTUP_TIMEOUT_SECONDS": "9999",
                                        "LLAMINAR_E2E_THINKING_MODES": "non-thinking",
                                        "LLAMINAR_E2E_MOVEMENT_EVIDENCE": "not_applicable"}):
            env = e2e.certification_environment(profile, Path("/tmp/e2e"))
        self.assertEqual(env["LLAMINAR_E2E_LONG_CONTEXT_TIER"], "full")
        self.assertEqual(env["LLAMINAR_E2E_CONTEXT_LENGTH"], "8192")
        self.assertEqual(env["LLAMINAR_E2E_LONG_MAX_TOKENS"], "2048")
        self.assertEqual(env["LLAMINAR_E2E_PERF_STATS"], "1")
        self.assertEqual(env["LLAMINAR_E2E_STARTUP_TIMEOUT_SECONDS"], "180")
        self.assertEqual(env["LLAMINAR_E2E_THINKING_MODES"], "both")
        self.assertEqual(env["LLAMINAR_E2E_MOVEMENT_EVIDENCE"], "required")

    def test_readiness_is_per_cell_and_cannot_extend_the_exact_cell_watchdog(self):
        profile = {"context_length": 8192, "minimum_prompt_tokens": 4096,
                   "generation_tokens": 2048, "request_timeout_seconds": 600,
                   "cell_timeout_seconds": {"AVX512": 900, "AVX2": 1200},
                   "thinking_modes": "both", "movement_evidence": "not_applicable"}
        for seconds in (60, 180, 900):
            env = e2e.certification_environment(profile | {"readiness_timeout_seconds": seconds}, Path("/tmp/e2e"))
            self.assertEqual(env["LLAMINAR_E2E_STARTUP_TIMEOUT_SECONDS"], str(seconds))
            self.assertEqual(env["LLAMINAR_E2E_LONG_REQUEST_TIMEOUT"], "600")
        for seconds in (None, 0, -1, 901, True, "180"):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                e2e.certification_environment(profile | {"readiness_timeout_seconds": seconds}, Path("/tmp/e2e"))

    def test_movement_obligation_rejects_missing_or_inferred_values(self):
        for evidence in ("not_applicable", "forbidden", "required"):
            self.assertEqual(e2e.movement_evidence({"movement_evidence": evidence}), evidence)
        for evidence in (None, True, "dynamic", "auto", ""):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                e2e.movement_evidence({"movement_evidence": evidence})

    def test_thinking_coverage_is_explicit_and_never_model_name_derived(self):
        for mode in ("both", "non-thinking"):
            self.assertEqual(e2e.thinking_modes({"thinking_modes": mode}), mode)
        for mode in (None, True, "auto", "ornith", ""):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                e2e.thinking_modes({"thinking_modes": mode})
        shell = (ROOT / "tests/v2/e2e/server/test_server_e2e.sh").read_text()
        self.assertNotIn("is_thinking_model", shell)
        self.assertIn('if [ "$THINKING_MODES" = "both" ]', shell)

    def test_full_helper_keeps_all_eight_objective_scenarios(self):
        observed = []
        def record(_runner, name, _check):
            observed.append(name)
        with patch.object(long_context.CheckRunner, "run", record):
            result = long_context.main(["--base-url", "http://unused", "--tier", "full"])
        self.assertEqual(result, 0)
        self.assertEqual(len(observed), 8)
        self.assertEqual(observed[:3], ["long needle recall beginning", "long needle recall middle", "long needle recall end"])
        self.assertIn("multi-needle strict JSON recall", observed)
        self.assertIn("structured long generation", observed)
        self.assertIn("cache-reset probe", observed)
        self.assertIn("valid near-boundary context", observed)
        self.assertIn("oversized context rejection", observed)

    def test_multi_needle_uses_the_same_filler_geometry_as_the_record_budget(self):
        with patch.object(long_context, "make_audit_record", wraps=long_context.make_audit_record) as filler:
            messages, sentinels, count = long_context.build_multi_needle_prompt(4096, 8192, 128, "full")
        self.assertEqual(filler.call_count, count - len(sentinels))
        prompt = messages[-1]["content"]
        self.assertEqual(prompt.count("status normal; this is not the requested value."), count - 3)
        for key, value in sentinels.items():
            self.assertIn(f"REQUIRED_JSON_FIELD {key} has exact value {value}.", prompt)


if __name__ == "__main__":
    unittest.main()
