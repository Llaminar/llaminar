#!/usr/bin/env python3
"""Device-free regressions for typed E2E discovery and full-check admission."""
import argparse
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
sys.path.insert(0, str(ROOT / "tests/v2/e2e/server"))
import run_model_parity_e2e as e2e
import long_context_checks as long_context


def inventory(records):
    """Produce Google's JSON discovery shape, including full parameter values."""
    return json.dumps({"testsuites": [{"name": "Suite", "testsuite": [
        {"name": name, "value_param": json.dumps(record)} for name, record in records]}]})


class E2EDiscoveryTests(unittest.TestCase):
    def test_cpu_node_tp_uses_cpu_evidence_without_adding_device_placement(self):
        self.assertEqual(e2e.harness_backend("CPU"), "cpu:0")
        for signature in ("CUDA", "ROCm", "CPU+CUDA", "CPU+ROCm", "CUDA+ROCm"):
            self.assertEqual(e2e.harness_backend(signature), "tp")
        shell = (ROOT / "tests/v2/e2e/server/test_server_e2e.sh").read_text()
        self.assertIn('if [[ -z "$SERVER_ARGS_FILE" && "$backend" != "tp" && "$backend" != "pp" ]]', shell)

    def test_exact_cell_timeout_retires_the_whole_server_group(self):
        process = MagicMock()
        process.__enter__.return_value = process
        process.wait.side_effect = e2e.subprocess.TimeoutExpired("harness", 600)
        with patch.object(e2e.subprocess, "Popen", return_value=process) as launch, \
             patch.object(e2e.parity, "_terminate_process_group") as retire:
            self.assertEqual(e2e.run_server_harness(["harness"], {}, None), 124)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        process.wait.assert_called_once_with(timeout=600)
        retire.assert_called_once_with(process)

    def test_successful_cell_does_not_retire_an_already_completed_server(self):
        process = MagicMock()
        process.__enter__.return_value = process
        process.wait.return_value = 0
        with patch.object(e2e.subprocess, "Popen", return_value=process), \
             patch.object(e2e.parity, "_terminate_process_group") as retire:
            self.assertEqual(e2e.run_server_harness(["harness"], {}, None), 0)
        retire.assert_not_called()

    def test_full_parameter_survives_large_domain_arguments_and_escaping(self):
        record = {"model_parity_schema": 1, "id": "cell", "model": "/models/a model.gguf",
                  "e2e": {"server_args": ["--define-domain", "name=" + "x" * 1000, "a\n\"b"]}}
        actual = e2e.parse_parameters(inventory([("ProductionParity/cell", record)]))
        self.assertEqual(actual["Suite.ProductionParity/cell"], record)

    def test_untagged_cells_and_focused_diagnostics_do_not_become_certificates(self):
        record = {"model_parity_schema": 1, "e2e": None}
        actual = e2e.parse_parameters(inventory([
            ("ProductionParity/off", record), ("FocusedDiagnostic", {"ignored": True})]))
        self.assertEqual(len(actual), 1)
        self.assertIsNone(actual["Suite.ProductionParity/off"]["e2e"])

    def test_duplicate_stale_or_truncated_metadata_fails_closed(self):
        record = {"model_parity_schema": 1, "e2e": None}
        with self.assertRaises(ValueError):
            e2e.parse_parameters(inventory([("ProductionParity/a", record)] * 2))
        with self.assertRaises(ValueError):
            e2e.parse_parameters(inventory([("ProductionParity/a", {"model_parity_schema": 0})]))
        with self.assertRaises(ValueError):
            e2e.parse_parameters('{"testsuites":')

    def test_inherited_lite_or_small_context_cannot_weaken_profile(self):
        profile = {"context_length": 8192, "minimum_prompt_tokens": 4096,
                   "generation_tokens": 2048, "request_timeout_seconds": 600,
                   "readiness_timeout_seconds": 180, "thinking_modes": "both", "movement_evidence": "required"}
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
                   "generation_tokens": 2048, "request_timeout_seconds": 600, "thinking_modes": "both", "movement_evidence": "not_applicable"}
        for seconds in (60, 180, 600):
            env = e2e.certification_environment(profile | {"readiness_timeout_seconds": seconds}, Path("/tmp/e2e"))
            self.assertEqual(env["LLAMINAR_E2E_STARTUP_TIMEOUT_SECONDS"], str(seconds))
            self.assertEqual(env["LLAMINAR_E2E_LONG_REQUEST_TIMEOUT"], "600")
        for seconds in (None, 0, -1, 601, True, "180"):
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
