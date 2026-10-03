#!/usr/bin/env python3
"""Device-free proofs of fail-first HTTP replay, ordered evidence and bounded diagnostics.

No model, driver or accelerator is initialized. A small local HTTP peer proves
the actual curl client's deadline/partial-SSE behavior; injected exchanges prove
the journal state machine without timing-dependent native inference.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests/v2/e2e/server"))
sys.path.insert(0, str(ROOT / "scripts/ci"))
import http_lifecycle_hammer as hammer
import run_http_lifecycle_hammer as driver


def sequence() -> dict:
    """Return distinct reset/restore requests with independent arithmetic answers."""
    return {"schema": 1, "requests": [{"name": name,
        "body": {"messages": [{"role": "user", "content": prompt}],
                 "temperature": 0, "max_tokens": 200, "stream": stream},
        "expected_numeric_answer": answer}
        for name, prompt, stream, answer in (("fresh", "What is 2+2?", False, "4"),
            ("changed", "What is 3+5?", False, "8"),
            ("restore_stream", "What is 2+2?", True, "4"))]}


def reply(answer: str, stream: bool = False) -> str:
    """Create a terminal response, including all mandatory SSE envelope fields."""
    if not stream:
        return json.dumps({"choices": [{"message": {"content": answer}, "finish_reason": "stop"}]})
    base = {"id": "chatcmpl-probe", "object": "chat.completion.chunk", "system_fingerprint": "llaminar-v2"}
    choices = ({"delta": {"role": "assistant"}, "finish_reason": None},
               {"delta": {"content": answer}, "finish_reason": None},
               {"delta": {}, "finish_reason": "stop"})
    return "".join("data: " + json.dumps({**base, "choices": [choice]}) + "\n\n"
                   for choice in choices) + "data: [DONE]\n\n"


def prefix_policy(minimum=4) -> dict:
    """Supply a typed hybrid/MTP policy without loading a model or inspecting its name."""
    return {"context_length": 8192, "minimum_prompt_tokens": minimum, "max_tokens": 128,
            "prefix_state": "hybrid_recurrent", "mtp_policy": "dynamic"}


def prefix_reply(request, prompt, *, matched=0) -> dict:
    """Return terminal authority evidence distinct from prompt-prefix eligibility."""
    full = request.prefix_probe is hammer.PrefixProbe.FULL
    restored = request.prefix_probe is not hammer.PrefixProbe.FRESH
    return {"object": "chat.completion", "choices": [{"message": {
        "content": json.dumps({"answer": request.expected_json_answer})}, "finish_reason": "stop"}],
        "token_ids": {"prompt": list(prompt), "completion": [100, 101]},
        "usage": {"prompt_tokens": len(prompt), "completion_tokens": 2, "total_tokens": len(prompt) + 2},
        "runtime_summary": {"schema": 1, "mtp": {"enabled": True}, "prefix_cache": {
            "enabled": True, "bypassed": False, "bypass_reason": "", "hit": full,
            "partial_hit": request.prefix_probe is hammer.PrefixProbe.PARTIAL,
            "requested_tokens": len(prompt), "matched_tokens": matched,
            "matched_blocks": 1 if restored else 0, "storage_tier": "ram" if restored else "none",
            "terminal_logits_restored": full, "terminal_hidden_restored": full,
            "hybrid_state_restored": restored, "mtp_state_restored": restored,
            "admission_epoch_earliest": 0, "admission_epoch_latest": 0,
            "completion_movement_epoch": 2}}}


@contextmanager
def http_peer(*, status=200, body="", delay=0):
    """Own a local HTTP peer and every test thread; no external network is used."""
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            """Capture one request and optionally stall after partial response publication."""
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                self.wfile.write(body.encode())
                self.wfile.flush()
                if delay:
                    time.sleep(delay)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args):
            """Keep the self-test output limited to unittest results."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class HTTPHammerTests(unittest.TestCase):
    """One writer and one terminal outcome must survive every interruption boundary."""

    def test_exact_order_is_repeated_without_relaunch_or_certification(self):
        """The ready server and immutable requests survive three complete workload cycles."""
        order = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "hammer"
            cases = sequence()["requests"]
            def send(_url, stem, *_args):
                body = json.loads(stem.with_suffix(".request.json").read_text())
                progress = json.loads((root / "progress.json").read_text())
                self.assertEqual(progress["state"], "request_in_flight")
                case = cases[len(order) % len(cases)]
                self.assertEqual(body, case["body"])
                order.append(case["name"])
                stem.with_suffix(".response.txt").write_text(reply(case["expected_numeric_answer"], body["stream"]))
                return {"return_code": 0, "http_status": "200"}
            self.assertEqual(hammer.run("http://unused", sequence(), root, cycle_limit=3, send=send), 0)
            progress = json.loads((root / "progress.json").read_text())
            self.assertEqual(order, [case["name"] for case in cases] * 3)
            self.assertEqual(progress["completed_cycles"], 3)
            self.assertEqual(progress["completed_requests"], 9)
            self.assertFalse(progress["certified"])
            self.assertIsNone(progress["active_request"])
            self.assertEqual(progress["state"], "bounded_diagnostic_complete")

    def test_first_failure_is_frozen_and_later_requests_never_run(self):
        """Wrong math is a failure even with a clean HTTP status; no replay retry follows."""
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "hammer"
            def send(_url, stem, *_args):
                calls.append(stem)
                stem.with_suffix(".response.txt").write_text(reply("4" if len(calls) == 1 else "99"))
                return {"return_code": 0, "http_status": "200"}
            self.assertEqual(hammer.run("http://unused", sequence(), root, send=send), 1)
            progress = json.loads((root / "progress.json").read_text())
            self.assertEqual(len(calls), 2)
            self.assertEqual(progress["completed_requests"], 1)
            self.assertEqual(progress["first_failure"]["name"], "changed")
            self.assertEqual(progress["active_request"]["ordinal"], 2)
            self.assertEqual(progress["state"], "failed")

    def test_interruption_retains_the_pending_request_and_previous_results(self):
        """A killed client cannot erase the failing request or manufacture completion."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "hammer"
            def cancelled(*_args):
                raise KeyboardInterrupt("cancel")
            with self.assertRaises(KeyboardInterrupt):
                hammer.run("http://unused", sequence(), root, send=cancelled)
            progress = json.loads((root / "progress.json").read_text())
            self.assertEqual(progress["state"], "interrupted")
            self.assertEqual(progress["active_request"]["name"], "fresh")
            self.assertEqual(progress["completed_requests"], 0)
            self.assertIsNone(progress["first_failure"])

    def test_invalid_transitions_cannot_overwrite_the_first_failure(self):
        """Journal ownership prevents double-completion and submission after a red."""
        with tempfile.TemporaryDirectory() as temporary:
            journal = hammer.ReplayJournal(Path(temporary) / "hammer", sequence())
            case = hammer.read_sequence(sequence())[0]
            journal.begin(case, 1, 1)
            with self.assertRaises(ValueError):
                journal.begin(case, 1, 2)
            with self.assertRaises(ValueError):
                journal.finish({"ordinal": 2}, failed=False)
            journal.finish({"ordinal": 1, "outcome": "failed"}, failed=True)
            with self.assertRaises(ValueError):
                journal.finish({"ordinal": 1}, failed=False)
            with self.assertRaises(ValueError):
                journal.begin(case, 2, 2)

    def test_existing_root_is_never_overwritten(self):
        """Restart requires a fresh evidence root, not replacement of an old red."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "hammer"
            hammer.ReplayJournal(root, sequence())
            original = (root / "progress.json").read_bytes()
            with self.assertRaises(FileExistsError):
                hammer.ReplayJournal(root, sequence())
            self.assertEqual((root / "progress.json").read_bytes(), original)

    def test_invalid_sequences_and_budgets_fail_before_http(self):
        """Missing or ambiguous oracles cannot weaken a replay into a smoke test."""
        bad = [{}, {"schema": True, "requests": []}, {"schema": 1, "requests": []},
               {"schema": 1, "requests": sequence()["requests"] * 2}]
        for body in bad:
            with self.subTest(body=body), self.assertRaises(ValueError):
                hammer.read_sequence(body)
        for options in ({"cycle_limit": -1}, {"cycle_limit": True},
                        {"request_timeout": float("nan")}, {"stall_seconds": 180},
                        {"on_stall": "shell text"}, {"on_stall": []}):
            with tempfile.TemporaryDirectory() as temporary, self.subTest(options=options), \
                    patch.object(hammer, "ReplayJournal") as journal:
                with self.assertRaises(ValueError):
                    hammer.run("http://unused", sequence(), Path(temporary), **options)
                journal.assert_not_called()

    def test_stream_errors_truncation_and_missing_done_are_not_success(self):
        """HTTP-200 SSE cannot hide a native failure or silently truncated arithmetic."""
        case = hammer.read_sequence(sequence())[-1]
        for raw in (reply("4", True).replace("data: [DONE]\n\n", ""),
                    reply("4", True).replace('"stop"', '"length"'),
                    'data: {"error":{"message":"native failure"}}\n\ndata: [DONE]\n\n'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                hammer.validate_response(case, raw)

    def test_real_client_saves_body_of_http_error_without_retry(self):
        """curl's status and body both survive a non-2xx response."""
        with tempfile.TemporaryDirectory() as temporary, http_peer(status=500, body="native error") as url:
            stem = Path(temporary) / "request"
            stem.with_suffix(".request.json").write_text("{}")
            result = hammer.send_request(url, stem, 1, 0.05, [])
            self.assertNotEqual(result["return_code"], 0)
            self.assertEqual(result["http_status"], "500")
            self.assertEqual(stem.with_suffix(".response.txt").read_text(), "native error")

    def test_partial_sse_deadline_runs_diagnostic_once_and_preserves_bytes(self):
        """The callback is dormant until a wait and never substitutes for the HTTP outcome."""
        with tempfile.TemporaryDirectory() as temporary, http_peer(body="data: partial\n\n", delay=0.2) as url:
            stem = Path(temporary) / "request"
            stem.with_suffix(".request.json").write_text("{}")
            with patch.object(hammer, "capture_stall", return_value={"return_code": 0}) as capture:
                result = hammer.send_request(url, stem, 0.12, 0.03, ["explicit-diagnostic"])
            self.assertEqual(result["return_code"], 28)
            self.assertEqual(capture.call_count, 1)
            self.assertEqual(stem.with_suffix(".response.txt").read_text(), "data: partial\n\n")

    def test_healthy_request_never_attaches_diagnostic(self):
        """No steady-state debugger, tracer or callback enters the success path."""
        with tempfile.TemporaryDirectory() as temporary, http_peer(body=reply("4")) as url:
            stem = Path(temporary) / "request"
            stem.with_suffix(".request.json").write_text("{}")
            with patch.object(hammer, "capture_stall") as capture:
                result = hammer.send_request(url, stem, 1, 0.2, ["explicit-diagnostic"])
            self.assertEqual(result["return_code"], 0)
            capture.assert_not_called()

    def test_outer_lifetime_uses_one_owned_group_not_a_certification_deadline(self):
        """An indefinite diagnostic still reaps its exact server group on exit."""
        with patch.object(driver.subprocess, "Popen") as popen, \
                patch.object(driver.e2e.parity, "_terminate_process_group") as retire:
            process = popen.return_value.__enter__.return_value
            process.wait.return_value = 1
            self.assertEqual(driver.run_owned_harness(["harness"], {}, None), 1)
            process.wait.assert_called_once_with()
            retire.assert_called_once_with(process)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_interrupted_outer_lifetime_still_reaps_only_its_owned_group(self):
        """An interrupted wait must not leave model workers running outside their lease."""
        with patch.object(driver.subprocess, "Popen") as popen, \
                patch.object(driver.e2e.parity, "_terminate_process_group") as retire:
            process = popen.return_value.__enter__.return_value
            process.wait.side_effect = KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                driver.run_owned_harness(["harness"], {}, None)
            retire.assert_called_once_with(process)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_actual_shell_workload_branch_keeps_shared_retirement(self):
        """The diagnostic branch does not implement a parallel launch/shutdown system."""
        shell = (ROOT / "tests/v2/e2e/server/test_server_e2e.sh").read_text()
        marker = 'elif [[ -n "$HTTP_LIFECYCLE_HAMMER_FILE" ]]; then'
        self.assertLess(shell.index('if [[ -n "$GENERATION_CONFIG_FILE" ]]; then', shell.index('run_backend_tests()')), shell.index(marker))
        self.assertLess(shell.index(marker), shell.index('shutdown_and_validate "$tag"'))
        subprocess.run(["bash", "-n", str(ROOT / "tests/v2/e2e/server/test_server_e2e.sh")], check=True)


class HTTPHammerDriverTests(unittest.TestCase):
    """Canonical discovery/staging/serve remain shared; only the diagnostic workload differs."""

    @contextmanager
    def canonical_fixture(self, root):
        """Inject one typed cell and a reusable staging lease, never a model or server."""
        with ExitStack() as mocks:
            model = root / "model.gguf"
            profile = {"context_length": 8192, "minimum_prompt_tokens": 4096,
                       "generation_tokens": 2048, "request_timeout_seconds": 180,
                       "readiness_timeout_seconds": 60,
                       "cell_timeout_seconds": {"AVX512": 900, "AVX2": 900},
                       "thinking_modes": "both", "movement_evidence": "not_applicable",
                       "tool_calling": "required", "server_args": ["--auto", "--mtp"],
                       "planning": {"mode": "auto", "strategy": "single", "mpi_ranks": 1,
                                    "device_counts": {"rocm": 1}}}
            record = {"id": "canonical-test", "model": str(model), "e2e": profile,
                      "runtime": {"context_length": 4096, "generation": {
                          "prefix_state": "hybrid_recurrent", "mtp_policy": "dynamic"}}}
            campaign = SimpleNamespace(group=SimpleNamespace(backends="ROCm"))
            mocks.enter_context(patch.object(driver.e2e, "discover", return_value=[(campaign, "exact-case", record)]))
            mocks.enter_context(patch.object(driver.e2e, "release_binary_cpu_isa", return_value=driver.e2e.CPUISA.AVX512))
            workspace = MagicMock(models=root, persistent=True)
            lease = mocks.enter_context(patch.object(driver.e2e.parity, "model_staging_workspace"))
            lease.return_value.__enter__.return_value = workspace
            stage = mocks.enter_context(patch.object(driver.e2e.parity, "stage_models_in_ramdisk",
                return_value=([SimpleNamespace(source_path=str(model), filename=model.name)], None)))
            run = mocks.enter_context(patch.object(driver, "run_owned_harness", return_value=1))
            driver_proof = mocks.enter_context(patch.object(driver.e2e, "validate_driver_diagnostics"))
            source = root / "sequence.json"
            source.write_text(json.dumps(sequence()))
            output = root / "diagnostic"
            arguments = ["--binary", str(root / "llaminar2"), "--manifest", str(root / "manifest.json"),
                         "--cell", "exact-case", "--sequence", str(source), "--output", str(output),
                         "--prefix-pressure"]
            yield SimpleNamespace(profile=profile, record=record, campaign=campaign, workspace=workspace,
                                  lease=lease, stage=stage, run=run, driver_proof=driver_proof,
                                  output=output, arguments=arguments)

    def test_one_canonical_cell_owns_one_model_lease_and_unmodified_launch_policy(self):
        """The wrapper cannot expand a topology or mint an image certificate from a replay."""
        with tempfile.TemporaryDirectory() as temporary, self.canonical_fixture(Path(temporary)) as fixture:
            code = driver.main(fixture.arguments)
            self.assertEqual(code, 1)
            fixture.lease.assert_called_once()
            self.assertEqual(fixture.stage.call_args.args[0], [fixture.campaign])
            fixture.workspace.protect_published_models.assert_called_once_with()
            fixture.run.assert_called_once()
            fixture.driver_proof.assert_not_called()
            command, environment, _ = fixture.run.call_args.args
            self.assertIn("test_server_e2e.sh", command[1])
            self.assertIn("--http-lifecycle-hammer", command)
            self.assertEqual(environment["LLAMINAR_E2E_CONTEXT_LENGTH"], "8192")
            self.assertEqual(environment["LLAMINAR_E2E_LOG_DIR"],
                             str(fixture.output / "lifetimes/000001"))
            self.assertEqual(json.loads((fixture.output / "server-args.json").read_text()),
                             fixture.profile["server_args"])
            config = json.loads((fixture.output / "hammer-configuration.json").read_text())
            self.assertEqual(config["sequence"], sequence())
            self.assertEqual(config["cycle_limit"], 0)
            self.assertEqual(config["prefix_pressure"], {**prefix_policy(4096), "context_length": 8192})
            report = json.loads((fixture.output / "report.json").read_text())
            self.assertEqual(report["configuration"], fixture.record)
            self.assertEqual(report["case"], "exact-case")
            self.assertEqual(report["return_code"], 1)
            self.assertFalse(report["certified"])
            self.assertEqual(report["completed_lifetimes"], 0)
            self.assertIsNone(report["active_lifetime"])
            self.assertEqual(report["lifetimes"][0]["state"], "failed")

    def test_fresh_lifetimes_reuse_one_lease_and_require_retired_driver_evidence(self):
        """Each normal retirement closes its own evidence before the next server starts."""
        with tempfile.TemporaryDirectory() as temporary, self.canonical_fixture(Path(temporary)) as fixture:
            fixture.run.side_effect = [0, 0, 0]
            ordering = MagicMock()
            ordering.attach_mock(fixture.run, "retire")
            ordering.attach_mock(fixture.driver_proof, "prove")
            self.assertEqual(driver.main(fixture.arguments + ["--cycle-limit", "2",
                                                               "--lifetime-limit", "3"]), 0)
            fixture.lease.assert_called_once()
            fixture.stage.assert_called_once()
            fixture.workspace.protect_published_models.assert_called_once_with()
            self.assertEqual([call[0] for call in ordering.mock_calls],
                             ["retire", "prove"] * 3)
            destinations = [fixture.output / "lifetimes" / f"{ordinal:06d}"
                            for ordinal in range(1, 4)]
            self.assertEqual([call.args[0] for call in fixture.driver_proof.call_args_list], destinations)
            commands = [call.args[0] for call in fixture.run.call_args_list]
            self.assertEqual(commands, [commands[0]] * 3)
            for call, destination in zip(fixture.run.call_args_list, destinations):
                self.assertEqual(call.args[1]["LLAMINAR_E2E_LOG_DIR"], str(destination))
                self.assertTrue((destination / "harness.log").is_file())
            report = json.loads((fixture.output / "report.json").read_text())
            self.assertEqual(report["completed_lifetimes"], 3)
            self.assertEqual(report["lifetime_limit"], 3)
            self.assertEqual(report["cycles_per_lifetime"], 2)
            self.assertEqual(report["state"], "bounded_diagnostic_complete")
            self.assertFalse(report["certified"])
            self.assertEqual([row["state"] for row in report["lifetimes"]], ["passed"] * 3)

    def test_unbounded_fresh_lifetimes_stop_on_the_first_failure_without_retry(self):
        """A failed third retirement cannot be overwritten by a fourth healthy launch."""
        with tempfile.TemporaryDirectory() as temporary, self.canonical_fixture(Path(temporary)) as fixture:
            fixture.run.side_effect = [0, 0, 7, 0]
            self.assertEqual(driver.main(fixture.arguments + ["--cycle-limit", "1",
                                                               "--lifetime-limit", "0"]), 7)
            self.assertEqual(fixture.run.call_count, 3)
            self.assertEqual(fixture.driver_proof.call_count, 2)
            fixture.lease.assert_called_once()
            fixture.stage.assert_called_once()
            report = json.loads((fixture.output / "report.json").read_text())
            self.assertEqual(report["completed_lifetimes"], 2)
            self.assertEqual(report["state"], "failed")
            self.assertEqual(report["return_code"], 7)
            self.assertEqual([row["state"] for row in report["lifetimes"]],
                             ["passed", "passed", "failed"])
            self.assertFalse((fixture.output / "lifetimes/000004").exists())

    def test_missing_final_driver_evidence_prevents_a_subsequent_lifetime(self):
        """Exit zero cannot manufacture a pass from an open or missing driver checkpoint."""
        with tempfile.TemporaryDirectory() as temporary, self.canonical_fixture(Path(temporary)) as fixture:
            fixture.run.side_effect = [0, 0]
            fixture.driver_proof.side_effect = ValueError("incomplete driver checkpoint")
            with self.assertRaisesRegex(ValueError, "incomplete driver checkpoint"):
                driver.main(fixture.arguments + ["--cycle-limit", "1", "--lifetime-limit", "0"])
            fixture.run.assert_called_once()
            report = json.loads((fixture.output / "report.json").read_text())
            self.assertEqual(report["completed_lifetimes"], 0)
            self.assertEqual(report["state"], "interrupted_or_infrastructure_error")
            self.assertEqual(report["lifetimes"][0]["state"], "interrupted_or_infrastructure_error")

    def test_interruption_preserves_completed_and_unfinished_lifetime_evidence(self):
        """An interrupted second server cannot erase the first pass or admit a third."""
        with tempfile.TemporaryDirectory() as temporary, self.canonical_fixture(Path(temporary)) as fixture:
            fixture.run.side_effect = [0, KeyboardInterrupt()]
            with self.assertRaises(KeyboardInterrupt):
                driver.main(fixture.arguments + ["--cycle-limit", "1", "--lifetime-limit", "0"])
            self.assertEqual(fixture.run.call_count, 2)
            self.assertEqual(fixture.driver_proof.call_count, 1)
            report = json.loads((fixture.output / "report.json").read_text())
            self.assertEqual(report["completed_lifetimes"], 1)
            self.assertEqual([row["state"] for row in report["lifetimes"]],
                             ["passed", "interrupted_or_infrastructure_error"])
            self.assertIsNone(report["lifetimes"][1]["return_code"])
            self.assertFalse((fixture.output / "lifetimes/000003").exists())

    def test_invalid_lifetime_geometry_fails_before_discovery_or_staging(self):
        """Every repeated lifetime needs a finite workload to reach normal retirement."""
        with patch.object(driver.e2e, "discover") as discover, \
                patch.object(driver.e2e.parity, "model_staging_workspace") as lease:
            for cycles, lifetimes in ((0, 0), (0, 2), (1, -1), (-1, 1)):
                with self.subTest(cycles=cycles, lifetimes=lifetimes), \
                        patch.object(sys, "stderr", new_callable=StringIO):
                    with self.assertRaises(SystemExit) as error:
                        driver.main(["--manifest", "unused", "--cell", "exact", "--sequence", "unused",
                                     "--output", "unused", "--cycle-limit", str(cycles),
                                     "--lifetime-limit", str(lifetimes)])
                    self.assertEqual(error.exception.code, 2)
            discover.assert_not_called()
            lease.assert_not_called()

    def test_ambiguous_selection_fails_before_model_staging(self):
        """An indefinite hammer cannot reserve hardware for an accidental wildcard matrix."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sequence.json"
            source.write_text(json.dumps(sequence()))
            for selected in ([], [object(), object()]):
                with self.subTest(count=len(selected)), \
                        patch.object(driver.e2e, "discover", return_value=selected), \
                        patch.object(driver.e2e.parity, "model_staging_workspace") as lease:
                    with self.assertRaisesRegex(ValueError, "exactly one"):
                        driver.main(["--manifest", str(root / "manifest.json"), "--cell", ".*",
                                     "--sequence", str(source), "--output", str(root / "diagnostic")])
                    lease.assert_not_called()


class PrefixPressureTests(unittest.TestCase):
    """Prefix lookup is never a restore certificate; exercise each required state plane."""

    def test_long_needles_rotate_and_cold_identity_changes_before_cached_boundaries(self):
        """The existing needle builder owns geometry and independently known answers."""
        pressure = hammer.PrefixPressure.from_document(prefix_policy(4096))
        cycles = [pressure.requests(cycle, "test-run") for cycle in range(1, 4)]
        self.assertEqual([cases[0].expected_json_answer for cases in cycles],
                         ["TUNDRA-84QX", "COBALT-27LM", "RIVER-93RN"])
        self.assertEqual(len({cases[0].body["messages"][0]["content"] for cases in cycles}), 3)
        self.assertNotEqual(cycles[0][0].body, pressure.requests(1, "another-run")[0].body)
        for cases in cycles:
            cold, full, partial, suffix, after_short, suffix_after = cases
            self.assertGreater(len(cold.body["messages"][-1]["content"]), 8000)
            self.assertEqual(cold.body, full.body)
            self.assertEqual(cold.body, after_short.body)
            self.assertEqual(partial.body, suffix.body)
            self.assertEqual(partial.body, suffix_after.body)
            self.assertEqual(partial.body["messages"][:2], cold.body["messages"])
            self.assertEqual(partial.body["messages"][-1]["role"], "assistant")
            for request in cases:
                self.assertTrue(request.body["return_token_ids"])
                self.assertTrue(request.body["return_runtime_summary"])
                self.assertFalse(request.body["stream"])

    def test_pressure_cycles_prove_receipts_and_preserve_the_original_http_sequence(self):
        """Large/small resets are surrounded by actual full/partial restore witnesses."""
        pressure = hammer.PrefixPressure.from_document(prefix_policy())
        order, callbacks = [], []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "hammer"
            def send(_url, stem, _timeout, _stall, diagnostic):
                cycle = len(order) // 9 + 1
                nonce = json.loads((root / "progress.json").read_text())["run_nonce"]
                prefix = pressure.requests(cycle, nonce)
                originals = hammer.read_sequence(sequence())
                expected = (*prefix[:4], *originals, *prefix[4:])[len(order) % 9]
                body = json.loads(stem.with_suffix(".request.json").read_text())
                self.assertEqual(body, expected.body)
                order.append(expected.name)
                callbacks.append(diagnostic)
                if expected.prefix_probe:
                    prompt = [cycle * 10 + n for n in range(4)]
                    if "suffix" in expected.name:
                        prompt += [900, 901]
                    matched = 0 if expected.prefix_probe is hammer.PrefixProbe.FRESH else (
                        4 if expected.prefix_probe is hammer.PrefixProbe.PARTIAL else len(prompt))
                    response = json.dumps(prefix_reply(expected, prompt, matched=matched))
                else:
                    response = reply(expected.expected_numeric_answer, body["stream"])
                stem.with_suffix(".response.txt").write_text(response)
                return {"return_code": 0, "http_status": "200"}
            self.assertEqual(hammer.run("http://unused", sequence(), root, cycle_limit=3,
                prefix_pressure=pressure.document(), on_stall=["passive-probe"], send=send), 0)
            progress = json.loads((root / "progress.json").read_text())
            self.assertEqual(progress["completed_requests"], 27)
            self.assertEqual(progress["prefix_probes"], {"fresh": 3, "full": 12, "partial": 3})
            self.assertEqual(progress["prefix_storage_tiers"], {"none": 3, "ram": 15})
            self.assertEqual(order, ["prefix_cold_long", "prefix_full_long", "prefix_partial_suffix",
                "prefix_full_suffix", "fresh", "changed", "restore_stream",
                "prefix_full_after_short", "prefix_suffix_after_short"] * 3)
            self.assertEqual(callbacks[0], [])  # Healthy long cold prefill must not be debug-attached.
            self.assertEqual(callbacks[1], ["passive-probe"])
            observed = json.loads((root / "http_000000009.result.json").read_text())["prefix_evidence"]
            self.assertEqual(observed["source_ordinal"], 3)
            self.assertTrue(observed["prefix_cache"]["mtp_state_restored"])

    def test_full_restore_requires_all_state_planes_and_byte_exact_tokens(self):
        """A matching answer cannot hide missing GDN/MTP state, a miss or one-token drift."""
        policy = hammer.PrefixPressure.from_document(prefix_policy())
        cold, full, *_ = policy.requests(1, "test-run")
        baseline = hammer.TokenTrace((1, 2, 3, 4), (100, 101), "stop")
        faults = ("hybrid_state_restored", "mtp_state_restored", "terminal_hidden_restored",
                  "terminal_logits_restored", "missing_summary", "token_drift", "usage_mismatch", "cache_miss")
        for fault in faults:
            data = prefix_reply(full, baseline.prompt, matched=4)
            if fault == "missing_summary":
                del data["runtime_summary"]
            elif fault == "token_drift":
                data["token_ids"]["completion"][-1] += 1
            elif fault == "usage_mismatch":
                data["usage"]["prompt_tokens"] += 1
            elif fault == "cache_miss":
                data["runtime_summary"]["prefix_cache"].update(hit=False, matched_tokens=0)
            else:
                data["runtime_summary"]["prefix_cache"][fault] = False
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                hammer.validate_prefix_response(full, data, policy, {cold.name: (baseline, 1)}, 2)

    def test_partial_restore_requires_the_completed_source_token_prefix(self):
        """Text similarity and a self-reported hit cannot replace the earlier full token boundary."""
        policy = hammer.PrefixPressure.from_document(prefix_policy())
        cold, _, partial, *_ = policy.requests(1, "test-run")
        baseline = hammer.TokenTrace((1, 2, 3, 4), (100, 101), "stop")
        for prompt, matched, source in (((1, 2, 3, 9, 5), 4, {cold.name: (baseline, 1)}),
            ((1, 2, 3, 4, 5), 3, {cold.name: (baseline, 1)}), ((1, 2, 3, 4, 5), 4, {})):
            with self.subTest(prompt=prompt, matched=matched), self.assertRaises(ValueError):
                hammer.validate_prefix_response(partial, prefix_reply(partial, prompt, matched=matched),
                                                policy, source, 3)

    def test_unknown_canonical_policy_fails_before_journal_creation(self):
        """Backend/model names cannot substitute for missing typed state or MTP intent."""
        for field in ("prefix_state", "mtp_policy", "context_length"):
            policy = prefix_policy()
            del policy[field]
            with self.subTest(field=field), patch.object(hammer, "ReplayJournal") as journal:
                with self.assertRaises(ValueError):
                    hammer.run("http://unused", sequence(), Path("unused"), prefix_pressure=policy)
                journal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
