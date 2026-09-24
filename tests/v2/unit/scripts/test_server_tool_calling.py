#!/usr/bin/env python3
"""Device-free protocol and fail-closed evidence tests for real HTTP tool probes.

A loopback HTTP server exercises the same client transport as production E2E;
synthetic messages test validator rejection, not model correctness. Every live
model cell still has to generate and consume its own tool call.
"""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests/v2/e2e/server"))
import tool_calling_checks as tools


def response(stream=False, followup=False):
    """Encode one valid response, splitting tool arguments across SSE events."""
    call = {"id": "call_17", "type": "function",
            "function": {"name": tools.TOOL_NAME, "arguments": json.dumps(tools.ARGUMENTS)}}
    message = {"role": "assistant", "content": tools.RECEIPT if followup else None}
    if not followup:
        message["tool_calls"] = [call]
    finish = "stop" if followup else "tool_calls"
    if not stream:
        return {"status": 200, "content_type": "application/json",
                "body": json.dumps({"choices": [{"message": message, "finish_reason": finish}]})}
    if followup:
        deltas = [{"role": "assistant"}, {"content": tools.RECEIPT[:4]}, {"content": tools.RECEIPT[4:]}]
    else:
        deltas = [{"role": "assistant"}, {"tool_calls": [{"index": 0, "id": call["id"],
            "type": "function", "function": {"name": tools.TOOL_NAME, "arguments": ""}}]}]
        deltas.extend({"tool_calls": [{"index": 0, "function": {"arguments": char}}]}
                      for char in call["function"]["arguments"])
    events = [{"choices": [{"delta": delta, "finish_reason": None}]} for delta in deltas]
    events.append({"choices": [{"delta": {}, "finish_reason": finish}]})
    body = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
    return {"status": 200, "content_type": "text/event-stream", "body": body}


def row_for(probe):
    """Create a complete evidence fixture bound to the oracle's exact requests."""
    name, stream, choice = probe
    first = response(stream)
    call = tools.required_call(first, stream)
    return {"name": name, "passed": True, "request": tools.request_body(stream, choice),
            "response": first, "followup_request": tools.followup_body(call, stream),
            "followup_response": response(stream, True)}


class ServerToolCallingTests(unittest.TestCase):
    """Cover transport, reconstruction, identity joins and incomplete evidence."""

    def test_real_loopback_http_exercises_every_mode_and_retains_exchanges(self):
        """The executable probe sends eight real requests and the evidence revalidates."""
        received = []

        class Handler(BaseHTTPRequestHandler):
            """A deterministic wire peer, never a substitute for a model-cell pass."""
            def do_POST(self):
                """Record incoming bytes and return the selected OpenAI wire form."""
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append((self.path, body))
                output = response(body["stream"], body["tool_choice"] == "none")
                payload = output["body"].encode()
                self.send_response(200)
                self.send_header("Content-Type", output["content_type"])
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                """Keep device-free gate output concise; assertions own failures."""

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
            worker.start()
            try:
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "tools.json"
                    result = tools.run_checks(f"http://127.0.0.1:{server.server_port}", output, 2)
                    self.assertEqual(result, json.loads(output.read_text()))
                    tools.validate_evidence(result)
            finally:
                server.shutdown()
                worker.join()
        self.assertEqual(len(received), 8)
        self.assertTrue(all(path == "/v1/chat/completions" for path, _ in received))
        self.assertNotIn(tools.RECEIPT, json.dumps(tools.MESSAGES))

    def test_json_and_fragmented_sse_have_the_same_semantic_call(self):
        """Chunk boundaries and empty initial deltas cannot alter function arguments."""
        self.assertEqual(tools.required_call(response(), False), tools.required_call(response(True), True))
        tools.validate_evidence({"schema": 1, "complete": True, "results": [row_for(p) for p in tools.PROBES]})

    def test_malformed_json_shapes_fail_cleanly_in_both_wire_modes(self):
        """Broken device/server output is a failed probe, not an uncaught aggregate crash."""
        for value in ([], None, 3, {"choices": [None]}, {"choices": [3]}):
            for stream in (False, True):
                raw = json.dumps(value)
                if stream: raw = "data: " + raw + "\n\ndata: [DONE]\n\n"
                with self.subTest(value=value, stream=stream), self.assertRaises(ValueError):
                    tools.response_message({"status": 200, "content_type": "text/event-stream", "body": raw}, stream)

    def test_sse_rejects_incomplete_duplicate_or_changed_identity(self):
        """A 200 response without authenticated terminal state never passes."""
        original = response(True)
        mutations = [original["body"].replace("data: [DONE]\n\n", ""),
                     original["body"] + 'data: {"choices": []}\n\n',
                     original["body"].replace('"index": 0', '"index": 1'),
                     original["body"].replace('"finish_reason": "tool_calls"', '"finish_reason": "length"'),
                     original["body"].replace('"function": {"arguments": "{"}',
                         '"id": "changed", "function": {"arguments": "{"}')]
        for raw in mutations:
            with self.subTest(raw=raw[-150:]), self.assertRaises(ValueError):
                tools.required_call({**original, "body": raw}, True)

    def test_saved_pass_flags_cannot_hide_wrong_requests_arguments_or_answers(self):
        """The outer evidence reader recomputes the oracle, not only its flags."""
        valid = {"schema": 1, "complete": True, "results": [row_for(p) for p in tools.PROBES]}
        for mutation in ("empty", "incomplete", "request", "join", "answer", "arguments", "finish"):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(valid)
                first = bad["results"][0]
                if mutation == "empty": bad["results"] = []
                elif mutation == "incomplete": bad["complete"] = False
                elif mutation == "request": first["request"]["tools"] = []
                elif mutation == "join": first["followup_request"]["messages"][-1]["tool_call_id"] = "unrelated"
                elif mutation == "answer": first["followup_response"]["body"] = response()["body"]
                elif mutation == "arguments": first["response"]["body"] = first["response"]["body"].replace("cobalt-widget", "wrong")
                elif mutation == "finish": first["response"]["body"] = first["response"]["body"].replace('"finish_reason": "tool_calls"', '"finish_reason": "stop"')
                with self.assertRaises(ValueError): tools.validate_evidence(bad)

    def test_failed_probe_does_not_hide_remaining_protocol_results(self):
        """Continue independent modes once, but preserve a failed aggregate."""
        def respond(base, body, timeout):
            if body["tool_choice"] == "required" and not body["stream"]:
                raise ValueError("broken non-streaming path")
            return response(body["stream"], body["tool_choice"] == "none")
        with tempfile.TemporaryDirectory() as directory, patch.object(tools, "exchange", side_effect=respond):
            result = tools.run_checks("http://unused", Path(directory) / "tools.json", 2)
        self.assertTrue(result["complete"])
        self.assertEqual([r["passed"] for r in result["results"]], [False, True, True, True])
        with self.assertRaisesRegex(ValueError, "failed"): tools.validate_evidence(result)

    def test_cancellation_retains_an_incomplete_probe(self):
        """A watchdog cannot leave an apparently complete certificate behind."""
        with tempfile.TemporaryDirectory() as directory, patch.object(tools, "exchange", side_effect=KeyboardInterrupt):
            path = Path(directory) / "tools.json"
            with self.assertRaises(KeyboardInterrupt): tools.run_checks("http://unused", path, 2)
            with self.assertRaisesRegex(ValueError, "incomplete"): tools.validate_evidence(json.loads(path.read_text()))


if __name__ == "__main__":
    unittest.main()
