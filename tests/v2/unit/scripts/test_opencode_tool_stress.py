#!/usr/bin/env python3
"""Device-free regressions for exact OpenCode transport and stress evidence.

Loopback peers prove that recording leaves request/response bytes untouched.
Small conversations test schema types, completed tool identities, continuation
joins and matching write/read arguments as well as exact final file bytes;
task-quality observations must never erase malformed or incomplete protocol
evidence. The protocol gate requires successful coverage independently of the
model's task score. These fixtures cannot certify a live model's behavior.
"""
import copy
import errno
import gc
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests/v2/e2e/server"))
import opencode_tool_stress as stress


def wire(message, stream=False, usage=None):
    """Encode JSON or character-fragmented SSE for the same semantic message."""
    finish = "tool_calls" if message.get("tool_calls") else "stop"
    if not stream:
        return {"status": 200, "content_type": "application/json",
                "body": json.dumps({"choices": [{"message": message, "finish_reason": finish}]})}
    deltas = [{"role": "assistant"}]
    for index, call in enumerate(message.get("tool_calls", [])):
        deltas.append({"tool_calls": [{"index": index, "id": call["id"], "type": "function",
                                      "function": {"name": call["function"]["name"], "arguments": ""}}]})
        deltas.extend({"tool_calls": [{"index": index, "function": {"arguments": char}}]}
                      for char in call["function"]["arguments"])
    if message.get("content"):
        deltas.append({"content": message["content"]})
    events = [{"choices": [{"delta": delta, "finish_reason": None}]} for delta in deltas]
    events.append({"choices": [{"delta": {}, "finish_reason": finish}]})
    if usage is not None:
        for event in events:
            event["usage"] = None
        events.append({"choices": [], "usage": usage})
        for event in events:
            event.update(id="chatcmpl-fixture", model="test-model", created=123)
    return {"status": 200, "content_type": "text/event-stream",
            "body": "".join("data: " + json.dumps(event) + "\n\n" for event in events)
                    + "data: [DONE]\n\n"}


def conversation(directory, stream=False):
    """Create one write/read/terminal conversation with real client event shapes."""
    tools = [{"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "required": list(keys),
        "properties": {key: {"type": "string"} for key in keys}}}}
        for name, keys in (("write", ("filePath", "content")), ("read", ("filePath",)))]
    calls = [{"id": "call_" + name, "type": "function", "function": {
        "name": name, "arguments": json.dumps({"filePath": "result.txt", **arguments})}}
        for name, arguments in (("write", {"content": "123"}), ("read", {}))]
    messages, rows, events = [{"role": "user", "content": "Write and read 123"}], [], []
    for call in calls:
        message = {"role": "assistant", "content": None, "tool_calls": [call]}
        rows.append({"path": "/v1/chat/completions", "request": {
            "stream": stream, "messages": copy.deepcopy(messages), "tools": tools},
            "response": wire(message, stream)})
        messages.extend([message, {"role": "tool", "tool_call_id": call["id"], "content": "ok"}])
        events.append({"type": "tool_use", "part": {"type": "tool", "tool": call["function"]["name"],
            "callID": call["id"], "state": {"status": "completed",
                "input": json.loads(call["function"]["arguments"])}}})
    rows.append({"path": "/v1/chat/completions", "request": {
        "stream": stream, "messages": messages, "tools": tools},
        "response": wire({"role": "assistant", "content": "Done"}, stream)})
    events.append({"type": "step_finish", "part": {"type": "step-finish", "reason": "stop"}})
    (directory / "client.jsonl").write_text("\n".join(json.dumps(event) for event in events))
    (directory / "expected.bin").write_bytes(b"123")
    (directory / "workspace").mkdir()
    (directory / "workspace/result.txt").write_bytes(b"123")
    (directory / "workspace-location.json").write_text(json.dumps({
        "live_workspace": str(directory / "workspace"), "copied_and_retired": True}))
    return rows, events


def amend_call(rows, events, index, arguments):
    """Change generated arguments consistently at wire, history and client boundaries."""
    message, _ = stress.response_message(rows[index]["response"], rows[index]["request"].get("stream", False))
    call = message["tool_calls"][0]
    call["function"]["arguments"] = json.dumps(arguments)
    rows[index]["response"] = wire(message, rows[index]["request"].get("stream", False))
    for row in rows[index + 1:]:
        for previous in row["request"]["messages"]:
            for recorded in previous.get("tool_calls", []):
                if recorded["id"] == call["id"]:
                    recorded["function"]["arguments"] = call["function"]["arguments"]
    events[index]["part"]["state"]["input"] = copy.deepcopy(arguments)


class OpenCodeStressTests(unittest.TestCase):
    """Exercise wire preservation and negative evidence without GPUs or models."""

    def test_interrupted_artifact_copy_preserves_live_workspace(self):
        """An ENOSPC exception cannot destroy the model-authored app during unwinding."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, live = [], []
            proxy = Mock()
            proxy.__enter__ = Mock(return_value=("http://127.0.0.1:1234/v1", rows))
            proxy.__exit__ = Mock(return_value=False)
            args = SimpleNamespace(output=root, base_url="http://127.0.0.1:1234",
                request_timeout=None, session_timeout=None, model="test-model",
                context_length=262144, max_tokens=32768, opencode="fixture-client")

            def client(command, *, prompt, workspace, env, output, error, wall_time_limit):
                """Complete one phase before the independent artifact copy fails."""
                live.append(workspace)
                (workspace / "authored.py").write_text('print("retained 🙂")\n')
                rows.append({"request": {"messages": [{"role": "user", "content": prompt.read_text()}]}})
                output.write_text(json.dumps({"type": "step_finish", "part": {"reason": "stop"}}) + "\n")
                error.write_text("")
                return 0, False

            try:
                with patch.object(stress, "recording_proxy", return_value=proxy), \
                     patch.object(stress, "run_client_with_prompt", side_effect=client), \
                     patch.object(stress.shutil, "copytree", side_effect=OSError(errno.ENOSPC, "no space")):
                    with self.assertRaises(OSError) as failure:
                        stress.run_session(args, 0, "number", "123")
                self.assertEqual(failure.exception.errno, errno.ENOSPC)
                gc.collect()
                self.assertEqual(len(live), 1)
                self.assertTrue(live[0].exists(), "exception cleanup deleted the unfinished app")
                self.assertEqual((live[0] / "authored.py").read_text(), 'print("retained 🙂")\n')
                receipt = json.loads((root / "session-0000-number/workspace-location.json").read_text())
                self.assertEqual(receipt["live_workspace"], str(live[0]))
                self.assertFalse(receipt["copied_and_retired"])
            finally:
                for workspace in live:
                    shutil.rmtree(workspace, ignore_errors=True)

    def test_client_deadlines_are_disabled_without_hiding_operator_limits(self):
        """A launched client must not impose its own five-minute header or SSE cap."""
        for limit in (None, 1.5):
            with self.subTest(operator_limit=limit), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                seed = root / "seed"
                seed.mkdir()
                evidence, events = conversation(seed, True)
                rows = []
                proxy = Mock()
                proxy.__enter__ = Mock(return_value=("http://127.0.0.1:1234/v1", rows))
                proxy.__exit__ = Mock(return_value=False)
                args = SimpleNamespace(output=root, base_url="http://127.0.0.1:1234",
                    request_timeout=limit, session_timeout=limit, model="test-model",
                    context_length=262144, max_tokens=32768, opencode="fixture-client")

                def client(command, *, prompt, workspace, env, output, error, wall_time_limit):
                    """Observe the actual launch configuration and publish an authenticated fixture."""
                    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
                    options = config["provider"]["llaminar"]["options"]
                    for key in ("timeout", "headerTimeout", "chunkTimeout"):
                        self.assertIs(options.get(key), False, key)
                    self.assertEqual(wall_time_limit, limit)
                    self.assertEqual(options["baseURL"], "http://127.0.0.1:1234/v1")
                    evidence[0]["request"]["messages"][0]["content"] = prompt.read_text()
                    rows.extend(evidence)
                    (workspace / "result.txt").write_bytes(b"123")
                    output.write_text("\n".join(json.dumps(event) for event in events))
                    error.write_text("")
                    return 0, False

                with patch.object(stress, "recording_proxy", return_value=proxy) as relay, \
                     patch.object(stress, "run_client_with_prompt", side_effect=client):
                    result = stress.run_session(args, 0, "number", "123")
                self.assertTrue(result["passed"], result["errors"])
                relay.assert_called_once_with(args.base_url, root / "session-0000-number", limit,
                                              measure_prefix_reuse=False)
                receipt = json.loads((root / "session-0000-number/workspace-location.json").read_text())
                self.assertTrue(receipt["copied_and_retired"])
                self.assertFalse(Path(receipt["live_workspace"]).exists())
                self.assertEqual((root / "session-0000-number/workspace/result.txt").read_bytes(), b"123")

    def test_prompt_stdin_preserves_quotes_unicode_and_large_inputs(self):
        """A real child receives exact input bytes with no prompt argument or pipe deadlock."""
        prompts = [stress.coding_prompt(case, value) for case, value in stress.fixture_cases().items()
                   if case != "webapp"]
        prompts.extend(stress.session_prompts("webapp", "123"))
        prompts += ['  \\"quoted\\"\r\n\t🙂 👩🏽‍💻 e\u0301\x00end  ', "long input 🙂\n" * 100000]
        child = ("import sys\n"
                 "assert len(sys.argv) == 1, 'prompt leaked into positional arguments'\n"
                 "sys.stdout.buffer.write(sys.stdin.buffer.read())\n")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index, prompt in enumerate(prompts):
                with self.subTest(index=index):
                    source, output, error = (directory / name for name in ("prompt", "stdout", "stderr"))
                    source.write_bytes(prompt.encode("utf-8"))
                    code, interrupted = stress.run_client_with_prompt(
                        [sys.executable, "-c", child], prompt=source, workspace=directory,
                        env=dict(os.environ), output=output, error=error)
                    self.assertEqual(code, 0, error.read_text())
                    self.assertFalse(interrupted)
                    self.assertEqual(output.read_bytes(), source.read_bytes())

    def test_phase_prompt_authenticates_actual_wire_and_rejects_cli_escaping(self):
        """Client argument escaping or a different latest user turn cannot certify a fixture."""
        for case, value in stress.fixture_cases().items():
            for prompt in stress.session_prompts(case, value):
                with self.subTest(case=case, prompt=prompt[:50]):
                    row = {"request": {"messages": [{"role": "user", "content": "old turn"},
                        {"role": "assistant", "content": "done"}, {"role": "user", "content": prompt}]}}
                    for content in (prompt, [{"type": "text", "text": prompt}]):
                        row["request"]["messages"][-1]["content"] = content
                        result = stress.authenticate_phase_prompt([row], prompt)
                        self.assertTrue(result["passed"])
                        self.assertEqual(result["observed_sha256"], result["expected_sha256"])
                    # This is the observed upstream run.ts positional-argument
                    # transformation, not JSON string decoding. The literal
                    # backslashes were present in the captured model request.
                    escaped = '"' + prompt.replace('"', '\\"') + '"'
                    for bad in (escaped, prompt + "\n", "old turn", None, 123,
                                [{"type": "image_url", "image_url": {"url": "fixture"}}],
                                [{"type": "text", "text": prompt}, {"type": "text", "text": "extra"}]):
                        row["request"]["messages"][-1]["content"] = bad
                        result = stress.authenticate_phase_prompt([row], prompt)
                        self.assertFalse(result["passed"])
                        self.assertIn("intended user prompt differs", result["error"])
                    self.assertFalse(stress.authenticate_phase_prompt([], prompt)["passed"])
                    row["request"]["messages"] = []
                    self.assertFalse(stress.authenticate_phase_prompt([row], prompt)["passed"])

    def test_shell_verification_is_allowed_without_replacing_required_file_tools(self):
        """Shell inspection is normal; its success cannot certify absent write/read calls."""
        for case in ("object", "unicode", "workflow", "webapp"):
            self.assertEqual(stress.session_permissions(case)["bash"], "allow")
        for replaced in ("write", "read"):
            with self.subTest(replaced=replaced), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                rows, events = conversation(directory, True)
                index = 0 if replaced == "write" else 1
                arguments = {"command": "wc -c result.txt && od -c result.txt | tail -3",
                             "description": "Inspect exact result bytes"}
                call = {"id": "call_" + replaced, "type": "function", "function": {
                    "name": "bash", "arguments": json.dumps(arguments)}}
                message = {"role": "assistant", "content": None, "tool_calls": [call]}
                rows[index]["response"] = wire(message, True)
                for row in rows:
                    row["request"]["tools"].append({"type": "function", "function": {
                        "name": "bash", "parameters": {"type": "object", "properties": {
                            key: {"type": "string"} for key in arguments}}}})
                for row in rows[index + 1:]:
                    for previous in row["request"]["messages"]:
                        if previous.get("tool_calls", [{}])[0].get("id") == call["id"]:
                            previous["tool_calls"] = [call]
                events[index]["part"].update(tool="bash", state={"status": "completed", "input": arguments})
                (directory / "client.jsonl").write_text("\n".join(json.dumps(event) for event in events))
                result = stress.check_session(directory, rows, 0)
                self.assertFalse(result["passed"])
                self.assertEqual(result["errors"], [f"session omitted required tools: ['{replaced}']"])

    def test_exact_file_requires_matching_completed_write_and_read(self):
        """Final bytes cannot substitute for the required tool's actual arguments."""
        for defect in ("write_content", "write_path", "read_path", "client_content", "client_path",
                       "read_before_write", "absolute"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                rows, events = conversation(directory, True)
                live = directory / "original-live-workspace"
                (directory / "workspace-location.json").write_text(json.dumps({"live_workspace": str(live)}))
                for index, name in enumerate(("write", "read")):
                    arguments = dict(events[index]["part"]["state"]["input"])
                    if defect == "absolute":
                        arguments["filePath"] = str(live / "result.txt")
                    if defect == name + "_path":
                        arguments["filePath"] = str(directory / "elsewhere/result.txt")
                    if name == "write" and defect == "write_content":
                        arguments["content"] = "wrong bytes"
                    call = {"id": "call_" + name, "type": "function", "function": {
                        "name": name, "arguments": json.dumps(arguments)}}
                    rows[index]["response"] = wire({"role": "assistant", "content": None,
                                                   "tool_calls": [call]}, True)
                    for row in rows[index + 1:]:
                        for message in row["request"]["messages"]:
                            if message.get("tool_calls", [{}])[0].get("id") == call["id"]:
                                message["tool_calls"] = [call]
                    events[index]["part"]["state"]["input"] = dict(arguments)
                    if name == "write" and defect == "client_content":
                        events[index]["part"]["state"]["input"]["content"] = "different client bytes"
                    if name == "read" and defect == "client_path":
                        events[index]["part"]["state"]["input"]["filePath"] = "another.txt"
                if defect == "read_before_write":
                    events[0], events[1] = events[1], events[0]
                (directory / "client.jsonl").write_text("\n".join(json.dumps(event) for event in events))
                result = stress.check_session(directory, rows, 0)
                self.assertEqual(result["passed"], defect == "absolute", result)
                if defect != "absolute":
                    self.assertTrue(any("matching completed" in error for error in result["errors"]), result)

    def test_workflow_requires_authenticated_passing_test_execution(self):
        """A bash call or final source bytes cannot prove that the requested tests passed."""
        for defect in (None, "default_workdir", "relative_workdir", "command", "client_command", "workdir", "client_workdir",
                       "exit", "missing_exit", "boolean_exit", "zero_tests", "missing_summary"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "session-0000-workflow"
                directory.mkdir()
                rows, events = conversation(directory, True)
                workspace = directory / "workspace"
                (workspace / "src").mkdir()
                (workspace / "src/answer.py").write_text("def answer():\n    return 123\n")
                terminal = rows.pop()
                tools = copy.deepcopy(terminal["request"]["tools"])
                messages = copy.deepcopy(terminal["request"]["messages"])
                events.pop()
                for name in ("edit", "glob", "grep", "todowrite", "bash"):
                    arguments = ({"command": "python3 -m unittest discover -s tests",
                                  "workdir": str(workspace)} if name == "bash" else {})
                    if name == "bash" and defect == "command":
                        arguments["command"] = "echo tests passed"
                    if name == "bash" and defect == "workdir":
                        arguments["workdir"] = str(Path(temporary) / "elsewhere")
                    if name == "bash" and defect == "default_workdir":
                        del arguments["workdir"]
                    if name == "bash" and defect == "relative_workdir":
                        arguments["workdir"] = "."
                    tools.append({"type": "function", "function": {"name": name,
                        "parameters": {"type": "object", "properties": {
                            key: {"type": "string"} for key in arguments}}}})
                    call = {"id": "call_" + name, "type": "function", "function": {
                        "name": name, "arguments": json.dumps(arguments)}}
                    message = {"role": "assistant", "content": None, "tool_calls": [call]}
                    rows.append({"path": "/v1/chat/completions", "request": {
                        "stream": True, "messages": copy.deepcopy(messages), "tools": tools},
                        "response": wire(message, True)})
                    messages.extend([message, {"role": "tool", "tool_call_id": call["id"], "content": "ok"}])
                    state = {"status": "completed", "input": copy.deepcopy(arguments)}
                    if name == "bash":
                        state.update(metadata={"exit": 0}, output=".\nRan 1 test in 0.001s\n\nOK\n")
                        if defect == "client_command":
                            state["input"]["command"] = "echo tests passed"
                        if defect == "client_workdir":
                            state["input"]["workdir"] = str(Path(temporary) / "elsewhere")
                        if defect == "exit":
                            state["metadata"]["exit"] = 1
                        if defect == "missing_exit":
                            state["metadata"] = {}
                        if defect == "boolean_exit":
                            state["metadata"]["exit"] = False
                        if defect == "zero_tests":
                            state["output"] = "Ran 0 tests in 0.000s\n\nOK\n"
                        if defect == "missing_summary":
                            state["output"] = "tests passed"
                    events.append({"type": "tool_use", "part": {"type": "tool", "tool": name,
                        "callID": call["id"], "state": state}})
                terminal["request"].update(messages=messages, tools=tools)
                rows.append(terminal)
                events.append({"type": "step_finish", "part": {"reason": "stop"}})
                (directory / "client.jsonl").write_text("\n".join(json.dumps(event) for event in events))
                result = stress.check_session(directory, rows, 0)
                valid = defect in (None, "default_workdir", "relative_workdir")
                self.assertEqual(result["passed"], valid, result)
                if not valid:
                    self.assertIn("workflow lacks a matching completed, passing unittest command",
                                  result["errors"])

    def test_buffered_tool_call_waits_without_an_implicit_socket_deadline(self):
        """A quiet upstream retains its raw tool payload; explicit deadlines still apply."""
        start = b'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        end = wire({"role": "assistant", "tool_calls": [{"id": "call_write", "type": "function",
            "function": {"name": "write", "arguments": json.dumps({"filePath": "result.txt",
                "content": "buffered tool output 🙂"})}}]}, True)["body"].encode()

        class Peer(BaseHTTPRequestHandler):
            """Keep generation quiet after the initial role, like a buffered XML call."""
            def do_POST(self):
                """Publish the remainder after a controlled short quiet interval."""
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(start) + len(end)))
                self.end_headers()
                self.wfile.write(start)
                self.wfile.flush()
                time.sleep(0.1)
                try:
                    self.wfile.write(end)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # The explicit-deadline negative control closed its peer.

            def log_message(self, *args):
                """Keep loopback diagnostics in the assertions and raw evidence."""

        with ThreadingHTTPServer(("127.0.0.1", 0), Peer) as upstream:
            worker = threading.Thread(target=upstream.serve_forever, kwargs={"poll_interval": 0.01})
            worker.start()
            try:
                for deadline in (None, 0.01):
                    with self.subTest(deadline=deadline), tempfile.TemporaryDirectory() as temporary:
                        directory = Path(temporary)
                        options = {} if deadline is None else {"timeout": deadline}
                        with stress.recording_proxy(f"http://127.0.0.1:{upstream.server_port}", directory,
                                                    **options) as (url, rows):
                            target = urlsplit(url)
                            client = http.client.HTTPConnection(target.hostname, target.port, timeout=2)
                            client.request("POST", "/v1/chat/completions", b'{"stream":true}')
                            observed = client.getresponse().read()
                            client.close()
                        if deadline is None:
                            self.assertEqual(observed, start + end)
                            self.assertNotIn("error", rows[0])
                        else:
                            self.assertEqual(observed, start)
                            self.assertIn("timed out", rows[0]["error"])
                        self.assertEqual((directory / "exchange-0000.sse").read_bytes(), observed)
            finally:
                upstream.shutdown()
                worker.join()

    def test_proxy_preserves_exact_json_and_fragmented_sse_bytes(self):
        """The recording layer forwards upstream bytes and credentials verbatim."""
        received = []
        output = wire({"role": "assistant", "content": "λ 中文 🙂"}, True)

        class Peer(BaseHTTPRequestHandler):
            """A transport peer whose irregular writes cross JSON/UTF-8 boundaries."""
            def do_POST(self):
                """Record the request and emit the authenticated wire in tiny chunks."""
                received.append((self.path, self.rfile.read(int(self.headers["Content-Length"])),
                                 self.headers.get("Authorization")))
                self.send_response(200)
                self.send_header("Content-Type", output["content_type"])
                self.send_header("Content-Length", str(len(output["body"].encode())))
                self.end_headers()
                raw = output["body"].encode()
                for index in range(0, len(raw), 3):
                    self.wfile.write(raw[index:index + 3])
                    self.wfile.flush()

            def log_message(self, *args):
                """Assertions own diagnostics; suppress routine access output."""

        with ThreadingHTTPServer(("127.0.0.1", 0), Peer) as upstream:
            worker = threading.Thread(target=upstream.serve_forever, kwargs={"poll_interval": 0.01})
            worker.start()
            try:
                with tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    base = f"http://127.0.0.1:{upstream.server_port}"
                    with stress.recording_proxy(base, directory, 2) as (url, rows):
                        for stream in (False, True):
                            raw = json.dumps({"stream": stream, "messages": [{"role": "user",
                                "content": "λ 中文 🙂"}]}, ensure_ascii=False, indent=2).encode()
                            parsed = urlsplit(url)
                            client = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=2)
                            client.request("POST", "/v1/chat/completions", raw,
                                           {"Authorization": "Bearer test-secret"})
                            self.assertEqual(client.getresponse().read(), output["body"].encode())
                            client.close()
                            self.assertEqual(received[-1], ("/v1/chat/completions", raw, "Bearer test-secret"))
                    self.assertEqual(len(rows), 2)
                    for index, row in enumerate(rows):
                        self.assertEqual(row["request_body"].encode(), received[index][1])
                        self.assertEqual(row["response"], output)
                        saved = (directory / f"exchange-{index:04d}.json").read_text()
                        self.assertNotIn("test-secret", saved)
            finally:
                upstream.shutdown()
                worker.join()

    def test_completed_json_and_sse_conversations_pass(self):
        """Both wire forms join the same calls and completed OpenCode events."""
        for stream in (False, True):
            with self.subTest(stream=stream), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                rows, _ = conversation(directory, stream)
                result = stress.check_session(directory, rows, 0)
                self.assertTrue(result["passed"], result["errors"])
                self.assertEqual(len(result["tool_calls"]), 2)

    def test_requested_usage_reaches_wire_and_real_client_accounting(self):
        """Positive wire totals and SDK input counts are both required evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows, events = conversation(directory, True)
            counts = {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168}
            for row in rows:
                message, _ = stress.response_message(row["response"], True)
                row["request"]["stream_options"] = {"include_usage": True}
                row["response"] = wire(message, True, counts)
            events[-1]["part"]["tokens"] = {
                "input": 123, "output": 45, "cache": {"read": 0, "write": 0}}
            def record():
                """Publish the controlled SDK observation without altering the wire."""
                (directory / "client.jsonl").write_text("\n".join(json.dumps(event) for event in events))
            record()
            result = stress.check_session(directory, rows, 0)
            self.assertTrue(result["passed"], result["errors"])
            self.assertEqual(result["streaming_usage"], [counts] * 3)
            for tokens in ({}, {"input": 0, "cache": {"read": 0, "write": 0}}):
                events[-1]["part"]["tokens"] = tokens
                record()
                result = stress.check_session(directory, rows, 0)
                self.assertFalse(result["passed"])
                self.assertTrue(any("SDK reported zero input tokens" in error for error in result["errors"]))
            events[-1]["part"]["tokens"] = {"input": 0, "cache": {"read": 123, "write": 0}}
            record()
            self.assertTrue(stress.check_session(directory, rows, 0)["passed"])

    def test_requested_usage_rejects_missing_corrupt_and_misordered_tails(self):
        """Malformed terminal accounting cannot masquerade as a completed stream."""
        counts = {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168}
        response = wire({"role": "assistant", "content": "🙂"}, True, counts)
        original = [json.loads(line[5:]) for line in response["body"].splitlines()
                    if line.startswith("data:") and line[5:].strip() != "[DONE]"]
        for mutation in ("missing", "duplicate", "early", "choice_usage", "missing_null", "identity",
                         "missing_identity", "bool", "negative", "zero_prompt", "wrong_sum",
                         "missing_count", "missing_choices", "after_done", "missing_done", "late_choice"):
            with self.subTest(mutation=mutation):
                chunks = copy.deepcopy(original)
                done = "data: [DONE]\n\n"
                if mutation == "missing":
                    chunks.pop()
                elif mutation == "duplicate":
                    chunks.append(copy.deepcopy(chunks[-1]))
                elif mutation == "early":
                    chunks.insert(0, chunks.pop())
                elif mutation == "choice_usage":
                    chunks[0]["usage"] = counts
                elif mutation == "missing_null":
                    del chunks[0]["usage"]
                elif mutation == "identity":
                    chunks[-1]["id"] = "changed"
                elif mutation == "missing_identity":
                    for chunk in chunks:
                        del chunk["model"]
                elif mutation == "bool":
                    chunks[-1]["usage"]["completion_tokens"] = True
                elif mutation == "negative":
                    chunks[-1]["usage"]["completion_tokens"] = -1
                elif mutation == "zero_prompt":
                    chunks[-1]["usage"].update(prompt_tokens=0, total_tokens=45)
                elif mutation == "wrong_sum":
                    chunks[-1]["usage"]["total_tokens"] = 167
                elif mutation == "missing_count":
                    del chunks[-1]["usage"]["completion_tokens"]
                elif mutation == "missing_choices":
                    del chunks[-1]["choices"]
                elif mutation == "after_done":
                    done += "data: " + json.dumps(chunks[-1]) + "\n\n"
                elif mutation == "missing_done":
                    done = ""
                elif mutation == "late_choice":
                    chunks.insert(-1, copy.deepcopy(chunks[1]))
                body = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + done
                exchange = {"request": {"stream": True, "stream_options": {"include_usage": True}},
                            "response": {**response, "body": body}}
                with self.assertRaises(ValueError):
                    stress.requested_streaming_usage(exchange)
        self.assertIsNone(stress.requested_streaming_usage({"request": {"stream": True}}))

    def test_compacted_history_keeps_the_immediate_tool_join_authoritative(self):
        """Consumed older calls may be pruned without losing the next continuation."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows, _ = conversation(directory)
            rows[-1]["request"]["messages"] = rows[-1]["request"]["messages"][-2:]
            result = stress.check_session(directory, rows, 0)
            self.assertTrue(result["passed"], result["errors"])
            rows[-1]["request"]["messages"] = []
            self.assertFalse(stress.check_session(directory, rows, 0)["passed"])

    def test_compaction_request_preserves_pending_tool_until_coding_continues(self):
        """Auxiliary summaries leave the latest tool join for the resumed coding request."""
        for stream in (False, True):
            for count in (1, 2):
                for explicit_empty in (False, True):
                    with self.subTest(stream=stream, summaries=count, explicit_empty=explicit_empty), \
                            tempfile.TemporaryDirectory() as temporary:
                        directory = Path(temporary)
                        rows, _ = conversation(directory, stream)
                        request = {"stream": stream, "messages": [
                            {"role": "system", "content": "You are a context summarization agent.\n"},
                            {"role": "user", "content": "Earlier history:\n<conversation>\n"
                             "[User]: Keep exact Unicode 🙂 and the current task.\n</conversation>\nSummarize."}]}
                        if explicit_empty:
                            request["tools"] = []
                        summary = {"path": "/v1/chat/completions", "request": request,
                                   "response": wire({"role": "assistant", "content": "Earlier work summarized."}, stream)}
                        # Real OpenCode compacts the older head, then reattaches
                        # the immediate assistant/tool pair as its retained tail.
                        rows[-1]["request"]["messages"] = rows[-1]["request"]["messages"][-2:]
                        rows[2:2] = [copy.deepcopy(summary) for _ in range(count)]
                        result = stress.check_session(directory, rows, 0)
                        self.assertTrue(result["passed"], result["errors"])
                        self.assertEqual(result["compaction_requests"], count)
                        self.assertEqual(result["request_count"], 3 + count)
                        self.assertEqual(len(result["tool_calls"]), 2)

    def test_compaction_cannot_erase_or_mutate_pending_tool_evidence(self):
        """Missing continuations, malformed summaries and changed identities stay fatal."""
        for mutation in ("missing_history", "changed_arguments", "changed_client_input", "no_continuation",
                         "wrong_system", "missing_transcript", "advertised_tools", "summary_tool_call"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                rows, events = conversation(directory, True)
                summary = {"path": "/v1/chat/completions", "request": {"stream": True, "messages": [
                    {"role": "system", "content": "You are a context summarization agent.\n"},
                    {"role": "user", "content": "Earlier history:\n<conversation>\nOlder work.\n</conversation>\nSummarize."}]},
                    "response": wire({"role": "assistant", "content": "Earlier work summarized."}, True)}
                if mutation == "missing_history":
                    rows[-1]["request"]["messages"] = []
                elif mutation == "changed_arguments":
                    rows[-1]["request"]["messages"][-2]["tool_calls"][0]["function"]["arguments"] = '{"filePath":"wrong.txt"}'
                elif mutation == "changed_client_input":
                    events[1]["part"]["state"]["input"] = {"filePath": "wrong.txt"}
                    (directory / "client.jsonl").write_text("\n".join(json.dumps(event) for event in events))
                elif mutation == "no_continuation":
                    rows.pop()
                elif mutation == "wrong_system":
                    summary["request"]["messages"][0]["content"] = "An unrelated assistant."
                elif mutation == "missing_transcript":
                    summary["request"]["messages"][1]["content"] = "No recorded conversation."
                elif mutation == "advertised_tools":
                    summary["request"]["tools"] = copy.deepcopy(rows[0]["request"]["tools"])
                elif mutation == "summary_tool_call":
                    summary["response"] = copy.deepcopy(rows[0]["response"])
                rows.insert(2, summary)
                result = stress.check_session(directory, rows, 0)
                self.assertFalse(result["protocol_passed"], result)
                self.assertTrue(result["protocol_errors"])

    def test_readiness_is_required_before_a_client_session(self):
        """An unready endpoint fails admission immediately, without client retries."""
        replies = []
        status = 200
        class Peer(BaseHTTPRequestHandler):
            """Publish a controlled readiness response and count observations."""
            def do_GET(self):
                """Respond once with the selected HTTP readiness status."""
                replies.append(self.path)
                self.send_response(status)
                self.send_header("Content-Length", "15")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            def log_message(self, *args):
                """Keep loopback diagnostics in assertions."""
        with ThreadingHTTPServer(("127.0.0.1", 0), Peer) as server:
            worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
            worker.start()
            try:
                endpoint = f"http://127.0.0.1:{server.server_port}"
                stress.require_ready(endpoint, 1)
                status = 503
                with self.assertRaises(OSError):
                    stress.require_ready(endpoint, 1)
                self.assertEqual(replies, ["/health", "/health"])
            finally:
                server.shutdown()
                worker.join()

    def test_repaired_tool_error_cannot_be_hidden_by_successful_files(self):
        """A later repair never erases the original SDK validation failure."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows, events = conversation(directory)
            events.insert(0, {"type": "tool_use", "part": {"type": "tool",
                "state": {"status": "error", "error": "Expected string, got 123"}}})
            (directory / "client.jsonl").write_text("\n".join(json.dumps(event) for event in events))
            result = stress.check_session(directory, rows, 0)
            self.assertFalse(result["passed"])
            self.assertFalse(result["protocol_passed"])
            self.assertTrue(any("Expected string" in error for error in result["errors"]))

    def test_coding_edit_conflict_requires_exact_authenticated_arguments(self):
        """A missing edit target is task quality only after exact argument authentication."""
        self.assert_coding_edit_conflict(
            "Could not find oldString in the file. It must match exactly, including whitespace, indentation, and line endings.",
            "edit_match_not_found")

    def test_ambiguous_edit_requires_exact_authenticated_arguments(self):
        """The native ambiguous-match error remains a task failure without hiding corruption."""
        self.assert_coding_edit_conflict(
            "Found multiple matches for oldString. Provide more surrounding context to make the match unique.",
            "edit_match_ambiguous")

    def test_unchanged_edit_requires_exact_authenticated_arguments(self):
        """Only actual unchanged edits may explain the native no-change domain error."""
        for replacement in ("same🙂\r\n", "same🙂\n"):
            self.assert_coding_edit_conflict(
                "No changes to apply: oldString and newString are identical.",
                "edit_no_change", {"filePath": "app.py", "oldString": "same🙂\r\n", "newString": replacement})

    def test_glob_execution_error_requires_exact_authenticated_arguments(self):
        """Client subprocess failure remains failed work without masking wire corruption."""
        for mutation in (None, "default_path", "input", "identity", "tool", "pattern", "path", "missing", "message"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                rows, events = conversation(directory)
                arguments = {"pattern": "*🙂*", "path": "/tmp/missing-search-directory"}
                if mutation == "pattern":
                    arguments["pattern"] = 42
                elif mutation == "path":
                    arguments["path"] = 42
                elif mutation == "missing":
                    arguments.pop("pattern")
                elif mutation == "default_path":
                    arguments.pop("path")
                call = {"id": "call_glob", "type": "function", "function": {
                    "name": "glob", "arguments": json.dumps(arguments)}}
                message = {"role": "assistant", "content": None, "tool_calls": [call]}
                conflict = "ripgrep execution failed"
                rows.insert(0, {"path": "/v1/chat/completions", "request": {
                    "messages": [{"role": "user", "content": "Find the result file"}], "tools": [{
                        "type": "function", "function": {"name": "glob", "parameters": {
                            "type": "object", "required": ["pattern"], "properties": {
                                "pattern": {"type": "string"}, "path": {"type": "string"}}}}}]},
                    "response": wire(message)})
                rows[1]["request"]["messages"].extend([
                    message, {"role": "tool", "tool_call_id": "call_glob", "content": conflict}])
                event = {"type": "tool_use", "part": {"type": "tool", "tool": "glob",
                    "callID": "call_glob", "state": {"status": "error", "input": copy.deepcopy(arguments),
                        "error": conflict}}}
                if mutation == "input":
                    event["part"]["state"]["input"]["pattern"] = "changed"
                elif mutation == "identity":
                    event["part"]["callID"] = "unknown"
                elif mutation == "tool":
                    event["part"]["tool"] = "read"
                elif mutation == "message":
                    event["part"]["state"]["error"] += ": SDK validation failed"
                events.insert(0, event)
                (directory / "client.jsonl").write_text("\n".join(json.dumps(value) for value in events))
                result = stress.check_session(directory, rows, 0)
                recognized = mutation in (None, "default_path")
                self.assertFalse(result["passed"])
                self.assertEqual(result["protocol_passed"], recognized, result["errors"])
                self.assertEqual(len(result["execution_errors"]), int(recognized))
                if recognized:
                    self.assertFalse(result["task_passed"])
                    self.assertEqual(result["execution_errors"][0]["kind"], "glob_execution_failed")

    def test_grep_regex_error_requires_exact_authenticated_pattern(self):
        """Native invalid escapes remain failed tasks; changed input and unknown errors fail protocol."""
        pattern = r"1\N{fe00}\u20e3|KEYCAP_ONE|\U000fe00"
        conflict = ("rg: regex parse error:\n    (?:" + pattern
                    + ")\n        ^^\nerror: unrecognized escape sequence")
        mutations = (None, "optional_fields", "input", "identity", "tool", "pattern_type",
                     "missing_pattern", "path_type", "include_type", "sdk", "echo", "suffix",
                     "caret_missing", "caret_outside", "wrong_reason", "wrong_prefix", "multiline")
        for stream in (False, True):
            for mutation in mutations:
                with self.subTest(stream=stream, mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    rows, events = conversation(directory, stream)
                    arguments = {"pattern": pattern}
                    if mutation == "optional_fields":
                        arguments.update(path=".", include="tests.py")
                    elif mutation == "pattern_type":
                        arguments["pattern"] = 42
                    elif mutation == "missing_pattern":
                        arguments.pop("pattern")
                    elif mutation in ("path_type", "include_type"):
                        arguments[mutation.removesuffix("_type")] = 42
                    elif mutation == "multiline":
                        arguments["pattern"] += "\n"
                    call = {"id": "call_grep", "type": "function", "function": {
                        "name": "grep", "arguments": json.dumps(arguments)}}
                    message = {"role": "assistant", "content": None, "tool_calls": [call]}
                    rows.insert(0, {"path": "/v1/chat/completions", "request": {
                        "stream": stream, "messages": [{"role": "user", "content": "Find the emoji test"}],
                        "tools": [{"type": "function", "function": {"name": "grep", "parameters": {
                            "type": "object", "required": ["pattern"], "properties": {
                                key: {"type": "string"} for key in ("pattern", "path", "include")}}}}]},
                        "response": wire(message, stream)})
                    rows[1]["request"]["messages"].extend([
                        message, {"role": "tool", "tool_call_id": "call_grep", "content": conflict}])
                    event = {"type": "tool_use", "part": {"type": "tool", "tool": "grep",
                        "callID": "call_grep", "state": {"status": "error", "input": copy.deepcopy(arguments),
                            "error": conflict}}}
                    if mutation == "input":
                        event["part"]["state"]["input"]["pattern"] = "changed"
                    elif mutation == "identity":
                        event["part"]["callID"] = "unknown"
                    elif mutation == "tool":
                        event["part"]["tool"] = "glob"
                    errors = {
                        "sdk": "Expected string, got 123",
                        "echo": conflict.replace("KEYCAP_ONE", "CHANGED"),
                        "suffix": conflict + "\nSDK validation failed",
                        "caret_missing": conflict.replace("^^", ""),
                        "caret_outside": conflict.replace("        ^^", " " * 100 + "^^"),
                        "wrong_reason": conflict.replace("unrecognized escape sequence", "SDK validation failed"),
                        "wrong_prefix": conflict.replace("rg: regex parse error:", "SDK regex parse error:"),
                    }
                    if mutation in errors:
                        event["part"]["state"]["error"] = errors[mutation]
                    events.insert(0, event)
                    (directory / "client.jsonl").write_text("\n".join(json.dumps(value) for value in events))
                    result = stress.check_session(directory, rows, 0)
                    recognized = mutation in (None, "optional_fields")
                    self.assertFalse(result["passed"])
                    self.assertEqual(result["protocol_passed"], recognized, result["errors"])
                    self.assertEqual(len(result["execution_errors"]), int(recognized))
                    self.assertNotIn("grep", result["successful_tools"])
                    if recognized:
                        self.assertFalse(result["task_passed"])
                        self.assertEqual(result["execution_errors"][0]["kind"], "grep_invalid_regex")

    def test_native_grep_escape_evidence_rejects_changed_backslashes(self):
        """The independent native argument join distinguishes literal escapes from Unicode conversion."""
        pattern = r"1\N{fe00}\u20e3|KEYCAP_ONE|\U000fe00"
        tools = [{"function": {"name": "grep", "parameters": {"required": ["pattern"],
            "properties": {"pattern": {"type": "string"}, "include": {"type": "string"}}}}}]
        native = ("<tool_call>\n<function=grep>\n<parameter=include>\ntests.py\n</parameter>\n"
                  "<parameter=pattern>\n" + pattern + "\n</parameter>\n</function>\n</tool_call>")
        for value in (pattern, pattern.replace("\\u20e3", "\u20e3"), pattern.replace("\\", "\\\\")):
            with self.subTest(pattern=value):
                result = stress.authenticate_native_qwen_calls(native,
                    [{"name": "grep", "arguments": {"pattern": value, "include": "tests.py"}}], tools)
                self.assertEqual(result["passed"], value == pattern)

    def assert_coding_edit_conflict(self, conflict, expected_kind, base_arguments=None):
        """Join exact wire/client inputs; reject unknown errors, SDK failures and missing fields."""
        mutations = (None, "sdk", "input", "identity", "tool", "argument", "missing", "message")
        if expected_kind == "edit_no_change":
            mutations += ("distinct_edit", "empty_path")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                rows, events = conversation(directory)
                arguments = copy.deepcopy(base_arguments) if base_arguments else {
                    "filePath": "app.py", "oldString": "absent", "newString": "replacement"}
                if mutation == "argument":
                    arguments["oldString"] = 123
                elif mutation == "missing":
                    arguments.pop("oldString")
                elif mutation == "distinct_edit":
                    arguments["newString"] = "a real change"
                elif mutation == "empty_path":
                    arguments["filePath"] = ""
                call = {"id": "call_edit", "type": "function", "function": {
                    "name": "edit", "arguments": json.dumps(arguments)}}
                message = {"role": "assistant", "content": None, "tool_calls": [call]}
                rows.insert(0, {"path": "/v1/chat/completions", "request": {
                    "messages": [{"role": "user", "content": "Edit app.py"}], "tools": [{
                        "type": "function", "function": {"name": "edit", "parameters": {
                            "type": "object", "properties": {key: {"type": "string"} for key in arguments}}}}]},
                    "response": wire(message)})
                rows[1]["request"]["messages"].extend([
                    message, {"role": "tool", "tool_call_id": "call_edit", "content": conflict}])
                error = {"type": "tool_use", "part": {"type": "tool", "tool": "edit",
                    "callID": "call_edit", "state": {"status": "error", "input": copy.deepcopy(arguments),
                        "error": conflict}}}
                if mutation == "sdk":
                    error["part"]["state"]["error"] = "Expected string, got 123"
                elif mutation == "message":
                    error["part"]["state"]["error"] = conflict + " SDK validation failed."
                elif mutation == "input":
                    error["part"]["state"]["input"]["oldString"] = "different"
                elif mutation == "identity":
                    error["part"]["callID"] = "unknown"
                elif mutation == "tool":
                    error["part"]["tool"] = "write"
                events.insert(0, error)
                (directory / "client.jsonl").write_text("\n".join(json.dumps(event) for event in events))
                result = stress.check_session(directory, rows, 0)
                self.assertFalse(result["passed"])
                self.assertEqual(result["protocol_passed"], mutation is None, result["errors"])
                self.assertEqual(len(result["execution_errors"]), int(mutation is None))
                if mutation is None:
                    self.assertFalse(result["task_passed"])
                    self.assertEqual(result["execution_errors"][0]["kind"], expected_kind)

    def test_task_quality_is_separate_from_tool_transport(self):
        """Wrong model text/path remains a task failure when the client received it exactly."""
        for stream in (False, True):
            for mistake in ("content", "path", "final_bytes"):
                with self.subTest(stream=stream, mistake=mistake), tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    rows, events = conversation(directory, stream)
                    if mistake == "content":
                        amend_call(rows, events, 0, {"filePath": "result.txt", "content": "123🙂\n"})
                        (directory / "workspace/result.txt").write_text("123🙂\n")
                    elif mistake == "path":
                        amend_call(rows, events, 0, {"filePath": "different.txt", "content": "123"})
                        amend_call(rows, events, 1, {"filePath": "different.txt"})
                        (directory / "workspace/result.txt").rename(directory / "workspace/different.txt")
                    else:
                        (directory / "workspace/result.txt").write_bytes(b"123\n")
                    (directory / "client.jsonl").write_text("\n".join(json.dumps(event) for event in events))
                    result = stress.check_session(directory, rows, 0)
                    self.assertTrue(result["protocol_passed"], result["errors"])
                    self.assertFalse(result["task_passed"])
                    self.assertFalse(result["passed"])
                    self.assertEqual(result["successful_tools"], ["read", "write"])
                    result["case"] = "number"
                    self.assertTrue(stress.campaign_assessment([result], ["number"], 1, "protocol")["gate_passed"])
                    self.assertFalse(stress.campaign_assessment([result], ["number"], 1, "strict")["gate_passed"])

    def test_native_argument_evidence_matches_complete_values_and_order(self):
        """An independent inverse rendering proves literal tags, emoji and typed JSON survive."""
        arguments = {"content": 'print("<tool_call></tool_call></parameter><parameter=filePath>")\n🙂',
                     "filePath": "/tmp/example/result.txt", "options": {"enabled": True, "limit": 2}}
        tools = [{"function": {"name": "write", "parameters": {"properties": {
            "content": {"type": "string"}, "filePath": {"type": "string"}, "options": {"type": "object"}}}}}]
        for newline in ("", "\n", "\r\n"):
            for order in (tuple(arguments), tuple(reversed(arguments))):
                with self.subTest(newline=newline, order=order):
                    body = "<tool_call>\n<function=write>\n"
                    for key in order:
                        value = arguments[key] if key != "options" else json.dumps(arguments[key], indent=2)
                        body += f"<parameter={key}>" + newline + value + newline + "</parameter>\n"
                    body += "</function>\n</tool_call>"
                    call = {"name": "write", "arguments": arguments}
                    result = stress.authenticate_native_qwen_calls(body + "\n" + body, [call, call], tools)
                    self.assertTrue(result["passed"], result)
                    self.assertEqual(result["verified_calls"], 2)
                    self.assertLessEqual(result["spans"][0]["end"], result["spans"][1]["begin"])

    def test_native_argument_evidence_rejects_parser_corruption(self):
        """A parser cannot certify itself by repeating a swallowed, clipped or mistyped argument."""
        tools = [{"function": {"name": "write", "parameters": {"required": ["content", "filePath"], "properties": {
            "content": {"type": "string"}, "filePath": {"type": "string"}}}}}]
        native = ("<tool_call>\n<function=write>\n<parameter=content>\n123🙂</parameter>\n"
                  "<parameter=filePath>\n/tmp/result.txt\n</parameter>\n</function>\n</tool_call>")
        for mutation in ("swallowed", "missing", "clipped", "type", "path", "extra", "duplicate_call", "unclosed"):
            with self.subTest(mutation=mutation):
                arguments = {"content": "123🙂", "filePath": "/tmp/result.txt"}
                content = native
                if mutation == "swallowed":
                    arguments = {"content": "123🙂</parameter>\n<parameter=filePath>\n/tmp/result.txt"}
                elif mutation == "missing":
                    arguments.pop("filePath")
                elif mutation == "clipped":
                    arguments["content"] = "123"
                elif mutation == "type":
                    arguments["content"] = 123
                elif mutation == "path":
                    arguments["filePath"] = "/tmp/elsewhere.txt"
                elif mutation == "extra":
                    arguments["unknown"] = "extra"
                elif mutation == "unclosed":
                    content = content.removesuffix("</tool_call>")
                calls = [{"name": "write", "arguments": arguments}] * (2 if mutation == "duplicate_call" else 1)
                result = stress.authenticate_native_qwen_calls(content, calls, tools)
                self.assertFalse(result["passed"], result)

    def test_domain_file_errors_require_matching_inputs_and_actual_error_kind(self):
        """Known path mistakes are observable task results; schema and identity defects stay fatal."""
        for domain in ("missing", "denied"):
            for mutation in (None, "input", "identity", "tool", "schema", "required", "unknown", "internal", "rule"):
                with self.subTest(domain=domain, mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    rows, events = conversation(directory, True)
                    index = 1 if domain == "missing" else 0
                    path = str(directory / ("workspace/absent.txt" if domain == "missing" else "outside.txt"))
                    if mutation == "internal":
                        path = str(directory / "workspace/inside.txt")
                    arguments = {"filePath": path, **({"content": "123"} if index == 0 else {})}
                    if mutation == "required":
                        arguments.pop("filePath")
                    amend_call(rows, events, index, arguments)
                    state = events[index]["part"]["state"]
                    state["status"] = "error"
                    rules = [{"permission": "external_directory", "action": "deny", "pattern": "*"}]
                    if mutation == "rule":
                        rules[0]["action"] = "allow"
                    state["error"] = ("File not found: " + path if domain == "missing" else
                        "The user has specified a rule which prevents you from using this specific tool call. "
                        "Here are some of the relevant rules " + json.dumps(rules))
                    if mutation == "input":
                        state["input"]["filePath"] = "different.txt"
                    elif mutation == "identity":
                        events[index]["part"]["callID"] = "unknown"
                    elif mutation == "tool":
                        events[index]["part"]["tool"] = "edit"
                    elif mutation == "schema":
                        state["error"] = 'The write tool was called with invalid arguments: SchemaError(Missing key at ["filePath"]).'
                    elif mutation == "unknown":
                        state["error"] = "Unexpected client failure"
                    (directory / "client.jsonl").write_text("\n".join(json.dumps(event) for event in events))
                    result = stress.check_session(directory, rows, 0)
                    recognized = mutation in (None, "internal") or (domain == "missing" and mutation == "rule")
                    self.assertEqual(result["protocol_passed"], recognized, result["errors"])
                    self.assertFalse(result["passed"])
                    self.assertEqual(len(result["execution_errors"]), int(recognized))

    def test_protocol_gate_requires_successful_coverage_for_every_planned_case(self):
        """Missing work and unexercised tools cannot turn a low task score into qualification."""
        good = {**stress.assessment([], []), "case": "number", "successful_tools": ["write", "read"]}
        bad_task = {**stress.assessment([], ["wrong bytes"]), "case": "unicode", "successful_tools": ["write"]}
        for mutation in ("coverage", "missing_session", "protocol", "case", "complete"):
            with self.subTest(mutation=mutation):
                rows = [copy.deepcopy(good), copy.deepcopy(bad_task)]
                if mutation != "coverage":
                    rows[1]["successful_tools"].append("read")
                if mutation == "missing_session":
                    rows.pop()
                elif mutation == "protocol":
                    rows[1].update(stress.assessment(["SSE omitted DONE"], ["wrong bytes"]))
                elif mutation == "case":
                    rows[1]["case"] = "number"
                result = stress.campaign_assessment(rows, ["number", "unicode"], 1, "protocol")
                self.assertEqual(result["gate_passed"], mutation == "complete", result)
                self.assertFalse(result["passed"])
                self.assertFalse(result["task_passed"])
        self.assertFalse(stress.campaign_assessment([], ["number"], 1, "protocol")["gate_passed"])

    def test_bash_missing_workdir_requires_exact_authenticated_directory(self):
        """Completed missing-workdir errors fail the task without certifying Bash coverage.

        OpenCode 1.18.34 resolves relative workdirs against the session workspace
        and drops trailing separators in its FileSystem.access diagnostic. The
        wire, continuation and client input must still agree byte for byte;
        normalization is permitted only when authenticating the error's path.
        No filesystem probe can establish whether a historical path existed.
        """
        mutations = (None, "input", "identity", "tool", "command_type", "missing_command",
                     "workdir_type", "missing_workdir", "empty_workdir", "wrong_path",
                     "suffix", "prefix", "operation", "reason", "history", "continuation")
        for stream in (False, True):
            for path_kind in ("relative", "absolute", "trailing_separator"):
                for mutation in mutations:
                    with self.subTest(stream=stream, path=path_kind, mutation=mutation), \
                            tempfile.TemporaryDirectory() as temporary:
                        directory = Path(temporary)
                        rows, events = conversation(directory, stream)
                        workspace = directory / "workspace"
                        workdir = ("./missing-🙂" if path_kind == "relative"
                                   else str(workspace / "missing-🙂"))
                        if path_kind == "trailing_separator":
                            workdir += "/"
                        arguments = {"command": "pwd", "workdir": workdir}
                        conflict = "NotFound: FileSystem.access (" + os.path.normpath(
                            os.path.join(workspace, workdir)) + ")"
                        if mutation in ("command_type", "workdir_type"):
                            arguments[mutation.removesuffix("_type")] = 123
                        elif mutation in ("missing_command", "missing_workdir"):
                            arguments.pop(mutation.removeprefix("missing_"))
                        elif mutation == "empty_workdir":
                            arguments["workdir"] = ""
                        call = {"id": "call_bash", "type": "function", "function": {
                            "name": "bash", "arguments": json.dumps(arguments)}}
                        message = {"role": "assistant", "content": None, "tool_calls": [call]}
                        rows.insert(0, {"path": "/v1/chat/completions", "request": {
                            "stream": stream, "messages": [{"role": "user", "content": "Check the directory"}],
                            "tools": [{"type": "function", "function": {"name": "bash", "parameters": {
                                "type": "object", "required": ["command"], "properties": {
                                    key: {"type": "string"} for key in ("command", "workdir")}}}}]},
                            "response": wire(message, stream)})
                        rows[1]["request"]["messages"].extend([
                            copy.deepcopy(message), {"role": "tool", "tool_call_id": "call_bash", "content": conflict}])
                        event = {"type": "tool_use", "part": {"type": "tool", "tool": "bash",
                            "callID": "call_bash", "state": {"status": "error",
                                "input": copy.deepcopy(arguments), "error": conflict}}}
                        if mutation == "input":
                            event["part"]["state"]["input"]["workdir"] = "changed"
                        elif mutation == "identity":
                            event["part"]["callID"] = "unknown"
                        elif mutation == "tool":
                            event["part"]["tool"] = "read"
                        elif mutation == "history":
                            rows[1]["request"]["messages"][-2]["tool_calls"][0]["function"]["arguments"] = "{}"
                        elif mutation == "continuation":
                            rows[1]["request"]["messages"].pop()
                        diagnostics = {
                            "wrong_path": conflict.replace("missing-🙂", "different"),
                            "suffix": conflict + " SDK failure",
                            "prefix": "SDK " + conflict,
                            "operation": conflict.replace("FileSystem.access", "FileSystem.stat"),
                            "reason": conflict.replace("NotFound", "PermissionDenied"),
                        }
                        if mutation in diagnostics:
                            event["part"]["state"]["error"] = diagnostics[mutation]
                        events.insert(0, event)
                        (directory / "client.jsonl").write_text("\n".join(json.dumps(value) for value in events))
                        result = stress.check_session(directory, rows, 0)
                        self.assertFalse(result["passed"])
                        self.assertEqual(result["protocol_passed"], mutation is None, result["errors"])
                        self.assertNotIn("bash", result["successful_tools"])
                        if mutation is None:
                            self.assertFalse(result["task_passed"])
                            self.assertEqual(len(result["execution_errors"]), 1)
                            self.assertEqual(result["execution_errors"][0]["kind"], "bash_workdir_not_found")

    def test_native_bash_workdir_evidence_rejects_changed_paths(self):
        """A matching error cannot excuse native-to-wire path normalization or corruption."""
        arguments = {"command": "pwd", "workdir": "./missing-🙂/"}
        tools = [{"function": {"name": "bash", "parameters": {"required": ["command"],
            "properties": {key: {"type": "string"} for key in arguments}}}}]
        native = ("<tool_call>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n"
                  "<parameter=workdir>\n./missing-🙂/\n</parameter>\n</function>\n</tool_call>")
        for path in (arguments["workdir"], "./missing-🙂", "/tmp/missing-🙂/", "./missing-?/", 123):
            with self.subTest(path=path):
                result = stress.authenticate_native_qwen_calls(native,
                    [{"name": "bash", "arguments": {**arguments, "workdir": path}}], tools)
                self.assertEqual(result["passed"], path == arguments["workdir"])

    def test_cli_protocol_gate_preserves_failed_task_score_and_fatal_protocol_exit(self):
        """The real CLI selector changes only its exit criterion, retaining both report outcomes."""
        for gate in ("strict", "protocol"):
            for failure in ("task", "protocol", "coverage"):
                with self.subTest(gate=gate, failure=failure), tempfile.TemporaryDirectory() as temporary:
                    output = Path(temporary) / "campaign"
                    result = {**stress.assessment(["invalid arguments"] if failure == "protocol" else [],
                                                 ["wrong file contents"]),
                              "case": "webapp", "successful_tools": sorted(stress.required_tools("webapp"))}
                    if failure == "coverage":
                        result["successful_tools"] = ["write"]
                    command = ["opencode_tool_stress.py", "--base-url", "http://127.0.0.1:1",
                               "--model", "fixture", "--output", str(output), "--case", "webapp", "--context-length", "65536",
                               "--iterations", "1", "--gate", gate]
                    with patch.object(sys, "argv", command), patch.object(stress, "require_ready"), \
                         patch.object(stress.subprocess, "check_output", return_value="1.18.34"), \
                         patch.object(stress, "run_session", return_value=result):
                        code = stress.main()
                    document = json.loads((output / "stress.json").read_text())
                    self.assertEqual(code == 0, gate == "protocol" and failure == "task")
                    self.assertEqual(document["gate_passed"], code == 0)
                    self.assertFalse(document["passed"])
                    self.assertFalse(document["task_passed"])
                    self.assertEqual(document["protocol_passed"], failure != "protocol")
                    self.assertEqual(document["results"][0]["task_errors"], ["wrong file contents"])

    def test_live_stream_observation_preserves_unicode_counts_and_repetition_signals(self):
        """Live output distinguishes SSE fragments, terminal usage and repeated text."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "progress.json"
            now = [100.0]
            observation = stress.StreamObservation(path, clock=lambda: now[0])
            chunks = [
                {"choices": [{"delta": {"reasoning_content": "Check the tests. " * 6}}]},
                {"choices": [{"delta": {"content": "🧪 Ready"}}]},
                {"choices": [{"delta": {"tool_calls": [{"function": {"name": "write", "arguments": '{"content":"🙂"}'}}]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
                {"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 20, "total_tokens": 1020}},
            ]
            for chunk in chunks:
                now[0] += 2
                observation.record(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode())
            observation.record(b"data: [DONE]\n\n")
            result = json.loads(path.read_text())
            self.assertTrue(result["complete"])
            self.assertEqual(result["delta_count"], 3)
            self.assertEqual(result["usage"]["completion_tokens"], 20)
            self.assertEqual(result["characters"]["content"], 7)
            self.assertEqual(result["tool_names"], ["write"])
            self.assertEqual(result["tail"]["content"], "🧪 Ready")
            self.assertGreaterEqual(result["repetition_signals"]["reasoning_content"]["repeats"], 6)
            self.assertEqual(stress.repeated_suffix(" " * 400)["repeats"], 0)
            self.assertEqual(stress.repeated_suffix("Write a test, run it, inspect the failure, fix it.")["repeats"], 0)

    def test_live_observation_does_not_interrupt_malformed_protocol_evidence(self):
        """Wrong JSON shapes survive observation for independent wire validation."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "progress.json"
            observation = stress.StreamObservation(path)
            malformed = [None, [], 3, "text", {"choices": None}, {"choices": {}},
                {"choices": [None, [], 3, {"delta": []}, {"delta": "bad"}]},
                {"choices": [{"delta": {"content": 3, "reasoning_content": [], "tool_calls": None}}]},
                {"choices": [{"delta": {"tool_calls": [None, 3, {"function": "bad"},
                    {"function": {"name": [], "arguments": {}}}]}},
                    {"delta": {"content": "still observing 🧪"}, "finish_reason": []}]},
                {"usage": "bad", "choices": []}]
            for event in malformed:
                observation.record(("data: " + json.dumps(event) + "\n").encode())
            observation.record(b"data: not-json\n")
            observation.record(b"data: [DONE]\n")
            result = json.loads(path.read_text())
            self.assertTrue(result["complete"])
            self.assertEqual(result["tail"]["content"], "still observing 🧪")
            self.assertEqual(result["delta_count"], 1)
            self.assertIsNone(result["usage"])

    def test_active_client_has_no_default_turn_deadline(self):
        """Long coding and reasoning turns finish on the client's own terminal event."""
        process = Mock()
        process.wait.return_value = 0
        self.assertEqual(stress.wait_for_client(process), (0, False))
        process.wait.assert_called_once_with(timeout=None)

    def test_explicit_turn_deadline_is_a_recorded_harness_interruption(self):
        """An operator stop request is retained separately from a model failure."""
        process = Mock(pid=123, returncode=-15)
        process.wait.side_effect = [stress.subprocess.TimeoutExpired("opencode", 3), -15]
        with patch.object(stress.os, "killpg") as kill:
            self.assertEqual(stress.wait_for_client(process, 3), (-15, True))
        kill.assert_called_once_with(123, stress.signal.SIGTERM)

    def test_cancelled_client_retires_real_process_group_before_reraising(self):
        """An interrupted owner leaves no live client despite unbounded generation."""
        process = stress.subprocess.Popen(
            [sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        original_wait = process.wait
        calls = 0
        def wait(timeout=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise KeyboardInterrupt()
            return original_wait(timeout=timeout)
        try:
            with patch.object(process, 'wait', side_effect=wait), self.assertRaises(KeyboardInterrupt):
                stress.wait_for_client(process)
            self.assertEqual(process.poll(), -stress.signal.SIGTERM)
        finally:
            if process.poll() is None:
                process.kill()
            original_wait()

    def test_cancelled_unresponsive_client_is_killed_and_cancellation_is_preserved(self):
        """Only exceptional cleanup has a deadline; an unresponsive group is reaped."""
        for cancellation in (KeyboardInterrupt(), SystemExit(130)):
            process = Mock(pid=123, returncode=-9)
            process.wait.side_effect = [cancellation, stress.subprocess.TimeoutExpired('opencode', 5), -9]
            with self.subTest(cancellation=type(cancellation)), patch.object(stress.os, 'killpg') as kill:
                with self.assertRaises(type(cancellation)):
                    stress.wait_for_client(process)
            self.assertEqual([call.args for call in kill.call_args_list],
                             [(123, stress.signal.SIGTERM), (123, stress.signal.SIGKILL)])
            self.assertEqual(process.wait.call_count, 3)

    def test_campaign_cancellation_retires_worker_before_executor_join(self):
        """A main-thread interruption must reach the process owned by another thread."""
        admitted = threading.Event()
        children = []
        def session(args, *job):
            with args.client_owners.launch([sys.executable, '-c', 'import time; time.sleep(30)']) as process:
                children.append(process)
                admitted.set()
                stress.wait_for_client(process)
            return {}
        def interrupt(futures):
            self.assertTrue(admitted.wait(5), 'worker did not publish its native client')
            raise KeyboardInterrupt()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                argv = ['stress', '--base-url', 'http://127.0.0.1:1', '--model', 'fixture',
                        '--context-length', '131072', '--output', str(Path(temporary) / 'app')]
                with patch.object(sys, 'argv', argv), patch.object(stress, 'require_ready'), \
                     patch.object(stress.subprocess, 'check_output', return_value='1.18.34'), \
                     patch.object(stress, 'run_session', side_effect=session), \
                     patch.object(stress, 'as_completed', side_effect=interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        stress.main()
                self.assertEqual(len(children), 1)
                self.assertEqual(children[0].poll(), -stress.signal.SIGTERM)
                self.assertFalse(json.loads((Path(temporary) / 'app/stress.json').read_text())['complete'])
        finally:
            for process in children:
                if process.poll() is None:
                    process.kill()
                process.wait()

    def test_cancelled_campaign_cannot_admit_another_phase(self):
        """Cancellation and new native ownership share one explicit state transition."""
        owners = stress.ClientOwners()
        owners.cancel()
        with patch.object(stress.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                with owners.launch(['opencode']):
                    self.fail('cancelled campaign admitted a client')
        launch.assert_not_called()

    def test_corrupt_or_incomplete_evidence_fails(self):
        """Bad identities, types, terminal state and byte damage are independent reds."""
        for mutation in ("join", "arguments", "completion", "terminal", "bytes", "events", "exit", "wire"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                rows, events = conversation(directory, True)
                code = 0
                if mutation == "join":
                    rows[1]["request"]["messages"][-1]["tool_call_id"] = "other"
                elif mutation == "arguments":
                    rows[0]["response"] = wire({"role": "assistant", "tool_calls": [{
                        "id": "call_write", "type": "function", "function": {
                            "name": "write", "arguments": '{"filePath":"result.txt","content":123}'}}]}, True)
                elif mutation == "completion":
                    events.pop(0)
                elif mutation == "terminal":
                    events.pop()
                elif mutation == "bytes":
                    (directory / "workspace/result.txt").write_bytes(b"123\n")
                elif mutation == "events":
                    events.append(None)
                elif mutation == "exit":
                    code = 1
                elif mutation == "wire":
                    rows[0]["response"]["body"] = rows[0]["response"]["body"].replace("data: [DONE]\n\n", "")
                (directory / "client.jsonl").write_text("\n".join(json.dumps(event) for event in events))
                result = stress.check_session(directory, rows, code)
                self.assertFalse(result["passed"])
                self.assertEqual(result["protocol_passed"], mutation == "bytes", result["errors"])


if __name__ == "__main__":
    unittest.main()
