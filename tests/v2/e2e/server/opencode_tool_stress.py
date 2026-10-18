#!/usr/bin/env python3
"""Stress a live Llaminar endpoint with the real OpenCode coding agent.

OpenCode owns its system prompt, tool descriptions, JSON schemas, argument
validation, execution and tool-result continuation. A transparent loopback
proxy preserves its exact requests and raw responses without rewriting them.
Each session operates in an isolated fixture. Protocol evidence and task quality
are graded independently: exact requested file bytes and app behavior measure
the task, while wire schemas, client inputs and continuations prove transport.
Requested streaming usage must reach the real SDK with positive input counts,
so the coding agent's normal context-compaction policy remains observable.
The final app review names concrete discovery and search operations so required
tool coverage is an explicit coding task, not an inferred preference.
Protocol and SDK failures stay fatal even after repair. Recognized tool-domain
errors require authenticated arguments and remain visible as task failures.
The protocol gate also requires successful coverage of every requested tool.
This is a focused diagnostic, not an image or model-matrix certificate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import http.client
import html
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlsplit
from urllib.request import urlopen, Request
from urllib.error import HTTPError

from tool_calling_checks import publish, response_message
from opencode_stress_policy import stress_assessment
from prefix_reuse_measurement import idle_snapshot, measure_request, summarize as summarize_prefix_reuse


def require_ready(base_url: str, timeout: float | None = None) -> None:
    """Admit client work only after the endpoint reports readiness; do not retry."""
    endpoint = urlsplit(base_url)
    url = f"{endpoint.scheme}://{endpoint.netloc}/health"
    with urlopen(url, timeout=5 if timeout is None else min(timeout, 5)) as response:
        if response.status != 200 or json.loads(response.read()).get("status") != "ok":
            raise ValueError("Llaminar endpoint is not ready")


def fixture_cases() -> dict[str, str]:
    """Exercise string arguments whose contents collide with JSON value syntax."""
    return {"number": "123", "boolean": "true", "null": "null",
            "object": '{"count":123,"ready":true}', "array": '[1,"two",null]',
            "quoted": '"quoted"', "code": 'def answer():\n    return 123\n',
            "whitespace": '    first\n\tsecond  \n', "empty": "", "unicode": "λ 中文 🙂",
            "delimiters": 'print("<tool_call></tool_call><parameter=x></parameter>")\n',
            "workflow": "123", "webapp": "123"}


def coding_prompt(case: str, content: str) -> str:
    """Describe a coding task while retaining OpenCode's own system/tool prompts."""
    if case == "workflow":
        return ("Fix the answer function in src/answer.py to return 123. Work only in this directory. "
                "Use todowrite to track the work, glob to find the Python files, grep to locate 'return 0', "
                "and read to inspect src/answer.py. Use edit to change only 'return 0' to 'return 123'. "
                "Run exactly `python3 -m unittest discover -s tests` using bash. Then use write to create "
                "result.txt containing exactly 123 with no newline, and read to verify it. Complete your todos.")
    return ("Work only in the current directory. "
            "Use the write tool to create result.txt with the exact UTF-8 content decoded "
            f"from this JSON string: {json.dumps(content, ensure_ascii=False)}. Do not add a newline or markdown. "
            "Any newline encoded in the JSON string is part of the requested content. "
            + ("For clarity, the decoded file content is:\n```text\n" + content + "```\n"
               "Use the literal characters shown, actual line breaks and the final newline. "
               if case in ("code", "whitespace", "delimiters") else "")
            + ("Use exactly four spaces before return. " if case == "code" else "")
            + ("The first line starts with exactly four spaces. The second starts with one TAB, "
               "ends with exactly two spaces, and is followed by one newline. " if case == "whitespace" else "")
            + "Use write, not shell commands or edit, for this creation. Then use read to "
            "read result.txt. Finish by reporting that you wrote and read the file.")


def session_prompts(case: str, content: str) -> list[str]:
    """Keep one real agent conversation across ten independently verified changes."""
    if case != "webapp":
        return [coding_prompt(case, content)]
    common = ("Work only in the current directory. Use Python's standard library only. "
              "Use your normal coding tools and track progress with todowrite. "
              "Use edit for incremental changes. Keep each increment small and run its unittests. "
              "After at most three repair attempts in this turn, finish with a brief status and any remaining "
              "failing tests; the next turn can continue the work. Do not repeat an unchanged failing test "
              "without a code or configuration change. Avoid redundant manual server probes. ")
    phases = [
        "Start Taskboard, a Python web app using ThreadingHTTPServer and SQLite. Create the taskboard package, "
        "a CLI runnable as `python3 -m taskboard --host 127.0.0.1 --port 8080 --database tasks.sqlite`, "
        "and a README. Implement GET /health returning JSON {\"status\":\"ok\"}. "
        "Keep this first increment small: add one HTTP unittest, run it and finish this turn. "
        "The task API, HTML interface and database features will be requested in subsequent turns.",
        "Add a JSON task API: GET /api/tasks returns a list ordered by id; POST /api/tasks accepts {title: string} "
        "and returns 201 with {id, title, completed: false}. Add HTTP unittests for creating/listing tasks.",
        "Extend the task API: GET /api/tasks/<id> returns that task. "
        "PATCH updates title and/or completed and returns 200; DELETE returns 204. IDs are integers. Test it.",
        "Add an HTML page at / with a task creation form and completion/deletion controls. Server-render "
        "the initial task titles and safely escape them; JavaScript uses the JSON API. Add CSS and document usage.",
        "Make Unicode and emoji round-trip exactly through HTTP, SQLite and HTML. Include tests with "
        "🙂 👩🏽‍💻 🇬🇧 ❤️ 1️⃣ 🚀 中文 λ, and JSON encoded with both literal UTF-8 and escaped surrogate pairs. "
        "Response Content-Length must count UTF-8 bytes. Test an HTML-injection title too.",
        "Harden input validation: malformed JSON, non-object bodies, missing/empty/whitespace titles, "
        "non-string titles and non-boolean completed values return JSON errors with status 400. "
        "Unknown task IDs return 404. An invalid request must not corrupt persisted tasks. Test each case.",
        "Add GET /api/tasks?completed=true or completed=false filtering plus limit and offset pagination. "
        "Filtering happens before pagination; id ordering is stable. Invalid completed, negative/non-numeric "
        "limits or offsets return 400. Add independent HTTP tests for all query variants.",
        "Verify task persistence across app restarts. Add tests that creates, updates and deletes survive restart.",
        "Add tests creating at least 12 tasks concurrently and checking unique IDs and all titles. "
        "Use explicit per-request SQLite connection ownership and configure the HTTP listener to handle "
        "that concurrent connection count. Distinguish HTTP connection failures from database errors.",
        "Review the entire app and tests. Begin by calling the glob tool with pattern '**/*.py', "
        "the grep tool with pattern 'def ' and include '*.py', and the read tool on a returned Python file. "
        "These three tool calls are required parts of the review. Add missing edge-case tests, update README, "
        "and run `python3 -m unittest discover -s tests -v`. Finish all todos. Finally use write to create "
        "result.txt containing exactly 123 with no newline, then read it; this is a tool-string regression sentinel.",
    ]
    return [common + phase for phase in phases]


def requested_streaming_usage(exchange: dict) -> dict | None:
    """Authenticate exact terminal accounting required by the coding client's SDK."""
    request = exchange["request"]
    if not request.get("stream") or not (request.get("stream_options") or {}).get("include_usage"):
        return None
    response = exchange["response"]
    identity, usage, finished, done = None, None, False, False
    for line in response["body"].splitlines():
        if not line.startswith("data:"):
            continue
        if done:
            raise ValueError("streaming usage data arrived after DONE")
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
            continue
        chunk = json.loads(payload)
        current = tuple(chunk.get(key) for key in ("id", "model", "created"))
        if (any(not isinstance(value, str) or not value for value in current[:2])
                or type(current[2]) is not int or current[2] < 0):
            raise ValueError("streaming usage lacks a valid completion identity")
        if identity is None:
            identity = current
        if current != identity:
            raise ValueError("streaming usage changed completion identity")
        choices = chunk.get("choices")
        if choices:
            if finished or usage is not None or chunk.get("usage", "missing") is not None:
                raise ValueError("streaming choice must publish null requested usage")
            finished = finished or choices[0].get("finish_reason") is not None
            continue
        if choices != [] or not finished or usage is not None or not isinstance(chunk.get("usage"), dict):
            raise ValueError("streaming usage must occur once after the terminal choice")
        usage = chunk["usage"]
        counts = [usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
        if (any(type(value) is not int or value < 0 for value in counts)
                or counts[0] == 0 or counts[2] != counts[0] + counts[1]):
            raise ValueError("streaming usage has invalid prompt/completion totals")
    if usage is None or not done:
        raise ValueError("requested streaming usage is missing before DONE")
    return usage


@contextmanager
def app_process(workspace: Path, database: Path, output: Path):
    """Run the generated CLI as a separate, bounded process for black-box HTTP checks."""
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    with output.open("ab") as log:
        process = subprocess.Popen(
            ["python3", "-m", "taskboard", "--host", "127.0.0.1", "--port", str(port),
             "--database", str(database)], cwd=workspace, stdout=log, stderr=log, start_new_session=True)
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 10
            while True:
                try:
                    with urlopen(base + "/health", timeout=1) as reply:
                        if reply.status == 200:
                            break
                except OSError:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise ValueError("generated web app did not become ready")
                    time.sleep(0.05)
            yield base
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def check_webapp(directory: Path) -> dict:
    """Assert behavior independently of the tests authored by the coding model."""
    workspace = directory / "workspace"
    checks, failures = [], []
    def expect(condition, label):
        checks.append(label)
        if not condition:
            failures.append(label)
    def exchange(base, method, path, value=None, *, raw=None, escaped=False):
        data = raw if raw is not None else (json.dumps(value, ensure_ascii=escaped).encode() if value is not None else None)
        try:
            reply = urlopen(Request(base + path, data=data, method=method,
                                    headers={"Content-Type": "application/json"}), timeout=10)
        except HTTPError as error:
            reply = error
        with reply:
            payload = reply.read()
            expect(reply.headers.get("Content-Length") == str(len(payload)) or (reply.status == 204 and not payload), f"{method} {path}: byte length")
            content = json.loads(payload) if payload and "application/json" in reply.headers.get("Content-Type", "") else payload.decode()
            return reply.status, content
    database = directory / "acceptance.sqlite"
    try:
        with app_process(workspace, database, directory / "app.stderr") as base:
            expect(exchange(base, "GET", "/health") == (200, {"status": "ok"}), "health contract")
            emoji = "🙂👩🏽‍💻🇬🇧❤️1️⃣🚀 中文 λ"
            tasks = []
            for escaped in (False, True):
                status, task = exchange(base, "POST", "/api/tasks", {"title": emoji}, escaped=escaped)
                expect(status == 201 and task.get("title") == emoji and task.get("completed") is False,
                       f"emoji creation escaped={escaped}")
                tasks.append(task)
                expect(exchange(base, "GET", f"/api/tasks/{task['id']}") == (200, task), "emoji get")
            identity = tasks[0]["id"]
            status, updated = exchange(base, "PATCH", f"/api/tasks/{identity}", {"completed": True, "title": "done " + emoji})
            expect(status == 200 and updated.get("completed") is True and updated.get("title") == "done " + emoji, "update round trip")
            expect(exchange(base, "GET", "/api/tasks?completed=true")[1] == [updated], "completed filter")
            expect(exchange(base, "GET", "/api/tasks?completed=false&limit=1&offset=0")[1] == [tasks[1]], "filter before pagination")
            status, injection = exchange(base, "POST", "/api/tasks", {"title": '<script>alert("x")</script>'})
            expect(status == 201, "HTML title creation")
            status, page = exchange(base, "GET", "/")
            expect(status == 200 and html.escape(injection["title"]) in page and injection["title"] not in page, "HTML escaping")
            for value in ({}, {"title": ""}, {"title": "  "}, {"title": 123}, [], None):
                status, body = exchange(base, "POST", "/api/tasks", value, raw=b"null" if value is None else None)
                expect(status == 400 and isinstance(body, dict), f"invalid create {value!r}")
            expect(exchange(base, "POST", "/api/tasks", raw=b"{broken")[0] == 400, "malformed JSON")
            expect(exchange(base, "PATCH", f"/api/tasks/{identity}", {"completed": "true"})[0] == 400, "boolean validation")
            expect(exchange(base, "GET", "/api/tasks/999999999")[0] == 404, "missing task")
            for query in ("completed=maybe", "limit=-1", "limit=no", "offset=-1", "offset=no"):
                expect(exchange(base, "GET", "/api/tasks?" + query)[0] == 400, f"invalid query {query}")
            with ThreadPoolExecutor(max_workers=12) as executor:
                created = list(executor.map(lambda index: exchange(base, "POST", "/api/tasks", {"title": f"parallel-{index}"}), range(12)))
            expect(all(status == 201 for status, _ in created), "concurrent creates")
            expect(len({task["id"] for _, task in created}) == 12, "unique concurrent IDs")
            expect({task["title"] for _, task in created} == {f"parallel-{index}" for index in range(12)}, "all concurrent titles")
            status, inventory = exchange(base, "GET", "/api/tasks")
            expect(status == 200 and [task["id"] for task in inventory] == sorted(task["id"] for task in inventory), "stable ordering")
            expect(exchange(base, "GET", "/api/tasks?limit=3&offset=2")[1] == inventory[2:5], "pagination")
            expect(exchange(base, "DELETE", f"/api/tasks/{tasks[1]['id']}")[0] == 204, "delete")
            expect(exchange(base, "GET", f"/api/tasks/{tasks[1]['id']}")[0] == 404, "deleted task absent")
        with app_process(workspace, database, directory / "app.stderr") as base:
            expect(exchange(base, "GET", f"/api/tasks/{identity}") == (200, updated), "update persisted across restart")
            expect(exchange(base, "GET", f"/api/tasks/{tasks[1]['id']}")[0] == 404, "delete persisted across restart")
            expect(len(exchange(base, "GET", "/api/tasks")[1]) == len(inventory) - 1, "complete inventory persisted")
        tests = subprocess.run(["python3", "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=workspace,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        (directory / "independent-unittest.log").write_text(tests.stdout)
        expect(tests.returncode == 0 and "Ran 0 tests" not in tests.stdout, "authored tests execute and pass")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, subprocess.TimeoutExpired) as error:
        failures.append(f"web app acceptance: {error}")
    result = {"passed": not failures, "checks": checks, "errors": failures}
    publish(directory / "webapp-acceptance.json", result)
    return result


def authenticated_workflow_tests(events: list[dict], calls: dict, workspace: str) -> list[dict]:
    """Prove a completed test command used the intended workspace and ran real tests.

    OpenCode reports a failed shell process as a completed tool. Tool presence
    and completion therefore cannot establish a passing test run. Join the
    exact requested command and client arguments with its native exit status
    and unittest's nonempty discovery summary; retain the successful call IDs.
    """
    evidence = []
    for event in events:
        part = event.get("part", {})
        state = part.get("state", {})
        call = calls.get(part.get("callID"))
        if (part.get("type") != "tool" or part.get("tool") != "bash"
                or state.get("status") != "completed" or call is None
                or call["function"]["name"] != "bash"):
            continue
        arguments = json.loads(call["function"]["arguments"])
        workdir = arguments.get("workdir", workspace)
        if (arguments != state.get("input")
                or arguments.get("command") != "python3 -m unittest discover -s tests"
                or not isinstance(workdir, str)
                or os.path.normpath(os.path.join(workspace, workdir)) != os.path.normpath(workspace)):
            continue
        status = state.get("metadata", {}).get("exit")
        output = state.get("output")
        summary = re.search(r"(?m)^Ran ([1-9][0-9]*) tests? in [^\r\n]+\r?$", output) if isinstance(output, str) else None
        if type(status) is int and status == 0 and summary is not None:
            evidence.append({"call_id": call["id"], "tests": int(summary[1]), "exit": status})
    return evidence


def assessment(protocol_errors: list[str], task_errors: list[str]) -> dict:
    """Keep strict task success visible even when qualifying only the protocol.

    Protocol success is evidence about the HTTP/client boundary, not numerical
    inference correctness. A native campaign must additionally authenticate raw
    model output, generation controls, PerfStats and the driver lifetime.
    """
    return {"passed": not protocol_errors and not task_errors,
            "errors": protocol_errors + task_errors,
            "protocol_passed": not protocol_errors, "protocol_errors": protocol_errors,
            "task_passed": not task_errors, "task_errors": task_errors}


def required_tools(case: str) -> set[str]:
    """Name the fixture's requested surface independently of model choices."""
    tools = {"write", "read"}
    if case in ("workflow", "webapp"):
        tools |= {"edit", "glob", "grep", "todowrite", "bash"}
    return tools


def campaign_assessment(results: list[dict], cases: list[str], iterations: int, gate: str) -> dict:
    """Grade every planned session and require successful tool coverage per case.

    Model omissions cannot manufacture a protocol pass by avoiding the tool
    surface under test. Coverage is reported separately from a protocol defect;
    an uncovered campaign is incomplete evidence, irrespective of task quality.
    ``passed`` always retains strict success. Only ``gate_passed`` and the CLI
    exit select the explicitly requested protocol or strict task gate.
    """
    coverage = []
    for case in sorted(set(cases)):
        rows = [row for row in results if row["case"] == case]
        observed = {name for row in rows for name in row["successful_tools"]}
        required = required_tools(case)
        coverage.append({"case": case, "sessions": len(rows),
                         "expected_sessions": cases.count(case) * iterations,
                         "required_tools": sorted(required), "successful_tools": sorted(observed),
                         "missing_tools": sorted(required - observed)})
    complete = (len(results) == len(cases) * iterations
                and all(row["sessions"] == row["expected_sessions"] for row in coverage))
    protocol = complete and all(row["protocol_passed"] for row in results)
    task = complete and all(row["task_passed"] for row in results)
    covered = complete and all(not row["missing_tools"] for row in coverage)
    if gate not in ("strict", "protocol", "stress"):
        raise ValueError("unknown OpenCode qualification gate")
    release_stress = stress_assessment(results) if gate == "stress" else None
    return {"passed": protocol and task and covered, "protocol_passed": protocol,
            "task_passed": task, "coverage_passed": covered, "coverage": coverage,
            "gate_passed": protocol and covered and (
                release_stress['passed'] if gate == "stress" else gate == "protocol" or task),
            **({"stress_policy": release_stress} if release_stress is not None else {}),
            "protocol_passed_sessions": sum(row["protocol_passed"] for row in results),
            "task_passed_sessions": sum(row["task_passed"] for row in results)}


def authenticate_native_qwen_calls(native_content: str, calls: list[dict], tools: list[dict]) -> dict:
    """Prove complete wire argument objects occur in independently decoded native text.

    This is an inverse renderer, not the production delimiter scanner: recorded
    string bytes must match in full before a closing tag can be consumed. Literal
    tags inside source code therefore cannot truncate a value or swallow its
    neighbor unnoticed. Typed JSON arguments are read by Python's JSON decoder.
    Accept only the native grammar's optional LF/CRLF framing and whitespace
    between structural elements. Call order, all arguments and the complete
    function/tool boundary must match. The caller owns token/log/request joins
    and must supply text after the thinking split, never an HTTP echo.

    A call-free result does not assert that every malformed native block was
    executable; required-key rejection remains a separate parser invariant.
    """
    definitions = {tool["function"]["name"]: tool["function"].get("parameters", {}) for tool in tools}
    whitespace = " \t\n\r\f\v"
    decoder = json.JSONDecoder()
    spans, errors, cursor = [], [], 0
    for index, call in enumerate(calls):
        name, arguments = call["name"], call["arguments"]
        schema = definitions.get(name, {})
        properties = schema.get("properties", {})
        if (name not in definitions or not isinstance(arguments, dict)
                or not all(key in properties for key in arguments)
                or not all(key in arguments for key in schema.get("required", []))):
            errors.append(f"native call {index}: unknown or incomplete argument schema")
            continue

        @lru_cache(maxsize=None)
        def match_parameters(position, remaining):
            """Match expected value bytes before admitting any following grammar token."""
            position += len(native_content[position:]) - len(native_content[position:].lstrip(whitespace))
            if not remaining:
                end = re.match(r"</function>\s*</tool_call>", native_content[position:], re.ASCII)
                return position + end.end() if end else None
            opening = re.match(r"<parameter=([^>]+)>", native_content[position:])
            if not opening:
                return None
            key = opening[1].strip(whitespace)
            if key not in remaining:
                return None
            begin = position + opening.end()
            value = arguments[key]
            ends = set()
            if properties[key].get("type") == "string" and isinstance(value, str):
                for leading in ("", "\n", "\r\n"):
                    for trailing in ("", "\n", "\r\n"):
                        body = leading + value + trailing + "</parameter>"
                        if native_content.startswith(body, begin):
                            ends.add(begin + len(body))
            elif properties[key].get("type") != "string":
                try:
                    start = begin + len(native_content[begin:]) - len(native_content[begin:].lstrip(whitespace))
                    decoded, end = decoder.raw_decode(native_content, start)
                    end += len(native_content[end:]) - len(native_content[end:].lstrip(whitespace))
                    if (json.dumps(decoded, sort_keys=True) == json.dumps(value, sort_keys=True)
                            and native_content.startswith("</parameter>", end)):
                        ends.add(end + len("</parameter>"))
                except ValueError:
                    pass
            for end in sorted(ends):
                matched = match_parameters(end, tuple(item for item in remaining if item != key))
                if matched is not None:
                    return matched
            return None

        opening = re.compile(r"<tool_call>\s*<function=\s*" + re.escape(name) + r"\s*>", re.ASCII)
        matched = None
        for block in opening.finditer(native_content, cursor):
            end = match_parameters(block.end(), tuple(sorted(arguments)))
            if end is not None:
                matched = (block.start(), end)
                break
        if matched is None:
            errors.append(f"native call {index} ({name}): complete wire arguments absent from native generation")
        else:
            spans.append({"call_index": index, "begin": matched[0], "end": matched[1]})
            cursor = matched[1]
    return {"passed": not errors, "verified_calls": len(spans), "spans": spans, "errors": errors}


def tool_domain_error(part: dict, arguments: dict, workspace: str) -> str | None:
    """Recognize narrow client-domain outcomes after wire/input authentication.

    An arbitrary tool error, including an SDK schema failure, is never assigned
    to task quality. A permission denial must contain the actual client rules;
    it is a tool outcome, not proof that the model or the rule was appropriate.
    No rule is changed and no denied operation is retried by this observer.
    """
    name, error = part.get("tool"), part.get("state", {}).get("error")
    if not isinstance(error, str):
        return None
    # OpenCode reports distinct outcomes for an absent edit target and an
    # ambiguous target. Both are model editing mistakes only after the caller
    # authenticates unchanged wire/client arguments. Exact messages keep SDK
    # failures and unknown tool errors outside this task-quality classification.
    edit_errors = {
        "Could not find oldString in the file. It must match exactly, including whitespace, indentation, and line endings.": "edit_match_not_found",
        "Found multiple matches for oldString. Provide more surrounding context to make the match unique.": "edit_match_ambiguous",
    }
    if (name == "edit" and all(isinstance(arguments.get(key), str)
            for key in ("filePath", "oldString", "newString"))
            and error in edit_errors):
        return edit_errors[error]
    # v1.18.34 edit.ts rejects identical input before execution, and its replace
    # helper applies the same check after CRLF normalization. Compare only the
    # already-authenticated values here; wire/client argument checks retain the
    # original bytes. A different edit or empty path cannot explain this error.
    # https://github.com/anomalyco/opencode/blob/v1.18.34/packages/opencode/src/tool/edit.ts
    if (name == "edit" and all(isinstance(arguments.get(key), str)
            for key in ("filePath", "oldString", "newString"))
            and arguments["filePath"]
            and error == "No changes to apply: oldString and newString are identical."
            and arguments["oldString"].replace("\r\n", "\n") == arguments["newString"].replace("\r\n", "\n")):
        return "edit_no_change"
    if (name == "read" and isinstance(arguments.get("filePath"), str)
            and error == "File not found: " + os.path.normpath(os.path.join(workspace, arguments["filePath"]))):
        return "file_not_found"
    if (name == "read" and isinstance(arguments.get("filePath"), str) and arguments['filePath']
            and error == "Cannot read binary file: " + os.path.normpath(os.path.join(workspace, arguments['filePath']))):
        return "read_binary_file"
    # A JSON string can contain NUL even though a process command cannot. The
    # observed pinned client quotes literal double quotes in its exact echoed
    # argument; it does not JSON-escape backslashes or control characters. Never
    # classify a different command/error or transform the authenticated input.
    command = arguments.get('command')
    if (name == 'bash' and isinstance(command, str) and '\x00' in command
            and error == "The argument 'file' must be a string without null bytes. Received "
                         + '"' + command.replace('"', '\\"') + '"'):
        return 'bash_command_contains_nul'
    # OpenCode 1.18.34 checks the resolved Bash workdir before spawning the
    # command. Its completed NotFound error is a failed task only when it names
    # the exact authored directory. Preserve wire/client bytes above; normalize
    # only this diagnostic path, without inspecting a retired session's files.
    # A missing default workspace or another filesystem failure is not covered.
    if (name == "bash" and isinstance(arguments.get("command"), str)
            and isinstance(arguments.get("workdir"), str) and arguments["workdir"]
            and error == "NotFound: FileSystem.access (" + os.path.normpath(
                os.path.join(workspace, arguments["workdir"])) + ")"):
        return "bash_workdir_not_found"
    # OpenCode's glob schema has already accepted these exact client arguments.
    # Its ripgrep adapter wraps process/stream failures with this exact message.
    # This authenticates a failed external operation, not its underlying cause;
    # the task still fails, and unknown errors/schema failures remain protocol
    # failures. Do not infer historical directory existence from the current FS.
    # https://github.com/anomalyco/opencode/blob/v1.18.34/packages/core/src/ripgrep.ts
    if (name == "glob" and error == "ripgrep execution failed"
            and isinstance(arguments.get("pattern"), str)
            and ("path" not in arguments or isinstance(arguments["path"], str))):
        return "glob_execution_failed"
    # OpenCode v1.18.34 forwards ripgrep's invalid-pattern diagnostic verbatim.
    # Authenticate its entire echoed expression and bounded caret annotation;
    # a substring such as "regex parse error" cannot explain an SDK failure or
    # an error for another pattern. Recognize the observed invalid-escape
    # contract only. This is failed search work and never successful coverage.
    # https://github.com/anomalyco/opencode/blob/v1.18.34/packages/core/src/ripgrep.ts
    pattern = arguments.get("pattern")
    if (name == "grep" and isinstance(pattern, str) and pattern
            and "\n" not in pattern and "\r" not in pattern
            and all(key not in arguments or isinstance(arguments[key], str)
                    for key in ("path", "include"))):
        prefix = "rg: regex parse error:\n    (?:" + pattern + ")\n"
        if error.startswith(prefix):
            marker = re.fullmatch(r"( +)(\^+)\nerror: unrecognized escape sequence", error[len(prefix):])
            if (marker is not None and len(marker[1]) >= 7
                    and len(marker[1]) + len(marker[2]) <= 7 + len(pattern)):
                return "grep_invalid_regex"
    prefix = ("The user has specified a rule which prevents you from using this specific tool call. "
              "Here are some of the relevant rules ")
    if not error.startswith(prefix):
        return None
    try:
        rules = json.loads(error[len(prefix):])
    except ValueError:
        return None
    if (isinstance(rules, list) and all(isinstance(rule, dict)
            and isinstance(rule.get("permission"), str) and isinstance(rule.get("pattern"), str)
            and rule.get("action") in ("allow", "ask", "deny") for rule in rules)
            and {"permission": "external_directory", "action": "deny", "pattern": "*"} in rules):
        return "external_directory_denied"
    return None


def is_compaction_request(request: dict) -> bool:
    """Identify OpenCode's auxiliary summary of older history, without consuming tool joins.

    OpenCode 1.18.34 sends the selected older conversation to a separate agent
    without tools. Its next coding request reattaches the recent assistant/tool
    tail. The summary is still validated and counted, but only that resumed
    coding request can satisfy the pending call's history obligation.
    """
    messages = request.get("messages")
    if (request.get("tools", []) != [] or not isinstance(messages, list) or len(messages) != 2
            or not all(isinstance(message, dict) for message in messages)):
        return False
    system, user = messages
    return (system.get("role") == "system" and user.get("role") == "user"
            and isinstance(system.get("content"), str) and isinstance(user.get("content"), str)
            and system["content"].startswith("You are a context summarization agent.")
            and "\n<conversation>\n" in user["content"] and "\n</conversation>\n" in user["content"])


def check_session(directory: Path, exchanges: list[dict], returncode: int) -> dict:
    """Join wire/client identities, separating protocol defects from task results."""
    errors, task_errors, calls, execution_errors, usage_records = [], [], [], [], []
    previous_calls, pending_calls, invalid_calls = {}, {}, set()
    compaction_requests = 0
    events = []
    for line in (directory / "client.jsonl").read_text().splitlines():
        try:
            event = json.loads(line)
            if not isinstance(event, dict) or not isinstance(event.get("part", {}), dict):
                raise ValueError("invalid client event shape")
            events.append(event)
        except ValueError:
            errors.append("client emitted malformed JSON")
    for event in events:
        if event.get("type") == "error":
            errors.append(f"OpenCode error: {event.get('error')}")
    if not any(event.get("type") == "step_finish" and event.get("part", {}).get("reason") == "stop"
               for event in events):
        errors.append("OpenCode did not finish the conversation normally")
    for row in exchanges:
        if row.get("error"):
            errors.append(row["error"])
            continue
        if row["path"] != "/v1/chat/completions":
            continue
        try:
            compaction = is_compaction_request(row["request"])
            history_calls, history_results = {}, {}
            for message in row["request"].get("messages", []):
                if message.get("role") == "assistant":
                    for call in message.get("tool_calls", []):
                        history_calls[call["id"]] = call
                if message.get("role") == "tool":
                    history_results[message["tool_call_id"]] = message
            if not compaction:
                for identity, emitted in pending_calls.items():
                    joined = history_calls.get(identity)
                    if (joined is None or identity not in history_results
                            or joined.get("function", {}).get("name") != emitted["function"]["name"]
                            or json.loads(joined["function"]["arguments"]) != json.loads(emitted["function"]["arguments"])):
                        errors.append(f"tool result/history did not preserve emitted call {identity}")
            if "response" not in row:
                raise ValueError("exchange has no completed response; conversation evidence is incomplete")
            message, finish = response_message(row["response"], row["request"].get("stream", False))
            usage = requested_streaming_usage(row)
            if usage is not None:
                usage_records.append(usage)
            if finish not in ("stop", "tool_calls"):
                errors.append(f"response ended with {finish}")
            definitions = {tool["function"]["name"]: tool["function"]
                           for tool in row["request"].get("tools", [])}
            if compaction:
                compaction_requests += 1
            else:
                pending_calls = {}
            for call in message.get("tool_calls", []):
                if (call.get("type") != "function" or not isinstance(call.get("id"), str)
                        or not call["id"] or call["id"] in previous_calls):
                    raise ValueError("response emitted a missing, reused or invalid tool identity")
                function = call["function"]
                arguments = json.loads(function["arguments"])
                if function["name"] not in definitions or not isinstance(arguments, dict):
                    raise ValueError("response called an unknown tool or emitted non-object arguments")
                # The real SDK performs complete schema validation. Retain an
                # independent check of the string fields implicated in Qwen's
                # native parameter decoding, including failures it may repair.
                schema = definitions[function["name"]].get("parameters", {})
                properties = schema.get("properties", {})
                for name in schema.get("required", []):
                    if name not in arguments:
                        errors.append(f"{function['name']}.{name}: missing required argument")
                        invalid_calls.add(call["id"])
                for name, value in arguments.items():
                    if properties.get(name, {}).get("type") == "string" and not isinstance(value, str):
                        errors.append(f"{function['name']}.{name}: expected string, got {type(value).__name__}")
                        invalid_calls.add(call["id"])
                calls.append(call)
                previous_calls[call["id"]] = call
                pending_calls[call["id"]] = call
            if message.get("tool_calls") and finish != "tool_calls":
                errors.append("tool response has the wrong finish reason")
            for message in row["request"].get("messages", []):
                if message.get("role") == "tool" and any(text in str(message.get("content", ""))
                        for text in ("Invalid arguments", "Type validation failed", "did not match the expected schema")):
                    errors.append("OpenCode returned a tool argument validation error")
        except (ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
    for identity in pending_calls:
        errors.append(f"tool result/history lacks a completed coding continuation for emitted call {identity}")
    if any((row["request"].get("stream_options") or {}).get("include_usage") for row in exchanges):
        for event in events:
            if event.get("type") != "step_finish":
                continue
            tokens = event.get("part", {}).get("tokens", {})
            cached = tokens.get("cache", {})
            if tokens.get("input", 0) + cached.get("read", 0) + cached.get("write", 0) <= 0:
                errors.append("OpenCode SDK reported zero input tokens despite requested streaming usage")
    if returncode:
        errors.append(f"OpenCode exited {returncode}")
    names = {call["function"]["name"] for call in calls}
    completed, successful_tools = set(), set()
    workspace = None
    try:
        workspace = json.loads((directory / "workspace-location.json").read_text())["live_workspace"]
        if not isinstance(workspace, str) or not os.path.isabs(workspace):
            raise ValueError("original workspace must be an absolute path")
        workspace = os.path.normpath(workspace)
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"cannot authenticate exact file tools: {error}")
    for event in events:
        part = event.get("part", {})
        state = part.get("state", {})
        if part.get("type") != "tool" or state.get("status") not in ("completed", "error"):
            continue
        identity = part.get("callID")
        call = previous_calls.get(identity)
        arguments = json.loads(call["function"]["arguments"]) if call else None
        if identity in invalid_calls:
            if state.get("status") == "error":
                errors.append(f"OpenCode tool error: {state.get('error')}")
            continue  # Invalid arguments cannot authenticate an execution outcome.
        if (call is None or part.get("tool") != call["function"]["name"]
                or arguments != state.get("input")):
            errors.append(f"client tool outcome did not preserve emitted call {identity}")
            if state.get("status") == "error":
                errors.append(f"OpenCode tool error: {state.get('error')}")
            continue
        if state["status"] == "completed":
            successful_tools.add(part["tool"])
        else:
            kind = tool_domain_error(part, arguments, workspace) if workspace is not None else None
            if kind is None:
                errors.append(f"OpenCode tool error: {state.get('error')}")
                continue
            execution_errors.append({"kind": kind, "call_id": identity,
                                     "error": state["error"], "input": arguments})
            task_errors.append(f"OpenCode task error ({kind}): {state['error']}")
        completed.add(identity)
    for identity in previous_calls:
        if identity not in completed:
            errors.append(f"OpenCode did not complete emitted tool {identity}")
    required = required_tools(directory.name.rsplit("-", 1)[-1])
    if directory.name.endswith("-workflow"):
        source = directory / "workspace/src/answer.py"
        if not source.is_file() or source.read_text() != "def answer():\n    return 123\n":
            task_errors.append("coding edit did not preserve the exact source bytes")
    if not required <= names:
        task_errors.append(f"session omitted required tools: {sorted(required - names)}")
    expected = (directory / "expected.bin").read_bytes()
    actual_path = directory / "workspace/result.txt"
    if not actual_path.is_file() or actual_path.read_bytes() != expected:
        task_errors.append("result.txt bytes differ from the requested content")
    matching_file_tools = set()
    workflow_tests = []
    try:
        if workspace is None:
            raise ValueError("original workspace could not be authenticated")
        if directory.name.endswith("-workflow"):
            workflow_tests = authenticated_workflow_tests(events, previous_calls, workspace)
        target = os.path.normpath(os.path.join(workspace, "result.txt"))
        for event in events:
            part = event.get("part", {})
            state = part.get("state", {})
            call = previous_calls.get(part.get("callID"))
            if (part.get("type") != "tool" or state.get("status") != "completed"
                    or call is None or part.get("tool") not in ("write", "read")
                    or part["tool"] != call["function"]["name"]):
                continue
            arguments = json.loads(call["function"]["arguments"])
            inputs = (arguments, state.get("input"))
            # Resolve against the recorded live workspace, which may already
            # be retired. Basenames or final bytes alone cannot prove that the
            # requested tool operated on the session's actual result file.
            if not all(isinstance(value, dict) and isinstance(value.get("filePath"), str)
                       and os.path.normpath(os.path.join(workspace, value["filePath"])) == target
                       for value in inputs):
                continue
            if part["tool"] == "read" and "write" not in matching_file_tools:
                continue  # A read preceding the requested write cannot verify it.
            if part["tool"] == "write" and not all(
                    isinstance(value.get("content"), str) and value["content"].encode("utf-8") == expected
                    for value in inputs):
                continue
            matching_file_tools.add(part["tool"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"cannot authenticate exact file tools: {error}")
    for name in sorted((names & {"write", "read"}) - matching_file_tools):
        if name == "read" and "write" not in names:
            continue  # The omitted-write diagnostic already explains this missing predecessor.
        task_errors.append(f"no matching completed {name} for the requested result.txt bytes/path")
    if directory.name.endswith("-workflow") and not workflow_tests:
        task_errors.append("workflow lacks a matching completed, passing unittest command")
    return {**assessment(errors, task_errors), "execution_errors": execution_errors, "tool_calls": calls,
            "tool_attempts": len(previous_calls), "tool_errors": len({row['call_id'] for row in execution_errors}),
            "successful_tools": sorted(successful_tools), "required_tools": sorted(required),
            "workflow_tests": workflow_tests,
            "compaction_requests": compaction_requests,
            "streaming_usage": usage_records,
            "request_count": len(exchanges), "client_event_count": len(events)}


def repeated_suffix(text: str) -> dict:
    """Report sustained exact suffix repetition as evidence, never a stop rule.

    Ignore single characters and short whitespace patterns common in code. A
    repeated phrase is only a diagnostic signal; source code and deliberate
    quotations can legitimately contain repetitions.
    """
    tail = text[-4096:]
    best = {"repeats": 0, "period_characters": 0, "sample": ""}
    for width in range(8, min(256, len(tail) // 4) + 1):
        phrase = tail[-width:]
        if len(phrase.strip()) < 8:
            continue
        count = 1
        while len(tail) >= width * (count + 1) and tail[-width * (count + 1):-width * count] == phrase:
            count += 1
        if count >= 4 and count > best["repeats"]:
            best = {"repeats": count, "period_characters": width, "sample": phrase[:160]}
    return best


class StreamObservation:
    """Bounded live diagnostics alongside the untouched raw response stream."""
    def __init__(self, path: Path, clock=time.time):
        """Bind one exchange's progress artifact and an injectable wall clock."""
        self.path, self.clock = path, clock
        self.started = clock()
        self.last_publish = 0.0
        self.characters = {"content": 0, "reasoning_content": 0, "tool_arguments": 0}
        self.tails = {key: "" for key in self.characters}
        self.delta_count, self.response_bytes = 0, 0
        self.tool_names = []
        self.finish_reason, self.usage = None, None
        self.repetition_peaks = {key: {"repeats": 0} for key in self.characters}

    def record(self, block: bytes) -> None:
        """Observe complete SSE lines without changing or delaying their payloads."""
        self.response_bytes += len(block)
        self.last_activity = self.clock()
        if not block.startswith(b"data:"):
            return
        payload = block[5:].strip()
        if payload == b"[DONE]":
            self.flush(complete=True)
            return
        try:
            event = json.loads(payload)
        except (ValueError, UnicodeError):
            # The final wire validator reports malformed JSON independently.
            return
        # Malformed protocol shapes belong to the final wire validator. Live
        # observation must still leave their raw bytes available for diagnosis.
        if not isinstance(event, dict):
            return
        if isinstance(event.get("usage"), dict):
            self.usage = event["usage"]
        choices = event.get("choices", [])
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            fragments = {key: delta.get(key) if isinstance(delta.get(key), str) else ""
                         for key in ("content", "reasoning_content")}
            fragments["tool_arguments"] = ""
            tool_calls = delta.get("tool_calls", [])
            for tool in tool_calls if isinstance(tool_calls, list) else []:
                if not isinstance(tool, dict):
                    continue
                function = tool.get("function") or {}
                if not isinstance(function, dict):
                    continue
                if isinstance(function.get("name"), str) and function["name"]:
                    self.tool_names.append(function["name"])
                if isinstance(function.get("arguments"), str):
                    fragments["tool_arguments"] += function["arguments"]
            for key, fragment in fragments.items():
                if fragment:
                    self.delta_count += 1
                    self.characters[key] += len(fragment)
                    self.tails[key] = (self.tails[key] + fragment)[-4096:]
            if isinstance(choice.get("finish_reason"), str) and choice["finish_reason"]:
                self.finish_reason = choice["finish_reason"]
        if self.last_activity - self.last_publish >= 1.0 or self.finish_reason:
            self.flush()

    def flush(self, *, complete=False, error=None) -> None:
        """Publish counts and short tails atomically; delta counts are not token IDs."""
        now = self.clock()
        for key, tail in self.tails.items():
            observed = repeated_suffix(tail)
            if observed["repeats"] > self.repetition_peaks[key]["repeats"]:
                self.repetition_peaks[key] = observed
        publish(self.path, {"started": self.started, "updated": now,
            "last_activity": getattr(self, "last_activity", self.started),
            "elapsed_seconds": now - self.started, "response_bytes": self.response_bytes,
            "delta_count": self.delta_count, "characters": self.characters,
            "tail": {key: tail[-512:] for key, tail in self.tails.items()},
            "tool_names": self.tool_names, "repetition_signals": self.repetition_peaks,
            "finish_reason": self.finish_reason, "usage": self.usage,
            "complete": complete, "error": error})
        self.last_publish = now


@contextmanager
def recording_proxy(base_url: str, output: Path, timeout: float | None = None,
                    *, measure_prefix_reuse: bool = False):
    """Forward each response as it arrives and preserve unmodified payloads."""
    upstream = urlsplit(base_url)
    if upstream.scheme != "http" or not upstream.hostname or upstream.username or upstream.password:
        raise ValueError("base URL must be an HTTP endpoint without embedded credentials")
    rows, lock = [], threading.Lock()
    measurement_lock = threading.Lock()

    def read_stats() -> dict:
        """Read bounded endpoint metadata; the finite timeout applies only to observation."""
        with urlopen(f"{upstream.scheme}://{upstream.netloc}/stats", timeout=5) as response:
            if response.status != 200:
                raise ValueError('Prefix measurement statistics endpoint failed')
            encoded = response.read(1024 * 1024 + 1)
            if len(encoded) > 1024 * 1024:
                raise ValueError('Prefix measurement statistics exceeded its metadata bound')
            return json.loads(encoded)

    class Handler(BaseHTTPRequestHandler):
        """One wire relay; credentials are forwarded but never written to artifacts."""
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            """Keep measured request boundaries exclusive while forwarding normal streaming."""
            if measure_prefix_reuse:
                with measurement_lock:
                    self.relay()
            else:
                self.relay()

        def relay(self):
            """Preserve OpenCode's exact request body and flush each SSE line."""
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            row = {"path": self.path, "request_body": raw.decode(), "request": json.loads(raw),
                   "started": time.time()}
            with lock:
                ordinal = len(rows)
                rows.append(row)
            path = output / f"exchange-{ordinal:04d}.json"
            publish(path, row)
            connection = http.client.HTTPConnection(upstream.hostname, upstream.port or 80, timeout=timeout)
            body = bytearray()
            observation = StreamObservation(output / f"exchange-{ordinal:04d}.progress.json")
            raw_stream = (output / f"exchange-{ordinal:04d}.sse").open("wb")
            observation.flush()
            try:
                before = None
                if measure_prefix_reuse:
                    before = read_stats()
                    idle_snapshot(before)
                headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
                if self.headers.get("Authorization"):
                    headers["Authorization"] = self.headers["Authorization"]
                connection.request("POST", upstream.path.rstrip("/") + self.path, raw, headers)
                response = connection.getresponse()
                self.send_response(response.status)
                self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
                self.send_header("Connection", "close")
                self.end_headers()
                while block := response.readline():
                    body.extend(block)
                    raw_stream.write(block)
                    raw_stream.flush()
                    self.wfile.write(block)
                    self.wfile.flush()
                    observation.record(block)
                row["response"] = {"status": response.status,
                                   "content_type": response.getheader("Content-Type", ""),
                                   "body": body.decode("utf-8")}
                if measure_prefix_reuse:
                    # EOF may precede the server's request-scope destructor.
                    # This short metadata-publication wait starts only after
                    # generation has ended; it never bounds a model turn.
                    deadline = time.monotonic() + 5
                    after = read_stats()
                    while after['requests']['active']:
                        if time.monotonic() >= deadline:
                            raise ValueError('Prefix measurement terminal statistics did not publish')
                        time.sleep(.01)
                        after = read_stats()
                    usage = requested_streaming_usage(row) if row['request'].get('stream') else json.loads(body)['usage']
                    if usage is None:
                        raise ValueError('Prefix measurement requires native wire token usage')
                    measurement = measure_request(before, after, usage)
                    compaction = is_compaction_request(row['request'])
                    previous_compaction = ordinal > 0 and is_compaction_request(rows[ordinal - 1]['request'])
                    measurement.update(exchange=ordinal, kind='compaction' if compaction else
                                       'post_compaction' if previous_compaction else 'ordinary')
                    row['prefix_measurement'] = measurement
            except (OSError, http.client.HTTPException, ValueError, KeyError, TypeError) as error:
                row["error"] = f"proxy: {error}"
                row["partial_response"] = body.decode("utf-8", errors="replace")
            finally:
                raw_stream.close()
                observation.flush(complete="response" in row, error=row.get("error"))
                connection.close()
                self.close_connection = True
                row["elapsed_seconds"] = time.time() - row["started"]
                publish(path, row)

        def log_message(self, *args):
            """The exchange files own diagnostics instead of HTTP access chatter."""

    class JoinedProxyServer(ThreadingHTTPServer):
        """Retire every relay before validating the final exchange evidence."""
        daemon_threads = False

    with JoinedProxyServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", rows
        finally:
            server.shutdown()
            worker.join()


def retire_client_group(process) -> None:
    """Retire one owned process group on cancellation, then reap its client."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


class ClientOwners:
    """Coordinate worker-owned client groups with cancellation on the main thread."""

    def __init__(self):
        """Keep admission and cancellation under one lock, with no global registry."""
        self.lock = threading.Lock()
        self.cancelled = False
        self.processes = {}

    @contextmanager
    def launch(self, command, **kwargs):
        """Publish the child before cancellation can close the admission frontier."""
        with self.lock:
            if self.cancelled:
                raise RuntimeError('OpenCode campaign was cancelled before client admission')
            process = subprocess.Popen(command, **kwargs, start_new_session=True)
            self.processes[process.pid] = process
        try:
            yield process
        finally:
            if process.poll() is None:
                retire_client_group(process)
            with self.lock:
                self.processes.pop(process.pid)

    def cancel(self) -> None:
        """Stop admitting phases and retire active clients before joining workers."""
        with self.lock:
            self.cancelled = True
            processes = tuple(self.processes.values())
        errors = []
        for process in processes:
            try:
                if process.poll() is None:
                    retire_client_group(process)
            except Exception as error:
                errors.append(repr(error))
        if errors:
            raise RuntimeError('OpenCode client retirement failed: ' + '; '.join(errors))


def wait_for_client(process, wall_time_limit=None) -> tuple[int, bool]:
    """Wait for real completion; an optional operator deadline marks interruption.

    Neither turn time nor upstream silence has a default deadline: prefill and
    buffered tool arguments may produce no SSE bytes while the model is active.
    Explicit operator limits remain available independently of live observation.
    """
    try:
        return process.wait(timeout=wall_time_limit), False
    except (subprocess.TimeoutExpired, KeyboardInterrupt, SystemExit) as error:
        # Each client owns a fresh process group, including its shell tools.
        # Cancellation must retire that group before the recording proxy joins
        # outstanding relays; otherwise an interrupted CI job leaves work alive.
        retire_client_group(process)
        if not isinstance(error, subprocess.TimeoutExpired):
            raise
        return process.returncode, True


def run_client_with_prompt(command: list[str], *, prompt: Path, workspace: Path,
                           env: dict[str, str], output: Path, error: Path,
                           wall_time_limit: float | None = None,
                           owners: ClientOwners | None = None) -> tuple[int, bool]:
    """Deliver UTF-8 stdin unchanged, avoiding OpenCode's positional-argument quoting.

    OpenCode run escapes quotes in arguments containing spaces before sending
    them as messages. Its stdin path preserves text. A retained binary input
    file also avoids pipe-size limits and gives every phase an exact input
    artifact, independently of its duration or the size of its output.
    """
    with prompt.open("rb") as stdin, output.open("wb") as stdout, error.open("wb") as stderr:
        with (owners or ClientOwners()).launch(command, cwd=workspace, env=env, stdin=stdin,
                                              stdout=stdout, stderr=stderr) as process:
            return wait_for_client(process, wall_time_limit)


def authenticate_phase_prompt(exchanges: list[dict], expected: str) -> dict:
    """Authenticate the first request's latest user text against the intended phase.

    Later requests may include genuine client compaction messages, so only the
    initial request binds a new user turn. A plain text message or one text part
    is admitted; missing, changed, reordered or non-text input fails explicitly.
    """
    observed = None
    if exchanges:
        request = exchanges[0].get("request")
        messages = request.get("messages", []) if isinstance(request, dict) else []
        if isinstance(messages, list):
            user = next((message for message in reversed(messages)
                         if isinstance(message, dict) and message.get("role") == "user"), {})
            content = user.get("content")
            if isinstance(content, str):
                observed = content
            elif (isinstance(content, list) and len(content) == 1
                  and isinstance(content[0], dict) and content[0].get("type") == "text"
                  and isinstance(content[0].get("text"), str)):
                observed = content[0]["text"]
    passed = observed == expected
    return {"passed": passed,
            "expected_sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
            "observed_sha256": None if observed is None else hashlib.sha256(observed.encode("utf-8")).hexdigest(),
            "error": None if passed else "intended user prompt differs from the first request's latest user text"}


def session_permissions(case: str) -> dict:
    """Permit native shell verification; required file tools are checked in evidence.

    Compound byte inspections such as ``wc -c result.txt && od -c result.txt |
    tail -3`` are legitimate model choices. A shell command allowlist changes
    that conversation and can manufacture permission errors. Exact write/read
    call identities, schema types and resulting bytes remain mandatory.
    """
    return {"external_directory": "allow" if case == "webapp" else "deny",
            "webfetch": "deny", "task": "deny", "question": "deny", "bash": "allow"}


def client_provider_options(base_url: str) -> dict:
    """Disable the client's independent deadlines for observed long-running work.

    OpenCode 1.18.34 defaults headerTimeout and chunkTimeout to 300,000 ms even
    when its total timeout is disabled. A legitimate large prefill or buffered
    tool call can exceed either. The harness owns any explicit operator limit;
    the client must not abort and retry an otherwise healthy request first.
    """
    return {"baseURL": base_url, "apiKey": "local", "timeout": False,
            "headerTimeout": False, "chunkTimeout": False}


def run_session(args, ordinal: int, case: str, content: str) -> dict:
    """Run an isolated conversation, retaining live work until its evidence is copied."""
    directory = args.output / f"session-{ordinal:04d}-{case}"
    directory.mkdir(parents=True)
    # OpenCode discovers repository roots and ancestor AGENTS.md files. A
    # fixture nested in the Llaminar checkout would redirect file tools into
    # that checkout and append its engineering instructions to the real prompt.
    # Retire this directory only after the artifact copy succeeds. Automatic
    # TemporaryDirectory cleanup would erase the model's unfinished app when
    # proxy publication or copying raises ENOSPC. Publish its location before
    # client work so operators can recover interrupted evidence independently.
    workspace = Path(tempfile.mkdtemp(prefix="llaminar-opencode-workspace-"))
    workspace_receipt = {"live_workspace": str(workspace), "copied_and_retired": False}
    publish(directory / "workspace-location.json", workspace_receipt)
    if case == "workflow":
        (workspace / "src").mkdir()
        (workspace / "tests").mkdir()
        (workspace / "src/answer.py").write_text("def answer():\n    return 0\n")
        (workspace / "tests/test_answer.py").write_text(
            "import sys, unittest\nfrom pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
            "from answer import answer\nclass AnswerTest(unittest.TestCase):\n"
            "    def test_answer(self):\n        self.assertEqual(answer(), 123)\n")
    (directory / "expected.bin").write_bytes(content.encode())
    prompts = session_prompts(case, content)
    prompt = "\n\n".join(prompts)
    (directory / "prompt.txt").write_text(prompt)
    started = time.monotonic()
    measure_prefix = getattr(args, 'measure_prefix_reuse', False)
    with recording_proxy(args.base_url, directory, args.request_timeout,
                         measure_prefix_reuse=measure_prefix) as (url, rows):
        config = {"$schema": "https://opencode.ai/config.json", "autoupdate": False,
                  "share": "disabled", "enabled_providers": ["llaminar"], "lsp": False,
                  "formatter": False, **({} if case == "webapp" else {"agent": {"build": {"steps": 10}}}),
                  "permission": session_permissions(case),
                  "provider": {"llaminar": {"npm": "@ai-sdk/openai-compatible",
                      "name": "Llaminar tool stress", "options": client_provider_options(url),
                      "models": {args.model: {"name": args.model, "tool_call": True,
                          "limit": {"context": args.context_length, "output": args.max_tokens}}}}}}
        env = {**os.environ, "OPENCODE_CONFIG_CONTENT": json.dumps(config),
               "OPENCODE_CONFIG_DIR": str(directory / "config"),
               "XDG_CONFIG_HOME": str(directory / "config"),
               "XDG_DATA_HOME": str(directory / "data"), "XDG_CACHE_HOME": str(directory / "cache"),
               "XDG_STATE_HOME": str(directory / "state"), "OPENCODE_DISABLE_AUTOUPDATE": "true",
               "OPENCODE_DISABLE_MODELS_FETCH": "true"}
        env.pop("OPENCODE_CONFIG", None)
        session_id, phases = None, []
        timed_out = False
        returncode = 0
        for phase, phase_prompt in enumerate(prompts):
            phase_started = time.monotonic()
            prompt_path = directory / f"prompt-{phase:02d}.txt"
            prompt_path.write_bytes(phase_prompt.encode("utf-8"))
            command = [args.opencode, "run", "--pure", "--dir", str(workspace), "--format", "json",
                       "--model", f"llaminar/{args.model}", "--title", f"tool-stress-{case}"]
            if session_id:
                command += ["--session", session_id]
            phase_output = directory / f"client-{phase:02d}.jsonl"
            first_exchange = len(rows)
            returncode, phase_interrupted = run_client_with_prompt(
                command, prompt=prompt_path, workspace=workspace, env=env, output=phase_output,
                error=directory / f"client-{phase:02d}.stderr", wall_time_limit=args.session_timeout,
                **({'owners': args.client_owners} if hasattr(args, 'client_owners') else {}))
            timed_out |= phase_interrupted
            prompt_evidence = authenticate_phase_prompt(rows[first_exchange:], phase_prompt)
            phase_events = [json.loads(line) for line in phase_output.read_text().splitlines()]
            normal_finish = any(event.get("type") == "step_finish" and event.get("part", {}).get("reason") == "stop" for event in phase_events)
            identities = {event["sessionID"] for event in phase_events if event.get("sessionID")}
            if len(identities) == 1:
                observed = identities.pop()
                if session_id and session_id != observed:
                    raise ValueError("OpenCode changed session identity during the coding conversation")
                session_id = observed
            phases.append({"phase": phase, "returncode": returncode, "finished": normal_finish,
                           "elapsed_seconds": time.monotonic() - phase_started,
                           "exchanges": len(rows) - first_exchange,
                           "interrupted_by_harness": phase_interrupted, "session_id": session_id,
                           "first_exchange": first_exchange, "prompt": prompt_evidence})
            publish(directory / "phases.json", phases)
            print(f"[opencode-stress] {ordinal} {case} phase {phase + 1}/{len(prompts)} finished={normal_finish}", flush=True)
            if returncode or not normal_finish or not prompt_evidence["passed"] or (case == "webapp" and not session_id):
                break
        (directory / "client.jsonl").write_text("".join(path.read_text() for path in sorted(directory.glob("client-*.jsonl"))))
        (directory / "client.stderr").write_text("".join(path.read_text() for path in sorted(directory.glob("client-*.stderr"))))
    shutil.copytree(workspace, directory / "workspace")
    shutil.rmtree(workspace)
    workspace_receipt["copied_and_retired"] = True
    publish(directory / "workspace-location.json", workspace_receipt)
    result = check_session(directory, rows, returncode)
    if measure_prefix:
        try:
            measurements = [row['prefix_measurement'] for row in rows]
            result['prefix_reuse'] = summarize_prefix_reuse(measurements)
            publish(directory / 'prefix-reuse.json', result['prefix_reuse'])
        except (ValueError, KeyError, TypeError) as error:
            result['protocol_errors'].append(f'prefix measurement: {error}')
    result["protocol_errors"].extend(f"phase {phase['phase'] + 1}: {phase['prompt']['error']}"
                                     for phase in phases if not phase["prompt"]["passed"])
    if case == "webapp":
        acceptance = check_webapp(directory)
        result["webapp_acceptance"] = acceptance
        result["task_errors"].extend(acceptance["errors"])
        if len(phases) != len(prompts):
            result["protocol_errors"].append("coding session did not complete every phase")
    result.update(assessment(result["protocol_errors"], result["task_errors"]))
    result.update(case=case, ordinal=ordinal, timed_out=timed_out, elapsed_seconds=time.monotonic() - started,
                  directory=str(directory), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    completed_phases = sum(row['finished'] and row['returncode'] == 0
        and not row['interrupted_by_harness'] and row['prompt']['passed'] for row in phases)
    result['phase_progress'] = {'expected': len(prompts), 'completed': completed_phases,
        'passed': completed_phases == len(prompts), 'phases': phases}
    publish(directory / "result.json", result)
    print(f"[opencode-stress] {ordinal} {case}: protocol={'PASS' if result['protocol_passed'] else 'FAIL'} "
          f"task={'PASS' if result['task_passed'] else 'FAIL'} {result['errors']}", flush=True)
    return result


def main() -> int:
    """Record version, repetition and concurrency alongside each raw conversation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--opencode", default="opencode")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--measure-prefix-reuse", action="store_true",
                        help="Join each sequential request to /stats and report compaction reuse, TTFT and tier traffic")
    parser.add_argument("--context-length", type=int, required=True,
                        help="The running server context admitted by PhysicalMemoryAuthority")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--gate", choices=("strict", "protocol", "stress"), default="stress",
                        help="Stress requires complete phases, protocol/coverage correctness and model tool errors below 5 percent")
    parser.add_argument("--request-timeout", type=float, default=None,
                        help="Optional upstream inactivity seconds; unset permits long prefill and buffered tool calls")
    parser.add_argument("--session-timeout", type=float, default=None,
                        help="Optional elapsed seconds per turn; unset permits active turns of any duration")
    parser.add_argument("--case", action="append", choices=("webapp",))
    args = parser.parse_args()
    if args.measure_prefix_reuse and args.concurrency != 1:
        parser.error('prefix measurements require concurrency 1 and an exclusive server')
    if any(value <= 0 for value in (args.iterations, args.concurrency, args.context_length, args.max_tokens)) or any(
            value is not None and (not math.isfinite(value) or value <= 0)
            for value in (args.request_timeout, args.session_timeout)):
        parser.error("counts, budgets and timeouts must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    require_ready(args.base_url, args.request_timeout)
    version = subprocess.check_output([args.opencode, "--version"], text=True, timeout=30).strip()
    document = {"schema": 2, "complete": False, "opencode_version": version, "gate": args.gate,
                "model": args.model, "base_url": args.base_url, "concurrency": args.concurrency,
                "measure_prefix_reuse": args.measure_prefix_reuse,
                "context_length": args.context_length, "max_tokens": args.max_tokens,
                "turn_wall_time_limit_seconds": args.session_timeout, "request_inactivity_timeout_seconds": args.request_timeout,
                "iterations": args.iterations, "results": []}
    publish(args.output / "stress.json", document)
    cases = fixture_cases()
    selected = args.case or ["webapp"]
    jobs = [(ordinal, case, cases[case]) for ordinal, case in enumerate(selected * args.iterations)]
    args.client_owners = ClientOwners()
    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    try:
        futures = [executor.submit(run_session, args, *job) for job in jobs]
        for future in as_completed(futures):
            document["results"].append(future.result())
            publish(args.output / "stress.json", document)
    except BaseException:
        # Workers do not receive the main thread's KeyboardInterrupt. Retire
        # their exact clients before executor shutdown waits for those workers.
        args.client_owners.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    document.update(complete=True, **campaign_assessment(document["results"], selected, args.iterations, args.gate))
    publish(args.output / "stress.json", document)
    return 0 if document["gate_passed"] else 1


if __name__ == "__main__":
    def cancel_campaign(signum, frame):
        """Translate container termination into ordered campaign cleanup."""
        raise KeyboardInterrupt('OpenCode campaign received signal ' + str(signum))
    signal.signal(signal.SIGTERM, cancel_campaign)
    raise SystemExit(main())
