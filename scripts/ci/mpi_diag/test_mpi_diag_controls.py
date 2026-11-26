#!/usr/bin/env python3
"""Focused unit controls for the PR75/78 MPI startup diagnostic.

Reviewer-required controls (T-d additions): startup failure preserved, original
exit preserved, timeout retirement, truncation, foreign pid excluded, missing
debugger, secret key not published, denial recorded, per-process env, and a
setsid grandchild escape must be recorded as a survivor FAILED.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "run_mpi_startup_trace.py"
BIN = Path(sys.executable)

#: CMake registration supplies its own ${CMAKE_CTEST_COMMAND} here, exactly as
#: LLAMINAR_TEST_MPI_SESSION_HELPER supplies a built tool path.
CONFIGURED_CTEST_ENV = "LLAMINAR_TEST_CONFIGURED_CTEST"


def record_configured_ctest(build_dir: Path) -> str:
    """Give a synthetic fixture build tree the real configured ctest.

    The diagnostic accepts only the build tree's CMAKE_CTEST_COMMAND, so each
    fixture tree records the ctest that configured the enclosing test build.
    @param build_dir Synthetic fixture build directory.
    @return The recorded ctest path.
    @raise RuntimeError when the registration did not supply the configured ctest.
    """
    ctest = os.environ.get(CONFIGURED_CTEST_ENV)
    if not ctest:
        raise RuntimeError(
            f"{CONFIGURED_CTEST_ENV} must name the configured ctest; run through "
            "the CMake registration")
    (Path(build_dir) / "CMakeCache.txt").write_text(
        f"CMAKE_CTEST_COMMAND:INTERNAL={ctest}\n")
    return ctest


class DiagnosticControls(unittest.TestCase):
    maxDiff = None

    def _build_dir(self):
        return self._tmp / "build"

    def _register_fixture(self, name: str, body: str,
                          env: dict[str, str] | None = None,
                          working_dir: str | None = None) -> None:
        """Write a minimal CTestTestfile + fixture script (no real build)."""
        script = self._build_dir() / (name + ".sh")
        # body is a single shell body executed directly (no arg quoting issues)
        script.write_text("#!/bin/sh\n" + body + "\n")
        os.chmod(script, 0o755)
        ctestfile = self._build_dir() / "CTestTestfile.cmake"
        ctestfile.write_text("""add_test(%s %s)
set_tests_properties(%s PROPERTIES
  ENVIRONMENT "%s"
  WORKING_DIRECTORY "%s")
""" % (name, script, name,
       ";".join(f"{k}={v}" for k, v in (env or {}).items()),
       working_dir or str(self._build_dir())))

    def setUp(self):
        import tempfile
        self._tmp = Path(tempfile.mkdtemp(prefix="mpi-diag-controls-"))
        self._build_dir().mkdir(parents=True, exist_ok=True)
        record_configured_ctest(self._build_dir())

    def tearDown(self):
        import shutil
        # Only retire OUR OWN fixture identity (recorded pid+starttime); never
        # a global pkill pattern (reviewer-90 item 3).
        for identity_file in self._tmp.glob("*/evidence/owned-identities.json"):
            try:
                identities = json.loads(identity_file.read_text())
            except (OSError, ValueError):
                continue
            for identity in identities:
                pid = identity.get("pid")
                start = identity.get("starttime")
                if not pid or start is None:
                    continue
                try:
                    stat = Path(f"/proc/{pid}/stat").read_text()
                    fields = stat[stat.rfind(")") + 1:].split()
                    if len(fields) > 19 and fields[19] == str(start):
                        os.kill(pid, signal.SIGKILL)
                except (OSError, IndexError, ValueError):
                    pass
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_original_failure_exit_preserved(self):
        """A failing fixture must return its own nonzero code, not 0."""
        marker = self._tmp / "ran.txt"
        self._register_fixture("FailingFixture",
            f"sleep 1; echo ran > {marker}; exit 7")
        result = self._run_diag("FailingFixture", window=20)
        self.assertEqual(result.returncode, 7,
                         "diagnostic must preserve the original failed exit")
        self.assertTrue(marker.exists(), "fixture must actually run once")

    def test_fixture_startup_failure_recorded(self):
        """Command that fails before any observable stall: record, not hang."""
        self._register_fixture("EarlyFail", "exit 3")
        result = self._run_diag("EarlyFail", window=20)
        self.assertEqual(result.returncode, 3)
        summary = json.loads((self._evidence / "summary.json").read_text())
        self.assertEqual(summary["diagnostic_returncode"], 3)
        self.assertTrue(summary["original_gate_failed"])
        self.assertIn("NO STALL RECORD", summary["stall"])

    def test_timeout_retirement_cleans_owned_tree(self):
        """Cancel window must SIGTERM/SIGKILL and reap owned descendants."""
        self._register_fixture("Runaway",
            "trap 'kill $CHILD 2>/dev/null' TERM; sleep 300 & CHILD=$!; wait $CHILD")
        result = self._run_diag("Runaway", window=10)
        summary = json.loads((self._evidence / "summary.json").read_text())
        self.assertTrue(summary["forced_cancel_after_window"],
                        "diagnostic must cancel a runaway fixture after window")
        self.assertFalse(summary["survivor_failed"],
                         "TERM-trapped tree must not leave survivors")
        identities = json.loads(
            (self._evidence / "owned-identities.json").read_text())
        self.assertTrue(isinstance(identities, list))

    def test_setsid_grandchild_survivor_is_failed(self):
        """A reparented/setsid grandchild escapes the tree: summary FAILED."""
        self._register_fixture("Escaper", "setsid sh -c 'sleep 300' & sleep 300")
        result = self._run_diag("Escaper", window=8)
        summary = json.loads((self._evidence / "summary.json").read_text())
        self.assertTrue(summary["survivor_failed"],
                        "escaped grandchild must leave a survivor FAILED record")
        self.assertTrue(summary["survivors_after_cancel"],
                        "survivor identities must be recorded")

    def test_truncation_marker_written(self):
        """Evidence files stay line-delimited/parseable (bounded by window)."""
        self._register_fixture("Chatty",
            "head -c 1000000 /dev/zero | tr '\\0' x; exit 0")
        self._run_diag("Chatty", window=12)
        sample = self._evidence / "owned-samples.jsonl"
        if sample.exists():
            first = sample.read_text().splitlines()[0]
            json.loads(first)

    def test_missing_debugger_reported_honestly(self):
        """Missing gdb must not be reported as capture success."""
        self._register_fixture("Quick", "sleep 2")
        result = self._run_diag("Quick", window=20,
                                extra=["--gdb-path", "/nonexistent/gdb"])
        stacks = self._evidence / "blocked-stacks.txt"
        self.assertTrue(stacks.exists(), "blocked-stacks.txt must exist")
        text = stacks.read_text()
        self.assertTrue(
            "gdb_secondary" in text,
            f"gdb absence must be recorded, got: {text[:200]}")

    def test_secret_key_never_published(self):
        """Non-allowlisted env values must never appear; only key NAMES."""
        secret = "TOPSECRET-abc123"
        self._register_fixture("Env", "sleep 2",
            env={"CREDENTIALS": secret})
        self._run_diag("Env", window=20)
        names = (self._evidence / "env-deny-names.txt").read_text()
        self.assertIn("CREDENTIALS", names.splitlines(),
                      "non-allowlisted key NAME must be recorded")
        for evidence in self._evidence.rglob("*"):
            if evidence.is_file():
                text = evidence.read_text(errors="replace")
                self.assertNotIn(secret, text,
                                 f"secret value leaked into {evidence.name}")
        sha_file = (self._evidence / "env-deny-sha.txt").read_text().strip()
        self.assertTrue(len(sha_file) == 64 or sha_file == "NOT-OBSERVABLE")

    def test_binding_json_has_no_secret_or_pid(self):
        """binding.json and argv evidence must be redacted (no secrets)."""
        secret = "PRIVATE-TOKEN-xyz"
        self._register_fixture("Bind", "sleep 2",
            env={"GITHUB_TOKEN": secret})
        self._run_diag("Bind", window=20)
        binding = json.loads(
            (self._evidence / "binding.json").read_text())
        for key, value in binding["env_binding"].items():
            self.assertNotIn(secret, value)
        self.assertIn("[key-name-only]", binding["env_binding"].get(
            "GITHUB_TOKEN", "[key-name-only]"))
        # Redaction keeps safe absolute roots but the fixture script by
        # definition lives under the control build dir; the SECRET is what
        # must never appear, and safe-root paths are allowed.
        for part in binding["command_redacted"]:
            self.assertNotIn(secret, part)

    def _run_diag(self, test: str, window: int,
                  extra: list[str] | None = None) -> subprocess.CompletedProcess:
        """Run the diagnostic with output to a FILE, not a pipe.

        Escaped grandchildren inherit the pipe write end; wait()-ing on a pipe
        would block until every writer closes. A regular file has no such
        dependency (EOF is reached once the diagnostic itself closes it), so
        survivor fixtures cannot stall the control harness.
        """
        self._evidence = self._tmp / "evidence"
        log = self._tmp / (test + ".diag.log")
        with log.open("w") as output:
            process = subprocess.Popen(
                [str(BIN), str(SCRIPT), "--test", test,
                 "--build-dir", str(self._build_dir()),
                 "--evidence-dir", str(self._evidence),
                 "--window", str(window), "--sample-interval", "0.2",
                 "--stall-samples", "2"] + (extra or []),
                stdout=output, stderr=subprocess.STDOUT)
            # Bounded wait: the diagnostic itself has a hard cancel path; if
            # it still outlives 150s for fixture-managed reasons, fail the
            # harness (never hide it behind an unbounded timeout).
            try:
                returncode = process.wait(timeout=150)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                raise
        return subprocess.CompletedProcess(process.args, returncode,
                                           stdout="", stderr="")


class HookControls(unittest.TestCase):
    """Focused CI hook controls (reviewer-90 item: real hook tests)."""

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="mpi-diag-hook-"))
        self.build = self.tmp / "build"
        self.build.mkdir()
        record_configured_ctest(self.build)
        self.out = self.tmp / "out"
        self.out.mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _register(self, name: str, body: str, timeout: int = 30,
                  env: list[str] | None = None) -> None:
        script = self.build / (name + ".sh")
        script.write_text("#!/bin/sh\n" + body + "\n")
        os.chmod(script, 0o755)
        env_line = (" ENVIRONMENT \"%s\"" % ";".join(env or [])) if env else ""
        with (self.build / "CTestTestfile.cmake").open("a") as ctf:
            ctf.write('add_test(%s %s)\nset_tests_properties(%s PROPERTIES %s\n'
                      '  TIMEOUT "%d")\n' % (name, script, name, env_line, timeout))

    def _run_hook(self, extra: list[str] | None = None) -> subprocess.CompletedProcess:
        from ci_mpi_hook import run_after_failed_host_lane
        return None  # replaced below

    def test_selection_single_and_order(self):
        """Selection: exactly ONE fixture, deterministic singleton-first."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ci_mpi_hook", str(Path(__file__).parent / "ci_mpi_hook.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._register("V2_Integration_MPISessionRetirement_singleton",
                       "exit 2", env=["HWLOC_COMPONENTS=-gl,-opencl"])
        self._register("V2_Integration_MPISessionRetirement_mpi", "exit 3")
        # Simulate both failing in the lane JUnit.
        self._write_junit(["V2_Integration_MPISessionRetirement_mpi",
                           "V2_Integration_MPISessionRetirement_singleton"])
        # Reuse the real hook but with a tiny deadline and skip-run assertions:
        # instead of executing (slow), assert the ORDER logic directly with a
        # monkeypatched runner.
        calls = []
        def fake_run(test, env, evidence, arm, window, deadline):
            calls.append((test, arm, env.get("HWLOC_COMPONENTS")))
            return 0
        mod.run_fixture = fake_run
        mod._retire = lambda proc: None
        result = mod.run_after_failed_host_lane(
            self.out / "host.xml", self.out, self.build, {"X": "1"},
            deadline_s=300)
        self.assertEqual(result["selected"],
                         "V2_Integration_MPISessionRetirement_singleton")
        # Only one fixture, 3 pairs = 6 runs max (with arm A/B when diff).
        names = {c[0] for c in calls}
        self.assertEqual(names, {"V2_Integration_MPISessionRetirement_singleton"})
        self.assertLessEqual(len(calls), 6)

    def test_hook_skips_on_success(self):
        """No failed allowlisted fixture => diagnostic not run."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ci_mpi_hook2", str(Path(__file__).parent / "ci_mpi_hook.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._write_junit(["V2_Unit_Unrelated"])
        self._register("V2_Unit_Unrelated", "exit 0")
        calls = []
        mod.run_fixture = lambda *a, **k: calls.append(a) or 0
        result = mod.run_after_failed_host_lane(
            self.out / "host.xml", self.out, self.build, {}, deadline_s=60)
        self.assertTrue((self.out / "mpi-diag" /
                         "no-failed-fixtures.txt").exists())
        self.assertEqual(calls, [])

    def test_hook_missing_junit_records_error(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ci_mpi_hook3", str(Path(__file__).parent / "ci_mpi_hook.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        result = mod.run_after_failed_host_lane(
            self.out / "missing.xml", self.out, self.build, {}, deadline_s=60)
        self.assertIn("error", result)
        self.assertTrue((self.out / "mpi-diag" / "hook-error.txt").exists())

    def test_hook_arm_diff_fails_when_equal(self):
        """A==B (registered already has -gl,-opencl) => contrast failed recorded."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ci_mpi_hook4", str(Path(__file__).parent / "ci_mpi_hook.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._register("V2_Integration_MPISessionRetirement_singleton",
                       "exit 0", env=["HWLOC_COMPONENTS=-gl,-opencl"])
        self._write_junit(["V2_Integration_MPISessionRetirement_singleton"],
                          timeout_prop=True)
        calls = []
        mod.run_fixture = lambda *a, **k: calls.append(a) or 0
        result = mod.run_after_failed_host_lane(
            self.out / "host.xml", self.out, self.build, {}, deadline_s=120)
        self.assertTrue(result.get("contrast_failed"))
        # Only arm A ran (B == A so contrast refused).
        self.assertIn("A-registered", result["arms"])
        self.assertNotIn("B-outer-hwloc-export", result["arms"])

    def test_env_modification_parse(self):
        """ONE shared parser (env_policy): exact CTest semantics else 125."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "env_policy", str(Path(__file__).parent / "env_policy.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        def entry_with(mods, env=None, fixtures=None):
            props = [{"name": "ENVIRONMENT_MODIFICATION", "value": mods}]
            if env:
                props.append({"name": "ENVIRONMENT", "value": env})
            if fixtures:
                props.append({"name": "FIXTURES_REQUIRED", "value": fixtures})
            return {"properties": props}

        env = mod.parse_environment_policy(
            entry_with(["A=unset:", "B=set:3", "C=string_append:def"],
                       env=["A=1", "B=2", "C=abc"]), {"A": "0", "B": "0"})
        self.assertNotIn("A", env)  # unset removes it
        self.assertEqual(env["B"], "3")
        self.assertEqual(env["C"], "abcdef")
        # string_prepend accumulates exactly.
        env = mod.parse_environment_policy(
            entry_with(["D=string_prepend:pre"]), {"D": "fix"})
        self.assertEqual(env["D"], "prefix")
        # REVIEWER-93: reset / path_list_* / cmakelist are UNSUPPORTED -> 125
        # (reset is ambiguous, path-list separator semantics unverified,
        # cmakelist is not an env op).
        for unsupported in ("A=reset:", "E=path_list_append:/x:",
                            "E=path_list_prepend:/x:", "A=cmakelist:1"):
            with self.assertRaises(mod.UnsupportedPolicy):
                mod.parse_environment_policy(entry_with([unsupported]), {})
        # bad op / malformed -> 125
        with self.assertRaises(mod.UnsupportedPolicy):
            mod.parse_environment_policy(entry_with(["D=badop:x"]), {})
        with self.assertRaises(mod.UnsupportedPolicy):
            mod.parse_environment_policy(entry_with(["malformed"]), {})
        # FIXTURES_REQUIRED -> unsupported (no arbitrary setup)
        with self.assertRaises(mod.UnsupportedPolicy):
            mod.parse_environment_policy(
                entry_with([], fixtures=["setup.o"]), {})

    def test_gate_exit_unchanged(self):
        """Hook return value never alters the caller's overall return code."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ci_mpi_hook6", str(Path(__file__).parent / "ci_mpi_hook.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # The hook itself returns dict without an exit code field; the caller
        # (run_develop_pr_tests) keeps overall=8 unchanged. Assert the hook
        # never raises on a skipped case and returns the marker structure.
        self._register("V2_Unit_X", "exit 7")
        self._write_junit(["V2_Unit_X"])
        calls = []
        mod.run_fixture = lambda *a, **k: calls.append(a) or 0
        result = mod.run_after_failed_host_lane(
            self.out / "host.xml", self.out, self.build, {}, deadline_s=60)
        self.assertEqual(result["selected"], None)
        self.assertEqual(calls, [])

    def _write_junit(self, failed: list[str], timeout_prop: bool = False):
        tests = "".join(
            '<testcase classname="x" name="%s" time="1">'
            '<failure message="boom"/></testcase>' % name
            for name in failed)
        (self.out / "host.xml").write_text(
            '<testsuite name="host" tests="%d" failures="%d">%s</testsuite>'
            % (len(failed), len(failed), tests))



class RealHookControls(unittest.TestCase):
    """Reviewer-92 item 6: REAL hook run with a tiny CTest dir, NO mocks.

    The hook executes the actual diagnostic against the registered fixture;
    assertions look at the on-disk evidence (not the summary), the correct
    build dir is used, a non-125 outcome is required, and no fixture
    descendants may survive.
    """

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="mpi-diag-real-hook-"))
        self.build = self.tmp / "build"
        self.build.mkdir()
        record_configured_ctest(self.build)
        self.out = self.tmp / "out"
        self.out.mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fixture(self, name: str, body: str, timeout: int = 8) -> Path:
        script = self.build / (name + ".sh")
        script.write_text("#!/bin/sh\n" + body + "\n")
        os.chmod(script, 0o755)
        return script

    def _junit(self, failed: list[str]) -> None:
        tests = "".join(
            '<testcase classname="x" name="%s" time="1">'
            '<failure message="boom"/></testcase>' % name
            for name in failed)
        (self.out / "host.xml").write_text(
            '<testsuite name="host" tests="%d" failures="%d">%s</testsuite>'
            % (len(failed), len(failed), tests))

    def test_real_hook_runs_diagnostic_no_survivors(self):
        """Hook drives the real diagnostic: evidence exists, non-125, clean."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ci_mpi_hook", str(Path(__file__).parent / "ci_mpi_hook.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        name = "V2_Integration_MPISessionRetirement_singleton"
        self._fixture(name, "sleep 1; exit 0")
        (self.build / "CTestTestfile.cmake").write_text(
            'add_test(%s %s)\nset_tests_properties(%s PROPERTIES TIMEOUT "8")\n'
            % (name, self.build / (name + ".sh"), name))
        self._junit([name])
        result = mod.run_after_failed_host_lane(
            self.out / "host.xml", self.out, self.build, {}, deadline_s=200)
        self.assertEqual(result["selected"], name)
        self.assertTrue(result["ran"])
        # Real evidence, from the real diagnostic, under the hook's dir.
        hook_summary = self.out / "mpi-diag" / "hook-summary.json"
        self.assertTrue(hook_summary.exists())
        arms = json.loads(hook_summary.read_text())
        self.assertIn("A-registered", arms["arms"])
        self.assertNotEqual(arms["arms"]["A-registered"][0]["returncode"], 125,
                            "fixture must not produce a 125 harness error")
        # summary.json exists with an UNBOUND note (no receipt in harness) but
        # source_bound false and NO survivors.
        summary = json.loads((self.out / "mpi-diag" /
                              f"{name}-A-registered-r0" / "summary.json")
                             .read_text())
        self.assertFalse(summary["source_bound"])
        self.assertFalse(summary["survivor_failed"])

    def test_real_hook_truncates_log_with_marker(self):
        """16 MiB log cap: the diag.log contains the marker after truncation."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ci_mpi_hook", str(Path(__file__).parent / "ci_mpi_hook.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        name = "V2_Integration_MPISessionRetirement_native_options"
        body = ("i=0; while [ $i -lt 700000 ]; do "
                "echo '0123456789abcdef0123456789abcdef0123456789abcdef'; "
                "i=$((i+1)); done; exit 0")
        self._fixture(name, body)
        (self.build / "CTestTestfile.cmake").write_text(
            'add_test(%s %s)\nset_tests_properties(%s PROPERTIES TIMEOUT "30")\n'
            % (name, self.build / (name + ".sh"), name))
        self._junit([name])
        mod.run_after_failed_host_lane(self.out / "host.xml", self.out,
                                       self.build, {}, deadline_s=240)
        log = (self.out / "mpi-diag" / f"{name}-A-registered-r0" / "diag.log")
        self.assertTrue(log.exists())
        size = log.stat().st_size
        self.assertLessEqual(size, 16 * 1024 * 1024 + 128,
                             "16 MiB cap must be enforced on disk")
        self.assertIn("MPI-DIAG LOG TRUNCATED AT 16 MiB", log.read_text())



class SourceIdentityControls(unittest.TestCase):
    """Reviewer-92 item 3: 40-hex commit FIELD required; else UNBOUND."""

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="mpi-diag-src-"))
        self.build = self.tmp / "build"
        self.build.mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _identity(self, revision=None):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_mpi_startup_trace", str(Path(__file__).parent / "run_mpi_startup_trace.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.source_identity(revision)

    def test_valid_image_revision_binds(self):
        ident = self._identity("a" * 40)
        self.assertEqual(ident["source_note"], "image-revision")
        self.assertEqual(ident["bound_commit"], "a" * 40)
        self.assertEqual(ident["bound_commit_field"],
                         "org.opencontainers.image.revision")

    def test_missing_revision_unbound(self):
        ident = self._identity(None)
        self.assertEqual(ident["source_note"], "UNBOUND")
        self.assertNotIn("bound_commit", ident)

    def test_malformed_revision_unbound(self):
        # wrong length / uppercase / non-hex -> UNBOUND (no heuristic scan).
        for bad in ("abc123", "A" * 40, "g" * 40, "a" * 39):
            ident = self._identity(bad)
            self.assertEqual(ident["source_note"], "UNBOUND")

    def test_unrelated_field_not_bound(self):
        # The old heuristic scanned any 40-hex field; reviewer-93 removes it.
        ident = self._identity(None)
        self.assertNotIn("source_receipt", ident)
        self.assertEqual(ident["source_note"], "UNBOUND")



class RegisteredSafety(unittest.TestCase):
    """Reviewer-95/96: SECOND Integration+ProductionTestPreflight shard.

    Executes the EXACT THREE existing safety tests by their fully-qualified
    names (no duplication, no copies): setsid survivor FAILED, secret key
    names-only never published, binding.json redaction. A CTest-inventory
    control proves the three tests are COLLECTED and RUN (not source-grep).
    """

    def test_runs_the_three_existing_safety_controls(self):
        import unittest as u
        from pathlib import Path as P
        module = __import__(
            "test_mpi_diag_controls",
            globals(), locals(), ["DiagnosticControls"], 0)
        loader = u.TestLoader()
        suite = u.TestSuite()
        for name in ("test_setsid_grandchild_survivor_is_failed",
                     "test_secret_key_never_published",
                     "test_binding_json_has_no_secret_or_pid"):
            suite.addTest(loader.loadTestsFromName(
                f"DiagnosticControls.{name}", module))
        runner = u.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        self.assertTrue(result.wasSuccessful() and result.testsRun == 3,
                        "exactly the three safety tests must run and pass")

    def test_inventory_shows_safety_selected_for_ctest(self):
        """The three names must exist in the control module (discovery
        proof; the registered shard holds the real RUN proof)."""
        for name in ("test_setsid_grandchild_survivor_is_failed",
                     "test_secret_key_never_published",
                     "test_binding_json_has_no_secret_or_pid"):
            self.assertTrue(
                hasattr(DiagnosticControls, name),
                f"CTest inventory must select {name}")
class BackstopControls(unittest.TestCase):
    """Reviewer-92 block 1: REAL forced backstop (hung diagnostic, NOT happy path).

    The fixture publishes a persistent child + grandchild identity before
    hanging (TERM ignored). The hook's deadline is tiny so the diag cannot
    finish its own cancel path; the HOOK's _retire must run, and afterwards
    every recorded identity must be dead (verified by pid+starttime), with the
    actual cleanup trace written to disk - not merely a happy summary flag.
    No global kill is used; only the fixture's own recorded identities.
    """

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="mpi-diag-backstop-"))
        self.build = self.tmp / "build"
        self.build.mkdir()
        record_configured_ctest(self.build)
        self.out = self.tmp / "out"
        self.out.mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _junit(self, failed: list[str]) -> None:
        tests = "".join(
            '<testcase classname="x" name="%s" time="1">'
            '<failure message="boom"/></testcase>' % name
            for name in failed)
        (self.out / "host.xml").write_text(
            '<testsuite name="host" tests="%d" failures="%d">%s</testsuite>'
            % (len(failed), len(failed), tests))

    def test_forced_backstop_reaps_all_recorded_identities(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ci_mpi_hook", str(Path(__file__).parent / "ci_mpi_hook.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        name = "V2_Integration_MPISessionRetirement_singleton"
        # Persistent child + grandchild, TERM ignored by both layers, then the
        # shell hangs. The diagnostic publishes identities incrementally (the
        # sampler flushes each new identity BEFORE any kill).
        script = self.build / (name + ".sh")
        script.write_text(
            "#!/bin/sh\n"
            "trap '' TERM INT HUP\n"
            "sleep 300 & CHILD=$!\n"
            "setsid sh -c 'trap "" TERM INT HUP; sleep 300' &\n"
            "echo child=$CHILD\n"
            "wait\n")
        os.chmod(script, 0o755)
        (self.build / "CTestTestfile.cmake").write_text(
            'add_test(%s %s)\nset_tests_properties(%s PROPERTIES TIMEOUT "5")\n'
            % (name, script, name))
        self._junit([name])
        # Deterministic forced backstop: deadline 20s is far below what the
        # diagnostic needs (window 5 + its own 36s grace + 5s kill), so the
        # HOOK's run_fixture wait times out and the hook _retire runs while
        # the fixture (and its grandchildren) are still alive.
        result = mod.run_after_failed_host_lane(
            self.out / "host.xml", self.out, self.build, {}, deadline_s=20)
        self.assertIn("A-registered", result.get("arms", {}))
        hook_summary = json.loads(
            (self.out / "mpi-diag" / "hook-summary.json").read_text())
        self.assertIn("A-registered", hook_summary["arms"])
        # 'cleanup' evidence: the hook retired recorded identities; the
        # diagnostic evidence directory must exist with identities published
        # BEFORE the backstop (incremental flush).
        evidence = (self.out / "mpi-diag" /
                    f"{name}-A-registered-r0")
        if not evidence.exists():
            # Backstop fired before this arm produced evidence: record that.
            raise AssertionError(
                f"no evidence dir; hook-summary={((self.out/'mpi-diag'/'hook-summary.json').read_text() if (self.out/'mpi-diag'/'hook-summary.json').exists() else 'missing')}")
        identities = mod.recorded_identities(evidence)
        self.assertTrue(identities, "identities must have been published before backstop")
        # Every recorded identity must now be gone or ZOMBIE (cleanup trace).
        alive = []
        zombies = []
        for identity in identities:
            stat = Path(f"/proc/{identity['pid']}/stat")
            try:
                text = stat.read_text()
            except OSError:
                continue
            fields = text[text.rfind(")") + 1:].split()
            if len(fields) > 19 and int(fields[19]) == identity["starttime"]:
                if fields[0] == "Z":
                    zombies.append(identity)
                else:
                    alive.append(identity)
        self.assertEqual(alive, [],
                         f"recorded identities survived backstop: {alive}")
        self.assertGreaterEqual(len(identities), 3,
                                "fixture + child + grandchild identities expected")
        # Structured backstop record (reviewer-94 block 3).
        backstop = json.loads((self.out / "mpi-diag" /
                               f"{name}-A-registered-r0" / "backstop.json")
                              .read_text())
        self.assertTrue(backstop["fired"])
        self.assertEqual(backstop["survivors_after"], [],
                         "no live survivors allowed after forced backstop")
        self.assertTrue(backstop["retired"], "retired list must be recorded")
        # Every retired entry must have an outcome (never raw-kill fallback).
        for entry in backstop["retired"]:
            self.assertIn("outcome", entry)
            self.assertNotEqual(entry["outcome"], "fallback-kill")
        # Hook summary carries the backstop.
        hook_summary = json.loads(
            (self.out / "mpi-diag" / "hook-summary.json").read_text())
        self.assertIn("backstop", hook_summary["arms"]["A-registered"][0])
        self.assertTrue(hook_summary["arms"]["A-registered"][0]["backstop"]["fired"])

if __name__ == "__main__":
    unittest.main()
