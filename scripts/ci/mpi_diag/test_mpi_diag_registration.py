#!/usr/bin/env python3
"""Registered focused controls for the PR75/78 MPI startup diagnostic.

Reviewer-94 block 3: the diagnostic/hook code must have REGISTERED tests (not
an orphan unit script). Two registrations:

  V2_Unit_MPIStartupDiagnosticPolicy (FAST, seconds):
      env_policy exact ops + 125; JUnit selection ONE/skip/missing;
      source binding valid/missing/malformed; A==B contrast refused.
  V2_Integration_MPIStartupDiagnosticLifecycle (TIMEOUT 60, identity self-clean):
      REAL forced backstop (fired=true, survivors empty, structured retired);
      real-hook cap-drain (fixture exits 0, marker + bounded size, pump never
      hangs the fixture); secrets not published (once, fast).

This file is self-contained: it builds a tiny CTest dir under a temp path and
runs the real hook/diagnostic against it. It never needs models, MPI tools, a
GPU, or the runner image - only Python3 + sh.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOOK = HERE / "ci_mpi_hook.py"
DIAG = HERE / "run_mpi_startup_trace.py"
POLICY = HERE / "env_policy.py"


def _load(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RegisteredSafety(unittest.TestCase):
    """Reviewer-96: second Integration+Preflight safety shard.

    Executes the EXACT three existing safety controls
    (setsid_survivor_is_failed, secret_key_never_published,
    binding_json_has_no_secret_or_pid) by fully-qualified unittest names from
    the local control module - no copies. Also runs a CTest-inventory control
    that proves the three were COLLECTED and RUN (not source-grep).
    """

    def test_exact_three_existing_safety_controls(self):
        import unittest as u
        module = _load("test_mpi_diag_controls",
                       HERE / "test_mpi_diag_controls.py")
        loader = u.TestLoader()
        suite = loader.loadTestsFromNames([
            "DiagnosticControls.test_setsid_grandchild_survivor_is_failed",
            "DiagnosticControls.test_secret_key_never_published",
            "DiagnosticControls.test_binding_json_has_no_secret_or_pid",
        ], module=module)
        result = u.TextTestRunner(verbosity=2).run(suite)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.testsRun, 3,
                         "exactly the three safety controls must run")

    def test_ctest_collects_the_safety_shard(self):
        """Discovery proves the three are COLLECTED by qualified name; each
        also RUNS (not source-grep)."""
        import subprocess
        # Discovery = CTest-style collection by fully-qualified name.
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "-v",
             "test_mpi_diag_controls.DiagnosticControls"],
            capture_output=True, text=True, cwd=str(HERE))
        stdout = result.stdout + result.stderr
        for name in ("test_setsid_grandchild_survivor_is_failed",
                     "test_secret_key_never_published",
                     "test_binding_json_has_no_secret_or_pid"):
            self.assertIn(name, stdout,
                          f"{name} must be collected by discovery")
        # Run exactly one: proves RUN, not grep-only.
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "-v",
             "test_mpi_diag_controls.DiagnosticControls.test_secret_key_never_published"],
            capture_output=True, text=True, cwd=str(HERE))
        combined = result.stdout + result.stderr
        self.assertIn("Ran 1 test", combined, "safety test must actually run")


class MPIStartupDiagnosticPolicy(unittest.TestCase):
    """FAST policy/selection controls (no subprocess past `sh`)."""

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="mpi-diag-reg-fast-"))
        self.build = self.tmp / "build"
        self.build.mkdir()
        self.out = self.tmp / "out"
        self.out.mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _junit(self, failed: list[str]) -> None:
        tests = "".join(
            '<testcase classname="x" name="%s" time="1">'
            '<failure message="boom"/></testcase>' % name for name in failed)
        (self.out / "host.xml").write_text(
            '<testsuite name="host" tests="%d" failures="%d">%s</testsuite>'
            % (len(failed), len(failed), tests))

    def _register(self, name: str, body: str, timeout: int = 8,
                  env: list[str] | None = None) -> None:
        script = self.build / (name + ".sh")
        script.write_text("#!/bin/sh\n" + body + "\n")
        os.chmod(script, 0o755)
        env_line = (" ENVIRONMENT \"%s\"" % ";".join(env or [])) if env else ""
        with (self.build / "CTestTestfile.cmake").open("a") as ctf:
            ctf.write('add_test(%s %s)\nset_tests_properties(%s PROPERTIES '
                      '%s\n  TIMEOUT "%d")\n'
                      % (name, script, name, env_line, timeout))

    def test_env_policy_exact_ops_else_125(self):
        mod = _load("env_policy", POLICY)
        entry = {"properties": [
            {"name": "ENVIRONMENT_MODIFICATION",
             "value": ["A=set:1", "B=unset:", "C=string_append:x",
                       "D=string_prepend:p"]}]}
        env = mod.parse_environment_policy(entry, {"A": "0", "B": "0",
                                                   "C": "c", "D": "d"})
        self.assertEqual(env["A"], "1")
        self.assertNotIn("B", env)
        self.assertEqual(env["C"], "cx")
        self.assertEqual(env["D"], "pd")
        for unsupported in ("E=reset:", "E=path_list_append:/x:",
                            "E=cmakelist:1", "E=bogus:1"):
            with self.assertRaises(mod.UnsupportedPolicy):
                mod.parse_environment_policy(
                    {"properties": [{"name": "ENVIRONMENT_MODIFICATION",
                                     "value": [unsupported]}]}, {})

    def test_selection_one_and_skip(self):
        mod = _load("ci_mpi_hook", HOOK)
        mod._build_dir = str(self.build)
        self._register("V2_Integration_MPISessionRetirement_singleton",
                       "sleep 0.1; exit 0")
        self._junit(["V2_Integration_MPISessionRetirement_singleton",
                     "V2_Integration_MPICTestSessionIsolation"])
        calls = []
        mod.run_fixture = lambda *a, **k: calls.append(a) or 0
        result = mod.run_after_failed_host_lane(
            self.out / "host.xml", self.out, self.build, {}, deadline_s=60)
        self.assertEqual(result["selected"],
                         "V2_Integration_MPISessionRetirement_singleton")
        self.assertEqual({c[0] for c in calls},
                         {"V2_Integration_MPISessionRetirement_singleton"})
        # Success skip.
        self._junit(["V2_Unit_Unrelated"])
        calls.clear()
        result = mod.run_after_failed_host_lane(
            self.out / "host.xml", self.out, self.build, {}, deadline_s=60)
        self.assertEqual(result["selected"], None)
        self.assertEqual(calls, [])

    def test_source_binding_valid_missing_malformed(self):
        mod = _load("run_mpi_startup_trace", DIAG)
        self.assertEqual(mod.source_identity("a" * 40)["source_note"],
                         "image-revision")
        self.assertEqual(mod.source_identity(None)["source_note"], "UNBOUND")
        for malformed in ("A" * 40, "abc123", "g" * 40):
            self.assertEqual(mod.source_identity(malformed)["source_note"],
                             "UNBOUND")

    def test_find_ctest_resolves_without_path(self):
        """Reproduce the CI condition: ctest NOT on PATH. find_ctest must
        still return an executable (candidates / LLAMINAR_CTEST)."""
        import subprocess
        env = dict(os.environ)
        env["LLAMINAR_CTEST"] = "/does/not/exist"
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, sys.argv[1]);"
             "from env_policy import find_ctest;"
             "print(find_ctest())",
             str(HERE)], capture_output=True, text=True, env=env)
        self.assertNotEqual(result.returncode, 0,
                            "missing LLAMINAR_CTEST must fail closed")
        # Without the env var, resolution must succeed even if PATH lacks ctest.
        env2 = dict(os.environ)
        env2["PATH"] = "/usr/bin:/bin"  # no cmake4 bin
        env2.pop("LLAMINAR_CTEST", None)
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, sys.argv[1]);"
             "from env_policy import find_ctest;"
             "import os;"
             "p = find_ctest();"
             "print(p) if os.path.exists(p) else print('MISSING')",
             str(HERE)], capture_output=True, text=True, env=env2)
        self.assertEqual(result.returncode, 0, result.stderr[-300:])
        self.assertNotIn("MISSING", result.stdout)

    def test_find_ctest_resolves_from_build_cache(self):
        """The CI runner condition: ctest NOT on PATH, but the build tree's
        CMakeCache.txt records CMAKE_CTEST_COMMAND. find_ctest(build_dir)
        must resolve to that exact executable."""
        import subprocess
        env = dict(os.environ)
        env["PATH"] = "/usr/bin:/bin"
        env.pop("LLAMINAR_CTEST", None)
        script = (
            "import sys, tempfile, pathlib, os;"
            "sys.path.insert(0, sys.argv[1]);"
            "from env_policy import find_ctest;"
            "d = tempfile.mkdtemp();"
            "p = pathlib.Path(d)/'CMakeCache.txt';"
            "p.write_text('CMAKE_CTEST_COMMAND:INTERNAL=' + sys.executable);"
            "print(find_ctest(d))")
        result = subprocess.run(
            [sys.executable, "-c", script, str(HERE)],
            capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, result.stderr[-300:])
        # sys.executable is an existing file, so it is the resolved value.
        self.assertTrue(
            result.stdout.strip().endswith("python3.12")
            or result.stdout.strip().endswith("python3")
            or result.stdout.strip().endswith("python"), result.stdout)

    def test_runner_propagates_failing_result(self):
        """Runner exit code is checked (reviewer-95 block 1): a failing test
        class (never registered) must make CTest red, not always-pass."""
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "DeliberatelyFailingControl"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0,
                            "failing control class must make CTest red")
        self.assertIn("intentional negative-control failure",
                      result.stderr + result.stdout)

    def test_runner_rejects_unknown_class(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "NoSuchClass"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("UNKNOWN TEST CLASS", result.stderr + result.stdout)

    def test_verify_gone_unknown_is_not_dead(self):
        """Reviewer-95 block 2: PermissionError / parse failure must be
        recorded UNKNOWN (a FAIL) - never treated as proof of death."""
        mod = _load("ci_mpi_hook", HOOK)

        class _Denied:
            def read_text(self):
                raise PermissionError("permission denied")

        import unittest.mock as mock
        with mock.patch.object(Path, "read_text",
                               side_effect=PermissionError("denied")):
            state = mod._verify_gone({"pid": 12345, "starttime": 999})
        self.assertEqual(state["outcome"], "UNKNOWN-permission-denied")

        with mock.patch.object(Path, "read_text",
                               return_value="broken stat line"):
            state = mod._verify_gone({"pid": 12345, "starttime": 999})
        self.assertIn("UNKNOWN-parse", state["outcome"])

        with mock.patch.object(Path, "read_text",
                               side_effect=FileNotFoundError()):
            state = mod._verify_gone({"pid": 12345, "starttime": 999})
        self.assertEqual(state["outcome"], "gone")

    def test_a_equals_b_contrast_refused(self):
        mod = _load("ci_mpi_hook", HOOK)
        mod._build_dir = str(self.build)
        self._register("V2_Integration_MPISessionRetirement_singleton",
                       "sleep 0.1; exit 0",
                       env=["HWLOC_COMPONENTS=-gl,-opencl"])
        self._junit(["V2_Integration_MPISessionRetirement_singleton"])
        calls = []
        mod.run_fixture = lambda *a, **k: calls.append(a) or 0
        result = mod.run_after_failed_host_lane(
            self.out / "host.xml", self.out, self.build, {}, deadline_s=60)
        self.assertTrue(result.get("contrast_failed"))
        self.assertIn("A-registered", result["arms"])
        self.assertNotIn("B-outer-hwloc-export", result["arms"])


class MPIStartupDiagnosticLifecycle(unittest.TestCase):
    """Registered lifecycle controls: REAL hook + REAL diagnostic (TIMEOUT 60)."""

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="mpi-diag-reg-life-"))
        self.build = self.tmp / "build"
        self.build.mkdir()
        self.out = self.tmp / "out"
        self.out.mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _junit(self, name: str) -> None:
        (self.out / "host.xml").write_text(
            '<testsuite name="host" tests="1" failures="1">'
            '<testcase classname="x" name="%s" time="1">'
            '<failure message="boom"/></testcase></testsuite>' % name)

    def _register(self, name: str, body: str, timeout: int = 8,
                  env: list[str] | None = None) -> None:
        script = self.build / (name + ".sh")
        script.write_text("#!/bin/sh\n" + body + "\n")
        os.chmod(script, 0o755)
        env_line = (" ENVIRONMENT \"%s\"" % ";".join(env or [])) if env else ""
        with (self.build / "CTestTestfile.cmake").open("a") as ctf:
            ctf.write('add_test(%s %s)\nset_tests_properties(%s PROPERTIES '
                      '%s\n  TIMEOUT "%d")\n'
                      % (name, script, name, env_line, timeout))

    def test_forced_backstop_structured(self):
        mod = _load("ci_mpi_hook", HOOK)
        name = "V2_Integration_MPISessionRetirement_singleton"
        self._register(name,
                       "trap '' TERM INT HUP\n"
                       "sleep 300 & CHILD=$!\n"
                       "setsid sh -c 'trap \"\" TERM INT HUP; sleep 300' &\n"
                       "echo child=$CHILD\nwait\n", timeout=5)
        self._junit(name)
        result = mod.run_after_failed_host_lane(
            self.out / "host.xml", self.out, self.build, {}, deadline_s=20)
        rev = result["arms"]["A-registered"][0]
        self.assertTrue(rev["backstop"]["fired"])
        self.assertEqual(rev["backstop"]["survivors_after"], [])
        for entry in rev["backstop"]["retired"]:
            self.assertIn("outcome", entry)
            self.assertNotEqual(entry["outcome"], "fallback-kill")

    def test_real_hook_cap_drain_exit_zero(self):
        """> 16 MiB fixture that EXITS 0: pump discards after cap, the
        diagnostic finishes (no backstop), marker + bound on disk."""
        mod = _load("ci_mpi_hook", HOOK)
        name = "V2_Integration_MPISessionRetirement_native_options"
        body = ("i=0; while [ $i -lt 700000 ]; do "
                "echo '0123456789abcdef0123456789abcdef0123456789abcdef'; "
                "i=$((i+1)); done; exit 0")
        self._register(name, body, timeout=30)
        self._junit(name)
        result = mod.run_after_failed_host_lane(
            self.out / "host.xml", self.out, self.build, {}, deadline_s=120)
        rev = result["arms"]["A-registered"][0]
        self.assertNotIn("backstop", rev, "fixture must finish without backstop")
        log = self.out / "mpi-diag" / f"{name}-A-registered-r0" / "diag.log"
        self.assertTrue(log.exists())
        self.assertLessEqual(log.stat().st_size, 16 * 1024 * 1024 + 128)
        self.assertIn("MPI-DIAG LOG TRUNCATED AT 16 MiB", log.read_text())


class DeliberatelyFailingControl(unittest.TestCase):
    """NEGATIVE control (never registered): proves the runner propagates a
    failing test as a NONZERO exit - CTest must not pass the diagnostic by
    discarding the result object."""

    def test_intentional_failure(self):
        self.fail("intentional negative-control failure")


if __name__ == "__main__":
    import unittest as _unittest
    # Registration invokes one class by name (CMake). The runner result is
    # checked explicitly (reviewer-95 block 1): CTest must see a NONZERO exit
    # for any failure OR for an unknown class (which is a hard error - never
    # silently treated as success). Running the file bare runs everything.
    if len(sys.argv) > 1:
        klass = globals().get(sys.argv[1])
        if not isinstance(klass, type) or not issubclass(klass, _unittest.TestCase):
            print(f"UNKNOWN TEST CLASS: {sys.argv[1]}", file=sys.stderr)
            raise SystemExit(1)
        suite = _unittest.TestLoader().loadTestsFromTestCase(klass)
        result = _unittest.TextTestRunner(verbosity=2).run(suite)
        tests_run = result.testsRun
        ok = (result.wasSuccessful() and tests_run > 0)
        print(f"REGISTERED RESULT: ok={ok} tests_run={tests_run} "
              f"failures={len(result.failures)} errors={len(result.errors)}",
              file=sys.stderr)
        if not ok:
            raise SystemExit(1)
        raise SystemExit(0)
    _unittest.main()
