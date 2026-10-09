#!/usr/bin/env python3
"""Exercise the real pre-commit script using device-free command recorders.

The hook runs exactly two CTest suites or authenticates their explicitly
selected prior receipt. It builds only their canonical targets, resolves the
workspace Ninja, and stops at the first failed phase. Temporary
repositories and executable shims keep these tests out of real builds, model
loading, Git registration, and accelerator execution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HOOK = Path(__file__).resolve().parents[4] / ".githooks/pre-commit"

# One recorder stands in for all external commands. Unknown invocations fail,
# making accidental new test/build paths observable instead of silently green.
RECORDER = r'''
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
record = {"tool": name, "args": args, "cwd": os.getcwd()}
with open(os.environ["HOOK_TEST_RECORD"], "a", encoding="utf-8") as out:
    out.write(json.dumps(record) + "\n")
if name == "git" and args == ["rev-parse", "--show-toplevel"]:
    print(os.environ["HOOK_TEST_ROOT"])
    sys.exit(0)
if name == "git" and args == ["rev-parse", "--local-env-vars"]:
    print("GIT_DIR\nGIT_WORK_TREE\nGIT__FILE\nGIT_PREFIX")
    sys.exit(0)
if name == "git" and args == ["rev-parse", "--abbrev-ref", "HEAD"]:
    print(os.environ["HOOK_TEST_BRANCH"])
    sys.exit(0)
if name == "cmake":
    phase = "build" if "--build" in args else "configure"
elif name == "ctest":
    phase = "unit" if "-R" in args else "preflight"
elif name == "python3":
    phase = "reuse"
else:
    sys.exit(99)
sys.exit(17 if os.environ.get("HOOK_TEST_FAIL") == phase else 0)
'''


def run_foreign_checkout_hook(hook: str) -> dict:
    """Run supplied hook text from a real linked worktree with a foreign checkout.

    Build/test shims inspect repository selection without invoking compilers or
    accelerators. Return the hook status, observed dependency identity and logs.
    """
    with tempfile.TemporaryDirectory(prefix="git-hook-dependency-") as temporary:
        directory = Path(temporary)
        base = directory / "main"
        root = directory / "linked source"
        dependency = root / "external/dependency"
        # Fixture setup must not inherit a caller's repository identity either.
        environment = dict(os.environ)
        local_variables = subprocess.check_output(
            ["git", "rev-parse", "--local-env-vars"], text=True).splitlines()
        for name in local_variables:
            environment.pop(name, None)

        def git(*args, cwd=directory):
            """Run real Git in the fixture with its own repository authority."""
            return subprocess.run(
                ["git", *map(str, args)], cwd=cwd, capture_output=True,
                text=True, check=True, env=environment).stdout.strip()

        git("init", "-q", base)
        git("-c", "user.name=Hook Test", "-c", "user.email=hook@example.invalid",
            "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
            "commit", "--allow-empty", "-qm", "fixture", cwd=base)
        git("worktree", "add", "--detach", root, cwd=base)
        git("remote", "add", "origin", "https://example.invalid/main.git", cwd=root)
        dependency.mkdir(parents=True)
        git("init", "-q", dependency)
        git("remote", "add", "origin", "https://example.invalid/dependency.git", cwd=dependency)
        hooks = root / ".githooks"
        hooks.mkdir()
        target = hooks / "pre-commit"
        target.write_text(hook)
        target.chmod(0o755)
        git("config", "core.hooksPath", ".githooks", cwd=root)
        tools = root / "tools"
        tools.mkdir()
        for name in ("ctest", "ninja"):
            tool = tools / name
            tool.write_text("#!/bin/sh\nexit 0\n")
            tool.chmod(0o755)
        cmake = tools / "cmake"
        cmake.write_text(f"#!{sys.executable}\n" + """import json, os, pathlib, subprocess, sys
if '--build' in sys.argv:
    sys.exit(0)
root = pathlib.Path.cwd()
dependency = root / 'external/dependency'
origin = subprocess.check_output(
    ['git', '-C', str(dependency), 'remote', 'get-url', 'origin'], text=True).strip()
repository_keys = ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_PREFIX', 'GIT_COMMON_DIR')
(root / 'observed.json').write_text(json.dumps({
    'origin': origin,
    'repository_variables': sorted(k for k in repository_keys if k in os.environ),
}))
sys.exit(0 if origin == 'https://example.invalid/dependency.git' else 23)
""")
        cmake.chmod(0o755)
        environment["PATH"] = str(tools) + os.pathsep + environment["PATH"]
        environment.pop("LLAMINAR_PRECOMMIT_BUILD_DIR", None)
        environment.pop("LLAMINAR_PRECOMMIT_PREREQUISITES", None)
        result = subprocess.run(
            ["git", "hook", "run", "pre-commit"], cwd=root, env=environment,
            capture_output=True, text=True)
        return {
            "returncode": result.returncode,
            "observed": json.loads((root / "observed.json").read_text()),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


class PreCommitHookTests(unittest.TestCase):
    """Prove selection, ordering, quoting, and fail-fast behavior end to end."""

    def run_hook(self, *, fail: str = "", branch: str = "feature/example", reuse: bool = False):
        """Run the checked-in shell script in an isolated path containing spaces."""
        with tempfile.TemporaryDirectory(prefix="llaminar-hook-") as directory:
            root = Path(directory) / "repository with spaces"
            nested = root / "nested"
            tools = root / "tool shims"
            nested.mkdir(parents=True)
            tools.mkdir()
            for tool in ("git", "cmake", "ctest", "ninja", "python3"):
                shim = tools / tool
                shim.write_text(f"#!{sys.executable}\n" + RECORDER, encoding="utf-8")
                shim.chmod(0o755)
            record = root / "commands.jsonl"
            env = dict(os.environ)
            env.pop("LLAMINAR_PRECOMMIT_BUILD_DIR", None)
            env.pop("LLAMINAR_PRECOMMIT_PREREQUISITES", None)
            if reuse:
                env["LLAMINAR_PRECOMMIT_BUILD_DIR"] = str(root / "selected integration build")
                env["LLAMINAR_PRECOMMIT_PREREQUISITES"] = str(root / "evidence directory/prerequisites.json")
            env.update({
                "PATH": str(tools) + os.pathsep + env["PATH"],
                "HOOK_TEST_ROOT": str(root),
                "HOOK_TEST_RECORD": str(record),
                "HOOK_TEST_BRANCH": branch,
                "HOOK_TEST_FAIL": fail,
                # Retired knobs must not enable another gate by accident.
                "LLAMINAR_PRECOMMIT_E2E_CONTAINER": "1",
                "LLAMINAR_PRECOMMIT_E2E_CONTAINER_BUILD": "1",
            })
            result = subprocess.run(
                ["bash", str(HOOK)], cwd=nested, env=env,
                text=True, capture_output=True, timeout=10,
            )
            calls = [json.loads(line) for line in record.read_text().splitlines()]
            return result, calls, root, tools

    def test_exact_two_gates_on_every_branch(self):
        """Branch names and retired E2E knobs cannot widen the hook's scope."""
        for branch in ("feature/example", "develop", "master", "main"):
            with self.subTest(branch=branch):
                result, calls, root, tools = self.run_hook(branch=branch)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual([c["tool"] for c in calls],
                                 ["git", "git", "cmake", "cmake", "ctest", "ctest"])
                build = str(root / "build_v2_integration")
                self.assertEqual(calls[2]["args"], [
                    "-B", build, "-S", str(root / "src/v2"), "-G", "Ninja",
                    "-DCMAKE_BUILD_TYPE=Integration",
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                    f"-DCMAKE_MAKE_PROGRAM:FILEPATH={tools / 'ninja'}",
                    "-DLLAMINAR_BUILD_TESTS=ON", "-DHAVE_CUDA=ON", "-DHAVE_ROCM=ON",
                ])
                self.assertEqual(calls[3]["args"], [
                    "--build", build, "--parallel", "--target",
                    "v2_unit_gate", "v2_production_test_preflight_gate",
                ])
                common = ["--test-dir", build, "--output-on-failure",
                          "--parallel", "--no-tests=error"]
                self.assertEqual(calls[4]["args"], common + ["-R", "^V2_Unit_"])
                self.assertEqual(calls[5]["args"], common + ["-L", "^ProductionTestPreflight$"])
                for call in calls[2:]:
                    self.assertEqual(call["cwd"], str(root))

    def test_compiler_metadata_is_explicitly_regenerated(self):
        """A stale cache cannot disable the compile database used by SDK audits."""
        result, calls, _, _ = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        configure = calls[2]["args"]
        self.assertEqual(configure.count("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"), 1)

    def test_each_failure_blocks_later_phases(self):
        """Configure/build/Unit/preflight failures all propagate to Git."""
        for phase, count in (("configure", 3), ("build", 4), ("unit", 5), ("preflight", 6)):
            with self.subTest(phase=phase):
                result, calls, _, _ = self.run_hook(fail=phase)
                self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
                self.assertEqual(len(calls), count)
                self.assertNotIn("Unit and ProductionTestPreflight passed.", result.stdout)

    def test_selected_build_reuses_only_authenticated_complete_evidence(self):
        """An explicit receipt uses the canonical validator after the ordinary build."""
        result, calls, root, _ = self.run_hook(reuse=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([c["tool"] for c in calls], ["git", "git", "cmake", "cmake", "python3"])
        build = str(root / "selected integration build")
        self.assertEqual(calls[2]["args"][1], build)
        self.assertEqual(calls[3]["args"][1], build)
        self.assertEqual(calls[4]["args"], [str(root / "scripts/ci/run_production_prerequisites.py"),
            "--build-dir", build, "--reuse-report", str(root / "evidence directory/prerequisites.json")])

    def test_rejected_receipt_blocks_commit_without_another_test_path(self):
        """A stale or invalid explicit receipt cannot succeed or start an alternate gate."""
        result, calls, _, _ = self.run_hook(reuse=True, fail="reuse")
        self.assertEqual(result.returncode, 17)
        self.assertEqual(len(calls), 5)
        self.assertNotIn("Unit and ProductionTestPreflight passed.", result.stdout)


    def test_real_git_hook_keeps_foreign_repository_identity(self):
        """A linked-worktree hook must not redirect CMake's dependency Git calls."""
        result = run_foreign_checkout_hook(HOOK.read_text())
        self.assertEqual(result['returncode'], 0, result)
        self.assertEqual(result['observed']['origin'], 'https://example.invalid/dependency.git')
        self.assertEqual(result['observed']['repository_variables'], [])


if __name__ == "__main__":
    unittest.main()
