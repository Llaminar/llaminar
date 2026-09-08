#!/usr/bin/env python3
"""Exercise the real pre-commit script using device-free command recorders.

The hook must run exactly two CTest suites, build only their canonical targets,
resolve the workspace Ninja, and stop at the first failed phase. Temporary
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
if name == "git" and args == ["rev-parse", "--abbrev-ref", "HEAD"]:
    print(os.environ["HOOK_TEST_BRANCH"])
    sys.exit(0)
if name == "cmake":
    phase = "build" if "--build" in args else "configure"
elif name == "ctest":
    phase = "unit" if "-R" in args else "preflight"
else:
    sys.exit(99)
sys.exit(17 if os.environ.get("HOOK_TEST_FAIL") == phase else 0)
'''


class PreCommitHookTests(unittest.TestCase):
    """Prove selection, ordering, quoting, and fail-fast behavior end to end."""

    def run_hook(self, *, fail: str = "", branch: str = "feature/example"):
        """Run the checked-in shell script in an isolated path containing spaces."""
        with tempfile.TemporaryDirectory(prefix="llaminar-hook-") as directory:
            root = Path(directory) / "repository with spaces"
            nested = root / "nested"
            tools = root / "tool shims"
            nested.mkdir(parents=True)
            tools.mkdir()
            for tool in ("git", "cmake", "ctest", "ninja"):
                shim = tools / tool
                shim.write_text(f"#!{sys.executable}\n" + RECORDER, encoding="utf-8")
                shim.chmod(0o755)
            record = root / "commands.jsonl"
            env = dict(os.environ)
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
                                 ["git", "cmake", "cmake", "ctest", "ctest"])
                build = str(root / "build_v2_integration")
                self.assertEqual(calls[1]["args"], [
                    "-B", build, "-S", str(root / "src/v2"), "-G", "Ninja",
                    "-DCMAKE_BUILD_TYPE=Integration",
                    f"-DCMAKE_MAKE_PROGRAM:FILEPATH={tools / 'ninja'}",
                    "-DLLAMINAR_BUILD_TESTS=ON", "-DHAVE_CUDA=ON", "-DHAVE_ROCM=ON",
                ])
                self.assertEqual(calls[2]["args"], [
                    "--build", build, "--parallel", "--target",
                    "v2_unit_gate", "v2_production_parity_preflight_gate",
                ])
                common = ["--test-dir", build, "--output-on-failure",
                          "--parallel", "--no-tests=error"]
                self.assertEqual(calls[3]["args"], common + ["-R", "^V2_Unit_"])
                self.assertEqual(calls[4]["args"], common + ["-L", "^ProductionParityPreflight$"])
                for call in calls[1:]:
                    self.assertEqual(call["cwd"], str(root))

    def test_each_failure_blocks_later_phases(self):
        """Configure/build/Unit/preflight failures all propagate to Git."""
        for phase, count in (("configure", 2), ("build", 3), ("unit", 4), ("preflight", 5)):
            with self.subTest(phase=phase):
                result, calls, _, _ = self.run_hook(fail=phase)
                self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
                self.assertEqual(len(calls), count)
                self.assertNotIn("Unit and ProductionParityPreflight passed.", result.stdout)


if __name__ == "__main__":
    unittest.main()
