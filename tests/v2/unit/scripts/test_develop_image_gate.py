#!/usr/bin/env python3
"""Device-free contract tests for the narrow develop image build/publish gate."""
from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))

import run_develop_image_gate as gate


def images(isa: str) -> dict:
    """Return just the immutable IDs needed by the outer develop transaction."""
    return {"test-runner": {"id": f"test-runner-{isa.lower()}"},
            "runtime": {"id": f"runtime-{isa.lower()}"}}


def prerequisite_report() -> dict:
    """Build one complete synthetic canonical prerequisite receipt."""
    return {"preflight_return_code": 0,
            "preflight_tests": ["V2_Unit_Contract", "V2_Integration_Preflight"],
            "preflight_test_count": 2}


class DevelopImageGateTests(unittest.TestCase):
    """Keep develop publishing smaller than, and incapable of replacing, certification."""

    def test_publish_requires_both_shipping_isas(self):
        """A partial local diagnosis can never overwrite a public develop tag."""
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(SystemExit) as error:
                gate.parse_arguments(["--output", str(Path(temporary) / "out"), "--publish",
                                      "--image", "ghcr.io/llaminar/llaminar:develop",
                                      "--cpu-isa", "AVX2"])
        self.assertEqual(error.exception.code, 2)

    def test_builds_and_tests_both_images_before_any_publication(self):
        """The two required gates finish before either mutable registry tag moves."""
        source = {"revision": "a" * 40, "tree": "tree", "dirty": False}
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"

            def build(args, admitted, directory, *, test_inventory):
                self.assertEqual(admitted, source)
                self.assertEqual(test_inventory, gate.TestRunnerInventory.MODEL_FREE)
                events.append(("build", args.cpu_isa))
                return images(args.cpu_isa)

            def prerequisites(built, directory):
                events.append(("prerequisites", built["runtime"]["id"]))
                return prerequisite_report()

            def publish(built, isa, tag, revision, directory):
                self.assertEqual(revision, source["revision"])
                if isa == "AVX512":
                    self.assertEqual(events, [
                        ("build", "AVX512"), ("prerequisites", "runtime-avx512"),
                        ("build", "AVX2"), ("prerequisites", "runtime-avx2"),
                    ])
                events.append(("publish", isa))
                return {"tag": gate.runtime_tag(tag, isa),
                        "ref_tag": gate.revision_tag(tag, isa, revision),
                        "image": built["runtime"]["id"]}

            with patch.object(gate, "source_identity", return_value=source), \
                 patch.object(gate, "device_lease", side_effect=lambda: nullcontext()), \
                 patch.object(gate, "build", side_effect=build), \
                 patch.object(gate, "require_image_pair"), \
                 patch.object(gate, "run_prerequisites", side_effect=prerequisites), \
                 patch.object(gate, "publish_runtime", side_effect=publish):
                self.assertEqual(gate.main(["--output", str(output), "--publish",
                                            "--image", "ghcr.io/llaminar/llaminar:develop"]), 0)

            self.assertEqual(events, [
                ("build", "AVX512"), ("prerequisites", "runtime-avx512"),
                ("build", "AVX2"), ("prerequisites", "runtime-avx2"),
                ("publish", "AVX512"), ("publish", "AVX2"),
            ])
            receipt = json.loads((output / "develop-image-gate.json").read_text())
            self.assertTrue(receipt["complete"])
            self.assertTrue(receipt["published"])
            self.assertEqual(receipt["variants"]["AVX512"]["test_runner_inventory"],
                             gate.TestRunnerInventory.MODEL_FREE.value)
            self.assertEqual(receipt["variants"]["AVX2"]["publication"]["tag"],
                             "ghcr.io/llaminar/llaminar:develop-avx2")
            self.assertEqual(receipt["variants"]["AVX2"]["publication"]["ref_tag"],
                             "ghcr.io/llaminar/llaminar:develop-" + "a" * 40 + "-avx2")

    def test_a_failed_preflight_prevents_every_registry_mutation(self):
        """No public tag moves until every image has passed its own installed gate."""
        source = {"revision": "a" * 40, "tree": "tree", "dirty": False}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            with patch.object(gate, "source_identity", return_value=source), \
                 patch.object(gate, "device_lease", side_effect=lambda: nullcontext()), \
                 patch.object(gate, "build", side_effect=lambda args, *_, **__: images(args.cpu_isa)), \
                 patch.object(gate, "require_image_pair"), \
                 patch.object(gate, "run_prerequisites", side_effect=RuntimeError("preflight red")), \
                 patch.object(gate, "publish_runtime") as publish:
                with self.assertRaisesRegex(RuntimeError, "preflight red"):
                    gate.main(["--output", str(output), "--publish",
                               "--image", "ghcr.io/llaminar/llaminar:develop"])
            publish.assert_not_called()

    def test_ref_tag_is_immutable_and_published_before_mutable_branch(self):
        source = "a" * 40
        runtime = {"runtime": {"id": "sha256:" + "1" * 64}}
        base = "ghcr.io/llaminar/llaminar:develop"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(gate, "remote_image_id", return_value=None), \
                 patch.object(gate, "run") as execute, \
                 patch.object(gate, "image_identity", return_value=runtime["runtime"]):
                result = gate.publish_runtime(runtime, "AVX2", base, source, directory)
            pinned = base + "-" + source + "-avx2"
            mutable = base + "-avx2"
            self.assertEqual(result["ref_tag"], pinned)
            self.assertEqual([call.args[0][:2] for call in execute.call_args_list],
                             [["docker", "tag"], ["docker", "push"],
                              ["docker", "tag"], ["docker", "push"]])
            self.assertEqual(execute.call_args_list[1].args[0][-1], pinned)
            self.assertEqual(execute.call_args_list[3].args[0][-1], mutable)
            with patch.object(gate, "remote_image_id", return_value="sha256:" + "2" * 64), \
                 patch.object(gate, "run") as execute:
                with self.assertRaisesRegex(ValueError, "different image bytes"):
                    gate.publish_runtime(runtime, "AVX2", base, source, directory)
                execute.assert_not_called()

    def test_registry_probe_distinguishes_missing_from_transport_failure(self):
        manifest = '{"config":{"digest":"sha256:' + "1" * 64 + '"}}'
        with patch.object(gate.subprocess, "run", return_value=type("Result", (), {
                "returncode": 0, "stdout": manifest, "stderr": ""})()):
            self.assertEqual(gate.remote_image_id("example:ref"), "sha256:" + "1" * 64)
        with patch.object(gate.subprocess, "run", return_value=type("Result", (), {
                "returncode": 1, "stdout": "", "stderr": "manifest unknown",
                "args": ["docker"]})()):
            self.assertIsNone(gate.remote_image_id("example:missing"))
        with patch.object(gate.subprocess, "run", return_value=type("Result", (), {
                "returncode": 1, "stdout": "", "stderr": "network unavailable",
                "args": ["docker"]})()):
            with self.assertRaises(gate.subprocess.CalledProcessError):
                gate.remote_image_id("example:failed")

    def test_canceled_build_retires_exact_child_group(self):
        """SIGTERM of the Actions driver cannot orphan its Buildx-style child."""
        driver_source = (
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from run_production_pipeline import run\n"
            "child = 'import os,sys,time;from pathlib import Path;"
            "Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(30)'\n"
            "run([sys.executable, '-c', child, sys.argv[2]], Path(sys.argv[3]))\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid_file, log = root / "child.pid", root / "build.log"
            driver = subprocess.Popen([sys.executable, "-c", driver_source,
                                       str(ROOT / "scripts/ci"), str(pid_file), str(log)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                      start_new_session=True)
            child_pid = None
            try:
                deadline = time.monotonic() + 5
                while not pid_file.exists() and driver.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(pid_file.exists(), "build child never reached its ready edge")
                child_pid = int(pid_file.read_text())
                os.kill(driver.pid, signal.SIGTERM)
                os.kill(driver.pid, signal.SIGTERM)
                self.assertNotEqual(driver.wait(timeout=5), 0)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if driver.poll() is None:
                    driver.terminate()
                    driver.wait(timeout=5)
                if child_pid is not None:
                    try:
                        os.kill(child_pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        os.killpg(child_pid, signal.SIGKILL)

    def test_ci_replaces_the_step_shell_with_the_cancellable_driver(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        self.assertIn("exec python3 scripts/ci/run_develop_image_gate.py", workflow)
        self.assertIn("if: ${{ !cancelled() }}", workflow)


if __name__ == "__main__":
    unittest.main()
