#!/usr/bin/env python3
"""Device-free driver-health regressions for the complete HTTP cell lifecycle.

Replay real warning shapes, log loss, teardown faults and unreadable logs.
No GPU, kernel-log access, model or elevated privilege is used by this gate.
"""
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests/v2/e2e/server"))
import gpu_driver_diagnostics as driver


def record(message="ordinary boot record", priority="info", timestamp=10.0, facility="kern"):
    """Construct one structured printk record, independent of wall time."""
    return driver.KernelRecord(timestamp, facility, priority, message)


def snapshot(*records, boot="same-boot", reader=driver.KernelReader.DIRECT):
    """Expose an immutable simulated kernel log through the real policy API."""
    return driver.KernelSnapshot(boot, reader, tuple(records))


def checkpoint(*records):
    """Create the on-disk begin contract for focused continuity tests."""
    return {"schema": 1, "state": "armed", "boot_id": "same-boot", "reader": "direct",
            "cursor": [asdict(row) for row in records]}


def clean_evidence():
    """Complete synthetic receipt for outer runners' device-free fixtures."""
    return {"schema": 1, "complete": True, "passed": True,
            "boot_id": "same-boot", "reader": "direct", "reader_container": None,
            "checkpoint": "fixture.driver-checkpoint.json", "new_record_count": 0,
            "records": [], "findings": [], "error": None}


class GPUDriverDiagnosticsTests(unittest.TestCase):
    """Driver warnings must fail otherwise successful production HTTP cells."""

    def setUp(self):
        """Never inherit a real CI observer while testing simulated kernels."""
        self.enterContext(patch.dict(driver.os.environ))
        driver.os.environ.pop(driver.READER_CONTAINER_ENV, None)

    def test_amd_and_nvidia_severity_and_misleveled_faults(self):
        """In particular, reproduce MI50 IH overflow and informational Xid."""
        for message in (
            "amdgpu 0000:8c:00.0: amdgpu: ih2 ring buffer overflow (0x80000)",
            "workqueue: svm_range_restore_work [amdgpu] hogged CPU for >10000us",
            "workqueue: amdgpu_amdkfd_restore_userptr_worker [amdgpu] hogged CPU for >10000us 35 times",
            "workqueue: kfd_process_wq_release [amdgpu] hogged CPU for >10000us 259 times, consider switching to WQ_UNBOUND",
            "vega20_ih_get_wptr: 156 callbacks suppressed",
            "amdgpu: [gfxhub] page fault (src_id:0 ring:0 vmid:1 pasid:123)",
            "amdgpu: ring gfx timeout, signaled seq=14, emitted seq=15",
            "amdgpu: GPU reset succeeded, trying to resume",
            "kfd: SVM restore failed",
            "NVRM: Xid (PCI:0000:01:00): 31, pid=120, MMU Fault",
            "NVRM: GPU has fallen off the bus.",
            "nvidia-modeset: WARNING: GPU:0: Failed to query display engine",
            "nvidia-uvm: error: out-of-bounds access",
            "WARNING: CPU: 1 PID: 6 at drivers/gpu/drm/amd/amdgpu/example.c:12",
        ):
            for priority in ("info", "warn", "err"):
                with self.subTest(message=message, priority=priority):
                    self.assertTrue(driver.driver_diagnostic(record(message, priority)))
        self.assertTrue(driver.driver_diagnostic(record("amdgpu: unexpected status 17", "warn")))
        self.assertTrue(driver.driver_diagnostic(record("NVRM: bad status 17", "crit")))

    def test_normal_initialization_and_unrelated_warnings_are_not_gpu_faults(self):
        """Loaded module names alone do not attribute another subsystem's WARN."""
        for message, priority in (
            ("amdgpu: VRAM: 32768M available", "info"),
            ("amdgpu: ring gfx uses VM inv eng 0", "info"),
            ("nvidia-modeset: Loading NVIDIA Kernel Mode Setting Driver", "info"),
            ("Modules linked in: nvidia amdgpu kfd", "warn"),
            ("overlayfs: xino feature enabled", "info"),
            ("wifi: link reset", "warn"),
        ):
            with self.subTest(message=message):
                self.assertFalse(driver.driver_diagnostic(record(message, priority)))
        self.assertFalse(driver.driver_diagnostic(record("amdgpu error mentioned by userspace", "err", facility="user")))

    def test_old_warnings_are_excluded_but_new_equal_time_records_are_checked(self):
        """Ordering, not time-of-day or a coarse seconds filter, defines a cell."""
        old = record("NVRM: Xid 13", "err")
        good = record("nvidia: normal diagnostic", timestamp=10.0)
        new = record("amdgpu: ih2 ring buffer overflow", timestamp=10.0)
        self.assertEqual(driver.new_records(checkpoint(old, good), snapshot(old, good)), ())
        self.assertEqual(driver.new_records(checkpoint(old, good), snapshot(old, good, new)), (new,))

    def test_reboot_cleared_wrapped_and_ambiguous_logs_cannot_certify(self):
        """Lost history is a failed observation, even when retained lines look clean."""
        original = record()
        for after in (snapshot(original, boot="new-boot"), snapshot(),
                      snapshot(record("new clean tail")), snapshot(original, original),
                      snapshot(original, reader=driver.KernelReader.SUDO)):
            with self.subTest(after=after), self.assertRaises(ValueError):
                driver.new_records(checkpoint(original), after)
        with self.assertRaises(ValueError):
            driver.new_records([], snapshot(original))

    def test_reader_requires_observable_kernel_and_retains_selected_access_scope(self):
        """A denied direct read may select sudo; malformed success is never ignored."""
        payload = json.dumps({"dmesg": [asdict(record())]})
        denied = SimpleNamespace(returncode=1, stderr="Operation not permitted", stdout="")
        valid = SimpleNamespace(returncode=0, stderr="", stdout=payload)
        with patch.object(driver, "_BOOT_ID") as boot, \
             patch.object(driver.subprocess, "run", side_effect=(denied, valid)) as run:
            boot.read_text.return_value = "same-boot"
            actual = driver.read_snapshot()
            self.assertEqual(actual.reader, driver.KernelReader.SUDO)
            self.assertEqual(run.call_args_list[1].args[0], ["sudo", "-n", "dmesg", "--json", "--decode"])
        with patch.object(driver, "_BOOT_ID") as boot, \
             patch.object(driver.subprocess, "run", return_value=denied) as run:
            boot.read_text.return_value = "same-boot"
            with self.assertRaisesRegex(RuntimeError, "Cannot read GPU driver diagnostics"):
                driver.read_snapshot(driver.KernelReader.DIRECT)
            self.assertEqual(run.call_count, 1)
        for output in ("not JSON", '{}', '{"dmesg":[]}', '{"dmesg":[{}]}'):
            with self.subTest(output=output), patch.object(driver, "_BOOT_ID") as boot, \
                 patch.object(driver.subprocess, "run", return_value=SimpleNamespace(
                     returncode=0, stdout=output, stderr="")):
                boot.read_text.return_value = "same-boot"
                with self.assertRaises(ValueError):
                    driver.read_snapshot()

    def test_complete_lifecycle_passes_and_teardown_warning_fails(self):
        """Use the public CLI lifecycle to reject a fault after successful HTTP."""
        old = record("amdgpu: old overflow", "err")
        for tail in (
            (),
            (record("NVRM: Xid 79, GPU has fallen off the bus", timestamp=20),),
            (record("workqueue: kfd_process_wq_release [amdgpu] hogged CPU for >10000us 259 times",
                    "warn", timestamp=20),),
        ):
            with self.subTest(tail=tail), tempfile.TemporaryDirectory() as directory:
                state = Path(directory) / "cell.driver-checkpoint.json"
                report = Path(directory) / "cell.driver-diagnostics.json"
                with patch.object(driver, "read_snapshot", side_effect=(snapshot(old), snapshot(old, *tail))):
                    self.assertEqual(driver.main(["begin", "--state", str(state)]), 0)
                    with self.assertRaises(ValueError):
                        driver.begin(state)
                    self.assertEqual(driver.main(["finish", "--state", str(state), "--report", str(report)]),
                                     1 if tail else 0)
                evidence = json.loads(report.read_text())
                self.assertEqual(evidence["findings"], [asdict(row) for row in tail])
                self.assertTrue(evidence["complete"])
                self.assertEqual(json.loads(state.read_text())["state"], "closed")
                if tail:
                    with self.assertRaises(ValueError):
                        driver.validate_evidence(Path(directory))
                else:
                    driver.validate_evidence(Path(directory))
                with self.assertRaises(ValueError):
                    driver.finish(state, report)

    def test_ci_reader_is_fixed_read_only_command_on_same_kernel_and_owned_container(self):
        """Non-root cache ownership cannot disable CI's kernel-health gate."""
        name = "llaminar-suite-driver-" + "a" * 32
        raw = json.dumps({"dmesg": [asdict(record())]})
        with patch.dict(driver.os.environ, {driver.READER_CONTAINER_ENV: name}), \
             patch.object(driver, "_BOOT_ID") as boot, \
             patch.object(driver.subprocess, "run", return_value=SimpleNamespace(
                 returncode=0, stderr="", stdout="same-boot\n" + raw)) as run:
            boot.read_text.return_value = "same-boot"
            actual = driver.read_snapshot()
            self.assertEqual(actual.reader, driver.KernelReader.TOOLS_CONTAINER)
            self.assertEqual(actual.container, name)
            command = run.call_args.args[0]
            self.assertEqual(command[:5], ["docker", "exec", "--user", "0:0", name])
            self.assertNotIn("--clear", command[-1])
            run.return_value.stdout = "another-host\n" + raw
            with self.assertRaisesRegex(ValueError, "boot"):
                driver.read_snapshot()
        for invalid in (None, "", "unrelated-server", name + ";true"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                driver.KernelReader.TOOLS_CONTAINER.command(invalid)

    def test_unreadable_finish_and_lost_history_preserve_failed_evidence(self):
        """Monitoring failure cannot leave a stale pass for the outer driver."""
        old = record()
        for ending in (RuntimeError("permission denied"), snapshot(record("ring wrapped"))):
            with self.subTest(ending=ending), tempfile.TemporaryDirectory() as directory:
                state = Path(directory) / "cell.driver-checkpoint.json"
                report = Path(directory) / "cell.driver-diagnostics.json"
                with patch.object(driver, "read_snapshot", side_effect=(snapshot(old), ending)):
                    driver.begin(state)
                    self.assertFalse(driver.finish(state, report))
                self.assertFalse(json.loads(report.read_text())["complete"])
                with self.assertRaises(ValueError):
                    driver.validate_evidence(Path(directory))

    def test_outer_http_admission_requires_driver_evidence(self):
        """A successful shell/needle/tool result cannot omit driver health."""
        sys.path.insert(0, str(ROOT / "scripts/ci"))
        import run_model_parity_e2e as e2e
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(e2e, "validate_long_context_evidence"), \
             patch.object(e2e, "validate_tool_calling_evidence"), \
             patch.object(e2e, "automatic_selection_policy", return_value={}), \
             patch.object(e2e, "validate_automatic_selection"):
            root = Path(directory)
            (root / "tool_calling_results.json").write_text('{}')
            (root / "automatic_selection_results.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "driver"):
                e2e.validate_http_cell_evidence(root, {})

    def test_outer_reader_rejects_a_passing_summary_with_faults_or_missing_records(self):
        """Retained raw evidence, not only a cached green flag, owns admission."""
        bad = asdict(record("NVRM: Xid 31, MMU Fault"))
        invalid = [[], {**clean_evidence(), "new_record_count": 1},
                   {**clean_evidence(), "new_record_count": 1, "records": [bad]},
                   {**clean_evidence(), "new_record_count": 1, "records": [{}]}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for evidence in invalid:
                with self.subTest(evidence=evidence):
                    driver.publish(root / "cell.driver-diagnostics.json", evidence)
                    with self.assertRaises(ValueError):
                        driver.validate_evidence(root)

    def test_real_shell_counter_fails_on_driver_warning_after_successful_http(self):
        """Exercise the shared shell's exit decision without a model or GPU.

        Only log reading is simulated. The real Python CLI, artifact publisher,
        shell pass/fail counters and final cell exit decision all execute.
        """
        shell = (ROOT / "tests/v2/e2e/server/test_server_e2e.sh").read_text()
        functions = "\n".join(name + "() {" + shell.split(name + "() {", 1)[1].split("\n}\n", 1)[0]
                              + "\n}" for name in ("pass", "fail", "finish_driver_diagnostics"))
        for warning in (None, "amdgpu: ih2 ring buffer overflow", "NVRM: Xid 79"):
            with self.subTest(warning=warning), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                state = root / "cell.driver-checkpoint.json"
                report = root / "cell.driver-diagnostics.json"
                driver.publish(state, checkpoint(record()))
                # Match the admitted direct reader; no sudo or host log access
                # may occur in this device-free protocol regression.
                boot = driver._BOOT_ID.read_text().strip()
                saved = json.loads(state.read_text())
                driver.publish(state, {**saved, "boot_id": boot})
                records = [asdict(record())]
                if warning:
                    records.append(asdict(record(warning, timestamp=20)))
                fake = root / "dmesg"
                fake.write_text("#!/bin/sh\nprintf '%s\\n' " + shlex.quote(json.dumps({"dmesg": records})) + "\n")
                fake.chmod(0o755)
                env = {**driver.os.environ, "PATH": str(root) + ":" + driver.os.environ["PATH"]}
                command = (functions + '\nTOTAL_TESTS=0; PASSED_TESTS=0; FAILED_TESTS=0\n'
                           'SCRIPT_DIR="$1"\npass "HTTP passed"\n'
                           'finish_driver_diagnostics cell "$2" "$3"\n'
                           '[ "$FAILED_TESTS" -eq 0 ]\n')
                result = subprocess.run(["bash", "-c", command, "guard-regression",
                                         str(ROOT / "tests/v2/e2e/server"), str(state), str(report)],
                                        env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 1 if warning else 0, result.stdout + result.stderr)
                self.assertEqual(json.loads(report.read_text())["passed"], warning is None)

    def test_shell_brackets_server_startup_and_teardown_and_has_no_disable_switch(self):
        """Keep this observer in the shared harness, including generation cells."""
        path = ROOT / "tests/v2/e2e/server/test_server_e2e.sh"
        subprocess.run(["bash", "-n", str(path)], check=True)
        shell = path.read_text()
        body = shell[shell.index("run_backend_tests() {"):]
        self.assertLess(body.index('gpu_driver_diagnostics.py" begin'), body.index('start_server_process "$tag"'))
        after_shutdown = body[body.index('shutdown_and_validate "$tag"'):]
        self.assertIn('finish_driver_diagnostics "$tag"', after_shutdown)
        self.assertNotIn("LLAMINAR_E2E_SKIP_DRIVER", shell)


if __name__ == "__main__":
    unittest.main()
