#!/usr/bin/env python3
"""Device-free driver-health and continuous HTTP lifetime regressions.

Exercise ring rotation, missing records, late teardown warnings, process
retirement, journal corruption and the actual CLI/shell closure protocol.
Kernel reads are simulated; no model, GPU or elevated privilege is used.
"""
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import socket
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
    """Expose an immutable simulated kernel ring through the real policy API."""
    return driver.KernelSnapshot(boot, reader, tuple(records))


def checkpoint(*records):
    """Construct the collector's internal predecessor cursor contract."""
    return {"schema": 1, "state": "armed", "boot_id": "same-boot", "reader": "direct",
            "cursor": [asdict(row) for row in records]}


def clean_evidence(directory: Path):
    """Write a complete minimal journal for outer-runner device-free fixtures."""
    directory.mkdir(parents=True, exist_ok=True)
    window = 'a' * 32
    cursor = [asdict(record())]
    rows = [{'schema': 2, 'kind': 'begin', 'window_id': window, 'boot_id': 'same-boot',
             'reader': 'direct', 'reader_container': None, 'cursor': cursor},
            {'kind': 'end', 'window_id': window, 'complete': True, 'new_record_count': 0,
             'snapshot_count': 2, 'cursor': cursor}]
    journal = directory / 'fixture.driver-records.jsonl'
    journal.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return {'schema': 2, 'complete': True, 'passed': True, 'window_id': window,
            'boot_id': 'same-boot', 'reader': 'direct', 'reader_container': None,
            'checkpoint': 'fixture.driver-checkpoint.json', 'new_record_count': 0,
            'snapshot_count': 2, 'journal': journal.name,
            'findings': [], 'finding_count': 0, 'error': None}


class GPUDriverDiagnosticsTests(unittest.TestCase):
    """Existing warning, scope and public-harness obligations stay mandatory."""
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


class GPUDriverWindowTests(unittest.TestCase):
    """Long lifetimes rotate the ring without losing collected diagnostics."""

    def run_collector(self, root, before, after, *, controls=None, owners=None):
        """Drive the real collector and journal through a deterministic ring."""
        state = root / 'cell.driver-checkpoint.json'
        window = 'b' * 32
        driver.publish(state, {'schema': 2, 'state': 'starting', 'window_id': window,
            'boot_id': before.boot_id, 'reader': before.reader.value, 'reader_container': before.container,
            'owner': {'pid': 1, 'start_ticks': 1}, 'cursor': [asdict(row) for row in before.records[-8:]]})
        close = json.dumps({'action': 'close', 'window_id': window}).encode()
        if controls is None:
            controls = [socket.timeout()] * max(0, len(after) - 2) + [close]
        with patch.object(driver, 'read_snapshot', side_effect=after), \
             patch.object(driver, 'process_is_live', side_effect=owners) as alive, \
             patch.object(driver.socket, 'socket') as control:
            alive.return_value = True
            control.return_value.__enter__.return_value.recv.side_effect = controls
            passed = driver.collect(state)
        terminal = json.loads(state.read_text())
        self.assertEqual(terminal['state'], 'retired')
        return state, terminal['outcome'], passed

    def test_continuous_collection_survives_repeated_ring_rotation(self):
        """Reproduce the five-hour bookend defect with 128 immediate rotations."""
        rows = [record('kernel row ' + str(i), timestamp=i) for i in range(1040)]
        before = snapshot(*rows[:8])
        after = [snapshot(*rows[i:i+16]) for i in range(0,1024,8)]
        with self.assertRaisesRegex(ValueError, 'cursor lost'):
            driver.new_records(checkpoint(*before.records), after[-1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, evidence, passed = self.run_collector(root, before, after)
            self.assertTrue(passed)
            self.assertEqual(evidence['new_record_count'], 1024)
            self.assertEqual(evidence['snapshot_count'], 128)
            driver.publish(root / 'cell.driver-diagnostics.json', evidence)
            driver.validate_evidence(root)

    def test_loss_between_observations_and_reboots_still_fail(self):
        """A short polling interval never excuses missing kernel history."""
        before = snapshot(record())
        for end in (snapshot(record('wrapped clean tail',timestamp=20)),
                    snapshot(record(),boot='new-boot'), RuntimeError('permission denied')):
            with self.subTest(end=end), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                _, evidence, passed = self.run_collector(root, before, [before,end])
                self.assertFalse(passed)
                self.assertFalse(evidence['complete'])
                self.assertIsNotNone(evidence['error'])
                driver.publish(root/'cell.driver-diagnostics.json',evidence)
                with self.assertRaises(ValueError):driver.validate_evidence(root)

    def test_final_read_captures_teardown_fault_and_preserves_bounded_preview(self):
        """Keep every warning in the journal while retaining only 32 in memory."""
        before = snapshot(record())
        warnings = [record('NVRM: nvAssertFailedNoLog: Assertion failed ' + str(i),
                           'warn',timestamp=20+i) for i in range(40)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            _, evidence, passed=self.run_collector(root,before,[before,snapshot(*before.records,*warnings)])
            self.assertFalse(passed)
            self.assertTrue(evidence['complete'])
            self.assertEqual(evidence['finding_count'],40)
            self.assertEqual(evidence['findings'],[asdict(row) for row in warnings[:32]])
            journal=[json.loads(line) for line in (root/evidence['journal']).read_text().splitlines()]
            self.assertEqual(len([row for row in journal if row['kind']=='record']),40)
            evidence.update(passed=True,findings=[],finding_count=0)
            driver.publish(root/'cell.driver-diagnostics.json',evidence)
            with self.assertRaises(ValueError):driver.validate_evidence(root)

    def test_owner_retirement_and_invalid_close_are_terminal_failures(self):
        """A detached collector cannot silently certify an abandoned cell."""
        before=snapshot(record())
        cases=[{'owners':[False]}, {'controls':[b'{}']},
               {'controls':[json.dumps({'action':'close','window_id':'c'*32}).encode()]}]
        for options in cases:
            with self.subTest(options=options),tempfile.TemporaryDirectory() as directory:
                _,evidence,passed=self.run_collector(Path(directory),before,[before,before],**options)
                self.assertFalse(passed)
                self.assertFalse(evidence['complete'])

    def test_journal_tampering_truncation_and_extra_data_fail(self):
        """A green summary cannot mask missing records, boundaries or identity."""
        for mutation in ('truncate','missing_end','extra','wrong_window','count','cursor','reorder'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                root=Path(directory); evidence=clean_evidence(root);path=root/evidence['journal']
                rows=[json.loads(line) for line in path.read_text().splitlines()]
                if mutation=='truncate':path.write_bytes(path.read_bytes()[:-1])
                else:
                    if mutation=='missing_end':rows.pop()
                    elif mutation=='extra':rows.append(rows[-1])
                    elif mutation=='wrong_window':rows[0]['window_id']='d'*32
                    elif mutation=='count':rows[-1]['new_record_count']=1
                    elif mutation=='cursor':rows[-1]['cursor']=[]
                    elif mutation=='reorder':rows.insert(1,{'kind':'record','ordinal':2,**asdict(record(timestamp=20))})
                    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                driver.publish(root/'cell.driver-diagnostics.json',evidence)
                with self.assertRaises(ValueError):driver.validate_evidence(root)

    def test_retired_or_reused_collector_cannot_make_a_fresh_pass(self):
        """A dead native PID never authorizes a substitute monitoring process."""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);state=root/'cell.driver-checkpoint.json';report=root/'cell.driver-diagnostics.json'
            driver.publish(state,{'schema':2,'state':'armed','window_id':'e'*32,
                                  'collector':{'pid':99999,'start_ticks':5}})
            with patch.object(driver,'process_identity',return_value={'pid':99999,'start_ticks':6}):
                self.assertFalse(driver.finish(state,report))
            self.assertFalse(json.loads(report.read_text())['complete'])
            with self.assertRaises(ValueError):driver.finish(state,report)

    def test_real_cli_and_shell_count_teardown_warnings(self):
        """Run real owned background processes with only dmesg substituted."""
        shell=(ROOT/'tests/v2/e2e/server/test_server_e2e.sh').read_text()
        functions='\n'.join(name+'() {'+shell.split(name+'() {',1)[1].split('\n}\n',1)[0]+'\n}'
                             for name in ('pass','fail','finish_driver_diagnostics'))
        for warning in (None,'amdgpu: ih2 ring buffer overflow','NVRM: Xid 79'):
            with self.subTest(warning=warning),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);state=root/'cell.driver-checkpoint.json';report=root/'cell.driver-diagnostics.json'
                current=root/'current.json';current.write_text(json.dumps({'dmesg':[asdict(record())]}))
                ending=root/'ending.json';ending.write_text(json.dumps({'dmesg':[asdict(record())]+(
                    [asdict(record(warning,timestamp=20))] if warning else [])}))
                fake=root/'dmesg';fake.write_text('#!/bin/sh\nexec cat '+shlex.quote(str(current))+'\n');fake.chmod(0o755)
                env={**driver.os.environ,'PATH':str(root)+':'+driver.os.environ['PATH']}
                env.pop(driver.READER_CONTAINER_ENV,None)
                command=(functions+'\nTOTAL_TESTS=0; PASSED_TESTS=0; FAILED_TESTS=0\n'
                    'SCRIPT_DIR="$1"\n'
                    'python3 "$SCRIPT_DIR/gpu_driver_diagnostics.py" begin --state "$2" || exit 3\n'
                    'pass "HTTP passed"\ncp "$4" "$5"\n'
                    'finish_driver_diagnostics cell "$2" "$3"\n'
                    '[ "$FAILED_TESTS" -eq 0 ]\n')
                result=subprocess.run(['bash','-c',command,'guard-regression',str(ROOT/'tests/v2/e2e/server'),
                    str(state),str(report),str(ending),str(current)],env=env,capture_output=True,text=True,timeout=10)
                self.assertEqual(result.returncode,1 if warning else 0,result.stdout+result.stderr)
                evidence=json.loads(report.read_text())
                self.assertEqual(evidence['passed'],warning is None)
                self.assertTrue(evidence['complete'])
                self.assertFalse(driver.process_is_live(evidence['collector']))
                self.assertEqual(json.loads(state.read_text())['state'],'closed')
                if warning is None:driver.validate_evidence(root)

    def test_journal_paths_and_line_bound_are_checked(self):
        """Reports cannot substitute an unrelated file or unbounded JSON row."""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('../outside','/tmp/elsewhere',''):
                evidence=clean_evidence(root);evidence['journal']=name
                with self.subTest(name=name),self.assertRaises(ValueError):driver.inspect_journal(root,evidence)
            evidence=clean_evidence(root)
            (root/evidence['journal']).write_bytes(b'x'*(driver._MAX_LINE_BYTES+1))
            with self.assertRaises(ValueError):driver.inspect_journal(root,evidence)


if __name__ == '__main__':
    unittest.main()
