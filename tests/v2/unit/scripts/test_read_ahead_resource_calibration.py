#!/usr/bin/env python3
"""Fast structural receipt/rejection replay, never a native resource certificate."""
import copy
import importlib.util
from pathlib import Path
import unittest
import argparse
import json
import os
import sys
import tempfile
import threading
import signal
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("calibration_driver", ROOT / "scripts/diagnostics/run_read_ahead_resource_calibration.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def records():
    """Model complete process protocol observations solely for parser/controller rejection tests."""
    closure = {"device": "CUDA:0", "physical_uuid": "0001", "architecture": "sm_test", "runtime_version": 1, "driver_api_version": 1, "resource_source_sha256": "a"*64, "calibration_source_sha256": "b"*64}
    geometry = {"side_streams": 1, "ordering_events": 2, "concurrent_parents": 1, "lifetimes": 1, "parent_terminal_lifetimes": 1, "replays": 2}
    stages = [("before_backend_lookup", {}), ("original_context_ready", {}), ("explicit_parent_payload_allocated", {}), ("before_parent_terminal", {"parent_lifetime": 0}), ("after_parent_terminal_created", {"parent_lifetime": 0}), ("first_parent_terminal_record_complete", {"parent_lifetime": 0}), ("parent_payload_and_terminal_baseline", {})]
    stages += [(name, {"lifetime": 0}) for name in ("before_stream", "after_stream", "after_fork", "after_join", "lazy_first_use_complete", "capture_definition_complete", "capture_instantiated")]
    stages += [("first_parent_replay_complete", {"lifetime": 0, "replay": 0}), ("parent_replay_complete", {"lifetime": 0, "replay": 1})]
    stages += [(name, {"lifetime": 0}) for name in ("capture_destroyed", "after_destroy_join", "after_destroy_fork", "after_destroy_stream")]
    stages += [("after_parent_terminal_destroyed", {"parent_lifetime": 0}), ("explicit_baseline_owners_destroyed", {}), ("exclusive_context_retirement_complete", {"reset_invoked": True, "retired_context_generation": 1, "retired_runtime_generation": 1, "successor_runtime_generation": 2, "runtime_post_reset_state": "quiescent", "injected_failure": False})]
    return [{"schema": driver.SCHEMA, "pid": 123, "sequence": index, "stage": stage, "closure": copy.deepcopy(closure), "geometry": copy.deepcopy(geometry), "qualified": False, "certified_limit": None, "production_admission": False, "extra": extra} for index, (stage, extra) in enumerate(stages)]


class CalibrationReceiptReplay(unittest.TestCase):
    def test_library_identity_never_opens_device_nodes_or_other_special_files(self):
        """Replay mapped driver nodes without touching them; keep ordinary ELF identity caching."""
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            library=root/"mapped-library.so"
            library.write_bytes(b"\x7fELFfixture-library")
            fifo=root/"mapped-fifo"
            os.mkfifo(fifo)
            special={Path("/dev/null"),root,fifo}
            paths=[*special,library]
            snapshot={"maps":"\n".join(f"1000-2000 r--p 0 00:00 1 {path}" for path in paths)}
            original_open=Path.open
            def checked_open(path,*args,**kwargs):
                if path in special:
                    self.fail(f"Passive library observer opened special file: {path}")
                return original_open(path,*args,**kwargs)
            cache={}
            with mock.patch.object(Path,"open",checked_open), mock.patch.object(
                    driver.subprocess,"run",return_value=argparse.Namespace(stdout="Build ID: abc123\n")) as readelf:
                first=driver.mapped_identities(snapshot,cache)
                second=driver.mapped_identities(snapshot,cache)
            self.assertEqual(set(first),{str(library)})
            self.assertEqual(first,second)
            self.assertEqual(first[str(library)]["elf_build_ids"],["abc123"])
            self.assertEqual(len(first[str(library)]["sha256"]),64)
            self.assertEqual(readelf.call_count,1)

    def test_complete_structural_replay_never_mints_certificate(self):
        for _ in range(20):
            value = records()
            driver.validate_records(value, "cuda:0", 0)
            self.assertTrue(all(record["certified_limit"] is None and record["qualified"] is False for record in value))

    def test_geometry_drift_rejects(self):
        value=records()
        value[2]["geometry"]["replays"]=3
        with self.assertRaises(ValueError):
            driver.validate_records(value,"cuda:0",0)

    def test_term_ignoring_owned_child_is_killed_reaped_and_receipt_survives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            binary=root/"owned-child"
            binary.write_text(f"#!{sys.executable}\nimport json,os,signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nprint(json.dumps(dict(schema={driver.SCHEMA!r},pid=os.getpid(),host=dict(maxrss_kib=0))),flush=True)\nwhile True: time.sleep(1)\n")
            binary.chmod(0o700)
            args=argparse.Namespace(output=root,device="cuda:0",binary=binary,bytes=4096,replays=1,
                external_observer=None,identity_cache={},sample_ms=5)
            original_threads={thread.ident for thread in threading.enumerate()}
            with mock.patch.object(driver,"TERMINATION_GRACE_SECONDS",0.05), mock.patch.object(
                    driver,"mapped_identities",side_effect=RuntimeError("owned observation rejection")):
                with self.assertRaisesRegex(RuntimeError,"partial evidence retained"):
                    driver.run_one(args,"owned-failure",1)
            receipt=json.loads((root/"owned-failure/process-result.json").read_text())
            self.assertTrue(receipt["cleanup"]["term_sent"])
            self.assertTrue(receipt["cleanup"]["kill_sent"])
            self.assertTrue(receipt["cleanup"]["reaped"])
            self.assertEqual(receipt["exit_code"],-9)
            self.assertFalse(receipt["sampler_alive"])
            self.assertFalse(receipt["stdout_reader_thread_created"])
            self.assertFalse(Path("/proc",str(receipt["pid"])).exists())
            with self.assertRaises(ChildProcessError):
                os.waitpid(receipt["pid"],os.WNOHANG)
            self.assertEqual({thread.ident for thread in threading.enumerate()},original_threads)

    def test_partial_stdout_cannot_hide_owned_child_progress_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            binary=root/"partial-child"
            binary.write_text(f"#!{sys.executable}\nimport os,signal,sys,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nsys.stdout.write('partial native line');sys.stdout.flush()\nwhile True:time.sleep(1)\n")
            binary.chmod(0o700)
            args=argparse.Namespace(output=root,device="rocm:0",binary=binary,bytes=4096,replays=1,
                external_observer=None,identity_cache={},sample_ms=5)
            original_threads={thread.ident for thread in threading.enumerate()}
            with mock.patch.object(driver,"TERMINATION_GRACE_SECONDS",0.05), mock.patch.object(
                    driver,"PROGRESS_TIMEOUT_SECONDS",0.2):
                with self.assertRaisesRegex(RuntimeError,"partial evidence retained"):
                    driver.run_one(args,"partial-failure",1)
            receipt=json.loads((root/"partial-failure/process-result.json").read_text())
            self.assertTrue(receipt["cleanup"]["reaped"])
            self.assertFalse(Path("/proc",str(receipt["pid"])).exists())
            self.assertFalse(receipt["sampler_alive"])
            self.assertFalse(receipt["stdout_reader_thread_created"])
            self.assertEqual({thread.ident for thread in threading.enumerate()},original_threads)

    def test_continuous_non_json_flood_cannot_extend_checkpoint_deadline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            binary=root/"flood-child"
            pidfile=root/"flood.pid"
            binary.write_text(f"#!{sys.executable}\nimport os,signal\nfrom pathlib import Path\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nPath({str(pidfile)!r}).write_text(str(os.getpid()))\npayload=b'non-json diagnostic line\\n'*4096\nwhile True:os.write(1,payload)\n")
            binary.chmod(0o700)
            args=argparse.Namespace(output=root,device="cuda:0",binary=binary,bytes=4096,replays=1,
                external_observer=None,identity_cache={},sample_ms=5)
            original_threads={thread.ident for thread in threading.enumerate()}
            watchdog_fired=[]
            def bound_original_code_failure():
                watchdog_fired.append(True)
                if pidfile.exists():
                    try:os.killpg(int(pidfile.read_text()),signal.SIGKILL)
                    except ProcessLookupError:pass
            watchdog=threading.Timer(2.0,bound_original_code_failure)
            watchdog.start()
            started=time.monotonic()
            try:
                with mock.patch.object(driver,"TERMINATION_GRACE_SECONDS",0.05), mock.patch.object(
                        driver,"PROGRESS_TIMEOUT_SECONDS",0.2):
                    with self.assertRaisesRegex(RuntimeError,"Native child emitted no progress"):
                        driver.run_one(args,"flood-failure",1)
            finally:
                watchdog.cancel()
                watchdog.join()
            self.assertFalse(watchdog_fired,"Readable noise bypassed the actual checkpoint deadline")
            self.assertLess(time.monotonic()-started,2.0)
            receipt=json.loads((root/"flood-failure/process-result.json").read_text())
            self.assertTrue(receipt["cleanup"]["reaped"])
            self.assertFalse(Path("/proc",str(receipt["pid"])).exists())
            self.assertFalse(receipt["sampler_alive"])
            self.assertEqual({thread.ident for thread in threading.enumerate()},original_threads)

    def test_native_probe_owns_process_entry_and_original_graph_commitment(self):
        source=(ROOT/"tests/v2/integration/backends/ReadAheadNativeResourceCalibration.cpp").read_text()
        main=source[source.index("int main("):]
        self.assertLess(main.index("GPUDeviceProcessLifetime process_lifetime;"),main.index("llaminar2::test::run("))
        self.assertIn("GPUGraphMemoryContract::reservationBytesPerExecutable",source)
        self.assertLess(source.index("PhysicalMemoryOwner::NativeGraphExecutable,graphBytes"),source.index("worker->createGraphCapture(primary)"))
        self.assertLess(source.index("auto graphReservation=memory->reserveNewAllocations"),source.index("worker->createGraphCapture(primary)"))
        self.assertLess(source.index("completeExclusiveModelRetirement(std::move(ticket))"),source.index("graphReservation={};"))

    def test_capture_uses_native_exact_stream_read_and_preserves_failure_retirement(self):
        folder=ROOT/"tests/v2/integration/backends"
        source=(folder/"ReadAheadNativeResourceCalibration.cpp").read_text()
        body=source[source.index("const auto body="):source.index("const auto complete=")]
        self.assertIn("backend->deviceCopyAsync(",body)
        self.assertNotIn("backend->deviceToDevice(",body)
        self.assertIn("options.device.ordinal,primary)",body)
        self.assertIn("diagnostic_failure_terminal_drained",source)
        self.assertIn("diagnostic_baseline_failure_cleanup_begin",source)
        self.assertLess(source.index("exclusive_context_retirement_complete"),source.index("std::rethrow_exception(failure)"))

    def test_every_missing_checkpoint_rejects(self):
        for missing in range(len(records())):
            value = records()
            del value[missing]
            with self.assertRaises((ValueError, IndexError, KeyError)):
                driver.validate_records(value, "cuda:0", 0)

    def test_closure_population_and_source_drift_reject(self):
        for key in ("device", "physical_uuid", "resource_source_sha256", "runtime_version"):
            value = records()
            value[2]["closure"][key] = "changed"
            with self.assertRaises(ValueError):
                driver.validate_records(value, "cuda:0", 0)
        value = records()
        value[1]["geometry"]["ordering_events"] = 3
        with self.assertRaises(ValueError):
            driver.validate_records(value, "cuda:0", 0)

    def test_incomplete_parent_and_family_lifetimes_reject(self):
        for field in ("lifetimes", "parent_terminal_lifetimes", "replays"):
            value = records()
            for record in value:
                record["geometry"][field] += 1
            with self.assertRaises(ValueError):
                driver.validate_records(value, "cuda:0", 0)

    def test_claimed_certificate_or_unretired_context_reject(self):
        for field, changed in (("qualified", True), ("certified_limit", 4096), ("production_admission", True)):
            value = records()
            value[1][field] = changed
            with self.assertRaises(ValueError):
                driver.validate_records(value, "cuda:0", 0)
        value = records()
        value[-1]["extra"]["runtime_post_reset_state"] = "unverified"
        with self.assertRaises(ValueError):
            driver.validate_records(value, "cuda:0", 0)


if __name__ == "__main__":
    unittest.main()
