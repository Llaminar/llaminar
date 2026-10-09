#!/usr/bin/env python3
"""Fail HTTP certification on new AMD/NVIDIA kernel-driver diagnostics.

The harness brackets the complete server lifetime, including planning, weight
loading and teardown. An owned collector advances an exact kernel-log cursor
throughout that lifetime and appends observed records to a bounded-memory
journal. Rotation after collection is harmless; loss between observations,
unreadable diagnostics or a retired collector cannot produce a certificate. This observer reads the serving host's shared kernel; it
does not clear logs, change driver settings, or participate in inference.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid

READER_CONTAINER_ENV = "LLAMINAR_E2E_KERNEL_READER_CONTAINER"


class KernelReader(str, Enum):
    """The authenticated local privilege scope, not an arbitrary shell command."""

    DIRECT = "direct"
    SUDO = "sudo"
    TOOLS_CONTAINER = "tools_container"

    def command(self, container: str | None = None) -> list[str]:
        """Read structured priorities and monotonic kernel timestamps."""
        if self is KernelReader.TOOLS_CONTAINER:
            if not container or not re.fullmatch(r"llaminar-suite-driver-[0-9a-f]{32}", container):
                raise ValueError("Kernel reader requires its explicit owned suite-driver container")
            # The tools process retains the model-cache UID. A root exec in
            # that same SYSLOG-capable tools container owns only this read;
            # neither the inference container nor its privileges are changed.
            return ["docker", "exec", "--user", "0:0", container, "/bin/sh", "-ec",
                    "cat /proc/sys/kernel/random/boot_id; exec /usr/bin/dmesg --json --decode"]
        return (["sudo", "-n"] if self is KernelReader.SUDO else []) + [
            "dmesg", "--json", "--decode"]


@dataclass(frozen=True)
class KernelRecord:
    """One immutable kernel record; message equality authenticates the cursor."""

    time: float
    fac: str
    pri: str
    msg: str

    @classmethod
    def parse(cls, value: dict) -> KernelRecord:
        """Reject malformed records instead of silently dropping diagnostics."""
        if (not isinstance(value, dict) or
                type(value.get("time")) not in (float, int) or
                not math.isfinite(value["time"]) or value["time"] < 0 or
                any(not isinstance(value.get(key), str) for key in ("fac", "pri", "msg"))):
            raise ValueError("Malformed kernel diagnostic record")
        return cls(float(value["time"]), value["fac"], value["pri"], value["msg"])


@dataclass(frozen=True)
class KernelSnapshot:
    """A read from one boot through one explicit kernel-log access scope."""

    boot_id: str
    reader: KernelReader
    records: tuple[KernelRecord, ...]
    container: str | None = None


_DRIVER = re.compile(r"\b(?:amdgpu|amdkfd|kfd|nvrm|nvidia(?:[-_](?:uvm|modeset|drm|peermem))?)\b", re.I)
_PROBLEM = re.compile(
    r"\b(?:warn(?:ing)?|error|fault|fail(?:ed|ure)?|timeout|timed out|overflow|"
    r"reset(?:ting)?|fallen off|xid)\b|callbacks suppressed|hogged CPU", re.I)
_BAD_PRIORITIES = {"emerg", "alert", "crit", "err", "warn", "warning"}
_BOOT_ID = Path("/proc/sys/kernel/random/boot_id")


def driver_diagnostic(record: KernelRecord) -> bool:
    """Recognize driver severity and faults that some drivers log at INFO.

    NVIDIA Xid, AMD IH overflow and SVM workqueue warnings must not depend on
    printk priority being correct. Conversely a generic backtrace's module
    inventory does not attribute an unrelated kernel warning to every driver.
    """
    if record.fac != "kern" or record.msg.startswith("Modules linked in:"):
        return False
    attributed = _DRIVER.search(record.msg) or re.search(
        r"\b\w*ih_get_wptr:.*callbacks suppressed", record.msg, re.I)
    return bool(attributed and (record.pri in _BAD_PRIORITIES or _PROBLEM.search(record.msg)))


def read_snapshot(reader: KernelReader | None = None) -> KernelSnapshot:
    """Read the local kernel, using noninteractive sudo only for log access.

    Once a scope is admitted, finish uses that same scope. Containers need
    CAP_SYSLOG and read-only /dev/kmsg; ordinary host users need readable dmesg
    or passwordless sudo for this command. No missing-permission opt-out is
    offered.
    """
    errors = []
    container = os.environ.get(READER_CONTAINER_ENV)
    scopes = ((reader,) if reader is not None else
              (KernelReader.TOOLS_CONTAINER,) if container else
              (KernelReader.DIRECT, KernelReader.SUDO))
    for scope in scopes:
        before = _BOOT_ID.read_text().strip()
        try:
            result = subprocess.run(scope.command(container), capture_output=True, text=True,
                                    check=False, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(f"{scope.value}: {error}")
            continue
        if result.returncode:
            errors.append(f"{scope.value}: {result.stderr.strip()}")
            continue
        after = _BOOT_ID.read_text().strip()
        if not before or before != after:
            raise ValueError("Kernel boot changed during diagnostic read")
        raw = result.stdout
        if scope is KernelReader.TOOLS_CONTAINER:
            remote_boot, separator, raw = raw.partition("\n")
            if not separator or remote_boot.strip() != before:
                raise ValueError("Kernel reader container does not share the serving harness's boot")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("dmesg"), list):
            raise ValueError("Kernel reader returned no structured dmesg records")
        records = tuple(KernelRecord.parse(row) for row in payload["dmesg"])
        if not records:
            raise ValueError("Empty kernel log cannot anchor a certification interval")
        return KernelSnapshot(before, scope, records,
                              container if scope is KernelReader.TOOLS_CONTAINER else None)
    raise RuntimeError("Cannot read GPU driver diagnostics; require readable /dev/kmsg "
                       "and CAP_SYSLOG or noninteractive sudo for dmesg. " + "; ".join(errors))


def new_records(checkpoint: dict, after: KernelSnapshot) -> tuple[KernelRecord, ...]:
    """Select only this interval, rejecting reboot, loss and ambiguous cursors."""
    if (not isinstance(checkpoint, dict) or checkpoint.get("schema") != 1 or checkpoint.get("state") != "armed" or
            checkpoint.get("boot_id") != after.boot_id or
            checkpoint.get("reader") != after.reader.value or
            checkpoint.get("reader_container") != after.container):
        raise ValueError("Driver checkpoint boot, reader or lifecycle does not match")
    cursor = tuple(KernelRecord.parse(row) for row in checkpoint.get("cursor", []))
    if not cursor:
        raise ValueError("Driver checkpoint has no retained-log cursor")
    matches = [index + len(cursor) for index in range(len(after.records) - len(cursor) + 1)
               if after.records[index:index + len(cursor)] == cursor]
    if len(matches) != 1:
        raise ValueError("Kernel log cursor lost or ambiguous: ring wrapped/cleared; cannot certify cell")
    return after.records[matches[0]:]


def publish(path: Path, value: dict) -> None:
    """Publish a complete artifact atomically, including failed observations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            name = handle.name
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(name, path)
        name = None
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


# These deadlines cover observer handshakes and at most two ten-second kernel
# reads. They never limit model loading, inference, reasoning or coding turns.
_CONTROL_SECONDS = 30.0
_POLL_SECONDS = 1.0
_MAX_LINE_BYTES = 1024 * 1024
_FINDING_PREVIEW = 32


def process_identity(pid: int) -> dict | None:
    """Bind a live native PID to its start tick; a zombie owns no collector."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("Invalid observer process identity")
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return None if fields[0] in ('Z', 'X') else {'pid': pid, 'start_ticks': int(fields[19])}


def process_is_live(identity: dict) -> bool:
    """Prevent PID reuse from authenticating a vanished owner or collector."""
    if (not isinstance(identity, dict) or set(identity) != {'pid', 'start_ticks'} or
            type(identity['start_ticks']) is not int or identity['start_ticks'] <= 0):
        raise ValueError("Incomplete observer process identity")
    return process_identity(identity['pid']) == identity


def window_paths(state: Path) -> tuple[Path, Path]:
    """Keep the journal and collector diagnostics beside their admission."""
    return (state.with_name(state.name + '.records.jsonl'),
            state.with_name(state.name + '.collector.log'))


def control_address(window_id: str) -> str:
    """Use an unguessable local socket without Unix pathname length limits."""
    if not isinstance(window_id, str) or re.fullmatch('[0-9a-f]{32}', window_id) is None:
        raise ValueError("Invalid driver observation window")
    return '\0llaminar-driver-' + window_id


def journal_line(stream, value: dict) -> None:
    """Append one bounded record; never retain a model-lifetime Python list."""
    raw = (json.dumps(value, ensure_ascii=True, separators=(',', ':')) + '\n').encode()
    if len(raw) > _MAX_LINE_BYTES:
        raise ValueError("Kernel record exceeds the observer record bound")
    stream.write(raw)


def collect(state: Path) -> bool:
    """Own continuous cursor advancement until authenticated normal closure.

    The collector is armed before server admission. Each read uses the original
    privilege scope and boot, and authenticates its predecessor's suffix. It
    preserves all intervening messages before advancing that cursor. A close
    request causes a fresh final read, covering server teardown even if another
    observation was already in progress when finish requested closure.
    """
    saved = json.loads(state.read_text())
    if saved.get('schema') != 2 or saved.get('state') != 'starting':
        raise ValueError("Collector requires a fresh starting admission")
    address = control_address(saved['window_id'])
    identity = process_identity(os.getpid())
    if identity is None:
        raise RuntimeError("Collector has no live process identity")
    saved = {**saved, 'collector': identity}
    journal, _ = window_paths(state)
    count = findings_count = snapshots = 0
    findings = []
    cursor = saved['cursor']
    complete = False
    error = None
    closing = False
    stream = None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as control:
            control.bind(address)
            control.settimeout(_POLL_SECONDS)
            stream = journal.open('xb')
            journal_line(stream, {'kind': 'begin', 'schema': 2,
                **{key: saved[key] for key in ('window_id', 'boot_id', 'reader', 'reader_container')},
                'cursor': cursor})
            while True:
                if not process_is_live(saved['owner']):
                    raise RuntimeError("Driver window owner retired before normal closure")
                after = read_snapshot(KernelReader(saved['reader']))
                rows = new_records({'schema': 1, 'state': 'armed',
                    **{key: saved[key] for key in ('boot_id', 'reader', 'reader_container')},
                    'cursor': cursor}, after)
                snapshots += 1
                for row in rows:
                    count += 1
                    journal_line(stream, {'kind': 'record', 'ordinal': count, **asdict(row)})
                    if driver_diagnostic(row):
                        findings_count += 1
                        if len(findings) < _FINDING_PREVIEW:
                            findings.append(asdict(row))
                cursor = [asdict(row) for row in after.records[-8:]]
                stream.flush()
                if closing:
                    complete = True
                    break
                if saved['state'] == 'starting':
                    saved = {**saved, 'state': 'armed'}
                    publish(state, saved)
                try:
                    raw = control.recv(4097)
                except socket.timeout:
                    continue
                if len(raw) > 4096 or json.loads(raw) != {
                        'action': 'close', 'window_id': saved['window_id']}:
                    raise ValueError("Driver collector received an invalid close request")
                closing = True
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as failure:
        error = str(failure)
    finally:
        if stream is not None:
            try:
                journal_line(stream, {'kind': 'end', 'window_id': saved['window_id'],
                    'complete': complete, 'new_record_count': count,
                    'snapshot_count': snapshots, 'cursor': cursor})
                stream.flush()
                os.fsync(stream.fileno())
            except (OSError, ValueError) as failure:
                complete = False
                error = str(failure)
            finally:
                stream.close()
        outcome = {'schema': 2, 'complete': complete, 'passed': complete and findings_count == 0,
            **{key: saved[key] for key in ('window_id', 'boot_id', 'reader', 'reader_container', 'collector')},
            'journal': journal.name, 'new_record_count': count, 'snapshot_count': snapshots,
            'findings': findings, 'finding_count': findings_count, 'error': error}
        publish(state, {**saved, 'state': 'retired', 'outcome': outcome})
    return outcome['passed']


def begin(state: Path) -> None:
    """Start and authenticate the observer before returning server admission."""
    if state.exists():
        raise ValueError("Driver checkpoint already exists; require a fresh cell artifact")
    snapshot = read_snapshot()
    owner = process_identity(os.getppid())
    if owner is None:
        raise RuntimeError("Driver window requires its live harness owner")
    admission = {'schema': 2, 'state': 'starting', 'window_id': uuid.uuid4().hex,
        'boot_id': snapshot.boot_id, 'reader': snapshot.reader.value,
        'reader_container': snapshot.container, 'owner': owner,
        'cursor': [asdict(row) for row in snapshot.records[-8:]]}
    state.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation claims the lifetime before any daemon is started.
    descriptor = os.open(state, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(admission, stream)
    _, log = window_paths(state)
    worker = None
    try:
        with log.open('xb') as stream:
            worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                'collect', '--state', str(state.resolve())], stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + _CONTROL_SECONDS
        while time.monotonic() < deadline:
            current = json.loads(state.read_text())
            if current.get('window_id') != admission['window_id']:
                raise ValueError("Driver observer admission changed during startup")
            if current.get('state') == 'armed':
                if current.get('collector') != process_identity(worker.pid):
                    raise RuntimeError("Driver collector did not authenticate its native process")
                return
            if worker.poll() is not None or current.get('state') == 'retired':
                raise RuntimeError("Driver collector failed startup: " + str(current.get('outcome', {})))
            time.sleep(.01)
        raise RuntimeError("Driver collector startup handshake did not complete")
    except BaseException:
        if worker is not None and worker.poll() is None:
            # This is only a failed observer startup; no inference was admitted.
            os.killpg(worker.pid, signal.SIGTERM)
            worker.wait(timeout=_CONTROL_SECONDS)
        raise


def inspect_journal(directory: Path, evidence: dict) -> None:
    """Stream and authenticate every retained record before admitting evidence.

    The journal proves its exact boot/window, all record ordinals, both ends and
    the terminal cursor. GPU findings are recomputed from the retained records;
    neither a stale summary nor truncating a log can produce a clean receipt.
    """
    name = evidence.get('journal')
    if not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..'):
        raise ValueError("Driver evidence has no local journal")
    header = footer = None
    count = finding_count = 0
    findings = []
    cursor = []
    with (directory / name).open('rb') as stream:
        while raw := stream.readline(_MAX_LINE_BYTES + 1):
            if len(raw) > _MAX_LINE_BYTES or not raw.endswith(b'\n'):
                raise ValueError("Truncated or oversized driver journal record")
            row = json.loads(raw)
            if not isinstance(row, dict) or footer is not None:
                raise ValueError("Malformed driver journal or data after closure")
            if header is None:
                if row.get('kind') != 'begin' or row.get('schema') != 2:
                    raise ValueError("Missing driver journal admission")
                for key in ('window_id', 'boot_id', 'reader', 'reader_container'):
                    if row.get(key) != evidence.get(key):
                        raise ValueError("Driver journal admission does not match its report")
                cursor = [KernelRecord.parse(value) for value in row.get('cursor', [])]
                if not 1 <= len(cursor) <= 8:
                    raise ValueError("Missing initial driver cursor")
                header = row
            elif row.get('kind') == 'record':
                count += 1
                if type(row.get('ordinal')) is not int or row['ordinal'] != count:
                    raise ValueError("Missing or reordered driver journal records")
                record = KernelRecord.parse(row)
                if record.time < cursor[-1].time:
                    raise ValueError("Driver journal timestamps moved backwards")
                cursor = (cursor + [record])[-8:]
                if driver_diagnostic(record):
                    finding_count += 1
                    if len(findings) < _FINDING_PREVIEW:
                        findings.append(asdict(record))
            elif row.get('kind') == 'end':
                if (row.get('window_id') != evidence.get('window_id') or row.get('complete') is not True or
                        type(row.get('new_record_count')) is not int or row['new_record_count'] != count or
                        type(row.get('snapshot_count')) is not int or row['snapshot_count'] < 2 or
                        row['snapshot_count'] != evidence.get('snapshot_count') or
                        row.get('cursor') != [asdict(value) for value in cursor]):
                    raise ValueError("Incomplete or mismatched driver journal closure")
                footer = row
            else:
                raise ValueError("Unknown driver journal record")
    if (header is None or footer is None or evidence.get('new_record_count') != count or
            evidence.get('finding_count') != finding_count or evidence.get('findings') != findings):
        raise ValueError("Missing driver journal boundary or incorrect summary")
    if finding_count:
        raise ValueError("HTTP cell contains a GPU driver diagnostic despite its passing summary")


def retire_failed_collector(identity: dict) -> None:
    """Retire only the authenticated failed observer before publishing closure.

    A pidfd binds signals and exit observation to one native lifetime. Recheck
    start ticks after opening it so PID reuse cannot select another process.
    These bounded cleanup waits apply only after observer failure, never to
    inference or its output. The original failed evidence remains failed even
    when exceptional retirement succeeds.
    """
    if not process_is_live(identity):
        return
    try:
        descriptor = os.pidfd_open(identity['pid'])
    except ProcessLookupError:
        return
    try:
        if not process_is_live(identity):
            return
        exited = select.poll()
        exited.register(descriptor, select.POLLIN)
        for action in (signal.SIGTERM, signal.SIGKILL):
            try:
                signal.pidfd_send_signal(descriptor, action)
            except ProcessLookupError:
                return
            events = exited.poll(max(1, int(_CONTROL_SECONDS * 1000)))
            if events:
                if any(fd != descriptor or mask & (select.POLLERR | select.POLLNVAL)
                       or not mask & (select.POLLIN | select.POLLHUP) for fd, mask in events):
                    raise RuntimeError('Failed driver collector exit observation lost its pidfd')
                return
        raise RuntimeError('Failed driver collector remains alive after exceptional retirement')
    finally:
        os.close(descriptor)


def finish(state: Path, report: Path) -> bool:
    """Close the live observer after teardown; never restart missing evidence."""
    if report.exists():
        raise ValueError("Driver report already exists; cannot certify an interval twice")
    if state.parent.resolve() != report.parent.resolve():
        raise ValueError("Driver report and its journal must share an artifact directory")
    evidence = {'schema': 2, 'complete': False, 'passed': False, 'checkpoint': str(state),
        'new_record_count': 0, 'finding_count': 0, 'findings': [], 'error': None}
    saved = None
    identity = None
    retired = False
    try:
        saved = json.loads(state.read_text())
        if saved.get('schema') != 2 or saved.get('state') not in ('armed', 'retired'):
            raise ValueError("Driver observation is not an armed or retired window")
        identity = saved['collector']
        if saved['state'] == 'armed':
            if not process_is_live(identity):
                raise RuntimeError("Driver collector retired without complete evidence")
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as control:
                control.connect(control_address(saved['window_id']))
                control.send(json.dumps({'action': 'close', 'window_id': saved['window_id']}).encode())
        deadline = time.monotonic() + _CONTROL_SECONDS
        while process_is_live(identity):
            if time.monotonic() >= deadline:
                raise RuntimeError("Driver collector did not retire after its final read")
            time.sleep(.01)
        retired = True
        terminal = json.loads(state.read_text())
        if (terminal.get('window_id') != saved['window_id'] or terminal.get('collector') != identity or
                terminal.get('state') != 'retired'):
            raise RuntimeError("Driver collector left no authenticated terminal outcome")
        evidence.update(terminal['outcome'])
        if evidence['passed']:
            inspect_journal(report.parent, evidence)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        evidence.update(complete=False, passed=False, error=str(error))
        if identity is not None:
            try:
                retire_failed_collector(identity)
                retired = True
            except (OSError, ValueError, RuntimeError) as cleanup_error:
                evidence['error'] += '; observer retirement failed: ' + str(cleanup_error)
    # A failed cleanup cannot publish a closed owner or overwrite a worker
    # that can still publish its own terminal state. Preserve the live admission
    # and failed report so the exact outstanding process remains reviewable.
    if retired:
        publish(state, {**saved, 'state': 'closed', 'outcome': evidence})
    publish(report, evidence)
    for row in evidence['findings']:
        print(f"[gpu-driver] FAIL {row['time']:.6f} {row['pri']}: {row['msg']}", flush=True)
    if evidence['error']:
        print(f"[gpu-driver] FAIL {evidence['error']}", flush=True)
    return evidence['passed']


def validate_evidence(directory: Path) -> None:
    """Require one complete, continuously observed and clean driver lifetime."""
    reports = list(directory.glob('*.driver-diagnostics.json'))
    if len(reports) != 1:
        raise ValueError("HTTP cell requires exactly one GPU driver diagnostic report")
    evidence = json.loads(reports[0].read_text())
    if (not isinstance(evidence, dict) or evidence.get('schema') != 2 or
            evidence.get('complete') is not True or evidence.get('passed') is not True or
            evidence.get('findings') != [] or evidence.get('finding_count') != 0 or
            evidence.get('error') is not None or not evidence.get('boot_id') or
            evidence.get('reader') not in {scope.value for scope in KernelReader} or
            type(evidence.get('new_record_count')) is not int or evidence['new_record_count'] < 0):
        raise ValueError("HTTP cell has missing, failed or incomplete GPU driver diagnostics")
    control_address(evidence.get('window_id'))
    inspect_journal(directory, evidence)


def main(argv: list[str] | None = None) -> int:
    """Expose the shared lifecycle and its owned continuous collector process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('begin', 'finish', 'collect'))
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    if args.phase == 'begin':
        begin(args.state)
        return 0
    if args.phase == 'collect':
        return 0 if collect(args.state) else 1
    if args.report is None:
        parser.error('finish requires --report')
    return 0 if finish(args.state, args.report) else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f'[gpu-driver] FAIL {error}', file=sys.stderr)
        sys.exit(1)
