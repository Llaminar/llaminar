#!/usr/bin/env python3
"""Fail HTTP certification on new AMD/NVIDIA kernel-driver diagnostics.

The harness brackets the complete server lifetime, including planning, weight
loading and teardown. Kernel boot identity and an exact retained-log cursor
separate this cell from older warnings without trusting wall-clock time. A
wrapped/cleared log or unreadable diagnostics is a failed observation, never a
clean certificate. This observer reads the serving host's shared kernel; it
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
import subprocess
import sys
import tempfile

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


def begin(state: Path) -> None:
    """Arm a fresh cell before any server work; never reuse a prior boundary."""
    if state.exists():
        raise ValueError("Driver checkpoint already exists; require a fresh cell artifact")
    snapshot = read_snapshot()
    # A short contiguous suffix stays unique even for equal-time printk lines.
    # Losing any of it is conservatively a failed observation, not a clean log.
    publish(state, {"schema": 1, "state": "armed", "boot_id": snapshot.boot_id,
                    "reader": snapshot.reader.value,
                    "reader_container": snapshot.container,
                    "cursor": [asdict(row) for row in snapshot.records[-8:]]})


def finish(state: Path, report: Path) -> bool:
    """Close the interval after teardown and persist every new driver warning."""
    if report.exists():
        raise ValueError("Driver report already exists; cannot certify an interval twice")
    evidence = {"schema": 1, "complete": False, "passed": False,
                "checkpoint": str(state), "new_record_count": 0,
                "records": [], "findings": [], "error": None}
    checkpoint = None
    try:
        checkpoint = json.loads(state.read_text())
        after = read_snapshot(KernelReader(checkpoint["reader"]))
        rows = new_records(checkpoint, after)
        evidence.update(complete=True, boot_id=after.boot_id, reader=after.reader.value,
                        reader_container=after.container,
                        new_record_count=len(rows),
                        records=[asdict(row) for row in rows],
                        findings=[asdict(row) for row in rows if driver_diagnostic(row)])
        evidence["passed"] = not evidence["findings"]
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        evidence["error"] = str(error)
    if isinstance(checkpoint, dict) and checkpoint.get("state") == "armed":
        publish(state, {**checkpoint, "state": "closed"})
    publish(report, evidence)
    for row in evidence["findings"]:
        print(f"[gpu-driver] FAIL {row['time']:.6f} {row['pri']}: {row['msg']}", flush=True)
    if evidence["error"]:
        print(f"[gpu-driver] FAIL {evidence['error']}", flush=True)
    return evidence["passed"]


def validate_evidence(directory: Path) -> None:
    """Require one complete clean interval before an outer HTTP certificate."""
    reports = list(directory.glob("*.driver-diagnostics.json"))
    if len(reports) != 1:
        raise ValueError("HTTP cell requires exactly one GPU driver diagnostic report")
    evidence = json.loads(reports[0].read_text())
    if (not isinstance(evidence, dict) or
            evidence.get("schema") != 1 or evidence.get("complete") is not True or
            evidence.get("passed") is not True or evidence.get("findings") != [] or
            evidence.get("error") is not None or not evidence.get("boot_id") or
            evidence.get("reader") not in {scope.value for scope in KernelReader} or
            not isinstance(evidence.get("records"), list) or
            type(evidence.get("new_record_count")) is not int or
            evidence["new_record_count"] != len(evidence["records"])):
        raise ValueError("HTTP cell has missing, failed or incomplete GPU driver diagnostics")
    # Inspect retained messages too: a stale/incorrect summary cannot certify
    # an artifact containing a fault, even if its producer marked it green.
    if any(driver_diagnostic(KernelRecord.parse(row)) for row in evidence["records"]):
        raise ValueError("HTTP cell contains a GPU driver diagnostic despite its passing summary")


def main(argv: list[str] | None = None) -> int:
    """Expose the same bookends to direct, generation and container HTTP runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("begin", "finish"))
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.phase == "begin":
        begin(args.state)
        return 0
    if args.report is None:
        parser.error("finish requires --report")
    return 0 if finish(args.state, args.report) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[gpu-driver] FAIL {error}", file=sys.stderr)
        sys.exit(1)
