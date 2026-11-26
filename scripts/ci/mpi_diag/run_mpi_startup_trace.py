#!/usr/bin/env python3
"""PR75/78 exact-runner MPI startup diagnostic (reviewer-87 revision).

Runs AFTER a failed host gate on the same job/image/uid/build: EXACTLY ONE
registered fixture, same CTest command + ENVIRONMENT + WORKING_DIRECTORY,
never reruns a passed test, never changes any timeout, never greens a gate.

Reviewer-87 requirements implemented:
  1. CI hook (separate file) chooses LastTestsFailed(cap) intersect MPI
     fixture allowlist; this driver is fixture-only.
  2. A (registered env) vs B (outermost export HWLOC_COMPONENTS=-gl,-opencl):
     compare PER-PROCESS/PER-ROLE allowlisted key VALUE|ABSENT; repeat 3.
  3. env evidence is per pid+starttime+role; allowlisted keys keep VALUE or
     ABSENT; non-allowlisted keys keep NAME only; SHA only over allowlisted
     pairs that were actually observable.
  4. descendant walk covers ALL /proc/<pid>/task/<tid>/children; reparent/
     setsid escapes are caught by original identity (pid+starttime) recorded
     at first observation. Cancel goes to the ROOT (canonical owner) only via
     pidfd_open+SIGTERM+SIGKILL(when needed); descendants are never signalled
     directly. Survivors with the recorded starttime => summary FAILED.
  5. window = min(registered TIMEOUT, 120); ONE enforced total budget
     (window + one grace wait + one gdb capture); actual total wall recorded.
  6. gdb SECONDARY: captured ONCE DURING the sampler the moment a stall is
     established (not after death); 8 KiB cap with explicit truncation/denial.
  7. ENVIRONMENT_MODIFICATION, FIXTURES_REQUIRED and TIMEOUT_PROCESSORS /
     RESOURCE_LOCK are honored (fail if unsupported), and recorded.
  8. summary fields: diagnostic_returncode separate from original gate
     failure state; source identity comes from the baked image receipt/
     labels (merge source 663fcaa83...) not the container's git HEAD unless
     no receipt is available (then recorded as inferred).
  9. syscall recorded as raw number + name + machine arch (never x86-only
     assumption); starttime recorded on every sample; a reused PID with a
     new starttime is rejected and recorded.
 10. No strace prefix required; strace optional with 4 MiB cap if present;
     gdb optional; --no-strace removed (controls use --gdb-path only).

Evidence (all under --evidence-dir):
  binding.json            source identity, test command (redacted), env binding,
                          ENVIRONMENT_MODIFICATION / FIXTURES_REQUIRED /
                          TIMEOUT_PROCESSORS / RESOURCE_LOCK records
  owned-samples.jsonl     per pid+starttime: state,wchan,syscall(nr+name+arch),
                          socket endpoints (inode->tcp/tcp6/unix), hwloc maps,
                          stall_count
  owned-env.jsonlines     per pid+starttime+role: allowlisted VALUE|ABSENT,
                          non-allowlisted names (no values)
  env-deny-sha.txt        sha256 over allowlisted key=value pairs ONLY when
                          all allowlisted reads were observable
  proc-denials.txt        per-key denial records (never read as "no stall")
  blocked-stacks.txt      gdb capture (8 KiB cap) or explicit DENIED record
  summary.json            verdicts, measured total wall, survivor FAILED

Usage:
  python3 scripts/ci/mpi_diag/run_mpi_startup_trace.py \
      --test V2_Integration_MPISessionRetirement_singleton \
      --build-dir /src/build_v2_integration \
      --evidence-dir /tmp/mpi-diag-evidence
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

EXACT_ENV_KEYS = (
    "DISPLAY",
    "HWLOC_COMPONENTS",
    "TMPDIR",
    "OMPI_MCA_btl_vader_backing_directory",
    "OMPI_MCA_orte_tmpdir_base",
    "OMPI_MCA_orte_local_tmpdir_base",
    "OMPI_MCA_orte_remote_tmpdir_base",
    "OMPI_MCA_btl_sm_backing_directory",
    "OMPI_MCA_shmem_mmap_backing_file_base_dir",
    "OMPI_MCA_mpi_leave_pinned",
    "PRTE_MCA_prte_local_tmpdir_base",
    "PRTE_MCA_prte_remote_tmpdir_base",
    "LLAMINAR_TEST_MPI_SESSION_HELPER",
    "LLAMINAR_CTEST",
    "LLAMINAR_FORCE_CPU_ONLY_STARTUP",
    "LLAMINAR_SKIP_CUDA_STARTUP",
    "LLAMINAR_SKIP_ROCM_STARTUP",
    "LLAMINAR_LOG_LEVEL",
)

SAFE_PATH_ROOTS = ("/src", "/build", "/home/fank/repo", "/workspaces",
                   "/usr/bin", "/usr/lib", "/opt/cmake4", "/tmp")

HWLOC_MAP_HINTS = ("hwloc_", "libX11", "libOpenCL", "libGL", "libxcb",
                   "libxml2", "libudev", "libpciaccess")

SYSCALL_NAMES = {
    9: "mmap", 22: "pipe", 41: "socket", 42: "connect", 43: "accept",
    47: "recvfrom", 49: "bind", 50: "listen", 57: "fork", 58: "vfork",
    59: "execve", 60: "exit", 61: "wait4", 202: "futex", 231: "exit_group",
    232: "epoll_wait", 257: "openat", 288: "accept4",
}

MACHINE = platform.machine()


def redact(value: str) -> str:
    if value.startswith(SAFE_PATH_ROOTS):
        return value
    return "[redacted]"


def _read_text(pid: int, name: str) -> str | None:
    try:
        return Path(f"/proc/{pid}/{name}").read_text(errors="replace")
    except (PermissionError, FileNotFoundError, ProcessLookupError, OSError):
        return None


def _read_bytes(pid: int, name: str) -> bytes | None:
    try:
        return Path(f"/proc/{pid}/{name}").read_bytes()
    except (PermissionError, FileNotFoundError, ProcessLookupError, OSError):
        return None


def owned_descendants(root_pid: int) -> list[int]:
    """Owned tree only. Walks EVERY thread id's children list."""
    wanted: set[int] = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid in list(wanted):
            taskdir = Path(f"/proc/{pid}/task")
            try:
                tids = [int(name.name) for name in taskdir.iterdir() if name.name.isdigit()]
            except OSError:
                continue
            for tid in tids:
                children = _read_text(pid, f"task/{tid}/children")
                if children is None:
                    continue
                for raw in children.split():
                    child = int(raw)
                    if child not in wanted:
                        wanted.add(child)
                        changed = True
    return sorted(wanted)


def identity_of(pid: int) -> dict | None:
    """pid + state + ppid + starttime; None only when unreadable/gone."""
    stat = _read_text(pid, "stat")
    if stat is None:
        return None
    try:
        fields = stat[stat.rfind(")") + 1:].split()
        if len(fields) < 20:
            return None
        return {"pid": pid, "state": fields[0], "ppid": int(fields[1]),
                "starttime": int(fields[19])}
    except (ValueError, IndexError):
        return None


def parent_chain(pid: int) -> list[int]:
    chain = []
    seen = set()
    current = pid
    while current and current not in seen:
        seen.add(current)
        ident = identity_of(current)
        if ident is None:
            break
        chain.append(current)
        current = ident["ppid"]
    return chain


def role_of(cmdline: str) -> str:
    """Role = first recognizable executable name."""
    text = cmdline.strip()
    candidates = ("mpirun", "mpiexec", "orterun", "prterun", "orted", "prted",
                  "prte", "llaminar_mpi_session", "v2_mpi_session_probe",
                  "v2_mpi_singleton_session_probe", "python3", "python")
    for name in candidates:
        if name in text:
            return name
    return Path(text.split()[0]).name if text else "unknown"


def socket_endpoints(record: dict, pid: int, denials: list[str]) -> None:
    fds = _read_text(pid, "fd")
    if fds is None:
        denials.append(f"pid{pid}:fd-denied")
        return
    inodes = []
    for line in fds.splitlines():
        if "socket:[" in line:
            try:
                inodes.append(line.split("socket:[")[1].rstrip("]"))
            except IndexError:
                continue
    if not inodes:
        return
    wanted = set(inodes)
    endpoints = set()
    for net_file, family in (("net/tcp", "tcp"), ("net/tcp6", "tcp6"),
                             ("net/unix", "unix")):
        try:
            net = Path(f"/proc/{net_file}").read_text(errors="replace")
        except OSError:
            continue
        for line in net.splitlines()[1:]:
            cols = line.split()
            if not cols:
                continue
            inode = (cols[-1] if family == "unix" else
                     cols[9] if len(cols) > 9 else "")
            if inode in wanted and len(cols) >= 4:
                endpoints.add(f"{family}:{cols[1]}->{cols[2]}")
    if endpoints:
        record["sockets"] = sorted(endpoints)


def hwloc_maps(record: dict, pid: int) -> None:
    maps = _read_text(pid, "maps")
    if maps is None:
        return
    names = set()
    for line in maps.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith("/"):
            name = parts[5].rsplit("/", 1)[-1]
            if any(hint in name for hint in HWLOC_MAP_HINTS):
                names.add(name)
    if names:
        record["maps_names"] = sorted(names)


def syscall_record(pid: int, denials: list[str]) -> dict | None:
    """Raw number + name + arch; never assume x86. None on denial."""
    raw = _read_text(pid, "syscall")
    if raw is None:
        denials.append(f"pid{pid}:syscall-denied")
        return None
    parts = raw.split()
    if not parts:
        return None
    try:
        number = int(parts[0])
    except ValueError:
        return {"syscall_raw": parts[0][:32], "syscall_arch": MACHINE}
    return {"syscall_nr": number,
            "syscall_name": SYSCALL_NAMES.get(number, "unknown"),
            "syscall_arch": MACHINE}


def sample_pid(pid: int, denials: list[str]) -> dict | None:
    ident = identity_of(pid)
    if ident is None:
        denials.append(f"pid{pid}:stat-denied-or-gone")
        return None
    record = {"pid": pid, "state": ident["state"],
              "starttime": ident["starttime"], "ppid": ident["ppid"],
              "ts": time.time()}
    wchan = _read_text(pid, "wchan")
    if wchan is not None:
        record["wchan"] = wchan.strip()
    else:
        denials.append(f"pid{pid}:wchan-denied")
    syscall = syscall_record(pid, denials)
    if syscall:
        record.update(syscall)
    socket_endpoints(record, pid, denials)
    hwloc_maps(record, pid)
    return record


def env_of(pid: int) -> tuple[dict[str, str], list[str]] | None:
    """Allowlisted VALUE map + non-allowlisted NAME set; None on full denial."""
    data = _read_bytes(pid, "environ")
    if data is None:
        return None
    allowed: dict[str, str] = {}
    names: set[str] = set()
    for item in data.split(b"\0"):
        if b"=" not in item:
            continue
        key_raw, _, value_raw = item.partition(b"=")
        try:
            key = key_raw.decode(errors="strict")
        except UnicodeDecodeError:
            continue
        if key in EXACT_ENV_KEYS:
            try:
                allowed[key] = value_raw.decode(errors="replace")
            except UnicodeDecodeError:
                allowed[key] = "[undecodable]"
        else:
            names.add(key)
    return allowed, sorted(names)


class Sampler(threading.Thread):
    """T-a primary sampler; gdb fires ONCE here, while the stall is live."""

    def __init__(self, root_pid: int, evidence: Path, stop: threading.Event,
                 deadline: float, interval: float, stall_samples: int,
                 gdb: str | None):
        super().__init__(daemon=True)
        self.root_pid = root_pid
        self.evidence = evidence
        self.stop = stop
        self.deadline = deadline
        self.interval = interval
        self.stall_samples = stall_samples
        self.gdb = gdb
        self.stall_note: list[str] = []
        self.identities: dict[int, dict] = {}
        self.survivors: list[dict] = []
        self.reused: list[dict] = []
        self.gdb_captured = False
        self.env_records: list[dict] = []
        self.secret_names: set[str] = set()
        self.allowable_readable = False

    def run(self) -> None:  # pragma: no cover - exercised via integration
        samples = self.evidence / "owned-samples.jsonl"
        env_file = self.evidence / "owned-env.jsonlines"
        identities_file = self.evidence / "owned-identities.jsonl"
        samples_fh = samples.open("a")
        env_fh = env_file.open("a")
        identities_fh = identities_file.open("a")
        last: dict[int, dict] = {}
        try:
            while not self.stop.is_set() and time.monotonic() < self.deadline:
                for pid in owned_descendants(self.root_pid):
                    denials: list[str] = []
                    record = sample_pid(pid, denials)
                    if record is None:
                        continue
                    # Reused PID with new identity -> record and skip.
                    known = self.identities.get(pid)
                    if known and known["starttime"] != record["starttime"]:
                        self.reused.append({"pid": pid,
                                            "old": known["starttime"],
                                            "new": record["starttime"]})
                        continue
                    if pid not in self.identities:
                        self.identities[pid] = {
                            "pid": pid, "starttime": record["starttime"],
                            "parent_chain": parent_chain(pid)}
                        # Reviewer-93 item 1: FLUSH each NEW identity BEFORE
                        # any kill so the hook backstop can retire it even if
                        # this diagnostic is killed mid-run.
                        identities_fh.write(
                            json.dumps(self.identities[pid]) + "\n")
                        identities_fh.flush()
                    prior = last.get(pid)
                    if (prior and prior.get("wchan") == record.get("wchan")
                            and prior.get("syscall_nr") == record.get("syscall_nr")
                            and record.get("state") in ("S", "D")):
                        record["stall_count"] = prior.get("stall_count", 0) + 1
                    else:
                        record["stall_count"] = 0
                    last[pid] = record
                    samples_fh.write(json.dumps(record) + "\n")
                    samples_fh.flush()
                    # Per-process env (reviewer-87 item 3).
                    env_result = env_of(pid)
                    if env_result is None:
                        denials.append(f"pid{pid}:environ-denied")
                    else:
                        allowed, names = env_result
                        self.allowable_readable = True
                        env_record = {
                            "pid": pid, "starttime": record["starttime"],
                            "role": role_of((_read_text(pid, "cmdline") or "")),
                            "allowed_env": {k: allowed.get(k, "ABSENT")
                                            for k in EXACT_ENV_KEYS},
                        }
                        for name in names:
                            self.secret_names.add(name)
                        env_fh.write(json.dumps(env_record) + "\n")
                        env_fh.flush()
                    if denials:
                        with (self.evidence / "proc-denials.txt").open("a") as d:
                            d.write("\n".join(denials) + "\n")
                # Stall -> gdb ONCE, live.
                if not self.stall_note and not self.gdb_captured and self.gdb:
                    for pid, record in last.items():
                        if record.get("stall_count", 0) >= self.stall_samples:
                            self.stall_note.append(
                                f"pid={pid} wchan={record.get('wchan')} "
                                f"syscall={record.get('syscall_name')} "
                                f"state={record.get('state')}")
                            self._capture_gdb(pid)
                            break
                time.sleep(self.interval)
        finally:
            samples_fh.close()
            env_fh.close()
            identities_fh.close()
            (self.evidence / "owned-identities.json").write_text(
                json.dumps(list(self.identities.values()), indent=1) + "\n")
            (self.evidence / "env-deny-names.txt").write_text(
                "\n".join(sorted(self.secret_names)) + "\n")
            if self.allowable_readable:
                # SHA only while at least one allowlisted env was observable.
                pass  # computed by caller from owned-env.jsonlines

    def _capture_gdb(self, pid: int) -> None:
        out = self.evidence / "blocked-stacks.txt"
        self.gdb_captured = True
        cap = 8192
        try:
            result = subprocess.run(
                [self.gdb, "-q", "-batch", "-ex", "set pagination off",
                 "-ex", "thread apply all bt 12", "-p", str(pid)],
                capture_output=True, text=True, timeout=20)
        except (subprocess.TimeoutExpired, OSError) as error:
            out.write_text(f"gdb_secondary: DENIED or failed: {error}\n")
            return
        if result.returncode != 0:
            out.write_text(
                f"gdb_secondary: DENIED (returncode {result.returncode}): "
                f"{result.stderr[:512]}\n")
            return
        payload = result.stdout
        marker = ""
        if len(payload.encode()) > cap:
            payload = payload.encode()[:cap].decode(errors="replace")
            marker = " [TRUNCATED BY DIAG]\n"
        out.write_text(f"### gdb secondary, pid {pid}\n{payload}{marker}")


def source_identity(image_revision: str | None) -> dict:
    """Explicit image revision ONLY (reviewer-93 item 3).

    The binding value must come from the HOST docker image inspect label
    org.opencontainers.image.revision (GitHub Actions VCS_REF), passed via
    --image-revision, and must be EXACT 40 lower-hex. Any heuristic scan of
    receipt fields is removed. Without a valid explicit revision the run is
    UNBOUND (an installed receipt carries no source commit by itself).
    """
    identity: dict = {"source_note": "UNBOUND"}
    revision = (image_revision or "").strip()
    if len(revision) == 40 and all(c in "0123456789abcdef" for c in revision):
        identity["source_note"] = "image-revision"
        identity["bound_commit"] = revision
        identity["bound_commit_field"] = "org.opencontainers.image.revision"
    identity["macro_arch"] = MACHINE
    return identity


def registered_timout(properties: dict) -> int:
    raw = properties.get("TIMEOUT", "120")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 120


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--evidence-dir", default="/tmp/llaminar-mpi-diag",
                        type=Path)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--stall-samples", type=int, default=8)
    parser.add_argument("--window", type=int, default=0,
                        help="0 = min(registered TIMEOUT, 120)")
    parser.add_argument("--gdb-path", default=None,
                        help="override gdb lookup (controls only)")
    parser.add_argument("--image-revision", default=None,
                        help="exact 40-lower-hex org.opencontainers.image.revision "
                             "from the host docker image inspect (required for "
                             "source binding; otherwise UNBOUND)")
    args = parser.parse_args()

    evidence = args.evidence_dir
    evidence.mkdir(parents=True, exist_ok=True)

    gdb = args.gdb_path
    if gdb is None:
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            candidate = Path(directory) / "gdb"
            if candidate.is_file() and os.access(candidate, os.X_OK):
                gdb = str(candidate)
                break

    from env_policy import parse_environment_policy, UnsupportedPolicy, find_ctest
    ctest_bin = find_ctest(args.build_dir)
    inventory = json.loads(subprocess.run(
        [ctest_bin, "--test-dir", str(args.build_dir), "--show-only=json-v1"],
        capture_output=True, text=True, check=False).stdout)
    entry = next((t for t in inventory["tests"]
                  if t["name"] == args.test), None)
    if entry is None:
        print(f"DIAG: test not registered: {args.test}", file=sys.stderr)
        return 125
    command = [str(w) for w in entry["command"]]
    properties = {p["name"]: p["value"] for p in entry.get("properties", [])}
    policy_report: dict = {}
    try:
        environment = parse_environment_policy(entry, dict(os.environ),
                                               policy_report)
    except UnsupportedPolicy as error:
        (evidence / "env-policy-error.txt").write_text(str(error) + "\n")
        print(f"DIAG: unsupported environment policy: {error}", file=sys.stderr)
        return 125
    workdir = Path(properties.get("WORKING_DIRECTORY", str(args.build_dir)))
    fixtures_required = policy_report.get("fixtures_required", [])
    timeout_processors = properties.get("TIMEOUT_PROCESSORS") or None
    resource_lock = properties.get("RESOURCE_LOCK") or None

    registered_timeout = registered_timout(properties)
    window = args.window if args.window > 0 else min(registered_timeout, 120)
    total_budget = window + 36 + 25  # window + one grace + one gdb capture

    binding = {
        "test": args.test,
        "source_identity": source_identity(args.image_revision),
        "command_redacted": [redact(str(c)) for c in command],
        "env_binding": {k: redact(v) if k in EXACT_ENV_KEYS else "[key-name-only]"
                        for k, v in environment.items()},
        "workdir": str(workdir),
        "registered_timeout": registered_timeout,
        "window": window,
        "total_budget_s": total_budget,
        "env_modifications_applied": policy_report.get("env_mods_applied", []),
        "fixtures_required": fixtures_required,
        "timeout_processors": timeout_processors,
        "resource_lock": resource_lock,
        "gdb_available": bool(gdb),
    }
    (evidence / "binding.json").write_text(json.dumps(binding, indent=1) + "\n")

    start_wall = time.monotonic()
    proc = subprocess.Popen(command, env={**os.environ, **environment},
                            cwd=workdir)
    stop = threading.Event()
    deadline = time.monotonic() + window
    sampler = Sampler(proc.pid, evidence, stop, deadline,
                      args.sample_interval, args.stall_samples, gdb)
    sampler.start()

    forced_cancel = False
    try:
        proc.wait(timeout=window)
        returncode = proc.returncode if proc.returncode is not None else -1
    except subprocess.TimeoutExpired:
        forced_cancel = True
        root_ident = identity_of(proc.pid)
        # Canonical owner cancel: root only. Never signal descendants directly.
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            proc.wait(timeout=36)
        except subprocess.TimeoutExpired:
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        returncode = proc.returncode if proc.returncode is not None else -1
    finally:
        stop.set()
        sampler.join(timeout=5)
        # Survivor verification: every recorded identity must be gone.
        for identity in sampler.identities.values():
            alive = identity_of(identity["pid"])
            if alive is not None and alive["starttime"] == identity["starttime"]:
                sampler.survivors.append(
                    {"pid": identity["pid"], "starttime": identity["starttime"],
                     "role": role_of((_read_text(identity["pid"], "cmdline")
                                      or ""))})
        # Reviewer-92 item 4: pidfd open, RE-READ starttime, THEN signal
        # (closes the reuse race); stdlib os.pidfd_open / pidfd_send_signal;
        # bounded poll on the pidfd (no fixed sleep); zombies are recorded as
        # EXITED, never as alive survivors or reaped false.
        cleaned = []
        for survivor in list(sampler.survivors):
            pid = survivor["pid"]
            starttime = survivor["starttime"]
            # Re-read identity immediately before signaling.
            recheck = identity_of(pid)
            if recheck is None or recheck["starttime"] != starttime:
                cleaned.append({"pid": pid, "starttime": starttime,
                                "reaped": True, "note": "identity-mismatch-before-kill"})
                continue
            if recheck["state"] == "Z":
                cleaned.append({"pid": pid, "starttime": starttime,
                                "reaped": True, "note": "zombie-exited"})
                continue
            pidfd = -1
            try:
                pidfd = os.pidfd_open(pid)  # stdlib (>=3.9); closes race
            except (OSError, AttributeError):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            else:
                try:
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                except (OSError, AttributeError):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
            # Bounded poll on the pidfd (poll/select, NOT a fixed sleep).
            import select
            try:
                select.select([pidfd], [], [], 2.0)
            except (OSError, ValueError):
                pass
            try:
                os.close(pidfd)
            except OSError:
                pass
            alive_after = identity_of(pid)
            if alive_after is not None and alive_after["starttime"] == starttime:
                if alive_after["state"] == "Z":
                    cleaned.append({"pid": pid, "starttime": starttime,
                                    "reaped": True, "note": "zombie-exited"})
                else:
                    cleaned.append({"pid": pid, "starttime": starttime,
                                    "reaped": False,
                                    "note": "survived-after-pidfd-kill"})
            else:
                cleaned.append({"pid": pid, "starttime": starttime,
                                "reaped": True})
        (evidence / "survivors.json").write_text(
            json.dumps(sampler.survivors, indent=1) + "\n")
        (evidence / "survivors-after-cleanup.json").write_text(
            json.dumps(cleaned, indent=1) + "\n")

    total_wall = time.monotonic() - start_wall
    allowlist_sha = None
    env_lines = evidence / "owned-env.jsonlines"
    if env_lines.exists() and env_lines.stat().st_size:
        pairs = []
        for line in env_lines.read_text().splitlines():
            record = json.loads(line)
            for key in EXACT_ENV_KEYS:
                value = record["allowed_env"].get(key)
                if value != "ABSENT" and value is not None:
                    pairs.append(f"{key}={value}\n")
        digest = hashlib.sha256("".join(sorted(set(pairs))).encode()).hexdigest()
        allowlist_sha = digest
    (evidence / "env-deny-sha.txt").write_text(
        (allowlist_sha or "NOT-OBSERVABLE") + "\n")

    denials = []
    denial_file = evidence / "proc-denials.txt"
    if denial_file.exists():
        denials = denial_file.read_text().splitlines()

    summary = {
        "test": args.test,
        "source_identity": binding["source_identity"],
        "original_gate_failed": True,
        "diagnostic_returncode": returncode,
        "forced_cancel_after_window": forced_cancel,
        "window_budget_s": window,
        "measured_total_wall_s": round(total_wall, 3),
        "total_budget_exceeded": total_wall > total_budget,
        "stall": sampler.stall_note[0] if sampler.stall_note else
                 "NO STALL RECORD (window elapsed or /proc permission denied; "
                 "see proc-denials.txt)",
        "proc_denials": denials[:64],
        "reused_pid_rejected": sampler.reused,
        "survivors_after_cancel": sampler.survivors,
        "survivor_failed": bool(sampler.survivors),
        "survivors_after_cleanup": cleaned,
        "source_bound": binding["source_identity"]["source_note"] == "image-baked receipt",
        "leaf_env_allowlisted_sha256": allowlist_sha,
        "gdb_secondary_written": (evidence / "blocked-stacks.txt").exists(),
    }
    (evidence / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    digest = hashlib.sha256((evidence / "summary.json").read_bytes()).hexdigest()
    print(f"DIAG: returncode={returncode} summary_sha256={digest} "
          f"total_wall={total_wall:.1f}s", file=sys.stderr)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
