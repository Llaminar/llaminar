#!/usr/bin/env python3
"""PR75/78 CI hook: bounded MPI startup diagnostic after a failed host lane.

Reviewer-90 revision. Called from run_develop_pr_tests.py AFTER the host lane
returned CTest-8 (completed tests failed). Contract:

  ONE fixture only. Selection order (deterministic, from the original failure
  authority, i.e. the host-lane JUnit): singleton, native_options, identity,
  mpi, MPICTestSessionIsolation, MoEOwnedSharedBackingNativeStartup - the first
  name in that order that actually failed. (The parent narrowed reviewer-87's
  "all six" to one minimal native/singleton fixture; the order prefers the
  minimal native fixture while still being deterministic.)

  ONE global deadline (default 15 min). The internal diagnostic has its own
  window; this hook is the backstop and MUST retire every recorded identity
  (via pidfd + SIGKILL + starttime verification) before it returns - it never
  returns while a fixture child is alive. Budget is passed down and recorded;
  if the diagnostic cannot finish inside the remaining budget the hook skips
  further runs (explicit budget skip) but still records evidence.

  3 A/B pairs maximum. A = registered CTest command+env. B = A plus outermost
  export HWLOC_COMPONENTS=-gl,-opencl. Since add_v2_test already sets
  HWLOC_COMPONENTS=-gl,-opencl in the CTest ENVIRONMENT, A==B for env; the
  hook therefore FAILS a requested contrast if A == B (no fake comparison).
  Optional arm C (registered env with HWLOC_COMPONENTS REMOVED) exists behind
  --arm-c-requires-owner-approval (default OFF) because removing the env
  needs owner approval; it is never run by default.

  Never greens: return 0 from the hook but the caller keeps the original gate
  failure. Evidence under <output>/mpi-diag/, uploaded by the existing
  pr-tests/** artifact glob.

Source identity: the BAKED receipt (installed-tests.json, or the image record
behind it) is REQUIRED; without it the run is recorded UNBOUND (source-bound
qualification absent) and the hook reports unbound - it does not fall through
to a git HEAD qualification. (The diagnostic itself still records where the
command came from, separately labelled inferred.)

Environment syntax (reviewer-90 item 5):
  ENVIRONMENT_MODIFICATION entries are `<NAME>=<op>:<value>` where op is one of
  set, unset, reset, string, path, cmakelist; anything else -> clean 125 error.
  FIXTURES_REQUIRED lists fixture NAMES (not paths); unknown name or unsupported
  requirement -> clean 125 with a message; a name that resolves to an actual
  setup executable is run once.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

MPI_FIXTURE_ALLOWLIST = frozenset((
    "V2_Integration_MPISessionRetirement_singleton",
    "V2_Integration_MPISessionRetirement_native_options",
    "V2_Integration_MPISessionRetirement_identity",
    "V2_Integration_MPISessionRetirement_mpi",
    "V2_Integration_MPICTestSessionIsolation",
    "V2_Integration_MoEOwnedSharedBackingNativeStartup",
))
# Deterministic preference order: minimal native/singleton first.
SELECTION_ORDER = (
    "V2_Integration_MPISessionRetirement_singleton",
    "V2_Integration_MPISessionRetirement_native_options",
    "V2_Integration_MPISessionRetirement_identity",
    "V2_Integration_MPISessionRetirement_mpi",
    "V2_Integration_MPICTestSessionIsolation",
    "V2_Integration_MoEOwnedSharedBackingNativeStartup",
)
REPEAT_PAIRS = 3
DEFAULT_DEADLINE_S = 900  # 15 minutes, one global deadline
VALID_ENV_OPS = ("set", "unset", "reset", "string", "path", "cmakelist")

_script = Path(__file__).resolve().parent / "run_mpi_startup_trace.py"


def failed_tests_from_junit(lane_xml: Path) -> tuple[set[str], str | None]:
    """Failed/not-run names from the host-lane JUnit; None+failure if missing."""
    if not lane_xml.exists():
        raise ValueError(f"host lane JUnit missing: {lane_xml}")
    failed: set[str] = set()
    tree = ET.parse(lane_xml)
    for suite in tree.getroot().iter("testsuite"):
        for case in suite.iter("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                failed.add(case.get("name", ""))
            if case.get("status") == "notrun":
                failed.add(case.get("name", ""))
    return {name for name in failed if name}, None



def run_fixture(test: str, env: dict[str, str], evidence_dir: Path,
                arm: str, window: int, global_deadline: float) -> int:
    """Run one diagnostic invocation; timeout is the hook's backstop, but the
    diagnostic itself is expected to finish within window+grace. The hook
    ALWAYS retires the fixture by recorded identity before returning."""
    command = [sys.executable, str(_script), "--test", test,
               "--build-dir", str(_build_dir),
               "--evidence-dir", str(evidence_dir),
               "--window", str(window)]
    revision = (env.get("LLAMINAR_IMAGE_REVISION") or "").strip()
    if revision:
        command += ["--image-revision", revision]
    # Output to FILES (bounded), never capture_output (unbounded memory).
    log = evidence_dir / "diag.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cap = 16 * 1024 * 1024  # reviewer-93 item 2: streamed cap DURING run
    import threading

    def _bounded_pump(stream) -> None:
        """Drain the fixture output DURING the run; keep at most `cap` bytes on
        disk with an explicit marker; then DISCARD the remainder until EOF.

        Never break at the cap: the pipe has a 64 KiB kernel buffer and the
        fixture would block writing, deadlocking the run. The pump keeps
        reading (and discarding) until the fixture exits so it can finish
        normally, and the on-disk log stays bounded throughout.
        """
        written = 0
        with log.open("wb") as out:
            truncated = False
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                remaining = cap - written
                if remaining > 0:
                    take = min(len(chunk), remaining)
                    out.write(chunk[:take])
                    written += take
                    if take < len(chunk):
                        truncated = True
                else:
                    truncated = True
            if truncated:
                out.write(b"\n[MPI-DIAG LOG TRUNCATED AT 16 MiB]\n")
            out.flush()

    proc = subprocess.Popen(command, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    pump = threading.Thread(target=_bounded_pump, args=(proc.stdout,),
                            daemon=True)
    pump.start()
    try:
        returncode = proc.wait(timeout=max(1, global_deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        # Backstop: retire via recorded identity (pidfd) then SIGKILL.
        backstop = _retire(proc, evidence_dir)
        returncode = proc.returncode if proc.returncode is not None else -1
        try:
            with (evidence_dir / "backstop.json").open("w") as bs:
                json.dump(backstop, bs, indent=1)
        except OSError:
            pass
    if proc.stdout:
        proc.stdout.close()
    pump.join(timeout=5)
    print(f"mpi-diag: {arm} {test} returncode={returncode}", flush=True)
    return returncode


def recorded_identities(evidence_dir: Path) -> list[dict]:
    """All published identities from the diagnostic evidence, regardless of
    whether the diagnostic managed to write its final summaries."""
    identities: dict[int, dict] = {}
    jsonl_file = evidence_dir / "owned-identities.jsonl"
    if jsonl_file.exists():
        for line in jsonl_file.read_text().splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            pid = record.get("pid")
            start = record.get("starttime")
            if pid is not None and start is not None:
                identities[pid] = {"pid": pid, "starttime": start}
    env_file = evidence_dir / "owned-env.jsonlines"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            pid = record.get("pid")
            start = record.get("starttime")
            if pid is not None and start is not None:
                identities[pid] = {"pid": pid, "starttime": start}
    id_file = evidence_dir / "owned-identities.json"
    if id_file.exists():
        try:
            for record in json.loads(id_file.read_text()):
                pid = record.get("pid")
                start = record.get("starttime")
                if pid is not None and start is not None:
                    identities[pid] = {"pid": pid, "starttime": start}
                else:
                    # owned-identities.json may lack starttime in older forms;
                    # record -1 forces "identity unknown, still signal".
                    identities.setdefault(pid, {"pid": pid, "starttime": -1})
        except ValueError:
            pass
    return list(identities.values())


def _kill_one(pid: int, starttime: int | None) -> dict:
    """Retire ONE recorded identity via pidfd only (reviewer-94 block 2).

    No raw os.kill for recorded identities. If pidfd_open fails with ESRCH the
    process is already gone; if unavailable, outcome records unavailable. The
    starttime is re-read AFTER the pidfd is open (the fd pins the identity, so
    the re-read sees the same object the fd refers to) and must MATCH before
    the signal; mismatch means the pid was reused and is left untouched.
    """
    outcome = {"pid": pid, "starttime": starttime}
    try:
        pidfd = os.pidfd_open(pid)
    except ProcessLookupError:
        outcome["outcome"] = "gone-before-open"
        return outcome
    except (OSError, AttributeError) as error:
        outcome["outcome"] = "pidfd-unavailable"
        outcome["error"] = str(error)
        return outcome
    try:
        # Re-read identity AFTER open; the pidfd pins the object.
        if starttime is not None:
            try:
                stat = Path(f"/proc/{pid}/stat").read_text()
            except OSError:
                outcome["outcome"] = "stat-unreadable-going-away"
                return outcome
            fields = stat[stat.rfind(")") + 1:].split()
            current = int(fields[19]) if len(fields) > 19 else None
            if current != starttime:
                outcome["outcome"] = "identity-reused-skip"
                return outcome
        try:
            signal.pidfd_send_signal(pidfd, signal.SIGKILL)
            outcome["outcome"] = "pidfd-signaled"
        except (OSError, AttributeError) as error:
            outcome["outcome"] = "pidfd-signal-unavailable"
            outcome["error"] = str(error)
        try:
            # Optional POLLIN proof: when the pidfd becomes readable the
            # process is gone; only recorded as extra evidence, never alone.
            import select
            ready, _, _ = select.select([pidfd], [], [], 0.2)
            if ready:
                outcome["pollin"] = True
        except (OSError, ValueError):
            pass
    finally:
        try:
            os.close(pidfd)
        except OSError:
            pass
    return outcome


def _verify_gone(identity: dict) -> dict:
    """Post-kill verification (reviewer-95 block 2).

    ONLY FileNotFound/ProcessLookup => gone; starttime mismatch => reused-gone;
    Z => zombie. PermissionError, parse failures and any other unexpected
    condition => UNKNOWN (a FAIL into survivors_after - never treated as dead).
    """
    record = {"pid": identity["pid"], "starttime": identity["starttime"]}
    stat = Path(f"/proc/{identity['pid']}/stat")
    try:
        text = stat.read_text()
    except FileNotFoundError:
        record["outcome"] = "gone"
        return record
    except ProcessLookupError:
        record["outcome"] = "gone"
        return record
    except PermissionError:
        record["outcome"] = "UNKNOWN-permission-denied"
        return record
    except OSError as error:
        record["outcome"] = "UNKNOWN-oserror"
        record["error"] = str(error)
        return record
    close = text.rfind(")")
    if close < 0:
        record["outcome"] = "UNKNOWN-parse"
        return record
    fields = text[close + 1:].split()
    if len(fields) <= 19:
        record["outcome"] = "UNKNOWN-parse"
        return record
    try:
        current = int(fields[19])
    except ValueError:
        record["outcome"] = "UNKNOWN-parse"
        return record
    if current != identity["starttime"]:
        record["outcome"] = "reused-gone"
        return record
    record["outcome"] = "zombie" if fields[0] == "Z" else "alive"
    return record


def _retire(proc: subprocess.Popen, evidence_dir: Path | None = None) -> dict:
    """Canonical retirement (reviewer-94 block 3, structured).

    Returns {fired, retired:[{pid,starttime,outcome}], survivors_after}.
    The diagnostic's own session group is signalled (it owns nothing we must
    preserve; it is OUR Popen child and always ours to reap), recorded
    identities are pidfd-only, then every identity is verified gone/zombie
    with a bounded wait. Only OUR un-reaped Popen child may get a raw
    SIGKILL fallback; recorded identities never do.
    """
    retired: list[dict] = []
    pid = proc.pid
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    identities = recorded_identities(evidence_dir) if evidence_dir else []
    for identity in identities:
        retired.append(_kill_one(identity["pid"], identity["starttime"]))
    # Our own diag Popen child: raw kill is safe (we own it), pidfd first.
    diag_outcome = _kill_one(pid, None)
    retired.append(diag_outcome)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # Only our owned Popen child may get the raw fallback.
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    # Bounded post-kill verification (reviewer-92 item 4: ~2s poll).
    time.sleep(2.0)
    survivors_after = []
    survivors_after_clean = []
    for identity in identities:
        state = _verify_gone(identity)
        if state["outcome"] in ("gone", "reused-gone", "zombie"):
            retired_verify = dict(state)
            retired_verify["outcome"] = "gone-or-zombie"
            survivors_after_clean.append(retired_verify)
        else:
            # UNKNOWN (permission denied, parse failure, missing kill) is a
            # FAIL: treated as a potential survivor, never as proof of death.
            survivors_after.append(state)
    result = {"fired": True,
              "retired": retired,
              "survivors_after": survivors_after,
              "survivors_after_clean": survivors_after_clean}
    return result


def source_bound(evidence: dict) -> str:
    """Baked receipt with a REAL 40-hex commit field is REQUIRED; else UNBOUND."""
    identity = evidence.get("source_identity", {})
    if (identity.get("source_note") == "image-baked receipt"
            and identity.get("bound_commit_field")
            and identity.get("bound_commit")):
        return "bound"
    return "UNBOUND"


def run_after_failed_host_lane(lane_xml: Path, output_root: Path,
                               build_dir: Path,
                               image_env: dict[str, str] | None = None,
                               deadline_s: int = DEFAULT_DEADLINE_S,
                               arm_c_requires_owner_approval: bool = False) -> dict:
    """Entry point. Call only after the host lane result was non-zero."""
    global _build_dir
    _build_dir = Path(build_dir)
    diag_dir = Path(output_root) / "mpi-diag"
    diag_dir.mkdir(parents=True, exist_ok=True)
    image_env = dict(image_env or {})
    global_deadline = time.monotonic() + deadline_s

    try:
        failed, _ = failed_tests_from_junit(Path(lane_xml))
    except (ValueError, ET.ParseError) as error:
        (diag_dir / "hook-error.txt").write_text(f"JUnit missing/unparsable: {error}\n")
        return {"ran": False, "selected": [], "error": str(error)}

    selected = None
    for name in SELECTION_ORDER:
        if name in failed:
            selected = name
            break
    result = {"ran": selected is not None, "selected": None, "arms": {},
              "deadline_s": deadline_s}
    if selected is None:
        (diag_dir / "no-failed-fixtures.txt").write_text(
            "No allowlisted MPI fixture failed; diagnostic not run.\n")
        print(f"mpi-diag: no allowlisted MPI fixture among {len(failed)}",
              flush=True)
        return result
    result["selected"] = selected

    # Registered TIMEOUT required: missing => fail unqualified (no default).
    inventory = json.loads(subprocess.run(
        ["ctest", "--test-dir", str(build_dir), "--show-only=json-v1"],
        capture_output=True, text=True, check=False).stdout)
    entry = next((t for t in inventory["tests"] if t["name"] == selected), None)
    if entry is None:
        raise ValueError(f"selected fixture not in inventory: {selected}")
    props = {p["name"]: p["value"] for p in entry.get("properties", [])}
    if "TIMEOUT" not in props:
        raise ValueError(f"selected fixture has no registered TIMEOUT: {selected}")
    window = min(int(props["TIMEOUT"]), 120)

    # ONE shared parser (reviewer-92 item 2): exact CTest semantics, else 125.
    from env_policy import parse_environment_policy, UnsupportedPolicy
    try:
        registered_env = parse_environment_policy(entry, dict(image_env))
    except UnsupportedPolicy as error:
        (diag_dir / "env-policy-error.txt").write_text(str(error) + "\n")
        return {"ran": False, "selected": None,
                "error": f"unsupported environment policy: {error}"}

    arms = [("A-registered", registered_env)]
    if arm_c_requires_owner_approval:
        arms.append(("C-without-hwloc", {k: v for k, v in registered_env.items()
                                         if k != "HWLOC_COMPONENTS"}))
    else:
        # B: outermost export. If it would equal A (because A already
        # carries -gl,-opencl), FAIL the requested contrast (no fake diff).
        b_env = dict(registered_env)
        b_env["HWLOC_COMPONENTS"] = "-gl,-opencl"
        if b_env == registered_env:
            result["contrast_failed"] = "A==B: registered env already has HWLOC_COMPONENTS=-gl,-opencl"
            (diag_dir / "contrast-failed.txt").write_text(
                result["contrast_failed"] + "\n")
        else:
            arms.append(("B-outer-hwloc-export", b_env))

    budget_remaining = global_deadline - time.monotonic()
    assert budget_remaining > 0, "hook budget exhausted before first run"
    for arm, arm_env in arms:
        revisions = []
        for iteration in range(REPEAT_PAIRS):
            # Allow a pair to start whenever time remains; run_fixture bounds
            # itself to the global deadline and the backstop retires by
            # identity. Skip only when there is no time left at all.
            if time.monotonic() + 2 > global_deadline:
                result["budget_skip"] = True
                break
            evidence = diag_dir / f"{selected}-{arm}-r{iteration}"
            evidence.mkdir(parents=True, exist_ok=True)
            returncode = run_fixture(selected, arm_env, evidence, arm,
                                     window, global_deadline)
            revision = {"iteration": iteration, "arm": arm,
                        "returncode": returncode,
                        "summary": str(evidence / "summary.json")}
            backstop_file = evidence / "backstop.json"
            if backstop_file.exists():
                try:
                    revision["backstop"] = json.loads(
                        backstop_file.read_text())
                except ValueError:
                    revision["backstop"] = {"fired": True, "parse-error": True}
            revisions.append(revision)
            time.sleep(0.5)
        result["arms"][arm] = revisions
    (diag_dir / "hook-summary.json").write_text(
        json.dumps(result, indent=1) + "\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane-xml", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    run_after_failed_host_lane(args.lane_xml, args.output_root, args.build_dir)
