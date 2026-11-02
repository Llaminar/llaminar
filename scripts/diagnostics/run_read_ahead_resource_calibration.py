#!/usr/bin/env python3
"""Process-isolated explicit native resource diagnostics; never certifies a limit.

Only the root/user runs this driver against hardware. It preserves partial
records and failures, samples process peaks outside the native child, and
requires the child's real retirement receipt. Optional external observations
are retained verbatim with provider identity; they are not authenticated by
merely supplying a script. No production certificate or PMA ledger is created.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import codecs
import selectors
import signal
import stat
import subprocess
import sys
import threading
import time

TERMINATION_GRACE_SECONDS = 5.0
PROGRESS_TIMEOUT_SECONDS = 30.0

SCHEMA = "llaminar.read_ahead_native_resource_calibration.v1"


def digest(path: Path) -> str:
    """Hash exact executable/source input once, without loading its native code."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def process_snapshot(pid: int) -> dict:
    """Observe process memory/maps without importing a CUDA/HIP library."""
    result = {"monotonic_ns": time.monotonic_ns(), "pid": pid}
    for name in ("status", "smaps_rollup", "maps"):
        try:
            result[name] = Path(f"/proc/{pid}/{name}").read_text()
        except OSError as error:
            result[name + "_error"] = str(error)
    return result


def mapped_identities(snapshot: dict, cache: dict) -> dict:
    """Bind actual mapped native libraries by content/build ID, with one read per file identity."""
    observed = {}
    for line in snapshot.get("maps", "").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        name = fields[5]
        try:
            path = Path(name)
            metadata = path.stat()
            # Driver mappings include character devices. Opening them can
            # invoke native driver operations; they are never ELF identities.
            # Directories/FIFOs likewise cannot participate in this observer.
            if not stat.S_ISREG(metadata.st_mode):
                continue
            key = (name, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
            if key not in cache:
                with path.open("rb") as stream:
                    if stream.read(4) != b"\x7fELF":
                        continue
                build = subprocess.run(["readelf", "-n", str(path)], capture_output=True, text=True, timeout=30)
                ids = [text.split("Build ID:", 1)[1].strip() for text in build.stdout.splitlines() if "Build ID:" in text]
                cache[key] = {"sha256": digest(path), "elf_build_ids": ids, "size": metadata.st_size}
            observed[name] = cache[key]
        except (OSError, subprocess.SubprocessError) as error:
            observed[name] = {"unavailable": str(error)}
    return observed


def validate_records(records: list[dict], device: str, fail_creator: int) -> None:
    """Reject incomplete/incompatible observations instead of inventing peaks or bounds."""
    if not records:
        raise ValueError("Child emitted no source-bound calibration checkpoints")
    pid = records[0]["pid"]
    for index, record in enumerate(records):
        if record.get("schema") != SCHEMA or record.get("sequence") != index or record.get("pid") != pid:
            raise ValueError("Malformed or reordered calibration checkpoint")
        if record.get("qualified") is not False or record.get("certified_limit") is not None or record.get("production_admission") is not False:
            raise ValueError("Diagnostic attempted to manufacture a production certificate")
        if str(record["closure"].get("device")).lower() != device.lower():
            raise ValueError("Calibration switched its actual native endpoint")
        geometry = record["geometry"]
        if geometry != records[0]["geometry"]:
            raise ValueError("Native calibration geometry drifted within one process")
        if (geometry.get("side_streams"), geometry.get("ordering_events"), geometry.get("concurrent_parents")) != (1, 2, 1):
            raise ValueError("Calibration population changed")
    stages = [record["stage"] for record in records]
    required = {"before_backend_lookup", "original_context_ready", "explicit_parent_payload_allocated", "before_parent_terminal", "after_parent_terminal_created", "first_parent_terminal_record_complete", "after_parent_terminal_destroyed", "parent_payload_and_terminal_baseline", "after_destroy_join", "after_destroy_fork", "after_destroy_stream", "explicit_baseline_owners_destroyed", "exclusive_context_retirement_complete"}
    required |= {"injected_creator_failure"} if fail_creator else {"after_stream", "after_fork", "after_join", "lazy_first_use_complete", "capture_definition_complete", "capture_instantiated", "first_parent_replay_complete", "capture_destroyed"}
    if not required.issubset(stages) or stages[-1] != "exclusive_context_retirement_complete":
        raise ValueError("Missing creation/use/capture/teardown/context retirement evidence")
    closure = next(record["closure"] for record in records if record["stage"] == "original_context_ready")
    for key in ("physical_uuid", "architecture", "runtime_version", "driver_api_version", "resource_source_sha256", "calibration_source_sha256"):
        if not closure.get(key):
            raise ValueError("Incomplete concrete native closure: " + key)
    for record in records[1:]:
        if record["closure"] != closure:
            raise ValueError("Native closure drifted within one process")
    for key in ("resource_source_sha256", "calibration_source_sha256"):
        value = closure[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Malformed source hash identity")
    if fail_creator:
        failures = [record for record in records if record["stage"] == "injected_creator_failure"]
        if len(failures) != 1 or failures[0]["extra"].get("creator") != fail_creator:
            raise ValueError("Wrong native creator rejection boundary")
    geometry = records[1]["geometry"]
    for parent in range(geometry["parent_terminal_lifetimes"]):
        for stage in ("before_parent_terminal", "after_parent_terminal_created", "first_parent_terminal_record_complete", "after_parent_terminal_destroyed"):
            if sum(record["stage"] == stage and record["extra"].get("parent_lifetime") == parent for record in records) != 1:
                raise ValueError("Incomplete independent parent terminal lifetime")
    for lifetime in range(1 if fail_creator else geometry["lifetimes"]):
        for stage in ("after_destroy_join", "after_destroy_fork", "after_destroy_stream"):
            if sum(record["stage"] == stage and record["extra"].get("lifetime") == lifetime for record in records) != 1:
                raise ValueError("Incomplete actual resource teardown lifetime")
        if not fail_creator:
            for stage in ("after_stream", "after_fork", "after_join", "lazy_first_use_complete", "capture_definition_complete", "capture_instantiated", "capture_destroyed"):
                if sum(record["stage"] == stage and record["extra"].get("lifetime") == lifetime for record in records) != 1:
                    raise ValueError("Incomplete resource creation/capture lifetime")
            replayed = [record["extra"].get("replay") for record in records if record["stage"] in ("first_parent_replay_complete", "parent_replay_complete") and record["extra"].get("lifetime") == lifetime]
            if replayed != list(range(geometry["replays"])):
                raise ValueError("Incomplete actual retained replay count")
    terminal = records[-1]["extra"]
    if terminal.get("runtime_post_reset_state") != "quiescent" or not terminal.get("reset_invoked") or not terminal.get("retired_context_generation") or not terminal.get("retired_runtime_generation") or not terminal.get("successor_runtime_generation"):
        raise ValueError("Actual context retirement receipt incomplete")
    if bool(fail_creator) != bool(terminal.get("injected_failure")):
        raise ValueError("Creator rejection control did not execute")


def external_observation(provider: Path | None, device: str) -> dict:
    """Preserve an explicitly selected external observer without claiming it is qualified."""
    if provider is None:
        return {"supported": False, "reason": "No external GPU observer selected", "authenticated": False}
    result = subprocess.run([str(provider), "--device", device], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError("External observer failed: " + result.stderr[-2000:])
    return {"supported": True, "authenticated": False, "provider_sha256": digest(provider), "stdout": result.stdout, "stderr": result.stderr}


def retire_owned_child(child: subprocess.Popen) -> dict:
    """Retire only the original new-session group, escalating TERM to KILL.

    This follows the canonical campaign's owned-session TERM/KILL/wait pattern.
    Keep the native child's PID directly (a guardian wrapper would change the
    PID whose /proc and runtime observations the receipt authenticates). Reap
    after escalation before closing any observation files or returning.
    """
    actions={"term_sent": False, "kill_sent": False, "reaped": False}
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
            actions["term_sent"]=True
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
                actions["kill_sent"]=True
            except ProcessLookupError:
                pass
            child.wait()
    else:
        child.wait()
    actions["reaped"]=child.returncode is not None
    return actions


def run_one(args: argparse.Namespace, label: str, lifetimes: int, fail_creator: int = 0, *, child_command: list[str] | None = None, expected_schema: str = SCHEMA, record_validator=validate_records, checkpoint_validator=None) -> dict:
    """Own one fresh subprocess and preserve all actual checkpoint/exit evidence.

    Other explicit raw diagnostics may supply their exact argv/schema/validator;
    process ownership, external prelaunch observation, regular ELF authentication,
    bounded handshake and retirement handling remain this single implementation.
    """
    folder = args.output / label
    folder.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(LLAMINAR_FORCE_CPU_ONLY_STARTUP="0", LLAMINAR_SKIP_CUDA_STARTUP="0" if args.device.startswith("cuda:") else "1", LLAMINAR_SKIP_ROCM_STARTUP="0" if args.device.startswith("rocm:") else "1")
    command = list(child_command) if child_command is not None else [str(args.binary), "--device", args.device, "--bytes", str(args.bytes), "--replays", str(args.replays), "--lifetimes", str(lifetimes), "--parent-terminal-lifetimes", str(lifetimes), "--handshake"]
    if fail_creator:
        command += ["--fail-creator", str(fail_creator)]
    before = external_observation(args.external_observer, args.device)
    records: list[dict] = []
    identity_cache = args.identity_cache
    stop = threading.Event()
    with (folder / "stderr.log").open("w") as stderr, (folder / "stdout.log").open("wb") as stdout, (folder / "process-samples.jsonl").open("w") as samples:
        child = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, bufsize=0, start_new_session=True)
        def sample() -> None:
            while not stop.is_set():
                samples.write(json.dumps(process_snapshot(child.pid)) + "\n")
                samples.flush()
                stop.wait(args.sample_ms / 1000)
        sampler = threading.Thread(target=sample, name="external-process-observer")
        sampler.start()
        failure = None
        code = None
        cleanup = None
        selector = selectors.DefaultSelector()
        assert child.stdout is not None
        os.set_blocking(child.stdout.fileno(), False)
        selector.register(child.stdout, selectors.EVENT_READ)
        decoder=codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending=""
        eof=False
        progress_deadline=time.monotonic()+PROGRESS_TIMEOUT_SECONDS
        try:
            assert child.stdin is not None
            while not eof:
                now=time.monotonic()
                if now>=progress_deadline:
                    raise RuntimeError("Native child emitted no progress for the standard 30-second boundary")
                if not selector.select(timeout=progress_deadline-now):
                    raise RuntimeError("Native child emitted no progress for the standard 30-second boundary")
                chunk=os.read(child.stdout.fileno(),65536)
                if chunk:
                    stdout.write(chunk)
                    stdout.flush()
                    pending+=decoder.decode(chunk)
                else:
                    eof=True
                    pending+=decoder.decode(b"",final=True)
                complete=pending.splitlines(keepends=True)
                pending=""
                if complete and not complete[-1].endswith("\n") and not eof:
                    pending=complete.pop()
                for line in complete:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue  # Retained verbatim logger output, not a checkpoint.
                    if not isinstance(record, dict) or record.get("schema") != expected_schema:
                        continue
                    if checkpoint_validator is not None:
                        checkpoint_validator(record)
                    records.append(record)
                    snapshot = process_snapshot(child.pid)
                    checkpoint = {"child": record, "external_process": snapshot, "mapped_library_identities": mapped_identities(snapshot, identity_cache), "external_device": external_observation(args.external_observer, args.device)}
                    (folder / f"checkpoint-{len(records)-1:04d}.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
                    child.stdin.write(b"continue\n")
                    child.stdin.flush()
                    progress_deadline=time.monotonic()+PROGRESS_TIMEOUT_SECONDS
            code = child.wait(timeout=PROGRESS_TIMEOUT_SECONDS)
        except BaseException as error:
            failure = error
        finally:
            stop.set()
            # No stdout reader thread exists: nonblocking reads share this
            # owner, so a partial native line cannot hide a progress timeout.
            try:
                cleanup=retire_owned_child(child)
                # Preserve the child's remaining raw output after it is reaped;
                # never start another reader or block on an inherited pipe.
                cleanup["drained_stdout_bytes"]=0
                try:
                    while True:
                        tail=os.read(child.stdout.fileno(),65536)
                        if not tail:
                            break
                        stdout.write(tail)
                        cleanup["drained_stdout_bytes"]+=len(tail)
                    stdout.flush()
                except BlockingIOError:
                    pass
            finally:
                code=child.returncode
                selector.close()
                child.stdin.close()
                child.stdout.close()
                sampler.join()
    try:
        after = external_observation(args.external_observer, args.device)
    except Exception as error:
        after={"error": str(error)}
        failure=failure or error
    device_samples = [record["device"]["driver_free_bytes"] for record in records if "device" in record]
    record = {"observed_driver_free_min": min(device_samples) if device_samples else None, "observed_driver_free_max": max(device_samples) if device_samples else None, "observed_process_maxrss_kib": max((record["host"]["maxrss_kib"] for record in records), default=None), "observations_are_not_limits": True, "label": label, "command": command, "pid": child.pid, "exit_code": code, "before": before, "after": after, "checkpoints": len(records), "peak_observation_resolution_ms": args.sample_ms, "qualified": False, "certified_limit": None, "failure": str(failure) if failure else None, "cleanup": cleanup, "sampler_alive": sampler.is_alive(), "stdout_reader_thread_created": False}
    (folder / "process-result.json").write_text(json.dumps(record, indent=2) + "\n")
    if failure:
        raise RuntimeError(f"{label} diagnostic failed; partial evidence retained in {folder}: {failure}") from failure
    if code:
        raise RuntimeError(f"{label} child exited {code}; partial evidence retained in {folder}")
    record_validator(records, args.device, fail_creator)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bytes", type=int, default=4096)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--cold-processes", type=int, default=20)
    parser.add_argument("--warm-lifetimes", type=int, default=20)
    parser.add_argument("--sample-ms", type=int, default=20)
    parser.add_argument("--external-observer", type=Path)
    parser.add_argument("--creator-rejection-controls", action="store_true")
    args = parser.parse_args()
    args.device = args.device.lower()
    args.binary = args.binary.resolve(strict=True)
    if args.external_observer:
        args.external_observer = args.external_observer.resolve(strict=True)
    if not args.device.startswith(("cuda:", "rocm:")) or int(args.device.split(":")[1]) < 0 or min(args.bytes, args.replays, args.cold_processes, args.warm_lifetimes, args.sample_ms) <= 0:
        parser.error("Invalid explicit device/positive diagnostic geometry")
    args.identity_cache = {}
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "llaminar.read_ahead_native_resource_dataset.v1", "binary": str(args.binary), "binary_sha256": digest(args.binary), "driver_script_sha256": digest(Path(__file__)), "platform": platform.uname()._asdict(), "device": args.device, "qualified": False, "certified_limit": None}
    for name, path in {"nvidia_kernel_driver": "/proc/driver/nvidia/version", "amdgpu_kernel_version": "/sys/module/amdgpu/version", "amdgpu_srcversion": "/sys/module/amdgpu/srcversion"}.items():
        try:
            manifest[name] = Path(path).read_text()
        except OSError as error:
            manifest[name] = {"unavailable": str(error)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    try:
        for index in range(args.cold_processes):
            results.append(run_one(args, f"cold-{index:03d}", 1))
        results.append(run_one(args, "warm-repeated", args.warm_lifetimes))
        if args.creator_rejection_controls:
            for creator in (1, 2, 3):
                results.append(run_one(args, f"reject-creator-{creator}", 1, creator))
    finally:
        (args.output / "dataset.json").write_text(json.dumps({"manifest": manifest, "processes": results, "qualified": False, "certified_limit": None}, indent=2) + "\n")


if __name__ == "__main__":
    main()
