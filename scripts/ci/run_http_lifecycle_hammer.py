#!/usr/bin/env python3
"""Stress one canonical Release HTTP cell's requests and complete server lifetimes.

This is an explicit diagnostic, not E2E/image certification. Inventory, model
staging, launch arguments, readiness, driver diagnostics and server retirement
remain owned by the existing production tooling. Repeat requests in one warm
server or repeat fresh lifetimes under one model-staging lease. Every failure
is terminal; each request and native retirement keep their bounded watchdogs.
"""
from __future__ import annotations

import argparse
from itertools import count
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_model_parity_e2e as e2e
from production_artifacts import image_cpu_isa, image_identity, write_json

sys.path.insert(0, str(e2e.ROOT / "tests/v2/e2e/server"))
from http_lifecycle_hammer import PrefixPressure, read_sequence
from long_context_checks import needle_max_tokens


def run_owned_harness(command: list[str], environment: dict[str, str], log) -> int:
    """Own one complete server lifetime until workload completion or interruption.

    The certified cell's overall deadline is intentionally not reused for an
    unbounded diagnostic. Native collective, readiness, request and shutdown
    watchdogs stay intact. This does not grant certification an infinite budget.
    """
    # Diagnostic attachment must be scoped to this exact lifetime. Export the
    # owner and Linux process birth identity, not a rank/socket assumption.
    owner_pid = os.getpid()
    owner_birth = Path(f"/proc/{owner_pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    environment = {**environment, "LLAMINAR_HTTP_HAMMER_OWNER_PID": str(owner_pid),
                   "LLAMINAR_HTTP_HAMMER_OWNER_BIRTH": owner_birth}
    with subprocess.Popen(command, cwd=e2e.ROOT, env=environment, stdout=log,
                          stderr=subprocess.STDOUT, start_new_session=True) as process:
        try:
            return process.wait()
        finally:
            # Also retire descendants if a harness fault bypassed its normal
            # shutdown. The shared helper addresses only this owned group.
            e2e.parity._terminate_process_group(process)


def main(argv: list[str] | None = None) -> int:
    """Select exactly one typed cell and reuse its full model/staging/serve contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=e2e.ROOT / "build_v2_integration")
    parser.add_argument("--binary", type=Path, default=e2e.ROOT / "build_v2_release/llaminar2")
    parser.add_argument("--container-image")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--backend", default=".*")
    parser.add_argument("--campaign", default=".*")
    parser.add_argument("--cell", required=True)
    parser.add_argument("--sequence", type=Path, required=True,
                        help="Preserved ordered request bodies with explicit independent oracles")
    parser.add_argument("--prefix-pressure", action="store_true",
                        help="Interleave canonical long needles and proved full/partial prefix restores")
    parser.add_argument("--cycle-limit", type=int, default=0,
                        help="Workload cycles per lifetime; 0 keeps one warm server until failure")
    parser.add_argument("--lifetime-limit", type=int, default=1,
                        help="Fresh server lifetimes; 0 repeats until failure, requiring finite cycles")
    parser.add_argument("--stall-seconds", type=float, default=10,
                        help="Invoke an optional passive diagnostic only after this HTTP wait")
    parser.add_argument("--on-stall-argv", type=Path,
                        help="JSON argv array, never a shell string; no steady-state tracing")
    parser.add_argument("--port", type=int, default=19080)
    parser.add_argument("--model-ramdisk-root", type=Path, default=Path("/mnt/llaminar-production-parity"))
    parser.add_argument("--persistent-model-cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cycle_limit < 0:
        parser.error("--cycle-limit must be zero or positive")
    if args.lifetime_limit < 0:
        parser.error("--lifetime-limit must be zero or positive")
    if args.lifetime_limit != 1 and args.cycle_limit == 0:
        parser.error("repeating fresh lifetimes requires a positive --cycle-limit")
    sequence = json.loads(args.sequence.read_text())
    read_sequence(sequence)
    selected = e2e.discover(args)
    if len(selected) != 1:
        raise ValueError("HTTP hammer must select exactly one canonical cell")
    campaign, exact, record = selected[0]
    profile = record["e2e"]
    pressure = None
    if args.prefix_pressure:
        # The model's generation definition owns GDN/MTP restore obligations;
        # E2E owns its context and long-prompt geometry. Never guess either
        # from a model name, backend or CLI flag substring.
        generation = record.get("runtime", {}).get("generation", {})
        pressure = PrefixPressure.from_document({
            "context_length": profile["context_length"],
            "minimum_prompt_tokens": profile["minimum_prompt_tokens"],
            "max_tokens": needle_max_tokens(profile["generation_tokens"]),
            "prefix_state": generation.get("prefix_state"),
            "mtp_policy": generation.get("mtp_policy")}).document()
    if args.container_image:
        identity = image_identity(args.container_image)
        cpu_isa = image_cpu_isa(identity)
        args.container_image = identity["id"]
        e2e.validate_attached_execution(args.container_image)
    else:
        cpu_isa = e2e.release_binary_cpu_isa(args.binary)
    if not 0 < args.stall_seconds < profile["request_timeout_seconds"]:
        parser.error("--stall-seconds must be within the request timeout")
    on_stall = json.loads(args.on_stall_argv.read_text()) if args.on_stall_argv else None
    if on_stall is not None and (not isinstance(on_stall, list) or not on_stall or
            any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in on_stall)):
        raise ValueError("--on-stall-argv requires a nonempty JSON argv array")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "kind": "http_lifecycle_hammer_diagnostic", "certified": False,
              "case": exact, "configuration": record, "cpu_isa": cpu_isa.value,
              "image": args.container_image, "state": "preparing", "return_code": None,
              "lifetime_limit": args.lifetime_limit, "cycles_per_lifetime": args.cycle_limit,
              "completed_lifetimes": 0, "active_lifetime": None, "lifetimes": []}
    started = time.monotonic()
    write_json(output / "report.json", report)
    try:
        # A single canonical lease owns all GGUF shards across the complete
        # diagnostic. Fresh server lifetimes never re-stage or duplicate weights.
        with e2e.parity.model_staging_workspace(args.model_ramdisk_root,
                args.persistent_model_cache_dir, None) as workspace:
            staged, _ = e2e.parity.stage_models_in_ramdisk([campaign], workspace.models,
                None, persistent=workspace.persistent)
            workspace.protect_published_models()
            staged_paths = {Path(item.source_path).resolve(): str(workspace.models / item.filename)
                            for item in staged}
            model = staged_paths[Path(record["model"]).resolve()]
            if any(ch in model + record["id"] for ch in "|\r\n"):
                raise ValueError("model/cell identity cannot contain harness delimiters")
            write_json(output / "server-args.json", profile["server_args"])
            write_json(output / "hammer-configuration.json", {
                "schema": 1, "kind": "http_lifecycle_hammer_diagnostic", "sequence": sequence,
                "cycle_limit": args.cycle_limit, "stall_seconds": args.stall_seconds,
                "request_timeout_seconds": profile["request_timeout_seconds"], "on_stall": on_stall,
                "prefix_pressure": pressure})
            command = ["bash", str(e2e.ROOT / "tests/v2/e2e/server/test_server_e2e.sh"),
                       "--binary", str(args.binary.resolve()), "--suite",
                       f"{model}|{e2e.harness_backend(campaign.group.backends)}|200||{record['id']}|e2e-certification",
                       "--server-args-file", str(output / "server-args.json"),
                       "--http-lifecycle-hammer", str(output / "hammer-configuration.json"),
                       "--port", str(args.port)]
            if args.container_image:
                command += ["--container-image", args.container_image]
            # Every launch goes through the mature harness, including its normal
            # retirement and one-shot driver checkpoint. Keep evidence separate:
            # a later healthy lifetime must never overwrite an earlier failure.
            for ordinal in count(1):
                lifetime_output = output / "lifetimes" / f"{ordinal:06d}"
                lifetime_output.mkdir(parents=True, exist_ok=False)
                lifetime = {"ordinal": ordinal, "output": str(lifetime_output),
                            "state": "running", "return_code": None}
                report["lifetimes"].append(lifetime)
                report.update(state="running", active_lifetime=ordinal)
                write_json(output / "report.json", report)
                print(f"[http-hammer] RUN {exact} lifetime={ordinal} output={lifetime_output}", flush=True)
                environment = e2e.certification_environment(profile, lifetime_output)
                environment["LLAMINAR_HTTP_HAMMER_BINARY"] = str(args.binary.resolve())
                lifetime_started = time.monotonic()
                try:
                    with (lifetime_output / "harness.log").open("w") as log:
                        code = run_owned_harness(command, environment, log)
                    lifetime["return_code"] = code
                    if code:
                        lifetime["state"] = "failed"
                        report.update(state="failed", return_code=code)
                        return code
                    # Process exit alone is not a complete lifetime proof. This
                    # read-only validator runs after the harness has retired the
                    # server and closed its checkpoint; it cannot finish it early.
                    e2e.validate_driver_diagnostics(lifetime_output)
                    lifetime["state"] = "passed"
                    report["completed_lifetimes"] += 1
                    print(f"[http-hammer] PASS {exact} lifetime={ordinal}", flush=True)
                    if ordinal == args.lifetime_limit:
                        report.update(state="bounded_diagnostic_complete", return_code=0)
                        return 0
                except BaseException as error:
                    lifetime["state"] = "interrupted_or_infrastructure_error"
                    lifetime["error"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    lifetime["elapsed_seconds"] = time.monotonic() - lifetime_started
                    report["active_lifetime"] = None
                    write_json(output / "report.json", report)
    except BaseException as error:
        report["state"] = "interrupted_or_infrastructure_error"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "report.json", report)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"[http-hammer] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
