#!/usr/bin/env python3
"""Run explicit, interleaved two-host GPU round-trip candidates through OpenMPI.

This diagnostic driver never installs dependencies or chooses a fallback. One
failed selected experiment stops the campaign. Result paths are local evidence,
not a production receipt. The same path must be mounted on both original hosts;
collect peer CSVs before removing those workspaces. Native MPI/session ownership
and the containing benchmark containers own final remote process retirement.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shlex
import statistics
import subprocess


VERBS_CANDIDATES = {"mapped-verbs", "write-combined-verbs", "mapped-dmabuf", "write-combined-dmabuf"}
CANDIDATES = {"mapped", "write-combined", "registered", "device"} | VERBS_CANDIDATES


def order_for_round(candidates: list[str], round_index: int) -> list[str]:
    """Rotate and reverse ordering to expose thermal/startup/order bias."""
    shift = round_index % len(candidates)
    values = candidates[shift:] + candidates[:shift]
    return list(reversed(values)) if round_index % 2 else values


def result_from_log(log: str, candidate: str, size: int, root: int) -> dict:
    """Authenticate the one terminal success row, never parse a partial timing."""
    rows = [json.loads(line) for line in log.splitlines() if line.startswith('{"candidate":')]
    if len(rows) != 1:
        raise ValueError("expected exactly one terminal timing row")
    row = rows[0]
    if (row["candidate"], row["bytes"], row["root"], row["correct"]) != (candidate, size, root, True):
        raise ValueError("terminal result does not match the requested experiment")
    for key in ("samples", "mean_us", "p50_us", "p95_us", "p99_us"):
        value = row[key]
        if not isinstance(value, (int, float)) or not 0 < value < float("inf"):
            raise ValueError(f"invalid measurement: {key}")
    return row


def summarize(rows: list[dict]) -> list[dict]:
    """Keep payload and direction separate; do not invent one universal winner."""
    summary = []
    cells = sorted({(r["bytes"], r["root"]) for r in rows})
    for size, root in cells:
        candidates = {}
        for candidate in sorted({r["candidate"] for r in rows}):
            group = [r for r in rows if (r["bytes"], r["root"], r["candidate"]) == (size, root, candidate)]
            if not group:
                raise ValueError("incomplete candidate matrix")
            values = [r["mean_us"] for r in group]
            candidates[candidate] = {"trials": len(values), "median_trial_mean_us": statistics.median(values),
                                     "min_trial_mean_us": min(values), "max_trial_mean_us": max(values)}
        if len({v["trials"] for v in candidates.values()}) != 1:
            raise ValueError("unequal trial counts")
        ranked = sorted(candidates, key=lambda c: candidates[c]["median_trial_mean_us"])
        winner = ranked[0]
        separated = len(ranked) > 1 and all(
            candidates[winner]["max_trial_mean_us"] < candidates[c]["min_trial_mean_us"] for c in ranked[1:])
        summary.append({"bytes": size, "root": root, "candidates": candidates,
                        "lowest_median": winner, "trial_ranges_separated": separated})
    return summary


def main() -> None:
    """Run the complete requested matrix with raw logs and immutable request metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpi", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--hosts", required=True, help="two original hosts with one slot each")
    parser.add_argument("--rsh-agent", required=True)
    parser.add_argument("--session-owner", required=True, help="canonical native llaminar_mpi_session helper")
    parser.add_argument("--daemon", required=True, help="orted from that same MPI installation")
    parser.add_argument("--net-devices", required=True, help="explicit UCX RDMA ports, never the management NIC")
    parser.add_argument("--library-path", required=True)
    parser.add_argument("--verbs-rails", nargs="+", default=[], help="original HCA:PORT:GID_INDEX for explicit verbs candidates")
    parser.add_argument("--tls", default="rc,self", help="include cuda for a CUDA-aware MPI experiment")
    parser.add_argument("--gpu-direct-rdma", choices=("yes", "try", "no"), default="try",
                        help="device candidate requires yes; protocol evidence must still prove zero-copy")
    parser.add_argument("--candidates", nargs="+", default=["mapped", "write-combined", "registered"])
    parser.add_argument("--bytes", type=int, nargs="+", default=[16384, 262144, 4194304])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--workers", default="all", help="positive resident worker cap or all (one per SM)")
    parser.add_argument("--rndv-threshold", default="0", help="explicit UCX policy; auto is a separate experiment")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (not args.candidates or len(set(args.candidates)) != len(args.candidates)
            or not set(args.candidates) <= CANDIDATES):
        parser.error("select unique explicit candidate names")
    if min(args.rounds, args.warmup, args.samples, *args.bytes) <= 0 or any(n % 4 for n in args.bytes):
        parser.error("positive rounds/samples and whole-FP32 payloads required")
    if "device" in args.candidates and args.gpu_direct_rdma != "yes":
        parser.error("device comparison requires --gpu-direct-rdma yes; no silently staged candidate")
    if set(args.candidates) & VERBS_CANDIDATES and not args.verbs_rails:
        parser.error("verbs candidates require explicit --verbs-rails")
    if len(set(args.verbs_rails)) != len(args.verbs_rails):
        parser.error("duplicate explicit verbs rails")
    if args.workers != "all" and (not args.workers.isdecimal() or int(args.workers) <= 0):
        parser.error("workers must be all or a positive decimal integer")
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, LD_LIBRARY_PATH=args.library_path, UCX_TLS=args.tls,
               UCX_NET_DEVICES=args.net_devices, UCX_PROTO_ENABLE="y", UCX_PROTO_INFO="n",
               UCX_LOG_LEVEL="warn", UCX_RNDV_THRESH=args.rndv_threshold,
               UCX_IB_GPU_DIRECT_RDMA=args.gpu_direct_rdma)
    identity = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    identity["binary_sha256"] = hashlib.sha256(Path(args.binary).read_bytes()).hexdigest()
    identity["ucx_env"] = {k: v for k, v in env.items() if k.startswith("UCX_")}
    (args.output / "request.json").write_text(json.dumps(identity, indent=2) + "\n")
    rows = []
    for trial in range(args.rounds):
        for size in args.bytes:
            for root in (0, 1):
                for candidate in order_for_round(args.candidates, trial):
                    name = f"{trial}-{size}-root{root}-{candidate}"
                    prefix = args.output / name
                    # Only the initiating rank opens its CSV. Both mounts must
                    # expose the experiment directory, created by the operator.
                    daemon = shlex.join([args.session_owner, "--mpi-daemon", "--session-base", "/tmp", "--", args.daemon])
                    command = [args.session_owner, "--session-base", "/tmp", "--", args.mpi,
                               "--launch-agent", daemon, "--host", args.hosts, "-np", "2", "--map-by", "ppr:1:node", "--bind-to", "core",
                               "--mca", "plm_rsh_agent", args.rsh_agent,
                               "--mca", "pml", "ucx", "--mca", "pml_ucx_tls", "any"]
                    for key in ("LD_LIBRARY_PATH", "UCX_TLS", "UCX_NET_DEVICES", "UCX_PROTO_ENABLE",
                                "UCX_PROTO_INFO", "UCX_LOG_LEVEL", "UCX_RNDV_THRESH", "UCX_IB_GPU_DIRECT_RDMA"):
                        command.extend(["-x", key])
                    command.extend(["--", args.binary, candidate, str(size), str(args.warmup),
                                    str(args.samples), str(root), args.workers, str(prefix)])
                    if candidate in VERBS_CANDIDATES:
                        command.extend(args.verbs_rails)
                    (prefix.with_suffix(".command.json")).write_text(json.dumps(command) + "\n")
                    with prefix.with_suffix(".log").open("w") as log:
                        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                        try:
                            code = process.wait(timeout=90)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                            raise RuntimeError("campaign stopped: timeout; verify remote container retirement before reuse")
                    if code:
                        raise RuntimeError(f"campaign stopped: {name} exited {code}; inspect exact log, no candidate fallback")
                    row = result_from_log(prefix.with_suffix(".log").read_text(), candidate, size, root)
                    row.update(trial=trial, name=name)
                    rows.append(row)
                    (args.output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
    (args.output / "summary.json").write_text(json.dumps(summarize(rows), indent=2) + "\n")
    print("ROUNDTRIP_CAMPAIGN_PASS", flush=True)


if __name__ == "__main__":
    main()
