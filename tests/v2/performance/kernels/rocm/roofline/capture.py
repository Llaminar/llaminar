"""Capture one scoped MI50 reference measurement and authenticate its hardware records."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "e2e/server"))
import gpu_driver_diagnostics as driver

PROBES = {
    "read": ("ReadBandwidth", "READ", 10),
    "issue": ("VectorIssue", "ISSUE", 9),
    "packet": ("QfnRoutedPacketRead", "PACKET", 12),
}

def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()

def physical(pci):
    device = Path("/sys/bus/pci/devices") / pci
    return {name: int((device / name).read_text())
            for name in ("gpu_busy_percent", "mem_info_vram_used")}

def require_isolation(pci):
    root = Path("/sys/bus/pci/devices") / pci
    selected = {p.name for p in (root / "drm").glob("renderD*")}
    exposed = {p.name for p in Path("/dev/dri").glob("renderD*")}
    if len(selected) != 1 or selected != exposed:
        raise RuntimeError(f"Expected the isolated PCI render node {selected}, got {exposed}")
    visible = set()
    for node in Path("/sys/class/kfd/kfd/topology/nodes").iterdir():
        try:
            properties = dict(line.split(maxsplit=1) for line in (node / "properties").read_text().splitlines())
        except PermissionError:
            continue
        minor = int(properties["drm_render_minor"])
        if minor >= 128:
            visible.add(f"renderD{minor}")
    if visible != selected:
        raise RuntimeError(f"KFD exposes {visible}, expected {selected}")

def scored_counter(output, kernel, metrics):
    payloads = list(output.rglob("*_results.json"))
    if len(payloads) != 1:
        raise ValueError("Expected one ROCm counter output")
    [data] = json.loads(payloads[0].read_text())["rocprofiler-sdk-tool"]
    [sample] = data["callback_records"]["counter_collection"]
    details = sample["dispatch_data"]["dispatch_info"]
    symbols = {row["kernel_id"]: row for row in data["kernel_symbols"]}
    named = symbols[details["kernel_id"]]
    if sample["replay_pass"] != 0 or not re.search(kernel, named["formatted_kernel_name"]):
        raise ValueError("Profiler record disagrees with the named dispatch")
    names = {row["id"]["handle"]: row["name"] for row in data["counters"]}
    values = {names[row["counter_id"]["handle"]]: row["value"] for row in sample["records"]}
    if len(values) != len(sample["records"]) or set(values) != set(metrics):
        raise ValueError("Profiler reported the wrong hardware counter set")
    return {"kernel": named["kernel_name"], "grid": details["grid_size"], "block": details["workgroup_size"],
            "values": values, "raw_file": str(payloads[0].relative_to(output)), "sha256": digest(payloads[0])}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", choices=PROBES)
    parser.add_argument("--pci", required=True, help="Domain:bus:device.function of the sole exposed MI50")
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routes", type=Path, help="Required for packet, three ordered 40-ID route histories")
    parser.add_argument("--issue-grid", type=int, help="Explicit calibrated workgroup count")
    parser.add_argument("--issue-block", type=int, choices=(256,1024), help="Explicit calibrated threads per workgroup")
    parser.add_argument("--kernel", help="Exact demangled kernel regex for a single profiled dispatch")
    parser.add_argument("--iteration", type=int, help="Warm kernel occurrence (one-based)")
    parser.add_argument("--metric", choices=("fetch", "write", "vector"))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", args.pci):
        parser.error("Supply an explicit PCI coordinate")
    profiling = bool(args.kernel or args.iteration or args.metric)
    if profiling and (not args.kernel or not args.metric or not args.iteration or args.iteration < 1):
        parser.error("Profiling requires kernel, positive iteration and metric")
    if (args.probe == "packet") != bool(args.routes):
        parser.error("Exactly the packet probe requires the original route IDs")

    name, tag, checks_per_round = PROBES[args.probe]
    exact_issue = args.issue_grid is not None or args.issue_block is not None
    if exact_issue:
        if args.probe != "issue" or args.issue_grid is None or args.issue_grid <= 0 or args.issue_block is None:
            parser.error("An explicit grid and block may only be used together with the issue probe")
        checks_per_round = 3
    binary = (args.build / name).resolve(strict=True)
    routes = args.routes.resolve(strict=True) if args.routes else None
    rounds = (0,) if profiling else (0, 1, 2)
    prefix = []
    metrics = ()
    output = args.output.resolve()
    if profiling:
        metrics = {"fetch": ("FetchSize",), "write": ("WriteSize",),
                   "vector": ("VALUInsts", "Wavefronts")}[args.metric]
        sdk = Path(os.environ["ROCM_PATH"]).resolve(strict=True)
        prefix = [str(sdk / "bin/rocprofv3"), "--kernel-trace", "--disable-signal-handlers", "true",
                  "--pmc", *(f"{metric}:device=0" for metric in metrics),
                  "--kernel-include-regex", args.kernel,
                  "--kernel-iteration-range", f"[{args.iteration}-{args.iteration}]",
                  "--output-format", "json", "--output-directory", str(output), "--"]
    probe_operands = ([str(routes)] if routes else [])
    if exact_issue:
        probe_operands = [str(args.issue_grid), str(args.issue_block)]
    argv = [[*prefix, str(binary), str(round_index), args.pci, *probe_operands]
            for round_index in rounds]
    started = datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True)
    identity = {"binary": str(binary), "binary_sha256": digest(binary), "pci": args.pci,
                "routes_sha256": digest(routes) if routes else None, "argv": argv, "started": started}
    (output / "invocation.json").write_text(json.dumps(identity, indent=2) + "\n")
    require_isolation(args.pci)
    before = physical(args.pci)
    if before["gpu_busy_percent"]:
        raise RuntimeError(f"Assigned MI50 is busy: {before}")

    state = output / "driver-state.json"
    log = output / "probe.log"
    driver.begin(state)
    returncode = 0
    try:
        with log.open("x") as stream:
            for command in argv:
                returncode = subprocess.run(command, cwd=output, stdout=stream, stderr=subprocess.STDOUT).returncode
                if returncode:
                    break
    finally:
        clean = driver.finish(state, output / "driver-report.json")
    for _ in range(20):
        after = physical(args.pci)
        if after == before:
            break
        time.sleep(.25)
    if not clean or after != before:
        raise RuntimeError("Driver health or physical retirement failed")
    if returncode:
        raise RuntimeError(f"Native probe exited {returncode} after physical retirement. See {log}")
    lines = log.read_text().splitlines()
    checks = [s for s in lines if s.startswith("CHECK,")]
    samples = [s for s in lines if s.startswith(tag + ",")]
    devices = [s.split(",") for s in lines if s.startswith("DEVICE,")]
    expected = checks_per_round * len(rounds)
    if len(set(checks)) != len(checks) or len(checks) != expected or len(samples) != expected * 13:
        raise RuntimeError("Incomplete independent exact-check or sample records")
    isa_index = 3 if args.probe == "read" else 2
    if len(devices) != len(rounds) or any(
            len(row) <= isa_index + 1 or row[1] != args.pci or
            row[isa_index].split(":")[0] != "gfx906" or row[isa_index + 1] != "60"
            for row in devices):
        raise RuntimeError("Probe selected the wrong PCI or a device outside the 60-CU gfx906 scope")
    if digest(binary) != identity["binary_sha256"] or (routes and digest(routes) != identity["routes_sha256"]):
        raise RuntimeError("Native input changed during capture")
    result = {"passed": True, "driver_clean": clean, "physical_before": before, "physical_after": after,
              "quiet_timing": not profiling, "checks": expected, "device_records": devices}
    if profiling:
        result["counter"] = scored_counter(output, args.kernel, metrics)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

if __name__ == "__main__":
    main()
