#!/usr/bin/env python3
"""Benchmark only canonical E2E candidates on an immutable Release image.

The typed E2E argument vector owns model execution policy. This runner owns
only a versioned workload, timing validation and the high-water comparison.
It never commits results, lowers a baseline, or mints a production certificate;
the full pipeline owns those operations after all correctness gates pass.
Full same-image E2E evidence is required unless --diagnostic explicitly selects
a one-off experiment that cannot certify an image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import uuid

import docker_paths
import run_model_parity_e2e as e2e
from production_artifacts import (digest, image_identity, positive, ratchet,
                                  validate_image_e2e, validate_manifest, write_json)

ROOT = Path(__file__).resolve().parents[2]


def hardware_identity() -> dict:
    """Identify compute/topology without keying ratchets on changing clocks."""
    cpu = json.loads(subprocess.check_output(["lscpu", "--json"], text=True))["lscpu"]
    fields = ("Architecture:", "CPU(s):", "Model name:", "Socket(s):",
              "Core(s) per socket:", "Thread(s) per core:", "NUMA node(s):")
    selected = {item["field"]: item["data"] for item in cpu
                if item["field"] in fields or re.fullmatch(r"NUMA node\d+ CPU\(s\):", item["field"])}
    pci = subprocess.check_output(["lspci", "-Dnn"], text=True)
    devices = sorted(line for line in pci.splitlines()
                     if any(kind in line for kind in ("VGA compatible controller", "3D controller", "Display controller")))
    return {"cpu": selected, "accelerators": devices}


def validate_workload(workload: dict) -> str:
    """Keep workload selection explicit and forbid empty/shortened evidence."""
    if workload.get("schema") != 1 or not workload.get("prompt_unit"):
        raise ValueError("invalid production benchmark workload")
    for field in ("prompt_repetitions", "decode_tokens", "warmup_iterations", "measurement_iterations"):
        if type(workload.get(field)) is not int or workload[field] <= 0:
            raise ValueError(f"invalid benchmark workload {field}")
    if workload["measurement_iterations"] < 3:
        raise ValueError("certification needs at least three measured requests")
    return workload["prompt_unit"] * workload["prompt_repetitions"]


def benchmark_arguments(record: dict, model: str, output: str, workload: dict) -> list[str]:
    """Use the server's exported policy verbatim; add only benchmark workload."""
    prompt = validate_workload(workload)
    if not record.get("e2e"):
        raise ValueError("only E2E-tagged cells may be benchmarked")
    return ["benchmark", *record["e2e"]["server_args"], "-m", model,
            "--context-length", str(record["e2e"]["context_length"]),
            "--prompt", prompt, "-n", str(workload["decode_tokens"]),
            "--seed", str(workload["seed"]), "--temperature", str(workload["temperature"]),
            "--top-k", str(workload["top_k"]), "--top-p", str(workload["top_p"]),
            "--benchmark-json-output", output]


def validate_measurement(data: dict, workload: dict) -> dict:
    """Use medians of complete unprofiled requests, never rounded console text."""
    prompt = validate_workload(workload).encode()
    if (data.get("schema") != "llaminar.benchmark.v1" or data.get("success") is not True
            or data.get("prefill_success") is not True or data.get("decode_success") is not True):
        raise ValueError("benchmark did not complete production prefill and decode")
    if (data.get("prompt", {}).get("sha256") != hashlib.sha256(prompt).hexdigest()
            or data["prompt"].get("bytes") != len(prompt)):
        raise ValueError("benchmark prompt identity mismatch")
    count = workload["measurement_iterations"]
    rows = data.get("iterations", [])
    if (data.get("measurement_iterations") != count or len(rows) != count
            or data.get("warmup_iterations") != workload["warmup_iterations"]):
        raise ValueError("benchmark omitted warmup or measured requests")
    samples = {"prefill": [], "decode": []}
    token_counts = set()
    for row in rows:
        if row["tokens"]["decode"] != workload["decode_tokens"]:
            raise ValueError("benchmark decode was truncated (including early EOS)")
        token_counts.add(positive(row["tokens"]["prefill"]))
        for phase, metric in (("prefill", "prefill"), ("decode", "decode_after_prefill")):
            samples[phase].append(positive(row["throughput_tokens_per_sec"][metric]))
    if len(token_counts) != 1:
        raise ValueError("benchmark prompt tokenization changed between requests")
    return {"tokens_per_second": {key: statistics.median(values) for key, values in samples.items()},
            "samples": samples, "prefill_tokens": int(token_counts.pop())}


def run_cell(image: str, backends: str, directory: Path, model: Path,
             record: dict, workload: dict) -> dict:
    """Run one isolated container and always retire it on timeout/interruption."""
    name = "llaminar-benchmark-" + uuid.uuid4().hex
    output = directory / "benchmark.json"
    command = ["docker", "run", "--rm", "--name", name,
               *docker_paths.device_args(image, backends),
               *docker_paths.mounts([(model.parent, str(model.parent), True),
                                     (directory, "/benchmark-output", False)]),
               "-e", f"LLAMINAR_BENCHMARK_ITERATIONS={workload['measurement_iterations']}",
               "-e", f"LLAMINAR_BENCHMARK_WARMUP_ITERATIONS={workload['warmup_iterations']}",
               image, *benchmark_arguments(record, str(model), "/benchmark-output/benchmark.json", workload)]
    write_json(directory / "invocation.json", {"argv": command})
    try:
        with (directory / "benchmark.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           timeout=e2e.parity.EXACT_CELL_TIMEOUT_SECONDS)
    finally:
        # A dead docker client does not necessarily retire server/MPI children.
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=30, check=False)
    return json.loads(output.read_text())


def main(argv: list[str] | None = None) -> int:
    """Emit independent cell progress and preserve partial evidence on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--image", required=True)
    admission = parser.add_mutually_exclusive_group()
    admission.add_argument("--e2e-report", type=Path,
                           help="passing full E2E report for this exact image and manifest")
    admission.add_argument("--diagnostic", action="store_true",
                           help="explicit one-off experiment; never eligible for certification")
    parser.add_argument("--cell", default=".*", help="diagnostic selector; partial runs cannot certify an image")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "benchmarks/production/high_water.json")
    parser.add_argument("--workload", type=Path, default=ROOT / "benchmarks/production/workload.json")
    parser.add_argument("--model-ramdisk-root", type=Path, default=Path("/mnt/llaminar-production-parity"))
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    all_cells = validate_manifest(manifest, args.source_revision)
    cells = [row for row in all_cells if re.fullmatch(args.cell, row["case"])]
    if not cells:
        raise ValueError("no eligible benchmark cells selected")
    for row in cells:
        print(f"[production-benchmark] selected {row['case']}", flush=True)
    if args.list:
        return 0
    if not args.diagnostic and args.e2e_report is None:
        parser.error("benchmarks require --e2e-report from the full server suite; "
                     "use --diagnostic only for an explicit one-off experiment")
    runtime = image_identity(args.image)
    if runtime["labels"].get("org.opencontainers.image.revision") != args.source_revision:
        raise ValueError("benchmark image has the wrong source revision")
    e2e_digest = None
    if args.e2e_report is not None:
        evidence = json.loads(args.e2e_report.read_text())
        validate_image_e2e(evidence, manifest, runtime["id"])
        e2e_digest = digest(evidence)
    workload = json.loads(args.workload.read_text())
    validate_workload(workload)
    hardware = hardware_identity()
    baseline = json.loads(args.baseline.read_text())
    report = {"schema": 1, "image": runtime["id"], "source_revision": args.source_revision,
              "manifest_digest": digest(manifest), "hardware": hardware,
              "diagnostic": args.diagnostic, "e2e_report_digest": e2e_digest,
              "workload": workload, "complete": False, "passed": False, "cells": []}
    args.report = args.report.resolve()
    run_root = args.report.parent / ("benchmarks-" + str(time.time_ns()))
    run_root.mkdir(parents=True)
    staging = [e2e.parity.CampaignCell(row["campaign"], e2e.parity.CampaignGroup(row["backends"], "ALL"),
                                     model_files=tuple(row["model_files"])) for row in cells]
    started = time.monotonic()
    try:
        with e2e.parity.model_staging_workspace(args.model_ramdisk_root, Path("cache"), None) as workspace:
            staged, _ = e2e.parity.stage_models_in_ramdisk(staging, workspace.models, None,
                                                         persistent=workspace.persistent)
            workspace.protect_published_models()
            paths = {Path(item.source_path).resolve(): workspace.models / item.filename for item in staged}
            for index, row in enumerate(cells, 1):
                print(f"[production-benchmark] {index}/{len(cells)} RUN {row['case']}", flush=True)
                directory = run_root / str(index)
                directory.mkdir()
                config = row["configuration"]
                model = paths[Path(config["model"]).resolve()]
                data = run_cell(runtime["id"], row["backends"], directory, model, config, workload)
                measured = validate_measurement(data, workload)
                # Preserve the model filename/size, not machine-specific mount text.
                identity = {"case": row["case"], "configuration": {**config, "model": model.name},
                            "model_bytes": model.stat().st_size, "hardware": hardware,
                            "model_files": {Path(name).name: Path(name).stat().st_size for name in row["model_files"]},
                            "cpu_isa": runtime["labels"]["org.llaminar.cpu_isa"],
                            "workload": workload, "prefill_tokens": measured["prefill_tokens"]}
                report["cells"].append({"case": row["case"], "identity": identity,
                                        "artifacts": str(directory), **measured})
                print(f"[production-benchmark] {index}/{len(cells)} PASS {row['case']} "
                      f"prefill={measured['tokens_per_second']['prefill']:.2f} "
                      f"decode={measured['tokens_per_second']['decode']:.2f} tok/s", flush=True)
                write_json(args.report, report)
        proposed, comparisons = ratchet(baseline, report["cells"])
        for comparison in comparisons:
            prior = comparison["high_water"]
            delta = "new" if prior is None else f"{100 * (comparison['current'] / prior - 1):+.2f}%"
            print(f"[production-benchmark] {comparison['status']} {comparison['case']} "
                  f"{comparison['phase']}={comparison['current']:.2f} tok/s delta={delta}", flush=True)
        report.update(complete=not args.diagnostic and len(cells) == len(all_cells), comparisons=comparisons,
                      passed=all(item["passed"] for item in comparisons),
                      proposed_high_water=proposed, baseline_digest=digest(baseline))
        return 0 if report["passed"] else 1
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(args.report, report)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f"[production-benchmark] ERROR: {error}", file=sys.stderr)
        sys.exit(1)
