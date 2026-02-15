#!/usr/bin/env python3

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import TypedDict


class StatsSummary(TypedDict):
    best: float
    median: float
    mean: float
    std: float
    all: list[float]


def run_cpu_only_parity_check(parity_binary: str, model: str, timeout_sec: int) -> tuple[int, str]:
    parity_env = os.environ.copy()
    parity_env["HIP_VISIBLE_DEVICES"] = ""
    parity_env["CUDA_VISIBLE_DEVICES"] = ""

    try:
        process = subprocess.run(
            [
                parity_binary,
                f"--model={model}",
                "--gtest_filter=*CPU_*",
            ],
            env=parity_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
        return process.returncode, process.stdout
    except subprocess.TimeoutExpired as exc:
        timed_output = exc.stdout or ""
        return 124, timed_output + f"\n[timeout] parity check exceeded {timeout_sec}s\n"


def detect_physical_cores() -> int:
    cmd = "lscpu -p=SOCKET,CORE | grep -v '^#' | sort -u | wc -l"
    out = subprocess.check_output(cmd, shell=True, text=True).strip()
    return int(out)


def build_env(pin_threads: bool, min_seq: int) -> dict[str, str]:
    env = os.environ.copy()
    env["LLAMINAR_FLASH_PREFILL_I16_I12_MIN_SEQ"] = str(min_seq)
    env["LLAMINAR_PROFILING"] = "0"

    if pin_threads:
        physical_cores = detect_physical_cores()
        env["OMP_NUM_THREADS"] = str(physical_cores)
        env["OMP_PLACES"] = "cores"
        env["OMP_PROC_BIND"] = "close"
        env["OMP_DYNAMIC"] = "FALSE"
        env["OPENBLAS_NUM_THREADS"] = str(physical_cores)
        env["MKL_NUM_THREADS"] = str(physical_cores)

    return env


def parse_metrics(output_text: str) -> tuple[str, str]:
    token_rate_pattern = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*tok/s")

    prefill_tps = ""
    overall_tps = ""
    in_prefill_block = False

    for line in output_text.splitlines():
        if "PREFILL" in line:
            in_prefill_block = True
        elif "DECODE" in line:
            in_prefill_block = False

        if in_prefill_block and "Throughput" in line and not prefill_tps:
            match = token_rate_pattern.search(line)
            if match:
                prefill_tps = match.group(1)

        if "Overall" in line and not overall_tps:
            match = token_rate_pattern.search(line)
            if match:
                overall_tps = match.group(1)

    return prefill_tps, overall_tps


def run_one(
    binary: str,
    model: str,
    prompt: str,
    decode_tokens: int,
    device: str,
    mode_flag: str,
    env: dict[str, str],
    timeout_sec: int,
) -> tuple[str, str, str, int]:
    run_env = env.copy()
    run_env["LLAMINAR_FLASH_PREFILL_I16_I12"] = mode_flag

    try:
        process = subprocess.run(
            [
                binary,
                "--benchmark",
                "-d",
                device,
                "-m",
                model,
                "-p",
                prompt,
                "-n",
                str(decode_tokens),
            ],
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
        prefill_tps, overall_tps = parse_metrics(process.stdout)
        return prefill_tps, overall_tps, process.stdout, process.returncode
    except subprocess.TimeoutExpired as exc:
        timed_output = exc.stdout or ""
        timeout_text = timed_output + f"\n[timeout] benchmark run exceeded {timeout_sec}s\n"
        return "", "", timeout_text, 124


def stats(values: list[float]) -> StatsSummary:
    sorted_values = sorted(values)
    return {
        "best": max(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "all": sorted_values,
    }


def safe_delta_percent(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("nan")
    return (numerator / denominator - 1.0) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run robust OFF/ON prefill A/B benchmark and summarize best/median/mean deltas."
    )
    parser.add_argument("--binary", default="/workspaces/llaminar/build_v2_release/llaminar2")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu:0")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--min-seq", type=int, default=1)
    parser.add_argument("--prompt", default="Llaminar cache aware sweep prompt.")
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=12,
        help="Repeat prompt text this many times separated by spaces.",
    )
    parser.add_argument(
        "--output-prefix",
        default="prefill_bestn",
        help="Prefix used for output files under --output-dir.",
    )
    parser.add_argument("--output-dir", default="/workspaces/llaminar/tmp")
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=30,
        help="Per-run timeout in seconds for parity precheck and benchmark subprocesses.",
    )
    parser.add_argument(
        "--no-pin-threads",
        action="store_true",
        help="Disable OMP/BLAS thread pinning env setup.",
    )
    parser.add_argument(
        "--parity-check",
        action="store_true",
        help="Run CPU-only single-device parity check before benchmarking.",
    )
    parser.add_argument(
        "--parity-binary",
        default="/workspaces/llaminar/build_v2_integration/tests/v2/v2_integration_parity_qwen2_single_device",
        help="Path to single-device parity test binary.",
    )

    args = parser.parse_args()

    if args.runs < 1:
        print("--runs must be >= 1", file=sys.stderr)
        return 2
    if args.timeout_sec < 1:
        print("--timeout-sec must be >= 1", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{args.output_prefix}_{timestamp}.csv"
    summary_path = output_dir / f"{args.output_prefix}_summary_{timestamp}.txt"
    log_path = output_dir / f"{args.output_prefix}_log_{timestamp}.txt"

    prompt_text = " ".join([args.prompt] * args.prompt_repeat)
    base_env = build_env(pin_threads=not args.no_pin_threads, min_seq=args.min_seq)

    rows: list[dict[str, str]] = []
    parity_status = "SKIPPED"
    parity_details = ""

    with log_path.open("w", encoding="utf-8") as log_file:
        if args.parity_check:
            if not Path(args.parity_binary).exists():
                print(
                    f"Parity binary not found: {args.parity_binary}",
                    file=sys.stderr,
                )
                return 4

            parity_cmd_line = (
                f"HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= {args.parity_binary} "
                f"--model={args.model} --gtest_filter=*CPU_*"
            )
            print("Running CPU-only parity check ...")
            log_file.write("Running CPU-only parity check ...\n")
            log_file.write(f"PARITY_CMD: {parity_cmd_line}\n")
            parity_code, parity_output = run_cpu_only_parity_check(
                parity_binary=args.parity_binary,
                model=args.model,
                timeout_sec=args.timeout_sec,
            )
            log_file.write(parity_output)
            log_file.write("\n")
            if parity_code != 0:
                parity_status = "FAILED"
                parity_details = f"CPU parity check failed with exit code {parity_code}"
                print(parity_details, file=sys.stderr)
                return 5

            parity_status = "PASSED"
            parity_details = "CPU-only parity filter (*CPU_*)"

        for mode_name, mode_flag in (("off", "0"), ("on", "1")):
            for run_index in range(1, args.runs + 1):
                line = f"Running {mode_name} #{run_index} ..."
                print(line)
                log_file.write(line + "\n")
                log_file.flush()

                prefill_tps, overall_tps, full_output, exit_code = run_one(
                    binary=args.binary,
                    model=args.model,
                    prompt=prompt_text,
                    decode_tokens=args.decode_tokens,
                    device=args.device,
                    mode_flag=mode_flag,
                    env=base_env,
                    timeout_sec=args.timeout_sec,
                )

                if not prefill_tps or not overall_tps:
                    log_file.write(f"Run failed (exit_code={exit_code})\n")
                    log_file.write(full_output)
                    log_file.write("\n")
                    if exit_code == 124:
                        print(
                            f"Timeout for mode={mode_name} run={run_index} after {args.timeout_sec}s. "
                            f"See log: {log_path}",
                            file=sys.stderr,
                        )
                        return 6
                    print(
                        f"Parse failure for mode={mode_name} run={run_index}. "
                        f"See log: {log_path}",
                        file=sys.stderr,
                    )
                    return 3

                rows.append(
                    {
                        "mode": mode_name,
                        "run": str(run_index),
                        "prefill_tps": prefill_tps,
                        "overall_tps": overall_tps,
                    }
                )

    with csv_path.open("w", newline="", encoding="utf-8") as output_csv:
        writer = csv.DictWriter(
            output_csv,
            fieldnames=["mode", "run", "prefill_tps", "overall_tps"],
        )
        writer.writeheader()
        writer.writerows(rows)

    values = defaultdict(lambda: {"prefill": [], "overall": []})
    for row in rows:
        values[row["mode"]]["prefill"].append(float(row["prefill_tps"]))
        values[row["mode"]]["overall"].append(float(row["overall_tps"]))

    off_prefill = stats(values["off"]["prefill"])
    on_prefill = stats(values["on"]["prefill"])
    off_overall = stats(values["off"]["overall"])
    on_overall = stats(values["on"]["overall"])

    summary_lines = [
        f"RAW={csv_path}",
        f"SUMMARY={summary_path}",
        f"LOG={log_path}",
        f"PARITY_CHECK={parity_status}",
        f"PARITY_DETAILS={parity_details}",
        "=== PREFILL (tok/s) ===",
        f"off: {off_prefill}",
        f"on : {on_prefill}",
        f"delta_best   = {safe_delta_percent(on_prefill['best'], off_prefill['best']):.2f}%",
        f"delta_median = {safe_delta_percent(on_prefill['median'], off_prefill['median']):.2f}%",
        f"delta_mean   = {safe_delta_percent(on_prefill['mean'], off_prefill['mean']):.2f}%",
        "",
        "=== OVERALL (tok/s) ===",
        f"off: {off_overall}",
        f"on : {on_overall}",
        f"delta_best   = {safe_delta_percent(on_overall['best'], off_overall['best']):.2f}%",
        f"delta_median = {safe_delta_percent(on_overall['median'], off_overall['median']):.2f}%",
        f"delta_mean   = {safe_delta_percent(on_overall['mean'], off_overall['mean']):.2f}%",
    ]

    with summary_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write("\n".join(summary_lines) + "\n")

    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
