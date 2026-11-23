#!/usr/bin/env python3
"""@file score_hc_ffn.py
@brief Apply each card's measured read and issue rates to ordered HC/FFN phases.

Each phase uses measurements from its exact launch geometry. A common grid
models a resident program. A per-phase grid map models a staged launch.
"""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


PHASES = ("hc_down", "hc_up", "router", "gate_up", "down")
REQUIRED = {"dram", "l2", "dp4a", "ffma"}
OPTIONAL = {"prmt", "int-mix"}
RATE_COLUMNS = {"dram": "dram_GBps", "l2": "l2_GBps", "dp4a": "dp4a_Ginst_s",
                "ffma": "ffma_Ginst_s", "prmt": "prmt_Ginst_s", "int-mix": "int_mix_Ginst_s"}


def calibration(path):
    """Read one checked capture and retain its hardware and launch identity."""
    with path.open(newline="") as stream:
        identifier = next(csv.reader(stream))
        rows = list(csv.DictReader(stream))
    kind, pci, _, isa, sms, _ = identifier
    if kind != "DEVICE" or not pci or int(sms) <= 0 or len(rows) != 22:
        raise ValueError(f"{path} is missing a complete native capture")
    first, *_, check = rows
    mode = first["probe"]
    blocks, threads = int(first["blocks"]), int(first["threads"])
    if mode not in REQUIRED | OPTIONAL or blocks <= 0 or threads <= 0:
        raise ValueError(f"{path} declares an invalid calibration dispatch")
    if (check["probe"], check["sample"], check["blocks"]) != ("CHECK", "pass", str(blocks * threads)):
        raise ValueError(f"{path} lacks a checked checksum for its native grid")
    rates = []
    for index, row in enumerate(rows[:-1]):
        if (row["probe"], int(row["blocks"]), int(row["threads"]), int(row["sample"])) != (
            mode, blocks, threads, index
        ):
            raise ValueError(f"{path} changes dispatch or sample identity")
        work, elapsed = int(row["count"]), float(row["us_per_launch"])
        if work <= 0 or elapsed <= 0 or not math.isfinite(elapsed) or row["count"] != first["count"]:
            raise ValueError(f"{path} has invalid calibration work or duration")
        rates.append(work / elapsed)
    return (pci.lower(), isa, int(sms), blocks, threads), mode, rates


def scored_work(work, phase_rates):
    """Use the dominant resource in each ordered phase, then add phase floors."""
    result = {}
    for phase in PHASES:
        rates = phase_rates[phase]
        count, int_macs, fp32_macs = (
            int(work[f"{phase}_{name}"]) for name in ("bytes", "int_macs", "fp32_macs")
        )
        if count <= 0 or min(int_macs, fp32_macs) < 0 or int_macs % 4:
            raise ValueError(f"Invalid work in {work['family']} phase {phase}")
        issue = max(int_macs / 4 / rates["dp4a"], fp32_macs / rates["ffma"])
        result[f"{phase}_issue_us"] = issue
        for tier in ("l2", "dram"):
            result[f"{phase}_{tier}_us"] = max(count / rates[tier], issue)
    for tier in ("l2", "dram"):
        result[f"total_{tier}_us"] = sum(result[f"{phase}_{tier}_us"] for phase in PHASES)
    return result


def phase_grid(argument):
    """Parse one named launch geometry without inferring missing dimensions."""
    phase, _, geometry = argument.partition("=")
    blocks, _, threads = geometry.partition("x")
    if (phase not in PHASES or not blocks.isascii() or not blocks.isdecimal() or
            not threads.isascii() or not threads.isdecimal() or int(blocks) <= 0 or int(threads) <= 0):
        raise argparse.ArgumentTypeError("expected PHASE=BLOCKSxTHREADS with positive dimensions")
    return phase, (int(blocks), int(threads))


def rate_columns(rates, prefix=""):
    """Report calibrated operations and bytes per second in billions."""
    return {prefix + RATE_COLUMNS[mode]: f"{rate / 1000:.6f}" for mode, rate in rates.items()}


def references(observations, phase_grids):
    """Build one checked reference per dispatch or per staged device."""
    if not phase_grids:
        for (pci, isa, sms, blocks, threads), rates in sorted(observations.items()):
            geometry = dict(pci=pci, isa=isa, sms=sms, blocks=blocks, threads=threads)
            yield geometry | rate_columns(rates), dict.fromkeys(PHASES, rates)
        return

    for pci, isa, sms in sorted({identity[:3] for identity in observations}):
        geometry = dict(pci=pci, isa=isa, sms=sms)
        phase_rates = {}
        for phase in PHASES:
            blocks, threads = phase_grids[phase]
            dispatch = (pci, isa, sms, blocks, threads)
            if dispatch not in observations:
                raise ValueError(f"{pci} lacks {phase} calibration at {blocks}×{threads}")
            rates = observations[dispatch]
            phase_rates[phase] = rates
            geometry.update({f"{phase}_blocks": blocks, f"{phase}_threads": threads})
            geometry.update(rate_columns(rates, f"{phase}_"))
        yield geometry, phase_rates


def main():
    """Print each source geometry's reference with its measured launch settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path, help="CSV from the probe's hc-ffn-work mode")
    parser.add_argument("calibrations", type=Path, nargs="+", help="Checked native probe CSVs")
    parser.add_argument("--phase-grid", type=phase_grid, action="append", default=[],
                        help="Use PHASE=BLOCKSxTHREADS once for each ordered phase (staged launches)")
    args = parser.parse_args()
    phase_grids = dict(args.phase_grid)
    if args.phase_grid and (set(phase_grids) != set(PHASES) or len(args.phase_grid) != len(PHASES)):
        parser.error(f"--phase-grid requires one entry for each of: {', '.join(PHASES)}")
    observations = defaultdict(lambda: defaultdict(list))
    for path in args.calibrations:
        identity, mode, rates = calibration(path)
        observations[identity][mode].extend(rates)
    for identity, samples in observations.items():
        pci, _, _, blocks, threads = identity
        if not REQUIRED <= samples.keys():
            raise ValueError(f"{pci} {blocks}×{threads} lacks {sorted(REQUIRED - samples.keys())}")
    rates_by_dispatch = {identity: {mode: statistics.median(values) for mode, values in samples.items()}
                         for identity, samples in observations.items()}
    calibrated = list(references(rates_by_dispatch, phase_grids))
    with args.work.open(newline="") as stream:
        work = csv.DictReader(stream)
        phase_times = [f"{phase}_{kind}_us" for phase in PHASES for kind in ("issue", "l2", "dram")]
        if phase_grids:
            identity_columns = ["pci", "isa", "sms"]
            rate_fields = [f"{phase}_{field}" for phase in PHASES
                           for field in ("blocks", "threads", *RATE_COLUMNS.values())]
        else:
            identity_columns = ["pci", "isa", "sms", "blocks", "threads"]
            rate_fields = RATE_COLUMNS.values()
        writer = csv.DictWriter(sys.stdout, [*identity_columns, *work.fieldnames, *rate_fields,
                                             *phase_times, "total_l2_us", "total_dram_us"])
        writer.writeheader()
        for entry in work:
            for geometry, phase_rates in calibrated:
                times = {}
                if entry["source_geometry"] == "complete":
                    times = {name: f"{time:.6f}" for name, time in scored_work(entry, phase_rates).items()}
                writer.writerow(geometry | entry | times)


if __name__ == "__main__":
    main()
