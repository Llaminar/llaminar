#!/usr/bin/env python3
"""Apply each card's measured read and issue rates to the ordered HC/FFN phases."""

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


def scored_work(work, rates):
    """Use the dominant resource in each ordered phase, then add phase floors."""
    result = {}
    for phase in PHASES:
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


def main():
    """Print one reference row per source geometry, PCI device, and CTA count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path, help="CSV from the probe's hc-ffn-work mode")
    parser.add_argument("calibrations", type=Path, nargs="+", help="Checked native probe CSVs")
    args = parser.parse_args()
    observations = defaultdict(lambda: defaultdict(list))
    for path in args.calibrations:
        identity, mode, rates = calibration(path)
        observations[identity][mode].extend(rates)
    with args.work.open(newline="") as stream:
        work = csv.DictReader(stream)
        reference = [f"{phase}_{kind}_us" for phase in PHASES for kind in ("issue", "l2", "dram")]
        writer = csv.DictWriter(sys.stdout, ["pci", "isa", "sms", "blocks", "threads", *work.fieldnames,
            "dram_GBps", "l2_GBps", "dp4a_Ginst_s", "ffma_Ginst_s", "prmt_Ginst_s", "int_mix_Ginst_s",
            *reference, "total_l2_us", "total_dram_us"])
        writer.writeheader()
        for entry in work:
            for (pci, isa, sms, blocks, threads), samples in sorted(observations.items()):
                if not REQUIRED <= samples.keys():
                    raise ValueError(f"{pci} {blocks}×{threads} lacks {sorted(REQUIRED - samples.keys())}")
                rates = {key: statistics.median(values) for key, values in samples.items()}
                mode_fields = {"dram": "dram_GBps", "l2": "l2_GBps", "dp4a": "dp4a_Ginst_s",
                    "ffma": "ffma_Ginst_s", "prmt": "prmt_Ginst_s", "int-mix": "int_mix_Ginst_s"}
                metrics = {mode_fields[key]: f"{rate / 1000:.6f}" for key, rate in rates.items()}
                if entry["source_geometry"] == "complete":
                    metrics.update({name: f"{time:.6f}" for name, time in scored_work(entry, rates).items()})
                writer.writerow(dict(pci=pci, isa=isa, sms=sms, blocks=blocks, threads=threads, **entry, **metrics))


if __name__ == "__main__":
    main()
