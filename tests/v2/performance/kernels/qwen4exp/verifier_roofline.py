#!/usr/bin/env python3
"""Compare measured verifier leaves with same-device sustainable resource rates.

Each work count names its counter or algorithmic definition. Its reference
uses that identical definition. Kernel timing comes from a separate timing
window, never the counter collection run. A reference may be exceeded; that
is evidence to revisit the calibration or work estimate, not a pass flag.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median, quantiles


def measurement(value, name, allow_zero=False):
    """Service rates and durations must be positive. A source may issue zero work."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def identity(record, names):
    """Return explicit matching dimensions without inventing missing labels."""
    result = tuple(record[name] for name in names)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"Missing measurement identity {names}")
    return result


def middle_half(samples):
    """Describe the observed spread, including singleton calibrations."""
    if len(samples) == 1:
        return samples[0], samples[0]
    lower, _, upper = quantiles(samples, n=4, method="inclusive")
    return lower, upper


def analyze(document):
    """Return one row per measured resource and the maximum reference time per leaf."""
    references = {}
    dimensions = ("device", "isa", "clock_policy", "resource", "unit", "counter", "regime")
    for reference in document["references"]:
        key, = identity(reference, ("id",))
        if key in references:
            raise ValueError(f"Duplicate calibration {key}")
        rates = [measurement(sample["count"], "calibration work") / measurement(sample["us"], "calibration time")
                 for sample in reference["samples"]]
        if not rates:
            raise ValueError(f"Empty calibration {key}")
        references[key] = identity(reference, dimensions), rates
    rows, summaries = [], []
    seen = set()
    for leaf in document["leaves"]:
        case, role, symbol, variant = identity(leaf, ("case", "role", "symbol", "variant"))
        device, isa, clock = identity(leaf, dimensions[:3])
        key = (case, role, symbol, variant, device, isa, clock)
        if key in seen:
            raise ValueError(f"Duplicate leaf {key}")
        seen.add(key)
        observed = [measurement(value, "native elapsed time") for value in leaf["timing_us"]]
        if not observed:
            raise ValueError(f"Missing separate native timings for {key}")
        elapsed = median(observed)
        elapsed_low, elapsed_high = middle_half(observed)
        limiting = None
        resources = set()
        for work in leaf["work"]:
            name = work["reference"]
            ref_dims, rates = references[name]
            dims = identity(work, dimensions[3:])
            resource, unit, counter, regime = dims
            if (device, isa, clock, *dims) != ref_dims or resource in resources:
                raise ValueError(f"Mismatched or duplicate resource {resource} on {key}")
            resources.add(resource)
            amount = measurement(work["count"], "native work per dispatch", allow_zero=True)
            rate = median(rates)
            rate_low, rate_high = middle_half(rates)
            reference_us = amount / rate
            row = dict(case=case, role=role, symbol=symbol, variant=variant, device=device,
                       isa=isa, clock_policy=clock, resource=resource, unit=unit, counter=counter, regime=regime, work=amount,
                       calibration=name, units_per_us=rate, measured_us=elapsed, reference_us=reference_us,
                       calibration_samples=len(rates), rate_p25=rate_low, rate_p75=rate_high,
                       timing_samples=len(observed), measured_us_p25=elapsed_low, measured_us_p75=elapsed_high,
                       fraction_of_reference=reference_us / elapsed)
            rows.append(row)
            if limiting is None or reference_us > limiting["reference_us"]:
                limiting = row
        if limiting is None:
            raise ValueError(f"No comparable resources for {key}")
        summaries.append(dict(case=case, role=role, symbol=symbol, variant=variant, device=device,
                              isa=isa, clock_policy=clock, measured_us=elapsed, measured_us_p25=elapsed_low,
                              measured_us_p75=elapsed_high, limiting_measured_resource=limiting["resource"],
                              reference_us=limiting["reference_us"],
                              fraction_of_reference=limiting["fraction_of_reference"]))
    if not summaries:
        raise ValueError("No verifier leaves")
    return rows, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--csv", type=Path, help="Write the counter-level comparison to this file")
    args = parser.parse_args()
    rows, summaries = analyze(json.loads(args.evidence.read_text()))
    if args.csv:
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summaries, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
