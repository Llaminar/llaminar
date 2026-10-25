#!/usr/bin/env python3
"""Matched Release-only timing of the projection-eliding shared CPU readout.

Time two full star verifications and final accepted-state publication, including
the deferred key projections, query preparation and final dense publication. All candidates use four-row physical tiles,
operands and precision. Setup, native oracle checks, warmup and artifact hashing
are outside the native steady-clock interval. Results are synthetic component
measurements, not model throughput or comparison with TreeWY/ReplaySSM kernels.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

from exact import check_guard, digest, guarded, inputs
from shared import Native
from harness import Arithmetic, Comparison, LCG, Precision


def measure_case(native, rows, dk, dv, seed, samples, iterations):
    """Retain identical buffers and rotate all six three-candidate orders."""
    rng, arithmetic = LCG(seed), Arithmetic(Precision.FP32)
    state = rng.vector(arithmetic, dk * dv, 128).reshape(dk, dv)
    q, k, v, gates = inputs(rng, (-1,) * (2 * rows), dk, dv, "ordinary")
    prepared = [native.prepare(q[i], gates[i]) for i in range(2 * rows)]
    scaled = np.stack([item[0] for item in prepared])
    decay = np.array([item[1][0] for item in prepared], dtype=np.float32)
    beta = np.array([item[1][1] for item in prepared], dtype=np.float32)
    output, output_guard = guarded((2 * rows, dv))
    corrections, correction_guard = guarded((2 * rows, dv))
    snapshots, snapshot_guard = guarded((2 * rows, dk, dv))
    resolved, resolved_guard = guarded((dk, dv))
    terminal, terminal_guard = guarded((dk, dv))
    adjusted, adjusted_guard = guarded(scaled.shape)
    eta, eta_guard = guarded(decay.shape)
    arrays = (state, scaled, k, v, decay, beta, output, corrections, snapshots, resolved, terminal, adjusted, eta)
    pointers = [native.ptr(array) for array in arrays]
    expected_output, legacy_output = np.empty_like(output), np.empty_like(output)
    parent, expected_terminal = state, None
    for round_index in range(2):
        accepted = None
        for row in range(rows):
            index = round_index * rows + row
            updated, expected_output[index], legacy_output[index] = native.serial(parent, q[index], k[index], v[index], gates[index])
            if row == (0 if round_index == 0 else rows - 1):
                accepted = updated
        parent = accepted
        expected_terminal = accepted
    comparison = Comparison()
    elapsed = ctypes.c_double()

    def invoke(mode, repeats):
        """Measure only native repeated lifecycles, excluding ctypes and checks."""
        native.call("gdn_shared_lifecycle", mode, repeats, rows, dk, dv, *pointers, ctypes.byref(elapsed))
        return elapsed.value

    def validate(mode):
        """Authenticate output/state bytes and the absence of compact snapshot writes."""
        comparison.observe(legacy_output if mode == 2 else expected_output, output, {"mode": mode, "surface": "outputs"})
        comparison.observe(expected_terminal, terminal, {"mode": mode, "surface": "terminal"})
        for guard in (output_guard, correction_guard, snapshot_guard, resolved_guard, terminal_guard, adjusted_guard, eta_guard):
            check_guard(guard)
        if mode in (0, 2) and np.any(snapshots.view(np.uint32) != 0x7FC12345):
            raise AssertionError("compact lifecycle wrote a dense speculative leaf state")
        if comparison.mismatches:
            raise AssertionError("timing candidate is not serial-byte-equivalent")

    for mode in (0, 1, 2):
        snapshots.view(np.uint32).fill(0x7FC12345)
        invoke(mode, 1)
        validate(mode)
        start = time.perf_counter()
        while time.perf_counter() - start < 0.1:
            invoke(mode, iterations)
    timings = {0: [], 1: [], 2: []}
    for sample in range(samples):
        for mode in ((0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0), (1, 0, 2), (0, 2, 1))[sample % 6]:
            # Poisoning is outside the native interval and occurs equally for
            # all modes. The explicit cache-hot warmup follows this reset.
            snapshots.view(np.uint32).fill(0x7FC12345)
            invoke(mode, 1)
            timings[mode].append(invoke(mode, iterations))
            validate(mode)
    summary = {}
    for mode, name in ((0, "shared_leaves"), (1, "shared_snapshots"), (2, "legacy_compact")):
        summary[name] = {"median_us": statistics.median(timings[mode]) / 1000,
                         "p10_us": float(np.quantile(timings[mode], 0.1)) / 1000,
                         "p90_us": float(np.quantile(timings[mode], 0.9)) / 1000,
                         "samples_ns": timings[mode]}
    return {"rows_per_round": rows, "dk": dk, "dv": dv, "rounds": 2,
            "iterations_per_sample": iterations, "samples_per_mode": samples,
            "native_comparison": vars(comparison), "timing": summary,
            "legacy_over_shared": summary["legacy_compact"]["median_us"] / summary["shared_leaves"]["median_us"],
            "snapshots_over_shared": summary["shared_snapshots"]["median_us"] / summary["shared_leaves"]["median_us"],
            "shared_projection_actions": 2*rows + 2, "legacy_projection_actions": 4*rows,
            "dense_state_write_bytes": (2 * rows + 1) * dk * dv * 4,
            "compact_state_write_bytes": 2 * dk * dv * 4,
            "legacy_correction_write_bytes": 2 * rows * dv * 4,
            "shared_correction_write_bytes": 0,
            "shared_query_workspace_write_bytes": 2 * rows * (dk + 1) * 4}


def main():
    """Require Release code and an explicit output artifact before measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 19, 39])
    parser.add_argument("--dk", type=int, default=128)
    parser.add_argument("--dv", type=int, default=128)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    if min(args.rows) < 1 or max(args.rows) > 4096 or not 1 <= args.dk <= 512 or not 16 <= args.dv <= 512 or args.dv % 16:
        parser.error("native-control geometry requires rows=1..4096, dk=1..512 and dv=16..512 in multiples of 16")
    if args.samples < 3 or args.iterations < 1:
        parser.error("at least three samples and one iteration are required")
    library = args.library.resolve(strict=True)
    manifest = json.loads(library.with_name("native-build.json").read_text())
    core = Path(manifest["core_library"])
    if manifest["core_build_type"] != "Release":
        parser.error("timing requires a core from a Release build")
    cache = (library.parent / "CMakeCache.txt").read_text()
    if "CMAKE_BUILD_TYPE:STRING=Release\n" not in cache:
        parser.error(f"timing requires a Release bridge: {library.parent}")
    native = Native(library)
    function = native.library.gdn_shared_lifecycle
    function.argtypes = [ctypes.c_int] * 5 + [native.pointer] * 13 + [ctypes.POINTER(ctypes.c_double)]
    function.restype = ctypes.c_int
    records = [measure_case(native, rows, args.dk, args.dv, args.seed, args.samples, args.iterations) for rows in args.rows]
    report = {"schema": "gdn-shared-readout-lifecycle-timing-v1", "seed": args.seed,
              "scope": "synthetic cache-hot, single-thread two-round core lifecycle; no production or model speedup claim",
              "measured_kernel_isa": "AVX2",
              "core_oracle_runtime_isa": os.environ.get("LLAMINAR_ISA_LEVEL", "auto"),
              "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
              "affinity": sorted(os.sched_getaffinity(0)), "build": manifest,
              "core_sha256": digest(core), "bridge_sha256": digest(library),
              "driver_sha256": digest(__file__), "cases": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for record in records:
        print(f"M={record['rows_per_round']}: new={record['timing']['shared_leaves']['median_us']:.3f} us, "
              f"legacy={record['timing']['legacy_compact']['median_us']:.3f} us, "
              f"speedup={record['legacy_over_shared']:.3f}x; outputs match their own serial definition")


if __name__ == "__main__":
    main()
