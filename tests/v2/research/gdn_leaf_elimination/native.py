#!/usr/bin/env python3
"""Compare actual production CPU decode with fused and FP64 leaf readouts.

The native bridge rejects comparisons unless an independent control matches
both native output and state bytes. Linked-core identity is recorded by the
build manifest; this is an existing-binary diagnostic, not a fresh build gate.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from harness import Arithmetic, Comparison, LCG, Precision


def run_native(library, seed):
    """Run a minimal witness and ordinary-gate trials against a real CPU binary."""
    library = Path(library).resolve(strict=True)
    manifest_path = library.with_name("native-build.json")
    manifest = json.loads(manifest_path.read_text())
    core = Path(manifest["core_library"]).resolve(strict=True)
    core_hash = hashlib.file_digest(core.open("rb"), "sha256").hexdigest()
    native = ctypes.CDLL(str(library))
    compare = native.gdn_leaf_cpu_compare
    pointer = ctypes.POINTER(ctypes.c_float)
    compare.argtypes = [ctypes.c_int, ctypes.c_int] + [pointer] * 8
    compare.restype = ctypes.c_int
    arithmetic = Arithmetic(Precision.FP32)
    rng = LCG(seed)
    results = []
    for dk, dv in ((1, 16), (7, 16), (64, 64), (128, 128)):
        fused_metric, wide_metric = Comparison(), Comparison()
        witness = None
        for trial in range(33):
            q, k, v = (rng.vector(arithmetic, width) for width in (dk, dk, dv))
            k = k / np.float32(np.sqrt(np.sum(k * k, dtype=np.float32) + np.float32(1e-6)))
            state = rng.vector(arithmetic, dk * dv, 128).reshape(dk, dv)
            gates = np.array([rng.integer(257) / 64 - 2, rng.integer(257) / 64 - 2,
                              -(rng.integer(127) + 1) / 64, 0], dtype=np.float32)
            if trial == 0 and dk == 1:
                q.fill(1)
                k.fill(1)
                v.fill(2 ** -25)
                state.fill(1)
                gates[:] = (0, 100, 0, 0)
            output, fused, wide = (np.empty(dv, dtype=np.float32) for _ in range(3))
            arrays = [np.ascontiguousarray(a, dtype=np.float32) for a in
                      (q, k, v, gates, state, output, fused, wide)]
            result = compare(dk, dv, *(a.ctypes.data_as(pointer) for a in arrays))
            if result != 1:
                raise RuntimeError(f"native comparison was not authenticated: dk={dk}, dv={dv}, trial={trial}, code={result}")
            fused_metric.observe(output, fused, {"trial": trial})
            wide_metric.observe(output, wide, {"trial": trial})
            if trial == 0 and dk == 1:
                if np.any(output != 0) or np.any(fused != np.float32(2 ** -25)):
                    raise AssertionError("native minimal witness did not reproduce")
                witness = {"serial": float(output[0]), "fused": float(fused[0]),
                           "fp64_readout": float(wide[0]), "serial_bits": hex(int(output.view(np.uint32)[0])),
                           "fused_bits": hex(int(fused.view(np.uint32)[0]))}
            # Every production step also authenticated its complete state.
        results.append({"dk": dk, "dv": dv, "trials": 33,
                        "fused_fp32": vars(fused_metric), "fp64_readout": vars(wide_metric),
                        "minimal_witness": witness})
    return {"seed": seed, "library": str(library), "build": manifest,
            "core_sha256": core_hash,
            "bridge_sha256": hashlib.file_digest(library.open("rb"), "sha256").hexdigest(),
            "requested_isa": os.environ.get("LLAMINAR_ISA_LEVEL", "native"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "runtime default"),
            "native_control": "every output and updated state matched the FMA control byte-for-byte",
            "cases": results}


def main():
    """Emit a reproducible native diagnostic without treating byte drift as success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_native(args.library, args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"native_control": result["native_control"],
                      "cases": result["cases"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
