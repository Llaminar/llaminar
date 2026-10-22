#!/usr/bin/env python3
"""Compare actual C++/Python trainer metadata loaders without backend startup.

Only the bound device-free projection executable runs. It links no inference
core or accelerator library, reads the shared JSON/catalog and emits metadata.
The full ordered records are compared, including existing sealed assignments.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests/v2/performance/kernels"))
from native_vnni_dispatch.shape_manifest import load_shape_manifest


def main() -> None:
    """Reject loader drift, truncated output or vacuous record comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    outcome = subprocess.run(
        [str(args.binary.resolve(strict=True))],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if outcome.returncode != 0 or outcome.stderr:
        raise RuntimeError("Original C++ manifest projection failed: " + outcome.stderr)
    actual = json.loads(outcome.stdout)
    expected = [
        dict(
            name=shape.name,
            n=shape.n,
            k=shape.k,
            model_family=shape.model_family,
            role=shape.role.value,
            exact_overlay=shape.exact_overlay,
            aspect_bucket=shape.aspect_bucket.value,
            fast_partition=shape.fast_partition.value if shape.fast_partition else None,
            verifier_partition=(
                shape.verifier_partition.value if shape.verifier_partition else None
            ),
            prefill_partition=(
                shape.prefill_partition.value if shape.prefill_partition else None
            ),
        )
        for shape in load_shape_manifest().shapes
    ]
    if not expected or actual != expected:
        raise RuntimeError(
            "C++/Python shared NativeVNNI shapes or partition assignments differ"
        )
    print(json.dumps(dict(
        scope="Device-free original loader metadata only",
        shape_count=len(expected),
        all_fields_equal=True,
    )))


if __name__ == "__main__":
    main()
