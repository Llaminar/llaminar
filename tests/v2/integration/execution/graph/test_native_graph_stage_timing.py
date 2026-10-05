#!/usr/bin/env python3
"""Certify opt-in stage timing on real model-free captured CUDA execution.

The canonical graph fixture proves captured output/snapshot byte equality.
This gate additionally requires exact model stage attribution and completed
GPU intervals from the native observer. It adds no alternate execution path,
and retains its JSON/CSV/HTML receipts in the build's testing directory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "performance" / "kernels"))
from native_graph_stage_report import write_report


def main() -> None:
    """Require native graph ownership, real intervals and the fixture's byte oracle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--gpu-test", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="cuda-native-stage-", dir=args.output_root))
    env = os.environ.copy()
    # A diagnostic may only inject its own observer. Existing preload policy is
    # rejected instead of silently combining unqualified instrumentation.
    if env.get("LD_PRELOAD"):
        parser.error("This focused gate requires an uninstrumented preload environment")
    env.update(LD_PRELOAD=str(args.observer.resolve()),
               LLAMINAR_NATIVE_EVENT_TRACE_DIR=str(output),
               LLAMINAR_NATIVE_EVENT_TRACE_ALL_FLAT="1",
               LLAMINAR_NATIVE_EVENT_TRACE_MIN_NODES="1",
               LLAMINAR_NATIVE_EVENT_TRACE_REQUIRE_STAGE_NAMES="1")
    env.pop("LLAMINAR_NATIVE_EVENT_TRACE_KERNEL_CONTAINS", None)
    command = [str(args.gpu_test.resolve()),
               "--gtest_filter=GPUGraphCaptureExecutionTest.SnapshotManifestUsesOneArenaAndPublishesCapturedBytes"]
    with (output / "capture.log").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((output / "capture.log").read_text(), file=sys.stderr)
        raise RuntimeError(f"Canonical captured byte fixture failed: {result.returncode}; receipts {output}")
    report = write_report(sorted(output.glob("cuda-event-*.json")), output)
    if not any({"rmsnorm", "residual_add"}.issubset({stage["name"] for stage in graph["stages"]})
               and graph["attributed_nodes"] >= 2 and graph["timed_native_nodes"] >= 2
               and all(stage["measurement"] == "complete" for stage in graph["stages"])
               for graph in report["graphs"]):
        raise RuntimeError("Canonical capture did not retain both complete named GPU stages")
    print(json.dumps({"status": "PASS", "captured_snapshot_byte_oracle": True,
                      "report": str(output / "stage-timing.html"), "graphs": len(report["graphs"])}))


if __name__ == "__main__":
    main()
