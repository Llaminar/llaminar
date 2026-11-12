#!/usr/bin/env python3
## @file extract_loader_control_affinity.py
# @brief Bind complete production loader-construction bodies to a host-only fixture.
"""Extract original loader control boundaries; installed tests need no compiler.

The staging pool is compiled as a complete original class, including its task
queue, worker body and retirement. Progress start/stop and the complete weight
preparation method remain original source. File/backend/model collaborators are
authored observation operands; no loading or quantization algorithm is copied.
"""
import argparse
import hashlib
import json
import pathlib

from extract_coordinator_readiness import extract


def main() -> None:
    """Generate source-bound original methods for the existing native compiler."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path)
    args = parser.parse_args()
    root = args.source_root / "src/v2"
    pipeline = root / "loaders/gpu_pipeline/DeviceLoadPipeline.cpp"
    source = pipeline.read_text()
    first = source.index("        struct StagedSource\n")
    last = source.index("        const char *repackFormatName(", first)
    staging = source[first:last].rstrip()
    if not staging.endswith("};") or "class StagingLanePool" not in staging:
        raise ValueError("Expected the complete original staging class before the format-name helper")
    bodies = ["// Generated complete production loader construction/task bodies.\nnamespace llaminar2 {\n", staging]
    inputs = [{"path": str(pipeline.resolve()), "sha256": hashlib.sha256(pipeline.read_bytes()).hexdigest(),
               "operands": "complete StagedSource and StagingLanePool definitions"}]
    for relative, owner, methods in [
        ("loaders/WeightLoadProgressAggregator.cpp", "WeightLoadProgressAggregator", ["startPolling", "stopPolling"]),
        ("loaders/WeightManager.cpp", "WeightManager", ["prepareWeightsForDeviceImpl"]),
    ]:
        path = root / relative
        bodies.extend(extract(path.read_text(), owner, method) for method in methods)
        inputs.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "methods": methods, "instrumentation": "none"})
    bodies.append("}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(bodies))
    if args.manifest:
        args.manifest.write_text(json.dumps({"inputs": inputs,
            "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
