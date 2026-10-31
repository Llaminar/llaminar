#!/usr/bin/env python3
"""Extract actual coordinator lifecycle methods for an SDK-free scheduling replay.

Only explicit mutex template arguments are rebound to the fixture's observable
mutex. Branches, flag operations, condition predicates and thread joins remain
the actual source. Hardware initialization/cleanup are fixture doubles.
"""
import argparse
import hashlib
import json
import pathlib
import re


def extract(source: str, owner: str, method: str) -> str:
    """Return one complete original C++ definition, preserving its source bytes."""
    token = f"{owner}::{method}("
    start = source.index(token)
    start = source.rfind("\n", 0, start) + 1
    opened = source.index("{", start)
    masked = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                    lambda match: " " * len(match.group()), source, flags=re.S)
    depth = 1
    end = opened + 1
    while depth:
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    return source[start:end]


def main() -> None:
    """Generate the immutable actual-method include and its source binding."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path)
    args = parser.parse_args()
    output = ["// Generated from actual original production lifecycle methods.\n"]
    declarations = ["// Actual header state operands; only synchronization types are observable.\n"]
    inputs = []
    for vendor, suffix in [("NCCL", "cu"), ("RCCL", "cpp")]:
        path = args.source_root / f"src/v2/collective/coordinators/{vendor}Coordinator.{suffix}"
        text = path.read_text()
        owner = vendor + "Coordinator"
        header = path.with_suffix(".h")
        header_text = header.read_text()
        fields = ["device_ordinals_", "num_devices_", "initialized_", "last_error_", "coordinator_thread_",
                  "running_", "init_success_", "init_complete_", "work_queue_", "queue_mutex_", "queue_cv_"]
        if vendor == "RCCL":
            fields.append("collective_performed_")
        original_fields = []
        for field in fields:
            matches = re.findall(r"^\s*((?:mutable )?std::[^\n;]+\b" + re.escape(field) +
                                 r"\b[^\n;]*;|int " + re.escape(field) + r"[^\n;]*;)", header_text, re.M)
            if len(matches) != 1:
                raise ValueError(f"Expected one actual {owner} state declaration for {field}: {matches}")
            original_fields.append(matches[0].replace("std::mutex", "ReplayMutex")
                                   .replace("std::condition_variable", "ReplayCondition"))
        declarations.append(f"struct {owner} {{\nSchedule schedule;\n" + "\n".join(original_fields) +
            f"\n{owner}(); ~{owner}();\nbool initialize(const std::vector<int>&);\n"
            "void shutdown(); void coordinatorLoop(); void initializeOnThread(); void cleanupOnThread();\n"
            + owner + "(const " + owner + "&)=delete;\n" + owner + "& operator=(const " + owner + "&)=delete;\n};\n")
        definitions = []
        for method in [owner, "~" + owner, "initialize", "shutdown", "coordinatorLoop"]:
            body = extract(text, owner, method)
            definitions.append(body.replace("<std::mutex>", "<ReplayMutex>"))
        output.append("namespace llaminar2 {\n" + "\n".join(definitions) + "\n}\n")
        inputs.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "header": str(header.resolve()), "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
                       "methods": [owner, "~" + owner, "initialize", "shutdown", "coordinatorLoop"],
                       "instrumentation": "explicit mutex template arguments only; hardware calls doubled"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output))
    args.output.with_name("CoordinatorReadinessFields.inc").write_text("\n".join(declarations))
    if args.manifest:
        args.manifest.write_text(json.dumps({"inputs": inputs, "output_sha256":
            hashlib.sha256(args.output.read_bytes()).hexdigest(), "fields_sha256":
            hashlib.sha256(args.output.with_name("CoordinatorReadinessFields.inc").read_bytes()).hexdigest()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
