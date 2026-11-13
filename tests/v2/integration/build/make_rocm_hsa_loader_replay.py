#!/usr/bin/env python3
"""Retain the five actual HSA caller bodies for SDK-free loader-boundary replay.

Only includes/class dependencies are replaced by inert host declarations. The
original control changes the five selected paths back to their literal former
bare name; native calls, symbol lookup and close/error bodies are not rewritten.
"""
import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests/v2/unit/scripts"))
from check_cpu_prepared_allocation_policy import executable_source


def replay(source, original=False):
    """Extract complete original method definitions after comment/literal masking."""
    code = executable_source(source)
    bodies = []
    for name in ("hsaMemoryLock", "hsaMemoryUnlock", "hsaInteropMapBuffer", "hsaInteropUnmapBuffer", "getHsaAgent"):
        matches = list(re.finditer(r"\b(?:bool|void)\s+ROCmBackend::" + name + r"\s*\([^)]*\)\s*\{", code))
        if len(matches) != 1:
            raise ValueError("Expected one actual HSA method: " + name)
        start = matches[0].start()
        end = code.index("{", start) + 1
        depth = 1
        while depth:
            if end == len(code):
                raise ValueError("Unclosed actual HSA method: " + name)
            depth += (code[end] == "{") - (code[end] == "}")
            end += 1
        body = source[start:end]
        if body.count("dlopen(LLAMINAR_HSA_RUNTIME_LIBRARY_PATH,") != 1:
            raise ValueError("Actual HSA caller lost configured path: " + name)
        if original:
            body = body.replace("dlopen(LLAMINAR_HSA_RUNTIME_LIBRARY_PATH,", 'dlopen("libhsa-runtime64.so",')
        bodies.append(body)
    return "\n\n".join(bodies) + "\n"


def main():
    """Write only the requested build-owned include; never modify production."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original", action="store_true")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Replay output must not replace source")
    output = replay(args.source.read_text(), args.original)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)


if __name__ == "__main__":
    main()
