#!/usr/bin/env python3
"""Bind exact cold production inspection bodies to a recorded SDK without loading it."""
import argparse
import hashlib
from pathlib import Path

FILES = (("CUDAGraphCapture", "src/v2/backends/cuda/CUDAGraphCapture.cu"),
         ("HIPGraphCapture", "src/v2/backends/rocm/HIPGraphCapture.cpp"))


def generate(source, output):
    bodies = []
    for name, relative in FILES:
        text = (source / relative).read_text()
        start = text.index("    bool " + name + "::inspectFlatGraphStructure(")
        opening = text.index("{", start)
        depth, end = 1, opening + 1
        while depth:
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            end += 1
        body = text[start:end]
        bodies.append("// Exact production body SHA256: " + hashlib.sha256(body.encode()).hexdigest() + "\n" + body)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("/** @file FlatGraphInspectionBodies.inc\n * @brief Exact build-time production methods; recorded SDK only. */\n" + "\n\n".join(bodies) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.output)
