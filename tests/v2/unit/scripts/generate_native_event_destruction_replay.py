#!/usr/bin/env python3
"""Compile exact production cleanup bodies into a fake-SDK replay; never load a runtime."""
import argparse
import hashlib
from pathlib import Path

FILES = (("NvidiaDeviceContext", "src/v2/backends/cuda/NvidiaDeviceContext.cu"),
         ("AMDDeviceContext", "src/v2/backends/rocm/AMDDeviceContext.cpp"),
         ("CUDABackend", "src/v2/backends/cuda/CUDABackend.cu"),
         ("ROCmBackend", "src/v2/backends/rocm/ROCmBackend.cpp"))

def generate(source, output):
    bodies = []
    for name, relative in FILES:
        text = (source / relative).read_text()
        start = text.index("    void " + name + "::destroyEvent(")
        opening = text.index("{", start)
        depth, end = 1, opening + 1
        while depth:
            if text[end] == "{": depth += 1
            elif text[end] == "}": depth -= 1
            end += 1
        body = text[start:end]
        bodies.append("// Exact production body SHA256: " + hashlib.sha256(body.encode()).hexdigest() + "\n" + body)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("/** @file NativeEventDestructionBodies.inc\n * @brief Build-time exact native cleanup bodies; fake SDK only. */\n" + "\n\n".join(bodies) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate(args.source, args.output)
