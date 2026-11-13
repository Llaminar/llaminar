#!/usr/bin/env python3
"""Build exact native host-method replay bodies without compiling or loading a GPU.

Only source extraction occurs here. The C++ replay supplies runtime entrypoint
and control-object doubles; all enqueue/error decisions remain literal production
method bodies. Missing/ambiguous signatures fail the build instead of certifying
an invented or stale implementation.
"""
import argparse
from pathlib import Path


def method(text, signature):
    """Extract one complete actual definition, rejecting a changed method boundary."""
    if text.count(signature) != 1:
        raise RuntimeError("Missing or ambiguous native method: " + signature)
    start = text.index(signature)
    opening = text.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def main():
    """Generate only the exact existing CUDA/ROCm host definitions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chunks = ["// Generated actual production host bodies; no native runtime is loaded.\n"]
    context = method((args.source / "src/v2/collective/LocalTPContext.cpp").read_text(),
                     "    bool LocalTPContext::nativeBroadcastBytesOnStream")
    for backend, extension in (("NCCL", "cu"), ("RCCL", "cpp")):
        chunks.append("namespace current_" + backend.lower() + " {\n")
        for family, name, suffix in (("coordinators", "Coordinator", extension),
                                     ("backends", "Backend", "cpp")):
            path = args.source / ("src/v2/collective/" + family + "/" + backend + name + "." + suffix)
            chunks.append(method(path.read_text(), "    NativeCollectiveEnqueueResult " +
                                 backend + name + "::nativeBroadcastBytesOnStream") + "\n")
        chunks.append(context + "\n}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(chunks))


if __name__ == "__main__":
    main()
