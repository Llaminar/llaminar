#!/usr/bin/env python3
"""Substitute a literal former NCCL/RCCL load body into a disposable source copy.

The configured-path tests compile that whole actual loader body in an isolated
namespace. This generator preserves the original bare-SONAME failure as a build
negative; it creates no production switch, alternate loader or native runtime.
"""
import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests/v2/unit/scripts"))
from check_cpu_prepared_allocation_policy import executable_source


def load_definition(source):
    """Find exactly one complete real load definition, ignoring comments/literals."""
    code = executable_source(source)
    matches = list(re.finditer(r"\bbool\s+load\s*\(\s*const\s+char\s*\*\s*library_path\s*\)\s*\{", code))
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one actual NCCL loader load definition")
    start = matches[0].start()
    end = code.index("{", start) + 1
    depth = 1
    while depth:
        if end == len(code):
            raise RuntimeError("Unclosed actual NCCL loader load definition")
        depth += (code[end] == "{") - (code[end] == "}")
        end += 1
    return start, end


def main():
    """Write only a requested build-owned source file after all validation passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("NCCL", "RCCL"), default="NCCL")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--original-load", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in (args.source.resolve(), args.original_load.resolve()):
        parser.error("Original control output must not replace an input")
    source = args.source.read_text()
    original = args.original_load.read_text()
    start, end = load_definition(source)
    old_start, old_end = load_definition(original)
    body = original[old_start:old_end]
    expected = ('library_path ? library_path : "libllaminar_nccl.so.2"'
                if args.backend == "NCCL" else "// Already loaded")
    if expected not in body:
        raise RuntimeError("Negative control lost its literal original selection defect")
    output = source[:start] + body + source[end:]
    if output == source:
        raise RuntimeError("Original control did not change production path selection")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)


if __name__ == "__main__":
    main()
