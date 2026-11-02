#!/usr/bin/env python3
"""Compile actual original/current alias guards against real CPU storage addresses.

The only semantic adapter is gpu_data_ptr()→raw_data(): the tests allocate real
CPU Tensor roots/leases and never forge a GPU allocation or backend generation.
Guard branches, role/range checks and physical overlap arithmetic are unchanged.
"""
import argparse
import pathlib
import re


def definition(source: str, name: str) -> str:
    """Preserve one complete actual free-function definition and reject ambiguity."""
    matches = list(re.finditer(r"^(?:bool|void) " + name + r"\(", source, re.M))
    if len(matches) != 1:
        raise ValueError(f"Expected one actual alias guard {name}")
    start = matches[0].start()
    masked = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', lambda m: " " * len(m[0]), source, flags=re.S)
    end = masked.index("{", start) + 1
    depth = 1
    while depth:
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    return source[start:end].replace("gpu_data_ptr()", "raw_data()")


def main() -> None:
    """Generate both original failure controls and current actual guard bodies at build time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    sections = ["// Actual native alias guards; CPU address accessor substitution only.\n#pragma once\n"]
    for name, path in [("candidate", args.source_root / "src/v2/kernels/qwen4exp/PooledImmutableNativeKernels.cpp"),
                       ("original", args.source_root / "tests/v2/fixtures/qwen4exp/original_PooledImmutableNativeKernels.inc")]:
        source = path.read_text()
        methods = ["overlaps"] + (["contained"] if name == "original" else []) + ["ownerAliases", "coordinatorAliases"]
        sections.append(f"namespace llaminar2::qwen4exp::alias_{name} {{\nnamespace n=pooled_native;\n" + "\n".join(definition(source, method) for method in methods) + "\n}\n")
    # Replay the actual private argument assembler, retaining all branches and
    # member operand names as explicit immutable function parameters. No seal,
    # prepared GPU owner or native allocation is synthesized by this adapter.
    for name, path in [("candidate", args.source_root / "src/v2/kernels/qwen4exp/PooledImmutableNativeOperands.cpp"),
                       ("original", args.source_root / "tests/v2/fixtures/qwen4exp/original_PooledImmutableNativeOperands.inc")]:
        source = path.read_text()
        def actual(signature):
            if source.count(signature) != 1:
                raise ValueError("Expected one actual native operand assembler: " + signature)
            start = source.index(signature)
            masked = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', lambda m: " " * len(m[0]), source, flags=re.S)
            opening = masked.index("{", start)
            end, depth = opening + 1, 1
            while depth:
                depth += (masked[end] == "{") - (masked[end] == "}")
                end += 1
            return source[start:end].replace("gpu_data_ptr()", "raw_data()"), source[opening:end]
        index, _ = actual("std::size_t index(PooledNativeOperand role)")
        address, _ = actual("const void* address(const PooledNativeOperandSpans& spans,PooledNativeOperand role)")
        _, body = actual("n::OwnerInputs PooledImmutableOwnerOperands::native()const")
        signature = "n::OwnerInputs ownerNativeInputs(const PooledNativeOperandSpans& spans_,n::OwnerAction action_,n::SourceCondition condition_,n::RecordLayout records_)"
        sections.append(f"namespace llaminar2::qwen4exp::arguments_{name} {{\nnamespace n=pooled_native;\n" + index + "\n" + address + "\n" + signature + body + "\n}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sections))


if __name__ == "__main__":
    main()
