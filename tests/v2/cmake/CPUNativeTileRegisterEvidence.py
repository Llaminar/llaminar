#!/usr/bin/env python3
"""Build and authenticate a register-allocation witness for the native NEON projection tiles.

The build phase compiles CPUNativeNEONProjectionTileRegisterProbe.cpp with the
real target's ordered compile command at -O3 and audits the assembly of every
tile K-loop program (CPUNativeNEONProjectionTile<...>::run/pairBlock): any
vector-register (q/v) load or store relative to the stack pointer is a spill
and fails the build. Runtime Unit/preflight verification needs only the
source/config-bound JSON and assembly, never a compiler.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess

_ENVELOPE = Path(__file__).resolve().with_name("CPUCodegenEnvelope.py")
_SPEC = importlib.util.spec_from_file_location("cpu_codegen_envelope", _ENVELOPE)
assert _SPEC and _SPEC.loader
envelope = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(envelope)

# A tile K-loop program: mangled CPUNativeNEONProjectionTile member run<...> or pairBlock<...>.
PROGRAM = re.compile(r"27CPUNativeNEONProjectionTileILm\d+EE(?:3run|9pairBlock)I")
LABEL = re.compile(r"^([A-Za-z_.$][\w.$]*):\s*(?:[#;/@].*)?$")
# Assembler-local labels (GNU ".L*", Apple "L*"/"l*") stay inside the current function.
LOCAL = re.compile(r"^(?:\.|L|l[A-Z_\d]|\$)")
VECTOR_STACK = re.compile(r"^\s*(?:ldr|str|ldp|stp|ldur|stur|ld1|st1)\s+[^\n]*\b[qv]\d+\b[^\n]*\[sp\b", re.IGNORECASE)


def program_spills(assembly: str) -> dict[str, int]:
    """Return vector stack-traffic counts for every tile K-loop function in compiler -S output."""
    counts: dict[str, int] = {}
    current = None
    for line in assembly.splitlines():
        label = LABEL.match(line)
        if label and not LOCAL.match(label.group(1)):
            current = label.group(1) if PROGRAM.search(label.group(1)) else None
            if current is not None:
                counts.setdefault(current, 0)
            continue
        if current is not None and VECTOR_STACK.match(line):
            counts[current] += 1
    return counts


def require_registers(counts: dict[str, int]) -> None:
    """Fail closed for a witness without tile programs or with any vector spill."""
    if not counts:
        raise ValueError("register witness contains no native NEON tile K-loop program")
    spilled = {name: count for name, count in counts.items() if count}
    if spilled:
        listing = ", ".join(f"{name}={count}" for name, count in sorted(spilled.items())[:8])
        raise ValueError(f"native NEON tile programs spill vector registers ({len(spilled)} functions): {listing}")


def build(args: argparse.Namespace) -> None:
    """Compile the probe with the target command and atomically publish the source/config-bound proof."""
    output = Path(args.proof)
    output.parent.mkdir(parents=True, exist_ok=True)
    assembly = output.with_suffix(".s")
    output.unlink(missing_ok=True)
    assembly.unlink(missing_ok=True)
    output.with_suffix(".tmp").unlink(missing_ok=True)
    command, directory = envelope.target_command(args)
    command += ["-O3", "-fno-sanitize=all", "-fno-lto", "-std=c++20", "-S", args.source, "-o", str(assembly)]
    subprocess.run(command, cwd=directory, check=True)
    counts = program_spills(assembly.read_text())
    require_registers(counts)
    result = {
        "schema": 1, "signature": args.signature,
        "source_sha256": envelope.digest(Path(args.source)),
        "checker_sha256": envelope.digest(Path(__file__)),
        "envelope_sha256": envelope.digest(_ENVELOPE),
        "target_command_sha256": envelope.command_digest(*envelope.target_command(args)),
        "assembly_file": assembly.name, "assembly_sha256": envelope.digest(assembly),
        "program_spills": counts,
    }
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    temporary.replace(output)


def verify(args: argparse.Namespace) -> None:
    """Validate the retained proof without invoking or requiring a compiler."""
    proof_path = Path(args.proof)
    proof = json.loads(proof_path.read_text())
    if proof.get("schema") != 1 or proof.get("signature") != args.signature:
        raise ValueError("tile register proof has a foreign schema/configuration")
    for key, path in (("source_sha256", Path(args.source)), ("checker_sha256", Path(__file__)),
                      ("envelope_sha256", _ENVELOPE)):
        if proof.get(key) != envelope.digest(path):
            raise ValueError(f"stale tile register input: {path}")
    if proof.get("target_command_sha256") != envelope.command_digest(*envelope.target_command(args)):
        raise ValueError("actual target compile command changed after the register witness")
    name = proof.get("assembly_file")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("tile register proof must name one retained local assembly artifact")
    assembly = proof_path.parent / name
    if proof.get("assembly_sha256") != envelope.digest(assembly):
        raise ValueError("tile register assembly is missing or changed")
    counts = program_spills(assembly.read_text())
    if proof.get("program_spills") != counts:
        raise ValueError("tile register evidence changed")
    require_registers(counts)


def main() -> None:
    """Run the requested build/verification phase; errors are fatal, never skips."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--proof", required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--compile-database", required=True)
    parser.add_argument("--target-source", required=True)
    args = parser.parse_args()
    if args.action == "build":
        build(args)
    verify(args)
    print(f"PASS: {len(program_spills((Path(args.proof).with_suffix('.s')).read_text()))} native NEON tile programs keep their working set in registers")


if __name__ == "__main__":
    main()
