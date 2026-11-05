#!/usr/bin/env python3
"""Build and authenticate an optimized native CPU compiler-envelope witness.

The build phase invokes the configured compiler on one SDK-free byte-dot motif
and audits its real assembly, not just preprocessor macros. Runtime Unit and
preflight verification need only the source-bound JSON/assembly artifacts, not
a compiler SDK. This detects Darwin's implicit target-CPU instruction leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


def digest(path: Path) -> str:
    """Return the exact source/artifact SHA256; missing input fails closed."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instruction_counts(assembly: str) -> dict[str, int]:
    """Count actual optional-instruction mnemonics in the isolated compiler output."""
    return {name: len(re.findall(r"(?m)^\s*" + name + r"(?:\.|\s)", assembly))
            for name in ("sdot", "udot", "usdot", "sudot", "smmla", "ummla", "usmmla")}


def require_envelope(mode: str, counts: dict[str, int]) -> None:
    """Reject implicit optional ISA, or a DotProd witness that never exercises dot instructions."""
    if any(counts[name] for name in ("usdot", "sudot", "smmla", "ummla", "usmmla")):
        raise ValueError(f"compiler emitted unadmitted matrix/dot extensions: {counts}")
    dots = counts["sdot"] + counts["udot"]
    if mode == "NEON" and dots:
        raise ValueError(f"baseline NEON compiler envelope leaked DotProd: {counts}")
    if mode == "NEON_DOTPROD" and not dots:
        raise ValueError("DotProd compiler witness did not exercise native dot instructions")


def require_native_assembly(assembly: str) -> None:
    """Reject empty/IR/LTO payloads that never prove an AArch64 instruction body."""
    if re.search(r"(?m)^\s*(?:define\s|target\s+(?:triple|datalayout))|\.gnu\.lto", assembly):
        raise ValueError("CPU envelope witness is IR/LTO data, not native assembly")
    if not re.search(r"(?m)^_?native_cpu_codegen_byte_dot:\s*(?:[#;/].*)?$", assembly):
        raise ValueError("CPU envelope witness lacks its native function label")
    if not re.search(r"(?m)^\s*ret\s*(?:[#;/].*)?$", assembly):
        raise ValueError("CPU envelope witness lacks a native return instruction")


def expand_responses(arguments: list[str], directory: str, active: tuple[Path, ...] = ()) -> list[str]:
    """Recursively retain effective response-file operands; missing/cyclic inputs fail closed."""
    expanded = []
    for argument in arguments:
        if not argument.startswith("@"):
            expanded.append(argument)
            continue
        response = (Path(directory) / argument[1:]).resolve()
        if response in active or len(active) >= 64:
            raise ValueError("cyclic or excessive CPU compiler response-file nesting")
        expanded.extend(expand_responses(shlex.split(response.read_text()), directory, active + (response,)))
    return expanded


def target_command(args: argparse.Namespace) -> tuple[list[str], str]:
    """Borrow the real target's ordered compile command, removing only its build outputs/input."""
    database = json.loads(Path(args.compile_database).read_text())
    source = Path(args.target_source).resolve()
    matches = [entry for entry in database if Path(entry["file"]).resolve() == source]
    if len(matches) != 1:
        raise ValueError("CPU envelope requires exactly one actual target compile command")
    entry = matches[0]
    directory = entry["directory"]
    raw = entry.get("arguments") or shlex.split(entry["command"])
    expanded = expand_responses(raw, directory)
    if Path(expanded[0]).resolve() != Path(args.compiler).resolve():
        raise ValueError("compile database names a foreign compiler")
    command = [expanded[0]]
    index, removed_source = 1, False
    while index < len(expanded):
        argument = expanded[index]
        if argument in ("-o", "-MF", "-MT", "-MQ"):
            index += 2
            continue
        if argument in ("-c", "-MD", "-MMD", "-MP"):
            index += 1
            continue
        candidate = Path(argument)
        if not argument.startswith("-") and (Path(directory) / candidate).resolve() == source:
            removed_source = True
        else:
            command.append(argument)
        index += 1
    if not removed_source:
        raise ValueError("CPU target command did not identify its original source operand")
    return command, directory


def command_digest(command: list[str], directory: str) -> str:
    """Bind every ordered effective target/TU flag, independently of compiler availability."""
    payload = json.dumps({"command": command, "directory": directory}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def build(args: argparse.Namespace) -> None:
    """Compile one optimized motif and atomically publish its complete source/config-bound proof."""
    output = Path(args.proof)
    output.parent.mkdir(parents=True, exist_ok=True)
    assembly = output.with_suffix(".s")
    # No failed rebuild may leave an earlier success consumable. Retire both
    # success artifacts before version checks, subprocesses or compilation.
    output.unlink(missing_ok=True)
    assembly.unlink(missing_ok=True)
    output.with_suffix(".tmp").unlink(missing_ok=True)
    command, directory = target_command(args)
    actual_command_hash = command_digest(command, directory)
    version = subprocess.check_output([args.compiler, "--version"], cwd=directory, text=True)
    predefined = subprocess.check_output(command + ["-dM", "-E", "-x", "c++", "-"],
                                        cwd=directory, input="", text=True)
    macros = dict(re.findall(r"(?m)^#define (\w+) ([0-9]+)$", predefined))
    if args.compiler_id == "AppleClang":
        keys = ("__clang_major__", "__clang_minor__", "__clang_patchlevel__", "__apple_build_version__")
    elif args.compiler_id == "Clang":
        keys = ("__clang_major__", "__clang_minor__", "__clang_patchlevel__")
    elif args.compiler_id == "GNU":
        keys = ("__GNUC__", "__GNUC_MINOR__", "__GNUC_PATCHLEVEL__")
    else:
        raise ValueError("unsupported compiler identity for native ISA evidence")
    observed_version = ".".join(macros[key] for key in keys)
    if observed_version != args.compiler_version:
        raise ValueError(f"compiler changed since configuration: {observed_version} != {args.compiler_version}")
    # Audit the effective global/configuration ISA flags under optimization.
    # Diagnostic instrumentation is explicitly removed, not architecture flags:
    # sanitizers can inhibit the auto-vectorization needed to expose this bug.
    command += ["-O3", "-fno-sanitize=all", "-fno-lto", "-std=c++20", "-S", args.source, "-o", str(assembly)]
    subprocess.run(command, cwd=directory, check=True)
    assembly_text = assembly.read_text()
    require_native_assembly(assembly_text)
    counts = instruction_counts(assembly_text)
    require_envelope(args.mode, counts)
    result = {
        "schema": 1, "mode": args.mode, "signature": args.signature,
        "source_sha256": digest(Path(args.source)), "policy_sha256": digest(Path(args.policy)),
        "checker_sha256": digest(Path(__file__)),
        "target_manifest_file": Path(args.target_manifest).name,
        "target_manifest_sha256": digest(Path(args.target_manifest)),
        "target_compile_directory": directory, "target_command_sha256": actual_command_hash,
        "compiler_version": version, "compiler_cmake_version": observed_version, "command": command,
        "assembly_file": assembly.name, "assembly_sha256": digest(assembly),
        "instruction_counts": counts,
    }
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    temporary.replace(output)


def verify(args: argparse.Namespace) -> None:
    """Validate the retained proof without invoking or requiring a compiler."""
    proof_path = Path(args.proof)
    proof = json.loads(proof_path.read_text())
    if proof.get("schema") != 1 or proof.get("mode") != args.mode or proof.get("signature") != args.signature:
        raise ValueError("CPU compiler-envelope proof has a foreign schema/configuration")
    for key, path in (("source_sha256", Path(args.source)), ("policy_sha256", Path(args.policy)),
                      ("checker_sha256", Path(__file__))):
        if proof.get(key) != digest(path):
            raise ValueError(f"stale CPU compiler-envelope input: {path}")
    command, directory = target_command(args)
    if proof.get("target_command_sha256") != command_digest(command, directory):
        raise ValueError("actual CPU target/TU compile command changed after the witness")
    manifest_name = proof.get("target_manifest_file")
    if not isinstance(manifest_name, str) or Path(manifest_name).name != manifest_name:
        raise ValueError("CPU proof requires one retained target configuration manifest")
    if proof.get("target_manifest_sha256") != digest(proof_path.parent / manifest_name):
        raise ValueError("CPU target configuration changed after its compiler witness")
    name = proof.get("assembly_file")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("CPU proof assembly must name one retained local artifact")
    assembly = proof_path.parent / name
    if proof.get("assembly_sha256") != digest(assembly):
        raise ValueError("CPU compiler-envelope assembly is missing or changed")
    assembly_text = assembly.read_text()
    require_native_assembly(assembly_text)
    counts = instruction_counts(assembly_text)
    if proof.get("instruction_counts") != counts:
        raise ValueError("CPU compiler-envelope instruction evidence changed")
    require_envelope(args.mode, counts)


def main() -> None:
    """Run the requested build/verification phase; errors are fatal, never skips."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--proof", required=True)
    parser.add_argument("--mode", choices=("NEON", "NEON_DOTPROD"), required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--compiler")
    parser.add_argument("--compiler-version")
    parser.add_argument("--compiler-id")
    parser.add_argument("--compile-database")
    parser.add_argument("--target-source")
    parser.add_argument("--target-manifest")
    args = parser.parse_args()
    if args.action == "build":
        if not all((args.compiler, args.compiler_version, args.compiler_id, args.compile_database,
                    args.target_source, args.target_manifest)):
            parser.error("build requires the exact configured compiler/identity/version")
        build(args)
    verify(args)
    print(f"PASS: source/config-bound {args.mode} optimized compiler-envelope proof")


if __name__ == "__main__":
    main()
