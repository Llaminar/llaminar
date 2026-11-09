#!/usr/bin/env python3
"""Build and authenticate a register-allocation witness for the native NEON projection tiles.

Build phase: compile CPUNativeNEONTileRegisterProbe.cpp with the real target's
ordered compile command at -O3, record the complete transitive include closure
(compiler-reported dependencies, hashed), and audit the assembly:

- inventory: every tile K-loop program the probe instantiates (row tile x leaf
  family x policy x encoding/map/minimum, as declared by the probe's inventory
  string) must be present exactly once;
- call closure: every function a program calls or tail-calls that the file
  defines is audited too; external callees are accepted only for the fatal
  diagnostics/abort, which never return;
- spills: FP/SIMD stack traffic of any width (b/h/s/d/q/v) through sp, the
  frame pointer or a register derived from either by a constant offset is
  counted, excluding ABI saves/restores of the callee-saved d8-d15 (each saved
  once, restored at most once per return); unclassifiable bodies are rejected;
- policy: single-row (decode) programs must not spill at all; grouped row tiles
  may not exceed the committed per-program budget for the exact compiler
  identity and ISA envelope (CPUNativeTileRegisterBudget.json, a ratchet).
  An unknown identity/envelope fails closed until it is qualified.

Verify phase (Unit/preflight, no compiler): every recorded input (probe, checker,
envelope helper, every header of the closure) must be unchanged, the actual
target command must match, and the retained assembly must re-audit clean.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess

_ENVELOPE = Path(__file__).resolve().with_name("CPUCodegenEnvelope.py")
_SPEC = importlib.util.spec_from_file_location("cpu_codegen_envelope", _ENVELOPE)
assert _SPEC and _SPEC.loader
envelope = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(envelope)

PROGRAM = re.compile(r"27CPUNativeNEONProjectionTileILm(\d)EE3runINS1_11multi_scale17NEONMultiScaleOpsILNS\d+_11NEONByteDotE(\d)EEE"
                     r"LNS_28CPUProjectionNumericalPolicyE(\d)ELNS1_21CPUNativeQuantisedEncodingE(\d)ELNS\d+_9NibbleMapE(\d)ELb(\d)E")
LABEL = re.compile(r"^([A-Za-z_.$][\w.$]*):\s*(?:[#;/@].*)?$")
LOCAL = re.compile(r"^(?:\.|L|l[A-Z_\d]|\$)")
CALL = re.compile(r"^\s*(bl|b)\s+([A-Za-z_.$][\w.$]*)\s*(?:[#;/@].*)?$")
RET = re.compile(r"^\s*ret\b")
FATAL_CALLEES = re.compile(r"rejectOutsideAdmittedPublication|^_?abort$")
INVENTORY = re.compile(r"llaminar-native-neon-tile-probe:([A-Za-z0-9,:]*)")
# (encoding, nibble map, with-minimum) dispatched per policy by Tile::dispatch().
DISPATCH = ((1, 0, 1), (1, 0, 0), (5, 0, 0), (2, 0, 0), (4, 2, 0), (0, 0, 0), (0, 2, 0), (0, 1, 1))
DOTS = {"ASIMD": 0, "DotProd": 1, "I8MM": 2}

FP_REG = r"(?:[bhsdq]\d{1,2}|v\d{1,2}(?:\.\w+)?)"
MEMORY = re.compile(r"\[\s*([a-z0-9]+)\s*(?:,\s*#?-?(?:0x[0-9a-fA-F]+|\d+))?\s*\]")
STORE_LOAD = re.compile(r"^\s*(st[rpu]\w*|ld[rpu]\w*|st1|ld1|st2|ld2|st3|ld3|st4|ld4)\b\s+(.*)$", re.IGNORECASE)
ALIAS = re.compile(r"^\s*(?:add|sub|mov)\s+(x\d+)\s*,\s*(sp|x29|x\d+)\s*(?:,\s*#?-?(?:0x[0-9a-fA-F]+|\d+)\s*)?(?:[#;/@].*)?$")
CALLEE_SAVED = {f"d{n}" for n in range(8, 16)}


def functions(assembly: str) -> dict[str, list[str]]:
    """Split compiler -S output into global function bodies; local labels stay inside."""
    bodies: dict[str, list[str]] = {}
    current = None
    for line in assembly.splitlines():
        label = LABEL.match(line)
        if label and not LOCAL.match(label.group(1)):
            current = label.group(1)
            if current in bodies:
                raise ValueError(f"symbol defined twice in the witness: {current}")
            bodies[current] = []
            continue
        if current is not None:
            bodies[current].append(line)
    return bodies


def stack_traffic(body: list[str]) -> tuple[int, dict[str, int], dict[str, int], int]:
    """Count non-ABI FP/SIMD stack traffic, callee-saved d8-d15 saves/restores and returns."""
    bases = {"sp", "x29"}
    spills, saves, restores, returns = 0, {}, {}, 0
    for line in body:
        code = re.split(r"\s(?:;|//|@)", line, maxsplit=1)[0]
        alias = ALIAS.match(code)
        if alias:
            if alias.group(2) in bases:
                bases.add(alias.group(1))
            else:
                bases.discard(alias.group(1))
            continue
        if RET.match(code):
            returns += 1
            continue
        access = STORE_LOAD.match(code)
        if not access:
            continue
        memory = MEMORY.search(access.group(2))
        if not memory or memory.group(1) not in bases:
            continue
        registers = re.findall(r"\b" + FP_REG + r"\b", access.group(2)[:memory.start()])
        if not registers:
            continue
        names = [register.split(".")[0] for register in registers]
        if all(name in CALLEE_SAVED for name in names):
            book = saves if access.group(1).lower().startswith("st") else restores
            for name in names:
                book[name] = book.get(name, 0) + 1
            continue
        spills += 1
    return spills, saves, restores, returns


def program_key(name: str) -> str:
    """Stable budget key rows:family:policy:encoding:map:minimum for a tile program symbol."""
    match = PROGRAM.search(name)
    if not match:
        raise ValueError(f"not a tile program: {name}")
    return ":".join(match.groups())


def audit(assembly: str) -> dict[str, int]:
    """Return per-program violation counts (0 = clean) over the probe's complete inventory and call closure."""
    inventory = INVENTORY.search(assembly)
    if not inventory:
        raise ValueError("register witness lacks the probe's instantiation inventory")
    expected = set()
    for entry in filter(None, inventory.group(1).split(",")):
        rows, dot = entry.split(":")
        leaf = 1 if dot == "I8MM" and rows == "1" else DOTS[dot]
        for policy in (0, 1):
            for encoding, nibble_map, minimum in DISPATCH:
                expected.add((int(rows), leaf, policy, encoding, nibble_map, minimum))
    if not expected:
        raise ValueError("register witness inventory names no tile instantiation")
    bodies = functions(assembly)
    found: dict[tuple, str] = {}
    for name in bodies:
        match = PROGRAM.search(name)
        if match:
            key = tuple(int(group) for group in match.groups())
            if key in found:
                raise ValueError(f"tile program appears twice: {name}")
            found[key] = name
    missing = expected - found.keys()
    if missing:
        raise ValueError(f"register witness is missing {len(missing)} expected tile programs, e.g. {sorted(missing)[:3]}")
    result: dict[str, int] = {}
    for key in sorted(expected):
        program = found[key]
        if not any(line.strip() and not line.strip().startswith((".", "#", ";", "//", "@")) for line in bodies[program]):
            raise ValueError(f"tile program has an empty or unclassifiable body: {program}")
        violations, pending, seen = 0, [program], set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            spills, saves, restores, returns = stack_traffic(bodies[name])
            violations += spills
            violations += sum(count - 1 for count in saves.values() if count > 1)
            violations += sum(count - max(returns, 1) for count in restores.values() if count > max(returns, 1))
            for line in bodies[name]:
                call = CALL.match(line)
                if not call or LOCAL.match(call.group(2)):
                    continue
                callee = call.group(2)
                if FATAL_CALLEES.search(callee):
                    continue  # Never returns: no live value of the program survives it.
                if callee in bodies:
                    pending.append(callee)
                else:
                    violations += 1  # An external callee cannot be audited for spills.
        result[program] = violations
    return result


BUDGET_PATH = Path(__file__).resolve().with_name("CPUNativeTileRegisterBudget.json")


def require_registers(counts: dict[str, int], identity: str, mode: str, budget: dict | None = None) -> None:
    """Single-row programs must not spill; grouped programs must stay within the committed ratchet."""
    if budget is None:
        budget = json.loads(BUDGET_PATH.read_text())
    limits = budget.get(identity, {}).get(mode)
    if limits is None:
        raise ValueError(f"no qualified tile register budget for compiler {identity} envelope {mode}")
    failures = []
    for name, count in counts.items():
        key = program_key(name)
        if key.startswith("1:"):
            if count:
                failures.append(f"{key} spills {count} (single-row programs must not spill)")
        elif key not in limits:
            failures.append(f"{key} has no budget")
        elif count > limits[key]:
            failures.append(f"{key} spills {count} > budget {limits[key]}")
    stale = set(limits) - {program_key(name) for name in counts}
    if stale:
        failures.append(f"budget names programs the witness did not emit: {sorted(stale)[:3]}")
    if failures:
        raise ValueError(f"native NEON tile register policy violated ({len(failures)}): " + "; ".join(failures[:6]))


def include_closure(command: list[str], directory: str, source: str) -> list[str]:
    """Return the compiler-reported transitive dependencies of the probe (all included headers)."""
    output = subprocess.check_output(command + ["-std=c++20", "-M", "-MT", "probe", source], cwd=directory, text=True)
    paths = shlex.split(output.replace("\\\n", " "))[1:]
    return sorted({str((Path(directory) / path).resolve()) for path in paths})


def bind_inputs(paths: list[str]) -> dict[str, str]:
    """Hash every dependency; a missing input fails closed."""
    return {path: envelope.digest(Path(path)) for path in paths}


def build(args: argparse.Namespace) -> None:
    """Compile the probe with the target command and atomically publish the source/closure/config-bound proof."""
    output = Path(args.proof)
    output.parent.mkdir(parents=True, exist_ok=True)
    assembly = output.with_suffix(".s")
    for stale in (output, assembly, output.with_suffix(".tmp"), output.with_suffix(".d")):
        stale.unlink(missing_ok=True)
    command, directory = envelope.target_command(args)
    closure = include_closure(command, directory, args.source)
    executed = command + ["-O3", "-fno-sanitize=all", "-fno-lto", "-std=c++20", "-S", args.source, "-o", str(assembly)]
    subprocess.run(executed, cwd=directory, check=True)
    counts = audit(assembly.read_text())
    require_registers(counts, args.identity, args.mode)
    version = subprocess.check_output([command[0], "--version"], cwd=directory, text=True)
    result = {
        "schema": 3, "signature": args.signature, "identity": args.identity, "mode": args.mode,
        "checker_sha256": envelope.digest(Path(__file__)),
        "envelope_sha256": envelope.digest(_ENVELOPE),
        "budget_sha256": envelope.digest(BUDGET_PATH),
        "compiler_version": version,
        "executed_command": executed, "executed_directory": directory,
        "target_command_sha256": envelope.command_digest(command, directory),
        "inputs": bind_inputs(closure),
        "assembly_file": assembly.name, "assembly_sha256": envelope.digest(assembly),
        "program_violations": counts,
    }
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    temporary.replace(output)
    # Ninja depfile: any header of the closure re-runs the witness.
    target = args.depfile_target or str(output)
    output.with_suffix(".d").write_text(f"{target}: " + " ".join(p.replace(" ", "\\ ") for p in closure) + "\n")


def verify(args: argparse.Namespace) -> None:
    """Validate the retained proof without invoking or requiring a compiler."""
    proof_path = Path(args.proof)
    proof = json.loads(proof_path.read_text())
    if (proof.get("schema") != 3 or proof.get("signature") != args.signature or
            proof.get("identity") != args.identity or proof.get("mode") != args.mode):
        raise ValueError("tile register proof has a foreign schema/configuration")
    for key, path in (("checker_sha256", Path(__file__)), ("envelope_sha256", _ENVELOPE),
                      ("budget_sha256", BUDGET_PATH)):
        if proof.get(key) != envelope.digest(path):
            raise ValueError(f"stale tile register tool: {path}")
    inputs = proof.get("inputs")
    if not isinstance(inputs, dict) or str(Path(args.source).resolve()) not in inputs:
        raise ValueError("tile register proof does not bind the probe's include closure")
    for path, expected in inputs.items():
        if not Path(path).is_file() or envelope.digest(Path(path)) != expected:
            raise ValueError(f"stale tile register input (production source changed): {path}")
    command, directory = envelope.target_command(args)
    if proof.get("target_command_sha256") != envelope.command_digest(command, directory):
        raise ValueError("actual target compile command changed after the register witness")
    executed = proof.get("executed_command")
    source = str(Path(args.source).resolve())
    if (not isinstance(executed, list) or proof.get("executed_directory") != directory or
            executed[:len(command)] != command or
            executed[len(command):-3] != ["-O3", "-fno-sanitize=all", "-fno-lto", "-std=c++20", "-S"] or
            str((Path(directory) / executed[-3]).resolve()) != source or executed[-2] != "-o" or
            Path(executed[-1]).name != proof.get("assembly_file")):
        raise ValueError("recorded witness invocation is not the target command over the probe")
    name = proof.get("assembly_file")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("tile register proof must name one retained local assembly artifact")
    assembly = proof_path.parent / name
    if proof.get("assembly_sha256") != envelope.digest(assembly):
        raise ValueError("tile register assembly is missing or changed")
    counts = audit(assembly.read_text())
    if proof.get("program_violations") != counts:
        raise ValueError("tile register evidence changed")
    require_registers(counts, args.identity, args.mode)


def census(args: argparse.Namespace) -> None:
    """Maintainer tool: print the per-program spill census for qualifying a budget (no policy applied)."""
    command, directory = envelope.target_command(args)
    assembly = Path(args.proof).with_suffix(".census.s")
    assembly.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command + ["-O3", "-fno-sanitize=all", "-fno-lto", "-std=c++20", "-S", args.source,
                              "-o", str(assembly)], cwd=directory, check=True)
    counts = audit(assembly.read_text())
    print(json.dumps({program_key(name): count for name, count in sorted(counts.items(), key=lambda item: program_key(item[0]))},
                     indent=1, sort_keys=True))


def main() -> None:
    """Run the requested build/verification phase; errors are fatal, never skips."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify", "census"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--proof", required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--compile-database", required=True)
    parser.add_argument("--target-source", required=True)
    parser.add_argument("--identity", required=True, help="compiler id and major version, e.g. GNU-14")
    parser.add_argument("--mode", required=True, help="native ISA envelope, e.g. NEON_I8MM")
    parser.add_argument("--depfile-target", help="depfile output path as the generator names it")
    args = parser.parse_args()
    if args.action == "census":
        census(args)
        return
    if args.action == "build":
        build(args)
    verify(args)
    count = len(json.loads(Path(args.proof).read_text())["program_violations"])
    print(f"PASS: {count} native NEON tile programs: single-row programs spill nothing, grouped programs within budget")


if __name__ == "__main__":
    main()
