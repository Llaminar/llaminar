#!/usr/bin/env python3
"""Generate one stack-traffic baseline record for the native NEON tile programs of a foreign (baseline) tree.

Maintainer tool for the develop-seeded stack-traffic table (CPUNativeTileStackTable.json). It never changes
witness semantics: it imports the committed witness checker (CPUNativeTileRegisterEvidence.py, whose SHA256 the
record binds) and runs its own build-phase building blocks - target command authentication, driver plan and
job-program identities, the preprocessing dependency closure bound before and after compiling, the inline
assembly guard, the executed compile, the B1 replay of the validated GNU plan that retains the assembler input
the bound object was assembled from, and the bound disassembly - over the baseline tree's own probe, with the
baseline tree's own compile database (same image, compiler identity, flags and envelope).

The program inventory is taken from the object and validated against the baseline tree's OWN admission: the
baseline checker file given with --admission (its ENVELOPE_PROGRAMS, DISPATCH, DOTS, TILE_RUN and TILE_PROGRAM;
hash recorded) - never the candidate tree's keys. Every expected program must exist exactly once; any other tile
symbol fails. Each program is then analysed by the committed checker (stack-traffic classes per natural loop,
loops named by the .loc locations of the retained assembly). Spill budgets and ordered-add sites are not
required of a baseline (recorded only: the FP/SIMD spill count per program).

Output: one JSON record (--out) with the tree SHA, the probe and closure hashes, the toolchain identities, the
checker and admission hashes, and per program the named loops and per-loop class counts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("cpu_native_tile_register_evidence",
                                               HERE / "CPUNativeTileRegisterEvidence.py")
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)
envelope = ev.envelope


def admission(path: Path):
    spec = importlib.util.spec_from_file_location("baseline_admission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(data: bytes, disassembly: str, mode: str, admitted) -> tuple[dict, set[str]]:
    """Tile programs of the object, validated exactly against the baseline tree's own admission."""
    if mode not in admitted.ENVELOPE_PROGRAMS:
        raise ValueError(f"baseline admission has no inventory for {mode}")
    image = ev.read_object(data)
    if ev.object_inventory(data) != [",".join(admitted.ENVELOPE_PROGRAMS[mode])]:
        raise ValueError("baseline probe inventory does not match the baseline admission")
    expected = set()
    for entry in admitted.ENVELOPE_PROGRAMS[mode]:
        rows, dot = entry.split(":")
        leaf = 1 if dot == "I8MM" and rows == "1" else admitted.DOTS[dot]
        for policy in (0, 1):
            for encoding, nibble_map, minimum in admitted.DISPATCH:
                expected.add((int(rows), leaf, policy, encoding, nibble_map, minimum, 0))
    bodies = ev.normalize_disassembly(disassembly, image)
    found: dict[tuple, str] = {}
    for symbol in sorted({s.name for s in image.symbols if s.kind != "section" and admitted.TILE_RUN.search(s.name)}):
        if symbol not in bodies:
            raise ValueError(f"tile symbol is not a function in an executable section: {symbol}")
        match = admitted.TILE_PROGRAM.fullmatch(symbol[len(image.prefix):]) if symbol.startswith(image.prefix) else None
        if not match:
            raise ValueError(f"unrecognized tile program symbol: {symbol}")
        key = tuple(int(field) for field in match.groups())
        if key in found or key not in expected:
            raise ValueError(f"duplicate or unadmitted tile program: {symbol}")
        found[key] = symbol
    if set(found) != expected:
        raise ValueError(f"baseline object misses {len(expected - set(found))} admitted programs")
    return bodies, set(found.values())


def annotate_loops(loops: list[dict], lines: dict, evidence: dict) -> None:
    """Attach each machine loop's member locations and control evidence. A helper loop (call model) is resolved
    from the role map's helper entry, never from evidence: it carries none."""
    for loop in loops:
        if "helper" in loop:
            loop.update(member_sources=[], latch_evidence=[], entry_evidence=[], empty=False, induction_evidence=[],
                        empty_source=None)
            continue
        loop["member_sources"] = lines[loop["id"]]
        loop["latch_evidence"] = evidence[loop["id"]]["latch"]
        loop["entry_evidence"] = evidence[loop["id"]]["entry"]
        loop["empty"] = evidence[loop["id"]]["empty"]
        loop["induction_evidence"] = evidence[loop["id"]]["induction"]
        loop["empty_source"] = evidence[loop["id"]]["empty_source"]


def loop_lines(analysis, where: list[str]) -> dict[int, list[str]]:
    """Per loop, the distinct .loc locations of every instruction of its blocks, nested loops included
    (instruction indices as traffic_record() numbers them: nops dropped)."""
    parents = {loop["id"]: loop["parent"] for loop in analysis.loops}
    lines: dict[int, set[str]] = {loop["id"]: set() for loop in analysis.loops}
    index = 0
    for number, block in enumerate(analysis.blocks):
        loop = analysis.innermost.get(number, 0)
        for instruction in block.instructions:
            if instruction.base == "nop":
                continue
            chain = loop
            while chain:
                lines[chain].add(where[index] if index < len(where) else "?")
                chain = parents[chain]
            index += 1
    return {loop: sorted(values) for loop, values in lines.items()}


FLAG_SETTERS = frozenset({"cmp", "cmn", "tst", "ccmp", "ccmn", "subs", "adds", "ands", "bics", "negs"})
CONDITIONAL = frozenset({"b.cond", "cbz", "cbnz", "tbz", "tbnz"})


def loop_evidence(analysis, where: list[str]) -> dict[int, dict[str, list[str]]]:
    """Per loop, the .loc locations of its control instructions - the evidence of which source loop it is:
    latch: every back-edge block's final instruction (the back-edge branch, or the last instruction before a
    fall-through into an entry) and, for a conditional branch on flags, the last flag-setting instruction of that
    block (the loop condition's compare); entry: the first instruction of every entry block (a top-tested
    loop's condition, a rotated loop's first body instruction). Instruction indices as traffic_record() numbers
    them (nops dropped)."""
    parents = {loop["id"]: loop["parent"] for loop in analysis.loops}

    def chain_of(identity: int) -> set[int]:
        found = set()
        while identity:
            identity = parents.get(identity, 0)
            found.add(identity)
        return found

    def on_every_cycle(step: int, loop: dict, body: set[int]) -> bool:
        """Every iteration of the loop passes `step`: no path inside the loop body (nested loops included) leads from
        one of its entries back to an entry without crossing the step block. For a single-entry loop this is the
        step dominating every latch; for a loop GCC enters at several blocks it is the same property over every
        entry."""
        entries = set(loop["entries"])
        for start in entries - {step}:
            seen, work = set(), [start]
            while work:
                node = work.pop()
                if node in seen:
                    continue
                seen.add(node)
                for successor in analysis.blocks[node].successors:
                    if successor < 0 or successor == step or successor not in body:
                        continue
                    if successor in entries:
                        return False
                    work.append(successor)
        return True

    def dominates(step: int, latch: int, loop: dict, body: set[int]) -> bool:
        if step == latch:
            return True
        seen, work = set(), [entry for entry in loop["entries"] if entry != step]
        while work:
            node = work.pop()
            if node in seen or node not in body:
                continue
            if node == latch:
                return False
            seen.add(node)
            work += [successor for successor in analysis.blocks[node].successors if successor != step and successor >= 0]
        return True
    positions, index = {}, 0
    for number, block in enumerate(analysis.blocks):
        real = []
        for instruction in block.instructions:
            if instruction.base != "nop":
                real.append((index, instruction))
                index += 1
        positions[number] = real

    def loc(position: int) -> str:
        return where[position] if position < len(where) else "?"
    result = {}
    for loop in analysis.loops:
        latch, entry = set(), set()
        for number in loop["latches"]:
            block = positions[number]
            if not block:
                continue
            last_index, last = block[-1]
            latch.add(loc(last_index))
            if last.base in CONDITIONAL and last.base == "b.cond":
                setter = next((i for i, instruction in reversed(block[:-1]) if instruction.base in FLAG_SETTERS), None)
                if setter is not None:
                    latch.add(loc(setter))
        for number in loop["entries"]:
            if positions[number]:
                entry.add(loc(positions[number][0][0]))
        own_blocks = [number for number in range(len(analysis.blocks)) if analysis.innermost.get(number) == loop["id"]]
        members = [instruction for number in own_blocks for _, instruction in positions[number]]
        # C1 - an empty self-loop: exactly one block holding exactly one instruction, an unconditional branch whose
        # only successor is that block (its own address: no exit edge, so the path never terminates), and no nested
        # loop. It gets no role (table() requires zero counts and it never covers a source loop).
        empty = (len(own_blocks) == 1 and len(members) == 1 and members[0].base == "b" and
                 analysis.blocks[own_blocks[0]].successors == [own_blocks[0]] and
                 not any(other["parent"] == loop["id"] for other in analysis.loops))
        body = {number for number in range(len(analysis.blocks)) if analysis.innermost.get(number) is not None and
                (analysis.innermost[number] == loop["id"] or loop["id"] in chain_of(analysis.innermost[number]))}
        body_instructions = [instruction for number in sorted(body) for _, instruction in positions[number]]
        invariant = ev.loop_invariant_registers(body_instructions)
        induction = set()
        for number in loop["latches"]:
            block = positions[number]
            if not block or block[-1][1].base != "b.cond":
                continue
            setter = next((instruction for _, instruction in reversed(block[:-1]) if instruction.base in FLAG_SETTERS),
                          None)
            if setter is None:
                continue
            reads = setter.operands[(0 if setter.base in ("cmp", "cmn", "tst", "ccmp", "ccmn") else 1):]
            for operand in reads:
                register = ev.gp(operand) or ev.gp_view(operand)
                if register is None:
                    continue
                # C2 dataflow (the header-extent condition (i) is checked by the table on the step's .loc):
                # (iii) the register's ONLY definition across the loop, nested loops included, is a step
                #       `r = r +/- #imm` (so (iv): the compare reads the stepped value, never a redefinition);
                definitions = [(block_number, i, instruction) for block_number in body
                               for i, instruction in positions[block_number]
                               if register in ev.defined_registers(instruction)]
                if len(definitions) != 1:
                    continue
                step_block, i, instruction = definitions[0]
                operands = instruction.operands
                if not (instruction.base in ("add", "sub", "adds", "subs") and len(operands) == 3 and
                        ev.immediate(operands[2]) is not None and
                        (ev.gp(operands[1]) or ev.gp_view(operands[1])) == register):
                    continue
                # (ii) the step lies on every iteration (dominates every latch; over every entry of a multi-entry loop)
                #      and in the loop's OWN innermost blocks, never a nested loop's. With (iii) - the sole definition
                #      across the loop including nested loops - it therefore runs exactly once per iteration.
                if step_block not in own_blocks or not on_every_cycle(step_block, loop, body):
                    continue
                # (v) every other operand the compare reads is an immediate (or shift amount) or loop invariant.
                others = [other for other in reads if other is not operand]
                if not all(ev.immediate(other) is not None or ev.SHIFT.match(other) or
                           (ev.gp(other) or ev.gp_view(other)) in invariant for other in others):
                    continue
                induction.add(loc(i))
        result[loop["id"]] = {"latch": sorted(latch), "entry": sorted(entry), "empty": empty,
                              "induction": sorted(induction),
                              "empty_source": loc(positions[own_blocks[0]][0][0]) if empty else None}
    return result


def generate(args: argparse.Namespace) -> dict:
    ev.require_absolute_source(args.source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    obj, dependencies = stem.with_suffix(".o"), stem.with_suffix(".deps")
    preprocessed, unit, assembly = stem.with_suffix(".pre.deps"), stem.with_suffix(".i"), stem.with_suffix(".asm")
    for stale in (out, obj, dependencies, preprocessed, unit, assembly):
        stale.unlink(missing_ok=True)
    command, directory = envelope.target_command(args)
    driver = ev.driver_identity(command)
    version = subprocess.check_output([command[0], "--version"], cwd=directory, text=True)
    identity = ev.compiler_identity(version)
    if identity != args.identity:
        raise ValueError(f"compiler reports {identity}, expected {args.identity}")
    executed = ev.executed_command(command, args.source, obj, dependencies)
    plan = ev.driver_plan(executed, directory, identity, ev.JobPaths(args.source, str(obj), str(dependencies),
                                                                     directory, None))
    programs = ev.program_identities(plan, command)
    if programs["driver"] != driver:
        raise ValueError("the compiler driver changed while its plan was observed")
    subprocess.run(command + ev.WITNESS_FLAGS + ["-M", "-MF", str(preprocessed), "-MT", "probe", args.source],
                   cwd=directory, check=True, stdout=subprocess.DEVNULL)
    before = ev.bind_inputs(ev.read_depfile(preprocessed, directory))
    paths = ev.JobPaths(args.source, str(obj), str(dependencies), directory, frozenset(before))
    ev.validate_plan(plan["jobs"], identity, paths)
    subprocess.run(command + ev.WITNESS_FLAGS + ["-E", args.source, "-o", str(unit)], cwd=directory, check=True)
    if ev.inline_assembly_violations(unit.read_text(errors="replace")):
        raise ValueError("baseline translation unit holds unproved inline assembly")
    compiled = subprocess.run(executed, cwd=directory, capture_output=True, text=True)
    if compiled.returncode or ev.precompiled_headers(compiled.stderr):
        raise ValueError(f"baseline compile failed or read precompiled headers: {compiled.stderr[-2000:]}")
    closure = ev.read_depfile(dependencies, directory)
    if closure != sorted(before) or ev.bind_inputs(closure) != before:
        raise ValueError("baseline dependency closure differs from preprocessing or changed while compiling")
    if ev.program_identities(plan, command) != programs or \
            ev.plan_shape(ev.driver_plan(executed, directory, identity, paths)) != ev.plan_shape(plan):
        raise ValueError("a compiler/assembler program or the driver plan changed while the baseline compiled")
    ev.replay_plan(plan, directory, obj, dependencies, assembly)
    if ev.bind_inputs(closure) != before or ev.program_identities(plan, command) != programs:
        raise ValueError("an input or program changed during the replay")
    disassembly, disassembly_command, tool_version = ev.disassemble(args.disassembler, obj)
    stem.with_suffix(".dis").write_text(disassembly)
    admitted = admission(Path(args.admission))
    bodies, symbols = inventory(obj.read_bytes(), disassembly, args.mode, admitted)
    image = ev.read_object(obj.read_bytes())
    noreturn = ev.noreturn_symbols(bodies, ev.GENERIC, ev.trusted_imports(image), image.prefix)
    locations: dict[str, list[str]] = {}
    ev.assembly_functions(assembly.read_text(errors="replace"), set(bodies), image.macho, locations)
    traffic, spills, structural, members = {}, {}, [], {}
    model, unit = ev.call_model(image, bodies)  # The same call model as the candidate witness (audit()).
    for symbol in sorted(symbols):
        analysis = ev.analysed(symbol, bodies, ev.GENERIC, noreturn, model, unit)
        key = ev.program_key(symbol)
        structural += [f"{key}: {error}" for error in analysis.errors]
        traffic[key] = ev.traffic_record(analysis)
        spills[key] = analysis.spills
        members[key] = (loop_lines(analysis, locations.get(symbol, [])),
                        loop_evidence(analysis, locations.get(symbol, [])))
    if structural:
        raise ValueError(f"baseline programs are structurally unprovable ({len(structural)}): {structural[:4]}")
    named = ev.name_loops(traffic, obj.read_bytes(), disassembly, assembly.read_text(errors="replace"))
    for key, record in named.items():
        annotate_loops(record["loops"], *members[key])
    tree_inputs = {path: value for path, value in before.items() if path.startswith(args.tree_root.rstrip("/") + "/")}
    return {
        "schema": 1, "tree_sha": args.tree_sha, "tree_root": args.tree_root, "mode": args.mode,
        "image": args.image, "identity": identity, "compiler_version": version,
        "checker_sha256": envelope.digest(HERE / "CPUNativeTileRegisterEvidence.py"),
        "generator_sha256": envelope.digest(Path(__file__)),
        "admission_sha256": envelope.digest(Path(args.admission)),
        "probe_sha256": envelope.digest(Path(args.source)),
        "closure_sha256": envelope.command_digest(sorted(f"{p}={v['sha256']}" for p, v in before.items()
                                                         if isinstance(v, dict) and "sha256" in v), ""),
        "tree_inputs": {path[len(args.tree_root.rstrip('/')) + 1:]: value.get("sha256") for path, value
                        in sorted(tree_inputs.items())},
        "executed_command": executed, "target_command_sha256": envelope.command_digest(command, directory),
        "programs": programs, "driver_plan_shape": ev.plan_shape(plan),
        "disassembler_version": tool_version, "disassembly_command": disassembly_command,
        "object_sha256": envelope.digest(obj), "assembly_sha256": envelope.digest(assembly),
        "fp_spills": spills, "stack_traffic": named,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source", "compiler", "compile-database", "target-source", "identity", "mode", "disassembler",
                   "admission", "tree-sha", "tree-root", "image", "out"):
        parser.add_argument(f"--{option}", required=True)
    parser.add_argument("--signature", default="baseline")
    args = parser.parse_args()
    record = generate(args)
    Path(args.out).write_text(json.dumps(record, sort_keys=True, indent=1) + "\n")
    print(f"baseline {args.mode}: {len(record['stack_traffic'])} programs, checker {record['checker_sha256'][:12]}")


if __name__ == "__main__":
    main()
