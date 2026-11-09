#!/usr/bin/env python3
"""Build and authenticate a register-allocation witness for the native NEON projection tiles.

Build phase: compile CPUNativeNEONTileRegisterProbe.cpp with the real target's
ordered compile command at -O3 and audit the assembly.

- Inventory: the expected tile K-loop programs (row tile x integer family x
  policy x encoding/map/minimum) derive from the admitted ISA envelope; the
  probe's inventory string must agree. Every expected program must be emitted
  exactly once with real instructions; any other tile program symbol fails.
- Analysis: each program and its call closure is analysed over its control-flow
  graph. Stack provenance of every general register (sp, the frame pointer and
  anything computed from them, including indexed/shifted forms, stack slots
  holding stack addresses and callee arguments) is a forward data-flow fact;
  any register definition not proven to be a constant stack offset invalidates
  or generalises it, and a path-dependent (joined) stack/non-stack base fails.
- Spills: each load/store moving FP/SIMD data (any width) through a stack
  address counts once, except proven ABI saves/restores of d8-d15: saved once
  before any clobber to a fixed slot, slot untouched while live, restored once
  from that slot, never clobbered after the restore, restored before return.
- Structural failures are fatal and never paid from a budget: indirect
  branches, external or unclassifiable callees (the only exemptions are abort
  and the exact fatal-diagnostic symbol, itself proven never to return),
  unclassifiable instructions/addressing, ambiguous stack bases, stack-address
  escapes, unreachable code, unproven ABI saves.
- Policy: single-row (decode) programs must not spill at all; grouped row tiles
  may not exceed the committed per-program budget for the exact compiler
  identity and ISA envelope (CPUNativeTileRegisterBudget.json, a ratchet). The
  identity is derived from the compiler's own version output and must equal the
  configured one; unknown identities/envelopes fail closed.
- Dependencies: the dependency closure is the one the compiler reports for this
  exact compilation (-MD on the -S run), cross-checked against a preprocessing
  pass with the same flags; every input is hashed before and after compiling and
  bound with its logical path, resolved path and symlink chain.

Verify phase (Unit/preflight): no compiler is invoked, but every recorded input
(including absolute system/SDK headers named by the compiler) must be present
and unchanged, so verification is tied to the build host's toolchain; it is not
an SDK-independent or installed-runner qualification. The actual target command
must match and the retained assembly must re-audit to the recorded counts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
INVENTORY = re.compile(r"llaminar-native-neon-tile-probe:([A-Za-z0-9,:]*)")
# (encoding, nibble map, with-minimum) dispatched per policy by Tile::dispatch().
DISPATCH = ((1, 0, 1), (1, 0, 0), (5, 0, 0), (2, 0, 0), (4, 2, 0), (0, 0, 0), (0, 2, 0), (0, 1, 1))
DOTS = {"ASIMD": 0, "DotProd": 1, "I8MM": 2}
# The one never-returning diagnostic a tile may call, and abort. Both must also be proven never to return.
FATAL_DIAGNOSTIC = "_ZN9llaminar23cpu11native_quantised23CPUNativeQuantisedWeightView32rejectOutsideAdmittedPublicationEv"
NORETURN_LEAVES = {"abort"}

LABEL = re.compile(r"^([A-Za-z_.$][\w.$]*):(.*)$")
CONDITIONS = "eq|ne|cs|hs|cc|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le|al|nv"
CONDITIONAL_BRANCH = re.compile(rf"^b\.?(?:{CONDITIONS})$")
INDIRECT = re.compile(r"^(?:br|blr)(?:aa|ab|aaz|abz)?$")
RETURN = re.compile(r"^ret(?:aa|ab)?$")
BRANCHES = ("b", "bl", "b.cond", "cbz", "cbnz", "tbz", "tbnz")
LOAD = re.compile(r"^(?:ldr|ldur|ldp|ldnp|ldpsw|ldrs?[bhw]|ldurs?[bhw]|ld[1-4]r?|ldar[bh]?|ldapr[bh]?|ldaxr[bh]?|ldxr[bh]?|ldaxp|ldxp)$")
STORE = re.compile(r"^(?:str|stur|stp|stnp|str[bh]|stur[bh]|st[1-4]|stlr[bh]?|stlxr[bh]?|stxr[bh]?|stlxp|stxp)$")
PREFETCH = re.compile(r"^prfu?m$")
NO_DESTINATION = {"cmp", "cmn", "tst", "ccmp", "ccmn", "fcmp", "fcmpe", "fccmp", "fccmpe", "nop", "hint", "dmb",
                  "dsb", "isb", "yield", "bti", "csdb", "brk", "hlt", "udf", "msr", "paciasp", "autiasp",
                  "pacibsp", "autibsp", "xpaclri"}
GP64 = re.compile(r"^(?:x([12]?[0-9]|30)|sp|fp|lr|xzr)$")
GP32 = re.compile(r"^(?:w([12]?[0-9]|30)|wsp|wzr)$")
FP = re.compile(r"^([bhsdqv])([12]?[0-9]|3[01])(?:\.(\d*)([bhsd]))?(?:\[\d+\])?$")
IMMEDIATE = re.compile(r"^#?(-?(?:0x[0-9a-fA-F]+|\d+))$")
SHIFT = re.compile(r"^lsl\s*#?(\d+)$")
RELOCATION = re.compile(r"^(?:#?:[a-z0-9_]+:[\w.$+-]+|[\w.$]+@\w+)$")
INDEX = re.compile(r"^([xw](?:[12]?[0-9]|30))(?:\s*,\s*(?:lsl|uxtw|sxtw|sxtx|uxtx)(?:\s*#?\d+)?)?$")
MEMORY = re.compile(r"^\[\s*(sp|fp|x(?:[12]?[0-9]|30))\s*(?:,\s*(.*?))?\s*\](!?)$")
SYMBOL = re.compile(r"^[A-Za-z_.$][\w.$@]*$")
WIDTH = {"b": 1, "h": 2, "s": 4, "d": 8, "q": 16}
CALLEE_SAVED = range(8, 16)
CALLER_CLOBBERED = [f"x{n}" for n in range(19)] + ["x30"]
NOT_STACK = frozenset({"N"})
WIDEN_AFTER = 4


class Dialect:
    """Assembler syntax of one witness: Mach-O (Apple) or ELF (GNU)."""

    def __init__(self, assembly: str) -> None:
        self.macho = ".subsections_via_symbols" in assembly or "__TEXT," in assembly

    def local(self, label: str) -> bool:
        """Assembler-local labels stay inside the enclosing function."""
        return label[:1] in ("L", "l") if self.macho else label.startswith(".L")

    def symbol(self, name: str) -> str:
        """Source-level symbol name (Mach-O prefixes C symbols with an underscore)."""
        return name[1:] if self.macho and name.startswith("_") else name

    def statements(self, line: str) -> list[str]:
        """Strip comments and split one source line into statements."""
        if self.macho:
            return [line.split(";", 1)[0].split("//", 1)[0]]
        line = line.split("//", 1)[0]
        if line.lstrip().startswith("#"):
            return []
        return line.split(";")


def functions(assembly: str) -> dict[str, list[str]]:
    """Split compiler -S output into global function bodies; local labels stay inside."""
    dialect = Dialect(assembly)
    bodies: dict[str, list[str]] = {}
    current = None
    for line in assembly.splitlines():
        label = LABEL.match(line)
        if label and not dialect.local(label.group(1)):
            current = label.group(1)
            if current in bodies:
                raise ValueError(f"symbol defined twice in the witness: {current}")
            bodies[current] = []
            continue
        if current is not None:
            bodies[current].append(line)
    return bodies


def split_operands(text: str) -> list[str]:
    """Split instruction operands on top-level commas (not inside [] or {})."""
    operands, depth, current = [], 0, []
    for character in text:
        if character in "[{":
            depth += 1
        elif character in "]}":
            depth -= 1
        if character == "," and depth == 0:
            operands.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if "".join(current).strip():
        operands.append("".join(current).strip())
    return operands


def gp(operand: str) -> str | None:
    """Canonical 64-bit general register name of an x/sp operand (fp=x29, lr=x30)."""
    if not GP64.match(operand):
        return None
    return {"fp": "x29", "lr": "x30"}.get(operand, operand)


def immediate(text: str) -> int | None:
    match = IMMEDIATE.match(text.strip())
    return int(match.group(1), 0) if match else None


def fp_registers(operand: str) -> list[tuple[int, int]] | None:
    """(register number, byte width) of an FP/SIMD operand or register list; None if not FP/SIMD."""
    operand = re.sub(r"\[\d+\]$", "", operand.strip())
    if operand.startswith("{"):
        inner = operand[1:operand.rindex("}")]
        result = []
        for element in (part.strip() for part in inner.split(",")):
            bounds = [part.strip() for part in element.split("-")]
            parsed = [FP.match(bound) for bound in bounds]
            if not all(parsed) or len(parsed) > 2:
                raise ValueError(f"unclassifiable register list: {operand}")
            first, last = int(parsed[0].group(2)), int(parsed[-1].group(2))
            width = _fp_width(parsed[0])
            number = first
            while True:
                result.append((number, width))
                if number == last:
                    break
                number = (number + 1) % 32
        return result
    match = FP.match(operand)
    return [(int(match.group(2)), _fp_width(match))] if match else None


def _fp_width(match: re.Match) -> int:
    if match.group(1) != "v":
        return WIDTH[match.group(1)]
    if match.group(4):
        return int(match.group(3) or 1) * WIDTH[match.group(4)]
    return 16


class Instruction:
    """One parsed machine instruction."""

    def __init__(self, text: str, dialect: Dialect) -> None:
        self.text = text.strip()
        parts = self.text.split(None, 1)
        mnemonic = parts[0].lower()
        self.operands = split_operands(parts[1]) if len(parts) > 1 else []
        if CONDITIONAL_BRANCH.match(mnemonic):
            self.base = "b.cond"
        else:
            self.base = mnemonic.split(".", 1)[0] if dialect.macho else mnemonic
        self.memory = next((index for index, operand in enumerate(self.operands) if operand.startswith("[")), None)


class Block:
    def __init__(self) -> None:
        self.labels: list[str] = []
        self.instructions: list[Instruction] = []
        self.successors: list[int] = []


def build_cfg(body: list[str], dialect: Dialect) -> tuple[list[Block], list[str]]:
    """Basic blocks of one function body; instructions outside executable sections are rejected."""
    blocks, errors, executable = [Block()], [], True
    for line in body:
        label = LABEL.match(line)
        statements = dialect.statements(label.group(2) if label else line)
        if label:
            if blocks[-1].instructions:
                blocks.append(Block())
            blocks[-1].labels.append(label.group(1))
        for statement in statements:
            statement = statement.strip()
            if not statement:
                continue
            if statement.startswith("."):
                directive = statement.split(None, 1)
                if directive[0] in (".text",):
                    executable = True
                elif directive[0] in (".data", ".bss", ".rodata"):
                    executable = False
                elif directive[0] == ".section":
                    section = directive[1] if len(directive) > 1 else ""
                    executable = (section.startswith(".text") or section.startswith("__TEXT,__text") or
                                  bool(re.search(r",\s*\"[^\"]*x[^\"]*\"", section)))
                elif directive[0] in (".inst", ".word", ".long", ".4byte") and executable:
                    errors.append(f"raw data or encoded instruction in code: {statement}")
                continue
            if not executable:
                errors.append(f"instruction outside an executable section: {statement}")
                continue
            instruction = Instruction(statement, dialect)
            blocks[-1].instructions.append(instruction)
            if instruction.base in BRANCHES or INDIRECT.match(instruction.base) or RETURN.match(instruction.base):
                blocks.append(Block())  # Every branch ends its block, so each edge carries the branch-point state.
    return blocks, errors


class State:
    """Per-program-point facts: general register stack provenance, slots holding stack addresses, ABI saves."""

    def __init__(self, registers: dict[str, frozenset], tainted: frozenset, saved: dict[int, frozenset]) -> None:
        self.registers, self.tainted, self.saved = registers, tainted, saved

    @staticmethod
    def entry(arguments: frozenset = frozenset()) -> "State":
        """Function entry: sp at offset 0, x29 the caller's frame; arguments holding caller stack addresses."""
        registers = {"sp": frozenset({("S", 0)}), "x29": frozenset({("S", None)})}
        for register, kind in arguments:
            registers[register] = frozenset({("S", None)}) if kind == "S" else frozenset({"N", ("S", None)})
        return State(registers, frozenset(), {n: frozenset({("U", None)}) for n in CALLEE_SAVED})

    def copy(self) -> "State":
        return State(dict(self.registers), self.tainted, dict(self.saved))

    def value(self, register: str) -> frozenset:
        return self.registers.get(register, NOT_STACK) if register != "xzr" else NOT_STACK

    def assign(self, register: str, values: frozenset) -> None:
        if register == "xzr":
            return
        if values == NOT_STACK:
            self.registers.pop(register, None)
        else:
            self.registers[register] = widen(values)

    def join(self, other: "State") -> "State":
        registers = {}
        for register in set(self.registers) | set(other.registers):
            registers[register] = widen(self.value(register) | other.value(register))
        saved = {n: self.saved[n] | other.saved[n] for n in CALLEE_SAVED}
        return State(registers, self.tainted | other.tainted, saved)

    def key(self) -> tuple:
        return (tuple(sorted((r, tuple(sorted(v, key=repr))) for r, v in self.registers.items())), self.tainted,
                tuple(tuple(sorted(self.saved[n], key=repr)) for n in CALLEE_SAVED))


def widen(values: frozenset) -> frozenset:
    """Keep the provenance lattice finite: many or unknown stack offsets collapse to one unknown offset."""
    stack = [value for value in values if value != "N"]
    if ("S", None) in values or len(stack) > WIDEN_AFTER:
        return frozenset({value for value in values if value == "N"} | {("S", None)})
    return values


def provenance(values: frozenset) -> str:
    """N: never a stack address; S: always a stack address; A: path-dependent (ambiguous)."""
    if values == NOT_STACK:
        return "N"
    return "S" if "N" not in values else "A"


def single_offset(values: frozenset) -> int | None:
    return next(iter(values))[1] if len(values) == 1 and "N" not in values else None


class Analysis:
    """Control-flow-aware audit of one function: counts FP/SIMD spills, records structural failures."""

    def __init__(self, name: str, body: list[str], dialect: Dialect, bodies: dict[str, list[str]],
                 noreturn: set[str], arguments: frozenset = frozenset()) -> None:
        self.name, self.dialect, self.bodies, self.noreturn = name, dialect, bodies, noreturn
        self.arguments = arguments
        self.blocks, self.errors = build_cfg(body, dialect)
        self.spills = 0
        self.callees: set[tuple[str, frozenset]] = set()
        self.report = False
        self.labels = {label: index for index, block in enumerate(self.blocks) for label in block.labels}

    def fail(self, instruction: Instruction | None, message: str) -> None:
        if self.report:
            where = f" at '{instruction.text}'" if instruction else ""
            self.errors.append(f"{message}{where}")

    # Control flow --------------------------------------------------------------------------------------------
    def classify_branch(self, instruction: Instruction) -> tuple[str, str | None]:
        """('fall'|'branch'|'jump'|'return'|'call'|'tail'|'noreturn'|'indirect', target)."""
        base = instruction.base
        if RETURN.match(base):
            return "return", None
        if INDIRECT.match(base):
            return "indirect", None
        if base in BRANCHES:
            target = instruction.operands[-1] if instruction.operands else ""
            if not SYMBOL.match(target):
                return "indirect", None
            if base == "bl":
                return ("noreturn" if self.is_fatal(target) else "call"), target
            local = target in self.labels
            if not local and self.dialect.local(target):
                self.fail(instruction, "branch to a local label outside the function")
                return "indirect", None
            if base == "b":
                if local:
                    return "jump", target
                return ("noreturn" if self.is_fatal(target) else "tail"), target
            return ("branch" if local else "conditional-tail"), target
        return "fall", None

    def is_fatal(self, symbol: str) -> bool:
        return symbol in self.noreturn

    def link(self) -> None:
        for index, block in enumerate(self.blocks):
            falls = True
            for instruction in block.instructions:
                kind, target = self.classify_branch(instruction)
                if kind in ("jump", "branch"):
                    block.successors.append(self.labels[target])
                if kind in ("jump", "return", "tail", "noreturn", "indirect"):
                    falls = False
            if falls:
                block.successors.append(index + 1 if index + 1 < len(self.blocks) else -1)

    # Data flow ---------------------------------------------------------------------------------------------------
    def run(self) -> None:
        self.link()
        states: dict[int, State] = {0: State.entry(self.arguments)}
        pending = [0]
        while pending:
            index = pending.pop()
            state = self.transfer(index, states[index].copy())
            if state is None:
                continue
            for successor in self.blocks[index].successors:
                if successor < 0:
                    continue
                merged = state if successor not in states else states[successor].join(state)
                if successor not in states or merged.key() != states[successor].key():
                    states[successor] = merged
                    pending.append(successor)
        self.report = True
        for index, block in enumerate(self.blocks):
            if index not in states:
                if block.instructions:
                    self.errors.append(f"unreachable instructions cannot be classified: '{block.instructions[0].text}'")
                continue
            self.transfer(index, states[index].copy())
            if -1 in block.successors:
                self.errors.append("control falls off the end of the function")

    def transfer(self, index: int, state: State) -> State | None:
        for instruction in self.blocks[index].instructions:
            if not self.step(instruction, state):
                return None
        return state

    def step(self, instruction: Instruction, state: State) -> bool:
        """Apply one instruction; False when control leaves the function."""
        kind, target = self.classify_branch(instruction)
        if kind == "indirect":
            self.fail(instruction, "indirect or uninspectable branch")
            return False
        if kind == "noreturn":
            return False
        if kind in ("return", "tail", "conditional-tail"):
            self.leave(instruction, state)
            if kind != "return":
                self.callee(instruction, target, state)
            return kind == "conditional-tail"
        if kind == "call":
            self.callee(instruction, target, state)
            for register in CALLER_CLOBBERED:
                state.assign(register, NOT_STACK)
            return True
        if kind in ("jump", "branch"):
            return True
        if instruction.memory is not None:
            self.access(instruction, state)
        elif LOAD.match(instruction.base) or STORE.match(instruction.base) or PREFETCH.match(instruction.base):
            self.literal(instruction, state)
        elif any(operand.startswith("[") or "[" in operand.split("]")[-1] for operand in instruction.operands):
            self.fail(instruction, "unclassifiable memory operand")
        else:
            self.compute(instruction, state)
        return True

    def callee(self, instruction: Instruction, target: str, state: State) -> None:
        """Audit a callee with the caller's stack provenance of its argument registers (interprocedural)."""
        arguments = frozenset((f"x{n}", provenance(state.value(f"x{n}"))) for n in range(8)
                              if provenance(state.value(f"x{n}")) != "N")
        if target in self.bodies and not self.dialect.local(target):
            if self.report:
                self.callees.add((target, arguments))
        else:
            self.fail(instruction, f"external callee cannot be audited: {target}")

    def leave(self, instruction: Instruction, state: State) -> None:
        for n in CALLEE_SAVED:
            if any(status == "D" for status, _ in state.saved[n]):
                self.fail(instruction, f"returns without restoring callee-saved d{n}")

    def clobber(self, instruction: Instruction, state: State, number: int) -> None:
        if number not in CALLEE_SAVED:
            return
        result = set()
        for status, slot in state.saved[number]:
            if status == "U":
                self.fail(instruction, f"clobbers callee-saved v{number} before saving it")
                result.add((status, slot))
                continue
            if status == "R":
                self.fail(instruction, f"clobbers callee-saved v{number} after its epilogue restore")
            result.add(("D", slot))
        state.saved[number] = frozenset(result)

    def literal(self, instruction: Instruction, state: State) -> None:
        if PREFETCH.match(instruction.base):
            return
        if not instruction.operands or not SYMBOL.match(instruction.operands[-1].lstrip("=")):
            self.fail(instruction, "unclassifiable memory instruction")
            return
        if LOAD.match(instruction.base):
            for operand in instruction.operands[:-1]:
                self.loaded(instruction, state, operand, False, None)

    def loaded(self, instruction: Instruction, state: State, operand: str, stack: bool, slot: int | None) -> None:
        register = gp(operand)
        if register:
            if stack and slot is not None:
                state.assign(register, frozenset({("S", None)}) if slot in state.tainted else NOT_STACK)
            elif stack and state.tainted:
                state.assign(register, frozenset({"N", ("S", None)}))
            else:
                state.assign(register, NOT_STACK)
            return
        if GP32.match(operand):
            state.assign("x" + operand[1:] if operand[1:].isdigit() else "xzr", NOT_STACK)
            return
        registers = fp_registers(operand)
        if registers is None:
            self.fail(instruction, f"unclassifiable load destination {operand}")
            return
        for number, _ in registers:
            self.clobber(instruction, state, number)

    def access(self, instruction: Instruction, state: State) -> None:
        """Loads/stores: classify FP/SIMD stack traffic, ABI saves/restores, and stack-address escapes."""
        base_name = instruction.base
        load, store = LOAD.match(base_name), STORE.match(base_name)
        if not (load or store or PREFETCH.match(base_name)):
            self.fail(instruction, "unclassifiable memory instruction")
            return
        memory = MEMORY.match(instruction.operands[instruction.memory])
        if not memory:
            self.fail(instruction, "unclassifiable addressing mode")
            return
        base = gp(memory.group(1))
        offset_text, writeback = memory.group(2), memory.group(3) == "!"
        post = instruction.operands[instruction.memory + 1:]
        if len(post) > 1:
            self.fail(instruction, "unclassifiable post-index form")
            return
        displacement: int | None = 0
        if offset_text:
            if immediate(offset_text) is not None:
                displacement = immediate(offset_text)
            elif RELOCATION.match(offset_text):
                displacement = None
                if provenance(state.value(base)) != "N":
                    self.fail(instruction, "relocated offset from a stack base")
            elif INDEX.match(offset_text):
                displacement = None
                index = INDEX.match(offset_text).group(1)
                if index.startswith("x") and provenance(state.value(index)) != "N":
                    self.fail(instruction, "stack address used as an index register")
            else:
                self.fail(instruction, "unclassifiable addressing offset")
                return
        if post and (writeback or offset_text):
            self.fail(instruction, "unclassifiable post-index form")
            return
        base_values = state.value(base)
        kind = provenance(base_values)
        if kind == "A":
            self.fail(instruction, "base register is a stack address on only some paths")
            return
        stack = kind == "S"
        origin = single_offset(base_values)
        address = None if origin is None or displacement is None else origin + (0 if post else displacement)
        data = instruction.operands[:instruction.memory]
        if PREFETCH.match(base_name):
            data = []
        if base_name.startswith(("stxr", "stlxr")) and data:
            data = data[1:]  # Exclusive-store status register.
        if store and stack:
            self.protect_slots(instruction, state, address, data)
        spilled = False
        position = address
        for operand in data:
            registers = fp_registers(operand)
            general = gp(operand) or (operand if GP32.match(operand) else None)
            if registers is None and general is None:
                self.fail(instruction, f"unclassifiable data operand {operand}")
                return
            if general:
                width = 8 if gp(operand) else 4
                if store and gp(operand) and provenance(state.value(gp(operand))) != "N":
                    if stack and position is not None:
                        state.tainted = state.tainted | {position}
                    else:
                        self.fail(instruction, "a stack address is stored where it cannot be tracked")
                elif store and stack and position is not None and gp(operand):
                    state.tainted = state.tainted - {position}
                if load:
                    self.loaded(instruction, state, operand, stack, position)
                position = None if position is None else position + width
                continue
            for number, width in registers:
                if stack:
                    if store and self.save(state, number, width, position):
                        pass
                    elif load and self.restore(state, number, width, position):
                        pass
                    else:
                        spilled = True
                        if load:
                            self.clobber(instruction, state, number)
                elif load:
                    self.clobber(instruction, state, number)
                position = None if position is None else position + width
        if spilled and self.report:
            self.spills += 1
        if writeback or post:
            step = displacement if writeback else immediate(post[0])
            if step is None:
                state.assign(base, frozenset({("S", None)}) if stack else NOT_STACK)
            else:
                state.assign(base, frozenset(value if value == "N" or value[1] is None else ("S", value[1] + step)
                                             for value in base_values))

    def protect_slots(self, instruction: Instruction, state: State, address: int | None, data: list[str]) -> None:
        """A live ABI save slot (saved, not yet restored) must not be overwritten before its restore."""
        live = [(n, slot) for n in CALLEE_SAVED for status, slot in state.saved[n]
                if status in ("S", "D") and slot is not None]
        if not live:
            return
        if address is None:
            self.fail(instruction, "stack store at an unknown offset while callee-saved slots are live")
            return
        size = 0
        for operand in data:
            registers = fp_registers(operand)
            size += sum(width for _, width in registers) if registers else (8 if gp(operand) else 4)
        for number, (start, width) in live:
            if address < start + width and start < address + size:
                self.fail(instruction, f"overwrites the live save slot of d{number}")

    def save(self, state: State, number: int, width: int, slot: int | None) -> bool:
        """An ABI save stores an unsaved, unclobbered callee-saved d8-d15 (or its q view) to a fixed frame slot."""
        if number not in CALLEE_SAVED or width < 8 or slot is None or state.saved[number] != frozenset({("U", None)}):
            return False
        state.saved[number] = frozenset({("S", (slot, width))})
        return True

    def restore(self, state: State, number: int, width: int, slot: int | None) -> bool:
        """An epilogue restore reloads a saved register from its own save slot, exactly once."""
        current = state.saved.get(number) if number in CALLEE_SAVED else None
        if not current or slot is None or any(status not in ("S", "D") or saved != (slot, width)
                                              for status, saved in current):
            return False
        state.saved[number] = frozenset({("R", (slot, width))})
        return True

    def compute(self, instruction: Instruction, state: State) -> None:
        """Register-to-register instructions: propagate stack provenance; FP writes clobber d8-d15."""
        if instruction.base in NO_DESTINATION or not instruction.operands:
            return
        destination, sources = instruction.operands[0], instruction.operands[1:]
        source_registers = [gp(operand) for operand in sources if gp(operand)]
        for operand in sources:
            for token in re.findall(r"\b(?:x(?:[12]?[0-9]|30)|sp|fp|lr)\b", operand):
                if not gp(operand) and provenance(state.value(gp(token))) != "N":
                    self.fail(instruction, "stack address in an unclassifiable operand")
        values = [state.value(register) for register in source_registers]
        stack_sources = [value for value in values if provenance(value) != "N"]
        registers = fp_registers(destination)
        if registers is not None:
            if stack_sources:
                self.fail(instruction, "a stack address moves into an FP/SIMD register")
            for number, _ in registers:
                self.clobber(instruction, state, number)
            return
        target = gp(destination)
        if target is None:
            if not GP32.match(destination):
                self.fail(instruction, f"unclassifiable destination {destination}")
            elif destination[1:].isdigit():
                state.assign("x" + destination[1:], NOT_STACK)
            return
        if not stack_sources:
            state.assign(target, NOT_STACK)
            return
        base = instruction.base
        if base == "mov" and len(sources) == 1:
            state.assign(target, values[0])
            return
        if base in ("add", "adds", "sub", "subs") and len(source_registers) == 1 and gp(sources[0]):
            amount = immediate(sources[1]) if len(sources) >= 2 else None
            if amount is not None and len(sources) == 3:
                shift = SHIFT.match(sources[2])
                amount = amount << int(shift.group(1)) if shift else None
            if amount is not None and len(sources) <= 3:
                sign = 1 if base.startswith("add") else -1
                state.assign(target, frozenset(value if value == "N" or value[1] is None
                                               else ("S", value[1] + sign * amount) for value in values[0]))
                return
        if base in ("csel", "csinc", "csinv", "csneg"):
            # A conditional select yields either operand: their provenances join (path-dependent).
            selected = state.value(gp(sources[0]) or "xzr") | state.value(gp(sources[1]) or "xzr")
            state.assign(target, frozenset(value if value == "N" else ("S", value[1]) for value in selected))
            return
        combined = frozenset({("S", None)})
        if any(provenance(value) == "A" for value in values):
            combined = combined | NOT_STACK
        state.assign(target, combined)


def proven_noreturn(symbol: str, bodies: dict[str, list[str]], dialect: Dialect) -> bool:
    """A fatal diagnostic is exempt only if every reachable path of its witness body ends in abort()."""
    body = bodies.get(symbol)
    if body is None:
        return False
    blocks, errors = build_cfg(body, dialect)
    labels = {label: index for index, block in enumerate(blocks) for label in block.labels}
    seen, pending, proven = set(), [0], not errors
    while pending and proven:
        index = pending.pop()
        if index in seen:
            continue
        seen.add(index)
        ends = False
        for instruction in blocks[index].instructions:
            target = instruction.operands[-1] if instruction.operands else ""
            if instruction.base == "bl" and dialect.symbol(target) in NORETURN_LEAVES:
                ends = True
                break
            if (RETURN.match(instruction.base) or INDIRECT.match(instruction.base) or
                    (instruction.base == "b" and target not in labels)):
                proven = False
                break
            if instruction.base in ("b", "b.cond", "cbz", "cbnz", "tbz", "tbnz"):
                if target not in labels:
                    proven = False
                    break
                pending.append(labels[target])
                if instruction.base == "b":
                    ends = True
                    break
        if not ends and proven:
            if index + 1 >= len(blocks):
                proven = False
            else:
                pending.append(index + 1)
    return proven


def closure(program: str, bodies: dict[str, list[str]], dialect: Dialect, noreturn: set[str],
            analyses: dict[tuple, Analysis]) -> tuple[int, list[str]]:
    """Spills and structural errors of a function and every callee it reaches, with argument provenance."""
    spills, errors, pending, seen = 0, [], [(program, frozenset())], set()
    while pending:
        entry = pending.pop()
        if entry in seen:
            continue
        seen.add(entry)
        name, arguments = entry
        if entry not in analyses:
            analysis = Analysis(name, bodies[name], dialect, bodies, noreturn, arguments)
            if not any(block.instructions for block in analysis.blocks):
                analysis.errors.append("empty or unclassifiable body")
            else:
                analysis.run()
            analyses[entry] = analysis
        analysis = analyses[entry]
        errors.extend(f"via {name}: {error}" for error in analysis.errors)
        spills += analysis.spills
        pending.extend(analysis.callees)
    return spills, errors


def program_key(name: str) -> str:
    """Stable budget key rows:family:policy:encoding:map:minimum for a tile program symbol."""
    match = PROGRAM.search(name)
    if not match:
        raise ValueError(f"not a tile program: {name}")
    return ":".join(match.groups())


ENVELOPE_PROGRAMS = {
    # Row tiles x integer family the probe instantiates in each admitted ISA envelope.
    "NEON": ("1:ASIMD", "2:ASIMD", "3:ASIMD", "4:ASIMD"),
    "NEON_DOTPROD": ("1:ASIMD", "2:ASIMD", "3:ASIMD", "4:ASIMD", "1:DotProd", "2:DotProd", "3:DotProd", "4:DotProd"),
    "NEON_I8MM": ("1:ASIMD", "2:ASIMD", "3:ASIMD", "4:ASIMD", "1:DotProd", "2:DotProd", "3:DotProd", "4:DotProd",
                  "2:I8MM", "3:I8MM", "4:I8MM"),
}
TILE_RUN = re.compile(r"27CPUNativeNEONProjectionTileILm\d+EE3run")


def audit(assembly: str, mode: str) -> dict[str, int]:
    """Per-program FP/SIMD spill counts over the probe's complete inventory and call closure.

    Structural failures (missing/duplicate/empty programs, unauditable callees, indirect branches, ambiguous or
    unclassifiable stack addressing, unproven ABI saves) are fatal and never paid from a spill budget.
    """
    if mode not in ENVELOPE_PROGRAMS:
        raise ValueError(f"no tile program inventory for ISA envelope {mode}")
    inventory = INVENTORY.findall(assembly)
    if inventory != [",".join(ENVELOPE_PROGRAMS[mode])]:
        raise ValueError(f"probe instantiation inventory {inventory} does not match the {mode} envelope")
    expected = set()
    for entry in ENVELOPE_PROGRAMS[mode]:
        rows, dot = entry.split(":")
        leaf = 1 if dot == "I8MM" and rows == "1" else DOTS[dot]
        for policy in (0, 1):
            for encoding, nibble_map, minimum in DISPATCH:
                expected.add((int(rows), leaf, policy, encoding, nibble_map, minimum))
    if not expected:
        raise ValueError("register witness inventory names no tile instantiation")
    dialect = Dialect(assembly)
    bodies = functions(assembly)
    noreturn = {name for name in bodies if dialect.symbol(name) == FATAL_DIAGNOSTIC and
                proven_noreturn(name, bodies, dialect)}
    noreturn |= {("_" if dialect.macho else "") + leaf for leaf in NORETURN_LEAVES}
    found: dict[tuple, str] = {}
    for name in bodies:
        match = PROGRAM.search(name)
        if match:
            key = tuple(int(group) for group in match.groups())
            if key in found:
                raise ValueError(f"tile program appears twice: {name}")
            if key not in expected:
                raise ValueError(f"tile program outside the {mode} inventory: {name}")
            found[key] = name
        elif TILE_RUN.search(name):
            raise ValueError(f"unrecognized tile program symbol: {name}")
    missing = expected - found.keys()
    if missing:
        raise ValueError(f"register witness is missing {len(missing)} expected tile programs, e.g. {sorted(missing)[:3]}")
    analyses: dict[tuple, Analysis] = {}
    result: dict[str, int] = {}
    structural: list[str] = []
    for key in sorted(expected):
        program = found[key]
        result[program], errors = closure(program, bodies, dialect, noreturn, analyses)
        structural.extend(f"{program_key(program)} {error}" for error in errors)
    if structural:
        raise ValueError(f"native NEON tile witness is structurally unprovable ({len(structural)}): "
                         + "; ".join(structural[:6]))
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


WITNESS_FLAGS = ["-O3", "-fno-sanitize=all", "-fno-lto", "-std=c++20"]


def compiler_identity(version: str) -> str:
    """CMake-style compiler identity (id-major) derived from the compiler's own --version output."""
    apple = re.search(r"^Apple clang version (\d+)\.", version, re.MULTILINE)
    if apple:
        return f"AppleClang-{apple.group(1)}"
    clang = re.search(r"^(?:\S+ )?clang version (\d+)\.", version, re.MULTILINE)
    if clang:
        return f"Clang-{clang.group(1)}"
    gnu = re.search(r"^\S*(?:g\+\+|c\+\+|gcc)\S* \(.*\) (\d+)\.\d+(?:\.\d+)?\s*$", version, re.MULTILINE)
    if gnu and "Free Software Foundation" in version:
        return f"GNU-{gnu.group(1)}"
    raise ValueError(f"unrecognized compiler identity: {version.splitlines()[0] if version else version!r}")


def read_depfile(path: Path, directory: str) -> list[str]:
    """Logical (unresolved, absolute) paths named by a compiler dependency file."""
    text = path.read_text().replace("\\\n", " ")
    target, _, prerequisites = text.partition(": ")
    if not prerequisites:
        raise ValueError(f"malformed compiler dependency file {path}")
    return sorted({os.path.normpath(os.path.join(directory, item)) for item in shlex.split(prerequisites)})


def input_identity(path: str) -> dict:
    """Content hash, resolved path and the symlink chain of every component of one logical input path."""
    links, current = [], Path(path).anchor
    for part in Path(path).parts[1:]:
        current = os.path.join(current, part)
        if os.path.islink(current):
            links.append([current, os.readlink(current)])
    return {"sha256": envelope.digest(Path(path)), "resolved": os.path.realpath(path), "links": links}


def bind_inputs(paths: list[str]) -> dict[str, dict]:
    """Identity of every dependency; a missing input fails closed."""
    return {path: input_identity(path) for path in paths}


def executed_command(command: list[str], source: str, assembly: Path, dependencies: Path) -> list[str]:
    return command + WITNESS_FLAGS + ["-S", source, "-o", str(assembly), "-MD", "-MF", str(dependencies),
                                      "-MT", "probe"]


def build(args: argparse.Namespace) -> None:
    """Compile the probe with the target command; publish the depfile, then the proof last and atomically."""
    output = Path(args.proof)
    output.parent.mkdir(parents=True, exist_ok=True)
    assembly, dependencies = output.with_suffix(".s"), output.with_suffix(".deps")
    preprocessed, depfile, temporary = output.with_suffix(".pre.deps"), output.with_suffix(".d"), output.with_suffix(".tmp")
    for stale in (output, temporary, assembly, dependencies, preprocessed, depfile):
        stale.unlink(missing_ok=True)
    command, directory = envelope.target_command(args)
    version = subprocess.check_output([command[0], "--version"], cwd=directory, text=True)
    identity = compiler_identity(version)
    if identity != args.identity:
        raise ValueError(f"compiler reports identity {identity}, configured budget identity is {args.identity}")
    subprocess.run(command + WITNESS_FLAGS + ["-M", "-MF", str(preprocessed), "-MT", "probe", args.source],
                   cwd=directory, check=True, stdout=subprocess.DEVNULL)
    before = bind_inputs(read_depfile(preprocessed, directory))
    executed = executed_command(command, args.source, assembly, dependencies)
    subprocess.run(executed, cwd=directory, check=True)
    closure = read_depfile(dependencies, directory)
    if closure != sorted(before):
        raise ValueError("the compilation's dependency closure differs from its preprocessing closure")
    after = bind_inputs(closure)
    if after != before:
        changed = sorted(path for path in closure if before[path] != after[path])
        raise ValueError(f"inputs changed while the witness compiled: {changed[:3]}")
    counts = audit(assembly.read_text(), args.mode)
    require_registers(counts, args.identity, args.mode)
    result = {
        "schema": 4, "signature": args.signature, "identity": identity, "mode": args.mode,
        "checker_sha256": envelope.digest(Path(__file__)),
        "envelope_sha256": envelope.digest(_ENVELOPE),
        "budget_sha256": envelope.digest(BUDGET_PATH),
        "compiler_version": version,
        "executed_command": executed, "executed_directory": directory,
        "target_command_sha256": envelope.command_digest(command, directory),
        "inputs": after,
        "assembly_file": assembly.name, "assembly_sha256": envelope.digest(assembly),
        "program_violations": counts,
    }
    # Ninja depfile: any input of the closure re-runs the witness. Written before the proof, which is published last.
    target = args.depfile_target or str(output)
    depfile.write_text(f"{target}: " + " ".join(path.replace(" ", "\\ ") for path in closure) + "\n")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    temporary.replace(output)


def verify(args: argparse.Namespace) -> None:
    """Validate the retained proof without invoking a compiler."""
    proof_path = Path(args.proof)
    proof = json.loads(proof_path.read_text())
    if (proof.get("schema") != 4 or proof.get("signature") != args.signature or
            proof.get("identity") != args.identity or proof.get("mode") != args.mode):
        raise ValueError("tile register proof has a foreign schema/configuration")
    if compiler_identity(str(proof.get("compiler_version", ""))) != args.identity:
        raise ValueError("recorded compiler identity does not match the configured budget identity")
    for key, path in (("checker_sha256", Path(__file__)), ("envelope_sha256", _ENVELOPE),
                      ("budget_sha256", BUDGET_PATH)):
        if proof.get(key) != envelope.digest(path):
            raise ValueError(f"stale tile register tool: {path}")
    inputs = proof.get("inputs")
    source = os.path.normpath(os.path.abspath(args.source))
    if not isinstance(inputs, dict) or source not in inputs:
        raise ValueError("tile register proof does not bind the probe's dependency closure")
    for path, expected in inputs.items():
        if not os.path.isfile(path) or input_identity(path) != expected:
            raise ValueError(f"stale tile register input (source, header or symlink changed): {path}")
    command, directory = envelope.target_command(args)
    if proof.get("target_command_sha256") != envelope.command_digest(command, directory):
        raise ValueError("actual target compile command changed after the register witness")
    executed = proof.get("executed_command")
    name = proof.get("assembly_file")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("tile register proof must name one retained local assembly artifact")
    assembly = proof_path.parent / name
    expected_command = executed_command(command, args.source, assembly, proof_path.with_suffix(".deps"))
    if (not isinstance(executed, list) or proof.get("executed_directory") != directory or
            executed != expected_command or executed[0] != command[0]):
        raise ValueError("recorded witness invocation is not the target command over the probe")
    if not assembly.is_file() or proof.get("assembly_sha256") != envelope.digest(assembly):
        raise ValueError("tile register assembly is missing or changed")
    counts = audit(assembly.read_text(), args.mode)
    if proof.get("program_violations") != counts:
        raise ValueError("tile register evidence changed")
    require_registers(counts, args.identity, args.mode)


def census(args: argparse.Namespace) -> None:
    """Maintainer tool: print the per-program spill census for qualifying a budget (no budget applied)."""
    command, directory = envelope.target_command(args)
    version = subprocess.check_output([command[0], "--version"], cwd=directory, text=True)
    assembly = Path(args.proof).with_suffix(".census.s")
    assembly.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command + WITNESS_FLAGS + ["-S", args.source, "-o", str(assembly)], cwd=directory, check=True)
    counts = audit(assembly.read_text(), args.mode)
    print(json.dumps({"identity": compiler_identity(version), "mode": args.mode,
                      "programs": {program_key(name): count for name, count in counts.items()}},
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
