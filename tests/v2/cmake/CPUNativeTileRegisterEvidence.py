#!/usr/bin/env python3
"""Build and authenticate a register-allocation witness for the native NEON projection tiles.

Build phase: compile CPUNativeNEONTileRegisterProbe.cpp with the real target's
ordered compile command at -O3 and audit the assembly.

- Inventory: the expected tile K-loop programs (row tile x integer family x
  policy x encoding/map/minimum) derive from the admitted ISA envelope; the
  probe's inventory string must agree. Every expected program must be emitted
  exactly once with real instructions; any other tile program symbol fails.
- Grammar: function bodies keep every statement, including one on a global
  label's line. Only admitted mnemonics, operand forms and addressing modes may
  appear; executable sections admit only directives that emit no code bytes
  (data directives belong to data sections, which control may not reach or fall
  into); unknown section control fails.
- Calls: the only admitted call is to the exact fatal-diagnostic symbol, whose
  body must be acyclic with every path ending in a call to abort(). Every other
  call boundary (local or external, direct or indirect, call or tail call) is a
  structural failure, so no interprocedural state needs to be modelled.
- Analysis: each program is analysed over its control-flow graph. Stack
  provenance of every general register is a forward data-flow fact; merge
  definitions (bfi, movk, ...) keep their destination as an input; any other
  computation on a stack address keeps it a stack address at an unknown offset;
  a base that is a stack address on only some paths fails. 8-byte slots holding
  stack addresses are tracked with their offsets; partial or reinterpreting
  access to such a slot, a 32-bit view of a stack address and a stack address
  leaving the tracked frame (to memory, an FP/SIMD register or an index) fail.
- Spills: each load/store moving FP/SIMD data (any width) through a stack
  address counts once, except proven ABI saves/restores of d8-d15: whole-register
  contiguous str/stp/ldr/ldp of the d or q view only, saved once before any
  clobber to a fixed slot, slot untouched while live, restored once from that
  slot, never clobbered after the restore, restored before return.
- Structural failures are fatal and never paid from a budget.
- Policy: single-row (decode) programs must not spill at all; grouped row tiles
  may not exceed the committed per-program budget for the exact compiler
  identity and ISA envelope (CPUNativeTileRegisterBudget.json, a ratchet). The
  identity is derived from the compiler's own version output and must equal the
  configured one; unknown identities/envelopes fail closed.
- Dependencies: the dependency closure is the one the compiler reports for this
  exact compilation (-MD on the -S run), cross-checked against a preprocessing
  pass with the same flags; every input is hashed before and after compiling and
  bound with its logical path, its resolved path (component-wise, as the kernel
  resolves symlinks and "..") and the symlinks followed. The Ninja depfile names
  resolved paths; symlink retargeting is caught by verification.

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
CONDITION = re.compile(rf"^(?:{CONDITIONS})$")
CONDITIONAL_BRANCH = re.compile(rf"^b\.?(?:{CONDITIONS})$")
INDIRECT = re.compile(r"^(?:br|blr)(?:aa|ab|aaz|abz)?$")
RETURN = re.compile(r"^ret(?:aa|ab)?$")
BRANCHES = ("b", "bl", "b.cond", "cbz", "cbnz", "tbz", "tbnz")

# Admitted instruction grammar. Anything else is a structural failure.
LOADS = frozenset("ldr ldur ldp ldpsw ldrb ldrh ldrsb ldrsh ldrsw ldurb ldurh ldursb ldursh ldursw "
                  "ld1 ld1r ld2 ld2r ld3 ld3r ld4 ld4r".split())
STORES = frozenset("str stur stp strb strh sturb sturh st1 st2 st3 st4".split())
PREFETCHES = frozenset({"prfm", "prfum"})
# Contiguous whole-register transfers: the only forms an ABI save/restore of d8-d15 may use.
ABI_TRANSFERS = frozenset({"str", "stur", "stp", "ldr", "ldur", "ldp"})
ABI_REGISTER = re.compile(r"^([dq])(8|9|1[0-5])$")
NO_DESTINATION = frozenset({"cmp", "cmn", "ccmp", "ccmn", "tst", "fcmp", "fcmpe"})
# The destination is also an input: only part of it is written.
MERGE_DEFINITIONS = frozenset({"bfi", "bfxil", "bfm", "bfc", "movk"})
CONDITIONAL_SELECTS = frozenset({"csel", "csinc", "csinv", "csneg"})
DEFINITIONS = frozenset(
    "add adds sub subs and ands orr eor bic bics orn eon mov mvn neg lsl lsr asr madd msub mul smaddl umaddl "
    "smull umull sxtw sxth sxtb uxtw uxth uxtb sbfiz sbfx ubfiz ubfx cset csetm adrp adr fmov "
    "addp dup ext fadd fsub fmul fmla fneg fcvtl scvtf ucvtf cmeq cmhi cmge cmgt bif bit bsl ins movi mvni "
    "saddlp sdot smmla shl sshr ushr usra sshll ushll smull2 umull2 uaddw tbl zip1 zip2".split())
ADMITTED = DEFINITIONS | MERGE_DEFINITIONS | CONDITIONAL_SELECTS | NO_DESTINATION | LOADS | STORES | PREFETCHES

# Directives admitted inside executable sections: none of them emits instruction bytes.
CODE_DIRECTIVES = re.compile(r"^\.(?:cfi_[a-z_]+|p2align|align|loc|file|type|size|globl|global|weak|hidden|"
                             r"private_extern|weak_def_can_be_hidden|weak_definition|loh|ident|addrsig|addrsig_sym|"
                             r"subsections_via_symbols|build_version)$")
SECTION_CONTROL = frozenset({".section", ".text", ".data", ".bss"})
UNSUPPORTED_SECTION_CONTROL = frozenset({".pushsection", ".popsection", ".previous", ".subsection"})
ELF_DATA_SECTIONS = (".rodata", ".data", ".bss", ".gcc_except_table", ".note", ".comment", ".debug", ".eh_frame",
                     ".init_array", ".fini_array", ".tdata", ".tbss")
MACHO_DATA_SECTIONS = ("__gcc_except_tab", "__const", "__cstring", "__literal4", "__literal8", "__literal16",
                       "__eh_frame", "__compact_unwind")

GP64 = re.compile(r"^(?:x([12]?[0-9]|30)|sp|fp|lr|xzr)$")
GP32 = re.compile(r"^(?:w([12]?[0-9]|30)|wsp|wzr)$")
FP = re.compile(r"^([bhsdqv])([12]?[0-9]|3[01])(?:\.(\d*)([bhsd]))?(?:\[\d+\])?$")
IMMEDIATE = re.compile(r"^#?(-?(?:0x[0-9a-fA-F]+|\d+))$")
FLOAT = re.compile(r"^#?-?\d+\.\d+(?:e[+-]?\d+)?$")
SHIFT = re.compile(r"^(lsl|lsr|asr|ror|msl|uxtb|uxth|uxtw|uxtx|sxtb|sxth|sxtw|sxtx)(?:\s*#?(\d+))?$")
RELOCATION = re.compile(r"^(?:#?:[a-z0-9_]+:[\w.$+-]+|[\w.$]+@\w+)$")
INDEX = re.compile(r"^([xw](?:[12]?[0-9]|30))(?:\s*,\s*(?:lsl|uxtw|sxtw|sxtx|uxtx)(?:\s*#?\d+)?)?$")
MEMORY = re.compile(r"^\[\s*(sp|fp|x(?:[12]?[0-9]|30))\s*(?:,\s*(.*?))?\s*\](!?)$")
SYMBOL = re.compile(r"^[A-Za-z_.$][\w.$@]*$")
WIDTH = {"b": 1, "h": 2, "s": 4, "d": 8, "q": 16}
CALLEE_SAVED = range(8, 16)
NOT_STACK = frozenset({"N"})
UNKNOWN_STACK = frozenset({("S", None)})
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

    def section(self, directive: str, arguments: str) -> tuple[str, bool | None]:
        """(section identity, executable) for a section directive; executable None = unsupported section."""
        if directive != ".section":
            return directive, directive == ".text"
        fields = [field.strip().strip('"') for field in arguments.split(",")]
        if self.macho:
            if len(fields) < 2 or fields[0] not in ("__TEXT", "__DATA", "__DATA_CONST", "__DWARF"):
                return arguments, None
            if fields[0] == "__TEXT" and fields[1] == "__text":
                return arguments, True
            return arguments, False if fields[0] != "__TEXT" or fields[1] in MACHO_DATA_SECTIONS else None
        name = fields[0]
        flags = re.search(r',\s*"([^"]*)"', arguments)
        if name.startswith(".text"):
            return name, True if not flags or "x" in flags.group(1) else None
        if flags:
            return name, "x" in flags.group(1)
        return name, False if name.startswith(ELF_DATA_SECTIONS) else None


def functions(assembly: str) -> dict[str, list[str]]:
    """Split compiler -S output into global function bodies; local labels and same-line statements stay inside."""
    dialect = Dialect(assembly)
    bodies: dict[str, list[str]] = {}
    current = None
    for line in assembly.splitlines():
        label = LABEL.match(line)
        if label and not dialect.local(label.group(1)):
            current = label.group(1)
            if current in bodies:
                raise ValueError(f"symbol defined twice in the witness: {current}")
            bodies[current] = [label.group(2)] if label.group(2).strip() else []
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


def gp_view(operand: str) -> str | None:
    """The 64-bit register a w operand is the low half of."""
    match = GP32.match(operand)
    if not match:
        return None
    return {"wsp": "sp", "wzr": "xzr"}.get(operand, f"x{match.group(1)}")


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


WHOLE_X = frozenset({"ldr", "ldur", "ldp", "str", "stur", "stp"})
NARROW = {"b": 1, "h": 2, "sb": 1, "sh": 2, "sw": 4}


def memory_width(mnemonic: str, operand: str) -> int:
    """Bytes a general-register operand of a load/store transfers (sign-extending forms read less)."""
    match = re.match(r"^(?:ldr|ldur|str|stur|ldp|stp)(b|h|sb|sh|sw)?$", mnemonic)
    if match and match.group(1):
        return NARROW[match.group(1)]
    if mnemonic == "ldpsw":
        return 4
    return 8 if gp(operand) else 4


def operand_kind(operand: str) -> str | None:
    """Kind of one operand in the admitted grammar; None if unclassifiable."""
    if gp(operand):
        return "gp"
    if GP32.match(operand):
        return "gp32"
    if operand.startswith("["):
        return "memory"
    try:
        if fp_registers(operand) is not None:
            return "fp"
    except ValueError:
        return None
    if IMMEDIATE.match(operand) or FLOAT.match(operand):
        return "immediate"
    if SHIFT.match(operand):
        return "shift"
    if CONDITION.match(operand):
        return "condition"
    if RELOCATION.match(operand) or SYMBOL.match(operand):
        return "symbol"
    return None


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
    def __init__(self, section: str, executable: bool | None) -> None:
        self.labels: list[str] = []
        self.instructions: list[Instruction] = []
        self.successors: list[int] = []
        self.section, self.executable = section, executable


def build_cfg(body: list[str], dialect: Dialect) -> tuple[list[Block], list[str]]:
    """Basic blocks of one function body; only admitted directives may appear in executable sections."""
    # The section the function label itself is in is declared before the label: an unknown identity.
    section, executable = "<function entry section>", True
    blocks, errors = [Block(section, executable)], []

    def start(new_section: str, new_executable: bool | None) -> None:
        if blocks[-1].instructions or blocks[-1].labels:
            blocks.append(Block(new_section, new_executable))
        else:
            blocks[-1].section, blocks[-1].executable = new_section, new_executable

    for line in body:
        label = LABEL.match(line)
        statements = dialect.statements(label.group(2) if label else line)
        if label:
            if blocks[-1].instructions:
                blocks.append(Block(section, executable))
            blocks[-1].labels.append(label.group(1))
        for statement in statements:
            statement = statement.strip()
            if not statement:
                continue
            if statement.startswith("."):
                directive, _, arguments = statement.partition(" ")
                directive = directive.split("\t", 1)[0]
                arguments = statement[len(directive):].strip()
                if directive in UNSUPPORTED_SECTION_CONTROL:
                    errors.append(f"unsupported section control: {statement}")
                elif directive in SECTION_CONTROL:
                    section, executable = dialect.section(directive, arguments)
                    if executable is None:
                        errors.append(f"unclassifiable section: {statement}")
                    start(section, executable)
                elif executable and not CODE_DIRECTIVES.match(directive):
                    errors.append(f"directive may emit code bytes: {statement}")
                continue
            if not executable:
                errors.append(f"instruction outside an executable section: {statement}")
                continue
            instruction = Instruction(statement, dialect)
            blocks[-1].instructions.append(instruction)
            if instruction.base in BRANCHES or INDIRECT.match(instruction.base) or RETURN.match(instruction.base):
                blocks.append(Block(section, executable))  # Every branch ends its block.
    return blocks, errors


class State:
    """Per-program-point facts: general register stack provenance, slots holding stack addresses, ABI saves.

    tainted holds (slot offset, stack offset or None) pairs: the 8-byte slots that hold a stack address.
    """

    def __init__(self, registers: dict[str, frozenset], tainted: frozenset, saved: dict[int, frozenset]) -> None:
        self.registers, self.tainted, self.saved = registers, tainted, saved

    @staticmethod
    def entry() -> "State":
        """Function entry: sp at offset 0 of the new frame, x29 the caller's frame pointer."""
        registers = {"sp": frozenset({("S", 0)}), "x29": UNKNOWN_STACK}
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
        return (tuple(sorted((r, tuple(sorted(v, key=repr))) for r, v in self.registers.items())),
                tuple(sorted(self.tainted, key=repr)), tuple(tuple(sorted(self.saved[n], key=repr)) for n in CALLEE_SAVED))


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
    """Control-flow-aware audit of one function: counts FP/SIMD spills, records structural failures.

    Calls are not modelled: the only admitted call is to a proven never-returning diagnostic. Every other call
    boundary (local or external, direct or indirect, call or tail call) is a structural failure.
    """

    def __init__(self, name: str, body: list[str], dialect: Dialect, noreturn: set[str]) -> None:
        self.name, self.dialect, self.noreturn = name, dialect, noreturn
        self.blocks, self.errors = build_cfg(body, dialect)
        self.spills = 0
        self.report = False
        self.labels = {label: index for index, block in enumerate(self.blocks) for label in block.labels}

    def fail(self, instruction: Instruction | None, message: str) -> None:
        if self.report:
            where = f" at '{instruction.text}'" if instruction else ""
            self.errors.append(f"{message}{where}")

    # Control flow --------------------------------------------------------------------------------------------
    def classify_branch(self, instruction: Instruction) -> tuple[str, str | None]:
        """('fall'|'branch'|'jump'|'return'|'call'|'noreturn'|'indirect'|'invalid', target)."""
        base = instruction.base
        if RETURN.match(base):
            return ("return", None) if not instruction.operands else ("indirect", None)
        if INDIRECT.match(base):
            return "indirect", None
        if base not in BRANCHES:
            return "fall", None
        target = instruction.operands[-1] if instruction.operands else ""
        if not SYMBOL.match(target):
            return "indirect", None
        if target in self.labels:
            if base == "bl" or not self.blocks[self.labels[target]].executable:
                return "invalid", target
            return ("jump" if base == "b" else "branch"), target
        if self.dialect.local(target):
            return "invalid", target
        if base == "bl" and target in self.noreturn:
            return "noreturn", target
        return "call", target

    def link(self) -> None:
        for index, block in enumerate(self.blocks):
            block.falls = True
            for instruction in block.instructions:
                kind, target = self.classify_branch(instruction)
                if kind in ("jump", "branch"):
                    block.successors.append(self.labels[target])
                if kind in ("jump", "return", "noreturn", "indirect", "invalid", "call"):
                    block.falls = False
            if block.falls:
                block.successors.append(index + 1 if index + 1 < len(self.blocks) else -1)

    # Data flow ---------------------------------------------------------------------------------------------------
    def run(self) -> None:
        self.link()
        states: dict[int, State] = {0: State.entry()}
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
            if self.transfer(index, states[index].copy()) is None or not block.falls or not block.executable:
                continue
            following = index + 1
            if following >= len(self.blocks):
                self.errors.append("control falls off the end of the function")
            elif not self.blocks[following].executable:
                self.errors.append("control falls into a data section")
            elif self.blocks[following].section != block.section:
                self.errors.append("control falls through a section switch")

    def transfer(self, index: int, state: State) -> State | None:
        for instruction in self.blocks[index].instructions:
            if not self.step(instruction, state):
                return None
        return state

    def step(self, instruction: Instruction, state: State) -> bool:
        """Apply one instruction; False when control leaves the function or the path cannot be classified."""
        kind, target = self.classify_branch(instruction)
        if kind == "indirect":
            self.fail(instruction, "indirect or uninspectable branch")
            return False
        if kind == "invalid":
            self.fail(instruction, "branch to a label that is not code of this function")
            return False
        if kind == "noreturn":
            return False
        if kind == "call":
            self.fail(instruction, f"unsupported call boundary (only the proven fatal diagnostic may be called): {target}")
            return False
        if kind == "return":
            self.leave(instruction, state)
            return False
        kinds = [operand_kind(operand) for operand in instruction.operands]
        if None in kinds:
            self.fail(instruction, "unclassifiable operand")
            return False
        if kind in ("jump", "branch"):
            return self.reads(instruction, state, instruction.operands[:-1])
        if instruction.base not in ADMITTED:
            self.fail(instruction, f"instruction outside the admitted grammar: {instruction.base}")
            return False
        if instruction.memory is not None:
            return self.access(instruction, state)
        if "memory" in kinds or instruction.base in LOADS | STORES | PREFETCHES:
            self.fail(instruction, "unclassifiable memory form")
            return False
        return self.compute(instruction, state)

    def reads(self, instruction: Instruction, state: State, operands: list[str]) -> bool:
        """A 32-bit view of a stack address cannot be classified: reading one is a structural failure."""
        for operand in operands:
            view = gp_view(operand)
            if view and provenance(state.value(view)) != "N":
                self.fail(instruction, f"32-bit view of a stack address {operand}")
                return False
        return True

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

    def access(self, instruction: Instruction, state: State) -> bool:
        """Loads/stores: FP/SIMD stack traffic, ABI saves/restores, slots holding stack addresses."""
        name = instruction.base
        load, store = name in LOADS, name in STORES
        if instruction.memory != len(instruction.operands) - 1 and not (
                instruction.memory == len(instruction.operands) - 2 and immediate(instruction.operands[-1]) is not None):
            self.fail(instruction, "unclassifiable post-index form")
            return False
        memory = MEMORY.match(instruction.operands[instruction.memory])
        if not memory:
            self.fail(instruction, "unclassifiable addressing mode")
            return False
        base = gp(memory.group(1))
        offset_text, writeback = memory.group(2), memory.group(3) == "!"
        post = instruction.operands[instruction.memory + 1:]
        displacement: int | None = 0
        if offset_text:
            if immediate(offset_text) is not None:
                displacement = immediate(offset_text)
            elif RELOCATION.match(offset_text):
                displacement = None
                if provenance(state.value(base)) != "N":
                    self.fail(instruction, "relocated offset from a stack base")
                    return False
            elif INDEX.match(offset_text):
                displacement = None
                index = INDEX.match(offset_text).group(1)
                if provenance(state.value(gp(index) or gp_view(index))) != "N":
                    self.fail(instruction, "stack address used as an index register")
                    return False
            else:
                self.fail(instruction, "unclassifiable addressing offset")
                return False
        if post and (writeback or offset_text):
            self.fail(instruction, "unclassifiable post-index form")
            return False
        base_values = state.value(base)
        kind = provenance(base_values)
        if kind == "A":
            self.fail(instruction, "base register is a stack address on only some paths")
            return False
        stack = kind == "S"
        origin = single_offset(base_values)
        address = None if origin is None or displacement is None else origin + (0 if post else displacement)
        data = [] if name in PREFETCHES else instruction.operands[:instruction.memory]
        if store and not self.reads(instruction, state, data):
            return False
        items, position = [], address
        for operand in data:
            registers = fp_registers(operand)
            if registers is not None:
                width = sum(width for _, width in registers)
                items.append(("fp", operand, registers, position, width))
            elif gp(operand) or GP32.match(operand):
                width = memory_width(name, operand)
                items.append(("gp" if gp(operand) else "gp32", operand, None, position, width))
            else:
                self.fail(instruction, f"unclassifiable data operand {operand}")
                return False
            position = None if position is None else position + width
        if stack and not self.slots(instruction, state, load, store, address, items):
            return False
        if store and not stack and any(provenance(state.value(gp(operand))) != "N"
                                       for category, operand, *_ in items if category == "gp"):
            self.fail(instruction, "a stack address is stored where it cannot be tracked")
            return False
        spilled = False
        eligible = name in ABI_TRANSFERS and all(ABI_REGISTER.match(operand) for _, operand, *_ in items)
        for category, operand, registers, start, width in items:
            if category != "fp":
                if load:
                    stored = frozenset(("S", offset) for slot, offset in state.tainted if slot == start)
                    exact = stack and start is not None and stored and category == "gp" and name in WHOLE_X
                    target = gp(operand) or gp_view(operand)
                    state.assign(target, stored if exact else
                                 (frozenset({"N", ("S", None)}) if stack and start is None and state.tainted
                                  and category == "gp" else NOT_STACK))
                continue
            for number, register_width in registers:
                if stack and eligible and store and self.save(state, number, register_width, start):
                    pass
                elif stack and eligible and load and self.restore(state, number, register_width, start):
                    pass
                elif stack:
                    spilled = True
                    if load:
                        self.clobber(instruction, state, number)
                elif load:
                    self.clobber(instruction, state, number)
                start = None if start is None else start + register_width
        if spilled and self.report:
            self.spills += 1
        if writeback or post:
            step = displacement if writeback else immediate(post[0])
            if step is None:
                state.assign(base, UNKNOWN_STACK if stack else NOT_STACK)
            else:
                state.assign(base, frozenset(value if value == "N" or value[1] is None else ("S", value[1] + step)
                                             for value in base_values))
        return True

    def slots(self, instruction: Instruction, state: State, load: bool, store: bool, address: int | None,
              items: list) -> bool:
        """Track 8-byte slots holding stack addresses and protect live ABI save slots; partial access fails."""
        live = [(n, slot) for n in CALLEE_SAVED for status, slot in state.saved[n]
                if status in ("S", "D") and slot is not None]
        if address is None:
            if store and (state.tainted or live):
                self.fail(instruction, "stack store at an unknown offset while tracked slots are live")
                return False
            if store and any(provenance(state.value(gp(operand))) != "N"
                             for category, operand, *_ in items if category == "gp"):
                self.fail(instruction, "a stack address is stored where it cannot be tracked")
                return False
            return True
        for category, operand, _, start, width in items:
            end = start + width
            for slot in {slot for slot, _ in state.tainted}:
                if not (slot < end and start < slot + 8):
                    continue
                exact = start == slot and width == 8 and category == "gp" and instruction.base in WHOLE_X
                if load and not exact:
                    self.fail(instruction, "partial or reinterpreting reload of a stored stack address")
                    return False
                if store and not exact and not (start <= slot and slot + 8 <= end):
                    self.fail(instruction, "partial overwrite of a stored stack address")
                    return False
                if store and not exact:
                    state.tainted = frozenset(entry for entry in state.tainted if entry[0] != slot)
            if store:
                for number, (slot, slot_width) in live:
                    if start < slot + slot_width and slot < end:
                        self.fail(instruction, f"overwrites the live save slot of d{number}")
                        return False
                if category == "gp":
                    value = provenance(state.value(gp(operand)))
                    if value == "A":
                        self.fail(instruction, "stores a stack address on only some paths")
                        return False
                    kept = frozenset(entry for entry in state.tainted if entry[0] != start)
                    state.tainted = kept | ({(start, stack[1]) for stack in state.value(gp(operand))}
                                            if value == "S" else set())
        return True

    def save(self, state: State, number: int, width: int, slot: int | None) -> bool:
        """An ABI save stores an unsaved, unclobbered d8-d15 (or its q view) whole to a fixed frame slot."""
        if number not in CALLEE_SAVED or width < 8 or slot is None or state.saved[number] != frozenset({("U", None)}):
            return False
        state.saved[number] = frozenset({("S", (slot, width))})
        return True

    def restore(self, state: State, number: int, width: int, slot: int | None) -> bool:
        """An epilogue restore reloads a saved register whole from its own save slot, exactly once."""
        current = state.saved.get(number) if number in CALLEE_SAVED else None
        if not current or slot is None or any(status not in ("S", "D") or saved != (slot, width)
                                              for status, saved in current):
            return False
        state.saved[number] = frozenset({("R", (slot, width))})
        return True

    def compute(self, instruction: Instruction, state: State) -> bool:
        """Register instructions: propagate stack provenance; FP writes clobber d8-d15."""
        base = instruction.base
        operands = instruction.operands
        if base in NO_DESTINATION or not operands:
            return self.reads(instruction, state, operands)
        destination = operands[0]
        inputs = operands if base in MERGE_DEFINITIONS else operands[1:]
        if not self.reads(instruction, state, inputs):
            return False
        values = [state.value(gp(operand)) for operand in inputs if gp(operand)]
        stack_inputs = any(provenance(value) != "N" for value in values)
        registers = fp_registers(destination)
        if registers is not None:
            if stack_inputs:
                self.fail(instruction, "a stack address moves into an FP/SIMD register")
                return False
            for number, _ in registers:
                self.clobber(instruction, state, number)
            return True
        target = gp(destination)
        if target is None:
            if gp_view(destination) is None:
                self.fail(instruction, f"unclassifiable destination {destination}")
                return False
            if stack_inputs:
                self.fail(instruction, "a stack address is truncated to 32 bits")
                return False
            state.assign(gp_view(destination), NOT_STACK)
            return True
        if not stack_inputs:
            state.assign(target, NOT_STACK)
            return True
        sources = operands[1:]
        if base == "mov" and len(sources) == 1:
            state.assign(target, values[0])
            return True
        if base in CONDITIONAL_SELECTS:
            selected = state.value(gp(sources[0]) or "xzr") | state.value(gp(sources[1]) or "xzr")
            state.assign(target, frozenset(value if value == "N" else ("S", value[1]) for value in selected))
            return True
        if base in ("add", "adds", "sub", "subs") and len(values) == 1 and gp(sources[0]) and len(sources) <= 3:
            amount = immediate(sources[1]) if len(sources) >= 2 else None
            if amount is not None and len(sources) == 3:
                shift = SHIFT.match(sources[2])
                amount = amount << int(shift.group(2) or 0) if shift and shift.group(1) == "lsl" else None
            if amount is not None:
                sign = 1 if base.startswith("add") else -1
                state.assign(target, frozenset(value if value == "N" or value[1] is None
                                               else ("S", value[1] + sign * amount) for value in values[0]))
                return True
        # Any other computation on a stack address: still a stack address, at an unknown offset.
        combined = UNKNOWN_STACK
        if any(provenance(value) == "A" for value in values):
            combined = combined | NOT_STACK
        state.assign(target, combined)
        return True


def proven_noreturn(symbol: str, bodies: dict[str, list[str]], dialect: Dialect) -> bool:
    """The fatal diagnostic is exempt only if its body is acyclic and every path ends in a call to abort()."""
    body = bodies.get(symbol)
    if body is None:
        return False
    blocks, errors = build_cfg(body, dialect)
    if errors:
        return False
    labels = {label: index for index, block in enumerate(blocks) for label in block.labels}
    successors: dict[int, list[int]] = {}
    for index, block in enumerate(blocks):
        edges, ends = [], False
        for instruction in block.instructions:
            target = instruction.operands[-1] if instruction.operands else ""
            if instruction.base == "bl" and target not in labels and dialect.symbol(target) in NORETURN_LEAVES:
                ends = True
                break
            if instruction.base == "bl" and target not in labels and not dialect.local(target):
                continue  # An ordinary call returns here; the path must still reach abort().
            if instruction.base in ("b", "b.cond", "cbz", "cbnz", "tbz", "tbnz"):
                if target not in labels or not blocks[labels[target]].executable:
                    return False
                edges.append(labels[target])
                if instruction.base == "b":
                    ends = True
                    break
            elif RETURN.match(instruction.base) or INDIRECT.match(instruction.base) or instruction.base in BRANCHES:
                return False
        if not ends:
            if index + 1 >= len(blocks):
                successors[index] = [-1]
                continue
            edges.append(index + 1)
        successors[index] = edges
    colors: dict[int, str] = {}

    def visit(index: int) -> bool:
        if index < 0:
            return False
        if colors.get(index) == "active":
            return False
        if colors.get(index) == "done":
            return True
        colors[index] = "active"
        result = all(visit(successor) for successor in successors[index])
        colors[index] = "done"
        return result

    return visit(0)


def analyse_function(name: str, bodies: dict[str, list[str]], dialect: Dialect,
                     noreturn: set[str]) -> tuple[int, list[str]]:
    """Spills and structural errors of one tile program."""
    analysis = Analysis(name, bodies[name], dialect, noreturn)
    if not any(block.instructions for block in analysis.blocks):
        analysis.errors.append("empty or unclassifiable body")
    else:
        analysis.run()
    return analysis.spills, analysis.errors


def noreturn_symbols(bodies: dict[str, list[str]], dialect: Dialect) -> set[str]:
    """Exactly the fatal diagnostic, when proven never to return, and abort()."""
    proven = {name for name in bodies if dialect.symbol(name) == FATAL_DIAGNOSTIC and
              proven_noreturn(name, bodies, dialect)}
    return proven | {("_" if dialect.macho else "") + leaf for leaf in NORETURN_LEAVES}


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
    noreturn = noreturn_symbols(bodies, dialect)
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
    result: dict[str, int] = {}
    structural: list[str] = []
    for key in sorted(expected):
        program = found[key]
        result[program], errors = analyse_function(program, bodies, dialect, noreturn)
        structural.extend(f"{program_key(program)}: {error}" for error in errors)
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
    # No lexical normalisation: "link/../h.h" must keep the filesystem's meaning (".." of the link target).
    return sorted({os.path.join(directory, item) for item in shlex.split(prerequisites)})


def resolve(path: str) -> tuple[str, list[list[str]]]:
    """Resolve a logical path component by component as the kernel does, recording every symlink followed."""
    if not os.path.isabs(path):
        raise ValueError(f"dependency path is not absolute: {path}")
    links, current, pending = [], os.sep, list(Path(path).parts[1:])
    while pending:
        part = pending.pop(0)
        if part in ("", "."):
            continue
        if part == "..":
            current = os.path.dirname(current) or os.sep
            continue
        candidate = os.path.join(current, part)
        if os.path.islink(candidate):
            if len(links) >= 64:
                raise ValueError(f"too many symbolic links resolving {path}")
            target = os.readlink(candidate)
            links.append([candidate, target])
            pending = list(Path(target).parts[1:] if os.path.isabs(target) else Path(target).parts) + pending
            current = os.sep if os.path.isabs(target) else current
        else:
            current = candidate
    if current != os.path.realpath(path):
        raise ValueError(f"dependency path resolution is ambiguous: {path}")
    return current, links


def input_identity(path: str) -> dict:
    """Content hash of the file a logical input path names, its resolved path and the symlinks followed."""
    resolved, links = resolve(path)
    return {"sha256": envelope.digest(Path(resolved)), "resolved": resolved, "links": links}


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
    # Resolved paths: Ninja canonicalises ".." lexically, which would name a different file behind a symlink.
    resolved = sorted({identity["resolved"] for identity in after.values()})
    depfile.write_text(f"{target}: " + " ".join(path.replace(" ", "\\ ") for path in resolved) + "\n")
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
    source = resolve(os.path.abspath(args.source))[0]
    if not isinstance(inputs, dict) or not any(isinstance(identity, dict) and identity.get("resolved") == source
                                               for identity in inputs.values()):
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
