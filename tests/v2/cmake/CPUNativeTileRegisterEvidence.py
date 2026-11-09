#!/usr/bin/env python3
"""Build and authenticate a register-allocation witness for the native NEON projection tiles.

Build phase: compile CPUNativeNEONTileRegisterProbe.cpp into an object with the
real target's ordered compile command at -O3, disassemble that object with the
configured objdump and audit the instructions actually assembled.

- Object: the format (ELF64 AArch64 or Mach-O arm64) is read from the object
  itself and must match the disassembly's format line. The disassembler prints
  generic AArch64 syntax (GNU objdump, or llvm-objdump --triple=aarch64, not
  Apple's abbreviated printer), so every operand is visible; assembler-level
  controls (macros, conditionals, directives) cannot change or hide audited
  instructions, because only the assembled bytes are read.
- Object image: sections (with executable extents), the complete symbol table
  (raw linker name, binding, type, defining section, value, ELF size) and the
  relocation tables of every executable section (type, raw target symbol,
  addend) are read from the object itself, which must be a relocatable object
  (ELF ET_REL / Mach-O MH_OBJECT). Relocation records are admitted by their
  native fields: complete, bounded ELF RELA records with valid section links,
  an admitted code relocation type and an aligned in-section offset; extern
  Mach-O records whose type has the r_pcrel / r_length combination required by
  mach-o/arm64/reloc.h. Anything else fails before comparison. Function starts and
  extents (ELF symbol size; Mach-O next distinct defined address) must be
  instruction aligned, lie in their section and never overlap; every
  non-padding instruction and every relocation belongs to one.
- Inventory: the expected tile K-loop programs (row tile x integer family x
  policy x encoding/map/minimum) derive from the admitted ISA envelope; the
  inventory string embedded in the object and the object's function symbols
  must agree. Every expected program must exist exactly once; any other tile
  program symbol fails.
- Disassembly contract: every executable section of the object is disassembled
  at every 4-byte address of its extent, in order, exactly once (no gaps,
  missing prefixes or tails, duplicates or reordering); every heading names a
  symbol defined at that address; function bodies are cut from the object's
  extents, not from printed headings. Every printed relocation must equal the
  object's relocation at that instruction (type, target, addend) and every
  object relocation must be printed; the object's record is the one analysed.
  Every line is a format, section, symbol, instruction or relocation line;
  in-function branches become local labels, relocated branches name their
  symbol, page and low-12 data relocations become symbolic operands. Unknown
  relocations, branch addends, unrelocated address operands, data words and
  ELF data mapping symbols in code fail; relocated memory or arithmetic on a
  stack address fails.
- Admitted forms: every instruction must be an admitted (mnemonic, operand
  signature) form: the forms qualified on the GNU-14 and AppleClang-21 witness
  objects plus the explicitly modelled merge definitions. This is an inventory
  of supported forms, not a complete AArch64 grammar; any other form fails.
- Calls: the only admitted call is to the exact fatal-diagnostic symbol (a
  defined function of the object under its raw linker name), whose body must
  consist of admitted forms, call only fwrite/fputs, be acyclic and end every
  path in a call to abort(). fwrite, fputs and abort are trusted only as the
  object's unique, undefined, strongly bound imports in the exact spelling of
  the format's C ABI (_fputs/_abort on Mach-O, fputs/abort on ELF); a foreign
  spelling, local definition, alias, rebinding or weak import disproves the
  diagnostic. Every other call boundary, including a direct
  abort(), is a structural failure, so no interprocedural state is modelled.
- Analysis: each program is analysed over its control-flow graph. Stack
  provenance of every general register is a forward data-flow fact; merge
  definitions (bfi, bfxil, movk) keep their destination as an input; csel joins
  its alternatives; any other computation on a stack address keeps it a stack
  address at an unknown offset; a base that is a stack address on only some
  paths fails. 8-byte frame slots that may hold stack addresses are tracked
  with may/must value sets (including the non-stack alternative); every memory
  access uses its exact opcode/arrangement/lane footprint; partial or
  reinterpreting access to such a slot, any stack access at an unknown offset
  while such slots are live, a 32-bit view of a stack address and a stack
  address leaving the tracked frame (to memory, an FP/SIMD register or an
  index) fail.
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
  exact compilation (-MD on the -c run), cross-checked against a preprocessing
  pass with the same flags; every input is hashed before and after compiling and
  bound with its logical path, its resolved path (component-wise, as the kernel
  resolves symlinks and "..") and the symlinks followed. The Ninja depfile names
  resolved paths; symlink retargeting is caught by verification. Those lists are
  preprocessing dependencies only, and the assembler can read further files or
  construct directives through macros and substitution. Rather than model the
  assembler, the preprocessed translation unit (same flags) is retained and
  C++-tokenized; every asm keyword (including ones nested in operands or
  lambdas) must be an empty-template barrier or a declaration asm label (one
  plain symbol, no qualifiers or operands, after a declarator, outside
  executable code); anything else fails closed. Opaque compiler inputs fail:
  PCH markers, clang module pragmas or remaining include/import/embed
  directives in the translation unit; precompiled headers the GNU compile
  reports reading (-H); PCH, module, plugin, specs, -B, wrapper, config and
  assembler options. The driver's expanded jobs (-###) of the exact compile are
  validated: one integrated clang -cc1 -emit-obj job, or one GNU cc1plus job
  and one as job whose only input is the compiler's output, no specs or
  configuration file read. Every job program is hashed into the proof.

Verify phase (Unit/preflight): no compiler is invoked, but every recorded input
(including absolute system/SDK headers named by the compiler) must be present
and unchanged, and the bound object is re-disassembled by the bound disassembler
(hashed) and must reproduce the retained disassembly; verification is therefore
tied to the build host's toolchain and binutils, not an SDK-independent or
installed-runner qualification. The actual target command must match and the
object must re-audit to the recorded counts. Probe paths must be absolute.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
from typing import NamedTuple
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

# Admitted instruction forms: (mnemonic, operand signature) pairs qualified on the GNU-14 and AppleClang-21
# witnesses (all build types and envelopes) plus the forms the semantics below model explicitly. Operand kinds:
# x/w general registers, b/h/s/d/q scalar FP views, v vector, v[] element, list / list[] register lists (lane),
# mem addressing, imm, shift, cond, sym label, reloc relocated symbol. Any other form is a structural failure.
FORMS = frozenset(line.strip() for line in """
add v,v,v|add w,w,imm|add w,w,w|add x,x,imm|add x,x,imm,shift|add x,x,reloc|add x,x,x|add x,x,x,shift
addp v,v,v|adrp x,reloc|adrp x,sym|and v,v,v|and w,w,imm|and x,x,imm|asr w,w,imm|b sym|b.cond sym|bic v,imm
bic v,v,v|bics w,w,w|bif v,v,v|bit v,v,v|bfi w,w,imm,imm|bfi x,x,imm,imm|bfxil x,x,imm,imm|bl sym|bsl v,v,v
cbnz w,sym|cbnz x,sym|cbz w,sym|cbz x,sym|ccmp w,w,imm,cond|cmeq v,v,imm|cmeq v,v,v|cmhi v,v,v|cmn w,imm
cmp w,imm|cmp w,w|cmp w,w,shift|cmp x,imm|cmp x,x|csel w,w,w,cond|csel x,x,x,cond|cset x,cond|dup d,v[]
dup v,v[]|dup v,w|dup v,x|eor x,x,x|ext v,v,v,imm|fadd v,v,v|fcvtl v,v|fmla v,v,v|fmov d,d|fmov d,x|fmov s,s
fmov s,w|fmov w,s|fmov x,d|fmul s,s,s|fmul v,v,v|fmul v,v,v[]|fneg v,v|ins v[],v[]|ins v[],x|ld1 list,mem
ld1 list,mem,imm|ld1 list[],mem|ld1 list[],mem,imm|ld1r list,mem|ld1r list,mem,imm|ld2 list,mem|ld2 list[],mem
ld2r list,mem|ldp d,d,mem|ldp d,d,mem,imm|ldp q,q,mem|ldp q,q,mem,imm|ldp s,s,mem|ldp w,w,mem|ldp x,x,mem
ldp x,x,mem,imm|ldpsw x,x,mem|ldr b,mem|ldr d,mem|ldr d,mem,imm|ldr h,mem|ldr q,mem|ldr q,mem,imm|ldr s,mem
ldr w,mem|ldr x,mem|ldrb w,mem|ldrh w,mem|ldrsh w,mem|ldrsw x,mem|ldur d,mem|ldur q,mem|lsl w,w,imm|lsl x,x,imm
lsr x,x,imm|lsr x,x,x|madd x,x,x,x|mov d,v[]|mov v,v|mov v[],v[]|mov w,imm|mov w,w|mov x,imm|mov x,x|movi d,imm|movi v,imm
movi v,imm,shift|movk x,imm,shift|mul w,w,w|mul x,x,x|mvni v,imm|mvni v,imm,shift|nop|orr v,imm|orr v,v,v|orr w,w,imm
orr w,w,w|orr w,w,w,shift|orr x,x,imm|orr x,x,x,shift|ret|saddlp v,v|sbfiz x,x,imm,imm|scvtf s,w|scvtf v,v
sdot v,v,v|shl v,v,imm|smaddl x,w,w,x|smmla v,v,v|smull v,v,v|smull x,w,w|smull2 v,v,v|sshll v,v,imm|sshr v,v,imm
st1 list,mem|st1 list,mem,imm|st1 list[],mem|stp d,d,mem|stp q,q,mem|stp w,w,mem|stp x,x,mem|str b,mem|str d,mem|str d,mem,imm|str q,mem,imm
str h,mem|str q,mem|str s,mem|str w,mem|str x,mem|stur d,mem|stur q,mem|sub v,v,v|sub w,w,imm|sub w,w,w
sub x,x,imm|sub x,x,imm,shift|sub x,x,x|sxtw x,w|tbl v,list,v|tbnz w,imm,sym|tbnz x,imm,sym|tbz w,imm,sym
uaddw v,v,v|ubfiz w,w,imm,imm|ubfx w,w,imm,imm|ubfx x,x,imm,imm|ucvtf v,v|umaddl x,w,w,x|umull x,w,w|ushll v,v,imm|ushr v,v,imm
usra v,v,imm|uxtw x,w|zip1 v,v,v|zip2 v,v,v
""".replace("\n", "|").split("|") if line.strip())
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
MERGE_DEFINITIONS = frozenset({"bfi", "bfxil", "movk"})
CONDITIONAL_SELECTS = frozenset({"csel"})  # csinc/csinv/csneg transform one alternative: not admitted.
STRUCTURE_ACCESSES = re.compile(r"^(ld|st)([1-4])(r?)$")
# Trusted calls of the fatal diagnostic body (they return; the path must still reach abort()).
DIAGNOSTIC_CALLS = frozenset({"fwrite", "fputs"})

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
    """Syntax of the normalised disassembly the analysis reads (generic AArch64 syntax, ELF-style labels)."""

    def local(self, label: str) -> bool:
        """Local labels (branch targets inside one function) stay inside the enclosing function."""
        return label.startswith(".L")

    def symbol(self, name: str) -> str:
        return name

    def statements(self, line: str) -> list[str]:
        return [line]


GENERIC = Dialect()
ENTRY_SECTION = re.compile(r'^\t\.section \.text\.\S+,"ax"$')

# Disassembly front end. The witness analyses the instructions actually assembled into the probe object, as
# printed by the configured disassembler in generic AArch64 syntax (GNU objdump, or llvm-objdump with
# --triple=aarch64, which does not use Apple's abbreviated printer). Assembler-level controls (macros,
# conditionals, directives) therefore cannot change or hide the audited instructions.
FORMAT_LINE = re.compile(r"^\S.*:\s+file format (\S.*?)\s*$")
SECTION_LINE = re.compile(r"^Disassembly of section (\S+):$")
SYMBOL_LINE = re.compile(r"^([0-9a-f]+) <(\S+)>:$")
RELOCATION_LINE = re.compile(r"^\s+([0-9a-f]+):\s+((?:R_AARCH64|ARM64_RELOC)_\w+)\s+(\S+)\s*$")
INSTRUCTION_LINE = re.compile(r"^\s+([0-9a-f]+):\s+(\S.*?)\s*$")
ADDRESS_OPERAND = re.compile(r"^(?:0x)?([0-9a-f]+)\s+<(\S+)>$")
BRANCH_RELOCATIONS = frozenset({"R_AARCH64_CALL26", "R_AARCH64_JUMP26", "ARM64_RELOC_BRANCH26"})
PAGE_RELOCATIONS = frozenset({"R_AARCH64_ADR_PREL_PG_HI21", "R_AARCH64_ADR_GOT_PAGE", "ARM64_RELOC_PAGE21",
                              "ARM64_RELOC_GOT_LOAD_PAGE21"})
LOW12_RELOCATIONS = frozenset({"R_AARCH64_ADD_ABS_LO12_NC", "R_AARCH64_LDST8_ABS_LO12_NC",
                               "R_AARCH64_LDST16_ABS_LO12_NC", "R_AARCH64_LDST32_ABS_LO12_NC",
                               "R_AARCH64_LDST64_ABS_LO12_NC", "R_AARCH64_LDST128_ABS_LO12_NC",
                               "R_AARCH64_LD64_GOT_LO12_NC", "ARM64_RELOC_PAGEOFF12", "ARM64_RELOC_GOT_LOAD_PAGEOFF12"})
OBJECT_FORMATS = {"elf64-littleaarch64": False, "mach-o arm64": True}


def object_format(data: bytes) -> bool:
    """True for a Mach-O arm64 object, False for an ELF64 AArch64 object; anything else fails."""
    if data[:4] == b"\x7fELF" and data[4] == 2 and data[5] == 1 and int.from_bytes(data[18:20], "little") == 183:
        return False
    if data[:4] == bytes.fromhex("cffaedfe") and int.from_bytes(data[4:8], "little") == 0x0100000C:
        return True
    raise ValueError("witness object is neither an ELF64 AArch64 nor a Mach-O arm64 object")


def object_inventory(data: bytes) -> list[str]:
    """The probe's instantiation inventory strings embedded in the object."""
    marker = b"llaminar-native-neon-tile-probe:"
    found, start = [], data.find(marker)
    while start >= 0:
        end = data.find(b"\0", start)
        found.append(data[start:end if end >= 0 else len(data)].decode("ascii", "replace"))
        start = data.find(marker, start + 1)
    return [entry[len(marker.decode()):] for entry in found]


class Section(NamedTuple):
    name: str
    address: int
    size: int
    executable: bool


class ObjectSymbol(NamedTuple):
    """One symbol-table entry. kind: function, object, notype, section, file, undefined, absolute, common,
    indirect or other; binding: local, global or weak; size is None where the format records none (Mach-O)."""
    name: str
    section: str | None
    value: int
    size: int | None
    kind: str
    binding: str


class ObjectImage(NamedTuple):
    """Sections, the symbol table (raw linker names) and the relocations of every executable section
    ({section: {address: (type name, raw target symbol, addend)}}), all read from the object."""
    macho: bool
    sections: dict[str, Section]
    symbols: list[ObjectSymbol]
    relocations: dict[str, dict[int, tuple[str, str, int]]]

    @property
    def prefix(self) -> str:
        """The C linker-name prefix of the object format (Mach-O prepends an underscore)."""
        return "_" if self.macho else ""


ELF_RELOCATIONS = {257: "R_AARCH64_ABS64", 275: "R_AARCH64_ADR_PREL_PG_HI21", 277: "R_AARCH64_ADD_ABS_LO12_NC",
                   278: "R_AARCH64_LDST8_ABS_LO12_NC", 279: "R_AARCH64_TSTBR14", 280: "R_AARCH64_CONDBR19",
                   282: "R_AARCH64_JUMP26", 283: "R_AARCH64_CALL26", 284: "R_AARCH64_LDST16_ABS_LO12_NC",
                   285: "R_AARCH64_LDST32_ABS_LO12_NC", 286: "R_AARCH64_LDST64_ABS_LO12_NC",
                   299: "R_AARCH64_LDST128_ABS_LO12_NC", 311: "R_AARCH64_ADR_GOT_PAGE",
                   312: "R_AARCH64_LD64_GOT_LO12_NC"}
MACHO_RELOCATIONS = {0: "ARM64_RELOC_UNSIGNED", 1: "ARM64_RELOC_SUBTRACTOR", 2: "ARM64_RELOC_BRANCH26",
                     3: "ARM64_RELOC_PAGE21", 4: "ARM64_RELOC_PAGEOFF12", 5: "ARM64_RELOC_GOT_LOAD_PAGE21",
                     6: "ARM64_RELOC_GOT_LOAD_PAGEOFF12", 7: "ARM64_RELOC_POINTER_TO_GOT",
                     8: "ARM64_RELOC_TLVP_LOAD_PAGE21", 9: "ARM64_RELOC_TLVP_LOAD_PAGEOFF12", 10: "ARM64_RELOC_ADDEND"}


# Relocations admitted on instructions, and (for Mach-O, from mach-o/arm64/reloc.h) their required
# r_pcrel / r_length (2 = 4 bytes).
CODE_RELOCATIONS = frozenset({"R_AARCH64_CALL26", "R_AARCH64_JUMP26", "R_AARCH64_CONDBR19", "R_AARCH64_TSTBR14",
                              "R_AARCH64_ADR_PREL_PG_HI21", "R_AARCH64_ADR_GOT_PAGE", "R_AARCH64_ADD_ABS_LO12_NC",
                              "R_AARCH64_LDST8_ABS_LO12_NC", "R_AARCH64_LDST16_ABS_LO12_NC",
                              "R_AARCH64_LDST32_ABS_LO12_NC", "R_AARCH64_LDST64_ABS_LO12_NC",
                              "R_AARCH64_LDST128_ABS_LO12_NC", "R_AARCH64_LD64_GOT_LO12_NC"})
MACHO_CODE_RELOCATIONS = {"ARM64_RELOC_BRANCH26": (1, 2), "ARM64_RELOC_PAGE21": (1, 2),
                          "ARM64_RELOC_PAGEOFF12": (0, 2), "ARM64_RELOC_GOT_LOAD_PAGE21": (1, 2),
                          "ARM64_RELOC_GOT_LOAD_PAGEOFF12": (0, 2)}


def _add_relocation(table: dict, section: str, address: int, record: tuple[str, str, int]) -> None:
    if address in table.setdefault(section, {}):
        raise ValueError(f"two relocations at {section}+{address:#x}")
    table[section][address] = record


def _field(data: bytes, offset: int, size: int) -> int:
    if offset < 0 or offset + size > len(data):
        raise ValueError("truncated witness object")
    return int.from_bytes(data[offset:offset + size], "little")


def _string(data: bytes, start: int, limit: int) -> str:
    end = data.find(b"\0", start, limit)
    if start < 0 or start >= limit or end < 0:
        raise ValueError("witness object string table entry out of bounds")
    return data[start:end].decode("utf-8", "replace")


def read_object(data: bytes) -> ObjectImage:
    """Sections (with executable extents) and the complete symbol table, read from the object itself."""
    return _read_macho(data) if object_format(data) else _read_elf(data)


def _read_elf(data: bytes) -> ObjectImage:
    if _field(data, 0x10, 2) != 1 or _field(data, 0x14, 4) != 1 or data[6] != 1:
        raise ValueError("the witness object is not an ELF relocatable object (ET_REL, version 1)")
    shoff, shentsize, shnum, shstrndx = (_field(data, 0x28, 8), _field(data, 0x3A, 2), _field(data, 0x3C, 2),
                                         _field(data, 0x3E, 2))
    if shentsize != 64 or not 0 < shstrndx < shnum:
        raise ValueError("malformed ELF section header table")
    headers = []
    for index in range(shnum):
        base = shoff + index * 64
        headers.append({"name": _field(data, base, 4), "type": _field(data, base + 4, 4),
                        "flags": _field(data, base + 8, 8), "address": _field(data, base + 16, 8),
                        "offset": _field(data, base + 24, 8), "size": _field(data, base + 32, 8),
                        "link": _field(data, base + 40, 4), "info": _field(data, base + 44, 4),
                        "entsize": _field(data, base + 56, 8)})
    strings = headers[shstrndx]
    names = [_string(data, strings["offset"] + header["name"], strings["offset"] + strings["size"])
             for header in headers]
    sections: dict[str, Section] = {}
    for index, header in enumerate(headers[1:], start=1):
        executable = bool(header["flags"] & 0x4)
        if names[index] in sections and (executable or sections[names[index]].executable):
            raise ValueError(f"ambiguous executable section name {names[index]}")
        sections.setdefault(names[index], Section(names[index], header["address"], header["size"], executable))
    tables = [header for header in headers if header["type"] == 2]
    if len(tables) != 1 or tables[0]["entsize"] != 24:
        raise ValueError("the witness object must have exactly one ELF symbol table")
    table, symbols = tables[0], []
    names_table = headers[table["link"]]
    kinds = {0: "notype", 1: "object", 2: "function", 3: "section", 4: "file"}
    indexed: list[str] = [""]
    for entry in range(1, table["size"] // 24):
        base = table["offset"] + entry * 24
        name = _string(data, names_table["offset"] + _field(data, base, 4),
                       names_table["offset"] + names_table["size"])
        if _field(data, base + 4, 1) & 0xF == 3 and 0 < _field(data, base + 6, 2) < len(headers):
            name = names[_field(data, base + 6, 2)]  # A section symbol is named by its section.
        indexed.append(name)
        info, index = _field(data, base + 4, 1), _field(data, base + 6, 2)
        binding = {0: "local", 1: "global", 2: "weak"}.get(info >> 4, "other")
        if index == 0:
            kind, section = "undefined", None
        elif index == 0xFFF1:
            kind, section = "absolute", None
        elif index == 0xFFF2:
            kind, section = "common", None
        elif index < len(headers):
            kind, section = kinds.get(info & 0xF, "other"), names[index]
        else:
            raise ValueError(f"symbol {name} names a nonexistent section")
        symbols.append(ObjectSymbol(name, section, _field(data, base + 8, 8), _field(data, base + 16, 8), kind, binding))
    relocations: dict[str, dict[int, tuple[str, str, int]]] = {}
    for number, header in enumerate(headers):
        if header["type"] not in (4, 9):
            continue
        if not 0 < header["info"] < len(headers) or not 0 < header["link"] < len(headers):
            raise ValueError(f"relocation section {names[number]} has an invalid section reference")
        target = headers[header["info"]]
        if not target["flags"] & 0x4:
            continue  # Relocations of data sections are not analysed.
        if header["type"] == 9 or header["entsize"] != 24 or headers[header["link"]] is not table:
            raise ValueError("executable-section relocations must be RELA entries against the symbol table")
        if header["size"] % 24 or header["offset"] + header["size"] > len(data):
            raise ValueError(f"relocation section {names[number]} holds a partial or truncated record")
        section = names[header["info"]]
        for entry in range(header["size"] // 24):
            base = header["offset"] + entry * 24
            offset, info = _field(data, base, 8), _field(data, base + 8, 8)
            if not 0 < info >> 32 < len(indexed):
                raise ValueError(f"relocation in {section} names no symbol")
            kind = ELF_RELOCATIONS.get(info & 0xFFFFFFFF)
            if kind not in CODE_RELOCATIONS:
                raise ValueError(f"relocation type {info & 0xFFFFFFFF} is not admitted in code ({section})")
            if offset % 4 or offset >= target["size"]:
                raise ValueError(f"relocation offset {offset:#x} is not an instruction of {section}")
            addend = _field(data, base + 16, 8)
            addend -= (1 << 64) if addend >= 1 << 63 else 0
            _add_relocation(relocations, section, target["address"] + offset, (kind, indexed[info >> 32], addend))
    return ObjectImage(False, sections, symbols, relocations)


def _read_macho(data: bytes) -> ObjectImage:
    if _field(data, 12, 4) != 1:
        raise ValueError("the witness object is not a Mach-O relocatable object (MH_OBJECT)")
    commands, offset = _field(data, 16, 4), 32
    sections: dict[str, Section] = {}
    ordered: list[str | None] = [None]
    relocation_tables: dict[str, tuple[int, int]] = {}
    symtab = None
    for _ in range(commands):
        command, size = _field(data, offset, 4), _field(data, offset + 4, 4)
        if size < 8:
            raise ValueError("malformed Mach-O load command")
        if command == 0x19:  # LC_SEGMENT_64
            for index in range(_field(data, offset + 64, 4)):
                base = offset + 72 + index * 80
                section = data[base:base + 16].rstrip(b"\0").decode()
                segment = data[base + 16:base + 32].rstrip(b"\0").decode()
                name = f"{segment},{section}"
                flags = _field(data, base + 64, 4)
                if name in sections:
                    raise ValueError(f"ambiguous Mach-O section name {name}")
                sections[name] = Section(name, _field(data, base + 32, 8), _field(data, base + 40, 8),
                                         bool(flags & 0x80000400))
                relocation_tables[name] = (_field(data, base + 56, 4), _field(data, base + 60, 4))
                ordered.append(name)
        elif command == 0x2:  # LC_SYMTAB
            symtab = tuple(_field(data, offset + 8 + 4 * field, 4) for field in range(4))
        elif command == 0x29 and _field(data, offset + 12, 4):  # LC_DATA_IN_CODE with entries
            raise ValueError("the witness object marks data inside code")
        offset += size
    if symtab is None:
        raise ValueError("the witness object has no Mach-O symbol table")
    symoff, count, stroff, strsize = symtab
    symbols, indexed = [], []
    for index in range(count):
        base = symoff + index * 16
        name = _string(data, stroff + _field(data, base, 4), stroff + strsize)
        indexed.append(name)
        kind_bits, number, description = _field(data, base + 4, 1), _field(data, base + 5, 1), _field(data, base + 6, 2)
        value = _field(data, base + 8, 8)
        if kind_bits & 0xE0:
            continue  # Debugger (stab) entry.
        section = None
        if kind_bits & 0x0E == 0x0:
            kind = "common" if value else "undefined"
        elif kind_bits & 0x0E == 0xE:
            if not 0 < number < len(ordered):
                raise ValueError(f"symbol {name} names a nonexistent section")
            section = ordered[number]
            kind = "function" if sections[section].executable else "object"
        else:
            kind = {0x2: "absolute", 0xA: "indirect"}.get(kind_bits & 0x0E, "other")
        binding = "weak" if description & 0xC0 else ("global" if kind_bits & 0x01 else "local")
        symbols.append(ObjectSymbol(name, section, value, None, kind, binding))
    relocations: dict[str, dict[int, tuple[str, str, int]]] = {}
    for name, (offset, count) in relocation_tables.items():
        if not sections[name].executable:
            continue
        for entry in range(count):
            address, info = _field(data, offset + entry * 8, 4), _field(data, offset + entry * 8 + 4, 4)
            symbol, pcrel, length = info & 0xFFFFFF, (info >> 24) & 1, (info >> 25) & 3
            external, kind = (info >> 27) & 1, info >> 28
            name_of = MACHO_RELOCATIONS.get(kind)
            if address & 0x80000000 or not external or not symbol < len(indexed):
                raise ValueError(f"unsupported relocation form in {name} (scattered or section-relative)")
            if name_of not in MACHO_CODE_RELOCATIONS or (pcrel, length) != MACHO_CODE_RELOCATIONS[name_of]:
                raise ValueError(f"relocation {name_of or kind} with pcrel={pcrel} length={length} is not admitted "
                                 f"in code ({name})")
            if address % 4 or address >= sections[name].size:
                raise ValueError(f"relocation address {address:#x} is not an instruction of {name}")
            _add_relocation(relocations, name, sections[name].address + address, (name_of, indexed[symbol], 0))
    return ObjectImage(True, sections, symbols, relocations)


def function_extents(image: ObjectImage) -> dict[str, tuple[str, int, int]]:
    """(section, start, end) of every function symbol defined in an executable section.

    ELF extents are the recorded symbol sizes; Mach-O records none, so a symbol extends to the next distinct
    defined address in its section (or the section end). Extents must lie inside their section; distinct
    function extents in one section must not overlap; a name may be defined only once.
    """
    extents: dict[str, tuple[str, int, int]] = {}
    for symbol in image.symbols:
        # ELF notype labels (mapping symbols) and section symbols are not functions.
        if symbol.section is None or not image.sections[symbol.section].executable or symbol.kind != "function":
            continue
        section = image.sections[symbol.section]
        if image.macho:
            later = sorted({other.value for other in image.symbols if other.section == symbol.section
                            and other.value > symbol.value})
            end = later[0] if later else section.address + section.size
        else:
            if not symbol.size:
                raise ValueError(f"function symbol {symbol.name} records no size")
            end = symbol.value + symbol.size
        if symbol.value % 4 or end % 4:
            raise ValueError(f"function symbol {symbol.name} has an unaligned start or extent")
        if not section.address <= symbol.value < end <= section.address + section.size:
            raise ValueError(f"function symbol {symbol.name} lies outside its section")
        if symbol.name in extents:
            raise ValueError(f"symbol defined twice in the witness: {symbol.name}")
        extents[symbol.name] = (symbol.section, symbol.value, end)
    spans = sorted(set(extents.values()))
    for (section, start, end), (other, other_start, other_end) in zip(spans, spans[1:]):
        if section == other and other_start < end:
            raise ValueError(f"function extents overlap in {section} at {other_start:#x}")
    return extents


def trusted_imports(image: ObjectImage) -> set[str]:
    """Raw linker names of the stdio/abort exceptions bound as the object's unique, undefined, strongly bound
    imports, in the exact spelling of the object format's C ABI (abort on ELF, _abort on Mach-O)."""
    names = [symbol.name for symbol in image.symbols]
    wanted = {image.prefix + name for name in DIAGNOSTIC_CALLS | NORETURN_LEAVES}
    return {symbol.name for symbol in image.symbols if symbol.name in wanted and symbol.kind == "undefined"
            and symbol.binding == "global" and names.count(symbol.name) == 1}


def parse_disassembly(disassembly: str, macho: bool) -> tuple[dict[str, dict[int, list]], list[tuple[str, int, str]]]:
    """Instruction map per section {address: [text, printed relocation]} and (section, address, raw name)
    headings."""
    def symbol(name: str) -> str:
        return name

    lines = disassembly.splitlines()
    formats = [FORMAT_LINE.match(line).group(1) for line in lines if FORMAT_LINE.match(line)]
    if len(formats) != 1 or OBJECT_FORMATS.get(formats[0]) != macho:
        raise ValueError(f"disassembly object format {formats} does not match the witness object")
    code: dict[str, dict[int, list]] = {}
    headings: list[tuple[str, int, str]] = []
    section, last = None, None
    for line in lines:
        if not line.strip() or FORMAT_LINE.match(line):
            continue
        if SECTION_LINE.match(line):
            section, last = SECTION_LINE.match(line).group(1), None
            if section in code:
                raise ValueError(f"section disassembled twice: {section}")
            code[section] = {}
            continue
        if SYMBOL_LINE.match(line):
            if section is None:
                raise ValueError(f"disassembly symbol outside a section: {line}")
            address, name = SYMBOL_LINE.match(line).groups()
            headings.append((section, int(address, 16), symbol(name)))
            continue
        relocation = RELOCATION_LINE.match(line)
        if relocation:
            if last is None or last != int(relocation.group(1), 16):
                raise ValueError(f"relocation does not follow its instruction: {line.strip()}")
            if code[section][last][1] is not None:
                raise ValueError(f"instruction carries two relocations: {line.strip()}")
            code[section][last][1] = (relocation.group(2), symbol(relocation.group(3)))
            continue
        instruction = INSTRUCTION_LINE.match(line)
        if instruction and section is not None and headings and headings[-1][0] == section:
            address = int(instruction.group(1), 16)
            if last is not None and address <= last:
                raise ValueError(f"disassembly addresses out of order in {section} at {address:#x}")
            code[section][address] = [instruction.group(2).split("//", 1)[0].strip(), None]
            last = address
            continue
        raise ValueError(f"unparseable disassembly line: {line.strip()}")
    return code, headings


def relocation_target(symbol: str, addend: int) -> str:
    """The relocation target as disassemblers print it: symbol, symbol+0xN or symbol-0xN."""
    return symbol if not addend else f"{symbol}{'+' if addend > 0 else '-'}0x{abs(addend):x}"


def normalize_disassembly(disassembly: str, image: ObjectImage) -> dict[str, list[str]]:
    """Translate disassembler output into one normalised body per function symbol of the object.

    Coverage is complete and bound to the object: every executable section is disassembled at every 4-byte
    address of its extent, in order, exactly once; every heading names a symbol defined at that address; and
    each function body is cut from its object symbol extent, not from the printed headings. Branch targets
    inside the function become local labels; relocated branches name their relocation symbol; page/low-12
    relocations become symbolic operands; ELF data mapping symbols inside a function mark data in code.
    """
    code, headings = parse_disassembly(disassembly, image.macho)
    for name, section in image.sections.items():
        if not section.executable:
            if name in code:
                raise ValueError(f"disassembly covers a non-executable section: {name}")
            continue
        if section.size % 4 or section.address % 4:
            raise ValueError(f"executable section {name} is not instruction aligned")
        expected = list(range(section.address, section.address + section.size, 4))
        if sorted(code.get(name, {})) != expected:
            missing = sorted(set(expected) - set(code.get(name, {})))
            raise ValueError(f"disassembly does not cover executable section {name} exactly "
                             f"(missing {[hex(address) for address in missing[:3]]})")
    for name in code:
        if name not in image.sections:
            raise ValueError(f"disassembly names a section the object does not define: {name}")
    defined = {(symbol.section, symbol.value, symbol.name) for symbol in image.symbols if symbol.section}
    for heading in headings:
        if heading not in defined:
            raise ValueError(f"disassembly heading is not a symbol defined at that address: {heading[2]}")
    for name, table in image.relocations.items():
        for address in table:
            if address not in code.get(name, {}):
                raise ValueError(f"object relocation at {name}+{address:#x} is not on a disassembled instruction")
    for name, instructions in code.items():
        table = image.relocations.get(name, {})
        for address, (_, printed) in instructions.items():
            actual = table.get(address)
            if (printed is None) != (actual is None) or \
                    (actual and printed != (actual[0], relocation_target(actual[1], actual[2]))):
                raise ValueError(f"printed relocation at {name}+{address:#x} does not match the object: "
                                 f"{printed} != {actual}")
    extents = function_extents(image)
    owned = {(section, address) for section, start, end in extents.values() for address in range(start, end, 4)}
    for name, table in image.relocations.items():
        orphans = [address for address in table if (name, address) not in owned]
        if orphans:
            raise ValueError(f"relocation outside every function extent in {name} at {orphans[0]:#x}")
    for name, instructions in code.items():
        orphans = [address for address, (text, _) in instructions.items()
                   if (name, address) not in owned and text != "nop"]
        if orphans:
            raise ValueError(f"code outside every function extent in {name} at {orphans[0]:#x}")
    starts: dict[tuple[str, int], str] = {}
    for name, (section, start, _) in sorted(extents.items()):
        starts.setdefault((section, start), name)
    data_marks = {(symbol.section, symbol.value) for symbol in image.symbols
                  if not image.macho and re.match(r"^\$d(?:\.|$)", symbol.name)}
    bodies: dict[str, list[str]] = {}
    for name, (section, start, end) in extents.items():
        function = {"section": section, "start": start}
        rewritten, targets = [], set()
        for address in range(start, end, 4):
            text = code[section][address][0]
            relocation = image.relocations.get(section, {}).get(address)
            rewritten.append((address, rewrite(text, relocation, function, end, starts, targets)))
        body = [f'\t.section .text.{section.replace(",", ".")},"ax"']
        for address, text in rewritten:
            if address in targets:
                body.append(f".L{address:x}:")
            if (section, address) in data_marks:
                body.append("\t.data-mapping-symbol")
            body.append(f"\t{text}")
        bodies[name] = body
    return bodies


def rewrite(text: str, relocation: tuple[str, str, int] | None, function: dict, end: int,
            starts: dict[tuple[str, int], str], targets: set[int]) -> str:
    """One disassembled instruction in normalised form (symbolic branch targets and relocations)."""
    parts = text.split(None, 1)
    mnemonic, operands = parts[0], (split_operands(parts[1]) if len(parts) > 1 else [])
    branch = mnemonic in ("b", "bl", "cbz", "cbnz", "tbz", "tbnz") or CONDITIONAL_BRANCH.match(mnemonic)
    kind, name, addend = relocation if relocation else (None, None, 0)
    if relocation and addend and kind in BRANCH_RELOCATIONS:
        raise ValueError(f"branch relocation with an addend is not supported: {text} {name}{addend:+#x}")
    if branch:
        target = ADDRESS_OPERAND.match(operands[-1]) if operands else None
        if not target:
            raise ValueError(f"unparseable branch target: {text}")
        if relocation:
            if kind not in BRANCH_RELOCATIONS or mnemonic not in ("b", "bl"):
                raise ValueError(f"unsupported branch relocation: {text} {kind}")
            operands[-1] = name
        else:
            address = int(target.group(1), 16)
            if function["start"] <= address < end:
                targets.add(address)
                operands[-1] = f".L{address:x}"
            else:
                operands[-1] = starts.get((function["section"], address), f".Loutside{address:x}")
    elif relocation and kind in PAGE_RELOCATIONS and mnemonic == "adrp" and len(operands) == 2 \
            and ADDRESS_OPERAND.match(operands[1]):
        operands[1] = name
    elif relocation and kind in LOW12_RELOCATIONS:
        memory = next((index for index, operand in enumerate(operands) if operand.startswith("[")), None)
        if memory is not None:
            inner = MEMORY.match(operands[memory])
            if not inner or (inner.group(2) and immediate(inner.group(2)) != 0) or inner.group(3):
                raise ValueError(f"unsupported relocated addressing: {text}")
            operands[memory] = f"[{inner.group(1)}, :lo12:{name}]"
        elif mnemonic == "add" and len(operands) == 3 and immediate(operands[2]) == 0:
            operands[2] = f":lo12:{name}"
        else:
            raise ValueError(f"unsupported low-12 relocation: {text}")
    elif relocation:
        raise ValueError(f"unsupported relocation {kind} on {text}")
    elif any(ADDRESS_OPERAND.match(operand) for operand in operands):
        raise ValueError(f"unrelocated address operand: {text}")
    return f"{mnemonic}\t{', '.join(operands)}" if operands else mnemonic


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
        return "x"
    if GP32.match(operand):
        return "w"
    if operand.startswith("["):
        return "mem"
    if operand.startswith("{"):
        try:
            fp_registers(operand)
        except ValueError:
            return None
        return "list[]" if operand.endswith("]") else "list"
    if re.match(r"^[bhsdq](?:[12]?[0-9]|3[01])$", operand):
        return operand[0]
    if re.match(r"^v(?:[12]?[0-9]|3[01])(?:\.\d+[bhsd])?$", operand):
        return "v"
    if re.match(r"^v(?:[12]?[0-9]|3[01])(?:\.[bhsd])?\[\d+\]$", operand):
        return "v[]"
    if IMMEDIATE.match(operand) or FLOAT.match(operand):
        return "imm"
    if SHIFT.match(operand):
        return "shift"
    if CONDITION.match(operand):
        return "cond"
    if RELOCATION.match(operand):
        return "reloc"
    if SYMBOL.match(operand):
        return "sym"
    return None


class Instruction:
    """One parsed machine instruction."""

    def __init__(self, text: str, dialect: Dialect) -> None:
        self.text = text.strip()
        parts = self.text.split(None, 1)
        mnemonic = parts[0].lower()
        self.operands = split_operands(parts[1]) if len(parts) > 1 else []
        self.base = "b.cond" if CONDITIONAL_BRANCH.match(mnemonic) else mnemonic
        self.memory = next((index for index, operand in enumerate(self.operands) if operand.startswith("[")), None)
        kinds = [operand_kind(operand) for operand in self.operands]
        self.form = None if None in kinds else f"{self.base} {','.join(kinds)}".strip()

    def admitted(self) -> bool:
        return self.form in FORMS


def footprint(instruction: Instruction, operand: str) -> int | None:
    """Bytes of memory one data operand of a load/store transfers; None if not exactly known."""
    base = instruction.base
    scalar = re.match(r"^([bhsdq])\d+$", operand)
    if scalar and base in ("ldr", "ldur", "ldp", "str", "stur", "stp"):
        return WIDTH[scalar.group(1)]
    if gp(operand) or GP32.match(operand):
        return memory_width(base, operand)
    structure = STRUCTURE_ACCESSES.match(base)
    if not structure or not operand.startswith("{"):
        return None
    count, replicate = int(structure.group(2)), bool(structure.group(3))
    lane = re.search(r"\]\s*$", operand) is not None and not operand.rstrip().endswith("}")
    registers = [element.strip() for element in operand[1:operand.rindex("}")].split(",")]
    if any("-" in element for element in registers) or len(registers) != (count if count > 1 else len(registers)):
        return None
    shapes = {re.sub(r"^v\d+", "", element) for element in registers}
    if len(shapes) != 1:
        return None
    shape = shapes.pop().lstrip(".")
    arrangement = re.match(r"^(\d*)([bhsd])$", shape)
    if not arrangement:
        return None
    element = WIDTH[arrangement.group(2)]
    if lane or replicate:
        if arrangement.group(1) and not replicate:
            return None
        return len(registers) * element
    if not arrangement.group(1):
        return None
    return len(registers) * int(arrangement.group(1)) * element


class Block:
    def __init__(self) -> None:
        self.labels: list[str] = []
        self.instructions: list[Instruction] = []
        self.successors: list[int] = []
        self.falls = True


def build_cfg(body: list[str], dialect: Dialect) -> tuple[list[Block], list[str]]:
    """Basic blocks of one normalised function body; data words or directives inside code fail."""
    blocks: list[Block] = [Block()]
    if not body or not ENTRY_SECTION.match(body[0]):
        return blocks, ["function body has no executable entry section"]
    errors: list[str] = []
    for line in body[1:]:
        label = LABEL.match(line)
        if label:
            if blocks[-1].instructions:
                blocks.append(Block())
            blocks[-1].labels.append(label.group(1))
            line = label.group(2)
        for statement in dialect.statements(line):
            statement = statement.strip()
            if not statement:
                continue
            if statement.startswith("."):
                errors.append(f"data or directive inside code: {statement}")
                continue
            instruction = Instruction(statement, dialect)
            blocks[-1].instructions.append(instruction)
            if instruction.base in BRANCHES or INDIRECT.match(instruction.base) or RETURN.match(instruction.base):
                blocks.append(Block())  # Every branch ends its block.
    return blocks, errors


class State:
    """Per-program-point facts: general register stack provenance, pointer slots, ABI saves.

    pointers maps the offset of each 8-byte frame slot that may hold a stack address to the set of values it
    may hold, including "N" when a non-stack value reaches the slot on some path (may/must information).
    """

    def __init__(self, registers: dict[str, frozenset], pointers: dict[int, frozenset],
                 saved: dict[int, frozenset]) -> None:
        self.registers, self.pointers, self.saved = registers, pointers, saved

    @staticmethod
    def entry() -> "State":
        """Function entry: sp at offset 0 of the new frame, x29 the caller's frame pointer."""
        registers = {"sp": frozenset({("S", 0)}), "x29": UNKNOWN_STACK}
        return State(registers, {}, {n: frozenset({("U", None)}) for n in CALLEE_SAVED})

    def copy(self) -> "State":
        return State(dict(self.registers), dict(self.pointers), dict(self.saved))

    def value(self, register: str) -> frozenset:
        return self.registers.get(register, NOT_STACK) if register != "xzr" else NOT_STACK

    def assign(self, register: str, values: frozenset) -> None:
        if register == "xzr":
            return
        if values == NOT_STACK:
            self.registers.pop(register, None)
        else:
            self.registers[register] = widen(values)

    def slot(self, offset: int) -> frozenset:
        return self.pointers.get(offset, NOT_STACK)

    def store_slot(self, offset: int, values: frozenset) -> None:
        if values == NOT_STACK:
            self.pointers.pop(offset, None)
        else:
            self.pointers[offset] = widen(values)

    def join(self, other: "State") -> "State":
        registers = {register: widen(self.value(register) | other.value(register))
                     for register in set(self.registers) | set(other.registers)}
        pointers = {offset: widen(self.slot(offset) | other.slot(offset))
                    for offset in set(self.pointers) | set(other.pointers)}
        saved = {n: self.saved[n] | other.saved[n] for n in CALLEE_SAVED}
        return State(registers, pointers, saved)

    def key(self) -> tuple:
        return (tuple(sorted((r, tuple(sorted(v, key=repr))) for r, v in self.registers.items())),
                tuple(sorted((o, tuple(sorted(v, key=repr))) for o, v in self.pointers.items())),
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
            if base == "bl":
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
                # Alignment padding after the last return is never executed; anything else is unclassifiable.
                if block.instructions and not all(instruction.base == "nop" for instruction in block.instructions):
                    self.errors.append(f"unreachable instructions cannot be classified: '{block.instructions[0].text}'")
                continue
            if self.transfer(index, states[index].copy()) is not None and block.falls and index + 1 >= len(self.blocks):
                self.errors.append("control falls off the end of the function")

    def transfer(self, index: int, state: State) -> State | None:
        for instruction in self.blocks[index].instructions:
            if not self.step(instruction, state):
                return None
        return state

    def step(self, instruction: Instruction, state: State) -> bool:
        """Apply one instruction; False when control leaves the function or the path cannot be classified."""
        if not instruction.admitted():
            self.fail(instruction, f"instruction form outside the admitted grammar: {instruction.form or instruction.base}")
            return False
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
        if kind in ("jump", "branch"):
            return self.reads(instruction, state, instruction.operands[:-1])
        if instruction.memory is not None:
            return self.access(instruction, state)
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
        """Loads/stores: FP/SIMD stack traffic, ABI saves/restores, pointer slots, with exact footprints."""
        name = instruction.base
        load, store = name in LOADS, name in STORES
        if not (load or store):
            self.fail(instruction, "unclassifiable memory instruction")
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
        if post and (writeback or offset_text or len(post) != 1 or immediate(post[0]) is None):
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
        data = instruction.operands[:instruction.memory]
        if store and not self.reads(instruction, state, data):
            return False
        items, position = [], address
        for operand in data:
            width = footprint(instruction, operand)
            registers = fp_registers(operand) if not (gp(operand) or GP32.match(operand)) else None
            if width is None or (registers is None and not (gp(operand) or GP32.match(operand))):
                self.fail(instruction, f"memory footprint of {operand} is not exactly known")
                return False
            items.append(("fp" if registers is not None else ("x" if gp(operand) else "w"),
                          operand, registers, position, width))
            position = None if position is None else position + width
        if stack and not self.slots(instruction, state, load, store, address, items):
            return False
        if store and not stack and any(provenance(state.value(gp(operand))) != "N"
                                       for category, operand, *_ in items if category == "x"):
            self.fail(instruction, "a stack address is stored where it cannot be tracked")
            return False
        spilled = False
        eligible = name in ABI_TRANSFERS and all(ABI_REGISTER.match(operand) for _, operand, *_ in items)
        for category, operand, registers, start, width in items:
            if category != "fp":
                if load:
                    exact = stack and start is not None and category == "x" and name in WHOLE_X
                    state.assign(gp(operand) or gp_view(operand), state.slot(start) if exact else NOT_STACK)
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
                if eligible and start is not None:
                    start += register_width
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
        """Pointer slots and live ABI save slots: any access that cannot be proven disjoint or exact fails."""
        live = [(n, slot) for n in CALLEE_SAVED for status, slot in state.saved[n]
                if status in ("S", "D") and slot is not None]
        if address is None:
            if state.pointers or (store and live):
                self.fail(instruction, "stack access at an unknown offset while pointer or save slots are live")
                return False
            if store and any(provenance(state.value(gp(operand))) != "N"
                             for category, operand, *_ in items if category == "x"):
                self.fail(instruction, "a stack address is stored where it cannot be tracked")
                return False
            return True
        for category, operand, _, start, width in items:
            end = start + width
            for slot in list(state.pointers):
                if not (slot < end and start < slot + 8):
                    continue
                exact = start == slot and width == 8 and category == "x" and instruction.base in WHOLE_X
                if load and not exact:
                    self.fail(instruction, "partial or reinterpreting reload of a stored stack address")
                    return False
                if store and not exact and not (start <= slot and slot + 8 <= end):
                    self.fail(instruction, "partial overwrite of a stored stack address")
                    return False
                if store and not exact:
                    state.store_slot(slot, NOT_STACK)
            if store:
                for number, (slot, slot_width) in live:
                    if start < slot + slot_width and slot < end:
                        self.fail(instruction, f"overwrites the live save slot of d{number}")
                        return False
                if category == "x":
                    values = state.value(gp(operand))
                    if provenance(values) == "A":
                        self.fail(instruction, "stores a stack address on only some paths")
                        return False
                    state.store_slot(start, values)
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
        if stack_inputs and any(operand_kind(operand) == "reloc" for operand in inputs):
            self.fail(instruction, "relocated arithmetic on a stack address")
            return False
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


def proven_noreturn(symbol: str, bodies: dict[str, list[str]], dialect: Dialect, imports: set[str],
                    prefix: str) -> bool:
    """The fatal diagnostic is exempt only if every reachable instruction is an admitted form, its only calls are
    the trusted diagnostic calls, and its body is acyclic with every path ending in a call to abort()."""
    body = bodies.get(symbol)
    if body is None:
        return False
    blocks, errors = build_cfg(body, dialect)
    if errors:
        return False
    labels = {label: index for index, block in enumerate(blocks) for label in block.labels}
    successors: dict[int, list[int] | None] = {}
    for index, block in enumerate(blocks):
        edges, ends, valid = [], False, True
        for instruction in block.instructions:
            target = instruction.operands[-1] if instruction.operands else ""
            if not instruction.admitted():
                valid = False
                break
            if instruction.base == "bl":
                if target in labels or dialect.local(target):
                    valid = False
                    break
                if target not in imports:
                    valid = False  # Trusted only as an undefined, strongly bound import; never a local body.
                    break
                if target in {prefix + leaf for leaf in NORETURN_LEAVES}:
                    ends = True
                    break
                if target not in {prefix + call for call in DIAGNOSTIC_CALLS}:
                    valid = False
                    break
                continue
            if instruction.base in ("b", "b.cond", "cbz", "cbnz", "tbz", "tbnz"):
                if target not in labels:
                    valid = False
                    break
                edges.append(labels[target])
                if instruction.base == "b":
                    ends = True
                    break
            elif instruction.base == "ret":
                valid = False
                break
        if not valid:
            successors[index] = None
            continue
        if not ends:
            if index + 1 >= len(blocks):
                successors[index] = None
                continue
            edges.append(index + 1)
        successors[index] = edges
    colors: dict[int, str] = {}

    def visit(index: int) -> bool:
        if successors.get(index) is None or colors.get(index) == "active":
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


def noreturn_symbols(bodies: dict[str, list[str]], dialect: Dialect, imports: set[str], prefix: str) -> set[str]:
    """The only symbol a tile program may call: the exact fatal diagnostic (raw linker name of the object
    format), when proven never to return."""
    return {name for name in bodies if name == prefix + FATAL_DIAGNOSTIC and
            proven_noreturn(name, bodies, dialect, imports, prefix)}


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
TILE_PROGRAM = re.compile(r"_ZNK9llaminar23cpu11native_quantised" + PROGRAM.pattern + r"EEvv")


def audit(data: bytes, disassembly: str, mode: str) -> dict[str, int]:
    """Per-program FP/SIMD spill counts of the tile programs actually assembled into the witness object.

    The expected programs derive from the admitted envelope; the object's embedded inventory, its function
    symbols and the complete disassembly of its executable sections must all agree. Structural failures
    (missing/duplicate/unexpected/empty programs, incomplete coverage, unsupported calls, indirect branches,
    unclassifiable forms, ambiguous or unclassifiable stack addressing, unproven ABI saves) are fatal and
    never paid from a spill budget.
    """
    if mode not in ENVELOPE_PROGRAMS:
        raise ValueError(f"no tile program inventory for ISA envelope {mode}")
    image = read_object(data)
    inventory = object_inventory(data)
    if inventory != [",".join(ENVELOPE_PROGRAMS[mode])]:
        raise ValueError(f"probe instantiation inventory {inventory} does not match the {mode} envelope")
    expected = set()
    for entry in ENVELOPE_PROGRAMS[mode]:
        rows, dot = entry.split(":")
        leaf = 1 if dot == "I8MM" and rows == "1" else DOTS[dot]
        for policy in (0, 1):
            for encoding, nibble_map, minimum in DISPATCH:
                expected.add((int(rows), leaf, policy, encoding, nibble_map, minimum))
    bodies = normalize_disassembly(disassembly, image)
    tile_symbols = sorted({symbol.name for symbol in image.symbols
                           if symbol.kind != "section" and TILE_RUN.search(symbol.name)})
    for symbol in tile_symbols:
        if symbol not in bodies:
            raise ValueError(f"tile program symbol is not a function in an executable section: {symbol}")
    noreturn = noreturn_symbols(bodies, GENERIC, trusted_imports(image), image.prefix)
    found: dict[tuple, str] = {}
    for name in tile_symbols:
        match = TILE_PROGRAM.fullmatch(name[len(image.prefix):]) if name.startswith(image.prefix) else None
        if not match:
            raise ValueError(f"unrecognized tile program symbol: {name}")
        key = tuple(int(group) for group in match.groups())
        if key in found:
            raise ValueError(f"tile program appears twice: {name}")
        if key not in expected:
            raise ValueError(f"tile program outside the {mode} inventory: {name}")
        found[key] = name
    missing = expected - found.keys()
    if missing:
        raise ValueError(f"register witness is missing {len(missing)} expected tile programs, e.g. {sorted(missing)[:3]}")
    result: dict[str, int] = {}
    structural: list[str] = []
    for key in sorted(expected):
        program = found[key]
        result[program], errors = analyse_function(program, bodies, GENERIC, noreturn)
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


def executed_command(command: list[str], source: str, obj: Path, dependencies: Path) -> list[str]:
    """The witness compile; -H reports every header and precompiled header the compiler actually read."""
    return command + WITNESS_FLAGS + ["-c", source, "-o", str(obj), "-MD", "-MF", str(dependencies), "-MT", "probe",
                                      "-H"]


def require_absolute_source(source: str) -> None:
    """The probe path is passed to the compiler as given; relative spellings depend on the cwd and are rejected."""
    if not os.path.isabs(source):
        raise ValueError(f"the witness probe path must be absolute: {source}")


def disassembler_command(tool: str, obj: Path) -> tuple[list[str], str]:
    """Disassembly command for the configured objdump, and its version text."""
    version = subprocess.check_output([tool, "--version"], text=True)
    if "GNU objdump" in version:
        options = ["-d", "-r", "-z", "--no-show-raw-insn"]
    elif "LLVM" in version:
        options = ["-d", "-r", "-z", "--no-show-raw-insn", "--triple=aarch64"]  # Generic, not Apple, printer.
    else:
        raise ValueError(f"unsupported disassembler: {version.splitlines()[0] if version else tool}")
    return [tool] + options + [str(obj)], version


def disassemble(tool: str, obj: Path) -> tuple[str, list[str], str]:
    command, version = disassembler_command(tool, obj)
    return subprocess.check_output(command, text=True), command, version


ASM_KEYWORDS = frozenset({"asm", "__asm", "__asm__"})
ASM_QUALIFIERS = frozenset({"volatile", "__volatile__", "__volatile", "inline", "__inline__", "__inline", "goto"})
# One C++ token of a preprocessed translation unit (comments are already removed by -E). Linemarkers and
# pragmas are skipped; pp-numbers include digit separators; string and character literals carry their
# encoding prefix; raw strings use their delimiter.
CXX_TOKEN = re.compile(r"""
    (?P<directive>^[ \t]*\#[^\n]*)
  | (?P<space>[ \t\r\f\v]*\n|[ \t\r\f\v]+)
  | (?P<raw>(?:u8|u|U|L)?R"(?P<delimiter>[^()\\\s"]{0,16})\((?P<body>.*?)\)(?P=delimiter)")
  | (?P<string>(?:u8|u|U|L)?"(?:[^"\\\n]|\\.)*")
  | (?P<character>(?:u8|u|U|L)?'(?:[^'\\\n]|\\.)+')
  | (?P<number>\.?[0-9](?:[0-9A-Za-z_.]|'[0-9A-Za-z_]|[eEpP][+-])*)
  | (?P<identifier>[A-Za-z_$][A-Za-z0-9_$]*)
  | (?P<punctuator>.)
""", re.MULTILINE | re.DOTALL | re.VERBOSE)
ASM_LABEL = re.compile(r"[A-Za-z_$][\w$]*")
ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "a": "\a", "b": "\b", "f": "\f", "v": "\v"}


def decode_literal(token: re.Match) -> str:
    """Contents of a string literal token, with escapes decoded."""
    if token.lastgroup == "raw":
        return token.group("body")
    text = token.group()
    body, decoded, position = text[text.index('"') + 1:-1], [], 0
    while position < len(body):
        character = body[position]
        if character != "\\":
            decoded.append(character)
            position += 1
            continue
        escape = body[position + 1]
        if escape == "x":
            digits = re.match(r"[0-9a-fA-F]+", body[position + 2:]).group()
            decoded.append(chr(int(digits, 16) % 0x110000))
            position += 2 + len(digits)
        elif escape in "01234567":
            digits = re.match(r"[0-7]{1,3}", body[position + 1:]).group()
            decoded.append(chr(int(digits, 8)))
            position += 1 + len(digits)
        elif escape in "uU":
            width = 4 if escape == "u" else 8
            decoded.append(chr(int(body[position + 2:position + 2 + width], 16) % 0x110000))
            position += 2 + width
        else:
            decoded.append(ESCAPES.get(escape, escape))
            position += 2
    return "".join(decoded)


OPAQUE_DIRECTIVES = re.compile(r"^[ \t]*#[ \t]*(?:pragma[ \t]+(?:GCC[ \t]+pch_preprocess|clang[ \t]+module)\b|import\b"
                               r"|include\b|include_next\b|embed\b)")
NAMESPACE_INTRODUCERS = frozenset({"namespace"})
CLASS_KEYS = frozenset({"class", "struct", "union", "enum"})
LABEL_PRECEDERS = frozenset({")", "]"})


def cxx_tokens(preprocessed: str) -> list[re.Match]:
    """Significant C++ tokens of a preprocessed translation unit.

    Linemarkers and ordinary pragmas are skipped. Directives that hand the compiler content -E does not show
    (a PCH marker, a clang module pragma, a remaining include/import/embed) fail; so does an unterminated
    literal.
    """
    tokens = []
    for token in CXX_TOKEN.finditer(preprocessed):
        kind = token.lastgroup
        if kind == "directive":
            if OPAQUE_DIRECTIVES.match(token.group()):
                raise ValueError(f"translation unit uses opaque compiler input: {token.group().strip()[:80]}")
            continue
        if kind == "space":
            continue
        if kind == "punctuator" and token.group() in "\"'":
            raise ValueError(f"unterminated literal at offset {token.start()}")
        tokens.append(token)
    return tokens


def scope_kinds(tokens: list[re.Match]) -> list[bool]:
    """For every token, whether it lies in executable code (a function body, lambda or initializer braces).

    A brace opens a declaration scope only when the declaration preceding it (since the last ';', '{' or '}')
    starts with 'namespace' / 'inline namespace' / 'extern "..."', or names a class key with no parenthesis;
    any other brace is treated as code, which only makes the label rule stricter.
    """
    stack, kinds, start = [], [], 0
    for index, token in enumerate(tokens):
        text = token.group()
        kinds.append(bool(stack) and stack[-1])
        if text == "{":
            head = [other.group() for other in tokens[start:index]]
            words = [word for word in head if word not in ("inline", "export")]
            declaration = bool(words) and (words[0] in NAMESPACE_INTRODUCERS or
                                           (words[0] == "extern" and len(words) == 2 and words[1].startswith('"')) or
                                           (any(word in CLASS_KEYS for word in words) and "(" not in words))
            stack.append(not declaration or bool(stack and stack[-1]))
            start = index + 1
        elif text == "}":
            if not stack:
                raise ValueError(f"unbalanced braces at offset {token.start()}")
            stack.pop()
            start = index + 1
        elif text == ";":
            start = index + 1
    return kinds


def inline_assembly_violations(preprocessed: str) -> list[str]:
    """Inline-assembly constructs the witness cannot prove free of assembler-side inputs.

    The compiler's dependency lists name preprocessing inputs only; the assembler can read further files and
    construct directives through macros or substitution. Rather than model the assembler, every asm keyword of
    the tokenized translation unit (including one nested in another asm's operands or in a lambda) must be
    either an empty-template statement (a compiler barrier) or a declaration asm label: a single plain symbol,
    without qualifiers or operands, directly after a declarator, outside executable code. Anything else fails.
    """
    tokens = cxx_tokens(preprocessed)
    code = scope_kinds(tokens)
    findings = []
    for index, token in enumerate(tokens):
        if token.lastgroup != "identifier" or token.group() not in ASM_KEYWORDS:
            continue
        position, qualified = index + 1, False
        while position < len(tokens) and tokens[position].group() in ASM_QUALIFIERS:
            position, qualified = position + 1, True
        if position >= len(tokens) or tokens[position].group() != "(":
            raise ValueError(f"unclassifiable inline assembly at offset {token.start()}")
        position += 1
        template = []
        while position < len(tokens) and tokens[position].lastgroup in ("string", "raw"):
            template.append(decode_literal(tokens[position]))
            position += 1
        if not template or position >= len(tokens) or tokens[position].group() not in (")", ":"):
            raise ValueError(f"inline assembly without a literal template at offset {token.start()}")
        text = "".join(template)
        if not text.strip():
            continue  # Empty template: emits nothing; operands are visited as ordinary tokens.
        previous = tokens[index - 1] if index else None
        label = (not qualified and tokens[position].group() == ")" and not code[index] and previous is not None
                 and (previous.group() in LABEL_PRECEDERS or
                      (previous.lastgroup == "identifier" and previous.group() not in ASM_KEYWORDS))
                 and ASM_LABEL.fullmatch(text))
        if not label:
            findings.append(f"offset {token.start()}: {text[:80]!r}")
    return findings


# Driver routing. The expanded jobs of the exact compile (-###) are the authority for what the compiler and
# assembler read: one compiler job and, for GNU, one assembler job whose sole input is the compiler's output.
REJECTED_JOB_ARGUMENTS = re.compile(r"^(?:-Wa,.*|-Xassembler|-save-temps.*|--param=as.*|-include-pch|-emit-pch|-fpch-preprocess|-fmodules|-fcxx-modules|"
                                    r"-fmodules-ts|-fmodule-file.*|-fmodule-map-file=.*|-fprebuilt-module-path.*|"
                                    r"-fimplicit-modules|-fmodule-mapper.*|-fplugin.*|-load|-plugin|-add-plugin|"
                                    r"-fpass-plugin.*|-specs.*|--specs.*|-B.*|-wrapper|@.*)$")
GNU_ASSEMBLER_OPTIONS = re.compile(r"^(?:-EL|--gdwarf-\d|-march=[\w+.-]+|-mcpu=[\w+.-]+|-mabi=lp64)$")
CONFIGURATION_LINES = re.compile(r"^(?:Reading specs from|Configuration file:|Using specs)", re.MULTILINE)


def driver_jobs(command: list[str], directory: str, identity: str, source: str, obj: Path) -> list[list[str]]:
    """Validate the expanded jobs (-###) of the exact witness compile and return them."""
    for argument in command:
        if REJECTED_JOB_ARGUMENTS.match(argument) or argument in ("-no-integrated-as", "-fno-integrated-as",
                                                                   "--config") or argument.startswith("--config="):
            raise ValueError(f"the witness compile command routes inputs outside the dependency closure: {argument}")
    expansion = subprocess.run(command + ["-###"], cwd=directory, capture_output=True, text=True)
    if expansion.returncode:
        raise ValueError(f"the compiler driver rejected the witness command: {expansion.stderr[-300:]}")
    if CONFIGURATION_LINES.search(expansion.stderr):
        raise ValueError("the compiler driver reads a specs or configuration file")
    jobs = [shlex.split(line) for line in expansion.stderr.splitlines() if line.startswith(" ")]
    for job in jobs:
        for argument in job:
            if REJECTED_JOB_ARGUMENTS.match(argument):
                raise ValueError(f"a compiler job reads opaque input: {argument}")
    if identity.startswith(("AppleClang-", "Clang-")):
        if len(jobs) != 1 or len(jobs[0]) < 2 or jobs[0][1] != "-cc1" or "-emit-obj" not in jobs[0]:
            raise ValueError(f"the clang driver must run exactly one integrated -cc1 -emit-obj job: {jobs}")
        if jobs[0][jobs[0].index("-o") + 1] != str(obj):
            raise ValueError("the clang job does not write the witness object")
        return jobs
    if len(jobs) != 2 or Path(jobs[0][0]).name not in ("cc1plus", "cc1") or Path(jobs[1][0]).name != "as":
        raise ValueError(f"the GNU driver must run exactly one cc1plus job and one as job: {[job[:1] for job in jobs]}")
    if source not in jobs[0]:
        raise ValueError("the GNU compiler job does not compile the probe")
    assembly = jobs[0][jobs[0].index("-o") + 1] if "-o" in jobs[0] else None
    assembler, position, inputs = jobs[1], 1, []
    while position < len(assembler):
        argument = assembler[position]
        if argument in ("-I", "-o"):
            if argument == "-o" and assembler[position + 1] != str(obj):
                raise ValueError("the GNU assembler job does not write the witness object")
            position += 2
            continue
        if argument.startswith("-"):
            if not GNU_ASSEMBLER_OPTIONS.match(argument):
                raise ValueError(f"unadmitted GNU assembler option: {argument}")
        else:
            inputs.append(argument)
        position += 1
    if assembly is None or inputs != [assembly]:
        raise ValueError(f"the GNU assembler job reads inputs other than the compiler output: {inputs}")
    return jobs


def resolve_program(program: str, directory: str, search: str) -> str:
    """Absolute path of a job's program (GNU prints the assembler bare; it is found on COMPILER_PATH/PATH)."""
    if os.path.isabs(program):
        return program
    for entry in [part for part in search.split(os.pathsep) if part]:
        candidate = os.path.join(entry if os.path.isabs(entry) else os.path.join(directory, entry), program)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.realpath(candidate)
    raise ValueError(f"cannot resolve job program {program}")


def precompiled_headers(header_trace: str) -> list[str]:
    """PCH files the GNU compiler used or tried: -H prints them after the include depth dots (possibly none)
    marked '!' (valid) or 'x' (invalid)."""
    return [line for line in header_trace.splitlines() if re.match(r"^\.*[!x] \S", line)]


def build(args: argparse.Namespace) -> None:
    """Compile the probe object with the target command, disassemble it, audit it, publish the proof last."""
    require_absolute_source(args.source)
    output = Path(args.proof)
    output.parent.mkdir(parents=True, exist_ok=True)
    obj, dependencies = output.with_suffix(".o"), output.with_suffix(".deps")
    disassembly_path, translation_unit = output.with_suffix(".dis"), output.with_suffix(".i")
    preprocessed, depfile, temporary = output.with_suffix(".pre.deps"), output.with_suffix(".d"), output.with_suffix(".tmp")
    # proof.s / proof.sym: retired artifacts of earlier schemas.
    for stale in (output, temporary, obj, dependencies, preprocessed, depfile, disassembly_path, translation_unit,
                  output.with_suffix(".s"), output.with_suffix(".sym")):
        stale.unlink(missing_ok=True)
    command, directory = envelope.target_command(args)
    version = subprocess.check_output([command[0], "--version"], cwd=directory, text=True)
    identity = compiler_identity(version)
    if identity != args.identity:
        raise ValueError(f"compiler reports identity {identity}, configured budget identity is {args.identity}")
    subprocess.run(command + WITNESS_FLAGS + ["-M", "-MF", str(preprocessed), "-MT", "probe", args.source],
                   cwd=directory, check=True, stdout=subprocess.DEVNULL)
    before = bind_inputs(read_depfile(preprocessed, directory))
    subprocess.run(command + WITNESS_FLAGS + ["-E", args.source, "-o", str(translation_unit)], cwd=directory, check=True)
    unproved = inline_assembly_violations(translation_unit.read_text(errors="replace"))
    if unproved:
        raise ValueError(f"inline assembly the dependency closure cannot bind (non-empty template): {unproved[:2]}")
    executed = executed_command(command, args.source, obj, dependencies)
    jobs = driver_jobs(executed, directory, identity, args.source, obj)
    search = os.pathsep.join(re.findall(r"^COMPILER_PATH=(.*)$", subprocess.run(
        executed + ["-###"], cwd=directory, capture_output=True, text=True).stderr, re.MULTILINE) +
        [os.environ.get("PATH", "")])
    programs = {program: envelope.digest(Path(program))
                for program in (resolve_program(job[0], directory, search) for job in jobs)}
    compiled = subprocess.run(executed, cwd=directory, capture_output=True, text=True)
    if compiled.returncode:
        raise ValueError(f"the witness compile failed: {compiled.stderr[-2000:]}")
    used = precompiled_headers(compiled.stderr)
    if used:
        raise ValueError(f"the witness compile read precompiled headers: {used[:2]}")
    closure = read_depfile(dependencies, directory)
    if closure != sorted(before):
        raise ValueError("the compilation's dependency closure differs from its preprocessing closure")
    after = bind_inputs(closure)
    if after != before:
        changed = sorted(path for path in closure if before[path] != after[path])
        raise ValueError(f"inputs changed while the witness compiled: {changed[:3]}")
    disassembly, disassembly_command, tool_version = disassemble(args.disassembler, obj)
    disassembly_path.write_text(disassembly)
    counts = audit(obj.read_bytes(), disassembly, args.mode)
    require_registers(counts, args.identity, args.mode)
    result = {
        "schema": 7, "signature": args.signature, "identity": identity, "mode": args.mode,
        "checker_sha256": envelope.digest(Path(__file__)),
        "envelope_sha256": envelope.digest(_ENVELOPE),
        "budget_sha256": envelope.digest(BUDGET_PATH),
        "compiler_version": version,
        "executed_command": executed, "executed_directory": directory,
        "target_command_sha256": envelope.command_digest(command, directory),
        "driver_jobs": jobs, "job_programs": programs,
        "inputs": after,
        "object_file": obj.name, "object_sha256": envelope.digest(obj),
        "disassembler": args.disassembler, "disassembler_version": tool_version,
        "disassembler_sha256": envelope.digest(Path(os.path.realpath(args.disassembler))),
        "disassembly_command": disassembly_command, "disassembly_sha256": envelope.digest(disassembly_path),
        "translation_unit_sha256": envelope.digest(translation_unit),
        "program_violations": counts,
    }
    # Ninja depfile: resolved paths (Ninja canonicalises ".." lexically, which would name another file).
    target = args.depfile_target or str(output)
    resolved = sorted({identity["resolved"] for identity in after.values()})
    depfile.write_text(f"{target}: " + " ".join(path.replace(" ", "\\ ") for path in resolved) + "\n")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    temporary.replace(output)


def verify(args: argparse.Namespace) -> None:
    """Validate the retained proof: inputs, command, the bound object, and its re-disassembly by the bound tool."""
    require_absolute_source(args.source)
    proof_path = Path(args.proof)
    proof = json.loads(proof_path.read_text())
    if (proof.get("schema") != 7 or proof.get("signature") != args.signature or
            proof.get("identity") != args.identity or proof.get("mode") != args.mode):
        raise ValueError("tile register proof has a foreign schema/configuration")
    if compiler_identity(str(proof.get("compiler_version", ""))) != args.identity:
        raise ValueError("recorded compiler identity does not match the configured budget identity")
    for key, path in (("checker_sha256", Path(__file__)), ("envelope_sha256", _ENVELOPE),
                      ("budget_sha256", BUDGET_PATH)):
        if proof.get(key) != envelope.digest(path):
            raise ValueError(f"stale tile register tool: {path}")
    inputs = proof.get("inputs")
    source = resolve(args.source)[0]
    if not isinstance(inputs, dict) or not any(isinstance(identity, dict) and identity.get("resolved") == source
                                               for identity in inputs.values()):
        raise ValueError("tile register proof does not bind the probe's dependency closure")
    for path, expected in inputs.items():
        if not os.path.isfile(path) or input_identity(path) != expected:
            raise ValueError(f"stale tile register input (source, header or symlink changed): {path}")
    command, directory = envelope.target_command(args)
    if proof.get("target_command_sha256") != envelope.command_digest(command, directory):
        raise ValueError("actual target compile command changed after the register witness")
    obj = proof_path.with_suffix(".o")
    if proof.get("object_file") != obj.name or proof.get("executed_directory") != directory or \
            proof.get("executed_command") != executed_command(command, args.source, obj, proof_path.with_suffix(".deps")):
        raise ValueError("recorded witness invocation is not the target command over the probe")
    if not obj.is_file() or proof.get("object_sha256") != envelope.digest(obj):
        raise ValueError("tile register object is missing or changed")
    programs = proof.get("job_programs")
    if not isinstance(programs, dict) or not programs or \
            any(not os.path.isfile(path) or envelope.digest(Path(path)) != digest for path, digest in programs.items()):
        raise ValueError("a compiler or assembler program of the witness compile changed")
    if proof.get("disassembler") != args.disassembler or \
            proof.get("disassembler_sha256") != envelope.digest(Path(os.path.realpath(args.disassembler))):
        raise ValueError("configured disassembler changed after the register witness")
    disassembly, disassembly_command, tool_version = disassemble(args.disassembler, obj)
    if disassembly_command != proof.get("disassembly_command") or tool_version != proof.get("disassembler_version"):
        raise ValueError("disassembler invocation changed after the register witness")
    retained = proof_path.with_suffix(".dis")
    if not retained.is_file() or retained.read_text() != disassembly or \
            proof.get("disassembly_sha256") != envelope.digest(retained):
        raise ValueError(f"retained disassembly does not match the bound object: {retained.name}")
    unit = proof_path.with_suffix(".i")
    if not unit.is_file() or proof.get("translation_unit_sha256") != envelope.digest(unit) or \
            inline_assembly_violations(unit.read_text(errors="replace")):
        raise ValueError("retained translation unit is missing, changed or has unproved inline assembly")
    counts = audit(obj.read_bytes(), disassembly, args.mode)
    if proof.get("program_violations") != counts:
        raise ValueError("tile register evidence changed")
    require_registers(counts, args.identity, args.mode)


def census(args: argparse.Namespace) -> None:
    """Maintainer tool: print the per-program spill census for qualifying a budget (no budget applied)."""
    require_absolute_source(args.source)
    command, directory = envelope.target_command(args)
    version = subprocess.check_output([command[0], "--version"], cwd=directory, text=True)
    obj = Path(args.proof).with_suffix(".census.o")
    obj.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command + WITNESS_FLAGS + ["-c", args.source, "-o", str(obj)], cwd=directory, check=True)
    disassembly, *_ = disassemble(args.disassembler, obj)
    counts = audit(obj.read_bytes(), disassembly, args.mode)
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
    parser.add_argument("--disassembler", required=True, help="objdump (GNU binutils or LLVM) for the target")
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
