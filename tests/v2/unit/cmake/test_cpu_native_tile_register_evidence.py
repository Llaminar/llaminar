#!/usr/bin/env python3
"""Compiler-free adversarial tests of the native NEON tile register-allocation witness.

Fixtures are synthetic ELF64 AArch64 (ET_REL) and Mach-O arm64 (MH_OBJECT)
objects - sections, symbol tables with raw linker names and relocation tables,
with zero-filled code bytes - paired with an llvm-objdump-style disassembly
rendered from the same layout; they are not native compiler/disassembler pairs.
Native object re-audits are separate owner evidence. Analysis tests run the
production front end over the pairs. Covered: the control-flow/provenance
analysis, the ABI save proof, admitted forms, call-boundary rejection, the
never-returning diagnostic and its trusted imports (binding and exact ABI
spelling), complete and aligned coverage, object-bound relocations and their
native field admission, object types, the embedded inventory, the
envelope-derived program inventory, the single-row/ratchet policy, compiler
identity, the tokenized inline-assembly guard (nested asm, label context,
opaque directives), and proof binding through scripted stand-in tools
(dependency closure, symlinks, inputs changing during the build, executed
command, driver jobs and their programs, precompiled headers, object,
disassembler, budget, translation unit). Expected program keys are written out
here, never derived from checker tables.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import textwrap
import unittest

CHECKER_PATH = Path(__file__).resolve().parents[2] / "cmake" / "CPUNativeTileRegisterEvidence.py"
SPEC = importlib.util.spec_from_file_location("cpu_native_tile_register_evidence", CHECKER_PATH)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)

# Tile::dispatch() programs per policy: (encoding, nibble map, with-minimum), written out independently.
DISPATCHED = [(1, 0, 1), (1, 0, 0), (5, 0, 0), (2, 0, 0), (4, 2, 0), (0, 0, 0), (0, 2, 0), (0, 1, 1)]
NEON_KEYS = sorted((rows, 0, policy, encoding, nibble_map, minimum)
                   for rows in (1, 2, 3, 4) for policy in (0, 1) for encoding, nibble_map, minimum in DISPATCHED)
NEON_INVENTORY = "1:ASIMD,2:ASIMD,3:ASIMD,4:ASIMD"
FATAL = "_ZN9llaminar23cpu16native_quantised28CPUNativeQuantisedWeightView32rejectOutsideAdmittedPublicationEv"
CLEAN = "\tstp\tx29, x30, [sp, #-16]!\n\tmov\tx29, sp\n\tsdot\tv0.4s, v1.16b, v2.16b\n\tldp\tx29, x30, [sp], #16\n\tret"
FATAL_BODY = "\tstp\tx29, x30, [sp, #-16]!\n\tmov\tx29, sp\n\tbl\tfwrite\n\tbl\tabort\n"
BRANCH = re.compile(r"^(?:b|bl|cbz|cbnz|tbz|tbnz|b\.\w+)$")


def name(rows: int, dot: int, policy: int, encoding: int, nibble_map: int, minimum: int) -> str:
    """Mangled tile run<> symbol for one program key."""
    return (f"_ZNK9llaminar23cpu16native_quantised27CPUNativeNEONProjectionTileILm{rows}EE3runINS1_11multi_scale17"
            f"NEONMultiScaleOpsILNS5_11NEONByteDotE{dot}EEELNS_28CPUProjectionNumericalPolicyE{policy}ELNS1_26"
            f"CPUNativeQuantisedEncodingE{encoding}ELNS3_9NibbleMapE{nibble_map}ELb{minimum}EEEvv")


ELF_TYPES = {name: number for number, name in CHECKER.ELF_RELOCATIONS.items()}
MACHO_TYPES = {name: number for number, name in CHECKER.MACHO_RELOCATIONS.items()}
RELOCATION_NOTE = re.compile(r"\s+!(\w+):(\S+?)(?:([+-])0x([0-9a-f]+))?$")


def branch_relocation(mnemonic: str, macho: bool) -> str:
    if macho:
        return "ARM64_RELOC_BRANCH26"
    return {"bl": "R_AARCH64_CALL26", "b": "R_AARCH64_JUMP26", "tbz": "R_AARCH64_TSTBR14",
            "tbnz": "R_AARCH64_TSTBR14"}.get(mnemonic, "R_AARCH64_CONDBR19")


class Layout:
    """Addresses, sections, symbols and relocations of rendered functions, shared by object and disassembly.

    Function names are raw linker names. An instruction line may end in '!TYPE:target[+0xN]' to carry that
    relocation; a branch to a non-local symbol carries the format's branch relocation automatically.
    """

    def __init__(self, functions: list[tuple[str, str]], macho: bool) -> None:
        self.macho, self.entries, address = macho, [], 0
        defined = {symbol for symbol, _ in functions}
        for symbol, body in functions:
            if not macho:
                address = 0
            start, labels, code, relocations = address, {}, [], {}
            for line in textwrap.dedent(body).splitlines():
                text = line.strip()
                if not text:
                    continue
                if re.match(r"^\.L\w+:$", text):
                    labels[text[:-1]] = address
                    continue
                note = RELOCATION_NOTE.search(text)
                if note:
                    text = text[:note.start()].strip()
                    addend = int(note.group(4), 16) * (-1 if note.group(3) == "-" else 1) if note.group(4) else 0
                    relocations[address] = (note.group(1), note.group(2), addend)
                else:
                    mnemonic, *rest = text.split(None, 1)
                    target = CHECKER.split_operands(rest[0])[-1] if rest else ""
                    if BRANCH.match(mnemonic) and target and not target.startswith(".L"):
                        relocations[address] = (branch_relocation(mnemonic, macho), target, 0)
                code.append((address, text))
                address += 4
            section = "__TEXT,__text" if macho else f".text.{symbol}"
            self.entries.append({"name": symbol, "section": section, "start": start, "end": address,
                                 "labels": labels, "code": code, "relocations": relocations})
        self.imports = sorted({target for entry in self.entries for _, target, _ in entry["relocations"].values()
                               if target not in defined and not target.startswith(".")})


def disassembly(functions: list[tuple[str, str]], macho: bool = False, path: str = "probe.o") -> str:
    """Render functions as llvm-objdump -d -r output (raw names, printed relocation records)."""
    layout = Layout(functions, macho)
    out = [f"{path}:\tfile format {'mach-o arm64' if macho else 'elf64-littleaarch64'}", ""]
    if macho:
        out.append("Disassembly of section __TEXT,__text:")
    for entry in layout.entries:
        symbol, start, labels = entry["name"], entry["start"], entry["labels"]
        if not macho:
            out += ["", f"Disassembly of section {entry['section']}:"]
        out += ["", f"{start:016x} <{symbol}>:"]
        for at, text in entry["code"]:
            parts = text.split(None, 1)
            mnemonic, operands = parts[0], (CHECKER.split_operands(parts[1]) if len(parts) > 1 else [])
            if BRANCH.match(mnemonic) and operands:
                target = operands[-1]
                if target in labels:
                    operands[-1] = f"0x{labels[target]:x} <{symbol}+0x{labels[target] - start:x}>"
                elif target.startswith(".L"):
                    operands[-1] = f"0xfffff0 <{symbol}+0x{0xfffff0 - start:x}>"
                else:
                    operands[-1] = f"0x{at:x} <{symbol}+0x{at - start:x}>"
            out.append(f"{at:8x}: \t{mnemonic}\t{', '.join(operands)}" if operands else f"{at:8x}: \t{mnemonic}")
            if at in entry["relocations"]:
                kind, target, addend = entry["relocations"][at]
                out.append(f"\t\t{at:016x}:  {kind}\t{CHECKER.relocation_target(target, addend)}")
    return "\n".join(out) + "\n"


def witness_object(functions: list[tuple[str, str]] | None = None, inventory: str | None = NEON_INVENTORY,
                   macho: bool = False, symbols: list[tuple] | None = None, sizes: dict[str, int] | None = None,
                   extra: bytes = b"", bindings: dict[str, str] | None = None,
                   relocations: bool = True) -> bytes:
    """A real ELF64 AArch64 or Mach-O arm64 object consistent with Layout(functions).

    Defined functions, undefined imports and relocation tables come from the layout; symbols appends raw
    (name, kind, binding, owning function or None, value) records; sizes overrides ELF symbol sizes;
    relocations=False omits the relocation tables.
    """
    layout = Layout(functions or [], macho)
    marker = b"" if inventory is None else b"llaminar-native-neon-tile-probe:" + inventory.encode() + b"\0"
    records = [(entry["name"], "function", "global", entry["name"], entry["start"]) for entry in layout.entries]
    records += [(name, "undefined", (bindings or {}).get(name, "global"), None, 0) for name in layout.imports]
    records += symbols or []
    if not relocations:
        for entry in layout.entries:
            entry["relocations"] = {}
    if macho:
        return _macho(layout, records, marker + extra)
    return _elf(layout, records, marker + extra, sizes or {})


def _elf(layout: Layout, records: list[tuple], data: bytes, sizes: dict[str, int]) -> bytes:
    relocated = [entry for entry in layout.entries if entry["relocations"]]
    names = [""] + [entry["section"] for entry in layout.entries] + [".rodata"] + \
        [".rela" + entry["section"] for entry in relocated] + [".symtab", ".strtab", ".shstrtab"]
    section_index = {entry["name"]: number + 1 for number, entry in enumerate(layout.entries)}
    section_index[".rodata"] = len(layout.entries) + 1
    symtab_index = len(names) - 3
    if any(target == ".rodata" for entry in layout.entries for _, target, _ in entry["relocations"].values()):
        records = [("", "section", "local", ".rodata", 0)] + records
    shstr = b"\0" + b"".join(name.encode() + b"\0" for name in names[1:])
    strtab, symtab, symbol_index = bytearray(b"\0"), bytearray(24), {}
    extents = {entry["name"]: entry["end"] - entry["start"] for entry in layout.entries}
    for number, (name, kind, binding, owner, value) in enumerate(records, start=1):
        offset = len(strtab)
        strtab += name.encode() + b"\0"
        symbol_index.setdefault(name or owner, number)
        info = ({"local": 0, "global": 1, "weak": 2}[binding] << 4) | {
            "function": 2, "notype": 0, "object": 1, "undefined": 0, "absolute": 0, "section": 3}[kind]
        section = section_index[owner] if owner else (0xFFF1 if kind == "absolute" else 0)
        size = sizes.get(name, extents.get(name, 0)) if kind == "function" else 0
        symtab += offset.to_bytes(4, "little") + bytes([info, 0]) + section.to_bytes(2, "little") + \
            value.to_bytes(8, "little") + size.to_bytes(8, "little")
    tables = []
    for entry in relocated:
        table = bytearray()
        for address, (kind, target, addend) in sorted(entry["relocations"].items()):
            table += address.to_bytes(8, "little") + \
                ((symbol_index[target] << 32) | ELF_TYPES[kind]).to_bytes(8, "little") + \
                addend.to_bytes(8, "little", signed=True)
        tables.append(bytes(table))
    contents = [b""] + [bytes(entry["end"] - entry["start"]) for entry in layout.entries] + [data] + tables + \
        [bytes(symtab), bytes(strtab), shstr]
    types = [0] + [1] * len(layout.entries) + [1] + [4] * len(tables) + [2, 3, 3]
    flags = [0] + [0x6] * len(layout.entries) + [0x2] + [0x40] * len(tables) + [0, 0, 0]
    links = [0] * (len(layout.entries) + 2) + [symtab_index] * len(tables) + [symtab_index + 1, 0, 0]
    infos = [0] * (len(layout.entries) + 2) + [section_index[entry["name"]] for entry in relocated] + [0, 0, 0]
    sizes_of = [24 if kind in (2, 4) else 0 for kind in types]
    body, offsets = bytearray(), []
    for content in contents:
        offsets.append(64 + len(body))
        body += content
    header = bytearray(64)
    header[0:7] = b"\x7fELF\x02\x01\x01"
    header[16:18] = (1).to_bytes(2, "little")  # ET_REL
    header[18:20] = (183).to_bytes(2, "little")  # EM_AARCH64
    header[20:24] = (1).to_bytes(4, "little")  # EV_CURRENT
    header[0x28:0x30] = (64 + len(body)).to_bytes(8, "little")
    header[0x3A:0x3C] = (64).to_bytes(2, "little")
    header[0x3C:0x3E] = len(names).to_bytes(2, "little")
    header[0x3E:0x40] = (len(names) - 1).to_bytes(2, "little")
    sections = bytearray()
    for number, content in enumerate(contents):
        name_offset = shstr.index(b"\0" + names[number].encode() + b"\0") + 1 if number else 0
        sections += name_offset.to_bytes(4, "little") + types[number].to_bytes(4, "little") + \
            flags[number].to_bytes(8, "little") + bytes(8) + offsets[number].to_bytes(8, "little") + \
            len(content).to_bytes(8, "little") + links[number].to_bytes(4, "little") + \
            infos[number].to_bytes(4, "little") + bytes(8) + sizes_of[number].to_bytes(8, "little")
    return bytes(header) + bytes(body) + bytes(sections)


def _macho(layout: Layout, records: list[tuple], data: bytes) -> bytes:
    text_size = max([entry["end"] for entry in layout.entries], default=0)
    commands = 72 + 2 * 80 + 24
    text_offset = 32 + commands
    const_offset = text_offset + text_size
    strtab, nlist, symbol_index = bytearray(b"\0"), bytearray(), {}
    starts = {entry["name"]: entry["start"] for entry in layout.entries}
    for number, (name, kind, binding, owner, value) in enumerate(records):
        offset = len(strtab)
        strtab += name.encode() + b"\0"
        symbol_index.setdefault(name, number)
        if kind == "undefined":
            kind_bits, section, value = 0x01, 0, 0
        elif kind == "absolute":
            kind_bits, section = 0x03, 0
        else:
            kind_bits, section = (0x0F if binding != "local" else 0x0E), 1
            value = value if owner == name else starts.get(owner, 0) + value
        description = 0x80 if binding == "weak" and kind != "undefined" else (0x40 if binding == "weak" else 0)
        nlist += offset.to_bytes(4, "little") + bytes([kind_bits, section]) + description.to_bytes(2, "little") + \
            value.to_bytes(8, "little")
    relocations = bytearray()
    for entry in layout.entries:
        for address, (kind, target, _) in sorted(entry["relocations"].items()):
            pcrel = 1 if kind in ("ARM64_RELOC_BRANCH26", "ARM64_RELOC_PAGE21", "ARM64_RELOC_GOT_LOAD_PAGE21") else 0
            info = symbol_index[target] | (pcrel << 24) | (2 << 25) | (1 << 27) | (MACHO_TYPES[kind] << 28)
            relocations += address.to_bytes(4, "little") + info.to_bytes(4, "little")
    reloff = const_offset + len(data)
    symoff = reloff + len(relocations)
    stroff = symoff + len(nlist)

    def section(sectname: bytes, address: int, size: int, offset: int, flags: int, relocation: tuple) -> bytes:
        return sectname.ljust(16, b"\0") + b"__TEXT".ljust(16, b"\0") + address.to_bytes(8, "little") + \
            size.to_bytes(8, "little") + offset.to_bytes(4, "little") + bytes(4) + \
            relocation[0].to_bytes(4, "little") + relocation[1].to_bytes(4, "little") + \
            flags.to_bytes(4, "little") + bytes(12)
    segment = (0x19).to_bytes(4, "little") + (72 + 160).to_bytes(4, "little") + b"".ljust(16, b"\0") + bytes(32) + \
        bytes(8) + (2).to_bytes(4, "little") + bytes(4)
    segment += section(b"__text", 0, text_size, text_offset, 0x80000400, (reloff, len(relocations) // 8))
    segment += section(b"__const", text_size, len(data), const_offset, 0, (0, 0))
    symtab = (0x2).to_bytes(4, "little") + (24).to_bytes(4, "little") + symoff.to_bytes(4, "little") + \
        len(records).to_bytes(4, "little") + stroff.to_bytes(4, "little") + len(strtab).to_bytes(4, "little")
    header = bytes.fromhex("cffaedfe") + (0x0100000C).to_bytes(4, "little") + bytes(4) + \
        (1).to_bytes(4, "little") + (2).to_bytes(4, "little") + commands.to_bytes(4, "little") + bytes(8)  # MH_OBJECT
    return header + segment + symtab + bytes(text_size) + data + bytes(relocations) + bytes(nlist) + bytes(strtab)


def abi(symbol: str, macho: bool) -> str:
    """Raw linker name of a C/C++ symbol in the object format."""
    return "_" + symbol if macho else symbol


def to_abi(functions: list[tuple[str, str]], macho: bool) -> list[tuple[str, str]]:
    """Spell function names, branch targets and relocation targets as raw linker names of the format."""
    if not macho:
        return functions
    def line(text: str) -> str:
        note = RELOCATION_NOTE.search(text)
        if note:
            target = note.group(2)
            return text[:note.start(2)] + (target if target.startswith(".") else abi(target, True)) + text[note.end(2):]
        parts = text.strip().split(None, 1)
        if parts and BRANCH.match(parts[0]) and len(parts) > 1:
            operands = CHECKER.split_operands(parts[1])
            if not operands[-1].startswith(".L"):
                operands[-1] = abi(operands[-1], True)
                return f"\t{parts[0]} {', '.join(operands)}"
        return text
    return [(abi(symbol, True), "\n".join(line(text) for text in textwrap.dedent(body).splitlines()))
            for symbol, body in functions]


def abi_options(options: dict, macho: bool) -> dict:
    if not macho:
        return options
    mapped = dict(options)
    if "symbols" in options:
        mapped["symbols"] = [(abi(name, True), kind, binding, abi(owner, True) if owner else None, value)
                             for name, kind, binding, owner, value in options["symbols"]]
    for key in ("sizes", "bindings"):
        if key in options:
            mapped[key] = {abi(name, True): value for name, value in options[key].items()}
    return mapped


def elf_header_offset(data: bytes, wanted: str) -> int:
    """File offset of the ELF section header named wanted."""
    shoff, count, names = (int.from_bytes(data[0x28:0x30], "little"), int.from_bytes(data[0x3C:0x3E], "little"),
                           int.from_bytes(data[0x3E:0x40], "little"))
    strings = shoff + names * 64
    table = int.from_bytes(data[strings + 24:strings + 32], "little")
    for index in range(count):
        base = shoff + index * 64
        start = table + int.from_bytes(data[base:base + 4], "little")
        if data[start:data.index(b"\0", start)].decode() == wanted:
            return base
    raise KeyError(wanted)


def patched(data: bytes, offset: int, value: int, size: int) -> bytes:
    return data[:offset] + value.to_bytes(size, "little") + data[offset + size:]


def image(functions: list[tuple[str, str]], macho: bool = False, **options) -> CHECKER.ObjectImage:
    return CHECKER.read_object(witness_object(functions, None, macho, **options))


def programs(bodies: dict[tuple, str] | None = None, keys: list[tuple] | None = None) -> list[tuple[str, str]]:
    return [(name(*key), (bodies or {}).get(key, CLEAN)) for key in (keys if keys is not None else NEON_KEYS)]


def audit_witness(bodies: dict[tuple, str] | None = None, extra: list[tuple[str, str]] | None = None,
                  keys: list[tuple] | None = None, inventory: str | None = NEON_INVENTORY, mode: str = "NEON",
                  macho: bool = False, symbols: list[tuple] | None = None,
                  sizes: dict[str, int] | None = None) -> dict[str, int]:
    functions = to_abi(programs(bodies, keys) + (extra or []), macho)
    options = abi_options({key: value for key, value in (("symbols", symbols), ("sizes", sizes)) if value}, macho)
    return CHECKER.audit(witness_object(functions, inventory, macho, **options), disassembly(functions, macho), mode)


def budget(limit: int = 4) -> dict:
    """Budget naming every grouped NEON program."""
    limits = {":".join(map(str, key)): limit for key in NEON_KEYS if key[0] != 1}
    return {"GNU-14": {"NEON": limits}}


class Result:
    """Spills and structural errors of one function, and the raw names proven never to return."""

    def __init__(self, spills: int, errors: list[str], noreturn: set[str], fatal: str) -> None:
        self.spills, self.errors, self.noreturn, self.fatal = spills, errors, noreturn, fatal


def analyse(body: str, extra: dict[str, str] | None = None, macho: bool = False, raw: bool = False,
            **options) -> Result:
    """Run the production front end and analysis over one function (plus optional other functions).

    Names are logical C/C++ names spelled in the format's ABI, unless raw=True (names are linker names).
    """
    functions = [("f", body)] + list((extra or {}).items())
    if not raw:
        functions, options = to_abi(functions, macho), abi_options(options, macho)
    target = functions[0][0]
    obj = CHECKER.read_object(witness_object(functions, macho=macho, **options))
    bodies = CHECKER.normalize_disassembly(disassembly(functions, macho), obj)
    noreturn = CHECKER.noreturn_symbols(bodies, CHECKER.GENERIC, CHECKER.trusted_imports(obj), obj.prefix)
    spills, errors = CHECKER.analyse_function(target, bodies, CHECKER.GENERIC, noreturn)
    return Result(spills, errors, noreturn, obj.prefix + FATAL)


class ClassificationTest(unittest.TestCase):
    """FP/SIMD traffic through any stack provenance is a spill; analysis is control-flow aware."""

    def spills(self, body: str, **options) -> int:
        analysis = analyse(body, **options)
        self.assertEqual(analysis.errors, [], body)
        return analysis.spills

    def fatal(self, body: str, fragment: str, **options) -> None:
        analysis = analyse(body, **options)
        self.assertTrue(any(fragment in error for error in analysis.errors), (body, analysis.errors))

    def test_every_base_and_width_is_classified(self) -> None:
        for line in ("str q31, [sp, 160]", "ldr q25, [sp, #0xa0]", "stp q0, q1, [sp, 32]", "stur q0, [x29, -16]",
                     "str q0, [x29, #16]", "str s3, [sp, 12]", "ldr d17, [x29, 40]", "str h2, [sp, 4]",
                     "st1 {v0.16b}, [sp]", "ldr b1, [sp, 1]", "st1 {v0.16b, v1.16b}, [sp]", "ld1 {v2.s}[1], [sp]",
                     "str q0, [sp, x9]", "ldr q0, [sp, w9, sxtw 4]", "str q0, [sp, -32]!", "ldr q0, [sp], 16"):
            with self.subTest(line=line):
                self.assertEqual(self.spills(f"\t{line}\n\tret\n"), 1)

    def test_stack_aliases_including_shifted_and_indexed_forms(self) -> None:
        self.assertEqual(self.spills("\tadd x9, sp, #64\n\tstr q0, [x9]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tmov x10, sp\n\tldr q4, [x10, 32]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tsub x11, x29, #32\n\tstr d18, [x11]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tadd x9, sp, #1, lsl #12\n\tstr q0, [x9]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tadd x9, sp, x3\n\tstr q0, [x9]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tmov x9, sp\n\tstr q0, [x9, x4, lsl 4]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tadd x9, sp, 64\n\teor x9, x9, x2\n\tstr q0, [x9]\n\tret\n"), 1)
        self.fatal("\tadd x9, sp, 64\n\tstr q0, [x1, x9]\n\tret\n", "index register")

    def test_any_other_definition_retires_an_alias(self) -> None:
        self.assertEqual(self.spills("\tadd x9, sp, #64\n\tadd x9, x1, #8\n\tstr q0, [x9]\n\tret\n"), 0)
        self.assertEqual(self.spills("\tadd x9, sp, #64\n\tldr x9, [x1]\n\tstr q0, [x9]\n\tret\n"), 0)
        self.assertEqual(self.spills("\tadd x9, sp, #64\n\tmov w9, 3\n\tstr q0, [x9]\n\tret\n"), 0)
        self.assertEqual(self.spills("\tadd x0, sp, #64\n\tldp x0, x1, [x2]\n\tstr q0, [x0]\n\tret\n"), 0)
        self.assertEqual(self.spills("\tldr q0, [x1, 16]\n\tstr q0, [x2]\n\tret\n"), 0)

    def test_branch_join_of_stack_and_heap_base_is_fatal(self) -> None:
        self.fatal("\tmov x9, sp\n\tcbz w0, .Luse\n\tmov x9, x1\n.Luse:\n\tldr q0, [x9]\n\tret\n", "only some paths")
        self.fatal("\tadd x9, sp, 64\n\tcsel x9, x9, x1, eq\n\tstr q0, [x9]\n\tret\n", "only some paths")
        self.assertEqual(self.spills("\tmov x9, sp\n\tcbz w0, .Luse\n\tadd x9, sp, 32\n.Luse:\n\tldr q0, [x9]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tmov x9, x1\n\tcbz w0, .Luse\n\tmov x9, x2\n.Luse:\n\tldr q0, [x9]\n\tret\n"), 0)

    def test_loops_reach_a_fixed_point(self) -> None:
        body = "\tmov x9, sp\n.Lloop:\n\tstr q0, [x9], 16\n\tsub x2, x2, 1\n\tcbnz x2, .Lloop\n\tret\n"
        self.assertEqual(self.spills(body), 1)

    def test_stack_addresses_stored_in_slots_and_escapes(self) -> None:
        self.assertEqual(self.spills("\tadd x9, sp, 32\n\tstr x9, [sp, 8]\n\tldr x10, [sp, 8]\n\tstr q0, [x10]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tadd x9, sp, 32\n\tstr x9, [sp, 8]\n\tstr x1, [sp, 8]\n\tldr x10, [sp, 8]\n"
                                     "\tstr q0, [x10]\n\tret\n"), 0)
        self.fatal("\tadd x9, sp, 32\n\tstr x9, [x1]\n\tret\n", "cannot be tracked")
        self.fatal("\tadd x9, sp, 32\n\tstr x9, [sp, x2]\n\tret\n", "cannot be tracked")
        self.fatal("\tmov x9, sp\n\tfmov d0, x9\n\tret\n", "FP/SIMD register")

    def test_unknown_offset_stack_access_with_live_pointer_slots_fails(self) -> None:
        frame = "\tsub sp, sp, #32\n\tadd x9, sp, #16\n\tstr x9, [sp]\n"
        for access in ("ldr w11, [sp, x10]", "ldp w10, w11, [sp, x10]", "ldr q1, [sp, x3]", "ldr x11, [sp, x3, lsl 3]"):
            with self.subTest(access=access):
                self.fatal(frame + f"\t{access}\n\tadd sp, sp, #32\n\tret\n", "unknown offset")
        self.assertEqual(self.spills("\tsub sp, sp, #32\n\tldr q1, [sp, x3]\n\tadd sp, sp, #32\n\tret\n"), 1)

    def test_pointer_slot_joins_keep_the_non_stack_alternative(self) -> None:
        joined = ("\tcbz w0, .La\n\tadd x9, sp, #32\n\tstr x9, [sp, 8]\n\tb .Lj\n.La:\n\tstr x1, [sp, 8]\n.Lj:\n"
                  "\tldr x10, [sp, 8]\n\tstr q0, [x10]\n\tret\n")
        self.fatal(joined, "only some paths")
        self.fatal(joined.replace("\tstr x1, [sp, 8]\n", ""), "only some paths")
        self.fatal("\tldr x10, [sp]\n\tadd x9, sp, #32\n\tstr x9, [x10]\n\tldr x11, [x1]\n\tstr q0, [x11]\n\tret\n",
                   "cannot be tracked")
        composed = ("\tcbz w0, .La\n\tadd x9, sp, #32\n\tstr x9, [sp, 8]\n\tb .Lj\n.La:\n\tstr x1, [sp, 8]\n.Lj:\n"
                    "\tldr x10, [sp, 8]\n\tadd x9, sp, #32\n\tstr x9, [x10]\n\tldr x11, [x1]\n\tstr q0, [x11]\n\tret\n")
        self.fatal(composed, "only some paths")  # The composed escape fails at its ambiguous base.

    def test_conditional_transforms_of_stack_addresses_are_not_admitted(self) -> None:
        for transform in ("csinc x9, x9, x9, eq", "csinv x9, x9, x1, ne", "csneg x9, x9, x9, lt", "cinc x9, x9, eq"):
            with self.subTest(transform=transform):
                self.fatal(f"\tmov x9, sp\n\t{transform}\n\tstr q0, [x9]\n\tret\n", "admitted grammar")
        self.fatal("\tmov x9, sp\n\tcsel x9, x9, x1, eq\n\tstr q0, [x9]\n\tret\n", "only some paths")

    def test_structure_footprints_use_the_exact_lane_and_arrangement(self) -> None:
        tail = "\tldr x10, [sp, #8]\n\tstr q0, [x10]\n\tadd sp, sp, #32\n\tret\n"
        frame = "\tsub sp, sp, #32\n\tadd x9, sp, #16\n\tstr x9, [sp, #8]\n"
        for macho in (False, True):
            for lane in ("\tst1 {v0.s}[0], [sp]\n", "\tst1 {v0.8b}, [sp]\n", "\tst1 {v0.4h}, [sp]\n"):
                with self.subTest(lane=lane, macho=macho):
                    analysis = analyse(frame + lane + tail, macho=macho)
                    self.assertEqual((analysis.spills, analysis.errors), (2, []))
            for overlap in ("\tadd x11, sp, #12\n\tst1 {v0.s}[0], [x11]\n", "\tadd x11, sp, #4\n\tst1 {v0.2s}, [x11]\n"):
                with self.subTest(overlap=overlap, macho=macho):
                    self.assertTrue(any("partial overwrite" in error
                                        for error in analyse(frame + overlap + tail, macho=macho).errors), overlap)
            with self.subTest(cover=macho):  # A whole 16-byte store retires the slot: x10 is no stack address.
                analysis = analyse(frame + "\tst1 {v0.16b}, [sp]\n" + tail, macho=macho)
                self.assertEqual((analysis.spills, analysis.errors), (1, []))
        for unknown in ("\tst1 {v0.16b-v1.16b}, [sp]\n", "\tst1 {v0}, [sp]\n", "\tld2 {v0.4s}, [sp]\n"):
            with self.subTest(unknown=unknown):
                self.fatal("\tsub sp, sp, #32\n" + unknown + "\tadd sp, sp, #32\n\tret\n", "footprint")

    def test_merge_definitions_keep_their_destination(self) -> None:
        self.assertEqual(self.spills("\tmov x9, sp\n\tbfi x9, xzr, #0, #1\n\tstr q0, [x9]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tmov x9, sp\n\tmovk x9, #0, lsl #48\n\tstr q0, [x9]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tmov x9, x1\n\tbfi x9, xzr, #0, #1\n\tstr q0, [x9]\n\tret\n"), 0)
        self.fatal("\tmov x9, sp\n\tbfi w9, wzr, #0, #1\n\tret\n", "32-bit view")

    def test_32_bit_views_and_partial_slot_reuse_fail(self) -> None:
        frame = "\tsub sp, sp, #32\n\tadd x9, sp, #16\n\tstr x9, [sp]\n"
        self.fatal(frame + "\tldp w10, w11, [sp]\n\torr x9, x10, x11, lsl #32\n\tstr q0, [x9]\n\tadd sp, sp, #32\n\tret\n",
                   "reload of a stored stack address")
        self.fatal(frame + "\tldrsw x10, [sp]\n\tadd sp, sp, #32\n\tret\n", "reload of a stored stack address")
        self.fatal(frame + "\tldr d1, [sp]\n\tadd sp, sp, #32\n\tret\n", "reload of a stored stack address")
        self.fatal(frame + "\tstr w1, [sp, 4]\n\tadd sp, sp, #32\n\tret\n", "partial overwrite")
        self.fatal("\tmov x9, sp\n\tmov w10, w9\n\tret\n", "32-bit view")
        self.fatal("\tmov x9, sp\n\tstr w9, [x1]\n\tret\n", "32-bit view")
        self.fatal("\tmov x9, sp\n\tldr q0, [x1, w9, sxtw]\n\tret\n", "index register")
        self.assertEqual(self.spills(frame + "\tldr x10, [sp]\n\tstr q0, [x10]\n\tadd sp, sp, #32\n\tret\n"), 1)
        self.assertEqual(self.spills(frame + "\tstr q1, [sp]\n\tldr x10, [sp]\n\tstr q0, [x10]\n"
                                     "\tadd sp, sp, #32\n\tret\n"), 1)  # The q store spills; x10 is no longer stack.

    def test_every_call_boundary_is_structural(self) -> None:
        for body, extra in (("\tmov x0, sp\n\tbl identity\n\tstr q0, [x0]\n\tret\n", {"identity": "\tret\n"}),
                            ("\tmov x8, sp\n\tbl helper\n\tstr q0, [x8]\n\tret\n", {"helper": "\tret\n"}),
                            ("\tstp d8, d9, [sp, -32]!\n\tfmov d8, d0\n\tmov x0, sp\n\tbl helper\n"
                             "\tldp d8, d9, [sp], 32\n\tret\n", {"helper": "\tstr xzr, [x0]\n\tret\n"}),
                            ("\tbl helper\n\tret\n", {"helper": "\tret\n"}),
                            ("\tb helper\n", {"helper": "\tret\n"}),
                            ("\tbl external\n\tret\n", None)):
            with self.subTest(body=body):
                self.fatal(body, "unsupported call boundary", extra=extra)
        with self.assertRaisesRegex(ValueError, "unsupported branch relocation"):
            analyse("\tcbz x0, helper\n\tret\n", {"helper": "\tret\n"})

    def test_relocated_data_addresses_are_never_stack(self) -> None:
        functions = [("f", "\tadrp x3, 0x0 <f> !R_AARCH64_ADR_GOT_PAGE:stderr\n"
                           "\tldr x3, [x3] !R_AARCH64_LD64_GOT_LO12_NC:stderr\n\tstr q0, [sp, #16]\n\tret\n")]
        bodies = CHECKER.normalize_disassembly(disassembly(functions), image(functions))
        self.assertEqual([line.strip() for line in bodies["f"][1:3]], ["adrp\tx3, stderr", "ldr\tx3, [x3, :lo12:stderr]"])
        self.assertEqual(CHECKER.analyse_function("f", bodies, CHECKER.GENERIC, set()), (1, []))
        for instruction in ("ldr q0, [sp] !R_AARCH64_LDST128_ABS_LO12_NC:x", "add x9, sp, #0 !R_AARCH64_ADD_ABS_LO12_NC:x"):
            with self.subTest(instruction=instruction):
                self.fatal(f"\t{instruction}\n\tstr q0, [x9]\n\tret\n", "relocated")


class ABITest(unittest.TestCase):
    """ABI saves of d8-d15 are exempt only when proven; count ceilings are not proof."""

    PROLOGUE = "\tstp d8, d9, [sp, -32]!\n"
    EPILOGUE = "\tldp d8, d9, [sp], 32\n\tret\n"

    def check(self, body: str) -> Result:
        return analyse(body)

    def test_proven_save_and_restore_are_not_spills(self) -> None:
        analysis = self.check(self.PROLOGUE + "\tfmov d8, d0\n\tfmul v9.4s, v1.4s, v2.4s\n" + self.EPILOGUE)
        self.assertEqual((analysis.spills, analysis.errors), (0, []))
        shrink = ("\tcbz w0, .Lout\n" + self.PROLOGUE + "\tfmov d8, d0\n\tldp d8, d9, [sp], 32\n.Lout:\n\tret\n")
        self.assertEqual(self.check(shrink).errors, [])

    def assertFatal(self, body: str, fragment: str) -> None:
        errors = self.check(body).errors
        self.assertTrue(any(fragment in error for error in errors), (body, errors))

    def test_unproven_saves_are_fatal(self) -> None:
        self.assertFatal("\tfmov d8, d0\n" + self.PROLOGUE + self.EPILOGUE, "before saving")
        self.assertFatal("\tfmov x9, d8\n\tfmov d8, d0\n\tfmov d8, x9\n\tret\n", "before saving")
        self.assertFatal(self.PROLOGUE + "\tfmov d8, d0\n\tret\n", "without restoring")
        self.assertFatal(self.PROLOGUE + "\tfmov d8, d0\n\tldr d8, [sp, 8]\n\tldr d9, [sp, 8]\n\tadd sp, sp, 32\n\tret\n",
                         "without restoring")
        self.assertFatal(self.PROLOGUE + "\tfmov d8, d0\n\tldp d8, d9, [sp], 32\n\tfmov d8, d1\n\tret\n",
                         "after its epilogue restore")
        self.assertFatal(self.PROLOGUE + "\tstr q0, [sp, 8]\n" + self.EPILOGUE, "live save slot")
        self.assertFatal(self.PROLOGUE + ".Lloop:\n\tstr d8, [sp]\n\tfmov d8, d0\n\tcbnz x0, .Lloop\n" + self.EPILOGUE,
                         "live save slot")
        self.assertFatal(self.PROLOGUE + "\tfmov d8, d0\n\tldp d8, d9, [sp]\n\tcbnz x0, .Lx\n.Lx:\n\tldr d8, [sp]\n"
                         "\tadd sp, sp, 32\n\tret\n", "after its epilogue restore")
        self.assertFatal(self.PROLOGUE + "\tstr q0, [sp, x3]\n" + self.EPILOGUE, "unknown offset")

    def test_lane_structure_and_replicating_forms_are_not_restores(self) -> None:
        for reload in ("ld1 {v8.d}[1], [sp]", "ld1r {v8.2d}, [sp]", "ld2 {v8.d, v9.d}[0], [sp]", "ld1 {v8.1d}, [sp]"):
            with self.subTest(reload=reload):
                analysis = self.check(f"\tsub sp, sp, #16\n\tstr d8, [sp]\n\tfmov d8, d0\n\t{reload}\n"
                                      "\tadd sp, sp, #16\n\tret\n")
                self.assertTrue(any("without restoring" in error for error in analysis.errors), analysis.errors)
        for save in ("st1 {v8.1d}, [sp]", "st1 {v8.d}[0], [sp]", "str s8, [sp]"):
            with self.subTest(save=save):
                analysis = self.check(f"\tsub sp, sp, #16\n\t{save}\n\tfmov d8, d0\n\tldr d8, [sp]\n"
                                      "\tadd sp, sp, #16\n\tret\n")
                self.assertTrue(any("before saving" in error for error in analysis.errors), analysis.errors)

    def test_unmatched_callee_saved_reloads_are_spills(self) -> None:
        body = self.PROLOGUE + "\tsub sp, sp, 16\n\tfmov d8, d0\n\tstr d8, [sp]\n\tldr d8, [sp]\n\tadd sp, sp, 16\n" + self.EPILOGUE
        analysis = self.check(body)
        self.assertEqual((analysis.spills, analysis.errors), (2, []))


class StructuralTest(unittest.TestCase):
    """Closure and classification failures are fatal, independent of any spill allowance."""

    def assertFatal(self, body: str, fragment: str, extra: dict[str, str] | None = None) -> None:
        errors = analyse(body, extra).errors
        self.assertTrue(any(fragment in error for error in errors), (body, errors))

    def test_callees(self) -> None:
        self.assertFatal("\tbl helper\n\tret\n", "unsupported call boundary")
        self.assertFatal("\tb helper\n", "unsupported call boundary")
        for indirect in ("\tblr x8\n\tret\n", "\tbr x8\n", "\tret x1\n", "\tblraaz x8\n\tret\n", "\tudf #0\n\tret\n",
                         "\t<unknown>\n\tret\n"):
            with self.subTest(indirect=indirect):
                self.assertFatal(indirect, "admitted grammar")
        self.assertFatal("\tb .Lnowhere\n", "not code of this function")
        self.assertFatal("\tbl .Llocal\n\tret\n.Llocal:\n\tret\n", "not code of this function")

    def test_fatal_diagnostic_must_be_exact_and_proven(self) -> None:
        caller = f"\tcbz x0, .Lok\n\tbl {FATAL}\n.Lok:\n\tret\n"
        self.assertEqual(analyse(caller, {FATAL: FATAL_BODY}).errors, [])
        self.assertEqual(analyse(caller, {FATAL: FATAL_BODY.replace("fwrite", "fputs")}, macho=True).errors, [])
        both = "\tcbz x0, .La\n\tbl fputs\n\tbl abort\n.La:\n\tbl abort\n"
        self.assertEqual(analyse(caller, {FATAL: both}).errors, [])
        self.assertFatal(f"\tbl {FATAL[:-1]}X\n\tret\n", "unsupported call boundary")
        self.assertFatal(f"\tbl {FATAL}\n\tret\n", "unsupported call boundary")
        self.assertFatal("\tbl abort\n", "unsupported call boundary")
        for body in ("\tret\n", ".Lspin:\n\tb .Lspin\n", "\tmov x8, #93\n\tmov x0, #0\n\tsvc #0\n\tbl abort\n",
                     "\tbl exit\n\tbl abort\n", "\tb.eq .Lx\n\tbl abort\n.Lx:\n\tret\n", "\tcbz x0, .Ldone\n.Lspin:\n\tb .Lspin\n.Ldone:\n\tbl abort\n",
                     ".Lloop:\n\tstr q0, [sp, -16]!\n\tcbnz x0, .Lloop\n\tbl abort\n", "\tcbz x0, .Lr\n\tbl abort\n.Lr:\n\tret\n",
                     "\tb fputs\n", "\tblr x1\n", "\tbl fputs\n", "\t.word 0x0\n\tbl abort\n"):
            with self.subTest(body=body):
                self.assertNotIn(FATAL, analyse("\tret\n", {FATAL: body}).noreturn)
                self.assertFatal(f"\tbl {FATAL}\n\tret\n", "unsupported call boundary", {FATAL: body})

    def test_unclassifiable_code_is_fatal(self) -> None:
        self.assertFatal("\tldr q0, [sp, w1, ror 3]\n\tret\n", "addressing")
        self.assertFatal("\tcasp x0, x1, x2, x3, [sp]\n\tret\n", "admitted grammar")
        self.assertFatal("\tldr q0, [x1], x2\n\tret\n", "admitted grammar")
        self.assertFatal("\tadd\n\tret\n", "admitted grammar")
        self.assertFatal("\tadd x9\n\tret\n", "admitted grammar")
        self.assertFatal("\tadd x9, x1, x2, x3\n\tret\n", "admitted grammar")
        self.assertFatal("\tsmmla\n\tret\n", "admitted grammar")
        self.assertFatal("\tsvc #0\n\tret\n", "admitted grammar")
        self.assertFatal("\twfi\n\tret\n", "admitted grammar")
        self.assertFatal("\tfrobnicate x9, x1\n\tret\n", "admitted grammar")
        self.assertFatal("\tadd x9, x1, @@\n\tret\n", "admitted grammar")
        self.assertFatal("\tret\n\tstr q0, [sp]\n", "unreachable")
        self.assertFatal("\tmov x0, x1\n", "falls off")

    def test_data_words_in_code_are_fatal(self) -> None:
        frame = "\tsub sp, sp, #16\n{}\n\tadd sp, sp, #16\n\tret\n"
        for word in ("\t.word 0x3d8003e0", "\t.inst 0x3d8003e0", "\t.byte 0xe0", "\t.short 0x3"):
            with self.subTest(word=word):
                self.assertTrue(any("inside code" in error for error in analyse(frame.format(word)).errors))
        self.assertEqual(analyse("\tret\n\tnop\n\tnop\n").errors, [])  # Unreachable alignment padding.

    def test_body_without_executable_entry_section_fails(self) -> None:
        self.assertTrue(CHECKER.analyse_function("f", {"f": ["\tret"]}, CHECKER.GENERIC, set())[1])
        self.assertTrue(CHECKER.analyse_function("f", {"f": ["\t.section .rodata", "\tret"]}, CHECKER.GENERIC, set())[1])


class DisassemblyTest(unittest.TestCase):
    """The disassembly contract: every line, relocation and operand is classified or fails."""

    TEXT = disassembly([("f", "\tadd x0, x1, #1\n\tret\n")])

    def test_unparseable_skipped_and_misplaced_lines_fail(self) -> None:
        for bad in (self.TEXT + "garbage line\n", self.TEXT + "\t...\n",
                    self.TEXT + "\t\t0000000000000040:  R_AARCH64_CALL26\tx\n",
                    self.TEXT.replace("ret\n", "ret\n\t\t0000000000000004:  R_AARCH64_CALL26\tx\n"
                                                "\t\t0000000000000004:  R_AARCH64_CALL26\ty\n"),
                    self.TEXT.replace("Disassembly of section .text.f:\n", ""),
                    self.TEXT + self.TEXT.split("\n", 1)[0] + "\n",
                    self.TEXT.replace("ret\n", "ret\n\t\t0000000000000004:  R_AARCH64_MOVW_UABS_G0\tx\n"),
                    self.TEXT.replace("add\tx0, x1, #1", "ldr\tx0, 0x40 <f+0x40>")):
            with self.subTest(bad=bad[-80:]):
                with self.assertRaises(ValueError):
                    CHECKER.normalize_disassembly(bad, image([("f", "\tadd x0, x1, #1\n\tret\n")]))

    def test_branch_relocation_addends_fail_and_data_addends_are_dropped(self) -> None:
        call = [("f", "\tbl helper !R_AARCH64_CALL26:helper+0x8\n\tret\n")]
        with self.assertRaisesRegex(ValueError, "addend"):
            CHECKER.normalize_disassembly(disassembly(call), image(call))
        page = [("f", "\tadrp x1, 0x0 <f> !R_AARCH64_ADR_PREL_PG_HI21:.rodata+0x38\n"
                      "\tadd x1, x1, #0 !R_AARCH64_ADD_ABS_LO12_NC:.rodata-0x8\n\tret\n")]
        text = disassembly(page)
        self.assertIn(".rodata+0x38", text)
        self.assertEqual([line.strip() for line in CHECKER.normalize_disassembly(text, image(page))["f"][1:3]],
                         ["adrp\tx1, .rodata", "add\tx1, x1, :lo12:.rodata"])  # Section symbol, named by section.

    def test_object_format_must_match_the_disassembly(self) -> None:
        single = [("f", "\tret\n")]
        with self.assertRaises(ValueError):
            CHECKER.normalize_disassembly(disassembly(single), image(single, macho=True))
        with self.assertRaises(ValueError):
            CHECKER.normalize_disassembly(disassembly(single, macho=True), image(single))
        self.assertEqual(sorted(CHECKER.normalize_disassembly(disassembly(single, macho=True), image(single, macho=True))),
                         ["f"])  # Mach-O symbols lose their leading underscore.

    def test_gnu_objdump_format(self) -> None:
        gnu = ("probe.o:     file format elf64-littleaarch64\n\n\nDisassembly of section .text.f:\n\n"
               "0000000000000000 <f>:\n"
               "   0:\tadrp\tx3, 0 <stderr>\n\t\t\t0: R_AARCH64_ADR_GOT_PAGE\tstderr\n"
               "   4:\tldr\tx3, [x3]\n\t\t\t4: R_AARCH64_LD64_GOT_LO12_NC\tstderr\n"
               "   8:\tmov\tx2, #0x45                  \t// #69\n"
               "   c:\tcmp\tx2, #0x0\n"
               "  10:\tb.ne\t8 <f+0x8>  // b.any\n"
               "  14:\tbl\t0 <fwrite>\n\t\t\t14: R_AARCH64_CALL26\tfwrite\n"
               "  18:\tbl\t0 <abort>\n\t\t\t18: R_AARCH64_CALL26\tabort\n")
        matching = [("f", "\tadrp x3, 0x0 <f> !R_AARCH64_ADR_GOT_PAGE:stderr\n"
                          "\tldr x3, [x3] !R_AARCH64_LD64_GOT_LO12_NC:stderr\n.L8:\n\tmov x2, #0x45\n"
                          "\tcmp x2, #0x0\n\tb.ne .L8\n\tbl fwrite\n\tbl abort\n")]
        bodies = CHECKER.normalize_disassembly(gnu, image(matching))
        self.assertEqual([line.strip() for line in bodies["f"][1:]],
                         ["adrp\tx3, stderr", "ldr\tx3, [x3, :lo12:stderr]", ".L8:", "mov\tx2, #0x45",
                          "cmp\tx2, #0x0", "b.ne\t.L8", "bl\tfwrite", "bl\tabort"])

    def test_object_format_and_embedded_inventory(self) -> None:
        self.assertFalse(CHECKER.object_format(witness_object()))
        self.assertTrue(CHECKER.object_format(witness_object(macho=True)))
        foreign = bytearray(witness_object())
        foreign[18:20] = (62).to_bytes(2, "little")  # x86-64
        for data in (bytes(foreign), b"MZ" + bytes(64), b""):
            with self.assertRaises(ValueError):
                CHECKER.object_format(data + bytes(32))
        self.assertEqual(CHECKER.object_inventory(witness_object()), [NEON_INVENTORY])
        self.assertEqual(CHECKER.object_inventory(witness_object(inventory=None)), [])
        doubled = witness_object(extra=b"llaminar-native-neon-tile-probe:" + NEON_INVENTORY.encode() + b"\0")
        self.assertEqual(len(CHECKER.object_inventory(doubled)), 2)


class CoverageTest(unittest.TestCase):
    """Function bodies are cut from object extents over a complete, ordered disassembly."""

    BODY = [("f", "\tmov x9, sp\n\tadd x9, x9, #16\n\tstr q0, [x9]\n\tret\n")]

    def normalize(self, text: str, **options) -> dict:
        return CHECKER.normalize_disassembly(text, image(self.BODY, **options))

    def lines(self) -> list[str]:
        return disassembly(self.BODY).splitlines()

    def test_internally_consistent_witness_passes_in_both_formats(self) -> None:
        for macho in (False, True):
            functions = self.BODY + [("g", "\tret\n")]
            bodies = CHECKER.normalize_disassembly(disassembly(functions, macho), image(functions, macho))
            self.assertEqual(CHECKER.analyse_function("f", bodies, CHECKER.GENERIC, set()), (1, []))

    def test_deleted_prefix_interior_tail_duplicate_and_reordered_instructions_fail(self) -> None:
        code = [index for index, line in enumerate(self.lines()) if re.match(r"^\s+[0-9a-f]+: ", line)]
        variants = {"prefix": [line for index, line in enumerate(self.lines()) if index != code[0]],
                    "interior": [line for index, line in enumerate(self.lines()) if index != code[1]],
                    "tail": [line for index, line in enumerate(self.lines()) if index != code[-1]],
                    "duplicate": self.lines() + [self.lines()[code[-1]]],
                    "reordered": self.lines()[:code[1]] + [self.lines()[code[2]], self.lines()[code[1]]]
                    + self.lines()[code[2] + 1:]}
        for label, lines in variants.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    self.normalize("\n".join(lines) + "\n")

    def test_symbol_extents_must_match_the_object(self) -> None:
        text = disassembly(self.BODY)
        with self.assertRaisesRegex(ValueError, "outside every function extent"):
            self.normalize(text, sizes={"f": 8})
        with self.assertRaisesRegex(ValueError, "outside its section"):
            self.normalize(text, sizes={"f": 32})
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.normalize(text, symbols=[("inner", "function", "global", "f", 4)], sizes={"inner": 8})
        self.normalize(text, symbols=[("alias", "function", "global", "f", 0)], sizes={"alias": 16})

    def test_headings_and_sections_must_be_the_objects(self) -> None:
        with self.assertRaisesRegex(ValueError, "heading"):
            self.normalize(disassembly(self.BODY).replace("<f>:", "<g>:"))
        with self.assertRaisesRegex(ValueError, "heading"):
            self.normalize(disassembly(self.BODY).replace("0000000000000000 <f>:", "0000000000000004 <f>:"))
        two = self.BODY + [("g", "\tret\n")]
        swapped = disassembly(two).replace("section .text.f:", "section .text.TMP:").replace(
            "section .text.g:", "section .text.f:").replace("section .text.TMP:", "section .text.g:")
        with self.assertRaises(ValueError):
            CHECKER.normalize_disassembly(swapped, image(two))
        with self.assertRaisesRegex(ValueError, "cover|does not define"):
            self.normalize(disassembly(self.BODY).replace("section .text.f:", "section .text.other:"))

    def test_elf_data_mapping_symbol_inside_a_function_is_data_in_code(self) -> None:
        bodies = self.normalize(disassembly(self.BODY), symbols=[("$d", "notype", "local", "f", 8)])
        self.assertTrue(any("inside code" in error
                            for error in CHECKER.analyse_function("f", bodies, CHECKER.GENERIC, set())[1]))


class TrustedImportTest(unittest.TestCase):
    """abort/fwrite/fputs are trusted only as unique, undefined, strongly bound imports of the object."""

    CALLER = f"\tcbz x0, .Lok\n\tbl {FATAL}\n.Lok:\n\tret\n"

    def outcome(self, macho: bool, extra: dict[str, str], **options) -> Result:
        return analyse(self.CALLER, {FATAL: FATAL_BODY.replace("fwrite", "fputs"), **extra}, macho=macho, **options)

    def test_undefined_imports_are_trusted(self) -> None:
        for macho in (False, True):
            with self.subTest(macho=macho):
                result = self.outcome(macho, {})
                self.assertEqual(result.errors, [])
                self.assertIn(result.fatal, result.noreturn)

    def test_trusted_imports_need_the_exact_abi_spelling(self) -> None:
        caller = f"\tcbz x0, .Lok\n\tbl {{fatal}}\n.Lok:\n\tret\n"
        for macho, calls, trusted in ((True, ("_fputs", "_abort"), True), (True, ("fputs", "_abort"), False),
                                      (True, ("_fputs", "abort"), False), (False, ("fputs", "abort"), True),
                                      (False, ("_fputs", "abort"), False), (False, ("fputs", "_abort"), False)):
            with self.subTest(macho=macho, calls=calls):
                fatal = abi(FATAL, macho)
                result = analyse(caller.format(fatal=fatal), {fatal: f"\tbl {calls[0]}\n\tbl {calls[1]}\n"},
                                 macho=macho, raw=True)
                self.assertEqual(fatal in result.noreturn, trusted)
                self.assertEqual(not result.errors, trusted, result.errors)

    def test_defined_rebound_weak_and_aliased_trusted_names_are_not(self) -> None:
        exit_body = "\tmov x8, #93\n\tmov x0, #0\n\tsvc #0\n"
        for macho in (False, True):
            for label, extra, options in (
                    ("defined fputs", {"fputs": exit_body}, {}),
                    ("defined abort", {"abort": "\tret\n"}, {}),
                    ("weak abort", {}, {"bindings": {"abort": "weak"}}),
                    ("local alias", {}, {"symbols": [("abort", "function", "local", FATAL, 0)],
                                         "sizes": {"abort": 16}}),
                    ("absolute alias", {}, {"symbols": [("fputs", "absolute", "global", None, 0)]})):
                with self.subTest(label=label, macho=macho):
                    result = self.outcome(macho, extra, **options)
                    self.assertNotIn(result.fatal, result.noreturn)
                    self.assertTrue(any("unsupported call boundary" in error for error in result.errors), result.errors)


class AssemblerInputTest(unittest.TestCase):
    """Files read by the assembler through inline assembly are outside the dependency closure: rejected."""

    def test_assembler_file_directives_are_found_in_every_spelling(self) -> None:
        for source in ('asm(".incbin \\"blob\\"");', 'asm(".inc" "bin \\"x\\"");', 'asm volatile("\\x2eincbin x");',
                       'asm(R"(.include "x")");', 'void g() { __asm__ __volatile__ (".INCLUDE \\"y\\"" ::: "memory"); }',
                       'asm("\\056include \\"z\\"");'):
            with self.subTest(source=source):
                self.assertTrue(CHECKER.inline_assembly_violations(source))

    def test_tokenization_and_assembler_construction(self) -> None:
        for source in ('constexpr auto marker = 1\'000; asm(".include \\"extra.s\\"");',
                       'auto r = R"x(a")x"; asm(".incbin \\"y\\"");',
                       "char c = '\"'; asm(\".incbin x\");",
                       'asm(R"(.macro external suffix\n.inc\\suffix "extra.s"\n.endm\nexternal lude)");',
                       'asm(".irpc c,lude\\n.inc\\\\c x\\n.endr");', 'asm(".p2align 2");', 'asm("nop; nop");',
                       'asm("nop");'):
            with self.subTest(source=source):
                self.assertTrue(CHECKER.inline_assembly_violations(source))

    def test_every_asm_is_visited_and_labels_need_declaration_context(self) -> None:
        for source in ('void f() { asm volatile("" : : "r"([] { asm volatile(".include \\"extra.s\\""); '
                       'return 0; }())); }',
                       'void f() { asm("nop"); }', 'asm("nop");', 'void f(int x) { if (x) asm("nop"); }',
                       'void f() { extern int g __asm("h"); }', 'int f() __asm__ volatile("g");',
                       'int f() __asm__("g" : );', 'auto l = [] { int v __asm("w"); };',
                       '__extension__ asm("nop");', 'void f() { __extension__ asm("nop"); }',
                       'auto l = []<class T> { extern int g __asm("h"); };', 'struct S { static int x __asm("y"); };',
                       'int y = 1; return asm("nop");'):
            with self.subTest(source=source):
                self.assertTrue(CHECKER.inline_assembly_violations(source))
        for source in ('extern "C" { int fputs(const char *, void *) __asm("_" "fputs"); }',
                       'namespace a { inline namespace b { long tz __asm("_timezone"); } }',
                       'int a[2] __asm("b");', 'int x __asm("y") = 1;'):
            with self.subTest(source=source):
                self.assertEqual(CHECKER.inline_assembly_violations(source), [])

    def test_opaque_preprocessor_input_fails(self) -> None:
        for source in ('#pragma GCC pch_preprocess "probe.h.gch"\nint a;', '#pragma clang module import x\nint a;',
                       '#include "x.h"\nint a;', '# include_next <x>\n', '#embed "blob"\n', "}"):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    CHECKER.inline_assembly_violations(source)
        self.assertEqual(CHECKER.inline_assembly_violations('# 1 "x.h" 1\n#pragma GCC diagnostic push\nint a;'), [])

    def test_ordinary_asm_and_labels_are_accepted_and_malformed_asm_fails(self) -> None:
        for source in ('int f() __asm__("g");', 'void h(float v) { asm volatile("" : "+w"(v)); }',
                       '# 1 "asm/unistd.h"\nint x;', 'const char* s = "asm(\\".incbin\\")";',
                       'FILE *fopen(const char *, const char *) __asm("_" "fopen" );',
                       'char *rp(const char *) __asm("_" "realpath" "$DARWIN_EXTSN");', "int digits = 1'000'000;"):
            with self.subTest(source=source):
                self.assertEqual(CHECKER.inline_assembly_violations(source), [])
        for source in ("int asm;", 'asm(".incbin', "asm(text);", 'asm volatile(x "y");'):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    CHECKER.inline_assembly_violations(source)


class RelocationTest(unittest.TestCase):
    """Printed relocations must equal the object's relocation tables exactly, in both formats."""

    CASES = {False: ("R_AARCH64_ADD_ABS_LO12_NC", "R_AARCH64_CALL26"),
             True: ("ARM64_RELOC_PAGEOFF12", "ARM64_RELOC_BRANCH26")}

    def body(self, macho: bool) -> list[tuple[str, str]]:
        lo12 = self.CASES[macho][0]
        return [("f", f"\tstp d8, d9, [sp, #-32]!\n\tadd x9, sp, #0 !{lo12}:{abi('table', macho)}\n"
                      "\tstr q0, [x9, #8]\n\tldp d8, d9, [sp], #32\n\tret\n")]

    def test_object_relocations_are_authoritative(self) -> None:
        for macho in (False, True):
            functions = self.body(macho)
            with self.subTest(macho=macho, case="faithful"):
                bodies = CHECKER.normalize_disassembly(disassembly(functions, macho), image(functions, macho))
                errors = CHECKER.analyse_function("f", bodies, CHECKER.GENERIC, set())[1]
                self.assertTrue(any("relocated arithmetic" in error for error in errors), errors)
            printed = disassembly(functions, macho)
            line = next(line for line in printed.splitlines() if self.CASES[macho][0] in line)
            for case, text, obj in (("deleted printed line", printed.replace(line + "\n", ""), image(functions, macho)),
                                    ("wrong target", printed.replace(line, line.replace("table", "other")),
                                     image(functions, macho)),
                                    ("wrong type", printed.replace(line, line.replace(self.CASES[macho][0],
                                                                                      self.CASES[macho][1])),
                                     image(functions, macho)),
                                    ("missing in object", printed, image(functions, macho, relocations=False))):
                with self.subTest(macho=macho, case=case):
                    with self.assertRaisesRegex(ValueError, "relocation"):
                        CHECKER.normalize_disassembly(text, obj)

    def test_invented_relocations_fail_even_in_relocation_free_sections(self) -> None:
        for macho in (False, True):
            functions = [("f", "\tadd x9, x1, #0\n\tret\n")]
            printed = disassembly(functions, macho).replace(
                "ret\n", f"ret\n\t\t0000000000000004:  {self.CASES[macho][1]}\t{abi('abort', macho)}\n")
            with self.subTest(macho=macho):
                with self.assertRaisesRegex(ValueError, "does not match the object"):
                    CHECKER.normalize_disassembly(printed, image(functions, macho))

    def test_printed_branch_target_must_be_the_objects(self) -> None:
        for macho in (False, True):
            functions = [("f", f"\tbl {abi('helper', macho)}\n\tret\n")]
            printed = disassembly(functions, macho)
            forged = printed.replace(f"\t{abi('helper', macho)}\n", f"\t{abi('abort', macho)}\n")
            with self.subTest(macho=macho):
                with self.assertRaisesRegex(ValueError, "does not match the object"):
                    CHECKER.normalize_disassembly(forged, image(functions, macho))

    def test_relocations_on_padding_fail(self) -> None:
        functions = [("f", "\tret\n\tnop !R_AARCH64_CALL26:abort\n")]
        with self.assertRaisesRegex(ValueError, "outside every function extent"):
            CHECKER.normalize_disassembly(disassembly(functions), image(functions, sizes={"f": 4}))
        plain = [("f", "\tret\n\tnop\n")]
        self.assertIn("f", CHECKER.normalize_disassembly(disassembly(plain), image(plain, sizes={"f": 4})))


class NativeRecordTest(unittest.TestCase):
    """Relocation and object-type records are admitted by their native fields, not by their rendering."""

    def relocated(self, macho: bool) -> tuple[list[tuple[str, str]], bytes]:
        functions = [("f", f"\tbl {abi('helper', macho)}\n\tret\n")]
        return functions, witness_object(functions, macho=macho)

    def test_macho_relocation_field_combinations(self) -> None:
        functions, data = self.relocated(True)
        reloff = int.from_bytes(data[32 + 72 + 56:32 + 72 + 60], "little")
        info = int.from_bytes(data[reloff + 4:reloff + 8], "little")
        self.assertIn("f", CHECKER.normalize_disassembly(disassembly(functions, True), CHECKER.read_object(data)))
        for label, value in (("pcrel cleared", info & ~(1 << 24)), ("length 8 bytes", info | (3 << 25)),
                             ("length 1 byte", info & ~(3 << 25)), ("UNSIGNED type", info & 0x0FFFFFFF),
                             ("ADDEND type", (info & 0x0FFFFFFF) | (10 << 28)), ("not extern", info & ~(1 << 27))):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "relocation"):
                    CHECKER.read_object(patched(data, reloff + 4, value, 4))
        for label, address in (("unaligned", 2), ("past the section", 64)):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "not an instruction"):
                    CHECKER.read_object(patched(data, reloff, address, 4))

    def test_elf_relocation_records(self) -> None:
        functions, data = self.relocated(False)
        header = elf_header_offset(data, ".rela.text.f")
        size, offset = (int.from_bytes(data[header + 32:header + 40], "little"),
                        int.from_bytes(data[header + 24:header + 32], "little"))
        CHECKER.read_object(data)
        for label, mutated in (("remainder", patched(data, header + 32, size + 8, 8)),
                               ("truncated", patched(data, header + 24, len(data) - 8, 8)),
                               ("invalid link", patched(data, header + 40, 999, 4)),
                               ("invalid info", patched(data, header + 44, 999, 4)),
                               ("REL table", patched(data, header + 4, 9, 4)),
                               ("data type in code", patched(data, offset + 8, (int.from_bytes(
                                   data[offset + 8:offset + 16], "little") & ~0xFFFFFFFF) | 257, 8)),
                               ("offset past the section", patched(data, offset, 64, 8)),
                               ("unaligned offset", patched(data, offset, 2, 8))):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    CHECKER.read_object(mutated)

    def test_object_types(self) -> None:
        _, elf = self.relocated(False)
        _, macho = self.relocated(True)
        for label, mutated in (("ET_EXEC", patched(elf, 16, 2, 2)), ("ET_DYN", patched(elf, 16, 3, 2)),
                               ("e_version 0", patched(elf, 20, 0, 4)), ("MH_EXECUTE", patched(macho, 12, 2, 4)),
                               ("MH_DYLIB", patched(macho, 12, 6, 4))):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "relocatable"):
                    CHECKER.read_object(mutated)


class AlignmentTest(unittest.TestCase):
    """Function starts and extents must be instruction aligned before ownership is computed."""

    def test_partial_final_instruction_and_unaligned_start_fail(self) -> None:
        functions = [("f", "\tnop\n\tret\n")]
        for options in ({"sizes": {"f": 5}}, {"sizes": {"f": 7}}, {"sizes": {"f": 6}},
                        {"symbols": [("g", "function", "global", "f", 2)], "sizes": {"g": 4}}):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "unaligned"):
                    CHECKER.normalize_disassembly(disassembly(functions), image(functions, **options))
        self.assertIn("f", CHECKER.normalize_disassembly(disassembly(functions), image(functions, sizes={"f": 8})))


class InventoryTest(unittest.TestCase):
    """The expected programs derive from the admitted envelope; object, symbol table and disassembly agree."""

    def test_envelope_inventory_matches_the_written_out_keys(self) -> None:
        for macho in (False, True):
            counts = audit_witness(macho=macho)
            self.assertEqual(sorted(tuple(map(int, CHECKER.program_key(n).split(":"))) for n in counts), NEON_KEYS)
        self.assertEqual(len(NEON_KEYS), 64)

    def test_missing_duplicate_empty_and_unexpected_programs(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            audit_witness(keys=NEON_KEYS[1:])
        with self.assertRaisesRegex(ValueError, "ambiguous executable section"):
            audit_witness(keys=NEON_KEYS + NEON_KEYS[:1])
        with self.assertRaisesRegex(ValueError, "twice"):
            audit_witness(keys=NEON_KEYS + NEON_KEYS[:1], macho=True)
        with self.assertRaisesRegex(ValueError, "records no size"):
            audit_witness({NEON_KEYS[3]: ""})
        with self.assertRaisesRegex(ValueError, "outside the NEON inventory"):
            audit_witness(extra=[(name(1, 1, 0, 1, 0, 1), "\tstr q0, [sp, #-16]!\n\tret\n")])
        with self.assertRaisesRegex(ValueError, "unrecognized tile program"):
            audit_witness(extra=[(name(2, 0, 0, 1, 0, 1).replace("3runI", "3runIX"), "\tret\n")])

    def test_object_function_symbols_must_be_disassembled(self) -> None:
        functions = programs()
        with self.assertRaisesRegex(ValueError, "cover"):
            CHECKER.audit(witness_object(functions), disassembly(functions[1:]), "NEON")
        with self.assertRaisesRegex(ValueError, "does not define"):
            CHECKER.audit(witness_object(functions[1:]), disassembly(functions), "NEON")

    def test_embedded_inventory_must_agree_with_the_envelope(self) -> None:
        with self.assertRaisesRegex(ValueError, "inventory"):
            audit_witness(inventory="2:ASIMD,3:ASIMD,4:ASIMD")
        with self.assertRaisesRegex(ValueError, "inventory"):
            audit_witness(mode="NEON_DOTPROD")
        with self.assertRaisesRegex(ValueError, "envelope"):
            audit_witness(mode="SVE")
        with self.assertRaisesRegex(ValueError, "inventory"):
            audit_witness(inventory=None)

    def test_structural_failure_is_not_paid_from_a_budget(self) -> None:
        key = (2, 0, 0, 1, 0, 1)
        with self.assertRaisesRegex(ValueError, "structurally"):
            audit_witness({key: "\tbl helper\n\tret"}, extra=[("helper", "\tret\n")])
        with self.assertRaisesRegex(ValueError, "structurally"):
            audit_witness({key: "\t.word 0x3d8003e0\n\tret"})
        counts = audit_witness({key: "\tstr q0, [sp, #-16]!\n\tadd sp, sp, #16\n\tret"})
        self.assertEqual(counts[name(*key)], 1)


class PolicyTest(unittest.TestCase):
    """Rows=1 must not spill; grouped programs ratchet against the identity/envelope budget."""

    def counts(self, grouped: int, single: int) -> dict[str, int]:
        return {name(*key): (single if key[0] == 1 else grouped) for key in NEON_KEYS}

    def test_single_row_spill_is_always_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "single-row"):
            CHECKER.require_registers(self.counts(0, 1), "GNU-14", "NEON", budget())

    def test_grouped_budget_is_a_ratchet(self) -> None:
        CHECKER.require_registers(self.counts(4, 0), "GNU-14", "NEON", budget())
        with self.assertRaisesRegex(ValueError, "budget"):
            CHECKER.require_registers(self.counts(5, 0), "GNU-14", "NEON", budget())

    def test_unknown_identity_envelope_missing_and_stale_budget_fail_closed(self) -> None:
        for identity, mode in (("GNU-15", "NEON"), ("GNU-14", "NEON_I8MM")):
            with self.assertRaisesRegex(ValueError, "no qualified"):
                CHECKER.require_registers(self.counts(0, 0), identity, mode, budget())
        missing = budget()
        missing["GNU-14"]["NEON"].pop("2:0:0:1:0:1")
        with self.assertRaisesRegex(ValueError, "no budget"):
            CHECKER.require_registers(self.counts(0, 0), "GNU-14", "NEON", missing)
        stale = budget()
        stale["GNU-14"]["NEON"]["3:1:0:1:0:0"] = 9
        with self.assertRaisesRegex(ValueError, "did not emit"):
            CHECKER.require_registers(self.counts(0, 0), "GNU-14", "NEON", stale)

    def test_committed_budget_is_well_formed(self) -> None:
        committed = json.loads(CHECKER.BUDGET_PATH.read_text())
        self.assertEqual(sorted(key for key in committed if not key.startswith("_")), ["AppleClang-21", "GNU-14"])
        for identity, envelopes in committed.items():
            if identity.startswith("_"):
                continue
            self.assertEqual(sorted(envelopes), ["NEON", "NEON_DOTPROD", "NEON_I8MM"])
            for mode, limits in envelopes.items():
                self.assertEqual(len(limits), {"NEON": 48, "NEON_DOTPROD": 96, "NEON_I8MM": 144}[mode])
                self.assertTrue(all(not key.startswith("1:") and value >= 0 for key, value in limits.items()))

    def test_compiler_identity_is_derived_from_the_compiler(self) -> None:
        self.assertEqual(CHECKER.compiler_identity("Apple clang version 21.0.0 (clang-2100.3.34.2)\nTarget: arm64"),
                         "AppleClang-21")
        self.assertEqual(CHECKER.compiler_identity("g++-14 (Ubuntu 14.2.0-4ubuntu2~24.04.1) 14.2.0\n"
                                                   "Copyright (C) 2024 Free Software Foundation, Inc.\n"), "GNU-14")
        self.assertEqual(CHECKER.compiler_identity("Ubuntu clang version 19.1.1 (1ubuntu1)\n"), "Clang-19")
        with self.assertRaises(ValueError):
            CHECKER.compiler_identity("icpx (ICX) 2025.0\n")


FAKE_COMPILER = """#!{python} -SE
import os, pathlib, shutil, sys
here = pathlib.Path(__file__).parent
args = sys.argv[1:]
if args == ["--version"]:
    sys.stdout.write((here / "version.txt").read_text()); sys.exit(0)
def value(flag):
    return args[args.index(flag) + 1]
def setting(name, default):
    path = here / name
    return path.read_text() if path.exists() else default
def event(phase):
    path = here / f"event-{{phase}}.txt"
    if not path.exists():
        return
    action, target, value = path.read_text().split(None, 2)
    path.unlink()
    if action == "retarget":
        os.unlink(target); os.symlink(value.strip(), target)
    else:
        pathlib.Path(target).write_text(value)
if args[-1] == "-print-search-dirs":
    if (here / "fail-search.txt").exists():
        sys.stderr.write("driver error\\n"); sys.exit(1)
    sys.stdout.write("install: /x/\\nprograms: =" + setting("progdirs.txt", str(here) + "/") + "\\n"); sys.exit(0)
if args[-1].startswith("-print-prog-name="):
    name = args[-1].split("=", 1)[1]
    sys.stdout.write(setting(f"progname-{{name}}.txt", str(here / name)) + "\\n"); sys.exit(0)
if "-###" in args:
    event("plan")
    source = args[args.index("-c") + 1] if "-c" in args else ""
    # Like GCC, every expansion names a fresh temporary assembly file (cc + six alphanumerics + .s).
    counter = here / "expansions.txt"
    count = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(count))
    temporary = f"/tmp/cc{{count:06d}}.s"
    default = (f" {{here}}/cc1plus -quiet -MD {{value('-MF')}} -MF {{value('-MF')}} -MT probe -H {{source}} "
               f"-o {{temporary}}\\n as -EL -march=armv8-a -o {{value('-o')}} {{temporary}}\\n")
    jobs = setting("jobs.txt", default).format(here=here, source=source, obj=value("-o"), deps=value("-MF"))
    sys.stderr.write("Using built-in specs.\\nCOMPILER_PATH=" + str(here) + "\\n" + jobs)
    sys.exit(0)
(here / "invocations.txt").open("a").write(" ".join(args) + "\\n")
if "-E" in args:
    event("preprocess")
    unit = here / "preprocessed.txt"
    pathlib.Path(value("-o")).write_text(unit.read_text() if unit.exists() else "int probe;\\n"); sys.exit(0)
compiling = "-c" in args
# Prerequisites are space-separated exactly as written (no Unicode splitting).
deps = (here / ("deps.c.txt" if compiling and (here / "deps.c.txt").exists() else "deps.txt")).read_text().strip().split(" ")
if compiling:
    event("compile")
    mutate = here / "mutate.txt"
    if mutate.exists():
        target = pathlib.Path(mutate.read_text().strip())
        target.write_text(target.read_text() + "// changed during the build\\n")
    shutil.copy(here / "witness.o", value("-o"))
    if (here / "headers.txt").exists():
        sys.stderr.write((here / "headers.txt").read_text())
pathlib.Path(value("-MF")).write_text(value("-MT") + ": " + " ".join(deps) + "\\n")
"""
FAKE_DISASSEMBLER = """#!{python} -SE
import pathlib, sys
here = pathlib.Path(__file__).parent
args = sys.argv[1:]
if args == ["--version"]:
    sys.stdout.write((here / "objdump-version.txt").read_text()); sys.exit(0)
sys.stdout.write((here / "disassembly.txt").read_text().replace("probe.o", args[-1]))
"""


class ProofBindingTest(unittest.TestCase):
    """build()/verify() bind the compilation's closure, identity, command, object, disassembler and budget."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = root = Path(self.temporary.name)
        self.tool = root / "tool"
        self.tool.mkdir()
        self.compiler = self.tool / "c++"
        self.compiler.write_text(FAKE_COMPILER.format(python=sys.executable))
        self.compiler.chmod(self.compiler.stat().st_mode | stat.S_IEXEC)
        (self.tool / "version.txt").write_text("g++-14 (Fixture 14.2.0) 14.2.0\nCopyright Free Software Foundation\n")
        self.disassembler = self.tool / "objdump"
        self.disassembler.write_text(FAKE_DISASSEMBLER.format(python=sys.executable))
        self.disassembler.chmod(self.disassembler.stat().st_mode | stat.S_IEXEC)
        (self.tool / "objdump-version.txt").write_text("GNU objdump (Fixture) 2.42\n")
        for program in ("cc1plus", "as"):
            (self.tool / program).write_text("#!/bin/sh\n")
            (self.tool / program).chmod(0o755)
        self.witness(programs())
        self.source, self.header, self.real = root / "probe.cpp", root / "view.h", root / "real" / "view.h"
        self.real.parent.mkdir()
        self.source.write_text('#include "view.h"\n')
        self.real.write_text("// accessor\n")
        self.header.symlink_to(self.real)
        (self.tool / "deps.txt").write_text("probe.cpp view.h")
        database = root / "compile_commands.json"
        database.write_text(json.dumps([{"directory": str(root), "file": str(self.source),
                                         "arguments": [str(self.compiler), "-O2", "-c", str(self.source), "-o", "x.o"]}]))
        self.args = argparse.Namespace(source=str(self.source), proof=str(root / "out" / "proof.json"), signature="sig",
                                       compiler=str(self.compiler), compile_database=str(database),
                                       target_source=str(self.source), identity="GNU-14", mode="NEON",
                                       depfile_target="out/proof.json", disassembler=str(self.disassembler))
        self.original_budget = CHECKER.BUDGET_PATH
        CHECKER.BUDGET_PATH = root / "budget.json"
        CHECKER.BUDGET_PATH.write_text(json.dumps(budget()))

    def tearDown(self) -> None:
        CHECKER.BUDGET_PATH = self.original_budget
        self.temporary.cleanup()

    def witness(self, functions: list[tuple[str, str]]) -> None:
        (self.tool / "witness.o").write_bytes(witness_object(functions))
        (self.tool / "disassembly.txt").write_text(disassembly(functions))

    def proof(self) -> dict:
        return json.loads(Path(self.args.proof).read_text())

    def test_build_binds_the_compilation_closure_and_verify_detects_changes(self) -> None:
        CHECKER.build(self.args)
        CHECKER.verify(self.args)
        proof = self.proof()
        self.assertEqual(proof["identity"], "GNU-14")
        self.assertIn("-MD", proof["executed_command"])
        self.assertEqual(sorted(proof["inputs"]), sorted([str(self.source), str(self.header)]))
        self.assertIn(str(self.real), [target for _, target in proof["inputs"][str(self.header)]["links"]])
        depfile = Path(self.args.proof).with_suffix(".d").read_text()
        self.assertTrue(depfile.startswith("out/proof.json: ") and os.path.realpath(self.real) in depfile)
        self.real.write_text("// accessor changed\n")
        with self.assertRaisesRegex(ValueError, "stale tile register input"):
            CHECKER.verify(self.args)
        self.real.write_text("// accessor\n")
        CHECKER.verify(self.args)
        twin = self.root / "real" / "twin.h"
        twin.write_text("// accessor\n")
        self.header.unlink()
        self.header.symlink_to(twin)
        with self.assertRaisesRegex(ValueError, "symlink"):
            CHECKER.verify(self.args)

    def test_symlink_then_parent_component_names_the_filesystem_target(self) -> None:
        link_root, real_root = self.root / "p", self.root / "q"
        (real_root / "sub").mkdir(parents=True)
        link_root.mkdir()
        (link_root / "link").symlink_to(real_root / "sub")
        real, decoy = real_root / "h.h", link_root / "h.h"
        real.write_text("// real\n")
        decoy.write_text("// real\n")
        (self.tool / "deps.txt").write_text("probe.cpp view.h p/link/../h.h")
        CHECKER.build(self.args)
        identity = self.proof()["inputs"][str(self.root / "p" / "link" / ".." / "h.h")]
        self.assertEqual(identity["resolved"], os.path.realpath(real))
        self.assertIn(os.path.realpath(real), Path(self.args.proof).with_suffix(".d").read_text())
        decoy.write_text("// decoy changed\n")
        CHECKER.verify(self.args)
        real.write_text("// real changed\n")
        with self.assertRaisesRegex(ValueError, "stale tile register input"):
            CHECKER.verify(self.args)

    def test_probe_source_spelling_through_a_symlink_is_preserved(self) -> None:
        (self.root / "q" / "sub").mkdir(parents=True)
        (self.root / "p").mkdir()
        (self.root / "p" / "link").symlink_to(self.root / "q" / "sub")
        (self.root / "q" / "probe.cpp").write_text('#include "view.h"\n')
        (self.root / "p" / "probe.cpp").write_text("// decoy\n")
        self.args.source = str(self.root / "p" / "link" / ".." / "probe.cpp")
        (self.tool / "deps.txt").write_text("p/link/../probe.cpp view.h")
        self.args.target_source = str(self.source)
        CHECKER.build(self.args)
        CHECKER.verify(self.args)
        (self.root / "p" / "probe.cpp").write_text("// decoy changed\n")
        CHECKER.verify(self.args)
        (self.root / "q" / "probe.cpp").write_text("// probe changed\n")
        with self.assertRaisesRegex(ValueError, "stale tile register input"):
            CHECKER.verify(self.args)

    def test_relative_probe_paths_are_rejected_before_compiling(self) -> None:
        for source in ("probe.cpp", "p/link/../probe.cpp"):
            with self.subTest(source=source):
                self.args.source = source
                with self.assertRaisesRegex(ValueError, "absolute"):
                    CHECKER.build(self.args)
                self.assertFalse((self.tool / "invocations.txt").exists())
                with self.assertRaisesRegex(ValueError, "absolute"):
                    CHECKER.verify(self.args)

    def test_object_and_disassembler_are_bound(self) -> None:
        CHECKER.build(self.args)
        CHECKER.verify(self.args)
        proof, out = self.proof(), Path(self.args.proof)
        self.assertIn("-c", proof["executed_command"])
        self.assertEqual(proof["disassembly_command"][-1], str(out.with_suffix(".o")))
        original = out.with_suffix(".o").read_bytes()
        out.with_suffix(".o").write_bytes(original + b"\0")
        with self.assertRaisesRegex(ValueError, "object"):
            CHECKER.verify(self.args)
        out.with_suffix(".o").write_bytes(original)
        CHECKER.verify(self.args)
        retained = out.with_suffix(".dis").read_text()
        out.with_suffix(".dis").write_text(retained.replace("sdot", "sdot "))
        with self.assertRaisesRegex(ValueError, "retained disassembly"):
            CHECKER.verify(self.args)
        out.with_suffix(".dis").write_text(retained)
        (self.tool / "disassembly.txt").write_text((self.tool / "disassembly.txt").read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "retained disassembly"):
            CHECKER.verify(self.args)
        (self.tool / "disassembly.txt").write_text(disassembly(programs()))
        CHECKER.verify(self.args)
        self.disassembler.write_text(self.disassembler.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "disassembler"):
            CHECKER.verify(self.args)
        out.with_suffix(".o").unlink()
        with self.assertRaisesRegex(ValueError, "object"):
            CHECKER.verify(self.args)

    def test_composed_assembler_constructions_are_rejected_before_compiling(self) -> None:
        for unit in ('constexpr auto marker = 1\'000; void f() { asm(".include \\"extra.s\\""); }\n',
                     'void f() { asm(R"(.macro external suffix\n.inc\\suffix "extra.s"\n.endm\nexternal lude)"); }\n'):
            with self.subTest(unit=unit[:40]):
                (self.tool / "preprocessed.txt").write_text(unit)
                with self.assertRaisesRegex(ValueError, "inline assembly"):
                    CHECKER.build(self.args)
                self.assertFalse(Path(self.args.proof).exists())
                self.assertNotIn(" -c ", (self.tool / "invocations.txt").read_text())

    def test_nested_asm_build_publishes_no_receipt(self) -> None:
        (self.tool / "preprocessed.txt").write_text(
            'void f() { asm volatile("" : : "r"([] { asm volatile(".include \\"extra.s\\""); return 0; }())); }\n')
        with self.assertRaisesRegex(ValueError, "inline assembly"):
            CHECKER.build(self.args)
        self.assertFalse(Path(self.args.proof).exists())
        self.assertNotIn(" -c ", (self.tool / "invocations.txt").read_text() if
                         (self.tool / "invocations.txt").exists() else "")

    def test_opaque_routes_are_rejected_before_or_by_the_compile(self) -> None:
        here = self.tool
        cc1 = f" {here}/cc1plus -quiet -MD {{deps}} -MF {{deps}} -MT probe -H {{source}} -o /tmp/ccTEST01.s"
        assembler = " as -EL -o {obj} /tmp/ccTEST01.s"
        cases = {
            "PCH marker in the translation unit": ("preprocessed.txt", '#pragma GCC pch_preprocess "p.h.gch"\nint a;\n'),
            "PCH used by the compile": ("headers.txt", ". /usr/include/stdio.h\n! /work/probe.h.gch\n"),
            "invalid PCH tried": ("headers.txt", "..x /work/x.h.gch\n"),
            "specs file read": ("jobs.txt", "Reading specs from /tmp/extra.specs\n" + cc1 + "\n" + assembler + "\n"),
            "extra assembler input (renamed asm spec)": ("jobs.txt", cc1 + "\n" + assembler + " /abs/extra.s\n"),
            "unadmitted assembler option": ("jobs.txt", cc1 + "\n as --defsym x=1 -o {obj} /tmp/ccTEST01.s\n"),
            "PCH in the compiler job": ("jobs.txt", cc1.replace("-quiet", "-quiet -include-pch p.pch") + "\n" +
                                        assembler + "\n"),
            "profile in the compiler job": ("jobs.txt", cc1.replace("-quiet", "-quiet -fprofile-use=/p.gcda") + "\n" +
                                            assembler + "\n"),
            "third job": ("jobs.txt", cc1 + "\n" + assembler + "\n" + cc1 + "\n"),
            "failed search-directory query": ("fail-search.txt", "1"),
            "driver selects another assembler": ("progname-as.txt", "/usr/local/bin/as"),
        }
        for label, (name, content) in cases.items():
            with self.subTest(label=label):
                (self.tool / name).write_text(content)
                with self.assertRaises(ValueError):
                    CHECKER.build(self.args)
                self.assertFalse(Path(self.args.proof).exists())
                (self.tool / name).unlink()
        CHECKER.build(self.args)
        self.assertEqual(len(self.proof()["driver_plan"]["jobs"]), 2)

    def test_job_programs_are_bound_by_content_symlink_and_selection(self) -> None:
        (self.tool / "versions").mkdir()
        for version in ("A", "B"):
            (self.tool / "versions" / version).write_text(f"#!/bin/sh\n# assembler {version}\n")
            (self.tool / "versions" / version).chmod(0o755)
        (self.tool / "as").unlink()
        (self.tool / "as").symlink_to(self.tool / "versions" / "A")
        (self.tool / "first").mkdir()
        (self.tool / "progdirs.txt").write_text(f"{self.tool}/first/:{self.tool}/")
        CHECKER.build(self.args)
        CHECKER.verify(self.args)
        programs = self.proof()["programs"]
        self.assertEqual(programs["job1"]["examined"], [f"{self.tool}/first/"])
        self.assertEqual(programs["job1"]["resolved"], os.path.realpath(self.tool / "versions" / "A"))
        (self.tool / "as").unlink()
        (self.tool / "as").symlink_to(self.tool / "versions" / "B")
        with self.assertRaisesRegex(ValueError, "program"):
            CHECKER.verify(self.args)
        (self.tool / "as").unlink()
        (self.tool / "as").symlink_to(self.tool / "versions" / "A")
        CHECKER.verify(self.args)
        (self.tool / "first" / "as").write_text("#!/bin/sh\n# shadow\n")
        (self.tool / "first" / "as").chmod(0o755)
        with self.assertRaisesRegex(ValueError, "program"):
            CHECKER.verify(self.args)
        (self.tool / "first" / "as").unlink()
        CHECKER.verify(self.args)
        (self.tool / "versions" / "A").write_text("#!/bin/sh\n# replaced assembler\n")
        with self.assertRaisesRegex(ValueError, "program"):
            CHECKER.verify(self.args)

    def test_receipt_plan_and_program_coverage_are_required(self) -> None:
        CHECKER.build(self.args)
        path, original = Path(self.args.proof), self.proof()
        plan, programs = original["driver_plan"], original["programs"]
        mutations = {
            "no plan": {key: value for key, value in original.items() if key != "driver_plan"},
            "plan without jobs": dict(original, driver_plan={"program_dirs": [], "path": []}),
            "incompatible plan": dict(original, driver_plan=dict(plan, jobs=plan["jobs"][:1])),
            "foreign job argument": dict(original, driver_plan=dict(plan, jobs=[
                plan["jobs"][0] + ["-fplugin=/x.so"], plan["jobs"][1]])),
            "assembler identity removed": dict(original, programs={key: value for key, value in programs.items()
                                                                     if key != "job1"}),
            "foreign program entry": dict(original, programs=dict(programs, job9=programs["job0"])),
            "mismatched job program": dict(original, programs=dict(programs, job1=programs["job0"])),
            "no programs": {key: value for key, value in original.items() if key != "programs"},
        }
        for label, receipt in mutations.items():
            with self.subTest(label=label):
                path.write_text(json.dumps(receipt))
                with self.assertRaises(ValueError):
                    CHECKER.verify(self.args)
        path.write_text(json.dumps(original))
        CHECKER.verify(self.args)

    def driver_symlink(self) -> Path:
        """Point the compile database at tool/cxx-driver -> tool/c++-A (a copy of the stand-in compiler)."""
        for version in ("A", "B"):
            (self.tool / f"c++-{version}").write_text(self.compiler.read_text())
            (self.tool / f"c++-{version}").chmod(0o755)
        link = self.tool / "cxx-driver"
        link.symlink_to(self.tool / "c++-A")
        database = json.loads(Path(self.args.compile_database).read_text())
        database[0]["arguments"][0] = str(link)
        Path(self.args.compile_database).write_text(json.dumps(database))
        self.args.compiler = str(link)
        return link

    def test_search_lists_must_be_absolute(self) -> None:
        original = os.environ["PATH"]
        try:
            for value in (":/usr/bin:/bin", "/usr/bin:/bin:", "/usr/bin::/bin", "bin:/usr/bin", ""):
                with self.subTest(path=value):
                    os.environ["PATH"] = value
                    with self.assertRaisesRegex(ValueError, "empty or relative"):
                        CHECKER.build(self.args)
                    self.assertFalse(Path(self.args.proof).exists())
        finally:
            os.environ["PATH"] = original
        (self.tool / "progdirs.txt").write_text(f"relative/dir/:{self.tool}/")
        with self.assertRaisesRegex(ValueError, "empty or relative"):
            CHECKER.build(self.args)
        (self.tool / "progdirs.txt").write_text(f"{self.tool}/::{self.tool}/")
        with self.assertRaisesRegex(ValueError, "empty or relative"):
            CHECKER.build(self.args)

    def test_checker_working_directory_does_not_matter(self) -> None:
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        here = os.getcwd()
        try:
            os.chdir(elsewhere)
            CHECKER.build(self.args)
            os.chdir(self.root)
            CHECKER.verify(self.args)
        finally:
            os.chdir(here)

    def test_epoch_mutations_publish_no_receipt(self) -> None:
        link = self.driver_symlink()
        cases = {
            "driver retargeted during plan observation": ("plan", f"retarget {link} {self.tool / 'c++-B'}"),
            "driver retargeted during preprocessing": ("preprocess", f"retarget {link} {self.tool / 'c++-B'}"),
            "compiler proper replaced during preprocessing": ("preprocess", f"write {self.tool / 'cc1plus'} #!/bin/sh\n# B\n"),
            "assembler replaced during compile": ("compile", f"write {self.tool / 'as'} #!/bin/sh\n# B\n"),
            "driver retargeted during compile": ("compile", f"retarget {link} {self.tool / 'c++-B'}"),
        }
        for label, (phase, action) in cases.items():
            with self.subTest(label=label):
                (self.tool / f"event-{phase}.txt").write_text(action)
                with self.assertRaisesRegex(ValueError, "changed"):
                    CHECKER.build(self.args)
                self.assertFalse(Path(self.args.proof).exists())
                link.unlink()
                link.symlink_to(self.tool / "c++-A")
                for program in ("cc1plus", "as"):
                    (self.tool / program).write_text("#!/bin/sh\n")
        CHECKER.build(self.args)
        CHECKER.verify(self.args)

    def test_make_escaped_dependency_names_are_rejected_before_binding(self) -> None:
        real, decoy = self.root / "a$b", self.root / "a$$b"
        for directory in (real, decoy):
            directory.mkdir()
            (directory / "view.h").write_text("// same\n")
        for deps in ("probe.cpp a$$b/view.h", "probe.cpp a\\ b/view.h", "probe.cpp a\\#b.h", "probe.cpp 'q.h'"):
            with self.subTest(deps=deps):
                (self.tool / "deps.txt").write_text(deps)
                with self.assertRaisesRegex(ValueError, "unsupported escaped names"):
                    CHECKER.build(self.args)
                self.assertFalse(Path(self.args.proof).exists())

    def test_replanning_masks_only_the_shared_driver_temporary(self) -> None:
        CHECKER.build(self.args)  # Each expansion of the stand-in names a fresh temporary, as GCC does.
        CHECKER.verify(self.args)
        jobs = self.proof()["driver_plan"]["jobs"]
        self.assertRegex(jobs[0][jobs[0].index("-o") + 1], r"^/tmp/cc\d{6}\.s$")
        plan = {"jobs": jobs, "program_dirs": [], "path": []}

        def renamed(compiler: str, assembler: str, compiler_jobs=None) -> dict:
            first = [compiler if argument == jobs[1][-1] else argument for argument in (compiler_jobs or jobs[0])]
            return dict(plan, jobs=[first, jobs[1][:-1] + [assembler]])
        self.assertEqual(CHECKER.plan_shape(renamed("/tmp/ccAb12Cd.s", "/tmp/ccAb12Cd.s")), CHECKER.plan_shape(plan))
        self.assertNotEqual(CHECKER.plan_shape(renamed("/var/ccAAAAAA.s", "/var/ccAAAAAA.s")), CHECKER.plan_shape(plan))
        for label, mutated in (("mismatched names", renamed("/tmp/ccAAAAAA.s", "/tmp/ccBBBBBB.s")),
                               ("not a driver temporary", renamed("/tmp/probe.s", "/tmp/probe.s")),
                               ("seven characters", renamed("/tmp/ccAAAAAAA.s", "/tmp/ccAAAAAAA.s")),
                               ("relative temporary", renamed("ccAAAAAA.s", "ccAAAAAA.s")),
                               ("trailing newline", renamed("/tmp/ccAb12Cd.s\n", "/tmp/ccAb12Cd.s\n"))):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    CHECKER.plan_shape(mutated)
        for label, mutated in (("compiler option", [jobs[0][:1] + ["-g"] + jobs[0][1:], jobs[1]]),
                               ("assembler option", [jobs[0], jobs[1][:1] + ["--gdwarf-5"] + jobs[1][1:]]),
                               ("compiler program", [["/other/cc1plus"] + jobs[0][1:], jobs[1]])):
            with self.subTest(label=label):
                self.assertNotEqual(CHECKER.plan_shape(dict(plan, jobs=mutated)), CHECKER.plan_shape(plan))
        self.assertNotEqual(CHECKER.plan_shape(dict(plan, path=["/usr/bin"])), CHECKER.plan_shape(plan))
        receipt = self.proof()
        temporary = jobs[0][jobs[0].index("-o") + 1]
        newline = [[argument + "\n" if argument == temporary else argument for argument in job] for job in jobs]
        Path(self.args.proof).write_text(json.dumps(dict(receipt, driver_plan=dict(receipt["driver_plan"],
                                                                                   jobs=newline))))
        with self.assertRaisesRegex(ValueError, "temporary"):
            CHECKER.verify(self.args)
        Path(self.args.proof).write_text(json.dumps(receipt))
        CHECKER.verify(self.args)
        (self.tool / "event-compile.txt").write_text(
            f"write {self.tool / 'jobs.txt'} replanned\n {{here}}/cc1plus -quiet -g -MD {{deps}} -MF {{deps}} -MT probe -H "
            "{source} -o /tmp/ccXXXXXX.s\n as -EL -o {obj} /tmp/ccXXXXXX.s\n")
        with self.assertRaisesRegex(ValueError, "driver plan"):
            CHECKER.build(self.args)
        self.assertFalse(Path(self.args.proof).exists())

    def test_non_ascii_dependency_names_are_rejected_before_splitting(self) -> None:
        real = self.root / "a\u00a0b"
        real.mkdir()
        (real / "view.h").write_text("// real\n")
        (self.root / "a").write_text("// decoy\n")
        (self.root / "b").mkdir()
        (self.root / "b" / "view.h").write_text("// decoy\n")
        (self.tool / "deps.txt").write_text("probe.cpp a\u00a0b/view.h")
        with self.assertRaisesRegex(ValueError, "outside the plain depfile subset"):
            CHECKER.build(self.args)
        self.assertFalse(Path(self.args.proof).exists())
        depfile = self.root / "native.d"
        for line in ("probe: /w/a\u00a0b/view.h\n", "probe: /w/a\u3000b.h\n", "probe: /w/a.h\x0b/w/b.h\n",
                     "probe: /w/\u00e9.h\n"):
            with self.subTest(line=line):
                depfile.write_bytes(line.encode())
                with self.assertRaisesRegex(ValueError, "plain depfile subset"):
                    CHECKER.read_depfile(depfile, "/w")
        depfile.write_text("probe: /w/a.h\t/w/b.h \\\n /w/c.h\n")
        self.assertEqual(CHECKER.read_depfile(depfile, "/w"), ["/w/a.h", "/w/b.h", "/w/c.h"])

    def test_retarget_during_the_final_plan_observation_publishes_no_receipt(self) -> None:
        link = self.driver_symlink()
        # The compile triggers a retarget armed for the next -### query: the final re-observation only.
        (self.tool / "event-compile.txt").write_text(
            f"write {self.tool / 'event-plan.txt'} retarget {link} {self.tool / 'c++-B'}")
        with self.assertRaisesRegex(ValueError, "changed"):
            CHECKER.build(self.args)
        self.assertFalse(Path(self.args.proof).exists())

    def test_relative_driver_is_rejected_directly(self) -> None:
        with self.assertRaisesRegex(ValueError, "relatively"):
            CHECKER.driver_identity(["c++", "-c"])
        self.assertEqual(CHECKER.driver_identity([str(self.compiler)])["selected"], str(self.compiler))

    def test_driver_role_comes_from_the_target_command(self) -> None:
        CHECKER.build(self.args)
        path, original = Path(self.args.proof), self.proof()
        programs = original["programs"]
        foreign = dict(programs["driver"], program="/usr/bin/true", selected="/usr/bin/true")
        for label, driver in (("job copied as driver", programs["job0"]), ("foreign driver", foreign)):
            with self.subTest(label=label):
                path.write_text(json.dumps(dict(original, programs=dict(programs, driver=driver))))
                with self.assertRaisesRegex(ValueError, "program"):
                    CHECKER.verify(self.args)
        path.write_text(json.dumps(original))
        CHECKER.verify(self.args)
        database = json.loads(Path(self.args.compile_database).read_text())
        database[0]["arguments"][0] = "c++"
        Path(self.args.compile_database).write_text(json.dumps(database))
        with self.assertRaises(ValueError):
            CHECKER.build(self.args)

    def test_typed_job_admission(self) -> None:
        paths = CHECKER.JobPaths("/w/probe.cpp", "/w/out/proof.o", "/w/out/proof.deps", "/w/build",
                                 frozenset({"/w/probe.cpp", "/sdk/SDKSettings.json"}))
        clang = ["/xc/usr/bin/clang", "-cc1", "-triple", "arm64-apple-macosx27.0.0", "-O3", "-emit-obj",
                 "-main-file-name", "probe.cpp", "-target-feature", "+i8mm", "-mllvm", "-extra-vectorizer-passes",
                 "-fdepfile-entry=/sdk/SDKSettings.json", "-I", "/w/src", "-I/usr/local/include",
                 "-fdebug-compilation-dir=/w/build", "-Wno-error=allocator-wrappers", "-D", "NDEBUG", "-H",
                 "-dependency-file", "/w/out/proof.deps", "-MT", "probe", "-o", "/w/out/proof.o", "-x", "c++",
                 "/w/probe.cpp"]
        gnu = [["/usr/libexec/gcc/cc1plus", "-quiet", "-I", "/w/src", "-MD", "/w/out/proof.d", "-MF",
                "/w/out/proof.deps", "-MT", "probe", "-D_GNU_SOURCE", "-D", "X=1", "/w/probe.cpp", "-march=armv8-a+i8mm",
                "-g", "-O3", "-dumpdir", "/tmp/", "-dumpbase", "proof.cpp", "-o", "/tmp/ccAb12Cd.s", "-H"],
               ["as", "-I", "/w/src", "--gdwarf-5", "-EL", "-march=armv8-a", "-mabi=lp64", "-o", "/w/out/proof.o",
                "/tmp/ccAb12Cd.s"]]
        CHECKER.validate_plan([clang], "AppleClang-21", paths)
        CHECKER.validate_plan(gnu, "GNU-14", paths)
        clang_mutations = {
            "remap-file": ["-remap-file", "/w/src/view.h;/outside/actual.h"],
            "vfs overlay": ["-ivfsoverlay", "/x/overlay.yaml"], "vfsoverlay": ["-vfsoverlay", "/x/o.yaml"],
            "profile use path": ["-fprofile-instrument-use-path=/x/default.profdata"],
            "sample profile": ["-fprofile-sample-use=/x/p.prof"], "include pch": ["-include-pch", "/x/p.pch"],
            "plugin": ["-load", "/x/plugin.so"], "other mllvm option": ["-mllvm", "-some-file=/x"],
            "depfile entry outside the closure": ["-fdepfile-entry=/x/other.json"],
            "relative include root": ["-I", "include"], "other compilation directory": ["-fdebug-compilation-dir=/x"],
            "second positional input": ["/x/other.cpp"], "ignorelist": ["-fsanitize-ignorelist=/x/list.txt"],
            "assembler-style warning": ["-Wa,-I/x"], "embed directory": ["--embed-dir=/x"],
        }
        for label, extra in clang_mutations.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    CHECKER.validate_plan([clang[:-1] + extra + clang[-1:]], "AppleClang-21", paths)
        for label, plan in {"other output": [clang[:clang.index("-o") + 1] + ["/x.o"] + clang[clang.index("-o") + 2:]],
                            "not integrated": [clang, ["/usr/bin/as", "-o", "/w/out/proof.o"]],
                            "relative program": [["clang"] + clang[1:]]}.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    CHECKER.validate_plan(plan, "AppleClang-21", paths)
        gnu_mutations = {
            "compiler profile": (0, ["-fprofile-use=/x/p.gcda"]), "compiler plugin": (0, ["-fplugin=/x/p.so"]),
            "compiler include": (0, ["-include", "/x/h.h"]), "assembler extra input": (1, ["/abs/extra.s"]),
            "assembler defsym": (1, ["--defsym", "x=1"]), "assembler warning flag": (1, ["-Wno-foo"]),
            "second assembly output": (0, ["-o", "/tmp/other.s"]),
        }
        for label, (index, extra) in gnu_mutations.items():
            with self.subTest(label=label):
                mutated = [list(job) for job in gnu]
                mutated[index] = mutated[index] + extra
                with self.assertRaises(ValueError):
                    CHECKER.validate_plan(mutated, "GNU-14", paths)

    def test_inline_assembly_file_input_is_rejected_before_compiling(self) -> None:
        (self.tool / "preprocessed.txt").write_text('void f() { asm(".incbin \\"blob.bin\\""); }\n')
        with self.assertRaisesRegex(ValueError, "inline assembly the dependency closure cannot bind"):
            CHECKER.build(self.args)
        self.assertFalse(Path(self.args.proof).exists())
        self.assertNotIn(" -c ", (self.tool / "invocations.txt").read_text())
        (self.tool / "preprocessed.txt").unlink()
        CHECKER.build(self.args)
        unit = Path(self.args.proof).with_suffix(".i")
        unit.write_text(unit.read_text() + 'asm(".include \\"other.s\\"");\n')
        with self.assertRaisesRegex(ValueError, "translation unit"):
            CHECKER.verify(self.args)

    def test_unsupported_disassembler_fails(self) -> None:
        (self.tool / "objdump-version.txt").write_text("Some Other Disassembler 1.0\n")
        with self.assertRaisesRegex(ValueError, "disassembler"):
            CHECKER.build(self.args)
        self.assertFalse(Path(self.args.proof).exists())

    def test_recorded_command_identity_budget_and_assembly_are_bound(self) -> None:
        CHECKER.build(self.args)
        path, original = Path(self.args.proof), self.proof()
        for field, value in (("executed_command", original["executed_command"][:-1] + ["other"]),
                             ("compiler_version", "g++-13 (Fixture 13.1.0) 13.1.0\nFree Software Foundation\n"),
                             ("identity", "GNU-15")):
            with self.subTest(field=field):
                path.write_text(json.dumps(dict(original, **{field: value})))
                with self.assertRaises(ValueError):
                    CHECKER.verify(self.args)
        path.write_text(json.dumps(original))
        CHECKER.BUDGET_PATH.write_text(json.dumps(budget(limit=9)))
        with self.assertRaisesRegex(ValueError, "stale tile register tool"):
            CHECKER.verify(self.args)
        CHECKER.BUDGET_PATH.write_text(json.dumps(budget()))
        CHECKER.verify(self.args)

    def test_build_rejects_foreign_identity_moving_inputs_and_closure_mismatch(self) -> None:
        (self.tool / "version.txt").write_text("g++-13 (Fixture 13.1.0) 13.1.0\nFree Software Foundation\n")
        with self.assertRaisesRegex(ValueError, "identity"):
            CHECKER.build(self.args)
        (self.tool / "version.txt").write_text("g++-14 (Fixture 14.2.0) 14.2.0\nFree Software Foundation\n")
        (self.tool / "mutate.txt").write_text(str(self.real))
        with self.assertRaisesRegex(ValueError, "changed while"):
            CHECKER.build(self.args)
        (self.tool / "mutate.txt").unlink()
        (self.tool / "deps.c.txt").write_text("probe.cpp")
        with self.assertRaisesRegex(ValueError, "differs"):
            CHECKER.build(self.args)
        self.assertFalse(Path(self.args.proof).exists())

    def test_failed_audit_publishes_no_receipt(self) -> None:
        CHECKER.build(self.args)
        self.witness(programs({(1, 0, 0, 0, 0, 0): "\tstr q0, [sp, #-16]!\n\tadd sp, sp, #16\n\tret"}))
        with self.assertRaisesRegex(ValueError, "single-row"):
            CHECKER.build(self.args)
        self.assertFalse(Path(self.args.proof).exists())


BINDING_SHARDS = 4


def shard_suite(shard: str) -> unittest.TestSuite:
    """'analysis' runs every compiler-free class; 'binding-K' runs every BINDING_SHARDS-th proof-binding test
    (each spawns stand-in tools), so every registered shard stays well inside the core-free test timeout."""
    loader = unittest.defaultTestLoader
    module = sys.modules[__name__]
    if shard == "analysis":
        return unittest.TestSuite(loader.loadTestsFromTestCase(case) for case in vars(module).values()
                                  if isinstance(case, type) and issubclass(case, unittest.TestCase)
                                  and case is not ProofBindingTest)
    index = int(shard.removeprefix("binding-"))
    if not 0 <= index < BINDING_SHARDS:
        raise SystemExit(f"unknown shard {shard}")
    names = sorted(loader.getTestCaseNames(ProofBindingTest))
    return unittest.TestSuite(ProofBindingTest(name) for position, name in enumerate(names)
                              if position % BINDING_SHARDS == index)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--shard":
        result = unittest.TextTestRunner(verbosity=1).run(shard_suite(sys.argv[2]))
        sys.exit(0 if result.wasSuccessful() and result.testsRun else 1)
    unittest.main()
