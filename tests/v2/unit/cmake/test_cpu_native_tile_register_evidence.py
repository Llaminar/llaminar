#!/usr/bin/env python3
"""Compiler-free adversarial tests of the native NEON tile register-allocation witness.

Fixtures are synthetic ELF64 AArch64 (ET_REL) and Mach-O arm64 (MH_OBJECT)
objects - sections, symbol tables with raw linker names and relocation tables,
with zero-filled code bytes - paired with an llvm-objdump-style disassembly
rendered from the same layout; they are not native compiler/disassembler pairs.
Native object re-audits are separate owner evidence. Analysis tests run the
production front end over the pairs. Covered: the control-flow/provenance
analysis, the ABI save proof, per-loop stack-traffic classes and loop naming, admitted forms, call-boundary rejection, the
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
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import signal
import stat
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

CHECKER_PATH = Path(__file__).resolve().parents[2] / "cmake" / "CPUNativeTileRegisterEvidence.py"
REPO = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location("cpu_native_tile_register_evidence", CHECKER_PATH)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)

# Tile::dispatch() programs per policy: (encoding, nibble map, with-minimum), written out independently.
DISPATCHED = [(1, 0, 0), (2, 0, 0), (4, 2, 0), (0, 0, 0), (0, 2, 0), (0, 1, 1), (0, 3, 0), (6, 1, 1), (7, 0, 0), (7, 1, 1),
              # runCoded<Codebook>: CompactMultiScale (3), map = source codebook 7 (Q5_K, minimum) / 11 (IQ3_S).
              (3, 7, 1), (3, 11, 0)]
NEON_KEYS = sorted((rows, 0, policy, encoding, nibble_map, minimum)
                   for rows in (1, 2, 3, 4) for policy in (0, 1) for encoding, nibble_map, minimum in DISPATCHED)
NEON_INVENTORY = "1:ASIMD,2:ASIMD,3:ASIMD,4:ASIMD"
FATAL = "_ZN9llaminar23cpu16native_quantised28CPUNativeQuantisedWeightView32rejectOutsideAdmittedPublicationEv"
CLEAN = "\tstp\tx29, x30, [sp, #-16]!\n\tmov\tx29, sp\n\tsdot\tv0.4s, v1.16b, v2.16b\n\tldp\tx29, x30, [sp], #16\n\tret"
# A GPU-aligned program carries one ordered add (the inline-asm site the assembly fixture marks).
ORDERED_CLEAN = CLEAN.replace("\tldp", "\tfadd\tv0.4s, v0.4s, v1.4s\n\tldp")
FATAL_BODY = "\tstp\tx29, x30, [sp, #-16]!\n\tmov\tx29, sp\n\tbl\tfwrite\n\tbl\tabort\n"
BRANCH = re.compile(r"^(?:b|bl|cbz|cbnz|tbz|tbnz|b\.\w+)$")


def name(rows: int, dot: int, policy: int, encoding: int, nibble_map: int, minimum: int, strips: int = 0) -> str:
    """Mangled tile run<..., Strips>() symbol (runCoded<..., Codebook, Strips>() for encoding 3); production
    instantiates Strips = 0 (the installed width)."""
    if encoding == 3:
        return (f"_ZNK9llaminar23cpu16native_quantised27CPUNativeNEONProjectionTileILm{rows}EE8runCodedINS1_11multi_scale"
                f"17NEONMultiScaleOpsILNS5_11NEONByteDotE{dot}EEELNS_28CPUProjectionNumericalPolicyE{policy}ELi"
                f"{nibble_map}ELi{strips}EEEvv")
    return (f"_ZNK9llaminar23cpu16native_quantised27CPUNativeNEONProjectionTileILm{rows}EE3runINS1_11multi_scale17"
            f"NEONMultiScaleOpsILNS5_11NEONByteDotE{dot}EEELNS_28CPUProjectionNumericalPolicyE{policy}ELNS1_26"
            f"CPUNativeQuantisedEncodingE{encoding}ELNS3_9NibbleMapE{nibble_map}ELb{minimum}ELi{strips}EEEvv")


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

    def __init__(self, functions: list[tuple[str, str]], macho: bool, shared: bool = False) -> None:
        self.macho, self.shared, self.entries, address = macho, shared, [], 0
        defined = {symbol for symbol, _ in functions}
        for symbol, body in functions:
            if not macho and not shared:
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
            section = "__TEXT,__text" if macho else (".text" if shared else f".text.{symbol}")
            self.entries.append({"name": symbol, "section": section, "start": start, "end": address,
                                 "labels": labels, "code": code, "relocations": relocations})
        self.imports = sorted({target for entry in self.entries for _, target, _ in entry["relocations"].values()
                               if target not in defined and not target.startswith(".")})


def disassembly(functions: list[tuple[str, str]], macho: bool = False, path: str = "probe.o",
                shared: bool = False) -> str:
    """Render functions as llvm-objdump -d -r output (raw names, printed relocation records); shared=True puts
    every ELF function into one .text section."""
    layout = Layout(functions, macho, shared)
    out = [f"{path}:\tfile format {'mach-o arm64' if macho else 'elf64-littleaarch64'}", ""]
    if macho:
        out.append("Disassembly of section __TEXT,__text:")
    printed = set()
    for entry in layout.entries:
        symbol, start, labels = entry["name"], entry["start"], entry["labels"]
        if not macho and entry["section"] not in printed:
            printed.add(entry["section"])
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
                   relocations: bool = True, shared: bool = False, visibility: dict[str, str] | None = None) -> bytes:
    """A real ELF64 AArch64 or Mach-O arm64 object consistent with Layout(functions).

    Defined functions, undefined imports and relocation tables come from the layout; symbols appends raw
    (name, kind, binding, owning function or None, value) records; sizes overrides ELF symbol sizes;
    relocations=False omits the relocation tables; bindings names the binding of an import or of a defined
    function (default global).
    """
    layout = Layout(functions or [], macho, shared)
    marker = b"" if inventory is None else b"llaminar-native-neon-tile-probe:" + inventory.encode() + b"\0"
    records = [(entry["name"], "function", (bindings or {}).get(entry["name"], "global"), entry["name"], entry["start"])
               for entry in layout.entries]
    records += [(name, "undefined", (bindings or {}).get(name, "global"), None, 0) for name in layout.imports]
    records += symbols or []
    if not relocations:
        for entry in layout.entries:
            entry["relocations"] = {}
    if macho:
        return _macho(layout, records, marker + extra)
    return _elf(layout, records, marker + extra, sizes or {}, visibility or {})


def _elf(layout: Layout, records: list[tuple], data: bytes, sizes: dict[str, int],
         visibility: dict[str, str] | None = None) -> bytes:
    # One section per function (duplicate names stay distinct sections), or one shared .text section.
    groups = ([[entry for entry in layout.entries if entry["section"] == section]
               for section in dict.fromkeys(entry["section"] for entry in layout.entries)]
              if layout.shared else [[entry] for entry in layout.entries])
    texts = list(range(len(groups)))
    members = dict(enumerate(groups))
    relocated = [group for group in texts if any(entry["relocations"] for entry in members[group])]
    names = [""] + [members[group][0]["section"] for group in texts] + [".rodata"] + \
        [".rela" + members[group][0]["section"] for group in relocated] + [".symtab", ".strtab", ".shstrtab"]
    section_index = {entry["name"]: group + 1 for group in texts for entry in members[group]}
    section_index[".rodata"] = len(texts) + 1
    symtab_index = len(names) - 3
    if any(target == ".rodata" for entry in layout.entries for _, target, _ in entry["relocations"].values()):
        records = [("", "section", "local", ".rodata", 0)] + records
    # Executable section symbols named as relocation targets (internal-linkage calls), owned by that section.
    targets = {target for entry in layout.entries for _, target, _ in entry["relocations"].values()}
    text_sections = [(members[group][0]["section"], members[group][0]["name"]) for group in texts
                     if members[group][0]["section"] in targets]
    records = [("", "section", "local", owner, 0) for _, owner in text_sections] + records
    shstr = b"\0" + b"".join(name.encode() + b"\0" for name in names[1:])
    strtab, symtab, symbol_index = bytearray(b"\0"), bytearray(24), {}
    extents = {entry["name"]: entry["end"] - entry["start"] for entry in layout.entries}
    for number, (name, kind, binding, owner, value) in enumerate(records, start=1):
        offset = len(strtab)
        strtab += name.encode() + b"\0"
        if kind == "section" and owner in dict((owner, section) for section, owner in text_sections):
            symbol_index[dict((owner, section) for section, owner in text_sections)[owner]] = number
        else:
            symbol_index.setdefault(name or owner, number)
        info = ({"local": 0, "global": 1, "weak": 2}[binding] << 4) | {
            "function": 2, "notype": 0, "object": 1, "undefined": 0, "absolute": 0, "section": 3}[kind]
        section = section_index[owner] if owner else (0xFFF1 if kind == "absolute" else 0)
        size = sizes.get(name, extents.get(name, 0)) if kind == "function" else 0
        other = {"default": 0, "internal": 1, "hidden": 2, "protected": 3}[(visibility or {}).get(name, "default")]
        symtab += offset.to_bytes(4, "little") + bytes([info, other]) + section.to_bytes(2, "little") + \
            value.to_bytes(8, "little") + size.to_bytes(8, "little")
    tables = []
    for section in relocated:
        table = bytearray()
        merged = {address: record for entry in members[section] for address, record in entry["relocations"].items()}
        for address, (kind, target, addend) in sorted(merged.items()):
            table += address.to_bytes(8, "little") + \
                ((symbol_index[target] << 32) | ELF_TYPES[kind]).to_bytes(8, "little") + \
                addend.to_bytes(8, "little", signed=True)
        tables.append(bytes(table))
    contents = [b""] + [bytes(max(entry["end"] for entry in members[section])) for section in texts] + [data] + \
        tables + [bytes(symtab), bytes(strtab), shstr]
    types = [0] + [1] * len(texts) + [1] + [4] * len(tables) + [2, 3, 3]
    flags = [0] + [0x6] * len(texts) + [0x2] + [0x40] * len(tables) + [0, 0, 0]
    links = [0] * (len(texts) + 2) + [symtab_index] * len(tables) + [symtab_index + 1, 0, 0]
    infos = [0] * (len(texts) + 2) + [group + 1 for group in relocated] + [0, 0, 0]
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


def dispatcher(rows: int, dot: int = 0) -> str:
    """Mangled execute<Dot>() dispatcher symbol of one probe inventory entry."""
    return (f"_ZNK9llaminar23cpu16native_quantised27CPUNativeNEONProjectionTileILm{rows}EE7executeILNS1_11multi_scale"
            f"11NEONByteDotE{dot}EEEvv")


DISPATCHER_BODY = "\tcmp\tw0, #1\n\tret"


def programs(bodies: dict[tuple, str] | None = None, keys: list[tuple] | None = None,
             dispatchers: bool = True) -> list[tuple[str, str]]:
    """Tile programs plus the NEON inventory's four execute<ASIMD>() dispatchers."""
    return [(name(*key), (bodies or {}).get(key, ORDERED_CLEAN if key[2] == 1 else CLEAN))
            for key in (keys if keys is not None else NEON_KEYS)] + \
        ([(dispatcher(rows), DISPATCHER_BODY) for rows in (1, 2, 3, 4)] if dispatchers else [])


ORDERED_LINE = re.compile(r"^\s*fadd\s+v(\d+)\.4s,\s*v\1\.4s,\s*v\d+\.4s\s*$")


def assembly(functions: list[tuple[str, str]], ordered: bool = True) -> str:
    """Compiler-style assembly of the functions; with ordered=True each in-place fadd .4s is an inline-asm block."""
    out = ["\t.text"]
    for function, body in functions:
        out += [f"\t.globl\t{function}", f"{function}:", "\t.cfi_startproc"]
        for line in body.splitlines():
            if ordered and ORDERED_LINE.match(line):
                out += ["#APP", '// 66 "contract.h" 1', line, '// 0 "" 2', "#NO_APP"]
            else:
                out.append(line)
        out += ["\t.cfi_endproc", f"\t.size\t{function}, .-{function}"]
    return "\n".join(out) + "\n"


def audit_witness(bodies: dict[tuple, str] | None = None, extra: list[tuple[str, str]] | None = None,
                  keys: list[tuple] | None = None, inventory: str | None = NEON_INVENTORY, mode: str = "NEON",
                  macho: bool = False, symbols: list[tuple] | None = None,
                  sizes: dict[str, int] | None = None) -> dict[str, int]:
    functions = to_abi(programs(bodies, keys) + (extra or []), macho)
    options = abi_options({key: value for key, value in (("symbols", symbols), ("sizes", sizes)) if value}, macho)
    return CHECKER.audit(witness_object(functions, inventory, macho, **options), disassembly(functions, macho), mode)


def ordered_sites(count: int = 1) -> dict:
    """Committed ordered-add site counts naming every GPU-aligned NEON program of the fixture."""
    return {"GNU-14": {"NEON": {":".join(map(str, key)): count for key in NEON_KEYS if key[2] == 1}}}


def chain_counts() -> dict:
    """Committed chain witness counts of the fixture, written out: each GPU-aligned program holds its one template
    site and no off-chain fadd; no audited function holds a fused instruction."""
    return {"GNU-14": {"NEON": {
        "tile_chain": {":".join(map(str, key)): {"sites": 1, "offchain": 0} for key in NEON_KEYS if key[2] == 1},
        "fused": {":".join(map(str, key)): {"fused": {}} for key in NEON_KEYS}}}}


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
        self.assertEqual(self.spills("\tcsneg w5, w1, w5, mi\n\tret\n"), 0)  # 32-bit form: no address involved.
        self.fatal("\tmov x9, sp\n\tcsneg w5, w9, w5, mi\n\tret\n", "32-bit view")
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
        for listing, spilled in (("\tld1 {v26.4s-v27.4s}, [x0]\n", 0), ("\tst1 {v0.16b-v1.16b}, [sp]\n", 1)):
            with self.subTest(listing=listing):
                self.assertEqual(self.spills("\tsub sp, sp, #32\n" + listing + "\tadd sp, sp, #32\n\tret\n"), spilled)
        covered = analyse(frame + "\tst1 {v0.16b-v1.16b}, [sp]\n" + tail)  # 32 bytes cover the sp+8 slot whole.
        self.assertEqual((covered.spills, covered.errors), (1, []))
        self.fatal(frame + "\tadd x11, sp, #12\n\tst1 {v0.4s-v1.4s}, [x11]\n" + tail, "partial overwrite")
        for unknown in ("\tst1 {v0.16b-v1.8b}, [sp]\n", "\tst1 {v0}, [sp]\n", "\tld2 {v0.4s}, [sp]\n"):
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

    def test_lane_moves_between_general_and_vector_registers(self) -> None:
        # UMOV (mov x, v.d[]) defines a general register that never holds a stack address.
        self.assertEqual(self.spills("\tmov\tx9, v1.d[0]\n\tstr\tx9, [x0]\n\tret\n"), 0)
        # INS (mov v.d[], x) and FCVTL2 are FP/SIMD writes; a stack address may not enter a vector lane.
        self.assertEqual(self.spills("\tmov\tv1.d[1], x2\n\tfcvtl2\tv0.4s, v1.8h\n\tret\n"), 0)
        self.fatal("\tadd\tx2, sp, #16\n\tmov\tv1.d[1], x2\n\tret\n", "a stack address moves into an FP/SIMD register")
        # A general register redefined by UMOV no longer carries the stack address it held.
        self.assertEqual(self.spills("\tadd\tx9, sp, #16\n\tmov\tx9, v1.d[0]\n\tstr\tq0, [x9]\n\tret\n"), 0)

    def test_fp_data_laundered_through_general_registers_is_spill_traffic(self) -> None:
        # UMOV/FMOV to a general register, a stack store, a stack load and INS/FMOV/DUP back: one spill, one reload.
        self.assertEqual(self.spills("\tmov\tx9, v1.d[0]\n\tstr\tx9, [sp, #16]\n\tldr\tx9, [sp, #16]\n"
                                     "\tmov\tv1.d[0], x9\n\tret\n"), 2)
        # A pair store, an intermediate register move on either side, other move forms and widths.
        cases = {"pair store": "\tmov\tx9, v1.d[0]\n\tstp\tx9, x10, [sp, #16]\n\tret\n",
                 "moved before the store": "\tmov\tx9, v1.d[0]\n\tmov\tx10, x9\n\tstr\tx10, [sp, #16]\n\tret\n",
                 "fmov x, d": "\tfmov\tx9, d1\n\tstur\tx9, [x29, #-8]\n\tret\n",
                 "fmov w, s": "\tfmov\tw9, s1\n\tstr\tw9, [sp, #8]\n\tret\n",
                 "reloaded into another register": "\tldr\tx9, [sp, #16]\n\tmov\tx10, x9\n\tmov\tv1.d[0], x10\n\tret\n",
                 "reloaded by ins": "\tldr\tx9, [sp, #16]\n\tins\tv1.d[1], x9\n\tret\n",
                 "reloaded by fmov": "\tldr\tw9, [sp, #8]\n\tfmov\ts1, w9\n\tret\n",
                 "reloaded by dup": "\tldp\tw9, w10, [sp, #8]\n\tdup\tv1.4s, w10\n\tret\n",
                 "tainted on one path": "\tcbz\tw0, .Ls\n\tmov\tx9, v1.d[0]\n.Ls:\n\tstr\tx9, [sp, #16]\n\tret\n"}
        for case, body in cases.items():
            with self.subTest(case=case):
                self.assertEqual(self.spills(body), 1)
        # Not FP/SIMD data: a computed value, a non-stack destination or source, a conversion.
        for body in ("\tmov\tx9, v1.d[0]\n\tadd\tx9, x9, #1\n\tstr\tx9, [sp, #16]\n\tret\n",
                     "\tmov\tx9, v1.d[0]\n\tstr\tx9, [x0]\n\tret\n",
                     "\tldr\tx9, [x0]\n\tmov\tv1.d[0], x9\n\tret\n",
                     "\tldr\tw9, [sp, #8]\n\tscvtf\ts1, w9\n\tret\n",
                     "\tldr\tx9, [sp, #16]\n\tldr\tx9, [x0]\n\tmov\tv1.d[0], x9\n\tret\n"):
            with self.subTest(body=body):
                self.assertEqual(self.spills(body), 0)

    def test_relocated_data_addresses_are_never_stack(self) -> None:
        functions = [("f", "\tadrp x3, 0x0 <f> !R_AARCH64_ADR_GOT_PAGE:stderr\n"
                           "\tldr x3, [x3] !R_AARCH64_LD64_GOT_LO12_NC:stderr\n\tstr q0, [sp, #16]\n\tret\n")]
        bodies = CHECKER.normalize_disassembly(disassembly(functions), image(functions))
        self.assertEqual([line.strip() for line in bodies["f"][1:3]], ["adrp\tx3, stderr", "ldr\tx3, [x3, :lo12:stderr]"])
        self.assertEqual(CHECKER.analyse_function("f", bodies, CHECKER.GENERIC, set()), (1, []))
        for instruction in ("ldr q0, [sp] !R_AARCH64_LDST128_ABS_LO12_NC:x", "add x9, sp, #0 !R_AARCH64_ADD_ABS_LO12_NC:x"):
            with self.subTest(instruction=instruction):
                self.fatal(f"\t{instruction}\n\tstr q0, [x9]\n\tret\n", "relocated")


class ObservedFormsTest(unittest.TestCase):
    """Forms admitted from the production compact inventory: each one by its existing semantics class."""

    def run_body(self, body: str) -> Result:
        return analyse(body)

    def fatal(self, body: str, fragment: str) -> None:
        errors = self.run_body(body).errors
        self.assertTrue(any(fragment in error for error in errors), (body, errors))

    def test_smov_carries_fp_data_into_a_general_register(self) -> None:
        stored = self.run_body("\tsmov\tw9, v1.h[2]\n\tstr\tw9, [sp, #8]\n\tret\n")
        self.assertEqual((stored.spills, stored.errors), (1, []))
        computed = self.run_body("\tsmov\tw9, v1.h[2]\n\tand\tw9, w9, w3\n\tstr\tw9, [sp, #8]\n\tret\n")
        self.assertEqual((computed.spills, computed.errors), (0, []))

    def test_scalar_pair_store_through_the_stack_is_fp_spill_never_an_abi_save(self) -> None:
        spill = self.run_body("\tsub\tsp, sp, #16\n\tstp\ts0, s1, [sp]\n\tadd\tsp, sp, #16\n\tret\n")
        self.assertEqual((spill.spills, spill.errors), (1, []))
        self.fatal("\tstp\ts8, s9, [sp, #-16]!\n\tfadd\ts8, s0, s1\n\tldp\ts8, s9, [sp], #16\n\tret\n",
                   "clobbers callee-saved v8 before saving it")

    def test_scalar_and_whole_vector_writes_clobber_the_callee_saved_low_half(self) -> None:
        for write in ("fadd\ts8, s0, s1", "ushr\td8, d1, #3", "saddlv\th8, v1.8h", "uzp1\tv8.8h, v1.8h, v2.8h",
                      "uzp2\tv8.16b, v1.16b, v2.16b", "mul\tv8.4s, v1.4s, v2.4s", "uxtl\tv8.8h, v1.8b",
                      "fmov\tv8.4s, #1.0"):
            with self.subTest(write=write):
                self.fatal(f"\t{write}\n\tret\n", "clobbers callee-saved v8 before saving it")
        self.assertEqual(self.run_body("\tfadd\ts16, s0, s1\n\tuzp1\tv17.8h, v1.8h, v2.8h\n\tret\n").errors, [])

    def test_general_forms_keep_stack_address_rules(self) -> None:
        for form in ("and\tw10, w9, w3", "lsr\tw10, w9, #2", "add\tw10, w3, w9, lsl #2", "ccmp\tw9, #0, #4, ne"):
            with self.subTest(form=form):
                self.fatal(f"\tadd\tx9, sp, #16\n\t{form}\n\tret\n", "32-bit view of a stack address")
        moved = self.run_body("\tadd\tx9, sp, #16\n\tsubs\tx10, x9, x3\n\tstr\tq0, [x10]\n\tret\n")
        self.assertEqual((moved.spills, moved.errors), (1, []))  # subs of a stack address: unknown-offset stack

    def test_unscaled_binary16_load_has_the_semantics_of_ldr_h(self) -> None:
        # ldur h is the unscaled-offset spelling of ldr h: same 2-byte footprint, stack class by base provenance.
        for spelling in ("ldr", "ldur"):
            instruction = CHECKER.Instruction(f"{spelling}\th30, [x4, #-32]", CHECKER.GENERIC)
            with self.subTest(spelling=spelling):
                self.assertTrue(instruction.admitted())
                self.assertEqual(CHECKER.footprint(instruction, "h30"), 2)
        frame, exit_ = "\tstp\tx29, x30, [sp, #-48]!\n\tmov\tx29, sp\n", "\tldp\tx29, x30, [sp], #48\n\tret\n"
        for base in ("[x29, #-2]", "[sp, #18]"):
            with self.subTest(base=base):
                # A stack reload of a binary16 value is FP/SIMD stack traffic, counted like ldr h.
                ldur = self.run_body(frame + f"\tstr\th0, [sp, #18]\n\tldur\th1, {base}\n" + exit_)
                ldr = self.run_body(frame + "\tstr\th0, [sp, #18]\n\tldr\th1, [sp, #18]\n" + exit_)
                self.assertEqual((ldur.spills, ldur.errors), (2, []))
                self.assertEqual((ldur.spills, ldur.errors), (ldr.spills, ldr.errors))
                counts, _, errors = traffic(frame + f"\tldur\th1, {base}\n" + exit_)
                self.assertEqual((errors, {k: v for k, v in counts.items() if v}), ([], {"fp": {"0": 1}}))
        # A heap (descriptor) load is no stack traffic, and the h write clobbers a callee-saved v8-v15 low half.
        heap = self.run_body("\tldur\th1, [x0, #-2]\n\tret\n")
        self.assertEqual((heap.spills, heap.errors), (0, []))
        self.fatal("\tldur\th8, [x0, #-2]\n\tret\n", "clobbers callee-saved v8 before saving it")

    def test_neighbouring_unobserved_forms_stay_outside_the_grammar(self) -> None:
        for form in ("smov\tx9, v1.h[2]", "fadd\th0, h1, h2", "stp\th0, h1, [sp]", "ccmp\tx1, #0, #4, ne",
                     "subs\tw1, w2, w3, lsl #2", "ushr\ts0, s1, #3", "stur\th0, [x29, #-2]", "ldur\tb0, [x29, #-1]"):
            with self.subTest(form=form):
                self.fatal(f"\t{form}\n\tret\n", "admitted grammar")


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

    def test_only_the_low_half_of_v8_to_v15_is_callee_saved(self) -> None:
        # AAPCS64 preserves bits 0-63 (d8-d15): a write confined to bits 64-127 is no clobber.
        for upper in ("mov\tv14.d[1], x0", "mov\tv14.s[2], w0", "mov\tv14.h[7], w0", "ld1\t{v14.s}[3], [x0]",
                      "mov\tv14.b[8], w0"):
            with self.subTest(upper=upper):
                self.assertEqual(self.check(f"\t{upper}\n\tret\n").errors, [])
        for low in ("mov\tv14.d[0], x0", "mov\tv14.s[1], w0", "mov\tv14.b[7], w0", "ld1\t{v14.s}[1], [x0]",
                    "fcvtl2\tv14.4s, v1.8h", "fmov\td14, x0", "mov\tv14.16b, v1.16b", "ldr\ts14, [x0]"):
            with self.subTest(low=low):
                self.assertFatal(f"\t{low}\n\tret\n", "clobbers callee-saved v14 before saving it")

    def test_shrink_wrapped_save_restore_inside_a_loop(self) -> None:
        # Each iteration saves the caller's d14 again after the previous iteration restored it.
        loop = ("\tstp\tx29, x30, [sp, #-32]!\n.Lloop:\n\tcbz\tx0, .Ldone\n\tstr\td14, [sp, #16]\n"
                "\tmovi\tv14.4s, #0x0\n\tldr\td14, [sp, #16]\n\tsub\tx0, x0, #1\n\tb\t.Lloop\n"
                ".Ldone:\n\tldp\tx29, x30, [sp], #32\n\tret\n")
        self.assertEqual((self.check(loop).spills, self.check(loop).errors), (0, []))
        # A clobber between the restore and the next save (or the return) is not the caller's value.
        self.assertFatal(loop.replace("\tsub\tx0", "\tmovi\tv14.4s, #0x0\n\tsub\tx0"), "d14")
        # Saving while the register may be clobbered on one path is no save.
        self.assertFatal(loop.replace("\tcbz\tx0, .Ldone\n", "\tcbz\tx0, .Ldone\n\tcbz\tx1, .Ls\n"
                                      "\tstr\td14, [sp, #16]\n\tmovi\tv14.4s, #0x0\n.Ls:\n", 1),
                         "live save slot")

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

    def test_jump_table_br_stays_fatal_after_adr_naming(self) -> None:
        # J-1: d08deb515 only names an unrelocated same-function ADR (a jump table base) by its local label so the
        # normaliser completes; the table's indirect `br` is still outside the grammar, in a program and in a
        # helper of the call model. Admitting a jump table needs its own bounded, reviewed rule.
        table = "\tadr x1, .Ltable\n\tldrb w2, [x1, w0, uxtw]\n\tadd x1, x1, w2, sxtb #2\n\tbr x1\n" \
                ".Ltable:\n\tret\n"
        # The named ADR itself is outside the grammar (adr x,sym), and the br alone is too: either way fatal.
        self.assertFatal(table, "admitted grammar")
        self.assertFatal("\tadd x1, x1, w2, sxtb #2\n\tbr x1\n", "admitted grammar")
        errors = called("\tbl helper\n\tret\n", {"helper": table})[1]
        self.assertTrue(any("helper helper" in error and "admitted grammar: adr x,sym" in error for error in errors),
                        errors)

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

    def test_source_names_resolve_relative_include_components(self) -> None:
        # One source file has one name whichever production directory includes it (../ components resolved).
        for path, name in (("src/v2/execution/stages/../../kernels/cpu/gemm/X.h", "src/v2/kernels/cpu/gemm/X.h"),
                           ("/t/tests/v2/portability/../../../src/v2/k/X.h", "src/v2/k/X.h"),
                           ("/src/src/v2/k/./X.h", "src/v2/k/X.h"), ("tests/v2/p/A.cpp", "tests/v2/p/A.cpp"),
                           ("/usr/include/c++/14/bits/x.h", "x.h")):
            with self.subTest(path=path):
                self.assertEqual(CHECKER.source_name(path), name)

    def test_unrelocated_adr_names_only_a_label_of_its_own_function(self) -> None:
        # GAS resolves an ADR of a label of the same function (a jump table base) without a relocation: it is
        # named by the function's local label; an unrelocated ADR (or literal load) anywhere else still fails.
        functions = [("f", "\tadd x0, x1, #1\n\tret\n")]
        inside = self.TEXT.replace("add\tx0, x1, #1", "adr\tx0, 0x4 <f+0x4>")
        body = CHECKER.normalize_disassembly(inside, image(functions))["f"]
        self.assertEqual([line.strip() for line in body[1:]], ["adr\tx0, .L4", ".L4:", "ret"])
        for bad in (self.TEXT.replace("add\tx0, x1, #1", "adr\tx0, 0x40 <f+0x40>"),
                    self.TEXT.replace("add\tx0, x1, #1", "ldr\tx0, 0x4 <f+0x4>")):
            with self.subTest(bad=bad[-60:]):
                with self.assertRaisesRegex(ValueError, "unrelocated address operand"):
                    CHECKER.normalize_disassembly(bad, image(functions))

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

    def test_section_relative_local_calls_name_only_an_exact_function_start(self) -> None:
        # GCC relocates a call to an internal-linkage function against its section's STT_SECTION symbol.
        def call(target: str, helper: str = "\tret\n", extra: list | None = None) -> list[tuple[str, str]]:
            return [("f", f"\tbl helper !R_AARCH64_CALL26:{target}\n\tret\n"), ("helper", helper)] + (extra or [])
        functions = call(".text+0x8")
        bodies = CHECKER.normalize_disassembly(disassembly(functions, shared=True),
                                               CHECKER.read_object(witness_object(functions, shared=True)))
        self.assertEqual(bodies["f"][1].strip(), "bl\thelper")
        # Aliases of one extent resolve; the call is then judged by the ordinary call rules (fatal in a tile).
        aliased = call(".text+0x8")
        aliased_object = witness_object(aliased, shared=True, symbols=[("helper2", "function", "global", "helper", 8)],
                                        sizes={"helper2": 4})
        self.assertEqual(CHECKER.normalize_disassembly(disassembly(aliased, shared=True),
                                                       CHECKER.read_object(aliased_object))["f"][1].strip(),
                         "bl\thelper")
        # An interior or foreign address, or a relocation against a non-section symbol, fails.
        for target, records in ((".text+0xc", None), (".text+0x10", None),
                                ("helper+0x4", None),
                                (".text+0x8", [(".text", "notype", "local", "f", 0)])):
            with self.subTest(target=target, records=records):
                bad = call(target, "\tnop\n\tret\n")
                obj = witness_object(bad, shared=True, symbols=records)
                text = disassembly(bad, shared=True)
                if records:  # The relocation names the notype symbol called .text, not the section symbol.
                    obj = witness_object([(n, b.replace(":.text+", ":.textalias+")) for n, b in bad], shared=True,
                                         symbols=[(".textalias", "notype", "local", "f", 0)])
                    obj = obj.replace(b".textalias\0", b".text\0\0\0\0\0\0")
                with self.assertRaisesRegex(ValueError, "addend"):
                    CHECKER.normalize_disassembly(text, CHECKER.read_object(obj))

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


HELPER = ("inline float32x4_t addOrderedNEON(float32x4_t earlier, float32x4_t later) noexcept\n"
          "{\n    asm(\"fadd %0.4s, %0.4s, %1.4s\" : \"+w\"(earlier) : \"w\"(later));\n    return earlier;\n}\n")


class OrderedAddSourceTest(unittest.TestCase):
    """Only the exact ordered-add helper body is an admitted non-empty template; every variation fails."""

    def test_the_exact_helper_is_admitted_once(self) -> None:
        self.assertEqual(CHECKER.inline_assembly_violations(HELPER), [])
        self.assertEqual(CHECKER.inline_assembly_violations("namespace n {\n" + HELPER + "}\n"), [])
        self.assertTrue(CHECKER.inline_assembly_violations(HELPER + "namespace m {\n" + HELPER + "}\n"))

    def test_every_variation_is_unproved(self) -> None:
        variants = {
            "second instruction": ('"fadd %0.4s, %0.4s, %1.4s"', '"fadd %0.4s, %0.4s, %1.4s\\n\\tnop"'),
            "two templates": ('"fadd %0.4s, %0.4s, %1.4s"', '"fadd %0.4s, %0.4s, " "%1.4s"'),
            "other opcode": ('"fadd %0.4s, %0.4s, %1.4s"', '"fmla %0.4s, %0.4s, %1.4s"'),
            "other arrangement": ('"fadd %0.4s, %0.4s, %1.4s"', '"fadd %0.2s, %0.2s, %1.2s"'),
            "destination not first source": ('"fadd %0.4s, %0.4s, %1.4s"', '"fadd %0.4s, %1.4s, %0.4s"'),
            "memory operand": (': \"w\"(later)', ': \"m\"(later)'),
            "memory clobber": ('\"w\"(later));', '\"w\"(later) : \"memory\");'),
            "register clobber": ('\"w\"(later));', '\"w\"(later) : \"v0\");'),
            "volatile": ("asm(", "asm volatile("),
            "goto": ("asm(", "asm goto("),
            "untied output": ('\"+w\"(earlier)', '\"=w\"(earlier)'),
            "extra input": ('\"w\"(later))', '\"w\"(later), \"w\"(earlier))'),
            "other helper name": ("addOrderedNEON", "addOrderedNEON2"),
            "other parameters": ("float32x4_t later)", "float32x4_t later, int x)"),
            "extra statement": ("    return earlier;", "    later = earlier;\n    return earlier;"),
        }
        for case, (old, new) in variants.items():
            with self.subTest(case=case):
                variant = HELPER.replace(old, new)
                self.assertNotEqual(variant, HELPER)
                self.assertTrue(CHECKER.inline_assembly_violations(variant), variant)
        outside = 'void f(float32x4_t earlier, float32x4_t later) { asm("fadd %0.4s, %0.4s, %1.4s" : "+w"(earlier) : "w"(later)); }'
        self.assertTrue(CHECKER.inline_assembly_violations(outside))


class OrderedAddObjectTest(unittest.TestCase):
    """Ordered-add sites are bound to the object's instruction at the same index; programs carry them by policy."""

    GA, BN = (2, 0, 1, 0, 0, 0), (2, 0, 0, 0, 0, 0)

    def evidence(self, functions: list[tuple[str, str]], text: str | None = None) -> dict:
        functions = list(functions)
        return CHECKER.ordered_add_evidence(witness_object(functions), disassembly(functions),
                                            assembly(functions) if text is None else text)

    def test_faithful_sites_are_recorded_per_program(self) -> None:
        sites = self.evidence(programs())
        self.assertEqual(set(sites), {":".join(map(str, key)) for key in NEON_KEYS if key[2] == 1})
        self.assertTrue(all(indices == [3] for indices in sites.values()))

    def test_policy_rules(self) -> None:
        with self.assertRaisesRegex(ValueError, "without an ordered add"):
            self.evidence(programs({self.GA: CLEAN}))
        with self.assertRaisesRegex(ValueError, "backend-native tile program holds"):
            self.evidence(programs({self.BN: ORDERED_CLEAN}))

    def test_block_contents_and_object_binding_fail_closed(self) -> None:
        functions = programs()
        target = name(*self.GA)
        good = assembly(functions)
        block = "#APP\n// 66 \"contract.h\" 1\n\tfadd\tv0.4s, v0.4s, v1.4s\n// 0 \"\" 2\n#NO_APP"
        self.assertIn(block, good)
        first = good.index(block, good.index(target + ":"))
        def edit(new: str) -> str:
            return good[:first] + new + good[first + len(block):]
        cases = {
            "second instruction": block.replace("v1.4s\n", "v1.4s\n\tmov\tx0, x1\n"),
            "wrong opcode": block.replace("fadd\t", "fmul\t"),
            "wrong arrangement": block.replace("v0.4s, v0.4s, v1.4s", "v0.2s, v0.2s, v1.2s"),
            "destination not first source": block.replace("v0.4s, v0.4s, v1.4s", "v0.4s, v1.4s, v0.4s"),
            "object differs": block.replace("v0.4s, v0.4s, v1.4s", "v2.4s, v2.4s, v1.4s"),
            "count differs": block.replace("v1.4s\n", "v1.4s\n#NO_APP\n\tnop\n#APP\n"),
            "unterminated": block.replace("#NO_APP", ""),
            "nop after the add": block.replace("v1.4s\n", "v1.4s\n\tnop\n"),
            "nop before the add": block.replace("\tfadd", "\tnop\n\tfadd"),
        }
        for case, new in cases.items():
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    self.evidence(functions, edit(new))
        with self.assertRaisesRegex(ValueError, "outside every object function"):
            self.evidence(functions, "#APP\n\tfadd\tv0.4s, v0.4s, v1.4s\n#NO_APP\n" + good)

    def test_alignment_nops_in_the_object_keep_the_binding(self) -> None:
        functions = programs()
        padded = programs({self.GA: ORDERED_CLEAN.replace("\tmov", "\tnop\n\tnop\n\tmov")})
        sites = CHECKER.ordered_add_evidence(witness_object(padded), disassembly(padded), assembly(functions))
        self.assertEqual(sites[":".join(map(str, self.GA))], [3])

    def test_extra_compiler_instruction_in_the_object_breaks_the_binding(self) -> None:
        functions = programs()
        text = assembly(functions)
        widened = programs({self.GA: ORDERED_CLEAN.replace("\tret", "\tmov\tx0, x1\n\tret")})
        with self.assertRaisesRegex(ValueError, "instruction counts differ"):
            CHECKER.ordered_add_evidence(witness_object(widened), disassembly(widened), text)


def traffic(body: str) -> tuple[dict, list, list[str]]:
    """Stack-traffic record of one function through the production front end: (counts, loops, errors)."""
    functions, options = to_abi([("f", body)], False), abi_options({}, False)
    obj = CHECKER.read_object(witness_object(functions, **options))
    bodies = CHECKER.normalize_disassembly(disassembly(functions), obj)
    noreturn = CHECKER.noreturn_symbols(bodies, CHECKER.GENERIC, CHECKER.trusted_imports(obj), obj.prefix)
    record, errors = CHECKER.analyse_traffic(functions[0][0], bodies, CHECKER.GENERIC, noreturn)
    return record["counts"], record["loops"], errors


TLS = "_ZN9llaminar224tls_graph_capture_activeE"


def tls_sequence(register: str = "x2", symbols: tuple = (TLS,) * 4, order: tuple = (0, 1, 2, 3),
                 call: str | None = None) -> str:
    """One TLS descriptor call sequence (adrp/ldr/add/blr with its four relocations), optionally perturbed."""
    kinds = CHECKER.TLSDESC_SEQUENCE
    lines = [f"\tadrp\tx0, 0x0 <f> !{kinds[0]}:{symbols[0]}",
             f"\tldr\t{register}, [x0] !{kinds[1]}:{symbols[1]}",
             f"\tadd\tx0, x0, #0x0 !{kinds[2]}:{symbols[2]}",
             f"\t{call or 'blr ' + register} !{kinds[3]}:{symbols[3]}"]
    return "\n".join(lines[index] for index in order) + "\n"


def tlsle_sequence(base: str = "x1", dest: str = "x2", addends: tuple[int, int] = (0, 0)) -> str:
    """Recorded local-exec relocation shape; literal operands are relocation placeholders."""
    high, low = (CHECKER.relocation_target(TLS, addend) for addend in addends)
    return (f"\tadd\t{dest}, {base}, #0x0, lsl #12 !{CHECKER.TLSLE_PAIR[0]}:{high}\n"
            f"\tadd\t{dest}, {dest}, #0x0 !{CHECKER.TLSLE_PAIR[1]}:{low}\n")


class TLSLocalExecPairTest(unittest.TestCase):
    """Exact pair modelling, without weakening the generic relocation or call grammar."""

    def test_pair_preserves_stack_facts_and_has_no_call_traffic(self) -> None:
        body = ("\tstp\tx29, x30, [sp, #-32]!\n\tmov\tx29, sp\n"
                "\tadd\tx9, sp, #16\n\tmrs\tx1, tpidr_el0\n.Lk:\n" + tlsle_sequence() +
                "\tstr\tq0, [x9]\n\tsub\tx5, x5, #1\n\tcbnz\tx5, .Lk\n"
                "\tldp\tx29, x30, [sp], #32\n\tret\n")
        for _ in range(3):
            counts, _, errors = traffic(body)
            self.assertEqual(errors, [])
            self.assertEqual(counts["fp"], {"1": 1})
            self.assertFalse(counts["tls_call"])

    def test_stack_source_is_not_erased_by_zero_placeholders(self) -> None:
        _, _, errors = traffic("\tadd\tx1, sp, #0\n" + tlsle_sequence() + "\tret\n")
        self.assertTrue(any("relocated arithmetic on a stack address" in error for error in errors), errors)

    def test_orphan_mismatch_wrong_register_and_unsupported_forms_fail(self) -> None:
        pair = tlsle_sequence()
        cases = {
            "orphan high": pair.splitlines()[0] + "\n",
            "orphan low": pair.splitlines()[1] + "\n",
            "mismatched symbol": pair.replace(CHECKER.TLSLE_PAIR[1] + ":" + TLS,
                                               CHECKER.TLSLE_PAIR[1] + ":other"),
            "wrong register": pair.replace("x2, x2,", "x2, x3,"),
            "wrong shift": pair.replace("lsl #12", "lsl #16"),
            "32-bit": pair.replace("x2", "w2"),
            "SP": pair.replace("x1", "sp"),
            "nonzero placeholder": pair.replace("#0x0", "#0x1"),
            "not consecutive": pair.replace("\n", "\n\tnop\n", 1),
        }
        for name, body in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "TLS local-exec"):
                traffic(body + "\tret\n")

    def test_non_local_exec_and_unsupported_low_pair_fail_in_object_reader(self) -> None:
        functions = [("f", tlsle_sequence() + "\tret\n")]
        original = witness_object(functions)
        header = elf_header_offset(original, ".rela.text.f")
        offset = int.from_bytes(original[header + 24:header + 32], "little")
        for record, kind in ((0, 539), (1, 550), (1, 552)):
            data = bytearray(original)
            data[offset + record * 24 + 8:offset + record * 24 + 12] = kind.to_bytes(4, "little")
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "not admitted in code"):
                CHECKER.read_object(bytes(data))

    def test_identical_24_bit_addends_preserve_both_symbolic_operands(self) -> None:
        for addend in (0, 1, 16, 4095, 4096, (1 << 24) - 1):
            for base, dest in (("x0", "x0"), ("x20", "x4"), ("x30", "x30")):
                with self.subTest(addend=addend, base=base, dest=dest):
                    functions = [("f", tlsle_sequence(base, dest, (addend, addend)) + "\tret\n")]
                    image = CHECKER.read_object(witness_object(functions))
                    bodies = CHECKER.normalize_disassembly(disassembly(functions), image)
                    target = CHECKER.relocation_target(TLS, addend)
                    self.assertIn(f":tprel_hi12:{target}", str(bodies))
                    self.assertIn(f":tprel_lo12_nc:{target}", str(bodies))
                    counts, _, errors = traffic(functions[0][1])
                    self.assertEqual(errors, [])
                    self.assertTrue(all(not value for value in counts.values()), counts)
        _, _, errors = traffic("\tadd\tx1, sp, #0\n" + tlsle_sequence(addends=(16, 16)) + "\tret\n")
        self.assertTrue(any("relocated arithmetic on a stack address" in error for error in errors), errors)

    def test_unequal_one_sided_negative_and_out_of_range_addends_fail(self) -> None:
        for addends in ((16, 17), (16, 0), (0, 16), (-1, -1), (1 << 24, 1 << 24)):
            with self.subTest(addends=addends), self.assertRaisesRegex(ValueError, "TLS local-exec"):
                traffic(tlsle_sequence(addends=addends) + "\tret\n")
        pair = tlsle_sequence(addends=(16, 16))
        cases = (pair.replace(CHECKER.TLSLE_PAIR[1] + ":" + TLS, CHECKER.TLSLE_PAIR[1] + ":other"),
                 pair.replace("x1", "xzr"), pair.replace("x2", "x31"),
                 pair.replace("#0x0", "#0x1"), pair.replace("\n", "\n\tnop\n", 1))
        for pair in cases:
            with self.subTest(pair=pair), self.assertRaisesRegex(ValueError, "TLS local-exec"):
                traffic(pair + "\tret\n")

    def test_addend_and_function_boundary_fail(self) -> None:
        functions = [("f", tlsle_sequence() + "\tret\n")]
        original = witness_object(functions)
        header = elf_header_offset(original, ".rela.text.f")
        offset = int.from_bytes(original[header + 24:header + 32], "little")
        data = bytearray(original)
        data[offset + 16:offset + 24] = (1).to_bytes(8, "little")
        with self.assertRaisesRegex(ValueError, "TLS local-exec"):
            CHECKER.normalize_disassembly(disassembly(functions).replace(
                CHECKER.TLSLE_PAIR[0] + "\t" + TLS, CHECKER.TLSLE_PAIR[0] + "\t" + TLS + "+0x1"),
                CHECKER.read_object(bytes(data)))
        with self.assertRaisesRegex(ValueError, "printed relocation"):
            CHECKER.normalize_disassembly(disassembly(functions), CHECKER.read_object(bytes(data)))
        image = CHECKER.read_object(original)
        code, _ = CHECKER.parse_disassembly(disassembly(functions), False)
        with self.assertRaisesRegex(ValueError, "function boundary"):
            CHECKER.tlsle_groups(code, image, {"f": (".text.f", 0, 4), "g": (".text.f", 4, 12)})


def tlsie_sequence(base: str = "x1", dest: str = "x2") -> str:
    """Exact native GOT page/load shape with relocation placeholders."""
    return (f"\tadrp\t{base}, 0 <{TLS}> !{CHECKER.TLSIE_PAIR[0]}:{TLS}\n"
            f"\tldr\t{dest}, [{base}] !{CHECKER.TLSIE_PAIR[1]}:{TLS}\n")


class TLSInitialExecPairTest(unittest.TestCase):
    """Only the consecutive 541/542 pair defines non-stack data without traffic credit."""

    def test_exact_pair_aliasing_and_register_boundaries(self) -> None:
        for base, dest in (("x0", "x0"), ("x1", "x2"), ("x30", "x30"), ("x0", "x30")):
            with self.subTest(base=base, dest=dest):
                functions = [("f", tlsie_sequence(base, dest) + "\tret\n")]
                image = CHECKER.read_object(witness_object(functions))
                bodies = CHECKER.normalize_disassembly(disassembly(functions), image)
                self.assertIn(f":gottprel:{TLS}", str(bodies))
                self.assertIn(f":gottprel_lo12:{TLS}", str(bodies))

    def test_load_is_not_stack_or_call_or_invariant_credit(self) -> None:
        body = ("\tmrs\tx3, tpidr_el0\n.Lk:\n" + tlsie_sequence() +
                "\tsub\tx5, x5, #1\n\tcbnz\tx5, .Lk\n\tret\n")
        counts, _, errors = traffic(body)
        self.assertEqual(errors, [])
        self.assertTrue(all(not value for value in counts.values()), counts)

    def test_stack_derived_load_operand_fails_even_when_page_is_skipped(self) -> None:
        pair = tlsie_sequence().replace("\tldr", ".Lload:\n\tldr")
        _, _, errors = traffic("\tadd\tx1, sp, #0\n\tb\t.Lload\n" + pair + "\tret\n")
        self.assertTrue(any("stack-derived" in error for error in errors), errors)

    def test_load_does_not_seed_stack_or_erase_other_stack_facts(self) -> None:
        body = ("\tstp\tx29, x30, [sp, #-32]!\n\tmov\tx29, sp\n"
                "\tadd\tx9, sp, #16\n" + tlsie_sequence() +
                "\tstr\tq0, [x9]\n\tldr\tx3, [x2]\n"
                "\tldp\tx29, x30, [sp], #32\n\tret\n")
        counts, _, errors = traffic(body)
        self.assertEqual(errors, [])
        self.assertEqual(counts["fp"], {"0": 1})
        self.assertFalse(counts["tls_call"])

    def test_orphans_mixed_symbols_and_instruction_variants_fail(self) -> None:
        pair = tlsie_sequence()
        cases = [pair.splitlines()[0] + "\n", pair.splitlines()[1] + "\n",
                 pair.replace(CHECKER.TLSIE_PAIR[1] + ":" + TLS, CHECKER.TLSIE_PAIR[1] + ":other"),
                 pair.replace("[x1]", "[x3]"), pair.replace("ldr\tx2", "ldr\tw2"),
                 pair.replace("x1", "sp"), pair.replace("x1", "xzr"),
                 pair.replace("x2", "xzr"), pair.replace("x1", "x31"),
                 pair.replace("[x1]", "[x1, x3]"), pair.replace("[x1]", "[x1, #8]"),
                 pair.replace("[x1]", "[x1]!"), pair.replace("[x1]", "[x1], #8"),
                 pair.replace("\n", "\n\tnop\n", 1)]
        for body in cases:
            with self.subTest(body=body), self.assertRaisesRegex(ValueError, "TLS initial-exec"):
                traffic(body + "\tret\n")

    def test_addends_and_function_boundary_fail(self) -> None:
        functions = [("f", tlsie_sequence() + "\tret\n")]
        original = witness_object(functions)
        header = elf_header_offset(original, ".rela.text.f")
        offset = int.from_bytes(original[header + 24:header + 32], "little")
        for record in (0, 1):
            for addend in (1, -1):
                data = bytearray(original)
                data[offset + record * 24 + 16:offset + record * 24 + 24] = addend.to_bytes(8, "little", signed=True)
                image = CHECKER.read_object(bytes(data))
                code, _ = CHECKER.parse_disassembly(disassembly(functions), False)
                with self.subTest(record=record, addend=addend), self.assertRaisesRegex(ValueError, "addend"):
                    CHECKER.tlsie_groups(code, image, CHECKER.function_extents(image))
        image = CHECKER.read_object(original)
        code, _ = CHECKER.parse_disassembly(disassembly(functions), False)
        with self.assertRaisesRegex(ValueError, "function boundary"):
            CHECKER.tlsie_groups(code, image, {"f": (".text.f", 0, 4), "g": (".text.f", 4, 12)})

    def test_assembly_text_without_object_pair_fails_closed(self) -> None:
        body = ['\t.section .text.f,"ax"',
                f"\tadrp\tx1, :gottprel:{TLS}",
                f"\tldr\tx2, [x1, :gottprel_lo12:{TLS}]", "\tret"]
        _, errors = CHECKER.analyse_traffic("f", {"f": body}, CHECKER.GENERIC, set())
        self.assertTrue(any("object-validated pair" in error for error in errors), errors)
        # Flattening authenticated text back to plain assembly must lose admission.
        functions = [("f", tlsie_sequence() + "\tret\n")]
        bodies = CHECKER.normalize_disassembly(disassembly(functions), CHECKER.read_object(witness_object(functions)))
        plain = {name: [str(line) for line in lines] for name, lines in bodies.items()}
        _, errors = CHECKER.analyse_traffic("f", plain, CHECKER.GENERIC, set())
        self.assertTrue(any("object-validated pair" in error for error in errors), errors)

    def test_other_tls_relocations_remain_rejected(self) -> None:
        functions = [("f", tlsie_sequence() + "\tret\n")]
        original = witness_object(functions)
        header = elf_header_offset(original, ".rela.text.f")
        offset = int.from_bytes(original[header + 24:header + 32], "little")
        for kind in (512, 513, 539, 543, 544, 545, 546, 547, 548, 550, 552):
            data = bytearray(original)
            data[offset + 8:offset + 12] = kind.to_bytes(4, "little")
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "not admitted in code"):
                CHECKER.read_object(bytes(data))


class TLSDescriptorTest(unittest.TestCase):
    """The only admitted indirect call: an exact TLS descriptor sequence. It clobbers x0/x30 only, is counted
    per loop, and every perturbation (reorder, mixed symbol, stray relocation, CALL on a non-blr, register
    mismatch) fails before analysis."""

    FRAME, EXIT = "\tstp\tx29, x30, [sp, #-32]!\n\tmov\tx29, sp\n", "\tldp\tx29, x30, [sp], #32\n\tret\n"

    def test_exact_sequence_is_counted_per_loop_and_clobbers_only_x0_and_x30(self) -> None:
        body = (self.FRAME + "\tadd\tx9, sp, #16\n.Lk:\n" + tls_sequence() +
                "\tmrs\tx1, tpidr_el0\n\tldrb\tw3, [x1, x0]\n\tstr\tq0, [x9]\n\tsub\tx5, x5, #1\n"
                "\tcbnz\tx5, .Lk\n" + self.EXIT)
        counts, loops, errors = traffic(body)
        self.assertEqual(errors, [])
        self.assertEqual(counts["tls_call"], {"1": 1})
        # x9 still holds its stack address across the call: the q0 store through it is an FP spill in the loop.
        self.assertEqual(counts["fp"], {"1": 1})

    def test_x0_and_x30_are_not_preserved(self) -> None:
        # x0 held a stack address before the call; after it, x0 is the TLS offset (no stack address).
        body = (self.FRAME + "\tadd\tx19, sp, #16\n" + tls_sequence() + "\tstr\tq0, [x0]\n" + self.EXIT)
        counts, _, errors = traffic(body)
        self.assertEqual((errors, counts["fp"]), ([], {}))
        bad = self.FRAME + "\tmrs\tx1, nzcv\n" + self.EXIT
        self.assertTrue(any("system register" in error for error in traffic(bad)[2]))

    def test_nzcv_is_undefined_after_the_resolver_call(self) -> None:
        # A compare before the call cannot feed a conditional after it; a fresh compare (or a path join where
        # one side is clobbered) decides.
        stale = (self.FRAME + "\tcmp\tx5, #3\n" + tls_sequence() + "\tcsel\tx6, x6, x7, eq\n" + self.EXIT)
        self.assertTrue(any("NZCV" in error for error in traffic(stale)[2]))
        latch = (self.FRAME + ".Lk:\n\tsub\tx5, x5, #1\n\tcmp\tx5, #0\n" + tls_sequence() + "\tb.ne\t.Lk\n" + self.EXIT)
        self.assertTrue(any("NZCV" in error for error in traffic(latch)[2]))
        joined = (self.FRAME + "\tcmp\tx5, #3\n\tcbz\tx9, .Lj\n" + tls_sequence() + ".Lj:\n\tb.eq\t.Lx\n.Lx:\n" + self.EXIT)
        self.assertTrue(any("NZCV" in error for error in traffic(joined)[2]))
        fresh = (self.FRAME + "\tcmp\tx5, #3\n" + tls_sequence() + "\tcmp\tx5, #4\n\tcsel\tx6, x6, x7, eq\n" + self.EXIT)
        self.assertEqual(traffic(fresh)[2], [])

    def test_subs_redefines_and_ccmp_reads_nzcv(self) -> None:
        fresh = (self.FRAME + tls_sequence() + "\tsubs\tx5, x5, x6\n\tccmp\tw7, #0, #4, ne\n\tb.eq\t.Lx\n.Lx:\n"
                 + self.EXIT)
        self.assertEqual(traffic(fresh)[2], [])
        stale = self.FRAME + "\tsubs\tx5, x5, x6\n" + tls_sequence() + "\tccmp\tw7, #0, #4, ne\n" + self.EXIT
        self.assertTrue(any("NZCV" in error for error in traffic(stale)[2]))

    def test_every_perturbation_fails(self) -> None:
        cases = {"reordered": tls_sequence(order=(1, 0, 2, 3)),
                 "mixed symbol": tls_sequence(symbols=(TLS, TLS, "other", TLS)),
                 "register mismatch": tls_sequence().replace("blr x2", "blr x3"),
                 "call on a non-blr": tls_sequence(call="bl helper"),
                 "partial": "\n".join(tls_sequence().splitlines()[:3]) + "\n",
                 "stray adrp page": f"\tadrp\tx0, 0x0 <f> !{CHECKER.TLSDESC_SEQUENCE[0]}:{TLS}\n",
                 "stray call": f"\tblr\tx2 !{CHECKER.TLSDESC_SEQUENCE[3]}:{TLS}\n",
                 "result not x0": tls_sequence().replace("adrp\tx0", "adrp\tx1")}
        for case, sequence in cases.items():
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    errors = traffic(self.FRAME + sequence + self.EXIT)[2]
                    if errors:
                        raise ValueError(errors)

    def test_other_indirect_calls_stay_structural(self) -> None:
        # A plain blr (no TLS descriptor relocations) is outside the grammar; with a register-relocation-free
        # call form it would be an indirect branch: both structural.
        errors = traffic(self.FRAME + "\tblr\tx2\n" + self.EXIT)[2]
        self.assertTrue(any("admitted grammar: blr x" in error or "indirect" in error for error in errors), errors)


def called(body: str, extra: dict[str, str] | None = None, local: tuple = ("helper",), selection: dict | None = None,
           roles: dict | None = None, units: dict | None = None, landing: list[tuple] | None = None,
           bindings: dict[str, str] | None = None, visibility: dict[str, str] | None = None,
           macho: bool = False) -> tuple[dict, list[str]]:
    """Stack-traffic record and errors of f under the call model (ELF, production front end); the `local`
    functions are STB_LOCAL definitions, every other one global. landing: (range start, range end, pad) labels
    of f, installed as the object's exception call-site records."""
    functions = [("f", body)] + list((extra or {}).items())
    binding = {name: "local" for name in local if name in dict(functions)} | (bindings or {})
    options = abi_options({"bindings": binding}, macho)
    functions = to_abi(functions, macho)
    obj = CHECKER.read_object(witness_object(functions, None, macho, visibility=visibility, **options))
    if landing:
        labels = Layout(functions, False).entries[0]["labels"]
        obj = obj._replace(landing={"f": [(labels[a], labels[b], labels[p]) for a, b, p in landing]})
    bodies = CHECKER.normalize_disassembly(disassembly(functions, macho), obj)
    model, unit = CHECKER.call_model(obj, bodies, selection=selection, helper_roles=roles, units=units)
    analysis = CHECKER.analysed(functions[0][0], bodies, CHECKER.GENERIC, set(), model, unit)
    return CHECKER.traffic_record(analysis), analysis.errors


def other_unit(copy: str, functions: dict[str, str], local: tuple = (),
               visibility: dict[str, str] | None = None) -> "CHECKER.Unit":
    """Another audited translation unit's copy (for link-selection records)."""
    listed = list(functions.items())
    obj = CHECKER.read_object(witness_object(listed, None, bindings={name: "local" for name in local},
                                             visibility=visibility))
    return CHECKER.make_unit(copy, obj, CHECKER.normalize_disassembly(disassembly(listed), obj))


HELPER_FRAME = "\tstp\tx29, x30, [sp, #-16]!\n\tmov\tx29, sp\n\tldp\tx29, x30, [sp], #16\n\tret\n"
HELPER_SPILL = "\tsub\tsp, sp, #16\n\tstr\tq0, [sp]\n\tldr\tq0, [sp]\n\tadd\tsp, sp, #16\n\tret\n"
HELPER_LOOP = ("\tsub\tsp, sp, #16\n\tmov\tx3, #4\n.Lh:\n\tstr\tq0, [sp]\n\tsub\tx3, x3, #1\n\tcbnz\tx3, .Lh\n"
               "\tadd\tsp, sp, #16\n\tret\n")
HELPER_ROLES = {"helper": {"source": "src/v2/kernels/cpu/gemm/Helper.h", "source_sha256": "0" * 64,
                           "loops": {"1": {"line": 12, "role": "block", "variable": "b", "trip": "K"}}}}


def scope(*entries: tuple[str, str, bool]) -> list[dict]:
    """A process lookup scope: (object, role, defines the symbol), in lookup order."""
    return [{"object": name, "role": role, "sha256": f"{number:064x}", "defines": defines}
            for number, (name, role, defines) in enumerate(entries, start=1)]


# The process resolution of llaminar2 + libllaminar.so where only the DSO defines the helper: its copy runs.
PROCESS = {"binding": "process", "resolution": {"copy_object": "libllaminar.so", "scope": scope(
    ("llaminar2", "executable", False), ("libllaminar.so", "dso", True))}}


def kernel(inner: str) -> str:
    """A caller frame of 48 bytes with one K loop (loop 1) around `inner`."""
    return ("\tstp\tx29, x30, [sp, #-48]!\n\tmov\tx29, sp\n\tmov\tx5, #4\n.Lk:\n" + inner +
            "\tsub\tx5, x5, #1\n\tcbnz\tx5, .Lk\n\tldp\tx29, x30, [sp], #48\n\tret\n")


def present(counts: dict) -> dict:
    return {kind: values for kind, values in counts.items() if values}


class CallModelTest(unittest.TestCase):
    """Call boundaries (call-model design v2, controls 1-12): helpers audited and charged at the call site,
    link-selected weak copies, memcpy accounting, exception edges and cold paths, the stack-protector canary."""

    def ok(self, *args, **kwargs) -> dict:
        record, errors = called(*args, **kwargs)
        self.assertEqual(errors, [], args[0])
        return record

    def fails(self, fragment: str, *args, **kwargs) -> None:
        errors = called(*args, **kwargs)[1]
        self.assertTrue(any(fragment in error for error in errors), (fragment, errors))

    def test_clobbered_registers_after_a_call(self) -> None:
        # Control 1. x9 keeps the caller's stack address when the audited body leaves it alone (GCC -fipa-ra may
        # rely on that), and loses it when the body redefines it; an import (memcpy) resets x0-x18 by the ABI.
        use = "\tadd\tx9, sp, #16\n\tbl\thelper\n\tstr\tq0, [x9]\n"
        self.assertEqual(present(self.ok(kernel(use), {"helper": "\tret\n"})["counts"])["fp"], {"1": 1})
        self.assertNotIn("fp", present(self.ok(kernel(use), {"helper": "\tmov\tx9, #0\n\tret\n"})["counts"]))
        laundered = "\tfmov\tx9, d0\n\tbl\thelper\n\tstr\tx9, [sp, #16]\n"
        counts = present(self.ok(kernel(laundered), {"helper": "\tmov\tx9, #0\n\tret\n"})["counts"])
        self.assertEqual((counts.get("fp"), counts["scalar"]), (None, {"1": 1}))
        imported = "\tadd\tx9, sp, #16\n\tmov\tx2, #8\n\tbl\tmemcpy\n\tstr\tq0, [x9]\n"
        self.assertNotIn("fp", present(self.ok(kernel(imported))["counts"]))
        # x0 written by the helper is a new value; a helper writing v0 or v16 needs no save.
        self.ok(kernel("\tadd\tx0, sp, #16\n\tbl\thelper\n\tstr\tq0, [x0]\n"), {"helper": "\tmovi\tv0.4s, #0\n\tret\n"})

    def test_only_the_low_half_of_v8_is_preserved_by_proof(self) -> None:
        # Control 2: a helper may write v8's upper half; writing its low half without a save fails.
        self.ok(kernel("\tbl\thelper\n"), {"helper": "\tins\tv8.d[1], x0\n\tret\n"})
        self.fails("callee-saved v8", kernel("\tbl\thelper\n"), {"helper": "\tfmov\td8, d0\n\tret\n"})
        self.fails("callee-saved x19", kernel("\tbl\thelper\n"), {"helper": "\tmov\tx19, #1\n\tret\n"})
        self.fails("sp not restored", kernel("\tbl\thelper\n"), {"helper": "\tsub\tsp, sp, #16\n\tret\n"})

    def test_nzcv_is_undefined_after_a_call(self) -> None:
        # Control 3.
        self.fails("NZCV", kernel("\tcmp\tx5, #3\n\tbl\thelper\n\tcsel\tx6, x6, x7, eq\n"), {"helper": "\tret\n"})
        self.ok(kernel("\tbl\thelper\n\tcmp\tx5, #3\n\tcsel\tx6, x6, x7, eq\n"), {"helper": "\tret\n"})

    def test_helper_frame_is_charged_at_the_call_site_loop(self) -> None:
        # Control 4: the helper's stp/ldp x29, x30 run on every call: call_frame at loop 1, never exempt.
        counts = present(self.ok(kernel("\tbl\thelper\n"), {"helper": HELPER_FRAME})["counts"])
        self.assertEqual(counts, {"call_frame": {"1": 2}, "int_call": {"1": 1}})
        outside = present(self.ok("\tbl\thelper\n\tret\n", {"helper": HELPER_FRAME})["counts"])
        self.assertEqual(outside, {"call_frame": {"0": 2}, "int_call": {"0": 1}})

    def test_helper_spills_are_charged_at_the_call_site_loop(self) -> None:
        # Control 5.
        counts = present(self.ok(kernel("\tbl\thelper\n"), {"helper": HELPER_SPILL})["counts"])
        self.assertEqual((counts["fp"], counts["int_call"]), ({"1": 2}, {"1": 1}))

    def test_helper_loops_need_a_role_entry_and_nest_under_the_call_site(self) -> None:
        # Control 6.
        self.fails("no role entry", kernel("\tbl\thelper\n"), {"helper": HELPER_LOOP})
        record = self.ok(kernel("\tbl\thelper\n"), {"helper": HELPER_LOOP}, roles=HELPER_ROLES)
        helper = [loop for loop in record["loops"] if "helper" in loop]
        self.assertEqual(len(helper), 1)
        self.assertEqual({key: helper[0][key] for key in ("id", "parent", "depth", "helper", "helper_loop")},
                         {"id": 2, "parent": 1, "depth": 2, "helper": "helper", "helper_loop": 1})
        self.assertEqual(helper[0]["helper_role"], HELPER_ROLES["helper"]["loops"]["1"])
        self.assertEqual(present(record["counts"])["fp"], {"2": 1})

    def test_weak_helpers_need_a_link_selection_record(self) -> None:
        # Control 7: a weak or global definition executes the link-selected copy.
        for binding in ("weak", "global"):
            with self.subTest(binding=binding):
                self.fails("link-selection record", kernel("\tbl\thelper\n"), {"helper": "\tret\n"}, local=(),
                           bindings={"helper": binding})
        other = other_unit("tuB", {"helper": HELPER_SPILL})
        record = self.ok(kernel("\tbl\thelper\n"), {"helper": "\tret\n"}, local=(), bindings={"helper": "weak"},
                         selection={"helper": {"copy": "tuB", **PROCESS}}, units={"tuB": other})
        self.assertEqual(present(record["counts"])["fp"], {"1": 2})  # tuB's copy, not this unit's.
        self.fails("binding 'interposable'", kernel("\tbl\thelper\n"), {"helper": "\tret\n"}, local=(),
                   bindings={"helper": "weak"}, selection={"helper": {"copy": "tuB", "binding": "interposable"}},
                   units={"tuB": other})
        self.fails("not audited", kernel("\tbl\thelper\n"), {"helper": "\tret\n"}, local=(),
                   bindings={"helper": "weak"}, selection={"helper": {"copy": "tuC", **PROCESS}})

    def test_link_selection_records_bind_the_process_resolution(self) -> None:
        # C-b: a PLT call inside a DSO is preemptible; the executed copy is the process's first definition.
        other = other_unit("tuB", {"helper": HELPER_SPILL})
        def outcome(record: dict, caller: dict | None = None, copy: "CHECKER.Unit" = other) -> list[str]:
            return called(kernel("\tbl\thelper\n"), {"helper": "\tret\n"}, local=(), bindings={"helper": "weak"},
                          selection={"helper": {"copy": "tuB", **record}}, units={"tuB": copy},
                          visibility=caller)[1]
        self.assertEqual(outcome(PROCESS), [])
        # The executable defines it first and its audited copy is the one named: that copy executes, so it passes.
        executable = {"binding": "process", "resolution": {"copy_object": "llaminar2", "scope": scope(
            ("llaminar2", "executable", True), ("libllaminar.so", "dso", True))}}
        self.assertEqual(outcome(executable), [])
        interposed = {"binding": "process", "resolution": {"copy_object": "libllaminar.so", "scope": scope(
            ("llaminar2", "executable", True), ("libllaminar.so", "dso", True))}}
        cases = {
            "PLT binding is preemptible": {"binding": "plt"},
            "interposition": interposed,  # The executable's own instance interposes on the named DSO copy.
            "selects 'libllaminar.so', not": {"binding": "process", "resolution": {  # Named the exe; DSO runs.
                "copy_object": "llaminar2", "scope": scope(("llaminar2", "executable", False),
                                                           ("libllaminar.so", "dso", True))}},
            "start with the executable": {"binding": "process", "resolution": {
                "copy_object": "libllaminar.so", "scope": scope(("libllaminar.so", "dso", True))}},
            "malformed process resolution": {"binding": "process", "resolution": {
                "copy_object": "libllaminar.so", "scope": [{"object": "llaminar2", "role": "executable"}]}},
            "without a non-preemptibility proof": {"binding": "direct"},
            "visibility the objects do not record": {"binding": "direct", "nonpreemptible": "hidden"},
            "without the link evidence digest": {"binding": "direct", "nonpreemptible": "Bsymbolic"},
        }
        for fragment, record in cases.items():
            with self.subTest(fragment=fragment):
                errors = outcome(record)
                self.assertTrue(any(fragment in error for error in errors), (fragment, errors))
        # Non-preemptible: hidden/protected recorded in both the caller's and the copy's symbol tables, or a
        # link-option proof bound by its digest.
        for proof in ("hidden", "protected"):
            with self.subTest(proof=proof):
                hidden = other_unit("tuB", {"helper": HELPER_SPILL}, visibility={"helper": proof})
                self.assertEqual(outcome({"binding": "direct", "nonpreemptible": proof}, {"helper": proof}, hidden), [])
                errors = outcome({"binding": "direct", "nonpreemptible": proof}, None, hidden)
                self.assertTrue(any("do not record" in error for error in errors), errors)
        self.assertEqual(outcome({"binding": "direct", "nonpreemptible": "dynamic-list", "link_evidence": "a" * 64}), [])

    def test_helper_return_facts_keep_caller_saved_classification(self) -> None:
        # D1: an audited helper that leaves a caller-saved register alone returns its facts exactly; an import
        # resets it (AAPCS). FP/SIMD registers hold no tracked fact: their stack transfers always count.
        fp_data = "\tfmov\tx9, d0\n\tbl\t{}\n\tstr\tx9, [sp, #16]\n"
        kept = present(self.ok(kernel(fp_data.format("helper")), {"helper": "\tret\n"})["counts"])
        # Still FP data: the GPR store is scalar traffic and also an FP spill.
        self.assertEqual((kept["fp"], kept["scalar"]), ({"1": 1}, {"1": 1}))
        stack_data = "\tldr\tx9, [sp, #24]\n\tbl\thelper\n\tfmov\td1, x9\n"
        reload = present(self.ok(kernel(stack_data), {"helper": "\tret\n"})["counts"])
        self.assertEqual((reload["fp"], reload["scalar"]), ({"1": 1}, {"1": 1}))  # A laundered FP reload.
        copied = kernel("\tfmov\tx9, d0\n\tadd\tx0, sp, #32\n\tmov\tx2, #8\n\tbl\tmemcpy\n"
                        "\tstr\tx9, [sp, #16]\n")
        reset = present(self.ok(copied)["counts"])
        self.assertEqual((reset.get("fp"), reset["scalar"]), (None, {"1": 1}))  # Import: AAPCS reset.
        vector = present(self.ok(kernel("\tbl\thelper\n\tstr\tq16, [sp, #16]\n\tstr\tq3, [sp, #32]\n"),
                                 {"helper": "\tret\n"})["counts"])
        self.assertEqual(vector["fp"], {"1": 2})

    def test_mach_o_calls_fail_closed(self) -> None:
        # C-c: no Mach-O exception tables are parsed, so every call boundary there fails.
        for extra, body in (({"helper": "\tret\n"}, "\tbl\thelper\n"), ({}, "\tadd\tx0, sp, #16\n\tmov\tx2, #8\n"
                                                                              "\tbl\tmemcpy\n")):
            with self.subTest(body=body):
                errors = called(kernel(body), extra, macho=True)[1]
                self.assertTrue(any("fails closed on Mach-O" in error for error in errors), errors)

    def test_stack_address_arguments(self) -> None:
        # Control 8: exact accesses through a stack-address argument count as caller traffic at the call site.
        exact = present(self.ok(kernel("\tadd\tx0, sp, #16\n\tbl\thelper\n"),
                                {"helper": "\tstr\txzr, [x0]\n\tret\n"})["counts"])
        self.assertEqual(exact["scalar"], {"1": 1})
        self.fails("cannot be tracked", kernel("\tadd\tx0, sp, #16\n\tbl\thelper\n"),
                   {"helper": "\tstr\tx0, [x1]\n\tret\n"})
        self.fails("into the caller frame", kernel("\tadd\tx0, sp, #16\n\tbl\thelper\n"),
                   {"helper": "\tstr\tx0, [x0, #8]\n\tret\n"})
        self.fails("outside the caller frame", kernel("\tadd\tx0, sp, #16\n\tbl\thelper\n"),
                   {"helper": "\tstr\txzr, [x0, #64]\n\tret\n"})
        self.fails("live pointer, save or canary slot",
                   kernel("\tadd\tx8, sp, #32\n\tstr\tx8, [sp, #16]\n\tadd\tx0, sp, #16\n\tbl\thelper\n"),
                   {"helper": "\tstr\txzr, [x0]\n\tret\n"})
        self.fails("unknown offset", kernel("\tadd\tx0, sp, #16\n\tbl\thelper\n"),
                   {"helper": "\tstr\txzr, [x0, x3]\n\tret\n"})

    def test_memcpy_sizes_and_stack_sides(self) -> None:
        # Control 9.
        copy = "\tadd\tx0, sp, #16\n\tmov\tx1, x6\n\tmov\tx2, #24\n\tbl\tmemcpy\n"
        counts = present(self.ok(kernel(copy))["counts"])
        self.assertEqual({kind: counts[kind] for kind in ("ext_call", "ext_bytes", "ext_accesses")},
                         {"ext_call": {"1": 1}, "ext_bytes": {"1": 24}, "ext_accesses": {"1": 3}})
        both = present(self.ok(kernel("\tadd\tx0, sp, #16\n\tadd\tx1, sp, #32\n\tmov\tx2, #9\n\tbl\tmemcpy\n"))["counts"])
        self.assertEqual((both["ext_bytes"], both["ext_accesses"]), ({"1": 18}, {"1": 4}))
        self.fails("not one constant", kernel(copy.replace("mov\tx2, #24", "mov\tx2, x7")))
        joined = ("\tcbz\tx9, .La\n\tmov\tx2, #24\n\tb\t.Lc\n.La:\n\tmov\tx2, #16\n.Lc:\n\tadd\tx0, sp, #16\n"
                  "\tbl\tmemcpy\n")
        self.fails("not one constant", kernel(joined))
        self.fails("live pointer, save or canary slot",
                   kernel("\tadd\tx8, sp, #32\n\tstr\tx8, [sp, #16]\n" + copy.replace("#24", "#8")))
        checked = copy.replace("bl\tmemcpy", "mov\tx3, #{}\n\tbl\t__memcpy_chk")
        self.ok(kernel(checked.format(32)))
        self.fails("bound", kernel(checked.format(16)))

    COMPACT = ("_ZN9llaminar23cpu16native_quantised11multi_scale4rowsILi10ELi4ELm1ELNS2_9ReductionE0ENS1_28CPUNative"
               "QuantisedWeightViewENS2_17NEONMultiScaleOpsILNS2_11NEONByteDotE0EEEEEvRKT3_RKSt5arrayIPKNS_9Q8_1BlockEXT1"
               "_EERKSC_IPfXT1_EEiiibiii.isra.0")
    KERNELS = "src/v2/kernels/cpu/gemm/CPUNativeQuantisedMultiScaleKernels.h"

    def bounded(self, line: int, source: str | None = None, name: str | None = None) -> tuple[dict, list[str]]:
        """A variable-size memcpy (x2 unknown) to a stack slot inside the K loop of a compact program symbol, its
        call located at `line` of the candidate tree's own compact kernel source (M-6 controls)."""
        name = name or self.COMPACT
        body = kernel("\tadd\tx0, sp, #16\n\tmov\tx1, x6\n\tmov\tx2, x7\n\tbl\tmemcpy\n")
        options = abi_options({}, False)
        obj = CHECKER.read_object(witness_object([(name, body)], **options))
        bodies = CHECKER.normalize_disassembly(disassembly([(name, body)]), obj)
        noreturn = CHECKER.noreturn_symbols(bodies, CHECKER.GENERIC, CHECKER.trusted_imports(obj), obj.prefix)
        model, unit = CHECKER.call_model(obj, bodies)
        text = source if source is not None else (REPO / self.KERNELS).read_text()
        where = [f"{self.KERNELS}:1"] * 16
        where[6] = f"{self.KERNELS}:{line}"  # stp, mov, mov, add, mov, mov, bl memcpy
        CHECKER.install_bounded_context({name: where}, {self.KERNELS: text})
        try:
            analysis = CHECKER.analysed(name, bodies, CHECKER.GENERIC, noreturn, model, unit)
        finally:
            CHECKER.install_bounded_context({}, {})
        return CHECKER.traffic_record(analysis), analysis.errors

    def test_bounded_memcpy_admission(self) -> None:
        # M-6 (reviewer 91): the admitted tail (candidate line 1110) is charged at sizeof(F) = 4 x Lanes = 16 bytes
        # with its lower bound recorded; any other line, another source version, or a non-compact function FAILS.
        record, errors = self.bounded(1110)
        self.assertEqual(errors, [])
        counts = present(record["counts"])
        self.assertEqual({kind: counts[kind] for kind in ("ext_call", "ext_bytes", "ext_accesses")},
                         {"ext_call": {"1": 1}, "ext_bytes": {"1": 16}, "ext_accesses": {"1": 2}})
        [site] = record["bounded"]
        self.assertEqual((site["lower_bytes"], site["upper_bytes"], site["loop"], site["placement"], site["sides"]),
                         (4, 16, 1, "column-tail", 1))
        self.assertEqual(site["text"], "std::memcpy(&acc[row],output[row]+column,(columns-column)*sizeof(float));")
        for line, fragment in ((1352, "not an admitted line"), (1109, "not an admitted line")):
            with self.subTest(line=line):
                self.assertTrue(any(fragment in error for error in self.bounded(line)[1]), self.bounded(line)[1])
        changed = (REPO / self.KERNELS).read_text().replace("Lanes <= columns", "Lanes<=columns", 1)
        self.assertTrue(any("not an admitted line of this source version" in error
                            for error in self.bounded(1110, changed)[1]))
        self.assertTrue(any("no program key" in error for error in self.bounded(1110, name="f")[1]))

    def test_cold_paths(self) -> None:
        # Control 10: a throw sequence is admitted only when every path ends in a never-returning call.
        throw = "\tcbz\tx9, .Lok\n\tmov\tx0, #16\n\tbl\t__cxa_allocate_exception\n\tbl\tctor\n\tbl\t__cxa_throw\n.Lok:\n"
        self.ok(kernel(throw))
        self.fails("outside a never-returning sequence", kernel("\tbl\t__cxa_allocate_exception\n"))
        record = self.ok(kernel("\tcbz\tx9, .Lok\n\tstr\tx1, [sp, #16]\n\tbl\tabort\n.Lok:\n"))
        self.assertEqual((record.get("cold"), present(record["counts"])), ({"scalar": 1}, {}))
        self.fails("unsupported call boundary (undefined", kernel("\tbl\tsome_import\n"), bindings={"some_import": "weak"})
        self.fails("tail call", "\tb\thelper\n", {"helper": "\tret\n"})

    def test_landing_pads(self) -> None:
        caught = kernel(".Lb:\n\tbl\thelper\n.Le:\n") + ".Lpad:\n\tbl\t_Unwind_Resume\n"
        self.ok(caught, {"helper": "\tret\n"}, landing=[(".Lb", ".Le", ".Lpad")])
        # Without its exception edge the pad is unreachable code.
        self.fails("unreachable", caught, {"helper": "\tret\n"})
        reenter = kernel(".Lb:\n\tbl\thelper\n.Le:\n") + ".Lpad:\n\tb\t.Lk\n"
        self.fails("landing pad path", reenter, {"helper": "\tret\n"}, landing=[(".Lb", ".Le", ".Lpad")])
        returning = kernel(".Lb:\n\tbl\thelper\n.Le:\n") + ".Lpad:\n\tret\n"
        self.fails("landing pad path", returning, {"helper": "\tret\n"}, landing=[(".Lb", ".Le", ".Lpad")])
        self.fails("unreachable", caught, {"helper": "\tret\n"}, landing=[(".Lb", ".Lb", ".Lpad")])

    def test_canary_slot_is_call_frame_and_protected_arrays_count(self) -> None:
        # Control 11.
        guard = ("\tadrp\tx1, 0x0 <f> !R_AARCH64_ADR_GOT_PAGE:__stack_chk_guard\n"
                 "\tldr\tx1, [x1] !R_AARCH64_LD64_GOT_LO12_NC:__stack_chk_guard\n\tldr\tx2, [x1]\n"
                 "\tstr\tx2, [sp, #40]\n")
        check = "\tldr\tx3, [sp, #40]\n\tldr\tx4, [x1]\n\tsubs\tx3, x3, x4\n\tb.ne\t.Lfail\n"
        body = ("\tstp\tx29, x30, [sp, #-48]!\n\tmov\tx29, sp\n" + guard + "\tmov\tx5, #4\n.Lk:\n\tstr\tw3, [sp, #16]\n"
                "\tsub\tx5, x5, #1\n\tcbnz\tx5, .Lk\n" + check + "\tldp\tx29, x30, [sp], #48\n\tret\n"
                ".Lfail:\n\tbl\t__stack_chk_fail\n")
        counts = present(self.ok(body)["counts"])
        self.assertEqual(counts, {"scalar": {"1": 1}, "call_frame": {"0": 2}})
        # An ordinary value stored to the slot is no canary: scalar traffic.
        plain = present(self.ok(body.replace("\tstr\tx2, [sp, #40]", "\tstr\tx6, [sp, #40]"))["counts"])
        self.assertEqual(plain, {"scalar": {"0": 2, "1": 1}})

    def test_recursion_and_depth_fail(self) -> None:
        # Control 12.
        recursive = "\tstp\tx29, x30, [sp, #-16]!\n\tbl\thelper\n\tldp\tx29, x30, [sp], #16\n\tret\n"
        self.fails("recursion", kernel("\tbl\thelper\n"), {"helper": recursive})
        chain = {f"chain{n}": f"\tstp\tx29, x30, [sp, #-16]!\n\tbl\tchain{n + 1}\n\tldp\tx29, x30, [sp], #16\n\tret\n"
                 for n in range(9)}
        chain["chain9"] = "\tret\n"
        self.fails("call depth", "\tbl\tchain0\n\tret\n", chain, local=tuple(chain))
        self.ok("\tbl\tchain2\n\tret\n", chain, local=tuple(chain))  # Depth 8 (chain2 ... chain9) is admitted.

    def test_new_call_classes_are_recorded(self) -> None:
        self.assertEqual(CHECKER.TRAFFIC[-5:], ("call_frame", "int_call", "ext_call", "ext_bytes", "ext_accesses"))


def eh_frame(lsda: int = 0x1B, relocate: bool = True, augmentation: bytes = b"zPLR", span: int = 16) \
        -> tuple[bytes, dict]:
    """A relocatable .eh_frame: one CIE and one FDE (pc_begin and LSDA relocated) for a `span`-byte function.
    Relocations: {field offset: (section index 3 / 5, offset)}; as_kinds() names them pc / lsda."""
    data = {b"P": b"\x9b" + bytes(4), b"L": bytes([lsda]), b"R": b"\x1b"}
    tail = b"".join(data.get(bytes([letter]), b"") for letter in augmentation[1:])
    body = bytes(4) + b"\x01" + augmentation + b"\0" + b"\x04\x78\x1e" + bytes([len(tail)]) + tail
    cie = len(body).to_bytes(4, "little") + body
    start = len(cie)
    fde = (start + 4).to_bytes(4, "little") + bytes(4) + span.to_bytes(4, "little") + b"\x04" + bytes(4)
    relocations = {start + 8: (3, 0), start + 17: (5, 8)} if relocate else {}
    return cie + len(fde).to_bytes(4, "little") + fde + bytes(4), relocations


class ExceptionTableTest(unittest.TestCase):
    """.eh_frame FDEs and LSDA call-site records are parsed from object bytes, fail closed on other encodings."""

    def test_fde_with_relocated_lsda(self) -> None:
        data, relocations = eh_frame()
        self.assertEqual(CHECKER.parse_eh_frame(data, relocations), [(3, 0, 16, 5, 8)])
        self.assertEqual(CHECKER.parse_eh_frame(*eh_frame(augmentation=b"zR")), [(3, 0, 16, None, 0)])
        for case, (data, relocations) in {"unrelocated": eh_frame(relocate=False), "lsda encoding": eh_frame(lsda=0),
                                          "augmentation": eh_frame(augmentation=b"zX")}.items():
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    CHECKER.parse_eh_frame(data, relocations)

    def test_lsda_call_sites(self) -> None:
        lsda = b"\xff\xff\x01\x08" + b"\x04\x04\x0c\x00" + b"\x08\x04\x00\x00"
        self.assertEqual(CHECKER.parse_lsda(lsda, 0, 16), [(4, 8, 12)])  # A record without a pad adds no edge.
        for case, data in {"lpstart": b"\x00" + lsda[1:], "encoding": lsda[:2] + b"\x0b" + lsda[3:],
                           "outside": lsda[:6] + b"\x40" + lsda[7:], "length": lsda[:3] + b"\x07" + lsda[4:]}.items():
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    CHECKER.parse_lsda(data, 0, 16)


def with_exception_tables(data: bytes, function: str, records: list[tuple[int, int, int]], relocate_lsda: bool = True,
                          lsda_encoding: int = 0x1B, span: int | None = None) -> bytes:
    """Add .gcc_except_table, .eh_frame and .rela.eh_frame to an ELF fixture: one zPLR CIE and one FDE for
    `function` (pc_begin relocated against its symbol, the LSDA pointer against a new .gcc_except_table section
    symbol) whose LSDA lists the call-site records (start, end, pad), function-relative byte offsets (pad 0: none).
    The object is rebuilt with the new sections and symbol; nothing else changes."""
    shoff, shnum, shstrndx = (int.from_bytes(data[0x28:0x30], "little"), int.from_bytes(data[0x3C:0x3E], "little"),
                              int.from_bytes(data[0x3E:0x40], "little"))
    headers = []
    for index in range(shnum):
        raw = data[shoff + 64 * index:shoff + 64 * (index + 1)]
        field = lambda at, size: int.from_bytes(raw[at:at + size], "little")
        headers.append({"name": field(0, 4), "type": field(4, 4), "flags": field(8, 8), "link": field(40, 4),
                        "info": field(44, 4), "entsize": field(56, 8),
                        "content": bytearray(data[field(24, 8):field(24, 8) + field(32, 8)])})
    symtab = next(index for index, header in enumerate(headers) if header["type"] == 2)
    strings = headers[headers[symtab]["link"]]["content"]
    entries = headers[symtab]["content"]
    symbol = next(entry for entry in range(1, len(entries) // 24)
                  if strings[int.from_bytes(entries[24 * entry:24 * entry + 4], "little"):].split(b"\0")[0]
                  == function.encode())
    size = int.from_bytes(entries[24 * symbol + 16:24 * symbol + 24], "little")
    table_index, frame_index = shnum, shnum + 1
    lsda_symbol = len(entries) // 24
    entries += bytes(4) + bytes([3, 0]) + table_index.to_bytes(2, "little") + bytes(16)  # STT_SECTION, local.
    sites = b"".join(bytes([start, end - start, pad, 0]) for start, end, pad in records)
    lsda = b"\xff\xff\x01" + bytes([len(sites)]) + sites
    frame, relocations = eh_frame(lsda=lsda_encoding, span=span if span is not None else size)
    rela = bytearray()
    for offset, (section, _) in sorted(relocations.items()):
        if section == 5 and not relocate_lsda:
            continue
        index = symbol if section == 3 else lsda_symbol
        rela += offset.to_bytes(8, "little") + ((index << 32) | 261).to_bytes(8, "little") + bytes(8)
    shstr = headers[shstrndx]["content"]
    def name(text: str) -> int:
        shstr.extend(text.encode() + b"\0")
        return len(shstr) - len(text) - 1
    headers += [{"name": name(".gcc_except_table"), "type": 1, "flags": 2, "link": 0, "info": 0, "entsize": 0,
                 "content": bytearray(lsda)},
                {"name": name(".eh_frame"), "type": 1, "flags": 2, "link": 0, "info": 0, "entsize": 0,
                 "content": bytearray(frame)},
                {"name": name(".rela.eh_frame"), "type": 4, "flags": 0x40, "link": symtab, "info": frame_index,
                 "entsize": 24, "content": rela}]
    body, offsets = bytearray(), []
    for header in headers:
        offsets.append(64 + len(body))
        body += header["content"]
    out = bytearray(data[:64])
    out[0x28:0x30] = (64 + len(body)).to_bytes(8, "little")
    out[0x3C:0x3E] = len(headers).to_bytes(2, "little")
    out += body
    for header, offset in zip(headers, offsets):
        out += header["name"].to_bytes(4, "little") + header["type"].to_bytes(4, "little") + \
            header["flags"].to_bytes(8, "little") + bytes(8) + offset.to_bytes(8, "little") + \
            len(header["content"]).to_bytes(8, "little") + header["link"].to_bytes(4, "little") + \
            header["info"].to_bytes(4, "little") + bytes(8) + header["entsize"].to_bytes(8, "little")
    return bytes(out)


def unwound(body: str, records: list[tuple[str, str, str]], extra: dict[str, str] | None = None, **options) \
        -> tuple[dict, list[str]]:
    """f's stack-traffic record and errors under the call model, its exception edges read from object bytes."""
    functions = [("f", body)] + list((extra or {}).items())
    labels = Layout(functions, False).entries[0]["labels"]
    data = witness_object(functions, None, bindings={name: "local" for name in (extra or {})})
    data = with_exception_tables(data, "f", [(labels[a], labels[b], labels[p] if p else 0) for a, b, p in records],
                                 **options)
    obj = CHECKER.read_object(data)
    bodies = CHECKER.normalize_disassembly(disassembly(functions), obj)
    model, unit = CHECKER.call_model(obj, bodies)
    analysis = CHECKER.analysed("f", bodies, CHECKER.GENERIC, set(), model, unit)
    return CHECKER.traffic_record(analysis), analysis.errors


class ObjectExceptionEdgeTest(unittest.TestCase):
    """Exception edges come from the object's own .eh_frame FDE and .gcc_except_table LSDA bytes."""

    CAUGHT = kernel(".Lb:\n\tbl\thelper\n.Le:\n") + ".Lpad:\n\tstr\tx1, [sp, #16]\n\tbl\t_Unwind_Resume\n"

    def test_landing_pad_from_object_bytes_is_cold(self) -> None:
        record, errors = unwound(self.CAUGHT, [(".Lb", ".Le", ".Lpad")], {"helper": "\tret\n"})
        self.assertEqual(errors, [])
        self.assertEqual((record.get("cold"), present(record["counts"])), ({"scalar": 1}, {"int_call": {"1": 1}}))

    def test_pad_paths_and_unlisted_pads_fail(self) -> None:
        # The same code without a call-site record for the call: the pad is unreachable.
        errors = unwound(self.CAUGHT, [(".Lb", ".Lb", None)], {"helper": "\tret\n"})[1]
        self.assertTrue(any("unreachable" in error for error in errors), errors)
        back = kernel(".Lb:\n\tbl\thelper\n.Le:\n") + ".Lpad:\n\tb\t.Lk\n"
        errors = unwound(back, [(".Lb", ".Le", ".Lpad")], {"helper": "\tret\n"})[1]
        self.assertTrue(any("landing pad path" in error for error in errors), errors)
        # A pad reached only after a returning cleanup call: the import must lie on a cold path too.
        cleanup = kernel(".Lb:\n\tbl\thelper\n.Le:\n") + ".Lpad:\n\tbl\tcleanup\n\tret\n"
        errors = unwound(cleanup, [(".Lb", ".Le", ".Lpad")], {"helper": "\tret\n"})[1]
        self.assertTrue(any("landing pad path" in error for error in errors), errors)

    def test_malformed_tables_fail_before_analysis(self) -> None:
        cases = {"unrelocated LSDA pointer": {"relocate_lsda": False}, "LSDA encoding": {"lsda_encoding": 0x03},
                 "FDE start is no function": {"span": 4}}
        for case, options in cases.items():
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    unwound(self.CAUGHT, [(".Lb", ".Le", ".Lpad")], {"helper": "\tret\n"}, **options)
        with self.assertRaisesRegex(ValueError, "outside its function"):
            unwound(self.CAUGHT, [(".Lb", ".Le", ".Lpad")], {"helper": "\tret\n"}, span=8)


class CallProvenanceTest(unittest.TestCase):
    """Provenance and binding negatives at call boundaries."""

    def errors(self, *args, **kwargs) -> list[str]:
        return called(*args, **kwargs)[1]

    def test_path_dependent_stack_arguments_fail(self) -> None:
        body = kernel("\tmov\tx0, x6\n\tcbz\tx9, .La\n\tadd\tx0, sp, #16\n.La:\n\tbl\thelper\n")
        self.assertTrue(any("on only some paths at a call" in e for e in self.errors(body, {"helper": "\tret\n"})))
        memcpy = kernel("\tmov\tx0, x6\n\tcbz\tx9, .La\n\tadd\tx0, sp, #16\n.La:\n\tmov\tx2, #8\n\tbl\tmemcpy\n")
        self.assertTrue(any("on only some paths" in e for e in self.errors(memcpy)))

    def test_callee_saved_stack_addresses_reach_the_helper(self) -> None:
        # x19 holds a caller stack address: a helper store through it is caller-frame FP traffic at the call site.
        record, errors = called(kernel("\tadd\tx19, sp, #16\n\tbl\thelper\n"), {"helper": "\tstr\tq0, [x19]\n\tret\n"})
        self.assertEqual((errors, present(record["counts"])["fp"]), ([], {"1": 1}))
        joined = kernel("\tmov\tx19, x6\n\tcbz\tx9, .La\n\tadd\tx19, sp, #16\n.La:\n\tbl\thelper\n")
        self.assertTrue(any("only some paths" in e for e in self.errors(joined, {"helper": "\tstr\tq0, [x19]\n\tret\n"})))

    def test_returned_own_frame_addresses_stay_stack_addresses(self) -> None:
        helper = "\tsub\tsp, sp, #16\n\tmov\tx0, sp\n\tadd\tsp, sp, #16\n\tret\n"
        # The dead address is a stack address at an unknown offset: an FP spill, never laundered to non-stack ...
        record, errors = called("\tbl\thelper\n\tstr\tq0, [x0]\n\tret\n", {"helper": helper})
        self.assertEqual((errors, present(record["counts"])["fp"]), ([], {"0": 1}))
        # ... and with the caller's frame-pointer slot live, an untrackable store.
        errors = self.errors(kernel("\tbl\thelper\n\tstr\tq0, [x0]\n"), {"helper": helper})
        self.assertTrue(any("unknown offset while pointer or save slots are live" in e for e in errors), errors)

    def test_import_bindings_and_spellings(self) -> None:
        copy = kernel("\tadd\tx0, sp, #16\n\tmov\tx2, #8\n\tbl\tmemcpy\n")
        self.assertTrue(any("undefined, weak or unbound" in e
                            for e in self.errors(copy, bindings={"memcpy": "weak"})))
        self.assertTrue(any("link-selection record" in e
                            for e in self.errors(copy, {"memcpy": "\tret\n"}, local=())))
        local = present(called(copy, {"memcpy": "\tret\n"}, local=("memcpy",))[0]["counts"])
        self.assertEqual((local.get("ext_call"), local["int_call"]), (None, {"1": 1}))  # Its own audited body.
        for name in ("__Unwind_Resume", "_abort"):
            with self.subTest(name=name):
                self.assertTrue(any("outside a never-returning sequence" in e
                                    for e in self.errors(kernel(f"\tcbz\tx9, .Lok\n\tbl\t{name}\n.Lok:\n"))))
        weak_abort = kernel("\tcbz\tx9, .Lok\n\tbl\tabort\n.Lok:\n")
        self.assertTrue(any("undefined, weak or unbound" in e for e in self.errors(weak_abort, bindings={"abort": "weak"})))
        checked = kernel("\tadd\tx0, sp, #16\n\tmov\tx2, #8\n\tmov\tx3, x7\n\tbl\t__memcpy_chk\n")
        self.assertTrue(any("bound" in e for e in self.errors(checked)))


def compact_name(ordered: int = 0, codebook: int = 16, lanes: int = 4, rows: int = 1, tree: int = 1, dot: int = 0,
                 clone: str = ".isra.0") -> str:
    """The production rowsProgram<...>() symbol shape."""
    return (f"_ZN9llaminar23cpu16native_quantised11multi_scale11rowsProgramILb{ordered}ELi{codebook}ELi{lanes}ELm{rows}"
            f"ELNS2_9ReductionE{tree}ENS1_28CPUNativeQuantisedWeightViewENS2_17NEONMultiScaleOpsILNS2_11NEONByteDotE"
            f"{dot}EEEEEvRKT4_RKSt5arrayIPKNS_9Q8_1BlockEXT2_EERKSC_IPfXT2_EEiiibiii{clone}")


class CompactProgramKeyTest(unittest.TestCase):
    """Compact programs are keyed mechanically from their exact symbol; per TU one copy per key, binding kept."""

    def test_keys_and_clones(self) -> None:
        self.assertEqual(CHECKER.compact_program_key(compact_name()), "compact:16:4:1:1:0:0")
        self.assertEqual(CHECKER.compact_program_key(compact_name(1, 7, 4, 3, 0, 2, "")), "compact:7:4:3:0:1:2")
        self.assertEqual(CHECKER.compact_program_key("_" + compact_name(), "_"), "compact:16:4:1:1:0:0")
        for bad in (compact_name(rows=5), compact_name(dot=3), compact_name(clone=".cold"),
                    compact_name().replace("WeightView", "WeightViewX"), compact_name() + "x"):
            with self.subTest(bad=bad[-40:]):
                self.assertIsNone(CHECKER.compact_program_key(bad))

    def test_develop_policy_free_program(self) -> None:
        # The shape g++-14 emits for develop's rows<10, 4, 1, OrderedPartitions, View, NEONMultiScaleOps<ASIMD>>.
        develop = ("_ZN9llaminar23cpu16native_quantised11multi_scale4rowsILi10ELi4ELm1ELNS2_9ReductionE1ENS1_28CPUNativeQ"
                   "uantisedWeightViewENS2_17NEONMultiScaleOpsILNS2_11NEONByteDotE0EEEEEvRKT3_RKSt5arrayIPKNS_9Q8_1Blo"
                   "ckEXT1_EERKSC_IPfXT1_EEiiibiii")
        self.assertEqual(CHECKER.compact_program_key(develop), "compact:10:4:1:1:-:0")
        self.assertEqual(CHECKER.compact_program_key(develop + ".isra.0"), "compact:10:4:1:1:-:0")
        self.assertIsNone(CHECKER.compact_program_key(develop.replace("T3_", "T4_")))
        odd = [(develop.replace("ELi4E", "ELi4ELi1E"), "\tret\n")]
        with self.assertRaisesRegex(ValueError, "unrecognized compact program"):
            CHECKER.compact_programs(CHECKER.read_object(witness_object(odd, None)))

    def test_per_object_inventory_fails_closed(self) -> None:
        body = "\tret\n"
        local = {compact_name(): "local", compact_name(rows=2): "weak"}
        functions = [(name, body) for name in local]
        found = CHECKER.compact_programs(CHECKER.read_object(witness_object(functions, None, bindings=local)))
        self.assertEqual(found, {"compact:16:4:1:1:0:0": (compact_name(), "local"),
                                 "compact:16:4:2:1:0:0": (compact_name(rows=2), "weak")})
        twice = [(compact_name(), body), (compact_name(clone=".isra.1"), body)]
        with self.assertRaisesRegex(ValueError, "defined twice"):
            CHECKER.compact_programs(CHECKER.read_object(witness_object(twice, None)))
        odd = [(compact_name(dot=3), body)]
        with self.assertRaisesRegex(ValueError, "unrecognized compact program"):
            CHECKER.compact_programs(CHECKER.read_object(witness_object(odd, None)))


KERNEL_FILE = "src/v2/kernels/cpu/gemm/CPUNativeQuantisedMultiScaleKernels.h"
NEON_HEADER = "/usr/lib/gcc/aarch64-linux-gnu/14/include/arm_neon.h"


def located(functions: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
    """(object functions, compiler assembly) from bodies whose lines may end in `@@N` (KERNEL_FILE line N) or
    `@@aN` (arm_neon.h line N): the marker becomes a `.loc` before that instruction in the assembly only."""
    clean, annotated = [], []
    for name, body in functions:
        plain, marked = [], []
        for line in body.splitlines():
            text, _, marker = line.partition("@@")
            if marker:
                file, number = (2, marker[1:]) if marker.startswith("a") else (1, marker)
                marked.append(f"\t.loc {file} {number.strip()} 0")
            plain.append(text.rstrip())
            marked.append(text.rstrip())
        clean.append((name, "\n".join(plain) + "\n"))
        annotated.append((name, "\n".join(marked) + "\n"))
    text = assembly(annotated).replace("\t.text\n", f'\t.text\n\t.file 1 "{KERNEL_FILE}"\n\t.file 2 "{NEON_HEADER}"\n', 1)
    return clean, text


class CompactOrderedAddTest(unittest.TestCase):
    """S2-4 (synthetic): ordered additions of compact programs per copy, bound to the assembler input; every
    compiler fadd (scalar or vector) proven in source order by global reaching definitions; exact counts."""

    SITE = "\tfadd\tv5.4s, v5.4s, v6.4s\n"
    SEED_SOURCE = "\n".join(["// kernel"] * 9 + ["                acc[row] = load<F>(output[row] + column);"]) + "\n"

    def evidence(self, bodies: dict, sources: dict | None = None) -> dict:
        functions, text = located(list(bodies.items()))
        return CHECKER.compact_ordered_add_evidence(witness_object(functions, None), disassembly(functions), text,
                                                    "KernelFactory", sources)

    def test_proven_sites_are_recorded_per_copy(self) -> None:
        ga, native = compact_name(ordered=1), compact_name(ordered=0, rows=2)
        body = "\tfadd\tv0.4s, v0.4s, v1.4s\n\tfadd\ts2, s0, s3\n\tfadd\ts4, s2, s5\n\tret\n"
        evidence = self.evidence({ga: body, native: "\tfadd\ts2, s3, s0\n\tret\n"})
        self.assertEqual(evidence, {"compact:16:4:1:1:1:0@KernelFactory": {"sites": [0], "scalar": [1, 2]}})
        CHECKER.require_compact_ordered_adds(evidence, {"compact:16:4:1:1:1:0@KernelFactory": {"sites": 1, "scalar": 2}})
        for expected in ({"compact:16:4:1:1:1:0@KernelFactory": {"sites": 1, "scalar": 1}}, {}):
            with self.assertRaisesRegex(ValueError, "differ from the committed"):
                CHECKER.require_compact_ordered_adds(evidence, expected)

    def test_loop_carried_chains_joins_and_copies(self) -> None:
        # S2-4a: acc = acc + x around the kb loop, from the exact +0.0 initialisation, proves itself.
        loop = "\tmovi\tv0.4s, #0\n" + self.SITE + ".Lk:\n\tldr\tq1, [x1], #16\n\t{}\n\tsubs\tx2, x2, #1\n\tb.ne\t.Lk\n\tret\n"
        ga = compact_name(ordered=1)
        self.assertEqual(self.evidence({ga: loop.format("fadd\ts0, s0, s1")})[
            "compact:16:4:1:1:1:0@KernelFactory"]["scalar"], [3])
        # An in-place `fadd v.4s` inside the inline-asm block is a template site, not a compiler fadd.
        self.assertEqual(self.evidence({ga: loop.format("fadd\tv0.4s, v0.4s, v1.4s")})
                         ["compact:16:4:1:1:1:0@KernelFactory"], {"sites": [1, 3], "scalar": []})
        # S2-4b: a compiler (non-template) vector fadd takes the same proof as a scalar one.
        vector = loop.format("fadd\tv2.4s, v0.4s, v1.4s\n\tmov\tv0.16b, v2.16b")
        self.assertEqual(self.evidence({ga: vector})["compact:16:4:1:1:1:0@KernelFactory"]["scalar"], [3])
        cases = {"commuted in the loop": loop.format("fadd\ts0, s1, s0"),
                 "commuted vector": loop.format("fadd\tv2.4s, v1.4s, v0.4s\n\tmov\tv0.16b, v2.16b"),
                 "join brings a non-chain value": "\tmovi\tv0.4s, #0\n" + self.SITE +
                     "\tcbz\tx3, .La\n\tldr\tq0, [x4]\n.La:\n\tfadd\ts2, s0, s1\n\tret\n",
                 "not exactly zero": loop.format("fadd\ts0, s0, s1").replace("movi\tv0.4s, #0", "movi\tv0.4s, #1"),
                 "incoming value": self.SITE + "\tfadd\ts2, s3, s4\n\tret\n",
                 "both on the chain": self.SITE + "\tfadd\ts2, s5, s5\n\tret\n",
                 "without an ordered add": "\tmovi\tv0.4s, #0\n\tfadd\ts2, s0, s3\n\tret\n"}
        for case, body in cases.items():
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, "commuted|unprovable|without an ordered add"):
                    self.evidence({ga: body})
        transparent = self.SITE + "\tmov\tv3.16b, v5.16b\n\tstr\tq3, [sp, #16]\n\tldr\tq4, [sp, #16]\n" \
                                  "\tfadd\ts2, s4, s1\n\tret\n"
        self.assertEqual(self.evidence({ga: transparent})["compact:16:4:1:1:1:0@KernelFactory"]["scalar"], [4])
        clobbered = transparent.replace("\tldr\tq4", "\tstr\tq1, [sp, #16]\n\tldr\tq4")
        with self.assertRaisesRegex(ValueError, "unprovable"):
            self.evidence({ga: clobbered})

    def test_contiguous_seeds_are_source_bound(self) -> None:
        ga = compact_name(ordered=1)
        seeded = self.SITE + "\tldr\tq0, [x5]@@10\n\tfadd\ts2, s0, s1\n\tret\n"
        self.assertEqual(self.evidence({ga: seeded}, {KERNEL_FILE: self.SEED_SOURCE})[
            "compact:16:4:1:1:1:0@KernelFactory"]["scalar"], [2])
        for case, sources in {"no source": None, "another line": {KERNEL_FILE: self.SEED_SOURCE.replace("output[row]",
                                                                                                         "bias[row]")}}.items():
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, "unprovable"):
                    self.evidence({ga: seeded}, sources)
        copied = self.SITE + "\tadd\tx0, sp, #16\n\tmov\tx2, #16\n\tbl\tmemcpy\n\tldr\tq0, [sp, #16]\n" \
                             "\tfadd\ts2, s0, s1\n\tret\n"
        self.assertEqual(self.evidence({ga: copied})["compact:16:4:1:1:1:0@KernelFactory"]["scalar"], [5])
        with self.assertRaisesRegex(ValueError, "unprovable"):
            self.evidence({ga: copied.replace("[sp, #16]\n\tfadd", "[sp, #48]\n\tfadd")})

    def test_policy_and_object_binding(self) -> None:
        with self.assertRaisesRegex(ValueError, "without the ordered policy holds an ordered add"):
            self.evidence({compact_name(ordered=0): self.SITE + "\tret\n"})
        develop = compact_name(ordered=0).replace("11rowsProgramILb0ELi16", "4rowsILi16").replace("T4_", "T3_") \
            .replace("EXT2_EERKSC_IPfXT2_", "EXT1_EERKSC_IPfXT1_")
        self.assertEqual(CHECKER.compact_program_key(develop), "compact:16:4:1:1:-:0")
        with self.assertRaisesRegex(ValueError, "without the ordered policy"):
            self.evidence({develop: self.SITE + "\tret\n"})
        ga = compact_name(ordered=1)
        functions = [(ga, self.SITE + "\tret\n")]
        text = assembly(functions).replace("fadd\tv5.4s, v5.4s, v6.4s", "fadd\tv5.4s, v5.4s, v7.4s")
        with self.assertRaisesRegex(ValueError, "does not hold the ordered add"):
            CHECKER.compact_ordered_add_evidence(witness_object(functions, None), disassembly(functions), text, "TU")


class FusedInstructionTest(unittest.TestCase):
    """S2-4c (synthetic): every fused multiply-add of the audited closure is an explicit source FMA on its located
    line; a contraction of a separate multiply and add fails; exact committed counts per copy."""

    SOURCE = "\n".join(["// kernel"] * 4 + ["    acc = vfmaq_f32(acc, a, b);",
                                            "    acc = vaddq_f32(acc, vmulq_f32(a, b));",
                                            "    acc = acc + a * b;",
                                            "    acc = std::fma(a, b, acc);"]) + "\n"
    NEON = "\n".join(["/* arm_neon */"] * 9 + ["  return __builtin_aarch64_fmav4sf (__b, __c, __a);",
                                                "  return __a + __b;"]) + "\n"
    SOURCES = {KERNEL_FILE: SOURCE, "arm_neon.h": NEON}

    def fused(self, bodies: list[tuple[str, str]], sources: dict | None = None) -> dict:
        functions, text = located(bodies)
        return CHECKER.fused_evidence(witness_object(functions, None, bindings={"helper": "local"}),
                                      disassembly(functions), text, "KernelFactory",
                                      self.SOURCES if sources is None else sources)

    def test_explicit_fma_sites_are_counted_through_the_closure(self) -> None:
        program = compact_name(ordered=0)
        bodies = [(program, "\tfmla\tv0.4s, v1.4s, v2.4s@@5\n\tfmla\tv0.4s, v1.4s, v2.s[1]@@a10\n\tbl\thelper\n\tret\n"),
                  ("helper", "\tfmadd\ts0, s1, s2, s0@@8\n\tret\n"),
                  ("unreached", "\tfmla\tv0.4s, v1.4s, v2.4s@@6\n\tret\n")]
        evidence = self.fused(bodies)
        self.assertEqual(evidence, {"compact:16:4:1:1:0:0@KernelFactory": {"fused": {"fmla": 2}},
                                    "helper@KernelFactory": {"fused": {"fmadd": 1}}})
        CHECKER.require_fused(evidence, json.loads(json.dumps(evidence)))
        lost = json.loads(json.dumps(evidence))
        lost["helper@KernelFactory"]["fused"] = {}
        with self.assertRaisesRegex(ValueError, "differ from the committed"):
            CHECKER.require_fused(evidence, lost)

    def test_contractions_and_unbound_fusions_fail(self) -> None:
        program = compact_name(ordered=0)
        for case, line in {"vmul then vadd": "@@6", "operators": "@@7", "arm_neon add body": "@@a11",
                           "unlocated": ""}.items():
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, "contraction|no bound source line"):
                    self.fused([(program, f"\tfmla\tv0.4s, v1.4s, v2.4s{line}\n\tret\n")])
        for mnemonic in ("fmls\tv0.4s, v1.4s, v2.4s", "fmsub\ts0, s1, s2, s3", "fnmadd\ts0, s1, s2, s3",
                         "fnmsub\td0, d1, d2, d3"):
            with self.subTest(mnemonic=mnemonic):
                with self.assertRaisesRegex(ValueError, "contraction"):
                    self.fused([(program, f"\t{mnemonic}@@6\n\tret\n")])
        with self.assertRaisesRegex(ValueError, "no bound source line"):
            self.fused([(program, "\tfmla\tv0.4s, v1.4s, v2.4s@@5\n\tret\n")], sources={})


class GroundedChainTest(unittest.TestCase):
    """F-1: a greatest-fixpoint chain must be grounded in a template, zero or seed origin."""

    def test_a_cycle_of_fadds_never_justifies_itself(self) -> None:
        # Two fadds whose A origins are only each other (no entry, no seed): accepted by the greatest fixpoint
        # alone, rejected as ungrounded.
        cycle = {10: ({("fadd", 20)}, {("other", 5)}), 20: ({("fadd", 10)}, {("other", 6)})}
        self.assertEqual(CHECKER.order_verdicts(cycle), (set(), {10: "ungrounded", 20: "ungrounded"}))
        anchored = {**cycle, 10: ({("fadd", 20), ("init", 1)}, {("other", 5)})}
        self.assertEqual(CHECKER.order_verdicts(anchored), ({10, 20}, {}))
        self.assertEqual(CHECKER.grounded({10, 20}, cycle), set())

    def test_seedless_two_fadd_loop_fails_end_to_end(self) -> None:
        # The entry pseudo-definition reaches A on the first iteration and is never a chain origin.
        body = (CompactOrderedAddTest.SITE + ".Lk:\n\tldr\tq3, [x1], #16\n\tfadd\ts0, s1, s3\n\tfadd\ts1, s0, s3\n"
                "\tsubs\tx2, x2, #1\n\tb.ne\t.Lk\n\tret\n")
        with self.assertRaisesRegex(ValueError, "unprovable|ungrounded"):
            CompactOrderedAddTest().evidence({compact_name(ordered=1): body})


class TileChainTest(unittest.TestCase):
    """T-1: in GPU-aligned tile programs every accumulator-chain addition is a template site; compiler fadds off the
    chain (within-block expression arithmetic) are listed and counted, never ordered."""

    SITE = "\tfadd\tv0.4s, v0.4s, v1.4s\n"
    GA, NATIVE = name(1, 0, 1, 1, 0, 0), name(1, 0, 0, 1, 0, 0)

    def evidence(self, body: str, native: str = "\tfadd\tv4.4s, v0.4s, v1.4s\n\tret\n") -> dict:
        functions = [(self.GA, body), (self.NATIVE, native)]
        return CHECKER.tile_chain_evidence(witness_object(functions, None), disassembly(functions),
                                           assembly(functions), "KernelFactory")

    def test_offchain_expression_adds_are_listed_and_counted(self) -> None:
        body = ("\tmovi\tv0.4s, #0\n.Lk:\n\tldr\tq2, [x1], #16\n\tldr\tq3, [x2], #16\n"
                "\tfadd\tv1.4s, v2.4s, v3.4s\n" + self.SITE + "\tsubs\tx3, x3, #1\n\tb.ne\t.Lk\n\tret\n")
        evidence = self.evidence(body)
        self.assertEqual(evidence, {"1:0:1:1:0:0@KernelFactory": {"sites": [4], "offchain": [3]}})
        CHECKER.require_tile_chain(evidence, {"1:0:1:1:0:0@KernelFactory": {"sites": 1, "offchain": 1}})
        with self.assertRaisesRegex(ValueError, "differ from the committed"):
            CHECKER.require_tile_chain(evidence, {"1:0:1:1:0:0@KernelFactory": {"sites": 1, "offchain": 0}})

    def test_compiler_adds_on_the_chain_fail(self) -> None:
        cases = {"reads a template result": self.SITE + "\tfadd\tv4.4s, v0.4s, v1.4s\n\tret\n",
                 "reads the zero init": "\tmovi\tv0.4s, #0\n\tfadd\tv4.4s, v2.4s, v0.4s\n" + self.SITE + "\tret\n",
                 "through a copy and a spill": self.SITE + "\tmov\tv7.16b, v0.16b\n\tstr\tq7, [sp, #16]\n"
                                                           "\tldr\tq8, [sp, #16]\n\tfadd\tv4.4s, v2.4s, v8.4s\n\tret\n",
                 "derived from a chain add": self.SITE + "\tfadd\tv4.4s, v0.4s, v1.4s\n\tfadd\tv5.4s, v2.4s, v4.4s\n"
                                                         "\tret\n",
                 # A compiler (non-template) accumulation in the loop: the in-place template form would be asm.
                 "loop-carried accumulator": "\tmovi\tv0.4s, #0\n.Lk:\n\tldr\tq1, [x1], #16\n" + self.SITE +
                                             "\tfadd\tv2.4s, v0.4s, v1.4s\n\tmov\tv0.16b, v2.16b\n"
                                             "\tsubs\tx3, x3, #1\n\tb.ne\t.Lk\n\tret\n"}
        # Every FP operation reading the chain fails, not only fadd.
        for op in ("fsub\tv4.4s, v2.4s, v0.4s", "fsub\tv4.4s, v0.4s, v2.4s", "fmul\tv4.4s, v0.4s, v2.4s",
                   "fneg\tv4.4s, v0.4s", "fabs\tv4.4s, v0.4s", "fmax\tv4.4s, v0.4s, v2.4s",
                   "fmin\tv4.4s, v2.4s, v0.4s", "fmla\tv0.4s, v2.4s, v3.4s", "fmla\tv4.4s, v0.4s, v3.4s",
                   "fmla\tv4.4s, v2.4s, v0.s[1]", "bsl\tv4.16b, v0.16b, v2.16b", "fcsel\ts4, s0, s2, eq",
                   "fcmeq\tv4.4s, v0.4s, v2.4s", "fcvtzs\tv4.4s, v0.4s"):
            cases[op] = self.SITE + f"\t{op}\n\tret\n"
        # ... also through a lane move (moves propagate the chain) and after a spill.
        cases["through a lane move"] = self.SITE + "\tdup\tv7.4s, v0.s[1]\n\tfsub\tv4.4s, v2.4s, v7.4s\n\tret\n"
        cases["through an FP->GP->lane move"] = self.SITE + "\tmov\tv7.s[0], v0.s[1]\n\tfmul\ts4, s7, s2\n\tret\n"
        for case, body in cases.items():
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, "touches the accumulator chain"):
                    self.evidence(body)
        # Allowed: moves, a spill and reload and the final store of the chain, and further template sites.
        allowed = (self.SITE + "\tmov\tv7.16b, v0.16b\n\tdup\tv8.4s, v7.s[0]\n\tstr\tq8, [sp, #16]\n"
                   "\tldr\tq9, [sp, #16]\n\tfadd\tv9.4s, v9.4s, v1.4s\n\tst1\t{v9.s}[1], [x0]\n"
                   "\tumov\tw3, v9.s[2]\n\tstr\tq0, [x0]\n\tret\n")
        self.assertEqual(self.evidence(allowed), {"1:0:1:1:0:0@KernelFactory": {"sites": [0, 5], "offchain": []}})
        # zip1 (or any FP/SIMD op) carrying the chain itself, or reading the accumulator's +0.0, touches it.
        for case, body in {"chain through zip1": self.SITE + "\tzip1\tv7.8h, v0.8h, v2.8h\n\tret\n",
                           "zip1 of the accumulator zero": "\tmovi\tv0.4s, #0\n\tzip1\tv7.8h, v2.8h, v0.8h\n"
                                                           + self.SITE + "\tret\n",
                           "zip1 of the chain, then fadd": self.SITE + "\tzip1\tv7.8h, v2.8h, v0.8h\n"
                                                           "\tfadd\tv4.4s, v7.4s, v2.4s\n\tret\n"}.items():
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, "touches the accumulator chain"):
                    self.evidence(body)
        # Only an all-zero constant that is not the accumulator operand is exempt (integer widening, a mask).
        widening = ("\tmovi\tv6.4s, #0\n\tmovi\tv0.4s, #0\n\tzip1\tv7.8h, v2.8h, v6.8h\n"
                    "\tfadd\tv4.4s, v6.4s, v2.4s\n" + self.SITE + "\tret\n")
        self.assertEqual(self.evidence(widening), {"1:0:1:1:0:0@KernelFactory": {"sites": [4], "offchain": [3]}})
        with self.assertRaisesRegex(ValueError, "without an ordered add"):
            self.evidence("\tfadd\tv1.4s, v2.4s, v3.4s\n\tret\n")


class LinkSelectedFusedTest(unittest.TestCase):
    """F-2: weak/global callees execute their link-selected copy: its fused evidence (P2) is required, never a
    quiet skip."""

    SOURCES = FusedInstructionTest.SOURCES

    def fused(self, helper: str, binding: str = "weak", **options) -> dict:
        program = compact_name(ordered=0)
        functions, text = located([(program, "\tbl\thelper\n\tret\n"), ("helper", helper)])
        return CHECKER.fused_evidence(witness_object(functions, None, bindings={"helper": binding}),
                                      disassembly(functions), text, "KernelFactory", self.SOURCES, **options)

    def test_unselected_fp_callee_is_incomplete(self) -> None:
        for binding in ("weak", "global"):
            with self.subTest(binding=binding):
                with self.assertRaisesRegex(ValueError, "no link-selection record .* incomplete"):
                    self.fused("\tfmul\ts0, s1, s2\n\tret\n", binding)
        self.assertEqual(self.fused("\tmov\tx0, x1\n\tret\n")["unselected@KernelFactory"],
                         {"fused": {}, "callees": ["helper"]})

    def test_selected_copy_evidence_is_required_and_included(self) -> None:
        other = other_unit("tuB", {"helper": "\tfmadd\ts0, s1, s2, s0\n\tret\n"})
        record = {"helper": {"copy": "tuB", **PROCESS}}
        selected = {"helper@tuB": {"fused": {"fmadd": 1}}}
        evidence = self.fused("\tfmul\ts0, s1, s2\n\tret\n", selection=record, units={"tuB": other},
                              selected=selected)
        self.assertEqual(evidence["helper@tuB"], {"fused": {"fmadd": 1}})
        self.assertNotIn("helper@KernelFactory", evidence)
        with self.assertRaisesRegex(ValueError, "fused evidence helper@tuB is not supplied"):
            self.fused("\tfmul\ts0, s1, s2\n\tret\n", selection=record, units={"tuB": other})
        with self.assertRaisesRegex(ValueError, "PLT binding is preemptible"):
            self.fused("\tfmul\ts0, s1, s2\n\tret\n", selection={"helper": {"copy": "tuB", "binding": "plt"}},
                       units={"tuB": other}, selected=selected)
        with self.assertRaisesRegex(ValueError, "not audited"):
            self.fused("\tfmul\ts0, s1, s2\n\tret\n", selection=record, selected=selected)

    def test_selected_copy_evidence_comes_from_its_own_object(self) -> None:
        source = FusedInstructionTest.SOURCES
        functions, text = located([("helper", "\tfmadd\ts0, s1, s2, s0@@8\n\tret\n")])
        evidence = CHECKER.fused_evidence(witness_object(functions, None, bindings={"helper": "weak"}),
                                          disassembly(functions), text, "tuB", source, roots=("helper",))
        self.assertEqual(evidence, {"helper@tuB": {"fused": {"fmadd": 1}}})


class PrelinkFusedScopeTest(unittest.TestCase):
    """Reviewer 102: the probe has no link authority, so its fused proof is selection-independent and stricter:
    this object's own copy of every weak/global helper must satisfy the witness; production (post-link) keeps the
    P2 selected-copy rule."""

    SOURCES = FusedInstructionTest.SOURCES

    def prelink(self, helper: str, binding: str = "weak", **options) -> dict:
        program = compact_name(ordered=0)
        functions, text = located([(program, "\tbl\thelper\n\tret\n"), ("helper", helper)])
        return CHECKER.fused_evidence(witness_object(functions, None, bindings={"helper": binding}),
                                      disassembly(functions), text, "probe", self.SOURCES, scope="prelink", **options)

    def test_every_weak_copy_is_audited_without_a_selection_record(self) -> None:
        for binding in ("weak", "global"):
            with self.subTest(binding=binding):
                evidence = self.prelink("\tfmul\ts0, s1, s2\n\tret\n", binding)
                self.assertEqual(evidence["helper@probe"], {"fused": {}})
                self.assertNotIn("unselected@probe", evidence)

    def test_a_bad_weak_copy_fails_even_when_another_copy_would_be_selected(self) -> None:
        # A contraction in this object's weak copy fails pre-link whatever copy a link would keep.
        with self.assertRaisesRegex(ValueError, "helper@probe: fused .* (not an explicit source FMA|no bound source)"):
            self.prelink("\tfmadd\ts0, s1, s2, s0@@7\n\tret\n")
        other = other_unit("tuB", {"helper": "\tfmul\ts0, s1, s2\n\tret\n"})
        with self.assertRaisesRegex(ValueError, "no link authority"):
            self.prelink("\tfmadd\ts0, s1, s2, s0@@7\n\tret\n", selection={"helper": {"copy": "tuB", **PROCESS}},
                         units={"tuB": other}, selected={"helper@tuB": {"fused": {}}})

    def test_postlink_selected_bad_copy_still_fails(self) -> None:
        functions, text = located([("helper", "\tfmadd\ts0, s1, s2, s0@@7\n\tret\n")])
        with self.assertRaisesRegex(ValueError, "helper@tuB: fused"):
            CHECKER.fused_evidence(witness_object(functions, None, bindings={"helper": "weak"}),
                                   disassembly(functions), text, "tuB", self.SOURCES, roots=("helper",))

    def test_unknown_scope_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown fused proof scope"):
            CHECKER.fused_evidence(b"", "", "", "probe", {}, scope="maybe")


class ChainCountsTest(unittest.TestCase):
    """The probe's chain witness is bound to exact committed counts per identity and envelope."""

    def test_counts_are_exact(self) -> None:
        committed = chain_counts()
        witness = json.loads(json.dumps(committed["GNU-14"]["NEON"]))
        CHECKER.require_chain_counts(witness, "GNU-14", "NEON", committed)
        for edit in (lambda w: w["tile_chain"]["1:0:1:0:0:0"].update(offchain=1),
                     lambda w: w["fused"]["1:0:0:0:0:0"]["fused"].update(fmla=1),
                     lambda w: w["tile_chain"].pop("1:0:1:0:0:0")):
            changed = json.loads(json.dumps(witness))
            edit(changed)
            with self.assertRaisesRegex(ValueError, "differs from the committed counts"):
                CHECKER.require_chain_counts(changed, "GNU-14", "NEON", committed)
        with self.assertRaisesRegex(ValueError, "no committed chain witness"):
            CHECKER.require_chain_counts(witness, "GNU-14", "NEON_I8MM", committed)

    def test_mismatch_detail_does_not_depend_on_key_order(self) -> None:
        # An unqualified analysis record is re-checked by comparing its stored failure details with the recomputed
        # ones: a freshly computed witness and its JSON round trip (sorted keys) must yield the same text.
        committed = chain_counts()
        fresh = {"tile_chain": {"1:0:1:0:0:0": {"sites": 9, "offchain": 0}}}
        stored = json.loads(json.dumps(fresh, sort_keys=True))
        details = []
        for witness in (fresh, stored):
            with self.assertRaisesRegex(ValueError, "differs from the committed counts") as raised:
                CHECKER.require_chain_counts(witness, "GNU-14", "NEON", committed)
            details.append(str(raised.exception))
        self.assertEqual(details[0], details[1])

    def test_closure_sources_drop_ambiguous_names(self) -> None:
        # Hermetic: name the files relative to the test's own temporary directory. Its absolute location (CTest
        # points TMPDIR into the build tree's tests/v2/mpi-sessions) must not reach source_name().
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                paths = {}
                for relative in ("a/src/v2/k.h", "a/include/arm_neon.h", "b/include/arm_neon.h", "a/include/other.h"):
                    path = Path(relative)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(relative)
                    paths[relative] = {}
                self.assertEqual(CHECKER.closure_sources(paths), {"src/v2/k.h": "a/src/v2/k.h",
                                                                  "other.h": "a/include/other.h"})
            finally:
                os.chdir(previous)


DRIVER_PATH = CHECKER_PATH.with_name("CPUNativeTileProofDriver.py")
DRIVER_SPEC = importlib.util.spec_from_file_location("cpu_native_tile_proof_driver", DRIVER_PATH)
DRIVER = importlib.util.module_from_spec(DRIVER_SPEC)
DRIVER_SPEC.loader.exec_module(DRIVER)


class ProofDriverTest(unittest.TestCase):
    """One driver: proof -> stack gate -> fused in one process; every stage runs, is logged and receipted, earlier
    outputs are retired first, any failure fails the driver, and the receipt is written atomically before and after
    every stage so that a killed attempt is never green."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def args(self, phase: str) -> argparse.Namespace:
        return argparse.Namespace(phase=phase, proof=str(self.directory / "proof.json"), mode="NEON",
                                  baseline=str(self.directory / "b.json"), roles=str(self.directory / "r.json"),
                                  keymap=str(self.directory / "k.json"))

    def drive(self, phase: str, outcomes: dict[str, str], leave: str | None = None) -> tuple[int, dict, list]:
        for name in ("proof.json", "proof.analysis.json", "stack-gate.json", "fused-stage.json", "driver.json",
                     "stage-proof.log", "stage-stack_gate.log", "stage-fused.log"):
            (self.directory / name).write_text("stale")  # An earlier attempt's outputs, all retired first.
        seen = []

        def stage(name: str):
            def function(context) -> dict:
                receipt = json.loads((self.directory / "driver.json").read_text())
                seen.append((name, receipt["stages"][name]["status"], receipt["status"],
                             {other for other in ("stack-gate.json", "fused-stage.json")
                              if (self.directory / other).exists()}))
                if name == "proof" and leave:
                    (self.directory / leave).write_text("{}")
                print(f"{name} output")
                if outcomes.get(name) == "raise":
                    raise ValueError(f"{name} broke")
                return {"status": outcomes.get(name, "passed")}
            return function
        status = DRIVER.drive(self.args(phase), ["--proof=x", "--depfile-target=y"],
                              {name: stage(name) for name in DRIVER.STAGES})
        return status, json.loads((self.directory / "driver.json").read_text()), seen

    def test_every_stage_runs_and_any_failure_fails(self) -> None:
        status, receipt, seen = self.drive("build", {}, leave="proof.json")
        self.assertEqual((status, receipt["status"], [name for name, *_ in seen]),
                         (0, "passed", ["proof", "stack_gate", "fused"]))
        self.assertEqual(DRIVER.classify(receipt)["status"], "passed")
        for failing in DRIVER.STAGES:
            for how in ("failed", "raise"):
                with self.subTest(failing=failing, how=how):
                    status, receipt, seen = self.drive("verify", {failing: how})
                    self.assertEqual((status, receipt["status"], len(seen)), (1, "failed", 3))
                    self.assertEqual({name: stage["status"] for name, stage in receipt["stages"].items()},
                                     {name: "failed" if name == failing else "passed" for name in DRIVER.STAGES})
                    if how == "raise":
                        log = (self.directory / f"stage-{failing}.log").read_text()
                        self.assertIn("Traceback", log)
                        self.assertIn(f"{failing} broke", receipt["stages"][failing]["reason"])

    def test_receipt_is_incomplete_until_the_end_and_running_before_each_stage(self) -> None:
        _, receipt, seen = self.drive("verify", {})
        self.assertEqual([(name, state, overall) for name, state, overall, _ in seen],
                         [(name, "running", "incomplete") for name in DRIVER.STAGES])
        self.assertTrue(all(found == set() for *_, found in seen))  # DR-1: no earlier receipt survives.
        for name, stage in receipt["stages"].items():
            log = self.directory / stage["log"]
            self.assertEqual(stage["log_sha256"], hashlib.sha256(log.read_bytes()).hexdigest())
            self.assertIn(f"{name} output", log.read_text())
        self.assertEqual(len(receipt["attempt"]), 32)
        self.assertTrue(all(len(receipt["tools_sha256"][name] or "") == 64 for name in DRIVER.TOOLS))
        self.assertEqual(receipt["checker_argv"], ["--proof=x", "--depfile-target=y"])

    def test_phase_build_retires_the_documents_and_verify_keeps_them(self) -> None:
        self.drive("build", {"proof": "failed"})
        self.assertFalse((self.directory / "proof.json").exists() or
                         (self.directory / "proof.analysis.json").exists())
        _, receipt, _ = self.drive("verify", {})
        self.assertEqual(receipt["stage1_document"], "sealed-proof")  # The build's seal is what verify reads.

    def test_a_killed_attempt_is_never_green(self) -> None:
        # V3: SIGKILL in the middle of the stack gate leaves the receipt incomplete with that stage running.
        script = f"""
import argparse, importlib.util, os, signal, sys
spec = importlib.util.spec_from_file_location("driver", {str(DRIVER_PATH)!r})
driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
def passed(context):
    return {{"status": "passed"}}
def killed(context):
    print("partial output", flush=True)
    os.kill(os.getpid(), signal.SIGKILL)
args = argparse.Namespace(phase="verify", proof={str(self.directory / "proof.json")!r}, mode="NEON",
                          baseline="b", roles="r", keymap="k")
driver.drive(args, [], {{"proof": passed, "stack_gate": killed, "fused": passed}})
"""
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertLess(completed.returncode, 0)
        receipt = json.loads((self.directory / "driver.json").read_text())
        self.assertEqual((receipt["status"], receipt["stages"]["proof"]["status"],
                          receipt["stages"]["stack_gate"]["status"], receipt["stages"]["fused"]["status"]),
                         ("incomplete", "passed", "running", "planned"))
        self.assertEqual(DRIVER.classify(receipt), {"status": "failed", "stages": {
            "proof": "passed", "stack_gate": "unknown", "fused": "not-run"}})
        self.assertIn("partial output", (self.directory / "stage-stack_gate.log").read_text())
        self.assertFalse((self.directory / "driver.json.tmp").exists())

    def test_command_line_runs_every_stage_without_inputs(self) -> None:
        proof = str(self.directory / "proof.json")
        command = [sys.executable, "-B", str(DRIVER_PATH), "verify", f"--proof={proof}", "--mode=NEON",
                   "--baseline=b", "--roles=r", "--keymap=k", "--", f"--proof={proof}", "--mode=NEON"]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 1, completed.stderr)
        receipt = json.loads(Path(proof).with_name("driver.json").read_text())
        self.assertEqual({name: stage["status"] for name, stage in receipt["stages"].items()},
                         {name: "failed" for name in DRIVER.STAGES})
        self.assertEqual(json.loads(Path(proof).with_name("stack-gate.json").read_text())["status"], "failed")
        self.assertEqual(json.loads(Path(proof).with_name("fused-stage.json").read_text())["input"], None)


class ParseCacheTest(unittest.TestCase):
    """M1: memoised parses hand every caller its own objects; mutating one result never changes another."""

    def test_mutating_a_result_never_reaches_the_cache(self) -> None:
        functions = to_abi(programs(), False)
        data, text = witness_object(functions), disassembly(functions)
        first = CHECKER.read_object(data)
        first.symbols.clear()
        first.sections.clear()
        next(iter(first.relocations.values()), {}).clear()
        first.landing["x"] = [(0, 0, 0)]
        second = CHECKER.read_object(data)
        self.assertTrue(second.symbols and second.sections and "x" not in second.landing)
        self.assertEqual(second, CHECKER._read_elf(data))
        bodies = CHECKER.normalize_disassembly(text, second)
        name = next(iter(bodies))
        bodies[name].append("\tinjected")
        bodies.eh["x"] = {0: ".L0"}
        bodies.pop(next(reversed(bodies)))
        again = CHECKER.normalize_disassembly(text, CHECKER.read_object(data))
        self.assertNotIn("\tinjected", again[name])
        self.assertEqual((dict(again), again.eh), (dict(CHECKER._normalize_disassembly(text, second)), {}))
        listing = assembly(functions)
        found, _ = CHECKER.assembly_functions(listing, set(again), False)
        found[name].append(("injected", 0))
        clean, _ = CHECKER.assembly_functions(listing, set(again), False)
        self.assertNotIn(("injected", 0), clean[name])

    def test_changed_inputs_are_parsed_afresh(self) -> None:
        functions = to_abi(programs(), False)
        data = witness_object(functions)
        other = witness_object(to_abi(programs(keys=NEON_KEYS[:4]), False))
        self.assertNotEqual(CHECKER.read_object(data), CHECKER.read_object(other))


class TestInventoryTest(unittest.TestCase):
    """The registered selections run the complete test inventory, measured by the unittest loader (not by reading
    the source): the shard union of this file is every test exactly once, and the stack-table script, run the way
    its CTest runs it, executes every test the loader finds in it (a second or early `unittest.main()` would stop
    later classes silently)."""

    def test_shards_cover_every_test_exactly_once(self) -> None:
        loader = unittest.defaultTestLoader
        everything = sorted(test.id() for test in iterate(loader.loadTestsFromModule(sys.modules[__name__])))
        shards = ["analysis"] + [f"binding-{index}" for index in range(BINDING_SHARDS)]
        selected = sorted(test.id() for shard in shards for test in iterate(shard_suite(shard)))
        self.assertEqual(selected, everything)
        self.assertEqual(len(selected), len(set(selected)))

    def test_the_stack_table_script_runs_its_whole_inventory(self) -> None:
        path = CHECKER_PATH.parents[1] / "unit" / "cmake" / "test_cpu_native_tile_stack_table.py"
        spec = importlib.util.spec_from_file_location("stack_table_inventory", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = sum(1 for _ in iterate(unittest.defaultTestLoader.loadTestsFromModule(module)))
        with tempfile.TemporaryDirectory() as directory:  # Another working directory, as CTest's.
            completed = subprocess.run([sys.executable, "-B", str(path)], capture_output=True, text=True, cwd=directory)
        ran = re.findall(r"^Ran (\d+) tests?", completed.stderr, re.MULTILINE)
        self.assertEqual(ran, [str(expected)], completed.stderr[-2000:])


def iterate(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iterate(item)
        else:
            yield item


class ParallelAnalysisTest(unittest.TestCase):
    """V2: per-program parallel analysis equals the serial reference in every field for 1, 2, an odd and the
    physical worker count; a worker's exception is the serial one; a dead worker fails; physical cores only."""

    GA = {(rows, 0, 1, encoding, nibble_map, minimum): "\tmovi\tv0.4s, #0\n\tfadd\tv1.4s, v2.4s, v3.4s\n"
          "\tfadd\tv0.4s, v0.4s, v1.4s\n\tstr\tq9, [sp, #-16]!\n\tadd\tsp, sp, #16\n\tret"
          for rows, _, policy, encoding, nibble_map, minimum in NEON_KEYS if policy == 1}

    def inputs(self, bodies: dict | None = None) -> tuple[bytes, str, str]:
        functions = to_abi(programs(bodies if bodies is not None else self.GA), False)
        return witness_object(functions), disassembly(functions), assembly(functions)

    def fields(self, workers: int, bodies: dict | None = None) -> tuple[str, str]:
        data, text, listing = self.inputs(bodies)
        analyses: dict = {}
        fields = CHECKER.analysis_fields(data, text, listing, {}, "NEON", analyses, workers)
        views = {name: {key: (repr(value) if key == "blocks" else value) for key, value in vars(view).items()}
                 for name, view in analyses.items()}
        return json.dumps(fields, sort_keys=True), json.dumps(views, sort_keys=True, default=repr)

    def test_every_worker_count_equals_the_serial_reference(self) -> None:
        serial = self.fields(1)
        self.assertIn('"offchain"', serial[0])
        for workers in (2, 3, 5, max(2, CHECKER.physical_workers()["workers"]), 64):
            with self.subTest(workers=workers):
                self.assertEqual(self.fields(workers), serial)
        data, text, listing = self.inputs()
        chain = [CHECKER.tile_chain_evidence(data, text, listing, "probe", {}, workers) for workers in (1, 3)]
        self.assertEqual(chain[0], chain[1])
        self.assertEqual(list(chain[0]), list(chain[1]))  # Same merge order.

    def test_input_mutation_and_failures_propagate_as_in_the_serial_run(self) -> None:
        broken = {**self.GA, NEON_KEYS[3]: "\tfrobnicate x9, x1\n\tret", NEON_KEYS[17]: "\tfrobnicate x9, x2\n\tret"}
        messages = []
        for workers in (1, 2, 3):
            with self.assertRaises(ValueError) as caught:
                self.fields(workers, broken)
            messages.append(str(caught.exception))
        self.assertEqual(len(set(messages)), 1, messages)
        self.assertIn("structurally unprovable", messages[0])
        # An exception raised in a worker is the first failing program's, in input order.
        original = CHECKER.analysed
        failing = {CHECKER.tile_match(name(*NEON_KEYS[key]), "").group(0) for key in (5, 9)}

        def raising(program, *args, **kwargs):
            if program in failing:
                raise ValueError(f"worker failure {program[-30:]}")
            return original(program, *args, **kwargs)
        CHECKER.analysed = raising
        try:
            outcomes = []
            for workers in (1, 2, 4):
                with self.assertRaises(ValueError) as caught:
                    self.fields(workers)
                outcomes.append(str(caught.exception))
        finally:
            CHECKER.analysed = original
        self.assertEqual(len(set(outcomes)), 1, outcomes)

    def test_a_dead_worker_fails_without_partial_results(self) -> None:
        original = CHECKER.analysed
        victim = CHECKER.tile_match(name(*NEON_KEYS[6]), "").group(0)

        def dying(program, *args, **kwargs):
            if program == victim:
                os.kill(os.getpid(), signal.SIGKILL)
            return original(program, *args, **kwargs)
        CHECKER.analysed = dying
        try:
            with self.assertRaisesRegex(ValueError, "an analysis worker died"):
                self.fields(3)
        finally:
            CHECKER.analysed = original
        self.assertEqual(CHECKER.map_programs(lambda item: item * 2, list(range(7)), 3), [0, 2, 4, 6, 8, 10, 12])
        self.assertEqual(CHECKER.map_programs(lambda item: item, [], 4), [])

    def run_parent(self, how: str) -> list[int]:
        """A parent process mapping slow items over 3 workers, which records the worker pids and then is ended
        mid-map (`how`: SIGKILL or SIGINT); returns the worker pids."""
        with tempfile.TemporaryDirectory() as directory:
            pids = Path(directory) / "pids"
            script = f"""
import importlib.util, os, sys, time
spec = importlib.util.spec_from_file_location("ev", {str(CHECKER_PATH)!r})
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
def slow(item):
    with open({str(pids)!r}, "a") as handle:
        handle.write(f"{{os.getpid()}}\\n")
    time.sleep(60)
    return item
ev.map_programs(slow, list(range(3)), 3)
"""
            process = subprocess.Popen([sys.executable, "-c", script], stderr=subprocess.PIPE)
            deadline = time.time() + 20
            while time.time() < deadline and (not pids.exists() or len(pids.read_text().split()) < 3):
                time.sleep(0.05)
            workers = [int(pid) for pid in pids.read_text().split()]
            process.send_signal(signal.SIGKILL if how == "kill" else signal.SIGINT)
            process.wait(timeout=20)
            return workers

    def assertGone(self, workers: list[int]) -> None:
        deadline = time.time() + 10
        alive = workers
        while alive and time.time() < deadline:
            alive = []
            for pid in workers:
                try:
                    os.kill(pid, 0)
                    alive.append(pid)
                except ProcessLookupError:
                    pass
            time.sleep(0.05)
        self.assertEqual(alive, [], "an analysis worker survived its parent")

    def test_no_worker_survives_a_killed_parent(self) -> None:
        # W1: the parent SIGKILLed mid-map (no finally runs): the workers die with it.
        workers = self.run_parent("kill")
        self.assertEqual(len(workers), 3)
        self.assertGone(workers)

    def test_an_interrupted_parent_kills_and_reaps_its_workers(self) -> None:
        # W1: KeyboardInterrupt in the parent mid-map: its finally kills and reaps every worker.
        workers = self.run_parent("interrupt")
        self.assertEqual(len(workers), 3)
        self.assertGone(workers)

    def test_workers_are_physical_cores_of_the_admitted_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Two packages x two cores x two hyperthreads: cpu = 4 * package + 2 * core + thread.
            for cpu in range(8):
                topology = root / f"cpu{cpu}" / "topology"
                topology.mkdir(parents=True)
                (topology / "physical_package_id").write_text(str(cpu // 4))
                (topology / "core_id").write_text(str((cpu // 2) % 2))
            cases = {(0, 1, 2, 3, 4, 5, 6, 7): (4, [0, 2, 4, 6]), (0, 1): (1, [0]), (1, 3, 5): (3, [1, 3, 5]),
                     (6, 7, 2): (2, [2, 6])}
            for admitted, (workers, cpus) in cases.items():
                with self.subTest(admitted=admitted):
                    observed = CHECKER.physical_workers(list(admitted), root)
                    self.assertEqual((observed["workers"], observed["cpus"], observed["admitted"]),
                                     (workers, cpus, len(admitted)))
            missing = CHECKER.physical_workers([0, 9], root)
            self.assertEqual((missing["workers"], missing["cpus"]), (1, []))  # Never one worker per logical CPU.


# Assembler-input parser pins (P1-P3): the exact outputs of _assembly_functions() captured from the parser before
# its data-structure/loop-shape optimisation, committed in cpu_native_tile_assembly_parser_expected.json. Every
# later parser must reproduce them byte for byte; there is no second parser in the tree to compare against.
PARSER_ELF = """\t.text
\t.file\t"probe.cpp"
\t.file 1 "/root/tree/src/v2/kernels/cpu/gemm/CPUNativeNEONPreparedProjection.h"
\t.file 2 "/usr/lib/gcc/aarch64-linux-gnu/14/include" "arm_neon.h"
\t.file 3 "/elsewhere/tests/v2/unit/fixture.h"
\t.globl\tfirst
first:\t// a comment after the label
.LFB0:
\t.cfi_startproc
\t.loc 1 42 7
\tstp\tx29, x30, [sp, -16]!\t// save
\tmov\tx29, sp ; add x0, x0, #1 ;  sub x1, x1, #2
\t.loc 2 1210 10 is_stmt 0
#APP
// 66 "contract.h" 1
\tfadd\tv0.4s, v0.4s, v1.4s
// 0 "" 2
#NO_APP
\t.loc 3 9 1
\t   ldp   x29,   x30, [sp], 16
\t.loc 9 5 0
\tnop
helper_local:
\tret
\t.cfi_endproc
.LFE0:
\t.size\tfirst, .-first
second:
\t.loc_mark_labels 1
\t.locview\t.LVU1
\t@ an at comment
\t; a semicolon comment
\t.loc 1 7
\tret
#APP
\tdmb ish
#NO_APP
not_audited:
\tmov x0, x1
#APP
\tisb
#NO_APP
"""

PARSER_MACHO = """\t.section\t__TEXT,__text,regular,pure_instructions
\t.file\t1 "/Users/x/tree/src/v2/kernels/cpu/gemm/CPUNativeQuantisedMultiScaleKernels.h"
\t.globl\t_first
_first:
\t.loc\t1 100 3
\tstp\tx29, x30, [sp, #-16]! ; a Mach-O comment ; not a statement
\tmov\tx29, sp
; InlineAsm Start
\tfadd\tv2.4s, v2.4s, v3.4s
; InlineAsm End
\tret
_second: ; trailing comment
\tret
"""

PARSER_NAMES = {"first", "second", "_first", "_second"}

PARSER_ERRORS = {
    "nested": "#APP\n#APP\n#NO_APP\n#NO_APP\n",
    "unbalanced": "first:\n#NO_APP\n",
    "label inside": "#APP\nfirst:\n#NO_APP\n",
    "directive inside": "first:\n#APP\n\tnop ; .word 0\n#NO_APP\n",
    "unterminated": "first:\n#APP\n\tnop\n",
}


def parser_generated(seed: int = 20261126, count: int = 400) -> tuple[str, set]:
    """A fixed-seed random ELF assembly: functions, local labels, .file/.loc (valid and malformed), directives,
    comments, ';'-separated statements and inline-asm blocks in random order."""
    rng = random.Random(seed)
    names = {f"f{index}" for index in range(12)}
    lines = ['\t.file 1 "/t/src/v2/a.h"', '\t.file 2 "/t/x/b.h" "c.h"']
    open_block = False
    for _ in range(count):
        choice = rng.randrange(14)
        if choice == 0 and not open_block:
            lines.append(f"f{rng.randrange(14)}:")  # f12/f13 are not object functions.
        elif choice == 1:
            lines.append(f".L{rng.randrange(99)}:")
        elif choice == 2:
            lines.append(f"\t.loc {rng.randrange(1, 4)} {rng.randrange(1, 900)} {rng.randrange(0, 9)}")
        elif choice == 3:
            lines.append(rng.choice(["\t.loc x 1", "\t.loc", "\t.locview .LVU3", "\t.file 9", "\t.cfi_offset 29, -16"]))
        elif choice == 4:
            lines.append(rng.choice(["\t// c", "\t@ c", "\t; c", "", "   "]))
        elif choice == 5 and not open_block:
            lines.append("#APP"); open_block = True
        elif choice == 6 and open_block:
            lines.append("#NO_APP"); open_block = False
        else:
            statements = [rng.choice(["add x0, x1, #2", "ldr  q0, [x1]", "fmla v0.4s, v1.4s, v2.4s", "ret",
                                      "mov   x2,   x3"]) for _ in range(rng.randrange(1, 4))]
            lines.append("\t" + rng.choice([" ; ", ";", " ;\t"]).join(statements) + rng.choice(["", "\t// tail"]))
    if open_block:
        lines.append("#NO_APP")
    return "\n".join(lines) + "\n", names


PARSER_EXPECTED = json.loads(CHECKER_PATH.parents[1].joinpath(
    "unit", "cmake", "cpu_native_tile_assembly_parser_expected.json").read_text())


class AssemblyParserGoldenTest(unittest.TestCase):
    """The assembler-input parser reproduces its committed outputs exactly: functions, inline-asm block numbers,
    stray inline assembly, source locations, every rejection message, and the fixed-seed generated input."""

    def parse(self, text: str, names: set, macho: bool = False) -> dict:
        locations: dict = {}
        functions, stray = CHECKER._assembly_functions(text, names, macho, locations)
        wrapped_locations: dict = {}
        wrapped = CHECKER.assembly_functions(text, names, macho, wrapped_locations)
        self.assertEqual((wrapped[0], wrapped[1], wrapped_locations), (functions, stray, locations))
        return json.loads(json.dumps({"functions": functions, "stray": stray, "locations": locations}))

    def test_handcrafted_elf_and_macho_outputs(self) -> None:
        self.assertEqual(self.parse(PARSER_ELF, PARSER_NAMES), PARSER_EXPECTED["elf"])
        self.assertEqual(self.parse(PARSER_MACHO, PARSER_NAMES, macho=True), PARSER_EXPECTED["macho"])

    def test_every_rejection_message_is_exact(self) -> None:
        for label, text in PARSER_ERRORS.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError) as caught:
                    CHECKER._assembly_functions(text, PARSER_NAMES, False, {})
                self.assertEqual(str(caught.exception), PARSER_EXPECTED["errors"][label])

    def test_fixed_seed_generated_input(self) -> None:
        text, names = parser_generated()
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), PARSER_EXPECTED["generated"]["input_sha256"])
        locations: dict = {}
        functions, stray = CHECKER._assembly_functions(text, names, False, locations)
        canonical = json.dumps([functions, stray, locations], sort_keys=True, separators=(",", ":"))
        self.assertEqual({"sha256": hashlib.sha256(canonical.encode()).hexdigest(), "functions": len(functions),
                          "instructions": sum(map(len, functions.values())), "stray": len(stray),
                          "input_sha256": PARSER_EXPECTED["generated"]["input_sha256"]},
                         PARSER_EXPECTED["generated"])


class StackTrafficTest(unittest.TestCase):
    """Per-class, per-loop stack traffic: counted where it executes per iteration, exempt only at entry level."""

    FRAME, EXIT = "\tstp\tx29, x30, [sp, #-48]!\n\tmov\tx29, sp\n", "\tldp\tx29, x30, [sp], #48\n\tret\n"

    def counts(self, body: str) -> dict:
        counts, _, errors = traffic(body)
        self.assertEqual(errors, [], body)
        return {kind: values for kind, values in counts.items() if values}

    def test_gpr_spill_in_the_k_loop_is_scalar_traffic_in_that_loop(self) -> None:
        body = (self.FRAME + ".Lk:\n\tstr\tx9, [sp, #16]\n\tadd\tx9, x9, #1\n\tldr\tx9, [sp, #16]\n"
                "\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n" + self.EXIT)
        self.assertEqual(self.counts(body), {"scalar": {"1": 2}})
        # The same spill and reload outside every loop is still traffic, at the entry level.
        self.assertEqual(self.counts(self.FRAME + "\tstr\tw9, [sp, #16]\n\tldr\tw9, [sp, #16]\n" + self.EXIT),
                         {"scalar": {"0": 2}})

    def test_back_edge_callee_saved_fp_save_restore_is_counted(self) -> None:
        loop = ("\tstp\tx29, x30, [sp, #-32]!\n.Lloop:\n\tcbz\tx0, .Ldone\n\tstr\td14, [sp, #16]\n"
                "\tmovi\tv14.4s, #0x0\n\tldr\td14, [sp, #16]\n\tsub\tx0, x0, #1\n\tb\t.Lloop\n"
                ".Ldone:\n\tldp\tx29, x30, [sp], #32\n\tret\n")
        self.assertEqual(self.counts(loop), {"abi_loop": {"1": 2}})
        # Saved once at entry and restored once at exit: exempt.
        entry = ("\tstp\tx29, x30, [sp, #-32]!\n\tstr\td14, [sp, #16]\n.Lloop:\n\tmovi\tv14.4s, #0x0\n"
                 "\tsub\tx0, x0, #1\n\tcbnz\tx0, .Lloop\n\tldr\td14, [sp, #16]\n\tldp\tx29, x30, [sp], #32\n\tret\n")
        self.assertEqual(self.counts(entry), {})

    def test_callee_saved_gpr_save_restore_inside_a_loop_is_counted(self) -> None:
        loop = ("\tstp\tx29, x30, [sp, #-32]!\n.Lloop:\n\tcbz\tx0, .Ldone\n\tstr\tx19, [sp, #16]\n"
                "\tmov\tx19, x1\n\tldr\tx19, [sp, #16]\n\tsub\tx0, x0, #1\n\tb\t.Lloop\n"
                ".Ldone:\n\tldp\tx29, x30, [sp], #32\n\tret\n")
        self.assertEqual(self.counts(loop), {"abi_loop": {"1": 2}})

    def test_entry_level_prologue_and_epilogue_saves_are_exempt(self) -> None:
        body = ("\tstp\tx29, x30, [sp, #-64]!\n\tmov\tx29, sp\n\tstp\tx19, x20, [sp, #16]\n"
                "\tstr\tx21, [sp, #32]\n\tstp\td8, d9, [sp, #40]\n.Lk:\n\tmov\tx19, x1\n\tmov\tw20, w2\n"
                "\tadd\tx21, x21, #1\n\tfmov\td8, d0\n\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n"
                "\tldp\td8, d9, [sp, #40]\n\tldr\tx21, [sp, #32]\n\tldp\tx19, x20, [sp, #16]\n"
                "\tldp\tx29, x30, [sp], #64\n\tret\n")
        self.assertEqual(self.counts(body), {})

    def test_void_or_foreign_saves_are_scalar_traffic(self) -> None:
        # A clobbered callee-saved register stored again is not a save; a store over a live save slot voids it.
        self.assertEqual(self.counts(self.FRAME + "\tmov\tx19, x1\n\tstr\tx19, [sp, #16]\n" + self.EXIT),
                         {"scalar": {"0": 1}})
        voided = (self.FRAME + "\tstr\tx19, [sp, #16]\n\tmov\tx19, x1\n\tstr\tx9, [sp, #16]\n"
                  "\tldr\tx19, [sp, #16]\n" + self.EXIT)
        self.assertEqual(self.counts(voided), {"scalar": {"0": 2}})
        # A caller-saved register is never an ABI save, and a mixed pair is one scalar transfer.
        self.assertEqual(self.counts(self.FRAME + "\tstp\tx19, x9, [sp, #16]\n\tldp\tx19, x9, [sp, #16]\n"
                                     + self.EXIT), {"scalar": {"0": 2}})

    def test_loop_invariant_descriptor_reloads_are_caller_alias_traffic(self) -> None:
        for setup, load in (("", "ldr\tw3, [x0, #8]"), ("\tmov\tx5, x0\n", "ldr\tw3, [x5, #8]"),
                            ("\tadd\tx5, x0, #16\n", "ldrsw\tx3, [x5, #4]"),
                            ("\tadd\tx5, x0, #1, lsl #12\n", "ldr\tq3, [x5]"),
                            ("\tsub\tx5, x0, #16\n", "ldp\tw3, w4, [x5, #24]")):
            with self.subTest(load=load):
                body = (setup + ".Lk:\n\t" + load + "\n\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tret\n")
                self.assertEqual(self.counts(body), {"alias_load": {"1": 1}, "invariant_load": {"1": 1}})
        # The descriptor pointer spilled at entry and reloaded per iteration: scalar traffic plus the alias load.
        spilled = (self.FRAME + "\tstr\tx0, [sp, #16]\n.Lk:\n\tldr\tx5, [sp, #16]\n\tldr\tw3, [x5, #8]\n"
                   "\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n" + self.EXIT)
        # The pointer is redefined by the reload each iteration, but the address is the same caller-owned field:
        # alias_load is a subset of invariant_load whatever register spells the base.
        self.assertEqual(self.counts(spilled), {"scalar": {"0": 1, "1": 1}, "alias_load": {"1": 1},
                                                "invariant_load": {"1": 1}})
        # Both alternatives of a select are fixed offsets of the argument: still caller-owned and invariant.
        selected = ("\tadd\tx5, x0, #8\n\tadd\tx6, x0, #24\n\tcmp\tw2, #1\n\tcsel\tx5, x5, x6, eq\n"
                    ".Lk:\n\tldr\tw3, [x5]\n\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tret\n")
        self.assertEqual(self.counts(selected), {"alias_load": {"1": 1}, "invariant_load": {"1": 1}})

    def test_caller_alias_stores_are_counted_at_every_level(self) -> None:
        body = ("\tstr\tw3, [x0, #12]\n.Lk:\n\tmov\tx5, x0\n\tstr\tw3, [x5, #16]\n\tsub\tx2, x2, #1\n"
                "\tcbnz\tx2, .Lk\n\tret\n")
        self.assertEqual(self.counts(body), {"alias_store": {"0": 1, "1": 1}})

    def test_hoisted_and_induction_addresses_are_not_invariant_traffic(self) -> None:
        cases = {"hoisted": "\tldr\tw3, [x0, #8]\n.Lk:\n\tadd\tw4, w4, w3\n\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tret\n",
                 "post-index walk": ".Lk:\n\tldr\tq0, [x1], #16\n\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tret\n",
                 "indexed": ".Lk:\n\tldr\tw3, [x0, x4]\n\tadd\tx4, x4, #4\n\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tret\n",
                 "walked copy": "\tmov\tx5, x0\n.Lk:\n\tldr\tq0, [x5]\n\tadd\tx5, x5, #16\n\tsub\tx2, x2, #1\n"
                                "\tcbnz\tx2, .Lk\n\tret\n",
                 "computed": ".Lk:\n\tmadd\tx5, x4, x4, x0\n\tldr\tw3, [x5, #4]\n\tsub\tx2, x2, #1\n"
                             "\tcbnz\tx2, .Lk\n\tret\n"}
        for case, body in cases.items():
            with self.subTest(case=case):
                self.assertEqual(self.counts(body), {})

    def test_reloads_through_descriptor_derived_pointers_are_invariant_traffic(self) -> None:
        # R1: provenance-free. The base is not an incoming argument, but nothing in the loop defines it, so every
        # iteration reloads the same address (a pointer or span held by the descriptor, a chain of them).
        loaded = "\tldr\tx5, [x0, #8]\n.Lk:\n\tldr\tw3, [x5, #4]\n\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tret\n"
        self.assertEqual(self.counts(loaded), {"invariant_load": {"1": 1}})
        chain = ("\tldr\tx5, [x0, #8]\n\tldr\tx6, [x5, #16]\n.Lk:\n\tldr\tw3, [x6, #4]\n\tldr\tq1, [x6]\n"
                 "\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tret\n")
        self.assertEqual(self.counts(chain), {"invariant_load": {"1": 2}})
        # A constant table addressed through a hoisted page: counted, and reported in the relocated sub-bucket.
        table = ("\tadrp\tx7, 0x0 <f> !R_AARCH64_ADR_PREL_PG_HI21:.rodata\n.Lk:\n"
                 "\tldr\tq2, [x7] !R_AARCH64_LDST128_ABS_LO12_NC:.rodata+0x10\n\tsub\tx2, x2, #1\n"
                 "\tcbnz\tx2, .Lk\n\tret\n")
        self.assertEqual(self.counts(table), {"invariant_load": {"1": 1}, "invariant_reloc": {"1": 1}})
        # A walked row pointer defined by the outer loop is invariant inside the inner loop only.
        outer = ("\tmov\tx5, x1\n.Lcol:\n\tadd\tx5, x5, #64\n\tmov\tx4, x2\n.Lk:\n\tldr\tw3, [x5, #4]\n"
                 "\tsub\tx4, x4, #1\n\tcbnz\tx4, .Lk\n\tsub\tx6, x6, #1\n\tcbnz\tx6, .Lcol\n\tret\n")
        self.assertEqual(self.counts(outer), {"invariant_load": {"2": 1}})

    def test_chains_reloaded_inside_the_loop_are_invariant_transitively(self) -> None:
        # R1b: a base the loop defines only by invariant loads (or constants, moves, add/sub of a constant, csel of
        # invariant registers) is itself invariant, to a fixed point.
        def invariant(body: str) -> dict:
            return {kind: values for kind, values in self.counts(body).items() if kind.startswith("invariant")}
        loop = lambda inner: ".Lk:\n" + inner + "\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tret\n"
        two = "\tldr\tx10, [x9, #8]\n\tldr\tw3, [x10, #4]\n"
        three = "\tldr\tx10, [x9, #8]\n\tldr\tx11, [x10, #16]\n\tldr\tw3, [x11, #4]\n"
        self.assertEqual(invariant(loop(two)), {"invariant_load": {"1": 2}})
        self.assertEqual(invariant(loop(three)), {"invariant_load": {"1": 3}})
        # The reviewer's example through `this`: both loads count, the first is also an alias_load.
        self.assertEqual(self.counts(loop("\tldr\tx5, [x0, #8]\n\tldr\tw3, [x5, #4]\n")),
                         {"alias_load": {"1": 1}, "invariant_load": {"1": 2}})
        # Moves, add/sub of constants (incl. relocated low-12), csel of invariant registers, adrp constants.
        derived = ("\tldr\tx10, [x9, #8]\n\tldr\tx12, [x9, #16]\n\tcmp\tw2, #3\n\tcsel\tx11, x10, x12, eq\n"
                   "\tadd\tx14, x11, #32\n\tmov\tx13, x14\n\tldr\tw3, [x13, #4]\n")
        self.assertEqual(invariant(loop(derived)), {"invariant_load": {"1": 3}})
        table = ("\tadrp\tx10, 0x0 <f> !R_AARCH64_ADR_PREL_PG_HI21:.rodata\n"
                 "\tadd\tx11, x10, #0 !R_AARCH64_ADD_ABS_LO12_NC:.rodata+0x10\n\tldr\tq2, [x11, #16]\n")
        self.assertEqual(invariant(loop(table)), {"invariant_load": {"1": 1}})

    def test_transitive_invariance_is_never_self_justified(self) -> None:
        def invariant(body: str) -> dict:
            return {kind: values for kind, values in self.counts(body).items() if kind.startswith("invariant")}
        loop = lambda inner: ".Lk:\n" + inner + "\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tret\n"
        # The middle register of a three-hop chain is also walked, or is loaded through an index: the hops after
        # it vary (only the first, invariant load counts).
        walked = ("\tldr\tx10, [x9, #8]\n\tldr\tx11, [x10, #16]\n\tldr\tw3, [x11, #4]\n"
                  "\tadd\tx10, x10, #64\n")
        self.assertEqual(invariant(loop(walked)), {"invariant_load": {"1": 1}})
        indexed = "\tldr\tx10, [x9, x4]\n\tldr\tx11, [x10, #16]\n\tldr\tw3, [x11, #4]\n\tadd\tx4, x4, #8\n"
        self.assertEqual(invariant(loop(indexed)), {})
        # A walk spelled through two registers, and a register loaded through itself, never prove themselves.
        self.assertEqual(invariant(loop("\tmov\tx10, x11\n\tadd\tx11, x10, #16\n\tldr\tw3, [x10]\n")), {})
        self.assertEqual(invariant(loop("\tldr\tx10, [x10, #8]\n\tldr\tw3, [x10]\n")), {})
        # A frame-slot reload never makes a register invariant (the reload itself is scalar traffic).
        frame = ("\tstp\tx29, x30, [sp, #-32]!\n\tstr\tx9, [sp, #16]\n.Lk:\n\tldr\tx10, [sp, #16]\n"
                 "\tldr\tw3, [x10, #4]\n\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tldp\tx29, x30, [sp], #32\n\tret\n")
        self.assertEqual(self.counts(frame), {"scalar": {"0": 1, "1": 1}})

    def test_transitive_invariance_is_attributed_per_loop(self) -> None:
        # Invariant in the outer loop but walked by the inner loop: the chain behind it does not count in the
        # outer loop (only its invariant first hop does).
        walked = (".Lcol:\n\tldr\tx10, [x9, #8]\n\tldr\tw3, [x10, #4]\n.Lk:\n\tadd\tx10, x10, #16\n"
                  "\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tsub\tx12, x12, #1\n\tcbnz\tx12, .Lcol\n\tret\n")
        self.assertEqual(self.counts(walked), {"invariant_load": {"1": 1}})
        # Walked by the outer loop, reloaded through in the inner loop: invariant in the inner loop only.
        inner = ("\tmov\tx9, x1\n.Lcol:\n\tldr\tx10, [x9, #8]\n.Lk:\n\tldr\tx11, [x10, #16]\n\tldr\tw3, [x11]\n"
                 "\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tadd\tx9, x9, #64\n\tsub\tx12, x12, #1\n"
                 "\tcbnz\tx12, .Lcol\n\tret\n")
        self.assertEqual(self.counts(inner), {"invariant_load": {"2": 2}})

    def test_bases_that_change_inside_the_loop_are_not_invariant_traffic(self) -> None:
        cases = {"redefined": ".Lk:\n\tldr\tx9, [x10, x11, lsl #3]\n\tldr\tw3, [x9, #4]\n\tadd\tx11, x11, #1\n"
                              "\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tret\n",
                 "redefined in a nested loop": ".Lcol:\n\tldr\tw3, [x9, #4]\n.Lk:\n\tadd\tx9, x9, #16\n"
                                               "\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tsub\tx12, x12, #1\n"
                                               "\tcbnz\tx12, .Lcol\n\tret\n",
                 "redefined through a w view": ".Lk:\n\tldr\tw3, [x9]\n\tmov\tw9, w3\n\tsub\tx2, x2, #1\n"
                                               "\tcbnz\tx2, .Lk\n\tret\n",
                 "pair load defines the base": ".Lk:\n\tldr\tw3, [x9, #4]\n\tldp\tx11, x9, [x10], #16\n"
                                               "\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tret\n",
                 "indexed": "\tldr\tx9, [x0, #8]\n.Lk:\n\tldr\tw3, [x9, x11]\n\tsub\tx2, x2, #1\n"
                            "\tcbnz\tx2, .Lk\n\tret\n",
                 "post-index": ".Lk:\n\tldr\tq0, [x9], #16\n\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tret\n",
                 "pre-index writeback": ".Lk:\n\tldr\tq0, [x9, #16]!\n\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tret\n",
                 "writeback elsewhere": ".Lk:\n\tldr\tw3, [x9, #4]\n\tstr\td0, [x9], #8\n\tsub\tx2, x2, #1\n"
                                        "\tcbnz\tx2, .Lk\n\tret\n",
                 "outside every loop": "\tldr\tx9, [x0, #8]\n\tldr\tw3, [x9, #4]\n\tret\n"}
        for case, body in cases.items():
            with self.subTest(case=case):
                self.assertEqual({kind: values for kind, values in self.counts(body).items()
                                  if kind.startswith("invariant")}, {})

    def test_nested_loops_attribute_to_the_innermost(self) -> None:
        body = (self.FRAME + ".Lcol:\n\tldr\tw7, [x0, #4]\n.Lk:\n\tstr\tw9, [sp, #16]\n\tldr\tw3, [x0, #8]\n"
                "\tldr\tw9, [sp, #16]\n\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tsub\tx6, x6, #1\n\tcbnz\tx6, .Lcol\n"
                + self.EXIT)
        counts, loops, errors = traffic(body)
        self.assertEqual(errors, [])
        self.assertEqual([(loop["id"], loop["parent"], loop["depth"]) for loop in loops], [(1, 0, 1), (2, 1, 2)])
        self.assertEqual({kind: values for kind, values in counts.items() if values},
                         {"alias_load": {"1": 1, "2": 1}, "invariant_load": {"1": 1, "2": 1}, "scalar": {"2": 2}})

    def test_a_loop_entered_at_two_blocks_is_one_attributed_loop(self) -> None:
        # GCC can enter a rotated loop both at its top and at its latch test (an irreducible region).
        body = (self.FRAME + ".Lcol:\n\tcbz\tx3, .Lb\n.La:\n\tldr\tw4, [x0, #4]\n.Lb:\n\tstr\tw9, [sp, #16]\n"
                "\tsub\tx2, x2, #1\n\tcbnz\tx2, .La\n\tldr\tw5, [x0, #8]\n\tsub\tx6, x6, #1\n\tcbnz\tx6, .Lcol\n"
                + self.EXIT)
        counts, loops, errors = traffic(body)
        self.assertEqual(errors, [])
        self.assertEqual([(loop["id"], loop["parent"], loop["depth"], len(loop["entries"])) for loop in loops],
                         [(1, 0, 1, 1), (2, 1, 2, 2)])
        self.assertEqual({kind: values for kind, values in counts.items() if values},
                         {"alias_load": {"1": 1, "2": 1}, "invariant_load": {"1": 1, "2": 1}, "scalar": {"2": 1}})

    def test_fp_spills_are_attributed_to_their_loop(self) -> None:
        body = (self.FRAME + "\tstr\tq16, [sp, #16]\n.Lk:\n\tmov\tx9, v1.d[0]\n\tstr\tx9, [sp, #32]\n"
                "\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n" + self.EXIT)
        self.assertEqual(self.counts(body), {"fp": {"0": 1, "1": 1}, "scalar": {"1": 1}})


class LoopNamingTest(unittest.TestCase):
    """Loops are named by the source locations the executed compile's assembly gives their entries and latches."""

    KEY = (2, 0, 0, 0, 0, 0)
    LOOP = "\tmov\tx5, x0\n.Lk:\n\tldr\tw3, [x5, #8]\n\tsub\tx2, x2, #1\n\tcbnz\tx2, .Lk\n\tret\n"

    def named(self, text_edit=None) -> dict:
        functions = programs({self.KEY: self.LOOP})
        obj, listing = witness_object(functions), disassembly(functions)
        traffic: dict = {}
        CHECKER.audit(obj, listing, "NEON", traffic)
        text = assembly(functions)
        target = name(*self.KEY) + ":\n"
        text = text.replace(target, target + '\t.file 3 "/e/trees/x/src/v2/kernels/K.h"\n\t.loc 3 10 1\n', 1)
        text = text.replace("\tldr\tw3, [x5, #8]", "\t.loc 3 12 5\n\tldr\tw3, [x5, #8]", 1)
        text = text.replace("\tcbnz\tx2, .Lk", "\t.loc 3 11 3\n\tcbnz\tx2, .Lk", 1)
        if text_edit:
            text = text_edit(text)
        return CHECKER.name_loops(traffic, obj, listing, text)[":".join(map(str, self.KEY))]

    def test_entries_and_latches_take_the_assembly_locations(self) -> None:
        record = self.named()
        self.assertEqual(record["counts"]["alias_load"], {"1": 1})
        (loop,) = record["loops"]
        self.assertEqual((loop["entry_sources"], loop["latch_sources"]),
                         (["src/v2/kernels/K.h:12"], ["src/v2/kernels/K.h:11"]))

    def test_an_assembly_that_is_not_the_objects_cannot_name_loops(self) -> None:
        with self.assertRaisesRegex(ValueError, "loops cannot be named"):
            self.named(lambda text: text.replace("\tsub\tx2, x2, #1", "\tsub\tx2, x2, #1\n\tmov\tx9, x9", 1))


class OrderedAddCountTest(unittest.TestCase):
    """Site counts per GPU-aligned program must equal the committed ones exactly, in both directions."""

    KEY = ":".join(map(str, (2, 0, 1, 0, 0, 0)))

    def sites(self, count: int = 1) -> dict[str, list[int]]:
        return {key: list(range(count)) for key in ordered_sites()["GNU-14"]["NEON"]}

    def test_exact_counts_pass(self) -> None:
        CHECKER.require_ordered_adds(self.sites(), "GNU-14", "NEON", ordered_sites())

    def test_one_site_lost_added_or_moved_between_programs_fails(self) -> None:
        cases = {"lost": dict(self.sites(), **{self.KEY: []}),
                 "added": dict(self.sites(), **{self.KEY: [0, 7]}),
                 "program dropped": {key: value for key, value in self.sites().items() if key != self.KEY},
                 "backend-native program named": dict(self.sites(), **{"2:0:0:0:0:0": [3]})}
        for case, sites in cases.items():
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, "ordered-add site counts differ"):
                    CHECKER.require_ordered_adds(sites, "GNU-14", "NEON", ordered_sites())

    def test_unknown_identity_or_envelope_fails_closed(self) -> None:
        for identity, mode in (("GNU-15", "NEON"), ("GNU-14", "NEON_I8MM"), ("AppleClang-21", "NEON")):
            with self.assertRaisesRegex(ValueError, "no committed ordered-add site counts"):
                CHECKER.require_ordered_adds(self.sites(), identity, mode, ordered_sites())


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

    def test_pinned_panel_widths_have_no_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-installed panel width"):
            CHECKER.program_key(name(1, 1, 0, 1, 0, 1, strips=2))
        self.assertEqual(CHECKER.program_key(name(1, 1, 0, 1, 0, 1)), "1:1:0:1:0:1")

    def test_envelope_inventory_matches_the_written_out_keys(self) -> None:
        for macho in (False, True):
            counts = audit_witness(macho=macho)
            self.assertEqual(sorted(tuple(map(int, CHECKER.program_key(n).split(":"))) for n in counts), NEON_KEYS)
        self.assertEqual(len(NEON_KEYS), 96)

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
        for retired in ("6strips", "5panel"):
            with self.subTest(retired=retired):
                with self.assertRaisesRegex(ValueError, "unrecognized tile program"):
                    audit_witness(extra=[(name(2, 0, 0, 1, 0, 1).replace("3run", retired), "\tret\n")])
        with self.assertRaisesRegex(ValueError, "outside the NEON inventory"):
            audit_witness(extra=[(name(1, 0, 0, 1, 0, 1, strips=2), "\tret\n")])  # A pinned tournament width.

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

    def test_coded_programs_are_audited_and_every_tile_member_is_admitted(self) -> None:
        coded = (2, 0, 1, 3, 11, 0)
        # A runCoded program is an audited tile program: its spills count, its structure is checked.
        counts = audit_witness({coded: "\tstr q0, [sp, #-16]!\n\tfadd\tv0.4s, v0.4s, v1.4s\n\tadd sp, sp, #16\n\tret"})
        self.assertEqual(counts[name(*coded)], 1)
        self.assertEqual(CHECKER.program_key(name(*coded)), "2:0:1:3:11:0")
        self.assertEqual(CHECKER.program_key(name(1, 0, 0, 3, 7, 1)), "1:0:0:3:7:1")
        with self.assertRaisesRegex(ValueError, "structurally"):
            audit_witness({coded: "\tbl helper\n\tret"}, extra=[("helper", "\tret\n")])
        with self.assertRaisesRegex(ValueError, "missing"):
            audit_witness(keys=[key for key in NEON_KEYS if key != coded])
        # Any other out-of-line tile member fails closed: an unlisted member, a pinned coded width, a codebook
        # the dispatch does not route to runCoded.
        member = "_ZNK9llaminar23cpu16native_quantised27CPUNativeNEONProjectionTileILm2EE6decodeEv"
        with self.assertRaisesRegex(ValueError, "outside the audited programs"):
            audit_witness(extra=[(member, "\tret")])
        with self.assertRaisesRegex(ValueError, "outside the NEON inventory"):
            audit_witness(extra=[(name(2, 0, 1, 3, 11, 0, 2), "\tret")])
        with self.assertRaisesRegex(ValueError, "outside the NEON inventory"):
            audit_witness(extra=[(name(2, 0, 1, 3, 9, 0), "\tret")])

    def test_dispatchers_branch_only_into_audited_programs_of_their_tile(self) -> None:
        base = [(dispatcher(rows), DISPATCHER_BODY) for rows in (1, 3, 4)]
        def run(body: str, extra: list | None = None) -> dict:
            return CHECKER.audit(witness_object(programs(dispatchers=False) + base + [(dispatcher(2), body)] + (extra or [])),
                                 disassembly(programs(dispatchers=False) + base + [(dispatcher(2), body)] + (extra or [])),
                                 "NEON")
        self.assertEqual(len(run(f"\tcbz\tw0, .La\n\tb\t{name(2, 0, 0, 0, 0, 0)}\n.La:\n\tb\t{name(2, 0, 1, 3, 7, 1)}")), 96)
        self.assertEqual(len(run("\tbl\tcompactKernel\n\tret", [("compactKernel", "\tret")])), 96)
        with self.assertRaisesRegex(ValueError, "unaudited tile member"):
            run(f"\tb\t{name(3, 0, 0, 0, 0, 0)}")  # another tile's program
        with self.assertRaisesRegex(ValueError, "holds a loop"):
            run("\t.La:\n\tsubs\tw0, w0, #1\n\tb.ne\t.La\n\tret".replace("\t.La:", ".La:"))
        with self.assertRaisesRegex(ValueError, "do not match the NEON inventory"):
            CHECKER.audit(witness_object(programs(dispatchers=False) + base), disassembly(programs(dispatchers=False) + base),
                          "NEON")
        with self.assertRaisesRegex(ValueError, "outside the NEON inventory or duplicated"):
            CHECKER.audit(witness_object(programs() + [(dispatcher(2, 1), DISPATCHER_BODY)]),
                          disassembly(programs() + [(dispatcher(2, 1), DISPATCHER_BODY)]), "NEON")

    def test_structural_failure_is_not_paid_from_a_budget(self) -> None:
        key = (2, 0, 0, 1, 0, 0)
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

    def test_q4k_coded_program_cannot_exceed_the_previous_identity_nibble_ceiling(self) -> None:
        counts = self.counts(0, 0)
        counts[name(3, 0, 0, 6, 1, 1)] = 5
        with self.assertRaisesRegex(ValueError, "3:0:0:6:1:1 spills 5 > budget 4"):
            CHECKER.require_registers(counts, "GNU-14", "NEON", budget())

    def test_native_q5_uses_its_own_grouped_bound(self) -> None:
        limits = budget()
        limits["GNU-14"]["NEON"]["3:0:1:7:1:1"] = 2
        counts = self.counts(0, 0)
        q5 = name(3, 0, 1, 7, 1, 1)
        counts[q5] = 2
        CHECKER.require_registers(counts, "GNU-14", "NEON", limits)
        counts[q5] = 3
        with self.assertRaisesRegex(ValueError, "3:0:1:7:1:1 spills 3 > budget 2"):
            CHECKER.require_registers(counts, "GNU-14", "NEON", limits)

    def test_unknown_identity_envelope_missing_and_stale_budget_fail_closed(self) -> None:
        for identity, mode in (("GNU-15", "NEON"), ("GNU-14", "NEON_I8MM")):
            with self.assertRaisesRegex(ValueError, "no qualified"):
                CHECKER.require_registers(self.counts(0, 0), identity, mode, budget())
        missing = budget()
        missing["GNU-14"]["NEON"].pop("2:0:0:1:0:0")
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
                # GNU-14 also bounds the coded runCoded programs (develop's actual Q5_K/IQ3_S counts); AppleClang has
                # no coded ceilings yet, so a coded program fails closed there ("has no budget").
                coded = {"NEON": 12, "NEON_DOTPROD": 24, "NEON_I8MM": 36}[mode] if identity == "GNU-14" else 0
                self.assertEqual(len(limits), {"NEON": 54, "NEON_DOTPROD": 108, "NEON_I8MM": 162}[mode] + coded)
                self.assertEqual(sum(key.split(":")[3] == "3" for key in limits), coded)
                self.assertTrue(all(not key.startswith("1:") and value >= 0 for key, value in limits.items()))

    def test_codegen_profile_lists_name_audited_programs(self) -> None:
        """kUnscheduledPrograms/kUnthreadedPrograms/kOpaqueStorePrograms: well-formed unique keys of programs the
        witness audits."""
        source = (REPO / "src/v2/kernels/cpu/gemm/CPUNativeNEONPreparedProjection.h").read_text()
        audited = set()
        for mode, entries in CHECKER.ENVELOPE_PROGRAMS.items():
            for entry in entries:
                rows, dot = entry.split(":")
                leaf = 1 if dot == "I8MM" and rows == "1" else CHECKER.DOTS[dot]
                for policy in (0, 1):
                    for encoding, nibble_map, minimum in CHECKER.DISPATCH:
                        audited.add(f"{rows}:{leaf}:{policy}:{encoding}:{nibble_map}:{minimum}")
                    for codebook, minimum in CHECKER.CODED_DISPATCH:
                        audited.add(f"{rows}:{leaf}:{policy}:{CHECKER.CODED_ENCODING}:{codebook}:{minimum}")
        for name in ("kUnscheduledPrograms", "kUnthreadedPrograms", "kOpaqueStorePrograms"):
            with self.subTest(name=name):
                match = re.search(name + r"\[\] = \{(.*?)\};", source, re.S)
                self.assertIsNotNone(match)
                body = re.sub(r"//[^\n]*", "", match.group(1))
                keys = re.findall(r'"([^"]*)"', body)
                self.assertTrue(keys)
                self.assertEqual(re.sub(r'"[^"]*"|[\s,]', "", body), "")
                self.assertEqual(len(keys), len(set(keys)))
                self.assertEqual(sorted(key for key in keys if key not in audited), [])
                if name == "kUnthreadedPrograms":
                    self.assertFalse(any(key.split(":")[3] == str(CHECKER.CODED_ENCODING) for key in keys))

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
    default = (f" {{here}}/cc1plus -quiet -std=c++20 -MD {{value('-MF')}} -MF {{value('-MF')}} -MT probe -H {{source}} "
               f"-o {{temporary}}\\n as -EL -march=armv8-a -o {{value('-o')}} {{temporary}}\\n")
    jobs = setting("jobs.txt", default).format(here=here, source=source, obj=value("-o"), deps=value("-MF"))
    sys.stderr.write("Using built-in specs.\\nCOMPILER_PATH=" + str(here) + "\\n" + jobs)
    sys.exit(0)
(here / "invocations.txt").open("a").write(" ".join(args) + "\\n")
if "-S" in args:
    sys.stderr.write("the witness must not compile the probe a second time\\n"); sys.exit(1)
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
FAKE_CC1PLUS = """#!{python} -SE
# {tag}
import pathlib, sys
here = pathlib.Path(__file__).parent
args = sys.argv[1:]
# The compiler proper of the plan: writes the probe assembly to its temporary output and the dependencies to -MF.
source = here / ("replay-assembly.txt" if (here / "replay-assembly.txt").exists() else "assembly.txt")
pathlib.Path(args[args.index("-o") + 1]).write_text(source.read_text())
if "-MF" in args:
    name = "deps.c.txt" if (here / "deps.c.txt").exists() else "deps.txt"
    deps = (here / name).read_text().strip().split(" ")
    pathlib.Path(args[args.index("-MF") + 1]).write_text(args[args.index("-MT") + 1] + ": " + " ".join(deps) + "\\n")
"""
FAKE_AS = """#!{python} -SE
# {tag}
import pathlib, sys
here = pathlib.Path(__file__).parent
args = sys.argv[1:]
# A deterministic stand-in assembler: the fixture's assembly yields the fixture object, other text another object.
text = pathlib.Path(args[-1]).read_text()
data = (here / "witness.o").read_bytes()
if text != (here / "assembly.txt").read_text():
    data += b"\\0reassembled from other assembly"
pathlib.Path(args[args.index("-o") + 1]).write_bytes(data)
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
        self.job_programs()
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
        self.original_sites = CHECKER.ORDERED_ADD_SITES_PATH
        CHECKER.ORDERED_ADD_SITES_PATH = root / "sites.json"
        CHECKER.ORDERED_ADD_SITES_PATH.write_text(json.dumps(ordered_sites()))
        self.original_chains = CHECKER.CHAIN_COUNTS_PATH
        CHECKER.CHAIN_COUNTS_PATH = root / "chains.json"
        CHECKER.CHAIN_COUNTS_PATH.write_text(json.dumps(chain_counts()))

    def tearDown(self) -> None:
        CHECKER.BUDGET_PATH = self.original_budget
        CHECKER.ORDERED_ADD_SITES_PATH = self.original_sites
        CHECKER.CHAIN_COUNTS_PATH = self.original_chains
        self.temporary.cleanup()

    def job_programs(self, tag: str = "A") -> None:
        """Working stand-ins for the plan's compiler proper and assembler (the witness replays them)."""
        for program, text in (("cc1plus", FAKE_CC1PLUS), ("as", FAKE_AS)):
            path = self.tool / program
            if path.is_symlink():
                path.unlink()
            path.write_text(text.format(python=sys.executable, tag=f"{program} {tag}"))
            path.chmod(0o755)

    def witness(self, functions: list[tuple[str, str]]) -> None:
        (self.tool / "witness.o").write_bytes(witness_object(functions))
        (self.tool / "disassembly.txt").write_text(disassembly(functions))
        (self.tool / "assembly.txt").write_text(assembly(functions))

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

    def test_ordered_add_sites_are_bound_and_reverified(self) -> None:
        CHECKER.build(self.args)
        self.assertEqual(set(self.proof()["ordered_adds"]),
                         {":".join(map(str, key)) for key in NEON_KEYS if key[2] == 1})
        retained = Path(self.args.proof).with_suffix(".asm")
        original = retained.read_text()
        retained.write_text(original.replace("v0.4s, v0.4s, v1.4s", "v0.4s, v1.4s, v0.4s", 1))
        with self.assertRaisesRegex(ValueError, "probe assembly"):
            CHECKER.verify(self.args)
        retained.write_text(original)
        CHECKER.verify(self.args)
        (self.tool / "assembly.txt").write_text(assembly(programs(), ordered=False))
        with self.assertRaisesRegex(ValueError, "without an ordered add"):
            CHECKER.build(self.args)

    def test_retained_assembly_is_the_executed_plans_assembler_input(self) -> None:
        CHECKER.build(self.args)
        retained = Path(self.args.proof).with_suffix(".asm")
        self.assertEqual(retained.read_text(), (self.tool / "assembly.txt").read_text())
        self.assertNotIn(" -S ", (self.tool / "invocations.txt").read_text())
        # An equal-length program that differs only at a non-site instruction is another program: the plan's
        # assembler does not reproduce the bound object from it.
        text = (self.tool / "assembly.txt").read_text()
        other = text.replace("\tsdot\tv0.4s, v1.16b, v2.16b", "\tsdot\tv3.4s, v1.16b, v2.16b", 1)
        self.assertNotEqual(other, text)
        (self.tool / "replay-assembly.txt").write_text(other)
        with self.assertRaisesRegex(ValueError, "does not reproduce the bound object"):
            CHECKER.build(self.args)
        self.assertFalse(Path(self.args.proof).exists())
        (self.tool / "replay-assembly.txt").unlink()
        # A replayed assembler that writes nothing cannot leave the driver's object standing.
        (self.tool / "as").write_text("#!/bin/sh\nexit 0\n")
        with self.assertRaisesRegex(ValueError, "does not reproduce the bound object"):
            CHECKER.build(self.args)
        self.job_programs()
        CHECKER.build(self.args)
        CHECKER.verify(self.args)

    def test_integrated_clang_plan_has_no_assembler_input_to_bind(self) -> None:
        obj, deps = self.root / "probe.o", self.root / "probe.deps"
        obj.write_bytes(b"o")
        deps.write_text("probe: x\n")
        plan = {"jobs": [["/usr/bin/clang", "-cc1", "-emit-obj", "-o", str(obj), "probe.cpp"]],
                "program_dirs": [], "path": []}
        with self.assertRaisesRegex(ValueError, "no assembler input to retain"):
            CHECKER.replay_plan(plan, str(self.root), obj, deps, self.root / "probe.asm")
        self.assertEqual(obj.read_bytes(), b"o")

    def test_committed_site_counts_are_exact_and_bound(self) -> None:
        CHECKER.build(self.args)
        CHECKER.verify(self.args)
        sites = CHECKER.ORDERED_ADD_SITES_PATH
        original = sites.read_text()
        sites.write_text(json.dumps(ordered_sites(2)))
        with self.assertRaisesRegex(ValueError, "stale tile register tool"):
            CHECKER.verify(self.args)
        with self.assertRaisesRegex(ValueError, "ordered-add site counts differ"):
            CHECKER.build(self.args)
        sites.write_text(original)
        CHECKER.build(self.args)
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
        cc1 = f" {here}/cc1plus -quiet -std=c++20 -MD {{deps}} -MF {{deps}} -MT probe -H {{source}} -o /tmp/ccTEST01.s"
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
            (self.tool / "versions" / version).write_text(FAKE_AS.format(python=sys.executable,
                                                                         tag=f"assembler {version}"))
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
                self.job_programs()
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
            f"write {self.tool / 'jobs.txt'} replanned\n {{here}}/cc1plus -quiet -std=c++20 -g -MD {{deps}} -MF {{deps}} -MT probe -H "
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
                "-g", "-O3", "-dumpdir", "/tmp/", "-dumpbase", "proof.cpp", "-o", "/tmp/ccAb12Cd.s", "-H", "-std=c++20"],
               ["as", "-I", "/w/src", "--gdwarf-5", "-EL", "-march=armv8-a", "-mabi=lp64", "-o", "/w/out/proof.o",
                "/tmp/ccAb12Cd.s"]]
        CHECKER.validate_plan([clang], "AppleClang-21", paths)
        CHECKER.validate_plan(gnu, "GNU-14", paths)
        CHECKER.validate_plan([gnu[0] + ["-fPIC", "-fopenmp"], gnu[1]], "GNU-14", paths)
        for unsupported in ("-funknown-witness", "-fPIC=1", "-fopenmp=libomp", "-fno-openmp"):
            with self.subTest(unsupported=unsupported), self.assertRaises(ValueError):
                CHECKER.validate_plan([gnu[0] + [unsupported], gnu[1]], "GNU-14", paths)
        for prior in ("gnu++23", "c++23"):
            CHECKER.validate_plan([gnu[0][:-1] + ["-std=" + prior, "-std=c++20"], gnu[1]], "GNU-14", paths)
        for dialects in (("c++20", "gnu++23"), ("gnu++23",), (), ("c++26", "c++20")):
            with self.subTest(dialects=dialects), self.assertRaises(ValueError):
                CHECKER.validate_plan([gnu[0][:-1] + ["-std=" + d for d in dialects], gnu[1]], "GNU-14", paths)
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

    SPILL = {(1, 0, 0, 0, 0, 0): "\tstr q0, [sp, #-16]!\n\tadd sp, sp, #16\n\tret"}

    def unqualified(self) -> Path:
        """A policy-red build: the structurally complete analysis is written as the unqualified record only."""
        CHECKER.build(self.args)
        self.witness(programs(self.SPILL))
        with self.assertRaisesRegex(ValueError, "admission failed .*unqualified analysis record written.*single-row"):
            CHECKER.build(self.args)
        self.assertFalse(Path(self.args.proof).exists())  # The earlier seal is retired, never left standing.
        return CHECKER.analysis_path(Path(self.args.proof))

    def test_admission_failure_writes_only_a_typed_unqualified_record(self) -> None:
        record_path = self.unqualified()
        record = json.loads(record_path.read_text())
        self.assertEqual((record["kind"], record["schema"], record["admission"]["status"], record["diagnostic"]),
                         ("unqualified-analysis", "12-unqualified-analysis", "failed", True))
        self.assertEqual([(failure["reason"], failure["programs"]) for failure in record["admission"]["failures"]],
                         [("spill", ["1:0:0:0:0:0"])])
        self.assertEqual(CHECKER.check_analysis_record(self.args)["kind"], "unqualified-analysis")
        with self.assertRaisesRegex(ValueError, "unqualified analysis record lies next to the proof"):
            CHECKER.verify(self.args)

    def test_records_and_seals_never_masquerade(self) -> None:
        record_path = self.unqualified()
        proof = Path(self.args.proof)
        record_path.rename(proof)  # A renamed record is no seal.
        with self.assertRaisesRegex(ValueError, "not a sealed proof"):
            CHECKER.verify(self.args)
        proof.unlink()
        self.witness(programs())
        CHECKER.build(self.args)
        with self.assertRaisesRegex(ValueError, "never consumed as an unqualified analysis record"):
            CHECKER.check_analysis_record(self.args)  # A sealed run has no record to consume.
        CHECKER.analysis_path(proof).write_text(proof.read_text())
        with self.assertRaisesRegex(ValueError, "both present"):
            CHECKER.check_analysis_record(self.args)
        with self.assertRaisesRegex(ValueError, "lies next to the proof"):
            CHECKER.verify(self.args)
        proof.unlink()  # A seal under the record's name.
        with self.assertRaisesRegex(ValueError, "not an unqualified analysis record"):
            CHECKER.check_analysis_record(self.args)

    def test_status_failures_fields_and_bindings_are_required(self) -> None:
        record_path = self.unqualified()
        original = json.loads(record_path.read_text())
        edits = {
            "no failed admission status": lambda r: r["admission"].pop("status"),
            "no failed admission status ": lambda r: r["admission"].update(status="passed"),
            "no typed admission failures": lambda r: r["admission"].update(failures=[]),
            "no typed admission failures ": lambda r: r["admission"]["failures"][0].update(reason="other"),
            "lacks fields": lambda r: r.pop("stack_traffic"),
            "not the recomputed ones": lambda r: r["admission"]["failures"].append(
                {"reason": "chain_count", "programs": [], "detail": "invented"}),
            "stale tile register tool": lambda r: r.update(checker_sha256="0" * 64),
            "tile register evidence changed": lambda r: r["program_violations"].update(
                {next(iter(r["program_violations"])): 99}),
        }
        for fragment, edit in edits.items():
            with self.subTest(fragment=fragment):
                record = json.loads(json.dumps(original))
                edit(record)
                record_path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, fragment.strip()):
                    CHECKER.check_analysis_record(self.args)
        record_path.write_text(json.dumps(original))
        CHECKER.check_analysis_record(self.args)
        # Tampered object, source and committed-count file.
        obj = Path(self.args.proof).with_suffix(".o")
        data = obj.read_bytes()
        obj.write_bytes(data + b"\0")
        with self.assertRaisesRegex(ValueError, "object is missing or changed"):
            CHECKER.check_analysis_record(self.args)
        obj.write_bytes(data)
        self.real.write_text("// accessor changed\n")
        with self.assertRaisesRegex(ValueError, "stale tile register input"):
            CHECKER.check_analysis_record(self.args)
        self.real.write_text("// accessor\n")
        CHECKER.CHAIN_COUNTS_PATH.write_text(json.dumps(chain_counts()) + " ")
        with self.assertRaisesRegex(ValueError, "stale tile register tool"):
            CHECKER.check_analysis_record(self.args)

    def test_an_admissible_analysis_is_never_recorded_as_unqualified(self) -> None:
        # A seal relabelled as a failed record (a forged admission) does not pass as unqualified input.
        CHECKER.build(self.args)
        proof = Path(self.args.proof)
        record = json.loads(proof.read_text())
        record.update(kind="unqualified-analysis", schema="12-unqualified-analysis", diagnostic=True,
                      admission={"status": "failed", "failures": [{"reason": "spill", "programs": ["1:0:0:0:0:0"],
                                                                    "detail": "forged"}]})
        CHECKER.analysis_path(proof).write_text(json.dumps(record))
        proof.unlink()
        with self.assertRaisesRegex(ValueError, "the analysis is admissible"):
            CHECKER.check_analysis_record(self.args)

    def test_structural_failure_writes_no_record_and_nothing_is_bound(self) -> None:
        CHECKER.build(self.args)
        self.witness(programs({(1, 0, 0, 0, 0, 0): "\tfrobnicate x9, x1\n\tret"}))
        with self.assertRaisesRegex(ValueError, "structurally unprovable"):
            CHECKER.build(self.args)
        self.assertFalse(Path(self.args.proof).exists() or CHECKER.analysis_path(Path(self.args.proof)).exists())
        with self.assertRaisesRegex(ValueError, "stage 1 failed structurally"):
            CHECKER.fused_stage(self.args)
        receipt = json.loads(Path(self.args.proof).with_name("fused-stage.json").read_text())
        self.assertEqual((receipt["status"], receipt["input"]), ("failed", None))

    def test_fused_stage_reads_only_this_runs_artifacts(self) -> None:
        CHECKER.build(self.args)
        CHECKER.fused_stage(self.args)
        receipt = json.loads(Path(self.args.proof).with_name("fused-stage.json").read_text())
        self.assertEqual((receipt["status"], receipt["input"], receipt["verified"]), ("passed", "sealed-proof", True))
        obj = Path(self.args.proof).with_suffix(".o")
        obj.write_bytes(obj.read_bytes() + b"\0")  # A prior run's (or foreign) object.
        with self.assertRaisesRegex(ValueError, "is not the artifact this run's sealed-proof binds"):
            CHECKER.fused_stage(self.args)
        self.unqualified()
        CHECKER.fused_stage(self.args)
        receipt = json.loads(Path(self.args.proof).with_name("fused-stage.json").read_text())
        self.assertEqual((receipt["input"], receipt["verified"]), ("unqualified-analysis", False))
        self.real.write_text("// accessor changed\n")
        with self.assertRaisesRegex(ValueError, "closure input changed"):
            CHECKER.fused_stage(self.args)



class LineWitnessTest(unittest.TestCase):
    """Release line witness (L-1..L-4): a Release argv without debug information gets a second compile with
    exactly one appended `-g`, whose object must be byte-identical in every allocated section and relocation."""

    RELEASE = ["/usr/bin/c++", "-O3", "-DNDEBUG", "-std=c++20", "-c", "/p/probe.cpp", "-o", "/w/proof.o",
               "-MD", "-MF", "/w/proof.deps", "-MT", "probe", "-H"]
    OUTPUTS = (Path("/w/proof.o"), Path("/w/proof.deps"), Path("/w/proof.lines.o"), Path("/w/proof.lines.deps"))
    RENAMED = {"/w/proof.o": "/w/proof.lines.o", "/w/proof.deps": "/w/proof.lines.deps"}
    CALL = [("f", "\tadrp x1, 0x0 <f> !R_AARCH64_ADR_PREL_PG_HI21:.rodata+0x8\n"
                  "\tadd x1, x1, #0x0 !R_AARCH64_ADD_ABS_LO12_NC:.rodata+0x8\n\tbl helper\n\tret\n")]

    def test_only_an_argv_without_debug_information_needs_the_witness(self) -> None:
        self.assertTrue(CHECKER.line_witness_needed(self.RELEASE))
        for flag in ("-g", "-g1", "-g3", "-ggdb", "-gdwarf-5", "-gline-tables-only"):
            with self.subTest(flag=flag):
                # A pre-existing debug level is analysed directly: the witness flag is never doubled.
                self.assertFalse(CHECKER.line_witness_needed(self.RELEASE + [flag]))
        with self.assertRaisesRegex(ValueError, "disables debug information"):
            CHECKER.line_witness_needed(self.RELEASE + ["-g", "-g0"])

    def test_witness_argv_is_the_release_argv_plus_exactly_one_g(self) -> None:
        witness = CHECKER.line_witness_command(self.RELEASE, *self.OUTPUTS)
        self.assertEqual(witness, [self.RENAMED.get(argument, argument) for argument in self.RELEASE] + ["-g"])
        self.assertEqual(witness.count("-g"), 1)

    def test_codegen_changing_or_extra_flags_are_rejected(self) -> None:
        renamed = [self.RENAMED.get(argument, argument) for argument in self.RELEASE]
        for label, witness in {
                "extra -fno-inline": renamed + ["-fno-inline", "-g"],
                "extra -O1": renamed + ["-O1", "-g"],
                "-O3 replaced": [("-O1" if argument == "-O3" else argument) for argument in renamed] + ["-g"],
                "-g1 instead of -g": renamed + ["-g1"],
                "no debug flag": renamed + ["-fno-inline"],
                "doubled -g": renamed + ["-g", "-g"],
                "object not renamed": list(self.RELEASE) + ["-g"]}.items():
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "exactly '-g'"):
                CHECKER.require_line_witness_delta(self.RELEASE, witness, self.RENAMED)

    def test_identical_allocated_image_is_admitted_whatever_the_unallocated_sections(self) -> None:
        release = witness_object(self.CALL)
        # Extra symbol-table entries (unallocated, as debug sections are) do not change the allocated image.
        witness = witness_object(self.CALL, symbols=[("debug_only", "notype", "local", None, 0)])
        self.assertNotEqual(release, witness)
        self.assertRegex(CHECKER.require_code_identity(release, witness), r"^[0-9a-f]{64}$")

    def test_allocated_section_mismatch_is_rejected_with_both_digests(self) -> None:
        release, witness = witness_object(self.CALL), witness_object(self.CALL, extra=b"\x01")
        with self.assertRaisesRegex(ValueError, r"changed code generation: allocated image differs in "
                                                r"\['\.rodata'\].*Release object [0-9a-f]{64}, witness object "
                                                r"[0-9a-f]{64}"):
            CHECKER.require_code_identity(release, witness)

    def test_relocation_mismatch_is_rejected(self) -> None:
        moved = [("f", self.CALL[0][1].replace(".rodata+0x8\n\tadd", ".rodata+0x10\n\tadd"))]
        with self.assertRaisesRegex(ValueError, "relocations of .text.f"):
            CHECKER.require_code_identity(witness_object(self.CALL), witness_object(moved))

    def test_missing_line_information_still_fails_binding(self) -> None:
        document = {"executed_command": self.RELEASE, "line_witness": None, "object_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "lacks line tables and the document binds no valid line witness"):
            CHECKER.bind_line_witness(None, document, Path("/w/proof.json"), [], None)
        with self.assertRaisesRegex(ValueError, "already carries line tables"):
            CHECKER.bind_line_witness(None, {"executed_command": self.RELEASE + ["-g"], "line_witness": {}},
                                      Path("/w/proof.json"), [], None)

    def test_unqualified_records_bind_the_line_witness_field(self) -> None:
        self.assertIn("line_witness", CHECKER.BOUND_FIELDS)


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


class ProductionCompileLineWitnessTests(unittest.TestCase):
    """Production argv comes from a corroborated Ninja edge, not a fabricated DB identity."""

    def test_normal_variable_launcher_requires_exact_source_block_and_unique_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "src/v2"
            home.mkdir(parents=True)
            (root / "cmake").mkdir()
            main = home / "CMakeLists.txt"
            main.write_text(CHECKER.PRODUCTION_CCACHE_BLOCK + "\nendif()\n")
            extra = root / "cmake/Extra.cmake"
            extra.write_text("# no launcher override\n")
            standalone = root / "tests/standalone/CMakeLists.txt"
            standalone.parent.mkdir(parents=True)
            standalone.write_text('set(CMAKE_CXX_COMPILER_LAUNCHER "/standalone-only")\n')
            def git(*args):
                return subprocess.check_output(["git", "-C", directory, *args])
            git("init", "-q")
            git("add", ".")
            git("-c", "user.name=Evidence Test", "-c", "user.email=evidence@example.invalid",
                "-c", "core.hooksPath=/dev/null", "commit", "-q", "--no-verify", "-m", "source")
            commit = git("rev-parse", "HEAD").decode().strip()
            cache = f"CMAKE_HOME_DIRECTORY:INTERNAL={home}\nCCACHE_PROGRAM:FILEPATH=/usr/bin/ccache\n"
            build = root / "build"
            build.mkdir()
            (build / "build.ninja").write_text(f"build build.ninja: RERUN_CMAKE | {main} {extra}\n")
            record = CHECKER.production_source_launcher(build, cache, commit)
            self.assertEqual(record["path"], "/usr/bin/ccache")
            manifest = build / "build.ninja"
            original_manifest = manifest.read_text()
            generated = build / "Generated.cmake"
            generated.write_text("# generated CMake module\n")
            manifest.write_text(original_manifest.rstrip() + f" {generated}\n")
            generated_record = CHECKER.production_source_launcher(build, cache, commit)
            self.assertIn(str(generated.resolve()), generated_record["cmake_sources"])
            generated.write_text('define_property(TARGET PROPERTY CXX_COMPILER_LAUNCHER BRIEF_DOCS "x" FULL_DOCS "x")\n')
            with self.assertRaisesRegex(ValueError, "extra or changed"):
                CHECKER.production_source_launcher(build, cache, commit)
            untracked = root / "ProcessedTest.cmake"
            untracked.write_text("# not tracked\n")
            manifest.write_text(original_manifest.rstrip() + f" {untracked}\n")
            with self.assertRaisesRegex(ValueError, "untracked processed"):
                CHECKER.production_source_launcher(build, cache, commit)
            untracked.unlink()
            manifest.write_text(original_manifest.rstrip() + f" {standalone}\n")
            with self.assertRaisesRegex(ValueError, "extra or changed"):
                CHECKER.production_source_launcher(build, cache, commit)
            manifest.write_text(original_manifest)
            for changed_cache, changed_commit in ((cache, None), (cache, "0" * 40),
                                                   (cache.replace("/usr/bin/ccache", "CCACHE_PROGRAM-NOTFOUND"), commit)):
                with self.subTest(cache=changed_cache), self.assertRaises(ValueError):
                    CHECKER.production_source_launcher(build, changed_cache, changed_commit)
            main.write_text(main.read_text().replace("find_program(CCACHE_PROGRAM ccache)",
                                                     "find_program(CCACHE_PROGRAM other)"))
            with self.assertRaisesRegex(ValueError, "block changed"):
                CHECKER.production_source_launcher(build, cache, commit)
            git("checkout", "--", "src/v2/CMakeLists.txt")
            extra.write_text('set_property(TARGET core PROPERTY CXX_COMPILER_LAUNCHER "/other")\n')
            git("add", ".")
            git("-c", "user.name=Evidence Test", "-c", "user.email=evidence@example.invalid",
                "-c", "core.hooksPath=/dev/null", "commit", "-q", "--no-verify", "-m", "extra launcher")
            commit = git("rev-parse", "HEAD").decode().strip()
            with self.assertRaisesRegex(ValueError, "extra or changed"):
                CHECKER.production_source_launcher(build, cache, commit)

    def test_processed_cmake_assignment_lexer_covers_property_and_genex_forms(self):
        for text in ('set_property(TARGET core PROPERTY CXX_COMPILER_LAUNCHER "$<1:/other>")',
                     'set_target_properties(core PROPERTIES CXX_COMPILER_LAUNCHER /other)',
                     'define_property(TARGET PROPERTY CXX_COMPILER_LAUNCHER BRIEF_DOCS "x" FULL_DOCS "x")',
                     'set(CMAKE_CXX_COMPILER_LAUNCHER [=[/other]=])'):
            with self.subTest(text=text):
                self.assertEqual(len(CHECKER.production_cmake_assignments(text)), 1)
        for text in ('#[=[set(CMAKE_CXX_COMPILER_LAUNCHER ignored)]=]',
                     'set(CMAKE_HIP_COMPILER_LAUNCHER /hip)',
                     'set(args "-DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}")'):
            self.assertEqual(CHECKER.production_cmake_assignments(text), [])

    def test_witness_redirects_only_output_operands_and_never_target(self):
        authority = {"directory": "/num/build", "direct": ["/usr/bin/c++", "-O3", "-MD", "-MT", "a.o",
                                                           "-MF", "a.o.d", "-o", "a.o", "-c", "/src/a.cpp"]}
        with mock.patch.object(CHECKER, "require_production_authority"):
            command = CHECKER.production_witness_command(authority, Path("/art/lines.o"), Path("/art/lines.d"))
            self.assertEqual(command, ["/usr/bin/c++", "-O3", "-MD", "-MT", "a.o", "-MF", "/art/lines.d",
                                       "-o", "/art/lines.o", "-c", "/src/a.cpp", "-g"])
            for obj, deps in (("/num/build/lines.o", "/art/lines.d"),
                              ("/art/lines.o", "/num/build/lines.d"), ("/art/x", "/art/x")):
                with self.subTest(obj=obj, deps=deps), self.assertRaises(ValueError):
                    CHECKER.production_witness_command(authority, Path(obj), Path(deps))

    def test_location_table_rejects_foreign_missing_and_duplicate_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "a.cpp"
            source.write_text("int a;\n")
            inputs = {str(source): {}}
            table = f'.file 1 "{source}"\n.loc 1 1 0\n'
            self.assertEqual(CHECKER.production_location_table(table, inputs, directory), {"1": str(source)})
            for changed in (table + table, '.loc 1 1 0', table.replace(str(source), "/foreign/a.cpp"),
                            table.replace('.loc 1 1', '.loc 2 1'), table.replace('.loc 1 1', '.loc 1 9')):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    CHECKER.production_location_table(changed, inputs, directory)

    def test_same_build_database_and_manifest_binding(self):
        with tempfile.TemporaryDirectory() as root:
            build = Path(root)
            database = build / "compile_commands.json"
            source, output = "/num/src/v2/loaders/WeightManager.cpp", "CMakeFiles/core.dir/WeightManager.cpp.o"
            command = ["/usr/bin/c++", "-O3", "-std=gnu++23", "-o", output, "-c", source]
            entry = {"file": source, "output": output, "directory": root, "arguments": command}
            database.write_text(json.dumps([entry]))
            (build / "build.ninja").write_text("include CMakeFiles/rules.ninja\n")
            (build / "CMakeFiles").mkdir()
            (build / "CMakeFiles/rules.ninja").write_text("# native rule\n")
            (build / "CMakeCache.txt").write_text("# no configured launcher\n")
            edge = command[:-4] + ["-MD", "-MT", output, "-MF", output + ".d"] + command[-4:]
            result = subprocess.CompletedProcess([], 0, shlex.join(edge) + "\n", "")
            digest = CHECKER.envelope.digest(database)
            identity = lambda argv: {"selected": argv[0], "sha256": "native-tool"}
            with mock.patch.object(CHECKER, "driver_identity", side_effect=identity), \
                    mock.patch.object(CHECKER.subprocess, "run", return_value=result):
                authority = CHECKER.production_compile_authority(database, digest, source, output, Path("/bin/ninja"))
                CHECKER.require_production_authority(authority)
                for key in ("file", "output"):
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        CHECKER.production_compile_authority(database, digest, source if key != "file" else "/wrong.cpp",
                                                             output if key != "output" else "wrong.o", Path("/bin/ninja"))
                (build / "CMakeFiles/rules.ninja").write_text("# changed rule\n")
                with self.assertRaisesRegex(ValueError, "authority changed"):
                    CHECKER.require_production_authority(authority)
                database.write_text(json.dumps([entry, entry]))
                with self.assertRaisesRegex(ValueError, "stale"):
                    CHECKER.production_compile_authority(database, digest, source, output, Path("/bin/ninja"))
                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    CHECKER.production_compile_authority(database, CHECKER.envelope.digest(database), source, output,
                                                         Path("/bin/ninja"))

    def test_configured_launcher_is_exact_single_elf_and_rechecked(self):
        with tempfile.TemporaryDirectory() as root:
            build = Path(root)
            launcher = build / "ccache"
            launcher.write_bytes(b"\x7fELFtest-launcher")
            database = build / "compile_commands.json"
            cache = build / "CMakeCache.txt"
            source, output = "/num/src/v2/loaders/WeightManager.cpp", "WeightManager.cpp.o"
            command = ["/usr/bin/c++", "-O3", "-std=gnu++23", "-o", output, "-c", source]
            database.write_text(json.dumps([{"file": source, "output": output,
                                             "directory": root, "arguments": command}]))
            (build / "build.ninja").write_text("# selected build\n")
            cache.write_text(f"CMAKE_CXX_COMPILER_LAUNCHER:FILEPATH={launcher}\n")
            edge = [str(launcher)] + command[:-4] + ["-MD", "-MT", output, "-MF", output + ".d"] + command[-4:]
            result = subprocess.CompletedProcess([], 0, shlex.join(edge) + "\n", "")
            digest = CHECKER.envelope.digest(database)
            def identity(argv):
                return {"selected": argv[0], "resolved": argv[0],
                        "sha256": CHECKER.envelope.digest(launcher) if argv[0] == str(launcher) else "native"}
            with mock.patch.object(CHECKER, "driver_identity", side_effect=identity), \
                    mock.patch.object(CHECKER, "observe", return_value="ccache 4.test"), \
                    mock.patch.object(CHECKER.subprocess, "run", return_value=result):
                authority = CHECKER.production_compile_authority(database, digest, source, output, Path("/bin/ninja"))
                self.assertEqual(authority["direct"], edge[1:])
                launcher.write_bytes(b"\x7fELFchanged-launcher")
                with self.assertRaisesRegex(ValueError, "authority changed"):
                    CHECKER.require_production_authority(authority)
                for configured in (str(launcher) + ";--args", str(launcher) + " --args", "/other/ccache", ""):
                    cache.write_text(f"CMAKE_CXX_COMPILER_LAUNCHER:FILEPATH={configured}\n")
                    with self.subTest(configured=configured), self.assertRaises((ValueError, FileNotFoundError)):
                        CHECKER.production_compile_authority(database, digest, source, output, Path("/bin/ninja"))

    def test_shared_adapter_binds_production_dialect_target_and_output_only_deltas(self):
        output = "CMakeFiles/core.dir/WeightManager.cpp.o"
        direct = ["/usr/bin/c++", "-O3", "-std=gnu++23", "-MD", "-MT", output, "-MF", output + ".d",
                  "-o", output, "-c", "/num/src/v2/loaders/WeightManager.cpp"]
        authority = {"directory": "/num/build", "output": output, "source": direct[-1], "direct": direct}
        command = list(direct)
        command[command.index("-o") + 1] = "/art/production.o"
        command[command.index("-MF") + 1] = "/art/production.d"
        with mock.patch.object(CHECKER, "require_production_authority"):
            paths = CHECKER.production_job_paths(authority, command, None)
            self.assertEqual(paths.production, authority)
            self.assertEqual(CHECKER.production_job_paths(authority, command + ["-g"], None), paths)
            for changed in (command + ["-funknown"], ["/usr/bin/ccache"] + command,
                            [arg.replace("gnu++23", "c++20") for arg in command], command + ["-MT", output]):
                with self.subTest(command=changed), self.assertRaises(ValueError):
                    CHECKER.production_job_paths(authority, changed, None)
        jobs = [["/usr/libexec/gcc/cc1plus", "-quiet", "-I", "/num/src/v2", "-isystem", "/sdk/include",
                 "-iquote", "/num/generated", "-D", "NDEBUG", "-MD", "/art/production.d", "-MF",
                 "/art/production.d", "-MT", output, direct[-1], "-march=armv8-a+i8mm", "-O3",
                 "-std=gnu++23", "-o", "/tmp/ccAb12Cd.s"],
                ["as", "-EL", "-march=armv8-a", "-mabi=lp64", "-o", "/art/production.o", "/tmp/ccAb12Cd.s"]]
        CHECKER.validate_plan(jobs, "GNU-14", paths)
        for changed in ([arg.replace("gnu++23", "c++20") for arg in jobs[0]],
                        jobs[0] + ["-funknown"], jobs[0] + ["-MT", output]):
            with self.subTest(job=changed), self.assertRaises(ValueError):
                CHECKER.validate_plan([changed, jobs[1]], "GNU-14", paths)
        with self.assertRaises(ValueError):
            CHECKER.validate_plan(jobs, "GNU-14", paths._replace(production=None))

    def test_native_cmake_metadata_relation_preserves_codegen_order(self):
        database = ["/usr/bin/c++", "-DNDEBUG", "-I/num/src/v2", "-isystem", "/sdk/include",
                    "-mcpu=generic", "-march=armv8-a+simd+dotprod+i8mm+fp16fml", "-O3",
                    "-std=gnu++23", "-fopenmp", "-o", "CMakeFiles/core.dir/WeightManager.cpp.o",
                    "-c", "/num/src/v2/loaders/WeightManager.cpp"]
        output = database[database.index("-o") + 1]
        metadata = ["-MD", "-MT", output, "-MF", output + ".d"]
        edge = database[:-4] + metadata + database[-4:]
        self.assertEqual(CHECKER.production_compile_relation(database, edge, output, None), edge)
        self.assertEqual(CHECKER.production_compile_relation(database, ["/usr/bin/ccache"] + edge,
                                                             output, "/usr/bin/ccache"), edge)
        mutations = [edge + ["-funknown"], edge + ["-O2"], edge + ["-MT", output],
                     [arg for arg in edge if arg != "-MD"],
                     ["wrong.o" if arg == output else arg for arg in edge],
                     ["wrong.d" if arg == output + ".d" else arg for arg in edge],
                     ["/usr/bin/ccache"] + edge, ["/usr/bin/ccache", "--args"] + edge]
        for changed in mutations:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                CHECKER.production_compile_relation(database, changed, output, None)
        for changed in (edge, ["/other/ccache"] + edge, ["/usr/bin/ccache"] * 2 + edge):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                CHECKER.production_compile_relation(database, changed, output, "/usr/bin/ccache")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--shard":
        result = unittest.TextTestRunner(verbosity=1).run(shard_suite(sys.argv[2]))
        sys.exit(0 if result.wasSuccessful() and result.testsRun else 1)
    unittest.main()
