#!/usr/bin/env python3
"""Exact DWARF source positions of the memcpy calls of compact programs (reviewer 98, Decision 1).

For each `bl` to memcpy/__memcpy_chk (by relocation symbol) in one function of a relocatable ELF64 AArch64 object:
- the DIEs are read from the object's own DWARF 5 (.debug_info/.debug_abbrev/.debug_rnglists/.debug_line with
  their RELA relocations applied section-aware: every address is (section index, offset), never a bare number);
- the covering DW_TAG_inlined_subroutine DIEs of the `bl` address (DW_AT_low_pc/high_pc or DW_AT_ranges) are
  collected; the innermost one must be unique and its abstract origin must be the glibc `memcpy` declared in
  bits/string_fortified.h - then the site's position is its DW_AT_call_file (through the CU's line-table file
  table) : DW_AT_call_line;
- a `bl` covered by no inlined subroutine is "direct" (its position is the line-table row at that address,
  supplied by the caller from the exact assembler input); a covered `bl` whose innermost instance is not glibc
  memcpy, or two innermost instances, FAIL;
- ownership: the nearest enclosing inlined_subroutine/subprogram of the memcpy instance must be an instance of a
  function named `function_names` (the compact kernel) whose declaration file is the call file, and the outermost
  subprogram covering the address must be the analysed function (its DW_AT_low_pc is the function symbol).
No nearest-line or fallback lookup exists. Returns one record per `bl` in address order.
"""
from __future__ import annotations

import os
import posixpath
import re
import struct
import sys
from pathlib import Path


TAG_SUBPROGRAM, TAG_INLINED, TAG_COMPILE_UNIT, TAG_VARIABLE = 0x2E, 0x1D, 0x11, 0x34
AT = {"sibling": 0x01, "name": 0x03, "low_pc": 0x11, "high_pc": 0x12, "comp_dir": 0x1B, "stmt_list": 0x10,
      "decl_file": 0x3A, "abstract_origin": 0x31, "specification": 0x47, "ranges": 0x55, "call_file": 0x58,
      "call_line": 0x59, "linkage_name": 0x6E, "declaration": 0x3C, "location": 0x02}
KEEP = {AT["name"], AT["low_pc"], AT["high_pc"], AT["comp_dir"], AT["stmt_list"], AT["decl_file"],
        AT["abstract_origin"], AT["specification"], AT["ranges"], AT["call_file"], AT["call_line"],
        AT["linkage_name"], AT["declaration"]}
R_ABS64, R_ABS32 = 257, 258
MEMCPY_CALLS = {"memcpy", "__memcpy_chk"}


class DwarfError(ValueError):
    pass


ElfError = DwarfError
SHT_NOBITS, SHT_SYMTAB, SHT_RELA = 8, 2, 4
BINDINGS = {0: "local", 1: "global", 2: "weak", 10: "unique"}
TYPES = {0: "notype", 1: "object", 2: "func", 3: "section", 4: "file", 5: "common", 6: "tls", 10: "ifunc"}
VISIBILITY = {0: "default", 1: "internal", 2: "hidden", 3: "protected"}
CALL_TYPES = {282: "JUMP26", 283: "CALL26"}




def _string(table: bytes, offset: int) -> str:
    end = table.index(b"\0", offset)
    return table[offset:end].decode("utf-8", "replace")


def read_elf(path_or_bytes) -> dict:
    data = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ElfError("not an ELF64 little-endian file")
    e_type, = struct.unpack_from("<H", data, 16)
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    sections = []
    for index in range(shnum):
        (name, kind, flags, address, offset, size, link, info, align,
         entsize) = struct.unpack_from("<IIQQQQIIQQ", data, shoff + index * shentsize)
        sections.append({"index": index, "name_offset": name, "type": kind, "flags": flags, "offset": offset,
                         "size": size, "link": link, "info": info, "align": align, "entsize": entsize,
                         "bytes": b"" if kind == SHT_NOBITS else data[offset:offset + size]})
    names = sections[shstrndx]["bytes"]
    for section in sections:
        section["name"] = _string(names, section["name_offset"])
    symbols = []
    for section in sections:
        if section["type"] != SHT_SYMTAB:
            continue
        strings = sections[section["link"]]["bytes"]
        for index in range(section["size"] // 24):
            name, info, other, shndx, value, size = struct.unpack_from("<IBBHQQ", section["bytes"], index * 24)
            symbols.append({"index": index, "name": _string(strings, name), "binding": BINDINGS.get(info >> 4, str(info >> 4)),
                            "type": TYPES.get(info & 15, str(info & 15)), "visibility": VISIBILITY[other & 3],
                            "shndx": shndx, "value": value, "size": size})
    relocations = {}
    for section in sections:
        if section["type"] != SHT_RELA:
            continue
        entries = []
        for index in range(section["size"] // 24):
            offset, info, addend = struct.unpack_from("<QQq", section["bytes"], index * 24)
            entries.append({"offset": offset, "type": info & 0xFFFFFFFF, "symbol": info >> 32, "addend": addend})
        relocations[section["index"]] = entries
    return {"type": e_type, "sections": sections, "symbols": symbols, "relocations": relocations}




def uleb(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return result, pos


def sleb(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            if byte & 0x40:
                result -= 1 << shift
            return result, pos


class Object:
    def __init__(self, elf: dict):
        self.elf = elf
        self.by_name = {}
        for section in elf["sections"]:
            if section["name"] in self.by_name and section["name"].startswith(".debug"):
                raise DwarfError(f"two {section['name']} sections")
            self.by_name.setdefault(section["name"], section)
        self.relocs: dict[str, dict[int, tuple]] = {}
        for section in elf["sections"]:
            if section["type"] == 4 and elf["sections"][section["info"]]["name"].startswith(".debug"):
                target = elf["sections"][section["info"]]["name"]
                table = self.relocs.setdefault(target, {})
                for entry in elf["relocations"][section["index"]]:
                    table[entry["offset"]] = (entry["type"], elf["symbols"][entry["symbol"]], entry["addend"])

    def data(self, name: str) -> bytes:
        section = self.by_name.get(name)
        if section is None:
            raise DwarfError(f"no {name} section")
        return section["bytes"]

    def offset_value(self, section: str, at: int, size: int) -> int:
        """A section-offset field (strp, line_strp, sec_offset): S + A of its relocation (a debug section symbol)."""
        relocation = self.relocs.get(section, {}).get(at)
        if relocation is None:
            return int.from_bytes(self.data(section)[at:at + size], "little")
        kind, symbol, addend = relocation
        if kind not in (R_ABS32, R_ABS64):
            raise DwarfError(f"{section}+{at:#x}: unsupported relocation {kind}")
        return (0 if symbol["type"] == "section" else symbol["value"]) + addend

    def address(self, section: str, at: int) -> tuple[int, int] | None:
        """A DW_FORM_addr: (section index, offset) from its relocation; None for an unrelocated 0 (no address)."""
        relocation = self.relocs.get(section, {}).get(at)
        if relocation is None:
            if int.from_bytes(self.data(section)[at:at + 8], "little"):
                raise DwarfError(f"{section}+{at:#x}: an address without a relocation")
            return None
        kind, symbol, addend = relocation
        if kind != R_ABS64 or not 0 < symbol["shndx"] < 0xFF00:
            raise DwarfError(f"{section}+{at:#x}: address relocation {kind} against {symbol['name']!r}")
        return symbol["shndx"], (0 if symbol["type"] == "section" else symbol["value"]) + addend


def cstring(data: bytes, at: int) -> str:
    return data[at:data.index(b"\0", at)].decode("utf-8", "replace")


def abbreviations(data: bytes, at: int) -> dict:
    table = {}
    while True:
        code, at = uleb(data, at)
        if not code:
            return table
        tag, at = uleb(data, at)
        children = data[at]
        at += 1
        specs = []
        while True:
            name, at = uleb(data, at)
            form, at = uleb(data, at)
            implicit = None
            if form == 0x21:
                implicit, at = sleb(data, at)
            if not name and not form:
                break
            specs.append((name, form, implicit))
        table[code] = (tag, bool(children), specs)


def read_dies(obj: Object) -> tuple[dict, list]:
    """{offset: DIE} of compile units, subprograms and inlined subroutines (kept attributes, parent = nearest kept
    ancestor, cu = its compile unit offset), and the compile units."""
    info, abbrev = obj.data(".debug_info"), obj.data(".debug_abbrev")
    strs = obj.by_name.get(".debug_str", {}).get("bytes", b"")
    line_strs = obj.by_name.get(".debug_line_str", {}).get("bytes", b"")
    dies, units, at = {}, [], 0
    while at < len(info):
        unit = at
        length = struct.unpack_from("<I", info, at)[0]
        if length >= 0xFFFFFFF0:
            raise DwarfError("64-bit DWARF is not supported")
        end = at + 4 + length
        version = struct.unpack_from("<H", info, at + 4)[0]
        if version != 5 or info[at + 6] != 1:
            raise DwarfError(f"unit at {unit:#x}: DWARF version {version} / unit type {info[at + 6]} not supported")
        if info[at + 7] != 8:
            raise DwarfError("address size is not 8")
        table = abbreviations(abbrev, obj.offset_value(".debug_info", at + 8, 4))
        at += 12
        stack: list[tuple[int, int | None]] = []  # (depth marker, kept DIE offset)
        kept_parent: list[int | None] = [None]
        cu = None
        while at < end:
            offset = at
            code, at = uleb(info, at)
            if not code:
                kept_parent.pop()
                continue
            tag, children, specs = table[code]
            attributes = {}
            for name, form, implicit in specs:
                value, at = form_value(obj, info, at, form, implicit, unit, strs, line_strs)
                if name in KEEP:
                    attributes[name] = value
            keep = tag in (TAG_SUBPROGRAM, TAG_INLINED, TAG_COMPILE_UNIT, TAG_VARIABLE)
            if tag == TAG_VARIABLE and any(name == AT["location"] for name, _, _ in specs):
                attributes["has_location"] = True
            if tag == TAG_COMPILE_UNIT:
                cu = offset
                units.append({"offset": offset, **{key: attributes.get(AT[key]) for key in ("comp_dir", "stmt_list",
                                                                                           "name")}})
            if keep:
                dies[offset] = {"tag": tag, "attributes": attributes, "parent": kept_parent[-1], "cu": cu}
            if children:
                kept_parent.append(offset if keep else kept_parent[-1])
        at = end
    return dies, units


def form_value(obj, info, at, form, implicit, unit, strs, line_strs):
    section = ".debug_info"
    fixed = {0x0B: 1, 0x05: 2, 0x06: 4, 0x07: 8, 0x11: 1, 0x12: 2, 0x13: 4, 0x14: 8, 0x0C: 1, 0x1E: 16, 0x20: 8,
             0x25: 1, 0x26: 2, 0x28: 4, 0x29: 1, 0x2A: 2, 0x2C: 4}
    if form == 0x01:  # addr
        return ("addr", obj.address(section, at)), at + 8
    if form in (0x0E, 0x1F):  # strp, line_strp
        value = obj.offset_value(section, at, 4)
        return cstring(strs if form == 0x0E else line_strs, value), at + 4
    if form == 0x17:  # sec_offset
        return obj.offset_value(section, at, 4), at + 4
    if form in (0x11, 0x12, 0x13, 0x14):  # ref1/2/4/8: unit-relative
        size = fixed[form]
        return ("ref", unit + int.from_bytes(info[at:at + size], "little")), at + size
    if form == 0x15:  # ref_udata
        value, at = uleb(info, at)
        return ("ref", unit + value), at
    if form == 0x10:  # ref_addr
        return ("ref", obj.offset_value(section, at, 4)), at + 4
    if form == 0x08:  # string
        end = info.index(b"\0", at)
        return info[at:end].decode("utf-8", "replace"), end + 1
    if form == 0x0F:  # udata
        return uleb(info, at)
    if form == 0x0D:  # sdata
        return sleb(info, at)
    if form == 0x19:  # flag_present
        return True, at
    if form == 0x21:  # implicit_const
        return implicit, at
    if form in (0x18, 0x09):  # exprloc, block
        size, at = uleb(info, at)
        return None, at + size
    if form == 0x0A:  # block1
        return None, at + 1 + info[at]
    if form == 0x03:  # block2
        return None, at + 2 + struct.unpack_from("<H", info, at)[0]
    if form == 0x04:  # block4
        return None, at + 4 + struct.unpack_from("<I", info, at)[0]
    if form in fixed:
        size = fixed[form]
        return int.from_bytes(info[at:at + size], "little"), at + size
    raise DwarfError(f"unsupported DWARF form {form:#x} at .debug_info+{at:#x}")


def ranges(obj: Object, die: dict, base: tuple | None) -> list[tuple[int, int, int]]:
    """[(section index, start, end)] of a DIE (low/high pc or DW_AT_ranges in .debug_rnglists)."""
    attributes = die["attributes"]
    low = attributes.get(AT["low_pc"])
    if low is not None and AT["ranges"] not in attributes:
        high = attributes.get(AT["high_pc"])
        if low[1] is None or not isinstance(high, int):
            raise DwarfError("low_pc without an address or a high_pc that is not a length")
        return [(low[1][0], low[1][1], low[1][1] + high)]
    if AT["ranges"] not in attributes:
        return []
    data, at, result = obj.data(".debug_rnglists"), attributes[AT["ranges"]], []
    while True:
        kind = data[at]
        at += 1
        if kind == 0:  # end_of_list
            return result
        if kind == 5:  # base_address
            base = obj.address(".debug_rnglists", at)
            at += 8
        elif kind == 4:  # offset_pair
            start, at = uleb(data, at)
            end, at = uleb(data, at)
            if base is None:
                raise DwarfError("offset_pair without a base address")
            result.append((base[0], base[1] + start, base[1] + end))
        elif kind == 7:  # start_length
            start = obj.address(".debug_rnglists", at)
            length, at = uleb(data, at + 8)
            result.append((start[0], start[1], start[1] + length))
        elif kind == 6:  # start_end
            start, end = obj.address(".debug_rnglists", at), obj.address(".debug_rnglists", at + 8)
            if start[0] != end[0]:
                raise DwarfError("a range spans two sections")
            result.append((start[0], start[1], end[1]))
            at += 16
        else:
            raise DwarfError(f"unsupported range list entry {kind}")


def file_table(obj: Object, unit: dict) -> list[str]:
    """DWARF 5 line-table file names of one unit, as absolute (comp_dir-joined) normalised paths."""
    data, at = obj.data(".debug_line"), unit["stmt_list"]
    line_strs = obj.by_name.get(".debug_line_str", {}).get("bytes", b"")
    version = struct.unpack_from("<H", data, at + 4)[0]
    if version != 5:
        raise DwarfError(f"line table version {version}")
    at += 4 + 2 + 1 + 1 + 4  # unit_length, version, address_size, seg_sel_size, header_length
    at += 4  # min_inst_length, max_ops, default_is_stmt, line_base
    line_range, opcode_base = data[at], data[at + 1]
    at += 2 + (opcode_base - 1)
    del line_range

    def entries(at: int) -> tuple[list[dict], int]:
        count_formats = data[at]
        at += 1
        formats = []
        for _ in range(count_formats):
            content, at = uleb(data, at)
            form, at = uleb(data, at)
            formats.append((content, form))
        count, at = uleb(data, at)
        result = []
        for _ in range(count):
            entry = {}
            for content, form in formats:
                if form == 0x1F:
                    entry[content] = cstring(line_strs, obj.offset_value(".debug_line", at, 4))
                    at += 4
                elif form == 0x08:
                    end = data.index(b"\0", at)
                    entry[content] = data[at:end].decode()
                    at = end + 1
                elif form == 0x0F:
                    entry[content], at = uleb(data, at)
                elif form == 0x0B:
                    entry[content] = data[at]
                    at += 1
                elif form == 0x1E:
                    entry[content] = data[at:at + 16].hex()
                    at += 16
                else:
                    raise DwarfError(f"line table entry form {form:#x}")
            result.append(entry)
        return result, at
    directories, at = entries(at)
    files, at = entries(at)
    comp_dir = unit["comp_dir"] or directories[0].get(1, "")
    dirs = [posixpath.join(comp_dir, d.get(1, "")) for d in directories]
    return [posixpath.normpath(posixpath.join(dirs[f.get(2, 0)], f.get(1, ""))) for f in files]


def origin(dies: dict, offset: int, seen: int = 0) -> dict:
    """The declaration DIE behind abstract_origin / specification chains (name, linkage name, decl_file)."""
    die = dies.get(offset)
    if die is None or seen > 8:
        raise DwarfError(f"unresolved origin DIE {offset:#x}")
    attributes = die["attributes"]
    for key in ("abstract_origin", "specification"):
        if AT[key] in attributes and (AT["name"] not in attributes or AT["decl_file"] not in attributes):
            found = origin(dies, attributes[AT[key]][1], seen + 1)
            return {"name": attributes.get(AT["name"], found["name"]),
                    "linkage": attributes.get(AT["linkage_name"], found["linkage"]),
                    "decl_file": attributes.get(AT["decl_file"], found["decl_file"]), "cu": found["cu"]}
    return {"name": attributes.get(AT["name"]), "linkage": attributes.get(AT["linkage_name"]),
            "decl_file": attributes.get(AT["decl_file"]), "cu": die["cu"]}


def definition_declarations(elf: dict, names: set[str]) -> dict[str, dict]:
    """Q-3 (reviewer 109): per non-local definition name, the declaration position of its DEFINITION DIE in the
    object's own DWARF: a subprogram with code (low_pc/ranges) or a variable with a location, whose linkage name
    (through abstract_origin/specification) is the symbol; decl_file through the unit's file table. A name with no
    such DIE, or more than one distinct position, is {"error": ...}."""
    obj = Object(elf)
    dies, units = read_dies(obj)
    units = {unit["offset"]: unit for unit in units}
    files: dict[int, list] = {}
    found: dict[str, set] = {}
    cu_names: dict[str, set] = {}
    for offset, die in dies.items():
        attributes = die["attributes"]
        if die["tag"] == TAG_SUBPROGRAM:
            if AT["low_pc"] not in attributes and AT["ranges"] not in attributes:
                continue
        elif die["tag"] == TAG_VARIABLE:
            if not attributes.get("has_location") or attributes.get(AT["declaration"]):
                continue
        else:
            continue
        declared = origin(dies, offset)
        if declared["linkage"] not in names or not isinstance(declared["decl_file"], int):
            continue
        cu = declared["cu"]
        if cu not in files:
            files[cu] = file_table(obj, units[cu])
        table = files[cu]
        if declared["decl_file"] >= len(table):
            found.setdefault(declared["linkage"], set()).add(("?", None))
            continue
        found.setdefault(declared["linkage"], set()).add((table[declared["decl_file"]], offset))
        cu_names.setdefault(declared["linkage"], set()).add(units[cu].get("name"))
    result = {}
    for name in names:
        positions = {path for path, _ in found.get(name, ())}
        if len(positions) != 1 or "?" in positions:
            result[name] = {"error": f"{len(positions)} definition declaration file(s) {sorted(positions)[:3]}"}
        else:
            result[name] = {"decl_file": positions.pop(), "dies": sorted(o for _, o in found[name]),
                            "cu_names": sorted(str(n) for n in cu_names.get(name, ()))}
    return result


def memcpy_sites(elf: dict, function: dict, function_names: set[str], fortify_suffix: str = "bits/string_fortified.h",
                 cache: dict | None = None) -> list[dict]:
    """Records of every `bl` to memcpy/__memcpy_chk inside `function` (a function symbol of `elf`), in address
    order: {"address", "callee", "kind": inlined|direct, "die", "call_file", "call_line"} or {"error": reason}."""
    cache = cache if cache is not None else {}
    if "dies" not in cache:
        obj = Object(elf)
        dies, units = read_dies(obj)
        cache.update(obj=obj, dies=dies, units={u["offset"]: u for u in units}, files={}, ranges={})
    obj, dies = cache["obj"], cache["dies"]

    def files_of(cu: int) -> list[str]:
        if cu not in cache["files"]:
            cache["files"][cu] = file_table(obj, cache["units"][cu])
        return cache["files"][cu]

    def covers(offset: int, shndx: int, address: int) -> bool:
        if offset not in cache["ranges"]:
            unit = cache["units"][dies[offset]["cu"]]
            base = dies[unit["offset"]]["attributes"].get(AT["low_pc"])
            cache["ranges"][offset] = ranges(obj, dies[offset], base[1] if base else None)
        return any(s == shndx and lo <= address < hi for s, lo, hi in cache["ranges"][offset])

    shndx, start, end = function["shndx"], function["value"], function["value"] + function["size"]
    calls = []
    for rela in elf["sections"]:
        if rela["type"] == 4 and rela["info"] == shndx:
            for entry in elf["relocations"][rela["index"]]:
                symbol = elf["symbols"][entry["symbol"]]
                if start <= entry["offset"] < end and entry["type"] in (282, 283) and symbol["name"] in MEMCPY_CALLS:
                    calls.append((entry["offset"], symbol["name"], entry["type"], entry["addend"]))
    if "subprograms" not in cache:
        cache["subprograms"] = [o for o, d in dies.items() if d["tag"] == TAG_SUBPROGRAM and
                                (AT["low_pc"] in d["attributes"] or AT["ranges"] in d["attributes"])]
        cache["inlined"] = [o for o, d in dies.items() if d["tag"] == TAG_INLINED]
    records = []
    for address, callee, kind, addend in sorted(calls):
        record = {"address": address, "callee": callee}
        if kind != 283 or addend:
            record["error"] = f"the call to {callee} is not a plain CALL26 (type {kind}, addend {addend})"
            records.append(record)
            continue
        outer = [o for o in cache["subprograms"] if covers(o, shndx, address)]
        if len(outer) != 1 or not covers(outer[0], shndx, start) or \
                not any(s == shndx and lo == start for s, lo, _ in cache["ranges"][outer[0]]):
            record["error"] = f"the covering subprogram is not exactly the analysed function ({len(outer)} found)"
            records.append(record)
            continue
        covering = [o for o in cache["inlined"] if covers(o, shndx, address)]

        def depth(offset: int) -> int:
            count = 0
            while dies[offset]["parent"] is not None:
                offset, count = dies[offset]["parent"], count + 1
            return count
        if not covering:
            record["kind"] = "direct"
            records.append(record)
            continue
        deepest = max(depth(o) for o in covering)
        innermost = [o for o in covering if depth(o) == deepest]
        if len(innermost) != 1:
            record["error"] = f"{len(innermost)} innermost inlined instances cover the call"
            records.append(record)
            continue
        die = innermost[0]
        declared = origin(dies, die)
        files = files_of(dies[die]["cu"])
        decl = files[declared["decl_file"]] if isinstance(declared["decl_file"], int) and \
            declared["decl_file"] < len(files) else None
        if declared["name"] != "memcpy" or decl is None or not decl.endswith("/" + fortify_suffix):
            record["error"] = (f"the innermost inlined instance {die:#x} is {declared['name']!r} from {decl!r}, not "
                               f"glibc memcpy of {fortify_suffix}")
            records.append(record)
            continue
        attributes = dies[die]["attributes"]
        call_file, call_line = attributes.get(AT["call_file"]), attributes.get(AT["call_line"])
        if not isinstance(call_file, int) or call_file >= len(files) or not isinstance(call_line, int):
            record["error"] = f"inlined instance {die:#x} has no resolvable call_file/call_line"
            records.append(record)
            continue
        parent = dies[die]["parent"]
        owner = origin(dies, parent) if parent is not None else None
        owner_decl = files[owner["decl_file"]] if owner and isinstance(owner["decl_file"], int) and \
            owner["decl_file"] < len(files) else None
        base = re.sub(r"(?:\.(?:isra|constprop|part|cold|lto_priv)\.\d+)+$", "", function["name"])
        if owner is None or str(owner["name"]).split("<", 1)[0] not in function_names or \
                owner_decl != files[call_file] or (owner["linkage"] is not None and owner["linkage"] != base):
            record["error"] = (f"the memcpy instance {die:#x} is inlined into {owner and owner['name']!r} declared in "
                               f"{owner_decl!r}, not the compact kernel function of its call file")
            records.append(record)
            continue
        record.update(kind="inlined", die=die, call_file=files[call_file], call_line=call_line,
                      fortify_file=decl, owner=owner["name"], owner_linkage=owner["linkage"], owner_die=parent)
        records.append(record)
    return records


if __name__ == "__main__":
    import json
    elf = read_elf(sys.argv[1])
    wanted = sys.argv[2]
    function = next(s for s in elf["symbols"] if s["name"] == wanted and s["type"] == "func")
    print(json.dumps(memcpy_sites(elf, function, set(sys.argv[3].split(","))), indent=1))
