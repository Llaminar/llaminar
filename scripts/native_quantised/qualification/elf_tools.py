"""Minimal ELF64 little-endian reader for the production B1/P2 tools (no external dependency).

- read_elf(): sections (with bytes), symbols, RELA entries.
- nondebug_view(): the comparable non-debug content of a relocatable object (B1-a): every non-debug section's
  type, flags, alignment, entry size, size and bytes; groups by member names; the relocations of non-debug
  sections as (offset, type, symbol name, binding, addend) with section symbols named by their section; and the
  non-debug symbol table (name, binding, type, visibility, section, value, size) - all keyed by section NAME and
  occurrence, so extra debug sections and their symbols (added by -g) do not shift the comparison.
  "Debug" is exactly the .debug_* sections, their .rela.debug_* relocation sections and the section symbols of
  those (C-3b); every other section (.comment, .note.*, .zdebug*, .gnu.debuglto*, ...) is compared, so any
  difference there FAILS.
- function_graph(): the fail-closed reference graph of one object (C-1): nodes are function symbols (cold/partition
  parts are their own STT_FUNC symbols), data objects and, for data not covered by an object symbol, whole
  sections; an edge is EVERY relocation of an allocated section (calls, tail calls, conditional branches,
  address-taking ADRP/ADD/GOT/ABS/PREL references - OpenMP outlined bodies passed to GOMP_* and function
  pointers in code or data). A relocation in an executable section that no function symbol covers, a CALL26
  whose target is not a function start, a branch or address into the interior of an unrelated function, and a
  reference into an executable section that no function covers are FAILURES, never skipped. Indirect branches
  (BLR/BR and their pointer-authentication forms) are enumerated per function from the code bytes (mapping-symbol
  data excluded); they are listed, not resolved.
  C-5 (reviewer 88): GAS resolves PC-relative references to local symbols in the same section at assembly time
  and emits NO relocation (production has no -ffunction-sections: static/anonymous functions and .isra/
  .constprop/.part clones share .text). So the code bytes of every function are decoded ($x/$d mapping):
  B/BL (imm26), B.cond/BC.cond/CBZ/CBNZ/TBZ/TBNZ, ADR (imm21), ADRP and LDR/PRFM literal. An instruction with
  a relocation must carry a relocation of its class (cross-check, both directions); one without a relocation
  that leaves its function must target a function start (an edge "decoded-<class>") or its .cold partner;
  an interior or unowned target, an ADRP without relocation, a BL to its own interior and any non-NOP code word
  outside every function FAIL. Every failure names the function, the offset, the word and the relocation.
"""
from __future__ import annotations

import bisect
import re
import struct
from pathlib import Path

SHT_NOBITS, SHT_SYMTAB, SHT_STRTAB, SHT_RELA, SHT_REL, SHT_GROUP = 8, 2, 3, 4, 9, 17
SHF_ALLOC, SHF_EXECINSTR = 0x2, 0x4
SHN_UNDEF, SHN_LORESERVE, SHN_ABS, SHN_COMMON = 0, 0xFF00, 0xFFF1, 0xFFF2
R_CALL26, R_BRANCHES = 283, {279: "TSTBR14", 280: "CONDBR19", 282: "JUMP26"}
# Relocation types a PC-relative instruction class may carry (AArch64 ELF ABI), C-5(b).
CLASS_RELOCATIONS = {
    "b": {282}, "bl": {283}, "bcond": {280}, "cb": {280}, "tb": {279},
    "adr": {274, 512, 517, 561},                       # ADR_PREL_LO21, TLSGD/TLSLD/TLSDESC_ADR_PREL21
    "adrp": {275, 276, 311, 513, 518, 541, 562},       # ADR_PREL_PG_HI21[_NC], ADR_GOT_PAGE, TLS*_PAGE21
    "ldr_literal": {273, 309, 543, 560},               # LD_PREL_LO19, GOT_LD_PREL19, TLSIE/TLSDESC literal
}
PC_RELATIVE_RELOCATIONS = set().union(*CLASS_RELOCATIONS.values())
NOP_WORDS = {0xD503201F, 0x00000000}


def _signed(value: int, bits: int) -> int:
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def _decode(word: int):
    """(class, byte displacement or None) of a PC-relative AArch64 instruction; None for any other word."""
    if word & 0xFC000000 == 0x14000000:
        return "b", _signed(word & 0x3FFFFFF, 26) * 4
    if word & 0xFC000000 == 0x94000000:
        return "bl", _signed(word & 0x3FFFFFF, 26) * 4
    if word & 0xFF000000 == 0x54000000:
        return "bcond", _signed((word >> 5) & 0x7FFFF, 19) * 4
    if word & 0x7E000000 == 0x34000000:
        return "cb", _signed((word >> 5) & 0x7FFFF, 19) * 4
    if word & 0x7E000000 == 0x36000000:
        return "tb", _signed((word >> 5) & 0x3FFF, 14) * 4
    if word & 0x9F000000 == 0x10000000:
        return "adr", _signed((((word >> 5) & 0x7FFFF) << 2) | ((word >> 29) & 3), 21)
    if word & 0x9F000000 == 0x90000000:
        return "adrp", None
    if word & 0x3B000000 == 0x18000000:
        return "ldr_literal", _signed((word >> 5) & 0x7FFFF, 19) * 4
    return None
BINDINGS = {0: "local", 1: "global", 2: "weak", 10: "unique"}
TYPES = {0: "notype", 1: "object", 2: "func", 3: "section", 4: "file", 5: "common", 6: "tls", 10: "ifunc"}
VISIBILITY = {0: "default", 1: "internal", 2: "hidden", 3: "protected"}
CALL_TYPES = {282: "JUMP26", 283: "CALL26"}


class ElfError(ValueError):
    pass


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


def is_debug(name: str) -> bool:
    """Exactly the DWARF sections and their RELA sections (C-3b)."""
    return name.startswith((".debug_", ".rela.debug_"))


def _keys(elf: dict) -> dict[int, str]:
    """Section index -> stable key: name#occurrence among sections of that name (debug ones included in neither)."""
    seen: dict[str, int] = {}
    keys = {}
    for section in elf["sections"]:
        name = section["name"]
        if section["index"] == 0:
            keys[0] = "<null>"
            continue
        if is_debug(name):
            keys[section["index"]] = None
            continue
        keys[section["index"]] = f"{name}#{seen.get(name, 0)}"
        seen[name] = seen.get(name, 0) + 1
    return keys


def _symbol_name(elf: dict, keys: dict, symbol: dict) -> str:
    if symbol["type"] == "section":
        key = keys.get(symbol["shndx"])
        return f"<section {key if key is not None else 'debug'}>"
    return symbol["name"]


def _resolved(keys: dict, symbol: dict) -> tuple:
    """A relocation's symbol resolved independently of symbol-table indices: (name, section key, value) (D-2)."""
    if symbol["type"] == "section":
        key = keys.get(symbol["shndx"])
        return (f"<section {key if key is not None else 'debug'}>", None, 0)
    section = keys.get(symbol["shndx"]) if 0 < symbol["shndx"] < 0xFF00 else f"<special {symbol['shndx']}>"
    return (symbol["name"], section, symbol["value"])


def nondebug_view(elf: dict) -> dict:
    keys = _keys(elf)
    sections, groups, relocations = {}, {}, {}
    for section in elf["sections"]:
        key = keys.get(section["index"])
        if section["index"] == 0 or key is None:
            continue
        kind = section["type"]
        if kind in (SHT_SYMTAB, SHT_STRTAB):
            continue
        if kind in (SHT_RELA, SHT_REL):
            target = keys.get(section["info"])
            if target is None:
                # A non-debug-named relocation section of a debug section is not excluded: compared raw.
                sections[key] = (kind, section["flags"], section["align"], section["entsize"], section["size"],
                                 section["bytes"].hex())
                continue
            symbols = elf["symbols"]
            relocations[target] = [
                (entry["offset"], entry["type"], _resolved(keys, symbols[entry["symbol"]]),
                 symbols[entry["symbol"]]["binding"], entry["addend"])
                for entry in sorted(elf["relocations"][section["index"]], key=lambda e: (e["offset"], e["type"]))]
            continue
        if kind == SHT_GROUP:
            words = struct.unpack_from(f"<{section['size'] // 4}I", section["bytes"])
            members = [keys.get(member) for member in words[1:]]
            groups[key] = (words[0], [member for member in members if member is not None],
                           _symbol_name(elf, keys, elf["symbols"][section["info"]]))
            continue
        sections[key] = (kind, section["flags"], section["align"], section["entsize"], section["size"],
                         section["bytes"].hex() if kind != SHT_NOBITS else None)
    symbols = sorted(
        (_symbol_name(elf, keys, symbol), symbol["binding"], symbol["type"], symbol["visibility"],
         keys.get(symbol["shndx"], f"<special {symbol['shndx']}>") if symbol["shndx"] < 0xFF00 else f"<special {symbol['shndx']}>",
         symbol["value"], symbol["size"])
        for symbol in elf["symbols"][1:]
        if not (symbol["type"] == "section" and 0 < symbol["shndx"] < 0xFF00 and keys.get(symbol["shndx"]) is None))
    return {"sections": sections, "groups": groups, "relocations": relocations, "symbols": symbols}


MERGE_STRING_LABEL = re.compile(r"\.LC[0-9]+")
SHF_MERGE_STRINGS = 0x30


def _symbol_tuple(keys: dict, symbol: dict) -> tuple:
    return (_symbol_name(None, keys, symbol), symbol["binding"], symbol["type"], symbol["visibility"],
            keys.get(symbol["shndx"], f"<special {symbol['shndx']}>") if symbol["shndx"] < 0xFF00
            else f"<special {symbol['shndx']}>", symbol["value"], symbol["size"])


def merge_string_label(reference: dict, candidate: dict, symbol: dict) -> dict | str:
    """D-2 (reviewer 98): an augmented-only symbol is a debug-only merge-string label iff STB_LOCAL, STT_NOTYPE,
    size 0, spelled .LC<digits>, in a SHF_MERGE|SHF_STRINGS section whose bytes, flags, entsize and alignment equal
    the build object's, at the start of a NUL-terminated string, named by no relocation of any non-debug section and
    by no group signature. Returns its record (with its debug referrers) or the reason it is not one."""
    keys_c, keys_r = _keys(candidate), _keys(reference)
    if symbol["binding"] != "local" or symbol["type"] != "notype" or symbol["size"] != 0:
        return "not a local NOTYPE size-0 symbol"
    if not MERGE_STRING_LABEL.fullmatch(symbol["name"]):
        return "not a compiler .LC<digits> label"
    if not 0 < symbol["shndx"] < 0xFF00 or keys_c.get(symbol["shndx"]) is None:
        return "not in a non-debug section"
    section = candidate["sections"][symbol["shndx"]]
    key = keys_c[symbol["shndx"]]
    if section["flags"] & SHF_MERGE_STRINGS != SHF_MERGE_STRINGS:
        return f"section {key} is not SHF_MERGE|SHF_STRINGS"
    twin = next((s for s in reference["sections"] if keys_r.get(s["index"]) == key), None)
    if twin is None or (twin["bytes"], twin["flags"], twin["entsize"], twin["align"], twin["type"]) != \
            (section["bytes"], section["flags"], section["entsize"], section["align"], section["type"]):
        return f"section {key} differs from the build object's (bytes, flags, entsize or alignment)"
    value, data = symbol["value"], section["bytes"]
    if not (value < len(data) and (value == 0 or data[value - 1] == 0) and data.find(b"\0", value) >= 0):
        return "not at the start of a NUL-terminated string"
    referrers: dict[str, int] = {}
    for rela in candidate["sections"]:
        if rela["type"] != SHT_RELA:
            continue
        target = candidate["sections"][rela["info"]]["name"]
        count = sum(1 for entry in candidate["relocations"][rela["index"]] if entry["symbol"] == symbol["index"])
        if not count:
            continue
        if not is_debug(target):
            return f"named by {count} relocation(s) of non-debug section {target}"
        referrers[rela["name"]] = count
    if any(group["type"] == SHT_GROUP and group["info"] == symbol["index"] for group in candidate["sections"]):
        return "a group signature"
    end = data.find(b"\0", value)
    return {"name": symbol["name"], "section": key, "value": value,
            "string_sha256": __import__("hashlib").sha256(data[value:end]).hexdigest(), "debug_referrers": referrers}


def compare_nondebug_detail(reference, candidate) -> tuple[list[str], list[dict]]:
    """(differences, excluded debug-only merge-string labels) between the non-debug views of two relocatable
    objects. Only augmented(candidate)-only symbols can be excluded, each by merge_string_label(); every other
    symbol is compared as a multiset and every non-debug relocation by resolved (name, section, value)."""
    ref_elf, cand_elf = read_elf(reference), read_elf(candidate)
    left, right = nondebug_view(ref_elf), nondebug_view(cand_elf)
    differences = []
    for part in ("sections", "groups", "relocations"):
        for key in sorted(set(left[part]) | set(right[part])):
            if left[part].get(key) != right[part].get(key):
                differences.append(f"{part} {key} differs" if key in left[part] and key in right[part]
                                   else f"{part} {key} only in {'reference' if key in left[part] else 'candidate'}")
    from collections import Counter
    only_left = Counter(left["symbols"]) - Counter(right["symbols"])
    only_right = Counter(right["symbols"]) - Counter(left["symbols"])
    excluded, unexplained = [], []
    if only_right:
        keys_c = _keys(cand_elf)
        by_tuple: dict[tuple, list] = {}
        for symbol in cand_elf["symbols"][1:]:
            by_tuple.setdefault(_symbol_tuple(keys_c, symbol), []).append(symbol)
        for item, count in sorted(only_right.items(), key=str):
            for symbol in by_tuple.get(item, [])[:count]:
                verdict = merge_string_label(ref_elf, cand_elf, symbol)
                if isinstance(verdict, dict):
                    excluded.append(verdict)
                else:
                    unexplained.append((item, verdict))
    if only_left or unexplained:
        differences.append(f"symbols differ: reference-only {sorted(only_left)[:3]}, candidate-only "
                           f"{[(item, why) for item, why in unexplained][:3]}")
    return differences, excluded


def compare_nondebug(reference, candidate) -> list[str]:
    """Differences between the non-debug views of two relocatable objects (empty: equal); D-2 exclusions apply."""
    return compare_nondebug_detail(reference, candidate)[0]


def read_symbols(path) -> list[dict]:
    """Symbols only (name, binding, type, shndx), reading just the section headers and the symbol/string tables."""
    with open(path, "rb") as stream:
        header = stream.read(64)
        if header[:4] != b"\x7fELF" or header[4] != 2 or header[5] != 1:
            raise ElfError(f"{path}: not an ELF64 little-endian file")
        shoff, = struct.unpack_from("<Q", header, 0x28)
        shentsize, shnum, _ = struct.unpack_from("<HHH", header, 0x3A)
        stream.seek(shoff)
        table = stream.read(shentsize * shnum)
        headers = [struct.unpack_from("<IIQQQQIIQQ", table, index * shentsize) for index in range(shnum)]
        def content(index):
            _, kind, _, _, offset, size, *_ = headers[index]
            stream.seek(offset)
            return stream.read(size)
        symbols = []
        for index, (_, kind, _, _, _, size, link, *_) in enumerate(headers):
            if kind != SHT_SYMTAB:
                continue
            data, strings = content(index), content(link)
            for position in range(size // 24):
                name, info, other, shndx, value, length = struct.unpack_from("<IBBHQQ", data, position * 24)
                symbols.append({"name": _string(strings, name), "binding": BINDINGS.get(info >> 4, str(info >> 4)),
                                "type": TYPES.get(info & 15, str(info & 15)), "shndx": shndx})
        return symbols


class _Pool:
    """Symbols of one section sorted by address: the symbol covering an address (prefix maximum of the ends bounds
    the backward scan, so an uncovered address is answered without a full scan)."""
    def __init__(self, symbols):
        self.symbols = sorted(symbols, key=lambda item: (item["value"], -item["size"]))
        self.starts = [item["value"] for item in self.symbols]
        self.reach, furthest = [], -1
        for item in self.symbols:
            furthest = max(furthest, item["value"] + max(item["size"], 1))
            self.reach.append(furthest)

    def covering(self, address: int):
        position = bisect.bisect_right(self.starts, address) - 1
        while position >= 0 and self.reach[position] > address:
            symbol = self.symbols[position]
            if symbol["value"] <= address < symbol["value"] + max(symbol["size"], 1):
                return symbol
            position -= 1
        return None

    def start(self, address: int):
        position = bisect.bisect_left(self.starts, address)
        return self.symbols[position] if position < len(self.starts) and self.starts[position] == address else None


def _partners(a: str, b: str) -> bool:
    """A function and its cold partition (GCC emits <f>.cold[.N] as its own STT_FUNC symbol)."""
    def base(name):
        head, dot, tail = name.rpartition(".cold")
        return head if dot and (tail == "" or tail.lstrip(".").isdigit()) else name
    return base(a) == base(b)


def _mapping(elf: dict) -> dict[int, tuple]:
    """Section -> (sorted starts, kinds) of its AArch64 mapping symbols ($x code, $d data)."""
    found: dict[int, list] = {}
    for symbol in elf["symbols"]:
        if symbol["name"][:2] in ("$x", "$d") and (len(symbol["name"]) == 2 or symbol["name"][2] == "."):
            found.setdefault(symbol["shndx"], []).append((symbol["value"], symbol["name"][:2]))
    return {shndx: ([start for start, _ in sorted(items)], [kind for _, kind in sorted(items)])
            for shndx, items in found.items()}


def _indirect_branches(elf: dict, function: dict, mapping: dict) -> dict:
    section = elf["sections"][function["shndx"]]
    starts, kinds = mapping.get(function["shndx"], ([], []))
    found = {"blr": [], "br": []}
    for offset in range(function["value"], function["value"] + function["size"] - 3, 4):
        position = bisect.bisect_right(starts, offset) - 1
        if position >= 0 and kinds[position] == "$d":
            continue
        word, = struct.unpack_from("<I", section["bytes"], offset)
        relative = offset - function["value"]
        if (word & 0xFFFFFC1F) == 0xD63F0000 or (word & 0xFEFFF800) == 0xD63F0800:
            found["blr"].append(relative)
        elif (word & 0xFFFFFC1F) == 0xD61F0000 or (word & 0xFEFFF800) == 0xD61F0800:
            found["br"].append(relative)
    return {kind: offsets for kind, offsets in found.items() if offsets}


def function_graph(elf: dict) -> dict:
    """{"functions": {key: {name, binding, section, value, size}}, "exports": {non-local name: node key},
    "edges": {node: [[relocation type, "node"|"name", target]]}, "indirect": {function key: {blr|br: [offsets]}},
    "failures": [...]}. A "name" target is a non-local or undefined symbol, resolved across objects by the caller
    (every non-local definition of it, or an import when nothing defines it)."""
    sections, symbols = elf["sections"], elf["symbols"]
    executable = {s["index"] for s in sections if s["flags"] & SHF_EXECINSTR}
    keys = _keys(elf)
    functions = [s for s in symbols if s["type"] in ("func", "ifunc") and 0 < s["shndx"] < SHN_LORESERVE]
    counts: dict[str, int] = {}
    for symbol in functions:
        counts[symbol["name"]] = counts.get(symbol["name"], 0) + 1
    def own_key(symbol):
        return symbol["name"] if counts[symbol["name"]] == 1 else f"{symbol['name']}@{symbol['shndx']}+{symbol['value']}"
    canonical: dict[tuple, str] = {}
    for symbol in sorted(functions, key=own_key):
        canonical.setdefault((symbol["shndx"], symbol["value"], symbol["size"]), own_key(symbol))
    def function_key(symbol):
        """Aliases (same section, address and size, e.g. C1/C2 constructors) share one node."""
        return canonical[(symbol["shndx"], symbol["value"], symbol["size"])]
    grouped: dict[int, list] = {}
    for symbol in functions:
        grouped.setdefault(symbol["shndx"], []).append(symbol)
    by_section = {shndx: _Pool(items) for shndx, items in grouped.items()}
    empty = _Pool([])
    grouped = {}
    data_counts: dict[str, int] = {}
    for symbol in symbols:
        if symbol["type"] in ("object", "tls") and 0 < symbol["shndx"] < SHN_LORESERVE and symbol["size"] > 0 \
                and symbol["shndx"] not in executable:
            grouped.setdefault(symbol["shndx"], []).append(symbol)
            data_counts[symbol["name"]] = data_counts.get(symbol["name"], 0) + 1
    data_by_section = {shndx: _Pool(items) for shndx, items in grouped.items()}
    def data_key(shndx, address):
        covering = data_by_section.get(shndx, empty).covering(address)
        if covering is None:
            return f"section:{keys.get(shndx) or sections[shndx]['name']}"
        name = covering["name"]
        return f"data:{name}" if data_counts[name] == 1 else f"data:{name}@{shndx}+{covering['value']}"
    exports = {}
    for symbol in symbols:
        if symbol["binding"] == "local" or not (0 < symbol["shndx"] < SHN_LORESERVE or symbol["shndx"] == SHN_COMMON):
            continue
        if symbol["shndx"] == SHN_COMMON:
            exports[symbol["name"]] = f"data:{symbol['name']}"
        elif symbol["shndx"] in executable:
            covering = by_section.get(symbol["shndx"], empty).covering(symbol["value"])
            if covering is not None:
                exports[symbol["name"]] = function_key(covering)
        else:
            exports[symbol["name"]] = data_key(symbol["shndx"], symbol["value"])
    edges: dict[str, set] = {}
    failures: list[str] = []
    mapping = _mapping(elf)

    def code_target(owner, rtype, shndx, address, where):
        """Target node of a reference into executable section shndx at address (None after a failure)."""
        pool = by_section.get(shndx, empty)
        start = pool.start(address)
        if start is not None:
            return function_key(start)
        if rtype == R_CALL26:
            failures.append(f"{where}: CALL26 target {sections[shndx]['name']}+{address:#x} is no function start")
            return None
        covering = pool.covering(address)
        if covering is None:
            failures.append(f"{where}: reference to {sections[shndx]['name']}+{address:#x}, which no function covers")
            return None
        if owner is not None and owner.startswith("fn:") and _partners(owner[3:], covering["name"]):
            return function_key(covering)
        failures.append(f"{where}: reference into the interior of {covering['name']} (+{address - covering['value']:#x})")
        return None

    for section in sections:
        if section["type"] != SHT_RELA or not section["info"]:
            continue
        target_section = sections[section["info"]]
        if not target_section["flags"] & SHF_ALLOC:
            continue
        in_code = target_section["index"] in executable
        for entry in elf["relocations"][section["index"]]:
            where = f"{target_section['name']}+{entry['offset']:#x}"
            if in_code:
                owner_symbol = by_section.get(target_section["index"], empty).covering(entry["offset"])
                if owner_symbol is None:
                    failures.append(f"{where}: relocation (type {entry['type']}) covered by no function symbol")
                    continue
                owner, owner_tag = function_key(owner_symbol), f"fn:{owner_symbol['name']}"
            else:
                owner, owner_tag = data_key(target_section["index"], entry["offset"]), None
            symbol = symbols[entry["symbol"]]
            rtype = entry["type"]
            if entry["symbol"] == 0:
                continue  # No symbol: an absolute or self-contained value (no reference).
            if symbol["shndx"] == SHN_UNDEF:
                if not symbol["name"]:
                    failures.append(f"{where}: relocation against an unnamed undefined symbol")
                    continue
                target = ("name", symbol["name"])
            elif symbol["shndx"] == SHN_ABS:
                continue  # An absolute value, no code or data.
            elif symbol["shndx"] == SHN_COMMON or symbol["binding"] != "local" and symbol["type"] != "section":
                if rtype == R_CALL26 and symbol["type"] in ("func", "ifunc") and entry["addend"]:
                    failures.append(f"{where}: CALL26 to {symbol['name']}+{entry['addend']:#x} is no function start")
                    continue
                target = ("name", symbol["name"])
            else:
                shndx = symbol["shndx"]
                address = (0 if symbol["type"] == "section" else symbol["value"]) + entry["addend"]
                if shndx in executable:
                    node = code_target(owner_tag, rtype, shndx, address, where)
                    if node is None:
                        continue
                    target = ("node", node)
                elif sections[shndx]["flags"] & SHF_ALLOC:
                    target = ("node", data_key(shndx, address))
                else:
                    failures.append(f"{where}: reference into non-allocated section {sections[shndx]['name']}")
                    continue
            kind = "CALL26" if rtype == R_CALL26 else R_BRANCHES.get(rtype, f"R{rtype}")
            edges.setdefault(owner, set()).add((kind, *target))
    # C-5: decode the code bytes (relocation-free PC-relative references) and cross-check relocated ones.
    relocated_at: dict[tuple, list] = {}
    for section in sections:
        if section["type"] == SHT_RELA and section["info"] in executable:
            for entry in elf["relocations"][section["index"]]:
                relocated_at.setdefault((section["info"], entry["offset"]), []).append(entry["type"])
    decoded_edges = 0
    seen_ranges: set = set()
    covered: dict[int, list] = {}
    for symbol in functions:
        span = (symbol["shndx"], symbol["value"], max(symbol["size"], 0))
        covered.setdefault(symbol["shndx"], []).append((symbol["value"], symbol["value"] + symbol["size"]))
        if span in seen_ranges:
            continue
        seen_ranges.add(span)
        owner = function_key(symbol)
        shndx, start, end = symbol["shndx"], symbol["value"], symbol["value"] + symbol["size"]
        code = sections[shndx]["bytes"]
        starts, kinds = mapping.get(shndx, ([], []))
        for offset in range(start, end - 3, 4):
            position = bisect.bisect_right(starts, offset) - 1
            if position >= 0 and kinds[position] == "$d":
                continue
            word, = struct.unpack_from("<I", code, offset)
            decoded = _decode(word)
            types = relocated_at.get((shndx, offset))
            where = f"{symbol['name']}+{offset - start:#x} ({sections[shndx]['name']}+{offset:#x}, word {word:#010x})"
            if decoded is None:
                if types and set(types) & PC_RELATIVE_RELOCATIONS:
                    failures.append(f"{where}: PC-relative relocation {types} on an instruction that is not PC-relative")
                continue
            kind, displacement = decoded
            if types:
                if not set(types) <= CLASS_RELOCATIONS[kind]:
                    failures.append(f"{where}: {kind} carries relocation(s) {types}, expected one of "
                                    f"{sorted(CLASS_RELOCATIONS[kind])}")
                continue  # Covered by the relocation edge.
            if kind == "adrp":
                failures.append(f"{where}: ADRP without a relocation")
                continue
            target = offset + displacement
            if start <= target < end:
                if kind == "bl" and target != start:
                    failures.append(f"{where}: BL without relocation into its own interior (+{target - start:#x})")
                elif kind == "bl":
                    edges.setdefault(owner, set()).add(("decoded-bl", "node", owner))
                    decoded_edges += 1
                continue  # A branch, address or literal within the function itself.
            if not 0 <= target < sections[shndx]["size"]:
                failures.append(f"{where}: {kind} without relocation leaves its section (target {target:#x})")
                continue
            pool = by_section.get(shndx, empty)
            hit = pool.start(target)
            if hit is None:
                covering = pool.covering(target)
                if covering is not None and _partners(symbol["name"], covering["name"]):
                    hit = covering
                elif covering is not None:
                    failures.append(f"{where}: {kind} without relocation into the interior of {covering['name']} "
                                    f"(+{target - covering['value']:#x})")
                    continue
                else:
                    failures.append(f"{where}: {kind} without relocation to {sections[shndx]['name']}+{target:#x}, "
                                    f"which no function covers")
                    continue
            edges.setdefault(owner, set()).add((f"decoded-{kind}", "node", function_key(hit)))
            decoded_edges += 1
    for shndx in sorted(executable):
        section = sections[shndx]
        if section["type"] == SHT_NOBITS or not section["size"]:
            continue
        starts, kinds = mapping.get(shndx, ([], []))
        intervals = sorted(covered.get(shndx, []))
        cursor, uncovered = 0, []
        for low, high in intervals + [(section["size"], section["size"])]:
            for offset in range(cursor - cursor % 4 if cursor % 4 == 0 else cursor + 4 - cursor % 4, low - 3, 4):
                position = bisect.bisect_right(starts, offset) - 1
                if position >= 0 and kinds[position] == "$d":
                    continue
                word, = struct.unpack_from("<I", section["bytes"], offset)
                if word not in NOP_WORDS:
                    uncovered.append(f"{section['name']}+{offset:#x} (word {word:#010x})")
            cursor = max(cursor, high)
        if uncovered:
            failures.append(f"{len(uncovered)} code word(s) outside every function symbol: {uncovered[:4]}")
    nodes: dict[str, dict] = {}
    for symbol in functions:
        node = nodes.setdefault(function_key(symbol), {"section": sections[symbol["shndx"]]["name"], "shndx": symbol["shndx"],
                                                       "value": symbol["value"], "size": symbol["size"], "names": {}})
        node["names"][symbol["name"]] = symbol["binding"]
    symbol_nodes, node_sections = {}, {}
    for symbol in symbols:
        if not 0 < symbol["shndx"] < SHN_LORESERVE or symbol["type"] in ("section", "file") or not symbol["name"]:
            continue
        if symbol["type"] in ("func", "ifunc"):
            key = function_key(symbol)
        elif symbol["shndx"] in executable:
            continue
        else:
            key = data_key(symbol["shndx"], symbol["value"])
        symbol_nodes.setdefault(symbol["name"], set()).add(key)
        node_sections[key] = sections[symbol["shndx"]]["name"]
    for targets in edges.values():
        for edge in targets:
            if edge[1] == "node" and edge[2].startswith("section:"):
                node_sections[edge[2]] = edge[2][len("section:"):].rsplit("#", 1)[0]
    return {"functions": nodes, "decoded_edges": decoded_edges,
            "symbol_nodes": {name: sorted(keys) for name, keys in sorted(symbol_nodes.items())},
            "node_sections": dict(sorted(node_sections.items())),
            "exports": exports,
            "edges": {node: sorted(list(edge) for edge in targets) for node, targets in sorted(edges.items())},
            "indirect": {key: found for key, node in nodes.items() if (found := _indirect_branches(elf, node, mapping))},
            "failures": failures}
