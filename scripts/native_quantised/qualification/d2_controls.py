#!/usr/bin/env python3
"""D-2 controls (reviewer 98) on the REAL candidate NEON Release WeightManager.cpp.o pair (build object and its
debug-augmented replay, both read-only): the real .LC1991-.LC1993 are excluded with their debug referrers; each
mutation of one excluded symbol (or its section / references) FAILS. Tool controls; never evidence."""
import json, struct, sys
from pathlib import Path
sys.path.insert(0, "/tools")
import elf_tools as E
REF, AUG, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
results = []
def check(name, expect, outcome):
    ok = str(outcome).startswith(expect); results.append(ok)
    print(f"{'ok ' if ok else 'BAD'} {name}: expected {expect}; got {str(outcome)[:260]}", flush=True)
def verdict(aug_path, ref_path=REF):
    differences, excluded = E.compare_nondebug_detail(ref_path, aug_path)
    return ("PASS " + str(sorted(e["name"] for e in excluded))) if not differences else f"FAIL {differences[0][:200]}"
data = bytearray(AUG.read_bytes()); elf = E.read_elf(bytes(data))
symtab = next(s for s in elf["sections"] if s["type"] == E.SHT_SYMTAB)
target = next(s for s in elf["symbols"] if s["name"] == ".LC1991")
entry = symtab["offset"] + target["index"] * 24
differences, excluded = E.compare_nondebug_detail(REF, AUG)
(OUT / "excluded.json").write_text(json.dumps(excluded, indent=1))
check("D2a-real-LC1991-1993-excluded", "PASS ['.LC1991', '.LC1992', '.LC1993']", verdict(AUG))
check("D2a2-debug-referrers-recorded", "PASS", "PASS" if all(e["debug_referrers"] and all(k.startswith(".rela.debug_") for k in e["debug_referrers"]) for e in excluded) else f"FAIL {excluded}")
def mutated(name, patch):
    copy = bytearray(data); patch(copy); path = OUT / f"{name}.o"; path.write_bytes(bytes(copy)); return verdict(path)
check("D2b-global-fails", "FAIL", mutated("global", lambda b: b.__setitem__(entry + 4, (1 << 4) | (b[entry + 4] & 15))))
check("D2c-object-type-fails", "FAIL", mutated("object", lambda b: b.__setitem__(entry + 4, (b[entry + 4] & 0xF0) | 1)))
check("D2d-nonzero-size-fails", "FAIL", mutated("size", lambda b: struct.pack_into("<Q", b, entry + 16, 8)))
check("D2e-not-string-start-fails", "FAIL", mutated("interior", lambda b: struct.pack_into("<Q", b, entry + 8, target["value"] + 1)))
other = next(s for s in elf["symbols"] if s["type"] == "file" and s["name"])
name_offset = struct.unpack_from("<I", data, symtab["offset"] + other["index"] * 24)[0]
check("D2f-non-LC-name-fails", "FAIL", mutated("name", lambda b: struct.pack_into("<I", b, entry, name_offset)))
section = elf["sections"][target["shndx"]]
header = struct.unpack_from("<Q", data, 0x28)[0] + section["index"] * struct.unpack_from("<H", data, 0x3A)[0]
check("D2g-non-merge-section-fails", "FAIL", mutated("flags", lambda b: struct.pack_into("<Q", b, header + 8, section["flags"] & ~0x10)))
check("D2h-section-bytes-differ-fails", "FAIL", mutated("bytes", lambda b: b.__setitem__(section["offset"] + target["value"], b[section["offset"] + target["value"]] ^ 1)))
rela = next(s for s in elf["sections"] if s["type"] == E.SHT_RELA and not E.is_debug(elf["sections"][s["info"]]["name"]) and elf["relocations"][s["index"]])
def nondebug_ref(b):
    info_at = rela["offset"] + 8
    info = struct.unpack_from("<Q", b, info_at)[0]
    struct.pack_into("<Q", b, info_at, (target["index"] << 32) | (info & 0xFFFFFFFF))
check("D2i-nondebug-relocation-fails", "FAIL", mutated("nondebug-ref", nondebug_ref))
check("D2j-reference-only-symbol-fails", "FAIL", verdict(REF, AUG))
print(json.dumps({"controls": len(results), "ok": sum(results)}))
sys.exit(0 if all(results) else 1)
