#!/usr/bin/env python3
"""D-1 controls (reviewer 98) on REAL production objects (read-only): develop and candidate positive sites, then
mutations of the exact DWARF evidence, each of which must FAIL. Tool controls, never evidence.
argv: <checker dir> <develop object> <develop kernel header> <candidate object> <candidate kernel header> <fortify header>"""
import copy, hashlib, importlib.util, json, sys
from pathlib import Path
C = Path(sys.argv[1])
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, C / file); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
ev = load("cpu_native_tile_register_evidence", "CPUNativeTileRegisterEvidence.py")
dw = load("cpu_native_tile_dwarf_sites", "CPUNativeTileDwarfSites.py")
DEV, DEVH, CAND, CANDH, FORT = (Path(a) for a in sys.argv[2:7])
KER = "src/v2/kernels/cpu/gemm/CPUNativeQuantisedMultiScaleKernels.h"
results = []
def check(name, expect, outcome):
    ok = str(outcome).startswith(expect); results.append(ok)
    print(f"{'ok ' if ok else 'BAD'} {name}: expected {expect}; got {str(outcome)[:280]}", flush=True)
def compact(obj, base):
    elf = dw.read_elf(obj.read_bytes())
    sym = next(s for s in elf["symbols"] if s["type"] == "func" and s["shndx"] and base in s["name"] and "ReductionE0" in s["name"])
    return elf, sym
def site(records, ordinal, callee, header, fort=FORT.read_text(), lanes=4):
    ev.install_bounded_context({"f": []}, {KER: header, "string_fortified.h": fort}, {"f": records})
    try:
        result = ev.bounded_memcpy_site("f", None, lanes, ordinal, callee)
    finally:
        ev.install_bounded_context({}, {}, None)
    return f"PASS {result['line'].split(':')[-1]}" if isinstance(result, dict) else f"FAIL {result}"
for tree, obj, header, base, lines in (("develop", DEV, DEVH, "multi_scale4rowsILi10ELi4ELm1E", "598,708"),
                                        ("candidate", CAND, CANDH, "multi_scale11rowsProgramILb0ELi10ELi4ELm1E", "1110,1354")):
    elf, sym = compact(obj, base)
    records = dw.memcpy_sites(elf, sym, {"rows", "rowsProgram"})
    text = header.read_text()
    got = sorted({site(records, i, r["callee"], text).split()[-1] for i, r in enumerate(records)})
    check(f"D1a-{tree}-real-sites-pass", f"PASS {lines}", f"PASS {','.join(got)}" if all(site(records, i, r['callee'], text).startswith("PASS") for i, r in enumerate(records)) else f"FAIL {[site(records, i, r['callee'], text) for i, r in enumerate(records)]}")
    good = next(i for i, r in enumerate(records) if r.get("kind") == "inlined")
    def mutated(**changes):
        recs = copy.deepcopy(records); recs[good].update(changes); return recs
    other = 1352 if tree == "candidate" else 706
    check(f"D1b-{tree}-call-line-{other}-fails", "FAIL", site(mutated(call_line=other), good, records[good]["callee"], text))
    check(f"D1b2-{tree}-call-line-1109-fails", "FAIL", site(mutated(call_line=1109), good, records[good]["callee"], text))
    check(f"D1c-{tree}-call-file-hash-fails", "FAIL", site(records, good, records[good]["callee"], text.replace("Lanes <= columns", "Lanes<=columns", 1)))
    check(f"D1d-{tree}-fortify-hash-fails", "FAIL", site(records, good, records[good]["callee"], text, FORT.read_text() + "\n"))
    check(f"D1e-{tree}-non-compact-owner-fails", "FAIL", site(mutated(owner="pack<4>"), good, records[good]["callee"], text))
    check(f"D1f-{tree}-non-memcpy-bl-fails", "FAIL", site(records, good, "plain_call", text))
    check(f"D1g-{tree}-dwarf-error-fails", "FAIL", site(mutated(error="x"), good, records[good]["callee"], text))
    # PC outside every inlined range: the call becomes "direct"; its line-table row (no row bound) cannot admit it.
    cache = {}
    dw.memcpy_sites(elf, sym, {"rows", "rowsProgram"}, cache=cache)
    die = records[good]["die"]
    shifted = copy.deepcopy(cache); shifted["ranges"][die] = [(s, lo + 0x100000, hi + 0x100000) for s, lo, hi in shifted["ranges"][die]]
    outside = dw.memcpy_sites(elf, sym, {"rows", "rowsProgram"}, cache=shifted)
    check(f"D1h-{tree}-pc-outside-ranges-fails", "FAIL", site(outside, good, records[good]["callee"], text) if outside[good].get("die") != die else "PASS (still covered)")
    # Duplicate overlapping innermost instance: a second DIE with the same parent and ranges.
    dup = copy.deepcopy(cache); twin = max(dup["dies"]) + 1
    dup["dies"][twin] = copy.deepcopy(dup["dies"][die]); dup["ranges"][twin] = dup["ranges"][die]; dup["inlined"] = dup["inlined"] + [twin]
    duplicated = dw.memcpy_sites(elf, sym, {"rows", "rowsProgram"}, cache=dup)
    check(f"D1i-{tree}-duplicate-instance-fails", "FAIL", site(duplicated, good, records[good]["callee"], text))
    # A memcpy inlined from the admitted line into a NON-compact function (owner name not admitted).
    check(f"D1j-{tree}-owner-name-set-excludes-other-functions", "FAIL",
          "FAIL " + str(dw.memcpy_sites(elf, sym, {"pack"})[good].get("error", "PASS"))[:120]
          if dw.memcpy_sites(elf, sym, {"pack"})[good].get("error") else "PASS")
print(json.dumps({"controls": len(results), "ok": sum(results)}))
sys.exit(0 if all(results) else 1)
