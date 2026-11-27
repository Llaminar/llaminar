#!/usr/bin/env python3
"""N-1/N-2 controls (reviewer 91): a real nvcc TU discarded / selected / reached / unprovable, and objects without
recorded deps (N-2) on the nv project. Tool controls only."""
import argparse, importlib.util, json, lzma, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, "/tools")
def load(name):
    spec = importlib.util.spec_from_file_location(name, f"/tools/{name}.py"); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
b1, pp = load("b1_replay"), load("postprocess")
BUILD, SRC = "/src/nv/build", "/src/nv"
CONTEXT = b1.load_context(argparse.Namespace(manifest=f"{SRC}/manifest.txt", audited_headers=f"{SRC}/audited.txt", src_root=SRC))
results = []
def check(name, expect, function):
    try:
        outcome = function(Path("/out") / name)
    except (b1.Failure, SystemExit) as error:
        outcome = f"FAIL: {error}"
    ok = str(outcome).startswith(expect); results.append(ok)
    print(f"{'ok ' if ok else 'BAD'} {name}: expected {expect}; got {str(outcome)[:300]}", flush=True)
def run(lib, evidence):
    evidence.mkdir(parents=True)
    return b1.run_b1(BUILD, [f"lib{lib}.so"], evidence / "b1", CONTEXT, 4) if (evidence / "b1").mkdir() is None else None
def nvrecord(summary):
    return next(t for t in summary["tus"] if t["object"].endswith(".cu.o"))
def link(lib, evidence, summary):
    pp.TARGETS = (f"lib{lib}.so",)
    artifacts = evidence / "link"; artifacts.mkdir()
    record = pp.link_record(BUILD, f"lib{lib}.so", artifacts)
    symbols, passed = pp.audited_symbols(summary)
    return record, pp.non_gnu_selection(BUILD, artifacts, summary, passed), pp.selected_copies(BUILD, artifacts, symbols, passed)
def discarded(evidence):
    summary = run("gnufirst", evidence)
    if summary["closure"]["non_gnu_shared_reached"] != {"_Z11helper_copyi": ["CMakeFiles/gnufirst.dir/k.cu.o"],
                                                         "kTable": ["CMakeFiles/gnufirst.dir/k.cu.o"],
                                                         "tls_counter": ["CMakeFiles/gnufirst.dir/k.cu.o"]}:
        return f"BAD shared-reached {summary['closure']['non_gnu_shared_reached']}"
    record = nvrecord(summary)
    _, selection, copies = link("gnufirst", evidence, summary)
    if "classification" not in record: return f"DIAG {record['status']} {record.get('reason')} {summary['closure'].get('failures',[])[:2]} {[(t['object'],t['status'],t.get('reason','')[:200]) for t in summary['tus'] if t['status']!='PASS']}"
    entry = selection["objects"][record["object"]]
    return (f"{summary['status']} {record['status']} {record['classification']['classes']} audited "
            f"{record['audited_shared']} map {selection['status']} {sorted(entry['shared'])} copies {copies['status']} "
            f"personality-not-audited {'DW.ref.__gxx_personality_v0' not in record['audited_shared'] and not any('personality' in f for f in selection['failures'])}")
check("N1a-discarded-nvcc-weak-copy-and-data-pass",
      "PASS PASS ['non-gnu-outside-closure'] audited ['_Z11helper_copyi', 'kTable', 'tls_counter'] map PASS "
      "['_Z11helper_copyi', 'kTable', 'tls_counter'] copies PASS personality-not-audited True", discarded)
def selected(evidence):
    summary = run("nvfirst", evidence)
    _, selection, copies = link("nvfirst", evidence, summary)
    return f"{selection['status']} {selection['failures'][:1]} copies {copies['status']}"
def selected_data(evidence):
    summary = run("nvfirst", evidence)
    _, selection, _ = link("nvfirst", evidence, summary)
    return "FAIL " + str([f for f in selection["failures"] if "kTable" in f][:1]) if any("kTable" in f for f in selection["failures"]) \
        else f"PASS {selection['failures']}"
check("Q2a-audited-data-selected-from-nvcc-fails", "FAIL ['CMakeFiles/nvfirst.dir/k.cu.o: kTable: FAIL (the non-GNU copy is selected)']", selected_data)
def missing_data_comdat(evidence):
    summary = run("gnufirst", evidence)
    pp.TARGETS = ("libgnufirst.so",)
    artifacts = evidence / "link"; artifacts.mkdir()
    pp.link_record(BUILD, "libgnufirst.so", artifacts)
    mapped = artifacts / "maps" / "libgnufirst.so.map.xz"
    head, _, body = lzma.open(mapped, "rt").read().partition("Linker script and memory map")
    kept = "\n".join(line for line in body.splitlines() if "kTable" not in line)
    with lzma.open(mapped, "wt") as stream:
        stream.write(head + "Linker script and memory map" + kept)
    symbols, passed = pp.audited_symbols(summary)
    selection = pp.non_gnu_selection(BUILD, artifacts, summary, passed)
    return "FAIL " + str([f for f in selection["failures"] if "kTable" in f][:1]) if selection["failures"] else "PASS"
check("Q2b-data-comdat-absent-from-map-fails", "FAIL ['CMakeFiles/gnufirst.dir/k.cu.o: kTable: FAIL (in no link map)']", missing_data_comdat)
def parser_finds_data_comdat(evidence):
    text = "Linker script and memory map\n .rodata._ZL6kTable\n                0x0000000000001000       0x10 CMakeFiles/x.dir/g.cpp.o\n"
    found = pp.map_definitions(text)
    return f"PASS {sorted(found.get('_ZL6kTable', []))}" if "_ZL6kTable" in found else "FAIL parser misses data COMDAT"
check("Q2c-parser-reads-data-comdat", "PASS ['CMakeFiles/x.dir/g.cpp.o']", parser_finds_data_comdat)
check("N1b-selected-nvcc-copy-fails", "FAIL ['CMakeFiles/nvfirst.dir/k.cu.o: _Z11helper_copyi: FAIL (the non-GNU copy is selected)']", selected)
def reached(evidence):
    summary = run("reached", evidence)
    return f"{summary['closure']['status']} {summary['closure']['failures'][:1]}"
check("N1c-reached-nvcc-function-fails", "FAIL ['the closure reaches _Z7nv_onlyi in CMakeFiles/reached.dir/k.cu.o, whose B1 is NON-GNU']", reached)
def declared(evidence):
    summary = run("declared", evidence)
    record = nvrecord(summary)
    return f"{record['status']}: {record.get('reason', '')}"
check("N1d-exclusive-definition-in-audited-header-fails-by-dwarf",
      "FAIL: non-GNU TU holds definitions not provably outside the audited headers (full nvcc provenance needed): ['_Z14nv_header_onlyi']", declared)
def defined_in_cu(evidence):
    summary = run("declared", evidence / "x")
    record = nvrecord(summary)
    entry = record["exclusive_classified"]["_Z11nv_declaredi"]
    return f"PASS {entry['rule']} {Path(entry['decl_file']).name} hashed {len(entry['sha256']) == 64}" \
        if isinstance(entry, dict) else f"FAIL {entry}"
check("Q3a-header-declared-cu-defined-classified-by-dwarf-decl-file", "PASS dwarf k2.cu hashed True", defined_in_cu)
def audited_header_entry(evidence):
    summary = run("declared", evidence / "y")
    return str(nvrecord(summary)["exclusive_classified"]["_Z14nv_header_onlyi"])
check("Q3b-audited-header-definition-named", "FAIL declared in audited header /src/nv/CPUNativeQuantisedMini.h", audited_header_entry)
import copy
KERNEL = {}
def kernel_summary(evidence):
    if "summary" not in KERNEL:
        KERNEL["summary"] = run("kernel", evidence)
    return KERNEL["summary"]
def generated(evidence):
    record = nvrecord(kernel_summary(evidence))
    classes = sorted({v["class"] for n, v in record["exclusive_classified"].items()
                      if isinstance(v, dict) and "class" in v})
    return f"{record['status']} {classes}"
check("G0-real-launch-stub-and-fatbin-wrapper-classified", "PASS ['nvcc-fatbin-wrapper', 'nvcc-generated-launch-stub']", generated)
def mutated(change):
    def control(evidence):
        record = copy.deepcopy(nvrecord(kernel_summary(Path("/out/G0-real-launch-stub-and-fatbin-wrapper-classified"))))
        stub = next(n for n in record["definitions"] if "__device_stub__" in n)
        wrapper = next(n for n in record["definitions"] if n.startswith("__fatbinwrap_"))
        shared, reached = set(), set()
        target = change(record, stub, wrapper, shared, reached)
        result = b1.classify_generated(target, record, CONTEXT, shared, reached)
        return result if isinstance(result, str) else f"PASS {result}"
    return control
def other_tu(record, stub, wrapper, shared, reached):
    declared = record["definition_declarations"][stub]
    declared["decl_file"] = declared["decl_file"].replace("tmpxft_", "tmpxft_ff", 1)
    return stub
check("G1c-stub-file-of-another-compile-fails", "FAIL the stub's unit", mutated(other_tu))
def no_kernel(record, stub, wrapper, shared, reached):
    record["stub_kernels"][stub]["defined_here"] = False
    return stub
check("G1d-stub-without-its-kernel-fails", "FAIL kernel", mutated(no_kernel))
check("G1e-stub-shared-with-gnu-fails", "FAIL launch stub shared", mutated(lambda r, s, w, sh, re: (sh.add(s), s)[1]))
check("G1b2-stub-reached-fails", "FAIL launch stub reached", mutated(lambda r, s, w, sh, re: (re.add(s), s)[1]))
def fatbin_function(record, stub, wrapper, shared, reached):
    record["definition_facts"][wrapper] = {"type": "func", "binding": "global", "section": ".text"}
    return wrapper
check("G2a-fatbinwrap-function-in-text-fails", "FAIL __fatbinwrap_", mutated(fatbin_function))
check("G2b-fatbinwrap-referenced-from-closure-fails", "FAIL fatbin wrapper referenced", mutated(lambda r, s, w, sh, re: (re.add(w), w)[1]))
def header_kernel(evidence):
    record = nvrecord(run("hdrkernel", evidence))
    stub = [v for n, v in record["exclusive_classified"].items() if "__device_stub__" in n]
    return f"{record['status']} {stub[0] if stub else 'no stub'}"
check("G1a-stub-of-kernel-in-audited-header-fails", "FAIL FAIL kernel declared in audited header", header_kernel)
def reached_stub(evidence):
    summary = run("reachstub", evidence)
    record = nvrecord(summary)
    stub = [v for n, v in record["exclusive_classified"].items() if "__device_stub__" in n]
    return f"{summary['closure']['status']} {record['status']} {stub[0] if stub else 'no stub'}"
check("G1b-stub-reached-from-a-program-fails", "FAIL FAIL FAIL launch stub reached from the closure", reached_stub)
# Q-2d: the -fPIC closure references audited data GOT-indirect (and TLS by descriptor); still resolved and audited.
def got_and_tls(evidence):
    summary = run("gnufirst", evidence)
    prog = next(t for t in summary["tus"] if t["object"].endswith("prog.cpp.o"))
    import elf_tools
    elf = elf_tools.read_elf(Path(BUILD) / prog["object"])
    kinds = {}
    for section in elf["sections"]:
        if section["type"] == elf_tools.SHT_RELA and not elf_tools.is_debug(elf["sections"][section["info"]]["name"]):
            for entry in elf["relocations"][section["index"]]:
                name = elf["symbols"][entry["symbol"]]["name"]
                if name in ("kTable", "tls_counter"):
                    kinds.setdefault(name, set()).add(entry["type"])
    data = sorted(summary["closure"]["nonlocal_data"])
    return f"PASS {sorted((n, sorted(k)) for n, k in kinds.items())} audited {data}"
check("Q2d-got-and-tls-references-resolve-to-audited-data", "PASS [('kTable', [311, 312]), ('tls_counter', [", got_and_tls)
def got_selected_from_nvcc(evidence):
    summary = run("nvfirst", evidence)
    _, selection, _ = link("nvfirst", evidence, summary)
    return "FAIL " + str(sorted(f.split(": ")[1] for f in selection["failures"] if "kTable" in f or "tls_counter" in f))
check("Q2d2-got-and-tls-audited-data-selected-from-nvcc-fails", "FAIL ['kTable', 'tls_counter']", got_selected_from_nvcc)
# N-2: an object without recorded deps (simulated: ninja_deps raises for it, as for cmake_device_link.o).
def no_deps(victim):
    def control(evidence):
        original = b1.ninja_deps
        def patched(build, output):
            if output.endswith(victim):
                raise b1.Failure("no recorded deps")
            return original(build, output)
        b1.ninja_deps = patched
        try:
            summary = run("gnufirst", evidence)
        finally:
            b1.ninja_deps = original
        entry = next(e for e in summary["no_deps_objects"] if e["object"].endswith(victim))
        return f"{entry['status']} {entry.get('reason', '')}"
    return control
check("N2a-no-deps-object-without-audited-names-passes", "PASS", no_deps("plain.cpp.o"))
check("N2b-no-deps-object-defining-an-audited-name-fails", "FAIL defines audited names ['_Z11helper_copyi', 'kTable', 'tls_counter']", no_deps("k.cu.o"))
print(json.dumps({"controls": len(results), "ok": sum(results)}))
sys.exit(0 if all(results) else 1)
