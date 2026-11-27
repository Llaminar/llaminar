#!/usr/bin/env python3
"""Controls of b1_replay.py v5 on a small git-committed CMake/Ninja/ccache project (tool controls only, never
production evidence). Mounted: /src/mini (sources + manifest), /src/mini/build (Integration-like -O2 -g),
/src/mini/build-release (Release-like -O2, no -g), /src/mini/build-lto (-O2 -flto), all build dirs READ-ONLY; the
tools at /tools; a fresh evidence root /evidence; a bind mount at /usr/local/ctl (C-4 control)."""
import importlib.util, json, os, shutil, sys, argparse
from pathlib import Path
sys.path.insert(0, "/tools")
spec = importlib.util.spec_from_file_location("b1", "/tools/b1_replay.py")
b1 = importlib.util.module_from_spec(spec); sys.modules["b1"] = b1; spec.loader.exec_module(b1)
import elf_tools  # noqa: E402
ROOT, M = Path("/evidence"), "/src/mini"
BUILD, RELEASE, LTO = f"{M}/build", f"{M}/build-release", f"{M}/build-lto"
OBJ = lambda name: f"CMakeFiles/prod.dir/{name}.cpp.o"
args = argparse.Namespace(manifest=f"{M}/manifest.txt", audited_headers=f"{M}/audited.txt", src_root=M)
CONTEXT = b1.load_context(args)
results = []

def check(name, expect, function):
    evidence = ROOT / name; evidence.mkdir()
    try:
        outcome = function(evidence)
    except b1.Failure as error:
        outcome = f"FAIL: {error}"
    ok = outcome.startswith(expect)
    results.append(ok)
    print(f"{'ok ' if ok else 'BAD'} {name}: expected {expect}; got {outcome[:240]}", flush=True)

def tu(build, obj, evidence, context=CONTEXT, **kw):
    return b1.replay_tu(build, obj, evidence, context, **kw)

def classes(build, obj, evidence, mode):
    record = tu(build, obj, evidence)
    assert record["mode"] == mode, record["mode"]
    return f"{record['status']} {'+'.join(record['classification']['classes'])}"

# Selection superset (B1-b): every target input with an audited header in its deps; b.cpp (no audited header) is out.
def selection(evidence):
    inputs, selected, missing, unrecorded = b1.selected_objects(BUILD, ["libprod.so"], CONTEXT["audited_paths"])
    assert not missing and not unrecorded, (missing, unrecorded)
    return "PASS " + ",".join(sorted(Path(o).stem.split(".")[0] for o in selected))
check("S1-selection-superset", "PASS a,c,d,e,p", selection)
check("S2-program", "PASS program", lambda e: classes(BUILD, OBJ("a"), e, "integration-native-debug"))
check("S3-fully-inlined", "PASS inlined", lambda e: classes(BUILD, OBJ("c"), e, "integration-native-debug"))
check("S4-none", "PASS none", lambda e: classes(BUILD, OBJ("d"), e, "integration-native-debug"))
check("S5-helper-copy", "PASS helper", lambda e: classes(BUILD, OBJ("e"), e, "integration-native-debug"))
def helper_binding(evidence):
    record = tu(BUILD, OBJ("e"), evidence)
    return f"PASS {sorted(record['helper_bindings'].values())}"
check("S6-helper-weak", "PASS ['weak']", helper_binding)
def closure(evidence):
    summary = b1.run_b1(BUILD, ["libprod.so"], evidence, CONTEXT, 4)
    return f"PASS {sorted(name for f in summary['closure']['functions'].values() for name in f['names'] if 'rowsProgramILi1' in name or 'scaled' in name)}"
check("S7-call-closure", "PASS ['_ZN9llaminar23cpu16native_quantised11multi_scale11rowsProgramILi1EEEii', '_ZN9llaminar23cpu16native_quantised11multi_scale6scaledEi']", closure)

# Release debug augmentation (B1-a).
check("A1-release-augmented-pass", "PASS inlined", lambda e: classes(RELEASE, OBJ("c"), e, "release-debug-augmented"))
check("A2-release-program", "PASS program", lambda e: classes(RELEASE, OBJ("a"), e, "release-debug-augmented"))
def codegen_flag(evidence):
    # Prove a candidate flag changes the non-debug code (a witness), then require the -g run with it to FAIL.
    for flag in ("-fno-schedule-insns2", "-fno-tree-vectorize", "-fno-inline", "-O1"):
        probe = evidence / flag.strip("-")
        probe.mkdir()
        try:
            tu(RELEASE, OBJ("c"), probe, cc1_extra_augment=(flag,))
            continue  # No change: not a witness.
        except b1.Failure as error:
            if "outside debug sections" not in str(error):
                raise
        final = evidence / ("g-" + flag.strip("-")); final.mkdir()
        try:
            tu(RELEASE, OBJ("c"), final, cc1_extra_augment=("-g", flag))
        except b1.Failure as error:
            return f"FAIL ({flag} witnessed as codegen-changing): {error}"
        return f"PASS with codegen-changing {flag} (comparison too weak)"
    return "NO WITNESS: no candidate flag changed the code"
check("A3-codegen-changing-flag", "FAIL (", codegen_flag)

# Snapshot identity (B1-c).
def manifest_variant(evidence, edit):
    lines = Path(args.manifest).read_text().splitlines()
    path = evidence / "manifest.txt"
    path.write_text("\n".join(edit(lines)) + "\n")
    return b1.load_context(argparse.Namespace(manifest=str(path), audited_headers=args.audited_headers, src_root=M))
def blob_changed(evidence):
    context = manifest_variant(evidence, lambda lines: [l.replace(l.split()[2], "0" * 40) if l.endswith("\tc.cpp") else l for l in lines])
    return tu(BUILD, OBJ("c"), evidence, context)["status"]
check("G1-source-not-commit-blob", "FAIL: closure file c.cpp differs from the snapshot commit's git blob", blob_changed)
def not_committed(evidence):
    context = manifest_variant(evidence, lambda lines: [l for l in lines if not l.endswith("\tc.cpp")])
    return tu(BUILD, OBJ("c"), evidence, context)["status"]
check("G2-source-not-in-commit", "FAIL: closure file c.cpp is not in the snapshot commit", not_committed)
def generated_and_toolchain(evidence):
    record = tu(BUILD, OBJ("a"), evidence)
    kinds = sorted({entry["kind"] for entry in record["closure"].values()})
    return f"PASS {kinds}"
check("G3-closure-kinds", "PASS ['source', 'system']", generated_and_toolchain)

# A configure-fetched git checkout under <src>/external (as oneDNN): tracked files must equal its committed blobs.
def external_ok(evidence):
    record = tu(BUILD, OBJ("f"), evidence)
    kinds = {Path(p).name: e["kind"] for p, e in record["closure"].items() if "/external/" in p}
    hashed = all(len(e["sha256"]) == 64 for p, e in record["closure"].items() if "/external/" in p)
    return f"PASS {sorted(kinds.items())}" if hashed else "FAIL: an external closure file is not hashed"
check("X1-external-checkout", "PASS [('dep.h', 'external-source'), ('gen.h', 'external-generated')]", external_ok)
def external_tampered(evidence):
    context = dict(CONTEXT, repositories={root: dict(info, blobs={k: "0" * 40 for k in info["blobs"]})
                                          for root, info in CONTEXT["repositories"].items()})
    return tu(BUILD, OBJ("f"), evidence, context)["status"]
check("X2-external-blob-mismatch", "FAIL: closure file /src/mini/external/dep/dep.h differs from its fetched checkout", external_tampered)

# C-1: the reference graph fails closed (objects compiled here; tool controls).
def graph_of(evidence, source, flags=("-O2",)):
    path = evidence / "x.cpp"; path.write_text(source); obj = evidence / "x.o"
    result = b1.run(["g++-14", *flags, "-c", str(path), "-o", str(obj)], str(evidence))
    if result.returncode:
        raise b1.Failure(f"control compile failed: {result.stderr[-400:]}")
    return elf_tools.function_graph(elf_tools.read_elf(obj))
def graph_failures(source):
    def control(evidence):
        failures = graph_of(evidence, source)["failures"]
        return f"FAIL {failures[0]}" if failures else "PASS (no failure: the graph is not fail-closed)"
    return control
TARGET = ('.section .text.tgt,"ax",%progbits\n.type tgt_fn,%function\ntgt_fn:\n\tnop\n\tnop\n\tret\n'
          '.size tgt_fn,.-tgt_fn\n.text\n.globl caller_fn\n.type caller_fn,%function\ncaller_fn:\n')
def asm_source(text):
    return "asm(" + json.dumps(text) + ");\n"
check("K1-uncovered-relocation-fails", "FAIL .text+",
      graph_failures("int plain(int);\nint f(int x) { return x + 1; }\n" + asm_source(".text\n\tbl plain\n")))
check("K2-call-not-at-function-start-fails", "FAIL ",
      graph_failures(asm_source(TARGET + "\tbl tgt_fn+4\n\tret\n.size caller_fn,.-caller_fn\n")))
def k2_reason(evidence):
    failures = graph_of(evidence, asm_source(TARGET + "\tbl tgt_fn+4\n\tret\n.size caller_fn,.-caller_fn\n"))["failures"]
    return "FAIL no function start" if failures and "is no function start" in failures[0] else f"PASS {failures}"
check("K2b-reason", "FAIL no function start", k2_reason)
def k3(evidence):
    failures = graph_of(evidence, asm_source(TARGET + "\tb tgt_fn+4\n\tret\n.size caller_fn,.-caller_fn\n"))["failures"]
    return "FAIL interior" if failures and "interior of tgt_fn" in failures[0] else f"PASS {failures}"
check("K3-branch-into-unrelated-interior-fails", "FAIL interior", k3)
def k4_cold(evidence):
    # GCC 14 on AArch64 partitions only with -freorder-blocks-and-partition (not default at -O2).
    graph = graph_of(evidence, '#include <stdexcept>\n#include <cstdio>\n__attribute__((noinline)) int hot(int x) {\n'
                     '  if (__builtin_expect(x == 12345, 0)) { std::printf("%d", x); throw std::runtime_error("x"); }\n'
                     '  return x * 3; }\n', ("-O2", "-freorder-blocks-and-partition"))
    cold = [key for key in graph["functions"] if key.endswith(".cold")]
    if not cold:
        return "NO WITNESS: no cold partition emitted"
    reached = any(target == cold[0] for edge in graph["edges"].get("_Z3hoti", []) for target in edge[2:])
    return f"PASS cold partition {cold[0]} reached={reached}" if not graph["failures"] else f"FAIL {graph['failures'][:2]}"
check("K4-cold-partition-attributed", "PASS cold partition _Z3hoti.cold reached=True", k4_cold)

# C-1/C-2: the global closure on the mini project (p.cpp: an OpenMP outlined body, a function pointer in code, a
# function-pointer table in data, a cross-TU call into b.cpp, an unaudited project header and a std:: header).
P2 = "_ZN9llaminar23cpu16native_quantised11multi_scale11rowsProgramILi2EEEii"
GLOBAL = {}
def global_run(evidence, audited):
    if audited not in GLOBAL:
        context = b1.load_context(argparse.Namespace(manifest=f"{M}/manifest.txt", audited_headers=f"{M}/{audited}", src_root=M))
        GLOBAL[audited] = b1.run_b1(BUILD, ["libprod.so"], evidence, context, 4)
    return GLOBAL[audited]
def k5_omp(evidence):
    summary = global_run(evidence, "audited.txt")
    record = next(t for t in summary["tus"] if t["object"] == OBJ("p"))
    graph = json.loads((evidence / record["graph"]).read_text())["graph"]
    omp = [edge for edge in graph["edges"][P2] if edge[1] == "node" and edge[2].endswith("._omp_fn.0")]
    included = any(key.endswith("._omp_fn.0") for key in summary["closure"]["functions"])
    return f"PASS edge {omp[0][0]} included {included}" if omp else "FAIL: no edge to the outlined body"
check("K5-omp-outlined-body-edge", "PASS edge R", k5_omp)
def k6_pointers(evidence):
    summary = global_run(ROOT / "K5-omp-outlined-body-edge", "audited.txt")
    names = {name.split("multi_scale")[-1] for f in summary["closure"]["functions"].values() for name in f["names"]}
    wanted = ["11via_pointerEi", "11via_table_aEi", "11via_table_bEi"]
    return f"PASS {wanted}" if all(w in names for w in wanted) else f"FAIL missing {[w for w in wanted if w not in names]}"
check("K6-function-pointers-in-code-and-data", "PASS", k6_pointers)
def k7_indirect(evidence):
    summary = global_run(ROOT / "K5-omp-outlined-body-edge", "audited.txt")
    listed = [label for label in summary["closure"]["indirect_branches"] if label.endswith(P2)]
    return f"PASS {summary['closure']['indirect_branches'][listed[0]]}" if listed else "FAIL: blr not listed"
check("K7-indirect-calls-listed", "PASS {'blr'", k7_indirect)
def k8_cross_tu(evidence):
    summary = global_run(ROOT / "K5-omp-outlined-body-edge", "audited.txt")
    return f"PASS {[Path(o).name for o in summary['selected_by_closure']]}"
check("K8-closure-selects-cross-tu-definition", "PASS ['b.cpp.o']", k8_cross_tu)
def m1(evidence):
    summary = global_run(ROOT / "K5-omp-outlined-body-edge", "audited.txt")
    closure = summary["closure"]
    system = sorted(Path(p).name for p in closure["loc_files"]["system"])
    return f"{closure['status']} {closure['unaudited_project_files']} system has stl_algobase.h {'stl_algobase.h' in system}"
check("M1-unlisted-project-files-fail", "FAIL ['a.cpp', 'b.cpp', 'h.h', 'p.cpp', 'ptr.h'] system has stl_algobase.h True", m1)
def m2(evidence):
    summary = global_run(evidence, "audited-all.txt")
    return f"{summary['closure']['status']} {summary['status']} unaudited {summary['closure']['unaudited_project_files']}"
check("M2-complete-list-passes", "PASS PASS unaudited []", m2)

# C-3: debug mode and the exact debug exclusion.
def d1(evidence):
    cases = {"none": ["g++", "-O2"], "g": ["g++", "-O2", "-g"], "gdwarf": ["g++", "-gdwarf-5"], "g0": ["g++", "-g0"],
             "g-g0": ["g++", "-g", "-g0"], "gno": ["g++", "-g", "-gno-column-info"], "gz": ["g++", "-g", "-gz"]}
    found = {}
    for name, argv in cases.items():
        try:
            found[name] = b1.debug_mode(argv)
        except b1.Failure:
            found[name] = "FAIL"
    return "PASS " + " ".join(f"{k}={v}" for k, v in found.items())
check("D1-debug-mode", "PASS none=augment g=native gdwarf=native g0=FAIL g-g0=FAIL gno=FAIL gz=FAIL", d1)
def objcopy_variant(evidence, name, *arguments):
    blob = evidence / "blob"; blob.write_bytes(b"variant\0")
    out = evidence / f"{name}.o"
    result = b1.run(["objcopy", *[a.replace("BLOB", str(blob)) for a in arguments], str(Path(RELEASE) / OBJ("c")), str(out)], str(evidence))
    if result.returncode:
        raise b1.Failure(f"objcopy failed: {result.stderr[-300:]}")
    differences = elf_tools.compare_nondebug(Path(RELEASE) / OBJ("c"), out)
    return f"FAIL {differences}" if differences else "PASS"
check("D2a-added-debug-section-excluded", "PASS",
      lambda e: objcopy_variant(e, "debug", "--add-section", ".debug_extra=BLOB"))
check("D2b-comment-difference-fails", "FAIL ['sections .comment#0 differs']",
      lambda e: objcopy_variant(e, "comment", "--update-section", ".comment=BLOB"))
check("D2c-note-section-fails", "FAIL ['sections .note.extra#0 only in candidate']",
      lambda e: objcopy_variant(e, "note", "--add-section", ".note.extra=BLOB"))
check("D2d-zdebug-not-excluded", "FAIL ['sections .zdebug_extra#0 only in candidate']",
      lambda e: objcopy_variant(e, "zdebug", "--add-section", ".zdebug_extra=BLOB"))
check("D2e-debuglto-not-excluded", "FAIL ['sections .gnu.debuglto_.debug_info#0 only in candidate']",
      lambda e: objcopy_variant(e, "debuglto", "--add-section", ".gnu.debuglto_.debug_info=BLOB"))
def d3(evidence):
    release = tu(RELEASE, OBJ("c"), evidence / "r" if (evidence / "r").mkdir() is None else None)
    native = tu(BUILD, OBJ("c"), evidence / "n" if (evidence / "n").mkdir() is None else None)
    return (f"PASS {release['analysed_assembly']['debug']['mode']} {release['analysed_assembly']['debug']['cc1plus_extra']} "
            f"{native['analysed_assembly']['debug']['mode']} {native['analysed_assembly']['debug']['executed_options']}")
check("D3-analysed-identity-records-debug", "PASS release-debug-augmented ['-g'] integration-native-debug ['-g']", d3)

# C-5: assembler-resolved (relocation-free) references, with the production compile flags (no -ffunction-sections).
PRODUCTION_FLAGS = ("-mcpu=generic", "-march=armv8-a+simd", "-O3", "-g", "-fPIC", "-fopenmp")
def compiled(evidence, source, flags=PRODUCTION_FLAGS, suffix="cpp"):
    path = evidence / f"x.{suffix}"; path.write_text(source)
    assembly, obj = evidence / "x.s", evidence / "x.o"
    if suffix == "cpp":
        result = b1.run(["g++-14", *flags, "-S", str(path), "-o", str(assembly)], str(evidence))
        if result.returncode:
            raise b1.Failure(f"control compile failed: {result.stderr[-400:]}")
    else:
        assembly = path
    result = b1.run(["g++-14", "-c", str(assembly), "-o", str(obj)], str(evidence))
    if result.returncode:
        raise b1.Failure(f"control assemble failed: {result.stderr[-400:]}")
    elf = elf_tools.read_elf(obj)
    return elf, elf_tools.function_graph(elf), assembly
def relocated_offsets(elf):
    return {(section["info"], entry["offset"]) for section in elf["sections"] if section["type"] == elf_tools.SHT_RELA
            for entry in elf["relocations"][section["index"]]}
LOCAL_CALLS = """struct Big { int a[64]; int k; };
static __attribute__((noinline)) int helper(int x) { return x * 3 + 1; }
static __attribute__((noinline)) int consume(const Big &b, int x) { return b.k * x + 3; }
namespace llaminar2::cpu::native_quantised::multi_scale {
int rowsProgram(int x) { return helper(x) + 2; }
int rowsTail(const Big &b, int x) { return consume(b, x + 1); }
}
"""
PROGRAM = "_ZN9llaminar23cpu16native_quantised11multi_scale11rowsProgramEi"
TAIL = "_ZN9llaminar23cpu16native_quantised11multi_scale8rowsTailERK3Bigi"
def c5a(evidence):
    elf, graph, _ = compiled(evidence, LOCAL_CALLS)
    edges = graph["edges"].get(PROGRAM, [])
    decoded = [edge for edge in edges if edge[0] == "decoded-bl" and edge[2] == "_ZL6helperi"]
    relocations = [e for e in edges if e[2] == "_ZL6helperi" and not e[0].startswith("decoded")]
    if graph["failures"]:
        return f"FAIL {graph['failures'][:2]}"
    return f"PASS decoded {len(decoded)} relocation-edges {len(relocations)}"
check("C5a-static-helper-same-text-no-relocation", "PASS decoded 1 relocation-edges 0", c5a)
def c5b(evidence):
    elf, graph, _ = compiled(evidence, LOCAL_CALLS)
    isra = [key for key in graph["functions"] if ".isra." in key]
    if not isra:
        return "NO WITNESS: no .isra clone"
    decoded = [edge[0] for edge in graph["edges"].get(TAIL, []) if edge[2] == isra[0]]
    return f"PASS {isra[0]} via {decoded}"
check("C5b-isra-tail-call-b", "PASS _ZL7consumeRK3Bigi.isra.0 via ['decoded-b']", c5b)
LOCAL_ASM = ('.text\n.type tgt,%function\ntgt:\n\tnop\n\tnop\n\tret\n.size tgt,.-tgt\n'
             '.globl caller\n.type caller,%function\ncaller:\n\t{insn}\n\tret\n.size caller,.-caller\n')
def asm_case(instruction, expect_text):
    def control(evidence):
        elf, graph, _ = compiled(evidence, LOCAL_ASM.format(insn=instruction), suffix="s")
        if relocated_offsets(elf):
            return f"NO WITNESS: the assembler emitted a relocation {sorted(relocated_offsets(elf))}"
        hits = [f for f in graph["failures"] if expect_text in f]
        if hits:
            return f"FAIL {hits[0]}"
        return f"PASS edges {graph['edges'].get('caller', [])} failures {graph['failures'][:1]}"
    return control
check("C5c-bl-into-local-interior-fails", "FAIL caller+0x0", asm_case("bl tgt+4", "into the interior of tgt"))
check("C5c2-b-into-local-interior-fails", "FAIL caller+0x0", asm_case("b tgt+4", "into the interior of tgt"))
check("C5c3-cbz-into-local-interior-fails", "FAIL caller+0x0", asm_case("cbz x0, tgt+4", "into the interior of tgt"))
check("C5e-adr-local-start-is-edge", "PASS edges [['decoded-adr', 'node', 'tgt']]", asm_case("adr x0, tgt", "@@none@@"))
check("C5e2-adr-local-interior-fails", "FAIL caller+0x0", asm_case("adr x0, tgt+8", "into the interior of tgt"))
check("C5e3-ldr-literal-interior-fails", "FAIL caller+0x0", asm_case("ldr x0, tgt+4", "into the interior of tgt"))
def c5f(evidence):
    elf, graph, _ = compiled(evidence, ".text\n\tmov x0, 1\n\tret\n" + LOCAL_ASM.format(insn="nop"), suffix="s")
    hits = [f for f in graph["failures"] if "outside every function symbol" in f]
    return f"FAIL {hits[0]}" if hits else f"PASS {graph['failures']}"
check("C5f-code-outside-functions-fails", "FAIL 2 code word(s) outside every function symbol", c5f)
def c5g(evidence):
    source = ".text\n.globl plain\n" + LOCAL_ASM.format(insn=".reloc ., R_AARCH64_CALL26, plain\n\tnop")
    elf, graph, _ = compiled(evidence, source, suffix="s")
    hits = [f for f in graph["failures"] if "not PC-relative" in f]
    return f"FAIL {hits[0]}" if hits else f"PASS {graph['failures']}"
check("C5g-pc-relocation-on-non-pc-instruction-fails", "FAIL caller+0x0", c5g)
def c5g2(evidence):
    source = ".text\n.globl plain\n" + LOCAL_ASM.format(insn=".reloc ., R_AARCH64_CALL26, plain\n\tb plain")
    elf, graph, _ = compiled(evidence, source, suffix="s")
    hits = [f for f in graph["failures"] if "carries relocation" in f]
    return f"FAIL {hits[0]}" if hits else f"PASS {graph['failures']}"
check("C5g2-relocation-class-mismatch-fails", "FAIL caller+0x0", c5g2)
# C-5d: the .s cross-check passes, and FAILS when one decoded edge is suppressed (a decode gap).
def c5d(evidence):
    elf, graph, assembly = compiled(evidence, LOCAL_CALLS)
    passed = b1.assembly_crosscheck(assembly, graph)
    original = elf_tools._decode
    elf_tools._decode = lambda word: None if word & 0xFC000000 == 0x94000000 else original(word)
    try:
        suppressed = elf_tools.function_graph(elf)
    finally:
        elf_tools._decode = original
    failed = b1.assembly_crosscheck(assembly, suppressed)
    return (f"PASS checked {passed['checked']} failures {len(passed['failures'])}; suppressed-bl: "
            f"{'FAIL ' + failed['failures'][0] if failed['failures'] else 'PASS (gap not caught)'}")
check("C5d-assembly-crosscheck", "PASS checked 2 failures 0; suppressed-bl: FAIL _ZN9llaminar23cpu16native_quantised11multi_scale11rowsProgramEi line", c5d)
def c5d_mini(evidence):
    summary = global_run(ROOT / "K5-omp-outlined-body-edge", "audited.txt")
    checks = [t["assembly_crosscheck"] for t in summary["tus"] if t["status"] == "PASS"]
    return f"PASS TUs {len(checks)} checked {sum(c['checked'] for c in checks)} failures {sum(len(c['failures']) for c in checks)}"
check("C5d2-crosscheck-on-mini-project", "PASS TUs 6 checked", c5d_mini)
# C-5d with GCC's assembler aliases: `.set .LANCHOR0,. + N` defines a label here; `.set .LC1,.LC0` aliases one.
SET_ASM = ('.data\n.set .LANCHOR0,. + 8\n\t.8byte 1\n\t.8byte 2\n.section .rodata\n.LC0:\n\t.8byte 3\n'
           '.set .LC1,.LC0\n.text\n.globl caller\n.type caller,%function\ncaller:\n'
           '\tadrp x0, .LANCHOR0\n\tadd x0, x0, :lo12:.LANCHOR0\n\tadrp x1, .LC1\n\tldr x1, [x1, :lo12:.LC1]\n'
           '\t{extra}\n\tret\n.size caller,.-caller\n')
def c5d_set(evidence):
    elf, graph, assembly = compiled(evidence, SET_ASM.format(extra="nop"), suffix="s")
    result = b1.assembly_crosscheck(assembly, graph)
    return f"PASS checked {result['checked']} failures {result['failures']}" if not result["failures"] else \
        f"FAIL {result['failures']}"
check("C5d3-set-labels-and-aliases", "PASS checked 4 failures []", c5d_set)
def c5d_set_unbound(evidence):
    elf, graph, _ = compiled(evidence, SET_ASM.format(extra="nop"), suffix="s")
    source = SET_ASM.format(extra="nop").replace(".set .LC1,.LC0", ".set .LC1,.Lnowhere")
    path = evidence / "unbound.s"; path.write_text(source)
    result = b1.assembly_crosscheck(path, graph)
    return f"FAIL {result['failures'][0]}" if result["failures"] else "PASS (unbound alias accepted)"
check("C5d4-unbound-alias-fails", "FAIL caller line", c5d_set_unbound)

# Actual production case (read-only develop r2 NEON Integration object, when mounted): the compact programs are
# local .isra clones; the graph is fail-closed and the decoded edges reach their local callees.
PROD = Path("/prod/CMakeFiles/llaminar2_core.dir/execution/compute_stages/stages/MoEExpertComputeStage.cpp.o")
def c5p(evidence):
    if not PROD.is_file():
        return "NO PRODUCTION OBJECT MOUNTED"
    elf = elf_tools.read_elf(PROD)
    graph = elf_tools.function_graph(elf)
    programs = [k for k in graph["functions"] if b1.PROGRAMS.search(k)]
    local_programs = [k for k in programs if all(b == "local" for b in graph["functions"][k]["names"].values())]
    decoded_from_programs = sorted({edge[2] for k in programs for edge in graph["edges"].get(k, [])
                                    if edge[0].startswith("decoded")})
    (evidence / "production-graph-summary.json").write_text(json.dumps(
        {"programs": len(programs), "local_programs": len(local_programs), "decoded_edges": graph["decoded_edges"],
         "decoded_targets_from_programs": decoded_from_programs, "failures": graph["failures"][:50]}, indent=1))
    return (f"{'FAIL' if graph['failures'] else 'PASS'} programs {len(programs)} local {len(local_programs)} "
            f"decoded-edges {graph['decoded_edges']} decoded-targets-from-programs {len(decoded_from_programs)} "
            f"failures {len(graph['failures'])} {graph['failures'][:2]}")
check("C5p-production-object", "PASS programs", c5p)

# C-4: closure files outside /src, the build, /src/external and the image's system paths are unclassified.
def classify_file(path):
    def control(evidence):
        return b1.closure_entry(path, BUILD, M, CONTEXT["manifest"], CONTEXT["repositories"])["kind"]
    return control
check("U1-bind-mounted-file-fails", "FAIL: closure file /tools/b1_replay.py is outside", classify_file("/tools/b1_replay.py"))
check("U2-mounted-usr-path-fails", "FAIL: closure file /usr/local/ctl/mounted.h is outside", classify_file("/usr/local/ctl/mounted.h"))
check("U3-image-system-header", "system", classify_file("/usr/include/stdio.h"))
check("U4-src-file-not-in-commit-fails", "FAIL: closure file manifest.txt is not in the snapshot commit", classify_file(f"{M}/manifest.txt"))

# LTO is fatal (B1-d).
check("L1-lto-fatal", "FAIL: LTO in the executed compile", lambda e: tu(LTO, "CMakeFiles/prod.dir/a.cpp.o", e)["status"])

# Earlier controls (v1), unchanged semantics.
def tampered(evidence):
    t = evidence / "tampered.o"; data = bytearray((Path(BUILD) / OBJ("a")).read_bytes()); data[-1] ^= 1; t.write_bytes(bytes(data))
    return tu(BUILD, OBJ("a"), evidence, build_object=t)["status"]
check("C2-tampered-object", "FAIL: the launcher route does not reproduce", tampered)
def source_changed(evidence):
    header = Path(f"{M}/h.h"); original = header.read_text(); header.write_text(original + "\nint changed_after_build = 1;\n")
    try:
        return tu(BUILD, OBJ("a"), evidence)["status"]
    finally:
        header.write_text(original)
check("C3-source-changed", "FAIL: the launcher route does not reproduce", source_changed)
check("C4-missing-object", "MISSING", lambda e: tu(BUILD, "CMakeFiles/prod.dir/none.o", e)["status"])
def plan(argument):
    return lambda e: (b1.gnu_plan(["/usr/bin/g++-14", argument, "-c", f"{M}/a.cpp", "-o", str(e / "x.o")], BUILD), "PASS")[1]
check("C5a-pipe", "FAIL: the compiler job output is not a driver temporary", plan("-pipe"))
check("C5b-save-temps", "FAIL: the driver plan is not exactly cc1plus then as", plan("-save-temps"))
def assembly_altered(evidence):
    original = b1.run
    def altering(argv, cwd, env=None):
        result = original(argv, cwd, env)
        if Path(argv[0]).name == "cc1plus":
            target = Path(argv[argv.index("-o") + 1]); target.write_text(target.read_text() + "\n\t.text\n\tnop\n")
        return result
    b1.run = altering
    try:
        return tu(BUILD, OBJ("a"), evidence)["status"]
    finally:
        b1.run = original
check("C6-assembly-altered", "FAIL: the replayed plan does not reproduce", assembly_altered)
def closure_mismatch(evidence):
    original = b1.ninja_deps
    b1.ninja_deps = lambda build, output: original(build, output)[1:]
    try:
        return tu(BUILD, OBJ("a"), evidence)["status"]
    finally:
        b1.ninja_deps = original
check("C7-closure-mismatch", "FAIL: the replay's dependency closure differs", closure_mismatch)
def no_log(evidence):
    os.environ["CCACHE_DISABLE"] = "1"
    try:
        return tu(BUILD, OBJ("a"), evidence)["status"]
    finally:
        del os.environ["CCACHE_DISABLE"]
check("C8-no-launcher-log", "FAIL: expected one ccache debug log", no_log)
def read_only(evidence):
    try:
        (Path(BUILD) / "probe-write").write_text("x")
    except OSError:
        return "PASS"
    return "FAIL: the build directory is writable"
check("C9-build-read-only", "PASS", read_only)
print(json.dumps({"controls": len(results), "ok": sum(results)}))
sys.exit(0 if all(results) else 1)
