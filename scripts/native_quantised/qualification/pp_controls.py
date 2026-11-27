#!/usr/bin/env python3
"""Controls of postprocess.py v3 (tool controls only)."""
import argparse, importlib.util, json, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, "/tools")
def load(name):
    spec = importlib.util.spec_from_file_location(name, f"/tools/{name}.py"); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
pp, b1 = load("postprocess"), load("b1_replay")
results = []
def check(name, expect, function):
    try:
        outcome = function()
    except SystemExit as error:
        outcome = f"FAIL: {error}"
    except b1.Failure as error:
        outcome = f"FAIL: {error}"
    ok = str(outcome).startswith(expect); results.append(ok)
    print(f"{'ok ' if ok else 'BAD'} {name}: expected {expect}; got {str(outcome)[:240]}", flush=True)

def b1_summary(build, src, targets, evidence):
    args = argparse.Namespace(manifest=f"{src}/manifest.txt", audited_headers=f"{src}/audited.txt", src_root=src)
    context = b1.load_context(args)
    Path(evidence).mkdir(parents=True)
    return b1.run_b1(build, targets, Path(evidence), context, 4)

MINI_B1 = b1_summary("/src/mini/build", "/src/mini", ["libprod.so"], "/pp/b1-mini")
PROC_B1 = b1_summary("/src/c/proc/build-proc", "/src/c/proc", ["llaminar2"], "/pp/b1-proc")

def links(build, targets, name, summary=MINI_B1):
    artifacts = Path(f"/pp/{name}"); artifacts.mkdir()
    records = {t: pp.link_record(build, t, artifacts) for t in targets}
    symbols, passed = pp.audited_symbols(summary)
    copies = pp.selected_copies(build, artifacts, symbols, passed) if all(r["status"] == "PASS" for r in records.values()) else None
    return records, copies, artifacts

def p1():
    records, copies, _ = links("/src/mini/build", ["libprod.so", "app"], "P1")
    return f"PASS {sorted(r['status'] for r in records.values())} copies {copies['status']} {sorted(copies['symbols'])}"
pp.TARGETS = ("libprod.so", "app")
check("P1-links-and-selected-copy", "PASS ['PASS', 'PASS'] copies PASS", p1)
check("P2-missing-target", "MISSING", lambda: links("/src/mini/build", ["absent.so"], "P2")[0]["absent.so"]["status"])
check("P3-lto-link-fatal", "FAIL: FAIL: LTO in the link", lambda: links("/src/mini/build-lto", ["libprod.so"], "P3"))
def p4():
    summary = dict(MINI_B1, tus=[t for t in MINI_B1["tus"] if not t["object"].endswith("e.cpp.o")]
                   + [dict(t, status="FAIL") for t in MINI_B1["tus"] if t["object"].endswith("e.cpp.o")])
    _, copies, _ = links("/src/mini/build", ["libprod.so"], "P4", summary)
    return f"{copies['status']} {copies['failures']}"
check("P4-copy-from-tu-without-b1", "FAIL ['_Z11helper_copyi']", p4)
check("P5-b1-without-closure-fails", "FAIL: FAIL: the B1 summary holds no reference closure",
      lambda: pp.audited_symbols({k: v for k, v in MINI_B1.items() if k != "closure"}))

pp.TARGETS = ("libllaminar2_core.so", "llaminar2")
SYMBOLS, _ = pp.audited_symbols(PROC_B1)
def resolution(build, name):
    artifacts = Path(f"/pp/{name}"); artifacts.mkdir()
    return pp.process_resolution(build, artifacts, SYMBOLS)
def r1():
    record = resolution("/src/c/proc/build-proc", "R1")
    helper = record["symbols"]["_Z11helper_copyi"]
    return f"{record['status']} helper {helper['status']} logs {len(record['process_logs'])}"
check("R1-observed-and-descendant-logs", "PASS helper observed logs 2", r1)
def r2():
    record = resolution("/src/c/proc/build-proc", "R2")
    unrelocated = sorted(n for n, e in record["symbols"].items() if e["status"].startswith("unobserved"))
    return f"{record['status']} unobserved {unrelocated}"
check("R2-unrelocated-is-unobserved", "PASS unobserved ['_ZN9llaminar23cpu16native_quantised11multi_scale11rowsProgramILi1EEEii'", r2)
def r3():
    original = pp.BINDING
    pp.BINDING = __import__("re").compile(r"(?!x)x(\S+)(\S+)(\S+)")
    try:
        record = resolution("/src/c/proc/build-proc", "R3")
    finally:
        pp.BINDING = original
    return f"{record['status']} {record['failures']}"
check("R3-relocated-unobserved-fails", "FAIL ['_Z11helper_copyi'", r3)
check("R4-nonzero-exit-fails", "FAILED (exit 1", lambda: resolution("/src/c/proc/build-proc-exit1", "R4")["status"])
def r5():
    original = pp.glob.glob
    pp.glob.glob = lambda pattern: [] if pattern.startswith("/tmp/lddebug") and pattern.endswith(".*") and r5.armed else original(pattern)
    r5.armed = False
    try:
        artifacts = Path("/pp/R5"); artifacts.mkdir()
        # Arm after the stale-log cleanup: only the post-run collection sees no log.
        real_run = pp.run
        def arming(*a, **k):
            result = real_run(*a, **k); r5.armed = True; return result
        pp.run = arming
        try:
            return pp.process_resolution("/src/c/proc/build-proc", artifacts, SYMBOLS)["status"]
        finally:
            pp.run = real_run
    finally:
        pp.glob.glob = original
check("R5-no-ld-debug-log-fails", "FAILED ( no LD_DEBUG log", r5)
print(json.dumps({"controls": len(results), "ok": sum(results)}))
sys.exit(0 if all(results) else 1)
