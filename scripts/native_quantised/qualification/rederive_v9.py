#!/usr/bin/env python3
"""Re-derive a retained B1 summary with the tools-v9 non-GNU rules (reviewer 109 Q-2d/G-1/G-2) WITHOUT any compile:
GNU TU records (replayed by the earlier, sealed tool version) are kept byte for byte; every non-GNU record is
recomputed from the read-only build object, its recorded deps and its DWARF (non_gnu_record); the closure fixpoint
is recomputed over the retained per-TU graphs; a closure that needs an object not replayed earlier FAILS (no
compile here). Output: a new evidence directory (graphs linked from the retained one) and b1-summary.json.
argv: <build> <retained b1 dir> <out dir> <manifest> <identity> <audited-headers> <src-root> [target...]"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_replay as b1  # noqa: E402

build, retained, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
args = argparse.Namespace(manifest=sys.argv[4], identity=sys.argv[5], audited_headers=sys.argv[6], src_root=sys.argv[7])
targets = sys.argv[8:] or ["llaminar2", "libllaminar2_core.so"]
out.mkdir(parents=True, exist_ok=False)
context = dict(b1.load_context(args), evidence=str(out))
old = json.loads((retained / "b1-summary.json").read_text())
inputs, objects, missing, unrecorded = b1.selected_objects(build, targets, context["audited_paths"])
definitions, roots = b1.symbol_index(build, [o for o in inputs if (Path(build) / o).is_file()])
records = []
for record in old["tus"]:
    if record.get("mode") == "non-gnu":
        command = b1.executed_command(build, record["object"])
        record = b1.non_gnu_record(build, record["object"], command, Path(build) / record["object"], context)
    else:
        graph = record.get("graph")
        if graph:
            (out / graph).symlink_to((retained / graph).resolve())
    records.append(record)
closure = b1.global_closure(records, out, definitions, roots, context, build)
if closure.get("needed"):
    closure = {"status": "FAIL", "failures": [f"needs objects not replayed by the retained run: {closure['needed'][:5]}"]}
no_deps, failing_no_deps = b1.finalize_non_gnu(records, closure, definitions, unrecorded, context, build)
failed = sum(r["status"] != "PASS" for r in records)
ok = records and not failed and not missing and not failing_no_deps and closure.get("status") == "PASS"
summary = dict(old, tus=records, closure=closure, no_deps_objects=no_deps, inputs_without_recorded_deps=failing_no_deps,
               passed=len(records) - failed, failed_or_missing=failed, status="PASS" if ok else "FAIL",
               closure_digest=__import__("hashlib").sha256(json.dumps(closure, sort_keys=True).encode()).hexdigest(),
               rederived={"from": str(retained), "tool": "tools-v9 rederive_v9.py", "compiles": 0})
(out / "b1-summary.json").write_text(json.dumps(summary, sort_keys=True, indent=1) + "\n")
for r in records:
    if r["status"] != "PASS":
        print(f"{r['status']}: {r['object']}: {r.get('reason', '')[:400]}")
for f in closure.get("failures", [])[:10]:
    print("CLOSURE FAIL:", f[:300])
print(f"B1 (rederived v9) {summary['status']}: {summary['passed']}/{len(records)}; closure {closure.get('status')}; "
      f"no-deps failing {failing_no_deps}")
