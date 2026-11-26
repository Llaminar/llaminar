#!/usr/bin/env python3
"""Stack-traffic records of the native NEON tile and compact programs AS LINKED INTO PRODUCTION (Gate7 tables).

Maintainer tool. It never changes witness semantics: it imports the committed witness checker
(CPUNativeTileRegisterEvidence.py) and the baseline generator's loop annotation (CPUNativeTileStackBaseline.py) and
runs the same per-program analysis (call model, stack-traffic classes per natural loop, loop naming by the .loc of
the retained assembler input, loop control evidence) on the PRODUCTION objects instead of a probe:

- Inputs are the production B1 evidence of one canonical build (b1_replay.py v4 summary: per TU the build
  object's digest, the retained analysed assembly that object was assembled from - byte-identical replay - and
  its digest) and the P2 post-processing summary (link maps, process resolution). Every digest is re-checked.
- Programs: every tile program (TILE_RUN) and every compact program (compact_programs()) of every B1 TU of class
  "program". A TU-local program is keyed <key>@<copy> (copy = the TU's object path; it serves that TU's call
  sites). A weak/global (COMDAT) program exists in the DSO only as the copy the executed link selected (GNU ld
  map): it is keyed <key>@link and recorded with its object and process binding; the discarded copies are
  listed, never analysed as executed code (P4: never another copy).
- Weak/global callees: the call model admits them only with a link-selection record. The records are built from
  P2 alone (never invented): the GNU ld map of the executed link names the selected input object; the process
  resolution under LD_BIND_NOW names the defining object (scope: the executable, then the core DSO), and the
  selected object must be a B1-PASS TU whose unit is loaded. Without that evidence no record exists and the call
  fails in the checker (reported per program).
- --per-copy (diagnostic, never qualified): every TU's compiled copy is keyed @<copy>, weak/COMDAT ones too, so
  two trees compare copy by copy before (or without) P2; --postprocess none is accepted only then.
- Output: one record per mode in the baseline record schema (structurally clean programs only) (CPUNativeTileStackTable.py consumes it), plus
  "structural" (per-key structural failures - a failing program is recorded, never dropped) and "qualified":
  true only when B1 is PASS and P2 is COMPLETE; an unqualified record is evidence for analysis, never a gate
  input.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import lzma
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ev = _load("cpu_native_tile_register_evidence", "CPUNativeTileRegisterEvidence.py")
baseline = _load("cpu_native_tile_stack_baseline", "CPUNativeTileStackBaseline.py")
envelope = ev.envelope


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_name(obj: str) -> str:
    """Copy identity (KEY grammar [A-Za-z0-9_.+-]): the object path below CMakeFiles/, '/' -> '+'."""
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", obj.replace("CMakeFiles/", "", 1).replace("/", "+"))


class Loaded:
    """One B1-verified production object: bytes, disassembly, analysed assembly, image, bodies, unit."""

    def __init__(self, tu: dict, build: Path, evidence: Path, disassembler: str):
        if tu.get("status") != "PASS":
            raise ValueError(f"{tu['object']}: B1 status {tu.get('status')}: not analysable")
        self.object = tu["object"]
        self.copy = copy_name(tu["object"])
        path = build / tu["object"]
        self.data = path.read_bytes()
        if hashlib.sha256(self.data).hexdigest() != tu["object_sha256"]:
            raise ValueError(f"{tu['object']}: object differs from its B1 digest")
        assembly = evidence / tu["analysed_assembly"]["file"]
        if sha256(assembly) != tu["analysed_assembly"]["sha256"]:
            raise ValueError(f"{tu['object']}: analysed assembly differs from its B1 digest")
        self.assembly_sha256 = tu["analysed_assembly"]["sha256"]
        self.assembly = assembly.read_text(errors="replace")
        self.disassembly, self.disassembly_command, self.disassembler_version = ev.disassemble(disassembler, path)
        self.image = ev.read_object(self.data)
        self.bodies = ev.normalize_disassembly(self.disassembly, self.image)
        self.unit = ev.make_unit(self.copy, self.image, self.bodies)


def map_selection(postprocess: dict, maps: Path) -> dict[str, str]:
    """Symbol -> the input object the executed link of the core DSO selected (GNU ld map, kept part)."""
    if not (maps / "libllaminar2_core.so.map.xz").is_file():
        return {}
    text = lzma.open(maps / "libllaminar2_core.so.map.xz", "rt", errors="replace").read()
    body = text.split("Linker script and memory map", 1)[-1]
    selected: dict[str, set] = {}
    current, pending = None, None
    section = re.compile(r"^ (\S+)(?:\s+0x[0-9a-f]+\s+0x[0-9a-f]+\s+(\S.*))?$")
    placement = re.compile(r"^\s+0x[0-9a-f]+\s+0x[0-9a-f]+\s+(\S.*)$")
    for line in body.splitlines():
        found = placement.match(line)
        if pending is not None and found:
            current = found.group(1).strip()
            if pending.startswith(".text."):
                selected.setdefault(pending[len(".text."):], set()).add(current)
            pending = None
            continue
        found = section.match(line)
        if found:
            if found.group(2):
                current, pending = found.group(2).strip(), None
                if found.group(1).startswith(".text."):
                    selected.setdefault(found.group(1)[len(".text."):], set()).add(current)
            else:
                pending = found.group(1)
            continue
        found = re.match(r"^\s+0x[0-9a-f]+\s+(\S.*)$", line)
        if found and current and not line.lstrip().startswith("0x0000000000000000 "):
            selected.setdefault(found.group(1).strip(), set()).add(current)
    return {name: objects.pop() for name, objects in selected.items() if len(objects) == 1}


def selection_records(postprocess: dict, maps: Path, loaded: dict[str, Loaded], needed: set[str]) -> tuple[dict, dict]:
    """(records, reasons): a process-binding record per needed weak/global callee with P2 evidence."""
    selected = map_selection(postprocess, maps)
    links = postprocess["links"]
    resolution = postprocess["process_resolution"].get("symbols", {})
    records, reasons = {}, {}
    by_object = {item.object: item for item in loaded.values()}
    for name in sorted(needed):
        obj = selected.get(name)
        if obj is None:
            reasons[name] = "no unique selected input object in the core DSO's link map"
            continue
        copy = by_object.get(obj)
        if copy is None:
            reasons[name] = f"the map-selected object {obj} is not a loaded B1-PASS unit"
            continue
        entry = resolution.get(name, {})
        bindings = entry.get("bindings", [])
        if entry.get("status") != "observed" or not bindings or \
                not all(binding.endswith("libllaminar2_core.so") for binding in bindings):
            reasons[name] = f"no observed process binding to the core DSO ({entry.get('status')})"
            continue
        records[name] = {"copy": copy.copy, "binding": "process", "resolution": {
            "scope": [{"object": "llaminar2", "role": "executable", "sha256": links["llaminar2"]["original_sha256"],
                       "defines": False},
                      {"object": "libllaminar2_core.so", "role": "dso",
                       "sha256": links["libllaminar2_core.so"]["original_sha256"], "defines": True}],
            "copy_object": "libllaminar2_core.so"}, "map_object": obj, "observed_bindings": bindings}
    return records, reasons


def name_loops(traffic: dict, item: Loaded, symbols: dict[str, str]) -> dict:
    """The checker's name_loops() for tile AND compact keys of one copy (same rule: instruction counts must
    agree, entries/latches take the .loc of the instruction at that index)."""
    locations: dict[str, list[str]] = {}
    functions, _ = ev.assembly_functions(item.assembly, set(item.bodies), item.image.macho, locations)
    named = {}
    for key, record in traffic.items():
        name = symbols[key]
        compiled = ev.object_instructions(item.bodies[name])
        lines, where = functions.get(name, []), locations.get(name, [])
        if len(compiled) != len(lines) or len(where) != len(lines):
            raise ValueError(f"{key}: compiler assembly ({len(lines)}) and object ({len(compiled)}) instruction counts "
                             f"differ; loops cannot be named")
        loops = []
        for loop in record["loops"]:
            loop = dict(loop)
            loop["entry_sources"] = [where[i] if 0 <= i < len(where) else "?" for i in loop["entries"]]
            loop["latch_sources"] = [where[i] if 0 <= i < len(where) else "?" for i in loop["latches"]]
            loops.append(loop)
        named[key] = {"loops": loops, "counts": record["counts"]}
    return named, locations


def generate(args: argparse.Namespace) -> dict:
    b1 = json.loads(Path(args.b1_summary).read_text())
    if args.postprocess == "none":
        if not args.per_copy:
            raise ValueError("without P2 post-processing only the per-copy diagnostic is possible (--per-copy)")
        postprocess = {"p2_status": "NONE (diagnostic: no P2)", "links": {}, "process_resolution": {}}
    else:
        postprocess = json.loads(Path(args.postprocess).read_text())
    build, evidence = Path(args.build), Path(args.b1_evidence)
    tus = {tu["object"]: tu for tu in b1["tus"]}
    program_tus = sorted(obj for obj, tu in tus.items() if "program" in tu.get("classification", {}).get("classes", []))
    loaded = {obj: Loaded(tus[obj], build, evidence, args.disassembler) for obj in program_tus}
    # Weak/global callees of the programs' units: their link-selected copies are loaded as units too.
    needed = {name for item in loaded.values() for name, binding in item.unit.functions.items()
              if binding in ("weak", "global")}
    selected = map_selection(postprocess, Path(args.maps))
    for name in sorted(needed):
        obj = selected.get(name)
        if obj and obj not in loaded and obj in tus and tus[obj].get("status") == "PASS":
            loaded[obj] = Loaded(tus[obj], build, evidence, args.disassembler)
    records, reasons = selection_records(postprocess, Path(args.maps), loaded, needed)
    units = {item.copy: item.unit for item in loaded.values()}
    traffic, spills, structural, members, copies = {}, {}, {}, {}, {}
    link_programs, discarded = {}, {}
    process = postprocess["process_resolution"].get("symbols", {})
    for obj in program_tus:
        item = loaded[obj]
        # M-1: the copy's instruction locations and its closure-bound source texts (each text verified against the
        # B1 closure hash first) admit a variable-size memcpy only at its exact committed source line.
        locations: dict[str, list[str]] = {}
        ev.assembly_functions(item.assembly, set(item.bodies), item.image.macho, locations)
        verified = {}
        for path, entry in tus[obj]["closure"].items():
            if entry["kind"] in ("source", "generated", "external-source", "external-generated"):
                if sha256(Path(path)) != entry["sha256"]:
                    raise ValueError(f"{obj}: closure file {path} differs from its B1 hash")
                verified[path] = entry
        ev.install_bounded_context(locations, ev.closure_sources(verified))
        model = ev.CallModel(units, records)
        noreturn = ev.noreturn_symbols(item.bodies, ev.GENERIC, ev.trusted_imports(item.image), item.image.prefix)
        symbols = {}
        found = [(ev.program_key(name), name) for name in sorted(item.bodies)
                 if ev.TILE_RUN.search(name) and ev.tile_match(name, item.image.prefix)]
        found += [(key, name) for key, (name, _) in sorted(ev.compact_programs(item.image).items())]
        for key, name in found:
            binding = item.unit.functions.get(name)
            if args.per_copy and binding in ("local", "weak", "global"):
                symbols[f"{key}@{item.copy}"] = name  # Diagnostic: every TU copy as compiled (not link-selected).
            elif binding == "local":
                symbols[f"{key}@{item.copy}"] = name  # A TU-local copy: it serves this TU's call sites.
            elif binding in ("weak", "global"):
                # A COMDAT/global copy: the link keeps exactly one; only the map-selected copy exists in the DSO.
                chosen = selected.get(name)
                if chosen == obj:
                    if f"{key}@link" in link_programs:
                        raise ValueError(f"{key}: two link-selected copies")
                    symbols[f"{key}@link"] = name
                    link_programs[f"{key}@link"] = {"symbol": name, "object": obj,
                                                    "process_binding": process.get(name, {}).get("status")}
                else:
                    discarded.setdefault(key, []).append({"object": obj, "selected": chosen})
            else:
                structural[f"{key}@{item.copy}"] = [f"program symbol binding {binding!r}"]
        copy_traffic = {}
        for key, name in sorted(symbols.items()):
            analysis = ev.analysed(name, item.bodies, ev.GENERIC, noreturn, model, item.unit)
            if analysis.errors:
                structural[key] = list(analysis.errors)
            copy_traffic[key] = ev.traffic_record(analysis)
            if key not in structural:
                spills[key] = analysis.spills
            members[key] = analysis
        # A structurally unprovable program has no table row: it is listed under "structural" (never dropped).
        copy_traffic = {key: record for key, record in copy_traffic.items() if key not in structural}
        named, locations = name_loops(copy_traffic, item, symbols)
        for key, record in named.items():
            where = locations.get(symbols[key], [])
            baseline.annotate_loops(record["loops"], baseline.loop_lines(members[key], where),
                                    baseline.loop_evidence(members[key], where))
            traffic[key] = record
        copies[item.copy] = {"object": obj, "object_sha256": tus[obj]["object_sha256"],
                             "assembly_sha256": item.assembly_sha256, "programs": len(symbols),
                             "disassembly_sha256": hashlib.sha256(item.disassembly.encode()).hexdigest()}
    first = next(iter(loaded.values()))
    tree_inputs = {}
    for obj in program_tus:
        for path, entry in tus[obj]["closure"].items():
            if path.startswith(args.tree_root.rstrip("/") + "/"):
                relative = path[len(args.tree_root.rstrip("/")) + 1:]
                if tree_inputs.setdefault(relative, entry["sha256"]) != entry["sha256"]:
                    raise ValueError(f"{relative}: two program TUs compiled different contents")

    def digest_of(values) -> str:
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    qualified = b1.get("status") == "PASS" and postprocess.get("p2_status") == "COMPLETE" and not args.per_copy
    return {
        "schema": 1, "kind": "production", "qualified": qualified,
        "qualification": {"b1_status": b1.get("status"), "p2_status": postprocess.get("p2_status")},
        "tree_sha": args.tree_sha, "tree_root": args.tree_root, "mode": args.mode, "image": args.image,
        "identity": args.identity, "compiler_version": args.identity,
        "checker_sha256": envelope.digest(HERE / "CPUNativeTileRegisterEvidence.py"),
        "generator_sha256": envelope.digest(Path(__file__)),
        "admission_sha256": envelope.digest(HERE / "CPUNativeTileRegisterEvidence.py"),
        "probe_sha256": digest_of({c: v["object_sha256"] for c, v in copies.items()}),
        "closure_sha256": digest_of({obj: tus[obj]["closure_sha256"] for obj in program_tus}),
        "object_sha256": digest_of({c: v["object_sha256"] for c, v in copies.items()}),
        "assembly_sha256": digest_of({c: v["assembly_sha256"] for c, v in copies.items()}),
        "target_command_sha256": digest_of({obj: tus[obj]["compiler_argv"] for obj in program_tus}),
        "disassembler_version": first.disassembler_version, "disassembly_command": first.disassembly_command[:-1],
        "b1_summary_sha256": sha256(Path(args.b1_summary)),
        "postprocess_sha256": None if args.postprocess == "none" else sha256(Path(args.postprocess)),
        "per_copy_diagnostic": bool(args.per_copy),
        "copies": copies, "tree_inputs": dict(sorted(tree_inputs.items())),
        "selection_records": records, "selection_missing": reasons,
        "link_selected_programs": link_programs, "discarded_copies": discarded,
        "fp_spills": spills, "stack_traffic": traffic, "structural": structural,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for option in ("b1-summary", "b1-evidence", "build", "postprocess", "maps", "mode", "tree-sha", "tree-root",
                   "image", "identity", "disassembler", "out"):
        parser.add_argument(f"--{option}", required=True)
    parser.add_argument("--per-copy", action="store_true",
                        help="diagnostic: key every program by its TU copy, also weak/COMDAT ones (never qualified)")
    args = parser.parse_args()
    record = generate(args)
    Path(args.out).write_text(json.dumps(record, sort_keys=True, indent=1) + "\n")
    print(f"production record {args.mode}: {len(record['stack_traffic'])} programs in {len(record['copies'])} copies; "
          f"structural failures {len(record['structural'])}; selection records {len(record['selection_records'])}, "
          f"missing {len(record['selection_missing'])}; qualified {record['qualified']}")


if __name__ == "__main__":
    main()
