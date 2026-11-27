#!/usr/bin/env python3
"""P1/P2 link-level evidence of one completed canonical build (run inside the toolchain image; the build directory
mounted READ-ONLY at its canonical /src/build_v2_<kind>; a new artifact directory).

- Build status: whether the canonical gate target finished (BUILD_DONE in its log) is recorded as given; a gate
  that failed before a production target was materialised leaves that target MISSING, which is reported and never
  counted as P2. An independent production-target build may acquire artifacts later but never qualifies the gate.
- Per production target (libllaminar2_core.so, llaminar2): the original file is copied and hashed BEFORE anything
  else; its link edge's rule (`ninja -t query`) and the rsp-expanded link command (`ninja -t compdb -x <rule>`) are
  recorded; every input file the command names is hashed; the command is re-run once with only `-o` redirected to
  a temporary path and `-Wl,-Map=<artifact>` added; the relinked file must be byte-identical to the original and the
  inputs and the original unchanged afterwards (else FAIL). The map is kept xz-compressed.
- LTO (B1-d): any -flto* in a production link command is FATAL.
- Selected weak copies (B1-b): every audited symbol defined with non-local binding by a replayed TU (B1 helper
  copies and the programs' call closures) is located in the link maps (the kept input section / defining object);
  its defining object must be a B1-PASS TU, and a symbol found in no map fails.
- Process symbol resolution (B1-e): the audited symbols are the B1 reference closure (non-local names of closure
  functions and the closure's imports, reviewer-86 C-1) plus the B1 helper copies - the FAMILIES regex is only
  a cross-check (symbols it matches outside that set are listed). The actual llaminar2 is started with `--help`
  (SubcommandRouter: a leading flag prints the top-level help and returns 0) with LD_BIND_NOW=1,
  LD_DEBUG=bindings,scopes and LLAMINAR_FORCE_CPU_ONLY_STARTUP=1 - an externally supported entry (read by
  DebugEnv::reload(), set by the ARM64 PR job as `--env`, covered by Test__BackendStartupConfig); no device claim
  is made or needed. A non-zero exit or no LD_DEBUG log fails; each process's (descendants included) log is kept
  separately. With LD_BIND_NOW every relocated import is bound at load, so a symbol may be "unobserved" only when
  no dynamic relocation (.rela.dyn/.rela.plt of llaminar2 or the core library) references it; a relocated symbol
  without an observed binding FAILS.
- Toolchain and configuration facts: versions and hashes of g++/ld/ninja/cmake/nvcc, the image id, the CMake cache
  build type / ISA / IPO / flag lines and the -flto census of build.ninja.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import lzma
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

TARGETS = ("libllaminar2_core.so", "llaminar2")
FAMILIES = re.compile(r"fp16_to_fp32|coded_metadata|dotHalf|CPUNativeQuantisedWeightView|multi_scale|"
                      r"CPUNativeNEONProjectionTile|addOrdered")
BINDING = re.compile(r"binding file (\S+) \[\d+\] to (\S+) \[\d+\]: normal symbol `([^']+)'")


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(argv, cwd, env=None, check=True, input_text=None):
    result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, input=input_text)
    if check and result.returncode:
        raise SystemExit(f"FAIL: {' '.join(argv[:3])}: {result.stderr[-800:]}")
    return result


def link_record(build: str, target: str, artifacts: Path) -> dict:
    original = Path(build) / target
    record: dict = {"target": target}
    if not original.is_file():
        record["status"] = "MISSING"
        return record
    keep = artifacts / "originals" / target
    keep.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, keep)
    record["original_sha256"] = sha256(original)
    if sha256(keep) != record["original_sha256"]:
        raise SystemExit(f"FAIL: copying {target} changed its bytes")
    query = run(["ninja", "-C", build, "-t", "query", target], build).stdout
    rules = re.findall(r"^\s+input: (\S+)$", query, re.MULTILINE)
    if len(rules) != 1:
        raise SystemExit(f"FAIL: {target} has {len(rules)} producing rules")
    entries = json.loads(run(["ninja", "-C", build, "-t", "compdb", "-x", rules[0]], build).stdout)
    entries = [entry for entry in entries if entry.get("output") == target]
    if len(entries) != 1:
        raise SystemExit(f"FAIL: compdb names {len(entries)} link commands for {target}")
    command = entries[0]["command"]
    record.update(rule=rules[0], link_command=command, directory=entries[0]["directory"])
    argv = shlex.split(command)
    if any(token in ("&&", "||", ";", "|") for token in argv):
        # CMake link rules may chain `: && ... && :`; only the one compiler-driver link is re-run.
        segments, current = [], []
        for token in argv:
            if token in ("&&", "||", ";", "|"):
                segments.append(current); current = []
            else:
                current.append(token)
        segments.append(current)
        links = [segment for segment in segments if "-o" in segment and target in segment]
        if len(links) != 1:
            raise SystemExit(f"FAIL: the link command of {target} holds {len(links)} linking segments")
        argv = links[0]
    inputs = sorted({token for token in argv if not token.startswith("-") and (Path(build) / token).is_file()
                     and token != target})
    before = {path: sha256(Path(build) / path) for path in inputs}
    record["inputs"] = before
    relinked = Path("/tmp") / f"relinked-{target}"
    mapped = artifacts / "maps" / f"{target}.map"
    mapped.parent.mkdir(parents=True, exist_ok=True)
    if any(token.startswith("-flto") or token.startswith("-Wl,-plugin") for token in argv):
        raise SystemExit(f"FAIL: LTO in the link of {target} (FATAL: B1-d)")
    index = argv.index("-o")
    if argv[index + 1] != target:
        raise SystemExit(f"FAIL: the link command of {target} writes {argv[index + 1]}")
    relink = argv[:index + 1] + [str(relinked)] + argv[index + 2:] + [f"-Wl,-Map={mapped}"]
    # The canonical link writes its own dependency file into the build tree (-Wl,--dependency-file=...): redirected
    # to /tmp (the build is read-only), and the relinked one must equal the original with only the output renamed.
    dependency = [i for i, token in enumerate(relink) if token.startswith("-Wl,--dependency-file=")]
    if len(dependency) > 1:
        raise SystemExit(f"FAIL: the link command of {target} names {len(dependency)} dependency files")
    original_dependency = None
    if dependency:
        original_dependency = relink[dependency[0]].split("=", 1)[1]
        relink[dependency[0]] = f"-Wl,--dependency-file=/tmp/relinked-{target}.d"
    record["relink_command"] = relink
    run(relink, build)
    if dependency:
        # Ninja consumed the original dependency file into its deps log: the relink's must name the same files.
        lines = run(["ninja", "-C", build, "-t", "deps", target], build).stdout.splitlines()
        if not lines or "(VALID)" not in lines[0]:
            raise SystemExit(f"FAIL: ninja holds no valid recorded link deps for {target}")
        recorded = sorted({os.path.realpath(os.path.join(build, line.strip())) for line in lines[1:] if line.strip()})
        text = Path(f"/tmp/relinked-{target}.d").read_text().replace("\\\n", " ")
        first_rule = text.strip().splitlines()[0]  # ld also writes one phony "file:" rule per dependency.
        relinked_deps = sorted({os.path.realpath(os.path.join(build, item)) for item in first_rule.partition(":")[2].split()})
        record["link_dependencies"] = {"recorded": len(recorded), "relink_equal": relinked_deps == recorded}
        if relinked_deps != recorded:
            raise SystemExit(f"FAIL: the relink's dependencies differ from ninja's recorded link deps for {target}: "
                             f"{sorted(set(recorded) ^ set(relinked_deps))[:4]}")
    after = {path: sha256(Path(build) / path) for path in inputs}
    if after != before or sha256(original) != record["original_sha256"]:
        raise SystemExit(f"FAIL: link inputs or the original {target} changed during the relink")
    record["relinked_sha256"] = sha256(relinked)
    if record["relinked_sha256"] != record["original_sha256"]:
        raise SystemExit(f"FAIL: the relink of {target} with -Map is not byte-identical to the original")
    with open(mapped, "rb") as source, lzma.open(f"{mapped}.xz", "wb") as target_file:
        shutil.copyfileobj(source, target_file)
    mapped.unlink()
    record["map"] = f"maps/{target}.map.xz"
    record["map_sha256"] = sha256(f"{mapped}.xz")
    relinked.unlink()
    record["status"] = "PASS"
    return record


def audited_symbols(b1: dict) -> tuple[dict, set]:
    """({symbol: {"kind", "tus"}}, B1-PASS objects): the B1 global reference closure (C-1) - non-local names of
    closure functions with their defining objects, and its imports (names no target object defines) - plus the
    non-local helper copies of every B1 TU. A B1 summary without a closure fails."""
    closure = b1.get("closure")
    if not closure or "nonlocal_definitions" not in closure:
        raise SystemExit("FAIL: the B1 summary holds no reference closure (v3 B1 required)")
    # A non-GNU TU (N-1) is never a valid selection source: only replayed GNU TUs are.
    symbols, passed = {}, {tu["object"] for tu in b1["tus"] if tu.get("status") == "PASS" and tu.get("mode") != "non-gnu"}
    for name, objects in closure["nonlocal_definitions"].items():
        symbols[name] = {"kind": "closure-definition", "tus": list(objects)}
    for name, objects in closure.get("nonlocal_data", {}).items():
        symbols.setdefault(name, {"kind": "closure-data", "tus": list(objects)})
    for name in closure["imports"]:
        symbols.setdefault(name, {"kind": "closure-import", "tus": []})
    for tu in b1["tus"]:
        for name, binding in tu.get("helper_bindings", {}).items():
            if binding != "local":
                symbols.setdefault(name, {"kind": "helper-copy", "tus": []})["tus"].append(tu["object"])
    return symbols, passed


MAP_SECTION = re.compile(r"^ (\S+)(?:\s+0x[0-9a-f]+\s+0x[0-9a-f]+\s+(\S.*))?$")
MAP_PLACEMENT = re.compile(r"^\s+0x[0-9a-f]+\s+0x[0-9a-f]+\s+(\S.*)$")
MAP_SYMBOL = re.compile(r"^\s+0x[0-9a-f]+\s+(\S.*)$")


COMDAT_SECTION = re.compile(r"^\.(?:text|rodata|data\.rel\.ro\.local|data\.rel\.ro|data\.rel\.local|data\.rel|data|bss|"
                            r"tdata|tbss)\.([A-Za-z_][\w$.]*)$")


def comdat_symbol(section: str) -> str | None:
    """The symbol a per-symbol (COMDAT) input section is named after: .text./.rodata./.data[.rel[.ro][.local]]./
    .bss./.tdata./.tbss. + a symbol name (code and data, mangled or not); never a .rela.* section. Section-only suffixes (str1.8, cst16,
    unlikely) name no symbol and are never looked up."""
    match = COMDAT_SECTION.match(section)
    return match.group(1) if match else None


def map_definitions(map_text: str) -> dict[str, set]:
    """Symbol -> objects defining it in the kept (memory map) part of a GNU ld map: global symbol lines under each
    input section, and comdat input sections named after a mangled symbol (code and data, comdat_symbol())."""
    definitions: dict[str, set] = {}
    body = map_text.split("Linker script and memory map", 1)[-1]
    current, pending = None, None
    for line in body.splitlines():
        placement = MAP_PLACEMENT.match(line)
        if pending is not None and placement:
            current = placement.group(1).strip()
            if comdat_symbol(pending):
                definitions.setdefault(comdat_symbol(pending), set()).add(current)
            pending = None
            continue
        section = MAP_SECTION.match(line)
        if section:
            name, obj = section.group(1), section.group(2)
            if obj:
                current = obj.strip()
                if comdat_symbol(name):
                    definitions.setdefault(comdat_symbol(name), set()).add(current)
                pending = None
            else:
                pending = name
            continue
        symbol = MAP_SYMBOL.match(line)
        if symbol and current and not line.lstrip().startswith("0x0000000000000000 "):
            definitions.setdefault(symbol.group(1).strip(), set()).add(current)
    return definitions


def selected_copies(build: str, artifacts: Path, symbols: dict, passed: set) -> dict:
    definitions: dict[str, set] = {}
    for target in TARGETS:
        mapped = artifacts / "maps" / f"{target}.map.xz"
        if mapped.is_file():
            with lzma.open(mapped, "rt", errors="replace") as stream:
                for name, objects in map_definitions(stream.read()).items():
                    definitions.setdefault(name, set()).update(objects)
    names = sorted(name for name, entry in symbols.items() if entry["kind"] != "closure-import")
    demangled = dict(zip(names, run(["c++filt"], build, check=False, input_text="\n".join(names)).stdout.splitlines()))
    result, failures = {}, []
    for name in names:
        objects = definitions.get(name, set()) | definitions.get(demangled.get(name, name), set())
        if not objects:
            result[name] = {"status": "FAIL (in no link map)"}
            failures.append(name)
            continue
        outside = sorted(obj for obj in objects if obj not in passed)
        result[name] = {"objects": sorted(objects), "status": "FAIL (selected copy from a TU without B1)" if outside else "PASS"}
        if outside:
            failures.append(name)
    return {"status": "FAIL" if failures else "PASS", "failures": failures[:50], "symbols": result}


def map_objects_named(obj: str, objects: set) -> list:
    """Map spellings of one build object: its path, or an archive member <lib>(<basename>)."""
    base = Path(obj).name
    return sorted(o for o in objects if o == obj or o.endswith(f"({base})"))


def non_gnu_selection(build: str, artifacts: Path, b1: dict, passed: set) -> dict:
    """N-1: every definition a non-GNU TU shares with GNU target objects is NON-selected in the link maps: the kept
    copy comes from a B1-PASS GNU TU, never from the non-GNU object. Bound to the map and B1 closure digests."""
    definitions: dict[str, set] = {}
    for target in TARGETS:
        mapped = artifacts / "maps" / f"{target}.map.xz"
        if mapped.is_file():
            with lzma.open(mapped, "rt", errors="replace") as stream:
                for name, objects in map_definitions(stream.read()).items():
                    definitions.setdefault(name, set()).update(objects)
    result, failures = {}, []
    for tu in b1["tus"]:
        if tu.get("mode") != "non-gnu":
            continue
        entry = {}
        # Reviewer-91 N-1 scope: only definitions that can come from an audited header (audited_shared, b1).
        for name in tu.get("audited_shared", []):
            kept = definitions.get(name, set())
            own = map_objects_named(tu["object"], kept)
            others = sorted(kept - set(own))
            if own:
                status = "FAIL (the non-GNU copy is selected)"
            elif not others:
                status = "FAIL (in no link map)"
            elif any(o not in passed for o in others):
                status = "FAIL (selected copy from a TU without B1)"
            else:
                status = "PASS (non-selected)"
            entry[name] = {"kept": others or own, "status": status}
            if not status.startswith("PASS"):
                failures.append(f"{tu['object']}: {name}: {status}")
        result[tu["object"]] = {"class": (tu.get("classification") or {}).get("classes"), "shared": entry,
                                "shared_recorded_not_audited": tu.get("shared_recorded", []),
                                "exclusive": len(tu.get("exclusive_definitions", []))}
    maps = {target: sha256(artifacts / "maps" / f"{target}.map.xz") for target in TARGETS
            if (artifacts / "maps" / f"{target}.map.xz").is_file()}
    return {"status": "FAIL" if failures else "PASS", "failures": failures[:50], "objects": result,
            "map_sha256": maps, "b1_closure_digest": b1.get("closure_digest")}


def dynamic_relocations(build: str, path: str) -> dict[str, list]:
    """Symbol -> dynamic relocation sections referencing it (readelf -rW)."""
    result, section = {}, None
    for line in run(["readelf", "-rW", path], build).stdout.splitlines():
        header = re.match(r"^Relocation section '([^']+)'", line)
        if header:
            section = header.group(1)
            continue
        fields = line.split()
        if section and len(fields) >= 5 and re.fullmatch(r"[0-9a-f]+", fields[0]):
            name = fields[4].split("@")[0]
            result.setdefault(name, []).append(section)
    return result


def process_resolution(build: str, artifacts: Path, symbols: dict) -> dict:
    executable = Path(build) / "llaminar2"
    if not executable.is_file() or not (Path(build) / "libllaminar2_core.so").is_file():
        return {"status": "INCOMPLETE (llaminar2 or the core library not materialised: no process resolution)"}
    prefix = "/tmp/lddebug"
    for stale in glob.glob(prefix + ".*"):
        os.remove(stale)
    env = dict(os.environ, LD_BIND_NOW="1", LD_DEBUG="bindings,scopes", LD_DEBUG_OUTPUT=prefix,
               LLAMINAR_FORCE_CPU_ONLY_STARTUP="1")
    started = run(["./llaminar2", "--help"], build, env, check=False)
    logs = sorted(glob.glob(prefix + ".*"))
    record = {"command": ["./llaminar2", "--help"], "environment": {"LD_BIND_NOW": "1", "LD_DEBUG": "bindings,scopes",
              "LLAMINAR_FORCE_CPU_ONLY_STARTUP": "1"}, "exit_code": started.returncode,
              "output_tail": (started.stdout + started.stderr)[-600:], "process_logs": []}
    text = ""
    for path in logs:
        content = Path(path).read_text(errors="replace")
        text += content
        name = f"process-ld-debug.pid{path.rsplit('.', 1)[-1]}.txt.xz"
        with lzma.open(artifacts / name, "wt") as stream:
            stream.write(content)
        record["process_logs"].append({"pid": path.rsplit(".", 1)[-1], "log": name, "lines": content.count("\n")})
    if started.returncode or not logs:
        record["status"] = f"FAILED ({'exit ' + str(started.returncode) if started.returncode else ''}" \
                           f"{' no LD_DEBUG log' if not logs else ''})"
        return record
    observed: dict[str, set] = {}
    for referencing, defining, symbol in BINDING.findall(text):
        observed.setdefault(symbol, set()).add(f"{referencing} -> {defining}")
    relocated: dict[str, list] = {}
    for owner in TARGETS:
        for name, sections in dynamic_relocations(build, owner).items():
            relocated.setdefault(name, []).extend(f"{owner}:{section}" for section in sections)
    entries, failures = {}, []
    for name, entry in sorted(symbols.items()):
        relocations, bindings = sorted(set(relocated.get(name, []))), sorted(observed.get(name, ()))
        if bindings:
            status = "observed"
        elif relocations:
            status = "FAIL (dynamically relocated but no binding observed under LD_BIND_NOW)"
            failures.append(name)
        else:
            status = "unobserved (no dynamic relocation references it)"
        entries[name] = {**entry, "relocations": relocations, "bindings": bindings, "status": status}
    regex_only = sorted(name for name in set(relocated) | set(observed) if FAMILIES.search(name) and name not in symbols)
    record.update(symbols=entries, regex_cross_check_outside_closure=regex_only[:200],
                  scopes=[line for line in text.splitlines() if "scope" in line][:200],
                  status="FAIL" if failures else "PASS", failures=failures[:50])
    return record


def facts(build: str, image_id: str) -> dict:
    tools = {}
    for name, argv in (("g++-14", ["g++-14", "--version"]), ("ld", ["ld", "--version"]), ("ninja", ["ninja", "--version"]),
                       ("cmake", ["cmake", "--version"]), ("nvcc", ["nvcc", "--version"])):
        path = shutil.which(argv[0])
        version = run(argv, build, check=False)
        tools[name] = {"path": path, "resolved": os.path.realpath(path) if path else None,
                       "sha256": sha256(os.path.realpath(path)) if path else None,
                       "version": (version.stdout + version.stderr).strip().splitlines()[:2]}
    cache = (Path(build) / "CMakeCache.txt").read_text(errors="replace")
    selected = [line for line in cache.splitlines() if re.match(
        r"^(CMAKE_BUILD_TYPE|LLAMINAR_CPU_ISA|CMAKE_INTERPROCEDURAL_OPTIMIZATION|CMAKE_CXX_FLAGS[A-Z_]*|"
        r"CMAKE_(SHARED|EXE)_LINKER_FLAGS[A-Z_]*|HAVE_CUDA|HAVE_ROCM|LLAMINAR_SHARED_CORE|CMAKE_CUDA_ARCHITECTURES|"
        r"CMAKE_CXX_COMPILER|CMAKE_CXX_COMPILER_LAUNCHER)[:=]", line)]
    lto = sorted(set(re.findall(r"-flto[=\w-]*", (Path(build) / "build.ninja").read_text(errors="replace"))))
    return {"image_id": image_id, "tools": tools, "cmake_cache": selected, "lto_flags": lto,
            "build_ninja_sha256": sha256(Path(build) / "build.ninja")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--build-log", required=True, help="the canonical build's log (read-only)")
    parser.add_argument("--targets", nargs="+", default=list(TARGETS), help="link targets (controls only)")
    parser.add_argument("--b1", required=True, help="the B1 summary of the same build")
    args = parser.parse_args()
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=False)
    log = Path(args.build_log).read_text(errors="replace")
    summary = {"build": args.build, "gate_build_completed": "BUILD_DONE" in log,
               "gate_build_note": "the canonical target's own build status; a failed gate is reported, never waived",
               "facts": facts(args.build, args.image_id)}
    summary["links"] = {target: link_record(args.build, target, artifacts) for target in args.targets}
    b1 = json.loads(Path(args.b1).read_text())
    symbols, passed = audited_symbols(b1)
    summary["b1_status"] = b1.get("status")
    summary["audited_symbols"] = len(symbols)
    summary["selected_copies"] = selected_copies(args.build, artifacts, symbols, passed)
    summary["non_gnu_selection"] = non_gnu_selection(args.build, artifacts, b1, passed)
    summary["process_resolution"] = process_resolution(args.build, artifacts, symbols)
    missing = [target for target, record in summary["links"].items() if record["status"] == "MISSING"]
    failed = [part for part in ("selected_copies", "non_gnu_selection", "process_resolution")
              if summary[part]["status"] not in ("PASS",)]
    if b1.get("status") != "PASS":
        failed.append("b1")
    summary["p2_status"] = ("INCOMPLETE (missing: " + ", ".join(missing) + ")" if missing else
                            "FAIL (" + ", ".join(failed) + ")" if failed else "COMPLETE")
    (artifacts / "postprocess-summary.json").write_text(json.dumps(summary, sort_keys=True, indent=1, default=sorted) + "\n")
    print(f"POSTPROCESS {summary['p2_status']}; gate build completed: {summary['gate_build_completed']}")
    raise SystemExit(0 if summary["p2_status"] == "COMPLETE" else 2)


if __name__ == "__main__":
    main()
