#!/usr/bin/env python3
"""B1 for production objects (v6: reviewer-83 B1-a..d, reviewer-86 C-1..C-4, reviewer-88 C-5, reviewer-91 N-1/N-2, reviewer-98 D-1/D-2, reviewer-109 G-1/G-2): retain the assembler input of every audited production TU's
executed compile and prove it is exactly what was assembled into the build's object.

Runs inside the canonical toolchain image with the source snapshot laid out at the build's /src paths, the build
directory mounted READ-ONLY at its canonical path and a separate evidence directory. Nothing in the build or source
tree is written.

Selection (B1-b): every object that is an input of the production target(s) (`ninja -t inputs`) whose recorded
ninja deps include at least one audited header (the explicit, hashed list --audited-headers: the CPUNativeNEON*,
CPUNativeQuantised* headers and .inc files, FP16Utils.h; a superset of the nm selection, so fully inlined audited
code and TUs holding only a helper copy are included). Each selected TU is classified from its analysed assembly:
  program  defines a native NEON tile program or a compact multi-scale program (symbol table);
  helper   defines a function whose definition (its first .loc) is in an audited header (an out-of-line copy);
  inlined  holds .loc lines into audited headers inside functions defined elsewhere;
  none     zero .loc into audited headers and zero audited definitions (included through the header only).
A TU that is not a GNU C++ compile (e.g. nvcc) cannot be classified: FAIL.

Per TU:
1. Executed command: `ninja -t commands -s <object>`, a single command; any -flto* in it is FATAL (B1-d).
2. Launcher route (ccache): the command through ccache with a fresh cache, CCACHE_DEBUG=1, only the VALUES of -o
   and -MF redirected; the route object must be byte-identical to the build's object; the debug log yields the
   one compiler argv executed.
3. GNU plan of that argv: exactly cc1plus + as, absolute cc??????.s temporary read once and last, programs equal
   to the driver's -print-prog-name answers.
4. Replay with the temporary replaced by a retained <tu>.s: the replayed object must be byte-identical to the
   build's object.
5. Debug mode (C-3a): no -g* option in the executed argv -> Release debug augmentation; only plain debug-enabling
   options (-g, -g1..3, -ggdb[1-3], -gdwarf[-N]) -> native; any other -g* (-g0, -gno-*, -gz, ...) FAILS as
   ambiguous. Augmentation (B1-a): the cc1plus job with -g appended, the as job unchanged, retains <tu>.aug.s and
   its object; every non-debug section (bytes, flags, alignment, entry size), every relocation of a non-debug
   section, every group and the non-debug symbol table must equal the build object's (elf_tools.compare_nondebug;
   "debug" is exactly .debug_*, .rela.debug_* and their section symbols, C-3b - .comment/.note.* differences
   FAIL). The analysed assembly is then <tu>.aug.s and its identity records the augmentation (C-3c).
6. Closure: the replay's depfile equals `ninja -t deps <object>`; each file hashed before 2 and after 4/5 (equal)
   and classified (C-4): source (git blob of the snapshot commit, B1-c), generated (build directory), external-
   source / external-generated (a configure-fetched checkout under /src/external: tracked files equal its
   committed blobs, untracked build products hashed), system (under /usr on the image's own filesystem, not a
   bind mount); anything else is unclassified and FAILS.
7. Reference graph (C-1, elf_tools.function_graph) of the build object: every relocation of an allocated section
   is an edge (calls, branches, address-taking code and data references); uncovered or unresolvable references
   FAIL the TU; indirect branches are listed per function; assembler-resolved (relocation-free) PC-relative
   references are decoded from the code bytes (C-5a-c). Kept as <tu>.graph.json with the per-function .loc
   files of the analysed assembly.
8. Assembly cross-check (C-5d, object-wide): every symbolic operand of a branch/ADR/ADRP/literal-load or a
   :modifier: operand in every function body of the analysed assembly must correspond to a graph edge of that
   function (relocation or decoded); a local label defined in the same function and section needs none. A
   missing correspondence FAILS, naming the function, the line and the operand.
Closure fixpoint (C-1/C-2): from every program function in any target object, the reference closure across all
target objects (all non-local copies of a reached name); reached objects not yet replayed are replayed and the
closure recomputed until none remain. Every closure function's .loc files must be in the audited list, the image's
system paths or a fetched external checkout; a project file outside the list (named), an unclassified file or a
closure function without .loc FAILS (the list is checked mechanically, not trusted).
The summary is PASS only if every selected TU passes and no target input object is missing.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import elf_tools  # noqa: E402

PROGRAMS = re.compile(r"CPUNativeNEONProjectionTile.*(?:3run|8runCoded)|11multi_scale11rowsProgram|11multi_scale4rows")
GNU_TEMPORARY = re.compile(r"cc[A-Za-z0-9]{6}\.s")
CONFIGURATION = re.compile(r"^(?:Reading specs from|Configuration file:|Using specs)", re.MULTILINE)
SHELL_OPERATORS = {"&&", "||", ";", "|", ">", "<", "&"}
ASM_FILE = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]*)"(?:\s+"([^"]*)")?')
ASM_LOC = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)")
ASM_LABEL = re.compile(r"^([A-Za-z_$][\w$.]*):")


class Failure(Exception):
    pass


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def run(argv, cwd, env=None):
    return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)


def ninja(build, *arguments):
    result = run(["ninja", "-C", build, *arguments], build)
    if result.returncode:
        raise Failure(f"ninja {' '.join(arguments)} failed: {result.stderr[-500:]}")
    return result.stdout


def executed_command(build, output):
    lines = [line for line in ninja(build, "-t", "commands", "-s", output).splitlines() if line.strip()]
    if len(lines) != 1:
        raise Failure(f"ninja names {len(lines)} commands for {output}, not exactly one")
    argv = shlex.split(lines[0])
    if any(token in SHELL_OPERATORS for token in argv):
        raise Failure(f"the executed command of {output} is a shell pipeline")
    return argv


def option_value(argv, option):
    places = [index for index, argument in enumerate(argv[:-1]) if argument == option]
    if len(places) != 1:
        raise Failure(f"the command names {option} {len(places)} times, not once")
    return argv[places[0] + 1]


def depfile_paths(path, cwd):
    text = Path(path).read_text().replace("\\\n", " ")
    if "\\ " in text or "$$" in text:
        raise Failure(f"depfile {Path(path).name} names escaped paths (unsupported)")
    return sorted({os.path.realpath(os.path.join(cwd, item)) for item in text.partition(":")[2].split()})


def ninja_deps(build, output):
    lines = ninja(build, "-t", "deps", output).splitlines()
    if not lines or "(VALID)" not in lines[0]:
        raise Failure(f"ninja holds no valid recorded deps for {output}: {lines[:1]}")
    return sorted({os.path.realpath(os.path.join(build, line.strip())) for line in lines[1:] if line.strip()})


def executing_argv(log):
    found = []
    for line in Path(log).read_text(errors="replace").splitlines():
        marker = line.find("Executing ")
        if marker < 0:
            continue
        argv = shlex.split(line[marker + len("Executing "):])
        if "-c" in argv and "-E" not in argv:
            found.append(argv)
    if len(found) != 1:
        raise Failure(f"the ccache debug log names {len(found)} compile executions, not exactly one")
    return found[0]


DEBUG_ENABLING = re.compile(r"-g(?:[123]|gdb[123]?|dwarf(?:-[2-5])?)?")


def debug_mode(argv) -> str:
    """C-3a: "augment" iff the executed argv carries no -g* option; "native" iff every -g* option is a plain
    debug-enabling one (-g, -g1..3, -ggdb[1-3], -gdwarf[-N]); anything else (-g0, -gno-*, -gz, -gsplit-dwarf, ...)
    is ambiguous for the .loc evidence: FAIL."""
    options = [argument for argument in argv[1:] if argument.startswith("-g")]
    if not options:
        return "augment"
    ambiguous = [option for option in options if not DEBUG_ENABLING.fullmatch(option)]
    if ambiguous:
        raise Failure(f"the executed compile carries debug options {ambiguous}: augmentation is ambiguous")
    return "native"


def program_identity(path):
    resolved = os.path.realpath(path)
    return {"path": path, "resolved": resolved, "sha256": sha256(resolved)}


def gnu_plan(argv, cwd):
    expansion = run(argv + ["-###"], cwd)
    if expansion.returncode:
        raise Failure(f"the driver plan query failed: {expansion.stderr[-300:]}")
    text = expansion.stdout + expansion.stderr
    if CONFIGURATION.search(text):
        raise Failure("the compiler driver reads a specs or configuration file")
    jobs = [shlex.split(line) for line in text.splitlines() if line.startswith(" ")]
    if len(jobs) != 2 or Path(jobs[0][0]).name != "cc1plus" or Path(jobs[1][0]).name != "as":
        raise Failure(f"the driver plan is not exactly cc1plus then as: {[Path(job[0]).name for job in jobs]}")
    compiler, assembler = jobs
    temporary = option_value(compiler, "-o")
    if not os.path.isabs(temporary) or not GNU_TEMPORARY.fullmatch(os.path.basename(temporary)):
        raise Failure(f"the compiler job output is not a driver temporary assembly file: {temporary}")
    if compiler.count(temporary) != 1 or assembler.count(temporary) != 1 or assembler[-1] != temporary:
        raise Failure("the assembler job must read exactly the compiler's temporary output, once, last")
    programs = {}
    for job in jobs:
        answer = run(argv + [f"-print-prog-name={Path(job[0]).name}"], cwd).stdout.strip()
        selected = job[0] if os.path.isabs(job[0]) else shutil.which(job[0])
        answer = answer if os.path.isabs(answer) else shutil.which(answer)
        if not selected or not answer or os.path.realpath(selected) != os.path.realpath(answer):
            raise Failure(f"the driver selects {answer} for {job[0]}, the plan names {selected}")
        programs[Path(job[0]).name] = program_identity(selected)
    return jobs, temporary, programs


def replay(jobs, temporary, mapping, cwd, cc1_extra=()):
    """Run the plan with the given path mapping; extra arguments are appended to the cc1plus job only."""
    for index, job in enumerate(jobs):
        argv = [mapping.get(argument, argument) for argument in job]
        if index == 0:
            argv = argv + list(cc1_extra)
        result = run(argv, cwd)
        if result.returncode:
            raise Failure(f"plan job {index} failed on replay: {result.stderr[-1500:]}")


def parse_assembly(assembly, function_names: set[str], cwd: str = "/") -> dict:
    """Per function (a label that is a function symbol of the object, up to its .size or the next label): the
    files of its .loc lines (relative .file entries - ccache base_dir rewriting - resolved against the compile's
    working directory); .loc lines outside every function are counted per file."""
    files, function, per_function, outside = {}, None, {}, {}
    for line in Path(assembly).read_text(errors="replace").splitlines():
        match = ASM_FILE.match(line)
        if match:
            path = os.path.join(match.group(2), match.group(3)) if match.group(3) is not None else match.group(2)
            files[match.group(1)] = os.path.realpath(os.path.join(cwd, path))
            continue
        match = ASM_LABEL.match(line)
        if match and not match.group(1).startswith(".L"):
            function = match.group(1) if match.group(1) in function_names else None
            if function is not None:
                per_function.setdefault(function, {"files": {}, "first": None})
            continue
        stripped = line.strip()
        if function is not None and stripped.startswith(".size") and stripped.split()[1].rstrip(",") == function:
            function = None
            continue
        match = ASM_LOC.match(line)
        if match:
            path = files.get(match.group(1), f"<unknown .file {match.group(1)}>")
            if function is None:
                outside[path] = outside.get(path, 0) + 1
                continue
            entry = per_function[function]
            entry["first"] = entry["first"] or path
            entry["files"][path] = entry["files"].get(path, 0) + 1
    return {"functions": per_function, "outside": outside}


ASM_ANY_LABEL = re.compile(r"^([A-Za-z_.$][\w$.]*):")
ASM_SET = re.compile(r"^\s*\.(?:set|equ)\s+([A-Za-z_.$][\w$.]*)\s*,\s*(.+)$")
ASM_HERE = re.compile(r"\.\s*(?:[+-]\s*(?:0x[0-9a-fA-F]+|\d+))?")
ASM_SECTION = re.compile(r'^\s*\.(?:section|pushsection)\s+("?)([^",\s]+)\1')
ASM_SIMPLE_SECTION = re.compile(r"^\s*\.(text|data|bss|rodata)\s*$")
ASM_INSTRUCTION = re.compile(r"^\s+([a-z][a-z0-9.]*)(?:\s+(.*))?$")
ASM_MODIFIED = re.compile(r":([a-z0-9_]+):([A-Za-z_.$][\w.$]*)")
ASM_SYMBOL = re.compile(r"^([A-Za-z_.$][\w.$]*)(?:[+-](?:0x[0-9a-fA-F]+|\d+))?$")
ASM_REGISTER = re.compile(r"^(?:[xwvqdshbz]\d+|sp|wsp|xzr|wzr|lr|fp|p\d+)(?:\.\w+)?(?:/[zm])?$")
BRANCH_MNEMONICS = re.compile(r"^(?:b|bl|b\.\w+|bc\.\w+|cbz|cbnz|tbz|tbnz|adr|adrp)$")
LITERAL_MNEMONICS = {"ldr", "ldrsw", "prfm"}


def assembly_references(assembly, function_names: set[str]):
    """Per function (label of a function symbol to its .size, lines in the function's own section only): the
    symbolic operands of its instructions [(line number, mnemonic, symbol)], and every label -> (function or
    None, section)."""
    section, stack, previous = ".text", [], ".text"
    function, function_section = None, None
    references, labels, aliases = {}, {}, {}
    for number, raw in enumerate(Path(assembly).read_text(errors="replace").splitlines(), 1):
        line = raw.split("//", 1)[0].rstrip()
        match = ASM_SECTION.match(line)
        simple = ASM_SIMPLE_SECTION.match(line)
        if match or simple:
            if line.lstrip().startswith(".pushsection"):
                stack.append(section)
            previous, section = section, (match.group(2) if match else "." + simple.group(1))
            continue
        stripped = line.strip()
        if stripped == ".popsection" and stack:
            previous, section = section, stack.pop()
            continue
        if stripped == ".previous":
            previous, section = section, previous
            continue
        assigned = ASM_SET.match(line)
        if assigned:
            # `.set NAME, . + N` defines NAME here; `.set NAME, OTHER[+N]` aliases another symbol (resolved below).
            name, value = assigned.group(1), assigned.group(2).strip()
            here = (function if function is not None and section == function_section else None, section)
            if ASM_HERE.fullmatch(value):
                labels[name] = here
            else:
                target = ASM_SYMBOL.match(value.replace(" ", ""))
                aliases[name] = target.group(1) if target else None
            continue
        label = ASM_ANY_LABEL.match(line)
        if label:
            name = label.group(1)
            if name in function_names:
                function, function_section = name, section
                references.setdefault(name, [])
            labels[name] = (function if function is not None and section == function_section else None, section)
            continue
        if function is not None and stripped.startswith(".size") and stripped[5:].split(",")[0].strip() == function:
            function = None
            continue
        if function is None or section != function_section:
            continue
        instruction = ASM_INSTRUCTION.match(line)
        if not instruction or instruction.group(1).startswith("."):
            continue
        mnemonic, operands = instruction.group(1), instruction.group(2) or ""
        found = [symbol for _, symbol in ASM_MODIFIED.findall(operands)]
        parts = [part.strip() for part in re.split(r",(?![^\[]*\])", operands)]
        candidate = None
        if BRANCH_MNEMONICS.match(mnemonic) and parts:
            candidate = parts[-1]
        elif mnemonic in LITERAL_MNEMONICS and len(parts) == 2 and not parts[1].startswith(("[", "#", "=")):
            candidate = parts[1]
        if candidate and not candidate.startswith(":"):
            symbolic = ASM_SYMBOL.match(candidate)
            if symbolic and not ASM_REGISTER.match(candidate):
                found.append(symbolic.group(1))
        for symbol in found:
            references[function].append((number, mnemonic, symbol))
    for name in aliases:
        seen, target = {name}, aliases[name]
        while target in aliases and target not in seen:
            seen.add(target)
            target = aliases[target]
        if target in labels:
            labels.setdefault(name, labels[target])
        elif target is not None and not target.startswith(".L"):
            labels.setdefault(name, ("=" + target, None))  # An alias of a named symbol: checked by that name.
    return references, labels


def assembly_crosscheck(assembly, graph: dict) -> dict:
    """C-5d: every symbolic operand in every function body of the analysed assembly corresponds to a graph edge."""
    names_to_key = {name: key for key, node in graph["functions"].items() for name in node["names"]}
    references, labels = assembly_references(assembly, set(names_to_key))
    failures, checked = [], 0
    for function, items in references.items():
        key = names_to_key[function]
        edges = graph["edges"].get(key, [])
        node_targets = {target for _, reference, target in edges if reference == "node"}
        name_targets = {target for _, reference, target in edges if reference == "name"}
        for number, mnemonic, symbol in items:
            if symbol.startswith(".L") and labels.get(symbol, ("", ""))[0] and \
                    str(labels[symbol][0]).startswith("="):
                symbol = labels[symbol][0][1:]  # An alias of a named symbol.
            if symbol.startswith(".L"):
                owner, section = labels.get(symbol, (None, None))
                if section is None:
                    failures.append(f"{function} line {number}: {mnemonic} {symbol}: label not defined in the assembly")
                    continue
                if owner == function:
                    continue  # A label of this function body: an internal branch or address.
                if owner is not None:
                    wanted = {names_to_key[owner]}
                else:
                    wanted = {node for node, node_section in graph["node_sections"].items() if node_section == section}
                ok = bool(node_targets & wanted)
            else:
                ok = symbol in name_targets or bool(node_targets & set(graph["symbol_nodes"].get(symbol, ())))
            checked += 1
            if not ok:
                failures.append(f"{function} line {number}: {mnemonic} {symbol} has no corresponding graph edge "
                                f"(relocation or decoded)")
    return {"checked": checked, "failures": failures}


def classify(parsed: dict, audited: set[str], program_symbols: list[str]) -> dict:
    """Classes of one TU from its analysed assembly."""
    per_function = parsed["functions"]
    helpers = sorted(name for name, entry in per_function.items()
                     if entry["first"] in audited and name not in program_symbols)
    inlined = sorted(name for name, entry in per_function.items()
                     if entry["first"] not in audited and audited & set(entry["files"]))
    audited_locs = sum(count for entry in per_function.values() for path, count in entry["files"].items()
                       if path in audited)
    if audited & set(parsed["outside"]):
        raise Failure("audited .loc lines outside every function")
    classes = []
    if program_symbols:
        classes.append("program")
    if helpers:
        classes.append("helper")
    if inlined:
        classes.append("inlined")
    if not classes:
        classes = ["none"]
    return {"classes": classes, "helpers": helpers, "inlined_into": inlined, "audited_locs": audited_locs}


def external_repositories(src_root):
    """Git checkouts the configure step fetched under <src>/external (e.g. oneDNN): {root: {commit, url, blobs}}.
    Their tracked files must equal the checkout's committed blobs; untracked files under them are generated."""
    repositories = {}
    external = Path(src_root) / "external"
    for git_dir in sorted(external.glob("*/.git")) if external.is_dir() else []:
        root = str(git_dir.parent.resolve())
        def git(*arguments):
            result = run(["git", "-c", f"safe.directory={root}", "-C", root, *arguments], root)
            if result.returncode:
                raise Failure(f"git {' '.join(arguments)} failed in {root}: {result.stderr[-300:]}")
            return result.stdout
        blobs = {}
        for line in git("ls-tree", "-r", "HEAD").splitlines():
            meta, _, path = line.partition("\t")
            mode, kind, blob = meta.split()
            if kind == "blob":
                blobs[path] = blob
        repositories[root] = {"commit": git("rev-parse", "HEAD").strip(), "tree": git("rev-parse", "HEAD^{tree}").strip(),
                              "url": git("config", "--get", "remote.origin.url").strip(), "blobs": blobs}
    return repositories


SYSTEM_PREFIXES = ("/usr/",)


def mount_points() -> list[str]:
    """Mount points of this container other than / (bind mounts are not the image)."""
    points = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        point = line.split()[4].replace("\\040", " ")
        if point != "/":
            points.append(point.rstrip("/") + "/")
    return points


def path_kind(real, build, src_root, repositories, manifest, mounts) -> str:
    """C-4: generated (build directory) | external-source / external-generated (a configure-fetched checkout under
    /src/external) | source (in the snapshot commit) | system (under /usr on the image's own filesystem, not a bind
    mount) | unclassified (anything else, never silently hashed)."""
    if real.startswith(build.rstrip("/") + "/"):
        return "generated"
    repository = next((root for root in repositories if real.startswith(root + "/")), None)
    if repository is not None:
        return "external-source" if os.path.relpath(real, repository) in repositories[repository]["blobs"] \
            else "external-generated"
    if real.startswith(src_root.rstrip("/") + "/"):
        return "source" if os.path.relpath(real, src_root) in manifest else "unclassified"
    if real.startswith(SYSTEM_PREFIXES) and not any(real.startswith(point) for point in mounts):
        return "system"
    return "unclassified"


def closure_entry(path, build, src_root, manifest, repositories=None, mounts=None):
    real = os.path.realpath(path)
    repositories = repositories or {}
    kind = path_kind(real, build, src_root, repositories, manifest, mount_points() if mounts is None else mounts)
    entry = {"sha256": sha256(real), "kind": kind}
    if kind == "unclassified":
        if real.startswith(src_root.rstrip("/") + "/"):
            raise Failure(f"closure file {os.path.relpath(real, src_root)} is not in the snapshot commit")
        raise Failure(f"closure file {real} is outside /src, the build directory, /src/external and the image's "
                      f"system paths (unclassified)")
    if kind.startswith("external-"):
        repository = next(root for root in repositories if real.startswith(root + "/"))
        entry["repository"] = repository
        if kind == "external-source":
            blob = repositories[repository]["blobs"][os.path.relpath(real, repository)]
            if blob != git_blob_sha1(real):
                raise Failure(f"closure file {real} differs from its fetched checkout's committed blob")
            entry["git_blob"] = blob
    elif kind == "source":
        relative = os.path.relpath(real, src_root)
        blob = git_blob_sha1(real)
        if blob != manifest[relative]:
            raise Failure(f"closure file {relative} differs from the snapshot commit's git blob")
        entry["git_blob"] = blob
    return entry


IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def function_identifier(demangled: str) -> str | None:
    """The unqualified name of a demangled function (before its parameter list at template depth 0, template
    arguments and qualifiers stripped); None when it cannot be determined (the caller fails)."""
    # Demangler spellings of unnamed scopes are not parameter lists: "(anonymous namespace)" is a scope component.
    demangled = demangled.replace("(anonymous namespace)", "{anonymous}")
    if not demangled.startswith("_Z") and "(" not in demangled and "::" not in demangled:
        # An unmangled name: a C identifier is checked as such; anything else (e.g. DW.ref.<x>, a compiler
        # generated symbol) cannot be spelled by a source definition.
        return demangled if IDENTIFIER.fullmatch(demangled) else ""
    depth, cut = 0, None
    for index, char in enumerate(demangled):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif char == "(" and depth == 0:
            cut = index
            break
    if cut is None:
        cut = len(demangled)  # A variable: no parameter list.
    head, depth, plain = demangled[:cut], 0, []
    for char in head:  # Drop template argument lists.
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif depth == 0:
            plain.append(char)
    name = "".join(plain).split("::")[-1].strip()
    if name.startswith("~"):
        name = name[1:]
    return name if IDENTIFIER.fullmatch(name) else None


STUB_PREFIX = "__device_stub__"
TMPXFT_STUB = re.compile(r"tmpxft_([0-9a-f]+_[0-9a-f]+)-\d+_(.+)\.cudafe1\.stub\.c")
TMPXFT_CU = re.compile(r"tmpxft_([0-9a-f]+_[0-9a-f]+)-\d+_(.+)\.cudafe1\.cpp")


def stub_kernel(name: str) -> str | None:
    """The kernel host symbol of an nvcc launch stub _Z[L]<n>__device_stub__<mangled kernel without _><params>."""
    match = re.match(r"_ZL?(\d+)", name)
    if not match:
        return None
    identifier = name[match.end():match.end() + int(match.group(1))]
    if not identifier.startswith(STUB_PREFIX) or len(identifier) != int(match.group(1)):
        return None
    return "_" + identifier[len(STUB_PREFIX):]


def classify_generated(name: str, record: dict, context: dict, shared: set, reached: set) -> dict | str | None:
    """G-1/G-2 (reviewer 109): nvcc-generated definitions of a non-GNU TU. None: not a generated name (the Q-3
    rules apply). A dict: the bounded class. A str: the FAIL reason (full nvcc provenance would be needed).
    G-1 binds the stub to this compile through the shared tmpxft id of its .stub.c declaration file and its
    unit's .cudafe1.cpp name (nvcc's CU DW_AT_name is that temporary, not the .cu) and the object's own hashed
    .cu; its kernel must be defined in this object and declared (Q-3) in a hashed non-audited file."""
    facts = record.get("definition_facts", {}).get(name, {})
    if name.startswith("__fatbinwrap_"):
        if name in shared:
            return "FAIL fatbin wrapper shared with another target object"
        if name in reached:
            return "FAIL fatbin wrapper referenced from the closure"
        if facts.get("type") != "object" or facts.get("section") != ".nvFatBinSegment":
            return f"FAIL {name} is a {facts.get('type')} in {facts.get('section')}, not an object in .nvFatBinSegment"
        return {"class": "nvcc-fatbin-wrapper", "section": ".nvFatBinSegment"}
    if STUB_PREFIX not in name:
        return None
    kernel_entry = record.get("stub_kernels", {}).get(name)
    if kernel_entry is None:
        return "FAIL a __device_stub__ name that does not spell its kernel"
    if name in shared:
        return "FAIL launch stub shared with a GNU translation unit"
    if name in reached:
        return "FAIL launch stub reached from the closure"
    own = record.get("own_source")
    if own is None or record["closure"].get(own, {}).get("kind") != "source":
        return "FAIL the object's own .cu source is not a hashed closure source"
    stem = Path(own).name[:-len(".cu")]
    declared = record.get("definition_declarations", {}).get(name, {})
    stub_file = TMPXFT_STUB.fullmatch(Path(str(declared.get("decl_file", ""))).name)
    units = [TMPXFT_CU.fullmatch(Path(unit).name) for unit in declared.get("cu_names", [])]
    if stub_file is None or stub_file.group(2) != stem:
        return f"FAIL stub declared in {declared.get('decl_file')!r}, not this object's tmpxft_*_{stem}.cudafe1.stub.c"
    if len(units) != 1 or units[0] is None or units[0].group(1) != stub_file.group(1) or units[0].group(2) != stem:
        return f"FAIL the stub's unit {declared.get('cu_names')} is not this object's own {stem} compile (same tmpxft id)"
    if not kernel_entry["defined_here"]:
        return f"FAIL kernel {kernel_entry['kernel']} is not defined in this object"
    kernel_declared = record.get("definition_declarations", {}).get(kernel_entry["kernel"], {})
    path = kernel_declared.get("decl_file")
    if path is None:
        return f"FAIL kernel {kernel_entry['kernel']} has no definition DIE ({kernel_declared.get('error')})"
    if path in context["audited_paths"]:
        return f"FAIL kernel declared in audited header {path}"
    if path not in record["closure"]:
        return f"FAIL kernel declared in {path}, outside the hashed closure"
    return {"class": "nvcc-generated-launch-stub", "kernel": kernel_entry["kernel"], "kernel_decl_file": path,
            "kernel_sha256": record["closure"][path]["sha256"], "stub_file": declared.get("decl_file"),
            "tmpxft_id": stub_file.group(1), "own_source": own}


def non_gnu_record(build, output, command, reference, context) -> dict:
    """N-1 (reviewer 91): a non-GNU TU (nvcc) is not replayed and never excluded. Its ELF is parsed into the
    fail-closed reference graph (the global closure must reach no function in it); its non-local definitions are
    listed with the audited headers in its recorded deps (hashed, closure-classified). The definitions shared
    with GNU target objects must be NON-selected in the link map (post-processing); a definition only this TU
    holds must be provably outside the audited headers: its unqualified name occurs as no identifier in any
    audited header of its deps (closure-bound text; token pasting in such a header fails). Status NON-GNU until
    run_b1() classifies it; anything selected, reached or unprovable FAILS (full nvcc provenance would be needed)."""
    if any(argument.startswith(("-flto", "-dlto")) for argument in command):
        raise Failure("LTO in the executed non-GNU compile (FATAL)")
    record = {"object": output, "executed_command": command, "object_sha256": sha256(reference),
              "compiler": Path(command[1] if Path(command[0]).name == "ccache" else command[0]).name}
    deps = ninja_deps(build, output)
    mounts = mount_points()
    record["closure"] = {path: closure_entry(path, build, context["src_root"], context["manifest"],
                                             context.get("repositories"), mounts) for path in deps}
    audited = sorted(path for path in deps if path in context["audited_paths"])
    record["audited_deps"] = audited
    elf = elf_tools.read_elf(reference)
    graph = elf_tools.function_graph(elf)
    if graph["failures"]:
        raise Failure(f"the non-GNU object's reference graph is not closed ({len(graph['failures'])} failures): "
                      f"{graph['failures'][:3]}")
    definitions = {symbol["name"]: symbol["binding"] for symbol in elf["symbols"]
                   if symbol["binding"] != "local" and 0 < symbol["shndx"] < elf_tools.SHN_LORESERVE
                   and symbol["type"] in ("func", "ifunc", "object", "tls")}  # Q-2d: TLS data too.
    record["definitions"] = definitions
    record["definition_facts"] = {symbol["name"]: {"type": symbol["type"], "binding": symbol["binding"],
                                                   "section": elf["sections"][symbol["shndx"]]["name"]}
                                  for symbol in elf["symbols"] if symbol["name"] in definitions
                                  and 0 < symbol["shndx"] < elf_tools.SHN_LORESERVE}
    defined_functions = {symbol["name"] for symbol in elf["symbols"] if symbol["type"] in ("func", "ifunc")
                         and 0 < symbol["shndx"] < elf_tools.SHN_LORESERVE}
    kernels = {name: stub_kernel(name) for name in definitions if stub_kernel(name)}
    record["stub_kernels"] = {stub: {"kernel": kernel, "defined_here": kernel in defined_functions}
                              for stub, kernel in kernels.items()}
    sources = [argument for argument in command if argument.endswith(".cu")]
    record["own_source"] = os.path.realpath(os.path.join(build, sources[-1])) if len(sources) == 1 else None
    texts = {}
    for path in audited:
        text = Path(path).read_text(errors="replace")
        if "##" in text:
            raise Failure(f"audited header {path} of a non-GNU TU uses token pasting: definitions not provable")
        texts[path] = set(IDENTIFIER.findall(text))
    names = sorted(definitions)
    result = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True)
    spec = importlib.util.spec_from_file_location("dwarf_sites", Path(__file__).resolve().with_name("dwarf_sites.py"))
    dwarf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dwarf)
    try:
        record["definition_declarations"] = dwarf.definition_declarations(
            dwarf.read_elf(reference.read_bytes()), set(definitions) | {k["kernel"] for k in record["stub_kernels"].values()})
    except dwarf.DwarfError as error:
        record["definition_declarations"] = {name: {"error": f"DWARF unreadable: {error}"} for name in definitions}
    record["identifiers"] = {name: function_identifier(plain)
                             for name, plain in zip(names, result.stdout.splitlines())}
    # "" = provably no source definition (not an identifier); None = undeterminable (fails).
    record["header_identifiers_in"] = {name: sorted(path for path, ids in texts.items() if identifier in ids)
                                       for name, identifier in record["identifiers"].items() if identifier}
    graph_path = Path(context["evidence"]) / f"{output.replace('/', '__')}.graph.json"
    graph_path.write_text(json.dumps({"graph": graph, "function_files": {}, "outside_files": {}}, sort_keys=True) + "\n")
    record["graph"], record["graph_sha256"] = graph_path.name, sha256(graph_path)
    record["mode"] = "non-gnu"
    record["status"] = "NON-GNU"
    return record


def replay_tu(build, output, evidence, context, build_object=None, cc1_extra_augment=("-g",)):
    """One TU (raises Failure)."""
    name = output.replace("/", "__")
    record = {"object": output}
    reference = Path(build_object) if build_object else Path(build) / output
    if not reference.is_file():
        record["status"] = "MISSING"
        return record
    record["object_sha256"] = reference_sha = sha256(reference)
    command = executed_command(build, output)
    record["executed_command"] = command
    if any(argument.startswith("-flto") for argument in command):
        raise Failure("LTO in the executed compile (FATAL: neither the assembly nor the object is final code)")
    launcher = Path(command[0]).name == "ccache"
    compiler_argv = command[1:] if launcher else command
    if Path(compiler_argv[0]).name not in ("g++", "g++-14", "c++") and not Path(compiler_argv[0]).name.startswith("g++"):
        return non_gnu_record(build, output, command, reference, context)
    if option_value(compiler_argv, "-o") != output:
        raise Failure("the executed command's -o is not the object")
    depfile = option_value(compiler_argv, "-MF")
    route_object, route_dep = str(evidence / f"{name}.route.o"), str(evidence / f"{name}.route.d")
    redirected = list(compiler_argv)
    redirected[compiler_argv.index("-o") + 1] = route_object
    redirected[compiler_argv.index("-MF") + 1] = route_dep
    recorded_deps = ninja_deps(build, output)
    before = {path: sha256(path) for path in recorded_deps}
    if launcher:
        cache = tempfile.mkdtemp(prefix=f"ccache-{name}-", dir=evidence)
        env = dict(os.environ, CCACHE_DIR=cache, CCACHE_DEBUG="1")
        routed = run([command[0]] + redirected, build, env)
        if routed.returncode:
            raise Failure(f"the launcher route compile failed: {routed.stderr[-1500:]}")
        logs = sorted(evidence.glob(f"{name}.route.o.*ccache-log"))
        if len(logs) != 1:
            raise Failure(f"expected one ccache debug log for the route compile, found {len(logs)}")
        actual = executing_argv(logs[0])
        record["launcher"] = {"program": program_identity(command[0]), "log": logs[0].name, "log_sha256": sha256(logs[0])}
        shutil.rmtree(cache)
    else:
        routed = run(redirected, build)
        if routed.returncode:
            raise Failure(f"the route compile failed: {routed.stderr[-1500:]}")
        actual = redirected
    if any(argument.startswith("-flto") for argument in actual):
        raise Failure("LTO in the executed compiler argv (FATAL)")
    record["route_object_sha256"] = sha256(route_object)
    if record["route_object_sha256"] != reference_sha:
        raise Failure("the launcher route does not reproduce the build's object (the executed compile is unknown)")
    record["compiler_argv"] = actual
    record["driver"] = program_identity(actual[0] if os.path.isabs(actual[0]) else shutil.which(actual[0]))
    jobs, temporary, programs = gnu_plan(actual, build)
    record["plan"], record["programs"] = jobs, programs
    assembly = evidence / f"{name}.s"
    replay_object, replay_dep = str(evidence / f"{name}.replay.o"), str(evidence / f"{name}.replay.d")
    replay(jobs, temporary, {temporary: str(assembly), route_object: replay_object, route_dep: replay_dep}, build)
    if not assembly.is_file() or not Path(replay_object).is_file():
        raise Failure("the replay wrote no assembly or no object")
    record["assembly"], record["assembly_sha256"] = assembly.name, sha256(assembly)
    record["replay_object_sha256"] = sha256(replay_object)
    if record["replay_object_sha256"] != reference_sha:
        raise Failure("the replayed plan does not reproduce the build's object: the retained assembly is not its "
                      "assembler input")
    analysed = assembly
    mode = debug_mode(actual)
    if mode == "augment":
        augmented = evidence / f"{name}.aug.s"
        augmented_object, augmented_dep = str(evidence / f"{name}.aug.o"), str(evidence / f"{name}.aug.d")
        replay(jobs, temporary, {temporary: str(augmented), route_object: augmented_object, route_dep: augmented_dep},
               build, cc1_extra_augment)
        differences, excluded = elf_tools.compare_nondebug_detail(reference, augmented_object)
        record["debug_augmented"] = {"cc1plus_extra": list(cc1_extra_augment), "assembly": augmented.name,
                                     "assembly_sha256": sha256(augmented), "object_sha256": sha256(augmented_object),
                                     "nondebug_differences": differences[:20],
                                     "excluded_debug_only_merge_string_labels": excluded}
        if differences:
            raise Failure(f"the debug-augmented object differs from the build object outside debug sections: "
                          f"{differences[:3]}")
        record["debug_augmented"]["object"] = Path(augmented_object).name  # Retained: Release DWARF (D-1.6).
        analysed = augmented
        record["mode"] = "release-debug-augmented"
    else:
        record["mode"] = "integration-native-debug"
    replayed_deps = depfile_paths(replay_dep, build)
    if replayed_deps != recorded_deps:
        raise Failure(f"the replay's dependency closure differs from ninja's recorded deps: "
                      f"{sorted(set(recorded_deps) ^ set(replayed_deps))[:4]}")
    after = {path: sha256(path) for path in recorded_deps}
    if after != before:
        raise Failure(f"closure inputs changed during the replay: {sorted(p for p in before if before[p] != after.get(p))[:3]}")
    mounts = mount_points()
    record["closure"] = {path: closure_entry(path, build, context["src_root"], context["manifest"],
                                             context.get("repositories"), mounts)
                         for path in recorded_deps}
    record["closure_sha256"] = hashlib.sha256(json.dumps(sorted(before.items())).encode()).hexdigest()
    record["working_directory"] = build
    elf = elf_tools.read_elf(reference)
    graph = elf_tools.function_graph(elf)
    if graph["failures"]:
        raise Failure(f"the object's reference graph is not closed ({len(graph['failures'])} failures): "
                      f"{graph['failures'][:3]}")
    crosscheck = assembly_crosscheck(analysed, graph)
    record["assembly_crosscheck"] = {"checked": crosscheck["checked"], "failures": crosscheck["failures"][:50],
                                     "decoded_edges": graph["decoded_edges"]}
    if crosscheck["failures"]:
        raise Failure(f"the analysed assembly names {len(crosscheck['failures'])} reference(s) without a graph edge "
                      f"(C-5d): {crosscheck['failures'][:3]}")
    function_names = {name for node in graph["functions"].values() for name in node["names"]}
    parsed = parse_assembly(analysed, function_names, build)
    program_symbols = sorted(name for name in function_names if PROGRAMS.search(name))
    record["classification"] = classify(parsed, context["audited_paths"], program_symbols)
    record["analysed_assembly"] = {
        "file": analysed.name, "sha256": sha256(analysed),
        "debug": ({"mode": "release-debug-augmented", "cc1plus_extra": list(cc1_extra_augment),
                   "note": "the .loc lines come from the augmentation; the code equals the build object's"}
                  if mode == "augment" else
                  {"mode": "integration-native-debug",
                   "executed_options": [a for a in actual[1:] if a.startswith("-g")]})}
    if program_symbols:
        record["programs_defined"] = program_symbols
    record["defined_nonlocal"] = sorted({symbol["name"] for symbol in elf["symbols"] if symbol["binding"] != "local"
                                         and 0 < symbol["shndx"] < elf_tools.SHN_LORESERVE})
    record["helper_bindings"] = {symbol["name"]: symbol["binding"] for symbol in elf["symbols"]
                                 if symbol["name"] in set(record["classification"]["helpers"])}
    names_to_key = {name: key for key, node in graph["functions"].items() for name in node["names"]}
    function_files = {}
    for function_name, entry in parsed["functions"].items():
        key = names_to_key[function_name]
        merged = function_files.setdefault(key, {})
        for path, count in entry["files"].items():
            merged[path] = merged.get(path, 0) + count
    graph_document = {"graph": graph, "function_files": function_files, "outside_files": parsed["outside"]}
    graph_path = evidence / f"{name}.graph.json"
    graph_path.write_text(json.dumps(graph_document, sort_keys=True) + "\n")
    record["graph"], record["graph_sha256"] = graph_path.name, sha256(graph_path)
    os.remove(replay_object)
    os.remove(route_object)
    record["status"] = "PASS"
    return record


def selected_objects(build, targets, audited_paths):
    inputs = sorted({line.strip() for line in ninja(build, "-t", "inputs", *targets).splitlines()
                     if line.strip().endswith(".o")})
    selected, missing, unrecorded = [], [], []
    for output in inputs:
        if not (Path(build) / output).is_file():
            missing.append(output)
            continue
        try:
            deps = ninja_deps(build, output)
        except Failure:
            unrecorded.append(output)
            continue
        if audited_paths & set(deps):
            selected.append(output)
    return inputs, selected, missing, unrecorded


def worker(arguments):
    build, output, evidence, context = arguments
    try:
        return replay_tu(build, output, Path(evidence), dict(context, evidence=evidence))
    except (Failure, OSError, elf_tools.ElfError) as error:
        return {"object": output, "status": "FAIL", "reason": str(error)}


def symbol_index(build, objects):
    """({non-local defined name: {objects}}, [(object, program function name)]) over the given objects."""
    definitions, roots = {}, []
    for output in objects:
        for symbol in elf_tools.read_symbols(Path(build) / output):
            if not (0 < symbol["shndx"] < elf_tools.SHN_LORESERVE or symbol["shndx"] == elf_tools.SHN_COMMON):
                continue
            if symbol["binding"] != "local":
                definitions.setdefault(symbol["name"], set()).add(output)
            if symbol["type"] in ("func", "ifunc") and PROGRAMS.search(symbol["name"]):
                roots.append((output, symbol["name"]))
    return definitions, sorted(set(roots))


def global_closure(records, evidence: Path, definitions, roots, context, build) -> dict:
    """C-1/C-2: the reference closure of every program over all target objects. A non-local name reaches EVERY
    non-local definition of it (all weak/COMDAT copies); a name no target object defines is an import (external).
    Objects reached but not yet replayed are returned as "needed" (the caller replays them and recomputes).
    Then every closure function's .loc files are classified: audited, system (image /usr), external (fetched
    checkout); a project source or generated file not in the audited list, an unclassified file, or a closure
    function without any .loc FAILS, naming it."""
    by_object = {record["object"]: record for record in records}
    documents = {}
    def document(obj):
        if obj not in documents:
            documents[obj] = json.loads((evidence / by_object[obj]["graph"]).read_text())
        return documents[obj]
    needed, failures, imports, seen, work = set(), [], {}, set(), []
    shared_reached: dict[str, set] = {}
    non_gnu_reached: dict[str, set] = {}

    def reach(obj, name):
        record = by_object.get(obj)
        if record is None:
            needed.add(obj)
            return
        if record["status"] != "PASS":
            if record["status"] == "NON-GNU":
                non_gnu_reached.setdefault(obj, set()).add(name)
            failures.append(f"the closure reaches {name} in {obj}, whose B1 is {record['status']}")
            return
        graph = document(obj)["graph"]
        key = graph["exports"].get(name) or next((k for k, node in graph["functions"].items()
                                                  if name in node["names"]), None)
        if key is None:
            failures.append(f"{obj} defines {name} but its graph has no node for it")
            return
        work.append((obj, key))

    for obj, name in roots:
        reach(obj, name)
    while work:
        obj, key = work.pop()
        if (obj, key) in seen:
            continue
        seen.add((obj, key))
        for _, reference, target in document(obj)["graph"]["edges"].get(key, []):
            if reference == "node":
                work.append((obj, target))
                continue
            owners = definitions.get(target)
            if not owners:
                imports.setdefault(target, set()).add(f"{obj}::{key}")
            owners = sorted(owners or ())
            non_gnu = [owner for owner in owners if by_object.get(owner, {}).get("status") == "NON-GNU"]
            if non_gnu and len(non_gnu) < len(owners):
                # N-1: a name with GNU copies executes the map-selected copy; the non-GNU copy must be proven
                # NON-selected by post-processing (non_gnu_selection), so only the GNU copies are audited here.
                for owner in non_gnu:
                    shared_reached.setdefault(target, set()).add(owner)
                owners = [owner for owner in owners if owner not in non_gnu]
            for owner in owners:
                reach(owner, target)
    if needed:
        return {"needed": sorted(needed)}
    mounts = mount_points()
    audited = context["audited_paths"]
    functions, indirect, nonlocal_definitions, no_loc = {}, {}, {}, []
    loc = {"audited": {}, "system": {}, "external": {}}
    unaudited, unclassified = {}, {}
    nonlocal_data: dict[str, set] = {}
    for obj, key in sorted(seen):
        graph = document(obj)["graph"]
        node = graph["functions"].get(key)
        if node is None:
            # A data or section node the closure code references (Q-2, reviewer 109): its non-local names are
            # audited data, under the same B1-PASS GNU selection rule as closure functions.
            for name, export in graph["exports"].items():
                if export == key:
                    nonlocal_data.setdefault(name, set()).add(obj)
            continue
        label = f"{obj}::{key}"
        functions[label] = {"names": node["names"], "section": node["section"]}
        for name, binding in node["names"].items():
            if binding != "local":
                nonlocal_definitions.setdefault(name, set()).add(obj)
        if key in graph["indirect"]:
            indirect[label] = graph["indirect"][key]
        files = document(obj)["function_files"].get(key, {})
        if not files:
            no_loc.append(label)
        for path in files:
            if path in audited:
                bucket = loc["audited"]
            else:
                kind = path_kind(path, build, context["src_root"], context["repositories"], context["manifest"], mounts)
                bucket = (loc["system"] if kind == "system" else loc["external"] if kind.startswith("external-")
                          else unaudited if kind in ("source", "generated") else unclassified)
            bucket.setdefault(path, []).append(label)
    for path, labels in sorted(unaudited.items()):
        failures.append(f"project file {os.path.relpath(path, context['src_root'])} has .loc lines in closure "
                        f"function(s) {labels[:2]} but is not in the audited list")
    for path, labels in sorted(unclassified.items()):
        failures.append(f"unclassified .loc file {path} in closure function(s) {labels[:2]}")
    for label in no_loc:
        failures.append(f"closure function {label} has no .loc line (its source cannot be audited)")
    def counted(bucket):
        return {path: len(labels) for path, labels in sorted(bucket.items())}
    return {"status": "FAIL" if failures else "PASS", "failures": failures[:200], "failure_count": len(failures),
            "roots": [f"{obj}::{name}" for obj, name in roots], "functions": functions,
            "nonlocal_definitions": {name: sorted(objs) for name, objs in sorted(nonlocal_definitions.items())},
            "nonlocal_data": {name: sorted(objs) for name, objs in sorted(nonlocal_data.items())},
            "imports": {name: sorted(refs)[:5] for name, refs in sorted(imports.items())},
            "indirect_branches": indirect,
            "non_gnu_shared_reached": {name: sorted(objs) for name, objs in sorted(shared_reached.items())},
            "non_gnu_reached": {obj: sorted(names) for obj, names in sorted(non_gnu_reached.items())},
            "indirect_branches_note": "listed per closure function, not resolved; the checker's call model is the "
                                      "authority for indirect targets",
            "loc_files": {kind: counted(bucket) for kind, bucket in loc.items()},
            "unaudited_project_files": sorted(os.path.relpath(path, context["src_root"]) for path in unaudited),
            "unclassified_files": sorted(unclassified)}


def finalize_non_gnu(records, closure, definitions, unrecorded, context, build):
    """N-1/N-2, Q-2/Q-3, G-1/G-2 classification of the non-GNU and no-deps objects after the closure fixpoint
    (no compile: re-derivable from retained B1 evidence). Returns (no-deps records, failing no-deps objects)."""
    gnu_definitions = {}
    for record in records:
        if record["status"] == "PASS":
            for name in record.get("defined_nonlocal", []):
                gnu_definitions.setdefault(name, []).append(record["object"])
    audited_names = {name for label, node in closure.get("functions", {}).items() for name in node["names"]}
    audited_names |= set(closure.get("nonlocal_data", {}))
    for record in records:
        audited_names |= set(record.get("classification", {}).get("helpers", []))
    for record in records:
        if record["status"] != "NON-GNU":
            continue
        # N-1: reached functions already failed the closure (status NON-GNU is not PASS).
        shared = sorted(name for name in record["definitions"] if name in definitions and
                        set(definitions[name]) - {record["object"]})
        exclusive = sorted(set(record["definitions"]) - set(shared))
        # Q-3 (reviewer 109): an exclusive definition is outside the audited headers when its definition DIE in the
        # object's own DWARF is declared in a closure-bound (hashed) file that is not an audited header; one
        # declared in an audited header needs full provenance (FAIL). Without such a DIE the identifier rule is the
        # fail-closed default.
        declarations = record.get("definition_declarations", {})
        unprovable, record["exclusive_classified"] = [], {}
        reached = set(closure.get("non_gnu_reached", {}).get(record["object"], []))
        for name in shared:
            if name.startswith("__fatbinwrap_") or STUB_PREFIX in name:
                unprovable.append(name)  # G-1/G-2: a generated name shared with another object FAILs.
                record["exclusive_classified"][name] = "FAIL nvcc-generated name shared with another target object"
        for name in exclusive:
            generated = classify_generated(name, record, context, set(shared), reached)
            if generated is not None:
                record["exclusive_classified"][name] = generated
                if isinstance(generated, str):
                    unprovable.append(name)
                continue
            declared = declarations.get(name, {})
            path = declared.get("decl_file")
            if path is not None:
                closure_entry_ = record["closure"].get(path)
                if path in context["audited_paths"]:
                    unprovable.append(name)
                    record["exclusive_classified"][name] = f"FAIL declared in audited header {path}"
                elif closure_entry_ is None:
                    unprovable.append(name)
                    record["exclusive_classified"][name] = f"FAIL declared in {path}, outside the hashed closure"
                else:
                    record["exclusive_classified"][name] = {"decl_file": path, "sha256": closure_entry_["sha256"],
                                                            "kind": closure_entry_["kind"], "rule": "dwarf"}
            elif record["identifiers"].get(name) is None or record["header_identifiers_in"].get(name):
                unprovable.append(name)
                record["exclusive_classified"][name] = f"FAIL no definition DIE ({declared.get('error')}); identifier rule"
            else:
                record["exclusive_classified"][name] = {"rule": "identifier", "dwarf": declared.get("error")}
        unprovable += [name for name in exclusive if name in audited_names]
        record["shared_definitions"] = {name: sorted(set(definitions[name]) - {record["object"]}) for name in shared}
        # The shared definitions that can come from an audited header: an audited GNU name (closure function or
        # helper copy), a name whose identifier occurs in an audited header of the TU's deps, or one whose
        # identifier is undeterminable. Only these must be proven non-selected (post-processing); a shared name
        # spelled in no audited header cannot be defined there (the same identifier rule as for exclusive ones).
        # Q-2 (reviewer 109): exactly the audited names - closure functions, helper copies and every data symbol
        # closure code references - stay under the B1-PASS GNU selection rule; other shared names are recorded.
        record["audited_shared"] = sorted(name for name in shared if name in audited_names)
        record["shared_recorded"] = sorted(name for name in shared if name not in audited_names)
        record["exclusive_definitions"] = exclusive
        if unprovable:
            record["status"] = "FAIL"
            record["reason"] = (f"non-GNU TU holds definitions not provably outside the audited headers (full nvcc "
                                f"provenance needed): {sorted(set(unprovable))[:5]}")
        else:
            record["status"] = "PASS"
            record["classification"] = {"classes": ["non-gnu-outside-closure"], "helpers": [], "inlined_into": [],
                                        "audited_locs": None}
            record["requires_map"] = "shared definitions non-selected; selected copy from a B1-PASS GNU TU"
    no_deps = []
    for output in unrecorded:
        # N-2: an object without recorded deps (cmake_device_link.o): no closure node (a reached name would have
        # been replayed and failed) and no audited name defined.
        names = {symbol["name"] for symbol in elf_tools.read_symbols(Path(build) / output)
                 if symbol["binding"] != "local" and 0 < symbol["shndx"] < elf_tools.SHN_LORESERVE}
        clash = sorted(names & audited_names)
        no_deps.append({"object": output, "object_sha256": sha256(Path(build) / output), "definitions": len(names),
                        "status": "FAIL" if clash else "PASS", "class": "no-deps-outside-closure",
                        **({"reason": f"defines audited names {clash[:5]}"} if clash else {})})
    unrecorded = [entry["object"] for entry in no_deps if entry["status"] != "PASS"]
    return no_deps, unrecorded


def run_b1(build, targets, evidence: Path, context, jobs, objects=None) -> dict:
    """Selection (audited deps) + replay, then the closure fixpoint: objects the closure reaches are replayed too."""
    missing, unrecorded = [], []
    if objects is None:
        inputs, objects, missing, unrecorded = selected_objects(build, targets, context["audited_paths"])
    else:
        inputs = list(objects)
    definitions, roots = symbol_index(build, [output for output in inputs if (Path(build) / output).is_file()])
    selected_by = {output: "audited-deps" for output in objects}
    records, pending, rounds, closure = [], list(objects), 0, {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        while True:
            rounds += 1
            records += list(pool.map(worker, [(build, output, str(evidence), context) for output in pending]))
            closure = global_closure(records, evidence, definitions, roots, context, build)
            pending = closure.get("needed", [])
            if not pending:
                break
            for output in pending:
                selected_by[output] = "closure"
    for record in records:
        record["selected_by"] = selected_by[record["object"]]
    no_deps, unrecorded = finalize_non_gnu(records, closure, definitions, unrecorded, context, build)
    failed = sum(record["status"] != "PASS" for record in records)
    classes = {}
    for record in records:
        for name in record.get("classification", {}).get("classes", []):
            classes[name] = classes.get(name, 0) + 1
    ok = records and not failed and not missing and not unrecorded and closure["status"] == "PASS"
    return {"targets": targets, "build": build, "target_input_objects": len(inputs), "selected": len(records),
            "selected_by_closure": sorted(o for o, why in selected_by.items() if why == "closure"),
            "closure_rounds": rounds, "passed": len(records) - failed, "failed_or_missing": failed,
            "missing_target_inputs": missing, "inputs_without_recorded_deps": unrecorded,
            "no_deps_objects": no_deps,
            "closure_digest": hashlib.sha256(json.dumps(closure, sort_keys=True).encode()).hexdigest(),
            "audited_headers": context["audited_headers"],
            "external_repositories": {root: {k: v for k, v in info.items() if k != "blobs"}
                                      for root, info in context["repositories"].items()},
            "audited_headers_list_sha256": context["audited_headers_list_sha256"], "classes": classes,
            "closure": closure, "tus": records, "status": "PASS" if ok else "FAIL"}


def load_context(args):
    manifest = {}
    for line in Path(args.manifest).read_text().splitlines():
        meta, _, path = line.partition("\t")
        mode, kind, blob = meta.split()
        if kind == "blob":
            manifest[path] = blob
    headers = [line.strip() for line in Path(args.audited_headers).read_text().splitlines() if line.strip()]
    audited_paths = {os.path.realpath(os.path.join(args.src_root, header)) for header in headers}
    present = {header: sha256(os.path.join(args.src_root, header)) for header in headers
               if os.path.isfile(os.path.join(args.src_root, header))}
    return {"manifest": manifest, "audited_paths": audited_paths, "src_root": args.src_root,
            "repositories": external_repositories(args.src_root),
            "audited_headers": present, "audited_headers_list_sha256": sha256(args.audited_headers)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--object", action="append", default=[], help="explicit object(s) (controls)")
    parser.add_argument("--manifest", required=True, help="git ls-tree -r of the snapshot commit")
    parser.add_argument("--identity", required=True, help="the snapshot's commit/tree identity file")
    parser.add_argument("--audited-headers", required=True)
    parser.add_argument("--src-root", default="/src")
    parser.add_argument("--image-id", default="")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()
    evidence = Path(args.evidence)
    evidence.mkdir(parents=True, exist_ok=False)
    context = load_context(args)
    summary = run_b1(args.build, args.target, evidence, context, args.jobs, args.object or None)
    summary.update(image_id=args.image_id, identity=Path(args.identity).read_text())
    for record in summary["tus"]:
        print(f"{record['status']}: {record['object']}" + (f": {record['reason']}" if record.get("reason") else ""))
    for failure in summary["closure"].get("failures", []):
        print(f"CLOSURE FAIL: {failure}")
    (evidence / "b1-summary.json").write_text(json.dumps(summary, sort_keys=True, indent=1) + "\n")
    print(f"B1 {summary['status']}: {summary['passed']}/{summary['selected']} TUs "
          f"({len(summary['selected_by_closure'])} added by the closure in {summary['closure_rounds']} rounds); "
          f"classes {summary['classes']}; closure {summary['closure']['status']} "
          f"({len(summary['closure'].get('functions', {}))} functions, {len(summary['closure'].get('imports', {}))} "
          f"imports, unaudited project files {summary['closure'].get('unaudited_project_files', [])}); missing "
          f"target inputs {len(summary['missing_target_inputs'])}; inputs without recorded deps "
          f"{len(summary['inputs_without_recorded_deps'])}")
    raise SystemExit(0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
