#!/usr/bin/env python3
"""Compare a candidate stack-traffic table with the develop baseline table: no new hot stack traffic.

Kept apart from the counting analysis (CPUNativeTileRegisterEvidence.py) and the table assembly
(CPUNativeTileStackTable.py) so that edits of the comparison never churn the baseline's bound hashes.

Rules (every class separately, never summed; classes: CLASSES below):
- Key map (CPUNativeTileStackKeyMap.json): a candidate program <ENV>/<rows:leaf:policy:tail> whose tail is
  `same` compares with the develop program of the same key; a `new` tail compares with its named develop
  analogue; any other tail fails (unmapped). A missing develop program fails.
- Existing keys: when both programs have the same loop structure (identical multiset of (role, parent role)
  pairs over their indexed roles) and neither has sibling copies, every class at every role (entry level
  included) must be <= develop's (static, per role). Otherwise every class must be <= develop's per-call dynamic
  count at every reference geometry (the candidate table is built with the candidate-max sibling rule, the
  baseline with develop-min, so the comparison is strict).
- New keys: every class must be zero at the K-walk roles (block, superblock, group) and the enclosing levels
  (entry, column, partition, strip, ...) must be <= the analogue: static per enclosing role when both enclosing
  structures match, else dynamic per geometry with the K-walk contributions of the analogue kept (the
  candidate's are zero by the first rule).
- Compact programs (P4, compare_compact()): a candidate compact:C:W:R:T:O:D@copy compares with every develop
  path of the formats the key map's "compact" entry lists, in the same copy (the same TU's local copy or the
  same link-selected copy; never another copy, never a maximum over copies): develop's policy-free compact
  program of the same Reduction enumerator (each table's source-derived values) or, for a format moved from a
  develop tile path, that develop tile program (dynamically per geometry). An undispatched codebook needs a P2
  unreachability record; a program without a copy identity fails.
A failure is reported per key with the class, the level (role or geometry) and both values. A false failure of
the strict sibling rule may only be refined by recorded CFG ownership (not implemented: no such key exists).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

CLASSES = ("fp", "scalar", "abi_loop", "alias_load", "alias_store", "invariant_load", "invariant_reloc",
           "tls_call", "call_frame", "int_call", "ext_call", "ext_bytes", "ext_accesses")
K_WALK = frozenset({"block", "superblock", "group"})


HEADERS = ("src/v2/kernels/cpu/gemm/CPUNativeQuantisedEncoding.h",
           "src/v2/kernels/cpu/gemm/CPUNativeQuantisedPreparedFootprint.h", "src/v2/tensors/NativeQuantisedFormatInfo.h")
TILE_SOURCE = "src/v2/kernels/cpu/gemm/CPUNativeNEONPreparedProjection.h"
DISPATCH_SIGNATURE = "int Strips = 0> void dispatch() const noexcept"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dispatch_text(source: str) -> str:
    """The text of the tile's dispatch() (signature through its closing brace): the transcribed rules' authority."""
    if source.count(DISPATCH_SIGNATURE) != 1:
        raise ValueError("the tile source has no unique dispatch()")
    start = source.index(DISPATCH_SIGNATURE)
    position = source.index("{", start)
    depth, position = 1, position + 1
    while depth:
        depth += {"{": 1, "}": -1}.get(source[position], 0)
        position += 1
    return source[start:position]


def tail_of(rules: list[dict], encoding: int, codebook: int, asymmetric: int) -> str:
    """A tree's dispatch(), transcribed as ordered (condition over E, C, A -> key tail) rules."""
    for rule in rules:
        if eval(rule["when"], {"__builtins__": {}}, {"E": encoding, "C": codebook, "A": asymmetric}):
            return rule["tail"]
    raise ValueError(f"dispatch rules select nothing for encoding {encoding} codebook {codebook}")


def parse_formats(text: str) -> dict[str, dict]:
    """Output of CPUNativeTileStackFormats.cpp: name codebook superblock encoding asymmetric per line."""
    formats = {}
    for line in text.strip().splitlines():
        name, codebook, superblock, encoding, asymmetric = line.split()
        formats[name] = {"codebook": int(codebook), "superblock": int(superblock), "encoding": int(encoding),
                         "asymmetric": int(asymmetric)}
    return formats


def derive_keys(keymap: dict) -> tuple[dict, dict]:
    """Per source format the develop and candidate key tails, and per candidate tile tail its formats and the
    develop tails those formats ran on - derived mechanically from the recorded per-tree encodings and the
    transcribed dispatch rules. Non-tile tails (compact, coded:N) are kept per format but are no candidate key."""
    develop, candidate = keymap["trees"]["develop"], keymap["trees"]["candidate"]
    if set(develop["formats"]) != set(candidate["formats"]):
        raise ValueError("the trees' format lists differ")
    formats, keys = {}, {}
    for name in sorted(candidate["formats"]):
        before, after = develop["formats"][name], candidate["formats"][name]
        if (before["codebook"], before["superblock"]) != (after["codebook"], after["superblock"]):
            raise ValueError(f"format {name} changed identity between the trees")
        old = tail_of(develop["dispatch"], before["encoding"], before["codebook"], before["asymmetric"])
        new = tail_of(candidate["dispatch"], after["encoding"], after["codebook"], after["asymmetric"])
        formats[name] = {"develop": old, "candidate": new}
        if TAIL.match(new):
            entry = keys.setdefault(new, {"formats": [], "develop": []})
            entry["formats"].append(name)
            if old not in entry["develop"]:
                entry["develop"].append(old)
    for entry in keys.values():
        entry["develop"] = sorted(entry["develop"])
    return formats, keys


TAIL = re.compile(r"^[0-9]:(?:[0-3]|7|11):[01]$")
COMPACT_TAIL = "compact"


def compact_tail(tail: str, codebook: int) -> str:
    """A dispatch tail with the compact implementation named by its program key prefix: rowsProgram is
    instantiated with the source codebook as its Codebook argument, so the shared compact tail of a format is
    compact:<codebook> (the checker's compact_program_key() prefix); every other tail is unchanged."""
    return f"{COMPACT_TAIL}:{codebook}" if tail == COMPACT_TAIL else tail


def derive_compact(keymap: dict) -> dict:
    """Per candidate compact program key prefix (compact:<codebook>), its formats and the develop tails those
    formats ran on (a develop compact tail as compact:<codebook>) - mechanically, as derive_keys(). The compact
    programs of each key prefix (lanes, rows, tree, ordered, dot) are compared per TU copy with the develop copies
    of these tails (step 2 of the call model; P4)."""
    develop, candidate = keymap["trees"]["develop"], keymap["trees"]["candidate"]
    keys: dict[str, dict] = {}
    for name in sorted(candidate["formats"]):
        before, after = develop["formats"][name], candidate["formats"][name]
        new = tail_of(candidate["dispatch"], after["encoding"], after["codebook"], after["asymmetric"])
        if new != COMPACT_TAIL:
            continue
        old = compact_tail(tail_of(develop["dispatch"], before["encoding"], before["codebook"], before["asymmetric"]),
                           before["codebook"])
        entry = keys.setdefault(compact_tail(new, after["codebook"]), {"formats": [], "develop": []})
        entry["formats"].append(name)
        if old not in entry["develop"]:
            entry["develop"].append(old)
    for entry in keys.values():
        entry["develop"] = sorted(entry["develop"])
    return keys


def check_keymap(keymap: dict, candidate_inputs: dict | None = None, candidate_source: str | None = None) -> None:
    """The key map is internally consistent: its formats and keys are those derive_keys() computes, every
    candidate key's develop tails are tile tails (or the key has no develop path, recorded as such). With a
    candidate (gate run), the recorded candidate header hashes must be the candidate compile's inputs and the
    recorded dispatch hash the candidate tile source's dispatch()."""
    formats, keys = derive_keys(keymap)
    if keymap.get("formats") != formats or keymap.get("keys") != keys or keymap.get("compact") != derive_compact(keymap):
        raise ValueError("key map formats/keys/compact keys are not the ones derived from the recorded encodings and "
                         "dispatch")
    for tree in ("develop", "candidate"):
        entry = keymap["trees"][tree]
        if set(entry["headers"]) != set(HEADERS) or not re.fullmatch(r"[0-9a-f]{64}", entry["dispatch_sha256"]):
            raise ValueError(f"key map {tree} entry lacks its header or dispatch hashes")
    check_leaf(keymap["trees"]["develop"].get("leaf"))
    if candidate_inputs is not None:
        for path in HEADERS:
            if candidate_inputs.get(path) != keymap["trees"]["candidate"]["headers"][path]:
                raise ValueError(f"key map was derived from another version of {path}")
    if candidate_source is not None and \
            digest(dispatch_text(candidate_source).encode()) != keymap["trees"]["candidate"]["dispatch_sha256"]:
        raise ValueError("the candidate dispatch() differs from the one the key map transcribes")


def structure(program: dict, roles: frozenset | None = None) -> list[tuple[str, str]]:
    names = {loop["id"]: loop["role"] for loop in program["loops"]}
    names[0] = "entry"
    return sorted((loop["role"], names[loop["parent"]]) for loop in program["loops"]
                  if not loop["role"].startswith("empty#")  # C1: empty self-loops carry no role and zero counts
                  and (roles is None or loop["role"].split("#")[0] in roles))


def base(role: str) -> str:
    return role.split("#")[0]


def subtrees(program: dict) -> dict[str, set[str]]:
    """Per indexed role node (and "entry"), the role names of its subtree: itself and every deeper loop."""
    parents = {loop["role"]: next((other["role"] for other in program["loops"] if other["id"] == loop["parent"]),
                                  "entry") for loop in program["loops"]}
    result = {"entry": {"entry"} | set(parents)}
    for role in parents:
        members, frontier = {role}, [role]
        while frontier:
            node = frontier.pop()
            for child, parent in parents.items():
                if parent == node and child not in members:
                    members.add(child)
                    frontier.append(child)
        result[role] = members
    return result


def dominance(candidate: dict, baseline: dict, skip: frozenset = frozenset()) -> list[str]:
    """Subtree dominance (matching structures): for every class and every role node r, the candidate's count
    summed over r's subtree is <= develop's. With every trip >= 1 the per-call multiplier of a loop is
    non-decreasing towards the leaves, so by Abel summation this implies candidate <= develop dynamically at
    every geometry: accesses may move outward (hoisting), never inward, and an inner increase is never paid by an
    outer decrease. Roles in `skip` (with their subtrees) are compared separately."""
    failures = []
    trees = subtrees(baseline)
    for cls in CLASSES:
        for role, members in sorted(trees.items()):
            if base(role) in skip:
                continue
            members = {member for member in members if base(member) not in skip}
            mine = sum(candidate["per_role"][cls].get(member, 0) for member in members)
            theirs = sum(baseline["per_role"][cls].get(member, 0) for member in members)
            if mine > theirs:
                failures.append(f"{cls} subtree {role} {mine} > develop {theirs}")
    return failures


def dynamic(candidate: dict, baseline: dict) -> list[str]:
    return [f"{cls} {label} {counts[cls]:g} > develop {baseline['dynamic'][label][cls]:g}"
            for label, counts in candidate["dynamic"].items() for cls in CLASSES
            if counts[cls] > baseline["dynamic"][label][cls]]


def strict_baseline(candidate: dict, baseline: dict) -> dict:
    """M-4 (reviewer 91): admitted variable-size memcpy sites are charged at their upper bound on both sides. A
    baseline site whose exact text the candidate also holds stays at the upper bound (symmetric: no artificial
    sizeof(F) - 4 regression); a baseline site whose text the candidate no longer holds is taken to its lower bound
    (strict: baseline minimum), in its static per-loop and per-role counts and in every dynamic count."""
    texts = {site["text"] for site in candidate.get("bounded_sites", [])}
    changed = [site for site in baseline.get("bounded_sites", []) if site["text"] not in texts]
    if not changed:
        return baseline
    adjusted = json.loads(json.dumps(baseline))
    for site in changed:
        for cls, delta in site["delta"].items():
            loop = str(site["loop"])
            adjusted["per_loop"][cls][loop] = adjusted["per_loop"][cls].get(loop, 0) - delta
            adjusted["per_role"][cls][site["role"]] = adjusted["per_role"][cls].get(site["role"], 0) - delta
        for geometry, values in site["dynamic"].items():
            for cls, delta in values.items():
                adjusted["dynamic"][geometry][cls] = round(adjusted["dynamic"][geometry][cls] - delta, 6)
    return adjusted


def compare_program(candidate: dict, baseline: dict, force_dynamic: bool = False) -> tuple[str, list[str]]:
    """(mode, failures) against one develop program: subtree dominance when the (role, parent role) structures
    match and neither side has sibling copies or alternative role assignments, else per-geometry dynamic
    counts (which take the side's rule over the alternatives). force_dynamic: programs of different kernels
    (a compact program against a develop tile program), whose role names do not denote the same source loops."""
    if not force_dynamic and not (candidate["sibling_copies"] or baseline["sibling_copies"] or
                                  candidate.get("alternatives") or baseline.get("alternatives")) and \
            structure(candidate) == structure(baseline):
        return "dominance", dominance(candidate, baseline)
    return "dynamic", dynamic(candidate, baseline)


def k_walk_zero(candidate: dict) -> list[str]:
    return [f"{cls} {role} {count} (no develop path: K walk must be zero)" for cls in CLASSES
            for role, count in sorted(candidate["per_role"][cls].items()) if base(role) in K_WALK and count]


COMPACT_NAME = re.compile(r"^(NEON|NEON_DOTPROD|NEON_I8MM)/compact:(\d+):(\d+):([1-4]):(\d+):([01-]):([0-2])"
                          r"(?:@([A-Za-z0-9_.+-]+))?$")


def tree_name(table: dict, value: str, name: str) -> str:
    """The Reduction enumerator a compact key's tree field names under that table's own source-derived values."""
    by_value = {str(number): enumerator for enumerator, number in table.get("compact_reduction", {}).items()}
    if value not in by_value:
        raise ValueError(f"{name}: tree value {value} names no Reduction enumerator of its table")
    return by_value[value]


HEX64 = re.compile(r"^[0-9a-f]{64}$")
OPS_SOURCE = "src/v2/kernels/cpu/gemm/CPUNativeNEONMultiScaleOps.h"
TU_CLASSES = ("inference-hot", "startup-planning", "load")


def normalised(text: str) -> str:
    """Comments removed, whitespace collapsed, one space around every punctuation character."""
    text = re.sub(r"//[^\n]*|/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"([^\w\s])", r" \1 ", text)).strip()


def derive_leaf(tile_source: str, ops_source: str) -> dict:
    """P4-a: the leaf instruction family rule of a tree, derived from its own source: the NEONByteDot enumerators
    (exactly ASIMD, DotProd, I8MM with implicit values) and the tile's one leafInstruction(), whose body must be
    exactly `return Instruction == Dot::X && Rows == N ? Dot::Y : Instruction;`. Anything else fails."""
    enums = re.findall(r"enum\s+class\s+NEONByteDot\s*\{([^}]*)\}", re.sub(r"//[^\n]*", " ", ops_source))
    if len(enums) != 1:
        raise ValueError(f"the ops source holds {len(enums)} NEONByteDot enums, not exactly one")
    names = [name.strip() for name in enums[0].split(",") if name.strip()]
    if sorted(names) != ["ASIMD", "DotProd", "I8MM"]:
        raise ValueError(f"NEONByteDot enumerators {names} are not exactly ASIMD, DotProd, I8MM with implicit values")
    values = {name: index for index, name in enumerate(names)}
    signature = "static constexpr Dot leafInstruction() noexcept"
    if tile_source.count(signature) != 1:
        raise ValueError("the tile source has no unique leafInstruction()")
    opening = tile_source.index("{", tile_source.index(signature))
    body = tile_source[opening + 1:tile_source.index("}", opening)]
    rule = re.fullmatch(r"return Instruction = = Dot : : (\w+) & & Rows = = (\d+) \? Dot : : (\w+) : Instruction ;",
                        normalised(body))
    if rule is None or rule.group(1) not in values or rule.group(3) not in values:
        raise ValueError(f"leafInstruction() is not the admitted single-rule form: {normalised(body)!r}")
    return {"tile_sha256": digest(tile_source.encode()), "ops_sha256": digest(ops_source.encode()),
            "families": values, "from": values[rule.group(1)], "rows": int(rule.group(2)),
            "to": values[rule.group(3)]}


def leaf_of(rule: dict, dot: int, rows: int) -> int:
    """The develop tile leaf family a (family, rows) program executes, by the derived rule."""
    return rule["to"] if dot == rule["from"] and rows == rule["rows"] else dot


def check_leaf(rule: dict | None) -> None:
    if not isinstance(rule, dict) or set(rule) != {"tile_sha256", "ops_sha256", "families", "from", "rows", "to"} or \
            not HEX64.match(str(rule["tile_sha256"])) or not HEX64.match(str(rule["ops_sha256"])) or \
            sorted(rule["families"].values()) != [0, 1, 2] or rule["from"] not in (0, 1, 2) or \
            rule["to"] not in (0, 1, 2) or rule["rows"] not in (1, 2, 3, 4):
        raise ValueError("the key map's develop leaf rule is missing or malformed (derive it: derive-leaf)")


def check_copy_map(copy_map: dict) -> None:
    """P4-b: a copy map is the P2 call graph's statement, never invented here: {"evidence_sha256": the P2
    call-graph artifact digest, "copies": {candidate copy: [{"call_site": source-level caller identity,
    "develop_copy": the develop copy that served that call site, or null, "owner_classification": {"owner",
    "class", "record_sha256"} for a call site with no develop counterpart}]}}."""
    if not isinstance(copy_map, dict) or not HEX64.match(str(copy_map.get("evidence_sha256", ""))) or \
            not isinstance(copy_map.get("copies"), dict):
        raise ValueError("the copy map carries no P2 call-graph evidence digest or no copies")
    for copy, sites in copy_map["copies"].items():
        if not isinstance(sites, list) or not sites:
            raise ValueError(f"copy map entry {copy} lists no call sites")
        for site in sites:
            if not isinstance(site, dict) or not isinstance(site.get("call_site"), str) or not site["call_site"] or \
                    "develop_copy" not in site or not (site["develop_copy"] is None or isinstance(site["develop_copy"], str)):
                raise ValueError(f"copy map entry {copy} has a malformed call site {site}")


def replaced_copies(copy: str, copy_map: dict | None) -> tuple[list[str], list[str]]:
    """(develop copies the candidate copy must pass, failures). Without a copy map the identity: the same copy.
    With one, every develop copy whose call sites the candidate copy now serves (strict: each one); a call site
    with no develop counterpart fails unless it carries an owner classification record."""
    if copy_map is None:
        return [copy], []
    sites = copy_map["copies"].get(copy)
    if sites is None:
        return [], [f"the copy map names no call sites served by {copy}"]
    develop, failures = [], []
    for site in sites:
        if site["develop_copy"] is None:
            owner = site.get("owner_classification")
            if not (isinstance(owner, dict) and owner.get("owner") and owner.get("class") and
                    HEX64.match(str(owner.get("record_sha256", "")))):
                failures.append(f"call site {site['call_site']} has no develop counterpart and no owner "
                                f"classification record")
            continue
        if site["develop_copy"] not in develop:
            develop.append(site["develop_copy"])
    return develop, failures


def check_unreachable(name: str, record) -> str | None:
    """P4-c: an unreachability record carries the P2 evidence: {"evidence_sha256" (call graph), "dso_sha256",
    "process_resolution_sha256", "callers": [every enumerated caller of the program's dispatch, each
    {"caller": identity, "reaches": false}]}. A bare listing, a missing digest or a reaching caller fails."""
    if not isinstance(record, dict) or any(not HEX64.match(str(record.get(field, ""))) for field in
                                           ("evidence_sha256", "dso_sha256", "process_resolution_sha256")):
        return f"{name}: the unreachability record carries no P2 evidence digests (call graph, DSO, process resolution)"
    callers = record.get("callers")
    if not isinstance(callers, list) or any(not isinstance(caller, dict) or not caller.get("caller") or
                                            caller.get("reaches") is not False for caller in callers):
        return f"{name}: the unreachability record does not enumerate its callers, or one reaches the program"
    return None


def compare_compact(name: str, program: dict, candidate: dict, baseline: dict, keymap: dict,
                    inputs: dict | None = None) -> dict:
    """P4: one candidate compact program <ENV>/compact:C:W:R:T:O:D@copy against every develop path its source
    formats ran on, in every develop copy it replaces - the same copy (TU-local or link-selected) without a copy
    map; under consolidation (P2 copy_map) every develop copy whose call sites it now serves. Never another
    copy, never a maximum: the candidate must pass each path in each replaced copy (strict minimum).

    Develop paths (key map "compact" entry of compact:C):
    - compact:C' (develop compact): develop's policy-free program compact:C':W:R:T':-:D@copy, where T' is the
      develop value of the same Reduction enumerator (each table's source-derived compact_reduction); it serves
      both candidate policies O = 0 and 1;
    - a tile tail (a format moved from a develop tile path): the develop tile program R:leaf:O:tail, leaf by the
      key map's develop leaf rule derived from develop's own source (derive_leaf(); bound to the develop table's
      tile source hash), of the replaced copy when the develop tile programs are per copy, compared dynamically.
    A compact program whose codebook no format dispatches to the compact implementation fails unless a P2
    unreachability record with its evidence digests lists it (check_unreachable()). A program without a copy
    identity fails. inputs: {"copy_map", "unreachable", "tu_classes"}; tu_class tags the result (P4-d)."""
    inputs = inputs or {}
    match = COMPACT_NAME.match(name)
    if not match:
        return {"develop": [], "formats": [], "modes": [], "failures": [f"malformed compact program name {name}"]}
    mode, codebook, lanes, rows, tree, ordered, dot, copy = match.groups()
    if copy is None:
        return {"develop": [], "formats": [], "modes": [],
                "failures": ["compact program has no copy identity (TU or link-selected copy)"]}
    entry = keymap.get("compact", {}).get(f"compact:{codebook}")
    if entry is None:
        unreachable = inputs.get("unreachable") or {}
        if name in unreachable:
            reason = check_unreachable(name, unreachable[name])
            if reason is None:
                return {"develop": [], "formats": [], "modes": ["unreachable (P2)"], "failures": []}
            return {"develop": [], "formats": [], "modes": [], "failures": [reason]}
        return {"develop": [], "formats": [], "modes": [],
                "failures": [f"no source format dispatches codebook {codebook} to the compact program: its "
                             f"reachability needs a P2 record"]}
    if ordered == "-":
        return {"develop": [], "formats": entry["formats"], "modes": [],
                "failures": ["a candidate compact program must be a policy program (ordered 0 or 1)"]}
    copies, failures = replaced_copies(copy, inputs.get("copy_map"))
    modes, develop = [], []
    enumerator = tree_name(candidate, tree, name)
    tile_copies = any("@" in other and "compact:" not in other for other in baseline["programs"])
    for old in entry["develop"]:
        for replaced in copies:
            if old.startswith("compact:"):
                values = {enumerator: number for enumerator, number in baseline.get("compact_reduction", {}).items()}
                if enumerator not in values:
                    raise ValueError(f"the develop table names no Reduction value for {enumerator}")
                reference_key = f"compact:{old.split(':')[1]}:{lanes}:{rows}:{values[enumerator]}:-:{dot}@{replaced}"
                force = False
            else:
                rule = keymap["trees"]["develop"].get("leaf")
                check_leaf(rule)
                if rule["tile_sha256"] != baseline.get("tile_source_sha256"):
                    raise ValueError("the key map's develop leaf rule was derived from another tile source than the "
                                     "develop table's")
                leaf = leaf_of(rule, int(dot), int(rows))
                reference_key = f"{rows}:{leaf}:{ordered}:{old}" + (f"@{replaced}" if tile_copies else "")
                force = True
            develop.append(reference_key)
            reference = baseline["programs"].get(f"{mode}/{reference_key}")
            if reference is None:
                failures.append(f"develop program {mode}/{reference_key} is missing (no other copy substitutes)")
                continue
            kind, found = compare_program(program, strict_baseline(program, reference), force_dynamic=force)
            modes.append(kind)
            failures += [f"[{reference_key}] {failure}" for failure in found]
    return {"develop": develop, "formats": entry["formats"], "modes": modes, "failures": failures}


def tu_class(name: str, classes: dict | None) -> str:
    """P4-d: the owner/parent TU classification of a compared copy (inference-hot, startup-planning, load);
    unclassified when the input does not name it. Classification never relaxes the comparison."""
    copy = name.split("@", 1)[1] if "@" in name else None
    value = (classes or {}).get(copy) if copy else None
    if value is not None and value not in TU_CLASSES:
        raise ValueError(f"TU class {value!r} of {copy} is not one of {TU_CLASSES}")
    return value or "unclassified"


def grouped(results: dict) -> dict:
    """P4-d: results grouped by TU class, each group listing its passing and failing programs (all strict)."""
    groups: dict[str, dict] = {}
    for name, result in sorted(results.items()):
        group = groups.setdefault(result.get("tu_class", "unclassified"), {"passing": [], "failing": []})
        group["failing" if result["failures"] else "passing"].append(name)
    return groups


def compare(candidate: dict, baseline: dict, keymap: dict, inputs: dict | None = None) -> dict:
    """Every candidate program against every develop program its source formats ran on (strict: the candidate
    must pass each, i.e. stay within the minimum). A candidate tile key some format of which had no develop tile
    path additionally needs a traffic-free K walk."""
    if baseline.get("sibling_rule") != "baseline" or candidate.get("sibling_rule") != "candidate":
        raise ValueError("the baseline must use the develop-min and the candidate the candidate-max sibling rule")
    if baseline["tree_sha"] != keymap["baseline"]:
        raise ValueError("the key map names another baseline tree")
    if candidate["geometries"] != baseline["geometries"]:
        raise ValueError("tables use different reference geometries")
    _, keys = derive_keys(keymap)
    inputs = inputs or {}
    if inputs.get("copy_map") is not None:
        check_copy_map(inputs["copy_map"])
    results = {}
    for name, program in sorted(candidate["programs"].items()):
        mode, key = name.split("/")
        if key.startswith("compact:"):
            results[name] = compare_compact(name, program, candidate, baseline, keymap, inputs)
            results[name]["tu_class"] = tu_class(name, inputs.get("tu_classes"))
            continue
        fields = key.split(":")
        tail = ":".join(fields[3:])
        entry = keys.get(tail)
        if entry is None:
            results[name] = {"develop": [], "formats": [], "modes": [], "failures": [f"unmapped key tail {tail}"]}
            continue
        failures, modes, develop = [], [], []
        for old in entry["develop"]:
            if not TAIL.match(old):
                failures += k_walk_zero(program)
                modes.append(f"no develop tile path ({old})")
                continue
            reference = baseline["programs"].get(f"{mode}/{':'.join(fields[:3])}:{old}")
            develop.append(f"{':'.join(fields[:3])}:{old}")
            if reference is None:
                failures.append(f"develop program {mode}/{develop[-1]} is missing")
                continue
            kind, found = compare_program(program, strict_baseline(program, reference))
            modes.append(kind)
            failures += [f"[{develop[-1]}] {failure}" for failure in found]
        results[name] = {"develop": develop, "formats": entry["formats"], "modes": modes, "failures": failures,
                         "tu_class": tu_class(name, inputs.get("tu_classes"))}
    return results


def load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), HERE / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_checker(baseline: dict, checker_sha256: str) -> None:
    """The develop table must have been measured by the running checker: the same semantics on both sides."""
    if baseline.get("checker_sha256") != checker_sha256:
        raise ValueError(f"the develop table was measured by checker {str(baseline.get('checker_sha256'))[:12]}, "
                         f"not the running checker {checker_sha256[:12]}: regenerate the table")


def gate_proof(args: argparse.Namespace) -> dict:
    """The gate step of the canonical proof: verify the proof with the checker (unchanged), re-analyse the bound
    object, build the candidate table with the committed role map (selected by the candidate tile source's
    SHA256), check the key map against the candidate compile's inputs and dispatch(), compare with the committed
    develop table and write a receipt binding every hash. Raises on any structural mismatch; returns the receipt
    (its failing programs are the gate result)."""
    checker = HERE / "CPUNativeTileRegisterEvidence.py"
    ev, baseline_tool, table_tool = (load("CPUNativeTileRegisterEvidence.py"), load("CPUNativeTileStackBaseline.py"),
                                     load("CPUNativeTileStackTable.py"))
    proof_path = Path(args.proof)
    record_path = ev.analysis_path(proof_path)
    if getattr(args, "analysis", False):
        # Unqualified input (UA-1..UA-4): the checker binds and re-analyses the record; the comparison below is
        # reported in full, and the gate result is always failed (finish_receipt()).
        kind, document_path, action, schema = ev.ANALYSIS_KIND, record_path, "check-analysis", ev.ANALYSIS_SCHEMA
    else:
        if not proof_path.exists():
            raise ValueError("no sealed proof of this run: " + (
                "an unqualified analysis record exists (admission failed; the driver passes --analysis)"
                if record_path.exists() else "and no unqualified analysis record (stage 1 failed structurally)"))
        kind, document_path, action, schema = ev.SEALED_KIND, proof_path, "verify", 12
    verified = subprocess.run([sys.executable, "-B", str(checker), action, *args.checker_args],
                              capture_output=True, text=True)
    if verified.returncode:
        raise ValueError(f"the tile register {kind} does not verify: "
                         f"{verified.stderr[-1500:] or verified.stdout[-1500:]}")
    proof = json.loads(document_path.read_text())
    obj, dis, asm = (proof_path.with_suffix(suffix) for suffix in (".o", ".dis", ".asm"))
    if proof.get("schema") != schema or proof.get("kind") != kind or proof.get("mode") != args.mode or \
            proof.get("checker_sha256") != digest(checker.read_bytes()) or \
            proof.get("object_sha256") != digest(obj.read_bytes()) or \
            proof.get("disassembly_sha256") != digest(dis.read_bytes()) or \
            proof.get("assembly_sha256") != digest(asm.read_bytes()):
        raise ValueError("the gate input is not the verified proof (schema, mode, checker, object, disassembly or "
                         "assembly hash)")
    data, disassembly, assembly = obj.read_bytes(), dis.read_text(), asm.read_text(errors="replace")
    analyses: dict = {}
    traffic: dict = {}
    counts = ev.audit(data, disassembly, args.mode, traffic, analyses=analyses)
    named = ev.name_loops(traffic, data, disassembly, assembly)
    if proof.get("stack_traffic") != named:
        raise ValueError("re-analysis of the bound object differs from the proof's recorded stack traffic")
    annotations = loop_annotations(ev, baseline_tool, analyses, data, disassembly, assembly)
    return gate_compare(args, kind, proof, digest(document_path.read_bytes()), counts, named, annotations)


def loop_annotations(ev, baseline_tool, analyses: dict, data: bytes, disassembly: str, assembly: str) -> dict:
    """Per program key, each machine loop's member locations and control evidence (loop_lines(), loop_evidence())
    from the analyses of one analysis pass, as plain JSON values (the V1 driver freezes them as canonical text)."""
    image = ev.read_object(data)
    bodies = ev.normalize_disassembly(disassembly, image)
    locations: dict[str, list[str]] = {}
    ev.assembly_functions(assembly, set(bodies), image.macho, locations)
    result = {}
    for name, analysis in sorted(analyses.items()):
        where = locations.get(name, [])
        result[ev.program_key(name)] = {
            "lines": {str(loop): value for loop, value in baseline_tool.loop_lines(analysis, where).items()},
            "evidence": {str(loop): value for loop, value in baseline_tool.loop_evidence(analysis, where).items()}}
    return json.loads(json.dumps(result))


def gate_from_analysis(args: argparse.Namespace, verified, annotations_text: str) -> dict:
    """The gate on one driver attempt's frozen verified analysis (V1): the checker that made it must be the checker
    file next to this gate, and its document's analysis fields must equal the frozen ones; no verification or
    analysis is repeated (they ran once in this attempt). Same comparison and receipt as gate_proof()."""
    checker = HERE / "CPUNativeTileRegisterEvidence.py"
    if verified.checker_sha256 != digest(checker.read_bytes()):
        raise ValueError("the verified analysis was made by another checker")
    proof = json.loads(verified.document_text)
    fields = json.loads(verified.fields_text)
    if any(proof.get(name) != fields.get(name) for name in ("program_violations", "ordered_adds", "stack_traffic",
                                                             "chain_witness")):
        raise ValueError("the verified analysis differs from its document")
    schema = 12 if verified.kind == "sealed-proof" else "12-unqualified-analysis"
    if proof.get("schema") != schema or proof.get("kind") != verified.kind or proof.get("mode") != args.mode or \
            proof.get("checker_sha256") != verified.checker_sha256:
        raise ValueError("the verified analysis is not this mode's document")
    return gate_compare(args, verified.kind, proof, digest(verified.document_text.encode()),
                        fields["program_violations"], fields["stack_traffic"], json.loads(annotations_text))


def gate_compare(args: argparse.Namespace, kind: str, proof: dict, document_sha256: str, counts: dict, named: dict,
                 annotations: dict) -> dict:
    """The candidate table of one verified analysis against the committed develop table, and the receipt."""
    ev, baseline_tool, table_tool = (load("CPUNativeTileRegisterEvidence.py"), load("CPUNativeTileStackBaseline.py"),
                                     load("CPUNativeTileStackTable.py"))
    inputs = {}
    for path, identity in proof["inputs"].items():
        for logical in HEADERS + (TILE_SOURCE,):
            if path.endswith("/" + logical):
                inputs[logical] = identity["sha256"]
    source_path = next(path for path in proof["inputs"] if path.endswith("/" + TILE_SOURCE))
    source = Path(source_path).read_text()
    if digest(source.encode()) != inputs.get(TILE_SOURCE):
        raise ValueError("the candidate tile source changed after the proof")
    keymap = json.loads(Path(args.keymap).read_text())
    check_keymap(keymap, inputs, source)
    for key, record in named.items():
        annotated = annotations[key]
        baseline_tool.annotate_loops(record["loops"], {int(loop): value for loop, value in annotated["lines"].items()},
                                     {int(loop): value for loop, value in annotated["evidence"].items()})
    record = {"tree_sha": "candidate", "mode": args.mode, "checker_sha256": proof["checker_sha256"],
              "tree_inputs": inputs, "stack_traffic": named,
              "fp_spills": {ev.program_key(name): count for name, count in counts.items()},
              "generator_sha256": digest(Path(__file__).read_bytes()), "admission_sha256": proof["checker_sha256"],
              "probe_sha256": next(identity["sha256"] for path, identity in proof["inputs"].items()
                                   if path.endswith("/tests/v2/portability/CPUNativeNEONTileRegisterProbe.cpp")),
              "closure_sha256": digest(json.dumps(sorted(f"{p}={i['sha256']}" for p, i in proof["inputs"].items()))
                                       .encode()),
              "object_sha256": proof["object_sha256"], "assembly_sha256": proof["assembly_sha256"],
              "image": "proof", "identity": proof["identity"], "compiler_version": proof["compiler_version"],
              "disassembler_version": proof["disassembler_version"],
              "target_command_sha256": proof["target_command_sha256"]}
    roles = json.loads(Path(args.roles).read_text())
    candidate = table_tool.table([record], roles, source, "candidate")
    baseline = json.loads(Path(args.baseline).read_text())
    table_tool.check(baseline)
    require_checker(baseline, proof["checker_sha256"])
    results = compare(candidate, {**baseline, "programs": {name: program for name, program in
                                                           baseline["programs"].items()
                                                           if name.startswith(args.mode + "/")}}, keymap)
    return finish_receipt({"schema": 1, "mode": args.mode, "proof_sha256": document_sha256,
            "checker_sha256": proof["checker_sha256"], "gate_sha256": digest(Path(__file__).read_bytes()),
            "table_tool_sha256": digest((HERE / "CPUNativeTileStackTable.py").read_bytes()),
            "loop_lines_sha256": digest((HERE / "CPUNativeTileStackBaseline.py").read_bytes()),
            "baseline_sha256": digest(Path(args.baseline).read_bytes()),
            "roles_sha256": digest(Path(args.roles).read_bytes()), "keymap_sha256": digest(Path(args.keymap).read_bytes()),
            "tile_source_sha256": inputs[TILE_SOURCE], "results": results, "by_tu_class": grouped(results),
            # C6: every program's alternative readings, empty self-loops and enumeration size.
            "alternatives": {name: program["alternatives"] for name, program in candidate["programs"].items()},
            "empty_loops": {name: program["empty_loops"] for name, program in candidate["programs"].items()
                            if program["empty_loops"]},
            "role_combinations": {name: program["role_combinations"] for name, program in candidate["programs"].items()},
            # S4: how every loop's source loop was resolved (evidence/mechanism) and nested-copy justification.
            "provenance": {name: [f"{loop['id']}:{loop['role']}@{loop['source_line']}:{loop['provenance']}"
                                  + ("" if loop["copy_justified"] else ":copy-unjustified")
                                  for loop in program["loops"]] for name, program in candidate["programs"].items()},
            "failing": sorted(name for name, result in results.items() if result["failures"])},
        kind, proof.get("admission"))


def run_gate(args: argparse.Namespace, compute) -> dict:
    """Retire the receipt, compute the gate, write its receipt (a failed one on any error, then re-raise)."""
    Path(args.receipt).unlink(missing_ok=True)
    try:
        receipt = compute()
    except (ValueError, OSError) as error:
        Path(args.receipt).write_text(json.dumps({"schema": 1, "mode": args.mode, "status": "failed",
                                                  "input": "unqualified-analysis" if getattr(args, "analysis", False)
                                                  else "sealed-proof", "error": str(error)}, indent=1) + "\n")
        raise
    Path(args.receipt).write_text(json.dumps(receipt, sort_keys=True, indent=1) + "\n")
    return receipt


def finish_receipt(receipt: dict, kind: str, admission: dict | None) -> dict:
    """UA-3/UA-4 vocabulary: `input` names the consumed document, `comparison` is clean or regressions, `status` is
    passed only for a sealed proof with a clean comparison. Unqualified input is always failed (its comparison is
    diagnostic: never Gate7 evidence, never a candidate verdict) and carries the record's admission."""
    receipt["input"] = kind
    receipt["comparison"] = "regressions" if receipt["failing"] else "clean"
    sealed = kind == "sealed-proof"
    receipt["status"] = "passed" if sealed and not receipt["failing"] else "failed"
    if not sealed:
        receipt["admission"] = admission
        receipt["diagnostic"] = "unqualified analysis: not Gate7 evidence and no input to any candidate verdict"
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    tables = sub.add_parser("compare", help="compare two tables")
    tables.add_argument("--baseline", required=True)
    tables.add_argument("--candidate", required=True)
    tables.add_argument("--keymap", required=True)
    tables.add_argument("--out")
    for option in ("copy-map", "unreachable", "tu-classes"):
        tables.add_argument(f"--{option}", help="P2/owner input (JSON), never invented here")
    keys = sub.add_parser("check-keymap", help="internal consistency of the key map")
    keys.add_argument("--keymap", required=True)
    derive = sub.add_parser("derive-compact", help="write the key map's derived compact program keys (mechanical)")
    derive.add_argument("--keymap", required=True)
    leaf = sub.add_parser("derive-leaf", help="write the develop leaf rule derived from develop's own sources")
    for option in ("keymap", "tile-source", "ops-source"):
        leaf.add_argument(f"--{option}", required=True)
    proof = sub.add_parser("proof", help="the gate step of the canonical proof (after the checker's verify)")
    for option in ("proof", "mode", "baseline", "roles", "keymap", "receipt"):
        proof.add_argument(f"--{option}", required=True)
    proof.add_argument("--analysis", action="store_true",
                       help="consume this run's unqualified analysis record (always a failed gate result)")
    proof.add_argument("checker_args", nargs=argparse.REMAINDER, help="-- then the checker's verify arguments")
    args = parser.parse_args()
    if args.action == "derive-leaf":
        keymap = json.loads(Path(args.keymap).read_text())
        keymap["trees"]["develop"]["leaf"] = derive_leaf(Path(args.tile_source).read_text(),
                                                         Path(args.ops_source).read_text())
        check_keymap(keymap)
        Path(args.keymap).write_text(json.dumps(keymap, sort_keys=True, indent=1) + "\n")
        print(f"key map: develop leaf rule {keymap['trees']['develop']['leaf']['from']}@"
              f"{keymap['trees']['develop']['leaf']['rows']} -> {keymap['trees']['develop']['leaf']['to']}")
        return
    if args.action == "derive-compact":
        keymap = json.loads(Path(args.keymap).read_text())
        keymap["compact"] = derive_compact(keymap)
        check_keymap(keymap)
        Path(args.keymap).write_text(json.dumps(keymap, sort_keys=True, indent=1) + "\n")
        print(f"key map: {len(keymap['compact'])} compact program key prefixes")
        return
    if args.action == "check-keymap":
        check_keymap(json.loads(Path(args.keymap).read_text()))
        print("key map: consistent with its recorded encodings and dispatch rules")
        return
    if args.action == "compare":
        keymap = json.loads(Path(args.keymap).read_text())
        check_keymap(keymap)
        inputs = {key: json.loads(Path(path).read_text()) for key, path in
                  (("copy_map", args.copy_map), ("unreachable", args.unreachable), ("tu_classes", args.tu_classes))
                  if path}
        results = compare(json.loads(Path(args.candidate).read_text()), json.loads(Path(args.baseline).read_text()),
                          keymap, inputs)
        if args.out:
            Path(args.out).write_text(json.dumps({"results": results, "by_tu_class": grouped(results)},
                                                 sort_keys=True, indent=1) + "\n")
        for group, members in grouped(results).items():
            print(f"{group}: {len(members['passing'])} passing, {len(members['failing'])} failing")
        failing = [name for name, result in results.items() if result["failures"]]
    else:
        args.checker_args = [argument for argument in args.checker_args if argument != "--"]
        receipt = run_gate(args, lambda: gate_proof(args))
        results, failing = receipt["results"], receipt["failing"]
        for name in failing[:12]:
            print(f"{name}: " + "; ".join(results[name]["failures"][:3]))
        print(f"{len(results)} candidate programs, {len(failing)} with new stack traffic; input {receipt['input']}, "
              f"comparison {receipt['comparison']}, status {receipt['status']}")
        if receipt["status"] != "passed":
            raise SystemExit(1)
        return
    print(f"{len(results)} candidate programs, {len(failing)} with new stack traffic")
    if failing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
