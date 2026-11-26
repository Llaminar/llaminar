#!/usr/bin/env python3
"""Assemble the stack-traffic table of the native NEON tile programs from baseline records and a role map.

Input: baseline records written by CPUNativeTileStackBaseline.py (one per ISA envelope, same tree) and the
committed role map CPUNativeTileStackRoles.json. Output: the table (per program: per-loop and per-role class
vectors and per-geometry dynamic counts), hashed with every input.

Classes overlap and are NEVER summed: each class is compared on its own against the same class of the baseline,
loop role by loop role (or by dynamic count where the role structures differ). Documented limits shared by
every tree measured with the same checker: a register redefined from itself plus a constant within one
iteration is treated as variant (invariant_load undercount); a pointer held in a descriptor field the loop
itself rewrites is treated as invariant (invariant_load overcount; the rewrite is counted as alias_store).

Role assignment (source-grounded, fail-closed): the role map names, per tree, every `for` statement of the
tile source file that may carry a machine loop, by header line, with a role from a fixed enum and the trip
expression of its source bounds; kernel functions (and inlined kernels with their call sites and the
instantiation condition of each) are named by unique source text. The `for` statements are parsed from the
tree's own source (header line and extent, loop variable, body extent); every mapped line must be a `for`
header over the role's variable. A machine loop's role is resolved from its CONTROL instructions only - the
.loc of its back-edge branch and the compare feeding it, and of its entry blocks' first instructions - never
from its body (resolve_roles); body instructions are charged to the machine loop holding them wherever their
.loc points. Unresolved or conflicting control locations fail. Loops of one program sharing a role are
indexed (role#0, role#1, ... in loop id order) and reported.

Helper loops (call model): a record's loops carrying "helper" are loops of an audited helper nested under the
call site's loop. Their role comes from the role map entry's "helpers" {raw helper symbol: {"source", "source_sha256",
"loops": {helper loop id: role}}}, which must equal the entry the checker admitted the call under, name the compiled
helper source (the record's closure hash and the supplied text, --helper-source) and a `for` header over the role's
variable there; their trips multiply under the caller's chain (caller chain x helper chain). Anything else fails.

Compact programs (keys compact:codebook:lanes:rows:tree:ordered:dot, the checker's compact_program_key(); ordered
"-" = develop's policy-free rows<>): their roles come from the role map entry's "compact" entry, written
mechanically from the tree's CPUNativeQuantisedMultiScaleKernels.h by compact_entry() (--compact-entry): the one
compact kernel (rowsProgram, else develop's rows; a PR rows() wrapper must hold no for) and its for statements by
variable - column (C / W), partition (P if T == 1 else 1), kb (block: K / P if T == 1 else K), row (R, declared
unrolled under `#pragma GCC unroll N`, N >= 4) - any other variable fails. The compact source must be supplied
(--helper-source PATH=FILE), equal the entry's hash and the records' closure hash. W = lanes, T = 1 for the
ordered partition tree. The table records compact_trips (and helper_trips) so check() recomputes every count.

Dynamic counts: for a geometry (columns, K blocks, partitions), executions(loop) = product over the distinct
roles on its loop chain (outermost first) of the role's trip; loop 0 executes once. Trips:
column = columns / column step (4, or 4*S on the panel path with the program's panel width S from the tree's own
panelStrips rule), partition = P, block = K / P (exact: the partitions together walk K blocks), group = its
constant bound, strip = S, row = Rows. A nested loop with the same role as an ancestor (a rotated or duplicated
copy of one source loop) does not multiply again. Geometries: 64 columns; K in {1, 3, 17}; P in {1, 2, K}
("split": one block per partition).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import json
import re
from pathlib import Path

TILE_FILE = "src/v2/kernels/cpu/gemm/CPUNativeNEONPreparedProjection.h"
COMPACT_FILE = "src/v2/kernels/cpu/gemm/CPUNativeQuantisedMultiScaleKernels.h"
COMPACT_TRIP_VARIABLES = ("W", "T")
# Compact program kernels by tree: the PR's rowsProgram<OrderedNaN, ...>() and develop's policy-free rows<...>().
COMPACT_KERNELS = ("inline void rowsProgram(", "inline void rows(")
# Mechanical compact role map (compact_entry()): every `for` of the compact kernel, admitted only when its
# normalised header and the normalised definitions of the bounds it reads equal these exact source patterns
# (S2-1); the trip is then the one the pattern implies. T is 1 for the ordered partition tree (compact_context()
# derives it from the source's Reduction enumerators, S2-2). Any drift fails: the map is extended by review.
COMPACT_PATTERNS = {
    "column": {"role": "column", "trip": "ceil(C / W)", "header": "int column = 0; column < columns; column += Lanes",
               "definitions": {}},
    "partition": {"role": "partition", "trip": "P if T == 1 else 1",
                  "header": "int partition = 0; partition < partition_count; ++partition",
                  "definitions": {"partition_count": "Tree == Reduction::OrderedPartitions ? partitions : 1"}},
    "kb": {"role": "block", "trip": "K / P if T == 1 else K", "header": "int kb = first; kb < last; ++kb",
           "definitions": {
               "first": "Tree == Reduction::OrderedPartitions ? begin + partition * blocks_per_partition : begin",
               "last": "Tree == Reduction::OrderedPartitions ? std::min(first + blocks_per_partition, end) : end"}},
    "row": {"role": "row", "trip": "R", "header": "size_t row = 0; row < Rows; ++row", "definitions": {}},
}
REDUCTION_ENUM = re.compile(r"\benum\s+class\s+Reduction\s*\{([^}]*)\}")
ROLES = {"column": "column", "partition": "partition", "superblock": "superblock", "block": "block", "group": "group", "strip": "strip",
         "row": "row", "pair": "pair", "half": "half"}
CLASSES = ("fp", "scalar", "abi_loop", "alias_load", "alias_store", "invariant_load", "invariant_reloc",
           "tls_call", "call_frame", "int_call", "ext_call", "ext_bytes", "ext_accesses")
GEOMETRIES = [{"columns": 64, "k": k, "partitions": p, "label": f"c64-k{k}-p{name}"}
              for k in (1, 3, 17) for p, name in ((1, "1"), (2, "2"), (k, "split"))]
FOR = re.compile(r"\bfor\s*\(")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_fors(text: str) -> dict[int, dict]:
    """Every `for` statement of a C++ file: header line -> loop variable, last line of its header (the closing
    parenthesis of init; condition; increment) and last line of its body."""
    lines = text.split("\n")
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line) + 1)
    def line_of(offset: int) -> int:
        low, high = 0, len(starts) - 1
        while low < high:
            middle = (low + high + 1) // 2
            if starts[middle] <= offset:
                low = middle
            else:
                high = middle - 1
        return low + 1
    result = {}
    for match in FOR.finditer(text):
        position, depth = match.end(), 1
        while depth:
            depth += {"(": 1, ")": -1}.get(text[position], 0)
            position += 1
        header = text[match.end():position - 1]
        header_end = line_of(position - 1)
        variable = re.search(r"(?:int|std::size_t|size_t|auto)\s+(\w+)\s*=", header) or \
            re.match(r"[^;]*;\s*(\w+)\s*(?:<|<=|>|>=|!=)", header)  # for (; block < end; ...): the tested variable.
        while text[position].isspace():
            position += 1
        if text[position] == "{":
            depth, position = 1, position + 1
            while depth:
                depth += {"{": 1, "}": -1}.get(text[position], 0)
                position += 1
        else:
            depth = 0
            while not (text[position] == ";" and depth == 0):
                depth += {"(": 1, ")": -1, "{": 1, "}": -1}.get(text[position], 0)
                position += 1
        header_line = line_of(match.start())
        previous = lines[header_line - 2].strip() if header_line >= 2 else ""
        unroll = re.fullmatch(r"#pragma GCC unroll (\d+)", previous)
        result[header_line] = {"variable": variable.group(1) if variable else None,
                               "header_end": header_end, "end": line_of(position),
                               "unroll": int(unroll.group(1)) if unroll else None}
    return result


def function_extents(text: str, kernels: list[dict]) -> list[dict]:
    """Body line extents of the kernel functions named by signature substrings (each must occur exactly once).

    An always-inlined kernel (a member function or a lambda) names the unique texts of its call sites (`calls`;
    `call` for one); each call site must lie in another kernel. A lambda passed to an inlined kernel and invoked
    there names that kernel's invocation sites as `calls` and the text of the call that passes it as `via`: only
    contexts reached through `via` are feasible for it.
    """
    def find(needle: str) -> int:
        if text.count(needle) != 1:
            raise ValueError(f"kernel text {needle!r} is not unique in the tile source")
        return text.index(needle)

    def line(offset: int) -> int:
        return text.count("\n", 0, offset) + 1
    extents = []
    for kernel in kernels:
        start = find(kernel["signature"])
        opening = text.index("{", start)
        depth, position = 1, opening + 1
        while depth:
            depth += {"{": 1, "}": -1}.get(text[position], 0)
            position += 1
        calls = kernel.get("calls", [kernel["call"]] if kernel.get("call") else [])
        # A call site may carry the condition (over the program key: R rows, L leaf, G policy, E encoding) under
        # which its if-constexpr branch is instantiated, transcribed from the source.
        calls = [call if isinstance(call, dict) else {"call": call} for call in calls]
        # The body runs from its opening brace: a lambda's call site shares the line of its signature.
        extents.append({"first": line(opening), "last": line(position),
                        "call_sites": [line(find(call["call"])) for call in calls],
                        "call_when": [call.get("when") for call in calls],
                        "via": line(find(kernel["via"])) if kernel.get("via") else None})
    for extent in extents:
        for site in extent["call_sites"] + ([extent["via"]] if extent["via"] else []):
            if not any(other is not extent and other["first"] <= site <= other["last"] for other in extents):
                raise ValueError("an inlined kernel's call site lies in no other kernel function")
    return extents


def chains(line: int, fors: dict[int, dict], kernels: list[dict], depth: int = 0,
           call: bool = False, context: dict | None = None) -> list[tuple[list[int], frozenset]]:
    """Feasible enclosing-for chains of a source line, outermost first, each with the call-site lines it passes:
    the line's innermost kernel's own for statements enclosing it, prefixed by every feasible chain of each of
    that kernel's call sites (a root kernel: no prefix). Empty when the line is in no kernel."""
    if depth > 8:
        raise ValueError("kernel call sites nest too deeply (a cycle?)")
    # A call site sharing its line with the opening of the lambda it passes belongs to the caller, not the lambda.
    containing = [k for k in kernels if k["first"] <= line <= k["last"] and not (call and k["via"] == line)]
    if not containing:
        return []
    kernel = min(containing, key=lambda k: k["last"] - k["first"])
    inner = [k for k in kernels if k is not kernel and kernel["first"] <= k["first"] and k["last"] <= kernel["last"]]
    own = sorted(header for header, body in fors.items()
                 if kernel["first"] <= header and header <= line <= body["end"]
                 and not any(k["first"] <= header <= k["last"] for k in inner))
    if not kernel["call_sites"]:
        return [(own, frozenset())]
    result = []
    for site, when in zip(kernel["call_sites"], kernel.get("call_when", [None] * len(kernel["call_sites"]))):
        if when is not None and context is not None and not eval(when, {"__builtins__": {}}, dict(context)):
            continue  # The call site's branch is not instantiated for this program.
        for prefix, used in chains(site, fors, kernels, depth + 1, call=True, context=context):
            used = used | {site}
            if kernel["via"] is None or kernel["via"] in used:
                result.append((prefix + own, used))
    return result


def chain(line: int, fors: dict[int, dict], kernels: list[dict]) -> list[int] | None:
    """The single feasible chain of a line (None when it is in no kernel); several feasible chains are an error."""
    found = {tuple(c) for c, _ in chains(line, fors, kernels)}
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(f"line {line} has several feasible enclosing chains")
    return list(found.pop())


def control_fors(line: int, fors: dict[int, dict], kernels: list[dict], mapped: dict[str, dict],
                 context: dict | None = None) -> set[int]:
    """The source loops a control instruction's .loc names: the for statements (inside kernel functions) whose
    header (init; condition; increment) spans the line; for a line outside every header - an induction variable
    GCC strength-reduced into an address computation of the body, the compare then carrying that line - the
    innermost mapped for statement enclosing it on each feasible chain (unmapped ones there are fully unrolled
    and single-trip ones for the program collapse: neither forms a machine loop)."""
    feasible = [chain for chain, _ in chains(line, fors, kernels, context=context)]
    if not feasible:
        return set()
    headers = {header for header, body in fors.items() if header <= line <= body["header_end"]
               and any(header in chain for chain in feasible)}
    if headers:
        return headers
    return {next((header for header in reversed(chain) if str(header) in mapped and
                  not (context is not None and single_trip(mapped[str(header)], context))), None)
            for chain in feasible} - {None}


def encloses(outer: int, inner: int, fors: dict[int, dict], kernels: list[dict], context: dict | None = None) -> bool:
    """Whether source for `outer` encloses source for `inner` on some feasible chain (inlined call sites
    included)."""
    return any(outer in chain[:-1] for chain, _ in chains(inner, fors, kernels, context=context))


def single_trip(role: dict, context: dict) -> bool:
    """A mapped source loop whose trip is exactly 1 for this program at every reference geometry: it forms no
    machine loop (e.g. the coded backend-native partition loop, `ordered || Coded == 0 ? partitions_ : 1`)."""
    return all(trip(role, geometry, context["R"], context["S"], context["G"], context["E"], context) == 1.0
               for geometry in GEOMETRIES)


def program_context(key: str, strips: int) -> dict:
    rows, leaf, policy, encoding, nibble_map, _ = (int(field) for field in key.split(":"))
    return {"R": rows, "L": leaf, "G": policy, "E": encoding, "C": nibble_map, "S": strips}


def declared_unrolled(line: int, mapped: dict[str, dict], fors: dict[int, dict], context: dict) -> bool:
    """The role map declares a mapped for statement fully unrolled for this program (`unrolled`: a condition over
    R, L, G, E, S), validated against the source: an explicit `#pragma GCC unroll N` directly before its header
    with N at least its trip at every reference geometry."""
    role = mapped.get(str(line))
    if not role or not role.get("unrolled") or not eval(role["unrolled"], {"__builtins__": {}}, dict(context)):
        return False
    factor = fors[line].get("unroll")
    if not factor or any(trip(role, geometry, context["R"], context["S"], context["G"], context["E"], context) > factor
                         for geometry in GEOMETRIES):
        raise ValueError(f"role map declares for statement {line} unrolled without a source pragma covering its trip")
    return True


def multi_trip(line: int, mapped: dict[str, dict], fors: dict[int, dict], context: dict) -> bool:
    """A mapped for statement that must form a machine loop for this program: not single-trip, not a declared,
    source-validated full unroll."""
    return str(line) in mapped and not single_trip(mapped[str(line)], context) and \
        not declared_unrolled(line, mapped, fors, context)


MAX_COMBINATIONS = 4096  # Enumeration bound per program; larger domains fail closed, never truncate.


EMPTY_ROLE = {"line": None, "role": "empty", "variable": None, "trip": "1", "provenance": "empty-self-loop",
              "copy_justified": False, "empty": True}


def resolve_roles(program: str, loops: list[dict], fors: dict[int, dict], mapped: dict[str, dict],
                  kernels: list[dict], context: dict, stats: dict | None = None) -> list[dict[int, dict]]:
    """Every assignment of machine loops to mapped source for statements that the loops' CONTROL evidence admits
    and the search-independent invariants prove; never derived from loop bodies.

    Domain of a loop: the multi-trip mapped for statements its control locations name (control_fors over the .loc
    lines of its latches - back-edge branch and the compare feeding it - and of its entry blocks' first
    instructions; feasible chains honour the call sites' instantiation conditions, `context`). The assignments
    are enumerated exhaustively (no heuristic search, so no order dependence: the enumeration runs over the
    domains forward and reversed and must yield the same set) and each must satisfy prove_assignment():
    (a) evidence, (b) embedding, (c) no skipped level, (d) no missing loop. No valid assignment fails. Several
    valid assignments (alternatives) are all returned: table() charges a candidate the maximum and the baseline
    the minimum over them, and they disable static dominance. Provenance per loop and alternative: evidence
    (latch-header, latch-body, entry-header, entry-body) / mechanism (candidate: its domain has one element;
    forced: one valid assignment; alternative: several); `copy_justified` only when its own latch names the for
    header directly and the assignment is forced or a sole candidate. Body instructions, wherever their .loc
    points, are charged to the machine loop holding them and never decide its role."""
    def tile_lines(sources: list[str]) -> list[int]:
        return [int(source.rsplit(":", 1)[1]) for source in sources
                if source.rsplit(":", 1)[0] == context.get("file", TILE_FILE)]

    def control(lines: list[int]) -> set[int]:
        return set().union(*(control_fors(line, fors, kernels, mapped, context) for line in lines)) if lines else set()

    def direct(lines: list[int]) -> set[int]:
        return {header for line in lines for header, body in fors.items() if header <= line <= body["header_end"]
                and any(header in chain for chain, _ in chains(line, fors, kernels, context=context))}
    empty = {loop["id"]: bool(loop.get("empty")) for loop in loops}
    for loop in loops:  # C1: an empty self-loop has no children and no role.
        if empty[loop["id"]] and any(other["parent"] == loop["id"] for other in loops):
            raise ValueError(f"{program}: empty self-loop {loop['id']} holds a nested loop")
    roled = [loop for loop in loops if not empty[loop["id"]]]
    candidates, evidence, induction = {}, {}, {}
    for loop in roled:
        latch, entry = tile_lines(loop["latch_evidence"]), tile_lines(loop["entry_evidence"])
        latch_any = control(latch)
        candidates[loop["id"]] = latch_any | control(entry)
        evidence[loop["id"]] = (direct(latch), latch_any, direct(entry))
        # C2 (i): induction evidence counts only where the step's .loc lies in a candidate's header extent.
        induction[loop["id"]] = direct(tile_lines(loop.get("induction_evidence", [])))
        # C5: a header declared fully unrolled for this program cannot be iterated by a machine loop.
        unrolled = sorted(line for line in evidence[loop["id"]][0] | induction[loop["id"]]
                          if line in fors and declared_unrolled(line, mapped, fors, context))
        if unrolled:
            raise ValueError(f"{program}: loop {loop['id']} iterates for statement {unrolled[0]}, which the role map "
                             f"declares fully unrolled")
    ids = [loop["id"] for loop in roled]

    for loop in roled:
        unmapped = sorted(line for line in evidence[loop["id"]][0] if str(line) not in mapped)
        if unmapped:
            raise ValueError(f"{program}: loop {loop['id']} resolves to the unmapped for statement at line "
                             f"{unmapped[0]} (its back edge names that header directly)")

    def domain_of(identity: int) -> list[int]:
        # The back edge tests a register the loop's own cycle steps: the loop iterates that source loop, so an
        # induction-named candidate restricts the domain.
        found = {line for line in candidates[identity]
                 if str(line) in mapped and not single_trip(mapped[str(line)], context)}
        stepped = found & induction[identity]
        return sorted(stepped or found)
    domains = [domain_of(identity) for identity in ids]

    def enumerate_valid(reverse: bool) -> list[tuple[int, ...]]:
        found = []
        for combo in itertools.product(*[list(reversed(domain)) if reverse else domain for domain in domains]):
            trial = {identity: {"line": line, "empty": False} for identity, line in zip(ids, combo)}
            try:
                prove_assignment(program, loops, trial, candidates, fors, kernels, mapped, context)
            except ValueError:
                continue
            found.append(combo)
        return found
    combinations = math.prod(len(domain) for domain in domains)
    if stats is not None:
        stats["combinations"] = combinations
    if combinations > MAX_COMBINATIONS:
        raise ValueError(f"{program}: loop role enumeration would try {combinations} combinations "
                         f"(> {MAX_COMBINATIONS}); refusing to truncate")
    valid = enumerate_valid(False)
    if sorted(valid) != sorted(enumerate_valid(True)):
        raise ValueError(f"{program}: loop role enumeration is not confluent (forward and reverse orders differ)")
    if not valid:
        try:  # Name the first violated invariant of the most direct assignment, for the diagnostic.
            if any(not domain for domain in domains):
                identity = ids[[not domain for domain in domains].index(True)]
                controls = sorted(set(next(l for l in roled if l["id"] == identity)["latch_evidence"] +
                                      next(l for l in roled if l["id"] == identity)["entry_evidence"]))
                raise ValueError(f"{program}: loop {identity} is unresolved: no multi-trip mapped source loop among "
                                 f"its latch/entry control locations {controls}")
            prove_assignment(program, loops, {identity: {"line": domain[0], "empty": False}
                                              for identity, domain in zip(ids, domains)},
                             candidates, fors, kernels, mapped, context)
        except ValueError as error:
            raise ValueError(f"no loop role assignment satisfies the invariants: {error}") from None
        raise ValueError(f"{program}: no loop role assignment satisfies the invariants")
    alternatives = []
    for combo in sorted(valid, key=lambda combo: [-int(line in evidence[identity][0])
                                                  for identity, line in zip(ids, combo)] + list(combo)):
        assignment = {}
        for identity, line, domain in zip(ids, combo, domains):
            latch_direct, latch_any, entry_direct = evidence[identity]
            kind = ("latch-header" if line in latch_direct else "latch-body" if line in latch_any else
                    "entry-header" if line in entry_direct else "entry-body")
            if line in induction[identity]:
                kind += "+induction"
            mechanism = "candidate" if len(domain) == 1 else "forced" if len(valid) == 1 else "alternative"
            # C3: an unmultiplied nested copy needs its own latch on the header AND sound induction evidence of it.
            assignment[identity] = {"line": line, **mapped[str(line)], "provenance": f"{kind}/{mechanism}",
                                    "copy_justified": kind == "latch-header+induction" and mechanism != "alternative",
                                    "empty": False}
        for identity in empty:
            if empty[identity]:
                assignment[identity] = dict(EMPTY_ROLE)
        alternatives.append(assignment)
    return alternatives


def prove_assignment(program: str, loops: list[dict], result: dict[int, dict], candidates: dict[int, set[int]],
                     fors: dict[int, dict], kernels: list[dict], mapped: dict[str, dict], context: dict) -> None:
    """The search-independent invariants (a)-(d) of resolve_roles() on a final assignment."""
    parent = {loop["id"]: loop["parent"] for loop in loops}
    assigned = {identity: role["line"] for identity, role in result.items() if not role.get("empty")}
    loops = [loop for loop in loops if loop["id"] in assigned]

    def nested(a: int, b: int) -> bool:
        return a == b or encloses(a, b, fors, kernels, context) or encloses(b, a, fors, kernels, context)
    for identity, line in assigned.items():
        if line not in candidates.get(identity, set()):
            raise ValueError(f"{program}: loop {identity} (S2a) is assigned for statement {line} outside its own "
                             f"control candidates {sorted(candidates[identity])}")
        unrelated = sorted(other for other in candidates[identity] if not nested(other, line))
        if unrelated:
            raise ValueError(f"{program}: loop {identity} (S2a) has conflicting control locations: for statement "
                             f"{line} and unrelated {unrelated}")
        above = assigned.get(parent[identity])
        if above == line:
            continue  # A nested copy of its parent's source loop: dynamic_count decides whether it multiplies.
        feasible = [chain for chain, _ in chains(line, fors, kernels, context=context)]
        if above is not None:
            feasible = [chain for chain in feasible if above in chain[:-1]]
            if not feasible:
                raise ValueError(f"{program}: loop {identity} (S2b) is not source-nested in its machine parent: for "
                                 f"statement {above} does not enclose {line}")
        for chain in feasible:
            start = chain.index(above) + 1 if above is not None else 0
            skipped = [header for header in chain[start:-1] if multi_trip(header, mapped, fors, context)]
            if skipped:
                raise ValueError(f"{program}: loop {identity} (S2c) skips multi-trip source loop(s) {skipped} "
                                 f"between its machine parent and for statement {line}")
    controls = {int(source.rsplit(":", 1)[1]) for loop in loops
                for source in loop["latch_evidence"] + loop["entry_evidence"]
                if source.rsplit(":", 1)[0] == context.get("file", TILE_FILE)}
    on_chains = {header for line in controls for chain, _ in chains(line, fors, kernels, context=context)
                 for header in chain}
    missing = sorted(header for header in on_chains
                     if multi_trip(header, mapped, fors, context) and header not in assigned.values())
    if missing:
        raise ValueError(f"{program}: (S2d) multi-trip source loop(s) {missing} on the program's control chains "
                         f"have no machine loop")


def trip(role: dict, geometry: dict, rows: int, strips: int, policy: int = 0, encoding: int = 0,
         extra: dict | None = None) -> float:
    """Iterations of one source loop per iteration of its enclosing loops, from the role map's trip expression
    over C (columns), K (blocks), P (partitions), S (panel strips), R (rows), G (1 for the GPU-aligned policy),
    E (the program key's encoding; 3 for a coded runCoded program or a compact program), for a compact program W
    (lanes) and T (1 for the ordered partition tree), and ceil()."""
    expression = role["trip"]
    values = {"C": geometry["columns"], "K": geometry["k"], "P": geometry["partitions"], "S": strips, "R": rows,
              "G": policy, "E": encoding}
    values.update({name: (extra or {})[name] for name in COMPACT_TRIP_VARIABLES if name in (extra or {})})
    return float(eval(expression, {"__builtins__": {}, "ceil": math.ceil}, values))  # Expressions come from the hashed role map.


def panel_strips(rule: list[dict], key: str) -> int:
    """The tree's own panelStrips rule, written out in the role map from its source as ordered (condition, S)
    entries over R (rows), L (integer leaf), E (encoding), M (minimum); the first true condition wins."""
    rows, leaf, policy, encoding, _, minimum = (int(field) for field in key.split(":"))
    for entry in rule:
        if eval(entry["when"], {"__builtins__": {}}, {"R": rows, "L": leaf, "E": encoding, "M": minimum, "G": policy}):
            return entry["S"]
    raise ValueError(f"panel strip rule names no width for {key}")


SIDES = {"baseline": min, "candidate": max}


def children_of(loops: list[dict]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {0: []}
    for loop in sorted(loops, key=lambda loop: loop["id"]):
        children.setdefault(loop["id"], [])
        children.setdefault(loop["parent"], []).append(loop["id"])
    return children


def sibling_groups(loops: list[dict], roles: dict[int, dict]) -> list[tuple[int, int, list[int]]]:
    """(parent loop, source line, copies): children of one loop (or of the entry level, 0) mapped to one source for
    statement - versioned or peeled copies of one source loop at the same nesting position."""
    groups = []
    for parent, children in sorted(children_of(loops).items()):
        lines: dict[int, list[int]] = {}
        for child in children:
            if roles[child]["line"] is not None:
                lines.setdefault(roles[child]["line"], []).append(child)
        groups += [(parent, line, copies) for line, copies in sorted(lines.items()) if len(copies) > 1]
    return groups


def subtree_static(loops: list[dict], counts: dict[str, int], root: int) -> int:
    children, total, work = children_of(loops), 0, [root]
    while work:
        node = work.pop()
        total += counts.get(str(node), 0)
        work += children.get(node, [])
    return total


def dynamic_count(loops: list[dict], roles: dict[int, dict], counts: dict[str, int], geometry: dict, rows: int,
                  strips: int, side: str, policy: int = 0, encoding: int = 0, extra: dict | None = None) -> float:
    """Per-call executions of one class: static count x product of the distinct source loops' trips on the loop
    chain (a nested copy of an ancestor's source loop is not multiplied again on the baseline side; on the
    candidate side only when copy_justified - its own latch names the for header directly - else it is), summed over the loop tree, except that
    sibling copies of one source loop (sibling_groups) are combined per parent iteration by the side's rule:
    baseline = minimum over the copies, candidate = maximum (strict: only false failures, never false passes)."""
    combine = SIDES[side]
    children = children_of(loops)

    def total(node: int, multiplier: float, used: frozenset) -> float:
        if node:
            role = roles[node]
            if role["line"] not in used:
                used = used | {role["line"]}
                multiplier *= trip(role, geometry, rows, strips, policy, encoding, extra)
            elif side == "candidate" and not role.get("copy_justified", False):
                # A nested copy of an ancestor's source loop runs each source iteration at most once only when
                # its own latch names that for's header directly; otherwise the candidate pays its full trip.
                multiplier *= trip(role, geometry, rows, strips, policy, encoding, extra)
        result = counts.get(str(node), 0) * multiplier
        lines: dict[int, list[int]] = {}
        for child in children.get(node, []):
            line = roles[child]["line"]
            lines.setdefault(line if line is not None else ("empty", child), []).append(child)
        for copies in lines.values():
            result += combine(total(child, multiplier, used) for child in copies)
        return result

    return total(0, 1.0, frozenset())


def helper_role(loop: dict, entry: dict, record: dict, helper_sources: dict[str, str]) -> dict:
    """The role of a helper loop (call model): the role map's helper entry {raw helper symbol: {"source": path,
    "source_sha256": sha, "loops": {helper loop id: {"line", "role", "variable", "trip"}}}} must be the entry the
    checker admitted the call under, its source must be the compiled one (the record's closure hash) and the
    supplied text, and the line must be a `for` header over the role's variable there. Its trip multiplies under
    the call site's loop chain (caller chain x helper chain)."""
    helper = entry.get("helpers", {}).get(loop["helper"])
    role = helper["loops"].get(str(loop["helper_loop"])) if helper else None
    if role is None or role != loop.get("helper_role"):
        raise ValueError(f"helper loop {loop['helper']}#{loop['helper_loop']} has no matching role map entry")
    text = helper_sources.get(helper["source"])
    if text is None or digest(text.encode()) != helper["source_sha256"] or \
            record["tree_inputs"].get(helper["source"]) != helper["source_sha256"]:
        raise ValueError(f"helper source {helper['source']} is not supplied or not the compiled version")
    statement = parse_fors(text).get(int(role["line"]))
    if statement is None or role["role"] not in ROLES or statement["variable"] != role["variable"]:
        raise ValueError(f"helper role line {role['line']} is not a for statement over {role['variable']!r}")
    return {"line": f"{loop['helper']}:{role['line']}", "role": role["role"], "variable": role["variable"],
            "trip": role["trip"], "provenance": "helper-role-map", "copy_justified": False, "empty": False}


def normalise(text: str) -> str:
    """Source text with comments removed and whitespace runs collapsed (spaces kept only between word
    characters), for exact pattern comparison."""
    text = re.sub(r"//[^\n]*|/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s*([^\w\s])\s*", r" \1 ", text).replace("  ", " ").strip()


def reduction_values(text: str) -> dict[str, int]:
    """The enumerator values of the compact source's `enum class Reduction` (implicit sequential values only; an
    explicit initialiser, a missing or repeated enum or another enumerator set fails)."""
    found = REDUCTION_ENUM.findall(re.sub(r"//[^\n]*|/\*.*?\*/", " ", text, flags=re.S))
    if len(found) != 1:
        raise ValueError(f"compact source holds {len(found)} Reduction enums, not exactly one")
    names = [name.strip() for name in found[0].split(",") if name.strip()]
    if any(not re.fullmatch(r"[A-Za-z_]\w*", name) for name in names) or \
            sorted(names) != ["Contiguous", "OrderedPartitions"]:
        raise ValueError(f"Reduction enumerators {names} are not exactly Contiguous and OrderedPartitions with "
                         f"implicit values")
    return {name: index for index, name in enumerate(names)}


def compact_entry(text: str) -> dict:
    """The compact role map entry of one tree, written mechanically from its compact source: the one compact
    kernel (COMPACT_KERNELS, exactly one present and unique) and each of its `for` statements by loop variable
    (COMPACT_VARIABLES); a row loop under an explicit `#pragma GCC unroll N` (N >= 4 = the maximum rows) is
    declared unrolled. Any other `for` in the kernel fails: the map must be extended by review, never guessed."""
    present = [kernel for kernel in COMPACT_KERNELS if kernel in text]
    if not present:
        raise ValueError("compact source names no compact kernel")
    # The PR's rows() only selects the rowsProgram<OrderedNaN> instantiation: it must hold no for statement.
    extents = function_extents(text, [{"signature": kernel} for kernel in present])
    fors = parse_fors(text)
    for kernel, wrapper in zip(present[1:], extents[1:]):
        if any(wrapper["first"] <= line <= wrapper["last"] for line in fors):
            raise ValueError(f"compact wrapper {kernel!r} holds a for statement")
    extent = extents[0]
    lines = text.split("\n")
    body = "\n".join(lines[extent["first"] - 1:extent["last"]])
    loops = {}
    for line, statement in sorted(fors.items()):
        if not extent["first"] <= line <= extent["last"]:
            continue
        pattern = COMPACT_PATTERNS.get(statement["variable"])
        if pattern is None:
            raise ValueError(f"compact kernel for statement {line} over {statement['variable']!r} has no mechanical role")
        source = "\n".join(lines[line - 1:statement["header_end"]])
        opening = source.index("for")
        header = re.match(r"for\s*\((.*)\)", source[opening:], flags=re.S)
        if header is None or normalise(header.group(1)) != normalise(pattern["header"]):
            raise ValueError(f"compact for statement {line} header {normalise(header.group(1) if header else source)!r} "
                             f"is not the admitted pattern {pattern['header']!r}")
        definitions = {}
        for name, expected in pattern["definitions"].items():
            # Exactly one definition `const int name = ...;` and no other assignment in the kernel.
            found = re.findall(rf"\bconst\s+int\s+{name}\s*=(?!=)\s*([^;]*);", body)
            assigned = re.findall(rf"\b{name}\s*(?:[-+*/%&|^]|<<|>>)?=(?!=)", body)
            if len(found) != 1 or len(assigned) != 1 or normalise(found[0]) != normalise(expected):
                raise ValueError(f"compact for statement {line}: the definition of {name} "
                                 f"{[normalise(text) for text in found]} is not exactly {expected!r}")
            definitions[name] = expected
        role = {"role": pattern["role"], "variable": statement["variable"], "trip": pattern["trip"],
                "header": pattern["header"], "definitions": definitions}
        if pattern["role"] == "row" and (statement["unroll"] or 0) >= 4:
            role["unrolled"] = "True"
        loops[str(line)] = role
    return {"source": COMPACT_FILE, "file_sha256": digest(text.encode()), "kernel_functions": [{"signature": present[0]}],
            "loops": loops, "reduction": reduction_values(text)}


def source_spec(spec: dict, records: list[dict], sources: dict[str, str], what: str) -> tuple[dict, list, dict]:
    """(fors, kernels, loops) of a role map source entry, validated: the supplied text is the hashed one, every
    record compiled it, every mapped line is a `for` header over its variable."""
    text = sources.get(spec["source"])
    if text is None or digest(text.encode()) != spec["file_sha256"]:
        raise ValueError(f"{what} source {spec['source']} is not supplied or not the mapped version")
    if any(record["tree_inputs"].get(spec["source"]) != spec["file_sha256"] for record in records):
        raise ValueError(f"records were compiled from another version of {spec['source']}")
    if what == "compact" and compact_entry(text) != spec:
        # S2-1/S2-2: the committed entry must be exactly the mechanical one (headers, bound definitions,
        # trips, Reduction enumerators) of the source the records compiled.
        raise ValueError("the compact role map entry is not the mechanical entry of its source")
    fors = parse_fors(text)
    for line, role in spec["loops"].items():
        statement = fors.get(int(line))
        if statement is None or role["role"] not in ROLES or statement["variable"] != role["variable"]:
            raise ValueError(f"{what} role map line {line} is not a for statement over {role['variable']!r}")
    return fors, function_extents(text, spec["kernel_functions"]), spec["loops"]


def compact_context(key: str, reduction: dict[str, int]) -> dict:
    """Context of a compact program key compact:codebook:lanes:rows:tree:ordered:dot[@copy] (ordered "-":
    develop's policy-free program; copy: the TU or link-selected copy). The key's tree field is the mangled
    Reduction enumerator value: T is 1 when it is the source's OrderedPartitions value, 0 for Contiguous; any
    other value fails (S2-2: derived from source, never assumed)."""
    _, codebook, lanes, rows, tree, ordered, dot = key.split("@", 1)[0].split(":")
    by_value = {value: name for name, value in reduction.items()}
    if int(tree) not in by_value:
        raise ValueError(f"{key}: tree value {tree} names no Reduction enumerator {reduction}")
    return {"R": int(rows), "W": int(lanes), "T": int(by_value[int(tree)] == "OrderedPartitions"),
            "G": 0 if ordered == "-" else int(ordered), "E": 3, "C": int(codebook), "L": int(dot), "S": 1,
            "file": COMPACT_FILE}


BOUNDED_CLASSES = ("ext_call", "ext_bytes", "ext_accesses")


def site_charges(site: dict, bytes_: int) -> dict[str, int]:
    """Stack-side charges of one admitted memcpy site at a byte count (D2: ceil(bytes/8) accesses per side)."""
    return {"ext_call": 1, "ext_bytes": bytes_ * site["sides"], "ext_accesses": -(-bytes_ // 8) * site["sides"]}


def place_bounded_sites(loops: list[dict], counts: dict, sites: list[dict], alternatives: list[dict]) -> \
        tuple[dict, list[dict]]:
    """M-3 (reviewer 91): an admitted column-tail memcpy (the tail branch is true on at most one column iteration,
    proven from its source by the checker) is charged ONCE per entry of the column loop: its upper-bound charges
    move from the machine loop holding it to that loop's parent when every role assignment names that loop a
    column loop; a site in any other loop (e.g. a kb loop) stays charged at its loop. Returns the adjusted counts
    and the placed sites (loop after placement, static upper-minus-lower deltas)."""
    counts = {cls: dict(values) for cls, values in counts.items()}
    parents = {loop["id"]: loop["parent"] for loop in loops}
    placed = []
    for site in sites:
        loop = site["loop"]
        kinds = {alternative[loop]["role"] for alternative in alternatives} if loop else {"entry"}
        target = loop
        if site["placement"] == "column-tail" and loop:
            if len(kinds) != 1:
                raise ValueError(f"bounded memcpy at {site['line']}: role assignments disagree on its loop {loop}")
            if next(iter(kinds)) == "column":
                target = parents[loop]
                for cls, amount in site_charges(site, site["upper_bytes"]).items():
                    counts[cls][str(loop)] = counts[cls].get(str(loop), 0) - amount
                    if counts[cls][str(loop)] < 0:
                        raise ValueError(f"bounded memcpy at {site['line']}: its loop holds less than its charge")
                    if not counts[cls][str(loop)]:
                        del counts[cls][str(loop)]
                    counts[cls][str(target)] = counts[cls].get(str(target), 0) + amount
        upper, lower = site_charges(site, site["upper_bytes"]), site_charges(site, site["lower_bytes"])
        placed.append({"line": site["line"], "text": site["text"], "loop": target, "held_by": loop,
                       "lower_bytes": site["lower_bytes"], "upper_bytes": site["upper_bytes"], "sides": site["sides"],
                       "delta": {cls: upper[cls] - lower[cls] for cls in BOUNDED_CLASSES if upper[cls] - lower[cls]}})
    return counts, placed


def table(records: list[dict], role_map: dict, source: str, side: str = "baseline",
          helper_sources: dict[str, str] | None = None) -> dict:
    tree = records[0]["tree_sha"]
    if any(record["tree_sha"] != tree or record["checker_sha256"] != records[0]["checker_sha256"]
           for record in records):
        raise ValueError("records of different trees or checkers")
    entry = role_map.get(tree)
    if entry is None:
        # A candidate tree (gate run) is identified by its tile source: exactly one entry must name it.
        matching = [value for value in role_map.values() if value["file_sha256"] == digest(source.encode())]
        if len(matching) != 1:
            raise ValueError(f"role map has no entry for tree {tree} or its tile source")
        entry = matching[0]
    if entry["file_sha256"] != digest(source.encode()):
        raise ValueError("role map was written for another version of the tile source")
    if any(record["tree_inputs"].get(TILE_FILE) != entry["file_sha256"] for record in records):
        raise ValueError("records were compiled from another version of the tile source")
    fors = parse_fors(source)
    kernels = function_extents(source, entry["kernel_functions"])
    for line, role in entry["loops"].items():
        statement = fors.get(int(line))
        if statement is None or role["role"] not in ROLES or statement["variable"] != role["variable"]:
            raise ValueError(f"role map line {line} is not a for statement over {role['variable']!r}")
    sources = dict(helper_sources or {})
    compact = None
    if any(key.startswith("compact:") for record in records for key in record["stack_traffic"]):
        if "compact" not in entry:
            raise ValueError("records hold compact programs but the role map entry has no compact entry")
        compact = source_spec(entry["compact"], records, sources, "compact")
    helper_trips: dict[str, dict] = {}
    programs, largest = {}, 1
    for record in records:
        for key, traffic in record["stack_traffic"].items():
            unknown = set(traffic["counts"]) - set(CLASSES)
            if unknown:
                raise ValueError(f"{key}: record carries unknown traffic classes {sorted(unknown)}")
            missing = set(CLASSES) - set(traffic["counts"])
            if missing and side != "baseline":
                # A candidate is always measured by the current checker: a missing class is no zero.
                raise ValueError(f"{key}: candidate record lacks traffic classes {sorted(missing)}")
            # Baseline only (strict for the comparison: zero can only fail a candidate): a class the record's
            # checker did not yet count reads as zero everywhere.
            traffic = {**traffic, "counts": {cls: traffic["counts"].get(cls, {}) for cls in CLASSES}}
            stats: dict = {}
            own = [loop for loop in traffic["loops"] if "helper" not in loop]
            helpers = {loop["id"]: helper_role(loop, entry, record, helper_sources or {})
                       for loop in traffic["loops"] if "helper" in loop}
            helper_trips.update({role["line"]: {"role": role["role"], "trip": role["trip"]} for role in helpers.values()})
            if key.startswith("compact:"):
                context = compact_context(key, entry["compact"]["reduction"])
                rows, strips, policy, encoding = context["R"], 1, context["G"], 3
                program_fors, program_kernels, program_loops = compact
            else:
                # A production tile key may carry its copy (@TU or @link-selected copy, P4): not a key field.
                plain = key.split("@", 1)[0]
                strips = panel_strips(entry["panel_strips"], plain)
                context = program_context(plain, strips)
                rows, policy, encoding = int(plain.split(":")[0]), int(plain.split(":")[2]), int(plain.split(":")[3])
                program_fors, program_kernels, program_loops = fors, kernels, entry["loops"]
            alternatives = [{**alternative, **helpers} for alternative in
                            resolve_roles(key, own, program_fors, program_loops, program_kernels, context, stats)]
            largest = max(largest, stats["combinations"])
            roles = alternatives[0]
            placed_counts, placed = place_bounded_sites(traffic["loops"], traffic["counts"], traffic.get("bounded", []),
                                                        alternatives)
            traffic = {**traffic, "counts": placed_counts}
            for identity, role in roles.items():
                if role["empty"] and any(traffic["counts"][cls].get(str(identity)) for cls in CLASSES):
                    raise ValueError(f"{key}: empty self-loop {identity} carries stack traffic")
            parents = {loop["id"]: loop["parent"] for loop in traffic["loops"]}
            names, seen = {}, {}
            for loop in sorted(traffic["loops"], key=lambda loop: loop["id"]):
                role = roles[loop["id"]]["role"]
                names[loop["id"]] = f"{role}#{seen.get(role, 0)}"
                seen[role] = seen.get(role, 0) + 1
            per_role = {cls: {} for cls in CLASSES}
            for cls in CLASSES:
                for loop, count in traffic["counts"][cls].items():
                    name = "entry" if loop == "0" else names[int(loop)]
                    per_role[cls][name] = per_role[cls].get(name, 0) + count
            # Several valid role assignments: the side's rule over them (baseline minimum, candidate maximum).
            dynamic = {geometry["label"]: {cls: round(SIDES[side](
                dynamic_count(traffic["loops"], alternative, traffic["counts"][cls], geometry, rows, strips, side,
                              policy, encoding, context) for alternative in alternatives), 6)
                for cls in CLASSES} for geometry in GEOMETRIES}
            for site in placed:
                # The site's upper-minus-lower charge per geometry (M-4: the gate takes a baseline site to its
                # minimum when the candidate's text differs), with the side's rule over role assignments.
                site["role"] = "entry" if site["loop"] == 0 else names[site["loop"]]
                site["dynamic"] = {geometry["label"]: {cls: round(SIDES[side](
                    dynamic_count(traffic["loops"], alternative, {str(site["loop"]): amount}, geometry, rows, strips,
                                  side, policy, encoding, context) for alternative in alternatives), 6)
                    for cls, amount in site["delta"].items()} for geometry in GEOMETRIES}
            siblings = sibling_groups(traffic["loops"], roles)
            programs[f"{record['mode']}/{key}"] = {
                "loops": [{"id": loop["id"], "parent": loop["parent"], "role": names[loop["id"]],
                           "source_line": roles[loop["id"]]["line"], "provenance": roles[loop["id"]]["provenance"],
                           "copy_justified": roles[loop["id"]]["copy_justified"]} for loop in traffic["loops"]],
                "per_loop": traffic["counts"], "per_role": per_role, "panel_strips": strips, "dynamic": dynamic,
                "repeated_roles": sorted(role for role, count in seen.items() if count > 1),
                "fp_spills": record["fp_spills"][key],
                "alternatives": [{str(identity): [role["line"], role["copy_justified"]]
                                  for identity, role in sorted(alternative.items()) if not role["empty"]}
                                 for alternative in alternatives[1:]],
                "empty_loops": sorted(loop["empty_source"] for loop in traffic["loops"] if loop.get("empty")),
                "role_combinations": stats["combinations"],
                **({"bounded_sites": placed} if placed else {}),
                "sibling_copies": [{"parent": parent, "source_line": line, "copies": [names[c] for c in copies],
                                    "classes": sorted({cls for cls in CLASSES for c in copies
                                                       if subtree_static(traffic["loops"], traffic["counts"][cls], c)})}
                                   for parent, line, copies in siblings],
            }
    return {
        "schema": 2, "tree_sha": tree, "sibling_rule": side, "max_role_combinations": largest,
        "header": ("Classes overlap and are never summed; each class is compared separately against the same class. "
                   "Limits (same semantics on every tree): a register redefined from itself plus a constant within "
                   "one iteration counts as variant (invariant_load undercount); a pointer in a descriptor field the "
                   "loop rewrites counts as invariant (invariant_load overcount; the rewrite is an alias_store). "
                   "Dynamic counts are static counts times source trip products per role chain: a nested copy of "
                   "an ancestor's source loop is not multiplied again; sibling copies of one source loop at one "
                   "nesting position (versioned or peeled copies) are combined per parent iteration strictly: the "
                   "baseline takes the minimum over the copies, a candidate the maximum (false failures only; a "
                   "failing key may be refined only by recorded CFG ownership: guard-matched alternatives or "
                   "sequential copies partitioning one trip); entry-level counts execute once per call. "
                   "Loop roles: every machine loop's source loop comes from its control evidence (latch, entry, induction) and "
                   "every assignment satisfying the search-independent invariants (evidence, embedding, no skipped "
                   "level, no missing multi-trip loop) is enumerated; several valid assignments are combined by the "
                   "side's rule (baseline minimum, candidate maximum) and disable static dominance; a nested copy of "
                   "an ancestor's source loop is not re-multiplied on the candidate side only when its own latch "
                   "names that header directly in a non-alternative assignment. Each loop records its provenance. "
                   "Reproduction: from a 63b053401 checkout in the pinned image, CPUNativeTileStackBaseline.py per "
                   "envelope (records, not committed), then this tool with --expect."),
        "checker_sha256": records[0]["checker_sha256"],
        "records": {record["mode"]: {name: record[name] for name in (
            "generator_sha256", "admission_sha256", "probe_sha256", "closure_sha256", "object_sha256",
            "assembly_sha256", "image", "identity", "compiler_version", "disassembler_version",
            "target_command_sha256")} for record in records},
        "role_map_sha256": digest(json.dumps(entry, sort_keys=True).encode()),
        "tile_source_sha256": entry["file_sha256"],
        "geometries": GEOMETRIES,
        "trips": {line: {"role": role["role"], "trip": role["trip"]} for line, role in entry["loops"].items()},
        **({"compact_trips": {line: {"role": role["role"], "trip": role["trip"]}
                              for line, role in entry["compact"]["loops"].items()},
            "compact_source_sha256": entry["compact"]["file_sha256"],
            "compact_reduction": entry["compact"]["reduction"]} if compact else {}),
        **({"helper_trips": dict(sorted(helper_trips.items()))} if helper_trips else {}),
        "programs": programs,
    }


KEY = re.compile(r"^(NEON|NEON_DOTPROD|NEON_I8MM)/(?:[1-4]:[0-2]:[01]:[0-9]:(?:[0-3]|7|11):[01]|"
                 r"compact:\d+:(?:4|8|16):[1-4]:[01]:[01-]:[0-2])(?:@[A-Za-z0-9_.+-]+)?$")
HASHES = ("generator_sha256", "admission_sha256", "probe_sha256", "closure_sha256", "object_sha256",
          "assembly_sha256", "target_command_sha256")


def check(result: dict) -> None:
    """Internal consistency of a table without its records (the CI check): every hash present, every program
    keyed, every loop role valid and its source line a mapped for statement with that role, indices dense in loop
    order, per_role equal to per_loop regrouped, dynamic counts equal to those recomputed from the per-loop counts
    and the recorded trips under the table's sibling rule."""
    for name in ("checker_sha256", "role_map_sha256", "tile_source_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(result.get(name, ""))):
            raise ValueError(f"table lacks {name}")
    if result.get("sibling_rule") not in SIDES or not result.get("records"):
        raise ValueError("table lacks its sibling rule or records")
    for mode, record in result["records"].items():
        if any(not re.fullmatch(r"[0-9a-f]{64}", str(record.get(name, ""))) for name in HASHES):
            raise ValueError(f"record {mode} lacks a hash")
    tile_trips, helper_trips = result["trips"], result.get("helper_trips", {})
    geometries = {geometry["label"]: geometry for geometry in result["geometries"]}
    if [g["label"] for g in result["geometries"]] != [g["label"] for g in GEOMETRIES]:
        raise ValueError("table geometries differ from the declared reference geometries")
    for key, program in result["programs"].items():
        if not KEY.match(key) or key.split("/")[0] not in result["records"]:
            raise ValueError(f"malformed program key {key}")
        compact = key.split("/")[1].startswith("compact:")
        if compact and ("compact_trips" not in result or "compact_reduction" not in result):
            raise ValueError(f"{key}: a compact program without the table's compact trips and Reduction values")
        trips = {**(result["compact_trips"] if compact else tile_trips), **helper_trips}
        roles, seen = {}, {}
        for loop in program["loops"]:
            role, _, index = loop["role"].partition("#")
            if role == "empty":
                if loop["source_line"] is not None or loop["provenance"] != "empty-self-loop" or \
                        index != str(seen.get(role, 0)) or any(other["parent"] == loop["id"] for other in program["loops"]) \
                        or any(program["per_loop"][cls].get(str(loop["id"])) for cls in CLASSES):
                    raise ValueError(f"{key}: loop {loop['id']} is not a valid empty self-loop")
                seen[role] = seen.get(role, 0) + 1
                roles[loop["id"]] = dict(EMPTY_ROLE)
                continue
            if loop["copy_justified"] != (loop["provenance"].startswith("latch-header+induction/")
                                          and not loop["provenance"].endswith("/alternative")):
                raise ValueError(f"{key}: loop {loop['id']} copy justification contradicts its provenance")
            mapped = trips.get(str(loop["source_line"]))
            if role not in ROLES or mapped is None or mapped["role"] != role or index != str(seen.get(role, 0)):
                raise ValueError(f"{key}: loop {loop['id']} has an invalid role {loop['role']}")
            seen[role] = seen.get(role, 0) + 1
            roles[loop["id"]] = {"line": loop["source_line"], **mapped, "copy_justified": loop["copy_justified"]}
        names = {str(loop["id"]): loop["role"] for loop in program["loops"]}
        for cls in CLASSES:
            regrouped: dict[str, int] = {}
            for loop, count in program["per_loop"][cls].items():
                name = "entry" if loop == "0" else names[loop]
                regrouped[name] = regrouped.get(name, 0) + count
            if regrouped != program["per_role"][cls]:
                raise ValueError(f"{key}: per_role {cls} is not per_loop regrouped")
        if compact:
            context = compact_context(key.split("/")[1], result["compact_reduction"])
            rows, policy, encoding = context["R"], context["G"], 3
        else:
            context = None
            plain = key.split("/")[1].split("@", 1)[0].split(":")
            rows, policy, encoding = int(plain[0]), int(plain[2]), int(plain[3])
        alternatives = [roles]
        for alternative in program["alternatives"]:
            if set(alternative) != {str(identity) for identity, role in roles.items() if not role.get("empty")}:
                raise ValueError(f"{key}: an alternative assignment does not cover every loop")
            other = {identity: role for identity, role in roles.items() if role.get("empty")}
            for identity, (line, justified) in alternative.items():
                mapped = trips.get(str(line))
                if mapped is None:
                    raise ValueError(f"{key}: an alternative names an unmapped for statement {line}")
                other[int(identity)] = {"line": line, **mapped, "copy_justified": justified}
            alternatives.append(other)
        for label, counts in program["dynamic"].items():
            for cls in CLASSES:
                expected = round(SIDES[result["sibling_rule"]](
                    dynamic_count(program["loops"], alternative, program["per_loop"][cls], geometries[label], rows,
                                  program["panel_strips"], result["sibling_rule"], policy, encoding, context)
                    for alternative in alternatives), 6)
                if counts[cls] != expected:
                    raise ValueError(f"{key}: dynamic {label} {cls} {counts[cls]} != recomputed {expected}")


def render(result: dict) -> str:
    """Stable text: sorted keys, one program per line (reviewable diffs)."""
    programs = result["programs"]
    head = json.dumps({key: value for key, value in result.items() if key != "programs"}, sort_keys=True, indent=1)
    lines = [f" {json.dumps(name)}: {json.dumps(programs[name], sort_keys=True, separators=(',', ':'))}"
             for name in sorted(programs)]
    return head[:-2] + ',\n "programs": {\n' + ",\n".join(lines) + "\n }\n}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", help="only check a table's internal consistency (no records needed)")
    parser.add_argument("--records", nargs="+")
    parser.add_argument("--roles")
    parser.add_argument("--source", help="the tree's tile source file (CPUNativeNEONPreparedProjection.h)")
    parser.add_argument("--helper-source", action="append", default=[], metavar="PATH=FILE",
                        help="a helper source file (role map path = file to read), for helper loops")
    parser.add_argument("--compact-entry", metavar="FILE",
                        help="print the mechanical compact role map entry of a compact source file and exit")
    parser.add_argument("--out")
    parser.add_argument("--side", choices=sorted(SIDES), default="baseline")
    parser.add_argument("--expect", help="committed table: fail unless the regenerated table is byte-identical")
    args = parser.parse_args()
    if args.compact_entry:
        print(json.dumps(compact_entry(Path(args.compact_entry).read_text()), sort_keys=True, indent=1))
        return
    if args.check:
        check(json.loads(Path(args.check).read_text()))
        print(f"table {args.check}: internally consistent")
        return
    if not (args.records and args.roles and args.source and args.out):
        parser.error("--records, --roles, --source and --out are required to build a table")
    records = [json.loads(Path(path).read_text()) for path in args.records]
    helper_sources = {path: Path(file).read_text() for path, file in
                      (item.split("=", 1) for item in args.helper_source)}
    result = table(records, json.loads(Path(args.roles).read_text()), Path(args.source).read_text(), args.side,
                   helper_sources)
    result["records_sha256"] = {json.loads(Path(path).read_text())["mode"]: digest(Path(path).read_bytes())
                                for path in args.records}
    check(result)
    text = render(result)
    Path(args.out).write_text(text)
    if args.expect and Path(args.expect).read_text() != text:
        raise SystemExit(f"regenerated table differs from the committed {args.expect} (tree, toolchain, checker, "
                         "role map or record hashes changed)")
    print(f"table {result['tree_sha']}: {len(result['programs'])} programs")


if __name__ == "__main__":
    main()
