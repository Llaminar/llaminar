#!/usr/bin/env python3
"""Compiler-free tests of the stack-traffic table tool: source for-statement parsing, the fail-closed role
assignment (kernel functions, inlined call sites, unmapped and uncommon loops), repeated-role indexing, role-map
validation, the panel strip rule and per-geometry dynamic counts, regeneration against a committed table, and every
committed compact role-map entry re-derived from its authenticated (worktree or pinned historical) source."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[2] / "cmake" / "CPUNativeTileStackTable.py"
SPEC = importlib.util.spec_from_file_location("cpu_native_tile_stack_table", TOOL)
TABLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TABLE)
FILE = TABLE.TILE_FILE

SOURCE = """struct Tile {
    void helper() const noexcept
    {
        for (int row = 0; row < 4; ++row)
            work(row);
    }
    void run() const noexcept
    {
        for (int column = 0; column < columns_; column += 4)
        {
            for (int partition = 0; partition < partitions_; ++partition)
            {
                for (int block = first; block < last; ++block)
                {
                    work(block);
                }
                fold();
            }
            store();
        }
        for (int column = 0; column < columns_; column += 8)
            walk<S>(column);
    }
    inline void walk(int column0) const noexcept
    {
        for (int block = 0; block < blocks; ++block)
            step(block);
    }
};
"""
KERNELS = [{"signature": "void run() const noexcept"},
           {"signature": "inline void walk(int column0)", "call": "walk<S>(column);"}]
ROLES = {"9": {"role": "column", "variable": "column", "trip": "C / 4"},
         "11": {"role": "partition", "variable": "partition", "trip": "P"},
         "13": {"role": "block", "variable": "block", "trip": "K / P"},
         "21": {"role": "column", "variable": "column", "trip": "C / 8"},
         "26": {"role": "block", "variable": "block", "trip": "K"}}


def loop(identity: int, parent: int, latch: tuple = (), entry: tuple = (), depth: int = 1,
         members: tuple = (), induction: tuple = (), empty: bool = False) -> dict:
    """A machine loop of a fixture record: its control evidence (latch, entry and induction .loc lines of the
    tile file, or 'file:line' strings) and, irrelevant to its role, the locations of its body."""
    def spell(lines: tuple) -> list[str]:
        return [line if isinstance(line, str) else f"{FILE}:{line}" for line in lines]
    return {"id": identity, "parent": parent, "depth": depth, "entry_sources": [], "latch_sources": [],
            "latch_evidence": spell(latch), "entry_evidence": spell(entry), "induction_evidence": spell(induction),
            "member_sources": spell(members), "empty": empty}


CONTEXT = {"R": 1, "L": 1, "G": 1, "E": 0, "C": 0, "S": 1}


def nest(*extra: dict) -> list[dict]:
    """The fixture's column (9) > partition (11) > block (13) machine nest, plus extra loops."""
    return [loop(1, 0, (9,), (11,), 1), loop(2, 1, (11,), (13,), 2, induction=(11,)),
            loop(3, 2, (13,), (15,), 3, induction=(13,))] + list(extra)


class RoleAssignmentTest(unittest.TestCase):
    """Roles come from control evidence; every returned assignment satisfies the search-independent invariants
    (S2 a-d), all valid ones are returned, and nothing depends on enumeration order."""

    def setUp(self) -> None:
        self.fors = TABLE.parse_fors(SOURCE)
        self.kernels = TABLE.function_extents(SOURCE, KERNELS)

    def alternatives(self, loops: list[dict], mapped: dict | None = None, context: dict | None = None,
                     source: str | None = None) -> list[dict]:
        fors = TABLE.parse_fors(source) if source else self.fors
        return TABLE.resolve_roles("p", loops, fors, ROLES if mapped is None else mapped, self.kernels,
                                   context or CONTEXT)

    def assign(self, loops: list[dict], **options) -> dict:
        found = self.alternatives(loops, **options)
        self.assertEqual(len(found), 1, found)
        return {identity: role["line"] for identity, role in found[0].items()}

    def test_for_statements_and_extents(self) -> None:
        self.assertEqual({line: entry["variable"] for line, entry in self.fors.items()},
                         {4: "row", 9: "column", 11: "partition", 13: "block", 21: "column", 26: "block"})
        self.assertEqual((self.fors[9]["end"], self.fors[13]["end"], self.fors[21]["end"]), (20, 16, 22))
        self.assertEqual(self.fors[13]["header_end"], 13)
        self.assertEqual([(k["first"], k["last"], k["call_sites"]) for k in self.kernels], [(8, 23, []), (25, 28, [22])])

    def test_back_edge_headers_name_the_loops_with_provenance(self) -> None:
        (found,) = self.alternatives(nest())
        self.assertEqual({i: (r["line"], r["provenance"]) for i, r in found.items()},
                         {1: (9, "latch-header/forced"), 2: (11, "latch-header+induction/candidate"),
                          3: (13, "latch-header+induction/candidate")})

    def test_body_locations_never_decide_the_role(self) -> None:
        loops = nest()
        loops[0] = loop(1, 0, (9,), (19,), 1, members=(9, 19, 13, 15, 26, 27, 4, "arm_neon.h:338"))
        self.assertEqual(self.assign(loops), {1: 9, 2: 11, 3: 13})

    def test_rotated_bottom_test_and_strength_reduced_induction(self) -> None:
        loops = nest()
        loops[2] = loop(3, 2, (15,), (15,), 3)  # compare carries a body address computation
        (found,) = self.alternatives(loops)
        self.assertEqual((found[3]["line"], found[3]["provenance"]), (13, "latch-body/candidate"))

    def test_induction_decides_a_nested_copy(self) -> None:
        # A partition sub-cycle whose back edge also carries the next block range's zero-trip test (13): the
        # partition increment on its cycle makes it a partition copy, justified (its latch names 11 directly).
        (found,) = self.alternatives(nest(loop(4, 2, (13, 11), (13,), 3, induction=(11,))))
        self.assertEqual((found[4]["line"], found[4]["copy_justified"]), (11, True))
        # Without the induction evidence both readings are valid: both are returned (no order-dependent pick).
        both = self.alternatives(nest(loop(4, 2, (13, 11), (13,), 3)))
        self.assertEqual(sorted(alternative[4]["line"] for alternative in both), [11, 13])
        self.assertTrue(all(alternative[4]["provenance"].endswith("/alternative") and
                            not alternative[4]["copy_justified"] for alternative in both))

    def test_non_monotone_configuration_yields_every_valid_reading(self) -> None:
        # S1: Y (ancestor X on 11) names {11, 13}; its child Z names 13. A set-aside search could give Y 13 and
        # later move it to 11; the enumeration returns both readings, and the candidate side pays the maximum.
        loops = [loop(1, 0, (9,), (11,), 1), loop(2, 1, (11,), (13,), 2, induction=(11,)),
                 loop(3, 2, (11, 13), (13,), 3), loop(4, 3, (13,), (15,), 4, induction=(13,))]
        found = self.alternatives(loops)
        self.assertEqual(sorted((a[3]["line"], a[4]["line"]) for a in found), [(11, 13), (13, 13)])
        counts = {"4": 1}
        geometry = {"columns": 64, "k": 17, "partitions": 2}
        values = [TABLE.dynamic_count(loops, a, counts, geometry, 1, 1, "candidate") for a in found]
        # (11, 13): Y is an unjustified partition copy, multiplied (16 * 2 * 2 * 8.5); (13, 13): Z's own latch names
        # 13 directly, a justified copy (16 * 2 * 8.5). The candidate pays the larger reading.
        self.assertEqual(values, [16 * 2 * 2 * 8.5, 16 * 2 * 8.5] if found[0][3]["line"] == 11 else [16 * 2 * 8.5, 16 * 2 * 2 * 8.5])
        self.assertEqual(max(values), 544)

    def test_a_tie_break_violating_the_invariants_fails(self) -> None:
        # Mutually waiting partition/block nest without the column loop: every reading skips or misses the
        # multi-trip column loop (S2c/S2d).
        with self.assertRaisesRegex(ValueError, "no loop role assignment satisfies the invariants.*S2"):
            self.alternatives([loop(1, 0, (13, 17), (11,), 1), loop(2, 1, (13,), (12,), 2)])

    def test_same_for_copy_justified_only_by_its_own_latch_header(self) -> None:
        geometry = {"columns": 64, "k": 17, "partitions": 2}
        # C3: unmultiplied only with latch-header evidence AND sound induction of that header.
        for latch, entry, induction, justified in (((11,), (11,), (11,), True), ((11,), (11,), (), False),
                                                   ((17,), (11,), (11,), False), (("arm_neon.h:1", 17), (11,), (), False)):
            with self.subTest(latch=latch, induction=induction):
                loops = nest(loop(4, 2, latch, entry, 3, induction=induction))
                (found,) = self.alternatives(loops)
                self.assertEqual((found[4]["line"], found[4]["copy_justified"]), (11, justified))
                baseline = TABLE.dynamic_count(loops, found, {"4": 1}, geometry, 1, 1, "baseline")
                candidate = TABLE.dynamic_count(loops, found, {"4": 1}, geometry, 1, 1, "candidate")
                self.assertEqual(baseline, 16 * 2)
                self.assertEqual(candidate, 16 * 2 * (1 if justified else 2))

    def test_missing_multi_trip_loop_fails_unless_a_validated_unroll(self) -> None:
        loops = [loop(1, 0, (9,), (11,), 1), loop(3, 1, (13,), (15,), 2, induction=(13,))]
        with self.assertRaisesRegex(ValueError, "S2c|S2d"):
            self.alternatives(loops)
        declared = {**ROLES, "11": {**ROLES["11"], "unrolled": "True", "trip": "2"}}
        with self.assertRaisesRegex(ValueError, "without a source pragma"):
            self.alternatives(loops, declared)
        pragma = SOURCE.replace("            for (int partition", "#pragma GCC unroll 2\n            for (int partition")
        pragma_roles = {str(int(k) + (1 if int(k) > 10 else 0)): v for k, v in declared.items()}
        moved = [loop(1, 0, (9,), (12,), 1), loop(3, 1, (14,), (16,), 2, induction=(14,))]
        fors = TABLE.parse_fors(pragma)
        self.assertEqual(fors[12]["unroll"], 2)
        found = TABLE.resolve_roles("p", moved, fors, pragma_roles, TABLE.function_extents(pragma, KERNELS), CONTEXT)
        self.assertEqual({i: r["line"] for i, r in found[0].items()}, {1: 9, 3: 14})

    def test_enumeration_must_be_confluent(self) -> None:
        original, calls = TABLE.prove_assignment, []
        def first_only(program, loops, trial, *rest):
            # Accepts only the first combination each enumeration pass tries (two combinations per pass).
            calls.append(1)
            if len(calls) % 2 == 0:
                raise ValueError("order-dependent")
        TABLE.prove_assignment = first_only
        try:
            with self.assertRaisesRegex(ValueError, "not confluent"):
                self.alternatives(nest(loop(4, 2, (13, 11), (13,), 3)))
        finally:
            TABLE.prove_assignment = original

    def test_enumeration_explosion_fails_closed(self) -> None:
        bound = TABLE.MAX_COMBINATIONS
        TABLE.MAX_COMBINATIONS = 1
        try:
            with self.assertRaisesRegex(ValueError, "refusing to truncate"):
                self.alternatives(nest(loop(4, 2, (13, 11), (13,), 3)))
        finally:
            TABLE.MAX_COMBINATIONS = bound

    def test_single_trip_source_loops_form_no_machine_loop(self) -> None:
        single = {**ROLES, "11": {**ROLES["11"], "trip": "P if G else 1"}}
        loops = [loop(1, 0, (9,), (11,), 1), loop(3, 1, (13,), (15,), 2, induction=(13,))]
        self.assertEqual(self.assign(loops, mapped=single, context={**CONTEXT, "G": 0}), {1: 9, 3: 13})
        with self.assertRaisesRegex(ValueError, "S2"):
            self.alternatives(loops, single, CONTEXT)

    def test_multi_entry_and_empty_self_loops(self) -> None:
        loops = nest()
        loops[1] = loop(2, 1, (11,), (11, 12, 17), 2, induction=(11,))
        self.assertEqual(self.assign(loops), {1: 9, 2: 11, 3: 13})
        (found,) = self.alternatives(nest(loop(4, 0, (13,), (13,), 1, empty=True)))
        self.assertEqual((found[4]["line"], found[4]["provenance"]), (None, "empty-self-loop"))

    def test_empty_self_loops_never_cover_a_source_loop(self) -> None:
        # C1: the block loop is missing; a `b .` whose .loc is the block header does not stand in for it.
        missing = [loop(1, 0, (9,), (11,), 1), loop(2, 1, (11,), (13,), 2, induction=(11,)),
                   loop(4, 2, (13,), (13,), 3, empty=True)]
        with self.assertRaisesRegex(ValueError, "S2d"):
            self.alternatives(missing)
        with self.assertRaisesRegex(ValueError, "holds a nested loop"):
            self.alternatives([loop(1, 0, (9,), (9,), 1, empty=True), loop(2, 1, (11,), (13,), 2)])

    def test_induction_counts_only_on_a_header(self) -> None:
        # C2 (i): a step whose .loc is body code (15) is no induction evidence: both readings stay.
        both = self.alternatives(nest(loop(4, 2, (13, 11), (13,), 3, induction=(15,))))
        self.assertEqual(sorted(alternative[4]["line"] for alternative in both), [11, 13])

    def test_declared_unroll_iterated_by_a_machine_loop_fails(self) -> None:
        # C5: a validated unroll declaration, but a machine loop's latch names that header directly.
        pragma = SOURCE.replace("            for (int partition", "#pragma GCC unroll 2\n            for (int partition")
        roles = {str(int(k) + (1 if int(k) > 10 else 0)): v for k, v in ROLES.items()}
        roles["12"] = {**roles["12"], "unrolled": "True", "trip": "2"}
        loops = [loop(1, 0, (9,), (12,), 1), loop(2, 1, (12,), (14,), 2, induction=(12,)),
                 loop(3, 2, (14,), (16,), 3, induction=(14,))]
        with self.assertRaisesRegex(ValueError, "declares fully unrolled"):
            TABLE.resolve_roles("p", loops, TABLE.parse_fors(pragma), roles, TABLE.function_extents(pragma, KERNELS),
                                CONTEXT)

    def test_inlined_kernel_nests_at_its_call_site(self) -> None:
        self.assertEqual(self.assign([loop(1, 0, (21,), (22,)), loop(2, 1, (26,), (27,), depth=2)]), {1: 21, 2: 26})

    def test_unresolved_conflicting_and_unmapped_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "unresolved"):
            self.alternatives([loop(1, 0, ("arm_neon.h:338",), (5,))])
        with self.assertRaisesRegex(ValueError, "S2a"):
            self.alternatives([loop(1, 0, (9,), (21,))])  # two unrelated column loops
        with self.assertRaisesRegex(ValueError, "unmapped for statement at line 13"):
            self.alternatives(nest(), {key: value for key, value in ROLES.items() if key != "13"})

    def test_kernel_and_call_site_text_must_be_unique(self) -> None:
        with self.assertRaisesRegex(ValueError, "not unique"):
            TABLE.function_extents(SOURCE, [{"signature": "for (int"}])
        with self.assertRaisesRegex(ValueError, "lies in no other kernel"):
            TABLE.function_extents(SOURCE, [{"signature": "inline void walk(int column0)", "call": "walk<S>(column);"}])


BASELINE_SPEC = importlib.util.spec_from_file_location("cpu_native_tile_stack_baseline",
                                                       TOOL.with_name("CPUNativeTileStackBaseline.py"))
BASELINE = importlib.util.module_from_spec(BASELINE_SPEC)
BASELINE_SPEC.loader.exec_module(BASELINE)


class LoopEvidenceTest(unittest.TestCase):
    """Induction evidence needs dataflow: the compared register's only definition in the loop is a step that
    dominates the latch. An empty loop is one unconditional branch whose only successor is itself."""

    def evidence(self, body: str) -> dict:
        lines = ['\t.section .text.f,"ax"'] + body.strip("\n").split("\n")
        analysis = BASELINE.ev.Analysis("f", lines, BASELINE.ev.GENERIC, set())
        analysis.run()
        self.assertEqual(analysis.errors, [])
        count = sum(1 for line in lines[1:] if line.startswith("\t"))
        where = [f"{FILE}:{100 + index}" for index in range(count)]
        return BASELINE.loop_evidence(analysis, where)

    def test_step_dominating_the_latch_is_induction(self) -> None:
        loop = "\tmov\tw15, #0\n.La:\n\tadd\tw15, w15, #1\n\tcmp\tw15, w3\n\tb.lt\t.La\n\tret"
        self.assertEqual(self.evidence(loop)[1]["induction"], [f"{FILE}:101"])

    def test_each_dataflow_condition_is_required(self) -> None:
        # (ii) the step must dominate EVERY latch: a second back edge that skips the step.
        two_latches = ("\tmov\tw15, #0\n.La:\n\tcbz\tw5, .Lb\n\tadd\tw15, w15, #1\n\tcmp\tw15, w3\n\tb.lt\t.La\n"
                       "\tret\n.Lb:\n\tcmp\tw15, w3\n\tb.lt\t.La\n\tret")
        self.assertEqual(self.evidence(two_latches)[1]["induction"], [])
        # (iii)/(iv) a second definition (between the step and the compare, or in a nested loop).
        between = "\tmov\tw15, #0\n.La:\n\tadd\tw15, w15, #1\n\tmov\tw15, w4\n\tcmp\tw15, w3\n\tb.lt\t.La\n\tret"
        self.assertEqual(self.evidence(between)[1]["induction"], [])
        nested = ("\tmov\tw15, #0\n.La:\n\tadd\tw15, w15, #1\n.Lc:\n\tsub\tw15, w15, #2\n\tsubs\tw6, w6, #1\n"
                  "\tb.ne\t.Lc\n\tcmp\tw15, w3\n\tb.lt\t.La\n\tret")
        self.assertEqual(self.evidence(nested)[1]["induction"], [])
        # (v) the compare's other operand varies in the loop.
        varying = "\tmov\tw15, #0\n.La:\n\tadd\tw15, w15, #1\n\tadd\tw3, w3, w15\n\tcmp\tw15, w3\n\tb.lt\t.La\n\tret"
        self.assertEqual(self.evidence(varying)[1]["induction"], [])
        immediate = "\tmov\tw15, #0\n.La:\n\tadd\tw15, w15, #1\n\tcmp\tw15, #9\n\tb.lt\t.La\n\tret"
        self.assertEqual(self.evidence(immediate)[1]["induction"], [f"{FILE}:101"])

    def test_a_path_through_a_nested_loop_that_bypasses_the_step_is_no_evidence(self) -> None:
        # The outer back edge is reachable from its entry through the nested loop without crossing the step.
        bypass = ("\tmov\tw15, #0\n.La:\n\tcbz\tw5, .Lin\n\tadd\tw15, w15, #1\n.Lj:\n\tcmp\tw15, w3\n\tb.lt\t.La\n"
                  "\tret\n.Lin:\n\tsubs\tw6, w6, #1\n\tb.ne\t.Lin\n\tb\t.Lj")
        found = self.evidence(bypass)
        self.assertEqual(found[1]["induction"], [])  # outer: bypassed through the nested loop
        self.assertEqual(found[2]["induction"], [f"{FILE}:106"])  # the nested loop's own `subs w6` step

    def test_a_step_in_a_nested_loop_is_no_evidence_of_the_outer_loop(self) -> None:
        # Sole definition, but it lies in the inner loop's own block: it runs many times per outer iteration.
        inner = ("\tmov\tw15, #0\n.La:\n\tmov\tw7, #0\n.Lb:\n\tadd\tw15, w15, #1\n\tsubs\tw6, w6, #1\n\tb.ne\t.Lb\n"
                 "\tcmp\tw15, w3\n\tb.lt\t.La\n\tret")
        found = self.evidence(inner)
        self.assertEqual(found[1]["induction"], [])

    def test_other_definitions_or_a_non_dominating_step_are_not(self) -> None:
        redefined = ("\tmov\tw15, #0\n.La:\n\tadd\tw15, w15, #1\n\tcsel\tw15, w15, w4, lt\n\tcmp\tw15, w3\n"
                     "\tb.lt\t.La\n\tret")
        self.assertEqual(self.evidence(redefined)[1]["induction"], [])
        bypassed = ("\tmov\tw15, #0\n.La:\n\tcbz\tw5, .Lb\n\tadd\tw15, w15, #1\n.Lb:\n\tcmp\tw15, w3\n"
                    "\tb.lt\t.La\n\tret")
        self.assertEqual(self.evidence(bypassed)[1]["induction"], [])
        computed = "\tmov\tw15, #0\n.La:\n\tmul\tw15, w15, w4\n\tcmp\tw15, w3\n\tb.lt\t.La\n\tret"
        self.assertEqual(self.evidence(computed)[1]["induction"], [])

    def test_empty_self_loop_needs_no_exit(self) -> None:
        self.assertTrue(self.evidence("\tcbz\tw0, .Lx\n\tret\n.Lx:\n\tb\t.Lx")[1]["empty"])
        self.assertFalse(self.evidence("\tmov\tw15, #0\n.La:\n\tsubs\tw15, w15, #1\n\tb.ne\t.La\n\tret")[1]["empty"])


class TableTest(unittest.TestCase):
    def record(self, loops: list[dict], counts: dict, key: str = "1:1:0:0:0:0", mode: str = "NEON") -> dict:
        full = {cls: {} for cls in TABLE.CLASSES}
        full.update(counts)
        return {"tree_sha": "abc", "checker_sha256": "c", "mode": mode, "fp_spills": {key: 0},
                "tree_inputs": {FILE: hashlib.sha256(SOURCE.encode()).hexdigest()},
                **{name: name for name in ("generator_sha256", "admission_sha256", "probe_sha256", "closure_sha256",
                                           "object_sha256", "assembly_sha256", "image", "identity",
                                           "compiler_version", "disassembler_version", "target_command_sha256")},
                "stack_traffic": {key: {"loops": loops, "counts": full}}}

    def role_map(self, **changes) -> dict:
        entry = {"file_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(), "kernel_functions": KERNELS,
                 "panel_strips": [{"when": "R == 1 and L == 1", "S": 2}, {"when": "True", "S": 1}], "loops": ROLES}
        entry.update(changes)
        return {"abc": entry}

    def test_per_role_vectors_repeated_roles_and_dynamic_counts(self) -> None:
        loops = nest(loop(4, 2, (11,), (17,), 3))
        record = self.record(loops, {"scalar": {"0": 1, "1": 2, "3": 1, "4": 5}, "invariant_load": {"3": 2}})
        program = TABLE.table([record], self.role_map(), SOURCE)["programs"]["NEON/1:1:0:0:0:0"]
        self.assertEqual([entry["role"] for entry in program["loops"]],
                         ["column#0", "partition#0", "block#0", "partition#1"])
        self.assertEqual(program["repeated_roles"], ["partition"])
        self.assertEqual(program["per_role"]["scalar"], {"entry": 1, "column#0": 2, "block#0": 1, "partition#1": 5})
        dynamic = program["dynamic"]
        # C/4 = 16 column iterations; P partitions; K blocks in total; the nested partition copy is not re-multiplied.
        self.assertEqual(dynamic["c64-k17-p2"]["scalar"], 1 + 2 * 16 + 1 * 16 * 17 + 5 * 16 * 2)
        self.assertEqual(dynamic["c64-k3-psplit"]["invariant_load"], 2 * 16 * 3)
        self.assertEqual(dynamic["c64-k1-p2"]["invariant_load"], 2 * 16 * 1)
        self.assertEqual(program["panel_strips"], 2)

    def test_role_map_is_validated_against_the_source(self) -> None:
        record = self.record(nest(), {})
        with self.assertRaisesRegex(ValueError, "not a for statement over"):
            TABLE.table([record], self.role_map(loops={**ROLES, "13": {**ROLES["13"], "variable": "column"}}), SOURCE)
        with self.assertRaisesRegex(ValueError, "not a for statement over"):
            TABLE.table([record], self.role_map(loops={**ROLES, "14": ROLES["13"]}), SOURCE)
        with self.assertRaisesRegex(ValueError, "another version of the tile source"):
            TABLE.table([record], self.role_map(file_sha256="0" * 64), SOURCE)
        with self.assertRaisesRegex(ValueError, "no entry for tree"):
            TABLE.table([record], {"other": self.role_map(file_sha256="0" * 64)["abc"]}, SOURCE)
        # A candidate (unknown tree) is found by its tile source's hash.
        self.assertEqual(TABLE.table([record], {"other": self.role_map()["abc"]}, SOURCE)["tree_sha"], "abc")
        stale = self.record(nest(), {})
        stale["tree_inputs"][FILE] = "1" * 64
        with self.assertRaisesRegex(ValueError, "compiled from another version"):
            TABLE.table([stale], self.role_map(), SOURCE)
        mixed = self.record(nest(), {}, mode="NEON_DOTPROD")
        mixed["checker_sha256"] = "d"
        with self.assertRaisesRegex(ValueError, "different trees or checkers"):
            TABLE.table([record, mixed], self.role_map(), SOURCE)

    def test_render_is_stable_json(self) -> None:
        result = TABLE.table([self.record(nest(), {"fp": {"1": 1}})], self.role_map(), SOURCE)
        text = TABLE.render(result)
        self.assertEqual(json.loads(text), json.loads(json.dumps(result)))
        self.assertEqual(text, TABLE.render(json.loads(text)))


class TLSCallClassTest(unittest.TestCase):
    """tls_call is a first-class traffic class: per role, dynamic, and compared like every other class."""

    def test_new_hot_loop_tls_call_fails_against_a_tls_free_baseline(self) -> None:
        loops = [{"id": 1, "parent": 0, "role": "column#0"}, {"id": 2, "parent": 1, "role": "block#0"}]
        def program(tls: dict) -> dict:
            full = {cls: {} for cls in GATE.CLASSES}
            full["tls_call"] = tls
            return {"loops": loops, "per_role": full, "dynamic": {"g": {cls: 0 for cls in GATE.CLASSES}},
                    "sibling_copies": []}
        self.assertIn("tls_call", TABLE.CLASSES)
        self.assertEqual(GATE.compare_program(program({}), program({}))[1], [])
        self.assertEqual(GATE.compare_program(program({"block#0": 1}), program({}))[1],
                         ["tls_call subtree block#0 1 > develop 0", "tls_call subtree column#0 1 > develop 0",
                          "tls_call subtree entry 1 > develop 0"])


HELPER_FILE = "src/v2/kernels/cpu/gemm/Helper.h"
HELPER_SOURCE = "inline void helper() noexcept\n{\n    for (int j = 0; j < 4; ++j)\n        work(j);\n}\n"
HELPER_ROLE = {"line": 3, "role": "group", "variable": "j", "trip": "4"}


class CallClassTest(unittest.TestCase):
    """Call-model classes are first-class (per role, dynamic, compared like every other class); helper loops take
    their role from the role map's helper entry and multiply under the call site's loop chain."""

    def program(self, cls: str, counts: dict) -> dict:
        loops = [{"id": 1, "parent": 0, "role": "column#0"}, {"id": 2, "parent": 1, "role": "block#0"}]
        full = {name: {} for name in GATE.CLASSES}
        full[cls] = counts
        return {"loops": loops, "per_role": full, "dynamic": {"g": {name: 0 for name in GATE.CLASSES}},
                "sibling_copies": []}

    def test_new_hot_loop_calls_fail_against_a_call_free_baseline(self) -> None:
        added = ("call_frame", "int_call", "ext_call", "ext_bytes", "ext_accesses")
        self.assertEqual(TABLE.CLASSES[-5:], added)
        self.assertEqual(GATE.CLASSES, TABLE.CLASSES)
        for cls in added:
            with self.subTest(cls=cls):
                self.assertEqual(GATE.compare_program(self.program(cls, {}), self.program(cls, {}))[1], [])
                self.assertIn(f"{cls} subtree block#0 1 > develop 0",
                              GATE.compare_program(self.program(cls, {"block#0": 1}), self.program(cls, {}))[1])

    def record(self, role: dict | None = HELPER_ROLE) -> tuple[dict, dict]:
        base = TableTest()
        helper = {"id": 4, "parent": 3, "depth": 4, "entries": [], "latches": [], "helper": "helper",
                  "helper_loop": 1, "helper_role": role}
        record = base.record(nest(helper), {"fp": {"3": 1, "4": 1}})
        record["tree_inputs"][HELPER_FILE] = hashlib.sha256(HELPER_SOURCE.encode()).hexdigest()
        helpers = {"helper": {"source": HELPER_FILE, "source_sha256": record["tree_inputs"][HELPER_FILE],
                              "loops": {"1": HELPER_ROLE}}}
        return record, base.role_map(helpers=helpers)

    def test_helper_loops_multiply_caller_by_helper_trips(self) -> None:
        record, roles = self.record()
        result = TABLE.table([record], roles, SOURCE, helper_sources={HELPER_FILE: HELPER_SOURCE})
        TABLE.check(checkable(result))  # Helper loops recompute from the table's own helper trips.
        program = result["programs"]["NEON/1:1:0:0:0:0"]
        self.assertEqual(program["loops"][-1]["role"], "group#0")
        # column 16 x partition 2 x block 17/2 per partition = 272 block iterations; the helper loop runs 4 each.
        self.assertEqual(program["dynamic"]["c64-k17-p2"]["fp"], 272 + 272 * 4)

    def test_helper_roles_are_validated(self) -> None:
        record, roles = self.record()
        with self.assertRaisesRegex(ValueError, "not supplied or not the compiled version"):
            TABLE.table([record], roles, SOURCE)
        unmatched, _ = self.record({**HELPER_ROLE, "trip": "8"})
        with self.assertRaisesRegex(ValueError, "no matching role map entry"):
            TABLE.table([unmatched], roles, SOURCE, helper_sources={HELPER_FILE: HELPER_SOURCE})
        roles["abc"]["helpers"]["helper"]["loops"]["1"] = {**HELPER_ROLE, "line": 4}
        moved, _ = self.record({**HELPER_ROLE, "line": 4})
        with self.assertRaisesRegex(ValueError, "not a for statement"):
            TABLE.table([moved], roles, SOURCE, helper_sources={HELPER_FILE: HELPER_SOURCE})


COMPACT_FILE = TABLE.COMPACT_FILE
COMPACT_SOURCE = """namespace ms {
enum class Reduction
{
    Contiguous,        ///< One accumulator.
    OrderedPartitions, ///< Ordered partials.
};
template <bool OrderedNaN, int Codebook, int Lanes, size_t Rows, Reduction Tree, typename Packed, typename Ops>
inline void rowsProgram(const Packed &packed, int columns, int partitions, int begin, int end, int blocks_per_partition)
{
    for (int column = 0; column < columns; column += Lanes)
    {
        const int partition_count = Tree == Reduction::OrderedPartitions ? partitions : 1;
        for (int partition = 0; partition < partition_count; ++partition)
        {
#pragma GCC unroll 4
            for (size_t row = 0; row < Rows; ++row)
                acc[row] = F{};
            const int first =
                Tree == Reduction::OrderedPartitions ? begin + partition * blocks_per_partition : begin;
            const int last =
                Tree == Reduction::OrderedPartitions ? std::min(first + blocks_per_partition, end) : end;
            for (int kb = first; kb < last; ++kb)
                step(kb);
        }
    }
}
template <int Codebook, int Lanes, size_t Rows, Reduction Tree, typename Packed, typename Ops>
inline void rows(const Packed &packed, int columns, int partitions)
{
    rowsProgram<false, Codebook, Lanes, Rows, Tree, Packed, Ops>(packed, columns, partitions);
}
}
"""
# Header lines of the fixture's column, partition, row and kb for statements.
COLUMN, PARTITION, ROW, KB = (next(number for number, line in enumerate(COMPACT_SOURCE.split("\n"), start=1)
                                   if f"for (" in line and needle in line)
                              for needle in ("int column", "int partition", "size_t row", "int kb"))


def compact_loop(identity: int, parent: int, line: int, depth: int, induction: bool = True) -> dict:
    spelled = f"{COMPACT_FILE}:{line}"
    return {"id": identity, "parent": parent, "depth": depth, "entry_sources": [], "latch_sources": [],
            "latch_evidence": [spelled], "entry_evidence": [], "induction_evidence": [spelled] if induction else [],
            "member_sources": [spelled], "empty": False}


COMPACT_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cpu_native_compact_sources"


def git_blob_id(data: bytes) -> str:
    """The git object ID of `data` as a blob, computed without git (installed runners carry none)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def pinned_compact_sources(directory: Path = COMPACT_FIXTURES) -> dict[str, str]:
    """{sha256: text} of the historical compact sources pinned in `directory`. Every `<sha256>.txt` must be
    named by its own SHA256 and must match a manifest pin: the compact source path and the git blob ID of the
    content at the tree that admitted it. A fixture without a pin, or a pin without a fixture, fails."""
    manifest = json.loads((directory / "manifest.json").read_text())
    files = sorted(path.name for path in directory.iterdir() if path.suffix == ".txt")
    if files != sorted(f"{digest}.txt" for digest in manifest):
        raise ValueError(f"pinned compact source files {files} differ from their manifest {sorted(manifest)}")
    sources = {}
    for digest, pin in sorted(manifest.items()):
        data = (directory / f"{digest}.txt").read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"pinned compact source {digest} is not named by its own SHA256")
        if pin.get("path") != COMPACT_FILE or git_blob_id(data) != pin.get("git_blob"):
            raise ValueError(f"pinned compact source {digest} is not the git blob its manifest pins")
        sources[digest] = data.decode()
    return sources


def verify_committed_compact_entries(roles: dict, worktree_text: str, pinned: dict[str, str]) -> dict[str, str]:
    """Re-derive every role-map tree's compact entry from its authenticated source and require it to be exactly
    the committed one (source, file hash, kernel function, every loop's role/variable/trip/header/definitions,
    Reduction values). The source named by an entry's file_sha256 is this worktree's compact header or a pinned
    historical version; any other hash is an unknown source and fails. The worktree's own source must be mapped,
    and every pinned source must be used by some entry. Returns {tree: authenticated source sha256}."""
    sources = {**pinned, TABLE.digest(worktree_text.encode()): worktree_text}
    resolved = {}
    for tree, entry in sorted(roles.items()):
        digest = entry["compact"]["file_sha256"]
        text = sources.get(digest)
        if text is None:
            raise ValueError(f"{tree}: compact source {digest} is neither this worktree's nor a pinned version")
        if TABLE.compact_entry(text) != entry["compact"]:
            raise ValueError(f"{tree}: the compact role map entry is not the mechanical entry of its source")
        resolved[tree] = digest
    named = set(resolved.values())
    if TABLE.digest(worktree_text.encode()) not in named:
        raise ValueError("this worktree's compact source has no role map entry")
    if set(pinned) - named:
        raise ValueError(f"pinned compact sources no role map entry uses: {sorted(set(pinned) - named)}")
    return resolved


def checkable(result: dict) -> dict:
    """A fixture table with well-formed hashes (fixture records carry placeholder ones)."""
    for name in ("checker_sha256", "role_map_sha256", "tile_source_sha256"):
        result[name] = "0" * 64
    for record in result["records"].values():
        record.update({name: "0" * 64 for name in TABLE.HASHES})
    return result


class CompactRoleMapTest(unittest.TestCase):
    """Compact programs take their roles from the mechanical compact role map of their tree's compact source."""

    KEY = "compact:16:4:2:1:1:0"

    def record(self, key: str = KEY, loops: list[dict] | None = None, counts: dict | None = None) -> dict:
        record = TableTest().record([], {}, key="1:1:0:0:0:0")
        record["fp_spills"] = {key: 0}
        record["tree_inputs"][COMPACT_FILE] = hashlib.sha256(COMPACT_SOURCE.encode()).hexdigest()
        full = {cls: {} for cls in TABLE.CLASSES}
        full.update(counts or {"scalar": {"1": 1, "3": 1}})
        record["stack_traffic"] = {key: {"loops": loops if loops is not None else
                                         [compact_loop(1, 0, COLUMN, 1), compact_loop(2, 1, PARTITION, 2),
                                          compact_loop(3, 2, KB, 3)], "counts": full}}
        return record

    def role_map(self) -> dict:
        return TableTest().role_map(compact=TABLE.compact_entry(COMPACT_SOURCE))

    def test_mechanical_entry(self) -> None:
        entry = TABLE.compact_entry(COMPACT_SOURCE)
        self.assertEqual(entry["kernel_functions"], [{"signature": "inline void rowsProgram("}])
        self.assertEqual(entry["reduction"], {"Contiguous": 0, "OrderedPartitions": 1})
        self.assertEqual({line: (role["role"], role["trip"], role.get("unrolled")) for line, role in entry["loops"].items()},
                         {str(COLUMN): ("column", "ceil(C / W)", None),
                          str(PARTITION): ("partition", "P if T == 1 else 1", None),
                          str(ROW): ("row", "R", "True"), str(KB): ("block", "K / P if T == 1 else K", None)})
        with self.assertRaisesRegex(ValueError, "no mechanical role"):
            TABLE.compact_entry(COMPACT_SOURCE.replace("int kb = first; kb < last; ++kb", "int j = 0; j < 4; ++j"))
        with self.assertRaisesRegex(ValueError, "wrapper"):
            TABLE.compact_entry(COMPACT_SOURCE.replace("    rowsProgram<false", "    for (int x = 0; x < 1; ++x) f();\n"
                                                                            "    rowsProgram<false"))
        with self.assertRaisesRegex(ValueError, "no compact kernel"):
            TABLE.compact_entry("void other() {}")
        develop = COMPACT_SOURCE.replace("rowsProgram", "rowsPrograX")
        self.assertEqual(TABLE.compact_entry(develop)["kernel_functions"], [{"signature": "inline void rows("}])
        # Formatting-only differences normalise away.
        spaced = COMPACT_SOURCE.replace("column < columns;", "column  <  columns ;")
        self.assertEqual(TABLE.compact_entry(spaced)["loops"], entry["loops"])

    def test_header_and_bound_drift_fails(self) -> None:
        # S2-1: a variable name alone never decides the trip: every header and bound definition must match.
        drifts = {
            "column stride": ("column += Lanes", "column += 2 * Lanes"),
            "column bound": ("column < columns;", "column < columns / 2;"),
            "column start": ("int column = 0;", "int column = 4;"),
            "partition bound": ("partition < partition_count;", "partition < partitions;"),
            "partition count": ("? partitions : 1;", "? partitions + 1 : 1;"),
            "kb range": ("int kb = first; kb < last;", "int kb = first; kb <= last;"),
            "kb step": ("kb < last; ++kb", "kb < last; kb += 2"),
            "first": ("begin + partition * blocks_per_partition", "begin + partition"),
            "last": ("std::min(first + blocks_per_partition, end)", "first + blocks_per_partition"),
            "row bound": ("row < Rows;", "row < 2 * Rows;"),
        }
        for case, (old, new) in drifts.items():
            with self.subTest(case=case):
                self.assertIn(old, COMPACT_SOURCE)
                with self.assertRaisesRegex(ValueError, "not the admitted pattern|is not exactly"):
                    TABLE.compact_entry(COMPACT_SOURCE.replace(old, new))
        rebound = COMPACT_SOURCE.replace("                step(kb);", "                step(kb), last = end;")
        with self.assertRaisesRegex(ValueError, "is not exactly"):
            TABLE.compact_entry(rebound)

    def test_reduction_values_come_from_source(self) -> None:
        # S2-2: the tree field maps through the source's enumerators, never a constant.
        swapped = COMPACT_SOURCE.replace("    Contiguous,        ///< One accumulator.\n    OrderedPartitions, ///< Ordered "
                                         "partials.", "    OrderedPartitions, ///< Ordered partials.\n    Contiguous,")
        self.assertEqual(TABLE.compact_entry(swapped)["reduction"], {"OrderedPartitions": 0, "Contiguous": 1})
        self.assertEqual(TABLE.compact_context("compact:16:4:2:1:1:0", {"OrderedPartitions": 0, "Contiguous": 1})["T"], 0)
        self.assertEqual(TABLE.compact_context("compact:16:4:2:0:1:0", {"OrderedPartitions": 0, "Contiguous": 1})["T"], 1)
        for case, text in {"explicit value": COMPACT_SOURCE.replace("Contiguous,", "Contiguous = 1,"),
                           "third enumerator": COMPACT_SOURCE.replace("OrderedPartitions, ///", "OrderedPartitions, Other, ///"),
                           "missing": COMPACT_SOURCE.replace("enum class Reduction", "enum class Reductio")}.items():
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, "Reduction"):
                    TABLE.compact_entry(text)
        with self.assertRaisesRegex(ValueError, "names no Reduction enumerator"):
            TABLE.compact_context("compact:16:4:2:2:1:0", {"Contiguous": 0, "OrderedPartitions": 1})
        # The table binds the mapping: a check() under swapped values recomputes different counts and fails.
        result = checkable(TABLE.table([self.record(counts={"scalar": {"2": 1}})], self.role_map(), SOURCE,
                                       helper_sources={COMPACT_FILE: COMPACT_SOURCE}))
        TABLE.check(result)
        result["compact_reduction"] = {"OrderedPartitions": 0, "Contiguous": 1}
        with self.assertRaisesRegex(ValueError, "recomputed"):
            TABLE.check(result)
        # A committed entry differing from its source's mechanical entry fails the table.
        tampered = self.role_map()
        tampered["abc"]["compact"]["reduction"] = {"OrderedPartitions": 0, "Contiguous": 1}
        with self.assertRaisesRegex(ValueError, "not the mechanical entry"):
            TABLE.table([self.record()], tampered, SOURCE, helper_sources={COMPACT_FILE: COMPACT_SOURCE})

    def test_compact_programs_resolve_and_count(self) -> None:
        result = TABLE.table([self.record()], self.role_map(), SOURCE, helper_sources={COMPACT_FILE: COMPACT_SOURCE})
        program = result["programs"]["NEON/" + self.KEY]
        self.assertEqual([loop["role"] for loop in program["loops"]], ["column#0", "partition#0", "block#0"])
        # 64 / 4 lanes = 16 columns; 2 ordered partitions; 17 blocks in total.
        self.assertEqual(program["dynamic"]["c64-k17-p2"]["scalar"], 16 + 17 * 16)
        TABLE.check(checkable(result))
        # Contiguous tree: the partition loop is single-trip, so no machine loop may claim it.
        contiguous = "compact:16:4:2:0:-:0"
        record = self.record(contiguous, [compact_loop(1, 0, COLUMN, 1), compact_loop(2, 1, KB, 2)], {"scalar": {"2": 1}})
        program = TABLE.table([record], self.role_map(), SOURCE,
                              helper_sources={COMPACT_FILE: COMPACT_SOURCE})["programs"]["NEON/" + contiguous]
        self.assertEqual(program["dynamic"]["c64-k17-p2"]["scalar"], 16 * 17)

    def test_compact_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "no compact entry"):
            TABLE.table([self.record()], TableTest().role_map(), SOURCE, helper_sources={COMPACT_FILE: COMPACT_SOURCE})
        with self.assertRaisesRegex(ValueError, "not supplied or not the mapped version"):
            TABLE.table([self.record()], self.role_map(), SOURCE)
        stale = self.record()
        stale["tree_inputs"][COMPACT_FILE] = "1" * 64
        with self.assertRaisesRegex(ValueError, "another version"):
            TABLE.table([stale], self.role_map(), SOURCE, helper_sources={COMPACT_FILE: COMPACT_SOURCE})
        result = TABLE.table([self.record()], self.role_map(), SOURCE, helper_sources={COMPACT_FILE: COMPACT_SOURCE})
        result = checkable(result)
        TABLE.check(result)
        result.pop("compact_trips")
        with self.assertRaisesRegex(ValueError, "compact trips"):
            TABLE.check(result)

    @staticmethod
    def committed() -> tuple[dict, str, dict[str, str]]:
        roles = json.loads(TOOL.with_name("CPUNativeTileStackRoles.json").read_text())
        return roles, (TOOL.parents[3] / COMPACT_FILE).read_bytes().decode(), pinned_compact_sources()

    def test_committed_compact_entries_are_mechanical(self) -> None:
        # Every committed tree's compact entry (whatever trees the role map admits) is exactly the mechanical entry
        # of its own authenticated compact source: this worktree's header or a pinned, blob-verified version.
        roles, text, pinned = self.committed()
        resolved = verify_committed_compact_entries(roles, text, pinned)
        self.assertEqual(set(resolved), set(roles))
        self.assertEqual({resolved[tree] for tree in roles}, set(pinned) | {TABLE.digest(text.encode())})

    def test_forged_or_unknown_compact_entries_fail(self) -> None:
        roles, text, pinned = self.committed()
        current = TABLE.digest(text.encode())
        for tree in sorted(roles):
            source = "worktree" if roles[tree]["compact"]["file_sha256"] == current else "pinned"
            # The right source hash with an altered loop role, kernel function or Reduction value.
            for field in ("role", "kernel", "reduction"):
                with self.subTest(tree=tree, source=source, forged=field):
                    forged = json.loads(json.dumps(roles))
                    compact = forged[tree]["compact"]
                    if field == "role":
                        loop = compact["loops"][sorted(compact["loops"], key=int)[0]]
                        loop["role"] = "block" if loop["role"] != "block" else "column"
                    elif field == "kernel":
                        compact["kernel_functions"][0]["signature"] += " "
                    else:
                        compact["reduction"] = {name: 1 - value for name, value in compact["reduction"].items()}
                    with self.assertRaisesRegex(ValueError, f"{tree}: .* not the mechanical entry"):
                        verify_committed_compact_entries(forged, text, pinned)
            # An entry naming a source that is neither the worktree's nor pinned.
            with self.subTest(tree=tree, source=source, forged="unknown source"):
                forged = json.loads(json.dumps(roles))
                forged[tree]["compact"]["file_sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, f"{tree}: compact source 0+ is neither this worktree's nor"):
                    verify_committed_compact_entries(forged, text, pinned)
        # A new tree: unknown source fails; a known source with a role that is not the mechanical one fails.
        reference = next(tree for tree in sorted(roles) if roles[tree]["compact"]["file_sha256"] in pinned)
        added = {**roles, "fffffffff": {**roles[reference], "compact": {**roles[reference]["compact"],
                                                                          "file_sha256": "1" * 64}}}
        with self.assertRaisesRegex(ValueError, "fffffffff: compact source .* neither"):
            verify_committed_compact_entries(added, text, pinned)
        wrong = json.loads(json.dumps(roles[reference]))
        wrong["compact"]["loops"][sorted(wrong["compact"]["loops"], key=int)[-1]]["trip"] = "1"
        with self.assertRaisesRegex(ValueError, "fffffffff: .* not the mechanical entry"):
            verify_committed_compact_entries({**roles, "fffffffff": wrong}, text, pinned)
        # The worktree's own source must be mapped, and every pinned source used.
        # (The committed worktree source pinned instead, so every entry still resolves.)
        with self.assertRaisesRegex(ValueError, "worktree's compact source has no role map entry"):
            verify_committed_compact_entries(roles, text + "\n", {**pinned, current: text})
        with self.assertRaisesRegex(ValueError, "no role map entry uses"):
            verify_committed_compact_entries(roles, text, {**pinned, "2" * 64: text})

    def test_pinned_compact_sources_are_authenticated(self) -> None:
        self.assertTrue(pinned_compact_sources())
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)

            def copy() -> None:
                for path in directory.iterdir():
                    path.unlink()
                for path in COMPACT_FIXTURES.iterdir():
                    if path.suffix in (".txt", ".json"):
                        (directory / path.name).write_bytes(path.read_bytes())

            copy()
            self.assertEqual(pinned_compact_sources(directory), pinned_compact_sources())
            fixture = sorted(directory.glob("*.txt"))[0]
            # One changed byte: the name is no longer its SHA256.
            data = bytearray(fixture.read_bytes())
            data[-2] ^= 1
            fixture.write_bytes(bytes(data))
            with self.assertRaisesRegex(ValueError, "not named by its own SHA256"):
                pinned_compact_sources(directory)
            # Renamed to the changed content's hash: the manifest has no pin for it.
            renamed = directory / f"{hashlib.sha256(bytes(data)).hexdigest()}.txt"
            fixture.rename(renamed)
            with self.assertRaisesRegex(ValueError, "differ from their manifest"):
                pinned_compact_sources(directory)
            # Re-pinned under that hash with the original git blob ID: the blob no longer matches.
            manifest = json.loads((directory / "manifest.json").read_text())
            manifest[renamed.stem] = manifest.pop(fixture.stem)
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "not the git blob its manifest pins"):
                pinned_compact_sources(directory)
            # A pin for another path fails too.
            copy()
            manifest = json.loads((directory / "manifest.json").read_text())
            manifest[fixture.stem]["path"] = TABLE.TILE_FILE
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "not the git blob its manifest pins"):
                pinned_compact_sources(directory)


class MissingClassTest(unittest.TestCase):
    """A record without a class reads it as zero only as the baseline; a candidate record must carry every class."""

    def test_only_the_baseline_defaults_a_missing_class(self) -> None:
        loops = nest()
        counts = {cls: {} for cls in TABLE.CLASSES if cls != "tls_call"}
        record = {"tree_sha": "abc", "checker_sha256": "c", "mode": "NEON", "fp_spills": {"1:1:0:0:0:0": 0},
                  "tree_inputs": {FILE: hashlib.sha256(SOURCE.encode()).hexdigest()},
                  **{name: name for name in ("generator_sha256", "admission_sha256", "probe_sha256", "closure_sha256",
                                             "object_sha256", "assembly_sha256", "image", "identity",
                                             "compiler_version", "disassembler_version", "target_command_sha256")},
                  "stack_traffic": {"1:1:0:0:0:0": {"loops": loops, "counts": counts}}}
        role_map = {"abc": {"file_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(), "kernel_functions": KERNELS,
                            "panel_strips": [{"when": "True", "S": 1}], "loops": ROLES}}
        baseline = TABLE.table([record], role_map, SOURCE, "baseline")
        self.assertEqual(baseline["programs"]["NEON/1:1:0:0:0:0"]["per_loop"]["tls_call"], {})
        with self.assertRaisesRegex(ValueError, "candidate record lacks traffic classes"):
            TABLE.table([record], role_map, SOURCE, "candidate")

    def test_every_call_class_is_strict_on_both_sides(self) -> None:
        # A baseline from a pre-call-model checker reads each call class as zero (strict: it can only fail a
        # candidate); a candidate lacking any one of them fails.
        for cls in ("call_frame", "int_call", "ext_call", "ext_bytes", "ext_accesses"):
            with self.subTest(cls=cls):
                record = TableTest().record(nest(), {})
                del record["stack_traffic"]["1:1:0:0:0:0"]["counts"][cls]
                role_map = TableTest().role_map()
                baseline = TABLE.table([record], role_map, SOURCE, "baseline")["programs"]["NEON/1:1:0:0:0:0"]
                self.assertEqual((baseline["per_loop"][cls], baseline["dynamic"]["c64-k17-p2"][cls]), ({}, 0))
                with self.assertRaisesRegex(ValueError, "candidate record lacks traffic classes"):
                    TABLE.table([record], role_map, SOURCE, "candidate")
                unknown = TableTest().record(nest(), {"frame_extra": {"1": 1}})
                with self.assertRaisesRegex(ValueError, "unknown traffic classes"):
                    TABLE.table([unknown], role_map, SOURCE, "baseline")

    def test_gate_rejects_a_table_measured_by_another_checker(self) -> None:
        committed = json.loads(TOOL.with_name("CPUNativeTileStackTable.json").read_text())
        running = hashlib.sha256(TOOL.with_name("CPUNativeTileRegisterEvidence.py").read_bytes()).hexdigest()
        GATE.require_checker(committed, running)  # The committed table is bound to the committed checker.
        with self.assertRaisesRegex(ValueError, "regenerate the table"):
            GATE.require_checker(committed, "0" * 64)
        with self.assertRaisesRegex(ValueError, "regenerate the table"):
            GATE.require_checker({**committed, "checker_sha256": "1" * 64}, running)

    def test_helper_loops_carry_no_control_evidence(self) -> None:
        loops = [{"id": 1}, {"id": 2, "helper": "h", "helper_loop": 1}]
        BASELINE.annotate_loops(loops, {1: ["f:9"]}, {1: {"latch": ["f:9"], "entry": [], "empty": False,
                                                          "induction": [], "empty_source": None}})
        self.assertEqual((loops[0]["member_sources"], loops[1]["member_sources"], loops[1]["latch_evidence"]),
                         (["f:9"], [], []))

    def test_checker_table_and_gate_classes_are_one_closure(self) -> None:
        # The checker's recorded classes, the table's and the gate's are the same ordered tuple.
        self.assertEqual(BASELINE.ev.TRAFFIC, TABLE.CLASSES)
        self.assertEqual(GATE.CLASSES, TABLE.CLASSES)


class SiblingRuleTest(unittest.TestCase):
    """Sibling copies of one source loop: the baseline takes the minimum per parent iteration, a candidate the
    maximum (strict: false failures only)."""

    ROLES = {1: {"role": "column", "line": 9, "trip": "C / 4"}, 2: {"role": "block", "line": 13, "trip": "K"},
             3: {"role": "block", "line": 13, "trip": "K"}, 4: {"role": "group", "line": 15, "trip": "8"}}
    LOOPS = [{"id": 1, "parent": 0}, {"id": 2, "parent": 1}, {"id": 3, "parent": 1}, {"id": 4, "parent": 3}]
    GEOMETRY = {"columns": 64, "k": 3, "partitions": 1}

    def count(self, counts: dict, side: str) -> float:
        return TABLE.dynamic_count(self.LOOPS, self.ROLES, counts, self.GEOMETRY, 1, 1, side)

    def test_baseline_minimum_and_candidate_maximum(self) -> None:
        # Copy 2 carries 1 per iteration; copy 3 carries 0 itself plus 1 per group iteration below it (8 per trip).
        counts = {"0": 1, "1": 2, "2": 1, "4": 1}
        column = 16
        self.assertEqual(self.count(counts, "baseline"), 1 + 2 * column + min(1 * column * 3, 1 * column * 3 * 8))
        self.assertEqual(self.count(counts, "candidate"), 1 + 2 * column + max(1 * column * 3, 1 * column * 3 * 8))
        # Without sibling copies both sides agree with the plain sum.
        single = [loop for loop in self.LOOPS if loop["id"] != 2]
        self.assertEqual(TABLE.dynamic_count(single, self.ROLES, counts, self.GEOMETRY, 1, 1, "baseline"),
                         TABLE.dynamic_count(single, self.ROLES, counts, self.GEOMETRY, 1, 1, "candidate"))

    def test_groups_name_the_copies_and_carried_classes(self) -> None:
        self.assertEqual(TABLE.sibling_groups(self.LOOPS, self.ROLES), [(1, 13, [2, 3])])
        self.assertEqual(TABLE.subtree_static(self.LOOPS, {"3": 1, "4": 2, "2": 5}, 3), 3)


class CommittedTableTest(unittest.TestCase):
    """The committed develop table is internally consistent; tampering with any derived field fails."""

    PATH = TOOL.with_name("CPUNativeTileStackTable.json")

    def setUp(self) -> None:
        self.table = json.loads(self.PATH.read_text())

    def test_committed_table_is_consistent_and_strict(self) -> None:
        TABLE.check(self.table)
        self.assertEqual(self.table["sibling_rule"], "baseline")
        self.assertEqual(self.PATH.read_text(), TABLE.render(self.table))
        self.assertEqual(set(self.table["records"]), {"NEON", "NEON_DOTPROD", "NEON_I8MM"})

    def test_tampering_fails(self) -> None:
        key = sorted(self.table["programs"])[0]
        cases = {
            "dynamic": lambda t: t["programs"][key]["dynamic"]["c64-k17-p2"].update(scalar=-1.0),
            "per_role": lambda t: t["programs"][key]["per_role"]["scalar"].update(entry=99),
            "role": lambda t: t["programs"][key]["loops"][0].update(role="block#0"),
            "trip": lambda t: next(iter(t["trips"].values())).update(trip="1"),
            "hash": lambda t: t.update(checker_sha256=""),
            "rule": lambda t: t.update(sibling_rule="lenient"),
            "key": lambda t: t["programs"].update({"NEON/9:0:0:0:0:0": t["programs"][key]}),
        }
        for case, edit in cases.items():
            with self.subTest(case=case):
                tampered = json.loads(json.dumps(self.table))
                edit(tampered)
                with self.assertRaises((ValueError, KeyError)):
                    TABLE.check(tampered)


LAMBDA_SOURCE = """struct Tile {
    template <typename Visit> void walk(int first, int last, Visit &&visit) const noexcept
    {
        if constexpr (A)
        {
            for (int block = first; block < last; )
            {
                const int end = next(block);
                for (; block < end; ++block)
                    visit(block, 1);
            }
        }
        else
        {
            for (int block = first; block < last; ++block)
                visit(block, 2);
        }
    }
    void run() const noexcept
    {
        for (int column = 0; column < columns_; column += 4)
            for (int partition = 0; partition < partitions_; ++partition)
                walk(first, last, [&](int block, int tag) {
                    for (int group = 0; group < 8; ++group)
                        consume(block, group);
                });
    }
    void other() const noexcept
    {
        for (int partition = 0; partition < partitions_; ++partition)
            walk(first, last, [&](int block, int kind) {
                consume(block, kind);
            });
    }
};
"""
LAMBDA_KERNELS = [{"signature": "void run() const noexcept"}, {"signature": "void other() const noexcept"},
                  {"signature": "void walk(int first", "calls": ["walk(first, last, [&](int block, int tag)",
                                                                 "walk(first, last, [&](int block, int kind)"]},
                  {"signature": "[&](int block, int tag)", "calls": ["visit(block, 1);", "visit(block, 2);"],
                   "via": "walk(first, last, [&](int block, int tag)"},
                  {"signature": "[&](int block, int kind)", "calls": ["visit(block, 1);", "visit(block, 2);"],
                   "via": "walk(first, last, [&](int block, int kind)"}]
LAMBDA_ROLES = {"6": {"role": "superblock", "variable": "block", "trip": "ceil((K / P - 1) / 8) + 1"},
                "9": {"role": "block", "variable": "block", "trip": "(K / P) / (ceil((K / P - 1) / 8) + 1)"},
                "15": {"role": "block", "variable": "block", "trip": "K / P"},
                "21": {"role": "column", "variable": "column", "trip": "C / 4"},
                "22": {"role": "partition", "variable": "partition", "trip": "P if G else 1"},
                "24": {"role": "group", "variable": "group", "trip": "8"}}


class InlinedVisitorTest(unittest.TestCase):
    """An inlined K walk called from two kernels and visitor lambdas nested at its invocation sites."""

    def setUp(self) -> None:
        self.fors = TABLE.parse_fors(LAMBDA_SOURCE)
        conditioned = [dict(kernel) for kernel in LAMBDA_KERNELS]
        for kernel in conditioned[3:]:
            kernel["calls"] = [{"call": "visit(block, 1);", "when": "E == 1"}, {"call": "visit(block, 2);", "when": "E != 1"}]
        self.kernels = TABLE.function_extents(LAMBDA_SOURCE, conditioned)
        self.unconditioned = TABLE.function_extents(LAMBDA_SOURCE, LAMBDA_KERNELS)

    def assign(self, *loops: dict, encoding: int = 2, kernels: list | None = None) -> dict:
        (roles,) = TABLE.resolve_roles("p", list(loops), self.fors, LAMBDA_ROLES, kernels or self.kernels,
                                       {**CONTEXT, "E": encoding})
        return {identity: role["line"] for identity, role in roles.items()}

    def test_untyped_inner_for_takes_its_tested_variable(self) -> None:
        self.assertEqual((self.fors[6]["variable"], self.fors[9]["variable"]), ("block", "block"))

    def test_visitor_lines_nest_inside_the_walk_through_their_own_call_only(self) -> None:
        run = [loop(1, 0, (21,), (22,), 1), loop(2, 1, (22,), (22,), 2)]
        self.assertEqual(self.assign(*run, loop(3, 2, (15,), (16,), 3), loop(4, 3, (24,), (25,), 4)),
                         {1: 21, 2: 22, 3: 15, 4: 24})
        self.assertEqual(self.assign(*run, loop(3, 2, (6,), (8,), 3), loop(4, 3, (9,), (10,), 4),
                                     loop(5, 4, (24,), (25,), 5), encoding=1), {1: 21, 2: 22, 3: 6, 4: 9, 5: 24})

    def test_call_site_conditions_select_the_instantiated_branch(self) -> None:
        run = [loop(1, 0, (21,), (22,), 1), loop(2, 1, (22,), (22,), 2)]
        # A visitor-body control location: the innermost mapped loop on the instantiated branch only.
        self.assertEqual(self.assign(*run, loop(3, 2, (26,), (26,), 3), encoding=2), {1: 21, 2: 22, 3: 15})
        with self.assertRaises(ValueError):  # E == 1: 26 names the walk's block loop 9, but 6 has no machine loop
            self.assign(*run, loop(3, 2, (26,), (26,), 3), encoding=1)
        with self.assertRaises(ValueError):  # unconditioned: both branches feasible, conflicting / missing loops
            self.assign(*run, loop(3, 2, (26,), (26,), 3), kernels=self.unconditioned)

    def test_infeasible_context_is_not_taken(self) -> None:
        # run's visitor is never passed from other(): other's partition and run's visitor are unrelated.
        with self.assertRaises(ValueError):
            self.assign(loop(1, 0, (30,), (24,)))

    def test_policy_and_superblock_trips(self) -> None:
        geometry = {"columns": 64, "k": 17, "partitions": 2}
        self.assertEqual(TABLE.trip(LAMBDA_ROLES["22"], geometry, 1, 1, 0), 1.0)
        self.assertEqual(TABLE.trip(LAMBDA_ROLES["22"], geometry, 1, 1, 1), 2.0)
        superblocks = TABLE.trip(LAMBDA_ROLES["6"], geometry, 1, 1)
        self.assertEqual(superblocks, 2.0)  # 8.5 (at most 9) blocks per partition meet at most 2 superblocks of 8.
        self.assertAlmostEqual(superblocks * TABLE.trip(LAMBDA_ROLES["9"], geometry, 1, 1), 8.5)


GATE_PATH = TOOL.with_name("CPUNativeTileStackGate.py")
GATE_SPEC = importlib.util.spec_from_file_location("cpu_native_tile_stack_gate", GATE_PATH)
GATE = importlib.util.module_from_spec(GATE_SPEC)
GATE_SPEC.loader.exec_module(GATE)


LEAF_OPS = "namespace ms {\n    enum class NEONByteDot { ASIMD, DotProd, I8MM };\n}\n"
LEAF_TILE = ("struct Tile {\n    using Dot = multi_scale::NEONByteDot;\n"
             "    template <Dot Instruction> static constexpr Dot leafInstruction() noexcept\n"
             "    { return Instruction == Dot::I8MM && Rows == 1 ? Dot::DotProd : Instruction; }\n};\n")
LEAF_SHA = hashlib.sha256(LEAF_TILE.encode()).hexdigest()


def keymap(develop_formats: dict, candidate_formats: dict) -> dict:
    """Fixture key map: format -> (codebook, superblock, encoding, asymmetric) on each tree, toy dispatch rules."""
    rules = [{"when": "E == 1 and A", "tail": "1:0:1"}, {"when": "E == 1", "tail": "1:0:0"},
             {"when": "E == 7", "tail": "7:0:0"}, {"when": "E == 3", "tail": "compact"},
             {"when": "E == 0", "tail": "0:0:0"}]
    def tree(formats: dict) -> dict:
        return {"headers": {path: "0" * 64 for path in GATE.HEADERS}, "dispatch_sha256": "1" * 64, "dispatch": rules,
                "formats": {name: dict(zip(("codebook", "superblock", "encoding", "asymmetric"), values))
                            for name, values in formats.items()}}
    result = {"baseline": "dev", "trees": {"develop": tree(develop_formats), "candidate": tree(candidate_formats)}}
    result["trees"]["develop"]["leaf"] = GATE.derive_leaf(LEAF_TILE, LEAF_OPS)
    result["formats"], result["keys"] = GATE.derive_keys(result)
    result["compact"] = GATE.derive_compact(result)
    return result


KEYMAP = keymap({"Q8_0": (19, 0, 1, 0), "Q5_0": (6, 0, 1, 0), "Q5_1": (7, 0, 1, 1), "Q4_0": (0, 0, 0, 0),
                 "IQ2": (15, 1, 3, 0)},
                {"Q8_0": (19, 0, 1, 0), "Q5_0": (6, 0, 7, 0), "Q5_1": (7, 0, 7, 1), "Q4_0": (0, 0, 0, 0),
                 "IQ2": (15, 1, 7, 0)})


class KeyMapTest(unittest.TestCase):
    """Per-source-format keys are derived mechanically from each tree's encodings and dispatch rules."""

    def test_keys_list_formats_and_every_develop_path(self) -> None:
        self.assertEqual(KEYMAP["formats"]["Q5_1"], {"develop": "1:0:1", "candidate": "7:0:0"})
        # 7:0:0 serves Q5_0 (develop 1:0:0), Q5_1 (develop 1:0:1) and IQ2 (no develop tile path).
        self.assertEqual(KEYMAP["keys"]["7:0:0"], {"formats": ["IQ2", "Q5_0", "Q5_1"],
                                                   "develop": ["1:0:0", "1:0:1", "compact"]})
        self.assertEqual(KEYMAP["keys"]["1:0:0"], {"formats": ["Q8_0"], "develop": ["1:0:0"]})

    def test_check_rejects_hand_edits_and_foreign_inputs(self) -> None:
        GATE.check_keymap(KEYMAP)
        edited = json.loads(json.dumps(KEYMAP))
        edited["keys"]["7:0:0"]["develop"] = ["1:0:0"]
        with self.assertRaisesRegex(ValueError, "not the ones derived"):
            GATE.check_keymap(edited)
        with self.assertRaisesRegex(ValueError, "another version"):
            GATE.check_keymap(KEYMAP, {path: "2" * 64 for path in GATE.HEADERS})
        changed = json.loads(json.dumps(KEYMAP))
        changed["trees"]["candidate"]["formats"]["Q8_0"]["codebook"] = 20
        with self.assertRaisesRegex(ValueError, "changed identity"):
            GATE.derive_keys(changed)

    def test_compact_program_keys_are_derived_per_source_format(self) -> None:
        # IQ2 runs the compact program of its codebook on both trees; IQ1 moved from a develop tile path to it.
        compact = keymap({"IQ2": (15, 1, 3, 0), "IQ1": (16, 1, 1, 1), "Q8_0": (19, 0, 1, 0)},
                         {"IQ2": (15, 1, 3, 0), "IQ1": (16, 1, 3, 1), "Q8_0": (19, 0, 1, 0)})
        self.assertEqual(compact["compact"], {"compact:15": {"formats": ["IQ2"], "develop": ["compact:15"]},
                                              "compact:16": {"formats": ["IQ1"], "develop": ["1:0:1"]}})
        GATE.check_keymap(compact)
        for edit in (lambda m: m["compact"]["compact:16"]["develop"].clear(), lambda m: m.pop("compact"),
                     lambda m: m["compact"].update({"compact:15": {"formats": ["IQ2"], "develop": ["compact:16"]}})):
            edited = json.loads(json.dumps(compact))
            edit(edited)
            with self.assertRaisesRegex(ValueError, "not the ones derived"):
                GATE.check_keymap(edited)
        committed = json.loads(TOOL.with_name("CPUNativeTileStackKeyMap.json").read_text())
        self.assertEqual(sorted(committed["compact"]), [f"compact:{c}" for c in (10, 12, 13, 14, 15, 16, 17, 9)])
        self.assertEqual(committed["compact"]["compact:12"], {"formats": ["IQ3_XXS"], "develop": ["1:0:0"]})

    def test_dispatch_text_is_the_functions_body(self) -> None:
        source = "x\n template <int Strips = 0> void dispatch() const noexcept\n { switch (e) { case 1: return; } }\n y"
        self.assertTrue(GATE.dispatch_text(source).endswith("} }"))
        with self.assertRaisesRegex(ValueError, "no unique dispatch"):
            GATE.dispatch_text(source + source)

    def test_committed_key_map_is_consistent(self) -> None:
        committed = json.loads(TOOL.with_name("CPUNativeTileStackKeyMap.json").read_text())
        GATE.check_keymap(committed)
        self.assertEqual(committed["keys"]["7:1:1"], {"formats": ["Q5_1"], "develop": ["1:0:1"]})
        self.assertEqual(committed["formats"]["Q5_K"], {"develop": "1:0:1", "candidate": "3:7:1"})
        self.assertEqual(committed["keys"]["3:11:0"], {"formats": ["IQ3_S"], "develop": ["5:0:0"]})


class GateTest(unittest.TestCase):
    """Per class: subtree dominance for matching structures (hoisting allowed, moving inward not), else dynamic;
    every develop path of a key's formats must be met (strict minimum); zero K only without a develop path."""

    def program(self, per_role: dict, loops: list[tuple[int, int, str]], dynamic: dict | None = None,
                siblings: bool = False) -> dict:
        full = {cls: {} for cls in GATE.CLASSES}
        full.update(per_role)
        return {"loops": [{"id": i, "parent": p, "role": r} for i, p, r in loops], "per_role": full,
                "dynamic": {"g": {cls: (dynamic or {}).get(cls, 0) for cls in GATE.CLASSES}},
                "sibling_copies": [{}] if siblings else []}

    LOOPS = [(1, 0, "column#0"), (2, 1, "partition#0"), (3, 2, "block#0")]

    def results(self, candidate: dict, develop: dict, key: str = "1:1:0:1:0:0", extra: dict | None = None) -> dict:
        programs = {f"NEON/1:1:0:{tail}": develop for tail in ("1:0:0", "1:0:1", "0:0:0")}
        programs.update(extra or {})
        return next(iter(GATE.compare({"sibling_rule": "candidate", "geometries": [], "programs": {f"NEON/{key}": candidate}},
                                      {"sibling_rule": "baseline", "geometries": [], "tree_sha": "dev",
                                       "programs": programs}, KEYMAP).values()))

    def failures(self, *args, **kwargs) -> list[str]:
        return self.results(*args, **kwargs)["failures"]

    def test_hoisting_passes_moving_inward_fails(self) -> None:
        develop = self.program({"scalar": {"column#0": 2, "block#0": 1}}, self.LOOPS)
        hoisted = self.program({"scalar": {"entry": 1, "column#0": 2}}, self.LOOPS)
        self.assertEqual(self.failures(hoisted, develop), [])
        inward = self.program({"scalar": {"column#0": 0, "block#0": 2}}, self.LOOPS)
        self.assertEqual(self.failures(inward, develop), ["[1:1:0:1:0:0] scalar subtree block#0 2 > develop 1",
                                                          "[1:1:0:1:0:0] scalar subtree partition#0 2 > develop 1"])
        # An outer decrease never pays for an inner increase.
        paid = self.program({"scalar": {"column#0": 0, "partition#0": 1, "block#0": 1}}, self.LOOPS)
        self.assertEqual(self.failures(paid, develop), ["[1:1:0:1:0:0] scalar subtree partition#0 2 > develop 1"])

    def test_different_structure_or_siblings_compare_dynamically(self) -> None:
        develop = self.program({"fp": {"block#0": 1}}, self.LOOPS, {"fp": 100})
        moved = self.program({"fp": {"partition#1": 5}}, self.LOOPS + [(4, 2, "partition#1")], {"fp": 90})
        self.assertEqual(self.results(moved, develop)["modes"], ["dynamic"])
        self.assertEqual(self.failures(moved, develop), [])
        hotter = self.program({"fp": {"partition#1": 5}}, self.LOOPS + [(4, 2, "partition#1")], {"fp": 101})
        self.assertEqual(self.failures(hotter, develop), ["[1:1:0:1:0:0] fp g 101 > develop 100"])
        self.assertEqual(self.failures(self.program({}, self.LOOPS, {"fp": 101}, siblings=True), develop),
                         ["[1:1:0:1:0:0] fp g 101 > develop 100"])

    def test_every_develop_path_is_met_and_zero_k_only_without_one(self) -> None:
        develop = self.program({"alias_load": {"block#0": 3}}, self.LOOPS)
        tighter = self.program({"alias_load": {"block#0": 1}}, self.LOOPS)
        candidate = self.program({"alias_load": {"block#0": 2}}, self.LOOPS)
        result = self.results(candidate, develop, "1:1:0:7:0:0", {"NEON/1:1:0:1:0:1": tighter})
        self.assertEqual(result["develop"], ["1:1:0:1:0:0", "1:1:0:1:0:1"])
        self.assertEqual(result["formats"], ["IQ2", "Q5_0", "Q5_1"])
        # Within develop 1:0:0 (Q5_0) but not within 1:0:1 (Q5_1), and IQ2 had no tile path: K walk must be zero.
        self.assertEqual(result["failures"], [
            "[1:1:0:1:0:1] alias_load subtree block#0 2 > develop 1",
            "[1:1:0:1:0:1] alias_load subtree column#0 2 > develop 1",
            "[1:1:0:1:0:1] alias_load subtree entry 2 > develop 1",
            "[1:1:0:1:0:1] alias_load subtree partition#0 2 > develop 1",
            "alias_load block#0 2 (no develop path: K walk must be zero)"])

    def test_unmapped_keys_missing_develop_and_rules_fail(self) -> None:
        self.assertEqual(self.failures(self.program({}, self.LOOPS), self.program({}, self.LOOPS), "1:1:0:9:0:0"),
                         ["unmapped key tail 9:0:0"])
        with self.assertRaisesRegex(ValueError, "sibling rule"):
            GATE.compare({"sibling_rule": "baseline", "geometries": [], "programs": {}},
                         {"sibling_rule": "baseline", "geometries": [], "tree_sha": "dev", "programs": {}}, KEYMAP)



class CompactComparisonTest(unittest.TestCase):
    """P4: every candidate compact copy against every develop path of its formats in the SAME copy (TU or
    link-selected), strict minimum, never another copy or a maximum; moved formats against their develop tile."""

    COMPACT = keymap({"IQ2": (15, 1, 3, 0), "IQ1": (16, 1, 1, 1), "Q8_0": (19, 0, 1, 0)},
                     {"IQ2": (15, 1, 3, 0), "IQ1": (16, 1, 3, 1), "Q8_0": (19, 0, 1, 0)})
    LOOPS = [(1, 0, "column#0"), (2, 1, "partition#0"), (3, 2, "block#0")]
    REDUCTION = {"Contiguous": 0, "OrderedPartitions": 1}

    def program(self, block: int, dynamic: int, loops: list | None = None) -> dict:
        return GateTest().program({"scalar": {"block#0": block}}, loops or self.LOOPS, {"scalar": dynamic})

    def compare(self, candidate: dict, develop: dict, unreachable: dict | None = None,
                develop_reduction: dict | None = None, keymap_: dict | None = None, copy_map: dict | None = None,
                tu_classes: dict | None = None, tile_sha: str = LEAF_SHA) -> dict:
        inputs = {"unreachable": unreachable, "copy_map": copy_map, "tu_classes": tu_classes}
        return GATE.compare({"sibling_rule": "candidate", "geometries": [], "programs": candidate,
                             "compact_reduction": self.REDUCTION},
                            {"sibling_rule": "baseline", "geometries": [], "tree_sha": "dev", "programs": develop,
                             "compact_reduction": develop_reduction or self.REDUCTION, "tile_source_sha256": tile_sha},
                            keymap_ or self.COMPACT, inputs)

    def test_same_copy_strict_and_both_policies(self) -> None:
        develop = {"NEON/compact:15:4:2:1:-:0@KernelFactory": self.program(2, 20)}
        for ordered in (0, 1):
            with self.subTest(ordered=ordered):
                name = f"NEON/compact:15:4:2:1:{ordered}:0@KernelFactory"
                within = self.compare({name: self.program(2, 20)}, develop)[name]
                self.assertEqual((within["failures"], within["develop"]),
                                 ([], ["compact:15:4:2:1:-:0@KernelFactory"]))
                hotter = self.compare({name: self.program(3, 30)}, develop)[name]["failures"]
                self.assertIn("[compact:15:4:2:1:-:0@KernelFactory] scalar subtree block#0 3 > develop 2", hotter)
                self.assertTrue(hotter and all(failure.startswith("[compact:15:4:2:1:-:0@KernelFactory] scalar")
                                               for failure in hotter))

    def test_no_other_copy_or_maximum_substitutes(self) -> None:
        # The develop copy of another TU (even a hotter one) never stands in for the candidate's own TU.
        develop = {"NEON/compact:15:4:2:1:-:0@WeightManager": self.program(9, 90)}
        name = "NEON/compact:15:4:2:1:1:0@KernelFactory"
        failures = self.compare({name: self.program(1, 10)}, develop)[name]["failures"]
        self.assertEqual(failures, ["develop program NEON/compact:15:4:2:1:-:0@KernelFactory is missing "
                                    "(no other copy substitutes)"])
        # Each TU is compared with its own develop copy: the colder develop TU decides for that TU.
        develop = {"NEON/compact:15:4:2:1:-:0@WeightManager": self.program(9, 90),
                   "NEON/compact:15:4:2:1:-:0@KernelFactory": self.program(1, 10)}
        results = self.compare({name: self.program(2, 20),
                                "NEON/compact:15:4:2:1:1:0@WeightManager": self.program(2, 20)}, develop)
        self.assertTrue(results[name]["failures"])
        self.assertEqual(results["NEON/compact:15:4:2:1:1:0@WeightManager"]["failures"], [])
        unnamed = "NEON/compact:15:4:2:1:1:0"
        self.assertEqual(self.compare({unnamed: self.program(0, 0)}, develop)[unnamed]["failures"],
                         ["compact program has no copy identity (TU or link-selected copy)"])

    def test_moved_format_compares_with_its_develop_tile_path(self) -> None:
        # IQ1 ran develop tile 1:0:1; the candidate compact:16 copy must stay within it, dynamically.
        name = "NEON_I8MM/compact:16:4:1:1:1:2@KernelFactory"
        tile = "NEON_I8MM/1:1:1:1:0:1"  # Single-row I8MM: the DotProd leaf; policy = the candidate's.
        result = self.compare({name: self.program(1, 50)}, {tile: self.program(0, 60)})[name]
        self.assertEqual((result["develop"], result["modes"], result["failures"]), (["1:1:1:1:0:1"], ["dynamic"], []))
        over = self.compare({name: self.program(0, 61)}, {tile: self.program(9, 60)})[name]["failures"]
        self.assertEqual(over, ["[1:1:1:1:0:1] scalar g 61 > develop 60"])
        # With per-copy develop tile programs, only the same copy's tile program counts.
        copies = {tile + "@WeightManager": self.program(0, 99), tile + "@KernelFactory": self.program(0, 40)}
        self.assertEqual(self.compare({name: self.program(1, 50)}, copies)[name]["failures"],
                         ["[1:1:1:1:0:1@KernelFactory] scalar g 50 > develop 40"])

    def test_tree_values_map_through_each_tables_source(self) -> None:
        # The develop source numbers OrderedPartitions 0: the candidate's tree 1 maps to develop tree 0.
        name = "NEON/compact:15:4:2:1:1:0@KernelFactory"
        develop = {"NEON/compact:15:4:2:0:-:0@KernelFactory": self.program(2, 20)}
        swapped = {"OrderedPartitions": 0, "Contiguous": 1}
        self.assertEqual(self.compare({name: self.program(2, 20)}, develop, develop_reduction=swapped)[name]["failures"],
                         [])
        self.assertTrue(self.compare({name: self.program(2, 20)}, develop)[name]["failures"])

    def test_undispatched_codebooks_need_p2_reachability(self) -> None:
        name = "NEON/compact:7:4:2:1:1:0@KernelFactory"
        failures = self.compare({name: self.program(0, 0)}, {})[name]["failures"]
        self.assertTrue(failures and "P2" in failures[0])
        # P4-c: the record must carry the P2 evidence digests and the enumerated callers.
        evidence = {"evidence_sha256": "a" * 64, "dso_sha256": "b" * 64, "process_resolution_sha256": "c" * 64,
                    "callers": [{"caller": "dispatch<CompactMultiScale>@KernelFactory", "reaches": False}]}
        result = self.compare({name: self.program(0, 0)}, {}, unreachable={name: evidence})[name]
        self.assertEqual((result["failures"], result["modes"]), ([], ["unreachable (P2)"]))
        for case, record in {"bare listing": True, "no DSO digest": {**evidence, "dso_sha256": "x"},
                             "no process resolution": {k: v for k, v in evidence.items()
                                                       if k != "process_resolution_sha256"},
                             "callers not enumerated": {k: v for k, v in evidence.items() if k != "callers"},
                             "a caller reaches it": {**evidence, "callers": [{"caller": "c", "reaches": True}]}}.items():
            with self.subTest(case=case):
                failures = self.compare({name: self.program(0, 0)}, {}, unreachable={name: record})[name]["failures"]
                self.assertTrue(failures and "unreachability record" in failures[0], failures)
        policy_free = "NEON/compact:15:4:2:1:-:0@KernelFactory"
        self.assertTrue(self.compare({policy_free: self.program(0, 0)}, {})[policy_free]["failures"])

    def test_baseline_rule_is_never_a_maximum(self) -> None:
        with self.assertRaisesRegex(ValueError, "develop-min"):
            GATE.compare({"sibling_rule": "candidate", "geometries": [], "programs": {}},
                         {"sibling_rule": "candidate", "geometries": [], "tree_sha": "dev", "programs": {}},
                         self.COMPACT)

    def test_copy_keys_resolve_in_the_table(self) -> None:
        key = "compact:16:4:2:1:1:0@KernelFactory"
        record = CompactRoleMapTest().record(key)
        result = TABLE.table([record], CompactRoleMapTest().role_map(), SOURCE, helper_sources={COMPACT_FILE: COMPACT_SOURCE})
        TABLE.check(checkable(result))
        self.assertIn("NEON/" + key, result["programs"])


class CompactConsolidationTest(unittest.TestCase):
    """P4-a (leaf rule from develop source), P4-b (P2 copy map under consolidation) and P4-d (TU classes)."""

    def setUp(self) -> None:
        self.case = CompactComparisonTest()
        self.program = self.case.program

    def test_leaf_rule_is_derived_from_develop_source(self) -> None:
        self.assertEqual({k: v for k, v in GATE.derive_leaf(LEAF_TILE, LEAF_OPS).items() if k not in
                          ("tile_sha256", "ops_sha256")},
                         {"families": {"ASIMD": 0, "DotProd": 1, "I8MM": 2}, "from": 2, "rows": 1, "to": 1})
        # A develop source mapping single-row I8MM to ASIMD instead: the moved format compares with 1:0:..., and a
        # develop table holding only the DotProd program fails as missing.
        elsewhere = LEAF_TILE.replace("? Dot::DotProd", "? Dot::ASIMD")
        mapped = json.loads(json.dumps(CompactComparisonTest.COMPACT))
        mapped["trees"]["develop"]["leaf"] = GATE.derive_leaf(elsewhere, LEAF_OPS)
        name = "NEON_I8MM/compact:16:4:1:1:1:2@KernelFactory"
        sha = hashlib.sha256(elsewhere.encode()).hexdigest()
        result = self.case.compare({name: self.program(0, 10)}, {"NEON_I8MM/1:1:1:1:0:1": self.program(0, 60)},
                                   keymap_=mapped, tile_sha=sha)[name]
        self.assertEqual((result["develop"], result["failures"]),
                         (["1:0:1:1:0:1"], ["develop program NEON_I8MM/1:0:1:1:0:1 is missing (no other copy "
                                            "substitutes)"]))
        # A rule derived from another tile source than the develop table's fails; malformed sources fail.
        with self.assertRaisesRegex(ValueError, "another tile source"):
            self.case.compare({name: self.program(0, 10)}, {}, tile_sha="0" * 64)
        for case, (tile, ops) in {"two rules": (LEAF_TILE.replace(": Instruction;", ": Rows == 2 ? Dot::ASIMD : "
                                                                  "Instruction;"), LEAF_OPS),
                                  "no leafInstruction": ("struct Tile {};", LEAF_OPS),
                                  "enum order value": (LEAF_TILE, LEAF_OPS.replace("ASIMD,", "ASIMD = 1,")),
                                  "unknown family": (LEAF_TILE.replace("Dot::DotProd", "Dot::SVE"), LEAF_OPS)}.items():
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    GATE.derive_leaf(tile, ops)
        stripped = json.loads(json.dumps(CompactComparisonTest.COMPACT))
        del stripped["trees"]["develop"]["leaf"]
        with self.assertRaisesRegex(ValueError, "leaf rule"):
            GATE.check_keymap(stripped)
        committed = json.loads(TOOL.with_name("CPUNativeTileStackKeyMap.json").read_text())
        table = json.loads(TOOL.with_name("CPUNativeTileStackTable.json").read_text())
        self.assertEqual(committed["trees"]["develop"]["leaf"]["tile_sha256"], table["tile_source_sha256"])

    def copy_map(self, sites: dict) -> dict:
        return {"evidence_sha256": "e" * 64, "copies": sites}

    def test_a_merged_copy_must_pass_every_replaced_copy(self) -> None:
        # C0': one consolidated copy now serves the call sites two develop TU copies served.
        develop = {"NEON/compact:15:4:2:1:-:0@KernelFactory": self.program(3, 30),
                   "NEON/compact:15:4:2:1:-:0@WeightManager": self.program(1, 10)}
        merged = self.copy_map({"Consolidated": [
            {"call_site": "KernelFactory::project", "develop_copy": "KernelFactory"},
            {"call_site": "WeightManager::prepare", "develop_copy": "WeightManager"}]})
        name = "NEON/compact:15:4:2:1:1:0@Consolidated"
        within = self.case.compare({name: self.program(1, 10)}, develop, copy_map=merged)[name]
        self.assertEqual((within["failures"], within["develop"]),
                         ([], ["compact:15:4:2:1:-:0@KernelFactory", "compact:15:4:2:1:-:0@WeightManager"]))
        # Within the hotter copy but above the colder one: the strict minimum fails it (never the maximum).
        between = self.case.compare({name: self.program(2, 20)}, develop, copy_map=merged)[name]["failures"]
        self.assertTrue(between and all("@WeightManager" in failure for failure in between), between)

    def test_copy_map_gaps_fail(self) -> None:
        develop = {"NEON/compact:15:4:2:1:-:0@KernelFactory": self.program(3, 30)}
        name = "NEON/compact:15:4:2:1:1:0@Consolidated"
        missing = self.copy_map({"Consolidated": [{"call_site": "KernelFactory::project",
                                                   "develop_copy": "KernelFactory"},
                                                  {"call_site": "Loader::warm", "develop_copy": "Loader"}]})
        self.assertEqual(self.case.compare({name: self.program(1, 10)}, develop, copy_map=missing)[name]["failures"],
                         ["develop program NEON/compact:15:4:2:1:-:0@Loader is missing (no other copy substitutes)"])
        new_site = self.copy_map({"Consolidated": [{"call_site": "KernelFactory::project",
                                                    "develop_copy": "KernelFactory"},
                                                   {"call_site": "Planner::probe", "develop_copy": None}]})
        failures = self.case.compare({name: self.program(1, 10)}, develop, copy_map=new_site)[name]["failures"]
        self.assertEqual(failures, ["call site Planner::probe has no develop counterpart and no owner classification "
                                    "record"])
        classified = json.loads(json.dumps(new_site))
        classified["copies"]["Consolidated"][1]["owner_classification"] = {"owner": "owner", "class": "planning-only",
                                                                           "record_sha256": "d" * 64}
        self.assertEqual(self.case.compare({name: self.program(1, 10)}, develop, copy_map=classified)[name]["failures"],
                         [])
        other = self.copy_map({"Elsewhere": [{"call_site": "x", "develop_copy": "KernelFactory"}]})
        self.assertEqual(self.case.compare({name: self.program(1, 10)}, develop, copy_map=other)[name]["failures"],
                         ["the copy map names no call sites served by Consolidated"])
        for bad in ({"copies": {}}, {"evidence_sha256": "e" * 64, "copies": {"Consolidated": []}},
                    self.copy_map({"Consolidated": [{"call_site": "x"}]})):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.case.compare({name: self.program(1, 10)}, develop, copy_map=bad)

    def test_results_are_grouped_by_tu_class_and_stay_strict(self) -> None:
        develop = {f"NEON/compact:15:4:2:1:-:0@{tu}": self.program(1, 10)
                   for tu in ("KernelFactory", "PlanningProjectionMeasurement", "WeightManager")}
        candidate = {f"NEON/compact:15:4:2:1:1:0@{tu}": self.program(2 if tu != "WeightManager" else 1, 20)
                     for tu in ("KernelFactory", "PlanningProjectionMeasurement", "WeightManager")}
        classes = {"KernelFactory": "inference-hot", "PlanningProjectionMeasurement": "startup-planning",
                   "WeightManager": "load"}
        results = self.case.compare(candidate, develop, tu_classes=classes)
        self.assertEqual(GATE.grouped(results), {
            "inference-hot": {"passing": [], "failing": ["NEON/compact:15:4:2:1:1:0@KernelFactory"]},
            "startup-planning": {"passing": [], "failing": ["NEON/compact:15:4:2:1:1:0@PlanningProjectionMeasurement"]},
            "load": {"passing": ["NEON/compact:15:4:2:1:1:0@WeightManager"], "failing": []}})
        # No class waives anything: the planning copy fails exactly as the hot one does.
        self.assertTrue(results["NEON/compact:15:4:2:1:1:0@PlanningProjectionMeasurement"]["failures"])
        self.assertEqual(GATE.grouped(self.case.compare(candidate, develop)).keys(), {"unclassified"})
        with self.assertRaisesRegex(ValueError, "not one of"):
            self.case.compare(candidate, develop, tu_classes={"KernelFactory": "cold"})


class UnqualifiedGateReceiptTest(unittest.TestCase):
    """UA-3/UA-4: unqualified input is always a failed gate result, its comparison reported separately."""

    ADMISSION = {"status": "failed", "failures": [{"reason": "spill", "programs": ["1:1:1:3:7:1"], "detail": "x"}]}

    def test_clean_comparison_on_unqualified_input_is_still_red(self) -> None:
        clean = GATE.finish_receipt({"failing": [], "results": {}}, "unqualified-analysis", self.ADMISSION)
        self.assertEqual((clean["input"], clean["comparison"], clean["status"]),
                         ("unqualified-analysis", "clean", "failed"))
        self.assertEqual(clean["admission"], self.ADMISSION)
        self.assertIn("not Gate7 evidence", clean["diagnostic"])
        regressed = GATE.finish_receipt({"failing": ["NEON/x"], "results": {}}, "unqualified-analysis", self.ADMISSION)
        self.assertEqual((regressed["comparison"], regressed["status"]), ("regressions", "failed"))

    def test_sealed_input_passes_only_when_clean(self) -> None:
        self.assertEqual(GATE.finish_receipt({"failing": [], "results": {}}, "sealed-proof", None)["status"], "passed")
        self.assertEqual(GATE.finish_receipt({"failing": ["a"], "results": {}}, "sealed-proof", None)["status"],
                         "failed")
        self.assertNotIn("admission", GATE.finish_receipt({"failing": [], "results": {}}, "sealed-proof", None))

    def test_gate_without_a_document_fails_and_receipts_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "stack-gate.json"
            receipt.write_text("stale")
            command = [sys.executable, "-B", str(TOOL.with_name("CPUNativeTileStackGate.py")), "proof",
                       f"--proof={root / 'proof.json'}", "--mode=NEON", "--baseline=b", "--roles=r", "--keymap=k",
                       f"--receipt={receipt}", "--"]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            written = json.loads(receipt.read_text())
            self.assertEqual(written["status"], "failed")
            self.assertIn("stage 1 failed structurally", written["error"])
            (root / "proof.analysis.json").write_text("{}")
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertIn("passes --analysis", json.loads(receipt.read_text())["error"])
            # The seal is never consumed through --analysis: the checker refuses (here: no bound record fields).
            completed = subprocess.run(command[:-1] + ["--analysis", "--"], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(json.loads(receipt.read_text())["input"], "unqualified-analysis")


class VerifiedAnalysisGateTest(unittest.TestCase):
    """V1: the gate takes one attempt's frozen analysis only from the current checker and only when it equals its
    document; it never re-verifies or re-analyses."""

    def verified(self, **changes):
        checker = TOOL.with_name("CPUNativeTileRegisterEvidence.py")
        document = {"schema": 12, "kind": "sealed-proof", "mode": "NEON",
                    "checker_sha256": hashlib.sha256(checker.read_bytes()).hexdigest(),
                    "program_violations": {}, "ordered_adds": {}, "stack_traffic": {}, "chain_witness": {}}
        fields = {name: document[name] for name in ("program_violations", "ordered_adds", "stack_traffic",
                                                    "chain_witness")}
        values = {"kind": "sealed-proof", "document_text": json.dumps(document), "data": b"", "disassembly": "",
                  "assembly": "", "fields_text": json.dumps(fields), "checker_sha256": document["checker_sha256"]}
        values.update(changes)
        return BASELINE.ev.VerifiedAnalysis(**values)

    def test_foreign_or_divergent_analyses_are_rejected(self) -> None:
        args = argparse.Namespace(mode="NEON")
        with self.assertRaisesRegex(ValueError, "made by another checker"):
            GATE.gate_from_analysis(args, self.verified(checker_sha256="0" * 64), "{}")
        with self.assertRaisesRegex(ValueError, "differs from its document"):
            GATE.gate_from_analysis(args, self.verified(fields_text=json.dumps(
                {"program_violations": {"x": 1}, "ordered_adds": {}, "stack_traffic": {}, "chain_witness": {}})), "{}")
        with self.assertRaisesRegex(ValueError, "not this mode's document"):
            GATE.gate_from_analysis(argparse.Namespace(mode="NEON_I8MM"), self.verified(), "{}")
        with self.assertRaisesRegex(ValueError, "not this mode's document"):
            GATE.gate_from_analysis(args, self.verified(kind="unqualified-analysis"), "{}")

    def test_the_verified_analysis_is_immutable(self) -> None:
        verified = self.verified()
        with self.assertRaises(AttributeError):
            verified.fields_text = "{}"
        self.assertTrue(all(isinstance(value, (str, bytes)) for value in verified))

class BoundedMemcpyPlacementTest(unittest.TestCase):
    """M-3/M-4/M-6 (reviewer 91): placement of admitted variable-size memcpy sites and the strict baseline rule."""

    SITE = {"line": "src/v2/k.h:1110", "text": "std::memcpy(&acc[row],output[row]+column,(columns-column)*sizeof(float));",
            "lower_bytes": 4, "upper_bytes": 16, "placement": "column-tail", "sides": 1, "loop": 2}
    LOOPS = [{"id": 1, "parent": 0}, {"id": 2, "parent": 1}]
    COUNTS = {"ext_call": {"2": 3}, "ext_bytes": {"2": 48}, "ext_accesses": {"2": 6}, "fp": {}}

    def test_column_tail_is_charged_once_per_column_loop_entry(self) -> None:
        counts, [placed] = TABLE.place_bounded_sites(self.LOOPS, self.COUNTS, [self.SITE],
                                                     [{1: {"role": "row"}, 2: {"role": "column"}}])
        self.assertEqual((counts["ext_call"], counts["ext_bytes"], counts["ext_accesses"]),
                         ({"2": 2, "1": 1}, {"2": 32, "1": 16}, {"2": 4, "1": 2}))
        self.assertEqual((placed["loop"], placed["held_by"], placed["delta"]),
                         (1, 2, {"ext_bytes": 12, "ext_accesses": 1}))

    def test_a_tail_inside_a_kb_loop_stays_charged_at_its_loop(self) -> None:
        counts, [placed] = TABLE.place_bounded_sites(self.LOOPS, self.COUNTS, [self.SITE],
                                                     [{1: {"role": "column"}, 2: {"role": "kb"}}])
        self.assertEqual(counts["ext_bytes"], {"2": 48})
        self.assertEqual(placed["loop"], 2)

    def test_disagreeing_role_assignments_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "role assignments disagree"):
            TABLE.place_bounded_sites(self.LOOPS, self.COUNTS, [self.SITE],
                                      [{1: {"role": "row"}, 2: {"role": "column"}},
                                       {1: {"role": "row"}, 2: {"role": "kb"}}])

    def program(self, text: str) -> dict:
        site = {"text": text, "loop": 1, "role": "row#0", "delta": {"ext_bytes": 12, "ext_accesses": 1},
                "dynamic": {"c64-k1-p1": {"ext_bytes": 12.0, "ext_accesses": 1.0}}}
        return {"per_loop": {"ext_bytes": {"1": 16}, "ext_accesses": {"1": 2}},
                "per_role": {"ext_bytes": {"row#0": 16}, "ext_accesses": {"row#0": 2}},
                "dynamic": {"c64-k1-p1": {"ext_bytes": 16.0, "ext_accesses": 2.0}}, "bounded_sites": [site]}

    def test_same_text_is_symmetric_and_changed_text_is_strict(self) -> None:
        baseline = self.program(self.SITE["text"])
        self.assertIs(GATE.strict_baseline(self.program(self.SITE["text"]), baseline), baseline)
        strict = GATE.strict_baseline(self.program("std::memcpy(other);"), baseline)
        self.assertEqual((strict["per_loop"]["ext_bytes"], strict["per_role"]["ext_accesses"],
                          strict["dynamic"]["c64-k1-p1"]), ({"1": 4}, {"row#0": 1},
                                                           {"ext_bytes": 4.0, "ext_accesses": 1.0}))
        self.assertEqual(baseline["per_loop"]["ext_bytes"], {"1": 16})  # The committed baseline is untouched.


if __name__ == "__main__":
    unittest.main()
