#!/usr/bin/env python3
"""Compiler-free adversarial tests of the native NEON tile register-allocation witness.

Every analysis test renders its functions as disassembler output (llvm-objdump
style, ELF or Mach-O; plus GNU objdump samples) and runs the production front
end over it: the control-flow/provenance analysis (stack bases and widths,
aliases, joins, merge definitions, pointer slots and their partial reuse), the
ABI save proof (opcode, lane and slot semantics), the admitted instruction
forms, call-boundary rejection, the never-returning diagnostic proof, data in
code, the disassembly/relocation contract, the object format and embedded
inventory, the symbol table, the envelope-derived program inventory, the
single-row/ratchet policy, compiler identity, and proof binding through a
scripted stand-in compiler and disassembler (dependency closure, symlinks,
inputs changing during the build, executed command, object, disassembler,
budget). Expected program keys are written out here, never derived from
checker tables.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import textwrap
import unittest

CHECKER_PATH = Path(__file__).resolve().parents[2] / "cmake" / "CPUNativeTileRegisterEvidence.py"
SPEC = importlib.util.spec_from_file_location("cpu_native_tile_register_evidence", CHECKER_PATH)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)

# Tile::dispatch() programs per policy: (encoding, nibble map, with-minimum), written out independently.
DISPATCHED = [(1, 0, 1), (1, 0, 0), (5, 0, 0), (2, 0, 0), (4, 2, 0), (0, 0, 0), (0, 2, 0), (0, 1, 1)]
NEON_KEYS = sorted((rows, 0, policy, encoding, nibble_map, minimum)
                   for rows in (1, 2, 3, 4) for policy in (0, 1) for encoding, nibble_map, minimum in DISPATCHED)
NEON_INVENTORY = "1:ASIMD,2:ASIMD,3:ASIMD,4:ASIMD"
FATAL = "_ZN9llaminar23cpu11native_quantised23CPUNativeQuantisedWeightView32rejectOutsideAdmittedPublicationEv"
CLEAN = "\tstp\tx29, x30, [sp, #-16]!\n\tmov\tx29, sp\n\tsdot\tv0.4s, v1.16b, v2.16b\n\tldp\tx29, x30, [sp], #16\n\tret"
FATAL_BODY = "\tstp\tx29, x30, [sp, #-16]!\n\tmov\tx29, sp\n\tbl\tfwrite\n\tbl\tabort\n"
BRANCH = re.compile(r"^(?:b|bl|cbz|cbnz|tbz|tbnz|b\.\w+)$")


def name(rows: int, dot: int, policy: int, encoding: int, nibble_map: int, minimum: int) -> str:
    """Mangled tile run<> symbol for one program key."""
    return (f"_ZNK9llaminar23cpu11native_quantised27CPUNativeNEONProjectionTileILm{rows}EE3runINS1_11multi_scale17"
            f"NEONMultiScaleOpsILNS5_11NEONByteDotE{dot}EEELNS_28CPUProjectionNumericalPolicyE{policy}ELNS1_21"
            f"CPUNativeQuantisedEncodingE{encoding}ELNS3_9NibbleMapE{nibble_map}ELb{minimum}EEEvv")


def disassembly(functions: list[tuple[str, str]], macho: bool = False, path: str = "probe.o") -> str:
    """Render functions (one instruction per line, .Lname: labels) as llvm-objdump -d -r output.

    Branches to local labels print their address; branches to other symbols carry a branch relocation; lines
    starting with '.' or '<' render as the disassembler prints data words.
    """
    prefix = "_" if macho else ""
    out = [f"{path}:\tfile format {'mach-o arm64' if macho else 'elf64-littleaarch64'}", ""]
    if macho:
        out.append("Disassembly of section __TEXT,__text:")
    address = 0
    for symbol, body in functions:
        if not macho:
            address = 0
            out += ["", f"Disassembly of section .text.{symbol}:"]
        start, labels, code = address, {}, []
        for line in textwrap.dedent(body).splitlines():
            text = line.strip()
            if not text:
                continue
            if re.match(r"^\.L\w+:$", text):
                labels[text[:-1]] = address
                continue
            code.append((address, text))
            address += 4
        out += ["", f"{start:016x} <{prefix}{symbol}>:"]
        for at, text in code:
            parts = text.split(None, 1)
            mnemonic, operands = parts[0], (CHECKER.split_operands(parts[1]) if len(parts) > 1 else [])
            relocation = None
            if BRANCH.match(mnemonic) and operands:
                target = operands[-1]
                if target in labels:
                    operands[-1] = f"0x{labels[target]:x} <{prefix}{symbol}+0x{labels[target] - start:x}>"
                elif target.startswith(".L"):
                    operands[-1] = f"0xfffff0 <{prefix}{symbol}+0x{0xfffff0 - start:x}>"
                else:
                    operands[-1] = f"0x{at:x} <{prefix}{symbol}+0x{at - start:x}>"
                    kind = "ARM64_RELOC_BRANCH26" if macho else {"bl": "R_AARCH64_CALL26",
                                                                 "b": "R_AARCH64_JUMP26"}.get(mnemonic,
                                                                                              "R_AARCH64_CONDBR19")
                    relocation = (kind, prefix + target)
            out.append(f"{at:8x}: \t{mnemonic}\t{', '.join(operands)}" if operands else f"{at:8x}: \t{mnemonic}")
            if relocation:
                out.append(f"\t\t{at:016x}:  {relocation[0]}\t{relocation[1]}")
    return "\n".join(out) + "\n"


def symbol_table(functions: list[tuple[str, str]], macho: bool = False) -> str:
    """objdump -t listing every function symbol."""
    prefix = "_" if macho else ""
    lines = ["", "probe.o:\tfile format x", "", "SYMBOL TABLE:"]
    for symbol, _ in functions:
        section = "__TEXT,__text" if macho else f".text.{symbol}\t0000000000000004"
        lines.append(f"0000000000000000 g     F {section} {prefix}{symbol}")
    return "\n".join(lines) + "\n"


def witness_object(inventory: str | None = NEON_INVENTORY, macho: bool = False, extra: bytes = b"") -> bytes:
    """Minimal object bytes: format header plus the embedded instantiation inventory string."""
    if macho:
        header = bytes.fromhex("cffaedfe") + (0x0100000C).to_bytes(4, "little") + bytes(24)
    else:
        header = bytearray(64)
        header[0:6] = b"\x7fELF\x02\x01"
        header[18:20] = (183).to_bytes(2, "little")
        header = bytes(header)
    marker = b"" if inventory is None else b"llaminar-native-neon-tile-probe:" + inventory.encode() + b"\0"
    return header + marker + extra


def programs(bodies: dict[tuple, str] | None = None, keys: list[tuple] | None = None) -> list[tuple[str, str]]:
    return [(name(*key), (bodies or {}).get(key, CLEAN)) for key in (keys if keys is not None else NEON_KEYS)]


def audit_witness(bodies: dict[tuple, str] | None = None, extra: list[tuple[str, str]] | None = None,
                  keys: list[tuple] | None = None, inventory: str | None = NEON_INVENTORY, mode: str = "NEON",
                  macho: bool = False, table: list[tuple[str, str]] | None = None) -> dict[str, int]:
    functions = programs(bodies, keys) + (extra or [])
    return CHECKER.audit(witness_object(inventory, macho), disassembly(functions, macho),
                         symbol_table(functions if table is None else table, macho), mode)


def budget(limit: int = 4) -> dict:
    """Budget naming every grouped NEON program."""
    limits = {":".join(map(str, key)): limit for key in NEON_KEYS if key[0] != 1}
    return {"GNU-14": {"NEON": limits}}


class Result:
    """Spills and structural errors of one function."""

    def __init__(self, spills: int, errors: list[str], noreturn: set[str]) -> None:
        self.spills, self.errors, self.noreturn = spills, errors, noreturn


def analyse(body: str, extra: dict[str, str] | None = None, macho: bool = False) -> Result:
    """Run the production front end and analysis over one function (plus optional other functions)."""
    functions = [("f", body)] + list((extra or {}).items())
    bodies = CHECKER.normalize_disassembly(disassembly(functions, macho), macho)
    noreturn = CHECKER.noreturn_symbols(bodies, CHECKER.GENERIC)
    spills, errors = CHECKER.analyse_function("f", bodies, CHECKER.GENERIC, noreturn)
    return Result(spills, errors, noreturn)


class ClassificationTest(unittest.TestCase):
    """FP/SIMD traffic through any stack provenance is a spill; analysis is control-flow aware."""

    def spills(self, body: str, **options) -> int:
        analysis = analyse(body, **options)
        self.assertEqual(analysis.errors, [], body)
        return analysis.spills

    def fatal(self, body: str, fragment: str, **options) -> None:
        analysis = analyse(body, **options)
        self.assertTrue(any(fragment in error for error in analysis.errors), (body, analysis.errors))

    def test_every_base_and_width_is_classified(self) -> None:
        for line in ("str q31, [sp, 160]", "ldr q25, [sp, #0xa0]", "stp q0, q1, [sp, 32]", "stur q0, [x29, -16]",
                     "str q0, [x29, #16]", "str s3, [sp, 12]", "ldr d17, [x29, 40]", "str h2, [sp, 4]",
                     "st1 {v0.16b}, [sp]", "ldr b1, [sp, 1]", "st1 {v0.16b, v1.16b}, [sp]", "ld1 {v2.s}[1], [sp]",
                     "str q0, [sp, x9]", "ldr q0, [sp, w9, sxtw 4]", "str q0, [sp, -32]!", "ldr q0, [sp], 16"):
            with self.subTest(line=line):
                self.assertEqual(self.spills(f"\t{line}\n\tret\n"), 1)

    def test_stack_aliases_including_shifted_and_indexed_forms(self) -> None:
        self.assertEqual(self.spills("\tadd x9, sp, #64\n\tstr q0, [x9]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tmov x10, sp\n\tldr q4, [x10, 32]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tsub x11, x29, #32\n\tstr d18, [x11]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tadd x9, sp, #1, lsl #12\n\tstr q0, [x9]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tadd x9, sp, x3\n\tstr q0, [x9]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tmov x9, sp\n\tstr q0, [x9, x4, lsl 4]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tadd x9, sp, 64\n\teor x9, x9, x2\n\tstr q0, [x9]\n\tret\n"), 1)
        self.fatal("\tadd x9, sp, 64\n\tstr q0, [x1, x9]\n\tret\n", "index register")

    def test_any_other_definition_retires_an_alias(self) -> None:
        self.assertEqual(self.spills("\tadd x9, sp, #64\n\tadd x9, x1, #8\n\tstr q0, [x9]\n\tret\n"), 0)
        self.assertEqual(self.spills("\tadd x9, sp, #64\n\tldr x9, [x1]\n\tstr q0, [x9]\n\tret\n"), 0)
        self.assertEqual(self.spills("\tadd x9, sp, #64\n\tmov w9, 3\n\tstr q0, [x9]\n\tret\n"), 0)
        self.assertEqual(self.spills("\tadd x0, sp, #64\n\tldp x0, x1, [x2]\n\tstr q0, [x0]\n\tret\n"), 0)
        self.assertEqual(self.spills("\tldr q0, [x1, 16]\n\tstr q0, [x2]\n\tret\n"), 0)

    def test_branch_join_of_stack_and_heap_base_is_fatal(self) -> None:
        self.fatal("\tmov x9, sp\n\tcbz w0, .Luse\n\tmov x9, x1\n.Luse:\n\tldr q0, [x9]\n\tret\n", "only some paths")
        self.fatal("\tadd x9, sp, 64\n\tcsel x9, x9, x1, eq\n\tstr q0, [x9]\n\tret\n", "only some paths")
        self.assertEqual(self.spills("\tmov x9, sp\n\tcbz w0, .Luse\n\tadd x9, sp, 32\n.Luse:\n\tldr q0, [x9]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tmov x9, x1\n\tcbz w0, .Luse\n\tmov x9, x2\n.Luse:\n\tldr q0, [x9]\n\tret\n"), 0)

    def test_loops_reach_a_fixed_point(self) -> None:
        body = "\tmov x9, sp\n.Lloop:\n\tstr q0, [x9], 16\n\tsub x2, x2, 1\n\tcbnz x2, .Lloop\n\tret\n"
        self.assertEqual(self.spills(body), 1)

    def test_stack_addresses_stored_in_slots_and_escapes(self) -> None:
        self.assertEqual(self.spills("\tadd x9, sp, 32\n\tstr x9, [sp, 8]\n\tldr x10, [sp, 8]\n\tstr q0, [x10]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tadd x9, sp, 32\n\tstr x9, [sp, 8]\n\tstr x1, [sp, 8]\n\tldr x10, [sp, 8]\n"
                                     "\tstr q0, [x10]\n\tret\n"), 0)
        self.fatal("\tadd x9, sp, 32\n\tstr x9, [x1]\n\tret\n", "cannot be tracked")
        self.fatal("\tadd x9, sp, 32\n\tstr x9, [sp, x2]\n\tret\n", "cannot be tracked")
        self.fatal("\tmov x9, sp\n\tfmov d0, x9\n\tret\n", "FP/SIMD register")

    def test_unknown_offset_stack_access_with_live_pointer_slots_fails(self) -> None:
        frame = "\tsub sp, sp, #32\n\tadd x9, sp, #16\n\tstr x9, [sp]\n"
        for access in ("ldr w11, [sp, x10]", "ldp w10, w11, [sp, x10]", "ldr q1, [sp, x3]", "ldr x11, [sp, x3, lsl 3]"):
            with self.subTest(access=access):
                self.fatal(frame + f"\t{access}\n\tadd sp, sp, #32\n\tret\n", "unknown offset")
        self.assertEqual(self.spills("\tsub sp, sp, #32\n\tldr q1, [sp, x3]\n\tadd sp, sp, #32\n\tret\n"), 1)

    def test_pointer_slot_joins_keep_the_non_stack_alternative(self) -> None:
        joined = ("\tcbz w0, .La\n\tadd x9, sp, #32\n\tstr x9, [sp, 8]\n\tb .Lj\n.La:\n\tstr x1, [sp, 8]\n.Lj:\n"
                  "\tldr x10, [sp, 8]\n\tstr q0, [x10]\n\tret\n")
        self.fatal(joined, "only some paths")
        self.fatal(joined.replace("\tstr x1, [sp, 8]\n", ""), "only some paths")
        self.fatal("\tldr x10, [sp]\n\tadd x9, sp, #32\n\tstr x9, [x10]\n\tldr x11, [x1]\n\tstr q0, [x11]\n\tret\n",
                   "cannot be tracked")
        composed = ("\tcbz w0, .La\n\tadd x9, sp, #32\n\tstr x9, [sp, 8]\n\tb .Lj\n.La:\n\tstr x1, [sp, 8]\n.Lj:\n"
                    "\tldr x10, [sp, 8]\n\tadd x9, sp, #32\n\tstr x9, [x10]\n\tldr x11, [x1]\n\tstr q0, [x11]\n\tret\n")
        self.fatal(composed, "only some paths")  # The composed escape fails at its ambiguous base.

    def test_conditional_transforms_of_stack_addresses_are_not_admitted(self) -> None:
        for transform in ("csinc x9, x9, x9, eq", "csinv x9, x9, x1, ne", "csneg x9, x9, x9, lt", "cinc x9, x9, eq"):
            with self.subTest(transform=transform):
                self.fatal(f"\tmov x9, sp\n\t{transform}\n\tstr q0, [x9]\n\tret\n", "admitted grammar")
        self.fatal("\tmov x9, sp\n\tcsel x9, x9, x1, eq\n\tstr q0, [x9]\n\tret\n", "only some paths")

    def test_structure_footprints_use_the_exact_lane_and_arrangement(self) -> None:
        tail = "\tldr x10, [sp, #8]\n\tstr q0, [x10]\n\tadd sp, sp, #32\n\tret\n"
        frame = "\tsub sp, sp, #32\n\tadd x9, sp, #16\n\tstr x9, [sp, #8]\n"
        for macho in (False, True):
            for lane in ("\tst1 {v0.s}[0], [sp]\n", "\tst1 {v0.8b}, [sp]\n", "\tst1 {v0.4h}, [sp]\n"):
                with self.subTest(lane=lane, macho=macho):
                    analysis = analyse(frame + lane + tail, macho=macho)
                    self.assertEqual((analysis.spills, analysis.errors), (2, []))
            for overlap in ("\tadd x11, sp, #12\n\tst1 {v0.s}[0], [x11]\n", "\tadd x11, sp, #4\n\tst1 {v0.2s}, [x11]\n"):
                with self.subTest(overlap=overlap, macho=macho):
                    self.assertTrue(any("partial overwrite" in error
                                        for error in analyse(frame + overlap + tail, macho=macho).errors), overlap)
            with self.subTest(cover=macho):  # A whole 16-byte store retires the slot: x10 is no stack address.
                analysis = analyse(frame + "\tst1 {v0.16b}, [sp]\n" + tail, macho=macho)
                self.assertEqual((analysis.spills, analysis.errors), (1, []))
        for unknown in ("\tst1 {v0.16b-v1.16b}, [sp]\n", "\tst1 {v0}, [sp]\n", "\tld2 {v0.4s}, [sp]\n"):
            with self.subTest(unknown=unknown):
                self.fatal("\tsub sp, sp, #32\n" + unknown + "\tadd sp, sp, #32\n\tret\n", "footprint")

    def test_merge_definitions_keep_their_destination(self) -> None:
        self.assertEqual(self.spills("\tmov x9, sp\n\tbfi x9, xzr, #0, #1\n\tstr q0, [x9]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tmov x9, sp\n\tmovk x9, #0, lsl #48\n\tstr q0, [x9]\n\tret\n"), 1)
        self.assertEqual(self.spills("\tmov x9, x1\n\tbfi x9, xzr, #0, #1\n\tstr q0, [x9]\n\tret\n"), 0)
        self.fatal("\tmov x9, sp\n\tbfi w9, wzr, #0, #1\n\tret\n", "32-bit view")

    def test_32_bit_views_and_partial_slot_reuse_fail(self) -> None:
        frame = "\tsub sp, sp, #32\n\tadd x9, sp, #16\n\tstr x9, [sp]\n"
        self.fatal(frame + "\tldp w10, w11, [sp]\n\torr x9, x10, x11, lsl #32\n\tstr q0, [x9]\n\tadd sp, sp, #32\n\tret\n",
                   "reload of a stored stack address")
        self.fatal(frame + "\tldrsw x10, [sp]\n\tadd sp, sp, #32\n\tret\n", "reload of a stored stack address")
        self.fatal(frame + "\tldr d1, [sp]\n\tadd sp, sp, #32\n\tret\n", "reload of a stored stack address")
        self.fatal(frame + "\tstr w1, [sp, 4]\n\tadd sp, sp, #32\n\tret\n", "partial overwrite")
        self.fatal("\tmov x9, sp\n\tmov w10, w9\n\tret\n", "32-bit view")
        self.fatal("\tmov x9, sp\n\tstr w9, [x1]\n\tret\n", "32-bit view")
        self.fatal("\tmov x9, sp\n\tldr q0, [x1, w9, sxtw]\n\tret\n", "index register")
        self.assertEqual(self.spills(frame + "\tldr x10, [sp]\n\tstr q0, [x10]\n\tadd sp, sp, #32\n\tret\n"), 1)
        self.assertEqual(self.spills(frame + "\tstr q1, [sp]\n\tldr x10, [sp]\n\tstr q0, [x10]\n"
                                     "\tadd sp, sp, #32\n\tret\n"), 1)  # The q store spills; x10 is no longer stack.

    def test_every_call_boundary_is_structural(self) -> None:
        for body, extra in (("\tmov x0, sp\n\tbl identity\n\tstr q0, [x0]\n\tret\n", {"identity": "\tret\n"}),
                            ("\tmov x8, sp\n\tbl helper\n\tstr q0, [x8]\n\tret\n", {"helper": "\tret\n"}),
                            ("\tstp d8, d9, [sp, -32]!\n\tfmov d8, d0\n\tmov x0, sp\n\tbl helper\n"
                             "\tldp d8, d9, [sp], 32\n\tret\n", {"helper": "\tstr xzr, [x0]\n\tret\n"}),
                            ("\tbl helper\n\tret\n", {"helper": "\tret\n"}),
                            ("\tb helper\n", {"helper": "\tret\n"}),
                            ("\tbl external\n\tret\n", None)):
            with self.subTest(body=body):
                self.fatal(body, "unsupported call boundary", extra=extra)
        with self.assertRaisesRegex(ValueError, "unsupported branch relocation"):
            analyse("\tcbz x0, helper\n\tret\n", {"helper": "\tret\n"})

    def test_relocated_data_addresses_are_never_stack(self) -> None:
        text = disassembly([("f", "\tadrp x3, 0x0 <f>\n\tldr x3, [x3]\n\tstr q0, [sp, #16]\n\tret\n")])
        text = text.replace("adrp\tx3, 0x0 <f>\n", "adrp\tx3, 0x0 <f>\n\t\t0000000000000000:  R_AARCH64_ADR_GOT_PAGE\tstderr\n")
        text = text.replace("ldr\tx3, [x3]\n", "ldr\tx3, [x3]\n\t\t0000000000000004:  R_AARCH64_LD64_GOT_LO12_NC\tstderr\n")
        bodies = CHECKER.normalize_disassembly(text, False)
        self.assertEqual([line.strip() for line in bodies["f"][1:3]], ["adrp\tx3, stderr", "ldr\tx3, [x3, :lo12:stderr]"])
        self.assertEqual(CHECKER.analyse_function("f", bodies, CHECKER.GENERIC, set()), (1, []))
        relocated_stack = disassembly([("f", "\tldr q0, [sp]\n\tret\n")]).replace(
            "ldr\tq0, [sp]\n", "ldr\tq0, [sp]\n\t\t0000000000000000:  R_AARCH64_LDST128_ABS_LO12_NC\tx\n")
        bodies = CHECKER.normalize_disassembly(relocated_stack, False)
        self.assertTrue(CHECKER.analyse_function("f", bodies, CHECKER.GENERIC, set())[1])


class ABITest(unittest.TestCase):
    """ABI saves of d8-d15 are exempt only when proven; count ceilings are not proof."""

    PROLOGUE = "\tstp d8, d9, [sp, -32]!\n"
    EPILOGUE = "\tldp d8, d9, [sp], 32\n\tret\n"

    def check(self, body: str) -> Result:
        return analyse(body)

    def test_proven_save_and_restore_are_not_spills(self) -> None:
        analysis = self.check(self.PROLOGUE + "\tfmov d8, d0\n\tfmul v9.4s, v1.4s, v2.4s\n" + self.EPILOGUE)
        self.assertEqual((analysis.spills, analysis.errors), (0, []))
        shrink = ("\tcbz w0, .Lout\n" + self.PROLOGUE + "\tfmov d8, d0\n\tldp d8, d9, [sp], 32\n.Lout:\n\tret\n")
        self.assertEqual(self.check(shrink).errors, [])

    def assertFatal(self, body: str, fragment: str) -> None:
        errors = self.check(body).errors
        self.assertTrue(any(fragment in error for error in errors), (body, errors))

    def test_unproven_saves_are_fatal(self) -> None:
        self.assertFatal("\tfmov d8, d0\n" + self.PROLOGUE + self.EPILOGUE, "before saving")
        self.assertFatal("\tfmov x9, d8\n\tfmov d8, d0\n\tfmov d8, x9\n\tret\n", "before saving")
        self.assertFatal(self.PROLOGUE + "\tfmov d8, d0\n\tret\n", "without restoring")
        self.assertFatal(self.PROLOGUE + "\tfmov d8, d0\n\tldr d8, [sp, 8]\n\tldr d9, [sp, 8]\n\tadd sp, sp, 32\n\tret\n",
                         "without restoring")
        self.assertFatal(self.PROLOGUE + "\tfmov d8, d0\n\tldp d8, d9, [sp], 32\n\tfmov d8, d1\n\tret\n",
                         "after its epilogue restore")
        self.assertFatal(self.PROLOGUE + "\tstr q0, [sp, 8]\n" + self.EPILOGUE, "live save slot")
        self.assertFatal(self.PROLOGUE + ".Lloop:\n\tstr d8, [sp]\n\tfmov d8, d0\n\tcbnz x0, .Lloop\n" + self.EPILOGUE,
                         "live save slot")
        self.assertFatal(self.PROLOGUE + "\tfmov d8, d0\n\tldp d8, d9, [sp]\n\tcbnz x0, .Lx\n.Lx:\n\tldr d8, [sp]\n"
                         "\tadd sp, sp, 32\n\tret\n", "after its epilogue restore")
        self.assertFatal(self.PROLOGUE + "\tstr q0, [sp, x3]\n" + self.EPILOGUE, "unknown offset")

    def test_lane_structure_and_replicating_forms_are_not_restores(self) -> None:
        for reload in ("ld1 {v8.d}[1], [sp]", "ld1r {v8.2d}, [sp]", "ld2 {v8.d, v9.d}[0], [sp]", "ld1 {v8.1d}, [sp]"):
            with self.subTest(reload=reload):
                analysis = self.check(f"\tsub sp, sp, #16\n\tstr d8, [sp]\n\tfmov d8, d0\n\t{reload}\n"
                                      "\tadd sp, sp, #16\n\tret\n")
                self.assertTrue(any("without restoring" in error for error in analysis.errors), analysis.errors)
        for save in ("st1 {v8.1d}, [sp]", "st1 {v8.d}[0], [sp]", "str s8, [sp]"):
            with self.subTest(save=save):
                analysis = self.check(f"\tsub sp, sp, #16\n\t{save}\n\tfmov d8, d0\n\tldr d8, [sp]\n"
                                      "\tadd sp, sp, #16\n\tret\n")
                self.assertTrue(any("before saving" in error for error in analysis.errors), analysis.errors)

    def test_unmatched_callee_saved_reloads_are_spills(self) -> None:
        body = self.PROLOGUE + "\tsub sp, sp, 16\n\tfmov d8, d0\n\tstr d8, [sp]\n\tldr d8, [sp]\n\tadd sp, sp, 16\n" + self.EPILOGUE
        analysis = self.check(body)
        self.assertEqual((analysis.spills, analysis.errors), (2, []))


class StructuralTest(unittest.TestCase):
    """Closure and classification failures are fatal, independent of any spill allowance."""

    def assertFatal(self, body: str, fragment: str, extra: dict[str, str] | None = None) -> None:
        errors = analyse(body, extra).errors
        self.assertTrue(any(fragment in error for error in errors), (body, errors))

    def test_callees(self) -> None:
        self.assertFatal("\tbl helper\n\tret\n", "unsupported call boundary")
        self.assertFatal("\tb helper\n", "unsupported call boundary")
        for indirect in ("\tblr x8\n\tret\n", "\tbr x8\n", "\tret x1\n", "\tblraaz x8\n\tret\n", "\tudf #0\n\tret\n",
                         "\t<unknown>\n\tret\n"):
            with self.subTest(indirect=indirect):
                self.assertFatal(indirect, "admitted grammar")
        self.assertFatal("\tb .Lnowhere\n", "not code of this function")
        self.assertFatal("\tbl .Llocal\n\tret\n.Llocal:\n\tret\n", "not code of this function")

    def test_fatal_diagnostic_must_be_exact_and_proven(self) -> None:
        caller = f"\tcbz x0, .Lok\n\tbl {FATAL}\n.Lok:\n\tret\n"
        self.assertEqual(analyse(caller, {FATAL: FATAL_BODY}).errors, [])
        self.assertEqual(analyse(caller, {FATAL: FATAL_BODY.replace("fwrite", "fputs")}, macho=True).errors, [])
        both = "\tcbz x0, .La\n\tbl fputs\n\tbl abort\n.La:\n\tbl abort\n"
        self.assertEqual(analyse(caller, {FATAL: both}).errors, [])
        self.assertFatal(f"\tbl {FATAL[:-1]}X\n\tret\n", "unsupported call boundary")
        self.assertFatal(f"\tbl {FATAL}\n\tret\n", "unsupported call boundary")
        self.assertFatal("\tbl abort\n", "unsupported call boundary")
        for body in ("\tret\n", ".Lspin:\n\tb .Lspin\n", "\tmov x8, #93\n\tmov x0, #0\n\tsvc #0\n\tbl abort\n",
                     "\tbl exit\n\tbl abort\n", "\tb.eq .Lx\n\tbl abort\n.Lx:\n\tret\n", "\tcbz x0, .Ldone\n.Lspin:\n\tb .Lspin\n.Ldone:\n\tbl abort\n",
                     ".Lloop:\n\tstr q0, [sp, -16]!\n\tcbnz x0, .Lloop\n\tbl abort\n", "\tcbz x0, .Lr\n\tbl abort\n.Lr:\n\tret\n",
                     "\tb fputs\n", "\tblr x1\n", "\tbl fputs\n", "\t.word 0x0\n\tbl abort\n"):
            with self.subTest(body=body):
                self.assertNotIn(FATAL, analyse("\tret\n", {FATAL: body}).noreturn)
                self.assertFatal(f"\tbl {FATAL}\n\tret\n", "unsupported call boundary", {FATAL: body})

    def test_unclassifiable_code_is_fatal(self) -> None:
        self.assertFatal("\tldr q0, [sp, w1, ror 3]\n\tret\n", "addressing")
        self.assertFatal("\tcasp x0, x1, x2, x3, [sp]\n\tret\n", "admitted grammar")
        self.assertFatal("\tldr q0, [x1], x2\n\tret\n", "admitted grammar")
        self.assertFatal("\tadd\n\tret\n", "admitted grammar")
        self.assertFatal("\tadd x9\n\tret\n", "admitted grammar")
        self.assertFatal("\tadd x9, x1, x2, x3\n\tret\n", "admitted grammar")
        self.assertFatal("\tsmmla\n\tret\n", "admitted grammar")
        self.assertFatal("\tsvc #0\n\tret\n", "admitted grammar")
        self.assertFatal("\twfi\n\tret\n", "admitted grammar")
        self.assertFatal("\tfrobnicate x9, x1\n\tret\n", "admitted grammar")
        self.assertFatal("\tadd x9, x1, @@\n\tret\n", "admitted grammar")
        self.assertFatal("\tret\n\tstr q0, [sp]\n", "unreachable")
        self.assertFatal("\tmov x0, x1\n", "falls off")

    def test_data_words_in_code_are_fatal(self) -> None:
        frame = "\tsub sp, sp, #16\n{}\n\tadd sp, sp, #16\n\tret\n"
        for word in ("\t.word 0x3d8003e0", "\t.inst 0x3d8003e0", "\t.byte 0xe0", "\t.short 0x3"):
            with self.subTest(word=word):
                self.assertTrue(any("inside code" in error for error in analyse(frame.format(word)).errors))
        self.assertEqual(analyse("\tret\n\tnop\n\tnop\n").errors, [])  # Unreachable alignment padding.

    def test_body_without_executable_entry_section_fails(self) -> None:
        self.assertTrue(CHECKER.analyse_function("f", {"f": ["\tret"]}, CHECKER.GENERIC, set())[1])
        self.assertTrue(CHECKER.analyse_function("f", {"f": ["\t.section .rodata", "\tret"]}, CHECKER.GENERIC, set())[1])


class DisassemblyTest(unittest.TestCase):
    """The disassembly contract: every line, relocation and operand is classified or fails."""

    TEXT = disassembly([("f", "\tadd x0, x1, #1\n\tret\n")])

    def test_unparseable_skipped_and_misplaced_lines_fail(self) -> None:
        for bad in (self.TEXT + "garbage line\n", self.TEXT + "\t...\n",
                    self.TEXT + "\t\t0000000000000040:  R_AARCH64_CALL26\tx\n",
                    self.TEXT.replace("ret\n", "ret\n\t\t0000000000000004:  R_AARCH64_CALL26\tx\n"
                                                "\t\t0000000000000004:  R_AARCH64_CALL26\ty\n"),
                    self.TEXT.replace("Disassembly of section .text.f:\n", ""),
                    self.TEXT + self.TEXT.split("\n", 1)[0] + "\n",
                    self.TEXT.replace("ret\n", "ret\n\t\t0000000000000004:  R_AARCH64_MOVW_UABS_G0\tx\n"),
                    self.TEXT.replace("add\tx0, x1, #1", "ldr\tx0, 0x40 <f+0x40>")):
            with self.subTest(bad=bad[-80:]):
                with self.assertRaises(ValueError):
                    CHECKER.normalize_disassembly(bad, False)

    def test_branch_relocation_addends_fail_and_data_addends_are_dropped(self) -> None:
        call = disassembly([("f", "\tbl helper\n\tret\n")]).replace("\thelper\n", "\thelper+0x8\n")
        with self.assertRaisesRegex(ValueError, "addend"):
            CHECKER.normalize_disassembly(call, False)
        page = disassembly([("f", "\tadrp x1, 0x0 <f>\n\tret\n")]).replace(
            "<f>\n", "<f>\n\t\t0000000000000000:  R_AARCH64_ADR_PREL_PG_HI21\t.rodata.str1.8+0x38\n", 1)
        self.assertEqual(CHECKER.normalize_disassembly(page, False)["f"][1].strip(), "adrp\tx1, .rodata.str1.8")

    def test_object_format_must_match_the_disassembly(self) -> None:
        with self.assertRaises(ValueError):
            CHECKER.normalize_disassembly(self.TEXT, True)
        with self.assertRaises(ValueError):
            CHECKER.normalize_disassembly(disassembly([("f", "\tret\n")], macho=True), False)
        self.assertEqual(sorted(CHECKER.normalize_disassembly(disassembly([("f", "\tret\n")], macho=True), True)),
                         ["f"])  # Mach-O symbols lose their leading underscore.

    def test_gnu_objdump_format(self) -> None:
        gnu = ("probe.o:     file format elf64-littleaarch64\n\n\nDisassembly of section .text.f:\n\n"
               "0000000000000000 <f>:\n"
               "   0:\tadrp\tx3, 0 <stderr>\n\t\t\t0: R_AARCH64_ADR_GOT_PAGE\tstderr\n"
               "   4:\tldr\tx3, [x3]\n\t\t\t4: R_AARCH64_LD64_GOT_LO12_NC\tstderr\n"
               "   8:\tmov\tx2, #0x45                  \t// #69\n"
               "   c:\tcmp\tx2, #0x0\n"
               "  10:\tb.ne\t8 <f+0x8>  // b.any\n"
               "  14:\tbl\t0 <fwrite>\n\t\t\t14: R_AARCH64_CALL26\tfwrite\n"
               "  18:\tbl\t0 <abort>\n\t\t\t18: R_AARCH64_CALL26\tabort\n")
        bodies = CHECKER.normalize_disassembly(gnu, False)
        self.assertEqual([line.strip() for line in bodies["f"][1:]],
                         ["adrp\tx3, stderr", "ldr\tx3, [x3, :lo12:stderr]", ".L8:", "mov\tx2, #0x45",
                          "cmp\tx2, #0x0", "b.ne\t.L8", "bl\tfwrite", "bl\tabort"])

    def test_object_format_and_embedded_inventory(self) -> None:
        self.assertFalse(CHECKER.object_format(witness_object()))
        self.assertTrue(CHECKER.object_format(witness_object(macho=True)))
        foreign = bytearray(witness_object())
        foreign[18:20] = (62).to_bytes(2, "little")  # x86-64
        for data in (bytes(foreign), b"MZ" + bytes(64), b""):
            with self.assertRaises(ValueError):
                CHECKER.object_format(data + bytes(32))
        self.assertEqual(CHECKER.object_inventory(witness_object()), [NEON_INVENTORY])
        self.assertEqual(CHECKER.object_inventory(witness_object(None)), [])
        doubled = witness_object(extra=b"llaminar-native-neon-tile-probe:" + NEON_INVENTORY.encode() + b"\0")
        self.assertEqual(len(CHECKER.object_inventory(doubled)), 2)


class InventoryTest(unittest.TestCase):
    """The expected programs derive from the admitted envelope; object, symbol table and disassembly agree."""

    def test_envelope_inventory_matches_the_written_out_keys(self) -> None:
        for macho in (False, True):
            counts = audit_witness(macho=macho)
            self.assertEqual(sorted(tuple(map(int, CHECKER.program_key(n).split(":"))) for n in counts), NEON_KEYS)
        self.assertEqual(len(NEON_KEYS), 64)

    def test_missing_duplicate_empty_and_unexpected_programs(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            audit_witness(keys=NEON_KEYS[1:])
        with self.assertRaisesRegex(ValueError, "twice"):
            audit_witness(keys=NEON_KEYS + NEON_KEYS[:1])
        with self.assertRaisesRegex(ValueError, "empty"):
            audit_witness({NEON_KEYS[3]: ""})
        with self.assertRaisesRegex(ValueError, "outside the NEON inventory"):
            audit_witness(extra=[(name(1, 1, 0, 1, 0, 1), "\tstr q0, [sp, #-16]!\n\tret\n")])
        with self.assertRaisesRegex(ValueError, "unrecognized tile program"):
            audit_witness(extra=[(name(2, 0, 0, 1, 0, 1).replace("3runI", "3runIX"), "\tret\n")])

    def test_symbol_table_must_agree_with_the_disassembly(self) -> None:
        functions = programs()
        with self.assertRaisesRegex(ValueError, "symbol table"):
            audit_witness(table=functions[1:])
        with self.assertRaisesRegex(ValueError, "symbol table"):
            audit_witness(table=functions + [(name(1, 1, 0, 1, 0, 1), "")])

    def test_embedded_inventory_must_agree_with_the_envelope(self) -> None:
        with self.assertRaisesRegex(ValueError, "inventory"):
            audit_witness(inventory="2:ASIMD,3:ASIMD,4:ASIMD")
        with self.assertRaisesRegex(ValueError, "inventory"):
            audit_witness(mode="NEON_DOTPROD")
        with self.assertRaisesRegex(ValueError, "envelope"):
            audit_witness(mode="SVE")
        with self.assertRaisesRegex(ValueError, "inventory"):
            audit_witness(inventory=None)

    def test_structural_failure_is_not_paid_from_a_budget(self) -> None:
        key = (2, 0, 0, 1, 0, 1)
        with self.assertRaisesRegex(ValueError, "structurally"):
            audit_witness({key: "\tbl helper\n\tret"}, extra=[("helper", "\tret\n")])
        with self.assertRaisesRegex(ValueError, "structurally"):
            audit_witness({key: "\t.word 0x3d8003e0\n\tret"})
        counts = audit_witness({key: "\tstr q0, [sp, #-16]!\n\tadd sp, sp, #16\n\tret"})
        self.assertEqual(counts[name(*key)], 1)


class PolicyTest(unittest.TestCase):
    """Rows=1 must not spill; grouped programs ratchet against the identity/envelope budget."""

    def counts(self, grouped: int, single: int) -> dict[str, int]:
        return {name(*key): (single if key[0] == 1 else grouped) for key in NEON_KEYS}

    def test_single_row_spill_is_always_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "single-row"):
            CHECKER.require_registers(self.counts(0, 1), "GNU-14", "NEON", budget())

    def test_grouped_budget_is_a_ratchet(self) -> None:
        CHECKER.require_registers(self.counts(4, 0), "GNU-14", "NEON", budget())
        with self.assertRaisesRegex(ValueError, "budget"):
            CHECKER.require_registers(self.counts(5, 0), "GNU-14", "NEON", budget())

    def test_unknown_identity_envelope_missing_and_stale_budget_fail_closed(self) -> None:
        for identity, mode in (("GNU-15", "NEON"), ("GNU-14", "NEON_I8MM")):
            with self.assertRaisesRegex(ValueError, "no qualified"):
                CHECKER.require_registers(self.counts(0, 0), identity, mode, budget())
        missing = budget()
        missing["GNU-14"]["NEON"].pop("2:0:0:1:0:1")
        with self.assertRaisesRegex(ValueError, "no budget"):
            CHECKER.require_registers(self.counts(0, 0), "GNU-14", "NEON", missing)
        stale = budget()
        stale["GNU-14"]["NEON"]["3:1:0:1:0:0"] = 9
        with self.assertRaisesRegex(ValueError, "did not emit"):
            CHECKER.require_registers(self.counts(0, 0), "GNU-14", "NEON", stale)

    def test_committed_budget_is_well_formed(self) -> None:
        committed = json.loads(CHECKER.BUDGET_PATH.read_text())
        self.assertEqual(sorted(key for key in committed if not key.startswith("_")), ["AppleClang-21", "GNU-14"])
        for identity, envelopes in committed.items():
            if identity.startswith("_"):
                continue
            self.assertEqual(sorted(envelopes), ["NEON", "NEON_DOTPROD", "NEON_I8MM"])
            for mode, limits in envelopes.items():
                self.assertEqual(len(limits), {"NEON": 48, "NEON_DOTPROD": 96, "NEON_I8MM": 144}[mode])
                self.assertTrue(all(not key.startswith("1:") and value >= 0 for key, value in limits.items()))

    def test_compiler_identity_is_derived_from_the_compiler(self) -> None:
        self.assertEqual(CHECKER.compiler_identity("Apple clang version 21.0.0 (clang-2100.3.34.2)\nTarget: arm64"),
                         "AppleClang-21")
        self.assertEqual(CHECKER.compiler_identity("g++-14 (Ubuntu 14.2.0-4ubuntu2~24.04.1) 14.2.0\n"
                                                   "Copyright (C) 2024 Free Software Foundation, Inc.\n"), "GNU-14")
        self.assertEqual(CHECKER.compiler_identity("Ubuntu clang version 19.1.1 (1ubuntu1)\n"), "Clang-19")
        with self.assertRaises(ValueError):
            CHECKER.compiler_identity("icpx (ICX) 2025.0\n")


FAKE_COMPILER = """#!{python}
import pathlib, shutil, sys
here = pathlib.Path(__file__).parent
args = sys.argv[1:]
if args == ["--version"]:
    sys.stdout.write((here / "version.txt").read_text()); sys.exit(0)
(here / "invocations.txt").open("a").write(" ".join(args) + "\\n")
def value(flag):
    return args[args.index(flag) + 1]
compiling = "-c" in args
deps = (here / ("deps.c.txt" if compiling and (here / "deps.c.txt").exists() else "deps.txt")).read_text().split()
if compiling:
    mutate = here / "mutate.txt"
    if mutate.exists():
        target = pathlib.Path(mutate.read_text().strip())
        target.write_text(target.read_text() + "// changed during the build\\n")
    shutil.copy(here / "witness.o", value("-o"))
pathlib.Path(value("-MF")).write_text(value("-MT") + ": " + " ".join(deps) + "\\n")
"""
FAKE_DISASSEMBLER = """#!{python}
import pathlib, sys
here = pathlib.Path(__file__).parent
args = sys.argv[1:]
if args == ["--version"]:
    sys.stdout.write((here / "objdump-version.txt").read_text()); sys.exit(0)
source = "symbols.txt" if args[0] == "-t" else "disassembly.txt"
sys.stdout.write((here / source).read_text().replace("probe.o", args[-1]))
"""


class ProofBindingTest(unittest.TestCase):
    """build()/verify() bind the compilation's closure, identity, command, object, disassembler and budget."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = root = Path(self.temporary.name)
        self.tool = root / "tool"
        self.tool.mkdir()
        self.compiler = self.tool / "c++"
        self.compiler.write_text(FAKE_COMPILER.format(python=sys.executable))
        self.compiler.chmod(self.compiler.stat().st_mode | stat.S_IEXEC)
        (self.tool / "version.txt").write_text("g++-14 (Fixture 14.2.0) 14.2.0\nCopyright Free Software Foundation\n")
        self.disassembler = self.tool / "objdump"
        self.disassembler.write_text(FAKE_DISASSEMBLER.format(python=sys.executable))
        self.disassembler.chmod(self.disassembler.stat().st_mode | stat.S_IEXEC)
        (self.tool / "objdump-version.txt").write_text("GNU objdump (Fixture) 2.42\n")
        self.witness(programs())
        self.source, self.header, self.real = root / "probe.cpp", root / "view.h", root / "real" / "view.h"
        self.real.parent.mkdir()
        self.source.write_text('#include "view.h"\n')
        self.real.write_text("// accessor\n")
        self.header.symlink_to(self.real)
        (self.tool / "deps.txt").write_text("probe.cpp view.h")
        database = root / "compile_commands.json"
        database.write_text(json.dumps([{"directory": str(root), "file": str(self.source),
                                         "arguments": [str(self.compiler), "-O2", "-c", str(self.source), "-o", "x.o"]}]))
        self.args = argparse.Namespace(source=str(self.source), proof=str(root / "out" / "proof.json"), signature="sig",
                                       compiler=str(self.compiler), compile_database=str(database),
                                       target_source=str(self.source), identity="GNU-14", mode="NEON",
                                       depfile_target="out/proof.json", disassembler=str(self.disassembler))
        self.original_budget = CHECKER.BUDGET_PATH
        CHECKER.BUDGET_PATH = root / "budget.json"
        CHECKER.BUDGET_PATH.write_text(json.dumps(budget()))

    def tearDown(self) -> None:
        CHECKER.BUDGET_PATH = self.original_budget
        self.temporary.cleanup()

    def witness(self, functions: list[tuple[str, str]]) -> None:
        (self.tool / "witness.o").write_bytes(witness_object())
        (self.tool / "disassembly.txt").write_text(disassembly(functions))
        (self.tool / "symbols.txt").write_text(symbol_table(functions))

    def proof(self) -> dict:
        return json.loads(Path(self.args.proof).read_text())

    def test_build_binds_the_compilation_closure_and_verify_detects_changes(self) -> None:
        CHECKER.build(self.args)
        CHECKER.verify(self.args)
        proof = self.proof()
        self.assertEqual(proof["identity"], "GNU-14")
        self.assertIn("-MD", proof["executed_command"])
        self.assertEqual(sorted(proof["inputs"]), sorted([str(self.source), str(self.header)]))
        self.assertIn(str(self.real), [target for _, target in proof["inputs"][str(self.header)]["links"]])
        depfile = Path(self.args.proof).with_suffix(".d").read_text()
        self.assertTrue(depfile.startswith("out/proof.json: ") and os.path.realpath(self.real) in depfile)
        self.real.write_text("// accessor changed\n")
        with self.assertRaisesRegex(ValueError, "stale tile register input"):
            CHECKER.verify(self.args)
        self.real.write_text("// accessor\n")
        CHECKER.verify(self.args)
        twin = self.root / "real" / "twin.h"
        twin.write_text("// accessor\n")
        self.header.unlink()
        self.header.symlink_to(twin)
        with self.assertRaisesRegex(ValueError, "symlink"):
            CHECKER.verify(self.args)

    def test_symlink_then_parent_component_names_the_filesystem_target(self) -> None:
        link_root, real_root = self.root / "p", self.root / "q"
        (real_root / "sub").mkdir(parents=True)
        link_root.mkdir()
        (link_root / "link").symlink_to(real_root / "sub")
        real, decoy = real_root / "h.h", link_root / "h.h"
        real.write_text("// real\n")
        decoy.write_text("// real\n")
        (self.tool / "deps.txt").write_text("probe.cpp view.h p/link/../h.h")
        CHECKER.build(self.args)
        identity = self.proof()["inputs"][str(self.root / "p" / "link" / ".." / "h.h")]
        self.assertEqual(identity["resolved"], os.path.realpath(real))
        self.assertIn(os.path.realpath(real), Path(self.args.proof).with_suffix(".d").read_text())
        decoy.write_text("// decoy changed\n")
        CHECKER.verify(self.args)
        real.write_text("// real changed\n")
        with self.assertRaisesRegex(ValueError, "stale tile register input"):
            CHECKER.verify(self.args)

    def test_probe_source_spelling_through_a_symlink_is_preserved(self) -> None:
        (self.root / "q" / "sub").mkdir(parents=True)
        (self.root / "p").mkdir()
        (self.root / "p" / "link").symlink_to(self.root / "q" / "sub")
        (self.root / "q" / "probe.cpp").write_text('#include "view.h"\n')
        (self.root / "p" / "probe.cpp").write_text("// decoy\n")
        self.args.source = str(self.root / "p" / "link" / ".." / "probe.cpp")
        (self.tool / "deps.txt").write_text("p/link/../probe.cpp view.h")
        self.args.target_source = str(self.source)
        CHECKER.build(self.args)
        CHECKER.verify(self.args)
        (self.root / "p" / "probe.cpp").write_text("// decoy changed\n")
        CHECKER.verify(self.args)
        (self.root / "q" / "probe.cpp").write_text("// probe changed\n")
        with self.assertRaisesRegex(ValueError, "stale tile register input"):
            CHECKER.verify(self.args)

    def test_relative_probe_paths_are_rejected_before_compiling(self) -> None:
        for source in ("probe.cpp", "p/link/../probe.cpp"):
            with self.subTest(source=source):
                self.args.source = source
                with self.assertRaisesRegex(ValueError, "absolute"):
                    CHECKER.build(self.args)
                self.assertFalse((self.tool / "invocations.txt").exists())
                with self.assertRaisesRegex(ValueError, "absolute"):
                    CHECKER.verify(self.args)

    def test_object_and_disassembler_are_bound(self) -> None:
        CHECKER.build(self.args)
        CHECKER.verify(self.args)
        proof, out = self.proof(), Path(self.args.proof)
        self.assertIn("-c", proof["executed_command"])
        self.assertEqual(proof["disassembly_command"][-1], str(out.with_suffix(".o")))
        original = out.with_suffix(".o").read_bytes()
        out.with_suffix(".o").write_bytes(original + b"\0")
        with self.assertRaisesRegex(ValueError, "object"):
            CHECKER.verify(self.args)
        out.with_suffix(".o").write_bytes(original)
        CHECKER.verify(self.args)
        retained = out.with_suffix(".dis").read_text()
        out.with_suffix(".dis").write_text(retained.replace("sdot", "sdot "))
        with self.assertRaisesRegex(ValueError, "retained disassembly"):
            CHECKER.verify(self.args)
        out.with_suffix(".dis").write_text(retained)
        (self.tool / "disassembly.txt").write_text((self.tool / "disassembly.txt").read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "retained disassembly"):
            CHECKER.verify(self.args)
        (self.tool / "disassembly.txt").write_text(disassembly(programs()))
        CHECKER.verify(self.args)
        self.disassembler.write_text(self.disassembler.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "disassembler"):
            CHECKER.verify(self.args)
        out.with_suffix(".o").unlink()
        with self.assertRaisesRegex(ValueError, "object"):
            CHECKER.verify(self.args)

    def test_unsupported_disassembler_fails(self) -> None:
        (self.tool / "objdump-version.txt").write_text("Some Other Disassembler 1.0\n")
        with self.assertRaisesRegex(ValueError, "disassembler"):
            CHECKER.build(self.args)
        self.assertFalse(Path(self.args.proof).exists())

    def test_recorded_command_identity_budget_and_assembly_are_bound(self) -> None:
        CHECKER.build(self.args)
        path, original = Path(self.args.proof), self.proof()
        for field, value in (("executed_command", original["executed_command"][:-1] + ["other"]),
                             ("compiler_version", "g++-13 (Fixture 13.1.0) 13.1.0\nFree Software Foundation\n"),
                             ("identity", "GNU-15")):
            with self.subTest(field=field):
                path.write_text(json.dumps(dict(original, **{field: value})))
                with self.assertRaises(ValueError):
                    CHECKER.verify(self.args)
        path.write_text(json.dumps(original))
        CHECKER.BUDGET_PATH.write_text(json.dumps(budget(limit=9)))
        with self.assertRaisesRegex(ValueError, "stale tile register tool"):
            CHECKER.verify(self.args)
        CHECKER.BUDGET_PATH.write_text(json.dumps(budget()))
        CHECKER.verify(self.args)

    def test_build_rejects_foreign_identity_moving_inputs_and_closure_mismatch(self) -> None:
        (self.tool / "version.txt").write_text("g++-13 (Fixture 13.1.0) 13.1.0\nFree Software Foundation\n")
        with self.assertRaisesRegex(ValueError, "identity"):
            CHECKER.build(self.args)
        (self.tool / "version.txt").write_text("g++-14 (Fixture 14.2.0) 14.2.0\nFree Software Foundation\n")
        (self.tool / "mutate.txt").write_text(str(self.real))
        with self.assertRaisesRegex(ValueError, "changed while"):
            CHECKER.build(self.args)
        (self.tool / "mutate.txt").unlink()
        (self.tool / "deps.c.txt").write_text("probe.cpp")
        with self.assertRaisesRegex(ValueError, "differs"):
            CHECKER.build(self.args)
        self.assertFalse(Path(self.args.proof).exists())

    def test_failed_audit_publishes_no_receipt(self) -> None:
        CHECKER.build(self.args)
        self.witness(programs({(1, 0, 0, 0, 0, 0): "\tstr q0, [sp, #-16]!\n\tadd sp, sp, #16\n\tret"}))
        with self.assertRaisesRegex(ValueError, "single-row"):
            CHECKER.build(self.args)
        self.assertFalse(Path(self.args.proof).exists())


if __name__ == "__main__":
    unittest.main()
