#!/usr/bin/env python3
"""Compiler-free adversarial tests of the native NEON tile register-allocation witness."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

CHECKER_PATH = Path(__file__).resolve().parents[2] / "cmake" / "CPUNativeTileRegisterEvidence.py"
SPEC = importlib.util.spec_from_file_location("cpu_native_tile_register_evidence", CHECKER_PATH)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)

RUN = "_ZNK9llaminar23cpu11native_quantised27CPUNativeNEONProjectionTileILm4EE3runINS1_11multi_scale17OpsEEEvv"
PAIR = "_ZNK9llaminar23cpu11native_quantised27CPUNativeNEONProjectionTileILm2EE9pairBlockINS1_OpsEEEvv"
CLEAN = f"""{RUN}:
\tstp\tx29, x30, [sp, -96]!
\tstp\td8, d9, [sp, 16]
\tsdot\tv0.4s, v1.16b, v2.16b
\tret
{PAIR}:
\tsmmla\tv0.4s, v1.16b, v2.16b
\tret
other_function:
\tstr\tq0, [sp, 16]
\tret
"""


class CPUNativeTileRegisterEvidenceTest(unittest.TestCase):
    """The witness must detect every vector spill form and fail closed on stale evidence."""

    def test_clean_programs_pass_and_unrelated_functions_are_ignored(self) -> None:
        counts = CHECKER.program_spills(CLEAN)
        self.assertEqual(counts, {RUN: 0, PAIR: 0})
        CHECKER.require_registers(counts)

    def test_every_vector_spill_form_is_rejected(self) -> None:
        for spill in ("\tstr\tq31, [sp, 160]", "\tldr\tq25, [sp, #0xa0]", "\tstp\tq0, q1, [sp, 32]",
                      "\tldur\tq2, [sp, 8]", "\tst1\t{v0.16b}, [sp]", "\tldr\tq7, [sp]"):
            with self.subTest(spill=spill):
                assembly = CLEAN.replace("\tsdot\tv0.4s, v1.16b, v2.16b", spill)
                counts = CHECKER.program_spills(assembly)
                self.assertEqual(counts[RUN], 1)
                with self.assertRaises(ValueError):
                    CHECKER.require_registers(counts)

    def test_apple_symbol_prefix_comments_and_local_labels_are_parsed(self) -> None:
        assembly = (f"_{RUN}:                ; @fn\nLFB12:\n\tsdot\tv0.4s, v1.16b, v2.16b\nLBB0_3:\n"
                    "Ltmp4:\n\tstr\tq3, [sp, #48]\n\tret\nLFE12:\n")
        self.assertEqual(CHECKER.program_spills(assembly), {"_" + RUN: 1})
        gnu = f"{RUN}:\n.LFB7:\n.L3:\n\tldr\tq5, [sp, 32]\n\tret\n.LFE7:\n"
        self.assertEqual(CHECKER.program_spills(gnu), {RUN: 1})

    def test_witness_without_programs_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CHECKER.require_registers(CHECKER.program_spills("other_function:\n\tret\n"))

    def test_verify_rejects_stale_or_modified_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, assembly, proof = root / "probe.cpp", root / "proof.s", root / "proof.json"
            source.write_text("int probe;\n")
            assembly.write_text(CLEAN)
            database = root / "compile_commands.json"
            compiler = root / "compiler"
            database.write_text(json.dumps([{"directory": str(root), "file": str(source),
                "arguments": [str(compiler), "-O2", "-c", str(source), "-o", "x.o"]}]))
            args = argparse.Namespace(source=str(source), proof=str(proof), signature="sig",
                compiler=str(compiler), compile_database=str(database), target_source=str(source))
            receipt = {
                "schema": 1, "signature": "sig", "source_sha256": CHECKER.envelope.digest(source),
                "checker_sha256": CHECKER.envelope.digest(CHECKER_PATH),
                "envelope_sha256": CHECKER.envelope.digest(CHECKER._ENVELOPE),
                "target_command_sha256": CHECKER.envelope.command_digest(*CHECKER.envelope.target_command(args)),
                "assembly_file": "proof.s", "assembly_sha256": CHECKER.envelope.digest(assembly),
                "program_spills": CHECKER.program_spills(CLEAN),
            }
            proof.write_text(json.dumps(receipt))
            CHECKER.verify(args)
            assembly.write_text(CLEAN.replace("\tret\n", "\tstr\tq0, [sp, 16]\n\tret\n", 1))
            with self.assertRaises(ValueError):
                CHECKER.verify(args)
            assembly.write_text(CLEAN)
            source.write_text("int changed;\n")
            with self.assertRaises(ValueError):
                CHECKER.verify(args)


if __name__ == "__main__":
    unittest.main()
