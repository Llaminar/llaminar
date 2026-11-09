#!/usr/bin/env python3
"""Compiler-free adversarial tests of the native NEON tile register-allocation witness.

Synthetic assembly exercises classification (every stack base and width, ABI
saves), the expected program inventory, callee closure, the single-row/ratchet
policy and proof binding (include closure, executed command, budget, assembly).
"""
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


def name(rows: int, dot: int, policy: int, encoding: int, nibble_map: int, minimum: int) -> str:
    """Mangled tile run<> symbol for one program key."""
    return (f"_ZNK9llaminar23cpu11native_quantised27CPUNativeNEONProjectionTileILm{rows}EE3runINS1_11multi_scale17"
            f"NEONMultiScaleOpsILNS5_11NEONByteDotE{dot}EEELNS_28CPUProjectionNumericalPolicyE{policy}ELNS1_21"
            f"CPUNativeQuantisedEncodingE{encoding}ELNS3_9NibbleMapE{nibble_map}ELb{minimum}EEEvv")


def keys(inventory: str) -> list[tuple]:
    """Expected program keys for an inventory string (mirrors the checker's expansion)."""
    result = []
    for entry in inventory.split(","):
        rows, dot = entry.split(":")
        leaf = 1 if dot == "I8MM" and rows == "1" else CHECKER.DOTS[dot]
        for policy in (0, 1):
            for encoding, nibble_map, minimum in CHECKER.DISPATCH:
                result.append((int(rows), leaf, policy, encoding, nibble_map, minimum))
    return result


def assembly(inventory: str, bodies: dict[tuple, str] | None = None, extra: str = "") -> str:
    """A complete witness: inventory string, every expected program (clean unless overridden), extra functions."""
    bodies = bodies or {}
    text = [f'\t.ascii\t"llaminar-native-neon-tile-probe:{inventory}"']
    for key in keys(inventory):
        text.append(f"{name(*key)}:")
        text.append(bodies.get(key, "\tstp\tx29, x30, [sp, -16]!\n\tmov\tx29, sp\n\tsdot\tv0.4s, v1.16b, v2.16b\n"
                                     "\tldp\tx29, x30, [sp], 16\n\tret"))
    return "\n".join(text) + "\n" + extra


def budget(inventory: str, limit: int = 4) -> dict:
    """Budget naming every grouped program of an inventory."""
    limits = {CHECKER.program_key(name(*key)): limit for key in keys(inventory) if key[0] != 1}
    return {"GNU-14": {"NEON_I8MM": limits}}


class ClassificationTest(unittest.TestCase):
    """Stack traffic through any sp/fp-derived base and any FP/SIMD width is a spill; ABI saves are not."""

    def spills(self, body: str) -> int:
        return CHECKER.stack_traffic(body.splitlines())[0]

    def test_every_base_and_width_is_classified(self) -> None:
        for line in ("str q31, [sp, 160]", "ldr q25, [sp, #0xa0]", "stp q0, q1, [sp, 32]", "stur q0, [x29, -16]",
                     "str q0, [x29, #16]", "str s3, [sp, 12]", "ldr d17, [x29, 40]", "str h2, [sp, 4]",
                     "st1 {v0.16b}, [sp]", "ldr b1, [sp, 1]"):
            with self.subTest(line=line):
                self.assertEqual(self.spills("\t" + line), 1)

    def test_sp_and_fp_derived_aliases_are_followed_and_retired(self) -> None:
        self.assertEqual(self.spills("\tadd x9, sp, #64\n\tstr q0, [x9]"), 1)
        self.assertEqual(self.spills("\tmov x10, sp\n\tldr q4, [x10, 32]"), 1)
        self.assertEqual(self.spills("\tsub x11, x29, #32\n\tstr d18, [x11]"), 1)
        self.assertEqual(self.spills("\tadd x9, sp, #64\n\tadd x9, x1, #8\n\tstr q0, [x9]"), 0)
        self.assertEqual(self.spills("\tldr q0, [x1, 16]\n\tstr q0, [x2]"), 0)

    def test_abi_saves_are_distinguished_from_spills(self) -> None:
        save = "\tstp d8, d9, [sp, 16]\n\tstr q0, [sp, 48]\n\tldp d8, d9, [sp, 16]\n\tret"
        spills, saves, restores, returns = CHECKER.stack_traffic(save.splitlines())
        self.assertEqual((spills, saves, restores, returns), (1, {"d8": 1, "d9": 1}, {"d8": 1, "d9": 1}, 1))
        self.assertEqual(self.spills("\tstr d7, [sp, 8]"), 1)  # Caller-saved: a spill.
        self.assertEqual(self.spills("\tstr q8, [sp, 8]"), 1)  # Full q8 is not the ABI save.


class InventoryAndClosureTest(unittest.TestCase):
    """Every expected program must exist once, be non-empty, and its callees must be clean."""

    def test_complete_clean_witness_passes(self) -> None:
        counts = CHECKER.audit(assembly("1:ASIMD,2:DotProd"))
        self.assertEqual(len(counts), 32)
        CHECKER.require_registers(counts, "GNU-14", "NEON_I8MM", budget("1:ASIMD,2:DotProd"))

    def test_missing_duplicate_empty_and_absent_inventory_are_rejected(self) -> None:
        full = assembly("1:ASIMD")
        missing = full.replace(f"{name(1, 0, 1, 2, 0, 0)}:", "unrelated:")
        with self.assertRaises(ValueError):
            CHECKER.audit(missing)
        with self.assertRaises(ValueError):
            CHECKER.audit(full + f"{name(1, 0, 0, 1, 0, 0)}:\n\tret\n")
        with self.assertRaises(ValueError):
            CHECKER.audit(assembly("1:ASIMD", {(1, 0, 0, 5, 0, 0): "\t.cfi_startproc"}))
        with self.assertRaises(ValueError):
            CHECKER.audit(full.replace("llaminar-native-neon-tile-probe", "something-else"))

    def test_partial_spill_in_one_program_is_detected(self) -> None:
        counts = CHECKER.audit(assembly("1:ASIMD,2:ASIMD", {(2, 0, 1, 4, 2, 0): "\tstr q3, [x29, 32]\n\tret"}))
        self.assertEqual(sum(counts.values()), 1)
        with self.assertRaises(ValueError):
            CHECKER.require_registers(counts, "GNU-14", "NEON_I8MM", budget("1:ASIMD,2:ASIMD", limit=0))

    def test_outlined_callee_external_callee_and_fatal_diagnostic(self) -> None:
        helper = "helper_body:\n\tstr q9, [sp, 16]\n\tret\n"
        outlined = {(1, 0, 0, 0, 0, 0): "\tbl helper_body\n\tret"}
        self.assertEqual(sum(CHECKER.audit(assembly("1:ASIMD", outlined, helper)).values()), 1)
        tail = {(1, 0, 0, 0, 0, 0): "\tb helper_body"}
        self.assertEqual(sum(CHECKER.audit(assembly("1:ASIMD", tail, helper)).values()), 1)
        external = {(1, 0, 0, 0, 0, 0): "\tbl memcpy\n\tret"}
        self.assertEqual(sum(CHECKER.audit(assembly("1:ASIMD", external)).values()), 1)
        fatal = {(1, 0, 0, 0, 0, 0): "\tbl _ZN9llaminar23cpu11native_quantised23CPUNativeQuantisedWeightView32rejectOutsideAdmittedPublicationEv\n\tret"}
        self.assertEqual(sum(CHECKER.audit(assembly("1:ASIMD", fatal)).values()), 0)

    def test_gnu_and_apple_local_labels_stay_inside_their_function(self) -> None:
        body = "LBB0_3:\n\tsdot v0.4s, v1.16b, v2.16b\n.L7:\nLtmp4:\n\tstr q3, [sp, #48]\n\tret"
        self.assertEqual(sum(CHECKER.audit(assembly("1:ASIMD", {(1, 0, 0, 0, 0, 0): body})).values()), 1)


class PolicyTest(unittest.TestCase):
    """Single-row programs never spill; grouped programs stay within the identity/envelope ratchet."""

    def setUp(self) -> None:
        self.inventory = "1:DotProd,2:DotProd"
        self.limits = budget(self.inventory, limit=2)

    def counts(self, rows: int, spills: int) -> dict[str, int]:
        result = {name(*key): 0 for key in keys(self.inventory)}
        result[name(rows, 1, 0, 1, 0, 0)] = spills
        return result

    def test_single_row_spill_is_always_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CHECKER.require_registers(self.counts(1, 1), "GNU-14", "NEON_I8MM", self.limits)

    def test_grouped_budget_is_a_ratchet(self) -> None:
        CHECKER.require_registers(self.counts(2, 2), "GNU-14", "NEON_I8MM", self.limits)
        with self.assertRaises(ValueError):
            CHECKER.require_registers(self.counts(2, 3), "GNU-14", "NEON_I8MM", self.limits)

    def test_unknown_identity_envelope_missing_and_stale_budget_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            CHECKER.require_registers(self.counts(2, 0), "GNU-15", "NEON_I8MM", self.limits)
        with self.assertRaises(ValueError):
            CHECKER.require_registers(self.counts(2, 0), "GNU-14", "NEON", self.limits)
        missing = json.loads(json.dumps(self.limits))
        missing["GNU-14"]["NEON_I8MM"].pop(CHECKER.program_key(name(2, 1, 0, 1, 0, 0)))
        with self.assertRaises(ValueError):
            CHECKER.require_registers(self.counts(2, 0), "GNU-14", "NEON_I8MM", missing)
        stale = json.loads(json.dumps(self.limits))
        stale["GNU-14"]["NEON_I8MM"]["3:1:0:1:0:0"] = 9
        with self.assertRaises(ValueError):
            CHECKER.require_registers(self.counts(2, 0), "GNU-14", "NEON_I8MM", stale)

    def test_committed_budget_is_well_formed(self) -> None:
        committed = json.loads(CHECKER.BUDGET_PATH.read_text())
        for identity, envelopes in committed.items():
            if identity.startswith("_"):
                continue
            for mode, limits in envelopes.items():
                self.assertTrue(limits, f"{identity}/{mode}")
                self.assertTrue(all(not key.startswith("1:") and value >= 0 for key, value in limits.items()))


class ProofBindingTest(unittest.TestCase):
    """verify() rejects changed headers, commands, budgets and assembly without a compiler."""

    def test_header_command_budget_and_assembly_changes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, header = root / "probe.cpp", root / "CPUNativeQuantisedWeightView.h"
            proof, asm = root / "proof.json", root / "proof.s"
            source.write_text('#include "CPUNativeQuantisedWeightView.h"\n')
            header.write_text("// accessor\n")
            inventory = "1:ASIMD,2:ASIMD"
            asm.write_text(assembly(inventory))
            compiler = root / "compiler"
            database = root / "compile_commands.json"
            database.write_text(json.dumps([{"directory": str(root), "file": str(source),
                "arguments": [str(compiler), "-O2", "-c", str(source), "-o", "x.o"]}]))
            args = argparse.Namespace(source=str(source), proof=str(proof), signature="sig", compiler=str(compiler),
                compile_database=str(database), target_source=str(source), identity="GNU-14", mode="NEON_I8MM")
            command, cwd = CHECKER.envelope.target_command(args)
            original_budget = CHECKER.BUDGET_PATH
            budget_file = root / "budget.json"
            budget_file.write_text(json.dumps(budget(inventory)))
            CHECKER.BUDGET_PATH = budget_file
            try:
                receipt = {
                    "schema": 3, "signature": "sig", "identity": "GNU-14", "mode": "NEON_I8MM",
                    "checker_sha256": CHECKER.envelope.digest(CHECKER_PATH),
                    "envelope_sha256": CHECKER.envelope.digest(CHECKER._ENVELOPE),
                    "budget_sha256": CHECKER.envelope.digest(budget_file),
                    "compiler_version": "fixture", "executed_command": command + [
                        "-O3", "-fno-sanitize=all", "-fno-lto", "-std=c++20", "-S", str(source), "-o", str(asm)],
                    "executed_directory": cwd, "target_command_sha256": CHECKER.envelope.command_digest(command, cwd),
                    "inputs": CHECKER.bind_inputs([str(source.resolve()), str(header.resolve())]),
                    "assembly_file": "proof.s", "assembly_sha256": CHECKER.envelope.digest(asm),
                    "program_violations": CHECKER.audit(asm.read_text()),
                }
                proof.write_text(json.dumps(receipt))
                CHECKER.verify(args)
                header.write_text("// accessor changed\n")
                with self.assertRaises(ValueError):
                    CHECKER.verify(args)
                header.write_text("// accessor\n")
                CHECKER.verify(args)
                tampered = dict(receipt, executed_command=receipt["executed_command"][:-4] + ["-O0", "-S", str(source), "-o", str(asm)])
                proof.write_text(json.dumps(tampered))
                with self.assertRaises(ValueError):
                    CHECKER.verify(args)
                proof.write_text(json.dumps(receipt))
                budget_file.write_text(json.dumps(budget(inventory, limit=9)))
                with self.assertRaises(ValueError):
                    CHECKER.verify(args)
                budget_file.write_text(json.dumps(budget(inventory)))
                asm.write_text(assembly(inventory, {(2, 0, 0, 0, 0, 0): "\tstr q0, [sp, 16]\n\tret"}))
                with self.assertRaises(ValueError):
                    CHECKER.verify(args)
            finally:
                CHECKER.BUDGET_PATH = original_budget


if __name__ == "__main__":
    unittest.main()
