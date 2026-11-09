#!/usr/bin/env python3
"""Device/compiler-free adversarial tests of retained CPU code-generation evidence.

Compilation belongs to the build-time witness. These tests mutate retained
source/config/assembly inputs and require runtime verification to reject stale,
missing or contradictory artifacts without invoking any compiler or device.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

CHECKER_PATH = Path(__file__).resolve().parents[2] / "cmake" / "CPUCodegenEnvelope.py"
SPEC = importlib.util.spec_from_file_location("cpu_codegen_envelope", CHECKER_PATH)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class CPUCodegenEnvelopeEvidenceTest(unittest.TestCase):
    """Prove the receipt verifier fails closed independently of build tooling."""

    def setUp(self) -> None:
        """Own one tiny isolated source/policy/proof fixture per adversarial case."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.source = root / "source.cpp"
        self.policy = root / "policy.cmake"
        self.proof = root / "proof.json"
        self.assembly = root / "proof.s"
        self.manifest = root / "target.txt"
        self.manifest.write_text("exact target options\n")
        self.source.write_text("int example;\n")
        self.policy.write_text("# exact envelope\n")
        self.assembly.write_text("native_cpu_codegen_byte_dot:\n\tsmull.8h v0, v1, v2\n\tret\n")
        self.args = argparse.Namespace(source=str(self.source), policy=str(self.policy),
            proof=str(self.proof), mode="NEON", signature="fixture-config",
            target_manifest=str(self.manifest), target_source=str(self.source),
            compile_database=str(root / "compile_commands.json"))
        self.args.compiler = str(root / "compiler")
        Path(self.args.compile_database).write_text(json.dumps([{
            "directory": str(root), "file": str(self.source),
            "arguments": [self.args.compiler, "-mcpu=generic", "-c", str(self.source), "-o", "unused.o"],
        }]))
        command, directory = CHECKER.target_command(self.args)
        self.receipt = {
            "schema": 1, "mode": "NEON", "signature": "fixture-config",
            "source_sha256": CHECKER.digest(self.source), "policy_sha256": CHECKER.digest(self.policy),
            "checker_sha256": CHECKER.digest(CHECKER_PATH),
            "target_manifest_file": self.manifest.name, "target_manifest_sha256": CHECKER.digest(self.manifest),
            "target_command_sha256": CHECKER.command_digest(command, directory),
            "assembly_file": "proof.s", "assembly_sha256": CHECKER.digest(self.assembly),
            "instruction_counts": CHECKER.instruction_counts(self.assembly.read_text()),
        }
        self.save()

    def save(self) -> None:
        """Publish exactly the current synthetic evidence, with no compiler dependency."""
        self.proof.write_text(json.dumps(self.receipt))

    def test_current_receipt_is_accepted(self) -> None:
        """A complete source/config-bound baseline witness verifies as supplied."""
        CHECKER.verify(self.args)

    def test_source_policy_checker_and_signature_changes_are_rejected(self) -> None:
        """Every bound input participates in freshness rather than pointer/path presence."""
        for key in ("source_sha256", "policy_sha256", "checker_sha256", "signature", "schema", "mode"):
            with self.subTest(key=key):
                original = self.receipt[key]
                self.receipt[key] = "foreign"
                self.save()
                with self.assertRaises(ValueError):
                    CHECKER.verify(self.args)
                self.receipt[key] = original
        self.save()
        self.source.write_text("int changed;\n")
        with self.assertRaises(ValueError):
            CHECKER.verify(self.args)

    def test_final_target_option_changes_are_rejected(self) -> None:
        """Target/configuration changes invalidate otherwise unchanged source evidence."""
        self.manifest.write_text("options=-mcpu=apple-m1\n")
        with self.assertRaises(ValueError):
            CHECKER.verify(self.args)

    def test_tu_specific_command_change_is_rejected(self) -> None:
        """Even a later source-level ISA flag cannot hide behind an unchanged global manifest."""
        path = Path(self.args.compile_database)
        entries = json.loads(path.read_text())
        entries[0]["arguments"].insert(-2, "-mcpu=apple-m1")
        path.write_text(json.dumps(entries))
        with self.assertRaises(ValueError):
            CHECKER.verify(self.args)

    def test_nested_response_changes_cycles_and_missing_inputs_are_rejected(self) -> None:
        """An inner response file participates even when the outer/DB/manifest stay unchanged."""
        root = self.proof.parent
        outer, inner = root / "outer.rsp", root / "inner.rsp"
        outer.write_text("@inner.rsp")
        inner.write_text("-mcpu=generic")
        database = Path(self.args.compile_database)
        entries = json.loads(database.read_text())
        entries[0]["arguments"][1] = "@outer.rsp"
        database.write_text(json.dumps(entries))
        command, directory = CHECKER.target_command(self.args)
        self.receipt["target_command_sha256"] = CHECKER.command_digest(command, directory)
        self.save()
        CHECKER.verify(self.args)
        inner.write_text("-mcpu=apple-m1")
        with self.assertRaises(ValueError):
            CHECKER.verify(self.args)
        inner.write_text("@outer.rsp")
        with self.assertRaises(ValueError):
            CHECKER.verify(self.args)
        inner.unlink()
        with self.assertRaises(FileNotFoundError):
            CHECKER.verify(self.args)

    def test_empty_and_ir_only_witnesses_are_rejected(self) -> None:
        """Zero optional-instruction counts alone are not a native ISA witness."""
        for text in ("", "; ModuleID = 'probe'\\ntarget triple = arm64\\n",
                     "define i32 @native_cpu_codegen_byte_dot() { ret i32 0 }",
                     '.section .gnu.lto_fixture,"e"'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                CHECKER.require_native_assembly(text)

    def test_missing_or_modified_assembly_is_rejected(self) -> None:
        """A retained receipt alone is insufficient without its exact assembly."""
        self.assembly.write_text("\tadd.4s v0, v1, v2\n")
        with self.assertRaises(ValueError):
            CHECKER.verify(self.args)
        self.assembly.unlink()
        with self.assertRaises(FileNotFoundError):
            CHECKER.verify(self.args)

    def test_baseline_optional_isa_is_rejected_even_with_consistent_hashes(self) -> None:
        """The actual SDOT regression cannot be hidden by refreshing artifact hashes."""
        self.assembly.write_text("native_cpu_codegen_byte_dot:\n\tsdot.4s v0, v1, v2\n\tret\n")
        self.receipt["assembly_sha256"] = CHECKER.digest(self.assembly)
        self.receipt["instruction_counts"] = CHECKER.instruction_counts(self.assembly.read_text())
        self.save()
        with self.assertRaises(ValueError):
            CHECKER.verify(self.args)

    def test_failed_rebuild_retires_previous_success(self) -> None:
        """Even an early compiler failure makes old success evidence unusable."""
        self.args.compiler = str(self.proof.parent / "missing-compiler")
        Path(self.args.compile_database).write_text(json.dumps([{
            "directory": str(self.proof.parent), "file": str(self.source),
            "arguments": [self.args.compiler, "-c", str(self.source), "-o", "unused.o"],
        }]))
        self.args.compiler_id = "Clang"
        self.args.compiler_version = "1.0.0"
        with self.assertRaises(FileNotFoundError):
            CHECKER.build(self.args)
        self.assertFalse(self.proof.exists())
        self.assertFalse(self.assembly.exists())
        with self.assertRaises(FileNotFoundError):
            CHECKER.verify(self.args)

    def test_dotprod_requires_real_dot_and_rejects_matrix_extensions(self) -> None:
        """DotProd cannot become an unexercised or wider I8MM capability claim."""
        with self.assertRaises(ValueError):
            CHECKER.require_envelope("NEON_DOTPROD", CHECKER.instruction_counts("\tsmull.8h v0,v1,v2\n"))
        CHECKER.require_envelope("NEON_DOTPROD", CHECKER.instruction_counts("\tsdot v0.4s,v1.16b,v2.16b\n"))
        with self.assertRaises(ValueError):
            CHECKER.require_envelope("NEON_DOTPROD", CHECKER.instruction_counts("\tsdot v0,v1,v2\n\tsmmla v0,v1,v2\n"))

    def test_i8mm_requires_real_dot_and_signed_matrix_only(self) -> None:
        """I8MM proves both SDOT and SMMLA; mixed-sign/unsigned matrix forms stay rejected."""
        gnu = "\tsdot\tv0.4s, v1.16b, v2.16b\n\tsmmla\tv0.4s, v1.16b, v2.16b\n"
        apple = "\tsdot.4s\tv0, v1, v2\n\tsmmla\tv0.4s, v1.16b, v2.16b\n"
        for assembly in (gnu, apple):
            CHECKER.require_envelope("NEON_I8MM", CHECKER.instruction_counts(assembly))
        for missing in ("\tsdot v0.4s,v1.16b,v2.16b\n", "\tsmmla v0.4s,v1.16b,v2.16b\n"):
            with self.assertRaises(ValueError):
                CHECKER.require_envelope("NEON_I8MM", CHECKER.instruction_counts(missing))
        for foreign in ("usmmla", "ummla", "usdot", "sudot"):
            with self.assertRaises(ValueError):
                CHECKER.require_envelope("NEON_I8MM", CHECKER.instruction_counts(
                    gnu + f"\t{foreign} v0.4s,v1.16b,v2.16b\n"))
        with self.assertRaises(ValueError):
            CHECKER.require_envelope("NEON", CHECKER.instruction_counts("\tsmmla v0.4s,v1.16b,v2.16b\n"))
        with self.assertRaises(ValueError):
            CHECKER.require_envelope("NEON_SVE", CHECKER.instruction_counts(gnu))


if __name__ == "__main__":
    unittest.main()
