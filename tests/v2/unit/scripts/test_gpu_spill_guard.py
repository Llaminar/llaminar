#!/usr/bin/env python3
"""Device-free tests of strict GPU spill evidence and compiler-output ownership."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location("gpu_spill_guard", ROOT / "scripts/build/gpu_spill_guard.py")
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


def metadata(name="kernel", sgpr="0", vgpr="0", private="0", dynamic="false"):
    """Canonical LLVM YAML including nested argument fields that are not kernels."""
    return ("---\namdhsa.kernels:\n  - .args:\n      - .name: ignored_argument\n"
            f"    .name: {name}\n    .private_segment_fixed_size: {private}\n"
            f"    .sgpr_spill_count: {sgpr}\n    .vgpr_spill_count: {vgpr}\n"
            f"    .uses_dynamic_stack: {dynamic}\n"
            "amdhsa.version:\n  - 1\n  - 2\n...\n")


class MetadataTest(unittest.TestCase):
    """A nonzero private allocation is not evidence of register spilling."""

    def test_clean_and_legitimate_private_memory(self):
        for private in ("0", "8", "4096"):
            self.assertEqual(guard.parse_metadata(metadata(private=private)),
                             [guard.KernelSpills("kernel", 0, 0, int(private))])

    def test_both_spill_banks_are_preserved(self):
        self.assertEqual(guard.parse_metadata(metadata(sgpr="2", vgpr="14")),
                         [guard.KernelSpills("kernel", 2, 14)])

    def test_multiple_specializations(self):
        text = metadata().split("amdhsa.version:")[0]
        text += metadata(name="second").split("amdhsa.kernels:\n")[1]
        self.assertEqual(len(guard.parse_metadata(text)), 2)

    def test_quoted_name(self):
        self.assertEqual(guard.parse_metadata(metadata(name="'_Z6kernelv'"))[0].name, "_Z6kernelv")

    def test_empty_kernel_inventory(self):
        self.assertEqual(guard.parse_metadata("---\namdhsa.kernels: []\n...\n"), [])

    def test_missing_invalid_and_duplicate_fields_fail(self):
        samples = ["", "amdhsa.kernels:\n", metadata().replace("    .vgpr_spill_count: 0\n", ""),
                   metadata(vgpr="-1"), metadata(sgpr="unknown"), metadata(dynamic="unknown"),
                   metadata().replace("    .name: kernel", "    .name: kernel\n    .name: again")]
        for sample in samples:
            with self.subTest(sample=sample), self.assertRaises(guard.SpillGuardError):
                guard.parse_metadata(sample)

    def test_duplicate_kernel_names_fail(self):
        text = metadata().split("amdhsa.version:")[0]
        text += metadata().split("amdhsa.kernels:\n")[1]
        with self.assertRaises(guard.SpillGuardError):
            guard.parse_metadata(text)


class LauncherTest(unittest.TestCase):
    """Compilation, cached output, and failure retirement share one authority."""

    def test_device_remarks_survive_ccache_preprocessor_classification(self):
        flags = ["-Xarch_device", "-foptimization-record-file=-", "-c", "kernel.hip"]
        self.assertEqual(guard.preserve_device_diagnostics(["clang++", *flags]),
                         ["clang++", *flags])
        self.assertEqual(
            guard.preserve_device_diagnostics(["/usr/bin/ccache", "clang++", *flags]),
            ["/usr/bin/ccache", "clang++", "--ccache-skip", "-Xarch_device",
             "--ccache-skip", "-foptimization-record-file=-", "-c", "kernel.hip"])

    def test_output_and_response_file(self):
        with tempfile.TemporaryDirectory() as directory:
            response = Path(directory) / "compile.rsp"
            response.write_text('-c "source with spaces.hip" -o "object with spaces.o"')
            self.assertEqual(guard.output_path(["ccache", "clang++", f"@{response}"]),
                             Path("object with spaces.o"))
        self.assertEqual(guard.output_path(["clang++", "--output-file=out.o"]), Path("out.o"))

    def test_missing_or_ambiguous_output_fails(self):
        for command in (["clang++"], ["clang++", "-o", "-"], ["clang++", "-o", "a", "-o", "b"]):
            with self.subTest(command=command), self.assertRaises(guard.SpillGuardError):
                guard.output_path(command)

    def test_cached_object_is_checked_and_failure_removes_only_that_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cached.o"
            neighbor = Path(directory) / "untouched.o"
            output.write_bytes(b"cached object")
            neighbor.write_bytes(b"unrelated")
            args = ["--llvm-bin", "/llvm", "--config", "Release", "--policy-id", "v1", "--",
                    "ccache", "clang++", "-o", str(output)]
            with patch.object(guard.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
                 patch.object(guard, "audit_rocm_object", side_effect=guard.SpillGuardError("VGPR spills=14")) as audit:
                self.assertEqual(guard.main(args), 1)
                audit.assert_called_once_with(output, Path("/llvm"))
            self.assertFalse(output.exists())
            self.assertEqual(neighbor.read_bytes(), b"unrelated")

    def test_debug_does_not_apply_optimized_resource_gate(self):
        args = ["--llvm-bin", "/llvm", "--config", "Debug", "--policy-id", "v1", "--",
                "clang++", "-o", "debug.o"]
        with patch.object(guard.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
             patch.object(guard, "audit_rocm_object") as audit:
            self.assertEqual(guard.main(args), 0)
            audit.assert_not_called()

    def test_compiler_failure_is_preserved(self):
        args = ["--llvm-bin", "/llvm", "--config", "Integration", "--policy-id", "v1", "--",
                "clang++", "-o", "bad.o"]
        with patch.object(guard.subprocess, "run", return_value=subprocess.CompletedProcess([], 9)), \
             patch.object(guard, "audit_rocm_object") as audit:
            self.assertEqual(guard.main(args), 9)
            audit.assert_not_called()

    def test_every_bundled_architecture_is_audited(self):
        notes = lambda text: guard.json.dumps([{"NoteSections": [{"NoteSection": {
            "Notes": [{"AMDGPU Metadata": text}]}}]}])
        results = [" [7] .hip_fatbin PROGBITS", "",
                   "hipv4-amdgcn-amd-amdhsa--gfx906\nhipv4-amdgcn-amd-amdhsa--gfx942\nhost-x86_64-unknown-linux-gnu-\n",
                   "", notes(metadata()), "", notes(metadata(name="other_arch", vgpr="1"))]
        with patch.object(guard, "run_tool", side_effect=results):
            kernels = guard.audit_rocm_object(Path("multi.o"), Path("/llvm"))
        self.assertEqual(kernels, [guard.KernelSpills("kernel", 0, 0), guard.KernelSpills("other_arch", 0, 1)])
        with self.assertRaisesRegex(guard.SpillGuardError, "other_arch"):
            guard.audit_device_allocations({"kernel": guard.FunctionAllocation(),
                "other_arch": guard.FunctionAllocation([guard.StackSlot("Spill", 4)])}, kernels)

    def test_host_only_object(self):
        with patch.object(guard, "run_tool", return_value=" [1] .text PROGBITS"):
            self.assertEqual(guard.audit_rocm_object(Path("host.o"), Path("/llvm")), [])

    def test_relocatable_ir_cannot_masquerade_as_final_resource_evidence(self):
        with patch.object(guard, "run_tool", return_value=" [1] __CLANG_OFFLOAD_BUNDLE__hip PROGBITS"):
            with self.assertRaisesRegex(guard.SpillGuardError, "final device-link"):
                guard.audit_rocm_object(Path("ir.o"), Path("/llvm"))

class AllocationTest(unittest.TestCase):
    """Final frame slot types distinguish memory spills from register moves."""

    @staticmethod
    def record(function="helper", slots=()):
        return (f"--- !Analysis\nPass: stack-frame-layout\nName: StackLayout\nFunction: {function}\nArgs:\n"
                + "".join(f"  - Type: {kind}\n  - Size: '{size}'\n" for kind, size in slots)
                + "...\n")

    def test_scalar_moves_have_no_memory_and_are_reported(self):
        reports = guard.audit_device_allocations(
            {"kernel": guard.FunctionAllocation()}, [guard.KernelSpills("kernel", 37, 0)])
        self.assertEqual(len(reports), 1)
        self.assertIn("37 scalar-to-vector register moves", reports[0])

    def test_memory_spills_in_helper_cannot_hide_behind_clean_kernel(self):
        for frame in (4, 128):
            with self.subTest(frame=frame), self.assertRaisesRegex(guard.SpillGuardError, "helper"):
                guard.audit_device_allocations({"kernel": guard.FunctionAllocation(),
                                               "helper": guard.FunctionAllocation([guard.StackSlot("Spill", frame)])},
                                              [guard.KernelSpills("kernel", 0, 0, 128)])

    def test_private_arrays_and_register_only_helpers_are_legal(self):
        self.assertEqual(guard.audit_device_allocations(
            {"kernel": guard.FunctionAllocation([guard.StackSlot("Variable", 1024)]),
             "helper": guard.FunctionAllocation()},
            [guard.KernelSpills("kernel", 0, 0, 1024)]), [])

    def test_scalar_moves_alongside_explicit_private_storage_are_legal(self):
        reports = guard.audit_device_allocations(
            {"kernel": guard.FunctionAllocation([guard.StackSlot("Variable", 1024),
                                                 guard.StackSlot("Fixed", 16)])},
            [guard.KernelSpills("kernel", 9, 0, 1040)])
        self.assertIn("9 scalar-to-vector", reports[0])

    def test_missing_kernel_frame_evidence_fails(self):
        with self.assertRaisesRegex(guard.SpillGuardError, "missing per-function"):
            guard.audit_device_allocations({}, [guard.KernelSpills("kernel", 0, 0)])

    def test_one_architecture_cannot_hide_another_architectures_missing_evidence(self):
        with self.assertRaisesRegex(guard.SpillGuardError, "for 2 targets"):
            guard.audit_device_allocations({"kernel": guard.FunctionAllocation()},
                [guard.KernelSpills("kernel", 0, 0), guard.KernelSpills("kernel", 0, 0)])

    def test_vector_to_accumulator_register_moves_are_not_memory_spills(self):
        reports = guard.audit_device_allocations({"kernel": guard.FunctionAllocation()},
                                                [guard.KernelSpills("kernel", 0, 7)])
        self.assertIn("7 vector register moves; no memory spill", reports[0])

    def test_records_are_cached_and_other_diagnostics_survive(self):
        text = "warning: keep this\n" + self.record(slots=[("Variable", 128)])
        evidence, other = guard.allocation_evidence(text)
        self.assertEqual(evidence["helper"], guard.FunctionAllocation([guard.StackSlot("Variable", 128)]))
        self.assertEqual(other, "warning: keep this\n")
        # Another target with real frame storage cannot be hidden by the clean one.
        evidence, _ = guard.allocation_evidence(text + self.record(slots=[("Spill", 32)]))
        with self.assertRaises(guard.SpillGuardError):
            guard.audit_device_allocations(evidence, [])

    def test_truncated_unknown_and_incomplete_records_fail(self):
        for text in ("--- !Analysis\nPass: stack-frame-layout\n",
                     self.record(slots=[("Unknown", 2)]),
                     self.record(slots=[("Spill", 2)]).replace("  - Size: '2'\n", ""),
                     self.record(slots=[("Spill", -1)])):
            with self.subTest(text=text), self.assertRaises(guard.SpillGuardError):
                guard.allocation_evidence(text)


if __name__ == "__main__":
    unittest.main()
