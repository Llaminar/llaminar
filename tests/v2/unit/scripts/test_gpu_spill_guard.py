#!/usr/bin/env python3
"""Device-free tests of strict GPU spill evidence and compiler-output ownership."""

import importlib.util
import argparse
import copy
import io
from pathlib import Path
import shlex
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
PROOF_SPEC = importlib.util.spec_from_file_location(
    "gpu_spill_compilation", ROOT / "tests/v2/integration/build/test_gpu_spill_compilation.py")
proof = importlib.util.module_from_spec(PROOF_SPEC)
PROOF_SPEC.loader.exec_module(proof)
CACHE_PROOF_SPEC = importlib.util.spec_from_file_location(
    "hip_device_input_identity", ROOT / "tests/v2/integration/build/test_hip_device_input_identity.py")
cache_proof = importlib.util.module_from_spec(CACHE_PROOF_SPEC)
CACHE_PROOF_SPEC.loader.exec_module(cache_proof)


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


class EmptyKernelMetadataTest(unittest.TestCase):
    """Replay kernel-free targets without weakening kernel or helper spill checks."""

    # Preserve LLVM's actual projection from the gfx906 side of the wave32
    # WMMA TU. The two spaces before [] triggered the production rejection.
    EMPTY_GFX906_METADATA = (
        "---\namdhsa.kernels:  []\n"
        "amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx906\n"
        "amdhsa.version:\n  - 1\n  - 2\n...\n")

    @staticmethod
    def _inspection_results(targets):
        """Double native tools while retaining the production bundle audit."""
        results = [" [7] .hip_fatbin PROGBITS", "",
                   "\n".join(f"hip-amdgcn-amd-amdhsa--{target}" for target, _ in targets)
                   + "\nhost-x86_64-unknown-linux-gnu-\n"]
        for _, text in targets:
            results.extend(["", guard.json.dumps([{"NoteSections": [{"NoteSection": {
                "Notes": [{"AMDGPU Metadata": text}]}}]}])])
        return results

    def test_actual_llvm_empty_inventory_accepts_horizontal_spacing(self):
        """Canonical LLVM alignment and horizontal whitespace denote one empty list."""
        for spacing in (" ", "  ", "\t", " \t "):
            text = self.EMPTY_GFX906_METADATA.replace("kernels:  []", f"kernels:{spacing}[]")
            with self.subTest(spacing=repr(spacing)):
                self.assertEqual(guard.parse_metadata(text), [])

    def test_empty_inventory_rejects_repeated_or_malformed_headers(self):
        """An empty declaration cannot conceal another inventory or nested records."""
        empty = self.EMPTY_GFX906_METADATA
        cases = {
            "repeated_empty": empty.replace(
                "amdhsa.kernels:  []\n", "amdhsa.kernels: []\namdhsa.kernels:  []\n"),
            "empty_then_nonempty": empty + metadata(),
            "nonempty_then_empty": metadata() + empty,
            "empty_with_nested_kernel": metadata().replace("amdhsa.kernels:\n", "amdhsa.kernels: []\n"),
            "trailing_tokens": empty.replace("kernels:  []", "kernels: [] trailing"),
            "nonempty_inline_list": empty.replace("kernels:  []", "kernels: [kernel]"),
            "mapping_instead_of_list": empty.replace("kernels:  []", "kernels: {}"),
            "missing_inventory": empty.replace("amdhsa.kernels:  []\n", ""),
        }
        for name, text in cases.items():
            with self.subTest(case=name), self.assertRaises(guard.SpillGuardError):
                guard.parse_metadata(text)

    def test_empty_target_preserves_other_target_kernel_audit(self):
        """Both bundle orders retain the nonempty target and its final spill evidence."""
        for empty_first in (True, False):
            for spilling in (False, True):
                targets = [("gfx906", self.EMPTY_GFX906_METADATA),
                           ("gfx1100", metadata(name="wmma_kernel", vgpr="1" if spilling else "0"))]
                if not empty_first:
                    targets.reverse()
                with self.subTest(empty_first=empty_first, spilling=spilling), \
                        patch.object(guard, "run_tool", side_effect=self._inspection_results(targets)):
                    kernels = guard.audit_rocm_object(Path("mixed-wave.o"), Path("/llvm"))
                self.assertEqual(kernels, [guard.KernelSpills("wmma_kernel", 0, int(spilling))])
                diagnostics = AllocationTest.record(
                    "wmma_kernel", [("Spill", 4)] if spilling else ())
                functions, _ = guard.allocation_evidence(diagnostics)
                if spilling:
                    with self.assertRaisesRegex(guard.SpillGuardError, "wmma_kernel: 1 memory spill slots"):
                        guard.audit_device_allocations(functions, kernels)
                else:
                    self.assertEqual(guard.audit_device_allocations(functions, kernels), [])

    def test_empty_inventory_keeps_helper_audit_and_failed_object_retirement(self):
        """A kernel-free code object still rejects spilling helpers on fresh and cached runs."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "kernel-free.o"
            neighbor = Path(directory) / "unrelated.o"
            neighbor.write_bytes(b"unrelated object")
            args = ["--llvm-bin", "/llvm", "--config", "Release", "--policy-id", "empty-metadata", "--",
                    "ccache", "clang++", "-o", str(output)]
            # Repeating the clean observation exercises cached-output auditing;
            # the final helper spill must retire only its own linkable output.
            for spilling in (False, False, True):
                output.write_bytes(b"owned kernel-free object")
                diagnostics = (AllocationTest.record(
                    "__cxa_pure_virtual", [("Spill", 4)] if spilling else ())
                    + AllocationTest.record("__cxa_deleted_virtual"))
                stdout, stderr = io.StringIO(), io.StringIO()
                with self.subTest(spilling=spilling), \
                        patch.object(guard, "compile_hip", return_value=subprocess.CompletedProcess(
                            [], 0, diagnostics, "")), \
                        patch.object(guard, "run_tool", side_effect=self._inspection_results(
                            [("gfx906", self.EMPTY_GFX906_METADATA)])), \
                        patch.object(sys, "stdout", stdout), patch.object(sys, "stderr", stderr):
                    self.assertEqual(guard.main(args), int(spilling))
                self.assertEqual(output.exists(), not spilling)
                self.assertEqual(neighbor.read_bytes(), b"unrelated object")
                if spilling:
                    self.assertIn("__cxa_pure_virtual: 1 memory spill slots", stderr.getvalue())


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

    def test_final_link_stdout_routing_survives_ccache_classification(self):
        """The new driver must retain the real linker's stdout option on cache misses."""
        flags = ["-Xarch_device", "-foptimization-record-file=-", "-Xoffload-linker",
                 "-plugin-opt=opt-remarks-filename=-", "-c", "kernel.hip"]
        self.assertEqual(guard.preserve_device_diagnostics(["clang++", *flags]), ["clang++", *flags])
        self.assertEqual(guard.preserve_device_diagnostics(["ccache", "clang++", *flags]),
            ["ccache", "clang++", "--ccache-skip", "-Xarch_device", "--ccache-skip",
             "-foptimization-record-file=-", "--ccache-skip", "-Xoffload-linker",
             "--ccache-skip", "-plugin-opt=opt-remarks-filename=-", "-c", "kernel.hip"])
        # Preserve complete native option pairs without enumerating every
        # future linker/device operand in this diagnostic transport layer.
        other = ["-Xoffload-linker", "--lto-partitions=2", "-Xarch_device", "-O3"]
        self.assertEqual(guard.preserve_device_diagnostics(["ccache", "clang++", *other]),
            ["ccache", "clang++", "--ccache-skip", "-Xoffload-linker", "--ccache-skip",
             "--lto-partitions=2", "--ccache-skip", "-Xarch_device", "--ccache-skip", "-O3"])

    def test_final_link_stdout_is_required_for_every_target_and_helper_on_replay(self):
        """Replay actual guard ownership with typed child output, never a compiler certificate."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, sidecar = root / "cached.o", root / "-.opt.ld.yaml"
            good = (AllocationTest.record("kernel") + AllocationTest.record("helper")) * 2
            sidecar.write_text(good)
            args = ["--llvm-bin", "/llvm", "--config", "Release", "--policy-id", "routing", "--",
                    "ccache", "clang++", "-Xoffload-linker", "-plugin-opt=opt-remarks-filename=-",
                    "-o", str(output)]
            kernels = [guard.KernelSpills("kernel", 0, 0)] * 2
            cases = [(good, 0), (good, 0), ("", 1),
                     (AllocationTest.record("kernel") + AllocationTest.record("helper"), 1),
                     (good + AllocationTest.record("helper", [("Spill", 4)]), 1)]
            for text, expected in cases:
                output.write_bytes(b"owned final object fixture")
                stdout, stderr = io.StringIO(), io.StringIO()
                with self.subTest(expected=expected, stdout_records=len(text)), \
                        patch.object(guard, "compile_hip", return_value=subprocess.CompletedProcess([], 0, text, "")), \
                        patch.object(guard, "audit_rocm_object", return_value=kernels), \
                        patch.object(sys, "stdout", stdout), patch.object(sys, "stderr", stderr):
                    self.assertEqual(guard.main(args), expected)
                self.assertEqual(output.exists(), expected == 0)
                self.assertEqual(sidecar.read_text(), good)  # Never adopt/delete shared sidecar evidence.
                if expected:
                    self.assertIn("memory register spills are forbidden", stderr.getvalue())

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
            with patch.object(guard, "compile_hip", return_value=subprocess.CompletedProcess([], 0)), \
                 patch.object(guard, "audit_rocm_object", side_effect=guard.SpillGuardError("VGPR spills=14")) as audit:
                self.assertEqual(guard.main(args), 1)
                audit.assert_called_once_with(output, Path("/llvm"))
            self.assertFalse(output.exists())
            self.assertEqual(neighbor.read_bytes(), b"unrelated")

    def test_debug_does_not_apply_optimized_resource_gate(self):
        args = ["--llvm-bin", "/llvm", "--config", "Debug", "--policy-id", "v1", "--",
                "clang++", "-o", "debug.o"]
        with patch.object(guard, "compile_hip", return_value=subprocess.CompletedProcess([], 0)), \
             patch.object(guard, "audit_rocm_object") as audit:
            self.assertEqual(guard.main(args), 0)
            audit.assert_not_called()

    def test_compiler_failure_is_preserved(self):
        args = ["--llvm-bin", "/llvm", "--config", "Integration", "--policy-id", "v1", "--",
                "clang++", "-o", "bad.o"]
        with patch.object(guard, "compile_hip", return_value=subprocess.CompletedProcess([], 9)), \
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

    @staticmethod
    def partition_notes(parts):
        """Preserve actual LLVM LTO note multiplicity rather than merging its records."""
        return guard.json.dumps([{"NoteSections": [{"NoteSection": {
            "Notes": [{"AMDGPU Metadata": part} for part in parts]}}]}])

    def audit_partition_notes(self, parts):
        """Replay actual final-object inspection with only the native inspection tools doubled."""
        results = [" [7] .hip_fatbin PROGBITS", "",
                   "hipv4-amdgcn-amd-amdhsa--gfx906\n", "", self.partition_notes(parts)]
        with patch.object(guard, "run_tool", side_effect=results):
            return guard.audit_rocm_object(Path("partitioned.o"), Path("/llvm"))

    def test_all_eight_lto_partition_notes_preserve_final_spilling_kernel(self):
        parts = [metadata(name=f"partition{i}", vgpr="1" if i == 7 else "0") for i in range(8)]
        kernels = self.audit_partition_notes(parts)
        self.assertEqual(len(kernels), 8)
        self.assertEqual(kernels[-1], guard.KernelSpills("partition7", 0, 1))
        functions = {kernel.name: guard.FunctionAllocation() for kernel in kernels}
        functions["partition7"] = guard.FunctionAllocation([guard.StackSlot("Spill", 4)])
        with self.assertRaisesRegex(guard.SpillGuardError, "partition7: 1 memory spill slots"):
            guard.audit_device_allocations(functions, kernels)

    def test_repeated_kernel_across_lto_notes_is_fatal_even_when_records_agree(self):
        for last in [metadata(), metadata(vgpr="9")]:
            with self.subTest(last=last), self.assertRaisesRegex(guard.SpillGuardError, "duplicate final kernel"):
                self.audit_partition_notes([metadata(), last])

    def test_missing_or_malformed_final_partition_metadata_is_fatal(self):
        for parts in [[], [metadata(name="first"), None], [metadata(name="first"), "not metadata"],
                      [metadata(name="first"), metadata(name="last").replace(".vgpr_spill_count: 0", ".vgpr_spill_count: invalid")]]:
            with self.subTest(parts=parts), self.assertRaises(guard.SpillGuardError):
                self.audit_partition_notes(parts)

    def test_host_only_object(self):
        with patch.object(guard, "run_tool", return_value=" [1] .text PROGBITS"):
            self.assertEqual(guard.audit_rocm_object(Path("host.o"), Path("/llvm")), [])

    def test_relocatable_ir_cannot_masquerade_as_final_resource_evidence(self):
        with patch.object(guard, "run_tool", return_value=" [1] __CLANG_OFFLOAD_BUNDLE__hip PROGBITS"):
            with self.assertRaisesRegex(guard.SpillGuardError, "final device-link"):
                guard.audit_rocm_object(Path("ir.o"), Path("/llvm"))

class InspectionDiagnosticTest(unittest.TestCase):
    """A native inspection replay supplies evidence without changing acceptance."""

    def test_exact_compile_command_and_private_working_directory_are_preserved(self):
        command = [sys.executable, str(ROOT / "scripts/build/gpu_spill_guard.py"),
                   "--llvm-bin", "/sdk/llvm", "--config", "Release", "--policy-id", "v1", "--",
                   "ccache", "clang++", "--offload-arch=gfx906", "--offload-arch=gfx1100",
                   "-o", "object with spaces.o", "-c", "source with spaces.hip"]
        args = argparse.Namespace(ninja="/active/ninja")
        with patch.object(proof.subprocess, "check_output", return_value=shlex.join(command)), \
                patch.object(proof.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 1, "", "HIP bundle inspection: targets=''")) as replay:
            diagnostic = proof.failed_hip_inspection(args, Path("/private/build"), "clean")
        self.assertIn("exit=1", diagnostic)
        self.assertIn("targets=''", diagnostic)
        actual = replay.call_args.args[0]
        self.assertEqual(actual[:2], [sys.executable, "-c"])
        self.assertEqual(actual[3:], command[2:])
        self.assertEqual(replay.call_args.kwargs["cwd"], Path("/private/build"))

    def test_ambiguous_or_indirect_launcher_is_rejected(self):
        command = shlex.join([sys.executable, str(ROOT / "scripts/build/gpu_spill_guard.py"), "--"])
        for commands in ("", command + "\n" + command, "env " + command):
            with self.subTest(commands=commands), \
                    patch.object(proof.subprocess, "check_output", return_value=commands), \
                    patch.object(proof.subprocess, "run") as replay, \
                    self.assertRaisesRegex(RuntimeError, "one direct production-guard"):
                proof.failed_hip_inspection(argparse.Namespace(ninja="ninja"), Path("/build"), "clean")
            replay.assert_not_called()

    def test_successful_diagnostic_cannot_replace_failed_compilation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "Release"
            build.mkdir()
            report = root / "proof.json"
            args = argparse.Namespace(report=report, language="HIP", compiler="clang++",
                                      architectures="gfx906;gfx1100", cmake="cmake", ninja="ninja")
            with patch.object(proof.tempfile, "TemporaryDirectory") as temporary, \
                    patch.object(proof.subprocess, "check_output", return_value="clang version"), \
                    patch.object(proof.subprocess, "run", return_value=subprocess.CompletedProcess(
                        [], 1, "original compiler failure", "")), \
                    patch.object(proof, "cases", return_value=[("Release", "clean", "initial")]), \
                    patch.object(proof, "failed_hip_inspection", return_value="diagnostic exit=0") as replay:
                temporary.return_value.__enter__.return_value = directory
                with self.assertRaisesRegex(RuntimeError, "original compiler failure"):
                    proof.compile_report(args)
                replay.assert_called_once_with(args, build, "clean")
            self.assertFalse(report.exists())


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


class DeviceInputIdentityTest(unittest.TestCase):
    """Device-only includes own both cache identity and native rebuild edges."""

    def test_each_target_changes_cache_identity_and_preserves_dependency_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, depfile = root / "object.o", root / "object.d"
            command = ["ccache", "clang++", "--offload-arch=gfx906", "--offload-arch=gfx1100",
                       "--hip-path=/selected/sdk", "-MD", "-MF", str(depfile),
                       "-o", str(output), "-x", "hip", "-c", "kernel.hip"]
            contents = {"gfx906": "old device body", "gfx1100": "other device body"}
            identities = []

            def run(argv, **kwargs):
                if "--offload-device-only" in argv:
                    target = next(arg.partition("=")[2] for arg in argv if arg.startswith("--offload-arch="))
                    self.assertIn("--hip-path=/selected/sdk", argv)
                    self.assertNotIn("-MD", argv)
                    self.assertIn("-sys-header-deps", argv)
                    Path(argv[argv.index("-o") + 1]).write_text(contents[target])
                    Path(argv[argv.index("-dependency-file") + 2]).write_text(
                        f"hip-device-input: device\\ {target}.inc\n")
                elif "--get-config" in argv:
                    return subprocess.CompletedProcess(argv, 0, "/user/cache-input\n", "")
                else:
                    extras = kwargs["env"]["CCACHE_EXTRAFILES"].split(guard.os.pathsep)
                    self.assertEqual(extras[0], "/user/cache-input")
                    identities.append(Path(extras[-1]).read_text())
                    depfile.write_text("original\\ object.o: host\\ header.h\nhost\\ header.h:\n")
                    output.write_bytes(b"compiler output")
                return subprocess.CompletedProcess(argv, 0, "", "")

            with patch.object(guard.subprocess, "run", side_effect=run):
                guard.compile_hip(command, output)
                guard.compile_hip(command, output)
                contents["gfx1100"] = "changed device-only body"
                guard.compile_hip(command, output)
            self.assertEqual(identities[0], identities[1])
            self.assertNotEqual(identities[1], identities[2])
            self.assertEqual(depfile.read_text(),
                "original\\ object.o: host\\ header.h device\\ gfx906.inc device\\ gfx1100.inc\nhost\\ header.h:\n")

    def test_incomplete_preprocessing_retires_old_object_before_cache_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "old.o"
            output.write_bytes(b"stale output")
            args = ["--llvm-bin", "/llvm", "--config", "Release", "--policy-id", "fixture", "--",
                    "ccache", "clang++", "--offload-arch=gfx1100", "-MD", "-MF", str(output)+".d",
                    "-o", str(output), "-c", "kernel.hip"]
            with patch.object(guard.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 1, "", "device header is missing")) as compile_call, \
                    patch.object(guard, "audit_rocm_object") as audit:
                self.assertEqual(guard.main(args), 1)
            compile_call.assert_called_once()
            audit.assert_not_called()
            self.assertFalse(output.exists())

    def test_make_rules_keep_escaped_colons_continuations_and_phony_targets(self):
        host = "object\\:name.o: first\\ file.h \\\n second.h\nfirst\\ file.h:\n"
        self.assertEqual(guard.merged_device_dependencies(host, ["device: dollar$$name.h\n"]),
                         "object\\:name.o: first\\ file.h  second.h dollar$$name.h\nfirst\\ file.h:\n")
        for invalid in ("", "target:", ": header.h", "no target rule"):
            with self.subTest(invalid=invalid), self.assertRaises(guard.SpillGuardError):
                guard.merged_device_dependencies(host, [invalid])

    def test_installed_proof_rejects_stale_missing_and_wrong_native_results_without_tools(self):
        """Synthetic receipts exercise admission only; native probes own qualification."""
        report = {"schema": 1, "compiler": "/absent/compiler", "compiler_version": "fixture",
                  "cache_version": "fixture", "architectures": "gfx906;gfx1100", "rocm_path": "/absent/sdk",
                  "sources": cache_proof.source_identity(), "observations": []}
        for stage, words in cache_proof.STAGES:
            report["observations"].append({"stage": stage, "words": words,
                "returncode": int(words is None), "object_present": words is not None,
                "diagnostic": "device input preprocessing failed" if words is None else "",
                "lds_bytes": {target: words * 4 for target in ("gfx906", "gfx1100")} if words else {},
                "device_header_dependency": True,
                "cache_results": ["cache_miss" if stage in ("cold512", "changed256") else "direct_cache_hit"]})
        verify = lambda value: cache_proof.verify_report(
            value, "/absent/compiler", "gfx906;gfx1100", "/absent/sdk")
        with patch.object(cache_proof.subprocess, "run", side_effect=AssertionError("must not compile")), \
                patch.object(cache_proof.subprocess, "check_output", side_effect=AssertionError("must not inspect")):
            verify(report)
            for mutation in ("sources", "missing_stage", "stale_lds", "missing_dependency", "retained_failure", "no_hit"):
                bad = copy.deepcopy(report)
                if mutation == "sources": bad["sources"] = {}
                elif mutation == "missing_stage": bad["observations"].pop()
                elif mutation == "stale_lds": bad["observations"][2]["lds_bytes"]["gfx1100"] = 2048
                elif mutation == "missing_dependency": bad["observations"][2]["device_header_dependency"] = False
                elif mutation == "retained_failure": bad["observations"][5]["object_present"] = True
                else: bad["observations"][1]["cache_results"] = ["cache_miss"]
                with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                    verify(bad)


class CompilerProofTest(unittest.TestCase):
    """Installed preflight verifies compiler evidence without needing SDKs."""

    @staticmethod
    def report(language="HIP"):
        """Synthetic evidence tests admission only, never certifies a compiler."""
        observations = []
        for config, variant, attempt in proof.cases():
            rejected = config != "Debug" and variant in ("spill", "helper")
            observations.append({
                "config": config, "variant": variant, "attempt": attempt,
                "returncode": 1 if rejected else 0,
                "objects_present": not rejected,
                "diagnostic": ("register spills are forbidden; Registers are spilled" if rejected
                               else "scalar-to-vector register moves; no memory spill"),
            })
        return {"schema": 1, "language": language, "compiler": "/absent/compiler",
                "architectures": "a;b", "compiler_version": "fixture compiler",
                "sources": proof.source_identity(), "observations": observations}

    def test_installed_verification_never_invokes_missing_compilers(self):
        for language in ("CUDA", "HIP"):
            with self.subTest(language=language), patch.object(
                    proof.subprocess, "run", side_effect=AssertionError("must not compile")), \
                    patch.object(proof.subprocess, "check_output",
                                 side_effect=AssertionError("must not probe compiler")):
                proof.verify_report(self.report(language), language, "/absent/compiler", "a;b")

    def test_missing_stale_foreign_and_partial_evidence_fail(self):
        original = self.report()
        changes = [(key, value) for key, value in (
            ("schema", 0), ("language", "CUDA"), ("compiler", "/other/compiler"),
            ("architectures", "a"), ("compiler_version", ""), ("sources", {}),
            ("observations", original["observations"][:-1]),
            ("observations", original["observations"] + original["observations"][:1]))]
        for key, value in changes:
            report = {**original, key: value}
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                proof.verify_report(report, "HIP", "/absent/compiler", "a;b")

    def test_wrong_rejection_or_retained_object_cannot_pass(self):
        original = self.report()
        failed_index = next(i for i, row in enumerate(original["observations"])
                            if row["returncode"] != 0)
        for index, key, value in ((0, "returncode", 1), (0, "objects_present", False),
                                  (failed_index, "diagnostic", "compiler missing"),
                                  (failed_index, "objects_present", True),
                                  (failed_index, "returncode", 0)):
            report = copy.deepcopy(original)
            report["observations"][index][key] = value
            with self.subTest(index=index, key=key), self.assertRaises(RuntimeError):
                proof.verify_report(report, "HIP", "/absent/compiler", "a;b")

    def test_rebuild_failure_retires_old_successful_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proof.json"
            path.write_text("old successful proof")
            args = argparse.Namespace(report=path, language="HIP", compiler="missing",
                                      architectures="gfx906")
            with patch.object(proof.subprocess, "check_output", side_effect=FileNotFoundError), \
                    self.assertRaises(FileNotFoundError):
                proof.compile_report(args)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
