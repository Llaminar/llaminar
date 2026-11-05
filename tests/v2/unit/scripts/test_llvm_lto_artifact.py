#!/usr/bin/env python3
"""Device-free negative traces for exact compiler/dependency artifact ownership.

These typed fixtures certify validation/dispatch behavior only. Actual compiler
records and original-code controls are produced by the independent SDK build.
No test manufactures native readiness or launches an accelerator kernel.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
HELPERS = ROOT / "scripts/docker/llvm-lto-linker"

def load(name, path):
    """Import the actual builder/guard implementation into a device-free fixture."""
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value

artifact = load("lto_artifact_test_body", HELPERS / "artifact.py")
builder = load("lto_builder_test_body", HELPERS / "build.py")
guard = artifact.load_guard(ROOT / "scripts/build/gpu_spill_guard.py")

class OutputRoleTest(unittest.TestCase):
    """Original output identity distinguishes the collective from ordinary host targets."""
    def test_shared_host_fmt_is_not_a_collective(self):
        argv = ["hipcc", "-shared", "-o", "_deps/fmt-build/libfmt.so.10.2.1", "fmt.o"]
        self.assertIn("-shared", argv)  # Original all-shared selection negative control.
        self.assertEqual(artifact.classify_output(argv, "/build/librccl.so.1.0", "/build"), artifact.ArtifactRole.HOST)

    def test_exact_collective_relative_and_absolute_output(self):
        for output in ("librccl.so.1.0", "/build/librccl.so.1.0"):
            self.assertEqual(artifact.classify_output(["hipcc", "-shared", "-o", output],
                             "/build/librccl.so.1.0", "/build"), artifact.ArtifactRole.COLLECTIVE)

    def test_foreign_collective_missing_output_and_wrong_link_role_reject(self):
        for argv in (["hipcc", "-shared", "-o", "/foreign/librccl.so.1.0"],
                     ["hipcc", "-shared", "-o", "librccl.so.2"], ["hipcc", "-shared"],
                     ["hipcc", "-o", "librccl.so.1.0"],
                     ["hipcc", "-shared", "-o", "a", "-o", "b"]):
            with self.subTest(argv=argv), self.assertRaises(RuntimeError):
                artifact.classify_output(argv, "/build/librccl.so.1.0", "/build")

    def test_host_never_borrows_native_driver_admission_and_collective_cannot_omit_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("lld", "receipt"):
                (root / name).write_bytes(name.encode())
            spec = {"linker": str(root / "lld"), "linker_sha256": artifact.digest(root / "lld"),
                    "linker_receipt": str(root / "receipt"), "linker_receipt_sha256": artifact.digest(root / "receipt"),
                    "library": str(root / "librccl.so.1.0"), "architectures": ["gfx906", "gfx1100"]}
            with patch.object(artifact.subprocess, "run") as native:
                result = artifact.authenticate_driver(spec, ["hipcc", "-shared", "-o", "libfmt.so"], root)
                self.assertEqual(result["role"], artifact.ArtifactRole.HOST.value)
                native.assert_not_called()
            response = argparse.Namespace(stdout="", stderr="", returncode=0)
            with patch.object(artifact.subprocess, "run", return_value=response), self.assertRaisesRegex(RuntimeError, "native architecture"):
                artifact.authenticate_driver(spec, ["hipcc", "-shared", "-o", "librccl.so.1.0"], root)

    def test_collective_target_entry_cannot_execute_a_foreign_host_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("lld", "receipt"):
                (root / name).write_bytes(name.encode())
            spec = {"linker": str(root / "lld"), "linker_sha256": artifact.digest(root / "lld"),
                    "linker_receipt": str(root / "receipt"), "linker_receipt_sha256": artifact.digest(root / "receipt"),
                    "library": str(root / "librccl.so.1.0")}
            path = root / "spec.json"
            path.write_text(json.dumps(spec))
            argv = ["artifact.py", "gate", "--spec", str(path), "--", "hipcc", "-shared", "-o", str(root / "libother.so")]
            with patch.object(sys, "argv", argv), patch.object(artifact.os, "execv") as execute, \
                    self.assertRaisesRegex(RuntimeError, "non-collective output"):
                artifact.main()
            execute.assert_not_called()

    def test_stale_partition_cannot_fill_missing_current_record(self):
        """Old eight-file success cannot certify a later link producing only seven files."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {"record_root": str(root), "architectures": ["gfx906", "gfx1100"]}
            foreign = root / "final-gfx999.opt.ld.yaml.part.7.obj.yaml"
            foreign.write_text("foreign target")
            for arch in spec["architectures"]:
                for index in range(8):
                    for kind in ("obj", "asm"):
                        (root / f"final-{arch}.opt.ld.yaml.part.{index}.{kind}.yaml").write_text("old")
                self.assertEqual(len(artifact.object_records(spec, arch)), 8)  # Original stale-file acceptance.
            with artifact.record_transaction(spec):
                with self.assertRaisesRegex(RuntimeError, "Concurrent"):
                    with artifact.record_transaction(spec):
                        self.fail("Overlapping link/proof transaction admitted")
                artifact.withdraw_records(spec)
                self.assertEqual(foreign.read_text(), "foreign target")
                for arch in spec["architectures"]:
                    for index in range(7):
                        (root / f"final-{arch}.opt.ld.yaml.part.{index}.obj.yaml").write_text("current")
                    with self.assertRaisesRegex(RuntimeError, "record is missing"):
                        artifact.object_records(spec, arch)
                    self.assertFalse((root / f"final-{arch}.opt.ld.yaml.part.7.asm.yaml").exists())
                    (root / f"final-{arch}.opt.ld.yaml.part.7.obj.yaml").write_text("current")
                    self.assertEqual(len(artifact.object_records(spec, arch)), 8)


class CompilerProofTest(unittest.TestCase):
    """Reject incomplete partition records and changed compiler observations."""
    @staticmethod
    def report():
        rows = []
        for arch in ("gfx906", "gfx1100"):
            for count in (2, 8):
                kernels = [{"name": name, "sgpr": 0, "vgpr": 0, "private_bytes": 0, "dynamic_stack": False}
                           for name in sorted(builder.KERNELS)]
                for variant in ("original", "patched"):
                    repaired = variant == "patched"
                    rows.append({"architecture": arch, "partitions": count, "variant": variant,
                        "metadata_partitions": count, "kernels": 16, "code_sha256": "1" * 64,
                        "allocation_functions": 16 if repaired else 0,
                        "record_files": count if repaired else 0, "assembly_record_files": count if repaired else 0,
                        "kernels_metadata": copy.deepcopy(kernels),
                        "allocations": {name: {"instances": 1, "slots": []} for name in builder.KERNELS} if repaired else {},
                        "guard_accepted_spill_free_frames": repaired})
        return {"schema": 1, "certificate": True, "architectures": ["gfx906", "gfx1100"], "observations": rows}

    def test_complete_typed_partition_trace(self):
        builder.verify(self.report(), ["gfx906", "gfx1100"], guard=guard)

    def test_missing_duplicate_or_changed_partition_trace_rejects(self):
        for change in (lambda rows: rows.pop(), lambda rows: rows.append(copy.deepcopy(rows[0])),
                       lambda rows: rows[1].update(record_files=1),
                       lambda rows: rows[1].update(code_sha256="2" * 64),
                       lambda rows: rows[0].update(allocation_functions=16),
                       lambda rows: rows[1]["allocations"].pop("partition_probe_7"),
                       lambda rows: rows[1]["kernels_metadata"][0].update(vgpr="unknown")):
            report = self.report()
            change(report["observations"])
            with self.assertRaises(RuntimeError):
                builder.verify(report, ["gfx906", "gfx1100"], guard=guard)

class FinalResourceEvidenceTest(unittest.TestCase):
    """External dependency evidence reports physical resources without certifying separate gates."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.library = self.root / "library"
        self.receipt = self.root / "receipt"
        self.library.write_bytes(b"exact selected library identity fixture")
        self.receipt.write_bytes(b"exact selected build receipt fixture")
        self.args = argparse.Namespace(report=self.root / "proof.json", library=self.library,
            receipt=self.receipt, spill_guard=ROOT / "scripts/build/gpu_spill_guard.py", architectures="gfx906;gfx1100")

    def tearDown(self):
        self.temporary.cleanup()

    def report(self):
        rows = []
        for arch in ("gfx906", "gfx1100"):
            rows.append({"architecture": arch, "partitions": 8,
                "record_files": [{"sha256": "3" * 64, "bytes": 17} for _ in range(8)],
                "kernels": [{"name": "kernel", "sgpr": 1, "vgpr": 0, "private_bytes": 16, "dynamic_stack": False}],
                "allocations": {"kernel": {"instances": 1, "slots": [{"kind": "Variable", "size": 16}]}}})
        for row in rows:
            functions = {name: guard.FunctionAllocation([guard.StackSlot(**slot) for slot in value["slots"]],
                         value["instances"]) for name, value in row["allocations"].items()}
            row["resource_usage"] = artifact.resource_usage(functions, [guard.KernelSpills(**v) for v in row["kernels"]])
        canonical = artifact.pins(HELPERS.parent / "rocm-release.env")
        compiler = CompilerProofTest.report()
        compiler.update(toolchain={"linker_sha256": "4" * 64}, source_inputs={
            path.name: artifact.digest(path) for path in
            (HELPERS / "build.py", HELPERS / "main.cpp", HELPERS / "PressureProbe.hip", self.args.spill_guard)},
            llvm_revision=canonical["LLAMINAR_ROCM_LLVM_REVISION"],
            llvm_patch_sha256=artifact.digest(HELPERS.parent / "patches/llvm-parallel-lto-remarks.patch"),
            backend_source_sha256=canonical["LLAMINAR_ROCM_LLVM_LTO_BACKEND_SHA256"],
            driver_source_sha256=canonical["LLAMINAR_ROCM_LLVM_LLD_DRIVER_SHA256"])
        proof_json = json.dumps(compiler)
        compiler_receipt = {"schema": 1, "linker_sha256": "4" * 64, "architectures": ["gfx906", "gfx1100"],
                            "proof_sha256": hashlib.sha256(proof_json.encode()).hexdigest()}
        receipt_json = json.dumps(compiler_receipt)
        return {"schema": 2, "scope": "external-rccl-resource-evidence",
                "resource_evidence_complete": True, "zero_spill_qualified": False,
                "native_runtime_qualified": False, "performance_qualified": False,
                "producer_sha256": artifact.digest(HELPERS / "artifact.py"), "library_sha256": artifact.digest(self.library),
                "receipt_sha256": artifact.digest(self.receipt), "architectures": ["gfx906", "gfx1100"],
                "observations": rows, "linker_sha256": "4" * 64,
                "linker_receipt_sha256": hashlib.sha256(receipt_json.encode()).hexdigest(),
                "compiler_artifact": {"receipt_json": receipt_json, "proof_json": proof_json}}

    def verify(self, report):
        self.args.report.write_text(json.dumps(report))
        artifact.verify_proof(self.args)

    def test_intentional_private_frames_and_register_moves_remain_legal(self):
        self.verify(self.report())

    def test_external_spills_reported_while_project_kernel_guard_remains_fatal(self):
        report = self.report()
        for row in report["observations"]:
            row["allocations"]["kernel"]["slots"] = [{"kind": "Spill", "size": 4}]
            functions = {"kernel": guard.FunctionAllocation([guard.StackSlot("Spill", 4)], 1)}
            kernels = [guard.KernelSpills(**value) for value in row["kernels"]]
            row["resource_usage"] = artifact.resource_usage(functions, kernels)
            self.assertEqual(row["resource_usage"]["physical_spills"]["bytes"], 4)
            with self.assertRaises(guard.SpillGuardError):
                guard.audit_device_allocations(functions, kernels)
        self.verify(report)
        self.assertFalse(report["zero_spill_qualified"])
        self.assertFalse(report["native_runtime_qualified"])
        self.assertFalse(report["performance_qualified"])

    def test_missing_conflicting_or_misreported_resources_reject(self):
        changes = [lambda r: r.update(schema=1),
                   lambda r: r.update(zero_spill_qualified=True),
                   lambda r: r.update(native_runtime_qualified=True),
                   lambda r: r.update(performance_qualified=True),
                   lambda r: r["observations"][0].pop("resource_usage"),
                   lambda r: r["observations"][0]["resource_usage"]["physical_spills"].update(bytes=1),
                   lambda r: r.pop("producer_sha256"),
                   lambda r: r.update(producer_sha256="0" * 64),
                   lambda r: r.pop("compiler_artifact"),
                   lambda r: r["compiler_artifact"].update(proof_json="{}"),
                   lambda r: r.update(linker_sha256="0" * 64),
                   lambda r: r.update(library_sha256="0" * 64),
                   lambda r: r.update(receipt_sha256="0" * 64),
                   lambda r: r["observations"].pop(),
                   lambda r: r["observations"][0]["record_files"].pop(),
                   lambda r: r["observations"][0]["record_files"][0].pop("sha256"),
                   lambda r: r["observations"][0]["kernels"].append(copy.deepcopy(r["observations"][0]["kernels"][0])),
                   lambda r: r["observations"][0]["kernels"][0].pop("private_bytes"),
                   lambda r: r["observations"][0]["kernels"][0].update(vgpr="unknown"),
                   lambda r: r["observations"][0]["allocations"].clear(),
                   lambda r: r["observations"][0]["allocations"]["kernel"]["slots"][0].update(kind="Spill"),
                   lambda r: r["observations"][0]["allocations"]["kernel"]["slots"][0].update(kind="Unknown")]
        for change in changes:
            report = self.report()
            change(report)
            with self.subTest(change=change), self.assertRaises((RuntimeError, guard.SpillGuardError)):
                self.verify(report)

if __name__ == "__main__":
    unittest.main()
