#!/usr/bin/env python3
"""Device-free adversarial validation of both native shared-file lifecycle proofs.

Fake library bytes test only the verifier interface. Actual whole-TU native and
literal-original execution belongs to the builder and cannot be inferred here.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "native_shared_file_proof", ROOT / "scripts/docker/shared-file-lifecycle/proof.py")
PROOF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROOF)


class NativeSharedFileProofTests(unittest.TestCase):
    """Mutate exactly one required claim while retaining complete baseline observations."""

    def setUp(self):
        """Construct verifier-only fake reports; no SDK/compiler/runtime is queried."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.library = self.root / "native.so"
        self.library.write_bytes(b"fake verifier-only library identity")

    def report(self, backend):
        """Create an in-memory format-valid fixture, never an installed certificate."""
        compiler = "/nonexistent/compiler"
        source = "/nonexistent/src/misc/shmutils.cc"
        closure = {compiler: "1" * 64, source: "2" * 64, "/nonexistent/include/comm.h": "3" * 64}
        if backend == "nccl":
            receipt = PROOF.load_module(PROOF.DOCKER / "nccl-gin-selection/proof.py").receipt_prefix() + "toolkit-default"
        else:
            receipt = "pin-shm-" + PROOF.digest(PROOF.patch(backend)) + "-selector-identity" + PROOF.rccl_helper().receipt_suffix()
        return dict(schema=1, certificate=True, backend=backend, scope=PROOF.SCOPE,
            native_runtime_qualified=False, sources=PROOF.source_identity(backend),
            library_sha256=PROOF.digest(self.library), build_receipt=receipt,
            compiler=dict(path=compiler, version="fixture compiler", sha256="1" * 64,
                          tools={compiler: "1" * 64}),
            sdk="/nonexistent/sdk", architecture="gfx906" if backend == "rccl" else "toolkit-default",
            compile_database_sha256="4" * 64, native_source=source,
            native_source_sha256="2" * 64, original_source_sha256="5" * 64,
            native_compile_commands=[[compiler, "-O3", "-c", source]] * 2,
            probe_compile_commands=[[compiler, "-MD", "-Wl,--wrap=malloc", "probe.cpp"]] * 2,
            dependency_closure=closure, dependency_closure_sha256=PROOF.closure_digest(closure),
            observations=[dict(variant=variant, compile_exit=0, returncode=0, stderr="",
                               stdout=json.dumps(PROOF.expected(variant)) + "\n")
                          for variant in ("candidate", "original")])

    def reject(self, mutation):
        """Require a specific malformed record to reject for both backend families."""
        for backend in ("nccl", "rccl"):
            with self.subTest(backend=backend):
                report = self.report(backend)
                mutation(report)
                with self.assertRaises(RuntimeError):
                    PROOF.verify_report(report, self.library, backend)

    def test_passive_verifier_needs_no_native_paths_or_subprocess(self):
        """Only policy/report/DSO bytes are read; nonexistent recorded SDK paths are valid."""
        with mock.patch.object(PROOF.subprocess, "run", side_effect=AssertionError("compiler/runtime invoked")):
            for backend in ("nccl", "rccl"):
                PROOF.verify_report(self.report(backend), self.library, backend)

    def test_relocated_same_library_bytes_pass(self):
        """Relocation changes no library identity or source authority."""
        moved = self.root / "moved.so"
        moved.write_bytes(self.library.read_bytes())
        for backend in ("nccl", "rccl"):
            PROOF.verify_report(self.report(backend), moved, backend)

    def test_wrong_library_rejects(self):
        """A same-filename substitution cannot retain certification."""
        reports = {backend: self.report(backend) for backend in ("nccl", "rccl")}
        self.library.write_bytes(b"different bytes")
        for backend, report in reports.items():
            with self.assertRaises(RuntimeError):
                PROOF.verify_report(report, self.library, backend)

    def test_source_only_and_schema_bool_reject(self):
        """Development source replay cannot be mistaken for an installed certificate."""
        self.reject(lambda r: r.update(certificate=False))
        self.reject(lambda r: r.update(schema=True))

    def test_missing_or_stale_source_patch_reject(self):
        """Every new patch/probe/helper remains part of one canonical policy identity."""
        self.reject(lambda r: r["sources"].pop(next(iter(r["sources"]))))
        self.reject(lambda r: r["sources"].update({next(iter(r["sources"])): "0" * 64}))
        self.reject(lambda r: r.update(build_receipt="old-unpatched-native-build"))

    def test_foreign_receipt_rejects(self):
        """Even a format-valid proof must agree with the selected installed receipt."""
        for backend in ("nccl", "rccl"):
            with self.assertRaises(RuntimeError):
                PROOF.verify_report(self.report(backend), self.library, backend, "foreign")

    def test_missing_compiler_header_or_command_rejects(self):
        """Do not accept a library hash without its actual native compilation closure."""
        self.reject(lambda r: r.update(compiler={}))
        self.reject(lambda r: r.update(dependency_closure={}))
        self.reject(lambda r: r.update(dependency_closure_sha256="0" * 64))
        self.reject(lambda r: r.update(native_compile_commands=[]))
        self.reject(lambda r: r["compiler"].update(tools={}))

    def test_wrong_architecture_or_sdk_rejects(self):
        """Architecture and selected SDK are required immutable builder facts."""
        self.reject(lambda r: r.update(architecture="wrong"))
        self.reject(lambda r: r.update(sdk="relative"))

    def test_identical_original_rejects(self):
        """The negative must be the distinct literal original implementation."""
        self.reject(lambda r: r.update(original_source_sha256=r["native_source_sha256"]))
        self.reject(lambda r: r.update(original_source_sha256=int("1" * 64)))

    def test_unrelated_crash_compile_error_or_diagnostic_rejects(self):
        """Nonzero exit or unrelated stderr cannot act as a defect witness."""
        for key, value in (("returncode", 87), ("returncode", -11), ("compile_exit", 1),
                           ("stderr", "unrelated failure"), ("returncode", False)):
            self.reject(lambda r, k=key, v=value: r["observations"][1].update({k: v}))

    def test_missing_duplicate_or_reordered_control_rejects(self):
        """Both complete expected polarities are mandatory exactly once."""
        self.reject(lambda r: r["observations"].pop())
        self.reject(lambda r: r["observations"].reverse())
        self.reject(lambda r: r["observations"][1].update(variant="candidate"))

    def test_every_case_and_last_replay_required(self):
        """Counts and booleans authenticate all original/candidate observations strictly."""
        for key, value in PROOF.expected("original").items():
            with self.subTest(field=key):
                def change(report):
                    measured = json.loads(report["observations"][1]["stdout"])
                    measured[key] = False if type(value) is bool else value - 1
                    report["observations"][1]["stdout"] = json.dumps(measured)
                self.reject(change)
        self.reject(lambda r: r["observations"][0].update(stdout="{}"))

    def test_bool_counts_and_unparsed_extra_output_reject(self):
        """Python bool/int equality must not permit malformed native measurements."""
        def change(report):
            measured = PROOF.expected("original")
            measured["schema"] = True
            report["observations"][1]["stdout"] = json.dumps(measured)
        self.reject(change)
        self.reject(lambda r: r["observations"][1].update(stdout=r["observations"][1]["stdout"] + "extra"))

    def test_interrupted_publication_is_not_a_final_report(self):
        """Only the exact pending path is retired; unrelated files survive."""
        final = self.root / "report.json"
        pending = final.with_suffix(".json.tmp")
        neighbor = self.root / "neighbor.json"
        final.write_text("stale")
        pending.write_text("interrupted")
        neighbor.write_text("preserve")
        PROOF.retire(final)
        self.assertFalse(final.exists())
        self.assertFalse(pending.exists())
        self.assertEqual(neighbor.read_text(), "preserve")
        with mock.patch.object(Path, "replace", side_effect=OSError("injected rename failure")):
            with self.assertRaises(OSError):
                PROOF.publish(self.report("nccl"), final)
        self.assertFalse(final.exists())
        PROOF.retire(final)

    def test_compile_rejects_unpatched_receipt_before_oracle(self):
        """A current unpatched native DSO cannot receive a new host-only certificate."""
        source = self.root / "native"
        library = source / "build/lib/lib.so.1"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"unpatched fake native build")
        receipt = self.root / "receipt.txt"
        receipt.write_text("old unpatched receipt")
        report = self.root / "report.json"
        report.write_text("stale final")
        report.with_suffix(".json.tmp").write_text("stale pending")
        args = argparse.Namespace(backend="nccl", report=report, library=library, source=source,
            receipt=receipt, compiler="g++", candidate_source=None, rccl_build=None, depfile=None)
        with mock.patch.object(PROOF, "run_oracles", side_effect=AssertionError("oracle should not run")):
            with self.assertRaisesRegex(RuntimeError, "lacks new source patch"):
                PROOF.compile_report(args)
        self.assertFalse(report.exists())
        self.assertFalse(report.with_suffix(".json.tmp").exists())
        self.assertEqual(receipt.read_text(), "old unpatched receipt")

    def test_output_must_not_delete_receipt(self):
        """Protect native inputs before retiring a stale destination."""
        receipt = self.root / "receipt.json"
        receipt.write_text("original receipt")
        args = argparse.Namespace(backend="nccl", report=receipt, receipt=receipt,
            library=self.library, source=None, compiler="g++", candidate_source=None,
            rccl_build=None, depfile=None)
        with self.assertRaisesRegex(RuntimeError, "aliases"):
            PROOF.compile_report(args)
        self.assertEqual(receipt.read_text(), "original receipt")

    def test_canonical_registration_is_explicit_and_builder_only(self):
        """Both family regressions and validator negatives enter canonical preflight."""
        text = (ROOT / "tests/v2/cmake/V2NativeSharedFileLifecycleTests.cmake").read_text()
        for name in ("NCCLSharedFileLifecycle", "RCCLSharedFileLifecycle", "NativeSharedFileProofPackaging"):
            self.assertIn("V2_Integration_" + name, text)
        self.assertIn("ProductionTestPreflight", text)
        self.assertIn("--backend nccl", text)
        self.assertIn("--backend rccl", text)
        self.assertIn("DEPFILE", text)
        self.assertIn("V2_UNIT_GATE_TARGETS", text)
        self.assertIn("include(cmake/V2NativeSharedFileLifecycleTests.cmake)", (ROOT / "tests/v2/CMakeLists.txt").read_text())

    def test_ordered_patch_receipt_and_installer_closure(self):
        """One actual ordered chain and one existing receipt authenticate the new delta."""
        shell = (PROOF.DOCKER / "apply-rccl-capture-patch.sh").read_text()
        ordered = re.search(r"for rccl_patch_name in (.*?); do", shell).group(1).split()
        self.assertEqual(ordered.count("rccl-partial-file-cleanup.patch"), 1)
        self.assertLess(ordered.index("rccl-native-host-storage.patch"), ordered.index("rccl-partial-file-cleanup.patch"))
        self.assertIn('--upgrade-patch "${rccl_patch_root}/rccl-partial-file-cleanup.patch"', shell)
        for path, marker in ((ROOT / "Dockerfile", "-storage-${rccl_storage_patch_sha}-shm-${rccl_shm_patch_sha}-selector-"),
                             (ROOT / "src/v2/CMakeLists.txt", "-storage-${RCCL_HOST_STORAGE_PATCH_HASH}-shm-${RCCL_SHM_PATCH_HASH}-selector-")):
            self.assertIn(marker, path.read_text())
        docker = (ROOT / "Dockerfile").read_text()
        for name in ("shared-file-lifecycle/", "patches/nccl-partial-file-cleanup.patch", "patches/rccl-partial-file-cleanup.patch"):
            self.assertIn("COPY scripts/docker/" + name, docker)
        installer = (PROOF.DOCKER / "install-nccl.sh").read_text()
        self.assertLess(installer.index('apply "${nccl_shm_patch}"'), installer.index('make -C'))
        self.assertIn('"${nccl_shm_probe}" compile', installer)
        self.assertIn('"${nccl_shm_probe}" verify', installer)


if __name__ == "__main__":
    unittest.main()
