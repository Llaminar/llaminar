#!/usr/bin/env python3
"""Adversarial device-free validation of native RCCL thread lifetime evidence.

Only fake identity bytes are created here. Actual native group/caller execution
and literal old-source controls belong to the builder-owned source proof.
"""
import copy
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "rccl_thread_proof", ROOT / "tests/v2/integration/build/test_rccl_thread_lifetime.py")
PROOF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROOF)


class RCCLThreadProofTests(unittest.TestCase):
    """Require complete authenticated outcomes and failure-safe publication."""

    def setUp(self):
        """Construct a format-valid report without loading a native library."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.library = self.directory / "librccl.so"
        self.library.write_bytes(b"thread-proof-fake-library-identity")
        files = dict(group="/build/hipify/src/group.cc", header="/build/hipify/src/include/group.h",
                     caller="/build/hipify/src/init.cc")
        closure = {files["group"]: "1" * 64, files["header"]: "2" * 64,
                   files["caller"]: "3" * 64, "/sdk/lib/llvm/bin/clang": "4" * 64}
        self.report = dict(
            schema=1, certificate=True, native_init_runtime_qualified=False,
            sources=PROOF.source_identity(), library_sha256=PROOF.digest(self.library),
            build_receipt="fixture" + PROOF.receipt_suffix(),
            native_source_sha256="1" * 64, native_header_sha256="2" * 64,
            original_source_sha256="5" * 64, original_header_sha256="6" * 64,
            literal_caller_sha256="3" * 64, compile_database_sha256="7" * 64,
            compiler=dict(path="/sdk/lib/llvm/bin/clang", stat=[1, 2, 3, 4, 5]),
            compiler_version="fixture compiler", sdk="/sdk", architecture="gfx906",
            dependency_closure=closure, dependency_closure_sha256=PROOF.closure_digest(closure),
            native_files=files,
            native_compile_inputs=[dict(source=files[name], flags=["-D__HIP_PLATFORM_AMD__=1", "-fgpu-rdc"])
                                   for name in ("group", "caller")],
            observations=[
                dict(variant="native", compile_exit=0, full_group_compile_exit=0,
                     group=self.execution("native", "group"), caller=self.execution("native", "caller")),
                dict(variant="original", compile_exit=0, rejections=[
                    dict(iteration=index, group=self.execution("original", "group"),
                         caller=self.execution("original", "caller"))
                    for index in range(PROOF.REPETITIONS)])])

    @staticmethod
    def execution(variant, suite):
        """Create exact oracle output; tests mutate one invariant at a time."""
        measured = copy.deepcopy(PROOF.EXPECTED[(variant, suite)])
        return dict(returncode=int(variant == "original"), stdout=json.dumps(measured) + "\n",
                    stderr="", measured=measured)

    def reject(self, change):
        """Apply one mutation while preserving the original valid fixture."""
        report = copy.deepcopy(self.report)
        change(report)
        with self.assertRaises(RuntimeError):
            PROOF.verify_report(report, self.library)

    def test_complete_report_and_relocated_library_pass(self):
        """Installed checking uses content identity without vendor/compiler access."""
        PROOF.verify_report(self.report, self.library)
        relocated = self.directory / "relocated.so"
        relocated.write_bytes(self.library.read_bytes())
        PROOF.verify_report(self.report, relocated)

    def test_container_stages_complete_applier_inputs_before_native_build(self):
        """The actual ordered applier must find its helper and every patch in Docker."""
        applier = (ROOT / "scripts/docker/apply-rccl-capture-patch.sh").read_text()
        patches = set(re.findall(r"rccl-[a-z0-9-]+\.patch", applier))
        required = {
            "scripts/docker/apply-rccl-capture-patch.sh": "/tmp/install-scripts/apply-rccl-capture-patch.sh",
            "scripts/docker/apply-rccl-patch-chain.py": "/tmp/install-scripts/apply-rccl-patch-chain.py",
            **{"scripts/docker/patches/" + name: "/tmp/install-scripts/patches/" + name
               for name in patches},
        }
        docker = (ROOT / "Dockerfile").read_text()
        before_build = docker[:docker.index("bash /tmp/install-scripts/apply-rccl-capture-patch.sh")]
        staged = {}
        for line in before_build.replace("\\\n", " ").splitlines():
            fields = shlex.split(line, comments=True)
            if not fields or fields[0] != "COPY" or fields[1].startswith("--"):
                continue
            destination = fields[-1]
            for source in fields[1:-1]:
                staged[source] = (str(PurePosixPath(destination) / PurePosixPath(source).name)
                                  if destination.endswith("/") else destination)
        for source, destination in required.items():
            with self.subTest(source=source):
                self.assertEqual(staged.get(source), destination)

    def test_source_only_cannot_certify_runtime(self):
        """A source-only result cannot be promoted through a copied report field."""
        for value in (None, False, 1, "true"):
            self.reject(lambda r, value=value: r.update(certificate=value))
        self.reject(lambda r: r.update(native_init_runtime_qualified=True))

    def test_repair_order_and_all_current_patch_hashes_are_required(self):
        """Thread identity precedes the existing strict protocol/smallAG suffix."""
        suffix = PROOF.receipt_suffix()
        self.assertLess(suffix.index("-thread-"), suffix.index("-protocol-"))
        for value in (None, "", suffix, "fixture" + suffix + "-unknown", "fixture" + suffix[:-1],
                      "fixture-protocol-" + "1" * 64 + "-smallag-" + "2" * 64 + "-thread-" + "3" * 64):
            self.reject(lambda r, value=value: r.update(build_receipt=value))

    def test_changed_runtime_or_each_regression_source_rejects(self):
        """A stale library or any source/extractor/compiler-driver input invalidates proof."""
        for name in PROOF.INPUTS:
            self.reject(lambda r, name=name: r["sources"].update({name: "0" * 64}))
        self.library.write_bytes(b"replaced native dependency")
        with self.assertRaises(RuntimeError):
            PROOF.verify_report(self.report, self.library)

    def test_missing_or_unchanged_original_source_and_header_reject(self):
        """The negative must reverse both actual source and its private job layout."""
        for field in ("native_source_sha256", "native_header_sha256", "original_source_sha256",
                      "original_header_sha256", "literal_caller_sha256", "compile_database_sha256"):
            self.reject(lambda r, field=field: r.pop(field))
        self.reject(lambda r: r.update(original_source_sha256=r["native_source_sha256"]))
        self.reject(lambda r: r.update(original_header_sha256=r["native_header_sha256"]))

    def test_missing_compiler_or_sdk_context_rejects(self):
        """Installed verification authenticates recorded context without probing it."""
        for field in ("compiler", "compiler_version", "sdk", "architecture"):
            self.reject(lambda r, field=field: r.pop(field))
        for value in (None, "relative", ""):
            self.reject(lambda r, value=value: r["compiler"].update(path=value))
        self.reject(lambda r: r["compiler"].update(stat=[1, 2]))

    def test_compiler_dependency_closure_is_complete_and_bound(self):
        """Dropping a true header or changing its recorded content cannot silently pass."""
        self.reject(lambda r: r["dependency_closure"].pop(r["native_files"]["header"]))
        self.reject(lambda r: r["dependency_closure"].update({"/new/header.h": "8" * 64}))
        def remove_actual_header(report):
            report["dependency_closure"].pop(report["native_files"]["header"])
            report["dependency_closure_sha256"] = PROOF.closure_digest(report["dependency_closure"])
        self.reject(remove_actual_header)
        self.reject(lambda r: r["native_compile_inputs"][0].update(flags=["-fno-gpu-rdc"]))
        self.reject(lambda r: r["native_compile_inputs"][1].update(flags=["-fgpu-rdc", "-fgpu-rdc"]))

    def test_missing_duplicate_or_reordered_variants_reject(self):
        """Native success alone never substitutes for the original controls."""
        self.reject(lambda r: r["observations"].pop())
        self.reject(lambda r: r["observations"].reverse())
        self.reject(lambda r: r["observations"].append(copy.deepcopy(r["observations"][0])))
        self.reject(lambda r: r["observations"][0].update(compile_exit=1))
        self.reject(lambda r: r["observations"][0].update(full_group_compile_exit=False))

    def test_every_original_rejection_is_required(self):
        """Twenty actual old-code rejections must retain exact polarity and iteration."""
        self.reject(lambda r: r["observations"][1]["rejections"].pop())
        self.reject(lambda r: r["observations"][1]["rejections"][1].update(iteration=0))
        for index in range(PROOF.REPETITIONS):
            for suite in ("group", "caller"):
                self.reject(lambda r, index=index, suite=suite:
                            r["observations"][1]["rejections"][index][suite].update(returncode=0))

    def test_exact_oracle_shape_stdout_and_nonvacuous_cases_required(self):
        """Correct totals cannot conceal missing real threads or the ignored-error caller."""
        for suite in ("group", "caller"):
            self.reject(lambda r, suite=suite: r["observations"][0][suite].update(stderr="unexpected"))
            self.reject(lambda r, suite=suite: r["observations"][0][suite].update(stdout="{}"))
            for field in PROOF.EXPECTED[("native", suite)]:
                self.reject(lambda r, suite=suite, field=field:
                            r["observations"][0][suite]["measured"].pop(field))
        self.reject(lambda r: r["observations"][0]["group"]["measured"].update(violations=False))
        self.reject(lambda r: r["observations"][0]["caller"]["measured"].update(ordinary_error_cases=0))

    def test_atomic_publication_and_failed_rebuild_retirement(self):
        """Only final verified evidence is published; stale names retire before work."""
        atomic = PROOF.module("test_rccl_small_allgather_policy")
        report = self.directory / "proof.json"
        report.write_text("stale")
        pending = report.with_suffix(".json.tmp")
        pending.write_text("interrupted")
        atomic.retire_proof(report)
        self.assertFalse(report.exists())
        self.assertFalse(pending.exists())
        atomic.publish_verified_report(self.report, report)
        self.assertEqual(json.loads(report.read_text()), self.report)
        self.assertFalse(pending.exists())

    def test_depfile_uses_real_compiler_dependencies_and_escapes_paths(self):
        """Make continuations, spaces, dollars and punctuation preserve exact files."""
        inputs = [self.directory / name for name in ("one.h", "two words.h", "cost$1.h", "tag#x:y.h")]
        for path in inputs:
            path.write_text("header")
        depfile = self.directory / "proof.d"
        target = self.directory / "proof.json"
        PROOF.write_depfile(depfile, target, inputs)
        self.assertEqual(PROOF.dependency_paths(depfile, self.directory), set(inputs))
        self.assertIn("\\ ", depfile.read_text())
        self.assertIn("$$", depfile.read_text())
        lines = depfile.read_text().replace("one.h ", "one.h \\\n  ")
        depfile.write_text(lines)
        self.assertEqual(PROOF.dependency_paths(depfile, self.directory), set(inputs))

    def test_rejected_or_failed_compile_retires_stale_report_and_depfile(self):
        """A failed rebuild cannot leave an earlier successful installed certificate."""
        from types import SimpleNamespace
        for override in (None, self.directory / "override"):
            report = self.directory / "stale.json"
            depfile = self.directory / "stale.d"
            for path in (report, depfile, report.with_suffix(".json.tmp"), depfile.with_suffix(".d.tmp")):
                path.write_text("stale")
            args = SimpleNamespace(report=report, depfile=depfile, library=self.library,
                                   candidate_source=override, rccl_build=self.directory / "absent")
            with self.assertRaises((RuntimeError, OSError)):
                PROOF.compile_report(args)
            for path in (report, depfile, report.with_suffix(".json.tmp"), depfile.with_suffix(".d.tmp")):
                self.assertFalse(path.exists())
            self.assertTrue(self.library.exists())

    def test_interrupted_publication_never_creates_final_evidence(self):
        """An interrupted rename leaves only a removable temporary record."""
        atomic = PROOF.module("test_rccl_small_allgather_policy")
        destination = self.directory / "interrupted.json"
        with mock.patch.object(Path, "replace", side_effect=OSError("injected rename failure")):
            with self.assertRaises(OSError):
                atomic.publish_verified_report(self.report, destination)
        self.assertFalse(destination.exists())
        self.assertTrue(destination.with_suffix(".json.tmp").exists())
        atomic.retire_proof(destination)
        self.assertFalse(destination.with_suffix(".json.tmp").exists())

    def test_diagnostic_mode_cannot_write_a_certificate(self):
        """Source override is a separate interface and cannot certify the selected DSO."""
        from types import SimpleNamespace
        args = SimpleNamespace(report=self.directory / "forbidden.json", depfile=None,
                               library=None)
        with self.assertRaisesRegex(RuntimeError, "cannot emit"):
            PROOF.source_only(args)
        self.assertFalse(args.report.exists())


if __name__ == "__main__":
    unittest.main()
