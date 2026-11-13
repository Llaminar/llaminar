#!/usr/bin/env python3
"""Adversarial passive validation of the native NCCL thread-lifetime certificate.

These tiny validator fixtures are not native DSOs or native certificates. The
real complete-source pthread replay and literal original execute only in the
dependency builder; installed Unit/preflight needs no compiler, SDK or device.
"""
import argparse
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location("nccl_thread_lifetime_proof",
    ROOT / "scripts/docker/nccl-thread-lifetime/proof.py")
PROOF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROOF)


class NativeThreadProof(unittest.TestCase):
    """Partial, stale or disconnected observations cannot authenticate native lifetime."""
    def setUp(self):
        """Construct an explicit validator-only byte identity and complete authored report."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.library = self.base / "validator-only-library"
        self.library.write_bytes(b"validator-only identity, never loaded as a DSO")
        self.receipt = PROOF.receipt_prefix() + "-gencode=arch=compute_86,code=sm_86"
        self.report = dict(schema=1, certificate=True, sources=PROOF.source_identity(),
            library_sha256=PROOF.digest(self.library), build_receipt=self.receipt,
            compiler_version="validator-only compiler identity", native_source_sha256="a" * 64,
            original_source_sha256="b" * 64, native_header_sha256="c" * 64,
            original_header_sha256="d" * 64, native_init_sha256="e" * 64,
            native_caller_sha256="f" * 64, observations=[
                dict(variant="native", compile_exit=0, returncode=0,
                     stdout=PROOF.NATIVE_OUTPUT, stderr=""),
                dict(variant="original", compile_exit=0, runs=[
                    dict(returncode=87, stdout="", stderr=PROOF.ORIGINAL_ERROR) for _ in range(20)])])
        self.report["compiler_identity"] = dict(executable="/validator-only/compiler",
            version=self.report["compiler_version"], sha256="1" * 64)

    def reject(self, report):
        """Require a precise fail-closed validator error for one altered observation."""
        with self.assertRaises(RuntimeError):
            PROOF.verify_report(report, self.library, self.receipt)

    def test_complete_matching_report(self):
        """Only the complete matching controller replay and literal original can pass."""
        PROOF.verify_report(self.report, self.library, self.receipt)

    def test_changed_dso(self):
        """An old proof cannot authenticate newly selected runtime bytes."""
        self.library.write_bytes(b"stale/different runtime")
        self.reject(self.report)

    def test_missing_all_required_identity_fields(self):
        """Every source/header/caller/compiler/receipt identity is mandatory."""
        for key in ("schema", "certificate", "sources", "library_sha256", "build_receipt",
                    "compiler_version", "native_source_sha256", "original_source_sha256",
                    "native_header_sha256", "original_header_sha256", "native_init_sha256",
                    "native_caller_sha256", "compiler_identity"):
            with self.subTest(key=key):
                report = copy.deepcopy(self.report)
                del report[key]
                self.reject(report)

    def test_stale_every_source_patch_or_probe(self):
        """The common source receipt pins every executable oracle and native patch."""
        for key in PROOF.INPUTS:
            with self.subTest(source=key):
                report = copy.deepcopy(self.report)
                report["sources"][key] = "0" * 64
                self.reject(report)

    def test_receipt_requires_source_chain_policy_and_architecture(self):
        """Matching arbitrary, old-policy or missing-architecture text is insufficient."""
        for receipt in ("unrelated", PROOF.receipt_prefix(),
                        self.receipt.replace(PROOF.GIN.REVISION, "0" * 40),
                        self.receipt.replace(PROOF.GIN.proof_identity(), "0" * 64)):
            with self.subTest(receipt=receipt):
                report = copy.deepcopy(self.report)
                report["build_receipt"] = receipt
                with self.assertRaises(RuntimeError):
                    PROOF.verify_report(report, self.library, receipt)

    def test_source_only_oracle_is_not_a_runtime_certificate(self):
        """Even complete source-only outcomes cannot attest an installed library."""
        self.report["certificate"] = False
        self.reject(self.report)

    def test_original_source_and_header_must_be_distinct(self):
        """Compiling the repaired implementation twice cannot prove the defect."""
        for native, original in (("native_source_sha256", "original_source_sha256"),
                                 ("native_header_sha256", "original_header_sha256")):
            report = copy.deepcopy(self.report)
            report[original] = report[native]
            self.reject(report)

    def test_malformed_source_and_caller_digests(self):
        """Missing or incomplete actual-caller extraction fails before reuse."""
        for key in ("native_source_sha256", "original_source_sha256", "native_header_sha256",
                    "original_header_sha256", "native_init_sha256", "native_caller_sha256"):
            for bad in (None, "", "short", "x" * 64):
                report = copy.deepcopy(self.report)
                report[key] = bad
                self.reject(report)

    def test_unique_complete_controls(self):
        """Missing original, duplicated controls and reordered observations fail."""
        native, original = self.report["observations"]
        for controls in (None, [], [native], [original], [native, native], [original, native],
                         [native, original, original], [native, None]):
            report = copy.deepcopy(self.report)
            report["observations"] = controls
            self.reject(report)

    def test_compile_failure_cannot_replace_execution(self):
        """The native and original implementations must both really compile."""
        for index in (0, 1):
            report = copy.deepcopy(self.report)
            report["observations"][index]["compile_exit"] = 1
            self.reject(report)

    def test_boolean_and_noninteger_outcome_codes_reject(self):
        """JSON false/0.0 cannot impersonate an actual native integer process status."""
        for index, field in ((0, "compile_exit"), (1, "compile_exit"), (0, "returncode")):
            for bad in (False, 0.0, "0"):
                report = copy.deepcopy(self.report)
                report["observations"][index][field] = bad
                self.reject(report)
        report = copy.deepcopy(self.report)
        report["observations"][1]["runs"][0]["returncode"] = 87.0
        self.reject(report)

    def test_compiler_identity_is_complete_and_consistent(self):
        """Installed validation needs the authenticated builder identity, not a local compiler."""
        for field, bad in (("version", "changed"), ("sha256", "short"), ("executable", "")):
            report = copy.deepcopy(self.report)
            report["compiler_identity"][field] = bad
            self.reject(report)

    def test_literal_aware_exact_caller_parser(self):
        """Comments, ordinary/raw literals, escaped quotes and separators cannot change scopes."""
        body = r'''ncclResult_t ncclCommEnsureReady(ncclComm_t comm) {
  const char* s = "} \" {";
  const char* raw = R"tag( } { " nested )tag";
  char brace = '}'; int separated = 1'000;
  /* } { */ // } {
  return ncclSuccess;
}'''
        fake = '/* ncclResult_t ncclCommEnsureReady(ncclComm_t comm) { } */\n'
        self.assertEqual(PROOF.extract_caller(fake + body + "\n"), body + "\n")

    def test_missing_duplicate_unbalanced_or_unbounded_caller_rejects(self):
        """A changed native caller cannot silently produce a different partial body."""
        body = "ncclResult_t ncclCommEnsureReady(ncclComm_t comm) { return ncclSuccess; }"
        for source in ("", body + "\n" + body, body[:-1], body + "\n}",
                       body[:-1] + " " * 8200 + "}", body[:-1] + '"unterminated',
                       body[:-1] + "/* unterminated", body[:-1] + 'R"tag(unterminated'):
            with self.subTest(source=source[:100]):
                with self.assertRaises(RuntimeError):
                    PROOF.extract_caller(source)

    def snapshot_fixture(self):
        """Create small source-identity fixtures only; no native compiler or DSO executes."""
        source = self.base / "snapshot-source"
        (source / "src/include").mkdir(parents=True, exist_ok=True)
        for name, data in (("src/group.cc", "group fixture"), ("src/include/group.h", "header fixture"),
                           ("src/init.cc", "ncclResult_t ncclCommEnsureReady(ncclComm_t comm) { return ncclSuccess; }")):
            (source / name).write_text(data)
        report = copy.deepcopy(self.report)
        for key, name in (("native_source_sha256", "src/group.cc"),
                          ("native_header_sha256", "src/include/group.h"),
                          ("native_init_sha256", "src/init.cc")):
            report[key] = PROOF.digest(source / name)
        report["native_caller_sha256"] = PROOF.hashlib.sha256(PROOF.caller_body(source).encode()).hexdigest()
        return source, report

    def test_postoracle_source_and_compiler_reauthentication(self):
        """Every source/caller/compiler replacement rejects the already completed observations."""
        source, report = self.snapshot_fixture()
        with patch.object(PROOF, "compiler_identity", return_value=report["compiler_identity"]):
            PROOF.authenticate_inputs(source, "validator-only", report)
            for name in ("src/group.cc", "src/include/group.h", "src/init.cc"):
                original = (source / name).read_bytes()
                (source / name).write_bytes(original + b" changed")
                with self.assertRaises(RuntimeError):
                    PROOF.authenticate_inputs(source, "validator-only", report)
                (source / name).write_bytes(original)
        changed = copy.deepcopy(report["compiler_identity"])
        changed["sha256"] = "2" * 64
        with patch.object(PROOF, "compiler_identity", return_value=changed):
            with self.assertRaises(RuntimeError):
                PROOF.authenticate_inputs(source, "validator-only", report)
        with patch.object(PROOF, "source_identity", return_value={}):
            with self.assertRaises(RuntimeError):
                PROOF.authenticate_inputs(source, "validator-only", report)

    def test_partial_or_unrelated_native_outcome(self):
        """Skipped replay, wrong counts, timeout and unexpected diagnostics fail."""
        for key, value in (("returncode", 1), ("returncode", 124), ("stdout", "PASS"),
                           ("stdout", PROOF.NATIVE_OUTPUT.replace("2360", "2359")),
                           ("stderr", "unexpected runtime warning")):
            report = copy.deepcopy(self.report)
            report["observations"][0][key] = value
            self.reject(report)

    def test_every_original_run_is_required(self):
        """One successful-looking rejection cannot replace all twenty original controls."""
        for count in (0, 1, 19, 21):
            report = copy.deepcopy(self.report)
            report["observations"][1]["runs"] = [
                dict(returncode=87, stdout="", stderr=PROOF.ORIGINAL_ERROR) for _ in range(count)]
            self.reject(report)

    def test_original_failure_must_be_exact_destructor_order_fault(self):
        """Passing, compile failures, crashes and unrelated exits cannot certify the repair."""
        for key, value in (("returncode", 0), ("returncode", 124), ("returncode", -6),
                           ("returncode", -11), ("stdout", "unexpected"),
                           ("stderr", "REJECT unrelated invariant\n")):
            report = copy.deepcopy(self.report)
            report["observations"][1]["runs"][13][key] = value
            self.reject(report)

    def test_unrelated_library_rejected_before_native_compilation(self):
        """A copied runtime outside the selected source build cannot mint a certificate."""
        source = self.base / "unbuilt-source"
        source.mkdir()
        output = self.base / "stale-report.json"
        output.write_text("stale previous report")
        pending = output.with_suffix(".tmp")
        pending.write_text("stale interrupted report")
        args = argparse.Namespace(source=source, library=self.library, report=output)
        with patch.object(PROOF, "run_oracles", side_effect=AssertionError("must not compile")):
            with self.assertRaisesRegex(RuntimeError, "actual DSO"):
                PROOF.compile_report(args)
        self.assertFalse(output.exists())
        self.assertFalse(pending.exists())

    def test_interrupted_publication_retires_both_outputs(self):
        """An interruption after pending write cannot leave a reusable final or pending report."""
        source, report = self.snapshot_fixture()
        library = source / "build/lib/validator-only.so.1"
        library.parent.mkdir(parents=True)
        library.write_bytes(self.library.read_bytes())
        receipt = source / "receipt"
        receipt.write_text(self.receipt)
        output = source / "report.json"
        output.write_text("stale final")
        output.with_suffix(".tmp").write_text("stale pending")
        args = argparse.Namespace(source=source, library=library, report=output,
                                  receipt=receipt, sdk=self.base, compiler="validator-only")
        with patch.object(PROOF, "run_oracles", return_value=report), \
             patch.object(PROOF, "authenticate_inputs", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                PROOF.compile_report(args)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".tmp").exists())

    def test_library_or_receipt_changes_during_oracle_prevent_publication(self):
        """A runtime replacement during replay cannot be rebound to old successful outcomes."""
        source, report = self.snapshot_fixture()
        library = source / "build/lib/validator-only.so.1"
        library.parent.mkdir(parents=True)
        receipt = source / "receipt"
        output = source / "report.json"
        args = argparse.Namespace(source=source, library=library, report=output,
                                  receipt=receipt, sdk=self.base, compiler="validator-only")
        for changed in (library, receipt):
            library.write_bytes(self.library.read_bytes())
            receipt.write_text(self.receipt)
            def oracle(*_):
                changed.write_bytes(b"changed during actual oracle interval")
                return copy.deepcopy(report)
            with patch.object(PROOF, "run_oracles", side_effect=oracle), \
                 patch.object(PROOF, "authenticate_inputs", return_value=None):
                with self.assertRaises(RuntimeError):
                    PROOF.compile_report(args)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".tmp").exists())

    def test_installer_docker_and_preflight_keep_the_same_proof(self):
        """Early dependency build, passive Unit and explicit preflight use one source identity."""
        installer = (ROOT / "scripts/docker/install-nccl.sh").read_text()
        docker = (ROOT / "Dockerfile").read_text()
        registration = (ROOT / "tests/v2/cmake/V2NCCLThreadLifetimeTests.cmake").read_text()
        for text in ('apply "${nccl_thread_patch}"', '"${nccl_thread_probe}" compile',
                     '"${nccl_thread_probe}" verify', '"${nccl_thread_report}"'):
            self.assertIn(text, installer)
        self.assertLess(docker.index("COPY scripts/docker/nccl-thread-lifetime/"),
                        docker.index("MODE=full /tmp/install-scripts/install-cuda.sh"))
        self.assertIn("V2_UNIT_GATE_TARGETS v2_nccl_thread_lifetime_proof", registration)
        self.assertIn("V2_Integration_NCCLThreadLifetime APPEND PROPERTY LABELS ProductionTestPreflight",
                      registration)
        self.assertIn("include(cmake/V2NCCLThreadLifetimeTests.cmake)",
                      (ROOT / "tests/v2/CMakeLists.txt").read_text())
        self.assertEqual(PROOF.source_identity(), PROOF.GIN.source_identity())


if __name__ == "__main__":
    unittest.main()
