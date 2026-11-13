#!/usr/bin/env python3
"""Adversarial verification of the source-bound NCCL GIN regression receipt.

These tests require no accelerator, SDK or compiler. Actual native selection and
the original defect execute in the dependency builder, then Unit/preflight
authenticates those observations against the selected runtime library.
"""

import copy
import importlib.util
from pathlib import Path
import signal
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "nccl_gin_selection_proof", ROOT / "scripts/docker/nccl-gin-selection/proof.py")
PROOF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROOF)


class NativeGinProof(unittest.TestCase):
    """Missing, stale or vacuous native observations must never become a pass."""

    def setUp(self):
        """Use a tiny authored DSO identity only for the receipt validator."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.library = Path(self.temporary.name) / "libllaminar_nccl.so"
        self.library.write_bytes(b"authored validator-only native library identity")
        self.receipt = PROOF.receipt_prefix() + "-gencode=arch=compute_86,code=sm_86"
        self.report = dict(schema=1, sources=PROOF.source_identity(),
            library_sha256=PROOF.digest(self.library), build_receipt=self.receipt,
            native_source_sha256="a" * 64, original_source_sha256="b" * 64,
            compiler_version="authored-validator-compiler", observations=[
                dict(variant="native", compile_exit=0, returncode=0,
                     stdout=PROOF.NATIVE_OUTPUT,
                     stderr="".join(f"replay {case}\n" for case in PROOF.CASES) * 20),
                dict(variant="original", compile_exit=0, returncode=-signal.SIGSEGV,
                     stdout="", stderr="replay auto-available\nreplay auto-zero-devices\n")])

    def test_complete_matching_report(self):
        """Only the exact full replay and reproduced original defect are accepted."""
        PROOF.verify_report(self.report, self.library, self.receipt)

    def test_changed_library(self):
        """An old proof cannot certify a newly installed dependency."""
        self.library.write_bytes(b"different native library")
        with self.assertRaisesRegex(RuntimeError, "source/runtime"):
            PROOF.verify_report(self.report, self.library, self.receipt)

    def test_missing_identity_or_compiler(self):
        """Every builder, source and original-code identity is mandatory."""
        for field in ("schema", "sources", "library_sha256", "build_receipt",
                      "compiler_version", "native_source_sha256", "original_source_sha256"):
            with self.subTest(field=field):
                report = copy.deepcopy(self.report)
                del report[field]
                with self.assertRaises(RuntimeError):
                    PROOF.verify_report(report, self.library, self.receipt)

    def test_stale_source_or_patch(self):
        """Every executable oracle and dependency patch participates in identity."""
        for name in PROOF.INPUTS:
            with self.subTest(source=name):
                report = copy.deepcopy(self.report)
                report["sources"][name] = "c" * 64
                with self.assertRaises(RuntimeError):
                    PROOF.verify_report(report, self.library, self.receipt)

    def test_receipt_authenticates_source_chain_and_architecture(self):
        """A matching arbitrary receipt is insufficient for canonical admission."""
        for receipt in ("unrelated", PROOF.receipt_prefix(),
                        self.receipt.replace(PROOF.REVISION, "0" * 40)):
            with self.subTest(receipt=receipt):
                report = copy.deepcopy(self.report)
                report["build_receipt"] = receipt
                with self.assertRaises(RuntimeError):
                    PROOF.verify_report(report, self.library, receipt)

    def test_native_and_original_sources_must_differ(self):
        """Compiling the fixed implementation twice cannot prove the regression."""
        self.report["original_source_sha256"] = self.report["native_source_sha256"]
        with self.assertRaisesRegex(RuntimeError, "distinct"):
            PROOF.verify_report(self.report, self.library, self.receipt)

    def test_complete_unique_controls(self):
        """Missing, duplicate or reordered observations fail closed."""
        native, original = self.report["observations"]
        for observations in ([], [native], [original], [native, native],
                             [original, native], [native, original, original]):
            with self.subTest(observations=observations):
                report = copy.deepcopy(self.report)
                report["observations"] = observations
                with self.assertRaises(RuntimeError):
                    PROOF.verify_report(report, self.library, self.receipt)

    def test_compile_failure_is_not_negative_control(self):
        """The old native implementation must compile and actually execute."""
        for index in (0, 1):
            with self.subTest(index=index):
                report = copy.deepcopy(self.report)
                report["observations"][index]["compile_exit"] = 1
                with self.assertRaises(RuntimeError):
                    PROOF.verify_report(report, self.library, self.receipt)

    def test_vacuous_or_partial_native_replay(self):
        """A shortened matrix, wrong counts or unrelated failure cannot pass."""
        for field, value in (("returncode", 1), ("stdout", "PASS"),
                             ("stderr", ""), ("stderr", "replay auto-zero-devices\n")):
            with self.subTest(field=field, value=value):
                report = copy.deepcopy(self.report)
                report["observations"][0][field] = value
                with self.assertRaises(RuntimeError):
                    PROOF.verify_report(report, self.library, self.receipt)

    def test_original_failure_must_be_exact_zero_device_fault(self):
        """A timeout, early crash or a control that passes cannot authenticate it."""
        for field, value in (("returncode", 0), ("returncode", 124),
                             ("returncode", -signal.SIGABRT), ("stdout", "unexpected"),
                             ("stderr", "replay auto-available\n")):
            with self.subTest(field=field, value=value):
                report = copy.deepcopy(self.report)
                report["observations"][1][field] = value
                with self.assertRaises(RuntimeError):
                    PROOF.verify_report(report, self.library, self.receipt)

    def test_installer_and_preflight_keep_the_proof(self):
        """The early Docker dependency build and installed gate share one oracle."""
        installer = (ROOT / "scripts/docker/install-nccl.sh").read_text()
        docker = (ROOT / "Dockerfile").read_text()
        registration = (ROOT / "tests/v2/cmake/V2NCCLGinSelectionTests.cmake").read_text()
        for fragment in ('apply "${nccl_gin_patch}"', '"${nccl_gin_probe}" compile',
                         '"${nccl_gin_probe}" verify', '"${nccl_gin_report}"'):
            self.assertIn(fragment, installer)
        self.assertLess(docker.index("COPY scripts/docker/nccl-gin-selection/"),
                        docker.index("MODE=full /tmp/install-scripts/install-cuda.sh"))
        self.assertIn("V2_UNIT_GATE_TARGETS v2_nccl_gin_selection_proof", registration)
        self.assertIn("V2_Integration_NCCLGinSelection APPEND PROPERTY LABELS ProductionTestPreflight",
                      registration)
        self.assertIn("include(cmake/V2NCCLGinSelectionTests.cmake)",
                      (ROOT / "tests/v2/CMakeLists.txt").read_text())


if __name__ == "__main__":
    unittest.main()
