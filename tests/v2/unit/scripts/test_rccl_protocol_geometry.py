#!/usr/bin/env python3
"""Device-free regressions for native RCCL protocol proof freshness and packaging.

Actual vendor consumers execute in the builder-owned functional probe. These
tests ensure installed preflight cannot accept stale libraries, omitted modes,
vacuous negative controls or a dependency layer missing the production patch.
"""

import copy
import hashlib
import os
import re
import subprocess
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "rccl_protocol_proof", ROOT / "tests/v2/integration/build/test_rccl_protocol_geometry.py")
PROOF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROOF)


class RCCLProtocolProofTests(unittest.TestCase):
    """Authenticate the complete native/negative outcome set against its inputs."""

    def setUp(self):
        """Create only a tiny fake identity artifact; never load any DSO or GPU."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.library = self.directory / "librccl.so"
        self.library.write_bytes(b"protocol-proof-library-identity")
        self.report = dict(
            schema=1, sources=PROOF.source_identity(), library_sha256=PROOF.digest(self.library),
            rccl_receipt=("fixture-protocol-" + PROOF.digest(ROOT / PROOF.PATCH)
                          + "-smallag-" + PROOF.digest(ROOT / PROOF.SMALL_ALLGATHER_PATCH)),
            compiler_version="fixture-compiler", native_source_sha256="source",
            native_arch_source_sha256="architecture-source", observations=[
                dict(variant=variant, mode=mode, returncode=0 if variant == "native" else 1,
                     measured=dict(mode=mode, checks=PROOF.CHECKS[mode],
                                   failures=0 if variant == "native" else 1))
                for variant in ("native", "original") for mode in PROOF.MODES])

    def test_complete_bound_report_passes(self):
        """All modes, both polarities and the exact runtime identity are required."""
        PROOF.verify_report(self.report, self.library)

    def test_only_exact_current_repair_suffix_is_accepted(self):
        """A known later repair is required; stale or arbitrary suffixes are rejected."""
        current = self.report["rccl_receipt"]
        old = "fixture-protocol-" + PROOF.digest(ROOT / PROOF.PATCH)
        for receipt in (None, False, [], "", old, old + "-smallag-" + "0" * 64,
                        current + "-unknown-policy", current[:-1],
                        current.replace("-protocol-", "-other-")):
            with self.subTest(receipt=receipt):
                changed = copy.deepcopy(self.report)
                changed["rccl_receipt"] = receipt
                with self.assertRaisesRegex(RuntimeError, "canonical repair suffix"):
                    PROOF.verify_report(changed, self.library)

    def test_receipt_suffix_dependencies_are_rebuilt_and_packaged(self):
        """A policy edit invalidates both the build proof and its installed verifier."""
        self.assertIn(PROOF.SMALL_ALLGATHER_PATCH, PROOF.INPUTS)
        registration = (ROOT / "tests/v2/cmake/V2RCCLProtocolDefaultsTests.cmake").read_text()
        self.assertIn(PROOF.SMALL_ALLGATHER_PATCH, registration)
        for path, suffix in (("Dockerfile", "-smallag-${rccl_small_allgather_patch_sha}"),
                             ("src/v2/CMakeLists.txt", "-smallag-${RCCL_SMALL_ALLGATHER_PATCH_HASH}")):
            self.assertIn(suffix, (ROOT / path).read_text())

    def test_actual_canonical_producers_preserve_strict_repair_receipts(self):
        """Execute real CMake/Docker identity producers against unchanged installed consumers."""
        def fixture_module(name):
            spec = importlib.util.spec_from_file_location(name, ROOT / "tests/v2/unit/scripts" / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        thread = fixture_module("test_rccl_thread_lifetime")
        smallag = fixture_module("test_rccl_small_allgather_policy")
        thread_case = thread.RCCLThreadProofTests()
        thread_case.setUp()
        self.addCleanup(thread_case.doCleanups)
        smallag_case = smallag.NativeSmallAllGatherProofTests()
        smallag_case.setUp()
        self.addCleanup(smallag_case.doCleanups)

        def accepted(receipt):
            protocol_report = copy.deepcopy(self.report)
            protocol_report["rccl_receipt"] = receipt
            PROOF.verify_report(protocol_report, self.library)
            thread_report = copy.deepcopy(thread_case.report)
            thread_report["build_receipt"] = receipt
            thread.PROOF.verify_report(thread_report, thread_case.library)
            smallag_report = copy.deepcopy(smallag_case.report)
            smallag_report["rccl_receipt"] = receipt
            smallag.P.verify(smallag_report, smallag_case.library)

        pairs = [
            ("COMPLETED", "completed", "rccl-device-completed-broadcast.patch"),
            ("CAPTURE", "capture", "rccl-hip-capture-event-wait.patch"),
            ("ROWS", "rows", "rccl-device-live-rows.patch"),
            ("HOST_STORAGE", "storage", "rccl-native-host-storage.patch"),
            ("SHM", "shm", "rccl-partial-file-cleanup.patch"),
            ("SELECTOR", "selector", "rccl-collective-selector-taxonomy.patch"),
            ("THREAD", "thread", "rccl-created-thread-lifetime.patch"),
            ("PROTOCOL", "protocol", "rccl-communicator-protocol-defaults.patch"),
            ("SMALL_ALLGATHER", "small_allgather", "rccl-small-allgather-ll-policy.patch"),
        ]
        cmake = (ROOT / "src/v2/CMakeLists.txt").read_text()
        pin = re.search(r'set\(RCCL_GIT_REF "([^"]+)"\)', cmake).group(1)
        variables = {"RCCL_COMMIT": pin, "RCCL_GPU_TARGETS": "gfx906;gfx1100",
                     "RCCL_LTO_RECEIPT_HASH": "a" * 64, "RCCL_LTO_PRODUCER_HASH": "b" * 64}
        environment = dict(os.environ, RCCL_GIT_REF=pin, rccl_lto_receipt_sha="a" * 64,
                           rccl_lto_producer_sha="b" * 64,
                           rccl_target_hash=hashlib.sha256(b"gfx906;gfx1100").hexdigest())
        for upper, lower, patch in pairs:
            value = PROOF.digest(ROOT / "scripts/docker/patches" / patch)
            variables["RCCL_" + upper + "_PATCH_HASH"] = value
            environment["rccl_" + lower + "_patch_sha"] = value
        start = cmake.index('    set(RCCL_BUILD_ID "')
        end = cmake.index("    # Check if RCCL build is up-to-date", start)
        script = self.directory / "receipt.cmake"
        output = self.directory / "receipt.txt"
        script.write_text("".join('set(' + key + ' "' + value + '")\n' for key, value in variables.items())
                          + cmake[start:end] + '\nfile(WRITE "' + str(output) + '" "${RCCL_BUILD_ID}")\n')
        result = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        cmake_receipt = output.read_text()
        docker = (ROOT / "Dockerfile").read_text()
        producer = next(line.strip().removesuffix("\\").rstrip() for line in docker.splitlines()
                        if line.strip().startswith("printf '%s\\n' \"targets-${rccl_target_hash}"))
        result = subprocess.run(["bash", "-c", producer], env=environment,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        docker_receipt = result.stdout.strip()
        self.assertEqual(cmake_receipt, docker_receipt)
        for receipt in (cmake_receipt, docker_receipt):
            accepted(receipt)
            self.assertLess(receipt.index("-llvm-lto-"), receipt.index("-thread-"))
            self.assertLess(receipt.index("-producer-"), receipt.index("-thread-"))
            for label, patch in (("thread", thread.PROOF.PATCH), ("protocol", PROOF.PATCH),
                                 ("smallag", smallag.P.PATCH)):
                exact = "-" + label + "-" + PROOF.digest(ROOT / patch)
                for changed in (receipt.replace(exact, ""), receipt.replace(exact, "-" + label + "-" + "0" * 64)):
                    with self.subTest(label=label, changed=changed), self.assertRaises(RuntimeError):
                        accepted(changed)
            compiler_metadata = "-llvm-lto-" + "a" * 64 + "-producer-" + "b" * 64
            original_order = receipt.replace(compiler_metadata, "") + compiler_metadata
            with self.assertRaises(RuntimeError):
                accepted(original_order)

    def test_changed_runtime_fails(self):
        """Replacing the selected DSO invalidates earlier proof immediately."""
        self.library.write_bytes(b"different-runtime")
        with self.assertRaisesRegex(RuntimeError, "identity"):
            PROOF.verify_report(self.report, self.library)

    def test_each_changed_source_fails(self):
        """Changing the fixture, repair or validator invalidates earlier proof."""
        for name in PROOF.INPUTS:
            with self.subTest(source=name):
                changed = copy.deepcopy(self.report)
                changed["sources"][name] = "stale"
                with self.assertRaisesRegex(RuntimeError, "identity"):
                    PROOF.verify_report(changed, self.library)

    def test_missing_or_duplicate_outcome_fails(self):
        """Neither a missing race probe nor a duplicated pass completes coverage."""
        for index in range(6):
            with self.subTest(index=index):
                changed = copy.deepcopy(self.report)
                changed["observations"].pop(index)
                with self.assertRaisesRegex(RuntimeError, "incomplete or duplicate"):
                    PROOF.verify_report(changed, self.library)
        self.report["observations"].append(self.report["observations"][0])
        with self.assertRaisesRegex(RuntimeError, "incomplete or duplicate"):
            PROOF.verify_report(self.report, self.library)

    def test_failed_or_vacuous_native_probe_fails(self):
        """A native crash, skipped matrix or false successful exit cannot pass."""
        for code, checks, failures in ((8, 1080, 0), (0, 0, 0), (0, 1080, 1), (1, 1080, 0)):
            with self.subTest(code=code, checks=checks, failures=failures):
                changed = copy.deepcopy(self.report)
                row = changed["observations"][0]
                row["returncode"] = code
                row["measured"].update(checks=checks, failures=failures)
                with self.assertRaises(RuntimeError):
                    PROOF.verify_report(changed, self.library)

    def test_negative_control_must_fail_the_geometry_oracle(self):
        """An old-code pass or signal is not evidence that the regression detects it."""
        for code, failures in ((0, 0), (1, 0), (-8, 1), (1, -1), (1, 1081), (1, True)):
            with self.subTest(code=code, failures=failures):
                changed = copy.deepcopy(self.report)
                row = changed["observations"][3]
                row["returncode"] = code
                row["measured"]["failures"] = failures
                with self.assertRaisesRegex(RuntimeError, "unexpected outcome"):
                    PROOF.verify_report(changed, self.library)

    def test_missing_compiler_or_native_identity_fails(self):
        """Keep the source-compiled witness distinguishable from fabricated counters."""
        for field in ("compiler_version", "native_source_sha256", "native_arch_source_sha256"):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.report)
                changed.pop(field)
                with self.assertRaisesRegex(RuntimeError, "identity"):
                    PROOF.verify_report(changed, self.library)

    def test_compile_inputs_preserve_definitions_and_headers_without_shell_execution(self):
        """Read the native compile database as structured argument data only."""
        source = self.directory / "hipify/src/rccl_wrap.cc"
        source.parent.mkdir(parents=True)
        source.write_text("// fixture source\n")
        words = ["ccache", "hipcc", "-O3", "-DENABLE_LL128", "-I/path with spaces",
                 "-isystem", "/sdk include", "-D", "MACRO=value", "-c", str(source)]
        (self.directory / "compile_commands.json").write_text(json.dumps([
            dict(directory=str(self.directory), file=str(source), arguments=words)]))
        selected, inputs = PROOF.native_compile_inputs(self.directory)
        self.assertEqual(selected, source)
        self.assertEqual(inputs, ["-DENABLE_LL128", "-I/path with spaces", "-isystem",
                                  "/sdk include", "-D", "MACRO=value"])

    def test_ambiguous_or_missing_native_consumer_fails(self):
        """A different translation unit cannot silently supply the regression."""
        entry = dict(file="/build/hipify/src/rccl_wrap.cc", arguments=["hipcc", "-DTEST=1"])
        for rows in ([], [entry, entry]):
            (self.directory / "compile_commands.json").write_text(json.dumps(rows))
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                PROOF.native_compile_inputs(self.directory)

    def test_patch_ships_in_both_build_identities(self):
        """Docker/local rebuilds and installed preflight share one protocol repair."""
        patch = Path(PROOF.PATCH).name
        installer = (ROOT / "scripts/docker/apply-rccl-capture-patch.sh").read_text()
        docker = (ROOT / "Dockerfile").read_text()
        cmake = (ROOT / "src/v2/CMakeLists.txt").read_text()
        self.assertIn(patch, installer)
        self.assertIn("COPY scripts/docker/patches/" + patch, docker)
        self.assertIn("-protocol-${rccl_protocol_patch_sha}", docker)
        self.assertIn("-protocol-${RCCL_PROTOCOL_PATCH_HASH}", cmake)
        self.assertIn("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", docker)
        self.assertIn("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", cmake)
        registration = (ROOT / "tests/v2/cmake/V2RCCLProtocolDefaultsTests.cmake").read_text()
        self.assertIn("V2_Integration_RCCLProtocolDefaults", registration)
        self.assertIn("ProductionTestPreflight", registration)
        self.assertIn("V2_UNIT_GATE_TARGETS v2_rccl_protocol_geometry", registration)


if __name__ == "__main__":
    unittest.main()
