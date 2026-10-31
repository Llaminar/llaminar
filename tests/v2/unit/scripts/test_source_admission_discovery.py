#!/usr/bin/env python3
"""Exercise setup-only source admission through the real CMake discovery path.

An executable emits only GTest inventory text; it never loads models, initializes
backends or runs a fixture. Actual generated CTest metadata must retain the
source closure, participant backends, MPI world and integration serialization,
while keeping numerical campaigns and their reference obligations separate.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[4]
DISCOVERY = SOURCE / "tests/v2/cmake/V2ParityTestDiscovery.cmake"


class SourceAdmissionDiscovery(unittest.TestCase):
    """Replay source-discovery inputs without SDK, model or hardware dependencies."""

    cmake = "cmake"
    ninja = "ninja"

    def discover(self, listing: str, *, sources: bool = True,
                 mtp_source_kind: str = "", predictor_files: tuple[str, ...] = (),
                 timeout: str | None = "30", source_timeout: str | None = None):
        """Run canonical discovery and parse the resulting real CTest properties."""
        with tempfile.TemporaryDirectory(prefix="llaminar-source-admission-") as directory:
            root = Path(directory)
            executable = root / "v2_integration_parity_admission_contract_matrix"
            executable.write_text(f"#!{sys.executable}\nprint({listing!r})\n")
            executable.chmod(0o755)
            generated = root / "discovered.cmake"
            command = [self.cmake, f"-DTEST_EXECUTABLE={executable}",
                       f"-DCTEST_FILE={generated}", "-DTEST_PREFIX=V2_AdmissionContract",
                       "-DLABELS=V2;Integration;Parity;CPU;CUDA;ROCm;PyTorch;GoldenReference;FullModel",
                       "-DMPI_PROCS=1", "-DNUM_SOCKETS=2", "-DCORES_PER_SOCKET=2",
                       f"-DWORKING_DIR={root}"]
            if timeout is not None:
                command.append(f"-DTIMEOUT={timeout}")
            if source_timeout is not None:
                command.append(f"-DSOURCE_ADMISSION_TIMEOUT={source_timeout}")
            common = ["/not-loaded/main 0.gguf", "/not-loaded/main 1.gguf"]
            backend_files = {backend: f"/not-loaded/{backend}.gguf"
                             for backend in ("CPU", "CUDA", "ROCm")}
            if sources:
                command.append("-DMODEL_FILES_SERIALIZED=" + "|".join(common))
                for backend, path in backend_files.items():
                    command.append(f"-D{backend.upper()}_MODEL_FILES_SERIALIZED={path}")
            command.append(f"-DSOURCE_ADMISSION_MTP_SOURCE_KIND={mtp_source_kind}")
            command.append("-DSOURCE_ADMISSION_MTP_MODEL_FILES_SERIALIZED="
                           + "|".join(predictor_files))
            command.extend(["-P", str(DISCOVERY)])
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            if result.returncode:
                return result, None
            (root / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.25)\n"
                "project(SourceAdmissionDiscovery LANGUAGES NONE)\n"
                "enable_testing()\n"
                f'include("{generated.as_posix()}")\n')
            configured = subprocess.run(
                [self.cmake, "-S", str(root), "-B", str(root / "build"),
                 "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={self.ninja}"],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
            inventory = subprocess.run(
                [str(Path(self.cmake).with_name("ctest")), "--test-dir", str(root / "build"),
                 "--show-only=json-v1"], capture_output=True, text=True, timeout=10, check=True)
            return result, json.loads(inventory.stdout)["tests"]

    def test_actual_participants_sources_and_world_are_preserved(self):
        """Every hardware signature owns exact sources and disabled foreign startup."""
        configurations = {
            "CPU0_NgramRAM": ({"CPU"}, 1),
            "CPU0_CPU1_PP_2xMPI_NgramMmap": ({"CPU"}, 2),
            "CUDA0_CUDA1_CPU0_TP2_NgramVRAM": ({"CPU", "CUDA"}, 1),
            "ROCm0_ROCm1_ROCm2_ROCm3_TP4_NgramRAM": ({"ROCm"}, 1),
            "CUDA0_ROCm0_CPU0_PP_NgramMmap": ({"CPU", "CUDA", "ROCm"}, 1),
        }
        listing = "SourceMatrix.\n" + "".join(
            f"  FullSourceAdmission/{name}\n" for name in configurations)
        result, tests = self.discover(listing)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(tests), len(configurations))
        for test in tests:
            name = test["name"].split("FullSourceAdmission_", 1)[1]
            backends, world = configurations[name]
            with self.subTest(name=name):
                props = {p["name"]: p["value"] for p in test["properties"]}
                self.assertEqual(set(props["LABELS"]) & {"CPU", "CUDA", "ROCm"}, backends)
                self.assertIn("Admission", props["LABELS"])
                self.assertIn("RealModel", props["LABELS"])
                self.assertFalse(set(props["LABELS"]) & {
                    "Parity", "PyTorch", "GoldenReference", "ProductionTestPreflight"})
                expected = {"/not-loaded/main 0.gguf", "/not-loaded/main 1.gguf"}
                expected.update(f"/not-loaded/{backend}.gguf" for backend in backends)
                self.assertEqual(set(props["REQUIRED_FILES"]), expected)
                self.assertEqual(props["FIXTURES_REQUIRED"], ["V2_Models"])
                self.assertEqual(props["RESOURCE_LOCK"], ["Integration_Serial"])
                self.assertTrue(props["RUN_SERIAL"])
                environment = dict(item.split("=", 1) for item in props["ENVIRONMENT"])
                self.assertEqual(environment["LLAMINAR_FORCE_CPU_ONLY_STARTUP"],
                                 "1" if backends == {"CPU"} else "0")
                for backend in ("CUDA", "ROCm"):
                    self.assertEqual(environment[f"LLAMINAR_SKIP_{backend.upper()}_STARTUP"],
                                     "0" if backend in backends else "1")
                self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", environment)
                if "ROCm" in backends:
                    self.assertIn("HSA_OVERRIDE_GFX_VERSION=unset:", props["ENVIRONMENT_MODIFICATION"])
                self.assertNotIn("LLAMINAR_PRODUCTION_PARITY_PROCESS_CAMPAIGN", environment)
                self.assertEqual(test["command"][test["command"].index("-np") + 1], str(world))
                self.assertEqual(test["command"][-1],
                                 "--gtest_filter=SourceMatrix.FullSourceAdmission/" + name)

    def test_reference_campaign_and_snapshot_contracts_remain_separate(self):
        """An N-prefixed next suite remains a suite and retains its numerical contract."""
        result, tests = self.discover(
            "SourceMatrix.\n  FullSourceAdmission/CPU0_NgramRAM\n"
            "NumericMatrix.\n  SnapshotAdmission/ROCm0_NgramRAM\n"
            "  ProductionParity/CUDA0_NgramRAM\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(tests), 3)
        self.assertIn("V2_AdmissionContract_NumericMatrix_SnapshotAdmission_ROCm0_NgramRAM",
                      {test["name"] for test in tests})
        for test in tests:
            props = {p["name"]: p["value"] for p in test["properties"]}
            if "FullSourceAdmission" in test["name"]:
                self.assertIn("Admission", props["LABELS"])
                self.assertNotIn("GoldenReference", props["LABELS"])
            else:
                self.assertNotIn("Admission", props["LABELS"])
                self.assertIn("GoldenReference", props["LABELS"])
                self.assertIn("PyTorch", props["LABELS"])

    def test_active_paired_product_keeps_off_and_numerical_closures(self):
        """All active typed suffixes add only predictor sources, across dispositions."""
        policies = ("MTPOff", "MTPDepth1", "MTPDepth2", "MTPDepth3",
                    "MTPDepth15", "MTPDynamicDepth")
        methods = ("FullSourceAdmission", "CandidateSourceAdmission",
                   "RejectUnsupportedSourceAdmission")
        listing = "SourceMatrix.\n" + "".join(
            f"  {method}/CPU0_NgramMmap_{policy}\n"
            for method in methods for policy in policies)
        listing += "NumericMatrix.\n  SnapshotAdmission/CPU0_NgramMmap_MTPDepth1\n"
        result, tests = self.discover(listing, mtp_source_kind="Paired",
                                      predictor_files=("/not-loaded/predictor.gguf",))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(tests), 19)
        for test in tests:
            props = {p["name"]: p["value"] for p in test["properties"]}
            source = "SourceMatrix" in test["name"]
            active = source and not test["name"].endswith("MTPOff")
            self.assertEqual("/not-loaded/predictor.gguf" in props.get("REQUIRED_FILES", []), active)
            environment = dict(item.split("=", 1) for item in props["ENVIRONMENT"])
            self.assertEqual("LLAMINAR_PERF_STATS_SUMMARY" in environment, active)
            self.assertEqual("LLAMINAR_PERF_STATS_FILTER" in environment, active)
            if active:
                self.assertEqual(environment["LLAMINAR_PERF_STATS_SUMMARY"], "1")
                self.assertEqual(environment["LLAMINAR_PERF_STATS_FILTER"], "forward_graph")
                self.assertEqual(sum(item.startswith("LLAMINAR_PERF_STATS_SUMMARY=")
                                     for item in props["ENVIRONMENT"]), 1)
                self.assertEqual(sum(item.startswith("LLAMINAR_PERF_STATS_FILTER=")
                                     for item in props["ENVIRONMENT"]), 1)
            if source:
                self.assertEqual(set(props["REQUIRED_FILES"]), {
                    "/not-loaded/main 0.gguf", "/not-loaded/main 1.gguf",
                    "/not-loaded/CPU.gguf"} | ({"/not-loaded/predictor.gguf"} if active else set()))
                self.assertFalse(set(props["LABELS"]) & {"Parity", "PyTorch", "GoldenReference"})
                self.assertEqual(props["FIXTURES_REQUIRED"], ["V2_Models"])
                self.assertEqual(props["RESOURCE_LOCK"], ["Integration_Serial"])
                self.assertTrue(props["RUN_SERIAL"])
                self.assertEqual("NegativeAdmission" in props["LABELS"],
                                 "RejectUnsupportedSourceAdmission" in test["name"])
                self.assertEqual("UnqualifiedCandidate" in props["LABELS"],
                                 "CandidateSourceAdmission" in test["name"])

    def test_unknown_source_mtp_policy_cannot_inherit_off_staging(self):
        """An unsupported policy must not silently lose its predictor dependency."""
        for suffix in ("MTPDepth5", "MTPDepth999", "MTPDynamicUnknown"):
            with self.subTest(suffix=suffix):
                result, tests = self.discover(
                    "SourceMatrix.\n  FullSourceAdmission/CPU0_" + suffix + "\n",
                    mtp_source_kind="Paired", predictor_files=("/not-loaded/predictor.gguf",))
                self.assertNotEqual(result.returncode, 0)
                self.assertIsNone(tests)
                self.assertIn("Unsupported source-admission MTP policy suffix", result.stderr)

    def test_embedded_active_mtp_needs_no_external_predictor(self):
        """An embedded predictor is explicit intent without an external GGUF obligation."""
        result, tests = self.discover(
            "SourceMatrix.\n  FullSourceAdmission/CPU0_MTPDynamicDepth\n",
            mtp_source_kind="Embedded")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        props = {p["name"]: p["value"] for p in tests[0]["properties"]}
        self.assertEqual(set(props["REQUIRED_FILES"]), {
            "/not-loaded/main 0.gguf", "/not-loaded/main 1.gguf", "/not-loaded/CPU.gguf"})

    def test_active_mtp_rejects_missing_or_invalid_source_intent(self):
        """Neither an unbound paired source nor an inferred embedded source is valid."""
        listing = "SourceMatrix.\n  FullSourceAdmission/CPU0_MTPDepth1\n"
        for kind, files, diagnostic in (
                ("", (), "requires explicit Embedded or Paired"),
                ("Paired", (), "no declared predictor inputs"),
                ("Unknown", (), "Invalid SOURCE_ADMISSION_MTP_SOURCE_KIND"),
                ("Embedded", ("/not-loaded/predictor.gguf",), "require Paired source kind"),
                ("", ("/not-loaded/predictor.gguf",), "require Paired source kind")):
            with self.subTest(kind=kind, files=files):
                result, tests = self.discover(listing, mtp_source_kind=kind,
                                              predictor_files=files)
                self.assertNotEqual(result.returncode, 0)
                self.assertIsNone(tests)
                self.assertIn(diagnostic, result.stderr)

    def test_source_deadline_is_explicit_without_changing_numerical_timeout(self):
        """Actual generated CTest metadata binds all source methods independently."""
        listing = ("SourceMatrix.\n  FullSourceAdmission/CPU0_MTPOff\n"
                   "  CandidateSourceAdmission/CPU0_MTPOff\n"
                   "  RejectUnsupportedSourceAdmission/CPU0_MTPOff\n"
                   "NumericMatrix.\n  SnapshotAdmission/CPU0_MTPOff\n")
        for ordinary in (None, "30"):
            with self.subTest(ordinary=ordinary):
                result, tests = self.discover(listing, timeout=ordinary, source_timeout="900")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(len(tests), 4)
                for test in tests:
                    props = {p["name"]: p["value"] for p in test["properties"]}
                    if "SourceMatrix" in test["name"]:
                        self.assertEqual(props["TIMEOUT"], 900)
                    elif ordinary is None:
                        self.assertNotIn("TIMEOUT", props)
                    else:
                        self.assertEqual(props["TIMEOUT"], 30)

    def test_source_deadline_missing_zero_negative_or_malformed_is_fatal(self):
        """Unbounded/invalid source setup is rejected instead of weakening admission."""
        for deadline in (None, "", "0", "-1", "NaN", "infinity", "not-seconds"):
            with self.subTest(deadline=deadline):
                result, tests = self.discover("SourceMatrix.\n  FullSourceAdmission/CPU0_MTPOff\n",
                                              timeout=None, source_timeout=deadline)
                self.assertNotEqual(result.returncode, 0)
                self.assertIsNone(tests)
                self.assertIn("requires a positive TIMEOUT or SOURCE_ADMISSION_TIMEOUT", " ".join(result.stderr.split()))

    def test_actual_target_timeout_participates_in_identity_and_post_build(self):
        """Execute the real argument parser/identity function without compiling a target."""
        source = (SOURCE / "tests/v2/CMakeLists.txt").read_text()
        begin = source.index("function(discover_v2_parity_tests TARGET)")
        end = source.index("endfunction()", begin) + len("endfunction()")
        helper = source[begin:end]
        begin = source.index("    discover_v2_parity_tests(v2_integration_parity_qwen4exp_production_matrix")
        end = source.index("\n    )", begin) + len("\n    )")
        declaration = source[begin:end]
        with tempfile.TemporaryDirectory(prefix="llaminar-source-deadline-identity-") as temporary:
            root = Path(temporary)
            def configure(declared):
                (root / "CMakeLists.txt").write_text(
                    "cmake_minimum_required(VERSION 3.25)\n"
                    "project(SourceDeadlineIdentity LANGUAGES NONE)\n"
                    "enable_testing()\n"
                    "add_custom_target(v2_integration_parity_qwen4exp_production_matrix)\n"
                    # Compilation and command scheduling are not the policy under
                    # test. Capture the actual POST_BUILD argv without a build.
                    "function(target_compile_definitions)\nendfunction()\n"
                    "function(add_custom_command)\n"
                    'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/post-build.txt" "${ARGV}")\n'
                    "endfunction()\n" + helper + "\n" + declared + "\n"
                    'get_target_property(deps v2_integration_parity_qwen4exp_production_matrix LINK_DEPENDS)\n'
                    'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dependencies.txt" "${deps}")\n')
                result = subprocess.run([self.cmake, "-S", str(root), "-B", str(root / "build"),
                                         "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={self.ninja}"],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return root / "build"
            build = configure(declaration)
            name = "v2_integration_parity_qwen4exp_production_matrix_discovery_identity.txt"
            identity = (build / name).read_text()
            self.assertIn("source_admission_timeout=900\n", identity)
            self.assertIn("SOURCE_ADMISSION_TIMEOUT=900", (build / "post-build.txt").read_text())
            self.assertIn(str(build / name), (build / "dependencies.txt").read_text())
            changed = declaration.replace("SOURCE_ADMISSION_TIMEOUT 900", "SOURCE_ADMISSION_TIMEOUT 901")
            self.assertNotEqual(changed, declaration)
            configure(changed)
            self.assertNotEqual((build / name).read_text(), identity)
            self.assertIn("source_admission_timeout=901\n", (build / name).read_text())
            self.assertIn("SOURCE_ADMISSION_TIMEOUT=901", (build / "post-build.txt").read_text())

    def test_capacity_rejection_is_distinct_and_keeps_typed_source_staging(self):
        """Capacity controls remain real native source contracts, not unsupported skips."""
        policies = ("MTPOff", "MTPDepth1", "MTPDepth3", "MTPDynamicDepth")
        listing = "CapacityMatrix.\n" + "".join(
            f"  RejectCapacitySourceAdmission/CUDA0_CUDA1_CPU0_Capacity13GiB_{policy}\n"
            for policy in policies)
        result, tests = self.discover(listing, mtp_source_kind="Paired",
            predictor_files=("/not-loaded/predictor.gguf",), timeout=None, source_timeout="900")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(tests), 4)
        for test in tests:
            props = {p["name"]: p["value"] for p in test["properties"]}
            self.assertTrue({"Admission", "NegativeAdmission", "CapacityAdmission", "CPU", "CUDA"} <= set(props["LABELS"]))
            self.assertFalse(set(props["LABELS"]) & {"Parity", "PyTorch", "GoldenReference", "UnqualifiedCandidate", "ROCm"})
            self.assertEqual(props["TIMEOUT"], 900)
            self.assertEqual(props["FIXTURES_REQUIRED"], ["V2_Models"])
            self.assertEqual(props["RESOURCE_LOCK"], ["Integration_Serial"])
            self.assertTrue(props["RUN_SERIAL"])
            self.assertNotIn("WILL_FAIL", props)
            self.assertNotIn("DISABLED", props)
            self.assertEqual("/not-loaded/predictor.gguf" in props["REQUIRED_FILES"],
                             not test["name"].endswith("MTPOff"))

    def test_missing_source_closure_fails_discovery(self):
        """A model fixture cannot publish an unbound source-admission test."""
        result, tests = self.discover("SourceMatrix.\n  FullSourceAdmission/CPU0\n", sources=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(tests)
        self.assertIn("has no declared MODEL_FILES", result.stderr)

    def test_rejected_source_request_keeps_its_exact_contract(self):
        """Expected rejection is explicit, source-bound and isolated from numerical campaigns."""
        result, tests = self.discover(
            "RejectedMatrix.\n  RejectUnsupportedSourceAdmission/ROCm0_ROCm1_CPU0_TP2_NgramRAM\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(tests), 1)
        props = {p["name"]: p["value"] for p in tests[0]["properties"]}
        self.assertIn("NegativeAdmission", props["LABELS"])
        self.assertIn("Admission", props["LABELS"])
        self.assertFalse(set(props["LABELS"]) & {"Parity", "PyTorch", "GoldenReference", "CUDA"})
        self.assertEqual(set(props["REQUIRED_FILES"]), {
            "/not-loaded/main 0.gguf", "/not-loaded/main 1.gguf",
            "/not-loaded/CPU.gguf", "/not-loaded/ROCm.gguf"})
        environment = dict(item.split("=", 1) for item in props["ENVIRONMENT"])
        self.assertEqual(environment["LLAMINAR_SKIP_CUDA_STARTUP"], "1")
        self.assertEqual(environment["LLAMINAR_SKIP_ROCM_STARTUP"], "0")
        self.assertEqual(tests[0]["command"][-1],
                         "--gtest_filter=RejectedMatrix.RejectUnsupportedSourceAdmission/ROCm0_ROCm1_CPU0_TP2_NgramRAM")

    def test_missing_backend_identity_fails_discovery(self):
        """Unknown participant ownership cannot acquire a generic startup policy."""
        result, tests = self.discover("SourceMatrix.\n  FullSourceAdmission/UnknownDevice\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(tests)
        self.assertIn("has no backend identity", result.stderr)

    def test_unqualified_candidate_keeps_positive_native_obligation(self):
        """Candidate labels cannot turn unavailable composition into a passing rejection."""
        result, tests = self.discover(
            "CandidateMatrix.\n  CandidateSourceAdmission/CUDA0_CUDA1_CPU0_CPU1_TP2PP2_NgramMmap\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(tests), 1)
        props = {p["name"]: p["value"] for p in tests[0]["properties"]}
        self.assertTrue({"Admission", "UnqualifiedCandidate", "FullModel", "RealModel"}
                        <= set(props["LABELS"]))
        self.assertFalse(set(props["LABELS"]) & {
            "NegativeAdmission", "Parity", "PyTorch", "GoldenReference", "ROCm",
            "ProductionTestPreflight"})
        self.assertEqual(set(props["REQUIRED_FILES"]), {
            "/not-loaded/main 0.gguf", "/not-loaded/main 1.gguf",
            "/not-loaded/CPU.gguf", "/not-loaded/CUDA.gguf"})
        self.assertEqual(props["RESOURCE_LOCK"], ["Integration_Serial"])
        self.assertTrue(props["RUN_SERIAL"])
        self.assertNotIn("WILL_FAIL", props)
        self.assertNotIn("DISABLED", props)
        environment = dict(item.split("=", 1) for item in props["ENVIRONMENT"])
        self.assertEqual(environment["LLAMINAR_SKIP_CUDA_STARTUP"], "0")
        self.assertEqual(environment["LLAMINAR_SKIP_ROCM_STARTUP"], "1")
        self.assertEqual(tests[0]["command"][-1],
                         "--gtest_filter=CandidateMatrix.CandidateSourceAdmission/"
                         "CUDA0_CUDA1_CPU0_CPU1_TP2PP2_NgramMmap")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--ninja", default="ninja")
    args, remaining = parser.parse_known_args()
    SourceAdmissionDiscovery.cmake = args.cmake
    SourceAdmissionDiscovery.ninja = args.ninja
    unittest.main(argv=[__file__, *remaining])
