#!/usr/bin/env python3
"""Execute QFN source admission registration in Integration and Release.

The real target selector, registration helper and QFN declaration run in a
language-free CMake project. CTest metadata proves exact model dependencies
and backend scope without loading models, compiling code or probing devices.
Release must omit the tests and their properties together; missing admitted
targets and malformed source closures remain fatal. The numerical MTP filter
also preserves the main matrix's exact arithmetic counter families without
broad-domain collection or changing execution policy.
"""
from pathlib import Path
import argparse
import json
import re
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[4]
SELECTOR = SOURCE / "tests/v2/cmake/V2TestTargetSelection.cmake"


def registration_source() -> str:
    """Execute the canonical declarations, rather than a replica of their policy."""
    source = (SOURCE / "tests/v2/CMakeLists.txt").read_text()
    start = source.index("function(add_v2_test TEST_NAME)")
    end = source.index("endfunction()", start) + len("endfunction()")
    helper = source[start:end]
    start = source.index("    # QFN paired-source admission registration begins here.")
    end = source.index("    # Every above registration intentionally requires V2_Models.", start)
    return helper + "\n" + source[start:end]


class Qwen4ExpAdmissionRegistration(unittest.TestCase):
    """Prove target selection and source metadata share one registration owner."""

    cmake = "cmake"
    ninja = "ninja"

    def configure(self, release: bool, cuda: bool, rocm: bool, *,
                  declare_target: bool = True, main_shards: int = 4):
        """Generate CTest declarations with imported, never-executed placeholders."""
        with tempfile.TemporaryDirectory(prefix="llaminar-qfn-registration-") as directory:
            root = Path(directory)
            sources = [f"/not-loaded/main shard {index}.gguf" for index in range(main_shards)]
            source_list = " ".join(f'"{path}"' for path in sources)
            target = """
add_executable(v2_integration_qwen4exp_mtp_admission IMPORTED GLOBAL)
if(TARGET v2_integration_qwen4exp_mtp_admission)
    set_target_properties(v2_integration_qwen4exp_mtp_admission PROPERTIES
        IMPORTED_LOCATION "${CMAKE_COMMAND}")
endif()
""" if declare_target else ""
            (root / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.25)
project(Qwen4ExpAdmissionRegistration LANGUAGES NONE)
enable_testing()
set(CMAKE_BUILD_TYPE {"Release" if release else "Integration"})
set(V2_PERF_TESTS_ONLY {"ON" if release else "OFF"})
set(HAVE_CUDA {"ON" if cuda else "OFF"})
set(HAVE_ROCM {"ON" if rocm else "OFF"})
set(V2_TEST_SOCKETS 2)
set(V2_TEST_CORES_PER_SOCKET 2)
set(V2_PARITY_MODEL_QWEN_FLASH_NEXT {source_list})
include("{SELECTOR.as_posix()}")
{target}
{registration_source()}
if(TARGET v2_integration_qwen4exp_mtp_admission)
    get_target_property(scope v2_integration_qwen4exp_mtp_admission V2_TEST_DEVICE_SCOPE)
    if(NOT scope STREQUAL "DeclaredBackends")
        message(FATAL_ERROR "Lost QFN admission startup ownership")
    endif()
endif()
""")
            result = subprocess.run(
                [self.cmake, "-S", str(root), "-B", str(root / "build"),
                 "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={self.ninja}"],
                capture_output=True, text=True, timeout=10)
            if result.returncode:
                return result, None
            inventory = subprocess.run(
                [str(Path(self.cmake).with_name("ctest")), "--test-dir", str(root / "build"),
                 "--show-only=json-v1"], capture_output=True, text=True,
                check=True, timeout=10)
            return result, json.loads(inventory.stdout)["tests"]

    def test_backend_source_closures_and_release_selection(self):
        """Every enabled topology retains five exact sources; Release retains none."""
        for release in (False, True):
            for cuda, rocm in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(release=release, cuda=cuda, rocm=rocm):
                    result, tests = self.configure(release, cuda, rocm)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    if release:
                        self.assertEqual(tests, [])
                        continue
                    topologies = ["CPU0"]
                    if cuda:
                        topologies.append("CUDA0_CPU0_ExpertOverlay")
                    if rocm:
                        topologies.append("ROCm0_CPU0_ExpertOverlay")
                    if cuda and rocm:
                        topologies.append("CUDA0_ROCm0_CPU0_PP")
                    names = {
                        f"V2_Integration_Qwen4ExpMTPAdmission_{topology}_{storage}"
                        for topology in topologies for storage in ("RAM", "Mmap")
                    } | {"V2_Integration_Qwen4ExpMTPAdmission_ExplicitNoMTP"}
                    self.assertEqual({test["name"] for test in tests}, names)
                    self.assertEqual(len(tests), len(names))
                    for test in tests:
                        props = {item["name"]: item["value"] for item in test["properties"]}
                        expected = [f"/not-loaded/main shard {index}.gguf" for index in range(4)]
                        expected.append("/opt/llaminar-models/Qwen3.8-Flash-Next-GGUF/"
                                        "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf")
                        self.assertEqual(props["REQUIRED_FILES"], expected)
                        self.assertEqual(props["FIXTURES_REQUIRED"], ["V2_Models"])
                        self.assertEqual(props["TIMEOUT"], 900)
                        self.assertNotIn("ProductionTestPreflight", props["LABELS"])
                        self.assertEqual("CUDA" in props["LABELS"], "CUDA" in test["name"])
                        self.assertEqual("ROCm" in props["LABELS"], "ROCm" in test["name"])

    def test_mtp_arithmetic_filter_matches_main_without_broad_domains(self):
        """Paired MTP exports the same precise default-arithmetic witnesses as main."""
        source = (SOURCE / "tests/v2/CMakeLists.txt").read_text()

        def numerical_filter(target: str) -> tuple[str, ...]:
            """Read one exact literal from its actual numerical registration."""
            start = source.index(f"    discover_v2_parity_tests({target}\n")
            end = source.index("MODEL_FILES", start)
            filters = re.findall(r'PRODUCTION_PERF_STATS_FILTER "([^"\n]+)"', source[start:end])
            self.assertEqual(len(filters), 1, target)
            fields = tuple(filters[0].split(","))
            self.assertEqual(len(fields), len(set(fields)), target)
            return fields

        main = numerical_filter("v2_integration_parity_qwen4exp_production_matrix")
        mtp = numerical_filter("v2_integration_parity_qwen4exp_mtp_matrix")
        arithmetic = tuple(field for field in main if field.startswith(
            ("kernel.", "weight_loading.", "cuda_moe.", "rocm_moe.")))
        self.assertEqual(len(arithmetic), 11)
        self.assertTrue(set(arithmetic).issubset(mtp))
        # Preserve existing graph/prefix/movement/lookup witnesses. Main's
        # additional memory inventory is not needed for this narrow MTP export.
        self.assertEqual(mtp, tuple(field for field in main if field not in
            {"memory", "moe_overlay_capacity", "gpu_graph_inventory"}))
        for broad in ("kernel", "weight_loading", "cuda_moe", "rocm_moe"):
            self.assertNotIn(broad, mtp)

    def test_missing_admitted_target_fails(self):
        """Release exclusion cannot excuse an absent Integration executable."""
        result, tests = self.configure(False, False, False, declare_target=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(tests)
        self.assertIn("v2_integration_qwen4exp_mtp_admission", result.stderr)

    def test_incomplete_original_source_closure_fails(self):
        """Admission cannot register a partial main model or omit its sidecar."""
        result, tests = self.configure(False, True, True, main_shards=3)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(tests)
        self.assertIn("QFN sidecar admission requires original4main shards", result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--ninja", default="ninja")
    args, remaining = parser.parse_known_args()
    Qwen4ExpAdmissionRegistration.cmake = args.cmake
    Qwen4ExpAdmissionRegistration.ninja = args.ninja
    unittest.main(argv=[__file__, *remaining])
