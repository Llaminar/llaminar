#!/usr/bin/env python3
"""Exercise the real CMake test-target selector without a model or GPU.

Release excludes Unit executables, including standalone host/device arithmetic
contracts that declare their own language level. Their property declarations
must be excluded with them, including startup device ownership, while admitted targets retain
every requirement and CMake must still reject invalid features. Native HTTP
affinity registrations follow the same selection and preserve their exact
OpenMP startup policy. These probes configure tiny CPU-only projects; they
never build or run an inference or test executable. QFN performance declarations
also remain available in Release without admitting functional test binaries.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[2] / "cmake/V2TestTargetSelection.cmake"
HTTP_REGISTRATION = MODULE.with_name("V2HttpServiceThreadAffinityTests.cmake")
QFN_PERFORMANCE = MODULE.with_name("V2Qwen4ExpPerformanceTargets.cmake")
CMAKE_LISTS = MODULE.parent.parent / "CMakeLists.txt"
OWNED_REGISTRATION = MODULE.with_name("OwnedSharedBackingTests.cmake")
COHORT_REGISTRATION = MODULE.with_name("V2PooledCompleteCohortTests.cmake")
HOST_SERVICE_REGISTRATION = MODULE.with_name("V2RetainedHostServiceTests.cmake")


class TestTargetSelection(unittest.TestCase):
    """Verify selection and requirement propagation using CMake itself."""

    cmake = "cmake"
    ninja = "ninja"

    def configure(self, performance_only: bool, feature: str = "cxx_std_20", *,
                  scope: str = "FullInventory", extra: str = "", native_targets: tuple[str, ...] = ()):
        """Configure an isolated fixture with the selected production module."""
        with tempfile.TemporaryDirectory(prefix="llaminar-test-targets-") as tmp:
            root = Path(tmp)
            (root / "main.cpp").write_text("int main() { return 0; }\n")
            for relative in ("performance", "unit", "integration", "fixtures", "scripts"):
                (root / relative).symlink_to(MODULE.parent.parent / relative,
                                            target_is_directory=True)
            # Reuse the module, not copied wrappers or source-string assertions.
            # Include twice to exercise its guard against nested macro wrapping.
            (root / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.27)
project(TestTargetSelection LANGUAGES CXX)
enable_testing()
set(LLAMINAR_RELEASE_NATIVE_TEST_TARGETS "{";".join(native_targets)}" CACHE STRING "" FORCE)
set(V2_PERF_TESTS_ONLY {"ON" if performance_only else "OFF"})
# The production guardian exists before the Release test-target selector.
add_executable(llaminar_mpi_session IMPORTED)
set_target_properties(llaminar_mpi_session PROPERTIES IMPORTED_LOCATION "${{CMAKE_COMMAND}}")
include("{MODULE.as_posix()}")
include("{MODULE.as_posix()}")
add_library(requirements INTERFACE)
foreach(target v2_test_contract v2_integration_contract v2_perf_contract)
    add_executable(${{target}} main.cpp)
    target_link_libraries(${{target}} PRIVATE requirements)
    target_include_directories(${{target}} PRIVATE "${{CMAKE_CURRENT_SOURCE_DIR}}")
    target_compile_definitions(${{target}} PRIVATE CONTRACT_CHECK=1)
    target_compile_options(${{target}} PRIVATE -Wall)
    target_compile_features(${{target}} PRIVATE {feature})
    v2_test_device_scope(${{target}} {scope})
    if(V2_PERF_TESTS_ONLY AND NOT target MATCHES "^v2_perf_" AND
       NOT target IN_LIST LLAMINAR_RELEASE_NATIVE_TEST_TARGETS)
        if(TARGET ${{target}})
            message(FATAL_ERROR "Excluded target was materialized: ${{target}}")
        endif()
    else()
        if(NOT TARGET ${{target}})
            message(FATAL_ERROR "Admitted target is missing: ${{target}}")
        endif()
        foreach(property LINK_LIBRARIES INCLUDE_DIRECTORIES COMPILE_DEFINITIONS
                         COMPILE_OPTIONS COMPILE_FEATURES V2_TEST_DEVICE_SCOPE)
            get_target_property(value ${{target}} ${{property}})
            if(NOT value)
                message(FATAL_ERROR "Lost ${{property}} for ${{target}}")
            endif()
        endforeach()
        get_target_property(features ${{target}} COMPILE_FEATURES)
        get_target_property(device_scope ${{target}} V2_TEST_DEVICE_SCOPE)
        if(NOT device_scope STREQUAL "{scope}")
            message(FATAL_ERROR "Lost startup device ownership for ${{target}}")
        endif()
        if(NOT "{feature}" IN_LIST features)
            message(FATAL_ERROR "Lost language contract for ${{target}}")
        endif()
    endif()
endforeach()
{extra}
''', encoding="utf-8")
            configured = subprocess.run(
                [self.cmake, "-S", str(root), "-B", str(root / "build"),
                 "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={self.ninja}"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            ninja_file = root / "build/build.ninja"
            configured.ninja_text = ninja_file.read_text() if ninja_file.exists() else ""
            return configured

    def test_explicit_release_native_owner_and_aliases(self):
        """Original owners and script aliases opt in without entering default all."""
        declaration = CMAKE_LISTS.read_text(encoding="utf-8")
        first = declaration.index("function(add_v2_test TEST_NAME)")
        stop = declaration.index("endfunction()", first) + len("endfunction()")
        helper = declaration[first:stop]
        for performance_only, native_targets in ((True, ()),
                (True, ("v2_integration_contract",)), (False, ())):
            with self.subTest(release=performance_only, selected=native_targets):
                extra = helper + f"""
set(V2_TEST_SOCKETS 1)
set(V2_TEST_CORES_PER_SOCKET 1)
set(CMAKE_BUILD_TYPE Release)
add_v2_test(V2_Perf_Contract COMMAND $<TARGET_FILE:v2_perf_contract>
    LABELS V2 Performance MPI_PROCS 1 NO_MPI NO_MODELS)
add_v2_test(V2_Integration_Contract COMMAND $<TARGET_FILE:v2_integration_contract>
    LABELS V2 Integration MPI_PROCS 1 NO_MPI NO_MODELS)
add_v2_test(V2_Integration_ScriptContract COMMAND "${{CMAKE_COMMAND}}" -E echo
    $<TARGET_FILE:v2_integration_contract>
    LABELS V2 Integration MPI_PROCS 1 NO_MPI NO_MODELS)
add_v2_test(V2_Integration_Unselected COMMAND $<TARGET_FILE:v2_test_contract>
    LABELS V2 Integration MPI_PROCS 1 NO_MPI NO_MODELS)
if(NOT TEST V2_Perf_Contract)
    message(FATAL_ERROR "Lost original performance alias")
endif()
foreach(alias V2_Integration_Contract V2_Integration_ScriptContract)
    if({"TRUE" if not performance_only or native_targets else "FALSE"})
        if(NOT TEST ${{alias}})
            message(FATAL_ERROR "Selected native alias missing: ${{alias}}")
        endif()
        get_property(owners GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_${{alias}})
        if(NOT "v2_integration_contract" IN_LIST owners OR
           NOT "llaminar_mpi_session" IN_LIST owners)
            message(FATAL_ERROR "Alias lost original owner/dependency: ${{alias}}")
        endif()
    elseif(TEST ${{alias}})
        message(FATAL_ERROR "Unselected native alias admitted: ${{alias}}")
    endif()
endforeach()
if(V2_PERF_TESTS_ONLY)
    if(TEST V2_Integration_Unselected)
        message(FATAL_ERROR "Foreign native alias admitted")
    endif()
    if(TARGET v2_integration_contract)
        get_target_property(excluded v2_integration_contract EXCLUDE_FROM_ALL)
        if(NOT excluded)
            message(FATAL_ERROR "Explicit native owner entered default all")
        endif()
        # A dependency-only qualification target must not widen default all.
        add_custom_target(qualify_native DEPENDS v2_integration_contract)
    endif()
endif()
"""
                result = self.configure(performance_only, scope="DeclaredBackends",
                                        native_targets=native_targets, extra=extra)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                if performance_only:
                    default_all = re.search(r"^build all:.*$", result.ninja_text, re.MULTILINE)
                    self.assertIsNotNone(default_all)
                    self.assertNotIn("v2_integration_contract", default_all.group())
                    self.assertNotIn("qualify_native", default_all.group())

    def test_explicit_release_native_alias_reachability(self):
        """Execute original selected-owner aliases and the script-only replay."""
        declaration = CMAKE_LISTS.read_text(encoding="utf-8")
        first = declaration.index("function(add_v2_test TEST_NAME)")
        stop = declaration.index("endfunction()", first) + len("endfunction()")
        helper = declaration[first:stop]
        first = declaration.index("# Native hardware identity and dependency target coverage")
        stop = declaration.index("# Optional dependency discovery", first)
        script_replay = declaration[first:stop]
        owners = ("v2_unit_rccl_rows_abi", "v2_unit_rccl_path_configured",
                  "v2_integration_captured_transfer_channel")
        aliases = {
            owners[0]: ("V2_Integration_RCCLNativeByteBroadcastABI",),
            owners[1]: ("V2_Integration_RCCLNativeByteMissingABINoFallback",
                        "V2_Integration_RCCLCompletedSemanticProtocolReplay"),
            owners[2]: ("V2_Integration_ROCmNativeCompletedCollective_WireReplay",
                        "V2_Integration_ROCmNativeCompletedCollective_ReaderAckReplay"),
        }
        for selected in ((), (owners[0],), (owners[1],), (owners[2],), owners):
            with self.subTest(selected=selected):
                checks = []
                for owner, names in aliases.items():
                    for name in names:
                        checks.append(f"if({'NOT ' if owner in selected else ''}TEST {name})\n"
                                      f'  message(FATAL_ERROR "Wrong original alias selection: {name}")\nendif()')
                checks_text = "\n".join(checks)
                extra = helper + f"""
set(Python3_EXECUTABLE "{sys.executable}")
set(V2_TEST_SOCKETS 1)
set(V2_TEST_CORES_PER_SOCKET 1)
set(CMAKE_BUILD_TYPE Release)
set(HAVE_ROCM ON)
set(HAVE_RCCL ON)
set(HAVE_CUDA OFF)
set(HAVE_NCCL OFF)
foreach(owner IN ITEMS {" ".join(owners)})
    add_executable(${{owner}} main.cpp)
endforeach()
include("{MODULE.with_name('V2NativeCountedBroadcastTests.cmake').as_posix()}")
{script_replay}
{checks_text}
if(NOT TEST V2_Integration_ROCmArchitectureIdentityReplay)
    message(FATAL_ERROR "Lost script-only architecture replay in Release")
endif()
get_test_property(V2_Integration_ROCmArchitectureIdentityReplay LABELS script_labels)
get_test_property(V2_Integration_ROCmArchitectureIdentityReplay TIMEOUT script_timeout)
if(NOT "ProductionTestPreflight" IN_LIST script_labels OR
   NOT "DeviceFree" IN_LIST script_labels OR NOT script_timeout EQUAL 30)
    message(FATAL_ERROR "Script replay lost original preflight contract")
endif()
foreach(foreign IN ITEMS V2_Integration_NCCLNativeByteBroadcastABI
        V2_Integration_NCCLNativeByteMissingABINoFallback
        V2_Integration_NativeCompleted_ROCm_Addressed2Source0Replay)
    if(TEST ${{foreign}})
        message(FATAL_ERROR "Unselected native alias admitted: ${{foreign}}")
    endif()
endforeach()
"""
                result = self.configure(True, scope="DeclaredBackends",
                                        native_targets=selected, extra=extra)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_explicit_release_rccl_proof_owners(self):
        """Configure actual proof modules in fresh language-free Release fixtures."""
        repository = CMAKE_LISTS.parents[2]
        production = (repository / "src/v2/CMakeLists.txt").read_text()
        first = production.index("    # Shared proof authority is published")
        stop = production.index("    # End shared RCCL proof authority.", first)
        authority = production[first:stop]
        modules = (
            ("V2RCCLProtocolDefaultsTests.cmake", "v2_rccl_protocol_geometry", "RCCLProtocolDefaults"),
            ("V2RCCLSelectorTests.cmake", "v2_rccl_selector_taxonomy", "RCCLCollectiveSelectors"),
            ("V2RCCLThreadLifetimeTests.cmake", "v2_rccl_thread_lifetime_proof", "RCCLThreadLifetimeReplay"),
            ("V2RCCLSmallAllGatherTests.cmake", "v2_rccl_small_allgather_policy", "RCCLSmallAllGatherPolicy"),
        )
        proof_targets = tuple(row[1] for row in modules) + (
            "v2_rccl_completed_broadcast_proof", "v2_rccl_final_allocation_proof")
        cases = (("none", (), False, ""), ("all", proof_targets, False, ""),
                 ("reordered", proof_targets, True, ""),
                 ("one", ("v2_rccl_selector_taxonomy",), False, ""),
                 ("unknown", ("v2_missing_proof",), False, ""),
                 ("missing_batch", ("v2_missing_proof_a", "v2_missing_proof_b"), False, ""),
                 ("unmarked", ("ordinary_custom",), False, "add_custom_target(ordinary_custom)"))
        for name, selected, reverse, extra in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory(prefix="llaminar-rccl-proof-owner-") as tmp:
                root = Path(tmp)
                source = root / "project"
                source.mkdir()
                for relative in ("unit", "integration"):
                    (source / relative).symlink_to(CMAKE_LISTS.parent / relative, target_is_directory=True)
                native = root / "canonical-rccl"
                (native / "src").mkdir(parents=True)
                build = native / "build"
                build.mkdir()
                (build / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=" + str(native) + "\n")
                library = build / "librccl.so.1.0"
                library.write_text("inert configuration-only library; never loaded")
                order = list(reversed(modules)) if reverse else list(modules)
                included = "\n".join(f'include("{MODULE.with_name(row[0]).as_posix()}")' for row in order)
                # The consumers run FIRST, before any selected/unselected module
                # can publish RCCL_PROTOCOL_BUILD_DIR as an accidental side effect.
                (source / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.27)
project(RCCLProofOwners NONE)
enable_testing()
set(CMAKE_BUILD_TYPE Release)
set(V2_PERF_TESTS_ONLY ON)
set(LLAMINAR_RELEASE_NATIVE_TEST_TARGETS "{";".join(selected)}" CACHE STRING "" FORCE)
include("{MODULE.as_posix()}")
set(REPO_ROOT "{repository.as_posix()}")
set(HAVE_ROCM ON)
set(HAVE_RCCL ON)
set(HAVE_CUDA OFF)
set(CMAKE_HIP_ARCHITECTURES gfx906 gfx1100)
set(CMAKE_HIP_COMPILER "${{CMAKE_COMMAND}}")
set(CMAKE_CXX_COMPILER "${{CMAKE_COMMAND}}")
set(Python3_EXECUTABLE "{sys.executable}")
set(ROCM_PATH "{(root / 'authored-sdk').as_posix()}")
set(RCCL_BUILD_DIR "{build.as_posix()}")
set(RCCL_LIBRARY "{library.as_posix()}")
{authority}
if(NOT RCCL_PROTOCOL_BUILD_DIR STREQUAL RCCL_BUILD_DIR)
    message(FATAL_ERROR "Canonical proof authority was not published before consumers")
endif()
include("{MODULE.with_name('V2NativeCompletedBroadcastProofTests.cmake').as_posix()}")
include("{MODULE.with_name('V2ROCmLTOArtifactTests.cmake').as_posix()}")
{included}
{extra}
get_directory_property(proof_tests TESTS)
file(WRITE "${{CMAKE_BINARY_DIR}}/tests.txt" "${{proof_tests}}")
file(WRITE "${{CMAKE_BINARY_DIR}}/owners.txt" "")
foreach(proof IN ITEMS {" ".join(proof_targets)})
    if(TARGET ${{proof}})
        get_target_property(kind ${{proof}} TYPE)
        get_target_property(owner ${{proof}} V2_RELEASE_NATIVE_PROOF_OWNER)
        file(APPEND "${{CMAKE_BINARY_DIR}}/owners.txt" "${{proof}}|${{kind}}|${{owner}}\n")
    endif()
endforeach()
""")
                output = root / "output"
                result = subprocess.run([self.cmake, "-S", str(source), "-B", str(output),
                    "-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=" + self.ninja],
                    capture_output=True, text=True, timeout=15, check=False)
                if name in ("unknown", "missing_batch", "unmarked"):
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("was not declared" if name != "unmarked" else "not an original declared test executable",
                                  " ".join((result.stdout + result.stderr).split()))
                    if name == "missing_batch":
                        self.assertIn("v2_missing_proof_a", result.stdout + result.stderr)
                        self.assertIn("v2_missing_proof_b", result.stdout + result.stderr)
                    continue
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                owners = (output / "owners.txt").read_text().splitlines()
                tests = set((output / "tests.txt").read_text().split(";"))
                for _, target, alias in modules:
                    expected = target in selected
                    self.assertEqual(any(row.startswith(target + "|") for row in owners), expected)
                    for suite in ("Unit", "Integration"):
                        self.assertEqual("V2_" + suite + "_" + alias in tests, expected)
                for target in selected:
                    self.assertIn(target + "|UTILITY|TRUE", owners)
                default_all = re.search(r"^build all:.*$", (output / "build.ninja").read_text(), re.MULTILINE)
                self.assertIsNotNone(default_all)
                for target in proof_targets:
                    self.assertNotIn(target, default_all.group())

    def test_explicit_release_native_owner_rejects_unknown_or_foreign_targets(self):
        """Unknown, imported, alias and non-test owners cannot masquerade as opt-ins."""
        cases = (
            ("v2_unknown", "", "was not declared"),
            ("requirements", "", "not an original declared test executable"),
            ("v2_imported", 'add_executable(v2_imported IMPORTED)',
             "not an original declared test executable"),
            ("v2_alias", 'add_executable(v2_alias ALIAS v2_perf_contract)',
             "not an original declared test executable"),
            ("llaminar2", '_add_executable(llaminar2 main.cpp)',
             "not an original declared test executable"),
        )
        for target, extra, diagnostic in cases:
            with self.subTest(target=target):
                result = self.configure(True, native_targets=(target,), extra=extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, " ".join((result.stdout + result.stderr).split()))

    def test_finite_storage_cases_keep_original_aggregate_and_owned_processes(self):
        """Actual declarations preserve every old suite and isolate all new source admissions."""
        declaration = CMAKE_LISTS.read_text(encoding="utf-8")
        first = declaration.index("function(add_v2_test TEST_NAME)")
        stop = declaration.index("endfunction()", first) + len("endfunction()")
        helper = declaration[first:stop]
        first = declaration.index("    add_v2_test(V2_Integration_GPUExpertTransfer")
        aggregate = declaration[first:declaration.index("    if(HAVE_CUDA)", first)]
        owner = MODULE.with_name("V2FiniteStorageParentTests.cmake")
        replay = CMAKE_LISTS.parent / "integration/transfer/FiniteBodySiblingNativeReplay.inc"
        new_cases = re.findall(r"TEST\(FiniteBodySiblingNativeIntegration,\s*(\w+)\)", replay.read_text())
        self.assertEqual(len(new_cases), 10)
        self.assertEqual(len(set(new_cases)), 10)
        source_first = declaration.index("set(V2_GPU_EXPERT_TRANSFER_TEST_SOURCES")
        source_stop = declaration.index("    add_executable(v2_unit_gpu_expert_transfer", source_first)
        old_sources = re.findall(r"(?:unit|integration)/[\w/.-]+[.]cpp", declaration[source_first:source_stop])
        old_suites = set()
        for relative in old_sources:
            old_suites.update(re.findall(r"TEST(?:_F|_P)?\(\s*(\w+)\s*,", (CMAKE_LISTS.parent / relative).read_text()))
        self.assertTrue(old_suites)
        self.assertNotIn("FiniteBodySiblingNativeIntegration", old_suites)
        for cuda, rocm in ((True, False), (False, True), (True, True)):
            with self.subTest(cuda=cuda, rocm=rocm), tempfile.TemporaryDirectory(prefix="llaminar-finite-selection-") as tmp:
                root = Path(tmp)
                for relative in ("unit", "integration", "fixtures"):
                    (root / relative).symlink_to(CMAKE_LISTS.parent / relative, target_is_directory=True)
                (root / "main.cpp").write_text("int main(){return 0;}\n")
                fixture = f"""
cmake_minimum_required(VERSION 3.27)
project(FiniteStorageSelection LANGUAGES CXX)
enable_testing()
set(V2_PERF_TESTS_ONLY OFF)
set(HAVE_CUDA {"ON" if cuda else "OFF"})
set(HAVE_ROCM {"ON" if rocm else "OFF"})
set(V2_TEST_SOCKETS 1)
set(V2_TEST_CORES_PER_SOCKET 1)
set(REPO_ROOT "{MODULE.parents[3].as_posix()}")
set(PROJECT_ROOT "${{REPO_ROOT}}")
set(MPIEXEC_EXECUTABLE "${{CMAKE_COMMAND}}")
set(Python3_EXECUTABLE "{sys.executable}")
add_library(llaminar2_core INTERFACE)
add_library(llaminar_mpi_session_owner INTERFACE)
add_library(GTest::gtest INTERFACE IMPORTED)
add_library(GTest::gtest_main INTERFACE IMPORTED)
add_executable(llaminar_mpi_session IMPORTED)
set_target_properties(llaminar_mpi_session PROPERTIES IMPORTED_LOCATION "${{CMAKE_COMMAND}}")
include("{MODULE.as_posix()}")
{helper}
foreach(target v2_unit_gpu_expert_transfer v2_test_compute_graph_execution)
    add_executable(${{target}} main.cpp)
    # CTest metadata requires an existing command. This stand-in is NEVER run;
    # actual native fixture execution remains a separate GPU qualification.
    file(CREATE_LINK "${{CMAKE_COMMAND}}" "${{CMAKE_CURRENT_BINARY_DIR}}/${{target}}" SYMBOLIC)
endforeach()
{aggregate}
include("{owner.as_posix()}")
"""
                (root / "CMakeLists.txt").write_text(fixture)
                built = subprocess.run([self.cmake, "-S", str(root), "-B", str(root / "build"),
                                        "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={self.ninja}"],
                                       capture_output=True, text=True, timeout=15, check=False)
                self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                inventory = subprocess.run([str(Path(self.cmake).with_name("ctest")), "--test-dir", str(root / "build"),
                                            "--show-only=json-v1"], capture_output=True, text=True, timeout=15, check=False)
                self.assertEqual(inventory.returncode, 0, inventory.stderr)
                tests = {entry["name"]: entry for entry in json.loads(inventory.stdout)["tests"]}
                old = tests["V2_Integration_GPUExpertTransfer"]["command"]
                filters = [arg for arg in old if arg.startswith("--gtest_filter=")]
                self.assertEqual(filters, ["--gtest_filter=-FiniteBodySiblingNativeIntegration.*"])
                # The ONLY exclusion is the newly introduced namespace; every
                # source-owned original suite remains admitted by that filter.
                for suite in old_suites:
                    self.assertFalse(suite.startswith("FiniteBodySiblingNativeIntegration"))
                expected = {case for case in new_cases if (cuda and case.startswith("CUDA_")) or
                            (rocm and case.startswith("ROCm_"))}
                native = {name: entry for name, entry in tests.items() if re.match(r"V2_Integration_(CUDA|ROCm)_FiniteStorage_", name)}
                self.assertEqual(len(native), len(expected))
                actual = set()
                for entry in native.values():
                    command = entry["command"]
                    filters = [arg for arg in command if arg.startswith("--gtest_filter=")]
                    self.assertEqual(len(filters), 1)
                    actual.add(filters[0].split("FiniteBodySiblingNativeIntegration.", 1)[1])
                    properties = {item["name"]: item["value"] for item in entry["properties"]}
                    self.assertEqual(properties["TIMEOUT"], 30)
                    self.assertIn("ProductionTestPreflight", properties["LABELS"])
                    if filters[0].endswith("RejectsOriginalForeignDevice"):
                        backend = "CUDA" if "CUDA_" in filters[0] else "ROCm"
                        self.assertIn(backend + "2", properties["LABELS"])
                    self.assertNotIn("--gtest_repeat=20", command)  # The actual native body performs 20 replays.
                self.assertEqual(actual, expected)

    def test_creation_and_all_requirements_follow_selection(self):
        """Both inventories configure and retain every admitted requirement."""
        for performance_only in (False, True):
            with self.subTest(performance_only=performance_only):
                result = self.configure(performance_only)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_host_consumer_target_keeps_canonical_entry_in_both_builds(self):
        """Import the actual module: both modes retain the diagnostic and exactly one lexical entry."""
        policy = MODULE.with_name("V2GPUProcessTestEntryPolicy.cmake")
        fingerprint = MODULE.parents[3] / "cmake/NativeSourceFingerprint.cmake"
        extra = f"""
if(V2_PERF_TESTS_ONLY)
    set(CMAKE_BUILD_TYPE Release)
else()
    set(CMAKE_BUILD_TYPE Integration)
endif()
set(LLAMINAR_SHARED_CORE ON)
add_library(llaminar2_core INTERFACE)
add_library(GTest::gtest INTERFACE IMPORTED)
add_library(GTest::gtest_main INTERFACE IMPORTED)
add_library(Threads::Threads INTERFACE IMPORTED)
add_library(v2_core_gtest_main INTERFACE)
include("{policy.as_posix()}")
include("{fingerprint.as_posix()}")
function(add_v2_test)
endfunction()
include("{HOST_SERVICE_REGISTRATION.as_posix()}")
set(consumer v2_perf_native_host_service_consumers)
if(NOT TARGET ${{consumer}})
    message(FATAL_ERROR "Original consumer diagnostic excluded by target selection")
endif()
get_target_property(sources ${{consumer}} SOURCES)
get_target_property(excluded ${{consumer}} EXCLUDE_FROM_ALL)
get_target_property(scope ${{consumer}} V2_TEST_DEVICE_SCOPE)
get_target_property(definitions ${{consumer}} COMPILE_DEFINITIONS)
get_target_property(links ${{consumer}} LINK_LIBRARIES)
set(entry_count 0)
foreach(source IN LISTS sources)
    get_filename_component(name "${{source}}" NAME)
    if(name STREQUAL "core_gtest_main.cpp")
        math(EXPR entry_count "${{entry_count}} + 1")
        file(REAL_PATH "${{source}}" entry BASE_DIRECTORY "${{CMAKE_CURRENT_SOURCE_DIR}}")
        file(REAL_PATH "${{CMAKE_CURRENT_SOURCE_DIR}}/integration/core_gtest_main.cpp" expected)
        if(NOT entry STREQUAL expected)
            message(FATAL_ERROR "Foreign canonical process entry")
        endif()
    endif()
endforeach()
if(NOT entry_count EQUAL 1)
    message(FATAL_ERROR "Original consumer diagnostic lost canonical process entry")
endif()
if("GTest::gtest_main" IN_LIST links OR "v2_core_gtest_main" IN_LIST links)
    message(FATAL_ERROR "Original consumer diagnostic has duplicate process entry")
endif()
if(NOT excluded OR NOT scope STREQUAL "DeclaredBackends" OR
   NOT "LLAMINAR_HOST_SERVICE_ORIGINAL_CONSUMERS=1" IN_LIST definitions)
    message(FATAL_ERROR "Original consumer diagnostic lost its explicit ownership contract")
endif()
"""
        for performance_only in (False, True):
            with self.subTest(performance_only=performance_only):
                configured = self.configure(performance_only, extra=extra)
                self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)

        # Re-run the real registration with each original defect independently.
        # Configuration properties catch both faults without compiling SDK/model code.
        registration = HOST_SERVICE_REGISTRATION.read_text(encoding="utf-8")
        variants = (
            (True, registration.replace("v2_perf_native_host_service_consumers",
                                        "v2_integration_native_host_service_consumers"),
             "Original consumer diagnostic excluded by target selection"),
            (False, registration.replace("    integration/core_gtest_main.cpp)", ")"),
             "Original consumer diagnostic lost canonical process entry"),
        )
        for performance_only, body, diagnostic in variants:
            with self.subTest(original_defect=diagnostic):
                with tempfile.TemporaryDirectory(prefix="llaminar-consumer-target-control-") as tmp:
                    module = Path(tmp) / "V2RetainedHostServiceTests.cmake"
                    module.write_text(body, encoding="utf-8")
                    configured = self.configure(performance_only, extra=extra.replace(
                        HOST_SERVICE_REGISTRATION.as_posix(), module.as_posix()))
                    self.assertNotEqual(configured.returncode, 0)
                    self.assertIn(diagnostic, configured.stdout + configured.stderr)

    def test_admitted_targets_still_reject_unknown_features(self):
        """The Release wrapper must not suppress validation on a live target."""
        result = self.configure(True, "cxx_not_a_real_compile_feature")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cxx_not_a_real_compile_feature", result.stderr)
        self.assertIn("v2_perf_contract", result.stderr)

    def test_unknown_scope_is_not_hidden_by_release_exclusion(self):
        """Invalid startup ownership fails even on a skipped declaration."""
        result = self.configure(True, scope="InventedScope")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown V2 test device scope", result.stderr)

    def test_missing_live_target_is_not_silently_ignored(self):
        """Only intentional Release exclusions may have no target owner."""
        for performance_only, target in ((False, "v2_integration_missing"),
                                         (True, "v2_perf_missing")):
            with self.subTest(performance_only=performance_only):
                result = self.configure(performance_only,
                    extra=f"v2_test_device_scope({target} FullInventory)")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(target, result.stderr)

    def test_qfn_performance_targets_survive_release_selection(self):
        """Native timing targets retain the same backend inventory in both builds."""
        for performance_only in (False, True):
            for cuda, rocm in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(performance_only=performance_only, cuda=cuda, rocm=rocm):
                    expected = ["v2_perf_immutable_host_lookup", "v2_perf_qwen_sparse_selection",
                                "v2_perf_q8_residual_prepared_projection"]
                    if cuda:
                        expected += ["v2_perf_cuda_qwen_sparse_selection",
                                     "v2_perf_cuda_qwen_sparse_observations"]
                    if rocm:
                        expected += ["v2_perf_rocm_qwen_sparse_selection",
                                     "v2_perf_rocm_qwen_sparse_observations",
                                     "v2_perf_rocm_floating_reduction",
                                     "v2_perf_qwen4exp_branch_norm"]
                    # Consume the actual outside-functional-umbrella declarations,
                    # including the original resident body, in both inventories.
                    declaration = CMAKE_LISTS.read_text(encoding="utf-8")
                    start = declaration.index("# Timing targets must also exist when Release excludes functional preflight.")
                    stop = declaration.index("# Use the same native oracle in Release", start)
                    performance_declarations = declaration[start:stop].replace(
                        "${CMAKE_CURRENT_LIST_DIR}/cmake/", MODULE.parent.as_posix() + "/")
                    helper_start = declaration.index("function(add_v2_test TEST_NAME)")
                    helper_end = declaration.index("endfunction()", helper_start) + len("endfunction()")
                    registration = declaration[helper_start:helper_end]
                    # Observe actual registration arguments without replacing the
                    # production registration body or its ownership checks.
                    registration = registration.replace(
                        "function(add_v2_test TEST_NAME)",
                        'function(add_v2_test TEST_NAME)\n'
                        'set_property(GLOBAL PROPERTY QFN_TEST_ARGS_${TEST_NAME} "${ARGN}")',
                        1)
                    if cuda or rocm:
                        expected += ["v2_perf_immutable_resident_lookup", "v2_perf_pooled_ngram_lookup"]
                    if cuda and rocm:
                        expected.append("v2_perf_captured_ngram_channel_floor")
                    extra = f"""
enable_testing()
set(HAVE_CUDA {"ON" if cuda else "OFF"})
set(HAVE_ROCM {"ON" if rocm else "OFF"})
set(CMAKE_BUILD_TYPE {"Release" if performance_only else "Integration"})
set(V2_TEST_SOCKETS 1)
set(V2_TEST_CORES_PER_SOCKET 1)
{registration}
add_library(llaminar2_core INTERFACE)
add_library(GTest::gtest INTERFACE IMPORTED)
add_library(GTest::gtest_main INTERFACE IMPORTED)
add_library(hip::host INTERFACE IMPORTED)
add_library(CUDA::cudart INTERFACE IMPORTED)
# Actual pooled declarations attach include requirements and registrations to
# these pre-existing functional owners; only their bodies are fixture doubles.
add_executable(v2_test_memory_planner main.cpp)
add_executable(v2_test_physical_memory_authority main.cpp)
{performance_declarations}
include("{QFN_PERFORMANCE.as_posix()}")
if(TARGET v2_perf_immutable_adaptive_lookup)
    message(FATAL_ERROR "Retired combined adaptive lookup performance entrypoint remains")
endif()
if(HAVE_CUDA)
    get_target_property(hc103_sources v2_perf_q8_residual_prepared_projection SOURCES)
    get_target_property(hc103_standard v2_perf_q8_residual_prepared_projection CUDA_STANDARD)
    if(NOT hc103_standard EQUAL 20)
        message(FATAL_ERROR "HC control lost its native CUDA language contract")
    endif()
    set(hc103_cuda_sources 0)
    foreach(source IN LISTS hc103_sources)
        if(source MATCHES "[.]cu$")
            math(EXPR hc103_cuda_sources "${{hc103_cuda_sources}} + 1")
            get_source_file_property(language "${{source}}" LANGUAGE)
            if(NOT language STREQUAL "CUDA")
                message(FATAL_ERROR "HC control source is not native CUDA")
            endif()
            set_source_files_properties("${{source}}" PROPERTIES HEADER_FILE_ONLY TRUE)
        endif()
    endforeach()
    if(NOT hc103_cuda_sources EQUAL 1)
        message(FATAL_ERROR "HC control lost its independent CUDA oracle")
    endif()
    set_target_properties(v2_perf_q8_residual_prepared_projection PROPERTIES LINKER_LANGUAGE CXX)
endif()
if(HAVE_ROCM)
    foreach(native_target IN ITEMS v2_perf_rocm_qwen_sparse_observations
                                   v2_perf_rocm_floating_reduction
                                   v2_perf_qwen4exp_branch_norm
                                   v2_perf_q8_residual_prepared_projection)
        get_target_property(native_sources ${{native_target}} SOURCES)
        get_target_property(native_standard ${{native_target}} HIP_STANDARD)
        if(NOT native_standard EQUAL 20)
            message(FATAL_ERROR "${{native_target}} lost its native HIP language contract")
        endif()
        set(native_source_count 0)
        foreach(source IN LISTS native_sources)
            if(source MATCHES "[.]hip$")
                math(EXPR native_source_count "${{native_source_count}} + 1")
                get_source_file_property(language "${{source}}" LANGUAGE)
                if(NOT language STREQUAL "HIP")
                    message(FATAL_ERROR "${{native_target}} source is not native HIP")
                endif()
                # This metadata-only fixture has no HIP compiler. Authenticate
                # the native oracle before omitting it from the generated stub.
                set_source_files_properties("${{source}}" PROPERTIES HEADER_FILE_ONLY TRUE)
            endif()
        endforeach()
        # The QSA observation target retains its test-owned evaluator probe.
        # Floating reduction retains its existing HIP fixture plus the frozen
        # pointer-pipeline oracle. Branch norm and HC control each retain one oracle.
        # All are native HIP in production; only this CXX-only metadata mock
        # marks the authenticated source header-only before CMake generation.
        set(expected_native_sources 1)
        if(native_target STREQUAL "v2_perf_rocm_floating_reduction")
            set(expected_native_sources 2)
        endif()
        if(NOT native_source_count EQUAL expected_native_sources)
            message(FATAL_ERROR "${{native_target}} lost its independent native source")
        endif()
        set_target_properties(${{native_target}} PROPERTIES LINKER_LANGUAGE CXX)
    endforeach()
endif()
get_property(all_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
list(FILTER all_targets INCLUDE REGEX "^v2_perf_")
list(REMOVE_ITEM all_targets v2_perf_contract)
set(expected {" ".join(expected)})
list(SORT all_targets)
list(SORT expected)
if(NOT all_targets STREQUAL expected)
    message(FATAL_ERROR "Lost QFN performance targets: ${{all_targets}}; expected ${{expected}}")
endif()
foreach(target IN LISTS expected)
    get_target_property(sources ${{target}} SOURCES)
    foreach(source IN LISTS sources)
        get_filename_component(source_absolute "${{source}}" ABSOLUTE
            BASE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}")
        if(NOT EXISTS "${{source_absolute}}")
            message(FATAL_ERROR "Missing performance source: ${{source}}")
        endif()
    endforeach()
endforeach()
# Inspect the actual shared replay actor before suppressing native compilation
# in this metadata-only CXX fixture. No production target or helper is replaced.
if(TARGET v2_perf_pooled_ngram_lookup)
    get_target_property(pooled_sources v2_perf_pooled_ngram_lookup SOURCES)
    set(expected_pooled_sources performance/kernels/qwen4exp/Measure__PooledNgramLookup.cpp)
    foreach(backend CUDA ROCm)
        if(backend STREQUAL "CUDA")
            set(extension cu)
            set(language CUDA)
        else()
            set(extension hip)
            set(language HIP)
        endif()
        if((backend STREQUAL "CUDA" AND HAVE_CUDA) OR (backend STREQUAL "ROCm" AND HAVE_ROCM))
            list(APPEND expected_pooled_sources integration/kernels/qwen4exp/PooledACKReadyCancellation${{backend}}.${{extension}})
        endif()
    endforeach()
    if(NOT pooled_sources STREQUAL expected_pooled_sources)
        message(FATAL_ERROR "Pooled timing native inventory drift: ${{pooled_sources}}")
    endif()
    get_target_property(pooled_includes v2_perf_pooled_ngram_lookup INCLUDE_DIRECTORIES)
    foreach(required_include "${{CMAKE_CURRENT_SOURCE_DIR}}" "${{CMAKE_CURRENT_SOURCE_DIR}}/integration/kernels/qwen4exp")
        if(NOT required_include IN_LIST pooled_includes)
            message(FATAL_ERROR "Pooled timing lost fixture include: ${{required_include}}")
        endif()
    endforeach()
    if(HAVE_ROCM)
        get_target_property(pooled_standard v2_perf_pooled_ngram_lookup HIP_STANDARD)
        if(NOT pooled_standard EQUAL 20)
            message(FATAL_ERROR "Pooled timing lost native HIP standard")
        endif()
    endif()
    foreach(source IN LISTS pooled_sources)
        if(source MATCHES "[.](cu|hip)$")
            get_source_file_property(native_language "${{source}}" LANGUAGE)
            # Unset language uses the compiler's native extension in production.
            # An explicit language must not override that extension with CXX.
            if((source MATCHES "[.]cu$" AND native_language AND NOT native_language STREQUAL "CUDA") OR
               (source MATCHES "[.]hip$" AND native_language AND NOT native_language STREQUAL "HIP"))
                message(FATAL_ERROR "Pooled timing source lost native language: ${{source}}")
            endif()
            set_source_files_properties("${{source}}" PROPERTIES HEADER_FILE_ONLY TRUE)
        endif()
    endforeach()
    set_target_properties(v2_perf_pooled_ngram_lookup PROPERTIES LINKER_LANGUAGE CXX)
endif()
if(TARGET v2_perf_captured_ngram_channel_floor)
    get_target_property(floor_sources v2_perf_captured_ngram_channel_floor SOURCES)
    get_target_property(floor_scope v2_perf_captured_ngram_channel_floor V2_TEST_DEVICE_SCOPE)
    set(expected_floor_sources
        integration/transfer/Test__CapturedTransferChannel.cpp
        integration/transfer/Test__DeviceCountedTransferChannel.cpp
        integration/transfer/CountedChannelTestKernels.cu
        integration/transfer/CountedChannelTestKernels.hip
        integration/mpi_gtest_main.cpp)
    if(NOT floor_sources STREQUAL expected_floor_sources OR
       NOT floor_scope STREQUAL "FullInventory")
        message(FATAL_ERROR "Channel floor lost production fixture or startup ownership")
    endif()
    # The shared fixture calls the counted peer replay even when timing selects
    # another case. Its complete native closure must link in both build modes.
    get_target_property(floor_hip_standard v2_perf_captured_ngram_channel_floor HIP_STANDARD)
    if(NOT floor_hip_standard EQUAL 20)
        message(FATAL_ERROR "Channel floor lost counted HIP fixture standard")
    endif()
    foreach(source IN LISTS floor_sources)
        if(source MATCHES "[.](cu|hip)$")
            get_source_file_property(native_language "${{source}}" LANGUAGE)
            if((source MATCHES "[.]cu$" AND native_language AND NOT native_language STREQUAL "CUDA") OR
               (source MATCHES "[.]hip$" AND native_language AND NOT native_language STREQUAL "HIP"))
                message(FATAL_ERROR "Channel floor lost native helper language: ${{source}}")
            endif()
            set_source_files_properties("${{source}}" PROPERTIES HEADER_FILE_ONLY TRUE)
        endif()
    endforeach()
    set_target_properties(v2_perf_captured_ngram_channel_floor PROPERTIES LINKER_LANGUAGE CXX)
endif()
get_property(tests DIRECTORY PROPERTY TESTS)
set(expected_tests)
if(NOT V2_PERF_TESTS_ONLY)
    list(APPEND expected_tests V2_Integration_ImmutableHostLookupFixtureOwnership
        V2_Integration_AuthoredRemoteDuplicateShapeReplay
        V2_Unit_PooledNgramReplayTrace V2_Integration_PooledNgramReplayTrace
        V2_Unit_PooledNgramRealTraceContract V2_Integration_PooledNgramRealTraceContract
        V2_Integration_PooledNgramCampaignSeedContract V2_Integration_PooledNgramNativeSourceExtent
        V2_Integration_PooledNgramFinalPMACostEvidence V2_Integration_PooledNgramMatchedPhysicalBudget
        V2_Integration_PooledNgramSourceTerminalCostWitness)
    if(HAVE_CUDA OR HAVE_ROCM)
        list(APPEND expected_tests V2_Unit_PooledConsumerLocalEconomyDeclarations
            V2_Integration_PooledConsumerLocalEconomyDeclarations)
    endif()
    foreach(backend CUDA ROCm)
        if((backend STREQUAL "CUDA" AND HAVE_CUDA) OR
           (backend STREQUAL "ROCm" AND HAVE_ROCM))
            foreach(storage ram mmap adaptive)
                list(APPEND expected_tests V2_Integration_${{backend}}ImmutableHostLookup_${{storage}}NativeActorReplay)
            endforeach()
        endif()
    endforeach()
    get_target_property(trace_sources v2_test_pooled_ngram_replay_trace SOURCES)
    get_target_property(trace_features v2_test_pooled_ngram_replay_trace COMPILE_FEATURES)
    if(NOT trace_sources STREQUAL "unit/kernels/qwen4exp/Test__PooledNgramReplayTrace.cpp" OR
       NOT "cxx_std_20" IN_LIST trace_features)
        message(FATAL_ERROR "Replay trace lost its source or language contract")
    endif()
    get_target_property(planner_includes v2_test_memory_planner INCLUDE_DIRECTORIES)
    if(NOT "${{CMAKE_CURRENT_SOURCE_DIR}}/../../external/cnpy" IN_LIST planner_includes)
        message(FATAL_ERROR "Pooled physical-budget fixture lost its reference include requirement")
    endif()
else()
    foreach(functional_owner v2_test_pooled_ngram_replay_trace
                             v2_test_memory_planner v2_test_physical_memory_authority)
        if(TARGET ${{functional_owner}})
            message(FATAL_ERROR "Release retained functional owner: ${{functional_owner}}")
        endif()
    endforeach()
endif()
set(placements)
if(HAVE_CUDA)
    list(APPEND placements LocalCUDA)
endif()
if(HAVE_ROCM)
    list(APPEND placements LocalROCm)
endif()
if(HAVE_CUDA AND HAVE_ROCM)
    list(APPEND placements CUDAConsumerROCmStorage ROCmConsumerCUDAStorage)
    foreach(direction CUDA_ROCm ROCm_CUDA)
        foreach(payload Decode1440 M512737280)
            list(APPEND expected_tests V2_Perf_CapturedNgramChannelFloor_${{direction}}_${{payload}})
        endforeach()
    endforeach()
    foreach(backend CUDA ROCm)
        foreach(transport Native Mapped)
            foreach(direction Forward Reverse)
                foreach(payload Header64 Decode1440 M512737280)
                    list(APPEND expected_tests V2_Perf_CapturedNgramChannelFloor_${{backend}}_${{transport}}${{direction}}_${{payload}})
                endforeach()
            endforeach()
        endforeach()
    endforeach()
endif()
foreach(backing SSD RAM)
    set(backing_suffix "")
    if(backing STREQUAL "RAM")
        set(backing_suffix _RAMSource)
    endif()
    foreach(placement IN LISTS placements)
        foreach(tokens 1 8 64 512)
            foreach(scenario warm_hits alternating_sets mixed churn empty duplicate_control)
                list(APPEND expected_tests V2_Perf_PooledNgram_${{placement}}_M${{tokens}}_${{scenario}}${{backing_suffix}})
            endforeach()
        endforeach()
    endforeach()
    foreach(backend CUDA ROCm)
        if((backend STREQUAL "CUDA" AND HAVE_CUDA) OR
           (backend STREQUAL "ROCm" AND HAVE_ROCM))
            foreach(participants 1 2 4 8)
                foreach(mode C0Alternating PositiveChurn)
                    list(APPEND expected_tests V2_Perf_PooledNgram_ConsumerLocal${{backend}}${{participants}}_${{mode}}ResetReplay${{backing_suffix}})
                endforeach()
            endforeach()
        endif()
    endforeach()
endforeach()
list(SORT tests)
list(SORT expected_tests)
if(NOT "${{tests}}" STREQUAL "${{expected_tests}}")
    message(FATAL_ERROR "Unexpected performance/fixture registrations: ${{tests}}; expected ${{expected_tests}}")
endif()
foreach(test IN LISTS tests)
    get_test_property(${{test}} LABELS labels)
    get_test_property(${{test}} FIXTURES_REQUIRED fixtures)
    if(NOT "ProcessOwned" IN_LIST labels OR "V2_Models" IN_LIST fixtures)
        message(FATAL_ERROR "Lost model-free process ownership: ${{test}}")
    endif()
    if(test MATCHES "^V2_Perf_PooledNgram_.*_RAMSource$")
        get_property(arguments GLOBAL PROPERTY QFN_TEST_ARGS_${{test}})
        foreach(option --source-backing --ram-slots)
            list(FIND arguments "${{option}}" option_index)
            if(option_index LESS 0)
                message(FATAL_ERROR "RAM source lost explicit ${{option}}: ${{test}}")
            endif()
            math(EXPR value_index "${{option_index}} + 1")
            list(GET arguments ${{value_index}} value)
            if((option STREQUAL "--source-backing" AND NOT value STREQUAL "ram") OR
               (option STREQUAL "--ram-slots" AND NOT value STREQUAL "0"))
                message(FATAL_ERROR "RAM source changed backing/cache authority: ${{test}} ${{option}}=${{value}}")
            endif()
        endforeach()
    endif()
    if(test MATCHES "^V2_Perf_")
        if(NOT "Performance" IN_LIST labels OR "Unit" IN_LIST labels OR "ProductionTestPreflight" IN_LIST labels)
            message(FATAL_ERROR "Timing entered a functional gate: ${{test}}")
        endif()
    elseif(test MATCHES "^V2_Integration_(CUDA|ROCm)ImmutableHostLookup_(ram|mmap|adaptive)NativeActorReplay$")
        set(actor_backend "${{CMAKE_MATCH_1}}")
        if(NOT actor_backend IN_LIST labels OR NOT "NativeReplay" IN_LIST labels OR
           NOT "ProductionTestPreflight" IN_LIST labels OR "DeviceFree" IN_LIST labels)
            message(FATAL_ERROR "Native Actor lost exact backend/gate contract: ${{test}}")
        endif()
        if((actor_backend STREQUAL "CUDA" AND NOT HAVE_CUDA) OR
           (actor_backend STREQUAL "ROCm" AND NOT HAVE_ROCM) OR V2_PERF_TESTS_ONLY)
            message(FATAL_ERROR "Native Actor registered outside its eligible Integration mode: ${{test}}")
        endif()
    elseif(NOT "DeviceFree" IN_LIST labels)
        message(FATAL_ERROR "Host fixture lost device-free contract: ${{test}}")
    endif()
endforeach()
"""
                    result = self.configure(performance_only, extra=extra)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_pooled_and_owned_registrations_follow_release_selection(self):
        """Actual pooled/MPI declarations retain Integration and omit Release owners.

        The real add_v2_test function and affected balanced source sections are
        projected from production CMake, not rewritten into a permissive stub.
        Only link dependencies and executable bodies are metadata-only doubles;
        no target, test executable, MPI or GPU runtime is built or executed.
        """
        declaration = CMAKE_LISTS.read_text(encoding="utf-8")

        def section(first: str, following: str) -> str:
            """Retain a complete actual declaration block between unique boundaries."""
            start = declaration.index(first)
            return declaration[start:declaration.index(following, start)]

        helper_start = declaration.index("function(add_v2_test TEST_NAME)")
        helper_end = declaration.index("endfunction()", helper_start) + len("endfunction()")
        registration = declaration[helper_start:helper_end]
        native_blocks = section("# Explicit native gates execute captured production phase callers;",
                                "# Actual retained external arena ownership is device-free;")
        native_blocks += section("# Real source gates use the actual retained service,",
                                 "foreach(QFN_PROVIDER_BACKEND IN ITEMS CUDA ROCm)")
        owner = section("add_executable(v2_test_pooled_owner_pin_chain ",
                        "foreach(_pooled_owner_chain_suite Unit Integration)")
        host_service = HOST_SERVICE_REGISTRATION.read_text(encoding="utf-8")
        # Include only the actual source scanner registration, without mocking
        # the canonical child-path rewriting in add_v2_test.
        scanner_marker = "# The actual environment authority scanner"
        scanner_start = host_service.index(scanner_marker)
        tool = host_service.rfind("find_program(", 0, scanner_start)
        scanner = host_service[tool if tool >= 0 else scanner_start:]
        mixed = section("set(_pooled_mixed_local_graph_cases",
                        "# Exact original release/retirement bodies:")
        owner_source = (CMAKE_LISTS.parent / "unit/kernels/qwen4exp/Test__PooledImmutableOwnerPinChain.cpp").read_text()
        mixed_cases = re.findall(r"TEST\(\s*PooledImmutableOwnerPinChain\s*,\s*(MixedLocal\w+)\s*\)", owner_source)
        self.assertTrue(mixed_cases, "No original mixed-local compiled cases discovered")
        self.assertEqual(len(mixed_cases), len(set(mixed_cases)))
        reader = section("add_executable(v2_test_pooled_hit_read_graph ",
                         "add_v2_test(V2_Unit_PooledImmutableHitReadGraphReplay")
        terminal = section("add_executable(v2_test_pooled_retirement_body ",
                           "add_v2_test(V2_Unit_PooledImmutableRetirementBodyReplay")
        for performance_only in (False, True):
            for cuda, rocm in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(performance_only=performance_only, cuda=cuda, rocm=rocm):
                    with tempfile.TemporaryDirectory(prefix="llaminar-pooled-selection-") as tmp:
                        root = Path(tmp)
                        for relative in ("unit", "fixtures", "integration"):
                            (root / relative).symlink_to(MODULE.parent.parent / relative,
                                                         target_is_directory=True)
                        (root / "main.cpp").write_text("int main(){return 0;}\n")
                        # Imported tools provide resolvable TARGET_FILE metadata;
                        # CTest --show-only never executes their stand-in body.
                        fixture = f'''
cmake_minimum_required(VERSION 3.27)
project(PooledReleaseSelection LANGUAGES C CXX)
enable_testing()
set(V2_PERF_TESTS_ONLY {"ON" if performance_only else "OFF"})
set(HAVE_CUDA {"ON" if cuda else "OFF"})
set(HAVE_ROCM {"ON" if rocm else "OFF"})
set(REPO_ROOT "{MODULE.parents[3].as_posix()}")
set(PROJECT_ROOT "${{REPO_ROOT}}")
set(Python3_EXECUTABLE "{sys.executable}")
set(MPIEXEC_EXECUTABLE "${{CMAKE_COMMAND}}")
set(V2_TEST_SOCKETS 1)
set(V2_TEST_CORES_PER_SOCKET 1)
add_library(llaminar2_core INTERFACE)
add_library(llaminar_mpi_session_owner INTERFACE)
add_library(GTest::gtest INTERFACE IMPORTED)
add_library(GTest::gtest_main INTERFACE IMPORTED)
add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
add_executable(llaminar_mpi_session IMPORTED)
set_target_properties(llaminar_mpi_session PROPERTIES IMPORTED_LOCATION "${{CMAKE_COMMAND}}")
include("{MODULE.as_posix()}")
{registration}
foreach(target v2_test_qwen4exp_device_primitives v2_unit_moe_overlay_node_local_rank_batch_transport
               v2_test_shmem_spin_backend v2_mpi_session_probe)
    add_executable(${{target}} main.cpp)
endforeach()
{native_blocks}
{owner}
{mixed}
{scanner}
{reader}
{terminal}
include("{OWNED_REGISTRATION.as_posix()}")
include("{COHORT_REGISTRATION.as_posix()}")
foreach(target v2_test_pooled_owner_pin_chain v2_test_pooled_hit_read_graph
               v2_test_pooled_retirement_body v2_test_pooled_retirement_graph
               v2_test_pooled_complete_cohort)
    if(V2_PERF_TESTS_ONLY)
        if(TARGET ${{target}})
            message(FATAL_ERROR "Release materialized functional target ${{target}}")
        endif()
    else()
        get_target_property(scope ${{target}} V2_TEST_DEVICE_SCOPE)
        get_target_property(options ${{target}} LINK_OPTIONS)
        get_target_property(features ${{target}} COMPILE_FEATURES)
        if(NOT scope STREQUAL "DeclaredBackends" OR
           NOT "LINKER:--export-dynamic" IN_LIST options OR
           NOT "cxx_std_23" IN_LIST features)
            message(FATAL_ERROR "Lost admitted requirements for ${{target}}")
        endif()
    endif()
endforeach()
if(TARGET v2_test_qwen4exp_device_primitives)
    get_target_property(sources v2_test_qwen4exp_device_primitives SOURCES)
    set(expected)
    if(HAVE_CUDA OR HAVE_ROCM)
        list(APPEND expected integration/kernels/qwen4exp/Test__PooledImmutableNativeDevice.cpp
            integration/kernels/qwen4exp/Test__PooledImmutableProviderSourceDevice.cpp
            integration/kernels/qwen4exp/Test__ImmutableLookupPreparationLifetime.cpp)
    endif()
    if(HAVE_CUDA)
        list(APPEND expected integration/kernels/qwen4exp/CUDAPooledImmutableNativeSourceFixture.cu)
    endif()
    if(HAVE_ROCM)
        list(APPEND expected integration/kernels/qwen4exp/ROCmPooledImmutableNativeSourceFixture.hip)
        get_source_file_property(language integration/kernels/qwen4exp/ROCmPooledImmutableNativeSourceFixture.hip LANGUAGE)
        if(NOT language STREQUAL "HIP")
            message(FATAL_ERROR "Lost HIP source identity")
        endif()
    endif()
    list(REMOVE_ITEM sources main.cpp)
    list(SORT sources)
    list(SORT expected)
    if(NOT "${{sources}}" STREQUAL "${{expected}}")
        message(FATAL_ERROR "Pooled native source inventory drift: ${{sources}} expected ${{expected}}")
    endif()
    # The actual source identity above is checked before this host-only fixture
    # marks native sources header-only. Production compiles them natively.
    foreach(source IN LISTS sources)
        if(source MATCHES "[.](cu|hip)$")
            set_source_files_properties("${{source}}" PROPERTIES HEADER_FILE_ONLY TRUE)
        endif()
    endforeach()
endif()
get_property(tests DIRECTORY PROPERTY TESTS)
if(V2_PERF_TESTS_ONLY AND tests)
    message(FATAL_ERROR "Release retained functional registrations: ${{tests}}")
endif()
if(NOT V2_PERF_TESTS_ONLY)
    # The canonical cohort module owns the cases. Every isolated original
    # process must remain in Unit after retiring the former aggregate alias.
    if(NOT _cohort_original_unit_cases)
        message(FATAL_ERROR "Cohort module declares no original Unit cases")
    endif()
    # Pooled sources are CUDA/ROCm-native: without a compiled GPU family the
    # cohort executable declares REQUIRES_BACKEND and every case is omitted
    # and counted, never registered to fail on "not compiled".
    get_property(omitted GLOBAL PROPERTY V2_BACKEND_OMITTED_TESTS)
    foreach(case IN LISTS _cohort_original_unit_cases)
        string(REPLACE "." "_" case_id "${{case}}")
        if(HAVE_CUDA OR HAVE_ROCM)
            if(NOT "V2_Unit_PooledOriginal_${{case_id}}" IN_LIST tests)
                message(FATAL_ERROR "Missing original cohort Unit registration: ${{case}}")
            endif()
        elseif("V2_Unit_PooledOriginal_${{case_id}}" IN_LIST tests OR
               NOT "V2_Unit_PooledOriginal_${{case_id}}" IN_LIST omitted)
            message(FATAL_ERROR "CPU-only build must omit and count cohort case: ${{case}}")
        endif()
    endforeach()
endif()
get_property(targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
if(V2_PERF_TESTS_ONLY)
    foreach(target IN LISTS targets)
        if(target MATCHES "^v2_(test_|mpi_owned_shared_|moe_owned_shared_)")
            message(FATAL_ERROR "Release retained functional build owner ${{target}}")
        endif()
    endforeach()
endif()
'''
                        (root / "CMakeLists.txt").write_text(fixture, encoding="utf-8")
                        configured = subprocess.run(
                            [self.cmake, "-S", str(root), "-B", str(root / "build"),
                             "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={self.ninja}"],
                            capture_output=True, text=True, timeout=15, check=False)
                        self.assertEqual(configured.returncode, 0,
                                         configured.stdout + configured.stderr)
                        inventory = subprocess.run(
                            [str(Path(self.cmake).with_name("ctest")), "--test-dir", str(root / "build"),
                             "--show-only=json-v1"], capture_output=True, text=True,
                            timeout=15, check=True)
                        tests = json.loads(inventory.stdout)["tests"]
                        prefix = "V2_Integration_PooledImmutableMixedLocalGraph_"
                        actual = [entry for entry in tests if entry["name"].startswith(prefix)]
                        scanners = [entry for entry in tests
                                    if entry["name"] == "V2_Integration_HostServiceDebugEnvAuthority"]
                        self.assertEqual(len(scanners), 0 if performance_only else 1)
                        for entry in scanners:
                            command = entry["command"]
                            self.assertIn("--session-base", command)
                            child = Path(command[command.index("--") + 1])
                            self.assertTrue(child.is_absolute(), command)
                            self.assertEqual(child.name, "bash")
                            self.assertTrue(child.is_file() and os.access(child, os.X_OK), command)
                            self.assertNotEqual(child, root / "build/bash")

                        self.assertNotIn("V2_Integration_PooledImmutableMixedLocalGraphReplay",
                                         [entry["name"] for entry in tests])
                        # The owner-pin executable requires a compiled GPU
                        # family; CPU-only builds omit (and count) its cases.
                        expected = (set() if performance_only or not (cuda or rocm)
                                    else {prefix + case + "Replay" for case in mixed_cases})
                        self.assertEqual({entry["name"] for entry in actual}, expected)
                        self.assertEqual(len(actual), len(expected), "Mixed-local registration duplicate")
                        for entry in actual:
                            case = entry["name"][len(prefix):-len("Replay")]
                            command = entry["command"]
                            self.assertIn("--gtest_filter=PooledImmutableOwnerPinChain." + case, command)
                            self.assertIn("--gtest_repeat=20", command)
                            self.assertTrue(any(arg.endswith("/scripts/run_cohort_original_cases.py") for arg in command))
                            properties = {prop["name"]: prop["value"] for prop in entry["properties"]}
                            self.assertEqual(properties["TIMEOUT"], 30.0)
                            self.assertIn("ProductionTestPreflight", properties["LABELS"])

                        if performance_only:
                            self.assertEqual(tests, [])
                        else:
                            names = {test["name"] for test in tests}
                            # CMake expands variable-based names in the inventory.
                            # Only complete literal names belong in this source check.
                            cohort_names = set(re.findall(
                                r"add_v2_test\((V2_[A-Za-z0-9_]+)(?=[\s)])",
                                COHORT_REGISTRATION.read_text(encoding="utf-8")))
                            self.assertTrue(cohort_names)
                            self.assertTrue(cohort_names <= names, cohort_names - names)
                            self.assertNotIn("V2_Unit_", names)
                            self.assertNotIn("V2_Integration_", names)
                            required = {"V2_Integration_MPINativeDaemonDiagnosticLifetime",
                                        "V2_Integration_OwnedSharedEvidenceLifetime",
                                        "V2_Integration_MPIRetirementDiagnostics",
                                        "V2_Integration_MPIRootRetirementBody"}
                            self.assertTrue(required <= names, required - names)
                            self.assertTrue(any(name.startswith("V2_Unit_PooledOriginal_")
                                                for name in names))
                            for test in tests:
                                properties = {p["name"]: p["value"] for p in test["properties"]}
                                self.assertIn("ProcessOwned", properties["LABELS"])
                                if test["name"].startswith("V2_Integration_"):
                                    self.assertIn("ProductionTestPreflight", properties["LABELS"])

    def configure_http_registration(self, performance_only: bool, *, declare_target: bool = True):
        """Generate actual production registrations without a compiler or device.

        The imported executable names CMake itself only to make CTest's
        inventory resolver independent of an unbuilt binary. It is never run.
        Production target selection still owns whether that target exists.
        """
        with tempfile.TemporaryDirectory(prefix="llaminar-http-registration-") as tmp:
            root = Path(tmp)
            target = """
add_executable(v2_test_server_mode IMPORTED)
if(TARGET v2_test_server_mode)
    set_target_properties(v2_test_server_mode PROPERTIES IMPORTED_LOCATION "${CMAKE_COMMAND}")
endif()
""" if declare_target else ""
            (root / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.20)
project(HttpServiceRegistration LANGUAGES NONE)
enable_testing()
set(V2_PERF_TESTS_ONLY {"ON" if performance_only else "OFF"})
include("{MODULE.as_posix()}")
{target}
include("{HTTP_REGISTRATION.as_posix()}")
include("{HTTP_REGISTRATION.as_posix()}")
""", encoding="utf-8")
            configured = subprocess.run(
                [self.cmake, "-S", str(root), "-B", str(root / "build"),
                 "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={self.ninja}"],
                capture_output=True, text=True, timeout=15, check=False)
            if configured.returncode:
                return configured, None
            inventory = subprocess.run(
                [str(Path(self.cmake).with_name("ctest")), "--test-dir", str(root / "build"),
                 "--show-only=json-v1"],
                capture_output=True, text=True, timeout=15, check=True)
            return configured, json.loads(inventory.stdout)["tests"]

    def test_http_service_registration_follows_target_selection(self):
        """Both native policies survive Integration and leave no Release target reference."""
        for performance_only in (False, True):
            with self.subTest(performance_only=performance_only):
                configured, tests = self.configure_http_registration(performance_only)
                self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
                expected = {} if performance_only else {
                    "V2_Integration_HTTPServiceThreadAffinity":
                        ["OMP_PROC_BIND=close", "OMP_PLACES=cores", "OMP_NUM_THREADS=2"],
                    "V2_Integration_HTTPServiceThreadAffinityUnbound":
                        ["OMP_PROC_BIND=false", "OMP_NUM_THREADS=2"],
                }
                self.assertEqual({test["name"] for test in tests}, set(expected))
                self.assertEqual(len(tests), len(expected))
                for test in tests:
                    properties = {value["name"]: value["value"] for value in test["properties"]}
                    self.assertEqual(set(properties["LABELS"]), {
                        "V2", "Integration", "ProductionTestPreflight", "HTTP",
                        "Observability", "Threading", "Regression", "DeviceFree"})
                    self.assertEqual(properties["TIMEOUT"], 30)
                    command = test["command"]
                    self.assertEqual(Path(command[0]).name, "cmake")
                    self.assertEqual(command[1:-2], ["-E", "env", *expected[test["name"]]])
                    self.assertEqual(command[-2], command[0])
                    self.assertEqual(command[-1], "--gtest_filter=HttpServiceThreadAffinity.*")

    def test_http_service_registration_rejects_missing_admitted_target(self):
        """An absent Integration executable remains a fatal configure error."""
        configured, tests = self.configure_http_registration(False, declare_target=False)
        self.assertNotEqual(configured.returncode, 0)
        self.assertIsNone(tests)
        self.assertIn('No target "v2_test_server_mode"', configured.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--ninja", default="ninja")
    args, remaining = parser.parse_known_args()
    TestTargetSelection.cmake = args.cmake
    TestTargetSelection.ninja = args.ninja
    unittest.main(argv=[__file__, *remaining])
