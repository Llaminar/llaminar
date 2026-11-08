#!/usr/bin/env python3
"""Replay actual native test-module selection with CMake, without building binaries.

The production selector and registration helpers own all mutations. Tiny fixture
owners supply the earlier target declarations only; every investigated module is
included verbatim. Backend flags exercise declaration branches, not hardware
qualification. Native CUDA/HIP sources are inspected then marked CXX solely so
this configure-only fixture never requires or claims a GPU compiler/runtime.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class NativeTestModuleSelection(unittest.TestCase):
    """Preserve functional selection and real performance command dependencies."""

    source_root = Path(__file__).resolve().parents[4]
    modules_root: Path | None = None
    cmake = "cmake"
    ninja = "ninja"
    modules = (
        "V2LocalHotNativeForkTests.cmake", "V2PooledLocalHotNativeTests.cmake",
        "V2CPUFragmentResidencyTests.cmake", "V2CPUShardRouteCollectiveTests.cmake",
        "V2NativeCPUCollectiveProgramTests.cmake", "V2CPUExpertDomainLifecycleTests.cmake",
        "V2NativeCollectiveOwnedErrorTests.cmake",
    )
    prerequisites = (
        "v2_unit_moe_overlay_participant_residency", "v2_unit_expert_gemm_registry",
        "v2_unit_moe_overlay_residency_authority", "v2_unit_moe_overlay_physical_residency_fabric",
        "v2_test_expert_tier_weight_stream", "v2_unit_moe_overlay_capacity_resolver",
        "v2_integration_moe_overlay_residency_consensus_mpi", "v2_integration_global_tp_mpi",
        "v2_test_cpu_expert_shard_execution", "v2_test_moe_overlay_tier_migration_transport",
        "v2_unit_moe_overlay_distributed_residency_transport",
        "v2_unit_moe_overlay_economy_calibration_controller",
        "v2_unit_moe_overlay_economy_profile_composer",
        "v2_unit_moe_overlay_migration_measurement_ledger",
    )

    def configure(self, performance: bool, cuda: bool, rocm: bool, *, missing_owner: bool = False):
        """Include actual modules and return real CTest metadata; never build or run a fixture executable."""
        source = self.source_root / "tests/v2"
        modules = self.modules_root or source / "cmake"
        declaration = (source / "CMakeLists.txt").read_text(encoding="utf-8")
        begin = declaration.index("function(add_v2_test TEST_NAME)")
        end = declaration.index("endfunction()", begin) + len("endfunction()")
        helper = declaration[begin:end]
        begin = declaration.index("function(add_v2_routed_expert_policy_unit TARGET_NAME TEST_NAME SOURCE_FILE)")
        end = declaration.index("endfunction()", begin) + len("endfunction()")
        helper += "\n" + declaration[begin:end]
        with tempfile.TemporaryDirectory(prefix="llaminar-native-module-selection-") as temp:
            project = Path(temp)
            root = project / "tests/v2"
            root.mkdir(parents=True)
            (project / "src").mkdir()
            (project / "src/v2").symlink_to(self.source_root / "src/v2", target_is_directory=True)
            for name in ("unit", "integration", "scripts"):
                (root / name).symlink_to(source / name, target_is_directory=True)
            (root / "mpi_session_gtest_main.cpp").symlink_to(source / "mpi_session_gtest_main.cpp")
            (root / "main.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
            owners = [name for name in self.prerequisites
                      if not (missing_owner and name == "v2_unit_moe_overlay_physical_residency_fabric")]
            includes = "\n".join(f'include("{(modules / name).as_posix()}")' for name in self.modules)
            (root / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.27)
project(NativeModuleSelection LANGUAGES CXX)
enable_testing()
set(CMAKE_BUILD_TYPE {"Release" if performance else "Integration"})
set(V2_PERF_TESTS_ONLY {"ON" if performance else "OFF"})
set(HAVE_CUDA {"ON" if cuda else "OFF"})
set(HAVE_ROCM {"ON" if rocm else "OFF"})
set(V2_TEST_SOCKETS 8)
set(V2_TEST_CORES_PER_SOCKET 1)
set(REPO_ROOT "{self.source_root.as_posix()}")
set(PROJECT_ROOT "${{REPO_ROOT}}")
set(Python3_EXECUTABLE "{sys.executable}")
set(MPIEXEC_EXECUTABLE "${{CMAKE_COMMAND}}")
foreach(library llaminar2_core v2_core_gtest_main)
    add_library(${{library}} INTERFACE)
endforeach()
foreach(library GTest::gtest GTest::gtest_main Threads::Threads MPI::MPI_CXX
                OpenMP::OpenMP_CXX CUDA::cudart hip::host)
    add_library(${{library}} INTERFACE IMPORTED)
endforeach()
add_executable(llaminar_mpi_session IMPORTED)
set_target_properties(llaminar_mpi_session PROPERTIES IMPORTED_LOCATION "${{CMAKE_COMMAND}}")
include("{(source / 'cmake/V2TestTargetSelection.cmake').as_posix()}")
include("{(source / 'cmake/V2GPUProcessTestEntryPolicy.cmake').as_posix()}")
{helper}
foreach(target {" ".join(owners)})
    add_executable(${{target}} main.cpp)
endforeach()
add_executable(v2_perf_selection_control main.cpp)
add_v2_test(V2_Perf_NativeModuleSelectionControl
    COMMAND $<TARGET_FILE:v2_perf_selection_control>
    LABELS "V2;Performance;CPU" MPI_PROCS 1 NO_MPI NO_MODELS)
{includes}
get_property(targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(target IN LISTS targets)
    if(V2_PERF_TESTS_ONLY AND target MATCHES "^v2_(test|unit|integration)_")
        message(FATAL_ERROR "Release admitted functional executable ${{target}}")
    endif()
    get_target_property(kind ${{target}} TYPE)
    if(kind STREQUAL "EXECUTABLE")
        # Only CTest command metadata is read, never this command executed.
        file(CREATE_LINK "${{CMAKE_COMMAND}}" "${{CMAKE_CURRENT_BINARY_DIR}}/${{target}}" SYMBOLIC)
    endif()
endforeach()
if(HAVE_CUDA OR HAVE_ROCM)
    if(V2_PERF_TESTS_ONLY)
        set(native_target v2_perf_pooled_local_hot_native)
    else()
        set(native_target v2_integration_pooled_local_hot_native)
        if(NOT TARGET v2_integration_local_hot_native_fork)
            message(FATAL_ERROR "Lost admitted native-fork fixture")
        endif()
    endif()
    if(NOT TARGET ${{native_target}})
        message(FATAL_ERROR "Lost selected pooled native fixture ${{native_target}}")
    endif()
    get_target_property(sources ${{native_target}} SOURCES)
    get_target_property(entry ${{native_target}} V2_GPU_PROCESS_ENTRY)
    if(NOT entry STREQUAL "v2_core_gtest_main")
        message(FATAL_ERROR "Lost lexical runtime owner for ${{native_target}}")
    endif()
    foreach(backend CUDA ROCm)
        if(backend STREQUAL "CUDA" AND HAVE_CUDA)
            set(response integration/kernels/qwen4exp/LocalHotRecordedResponse.cu)
        elseif(backend STREQUAL "ROCm" AND HAVE_ROCM)
            set(response integration/kernels/qwen4exp/LocalHotRecordedResponse.hip)
            get_target_property(hip_standard ${{native_target}} HIP_STANDARD)
            if(NOT hip_standard EQUAL 20)
                message(FATAL_ERROR "Lost native HIP language contract")
            endif()
        else()
            continue()
        endif()
        if(NOT response IN_LIST sources)
            message(FATAL_ERROR "Lost native response source ${{response}}")
        endif()
        set_source_files_properties(${{response}} PROPERTIES LANGUAGE CXX)
    endforeach()
endif()
if(NOT V2_PERF_TESTS_ONLY)
    get_target_property(fragment v2_unit_moe_overlay_physical_residency_fabric SOURCES)
    if(NOT "unit/execution/moe/Test__MoEOverlayPhysicalProjectionPieces.cpp" IN_LIST fragment)
        message(FATAL_ERROR "Lost original fragment regression source")
    endif()
    get_target_property(shards v2_test_cpu_shard_route_collective SOURCES)
    if(NOT "unit/execution/moe/Test__CPUJointServingMetadata.cpp" IN_LIST shards)
        message(FATAL_ERROR "Lost original shard metadata regression source")
    endif()
    foreach(target v2_test_native_cpu_collective_program v2_integration_cpu_expert_domain_lifecycle
                   v2_test_native_collective_owned_errors v2_unit_cpu_expert_socket_membership)
        if(NOT TARGET ${{target}})
            message(FATAL_ERROR "Lost functional fixture ${{target}}")
        endif()
    endforeach()
    get_property(preflight GLOBAL PROPERTY V2_UNIT_GATE_TARGETS)
    if(NOT "v2_test_native_collective_owned_errors" IN_LIST preflight)
        message(FATAL_ERROR "Lost original owned-error Unit gate dependency")
    endif()
endif()
''', encoding="utf-8")
            configured = subprocess.run(
                [self.cmake, "-S", str(root), "-B", str(project / "build"), "-G", "Ninja",
                 f"-DCMAKE_MAKE_PROGRAM={self.ninja}"], capture_output=True, text=True,
                timeout=15, check=False)
            if configured.returncode:
                return configured, None
            listed = subprocess.run(["ctest", "--test-dir", str(project / "build"),
                                     "--show-only=json-v1"], capture_output=True, text=True,
                                    timeout=10, check=True)
            return configured, json.loads(listed.stdout)["tests"]

    def test_backend_matrix_preserves_selected_targets_and_aliases(self):
        """CPU/CUDA/ROCm/both retain original Integration cells and valid Release performance commands."""
        for performance in (True, False):
            for cuda, rocm in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(performance=performance, cuda=cuda, rocm=rocm):
                    configured, tests = self.configure(performance, cuda, rocm)
                    self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
                    names = {test["name"] for test in tests}
                    expected_perf = {"V2_Perf_NativeModuleSelectionControl"}
                    for backend, present in (("CUDA", cuda), ("ROCm", rocm)):
                        if present:
                            expected_perf.update(f"V2_Perf_{backend}_PooledLocalHotNative_{case}"
                                                 for case in ("SmallProbeTiming",))
                    self.assertEqual({name for name in names if name.startswith("V2_Perf_")}, expected_perf)
                    if performance:
                        self.assertEqual(names, expected_perf)
                    else:
                        for expected in ("V2_Integration_CPUFragmentPayloadContract",
                                         "V2_Integration_NativeCPUCollectiveProgramReplay",
                                         "V2_Integration_CPUExpertSocketMembershipReplay",
                                         "V2_Integration_NativeCollectiveOwnedErrorReplay"):
                            self.assertIn(expected, names)
                    for test in tests:
                        if "_PooledLocalHotNative_" in test["name"]:
                            target = ("v2_perf_pooled_local_hot_native" if performance
                                      else "v2_integration_pooled_local_hot_native")
                            self.assertTrue(any(Path(arg).name == target for arg in test["command"]), test)

    def test_missing_integration_owner_still_fails(self):
        """A required missing owner remains an actual target_sources configure failure."""
        configured, tests = self.configure(False, False, False, missing_owner=True)
        self.assertNotEqual(configured.returncode, 0)
        self.assertIsNone(tests)
        self.assertIn('Cannot specify sources for target', configured.stderr)
        self.assertIn('v2_unit_moe_overlay_physical_residency_fabric', configured.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--modules-root", type=Path)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--ninja", default="ninja")
    args, remaining = parser.parse_known_args()
    if args.source_root:
        NativeTestModuleSelection.source_root = args.source_root.resolve()
    NativeTestModuleSelection.modules_root = args.modules_root
    NativeTestModuleSelection.cmake = args.cmake
    NativeTestModuleSelection.ninja = args.ninja
    unittest.main(argv=[__file__, *remaining])
