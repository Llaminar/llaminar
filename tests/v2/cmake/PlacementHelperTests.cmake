# Placement helper isolated launch-chain tests (bounded, approved scope).
# The helper binary is built and co-located with the rank runtime; the tests
# exec the ACTUAL helper across the agreed per-rank boundary and assert the
# typed failure branches. NO production launcher wiring exists.
if(V2_PERF_TESTS_ONLY)
    return()
endif()

set(_placement_helper_env
    "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")

add_executable(v2_placement_helper_child_check unit/planning/PlacementHelperChildCheck.cpp)
target_compile_features(v2_placement_helper_child_check PRIVATE cxx_std_20)
target_link_libraries(v2_placement_helper_child_check PRIVATE llaminar_placement_topology)

add_executable(v2_placement_helper_native_child unit/planning/PlacementHelperNativeChild.cpp)
target_compile_features(v2_placement_helper_native_child PRIVATE cxx_std_20)
# The native child deliberately links OpenMP: it verifies the ACTUAL libgomp
# constructor contract after the helper's exec.
target_link_libraries(v2_placement_helper_native_child PRIVATE llaminar2_core llaminar_placement_topology OpenMP::OpenMP_CXX)

add_executable(v2_test_placement_helper_launch unit/planning/Test__PlacementHelperLaunch.cpp)
target_compile_features(v2_test_placement_helper_launch PRIVATE cxx_std_20)
target_link_libraries(v2_test_placement_helper_launch PRIVATE llaminar_placement_topology GTest::gtest_main)
target_compile_definitions(v2_test_placement_helper_launch PRIVATE
    LLAMINAR_PLACEMENT_HELPER_PATH="$<TARGET_FILE:llaminar_placement_helper>"
    LLAMINAR_PLACEMENT_CHILD_PATH="$<TARGET_FILE:v2_placement_helper_child_check>"
    LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH="$<TARGET_FILE:v2_placement_helper_native_child>")
# The launch chain execs these binaries by absolute path at runtime; make the
# test target depend on them so a test build always produces fresh candidates.
add_dependencies(v2_test_placement_helper_launch
    llaminar_placement_helper v2_placement_helper_child_check
    v2_placement_helper_native_child)

add_v2_test(V2_Unit_PlacementHelperLaunch
    COMMAND $<TARGET_FILE:v2_test_placement_helper_launch>
    LABELS "V2;Unit;CPU;Placement;ProcessOwned;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 120 ENVIRONMENT "${_placement_helper_env}")

# Diagnostic MPI launches explicitly compose the placement helper and its
# observing child. Product placement and staged sibling closure use the
# native frontend/guardian/component path, not this helper.
add_executable(v2_test_placement_integration integration/app/Test__PlacementIntegration.cpp)
target_compile_features(v2_test_placement_integration PRIVATE cxx_std_20)
target_link_libraries(v2_test_placement_integration PRIVATE llaminar2_core llaminar_placement_topology GTest::gtest_main)
target_compile_definitions(v2_test_placement_integration PRIVATE
    LLAMINAR_PLACEMENT_HELPER_PATH="$<TARGET_FILE:llaminar_placement_helper>"
    LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH="$<TARGET_FILE:v2_placement_helper_native_child>"
    LLAMINAR_MPI_SESSION_PATH="$<TARGET_FILE:llaminar_mpi_session>")
add_dependencies(v2_test_placement_integration
    llaminar_placement_helper v2_placement_helper_native_child
    llaminar_mpi_session)
add_v2_test(V2_Integration_PlacementWiring
    COMMAND $<TARGET_FILE:v2_test_placement_integration>
    LABELS "V2;Integration;CPU;Placement;MPI;ProcessOwned;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 300
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
