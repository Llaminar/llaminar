# Actual production enumeration body; native entrypoints are device-free doubles.
# The private fixture include path never propagates to serving/backend targets.
foreach(V2_ROCM_ENUMERATION_VARIANT IN ITEMS failures original)
    add_executable(v2_unit_rocm_enumeration_${V2_ROCM_ENUMERATION_VARIANT}
        unit/backends/Test__ROCmEnumerationFailures.cpp)
    target_include_directories(v2_unit_rocm_enumeration_${V2_ROCM_ENUMERATION_VARIANT} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/rocm_enumeration"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2/backends"
        "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_features(v2_unit_rocm_enumeration_${V2_ROCM_ENUMERATION_VARIANT} PRIVATE cxx_std_23)
    target_compile_definitions(v2_unit_rocm_enumeration_${V2_ROCM_ENUMERATION_VARIANT} PRIVATE HAVE_ROCM)
    target_link_libraries(v2_unit_rocm_enumeration_${V2_ROCM_ENUMERATION_VARIANT} PRIVATE GTest::gtest_main Threads::Threads)
endforeach()
target_compile_definitions(v2_unit_rocm_enumeration_original PRIVATE
    ROCM_ENUMERATION_BODY="fixtures/rocm_enumeration/original_ROCmEnumeration.inc")
add_v2_test(V2_Unit_ROCmEnumerationFailures
    COMMAND $<TARGET_FILE:v2_unit_rocm_enumeration_failures>
    LABELS "V2;Unit;DeviceFree;ROCm;Enumeration;Ownership;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
add_v2_test(V2_Integration_ROCmEnumerationFailureReplay
    COMMAND $<TARGET_FILE:v2_unit_rocm_enumeration_failures>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;ROCm;Enumeration;Ownership;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
add_v2_test(V2_Unit_ROCmEnumerationOriginalFailureControls
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_rocm_enumeration_original_controls.py"
        --binary $<TARGET_FILE:v2_unit_rocm_enumeration_original>
    LABELS "V2;Unit;DeviceFree;ROCm;Enumeration;Ownership;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
add_v2_test(V2_Integration_ROCmEnumerationOriginalFailureControls
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_rocm_enumeration_original_controls.py"
        --binary $<TARGET_FILE:v2_unit_rocm_enumeration_original>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;ROCm;Enumeration;Ownership;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_ROCmEnumerationFailureReplay
    V2_Integration_ROCmEnumerationOriginalFailureControls)
