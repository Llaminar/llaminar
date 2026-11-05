#[=[
@file V2NativeCollectiveOwnedErrorTests.cmake
@brief Build and replay actual native host error paths without GPU or SDK execution.

The builder extracts exact coordinator/backend/LocalTP method bodies. The device-
free executable supplies only recorded control fields and runtime entrypoint
responses; no core/backend object or native readiness is fabricated. Original
shared-error bodies remain explicit deferred-readback negative controls.
]=]
include_guard(DIRECTORY)
set(V2_NATIVE_ERROR_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/native_owned_error_replay")
set(V2_NATIVE_ERROR_GENERATED "${V2_NATIVE_ERROR_GENERATED_DIR}/NativeCollectiveDiagnosticBodies.inc")
set(V2_NATIVE_ERROR_GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/scripts/extract_native_collective_error_bodies.py")
add_custom_command(OUTPUT "${V2_NATIVE_ERROR_GENERATED}"
    COMMAND ${Python3_EXECUTABLE} -B "${V2_NATIVE_ERROR_GENERATOR}"
        --source "${REPO_ROOT}" --output "${V2_NATIVE_ERROR_GENERATED}"
    DEPENDS "${V2_NATIVE_ERROR_GENERATOR}"
        "${REPO_ROOT}/src/v2/collective/LocalTPContext.cpp"
        "${REPO_ROOT}/src/v2/collective/backends/NCCLBackend.cpp"
        "${REPO_ROOT}/src/v2/collective/backends/RCCLBackend.cpp"
        "${REPO_ROOT}/src/v2/collective/coordinators/NCCLCoordinator.cu"
        "${REPO_ROOT}/src/v2/collective/coordinators/RCCLCoordinator.cpp"
    VERBATIM)
add_executable(v2_test_native_collective_owned_errors
    unit/collective/Test__NativeCollectiveOwnedErrors.cpp "${V2_NATIVE_ERROR_GENERATED}")
target_compile_features(v2_test_native_collective_owned_errors PRIVATE cxx_std_20)
target_include_directories(v2_test_native_collective_owned_errors PRIVATE
    "${REPO_ROOT}/src/v2" "${V2_NATIVE_ERROR_GENERATED_DIR}")
target_link_libraries(v2_test_native_collective_owned_errors PRIVATE GTest::gtest_main Threads::Threads)
set_property(GLOBAL APPEND PROPERTY V2_UNIT_GATE_TARGETS v2_test_native_collective_owned_errors)
foreach(V2_NATIVE_ERROR_SUITE IN ITEMS Unit Integration)
    set(V2_NATIVE_ERROR_TEST "V2_${V2_NATIVE_ERROR_SUITE}_NativeCollectiveOwnedErrorReplay")
    add_v2_test("${V2_NATIVE_ERROR_TEST}"
        COMMAND $<TARGET_FILE:v2_test_native_collective_owned_errors> --gtest_filter=NativeCollectiveOwnedErrors.*
        MPI_PROCS 1 NO_MPI NO_MODELS LABELS "V2;${V2_NATIVE_ERROR_SUITE};DeviceFree;Collective;Replay;Regression" TIMEOUT 10)
    if(V2_NATIVE_ERROR_SUITE STREQUAL "Integration")
        set_property(TEST "${V2_NATIVE_ERROR_TEST}" APPEND PROPERTY LABELS ProductionTestPreflight)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS "${V2_NATIVE_ERROR_TEST}")
    endif()
endforeach()
