# @file V2NativeEventDestructionTests.cmake
# @brief Compile original cleanup bodies without a native runtime, retaining canonical source dependencies.
# Ninja normalizes '..' in output paths; CMake preserves spelling in missing-input
# phony rules. Share the normalized source identity with the native source catalog.
if(NOT V2_PERF_TESTS_ONLY)
get_filename_component(_event_replay_source "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(_event_replay_generated "${CMAKE_CURRENT_BINARY_DIR}/native_event_destruction")
set(_event_replay_generator "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/generate_native_event_destruction_replay.py")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_event_replay_source}/src/v2/backends/cuda/NvidiaDeviceContext.cu"
    "${_event_replay_source}/src/v2/backends/cuda/CUDABackend.cu"
    "${_event_replay_source}/src/v2/backends/rocm/AMDDeviceContext.cpp"
    "${_event_replay_source}/src/v2/backends/rocm/ROCmBackend.cpp"
    "${_event_replay_generator}")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${_event_replay_generator}"
    --source "${_event_replay_source}" --output "${_event_replay_generated}/NativeEventDestructionBodies.inc"
    RESULT_VARIABLE _event_replay_result)
if(NOT _event_replay_result EQUAL 0)
    message(FATAL_ERROR "Actual native event destruction replay extraction failed")
endif()
add_executable(v2_unit_native_event_destruction "${CMAKE_CURRENT_SOURCE_DIR}/unit/backends/Test__NativeEventDestruction.cpp")
target_include_directories(v2_unit_native_event_destruction PRIVATE "${_event_replay_generated}")
target_compile_features(v2_unit_native_event_destruction PRIVATE cxx_std_20)
target_link_libraries(v2_unit_native_event_destruction PRIVATE GTest::gtest_main Threads::Threads)
add_v2_test(V2_Unit_NativeEventDestruction
    COMMAND $<TARGET_FILE:v2_unit_native_event_destruction>
    LABELS "V2;Unit;DeviceFree;CUDA;ROCm;Ownership;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
add_v2_test(V2_Integration_NativeEventDestructionReplay
    COMMAND $<TARGET_FILE:v2_unit_native_event_destruction>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CUDA;ROCm;Ownership;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NativeEventDestructionReplay)
endif()
