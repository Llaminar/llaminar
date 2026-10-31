# Narrow source ownership plus actual POSIX signal replay; no driver or hardware.
add_executable(v2_unit_gpu_cleanup_failure_qualification
    unit/backends/Test__GPUCleanupFailureQualification.cpp)
target_compile_features(v2_unit_gpu_cleanup_failure_qualification PRIVATE cxx_std_20)
target_compile_definitions(v2_unit_gpu_cleanup_failure_qualification PRIVATE
    LLAMINAR_TEST_V2_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(v2_unit_gpu_cleanup_failure_qualification PRIVATE GTest::gtest_main)
add_v2_test(V2_Unit_GPUCleanupFailureQualification
    COMMAND $<TARGET_FILE:v2_unit_gpu_cleanup_failure_qualification>
    LABELS "V2;Unit;DeviceFree;Lifecycle;Signals;Ownership;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
add_v2_test(V2_Integration_GPUCleanupFailureQualificationReplay
    COMMAND $<TARGET_FILE:v2_unit_gpu_cleanup_failure_qualification>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Lifecycle;Signals;Ownership;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_GPUCleanupFailureQualificationReplay)
