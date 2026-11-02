# Isolate fake canonical pool factories from every other native/unit fixture.
add_executable(v2_unit_read_ahead_native_resources
    unit/backends/Test__ReadAheadNativeResourceFamily.cpp)
if(TARGET v2_unit_read_ahead_native_resources)
    target_link_libraries(v2_unit_read_ahead_native_resources PRIVATE
        llaminar2_core GTest::gtest GTest::gtest_main Threads::Threads)
    v2_bind_gpu_process_test_entry(v2_unit_read_ahead_native_resources v2_core_gtest_main)
    target_include_directories(v2_unit_read_ahead_native_resources PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2")
    # Canonical directory binding replaces GTest main with the process-owned
    # entry; add_v2_test supplies the original process/session guardian.
    add_v2_test(V2_Unit_ReadAheadNativeResources
        COMMAND $<TARGET_FILE:v2_unit_read_ahead_native_resources>
        LABELS "V2;Unit;DeviceFree;Ownership;Memory;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    add_v2_test(V2_Integration_ReadAheadNativeResourceFamilyReplay
        COMMAND $<TARGET_FILE:v2_unit_read_ahead_native_resources>
            --gtest_filter=ReadAheadNativeResourceFamily.*:ReadAheadNativeParentTerminal.*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ownership;Memory;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    add_v2_test(V2_Integration_ReadAheadNativeProcessRuntimeReplay
        COMMAND $<TARGET_FILE:v2_unit_read_ahead_native_resources>
            --gtest_filter=ReadAheadNativeProcessRuntime.*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ownership;Memory;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
        V2_Integration_ReadAheadNativeResourceFamilyReplay V2_Integration_ReadAheadNativeProcessRuntimeReplay)
endif()
