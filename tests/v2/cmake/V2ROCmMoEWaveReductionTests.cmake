# Replay the exact production helper with original-tree controls on real hardware.
if(HAVE_ROCM)
    set_source_files_properties(integration/kernels/rocm/Test__ROCmMoEWaveReduction.hip PROPERTIES LANGUAGE HIP)
    add_executable(v2_integration_rocm_moe_wave_reduction
        integration/kernels/rocm/Test__ROCmMoEWaveReduction.hip)
    if(LLAMINAR_SHARED_CORE)
        target_link_libraries(v2_integration_rocm_moe_wave_reduction PRIVATE
            llaminar2_core GTest::gtest_main hip::host)
    else()
        target_link_libraries(v2_integration_rocm_moe_wave_reduction PRIVATE
            -Wl,--start-group llaminar2_core -Wl,--whole-archive rocm_backend -Wl,--no-whole-archive
            -Wl,--end-group GTest::gtest_main hip::host)
    endif()
    set_target_properties(v2_integration_rocm_moe_wave_reduction PROPERTIES
        HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON V2_TEST_DEVICE_SCOPE FullInventory)
    add_v2_test(V2_Integration_ROCmMoEWaveReductionNativeReplay
        COMMAND $<TARGET_FILE:v2_integration_rocm_moe_wave_reduction>
            --gtest_filter=ROCmMoEWaveReductionNative.CapturedProductionSumUsesCanonicalAscendingXorOrder
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;Router;GraphCapture;ByteExact;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 60
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=0")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ROCmMoEWaveReductionNativeReplay)
endif()
