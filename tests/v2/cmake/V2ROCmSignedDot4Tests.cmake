# Native signed-byte arithmetic is a distinct first-class implementation on
# gfx906 and gfx1100. The device regression replays the actual shared helper.
if(HAVE_ROCM)
    set_source_files_properties(integration/kernels/rocm/Test__ROCmSignedDot4.hip
        PROPERTIES LANGUAGE HIP)
    add_executable(v2_integration_rocm_signed_dot4
        integration/kernels/rocm/Test__ROCmSignedDot4.hip)
    target_link_libraries(v2_integration_rocm_signed_dot4 PRIVATE
        llaminar2_core GTest::gtest GTest::gtest_main hip::host ${CMAKE_DL_LIBS})
    # Observe calls in a shared backend as well as static linkage; forward to the exact SDK runtime.
    target_link_options(v2_integration_rocm_signed_dot4 PRIVATE "LINKER:--export-dynamic-symbol=hipDeviceGetAttribute")
    set_target_properties(v2_integration_rocm_signed_dot4 PROPERTIES
        HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
    add_v2_test(V2_Integration_ROCmSignedDot4NativeReplay
        COMMAND $<TARGET_FILE:v2_integration_rocm_signed_dot4>
            --gtest_filter=ROCmSignedDot4.CapturedSignedModuloLaneReplay
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;Capture;ByteExact;Regression;INT8"
        MPI_PROCS 1 NO_MODELS TIMEOUT 120)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
        V2_Integration_ROCmSignedDot4NativeReplay)
endif()

if(TARGET v2_integration_rocm_signed_dot4)
    add_v2_test(V2_Integration_ROCmStreamWaitValueCapabilityNativeReplay
        COMMAND $<TARGET_FILE:v2_integration_rocm_signed_dot4>
            --gtest_filter=ROCmHardwareMetadataNative.StreamWaitValueCapabilityUsesAdmittedFactsWithoutRediscovery
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;HardwareMetadata;StreamTimeline;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 60
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=0")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ROCmStreamWaitValueCapabilityNativeReplay)
endif()

# Actual production Q4_0 arithmetic replay; widening variant is diagnostic only.
if(HAVE_ROCM)
    set_source_files_properties(integration/kernels/rocm/Test__ROCmQ4ScaleContribution.hip PROPERTIES LANGUAGE HIP)
    target_sources(v2_integration_rocm_signed_dot4 PRIVATE integration/kernels/rocm/Test__ROCmQ4ScaleContribution.hip)
    add_v2_test(V2_Integration_ROCmQ4ScaleContributionNativeReplay
        COMMAND $<TARGET_FILE:v2_integration_rocm_signed_dot4>
            --gtest_filter=ROCmQ4ScaleContribution.CapturedProductionAndExplicitWideningDiagnostic
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;Capture;ByteExact;Regression;Q4_0"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=0")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ROCmQ4ScaleContributionNativeReplay)
endif()
