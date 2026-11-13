#[[
@file V2CUDANativeProgramStackTests.cmake
@brief Original-source native-program stack capacity and actual NCCL capture replay.
No timing threshold or variable-length lookup qualification is implied by the
fixed-payload native replay. The original source late-growth rejection remains.
]]
if(TARGET v2_test_physical_memory_authority)
    add_v2_test(V2_Integration_CUDANativeProgramStackContract
        COMMAND $<TARGET_FILE:v2_test_physical_memory_authority>
            --gtest_filter=NativeContextPreparation.CUDANativeProgramStackCapacityContract:NativeContextPreparation.ActualOwnerSamePMAAdmissionAndRepeatedBorrow
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;NativeContext;Contract;Lifecycle;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CUDANativeProgramStackContract)
endif()
if(HAVE_CUDA AND TARGET v2_integration_captured_transfer_channel)
    add_v2_test(V2_Integration_CUDANativeProgramStackReplay
        COMMAND $<TARGET_FILE:v2_integration_captured_transfer_channel>
            --gtest_filter=CapturedTransferNativeProgramStack.CUDAOriginalSourceReplay
        LABELS "V2;Integration;ProductionTestPreflight;CUDA;GPU;NativeCollective;GraphCapture;Replay;Regression"
        MPI_PROCS 2 NO_MODELS TIMEOUT 30 RESOURCE_LOCK "cuda_gpu_0;cuda_gpu_1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CUDANativeProgramStackReplay)
endif()
