# Native architecture replay shares the existing grouped projection owner.
if(HAVE_CUDA AND TARGET v2_integration_cuda_gemm_parity)
    add_v2_test(V2_Integration_CUDAVerifierDispatchTableNativeReplay
        COMMAND $<TARGET_FILE:v2_integration_cuda_gemm_parity>
            --gtest_filter=Test__CUDAGemmParity.GroupedVerifierDispatchTableNativeReplay
        LABELS "V2;Integration;ProductionTestPreflight;CUDA;GraphCapture;GroupedVerifierRows;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 60
        ENVIRONMENT "LLAMINAR_PERF_STATS_SUMMARY=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CUDAVerifierDispatchTableNativeReplay)
endif()
