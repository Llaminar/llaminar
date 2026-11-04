# @file V2TensorRawViewLifecycleTests.cmake
# @brief Device-free native raw-owner/view retirement gate; prepared engines remain independently owned.
if(TARGET v2_test_tensor_host_release)
    add_v2_test(V2_Integration_TensorRawViewLifecycle
        COMMAND $<TARGET_FILE:v2_test_tensor_host_release>
            --gtest_filter=Test__TensorHostRelease.AllNativeRawViewsRetireIndependentlyAndFollowRoot --gtest_brief=1
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;TensorOperations;Source;Lifecycle;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false")
    if(TEST V2_Integration_TensorRawViewLifecycle)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_TensorRawViewLifecycle)
    endif()
endif()
