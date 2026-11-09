# Shared PLE query-only graphs borrow an original source cohort, never a fake CPU channel.
# This device-free actual-builder rejection complements full native cohort activation;
# it proves missing source authority fails before mutating the participant graph.
if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_SharedPLEReaderModelAuthority REQUIRES_BACKEND GPU
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=Qwen4ExpPLEBuilder.SharedReaderRequiresOriginalCohortBeforeGraphMutation
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1"
            "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_SharedPLEReaderModelAuthority)
endif()
