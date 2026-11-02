# Actual source-consumption predicate replay, never native/model qualification.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpConsumedNativeSourceReplay
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.ConsumedNativeSourceRejectsAbsentOriginalPreparation:Qwen4ExpSourceAdmissionCases.ConsumedNativeSourceRejectsPreboundUnusedOriginalAuthority
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Admission;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpConsumedNativeSourceReplay)
endif()
