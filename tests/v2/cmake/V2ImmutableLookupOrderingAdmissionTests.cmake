# Append after the existing complete cohort declarations; preserve Release selection.
if(TARGET v2_test_pooled_complete_cohort)
    target_sources(v2_test_pooled_complete_cohort PRIVATE
        unit/execution/Test__ImmutableLookupOrderingAdmission.cpp)
    add_v2_test(V2_Integration_ImmutableLookupOrderingAdmissionReplay
        COMMAND $<TARGET_FILE:v2_test_pooled_complete_cohort>
            --gtest_filter=ImmutableLookupOrderingAdmission.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ImmutableLookupOrderingAdmissionReplay)
endif()
