#[=[
@file V2TypedHistogramOwnershipTests.cmake
@brief Replay complete-target publication through the original logical histogram banks.
]=]
add_v2_test(V2_Integration_DecodeExpertHistogramCompleteTargetOwnershipReplay
    COMMAND $<TARGET_FILE:v2_test_decode_expert_histogram>
        --gtest_filter=Test__DecodeExpertHistogram.CompleteTargetOwnershipPublicationReplay
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;ExpertOverlay;Ownership;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_DecodeExpertHistogramCompleteTargetOwnershipReplay)
