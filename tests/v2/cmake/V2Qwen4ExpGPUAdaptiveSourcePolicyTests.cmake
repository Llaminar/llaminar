# Source-only typed discovery is distinct from numerical/expert movement qualification.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpGPUAdaptiveSourcePolicyDiscoveryReplay
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.GPUAdaptiveSourceHasTotalExpertPolicyAndMTPProduct:Qwen4ExpSourceAdmissionCases.GPUAdaptiveSourcePreservesOriginalSixteenAndNumericalFour:Qwen4ExpSourceAdmissionCases.GPUAdaptiveDynamicDepthDoesNotCertifyPhysicalExpertMovement:Qwen4ExpSourceAdmissionCases.FullSourceMatrixHasTotalFourPolicyMTPDimension:Qwen4ExpSourceAdmissionCI.GPUAdaptiveDiscoveryRetainsAllTypedExpertAndMTPPolicies
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Admission;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpGPUAdaptiveSourcePolicyDiscoveryReplay)
endif()

if(TARGET v2_test_qwen4exp_metadata)
    add_v2_test(V2_Integration_RetainedPredictorSourceReuseContract
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_metadata>
            --gtest_filter=ModelLoaderSharedPredictor.OriginalUnsplitSourceIdentityRejectsForeignAndMissingReuse:ModelLoaderSharedPredictor.RejectedPredictorBorrowPreservesReusableModelAndAuthority
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Admission;Ownership;Lifecycle;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RetainedPredictorSourceReuseContract)
endif()
