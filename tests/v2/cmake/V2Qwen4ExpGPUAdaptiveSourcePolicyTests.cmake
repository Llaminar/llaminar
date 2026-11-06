#[[
@file V2Qwen4ExpGPUAdaptiveSourcePolicyTests.cmake
@brief Register original GPU-adaptive source, ownership, reuse and exhaustive subset selector replays.

These device-free contracts do not qualify the separately discovered real-model
source/numerical/exact candidates or absent native hardware.
]]
# Source-only typed discovery is distinct from numerical/expert movement qualification.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpAllGPUQ6MTPRegistrationContract
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.AllGPUQ6* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;MTP;Admission;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpAllGPUQ6MTPRegistrationContract)
endif()

if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpGPUAdaptiveSourcePolicyDiscoveryReplay
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.ExpertLookupOwnerSelectionRequiresOriginalKnownRank:Qwen4ExpSourceAdmissionCases.ExpertFollowerOwnersRetainOriginalQ6AndAllMTPPolicies:Qwen4ExpSourceAdmissionCases.GPUAdaptiveSourceHasTotalExpertPolicyAndMTPProduct:Qwen4ExpSourceAdmissionCases.GPUAdaptiveSourcePreservesOriginalSixteenAndNumericalFour:Qwen4ExpSourceAdmissionCases.GPUAdaptiveDynamicDepthDoesNotCertifyPhysicalExpertMovement:Qwen4ExpSourceAdmissionCases.FullSourceMatrixHasTotalFourPolicyMTPDimension:Qwen4ExpSourceAdmissionCI.GPUAdaptiveDiscoveryRetainsAllTypedExpertAndMTPPolicies
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Admission;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpGPUAdaptiveSourcePolicyDiscoveryReplay)
endif()

# Missing rank placement is legitimate for original expert-only CPU followers.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpSourceReuseEligibilityContract
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.SharedContinuationReuseEligibilityPreservesFollowerAndOwnerRoles:Qwen4ExpSourceAdmissionCases.SharedContinuationMTPProjectionPreservesCanonicalMatchedSeeds
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Admission;Ownership;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpSourceReuseEligibilityContract)
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

# Shared source/reader and embedded expert evidence use actual terminal schemas.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpSharedAdaptiveInferenceEvidenceContract
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpGPUAdaptiveLookupEvidence.SharedCoordinatorRetainsSoleSourceAndExactReaderRoles:Qwen4ExpGPUAdaptiveLookupEvidence.NativeSubmissionsExcludeActualEmbeddedOwners:Qwen4ExpGPUAdaptiveLookupEvidence.EveryEmbeddedExpertRequiresOriginalCompletedUsefulWork
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Ownership;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpSharedAdaptiveInferenceEvidenceContract)
endif()

# Metadata enumerates every mask; the separately discovered40/40/30 real-model
# candidate cells require actual source, numerical and native exact execution.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpOwnerSubsetSourceSelectionReplay
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.OwnerSubset* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Admission;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpOwnerSubsetSourceSelectionReplay)
endif()
