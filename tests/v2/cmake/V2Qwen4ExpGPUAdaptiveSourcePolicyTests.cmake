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
            --gtest_filter=Qwen4ExpSourceAdmissionCases.ConsumerLocalRecreationRetainsOriginalGPUOwners:Qwen4ExpSourceAdmissionCases.SharedContinuationReuseEligibilityPreservesFollowerAndOwnerRoles:Qwen4ExpSourceAdmissionCases.SharedContinuationMTPProjectionPreservesCanonicalMatchedSeeds
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

# Source declarations remain unqualified until their separate actual source/exact jobs execute.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_PooledCurrentDemandMTPInventory
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.PooledCurrentDemand*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;MTP;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledCurrentDemandMTPInventory)
endif()
add_v2_test(V2_Integration_ParityExplicitEnvironmentContract
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_source_admission_discovery.py
        SourceAdmissionDiscovery.test_explicit_environment_is_literal_for_focused_and_campaign_children
        SourceAdmissionDiscovery.test_explicit_environment_rejects_invalid_duplicate_and_canonical_override
        SourceAdmissionDiscovery.test_actual_target_timeout_participates_in_identity_and_post_build
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Discovery;Contract;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ParityExplicitEnvironmentContract)

# Functional generation discovery uses real originals and never acquires HF labels or a download fixture.
add_v2_test(V2_Integration_Qwen4ExpOrdinaryGenerationRegistration
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_source_admission_discovery.py
        --cmake ${CMAKE_COMMAND} --ninja ${CMAKE_MAKE_PROGRAM}
        SourceAdmissionDiscovery.test_ordinary_generation_uses_original_inputs_and_functional_labels
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;BuildSystem;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpOrdinaryGenerationRegistration)
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpOrdinaryGenerationProjection
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.OwnerSubsetOrdinaryGenerationKeepsOriginalOffRequests
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpOrdinaryGenerationProjection)
endif()

# Exact local-cache intent must survive canonical runtime export before native
# qualification; this registration makes no claim that a device path executed.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_QFNLocalHotRuntimeProjection
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.LocalHotCache*
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;Ngram;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNLocalHotRuntimeProjection)
endif()

# Focused declaration control; real source/ordinary/exact candidates remain hardware/model qualified separately.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpCombinedLocalHotC0ProjectionReplay
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.CombinedLocalHotC0CandidatesRetainOriginalRequests --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Ngram;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpCombinedLocalHotC0ProjectionReplay)
endif()

# Positive/unequal cases retain separate actual source/runtime obligations; metadata proves no performance claim.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpLocalHotBudgetProjectionReplay
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.LocalHotBudget* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Ngram;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpLocalHotBudgetProjectionReplay)
endif()
add_v2_test(V2_Integration_ParityLosslessRuntimeDocumentReplay
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/../unit/scripts/test_model_parity_inventory.py LosslessRuntimeDocumentTests
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Parity;Configuration;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ParityLosslessRuntimeDocumentReplay)

# Recurrent R3 candidates retain original source policies and lossless unequal-cache runtime exports.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpLocalHotRecurrentProjection
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.LocalHotRecurrent* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Parity;Inventory;Configuration;Ngram"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpLocalHotRecurrentProjection)
endif()

# CUDA combined-cache membership/configuration is a separate model-free inventory obligation.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpCUDALocalHotProjection
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.CUDALocalHot* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Parity;Inventory;Configuration;Ngram"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpCUDALocalHotProjection)
endif()

# GPU-only Q6 source/capture candidates retain exact original roles and configuration.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpAllGPUQ6LocalHotProjection
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.AllGPUQ6LocalHot* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Parity;Inventory;Configuration;Ngram"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpAllGPUQ6LocalHotProjection)
endif()

# Metadata/configuration validation only; actual complete-history panels remain separate.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpLocalHotNumericalProjection
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpLocalPairedCases.LocalHotNumericalPoliciesRequireExactPairedBindings --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Numerics;Inventory;Configuration"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpLocalHotNumericalProjection)
endif()

# Declaration/ownership proof only; actual SSD source and inference run separately.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpLocalHotSSDProjection
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.LocalHotSSDProjectionPreservesBoundedSourceAndConsumerOwnership --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Ngram;Configuration;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpLocalHotSSDProjection)
endif()

if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpLocalHotExplicitBudgetProjection
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.LocalHotExplicitBudgetCandidatesPreserveOriginalPolicies:Qwen4ExpSourceAdmissionCases.LocalHotRefinedBudgetPreservesMeasuredPolicies --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Ngram;Configuration;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpLocalHotExplicitBudgetProjection)
endif()

# Actual request declaration only; real movement remains the native/model obligation.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpOrdinaryDeclaredMovementWorkload
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.OrdinaryGenerationBindsDeclaredMovementHorizonAndInput --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Inventory;Configuration;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpOrdinaryDeclaredMovementWorkload)
endif()

if(TARGET v2_test_model_parity_definition)
    target_sources(v2_test_model_parity_definition PRIVATE unit/parity/Test__Qwen4ExpLocalHotHeldSourceCases.cpp)
    add_v2_test(V2_Integration_Qwen4ExpLocalHotHeldSourcePolicies
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpLocalHotHeldSourceCases.OriginalPoliciesRequirePoolC0AndOnePositiveOriginalConsumer --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Ngram;Configuration;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpLocalHotHeldSourcePolicies)
endif()
