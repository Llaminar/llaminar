#[=[
@file V2PooledDirectedTransportTests.cmake
@brief Focused source-owned directed geometry, native backing and cold transport lifecycle regressions.
]=]
if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledCompactTerminalCounterUnits
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=QwenAdaptiveTerminalCounters.CompactMissCountsKeepOriginalUnitsAndUnpaddedBounds
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Counters;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledCompactTerminalCounterUnits)
endif()
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_PooledCompactTerminalCounterEvidence
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpGPUAdaptiveLookupEvidence.CompactInputsPreserveExactCountsAndRejectRelabeledEvidence
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Counters;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledCompactTerminalCounterEvidence)
endif()
if(TARGET v2_test_qwen4exp_primitives)
    target_sources(v2_test_qwen4exp_primitives PRIVATE
        unit/kernels/qwen4exp/Test__PooledImmutableDirectedTransport.cpp)
    add_v2_test(V2_Integration_PooledImmutableDirectedTransportReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableDirectedTransport.*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Transfer;LiveExtent;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledImmutableDirectedTransportReplay)
endif()

if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledNativeMappedBackingReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableDirectedTransport.NativeCompletionMappedBackingKeepsLogicalSpanSeparate
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Transfer;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNativeMappedBackingReplay)
endif()
if(TARGET v2_test_captured_transfer_channel_binding)
    add_v2_test(V2_Integration_CapturedChannelColdInitializationReplay
        COMMAND $<TARGET_FILE:v2_test_captured_transfer_channel_binding>
            --gtest_filter=CapturedTransferChannelBindingTest.ColdInitialization*:CapturedTransferChannelBindingTest.PartialSetup*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Transfer;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CapturedChannelColdInitializationReplay)
endif()

if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_PooledNoPeerInventoryReplay
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=DeclaredGPUCachePlacement.ExplicitNoPeerUsesOriginalInventoryAndSurvivesWireWithoutComputeRoles
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Transfer;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNoPeerInventoryReplay)
endif()
if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledNoPeerNativeOwnerGeometryReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableDirectedTransport.StorageOnlyNativeMembersHaveExactBOMWithoutModelReaderRoles:PooledImmutableDirectedTransport.NativeUnionPreservesReaderOrderAndExcludesStorageOnlyResults
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Transfer;Memory;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNoPeerNativeOwnerGeometryReplay)
endif()
if(TARGET v2_test_physical_memory_authority)
    add_v2_test(V2_Integration_PooledNativeOwnerAdmissionContractReplay
        COMMAND $<TARGET_FILE:v2_test_physical_memory_authority>
            --gtest_filter=PooledImmutableDirectedOwnerProgram.NativeGeneralPreservesOriginalGeometryWithoutFlatCertificate
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Transfer;Memory;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNativeOwnerAdmissionContractReplay)
endif()

# Exact static completion-bank population used by native inline continuation owners.
if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledNativeCompletionLayoutReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableDirectedTransport.NativeCompletionLayoutHasExactPositiveMessagePopulation:PooledImmutableDirectedTransport.NativeCompletionC0RetainsOnlyOriginalReaderResult:PooledImmutableDirectedTransport.NativeCompletionAbsentViewRejectsMatching:PooledImmutableDirectedTransport.NativeCompletionOperandCannotBeReboundAcrossVariants
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Transfer;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNativeCompletionLayoutReplay)
endif()

if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledNativeOriginalAbortReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableNative.OriginalNativeAbortPreservesPinnedDutiesAndEarlierOwnerFaults
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Transfer;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNativeOriginalAbortReplay)
endif()

# Recorded fixed-header/Completed-payload arithmetic; native capture remains a separate gate.
if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledNativeReaderReceiptReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableCoordinator.NativeFixedHeaderPreservesOriginalFailureReplay:PooledImmutableCoordinator.NativeReaderFreshReceiptAndCurrentACKReplay
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Transfer;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNativeReaderReceiptReplay)
endif()

if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledNativeOriginalTicketAbortReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=ImmutableLookupSubordinate.OriginalNativeAbortBindingRejectsSuccessorAndPreservesCPUOutcome
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Transfer;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "GTEST_DEATH_TEST_STYLE=threadsafe")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNativeOriginalTicketAbortReplay)
    add_v2_test(V2_Integration_PooledNativeInlineBOMReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableDirectedTransport.NativeInlineOwnersKeepExactProtocolFloorAndNoSecondExecutable
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Transfer;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNativeInlineBOMReplay)
endif()
if(TARGET v2_test_memory_planner)
    add_v2_test(V2_Integration_PooledNativeAllContinuationRolesReplay
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=RankMemoryPlanInputs.SharedAllContinuationOwnersUseOriginalNativeTP2AcrossPeerCoverage
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Transfer;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNativeAllContinuationRolesReplay)
endif()

# Immutable parent ownership classification grants no native readiness.
if(TARGET v2_test_qwen4exp_metadata)
    add_v2_test(V2_Integration_PooledNativeParentMaterializationReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_metadata>
            --gtest_filter=Qwen4ExpReplicatedPLECohort.SharedNativeFactoryRequiresItsOriginalRankOwnedMaterialization
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNativeParentMaterializationReplay)
endif()

# Unsupported original native cohort geometry rejects before any worker/materialization.
if(TARGET v2_test_memory_planner)
    add_v2_test(V2_Integration_PooledOutsideNativeCohortAdmissionContract
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=RankMemoryPlanInputs.SharedOutsideNativeCohortOwnerRejectsAtAdmission:RankMemoryPlanInputs.SharedAllContinuationOwnersUseOriginalNativeTP2AcrossPeerCoverage:RankMemoryPlanInputs.SharedCoordinatorCUDATP2KeepsOppositeVendorStorageAndNoPeerReader
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledOutsideNativeCohortAdmissionContract)
endif()

# Cold graph construction uses real original source/owner endpoints; no native execution claim.
if(TARGET v2_test_pooled_owner_pin_chain)
    add_v2_test(V2_Integration_PooledCurrentDemandGraphLifecycle
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py
            $<TARGET_FILE:v2_test_pooled_owner_pin_chain> --gtest_filter=PooledImmutableCurrentDemand.*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Transfer;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledCurrentDemandGraphLifecycle)
endif()

# Owner subsets retain original native reader/materialization authority; no GPU work.
if(TARGET v2_test_memory_planner)
    add_v2_test(V2_Integration_PooledNativeOwnerSubsetPlanningReplay
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=RankMemoryPlanInputs.SharedOwnerSubsetsPreserveOriginalNativeReadersAcrossBackends
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Transfer;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNativeOwnerSubsetPlanningReplay)
endif()
if(TARGET v2_test_qwen4exp_metadata)
    add_v2_test(V2_Integration_PooledNativeOwnerSubsetMaterializationReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_metadata>
            --gtest_filter=Qwen4ExpReplicatedPLECohort.SharedOwnerSubsetsKeepParentMaterializationAndExactPhysicalGeometry
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNativeOwnerSubsetMaterializationReplay)
endif()
if(TARGET v2_test_qwen4exp_metadata)
    add_v2_test(V2_Integration_PooledNativeOwnerSubsetSourceGeometryReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_metadata>
            --gtest_filter=ImmutableRowTableLayoutGeometry.SharedOwnerSubsetSourceProjectionSeparatesReadersFromRemoteOwners
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNativeOwnerSubsetSourceGeometryReplay)
endif()

if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledNativeOwnerSubsetBOMReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableDirectedTransport.NativeOwnerSubsetsKeepExactOriginalCompletionBOM
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNativeOwnerSubsetBOMReplay)
endif()

# Addressed native transport is qualified independently of two-endpoint geometry.
if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledAddressedNativeCompletionBOMReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableDirectedTransport.Addressed*
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Transfer;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 60
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledAddressedNativeCompletionBOMReplay)
endif()
if(TARGET v2_test_memory_planner)
    add_v2_test(V2_Integration_PooledAddressedNativeOriginalPlanningReplay
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=RankMemoryPlanInputs.AddressedSharedOwnersRetainOriginalReadersEveryDegree
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Transfer;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledAddressedNativeOriginalPlanningReplay)
endif()
if(TARGET v2_test_qwen4exp_metadata)
    add_v2_test(V2_Integration_PooledAddressedNativeParentMaterializationReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_metadata>
            --gtest_filter=Qwen4ExpReplicatedPLECohort.AddressedReadersEveryDegreeKeepOriginalParentMaterialization
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Transfer;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledAddressedNativeParentMaterializationReplay)
endif()

foreach(_addressed_backend IN ITEMS nccl rccl)
    if(TARGET v2_unit_${_addressed_backend}_rows_abi)
        add_v2_test(V2_Integration_PooledAddressed_${_addressed_backend}_NativeABIReplay
            COMMAND $<TARGET_FILE:v2_unit_${_addressed_backend}_rows_abi>
                --gtest_filter=NativeRowsABI.AddressedBytesUseOnlyActualOriginalEndpointsAndExactOperands
                --gtest_repeat=20
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Collective;Transfer;Ownership;TraceReplay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledAddressed_${_addressed_backend}_NativeABIReplay)
    endif()
endforeach()

foreach(_addressed_backend IN ITEMS nccl rccl)
    if(TARGET v2_unit_${_addressed_backend}_path_configured)
        string(TOUPPER "${_addressed_backend}" _addressed_upper)
        add_v2_test(V2_Integration_PooledAddressed_${_addressed_backend}_NativeAdmissionReplay
            COMMAND $<TARGET_FILE:v2_unit_${_addressed_backend}_path_configured>
                --gtest_filter=${_addressed_upper}ConfiguredLibrary.AddressedExportsRejectPartialAdmissionAndWithdrawCallbacks
                --gtest_repeat=20
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Collective;Transfer;Ownership;TraceReplay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledAddressed_${_addressed_backend}_NativeAdmissionReplay)
    endif()
endforeach()

# Same genuine prior/count authority for grouped pooled current lookup; no device work.
if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_Qwen4ExpPooledCurrentVerifierReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PLELifecycle.EarlyCurrentPriorHashRetainReplay:PLELifecycle.EarlyCurrentPriorRejectsSubstitutedAuthority
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;MTP;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpPooledCurrentVerifierReplay)
endif()
if(TARGET v2_test_pooled_owner_pin_chain)
    add_v2_test(V2_Integration_PooledCurrentVerifierAuthorityContract
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py
            $<TARGET_FILE:v2_test_pooled_owner_pin_chain>
            --gtest_filter=PooledImmutableCurrentDemand.GroupedPrefixRejectsMissingPriorAndSubstitutedProjectionBeforeMutation
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;MTP;Ownership;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledCurrentVerifierAuthorityContract)
endif()

# The real source family cold-builds all original scalar MTP condition purposes.
if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_Qwen4ExpPooledCurrentConditionReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PLELifecycle.EarlyCurrentConditionUsesOriginalScalarAcceptedHistory
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;MTP;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpPooledCurrentConditionReplay)
endif()

# Compact local-hot messages retain the original pool coalescer and receipt law.
# This component proof is device-free; it advertises no native graph or model qualification.
if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledLocalHotCompactConsumerContract
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableCoordinator.LocalHot* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Transfer;LiveExtent;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledLocalHotCompactConsumerContract)
endif()

# Request88/Lane24 completion is a distinct typed purpose, never a pool interval.
if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledNativeRequestCompletionBindingReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableDirectedTransport.NativeRequestCompletionFramePurposeAndOwnershipReplay:PooledImmutableDirectedTransport.NativeRequestCompletionRejectsMalformedAndAliasedFieldsReplay:PooledImmutableDirectedTransport.NativeRequestCompletionSubspanKeepsOriginalRoleAndBoundsReplay
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Transfer;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNativeRequestCompletionBindingReplay)
endif()

# The original source admission and current request lane remain distinct from the pool interval.
if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledLocalHotRequestPublicationReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableCoordinator.LocalHotRequestPublicationRejectsStaleAdmissionAndKeepsEmptyProgress
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Transfer;LiveExtent;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledLocalHotRequestPublicationReplay)
endif()

# Storage ownership is independent from compute width; the original planner must
# fit every feasible native owner while retaining exact protocol growth.
if(TARGET v2_test_memory_planner)
    add_v2_test(V2_Integration_PooledStorageOnlyOwnerAdmissionReplay
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=GPUAdaptiveImmutableLookupAdmission.SharedDirectoryPressureBalancesEveryFeasibleOwner
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "GTEST_DEATH_TEST_STYLE=threadsafe")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledStorageOnlyOwnerAdmissionReplay)
endif()
