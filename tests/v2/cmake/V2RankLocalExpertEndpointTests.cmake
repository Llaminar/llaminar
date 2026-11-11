#[[
@file V2RankLocalExpertEndpointTests.cmake
@brief Register original same-rank expert endpoint identity and exact physical lane replay.

These gates are device-free protocol/planning proofs, not native follower
materialization or successful inference qualification.
]]
if(TARGET v2_unit_moe_overlay_participant_residency)
    add_v2_test(V2_Integration_MoEDeferredContinuationInitialBanks
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_participant_residency>
            --gtest_filter=Test__MoEOverlayParticipantResidencyRegistry.FinalizesDeferredContinuationAndRetainedSidecarBeforeAuxiliaryReadiness:Test__MoEOverlayParticipantResidencyRegistry.FinalizesDormantLayersFromModelOwnedPreparedEngines
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;MoE;Ownership;Lifecycle;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
            "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEDeferredContinuationInitialBanks)
endif()

if(TARGET v2_unit_moe_overlay_inference_transaction)
    add_v2_test(V2_Integration_RankLocalExpertEndpointOwnershipReplay
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_inference_transaction>
            --gtest_filter=Test__MoEOverlayInferenceTransaction.RankLocalEndpointIdentityRetainsGraphFamiliesAndRejectsForeignMembership:Test__MoEOverlayInferenceTransaction.RankLocalExpertsShareOriginalCoordinatorWithoutAliasingDenseSegments:Test__MoEOverlayInferenceTransaction.RankLocalGPUExpertParticipationUsesFrozenAdmittedTiersAndOriginalRoles:Test__MoEOverlayInferenceTransaction.LocalEndpointRetirementAttemptsEveryPeerAndPreservesFirstNativeFailure:Test__MoEOverlayInferenceTransaction.RankLocalExpertSubsetRequiresOriginalFullRankRegistryAndEpoch
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;MoE;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1"
            "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1"
            "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RankLocalExpertEndpointOwnershipReplay)
endif()
if(TARGET v2_unit_moe_overlay_capacity_resolver)
    add_v2_test(V2_Integration_RankLocalExpertEndpointPhysicalBOMReplay
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_capacity_resolver>
            --gtest_filter=MoEOverlayLocalCapacityPlanner.RankLocalGPUActivationBOMKeepsOnePhysicalOwnerAndExactEndpointGrants
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;MoE;Memory;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1"
            "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1"
            "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RankLocalExpertEndpointPhysicalBOMReplay)
endif()

# Actual callable publisher protocol; recorded native outcomes are not GPU proof.
if(TARGET v2_unit_moe_overlay_inference_transaction)
    foreach(_publisher_case RankLocalCallablePublisherRetiresExactIdentityAcrossCommands
            RankLocalCallablePublisherPartialSubmitRequiresOriginalDrain
            RankLocalCallablePublisherJoinFailureStaysFailedUntilAllPeerDrain
            RankLocalCallablePublisherRejectsStaleAbortBeforeNativeDrain
            RankLocalCallablePublisherKeepsQueuedSuccessUntilOriginalJoins
            RankLocalCallablePublisherCancelsDelayedFailureBeforeAnyJoin)
        set(_publisher_gate "V2_Integration_${_publisher_case}Replay")
        add_v2_test(${_publisher_gate}
            COMMAND $<TARGET_FILE:v2_unit_moe_overlay_inference_transaction>
                --gtest_filter=Test__MoEOverlayInferenceTransaction.${_publisher_case} --gtest_repeat=20
            LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;MoE;Ownership;TraceReplay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
                "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_publisher_gate})
    endforeach()
endif()

if(TARGET v2_unit_moe_overlay_activation_epoch_protocol)
    add_v2_test(V2_Integration_OriginalExpertSchedulerCancellationReplay
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_activation_epoch_protocol>
            --gtest_filter=Test__MoEOverlayActivationEpochProtocol.SchedulerCancellationPreservesOriginalEndpointErrorsAndRejectsForeignIdentity:Test__MoEOverlayActivationEpochProtocol.SchedulerCancellationIsConcurrentAbsorbingAndPreservesWatchdogReason:Test__MoEOverlayActivationEpochProtocol.SchedulerCancellationBeforeActivationDoesNotForgeGPUCompletion
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;MoE;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1"
            "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1"
            "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_OriginalExpertSchedulerCancellationReplay)
endif()



if(TARGET v2_unit_moe_overlay_inference_transaction)
    add_v2_test(V2_Integration_RankLocalExpertSubmissionResetReplay
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_inference_transaction>
            --gtest_filter=Test__MoEOverlayInferenceTransaction.RankLocalCallablePublisherResetRetainsQueuedAndUnprovenOriginalWork
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;MoE;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1"
            "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RankLocalExpertSubmissionResetReplay)
endif()

# Original host sequence ownership, including pending publication and terminal failures.
if(TARGET v2_unit_moe_overlay_residency_authority)
    add_v2_test(V2_Integration_MoEGraphSequenceDescendantLifecycle
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_residency_authority>
            --gtest_filter=Test__MoEOverlayResidencyAuthority.GraphSequenceDescendantsDrainReservedPublication:Test__MoEOverlayResidencyAuthority.GraphSequenceDescendantsRejectInvalidParents
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Lifecycle;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEGraphSequenceDescendantLifecycle)
endif()
if(TARGET v2_unit_moe_overlay_inference_transaction)
    add_v2_test(V2_Integration_MoEFinitePlacementScopeLifecycle
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_inference_transaction>
            --gtest_filter=Test__MoEOverlayInferenceTransaction.FinitePlacementScope* --gtest_repeat=20 --gtest_brief=1
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Lifecycle;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEFinitePlacementScopeLifecycle)
endif()

# Original host SDK enqueues precede continuation; native execution remains asynchronous.
if(TARGET v2_unit_moe_overlay_inference_transaction)
    add_v2_test(V2_Integration_RankLocalOriginalEnqueueBeforeSourceReplay
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_inference_transaction>
            --gtest_filter=Test__MoEOverlayInferenceTransaction.RankLocalOriginalEnqueuesCompleteBeforeContinuationReplay
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ownership;MoE;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RankLocalOriginalEnqueueBeforeSourceReplay)
endif()

# Outer command admission authenticates original sealed rank identity even without MPI peers.
if(TARGET v2_unit_moe_overlay_inference_transaction)
    add_v2_test(V2_Integration_RankLocalRootCommandAdmissionReplay
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_inference_transaction>
            --gtest_filter=Test__MoEOverlayInferenceTransaction.RootCommandAdmissionUsesOriginalPlanAndGeneration:Test__MoEOverlayInferenceTransaction.RankLocalExpertsShareOriginalCoordinatorWithoutAliasingDenseSegments
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;MoE;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RankLocalRootCommandAdmissionReplay)
endif()

if(TARGET v2_unit_moe_overlay_inference_transaction)
    add_v2_test(V2_Integration_RankLocalOriginalMTPPublicationRingReplay
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_inference_transaction>
            --gtest_filter=Test__MoEOverlayInferenceTransaction.RankLocalCallablePublisherRetainsOriginalMTPRing:Test__MoEOverlayInferenceTransaction.RankLocalCallablePublisherMTPRingRetainsPartialFailure:Test__MoEOverlayInferenceTransaction.RankLocalCallablePublisherMTPRingRejectsCapacityOverrun:Test__MoEOverlayInferenceTransaction.RankLocalCallablePublisherAbortRetainsSourceFailureAfterNativeProof
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;MoE;MTP;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RankLocalOriginalMTPPublicationRingReplay)
endif()

# Native resource ownership follows the original accepted transaction slots.
add_v2_test(V2_Integration_MoEOriginalNativeSlotOwnershipReplay
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_inference_transaction>
        --gtest_filter=Test__MoEOverlayInferenceTransaction.OriginalNativeSlots*
        --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Ownership;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOriginalNativeSlotOwnershipReplay)

# One original K-slot native metadata roster per participant, never per family.
if(TARGET v2_unit_moe_overlay_node_local_rank_batch_transport)
    add_v2_test(V2_Integration_MoEOriginalNativeSlotLayoutReplay
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_node_local_rank_batch_transport>
            --gtest_filter=Test__MoEOverlayNodeLocalRankBatchTransport.OriginalNativeSlotLayout*
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOriginalNativeSlotLayoutReplay)
endif()

# Describe only immutable original admission; native packet retirement remains a separate gate.
if(TARGET v2_unit_moe_overlay_activation_epoch_protocol)
    add_v2_test(V2_Integration_MoEOriginalNativeAdmissionIdentityReplay
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_activation_epoch_protocol>
            --gtest_filter=Test__MoEOverlayActivationEpochProtocol.OriginalNativeAdmission*
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOriginalNativeAdmissionIdentityReplay)
endif()

# Actual cold owner and three-phase aggregate retirement, including absorbing Abort.
if(TARGET v2_unit_moe_overlay_inference_transaction)
    add_v2_test(V2_Integration_RankLocalOriginalAggregateRetirementReplay
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_inference_transaction>
            --gtest_filter=Test__MoEOverlayInferenceTransaction.RankLocalColdOriginalSlotsRejectQueuedAdmissionBeforeDrain:Test__MoEOverlayInferenceTransaction.RankLocalOriginalAggregateObservesEverySlotBeforeRetiringAny:Test__MoEOverlayInferenceTransaction.OriginalNativeSlotsSerializePublicationWithAbortReplay
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RankLocalOriginalAggregateRetirementReplay)
endif()
