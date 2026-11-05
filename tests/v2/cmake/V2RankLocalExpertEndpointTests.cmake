#[[
@file V2RankLocalExpertEndpointTests.cmake
@brief Register original same-rank expert endpoint identity and exact physical lane replay.

These gates are device-free protocol/planning proofs, not native follower
materialization or successful inference qualification.
]]
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

if(TARGET v2_integration_mapped_activation_packet_cuda_rocm)
    add_v2_test(V2_Integration_OriginalExpertSchedulerCancellationNativeReplay
        COMMAND $<TARGET_FILE:v2_integration_mapped_activation_packet_cuda_rocm>
            --gtest_filter=Test__MappedActivationPacketCUDAAndROCm.CUDAForkedPrefillAbortDrainsPendingAdmission:Test__MappedActivationPacketCUDAAndROCm.ROCmForkedPrefillAbortDrainsPendingAdmission
        LABELS "V2;Integration;ProductionTestPreflight;CUDA;ROCm;ExpertOverlay;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MODELS TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_OriginalExpertSchedulerCancellationNativeReplay)
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
