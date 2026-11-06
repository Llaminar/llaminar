# @file V2RetainedParticipantSubmissionTests.cmake
# @brief Actual bounded recurring host enqueue ordering and reader ownership replay.
if(TARGET v2_test_physical_memory_authority)
    target_sources(v2_test_physical_memory_authority PRIVATE
        unit/execution/local_execution/graph/Test__RetainedGraphParticipantEnqueueGroup.cpp
        unit/execution/qwen4exp/Test__PooledImmutableParticipantExecutionPolicy.cpp)
    foreach(_case IN ITEMS
        RetainedGraphParticipantEnqueueGroup.ColdStorageValidationSealsOnceAndRejectsReplacedGraphs
        RetainedGraphParticipantEnqueueGroup.ColdStorageAdapterRetainsBorrowAcrossPartialNativeFailure
        RetainedGraphParticipantEnqueueGroup.ColdStorageAdapterUsesExactMatchedParentAndRetiresNativeFirst
        RetainedGraphParticipantEnqueueGroup.ColdStorageAdapterRejectsForeignAndPreservesAssemblyFailure
        RetainedGraphParticipantEnqueueGroup.EarlyBinderUsesActualImmutableGraphIdentity
        RetainedGraphParticipantEnqueueGroup.ColdDefinitionNamesEveryMissingDeclaration
        RetainedGraphParticipantEnqueueGroup.ColdDefinitionUsesOriginalMembershipAndGeneration
        RetainedGraphParticipantEnqueueGroup.ColdDefinitionFailureWakesOriginalMembersWithoutSeal
        RetainedGraphParticipantEnqueueGroup.SourceBeforeRemoteHostJoinResetAndRepeat
        RetainedGraphParticipantEnqueueGroup.PreDispatchRejectionOwnsNoNativeReader
        RetainedGraphParticipantEnqueueGroup.PartialDispatchAndMissingSourceCannotUnwindReaders
        RetainedGraphParticipantEnqueueGroup.PreparationStreamRejectsForeignAndRetiredSource
        RetainedGraphParticipantEnqueueGroup.ColdOriginalPreparationMarshalsOffWorkerAndRunsInlineOnWorker
        RetainedGraphParticipantEnqueueGroup.ColdOriginalPreparationRejectsForeignNullAndBrokenDispatch
        RetainedGraphParticipantEnqueueGroup.SetupRecorderUsesOriginalWorkerAndNeverPublishesExecution
        RetainedGraphParticipantEnqueueGroup.SetupRecorderFailureAbortsWithoutCompletion
        RetainedGraphParticipantEnqueueGroup.FaultedRemoteCannotLeaveSourceWaiting
        PooledImmutableParticipantExecutionPolicy.OffRetainsCapacityAndBoundsRequestSwitches
        PooledImmutableParticipantExecutionPolicy.ActiveCanonicalPoliciesRequireOriginalRetainedCapacity)
        string(REPLACE "." "_" _name "${_case}")
        add_v2_test(V2_Unit_${_name}
            COMMAND $<TARGET_FILE:v2_test_physical_memory_authority> --gtest_filter=${_case}
            LABELS "V2;Unit;DeviceFree;Memory;Ownership;Ngram;TraceReplay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;GTEST_DEATH_TEST_STYLE=threadsafe")
        add_v2_test(V2_Integration_${_name}Replay
            COMMAND $<TARGET_FILE:v2_test_physical_memory_authority> --gtest_filter=${_case} --gtest_repeat=20
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;Ngram;TraceReplay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;GTEST_DEATH_TEST_STYLE=threadsafe")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_${_name}Replay)
    endforeach()
    add_v2_test(V2_Integration_PooledSharedReaderExecutionDescriptor
        COMMAND $<TARGET_FILE:v2_test_physical_memory_authority>
            --gtest_filter=PooledImmutableParticipantExecutionPolicy.SharedReaderUsesCompleteDescriptorNotAllocationIdentity
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ownership;Ngram;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledSharedReaderExecutionDescriptor)
    add_v2_test(V2_Unit_PooledSharedReaderMaterialization
        COMMAND $<TARGET_FILE:v2_test_physical_memory_authority>
            --gtest_filter=PooledImmutableParticipantExecutionPolicy.SharedReadersRequireTransportWithoutRemoteStorage
        LABELS "V2;Unit;DeviceFree;Ownership;Ngram;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    add_v2_test(V2_Integration_PooledSharedReaderMaterialization
        COMMAND $<TARGET_FILE:v2_test_physical_memory_authority>
            --gtest_filter=PooledImmutableParticipantExecutionPolicy.SharedReadersRequireTransportWithoutRemoteStorage
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ownership;Ngram;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledSharedReaderMaterialization)
endif()

# Native endpoint capture remains independent from CPU/runtime collective capture support.
if(TARGET v2_test_forward_graph_types)
    foreach(_case CaptureHookExceptionPreservesStageContextAndUnwindsActivity ColdCaptureOriginalWorkerLifecycle NativeTransfersPreserveFortyEightCPUServiceCutpoints NativeTransferAuthorityKeepsReadinessAndRuntimeCollectiveGuards NativeTransferReplayRetainsCaptureStreamTerminalAuthority)
        add_v2_test(V2_Integration_${_case}Replay
            COMMAND $<TARGET_FILE:v2_test_forward_graph_types> --gtest_filter=Test__GraphSegmentCache.${_case} --gtest_repeat=20
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Graph;Ownership;Ngram;TraceReplay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_${_case}Replay)
    endforeach()
endif()

# Exact cold source-family association; backend doubles grant no native completion proof.
if(TARGET v2_test_physical_memory_authority)
    add_v2_test(V2_Integration_PooledExpertSourceFamilyLifecycle
        COMMAND $<TARGET_FILE:v2_test_physical_memory_authority>
            --gtest_filter=RetainedGraphParticipantEnqueueGroup.ExpertSourceFamily*:RetainedGraphParticipantEnqueueGroup.ExpertOwnerRequiresOriginalExecutionRole
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;Ngram;Lifecycle;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledExpertSourceFamilyLifecycle)
endif()

if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledExpertOriginalSourceFailureLifecycle
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=ImmutableLookupSubordinate.ExpertEnqueueFailureCancelsOriginalSourceBeforePartialDrain:ImmutableLookupSubordinate.ExpertAliasRefreshRejectsExecutionAdmission
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;Ngram;Lifecycle;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;GTEST_DEATH_TEST_STYLE=threadsafe")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledExpertOriginalSourceFailureLifecycle)
endif()

if(TARGET v2_test_forward_graph_types)
    add_v2_test(V2_Integration_QFNPredictorLocalTPCaptureWaveReplay
        COMMAND $<TARGET_FILE:v2_test_forward_graph_types>
            --gtest_filter=Test__GraphSegmentCache.QFNPredictorWaveAuthenticatesOriginalTransactionAcrossLocalTP:Test__GraphSegmentCache.NativeTransactionCaptureContractRetainsScopeAndCacheIdentity
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;MTP;GraphCapture;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNPredictorLocalTPCaptureWaveReplay)
endif()

# A successor stream consumes the exact predecessor event while aggregate host retirement stays deferred.
if(TARGET v2_test_forward_graph_types)
    add_v2_test(V2_Integration_CaptureTerminalDeviceHandoffLifecycle
        COMMAND $<TARGET_FILE:v2_test_forward_graph_types>
            --gtest_filter=Test__GraphSegmentCache.PublishedTerminalDeviceHandoffDoesNotObserveOrRerecord:Test__GraphSegmentCache.PublishedTerminalDeviceHandoffRejectsChangedOriginalOwners
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Graph;Lifecycle;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CaptureTerminalDeviceHandoffLifecycle)
endif()

# Exact singleton capture terminal creators use the original admitted source and PMA.
if(TARGET v2_test_native_ordering_event_group)
    add_v2_test(V2_Integration_CaptureTerminalOriginalSourceLifecycle
        COMMAND $<TARGET_FILE:v2_test_native_ordering_event_group>
            --gtest_filter=NativeOrderingOriginalSource.CaptureTerminalSingletonsUseOriginalPMAAndExactCreatorPopulation
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Graph;Memory;Lifecycle;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CaptureTerminalOriginalSourceLifecycle)
endif()

# Exact terminal receipts use only the original admitted publication-slot roster.
if(TARGET v2_test_forward_graph_types)
    add_v2_test(V2_Integration_OriginalSlotCaptureTerminalLifecycle
        COMMAND $<TARGET_FILE:v2_test_forward_graph_types>
            --gtest_filter=Test__GraphSegmentCache.OriginalSlotTerminalsRetainEveryDeferredInvocationLifecycle:Test__GraphSegmentCache.OriginalSlotTerminalsRejectForeignAndWithdrawnOwners:Test__GraphSegmentCache.OriginalSlotTerminalBOMUsesCertifiedUnitCount
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Graph;Lifecycle;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_OriginalSlotCaptureTerminalLifecycle)
endif()

# The actual hosted fragment value must name the complete retained verifier key.
if(TARGET v2_test_forward_graph_types)
    add_v2_test(V2_Integration_HostedVerifierForwardIdentity
        COMMAND $<TARGET_FILE:v2_test_forward_graph_types>
            --gtest_filter=HostedDeviceGenerationFragment.ExactVerifierIdentityAcrossFixedAndDynamicWidths
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MTP;Graph;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_HostedVerifierForwardIdentity)
endif()

# An ordinary hosted forward is selected by its key AND the original owned sampler.
if(TARGET v2_test_forward_graph_types)
    add_v2_test(V2_Integration_HostedOrdinaryForwardIdentity
        COMMAND $<TARGET_FILE:v2_test_forward_graph_types>
            --gtest_filter=HostedDeviceGenerationFragment.OrdinaryForwardRequiresOriginalSamplerAndCompleteKey
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Graph;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_HostedOrdinaryForwardIdentity)
endif()

# Actual graph/controller partition replay; metadata grants no native/source permit.
if(TARGET v2_test_forward_graph_types)
    add_v2_test(V2_Integration_RetainedParticipantAuxiliaryMainPartitionReplay
        COMMAND $<TARGET_FILE:v2_test_forward_graph_types>
            --gtest_filter=Test__GraphSegmentCache.RetainedParticipantAuxiliary*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Graph;Ngram;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RetainedParticipantAuxiliaryMainPartitionReplay)
endif()

# Probe is a graph-only compiler fork point inside the same original retained parent.
if(TARGET v2_test_forward_graph_types)
    add_v2_test(V2_Integration_RetainedParticipantAuxiliaryProbeFrontierReplay
        COMMAND $<TARGET_FILE:v2_test_forward_graph_types>
            --gtest_filter=Test__GraphSegmentCache.RetainedParticipantAuxiliaryProbeFrontierPartitionReplay:Test__GraphSegmentCache.RetainedParticipantAuxiliaryProbeEndIsNotSplitAgain:Test__GraphSegmentCache.RetainedParticipantAuxiliaryProbeRequiresLaterMainRead:Test__GraphSegmentCache.RetainedParticipantAuxiliarySplitTailHasSharedBoundaryIdentityReplay
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Graph;Ngram;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RetainedParticipantAuxiliaryProbeFrontierReplay)
endif()

# Main children cover the exact main-stage projection; only their complete parent owns the auxiliary body.
if(TARGET v2_test_forward_graph_types)
    add_v2_test(V2_Integration_RetainedParticipantAuxiliaryChildExportReplay
        COMMAND $<TARGET_FILE:v2_test_forward_graph_types>
            --gtest_filter=Test__GraphSegmentCache.RetainedParticipantAuxiliaryChildExport*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Graph;Ngram;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RetainedParticipantAuxiliaryChildExportReplay)
endif()

# MoE composition consumes the same main projection and revalidates the original disjoint body.
if(TARGET v2_test_forward_graph_types)
    add_v2_test(V2_Integration_RetainedParticipantAuxiliaryParentCompositionReplay
        COMMAND $<TARGET_FILE:v2_test_forward_graph_types>
            --gtest_filter=Test__GraphSegmentCache.RetainedParticipantAuxiliaryParentComposition*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Graph;Ngram;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RetainedParticipantAuxiliaryParentCompositionReplay)
endif()

# Model epoch release depends only on main work; original parent still joins finite auxiliary retirement.
if(TARGET v2_test_forward_graph_types)
    add_v2_test(V2_Integration_RetainedParticipantMainLeafProjectionReplay
        COMMAND $<TARGET_FILE:v2_test_forward_graph_types>
            --gtest_filter=Test__GraphSegmentCache.RetainedParticipantAuxiliaryMainLeafProjectionReplay
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Graph;Ngram;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RetainedParticipantMainLeafProjectionReplay)
endif()

# Actual Rank admission enters all original workers before any peer enqueue join.
if(TARGET v2_test_multi_device_orchestrator)
    add_v2_test(V2_Integration_RankRetainedStorageAdmissionFanoutReplay
        COMMAND $<TARGET_FILE:v2_test_multi_device_orchestrator>
            --gtest_filter=Test__RankOrchestratorDeviceGeneration.AdmissionEntersAllOriginalParticipantsBeforeJoiningReplay:Test__RankOrchestratorDeviceGeneration.AdmissionDrainsAllCallbacksBeforeFailureReplay
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;LocalTP;Ngram;Admission;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RankRetainedStorageAdmissionFanoutReplay)
endif()

# Executable export preserves full identities while projecting only authenticated auxiliary work from main units.
if(TARGET v2_test_forward_graph_types)
    add_v2_test(V2_Integration_RetainedParticipantAuxiliaryExecutableTemplateReplay
        COMMAND $<TARGET_FILE:v2_test_forward_graph_types>
            --gtest_filter=Test__GraphSegmentCache.RetainedParticipantAuxiliaryExecutableTemplate*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Graph;Ngram;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RetainedParticipantAuxiliaryExecutableTemplateReplay)
endif()

# Native graph-only compiler fragments still compose one original device-owned executable.
if(TARGET v2_test_forward_graph_types)
    add_v2_test(V2_Integration_NativeAuxiliaryOriginalParentReplay
        COMMAND $<TARGET_FILE:v2_test_forward_graph_types>
            --gtest_filter=Test__GraphSegmentCache.NativeAuxiliaryParent*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Graph;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NativeAuxiliaryOriginalParentReplay)
endif()
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_PooledOriginalHostedSourceContract
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.OriginalHostedSourceUsesActualMixedGPUContract
            --gtest_repeat=20 --gtest_break_on_failure
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Configuration;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledOriginalHostedSourceContract)
endif()

# Exact reader cache/source association before original expert-source discovery.
if(TARGET v2_test_physical_memory_authority)
    add_v2_test(V2_Integration_PooledAssociatedReaderSourceProjection
        COMMAND $<TARGET_FILE:v2_test_physical_memory_authority>
            --gtest_filter=RetainedGraphParticipantEnqueueGroup.ExpertSourceFamilyAssociatedReaderProjection
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ownership;Ngram;Lifecycle;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledAssociatedReaderSourceProjection)
endif()

# Initial request policy cannot remove the admitted exact source/follower rows.
if(TARGET v2_test_mtp_verifier_policy)
    add_v2_test(V2_Integration_MTPRetainedSourceCaptureEnvelope
        COMMAND $<TARGET_FILE:v2_test_mtp_verifier_policy>
            --gtest_filter=Test__MTPVerifierPolicy.OffAndActiveInitialPoliciesRetainActualSourceRows:Test__MTPVerifierPolicy.ExactHostedFamilyAdmissionIsIndependentOfActiveDepth
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;MTP;GraphCapture;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MTPRetainedSourceCaptureEnvelope)
endif()
if(TARGET v2_test_physical_memory_authority)
    add_v2_test(V2_Integration_PooledRetainedSourceRowsAcrossInitialPolicies
        COMMAND $<TARGET_FILE:v2_test_physical_memory_authority>
            --gtest_filter=RetainedGraphParticipantEnqueueGroup.ExpertSourceFamilyRetainedRowsAcrossInitialPolicies
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;MTP;Ownership;Ngram;Lifecycle;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledRetainedSourceRowsAcrossInitialPolicies)
endif()

# Original prepared-model event handles survive serial runner replacement.
if(TARGET v2_test_native_ordering_event_group)
    add_v2_test(V2_Integration_PreparedFollowerTerminalRecreation
        COMMAND $<TARGET_FILE:v2_test_native_ordering_event_group>
            --gtest_filter=NativeOrderingOriginalSource.PreparedFollowerRecreationRetainsOneOriginalEventPopulation:NativeOrderingOriginalSource.PreparedFollowerPartialCreatorFailureCannotAllocateReplacement
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;Replay;Lifecycle;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;GTEST_DEATH_TEST_STYLE=threadsafe")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PreparedFollowerTerminalRecreation)
endif()
