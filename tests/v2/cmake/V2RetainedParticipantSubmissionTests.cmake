# @file V2RetainedParticipantSubmissionTests.cmake
# @brief Actual bounded recurring host enqueue ordering and reader ownership replay.
if(TARGET v2_test_physical_memory_authority)
    target_sources(v2_test_physical_memory_authority PRIVATE
        unit/execution/local_execution/graph/Test__RetainedGraphParticipantEnqueueGroup.cpp
        unit/execution/qwen4exp/Test__PooledImmutableParticipantExecutionPolicy.cpp)
    foreach(_case IN ITEMS
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
endif()

# Native endpoint capture remains independent from CPU/runtime collective capture support.
if(TARGET v2_test_forward_graph_types)
    foreach(_case NativeTransfersPreserveFortyEightCPUServiceCutpoints NativeTransferAuthorityKeepsReadinessAndRuntimeCollectiveGuards NativeTransferReplayRetainsCaptureStreamTerminalAuthority)
        add_v2_test(V2_Integration_${_case}Replay
            COMMAND $<TARGET_FILE:v2_test_forward_graph_types> --gtest_filter=Test__GraphSegmentCache.${_case} --gtest_repeat=20
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Graph;Ownership;Ngram;TraceReplay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_${_case}Replay)
    endforeach()
endif()
