# @file V2RetainedStorageRequestWorkerTests.cmake
# @brief Original request publications/whole publisher worker scope; no native runtime is used.
if(NOT V2_PERF_TESTS_ONLY)
    add_executable(v2_test_retained_storage_request_worker
        unit/execution/qwen4exp/Test__RetainedStorageRequestWorker.cpp integration/core_gtest_main.cpp)
    target_include_directories(v2_test_retained_storage_request_worker PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(v2_test_retained_storage_request_worker PRIVATE llaminar2_core GTest::gtest)
    target_compile_features(v2_test_retained_storage_request_worker PRIVATE cxx_std_23)
    v2_test_device_scope(v2_test_retained_storage_request_worker DeclaredBackends)
    add_v2_test(V2_Unit_RetainedStorageRequestWorker
        COMMAND $<TARGET_FILE:v2_test_retained_storage_request_worker>
        LABELS "V2;Unit;DeviceFree;Ngram;Ownership;RequestReset;GenerationController;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "GTEST_DEATH_TEST_STYLE=threadsafe")
    foreach(_request_worker_case RankMainConditionArmsOriginalGroupBeforeFanoutAndAbortsFailures RankRetiresEveryOriginalPeerAndGroupWhilePreservingFirstAwaitFailure SourceGroupMembershipPreservesAuthoredTPOrderAndOriginalOwnership EveryMainSourceRoleArmsBeforeLiveStateMutation ActiveParentBindingKeepsOriginalStreamAndRejectsForeignAuthority ActivePoliciesReuseOriginalWholePublisherAndTerminalPredicates BrokenPublisherDispatchRejectsBeforeRecursiveCapture AdmissionDispatchesOriginalOffAndOnWorkerAndPropagatesFailures
            GroupedResetPublishesEveryOriginalCloseBeforeFirstServiceAwait
            ResetClosesOnWorkerAndAggregatesOnlyAfterOriginalTerminalEvent
            WholePublisherUsesOriginalWorkerAndRejectsForeignContextBeforeCapture)
        set(_request_worker_gate "V2_Integration_RetainedStorageRequestWorker_${_request_worker_case}Replay")
        add_v2_test(${_request_worker_gate}
            COMMAND $<TARGET_FILE:v2_test_retained_storage_request_worker>
                --gtest_filter=RetainedStorageRequestWorker.${_request_worker_case} --gtest_repeat=20
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;RequestReset;GenerationController;Replay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "GTEST_DEATH_TEST_STYLE=threadsafe")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_request_worker_gate})
    endforeach()
endif()
