# @file V2FiniteStorageParentTests.cmake
# @brief Exact original native finite-body ownership and SDMA/progress composition replay.
if(TARGET v2_test_compute_graph_execution)
    foreach(_case CapturedProducerForkDoesNotGateLaterConsumerUnit CapturedProducerMappingRejectsStaleForeignAndAmbiguousDefinitions CapturedProducerForkRejectsUnknownOrDuplicateNodesBeforeMutation FiniteSiblingWithoutProgressPreservesOriginalMainAndAllTerminals FiniteSiblingRejectsEmptySelfDuplicateBeforeMutation)
        add_v2_test(V2_Integration_FiniteStorage_${_case}Replay
            COMMAND $<TARGET_FILE:v2_test_compute_graph_execution> --gtest_filter=NativeParallelGraphBranch.${_case} --gtest_repeat=20
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Graph;Ownership;TraceReplay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FiniteStorage_${_case}Replay)
    endforeach()
endif()
if(TARGET v2_unit_gpu_expert_transfer)
    target_include_directories(v2_unit_gpu_expert_transfer PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach(_backend CUDA ROCm)
        string(TOUPPER "${_backend}" _key)
        if(HAVE_${_key})
            foreach(_case Live0 Live1 Live257 Live8193 RejectsOriginalForeignDevice)
                set(_hardware_label "${_backend}")
                if(_case STREQUAL "RejectsOriginalForeignDevice")
                    set(_hardware_label "${_backend}2")
                endif()
                add_v2_test(V2_Integration_${_backend}_FiniteStorage_${_case}
                    COMMAND $<TARGET_FILE:v2_unit_gpu_expert_transfer> --gtest_filter=FiniteBodySiblingNativeIntegration.${_backend}_${_case}
                    LABELS "V2;Integration;ProductionTestPreflight;GPU;${_backend};${_hardware_label};Graph;Ownership;TraceReplay;Regression"
                    MPI_PROCS 1 NO_MODELS TIMEOUT 30)
                list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_${_backend}_FiniteStorage_${_case})
            endforeach()
        endif()
    endforeach()
endif()
if(NOT V2_PERF_TESTS_ONLY AND Python3_EXECUTABLE)
    add_v2_test(V2_Integration_FiniteStorageTargetSelection
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_test_target_selection.py"
            --cmake "${CMAKE_COMMAND}" --ninja "${CMAKE_MAKE_PROGRAM}"
            TestTargetSelection.test_finite_storage_cases_keep_original_aggregate_and_owned_processes
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CMake;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FiniteStorageTargetSelection)
endif()
