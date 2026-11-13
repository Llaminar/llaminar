# @file V2PooledRemoteGraphHostTests.cmake
# @brief Actual native-query bodies over recorded SDK metadata; no SDK is loaded at runtime.
if(NOT V2_PERF_TESTS_ONLY)
    set(_flat_graph_bodies "${CMAKE_CURRENT_BINARY_DIR}/generated/flat-graph-inspection/FlatGraphInspectionBodies.inc")
    add_custom_command(OUTPUT "${_flat_graph_bodies}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/generate_flat_graph_inspection_replay.py"
            --source "${REPO_ROOT}" --output "${_flat_graph_bodies}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/generate_flat_graph_inspection_replay.py"
            "${REPO_ROOT}/src/v2/backends/cuda/CUDAGraphCapture.cu"
            "${REPO_ROOT}/src/v2/backends/rocm/HIPGraphCapture.cpp"
        VERBATIM)
    add_executable(v2_test_gpu_flat_graph_inspection unit/backends/Test__GPUFlatGraphInspection.cpp "${_flat_graph_bodies}")
    if(TARGET v2_test_gpu_flat_graph_inspection)
        target_include_directories(v2_test_gpu_flat_graph_inspection PRIVATE "${REPO_ROOT}/src/v2" "${CMAKE_CURRENT_BINARY_DIR}/generated/flat-graph-inspection")
        target_link_libraries(v2_test_gpu_flat_graph_inspection PRIVATE GTest::gtest_main)
        target_compile_features(v2_test_gpu_flat_graph_inspection PRIVATE cxx_std_23)
        foreach(_flat_suite Unit Integration)
            set(_flat_gate "V2_${_flat_suite}_GPUFlatGraphInspectionReplay")
            set(_flat_labels "V2;${_flat_suite};DeviceFree;Ngram;Ownership;Replay;Regression")
            if(_flat_suite STREQUAL "Integration")
                list(APPEND _flat_labels ProductionTestPreflight)
                list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_flat_gate})
            endif()
            add_v2_test(${_flat_gate}
                COMMAND $<TARGET_FILE:v2_test_gpu_flat_graph_inspection> --gtest_repeat=20
                LABELS "${_flat_labels}" MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
        endforeach()
    endif()
    add_executable(v2_test_pooled_remote_graph_host unit/backends/Test__PooledRemoteGraphHostAllowance.cpp unit/backends/Test__PooledRemoteGraphHost.cpp integration/core_gtest_main.cpp)
    # Pooled immutable sources are CUDA/ROCm-native objects (no CPU
    # implementation); every registration requires a GPU backend family.
    if(TARGET v2_test_pooled_remote_graph_host)
        set_target_properties(v2_test_pooled_remote_graph_host PROPERTIES V2_TEST_REQUIRED_BACKEND GPU)
    endif()
    if(TARGET v2_test_pooled_remote_graph_host)
        target_compile_definitions(v2_test_pooled_remote_graph_host PRIVATE LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC=1)
        target_include_directories(v2_test_pooled_remote_graph_host PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        target_link_libraries(v2_test_pooled_remote_graph_host PRIVATE llaminar2_core GTest::gtest)
        target_compile_features(v2_test_pooled_remote_graph_host PRIVATE cxx_std_23)
        v2_test_device_scope(v2_test_pooled_remote_graph_host DeclaredBackends)
        add_v2_test(V2_Unit_PooledRemoteGraphHostReplay
            COMMAND $<TARGET_FILE:v2_test_pooled_remote_graph_host>
            LABELS "V2;Unit;DeviceFree;Ngram;Ownership;Replay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "GTEST_DEATH_TEST_STYLE=threadsafe")
        foreach(_allowance_case StartedBucketsAndCheckedArithmetic ActualC0PositiveAndLargerProgramsUseInstalledBackendFamily)
            set(_allowance_gate "V2_Integration_PooledRemoteGraphHostAllowance_${_allowance_case}Replay")
            add_v2_test(${_allowance_gate} COMMAND $<TARGET_FILE:v2_test_pooled_remote_graph_host> --gtest_filter=PooledRemoteGraphHostAllowance.${_allowance_case} --gtest_repeat=20
                LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;Replay;Regression"
                MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
                ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "GTEST_DEATH_TEST_STYLE=threadsafe")
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_allowance_gate})
        endforeach()
        # Serial source definitions retain the same original opaque native slot.
        add_v2_test(V2_Integration_PooledOriginalSourceSerialSlotRecreation
            COMMAND $<TARGET_FILE:v2_test_pooled_remote_graph_host>
                --gtest_filter=PooledRemoteGraphHost.OriginalSourceSerialDefinitionRecreationKeepsOneSlot
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;Ngram;Lifecycle;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;GTEST_DEATH_TEST_STYLE=threadsafe")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledOriginalSourceSerialSlotRecreation)
        set(_remote_host_cases SharedNativeChildKeepsExactOriginalOwnerUntilAllParentsRetire
            NativeGeneralSourceSlotsRetainPurposeAndOriginalIdentity NativeGeneralStorageCannotConsumeModelBudget
            StoragePurposeSurvivesFullyReservedModelAndFailedCreation FlatPreparationRejectsForeignAuthorityAndRetainsNativeFailures RecordedOriginalC0ClaimsSealReplayAndSurviveReset RecordedOriginalPositiveClaimsSealReplayAndSurviveReset
            WrongSourceAndFlatShapeRejectBeforeNativeInstantiation MissingPlannedBytesAndCreatorPrefixKeepPreciseOriginalOwnership
            ExplicitRawDescriptorUsesSameSealWithoutInventedHostLimit FamilySelectionKeepsExactCaptureAndAdmitsOnlyPlannedInstances)
        foreach(_remote_host_case IN LISTS _remote_host_cases)
            set(_remote_host_gate "V2_Integration_PooledRemoteGraphHost_${_remote_host_case}Replay")
            add_v2_test(${_remote_host_gate}
                COMMAND $<TARGET_FILE:v2_test_pooled_remote_graph_host> --gtest_filter=PooledRemoteGraphHost.${_remote_host_case} --gtest_repeat=20
                LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;Replay;Regression"
                MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
                ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "GTEST_DEATH_TEST_STYLE=threadsafe")
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_remote_host_gate})
        endforeach()
    endif()

    add_executable(v2_test_pooled_remote_family_scope unit/backends/Test__PooledRemoteFamilyScope.cpp)
    if(TARGET v2_test_pooled_remote_family_scope)
        target_include_directories(v2_test_pooled_remote_family_scope PRIVATE "${REPO_ROOT}/src/v2")
        target_link_libraries(v2_test_pooled_remote_family_scope PRIVATE llaminar2_core GTest::gtest_main)
        target_compile_features(v2_test_pooled_remote_family_scope PRIVATE cxx_std_23)
        foreach(_family_scope_suite Unit Integration)
            set(_family_scope_gate "V2_${_family_scope_suite}_PooledRemoteFamilyScopeReplay")
            set(_family_scope_labels "V2;${_family_scope_suite};DeviceFree;Ngram;Ownership;TraceReplay;Regression")
            if(_family_scope_suite STREQUAL "Integration")
                list(APPEND _family_scope_labels ProductionTestPreflight)
                list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_family_scope_gate})
            endif()
            add_v2_test(${_family_scope_gate} COMMAND $<TARGET_FILE:v2_test_pooled_remote_family_scope> --gtest_repeat=20
                LABELS "${_family_scope_labels}" MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
        endforeach()
    endif()

endif()
