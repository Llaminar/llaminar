## @file V2PooledNgramPerformanceTests.cmake
## @brief Retain the actual pooled measurement target, original native trajectories and device-free group declarations.
# The same actual pooled fixture serves correctness and explicit Release timing.
# Include before canonical process-entry finalization; root owns the top-level include.
if(NOT V2_PERF_TESTS_ONLY)
    add_executable(v2_test_pooled_ngram_replay_trace
        unit/kernels/qwen4exp/Test__PooledNgramReplayTrace.cpp)
    target_include_directories(v2_test_pooled_ngram_replay_trace PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_features(v2_test_pooled_ngram_replay_trace PRIVATE cxx_std_20)
    target_link_libraries(v2_test_pooled_ngram_replay_trace PRIVATE GTest::gtest_main)
    add_v2_test(V2_Unit_PooledNgramReplayTrace
        COMMAND $<TARGET_FILE:v2_test_pooled_ngram_replay_trace>
        LABELS "V2;Unit;CPU;DeviceFree;Ngram;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    add_v2_test(V2_Integration_PooledNgramReplayTrace
        COMMAND $<TARGET_FILE:v2_test_pooled_ngram_replay_trace>
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;Ngram;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNgramReplayTrace)
endif()
if(HAVE_CUDA OR HAVE_ROCM)
    add_executable(v2_perf_pooled_ngram_lookup
        performance/kernels/qwen4exp/Measure__PooledNgramLookup.cpp)
    target_include_directories(v2_perf_pooled_ngram_lookup PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/integration/kernels/qwen4exp")
    target_link_libraries(v2_perf_pooled_ngram_lookup PRIVATE llaminar2_core GTest::gtest)
    target_compile_definitions(v2_perf_pooled_ngram_lookup PRIVATE $<$<CONFIG:Release>:LLAMINAR_NGRAM_PERF_RELEASE=1>)
    # Original diagnostic helper definitions are shared with the correctness TU;
    # no new kernel or modified device function is introduced for measurement.
    if(HAVE_CUDA)
        target_sources(v2_perf_pooled_ngram_lookup PRIVATE integration/kernels/qwen4exp/PooledACKReadyCancellationCUDA.cu)
    endif()
    if(HAVE_ROCM)
        target_sources(v2_perf_pooled_ngram_lookup PRIVATE integration/kernels/qwen4exp/PooledACKReadyCancellationROCm.hip)
        set_target_properties(v2_perf_pooled_ngram_lookup PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
        target_link_libraries(v2_perf_pooled_ngram_lookup PRIVATE hip::host)
    endif()
    set(_ngram_placements)
    if(HAVE_CUDA)
        list(APPEND _ngram_placements LocalCUDA)
    endif()
    if(HAVE_ROCM)
        list(APPEND _ngram_placements LocalROCm)
    endif()
    if(HAVE_CUDA AND HAVE_ROCM)
        list(APPEND _ngram_placements CUDAConsumerROCmStorage ROCmConsumerCUDAStorage)
    endif()
    foreach(_placement IN LISTS _ngram_placements)
        if(_placement MATCHES "^Local")
            set(_kind local)
        else()
            set(_kind cross_vendor)
        endif()
        if(_placement STREQUAL "LocalCUDA" OR _placement STREQUAL "CUDAConsumerROCmStorage")
            set(_source cuda:0)
        else()
            set(_source rocm:0)
        endif()
        if(_kind STREQUAL "local")
            set(_owner "${_source}")
        elseif(_source STREQUAL "cuda:0")
            set(_owner rocm:0)
        else()
            set(_owner cuda:0)
        endif()
        set(_skip_cuda 1)
        set(_skip_rocm 1)
        set(_locks)
        set(_backends)
        foreach(_endpoint "${_source}" "${_owner}")
            if(_endpoint STREQUAL "cuda:0")
                set(_skip_cuda 0)
                list(APPEND _locks cuda_gpu_0)
                list(APPEND _backends CUDA)
            else()
                set(_skip_rocm 0)
                list(APPEND _locks rocm_gpu_0)
                list(APPEND _backends ROCm)
            endif()
        endforeach()
        list(REMOVE_DUPLICATES _locks)
        list(REMOVE_DUPLICATES _backends)
        foreach(_tokens 1 8 64 512)
            math(EXPR _ids "16 * ${_tokens}")
            foreach(_scenario warm_hits alternating_sets mixed churn empty duplicate_control)
                set(_slots "${_ids}")
                if(_scenario STREQUAL "churn")
                    math(EXPR _slots "${_ids} / 2")
                endif()
                add_v2_test(V2_Perf_PooledNgram_${_placement}_M${_tokens}_${_scenario}
                    COMMAND $<TARGET_FILE:v2_perf_pooled_ngram_lookup>
                        --placement ${_kind} --continuation ${_source} --owner ${_owner}
                        --tokens ${_tokens} --capacity-tokens ${_tokens} --gpu-slots ${_slots}
                        --ram-slots 2 --locality scattered --scenario ${_scenario} --samples 5 --table-rows 1048576
                    LABELS "V2;Performance;Ngram;Pooled;NativeReplay;${_backends};JSON"
                    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30 RESOURCE_LOCK "${_locks}"
                    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=${_skip_cuda};LLAMINAR_SKIP_ROCM_STARTUP=${_skip_rocm};LLAMINAR_LOG_LEVEL=ERROR")
            endforeach()
        endforeach()
    endforeach()
endif()

# Inherited fixture backing must remain under the existing process guardian.
if(TARGET v2_test_qwen4exp_device_primitives)
    add_v2_test(V2_Integration_PooledProviderOwnedTemporaryBacking
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_device_primitives>
            --gtest_filter=PooledImmutableProviderSourceDevice.ProviderBackingUsesOwnedTemporaryRoot
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;Ngram;ProcessOwned;Storage;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledProviderOwnedTemporaryBacking)
endif()

# Exact actual authored-remote duplicate trajectory regression; no device or SDK required.
if(TARGET v2_test_pooled_ngram_replay_trace)
    add_v2_test(V2_Integration_AuthoredRemoteDuplicateShapeReplay
        COMMAND $<TARGET_FILE:v2_test_pooled_ngram_replay_trace>
            --gtest_filter=PooledNgramReplayTrace.AuthoredRemoteDuplicateHeavyRetainsFullAdmittedCapacity --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_AuthoredRemoteDuplicateShapeReplay)
endif()

# Same performance executable replays only its cold CLI/layout declaration here.
# No original native source is prepared and no model or accelerator is queried.
if(TARGET v2_perf_pooled_ngram_lookup AND NOT V2_PERF_TESTS_ONLY)
    foreach(_consumer_declaration_suite Unit Integration)
        add_v2_test(V2_${_consumer_declaration_suite}_PooledConsumerLocalEconomyDeclarations
            COMMAND $<TARGET_FILE:v2_perf_pooled_ngram_lookup> --self-test
            LABELS "V2;${_consumer_declaration_suite};CPU;DeviceFree;Ngram;ConsumerLocal;Replay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    endforeach()
    set_property(TEST V2_Integration_PooledConsumerLocalEconomyDeclarations APPEND PROPERTY LABELS ProductionTestPreflight)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledConsumerLocalEconomyDeclarations)
endif()

# Four discoverable actual shared-source cells retain the existing source trace,
# original process owner and two request intervals on the same captured graphs.
# Performance is deliberately separate from production-preflight correctness.
if(TARGET v2_perf_pooled_ngram_lookup)
    foreach(_group_backend CUDA ROCm)
        if((_group_backend STREQUAL "CUDA" AND HAVE_CUDA) OR
           (_group_backend STREQUAL "ROCm" AND HAVE_ROCM))
            string(TOLOWER "${_group_backend}" _group_cli)
            set(_group_skip_cuda 1)
            set(_group_skip_rocm 1)
            if(_group_backend STREQUAL "CUDA")
                set(_group_skip_cuda 0)
            else()
                set(_group_skip_rocm 0)
            endif()
            foreach(_group_mode C0Alternating PositiveChurn)
                if(_group_mode STREQUAL "C0Alternating")
                    set(_group_slots 0)
                    set(_group_scenario alternating_sets)
                else()
                    set(_group_slots 64)
                    set(_group_scenario churn)
                endif()
                add_v2_test(V2_Perf_PooledNgram_ConsumerLocal${_group_backend}2_${_group_mode}ResetReplay
                    COMMAND $<TARGET_FILE:v2_perf_pooled_ngram_lookup>
                        --placement consumer_local --continuation ${_group_cli}:0
                        --consumers ${_group_cli}:0,${_group_cli}:1
                        --tokens 8 --capacity-tokens 64 --gpu-slots ${_group_slots}
                        --ram-slots 8 --locality scattered --scenario ${_group_scenario}
                        --samples 5 --requests 2 --cold-attempt 0 --table-rows 1048576
                    LABELS "V2;Performance;Ngram;Pooled;ConsumerLocal;SharedSource;NativeReplay;Diagnostic;Unprofiled;${_group_backend};JSON"
                    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
                    RESOURCE_LOCK "${_group_cli}_gpu_0;${_group_cli}_gpu_1"
                    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=${_group_skip_cuda};LLAMINAR_SKIP_ROCM_STARTUP=${_group_skip_rocm};LLAMINAR_LOG_LEVEL=ERROR")
            endforeach()
        endif()
    endforeach()
endif()
