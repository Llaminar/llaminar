# @file V2PooledLocalHotNativeTests.cmake
# @brief Actual L1 production kernels with original roots and explicitly recorded response inputs.
if(HAVE_CUDA OR HAVE_ROCM)
    # The same native fixture owns functional assertions and explicit timing.
    # Release admits its performance role; Integration retains its original
    # functional role. Every source and runtime-owner property follows that role.
    if(V2_PERF_TESTS_ONLY)
        set(_v2_pooled_local_hot_native_target v2_perf_pooled_local_hot_native)
    else()
        set(_v2_pooled_local_hot_native_target v2_integration_pooled_local_hot_native)
    endif()
    add_executable(${_v2_pooled_local_hot_native_target} EXCLUDE_FROM_ALL
        integration/kernels/qwen4exp/Test__PooledLocalHotNativeDevice.cpp)
    if(HAVE_CUDA)
        target_sources(${_v2_pooled_local_hot_native_target} PRIVATE integration/kernels/qwen4exp/LocalHotRecordedResponse.cu)
        target_link_libraries(${_v2_pooled_local_hot_native_target} PRIVATE CUDA::cudart)
    endif()
    if(HAVE_ROCM)
        target_sources(${_v2_pooled_local_hot_native_target} PRIVATE integration/kernels/qwen4exp/LocalHotRecordedResponse.hip)
        set_source_files_properties(integration/kernels/qwen4exp/LocalHotRecordedResponse.hip PROPERTIES LANGUAGE HIP)
        set_target_properties(${_v2_pooled_local_hot_native_target} PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
        target_link_libraries(${_v2_pooled_local_hot_native_target} PRIVATE hip::host)
    endif()
    target_include_directories(${_v2_pooled_local_hot_native_target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${_v2_pooled_local_hot_native_target} PRIVATE llaminar2_core GTest::gtest GTest::gtest_main)
    v2_bind_gpu_process_test_entry(${_v2_pooled_local_hot_native_target} v2_core_gtest_main)
    foreach(_backend CUDA ROCm)
        if((_backend STREQUAL "CUDA" AND HAVE_CUDA) OR (_backend STREQUAL "ROCm" AND HAVE_ROCM))
            if(_backend STREQUAL "CUDA")
                set(_env "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=0;LLAMINAR_SKIP_ROCM_STARTUP=1")
                set(_lock cuda_gpu_0)
            else()
                set(_env "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=0")
                set(_lock rocm_gpu_0)
            endif()
            if(NOT V2_PERF_TESTS_ONLY)
                # Each source/PMA root has its own process lifetime. Register cache
                # capacity and key-width cells separately instead of preparing an
                # unrelated authority after a native runtime has already existed.
                foreach(_case AllFormatsC0 AllFormatsPositive WideKeyPositive TerminalObservation LargeRadixScan LargeRadixWideKey PrefillNextWindowCount LargeFutureAdoption SmallProbeBoundary SmallProbeBoundaryC0 SmallProbeBoundaryWideKey SmallProbeBoundaryWideKeyC0)
                    add_v2_test(V2_Integration_${_backend}_PooledLocalHotNative_${_case}
                        COMMAND $<TARGET_FILE:${_v2_pooled_local_hot_native_target}>
                            --gtest_filter=PooledLocalHotNative.${_backend}_${_case}
                        LABELS "V2;Integration;ProductionTestPreflight;GPU;${_backend};Ngram;Pooled;TraceReplay;Regression"
                        MPI_PROCS 1 NO_MPI NO_MODELS RESOURCE_LOCK ${_lock} TIMEOUT 120 ENVIRONMENT ${_env})
                    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_${_backend}_PooledLocalHotNative_${_case})
                endforeach()
            endif()
            # Explicit native-event diagnostic; correctness registrations above stay untimed.
            add_v2_test(V2_Perf_${_backend}_PooledLocalHotNative_SmallProbeTiming
                COMMAND $<TARGET_FILE:${_v2_pooled_local_hot_native_target}>
                    --gtest_filter=PooledLocalHotNative.${_backend}_SmallProbeTiming
                LABELS "V2;Performance;GPU;${_backend};Ngram;Pooled;NativeReplay;Diagnostic;Unprofiled"
                MPI_PROCS 1 NO_MPI NO_MODELS RESOURCE_LOCK ${_lock} TIMEOUT 120 ENVIRONMENT ${_env})
        endif()
    endforeach()
endif()
unset(_v2_pooled_local_hot_native_target)
