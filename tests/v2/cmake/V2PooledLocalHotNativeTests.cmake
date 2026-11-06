# @file V2PooledLocalHotNativeTests.cmake
# @brief Actual L1 production kernels with original roots and explicitly recorded response inputs.
if(HAVE_CUDA OR HAVE_ROCM)
    add_executable(v2_integration_pooled_local_hot_native EXCLUDE_FROM_ALL
        integration/kernels/qwen4exp/Test__PooledLocalHotNativeDevice.cpp)
    if(HAVE_CUDA)
        target_sources(v2_integration_pooled_local_hot_native PRIVATE integration/kernels/qwen4exp/LocalHotRecordedResponse.cu)
        target_link_libraries(v2_integration_pooled_local_hot_native PRIVATE CUDA::cudart)
    endif()
    if(HAVE_ROCM)
        target_sources(v2_integration_pooled_local_hot_native PRIVATE integration/kernels/qwen4exp/LocalHotRecordedResponse.hip)
        set_source_files_properties(integration/kernels/qwen4exp/LocalHotRecordedResponse.hip PROPERTIES LANGUAGE HIP)
        set_target_properties(v2_integration_pooled_local_hot_native PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
        target_link_libraries(v2_integration_pooled_local_hot_native PRIVATE hip::host)
    endif()
    target_include_directories(v2_integration_pooled_local_hot_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(v2_integration_pooled_local_hot_native PRIVATE llaminar2_core GTest::gtest GTest::gtest_main)
    v2_bind_gpu_process_test_entry(v2_integration_pooled_local_hot_native v2_core_gtest_main)
    foreach(_backend CUDA ROCm)
        if((_backend STREQUAL "CUDA" AND HAVE_CUDA) OR (_backend STREQUAL "ROCm" AND HAVE_ROCM))
            if(_backend STREQUAL "CUDA")
                set(_env "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=0;LLAMINAR_SKIP_ROCM_STARTUP=1")
                set(_lock cuda_gpu_0)
            else()
                set(_env "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=0")
                set(_lock rocm_gpu_0)
            endif()
            foreach(_case AllFormatsC0 AllFormatsPositive WideKeyPositive TerminalObservation LargeRadixScan LargeRadixWideKey)
                add_v2_test(V2_Integration_${_backend}_PooledLocalHotNative_${_case}
                    COMMAND $<TARGET_FILE:v2_integration_pooled_local_hot_native>
                        --gtest_filter=PooledLocalHotNative.${_backend}_${_case}
                    LABELS "V2;Integration;ProductionTestPreflight;GPU;${_backend};Ngram;Pooled;TraceReplay;Regression"
                    MPI_PROCS 1 NO_MPI NO_MODELS RESOURCE_LOCK ${_lock} TIMEOUT 120 ENVIRONMENT ${_env})
                list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_${_backend}_PooledLocalHotNative_${_case})
            endforeach()
        endif()
    endforeach()
endif()
