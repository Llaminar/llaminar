# @file V2LocalHotNativeForkTests.cmake
# @brief Exact native imported/owned parent forks with no auxiliary executable or model authority.
if(HAVE_CUDA OR HAVE_ROCM)
    add_executable(v2_integration_local_hot_native_fork EXCLUDE_FROM_ALL
        integration/transfer/Test__LocalHotNativeParentFork.cpp)
    target_include_directories(v2_integration_local_hot_native_fork PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(v2_integration_local_hot_native_fork PRIVATE llaminar2_core GTest::gtest GTest::gtest_main)
    v2_bind_gpu_process_test_entry(v2_integration_local_hot_native_fork v2_core_gtest_main)
    foreach(_backend CUDA ROCm)
        if((_backend STREQUAL "CUDA" AND HAVE_CUDA) OR (_backend STREQUAL "ROCm" AND HAVE_ROCM))
            if(_backend STREQUAL "CUDA")
                set(_env "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=0;LLAMINAR_SKIP_ROCM_STARTUP=1")
                set(_lock cuda_gpu_0)
            else()
                set(_env "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=0")
                set(_lock rocm_gpu_0)
            endif()
            foreach(_case OwnedAllHit ImportedAllHit OwnedMiss ImportedEmpty OldNoEdgeControl OriginalNativeComposerAllHit OriginalSourceAllHitAndMixed)
                add_v2_test(V2_Integration_${_backend}_LocalHotParentFork_${_case}
                    COMMAND $<TARGET_FILE:v2_integration_local_hot_native_fork>
                        --gtest_filter=LocalHotNativeParentFork.${_backend}_${_case}
                    LABELS "V2;Integration;ProductionTestPreflight;GPU;${_backend};Graph;Ownership;TraceReplay;Regression"
                    MPI_PROCS 1 NO_MPI NO_MODELS RESOURCE_LOCK ${_lock} TIMEOUT 30 ENVIRONMENT ${_env})
                list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_${_backend}_LocalHotParentFork_${_case})
            endforeach()
        endif()
    endforeach()
endif()
