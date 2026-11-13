# @file V2PooledOriginalSourceSetupTests.cmake
# @brief Register actual original source/PMA component setup without graph or timing resource bypass.
if(TARGET v2_test_physical_memory_authority)
    target_sources(v2_test_physical_memory_authority PRIVATE unit/backends/Test__PreparedNativeLookupFixture.cpp)
    add_v2_test(V2_Unit_PreparedNativeLookupFixture
        COMMAND $<TARGET_FILE:v2_test_physical_memory_authority> --gtest_filter=PreparedNativeLookupFixture.*
        LABELS "V2;Unit;DeviceFree;Memory;Ownership;TraceReplay;Regression" MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    add_v2_test(V2_Integration_PreparedNativeLookupFixtureReplay
        COMMAND $<TARGET_FILE:v2_test_physical_memory_authority> --gtest_filter=PreparedNativeLookupFixture.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;TraceReplay;Regression" MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PreparedNativeLookupFixtureReplay)
endif()
if(HAVE_CUDA OR HAVE_ROCM)
    add_executable(v2_perf_pooled_original_source_setup EXCLUDE_FROM_ALL integration/kernels/qwen4exp/Test__PooledOriginalSourceSetup.cpp)
    if(TARGET v2_perf_pooled_original_source_setup)
        target_include_directories(v2_perf_pooled_original_source_setup PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        target_link_libraries(v2_perf_pooled_original_source_setup PRIVATE llaminar2_core GTest::gtest GTest::gtest_main)
        v2_bind_gpu_process_test_entry(v2_perf_pooled_original_source_setup v2_core_gtest_main)
        if(NOT LLAMINAR_SHARED_CORE)
            target_link_libraries(v2_perf_pooled_original_source_setup PRIVATE -Wl,--start-group)
            if(HAVE_CUDA)
                target_link_libraries(v2_perf_pooled_original_source_setup PRIVATE -Wl,--whole-archive cuda_backend -Wl,--no-whole-archive)
            endif()
            if(HAVE_ROCM)
                target_link_libraries(v2_perf_pooled_original_source_setup PRIVATE -Wl,--whole-archive rocm_backend -Wl,--no-whole-archive)
            endif()
            target_link_libraries(v2_perf_pooled_original_source_setup PRIVATE -Wl,--end-group)
        endif()
    endif()
    foreach(_pooled_source_backend IN ITEMS CUDA ROCm)
        if((_pooled_source_backend STREQUAL "CUDA" AND HAVE_CUDA) OR (_pooled_source_backend STREQUAL "ROCm" AND HAVE_ROCM))
            if(_pooled_source_backend STREQUAL "CUDA")
                set(_pooled_source_env "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=0;LLAMINAR_SKIP_ROCM_STARTUP=1")
                set(_pooled_source_lock cuda_gpu_0)
            else()
                set(_pooled_source_env "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=0")
                set(_pooled_source_lock rocm_gpu_0)
            endif()
            add_v2_test(V2_Integration_${_pooled_source_backend}_PooledCompactQueryDeclarationBeforeSeal
                COMMAND $<TARGET_FILE:v2_perf_pooled_original_source_setup> --gtest_filter=PooledOriginalSourceSetup.${_pooled_source_backend}CompactDeclarationBeforeSeal
                LABELS "V2;Integration;ProductionTestPreflight;${_pooled_source_backend};Memory;Ownership;Ngram;Pooled;NativeReplay;Regression"
                MPI_PROCS 1 NO_MPI NO_MODELS RESOURCE_LOCK ${_pooled_source_lock} TIMEOUT 30 ENVIRONMENT ${_pooled_source_env})
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_${_pooled_source_backend}_PooledCompactQueryDeclarationBeforeSeal)
            foreach(_pooled_source_case IN ITEMS C0 Positive)
                add_v2_test(V2_Integration_${_pooled_source_backend}_PooledOriginalSource${_pooled_source_case}Setup
                    COMMAND $<TARGET_FILE:v2_perf_pooled_original_source_setup> --gtest_filter=PooledOriginalSourceSetup.${_pooled_source_backend}${_pooled_source_case}
                    LABELS "V2;Integration;ProductionTestPreflight;${_pooled_source_backend};Memory;Ownership;Ngram;Pooled;NativeReplay;Regression"
                    MPI_PROCS 1 NO_MPI NO_MODELS RESOURCE_LOCK ${_pooled_source_lock} TIMEOUT 30 ENVIRONMENT ${_pooled_source_env})
                list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_${_pooled_source_backend}_PooledOriginalSource${_pooled_source_case}Setup)
            endforeach()
        endif()
    endforeach()
endif()
