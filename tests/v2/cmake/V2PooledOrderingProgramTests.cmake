# Original root geometry defines the closed event program before resource admission.
if(TARGET v2_test_qwen4exp_primitives)
    target_sources(v2_test_qwen4exp_primitives PRIVATE
        unit/kernels/qwen4exp/Test__PooledImmutableOrderingProgram.cpp
        unit/planning/Test__PooledImmutableOrderingGeometry.cpp)
    add_v2_test(V2_Integration_PooledImmutableOrderingProgramReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableOrderingProgram.*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    add_v2_test(V2_Integration_PooledImmutableOrderingGeometryReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives> --gtest_filter=PooledImmutableOrderingGeometry.*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    add_v2_test(V2_Integration_PooledImmutableSourceComputeReaderReadiness
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableOrderingProgram.SourceReadinessPreservesDeclaredComputeReaderPopulation
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    add_v2_test(V2_Integration_PooledImmutableLiveRootRetention
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableOrderingProgram.RetainedRootProjectionDistinguishesPhysicalOwnersAndReaders
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledImmutableLiveRootRetention)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledImmutableSourceComputeReaderReadiness)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledImmutableOrderingProgramReplay V2_Integration_PooledImmutableOrderingGeometryReplay)
endif()

# Isolated backend-resolution boundary for actual root/Tensor lifetime replay.
if(NOT V2_PERF_TESTS_ONLY)
add_executable(v2_test_pooled_ordering_root_lifetime unit/backends/Test__PooledImmutableOrderingRootLifetime.cpp)
if(TARGET v2_test_pooled_ordering_root_lifetime)
    target_include_directories(v2_test_pooled_ordering_root_lifetime PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${REPO_ROOT}/src/v2")
    target_link_libraries(v2_test_pooled_ordering_root_lifetime PRIVATE llaminar2_core GTest::gtest_main)
    target_compile_features(v2_test_pooled_ordering_root_lifetime PRIVATE cxx_std_23)
    target_link_options(v2_test_pooled_ordering_root_lifetime PRIVATE "LINKER:--export-dynamic")
    v2_test_device_scope(v2_test_pooled_ordering_root_lifetime DeclaredBackends)
    v2_bind_gpu_process_test_entry(v2_test_pooled_ordering_root_lifetime v2_core_gtest_main)
    foreach(_ordering_suite Unit Integration)
        set(_ordering_test "V2_${_ordering_suite}_PooledImmutableOrderingRootLifetimeReplay")
        set(_ordering_labels "V2;${_ordering_suite};DeviceFree;Memory;Ngram;Ownership;TraceReplay;Regression")
        if(_ordering_suite STREQUAL "Integration")
            list(APPEND _ordering_labels ProductionTestPreflight)
        endif()
        add_v2_test(${_ordering_test}
            COMMAND $<TARGET_FILE:v2_test_pooled_ordering_root_lifetime> --gtest_filter=PooledImmutableOrderingRootLifetime.*
            LABELS "${_ordering_labels}" MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    endforeach()
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledImmutableOrderingRootLifetimeReplay)
endif()
endif()

# The actual parent graph fork needs no extra Probe event; original setup3/4 remains.
if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledLocalHotOriginalSetupEventBOMReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableOrderingProgram.LocalHotForkRetainsOnlyOriginalSetupEventBOMReplay
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ngram;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledLocalHotOriginalSetupEventBOMReplay)
endif()
