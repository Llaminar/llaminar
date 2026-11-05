# Closed recorded claims authenticate the same original factory bodies before native creators.
foreach(_factory_target v2_test_pooled_owner_pin_chain v2_test_pooled_hit_read_graph)
    if(TARGET ${_factory_target})
        target_include_directories(${_factory_target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
endforeach()
if(TARGET v2_test_pooled_owner_pin_chain)
    add_v2_test(V2_Integration_PooledImmutableRecordedFactoryClaimReplay
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py $<TARGET_FILE:v2_test_pooled_owner_pin_chain>
            --gtest_filter=PooledImmutableOwnerPinChain.RecordedOrdering* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ngram;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledImmutableRecordedFactoryClaimReplay)
endif()
if(TARGET v2_test_pooled_hit_read_graph)
    foreach(_original_source_partition RANGE 1 4)
    add_v2_test(V2_Integration_PooledImmutableRecordedSourceClaimReplay${_original_source_partition}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py $<TARGET_FILE:v2_test_pooled_hit_read_graph>
            --gtest_filter=PooledImmutableReadGraph.RecordedOrdering* --gtest_repeat=5
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ngram;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledImmutableRecordedSourceClaimReplay${_original_source_partition})
    endforeach()
endif()

# Exact source storage survives serial runner-controller lifetimes on backend doubles.
if(TARGET v2_test_pooled_hit_read_graph)
    add_v2_test(V2_Integration_PooledImmutableGenerationBorrowLifecycleContract
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py $<TARGET_FILE:v2_test_pooled_hit_read_graph>
            --gtest_filter=PooledImmutableReadGraph.CanonicalGenerationBorrowRetiresBeforeRunnerAddressReplacement
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ngram;Ownership;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledImmutableGenerationBorrowLifecycleContract)
endif()

# Actual P1 graph construction and unchanged mixed-topology rejection.
# Native full shared result/ACK execution remains the registered real-model source gate.
if(TARGET v2_test_pooled_owner_pin_chain)
    add_v2_test(V2_Integration_PooledSoleLocalSourceGraphLifecycleContract
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py $<TARGET_FILE:v2_test_pooled_owner_pin_chain>
            --gtest_filter=PooledImmutableOwnerPinChain.SoleLocalDirectedSourceKeepsOriginalPhasesAndModelJoins:PooledImmutableOwnerPinChain.MixedLocalP1RetainsOriginalUnallocatedFrontier
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ngram;Ownership;Lifecycle;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledSoleLocalSourceGraphLifecycleContract)
endif()
