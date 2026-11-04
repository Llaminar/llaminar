# Actual source/cohort/owner-map/PMA production bodies; no models or native readiness.
if(TARGET v2_unit_moe_overlay_capacity_resolver)
    add_v2_test(V2_Integration_CPUExpertShardAccounting
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_capacity_resolver>
            --gtest_filter=CPUExpertShardAccounting.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;TensorParallel;NUMA;Memory;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUExpertShardAccounting)
endif()

# Shared source-format fixtures are owned by the existing tests/v2 utility root.
target_include_directories(v2_unit_moe_overlay_capacity_resolver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
