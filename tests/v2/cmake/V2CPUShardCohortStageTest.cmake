#[[
@file V2CPUShardCohortStageTest.cmake
@brief Register the original CPU expert cohort and its shutdown lifecycle on qualified sockets.
Two MPI ranks on a single socket cannot supply the distinct physical owners that
these native tests attest. The separate device-free arithmetic contract runs on
both topologies.
]]
if(V2_TEST_SOCKETS LESS 2)
    message(STATUS "CPU shard cohort stage: unqualified (requires two physical sockets)")
    return()
endif()
add_v2_test(V2_Integration_CPUShardOriginalCohortStage
    COMMAND $<TARGET_FILE:v2_integration_global_tp_mpi>
        --gtest_filter=Test__GlobalTPContext_MPI.CPUShardStageUsesOriginalCohortAndUnweightedRows
    LABELS "V2;Integration;ProductionTestPreflight;CPU;MoE;Shard;MPI;Collective;Replay;Economy;Contract"
    MPI_PROCS 2 NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
        "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUShardOriginalCohortStage)

# Terminal prefix persistence must run after both physical CPU owners retire.
add_v2_test(V2_Integration_PrefixCacheAfterCPUExpertRetirement_MPI
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_shutdown_mpi>
        --gtest_filter=Test__PrefixCacheShutdownMPI.RetiresOriginalExpertDomainBeforeAsymmetricPersistence
    LABELS "V2;Integration;ProductionTestPreflight;PrefixCache;Lifecycle;MPI;Regression;DeviceFree"
    MPI_PROCS 2 NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
        "LLAMINAR_SKIP_ROCM_STARTUP=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PrefixCacheAfterCPUExpertRetirement_MPI)
