#[[
@file V2CPUShardRouteCollectiveTests.cmake
@brief Explicit compact expert-shard route and original rooted transport regressions.
The two-rank case proves the real original MPI/shared-memory lifecycle. Neither
it nor device-free recorded partials certify full NUMA model serving or economy.
]]
# Release deliberately owns only performance targets. Keep functional target
# creation, source/property mutations and registration under the same policy.
# Integration still executes every declaration and rejects missing owners.
if(V2_PERF_TESTS_ONLY)
    return()
endif()

add_executable(v2_test_cpu_shard_route_collective unit/kernels/cpu/moe/Test__CPUShardRouteCollective.cpp)
target_compile_features(v2_test_cpu_shard_route_collective PRIVATE cxx_std_20)
target_include_directories(v2_test_cpu_shard_route_collective PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(v2_test_cpu_shard_route_collective PRIVATE llaminar2_core v2_core_gtest_main GTest::gtest)
add_v2_test(V2_Unit_CPUShardRouteCollective
    COMMAND $<TARGET_FILE:v2_test_cpu_shard_route_collective> --gtest_filter=CPUShardRouteCollective.*
    LABELS "V2;Unit;DeviceFree;CPU;MoE;Shard;Replay"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 10
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
        "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
foreach(cpu_shard_route_case IN ITEMS ArrivalOrderPartialAndRepeatedReset AllCohortWidthsUseSourceOrder
    InvalidCompletePrefixNeverPublishes EmptyAndLargeToSmallPreserveLiveExtent
    FixedParticipantSumPrecedesOriginalWeighting OriginalSparseRoutesMatchDenseAndRejectPartialIdentity)
    set(cpu_shard_route_name "V2_Integration_CPUShardRouteCollective_${cpu_shard_route_case}Replay")
    add_v2_test(${cpu_shard_route_name}
        COMMAND $<TARGET_FILE:v2_test_cpu_shard_route_collective>
            "--gtest_filter=CPUShardRouteCollective.${cpu_shard_route_case}" --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Shard;Replay"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 10
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
            "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${cpu_shard_route_name})
endforeach()
add_v2_test(V2_Integration_CPUShardSourceRangesLivePrefix
    COMMAND $<TARGET_FILE:v2_integration_global_tp_mpi>
        --gtest_filter=Test__GlobalTPContext_MPI.CPUShardSourceRangesPreserveLivePrefixes
    LABELS "V2;Integration;ProductionTestPreflight;CPU;MoE;Shard;MPI;Collective;Replay"
    MPI_PROCS 2 NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
        "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUShardSourceRangesLivePrefix)
add_v2_test(V2_Integration_CPUShardPreparedPackedRecords
    COMMAND $<TARGET_FILE:v2_test_cpu_expert_shard_execution>
        --gtest_filter=CPUExpertShardExecution.PreparedPackedSourceRecordsAllFormatsAndPhases
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Shard;Replay"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
        "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUShardPreparedPackedRecords)
add_v2_test(V2_Integration_CPUShardPreparedPackedQFNClasses
    COMMAND $<TARGET_FILE:v2_test_cpu_expert_shard_execution>
        --gtest_filter=CPUExpertShardExecution.PreparedPackedQFNMixedClasses
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Shard;Replay"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
        "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUShardPreparedPackedQFNClasses)
# Two ranks on distinct NUMA sockets; single-socket hosts cannot form this cohort.
if(V2_TEST_SOCKETS GREATER_EQUAL 2)
add_v2_test(V2_Integration_CPUShardOriginalCohortStage
    COMMAND $<TARGET_FILE:v2_integration_global_tp_mpi>
        --gtest_filter=Test__GlobalTPContext_MPI.CPUShardStageUsesOriginalCohortAndUnweightedRows
    LABELS "V2;Integration;ProductionTestPreflight;CPU;MoE;Shard;MPI;Collective;Replay"
    MPI_PROCS 2 NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
        "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUShardOriginalCohortStage)
endif()

# Metadata checks call the same source/owner-map and immutable return-role projections as serving.
target_sources(v2_test_cpu_shard_route_collective PRIVATE unit/execution/moe/Test__CPUJointServingMetadata.cpp)
foreach(cpu_joint_metadata_case IN ITEMS ExplicitTensorTP1UsesOriginalWholeMatrixWithoutJointAuthority OriginalPopulationAndReturnRolesThroughEight ActualStageSharesRootExtentAcrossAsymmetricParticipantBanks ActualStageRejectsSparsePhaseWithoutOriginalPreparedBank RootTimingLifecycle)
    set(cpu_joint_metadata_name "V2_Integration_CPUJointServingMetadata_${cpu_joint_metadata_case}")
    add_v2_test(${cpu_joint_metadata_name}
        COMMAND $<TARGET_FILE:v2_test_cpu_shard_route_collective> "--gtest_filter=CPUJointServingMetadata.${cpu_joint_metadata_case}" --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Shard;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 10
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${cpu_joint_metadata_name})
endforeach()

# Capacity is storage only: the contract and original two-socket stage use live9/0/1.
add_v2_test(V2_Integration_CPUShardPaddedCompactGatherEconomyContract
    COMMAND $<TARGET_FILE:v2_test_cpu_shard_route_collective>
        --gtest_filter=CPUShardRouteCollective.Padded64CompactGatherAndReducerEconomyContract --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Economy;Contract;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 10
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUShardPaddedCompactGatherEconomyContract)

# Actual original two-socket transport/stage; no prepared/ticket readiness is invented.
# Two ranks on distinct NUMA sockets; single-socket hosts cannot form this cohort.
if(V2_TEST_SOCKETS GREATER_EQUAL 2)
add_v2_test(V2_Integration_CPUShardPaddedCompactGatherNativeEconomy
    COMMAND $<TARGET_FILE:v2_integration_global_tp_mpi>
        --gtest_filter=Test__GlobalTPContext_MPI.CPUShardStageUsesOriginalCohortAndUnweightedRows
    LABELS "V2;Integration;ProductionTestPreflight;CPU;MoE;Shard;MPI;Collective;Economy;Contract"
    MPI_PROCS 2 NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUShardPaddedCompactGatherNativeEconomy)
endif()

# Exercise the actual original-domain constructor's rejection before traffic.
add_v2_test(V2_Integration_CPUJointPreparedComputeOriginalDomainContract
    COMMAND $<TARGET_FILE:v2_test_cpu_shard_route_collective>
        --gtest_filter=CPUJointServingMetadata.PreparedComputeRejectsForeignDomainBeforeCollectives
        --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Shard;Measurement;Contract;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 10
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
        "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointPreparedComputeOriginalDomainContract)
