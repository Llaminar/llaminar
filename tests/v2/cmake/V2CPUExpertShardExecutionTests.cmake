#[[
@file V2CPUExpertShardExecutionTests.cmake
@brief Real CPU expert source-fragment execution and pre-weighting partial-fold replay.

The ordinary Unit registration stays small. The explicit CPU integration case
executes prepared NativeQuantised and floating engines over original source slices;
it does not certify MPI/NUMA serving placement, source loading or performance.
]]
add_executable(v2_test_cpu_expert_shard_execution
    unit/kernels/cpu/moe/Test__CPUExpertShardExecution.cpp)
target_compile_features(v2_test_cpu_expert_shard_execution PRIVATE cxx_std_20)
target_include_directories(v2_test_cpu_expert_shard_execution PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_test_cpu_expert_shard_execution PRIVATE
    llaminar2_core v2_core_gtest_main GTest::gtest)
add_v2_test(V2_Unit_CPUExpertShardExecution
    COMMAND $<TARGET_FILE:v2_test_cpu_expert_shard_execution>
        --gtest_filter=CPUExpertShardExecution.OrderedPartialRowsPrecedeTopKAndReset:CPUExpertShardExecution.RejectsFoldBeforePartialReductionOracle
    LABELS "V2;Unit;DeviceFree;CPU;MoE;Shard;Replay"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 10
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
foreach(cpu_shard_case IN ITEMS OrderedPartialRowsPrecedeTopKAndReset
    RejectsFoldBeforePartialReductionOracle PreparedNativeSourceAllFormatsAndSerialRows
    QFNOriginalMixedFormatsAndNative320Cuts QFNOriginalQ4KQ8_0Native320Cuts
    QFNOriginalQ5KQ8_0Native320Cuts)
    if(cpu_shard_case STREQUAL "PreparedNativeSourceAllFormatsAndSerialRows" OR
       cpu_shard_case STREQUAL "QFNOriginalMixedFormatsAndNative320Cuts" OR
       cpu_shard_case STREQUAL "QFNOriginalQ4KQ8_0Native320Cuts" OR
       cpu_shard_case STREQUAL "QFNOriginalQ5KQ8_0Native320Cuts")
        set(cpu_shard_repeats 20)
        set(cpu_shard_timeout 30)
    else()
        set(cpu_shard_repeats 20)
        set(cpu_shard_timeout 10)
    endif()
    set(cpu_shard_name "V2_Integration_CPUExpertShardExecution_${cpu_shard_case}Replay")
    add_v2_test(${cpu_shard_name}
        COMMAND $<TARGET_FILE:v2_test_cpu_expert_shard_execution>
            "--gtest_filter=CPUExpertShardExecution.${cpu_shard_case}" "--gtest_repeat=${cpu_shard_repeats}"
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Shard;Replay"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT ${cpu_shard_timeout}
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${cpu_shard_name})
endforeach()

# Projection/source identity is strict; TP reassociation and FP64 distances are diagnostic, not model KLD acceptance.
add_v2_test(V2_Integration_CPUFullFragmentProjectionContract
    COMMAND $<TARGET_FILE:v2_test_cpu_expert_shard_execution>
        --gtest_filter=CPUExpertShardExecution.IndependentFullExpertVersusNative320FragmentContract --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Projection;Contract;NumericalDiagnostic"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUFullFragmentProjectionContract)
