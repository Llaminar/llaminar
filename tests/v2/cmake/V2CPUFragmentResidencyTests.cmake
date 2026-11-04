# @file V2CPUFragmentResidencyTests.cmake
# @brief Register source-bound fragment, placement and RCU ownership regressions.
#
# These device-free component tests exercise the actual prepared-bank interfaces
# and complete-piece publication. They do not claim native shard migration or
# global placement commit, which require the corresponding physical path gates.
set(V2_CPU_FRAGMENT_TEST_ENV
    "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")

add_v2_test(V2_Integration_PreparedProjectionIntersectionContract
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_participant_residency>
        --gtest_filter=PreparedProjectionIntersectionContract.* --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Contract;Lifecycle;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PreparedProjectionIntersectionContract)

add_v2_test(V2_Integration_CPUFragmentPayloadContract
    COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
        --gtest_filter=CPUFragmentPayloadContract.* --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;PreparedWeights;Contract;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUFragmentPayloadContract)

add_v2_test(V2_Integration_CPUJointPlacementCandidateContract
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_residency_authority>
        --gtest_filter=CPUJointPlacementCandidateContract.* --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Contract;SourceBinding;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointPlacementCandidateContract)

add_v2_test(V2_Integration_CPUFragmentEpochBankContract
    COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
        --gtest_filter=CPUFragmentEpochBankContract.* --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;PreparedWeights;Lifecycle;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUFragmentEpochBankContract)

add_v2_test(V2_Integration_CPUJointInitialBankPrepassContract
    COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
        --gtest_filter=CPUJointInitialBankPrepassContract.* --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Contract;Lifecycle;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointInitialBankPrepassContract)
unset(V2_CPU_FRAGMENT_TEST_ENV)
