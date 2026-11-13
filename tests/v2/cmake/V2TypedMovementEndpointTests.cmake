#[=[
@file V2TypedMovementEndpointTests.cmake
@brief Replay original complete-target endpoint publication/export and actual per-member terminal joins.
]=]
add_v2_routed_expert_policy_unit(v2_unit_moe_movement_endpoint V2_Unit_MoEMovementEndpoint
    unit/execution/moe/Test__MoEOptimizationMovementEndpoint.cpp)
add_v2_test(V2_Integration_MoEMovementEndpointOriginalCohortReplay
    COMMAND $<TARGET_FILE:v2_unit_moe_movement_endpoint>
        --gtest_filter=MoEOptimizationMovementEndpoint.OriginalCohortLifetimeAndLosslessExportReplay
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;ExpertOverlay;Ownership;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEMovementEndpointOriginalCohortReplay)
add_v2_test(V2_Integration_MoEMovementEndpointMemberTransportReplay
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_generation_movement_ledger.py
        CompleteCohortMovementTests
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;ExpertOverlay;Transport;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEMovementEndpointMemberTransportReplay)

add_v2_test(V2_Integration_CohortHostFollowerEvidenceReplay
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_generation_movement_ledger.py
        CohortHostFollowerEvidenceTests
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;ExpertOverlay;Evidence;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CohortHostFollowerEvidenceReplay)
