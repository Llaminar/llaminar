# @file V2MoEOverlayPlacementTargetTests.cmake
# @brief Register actual whole/cohort migration placement and controller replay contracts.
# These tests prove metadata/slot obligations; native execution remains separately gated.
add_v2_routed_expert_policy_unit(
    v2_unit_moe_overlay_placement_targets
    V2_Unit_MoEOverlayPlacementTargets
    unit/execution/moe/Test__MoEOverlayPlacementTargets.cpp)
add_v2_test(V2_Integration_MoEOverlayPlacementCycleReplay
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_placement_targets>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Residency;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOverlayPlacementCycleReplay)
