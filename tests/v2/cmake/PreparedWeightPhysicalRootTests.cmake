# Replay the actual store identity and canonical physical creator/alias BOM lowering.
add_executable(v2_test_prepared_weight_physical_roots
    unit/planning/Test__PreparedWeightPhysicalRoots.cpp)
target_link_libraries(v2_test_prepared_weight_physical_roots PRIVATE llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_PreparedWeightPhysicalRoots
    COMMAND $<TARGET_FILE:v2_test_prepared_weight_physical_roots>
    LABELS "V2;Unit;Planning;PreparedWeights;Ownership;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PreparedWeightPhysicalRootReplay
    COMMAND $<TARGET_FILE:v2_test_prepared_weight_physical_roots>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Planning;PreparedWeights;Ownership;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PreparedWeightPhysicalRootReplay)
