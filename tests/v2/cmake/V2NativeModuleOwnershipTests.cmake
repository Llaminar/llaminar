# Replay actual focused native link declarations with language-free imported targets.
add_v2_test(V2_Unit_NativeModuleOwnership
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/../unit/scripts/test_native_module_ownership.py
    LABELS "V2;Unit;DeviceFree;Ownership;BuildSystem;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
add_v2_test(V2_Integration_NativeModuleOwnershipReplay
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/../unit/scripts/test_native_module_ownership.py
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ownership;BuildSystem;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NativeModuleOwnershipReplay)
