# @file V2NativeSourceFingerprintTests.cmake
# @brief Prove actual native creator dependencies remain unique and fresh in Ninja.
# This replay invokes only CMake's compiler-free project and Ninja's help target;
# the original bad module is a negative control, not an alternate production path.
if(NOT V2_PERF_TESTS_ONLY)
    add_v2_test(V2_Unit_NativeSourceFingerprint
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_native_source_fingerprint.py"
        LABELS "V2;Unit;DeviceFree;BuildSystem;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    add_v2_test(V2_Integration_NativeSourceFingerprintReplay
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_native_source_fingerprint.py"
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;BuildSystem;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NativeSourceFingerprintReplay)
endif()
