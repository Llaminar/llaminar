#[[
@file V2PreparedCalibrationOwnerTests.cmake
@brief Register actual PMA isolation between serving graphs and startup calibration.

The preparation slot keeps the original General limits and lifetime while a
model holds its complete serving reservation. This model-free gate complements
native serial-definition/reset and prepared-weight measurement replays.
]]
# Startup calibration owns a distinct PMA graph family, so a fully reserved
# serving model cannot consume its one admitted cold-reusable General slot.
# The exact original slot and native retirement remain exercised by the existing
# CUDA/ROCm prepared-calibration owner and production measurement registrations.
if(TARGET v2_test_pooled_native_program_diagnostic)
    add_v2_test(V2_Integration_PooledNativeProgram_PreparedCalibrationBOMIsolatedFromServingReservationReplay
        COMMAND $<TARGET_FILE:v2_test_pooled_native_program_diagnostic>
            --gtest_filter=PooledNativeProgramDiagnostic.PreparedCalibrationBOMIsolatedFromServingReservation
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1"
            "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1"
            "GTEST_DEATH_TEST_STYLE=threadsafe")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNativeProgram_PreparedCalibrationBOMIsolatedFromServingReservationReplay)
endif()
