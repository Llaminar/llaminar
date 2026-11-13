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

# Each original source admits the process-lifetime native root once. Floating
# and native source fixtures therefore run as separate processes rather than
# attempting to adopt a second PMA after the first fixture retires its workers.
foreach(_gateup_service_backend IN ITEMS CUDA ROCm)
    string(TOUPPER "${_gateup_service_backend}" _gateup_service_feature)
    if(HAVE_${_gateup_service_feature})
        foreach(_gateup_source_kind IN ITEMS Floating Native)
            set(_gateup_source "")
            if(_gateup_source_kind STREQUAL "Native")
                set(_gateup_source Native)
            endif()
            set(_gateup_test "V2_Integration_${_gateup_service_backend}_PreparedGPUService${_gateup_source}GateUpOriginalReplay")
            add_v2_test(${_gateup_test}
                COMMAND $<TARGET_FILE:v2_integration_prepared_expert_service_execution>
                    --gtest_filter=PreparedExpertServiceExecution.${_gateup_service_backend}${_gateup_source}GateUpOriginalMeasurement
                LABELS "V2;Integration;ProductionTestPreflight;${_gateup_service_backend};MoE;PreparedWeights;GraphCapture;Ownership;Lifecycle;Regression"
                ENVIRONMENT "OMP_NUM_THREADS=4"
                MPI_PROCS 1 NO_MODELS TIMEOUT 30)
            if(TEST ${_gateup_test})
                list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_gateup_test})
            endif()
        endforeach()
    endif()
endforeach()
