# @file V2PooledNativeProgramDiagnosticTests.cmake
# @brief Bound original resource-owner replays per actual case, with executable/CTest coverage parity.
# Recorded actual owner lifecycles; no native runtime/device/model is exercised.
if(NOT V2_PERF_TESTS_ONLY)
    add_executable(v2_test_pooled_native_program_diagnostic
        unit/backends/Test__PooledNativeProgramDiagnostic.cpp
        integration/core_gtest_main.cpp)
    # Pooled immutable sources are CUDA/ROCm-native objects (no CPU
    # implementation); every registration requires a GPU backend family.
    if(TARGET v2_test_pooled_native_program_diagnostic)
        set_target_properties(v2_test_pooled_native_program_diagnostic PROPERTIES V2_TEST_REQUIRED_BACKEND GPU)
    endif()
    if(TARGET v2_test_pooled_native_program_diagnostic)
        target_compile_definitions(v2_test_pooled_native_program_diagnostic PRIVATE LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC=1)
        target_include_directories(v2_test_pooled_native_program_diagnostic PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        target_link_libraries(v2_test_pooled_native_program_diagnostic PRIVATE llaminar2_core GTest::gtest)
        target_compile_features(v2_test_pooled_native_program_diagnostic PRIVATE cxx_std_23)
        add_v2_test(V2_Unit_PooledNativeProgramDiagnostic
            COMMAND $<TARGET_FILE:v2_test_pooled_native_program_diagnostic>
            LABELS "V2;Unit;DeviceFree;Ngram;Ownership;Replay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "GTEST_DEATH_TEST_STYLE=threadsafe")
        # Each original case retains 20 threadsafe replays while native process
        # bootstrap remains bounded independently of the other cases.
        set(_native_program_cases
            RawFamilyParentHaveNoInventedCommitmentAndRetireActualTerminal
            ActualCreatorFailureReturnsDestroyOnlySuccessfulPrefix
            InvalidPrimaryAndPreparingSourceRejectBeforeAnyCreator
            GeneralOriginalAllowanceEnforcesPlannedSlotsAndZeroAttemptCleanup
            GeneralHostBOMRejectsMissingForeignAndOverflowBeforeNative
            GeneralAttemptedHostAndDeviceBackingSurviveOriginalReset
            GeneralAttemptedBackingRemainsProcessRetainedAcrossActualReset
            GeneralLiveReadersRejectTeardownBeforeActualCaptureDestruction
            RecordingFacadeRejectsUntrackedDuplicateAndResetLaunches
            RecordingNativeFailuresIrreversiblyRejectEveryRetry
            RecordingRejectsForeignControlBlockAndWrongWorkerBeforeNative
            RecordingDelegatesBoundedFlatNativeValidation
            PreparedCalibrationCUDATimingOwner PreparedCalibrationROCmTimingOwner
    PreparedCalibrationCUDASequential
            PreparedCalibrationROCmSequential
            PreparedCalibrationCUDARejectsUnproved
            PreparedCalibrationROCmRejectsUnproved
            PreparedCalibrationSourceCUDARecreation
            PreparedCalibrationSourceROCmRecreation
            PreparedCalibrationSourceCUDAHolderRetirement
            PreparedCalibrationSourceROCmHolderRetirement
            PreparedCalibrationSourceCUDAModelTicketOrder
            PreparedCalibrationSourceROCmModelTicketOrder
        )
        foreach(_native_program_case IN LISTS _native_program_cases)
            set(_native_program_gate "V2_Integration_PooledNativeProgram_${_native_program_case}Replay")
            add_v2_test(${_native_program_gate}
                COMMAND $<TARGET_FILE:v2_test_pooled_native_program_diagnostic>
                    --gtest_filter=PooledNativeProgramDiagnostic.${_native_program_case} --gtest_repeat=20
                LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;Replay;Regression"
                MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
                ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "GTEST_DEATH_TEST_STYLE=threadsafe")
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_native_program_gate})
        endforeach()
        add_v2_test(V2_Unit_PooledNativeProgramRegistrationCoverage
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_native_program_registration.py" --self-test
            LABELS "V2;Unit;DeviceFree;Ownership;SourcePolicy;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
        add_v2_test(V2_Integration_PooledNativeProgramRegistrationCoverage
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_native_program_registration.py"
                --binary $<TARGET_FILE:v2_test_pooled_native_program_diagnostic>
                --ctest "${CMAKE_CTEST_COMMAND}" --build-dir "${CMAKE_BINARY_DIR}"
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ownership;SourcePolicy;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "GTEST_DEATH_TEST_STYLE=threadsafe")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNativeProgramRegistrationCoverage)
    endif()
    add_v2_test(V2_Integration_NativeAuxiliaryDiagnosticSourcePolicy
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_native_auxiliary_diagnostic_boundary.py" --source "${REPO_ROOT}"
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ownership;SourcePolicy;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NativeAuxiliaryDiagnosticSourcePolicy)
endif()
