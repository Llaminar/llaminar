# Exercise the installed selector with recorded identities, never a native runtime.
if(TARGET v2_test_physical_memory_authority)
    target_sources(v2_test_physical_memory_authority PRIVATE
        unit/backends/Test__NativeContextQualifiedClass.cpp)
    add_v2_test(V2_Integration_NativeContextQualifiedClassReplay
        COMMAND $<TARGET_FILE:v2_test_physical_memory_authority>
            --gtest_filter=NativeContextQualifiedClass.*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1"
            "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
        V2_Integration_NativeContextQualifiedClassReplay)
endif()

# Closed creator ABI eligibility is independent from complete live source provenance.
if(TARGET v2_test_physical_memory_authority)
    target_sources(v2_test_physical_memory_authority PRIVATE
        unit/backends/Test__NativeAllowanceEligibility.cpp)
    add_v2_test(V2_Integration_NativeAllowanceEligibilityReplay
        COMMAND $<TARGET_FILE:v2_test_physical_memory_authority>
            --gtest_filter=NativeAllowanceEligibility.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1"
            "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NativeAllowanceEligibilityReplay)
endif()

# Direct family allowances tolerate ordinary development/runtime/object variation;
# original explicit pages, capacity and invalid-owner checks remain authoritative.
if(TARGET v2_test_physical_memory_authority)
    add_v2_test(V2_Integration_NativeResourceFamilyAllowanceReplay
        COMMAND $<TARGET_FILE:v2_test_physical_memory_authority>
            --gtest_filter=NativeAllowanceEligibility.ResourceFamiliesIgnoreDevelopmentFingerprintsButKeepCreatorCounts:NativeAllowanceEligibility.ExplicitObjectPagesAndRealCapacityRemainOwned:NativeAllowanceEligibility.InvalidResourceFamilyAndLifetimeStillReject
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NativeResourceFamilyAllowanceReplay)
endif()

# Original source teardown includes storage-only births after partial model construction.
if(NOT V2_PERF_TESTS_ONLY)
    add_executable(v2_test_native_source_final_retirement unit/backends/Test__NativeSourceFinalRetirement.cpp)
    if(TARGET v2_test_native_source_final_retirement)
    target_include_directories(v2_test_native_source_final_retirement PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${REPO_ROOT}/src/v2")
    target_link_libraries(v2_test_native_source_final_retirement PRIVATE llaminar2_core GTest::gtest_main)
    # The actual QFN fixture vtable retains its existing snapshot reader closure;
    # these tests run only its metadata owner edges, never model/backend setup.
    target_sources(v2_test_native_source_final_retirement PRIVATE "${REPO_ROOT}/external/cnpy/cnpy.cpp")
    target_include_directories(v2_test_native_source_final_retirement PRIVATE "${REPO_ROOT}/external/cnpy")
    target_link_libraries(v2_test_native_source_final_retirement PRIVATE v2_test_utils ZLIB::ZLIB)
    target_compile_features(v2_test_native_source_final_retirement PRIVATE cxx_std_23)
    target_link_options(v2_test_native_source_final_retirement PRIVATE "LINKER:--export-dynamic")
    v2_test_device_scope(v2_test_native_source_final_retirement DeclaredBackends)
    v2_bind_gpu_process_test_entry(v2_test_native_source_final_retirement v2_core_gtest_main)
    add_v2_test(V2_Unit_NativeSourceFinalRetirement
        COMMAND $<TARGET_FILE:v2_test_native_source_final_retirement> --gtest_filter=NativeSourceFinalRetirement.*
        LABELS "V2;Unit;DeviceFree;Memory;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "GTEST_DEATH_TEST_STYLE=threadsafe")
    foreach(_case QFNObserverPrimaryAndTerminalFailuresRemainExplicit QFNObserverOffCapacitySelectsOriginalPredictorMaterialization QFNObserverSuccessRetiresSoleOriginalOwner QFNObserverBeforeHistoryFailureRetiresSoleOriginalOwner ExportedContractRetiresOriginalStorageOnlySourceBirths ActualRunnerExportedSourceShutdownThenDestructorIsIdempotent ActualRunnerPostInitializationFailureThenDestructorRetiresOnce RetiredRunnerCannotSealSuccessorActiveReuseAuthority LiveRunnerForeignReusableAuthorityStillRejectsShutdown AdmittedMixedSourceRetiresStorageOnlyBirthAndRepeatsSafely LiveModelAllocationAndBorrowedWorkerRejectBeforeReset FailedAdmittedInitializationPreservesOriginalProcessObligation ActualRunnerFailedInitializationRetiresItsOriginalMixedSource ActualRunnerBorrowedSourceRemainsWithItsOriginalOwner)
        set(_name "V2_Integration_NativeSourceFinalRetirement_${_case}Replay")
        add_v2_test(${_name} COMMAND $<TARGET_FILE:v2_test_native_source_final_retirement> --gtest_filter=NativeSourceFinalRetirement.${_case} --gtest_repeat=20
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;Replay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "GTEST_DEATH_TEST_STYLE=threadsafe")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_name})
    endforeach()
    endif()
endif()

if(TARGET v2_test_native_source_final_retirement)
    foreach(_case QFNParityFixtureSuccessRetiresOriginalObservationOwner QFNParityFixtureFailureRetiresOriginalObservationOwner)
        set(_name "V2_Integration_NativeSourceFinalRetirement_${_case}Replay")
        add_v2_test(${_name} COMMAND $<TARGET_FILE:v2_test_native_source_final_retirement>
            --gtest_filter=NativeSourceFinalRetirement.${_case} --gtest_repeat=20
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Lifecycle;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_name})
    endforeach()
endif()
