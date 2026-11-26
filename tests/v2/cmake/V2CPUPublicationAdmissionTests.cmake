# Model-free publication admission owns no MPI environment or model fixtures.
# The executable retains the native entrypoint tests and rejects MPI startup;
# one focused registration per actual runtime ISA preserves existing coverage.
add_executable(v2_integration_cpu_native_quantised_publication_admission
    integration/kernels/cpu/native_quantised/Test__CPUNativeQuantisedPublicationAdmission.cpp)
target_link_libraries(v2_integration_cpu_native_quantised_publication_admission
    llaminar2_core GTest::gtest ${CMAKE_DL_LIBS})
if(TARGET v2_integration_cpu_native_quantised_publication_admission)
    # Publish the test-only MPI tripwires into the actual process symbol scope.
    set_target_properties(v2_integration_cpu_native_quantised_publication_admission
        PROPERTIES ENABLE_EXPORTS TRUE)
endif()
target_include_directories(v2_integration_cpu_native_quantised_publication_admission PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})

# Every CPU NativeQuantised entry rejects unsealed or stale publications, on
# each native runtime ISA of this build (NEON, AVX2, AVX512).
set(V2_CPU_PUBLICATION_ADMISSION_ISAS ${V2_CPU_BASELINE_RUNTIME_ISA})
if(LLAMINAR_CPU_ISA_UPPER STREQUAL "AVX512")
    list(APPEND V2_CPU_PUBLICATION_ADMISSION_ISAS avx2 avx512)
endif()
list(REMOVE_DUPLICATES V2_CPU_PUBLICATION_ADMISSION_ISAS)
foreach(V2_ADMISSION_ISA IN LISTS V2_CPU_PUBLICATION_ADMISSION_ISAS)
    string(TOUPPER "${V2_ADMISSION_ISA}" V2_ADMISSION_ISA_LABEL)
    string(REPLACE "-" "_" V2_ADMISSION_ISA_LABEL "${V2_ADMISSION_ISA_LABEL}")
    add_v2_test(V2_Integration_CPU_NativeQuantisedPublicationAdmission_${V2_ADMISSION_ISA_LABEL}
        COMMAND $<TARGET_FILE:v2_integration_cpu_native_quantised_publication_admission>
                --gtest_filter=CPUNativeQuantisedPublicationAdmission.*
                --gtest_brief=1
        LABELS "V2;Integration;CPU;NativeQuantised;MXFP4;Publication;${V2_ADMISSION_ISA_LABEL};ProductionTestPreflight;Regression"
        MPI_PROCS 1
        NO_MPI
        NO_MODELS
        ENVIRONMENT "LLAMINAR_ISA_LEVEL=${V2_ADMISSION_ISA}")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
        V2_Integration_CPU_NativeQuantisedPublicationAdmission_${V2_ADMISSION_ISA_LABEL})
endforeach()
add_v2_test(V2_Integration_CPU_NativeQuantisedPublicationStartupGuard
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/integration/scripts/test_cpu_publication_startup_guard.py
        --binary $<TARGET_FILE:v2_integration_cpu_native_quantised_publication_admission>
    LABELS "V2;Integration;CPU;DeviceFree;NativeQuantised;Publication;ProductionTestPreflight;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_CPU_NativeQuantisedPublicationStartupGuard)
