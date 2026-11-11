# @file V2CPUArchitectureTests.cmake
# @brief Single registration authority for core-free CPU architecture regressions.
#
# The ordinary Unit/preflight gates and the focused portability build consume
# these same targets and contracts. No GPU, model or MPI initialization/session
# is needed; the ownership-header proof uses the real external MPI SDK.
# Every numerical test compiles the production implementation, not a substitute.
if(V2_PERF_TESTS_ONLY)
    return()
endif()
get_filename_component(V2_CPU_ARCHITECTURE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

include("${CMAKE_CURRENT_LIST_DIR}/V2CoreFreeTestRegistration.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/V2CPUImmutableLookupTests.cmake")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_executable(v2_test_cpu_compiler_configuration
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/Test__CPUCompilerConfiguration.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/CPUCompilerConfigurationProbe.c")
target_compile_features(v2_test_cpu_compiler_configuration PRIVATE c_std_11 cxx_std_20)
target_compile_definitions(v2_test_cpu_compiler_configuration PRIVATE
    LLAMINAR_TEST_EXPECT_OPTIMIZED_CONFIGURATION=$<NOT:$<CONFIG:Debug>>)
target_link_libraries(v2_test_cpu_compiler_configuration PRIVATE GTest::gtest_main)
foreach(_ctest_metadata_suite IN ITEMS Unit Integration)
    if(_ctest_metadata_suite STREQUAL "Unit")
        set(_ctest_metadata_labels V2 Unit CPU DeviceFree Build Ownership)
    else()
        set(_ctest_metadata_labels V2 Integration ProductionTestPreflight CPU DeviceFree Build Ownership Regression)
    endif()
    v2_add_core_free_test(V2_${_ctest_metadata_suite}_CTestTimeoutOwnerMetadata
        COMMAND "${Python3_EXECUTABLE}" -B
            "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_ctest_timeout_metadata.py"
            --cmake "${CMAKE_COMMAND}" --ctest "${CMAKE_CTEST_COMMAND}" --ninja "${CMAKE_MAKE_PROGRAM}"
        LABELS ${_ctest_metadata_labels})
    v2_add_core_free_test(V2_${_ctest_metadata_suite}_CPUCoreFreeSanitizerFailurePolicy
        COMMAND "${Python3_EXECUTABLE}" -B
            "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_core_free_sanitizer_failure_policy.py"
            --cmake "${CMAKE_COMMAND}" --ctest "${CMAKE_CTEST_COMMAND}" --ninja "${CMAKE_MAKE_PROGRAM}"
        LABELS ${_ctest_metadata_labels})
    v2_add_core_free_test(V2_${_ctest_metadata_suite}_CPUPortabilityIntegrationFlags
        COMMAND "${Python3_EXECUTABLE}" -B
            "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_portability_build_types.py"
            --cmake "${CMAKE_COMMAND}" --ninja "${CMAKE_MAKE_PROGRAM}"
        LABELS ${_ctest_metadata_labels})
    v2_add_core_free_test(V2_${_ctest_metadata_suite}_CPUCompilerConfiguration
        COMMAND $<TARGET_FILE:v2_test_cpu_compiler_configuration>
        LABELS ${_ctest_metadata_labels})
endforeach()
v2_add_core_free_test(V2_Integration_CTestBuildToolImageContracts
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/scripts/test_production_pipeline.py"
        ImageIdentityTests
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Build Ownership Regression)
v2_add_core_free_test(V2_Integration_DeviceFreeImageFixtureOwnership
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/scripts/test_production_pipeline.py"
        CrossHostRunnerTests.test_report_artifacts_survive_staging_cleanup_on_success_and_failure
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Build Ownership Regression)
foreach(_stage_header_suite IN ITEMS Unit Integration)
    if(_stage_header_suite STREQUAL "Unit")
        set(_stage_header_labels V2 Unit CPU DeviceFree Architecture)
    else()
        set(_stage_header_labels V2 Integration ProductionTestPreflight CPU DeviceFree Architecture Regression)
    endif()
    v2_add_core_free_test(V2_${_stage_header_suite}_CPUCacheAppendHeaderBoundary
        COMMAND "${Python3_EXECUTABLE}" -B
            "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_cpu_stage_header_boundary.py"
        LABELS ${_stage_header_labels})
endforeach()
v2_add_core_free_test(V2_Unit_CPUHostTopology
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_cpu_host_topology.py"
    LABELS V2 Unit CPU DeviceFree Topology Build)
v2_add_core_free_test(V2_Integration_CPUHybridTopologyAdmission
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_cpu_host_topology.py"
        CPUHostTopologyTests.test_hybrid_single_package_has_twenty_distinct_sibling_groups
        CPUHostTopologyTests.test_actual_cmake_gate_ignores_multiline_presentation_geometry
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Topology Build Regression)
v2_add_core_free_test(V2_Integration_CPUHostTopologyRejectUnknownGeometry
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_cpu_host_topology.py"
        CPUHostTopologyTests.test_missing_conflicting_and_unknown_identity_fail_closed
        CPUHostTopologyTests.test_capture_requires_complete_stable_original_affinity
        CPUHostTopologyTests.test_cmake_rejects_asymmetric_projection_and_noninteger_counts
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Topology Build Regression)
v2_add_core_free_test(V2_Integration_CPUHostTopologyReceiptRetirement
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_cpu_host_topology.py"
        CPUHostTopologyTests.test_actual_owner_retires_receipt_before_failed_readmission
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Topology Build Ownership Regression)
v2_add_core_free_test(V2_Unit_OneDNNBuildLifecycle
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_onednn_build_lifecycle.py"
        --cmake "${CMAKE_COMMAND}" --ninja "${CMAKE_MAKE_PROGRAM}"
    LABELS V2 Unit CPU DeviceFree Build Ownership)
v2_add_core_free_test(V2_Integration_OneDNNConfiguredBuildResume
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_onednn_build_lifecycle.py"
        --cmake "${CMAKE_COMMAND}" --ninja "${CMAKE_MAKE_PROGRAM}"
        OneDNNBuildLifecycleTest.test_failed_build_resumes_and_complete_install_reuses
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Build Ownership Regression)
v2_add_core_free_test(V2_Integration_OneDNNFailedPublication
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_onednn_build_lifecycle.py"
        --cmake "${CMAKE_COMMAND}" --ninja "${CMAKE_MAKE_PROGRAM}"
        OneDNNBuildLifecycleTest.test_failed_configure_exposes_both_streams_without_publication
        OneDNNBuildLifecycleTest.test_identity_change_invalidates_install_and_configured_artifacts
        OneDNNBuildLifecycleTest.test_missing_artifact_retires_success_before_failed_producer
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Build Ownership Regression)
v2_add_core_free_test(V2_Integration_OneDNNOriginAuthentication
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_onednn_build_lifecycle.py"
        --cmake "${CMAKE_COMMAND}" --ninja "${CMAKE_MAKE_PROGRAM}"
        OneDNNBuildLifecycleTest.test_origin_authentication_accepts_the_real_local_git_observation
        OneDNNBuildLifecycleTest.test_origin_mismatch_preserves_existing_source_and_configuration
        OneDNNBuildLifecycleTest.test_failed_git_probe_is_not_a_destructive_origin_mismatch
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Build Ownership Regression)
v2_add_core_free_test(V2_Integration_OneDNNRequiredInstallClosure
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_onednn_build_lifecycle.py"
        --cmake "${CMAKE_COMMAND}" --ninja "${CMAKE_MAKE_PROGRAM}"
        OneDNNBuildLifecycleTest.test_missing_library_retires_receipt_and_reinstalls_the_owned_closure
        OneDNNBuildLifecycleTest.test_missing_other_required_header_cannot_reuse_installation
        OneDNNBuildLifecycleTest.test_modified_artifact_is_not_an_authenticated_reuse
        OneDNNBuildLifecycleTest.test_successful_repair_restores_same_size_same_mtime_modified_header
        OneDNNBuildLifecycleTest.test_consumer_library_is_manifested_not_an_unlisted_priority_shadow
        OneDNNBuildLifecycleTest.test_rejected_surface_stays_rejected_across_two_invocations
        OneDNNBuildLifecycleTest.test_interrupted_partial_retirement_resumes_before_recertifying
        OneDNNBuildLifecycleTest.test_nested_directory_link_rejects_before_unpublished_retry_writes
        OneDNNBuildLifecycleTest.test_owned_versioned_library_file_links_remain_legal
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Build Ownership Regression)
v2_add_core_free_test(V2_Integration_OneDNNCompilerArgumentIdentity
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_onednn_build_lifecycle.py"
        --cmake "${CMAKE_COMMAND}" --ninja "${CMAKE_MAKE_PROGRAM}"
        OneDNNBuildLifecycleTest.test_compiler_arg1_is_forwarded_and_changes_identity
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Build ISA Regression)
add_executable(v2_test_q16_1_rmsnorm
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/ops/Test__Q16_1RMSNorm.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/cpu/primitives/RMSNormPrimitives.cpp")
target_compile_features(v2_test_q16_1_rmsnorm PRIVATE cxx_std_23)
target_include_directories(v2_test_q16_1_rmsnorm PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_test_q16_1_rmsnorm PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
v2_add_core_free_test(V2_Unit_Q16_1RMSNorm
    COMMAND $<TARGET_FILE:v2_test_q16_1_rmsnorm>
    LABELS V2 Unit CPU DeviceFree Kernels RMSNorm Q16_1 TypedResiduals)
v2_add_core_free_test(V2_Integration_CPURMSNormQ16LegacyCoverage
    COMMAND $<TARGET_FILE:v2_test_q16_1_rmsnorm>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm Q16_1 Regression)
# Pure-integer Q8_1 gamma is cached by owner control block, never by raw address.
add_executable(v2_test_cpu_rmsnorm_gamma_q11_identity
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/ops/Test__RMSNormGammaQ11Identity.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/cpu/primitives/RMSNormPrimitives.cpp")
target_compile_features(v2_test_cpu_rmsnorm_gamma_q11_identity PRIVATE cxx_std_23)
target_include_directories(v2_test_cpu_rmsnorm_gamma_q11_identity PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_test_cpu_rmsnorm_gamma_q11_identity PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
v2_add_core_free_test(V2_Unit_CPURMSNormGammaQ11Identity
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_gamma_q11_identity>
    LABELS V2 Unit CPU DeviceFree RMSNorm Q8_1 Ownership)
v2_add_core_free_test(V2_Integration_CPURMSNormGammaQ11OwnerIdentity
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_gamma_q11_identity>
        --gtest_filter=RMSNormGammaQ11Identity.NewOwnerAtReusedAddressNeverStaleHits:RMSNormGammaQ11Identity.ExpiredOwnerRequantizes:RMSNormGammaQ11Identity.UnownedGammaIsNeverCached
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm Q8_1 Ownership Regression)
# Compile the actual legacy precision caller TU, not a hand-copied surrogate.
# Its remaining precision/ISA cases retain their canonical full-core Unit gate;
# this core-free selection certifies only the admitted original-INT32 publisher.
add_executable(v2_test_cpu_rmsnorm_precision
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/ops/Test__RMSNormPrecision.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/cpu/primitives/RMSNormPrimitives.cpp")
target_compile_features(v2_test_cpu_rmsnorm_precision PRIVATE cxx_std_23)
target_include_directories(v2_test_cpu_rmsnorm_precision PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_test_cpu_rmsnorm_precision PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
v2_add_core_free_test(V2_Unit_CPURMSNormInt32PrecisionCaller
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_precision>
        --gtest_filter=Test__RMSNormPrecision.ISAParity_Int32OriginalPublication
    LABELS V2 Unit CPU DeviceFree RMSNorm INT32 Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormInt32PrecisionCaller
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_precision>
        --gtest_filter=Test__RMSNormPrecision.ISAParity_Int32OriginalPublication
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm INT32 Regression)
add_executable(v2_test_cpu_rmsnorm_workshare
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPURMSNormWorkshare.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUQuantizedRMSNorm.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUInt32RMSNorm.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUInt32RMSNormPublication.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/ops/Test__RMSNormINT32.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/cpu/primitives/RMSNormPrimitives.cpp")
target_compile_features(v2_test_cpu_rmsnorm_workshare PRIVATE cxx_std_23)
target_include_directories(v2_test_cpu_rmsnorm_workshare PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_test_cpu_rmsnorm_workshare PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
v2_add_core_free_test(V2_Unit_CPURMSNormWorkshare
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare>
        --gtest_filter=-*AllPositiveWorkersThroughTen*
    LABELS V2 Unit CPU DeviceFree RMSNorm Ownership)
v2_add_core_free_test(V2_Integration_CPURMSNormFP32Application
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare>
        --gtest_filter=CPURMSNormWorkshare.FP32ApplicationRetainsOriginalRoundedEdges
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm ByteExact Ownership Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormScalarFP32Summary
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare>
        --gtest_filter=CPURMSNormWorkshare.ScalarFP32SummaryRetainsIncreasingColumnFMA
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormInt32Precision
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare>
        --gtest_filter=CPUInt32RMSNorm.FullWidthSummaryNeverRoundsOriginalIntegersThroughFP32:CPUInt32RMSNorm.ExplicitScalarUsesOriginalIncreasingColumnFP64Program
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm INT32 Native ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormInt32Admission
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare>
        --gtest_filter=CPUInt32RMSNorm.OriginalBankIntersectionsAndUndeclaredPolicyRejectBeforeStores
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm INT32 Ownership Geometry Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormInt32Completion
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare>
        --gtest_filter=CPUInt32RMSNorm.EmptyRetainedSummaryCompletesBeforeRetirement:CPUInt32RMSNorm.SmallRetainedSummaryHasOnePublicationOwner
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm INT32 Ownership Workshare Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormInt32AllPositiveWorkers WHOLE_TEAM
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare>
        --gtest_filter=CPUInt32RMSNorm.AllPositiveWorkersThroughTenRetainOriginalSummaries
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm INT32 ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormInt32LegacyCoverage
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare> --gtest_filter=Test__RMSNormINT32.*
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm INT32 Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormInt32PublicationProgram
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare>
        --gtest_filter=CPUInt32RMSNormPublication.OriginalNativeRNEFragmentsAndAwayTails:CPUInt32RMSNormPublication.OriginalScaleHasNoForeignFloorAndZeroInverseDoesNotReadGamma:CPUInt32RMSNormPublication.ExplicitScalarFusedUsesSerialSummaryAndOriginalAwayPublication
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm INT32 ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormInt32PublicationAdmission
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare>
        --gtest_filter=CPUInt32RMSNormPublication.CompleteOriginalBanksAndPolicyRejectBeforePublication
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm INT32 Ownership Geometry Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormInt32PublicationCompletion
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare>
        --gtest_filter=CPUInt32RMSNormPublication.EmptyStagedAndFusedCallsCompleteRetainedTeam:CPUInt32RMSNormPublication.NonemptyStagedAndFusedCompleteBeforeImmediateReuse
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm INT32 Ownership Workshare Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormInt32PublicationNumeric
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare>
        --gtest_filter=CPUInt32RMSNormPublication.NonfiniteAndUnrepresentableRowsAreFatalNotZeroed
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm INT32 Numerical Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormInt32PublicationAllPositiveWorkers WHOLE_TEAM
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare>
        --gtest_filter=CPUInt32RMSNormPublication.AllPositiveWorkersThroughTenRetainNativeStagedFusedBytesAndGuards
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm INT32 ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormRetainedSummary
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare> --gtest_filter=*WideRetainedRowUsesCompleteOriginalSummary
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm Ownership Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormRetainedAlias
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare> --gtest_filter=*SmallRetainedAliasCompletesExactlyOnce
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm Ownership Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormEmptyCompletion
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare> --gtest_filter=*EmptyRetainedInvocationCompletesBeforeReturn
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm Ownership Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormOriginalFloatingRows
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare> --gtest_filter=*FloatingMatricesPreserveOriginalRowsAndSerialBytes
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm ByteExact)
v2_add_core_free_test(V2_Integration_CPURMSNormAllPositiveWorkers WHOLE_TEAM
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare> --gtest_filter=*AllPositiveWorkersThroughTenRetainSerialBytes
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm ByteExact)
v2_add_core_free_test(V2_Integration_CPURMSNormOriginalBankAliases
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare> --gtest_filter=*OriginalBankAndPolicyAliasesRejectBeforeMutation
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm Ownership Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormQ16Geometry
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare> --gtest_filter=*Q16CompleteGeometryRejectsBeforeStores
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm Q16 Geometry Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormQ16Expansion
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare> --gtest_filter=*Q16ExpansionProtectsCompleteOriginalBanks
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm Q16 Ownership Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormQ16EmptyCompletion
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare> --gtest_filter=*Q16EmptyRetainedInvocationCompletes
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm Q16 Ownership Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormQ16AllPositiveWorkers WHOLE_TEAM
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare> --gtest_filter=*Q16AllPositiveWorkersThroughTenRetainSerialBytes
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm Q16 ByteExact)
v2_add_core_free_test(V2_Integration_CPURMSNormQ16NumericalProgram
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare> --gtest_filter=*Q16RoundedSquaresAndOriginalMetadataMatchIndependentProgram
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm Q16 ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormQ16ReductionTreeIdentity
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare> --gtest_filter=*Q16ReductionTreeIdentityRemainsISASpecific
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm Q16 ISA ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormQ16Policy
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare> --gtest_filter=*Q16PolicyAndNonfiniteSummaryFailHard
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm Q16 Ownership Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormPortableHalfTail
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_workshare> --gtest_filter=*PortableHalfNormalRoundingIsLaneIndependent
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm FP16 ByteExact Regression)
add_executable(v2_test_cpu_workshare_timing
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUWorkshareTiming.cpp")
target_compile_features(v2_test_cpu_workshare_timing PRIVATE cxx_std_20)
target_include_directories(v2_test_cpu_workshare_timing PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_test_cpu_workshare_timing PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
v2_add_core_free_test(V2_Unit_CPUWorkshareTiming
    COMMAND $<TARGET_FILE:v2_test_cpu_workshare_timing>
    LABELS V2 Unit CPU DeviceFree Profiling Ownership)
v2_add_core_free_test(V2_Integration_CPUWorkshareTiming
    COMMAND $<TARGET_FILE:v2_test_cpu_workshare_timing>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Profiling Ownership)
v2_add_core_free_test(V2_Integration_CPUWorkshareTimingRetainedRelease
    COMMAND $<TARGET_FILE:v2_test_cpu_workshare_timing> --gtest_filter=*RetainedPublicationCompletesBeforeStackReuse
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Profiling Ownership Regression)
v2_add_core_free_test(V2_Integration_CPUWorkshareTimingLifecycle
    COMMAND $<TARGET_FILE:v2_test_cpu_workshare_timing> --gtest_filter=*PublicationStateRejectsRepeatedOperations
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Profiling Ownership Regression)
add_executable(v2_test_cpu_openmp_workshare
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__OpenMPWorkshare.cpp")
target_compile_features(v2_test_cpu_openmp_workshare PRIVATE cxx_std_23)
target_include_directories(v2_test_cpu_openmp_workshare PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_test_cpu_openmp_workshare PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
v2_add_core_free_test(V2_Unit_OpenMPWorkshareLexicalRetained
    COMMAND $<TARGET_FILE:v2_test_cpu_openmp_workshare>
    LABELS V2 Unit CPU DeviceFree OpenMP Workshare Ownership Regression)
v2_add_core_free_test(V2_Integration_OpenMPWorkshareLexicalRetained
    COMMAND $<TARGET_FILE:v2_test_cpu_openmp_workshare>
        --gtest_filter=OpenMPWorkshare.RetainedSerializedTeamReusesExactTeam:OpenMPWorkshare.RetainedInactiveTeamReusesExactTeam:OpenMPWorkshare.RegionIfRetainedVariantsStayDirect:OpenMPWorkshare.BeginWorkshareEmptySemantics
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree OpenMP Workshare Ownership Regression)
v2_add_core_free_test(V2_Integration_OpenMPWorkshareRetainedDepth
    COMMAND $<TARGET_FILE:v2_test_cpu_openmp_workshare>
        --gtest_filter=OpenMPWorkshare.StandalonePolicyIsUnchanged:OpenMPWorkshare.RetainedActiveTeamReusesExactTeam:OpenMPWorkshare.NestedSerializedTeamDepthIsStable:OpenMPWorkshare.CollapseFormsReuseRetainedTeam:OpenMPWorkshare.RepeatedRetainedCallsKeepConstantDepth:OpenMPWorkshare.ImmediateAfterReturnBeforeOuterJoin:OpenMPWorkshare.RetainedSingleExecutesExactlyOnce:OpenMPWorkshare.SingleNoWaitHasNoForcedBarrier:OpenMPWorkshare.RegionSyncAddsExplicitBarrierAfterNowaitWork:OpenMPWorkshare.LayerParallelReusesRetainedTeam:OpenMPWorkshare.CanonicalPlacementAuthority
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree OpenMP Workshare Ownership Regression)
v2_add_core_free_test(V2_Integration_OpenMPTeamForkCounterContract
    COMMAND $<TARGET_FILE:v2_test_cpu_openmp_workshare>
        --gtest_filter=OpenMPWorkshare.TeamForkCounterCountsExactlyTheRealForks:OpenMPWorkshare.TeamForkCounterPublicationCostAndDefault:OpenMPWorkshare.TeamForkScopeAttributesExactlyUnderOverlapAndNesting:OpenMPWorkshare.TeamForkWorkerContextForksNeverReachOwnerCells
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree OpenMP Workshare Ownership Regression)
v2_add_core_free_test(V2_Integration_OpenMPForkNoteSourcePolicy
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/scripts/check_cpu_openmp_fork_note_policy.py"
        --repo-root "${V2_CPU_ARCHITECTURE_ROOT}"
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree OpenMP Workshare SourcePolicy Regression)
v2_add_core_free_test(V2_Integration_OpenMPForkNoteSourcePolicyNegativeControls
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/scripts/test_cpu_openmp_fork_note_policy.py"
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree OpenMP Workshare SourcePolicy Regression)
add_executable(v2_test_cpu_rmsnorm_numeric_fatal
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPURMSNormNumericFatal.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/cpu/primitives/RMSNormPrimitives.cpp")
target_compile_features(v2_test_cpu_rmsnorm_numeric_fatal PRIVATE cxx_std_23)
target_include_directories(v2_test_cpu_rmsnorm_numeric_fatal PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_test_cpu_rmsnorm_numeric_fatal PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
v2_add_core_free_test(V2_Unit_CPURMSNormNumericFatal
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_numeric_fatal>
    LABELS V2 Unit CPU DeviceFree RMSNorm Numerical Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormNumericFatal
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_numeric_fatal>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm Numerical Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormHalfPolicyIdentity
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_numeric_fatal>
        --gtest_filter=CPURMSNormNumericFatal.HalfT5RejectedColdAndFatalInRetainedTeam:CPURMSNormNumericFatal.HalfNullGammaIdentityMatchesExplicitOnes:CPURMSNormNumericFatal.RowNullGammaIdentityMatchesExplicitOnes
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm FP16 Policy Identity Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormExplicitFatalSink
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_numeric_fatal>
        --gtest_filter=CPURMSNormNumericFatal.FP32MatrixNonfiniteSummaryIsExplicitFatal:CPURMSNormNumericFatal.FP32MatrixNonfiniteRetainedTeamIsExplicitFatal:CPURMSNormNumericFatal.FP32MatrixNonfiniteNestedRetainedIsExplicitFatal:CPURMSNormNumericFatal.FP32DirectRowZeroDenominatorIsExplicitFatal:CPURMSNormNumericFatal.HalfMatrixNonfiniteSummaryIsExplicitFatal:CPURMSNormNumericFatal.HalfMatrixNonfiniteRetainedTeamIsExplicitFatal:CPURMSNormNumericFatal.HalfZeroDenominatorIsFatalAndPositiveEpsilonLegal:CPURMSNormNumericFatal.T5FiniteOverflowSummaryIsFatal:CPURMSNormNumericFatal.TinyOnlyRowRoundedFP32InverseOverflowIsFatal:CPURMSNormNumericFatal.DirectSIMDLeavesGateRangesAndTails:CPURMSNormNumericFatal.MultiOwnerMatrixEmitsIntactTypedFatal:CPURMSNormNumericFatal.Q16NonfiniteScaleIsExplicitFatal
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm Numerical Fatal Regression)
v2_add_core_free_test(V2_Integration_CPURMSNormLegalNumericPositives
    COMMAND $<TARGET_FILE:v2_test_cpu_rmsnorm_numeric_fatal>
        --gtest_filter=CPURMSNormNumericFatal.GuardedZeroAndLargeFinitePositivesRemainLegal:CPURMSNormNumericFatal.T5ZeroRowPublishesLegalGuardedZeros:CPURMSNormNumericFatal.GuardedZeroRowOutputRemainsZeroWithNonfiniteGamma:CPURMSNormNumericFatal.ZeroRowWithPositiveEpsilonAndValidGammaStaysLegal:CPURMSNormNumericFatal.ColdEpsilonPolicyRemainsStandaloneRecoverable
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RMSNorm Numerical Positive Regression)
add_executable(v2_unit_cpu_aligned_storage_header
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUAlignedStorageHeader.cpp")
target_compile_features(v2_unit_cpu_aligned_storage_header PRIVATE cxx_std_23)
target_include_directories(v2_unit_cpu_aligned_storage_header PRIVATE "${V2_CPU_ARCHITECTURE_ROOT}/src/v2")
target_link_libraries(v2_unit_cpu_aligned_storage_header PRIVATE GTest::gtest_main)
v2_add_core_free_test(V2_Unit_CPUAlignedStorageHeader
    COMMAND $<TARGET_FILE:v2_unit_cpu_aligned_storage_header>
    LABELS V2 Unit CPU DeviceFree Interface Ownership)
v2_add_core_free_test(V2_Integration_CPUAlignedStorageHeader
    COMMAND $<TARGET_FILE:v2_unit_cpu_aligned_storage_header>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Interface Ownership Regression)

# Header ownership is tested with the real external MPI SDK, without MPI_Init,
# device discovery or an alternate topology definition. Full runtime ABI gates
# remain separate; this does not qualify the native ORTE/session closure.
if(NOT TARGET MPI::MPI_CXX)
    find_package(MPI REQUIRED COMPONENTS CXX)
endif()
add_executable(v2_unit_mpi_context_header
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/utils/Test__MPIContextHeader.cpp")
target_compile_features(v2_unit_mpi_context_header PRIVATE cxx_std_23)
target_include_directories(v2_unit_mpi_context_header PRIVATE "${V2_CPU_ARCHITECTURE_ROOT}/src/v2")
target_link_libraries(v2_unit_mpi_context_header PRIVATE GTest::gtest_main MPI::MPI_CXX OpenMP::OpenMP_CXX)
v2_add_core_free_test(V2_Unit_MPIContextOwnershipHeader
    COMMAND $<TARGET_FILE:v2_unit_mpi_context_header>
    LABELS V2 Unit CPU DeviceFree Interface Ownership MPI)
v2_add_core_free_test(V2_Integration_MPIContextOwnershipHeader
    COMMAND $<TARGET_FILE:v2_unit_mpi_context_header>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Interface Ownership MPI Regression)
v2_add_core_free_test(V2_Integration_CPUCodegenTargetManifestMultiLanguage
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_cpu_codegen_target_manifest.py"
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree ISA Build Regression)
v2_add_core_free_test(V2_Unit_CPUCodegenEvidenceLifecycle
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_cpu_codegen_envelope.py"
    LABELS V2 Unit CPU DeviceFree ISA Build Regression)
# The witness checker tests run in shards: the compiler-free analysis classes, and the proof-binding tests
# (which spawn stand-in compiler/disassembler processes) split four ways, each well inside the timeout.
foreach(v2_tile_register_shard IN ITEMS analysis binding-0 binding-1 binding-2 binding-3)
    string(REPLACE "-" "" v2_tile_register_shard_suffix "${v2_tile_register_shard}")
    string(SUBSTRING "${v2_tile_register_shard_suffix}" 0 1 v2_tile_register_first)
    string(TOUPPER "${v2_tile_register_first}" v2_tile_register_first)
    string(SUBSTRING "${v2_tile_register_shard_suffix}" 1 -1 v2_tile_register_rest)
    set(v2_tile_register_name "${v2_tile_register_first}${v2_tile_register_rest}")
    v2_add_core_free_test(V2_Unit_CPUNativeTileRegisterEvidence${v2_tile_register_name}
        COMMAND "${Python3_EXECUTABLE}" -B
            "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_cpu_native_tile_register_evidence.py"
            --shard "${v2_tile_register_shard}"
        LABELS V2 Unit CPU DeviceFree ISA Build Regression)
    v2_add_core_free_test(V2_Integration_CPUNativeTileRegisterEvidence${v2_tile_register_name}
        COMMAND "${Python3_EXECUTABLE}" -B
            "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_cpu_native_tile_register_evidence.py"
            --shard "${v2_tile_register_shard}"
        LABELS V2 Integration ProductionTestPreflight CPU DeviceFree ISA Build Regression)
endforeach()
v2_add_core_free_test(V2_Integration_CPUCodegenFailedRebuildInvalidation
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_cpu_codegen_envelope.py"
        CPUCodegenEnvelopeEvidenceTest.test_failed_rebuild_retires_previous_success
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree ISA Build Regression)
v2_add_core_free_test(V2_Integration_CPUCodegenNestedResponseExtent
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_cpu_codegen_envelope.py"
        CPUCodegenEnvelopeEvidenceTest.test_nested_response_changes_cycles_and_missing_inputs_are_rejected
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree ISA Build Regression)
v2_add_core_free_test(V2_Integration_CPUCodegenNativeAssemblyEvidence
    COMMAND "${Python3_EXECUTABLE}" -B
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/test_cpu_codegen_envelope.py"
        CPUCodegenEnvelopeEvidenceTest.test_empty_and_ir_only_witnesses_are_rejected
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree ISA Build Regression)

if(LLAMINAR_CPU_TARGET_PROCESSOR MATCHES "^(arm64|aarch64|ARM64|AARCH64)$")
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    add_executable(v2_test_neon_rmsnorm_reduction
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__NEONRMSNormReduction.cpp")
    target_compile_features(v2_test_neon_rmsnorm_reduction PRIVATE cxx_std_20)
    target_include_directories(v2_test_neon_rmsnorm_reduction PRIVATE "${V2_CPU_ARCHITECTURE_ROOT}/src/v2")
    target_link_libraries(v2_test_neon_rmsnorm_reduction PRIVATE GTest::gtest_main)
    v2_add_core_free_test(V2_Unit_NEONRMSNormReduction
        COMMAND $<TARGET_FILE:v2_test_neon_rmsnorm_reduction>
        LABELS V2 Unit CPU DeviceFree NEON RMSNorm ByteExact)
    v2_add_core_free_test(V2_Integration_NEONRMSNormReduction
        COMMAND $<TARGET_FILE:v2_test_neon_rmsnorm_reduction>
        LABELS V2 Integration ProductionTestPreflight CPU DeviceFree NEON RMSNorm ByteExact)
    v2_add_core_free_test(V2_Integration_NEONRMSNormOriginalBorrow
        COMMAND $<TARGET_FILE:v2_test_neon_rmsnorm_reduction>
            --gtest_filter=*OriginalBorrowGeometryRejectsBeforeArithmetic
        LABELS V2 Integration ProductionTestPreflight CPU DeviceFree NEON RMSNorm Ownership Regression)
    v2_add_core_free_test(V2_Integration_NEONRMSNormInt32Width
        COMMAND $<TARGET_FILE:v2_test_neon_rmsnorm_reduction>
            --gtest_filter=*IntegerOriginalRowsRetainFullWidthAndSerialBytes
        LABELS V2 Integration ProductionTestPreflight CPU DeviceFree NEON RMSNorm ByteExact Regression)
    add_executable(v2_test_neon_gdn_normalization
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__NEONGDNNormalization.cpp")
    target_compile_features(v2_test_neon_gdn_normalization PRIVATE cxx_std_20)
    target_include_directories(v2_test_neon_gdn_normalization PRIVATE "${V2_CPU_ARCHITECTURE_ROOT}/src/v2")
    target_link_libraries(v2_test_neon_gdn_normalization PRIVATE GTest::gtest_main)
    v2_add_core_free_test(V2_Unit_NEONGDNNormalization
        COMMAND $<TARGET_FILE:v2_test_neon_gdn_normalization>
        LABELS V2 Unit CPU DeviceFree NEON GDN ByteExact)
    v2_add_core_free_test(V2_Integration_NEONGDNNormalization
        COMMAND $<TARGET_FILE:v2_test_neon_gdn_normalization>
        LABELS V2 Integration ProductionTestPreflight CPU DeviceFree NEON GDN ByteExact Regression)
    add_executable(v2_test_neon_attention_math
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__NEONAttentionMath.cpp"
        "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/cpu/primitives/VectorPrimitives.cpp")
    target_compile_features(v2_test_neon_attention_math PRIVATE cxx_std_20)
    target_include_directories(v2_test_neon_attention_math PRIVATE
        "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
    target_link_libraries(v2_test_neon_attention_math PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
    v2_add_core_free_test(V2_Unit_NEONAttentionMath
        COMMAND $<TARGET_FILE:v2_test_neon_attention_math>
        LABELS V2 Unit CPU DeviceFree Attention ByteExact)
    v2_add_core_free_test(V2_Integration_NEONAttentionMath
        COMMAND $<TARGET_FILE:v2_test_neon_attention_math>
        LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention ByteExact)
    v2_add_core_free_test(V2_Integration_NEONAttentionFloatingRows
        COMMAND $<TARGET_FILE:v2_test_neon_attention_math> --gtest_filter=*FloatingOriginalCacheRowsRetainSerialBytes
        LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention ByteExact)
    # The row programs that trust admitted geometry are private to the owning
    # kernel (compile-time witness) and the integer width guard is fatal.
    v2_add_core_free_test(V2_Integration_NEONAttentionAdmittedRowsBoundary
        COMMAND $<TARGET_FILE:v2_test_neon_attention_math> --gtest_filter=*PrivateIntegerRowsRejectUnprovenWidths
        LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention Regression)
    # Int32 accumulation is exact only under the typed Q16 reduction bound:
    # equality with int64 sums, closed-form worst cases and fatal rejection.
    v2_add_core_free_test(V2_Integration_NEONAttentionQ16Int32Bound
        COMMAND $<TARGET_FILE:v2_test_neon_attention_math>
            --gtest_filter=*IntegerOriginalRowsAccumulateExactlyUnderTheTypedBound:*Q16BoundWorstCaseWitnessesStayExact:*Q16BoundRejectsUnprovenReductions:*Q16BoundRejectsInvalidTailCodesBeforeTailArithmetic
        LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention ByteExact Regression)
    v2_add_core_free_test(V2_Integration_NEONAttentionQueryQuantization
        COMMAND $<TARGET_FILE:v2_test_neon_attention_math> --gtest_filter=*OriginalQueryQuantizationProgram
        LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention ByteExact)
    v2_add_core_free_test(V2_Integration_NEONAttentionStorageAdmission
        COMMAND $<TARGET_FILE:v2_test_neon_attention_math> --gtest_filter=*OriginalStorageAndDomainAdmission:*NonfiniteWeightIsFatalBeforeAnySharedNumeratorWrite
        LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention Regression)
endif()
add_executable(v2_unit_cpu_isa_profile
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/utils/Test__CPUISAProfile.cpp")
target_compile_features(v2_unit_cpu_isa_profile PRIVATE cxx_std_20)
target_include_directories(v2_unit_cpu_isa_profile PRIVATE "${V2_CPU_ARCHITECTURE_ROOT}/src/v2")
target_link_libraries(v2_unit_cpu_isa_profile PRIVATE GTest::gtest_main)

if(LLAMINAR_CPU_TARGET_PROCESSOR MATCHES "^(arm64|aarch64|ARM64|AARCH64)$")
    set(V2_CPU_CODEGEN_SOURCE "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/portability/CPUCodegenEnvelopeProbe.cpp")
    set(V2_CPU_CODEGEN_POLICY "${V2_CPU_ARCHITECTURE_ROOT}/cmake/CPUInstructionSet.cmake")
    set(V2_CPU_CODEGEN_CHECKER "${CMAKE_CURRENT_LIST_DIR}/CPUCodegenEnvelope.py")
    set(V2_CPU_CODEGEN_PROOF "${CMAKE_CURRENT_BINARY_DIR}/cpu_codegen_envelope/proof.json")
    set(V2_CPU_CODEGEN_TARGET_SOURCE "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/utils/Test__CPUISAProfile.cpp")
    set(V2_CPU_CODEGEN_DATABASE "${CMAKE_BINARY_DIR}/compile_commands.json")
    set(V2_CPU_CODEGEN_TARGET_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/cpu_codegen_envelope/target.txt")
    string(TOUPPER "${CMAKE_BUILD_TYPE}" V2_CPU_CODEGEN_CONFIGURATION)
    set(V2_CPU_CODEGEN_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${V2_CPU_CODEGEN_CONFIGURATION}}")
    if(APPLE AND CMAKE_OSX_ARCHITECTURES)
        string(APPEND V2_CPU_CODEGEN_FLAGS " -arch ${CMAKE_OSX_ARCHITECTURES}")
    endif()
    string(SHA256 V2_CPU_CODEGEN_SIGNATURE
        "${CMAKE_CXX_COMPILER}|${CMAKE_CXX_COMPILER_ID}|${CMAKE_CXX_COMPILER_VERSION}|${V2_CPU_CODEGEN_FLAGS}|${CMAKE_CXX_COMPILER_TARGET}|${CMAKE_SYSROOT}|${CMAKE_OSX_ARCHITECTURES}")
    # Generation observes final target properties, including options added by
    # callers after this module. The build borrows the actual ordered command
    # from CMake's compilation database rather than reconstructing its ISA flags.
    include("${CMAKE_CURRENT_LIST_DIR}/V2CPUCodegenTargetManifest.cmake")
    v2_write_cpu_codegen_target_manifest(v2_unit_cpu_isa_profile
        "${V2_CPU_CODEGEN_TARGET_MANIFEST}" "${V2_CPU_CODEGEN_FLAGS}")
    # Compile the isolated optimized ISA motif during the build, independently
    # of sanitizer/debug instrumentation. Verification never needs a compiler
    # in an installed runner; retain the small JSON and assembly together.
    add_custom_command(OUTPUT "${V2_CPU_CODEGEN_PROOF}"
            "${CMAKE_CURRENT_BINARY_DIR}/cpu_codegen_envelope/proof.s"
        COMMAND "${Python3_EXECUTABLE}" "${V2_CPU_CODEGEN_CHECKER}" build
            "--compiler=${CMAKE_CXX_COMPILER}" "--compiler-version=${CMAKE_CXX_COMPILER_VERSION}"
            "--compiler-id=${CMAKE_CXX_COMPILER_ID}"
            "--compile-database=${V2_CPU_CODEGEN_DATABASE}" "--target-source=${V2_CPU_CODEGEN_TARGET_SOURCE}"
            "--target-manifest=${V2_CPU_CODEGEN_TARGET_MANIFEST}" "--mode=${LLAMINAR_CPU_ISA_UPPER}"
            "--source=${V2_CPU_CODEGEN_SOURCE}" "--policy=${V2_CPU_CODEGEN_POLICY}"
            "--signature=${V2_CPU_CODEGEN_SIGNATURE}" "--proof=${V2_CPU_CODEGEN_PROOF}"
        DEPENDS "${V2_CPU_CODEGEN_SOURCE}" "${V2_CPU_CODEGEN_POLICY}"
            "${V2_CPU_CODEGEN_CHECKER}" "${CMAKE_CXX_COMPILER}"
            "${V2_CPU_CODEGEN_DATABASE}" "${V2_CPU_CODEGEN_TARGET_MANIFEST}"
        VERBATIM)
    add_custom_target(v2_cpu_codegen_envelope_proof DEPENDS "${V2_CPU_CODEGEN_PROOF}"
        "${CMAKE_CURRENT_BINARY_DIR}/cpu_codegen_envelope/proof.s")
    add_dependencies(v2_unit_cpu_isa_profile v2_cpu_codegen_envelope_proof)
    # Register-allocation witness for every native NEON tile K-loop program:
    # the probe object is compiled with the same target command at -O3 and its
    # disassembly (the configured objdump) is audited. Structural failures are
    # fatal; single-row (decode) programs must not spill; grouped row tiles may
    # not exceed the committed ratchet budget for the compiler identity (derived
    # from the compiler and checked against the configured one) and ISA
    # envelope. The proof binds the compilation's own dependency closure (also
    # the depfile, so any header change re-runs the witness), the executed
    # command, the object, the disassembler and the budget; preflight
    # re-disassembles the bound object and re-verifies it without a compiler.
    if(NOT CMAKE_OBJDUMP)
        message(FATAL_ERROR "the native NEON tile register witness requires CMAKE_OBJDUMP (GNU binutils or LLVM)")
    endif()
    set(V2_CPU_TILE_REGISTER_SOURCE "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/portability/CPUNativeNEONTileRegisterProbe.cpp")
    set(V2_CPU_TILE_REGISTER_CHECKER "${CMAKE_CURRENT_LIST_DIR}/CPUNativeTileRegisterEvidence.py")
    set(V2_CPU_TILE_REGISTER_BUDGET "${CMAKE_CURRENT_LIST_DIR}/CPUNativeTileRegisterBudget.json")
    set(V2_CPU_TILE_REGISTER_PROOF "${CMAKE_CURRENT_BINARY_DIR}/cpu_native_tile_registers/proof.json")
    string(REGEX MATCH "^[0-9]+" V2_CPU_TILE_REGISTER_MAJOR "${CMAKE_CXX_COMPILER_VERSION}")
    set(V2_CPU_TILE_REGISTER_IDENTITY "${CMAKE_CXX_COMPILER_ID}-${V2_CPU_TILE_REGISTER_MAJOR}")
    set(V2_CPU_TILE_REGISTER_ARGS
        "--compiler=${CMAKE_CXX_COMPILER}" "--compile-database=${V2_CPU_CODEGEN_DATABASE}"
        "--target-source=${V2_CPU_CODEGEN_TARGET_SOURCE}" "--source=${V2_CPU_TILE_REGISTER_SOURCE}"
        "--signature=${V2_CPU_CODEGEN_SIGNATURE}" "--proof=${V2_CPU_TILE_REGISTER_PROOF}"
        "--identity=${V2_CPU_TILE_REGISTER_IDENTITY}" "--mode=${LLAMINAR_CPU_ISA_UPPER}"
        "--disassembler=${CMAKE_OBJDUMP}")
    # Ninja depfile targets are relative to the current binary directory under
    # CMP0116 NEW and to the top binary directory otherwise.
    cmake_policy(GET CMP0116 V2_CPU_TILE_REGISTER_DEPFILE_POLICY)
    if(V2_CPU_TILE_REGISTER_DEPFILE_POLICY STREQUAL "NEW")
        file(RELATIVE_PATH V2_CPU_TILE_REGISTER_DEPFILE_TARGET "${CMAKE_CURRENT_BINARY_DIR}" "${V2_CPU_TILE_REGISTER_PROOF}")
    else()
        file(RELATIVE_PATH V2_CPU_TILE_REGISTER_DEPFILE_TARGET "${CMAKE_BINARY_DIR}" "${V2_CPU_TILE_REGISTER_PROOF}")
    endif()
    set(V2_CPU_TILE_REGISTER_ARTIFACTS
        "${CMAKE_CURRENT_BINARY_DIR}/cpu_native_tile_registers/proof.o"
        "${CMAKE_CURRENT_BINARY_DIR}/cpu_native_tile_registers/proof.dis"
        "${CMAKE_CURRENT_BINARY_DIR}/cpu_native_tile_registers/proof.i")
    add_custom_command(OUTPUT "${V2_CPU_TILE_REGISTER_PROOF}" ${V2_CPU_TILE_REGISTER_ARTIFACTS}
        BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/cpu_native_tile_registers/proof.deps"
            "${CMAKE_CURRENT_BINARY_DIR}/cpu_native_tile_registers/proof.pre.deps"
        COMMAND "${Python3_EXECUTABLE}" "${V2_CPU_TILE_REGISTER_CHECKER}" build ${V2_CPU_TILE_REGISTER_ARGS}
            "--depfile-target=${V2_CPU_TILE_REGISTER_DEPFILE_TARGET}"
        DEPENDS "${V2_CPU_TILE_REGISTER_SOURCE}" "${V2_CPU_TILE_REGISTER_CHECKER}" "${V2_CPU_CODEGEN_CHECKER}"
            "${V2_CPU_TILE_REGISTER_BUDGET}" "${CMAKE_CXX_COMPILER}" "${V2_CPU_CODEGEN_DATABASE}" "${CMAKE_OBJDUMP}"
        DEPFILE "${CMAKE_CURRENT_BINARY_DIR}/cpu_native_tile_registers/proof.d"
        VERBATIM)
    add_custom_target(v2_cpu_native_tile_register_proof
        DEPENDS "${V2_CPU_TILE_REGISTER_PROOF}" ${V2_CPU_TILE_REGISTER_ARTIFACTS})
    add_dependencies(v2_unit_cpu_isa_profile v2_cpu_native_tile_register_proof)
    foreach(scope IN ITEMS Unit Integration)
        set(labels V2 ${scope} CPU DeviceFree ISA Build Regression)
        if(scope STREQUAL "Integration")
            list(APPEND labels ProductionTestPreflight)
        endif()
        v2_add_core_free_test(V2_${scope}_CPUNativeNEONTileRegisters
            COMMAND "${Python3_EXECUTABLE}" "${V2_CPU_TILE_REGISTER_CHECKER}" verify ${V2_CPU_TILE_REGISTER_ARGS}
            LABELS ${labels})
    endforeach()
    foreach(scope IN ITEMS Unit Integration)
        set(labels V2 ${scope} CPU DeviceFree ISA Build Regression)
        if(scope STREQUAL "Integration")
            list(APPEND labels ProductionTestPreflight)
        endif()
        v2_add_core_free_test(V2_${scope}_CPUCodegenEnvelope
            COMMAND "${Python3_EXECUTABLE}" "${V2_CPU_CODEGEN_CHECKER}" verify
                "--compiler=${CMAKE_CXX_COMPILER}" "--compile-database=${V2_CPU_CODEGEN_DATABASE}"
                "--target-source=${V2_CPU_CODEGEN_TARGET_SOURCE}"
                "--source=${V2_CPU_CODEGEN_SOURCE}" "--policy=${V2_CPU_CODEGEN_POLICY}"
                "--mode=${LLAMINAR_CPU_ISA_UPPER}" "--signature=${V2_CPU_CODEGEN_SIGNATURE}"
                "--proof=${V2_CPU_CODEGEN_PROOF}"
            LABELS ${labels})
    endforeach()
endif()

v2_add_core_free_test(V2_Unit_CPUISAProfile
    COMMAND $<TARGET_FILE:v2_unit_cpu_isa_profile>
    LABELS V2 Unit CPU DeviceFree ISA)
v2_add_core_free_test(V2_Integration_CPUISAProfile
    COMMAND $<TARGET_FILE:v2_unit_cpu_isa_profile>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree ISA Regression)

v2_add_core_free_test(V2_Integration_CPUCacheObservationNoInventedCapacity
    COMMAND $<TARGET_FILE:v2_unit_cpu_isa_profile> --gtest_filter=CPUCacheObservation.*
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree ISA Geometry Regression)

include("${CMAKE_CURRENT_LIST_DIR}/V2CPUAttentionKeyQ8Tests.cmake")

add_executable(v2_test_cpu_softmax
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUSoftmax.cpp")
target_compile_features(v2_test_cpu_softmax PRIVATE cxx_std_20)
target_include_directories(v2_test_cpu_softmax PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_test_cpu_softmax PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
v2_add_core_free_test(V2_Unit_CPUSoftmax WHOLE_TEAM
    COMMAND $<TARGET_FILE:v2_test_cpu_softmax> --gtest_filter=-*AllPositiveWorkers*
    LABELS V2 Unit CPU DeviceFree Softmax ByteExact)
v2_add_core_free_test(V2_Integration_CPUSoftmax WHOLE_TEAM
    COMMAND $<TARGET_FILE:v2_test_cpu_softmax> --gtest_filter=-*AllPositiveWorkers*
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Softmax ByteExact)
v2_add_core_free_test(V2_Integration_CPUSoftmaxPolynomialLinearTerm
    COMMAND $<TARGET_FILE:v2_test_cpu_softmax> --gtest_filter=*ExponentialPolynomialLinearTermAndAccuracy
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Softmax NumericalContract Regression)
v2_add_core_free_test(V2_Integration_CPUSoftmaxQ8FP16Metadata
    COMMAND $<TARGET_FILE:v2_test_cpu_softmax> --gtest_filter=*Q8FP16MetadataAndSignedEndpoint
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Softmax Quantization Regression)
v2_add_core_free_test(V2_Integration_CPUSoftmaxQ8MinimumScale
    COMMAND $<TARGET_FILE:v2_test_cpu_softmax> --gtest_filter=*Q8FP16MinimumScaleRetainsLongUniformRows
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Softmax Quantization Regression)
v2_add_core_free_test(V2_Integration_CPUSoftmaxProbabilityAccuracy
    COMMAND $<TARGET_FILE:v2_test_cpu_softmax> --gtest_filter=*ProbabilityAccuracyAgainstDoubleExponentials
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Softmax NumericalContract)
v2_add_core_free_test(V2_Integration_CPUSoftmaxOriginalColumnPredicate
    COMMAND $<TARGET_FILE:v2_test_cpu_softmax> --gtest_filter=*OriginalColumnPredicateRetainsINTMAXBoundary
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Softmax Regression)
v2_add_core_free_test(V2_Integration_CPUSoftmaxFP16FragmentAlignment
    COMMAND $<TARGET_FILE:v2_test_cpu_softmax> --gtest_filter=*UnalignedFP16OriginalAndProbabilityFragments
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Softmax Regression)
v2_add_core_free_test(V2_Integration_CPUSoftmaxGeometry
    COMMAND $<TARGET_FILE:v2_test_cpu_softmax> --gtest_filter=*OriginalGeometryRejectsBeforeMutation
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Softmax Regression)
# One storage per cell keeps each whole-team sweep inside the 30 s budget on
# hosts with many logical CPUs; together they cover the original program.
foreach(_softmax_storage IN ITEMS FP32 BF16 FP16 Q8_1)
    v2_add_core_free_test(V2_Integration_CPUSoftmaxRetainedOriginalRows${_softmax_storage} WHOLE_TEAM
        COMMAND $<TARGET_FILE:v2_test_cpu_softmax>
            --gtest_filter=CPUSoftmax.AllPositiveWorkersRetainOriginalSerialRows${_softmax_storage}
        LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Softmax ByteExact Regression)
endforeach()
v2_add_core_free_test(V2_Integration_CPUSoftmaxEmptyCompletion
    COMMAND $<TARGET_FILE:v2_test_cpu_softmax> --gtest_filter=*EmptyRetainedWorkshareCompletesBeforeRelease
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Softmax Regression)
if(LLAMINAR_CPU_TARGET_PROCESSOR MATCHES "^(arm64|aarch64|ARM64|AARCH64)$")
    v2_add_core_free_test(V2_Integration_NEONSoftmaxOrderedRows
        COMMAND $<TARGET_FILE:v2_test_cpu_softmax> --gtest_filter=NEONSoftmax.OrderedOriginalRowsAllStorage
        LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Softmax ByteExact)
    v2_add_core_free_test(V2_Integration_NEONSoftmaxExpAccuracy
        COMMAND $<TARGET_FILE:v2_test_cpu_softmax> --gtest_filter=NEONSoftmax.ExponentialAccuracyAndSubnormalProgram
        LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Softmax NumericalContract)
    v2_add_core_free_test(V2_Integration_NEONSoftmaxNumericAdmission
        COMMAND $<TARGET_FILE:v2_test_cpu_softmax> --gtest_filter=NEONSoftmax.ActiveNumericFailuresRejectBeforeMutation
        LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Softmax Regression)
endif()

add_executable(v2_test_cpu_tq_rotation
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUTQRotation.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/cpu/primitives/VectorPrimitives.cpp")
target_compile_features(v2_test_cpu_tq_rotation PRIVATE cxx_std_20)
target_include_directories(v2_test_cpu_tq_rotation PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_test_cpu_tq_rotation PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
v2_add_core_free_test(V2_Unit_CPUTQRotation WHOLE_TEAM
    COMMAND $<TARGET_FILE:v2_test_cpu_tq_rotation> --gtest_filter=-*AllPositiveWorkers*
    LABELS V2 Unit CPU DeviceFree Attention TurboQuant Rotation ByteExact)
v2_add_core_free_test(V2_Integration_CPUTQRotation WHOLE_TEAM
    COMMAND $<TARGET_FILE:v2_test_cpu_tq_rotation>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention TurboQuant Rotation ByteExact)
v2_add_core_free_test(V2_Integration_CPUTQRotationFringes
    COMMAND $<TARGET_FILE:v2_test_cpu_tq_rotation> --gtest_filter=*OriginalGeometryFringesRetainArithmeticAndCanaries
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention TurboQuant Rotation ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPUTQRotationStorageAdmission
    COMMAND $<TARGET_FILE:v2_test_cpu_tq_rotation> --gtest_filter=*OriginalStorageAndGeometryRejectBeforeMutation
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention TurboQuant Rotation Regression)
v2_add_core_free_test(V2_Integration_CPUTQRotationFiniteVerifier
    COMMAND $<TARGET_FILE:v2_test_cpu_tq_rotation> --gtest_filter=*NonfiniteVerifierObservationsFailClosed
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention TurboQuant Rotation Regression)
v2_add_core_free_test(V2_Integration_CPUTQRotationOriginalHeads WHOLE_TEAM
    COMMAND $<TARGET_FILE:v2_test_cpu_tq_rotation> --gtest_filter=*AllPositiveWorkersRetainOriginalSerialHeadBytes
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention TurboQuant Rotation ByteExact)

add_executable(v2_test_qwen_sparse_native_bank
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/attention/Test__QwenSparseNativeBank.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/qwen4exp/QwenSparseAttention.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/qwen4exp/Qwen4ExpStateIO.cpp")
target_compile_features(v2_test_qwen_sparse_native_bank PRIVATE cxx_std_23)
target_include_directories(v2_test_qwen_sparse_native_bank PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_test_qwen_sparse_native_bank PRIVATE GTest::gtest_main)
v2_add_core_free_test(V2_Unit_QwenSparseNativeBank
    COMMAND $<TARGET_FILE:v2_test_qwen_sparse_native_bank>
    LABELS V2 Unit CPU DeviceFree Attention QwenSparse Ownership)
v2_add_core_free_test(V2_Integration_QwenSparseNativeBank
    COMMAND $<TARGET_FILE:v2_test_qwen_sparse_native_bank>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse Ownership Regression)
v2_add_core_free_test(V2_Integration_QwenSparseNativeBankAddressAdmission
    COMMAND $<TARGET_FILE:v2_test_qwen_sparse_native_bank>
        --gtest_filter=*WrappingAndUnalignedOriginalAddressDomainsReject:*ActualCommonValidatorRejectsWrappingOriginalBank
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse Ownership Regression)
v2_add_core_free_test(V2_Integration_QwenSparseNativeBankPolicyAdmission
    COMMAND $<TARGET_FILE:v2_test_qwen_sparse_native_bank>
        --gtest_filter=*UnknownOrIncompleteNativePoliciesReject:*ActualCommonValidatorRejectsUnrepresentableGQAAndRotaryPolicies:*ExactPhysicalElementAlignmentIsNotSIMDOrUniversalAlignment
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse Ownership Regression)
v2_add_core_free_test(V2_Integration_QwenSparseNativeBankOriginalExclusions
    COMMAND $<TARGET_FILE:v2_test_qwen_sparse_native_bank>
        --gtest_filter=*EveryCodecRoleAndLayoutRetainsOriginalSlices:*CompleteOriginalCapacityAndMetadataExclusions:*PostAdmissionCoordinateCorruptionIsFatal
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse Ownership Regression)
add_executable(v2_test_cpu_qwen_sparse_native_codec
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/attention/Test__CPUQwenSparseNativeCodec.cpp")
target_compile_features(v2_test_cpu_qwen_sparse_native_codec PRIVATE cxx_std_23)
target_include_directories(v2_test_cpu_qwen_sparse_native_codec PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_test_cpu_qwen_sparse_native_codec PRIVATE GTest::gtest_main)
v2_add_core_free_test(V2_Unit_CPUQwenSparseNativeCodec
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_native_codec>
    LABELS V2 Unit CPU DeviceFree Attention QwenSparse ByteExact)
v2_add_core_free_test(V2_Integration_CPUQwenSparseNativeCodec
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_native_codec>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPUQwenSparseNativeCodecRadialOrder
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_native_codec>
        --gtest_filter=*OriginalRadialScaleOrderCannotBeReassociated
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPUQwenSparseNativeCodecNumeric
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_native_codec>
        --gtest_filter=*InvalidOriginalNumericDataIsFatalBeforeFragmentPublication
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse Numeric Regression)
add_executable(v2_test_cpu_qwen_sparse_attention
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/Test__QwenSparseAttention.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/attention/Test__CPUQwenSparseLifetime.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/attention/Test__CPUQwenSparseExactBasis.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/attention/Test__CPUQwenSparseIndexerProgram.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/Test__QwenSparseTerminalCounters.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/cpu/attention/CPUQwenSparseAttention.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/config/TensorParallelConfig.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/backends/GlobalDeviceAddress.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/qwen4exp/QwenSparseAttention.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/qwen4exp/QwenSparseReplay.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/qwen4exp/Qwen4ExpStateIO.cpp")
target_compile_features(v2_test_cpu_qwen_sparse_attention PRIVATE cxx_std_23)
target_include_directories(v2_test_cpu_qwen_sparse_attention PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_test_cpu_qwen_sparse_attention PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
v2_add_core_free_test(V2_Unit_CPUQwenSparseNativeEntries
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_attention>
        --gtest_filter=QwenSparseAttention.ExplicitRepresentationAndStaticBindingValidation:QwenSparseAttention.OrdinaryTailNoOpAndExactAcceptedPrefix:QwenSparseAttention.NativeSelectedAttentionIsGroupedSerialByteEquivalent:CPUQwenSparseLifetime.CompleteOriginalExclusionsRejectBeforePublication:CPUQwenSparseLifetime.OriginalReplayIntervalRejectsBeforeAnyWordLoad:CPUQwenSparseLifetime.FullSameHeadQueryAliasMatchesOriginalNativeProgram:CPUQwenSparseLifetime.EmptyAndNonemptyCallsCompleteBeforeOuterJoin:CPUQwenSparseLifetime.CanonicalCounterOverflowIsFatalAtPublicPublication:CPUQwenSparseLifetime.ConsumedIndexerNumericCorruptionIsFatalNotClamped:CPUQwenSparseExactBasis.OriginalNonsymmetricBasisAndQueryAliasMatchOrderedFMABytes:CPUQwenSparseIndexerProgram.FP32MeanAndRopeCastEdgesCannotContractAfterInlining:QwenSparseAttention.ProvenNoRankingPreservesRequiredStateWithoutQueryBanks:CPUQwenSparseIndexerProgram.IndexerAnalyticElementariesCannotMerge:CPUQwenSparseIndexerProgram.NativeRankingScoresRetainOriginalHeadProgram
    LABELS V2 Unit CPU DeviceFree Attention QwenSparse ByteExact)
v2_add_core_free_test(V2_Integration_CPUQwenSparseOriginalCacheProgram
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_attention>
        --gtest_filter=QwenSparseAttention.NativeSelectedAttentionIsGroupedSerialByteEquivalent:QwenSparseAttention.InactiveRowsPreserveOutputAndTrafficAcrossDirtyReplays:QwenSparseAttention.EveryNativeCodecAndQ16BlockUsesCanonicalRingAndBasis
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPUQwenSparseAdmission
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_attention>
        --gtest_filter=CPUQwenSparseLifetime.CompleteOriginalExclusionsRejectBeforePublication
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse Ownership Regression)
v2_add_core_free_test(V2_Unit_CPUQwenSparseRepresentationCast
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_attention>
        --gtest_filter=CPUQwenSparseIndexerProgram.NativeRepresentationCastsRetainIndependentIEEEWordBoundaries
    LABELS V2 Unit CPU DeviceFree Attention QwenSparse ByteExact)
v2_add_core_free_test(V2_Integration_CPUQwenSparseNoRankingNullScoreAddress
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_attention>
        --gtest_filter=QwenSparseAttention.ProvenNoRankingPreservesRequiredStateWithoutQueryBanks
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse Ownership Regression)
v2_add_core_free_test(V2_Integration_CPUQwenSparseIndexerAnalyticElementaries
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_attention>
        --gtest_filter=CPUQwenSparseIndexerProgram.IndexerAnalyticElementariesCannotMerge
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse ByteExact NumericalContract Regression)
v2_add_core_free_test(V2_Integration_CPUQwenSparseIndexerScoreProgram
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_attention>
        --gtest_filter=CPUQwenSparseIndexerProgram.NativeRankingScoresRetainOriginalHeadProgram
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse ByteExact NumericalContract Regression)
v2_add_core_free_test(V2_Integration_CPUQwenSparseIndexerNoContraction
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_attention>
        --gtest_filter=CPUQwenSparseIndexerProgram.FP32MeanAndRopeCastEdgesCannotContractAfterInlining
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPUQwenSparseRepresentationCast
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_attention>
        --gtest_filter=CPUQwenSparseIndexerProgram.NativeRepresentationCastsRetainIndependentIEEEWordBoundaries
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse ByteExact Regression)
foreach(_indexer_rep IN ITEMS FP32 FP16 BF16)
    v2_add_core_free_test(V2_Integration_CPUQwenSparseIndexerRepresentation${_indexer_rep}
        COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_attention>
            --gtest_filter=CPUQwenSparseIndexerProgram.OriginalRepresentationAndPartialRopeMatchIndependentBytes${_indexer_rep}
        LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse ByteExact Regression)
endforeach()
v2_add_core_free_test(V2_Integration_CPUQwenSparseExactOriginalBasis
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_attention>
        --gtest_filter=CPUQwenSparseExactBasis.OriginalNonsymmetricBasisAndQueryAliasMatchOrderedFMABytes
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPUQwenSparseReplayAdmission
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_attention>
        --gtest_filter=CPUQwenSparseLifetime.OriginalReplayIntervalRejectsBeforeAnyWordLoad
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse Ownership Regression)
v2_add_core_free_test(V2_Integration_CPUQwenSparseQueryAlias
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_attention>
        --gtest_filter=CPUQwenSparseLifetime.FullSameHeadQueryAliasMatchesOriginalNativeProgram
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPUQwenSparseCompletion
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_attention>
        --gtest_filter=CPUQwenSparseLifetime.EmptyAndNonemptyCallsCompleteBeforeOuterJoin
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse Ordering Regression)
v2_add_core_free_test(V2_Integration_CPUQwenSparseTerminalCounters
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_attention>
        --gtest_filter=QwenSparseTerminalCounters.*:CPUQwenSparseLifetime.CanonicalCounterOverflowIsFatalAtPublicPublication
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse StateMachine Regression)
v2_add_core_free_test(V2_Integration_CPUQwenSparseIndexerNumeric
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_attention>
        --gtest_filter=CPUQwenSparseLifetime.ConsumedIndexerNumericCorruptionIsFatalNotClamped
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse Numeric Regression)
v2_add_core_free_test(V2_Integration_CPUQwenSparseAllPositiveWorkers WHOLE_TEAM
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_attention>
        --gtest_filter=CPUQwenSparseLifetime.AllPositiveWorkersRetainOriginalSerialRowsAndGuards
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse ByteExact ThreadBudget)
# Each real representation/tail owns its timeout and every caller-admitted team
# size. One named resource prevents these whole-team sweeps competing with each
# other; it does not reduce workers or cap ordinary build/CTest parallelism.
foreach(_indexer_rep IN ITEMS FP32 FP16 BF16)
    foreach(_indexer_tail RANGE 0 3)
        v2_add_core_free_test(V2_Integration_CPUQwenSparseIndexer${_indexer_rep}Tail${_indexer_tail}AllPositiveWorkers WHOLE_TEAM
            COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_attention>
                "--gtest_filter=OriginalInputs/CPUQwenSparseIndexerTotality.NativeRowsAndWorkers/${_indexer_rep}_Tail${_indexer_tail}"
            LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse ByteExact ThreadBudget Regression
            RESOURCE_LOCK CPUQwenSparseIndexer_AllTeams)
    endforeach()
endforeach()
add_executable(v2_test_cpu_qwen_sparse_fp32
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/attention/Test__CPUQwenSparseFP32.cpp")
target_compile_features(v2_test_cpu_qwen_sparse_fp32 PRIVATE cxx_std_20)
target_include_directories(v2_test_cpu_qwen_sparse_fp32 PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_test_cpu_qwen_sparse_fp32 PRIVATE GTest::gtest_main)
v2_add_core_free_test(V2_Unit_CPUQwenSparseFP32
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_fp32>
    LABELS V2 Unit CPU DeviceFree Attention QwenSparse ByteExact)
v2_add_core_free_test(V2_Integration_CPUQwenSparseFP32OriginalProgram
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_fp32>
        --gtest_filter=CPUQwenSparseFP32.OriginalRowsRetainDeclaredNativeFMAAndButterfly:CPUQwenSparseFP32.QsaButterflyIsNotGenericAttentionFold
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPUQwenSparseFP32Admission
    COMMAND $<TARGET_FILE:v2_test_cpu_qwen_sparse_fp32>
        --gtest_filter=CPUQwenSparseFP32.ScalarAndForeignProgramsAreNotAdvertised
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention QwenSparse Regression)
add_executable(v2_test_cpu_tq_attention
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUTQAttention.cpp")
target_compile_features(v2_test_cpu_tq_attention PRIVATE cxx_std_20)
target_include_directories(v2_test_cpu_tq_attention PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_test_cpu_tq_attention PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
v2_add_core_free_test(V2_Unit_CPUTQAttention WHOLE_TEAM
    COMMAND $<TARGET_FILE:v2_test_cpu_tq_attention> --gtest_filter=-*AllPositiveWorkers*
    LABELS V2 Unit CPU DeviceFree Attention TurboQuant ByteExact)
v2_add_core_free_test(V2_Integration_CPUTQAttention WHOLE_TEAM
    COMMAND $<TARGET_FILE:v2_test_cpu_tq_attention>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention TurboQuant ByteExact)
v2_add_core_free_test(V2_Integration_CPUTQAttentionStorageAdmission
    COMMAND $<TARGET_FILE:v2_test_cpu_tq_attention> --gtest_filter=*StorageAndGeometryAdmissionPrecedeWrites
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention TurboQuant Regression)
v2_add_core_free_test(V2_Integration_CPUTQAttentionMetadataRoles
    COMMAND $<TARGET_FILE:v2_test_cpu_tq_attention>
        --gtest_filter=*OriginalMetadataRolesThresholdsAndNumeratorBytes:*InvalidSelectedMetadataAndWeightRejectBeforeWrites
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention TurboQuant ByteExact Regression)
if(LLAMINAR_CPU_TARGET_PROCESSOR MATCHES "^(arm64|aarch64|ARM64|AARCH64)$")
    v2_add_core_free_test(V2_Integration_NEONTQAttentionOrderedScore
        COMMAND $<TARGET_FILE:v2_test_cpu_tq_attention> --gtest_filter=*NativeScoreMatchesIndependentOrderedFMABytes
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree Attention TurboQuant ByteExact)
    v2_add_core_free_test(V2_Integration_NEONTQAttentionCentroidDecode
        COMMAND $<TARGET_FILE:v2_test_cpu_tq_attention> --gtest_filter=*NativeCentroidDecodeByteDomains
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree Attention TurboQuant ByteExact)
    v2_add_core_free_test(V2_Integration_NEONTQAttentionOriginalCodebooks
        COMMAND $<TARGET_FILE:v2_test_cpu_tq_attention> --gtest_filter=*NativeOriginalCodebookWordAuthorities
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree Attention TurboQuant ByteExact Ownership Regression)
endif()

add_executable(v2_test_cpu_swiglu
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUSwiGLU.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/cpu/primitives/SwiGLUPrimitives.cpp")
target_compile_features(v2_test_cpu_swiglu PRIVATE cxx_std_20)
target_include_directories(v2_test_cpu_swiglu PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_test_cpu_swiglu PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
v2_add_core_free_test(V2_Unit_CPUSwiGLU
    COMMAND $<TARGET_FILE:v2_test_cpu_swiglu>
        --gtest_filter=-*AllPositiveWorkersThroughTen*
    LABELS V2 Unit CPU DeviceFree SwiGLU Numerics ByteExact)
v2_add_core_free_test(V2_Integration_CPUSwiGLU
    COMMAND $<TARGET_FILE:v2_test_cpu_swiglu>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree SwiGLU Numerics ByteExact)
v2_add_core_free_test(V2_Integration_CPUSwiGLUBlockGeometry
    COMMAND $<TARGET_FILE:v2_test_cpu_swiglu>
        --gtest_filter=CPUSwiGLU.CompleteBlockGeometryRejectsBeforeMutation
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree SwiGLU Regression)
v2_add_core_free_test(V2_Integration_CPUFusedSwiGLUPublicationStorage
    COMMAND $<TARGET_FILE:v2_test_cpu_swiglu>
        --gtest_filter=CPUSwiGLU.FusedPublicationRejectsPolicyAndSourceAliasesBeforeMutation
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree NativeQuantised SwiGLU Regression)
v2_add_core_free_test(V2_Integration_CPUSwiGLUStorageAliases
    COMMAND $<TARGET_FILE:v2_test_cpu_swiglu>
        --gtest_filter=CPUSwiGLU.PartialStorageAliasesRejectBeforeMutation
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree SwiGLU Regression)
v2_add_core_free_test(V2_Integration_CPUSwiGLUEmptyWorkshare
    COMMAND $<TARGET_FILE:v2_test_cpu_swiglu>
        --gtest_filter=CPUSwiGLU.EmptyRetainedWorksharesCompleteBeforeReturn
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree SwiGLU ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPUSwiGLURetainedQ8
    COMMAND $<TARGET_FILE:v2_test_cpu_swiglu>
        --gtest_filter=CPUSwiGLU.Q8RetainedTeamsPublishOnceAndCompleteEmpty
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree SwiGLU ByteExact Regression)
if(LLAMINAR_CPU_TARGET_PROCESSOR MATCHES "^(arm64|aarch64|ARM64|AARCH64)$")
    v2_add_core_free_test(V2_Integration_NEONSwiGLUTypedRetained
        COMMAND $<TARGET_FILE:v2_test_cpu_swiglu>
            --gtest_filter=NativeSwiGLU.TypedRetainedTeamsAndEmptyCompletion
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree SwiGLU ByteExact Regression)
    v2_add_core_free_test(V2_Integration_NEONHalfNarrowing
        COMMAND $<TARGET_FILE:v2_test_cpu_swiglu>
            --gtest_filter=NativeSwiGLU.PortableHalfNarrowingAllExponentAndMantissaDomains
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree Half Numerics ByteExact Regression)
    v2_add_core_free_test(V2_Integration_NEONSwiGLUAccuracy
        COMMAND $<TARGET_FILE:v2_test_cpu_swiglu>
            --gtest_filter=NativeSwiGLU.OrdinaryAccuracyAgainstDoubleExponential
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree SwiGLU Numerics Regression)
    v2_add_core_free_test(V2_Integration_NEONSwiGLUDeviceContract
        COMMAND $<TARGET_FILE:v2_test_cpu_swiglu>
            --gtest_filter=NativeSwiGLU.DeviceContractWordsAndExceptionalDomains
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree SwiGLU ByteExact Regression)
    v2_add_core_free_test(V2_Integration_NEONSwiGLUSeparateNewtonRounding
        COMMAND $<TARGET_FILE:v2_test_cpu_swiglu>
            --gtest_filter=NativeSwiGLU.DeviceNewtonRoundingSurvivesInlining
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree SwiGLU ByteExact Regression)
    v2_add_core_free_test(V2_Integration_NEONSwiGLUOriginalRowTails
        COMMAND $<TARGET_FILE:v2_test_cpu_swiglu>
            --gtest_filter=NativeSwiGLU.OriginalRowsTailsAliasesAndNativeEntryPoints
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree SwiGLU ByteExact Regression)
endif()

add_executable(v2_test_cpu_rope_workshare
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPURoPEWorkshare.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/cpu/primitives/RoPEPrimitives.cpp")
target_compile_features(v2_test_cpu_rope_workshare PRIVATE cxx_std_20)
target_include_directories(v2_test_cpu_rope_workshare PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_test_cpu_rope_workshare PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
v2_add_core_free_test(V2_Unit_CPURoPEWorkshare
    COMMAND $<TARGET_FILE:v2_test_cpu_rope_workshare>
    LABELS V2 Unit CPU DeviceFree RoPE ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPURoPECachedRetained
    COMMAND $<TARGET_FILE:v2_test_cpu_rope_workshare>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RoPE ByteExact Regression)

# CPU RoPE angles live only in the admitted cpu_rope_angles workspace.
foreach(_rope_storage_suite IN ITEMS Unit Integration)
    if(_rope_storage_suite STREQUAL "Unit")
        set(_rope_storage_labels V2 Unit CPU DeviceFree RoPE SourcePolicy)
    else()
        set(_rope_storage_labels V2 Integration ProductionTestPreflight CPU DeviceFree RoPE SourcePolicy Regression)
    endif()
    v2_add_core_free_test(V2_${_rope_storage_suite}_CPURoPEAngleStorageSourcePolicy
        COMMAND "${Python3_EXECUTABLE}" -B
            "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/scripts/check_cpu_rope_angle_storage_policy.py"
            --repo-root "${V2_CPU_ARCHITECTURE_ROOT}"
        LABELS ${_rope_storage_labels})
    v2_add_core_free_test(V2_${_rope_storage_suite}_CPURoPEAngleStorageSourcePolicyNegativeControls
        COMMAND "${Python3_EXECUTABLE}" -B
            "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/scripts/test_cpu_rope_angle_storage_policy.py"
        LABELS ${_rope_storage_labels})
endforeach()
v2_add_core_free_test(V2_Integration_CPURoPESharedAngleFill
    COMMAND $<TARGET_FILE:v2_test_cpu_rope_workshare>
        --gtest_filter=CPURoPEWorkshare.SharedAngleFillCompletesBeforeRotationInRetainedTeams
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RoPE ByteExact Ownership Regression)
v2_add_core_free_test(V2_Integration_CPURoPENegativePositionDirtyWorkspace
    COMMAND $<TARGET_FILE:v2_test_cpu_rope_workshare>
        --gtest_filter=CPURoPEWorkshare.OrdinaryNegativePositionsIgnoreDirtyAngleWorkspace
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RoPE FP32 BF16 FP16 ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPURoPEQ8OneRowTeamHeadOwnership
    COMMAND $<TARGET_FILE:v2_test_cpu_rope_workshare>
        --gtest_filter=CPURoPEWorkshare.Q8OneRowRetainedTeamRotatesEachHeadOnce
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RoPE Q8_1 ByteExact Ownership Regression)
v2_add_core_free_test(V2_Integration_CPURoPEQ8OneRowNegativePosition
    COMMAND $<TARGET_FILE:v2_test_cpu_rope_workshare>
        --gtest_filter=CPURoPEWorkshare.Q8OneRowNegativePositionIsIndependentOfPriorCall
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RoPE Q8_1 ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPURoPEAngleBindingRejection
    COMMAND $<TARGET_FILE:v2_test_cpu_rope_workshare>
        --gtest_filter=CPURoPEWorkshare.AngleRowsRejectMissingOrUndersizedPlanes
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RoPE Ownership Geometry Regression)
v2_add_core_free_test(V2_Integration_CPURoPEInverseFrequencyLookup
    COMMAND $<TARGET_FILE:v2_test_cpu_rope_workshare>
        --gtest_filter=CPURoPEWorkshare.InverseFrequencyLookupIsStableAcrossThreadsAndEvictions:CPURoPEWorkshare.InverseFrequencyLookupAlternatesConfigurationsInRetainedTeams
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RoPE Ownership Regression)
v2_add_core_free_test(V2_Integration_CPURoPEInlineBoundedCoefficients
    COMMAND $<TARGET_FILE:v2_test_cpu_rope_workshare>
        --gtest_filter=CPURoPEWorkshare.InlineWideHeadsAndPartialPrefixesRetainProductionBytes
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RoPE ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPURoPEInlineEmptyCompletion
    COMMAND $<TARGET_FILE:v2_test_cpu_rope_workshare>
        --gtest_filter=CPURoPEWorkshare.InlineEmptyCompletionPrecedesObserverRelease
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RoPE Regression)
v2_add_core_free_test(V2_Integration_CPURoPEInlineGeometry
    COMMAND $<TARGET_FILE:v2_test_cpu_rope_workshare>
        --gtest_filter=CPURoPEWorkshare.InlineGeometryRejectsBeforeAccess
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RoPE Geometry Regression)

v2_add_core_free_test(V2_Integration_CPURoPEHybridOriginalRows
    COMMAND $<TARGET_FILE:v2_test_cpu_rope_workshare>
        --gtest_filter=CPURoPEWorkshare.HybridWideHeadsRetainSerialOriginalRows
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RoPE ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPURoPEHybridStorageAdmission
    COMMAND $<TARGET_FILE:v2_test_cpu_rope_workshare>
        --gtest_filter=CPURoPEWorkshare.HybridStorageAdmissionAndEmptyCompletion
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree RoPE Regression)

add_executable(v2_test_cpu_flash_attention_launch_policy
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/attention/Test__CPUFlashAttentionLaunchPolicy.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/attention/Test__CPUQ16AttentionMath.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/attention/Test__CPUAttentionInvocationAdmission.cpp")
target_compile_features(v2_test_cpu_flash_attention_launch_policy PRIVATE cxx_std_20)
target_include_directories(v2_test_cpu_flash_attention_launch_policy PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src" "${V2_CPU_ARCHITECTURE_ROOT}/src/v2")
target_link_libraries(v2_test_cpu_flash_attention_launch_policy PRIVATE GTest::gtest_main)
v2_add_core_free_test(V2_Unit_CPUFlashAttentionLaunchPolicy
    COMMAND $<TARGET_FILE:v2_test_cpu_flash_attention_launch_policy>
    LABELS V2 Unit CPU DeviceFree Attention Dispatch Geometry)
v2_add_core_free_test(V2_Integration_CPUFlashAttentionLaunchPolicy
    COMMAND $<TARGET_FILE:v2_test_cpu_flash_attention_launch_policy>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention Dispatch Geometry)
# Every CPU attention entry admits banks, head mapping and output disjointness
# once per invocation; the row programs (NEON included) carry no per-call checks.
v2_add_core_free_test(V2_Integration_CPUAttentionInvocationAdmission
    COMMAND $<TARGET_FILE:v2_test_cpu_flash_attention_launch_policy>
        --gtest_filter=CPUAttentionInvocationAdmission.*
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention Geometry Regression)
v2_add_core_free_test(V2_Integration_CPUFA2NativeISAAdmission
    COMMAND $<TARGET_FILE:v2_test_cpu_flash_attention_launch_policy>
        --gtest_filter=*NativeISAIdentityRejectsForeignEnvelopes
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention ISA Regression)
v2_add_core_free_test(V2_Integration_CPUFA2ARMCanonicalSpanPolicy
    COMMAND $<TARGET_FILE:v2_test_cpu_flash_attention_launch_policy>
        --gtest_filter=*ARMCanonicalSpanDoesNotBorrowX86CacheEvidence
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPUFA2CheckedGeometry
    COMMAND $<TARGET_FILE:v2_test_cpu_flash_attention_launch_policy>
        --gtest_filter=*WorkingSetOverflowRejectsWithoutNarrowing:*ParallelGeometryBoundsPrecedeMultiplication
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention Geometry Regression)
v2_add_core_free_test(V2_Integration_CPUQ16AttentionIntegerRange
    COMMAND $<TARGET_FILE:v2_test_cpu_flash_attention_launch_policy>
        --gtest_filter=CPUQ16AttentionMath.*
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention Quantization Regression)

add_executable(v2_unit_cpu_native_block_decode
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUNativeBlockDecode.cpp")
target_compile_features(v2_unit_cpu_native_block_decode PRIVATE cxx_std_20)
target_include_directories(v2_unit_cpu_native_block_decode PRIVATE "${V2_CPU_ARCHITECTURE_ROOT}/src/v2")
target_link_libraries(v2_unit_cpu_native_block_decode PRIVATE GTest::gtest_main)
v2_add_core_free_test(V2_Unit_CPUNativeBlockDecode
    COMMAND $<TARGET_FILE:v2_unit_cpu_native_block_decode>
    LABELS V2 Unit CPU DeviceFree Quantization)
v2_add_core_free_test(V2_Integration_CPUNativeBlockDecode
    COMMAND $<TARGET_FILE:v2_unit_cpu_native_block_decode>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Quantization Regression)

add_executable(v2_unit_cpu_native_weight_view
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUNativeQuantisedWeightView.cpp")
target_compile_features(v2_unit_cpu_native_weight_view PRIVATE cxx_std_20)
target_include_directories(v2_unit_cpu_native_weight_view PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_unit_cpu_native_weight_view PRIVATE GTest::gtest_main)
v2_add_core_free_test(V2_Unit_CPUNativeQuantisedWeightView
    COMMAND $<TARGET_FILE:v2_unit_cpu_native_weight_view>
    LABELS V2 Unit CPU DeviceFree NativeQuantised Quantization)
v2_add_core_free_test(V2_Integration_CPUNativeQuantisedWeightView
    COMMAND $<TARGET_FILE:v2_unit_cpu_native_weight_view>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree NativeQuantised Quantization Regression)

# Prepared metadata is admitted once per publication generation; execution
# never tests reserved bits inside a worker team. Coverage equals the former
# in-loop rules on every lane, and every mutation (slot rewrite included)
# withdraws admission until a producer reseals.
add_executable(v2_unit_cpu_native_metadata_seal
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUNativeQuantisedMetadataSeal.cpp")
target_compile_features(v2_unit_cpu_native_metadata_seal PRIVATE cxx_std_20)
target_include_directories(v2_unit_cpu_native_metadata_seal PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_unit_cpu_native_metadata_seal PRIVATE GTest::gtest_main)
v2_add_core_free_test(V2_Unit_CPUNativeQuantisedMetadataSeal
    COMMAND $<TARGET_FILE:v2_unit_cpu_native_metadata_seal>
    LABELS V2 Unit CPU DeviceFree NativeQuantised Metadata)
v2_add_core_free_test(V2_Integration_CPUNativeQuantisedMetadataSealCoverage
    COMMAND $<TARGET_FILE:v2_unit_cpu_native_metadata_seal>
        --gtest_filter=CPUNativeQuantisedMetadataSeal.CoversExactlyTheFormerInLoopRulesOnEveryLane:CPUNativeQuantisedMetadataSeal.ViewRequiresTheSealOfItsExactBytesAndInterpretation
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree NativeQuantised Metadata Regression)
# Partial final chunks: producers normalize padding words in the owned
# generation; raw poisoned padding is rejected, normalized arrivals admitted.
# Admitted in-team metadata access keeps a fatal (never throwing) coordinate bound.
v2_add_core_free_test(V2_Integration_CPUNativeQuantisedAdmittedAccessBound
    COMMAND $<TARGET_FILE:v2_unit_cpu_native_metadata_seal>
        --gtest_filter=CPUNativeQuantisedMetadataSeal.AdmittedMetadataAccessIsFatalOutsideThePublication
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree NativeQuantised Metadata Regression)
v2_add_core_free_test(V2_Integration_CPUNativeQuantisedPaddingNormalization
    COMMAND $<TARGET_FILE:v2_unit_cpu_native_metadata_seal>
        --gtest_filter=CPUNativeQuantisedMetadataSeal.PaddingNormalization*
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree NativeQuantised Metadata Regression)
v2_add_core_free_test(V2_Integration_CPUNativeQuantisedMetadataSealLifecycle
    COMMAND $<TARGET_FILE:v2_unit_cpu_native_metadata_seal>
        --gtest_filter=CPUNativeQuantisedMetadataSeal.EveryMutationWithdrawsAdmission:CPUNativeQuantisedMetadataSeal.RejectedResealLeavesThePublicationUnadmitted:CPUNativeQuantisedMetadataSeal.UnsealedAndStalePublicationsAreFatal:CPUNativeQuantisedMetadataSeal.SlotRewriteAtTheSameAddressNeedsAReseal
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree NativeQuantised Metadata Lifecycle Regression)

if(LLAMINAR_CPU_TARGET_PROCESSOR MATCHES "^(arm64|aarch64|ARM64|AARCH64)$")
    add_executable(v2_test_neon_integer_rope
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__NEONIntegerRoPE.cpp"
        "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/cpu/primitives/RoPEPrimitives.cpp")
    target_compile_features(v2_test_neon_integer_rope PRIVATE cxx_std_20)
    target_include_directories(v2_test_neon_integer_rope PRIVATE
        "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
    target_link_libraries(v2_test_neon_integer_rope PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
    v2_add_core_free_test(V2_Unit_NEONIntegerRoPE
        COMMAND $<TARGET_FILE:v2_test_neon_integer_rope>
        LABELS V2 Unit CPU NEON DeviceFree RoPE Quantization ByteExact)
    v2_add_core_free_test(V2_Integration_NEONIntegerRoPE
        COMMAND $<TARGET_FILE:v2_test_neon_integer_rope>
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree RoPE Quantization ByteExact)
    v2_add_core_free_test(V2_Integration_NEONRoPEIntegerMetadataDeferral
        COMMAND $<TARGET_FILE:v2_test_neon_integer_rope>
            --gtest_filter=*FixedDynamicQ16WidenedProductsAndDeferredScales
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree RoPE Quantization ByteExact Regression)
    v2_add_core_free_test(V2_Integration_NEONRoPEIntegerBiasAndWidening
        COMMAND $<TARGET_FILE:v2_test_neon_integer_rope>
            --gtest_filter=*LegacySignedBiasAndScaleIdentity
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree RoPE Quantization ByteExact Regression)
    v2_add_core_free_test(V2_Integration_NEONRoPEIntegerStorageAdmission
        COMMAND $<TARGET_FILE:v2_test_neon_integer_rope>
            --gtest_filter=*GeometryAliasesAndInvalidScaleRejection
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree RoPE Quantization Regression)

    v2_add_core_free_test(V2_Integration_NEONRoPEQ8SignedRatioExtrema
        COMMAND $<TARGET_FILE:v2_test_neon_integer_rope>
            --gtest_filter=*Q8CommonScalePairRetainsWidenedSignedByteExtrema
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree RoPE Quantization ByteExact Regression)
    v2_add_core_free_test(V2_Integration_NEONRoPEHybridPairRounding
        COMMAND $<TARGET_FILE:v2_test_neon_integer_rope>
            --gtest_filter=*HybridPairsMatchIndependentMemoryRoundedBytes
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree RoPE ByteExact Regression)

    add_executable(v2_test_neon_rope
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__NEONRoPE.cpp"
        "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/cpu/primitives/RoPEPrimitives.cpp")
    target_compile_features(v2_test_neon_rope PRIVATE cxx_std_20)
    target_include_directories(v2_test_neon_rope PRIVATE
        "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
    target_link_libraries(v2_test_neon_rope PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
    v2_add_core_free_test(V2_Unit_NEONRoPE
        COMMAND $<TARGET_FILE:v2_test_neon_rope>
        LABELS V2 Unit CPU NEON DeviceFree RoPE Numerics ByteExact)
    v2_add_core_free_test(V2_Integration_NEONRoPE
        COMMAND $<TARGET_FILE:v2_test_neon_rope>
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree RoPE Numerics ByteExact)
    v2_add_core_free_test(V2_Integration_NEONRoPEPairRounding
        COMMAND $<TARGET_FILE:v2_test_neon_rope>
            --gtest_filter=*FloatingStoragePairsMatchMemoryRoundedBytes
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree RoPE Numerics ByteExact Regression)

    v2_add_core_free_test(V2_Integration_NEONRoPEQuantizedBlockCycles
        COMMAND $<TARGET_FILE:v2_test_neon_rope>
            --gtest_filter=*QuantizedSingleOddEvenBlockCyclesRetainIndependentBytes
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree RoPE Quantization ByteExact Regression)
    v2_add_core_free_test(V2_Integration_NEONRoPEHalfScaleOverflow
        COMMAND $<TARGET_FILE:v2_test_neon_rope>
            --gtest_filter=*Q8OutputScaleOverflowIsFatalBeforeBlockPublication
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree RoPE Quantization Regression)

    add_executable(v2_test_native_neon_execution
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUNativeNEONExecution.cpp"
        "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/cpu/primitives/SwiGLUPrimitives.cpp")
    target_compile_features(v2_test_native_neon_execution PRIVATE cxx_std_20)
    target_include_directories(v2_test_native_neon_execution PRIVATE
        "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
    target_link_libraries(v2_test_native_neon_execution PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
    v2_add_core_free_test(V2_Unit_CPUNativeNEONExecution WHOLE_TEAM
        COMMAND $<TARGET_FILE:v2_test_native_neon_execution>
        LABELS V2 Unit CPU NEON DeviceFree NativeQuantised Numerics ByteExact)
    v2_add_core_free_test(V2_Integration_CPUNativeNEONExecution WHOLE_TEAM
        COMMAND $<TARGET_FILE:v2_test_native_neon_execution>
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree NativeQuantised Numerics ByteExact AllFormats)
    v2_add_core_free_test(V2_Integration_CPUProjectionPublicationStorageAliases
        COMMAND $<TARGET_FILE:v2_test_native_neon_execution>
            --gtest_filter=*RejectsInputPublicationAndBetaWorkAliasesBeforeWrites
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree NativeQuantised ByteExact Regression)
    v2_add_core_free_test(V2_Integration_CPUQ8ResidualRetainedPrimaryReuse
        COMMAND $<TARGET_FILE:v2_test_native_neon_execution>
            --gtest_filter=*RetainedTwoTermCompletesBeforePrimaryReuse
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree NativeQuantised ByteExact Regression)
    v2_add_core_free_test(V2_Integration_CPUPreparedOwnerSingleScaleMin
        COMMAND $<TARGET_FILE:v2_test_native_neon_execution>
            --gtest_filter=*ResidentSingleScaleOwnerMinIsExactZero
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree NativeQuantised ByteExact Regression)

    v2_add_core_free_test(V2_Integration_CPUNativeNEONSwiGLUDownPublication
        COMMAND $<TARGET_FILE:v2_test_native_neon_execution>
            --gtest_filter=*AllSourcesFusedSwiGLUQ8DownRetainSerialRows
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree NativeQuantised SwiGLU ByteExact AllFormats)

    add_executable(v2_test_native_neon_projection_batch
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUNativeNEONProjectionBatch.cpp")
    target_compile_features(v2_test_native_neon_projection_batch PRIVATE cxx_std_20)
    target_include_directories(v2_test_native_neon_projection_batch PRIVATE
        "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
    target_link_libraries(v2_test_native_neon_projection_batch PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
    v2_add_core_free_test(V2_Unit_CPUNativeNEONProjectionBatch WHOLE_TEAM
        COMMAND $<TARGET_FILE:v2_test_native_neon_projection_batch>
            --gtest_filter=-*AllPositiveWorkersThroughTen*
        LABELS V2 Unit CPU NEON DeviceFree NativeQuantised Numerics ByteExact)
    v2_add_core_free_test(V2_Integration_CPUNativeNEONProjectionBatch WHOLE_TEAM
        COMMAND $<TARGET_FILE:v2_test_native_neon_projection_batch>
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree NativeQuantised Numerics ByteExact AllFormats)
    v2_add_core_free_test(V2_Integration_CPUNativeNEONEmptyWorkshare
        COMMAND $<TARGET_FILE:v2_test_native_neon_projection_batch>
            --gtest_filter=*EmptyRetainedTeamCompletesBeforeNextPublication
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree NativeQuantised ByteExact Regression)
    v2_add_core_free_test(V2_Integration_CPUNativeNEONOriginalRowPublication
        COMMAND $<TARGET_FILE:v2_test_native_neon_projection_batch>
            --gtest_filter=*OriginalRequestSlicesRepeatsAndDirtyInactiveRows:*RejectsWholeBankAliasesAndInvalidGeometry
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree NativeQuantised ByteExact Regression)

    add_executable(v2_unit_neon_typed_tensor_ops
        "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__NEONTypedTensorOps.cpp")
    target_compile_features(v2_unit_neon_typed_tensor_ops PRIVATE cxx_std_20)
    target_include_directories(v2_unit_neon_typed_tensor_ops PRIVATE
        "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
    target_link_libraries(v2_unit_neon_typed_tensor_ops PRIVATE GTest::gtest_main)
    v2_add_core_free_test(V2_Unit_NEONTypedTensorOps
        COMMAND $<TARGET_FILE:v2_unit_neon_typed_tensor_ops>
        LABELS V2 Unit CPU NEON DeviceFree Quantization Numerics ByteExact)
    v2_add_core_free_test(V2_Integration_NEONTypedTensorOps
        COMMAND $<TARGET_FILE:v2_unit_neon_typed_tensor_ops>
        LABELS V2 Integration ProductionTestPreflight CPU NEON DeviceFree Quantization Numerics ByteExact Regression)
endif()

add_executable(v2_test_recursive_stage_snapshot
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__RecursiveStageSnapshot.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/backends/GlobalDeviceAddress.cpp")
target_compile_features(v2_test_recursive_stage_snapshot PRIVATE cxx_std_20)
target_include_directories(v2_test_recursive_stage_snapshot PRIVATE "${V2_CPU_ARCHITECTURE_ROOT}/src/v2")
target_link_libraries(v2_test_recursive_stage_snapshot PRIVATE GTest::gtest_main)
v2_add_core_free_test(V2_Unit_RecursiveStageSnapshot
    COMMAND $<TARGET_FILE:v2_test_recursive_stage_snapshot>
    LABELS V2 Unit CPU DeviceFree Metadata)
v2_add_core_free_test(V2_Integration_RecursiveStageSnapshotCompletion
    COMMAND $<TARGET_FILE:v2_test_recursive_stage_snapshot>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Metadata Regression)

add_executable(v2_test_cpu_projection_native_workspace
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUProjectionNativeWorkspace.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/backends/CPUExecutionGeometry.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/cpu/gemm/CPUProjectionWorkspaceContract.cpp")
target_compile_features(v2_test_cpu_projection_native_workspace PRIVATE cxx_std_20)
target_include_directories(v2_test_cpu_projection_native_workspace PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_test_cpu_projection_native_workspace PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
v2_add_core_free_test(V2_Unit_CPUProjectionNativeWorkspace
    COMMAND $<TARGET_FILE:v2_test_cpu_projection_native_workspace>
    LABELS V2 Unit CPU DeviceFree Memory NativeQuantised)
v2_add_core_free_test(V2_Integration_CPUProjectionNativeWorkspaceABI
    COMMAND $<TARGET_FILE:v2_test_cpu_projection_native_workspace>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Memory NativeQuantised Regression)

add_executable(v2_test_cpu_typed_tensor_arithmetic
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUTypedTensorArithmetic.cpp")
target_compile_features(v2_test_cpu_typed_tensor_arithmetic PRIVATE cxx_std_20)
target_include_directories(v2_test_cpu_typed_tensor_arithmetic PRIVATE
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2" "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2")
target_link_libraries(v2_test_cpu_typed_tensor_arithmetic PRIVATE GTest::gtest_main)
v2_add_core_free_test(V2_Unit_CPUTypedTensorArithmetic
    COMMAND $<TARGET_FILE:v2_test_cpu_typed_tensor_arithmetic>
    LABELS V2 Unit CPU DeviceFree Quantization Numerics ByteExact)
v2_add_core_free_test(V2_Integration_CPUTypedTensorDecodedAddRounding
    COMMAND $<TARGET_FILE:v2_test_cpu_typed_tensor_arithmetic>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Quantization ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPUTypedTensorProductRoundingAliases
    COMMAND $<TARGET_FILE:v2_test_cpu_typed_tensor_arithmetic>
        --gtest_filter=CPUTypedTensorArithmetic.DecodedPairRoundingSurvivesOperandOrderAndAliases
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Quantization ByteExact Regression)

add_executable(v2_unit_typed_tensor_block_geometry
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/Test__TypedTensorBlockGeometry.cpp")
target_compile_features(v2_unit_typed_tensor_block_geometry PRIVATE cxx_std_20)
target_include_directories(v2_unit_typed_tensor_block_geometry PRIVATE "${V2_CPU_ARCHITECTURE_ROOT}/src/v2")
target_link_libraries(v2_unit_typed_tensor_block_geometry PRIVATE GTest::gtest_main)
v2_add_core_free_test(V2_Unit_TypedTensorBlockGeometry
    COMMAND $<TARGET_FILE:v2_unit_typed_tensor_block_geometry>
    LABELS V2 Unit CPU DeviceFree Quantization)
v2_add_core_free_test(V2_Integration_TypedTensorBlockGeometry
    COMMAND $<TARGET_FILE:v2_unit_typed_tensor_block_geometry>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Quantization Regression)

add_executable(v2_unit_portable_vector_primitives
    "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/kernels/cpu/primitives/Test__PortableVectorPrimitives.cpp"
    "${V2_CPU_ARCHITECTURE_ROOT}/src/v2/kernels/cpu/primitives/VectorPrimitives.cpp")
target_compile_features(v2_unit_portable_vector_primitives PRIVATE cxx_std_20)
target_include_directories(v2_unit_portable_vector_primitives PRIVATE "${V2_CPU_ARCHITECTURE_ROOT}/src/v2")
v2_add_core_free_test(V2_Unit_PortableVectorPrimitives
    COMMAND $<TARGET_FILE:v2_unit_portable_vector_primitives>
    LABELS V2 Unit CPU DeviceFree Primitives)
v2_add_core_free_test(V2_Integration_PortableVectorGroupedByteEquivalence
    COMMAND $<TARGET_FILE:v2_unit_portable_vector_primitives> --grouped
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Primitives Regression)

foreach(scope IN ITEMS Unit Integration)
    set(labels V2 ${scope} CPU DeviceFree Build Regression)
    if(scope STREQUAL "Integration")
        list(APPEND labels ProductionTestPreflight)
    endif()
    v2_add_core_free_test(V2_${scope}_CPUInstructionSetContract
        COMMAND "${CMAKE_COMMAND}" -P
            "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/TestCPUInstructionSet.cmake"
        LABELS ${labels})
    v2_add_core_free_test(V2_${scope}_OpenMPNativeIdentityContract
        COMMAND "${CMAKE_COMMAND}" "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/cpu_native_identity_${scope}"
            -P "${V2_CPU_ARCHITECTURE_ROOT}/tests/v2/unit/cmake/TestOpenMPNativeIdentity.cmake"
        LABELS ${labels})
endforeach()
