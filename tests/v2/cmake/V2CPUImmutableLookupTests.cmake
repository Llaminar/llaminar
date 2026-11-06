# @file V2CPUImmutableLookupTests.cmake
# @brief Compile the real immutable CPU row decoder in both canonical and focused gates.
#
# Native header/ISA closure is a compilation obligation, not a source allowlist.
# Fast device-free payload, original-row and retained-team regressions exercise
# exactly the same production TU as llaminar2_core, without model/service setup.
if(V2_PERF_TESTS_ONLY)
    return()
endif()
get_filename_component(V2_CPU_LOOKUP_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
include("${CMAKE_CURRENT_LIST_DIR}/V2CoreFreeTestRegistration.cmake")
add_executable(v2_test_cpu_immutable_lookup_rows
    "${V2_CPU_LOOKUP_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUImmutableLookupRows.cpp"
    "${V2_CPU_LOOKUP_ROOT}/src/v2/kernels/cpu/qwen4exp/CPUImmutableLookupKernels.cpp")
target_compile_features(v2_test_cpu_immutable_lookup_rows PRIVATE cxx_std_23)
target_include_directories(v2_test_cpu_immutable_lookup_rows PRIVATE "${V2_CPU_LOOKUP_ROOT}/src/v2")
target_link_libraries(v2_test_cpu_immutable_lookup_rows PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
v2_add_core_free_test(V2_Unit_CPUImmutableLookupRows
    COMMAND $<TARGET_FILE:v2_test_cpu_immutable_lookup_rows>
    LABELS V2 Unit CPU DeviceFree ImmutableLookup)
v2_add_core_free_test(V2_Integration_CPUImmutableLookupNativePayloads
    COMMAND $<TARGET_FILE:v2_test_cpu_immutable_lookup_rows>
        --gtest_filter=CPUImmutableLookupRows.FloatingAllOriginalPayloads:CPUImmutableLookupRows.FloatingIndependentNonfiniteFringes:CPUImmutableLookupRows.IQ4OriginalCodebookScalesAndPartialRows
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree ImmutableLookup Native ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPUImmutableLookupOriginalRows
    COMMAND $<TARGET_FILE:v2_test_cpu_immutable_lookup_rows>
        --gtest_filter=CPUImmutableLookupRows.EverySourceAndPositiveTeamRetainSerialOriginalRows:CPUImmutableLookupRows.RetainedNativePublicationCompletesBeforeOutputReuse
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree ImmutableLookup AllFormats ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPUImmutableLookupAdmission
    COMMAND $<TARGET_FILE:v2_test_cpu_immutable_lookup_rows>
        --gtest_filter=CPUImmutableLookupRows.GeometryAndOriginalOutputIntersectionsRejectBeforeStores:CPUImmutableLookupRows.GuardAuthenticationRejectsAdjacentOverwrites
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree ImmutableLookup Ownership Geometry Regression)
v2_add_core_free_test(V2_Integration_CPUImmutableLookupEmptyCompletion
    COMMAND $<TARGET_FILE:v2_test_cpu_immutable_lookup_rows>
        --gtest_filter=CPUImmutableLookupRows.EmptyRetainedCallsCompleteBeforeImmediateReuse
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree ImmutableLookup Workshare Ownership Regression)
