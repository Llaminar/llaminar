# @file V2CPUAttentionKeyQ8Tests.cmake
# @brief Shared native AQ8 codec targets and device-free Unit/preflight registrations.
#
# The complete V2 build and focused CPU-port project compile the identical
# production header and conformance source. The canonical Perf__ target is
# available in both; only the complete V2 owner registers its economy CTest.
# Kernel evidence does not certify cache/request lifetime or whole-engine serving.
include_guard(GLOBAL)
get_filename_component(V2_CPU_AQ8_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(v2_perf_cpu_attention_key_q8
    "${V2_CPU_AQ8_ROOT}/tests/v2/performance/kernels/cpu/attention/Perf__CPUAttentionKeyQ8.cpp")
target_include_directories(v2_perf_cpu_attention_key_q8 PRIVATE "${V2_CPU_AQ8_ROOT}/src/v2")
target_compile_features(v2_perf_cpu_attention_key_q8 PRIVATE cxx_std_20)
target_link_libraries(v2_perf_cpu_attention_key_q8 PRIVATE OpenMP::OpenMP_CXX)

if(V2_PERF_TESTS_ONLY)
    return()
endif()
include("${CMAKE_CURRENT_LIST_DIR}/V2CoreFreeTestRegistration.cmake")
add_executable(v2_integration_cpu_attention_key_q8
    "${V2_CPU_AQ8_ROOT}/tests/v2/integration/kernels/cpu/Test__CPUAttentionKeyQ8.cpp"
    "${V2_CPU_AQ8_ROOT}/tests/v2/integration/kernels/cpu/Test__CPUAttentionKeyQ8Dot.cpp"
    "${V2_CPU_AQ8_ROOT}/tests/v2/integration/kernels/cpu/Test__CPUQ8AttentionValue.cpp")
target_compile_features(v2_integration_cpu_attention_key_q8 PRIVATE cxx_std_20)
target_include_directories(v2_integration_cpu_attention_key_q8 PRIVATE "${V2_CPU_AQ8_ROOT}/src/v2" "${V2_CPU_AQ8_ROOT}/tests/v2")
target_link_libraries(v2_integration_cpu_attention_key_q8 PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
v2_add_core_free_test(V2_Unit_CPUAttentionKeyQ8
    COMMAND $<TARGET_FILE:v2_integration_cpu_attention_key_q8>
        --gtest_filter=-*AllWidthsISAsAndWorkerCountsMatchScalarBytes
    LABELS V2 Unit CPU DeviceFree Attention KVCache Q8 AllHeadDims ByteExact)
v2_add_core_free_test(V2_Integration_CPUAttentionKeyQ8
    COMMAND $<TARGET_FILE:v2_integration_cpu_attention_key_q8>
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention KVCache Q8 AllHeadDims ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPUQ8AttentionValueProgram
    COMMAND $<TARGET_FILE:v2_integration_cpu_attention_key_q8>
        --gtest_filter=CPUQ8AttentionValue.NativeOriginalCodesMatchIndependentCombinedWeightFMA
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention KVCache Q8 ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPUQ8AttentionValueAdmission
    COMMAND $<TARGET_FILE:v2_integration_cpu_attention_key_q8>
        --gtest_filter=CPUQ8AttentionValue.CompleteNumeratorAndSourceAliasesRejectBeforeStores
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention KVCache Q8 Ownership Geometry Regression)
v2_add_core_free_test(V2_Integration_CPUAttentionKeyQ8OriginalDots
    COMMAND $<TARGET_FILE:v2_integration_cpu_attention_key_q8>
        --gtest_filter=CPUAttentionKeyQ8Dot.OriginalQueriesMatchSerialAndPreviousPipeline
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention KVCache Q8 ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPUAttentionKeyQ8DotRounding
    COMMAND $<TARGET_FILE:v2_integration_cpu_attention_key_q8>
        --gtest_filter=CPUAttentionKeyQ8Dot.SeparateDecodeRoundingAndPublishedScales
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention KVCache Q8 ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPUAttentionKeyQ8DotAuthentication
    COMMAND $<TARGET_FILE:v2_integration_cpu_attention_key_q8>
        --gtest_filter=CPUAttentionKeyQ8Dot.PoisonedAuthenticationRejectsNoOpAndPartialWriters
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention KVCache Q8 Regression)
v2_add_core_free_test(V2_Integration_CPUAttentionKeyQ8DotAdmission
    COMMAND $<TARGET_FILE:v2_integration_cpu_attention_key_q8>
        --gtest_filter=CPUAttentionKeyQ8Dot.OriginalHeadGeometryAndUnavailableProgramsReject
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention KVCache Q8 ISA Geometry Regression)
v2_add_core_free_test(V2_Integration_CPUAttentionKeyQ8StorageAliases
    COMMAND $<TARGET_FILE:v2_integration_cpu_attention_key_q8>
        --gtest_filter=*SourceAnchorAndCompressedStorageAliasesRejectBeforeMutation
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention KVCache Q8 Regression)
v2_add_core_free_test(V2_Integration_CPUAttentionKeyQ8AnchoredFailures
    COMMAND $<TARGET_FILE:v2_integration_cpu_attention_key_q8>
        --gtest_filter=*AnchoredFailuresRejectBeforePublicationForEveryWidth
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention KVCache Q8 Regression)
v2_add_core_free_test(V2_Integration_CPUAttentionKeyQ8ZeroResidual
    COMMAND $<TARGET_FILE:v2_integration_cpu_attention_key_q8>
        --gtest_filter=*ZeroResidualRetainsAnchorsAndAuthorsEveryByte
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention KVCache Q8 AllHeadDims ByteExact Regression)
v2_add_core_free_test(V2_Integration_CPUAttentionKeyQ8RationalBoundaries
    COMMAND $<TARGET_FILE:v2_integration_cpu_attention_key_q8>
        --gtest_filter=*EveryRationalBoundaryMatchesScalarBytes:*FiniteExponentRangeMatchesScalarBytes:*EveryLegalCodeAndPhysicalScaleMatchesScalarBytes
    LABELS V2 Integration ProductionTestPreflight CPU DeviceFree Attention KVCache Q8 AllHeadDims ByteExact Regression)
