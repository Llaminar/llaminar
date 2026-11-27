# @file V2CPUExplicitRoundingTests.cmake
# @brief Canonical native CPU explicit-rounding/Q8/half numerical proof registration.
#
# Header-only production arithmetic does not require a model, MPI or the core
# library. Share these exact targets with the focused ARM build. ISA ceilings
# use real compiler flags and native feature detection, never spoofed CPUID or
# an ARM target mislabeled as scalar/AVX2.
#
# The Q8 finite-scale copy whose ISA equals the canonical build envelope
# compiles in this directory. Every reduced x86 copy (SCALAR, and AVX2 inside an
# AVX512 build) compiles in tests/v2/cpu_secondary_envelopes under one coherent
# declared envelope of its own; CTest registration for all copies stays here.
include("${CMAKE_CURRENT_LIST_DIR}/V2CoreFreeTestRegistration.cmake")
get_filename_component(V2_NUMERIC_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
# The NEON projection tests reach CPUNativeOrderedPartitionPlan.h; they link its reporter authority.
include("${V2_NUMERIC_ROOT}/cmake/NativePartitionPlan.cmake")

# @brief Build one copy of the existing production numerical contract tests.
# @param target Exact executable identity used by Unit/preflight registration.
function(v2_add_cpu_rounding_target target)
    add_executable(${target}
        "${V2_NUMERIC_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUExplicitRoundingContract.cpp")
    target_include_directories(${target} PRIVATE
        "${V2_NUMERIC_ROOT}/src/v2" "${V2_NUMERIC_ROOT}/tests/v2")
    target_compile_features(${target} PRIVATE cxx_std_20)
    # A conversion branch that falls off its end is undefined behaviour the
    # optimizer exploits (the F16C half-scale path published 65504 as +inf).
    target_compile_options(${target} PRIVATE -Werror=return-type)
    target_link_libraries(${target} PRIVATE GTest::gtest_main)
endfunction()

v2_add_cpu_rounding_target(v2_test_cpu_explicit_rounding_contract)
v2_add_core_free_test(V2_Unit_CPUExplicitRoundingContract
    COMMAND $<TARGET_FILE:v2_test_cpu_explicit_rounding_contract>
    LABELS V2 Unit CPU Numerics ByteExact)

set(V2_Q8_FINITE_SCALE_CPU_PREFLIGHT_TESTS)
if(LLAMINAR_CPU_TARGET_PROCESSOR MATCHES "^(arm64|aarch64|ARM64|AARCH64)$")
    set(V2_Q8_FINITE_SCALE_CPU_ISAS NEON)
    target_compile_definitions(v2_test_cpu_explicit_rounding_contract PRIVATE
        LLAMINAR_Q8_SCALE_TEST_NEON=1)
else()
    set(V2_Q8_FINITE_SCALE_CPU_ISAS SCALAR AVX2)
    if(LLAMINAR_CPU_ISA_UPPER STREQUAL "AVX512")
        list(APPEND V2_Q8_FINITE_SCALE_CPU_ISAS AVX512)
    endif()
endif()
# Canonical-envelope copies compile here; reduced copies use their own scope.
set(V2_Q8_FINITE_SCALE_SECONDARY_ISAS)
foreach(V2_Q8_SCALE_ISA IN LISTS V2_Q8_FINITE_SCALE_CPU_ISAS)
    if(V2_Q8_SCALE_ISA STREQUAL "NEON" OR V2_Q8_SCALE_ISA STREQUAL LLAMINAR_CPU_ISA_UPPER)
        string(TOLOWER "${V2_Q8_SCALE_ISA}" V2_Q8_SCALE_TARGET_ISA)
        set(V2_Q8_SCALE_TARGET v2_test_q8_finite_scale_contract_${V2_Q8_SCALE_TARGET_ISA})
        v2_add_cpu_rounding_target(${V2_Q8_SCALE_TARGET})
        target_compile_definitions(${V2_Q8_SCALE_TARGET} PRIVATE
            LLAMINAR_Q8_SCALE_TEST_${V2_Q8_SCALE_ISA}=1)
    else()
        list(APPEND V2_Q8_FINITE_SCALE_SECONDARY_ISAS ${V2_Q8_SCALE_ISA})
    endif()
endforeach()
if(V2_Q8_FINITE_SCALE_SECONDARY_ISAS)
    add_subdirectory("${V2_NUMERIC_ROOT}/tests/v2/cpu_secondary_envelopes"
        "${CMAKE_CURRENT_BINARY_DIR}/cpu_secondary_envelopes")
endif()

foreach(V2_Q8_SCALE_ISA IN LISTS V2_Q8_FINITE_SCALE_CPU_ISAS)
    string(TOLOWER "${V2_Q8_SCALE_ISA}" V2_Q8_SCALE_TARGET_ISA)
    set(V2_Q8_SCALE_TARGET v2_test_q8_finite_scale_contract_${V2_Q8_SCALE_TARGET_ISA})
    v2_add_core_free_test(V2_Integration_Q8FiniteScaleContract_CPU_${V2_Q8_SCALE_ISA}
        COMMAND $<TARGET_FILE:${V2_Q8_SCALE_TARGET}>
            --gtest_filter=CPUExplicitRoundingContract.Q8FiniteScale*
        LABELS V2 Integration CPU ${V2_Q8_SCALE_ISA} Quantization Numerics ScaleRange Regression)
    list(APPEND V2_Q8_FINITE_SCALE_CPU_PREFLIGHT_TESTS
        V2_Integration_Q8FiniteScaleContract_CPU_${V2_Q8_SCALE_ISA})

    v2_add_core_free_test(V2_Integration_FP32SubnormalContract_CPU_${V2_Q8_SCALE_ISA}
        COMMAND $<TARGET_FILE:${V2_Q8_SCALE_TARGET}>
            --gtest_filter=CPUExplicitRoundingContract.Q8FiniteScaleCompiledISAPath:CPUExplicitRoundingContract.DeviceFP32SubnormalFixedWords
        LABELS V2 Integration CPU ${V2_Q8_SCALE_ISA} NumericalContract ByteExact Regression)
    list(APPEND V2_FP32_SUBNORMAL_PREFLIGHT_TESTS
        V2_Integration_FP32SubnormalContract_CPU_${V2_Q8_SCALE_ISA})
endforeach()

v2_add_core_free_test(V2_Integration_CPUExplicitRoundingContract
    COMMAND $<TARGET_FILE:v2_test_cpu_explicit_rounding_contract>
    LABELS V2 Integration ProductionTestPreflight CPU Numerics MoE AllFormats ByteExact Regression)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUExplicitRoundingContract)
# Focused regression: the native half conversion's finite/non-finite seam
# (max-finite 65504 saturates to 65504, never +inf; NaN payloads are kept).
v2_add_core_free_test(V2_Integration_CPUNativeHalfScaleFiniteSeam
    COMMAND $<TARGET_FILE:v2_test_cpu_explicit_rounding_contract>
        --gtest_filter=CPUExplicitRoundingContract.NativeHalfScaleFiniteSeamSaturatesAndKeepsPayloads:CPUExplicitRoundingContract.NativeHalfScaleEveryEncodingIsByteExact
    LABELS V2 Integration ProductionTestPreflight CPU Numerics ByteExact Regression)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUNativeHalfScaleFiniteSeam)

if(LLAMINAR_CPU_TARGET_PROCESSOR MATCHES "^(arm64|aarch64|ARM64|AARCH64)$")
    add_executable(v2_test_cpu_native_neon_numerical_contract
        "${V2_NUMERIC_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUNativeNEONNumericalContract.cpp")
    target_include_directories(v2_test_cpu_native_neon_numerical_contract PRIVATE
        "${V2_NUMERIC_ROOT}/src/v2" "${V2_NUMERIC_ROOT}/tests/v2")
    target_compile_features(v2_test_cpu_native_neon_numerical_contract PRIVATE cxx_std_20)
    target_link_libraries(v2_test_cpu_native_neon_numerical_contract PRIVATE GTest::gtest_main)
    v2_add_core_free_test(V2_Unit_CPUNativeNEONNumericalContract
        COMMAND $<TARGET_FILE:v2_test_cpu_native_neon_numerical_contract>
        LABELS V2 Unit CPU NEON Numerics ByteExact)
    v2_add_core_free_test(V2_Integration_CPUNativeNEONNumericalContract
        COMMAND $<TARGET_FILE:v2_test_cpu_native_neon_numerical_contract>
        LABELS V2 Integration ProductionTestPreflight CPU NEON NativeQuantised Numerics ByteExact Regression)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUNativeNEONNumericalContract)

    add_executable(v2_test_cpu_native_neon_prepared_projection
        "${V2_NUMERIC_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUNativeNEONPreparedProjection.cpp")
    target_include_directories(v2_test_cpu_native_neon_prepared_projection PRIVATE
        "${V2_NUMERIC_ROOT}/src/v2" "${V2_NUMERIC_ROOT}/tests/v2")
    target_compile_features(v2_test_cpu_native_neon_prepared_projection PRIVATE cxx_std_20)
    target_link_libraries(v2_test_cpu_native_neon_prepared_projection PRIVATE GTest::gtest_main
        llaminar2_native_partition_plan)
    v2_add_core_free_test(V2_Unit_CPUNativeNEONPreparedProjection
        COMMAND $<TARGET_FILE:v2_test_cpu_native_neon_prepared_projection>
        LABELS V2 Unit CPU NEON NativeQuantised Numerics ByteExact)
    v2_add_core_free_test(V2_Integration_CPUNativeNEONPreparedProjection
        COMMAND $<TARGET_FILE:v2_test_cpu_native_neon_prepared_projection>
        LABELS V2 Integration ProductionTestPreflight CPU NEON NativeQuantised Numerics ByteExact Regression)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUNativeNEONPreparedProjection)
    # Padding contract at N%4 = 1/2/3 tails: poisoned loaded padding is rejected
    # before any write; benign padding keeps serial bytes for every family/policy.
    v2_add_core_free_test(V2_Integration_CPUNativeNEONTailPaddingContract
        COMMAND $<TARGET_FILE:v2_test_cpu_native_neon_prepared_projection>
            --gtest_filter=*TailPaddingContract
        LABELS V2 Integration ProductionTestPreflight CPU NEON NativeQuantised Metadata ByteExact Regression)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUNativeNEONTailPaddingContract)
    # Native quantized NEON entries admit the calling thread's FPCR profile
    # (RNE, FZ/FZ16/AHP clear) and fail fatally under any other: the staged
    # coded factors and the FP32 contribution program round under it.
    v2_add_core_free_test(V2_Integration_CPUNativeNEONFloatingPointProfile
        COMMAND $<TARGET_FILE:v2_test_cpu_native_neon_prepared_projection>
            --gtest_filter=CPUNativeNEONFloatingPointProfile.*
        LABELS V2 Integration ProductionTestPreflight CPU NEON NativeQuantised Numerics Regression)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUNativeNEONFloatingPointProfile)
    v2_add_core_free_test(V2_Integration_CPUNativeSourceCoefficientMetadata_NEON
        COMMAND $<TARGET_FILE:v2_test_cpu_native_neon_prepared_projection>
            --gtest_filter=CPUNativeNEONPreparedProjection.RejectsInvalidSourceMetadataBeforeAnyWrite
        LABELS V2 Integration ProductionTestPreflight CPU NEON NativeQuantised Metadata Regression)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUNativeSourceCoefficientMetadata_NEON)
    # Decode panels (idea 04): every panel width writes the strip walk's bytes,
    # including partial strips and strips wholly in sealed padding columns, and
    # the lane-indexed activation leaf equals the broadcast-word leaf.
    v2_add_core_free_test(V2_Integration_CPUNativeNEONPanelTailsMatchSerial
        COMMAND $<TARGET_FILE:v2_test_cpu_native_neon_prepared_projection>
            --gtest_filter=*PanelWidthsMatchTheStripProgram:CPUNativeNEONMultiScaleOps.*
        LABELS V2 Integration ProductionTestPreflight CPU NEON NativeQuantised ByteExact AllFormats Regression)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUNativeNEONPanelTailsMatchSerial)
    add_executable(v2_test_cpu_native_neon_packed_prefill
        "${V2_NUMERIC_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUNativeNEONPackedPrefill.cpp")
    target_include_directories(v2_test_cpu_native_neon_packed_prefill PRIVATE
        "${V2_NUMERIC_ROOT}/src/v2" "${V2_NUMERIC_ROOT}/tests/v2")
    target_compile_features(v2_test_cpu_native_neon_packed_prefill PRIVATE cxx_std_20)
    target_link_libraries(v2_test_cpu_native_neon_packed_prefill PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX
        llaminar2_native_partition_plan)
    v2_add_core_free_test(V2_Unit_CPUNativeNEONPackedPrefill
        COMMAND $<TARGET_FILE:v2_test_cpu_native_neon_packed_prefill>
        LABELS V2 Unit CPU NEON NativeQuantised ByteExact)
    v2_add_core_free_test(V2_Integration_CPUNativeNEONPackedPrefill
        COMMAND $<TARGET_FILE:v2_test_cpu_native_neon_packed_prefill>
        LABELS V2 Integration ProductionTestPreflight CPU NEON NativeQuantised ByteExact Regression)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUNativeNEONPackedPrefill)
    # Row-pair SMMLA exists only in the I8MM envelope. These focused gates
    # prove its operands are the serial SDOT walk's product terms and that its
    # N-tail/odd-row bytes, including canaries past N, equal SDOT's.
    if(LLAMINAR_CPU_ISA_UPPER STREQUAL "NEON_I8MM")
        v2_add_core_free_test(V2_Integration_CPUNativeNEONI8MMOperandGrouping
            COMMAND $<TARGET_FILE:v2_test_cpu_native_neon_prepared_projection>
                --gtest_filter=*I8MMOperandGroupingMatchesSerialSDOTWalk:*I8MMRowPairPrimitivePlacesEveryProduct
            LABELS V2 Integration ProductionTestPreflight CPU NEON NativeQuantised I8MM ByteExact Regression)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUNativeNEONI8MMOperandGrouping)
        v2_add_core_free_test(V2_Integration_CPUNativeNEONI8MMTailsMatchSDOT
            COMMAND $<TARGET_FILE:v2_test_cpu_native_neon_prepared_projection>
                --gtest_filter=*I8MMRowPairsMatchSDOTAcrossNTailsAndOddRows
            LABELS V2 Integration ProductionTestPreflight CPU NEON NativeQuantised I8MM ByteExact AllFormats Regression)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUNativeNEONI8MMTailsMatchSDOT)
    endif()
    # Native MXFP4 primary scales are E8M0 bytes, never FP16: the NEON
    # tile must select the exact placement widening from the prepared identity.
    v2_add_core_free_test(V2_Integration_MXFP4NEONScaleWords
        COMMAND $<TARGET_FILE:v2_test_cpu_native_neon_prepared_projection>
            --gtest_filter=CPUNativeNEONPreparedProjection.EveryAdmittedE8M0ByteWidensExactly:CPUNativeNEONPreparedProjection.MXFP4ScaleFieldsFollowThePreparedIdentity
        LABELS V2 Integration ProductionTestPreflight CPU NEON NativeQuantised MXFP4 Metadata Regression)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MXFP4NEONScaleWords)
endif()
