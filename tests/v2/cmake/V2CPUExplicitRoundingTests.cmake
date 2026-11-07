# @file V2CPUExplicitRoundingTests.cmake
# @brief Canonical native CPU explicit-rounding/Q8/half numerical proof registration.
#
# Header-only production arithmetic does not require a model, MPI or the core
# library. Share these exact targets with the focused ARM build. ISA ceilings
# use real compiler flags and native feature detection, never spoofed CPUID or
# an ARM target mislabeled as scalar/AVX2.
include("${CMAKE_CURRENT_LIST_DIR}/V2CoreFreeTestRegistration.cmake")
get_filename_component(V2_NUMERIC_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# @brief Build one copy of the existing production numerical contract tests.
# @param target Exact executable identity used by Unit/preflight registration.
function(v2_add_cpu_rounding_target target)
    add_executable(${target}
        "${V2_NUMERIC_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUExplicitRoundingContract.cpp")
    target_include_directories(${target} PRIVATE
        "${V2_NUMERIC_ROOT}/src/v2" "${V2_NUMERIC_ROOT}/tests/v2")
    target_compile_features(${target} PRIVATE cxx_std_20)
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
foreach(V2_Q8_SCALE_ISA IN LISTS V2_Q8_FINITE_SCALE_CPU_ISAS)
    string(TOLOWER "${V2_Q8_SCALE_ISA}" V2_Q8_SCALE_TARGET_ISA)
    set(V2_Q8_SCALE_TARGET v2_test_q8_finite_scale_contract_${V2_Q8_SCALE_TARGET_ISA})
    v2_add_cpu_rounding_target(${V2_Q8_SCALE_TARGET})
    target_compile_definitions(${V2_Q8_SCALE_TARGET} PRIVATE
        LLAMINAR_Q8_SCALE_TEST_${V2_Q8_SCALE_ISA}=1)
    if(V2_Q8_SCALE_ISA STREQUAL "SCALAR")
        target_compile_options(${V2_Q8_SCALE_TARGET} PRIVATE
            -mno-avx -mno-avx2 -mno-avx512f -mno-fma)
    elseif(V2_Q8_SCALE_ISA STREQUAL "AVX2")
        target_compile_options(${V2_Q8_SCALE_TARGET} PRIVATE
            -mavx2 -mfma -mno-avx512f)
    elseif(V2_Q8_SCALE_ISA STREQUAL "AVX512")
        target_compile_options(${V2_Q8_SCALE_TARGET} PRIVATE
            -mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma)
    endif()
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
        LABELS V2 Integration ProductionTestPreflight CPU NEON NativeVNNI Numerics ByteExact Regression)

    add_executable(v2_test_cpu_native_neon_prepared_projection
        "${V2_NUMERIC_ROOT}/tests/v2/unit/kernels/cpu/Test__CPUNativeNEONPreparedProjection.cpp")
    target_include_directories(v2_test_cpu_native_neon_prepared_projection PRIVATE
        "${V2_NUMERIC_ROOT}/src/v2" "${V2_NUMERIC_ROOT}/tests/v2")
    target_compile_features(v2_test_cpu_native_neon_prepared_projection PRIVATE cxx_std_20)
    target_link_libraries(v2_test_cpu_native_neon_prepared_projection PRIVATE GTest::gtest_main)
    v2_add_core_free_test(V2_Unit_CPUNativeNEONPreparedProjection
        COMMAND $<TARGET_FILE:v2_test_cpu_native_neon_prepared_projection>
        LABELS V2 Unit CPU NEON NativeVNNI Numerics ByteExact)
    v2_add_core_free_test(V2_Integration_CPUNativeNEONPreparedProjection
        COMMAND $<TARGET_FILE:v2_test_cpu_native_neon_prepared_projection>
        LABELS V2 Integration ProductionTestPreflight CPU NEON NativeVNNI Numerics ByteExact Regression)
    v2_add_core_free_test(V2_Integration_CPUNativeSourceCoefficientMetadata_NEON
        COMMAND $<TARGET_FILE:v2_test_cpu_native_neon_prepared_projection>
            --gtest_filter=CPUNativeNEONPreparedProjection.RejectsReservedSourceCoefficientBits
        LABELS V2 Integration ProductionTestPreflight CPU NEON NativeVNNI Metadata Regression)
    # Native MXFP4 primary words are BF16 powers of two, never FP16: the NEON
    # tile must select the exact placement widening from the prepared identity.
    v2_add_core_free_test(V2_Integration_MXFP4NEONScaleWords
        COMMAND $<TARGET_FILE:v2_test_cpu_native_neon_prepared_projection>
            --gtest_filter=CPUNativeNEONPreparedProjection.EveryAdmittedPowerOfTwoBF16WordWidensExactly:CPUNativeNEONPreparedProjection.MXFP4ScaleWordsFollowThePreparedIdentity:CPUNativeNEONPreparedProjection.RotatedMXFP4ExecutesExpandedInt8WithFP16Scales
        LABELS V2 Integration ProductionTestPreflight CPU NEON NativeVNNI MXFP4 Metadata Regression)
endif()
