/**
 * @file CPUCodegenEnvelopeProbe.cpp
 * @brief Optimized signed-byte motif that exposes implicit target-CPU ISA leakage.
 *
 * This compiler-only fixture needs no SDK headers, model, device or execution.
 * AppleClang's implicit apple-m1 backend can select SDOT despite ARMv8.0
 * frontend macros. Build-time assembly evidence authenticates the actual code.
 */
#if !defined(__aarch64__)
#error "Native ARM code-generation envelope proof requires AArch64"
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8) || defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_ATOMICS) || \
    defined(__ARM_FEATURE_BF16_SCALAR_ARITHMETIC) || defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#error "Unexpected optional ISA outside the declared NEON/DotProd envelopes"
#endif

/**
 * @brief Sum a complete block's exact signed-byte products for auto-vectorization.
 * @param a Thirty-two readable signed bytes.
 * @param b Thirty-two readable signed bytes.
 * @return Exact bounded INT32 sum.
 */
extern "C" int native_cpu_codegen_byte_dot(const signed char *a, const signed char *b) noexcept
{
    int sum = 0;
    for (int k = 0; k < 32; ++k) sum += int(a[k]) * int(b[k]);
    return sum;
}
