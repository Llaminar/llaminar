/**
 * @file CPUCodegenEnvelopeProbe.cpp
 * @brief Optimized signed-byte motif that exposes implicit target-CPU ISA leakage.
 *
 * This compiler-only fixture needs no SDK headers, model, device or execution.
 * AppleClang's implicit apple-m1 backend can select SDOT despite ARMv8.0
 * frontend macros. Build-time assembly evidence authenticates the actual code.
 * The I8MM envelope additionally compiles one row-pair SMMLA motif, so its
 * proof shows the configured compiler really emits the admitted instruction.
 */
#if !defined(__aarch64__)
#error "Native ARM code-generation envelope proof requires AArch64"
#endif
#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_ATOMICS) || \
    defined(__ARM_FEATURE_BF16_SCALAR_ARITHMETIC) || defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#error "Unexpected optional ISA outside the declared NEON/DotProd/I8MM envelopes"
#endif
#if !defined(LLAMINAR_COMPILED_WITH_NEON_I8MM)
#error "The envelope witness must compile with the target's declared I8MM envelope"
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8) != (LLAMINAR_COMPILED_WITH_NEON_I8MM == 1)
#error "Compiler I8MM availability differs from the declared NEON_I8MM envelope"
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
#include <arm_neon.h>
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

#if defined(__ARM_FEATURE_MATMUL_INT8)
/**
 * @brief Two rows by two columns of exact eight-byte signed dots in one SMMLA.
 * @param a Two rows of eight readable signed bytes, row-major.
 * @param b Two columns of eight readable signed bytes, column-major.
 * @param out Receives the four exact INT32 sums in row-major order.
 */
extern "C" void native_cpu_codegen_row_pair_dot(const signed char *a, const signed char *b, int *out) noexcept
{
    vst1q_s32(out, vmmlaq_s32(vdupq_n_s32(0), vld1q_s8(a), vld1q_s8(b)));
}
#endif
