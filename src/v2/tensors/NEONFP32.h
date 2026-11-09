/**
 * @file NEONFP32.h
 * @brief Allocation-free AArch64 register boundaries for separately rounded FP32 programs.
 *
 * A function-local contraction option is not a retained arithmetic boundary:
 * GCC can contract operations after an always-inline helper enters a caller
 * compiled with its ordinary contraction policy. The opaque SIMD-register
 * dependency below preserves the original binary32 result without memory,
 * synchronization, an outlined call, or a different floating-point environment.
 * Explicit FMA instructions remain explicit at the sites whose programs require
 * them; these helpers protect only declared separate multiply/add/subtract edges.
 */
#pragma once

#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "Native FP32 register boundaries require AArch64 ASIMD"
#endif
#if !defined(__GNUC__) && !defined(__clang__)
#error "Native FP32 register boundaries require GNU-compatible register constraints"
#endif
#include <arm_neon.h>

namespace llaminar2::neon_fp32
{
/** @brief Retain one completed binary32 operation in each native lane.
 * @param value Four already rounded FP32 results.
 * @return Identical lane words, opaque to subsequent compiler contraction.
 */
[[gnu::always_inline]] inline float32x4_t persistRounded(float32x4_t value) noexcept
{
    asm volatile("" : "+w"(value));
    return value;
}

/** @brief Multiply lanes with a retained separate binary32 rounding edge.
 * @param lhs Four original multiplicands.
 * @param rhs Four original multipliers.
 * @return Separately rounded products, not inputs to an implicit FMA.
 */
[[gnu::always_inline]] inline float32x4_t multiply(float32x4_t lhs, float32x4_t rhs) noexcept
{ return persistRounded(vmulq_f32(lhs, rhs)); }

/** @brief Add lanes with a retained separate binary32 rounding edge.
 * @param lhs Four original addends.
 * @param rhs Four original addends.
 * @return Separately rounded sums.
 */
[[gnu::always_inline]] inline float32x4_t add(float32x4_t lhs, float32x4_t rhs) noexcept
{ return persistRounded(vaddq_f32(lhs, rhs)); }

/** @brief Subtract lanes without contracting an already completed product.
 * @param lhs Four minuends.
 * @param rhs Four separately rounded subtrahends.
 * @return Separately rounded differences.
 */
[[gnu::always_inline]] inline float32x4_t subtract(float32x4_t lhs, float32x4_t rhs) noexcept
{ return persistRounded(vsubq_f32(lhs, rhs)); }
} // namespace llaminar2::neon_fp32
