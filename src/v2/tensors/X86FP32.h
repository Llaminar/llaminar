/**
 * @file X86FP32.h
 * @brief Register-only rounding edges for explicitly separate x86 FP32 programs.
 *
 * A decoded FP32 product must finish before typed residual addition. Ordinary
 * compiler contraction can otherwise turn the intrinsics' multiply/add source
 * sequence into an FMA, changing the requantized scale and payload. These opaque
 * vector-register dependencies preserve the declared product without temporary
 * banks, memory fences, global compiler switches or an outlined arithmetic path.
 * They do not replace intentional FMA programs or change per-ISA quantization.
 */
#pragma once

#if !defined(__AVX2__) && !defined(__AVX512F__)
#error "Native x86 FP32 register boundaries require an AVX2 or AVX512 compiler envelope"
#endif
#if !defined(__GNUC__) && !defined(__clang__)
#error "Native x86 FP32 register boundaries require GNU-compatible vector constraints"
#endif
#include <immintrin.h>

namespace llaminar2::x86_fp32
{
#if defined(__AVX2__)
/** @brief Retain eight completed FP32 results before a subsequent operation.
 * @param value Already rounded binary32 lanes in the original program order.
 * @return Identical lane words opaque to subsequent compiler contraction.
 * @details The vector constraint admits only registers enabled by this compiler
 * envelope; it does not grant an AVX2 object access to AVX512 hardware.
 */
[[gnu::always_inline]] inline __m256 persistRounded(__m256 value) noexcept
{
    asm volatile("" : "+v"(value));
    return value;
}
#endif
#if defined(__AVX512F__)
/** @brief Retain sixteen completed FP32 results before a subsequent operation.
 * @param value Already rounded binary32 lanes in the original program order.
 * @return Identical lane words without a stack publication or FP-environment change.
 */
[[gnu::always_inline]] inline __m512 persistRounded(__m512 value) noexcept
{
    asm volatile("" : "+v"(value));
    return value;
}
#endif
} // namespace llaminar2::x86_fp32
