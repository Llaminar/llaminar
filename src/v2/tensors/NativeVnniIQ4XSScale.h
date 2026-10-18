/**
 * @file NativeVnniIQ4XSScale.h
 * @brief Lossless raw IQ4_XS source coefficient metadata shared by CPU and GPU.
 *
 * The primary word is the original half d. The secondary word stores the
 * original encoded six-bit ls, with ten reserved zero bits. It is never a half
 * or additive minimum. Consumers widen d and multiply by coefficient once in
 * FP32 before the unchanged activation-scale/integer-dot contribution.
 */
#pragma once
#include <cstdint>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_IQ4XS_SCALE_HD __host__ __device__
#else
#define LLAMINAR_IQ4XS_SCALE_HD
#endif
namespace llaminar2::native_vnni_iq4xs_scale
{
    /** @brief Widen an original encoded source coefficient without re-encoding.
     * @param source_ls Original six-bit source value0..63.
     * @return Widened source word. Invalid high bits remain detectable by valid().
     */
    LLAMINAR_IQ4XS_SCALE_HD constexpr uint16_t
    encodeSourceCoefficient(uint8_t source_ls) noexcept { return source_ls; }

    /** @brief Decode the original signed factor.
     * @param metadata Valid raw source coefficient word.
     * @return Exact integer factor in[-32,31].
     */
    LLAMINAR_IQ4XS_SCALE_HD constexpr int
    coefficient(uint16_t metadata) noexcept { return static_cast<int>(metadata & 63u) - 32; }

    /** @brief Reject non-source bits before adopting raw coefficient metadata.
     * @param metadata Candidate secondary word.
     * @return Whether all ten reserved bits are zero.
     */
    LLAMINAR_IQ4XS_SCALE_HD constexpr bool
    valid(uint16_t metadata) noexcept { return (metadata & 0xffc0u) == 0; }
    static_assert(coefficient(encodeSourceCoefficient(35)) == 3);
    static_assert(valid(63) && !valid(64));
}
#undef LLAMINAR_IQ4XS_SCALE_HD
