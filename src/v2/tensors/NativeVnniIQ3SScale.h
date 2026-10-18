/**
 * @file NativeVnniIQ3SScale.h
 * @brief Lossless IQ3_S source factors and exact inverse of its signed grid.
 *
 * Primary metadata retains the original half d; secondary metadata is the
 * original scale nibble in low4, never a half or an additive minimum. Consumers
 * widen d and multiply by coefficient() in FP32 before the unchanged block
 * contribution. Tier movement inversely encodes the existing decoded INT8 grid
 * with bounded exact lookup in the already resident canonical512-entry table.
 * No nearest-value search, re-quantization, extra table or allocation is used.
 */
#pragma once
#include <cstdint>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_IQ3S_SCALE_HD __host__ __device__
#else
#define LLAMINAR_IQ3S_SCALE_HD
#endif
namespace llaminar2::native_vnni_iq3s_scale
{
    /** @brief Preserve a source nibble; invalid high bits remain detectable.
     * @param nibble Original source scale nibble0..15.
     * @return Raw widened metadata, without half conversion.
     */
    LLAMINAR_IQ3S_SCALE_HD constexpr uint16_t
    encodeSourceCoefficient(uint8_t nibble) noexcept { return nibble; }

    /** @brief Resolve the exact original odd multiplier.
     * @param metadata Valid source nibble word.
     * @return Integer1+2*nibble in[1,31].
     */
    LLAMINAR_IQ3S_SCALE_HD constexpr int
    coefficient(uint16_t metadata) noexcept { return 1+2*static_cast<int>(metadata&15u); }

    /** @brief Reject reserved/non-source coefficient bits.
     * @param metadata Candidate raw source word.
     * @return Whether high12 bits are zero.
     */
    LLAMINAR_IQ3S_SCALE_HD constexpr bool
    valid(uint16_t metadata) noexcept { return (metadata&0xfff0u)==0; }

    /** Grid size doubles as the explicit absent-index sentinel. */
    inline constexpr uint16_t kInvalidGridIndex=512;

    /** @brief Exact source index/sign result for one four-value grid group.
     * Invalid results cannot be published as payload: the caller must reject
     * them through its existing fatal movement path before storing a block.
     */
    struct EncodedGridQuartet
    {
        uint16_t index=kInvalidGridIndex; ///< Nine-bit source index or invalid sentinel.
        uint8_t signs=0; ///< Sign bit i belongs to original value i.
        /** @return Whether the exact source-grid membership proof succeeded. */
        LLAMINAR_IQ3S_SCALE_HD constexpr bool valid() const noexcept
        { return index<kInvalidGridIndex && (signs&0xf0u)==0; }
    };

    /** @brief Find an exact packed positive quartet in the canonical sorted grid.
     * @param magnitudes Four little-endian positive magnitude bytes.
     * @param grid Existing canonical512-word table, strictly ordered and unique.
     * @return Exact9-bit source index, or kInvalidGridIndex. At most10 bounded
     *         lower-bound comparisons plus one equality check; no approximation.
     */
    LLAMINAR_IQ3S_SCALE_HD constexpr uint16_t
    sourceGridIndex(uint32_t magnitudes,const uint32_t *grid) noexcept
    {
        if (!grid) return kInvalidGridIndex;
        unsigned first=0,last=kInvalidGridIndex;
        for (unsigned iteration=0;iteration<10 && first<last;++iteration)
        {
            const unsigned middle=first+(last-first)/2;
            if (grid[middle]<magnitudes) first=middle+1;
            else last=middle;
        }
        return first<kInvalidGridIndex && grid[first]==magnitudes
            ? static_cast<uint16_t>(first) : kInvalidGridIndex;
    }

    /** @brief Recover one original index and its four source sign bits exactly.
     * @param values Four contiguous decoded signed grid bytes in original order.
     * @param grid Already resident canonical IQ3_S grid; no new lookup allocation.
     * @return Typed exact encoding, or an invalid result for null, zero, even,
     *         out-of-range or non-grid input. No output storage is modified.
     */
    LLAMINAR_IQ3S_SCALE_HD constexpr EncodedGridQuartet
    encodeGridQuartet(const int8_t *values,const uint32_t *grid) noexcept
    {
        if (!values || !grid) return {};
        uint32_t magnitudes=0;uint8_t signs=0;
        for (unsigned i=0;i<4;++i)
        {
            const int value=values[i];const int magnitude=value<0?-value:value;
            if (magnitude==0 || magnitude>15 || (magnitude&1)==0) return {};
            magnitudes|=static_cast<uint32_t>(magnitude)<<(8*i);
            if (value<0) signs|=static_cast<uint8_t>(1u<<i);
        }
        return {sourceGridIndex(magnitudes,grid),signs};
    }
    static_assert(coefficient(encodeSourceCoefficient(0))==1);
    static_assert(coefficient(encodeSourceCoefficient(15))==31);
    static_assert(valid(15) && !valid(16));
}
#undef LLAMINAR_IQ3S_SCALE_HD
