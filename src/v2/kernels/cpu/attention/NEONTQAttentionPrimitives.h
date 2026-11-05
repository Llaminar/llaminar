/**
 * @file NEONTQAttentionPrimitives.h
 * @brief Native ASIMD centroid scoring and numerator updates from checked TQ cache heads.
 *
 * TQ4 reconstructs eight original 3+1-bit indices in registers and gathers exact
 * float words from the original 64-byte codebook with TBL. TQ8 uses four native
 * indexed lane loads because ASIMD has no floating gather instruction. Both
 * feed native FP32 vector arithmetic; neither decodes a head/cache mirror nor
 * substitutes a whole-operation scalar routine. Checked operands retain the
 * cache/codebook authorities through the last store without owning their bytes.
 */
#pragma once
#include "kernels/cpu/attention/CPUTQAttentionOperands.h"
#include "utils/CPUFeatures.h"
#if !LLAMINAR_COMPILED_WITH_NEON
#error "Native TQ attention requires an AArch64 ASIMD compiler envelope"
#endif
#include <arm_neon.h>
#include <bit>
#include <type_traits>

namespace llaminar2::cpu::neon_tq_attention
{
using tq_attention::Format;
/** @brief Two original adjacent four-coordinate fragments, never a decoded head. */
struct Centroids8 { float32x4_t lo, hi; };

/** @brief Four register fragments of the original immutable sixteen-entry codebook. */
class TQ4Lookup final
{
public:
    /** @brief Load the sole published codebook; no new physical storage owner is created. */
    TQ4Lookup() noexcept
    {
        static_assert(std::endian::native == std::endian::little);
        const auto *bytes = reinterpret_cast<const std::uint8_t *>(TQ4_CENTROIDS.data());
        for (int i = 0; i < 4; ++i) table_.val[i] = vld1q_u8(bytes + i * 16);
    }
private:
    /** @brief Gather four exact FP32 words without floating approximation or scalar lookup loops.
     * @param indices Four original indices, each proven in [0,15] by decode masking.
     * @return Exact codebook words at those original coordinates.
     */
    [[gnu::always_inline]] float32x4_t lookup(uint32x4_t indices) const noexcept
    {
        const auto bases = vreinterpretq_u8_u32(vmulq_n_u32(indices, 0x04040404U));
        const auto offsets = vaddq_u8(bases, vreinterpretq_u8_u32(vdupq_n_u32(0x03020100U)));
        return vreinterpretq_f32_u8(vqtbl4q_u8(table_, offsets));
    }
public:
    /** @brief Consume one bounded three-byte group and its matching high-bit byte.
     * @param packed Original consecutive 24 low bits; exactly three bytes are read.
     * @param high Original high-bit plane byte for these same eight coordinates.
     * @return Original-order native centroid fragments for coordinates 0..3 and 4..7.
     */
    [[gnu::always_inline]] Centroids8 decode(const std::uint8_t *packed, std::uint8_t high) const noexcept
    {
        const std::uint32_t word = packed[0] | (std::uint32_t(packed[1]) << 8) | (std::uint32_t(packed[2]) << 16);
        const int32x4_t low_shifts = {0, -3, -6, -9}, high_shifts = {-12, -15, -18, -21};
        const int32x4_t bit_lo = {0, -1, -2, -3}, bit_hi = {-4, -5, -6, -7};
        const auto low = vdupq_n_u32(word), plane = vdupq_n_u32(high);
        const auto lo = vorrq_u32(vandq_u32(vshlq_u32(low, low_shifts), vdupq_n_u32(7)),
            vshlq_n_u32(vandq_u32(vshlq_u32(plane, bit_lo), vdupq_n_u32(1)), 3));
        const auto hi = vorrq_u32(vandq_u32(vshlq_u32(low, high_shifts), vdupq_n_u32(7)),
            vshlq_n_u32(vandq_u32(vshlq_u32(plane, bit_hi), vdupq_n_u32(1)), 3));
        return {lookup(lo), lookup(hi)};
    }
private:
    uint8x16x4_t table_;
};

/** @brief Perform four indexed native lane loads from the original TQ8 codebook.
 * @param indices Four immutable original byte indices, intrinsically within [0,255].
 * @return A native four-lane centroid vector; no scalar arithmetic replaces SIMD math.
 */
[[gnu::always_inline]] inline float32x4_t lookupTQ8(const std::uint8_t *indices) noexcept
{
    auto values = vdupq_n_f32(0);
    values = vld1q_lane_f32(TQ8_CENTROIDS.data() + indices[0], values, 0);
    values = vld1q_lane_f32(TQ8_CENTROIDS.data() + indices[1], values, 1);
    values = vld1q_lane_f32(TQ8_CENTROIDS.data() + indices[2], values, 2);
    return vld1q_lane_f32(TQ8_CENTROIDS.data() + indices[3], values, 3);
}

/** @brief Empty compile-time storage: TQ8 never loads or retains the unrelated TQ4 table. */
struct NoTQ4Lookup {};

/** @brief Decode original eight-coordinate fragments with format-exact register resources.
 * @tparam Encoding Existing physical layout; no source/codebook ownership is extended.
 */
template <Format Encoding> class CentroidDecoder final
{
    static_assert(Encoding == Format::TQ4 || Encoding == Format::TQ8);
public:
    /** @brief Read one original eight-coordinate fragment without a decoded head mirror.
     * @param block Original immutable block; @param dimensions Checked complete head width.
     * @param coordinate Original multiple-of-eight coordinate.
     * @return Exact original centroid words in two native registers.
     */
    [[gnu::always_inline]] Centroids8 decode(const std::uint8_t *block, int dimensions, int coordinate) const noexcept
    {
        if constexpr (Encoding == Format::TQ4)
            return table_.decode(block + 8 + (coordinate / 8) * 3, block[8 + dimensions * 3 / 8 + coordinate / 8]);
        else
            return {lookupTQ8(block + 8 + coordinate), lookupTQ8(block + 8 + coordinate + 4)};
    }
private:
    [[no_unique_address]] std::conditional_t<Encoding == Format::TQ4, TQ4Lookup, NoTQ4Lookup> table_;
};

/** @brief Execute one native source-norm score program with a fixed same-backend reduction tree.
 * @tparam Encoding Original cache layout.
 * @param input Checked immutable query/cache borrows and original source norm.
 * @return Ordered native two-chain dot, pairwise folded and multiplied once by the source norm.
 * @details Grouping may change ownership, never coordinate order or this arithmetic tree.
 */
template <Format Encoding>
inline float dot(const tq_attention::DotOperand<Encoding> &input) noexcept
{
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
    if (!input.active()) return 0;
    const CentroidDecoder<Encoding> decoder;
    auto lo = vdupq_n_f32(0), hi = lo;
    for (int coordinate = 0; coordinate < input.head().dimensions(); coordinate += 8)
    {
        const auto values = decoder.decode(input.head().data(), input.head().dimensions(), coordinate);
        lo = vfmaq_f32(lo, values.lo, vld1q_f32(input.query() + coordinate));
        hi = vfmaq_f32(hi, values.hi, vld1q_f32(input.query() + coordinate + 4));
    }
    const auto combined = vaddq_f32(lo, hi);
    const auto pairs = vpadd_f32(vget_low_f32(combined), vget_high_f32(combined));
    const float sum = vget_lane_f32(vpadd_f32(pairs, pairs), 0);
    return device_fp32_contract::multiply(sum, input.head().norm());
}

/** @brief Update original numerator coordinates with the fixed weighted-centroid FMA program.
 * @tparam Encoding Original value layout.
 * @param input Checked writable numerator and disjoint cache/codebook borrows.
 * @details One rounded weight/radius product feeds every coordinate. No output
 * becomes visible through a new cache publication; the caller owns completion.
 */
template <Format Encoding>
inline void accumulate(const tq_attention::AccumOperand<Encoding> &input) noexcept
{
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
    if (!input.active()) return;
    const CentroidDecoder<Encoding> decoder;
    for (int coordinate = 0; coordinate < input.head().dimensions(); coordinate += 8)
    {
        const auto values = decoder.decode(input.head().data(), input.head().dimensions(), coordinate);
        auto *output = input.output() + coordinate;
        vst1q_f32(output, vfmaq_n_f32(vld1q_f32(output), values.lo, input.combinedWeight()));
        vst1q_f32(output + 4, vfmaq_n_f32(vld1q_f32(output + 4), values.hi, input.combinedWeight()));
    }
}
} // namespace llaminar2::cpu::neon_tq_attention
