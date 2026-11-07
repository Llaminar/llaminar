/**
 * @file CPUNativeNEONMultiScaleOps.h
 * @brief Native four-column ASIMD/DotProd operations for shared multi-scale kernels.
 *
 * The higher-level decoder and rounded arithmetic tree are ISA-independent.
 * AArch64 has no general integer gather: each immutable eight-byte codebook
 * entry is loaded once into a vector lane, then transposed into four parallel
 * column dots. This is bounded address generation, not a scalar projection or
 * a decoded-weight cache. Signed dots never use saturating pair arithmetic.
 */
#pragma once

#include "CPUNativeNEONContributionContract.h"
#include "tensors/NEONFP16.h"
#include <arm_neon.h>
#include <bit>
#include <cstdint>

namespace llaminar2::cpu::native_vnni::multi_scale
{
    template <int Lanes> struct SIMD;

    /** @brief Exact signed-byte instruction family, sealed outside the K traversal. */
    enum class NEONByteDot { ASIMD, DotProd };

    /** @brief Four native AArch64 columns over the existing four-byte interleaving.
     * @tparam Dot Admitted native integer instruction family.
     */
    template <NEONByteDot Dot> struct NEONMultiScaleOps
    {
        static_assert(Dot == NEONByteDot::ASIMD || Dot == NEONByteDot::DotProd);
        static constexpr NEONByteDot instruction = Dot; ///< Exact native instruction-family identity.
        using I = int __attribute__((vector_size(16)));
        using F = float __attribute__((vector_size(16)));

        /** @brief Accumulate four exact signed column dots.
         * @param acc Existing INT32 dots.
         * @param weights Four signed bytes per column.
         * @param activation One four-byte activation word repeated per column.
         * @return Updated INT32 dots without saturation or cross-column reduction.
         */
        static I dot(I acc, I weights, I activation) noexcept
        {
#if defined(__ARM_FEATURE_DOTPROD)
            if constexpr (Dot == NEONByteDot::DotProd)
                return std::bit_cast<I>(dotFourColumnsDotProd(std::bit_cast<int32x4_t>(acc),
                    std::bit_cast<int8x16_t>(activation), std::bit_cast<int8x16_t>(weights)));
#else
            static_assert(Dot == NEONByteDot::ASIMD, "DotProd operations require a DotProd-targeted compiler envelope");
#endif
            return std::bit_cast<I>(dotFourColumnsASIMD(std::bit_cast<int32x4_t>(acc),
                std::bit_cast<int8x16_t>(activation), std::bit_cast<int8x16_t>(weights)));
        }

        /** @brief Subtract within each byte without borrowing across weight elements.
         * @param a Packed four-byte integer words.
         * @param b Per-byte centering biases.
         * @return Independently centered signed bytes.
         */
        static I subtractBytes(I a, I b) noexcept
        { return std::bit_cast<I>(vsubq_s8(std::bit_cast<int8x16_t>(a), std::bit_cast<int8x16_t>(b))); }

        /** @brief Broadcast one metadata byte into each of its four column bytes.
         * @tparam Byte Byte index within each INT32 metadata word.
         * @param metadata Four packed column words.
         * @return Original selected bytes, repeated four times per column.
         */
        template <int Byte> static I repeatByte(I metadata) noexcept
        {
            static_assert(Byte >= 0 && Byte < 4);
            const uint8x16_t index{Byte, Byte, Byte, Byte, 4 + Byte, 4 + Byte, 4 + Byte, 4 + Byte,
                8 + Byte, 8 + Byte, 8 + Byte, 8 + Byte, 12 + Byte, 12 + Byte, 12 + Byte, 12 + Byte};
            return std::bit_cast<I>(vqtbl1q_u8(std::bit_cast<uint8x16_t>(metadata), index));
        }

        /** @brief Apply packed source sign bits to four-byte codebook groups.
         * @tparam Half Low/high four values of each eight-byte grid entry.
         * @param values Four columns of decoded signed grid bytes, excluding -128.
         * @param signs Per-column sign bytes repeated by repeatByte().
         * @return Exactly signed source grid values.
         */
        template <int Half> static I signGrid(I values, I signs) noexcept
        {
            static_assert(Half == 0 || Half == 1);
            const auto bit_positions = vreinterpretq_u8_u32(vdupq_n_u32(0x08040201u << (Half * 4)));
            const auto negative = vtstq_u8(std::bit_cast<uint8x16_t>(signs), bit_positions);
            const auto bytes = std::bit_cast<int8x16_t>(values);
            return std::bit_cast<I>(vbslq_s8(negative, vnegq_s8(bytes), bytes));
        }

        /** @brief Load each complete eight-byte immutable grid entry once.
         * @param table Canonical source grid with validated decoder indices.
         * @param index Four entry indices, one per output column.
         * @param low Receives the first four source bytes per column.
         * @param high Receives the last four source bytes per column.
         */
        static void gridPair(const std::uint64_t *table, I index, I &low, I &high) noexcept
        {
            const auto indices = std::bit_cast<int32x4_t>(index);
            auto first = vdupq_n_u64(0);
            auto second = vdupq_n_u64(0);
            first = vld1q_lane_u64(table + vgetq_lane_s32(indices, 0), first, 0);
            first = vld1q_lane_u64(table + vgetq_lane_s32(indices, 1), first, 1);
            second = vld1q_lane_u64(table + vgetq_lane_s32(indices, 2), second, 0);
            second = vld1q_lane_u64(table + vgetq_lane_s32(indices, 3), second, 1);
            low = std::bit_cast<I>(vuzp1q_u32(vreinterpretq_u32_u64(first), vreinterpretq_u32_u64(second)));
            high = std::bit_cast<I>(vuzp2q_u32(vreinterpretq_u32_u64(first), vreinterpretq_u32_u64(second)));
        }

        /** @brief Widen four persisted halves without changing nonfinite payload bits.
         * @param source Four readable canonical half words.
         * @return Four exact binary32 conversions.
         */
        static F half(const std::uint16_t *source) noexcept
        { return expandHalfBits(vld1_u16(source)); }

        /** @brief Widen opaque metadata without interpreting its half exponent.
         * @param source Four readable raw metadata words.
         * @return Four zero-extended INT32 metadata words.
         */
        static I words(const std::uint16_t *source) noexcept
        { return std::bit_cast<I>(vmovl_u16(vld1_u16(source))); }

        /** @brief Reject reserved bits in every opaque source metadata lane.
         * @param metadata Four zero-extended original source words.
         * @param permitted_bits Bit positions owned by the source coefficient ABI.
         * @return True only when no lane contains an unowned bit.
         */
        static bool metadataFits(I metadata, std::uint32_t permitted_bits) noexcept
        {
            const auto reserved = vandq_u32(std::bit_cast<uint32x4_t>(metadata), vdupq_n_u32(~permitted_bits));
            return vmaxvq_u32(reserved) == 0;
        }

        /** @brief Convert the low half word of each packed INT32 lane.
         * @param packed Four raw low/high half-word pairs.
         * @return Four exact binary32 low-word conversions.
         */
        static F halfWords(I packed) noexcept
        { return expandHalfBits(vmovn_u32(std::bit_cast<uint32x4_t>(packed))); }

    private:
        /** @brief Preserve the portable half contract while using native finite conversion.
         * @param half Four raw binary16 words.
         * @return Exact FP32 words, including unmodified NaN payload/signaling bits.
         */
        static F expandHalfBits(uint16x4_t half) noexcept
        { return std::bit_cast<F>(neon_fp16::widen(half)); }
    };

    /** @brief Baseline four-column specialization; DotProd is an explicitly selected type. */
    template <> struct SIMD<4> : NEONMultiScaleOps<NEONByteDot::ASIMD> {};
} // namespace llaminar2::cpu::native_vnni::multi_scale
