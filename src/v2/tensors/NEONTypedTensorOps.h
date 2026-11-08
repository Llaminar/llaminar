/**
 * @file NEONTypedTensorOps.h
 * @brief Native AArch64 typed arithmetic, conversion and activation Q8 leaves.
 *
 * Values stay in bounded native vector tiles. Typed residual requantization
 * retains its existing half/full scale, zero thresholds and ties-away rounding;
 * it is not the distinct GPU-aligned expert activation quantizer. Full ordinary
 * activation and SwiGLU blocks retain separate ties-even/division thresholds. Block-only
 * operations reject malformed counts rather than dropping elements. Floating
 * outputs support bounded final-vector tails and never require a tensor mirror.
 * Decoded products carry explicit register rounding edges into their addition;
 * inlining must not let the compiler turn either source decode into an FMA.
 */
#pragma once

#include "BlockStructures.h"
#include "FP16Utils.h"
#include "NEONFP16.h"
#include "NEONFP32.h"
#include "kernels/common/DeviceFP32NumericalContract.h"
#include "TypedBlockGeometry.h"
#include <arm_neon.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#if defined(__clang__)
#define LLAMINAR_NEON_TYPED_FP _Pragma("clang fp contract(off)") _Pragma("clang fp reassociate(off)")
#define LLAMINAR_NEON_TYPED_EXACT
#elif defined(__GNUC__)
#define LLAMINAR_NEON_TYPED_FP
#define LLAMINAR_NEON_TYPED_EXACT __attribute__((optimize("fp-contract=off", "no-associative-math")))
#else
#error "Native typed tensor arithmetic requires explicit FP contraction controls"
#endif

namespace llaminar2::simd
{
    namespace neon_typed
    {
        using Values = std::array<float32x4_t, 8>; ///< One 32-element private register tile.
        /** @brief Original typed Q8 scale/inverse programs, not an inferred precision choice. */
        enum class Q8Program { Residual, Tensor, FullBlockRNE, SwiGLUBlockRNE };

        /** @brief Reject a partial count for an API whose unit is one complete typed block.
         * @param count Logical element count, including zero for an empty operation.
         * @throws std::invalid_argument Unless count is a multiple of 32.
         */
        inline void requireBlocks(std::size_t count)
        { requireCompleteTypedBlocks(count); }

        /** @brief Widen exactly four signed bytes without overread or aliasing.
         * @param source Four readable original signed bytes.
         * @return Four exact FP32 integers.
         */
        [[gnu::always_inline]] inline float32x4_t load4(const std::int8_t *source) noexcept
        {
            std::int32_t word;
            std::memcpy(&word, source, sizeof(word));
            const auto bytes = vreinterpret_s8_s32(vdup_n_s32(word));
            return vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(bytes))));
        }

        /** @brief Widen four signed half words exactly.
         * @param source Four readable original INT16 elements.
         * @return Four exact FP32 integers, including -32768.
         */
        [[gnu::always_inline]] inline float32x4_t load4(const std::int16_t *source) noexcept
        { return vcvtq_f32_s32(vmovl_s16(vld1_s16(source))); }

        /** @brief Decode one typed source block into a bounded register tile.
         * @tparam Block Q8_1 or Q16_1 storage type.
         * @param block Original retained block, read before an aliased output is modified.
         * @return Thirty-two separately multiplied FP32 values in source order.
         */
        template <typename Block>
        [[gnu::always_inline]] LLAMINAR_NEON_TYPED_EXACT inline Values decode(const Block &block) noexcept
        {
            LLAMINAR_NEON_TYPED_FP
            static_assert(std::is_same_v<Block, Q8_1Block> || std::is_same_v<Block, Q16_1Block>);
            const float scale = [&] {
                if constexpr (std::is_same_v<Block, Q8_1Block>) return fp16_to_fp32(block.d);
                else return block.d;
            }();
            Values values;
#pragma GCC unroll 8
            for (int group = 0; group < 8; ++group)
                values[group] = neon_fp32::multiply(load4(block.qs + group * 4), vdupq_n_f32(scale));
            return values;
        }

        /** @brief Add complete register tiles without fusing either source decode multiply.
         * @param a First separately decoded values.
         * @param b Second separately decoded values.
         * @return Thirty-two original-order sums.
         */
        [[gnu::always_inline]] LLAMINAR_NEON_TYPED_EXACT inline Values add(Values a, const Values &b) noexcept
        {
            LLAMINAR_NEON_TYPED_FP
#pragma GCC unroll 8
            for (int group = 0; group < 8; ++group) a[group] = neon_fp32::add(a[group], b[group]);
            return a;
        }

        /** @brief Load one full floating input block into a native tile.
         * @param source Thirty-two readable finite FP32 values.
         * @return Source-order values, without arithmetic or allocation.
         */
        [[gnu::always_inline]] inline Values load(const float *source) noexcept
        {
            Values values;
#pragma GCC unroll 8
            for (int group = 0; group < 8; ++group) values[group] = vld1q_f32(source + group * 4);
            return values;
        }

        /** @brief Reduce a finite tile's absolute maximum without FP additions.
         * @param values Thirty-two finite input values.
         * @return Nonnegative original binary32 maximum.
         */
        [[gnu::always_inline]] inline float maximum(const Values &values) noexcept
        {
            auto result = vdupq_n_f32(0.0f);
#pragma GCC unroll 8
            for (int group = 0; group < 8; ++group) result = vmaxq_f32(result, vabsq_f32(values[group]));
            return vmaxvq_f32(result);
        }

        /** @brief Preserve the declared residual, tensor or full-block activation Q8 program.
         * @tparam Program Original API's declared scale/inverse calculation.
         * @param values Complete finite block, already independently added where required.
         * @param output Sole destination block, which may alias a fully consumed source.
         */
        template <Q8Program Program>
        [[gnu::always_inline]] LLAMINAR_NEON_TYPED_EXACT inline void quantize8(const Values &values, Q8_1Block &output)
        {
            LLAMINAR_NEON_TYPED_FP
            static_assert(Program == Q8Program::Residual || Program == Q8Program::Tensor ||
                          Program == Q8Program::FullBlockRNE || Program == Q8Program::SwiGLUBlockRNE);
            const float max_abs = maximum(values);
            constexpr float threshold = Program == Q8Program::FullBlockRNE ? 1e-10f : 1e-6f;
            // SwiGLU compares the resulting scale, not the original maximum;
            // its stored half scale is still authored when every byte is zero.
            if (Program != Q8Program::SwiGLUBlockRNE && max_abs < threshold)
            {
                output.d = 0;
                output.sum_qs = 0;
                vst1q_s8(output.qs, vdupq_n_s8(0));
                vst1q_s8(output.qs + 16, vdupq_n_s8(0));
                return;
            }
            float scale = max_abs / 127.0f;
            if constexpr (Program == Q8Program::Tensor)
                if (scale > 65504.0f) scale = 65504.0f;
            const float inverse = [&] {
                if constexpr (Program == Q8Program::Residual) return 127.0f / max_abs;
                else if constexpr (Program == Q8Program::SwiGLUBlockRNE) return scale > 1e-10f ? 1.0f / scale : 0.0f;
                else return 1.0f / scale;
            }();
            auto sum = vdupq_n_s32(0);
#pragma GCC unroll 8
            for (int group = 0; group < 8; ++group)
            {
                const auto scaled = vmulq_n_f32(values[group], inverse);
                const auto bounded = vmaxq_f32(vdupq_n_f32(-127.0f), vminq_f32(vdupq_n_f32(127.0f), scaled));
                // The full-block activation API uses native ties-even, unlike
                // typed residual/tensor requantization's std::round program.
                const auto q = [&] {
                    if constexpr (Program == Q8Program::FullBlockRNE || Program == Q8Program::SwiGLUBlockRNE)
                        return vcvtnq_s32_f32(bounded);
                    else return vcvtaq_s32_f32(bounded);
                }();
                sum = vaddq_s32(sum, q);
                const auto bytes = vqmovn_s16(vcombine_s16(vqmovn_s32(q), vdup_n_s16(0)));
                const auto word = vget_lane_s32(vreinterpret_s32_s8(bytes), 0);
                std::memcpy(output.qs + group * 4, &word, sizeof(word));
            }
            output.d = fp32_to_fp16(scale);
            output.sum_qs = static_cast<std::int16_t>(vaddvq_s32(sum));
        }

        /** @brief Preserve full-scale Q16 residual requantization with exact signed sums.
         * @param values Complete independently decoded/added finite source block.
         * @param output Sole Q16 destination block.
         */
        [[gnu::always_inline]] LLAMINAR_NEON_TYPED_EXACT inline void quantize16(const Values &values, Q16_1Block &output)
        {
            LLAMINAR_NEON_TYPED_FP
            const float max_abs = maximum(values);
            if (max_abs < 1e-10f)
            {
                output.d = 0.0f;
                output.sum_qs = 0;
                for (int group = 0; group < 4; ++group) vst1q_s16(output.qs + group * 8, vdupq_n_s16(0));
                return;
            }
            const float scale = max_abs / 32767.0f;
            const float inverse = 32767.0f / max_abs;
            auto sum = vdupq_n_s32(0);
#pragma GCC unroll 8
            for (int group = 0; group < 8; ++group)
            {
                const auto scaled = vmulq_n_f32(values[group], inverse);
                const auto bounded = vmaxq_f32(vdupq_n_f32(-32767.0f), vminq_f32(vdupq_n_f32(32767.0f), scaled));
                const auto q = vcvtaq_s32_f32(bounded);
                vst1_s16(output.qs + group * 4, vqmovn_s32(q));
                sum = vaddq_s32(sum, q);
            }
            output.d = scale;
            output.sum_qs = vaddvq_s32(sum); // 32*32767 fits INT32, independently of worker geometry.
        }
    } // namespace neon_typed

    /** @brief Quantize one complete activation block using the native ties-even program.
     * @param source Exactly 32 readable finite FP32 values.
     * @param destination Exclusive complete Q8 block publication.
     * @details Partial blocks retain the public API's separate bounded-tail
     * program. This leaf matches the existing full-block SIMD activation
     * quantizer, not the typed tensor/residual ties-away quantizers.
     */
    inline void quantize_single_block_neon(const float *source, Q8_1Block &destination)
    {
        neon_typed::quantize8<neon_typed::Q8Program::FullBlockRNE>(neon_typed::load(source), destination);
    }

    /** @brief Add FP32 residuals in native four-value groups with at most three scalar tail values.
     * @param residual Original FP32 residual; exact output aliasing is permitted.
     * @param input Original FP32 addend.
     * @param output Exclusive destination for count values.
     * @param count Logical element count, including zero.
     */
    inline void fused_fp32_residual_add_neon(const float *residual, const float *input, float *output, std::size_t count)
    {
        std::size_t i = 0;
        for (; i + 4 <= count; i += 4) vst1q_f32(output + i, vaddq_f32(vld1q_f32(residual + i), vld1q_f32(input + i)));
        for (; i < count; ++i) output[i] = residual[i] + input[i];
    }

    /** @brief Fuse bit-exact BF16 expansion and FP32 residual addition without intermediate storage.
     * @param residual Original BF16 storage words.
     * @param input FP32 addend.
     * @param output Destination for exactly count floats.
     * @param count Live values, with bounded final-vector tails.
     */
    inline void fused_bf16_residual_add_neon(const std::uint16_t *residual, const float *input, float *output, std::size_t count)
    {
        std::size_t i = 0;
        for (; i + 4 <= count; i += 4)
            vst1q_f32(output + i, vaddq_f32(vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(residual + i), 16)), vld1q_f32(input + i)));
        for (; i < count; ++i)
        {
            const auto bits = std::uint32_t(residual[i]) << 16;
            float value;
            std::memcpy(&value, &bits, sizeof(value));
            output[i] = value + input[i];
        }
    }

    /** @brief Fuse portable-contract half expansion and FP32 residual addition.
     * @param residual Original binary16 words.
     * @param input FP32 addend.
     * @param output Exclusive live FP32 destination.
     * @param count Logical values; no source/output overread is permitted.
     */
    inline void fused_fp16_residual_add_neon(const std::uint16_t *residual, const float *input, float *output, std::size_t count)
    {
        std::size_t i = 0;
        for (; i + 4 <= count; i += 4)
            vst1q_f32(output + i, vaddq_f32(neon_fp16::widen(vld1_u16(residual + i)), vld1q_f32(input + i)));
        for (; i < count; ++i) output[i] = fp16_to_fp32(residual[i]) + input[i];
    }

    /** @brief Fuse complete Q8 residual decode and independent FP32 addition in native registers.
     * @param residual Retained Q8 blocks, each containing 32 source values.
     * @param input Live FP32 addend.
     * @param output Destination for exactly count floats.
     * @param count Logical count, allowing a bounded final block.
     */
    LLAMINAR_NEON_TYPED_EXACT inline void fused_q8_1_residual_add_neon(
        const Q8_1Block *residual, const float *input, float *output, std::size_t count)
    {
        LLAMINAR_NEON_TYPED_FP
        for (std::size_t first = 0; first < count; first += 32)
        {
            const auto &block = residual[first / 32];
            const float scale = fp16_to_fp32(block.d);
            const std::size_t live = count - first < 32 ? count - first : 32;
            std::size_t i = 0;
            for (; i + 4 <= live; i += 4)
                vst1q_f32(output + first + i, neon_fp32::add(neon_fp32::multiply(neon_typed::load4(block.qs + i), vdupq_n_f32(scale)), vld1q_f32(input + first + i)));
            for (; i < live; ++i)
            {
                const float decoded = device_fp32_contract::multiply(scale, static_cast<float>(block.qs[i]));
                output[first + i] = device_fp32_contract::add(decoded, input[first + i]);
            }
        }
    }

    /** @brief Add complete Q8 blocks with the existing residual scale/rounding program.
     * @param a First Q8 source.
     * @param b Second Q8 source.
     * @param output Destination; either exact source alias is permitted.
     * @param count Full-block element count; malformed partial counts fail.
     */
    inline void q8_1_add_q8_1_neon(const Q8_1Block *a, const Q8_1Block *b, Q8_1Block *output, std::size_t count)
    {
        neon_typed::requireBlocks(count);
        for (std::size_t block = 0; block < count / 32; ++block)
            neon_typed::quantize8<neon_typed::Q8Program::Residual>(neon_typed::add(neon_typed::decode(a[block]), neon_typed::decode(b[block])), output[block]);
    }

    /** @brief Add complete Q16 blocks while preserving full FP32 scale and signed sums.
     * @param a First source block bank.
     * @param b Second source block bank.
     * @param output Sole destination, with exact source aliasing permitted.
     * @param count Full-block element count.
     */
    inline void q16_1_add_q16_1_neon(const Q16_1Block *a, const Q16_1Block *b, Q16_1Block *output, std::size_t count)
    {
        neon_typed::requireBlocks(count);
        for (std::size_t block = 0; block < count / 32; ++block)
            neon_typed::quantize16(neon_typed::add(neon_typed::decode(a[block]), neon_typed::decode(b[block])), output[block]);
    }

    /** @brief Add FP32 delta directly to Q16 residuals without a decoded tensor.
     * @param residual Sole mutable Q16 bank.
     * @param delta Read-only FP32 addend.
     * @param count Full-block logical count.
     */
    inline void q16_1_add_fp32_neon(Q16_1Block *residual, const float *delta, std::size_t count)
    {
        neon_typed::requireBlocks(count);
        for (std::size_t block = 0; block < count / 32; ++block)
            neon_typed::quantize16(neon_typed::add(neon_typed::decode(residual[block]), neon_typed::load(delta + block * 32)), residual[block]);
    }

    /** @brief Convert complete Q16 blocks into the existing typed-residual Q8 representation.
     * @param source Original complete Q16 blocks.
     * @param destination Exclusive Q8 block bank.
     * @param blocks Physical block count, including zero.
     */
    inline void q16_1_to_q8_1_packed_neon(const Q16_1Block *source, Q8_1Block *destination, std::size_t blocks)
    {
        for (std::size_t block = 0; block < blocks; ++block)
            neon_typed::quantize8<neon_typed::Q8Program::Residual>(neon_typed::decode(source[block]), destination[block]);
    }

    /** @brief Add retained Q8 delta to complete Q16 residual blocks in one native tile.
     * @param residual Sole mutable Q16 block bank.
     * @param delta Original Q8 delta blocks.
     * @param count Full-block element count.
     */
    inline void q16_1_add_q8_1_neon(Q16_1Block *residual, const Q8_1Block *delta, std::size_t count)
    {
        neon_typed::requireBlocks(count);
        for (std::size_t block = 0; block < count / 32; ++block)
            neon_typed::quantize16(neon_typed::add(neon_typed::decode(residual[block]), neon_typed::decode(delta[block])), residual[block]);
    }

    /** @brief Decode Q8 blocks directly into the live FP32 output extent.
     * @param source Retained Q8 blocks including a complete final physical block.
     * @param output Exactly count writable logical floats.
     * @param count Logical extent with bounded final-block/vector tails.
     */
    LLAMINAR_NEON_TYPED_EXACT inline void dequantize_q8_1_to_fp32_neon(const Q8_1Block *source, float *output, std::size_t count)
    {
        LLAMINAR_NEON_TYPED_FP
        for (std::size_t first = 0; first < count; first += 32)
        {
            const auto &block = source[first / 32];
            const float scale = fp16_to_fp32(block.d);
            const auto live = count - first < 32 ? count - first : 32;
            std::size_t i = 0;
            for (; i + 4 <= live; i += 4) vst1q_f32(output + first + i, vmulq_n_f32(neon_typed::load4(block.qs + i), scale));
            for (; i < live; ++i) output[first + i] = scale * static_cast<float>(block.qs[i]);
        }
    }

    /** @brief Quantize full FP32 blocks with the existing tensor scale clamp and ties-away program.
     * @param source Read-only finite FP32 values.
     * @param destination Exclusive Q8 block bank.
     * @param count Complete-block element count; partial counts are rejected.
     */
    inline void quantize_fp32_to_q8_1_blocks_neon(const float *source, Q8_1Block *destination, std::size_t count)
    {
        neon_typed::requireBlocks(count);
        for (std::size_t block = 0; block < count / 32; ++block)
            neon_typed::quantize8<neon_typed::Q8Program::Tensor>(neon_typed::load(source + block * 32), destination[block]);
    }
} // namespace llaminar2::simd
#undef LLAMINAR_NEON_TYPED_FP
#undef LLAMINAR_NEON_TYPED_EXACT
