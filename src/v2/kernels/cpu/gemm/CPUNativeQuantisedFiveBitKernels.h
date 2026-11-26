/**
 * @file CPUNativeQuantisedFiveBitKernels.h
 * @brief Byte-exact compact five-bit decode kernels (Q5_0, Q5_1, Q5_K) for CPU NativeQuantised.
 *
 * CompactFiveBit retains the native 20-byte five-bit payload for all three
 * five-bit source formats (Q5_0 codebook 6, Q5_1 and Q5_K codebook 7) without
 * expanding weights to signed INT8:
 * - Bytes [0..1023]: 4 groups × 4 ZMMs × 64 bytes of native `qs[0..15]`
 *   (low nibbles = K0..15, high nibbles = K16..31).
 * - Bytes [1024..1279]: 8 VNNI groups × 4 ZMMs × 8-byte `uint64_t` bitmasks
 *   for bit 4 of each weight.
 * - Bytes [1280..1407]: 64 × FP16 scales (`chunkScales`).
 * - Bytes [1408..1535]: 64 × FP16 minima (`chunkMins`, asymmetric Q5_1/Q5_K only).
 *
 * Total unit stride is 1,408 B (Q5_0, −38.9% vs 2,304 B ExpandedInt8) or
 * 1,536 B (Q5_1/Q5_K, −36.8% vs 2,432 B ExpandedInt8).
 *
 * VNNI operands are ordered with unsigned five-bit weights `u ∈ [0, 31]` as
 * the unsigned operand and raw signed Q8_1 activations `a ∈ [-128, 127]` as
 * the signed operand:
 * - Asymmetric (Q5_1/Q5_K): `dot(u, a)` feeds directly into
 *   `accumulateCorrectedSingleScaleBlock*`, eliminating +128 activation shift
 *   and per-column compensation metadata.
 * - Symmetric (Q5_0): `dot(u - 16, a) = dot(u, a) - 16 * a.sum_qs`, applied
 *   once per K block in INT32 before `accumulateSingleScaleBlock*`.
 * - On AVX2 without VNNI, `|u * a| <= 31 * 128 = 3,968`, so pair sums in
 *   `_mm256_maddubs_epi16` never exceed `7,936 < 32,767` (zero saturation).
 *
 * Serial M=1, ordered K-partition trees, and grouped M=2/3/4 tiles preserve
 * exact serial-row arithmetic order and byte equivalence with CUDA/ROCm.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include <immintrin.h>

#include "CPUNativeQuantisedContributionContract.h"
#include "CPUNativeQuantisedFP16.h"
#include "CPUNativeQuantisedOutputVector.h"
#include "CPUNativeQuantisedWeightPacker.h"

namespace llaminar2::cpu::native_quantised
{
    namespace fivebit_detail
    {
        /** @brief Pack four signed Q8_1 activation bytes without altering their bits. */
        inline int32_t packSignedActivationWord(
            const Q8_1Block &block, int base_index) noexcept
        {
            int32_t packed_word = 0;
            std::memcpy(&packed_word, block.qs + base_index, sizeof(packed_word));
            return packed_word;
        }

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        /**
         * @brief Decode one 64-byte five-bit weight vector from nibbles + 64-bit high mask.
         *
         * Low nibbles lie in `[0, 15]`, so masked addition of `0x10` is exact
         * bitwise OR into `[0, 31]` in one instruction and zero extra ZMMs.
         */
        [[gnu::always_inline]] inline __m512i mergeFiveBitMaskAVX512(
            __m512i nibbles,
            const uint8_t *mask_ptr,
            __m512i bit16) noexcept
        {
            uint64_t mask = 0;
            std::memcpy(&mask, mask_ptr, sizeof(mask));
            return _mm512_mask_add_epi8(
                nibbles, static_cast<__mmask64>(mask), nibbles, bit16);
        }

        /**
         * @brief Apply one K-block's CompactFiveBit scaling/centering contribution.
         * @tparam Asymmetric True for Q5_1/Q5_K (unsigned [0,31] + min), false for Q5_0 ([-16,15]).
         * @tparam Policy BackendNative or GPUAlignedExpert FP32 contract.
         */
        template <bool Asymmetric, CPUProjectionNumericalPolicy Policy>
        [[gnu::always_inline]] inline __m512 applyFiveBitBlockAVX512(
            __m512 fp_acc,
            __m512i dot,
            const Q8_1Block &activation,
            float activation_scale,
            __m512 weight_scale,
            __m512 weight_min) noexcept
        {
            if constexpr (Asymmetric)
            {
                return accumulateCorrectedSingleScaleBlockAVX512(
                    fp_acc, dot, weight_scale, activation.sum_qs,
                    weight_min, activation_scale, Policy);
            }
            else
            {
                (void)weight_min;
                const __m512i centered = _mm512_sub_epi32(
                    dot, _mm512_set1_epi32(static_cast<int32_t>(activation.sum_qs) * 16));
                return accumulateSingleScaleBlockAVX512(
                    fp_acc, centered, weight_scale, activation_scale, Policy);
            }
        }

        /**
         * @brief Accumulate one 64-column CompactFiveBit M=1 chunk over `[kb_start, kb_end)`.
         *
         * Keeps all four FP32 accumulators and four INT32 accumulators in ZMM
         * registers (peak ~14 ZMMs, zero spills).
         */
        template <bool Asymmetric, CPUProjectionNumericalPolicy Policy>
        [[gnu::always_inline]] inline void accumulateOneRowRangeAVX512(
            const CPUNativeQuantisedPackedWeights &packed,
            const Q8_1Block *A_q8,
            int chunk,
            int kb_start,
            int kb_end,
            __m512 &fp0,
            __m512 &fp1,
            __m512 &fp2,
            __m512 &fp3) noexcept
        {
            const __m512i mask_0F = _mm512_set1_epi8(0x0F);
            const __m512i bit16 = _mm512_set1_epi8(0x10);
            const uint8_t *const base = packed.interleavedBase();
            const size_t stride = static_cast<size_t>(packed.interleaved_block_stride);
            const size_t chunk_origin = static_cast<size_t>(chunk) * packed.blocks_per_row;

            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                if (kb + 2 < packed.blocks_per_row)
                    packed.prefetchInterleavedUnitT0(chunk, kb + 2);
                else if (chunk + 1 < packed.N_padded / 64 && kb + 2 - packed.blocks_per_row < packed.blocks_per_row)
                    packed.prefetchInterleavedUnitT0(chunk + 1, kb + 2 - packed.blocks_per_row);
                const Q8_1Block &a_blk = A_q8[kb];
                const float a_scale = nativeQuantisedFP16ScaleToFP32(a_blk.d);
                const uint8_t *const unit = base + (chunk_origin + kb) * stride;

                __m512i d0 = _mm512_setzero_si512();
                __m512i d1 = _mm512_setzero_si512();
                __m512i d2 = _mm512_setzero_si512();
                __m512i d3 = _mm512_setzero_si512();

#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
                for (int group = 0; group < 4; ++group)
                {
                    const __m512i a_lo = _mm512_set1_epi32(
                        packSignedActivationWord(a_blk, group * 4));
                    const __m512i a_hi = _mm512_set1_epi32(
                        packSignedActivationWord(a_blk, group * 4 + 16));
                    const uint8_t *const low_grp = unit + static_cast<size_t>(group) * 256u;
                    const uint8_t *const hi_lo_grp = unit + 1024u + static_cast<size_t>(group) * 32u;
                    const uint8_t *const hi_hi_grp = unit + 1024u + static_cast<size_t>(group + 4) * 32u;

#define LLAMINAR_FIVEBIT_M1_Z(Z, DOT)                                                       \
                    {                                                                       \
                        const __m512i raw = _mm512_load_si512(low_grp + (Z) * 64u);         \
                        const __m512i w_lo = mergeFiveBitMaskAVX512(                        \
                            _mm512_and_si512(raw, mask_0F), hi_lo_grp + (Z) * 8u, bit16);   \
                        DOT = _mm512_dpbusd_epi32(DOT, w_lo, a_lo);                         \
                        const __m512i w_hi = mergeFiveBitMaskAVX512(                        \
                            _mm512_and_si512(_mm512_srli_epi16(raw, 4), mask_0F),           \
                            hi_hi_grp + (Z) * 8u, bit16);                                   \
                        DOT = _mm512_dpbusd_epi32(DOT, w_hi, a_hi);                         \
                    }
                    LLAMINAR_FIVEBIT_M1_Z(0, d0)
                    LLAMINAR_FIVEBIT_M1_Z(1, d1)
                    LLAMINAR_FIVEBIT_M1_Z(2, d2)
                    LLAMINAR_FIVEBIT_M1_Z(3, d3)
#undef LLAMINAR_FIVEBIT_M1_Z
                }

                const uint16_t *const scales = reinterpret_cast<const uint16_t *>(unit + 1280u);
                const __m512 s0 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scales)));
                const __m512 s1 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scales + 16)));
                const __m512 s2 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scales + 32)));
                const __m512 s3 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scales + 48)));

                if constexpr (Asymmetric)
                {
                    const uint16_t *const mins = reinterpret_cast<const uint16_t *>(unit + 1408u);
                    const __m512 m0 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(mins)));
                    const __m512 m1 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(mins + 16)));
                    const __m512 m2 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(mins + 32)));
                    const __m512 m3 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(mins + 48)));
                    fp0 = applyFiveBitBlockAVX512<true, Policy>(fp0, d0, a_blk, a_scale, s0, m0);
                    fp1 = applyFiveBitBlockAVX512<true, Policy>(fp1, d1, a_blk, a_scale, s1, m1);
                    fp2 = applyFiveBitBlockAVX512<true, Policy>(fp2, d2, a_blk, a_scale, s2, m2);
                    fp3 = applyFiveBitBlockAVX512<true, Policy>(fp3, d3, a_blk, a_scale, s3, m3);
                }
                else
                {
                    const __m512 zero = _mm512_setzero_ps();
                    fp0 = applyFiveBitBlockAVX512<false, Policy>(fp0, d0, a_blk, a_scale, s0, zero);
                    fp1 = applyFiveBitBlockAVX512<false, Policy>(fp1, d1, a_blk, a_scale, s1, zero);
                    fp2 = applyFiveBitBlockAVX512<false, Policy>(fp2, d2, a_blk, a_scale, s2, zero);
                    fp3 = applyFiveBitBlockAVX512<false, Policy>(fp3, d3, a_blk, a_scale, s3, zero);
                }
            }
        }

        /** @brief Register-resident ordered K-partition M=1 CompactFiveBit AVX-512 kernel. */
        template <bool Asymmetric, CPUProjectionNumericalPolicy Policy, bool Bounded>
        inline void oneRowOrderedAVX512(
            const CPUNativeQuantisedPackedWeights &packed,
            const Q8_1Block *row,
            float *output,
            int chunk,
            int K_blocks,
            int k_tiles,
            int k_blocks_per_tile,
            int columns) noexcept
        {
            __m512 r0 = _mm512_setzero_ps(), r1 = _mm512_setzero_ps();
            __m512 r2 = _mm512_setzero_ps(), r3 = _mm512_setzero_ps();
            for (int k_tile = 0; k_tile < k_tiles; ++k_tile)
            {
                const int kb_start = k_tile * k_blocks_per_tile;
                const int kb_end = std::min(kb_start + k_blocks_per_tile, K_blocks);
                __m512 p0 = _mm512_setzero_ps(), p1 = _mm512_setzero_ps();
                __m512 p2 = _mm512_setzero_ps(), p3 = _mm512_setzero_ps();
                accumulateOneRowRangeAVX512<Asymmetric, Policy>(
                    packed, row, chunk, kb_start, kb_end, p0, p1, p2, p3);
                if (k_tile == 0)
                {
                    r0 = p0; r1 = p1; r2 = p2; r3 = p3;
                }
                else
                {
                    r0 = addOrderedAVX512(r0, p0);
                    r1 = addOrderedAVX512(r1, p1);
                    r2 = addOrderedAVX512(r2, p2);
                    r3 = addOrderedAVX512(r3, p3);
                }
            }
            publishProjectionOutputVector<Bounded, 16>(output, columns, 0, r0);
            publishProjectionOutputVector<Bounded, 16>(output, columns, 16, r1);
            publishProjectionOutputVector<Bounded, 16>(output, columns, 32, r2);
            publishProjectionOutputVector<Bounded, 16>(output, columns, 48, r3);
        }

        /**
         * @brief Accumulate two rows across two 16-column vectors (`zbase`, `zbase+1`) over `[kb_start, kb_end)`.
         *
         * Keeps 4 FP32 + 4 INT32 + 4 A + 2 consts + 2 temps = 16 ZMMs (zero spills).
         */
        template <bool Asymmetric, CPUProjectionNumericalPolicy Policy>
        [[gnu::always_inline]] inline void accumulateTwoRowsTwoZRangeAVX512(
            const CPUNativeQuantisedPackedWeights &packed,
            const Q8_1Block *A0,
            const Q8_1Block *A1,
            int chunk,
            int zbase,
            int kb_start,
            int kb_end,
            __m512 &fp00,
            __m512 &fp01,
            __m512 &fp10,
            __m512 &fp11) noexcept
        {
            const __m512i mask_0F = _mm512_set1_epi8(0x0F);
            const __m512i bit16 = _mm512_set1_epi8(0x10);
            const uint8_t *const base = packed.interleavedBase();
            const size_t stride = static_cast<size_t>(packed.interleaved_block_stride);
            const size_t chunk_origin = static_cast<size_t>(chunk) * packed.blocks_per_row;

            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                if (kb + 1 < packed.blocks_per_row)
                    packed.prefetchInterleavedUnitT0(chunk, kb + 1);
                const Q8_1Block &a0 = A0[kb];
                const Q8_1Block &a1 = A1[kb];
                const float a0_scale = nativeQuantisedFP16ScaleToFP32(a0.d);
                const float a1_scale = nativeQuantisedFP16ScaleToFP32(a1.d);
                const uint8_t *const unit = base + (chunk_origin + kb) * stride;

                __m512i d00 = _mm512_setzero_si512(), d01 = _mm512_setzero_si512();
                __m512i d10 = _mm512_setzero_si512(), d11 = _mm512_setzero_si512();

#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
                for (int group = 0; group < 4; ++group)
                {
                    const int32_t a0_lo_w = packSignedActivationWord(a0, group * 4);
                    const int32_t a1_lo_w = packSignedActivationWord(a1, group * 4);
                    const int32_t a0_hi_w = packSignedActivationWord(a0, group * 4 + 16);
                    const int32_t a1_hi_w = packSignedActivationWord(a1, group * 4 + 16);
                    const uint8_t *const low_grp = unit + static_cast<size_t>(group) * 256u;
                    const uint8_t *const hi_lo_grp = unit + 1024u + static_cast<size_t>(group) * 32u;
                    const uint8_t *const hi_hi_grp = unit + 1024u + static_cast<size_t>(group + 4) * 32u;

                    const __m512i raw0 = _mm512_load_si512(low_grp + static_cast<size_t>(zbase) * 64u);
                    const __m512i w0_lo = mergeFiveBitMaskAVX512(
                        _mm512_and_si512(raw0, mask_0F), hi_lo_grp + static_cast<size_t>(zbase) * 8u, bit16);
                    d00 = _mm512_dpbusd_epi32(d00, w0_lo, _mm512_set1_epi32(a0_lo_w));
                    d10 = _mm512_dpbusd_epi32(d10, w0_lo, _mm512_set1_epi32(a1_lo_w));
                    const __m512i w0_hi = mergeFiveBitMaskAVX512(
                        _mm512_and_si512(_mm512_srli_epi16(raw0, 4), mask_0F),
                        hi_hi_grp + static_cast<size_t>(zbase) * 8u, bit16);
                    d00 = _mm512_dpbusd_epi32(d00, w0_hi, _mm512_set1_epi32(a0_hi_w));
                    d10 = _mm512_dpbusd_epi32(d10, w0_hi, _mm512_set1_epi32(a1_hi_w));

                    const __m512i raw1 = _mm512_load_si512(low_grp + static_cast<size_t>(zbase + 1) * 64u);
                    const __m512i w1_lo = mergeFiveBitMaskAVX512(
                        _mm512_and_si512(raw1, mask_0F), hi_lo_grp + static_cast<size_t>(zbase + 1) * 8u, bit16);
                    d01 = _mm512_dpbusd_epi32(d01, w1_lo, _mm512_set1_epi32(a0_lo_w));
                    d11 = _mm512_dpbusd_epi32(d11, w1_lo, _mm512_set1_epi32(a1_lo_w));
                    const __m512i w1_hi = mergeFiveBitMaskAVX512(
                        _mm512_and_si512(_mm512_srli_epi16(raw1, 4), mask_0F),
                        hi_hi_grp + static_cast<size_t>(zbase + 1) * 8u, bit16);
                    d01 = _mm512_dpbusd_epi32(d01, w1_hi, _mm512_set1_epi32(a0_hi_w));
                    d11 = _mm512_dpbusd_epi32(d11, w1_hi, _mm512_set1_epi32(a1_hi_w));
                }

                const uint16_t *const scales = reinterpret_cast<const uint16_t *>(unit + 1280u) + zbase * 16;
                const __m512 s0 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scales)));
                const __m512 s1 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scales + 16)));
                if constexpr (Asymmetric)
                {
                    const uint16_t *const mins = reinterpret_cast<const uint16_t *>(unit + 1408u) + zbase * 16;
                    const __m512 m0 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(mins)));
                    const __m512 m1 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(mins + 16)));
                    fp00 = applyFiveBitBlockAVX512<true, Policy>(fp00, d00, a0, a0_scale, s0, m0);
                    fp01 = applyFiveBitBlockAVX512<true, Policy>(fp01, d01, a0, a0_scale, s1, m1);
                    fp10 = applyFiveBitBlockAVX512<true, Policy>(fp10, d10, a1, a1_scale, s0, m0);
                    fp11 = applyFiveBitBlockAVX512<true, Policy>(fp11, d11, a1, a1_scale, s1, m1);
                }
                else
                {
                    const __m512 zero = _mm512_setzero_ps();
                    fp00 = applyFiveBitBlockAVX512<false, Policy>(fp00, d00, a0, a0_scale, s0, zero);
                    fp01 = applyFiveBitBlockAVX512<false, Policy>(fp01, d01, a0, a0_scale, s1, zero);
                    fp10 = applyFiveBitBlockAVX512<false, Policy>(fp10, d10, a1, a1_scale, s0, zero);
                    fp11 = applyFiveBitBlockAVX512<false, Policy>(fp11, d11, a1, a1_scale, s1, zero);
                }
            }
        }

        /** @brief Register-resident ordered K-partition M=2 CompactFiveBit AVX-512 kernel. */
        template <bool Asymmetric, CPUProjectionNumericalPolicy Policy, bool Bounded>
        inline void twoRowsOrderedAVX512(
            const CPUNativeQuantisedPackedWeights &packed,
            const Q8_1Block *row0,
            const Q8_1Block *row1,
            float *output0,
            float *output1,
            int chunk,
            int K_blocks,
            int k_tiles,
            int k_blocks_per_tile,
            int columns) noexcept
        {
            for (int zbase = 0; zbase < (Bounded ? (columns + 15) / 16 : 4); zbase += 2)
            {
                __m512 r00 = _mm512_setzero_ps(), r01 = _mm512_setzero_ps();
                __m512 r10 = _mm512_setzero_ps(), r11 = _mm512_setzero_ps();
                for (int k_tile = 0; k_tile < k_tiles; ++k_tile)
                {
                    const int kb_start = k_tile * k_blocks_per_tile;
                    const int kb_end = std::min(kb_start + k_blocks_per_tile, K_blocks);
                    __m512 p00 = _mm512_setzero_ps(), p01 = _mm512_setzero_ps();
                    __m512 p10 = _mm512_setzero_ps(), p11 = _mm512_setzero_ps();
                    accumulateTwoRowsTwoZRangeAVX512<Asymmetric, Policy>(
                        packed, row0, row1, chunk, zbase, kb_start, kb_end, p00, p01, p10, p11);
                    if (k_tile == 0)
                    {
                        r00 = p00; r01 = p01; r10 = p10; r11 = p11;
                    }
                    else
                    {
                        r00 = addOrderedAVX512(r00, p00);
                        r01 = addOrderedAVX512(r01, p01);
                        r10 = addOrderedAVX512(r10, p10);
                        r11 = addOrderedAVX512(r11, p11);
                    }
                }
                publishProjectionOutputVector<Bounded, 16>(output0, columns, zbase * 16, r00);
                publishProjectionOutputVector<Bounded, 16>(output0, columns, (zbase + 1) * 16, r01);
                publishProjectionOutputVector<Bounded, 16>(output1, columns, zbase * 16, r10);
                publishProjectionOutputVector<Bounded, 16>(output1, columns, (zbase + 1) * 16, r11);
            }
        }

        /** @brief Three-row 2z CompactFiveBit AVX-512 kernel (peak ~20 ZMMs, zero spills). */
        template <bool Asymmetric, CPUProjectionNumericalPolicy Policy, bool Bounded>
        inline void threeRowsChunkAVX512(
            const CPUNativeQuantisedPackedWeights &packed,
            const Q8_1Block *A0,
            const Q8_1Block *A1,
            const Q8_1Block *A2,
            float *C0,
            float *C1,
            float *C2,
            int chunk,
            int kb_start,
            int kb_end,
            bool accumulate,
            int columns) noexcept
        {
            const __m512i mask_0F = _mm512_set1_epi8(0x0F);
            const __m512i bit16 = _mm512_set1_epi8(0x10);
            const uint8_t *const base = packed.interleavedBase();
            const size_t stride = static_cast<size_t>(packed.interleaved_block_stride);
            const size_t chunk_origin = static_cast<size_t>(chunk) * packed.blocks_per_row;

            for (int zbase = 0; zbase < (Bounded ? (columns + 15) / 16 : 4); zbase += 2)
            {
                __m512 fp00 = readProjectionOutputVector<Bounded, 16, __m512>(C0, columns, zbase * 16, accumulate);
                __m512 fp01 = readProjectionOutputVector<Bounded, 16, __m512>(C0, columns, (zbase + 1) * 16, accumulate);
                __m512 fp10 = readProjectionOutputVector<Bounded, 16, __m512>(C1, columns, zbase * 16, accumulate);
                __m512 fp11 = readProjectionOutputVector<Bounded, 16, __m512>(C1, columns, (zbase + 1) * 16, accumulate);
                __m512 fp20 = readProjectionOutputVector<Bounded, 16, __m512>(C2, columns, zbase * 16, accumulate);
                __m512 fp21 = readProjectionOutputVector<Bounded, 16, __m512>(C2, columns, (zbase + 1) * 16, accumulate);

                for (int kb = kb_start; kb < kb_end; ++kb)
                {
                    if (kb + 1 < packed.blocks_per_row)
                        packed.prefetchInterleavedUnitT0(chunk, kb + 1);
                    const Q8_1Block &a0 = A0[kb];
                    const Q8_1Block &a1 = A1[kb];
                    const Q8_1Block &a2 = A2[kb];
                    const float a0_scale = nativeQuantisedFP16ScaleToFP32(a0.d);
                    const float a1_scale = nativeQuantisedFP16ScaleToFP32(a1.d);
                    const float a2_scale = nativeQuantisedFP16ScaleToFP32(a2.d);
                    const uint8_t *const unit = base + (chunk_origin + kb) * stride;

                    __m512i d00 = _mm512_setzero_si512(), d01 = _mm512_setzero_si512();
                    __m512i d10 = _mm512_setzero_si512(), d11 = _mm512_setzero_si512();
                    __m512i d20 = _mm512_setzero_si512(), d21 = _mm512_setzero_si512();

                    for (int group = 0; group < 4; ++group)
                    {
                        const int32_t w0_lo_a = packSignedActivationWord(a0, group * 4);
                        const int32_t w1_lo_a = packSignedActivationWord(a1, group * 4);
                        const int32_t w2_lo_a = packSignedActivationWord(a2, group * 4);
                        const int32_t w0_hi_a = packSignedActivationWord(a0, group * 4 + 16);
                        const int32_t w1_hi_a = packSignedActivationWord(a1, group * 4 + 16);
                        const int32_t w2_hi_a = packSignedActivationWord(a2, group * 4 + 16);
                        const uint8_t *const low_grp = unit + static_cast<size_t>(group) * 256u;
                        const uint8_t *const hi_lo_grp = unit + 1024u + static_cast<size_t>(group) * 32u;
                        const uint8_t *const hi_hi_grp = unit + 1024u + static_cast<size_t>(group + 4) * 32u;

                        const __m512i raw0 = _mm512_load_si512(low_grp + static_cast<size_t>(zbase) * 64u);
                        const __m512i b0_lo = mergeFiveBitMaskAVX512(
                            _mm512_and_si512(raw0, mask_0F), hi_lo_grp + static_cast<size_t>(zbase) * 8u, bit16);
                        d00 = _mm512_dpbusd_epi32(d00, b0_lo, _mm512_set1_epi32(w0_lo_a));
                        d10 = _mm512_dpbusd_epi32(d10, b0_lo, _mm512_set1_epi32(w1_lo_a));
                        d20 = _mm512_dpbusd_epi32(d20, b0_lo, _mm512_set1_epi32(w2_lo_a));
                        const __m512i b0_hi = mergeFiveBitMaskAVX512(
                            _mm512_and_si512(_mm512_srli_epi16(raw0, 4), mask_0F),
                            hi_hi_grp + static_cast<size_t>(zbase) * 8u, bit16);
                        d00 = _mm512_dpbusd_epi32(d00, b0_hi, _mm512_set1_epi32(w0_hi_a));
                        d10 = _mm512_dpbusd_epi32(d10, b0_hi, _mm512_set1_epi32(w1_hi_a));
                        d20 = _mm512_dpbusd_epi32(d20, b0_hi, _mm512_set1_epi32(w2_hi_a));

                        const __m512i raw1 = _mm512_load_si512(low_grp + static_cast<size_t>(zbase + 1) * 64u);
                        const __m512i b1_lo = mergeFiveBitMaskAVX512(
                            _mm512_and_si512(raw1, mask_0F), hi_lo_grp + static_cast<size_t>(zbase + 1) * 8u, bit16);
                        d01 = _mm512_dpbusd_epi32(d01, b1_lo, _mm512_set1_epi32(w0_lo_a));
                        d11 = _mm512_dpbusd_epi32(d11, b1_lo, _mm512_set1_epi32(w1_lo_a));
                        d21 = _mm512_dpbusd_epi32(d21, b1_lo, _mm512_set1_epi32(w2_lo_a));
                        const __m512i b1_hi = mergeFiveBitMaskAVX512(
                            _mm512_and_si512(_mm512_srli_epi16(raw1, 4), mask_0F),
                            hi_hi_grp + static_cast<size_t>(zbase + 1) * 8u, bit16);
                        d01 = _mm512_dpbusd_epi32(d01, b1_hi, _mm512_set1_epi32(w0_hi_a));
                        d11 = _mm512_dpbusd_epi32(d11, b1_hi, _mm512_set1_epi32(w1_hi_a));
                        d21 = _mm512_dpbusd_epi32(d21, b1_hi, _mm512_set1_epi32(w2_hi_a));
                    }

                    const uint16_t *const scales = reinterpret_cast<const uint16_t *>(unit + 1280u) + zbase * 16;
                    const __m512 s0 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scales)));
                    const __m512 s1 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scales + 16)));
                    if constexpr (Asymmetric)
                    {
                        const uint16_t *const mins = reinterpret_cast<const uint16_t *>(unit + 1408u) + zbase * 16;
                        const __m512 m0 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(mins)));
                        const __m512 m1 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(mins + 16)));
                        fp00 = applyFiveBitBlockAVX512<true, Policy>(fp00, d00, a0, a0_scale, s0, m0);
                        fp01 = applyFiveBitBlockAVX512<true, Policy>(fp01, d01, a0, a0_scale, s1, m1);
                        fp10 = applyFiveBitBlockAVX512<true, Policy>(fp10, d10, a1, a1_scale, s0, m0);
                        fp11 = applyFiveBitBlockAVX512<true, Policy>(fp11, d11, a1, a1_scale, s1, m1);
                        fp20 = applyFiveBitBlockAVX512<true, Policy>(fp20, d20, a2, a2_scale, s0, m0);
                        fp21 = applyFiveBitBlockAVX512<true, Policy>(fp21, d21, a2, a2_scale, s1, m1);
                    }
                    else
                    {
                        const __m512 zero = _mm512_setzero_ps();
                        fp00 = applyFiveBitBlockAVX512<false, Policy>(fp00, d00, a0, a0_scale, s0, zero);
                        fp01 = applyFiveBitBlockAVX512<false, Policy>(fp01, d01, a0, a0_scale, s1, zero);
                        fp10 = applyFiveBitBlockAVX512<false, Policy>(fp10, d10, a1, a1_scale, s0, zero);
                        fp11 = applyFiveBitBlockAVX512<false, Policy>(fp11, d11, a1, a1_scale, s1, zero);
                        fp20 = applyFiveBitBlockAVX512<false, Policy>(fp20, d20, a2, a2_scale, s0, zero);
                        fp21 = applyFiveBitBlockAVX512<false, Policy>(fp21, d21, a2, a2_scale, s1, zero);
                    }
                }

                publishProjectionOutputVector<Bounded, 16>(C0, columns, zbase * 16, fp00);
                publishProjectionOutputVector<Bounded, 16>(C0, columns, (zbase + 1) * 16, fp01);
                publishProjectionOutputVector<Bounded, 16>(C1, columns, zbase * 16, fp10);
                publishProjectionOutputVector<Bounded, 16>(C1, columns, (zbase + 1) * 16, fp11);
                publishProjectionOutputVector<Bounded, 16>(C2, columns, zbase * 16, fp20);
                publishProjectionOutputVector<Bounded, 16>(C2, columns, (zbase + 1) * 16, fp21);
            }
        }

        /** @brief Four-row 1z CompactFiveBit AVX-512 kernel (peak ~14 ZMMs, zero spills). */
        template <bool Asymmetric, CPUProjectionNumericalPolicy Policy, bool SumPairs, bool Bounded>
        inline void fourRowsChunkAVX512(
            const CPUNativeQuantisedPackedWeights &packed,
            const Q8_1Block *A0,
            const Q8_1Block *A1,
            const Q8_1Block *A2,
            const Q8_1Block *A3,
            float *C0,
            float *C1,
            float *C2,
            float *C3,
            int chunk,
            int kb_start,
            int kb_end,
            bool accumulate,
            int columns) noexcept
        {
            const __m512i mask_0F = _mm512_set1_epi8(0x0F);
            const __m512i bit16 = _mm512_set1_epi8(0x10);
            const uint8_t *const base = packed.interleavedBase();
            const size_t stride = static_cast<size_t>(packed.interleaved_block_stride);
            const size_t chunk_origin = static_cast<size_t>(chunk) * packed.blocks_per_row;

            for (int z = 0; z < (Bounded ? (columns + 15) / 16 : 4); ++z)
            {
                __m512 fp0 = readProjectionOutputVector<Bounded, 16, __m512>(C0, columns, z * 16, accumulate);
                __m512 fp1 = readProjectionOutputVector<Bounded, 16, __m512>(C1, columns, z * 16, accumulate);
                __m512 fp2 = readProjectionOutputVector<Bounded, 16, __m512>(C2, columns, z * 16, accumulate);
                __m512 fp3 = readProjectionOutputVector<Bounded, 16, __m512>(C3, columns, z * 16, accumulate);

                for (int kb = kb_start; kb < kb_end; ++kb)
                {
                    if (kb + 1 < packed.blocks_per_row)
                        packed.prefetchInterleavedUnitT0(chunk, kb + 1);
                    const Q8_1Block &a0 = A0[kb];
                    const Q8_1Block &a1 = A1[kb];
                    const Q8_1Block &a2 = A2[kb];
                    const Q8_1Block &a3 = A3[kb];
                    const float a0_scale = nativeQuantisedFP16ScaleToFP32(a0.d);
                    const float a1_scale = nativeQuantisedFP16ScaleToFP32(a1.d);
                    const float a2_scale = nativeQuantisedFP16ScaleToFP32(a2.d);
                    const float a3_scale = nativeQuantisedFP16ScaleToFP32(a3.d);
                    const uint8_t *const unit = base + (chunk_origin + kb) * stride;

                    __m512i d0 = _mm512_setzero_si512();
                    __m512i d1 = _mm512_setzero_si512();
                    __m512i d2 = _mm512_setzero_si512();
                    __m512i d3 = _mm512_setzero_si512();

                    for (int group = 0; group < 4; ++group)
                    {
                        const __m512i raw = _mm512_load_si512(
                            unit + static_cast<size_t>(group) * 256u + static_cast<size_t>(z) * 64u);
                        const __m512i b_lo = mergeFiveBitMaskAVX512(
                            _mm512_and_si512(raw, mask_0F),
                            unit + 1024u + static_cast<size_t>(group) * 32u + static_cast<size_t>(z) * 8u,
                            bit16);
                        d0 = _mm512_dpbusd_epi32(d0, b_lo, _mm512_set1_epi32(packSignedActivationWord(a0, group * 4)));
                        d1 = _mm512_dpbusd_epi32(d1, b_lo, _mm512_set1_epi32(packSignedActivationWord(a1, group * 4)));
                        d2 = _mm512_dpbusd_epi32(d2, b_lo, _mm512_set1_epi32(packSignedActivationWord(a2, group * 4)));
                        d3 = _mm512_dpbusd_epi32(d3, b_lo, _mm512_set1_epi32(packSignedActivationWord(a3, group * 4)));

                        const __m512i b_hi = mergeFiveBitMaskAVX512(
                            _mm512_and_si512(_mm512_srli_epi16(raw, 4), mask_0F),
                            unit + 1024u + static_cast<size_t>(group + 4) * 32u + static_cast<size_t>(z) * 8u,
                            bit16);
                        d0 = _mm512_dpbusd_epi32(d0, b_hi, _mm512_set1_epi32(packSignedActivationWord(a0, group * 4 + 16)));
                        d1 = _mm512_dpbusd_epi32(d1, b_hi, _mm512_set1_epi32(packSignedActivationWord(a1, group * 4 + 16)));
                        d2 = _mm512_dpbusd_epi32(d2, b_hi, _mm512_set1_epi32(packSignedActivationWord(a2, group * 4 + 16)));
                        d3 = _mm512_dpbusd_epi32(d3, b_hi, _mm512_set1_epi32(packSignedActivationWord(a3, group * 4 + 16)));
                    }

                    const uint16_t *const scales = reinterpret_cast<const uint16_t *>(unit + 1280u) + z * 16;
                    const __m512 sc = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scales)));
                    if constexpr (Asymmetric)
                    {
                        const uint16_t *const mins = reinterpret_cast<const uint16_t *>(unit + 1408u) + z * 16;
                        const __m512 mn = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(mins)));
                        fp0 = applyFiveBitBlockAVX512<true, Policy>(fp0, d0, a0, a0_scale, sc, mn);
                        fp1 = applyFiveBitBlockAVX512<true, Policy>(fp1, d1, a1, a1_scale, sc, mn);
                        fp2 = applyFiveBitBlockAVX512<true, Policy>(fp2, d2, a2, a2_scale, sc, mn);
                        fp3 = applyFiveBitBlockAVX512<true, Policy>(fp3, d3, a3, a3_scale, sc, mn);
                    }
                    else
                    {
                        const __m512 zero = _mm512_setzero_ps();
                        fp0 = applyFiveBitBlockAVX512<false, Policy>(fp0, d0, a0, a0_scale, sc, zero);
                        fp1 = applyFiveBitBlockAVX512<false, Policy>(fp1, d1, a1, a1_scale, sc, zero);
                        fp2 = applyFiveBitBlockAVX512<false, Policy>(fp2, d2, a2, a2_scale, sc, zero);
                        fp3 = applyFiveBitBlockAVX512<false, Policy>(fp3, d3, a3, a3_scale, sc, zero);
                    }
                }

                if constexpr (SumPairs)
                {
                    const int live = packed.N - chunk * 64 - z * 16;
                    const auto store_live = [&](float *dst, __m512 val) {
                        if (live >= 16)
                            _mm512_storeu_ps(dst + z * 16, val);
                        else if (live > 0)
                            _mm512_mask_storeu_ps(dst + z * 16, static_cast<__mmask16>((1u << live) - 1u), val);
                    };
                    store_live(C0, _mm512_add_ps(fp0, fp1));
                    store_live(C2, _mm512_add_ps(fp2, fp3));
                }
                else
                {
                    publishProjectionOutputVector<Bounded, 16>(C0, columns, z * 16, fp0);
                    publishProjectionOutputVector<Bounded, 16>(C1, columns, z * 16, fp1);
                    publishProjectionOutputVector<Bounded, 16>(C2, columns, z * 16, fp2);
                    publishProjectionOutputVector<Bounded, 16>(C3, columns, z * 16, fp3);
                }
            }
        }
#endif // __AVX512F__ && __AVX512VNNI__ && __AVX512BW__

#if defined(__AVX2__)
        /**
         * @brief Expand a 32-bit high-bit mask to 32 bytes of `{0x00, 0x10}` in AVX2.
         */
        inline __m256i expandMask32ToBit16AVX2(
            uint32_t mask,
            __m256i shuf_bytes,
            __m256i bit_positions,
            __m256i bit16) noexcept
        {
            const __m256i v = _mm256_set1_epi32(static_cast<int32_t>(mask));
            const __m256i replicated = _mm256_shuffle_epi8(v, shuf_bytes);
            const __m256i isolated = _mm256_and_si256(replicated, bit_positions);
            const __m256i cmp = _mm256_cmpeq_epi8(isolated, bit_positions);
            return _mm256_and_si256(cmp, bit16);
        }

        /**
         * @brief Exact saturation-free AVX2 unsigned-5-bit × signed-int8 dot product.
         *
         * Because `u ∈ [0, 31]` and `a ∈ [-128, 127]`, every pair sum in
         * `vpmaddubsw` has magnitude at most `2 * 31 * 128 = 7,936 < 32,767`.
         * No INT16 saturation is possible, so two instructions replace the
         * six-instruction saturation-safe VNNI emulation.
         */
        inline __m256i dpbusdFiveBitAVX2(
            __m256i acc, __m256i u_fivebit, __m256i a_signed, __m256i ones16) noexcept
        {
            const __m256i pairs = _mm256_maddubs_epi16(u_fivebit, a_signed);
            return _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, ones16));
        }

        template <bool Asymmetric, CPUProjectionNumericalPolicy Policy>
        inline __m256 applyFiveBitBlockAVX2(
            __m256 fp_acc,
            __m256i dot,
            const Q8_1Block &activation,
            float activation_scale,
            __m256 weight_scale,
            __m256 weight_min) noexcept
        {
            if constexpr (Asymmetric)
            {
                return accumulateCorrectedSingleScaleBlockAVX2(
                    fp_acc, dot, weight_scale, activation.sum_qs,
                    weight_min, activation_scale, Policy);
            }
            else
            {
                (void)weight_min;
                const __m256i centered = _mm256_sub_epi32(
                    dot, _mm256_set1_epi32(static_cast<int32_t>(activation.sum_qs) * 16));
                return accumulateSingleScaleBlockAVX2(
                    fp_acc, centered, weight_scale, activation_scale, Policy);
            }
        }
#endif // __AVX2__
    } // namespace fivebit_detail

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
    /** @brief Single-row CompactFiveBit AVX-512 GEMV chunk (`Q5_0`, `Q5_1`, `Q5_K`). */
    template <bool Bounded = false>
    inline void gemvCompactFiveBitAVX512Chunk(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate = false,
        int columns = 64)
    {
        if (!packed.usesCompactFiveBit())
            throw std::invalid_argument("CompactFiveBit AVX-512 M=1 kernel received another encoding");
        __m512 fp0 = readProjectionOutputVector<Bounded, 16, __m512>(C, columns, 0, accumulate);
        __m512 fp1 = readProjectionOutputVector<Bounded, 16, __m512>(C, columns, 16, accumulate);
        __m512 fp2 = readProjectionOutputVector<Bounded, 16, __m512>(C, columns, 32, accumulate);
        __m512 fp3 = readProjectionOutputVector<Bounded, 16, __m512>(C, columns, 48, accumulate);
        if (packed.is_asymmetric)
        {
            if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                fivebit_detail::accumulateOneRowRangeAVX512<true, CPUProjectionNumericalPolicy::GPUAlignedExpert>(
                    packed, A_q8, chunk, kb_start, kb_end, fp0, fp1, fp2, fp3);
            else
                fivebit_detail::accumulateOneRowRangeAVX512<true, CPUProjectionNumericalPolicy::BackendNative>(
                    packed, A_q8, chunk, kb_start, kb_end, fp0, fp1, fp2, fp3);
        }
        else
        {
            if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                fivebit_detail::accumulateOneRowRangeAVX512<false, CPUProjectionNumericalPolicy::GPUAlignedExpert>(
                    packed, A_q8, chunk, kb_start, kb_end, fp0, fp1, fp2, fp3);
            else
                fivebit_detail::accumulateOneRowRangeAVX512<false, CPUProjectionNumericalPolicy::BackendNative>(
                    packed, A_q8, chunk, kb_start, kb_end, fp0, fp1, fp2, fp3);
        }
        publishProjectionOutputVector<Bounded, 16>(C, columns, 0, fp0);
        publishProjectionOutputVector<Bounded, 16>(C, columns, 16, fp1);
        publishProjectionOutputVector<Bounded, 16>(C, columns, 32, fp2);
        publishProjectionOutputVector<Bounded, 16>(C, columns, 48, fp3);
    }

    /** @brief Register-resident ordered K-partition M=1 CompactFiveBit AVX-512 kernel. */
    template <bool Bounded = false>
    inline void gemvCompactFiveBitAVX512OrderedPartitions(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *row,
        float *output,
        int chunk,
        int K_blocks,
        int k_tiles,
        int k_blocks_per_tile,
        int columns = 64)
    {
        if (!packed.usesCompactFiveBit())
            throw std::invalid_argument("CompactFiveBit AVX-512 ordered M=1 kernel received another encoding");
        if (packed.is_asymmetric)
        {
            if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                return fivebit_detail::oneRowOrderedAVX512<true, CPUProjectionNumericalPolicy::GPUAlignedExpert, Bounded>(
                    packed, row, output, chunk, K_blocks, k_tiles, k_blocks_per_tile, columns);
            return fivebit_detail::oneRowOrderedAVX512<true, CPUProjectionNumericalPolicy::BackendNative, Bounded>(
                packed, row, output, chunk, K_blocks, k_tiles, k_blocks_per_tile, columns);
        }
        if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
            return fivebit_detail::oneRowOrderedAVX512<false, CPUProjectionNumericalPolicy::GPUAlignedExpert, Bounded>(
                packed, row, output, chunk, K_blocks, k_tiles, k_blocks_per_tile, columns);
        return fivebit_detail::oneRowOrderedAVX512<false, CPUProjectionNumericalPolicy::BackendNative, Bounded>(
            packed, row, output, chunk, K_blocks, k_tiles, k_blocks_per_tile, columns);
    }

    /** @brief Two-row CompactFiveBit AVX-512 chunk (`Q5_0`, `Q5_1`, `Q5_K`). */
    template <bool SumPairs = false, bool Bounded = false>
    inline void gemmCompactFiveBitTwoRowsAVX512Chunk(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *A0,
        const Q8_1Block *A1,
        float *C0,
        float *C1,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate,
        int columns = 64)
    {
        if (!packed.usesCompactFiveBit())
            throw std::invalid_argument("CompactFiveBit AVX-512 M=2 kernel received another encoding");
        for (int zbase = 0; zbase < (Bounded ? (columns + 15) / 16 : 4); zbase += 2)
        {
            __m512 fp00 = readProjectionOutputVector<Bounded, 16, __m512>(C0, columns, zbase * 16, accumulate);
            __m512 fp01 = readProjectionOutputVector<Bounded, 16, __m512>(C0, columns, (zbase + 1) * 16, accumulate);
            __m512 fp10 = readProjectionOutputVector<Bounded, 16, __m512>(C1, columns, zbase * 16, accumulate);
            __m512 fp11 = readProjectionOutputVector<Bounded, 16, __m512>(C1, columns, (zbase + 1) * 16, accumulate);
            if (packed.is_asymmetric)
            {
                if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                    fivebit_detail::accumulateTwoRowsTwoZRangeAVX512<true, CPUProjectionNumericalPolicy::GPUAlignedExpert>(
                        packed, A0, A1, chunk, zbase, kb_start, kb_end, fp00, fp01, fp10, fp11);
                else
                    fivebit_detail::accumulateTwoRowsTwoZRangeAVX512<true, CPUProjectionNumericalPolicy::BackendNative>(
                        packed, A0, A1, chunk, zbase, kb_start, kb_end, fp00, fp01, fp10, fp11);
            }
            else
            {
                if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                    fivebit_detail::accumulateTwoRowsTwoZRangeAVX512<false, CPUProjectionNumericalPolicy::GPUAlignedExpert>(
                        packed, A0, A1, chunk, zbase, kb_start, kb_end, fp00, fp01, fp10, fp11);
                else
                    fivebit_detail::accumulateTwoRowsTwoZRangeAVX512<false, CPUProjectionNumericalPolicy::BackendNative>(
                        packed, A0, A1, chunk, zbase, kb_start, kb_end, fp00, fp01, fp10, fp11);
            }
            if constexpr (SumPairs)
            {
                const auto store_live = [&](int z, __m512 val) {
                    const int live = packed.N - chunk * 64 - z * 16;
                    if (live >= 16)
                        _mm512_storeu_ps(C0 + z * 16, val);
                    else if (live > 0)
                        _mm512_mask_storeu_ps(C0 + z * 16, static_cast<__mmask16>((1u << live) - 1u), val);
                };
                store_live(zbase, _mm512_add_ps(fp00, fp10));
                store_live(zbase + 1, _mm512_add_ps(fp01, fp11));
            }
            else
            {
                publishProjectionOutputVector<Bounded, 16>(C0, columns, zbase * 16, fp00);
                publishProjectionOutputVector<Bounded, 16>(C0, columns, (zbase + 1) * 16, fp01);
                publishProjectionOutputVector<Bounded, 16>(C1, columns, zbase * 16, fp10);
                publishProjectionOutputVector<Bounded, 16>(C1, columns, (zbase + 1) * 16, fp11);
            }
        }
    }

    /** @brief Register-resident ordered K-partition M=2 CompactFiveBit AVX-512 kernel. */
    template <bool Bounded = false>
    inline void gemmCompactFiveBitTwoRowsAVX512OrderedPartitions(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *row0,
        const Q8_1Block *row1,
        float *output0,
        float *output1,
        int chunk,
        int K_blocks,
        int k_tiles,
        int k_blocks_per_tile,
        int columns = 64)
    {
        if (!packed.usesCompactFiveBit())
            throw std::invalid_argument("CompactFiveBit AVX-512 ordered M=2 kernel received another encoding");
        if (packed.is_asymmetric)
        {
            if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                return fivebit_detail::twoRowsOrderedAVX512<true, CPUProjectionNumericalPolicy::GPUAlignedExpert, Bounded>(
                    packed, row0, row1, output0, output1, chunk, K_blocks, k_tiles, k_blocks_per_tile, columns);
            return fivebit_detail::twoRowsOrderedAVX512<true, CPUProjectionNumericalPolicy::BackendNative, Bounded>(
                packed, row0, row1, output0, output1, chunk, K_blocks, k_tiles, k_blocks_per_tile, columns);
        }
        if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
            return fivebit_detail::twoRowsOrderedAVX512<false, CPUProjectionNumericalPolicy::GPUAlignedExpert, Bounded>(
                packed, row0, row1, output0, output1, chunk, K_blocks, k_tiles, k_blocks_per_tile, columns);
        return fivebit_detail::twoRowsOrderedAVX512<false, CPUProjectionNumericalPolicy::BackendNative, Bounded>(
            packed, row0, row1, output0, output1, chunk, K_blocks, k_tiles, k_blocks_per_tile, columns);
    }

    /** @brief Three-row CompactFiveBit AVX-512 chunk (`Q5_0`, `Q5_1`, `Q5_K`). */
    template <bool Bounded = false>
    inline void gemmCompactFiveBitThreeRowsAVX512Chunk(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *A0,
        const Q8_1Block *A1,
        const Q8_1Block *A2,
        float *C0,
        float *C1,
        float *C2,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate,
        int columns = 64)
    {
        if (!packed.usesCompactFiveBit())
            throw std::invalid_argument("CompactFiveBit AVX-512 M=3 kernel received another encoding");
        if (packed.is_asymmetric)
        {
            if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                return fivebit_detail::threeRowsChunkAVX512<true, CPUProjectionNumericalPolicy::GPUAlignedExpert, Bounded>(
                    packed, A0, A1, A2, C0, C1, C2, chunk, kb_start, kb_end, accumulate, columns);
            return fivebit_detail::threeRowsChunkAVX512<true, CPUProjectionNumericalPolicy::BackendNative, Bounded>(
                packed, A0, A1, A2, C0, C1, C2, chunk, kb_start, kb_end, accumulate, columns);
        }
        if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
            return fivebit_detail::threeRowsChunkAVX512<false, CPUProjectionNumericalPolicy::GPUAlignedExpert, Bounded>(
                packed, A0, A1, A2, C0, C1, C2, chunk, kb_start, kb_end, accumulate, columns);
        return fivebit_detail::threeRowsChunkAVX512<false, CPUProjectionNumericalPolicy::BackendNative, Bounded>(
            packed, A0, A1, A2, C0, C1, C2, chunk, kb_start, kb_end, accumulate, columns);
    }

    /** @brief Four-row CompactFiveBit AVX-512 chunk (`Q5_0`, `Q5_1`, `Q5_K`). */
    template <bool SumPairs = false, bool Bounded = false>
    inline void gemmCompactFiveBitFourRowsAVX512Chunk(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *A0,
        const Q8_1Block *A1,
        const Q8_1Block *A2,
        const Q8_1Block *A3,
        float *C0,
        float *C1,
        float *C2,
        float *C3,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate,
        int columns = 64)
    {
        if (!packed.usesCompactFiveBit())
            throw std::invalid_argument("CompactFiveBit AVX-512 M=4 kernel received another encoding");
        if (packed.is_asymmetric)
        {
            if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                return fivebit_detail::fourRowsChunkAVX512<true, CPUProjectionNumericalPolicy::GPUAlignedExpert, SumPairs, Bounded>(
                    packed, A0, A1, A2, A3, C0, C1, C2, C3, chunk, kb_start, kb_end, accumulate, columns);
            return fivebit_detail::fourRowsChunkAVX512<true, CPUProjectionNumericalPolicy::BackendNative, SumPairs, Bounded>(
                packed, A0, A1, A2, A3, C0, C1, C2, C3, chunk, kb_start, kb_end, accumulate, columns);
        }
        if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
            return fivebit_detail::fourRowsChunkAVX512<false, CPUProjectionNumericalPolicy::GPUAlignedExpert, SumPairs, Bounded>(
                packed, A0, A1, A2, A3, C0, C1, C2, C3, chunk, kb_start, kb_end, accumulate, columns);
        return fivebit_detail::fourRowsChunkAVX512<false, CPUProjectionNumericalPolicy::BackendNative, SumPairs, Bounded>(
            packed, A0, A1, A2, A3, C0, C1, C2, C3, chunk, kb_start, kb_end, accumulate, columns);
    }
#endif // __AVX512F__ && __AVX512VNNI__ && __AVX512BW__

#if defined(__AVX2__)
    /** @brief Single-row CompactFiveBit AVX2 chunk (`Q5_0`, `Q5_1`, `Q5_K`). */
    template <bool Bounded = false>
    inline void gemvCompactFiveBitAVX2Chunk(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate = false,
        int columns = 64)
    {
        if (!packed.usesCompactFiveBit())
            throw std::invalid_argument("CompactFiveBit AVX2 M=1 kernel received another encoding");
        const __m256i mask_0F = _mm256_set1_epi8(0x0F);
        const __m256i bit16 = _mm256_set1_epi8(0x10);
        const __m256i ones16 = _mm256_set1_epi16(1);
        const __m256i shuf_bytes = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
        const __m256i bit_positions = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, static_cast<char>(128),
            1, 2, 4, 8, 16, 32, 64, static_cast<char>(128),
            1, 2, 4, 8, 16, 32, 64, static_cast<char>(128),
            1, 2, 4, 8, 16, 32, 64, static_cast<char>(128));
        const uint8_t *const base = packed.interleavedBase();
        const size_t stride = static_cast<size_t>(packed.interleaved_block_stride);
        const size_t chunk_origin = static_cast<size_t>(chunk) * packed.blocks_per_row;

        for (int segment = 0; segment < (Bounded ? (columns + 7) / 8 : 8); ++segment)
        {
            const int z = segment / 2;
            const int half = segment % 2;
            __m256 fp = readProjectionOutputVector<Bounded, 8, __m256>(C, columns, segment * 8, accumulate);
            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                const Q8_1Block &a = A_q8[kb];
                const uint8_t *const unit = base + (chunk_origin + kb) * stride;
                __m256i dot = _mm256_setzero_si256();
                for (int group = 0; group < 4; ++group)
                {
                    const __m256i raw = _mm256_load_si256(reinterpret_cast<const __m256i *>(
                        unit + static_cast<size_t>(group) * 256u + static_cast<size_t>(z) * 64u + half * 32u));
                    uint32_t lo_m = 0, hi_m = 0;
                    std::memcpy(&lo_m, unit + 1024u + static_cast<size_t>(group) * 32u + z * 8u + half * 4u, 4);
                    std::memcpy(&hi_m, unit + 1024u + static_cast<size_t>(group + 4) * 32u + z * 8u + half * 4u, 4);
                    const __m256i w_lo = _mm256_or_si256(
                        _mm256_and_si256(raw, mask_0F),
                        fivebit_detail::expandMask32ToBit16AVX2(lo_m, shuf_bytes, bit_positions, bit16));
                    const __m256i w_hi = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask_0F),
                        fivebit_detail::expandMask32ToBit16AVX2(hi_m, shuf_bytes, bit_positions, bit16));
                    const __m256i a_lo = _mm256_set1_epi32(fivebit_detail::packSignedActivationWord(a, group * 4));
                    const __m256i a_hi = _mm256_set1_epi32(fivebit_detail::packSignedActivationWord(a, group * 4 + 16));
                    dot = fivebit_detail::dpbusdFiveBitAVX2(dot, w_lo, a_lo, ones16);
                    dot = fivebit_detail::dpbusdFiveBitAVX2(dot, w_hi, a_hi, ones16);
                }
                const __m256 sc = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i *>(
                    reinterpret_cast<const uint16_t *>(unit + 1280u) + segment * 8)));
                const float a_scale = nativeQuantisedFP16ScaleToFP32(a.d);
                if (packed.is_asymmetric)
                {
                    const __m256 mn = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i *>(
                        reinterpret_cast<const uint16_t *>(unit + 1408u) + segment * 8)));
                    fp = accumulateCorrectedSingleScaleBlockAVX2(
                        fp, dot, sc, a.sum_qs, mn, a_scale, packed.numerical_policy);
                }
                else
                {
                    const __m256i centered = _mm256_sub_epi32(
                        dot, _mm256_set1_epi32(static_cast<int32_t>(a.sum_qs) * 16));
                    fp = accumulateSingleScaleBlockAVX2(
                        fp, centered, sc, a_scale, packed.numerical_policy);
                }
            }
            publishProjectionOutputVector<Bounded, 8>(C, columns, segment * 8, fp);
        }
    }

    /** @brief Two-row CompactFiveBit AVX2 chunk (`Q5_0`, `Q5_1`, `Q5_K`). */
    template <bool SumPairs = false, bool Bounded = false>
    inline void gemmCompactFiveBitTwoRowsAVX2Chunk(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *A0,
        const Q8_1Block *A1,
        float *C0,
        float *C1,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate,
        int columns = 64)
    {
        if (!packed.usesCompactFiveBit())
            throw std::invalid_argument("CompactFiveBit AVX2 M=2 kernel received another encoding");
        const __m256i mask_0F = _mm256_set1_epi8(0x0F);
        const __m256i bit16 = _mm256_set1_epi8(0x10);
        const __m256i ones16 = _mm256_set1_epi16(1);
        const __m256i shuf_bytes = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
        const __m256i bit_positions = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, static_cast<char>(128),
            1, 2, 4, 8, 16, 32, 64, static_cast<char>(128),
            1, 2, 4, 8, 16, 32, 64, static_cast<char>(128),
            1, 2, 4, 8, 16, 32, 64, static_cast<char>(128));
        const uint8_t *const base = packed.interleavedBase();
        const size_t stride = static_cast<size_t>(packed.interleaved_block_stride);
        const size_t chunk_origin = static_cast<size_t>(chunk) * packed.blocks_per_row;

        for (int segment = 0; segment < (Bounded ? (columns + 7) / 8 : 8); ++segment)
        {
            const int z = segment / 2;
            const int half = segment % 2;
            __m256 fp0 = readProjectionOutputVector<Bounded, 8, __m256>(C0, columns, segment * 8, accumulate);
            __m256 fp1 = readProjectionOutputVector<Bounded, 8, __m256>(C1, columns, segment * 8, accumulate);
            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                if (kb + 1 < packed.blocks_per_row)
                    packed.prefetchInterleavedUnitT0(chunk, kb + 1);
                const Q8_1Block &a0 = A0[kb];
                const Q8_1Block &a1 = A1[kb];
                const uint8_t *const unit = base + (chunk_origin + kb) * stride;
                __m256i d0 = _mm256_setzero_si256();
                __m256i d1 = _mm256_setzero_si256();
                for (int group = 0; group < 4; ++group)
                {
                    const __m256i raw = _mm256_load_si256(reinterpret_cast<const __m256i *>(
                        unit + static_cast<size_t>(group) * 256u + static_cast<size_t>(z) * 64u + half * 32u));
                    uint32_t lo_m = 0, hi_m = 0;
                    std::memcpy(&lo_m, unit + 1024u + static_cast<size_t>(group) * 32u + z * 8u + half * 4u, 4);
                    std::memcpy(&hi_m, unit + 1024u + static_cast<size_t>(group + 4) * 32u + z * 8u + half * 4u, 4);
                    const __m256i w_lo = _mm256_or_si256(
                        _mm256_and_si256(raw, mask_0F),
                        fivebit_detail::expandMask32ToBit16AVX2(lo_m, shuf_bytes, bit_positions, bit16));
                    d0 = fivebit_detail::dpbusdFiveBitAVX2(
                        d0, w_lo, _mm256_set1_epi32(fivebit_detail::packSignedActivationWord(a0, group * 4)), ones16);
                    d1 = fivebit_detail::dpbusdFiveBitAVX2(
                        d1, w_lo, _mm256_set1_epi32(fivebit_detail::packSignedActivationWord(a1, group * 4)), ones16);
                    const __m256i w_hi = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask_0F),
                        fivebit_detail::expandMask32ToBit16AVX2(hi_m, shuf_bytes, bit_positions, bit16));
                    d0 = fivebit_detail::dpbusdFiveBitAVX2(
                        d0, w_hi, _mm256_set1_epi32(fivebit_detail::packSignedActivationWord(a0, group * 4 + 16)), ones16);
                    d1 = fivebit_detail::dpbusdFiveBitAVX2(
                        d1, w_hi, _mm256_set1_epi32(fivebit_detail::packSignedActivationWord(a1, group * 4 + 16)), ones16);
                }
                const __m256 sc = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i *>(
                    reinterpret_cast<const uint16_t *>(unit + 1280u) + segment * 8)));
                const float a0_scale = nativeQuantisedFP16ScaleToFP32(a0.d);
                const float a1_scale = nativeQuantisedFP16ScaleToFP32(a1.d);
                if (packed.is_asymmetric)
                {
                    const __m256 mn = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i *>(
                        reinterpret_cast<const uint16_t *>(unit + 1408u) + segment * 8)));
                    fp0 = accumulateCorrectedSingleScaleBlockAVX2(
                        fp0, d0, sc, a0.sum_qs, mn, a0_scale, packed.numerical_policy);
                    fp1 = accumulateCorrectedSingleScaleBlockAVX2(
                        fp1, d1, sc, a1.sum_qs, mn, a1_scale, packed.numerical_policy);
                }
                else
                {
                    const __m256i c0 = _mm256_sub_epi32(d0, _mm256_set1_epi32(static_cast<int32_t>(a0.sum_qs) * 16));
                    const __m256i c1 = _mm256_sub_epi32(d1, _mm256_set1_epi32(static_cast<int32_t>(a1.sum_qs) * 16));
                    fp0 = accumulateSingleScaleBlockAVX2(fp0, c0, sc, a0_scale, packed.numerical_policy);
                    fp1 = accumulateSingleScaleBlockAVX2(fp1, c1, sc, a1_scale, packed.numerical_policy);
                }
            }
            if constexpr (SumPairs)
            {
                const int live = packed.N - chunk * 64 - segment * 8;
                const __m256 sum = _mm256_add_ps(fp0, fp1);
                if (live >= 8)
                    _mm256_storeu_ps(C0 + segment * 8, sum);
                else if (live > 0)
                {
                    const auto mask = _mm256_cmpgt_epi32(
                        _mm256_set1_epi32(live), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
                    _mm256_maskstore_ps(C0 + segment * 8, mask, sum);
                }
            }
            else
            {
                publishProjectionOutputVector<Bounded, 8>(C0, columns, segment * 8, fp0);
                publishProjectionOutputVector<Bounded, 8>(C1, columns, segment * 8, fp1);
            }
        }
    }
#endif // __AVX2__
} // namespace llaminar2::cpu::native_quantised
