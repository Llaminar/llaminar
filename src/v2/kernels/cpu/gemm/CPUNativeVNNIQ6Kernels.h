/**
 * @file CPUNativeVNNIQ6Kernels.h
 * @brief Byte-stable native Q6_K decode kernels for CPU NativeVNNI.
 *
 * Q6_K stores each logical 32-value block as two independently scaled groups
 * of sixteen six-bit values.  Expanding those values to a requantized INT8
 * stream makes decode bandwidth-bound at one byte per weight and also discards
 * the original dual-scale arithmetic.  These kernels consume the packed
 * six-bit representation directly for AVX-512 VNNI and AVX2-emulated VNNI.
 *
 * Serial M=1 and grouped MTP verification call the same templated primitive.
 * Grouping changes only how many independent row accumulators reuse each
 * decoded weight vector; every row retains the following fixed contribution
 * order for each logical K block:
 *
 * @code
 * dot_term    = scale_low * dot_low + scale_high * dot_high;
 * contribution = activation_scale * dot_term;
 * output      = output + contribution;
 * @endcode
 *
 * The VNNI operands are intentionally reversed from the older expanded-INT8
 * kernel.  Unsigned raw Q6 values multiply signed Q8_1 activations, followed
 * by a common `32 * activation_half_sum` centering correction.  This removes
 * all per-column compensation bytes and keeps the prepared block at 1792
 * bytes per 64 columns instead of 2304 bytes.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include <immintrin.h>

#include "CPUNativeVNNIFP16.h"
#include "CPUNativeVNNIWeightPacker.h"
#include "VNNIEmulation.h"

namespace llaminar2::cpu::native_vnni
{
#if defined(__GNUC__) && !defined(__clang__)
    /** Keep serial decode arithmetic intact under the global fast-math flags. */
#define LLAMINAR_Q6K_EXACT_FP                                                  \
    __attribute__((optimize("fp-contract=off", "no-associative-math")))
#else
#define LLAMINAR_Q6K_EXACT_FP
#endif

    namespace q6k_detail
    {
        /** @brief Pack four signed activation bytes without changing their bits. */
        inline int32_t packSignedActivationWord(
            const Q8_1Block &block, int base_index) noexcept
        {
            int32_t packed_word = 0;
            std::memcpy(&packed_word, block.qs + base_index, sizeof(packed_word));
            return packed_word;
        }

        /**
         * @brief Sum the first sixteen signed activation bytes exactly.
         *
         * Q6_K owns one scale for each sixteen-value half.  Q8_1 stores only
         * the total 32-value sum, so the low-half sum is recovered with one
         * fixed SIMD reduction and the high-half sum is `sum_qs - low`.  The
         * reduction is integer-only and therefore identical across AVX2 and
         * AVX-512 execution.
         */
        inline int activationLowHalfSum(const Q8_1Block &block) noexcept
        {
            const __m128i values = _mm_loadu_si128(
                reinterpret_cast<const __m128i *>(block.qs));
            const __m128i byte_ones = _mm_set1_epi8(1);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
            /*
             * Unsigned one times signed activation byte produces four exact
             * signed INT32 partial sums.  Besides replacing PMADDUBSW plus
             * PMADDWD, this needs one fewer persistent SIMD constant in the
             * register-constrained grouped Q6 kernels.
             */
            const __m128i quad_sums = _mm_dpbusd_epi32(
                _mm_setzero_si128(), byte_ones, values);
#else
            const __m128i pair_sums = _mm_maddubs_epi16(byte_ones, values);
            const __m128i word_ones = _mm_set1_epi16(1);
            const __m128i quad_sums = _mm_madd_epi16(pair_sums, word_ones);
#endif
            const __m128i pair_totals = _mm_hadd_epi32(quad_sums, quad_sums);
            const __m128i total = _mm_hadd_epi32(pair_totals, pair_totals);
            return _mm_cvtsi128_si32(total);
        }

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        /**
         * @brief Merge two Q6 high-bit planes into decoded low nibbles.
         *
         * Each mask bit addresses the corresponding byte in the VNNI vector.
         * The low nibble is known to be in `[0, 15]`, so masked additions of
         * sixteen and thirty-two are bit-identical to OR-ing materialized bit
         * vectors.  Updating the decode vector in place avoids keeping two
         * extra ZMM temporaries live beside grouped dot accumulators.
         *
         * @param low_nibbles Sixty-four decoded values containing bits 0..3.
         * @param source Two consecutive sixty-four-bit masks for bits 4 and 5.
         * @return Fully decoded unsigned Q6 values in the range `[0, 63]`.
         */
        inline __m512i mergeHighBitplanesAVX512(
            __m512i low_nibbles,
            const uint8_t *source) noexcept
        {
            uint64_t low_mask = 0;
            uint64_t high_mask = 0;
            std::memcpy(&low_mask, source, sizeof(low_mask));
            std::memcpy(
                &high_mask, source + sizeof(low_mask), sizeof(high_mask));
            low_nibbles = _mm512_mask_add_epi8(
                low_nibbles,
                static_cast<__mmask64>(low_mask),
                low_nibbles,
                _mm512_set1_epi8(0x10));
            return _mm512_mask_add_epi8(
                low_nibbles,
                static_cast<__mmask64>(high_mask),
                low_nibbles,
                _mm512_set1_epi8(0x20));
        }

        /**
         * @brief Apply one native-Q6 K-block contribution to one output vector.
         *
         * The helper is shared by the explicit three- and four-row kernels so
         * their floating-point expression remains textually identical to the
         * serial kernel.  Keeping this operation as a force-inlined leaf also
         * shortens the lifetime of corrected dot and scale temporaries; the
         * persistent output accumulator can remain in a ZMM register across
         * the complete increasing-K loop.
         *
         * @param accumulator Running FP32 output for sixteen columns.
         * @param dot_low Raw unsigned-Q6/signed-Q8 dot for values 0..15.
         * @param dot_high Raw unsigned-Q6/signed-Q8 dot for values 16..31.
         * @param activation Source Q8_1 activation block for this row and K.
         * @param scale_low Native Q6 scale for values 0..15.
         * @param scale_high Native Q6 scale for values 16..31.
         * @return Updated accumulator with exactly one K-block contribution.
         */
        LLAMINAR_Q6K_EXACT_FP
        inline __m512 addScaledDotsAVX512(
            __m512 accumulator,
            __m512i dot_low,
            __m512i dot_high,
            const Q8_1Block &activation,
            __m512 scale_low,
            __m512 scale_high) noexcept
        {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
            const int low_sum = activationLowHalfSum(activation);
            const int high_sum =
                static_cast<int>(activation.sum_qs) - low_sum;
            const __m512i corrected_low = _mm512_sub_epi32(
                dot_low,
                _mm512_slli_epi32(_mm512_set1_epi32(low_sum), 5));
            const __m512i corrected_high = _mm512_sub_epi32(
                dot_high,
                _mm512_slli_epi32(_mm512_set1_epi32(high_sum), 5));
            const __m512 low_term = _mm512_mul_ps(
                scale_low, _mm512_cvtepi32_ps(corrected_low));
            const __m512 high_term = _mm512_mul_ps(
                scale_high, _mm512_cvtepi32_ps(corrected_high));
            const __m512 dot_term = _mm512_add_ps(low_term, high_term);
            const __m512 activation_scale = _mm512_set1_ps(
                nativeVNNIFP16ScaleToFP32(activation.d));
            const __m512 contribution =
                _mm512_mul_ps(activation_scale, dot_term);
            return _mm512_add_ps(accumulator, contribution);
        }

        /**
         * @brief Execute one native Q6_K chunk for one to four verifier rows.
         *
         * @tparam Rows Number of independent activation rows sharing weights.
         * @tparam ZSpan Number of sixteen-column vectors kept live together.
         * @param packed Native dual-scale Q6_K prepared weights.
         * @param activations One Q8_1 block stream per row.
         * @param outputs One 64-column output chunk per row.
         * @param chunk Output-column chunk index.
         * @param kb_start First logical K block.
         * @param kb_end One-past-last logical K block.
         * @param accumulate Load existing outputs before adding this K range.
         */
        template <int Rows, int ZSpan>
        LLAMINAR_Q6K_EXACT_FP
        inline void rowsChunkAVX512(
            const CPUNativeVNNIPackedWeights &packed,
            const std::array<const Q8_1Block *, Rows> &activations,
            const std::array<float *, Rows> &outputs,
            int chunk,
            int kb_start,
            int kb_end,
            bool accumulate)
        {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
            static_assert(Rows >= 1 && Rows <= 4);
            static_assert(ZSpan >= 1 && ZSpan <= 4);
            static_assert(4 % ZSpan == 0);
            if (!packed.usesQ6KNativeDualScale())
                throw std::invalid_argument(
                    "Native Q6_K AVX-512 kernel received another packed encoding");

            const __m512i nibble_mask = _mm512_set1_epi8(0x0F);
            const __m512i center = _mm512_set1_epi32(32);

            for (int z_base = 0; z_base < 4; z_base += ZSpan)
            {
                __m512 fp_accumulators[Rows][ZSpan];
                for (int row = 0; row < Rows; ++row)
                {
                    for (int zi = 0; zi < ZSpan; ++zi)
                    {
                        fp_accumulators[row][zi] = accumulate
                            ? _mm512_loadu_ps(
                                  outputs[row] + (z_base + zi) * 16)
                            : _mm512_setzero_ps();
                    }
                }

                for (int kb = kb_start; kb < kb_end; ++kb)
                {
                    __m512i dot_low[Rows][ZSpan];
                    __m512i dot_high[Rows][ZSpan];
                    for (int row = 0; row < Rows; ++row)
                    {
                        for (int zi = 0; zi < ZSpan; ++zi)
                        {
                            dot_low[row][zi] = _mm512_setzero_si512();
                            dot_high[row][zi] = _mm512_setzero_si512();
                        }
                    }

#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
                    for (int group = 0; group < 4; ++group)
                    {
                        /*
                         * Keep reusable activation words scalar. Retaining all
                         * row broadcasts beside the dot accumulators exhausts
                         * the ZMM register file at grouped widths and spills
                         * the persistent FP accumulators. The compiler can
                         * retain broadcasts when registers are plentiful or
                         * cheaply rematerialize them immediately before DPBUSD.
                         */
                        int32_t activation_low_words[Rows];
                        int32_t activation_high_words[Rows];
                        for (int row = 0; row < Rows; ++row)
                        {
                            activation_low_words[row] =
                                packSignedActivationWord(
                                    activations[row][kb], group * 4);
                            activation_high_words[row] =
                                packSignedActivationWord(
                                    activations[row][kb], group * 4 + 16);
                        }

                        for (int zi = 0; zi < ZSpan; ++zi)
                        {
                            const int z = z_base + zi;
                            const __m512i raw = _mm512_load_si512(
                                packed.interleavedB(chunk, kb, group, z));
                            const __m512i low_nibbles =
                                _mm512_and_si512(raw, nibble_mask);
                            const __m512i high_nibbles = _mm512_and_si512(
                                _mm512_srli_epi16(raw, 4), nibble_mask);
                            const __m512i raw_low =
                                mergeHighBitplanesAVX512(
                                    low_nibbles,
                                    packed.q6KHighBitplanes(
                                        chunk, kb, group, z));
                            const __m512i raw_high =
                                mergeHighBitplanesAVX512(
                                    high_nibbles,
                                    packed.q6KHighBitplanes(
                                        chunk, kb, group + 4, z));

                            for (int row = 0; row < Rows; ++row)
                            {
                                const __m512i activation_low =
                                    _mm512_set1_epi32(
                                        activation_low_words[row]);
                                const __m512i activation_high =
                                    _mm512_set1_epi32(
                                        activation_high_words[row]);
                                dot_low[row][zi] = _mm512_dpbusd_epi32(
                                    dot_low[row][zi], raw_low,
                                    activation_low);
                                dot_high[row][zi] = _mm512_dpbusd_epi32(
                                    dot_high[row][zi], raw_high,
                                    activation_high);
                            }
                        }
                    }

                    for (int row = 0; row < Rows; ++row)
                    {
                        const Q8_1Block &activation = activations[row][kb];
                        const int low_sum = activationLowHalfSum(activation);
                        const int high_sum =
                            static_cast<int>(activation.sum_qs) - low_sum;
                        const __m512i low_center = _mm512_mullo_epi32(
                            center, _mm512_set1_epi32(low_sum));
                        const __m512i high_center = _mm512_mullo_epi32(
                            center, _mm512_set1_epi32(high_sum));
                        const __m512 activation_scale = _mm512_set1_ps(
                            nativeVNNIFP16ScaleToFP32(activation.d));

                        for (int zi = 0; zi < ZSpan; ++zi)
                        {
                            const int z = z_base + zi;
                            const __m512 corrected_low = _mm512_cvtepi32_ps(
                                _mm512_sub_epi32(
                                    dot_low[row][zi], low_center));
                            const __m512 corrected_high = _mm512_cvtepi32_ps(
                                _mm512_sub_epi32(
                                    dot_high[row][zi], high_center));
                            const __m512 scale_low = _mm512_cvtph_ps(
                                _mm256_load_si256(
                                    reinterpret_cast<const __m256i *>(
                                        packed.chunkScales(chunk, kb) +
                                        z * 16)));
                            const __m512 scale_high = _mm512_cvtph_ps(
                                _mm256_load_si256(
                                    reinterpret_cast<const __m256i *>(
                                        packed.chunkMins(chunk, kb) +
                                        z * 16)));

                            // Keep these operations separate and ordered.  The
                            // disassembly gate verifies vmulps/vaddps rather
                            // than a reassociated or contracted expression.
                            const __m512 low_term =
                                _mm512_mul_ps(scale_low, corrected_low);
                            const __m512 high_term =
                                _mm512_mul_ps(scale_high, corrected_high);
                            const __m512 dot_term =
                                _mm512_add_ps(low_term, high_term);
                            const __m512 contribution =
                                _mm512_mul_ps(activation_scale, dot_term);
                            fp_accumulators[row][zi] = _mm512_add_ps(
                                fp_accumulators[row][zi], contribution);
                        }
                    }
                }

                for (int row = 0; row < Rows; ++row)
                {
                    for (int zi = 0; zi < ZSpan; ++zi)
                    {
                        _mm512_storeu_ps(
                            outputs[row] + (z_base + zi) * 16,
                            fp_accumulators[row][zi]);
                    }
                }
            }
        }

        /**
         * @brief Execute a spill-free two-row, two-vector native-Q6 tile.
         *
         * The generic array-shaped primitive is compact source, but GCC gives
         * several short-lived decoded-weight vectors stack homes when it is
         * instantiated as `Rows=2, ZSpan=2`.  Those stores sit inside the
         * four-group loop and therefore repeat for every K block.  Naming the
         * eight integer dot chains and four floating-point accumulators makes
         * their disjoint lifetimes visible to register allocation while still
         * providing enough independent work to cover DPBUSD latency.
         *
         * Column scheduling is deliberately the only difference from serial
         * decode.  Every `(row, column lane)` accumulator visits K blocks in
         * ascending order and applies `addScaledDotsAVX512()` exactly once per
         * block.  Consequently the grouped result remains byte-identical to
         * two independent M=1 launches.
         *
         * @param packed Native dual-scale Q6_K prepared weights.
         * @param activation0 First Q8_1 activation row.
         * @param activation1 Second Q8_1 activation row.
         * @param output0 First FP32 output row.
         * @param output1 Second FP32 output row.
         * @param chunk Output-column chunk index.
         * @param kb_start First logical K block.
         * @param kb_end One-past-last logical K block.
         * @param accumulate Load existing outputs before adding this K range.
         */
        LLAMINAR_Q6K_EXACT_FP
        inline void twoRowsTwoVectorsChunkAVX512(
            const CPUNativeVNNIPackedWeights &packed,
            const Q8_1Block *activation0,
            const Q8_1Block *activation1,
            float *output0,
            float *output1,
            int chunk,
            int kb_start,
            int kb_end,
            bool accumulate)
        {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
            if (!packed.usesQ6KNativeDualScale())
                throw std::invalid_argument(
                    "Native Q6_K AVX-512 two-row kernel received another "
                    "packed encoding");

            const __m512i nibble_mask = _mm512_set1_epi8(0x0F);
            for (int z_base = 0; z_base < 4; z_base += 2)
            {
                __m512 fp00 = accumulate
                    ? _mm512_loadu_ps(output0 + z_base * 16)
                    : _mm512_setzero_ps();
                __m512 fp01 = accumulate
                    ? _mm512_loadu_ps(output0 + (z_base + 1) * 16)
                    : _mm512_setzero_ps();
                __m512 fp10 = accumulate
                    ? _mm512_loadu_ps(output1 + z_base * 16)
                    : _mm512_setzero_ps();
                __m512 fp11 = accumulate
                    ? _mm512_loadu_ps(output1 + (z_base + 1) * 16)
                    : _mm512_setzero_ps();

                for (int kb = kb_start; kb < kb_end; ++kb)
                {
                    __m512i dot_low00 = _mm512_setzero_si512();
                    __m512i dot_high00 = _mm512_setzero_si512();
                    __m512i dot_low01 = _mm512_setzero_si512();
                    __m512i dot_high01 = _mm512_setzero_si512();
                    __m512i dot_low10 = _mm512_setzero_si512();
                    __m512i dot_high10 = _mm512_setzero_si512();
                    __m512i dot_low11 = _mm512_setzero_si512();
                    __m512i dot_high11 = _mm512_setzero_si512();

#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
                    for (int group = 0; group < 4; ++group)
                    {
                        /*
                         * Activation words are scalar so each broadcast dies
                         * immediately after its DPBUSD.  Keeping eight
                         * broadcasts live across both column vectors would
                         * merely trade decoded-weight spills for activation
                         * spills without reducing arithmetic.
                         */
                        const int32_t activation0_low =
                            packSignedActivationWord(
                                activation0[kb], group * 4);
                        const int32_t activation0_high =
                            packSignedActivationWord(
                                activation0[kb], group * 4 + 16);
                        const int32_t activation1_low =
                            packSignedActivationWord(
                                activation1[kb], group * 4);
                        const int32_t activation1_high =
                            packSignedActivationWord(
                                activation1[kb], group * 4 + 16);

                        const int z0 = z_base;
                        const __m512i raw0 = _mm512_load_si512(
                            packed.interleavedB(chunk, kb, group, z0));
                        const __m512i raw_low0 = mergeHighBitplanesAVX512(
                            _mm512_and_si512(raw0, nibble_mask),
                            packed.q6KHighBitplanes(
                                chunk, kb, group, z0));
                        dot_low00 = _mm512_dpbusd_epi32(
                            dot_low00,
                            raw_low0,
                            _mm512_set1_epi32(activation0_low));
                        dot_low10 = _mm512_dpbusd_epi32(
                            dot_low10,
                            raw_low0,
                            _mm512_set1_epi32(activation1_low));

                        /*
                         * Consume the low-half decode before constructing the
                         * high half.  Keeping both decoded vectors live beside
                         * eight dot accumulators forces GCC to stack-home one
                         * of them in every group iteration.
                         */
                        const __m512i raw_high0 = mergeHighBitplanesAVX512(
                            _mm512_and_si512(
                                _mm512_srli_epi16(raw0, 4), nibble_mask),
                            packed.q6KHighBitplanes(
                                chunk, kb, group + 4, z0));
                        dot_high00 = _mm512_dpbusd_epi32(
                            dot_high00,
                            raw_high0,
                            _mm512_set1_epi32(activation0_high));
                        dot_high10 = _mm512_dpbusd_epi32(
                            dot_high10,
                            raw_high0,
                            _mm512_set1_epi32(activation1_high));

                        const int z1 = z_base + 1;
                        const __m512i raw1 = _mm512_load_si512(
                            packed.interleavedB(chunk, kb, group, z1));
                        const __m512i raw_low1 = mergeHighBitplanesAVX512(
                            _mm512_and_si512(raw1, nibble_mask),
                            packed.q6KHighBitplanes(
                                chunk, kb, group, z1));
                        dot_low01 = _mm512_dpbusd_epi32(
                            dot_low01,
                            raw_low1,
                            _mm512_set1_epi32(activation0_low));
                        dot_low11 = _mm512_dpbusd_epi32(
                            dot_low11,
                            raw_low1,
                            _mm512_set1_epi32(activation1_low));

                        const __m512i raw_high1 = mergeHighBitplanesAVX512(
                            _mm512_and_si512(
                                _mm512_srli_epi16(raw1, 4), nibble_mask),
                            packed.q6KHighBitplanes(
                                chunk, kb, group + 4, z1));
                        dot_high01 = _mm512_dpbusd_epi32(
                            dot_high01,
                            raw_high1,
                            _mm512_set1_epi32(activation0_high));
                        dot_high11 = _mm512_dpbusd_epi32(
                            dot_high11,
                            raw_high1,
                            _mm512_set1_epi32(activation1_high));
                    }

                    const __m512 scale_low0 = _mm512_cvtph_ps(
                        _mm256_load_si256(
                            reinterpret_cast<const __m256i *>(
                                packed.chunkScales(chunk, kb) +
                                z_base * 16)));
                    const __m512 scale_high0 = _mm512_cvtph_ps(
                        _mm256_load_si256(
                            reinterpret_cast<const __m256i *>(
                                packed.chunkMins(chunk, kb) +
                                z_base * 16)));
                    const __m512 scale_low1 = _mm512_cvtph_ps(
                        _mm256_load_si256(
                            reinterpret_cast<const __m256i *>(
                                packed.chunkScales(chunk, kb) +
                                (z_base + 1) * 16)));
                    const __m512 scale_high1 = _mm512_cvtph_ps(
                        _mm256_load_si256(
                            reinterpret_cast<const __m256i *>(
                                packed.chunkMins(chunk, kb) +
                                (z_base + 1) * 16)));

                    fp00 = addScaledDotsAVX512(
                        fp00, dot_low00, dot_high00, activation0[kb],
                        scale_low0, scale_high0);
                    fp01 = addScaledDotsAVX512(
                        fp01, dot_low01, dot_high01, activation0[kb],
                        scale_low1, scale_high1);
                    fp10 = addScaledDotsAVX512(
                        fp10, dot_low10, dot_high10, activation1[kb],
                        scale_low0, scale_high0);
                    fp11 = addScaledDotsAVX512(
                        fp11, dot_low11, dot_high11, activation1[kb],
                        scale_low1, scale_high1);
                }

                _mm512_storeu_ps(output0 + z_base * 16, fp00);
                _mm512_storeu_ps(output0 + (z_base + 1) * 16, fp01);
                _mm512_storeu_ps(output1 + z_base * 16, fp10);
                _mm512_storeu_ps(output1 + (z_base + 1) * 16, fp11);
            }
        }

        /**
         * @brief Execute the register-resident AVX-512 Q6 verifier tile.
         *
         * The generic array-based kernel is ideal for M=1 and M=2, but GCC
         * gives its three- and four-row accumulator arrays stack homes inside
         * the K loop.  Those loads and stores erase much of the grouped weight
         * reuse.  This specialization names every live dot and FP accumulator,
         * limiting the physical tile to one sixteen-column ZMM while exposing
         * three or four independent DPBUSD dependency chains to the scheduler.
         *
         * The prepared Q6 bytes are decoded once per group and then consumed by
         * every row.  Each row still visits K blocks in ascending order and
         * calls addScaledDotsAVX512() once per block, exactly as serial decode.
         * A null fourth row is structurally valid only for the Rows=3 template.
         *
         * @tparam Rows Three or four verifier rows sharing one weight decode.
         * @param packed Native dual-scale Q6_K prepared weights.
         * @param activation0 First Q8_1 activation row.
         * @param activation1 Second Q8_1 activation row.
         * @param activation2 Third Q8_1 activation row.
         * @param activation3 Fourth row, required only when Rows is four.
         * @param output0 First FP32 output row.
         * @param output1 Second FP32 output row.
         * @param output2 Third FP32 output row.
         * @param output3 Fourth row, required only when Rows is four.
         * @param chunk Output-column chunk index.
         * @param kb_start First logical K block.
         * @param kb_end One-past-last logical K block.
         * @param accumulate Load existing outputs before adding this K range.
         */
        template <int Rows>
        LLAMINAR_Q6K_EXACT_FP
        inline void wideRowsChunkAVX512(
            const CPUNativeVNNIPackedWeights &packed,
            const Q8_1Block *activation0,
            const Q8_1Block *activation1,
            const Q8_1Block *activation2,
            const Q8_1Block *activation3,
            float *output0,
            float *output1,
            float *output2,
            float *output3,
            int chunk,
            int kb_start,
            int kb_end,
            bool accumulate)
        {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
            static_assert(Rows == 3 || Rows == 4);
            if (!packed.usesQ6KNativeDualScale())
                throw std::invalid_argument(
                    "Native Q6_K AVX-512 wide-row kernel received another "
                    "packed encoding");

            const __m512i nibble_mask = _mm512_set1_epi8(0x0F);
            for (int z = 0; z < 4; ++z)
            {
                __m512 fp0 = accumulate
                    ? _mm512_loadu_ps(output0 + z * 16)
                    : _mm512_setzero_ps();
                __m512 fp1 = accumulate
                    ? _mm512_loadu_ps(output1 + z * 16)
                    : _mm512_setzero_ps();
                __m512 fp2 = accumulate
                    ? _mm512_loadu_ps(output2 + z * 16)
                    : _mm512_setzero_ps();
                __m512 fp3 = _mm512_setzero_ps();
                if constexpr (Rows == 4)
                {
                    fp3 = accumulate
                        ? _mm512_loadu_ps(output3 + z * 16)
                        : _mm512_setzero_ps();
                }

                for (int kb = kb_start; kb < kb_end; ++kb)
                {
                    __m512i dot_low0 = _mm512_setzero_si512();
                    __m512i dot_high0 = _mm512_setzero_si512();
                    __m512i dot_low1 = _mm512_setzero_si512();
                    __m512i dot_high1 = _mm512_setzero_si512();
                    __m512i dot_low2 = _mm512_setzero_si512();
                    __m512i dot_high2 = _mm512_setzero_si512();
                    __m512i dot_low3 = _mm512_setzero_si512();
                    __m512i dot_high3 = _mm512_setzero_si512();

#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
                    for (int group = 0; group < 4; ++group)
                    {
                        const __m512i raw = _mm512_load_si512(
                            packed.interleavedB(chunk, kb, group, z));
                        const __m512i low_nibbles =
                            _mm512_and_si512(raw, nibble_mask);
                        const __m512i high_nibbles = _mm512_and_si512(
                            _mm512_srli_epi16(raw, 4), nibble_mask);
                        const __m512i raw_low = mergeHighBitplanesAVX512(
                            low_nibbles,
                            packed.q6KHighBitplanes(
                                chunk, kb, group, z));
                        const __m512i raw_high = mergeHighBitplanesAVX512(
                            high_nibbles,
                            packed.q6KHighBitplanes(
                                chunk, kb, group + 4, z));

                        dot_low0 = _mm512_dpbusd_epi32(
                            dot_low0,
                            raw_low,
                            _mm512_set1_epi32(packSignedActivationWord(
                                activation0[kb], group * 4)));
                        dot_high0 = _mm512_dpbusd_epi32(
                            dot_high0,
                            raw_high,
                            _mm512_set1_epi32(packSignedActivationWord(
                                activation0[kb], group * 4 + 16)));
                        dot_low1 = _mm512_dpbusd_epi32(
                            dot_low1,
                            raw_low,
                            _mm512_set1_epi32(packSignedActivationWord(
                                activation1[kb], group * 4)));
                        dot_high1 = _mm512_dpbusd_epi32(
                            dot_high1,
                            raw_high,
                            _mm512_set1_epi32(packSignedActivationWord(
                                activation1[kb], group * 4 + 16)));
                        dot_low2 = _mm512_dpbusd_epi32(
                            dot_low2,
                            raw_low,
                            _mm512_set1_epi32(packSignedActivationWord(
                                activation2[kb], group * 4)));
                        dot_high2 = _mm512_dpbusd_epi32(
                            dot_high2,
                            raw_high,
                            _mm512_set1_epi32(packSignedActivationWord(
                                activation2[kb], group * 4 + 16)));
                        if constexpr (Rows == 4)
                        {
                            dot_low3 = _mm512_dpbusd_epi32(
                                dot_low3,
                                raw_low,
                                _mm512_set1_epi32(packSignedActivationWord(
                                    activation3[kb], group * 4)));
                            dot_high3 = _mm512_dpbusd_epi32(
                                dot_high3,
                                raw_high,
                                _mm512_set1_epi32(packSignedActivationWord(
                                    activation3[kb], group * 4 + 16)));
                        }
                    }

                    const __m512 scale_low = _mm512_cvtph_ps(
                        _mm256_load_si256(
                            reinterpret_cast<const __m256i *>(
                                packed.chunkScales(chunk, kb) + z * 16)));
                    const __m512 scale_high = _mm512_cvtph_ps(
                        _mm256_load_si256(
                            reinterpret_cast<const __m256i *>(
                                packed.chunkMins(chunk, kb) + z * 16)));
                    fp0 = addScaledDotsAVX512(
                        fp0, dot_low0, dot_high0, activation0[kb],
                        scale_low, scale_high);
                    fp1 = addScaledDotsAVX512(
                        fp1, dot_low1, dot_high1, activation1[kb],
                        scale_low, scale_high);
                    fp2 = addScaledDotsAVX512(
                        fp2, dot_low2, dot_high2, activation2[kb],
                        scale_low, scale_high);
                    if constexpr (Rows == 4)
                    {
                        fp3 = addScaledDotsAVX512(
                            fp3, dot_low3, dot_high3, activation3[kb],
                            scale_low, scale_high);
                    }
                }

                _mm512_storeu_ps(output0 + z * 16, fp0);
                _mm512_storeu_ps(output1 + z * 16, fp1);
                _mm512_storeu_ps(output2 + z * 16, fp2);
                if constexpr (Rows == 4)
                    _mm512_storeu_ps(output3 + z * 16, fp3);
            }
        }
#endif

        /**
         * @brief Expand one 32-byte half of two Q6 high-bit planes for AVX2.
         *
         * Four mask bytes each own eight consecutive output bytes. A byte
         * shuffle replicates each mask byte over its eight destinations, then
         * parallel bit tests materialize bit four and bit five. This preserves
         * the compact bitplane representation without scalar per-weight work.
         */
        inline __m256i expandHighBitplanesAVX2(
            const uint8_t *source, int half) noexcept
        {
            int32_t low_mask = 0;
            int32_t high_mask = 0;
            std::memcpy(
                &low_mask,
                source + static_cast<size_t>(half) * sizeof(low_mask),
                sizeof(low_mask));
            std::memcpy(
                &high_mask,
                source + sizeof(uint64_t) +
                    static_cast<size_t>(half) * sizeof(high_mask),
                sizeof(high_mask));

            const __m256i replicate_mask = _mm256_setr_epi8(
                0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2,
                3, 3, 3, 3, 3, 3, 3, 3);
            const __m256i bit_selectors = _mm256_setr_epi8(
                1, 2, 4, 8, 16, 32, 64, static_cast<char>(0x80),
                1, 2, 4, 8, 16, 32, 64, static_cast<char>(0x80),
                1, 2, 4, 8, 16, 32, 64, static_cast<char>(0x80),
                1, 2, 4, 8, 16, 32, 64, static_cast<char>(0x80));
            const __m256i replicated_low = _mm256_shuffle_epi8(
                _mm256_set1_epi32(low_mask), replicate_mask);
            const __m256i replicated_high = _mm256_shuffle_epi8(
                _mm256_set1_epi32(high_mask), replicate_mask);
            const __m256i low_present = _mm256_cmpeq_epi8(
                _mm256_and_si256(replicated_low, bit_selectors),
                bit_selectors);
            const __m256i high_present = _mm256_cmpeq_epi8(
                _mm256_and_si256(replicated_high, bit_selectors),
                bit_selectors);
            return _mm256_or_si256(
                _mm256_and_si256(
                    low_present, _mm256_set1_epi8(0x10)),
                _mm256_and_si256(
                    high_present, _mm256_set1_epi8(0x20)));
        }

        /**
         * @brief Execute one native Q6_K chunk with AVX2-emulated VNNI.
         *
         * One eight-column segment is live at a time.  This is intentional:
         * AVX2 exposes sixteen architectural YMM registers, so retaining the
         * whole 64-column/two-row tile would force stack spills.  Weight bytes
         * remain sequential and each loaded vector is reused across all rows.
         */
        template <int Rows>
        LLAMINAR_Q6K_EXACT_FP
        inline void rowsChunkAVX2(
            const CPUNativeVNNIPackedWeights &packed,
            const std::array<const Q8_1Block *, Rows> &activations,
            const std::array<float *, Rows> &outputs,
            int chunk,
            int kb_start,
            int kb_end,
            bool accumulate)
        {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
            static_assert(Rows >= 1 && Rows <= 2);
            if (!packed.usesQ6KNativeDualScale())
                throw std::invalid_argument(
                    "Native Q6_K AVX2 kernel received another packed encoding");

            const __m256i nibble_mask = _mm256_set1_epi8(0x0F);
            const __m256i center = _mm256_set1_epi32(32);

            for (int segment = 0; segment < 8; ++segment)
            {
                const int z = segment / 2;
                const int z_half = segment % 2;
                __m256 fp_accumulators[Rows];
                for (int row = 0; row < Rows; ++row)
                {
                    fp_accumulators[row] = accumulate
                        ? _mm256_loadu_ps(outputs[row] + segment * 8)
                        : _mm256_setzero_ps();
                }

                for (int kb = kb_start; kb < kb_end; ++kb)
                {
                    __m256i dot_low[Rows];
                    __m256i dot_high[Rows];
                    for (int row = 0; row < Rows; ++row)
                    {
                        dot_low[row] = _mm256_setzero_si256();
                        dot_high[row] = _mm256_setzero_si256();
                    }

                    for (int group = 0; group < 4; ++group)
                    {
                        const uint8_t *const raw_base =
                            packed.interleavedB(chunk, kb, group, z) +
                            z_half * 32;
                        const __m256i raw = _mm256_load_si256(
                            reinterpret_cast<const __m256i *>(raw_base));
                        const __m256i low_nibbles =
                            _mm256_and_si256(raw, nibble_mask);
                        const __m256i high_nibbles = _mm256_and_si256(
                            _mm256_srli_epi16(raw, 4), nibble_mask);
                        const uint8_t *const low_high_source =
                            packed.q6KHighBitplanes(
                                chunk, kb, group, z);
                        const uint8_t *const high_high_source =
                            packed.q6KHighBitplanes(
                                chunk, kb, group + 4, z);
                        const __m256i raw_low = _mm256_or_si256(
                            low_nibbles,
                            expandHighBitplanesAVX2(
                                low_high_source, z_half));
                        const __m256i raw_high = _mm256_or_si256(
                            high_nibbles,
                            expandHighBitplanesAVX2(
                                high_high_source, z_half));

                        for (int row = 0; row < Rows; ++row)
                        {
                            const __m256i activation_low =
                                _mm256_set1_epi32(packSignedActivationWord(
                                    activations[row][kb], group * 4));
                            const __m256i activation_high =
                                _mm256_set1_epi32(packSignedActivationWord(
                                    activations[row][kb], group * 4 + 16));
                            dot_low[row] = isa::avx2_dpbusd_epi32(
                                dot_low[row], raw_low, activation_low);
                            dot_high[row] = isa::avx2_dpbusd_epi32(
                                dot_high[row], raw_high, activation_high);
                        }
                    }

                    for (int row = 0; row < Rows; ++row)
                    {
                        const Q8_1Block &activation = activations[row][kb];
                        const int low_sum = activationLowHalfSum(activation);
                        const int high_sum =
                            static_cast<int>(activation.sum_qs) - low_sum;
                        const __m256 corrected_low = _mm256_cvtepi32_ps(
                            _mm256_sub_epi32(
                                dot_low[row],
                                _mm256_mullo_epi32(
                                    center, _mm256_set1_epi32(low_sum))));
                        const __m256 corrected_high = _mm256_cvtepi32_ps(
                            _mm256_sub_epi32(
                                dot_high[row],
                                _mm256_mullo_epi32(
                                    center, _mm256_set1_epi32(high_sum))));
                        const __m256 scale_low = _mm256_cvtph_ps(
                            _mm_load_si128(
                                reinterpret_cast<const __m128i *>(
                                    packed.chunkScales(chunk, kb) +
                                    segment * 8)));
                        const __m256 scale_high = _mm256_cvtph_ps(
                            _mm_load_si128(
                                reinterpret_cast<const __m128i *>(
                                    packed.chunkMins(chunk, kb) +
                                    segment * 8)));
                        const __m256 low_term =
                            _mm256_mul_ps(scale_low, corrected_low);
                        const __m256 high_term =
                            _mm256_mul_ps(scale_high, corrected_high);
                        const __m256 dot_term =
                            _mm256_add_ps(low_term, high_term);
                        const __m256 contribution = _mm256_mul_ps(
                            _mm256_set1_ps(
                                nativeVNNIFP16ScaleToFP32(activation.d)),
                            dot_term);
                        fp_accumulators[row] = _mm256_add_ps(
                            fp_accumulators[row], contribution);
                    }
                }

                for (int row = 0; row < Rows; ++row)
                {
                    _mm256_storeu_ps(
                        outputs[row] + segment * 8,
                        fp_accumulators[row]);
                }
            }
        }
    } // namespace q6k_detail

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
    /**
     * @brief Native Q6_K serial-shaped AVX-512 chunk.
     *
     * @param accumulate Load the existing output before consuming this K
     *        range. This preserves the serial per-K-block arithmetic when an
     *        ordinary prefill tile boundary materializes the accumulator.
     */
    inline void gemvQ6KNativeAVX512Chunk(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *activation,
        float *output,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate = false)
    {
        q6k_detail::rowsChunkAVX512<1, 4>(
            packed,
            {activation},
            {output},
            chunk,
            kb_start,
            kb_end,
            accumulate);
    }

    /**
     * @brief Native Q6_K two-row AVX-512 verifier chunk.
     *
     * Two adjacent sixteen-column vectors are kept live together.  A
     * one-vector tile leaves too few independent DPBUSD chains to cover the
     * instruction latency and made grouped M=2 slower than two serial M=1
     * launches.  Two vectors provide eight integer dot chains plus four FP32
     * output accumulators while remaining within the AVX-512 register file.
     * This changes only column scheduling: each output lane still consumes K
     * blocks in the exact ascending order used by serial decode.
     */
    inline void gemmQ6KNativeTwoRowsAVX512Chunk(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *activation0,
        const Q8_1Block *activation1,
        float *output0,
        float *output1,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate)
    {
        q6k_detail::twoRowsTwoVectorsChunkAVX512(
            packed,
            activation0,
            activation1,
            output0,
            output1,
            chunk,
            kb_start,
            kb_end,
            accumulate);
    }

    /** @brief Native Q6_K three-row AVX-512 verifier chunk. */
    inline void gemmQ6KNativeThreeRowsAVX512Chunk(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *activation0,
        const Q8_1Block *activation1,
        const Q8_1Block *activation2,
        float *output0,
        float *output1,
        float *output2,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate)
    {
        q6k_detail::wideRowsChunkAVX512<3>(
            packed,
            activation0,
            activation1,
            activation2,
            nullptr,
            output0,
            output1,
            output2,
            nullptr,
            chunk,
            kb_start,
            kb_end,
            accumulate);
    }

    /** @brief Native Q6_K four-row AVX-512 verifier chunk. */
    inline void gemmQ6KNativeFourRowsAVX512Chunk(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *activation0,
        const Q8_1Block *activation1,
        const Q8_1Block *activation2,
        const Q8_1Block *activation3,
        float *output0,
        float *output1,
        float *output2,
        float *output3,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate)
    {
        q6k_detail::wideRowsChunkAVX512<4>(
            packed,
            activation0,
            activation1,
            activation2,
            activation3,
            output0,
            output1,
            output2,
            output3,
            chunk,
            kb_start,
            kb_end,
            accumulate);
    }
#endif

    /**
     * @brief Native Q6_K serial-shaped AVX2 chunk.
     *
     * @param accumulate Load the existing output before consuming this K
     *        range. This preserves the serial per-K-block arithmetic when an
     *        ordinary prefill tile boundary materializes the accumulator.
     */
    inline void gemvQ6KNativeAVX2Chunk(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *activation,
        float *output,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate = false)
    {
        q6k_detail::rowsChunkAVX2<1>(
            packed,
            {activation},
            {output},
            chunk,
            kb_start,
            kb_end,
            accumulate);
    }

    /** @brief Native Q6_K two-row AVX2 verifier chunk. */
    inline void gemmQ6KNativeTwoRowsAVX2Chunk(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *activation0,
        const Q8_1Block *activation1,
        float *output0,
        float *output1,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate)
    {
        q6k_detail::rowsChunkAVX2<2>(
            packed, {activation0, activation1}, {output0, output1},
            chunk, kb_start, kb_end, accumulate);
    }

#undef LLAMINAR_Q6K_EXACT_FP

} // namespace llaminar2::cpu::native_vnni
