/**
 * @file CPUNativeAVX2Gemv.h
 * @brief AVX2 GEMV/GEMM kernels using saturation-safe emulated VNNI.
 *
 * These kernels read the SAME packed weight format as the AVX512 VNNI kernels
 * (CPUNativeVNNIGemv.h). Each 64-byte ZMM slot is processed as two 32-byte
 * YMM halves, and 8 YMM accumulators cover the same 64 columns that 4 ZMM
 * accumulators cover in AVX512.
 *
 * Both AVX512 and AVX2 paths are always compiled (no #ifdef gates) so they
 * can be tested against each other on AVX512 hardware via runtime dispatch.
 *
 * Performance note: AVX2 processes half the data per instruction vs AVX512,
 * so expect ~1.5-2× slower throughput. The packed weight layout is unchanged.
 */

#pragma once

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

#include "CPUNativeVNNIWeightPacker.h"
#include "CPUNativeVNNIContributionContract.h"
#include "CPUNativeVNNIFP16.h"
#include "CPUNativeVNNIQ6Kernels.h"
#include "CPUNativeVNNIMultiScaleKernels.h"
#include "VNNIEmulation.h"
#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"

namespace llaminar2::cpu::native_vnni
{

    // =========================================================================
    // Decode LUT tables for nibble → signed int8 mapping
    // =========================================================================
    // These are duplicated from CPUNativeVNNIGemv.h so this header is
    // self-contained. The values are identical to the AVX512 versions.
    // =========================================================================

    namespace avx2_luts
    {
        // Q4_0: nibble → [-8..7]
        alignas(16) static constexpr int8_t Q4_0_DECODE_LUT[16] = {
            -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};
        // IQ4_NL: nibble → non-linear codebook
        alignas(16) static constexpr int8_t IQ4_NL_DECODE_LUT[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
        // Q4_1: nibble → [0..15] unsigned identity
        alignas(16) static constexpr int8_t Q4_1_DECODE_LUT[16] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    } // namespace avx2_luts

    // =========================================================================
    // AVX2 decode LUT builder (selects correct LUT by codebook_id)
    // =========================================================================

    inline __m256i build_decode_lut_avx2_for_codebook(uint8_t codebook_id)
    {
        const int8_t *lut_data;
        switch (codebook_id)
        {
        case 4: // IQ4_NL / IQ4_XS
            lut_data = avx2_luts::IQ4_NL_DECODE_LUT;
            break;
        case 5: // Q4_1
            lut_data = avx2_luts::Q4_1_DECODE_LUT;
            break;
        default: // Q4_0 (codebook 0)
            lut_data = avx2_luts::Q4_0_DECODE_LUT;
            break;
        }
        return isa::build_decode_lut_avx2(lut_data);
    }

    /**
     * @brief Pack four signed Q8_1 bytes as unsigned VNNI input lanes.
     *
     * Flipping each sign bit is exactly equivalent to adding 128 modulo 256,
     * but keeps activation preparation in general-purpose registers.  The
     * corresponding `128 * sum(weights)` compensation is applied after every
     * complete INT32 dot product.
     */
    inline int32_t pack_q8_1_unsigned_word_avx2(
        const Q8_1Block &block, int base_index) noexcept
    {
        uint32_t raw = 0;
        std::memcpy(&raw, block.qs + base_index, sizeof(raw));
        return static_cast<int32_t>(raw ^ 0x80808080u);
    }

    // =========================================================================
    // AVX2 GEMV: nibble-LUT path (Q4_0, IQ4_NL, Q4_1, IQ4_XS)
    // =========================================================================
    //
    // Processes one 64-column N-chunk using 8 YMM accumulators (8 cols each).
    // Each AVX512 ZMM load (64 bytes) is split into two YMM loads (32 bytes).
    // The nibble decode and VNNI accumulation use AVX2 equivalents.
    // =========================================================================

    /**
     * @brief Accumulate one eight-column AVX2 nibble segment over a K range.
     *
     * This is the single nibble dot/scale implementation shared by ordinary
     * GEMV and the device-ordered ExpertOverlay tree. Keeping the segment in a
     * register lets the latter reset and fold exact K partitions without a
     * scratch plane or an alternate arithmetic body.
     *
     * @param packed Prepared nibble matrix.
     * @param A_q8 Q8_1 activation row.
     * @param chunk Sixty-four-column output chunk.
     * @param kb_start First K block, inclusive.
     * @param kb_end Final K block, exclusive.
     * @param decode_lut Broadcast nibble codebook.
     * @param segment Eight-column segment within the chunk.
     * @param accumulator Initial FP32 value for the segment.
     * @return Updated FP32 segment.
     */
    inline __m256 accumulateNibbleSegmentAVX2(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        int chunk,
        int kb_start,
        int kb_end,
        const __m256i decode_lut,
        int segment,
        __m256 accumulator)
    {
        const __m256i bias_128_i32 = _mm256_set1_epi32(128);
        const __m256i mask_0F = _mm256_set1_epi8(0x0F);
        const int z = segment / 2;
        const int byte_offset = (segment % 2) * 32;

        for (int kb = kb_start; kb < kb_end; ++kb)
        {
            const Q8_1Block &activation = A_q8[kb];
            __m256i dot_low = _mm256_setzero_si256();
            __m256i dot_high = _mm256_setzero_si256();
#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
            for (int group = 0; group < 4; ++group)
            {
                const uint8_t *base =
                    packed.interleavedB(chunk, kb, group, z) + byte_offset;
                const __m256i raw = _mm256_load_si256(
                    reinterpret_cast<const __m256i *>(base));
                const __m256i low = _mm256_shuffle_epi8(
                    decode_lut, _mm256_and_si256(raw, mask_0F));
                const __m256i high = _mm256_shuffle_epi8(
                    decode_lut,
                    _mm256_and_si256(
                        _mm256_srli_epi16(raw, 4), mask_0F));
                dot_low = isa::avx2_dpbusd_epi32(
                    dot_low,
                    _mm256_set1_epi32(pack_q8_1_unsigned_word_avx2(
                        activation, group * 4)),
                    low);
                dot_high = isa::avx2_dpbusd_epi32(
                    dot_high,
                    _mm256_set1_epi32(pack_q8_1_unsigned_word_avx2(
                        activation, group * 4 + 16)),
                    high);
            }

            const __m256i compensation = _mm256_cvtepi16_epi32(
                _mm_load_si128(reinterpret_cast<const __m128i *>(
                    packed.chunkComp(chunk, kb) + segment * 8)));
            const __m256i corrected = _mm256_sub_epi32(
                _mm256_add_epi32(dot_low, dot_high),
                _mm256_mullo_epi32(bias_128_i32, compensation));
            const __m256 weight_scale = _mm256_cvtph_ps(
                _mm_load_si128(reinterpret_cast<const __m128i *>(
                    packed.chunkScales(chunk, kb) + segment * 8)));
            const float activation_scale =
                nativeVNNIFP16ScaleToFP32(activation.d);
            if (packed.is_asymmetric)
            {
                const __m256 weight_min = _mm256_cvtph_ps(
                    _mm_load_si128(reinterpret_cast<const __m128i *>(
                        packed.chunkMins(chunk, kb) + segment * 8)));
                accumulator = accumulateCorrectedSingleScaleBlockAVX2(
                    accumulator,
                    corrected,
                    weight_scale,
                    activation.sum_qs,
                    weight_min,
                    activation_scale,
                    packed.numerical_policy);
            }
            else
            {
                accumulator = accumulateSingleScaleBlockAVX2(
                    accumulator,
                    corrected,
                    weight_scale,
                    activation_scale,
                    packed.numerical_policy);
            }
        }
        return accumulator;
    }

    /**
     * @brief Execute one serial-shaped AVX2 nibble chunk over a K range.
     *
     * @param accumulate Load `C` as the initial FP32 accumulator. Tile callers
     *        use this mode to materialize cache boundaries without changing
     *        the serial per-K-block reduction tree.
     */
    inline void gemv_avx2_chunk_native(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk,
        int kb_start,
        int kb_end,
        const __m256i decode_lut,
        bool accumulate = false)
    {
        for (int segment = 0; segment < 8; ++segment)
        {
            const __m256 initial = accumulate
                ? _mm256_loadu_ps(C + segment * 8)
                : _mm256_setzero_ps();
            _mm256_storeu_ps(
                C + segment * 8,
                accumulateNibbleSegmentAVX2(
                    packed,
                    A_q8,
                    chunk,
                    kb_start,
                    kb_end,
                    decode_lut,
                    segment,
                    initial));
        }
    }

    // =========================================================================
    // AVX2 GEMV: non-nibble prepared encodings
    // =========================================================================

    /**
     * @brief Dispatch one non-nibble AVX2 chunk by its typed prepared encoding.
     *
     * Q6_K consumes its native dual-scale payload. Remaining codebooks use the
     * explicitly expanded INT8 encoding below. Keeping the decision here makes
     * every serial and K-partition caller exhaustive over the packed ABI.
     *
     * @param accumulate Load `C` as the initial FP32 accumulator. The flag is
     *        forwarded to native Q6 as well as the expanded-INT8 path so every
     *        prepared encoding has identical tile-boundary semantics.
     */
    inline void gemv_avx2_chunk_non_nibble(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate = false)
    {
        if (packed.usesCompactMultiScale())
        {
            multi_scale::dispatch<8>(packed, std::array{A_q8}, std::array{C},
                                     chunk, kb_start, kb_end, accumulate);
            return;
        }
        if (packed.usesQ6KNativeDualScale())
        {
            gemvQ6KNativeAVX2Chunk(
                packed, A_q8, C, chunk, kb_start, kb_end, accumulate);
            return;
        }
        if (!packed.usesExpandedInt8())
            throw std::invalid_argument(
                "AVX2 non-nibble GEMV received an unsupported packed encoding");

        const __m256i bias_128_i32 = _mm256_set1_epi32(128);
        for (int segment = 0; segment < 8; ++segment)
        {
            const int z = segment / 2;
            const int byte_offset = (segment % 2) * 32;
            __m256 fp_accumulator = accumulate
                ? _mm256_loadu_ps(C + segment * 8)
                : _mm256_setzero_ps();
            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                const Q8_1Block &activation = A_q8[kb];
                __m256i dot_even = _mm256_setzero_si256();
                __m256i dot_odd = _mm256_setzero_si256();
#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
                for (int group = 0; group < 8; group += 2)
                {
                    const __m256i weight_even = _mm256_load_si256(
                        reinterpret_cast<const __m256i *>(
                            packed.interleavedB(
                                chunk, kb, group, z) + byte_offset));
                    const __m256i weight_odd = _mm256_load_si256(
                        reinterpret_cast<const __m256i *>(
                            packed.interleavedB(
                                chunk, kb, group + 1, z) + byte_offset));
                    dot_even = isa::avx2_dpbusd_epi32(
                        dot_even,
                        _mm256_set1_epi32(pack_q8_1_unsigned_word_avx2(
                            activation, group * 4)),
                        weight_even);
                    dot_odd = isa::avx2_dpbusd_epi32(
                        dot_odd,
                        _mm256_set1_epi32(pack_q8_1_unsigned_word_avx2(
                            activation, (group + 1) * 4)),
                        weight_odd);
                }

                const __m256i compensation = _mm256_cvtepi16_epi32(
                    _mm_load_si128(reinterpret_cast<const __m128i *>(
                        packed.chunkComp(chunk, kb) + segment * 8)));
                const __m256i corrected = _mm256_sub_epi32(
                    _mm256_add_epi32(dot_even, dot_odd),
                    _mm256_mullo_epi32(bias_128_i32, compensation));
                const __m256 weight_scale = _mm256_cvtph_ps(
                    _mm_load_si128(reinterpret_cast<const __m128i *>(
                        packed.chunkScales(chunk, kb) + segment * 8)));
                const float activation_scale =
                    nativeVNNIFP16ScaleToFP32(activation.d);
                if (packed.is_asymmetric)
                {
                    const __m256 weight_min = _mm256_cvtph_ps(
                        _mm_load_si128(reinterpret_cast<const __m128i *>(
                            packed.chunkMins(chunk, kb) + segment * 8)));
                    fp_accumulator =
                        accumulateCorrectedSingleScaleBlockAVX2(
                            fp_accumulator,
                            corrected,
                            weight_scale,
                            activation.sum_qs,
                            weight_min,
                            activation_scale,
                            packed.numerical_policy);
                }
                else
                {
                    fp_accumulator = accumulateSingleScaleBlockAVX2(
                        fp_accumulator,
                        corrected,
                        weight_scale,
                        activation_scale,
                        packed.numerical_policy);
                }
            }
            _mm256_storeu_ps(C + segment * 8, fp_accumulator);
        }
    }

    // =========================================================================
    // AVX2 multi-chunk GEMV block (processes consecutive 64-column chunks)
    // =========================================================================

    inline void gemv_avx2_block(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk_start,
        int chunk_count,
        int K_blocks,
        int N,
        const __m256i decode_lut)
    {
        const bool use_nibble_lut = packed.usesNibbleLUT();

        for (int ci = 0; ci < chunk_count; ++ci)
        {
            int chunk = chunk_start + ci;
            int n_start = chunk * 64;
            int n_cols = std::min(64, N - n_start);

            if (n_cols < 64)
            {
                alignas(64) float tmp[64];
                if (use_nibble_lut)
                    gemv_avx2_chunk_native(packed, A_q8, tmp, chunk, 0, K_blocks, decode_lut);
                else
                    gemv_avx2_chunk_non_nibble(packed, A_q8, tmp, chunk, 0, K_blocks);
                std::memcpy(C + n_start, tmp, n_cols * sizeof(float));
            }
            else
            {
                if (use_nibble_lut)
                    gemv_avx2_chunk_native(packed, A_q8, C + n_start, chunk, 0, K_blocks, decode_lut);
                else
                    gemv_avx2_chunk_non_nibble(packed, A_q8, C + n_start, chunk, 0, K_blocks);
            }
        }
    }

    // =========================================================================
    // AVX2 2-Row GEMM Microkernels (share B loads across 2 M rows)
    // =========================================================================

    /**
     * @brief Accumulate one shared-decode AVX2 segment for two rows.
     *
     * This register-returning primitive owns the two-row nibble dot and scale
     * program for both ordinary grouped execution and the exact device K tree.
     * The caller supplies each row's initial value, allowing a partition to
     * begin at positive zero without materializing a scratch plane.
     *
     * @param packed Prepared nibble matrix.
     * @param A_q8_row0 First Q8_1 activation row.
     * @param A_q8_row1 Second Q8_1 activation row.
     * @param chunk Sixty-four-column output chunk.
     * @param kb_start First K block, inclusive.
     * @param kb_end Final K block, exclusive.
     * @param decode_lut Broadcast nibble codebook.
     * @param segment Eight-column segment within the chunk.
     * @param fp0 Running first-row FP32 value, updated in place.
     * @param fp1 Running second-row FP32 value, updated in place.
     */
    inline void accumulateTwoRowsNibbleSegmentAVX2(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_row0,
        const Q8_1Block *A_q8_row1,
        int chunk,
        int kb_start,
        int kb_end,
        const __m256i decode_lut,
        int segment,
        __m256 &fp0,
        __m256 &fp1)
    {
        const __m256i bias_128_i32 = _mm256_set1_epi32(128);
        const __m256i mask_0F = _mm256_set1_epi8(0x0F);
        const int z = segment / 2;
        const int byte_offset = (segment % 2) * 32;

        for (int kb = kb_start; kb < kb_end; ++kb)
        {
                const Q8_1Block &a0 = A_q8_row0[kb];
                const Q8_1Block &a1 = A_q8_row1[kb];
                __m256i dot0_low = _mm256_setzero_si256();
                __m256i dot0_high = _mm256_setzero_si256();
                __m256i dot1_low = _mm256_setzero_si256();
                __m256i dot1_high = _mm256_setzero_si256();

#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
                for (int group = 0; group < 4; ++group)
                {
                    const uint8_t *base =
                        packed.interleavedB(chunk, kb, group, z) + byte_offset;
                    const __m256i raw = _mm256_load_si256(
                        reinterpret_cast<const __m256i *>(base));
                    const __m256i low = _mm256_shuffle_epi8(
                        decode_lut, _mm256_and_si256(raw, mask_0F));
                    const __m256i high = _mm256_shuffle_epi8(
                        decode_lut,
                        _mm256_and_si256(
                            _mm256_srli_epi16(raw, 4), mask_0F));
                    dot0_low = isa::avx2_dpbusd_epi32(
                        dot0_low,
                        _mm256_set1_epi32(pack_q8_1_unsigned_word_avx2(
                            a0, group * 4)),
                        low);
                    dot1_low = isa::avx2_dpbusd_epi32(
                        dot1_low,
                        _mm256_set1_epi32(pack_q8_1_unsigned_word_avx2(
                            a1, group * 4)),
                        low);
                    dot0_high = isa::avx2_dpbusd_epi32(
                        dot0_high,
                        _mm256_set1_epi32(pack_q8_1_unsigned_word_avx2(
                            a0, group * 4 + 16)),
                        high);
                    dot1_high = isa::avx2_dpbusd_epi32(
                        dot1_high,
                        _mm256_set1_epi32(pack_q8_1_unsigned_word_avx2(
                            a1, group * 4 + 16)),
                        high);
                }

                const __m256i compensation = _mm256_cvtepi16_epi32(
                    _mm_load_si128(reinterpret_cast<const __m128i *>(
                        packed.chunkComp(chunk, kb) + segment * 8)));
                const __m256i bias =
                    _mm256_mullo_epi32(bias_128_i32, compensation);
                const __m256i corrected0 = _mm256_sub_epi32(
                    _mm256_add_epi32(dot0_low, dot0_high), bias);
                const __m256i corrected1 = _mm256_sub_epi32(
                    _mm256_add_epi32(dot1_low, dot1_high), bias);
                const __m256 weight_scale = _mm256_cvtph_ps(
                    _mm_load_si128(reinterpret_cast<const __m128i *>(
                        packed.chunkScales(chunk, kb) + segment * 8)));
                const float a0_scale = nativeVNNIFP16ScaleToFP32(a0.d);
                const float a1_scale = nativeVNNIFP16ScaleToFP32(a1.d);
                if (packed.is_asymmetric)
                {
                    const __m256 weight_min = _mm256_cvtph_ps(
                        _mm_load_si128(reinterpret_cast<const __m128i *>(
                            packed.chunkMins(chunk, kb) + segment * 8)));
                    fp0 = accumulateCorrectedSingleScaleBlockAVX2(
                        fp0, corrected0, weight_scale, a0.sum_qs,
                        weight_min, a0_scale, packed.numerical_policy);
                    fp1 = accumulateCorrectedSingleScaleBlockAVX2(
                        fp1, corrected1, weight_scale, a1.sum_qs,
                        weight_min, a1_scale, packed.numerical_policy);
                }
                else
                {
                    fp0 = accumulateSingleScaleBlockAVX2(
                        fp0, corrected0, weight_scale, a0_scale,
                        packed.numerical_policy);
                    fp1 = accumulateSingleScaleBlockAVX2(
                        fp1, corrected1, weight_scale, a1_scale,
                        packed.numerical_policy);
                }
        }
    }

    /**
     * @brief Register-resident AVX2 two-row nibble-LUT verifier kernel.
     *
     * One eight-column segment is carried through the complete increasing-K
     * loop. Independent row chains provide two-way ILP while each decoded
     * weight vector is shared across both rows. Low and high nibbles are folded
     * into those row chains so the saturation-safe emulation fits the sixteen-
     * register AVX2 ABI without hot-loop spills.
     */
    inline void gemm_2row_native_chunk_avx2(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_row0,
        const Q8_1Block *A_q8_row1,
        float *C_row0,
        float *C_row1,
        int chunk,
        int kb_start,
        int kb_end,
        const __m256i decode_lut,
        bool accumulate)
    {
        for (int segment = 0; segment < 8; ++segment)
        {
            __m256 fp0 = accumulate
                ? _mm256_loadu_ps(C_row0 + segment * 8)
                : _mm256_setzero_ps();
            __m256 fp1 = accumulate
                ? _mm256_loadu_ps(C_row1 + segment * 8)
                : _mm256_setzero_ps();
            accumulateTwoRowsNibbleSegmentAVX2(
                packed,
                A_q8_row0,
                A_q8_row1,
                chunk,
                kb_start,
                kb_end,
                decode_lut,
                segment,
                fp0,
                fp1);
            _mm256_storeu_ps(C_row0 + segment * 8, fp0);
            _mm256_storeu_ps(C_row1 + segment * 8, fp1);
        }
    }

    /**
     * @brief AVX2 2-row INT8 pre-decoded GEMM microkernel for one 64-col chunk.
     */
    /** @brief Dispatch a two-row non-nibble AVX2 verifier chunk. */
    inline void gemm_2row_non_nibble_chunk_avx2(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_row0,
        const Q8_1Block *A_q8_row1,
        float *C_row0,
        float *C_row1,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate)
    {
        if (packed.usesCompactMultiScale())
        {
            multi_scale::dispatch<8>(packed, std::array{A_q8_row0, A_q8_row1},
                                     std::array{C_row0, C_row1}, chunk, kb_start, kb_end, accumulate);
            return;
        }
        if (packed.usesQ6KNativeDualScale())
        {
            gemmQ6KNativeTwoRowsAVX2Chunk(
                packed,
                A_q8_row0,
                A_q8_row1,
                C_row0,
                C_row1,
                chunk,
                kb_start,
                kb_end,
                accumulate);
            return;
        }
        if (!packed.usesExpandedInt8())
            throw std::invalid_argument(
                "AVX2 non-nibble two-row kernel received an unsupported packed encoding");

        const __m256i bias_128_i32 = _mm256_set1_epi32(128);
        for (int segment = 0; segment < 8; ++segment)
        {
            const int z = segment / 2;
            const int byte_offset = (segment % 2) * 32;
            __m256 fp0 = accumulate
                ? _mm256_loadu_ps(C_row0 + segment * 8)
                : _mm256_setzero_ps();
            __m256 fp1 = accumulate
                ? _mm256_loadu_ps(C_row1 + segment * 8)
                : _mm256_setzero_ps();
            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                const Q8_1Block &a0 = A_q8_row0[kb];
                const Q8_1Block &a1 = A_q8_row1[kb];
                __m256i dot0_even = _mm256_setzero_si256();
                __m256i dot0_odd = _mm256_setzero_si256();
                __m256i dot1_even = _mm256_setzero_si256();
                __m256i dot1_odd = _mm256_setzero_si256();
#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
                for (int group = 0; group < 8; group += 2)
                {
                    const __m256i weight_even = _mm256_load_si256(
                        reinterpret_cast<const __m256i *>(
                            packed.interleavedB(
                                chunk, kb, group, z) + byte_offset));
                    const __m256i weight_odd = _mm256_load_si256(
                        reinterpret_cast<const __m256i *>(
                            packed.interleavedB(
                                chunk, kb, group + 1, z) + byte_offset));
                    dot0_even = isa::avx2_dpbusd_epi32(
                        dot0_even,
                        _mm256_set1_epi32(pack_q8_1_unsigned_word_avx2(
                            a0, group * 4)),
                        weight_even);
                    dot1_even = isa::avx2_dpbusd_epi32(
                        dot1_even,
                        _mm256_set1_epi32(pack_q8_1_unsigned_word_avx2(
                            a1, group * 4)),
                        weight_even);
                    dot0_odd = isa::avx2_dpbusd_epi32(
                        dot0_odd,
                        _mm256_set1_epi32(pack_q8_1_unsigned_word_avx2(
                            a0, (group + 1) * 4)),
                        weight_odd);
                    dot1_odd = isa::avx2_dpbusd_epi32(
                        dot1_odd,
                        _mm256_set1_epi32(pack_q8_1_unsigned_word_avx2(
                            a1, (group + 1) * 4)),
                        weight_odd);
                }

                const __m256i compensation = _mm256_cvtepi16_epi32(
                    _mm_load_si128(reinterpret_cast<const __m128i *>(
                        packed.chunkComp(chunk, kb) + segment * 8)));
                const __m256i bias =
                    _mm256_mullo_epi32(bias_128_i32, compensation);
                const __m256i corrected0 = _mm256_sub_epi32(
                    _mm256_add_epi32(dot0_even, dot0_odd), bias);
                const __m256i corrected1 = _mm256_sub_epi32(
                    _mm256_add_epi32(dot1_even, dot1_odd), bias);
                const __m256 weight_scale = _mm256_cvtph_ps(
                    _mm_load_si128(reinterpret_cast<const __m128i *>(
                        packed.chunkScales(chunk, kb) + segment * 8)));
                const float a0_scale = nativeVNNIFP16ScaleToFP32(a0.d);
                const float a1_scale = nativeVNNIFP16ScaleToFP32(a1.d);
                if (packed.is_asymmetric)
                {
                    const __m256 weight_min = _mm256_cvtph_ps(
                        _mm_load_si128(reinterpret_cast<const __m128i *>(
                            packed.chunkMins(chunk, kb) + segment * 8)));
                    fp0 = accumulateCorrectedSingleScaleBlockAVX2(
                        fp0, corrected0, weight_scale, a0.sum_qs,
                        weight_min, a0_scale, packed.numerical_policy);
                    fp1 = accumulateCorrectedSingleScaleBlockAVX2(
                        fp1, corrected1, weight_scale, a1.sum_qs,
                        weight_min, a1_scale, packed.numerical_policy);
                }
                else
                {
                    fp0 = accumulateSingleScaleBlockAVX2(
                        fp0, corrected0, weight_scale, a0_scale,
                        packed.numerical_policy);
                    fp1 = accumulateSingleScaleBlockAVX2(
                        fp1, corrected1, weight_scale, a1_scale,
                        packed.numerical_policy);
                }
            }
            _mm256_storeu_ps(C_row0 + segment * 8, fp0);
            _mm256_storeu_ps(C_row1 + segment * 8, fp1);
        }
    }

    /**
     * @brief Fold one 64-column AVX2 K partition in device order.
     *
     * The destination is initialized by partition zero and subsequently read,
     * added, and written exactly once for every increasing partition. This is
     * the AVX2 expression of the same placement-invariant tree used by AVX-512,
     * CUDA, and ROCm.
     *
     * @param destination Running 64-column reduction.
     * @param partial Newly completed 64-column partition.
     * @param first_partition Whether @p partial initializes the reduction.
     */
    inline void foldOrderedAVX2Partition(
        float *destination,
        const float *partial,
        bool first_partition)
    {
        for (int segment = 0; segment < 8; ++segment)
        {
            const __m256 value = _mm256_load_ps(partial + segment * 8);
            _mm256_storeu_ps(
                destination + segment * 8,
                first_partition
                    ? value
                    : _mm256_add_ps(
                          _mm256_loadu_ps(destination + segment * 8),
                          value));
        }
    }

    /**
     * @brief Execute one AVX2 row with the exact device K-partition tree.
     *
     * The encoding's established AVX2 microkernel produces one partition from
     * positive zero. Immediate task-local folding avoids the complete partial
     * plane and applies equally to nibble, expanded-INT8, and native-Q6 weights.
     *
     * @param packed Prepared matrix carrying its typed physical encoding.
     * @param row Q8_1 activation row.
     * @param output FP32 destination row.
     * @param chunk Sixty-four-column output chunk.
     * @param K_blocks Number of source Q8_1 blocks.
     * @param k_tiles Ordered K-partition count.
     * @param k_blocks_per_tile Consecutive blocks in one partition.
     * @param decode_lut Broadcast decoder for nibble formats.
     */
    inline void gemvAVX2OrderedPartitions(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *row,
        float *output,
        int chunk,
        int K_blocks,
        int k_tiles,
        int k_blocks_per_tile,
        const __m256i decode_lut)
    {
        if (MoEProjectionNumericalContract::
                oneNativeVNNIBlockPerOrderedPartition(
                    K_blocks *
                    MoEProjectionNumericalContract::
                        native_vnni_values_per_block))
        {
            if (packed.usesNibbleLUT())
            {
                gemv_avx2_chunk_native(
                    packed,
                    row,
                    output,
                    chunk,
                    0,
                    K_blocks,
                    decode_lut,
                    /*accumulate=*/false);
            }
            else
            {
                gemv_avx2_chunk_non_nibble(
                    packed,
                    row,
                    output,
                    chunk,
                    0,
                    K_blocks,
                    /*accumulate=*/false);
            }
            return;
        }

        if (packed.usesNibbleLUT())
        {
            for (int segment = 0; segment < 8; ++segment)
            {
                __m256 reduced = _mm256_setzero_ps();
                for (int k_tile = 0; k_tile < k_tiles; ++k_tile)
                {
                    const int kb_start = k_tile * k_blocks_per_tile;
                    const int kb_end = std::min(
                        kb_start + k_blocks_per_tile, K_blocks);
                    const __m256 partial = accumulateNibbleSegmentAVX2(
                        packed,
                        row,
                        chunk,
                        kb_start,
                        kb_end,
                        decode_lut,
                        segment,
                        _mm256_setzero_ps());
                    reduced = k_tile == 0
                        ? partial
                        : _mm256_add_ps(reduced, partial);
                }
                _mm256_storeu_ps(output + segment * 8, reduced);
            }
            return;
        }

        alignas(32) float partial[64];
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile)
        {
            const int kb_start = k_tile * k_blocks_per_tile;
            const int kb_end = std::min(
                kb_start + k_blocks_per_tile, K_blocks);
            gemv_avx2_chunk_non_nibble(
                packed,
                row,
                partial,
                chunk,
                kb_start,
                kb_end,
                /*accumulate=*/false);
            foldOrderedAVX2Partition(
                output, partial, k_tile == 0);
        }
    }

    /**
     * @brief Execute two AVX2 rows with shared decode and exact K folding.
     *
     * Each physical encoding keeps its canonical two-row microkernel. Only one
     * 512-byte task-local partition is materialized before an immediate ordered
     * fold, retaining shared weight decode without a layer-sized scratch plane.
     *
     * @param packed Prepared matrix carrying its typed physical encoding.
     * @param row0 First Q8_1 activation row.
     * @param row1 Second Q8_1 activation row.
     * @param output0 First FP32 destination row.
     * @param output1 Second FP32 destination row.
     * @param chunk Sixty-four-column output chunk.
     * @param K_blocks Number of source Q8_1 blocks.
     * @param k_tiles Ordered K-partition count.
     * @param k_blocks_per_tile Consecutive blocks in one partition.
     * @param decode_lut Broadcast decoder for nibble formats.
     */
    inline void gemmTwoRowsAVX2OrderedPartitions(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *row0,
        const Q8_1Block *row1,
        float *output0,
        float *output1,
        int chunk,
        int K_blocks,
        int k_tiles,
        int k_blocks_per_tile,
        const __m256i decode_lut)
    {
        if (MoEProjectionNumericalContract::
                oneNativeVNNIBlockPerOrderedPartition(
                    K_blocks *
                    MoEProjectionNumericalContract::
                        native_vnni_values_per_block))
        {
            if (packed.usesNibbleLUT())
            {
                gemm_2row_native_chunk_avx2(
                    packed,
                    row0,
                    row1,
                    output0,
                    output1,
                    chunk,
                    0,
                    K_blocks,
                    decode_lut,
                    /*accumulate=*/false);
            }
            else
            {
                gemm_2row_non_nibble_chunk_avx2(
                    packed,
                    row0,
                    row1,
                    output0,
                    output1,
                    chunk,
                    0,
                    K_blocks,
                    /*accumulate=*/false);
            }
            return;
        }

        if (packed.usesNibbleLUT())
        {
            for (int segment = 0; segment < 8; ++segment)
            {
                __m256 reduced0 = _mm256_setzero_ps();
                __m256 reduced1 = _mm256_setzero_ps();
                for (int k_tile = 0; k_tile < k_tiles; ++k_tile)
                {
                    const int kb_start = k_tile * k_blocks_per_tile;
                    const int kb_end = std::min(
                        kb_start + k_blocks_per_tile, K_blocks);
                    __m256 partial0 = _mm256_setzero_ps();
                    __m256 partial1 = _mm256_setzero_ps();
                    accumulateTwoRowsNibbleSegmentAVX2(
                        packed,
                        row0,
                        row1,
                        chunk,
                        kb_start,
                        kb_end,
                        decode_lut,
                        segment,
                        partial0,
                        partial1);
                    if (k_tile == 0)
                    {
                        reduced0 = partial0;
                        reduced1 = partial1;
                    }
                    else
                    {
                        reduced0 = _mm256_add_ps(reduced0, partial0);
                        reduced1 = _mm256_add_ps(reduced1, partial1);
                    }
                }
                _mm256_storeu_ps(output0 + segment * 8, reduced0);
                _mm256_storeu_ps(output1 + segment * 8, reduced1);
            }
            return;
        }

        alignas(32) float partial0[64];
        alignas(32) float partial1[64];
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile)
        {
            const int kb_start = k_tile * k_blocks_per_tile;
            const int kb_end = std::min(
                kb_start + k_blocks_per_tile, K_blocks);
            gemm_2row_non_nibble_chunk_avx2(
                packed,
                row0,
                row1,
                partial0,
                partial1,
                chunk,
                kb_start,
                kb_end,
                /*accumulate=*/false);
            foldOrderedAVX2Partition(
                output0, partial0, k_tile == 0);
            foldOrderedAVX2Partition(
                output1, partial1, k_tile == 0);
        }
    }

} // namespace llaminar2::cpu::native_vnni
