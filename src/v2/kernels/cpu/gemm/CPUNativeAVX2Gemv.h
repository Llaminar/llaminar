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
#include "CPUNativeVNNIFP16.h"
#include "CPUNativeVNNIQ6Kernels.h"
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
        const __m256i bias_128_i32 = _mm256_set1_epi32(128);
        const __m256i mask_0F = _mm256_set1_epi8(0x0F);
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
                    /*
                     * The even/odd-byte products inside `avx2_dpbusd_epi32`
                     * have a short enough live set for two independent nibble
                     * chains when this fixed group loop is not unrolled. This
                     * restores dot-product ILP without exceeding AVX2's sixteen
                     * architectural registers.
                     */
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
                const __m256 combined_scale = _mm256_mul_ps(
                    _mm256_set1_ps(nativeVNNIFP16ScaleToFP32(activation.d)),
                    weight_scale);
                fp_accumulator = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(corrected),
                    combined_scale,
                    fp_accumulator);

                if (packed.is_asymmetric)
                {
                    const __m256 weight_min = _mm256_cvtph_ps(
                        _mm_load_si128(reinterpret_cast<const __m128i *>(
                            packed.chunkMins(chunk, kb) + segment * 8)));
                    const float correction =
                        static_cast<float>(activation.sum_qs) *
                        nativeVNNIFP16ScaleToFP32(activation.d);
                    fp_accumulator = _mm256_fmadd_ps(
                        _mm256_set1_ps(correction),
                        weight_min,
                        fp_accumulator);
                }
            }
            _mm256_storeu_ps(C + segment * 8, fp_accumulator);
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
                fp_accumulator = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(corrected),
                    _mm256_mul_ps(
                        _mm256_set1_ps(activation_scale), weight_scale),
                    fp_accumulator);
                if (packed.is_asymmetric)
                {
                    const __m256 weight_min = _mm256_cvtph_ps(
                        _mm_load_si128(reinterpret_cast<const __m128i *>(
                            packed.chunkMins(chunk, kb) + segment * 8)));
                    fp_accumulator = _mm256_fmadd_ps(
                        _mm256_set1_ps(
                            static_cast<float>(activation.sum_qs) *
                            activation_scale),
                        weight_min,
                        fp_accumulator);
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
        const __m256i bias_128_i32 = _mm256_set1_epi32(128);
        const __m256i mask_0F = _mm256_set1_epi8(0x0F);
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
                fp0 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(corrected0),
                    _mm256_mul_ps(_mm256_set1_ps(a0_scale), weight_scale),
                    fp0);
                fp1 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(corrected1),
                    _mm256_mul_ps(_mm256_set1_ps(a1_scale), weight_scale),
                    fp1);
                if (packed.is_asymmetric)
                {
                    const __m256 weight_min = _mm256_cvtph_ps(
                        _mm_load_si128(reinterpret_cast<const __m128i *>(
                            packed.chunkMins(chunk, kb) + segment * 8)));
                    fp0 = _mm256_fmadd_ps(
                        _mm256_set1_ps(
                            static_cast<float>(a0.sum_qs) * a0_scale),
                        weight_min,
                        fp0);
                    fp1 = _mm256_fmadd_ps(
                        _mm256_set1_ps(
                            static_cast<float>(a1.sum_qs) * a1_scale),
                        weight_min,
                        fp1);
                }
            }
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
                fp0 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(corrected0),
                    _mm256_mul_ps(_mm256_set1_ps(a0_scale), weight_scale),
                    fp0);
                fp1 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(corrected1),
                    _mm256_mul_ps(_mm256_set1_ps(a1_scale), weight_scale),
                    fp1);
                if (packed.is_asymmetric)
                {
                    const __m256 weight_min = _mm256_cvtph_ps(
                        _mm_load_si128(reinterpret_cast<const __m128i *>(
                            packed.chunkMins(chunk, kb) + segment * 8)));
                    fp0 = _mm256_fmadd_ps(
                        _mm256_set1_ps(
                            static_cast<float>(a0.sum_qs) * a0_scale),
                        weight_min,
                        fp0);
                    fp1 = _mm256_fmadd_ps(
                        _mm256_set1_ps(
                            static_cast<float>(a1.sum_qs) * a1_scale),
                        weight_min,
                        fp1);
                }
            }
            _mm256_storeu_ps(C_row0 + segment * 8, fp0);
            _mm256_storeu_ps(C_row1 + segment * 8, fp1);
        }
    }

} // namespace llaminar2::cpu::native_vnni
