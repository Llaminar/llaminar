/**
 * @file CPUNativeAVX2Gemv.h
 * @brief AVX2 GEMV/GEMM kernels using emulated VNNI (maddubs+madd pattern).
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

    // =========================================================================
    // AVX2 GEMV: nibble-LUT path (Q4_0, IQ4_NL, Q4_1, IQ4_XS)
    // =========================================================================
    //
    // Processes one 64-column N-chunk using 8 YMM accumulators (8 cols each).
    // Each AVX512 ZMM load (64 bytes) is split into two YMM loads (32 bytes).
    // The nibble decode and VNNI accumulation use AVX2 equivalents.
    // =========================================================================

    inline void gemv_avx2_chunk_native(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk,
        int kb_start,
        int kb_end,
        const __m256i decode_lut)
    {
        // 8 FP32 accumulators covering 64 columns (8 columns each)
        __m256 fp_acc0 = _mm256_setzero_ps();
        __m256 fp_acc1 = _mm256_setzero_ps();
        __m256 fp_acc2 = _mm256_setzero_ps();
        __m256 fp_acc3 = _mm256_setzero_ps();
        __m256 fp_acc4 = _mm256_setzero_ps();
        __m256 fp_acc5 = _mm256_setzero_ps();
        __m256 fp_acc6 = _mm256_setzero_ps();
        __m256 fp_acc7 = _mm256_setzero_ps();

        const __m256i bias_128_i32 = _mm256_set1_epi32(128);
        const __m256i mask_0F = _mm256_set1_epi8(0x0F);

        for (int kb = kb_start; kb < kb_end; ++kb)
        {
            const Q8_1Block &a_blk = A_q8[kb];
            float a_scale = simd::fp16_to_fp32(a_blk.d);
            int16_t a_sum = a_blk.sum_qs;

            // 8 INT32 accumulators
            __m256i int_acc0 = _mm256_setzero_si256();
            __m256i int_acc1 = _mm256_setzero_si256();
            __m256i int_acc2 = _mm256_setzero_si256();
            __m256i int_acc3 = _mm256_setzero_si256();
            __m256i int_acc4 = _mm256_setzero_si256();
            __m256i int_acc5 = _mm256_setzero_si256();
            __m256i int_acc6 = _mm256_setzero_si256();
            __m256i int_acc7 = _mm256_setzero_si256();

            // 4 groups: each covers 4 native bytes per column.
            // Low nibbles → K-elements [g*4..g*4+3]
            // High nibbles → K-elements [g*4+16..g*4+19]
            for (int group = 0; group < 4; ++group)
            {
                // A broadcast for low-nibble sub
                uint8_t a_lo[4];
                a_lo[0] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 0]) + 128);
                a_lo[1] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 1]) + 128);
                a_lo[2] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 2]) + 128);
                a_lo[3] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 3]) + 128);
                int32_t a_lo_i32;
                std::memcpy(&a_lo_i32, a_lo, 4);
                __m256i a_lo_bcast = _mm256_set1_epi32(a_lo_i32);

                // A broadcast for high-nibble sub
                uint8_t a_hi[4];
                a_hi[0] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 16]) + 128);
                a_hi[1] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 17]) + 128);
                a_hi[2] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 18]) + 128);
                a_hi[3] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 19]) + 128);
                int32_t a_hi_i32;
                std::memcpy(&a_hi_i32, a_hi, 4);
                __m256i a_hi_bcast = _mm256_set1_epi32(a_hi_i32);

                // Process 4 ZMM slots, each split into 2 YMM halves
                for (int z = 0; z < 4; ++z)
                {
                    const uint8_t *base = packed.interleavedB(chunk, kb, group, z);
                    __m256i raw_lo = _mm256_load_si256(reinterpret_cast<const __m256i *>(base));
                    __m256i raw_hi = _mm256_load_si256(reinterpret_cast<const __m256i *>(base + 32));

                    // Decode low nibbles
                    __m256i lo_lo = _mm256_shuffle_epi8(decode_lut, _mm256_and_si256(raw_lo, mask_0F));
                    __m256i lo_hi = _mm256_shuffle_epi8(decode_lut, _mm256_and_si256(raw_hi, mask_0F));

                    // Accumulate low nibbles
                    int idx = z * 2;
                    __m256i *acc_ptr = &int_acc0 + idx; // Won't work with stack vars...
                    // Use explicit indexing instead
                    switch (z)
                    {
                    case 0:
                        int_acc0 = isa::avx2_dpbusd_epi32(int_acc0, a_lo_bcast, lo_lo);
                        int_acc1 = isa::avx2_dpbusd_epi32(int_acc1, a_lo_bcast, lo_hi);
                        break;
                    case 1:
                        int_acc2 = isa::avx2_dpbusd_epi32(int_acc2, a_lo_bcast, lo_lo);
                        int_acc3 = isa::avx2_dpbusd_epi32(int_acc3, a_lo_bcast, lo_hi);
                        break;
                    case 2:
                        int_acc4 = isa::avx2_dpbusd_epi32(int_acc4, a_lo_bcast, lo_lo);
                        int_acc5 = isa::avx2_dpbusd_epi32(int_acc5, a_lo_bcast, lo_hi);
                        break;
                    case 3:
                        int_acc6 = isa::avx2_dpbusd_epi32(int_acc6, a_lo_bcast, lo_lo);
                        int_acc7 = isa::avx2_dpbusd_epi32(int_acc7, a_lo_bcast, lo_hi);
                        break;
                    }

                    // Decode high nibbles
                    __m256i hi_lo = _mm256_shuffle_epi8(decode_lut,
                                                        _mm256_and_si256(_mm256_srli_epi16(raw_lo, 4), mask_0F));
                    __m256i hi_hi = _mm256_shuffle_epi8(decode_lut,
                                                        _mm256_and_si256(_mm256_srli_epi16(raw_hi, 4), mask_0F));

                    // Accumulate high nibbles
                    switch (z)
                    {
                    case 0:
                        int_acc0 = isa::avx2_dpbusd_epi32(int_acc0, a_hi_bcast, hi_lo);
                        int_acc1 = isa::avx2_dpbusd_epi32(int_acc1, a_hi_bcast, hi_hi);
                        break;
                    case 1:
                        int_acc2 = isa::avx2_dpbusd_epi32(int_acc2, a_hi_bcast, hi_lo);
                        int_acc3 = isa::avx2_dpbusd_epi32(int_acc3, a_hi_bcast, hi_hi);
                        break;
                    case 2:
                        int_acc4 = isa::avx2_dpbusd_epi32(int_acc4, a_hi_bcast, hi_lo);
                        int_acc5 = isa::avx2_dpbusd_epi32(int_acc5, a_hi_bcast, hi_hi);
                        break;
                    case 3:
                        int_acc6 = isa::avx2_dpbusd_epi32(int_acc6, a_hi_bcast, hi_lo);
                        int_acc7 = isa::avx2_dpbusd_epi32(int_acc7, a_hi_bcast, hi_hi);
                        break;
                    }
                }
            }

            // Bias correction: corrected = int_acc - 128 * comp
            // comp is 64 contiguous INT16; process as 8 groups of 8
            const int16_t *comp_ptr = packed.chunkComp(chunk, kb);
            for (int z = 0; z < 8; ++z)
            {
                __m256i comp = _mm256_cvtepi16_epi32(
                    _mm_load_si128(reinterpret_cast<const __m128i *>(comp_ptr + z * 8)));
                __m256i bias_correction = _mm256_mullo_epi32(bias_128_i32, comp);
                __m256i *acc;
                switch (z)
                {
                case 0:
                    int_acc0 = _mm256_sub_epi32(int_acc0, bias_correction);
                    break;
                case 1:
                    int_acc1 = _mm256_sub_epi32(int_acc1, bias_correction);
                    break;
                case 2:
                    int_acc2 = _mm256_sub_epi32(int_acc2, bias_correction);
                    break;
                case 3:
                    int_acc3 = _mm256_sub_epi32(int_acc3, bias_correction);
                    break;
                case 4:
                    int_acc4 = _mm256_sub_epi32(int_acc4, bias_correction);
                    break;
                case 5:
                    int_acc5 = _mm256_sub_epi32(int_acc5, bias_correction);
                    break;
                case 6:
                    int_acc6 = _mm256_sub_epi32(int_acc6, bias_correction);
                    break;
                case 7:
                    int_acc7 = _mm256_sub_epi32(int_acc7, bias_correction);
                    break;
                }
            }

            // Convert to FP32 and scale: fp_val = int32_val * a_scale * b_scale[n]
            // scales are 64 contiguous FP16; process as 8 groups of 8
            __m256 a_scale_v = _mm256_set1_ps(a_scale);
            const uint16_t *b_scales = packed.chunkScales(chunk, kb);

#define AVX2_SCALE_ACC(IDX, INT_ACC, FP_ACC)                                                      \
    {                                                                                              \
        __m256 cs = _mm256_mul_ps(a_scale_v,                                                       \
                                  _mm256_cvtph_ps(_mm_load_si128(                                  \
                                      reinterpret_cast<const __m128i *>(b_scales + (IDX) * 8))));  \
        FP_ACC = _mm256_fmadd_ps(_mm256_cvtepi32_ps(INT_ACC), cs, FP_ACC);                        \
    }

            AVX2_SCALE_ACC(0, int_acc0, fp_acc0)
            AVX2_SCALE_ACC(1, int_acc1, fp_acc1)
            AVX2_SCALE_ACC(2, int_acc2, fp_acc2)
            AVX2_SCALE_ACC(3, int_acc3, fp_acc3)
            AVX2_SCALE_ACC(4, int_acc4, fp_acc4)
            AVX2_SCALE_ACC(5, int_acc5, fp_acc5)
            AVX2_SCALE_ACC(6, int_acc6, fp_acc6)
            AVX2_SCALE_ACC(7, int_acc7, fp_acc7)
#undef AVX2_SCALE_ACC

            // Asymmetric correction: acc += a_scale * sum_qs * b_min[n]
            if (packed.is_asymmetric)
            {
                const uint16_t *b_mins = packed.chunkMins(chunk, kb);
                __m256 a_corr_v = _mm256_set1_ps(static_cast<float>(a_sum) * a_scale);

#define AVX2_ASYM_CORR(IDX, FP_ACC)                                                             \
    FP_ACC = _mm256_fmadd_ps(a_corr_v,                                                          \
                              _mm256_cvtph_ps(_mm_load_si128(                                    \
                                  reinterpret_cast<const __m128i *>(b_mins + (IDX) * 8))),       \
                              FP_ACC);

                AVX2_ASYM_CORR(0, fp_acc0)
                AVX2_ASYM_CORR(1, fp_acc1)
                AVX2_ASYM_CORR(2, fp_acc2)
                AVX2_ASYM_CORR(3, fp_acc3)
                AVX2_ASYM_CORR(4, fp_acc4)
                AVX2_ASYM_CORR(5, fp_acc5)
                AVX2_ASYM_CORR(6, fp_acc6)
                AVX2_ASYM_CORR(7, fp_acc7)
#undef AVX2_ASYM_CORR
            }
        }

        // Store 64 floats (8 × 8)
        _mm256_storeu_ps(C, fp_acc0);
        _mm256_storeu_ps(C + 8, fp_acc1);
        _mm256_storeu_ps(C + 16, fp_acc2);
        _mm256_storeu_ps(C + 24, fp_acc3);
        _mm256_storeu_ps(C + 32, fp_acc4);
        _mm256_storeu_ps(C + 40, fp_acc5);
        _mm256_storeu_ps(C + 48, fp_acc6);
        _mm256_storeu_ps(C + 56, fp_acc7);
    }

    // =========================================================================
    // AVX2 GEMV: INT8 pre-decoded path (Q5_0, Q5_1, Q6_K, Q3_K, Q2_K, etc.)
    // =========================================================================

    inline void gemv_avx2_chunk_int8(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk,
        int kb_start,
        int kb_end)
    {
        __m256 fp_acc0 = _mm256_setzero_ps();
        __m256 fp_acc1 = _mm256_setzero_ps();
        __m256 fp_acc2 = _mm256_setzero_ps();
        __m256 fp_acc3 = _mm256_setzero_ps();
        __m256 fp_acc4 = _mm256_setzero_ps();
        __m256 fp_acc5 = _mm256_setzero_ps();
        __m256 fp_acc6 = _mm256_setzero_ps();
        __m256 fp_acc7 = _mm256_setzero_ps();

        const __m256i bias_128_i32 = _mm256_set1_epi32(128);

        for (int kb = kb_start; kb < kb_end; ++kb)
        {
            const Q8_1Block &a_blk = A_q8[kb];
            float a_scale = simd::fp16_to_fp32(a_blk.d);
            int16_t a_sum = a_blk.sum_qs;

            __m256i int_acc0 = _mm256_setzero_si256();
            __m256i int_acc1 = _mm256_setzero_si256();
            __m256i int_acc2 = _mm256_setzero_si256();
            __m256i int_acc3 = _mm256_setzero_si256();
            __m256i int_acc4 = _mm256_setzero_si256();
            __m256i int_acc5 = _mm256_setzero_si256();
            __m256i int_acc6 = _mm256_setzero_si256();
            __m256i int_acc7 = _mm256_setzero_si256();

            // 8 groups × 4 K-elements each = 32 K-elements per block
            for (int group = 0; group < 8; ++group)
            {
                // A broadcast: convert signed→unsigned via +128
                uint8_t a_u8[4];
                a_u8[0] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 0]) + 128);
                a_u8[1] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 1]) + 128);
                a_u8[2] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 2]) + 128);
                a_u8[3] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 3]) + 128);
                int32_t a_i32;
                std::memcpy(&a_i32, a_u8, 4);
                __m256i a_bcast = _mm256_set1_epi32(a_i32);

                // Process 4 ZMM slots, each split into 2 YMM halves
                for (int z = 0; z < 4; ++z)
                {
                    const uint8_t *base = packed.interleavedB(chunk, kb, group, z);
                    __m256i b_lo = _mm256_load_si256(reinterpret_cast<const __m256i *>(base));
                    __m256i b_hi = _mm256_load_si256(reinterpret_cast<const __m256i *>(base + 32));

                    switch (z)
                    {
                    case 0:
                        int_acc0 = isa::avx2_dpbusd_epi32(int_acc0, a_bcast, b_lo);
                        int_acc1 = isa::avx2_dpbusd_epi32(int_acc1, a_bcast, b_hi);
                        break;
                    case 1:
                        int_acc2 = isa::avx2_dpbusd_epi32(int_acc2, a_bcast, b_lo);
                        int_acc3 = isa::avx2_dpbusd_epi32(int_acc3, a_bcast, b_hi);
                        break;
                    case 2:
                        int_acc4 = isa::avx2_dpbusd_epi32(int_acc4, a_bcast, b_lo);
                        int_acc5 = isa::avx2_dpbusd_epi32(int_acc5, a_bcast, b_hi);
                        break;
                    case 3:
                        int_acc6 = isa::avx2_dpbusd_epi32(int_acc6, a_bcast, b_lo);
                        int_acc7 = isa::avx2_dpbusd_epi32(int_acc7, a_bcast, b_hi);
                        break;
                    }
                }
            }

            // Bias correction
            const int16_t *comp_ptr = packed.chunkComp(chunk, kb);

#define AVX2_BIAS_CORRECT(IDX, INT_ACC)                                                           \
    {                                                                                              \
        __m256i comp = _mm256_cvtepi16_epi32(                                                      \
            _mm_load_si128(reinterpret_cast<const __m128i *>(comp_ptr + (IDX) * 8)));              \
        INT_ACC = _mm256_sub_epi32(INT_ACC, _mm256_mullo_epi32(bias_128_i32, comp));              \
    }

            AVX2_BIAS_CORRECT(0, int_acc0)
            AVX2_BIAS_CORRECT(1, int_acc1)
            AVX2_BIAS_CORRECT(2, int_acc2)
            AVX2_BIAS_CORRECT(3, int_acc3)
            AVX2_BIAS_CORRECT(4, int_acc4)
            AVX2_BIAS_CORRECT(5, int_acc5)
            AVX2_BIAS_CORRECT(6, int_acc6)
            AVX2_BIAS_CORRECT(7, int_acc7)
#undef AVX2_BIAS_CORRECT

            // Scale conversion
            __m256 a_scale_v = _mm256_set1_ps(a_scale);
            const uint16_t *b_scales = packed.chunkScales(chunk, kb);

#define AVX2_SCALE_ACC(IDX, INT_ACC, FP_ACC)                                                      \
    {                                                                                              \
        __m256 cs = _mm256_mul_ps(a_scale_v,                                                       \
                                  _mm256_cvtph_ps(_mm_load_si128(                                  \
                                      reinterpret_cast<const __m128i *>(b_scales + (IDX) * 8))));  \
        FP_ACC = _mm256_fmadd_ps(_mm256_cvtepi32_ps(INT_ACC), cs, FP_ACC);                        \
    }

            AVX2_SCALE_ACC(0, int_acc0, fp_acc0)
            AVX2_SCALE_ACC(1, int_acc1, fp_acc1)
            AVX2_SCALE_ACC(2, int_acc2, fp_acc2)
            AVX2_SCALE_ACC(3, int_acc3, fp_acc3)
            AVX2_SCALE_ACC(4, int_acc4, fp_acc4)
            AVX2_SCALE_ACC(5, int_acc5, fp_acc5)
            AVX2_SCALE_ACC(6, int_acc6, fp_acc6)
            AVX2_SCALE_ACC(7, int_acc7, fp_acc7)
#undef AVX2_SCALE_ACC

            // Asymmetric correction
            if (packed.is_asymmetric)
            {
                const uint16_t *b_mins = packed.chunkMins(chunk, kb);
                __m256 a_corr_v = _mm256_set1_ps(static_cast<float>(a_sum) * a_scale);

#define AVX2_ASYM_CORR(IDX, FP_ACC)                                                             \
    FP_ACC = _mm256_fmadd_ps(a_corr_v,                                                          \
                              _mm256_cvtph_ps(_mm_load_si128(                                    \
                                  reinterpret_cast<const __m128i *>(b_mins + (IDX) * 8))),       \
                              FP_ACC);

                AVX2_ASYM_CORR(0, fp_acc0)
                AVX2_ASYM_CORR(1, fp_acc1)
                AVX2_ASYM_CORR(2, fp_acc2)
                AVX2_ASYM_CORR(3, fp_acc3)
                AVX2_ASYM_CORR(4, fp_acc4)
                AVX2_ASYM_CORR(5, fp_acc5)
                AVX2_ASYM_CORR(6, fp_acc6)
                AVX2_ASYM_CORR(7, fp_acc7)
#undef AVX2_ASYM_CORR
            }
        }

        _mm256_storeu_ps(C, fp_acc0);
        _mm256_storeu_ps(C + 8, fp_acc1);
        _mm256_storeu_ps(C + 16, fp_acc2);
        _mm256_storeu_ps(C + 24, fp_acc3);
        _mm256_storeu_ps(C + 32, fp_acc4);
        _mm256_storeu_ps(C + 40, fp_acc5);
        _mm256_storeu_ps(C + 48, fp_acc6);
        _mm256_storeu_ps(C + 56, fp_acc7);
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
        const bool use_nibble_lut = packed.is_nibble_lut;

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
                    gemv_avx2_chunk_int8(packed, A_q8, tmp, chunk, 0, K_blocks);
                std::memcpy(C + n_start, tmp, n_cols * sizeof(float));
            }
            else
            {
                if (use_nibble_lut)
                    gemv_avx2_chunk_native(packed, A_q8, C + n_start, chunk, 0, K_blocks, decode_lut);
                else
                    gemv_avx2_chunk_int8(packed, A_q8, C + n_start, chunk, 0, K_blocks);
            }
        }
    }

    // =========================================================================
    // AVX2 2-Row GEMM Microkernels (share B loads across 2 M rows)
    // =========================================================================

    /**
     * @brief AVX2 2-row nibble-LUT GEMM microkernel for one 64-col chunk.
     *
     * Uses 16 YMM FP accumulators (8 per row) + 16 YMM INT accumulators
     * This exceeds 16 YMM registers, so the compiler will spill some to stack.
     * For correctness-first AVX2, this is acceptable.
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
        // 8 FP accumulators per row = 16 total
        __m256 fp0[8], fp1[8];
        for (int i = 0; i < 8; ++i)
        {
            fp0[i] = accumulate ? _mm256_loadu_ps(C_row0 + i * 8) : _mm256_setzero_ps();
            fp1[i] = accumulate ? _mm256_loadu_ps(C_row1 + i * 8) : _mm256_setzero_ps();
        }

        const __m256i bias_128_i32 = _mm256_set1_epi32(128);
        const __m256i mask_0F = _mm256_set1_epi8(0x0F);

        for (int kb = kb_start; kb < kb_end; ++kb)
        {
            const Q8_1Block &a0 = A_q8_row0[kb];
            const Q8_1Block &a1 = A_q8_row1[kb];
            float a0_scale = simd::fp16_to_fp32(a0.d);
            float a1_scale = simd::fp16_to_fp32(a1.d);

            __m256i ia0[8], ia1[8];
            for (int i = 0; i < 8; ++i)
            {
                ia0[i] = _mm256_setzero_si256();
                ia1[i] = _mm256_setzero_si256();
            }

            for (int group = 0; group < 4; ++group)
            {
                // Row 0 & 1 A broadcasts for low nibbles
                auto make_a_bcast = [](const Q8_1Block &blk, int base_idx) -> __m256i
                {
                    uint8_t vals[4];
                    vals[0] = static_cast<uint8_t>(static_cast<int16_t>(blk.qs[base_idx + 0]) + 128);
                    vals[1] = static_cast<uint8_t>(static_cast<int16_t>(blk.qs[base_idx + 1]) + 128);
                    vals[2] = static_cast<uint8_t>(static_cast<int16_t>(blk.qs[base_idx + 2]) + 128);
                    vals[3] = static_cast<uint8_t>(static_cast<int16_t>(blk.qs[base_idx + 3]) + 128);
                    int32_t v;
                    std::memcpy(&v, vals, 4);
                    return _mm256_set1_epi32(v);
                };

                __m256i a0_lo = make_a_bcast(a0, group * 4);
                __m256i a1_lo = make_a_bcast(a1, group * 4);
                __m256i a0_hi = make_a_bcast(a0, group * 4 + 16);
                __m256i a1_hi = make_a_bcast(a1, group * 4 + 16);

                for (int z = 0; z < 4; ++z)
                {
                    const uint8_t *base = packed.interleavedB(chunk, kb, group, z);
                    __m256i raw_lo = _mm256_load_si256(reinterpret_cast<const __m256i *>(base));
                    __m256i raw_hi = _mm256_load_si256(reinterpret_cast<const __m256i *>(base + 32));

                    // Decode nibbles
                    __m256i lo_dec_lo = _mm256_shuffle_epi8(decode_lut, _mm256_and_si256(raw_lo, mask_0F));
                    __m256i lo_dec_hi = _mm256_shuffle_epi8(decode_lut, _mm256_and_si256(raw_hi, mask_0F));
                    __m256i hi_dec_lo = _mm256_shuffle_epi8(decode_lut,
                                                            _mm256_and_si256(_mm256_srli_epi16(raw_lo, 4), mask_0F));
                    __m256i hi_dec_hi = _mm256_shuffle_epi8(decode_lut,
                                                            _mm256_and_si256(_mm256_srli_epi16(raw_hi, 4), mask_0F));

                    int idx = z * 2;
                    // Row 0 accumulation
                    ia0[idx] = isa::avx2_dpbusd_epi32(ia0[idx], a0_lo, lo_dec_lo);
                    ia0[idx + 1] = isa::avx2_dpbusd_epi32(ia0[idx + 1], a0_lo, lo_dec_hi);
                    ia0[idx] = isa::avx2_dpbusd_epi32(ia0[idx], a0_hi, hi_dec_lo);
                    ia0[idx + 1] = isa::avx2_dpbusd_epi32(ia0[idx + 1], a0_hi, hi_dec_hi);

                    // Row 1 accumulation (same B data, different A)
                    ia1[idx] = isa::avx2_dpbusd_epi32(ia1[idx], a1_lo, lo_dec_lo);
                    ia1[idx + 1] = isa::avx2_dpbusd_epi32(ia1[idx + 1], a1_lo, lo_dec_hi);
                    ia1[idx] = isa::avx2_dpbusd_epi32(ia1[idx], a1_hi, hi_dec_lo);
                    ia1[idx + 1] = isa::avx2_dpbusd_epi32(ia1[idx + 1], a1_hi, hi_dec_hi);
                }
            }

            // Bias correction + scale (shared comp/scales, per-row a_scale)
            const int16_t *comp_ptr = packed.chunkComp(chunk, kb);
            const uint16_t *b_scales = packed.chunkScales(chunk, kb);

            for (int z = 0; z < 8; ++z)
            {
                __m256i comp = _mm256_cvtepi16_epi32(
                    _mm_load_si128(reinterpret_cast<const __m128i *>(comp_ptr + z * 8)));
                __m256i bc = _mm256_mullo_epi32(bias_128_i32, comp);
                ia0[z] = _mm256_sub_epi32(ia0[z], bc);
                ia1[z] = _mm256_sub_epi32(ia1[z], bc);

                __m256 bs = _mm256_cvtph_ps(
                    _mm_load_si128(reinterpret_cast<const __m128i *>(b_scales + z * 8)));

                __m256 cs0 = _mm256_mul_ps(_mm256_set1_ps(a0_scale), bs);
                __m256 cs1 = _mm256_mul_ps(_mm256_set1_ps(a1_scale), bs);
                fp0[z] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia0[z]), cs0, fp0[z]);
                fp1[z] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia1[z]), cs1, fp1[z]);
            }

            if (packed.is_asymmetric)
            {
                const uint16_t *b_mins = packed.chunkMins(chunk, kb);
                __m256 corr0 = _mm256_set1_ps(static_cast<float>(a0.sum_qs) * a0_scale);
                __m256 corr1 = _mm256_set1_ps(static_cast<float>(a1.sum_qs) * a1_scale);
                for (int z = 0; z < 8; ++z)
                {
                    __m256 bm = _mm256_cvtph_ps(
                        _mm_load_si128(reinterpret_cast<const __m128i *>(b_mins + z * 8)));
                    fp0[z] = _mm256_fmadd_ps(corr0, bm, fp0[z]);
                    fp1[z] = _mm256_fmadd_ps(corr1, bm, fp1[z]);
                }
            }
        }

        for (int i = 0; i < 8; ++i)
        {
            _mm256_storeu_ps(C_row0 + i * 8, fp0[i]);
            _mm256_storeu_ps(C_row1 + i * 8, fp1[i]);
        }
    }

    /**
     * @brief AVX2 2-row INT8 pre-decoded GEMM microkernel for one 64-col chunk.
     */
    inline void gemm_2row_int8_chunk_avx2(
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
        __m256 fp0[8], fp1[8];
        for (int i = 0; i < 8; ++i)
        {
            fp0[i] = accumulate ? _mm256_loadu_ps(C_row0 + i * 8) : _mm256_setzero_ps();
            fp1[i] = accumulate ? _mm256_loadu_ps(C_row1 + i * 8) : _mm256_setzero_ps();
        }

        const __m256i bias_128_i32 = _mm256_set1_epi32(128);

        for (int kb = kb_start; kb < kb_end; ++kb)
        {
            const Q8_1Block &a0 = A_q8_row0[kb];
            const Q8_1Block &a1 = A_q8_row1[kb];
            float a0_scale = simd::fp16_to_fp32(a0.d);
            float a1_scale = simd::fp16_to_fp32(a1.d);

            __m256i ia0[8], ia1[8];
            for (int i = 0; i < 8; ++i)
            {
                ia0[i] = _mm256_setzero_si256();
                ia1[i] = _mm256_setzero_si256();
            }

            for (int group = 0; group < 8; ++group)
            {
                // GPR-only A-prep: XOR with 0x80 converts signed→unsigned
                int32_t raw0, raw1;
                std::memcpy(&raw0, &a0.qs[group * 4], 4);
                std::memcpy(&raw1, &a1.qs[group * 4], 4);
                __m256i a0_bc = _mm256_set1_epi32(static_cast<int32_t>(
                    static_cast<uint32_t>(raw0) ^ 0x80808080u));
                __m256i a1_bc = _mm256_set1_epi32(static_cast<int32_t>(
                    static_cast<uint32_t>(raw1) ^ 0x80808080u));

                for (int z = 0; z < 4; ++z)
                {
                    const uint8_t *base = packed.interleavedB(chunk, kb, group, z);
                    __m256i b_lo = _mm256_load_si256(reinterpret_cast<const __m256i *>(base));
                    __m256i b_hi = _mm256_load_si256(reinterpret_cast<const __m256i *>(base + 32));

                    int idx = z * 2;
                    ia0[idx] = isa::avx2_dpbusd_epi32(ia0[idx], a0_bc, b_lo);
                    ia0[idx + 1] = isa::avx2_dpbusd_epi32(ia0[idx + 1], a0_bc, b_hi);
                    ia1[idx] = isa::avx2_dpbusd_epi32(ia1[idx], a1_bc, b_lo);
                    ia1[idx + 1] = isa::avx2_dpbusd_epi32(ia1[idx + 1], a1_bc, b_hi);
                }
            }

            // Bias correction + scale
            const int16_t *comp_ptr = packed.chunkComp(chunk, kb);
            const uint16_t *b_scales = packed.chunkScales(chunk, kb);

            for (int z = 0; z < 8; ++z)
            {
                __m256i comp = _mm256_cvtepi16_epi32(
                    _mm_load_si128(reinterpret_cast<const __m128i *>(comp_ptr + z * 8)));
                __m256i bc = _mm256_mullo_epi32(bias_128_i32, comp);
                ia0[z] = _mm256_sub_epi32(ia0[z], bc);
                ia1[z] = _mm256_sub_epi32(ia1[z], bc);

                __m256 bs = _mm256_cvtph_ps(
                    _mm_load_si128(reinterpret_cast<const __m128i *>(b_scales + z * 8)));

                fp0[z] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia0[z]),
                                          _mm256_mul_ps(_mm256_set1_ps(a0_scale), bs), fp0[z]);
                fp1[z] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia1[z]),
                                          _mm256_mul_ps(_mm256_set1_ps(a1_scale), bs), fp1[z]);
            }

            if (packed.is_asymmetric)
            {
                const uint16_t *b_mins = packed.chunkMins(chunk, kb);
                __m256 corr0 = _mm256_set1_ps(static_cast<float>(a0.sum_qs) * a0_scale);
                __m256 corr1 = _mm256_set1_ps(static_cast<float>(a1.sum_qs) * a1_scale);
                for (int z = 0; z < 8; ++z)
                {
                    __m256 bm = _mm256_cvtph_ps(
                        _mm_load_si128(reinterpret_cast<const __m128i *>(b_mins + z * 8)));
                    fp0[z] = _mm256_fmadd_ps(corr0, bm, fp0[z]);
                    fp1[z] = _mm256_fmadd_ps(corr1, bm, fp1[z]);
                }
            }
        }

        for (int i = 0; i < 8; ++i)
        {
            _mm256_storeu_ps(C_row0 + i * 8, fp0[i]);
            _mm256_storeu_ps(C_row1 + i * 8, fp1[i]);
        }
    }

} // namespace llaminar2::cpu::native_vnni
