/**
 * @file CPUNativeVNNIGemv.h
 * @brief Optimized M=1 GEMV kernels for native-interleaved VNNI weights + AVX-512.
 *
 * ## Packed VNNI Path (all deferred formats: Q4_0, IQ4_NL, Q4_1, Q8_0, Q5_0, Q5_1)
 *
 * Activations are quantized FP32→Q8_1, weights are pre-packed into VNNI
 * interleaved order. Inner loop uses vpdpbusd (INT8 dot product).
 *
 * Entry points:
 *   - gemv_native_vnni()           — quantize FP32→Q8_1 + dispatch to packed VNNI GEMV
 *
 * ## Packed VNNI Architecture
 *
 * The weight packer stores native payload bytes (Q4_0 nibbles, IQ4_NL nibbles)
 * in VNNI-interleaved order at pack time:
 *   native_interleaved: [N_chunks][bpr][4 groups][4 ZMMs][64 bytes]
 *
 * Each 64-byte ZMM holds 16 columns × 4 consecutive native bytes.
 * Total weight memory = native payload size (zero expansion).
 *
 * At runtime, AVX-512 vpshufb decodes 4-bit nibbles to signed INT8 directly
 * in VNNI lane order. This gives native-size bandwidth (0.5 byte/element
 * for Q4_0) with vectorized VNNI accumulation — matching the GPU approach.
 *
 * ## Inner Loop (64 columns per N-chunk, 32 K-elements per block)
 *
 * Per K-block:
 *   4 groups × (4 ZMM loads + 8 vpshufb + 8 vpdpbusd) = 16 loads + 32 decode + 32 VNNI
 *   Memory traffic: 1024 bytes (native size — half of pre-decoded INT8)
 *
 * Post-K-block:
 *   4 comp loads + 4 mullo + 4 sub (bias correction)
 *   4 scale loads + 4 mul + 4 fmadd (FP32 accumulation)
 *
 * ## Cache-Aware Tiling
 *
 * Uses NativeVNNITileConfig (from CPUNativeVNNITileConfig.h) to select
 * N-block size based on detected L1/L2/L3 and shape category.
 */

#pragma once

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <omp.h>

#include "CPUNativeVNNIDecode.h"
#include "CPUNativeVNNIWeightPacker.h"
#include "CPUNativeVNNITileConfig.h"
#include "tensors/AlignedVector.h"
#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"
#include "utils/CPUFeatures.h"
#include "utils/OpenMPUtils.h"

// AVX2 GEMV/GEMM kernels (same packed weight format, uses emulated VNNI)
#include "CPUNativeAVX2Gemv.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNIVerifierRowsPolicyGenerated.inc"

namespace llaminar2::cpu::native_vnni
{

    // =========================================================================
    // ISA dispatch enum for runtime-selectable GEMV/GEMM paths
    // =========================================================================
    //
    // Both AVX512 and AVX2 paths are always compiled (no #ifdef gates around
    // the dispatch logic). Tests can force either path on AVX512 hardware for
    // parity comparison. AUTO selects the best available at runtime.
    // =========================================================================

    enum class ISAPath
    {
        AUTO,   // Runtime detection: AVX512-VNNI if available, else AVX2
        AVX512, // Force AVX512-VNNI path (only valid on AVX512 hardware)
        AVX2,   // Force AVX2 emulated-VNNI path
        SCALAR  // Force scalar decode path; used as the correctness floor
    };

    /**
     * @brief Runtime policy for grouped verifier-row CPU NativeVNNI kernels.
     *
     * Pairwise uses the two-row chunk kernels as the active floor for M=2..4,
     * with an odd tail row when M=3. WideRows uses the verifier-specialized
     * three-row or four-row AVX512 kernels that share each decoded B chunk
     * across more rows.  The generated table is trained from strict
     * decode-equivalent microbench CSVs.  Shapes that have not been trained use
     * Pairwise, the conservative economy floor, because the wide-row kernels
     * are only safe to promote once a shape/codebook row proves faster than
     * serial decode in the trainer.
     */
    enum class VerifierRowsPolicy
    {
        /**
         * Use the generated production selector.  Perf harnesses may pass an
         * explicit policy to compare candidates on the same k-tiled path, but
         * normal inference should leave this as Auto.
         */
        Auto,
        Pairwise,
        WideRows,
    };

    inline VerifierRowsPolicy selectVerifierRowsPolicy(
        const CPUNativeVNNIPackedWeights &packed,
        int M,
        int N,
        int K)
    {
        generated::CPUNativeVNNIVerifierRowsPolicy generated_policy{};
        if (generated::selectCPUNativeVNNIVerifierRowsGeneratedPolicy(
                packed.codebook_id,
                M,
                N,
                K,
                generated_policy))
        {
            return generated_policy ==
                       generated::CPUNativeVNNIVerifierRowsPolicy::WideRows
                       ? VerifierRowsPolicy::WideRows
                       : VerifierRowsPolicy::Pairwise;
        }

        return VerifierRowsPolicy::Pairwise;
    }

    // =========================================================================
    // Scalar reference GEMV (any format, for correctness verification)
    // =========================================================================

    /**
     * @brief Scalar NativeVNNI GEMV for correctness reference.
     *
     * For nibble-LUT formats (Q4_0, IQ4_NL, Q4_1, IQ4_XS): decodes from payload.
     * For INT8 pre-decoded formats: reads from int8_flat buffer.
     */
    inline void gemv_native_vnni_scalar(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int N,
        int K_blocks)
    {
        const int N_chunks = (N + 63) / 64;

        for (int chunk = 0; chunk < N_chunks; ++chunk)
        {
            const int n_start = chunk * 64;
            const int n_end = std::min(n_start + 64, N);

            for (int n_local = 0; n_local < (n_end - n_start); ++n_local)
            {
                const int n = n_start + n_local;
                float acc = 0.0f;

                for (int kb = 0; kb < K_blocks; ++kb)
                {
                    const Q8_1Block &a_blk = A_q8[kb];
                    float a_scale = simd::fp16_to_fp32(a_blk.d);

                    int8_t b_vals[32];
                    if (packed.is_nibble_lut)
                    {
                        const uint8_t *payload = packed.blockPayload(chunk, kb, n_local);
                        decode_native_block(packed.codebook_id, payload, b_vals);
                    }
                    else
                    {
                        if (!packed.int8_flat.empty())
                        {
                            std::memcpy(b_vals, packed.blockInt8(chunk, kb, n_local), 32);
                        }
                        else
                        {
                            /*
                             * Modern packed weights release int8_flat after
                             * interleaving to avoid carrying two full decoded
                             * copies.  Reconstruct the scalar reference block
                             * from the same native_interleaved layout used by
                             * AVX2/AVX512: group covers four K values, and z
                             * selects the 16-column lane group inside a chunk.
                             */
                            const int z = n_local / 16;
                            const int lane = n_local % 16;
                            for (int group = 0; group < 8; ++group)
                            {
                                const auto *group_data = reinterpret_cast<const int8_t *>(
                                    packed.interleavedB(chunk, kb, group, z));
                                const int src = lane * 4;
                                const int dst = group * 4;
                                b_vals[dst + 0] = group_data[src + 0];
                                b_vals[dst + 1] = group_data[src + 1];
                                b_vals[dst + 2] = group_data[src + 2];
                                b_vals[dst + 3] = group_data[src + 3];
                            }
                        }
                    }

                    float b_scale = packed.blockScale(chunk, kb, n_local);
                    float b_min = packed.blockMin(chunk, kb, n_local);

                    int32_t dot = 0;
                    int32_t b_comp = 0;
                    for (int i = 0; i < 32; ++i)
                    {
                        uint8_t a_u8 = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[i]) + 128);
                        dot += static_cast<int32_t>(a_u8) * static_cast<int32_t>(b_vals[i]);
                        b_comp += b_vals[i];
                    }

                    int32_t corrected = dot - 128 * b_comp;
                    acc += static_cast<float>(corrected) * a_scale * b_scale;

                    if (packed.is_asymmetric && b_min != 0.0f)
                    {
                        acc += static_cast<float>(a_blk.sum_qs) * a_scale * b_min;
                    }
                }

                C[n] = acc;
            }
        }
    }

    // =========================================================================
    // AVX-512 VNNI GEMV with native-interleaved weights (optimized hot path)
    // =========================================================================
    //
    // The weight packer stores native payload bytes in VNNI-interleaved order:
    //   native_interleaved: [N_chunks][bpr][4 groups][4 ZMMs][64 bytes]
    //
    // Each 64-byte ZMM holds 16 columns × 4 consecutive native bytes.
    // At runtime, vpshufb decodes nibbles→INT8 directly in VNNI lane order:
    //   - Low nibbles  → K-elements [group*4 .. group*4+3]
    //   - High nibbles → K-elements [group*4+16 .. group*4+19]
    //
    // Memory traffic per K-block per 64-col chunk: 1024 bytes (= native size!)
    // vs 2048 bytes for the old pre-decoded INT8 path.
    // =========================================================================

    // Decode LUT tables for nibble→INT8 conversion via vpshufb.
    // Each 16-entry table maps a 4-bit nibble to a signed INT8 value.
    // Q4_0: nibble → (nibble - 8) = [-8, -7, ..., +7]
    alignas(16) static constexpr int8_t Q4_0_DECODE_LUT[16] = {
        -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};
    // IQ4_NL / IQ4_XS: nibble → kvalues_iq4nl[nibble]
    alignas(16) static constexpr int8_t IQ4_NL_DECODE_LUT[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    // Q4_1: nibble → nibble (unsigned identity [0..15])
    alignas(16) static constexpr int8_t Q4_1_DECODE_LUT[16] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)

    /**
     * @brief Build the 512-bit decode LUT for a given codebook_id.
     *
     * Broadcasts a 16-byte LUT to all four 128-bit lanes of a ZMM register.
     * Used with vpshufb to decode 4-bit nibbles to signed INT8 in one instruction.
     * Only valid for nibble-LUT formats (Q4_0, IQ4_NL, Q4_1, IQ4_XS).
     */
    inline __m512i build_decode_lut(uint8_t codebook_id)
    {
        const int8_t *lut_data;
        switch (codebook_id)
        {
        case 4: // IQ4_NL / IQ4_XS (both use kvalues_iq4nl LUT)
            lut_data = IQ4_NL_DECODE_LUT;
            break;
        case 5: // Q4_1
            lut_data = Q4_1_DECODE_LUT;
            break;
        default: // Q4_0 (codebook 0)
            lut_data = Q4_0_DECODE_LUT;
            break;
        }
        __m128i lut_128 = _mm_load_si128(reinterpret_cast<const __m128i *>(lut_data));
        return _mm512_broadcast_i32x4(lut_128);
    }

    enum class NibbleDecodeKind : uint8_t
    {
        LUT,
        Q4_0_LINEAR,
        Q4_1_IDENTITY,
    };

    /**
     * @brief Classify nibble decode math for AVX512 NativeVNNI kernels.
     *
     * Q4_0 and Q4_1-family payloads are linear nibble mappings, so they can be
     * decoded with byte arithmetic instead of `vpshufb`.  IQ4 formats keep the
     * lookup-table path because their codebook is non-linear.  This is a
     * format-semantic choice, not a dispatch fallback: every branch is still
     * mathematically identical to the scalar NativeVNNI decode.
     */
    inline NibbleDecodeKind nibbleDecodeKind(uint8_t codebook_id)
    {
        switch (codebook_id)
        {
        case 0:
            return NibbleDecodeKind::Q4_0_LINEAR;
        case 5:
            return NibbleDecodeKind::Q4_1_IDENTITY;
        default:
            return NibbleDecodeKind::LUT;
        }
    }

    /**
     * @brief Pack four signed Q8_1 activation bytes exactly like M=1 GEMV.
     *
     * The VNNI dot instruction consumes unsigned activation bytes.  The decode
     * GEMV path converts each signed Q8_1 byte with `int16(q) + 128` before
     * broadcasting the four-byte word.  Multi-row verifier kernels must use the
     * same helper instead of clever bit tricks: even mathematically equivalent
     * shortcuts make it harder to reason about strict decode equivalence when
     * a verifier row later publishes KV/GDN state.
     */
    inline int32_t pack_q8_1_unsigned_word(const Q8_1Block &block, int base_idx)
    {
        uint8_t vals[4];
        vals[0] = static_cast<uint8_t>(static_cast<int16_t>(block.qs[base_idx + 0]) + 128);
        vals[1] = static_cast<uint8_t>(static_cast<int16_t>(block.qs[base_idx + 1]) + 128);
        vals[2] = static_cast<uint8_t>(static_cast<int16_t>(block.qs[base_idx + 2]) + 128);
        vals[3] = static_cast<uint8_t>(static_cast<int16_t>(block.qs[base_idx + 3]) + 128);
        int32_t packed_word;
        std::memcpy(&packed_word, vals, sizeof(packed_word));
        return packed_word;
    }

    /**
     * @brief Decode low nibbles to signed INT8 VNNI lanes.
     */
    inline __m512i decode_low_nibbles_avx512(
        __m512i raw,
        __m512i decode_lut,
        NibbleDecodeKind kind,
        __m512i mask_0F,
        __m512i q4_zero_offset)
    {
        const __m512i lo = _mm512_and_si512(raw, mask_0F);
        switch (kind)
        {
        case NibbleDecodeKind::Q4_0_LINEAR:
            return _mm512_sub_epi8(lo, q4_zero_offset);
        case NibbleDecodeKind::Q4_1_IDENTITY:
            return lo;
        case NibbleDecodeKind::LUT:
        default:
            return _mm512_shuffle_epi8(decode_lut, lo);
        }
    }

    /**
     * @brief Decode high nibbles to signed INT8 VNNI lanes.
     */
    inline __m512i decode_high_nibbles_avx512(
        __m512i raw,
        __m512i decode_lut,
        NibbleDecodeKind kind,
        __m512i mask_0F,
        __m512i q4_zero_offset)
    {
        const __m512i hi =
            _mm512_and_si512(_mm512_srli_epi16(raw, 4), mask_0F);
        switch (kind)
        {
        case NibbleDecodeKind::Q4_0_LINEAR:
            return _mm512_sub_epi8(hi, q4_zero_offset);
        case NibbleDecodeKind::Q4_1_IDENTITY:
            return hi;
        case NibbleDecodeKind::LUT:
        default:
            return _mm512_shuffle_epi8(decode_lut, hi);
        }
    }

    /**
     * @brief AVX-512 VNNI GEMV for one N-chunk of 64 columns.
     *
     * Reads native-interleaved bytes (1024 B/K-block = native payload size),
     * decodes nibbles to INT8 via vpshufb, and feeds directly into vpdpbusd.
     *
     * Per K-block inner loop (4 groups × 2 subs each = 8 sub-iterations):
     *   16 ZMM loads (native data) + 32 vpshufb (decode) + 32 vpdpbusd
     *   Memory traffic: 1024 bytes (native size — zero expansion)
     */
    inline void gemv_native_vnni_avx512_chunk_native(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk,
        int kb_start,
        int kb_end,
        const __m512i decode_lut)
    {
        __m512 fp_acc0 = _mm512_setzero_ps();
        __m512 fp_acc1 = _mm512_setzero_ps();
        __m512 fp_acc2 = _mm512_setzero_ps();
        __m512 fp_acc3 = _mm512_setzero_ps();

        const __m512i bias_128_i32 = _mm512_set1_epi32(128);
        const NibbleDecodeKind decode_kind = nibbleDecodeKind(packed.codebook_id);
        const __m512i mask_0F = _mm512_set1_epi8(0x0F);
        const __m512i q4_zero_offset = _mm512_set1_epi8(8);

        for (int kb = kb_start; kb < kb_end; ++kb)
        {
            const Q8_1Block &a_blk = A_q8[kb];
            float a_scale = simd::fp16_to_fp32(a_blk.d);
            int16_t a_sum = a_blk.sum_qs;

            __m512i int_acc0 = _mm512_setzero_si512();
            __m512i int_acc1 = _mm512_setzero_si512();
            __m512i int_acc2 = _mm512_setzero_si512();
            __m512i int_acc3 = _mm512_setzero_si512();

            // 4 groups: each loads 4 native bytes per column and extracts lo+hi nibbles.
            // Group g covers native bytes [g*4..g*4+3]:
            //   low  nibbles → K-elements [g*4 .. g*4+3]     (1st sub)
            //   high nibbles → K-elements [g*4+16 .. g*4+19]  (2nd sub)
            for (int group = 0; group < 4; ++group)
            {
                // Load 4 ZMMs of interleaved native bytes (64 cols × 4 bytes)
                __m512i raw0 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 0));
                __m512i raw1 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 1));
                __m512i raw2 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 2));
                __m512i raw3 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 3));

                // Decode low nibbles via vpshufb LUT → signed INT8 in VNNI lane order
                __m512i lo0 = decode_low_nibbles_avx512(raw0, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i lo1 = decode_low_nibbles_avx512(raw1, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i lo2 = decode_low_nibbles_avx512(raw2, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i lo3 = decode_low_nibbles_avx512(raw3, decode_lut, decode_kind, mask_0F, q4_zero_offset);

                // A broadcast for low-nibble sub (K-elements group*4..group*4+3)
                uint8_t a_lo[4];
                a_lo[0] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 0]) + 128);
                a_lo[1] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 1]) + 128);
                a_lo[2] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 2]) + 128);
                a_lo[3] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 3]) + 128);
                int32_t a_lo_i32;
                std::memcpy(&a_lo_i32, a_lo, 4);
                __m512i a_lo_bcast = _mm512_set1_epi32(a_lo_i32);

                int_acc0 = _mm512_dpbusd_epi32(int_acc0, a_lo_bcast, lo0);
                int_acc1 = _mm512_dpbusd_epi32(int_acc1, a_lo_bcast, lo1);
                int_acc2 = _mm512_dpbusd_epi32(int_acc2, a_lo_bcast, lo2);
                int_acc3 = _mm512_dpbusd_epi32(int_acc3, a_lo_bcast, lo3);

                // Decode high nibbles: shift right 4, mask, then vpshufb LUT
                __m512i hi0 = decode_high_nibbles_avx512(raw0, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i hi1 = decode_high_nibbles_avx512(raw1, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i hi2 = decode_high_nibbles_avx512(raw2, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i hi3 = decode_high_nibbles_avx512(raw3, decode_lut, decode_kind, mask_0F, q4_zero_offset);

                // A broadcast for high-nibble sub (K-elements group*4+16..group*4+19)
                uint8_t a_hi[4];
                a_hi[0] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 16]) + 128);
                a_hi[1] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 17]) + 128);
                a_hi[2] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 18]) + 128);
                a_hi[3] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 19]) + 128);
                int32_t a_hi_i32;
                std::memcpy(&a_hi_i32, a_hi, 4);
                __m512i a_hi_bcast = _mm512_set1_epi32(a_hi_i32);

                int_acc0 = _mm512_dpbusd_epi32(int_acc0, a_hi_bcast, hi0);
                int_acc1 = _mm512_dpbusd_epi32(int_acc1, a_hi_bcast, hi1);
                int_acc2 = _mm512_dpbusd_epi32(int_acc2, a_hi_bcast, hi2);
                int_acc3 = _mm512_dpbusd_epi32(int_acc3, a_hi_bcast, hi3);
            }

            // Bias correction: corrected = int_acc - 128 * comp
            // comp is INT16 (halved metadata); load via sign-extend to INT32
            const int16_t *comp_ptr = packed.chunkComp(chunk, kb);
            __m512i comp0 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr)));
            __m512i comp1 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 16)));
            __m512i comp2 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 32)));
            __m512i comp3 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 48)));

            int_acc0 = _mm512_sub_epi32(int_acc0, _mm512_mullo_epi32(bias_128_i32, comp0));
            int_acc1 = _mm512_sub_epi32(int_acc1, _mm512_mullo_epi32(bias_128_i32, comp1));
            int_acc2 = _mm512_sub_epi32(int_acc2, _mm512_mullo_epi32(bias_128_i32, comp2));
            int_acc3 = _mm512_sub_epi32(int_acc3, _mm512_mullo_epi32(bias_128_i32, comp3));

            // Convert to FP32 and scale: fp_val = int32_val * a_scale * b_scale[n]
            // scales are FP16 (halved metadata); load via F16C convert to FP32
            __m512 a_scale_v = _mm512_set1_ps(a_scale);

            const uint16_t *b_scales = packed.chunkScales(chunk, kb);
            __m512 cs0 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales))));
            __m512 cs1 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 16))));
            __m512 cs2 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 32))));
            __m512 cs3 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 48))));

            fp_acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc0), cs0, fp_acc0);
            fp_acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc1), cs1, fp_acc1);
            fp_acc2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc2), cs2, fp_acc2);
            fp_acc3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc3), cs3, fp_acc3);

            // Asymmetric correction: acc += a_scale * sum_qs * b_min[n]
            // Formula: weight = scale * int_val + min, so the offset term
            // contributes min * Σ A[k] ≈ min * a_scale * sum_qs per block.
            // mins are FP16; load via F16C convert to FP32
            if (packed.is_asymmetric)
            {
                const uint16_t *b_mins = packed.chunkMins(chunk, kb);
                __m512 a_corr_v = _mm512_set1_ps(static_cast<float>(a_sum) * a_scale);
                fp_acc0 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins))), fp_acc0);
                fp_acc1 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 16))), fp_acc1);
                fp_acc2 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 32))), fp_acc2);
                fp_acc3 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 48))), fp_acc3);
            }
        }

        _mm512_storeu_ps(C, fp_acc0);
        _mm512_storeu_ps(C + 16, fp_acc1);
        _mm512_storeu_ps(C + 32, fp_acc2);
        _mm512_storeu_ps(C + 48, fp_acc3);
    }

    // =========================================================================
    // AVX-512 VNNI GEMV with pre-decoded INT8 weights (non-4-bit formats)
    // =========================================================================
    //
    // For formats that cannot use vpshufb LUT decode (Q5_0, Q5_1, Q6_K, Q3_K,
    // Q2_K, IQ2/3/1 formats), the weight packer pre-decodes to INT8 at pack
    // time and stores them in VNNI-interleaved order:
    //   native_interleaved: [N_chunks][bpr][8 groups][4 ZMMs][64 bytes]
    //
    // Each group covers 4 consecutive K-elements (vs the nibble path which
    // covers 8 K-elements per group via lo/hi nibble split).
    //
    // Memory traffic per K-block: 2048 bytes (1.0 byte/element INT8)
    // This is 2× the native nibble path, but universally applicable.
    // =========================================================================

    /**
     * @brief AVX-512 VNNI GEMV for one N-chunk of 64 columns (INT8 pre-decoded path).
     *
     * Loads pre-decoded INT8 values directly from the interleaved buffer.
     * 8 groups × 4 K-elements per group = 32 K-elements per block.
     */
    inline void gemv_native_vnni_avx512_chunk_int8(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk,
        int kb_start,
        int kb_end)
    {
        __m512 fp_acc0 = _mm512_setzero_ps();
        __m512 fp_acc1 = _mm512_setzero_ps();
        __m512 fp_acc2 = _mm512_setzero_ps();
        __m512 fp_acc3 = _mm512_setzero_ps();

        const __m512i bias_128_i32 = _mm512_set1_epi32(128);

        for (int kb = kb_start; kb < kb_end; ++kb)
        {
            const Q8_1Block &a_blk = A_q8[kb];
            float a_scale = simd::fp16_to_fp32(a_blk.d);
            int16_t a_sum = a_blk.sum_qs;

            __m512i int_acc0 = _mm512_setzero_si512();
            __m512i int_acc1 = _mm512_setzero_si512();
            __m512i int_acc2 = _mm512_setzero_si512();
            __m512i int_acc3 = _mm512_setzero_si512();

            // 8 groups × 4 K-elements each = 32 K-elements per block.
            // Each group loads pre-decoded signed INT8 in VNNI-interleaved order.
            for (int group = 0; group < 8; ++group)
            {
                // Load 4 ZMMs of pre-decoded INT8 (16 cols × 4 INT8 values each)
                __m512i b0 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 0));
                __m512i b1 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 1));
                __m512i b2 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 2));
                __m512i b3 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 3));

                // A broadcast for this group's 4 consecutive K-elements [group*4 .. group*4+3]
                uint8_t a_u8[4];
                a_u8[0] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 0]) + 128);
                a_u8[1] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 1]) + 128);
                a_u8[2] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 2]) + 128);
                a_u8[3] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 3]) + 128);
                int32_t a_i32;
                std::memcpy(&a_i32, a_u8, 4);
                __m512i a_bcast = _mm512_set1_epi32(a_i32);

                int_acc0 = _mm512_dpbusd_epi32(int_acc0, a_bcast, b0);
                int_acc1 = _mm512_dpbusd_epi32(int_acc1, a_bcast, b1);
                int_acc2 = _mm512_dpbusd_epi32(int_acc2, a_bcast, b2);
                int_acc3 = _mm512_dpbusd_epi32(int_acc3, a_bcast, b3);
            }

            // Bias correction: corrected = int_acc - 128 * comp
            // comp is INT16 (halved metadata); load via sign-extend to INT32
            const int16_t *comp_ptr = packed.chunkComp(chunk, kb);
            __m512i comp0 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr)));
            __m512i comp1 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 16)));
            __m512i comp2 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 32)));
            __m512i comp3 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 48)));

            int_acc0 = _mm512_sub_epi32(int_acc0, _mm512_mullo_epi32(bias_128_i32, comp0));
            int_acc1 = _mm512_sub_epi32(int_acc1, _mm512_mullo_epi32(bias_128_i32, comp1));
            int_acc2 = _mm512_sub_epi32(int_acc2, _mm512_mullo_epi32(bias_128_i32, comp2));
            int_acc3 = _mm512_sub_epi32(int_acc3, _mm512_mullo_epi32(bias_128_i32, comp3));

            // Convert to FP32 and scale: fp_val = int32_val * a_scale * b_scale[n]
            // scales are FP16 (halved metadata); load via F16C convert to FP32
            __m512 a_scale_v = _mm512_set1_ps(a_scale);

            const uint16_t *b_scales = packed.chunkScales(chunk, kb);
            __m512 cs0 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales))));
            __m512 cs1 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 16))));
            __m512 cs2 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 32))));
            __m512 cs3 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 48))));

            fp_acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc0), cs0, fp_acc0);
            fp_acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc1), cs1, fp_acc1);
            fp_acc2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc2), cs2, fp_acc2);
            fp_acc3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc3), cs3, fp_acc3);

            // Asymmetric correction: acc += a_scale * sum_qs * b_min[n]
            // mins are FP16; load via F16C convert to FP32
            if (packed.is_asymmetric)
            {
                const uint16_t *b_mins = packed.chunkMins(chunk, kb);
                __m512 a_corr_v = _mm512_set1_ps(static_cast<float>(a_sum) * a_scale);
                fp_acc0 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins))), fp_acc0);
                fp_acc1 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 16))), fp_acc1);
                fp_acc2 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 32))), fp_acc2);
                fp_acc3 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 48))), fp_acc3);
            }
        }

        _mm512_storeu_ps(C, fp_acc0);
        _mm512_storeu_ps(C + 16, fp_acc1);
        _mm512_storeu_ps(C + 32, fp_acc2);
        _mm512_storeu_ps(C + 48, fp_acc3);
    }

    /**
     * @brief Multi-chunk GEMV processing a block of consecutive N-chunks.
     *
     * Processes n_block_chunks consecutive 64-column chunks with L2 prefetch.
     */
    inline void gemv_native_vnni_avx512_block(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk_start,
        int chunk_count,
        int K_blocks,
        int N,
        const __m512i decode_lut)
    {
        const bool use_nibble_lut = packed.is_nibble_lut;

        for (int ci = 0; ci < chunk_count; ++ci)
        {
            int chunk = chunk_start + ci;
            int n_start = chunk * 64;
            int n_cols = std::min(64, N - n_start);

            // Prefetch next chunk's first group data into L2
            if (ci + 1 < chunk_count)
            {
                _mm_prefetch(reinterpret_cast<const char *>(packed.interleavedB(chunk + 1, 0, 0, 0)),
                             _MM_HINT_T1);
                _mm_prefetch(reinterpret_cast<const char *>(packed.chunkScales(chunk + 1, 0)),
                             _MM_HINT_T1);
            }

            if (n_cols < 64)
            {
                // Tail chunk: kernel writes 64 floats, but only n_cols are valid.
                // Use a temp buffer to avoid overflowing the caller's C buffer.
                alignas(64) float tmp[64];
                if (use_nibble_lut)
                    gemv_native_vnni_avx512_chunk_native(packed, A_q8, tmp, chunk, 0, K_blocks, decode_lut);
                else
                    gemv_native_vnni_avx512_chunk_int8(packed, A_q8, tmp, chunk, 0, K_blocks);
                std::memcpy(C + n_start, tmp, n_cols * sizeof(float));
            }
            else
            {
                if (use_nibble_lut)
                    gemv_native_vnni_avx512_chunk_native(packed, A_q8, C + n_start, chunk, 0, K_blocks, decode_lut);
                else
                    gemv_native_vnni_avx512_chunk_int8(packed, A_q8, C + n_start, chunk, 0, K_blocks);
            }
        }
    }

#endif // __AVX512F__ && __AVX512VNNI__ && __AVX512BW__

    // =========================================================================
    // Pre-quantized GEMV (M=1) — compute only, skips quantization
    // =========================================================================
    //
    // Caller is responsible for providing pre-quantized Q8_1 blocks.
    // Used by multiply_fused() to avoid redundant quantization when
    // the same input is projected through multiple weight matrices.
    // =========================================================================

    inline void gemv_native_vnni_preq(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        ISAPath isa_path = ISAPath::AUTO)
    {
        const int N = packed.N;
        const int K = packed.K;
        const int K_blocks = packed.blocks_per_row;
        const int N_chunks = (N + 63) / 64;

        // Runtime ISA selection
        const ISALevel active_isa = activeISALevel();
        bool use_avx512 = false;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        use_avx512 = (isa_path == ISAPath::AUTO) ? (active_isa >= ISALevel::AVX512)
                                                 : (isa_path == ISAPath::AVX512);
#endif
        const bool use_avx2 =
            !use_avx512 &&
            ((isa_path == ISAPath::AUTO && active_isa >= ISALevel::AVX2) ||
             isa_path == ISAPath::AVX2);

        // Compute tile configuration
        int num_threads = omp_get_max_threads();
        NativeVNNITileConfig cfg = computeTileConfig(N, K, 1, packed.payload_bytes, num_threads);

        // Initialize decode LUTs for selected ISA
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        __m512i decode_lut_512 = _mm512_setzero_si512();
        if (use_avx512 && packed.is_nibble_lut)
            decode_lut_512 = build_decode_lut(packed.codebook_id);
#endif
        __m256i decode_lut_256 = _mm256_setzero_si256();
        if (use_avx2 && packed.is_nibble_lut)
            decode_lut_256 = build_decode_lut_avx2_for_codebook(packed.codebook_id);

        if (!use_avx512 && !use_avx2)
        {
            gemv_native_vnni_scalar(packed, A_q8, C, N, K_blocks);
            return;
        }

        // K-parallel GEMV when N-parallelism is insufficient
        if (cfg.k_tiles > 1)
        {
            int k_tiles = cfg.k_tiles;
            int k_blocks_per_tile = (K_blocks + k_tiles - 1) / k_tiles;
            /*
             * Each [N chunk, K tile] partial is fully overwritten before the
             * reduction pass.  Use default-initialized storage so long-K decode
             * verifier rows do not pay a redundant memset on every GEMV call.
             */
            std::unique_ptr<float[]> partial_sums(
                new float[static_cast<size_t>(N_chunks) * k_tiles * 64]);

            auto do_gemv_kpar = [&]()
            {
                int total_2d = N_chunks * k_tiles;
#pragma omp for schedule(static)
                for (int task = 0; task < total_2d; ++task)
                {
                    int chunk_idx = task / k_tiles;
                    int kt = task % k_tiles;
                    int kb_start = kt * k_blocks_per_tile;
                    int kb_end = std::min(kb_start + k_blocks_per_tile, K_blocks);
                    float *dest = partial_sums.get() +
                                  (static_cast<size_t>(chunk_idx) * k_tiles + kt) * 64;

                    if (use_avx512)
                    {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                        if (packed.is_nibble_lut)
                            gemv_native_vnni_avx512_chunk_native(packed, A_q8, dest,
                                                                 chunk_idx, kb_start, kb_end, decode_lut_512);
                        else
                            gemv_native_vnni_avx512_chunk_int8(packed, A_q8, dest,
                                                               chunk_idx, kb_start, kb_end);
#endif
                    }
                    else
                    {
                        if (packed.is_nibble_lut)
                            gemv_avx2_chunk_native(packed, A_q8, dest,
                                                   chunk_idx, kb_start, kb_end, decode_lut_256);
                        else
                            gemv_avx2_chunk_int8(packed, A_q8, dest,
                                                 chunk_idx, kb_start, kb_end);
                    }
                }

                // Reduce partial sums across K-tiles
#pragma omp for schedule(static)
                for (int chunk_idx = 0; chunk_idx < N_chunks; ++chunk_idx)
                {
                    int n_start = chunk_idx * 64;
                    int n_cols = std::min(64, N - n_start);
                    const float *base = partial_sums.get() +
                                        static_cast<size_t>(chunk_idx) * k_tiles * 64;
#if defined(__AVX512F__)
                    __m512 sum0 = _mm512_loadu_ps(base);
                    __m512 sum1 = _mm512_loadu_ps(base + 16);
                    __m512 sum2 = _mm512_loadu_ps(base + 32);
                    __m512 sum3 = _mm512_loadu_ps(base + 48);
                    for (int kt = 1; kt < k_tiles; ++kt)
                    {
                        const float *src = base + kt * 64;
                        sum0 = _mm512_add_ps(sum0, _mm512_loadu_ps(src));
                        sum1 = _mm512_add_ps(sum1, _mm512_loadu_ps(src + 16));
                        sum2 = _mm512_add_ps(sum2, _mm512_loadu_ps(src + 32));
                        sum3 = _mm512_add_ps(sum3, _mm512_loadu_ps(src + 48));
                    }
                    if (n_cols < 64)
                    {
                        alignas(64) float tmp[64];
                        _mm512_store_ps(tmp, sum0);
                        _mm512_store_ps(tmp + 16, sum1);
                        _mm512_store_ps(tmp + 32, sum2);
                        _mm512_store_ps(tmp + 48, sum3);
                        std::memcpy(C + n_start, tmp, n_cols * sizeof(float));
                    }
                    else
                    {
                        _mm512_storeu_ps(C + n_start, sum0);
                        _mm512_storeu_ps(C + n_start + 16, sum1);
                        _mm512_storeu_ps(C + n_start + 32, sum2);
                        _mm512_storeu_ps(C + n_start + 48, sum3);
                    }
#else
                    // AVX2 FP32 partial sum reduction
                    __m256 s0 = _mm256_loadu_ps(base);
                    __m256 s1 = _mm256_loadu_ps(base + 8);
                    __m256 s2 = _mm256_loadu_ps(base + 16);
                    __m256 s3 = _mm256_loadu_ps(base + 24);
                    __m256 s4 = _mm256_loadu_ps(base + 32);
                    __m256 s5 = _mm256_loadu_ps(base + 40);
                    __m256 s6 = _mm256_loadu_ps(base + 48);
                    __m256 s7 = _mm256_loadu_ps(base + 56);
                    for (int kt = 1; kt < k_tiles; ++kt)
                    {
                        const float *src = base + kt * 64;
                        s0 = _mm256_add_ps(s0, _mm256_loadu_ps(src));
                        s1 = _mm256_add_ps(s1, _mm256_loadu_ps(src + 8));
                        s2 = _mm256_add_ps(s2, _mm256_loadu_ps(src + 16));
                        s3 = _mm256_add_ps(s3, _mm256_loadu_ps(src + 24));
                        s4 = _mm256_add_ps(s4, _mm256_loadu_ps(src + 32));
                        s5 = _mm256_add_ps(s5, _mm256_loadu_ps(src + 40));
                        s6 = _mm256_add_ps(s6, _mm256_loadu_ps(src + 48));
                        s7 = _mm256_add_ps(s7, _mm256_loadu_ps(src + 56));
                    }
                    if (n_cols < 64)
                    {
                        alignas(64) float tmp[64];
                        _mm256_store_ps(tmp, s0);
                        _mm256_store_ps(tmp + 8, s1);
                        _mm256_store_ps(tmp + 16, s2);
                        _mm256_store_ps(tmp + 24, s3);
                        _mm256_store_ps(tmp + 32, s4);
                        _mm256_store_ps(tmp + 40, s5);
                        _mm256_store_ps(tmp + 48, s6);
                        _mm256_store_ps(tmp + 56, s7);
                        std::memcpy(C + n_start, tmp, n_cols * sizeof(float));
                    }
                    else
                    {
                        _mm256_storeu_ps(C + n_start, s0);
                        _mm256_storeu_ps(C + n_start + 8, s1);
                        _mm256_storeu_ps(C + n_start + 16, s2);
                        _mm256_storeu_ps(C + n_start + 24, s3);
                        _mm256_storeu_ps(C + n_start + 32, s4);
                        _mm256_storeu_ps(C + n_start + 40, s5);
                        _mm256_storeu_ps(C + n_start + 48, s6);
                        _mm256_storeu_ps(C + n_start + 56, s7);
                    }
#endif
                }
            };

            OMP_WORKSHARE_REGION(do_gemv_kpar);
            return;
        }

        // Standard N-parallel GEMV
        int n_block_chunks = cfg.n_block_chunks;
        int total_blocks = (N_chunks + n_block_chunks - 1) / n_block_chunks;

        // Serial fast path: when there are fewer tasks than threads and we're
        // not already inside a parallel region, skip OMP entirely.  For MoE
        // expert decode (N=512 → 8 tasks, N=2048 → 32 tasks with 28 threads)
        // the fork/join overhead (~5-10µs) dominates the per-call compute.
        bool serialize = !omp_in_parallel() && total_blocks < num_threads;
        if (serialize)
        {
            for (int block_idx = 0; block_idx < total_blocks; ++block_idx)
            {
                int chunk_start = block_idx * n_block_chunks;
                int chunk_count = std::min(n_block_chunks, N_chunks - chunk_start);

                if (use_avx512)
                {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                    gemv_native_vnni_avx512_block(packed, A_q8, C,
                                                  chunk_start, chunk_count, K_blocks, N,
                                                  decode_lut_512);
#endif
                }
                else
                {
                    gemv_avx2_block(packed, A_q8, C,
                                    chunk_start, chunk_count, K_blocks, N,
                                    decode_lut_256);
                }
            }
            return;
        }

        auto do_gemv = [&]()
        {
#pragma omp for schedule(static)
            for (int block_idx = 0; block_idx < total_blocks; ++block_idx)
            {
                int chunk_start = block_idx * n_block_chunks;
                int chunk_count = std::min(n_block_chunks, N_chunks - chunk_start);

                if (use_avx512)
                {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                    gemv_native_vnni_avx512_block(packed, A_q8, C,
                                                  chunk_start, chunk_count, K_blocks, N,
                                                  decode_lut_512);
#endif
                }
                else
                {
                    gemv_avx2_block(packed, A_q8, C,
                                    chunk_start, chunk_count, K_blocks, N,
                                    decode_lut_256);
                }
            }
        };

        OMP_WORKSHARE_REGION(do_gemv);
    }

    // =========================================================================
    // Forward declaration: Q8_0 native GEMV (defined later in this file)
    // =========================================================================
    inline void q8_0_native_gemv(
        const Q8_0Block *__restrict blocks,
        const float *__restrict A,
        float *__restrict C,
        int N, int K, int bpr);

    // =========================================================================
    // Full GEMV dispatcher (M=1) — unified entrypoint
    //
    // When q8_0_blocks is non-null, uses the Q8_0 native path (direct block
    // access + FP32 dequant + FMA). Otherwise uses the packed VNNI path
    // (FP32→Q8_1 quantize + vpdpbusd). The caller does not need to know
    // which format is active.
    // =========================================================================

    inline void gemv_native_vnni(
        const CPUNativeVNNIPackedWeights &packed,
        const float *A_fp32,
        float *C)
    {
        // Q8_0 fast path: workspace contains raw Q8_0 blocks (no interleave).
        // Use dedicated raw-block GEMV for both AVX512 and AVX2 builds.
        if (packed.codebook_id == 19 && packed.workspace_data_)
        {
            auto *blocks = reinterpret_cast<const Q8_0Block *>(packed.workspace_data_);
            q8_0_native_gemv(blocks, A_fp32, C, packed.N, packed.K, packed.blocks_per_row);
            return;
        }

        const int K = packed.K;
        const int K_blocks = packed.blocks_per_row;

        // Step 1: Quantize activations to Q8_1 (thread-local buffer avoids heap alloc per call)
        thread_local std::vector<Q8_1Block> A_q8_tls;
        if (static_cast<int>(A_q8_tls.size()) < K_blocks)
            A_q8_tls.resize(K_blocks);
        Q8_1Block *A_q8 = A_q8_tls.data();

        // Process blocks in pairs for 2-way ILP (AVX-512)
        const bool k_aligned = (K % 32 == 0);
        int kb = 0;
#if defined(__AVX512F__)
        if (k_aligned)
        {
            for (; kb + 1 < K_blocks; kb += 2)
            {
                simd::quantize_two_blocks_avx512(A_fp32 + kb * 32, A_q8[kb], A_q8[kb + 1]);
            }
        }
#endif
        for (; kb < K_blocks; ++kb)
        {
            int block_start = kb * 32;
            int block_len = std::min(32, K - block_start);
            simd::quantize_single_block(A_fp32 + block_start, A_q8[kb], block_len);
        }

        // Step 2+: Delegate to pre-quantized compute path
        gemv_native_vnni_preq(packed, A_q8, C);
    }

    // =========================================================================
    // 2-Row GEMM Microkernels (share B loads across 2 M rows)
    // =========================================================================
    //
    // The key optimization for M>1 GEMM: load each B weight block once and
    // compute dot products for 2 rows simultaneously. This doubles the
    // compute-to-load ratio vs row-by-row GEMV.
    //
    // Register layout per 64-col chunk (2 rows):
    //   Row 0: fp_acc0..fp_acc3 (4 × __m512, 64 FP32 accumulators)
    //   Row 1: fp_acc4..fp_acc7 (4 × __m512, 64 FP32 accumulators)
    //   B data: loaded once, used for both rows
    //   A broadcasts: separate per row (different activation values)
    // =========================================================================

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)

    /**
     * @brief 2-row nibble-LUT GEMM microkernel for one 64-col chunk.
     *
     * Processes rows m0 and m1 simultaneously, sharing B loads.
     * Accumulates partial results for a K-block range [kb_start, kb_end).
     * Caller must add results to existing C values when K-tiling.
     */
    inline void gemm_2row_native_chunk(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_row0,
        const Q8_1Block *A_q8_row1,
        float *C_row0,
        float *C_row1,
        int chunk,
        int kb_start,
        int kb_end,
        const __m512i decode_lut,
        bool accumulate)
    {
        __m512 fp0_0 = accumulate ? _mm512_loadu_ps(C_row0) : _mm512_setzero_ps();
        __m512 fp0_1 = accumulate ? _mm512_loadu_ps(C_row0 + 16) : _mm512_setzero_ps();
        __m512 fp0_2 = accumulate ? _mm512_loadu_ps(C_row0 + 32) : _mm512_setzero_ps();
        __m512 fp0_3 = accumulate ? _mm512_loadu_ps(C_row0 + 48) : _mm512_setzero_ps();
        __m512 fp1_0 = accumulate ? _mm512_loadu_ps(C_row1) : _mm512_setzero_ps();
        __m512 fp1_1 = accumulate ? _mm512_loadu_ps(C_row1 + 16) : _mm512_setzero_ps();
        __m512 fp1_2 = accumulate ? _mm512_loadu_ps(C_row1 + 32) : _mm512_setzero_ps();
        __m512 fp1_3 = accumulate ? _mm512_loadu_ps(C_row1 + 48) : _mm512_setzero_ps();

        const __m512i bias_128_i32 = _mm512_set1_epi32(128);
        const NibbleDecodeKind decode_kind = nibbleDecodeKind(packed.codebook_id);
        const __m512i mask_0F = _mm512_set1_epi8(0x0F);
        const __m512i q4_zero_offset = _mm512_set1_epi8(8);

        for (int kb = kb_start; kb < kb_end; ++kb)
        {
            const Q8_1Block &a0 = A_q8_row0[kb];
            const Q8_1Block &a1 = A_q8_row1[kb];
            float a0_scale = simd::fp16_to_fp32(a0.d);
            float a1_scale = simd::fp16_to_fp32(a1.d);

            __m512i ia0_0 = _mm512_setzero_si512(), ia0_1 = _mm512_setzero_si512();
            __m512i ia0_2 = _mm512_setzero_si512(), ia0_3 = _mm512_setzero_si512();
            __m512i ia1_0 = _mm512_setzero_si512(), ia1_1 = _mm512_setzero_si512();
            __m512i ia1_2 = _mm512_setzero_si512(), ia1_3 = _mm512_setzero_si512();

            for (int group = 0; group < 4; ++group)
            {
                // Load B once — shared across both rows
                __m512i raw0 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 0));
                __m512i raw1 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 1));
                __m512i raw2 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 2));
                __m512i raw3 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 3));

                // Decode nibbles — shared across rows
                __m512i lo0 = decode_low_nibbles_avx512(raw0, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i lo1 = decode_low_nibbles_avx512(raw1, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i lo2 = decode_low_nibbles_avx512(raw2, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i lo3 = decode_low_nibbles_avx512(raw3, decode_lut, decode_kind, mask_0F, q4_zero_offset);

                __m512i hi0 = decode_high_nibbles_avx512(raw0, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i hi1 = decode_high_nibbles_avx512(raw1, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i hi2 = decode_high_nibbles_avx512(raw2, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i hi3 = decode_high_nibbles_avx512(raw3, decode_lut, decode_kind, mask_0F, q4_zero_offset);

                // Row 0 A broadcasts + VPDPBUSD
                {
                    uint8_t vals[4];
                    vals[0] = static_cast<uint8_t>(static_cast<int16_t>(a0.qs[group * 4 + 0]) + 128);
                    vals[1] = static_cast<uint8_t>(static_cast<int16_t>(a0.qs[group * 4 + 1]) + 128);
                    vals[2] = static_cast<uint8_t>(static_cast<int16_t>(a0.qs[group * 4 + 2]) + 128);
                    vals[3] = static_cast<uint8_t>(static_cast<int16_t>(a0.qs[group * 4 + 3]) + 128);
                    int32_t v;
                    std::memcpy(&v, vals, 4);
                    __m512i ab = _mm512_set1_epi32(v);
                    ia0_0 = _mm512_dpbusd_epi32(ia0_0, ab, lo0);
                    ia0_1 = _mm512_dpbusd_epi32(ia0_1, ab, lo1);
                    ia0_2 = _mm512_dpbusd_epi32(ia0_2, ab, lo2);
                    ia0_3 = _mm512_dpbusd_epi32(ia0_3, ab, lo3);

                    vals[0] = static_cast<uint8_t>(static_cast<int16_t>(a0.qs[group * 4 + 16]) + 128);
                    vals[1] = static_cast<uint8_t>(static_cast<int16_t>(a0.qs[group * 4 + 17]) + 128);
                    vals[2] = static_cast<uint8_t>(static_cast<int16_t>(a0.qs[group * 4 + 18]) + 128);
                    vals[3] = static_cast<uint8_t>(static_cast<int16_t>(a0.qs[group * 4 + 19]) + 128);
                    std::memcpy(&v, vals, 4);
                    ab = _mm512_set1_epi32(v);
                    ia0_0 = _mm512_dpbusd_epi32(ia0_0, ab, hi0);
                    ia0_1 = _mm512_dpbusd_epi32(ia0_1, ab, hi1);
                    ia0_2 = _mm512_dpbusd_epi32(ia0_2, ab, hi2);
                    ia0_3 = _mm512_dpbusd_epi32(ia0_3, ab, hi3);
                }

                // Row 1 A broadcasts + VPDPBUSD (same B data, different A)
                {
                    uint8_t vals[4];
                    vals[0] = static_cast<uint8_t>(static_cast<int16_t>(a1.qs[group * 4 + 0]) + 128);
                    vals[1] = static_cast<uint8_t>(static_cast<int16_t>(a1.qs[group * 4 + 1]) + 128);
                    vals[2] = static_cast<uint8_t>(static_cast<int16_t>(a1.qs[group * 4 + 2]) + 128);
                    vals[3] = static_cast<uint8_t>(static_cast<int16_t>(a1.qs[group * 4 + 3]) + 128);
                    int32_t v;
                    std::memcpy(&v, vals, 4);
                    __m512i ab = _mm512_set1_epi32(v);
                    ia1_0 = _mm512_dpbusd_epi32(ia1_0, ab, lo0);
                    ia1_1 = _mm512_dpbusd_epi32(ia1_1, ab, lo1);
                    ia1_2 = _mm512_dpbusd_epi32(ia1_2, ab, lo2);
                    ia1_3 = _mm512_dpbusd_epi32(ia1_3, ab, lo3);

                    vals[0] = static_cast<uint8_t>(static_cast<int16_t>(a1.qs[group * 4 + 16]) + 128);
                    vals[1] = static_cast<uint8_t>(static_cast<int16_t>(a1.qs[group * 4 + 17]) + 128);
                    vals[2] = static_cast<uint8_t>(static_cast<int16_t>(a1.qs[group * 4 + 18]) + 128);
                    vals[3] = static_cast<uint8_t>(static_cast<int16_t>(a1.qs[group * 4 + 19]) + 128);
                    std::memcpy(&v, vals, 4);
                    ab = _mm512_set1_epi32(v);
                    ia1_0 = _mm512_dpbusd_epi32(ia1_0, ab, hi0);
                    ia1_1 = _mm512_dpbusd_epi32(ia1_1, ab, hi1);
                    ia1_2 = _mm512_dpbusd_epi32(ia1_2, ab, hi2);
                    ia1_3 = _mm512_dpbusd_epi32(ia1_3, ab, hi3);
                }
            }

            // Bias correction + scale (shared comp/scales loads, per-row a_scale)
            const int16_t *comp_ptr = packed.chunkComp(chunk, kb);
            __m512i c0 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr)));
            __m512i c1 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 16)));
            __m512i c2 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 32)));
            __m512i c3 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 48)));

            __m512i bias_c0 = _mm512_mullo_epi32(bias_128_i32, c0);
            __m512i bias_c1 = _mm512_mullo_epi32(bias_128_i32, c1);
            __m512i bias_c2 = _mm512_mullo_epi32(bias_128_i32, c2);
            __m512i bias_c3 = _mm512_mullo_epi32(bias_128_i32, c3);

            ia0_0 = _mm512_sub_epi32(ia0_0, bias_c0);
            ia0_1 = _mm512_sub_epi32(ia0_1, bias_c1);
            ia0_2 = _mm512_sub_epi32(ia0_2, bias_c2);
            ia0_3 = _mm512_sub_epi32(ia0_3, bias_c3);
            ia1_0 = _mm512_sub_epi32(ia1_0, bias_c0);
            ia1_1 = _mm512_sub_epi32(ia1_1, bias_c1);
            ia1_2 = _mm512_sub_epi32(ia1_2, bias_c2);
            ia1_3 = _mm512_sub_epi32(ia1_3, bias_c3);

            const uint16_t *b_scales = packed.chunkScales(chunk, kb);
            __m512 bs0 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales)));
            __m512 bs1 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 16)));
            __m512 bs2 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 32)));
            __m512 bs3 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 48)));

            __m512 as0 = _mm512_set1_ps(a0_scale);
            __m512 cs0_0 = _mm512_mul_ps(as0, bs0), cs0_1 = _mm512_mul_ps(as0, bs1);
            __m512 cs0_2 = _mm512_mul_ps(as0, bs2), cs0_3 = _mm512_mul_ps(as0, bs3);
            fp0_0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia0_0), cs0_0, fp0_0);
            fp0_1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia0_1), cs0_1, fp0_1);
            fp0_2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia0_2), cs0_2, fp0_2);
            fp0_3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia0_3), cs0_3, fp0_3);

            __m512 as1 = _mm512_set1_ps(a1_scale);
            __m512 cs1_0 = _mm512_mul_ps(as1, bs0), cs1_1 = _mm512_mul_ps(as1, bs1);
            __m512 cs1_2 = _mm512_mul_ps(as1, bs2), cs1_3 = _mm512_mul_ps(as1, bs3);
            fp1_0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia1_0), cs1_0, fp1_0);
            fp1_1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia1_1), cs1_1, fp1_1);
            fp1_2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia1_2), cs1_2, fp1_2);
            fp1_3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia1_3), cs1_3, fp1_3);

            if (packed.is_asymmetric)
            {
                const uint16_t *b_mins = packed.chunkMins(chunk, kb);
                __m512 bm0 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins)));
                __m512 bm1 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 16)));
                __m512 bm2 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 32)));
                __m512 bm3 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 48)));

                __m512 corr0 = _mm512_set1_ps(static_cast<float>(a0.sum_qs) * a0_scale);
                fp0_0 = _mm512_fmadd_ps(corr0, bm0, fp0_0);
                fp0_1 = _mm512_fmadd_ps(corr0, bm1, fp0_1);
                fp0_2 = _mm512_fmadd_ps(corr0, bm2, fp0_2);
                fp0_3 = _mm512_fmadd_ps(corr0, bm3, fp0_3);

                __m512 corr1 = _mm512_set1_ps(static_cast<float>(a1.sum_qs) * a1_scale);
                fp1_0 = _mm512_fmadd_ps(corr1, bm0, fp1_0);
                fp1_1 = _mm512_fmadd_ps(corr1, bm1, fp1_1);
                fp1_2 = _mm512_fmadd_ps(corr1, bm2, fp1_2);
                fp1_3 = _mm512_fmadd_ps(corr1, bm3, fp1_3);
            }
        }

        _mm512_storeu_ps(C_row0, fp0_0);
        _mm512_storeu_ps(C_row0 + 16, fp0_1);
        _mm512_storeu_ps(C_row0 + 32, fp0_2);
        _mm512_storeu_ps(C_row0 + 48, fp0_3);
        _mm512_storeu_ps(C_row1, fp1_0);
        _mm512_storeu_ps(C_row1 + 16, fp1_1);
        _mm512_storeu_ps(C_row1 + 32, fp1_2);
        _mm512_storeu_ps(C_row1 + 48, fp1_3);
    }

    /**
     * @brief 2-row INT8 pre-decoded GEMM microkernel for one 64-col chunk.
     */
    inline void gemm_2row_int8_chunk(
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
        __m512 fp0_0 = accumulate ? _mm512_loadu_ps(C_row0) : _mm512_setzero_ps();
        __m512 fp0_1 = accumulate ? _mm512_loadu_ps(C_row0 + 16) : _mm512_setzero_ps();
        __m512 fp0_2 = accumulate ? _mm512_loadu_ps(C_row0 + 32) : _mm512_setzero_ps();
        __m512 fp0_3 = accumulate ? _mm512_loadu_ps(C_row0 + 48) : _mm512_setzero_ps();
        __m512 fp1_0 = accumulate ? _mm512_loadu_ps(C_row1) : _mm512_setzero_ps();
        __m512 fp1_1 = accumulate ? _mm512_loadu_ps(C_row1 + 16) : _mm512_setzero_ps();
        __m512 fp1_2 = accumulate ? _mm512_loadu_ps(C_row1 + 32) : _mm512_setzero_ps();
        __m512 fp1_3 = accumulate ? _mm512_loadu_ps(C_row1 + 48) : _mm512_setzero_ps();

        const __m512i bias_128_i32 = _mm512_set1_epi32(128);

        for (int kb = kb_start; kb < kb_end; ++kb)
        {
            const Q8_1Block &a0 = A_q8_row0[kb];
            const Q8_1Block &a1 = A_q8_row1[kb];
            float a0_scale = simd::fp16_to_fp32(a0.d);
            float a1_scale = simd::fp16_to_fp32(a1.d);

            __m512i ia0_0 = _mm512_setzero_si512(), ia0_1 = _mm512_setzero_si512();
            __m512i ia0_2 = _mm512_setzero_si512(), ia0_3 = _mm512_setzero_si512();
            __m512i ia1_0 = _mm512_setzero_si512(), ia1_1 = _mm512_setzero_si512();
            __m512i ia1_2 = _mm512_setzero_si512(), ia1_3 = _mm512_setzero_si512();

            for (int group = 0; group < 8; ++group)
            {
                // GPR-only A-prep: XOR with 0x80 converts signed→unsigned bytes,
                // avoids xmm intermediaries that alias zmm accumulators.
                // vpbroadcastd has a GPR source form on AVX-512 (no xmm needed).
                __m512i a0_bc =
                    _mm512_set1_epi32(pack_q8_1_unsigned_word(a0, group * 4));
                __m512i a1_bc =
                    _mm512_set1_epi32(pack_q8_1_unsigned_word(a1, group * 4));

// Load-use-discard: 1 B register live at a time (not 4).
// Peak: 8 INT32 + 8 FP32 + 2 A + 1 B + 1 const = 20 ZMMs.
#define INT8_SUBCHUNK(Z, IA0, IA1)                                               \
    {                                                                            \
        __m512i b = _mm512_load_si512(packed.interleavedB(chunk, kb, group, Z)); \
        IA0 = _mm512_dpbusd_epi32(IA0, a0_bc, b);                                \
        IA1 = _mm512_dpbusd_epi32(IA1, a1_bc, b);                                \
    }
                INT8_SUBCHUNK(0, ia0_0, ia1_0)
                INT8_SUBCHUNK(1, ia0_1, ia1_1)
                INT8_SUBCHUNK(2, ia0_2, ia1_2)
                INT8_SUBCHUNK(3, ia0_3, ia1_3)
#undef INT8_SUBCHUNK
            }

            // Bias correction (shared comp loads)
            const int16_t *comp_ptr = packed.chunkComp(chunk, kb);
            __m512i cc0 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr)));
            __m512i cc1 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 16)));
            __m512i cc2 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 32)));
            __m512i cc3 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 48)));

            __m512i bc0 = _mm512_mullo_epi32(bias_128_i32, cc0);
            __m512i bc1 = _mm512_mullo_epi32(bias_128_i32, cc1);
            __m512i bc2 = _mm512_mullo_epi32(bias_128_i32, cc2);
            __m512i bc3 = _mm512_mullo_epi32(bias_128_i32, cc3);

            ia0_0 = _mm512_sub_epi32(ia0_0, bc0);
            ia0_1 = _mm512_sub_epi32(ia0_1, bc1);
            ia0_2 = _mm512_sub_epi32(ia0_2, bc2);
            ia0_3 = _mm512_sub_epi32(ia0_3, bc3);
            ia1_0 = _mm512_sub_epi32(ia1_0, bc0);
            ia1_1 = _mm512_sub_epi32(ia1_1, bc1);
            ia1_2 = _mm512_sub_epi32(ia1_2, bc2);
            ia1_3 = _mm512_sub_epi32(ia1_3, bc3);

            // Scale (shared b_scales loads)
            const uint16_t *b_scales = packed.chunkScales(chunk, kb);
            __m512 bs0 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales)));
            __m512 bs1 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 16)));
            __m512 bs2 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 32)));
            __m512 bs3 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 48)));

            __m512 as0 = _mm512_set1_ps(a0_scale);
            fp0_0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia0_0), _mm512_mul_ps(as0, bs0), fp0_0);
            fp0_1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia0_1), _mm512_mul_ps(as0, bs1), fp0_1);
            fp0_2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia0_2), _mm512_mul_ps(as0, bs2), fp0_2);
            fp0_3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia0_3), _mm512_mul_ps(as0, bs3), fp0_3);

            __m512 as1 = _mm512_set1_ps(a1_scale);
            fp1_0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia1_0), _mm512_mul_ps(as1, bs0), fp1_0);
            fp1_1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia1_1), _mm512_mul_ps(as1, bs1), fp1_1);
            fp1_2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia1_2), _mm512_mul_ps(as1, bs2), fp1_2);
            fp1_3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia1_3), _mm512_mul_ps(as1, bs3), fp1_3);

            if (packed.is_asymmetric)
            {
                const uint16_t *b_mins = packed.chunkMins(chunk, kb);
                __m512 bm0 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins)));
                __m512 bm1 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 16)));
                __m512 bm2 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 32)));
                __m512 bm3 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 48)));

                __m512 corr0 = _mm512_set1_ps(static_cast<float>(a0.sum_qs) * a0_scale);
                fp0_0 = _mm512_fmadd_ps(corr0, bm0, fp0_0);
                fp0_1 = _mm512_fmadd_ps(corr0, bm1, fp0_1);
                fp0_2 = _mm512_fmadd_ps(corr0, bm2, fp0_2);
                fp0_3 = _mm512_fmadd_ps(corr0, bm3, fp0_3);

                __m512 corr1 = _mm512_set1_ps(static_cast<float>(a1.sum_qs) * a1_scale);
                fp1_0 = _mm512_fmadd_ps(corr1, bm0, fp1_0);
                fp1_1 = _mm512_fmadd_ps(corr1, bm1, fp1_1);
                fp1_2 = _mm512_fmadd_ps(corr1, bm2, fp1_2);
                fp1_3 = _mm512_fmadd_ps(corr1, bm3, fp1_3);
            }
        }

        _mm512_storeu_ps(C_row0, fp0_0);
        _mm512_storeu_ps(C_row0 + 16, fp0_1);
        _mm512_storeu_ps(C_row0 + 32, fp0_2);
        _mm512_storeu_ps(C_row0 + 48, fp0_3);
        _mm512_storeu_ps(C_row1, fp1_0);
        _mm512_storeu_ps(C_row1 + 16, fp1_1);
        _mm512_storeu_ps(C_row1 + 32, fp1_2);
        _mm512_storeu_ps(C_row1 + 48, fp1_3);
    }

    /**
     * @brief Three-row AVX512 verifier microkernel for nibble-LUT formats.
     *
     * MTP commonly verifies `draft_count + 1` rows.  For draft depth two, that
     * means M=3 target rows.  The older safe path ran one 2-row verifier chunk
     * plus one serial GEMV tail, which preserved decode equivalence but loaded
     * and decoded the same packed B chunk twice.  This kernel shares each
     * decoded B vector across three independent row accumulators while keeping
     * the per-row K-block, compensation, scale, and min-correction order
     * identical to serial decode GEMV.
     */
    inline void gemm_3row_native_2z_chunk(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_row0,
        const Q8_1Block *A_q8_row1,
        const Q8_1Block *A_q8_row2,
        float *C_row0,
        float *C_row1,
        float *C_row2,
        int chunk,
        int kb_start,
        int kb_end,
        const __m512i decode_lut,
        bool accumulate)
    {
        const __m512i bias_128_i32 = _mm512_set1_epi32(128);
        const NibbleDecodeKind decode_kind = nibbleDecodeKind(packed.codebook_id);
        const __m512i mask_0F = _mm512_set1_epi8(0x0F);
        const __m512i q4_zero_offset = _mm512_set1_epi8(8);

        for (int zbase = 0; zbase < 4; zbase += 2)
        {
            __m512 fp0_0 = accumulate ? _mm512_loadu_ps(C_row0 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp0_1 = accumulate ? _mm512_loadu_ps(C_row0 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp1_0 = accumulate ? _mm512_loadu_ps(C_row1 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp1_1 = accumulate ? _mm512_loadu_ps(C_row1 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp2_0 = accumulate ? _mm512_loadu_ps(C_row2 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp2_1 = accumulate ? _mm512_loadu_ps(C_row2 + (zbase + 1) * 16) : _mm512_setzero_ps();

            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                const Q8_1Block &a0 = A_q8_row0[kb];
                const Q8_1Block &a1 = A_q8_row1[kb];
                const Q8_1Block &a2 = A_q8_row2[kb];

                __m512i ia0_0 = _mm512_setzero_si512(), ia0_1 = _mm512_setzero_si512();
                __m512i ia1_0 = _mm512_setzero_si512(), ia1_1 = _mm512_setzero_si512();
                __m512i ia2_0 = _mm512_setzero_si512(), ia2_1 = _mm512_setzero_si512();

                for (int group = 0; group < 4; ++group)
                {
                    const __m512i raw0 =
                        _mm512_load_si512(packed.interleavedB(chunk, kb, group, zbase));
                    const __m512i raw1 =
                        _mm512_load_si512(packed.interleavedB(chunk, kb, group, zbase + 1));
                    const __m512i lo0 =
                        decode_low_nibbles_avx512(raw0, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                    const __m512i lo1 =
                        decode_low_nibbles_avx512(raw1, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                    const __m512i hi0 =
                        decode_high_nibbles_avx512(raw0, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                    const __m512i hi1 =
                        decode_high_nibbles_avx512(raw1, decode_lut, decode_kind, mask_0F, q4_zero_offset);

#define NIBBLE_3ROW_ACCUM(ROW, ABLK)                                             \
                    {                                                            \
                        const __m512i a_lo = _mm512_set1_epi32(                  \
                            pack_q8_1_unsigned_word(ABLK, group * 4));           \
                        ia##ROW##_0 = _mm512_dpbusd_epi32(ia##ROW##_0, a_lo, lo0); \
                        ia##ROW##_1 = _mm512_dpbusd_epi32(ia##ROW##_1, a_lo, lo1); \
                        const __m512i a_hi = _mm512_set1_epi32(                  \
                            pack_q8_1_unsigned_word(ABLK, group * 4 + 16));      \
                        ia##ROW##_0 = _mm512_dpbusd_epi32(ia##ROW##_0, a_hi, hi0); \
                        ia##ROW##_1 = _mm512_dpbusd_epi32(ia##ROW##_1, a_hi, hi1); \
                    }
                    NIBBLE_3ROW_ACCUM(0, a0)
                    NIBBLE_3ROW_ACCUM(1, a1)
                    NIBBLE_3ROW_ACCUM(2, a2)
#undef NIBBLE_3ROW_ACCUM
                }

                const int16_t *comp_ptr = packed.chunkComp(chunk, kb) + zbase * 16;
                const __m512i comp0 =
                    _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr)));
                const __m512i comp1 =
                    _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 16)));
                const __m512i bias0 = _mm512_mullo_epi32(bias_128_i32, comp0);
                const __m512i bias1 = _mm512_mullo_epi32(bias_128_i32, comp1);

                ia0_0 = _mm512_sub_epi32(ia0_0, bias0);
                ia1_0 = _mm512_sub_epi32(ia1_0, bias0);
                ia2_0 = _mm512_sub_epi32(ia2_0, bias0);
                ia0_1 = _mm512_sub_epi32(ia0_1, bias1);
                ia1_1 = _mm512_sub_epi32(ia1_1, bias1);
                ia2_1 = _mm512_sub_epi32(ia2_1, bias1);

                const uint16_t *scale_ptr = packed.chunkScales(chunk, kb) + zbase * 16;
                const __m512 bscale0 =
                    _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scale_ptr)));
                const __m512 bscale1 =
                    _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scale_ptr + 16)));
                const float a0_scale = simd::fp16_to_fp32(a0.d);
                const float a1_scale = simd::fp16_to_fp32(a1.d);
                const float a2_scale = simd::fp16_to_fp32(a2.d);

#define FMA_3ROW_NATIVE(ROW, ASCALE)                                            \
                {                                                               \
                    const __m512 as = _mm512_set1_ps(ASCALE);                   \
                    fp##ROW##_0 = _mm512_fmadd_ps(                              \
                        _mm512_cvtepi32_ps(ia##ROW##_0),                        \
                        _mm512_mul_ps(as, bscale0),                             \
                        fp##ROW##_0);                                           \
                    fp##ROW##_1 = _mm512_fmadd_ps(                              \
                        _mm512_cvtepi32_ps(ia##ROW##_1),                        \
                        _mm512_mul_ps(as, bscale1),                             \
                        fp##ROW##_1);                                           \
                }
                FMA_3ROW_NATIVE(0, a0_scale)
                FMA_3ROW_NATIVE(1, a1_scale)
                FMA_3ROW_NATIVE(2, a2_scale)
#undef FMA_3ROW_NATIVE

                if (packed.is_asymmetric)
                {
                    const uint16_t *min_ptr = packed.chunkMins(chunk, kb) + zbase * 16;
                    const __m512 bmin0 =
                        _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(min_ptr)));
                    const __m512 bmin1 =
                        _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(min_ptr + 16)));

#define MIN_3ROW_NATIVE(ROW, ABLK, ASCALE)                                      \
                    {                                                           \
                        const __m512 corr = _mm512_set1_ps(                     \
                            static_cast<float>(ABLK.sum_qs) * ASCALE);          \
                        fp##ROW##_0 = _mm512_fmadd_ps(corr, bmin0, fp##ROW##_0); \
                        fp##ROW##_1 = _mm512_fmadd_ps(corr, bmin1, fp##ROW##_1); \
                    }
                    MIN_3ROW_NATIVE(0, a0, a0_scale)
                    MIN_3ROW_NATIVE(1, a1, a1_scale)
                    MIN_3ROW_NATIVE(2, a2, a2_scale)
#undef MIN_3ROW_NATIVE
                }
            }

            _mm512_storeu_ps(C_row0 + zbase * 16, fp0_0);
            _mm512_storeu_ps(C_row0 + (zbase + 1) * 16, fp0_1);
            _mm512_storeu_ps(C_row1 + zbase * 16, fp1_0);
            _mm512_storeu_ps(C_row1 + (zbase + 1) * 16, fp1_1);
            _mm512_storeu_ps(C_row2 + zbase * 16, fp2_0);
            _mm512_storeu_ps(C_row2 + (zbase + 1) * 16, fp2_1);
        }
    }

    /**
     * @brief Three-row AVX512 verifier microkernel for INT8-predecoded formats.
     *
     * This is the INT8-layout sibling of gemm_3row_native_2z_chunk().  It is
     * decode-equivalent to three serial M=1 NativeVNNI GEMVs and shares each
     * interleaved B vector across all three verifier rows.
     */
    inline void gemm_3row_int8_2z_chunk(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_row0,
        const Q8_1Block *A_q8_row1,
        const Q8_1Block *A_q8_row2,
        float *C_row0,
        float *C_row1,
        float *C_row2,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate)
    {
        const __m512i bias_128_i32 = _mm512_set1_epi32(128);

        for (int zbase = 0; zbase < 4; zbase += 2)
        {
            __m512 fp0_0 = accumulate ? _mm512_loadu_ps(C_row0 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp0_1 = accumulate ? _mm512_loadu_ps(C_row0 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp1_0 = accumulate ? _mm512_loadu_ps(C_row1 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp1_1 = accumulate ? _mm512_loadu_ps(C_row1 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp2_0 = accumulate ? _mm512_loadu_ps(C_row2 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp2_1 = accumulate ? _mm512_loadu_ps(C_row2 + (zbase + 1) * 16) : _mm512_setzero_ps();

            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                const Q8_1Block &a0 = A_q8_row0[kb];
                const Q8_1Block &a1 = A_q8_row1[kb];
                const Q8_1Block &a2 = A_q8_row2[kb];

                __m512i ia0_0 = _mm512_setzero_si512(), ia0_1 = _mm512_setzero_si512();
                __m512i ia1_0 = _mm512_setzero_si512(), ia1_1 = _mm512_setzero_si512();
                __m512i ia2_0 = _mm512_setzero_si512(), ia2_1 = _mm512_setzero_si512();

                for (int group = 0; group < 8; ++group)
                {
                    const __m512i a0_bc =
                        _mm512_set1_epi32(pack_q8_1_unsigned_word(a0, group * 4));
                    const __m512i a1_bc =
                        _mm512_set1_epi32(pack_q8_1_unsigned_word(a1, group * 4));
                    const __m512i a2_bc =
                        _mm512_set1_epi32(pack_q8_1_unsigned_word(a2, group * 4));

                    const __m512i b0 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, zbase));
                    ia0_0 = _mm512_dpbusd_epi32(ia0_0, a0_bc, b0);
                    ia1_0 = _mm512_dpbusd_epi32(ia1_0, a1_bc, b0);
                    ia2_0 = _mm512_dpbusd_epi32(ia2_0, a2_bc, b0);

                    const __m512i b1 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, zbase + 1));
                    ia0_1 = _mm512_dpbusd_epi32(ia0_1, a0_bc, b1);
                    ia1_1 = _mm512_dpbusd_epi32(ia1_1, a1_bc, b1);
                    ia2_1 = _mm512_dpbusd_epi32(ia2_1, a2_bc, b1);
                }

                const int16_t *comp_ptr = packed.chunkComp(chunk, kb) + zbase * 16;
                const __m512i comp0 =
                    _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr)));
                const __m512i comp1 =
                    _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 16)));
                const __m512i bias0 = _mm512_mullo_epi32(bias_128_i32, comp0);
                const __m512i bias1 = _mm512_mullo_epi32(bias_128_i32, comp1);

                ia0_0 = _mm512_sub_epi32(ia0_0, bias0);
                ia1_0 = _mm512_sub_epi32(ia1_0, bias0);
                ia2_0 = _mm512_sub_epi32(ia2_0, bias0);
                ia0_1 = _mm512_sub_epi32(ia0_1, bias1);
                ia1_1 = _mm512_sub_epi32(ia1_1, bias1);
                ia2_1 = _mm512_sub_epi32(ia2_1, bias1);

                const uint16_t *scale_ptr = packed.chunkScales(chunk, kb) + zbase * 16;
                const __m512 bscale0 =
                    _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scale_ptr)));
                const __m512 bscale1 =
                    _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scale_ptr + 16)));
                const float a0_scale = simd::fp16_to_fp32(a0.d);
                const float a1_scale = simd::fp16_to_fp32(a1.d);
                const float a2_scale = simd::fp16_to_fp32(a2.d);

#define FMA_3ROW_INT8(ROW, ASCALE)                                              \
                {                                                               \
                    const __m512 as = _mm512_set1_ps(ASCALE);                   \
                    fp##ROW##_0 = _mm512_fmadd_ps(                              \
                        _mm512_cvtepi32_ps(ia##ROW##_0),                        \
                        _mm512_mul_ps(as, bscale0),                             \
                        fp##ROW##_0);                                           \
                    fp##ROW##_1 = _mm512_fmadd_ps(                              \
                        _mm512_cvtepi32_ps(ia##ROW##_1),                        \
                        _mm512_mul_ps(as, bscale1),                             \
                        fp##ROW##_1);                                           \
                }
                FMA_3ROW_INT8(0, a0_scale)
                FMA_3ROW_INT8(1, a1_scale)
                FMA_3ROW_INT8(2, a2_scale)
#undef FMA_3ROW_INT8

                if (packed.is_asymmetric)
                {
                    const uint16_t *min_ptr = packed.chunkMins(chunk, kb) + zbase * 16;
                    const __m512 bmin0 =
                        _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(min_ptr)));
                    const __m512 bmin1 =
                        _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(min_ptr + 16)));

#define MIN_3ROW_INT8(ROW, ABLK, ASCALE)                                        \
                    {                                                           \
                        const __m512 corr = _mm512_set1_ps(                     \
                            static_cast<float>(ABLK.sum_qs) * ASCALE);          \
                        fp##ROW##_0 = _mm512_fmadd_ps(corr, bmin0, fp##ROW##_0); \
                        fp##ROW##_1 = _mm512_fmadd_ps(corr, bmin1, fp##ROW##_1); \
                    }
                    MIN_3ROW_INT8(0, a0, a0_scale)
                    MIN_3ROW_INT8(1, a1, a1_scale)
                    MIN_3ROW_INT8(2, a2, a2_scale)
#undef MIN_3ROW_INT8
                }
            }

            _mm512_storeu_ps(C_row0 + zbase * 16, fp0_0);
            _mm512_storeu_ps(C_row0 + (zbase + 1) * 16, fp0_1);
            _mm512_storeu_ps(C_row1 + zbase * 16, fp1_0);
            _mm512_storeu_ps(C_row1 + (zbase + 1) * 16, fp1_1);
            _mm512_storeu_ps(C_row2 + zbase * 16, fp2_0);
            _mm512_storeu_ps(C_row2 + (zbase + 1) * 16, fp2_1);
        }
    }

    /**
     * @brief Four-row AVX512 verifier microkernel for nibble-LUT formats.
     *
     * Q4_0, Q4_1/Q4_K, IQ4_NL/IQ4_XS and their aliases decode packed nibbles
     * through the same semantic helper as the M=1 GEMV path: linear Q4
     * codebooks use byte arithmetic, while non-linear IQ4 codebooks use the
     * per-codebook LUT.  The verifier-specialized kernel keeps the exact M=1
     * accumulation order: each K block first builds the INT32 dot-product
     * accumulators, then applies compensation, scale and optional min
     * correction once.  It simply shares the decoded B vectors across four
     * independent verifier rows.
     *
     * The implementation processes two 16-column z-lanes at a time.  That is a
     * deliberate register-pressure compromise: it gives B decode reuse for M=4
     * without carrying the full 4-row x 64-column accumulator set live at once.
     */
    inline void gemm_4row_native_2z_chunk(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_row0,
        const Q8_1Block *A_q8_row1,
        const Q8_1Block *A_q8_row2,
        const Q8_1Block *A_q8_row3,
        float *C_row0,
        float *C_row1,
        float *C_row2,
        float *C_row3,
        int chunk,
        int kb_start,
        int kb_end,
        const __m512i decode_lut,
        bool accumulate)
    {
        const __m512i bias_128_i32 = _mm512_set1_epi32(128);
        const NibbleDecodeKind decode_kind = nibbleDecodeKind(packed.codebook_id);
        const __m512i mask_0F = _mm512_set1_epi8(0x0F);
        const __m512i q4_zero_offset = _mm512_set1_epi8(8);

        for (int zbase = 0; zbase < 4; zbase += 2)
        {
            __m512 fp0_0 = accumulate ? _mm512_loadu_ps(C_row0 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp0_1 = accumulate ? _mm512_loadu_ps(C_row0 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp1_0 = accumulate ? _mm512_loadu_ps(C_row1 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp1_1 = accumulate ? _mm512_loadu_ps(C_row1 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp2_0 = accumulate ? _mm512_loadu_ps(C_row2 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp2_1 = accumulate ? _mm512_loadu_ps(C_row2 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp3_0 = accumulate ? _mm512_loadu_ps(C_row3 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp3_1 = accumulate ? _mm512_loadu_ps(C_row3 + (zbase + 1) * 16) : _mm512_setzero_ps();

            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                const Q8_1Block &a0 = A_q8_row0[kb];
                const Q8_1Block &a1 = A_q8_row1[kb];
                const Q8_1Block &a2 = A_q8_row2[kb];
                const Q8_1Block &a3 = A_q8_row3[kb];

                __m512i ia0_0 = _mm512_setzero_si512(), ia0_1 = _mm512_setzero_si512();
                __m512i ia1_0 = _mm512_setzero_si512(), ia1_1 = _mm512_setzero_si512();
                __m512i ia2_0 = _mm512_setzero_si512(), ia2_1 = _mm512_setzero_si512();
                __m512i ia3_0 = _mm512_setzero_si512(), ia3_1 = _mm512_setzero_si512();

                for (int group = 0; group < 4; ++group)
                {
                    const __m512i raw0 =
                        _mm512_load_si512(packed.interleavedB(chunk, kb, group, zbase));
                    const __m512i raw1 =
                        _mm512_load_si512(packed.interleavedB(chunk, kb, group, zbase + 1));
                    const __m512i lo0 =
                        decode_low_nibbles_avx512(raw0, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                    const __m512i lo1 =
                        decode_low_nibbles_avx512(raw1, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                    const __m512i hi0 =
                        decode_high_nibbles_avx512(raw0, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                    const __m512i hi1 =
                        decode_high_nibbles_avx512(raw1, decode_lut, decode_kind, mask_0F, q4_zero_offset);

#define NIBBLE_4ROW_ACCUM(ROW, ABLK)                                             \
                    {                                                            \
                        const __m512i a_lo = _mm512_set1_epi32(                  \
                            pack_q8_1_unsigned_word(ABLK, group * 4));           \
                        ia##ROW##_0 = _mm512_dpbusd_epi32(ia##ROW##_0, a_lo, lo0); \
                        ia##ROW##_1 = _mm512_dpbusd_epi32(ia##ROW##_1, a_lo, lo1); \
                        const __m512i a_hi = _mm512_set1_epi32(                  \
                            pack_q8_1_unsigned_word(ABLK, group * 4 + 16));      \
                        ia##ROW##_0 = _mm512_dpbusd_epi32(ia##ROW##_0, a_hi, hi0); \
                        ia##ROW##_1 = _mm512_dpbusd_epi32(ia##ROW##_1, a_hi, hi1); \
                    }
                    NIBBLE_4ROW_ACCUM(0, a0)
                    NIBBLE_4ROW_ACCUM(1, a1)
                    NIBBLE_4ROW_ACCUM(2, a2)
                    NIBBLE_4ROW_ACCUM(3, a3)
#undef NIBBLE_4ROW_ACCUM
                }

                const int16_t *comp_ptr = packed.chunkComp(chunk, kb) + zbase * 16;
                const __m512i comp0 =
                    _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr)));
                const __m512i comp1 =
                    _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 16)));
                const __m512i bias0 = _mm512_mullo_epi32(bias_128_i32, comp0);
                const __m512i bias1 = _mm512_mullo_epi32(bias_128_i32, comp1);

                ia0_0 = _mm512_sub_epi32(ia0_0, bias0);
                ia1_0 = _mm512_sub_epi32(ia1_0, bias0);
                ia2_0 = _mm512_sub_epi32(ia2_0, bias0);
                ia3_0 = _mm512_sub_epi32(ia3_0, bias0);
                ia0_1 = _mm512_sub_epi32(ia0_1, bias1);
                ia1_1 = _mm512_sub_epi32(ia1_1, bias1);
                ia2_1 = _mm512_sub_epi32(ia2_1, bias1);
                ia3_1 = _mm512_sub_epi32(ia3_1, bias1);

                const uint16_t *scale_ptr = packed.chunkScales(chunk, kb) + zbase * 16;
                const __m512 bscale0 =
                    _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scale_ptr)));
                const __m512 bscale1 =
                    _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scale_ptr + 16)));

#define FMA_4ROW_NATIVE(ROW, ABLK, ASCALE)                                      \
                {                                                               \
                    const __m512 as = _mm512_set1_ps(ASCALE);                   \
                    fp##ROW##_0 = _mm512_fmadd_ps(                              \
                        _mm512_cvtepi32_ps(ia##ROW##_0),                        \
                        _mm512_mul_ps(as, bscale0),                             \
                        fp##ROW##_0);                                           \
                    fp##ROW##_1 = _mm512_fmadd_ps(                              \
                        _mm512_cvtepi32_ps(ia##ROW##_1),                        \
                        _mm512_mul_ps(as, bscale1),                             \
                        fp##ROW##_1);                                           \
                }
                const float a0_scale = simd::fp16_to_fp32(a0.d);
                const float a1_scale = simd::fp16_to_fp32(a1.d);
                const float a2_scale = simd::fp16_to_fp32(a2.d);
                const float a3_scale = simd::fp16_to_fp32(a3.d);
                FMA_4ROW_NATIVE(0, a0, a0_scale)
                FMA_4ROW_NATIVE(1, a1, a1_scale)
                FMA_4ROW_NATIVE(2, a2, a2_scale)
                FMA_4ROW_NATIVE(3, a3, a3_scale)
#undef FMA_4ROW_NATIVE

                if (packed.is_asymmetric)
                {
                    const uint16_t *min_ptr = packed.chunkMins(chunk, kb) + zbase * 16;
                    const __m512 bmin0 =
                        _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(min_ptr)));
                    const __m512 bmin1 =
                        _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(min_ptr + 16)));

#define MIN_4ROW_NATIVE(ROW, ABLK, ASCALE)                                      \
                    {                                                           \
                        const __m512 corr = _mm512_set1_ps(                     \
                            static_cast<float>(ABLK.sum_qs) * ASCALE);          \
                        fp##ROW##_0 = _mm512_fmadd_ps(corr, bmin0, fp##ROW##_0); \
                        fp##ROW##_1 = _mm512_fmadd_ps(corr, bmin1, fp##ROW##_1); \
                    }
                    MIN_4ROW_NATIVE(0, a0, a0_scale)
                    MIN_4ROW_NATIVE(1, a1, a1_scale)
                    MIN_4ROW_NATIVE(2, a2, a2_scale)
                    MIN_4ROW_NATIVE(3, a3, a3_scale)
#undef MIN_4ROW_NATIVE
                }
            }

            _mm512_storeu_ps(C_row0 + zbase * 16, fp0_0);
            _mm512_storeu_ps(C_row0 + (zbase + 1) * 16, fp0_1);
            _mm512_storeu_ps(C_row1 + zbase * 16, fp1_0);
            _mm512_storeu_ps(C_row1 + (zbase + 1) * 16, fp1_1);
            _mm512_storeu_ps(C_row2 + zbase * 16, fp2_0);
            _mm512_storeu_ps(C_row2 + (zbase + 1) * 16, fp2_1);
            _mm512_storeu_ps(C_row3 + zbase * 16, fp3_0);
            _mm512_storeu_ps(C_row3 + (zbase + 1) * 16, fp3_1);
        }
    }

    /**
     * @brief Four-row AVX512 verifier microkernel for INT8-predecoded formats.
     *
     * This is the first real M=4 CPU verifier candidate.  It keeps the same
     * decode-equivalent order as four independent one-token GEMVs:
     * for each K block, compute the INT32 dot for a column group, apply the
     * same compensation/scales/mins, then accumulate into FP32.  The economy
     * comes from loading each interleaved B vector once and applying it to four
     * independent activation rows.
     *
     * The kernel works on two 16-column subchunks at a time.  That gives useful
     * B reuse without the register pressure of a full 4-row x 64-column kernel.
     */
    inline void gemm_4row_int8_2z_chunk(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_row0,
        const Q8_1Block *A_q8_row1,
        const Q8_1Block *A_q8_row2,
        const Q8_1Block *A_q8_row3,
        float *C_row0,
        float *C_row1,
        float *C_row2,
        float *C_row3,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate)
    {
        const __m512i bias_128_i32 = _mm512_set1_epi32(128);

        for (int zbase = 0; zbase < 4; zbase += 2)
        {
            __m512 fp0_0 = accumulate ? _mm512_loadu_ps(C_row0 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp0_1 = accumulate ? _mm512_loadu_ps(C_row0 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp1_0 = accumulate ? _mm512_loadu_ps(C_row1 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp1_1 = accumulate ? _mm512_loadu_ps(C_row1 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp2_0 = accumulate ? _mm512_loadu_ps(C_row2 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp2_1 = accumulate ? _mm512_loadu_ps(C_row2 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp3_0 = accumulate ? _mm512_loadu_ps(C_row3 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp3_1 = accumulate ? _mm512_loadu_ps(C_row3 + (zbase + 1) * 16) : _mm512_setzero_ps();

            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                const Q8_1Block &a0 = A_q8_row0[kb];
                const Q8_1Block &a1 = A_q8_row1[kb];
                const Q8_1Block &a2 = A_q8_row2[kb];
                const Q8_1Block &a3 = A_q8_row3[kb];

                __m512i ia0_0 = _mm512_setzero_si512(), ia0_1 = _mm512_setzero_si512();
                __m512i ia1_0 = _mm512_setzero_si512(), ia1_1 = _mm512_setzero_si512();
                __m512i ia2_0 = _mm512_setzero_si512(), ia2_1 = _mm512_setzero_si512();
                __m512i ia3_0 = _mm512_setzero_si512(), ia3_1 = _mm512_setzero_si512();

                for (int group = 0; group < 8; ++group)
                {
                    const __m512i a0_bc =
                        _mm512_set1_epi32(pack_q8_1_unsigned_word(a0, group * 4));
                    const __m512i a1_bc =
                        _mm512_set1_epi32(pack_q8_1_unsigned_word(a1, group * 4));
                    const __m512i a2_bc =
                        _mm512_set1_epi32(pack_q8_1_unsigned_word(a2, group * 4));
                    const __m512i a3_bc =
                        _mm512_set1_epi32(pack_q8_1_unsigned_word(a3, group * 4));

                    const __m512i b0 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, zbase));
                    ia0_0 = _mm512_dpbusd_epi32(ia0_0, a0_bc, b0);
                    ia1_0 = _mm512_dpbusd_epi32(ia1_0, a1_bc, b0);
                    ia2_0 = _mm512_dpbusd_epi32(ia2_0, a2_bc, b0);
                    ia3_0 = _mm512_dpbusd_epi32(ia3_0, a3_bc, b0);

                    const __m512i b1 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, zbase + 1));
                    ia0_1 = _mm512_dpbusd_epi32(ia0_1, a0_bc, b1);
                    ia1_1 = _mm512_dpbusd_epi32(ia1_1, a1_bc, b1);
                    ia2_1 = _mm512_dpbusd_epi32(ia2_1, a2_bc, b1);
                    ia3_1 = _mm512_dpbusd_epi32(ia3_1, a3_bc, b1);
                }

                const int16_t *comp_ptr = packed.chunkComp(chunk, kb) + zbase * 16;
                const __m512i comp0 =
                    _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr)));
                const __m512i comp1 =
                    _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 16)));
                const __m512i bias0 = _mm512_mullo_epi32(bias_128_i32, comp0);
                const __m512i bias1 = _mm512_mullo_epi32(bias_128_i32, comp1);

                ia0_0 = _mm512_sub_epi32(ia0_0, bias0);
                ia1_0 = _mm512_sub_epi32(ia1_0, bias0);
                ia2_0 = _mm512_sub_epi32(ia2_0, bias0);
                ia3_0 = _mm512_sub_epi32(ia3_0, bias0);
                ia0_1 = _mm512_sub_epi32(ia0_1, bias1);
                ia1_1 = _mm512_sub_epi32(ia1_1, bias1);
                ia2_1 = _mm512_sub_epi32(ia2_1, bias1);
                ia3_1 = _mm512_sub_epi32(ia3_1, bias1);

                const uint16_t *scale_ptr = packed.chunkScales(chunk, kb) + zbase * 16;
                const __m512 bscale0 =
                    _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scale_ptr)));
                const __m512 bscale1 =
                    _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scale_ptr + 16)));

#define FMA_4ROW_INT8(ROW, ABLK, ASCALE)                                           \
                {                                                                  \
                    const __m512 as = _mm512_set1_ps(ASCALE);                      \
                    fp##ROW##_0 = _mm512_fmadd_ps(                                 \
                        _mm512_cvtepi32_ps(ia##ROW##_0), _mm512_mul_ps(as, bscale0), fp##ROW##_0); \
                    fp##ROW##_1 = _mm512_fmadd_ps(                                 \
                        _mm512_cvtepi32_ps(ia##ROW##_1), _mm512_mul_ps(as, bscale1), fp##ROW##_1); \
                }
                const float a0_scale = simd::fp16_to_fp32(a0.d);
                const float a1_scale = simd::fp16_to_fp32(a1.d);
                const float a2_scale = simd::fp16_to_fp32(a2.d);
                const float a3_scale = simd::fp16_to_fp32(a3.d);
                FMA_4ROW_INT8(0, a0, a0_scale)
                FMA_4ROW_INT8(1, a1, a1_scale)
                FMA_4ROW_INT8(2, a2, a2_scale)
                FMA_4ROW_INT8(3, a3, a3_scale)
#undef FMA_4ROW_INT8

                if (packed.is_asymmetric)
                {
                    const uint16_t *min_ptr = packed.chunkMins(chunk, kb) + zbase * 16;
                    const __m512 bmin0 =
                        _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(min_ptr)));
                    const __m512 bmin1 =
                        _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(min_ptr + 16)));

#define MIN_4ROW_INT8(ROW, ABLK, ASCALE)                                  \
                    {                                                      \
                        const __m512 corr = _mm512_set1_ps(                \
                            static_cast<float>(ABLK.sum_qs) * ASCALE);     \
                        fp##ROW##_0 = _mm512_fmadd_ps(corr, bmin0, fp##ROW##_0); \
                        fp##ROW##_1 = _mm512_fmadd_ps(corr, bmin1, fp##ROW##_1); \
                    }
                    MIN_4ROW_INT8(0, a0, a0_scale)
                    MIN_4ROW_INT8(1, a1, a1_scale)
                    MIN_4ROW_INT8(2, a2, a2_scale)
                    MIN_4ROW_INT8(3, a3, a3_scale)
#undef MIN_4ROW_INT8
                }
            }

            _mm512_storeu_ps(C_row0 + zbase * 16, fp0_0);
            _mm512_storeu_ps(C_row0 + (zbase + 1) * 16, fp0_1);
            _mm512_storeu_ps(C_row1 + zbase * 16, fp1_0);
            _mm512_storeu_ps(C_row1 + (zbase + 1) * 16, fp1_1);
            _mm512_storeu_ps(C_row2 + zbase * 16, fp2_0);
            _mm512_storeu_ps(C_row2 + (zbase + 1) * 16, fp2_1);
            _mm512_storeu_ps(C_row3 + zbase * 16, fp3_0);
            _mm512_storeu_ps(C_row3 + (zbase + 1) * 16, fp3_1);
        }
    }

#endif // __AVX512F__ && __AVX512VNNI__ && __AVX512BW__

    // =========================================================================
    // Full GEMM dispatcher (M>1) — dual-strategy
    // =========================================================================
    //
    // Strategy 1 — Small-N (N-tasks <= threads/4):
    //   M×N 2D task grid. Tasks ordered (chunk, m-row) so consecutive
    //   M rows for the same chunk land on the same thread (B stays in L2).
    //   Each task calls the GEMV chunk kernel directly — no new SIMD code.
    //   Activates when N-only parallelism fills <25% of threads.
    //
    // Strategy 2 — Tiled GEMM (N-tasks > threads/4):
    //   Outer loop: parallel over N-block chunks
    //   Middle loop: K-tiles (each fits in L2, reused across all M rows)
    //   Inner loop: M rows (2 at a time via 2-row microkernel)
    //   B-tile reuse: each N-chunk × K-tile loaded once, scanned M times.
    // =========================================================================

    // =========================================================================
    // Shared activation quantization (for multiply_fused quantize-once)
    // =========================================================================

    inline void quantize_activations_to_q8_1(
        const float *A_fp32,
        Q8_1Block *A_q8,
        int M,
        int K,
        int K_blocks)
    {
        const bool k_aligned = (K % 32 == 0);

        auto do_quantize = [&]()
        {
#pragma omp for schedule(static)
            for (int m = 0; m < M; ++m)
            {
                const float *row_a = A_fp32 + m * K;
                Q8_1Block *row_q8 = A_q8 + static_cast<size_t>(m) * K_blocks;
                int kb = 0;
#if defined(__AVX512F__)
                if (k_aligned)
                {
                    for (; kb + 1 < K_blocks; kb += 2)
                    {
                        simd::quantize_two_blocks_avx512(row_a + kb * 32, row_q8[kb], row_q8[kb + 1]);
                    }
                }
#endif
                for (; kb < K_blocks; ++kb)
                {
                    int block_start = kb * 32;
                    int block_len = std::min(32, K - block_start);
                    simd::quantize_single_block(row_a + block_start, row_q8[kb], block_len);
                }
            }
        };

        OMP_WORKSHARE_REGION(do_quantize);
    }

    // =========================================================================
    // Pre-quantized GEMM (M>1) — compute only, skips quantization
    // =========================================================================
    //
    // A_q8_all: Pre-quantized Q8_1 blocks, [M * K_blocks] contiguous layout.
    //           Row m starts at A_q8_all + m * K_blocks.
    // =========================================================================

    inline void gemm_native_vnni_preq(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_all,
        float *C,
        int M,
        int ldc,
        ISAPath isa_path = ISAPath::AUTO,
        VerifierRowsPolicy verifier_policy_override = VerifierRowsPolicy::Auto)
    {
        const int N = packed.N;
        const int K_blocks = packed.blocks_per_row;
        const int N_chunks = (N + 63) / 64;

        // Runtime ISA selection
        const ISALevel active_isa = activeISALevel();
        bool use_avx512 = false;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        use_avx512 = (isa_path == ISAPath::AUTO) ? (active_isa >= ISALevel::AVX512)
                                                 : (isa_path == ISAPath::AVX512);
#endif
        const bool use_avx2 =
            !use_avx512 &&
            ((isa_path == ISAPath::AUTO && active_isa >= ISALevel::AVX2) ||
             isa_path == ISAPath::AVX2);

        int num_threads = omp_get_max_threads();
        NativeVNNITileConfig cfg = computeTileConfig(N, packed.K, M, packed.payload_bytes, num_threads);
        int n_block_chunks = cfg.n_block_chunks;
        int total_n_blocks = (N_chunks + n_block_chunks - 1) / n_block_chunks;

        int k_tile_blocks = cfg.k_tile_blocks > 0 ? cfg.k_tile_blocks : K_blocks;
        int num_k_tiles = (K_blocks + k_tile_blocks - 1) / k_tile_blocks;

        // Initialize decode LUTs for selected ISA
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        __m512i decode_lut_512 = _mm512_setzero_si512();
        if (use_avx512 && packed.is_nibble_lut)
            decode_lut_512 = build_decode_lut(packed.codebook_id);
#endif
        __m256i decode_lut_256 = _mm256_setzero_si256();
        if (!use_avx512 && packed.is_nibble_lut)
            decode_lut_256 = build_decode_lut_avx2_for_codebook(packed.codebook_id);

        // Small-N dispatch: M×N 2D parallel grid
        // Also forced when ldc < 64: the 2-row tiled microkernel always stores
        // 64 floats per row via _mm512_storeu_ps, which overflows into adjacent
        // rows when the output stride is narrower than one VNNI chunk.
        if ((total_n_blocks <= num_threads / 4 && M >= 2) || ldc < 64)
        {
            auto do_compute = [&]()
            {
                int total_tasks = N_chunks * M;
#pragma omp for schedule(static)
                for (int task = 0; task < total_tasks; ++task)
                {
                    int chunk = task / M;
                    int m = task % M;
                    const Q8_1Block *aq = A_q8_all + static_cast<size_t>(m) * K_blocks;
                    float *c_out = C + m * ldc + chunk * 64;

                    int n_cols_actual = std::min(64, N - chunk * 64);
                    if (n_cols_actual < 64)
                    {
                        alignas(64) float tmp[64];
                        if (use_avx512)
                        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                            if (packed.is_nibble_lut)
                                gemv_native_vnni_avx512_chunk_native(packed, aq, tmp, chunk, 0, K_blocks, decode_lut_512);
                            else
                                gemv_native_vnni_avx512_chunk_int8(packed, aq, tmp, chunk, 0, K_blocks);
#endif
                        }
                        else
                        {
                            if (packed.is_nibble_lut)
                                gemv_avx2_chunk_native(packed, aq, tmp, chunk, 0, K_blocks, decode_lut_256);
                            else
                                gemv_avx2_chunk_int8(packed, aq, tmp, chunk, 0, K_blocks);
                        }
                        std::memcpy(c_out, tmp, n_cols_actual * sizeof(float));
                    }
                    else
                    {
                        if (use_avx512)
                        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                            if (packed.is_nibble_lut)
                                gemv_native_vnni_avx512_chunk_native(packed, aq, c_out, chunk, 0, K_blocks, decode_lut_512);
                            else
                                gemv_native_vnni_avx512_chunk_int8(packed, aq, c_out, chunk, 0, K_blocks);
#endif
                        }
                        else
                        {
                            if (packed.is_nibble_lut)
                                gemv_avx2_chunk_native(packed, aq, c_out, chunk, 0, K_blocks, decode_lut_256);
                            else
                                gemv_avx2_chunk_int8(packed, aq, c_out, chunk, 0, K_blocks);
                        }
                    }
                }
            };
            OMP_WORKSHARE_REGION(do_compute);
            return;
        }

        // Tiled GEMM path: N-parallel with 2-row microkernels and K-tiling
        auto do_compute = [&]()
        {
#pragma omp for schedule(static)
            for (int block_idx = 0; block_idx < total_n_blocks; ++block_idx)
            {
                int chunk_start = block_idx * n_block_chunks;
                int chunk_count = std::min(n_block_chunks, N_chunks - chunk_start);

                for (int kt = 0; kt < num_k_tiles; ++kt)
                {
                    int kb_start = kt * k_tile_blocks;
                    int kb_end = std::min(kb_start + k_tile_blocks, K_blocks);
                    bool accum = (kt > 0);

                    int m = 0;
                    for (; m + 1 < M; m += 2)
                    {
                        const Q8_1Block *aq0 = A_q8_all + static_cast<size_t>(m) * K_blocks;
                        const Q8_1Block *aq1 = A_q8_all + static_cast<size_t>(m + 1) * K_blocks;

                        for (int ci = 0; ci < chunk_count; ++ci)
                        {
                            int chunk = chunk_start + ci;
                            int n_start = chunk * 64;
                            float *c0 = C + m * ldc + n_start;
                            float *c1 = C + (m + 1) * ldc + n_start;

                            if (use_avx512)
                            {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                                if (packed.is_nibble_lut)
                                    gemm_2row_native_chunk(packed, aq0, aq1, c0, c1,
                                                           chunk, kb_start, kb_end,
                                                           decode_lut_512, accum);
                                else
                                    gemm_2row_int8_chunk(packed, aq0, aq1, c0, c1,
                                                         chunk, kb_start, kb_end, accum);
#endif
                            }
                            else
                            {
                                if (packed.is_nibble_lut)
                                    gemm_2row_native_chunk_avx2(packed, aq0, aq1, c0, c1,
                                                                chunk, kb_start, kb_end,
                                                                decode_lut_256, accum);
                                else
                                    gemm_2row_int8_chunk_avx2(packed, aq0, aq1, c0, c1,
                                                              chunk, kb_start, kb_end, accum);
                            }
                        }
                    }

                    // Odd M tail: single row
                    if (m < M)
                    {
                        const Q8_1Block *aq = A_q8_all + static_cast<size_t>(m) * K_blocks;
                        for (int ci = 0; ci < chunk_count; ++ci)
                        {
                            int chunk = chunk_start + ci;
                            int n_start = chunk * 64;
                            float *c_row = C + m * ldc + n_start;

                            if (use_avx512)
                            {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                                if (accum)
                                {
                                    alignas(64) float tmp[64] = {};
                                    if (packed.is_nibble_lut)
                                        gemv_native_vnni_avx512_chunk_native(packed, aq, tmp, chunk, kb_start, kb_end, decode_lut_512);
                                    else
                                        gemv_native_vnni_avx512_chunk_int8(packed, aq, tmp, chunk, kb_start, kb_end);
                                    __m512 s0 = _mm512_add_ps(_mm512_loadu_ps(c_row), _mm512_load_ps(tmp));
                                    __m512 s1 = _mm512_add_ps(_mm512_loadu_ps(c_row + 16), _mm512_load_ps(tmp + 16));
                                    __m512 s2 = _mm512_add_ps(_mm512_loadu_ps(c_row + 32), _mm512_load_ps(tmp + 32));
                                    __m512 s3 = _mm512_add_ps(_mm512_loadu_ps(c_row + 48), _mm512_load_ps(tmp + 48));
                                    _mm512_storeu_ps(c_row, s0);
                                    _mm512_storeu_ps(c_row + 16, s1);
                                    _mm512_storeu_ps(c_row + 32, s2);
                                    _mm512_storeu_ps(c_row + 48, s3);
                                }
                                else
                                {
                                    if (packed.is_nibble_lut)
                                        gemv_native_vnni_avx512_chunk_native(packed, aq, c_row, chunk, kb_start, kb_end, decode_lut_512);
                                    else
                                        gemv_native_vnni_avx512_chunk_int8(packed, aq, c_row, chunk, kb_start, kb_end);
                                }
#endif
                            }
                            else
                            {
                                if (accum)
                                {
                                    alignas(64) float tmp[64] = {};
                                    if (packed.is_nibble_lut)
                                        gemv_avx2_chunk_native(packed, aq, tmp, chunk, kb_start, kb_end, decode_lut_256);
                                    else
                                        gemv_avx2_chunk_int8(packed, aq, tmp, chunk, kb_start, kb_end);
                                    for (int i = 0; i < 64; i += 8)
                                    {
                                        __m256 s = _mm256_add_ps(
                                            _mm256_loadu_ps(c_row + i),
                                            _mm256_load_ps(tmp + i));
                                        _mm256_storeu_ps(c_row + i, s);
                                    }
                                }
                                else
                                {
                                    if (packed.is_nibble_lut)
                                        gemv_avx2_chunk_native(packed, aq, c_row, chunk, kb_start, kb_end, decode_lut_256);
                                    else
                                        gemv_avx2_chunk_int8(packed, aq, c_row, chunk, kb_start, kb_end);
                                }
                            }
                        }
                    }
                }

                int last_chunk_start = (chunk_start + chunk_count - 1) * 64;
                int last_n_cols = std::min(64, N - last_chunk_start);
                if (last_n_cols < 64)
                {
                    for (int m_row = 0; m_row < M; ++m_row)
                    {
                        for (int i = last_n_cols; i < 64; ++i)
                            C[m_row * ldc + last_chunk_start + i] = 0.0f;
                    }
                }
            }
        };

        OMP_WORKSHARE_REGION(do_compute);
    }

    /**
     * @brief Grouped M=2..4 verifier GEMM with M=1 decode-equivalent row math.
     *
     * The regular M>1 GEMM path is free to use 2-row microkernels and K tiling
     * that change FP32 accumulation order.  That is fine for ordinary prefill,
     * but not for MTP verifier rows that later publish recurrent/KV state:
     * those rows must match running M individual decode GEMVs.  This helper
     * keeps the work grouped by parallelizing across `(row, N-block[, K-tile])`
     * tasks while preserving the exact per-row GEMV chunk and reduction order
     * used by gemv_native_vnni_preq().
     */
    inline void gemm_native_vnni_preq_decode_equivalent_rows(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_all,
        float *C,
        int M,
        int ldc,
        ISAPath isa_path = ISAPath::AUTO,
        VerifierRowsPolicy verifier_policy_override = VerifierRowsPolicy::Auto)
    {
        const int N = packed.N;
        const int K = packed.K;
        const int K_blocks = packed.blocks_per_row;
        const int N_chunks = (N + 63) / 64;

        const ISALevel active_isa = activeISALevel();
        bool use_avx512 = false;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        use_avx512 = (isa_path == ISAPath::AUTO) ? (active_isa >= ISALevel::AVX512)
                                                 : (isa_path == ISAPath::AVX512);
#endif
        const bool use_avx2 =
            !use_avx512 &&
            ((isa_path == ISAPath::AUTO && active_isa >= ISALevel::AVX2) ||
             isa_path == ISAPath::AVX2);
        const VerifierRowsPolicy verifier_policy =
            verifier_policy_override == VerifierRowsPolicy::Auto
                ? selectVerifierRowsPolicy(packed, M, N, K)
                : verifier_policy_override;
        const bool use_wide_rows =
            verifier_policy == VerifierRowsPolicy::WideRows;

        const int num_threads = omp_get_max_threads();
        NativeVNNITileConfig cfg = computeTileConfig(N, K, 1, packed.payload_bytes, num_threads);

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        __m512i decode_lut_512 = _mm512_setzero_si512();
        if (use_avx512 && packed.is_nibble_lut)
            decode_lut_512 = build_decode_lut(packed.codebook_id);
#endif
        __m256i decode_lut_256 = _mm256_setzero_si256();
        if (use_avx2 && packed.is_nibble_lut)
            decode_lut_256 = build_decode_lut_avx2_for_codebook(packed.codebook_id);

        if (!use_avx512 && !use_avx2)
        {
            auto do_scalar_rows = [&]()
            {
#pragma omp for schedule(static)
                for (int row = 0; row < M; ++row)
                {
                    gemv_native_vnni_scalar(
                        packed,
                        A_q8_all + static_cast<size_t>(row) * K_blocks,
                        C + static_cast<size_t>(row) * ldc,
                        N,
                        K_blocks);
                }
            };
            OMP_WORKSHARE_REGION(do_scalar_rows);
            return;
        }

        if (cfg.k_tiles > 1)
        {
            const int k_tiles = cfg.k_tiles;
            const int k_blocks_per_tile = (K_blocks + k_tiles - 1) / k_tiles;
            const int n_block_chunks = cfg.n_block_chunks;
            const int total_blocks = (N_chunks + n_block_chunks - 1) / n_block_chunks;

            const size_t partial_sum_floats =
                static_cast<size_t>(M) * static_cast<size_t>(N_chunks) *
                static_cast<size_t>(k_tiles) * 64;
            /*
             * Tile kernels write every partial slot before reduction. Reuse
             * thread-local aligned storage so tiny verifier rows do not pay
             * malloc/free or zero-fill cost on every decode step.
             */
            thread_local AlignedVector<float> partial_sums_tls;
            if (partial_sums_tls.size() < partial_sum_floats)
                partial_sums_tls.resize_uninitialized(partial_sum_floats);
            float *partial_sums = partial_sums_tls.data();

            auto do_kpar_partial_rows = [&]()
            {
                /*
                 * Preserve the exact serial decode K-parallel contract:
                 *
                 *   1. compute independent [chunk, K-tile] partial sums
                 *   2. reduce tile0 + tile1 + ... in increasing tile order
                 *
                 * This intentionally uses the same single-row tile kernels as
                 * gemv_native_vnni_preq().  The workspace is modest for the
                 * long-K verifier shapes that need it, and it buys back
                 * K-tile parallelism without taking the non-equivalent
                 * multi-row microkernels.
                 */
                /*
                 * AVX512 can share each B tile across M=2..4 rows as long as
                 * the shared kernel writes an independent tile partial.  The
                 * reduction below is still the M=1 reduction, so the only
                 * arithmetic that must match is one tile's partial output.
                 */
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                if (use_avx512 && M >= 2 && M <= 4 &&
                    (M == 2 || use_wide_rows))
                {
                    const int total_shared_tile_tasks = N_chunks * k_tiles;
#pragma omp for schedule(static)
                    for (int task = 0; task < total_shared_tile_tasks; ++task)
                    {
                        const int kt = task % k_tiles;
                        const int chunk = task / k_tiles;
                        const int kb_start = kt * k_blocks_per_tile;
                        const int kb_end =
                            std::min(kb_start + k_blocks_per_tile, K_blocks);
                        float *dst0 =
                            partial_sums +
                            ((static_cast<size_t>(chunk) * k_tiles) + kt) * 64;
                        float *dst1 =
                            partial_sums +
                            (((static_cast<size_t>(N_chunks) + chunk) *
                              k_tiles) +
                             kt) *
                                64;
                        float *dst2 = nullptr;
                        float *dst3 = nullptr;
                        if (M >= 3)
                        {
                            dst2 =
                                partial_sums +
                                (((static_cast<size_t>(2) * N_chunks + chunk) *
                                  k_tiles) +
                                 kt) *
                                    64;
                        }
                        if (M >= 4)
                        {
                            dst3 =
                                partial_sums +
                                (((static_cast<size_t>(3) * N_chunks + chunk) *
                                  k_tiles) +
                                 kt) *
                                    64;
                        }

                        const Q8_1Block *row0_q8 = A_q8_all;
                        const Q8_1Block *row1_q8 = A_q8_all + K_blocks;
                        const Q8_1Block *row2_q8 =
                            A_q8_all + static_cast<size_t>(2) * K_blocks;
                        const Q8_1Block *row3_q8 =
                            A_q8_all + static_cast<size_t>(3) * K_blocks;

                        if (M == 2)
                        {
                            if (packed.is_nibble_lut)
                                gemm_2row_native_chunk(
                                    packed, row0_q8, row1_q8, dst0, dst1,
                                    chunk, kb_start, kb_end, decode_lut_512,
                                    /*accumulate=*/false);
                            else
                                gemm_2row_int8_chunk(
                                    packed, row0_q8, row1_q8, dst0, dst1,
                                    chunk, kb_start, kb_end,
                                    /*accumulate=*/false);
                        }
                        else if (M == 3)
                        {
                            if (packed.is_nibble_lut)
                                gemm_3row_native_2z_chunk(
                                    packed, row0_q8, row1_q8, row2_q8, dst0,
                                    dst1, dst2, chunk, kb_start, kb_end,
                                    decode_lut_512, /*accumulate=*/false);
                            else
                                gemm_3row_int8_2z_chunk(
                                    packed, row0_q8, row1_q8, row2_q8, dst0,
                                    dst1, dst2, chunk, kb_start, kb_end,
                                    /*accumulate=*/false);
                        }
                        else
                        {
                            if (packed.is_nibble_lut)
                                gemm_4row_native_2z_chunk(
                                    packed, row0_q8, row1_q8, row2_q8,
                                    row3_q8, dst0, dst1, dst2, dst3, chunk,
                                    kb_start, kb_end, decode_lut_512,
                                    /*accumulate=*/false);
                            else
                                gemm_4row_int8_2z_chunk(
                                    packed, row0_q8, row1_q8, row2_q8,
                                    row3_q8, dst0, dst1, dst2, dst3, chunk,
                                    kb_start, kb_end, /*accumulate=*/false);
                        }
                    }
                }
                else
#endif
                {
                const int total_tile_tasks = M * N_chunks * k_tiles;
#pragma omp for schedule(static)
                for (int task = 0; task < total_tile_tasks; ++task)
                {
                    const int kt = task % k_tiles;
                    const int chunk = (task / k_tiles) % N_chunks;
                    const int row = task / (N_chunks * k_tiles);
                    const Q8_1Block *row_q8 =
                        A_q8_all + static_cast<size_t>(row) * K_blocks;
                    const int kb_start = kt * k_blocks_per_tile;
                    const int kb_end =
                        std::min(kb_start + k_blocks_per_tile, K_blocks);
                    float *dest =
                        partial_sums +
                        (((static_cast<size_t>(row) * N_chunks + chunk) *
                          k_tiles) +
                         kt) *
                            64;

                    if (use_avx512)
                    {
#if defined(__AVX512F__)
                        if (packed.is_nibble_lut)
                            gemv_native_vnni_avx512_chunk_native(
                                packed, row_q8, dest, chunk, kb_start, kb_end,
                                decode_lut_512);
                        else
                            gemv_native_vnni_avx512_chunk_int8(
                                packed, row_q8, dest, chunk, kb_start, kb_end);
#endif
                    }
                    else
                    {
                        if (packed.is_nibble_lut)
                            gemv_avx2_chunk_native(
                                packed, row_q8, dest, chunk, kb_start, kb_end,
                                decode_lut_256);
                        else
                            gemv_avx2_chunk_int8(
                                packed, row_q8, dest, chunk, kb_start, kb_end);
                    }
                }
                }

#pragma omp for schedule(static)
                for (int task = 0; task < M * N_chunks; ++task)
                {
                    const int chunk = task % N_chunks;
                    const int row = task / N_chunks;
                    const int n_start = chunk * 64;
                    const int n_cols = std::min(64, N - n_start);
                    const float *base =
                        partial_sums +
                        (static_cast<size_t>(row) * N_chunks + chunk) *
                            k_tiles * 64;
                    float *dst = C + static_cast<size_t>(row) * ldc + n_start;
#if defined(__AVX512F__)
                    __m512 sum0 = _mm512_loadu_ps(base);
                    __m512 sum1 = _mm512_loadu_ps(base + 16);
                    __m512 sum2 = _mm512_loadu_ps(base + 32);
                    __m512 sum3 = _mm512_loadu_ps(base + 48);
                    for (int kt = 1; kt < k_tiles; ++kt)
                    {
                        const float *src = base + static_cast<size_t>(kt) * 64;
                        sum0 = _mm512_add_ps(sum0, _mm512_loadu_ps(src));
                        sum1 = _mm512_add_ps(sum1, _mm512_loadu_ps(src + 16));
                        sum2 = _mm512_add_ps(sum2, _mm512_loadu_ps(src + 32));
                        sum3 = _mm512_add_ps(sum3, _mm512_loadu_ps(src + 48));
                    }
                    if (n_cols < 64)
                    {
                        alignas(64) float tmp[64];
                        _mm512_store_ps(tmp, sum0);
                        _mm512_store_ps(tmp + 16, sum1);
                        _mm512_store_ps(tmp + 32, sum2);
                        _mm512_store_ps(tmp + 48, sum3);
                        std::memcpy(dst, tmp, static_cast<size_t>(n_cols) * sizeof(float));
                    }
                    else
                    {
                        _mm512_storeu_ps(dst, sum0);
                        _mm512_storeu_ps(dst + 16, sum1);
                        _mm512_storeu_ps(dst + 32, sum2);
                        _mm512_storeu_ps(dst + 48, sum3);
                    }
#else
                    __m256 s0 = _mm256_loadu_ps(base);
                    __m256 s1 = _mm256_loadu_ps(base + 8);
                    __m256 s2 = _mm256_loadu_ps(base + 16);
                    __m256 s3 = _mm256_loadu_ps(base + 24);
                    __m256 s4 = _mm256_loadu_ps(base + 32);
                    __m256 s5 = _mm256_loadu_ps(base + 40);
                    __m256 s6 = _mm256_loadu_ps(base + 48);
                    __m256 s7 = _mm256_loadu_ps(base + 56);
                    for (int kt = 1; kt < k_tiles; ++kt)
                    {
                        const float *src = base + static_cast<size_t>(kt) * 64;
                        s0 = _mm256_add_ps(s0, _mm256_loadu_ps(src));
                        s1 = _mm256_add_ps(s1, _mm256_loadu_ps(src + 8));
                        s2 = _mm256_add_ps(s2, _mm256_loadu_ps(src + 16));
                        s3 = _mm256_add_ps(s3, _mm256_loadu_ps(src + 24));
                        s4 = _mm256_add_ps(s4, _mm256_loadu_ps(src + 32));
                        s5 = _mm256_add_ps(s5, _mm256_loadu_ps(src + 40));
                        s6 = _mm256_add_ps(s6, _mm256_loadu_ps(src + 48));
                        s7 = _mm256_add_ps(s7, _mm256_loadu_ps(src + 56));
                    }
                    if (n_cols < 64)
                    {
                        alignas(64) float tmp[64];
                        _mm256_store_ps(tmp, s0);
                        _mm256_store_ps(tmp + 8, s1);
                        _mm256_store_ps(tmp + 16, s2);
                        _mm256_store_ps(tmp + 24, s3);
                        _mm256_store_ps(tmp + 32, s4);
                        _mm256_store_ps(tmp + 40, s5);
                        _mm256_store_ps(tmp + 48, s6);
                        _mm256_store_ps(tmp + 56, s7);
                        std::memcpy(dst, tmp, static_cast<size_t>(n_cols) * sizeof(float));
                    }
                    else
                    {
                        _mm256_storeu_ps(dst, s0);
                        _mm256_storeu_ps(dst + 8, s1);
                        _mm256_storeu_ps(dst + 16, s2);
                        _mm256_storeu_ps(dst + 24, s3);
                        _mm256_storeu_ps(dst + 32, s4);
                        _mm256_storeu_ps(dst + 40, s5);
                        _mm256_storeu_ps(dst + 48, s6);
                        _mm256_storeu_ps(dst + 56, s7);
                    }
#endif
                }
            };
            OMP_WORKSHARE_REGION(do_kpar_partial_rows);
            return;
        }

        const int n_block_chunks = cfg.n_block_chunks;
        const int total_blocks = (N_chunks + n_block_chunks - 1) / n_block_chunks;

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        if (use_avx512 && M == 3 && use_wide_rows)
        {
            auto do_three_rows = [&]()
            {
#pragma omp for schedule(static)
                for (int block_idx = 0; block_idx < total_blocks; ++block_idx)
                {
                    const int chunk_start = block_idx * n_block_chunks;
                    const int chunk_count = std::min(n_block_chunks, N_chunks - chunk_start);
                    const Q8_1Block *row0_q8 = A_q8_all;
                    const Q8_1Block *row1_q8 = A_q8_all + K_blocks;
                    const Q8_1Block *row2_q8 = A_q8_all + static_cast<size_t>(2) * K_blocks;
                    float *row0_out = C;
                    float *row1_out = C + static_cast<size_t>(ldc);
                    float *row2_out = C + static_cast<size_t>(2) * ldc;

                    for (int ci = 0; ci < chunk_count; ++ci)
                    {
                        const int chunk = chunk_start + ci;
                        const int n_start = chunk * 64;
                        const int n_cols = std::min(64, N - n_start);
                        float *c0 = row0_out + n_start;
                        float *c1 = row1_out + n_start;
                        float *c2 = row2_out + n_start;
                        float *dst0 = c0;
                        float *dst1 = c1;
                        float *dst2 = c2;
                        alignas(64) float tmp0[64];
                        alignas(64) float tmp1[64];
                        alignas(64) float tmp2[64];
                        if (n_cols < 64)
                        {
                            dst0 = tmp0;
                            dst1 = tmp1;
                            dst2 = tmp2;
                        }

                        if (packed.is_nibble_lut)
                        {
                            gemm_3row_native_2z_chunk(
                                packed,
                                row0_q8,
                                row1_q8,
                                row2_q8,
                                dst0,
                                dst1,
                                dst2,
                                chunk,
                                0,
                                K_blocks,
                                decode_lut_512,
                                /*accumulate=*/false);
                        }
                        else
                        {
                            gemm_3row_int8_2z_chunk(
                                packed,
                                row0_q8,
                                row1_q8,
                                row2_q8,
                                dst0,
                                dst1,
                                dst2,
                                chunk,
                                0,
                                K_blocks,
                                /*accumulate=*/false);
                        }

                        if (n_cols < 64)
                        {
                            std::memcpy(c0, tmp0, static_cast<size_t>(n_cols) * sizeof(float));
                            std::memcpy(c1, tmp1, static_cast<size_t>(n_cols) * sizeof(float));
                            std::memcpy(c2, tmp2, static_cast<size_t>(n_cols) * sizeof(float));
                        }
                    }
                }
            };
            OMP_WORKSHARE_REGION(do_three_rows);
            return;
        }

        if (use_avx512 && M == 4 && use_wide_rows)
        {
            auto do_four_rows = [&]()
            {
#pragma omp for schedule(static)
                for (int block_idx = 0; block_idx < total_blocks; ++block_idx)
                {
                    const int chunk_start = block_idx * n_block_chunks;
                    const int chunk_count = std::min(n_block_chunks, N_chunks - chunk_start);
                    const Q8_1Block *row0_q8 = A_q8_all;
                    const Q8_1Block *row1_q8 = A_q8_all + K_blocks;
                    const Q8_1Block *row2_q8 = A_q8_all + static_cast<size_t>(2) * K_blocks;
                    const Q8_1Block *row3_q8 = A_q8_all + static_cast<size_t>(3) * K_blocks;
                    float *row0_out = C;
                    float *row1_out = C + static_cast<size_t>(ldc);
                    float *row2_out = C + static_cast<size_t>(2) * ldc;
                    float *row3_out = C + static_cast<size_t>(3) * ldc;

                    for (int ci = 0; ci < chunk_count; ++ci)
                    {
                        const int chunk = chunk_start + ci;
                        const int n_start = chunk * 64;
                        const int n_cols = std::min(64, N - n_start);
                        float *c0 = row0_out + n_start;
                        float *c1 = row1_out + n_start;
                        float *c2 = row2_out + n_start;
                        float *c3 = row3_out + n_start;
                        float *dst0 = c0;
                        float *dst1 = c1;
                        float *dst2 = c2;
                        float *dst3 = c3;
                        alignas(64) float tmp0[64];
                        alignas(64) float tmp1[64];
                        alignas(64) float tmp2[64];
                        alignas(64) float tmp3[64];
                        if (n_cols < 64)
                        {
                            dst0 = tmp0;
                            dst1 = tmp1;
                            dst2 = tmp2;
                            dst3 = tmp3;
                        }

                        const bool use_four_row_native = packed.is_nibble_lut;
                        if (use_four_row_native)
                        {
                            gemm_4row_native_2z_chunk(
                                packed,
                                row0_q8,
                                row1_q8,
                                row2_q8,
                                row3_q8,
                                dst0,
                                dst1,
                                dst2,
                                dst3,
                                chunk,
                                0,
                                K_blocks,
                                decode_lut_512,
                                /*accumulate=*/false);
                        }
                        else if (!packed.is_nibble_lut)
                        {
                            gemm_4row_int8_2z_chunk(
                                packed,
                                row0_q8,
                                row1_q8,
                                row2_q8,
                                row3_q8,
                                dst0,
                                dst1,
                                dst2,
                                dst3,
                                chunk,
                                0,
                                K_blocks,
                                /*accumulate=*/false);
                        }
                        else
                        {
                            gemm_2row_native_chunk(
                                packed,
                                row0_q8,
                                row1_q8,
                                dst0,
                                dst1,
                                chunk,
                                0,
                                K_blocks,
                                decode_lut_512,
                                /*accumulate=*/false);
                            gemm_2row_native_chunk(
                                packed,
                                row2_q8,
                                row3_q8,
                                dst2,
                                dst3,
                                chunk,
                                0,
                                K_blocks,
                                decode_lut_512,
                                /*accumulate=*/false);
                        }

                        if (n_cols < 64)
                        {
                            std::memcpy(c0, tmp0, static_cast<size_t>(n_cols) * sizeof(float));
                            std::memcpy(c1, tmp1, static_cast<size_t>(n_cols) * sizeof(float));
                            std::memcpy(c2, tmp2, static_cast<size_t>(n_cols) * sizeof(float));
                            std::memcpy(c3, tmp3, static_cast<size_t>(n_cols) * sizeof(float));
                        }
                    }
                }
            };
            OMP_WORKSHARE_REGION(do_four_rows);
            return;
        }
#endif

        /*
         * Pair verifier rows for AVX2 and AVX512.
         *
         * Each pair task still visits K blocks in the exact serial decode
         * order for each row.  The only difference from row-by-row GEMV is
         * that the 2-row chunk microkernel loads and decodes the packed B
         * chunk once, then applies it to two independent row accumulators.
         * Tail chunks use temporary 64-float rows so the microkernel never
         * writes past a compact verifier output buffer.
         */
        const int row_pairs = (M + 1) / 2;
        const int total_tasks = row_pairs * total_blocks;

        auto process_pair_task = [&](int task)
        {
            /*
             * Keep block as the outer scheduling dimension so odd-M cases
             * such as M=3 interleave pair work and tail-row work across the
             * OpenMP team.  A pair task is heavier than a one-row tail task;
             * grouping all pairs first leaves half the team waiting at the
             * barrier on realistic verifier shapes.
             */
            const int block_idx = task / row_pairs;
            const int pair = task % row_pairs;
            const int row0 = pair * 2;
            const int row1 = row0 + 1;
            const int chunk_start = block_idx * n_block_chunks;
            const int chunk_count = std::min(n_block_chunks, N_chunks - chunk_start);

            if (row1 < M)
            {
                const Q8_1Block *row0_q8 =
                    A_q8_all + static_cast<size_t>(row0) * K_blocks;
                const Q8_1Block *row1_q8 =
                    A_q8_all + static_cast<size_t>(row1) * K_blocks;
                float *row0_out = C + static_cast<size_t>(row0) * ldc;
                float *row1_out = C + static_cast<size_t>(row1) * ldc;

                for (int ci = 0; ci < chunk_count; ++ci)
                {
                    const int chunk = chunk_start + ci;
                    const int n_start = chunk * 64;
                    const int n_cols = std::min(64, N - n_start);
                    float *c0 = row0_out + n_start;
                    float *c1 = row1_out + n_start;
                    float *dst0 = c0;
                    float *dst1 = c1;
                    alignas(64) float tmp0[64];
                    alignas(64) float tmp1[64];
                    if (n_cols < 64)
                    {
                        dst0 = tmp0;
                        dst1 = tmp1;
                    }

                    if (use_avx512)
                    {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                        if (packed.is_nibble_lut)
                            gemm_2row_native_chunk(
                                packed, row0_q8, row1_q8, dst0, dst1,
                                chunk, 0, K_blocks, decode_lut_512,
                                /*accumulate=*/false);
                        else
                            gemm_2row_int8_chunk(
                                packed, row0_q8, row1_q8, dst0, dst1,
                                chunk, 0, K_blocks,
                                /*accumulate=*/false);
#endif
                    }
                    else
                    {
                        if (packed.is_nibble_lut)
                            gemm_2row_native_chunk_avx2(
                                packed, row0_q8, row1_q8, dst0, dst1,
                                chunk, 0, K_blocks, decode_lut_256,
                                /*accumulate=*/false);
                        else
                            gemm_2row_int8_chunk_avx2(
                                packed, row0_q8, row1_q8, dst0, dst1,
                                chunk, 0, K_blocks,
                                /*accumulate=*/false);
                    }

                    if (n_cols < 64)
                    {
                        std::memcpy(c0, tmp0, static_cast<size_t>(n_cols) * sizeof(float));
                        std::memcpy(c1, tmp1, static_cast<size_t>(n_cols) * sizeof(float));
                    }
                }
                return;
            }

            const Q8_1Block *row_q8 =
                A_q8_all + static_cast<size_t>(row0) * K_blocks;
            float *row_out = C + static_cast<size_t>(row0) * ldc;
            if (use_avx512)
            {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                gemv_native_vnni_avx512_block(
                    packed, row_q8, row_out,
                    chunk_start, chunk_count, K_blocks, N,
                    decode_lut_512);
#endif
            }
            else
            {
                gemv_avx2_block(
                    packed, row_q8, row_out,
                    chunk_start, chunk_count, K_blocks, N,
                    decode_lut_256);
            }
        };

        /*
         * MoE expert verifier projections are small enough that OpenMP
         * fork/join and barrier cost can dominate the M=2 compute.  Serial
         * decode already has a direct path for these shapes; keep grouped M=2
         * on the same footing by running the exact same 2-row chunk kernel
         * directly when the scheduler would create only a small number of
         * blocks.  This is still the grouped/economical kernel, not a hidden
         * fallback to serial row-by-row GEMV.
         */
        const bool direct_small_m2 =
            M == 2 && !omp_in_parallel() &&
            total_tasks <= std::max(1, num_threads * 2) &&
            K_blocks <= 64;
        if (direct_small_m2)
        {
            for (int task = 0; task < total_tasks; ++task)
                process_pair_task(task);
            return;
        }

        auto do_rows = [&]()
        {
#pragma omp for schedule(static)
            for (int task = 0; task < total_tasks; ++task)
                process_pair_task(task);
        };
        OMP_WORKSHARE_REGION(do_rows);
    }

    /**
     * @brief Descriptor for one projection in a grouped verifier-row bundle.
     *
     * All descriptors share the same pre-quantized activation rows.  Each
     * projection keeps its own packed weight matrix, output stride, and optional
     * bias.  The helper below schedules M=2..4 verifier rows for several
     * projections in one OpenMP region while preserving the exact M=1 GEMV
     * chunk kernels used by serial decode.
     */
    struct FusedVerifierRowsDesc
    {
        const CPUNativeVNNIPackedWeights *packed = nullptr;
        float *output = nullptr;
        const float *bias = nullptr;
        int N = 0;
        int ldc = 0;
    };

    /**
     * @brief Fused multi-projection M=2..4 verifier rows.
     *
     * This is the projection-bundle analogue of
     * gemm_native_vnni_preq_decode_equivalent_rows().  It does not use the
     * regular M>1 GEMM microkernels; every output row is still computed by the
     * same per-row GEMV block function as serial decode.  The economy win comes
     * from grouping projections and rows under one OpenMP team and using
     * nowait scheduling between projections.
     *
     * K-parallel GEMV shapes are intentionally not handled here yet because the
     * partial-sum reduction order must remain identical to serial decode.  The
     * caller should use gemm_native_vnni_preq_decode_equivalent_rows() for those
     * single-projection or K-parallel cases.
     */
    inline bool gemm_native_vnni_fused_verifier_rows_preq(
        const Q8_1Block *A_q8_all,
        const FusedVerifierRowsDesc *descs,
        int num_descs,
        int M,
        int K_blocks,
        ISAPath isa_path = ISAPath::AUTO)
    {
        if (!A_q8_all || !descs || num_descs <= 0 || M <= 1 || M > 4 || K_blocks <= 0)
            return false;

        const int num_threads = omp_get_max_threads();
        for (int p = 0; p < num_descs; ++p)
        {
            const auto &d = descs[p];
            if (!d.packed || !d.output || d.N <= 0 || d.ldc < d.N ||
                d.packed->blocks_per_row != K_blocks)
            {
                return false;
            }

        }

        const ISALevel active_isa = activeISALevel();
        bool use_avx512 = false;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        use_avx512 = (isa_path == ISAPath::AUTO) ? (active_isa >= ISALevel::AVX512)
                                                 : (isa_path == ISAPath::AVX512);
#endif
        const bool use_avx2 =
            !use_avx512 &&
            ((isa_path == ISAPath::AUTO && active_isa >= ISALevel::AVX2) ||
             isa_path == ISAPath::AVX2);

        struct FusedVerifierRowsPlan
        {
            int n_chunks = 0;
            int n_block_chunks = 1;
            int total_blocks = 0;
            int k_tiles = 0;
            int k_blocks_per_tile = 0;
            size_t partial_sums_offset = 0;
            size_t partial_sums_size = 0;
        };

        std::vector<FusedVerifierRowsPlan> plans(static_cast<size_t>(num_descs));
        size_t fused_partial_sum_floats = 0;
        for (int p = 0; p < num_descs; ++p)
        {
            const auto &d = descs[p];
            const NativeVNNITileConfig cfg =
                computeTileConfig(d.packed->N, d.packed->K, 1,
                                  d.packed->payload_bytes, num_threads);
            auto &plan = plans[static_cast<size_t>(p)];
            plan.n_chunks = (d.packed->N + 63) / 64;
            plan.n_block_chunks = std::max(1, cfg.n_block_chunks);
            plan.total_blocks =
                (plan.n_chunks + plan.n_block_chunks - 1) /
                plan.n_block_chunks;
            plan.k_tiles = cfg.k_tiles;
            plan.k_blocks_per_tile =
                (plan.k_tiles > 1)
                    ? (K_blocks + plan.k_tiles - 1) / plan.k_tiles
                    : K_blocks;
            if (plan.k_tiles > 1)
            {
                /*
                 * CPU-only verifier workspace.  GPU stages must use the graph
                 * workspace binding contract instead; this helper never runs on
                 * device backends.  The layout mirrors the single-projection
                 * exact verifier path: [row][N chunk][K tile][64 columns].
                 */
                plan.partial_sums_offset = fused_partial_sum_floats;
                plan.partial_sums_size =
                    static_cast<size_t>(M) *
                    static_cast<size_t>(plan.n_chunks) *
                    static_cast<size_t>(plan.k_tiles) * 64;
                fused_partial_sum_floats += plan.partial_sums_size;
            }
        }
        /*
         * Every K-tile partial is overwritten before reduction.  A single
         * thread-local arena avoids repeated per-projection allocations while
         * keeping the layout explicit and easy to audit.
         */
        thread_local AlignedVector<float> fused_partial_sums_tls;
        if (fused_partial_sums_tls.size() < fused_partial_sum_floats)
            fused_partial_sums_tls.resize_uninitialized(fused_partial_sum_floats);

        auto do_fused_rows = [&]()
        {
            for (int p = 0; p < num_descs; ++p)
            {
                const auto &d = descs[p];
                const auto &packed = *d.packed;
                const int N = packed.N;
                auto &plan = plans[static_cast<size_t>(p)];
                const int N_chunks = plan.n_chunks;
                const int n_block_chunks = plan.n_block_chunks;
                const int total_blocks = plan.total_blocks;

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                const __m512i decode_lut_512 =
                    (use_avx512 && packed.is_nibble_lut)
                        ? build_decode_lut(packed.codebook_id)
                        : _mm512_setzero_si512();
#endif
                const __m256i decode_lut_256 =
                    (use_avx2 && packed.is_nibble_lut)
                        ? build_decode_lut_avx2_for_codebook(packed.codebook_id)
                        : _mm256_setzero_si256();

                if (plan.k_tiles > 1)
                {
                    /*
                     * Long-K fused verifier projections need K-parallel work
                     * for economy, but they must reduce partials in the exact
                     * same order as the M=1 decode GEMV.  This mirrors
                     * gemm_native_vnni_preq_decode_equivalent_rows(), only
                     * keeping all projections under this one OpenMP team.
                     */
                    const int k_tiles = plan.k_tiles;
                    const int k_blocks_per_tile = plan.k_blocks_per_tile;
                    float *partial_sums =
                        fused_partial_sums_tls.data() + plan.partial_sums_offset;

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                    if (use_avx512 && M >= 2 && M <= 4)
                    {
                        const int total_shared_tile_tasks = N_chunks * k_tiles;
#pragma omp for schedule(static)
                        for (int task = 0; task < total_shared_tile_tasks; ++task)
                        {
                            const int kt = task % k_tiles;
                            const int chunk = task / k_tiles;
                            const int kb_start = kt * k_blocks_per_tile;
                            const int kb_end =
                                std::min(kb_start + k_blocks_per_tile, K_blocks);
                            float *dst0 =
                                partial_sums +
                                ((static_cast<size_t>(chunk) * k_tiles) + kt) * 64;
                            float *dst1 =
                                partial_sums +
                                (((static_cast<size_t>(N_chunks) + chunk) *
                                  k_tiles) +
                                 kt) *
                                    64;
                            float *dst2 = nullptr;
                            float *dst3 = nullptr;
                            if (M >= 3)
                            {
                                dst2 =
                                    partial_sums +
                                    (((static_cast<size_t>(2) * N_chunks + chunk) *
                                      k_tiles) +
                                     kt) *
                                        64;
                            }
                            if (M >= 4)
                            {
                                dst3 =
                                    partial_sums +
                                    (((static_cast<size_t>(3) * N_chunks + chunk) *
                                      k_tiles) +
                                     kt) *
                                        64;
                            }

                            const Q8_1Block *row0_q8 = A_q8_all;
                            const Q8_1Block *row1_q8 = A_q8_all + K_blocks;
                            const Q8_1Block *row2_q8 =
                                A_q8_all + static_cast<size_t>(2) * K_blocks;
                            const Q8_1Block *row3_q8 =
                                A_q8_all + static_cast<size_t>(3) * K_blocks;

                            if (M == 2)
                            {
                                if (packed.is_nibble_lut)
                                    gemm_2row_native_chunk(
                                        packed, row0_q8, row1_q8, dst0, dst1,
                                        chunk, kb_start, kb_end,
                                        decode_lut_512, /*accumulate=*/false);
                                else
                                    gemm_2row_int8_chunk(
                                        packed, row0_q8, row1_q8, dst0, dst1,
                                        chunk, kb_start, kb_end,
                                        /*accumulate=*/false);
                            }
                            else if (M == 3)
                            {
                                if (packed.is_nibble_lut)
                                    gemm_3row_native_2z_chunk(
                                        packed, row0_q8, row1_q8, row2_q8,
                                        dst0, dst1, dst2, chunk, kb_start,
                                        kb_end, decode_lut_512,
                                        /*accumulate=*/false);
                                else
                                    gemm_3row_int8_2z_chunk(
                                        packed, row0_q8, row1_q8, row2_q8,
                                        dst0, dst1, dst2, chunk, kb_start,
                                        kb_end, /*accumulate=*/false);
                            }
                            else
                            {
                                if (packed.is_nibble_lut)
                                    gemm_4row_native_2z_chunk(
                                        packed, row0_q8, row1_q8, row2_q8,
                                        row3_q8, dst0, dst1, dst2, dst3,
                                        chunk, kb_start, kb_end,
                                        decode_lut_512, /*accumulate=*/false);
                                else
                                    gemm_4row_int8_2z_chunk(
                                        packed, row0_q8, row1_q8, row2_q8,
                                        row3_q8, dst0, dst1, dst2, dst3,
                                        chunk, kb_start, kb_end,
                                        /*accumulate=*/false);
                            }
                        }
                    }
                    else
#endif
                    {
                        const int total_tile_tasks = M * N_chunks * k_tiles;
#pragma omp for schedule(static)
                        for (int task = 0; task < total_tile_tasks; ++task)
                        {
                            const int kt = task % k_tiles;
                            const int chunk = (task / k_tiles) % N_chunks;
                            const int row = task / (N_chunks * k_tiles);
                            const Q8_1Block *row_q8 =
                                A_q8_all + static_cast<size_t>(row) * K_blocks;
                            const int kb_start = kt * k_blocks_per_tile;
                            const int kb_end =
                                std::min(kb_start + k_blocks_per_tile, K_blocks);
                            float *dest =
                                partial_sums +
                                (((static_cast<size_t>(row) * N_chunks + chunk) *
                                  k_tiles) +
                                 kt) *
                                    64;

                            if (use_avx512)
                            {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                                if (packed.is_nibble_lut)
                                    gemv_native_vnni_avx512_chunk_native(
                                        packed, row_q8, dest, chunk, kb_start,
                                        kb_end, decode_lut_512);
                                else
                                    gemv_native_vnni_avx512_chunk_int8(
                                        packed, row_q8, dest, chunk, kb_start,
                                        kb_end);
#endif
                            }
                            else
                            {
                                if (packed.is_nibble_lut)
                                    gemv_avx2_chunk_native(
                                        packed, row_q8, dest, chunk, kb_start,
                                        kb_end, decode_lut_256);
                                else
                                    gemv_avx2_chunk_int8(
                                        packed, row_q8, dest, chunk, kb_start,
                                        kb_end);
                            }
                        }
                    }

#pragma omp for schedule(static)
                    for (int task = 0; task < M * N_chunks; ++task)
                    {
                        const int chunk = task % N_chunks;
                        const int row = task / N_chunks;
                        const int n_start = chunk * 64;
                        const int n_cols = std::min(64, N - n_start);
                        const float *base =
                            partial_sums +
                            (static_cast<size_t>(row) * N_chunks + chunk) *
                                k_tiles * 64;
                        float *dst =
                            d.output + static_cast<size_t>(row) * d.ldc + n_start;
#if defined(__AVX512F__)
                        __m512 sum0 = _mm512_loadu_ps(base);
                        __m512 sum1 = _mm512_loadu_ps(base + 16);
                        __m512 sum2 = _mm512_loadu_ps(base + 32);
                        __m512 sum3 = _mm512_loadu_ps(base + 48);
                        for (int kt = 1; kt < k_tiles; ++kt)
                        {
                            const float *src = base + static_cast<size_t>(kt) * 64;
                            sum0 = _mm512_add_ps(sum0, _mm512_loadu_ps(src));
                            sum1 = _mm512_add_ps(sum1, _mm512_loadu_ps(src + 16));
                            sum2 = _mm512_add_ps(sum2, _mm512_loadu_ps(src + 32));
                            sum3 = _mm512_add_ps(sum3, _mm512_loadu_ps(src + 48));
                        }
                        if (n_cols < 64)
                        {
                            alignas(64) float tmp[64];
                            _mm512_store_ps(tmp, sum0);
                            _mm512_store_ps(tmp + 16, sum1);
                            _mm512_store_ps(tmp + 32, sum2);
                            _mm512_store_ps(tmp + 48, sum3);
                            std::memcpy(
                                dst, tmp,
                                static_cast<size_t>(n_cols) * sizeof(float));
                        }
                        else
                        {
                            _mm512_storeu_ps(dst, sum0);
                            _mm512_storeu_ps(dst + 16, sum1);
                            _mm512_storeu_ps(dst + 32, sum2);
                            _mm512_storeu_ps(dst + 48, sum3);
                        }
#else
                        __m256 s0 = _mm256_loadu_ps(base);
                        __m256 s1 = _mm256_loadu_ps(base + 8);
                        __m256 s2 = _mm256_loadu_ps(base + 16);
                        __m256 s3 = _mm256_loadu_ps(base + 24);
                        __m256 s4 = _mm256_loadu_ps(base + 32);
                        __m256 s5 = _mm256_loadu_ps(base + 40);
                        __m256 s6 = _mm256_loadu_ps(base + 48);
                        __m256 s7 = _mm256_loadu_ps(base + 56);
                        for (int kt = 1; kt < k_tiles; ++kt)
                        {
                            const float *src = base + static_cast<size_t>(kt) * 64;
                            s0 = _mm256_add_ps(s0, _mm256_loadu_ps(src));
                            s1 = _mm256_add_ps(s1, _mm256_loadu_ps(src + 8));
                            s2 = _mm256_add_ps(s2, _mm256_loadu_ps(src + 16));
                            s3 = _mm256_add_ps(s3, _mm256_loadu_ps(src + 24));
                            s4 = _mm256_add_ps(s4, _mm256_loadu_ps(src + 32));
                            s5 = _mm256_add_ps(s5, _mm256_loadu_ps(src + 40));
                            s6 = _mm256_add_ps(s6, _mm256_loadu_ps(src + 48));
                            s7 = _mm256_add_ps(s7, _mm256_loadu_ps(src + 56));
                        }
                        if (n_cols < 64)
                        {
                            alignas(64) float tmp[64];
                            _mm256_store_ps(tmp, s0);
                            _mm256_store_ps(tmp + 8, s1);
                            _mm256_store_ps(tmp + 16, s2);
                            _mm256_store_ps(tmp + 24, s3);
                            _mm256_store_ps(tmp + 32, s4);
                            _mm256_store_ps(tmp + 40, s5);
                            _mm256_store_ps(tmp + 48, s6);
                            _mm256_store_ps(tmp + 56, s7);
                            std::memcpy(
                                dst, tmp,
                                static_cast<size_t>(n_cols) * sizeof(float));
                        }
                        else
                        {
                            _mm256_storeu_ps(dst, s0);
                            _mm256_storeu_ps(dst + 8, s1);
                            _mm256_storeu_ps(dst + 16, s2);
                            _mm256_storeu_ps(dst + 24, s3);
                            _mm256_storeu_ps(dst + 32, s4);
                            _mm256_storeu_ps(dst + 40, s5);
                            _mm256_storeu_ps(dst + 48, s6);
                            _mm256_storeu_ps(dst + 56, s7);
                        }
#endif
                    }

                    if (d.bias)
                    {
#pragma omp for schedule(static) nowait
                        for (int row = 0; row < M; ++row)
                        {
                            float *row_out =
                                d.output + static_cast<size_t>(row) * d.ldc;
                            int j = 0;
#if defined(__AVX512F__)
                            for (; j + 15 < d.N; j += 16)
                            {
                                __m512 v = _mm512_add_ps(
                                    _mm512_loadu_ps(row_out + j),
                                    _mm512_loadu_ps(d.bias + j));
                                _mm512_storeu_ps(row_out + j, v);
                            }
#endif
                            for (; j < d.N; ++j)
                                row_out[j] += d.bias[j];
                        }
                    }
                    continue;
                }

                if (!use_avx512 && !use_avx2)
                {
#pragma omp for schedule(static) nowait
                    for (int row = 0; row < M; ++row)
                    {
                        gemv_native_vnni_scalar(
                            packed,
                            A_q8_all + static_cast<size_t>(row) * K_blocks,
                            d.output + static_cast<size_t>(row) * d.ldc,
                            N,
                            K_blocks);
                    }
                }
                else
                {
                    const int row_pairs = (M + 1) / 2;
                    const int total_tasks = row_pairs * total_blocks;
#pragma omp for schedule(static) nowait
                    for (int task = 0; task < total_tasks; ++task)
                    {
                    /*
                     * Interleave pair/tail-row tasks per N-block.  This keeps
                     * fused projection bundles economical for M=3, where the
                     * final verifier row is necessarily a single-row task unless
                     * a wider 3-row microkernel is selected.
                     */
                    const int block_idx = task / row_pairs;
                    const int pair = task % row_pairs;
                    const int row0 = pair * 2;
                    const int row1 = row0 + 1;
                    const int chunk_start = block_idx * n_block_chunks;
                    const int chunk_count = std::min(n_block_chunks, N_chunks - chunk_start);

                    if (row1 < M)
                    {
                        const Q8_1Block *row0_q8 =
                            A_q8_all + static_cast<size_t>(row0) * K_blocks;
                        const Q8_1Block *row1_q8 =
                            A_q8_all + static_cast<size_t>(row1) * K_blocks;
                        float *row0_out = d.output + static_cast<size_t>(row0) * d.ldc;
                        float *row1_out = d.output + static_cast<size_t>(row1) * d.ldc;

                        for (int ci = 0; ci < chunk_count; ++ci)
                        {
                            const int chunk = chunk_start + ci;
                            const int n_start = chunk * 64;
                            const int n_cols = std::min(64, N - n_start);
                            float *c0 = row0_out + n_start;
                            float *c1 = row1_out + n_start;
                            float *dst0 = c0;
                            float *dst1 = c1;
                            alignas(64) float tmp0[64];
                            alignas(64) float tmp1[64];
                            if (n_cols < 64)
                            {
                                dst0 = tmp0;
                                dst1 = tmp1;
                            }

                            if (use_avx512)
                            {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                                if (packed.is_nibble_lut)
                                    gemm_2row_native_chunk(
                                        packed, row0_q8, row1_q8, dst0, dst1,
                                        chunk, 0, K_blocks, decode_lut_512,
                                        /*accumulate=*/false);
                                else
                                    gemm_2row_int8_chunk(
                                        packed, row0_q8, row1_q8, dst0, dst1,
                                        chunk, 0, K_blocks,
                                        /*accumulate=*/false);
#endif
                            }
                            else
                            {
                                if (packed.is_nibble_lut)
                                    gemm_2row_native_chunk_avx2(
                                        packed, row0_q8, row1_q8, dst0, dst1,
                                        chunk, 0, K_blocks, decode_lut_256,
                                        /*accumulate=*/false);
                                else
                                    gemm_2row_int8_chunk_avx2(
                                        packed, row0_q8, row1_q8, dst0, dst1,
                                        chunk, 0, K_blocks,
                                        /*accumulate=*/false);
                            }

                            if (n_cols < 64)
                            {
                                std::memcpy(c0, tmp0, static_cast<size_t>(n_cols) * sizeof(float));
                                std::memcpy(c1, tmp1, static_cast<size_t>(n_cols) * sizeof(float));
                            }
                        }
                        continue;
                    }

                    const Q8_1Block *row_q8 =
                        A_q8_all + static_cast<size_t>(row0) * K_blocks;
                    float *row_out = d.output + static_cast<size_t>(row0) * d.ldc;
                    if (use_avx512)
                    {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                        gemv_native_vnni_avx512_block(
                            packed, row_q8, row_out,
                            chunk_start, chunk_count, K_blocks, N,
                            decode_lut_512);
#endif
                    }
                    else
                    {
                        gemv_avx2_block(
                            packed, row_q8, row_out,
                        chunk_start, chunk_count, K_blocks, N,
                        decode_lut_256);
                    }
                    }
                }

                if (d.bias)
                {
#pragma omp barrier
#pragma omp for schedule(static) nowait
                    for (int row = 0; row < M; ++row)
                    {
                        float *row_out = d.output + static_cast<size_t>(row) * d.ldc;
                        int j = 0;
#if defined(__AVX512F__)
                        for (; j + 15 < d.N; j += 16)
                        {
                            __m512 v = _mm512_add_ps(
                                _mm512_loadu_ps(row_out + j),
                                _mm512_loadu_ps(d.bias + j));
                            _mm512_storeu_ps(row_out + j, v);
                        }
#endif
                        for (; j < d.N; ++j)
                            row_out[j] += d.bias[j];
                    }
                }
            }
#pragma omp barrier
        };

        OMP_WORKSHARE_REGION(do_fused_rows);
        return true;
    }

    // =========================================================================
    // Full GEMM dispatcher (M>1) — quantizes then delegates to preq path
    // =========================================================================

    inline void gemm_native_vnni(
        const CPUNativeVNNIPackedWeights &packed,
        const float *A_fp32,
        float *C,
        int M,
        int ldc)
    {
        const int K = packed.K;
        const int K_blocks = packed.blocks_per_row;

        // Pre-quantize all M rows of A
        std::vector<Q8_1Block> all_A_q8(static_cast<size_t>(M) * K_blocks);
        quantize_activations_to_q8_1(A_fp32, all_A_q8.data(), M, K, K_blocks);

        // Delegate to pre-quantized compute path
        gemm_native_vnni_preq(packed, all_A_q8.data(), C, M, ldc);
    }

    // =========================================================================
    // Bias epilogue: C[m, j] += bias[j] for all M rows
    // =========================================================================

    inline void apply_bias_epilogue(float *C, const float *bias, int M, int N, int ldc)
    {
        for (int m = 0; m < M; ++m)
        {
            float *row = C + m * ldc;
            int j = 0;
#if defined(__AVX512F__)
            for (; j + 15 < N; j += 16)
            {
                __m512 c = _mm512_loadu_ps(row + j);
                __m512 b = _mm512_loadu_ps(bias + j);
                _mm512_storeu_ps(row + j, _mm512_add_ps(c, b));
            }
#endif
            for (; j < N; ++j)
                row[j] += bias[j];
        }
    }

    // =========================================================================
    // Native Q8_0 GEMV — internal implementation
    // =========================================================================
    //
    // These are implementation details called by gemv_native_vnni() when
    // Q8_0 blocks are provided. For Q8_0 weights (8-bit symmetric, 34
    // bytes/block), the packed VNNI format's 3-array layout creates too
    // many memory streams for the hardware prefetcher. Reading native
    // blocks directly is faster: one contiguous stream per row.
    //
    // VNNI path (AVX512_VNNI + AVX512VL): Pre-quantize FP32 activation
    // to Q8_1 once, then use 256-bit vpdpbusd via abs()+sign() trick for
    // signed×signed dot product. ~2× fewer instructions than FP32 FMA.
    //
    // FMA fallback (AVX512F only): F16C hardware scale conversion,
    // 4-block unrolled FMA with FP32 activation.

#if defined(__AVX512F__)

    /**
     * @brief Hardware FP16→FP32 conversion using F16C (vcvtph2ps).
     * Single instruction vs the software path which has branches and bit ops.
     */
    static inline float hw_fp16_to_fp32(uint16_t h)
    {
        return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
    }

    /**
     * @brief VNNI-accelerated Q8_0 row dot product with pre-quantized INT8 activation.
     *
     * Uses 256-bit AVX512_VNNI vpdpbusd via the abs()+sign() trick for true
     * signed×signed dot product without bias correction. ~2× fewer instructions
     * than an FP32 FMA approach: 8 instr/block vs ~15.
     *
     * Q8_0 weight values are in [-127,127] (never -128), so abs() is exact.
     *
     * @param row     Q8_0 weight blocks for one output row (bpr blocks)
     * @param a_q8    Q8_1 activation blocks (pre-quantized from FP32)
     * @param bpr     Blocks per row (K / 32)
     * @return        Scalar dot product with scale weighting
     */
    static inline float gemv_dot_row_q8_0_vnni(
        const Q8_0Block *__restrict row,
        const Q8_1Block *__restrict a_q8,
        int bpr)
    {
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();

        int kb = 0;

        // Main loop: 4-block unroll for ILP across independent FMA chains
        for (; kb + 3 < bpr; kb += 4)
        {
            // Block 0: abs(w)*sign(a,w) → vpdpbusd → scale → FMA accumulate
            {
                const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row[kb].qs));
                const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_q8[kb].qs));
                const __m256i sumi = _mm256_dpbusd_epi32(
                    _mm256_setzero_si256(), _mm256_abs_epi8(w), _mm256_sign_epi8(a, w));
                const float sp = hw_fp16_to_fp32(row[kb].d) * hw_fp16_to_fp32(a_q8[kb].d);
                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(sp), _mm256_cvtepi32_ps(sumi), acc0);
            }
            // Block 1
            {
                const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row[kb + 1].qs));
                const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_q8[kb + 1].qs));
                const __m256i sumi = _mm256_dpbusd_epi32(
                    _mm256_setzero_si256(), _mm256_abs_epi8(w), _mm256_sign_epi8(a, w));
                const float sp = hw_fp16_to_fp32(row[kb + 1].d) * hw_fp16_to_fp32(a_q8[kb + 1].d);
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(sp), _mm256_cvtepi32_ps(sumi), acc1);
            }
            // Block 2
            {
                const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row[kb + 2].qs));
                const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_q8[kb + 2].qs));
                const __m256i sumi = _mm256_dpbusd_epi32(
                    _mm256_setzero_si256(), _mm256_abs_epi8(w), _mm256_sign_epi8(a, w));
                const float sp = hw_fp16_to_fp32(row[kb + 2].d) * hw_fp16_to_fp32(a_q8[kb + 2].d);
                acc2 = _mm256_fmadd_ps(_mm256_set1_ps(sp), _mm256_cvtepi32_ps(sumi), acc2);
            }
            // Block 3
            {
                const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row[kb + 3].qs));
                const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_q8[kb + 3].qs));
                const __m256i sumi = _mm256_dpbusd_epi32(
                    _mm256_setzero_si256(), _mm256_abs_epi8(w), _mm256_sign_epi8(a, w));
                const float sp = hw_fp16_to_fp32(row[kb + 3].d) * hw_fp16_to_fp32(a_q8[kb + 3].d);
                acc3 = _mm256_fmadd_ps(_mm256_set1_ps(sp), _mm256_cvtepi32_ps(sumi), acc3);
            }
        }

        // Tail: remaining 1-3 blocks
        for (; kb < bpr; ++kb)
        {
            const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row[kb].qs));
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_q8[kb].qs));
            const __m256i sumi = _mm256_dpbusd_epi32(
                _mm256_setzero_si256(), _mm256_abs_epi8(w), _mm256_sign_epi8(a, w));
            const float sp = hw_fp16_to_fp32(row[kb].d) * hw_fp16_to_fp32(a_q8[kb].d);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(sp), _mm256_cvtepi32_ps(sumi), acc0);
        }

        // Horizontal reduce: 4 × YMM (8 FP32 each) → 1 scalar
        const __m256 sum01 = _mm256_add_ps(acc0, acc1);
        const __m256 sum23 = _mm256_add_ps(acc2, acc3);
        const __m256 total = _mm256_add_ps(sum01, sum23);
        const __m128 hi = _mm256_extractf128_ps(total, 1);
        const __m128 lo = _mm256_castps256_ps128(total);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        return _mm_cvtss_f32(sum128);
    }

    /**
     * @brief Row-parallel Q8_0 GEMV from native blocks.
     *
     * C[n] = Σ_kb  scale[n][kb] * dot(qs[n][kb], A[kb*32..(kb+1)*32-1])
     *
     * @param blocks  Native Q8_0 blocks [N × bpr], contiguous in memory
     * @param A       FP32 activation vector [K]
     * @param C       FP32 output vector [N]
     * @param N       Number of output rows
     * @param K       Input dimension
     * @param bpr     Blocks per row (= K / 32)
     */
    inline void q8_0_native_gemv(
        const Q8_0Block *__restrict blocks,
        const float *__restrict A,
        float *__restrict C,
        int N,
        int K,
        int bpr)
    {
        (void)K;

        // VNNI path: pre-quantize activation to Q8_1, then use vpdpbusd
        static thread_local std::vector<Q8_1Block> a_q8_tls;
        if (static_cast<int>(a_q8_tls.size()) < bpr)
            a_q8_tls.resize(bpr);
        Q8_1Block *A_q8 = a_q8_tls.data();

        // Quantize FP32 activation → Q8_1 (single-threaded, amortized over all rows)
        {
            int kb = 0;
            for (; kb + 1 < bpr; kb += 2)
                simd::quantize_two_blocks_avx512(A + kb * 32, A_q8[kb], A_q8[kb + 1]);
            for (; kb < bpr; ++kb)
                simd::quantize_single_block(A + kb * 32, A_q8[kb], 32);
        }

        // Serial fast path for small N (same rationale as gemv_native_vnni_preq)
        int num_threads = omp_get_max_threads();
        if (!omp_in_parallel() && N < num_threads)
        {
            for (int n = 0; n < N; ++n)
            {
                const Q8_0Block *__restrict row = blocks + static_cast<size_t>(n) * bpr;
                C[n] = gemv_dot_row_q8_0_vnni(row, A_q8, bpr);
            }
            return;
        }

        auto do_gemv = [&]()
        {
#pragma omp for schedule(static)
            for (int n = 0; n < N; ++n)
            {
                const Q8_0Block *__restrict row = blocks + static_cast<size_t>(n) * bpr;
                C[n] = gemv_dot_row_q8_0_vnni(row, A_q8, bpr);
            }
        };
        OMP_WORKSHARE_REGION(do_gemv);
    }

    // =========================================================================
    // Fused multi-projection GEMV (single OMP region, all formats)
    // =========================================================================

    /**
     * @brief Descriptor for one projection in a fused GEMV call.
     *
     * Supports both Q8_0-raw (row-parallel on native blocks) and interleaved
     * (chunk-parallel on VNNI-packed data) weight layouts.
     */
    struct FusedGemvDesc
    {
        const CPUNativeVNNIPackedWeights *packed; // interleaved packed weights (always set)
        const Q8_0Block *q8_0_raw;                // non-null for Q8_0 zero-copy (row-parallel)
        float *output;                            // output vector [N]
        const float *bias;                        // optional bias [N], nullptr if none
        int N;                                    // number of output rows
        int bpr;                                  // blocks per row (for Q8_0 raw path)
    };

    /**
     * @brief Fused multi-projection GEMV in a single OMP region.
     *
     * Processes all projections without re-entering the OMP parallel region.
     * Uses `nowait` between projections so threads finishing one projection
     * can immediately start the next — critical for work balancing when
     * projection sizes differ (e.g., Q=3584, K=512, V=512 with 56 threads).
     *
     * Supports mixed formats: Q8_0 deferred (row-parallel on raw blocks) and
     * any interleaved format (chunk-parallel on VNNI-packed data).
     *
     * @param A_q8              Pre-quantized Q8_1 activations [K_blocks]
     * @param descs             Array of projection descriptors
     * @param num_descs         Number of projections
     */
    inline void gemv_native_vnni_fused_preq(
        const Q8_1Block *__restrict A_q8,
        const FusedGemvDesc *descs,
        int num_descs)
    {
        auto do_fused = [&]()
        {
            for (int p = 0; p < num_descs; ++p)
            {
                const auto &d = descs[p];

                if (d.q8_0_raw)
                {
                    // Q8_0 raw blocks: row-parallel (one row per thread)
                    const int proj_N = d.N;
                    const int proj_bpr = d.bpr;
                    const Q8_0Block *__restrict blocks = d.q8_0_raw;

#pragma omp for schedule(static) nowait
                    for (int n = 0; n < proj_N; ++n)
                    {
                        const Q8_0Block *__restrict row = blocks + static_cast<size_t>(n) * proj_bpr;
                        float val = gemv_dot_row_q8_0_vnni(row, A_q8, proj_bpr);
                        if (d.bias)
                            val += d.bias[n];
                        d.output[n] = val;
                    }
                }
                else
                {
                    // Interleaved: chunk-parallel (64 columns per chunk)
                    const auto &packed = *d.packed;
                    const int N = packed.N;
                    const int N_chunks = (N + 63) / 64;
                    const int K_blocks = packed.blocks_per_row;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                    const __m512i decode_lut = packed.is_nibble_lut
                                                   ? build_decode_lut(packed.codebook_id)
                                                   : _mm512_setzero_si512();

#pragma omp for schedule(static) nowait
                    for (int chunk = 0; chunk < N_chunks; ++chunk)
                    {
                        gemv_native_vnni_avx512_block(packed, A_q8, d.output,
                                                      chunk, 1, K_blocks, N, decode_lut);
                    }
#else
                    const __m256i decode_lut = packed.is_nibble_lut
                                                   ? build_decode_lut_avx2_for_codebook(packed.codebook_id)
                                                   : _mm256_setzero_si256();
#pragma omp for schedule(static) nowait
                    for (int chunk = 0; chunk < N_chunks; ++chunk)
                    {
                        gemv_avx2_block(packed, A_q8, d.output, chunk, 1, K_blocks, N, decode_lut);
                    }
#endif
                    // Barrier: all GEMV chunks must complete before bias pass
                    // reads d.output. Without this, threads with 0 GEMV chunks
                    // (when N_chunks < num_threads) race into the bias loop
                    // and read output locations still being written by GEMV threads.
                    if (d.bias)
                    {
#pragma omp barrier
#pragma omp for schedule(static) nowait
                        for (int i = 0; i < d.N; ++i)
                            d.output[i] += d.bias[i];
                    }
                }
            }

            // Final barrier: ensure all projections are complete before
            // the caller reads outputs.
#pragma omp barrier
        };
        OMP_WORKSHARE_REGION(do_fused);
    }

#else // !defined(__AVX512F__)

    static inline float gemv_dot_row_q8_0_vnni(
        const Q8_0Block *__restrict row,
        const Q8_1Block *__restrict a_q8,
        int bpr)
    {
        float acc = 0.0f;
        for (int kb = 0; kb < bpr; ++kb)
        {
            int32_t sumi = 0;
            for (int i = 0; i < Q8_0Block::BLOCK_SIZE; ++i)
                sumi += static_cast<int32_t>(row[kb].qs[i]) *
                        static_cast<int32_t>(a_q8[kb].qs[i]);
            const float sp = simd::fp16_to_fp32(row[kb].d) *
                             simd::fp16_to_fp32(a_q8[kb].d);
            acc += sp * static_cast<float>(sumi);
        }
        return acc;
    }

    inline void q8_0_native_gemv(
        const Q8_0Block *__restrict blocks,
        const float *__restrict A,
        float *__restrict C,
        int N,
        int K,
        int bpr)
    {
        static thread_local std::vector<Q8_1Block> a_q8_tls;
        if (static_cast<int>(a_q8_tls.size()) < bpr)
            a_q8_tls.resize(bpr);
        Q8_1Block *A_q8 = a_q8_tls.data();

        for (int kb = 0; kb < bpr; ++kb)
        {
            const int block_start = kb * 32;
            const int block_len = std::min(32, K - block_start);
            simd::quantize_single_block(A + block_start, A_q8[kb], block_len);
        }

        auto do_gemv = [&]()
        {
#pragma omp for schedule(static)
            for (int n = 0; n < N; ++n)
            {
                const Q8_0Block *__restrict row = blocks + static_cast<size_t>(n) * bpr;
                C[n] = gemv_dot_row_q8_0_vnni(row, A_q8, bpr);
            }
        };
        OMP_WORKSHARE_REGION(do_gemv);
    }

    struct FusedGemvDesc
    {
        const CPUNativeVNNIPackedWeights *packed;
        const Q8_0Block *q8_0_raw;
        float *output;
        const float *bias;
        int N;
        int bpr;
    };

    inline void gemv_native_vnni_fused_preq(
        const Q8_1Block *__restrict A_q8,
        const FusedGemvDesc *descs,
        int num_descs)
    {
        for (int p = 0; p < num_descs; ++p)
        {
            const auto &d = descs[p];
            if (d.q8_0_raw)
            {
                auto do_rows = [&]()
                {
#pragma omp for schedule(static)
                    for (int n = 0; n < d.N; ++n)
                    {
                        const Q8_0Block *__restrict row =
                            d.q8_0_raw + static_cast<size_t>(n) * d.bpr;
                        float val = gemv_dot_row_q8_0_vnni(row, A_q8, d.bpr);
                        if (d.bias)
                            val += d.bias[n];
                        d.output[n] = val;
                    }
                };
                OMP_WORKSHARE_REGION(do_rows);
            }
            else
            {
                gemv_native_vnni_preq(*d.packed, A_q8, d.output);
                if (d.bias)
                {
                    for (int n = 0; n < d.N; ++n)
                        d.output[n] += d.bias[n];
                }
            }
        }
    }

#endif // __AVX512F__

    // =========================================================================
    // Fused multi-input GEMV (single OMP region, different Q8_1 input per desc)
    // =========================================================================

    /**
     * @brief Descriptor for one projection with its own pre-quantized input.
     *
     * Unlike FusedGemvDesc (which shares a single Q8_1 input across all
     * projections), each descriptor here carries its own A_q8 pointer.
     * Used for MoE expert down projections where each expert has a different
     * SwiGLU activation as input.
     */
    struct FusedGemvMultiInputDesc
    {
        const Q8_1Block *A_q8;                    // per-projection Q8_1 input
        const CPUNativeVNNIPackedWeights *packed;  // packed weights
        const Q8_0Block *q8_0_raw;                // non-null for Q8_0 zero-copy path
        float *output;                            // output vector [N]
        int N;                                    // number of output rows
        int bpr;                                  // blocks per row
    };

    /**
     * @brief Fused multi-input GEMV: multiple projections with different inputs.
     *
     * Saves OMP fork/join overhead (3×~8µs per MoE layer for 4 experts)
     * and improves load balance via nowait between projections
     * (128 total chunks vs 4×32 = better utilization with 28 threads).
     */
    inline void gemv_fused_multi_input_preq(
        const FusedGemvMultiInputDesc *descs,
        int num_descs)
    {
        auto do_fused = [&]()
        {
            for (int p = 0; p < num_descs; ++p)
            {
                const auto &d = descs[p];

                if (d.q8_0_raw)
                {
                    const int proj_N = d.N;
                    const int proj_bpr = d.bpr;
                    const Q8_0Block *__restrict blocks = d.q8_0_raw;
                    const Q8_1Block *__restrict A_q8 = d.A_q8;

#pragma omp for schedule(static) nowait
                    for (int n = 0; n < proj_N; ++n)
                    {
                        const Q8_0Block *__restrict row = blocks + static_cast<size_t>(n) * proj_bpr;
                        float val = gemv_dot_row_q8_0_vnni(row, A_q8, proj_bpr);
                        d.output[n] = val;
                    }
                }
                else
                {
                    const auto &packed = *d.packed;
                    const int N = packed.N;
                    const int N_chunks = (N + 63) / 64;
                    const int K_blocks = packed.blocks_per_row;
                    const Q8_1Block *__restrict A_q8 = d.A_q8;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                    const __m512i decode_lut = packed.is_nibble_lut
                                                   ? build_decode_lut(packed.codebook_id)
                                                   : _mm512_setzero_si512();

#pragma omp for schedule(static) nowait
                    for (int chunk = 0; chunk < N_chunks; ++chunk)
                    {
                        gemv_native_vnni_avx512_block(packed, A_q8, d.output,
                                                      chunk, 1, K_blocks, N, decode_lut);
                    }
#else
                    const __m256i decode_lut = packed.is_nibble_lut
                                                   ? build_decode_lut_avx2_for_codebook(packed.codebook_id)
                                                   : _mm256_setzero_si256();
#pragma omp for schedule(static) nowait
                    for (int chunk = 0; chunk < N_chunks; ++chunk)
                    {
                        gemv_avx2_block(packed, A_q8, d.output, chunk, 1, K_blocks, N, decode_lut);
                    }
#endif
                }
            }

#pragma omp barrier
        };
        OMP_WORKSHARE_REGION(do_fused);
    }

} // namespace llaminar2::cpu::native_vnni
