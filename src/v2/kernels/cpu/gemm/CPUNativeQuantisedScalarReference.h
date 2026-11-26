/**
 * @file CPUNativeQuantisedScalarReference.h
 * @brief ISA-neutral scalar NativeQuantised GEMV oracle shared by x86 and ARM tests.
 *
 * The oracle reconstructs each 32-element block from the same prepared
 * interleaved payload the production kernels consume, so it needs no second
 * source-order matrix. It is a diagnostic reference for parity tests, never a
 * production dispatch target, and contains no SIMD intrinsics so every CPU
 * architecture compiles the identical program.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "CPUNativeQuantisedDecode.h"
#include "CPUNativeQuantisedFP16.h"
#include "CPUNativeQuantisedMultiScaleKernels.h"
#include "CPUNativeQuantisedPackedWeights.h"
#include "kernels/common/DeviceFP32NumericalContract.h"
#include "tensors/BlockStructures.h"

namespace llaminar2::cpu::native_quantised
{
    // =========================================================================
    // Scalar reference GEMV (any format, for correctness verification)
    // =========================================================================

    /**
     * @brief Scalar NativeQuantised GEMV for correctness reference.
     *
     * For nibble-LUT formats (Q4_0, IQ4_NL, Q4_1, IQ4_XS): decodes from payload.
     * For INT8 pre-decoded formats: reads from int8_flat buffer.
     * Q6 uses the original base/coefficient metadata with separate FP32 edges.
     * @param packed Prepared source payload and per-block metadata.
     * @param A_q8 Unchanged Q8 activation blocks for one row.
     * @param C Destination for N output values.
     * @param N Live output columns, excluding padded allocation capacity.
     * @param K_blocks Number of live 32-element reduction blocks.
     */
    inline void gemv_native_quantised_scalar(
        const CPUNativeQuantisedPackedWeights &packed,
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
                    float a_scale = nativeQuantisedFP16ScaleToFP32(a_blk.d);

                    if (packed.usesCompactMultiScale())
                    {
                        acc += multi_scale::scalarReferenceBlock(packed, a_blk, chunk, kb, n_local);
                        continue;
                    }
                    if (packed.usesQ6KNativeDualScale())
                    {
                        int32_t dot_low = 0;
                        int32_t dot_high = 0;
                        const int z = n_local / 16;
                        const int lane = n_local % 16;
                        for (int group = 0; group < 4; ++group)
                        {
                            const uint8_t *const interleaved =
                                packed.interleavedB(chunk, kb, group, z) +
                                lane * 4;
                            uint64_t low_bitplane0 = 0;
                            uint64_t low_bitplane1 = 0;
                            uint64_t high_bitplane0 = 0;
                            uint64_t high_bitplane1 = 0;
                            const uint8_t *const low_bitplanes =
                                packed.q6KHighBitplanes(
                                    chunk, kb, group, z);
                            const uint8_t *const high_bitplanes =
                                packed.q6KHighBitplanes(
                                    chunk, kb, group + 4, z);
                            std::memcpy(
                                &low_bitplane0,
                                low_bitplanes,
                                sizeof(low_bitplane0));
                            std::memcpy(
                                &low_bitplane1,
                                low_bitplanes + sizeof(low_bitplane0),
                                sizeof(low_bitplane1));
                            std::memcpy(
                                &high_bitplane0,
                                high_bitplanes,
                                sizeof(high_bitplane0));
                            std::memcpy(
                                &high_bitplane1,
                                high_bitplanes + sizeof(high_bitplane0),
                                sizeof(high_bitplane1));
                            for (int index = 0; index < 4; ++index)
                            {
                                const int bit_index = lane * 4 + index;
                                const int low_high_part =
                                    static_cast<int>(
                                        (low_bitplane0 >> bit_index) & 1u) |
                                    (static_cast<int>(
                                         (low_bitplane1 >> bit_index) & 1u)
                                     << 1);
                                const int high_high_part =
                                    static_cast<int>(
                                        (high_bitplane0 >> bit_index) & 1u) |
                                    (static_cast<int>(
                                         (high_bitplane1 >> bit_index) & 1u)
                                     << 1);
                                const int weight_low =
                                    static_cast<int>(interleaved[index] & 0x0F) |
                                    (low_high_part << 4);
                                const int weight_high =
                                    static_cast<int>(interleaved[index] >> 4) |
                                    (high_high_part << 4);
                                dot_low +=
                                    static_cast<int>(
                                        a_blk.qs[group * 4 + index]) *
                                    (weight_low - 32);
                                dot_high +=
                                    static_cast<int>(
                                        a_blk.qs[group * 4 + index + 16]) *
                                    (weight_high - 32);
                            }
                        }

                        // The scalar diagnostic must retain the same separately
                        // rounded dependencies as the SIMD and device Q6 paths.
                        // Ordinary expressions permit contraction under -mfma,
                        // hiding source-factor errors behind a different oracle.
                        const float low_term = device_fp32_contract::multiply(
                            packed.blockScale(chunk, kb, n_local),
                            static_cast<float>(dot_low));
                        const float high_term = device_fp32_contract::multiply(
                            packed.blockMin(chunk, kb, n_local),
                            static_cast<float>(dot_high));
                        const float dot_term = device_fp32_contract::add(low_term, high_term);
                        const float contribution = device_fp32_contract::multiply(a_scale, dot_term);
                        acc = device_fp32_contract::add(acc, contribution);
                        continue;
                    }

                    int8_t b_vals[32];
                    if (packed.usesCompactFiveBit())
                    {
                        const int z = n_local / 16;
                        const int lane = n_local % 16;
                        const int center = packed.is_asymmetric ? 0 : 16;
                        for (int group = 0; group < 4; ++group)
                        {
                            const uint8_t *const source =
                                packed.interleavedB(chunk, kb, group, z) + lane * 4;
                            uint64_t lo_mask = 0, hi_mask = 0;
                            std::memcpy(&lo_mask, packed.fiveBitHighBitplane(chunk, kb, group, z), sizeof(lo_mask));
                            std::memcpy(&hi_mask, packed.fiveBitHighBitplane(chunk, kb, group + 4, z), sizeof(hi_mask));
                            for (int idx = 0; idx < 4; ++idx)
                            {
                                const int bit = lane * 4 + idx;
                                const int lo = (source[idx] & 0x0F) | (static_cast<int>((lo_mask >> bit) & 1u) << 4);
                                const int hi = (source[idx] >> 4) | (static_cast<int>((hi_mask >> bit) & 1u) << 4);
                                b_vals[group * 4 + idx] = static_cast<int8_t>(lo - center);
                                b_vals[16 + group * 4 + idx] = static_cast<int8_t>(hi - center);
                            }
                        }
                    }
                    else if (packed.usesNibbleLUT())
                    {
                        // Four groups contain the original sixteen native
                        // bytes for this column. Reconstruct only this bounded
                        // diagnostic block; retaining another matrix solely for
                        // the scalar oracle would violate prepared-memory BOM.
                        uint8_t payload[16];
                        const int z = n_local / 16;
                        const int lane = n_local % 16;
                        for (int group = 0; group < 4; ++group)
                        {
                            const auto *source = packed.interleavedB(chunk, kb, group, z);
                            std::memcpy(payload + group * 4, source + lane * 4, 4);
                        }
                        decode_nibble_lut_block(packed.codebook_id, payload, b_vals);
                    }
                    else if (packed.usesUnsignedInt8())
                    {
                        const int z = n_local / 16;
                        const int lane = n_local % 16;
                        for (int group = 0; group < 8; ++group)
                        {
                            const auto *group_data = packed.interleavedB(chunk, kb, group, z);
                            const int src = lane * 4;
                            const int dst = group * 4;
                            b_vals[dst + 0] = static_cast<int8_t>(group_data[src + 0] ^ 0x80u);
                            b_vals[dst + 1] = static_cast<int8_t>(group_data[src + 1] ^ 0x80u);
                            b_vals[dst + 2] = static_cast<int8_t>(group_data[src + 2] ^ 0x80u);
                            b_vals[dst + 3] = static_cast<int8_t>(group_data[src + 3] ^ 0x80u);
                        }
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

    /**
     * @brief Exact single-chunk, single-K-interval scalar NativeQuantised GEMV.
     *
     * Mirrors the exact per-block integer dot, bias correction, scale/min
     * products, and ascending FP32 accumulation order of the AVX2 and AVX-512
     * microkernels across all 18 NativeQuantised codebooks, using
     * `device_fp32_contract` arithmetic so compiler FMA contraction cannot
     * drift from the SIMD/GPU contract.
     *
     * @param packed Prepared NativeQuantised matrix.
     * @param A_q8 Contiguous Q8_1 blocks for one activation row.
     * @param C_chunk Destination span for up to `columns` (<= 64) output values.
     * @param chunk 64-column output chunk index.
     * @param kb_start Inclusive starting K-block index.
     * @param kb_end Exclusive ending K-block index.
     * @param columns Live output columns within this chunk (`1..64`).
     */
    inline void gemv_native_quantised_scalar_chunk(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C_chunk,
        int chunk,
        int kb_start,
        int kb_end,
        int columns = 64) noexcept
    {
        const bool gpu_aligned =
            packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert;
        for (int n_local = 0; n_local < columns; ++n_local)
        {
            float acc = 0.0f;
            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                const Q8_1Block &a_blk = A_q8[kb];
                const float a_scale = nativeQuantisedFP16ScaleToFP32(a_blk.d);

                if (packed.usesCompactMultiScale())
                {
                    const float contrib =
                        multi_scale::scalarReferenceBlock(packed, a_blk, chunk, kb, n_local);
                    acc = device_fp32_contract::add(acc, contrib);
                    continue;
                }
                if (packed.usesQ6KNativeDualScale())
                {
                    int32_t dot_low = 0;
                    int32_t dot_high = 0;
                    const int z = n_local / 16;
                    const int lane = n_local % 16;
                    for (int group = 0; group < 4; ++group)
                    {
                        const uint8_t *const interleaved =
                            packed.interleavedB(chunk, kb, group, z) + lane * 4;
                        uint64_t low_bp0 = 0, low_bp1 = 0, high_bp0 = 0, high_bp1 = 0;
                        const uint8_t *const low_bp = packed.q6KHighBitplanes(chunk, kb, group, z);
                        const uint8_t *const high_bp = packed.q6KHighBitplanes(chunk, kb, group + 4, z);
                        std::memcpy(&low_bp0, low_bp, sizeof(low_bp0));
                        std::memcpy(&low_bp1, low_bp + sizeof(low_bp0), sizeof(low_bp1));
                        std::memcpy(&high_bp0, high_bp, sizeof(high_bp0));
                        std::memcpy(&high_bp1, high_bp + sizeof(high_bp0), sizeof(high_bp1));
                        for (int index = 0; index < 4; ++index)
                        {
                            const int bit_index = lane * 4 + index;
                            const int low_hi = static_cast<int>((low_bp0 >> bit_index) & 1u) |
                                               (static_cast<int>((low_bp1 >> bit_index) & 1u) << 1);
                            const int high_hi = static_cast<int>((high_bp0 >> bit_index) & 1u) |
                                                (static_cast<int>((high_bp1 >> bit_index) & 1u) << 1);
                            const int w_low = static_cast<int>(interleaved[index] & 0x0F) | (low_hi << 4);
                            const int w_high = static_cast<int>(interleaved[index] >> 4) | (high_hi << 4);
                            dot_low += static_cast<int>(a_blk.qs[group * 4 + index]) * (w_low - 32);
                            dot_high += static_cast<int>(a_blk.qs[group * 4 + index + 16]) * (w_high - 32);
                        }
                    }
                    const float low_term = device_fp32_contract::multiply(
                        packed.blockScale(chunk, kb, n_local), static_cast<float>(dot_low));
                    const float high_term = device_fp32_contract::multiply(
                        packed.blockMin(chunk, kb, n_local), static_cast<float>(dot_high));
                    const float dot_term = device_fp32_contract::add(low_term, high_term);
                    const float contribution = device_fp32_contract::multiply(a_scale, dot_term);
                    acc = device_fp32_contract::add(acc, contribution);
                    continue;
                }

                int8_t b_vals[32];
                if (packed.usesCompactFiveBit())
                {
                    const int z = n_local / 16;
                    const int lane = n_local % 16;
                    const int center = packed.is_asymmetric ? 0 : 16;
                    for (int group = 0; group < 4; ++group)
                    {
                        const uint8_t *const source =
                            packed.interleavedB(chunk, kb, group, z) + lane * 4;
                        uint64_t lo_mask = 0, hi_mask = 0;
                        std::memcpy(&lo_mask, packed.fiveBitHighBitplane(chunk, kb, group, z), sizeof(lo_mask));
                        std::memcpy(&hi_mask, packed.fiveBitHighBitplane(chunk, kb, group + 4, z), sizeof(hi_mask));
                        for (int idx = 0; idx < 4; ++idx)
                        {
                            const int bit = lane * 4 + idx;
                            const int lo = (source[idx] & 0x0F) | (static_cast<int>((lo_mask >> bit) & 1u) << 4);
                            const int hi = (source[idx] >> 4) | (static_cast<int>((hi_mask >> bit) & 1u) << 4);
                            b_vals[group * 4 + idx] = static_cast<int8_t>(lo - center);
                            b_vals[16 + group * 4 + idx] = static_cast<int8_t>(hi - center);
                        }
                    }
                }
                else if (packed.usesNibbleLUT())
                {
                    uint8_t payload[16];
                    const int z = n_local / 16;
                    const int lane = n_local % 16;
                    for (int group = 0; group < 4; ++group)
                    {
                        const auto *source = packed.interleavedB(chunk, kb, group, z);
                        std::memcpy(payload + group * 4, source + lane * 4, 4);
                    }
                    decode_nibble_lut_block(packed.codebook_id, payload, b_vals);
                }
                else if (packed.usesUnsignedInt8())
                {
                    const int z = n_local / 16;
                    const int lane = n_local % 16;
                    for (int group = 0; group < 8; ++group)
                    {
                        const auto *group_data = packed.interleavedB(chunk, kb, group, z);
                        const int src = lane * 4;
                        const int dst = group * 4;
                        b_vals[dst + 0] = static_cast<int8_t>(group_data[src + 0] ^ 0x80u);
                        b_vals[dst + 1] = static_cast<int8_t>(group_data[src + 1] ^ 0x80u);
                        b_vals[dst + 2] = static_cast<int8_t>(group_data[src + 2] ^ 0x80u);
                        b_vals[dst + 3] = static_cast<int8_t>(group_data[src + 3] ^ 0x80u);
                    }
                }
                else if (!packed.int8_flat.empty())
                {
                    std::memcpy(b_vals, packed.blockInt8(chunk, kb, n_local), 32);
                }
                else
                {
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

                int32_t dot = 0;
                int32_t b_comp = 0;
                for (int i = 0; i < 32; ++i)
                {
                    const uint8_t a_u8 = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[i]) + 128);
                    dot += static_cast<int32_t>(a_u8) * static_cast<int32_t>(b_vals[i]);
                    b_comp += b_vals[i];
                }
                const int32_t corrected = dot - 128 * b_comp;
                const float b_scale = packed.blockScale(chunk, kb, n_local);
                const float b_min = packed.blockMin(chunk, kb, n_local);

                if (gpu_aligned)
                {
                    const float scaled_weight = device_fp32_contract::multiply(a_scale, b_scale);
                    const float dot_contrib =
                        device_fp32_contract::multiply(scaled_weight, static_cast<float>(corrected));
                    if (packed.is_asymmetric)
                    {
                        const float scaled_min = device_fp32_contract::multiply(a_scale, b_min);
                        const float correction = device_fp32_contract::multiply(
                            scaled_min, static_cast<float>(a_blk.sum_qs));
                        const float completed = device_fp32_contract::add(dot_contrib, correction);
                        acc = device_fp32_contract::add(acc, completed);
                    }
                    else
                    {
                        acc = device_fp32_contract::add(acc, dot_contrib);
                    }
                }
                else
                {
                    const float scaled_weight = device_fp32_contract::multiply(a_scale, b_scale);
                    acc = std::fma(static_cast<float>(corrected), scaled_weight, acc);
                    if (packed.is_asymmetric)
                    {
                        const float scaled_min = device_fp32_contract::multiply(a_scale, b_min);
                        acc = std::fma(static_cast<float>(a_blk.sum_qs), scaled_min, acc);
                    }
                }
            }
            C_chunk[n_local] = acc;
        }
    }

    /**
     * @brief Exact scalar ordered-K-partition GEMV for one 64-column chunk.
     *
     * Computes each partition `[kb_start, kb_end)` starting from positive zero
     * and folds left-to-right with `device_fp32_contract::add`, matching the
     * AVX2, AVX-512, CUDA, and ROCm ordered-partition contract bitwise.
     */
    inline void gemvScalarOrderedPartitions(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *row,
        float *output,
        int chunk,
        int K_blocks,
        int k_tiles,
        int k_blocks_per_tile,
        int columns = 64) noexcept
    {
        if (k_tiles <= 1)
        {
            gemv_native_quantised_scalar_chunk(packed, row, output, chunk, 0, K_blocks, columns);
            return;
        }
        alignas(64) float partial[64];
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile)
        {
            const int kb_start = k_tile * k_blocks_per_tile;
            const int kb_end = std::min(kb_start + k_blocks_per_tile, K_blocks);
            gemv_native_quantised_scalar_chunk(
                packed, row, k_tile == 0 ? output : partial, chunk, kb_start, kb_end, columns);
            if (k_tile > 0)
            {
                for (int c = 0; c < columns; ++c)
                    output[c] = device_fp32_contract::add(output[c], partial[c]);
            }
        }
    }
} // namespace llaminar2::cpu::native_quantised
