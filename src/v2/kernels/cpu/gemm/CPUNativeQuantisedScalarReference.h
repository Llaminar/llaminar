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
#include <cmath>
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
     * For signed Q8: reads from the same interleaved publication as SIMD.
     * Q6_K and IQ4_XS reconstruct the original half base and coefficients
     * before applying their respective FP32 contribution programs.
     * @param packed Prepared source payload and metadata.
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
                    if (packed.encoding == CPUNativeQuantisedEncoding::Q5NativeSingleScale)
                    {
                        const auto *unit = packed.interleavedBase() + packed.blockOffset(chunk, kb);
                        for (int group = 0; group < 4; ++group)
                        {
                            const auto *low = unit + Q5NativePayload::nibbleOffset(group, n_local);
                            const unsigned fifth = unit[Q5NativePayload::fifthOffset(group, n_local)];
                            const int bias = packed.is_asymmetric ? 0 : 16;
                            for (int i = 0; i < 4; ++i)
                            {
                                b_vals[group * 4 + i] = (low[i] & 15) + 16 * ((fifth >> i) & 1) - bias;
                                b_vals[group * 4 + 16 + i] = (low[i] >> 4) + 16 * ((fifth >> (i + 4)) & 1) - bias;
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
                        decode_native_block(packed.codebook_id, payload, b_vals);
                    }
                    else
                    {
                        const int z = n_local / 16;
                        const int lane = n_local % 16;
                        for (int group = 0; group < 8; ++group)
                        {
                            const auto *group_data = packed.interleavedB(chunk, kb, group, z);
                            std::memcpy(b_vals + group * 4, group_data + lane * 4, 4);
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
                    if (packed.usesNibbleLUTSourceScale())
                    {
                        const float combined = device_fp32_contract::multiply(a_scale, b_scale);
                        const float dot_fp32 = static_cast<float>(corrected);
                        acc = packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert
                            ? device_fp32_contract::add(acc, device_fp32_contract::multiply(combined, dot_fp32))
                            : std::fma(combined, dot_fp32, acc);
                        continue;
                    }
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
} // namespace llaminar2::cpu::native_quantised
