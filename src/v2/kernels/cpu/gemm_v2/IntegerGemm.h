/**
 * @file IntegerGemm.h
 * @brief Integer-domain quantized GEMM (llama.cpp strategy)
 * @author David Sanftenberg
 * @date November 2025
 *
 * Strategy: Keep weights quantized (IQ4_NL/Q6_K/etc.), quantize activations
 * to Q8_0 on-the-fly, perform GEMM entirely in INT8 domain using AVX512-VNNI,
 * then dequantize only the final result to FP32.
 *
 * Benefits:
 * - Faster INT8×INT8 math (VPDPBUSD on AVX512-VNNI)
 * - Reduced memory bandwidth (Q8_0 activations vs FP32)
 * - Single dequantization at the end (vs per-weight-block)
 *
 * Algorithm:
 * 1. Quantize FP32 activation matrix A → Q8_0 (per-32-element blocks)
 * 2. For each weight block (IQ4_NL/Q6_K/etc.):
 *    a. Load quantized weight block
 *    b. Load corresponding Q8_0 activation block
 *    c. Compute INT8×INT8 dot product → INT32 accumulator (VNNI)
 * 3. Dequantize INT32 accumulator → FP32 using combined scale factors
 *
 * Instruction: VPDPBUSD (Dot Product of Signed Bytes and Unsigned Bytes with Dword Accumulation)
 *   _mm512_dpbusd_epi32(acc, a_u8, b_i8)
 *   acc[i] += a_u8[4*i+0]*b_i8[4*i+0] + a_u8[4*i+1]*b_i8[4*i+1] +
 *             a_u8[4*i+2]*b_i8[4*i+2] + a_u8[4*i+3]*b_i8[4*i+3]
 */

#pragma once

#include "tensors/Tensors.h"
#include "utils/CPUFeatures.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace llaminar2
{
    namespace kernels
    {

        /**
         * @brief Quantize FP32 activations to Q8_0 format
         *
         * @param src FP32 source data
         * @param dst Q8_0 destination blocks
         * @param n Number of elements (must be multiple of 32)
         *
         * Q8_0 format: Each block has 32 INT8 values + 1 FP16 scale
         * scale = max(abs(block)) / 127
         * quantized[i] = round(src[i] / scale)
         */
        void quantize_fp32_to_q8_0(const float *src, Q8_0Block *dst, size_t n);

        /**
         * @brief Integer-domain GEMM: C = A_fp32 × B_quantized (IQ4_NL weights)
         *
         * Strategy:
         * 1. Quantize each row of A to Q8_0 on-the-fly
         * 2. Perform INT8×INT8 dot products with B (kept in IQ4_NL format)
         * 3. Accumulate in INT32, dequantize at the end
         *
         * @param A FP32 activation matrix [M × K]
         * @param B IQ4_NL quantized weight matrix [K × N] (transposed: [N × K])
         * @param C FP32 output matrix [M × N]
         * @param M Number of output rows
         * @param N Number of output columns
         * @param K Inner dimension
         * @param lda Leading dimension of A (usually K)
         * @param ldc Leading dimension of C (usually N)
         *
         * @return true if successful, false if AVX512-VNNI not available
         *
         * Requires: AVX512-VNNI CPU support (Ice Lake+, Zen 4+)
         */
        bool gemm_int8_iq4nl_vnni(
            const float *A,
            const IQ4_NLTensor *B,
            float *C,
            int M, int N, int K,
            int lda, int ldc);

        /**
         * @brief Integer-domain GEMM: C = A_fp32 × B_quantized (Q6_K weights)
         *
         * Same strategy as IQ4_NL variant but for Q6_K format.
         *
         * @param A FP32 activation matrix [M × K]
         * @param B Q6_K quantized weight matrix [K × N] (transposed: [N × K])
         * @param C FP32 output matrix [M × N]
         * @param M Number of output rows
         * @param N Number of output columns
         * @param K Inner dimension
         * @param lda Leading dimension of A (usually K)
         * @param ldc Leading dimension of C (usually N)
         *
         * @return true if successful, false if AVX512-VNNI not available
         */
        bool gemm_int8_q6k_vnni(
            const float *A,
            const Q6_KTensor *B,
            float *C,
            int M, int N, int K,
            int lda, int ldc);

        /**
         * @brief Generic integer-domain GEMM dispatcher
         *
         * Automatically selects the appropriate GEMM variant based on
         * weight tensor type and CPU capabilities.
         *
         * @param A FP32 activation matrix [M × K]
         * @param B Quantized weight tensor (IQ4_NL, Q6_K, Q8_0, etc.)
         * @param C FP32 output matrix [M × N]
         * @param M Number of output rows
         * @param N Number of output columns
         * @param K Inner dimension
         *
         * @return true if successful, false if unsupported format or no VNNI
         */
        bool gemm_int8_dispatch(
            const float *A,
            const TensorBase *B,
            float *C,
            int M, int N, int K);

    } // namespace kernels
} // namespace llaminar2
