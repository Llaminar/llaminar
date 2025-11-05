/**
 * @file CudaGemmKernelPhase5ASimple.h
 * @brief Phase 5A: Streaming dequantization (overlap decode with MMA)
 *
 * @author David Sanftenberg
 * @date November 4, 2025
 */

#pragma once

// IQ4_NL block forward declaration
struct IQ4_NLBlock;

/**
 * @brief Launch Phase 5A streaming dequantization GEMM kernel
 *
 * Key optimization: Decode B sub-tiles inside K-loop
 * - CUDA cores (dequant) work while Tensor Cores (MMA) are busy
 * - Expected +15-25% gain from instruction-level parallelism
 *
 * @param A Input matrix A (M×K, FP32)
 * @param B_blocks Input matrix B (N×K, IQ4_NL quantized)
 * @param C Output matrix C (M×N, FP32)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 */
template <int TILE_M = 64, int TILE_N = 64, int TILE_K = 64, int SUB_K = 16>
void launch_iq4nl_gemm_phase5a(
    const float *A,
    const IQ4_NLBlock *B_blocks,
    float *C,
    int M, int N, int K);
