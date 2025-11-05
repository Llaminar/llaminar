/**
 * @file CudaGemmKernelPhase3.h
 * @brief Header for Phase 3 large-tile GEMM kernel
 *
 * @author David Sanftenberg
 * @date November 3, 2025
 */

#pragma once

#include <cuda_runtime.h>

// Forward declarations
struct IQ4_NLBlock;

/**
 * @brief Launch Phase 3 large-tile GEMM kernel (128×128×64)
 *
 * @param A Input matrix A (M×K, row-major FP32)
 * @param B_blocks Input matrix B (N×K, IQ4_NL quantized)
 * @param C Output matrix C (M×N, row-major FP32)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param stream CUDA stream (default 0)
 */
template <
    int TILE_M = 128,
    int TILE_N = 128,
    int TILE_K = 64,
    int MMA_M = 2,
    int MMA_N = 2,
    int THREADS_PER_BLOCK = 128>
void launch_iq4nl_gemm_phase3(
    const float *A,
    const IQ4_NLBlock *B_blocks,
    float *C,
    int M, int N, int K,
    cudaStream_t stream = 0);
