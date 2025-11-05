/**
 * @file CudaGemmKernelPhase3Pipelined.h
 * @brief Header for Phase 3 Part 2 pipelined GEMM kernel
 *
 * @author David Sanftenberg
 * @date November 4, 2025
 */

#pragma once

#include <cuda_runtime.h>

// Forward declarations
struct IQ4_NLBlock;

/**
 * @brief Launch Phase 3 Part 2 pipelined GEMM kernel
 *
 * 3-stage pipeline with double-buffered shared memory
 * Tile size: 64×64×64 (reduced from 128×128×64 to fit shared mem limit)
 * Expected: 700-800 GFLOPS (similar to Phase 3 Part 1 but with pipelining)
 */
template <
    int TILE_M = 64,
    int TILE_N = 64,
    int TILE_K = 64,
    int MMA_M = 2,
    int MMA_N = 2,
    int THREADS_PER_BLOCK = 128>
void launch_iq4nl_gemm_pipelined(
    const float *A,
    const IQ4_NLBlock *B_blocks,
    float *C,
    int M, int N, int K,
    cudaStream_t stream = 0);
