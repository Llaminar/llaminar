/**
 * @file CudaGemmKernelPhase4QuickWins.h
 * @brief Phase 4 quick wins kernel header
 *
 * @author David Sanftenberg
 * @date November 4, 2025
 */

#pragma once

struct IQ4_NLBlock;

/**
 * @brief Launch Phase 4 kernel (swizzled + cp.async optimized)
 *
 * Expected: 7.6 TFLOPS at M=1024 (+16% over Phase 3 Part 2)
 *
 * @tparam TILE_M Output tile M (default 64)
 * @tparam TILE_N Output tile N (default 64)
 * @tparam TILE_K K-tile size (default 64)
 */
template <int TILE_M = 64, int TILE_N = 64, int TILE_K = 64>
void launch_iq4nl_gemm_phase4(
    const float *A,
    const IQ4_NLBlock *B_blocks,
    float *C,
    int M, int N, int K);
