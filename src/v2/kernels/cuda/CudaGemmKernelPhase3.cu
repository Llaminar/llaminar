/**
 * @file CudaGemmKernelPhase3.h
 * @brief Phase 3: Larger tiles for 2-3× performance improvement
 *
 * Phase 3 Implementation: Large tile optimization
 * - Tile size: 128×128×64 (up from 32×32×32)
 * - MMA atom layout: 2×2 (4 warps, up from 1×1)
 * - Thread count: 128 (4 warps, up from 1 warp)
 * - K-dimension blocking: Process K in 64-element chunks
 * - Shared memory: ~98 KB (within 100 KB limit)
 *
 * Expected Performance: 800-1,000 GFLOPS (2-3× improvement)
 *
 * @author David Sanftenberg
 * @date November 3, 2025
 */

#pragma once

#include <cuda_fp16.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

// IQ4_NL block structure (32 elements, 4-bit quantized)
struct IQ4_NLBlock
{
    float scale;        // FP32 scale factor
    uint8_t quants[16]; // 16 bytes for 32 4-bit values
};

// IQ4_NL quantization values lookup table (device constant)
__constant__ float iq4nl_values_p3[16] = {
    -127.0f / 127.0f, -104.0f / 127.0f, -83.0f / 127.0f, -65.0f / 127.0f,
    -49.0f / 127.0f, -35.0f / 127.0f, -22.0f / 127.0f, -10.0f / 127.0f,
    0.0f, 10.0f / 127.0f, 22.0f / 127.0f, 35.0f / 127.0f,
    49.0f / 127.0f, 65.0f / 127.0f, 83.0f / 127.0f, 104.0f / 127.0f};

/**
 * @brief Vectorized IQ4_NL decode (Phase 3 optimized)
 */
__device__ __forceinline__ void decode_iq4nl_to_fp16_p3(
    const IQ4_NLBlock *block,
    __half *output)
{
    const float scale = block->scale;
    const uint8_t *quants = block->quants;

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        uint32_t q4 = *reinterpret_cast<const uint32_t *>(&quants[i * 4]);

#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            uint8_t q = (q4 >> (j * 8)) & 0xFF;
            float val0 = scale * iq4nl_values_p3[q & 0xF];
            float val1 = scale * iq4nl_values_p3[q >> 4];

            reinterpret_cast<__half2 *>(output)[i * 4 + j] =
                __floats2half2_rn(val0, val1);
        }
    }
}

/**
 * @brief Phase 3: Large tile GEMM kernel
 *
 * Key improvements over Phase 2:
 * - 16× larger tiles (128×128 vs 32×32)
 * - 4× more threads per block (128 vs 32)
 * - K-dimension blocking for better arithmetic intensity
 * - 2×2 MMA atom layout for better warp cooperation
 *
 * @tparam TILE_M Output tile M dimension (128)
 * @tparam TILE_N Output tile N dimension (128)
 * @tparam TILE_K K-dimension tile (64)
 * @tparam MMA_M MMA atom layout M (2)
 * @tparam MMA_N MMA atom layout N (2)
 */
template <
    int TILE_M = 128,
    int TILE_N = 128,
    int TILE_K = 64,
    int MMA_M = 2,
    int MMA_N = 2>
__global__ void iq4nl_gemm_phase3_kernel(
    const float *__restrict__ A,
    const IQ4_NLBlock *__restrict__ B_blocks,
    float *__restrict__ C,
    int M, int N, int K)
{
    using namespace cute;

    // ========== 1. Define MMA Atom with 2×2 Layout ==========
    using MMA_Atom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
    using TiledMMA = TiledMMA<
        MMA_Atom,
        Layout<Shape<Int<MMA_M>, Int<MMA_N>, Int<1>>> // 2×2×1 layout
        >;
    TiledMMA tiled_mma;

    // ========== 2. Shared Memory Allocation ==========
    // A: 128×64 FP16 = 16,384 bytes
    // B: 128×64 FP16 = 16,384 bytes
    // Total: ~32 KB per K-tile (well within limits)
    __shared__ __half s_A[TILE_M][TILE_K];
    __shared__ __half s_B[TILE_N][TILE_K];

    // ========== 3. Block/Thread Indices ==========
    int block_m = blockIdx.x;
    int block_n = blockIdx.y;
    int tid = threadIdx.x;
    int num_threads = blockDim.x; // 128 threads

    int gm_start = block_m * TILE_M;
    int gn_start = block_n * TILE_N;

    // ========== 4. Create Global C Tensor ==========
    auto mC = make_tensor(
        make_gmem_ptr(C),
        make_shape(M, N),
        make_stride(N, Int<1>{}));

    auto cta_tiler = make_shape(Int<TILE_M>{}, Int<TILE_N>{}, Int<TILE_K>{});
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    // ========== 5. Partition for MMA and Create Accumulator ==========
    auto thr_mma = tiled_mma.get_slice(tid);
    auto tCgC = thr_mma.partition_C(gC);
    auto tCrC = thr_mma.make_fragment_C(tCgC);

    clear(tCrC); // Zero accumulator

    // ========== 6. K-Dimension Loop ==========
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile)
    {
        int gk_start = k_tile * TILE_K;
        int k_remaining = K - gk_start;
        int k_valid = (k_remaining < TILE_K) ? k_remaining : TILE_K;

        // ========== 7. Load A Tile (FP32→FP16) ==========
        for (int idx = tid; idx < TILE_M * TILE_K; idx += num_threads)
        {
            int m = idx / TILE_K;
            int k = idx % TILE_K;
            int gm = gm_start + m;
            int gk = gk_start + k;

            if (gm < M && gk < K)
            {
                s_A[m][k] = __float2half(A[gm * K + gk]);
            }
            else
            {
                s_A[m][k] = __float2half(0.0f);
            }
        }

        // ========== 8. Load B Tile (IQ4_NL→FP16) ==========
        constexpr int BLOCKS_PER_TILE = TILE_K / 32;
        int blocks_per_row = K / 32;
        int kb_start = k_tile * BLOCKS_PER_TILE;

        for (int n = tid; n < TILE_N; n += num_threads)
        {
            int gn = gn_start + n;
            if (gn < N)
            {
                for (int kb = 0; kb < BLOCKS_PER_TILE; kb++)
                {
                    int global_kb = kb_start + kb;
                    if (global_kb < blocks_per_row)
                    {
                        const IQ4_NLBlock *block = &B_blocks[gn * blocks_per_row + global_kb];
                        decode_iq4nl_to_fp16_p3(block, &s_B[n][kb * 32]);
                    }
                    else
                    {
#pragma unroll
                        for (int i = 0; i < 32; i++)
                        {
                            s_B[n][kb * 32 + i] = __float2half(0.0f);
                        }
                    }
                }
            }
            else
            {
                for (int k = 0; k < TILE_K; k++)
                {
                    s_B[n][k] = __float2half(0.0f);
                }
            }
        }

        __syncthreads();

        // ========== 9. Create Shared Memory Tensor Views ==========
        auto sA_tensor = make_tensor(
            make_smem_ptr(s_A[0]),
            make_shape(Int<TILE_M>{}, Int<TILE_K>{}),
            make_stride(Int<TILE_K>{}, Int<1>{}));

        auto sB_tensor = make_tensor(
            make_smem_ptr(s_B[0]),
            make_shape(Int<TILE_N>{}, Int<TILE_K>{}),
            make_stride(Int<TILE_K>{}, Int<1>{}));

        // ========== 10. Create Register Fragments and Copy ==========
        auto tCrA = thr_mma.partition_fragment_A(sA_tensor);
        auto tCrB = thr_mma.partition_fragment_B(sB_tensor);

        auto tCsA = thr_mma.partition_A(sA_tensor);
        auto tCsB = thr_mma.partition_B(sB_tensor);

        cute::copy(tCsA, tCrA);
        cute::copy(tCsB, tCrB);

        // ========== 11. Execute MMA and Accumulate ==========
        cute::gemm(tiled_mma, tCrA, tCrB, tCrC);

        __syncthreads();
    }

    // ========== 12. Write Results to Global Memory ==========
    cute::axpby(1.0f, tCrC, 0.0f, tCgC);
}

/**
 * @brief Phase 3 launcher
 */
template <
    int TILE_M,
    int TILE_N,
    int TILE_K,
    int MMA_M,
    int MMA_N,
    int THREADS_PER_BLOCK>
void launch_iq4nl_gemm_phase3(
    const float *A,
    const IQ4_NLBlock *B_blocks,
    float *C,
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 grid(
        (M + TILE_M - 1) / TILE_M,
        (N + TILE_N - 1) / TILE_N);
    dim3 block(THREADS_PER_BLOCK);

    iq4nl_gemm_phase3_kernel<TILE_M, TILE_N, TILE_K, MMA_M, MMA_N>
        <<<grid, block, 0, stream>>>(A, B_blocks, C, M, N, K);
}

// Explicit instantiation for common configurations
template void launch_iq4nl_gemm_phase3<128, 128, 64, 2, 2, 128>(
    const float *, const IQ4_NLBlock *, float *, int, int, int, cudaStream_t);
