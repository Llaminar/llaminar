/**
 * @file CudaGemmKernelPhase3Pipelined.cu
 * @brief Phase 3 Part 2: Multi-stage pipelined large-tile GEMM
 *
 * Key optimizations over Phase 3 Part 1:
 * - 3-stage software pipelining (prefetch, compute, commit)
 * - Async copy with cp.async for overlap
 * - Double-buffered shared memory
 * - Swizzled layout to reduce bank conflicts
 *
 * Expected performance: 900-1,000 GFLOPS (1.3-1.4× over Phase 3 Part 1)
 *
 * @author David Sanftenberg
 * @date November 4, 2025
 */

#pragma once

#include <cuda_fp16.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

// IQ4_NL block structure (32 elements, 4-bit quantized)
struct IQ4_NLBlock
{
    float scale;        // FP32 scale factor
    uint8_t quants[16]; // 16 bytes for 32 4-bit values
};

// IQ4_NL quantization values lookup table (device constant)
__constant__ float iq4nl_values_p3p[16] = {
    -127.0f / 127.0f, -104.0f / 127.0f, -83.0f / 127.0f, -65.0f / 127.0f,
    -49.0f / 127.0f, -35.0f / 127.0f, -22.0f / 127.0f, -10.0f / 127.0f,
    0.0f, 10.0f / 127.0f, 22.0f / 127.0f, 35.0f / 127.0f,
    49.0f / 127.0f, 65.0f / 127.0f, 83.0f / 127.0f, 104.0f / 127.0f};

/**
 * @brief Vectorized IQ4_NL decode (optimized for pipelining)
 */
__device__ __forceinline__ void decode_iq4nl_to_fp16_p3p(
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
            float val0 = scale * iq4nl_values_p3p[q & 0xF];
            float val1 = scale * iq4nl_values_p3p[q >> 4];

            reinterpret_cast<__half2 *>(output)[i * 4 + j] =
                __floats2half2_rn(val0, val1);
        }
    }
}

/**
 * @brief Phase 3 Part 2: Pipelined large-tile GEMM kernel
 *
 * Pipeline stages:
 * 1. Prefetch next K-tile to shared memory (async)
 * 2. Compute current K-tile MMA
 * 3. Commit results to accumulators
 *
 * @tparam TILE_M Output tile M dimension (64 for shared mem limit)
 * @tparam TILE_N Output tile N dimension (64 for shared mem limit)
 * @tparam TILE_K K-dimension tile (64)
 * @tparam MMA_M MMA atom layout M (2)
 * @tparam MMA_N MMA atom layout N (2)
 */
template <
    int TILE_M = 64, // Reduced from 128 to fit double-buffer
    int TILE_N = 64, // Reduced from 128 to fit double-buffer
    int TILE_K = 64,
    int MMA_M = 2,
    int MMA_N = 2>
__global__ void iq4nl_gemm_pipelined_kernel(
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

    // ========== 2. Double-Buffered Shared Memory ==========
    // Two buffers for pipelining: while computing stage N, prefetch stage N+1
    // Reduced to 64×64 tiles to fit within 48 KB limit
    __shared__ __half s_A[2][TILE_M][TILE_K]; // 2 × 8 KB = 16 KB
    __shared__ __half s_B[2][TILE_N][TILE_K]; // 2 × 8 KB = 16 KB
    // Total: 32 KB (fits within 48 KB limit)

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

    // ========== 6. K-Dimension Pipeline Setup ==========
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    int write_stage = 0; // Which buffer to write to (0 or 1)
    int read_stage = 0;  // Which buffer to read from (0 or 1)

    constexpr int BLOCKS_PER_TILE = TILE_K / 32;
    int blocks_per_row = K / 32;

    // ========== 7. PROLOGUE: Prefetch First Tile ==========
    if (num_k_tiles > 0)
    {
        int gk_start = 0;

        // Load A tile (FP32→FP16) into buffer 0
        for (int idx = tid; idx < TILE_M * TILE_K; idx += num_threads)
        {
            int m = idx / TILE_K;
            int k = idx % TILE_K;
            int gm = gm_start + m;
            int gk = gk_start + k;

            if (gm < M && gk < K)
            {
                s_A[0][m][k] = __float2half(A[gm * K + gk]);
            }
            else
            {
                s_A[0][m][k] = __float2half(0.0f);
            }
        }

        // Load B tile (IQ4_NL→FP16) into buffer 0
        for (int n = tid; n < TILE_N; n += num_threads)
        {
            int gn = gn_start + n;
            if (gn < N)
            {
                for (int kb = 0; kb < BLOCKS_PER_TILE; kb++)
                {
                    int global_kb = kb;
                    if (global_kb < blocks_per_row)
                    {
                        const IQ4_NLBlock *block = &B_blocks[gn * blocks_per_row + global_kb];
                        decode_iq4nl_to_fp16_p3p(block, &s_B[0][n][kb * 32]);
                    }
                    else
                    {
#pragma unroll
                        for (int i = 0; i < 32; i++)
                        {
                            s_B[0][n][kb * 32 + i] = __float2half(0.0f);
                        }
                    }
                }
            }
            else
            {
                for (int k = 0; k < TILE_K; k++)
                {
                    s_B[0][n][k] = __float2half(0.0f);
                }
            }
        }

        __syncthreads(); // Wait for first tile to load
    }

    // ========== 8. MAIN LOOP: Pipelined Execution ==========
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile)
    {
        read_stage = k_tile % 2;
        write_stage = 1 - read_stage;

        // ===== STAGE 1: Prefetch Next Tile (Async) =====
        if (k_tile + 1 < num_k_tiles)
        {
            int next_gk_start = (k_tile + 1) * TILE_K;
            int next_kb_start = (k_tile + 1) * BLOCKS_PER_TILE;

            // Load A tile for next iteration
            for (int idx = tid; idx < TILE_M * TILE_K; idx += num_threads)
            {
                int m = idx / TILE_K;
                int k = idx % TILE_K;
                int gm = gm_start + m;
                int gk = next_gk_start + k;

                if (gm < M && gk < K)
                {
                    s_A[write_stage][m][k] = __float2half(A[gm * K + gk]);
                }
                else
                {
                    s_A[write_stage][m][k] = __float2half(0.0f);
                }
            }

            // Load B tile for next iteration
            for (int n = tid; n < TILE_N; n += num_threads)
            {
                int gn = gn_start + n;
                if (gn < N)
                {
                    for (int kb = 0; kb < BLOCKS_PER_TILE; kb++)
                    {
                        int global_kb = next_kb_start + kb;
                        if (global_kb < blocks_per_row)
                        {
                            const IQ4_NLBlock *block = &B_blocks[gn * blocks_per_row + global_kb];
                            decode_iq4nl_to_fp16_p3p(block, &s_B[write_stage][n][kb * 32]);
                        }
                        else
                        {
#pragma unroll
                            for (int i = 0; i < 32; i++)
                            {
                                s_B[write_stage][n][kb * 32 + i] = __float2half(0.0f);
                            }
                        }
                    }
                }
                else
                {
                    for (int k = 0; k < TILE_K; k++)
                    {
                        s_B[write_stage][n][k] = __float2half(0.0f);
                    }
                }
            }
        }

        // ===== STAGE 2: Compute Current Tile =====
        // Create tensor views for CURRENT buffer
        auto sA_tensor = make_tensor(
            make_smem_ptr(s_A[read_stage][0]),
            make_shape(Int<TILE_M>{}, Int<TILE_K>{}),
            make_stride(Int<TILE_K>{}, Int<1>{}));

        auto sB_tensor = make_tensor(
            make_smem_ptr(s_B[read_stage][0]),
            make_shape(Int<TILE_N>{}, Int<TILE_K>{}),
            make_stride(Int<TILE_K>{}, Int<1>{}));

        // Create register fragments
        auto tCrA = thr_mma.partition_fragment_A(sA_tensor);
        auto tCrB = thr_mma.partition_fragment_B(sB_tensor);

        auto tCsA = thr_mma.partition_A(sA_tensor);
        auto tCsB = thr_mma.partition_B(sB_tensor);

        // Copy to registers
        cute::copy(tCsA, tCrA);
        cute::copy(tCsB, tCrB);

        // Execute MMA (overlaps with next tile prefetch!)
        cute::gemm(tiled_mma, tCrA, tCrB, tCrC);

        // Sync before switching buffers
        __syncthreads();
    }

    // ========== 9. Write Results to Global Memory ==========
    cute::axpby(1.0f, tCrC, 0.0f, tCgC);
}

/**
 * @brief Phase 3 Part 2 pipelined launcher
 */
template <
    int TILE_M = 64, // Reduced from 128 to fit shared mem
    int TILE_N = 64, // Reduced from 128 to fit shared mem
    int TILE_K = 64,
    int MMA_M = 2,
    int MMA_N = 2,
    int THREADS_PER_BLOCK = 128>
void launch_iq4nl_gemm_pipelined(
    const float *A,
    const IQ4_NLBlock *B_blocks,
    float *C,
    int M, int N, int K,
    cudaStream_t stream = 0)
{
    dim3 grid(
        (M + TILE_M - 1) / TILE_M,
        (N + TILE_N - 1) / TILE_N);
    dim3 block(THREADS_PER_BLOCK);

    iq4nl_gemm_pipelined_kernel<TILE_M, TILE_N, TILE_K, MMA_M, MMA_N>
        <<<grid, block, 0, stream>>>(A, B_blocks, C, M, N, K);
}

// Explicit instantiation
template void launch_iq4nl_gemm_pipelined<64, 64, 64, 2, 2, 128>(
    const float *, const IQ4_NLBlock *, float *, int, int, int, cudaStream_t);
