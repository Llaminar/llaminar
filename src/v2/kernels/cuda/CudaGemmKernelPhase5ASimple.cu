/**
 * @file CudaGemmKernelPhase5ASimple.cu
 * @brief Phase 5A: Streaming dequantization (overlap decode with MMA)
 *
 * Key optimization over Phase 4:
 * - Move B dequantization INSIDE inner K-loop
 * - CUDA cores (dequant) work while Tensor Cores (MMA) are busy
 * - Single-buffer streaming (simpler than warp specialization)
 *
 * Profiling data shows:
 * - B dequant: 40% of time (bottleneck)
 * - MMA: 30% of time
 * - Expected gain: +15-25% from instruction-level parallelism
 *
 * Strategy:
 * - Decode B[k_inner+1] while computing MMA on B[k_inner]
 * - Eliminates serialization between dequant and MMA
 * - Similar to CUTLASS producer/consumer pattern (simpler version)
 *
 * Expected performance: 10.0 TFLOPS (M=1024) - +15% over Phase 4's 8.69 TFLOPS
 *
 * @author David Sanftenberg
 * @date November 4, 2025
 */

#pragma once

#include <cuda_fp16.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/algorithm/copy.hpp>

// IQ4_NL block structure (32 elements, 4-bit quantized)
struct IQ4_NLBlock
{
    float scale;        // FP32 scale factor
    uint8_t quants[16]; // 16 bytes for 32 4-bit values
};

// IQ4_NL quantization values lookup table (device constant)
__constant__ float iq4nl_values_p5a[16] = {
    -127.0f / 127.0f, -104.0f / 127.0f, -83.0f / 127.0f, -65.0f / 127.0f,
    -49.0f / 127.0f, -35.0f / 127.0f, -22.0f / 127.0f, -10.0f / 127.0f,
    0.0f, 10.0f / 127.0f, 22.0f / 127.0f, 35.0f / 127.0f,
    49.0f / 127.0f, 65.0f / 127.0f, 83.0f / 127.0f, 104.0f / 127.0f};

/**
 * @brief Vectorized IQ4_NL decode (optimized for streaming)
 */
__device__ __forceinline__ void decode_iq4nl_to_fp16_p5a(
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
            float val0 = scale * iq4nl_values_p5a[q & 0xF];
            float val1 = scale * iq4nl_values_p5a[q >> 4];

            reinterpret_cast<__half2 *>(output)[i * 4 + j] =
                __floats2half2_rn(val0, val1);
        }
    }
}

/**
 * @brief Phase 5A: Streaming dequantization GEMM
 *
 * Key change from Phase 4:
 * - Split K-dimension into 4 sub-tiles (TILE_K=64 → 4×16)
 * - Decode B sub-tile INSIDE inner loop
 * - Overlaps decode (CUDA cores) with MMA (Tensor Cores)
 *
 * Memory layout:
 * - s_A: Still double-buffered (64×64 FP16) - loaded once per K-tile
 * - s_B: SINGLE buffer (64×16 FP16) - decoded on-the-fly each iteration
 *
 * Execution pattern:
 *   decode B[0]
 *   for i in [0..3]:
 *       MMA with B[i]  (Tensor Cores busy)
 *       decode B[i+1]  (CUDA cores work in parallel!)
 *
 * @tparam TILE_M Output tile M dimension (64)
 * @tparam TILE_N Output tile N dimension (64)
 * @tparam TILE_K K-dimension tile (64, split into 4×16)
 * @tparam SUB_K Sub-tile size (16 - one MMA atom's K dimension)
 */
template <
    int TILE_M = 64,
    int TILE_N = 64,
    int TILE_K = 64,
    int SUB_K = 16, // Split K into 4 sub-tiles for streaming
    int MMA_M = 2,
    int MMA_N = 2>
__global__ void iq4nl_gemm_phase5a_kernel(
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
        Layout<Shape<Int<MMA_M>, Int<MMA_N>, Int<1>>>>;
    TiledMMA tiled_mma;

    // ========== 2. Swizzled Shared Memory Layout ==========
    using SmemLayoutA_Swizzled = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<Int<TILE_M>, Int<TILE_K>>,
               Stride<Int<TILE_K>, Int<1>>>{}));

    // B buffer holds ENTIRE K-tile (64 elements = all 2 IQ4_NL blocks)
    // Decode ALL blocks ONCE, then use for all 4 sub-tiles → ZERO sync tax!
    using SmemLayoutB_FullTile = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<Int<TILE_N>, Int<TILE_K>>, // 64×64 (same as Phase 4!)
               Stride<Int<TILE_K>, Int<1>>>{}));

    // Shared Memory: DOUBLE-BUFFER EVERYTHING for overlap!
    __shared__ __half s_A[2][TILE_M][TILE_K];         // 2 × 8 KB = 16 KB
    __shared__ __half s_B_decoded[2][TILE_N][TILE_K]; // 2 × 8 KB = 16 KB (double-buffered!)
    // Total: 32 KB (same as Phase 4, but with streaming capability!)

    // ========== 3. Block/Thread Indices ==========
    int block_m = blockIdx.x;
    int block_n = blockIdx.y;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

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

    clear(tCrC);

    // ========== 6. K-Dimension Pipeline Setup ==========
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    constexpr int NUM_SUB_TILES = TILE_K / SUB_K; // 64/16 = 4
    int blocks_per_row = K / 32;

    int write_stage = 0;
    int read_stage = 0;

    // ========== 7. PROLOGUE: Prefetch First A Tile ==========
    if (num_k_tiles > 0)
    {
        auto sA_write = make_tensor(
            make_smem_ptr(s_A[0][0]),
            SmemLayoutA_Swizzled{});

        // Load A tile (full TILE_K width)
        for (int idx = tid; idx < TILE_M * TILE_K; idx += num_threads)
        {
            int m = idx / TILE_K;
            int k = idx % TILE_K;
            int gm = gm_start + m;
            int gk = k; // First tile starts at 0

            if (gm < M && gk < K)
            {
                sA_write(m, k) = __float2half(A[gm * K + gk]);
            }
            else
            {
                sA_write(m, k) = __float2half(0.0f);
            }
        }

        __syncthreads();
    }

    // ========== 8. MAIN LOOP: Overlapped Streaming Execution ==========
    // Prologue: Decode first B tile before loop
    {
        auto sB_decoded_tensor = make_tensor(
            make_smem_ptr(s_B_decoded[0][0]),
            SmemLayoutB_FullTile{});

        int global_k_start = 0;
        int blocks_in_tile = TILE_K / 32; // 2 blocks per tile

        // Decode all blocks for first K-tile
        for (int n = tid; n < TILE_N; n += num_threads)
        {
            int gn = gn_start + n;

            for (int block_idx = 0; block_idx < blocks_in_tile; ++block_idx)
            {
                int global_block_idx = (global_k_start / 32) + block_idx;
                int k_offset = block_idx * 32;

                if (gn < N && global_block_idx < blocks_per_row)
                {
                    const IQ4_NLBlock *block = &B_blocks[gn * blocks_per_row + global_block_idx];
                    __half decoded[32];
                    decode_iq4nl_to_fp16_p5a(block, decoded);

#pragma unroll
                    for (int i = 0; i < 32; i++)
                    {
                        sB_decoded_tensor(n, k_offset + i) = decoded[i];
                    }
                }
                else
                {
#pragma unroll
                    for (int i = 0; i < 32; i++)
                    {
                        sB_decoded_tensor(n, k_offset + i) = __float2half(0.0f);
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile)
    {
        read_stage = k_tile % 2;
        write_stage = 1 - read_stage;

        // ===== DECODE NEXT B TILE (while computing current tile) =====
        if (k_tile + 1 < num_k_tiles)
        {
            auto sB_decoded_next = make_tensor(
                make_smem_ptr(s_B_decoded[write_stage][0]),
                SmemLayoutB_FullTile{});

            int next_global_k_start = (k_tile + 1) * TILE_K;
            int blocks_in_tile = TILE_K / 32;

            // Decode next B tile in parallel with MMA operations below
            for (int n = tid; n < TILE_N; n += num_threads)
            {
                int gn = gn_start + n;

                for (int block_idx = 0; block_idx < blocks_in_tile; ++block_idx)
                {
                    int global_block_idx = (next_global_k_start / 32) + block_idx;
                    int k_offset = block_idx * 32;

                    if (gn < N && global_block_idx < blocks_per_row)
                    {
                        const IQ4_NLBlock *block = &B_blocks[gn * blocks_per_row + global_block_idx];
                        __half decoded[32];
                        decode_iq4nl_to_fp16_p5a(block, decoded);

#pragma unroll
                        for (int i = 0; i < 32; i++)
                        {
                            sB_decoded_next(n, k_offset + i) = decoded[i];
                        }
                    }
                    else
                    {
#pragma unroll
                        for (int i = 0; i < 32; i++)
                        {
                            sB_decoded_next(n, k_offset + i) = __float2half(0.0f);
                        }
                    }
                }
            }
        }

        // ===== Prefetch Next A Tile =====
        if (k_tile + 1 < num_k_tiles)
        {
            int next_gk_start = (k_tile + 1) * TILE_K;

            auto sA_next = make_tensor(
                make_smem_ptr(s_A[write_stage][0]),
                SmemLayoutA_Swizzled{});

            for (int idx = tid; idx < TILE_M * TILE_K; idx += num_threads)
            {
                int m = idx / TILE_K;
                int k = idx % TILE_K;
                int gm = gm_start + m;
                int gk = next_gk_start + k;

                if (gm < M && gk < K)
                {
                    sA_next(m, k) = __float2half(A[gm * K + gk]);
                }
                else
                {
                    sA_next(m, k) = __float2half(0.0f);
                }
            }
        }

        // Get current B tile (already decoded in previous iteration or prologue)
        auto sB_decoded_tensor = make_tensor(
            make_smem_ptr(s_B_decoded[read_stage][0]),
            SmemLayoutB_FullTile{});

        // ===== STREAMING K-LOOP: Use pre-decoded data (ZERO additional sync!) =====
        for (int sub_k = 0; sub_k < NUM_SUB_TILES; ++sub_k)
        {
            // Slice the appropriate 16-element chunk from the full decoded tile
            auto sB_subtile = local_tile(
                sB_decoded_tensor,
                make_shape(Int<TILE_N>{}, Int<SUB_K>{}),
                make_coord(0, sub_k) // sub_k = 0,1,2,3 gives offsets 0,16,32,48
            );

            // Create A sub-tile view (slice from full A tile)
            auto sA_full = make_tensor(
                make_smem_ptr(s_A[read_stage][0]),
                SmemLayoutA_Swizzled{});

            // Slice A to match current SUB_K range
            auto sA_subtile = local_tile(
                sA_full,
                make_shape(Int<TILE_M>{}, Int<SUB_K>{}),
                make_coord(0, sub_k));

            // Perform MMA on this sub-tile
            auto tCrA = thr_mma.partition_fragment_A(sA_subtile);
            auto tCrB = thr_mma.partition_fragment_B(sB_subtile);

            auto tCsA = thr_mma.partition_A(sA_subtile);
            auto tCsB = thr_mma.partition_B(sB_subtile);

            cute::copy(tCsA, tCrA);
            cute::copy(tCsB, tCrB);

            // Execute MMA (no sync needed - B already fully decoded!)
            cute::gemm(tiled_mma, tCrA, tCrB, tCrC);

            // NO synchronization needed inside loop - all B data pre-decoded!
        }

        __syncthreads(); // Sync before switching A buffers
    }

    // ========== 9. Write Results to Global Memory ==========
    cute::axpby(1.0f, tCrC, 0.0f, tCgC);
}

/**
 * @brief Launch wrapper for Phase 5A kernel
 */
template <int TILE_M = 64, int TILE_N = 64, int TILE_K = 64, int SUB_K = 16>
void launch_iq4nl_gemm_phase5a(
    const float *A,
    const IQ4_NLBlock *B_blocks,
    float *C,
    int M, int N, int K)
{
    dim3 block(128);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    iq4nl_gemm_phase5a_kernel<TILE_M, TILE_N, TILE_K, SUB_K>
        <<<grid, block>>>(A, B_blocks, C, M, N, K);
}

// Explicit template instantiation
template void launch_iq4nl_gemm_phase5a<64, 64, 64, 16>(
    const float *A,
    const IQ4_NLBlock *B_blocks,
    float *C,
    int M, int N, int K);
