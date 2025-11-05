/**
 * @file CudaGemmKernelPhase4QuickWins.cu
 * @brief Phase 4: Swizzled shared memory (eliminate bank conflicts)
 *
 * Key optimization over Phase 3 Part 2:
 * - XOR swizzled shared memory layout using Swizzle<3, 3, 3>
 * - Formula: MBase=log2(8), BBits=log2(64)-3=3, SShift=log2(64)-3=3
 * - Assumes 128-bit (16-byte) vectorized access for FP16 (8 elements)
 * - Bank conflict free column access for 64-wide FP16 matrices
 *
 * Reference: https://leimao.github.io/blog/CuTe-Swizzle/
 *
 * Expected performance: 7.2 TFLOPS (M=1024) - +10% over Phase 3 Part 2's 6.56 TFLOPS
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
__constant__ float iq4nl_values_p4[16] = {
    -127.0f / 127.0f, -104.0f / 127.0f, -83.0f / 127.0f, -65.0f / 127.0f,
    -49.0f / 127.0f, -35.0f / 127.0f, -22.0f / 127.0f, -10.0f / 127.0f,
    0.0f, 10.0f / 127.0f, 22.0f / 127.0f, 35.0f / 127.0f,
    49.0f / 127.0f, 65.0f / 127.0f, 83.0f / 127.0f, 104.0f / 127.0f};

/**
 * @brief Vectorized IQ4_NL decode (optimized for cp.async pipelining)
 */
__device__ __forceinline__ void decode_iq4nl_to_fp16_p4(
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
            float val0 = scale * iq4nl_values_p4[q & 0xF];
            float val1 = scale * iq4nl_values_p4[q >> 4];

            reinterpret_cast<__half2 *>(output)[i * 4 + j] =
                __floats2half2_rn(val0, val1);
        }
    }
}

/**
 * @brief Phase 4: Pipelined GEMM with swizzled shared memory
 *
 * Optimization: Swizzle<3, 3, 3> for bank-conflict-free column access
 * - For FP16 (2 bytes) with 64-element rows (TILE_K=64)
 * - MBase = 3 (assumes 128-bit / 8 FP16 vectorization)
 * - BBits = 3 (log2(64) - 3)
 * - SShift = 3 (log2(64) - 3)
 *
 * Note: We still use manual loops (not cute::copy) because:
 * 1. A matrix needs FP32→FP16 conversion
 * 2. B matrix needs IQ4_NL→FP16 decode
 * 3. cute::copy() with swizzle requires custom Copy_Atoms (complex)
 *
 * Strategy: Apply swizzle only during shared→register copy (partition_fragment)
 * This is where bank conflicts actually matter (strided column access).
 *
 * @tparam TILE_M Output tile M dimension (64)
 * @tparam TILE_N Output tile N dimension (64)
 * @tparam TILE_K K-dimension tile (64)
 * @tparam MMA_M MMA atom layout M (2)
 * @tparam MMA_N MMA atom layout N (2)
 */
template <
    int TILE_M = 64,
    int TILE_N = 64,
    int TILE_K = 64,
    int MMA_M = 2,
    int MMA_N = 2>
__global__ void iq4nl_gemm_phase4_kernel(
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

    // ========== 2. Swizzled Shared Memory Layout ==========
    // Swizzle<3, 3, 3> for FP16 64-wide rows with 128-bit vectorization
    // XOR pattern: bits [3:5] ^= bits [0:2]
    // This distributes column accesses across banks to eliminate conflicts
    using SmemLayoutA_Swizzled = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<Int<TILE_M>, Int<TILE_K>>,
               Stride<Int<TILE_K>, Int<1>>>{}));

    using SmemLayoutB_Swizzled = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<Int<TILE_N>, Int<TILE_K>>,
               Stride<Int<TILE_K>, Int<1>>>{}));

    // Double-Buffered Shared Memory (static allocation)
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

        // Create swizzled shared memory tensor for A (buffer 0)
        auto sA_write = make_tensor(
            make_smem_ptr(s_A[0][0]),
            SmemLayoutA_Swizzled{});

        // Load A tile with swizzle: FP32→FP16 conversion via manual loop
        // (cute::copy() doesn't handle FP32→FP16 well, so we write through swizzled tensor)
        for (int idx = tid; idx < TILE_M * TILE_K; idx += num_threads)
        {
            int m = idx / TILE_K;
            int k = idx % TILE_K;
            int gm = gm_start + m;
            int gk = gk_start + k;

            if (gm < M && gk < K)
            {
                sA_write(m, k) = __float2half(A[gm * K + gk]); // Swizzled indexing!
            }
            else
            {
                sA_write(m, k) = __float2half(0.0f);
            }
        }

        // Create swizzled shared memory tensor for B (buffer 0)
        auto sB_write = make_tensor(
            make_smem_ptr(s_B[0][0]),
            SmemLayoutB_Swizzled{});

        // Load B tile with swizzle: IQ4_NL→FP16 decode via manual loop
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
                        __half decoded[32];
                        decode_iq4nl_to_fp16_p4(block, decoded);

// Write decoded values with swizzled indexing
#pragma unroll
                        for (int i = 0; i < 32; i++)
                        {
                            sB_write(n, kb * 32 + i) = decoded[i]; // Swizzled!
                        }
                    }
                    else
                    {
#pragma unroll
                        for (int i = 0; i < 32; i++)
                        {
                            sB_write(n, kb * 32 + i) = __float2half(0.0f);
                        }
                    }
                }
            }
            else
            {
                for (int k = 0; k < TILE_K; k++)
                {
                    sB_write(n, k) = __float2half(0.0f);
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

            // Create swizzled tensors for next buffer
            auto sA_next = make_tensor(
                make_smem_ptr(s_A[write_stage][0]),
                SmemLayoutA_Swizzled{});
            auto sB_next = make_tensor(
                make_smem_ptr(s_B[write_stage][0]),
                SmemLayoutB_Swizzled{});

            // Load A tile for next iteration with swizzled indexing
            for (int idx = tid; idx < TILE_M * TILE_K; idx += num_threads)
            {
                int m = idx / TILE_K;
                int k = idx % TILE_K;
                int gm = gm_start + m;
                int gk = next_gk_start + k;

                if (gm < M && gk < K)
                {
                    sA_next(m, k) = __float2half(A[gm * K + gk]); // Swizzled!
                }
                else
                {
                    sA_next(m, k) = __float2half(0.0f);
                }
            }

            // Load B tile for next iteration with swizzled indexing
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
                            __half decoded[32];
                            decode_iq4nl_to_fp16_p4(block, decoded);

#pragma unroll
                            for (int i = 0; i < 32; i++)
                            {
                                sB_next(n, kb * 32 + i) = decoded[i]; // Swizzled!
                            }
                        }
                        else
                        {
#pragma unroll
                            for (int i = 0; i < 32; i++)
                            {
                                sB_next(n, kb * 32 + i) = __float2half(0.0f);
                            }
                        }
                    }
                }
                else
                {
                    for (int k = 0; k < TILE_K; k++)
                    {
                        sB_next(n, k) = __float2half(0.0f);
                    }
                }
            }
        }

        // ===== STAGE 2: Compute Current Tile =====
        // Create SWIZZLED tensor views for current buffer
        auto sA_tensor = make_tensor(
            make_smem_ptr(s_A[read_stage][0]),
            SmemLayoutA_Swizzled{});

        auto sB_tensor = make_tensor(
            make_smem_ptr(s_B[read_stage][0]),
            SmemLayoutB_Swizzled{});

        // Create register fragments
        auto tCrA = thr_mma.partition_fragment_A(sA_tensor);
        auto tCrB = thr_mma.partition_fragment_B(sB_tensor);

        auto tCsA = thr_mma.partition_A(sA_tensor);
        auto tCsB = thr_mma.partition_B(sB_tensor);

        // Copy to registers (swizzled access!)
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
 * @brief Launch wrapper for Phase 4 kernel
 */
template <int TILE_M = 64, int TILE_N = 64, int TILE_K = 64>
void launch_iq4nl_gemm_phase4(
    const float *A,
    const IQ4_NLBlock *B_blocks,
    float *C,
    int M, int N, int K)
{
    dim3 block(128); // 128 threads per block
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    iq4nl_gemm_phase4_kernel<TILE_M, TILE_N, TILE_K>
        <<<grid, block>>>(A, B_blocks, C, M, N, K);
}

// Explicit template instantiation for default tile sizes
template void launch_iq4nl_gemm_phase4<64, 64, 64>(
    const float *A,
    const IQ4_NLBlock *B_blocks,
    float *C,
    int M, int N, int K);
