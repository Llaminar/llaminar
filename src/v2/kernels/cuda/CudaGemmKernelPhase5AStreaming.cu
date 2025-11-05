#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/gemm.hpp>
#include "IQ4_NL.h"

using namespace cute;

// ============================================================
// Phase 5A: On-Demand B Dequant (Streaming Optimization)
// ============================================================
//
// Simplest streaming approach: Decode B in the inner K-loop
//
// Phase 4 approach:
//   for k_tile:
//     load_A(); load_B(); sync()  ← All dequant upfront
//     for ki in K_tile:
//       MMA(A[ki], B[ki])
//
// Phase 5A approach:
//   for k_tile:
//     load_A(); sync()  ← Only A upfront
//     for ki in K_tile:
//       decode_B(ki)  ← Decode on-demand (overlaps with MMA)
//       MMA(A[ki], B[ki])
//
// Benefit: Instruction-level parallelism between decode and MMA
// Hardware can execute decode (CUDA cores) while MMA runs (Tensor Cores)
//
// Expected: +10-15% from better resource utilization
// ============================================================

__constant__ float iq4nl_values_p5a[16] = {
    -127.0f / 127.0f, -104.0f / 127.0f, -83.0f / 127.0f, -65.0f / 127.0f,
    -49.0f / 127.0f, -35.0f / 127.0f, -22.0f / 127.0f, -10.0f / 127.0f,
    1.0f / 127.0f, 13.0f / 127.0f, 25.0f / 127.0f, 38.0f / 127.0f,
    53.0f / 127.0f, 69.0f / 127.0f, 89.0f / 127.0f, 113.0f / 127.0f};

/**
 * @brief Decode single IQ4_NL block to shared memory (vectorized)
 */
__device__ __forceinline__ void decode_iq4nl_block_p5a(
    const IQ4_NLBlock *block,
    __half *output)
{
    const float scale = block->scale;
    const uint8_t *quants = block->quants;

#pragma unroll
    for (int i = 0; i < 16; i++)
    {
        uint8_t q = quants[i];
        float val0 = scale * iq4nl_values_p5a[q & 0xF];
        float val1 = scale * iq4nl_values_p5a[q >> 4];
        output[i * 2] = __float2half(val0);
        output[i * 2 + 1] = __float2half(val1);
    }
}

/**
 * @brief Phase 5A: Streaming dequant with on-demand B decode
 *
 * Key change: Decode B during inner K-loop, not in prologue
 * This creates instruction-level overlap between decode and MMA
 */
template <
    int TILE_M = 64,
    int TILE_N = 64,
    int TILE_K = 64>
__global__ void iq4nl_gemm_phase5a_streaming(
    const float *__restrict__ A,
    const IQ4_NLBlock *__restrict__ B,
    float *__restrict__ C,
    int M, int N, int K)
{
    // ========== 1. MMA Setup ==========
    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    using tiled_mma = decltype(make_tiled_mma(
        mma_atom{},
        make_layout(Shape<_4, _8, _1>{}),
        make_tile(Layout<Shape<_1, _1, _1>>{})));

    tiled_mma mma;

    // ========== 2. Shared Memory (swizzled layouts) ==========
    using SmemLayoutA = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<Int<TILE_M>, Int<TILE_K>>, Stride<Int<TILE_K>, Int<1>>>{}));

    using SmemLayoutB = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<Int<TILE_N>, Int<TILE_K>>, Stride<Int<TILE_K>, Int<1>>>{}));

    __shared__ __half s_A[2][TILE_M][TILE_K]; // Double-buffered A
    __shared__ __half s_B[TILE_N][TILE_K];    // Single-buffered B (decoded on-demand!)

    // ========== 3. Block/Thread Setup ==========
    int tid = threadIdx.x;
    int tx = tid % 32;
    int ty = tid / 32;

    int gm_start = blockIdx.x * TILE_M;
    int gn_start = blockIdx.y * TILE_N;

    // ========== 4. MMA Partitioning ==========
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));
    auto cta_tiler = make_shape(Int<TILE_M>{}, Int<TILE_N>{}, Int<TILE_K>{});
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    auto thr_mma = mma.get_slice(tid);
    auto tCgC = thr_mma.partition_C(gC);
    auto tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    // ========== 5. Main K-tile Loop ==========
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Prologue: Load first A tile
    {
        auto sA_write = make_tensor(make_smem_ptr(s_A[0][0]), SmemLayoutA{});
        for (int k = tx; k < TILE_K; k += 32)
        {
            for (int m = ty; m < TILE_M; m += 4)
            {
                int gm = gm_start + m;
                int gk = k;
                if (gm < M && gk < K)
                {
                    sA_write(m, k) = __float2half(A[gm * K + gk]);
                }
                else
                {
                    sA_write(m, k) = __float2half(0.0f);
                }
            }
        }
    }
    __syncthreads();

    // Main loop
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++)
    {
        int k0 = k_tile * TILE_K;
        int read_stage = k_tile % 2;
        int write_stage = 1 - read_stage;

        // ===== NEW: Decode B on-demand during compute =====
        // Instead of pre-loading entire B tile, decode as we go
        // This creates overlap: decode B[ki+1] while MMA on B[ki]

        // Inner K-loop: 64 elements in chunks of 16 (MMA atom K-dim)
        for (int ki = 0; ki < TILE_K; ki += 16)
        {
            // Decode next B slice (32 elements = 1 IQ4_NL block)
            // Do this BEFORE MMA so hardware can overlap
            if (ki + 16 <= TILE_K)
            {
                auto sB_write = make_tensor(make_smem_ptr(s_B[0][ki]), SmemLayoutB{});

                for (int n = ty * 32 + tx; n < TILE_N; n += 128)
                {
                    int gn = gn_start + n;
                    if (gn < N)
                    {
                        for (int local_k = 0; local_k < 16; local_k += 32)
                        {
                            int gk = k0 + ki + local_k;
                            if (gk < K)
                            {
                                const IQ4_NLBlock *block = &B[gn * (K / 32) + gk / 32];
                                __half temp[32];
                                decode_iq4nl_block_p5a(block, temp);

// Copy decoded values to shared memory
#pragma unroll
                                for (int idx = 0; idx < 32 && local_k + idx < 16; idx++)
                                {
                                    s_B[n][ki + local_k + idx] = temp[idx];
                                }
                            }
                        }
                    }
                }
            }
            __syncthreads();

            // Partition A and B for MMA
            auto sA_tensor = make_tensor(make_smem_ptr(s_A[read_stage][0]), SmemLayoutA{});
            auto sB_tensor = make_tensor(make_smem_ptr(s_B[0][ki]), SmemLayoutB{});

            auto tCsA = thr_mma.partition_A(sA_tensor);
            auto tCsB = thr_mma.partition_B(sB_tensor);

            auto tCrA = make_fragment_like(tCsA);
            auto tCrB = make_fragment_like(tCsB);

            cute::copy(tCsA, tCrA);
            cute::copy(tCsB, tCrB);

            // MMA (overlaps with next decode!)
            cute::gemm(mma, tCrA, tCrB, tCrC);
            __syncthreads();
        }

        // Prefetch next A tile (same as Phase 4)
        if (k_tile + 1 < num_k_tiles)
        {
            auto sA_next = make_tensor(make_smem_ptr(s_A[write_stage][0]), SmemLayoutA{});
            int k_next = (k_tile + 1) * TILE_K;

            for (int k = tx; k < TILE_K; k += 32)
            {
                for (int m = ty; m < TILE_M; m += 4)
                {
                    int gm = gm_start + m;
                    int gk = k_next + k;
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
        }
        __syncthreads();
    }

    // ========== 6. Write Results ==========
    for (int i = 0; i < size(tCgC); i++)
    {
        tCgC(i) = tCrC(i);
    }
}

void launch_iq4nl_gemm_phase5a_streaming(
    const float *A,
    const IQ4_NLBlock *B,
    float *C,
    int M, int N, int K)
{
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    constexpr int THREADS = 128;

    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    dim3 threads(THREADS);

    iq4nl_gemm_phase5a_streaming<TILE_M, TILE_N><<<grid, threads>>>(A, B, C, M, N, K);
}
