/**
 * @file CUDANativeVNNIPrefillKernels.cu
 * @brief Q4_0 native-vnni tensor-core prefill kernels.
 *
 * Weights stay in native payload form in VRAM. Each CTA loads compact Q4_0
 * payload blocks, decodes them into a transient shared-memory INT8 tile, then
 * reuses the existing mma.sync.m16n8k32 fragment path for compute.
 */

#include <cuda_runtime.h>

#include "CUDANativeVNNIDecodeCommon.cuh"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace
{
    using llaminar2::cuda_native_vnni::fp16_bits_to_float;

    constexpr int BK = 32;
    constexpr int Q40_PAYLOAD_BYTES = llaminar2::cuda_native_vnni::CodebookTraits<0>::payload_bytes;
    constexpr int SMEM_PAD = 16;
    constexpr int SMEM_STRIDE = BK + SMEM_PAD;
    constexpr int STAGES = 2;
    constexpr int MIN_GRID_BLOCKS = 128;

    // BK=64 constants (CUTLASS-standard K-tile for INT8 on SM80+)
    constexpr int BK64 = 64;
    constexpr int SMEM_PAD_64 = 16;
    constexpr int SMEM_STRIDE_64 = BK64 + SMEM_PAD_64; // 80, 16-byte aligned for ldmatrix

    enum class Q40PrefillShape
    {
        TpNarrow,
        Balanced,
        Wide,
    };

    static thread_local Q40PrefillShape g_last_q40_prefill_shape = Q40PrefillShape::Balanced;
    static std::atomic<uint64_t> g_q40_prefill_shape_counts[3] = {};

    // ─── Sweep-derived tile dispatch ───────────────────────────────────
    // Tile configurations validated via exhaustive sweep across 336 shapes,
    // 12 tile/warp configs, and 4 split_k values (15,984 measurements).
    // Overall penalty vs per-shape oracle: +2.2%.

    enum class TileId : uint8_t
    {
        T64x64_w2x2,   // BM=64  BN=64  WM=2 WN=2  (128 threads)
        T64x128_w2x2,  // BM=64  BN=128 WM=2 WN=2  (128 threads)
        T64x128_w4x2,  // BM=64  BN=128 WM=4 WN=2  (256 threads)
        T64x128_w2x4,  // BM=64  BN=128 WM=2 WN=4  (256 threads)
        T128x128_w4x2, // BM=128 BN=128 WM=4 WN=2  (256 threads)
        T128x128_w4x4, // BM=128 BN=128 WM=4 WN=4  (512 threads)
    };

    struct TileChoice
    {
        TileId tile;
        int split_k;
    };

    __device__ __forceinline__ int frag_row(int lane_id, int elem)
    {
        return (elem >> 1) * 8 + (lane_id >> 2);
    }

    __device__ __forceinline__ int frag_col(int lane_id, int elem)
    {
        return (lane_id & 3) * 2 + (elem & 1);
    }

    __device__ __forceinline__ void cp_async_cg_16_zfill_128(
        void *smem_dst, const void *gmem_src, int src_size)
    {
        const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::"r"(smem_addr), "l"(gmem_src), "r"(src_size) : "memory");
    }

    __device__ __forceinline__ void cp_async_commit()
    {
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    template <int N>
    __device__ __forceinline__ void cp_async_wait()
    {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
    }

    // Named barrier: sync a subset of threads within a CTA.
    // barrier_id: 0-15 (hardware barrier slot), thread_count: participating threads.
    // Has acquire/release memory ordering (like __syncthreads but scoped to participants).
    __device__ __forceinline__ void named_bar_sync(int barrier_id, int thread_count)
    {
        asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(thread_count) : "memory");
    }

    __device__ __forceinline__ void load_ldmatrix_a_m16n8k32(
        uint32_t frag[4],
        const int *smem_base,
        int stride_words,
        int lane_id)
    {
#if __CUDA_ARCH__ >= 800
        const int *xs = smem_base + (lane_id % 16) * stride_words + (lane_id / 16) * 4;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
                     : "l"(xs));
#else
        (void)frag;
        (void)smem_base;
        (void)stride_words;
        (void)lane_id;
#endif
    }

    __device__ __forceinline__ void load_ldmatrix_b_m16n8k32(
        uint32_t frag[2],
        const int *smem_base,
        int stride_words,
        int lane_id)
    {
#if __CUDA_ARCH__ >= 800
        const int *xs = smem_base + (lane_id % 8) * stride_words + ((lane_id / 8) * 4) % 8;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                     : "=r"(frag[0]), "=r"(frag[1])
                     : "l"(xs));
#else
        (void)frag;
        (void)smem_base;
        (void)stride_words;
        (void)lane_id;
#endif
    }

    __device__ __forceinline__ void mma_m16n8k32_s8(
        int32_t D[4],
        const uint32_t A[4],
        const uint32_t B[2])
    {
#if __CUDA_ARCH__ >= 800
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%0, %1, %2, %3};\n"
            : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
              "r"(B[0]), "r"(B[1]));
#else
        (void)D;
        (void)A;
        (void)B;
#endif
    }

    template <int BN, int BLOCK_SIZE>
    __device__ __forceinline__ void load_q40_tile_to_smem(
        const uint8_t *raw_payload,
        const uint16_t *scales_B,
        int8_t *decoded_tile,
        uint16_t *decoded_scales,
        int kb,
        int block_n,
        int N)
    {
        for (int col = threadIdx.x; col < BN; col += BLOCK_SIZE)
        {
            int32_t *dst_words = reinterpret_cast<int32_t *>(decoded_tile + col * SMEM_STRIDE);
            const int gcol = block_n + col;
            if (gcol < N)
            {
                int32_t packed_groups[8];
                const size_t linear = static_cast<size_t>(kb) * N + gcol;
                llaminar2::cuda_native_vnni::decode_groups_vec<0>(
                    raw_payload + static_cast<size_t>(col) * Q40_PAYLOAD_BYTES,
                    packed_groups);
#pragma unroll
                for (int g = 0; g < 8; ++g)
                    dst_words[g] = packed_groups[g];
                decoded_scales[col] = scales_B[linear];
            }
            else
            {
#pragma unroll
                for (int g = 0; g < 8; ++g)
                    dst_words[g] = 0;
                decoded_scales[col] = uint16_t{0};
            }
        }
    }

    template <int BN, int BLOCK_SIZE>
    __device__ __forceinline__ void decode_q40_raw_tile_to_smem(
        const uint8_t *raw_payload,
        int8_t *decoded_tile,
        int valid_cols)
    {
        for (int col = threadIdx.x; col < BN; col += BLOCK_SIZE)
        {
            int32_t *dst_words = reinterpret_cast<int32_t *>(decoded_tile + col * SMEM_STRIDE);
            if (col < valid_cols)
            {
                int32_t packed_groups[8];
                llaminar2::cuda_native_vnni::decode_groups_vec<0>(
                    raw_payload + static_cast<size_t>(col) * Q40_PAYLOAD_BYTES,
                    packed_groups);
#pragma unroll
                for (int g = 0; g < 8; ++g)
                    dst_words[g] = packed_groups[g];
            }
            else
            {
#pragma unroll
                for (int g = 0; g < 8; ++g)
                    dst_words[g] = 0;
            }
        }
    }

    template <int BM, int BN, int WARPS_M, int WARPS_N, int SPLIT_K = 1, bool SINGLE_PASS_MATERIALIZE = false>
    __global__ void q40NativeVNNITensorCoreKernel(
        const int8_t *__restrict__ A,
        const uint8_t *__restrict__ payload,
        const uint16_t *__restrict__ scales_B,
        float *__restrict__ C,
        const float *__restrict__ scales_A,
        const float *__restrict__ C_existing,
        const float *__restrict__ bias,
        int M,
        int N,
        int K,
        float alpha,
        float beta)
    {
#if __CUDA_ARCH__ >= 800
        constexpr int NUM_WARPS = WARPS_M * WARPS_N;
        constexpr int BLOCK_SIZE = NUM_WARPS * 32;
        constexpr int WARP_M = BM / WARPS_M;
        constexpr int WARP_N = BN / WARPS_N;
        constexpr int WM = WARP_M / 16;
        constexpr int WN = WARP_N / 8;
        constexpr int A_VEC_LOADS = BM * BK / 16;

        static_assert(BM % WARPS_M == 0 && BN % WARPS_N == 0);
        static_assert(WARP_M % 16 == 0);
        static_assert(WARP_N % 8 == 0);

        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 31;
        const int wr = warp_id / WARPS_N;
        const int wc = warp_id % WARPS_N;
        const int gid = lane_id >> 2;

        const int block_m = blockIdx.x * BM;
        const int block_n = blockIdx.y * BN;

        const int num_k_blocks_total = K / BK;
        int kb_begin = 0;
        int kb_end = num_k_blocks_total;
        if constexpr (SPLIT_K > 1)
        {
            const int blocks_per_part = (num_k_blocks_total + SPLIT_K - 1) / SPLIT_K;
            kb_begin = static_cast<int>(blockIdx.z) * blocks_per_part;
            kb_end = min(kb_begin + blocks_per_part, num_k_blocks_total);
        }
        const int num_k_iters = kb_end - kb_begin;
        if (num_k_iters <= 0)
            return;

        __shared__ int8_t smem_A[STAGES][BM * SMEM_STRIDE];
        __shared__ int8_t smem_B[STAGES][BN * SMEM_STRIDE];
        __shared__ uint8_t smem_B_raw[STAGES][BN * Q40_PAYLOAD_BYTES];
        __shared__ uint16_t smem_scales_B[STAGES][BN];

        float acc[WM][WN][4];
#pragma unroll
        for (int i = 0; i < WM; ++i)
#pragma unroll
            for (int j = 0; j < WN; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[i][j][e] = 0.0f;

        auto load_A_tile = [&](int stage, int kb) __attribute__((always_inline))
        {
#pragma unroll 4
            for (int idx = threadIdx.x; idx < A_VEC_LOADS; idx += BLOCK_SIZE)
            {
                const int linear = idx << 4;
                const int row = linear >> 5;
                const int col = linear & 31;
                const int grow = block_m + row;
                void *dst = &smem_A[stage][row * SMEM_STRIDE + col];
                const bool valid = grow < M;
                const void *src = valid
                                      ? static_cast<const void *>(&A[static_cast<size_t>(grow) * K + kb * BK + col])
                                      : static_cast<const void *>(A);
                cp_async_cg_16_zfill_128(dst, src, valid ? 16 : 0);
            }
        };

        auto load_B_payload_tile = [&](int stage, int kb) __attribute__((always_inline))
        {
#pragma unroll 2
            for (int col = threadIdx.x; col < BN; col += BLOCK_SIZE)
            {
                const int gcol = block_n + col;
                void *dst = &smem_B_raw[stage][col * Q40_PAYLOAD_BYTES];
                const bool valid = gcol < N;
                const void *src = valid
                                      ? static_cast<const void *>(&payload[(static_cast<size_t>(kb) * N + gcol) * Q40_PAYLOAD_BYTES])
                                      : static_cast<const void *>(payload);
                cp_async_cg_16_zfill_128(dst, src, valid ? Q40_PAYLOAD_BYTES : 0);
            }
        };

        auto materialize_B_stage = [&](int stage, int kb) __attribute__((always_inline))
        {
            if constexpr (SINGLE_PASS_MATERIALIZE)
            {
                load_q40_tile_to_smem<BN, BLOCK_SIZE>(
                    smem_B_raw[stage],
                    scales_B,
                    smem_B[stage],
                    smem_scales_B[stage],
                    kb,
                    block_n,
                    N);
            }
            else
            {
                const int valid_cols = max(0, min(BN, N - block_n));
                for (int col = threadIdx.x; col < BN; col += BLOCK_SIZE)
                {
                    const int gcol = block_n + col;
                    smem_scales_B[stage][col] = (gcol < N)
                                                    ? scales_B[static_cast<size_t>(kb) * N + gcol]
                                                    : uint16_t{0};
                }
                __syncthreads();
                decode_q40_raw_tile_to_smem<BN, BLOCK_SIZE>(smem_B_raw[stage], smem_B[stage], valid_cols);
            }
        };

        const bool is_interior_tile = (block_m + BM <= M) && (block_n + BN <= N);

        auto compute_k_block_interior = [&](int stage, int kb) __attribute__((always_inline))
        {
#pragma unroll
            for (int wi = 0; wi < WM; ++wi)
            {
                const int a_row_base = wr * WARP_M + wi * 16;

                uint32_t A_frag[4];
                load_ldmatrix_a_m16n8k32(
                    A_frag,
                    reinterpret_cast<const int *>(&smem_A[stage][a_row_base * SMEM_STRIDE]),
                    SMEM_STRIDE / 4,
                    lane_id);

                const int grow0 = block_m + a_row_base + gid;
                const float sa0 = scales_A[grow0 * num_k_blocks_total + kb];
                const float sa1 = scales_A[(grow0 + 8) * num_k_blocks_total + kb];

#pragma unroll
                for (int wj = 0; wj < WN; ++wj)
                {
                    const int b_col_base = wc * WARP_N + wj * 8;
                    const float sb0 = fp16_bits_to_float(smem_scales_B[stage][b_col_base + frag_col(lane_id, 0)]);
                    const float sb1 = fp16_bits_to_float(smem_scales_B[stage][b_col_base + frag_col(lane_id, 1)]);

                    const float cs00 = sa0 * sb0;
                    const float cs01 = sa0 * sb1;
                    const float cs10 = sa1 * sb0;
                    const float cs11 = sa1 * sb1;

                    uint32_t B_frag[2];
                    load_ldmatrix_b_m16n8k32(
                        B_frag,
                        reinterpret_cast<const int *>(&smem_B[stage][b_col_base * SMEM_STRIDE]),
                        SMEM_STRIDE / 4,
                        lane_id);

                    int32_t D_frag[4] = {0, 0, 0, 0};
                    mma_m16n8k32_s8(D_frag, A_frag, B_frag);

                    acc[wi][wj][0] += static_cast<float>(D_frag[0]) * cs00;
                    acc[wi][wj][1] += static_cast<float>(D_frag[1]) * cs01;
                    acc[wi][wj][2] += static_cast<float>(D_frag[2]) * cs10;
                    acc[wi][wj][3] += static_cast<float>(D_frag[3]) * cs11;
                }
            }
        };

        auto compute_k_block_border = [&](int stage, int kb) __attribute__((always_inline))
        {
#pragma unroll
            for (int wi = 0; wi < WM; ++wi)
            {
                const int a_row_base = wr * WARP_M + wi * 16;

                uint32_t A_frag[4];
                load_ldmatrix_a_m16n8k32(
                    A_frag,
                    reinterpret_cast<const int *>(&smem_A[stage][a_row_base * SMEM_STRIDE]),
                    SMEM_STRIDE / 4,
                    lane_id);

                const int grow0 = block_m + a_row_base + gid;
                const int grow1 = grow0 + 8;
                const float sa0 = (grow0 < M) ? scales_A[grow0 * num_k_blocks_total + kb] : 0.0f;
                const float sa1 = (grow1 < M) ? scales_A[grow1 * num_k_blocks_total + kb] : 0.0f;

#pragma unroll
                for (int wj = 0; wj < WN; ++wj)
                {
                    const int b_col_base = wc * WARP_N + wj * 8;
                    const int gc0 = block_n + b_col_base + frag_col(lane_id, 0);
                    const int gc1 = block_n + b_col_base + frag_col(lane_id, 1);
                    const float sb0 = (gc0 < N) ? fp16_bits_to_float(smem_scales_B[stage][gc0 - block_n]) : 0.0f;
                    const float sb1 = (gc1 < N) ? fp16_bits_to_float(smem_scales_B[stage][gc1 - block_n]) : 0.0f;

                    uint32_t B_frag[2];
                    load_ldmatrix_b_m16n8k32(
                        B_frag,
                        reinterpret_cast<const int *>(&smem_B[stage][b_col_base * SMEM_STRIDE]),
                        SMEM_STRIDE / 4,
                        lane_id);

                    int32_t D_frag[4] = {0, 0, 0, 0};
                    mma_m16n8k32_s8(D_frag, A_frag, B_frag);

#pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        const float sa = (e < 2) ? sa0 : sa1;
                        const float sb = (e & 1) ? sb1 : sb0;
                        acc[wi][wj][e] += static_cast<float>(D_frag[e]) * sa * sb;
                    }
                }
            }
        };

        // ── 2-stage async pipeline ──────────────────────────────────
        load_A_tile(0, kb_begin);
        load_B_payload_tile(0, kb_begin);
        cp_async_commit();
        cp_async_wait<0>();
        materialize_B_stage(0, kb_begin);
        __syncthreads();

        for (int ki = 0; ki < num_k_iters; ++ki)
        {
            const int stage = ki & 1;
            const int kb = kb_begin + ki;

            if (ki + 1 < num_k_iters)
            {
                load_A_tile(stage ^ 1, kb + 1);
                load_B_payload_tile(stage ^ 1, kb + 1);
                cp_async_commit();
            }

            if (is_interior_tile)
                compute_k_block_interior(stage, kb);
            else
                compute_k_block_border(stage, kb);

            if (ki + 1 < num_k_iters)
            {
                cp_async_wait<0>();
                materialize_B_stage(stage ^ 1, kb + 1);
                __syncthreads();
            }
        }

        const bool simple_epilogue = (beta == 0.0f) && (bias == nullptr);

#pragma unroll
        for (int wj = 0; wj < WN; ++wj)
        {
            const int tile_n = block_n + wc * WARP_N + wj * 8;
            const int gc0 = tile_n + frag_col(lane_id, 0);
            const int gc1 = tile_n + frag_col(lane_id, 1);
            const bool gc0_valid = gc0 < N;
            const bool gc1_valid = gc1 < N;
            const float bias0 = (bias && gc0_valid) ? bias[gc0] : 0.0f;
            const float bias1 = (bias && gc1_valid) ? bias[gc1] : 0.0f;

#pragma unroll
            for (int wi = 0; wi < WM; ++wi)
            {
                const int tile_m = block_m + wr * WARP_M + wi * 16;
                const bool interior = (tile_m + 15 < M) && (tile_n + 7 < N);

                if (interior && simple_epilogue)
                {
                    const int out_idx0 = (tile_m + frag_row(lane_id, 0)) * N + gc0;
                    const int out_idx1 = (tile_m + frag_row(lane_id, 1)) * N + gc1;
                    const int out_idx2 = (tile_m + frag_row(lane_id, 2)) * N + gc0;
                    const int out_idx3 = (tile_m + frag_row(lane_id, 3)) * N + gc1;

                    if constexpr (SPLIT_K > 1)
                    {
                        atomicAdd(&C[out_idx0], acc[wi][wj][0] * alpha);
                        atomicAdd(&C[out_idx1], acc[wi][wj][1] * alpha);
                        atomicAdd(&C[out_idx2], acc[wi][wj][2] * alpha);
                        atomicAdd(&C[out_idx3], acc[wi][wj][3] * alpha);
                    }
                    else
                    {
                        C[out_idx0] = acc[wi][wj][0] * alpha;
                        C[out_idx1] = acc[wi][wj][1] * alpha;
                        C[out_idx2] = acc[wi][wj][2] * alpha;
                        C[out_idx3] = acc[wi][wj][3] * alpha;
                    }
                    continue;
                }

#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    const int gr = tile_m + frag_row(lane_id, e);
                    const int gc = (e & 1) ? gc1 : gc0;

                    if (gr < M && gc < N)
                    {
                        const int out_idx = gr * N + gc;
                        float val = acc[wi][wj][e] * alpha;

                        if constexpr (SPLIT_K > 1)
                        {
                            if (blockIdx.z == 0)
                            {
                                if (beta != 0.0f && C_existing)
                                    val += beta * C_existing[out_idx];
                                if (bias)
                                    val += (e & 1) ? bias1 : bias0;
                            }
                            atomicAdd(&C[out_idx], val);
                        }
                        else
                        {
                            if (beta != 0.0f && C_existing)
                                val += beta * C_existing[out_idx];
                            if (bias)
                                val += (e & 1) ? bias1 : bias0;
                            C[out_idx] = val;
                        }
                    }
                }
            }
        }
#else
        (void)A;
        (void)payload;
        (void)scales_B;
        (void)C;
        (void)scales_A;
        (void)C_existing;
        (void)bias;
        (void)M;
        (void)N;
        (void)K;
        (void)alpha;
        (void)beta;
#endif
    }

    // =========================================================================
    // =========================================================================
    // BK=64 kernel: processes 2 quantized blocks per K-tile iteration.
    // Halves the number of K-loop iterations, halving barrier and decode
    // overhead while doubling MMA work per iteration.
    // Templatized on CODEBOOK_ID for any single-scale or asymmetric format.
    // For 16-byte payloads (CB 0,4,5): vectorized int4 load + specialized decode.
    // For other payloads: generic decode_groups<CB>() fallback.
    // Asymmetric formats (CB 5,7): adds min correction via sum_A.
    // B payload is decoded directly from global → registers → smem_B
    // (no staging buffer), saving 8KB smem and eliminating extra traffic.
    // =========================================================================
    // Occupancy hint: BM=128 (8 warps) → target 2 blocks/SM (≤128 regs/thread).
    //                 BM=64  (4 warps) → target 3 blocks/SM (≤170 regs/thread).
    // STAGES_=1: single-buffered (half smem, no load/compute overlap, higher occupancy).
    // STAGES_=2: double-buffered (overlaps decode(next) with compute(current)).
    template <uint8_t CODEBOOK_ID, int BM, int BN, int WARPS_M, int WARPS_N, int SPLIT_K = 1,
              int STAGES_ = 2,
              bool STREAM_K = false,
              int BLOCK_SIZE_ = WARPS_M * WARPS_N * 32,
              // Stream-K persistent loop needs more live registers for loop bookkeeping
              // (kbc, kbc_stop, tile bounds, fixup writes). Using MIN_BLOCKS_HINT=1
              // gives the compiler 128 regs/thread (vs 64 with hint=2), eliminating
              // local memory spilling that was causing 4.5x more DRAM traffic.
              int MIN_BLOCKS_HINT = STREAM_K ? 1
                                    : (STAGES_ == 1)
                                        ? ((BLOCK_SIZE_ >= 256) ? 3 : 4)
                                        : ((BLOCK_SIZE_ >= 256) ? 2 : 3)>
    __global__ __launch_bounds__(BLOCK_SIZE_, MIN_BLOCKS_HINT) void nativeVnniTC_BK64(
        const int8_t *__restrict__ A,
        const uint8_t *__restrict__ payload,
        const uint16_t *__restrict__ scales_B,
        const uint16_t *__restrict__ mins_B,
        const uint32_t *__restrict__ emins_B,
        float *__restrict__ C,
        const float *__restrict__ scales_A,
        const float *__restrict__ C_existing,
        const float *__restrict__ bias,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        float *__restrict__ tmp_fixup)
    {
#if __CUDA_ARCH__ >= 800
        constexpr int NUM_WARPS = WARPS_M * WARPS_N;
        constexpr int BLOCK_SIZE = NUM_WARPS * 32;
        constexpr int WARP_M = BM / WARPS_M;
        constexpr int WARP_N = BN / WARPS_N;
        constexpr int WM = WARP_M / 16;
        constexpr int WN = WARP_N / 8;
        constexpr int A_VEC_LOADS = BM * BK64 / 16;

        // Named barriers replace __syncthreads() when load/decode are warp-group-affine:
        // - A loads: row-group-affine when A_VEC_LOADS == BLOCK_SIZE (1 cp.async/thread)
        // - B decode: col-group-affine when BLOCK_SIZE >= 2*BN (split-thread path)
        // Each warp hits a row barrier (syncs WARPS_N warps) + col barrier (syncs WARPS_M warps)
        // instead of one full-CTA barrier, reducing barrier stall from 16→4 warp convergence.
        // PERF NOTE: benchmarking showed ~1% regression due to 2× barrier instruction overhead
        // outweighing the smaller sync group benefit. Disabled pending a single-barrier solution.
        constexpr bool USE_NAMED_BARRIERS = false; // (A_VEC_LOADS == BLOCK_SIZE) && (BLOCK_SIZE >= 2 * BN);

        static_assert(BM % WARPS_M == 0 && BN % WARPS_N == 0);
        static_assert(WARP_M % 16 == 0);
        static_assert(WARP_N % 8 == 0);
        static_assert(!STREAM_K || SPLIT_K == 1, "Stream-K and Split-K are mutually exclusive");

        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 31;
        const int wr = warp_id / WARPS_N;
        const int wc = warp_id % WARPS_N;
        const int gid = lane_id >> 2;

        int block_m = 0, block_n = 0;
        if constexpr (!STREAM_K)
        {
            block_m = blockIdx.x * BM;
            block_n = blockIdx.y * BN;
        }

        const int num_q40_blocks = K / 32;
        const int num_k_tiles_total = (num_q40_blocks + 1) / 2; // ceil: handles K%64!=0
        int kt_begin = 0;
        int kt_end = num_k_tiles_total;
        int num_k_iters = num_k_tiles_total;
        if constexpr (!STREAM_K)
        {
            if constexpr (SPLIT_K > 1)
            {
                const int tiles_per_part = (num_k_tiles_total + SPLIT_K - 1) / SPLIT_K;
                kt_begin = static_cast<int>(blockIdx.z) * tiles_per_part;
                kt_end = min(kt_begin + tiles_per_part, num_k_tiles_total);
            }
            num_k_iters = kt_end - kt_begin;
            if (num_k_iters <= 0)
                return;
        }

        // Compile-time traits for this codebook
        using Traits = llaminar2::cuda_native_vnni::CodebookTraits<CODEBOOK_ID>;
        constexpr bool IS_ASYMMETRIC = Traits::is_asymmetric;
        constexpr bool IS_DUAL_SCALE = Traits::is_dual_scale;
        constexpr bool IS_DUAL_SCALE_ASYM = Traits::is_dual_scale_asym;
        constexpr bool IS_IQ1_M = Traits::is_iq1_m;
        constexpr bool NEEDS_MINS = IS_ASYMMETRIC || IS_DUAL_SCALE;

        // Shared memory: no smem_B_raw staging buffer needed.
        // STAGES_=1 (single-buffered) halves smem → potential 3 blocks/SM.
        // STAGES_=2 (double-buffered) overlaps load/decode of next tile with compute.
        __shared__ int8_t smem_A[STAGES_][BM * SMEM_STRIDE_64];
        __shared__ int8_t smem_B[STAGES_][BN * SMEM_STRIDE_64];
        __shared__ uint16_t smem_scales_B[STAGES_][2 * BN];

        // Activation scales cached in smem to decouple from L1 data cache.
        // Stores 2 FP32 scale values per M-row per K-tile (one per Q4_0 block half).
        __shared__ float smem_sa[STAGES_][BM * 2];

        // Asymmetric formats need per-block mins; dual-scale formats need per-block scale_hi.
        // Both use the same smem_mins_B buffer (mins_B pointer carries scale_hi for dual-scale).
        [[maybe_unused]] __shared__ uint16_t smem_mins_B[STAGES_][NEEDS_MINS ? 2 * BN : 1];

        // Q2_K (dual_scale_asym) needs per-block emins: packed {min_lo_fp16, min_hi_fp16} as uint32_t
        [[maybe_unused]] __shared__ uint32_t smem_emins_B[STAGES_][IS_DUAL_SCALE_ASYM ? 2 * BN : 1];

        // IQ1_M needs per-block delta sign bytes from payload (qh0, qh1)
        [[maybe_unused]] __shared__ uint16_t smem_iq1m_qh[STAGES_][IS_IQ1_M ? 2 * BN : 1];

        float acc[WM][WN][4];
        auto zero_acc = [&]() __attribute__((always_inline))
        {
#pragma unroll
            for (int i = 0; i < WM; ++i)
#pragma unroll
                for (int j = 0; j < WN; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        acc[i][j][e] = 0.0f;
        };

        // Load A tile: BM × BK64 bytes via 16-byte async copies
        auto load_A_tile = [&](int stage, int kt) __attribute__((always_inline))
        {
#pragma unroll 2
            for (int idx = threadIdx.x; idx < A_VEC_LOADS; idx += BLOCK_SIZE)
            {
                const int linear = idx << 4;
                const int row = linear >> 6;
                const int col = linear & 63;
                const int grow = block_m + row;
                void *dst = &smem_A[stage][row * SMEM_STRIDE_64 + col];
                const bool valid = grow < M && (kt * BK64 + col + 16 <= K);
                const void *src = valid
                                      ? static_cast<const void *>(&A[static_cast<size_t>(grow) * K + kt * BK64 + col])
                                      : static_cast<const void *>(A);
                cp_async_cg_16_zfill_128(dst, src, valid ? 16 : 0);
            }
        };

        // Decode B: load payloads from global memory, decode to int8, write to smem.
        // Split-thread strategy: when BLOCK_SIZE >= 2*BN, first BN threads handle
        // K-block 0 and next BN threads handle K-block 1 in parallel. This doubles
        // warp-level parallelism during decode (100% vs 50% thread utilization).
        // Fallback: original per-column loop for smaller thread counts.
        auto decode_B_direct = [&](int stage, int kt) __attribute__((always_inline))
        {
            const int kb0 = kt * 2;
            const bool has_block1 = (kb0 + 1 < num_q40_blocks);

            if constexpr (BLOCK_SIZE >= 2 * BN)
            {
                // Split: threads [0, BN) → block 0; threads [BN, 2*BN) → block 1.
                // All warps participate, halving per-thread work.
                const int col = threadIdx.x & (BN - 1);  // BN is power of 2
                const int block_half = threadIdx.x / BN; // 0 or 1

                if (block_half >= 2)
                    return; // guard for BLOCK_SIZE > 2*BN

                const int gcol = block_n + col;
                int32_t *dst_words = reinterpret_cast<int32_t *>(&smem_B[stage][col * SMEM_STRIDE_64]);
                const int word_offset = block_half * 8;
                const int scale_slot = block_half;
                const int kb = kb0 + block_half;
                const bool has_this_block = (block_half == 0) || has_block1;

                if (gcol < N && has_this_block)
                {
                    constexpr int PAYLOAD_BYTES = Traits::payload_bytes;
                    const size_t base = (static_cast<size_t>(kb) * N + gcol) * PAYLOAD_BYTES;
                    const uint16_t s = scales_B[static_cast<size_t>(kb) * N + gcol];

                    [[maybe_unused]] uint16_t m = 0;
                    if constexpr (NEEDS_MINS)
                        m = mins_B[static_cast<size_t>(kb) * N + gcol];

                    [[maybe_unused]] uint32_t em = 0;
                    if constexpr (IS_DUAL_SCALE_ASYM)
                        em = emins_B[static_cast<size_t>(kb) * N + gcol];

                    [[maybe_unused]] uint16_t qh_packed = 0;
                    if constexpr (IS_IQ1_M)
                    {
                        qh_packed = static_cast<uint16_t>(payload[base + 4]) |
                                    (static_cast<uint16_t>(payload[base + 5]) << 8);
                    }

                    // ── Decode one Q-block ──────────────────────────────
                    if constexpr (PAYLOAD_BYTES == 16)
                    {
                        const int4 raw = *reinterpret_cast<const int4 *>(payload + base);
                        const uint32_t r0 = static_cast<uint32_t>(raw.x);
                        const uint32_t r1 = static_cast<uint32_t>(raw.y);
                        const uint32_t r2 = static_cast<uint32_t>(raw.z);
                        const uint32_t r3 = static_cast<uint32_t>(raw.w);
                        if constexpr (CODEBOOK_ID == 0)
                        {
                            *reinterpret_cast<int4 *>(&dst_words[word_offset]) = make_int4(
                                static_cast<int32_t>(__vsub4(r0 & 0x0F0F0F0Fu, 0x08080808u)),
                                static_cast<int32_t>(__vsub4(r1 & 0x0F0F0F0Fu, 0x08080808u)),
                                static_cast<int32_t>(__vsub4(r2 & 0x0F0F0F0Fu, 0x08080808u)),
                                static_cast<int32_t>(__vsub4(r3 & 0x0F0F0F0Fu, 0x08080808u)));
                            *reinterpret_cast<int4 *>(&dst_words[word_offset + 4]) = make_int4(
                                static_cast<int32_t>(__vsub4((r0 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                                static_cast<int32_t>(__vsub4((r1 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                                static_cast<int32_t>(__vsub4((r2 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                                static_cast<int32_t>(__vsub4((r3 >> 4) & 0x0F0F0F0Fu, 0x08080808u)));
                        }
                        else if constexpr (CODEBOOK_ID == 4)
                        {
                            using llaminar2::cuda_native_vnni::iq4nl_decode_word;
                            uint32_t lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
                            iq4nl_decode_word(r0, lo0, hi0);
                            iq4nl_decode_word(r1, lo1, hi1);
                            iq4nl_decode_word(r2, lo2, hi2);
                            iq4nl_decode_word(r3, lo3, hi3);
                            *reinterpret_cast<int4 *>(&dst_words[word_offset]) = make_int4(
                                static_cast<int32_t>(lo0), static_cast<int32_t>(lo1),
                                static_cast<int32_t>(lo2), static_cast<int32_t>(lo3));
                            *reinterpret_cast<int4 *>(&dst_words[word_offset + 4]) = make_int4(
                                static_cast<int32_t>(hi0), static_cast<int32_t>(hi1),
                                static_cast<int32_t>(hi2), static_cast<int32_t>(hi3));
                        }
                        else
                        {
                            int32_t groups[8];
                            llaminar2::cuda_native_vnni::decode_groups_vec<CODEBOOK_ID>(
                                payload + base, groups);
                            *reinterpret_cast<int4 *>(&dst_words[word_offset]) = make_int4(
                                groups[0], groups[1], groups[2], groups[3]);
                            *reinterpret_cast<int4 *>(&dst_words[word_offset + 4]) = make_int4(
                                groups[4], groups[5], groups[6], groups[7]);
                        }
                    }
                    else
                    {
                        int32_t groups[8];
                        llaminar2::cuda_native_vnni::decode_groups<CODEBOOK_ID>(
                            payload + base, groups);
                        *reinterpret_cast<int4 *>(&dst_words[word_offset]) = make_int4(
                            groups[0], groups[1], groups[2], groups[3]);
                        *reinterpret_cast<int4 *>(&dst_words[word_offset + 4]) = make_int4(
                            groups[4], groups[5], groups[6], groups[7]);
                    }

                    smem_scales_B[stage][2 * col + scale_slot] = s;
                    if constexpr (NEEDS_MINS)
                        smem_mins_B[stage][2 * col + scale_slot] = m;
                    if constexpr (IS_DUAL_SCALE_ASYM)
                        smem_emins_B[stage][2 * col + scale_slot] = em;
                    if constexpr (IS_IQ1_M)
                        smem_iq1m_qh[stage][2 * col + scale_slot] = qh_packed;
                }
                else if (gcol < N)
                {
                    // K-tail or out-of-bounds block: zero-fill this half
                    *reinterpret_cast<int4 *>(&dst_words[word_offset]) = make_int4(0, 0, 0, 0);
                    *reinterpret_cast<int4 *>(&dst_words[word_offset + 4]) = make_int4(0, 0, 0, 0);
                    smem_scales_B[stage][2 * col + scale_slot] = uint16_t{0};
                    if constexpr (NEEDS_MINS)
                        smem_mins_B[stage][2 * col + scale_slot] = uint16_t{0};
                    if constexpr (IS_DUAL_SCALE_ASYM)
                        smem_emins_B[stage][2 * col + scale_slot] = 0u;
                    if constexpr (IS_IQ1_M)
                        smem_iq1m_qh[stage][2 * col + scale_slot] = 0;
                }
                // gcol >= N: no smem write needed (border tiles use compute_k_tile_border)
            }
            else
            {
                // Fallback: original per-column loop (BLOCK_SIZE < 2*BN)
                for (int col = threadIdx.x; col < BN; col += BLOCK_SIZE)
                {
                    const int gcol = block_n + col;
                    int32_t *dst_words = reinterpret_cast<int32_t *>(&smem_B[stage][col * SMEM_STRIDE_64]);

                    if (gcol < N)
                    {
                        constexpr int PAYLOAD_BYTES = Traits::payload_bytes;
                        const size_t base0 = (static_cast<size_t>(kb0) * N + gcol) * PAYLOAD_BYTES;
                        const size_t base1 = has_block1
                                                 ? (static_cast<size_t>(kb0 + 1) * N + gcol) * PAYLOAD_BYTES
                                                 : size_t{0};

                        const uint16_t s0 = scales_B[static_cast<size_t>(kb0) * N + gcol];
                        const uint16_t s1 = has_block1
                                                ? scales_B[static_cast<size_t>(kb0 + 1) * N + gcol]
                                                : uint16_t{0};

                        [[maybe_unused]] uint16_t m0 = 0, m1 = 0;
                        if constexpr (NEEDS_MINS)
                        {
                            m0 = mins_B[static_cast<size_t>(kb0) * N + gcol];
                            if (has_block1)
                                m1 = mins_B[static_cast<size_t>(kb0 + 1) * N + gcol];
                        }

                        [[maybe_unused]] uint32_t em0 = 0, em1 = 0;
                        if constexpr (IS_DUAL_SCALE_ASYM)
                        {
                            em0 = emins_B[static_cast<size_t>(kb0) * N + gcol];
                            if (has_block1)
                                em1 = emins_B[static_cast<size_t>(kb0 + 1) * N + gcol];
                        }

                        [[maybe_unused]] uint16_t qh_packed0 = 0, qh_packed1 = 0;
                        if constexpr (IS_IQ1_M)
                        {
                            qh_packed0 = static_cast<uint16_t>(payload[base0 + 4]) |
                                         (static_cast<uint16_t>(payload[base0 + 5]) << 8);
                            if (has_block1)
                            {
                                qh_packed1 = static_cast<uint16_t>(payload[base1 + 4]) |
                                             (static_cast<uint16_t>(payload[base1 + 5]) << 8);
                            }
                        }

                        // ── Decode block 0 ──────────────────────────────────
                        if constexpr (PAYLOAD_BYTES == 16)
                        {
                            const int4 raw0 = *reinterpret_cast<const int4 *>(payload + base0);
                            const uint32_t r0 = static_cast<uint32_t>(raw0.x);
                            const uint32_t r1 = static_cast<uint32_t>(raw0.y);
                            const uint32_t r2 = static_cast<uint32_t>(raw0.z);
                            const uint32_t r3 = static_cast<uint32_t>(raw0.w);
                            if constexpr (CODEBOOK_ID == 0)
                            {
                                *reinterpret_cast<int4 *>(&dst_words[0]) = make_int4(
                                    static_cast<int32_t>(__vsub4(r0 & 0x0F0F0F0Fu, 0x08080808u)),
                                    static_cast<int32_t>(__vsub4(r1 & 0x0F0F0F0Fu, 0x08080808u)),
                                    static_cast<int32_t>(__vsub4(r2 & 0x0F0F0F0Fu, 0x08080808u)),
                                    static_cast<int32_t>(__vsub4(r3 & 0x0F0F0F0Fu, 0x08080808u)));
                                *reinterpret_cast<int4 *>(&dst_words[4]) = make_int4(
                                    static_cast<int32_t>(__vsub4((r0 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                                    static_cast<int32_t>(__vsub4((r1 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                                    static_cast<int32_t>(__vsub4((r2 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                                    static_cast<int32_t>(__vsub4((r3 >> 4) & 0x0F0F0F0Fu, 0x08080808u)));
                            }
                            else if constexpr (CODEBOOK_ID == 4)
                            {
                                using llaminar2::cuda_native_vnni::iq4nl_decode_word;
                                uint32_t lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
                                iq4nl_decode_word(r0, lo0, hi0);
                                iq4nl_decode_word(r1, lo1, hi1);
                                iq4nl_decode_word(r2, lo2, hi2);
                                iq4nl_decode_word(r3, lo3, hi3);
                                *reinterpret_cast<int4 *>(&dst_words[0]) = make_int4(
                                    static_cast<int32_t>(lo0), static_cast<int32_t>(lo1),
                                    static_cast<int32_t>(lo2), static_cast<int32_t>(lo3));
                                *reinterpret_cast<int4 *>(&dst_words[4]) = make_int4(
                                    static_cast<int32_t>(hi0), static_cast<int32_t>(hi1),
                                    static_cast<int32_t>(hi2), static_cast<int32_t>(hi3));
                            }
                            else
                            {
                                int32_t groups0[8];
                                llaminar2::cuda_native_vnni::decode_groups_vec<CODEBOOK_ID>(
                                    payload + base0, groups0);
                                *reinterpret_cast<int4 *>(&dst_words[0]) = make_int4(
                                    groups0[0], groups0[1], groups0[2], groups0[3]);
                                *reinterpret_cast<int4 *>(&dst_words[4]) = make_int4(
                                    groups0[4], groups0[5], groups0[6], groups0[7]);
                            }
                        }
                        else
                        {
                            int32_t groups0[8];
                            llaminar2::cuda_native_vnni::decode_groups<CODEBOOK_ID>(
                                payload + base0, groups0);
                            *reinterpret_cast<int4 *>(&dst_words[0]) = make_int4(
                                groups0[0], groups0[1], groups0[2], groups0[3]);
                            *reinterpret_cast<int4 *>(&dst_words[4]) = make_int4(
                                groups0[4], groups0[5], groups0[6], groups0[7]);
                        }

                        // ── Decode block 1 (K-tail: zero-fill if no second block) ──
                        if (has_block1)
                        {
                            if constexpr (PAYLOAD_BYTES == 16)
                            {
                                const int4 raw1 = *reinterpret_cast<const int4 *>(payload + base1);
                                const uint32_t r0 = static_cast<uint32_t>(raw1.x);
                                const uint32_t r1 = static_cast<uint32_t>(raw1.y);
                                const uint32_t r2 = static_cast<uint32_t>(raw1.z);
                                const uint32_t r3 = static_cast<uint32_t>(raw1.w);
                                if constexpr (CODEBOOK_ID == 0)
                                {
                                    *reinterpret_cast<int4 *>(&dst_words[8]) = make_int4(
                                        static_cast<int32_t>(__vsub4(r0 & 0x0F0F0F0Fu, 0x08080808u)),
                                        static_cast<int32_t>(__vsub4(r1 & 0x0F0F0F0Fu, 0x08080808u)),
                                        static_cast<int32_t>(__vsub4(r2 & 0x0F0F0F0Fu, 0x08080808u)),
                                        static_cast<int32_t>(__vsub4(r3 & 0x0F0F0F0Fu, 0x08080808u)));
                                    *reinterpret_cast<int4 *>(&dst_words[12]) = make_int4(
                                        static_cast<int32_t>(__vsub4((r0 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                                        static_cast<int32_t>(__vsub4((r1 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                                        static_cast<int32_t>(__vsub4((r2 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                                        static_cast<int32_t>(__vsub4((r3 >> 4) & 0x0F0F0F0Fu, 0x08080808u)));
                                }
                                else if constexpr (CODEBOOK_ID == 4)
                                {
                                    using llaminar2::cuda_native_vnni::iq4nl_decode_word;
                                    uint32_t lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
                                    iq4nl_decode_word(r0, lo0, hi0);
                                    iq4nl_decode_word(r1, lo1, hi1);
                                    iq4nl_decode_word(r2, lo2, hi2);
                                    iq4nl_decode_word(r3, lo3, hi3);
                                    *reinterpret_cast<int4 *>(&dst_words[8]) = make_int4(
                                        static_cast<int32_t>(lo0), static_cast<int32_t>(lo1),
                                        static_cast<int32_t>(lo2), static_cast<int32_t>(lo3));
                                    *reinterpret_cast<int4 *>(&dst_words[12]) = make_int4(
                                        static_cast<int32_t>(hi0), static_cast<int32_t>(hi1),
                                        static_cast<int32_t>(hi2), static_cast<int32_t>(hi3));
                                }
                                else
                                {
                                    int32_t groups1[8];
                                    llaminar2::cuda_native_vnni::decode_groups_vec<CODEBOOK_ID>(
                                        payload + base1, groups1);
                                    *reinterpret_cast<int4 *>(&dst_words[8]) = make_int4(
                                        groups1[0], groups1[1], groups1[2], groups1[3]);
                                    *reinterpret_cast<int4 *>(&dst_words[12]) = make_int4(
                                        groups1[4], groups1[5], groups1[6], groups1[7]);
                                }
                            }
                            else
                            {
                                int32_t groups1[8];
                                llaminar2::cuda_native_vnni::decode_groups<CODEBOOK_ID>(
                                    payload + base1, groups1);
                                *reinterpret_cast<int4 *>(&dst_words[8]) = make_int4(
                                    groups1[0], groups1[1], groups1[2], groups1[3]);
                                *reinterpret_cast<int4 *>(&dst_words[12]) = make_int4(
                                    groups1[4], groups1[5], groups1[6], groups1[7]);
                            }
                        } // has_block1
                        else
                        {
                            *reinterpret_cast<int4 *>(&dst_words[8]) = make_int4(0, 0, 0, 0);
                            *reinterpret_cast<int4 *>(&dst_words[12]) = make_int4(0, 0, 0, 0);
                        }

                        smem_scales_B[stage][2 * col] = s0;
                        smem_scales_B[stage][2 * col + 1] = s1;

                        if constexpr (NEEDS_MINS)
                        {
                            smem_mins_B[stage][2 * col] = m0;
                            smem_mins_B[stage][2 * col + 1] = m1;
                        }

                        if constexpr (IS_DUAL_SCALE_ASYM)
                        {
                            smem_emins_B[stage][2 * col] = em0;
                            smem_emins_B[stage][2 * col + 1] = em1;
                        }

                        if constexpr (IS_IQ1_M)
                        {
                            smem_iq1m_qh[stage][2 * col] = qh_packed0;
                            smem_iq1m_qh[stage][2 * col + 1] = qh_packed1;
                        }
                    }
                    else
                    {
#pragma unroll
                        for (int g = 0; g < 16; ++g)
                            dst_words[g] = 0;
                        smem_scales_B[stage][2 * col] = uint16_t{0};
                        smem_scales_B[stage][2 * col + 1] = uint16_t{0};
                        if constexpr (NEEDS_MINS)
                        {
                            smem_mins_B[stage][2 * col] = uint16_t{0};
                            smem_mins_B[stage][2 * col + 1] = uint16_t{0};
                        }
                        if constexpr (IS_DUAL_SCALE_ASYM)
                        {
                            smem_emins_B[stage][2 * col] = 0u;
                            smem_emins_B[stage][2 * col + 1] = 0u;
                        }
                        if constexpr (IS_IQ1_M)
                        {
                            smem_iq1m_qh[stage][2 * col] = 0;
                            smem_iq1m_qh[stage][2 * col + 1] = 0;
                        }
                    }
                }
            }
        };

        // Load activation scales for the current K-tile into smem.
        // BM rows × 2 halves (kb0, kb0+1) per K-tile → BM*2 floats.
        auto load_scales_A = [&](int stage, int kt) __attribute__((always_inline))
        {
            const int kb0 = kt * 2;
#pragma unroll 2
            for (int idx = threadIdx.x; idx < BM * 2; idx += BLOCK_SIZE)
            {
                const int row = idx >> 1;
                const int half = idx & 1;
                const int kb = kb0 + half;
                const int grow = block_m + row;
                smem_sa[stage][idx] = (grow < M && kb < num_q40_blocks)
                                          ? scales_A[grow * num_q40_blocks + kb]
                                          : 0.0f;
            }
        };

        const bool is_interior_tile = (!STREAM_K) ? ((block_m + BM <= M) && (block_n + BN <= N)) : false;
        // Mutable copy for stream-K tile loop (modified per-tile; standard path uses const above)
        bool is_interior_tile_mut = is_interior_tile;

        auto compute_k_tile_interior = [&](int stage, int kt) __attribute__((always_inline))
        {
            const int kb0 = kt * 2;
#pragma unroll
            for (int half = 0; half < 2; ++half)
            {
                const int kb = kb0 + half;
                if (kb >= num_q40_blocks)
                    break; // K-tail: second q-block doesn't exist
                const int k_offset = half * 32;
                const int scale_slot = half;

                // Pre-load ALL A scales for this half from smem
                float sa_pre[WM][2];
#pragma unroll
                for (int wi = 0; wi < WM; ++wi)
                {
                    const int local_row0 = wr * WARP_M + wi * 16 + gid;
                    sa_pre[wi][0] = smem_sa[stage][local_row0 * 2 + half];
                    sa_pre[wi][1] = smem_sa[stage][(local_row0 + 8) * 2 + half];
                }

                // Pre-load ALL B scales for this half into registers
                float sb_pre[WN][2];
                [[maybe_unused]] float mb_pre[WN][2];      // min_B (asymmetric) or scale_hi (dual-scale)
                [[maybe_unused]] float emin_lo_pre[WN][2]; // Q2_K: emins lo half
                [[maybe_unused]] float emin_hi_pre[WN][2]; // Q2_K: emins hi half
#pragma unroll
                for (int wj = 0; wj < WN; ++wj)
                {
                    const int b_col_base = wc * WARP_N + wj * 8;
                    sb_pre[wj][0] = fp16_bits_to_float(smem_scales_B[stage][2 * (b_col_base + frag_col(lane_id, 0)) + scale_slot]);
                    sb_pre[wj][1] = fp16_bits_to_float(smem_scales_B[stage][2 * (b_col_base + frag_col(lane_id, 1)) + scale_slot]);
                    if constexpr (NEEDS_MINS)
                    {
                        mb_pre[wj][0] = fp16_bits_to_float(smem_mins_B[stage][2 * (b_col_base + frag_col(lane_id, 0)) + scale_slot]);
                        mb_pre[wj][1] = fp16_bits_to_float(smem_mins_B[stage][2 * (b_col_base + frag_col(lane_id, 1)) + scale_slot]);
                    }
                    if constexpr (IS_DUAL_SCALE_ASYM)
                    {
                        const uint32_t em_c0 = smem_emins_B[stage][2 * (b_col_base + frag_col(lane_id, 0)) + scale_slot];
                        const uint32_t em_c1 = smem_emins_B[stage][2 * (b_col_base + frag_col(lane_id, 1)) + scale_slot];
                        emin_lo_pre[wj][0] = fp16_bits_to_float(static_cast<uint16_t>(em_c0));
                        emin_hi_pre[wj][0] = fp16_bits_to_float(static_cast<uint16_t>(em_c0 >> 16));
                        emin_lo_pre[wj][1] = fp16_bits_to_float(static_cast<uint16_t>(em_c1));
                        emin_hi_pre[wj][1] = fp16_bits_to_float(static_cast<uint16_t>(em_c1 >> 16));
                    }
                }

                // ── Pre-load ALL A fragments for this half ────────────
                // Decouples A ldmatrix from MMA, enabling the restructured
                // wj→wi loop that loads each B fragment ONCE instead of WM times.
                uint32_t A_frag_all[WM][4];
                [[maybe_unused]] float sum_A_row0_all[WM], sum_A_row1_all[WM];
                [[maybe_unused]] float sum_A_lo_row0_all[WM], sum_A_lo_row1_all[WM];
                [[maybe_unused]] float sum_A_hi_row0_all[WM], sum_A_hi_row1_all[WM];
                [[maybe_unused]] int sg0_r0_all[WM], sg1_r0_all[WM], sg2_r0_all[WM], sg3_r0_all[WM];
                [[maybe_unused]] int sg0_r1_all[WM], sg1_r1_all[WM], sg2_r1_all[WM], sg3_r1_all[WM];

#pragma unroll
                for (int wi = 0; wi < WM; ++wi)
                {
                    const int a_row_base = wr * WARP_M + wi * 16;
                    load_ldmatrix_a_m16n8k32(
                        A_frag_all[wi],
                        reinterpret_cast<const int *>(&smem_A[stage][a_row_base * SMEM_STRIDE_64 + k_offset]),
                        SMEM_STRIDE_64 / 4, lane_id);

                    if constexpr (IS_ASYMMETRIC)
                    {
                        const int8_t *row0_ptr = &smem_A[stage][(a_row_base + gid) * SMEM_STRIDE_64 + k_offset];
                        const int8_t *row1_ptr = &smem_A[stage][(a_row_base + gid + 8) * SMEM_STRIDE_64 + k_offset];
                        int32_t s0 = 0, s1 = 0;
#pragma unroll
                        for (int w = 0; w < 8; ++w)
                        {
                            s0 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row0_ptr)[w], s0);
                            s1 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row1_ptr)[w], s1);
                        }
                        sum_A_row0_all[wi] = static_cast<float>(s0);
                        sum_A_row1_all[wi] = static_cast<float>(s1);
                    }

                    if constexpr (IS_DUAL_SCALE_ASYM || IS_IQ1_M)
                    {
                        const int8_t *row0_ptr = &smem_A[stage][(a_row_base + gid) * SMEM_STRIDE_64 + k_offset];
                        const int8_t *row1_ptr = &smem_A[stage][(a_row_base + gid + 8) * SMEM_STRIDE_64 + k_offset];
                        int32_t slo0 = 0, slo1 = 0, shi0 = 0, shi1 = 0;
#pragma unroll
                        for (int w = 0; w < 4; ++w)
                        {
                            slo0 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row0_ptr)[w], slo0);
                            slo1 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row1_ptr)[w], slo1);
                        }
#pragma unroll
                        for (int w = 4; w < 8; ++w)
                        {
                            shi0 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row0_ptr)[w], shi0);
                            shi1 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row1_ptr)[w], shi1);
                        }
                        sum_A_lo_row0_all[wi] = static_cast<float>(slo0);
                        sum_A_lo_row1_all[wi] = static_cast<float>(slo1);
                        sum_A_hi_row0_all[wi] = static_cast<float>(shi0);
                        sum_A_hi_row1_all[wi] = static_cast<float>(shi1);
                    }

                    if constexpr (IS_IQ1_M)
                    {
                        const int8_t *row0_ptr = &smem_A[stage][(a_row_base + gid) * SMEM_STRIDE_64 + k_offset];
                        const int8_t *row1_ptr = &smem_A[stage][(a_row_base + gid + 8) * SMEM_STRIDE_64 + k_offset];
                        sg0_r0_all[wi] = sg1_r0_all[wi] = sg2_r0_all[wi] = sg3_r0_all[wi] = 0;
                        sg0_r1_all[wi] = sg1_r1_all[wi] = sg2_r1_all[wi] = sg3_r1_all[wi] = 0;
                        for (int w = 0; w < 2; ++w)
                        {
                            sg0_r0_all[wi] += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row0_ptr)[w]);
                            sg0_r1_all[wi] += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row1_ptr)[w]);
                        }
                        for (int w = 2; w < 4; ++w)
                        {
                            sg1_r0_all[wi] += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row0_ptr)[w]);
                            sg1_r1_all[wi] += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row1_ptr)[w]);
                        }
                        for (int w = 4; w < 6; ++w)
                        {
                            sg2_r0_all[wi] += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row0_ptr)[w]);
                            sg2_r1_all[wi] += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row1_ptr)[w]);
                        }
                        for (int w = 6; w < 8; ++w)
                        {
                            sg3_r0_all[wi] += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row0_ptr)[w]);
                            sg3_r1_all[wi] += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row1_ptr)[w]);
                        }
                    }
                }

                // ── Compute: wj (outer) → wi (inner) ─────────────────
                // B fragment loaded ONCE per wj, reused across all WM rows.
                // Halves B ldmatrix loads vs the original wi→wj order.
#pragma unroll
                for (int wj = 0; wj < WN; ++wj)
                {
                    uint32_t B_frag[2];
                    load_ldmatrix_b_m16n8k32(
                        B_frag,
                        reinterpret_cast<const int *>(&smem_B[stage][(wc * WARP_N + wj * 8) * SMEM_STRIDE_64 + k_offset]),
                        SMEM_STRIDE_64 / 4, lane_id);

#pragma unroll
                    for (int wi = 0; wi < WM; ++wi)
                    {
                        const float sa0 = sa_pre[wi][0];
                        const float sa1 = sa_pre[wi][1];

                        if constexpr (IS_DUAL_SCALE)
                        {
                            const uint32_t B_lo[2] = {B_frag[0], 0u};
                            const uint32_t B_hi[2] = {0u, B_frag[1]};
                            int32_t D_lo[4] = {0, 0, 0, 0};
                            int32_t D_hi[4] = {0, 0, 0, 0};
                            mma_m16n8k32_s8(D_lo, A_frag_all[wi], B_lo);
                            mma_m16n8k32_s8(D_hi, A_frag_all[wi], B_hi);

                            acc[wi][wj][0] += sa0 * (sb_pre[wj][0] * static_cast<float>(D_lo[0]) + mb_pre[wj][0] * static_cast<float>(D_hi[0]));
                            acc[wi][wj][1] += sa0 * (sb_pre[wj][1] * static_cast<float>(D_lo[1]) + mb_pre[wj][1] * static_cast<float>(D_hi[1]));
                            acc[wi][wj][2] += sa1 * (sb_pre[wj][0] * static_cast<float>(D_lo[2]) + mb_pre[wj][0] * static_cast<float>(D_hi[2]));
                            acc[wi][wj][3] += sa1 * (sb_pre[wj][1] * static_cast<float>(D_lo[3]) + mb_pre[wj][1] * static_cast<float>(D_hi[3]));

                            if constexpr (IS_DUAL_SCALE_ASYM)
                            {
                                acc[wi][wj][0] += sa0 * (emin_lo_pre[wj][0] * sum_A_lo_row0_all[wi] + emin_hi_pre[wj][0] * sum_A_hi_row0_all[wi]);
                                acc[wi][wj][1] += sa0 * (emin_lo_pre[wj][1] * sum_A_lo_row0_all[wi] + emin_hi_pre[wj][1] * sum_A_hi_row0_all[wi]);
                                acc[wi][wj][2] += sa1 * (emin_lo_pre[wj][0] * sum_A_lo_row1_all[wi] + emin_hi_pre[wj][0] * sum_A_hi_row1_all[wi]);
                                acc[wi][wj][3] += sa1 * (emin_lo_pre[wj][1] * sum_A_lo_row1_all[wi] + emin_hi_pre[wj][1] * sum_A_hi_row1_all[wi]);
                            }

                            if constexpr (IS_IQ1_M)
                            {
                                constexpr float IQ1S_DELTA_VAL = 0.125f;
                                const int b_col_base = wc * WARP_N + wj * 8;
                                const uint16_t qh_c0 = smem_iq1m_qh[stage][2 * (b_col_base + frag_col(lane_id, 0)) + scale_slot];
                                const uint16_t qh_c1 = smem_iq1m_qh[stage][2 * (b_col_base + frag_col(lane_id, 1)) + scale_slot];

                                auto iq1m_corr = [&](uint16_t qh_packed, float sg0, float sg1, float sg2, float sg3,
                                                     float s_lo, float s_hi) -> float
                                {
                                    const uint8_t qh0 = static_cast<uint8_t>(qh_packed);
                                    const uint8_t qh1 = static_cast<uint8_t>(qh_packed >> 8);
                                    const float d0 = (qh0 & 0x08) ? -IQ1S_DELTA_VAL : IQ1S_DELTA_VAL;
                                    const float d1 = (qh0 & 0x80) ? -IQ1S_DELTA_VAL : IQ1S_DELTA_VAL;
                                    const float d2 = (qh1 & 0x08) ? -IQ1S_DELTA_VAL : IQ1S_DELTA_VAL;
                                    const float d3 = (qh1 & 0x80) ? -IQ1S_DELTA_VAL : IQ1S_DELTA_VAL;
                                    return (d0 * sg0 + d1 * sg1) * s_lo + (d2 * sg2 + d3 * sg3) * s_hi;
                                };

                                acc[wi][wj][0] += sa0 * iq1m_corr(qh_c0,
                                                                  static_cast<float>(sg0_r0_all[wi]), static_cast<float>(sg1_r0_all[wi]),
                                                                  static_cast<float>(sg2_r0_all[wi]), static_cast<float>(sg3_r0_all[wi]),
                                                                  sb_pre[wj][0], mb_pre[wj][0]);
                                acc[wi][wj][1] += sa0 * iq1m_corr(qh_c1,
                                                                  static_cast<float>(sg0_r0_all[wi]), static_cast<float>(sg1_r0_all[wi]),
                                                                  static_cast<float>(sg2_r0_all[wi]), static_cast<float>(sg3_r0_all[wi]),
                                                                  sb_pre[wj][1], mb_pre[wj][1]);
                                acc[wi][wj][2] += sa1 * iq1m_corr(qh_c0,
                                                                  static_cast<float>(sg0_r1_all[wi]), static_cast<float>(sg1_r1_all[wi]),
                                                                  static_cast<float>(sg2_r1_all[wi]), static_cast<float>(sg3_r1_all[wi]),
                                                                  sb_pre[wj][0], mb_pre[wj][0]);
                                acc[wi][wj][3] += sa1 * iq1m_corr(qh_c1,
                                                                  static_cast<float>(sg0_r1_all[wi]), static_cast<float>(sg1_r1_all[wi]),
                                                                  static_cast<float>(sg2_r1_all[wi]), static_cast<float>(sg3_r1_all[wi]),
                                                                  sb_pre[wj][1], mb_pre[wj][1]);
                            }
                        }
                        else
                        {
                            int32_t D[4] = {0, 0, 0, 0};
                            mma_m16n8k32_s8(D, A_frag_all[wi], B_frag);

                            if constexpr (IS_ASYMMETRIC)
                            {
                                const float sa0_sum0 = sa0 * sum_A_row0_all[wi];
                                const float sa1_sum1 = sa1 * sum_A_row1_all[wi];
                                acc[wi][wj][0] += static_cast<float>(D[0]) * sa0 * sb_pre[wj][0] + sa0_sum0 * mb_pre[wj][0];
                                acc[wi][wj][1] += static_cast<float>(D[1]) * sa0 * sb_pre[wj][1] + sa0_sum0 * mb_pre[wj][1];
                                acc[wi][wj][2] += static_cast<float>(D[2]) * sa1 * sb_pre[wj][0] + sa1_sum1 * mb_pre[wj][0];
                                acc[wi][wj][3] += static_cast<float>(D[3]) * sa1 * sb_pre[wj][1] + sa1_sum1 * mb_pre[wj][1];
                            }
                            else
                            {
                                const float cs00 = sa0 * sb_pre[wj][0];
                                const float cs01 = sa0 * sb_pre[wj][1];
                                const float cs10 = sa1 * sb_pre[wj][0];
                                const float cs11 = sa1 * sb_pre[wj][1];
                                acc[wi][wj][0] += static_cast<float>(D[0]) * cs00;
                                acc[wi][wj][1] += static_cast<float>(D[1]) * cs01;
                                acc[wi][wj][2] += static_cast<float>(D[2]) * cs10;
                                acc[wi][wj][3] += static_cast<float>(D[3]) * cs11;
                            }
                        }
                    }
                }
            }
        };

        auto compute_k_tile_border = [&](int stage, int kt) __attribute__((always_inline))
        {
            const int kb0 = kt * 2;
#pragma unroll
            for (int half = 0; half < 2; ++half)
            {
                const int kb = kb0 + half;
                if (kb >= num_q40_blocks)
                    break; // K-tail: second q-block doesn't exist
                const int k_offset = half * 32;
                const int scale_slot = half;

#pragma unroll
                for (int wi = 0; wi < WM; ++wi)
                {
                    const int a_row_base = wr * WARP_M + wi * 16;
                    uint32_t A_frag[4];
                    load_ldmatrix_a_m16n8k32(
                        A_frag,
                        reinterpret_cast<const int *>(&smem_A[stage][a_row_base * SMEM_STRIDE_64 + k_offset]),
                        SMEM_STRIDE_64 / 4, lane_id);

                    const int grow0 = block_m + a_row_base + gid;
                    const int grow1 = grow0 + 8;
                    const int local_row0 = a_row_base + gid;
                    const float sa0 = (grow0 < M) ? smem_sa[stage][local_row0 * 2 + half] : 0.0f;
                    const float sa1 = (grow1 < M) ? smem_sa[stage][(local_row0 + 8) * 2 + half] : 0.0f;

                    // sum_A for asymmetric correction (border variant with bounds check)
                    [[maybe_unused]] float sum_A_row0 = 0.0f, sum_A_row1 = 0.0f;
                    if constexpr (IS_ASYMMETRIC)
                    {
                        if (grow0 < M)
                        {
                            const int8_t *row0_ptr = &smem_A[stage][(a_row_base + gid) * SMEM_STRIDE_64 + k_offset];
                            int32_t s0 = 0;
#pragma unroll
                            for (int w = 0; w < 8; ++w)
                                s0 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row0_ptr)[w], s0);
                            sum_A_row0 = static_cast<float>(s0);
                        }
                        if (grow1 < M)
                        {
                            const int8_t *row1_ptr = &smem_A[stage][(a_row_base + gid + 8) * SMEM_STRIDE_64 + k_offset];
                            int32_t s1 = 0;
#pragma unroll
                            for (int w = 0; w < 8; ++w)
                                s1 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row1_ptr)[w], s1);
                            sum_A_row1 = static_cast<float>(s1);
                        }
                    }

                    // Split sums for dual_scale_asym (Q2_K) and IQ1_M
                    [[maybe_unused]] float sum_A_lo_row0 = 0.0f, sum_A_lo_row1 = 0.0f;
                    [[maybe_unused]] float sum_A_hi_row0 = 0.0f, sum_A_hi_row1 = 0.0f;
                    if constexpr (IS_DUAL_SCALE_ASYM || IS_IQ1_M)
                    {
                        if (grow0 < M)
                        {
                            const int8_t *row0_ptr = &smem_A[stage][(a_row_base + gid) * SMEM_STRIDE_64 + k_offset];
                            int32_t slo = 0, shi = 0;
                            for (int w = 0; w < 4; ++w)
                                slo = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row0_ptr)[w], slo);
                            for (int w = 4; w < 8; ++w)
                                shi = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row0_ptr)[w], shi);
                            sum_A_lo_row0 = static_cast<float>(slo);
                            sum_A_hi_row0 = static_cast<float>(shi);
                        }
                        if (grow1 < M)
                        {
                            const int8_t *row1_ptr = &smem_A[stage][(a_row_base + gid + 8) * SMEM_STRIDE_64 + k_offset];
                            int32_t slo = 0, shi = 0;
                            for (int w = 0; w < 4; ++w)
                                slo = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row1_ptr)[w], slo);
                            for (int w = 4; w < 8; ++w)
                                shi = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row1_ptr)[w], shi);
                            sum_A_lo_row1 = static_cast<float>(slo);
                            sum_A_hi_row1 = static_cast<float>(shi);
                        }
                    }

#pragma unroll
                    for (int wj = 0; wj < WN; ++wj)
                    {
                        const int b_col_base = wc * WARP_N + wj * 8;
                        const int lc0 = b_col_base + frag_col(lane_id, 0);
                        const int lc1 = b_col_base + frag_col(lane_id, 1);
                        const float sb0 = (block_n + lc0 < N)
                                              ? fp16_bits_to_float(smem_scales_B[stage][2 * lc0 + scale_slot])
                                              : 0.0f;
                        const float sb1 = (block_n + lc1 < N)
                                              ? fp16_bits_to_float(smem_scales_B[stage][2 * lc1 + scale_slot])
                                              : 0.0f;

                        [[maybe_unused]] float mb0 = 0.0f, mb1 = 0.0f;
                        if constexpr (NEEDS_MINS)
                        {
                            mb0 = (block_n + lc0 < N)
                                      ? fp16_bits_to_float(smem_mins_B[stage][2 * lc0 + scale_slot])
                                      : 0.0f;
                            mb1 = (block_n + lc1 < N)
                                      ? fp16_bits_to_float(smem_mins_B[stage][2 * lc1 + scale_slot])
                                      : 0.0f;
                        }

                        uint32_t B_frag[2];
                        load_ldmatrix_b_m16n8k32(
                            B_frag,
                            reinterpret_cast<const int *>(&smem_B[stage][b_col_base * SMEM_STRIDE_64 + k_offset]),
                            SMEM_STRIDE_64 / 4, lane_id);

                        if constexpr (IS_DUAL_SCALE)
                        {
                            const uint32_t B_lo[2] = {B_frag[0], 0u};
                            const uint32_t B_hi[2] = {0u, B_frag[1]};
                            int32_t D_lo[4] = {0, 0, 0, 0};
                            int32_t D_hi[4] = {0, 0, 0, 0};
                            mma_m16n8k32_s8(D_lo, A_frag, B_lo);
                            mma_m16n8k32_s8(D_hi, A_frag, B_hi);

#pragma unroll
                            for (int e = 0; e < 4; ++e)
                            {
                                const float sa = (e < 2) ? sa0 : sa1;
                                const float sb = (e & 1) ? sb1 : sb0; // scale_lo
                                const float mb = (e & 1) ? mb1 : mb0; // scale_hi
                                acc[wi][wj][e] += sa * (sb * static_cast<float>(D_lo[e]) + mb * static_cast<float>(D_hi[e]));
                            }

                            if constexpr (IS_DUAL_SCALE_ASYM)
                            {
                                [[maybe_unused]] float emlo0 = 0.0f, emhi0 = 0.0f, emlo1 = 0.0f, emhi1 = 0.0f;
                                if (block_n + lc0 < N)
                                {
                                    const uint32_t em = smem_emins_B[stage][2 * lc0 + scale_slot];
                                    emlo0 = fp16_bits_to_float(static_cast<uint16_t>(em));
                                    emhi0 = fp16_bits_to_float(static_cast<uint16_t>(em >> 16));
                                }
                                if (block_n + lc1 < N)
                                {
                                    const uint32_t em = smem_emins_B[stage][2 * lc1 + scale_slot];
                                    emlo1 = fp16_bits_to_float(static_cast<uint16_t>(em));
                                    emhi1 = fp16_bits_to_float(static_cast<uint16_t>(em >> 16));
                                }
#pragma unroll
                                for (int e = 0; e < 4; ++e)
                                {
                                    const float sa = (e < 2) ? sa0 : sa1;
                                    const float elo = (e & 1) ? emlo1 : emlo0;
                                    const float ehi = (e & 1) ? emhi1 : emhi0;
                                    const float slo = (e < 2) ? sum_A_lo_row0 : sum_A_lo_row1;
                                    const float shi = (e < 2) ? sum_A_hi_row0 : sum_A_hi_row1;
                                    acc[wi][wj][e] += sa * (elo * slo + ehi * shi);
                                }
                            }

                            if constexpr (IS_IQ1_M)
                            {
                                constexpr float IQ1S_DELTA_VAL = 0.125f;
                                // Sub-group sums: 4 groups of 8 A-elements each (bounds-checked)
                                int sg0_r0b = 0, sg1_r0b = 0, sg2_r0b = 0, sg3_r0b = 0;
                                int sg0_r1b = 0, sg1_r1b = 0, sg2_r1b = 0, sg3_r1b = 0;
                                if (grow0 < M)
                                {
                                    const int8_t *row0_ptr = &smem_A[stage][(a_row_base + gid) * SMEM_STRIDE_64 + k_offset];
                                    for (int w = 0; w < 2; ++w)
                                        sg0_r0b += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row0_ptr)[w]);
                                    for (int w = 2; w < 4; ++w)
                                        sg1_r0b += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row0_ptr)[w]);
                                    for (int w = 4; w < 6; ++w)
                                        sg2_r0b += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row0_ptr)[w]);
                                    for (int w = 6; w < 8; ++w)
                                        sg3_r0b += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row0_ptr)[w]);
                                }
                                if (grow1 < M)
                                {
                                    const int8_t *row1_ptr = &smem_A[stage][(a_row_base + gid + 8) * SMEM_STRIDE_64 + k_offset];
                                    for (int w = 0; w < 2; ++w)
                                        sg0_r1b += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row1_ptr)[w]);
                                    for (int w = 2; w < 4; ++w)
                                        sg1_r1b += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row1_ptr)[w]);
                                    for (int w = 4; w < 6; ++w)
                                        sg2_r1b += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row1_ptr)[w]);
                                    for (int w = 6; w < 8; ++w)
                                        sg3_r1b += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row1_ptr)[w]);
                                }

                                const uint16_t qh_c0 = (block_n + lc0 < N) ? smem_iq1m_qh[stage][2 * lc0 + scale_slot] : 0;
                                const uint16_t qh_c1 = (block_n + lc1 < N) ? smem_iq1m_qh[stage][2 * lc1 + scale_slot] : 0;

                                auto iq1m_corr = [&](uint16_t qh_packed, float s0, float s1, float s2, float s3,
                                                     float s_lo, float s_hi) -> float
                                {
                                    const uint8_t qh0 = static_cast<uint8_t>(qh_packed);
                                    const uint8_t qh1 = static_cast<uint8_t>(qh_packed >> 8);
                                    const float d0 = (qh0 & 0x08) ? -IQ1S_DELTA_VAL : IQ1S_DELTA_VAL;
                                    const float d1 = (qh0 & 0x80) ? -IQ1S_DELTA_VAL : IQ1S_DELTA_VAL;
                                    const float d2 = (qh1 & 0x08) ? -IQ1S_DELTA_VAL : IQ1S_DELTA_VAL;
                                    const float d3 = (qh1 & 0x80) ? -IQ1S_DELTA_VAL : IQ1S_DELTA_VAL;
                                    return (d0 * s0 + d1 * s1) * s_lo + (d2 * s2 + d3 * s3) * s_hi;
                                };

#pragma unroll
                                for (int e = 0; e < 4; ++e)
                                {
                                    const float sa = (e < 2) ? sa0 : sa1;
                                    const uint16_t qh = (e & 1) ? qh_c1 : qh_c0;
                                    const float sg0v = static_cast<float>((e < 2) ? sg0_r0b : sg0_r1b);
                                    const float sg1v = static_cast<float>((e < 2) ? sg1_r0b : sg1_r1b);
                                    const float sg2v = static_cast<float>((e < 2) ? sg2_r0b : sg2_r1b);
                                    const float sg3v = static_cast<float>((e < 2) ? sg3_r0b : sg3_r1b);
                                    const float sbl = (e & 1) ? sb1 : sb0;
                                    const float sbh = (e & 1) ? mb1 : mb0;
                                    acc[wi][wj][e] += sa * iq1m_corr(qh, sg0v, sg1v, sg2v, sg3v, sbl, sbh);
                                }
                            }
                        }
                        else
                        {
                            int32_t D[4] = {0, 0, 0, 0};
                            mma_m16n8k32_s8(D, A_frag, B_frag);

                            if constexpr (IS_ASYMMETRIC)
                            {
#pragma unroll
                                for (int e = 0; e < 4; ++e)
                                {
                                    const float sa = (e < 2) ? sa0 : sa1;
                                    const float sb = (e & 1) ? sb1 : sb0;
                                    const float mb = (e & 1) ? mb1 : mb0;
                                    const float sum_A = (e < 2) ? sum_A_row0 : sum_A_row1;
                                    acc[wi][wj][e] += static_cast<float>(D[e]) * sa * sb + sa * mb * sum_A;
                                }
                            }
                            else
                            {
#pragma unroll
                                for (int e = 0; e < 4; ++e)
                                {
                                    const float sa = (e < 2) ? sa0 : sa1;
                                    const float sb = (e & 1) ? sb1 : sb0;
                                    acc[wi][wj][e] += static_cast<float>(D[e]) * sa * sb;
                                }
                            }
                        }
                    }
                }
            }
        };

        // ── Dispatch: stream-K tile loop or standard single-tile ─────
        if constexpr (STREAM_K)
        {
            // -- Stream-K lambdas (defined here to avoid capture overhead on standard path) --
            auto zero_acc = [&]() __attribute__((always_inline))
            {
#pragma unroll
                for (int i = 0; i < WM; ++i)
#pragma unroll
                    for (int j = 0; j < WN; ++j)
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            acc[i][j][e] = 0.0f;
            };

            auto run_pipeline = [&]() __attribute__((always_inline))
            {
                if constexpr (STAGES_ == 1)
                {
                    for (int ki = 0; ki < num_k_iters; ++ki)
                    {
                        const int kt = kt_begin + ki;
                        load_A_tile(0, kt);
                        cp_async_commit();
                        decode_B_direct(0, kt);
                        load_scales_A(0, kt);
                        cp_async_wait<0>();
                        __syncthreads();
                        if (is_interior_tile_mut)
                            compute_k_tile_interior(0, kt);
                        else
                            compute_k_tile_border(0, kt);
                        if (ki + 1 < num_k_iters)
                            __syncthreads();
                    }
                }
                else
                {
                    load_A_tile(0, kt_begin);
                    cp_async_commit();
                    decode_B_direct(0, kt_begin);
                    load_scales_A(0, kt_begin);
                    cp_async_wait<0>();
                    __syncthreads();
                    for (int ki = 0; ki < num_k_iters; ++ki)
                    {
                        const int stage = ki & 1;
                        const int kt = kt_begin + ki;
                        if (ki + 1 < num_k_iters)
                        {
                            load_A_tile(stage ^ 1, kt + 1);
                            cp_async_commit();
                            decode_B_direct(stage ^ 1, kt + 1);
                            load_scales_A(stage ^ 1, kt + 1);
                        }
                        if (is_interior_tile_mut)
                            compute_k_tile_interior(stage, kt);
                        else
                            compute_k_tile_border(stage, kt);
                        if (ki + 1 < num_k_iters)
                        {
                            cp_async_wait<0>();
                            __syncthreads();
                        }
                    }
                }
            };

            auto write_epilogue = [&]() __attribute__((always_inline))
            {
                const bool simple_epilogue = (beta == 0.0f) && (bias == nullptr);
#pragma unroll
                for (int wj = 0; wj < WN; ++wj)
                {
                    const int tile_n = block_n + wc * WARP_N + wj * 8;
                    const int gc0 = tile_n + frag_col(lane_id, 0);
                    const int gc1 = tile_n + frag_col(lane_id, 1);
                    const bool gc0_valid = gc0 < N;
                    const bool gc1_valid = gc1 < N;
                    const float bias0 = (bias && gc0_valid) ? bias[gc0] : 0.0f;
                    const float bias1 = (bias && gc1_valid) ? bias[gc1] : 0.0f;
#pragma unroll
                    for (int wi = 0; wi < WM; ++wi)
                    {
                        const int tile_m = block_m + wr * WARP_M + wi * 16;
                        const bool interior = (tile_m + 15 < M) && (tile_n + 7 < N);
                        if (interior && simple_epilogue)
                        {
                            const int out_idx0 = (tile_m + frag_row(lane_id, 0)) * N + gc0;
                            const int out_idx1 = (tile_m + frag_row(lane_id, 1)) * N + gc1;
                            const int out_idx2 = (tile_m + frag_row(lane_id, 2)) * N + gc0;
                            const int out_idx3 = (tile_m + frag_row(lane_id, 3)) * N + gc1;
                            C[out_idx0] = acc[wi][wj][0] * alpha;
                            C[out_idx1] = acc[wi][wj][1] * alpha;
                            C[out_idx2] = acc[wi][wj][2] * alpha;
                            C[out_idx3] = acc[wi][wj][3] * alpha;
                            continue;
                        }
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                        {
                            const int gr = tile_m + frag_row(lane_id, e);
                            const int gc = (e & 1) ? gc1 : gc0;
                            if (gr < M && gc < N)
                            {
                                const int out_idx = gr * N + gc;
                                float val = acc[wi][wj][e] * alpha;
                                if (beta != 0.0f && C_existing)
                                    val += beta * C_existing[out_idx];
                                if (bias)
                                    val += (e & 1) ? bias1 : bias0;
                                C[out_idx] = val;
                            }
                        }
                    }
                }
            };

            auto write_to_fixup = [&]() __attribute__((always_inline))
            {
                float *fix = tmp_fixup + blockIdx.x * (BM * BN);
#pragma unroll
                for (int wj = 0; wj < WN; ++wj)
                {
                    const int fc0 = wc * WARP_N + wj * 8 + frag_col(lane_id, 0);
                    const int fc1 = wc * WARP_N + wj * 8 + frag_col(lane_id, 1);
#pragma unroll
                    for (int wi = 0; wi < WM; ++wi)
                    {
                        const int fr_base = wr * WARP_M + wi * 16;
                        fix[(fr_base + frag_row(lane_id, 0)) * BN + fc0] = acc[wi][wj][0] * alpha;
                        fix[(fr_base + frag_row(lane_id, 1)) * BN + fc1] = acc[wi][wj][1] * alpha;
                        fix[(fr_base + frag_row(lane_id, 2)) * BN + fc0] = acc[wi][wj][2] * alpha;
                        fix[(fr_base + frag_row(lane_id, 3)) * BN + fc1] = acc[wi][wj][3] * alpha;
                    }
                }
            };

            // Stream-K tile loop
            const int ntx = (N + BN - 1) / BN;
            const int nty = (M + BM - 1) / BM;
            const int num_k_tiles = num_k_tiles_total;
            const long long total_work = (long long)nty * ntx * num_k_tiles;
            long long kbc = (long long)blockIdx.x * total_work / gridDim.x;
            long long kbc_stop = (long long)(blockIdx.x + 1) * total_work / gridDim.x;

            while (kbc < kbc_stop)
            {
                const int tile_idx = static_cast<int>(kbc / num_k_tiles);
                const int local_kt_begin = static_cast<int>(kbc % num_k_tiles);
                const int local_kt_end = min(num_k_tiles,
                                             local_kt_begin + static_cast<int>(kbc_stop - kbc));

                const int ty = tile_idx / ntx;
                const int tx = tile_idx % ntx;
                block_m = ty * BM;
                block_n = tx * BN;
                kt_begin = local_kt_begin;
                kt_end = local_kt_end;
                num_k_iters = kt_end - kt_begin;
                is_interior_tile_mut = (block_m + BM <= M) && (block_n + BN <= N);

                const bool completes_tile = (kt_end == num_k_tiles);

                zero_acc();
                run_pipeline();

                if (completes_tile)
                    write_epilogue();
                else
                    write_to_fixup();

                __syncthreads(); // ensure smem safe before next tile

                kbc += (num_k_tiles - local_kt_begin);
            }
        }
        else
        {
            // ── Standard path: inline code matching baseline exactly ─────
            // No lambda wrapping — keeps variables const and avoids capture overhead.

#pragma unroll
            for (int i = 0; i < WM; ++i)
#pragma unroll
                for (int j = 0; j < WN; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        acc[i][j][e] = 0.0f;

            if constexpr (STAGES_ == 1)
            {
                // ── Single-buffered main loop ────────────────────────────────
                for (int ki = 0; ki < num_k_iters; ++ki)
                {
                    const int kt = kt_begin + ki;

                    load_A_tile(0, kt);
                    cp_async_commit();
                    decode_B_direct(0, kt); // runs while cp.async for A is in-flight
                    load_scales_A(0, kt);
                    cp_async_wait<0>();
                    __syncthreads(); // A load + B decode both complete

                    if (is_interior_tile)
                        compute_k_tile_interior(0, kt);
                    else
                        compute_k_tile_border(0, kt);

                    if (ki + 1 < num_k_iters)
                        __syncthreads(); // ensure all warps done before overwriting smem
                }
            }
            else
            {
                // ── Double-buffered pipeline (STAGES_=2) ─────────────────────
                load_A_tile(0, kt_begin);
                cp_async_commit();
                decode_B_direct(0, kt_begin);
                load_scales_A(0, kt_begin);
                cp_async_wait<0>();
                __syncthreads();

                for (int ki = 0; ki < num_k_iters; ++ki)
                {
                    const int stage = ki & 1;
                    const int kt = kt_begin + ki;

                    if (ki + 1 < num_k_iters)
                    {
                        load_A_tile(stage ^ 1, kt + 1);
                        cp_async_commit();
                        decode_B_direct(stage ^ 1, kt + 1);
                        load_scales_A(stage ^ 1, kt + 1);
                    }

                    if (is_interior_tile)
                        compute_k_tile_interior(stage, kt);
                    else
                        compute_k_tile_border(stage, kt);

                    if (ki + 1 < num_k_iters)
                    {
                        cp_async_wait<0>();
                        __syncthreads();
                    }
                }
            }

            // Epilogue: write accumulators to global memory
            const bool simple_epilogue = (beta == 0.0f) && (bias == nullptr);

#pragma unroll
            for (int wj = 0; wj < WN; ++wj)
            {
                const int tile_n = block_n + wc * WARP_N + wj * 8;
                const int gc0 = tile_n + frag_col(lane_id, 0);
                const int gc1 = tile_n + frag_col(lane_id, 1);
                const bool gc0_valid = gc0 < N;
                const bool gc1_valid = gc1 < N;
                const float bias0 = (bias && gc0_valid) ? bias[gc0] : 0.0f;
                const float bias1 = (bias && gc1_valid) ? bias[gc1] : 0.0f;

#pragma unroll
                for (int wi = 0; wi < WM; ++wi)
                {
                    const int tile_m = block_m + wr * WARP_M + wi * 16;
                    const bool interior = (tile_m + 15 < M) && (tile_n + 7 < N);

                    if (interior && simple_epilogue)
                    {
                        const int out_idx0 = (tile_m + frag_row(lane_id, 0)) * N + gc0;
                        const int out_idx1 = (tile_m + frag_row(lane_id, 1)) * N + gc1;
                        const int out_idx2 = (tile_m + frag_row(lane_id, 2)) * N + gc0;
                        const int out_idx3 = (tile_m + frag_row(lane_id, 3)) * N + gc1;

                        if constexpr (SPLIT_K > 1)
                        {
                            atomicAdd(&C[out_idx0], acc[wi][wj][0] * alpha);
                            atomicAdd(&C[out_idx1], acc[wi][wj][1] * alpha);
                            atomicAdd(&C[out_idx2], acc[wi][wj][2] * alpha);
                            atomicAdd(&C[out_idx3], acc[wi][wj][3] * alpha);
                        }
                        else
                        {
                            C[out_idx0] = acc[wi][wj][0] * alpha;
                            C[out_idx1] = acc[wi][wj][1] * alpha;
                            C[out_idx2] = acc[wi][wj][2] * alpha;
                            C[out_idx3] = acc[wi][wj][3] * alpha;
                        }
                        continue;
                    }

#pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        const int gr = tile_m + frag_row(lane_id, e);
                        const int gc = (e & 1) ? gc1 : gc0;

                        if (gr < M && gc < N)
                        {
                            const int out_idx = gr * N + gc;
                            float val = acc[wi][wj][e] * alpha;

                            if constexpr (SPLIT_K > 1)
                            {
                                if (blockIdx.z == 0)
                                {
                                    if (beta != 0.0f && C_existing)
                                        val += beta * C_existing[out_idx];
                                    if (bias)
                                        val += (e & 1) ? bias1 : bias0;
                                }
                                atomicAdd(&C[out_idx], val);
                            }
                            else
                            {
                                if (beta != 0.0f && C_existing)
                                    val += beta * C_existing[out_idx];
                                if (bias)
                                    val += (e & 1) ? bias1 : bias0;
                                C[out_idx] = val;
                            }
                        }
                    }
                }
            }
        }

#else
        (void)A;
        (void)payload;
        (void)scales_B;
        (void)C;
        (void)scales_A;
        (void)C_existing;
        (void)bias;
        (void)M;
        (void)N;
        (void)K;
        (void)alpha;
        (void)beta;
        (void)tmp_fixup;
#endif
    }

    // =========================================================================
    // BK=256 kernel: processes 8 quantized blocks per outer K-tile iteration.
    // Weights (B) loaded once for full K=256; activations (A) loaded in two
    // K=128 halves (llama.cpp-style). Per half, all A MMA fragments are
    // pre-loaded into registers, then the compute loop iterates
    // k→wj(loads B once)→wi(reuses pre-loaded A), halving B ldmatrix loads.
    //
    // smem layout (BM=128, BN=128):
    //   smem_A:        128 × 144 = 18,432 bytes  (activations K=128 half, INT8)
    //   smem_B:        128 × 272 = 34,816 bytes  (decoded Q4_0 weights, INT8)
    //   smem_scales_B: 8 × 128 × 2 = 2,048 bytes (FP16 weight scales)
    //   smem_sa:       128 × 8 × 4 = 4,096 bytes (FP32 activation scales)
    //   Total: ~59,392 bytes → 1 block/SM (requires >48KB smem opt-in)
    //
    // Q4_0 only (CODEBOOK_ID=0). For other codebook types, use BK=64.
    // =========================================================================
    // BK=256 / BK=128 constants
    constexpr int BK256 = 256;
    constexpr int BK256_PAD = 16;
    constexpr int BK256_STRIDE = BK256 + BK256_PAD; // 272, 16-byte aligned
    constexpr int BK128 = 128;
    constexpr int BK128_PAD = 16;
    constexpr int BK128_STRIDE = BK128 + BK128_PAD; // 144, 16-byte aligned

    template <int BM, int BN, int WARPS_M, int WARPS_N, int SPLIT_K = 1,
              int BLOCK_SIZE_ = WARPS_M * WARPS_N * 32>
    __global__ __launch_bounds__(BLOCK_SIZE_, 1) // 1 block/SM: max register budget
        void nativeVnniTC_BK256(
            const int8_t *__restrict__ A,
            const uint8_t *__restrict__ payload,
            const uint16_t *__restrict__ scales_B,
            float *__restrict__ C,
            const float *__restrict__ scales_A,
            const float *__restrict__ C_existing,
            const float *__restrict__ bias,
            int M,
            int N,
            int K,
            float alpha,
            float beta)
    {
#if __CUDA_ARCH__ >= 800
        constexpr int NUM_WARPS = WARPS_M * WARPS_N;
        constexpr int BLOCK_SIZE = NUM_WARPS * 32;
        constexpr int WARP_M = BM / WARPS_M;
        constexpr int WARP_N = BN / WARPS_N;
        constexpr int WM = WARP_M / 16;
        constexpr int WN = WARP_N / 8;
        constexpr int A_HALF_VEC_LOADS = BM * BK128 / 16;
        constexpr int Q40_PL = 16; // Q4_0 payload bytes per 32-element block

        static_assert(BM % WARPS_M == 0 && BN % WARPS_N == 0);
        static_assert(WARP_M % 16 == 0 && WARP_N % 8 == 0);

        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 31;
        const int wr = warp_id / WARPS_N;
        const int wc = warp_id % WARPS_N;
        const int gid = lane_id >> 2;

        const int block_m = blockIdx.x * BM;
        const int block_n = blockIdx.y * BN;

        const int num_q40_blocks = K / 32;
        const int num_outer_tiles = (num_q40_blocks + 7) / 8;
        int ot_begin = 0;
        int ot_end = num_outer_tiles;
        if constexpr (SPLIT_K > 1)
        {
            const int tiles_per_part = (num_outer_tiles + SPLIT_K - 1) / SPLIT_K;
            ot_begin = static_cast<int>(blockIdx.z) * tiles_per_part;
            ot_end = min(ot_begin + tiles_per_part, num_outer_tiles);
        }
        if (ot_end <= ot_begin)
            return;

        // Dynamic shared memory: exceeds 48KB static limit for BM=128
        // Layout: smem_A (K=128 half) | smem_B (K=256 full) | smem_scales_B | smem_sa
        extern __shared__ char smem_raw[];
        int8_t *smem_A = reinterpret_cast<int8_t *>(smem_raw);
        int8_t *smem_B = reinterpret_cast<int8_t *>(smem_raw + BM * BK128_STRIDE);
        constexpr int SCALES_B_OFFSET = BM * BK128_STRIDE + BN * BK256_STRIDE;
        uint16_t *smem_scales_B = reinterpret_cast<uint16_t *>(smem_raw + SCALES_B_OFFSET);
        constexpr int SA_OFFSET = SCALES_B_OFFSET + 8 * BN * sizeof(uint16_t);
        // Align smem_sa to 4-byte boundary (float alignment)
        constexpr int SA_ALIGNED = (SA_OFFSET + 3) & ~3;
        float *smem_sa = reinterpret_cast<float *>(smem_raw + SA_ALIGNED);

        float acc[WM][WN][4];
#pragma unroll
        for (int i = 0; i < WM; ++i)
#pragma unroll
            for (int j = 0; j < WN; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[i][j][e] = 0.0f;

        // Load A: BM × 128 bytes (one K-half) via 16-byte async copies
        auto load_A_half = [&](int outer_kt, int half_idx) __attribute__((always_inline))
        {
            const int k_start = outer_kt * BK256 + half_idx * BK128;
#pragma unroll 4
            for (int idx = threadIdx.x; idx < A_HALF_VEC_LOADS; idx += BLOCK_SIZE)
            {
                const int row = idx >> 3;       // / 8 (128 bytes per row / 16 bytes per load)
                const int col = (idx & 7) << 4; // (% 8) * 16
                const int grow = block_m + row;
                void *dst = &smem_A[row * BK128_STRIDE + col];
                const bool valid = grow < M && (k_start + col + 16 <= K);
                const void *src = valid
                                      ? static_cast<const void *>(&A[static_cast<size_t>(grow) * K + k_start + col])
                                      : static_cast<const void *>(A);
                cp_async_cg_16_zfill_128(dst, src, valid ? 16 : 0);
            }
        };

        // Decode B: 8 Q4_0 blocks per column, linearized across all threads
        auto decode_B_big = [&](int outer_kt) __attribute__((always_inline))
        {
            const int kb_start = outer_kt * 8;
            // BN * 8 items: each item = decode 1 Q4_0 block for 1 column.
            // Column-major mapping: adjacent threads → adjacent columns → coalesced global loads.
            for (int idx = threadIdx.x; idx < BN * 8; idx += BLOCK_SIZE)
            {
                const int bi = idx / BN;        // K-block index 0..7 (outer, varies slowly)
                const int col = idx & (BN - 1); // column 0..BN-1 (inner, varies quickly)
                const int gcol = block_n + col;
                const int kb = kb_start + bi;

                int32_t *dst = reinterpret_cast<int32_t *>(&smem_B[col * BK256_STRIDE + bi * 32]);

                if (gcol < N && kb < num_q40_blocks)
                {
                    const size_t base = (static_cast<size_t>(kb) * N + gcol) * Q40_PL;
                    const int4 raw = *reinterpret_cast<const int4 *>(payload + base);
                    const uint32_t r0 = static_cast<uint32_t>(raw.x);
                    const uint32_t r1 = static_cast<uint32_t>(raw.y);
                    const uint32_t r2 = static_cast<uint32_t>(raw.z);
                    const uint32_t r3 = static_cast<uint32_t>(raw.w);
                    *reinterpret_cast<int4 *>(&dst[0]) = make_int4(
                        static_cast<int32_t>(__vsub4(r0 & 0x0F0F0F0Fu, 0x08080808u)),
                        static_cast<int32_t>(__vsub4(r1 & 0x0F0F0F0Fu, 0x08080808u)),
                        static_cast<int32_t>(__vsub4(r2 & 0x0F0F0F0Fu, 0x08080808u)),
                        static_cast<int32_t>(__vsub4(r3 & 0x0F0F0F0Fu, 0x08080808u)));
                    *reinterpret_cast<int4 *>(&dst[4]) = make_int4(
                        static_cast<int32_t>(__vsub4((r0 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                        static_cast<int32_t>(__vsub4((r1 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                        static_cast<int32_t>(__vsub4((r2 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                        static_cast<int32_t>(__vsub4((r3 >> 4) & 0x0F0F0F0Fu, 0x08080808u)));

                    smem_scales_B[bi * BN + col] = scales_B[static_cast<size_t>(kb) * N + gcol];
                }
                else
                {
                    *reinterpret_cast<int4 *>(&dst[0]) = make_int4(0, 0, 0, 0);
                    *reinterpret_cast<int4 *>(&dst[4]) = make_int4(0, 0, 0, 0);
                    smem_scales_B[bi * BN + col] = 0;
                }
            }
        };

        // Load all 8 activation scales per row for the K=256 chunk
        auto load_scales_A_big = [&](int outer_kt) __attribute__((always_inline))
        {
            const int kb_start = outer_kt * 8;
            for (int idx = threadIdx.x; idx < BM * 8; idx += BLOCK_SIZE)
            {
                const int row = idx >> 3;
                const int si = idx & 7;
                const int kb = kb_start + si;
                const int grow = block_m + row;
                smem_sa[idx] = (grow < M && kb < num_q40_blocks)
                                   ? scales_A[grow * num_q40_blocks + kb]
                                   : 0.0f;
            }
        };

        // Main loop: outer K=256 tiles
        for (int ot = ot_begin; ot < ot_end; ++ot)
        {
            // Phase 1: Issue A half-0 cp.async, overlapped with B decode
            load_A_half(ot, 0);
            cp_async_commit();
            decode_B_big(ot);
            load_scales_A_big(ot);
            cp_async_wait<0>();
            __syncthreads(); // sync 1: B decoded + A half 0 ready

            // Phase 2: two K=128 halves, each with pre-loaded A fragments
#pragma unroll
            for (int hi = 0; hi < 2; ++hi)
            {
                // Pre-load ALL A MMA fragments for this half (4 k-subtiles × WM rows).
                // This decouples A ldmatrix loads from MMA, enabling the compiler
                // to pipeline them. The restructured loop iterates wj in the middle,
                // loading each B fragment once per (k, wj) instead of WM times.
                uint32_t A_frag_all[WM][4][4];
                float sa_all[WM][2][4];
#pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    const int k_offset = k * 32;
                    const int scale_idx = hi * 4 + k;
#pragma unroll
                    for (int wi = 0; wi < WM; ++wi)
                    {
                        const int a_row_base = wr * WARP_M + wi * 16;
                        load_ldmatrix_a_m16n8k32(
                            A_frag_all[wi][k],
                            reinterpret_cast<const int *>(&smem_A[a_row_base * BK128_STRIDE + k_offset]),
                            BK128_STRIDE / 4, lane_id);

                        const int local_row0 = a_row_base + gid;
                        sa_all[wi][0][k] = smem_sa[local_row0 * 8 + scale_idx];
                        sa_all[wi][1][k] = smem_sa[(local_row0 + 8) * 8 + scale_idx];
                    }
                }

                // Compute: k (outer) → wj (middle, loads B) → wi (inner, reuses pre-loaded A)
#pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    const int b_k_offset = hi * 128 + k * 32;
                    const int scale_idx_b = hi * 4 + k;

                    float sb_pre[WN][2];
#pragma unroll
                    for (int wj = 0; wj < WN; ++wj)
                    {
                        const int b_col_base = wc * WARP_N + wj * 8;
                        sb_pre[wj][0] = fp16_bits_to_float(
                            smem_scales_B[scale_idx_b * BN + b_col_base + frag_col(lane_id, 0)]);
                        sb_pre[wj][1] = fp16_bits_to_float(
                            smem_scales_B[scale_idx_b * BN + b_col_base + frag_col(lane_id, 1)]);
                    }

#pragma unroll
                    for (int wj = 0; wj < WN; ++wj)
                    {
                        uint32_t B_frag[2];
                        load_ldmatrix_b_m16n8k32(
                            B_frag,
                            reinterpret_cast<const int *>(&smem_B[(wc * WARP_N + wj * 8) * BK256_STRIDE + b_k_offset]),
                            BK256_STRIDE / 4, lane_id);

#pragma unroll
                        for (int wi = 0; wi < WM; ++wi)
                        {
                            int32_t D[4] = {0, 0, 0, 0};
                            mma_m16n8k32_s8(D, A_frag_all[wi][k], B_frag);

                            const float cs00 = sa_all[wi][0][k] * sb_pre[wj][0];
                            const float cs01 = sa_all[wi][0][k] * sb_pre[wj][1];
                            const float cs10 = sa_all[wi][1][k] * sb_pre[wj][0];
                            const float cs11 = sa_all[wi][1][k] * sb_pre[wj][1];
                            acc[wi][wj][0] += static_cast<float>(D[0]) * cs00;
                            acc[wi][wj][1] += static_cast<float>(D[1]) * cs01;
                            acc[wi][wj][2] += static_cast<float>(D[2]) * cs10;
                            acc[wi][wj][3] += static_cast<float>(D[3]) * cs11;
                        }
                    }
                }

                // Transition: reload smem_A with next half
                if (hi == 0)
                {
                    __syncthreads(); // sync 2: protect smem_A reads before reload
                    load_A_half(ot, 1);
                    cp_async_commit();
                    cp_async_wait<0>();
                    __syncthreads(); // sync 3: A half 1 ready
                }
            }

            // Barrier before next outer iteration (smem_B will be overwritten)
            if (ot + 1 < ot_end)
                __syncthreads(); // sync 4
        }

        // Epilogue: write accumulators to global memory
        const bool simple_epilogue = (beta == 0.0f) && (bias == nullptr);

#pragma unroll
        for (int wj = 0; wj < WN; ++wj)
        {
            const int tile_n = block_n + wc * WARP_N + wj * 8;
            const int gc0 = tile_n + frag_col(lane_id, 0);
            const int gc1 = tile_n + frag_col(lane_id, 1);
            const bool gc0_valid = gc0 < N;
            const bool gc1_valid = gc1 < N;
            const float bias0 = (bias && gc0_valid) ? bias[gc0] : 0.0f;
            const float bias1 = (bias && gc1_valid) ? bias[gc1] : 0.0f;

#pragma unroll
            for (int wi = 0; wi < WM; ++wi)
            {
                const int tile_m = block_m + wr * WARP_M + wi * 16;
                const bool interior = (tile_m + 15 < M) && (tile_n + 7 < N);

                if (interior && simple_epilogue)
                {
                    const int out_idx0 = (tile_m + frag_row(lane_id, 0)) * N + gc0;
                    const int out_idx1 = (tile_m + frag_row(lane_id, 1)) * N + gc1;
                    const int out_idx2 = (tile_m + frag_row(lane_id, 2)) * N + gc0;
                    const int out_idx3 = (tile_m + frag_row(lane_id, 3)) * N + gc1;

                    if constexpr (SPLIT_K > 1)
                    {
                        atomicAdd(&C[out_idx0], acc[wi][wj][0] * alpha);
                        atomicAdd(&C[out_idx1], acc[wi][wj][1] * alpha);
                        atomicAdd(&C[out_idx2], acc[wi][wj][2] * alpha);
                        atomicAdd(&C[out_idx3], acc[wi][wj][3] * alpha);
                    }
                    else
                    {
                        C[out_idx0] = acc[wi][wj][0] * alpha;
                        C[out_idx1] = acc[wi][wj][1] * alpha;
                        C[out_idx2] = acc[wi][wj][2] * alpha;
                        C[out_idx3] = acc[wi][wj][3] * alpha;
                    }
                    continue;
                }

#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    const int gr = tile_m + frag_row(lane_id, e);
                    const int gc = (e & 1) ? gc1 : gc0;

                    if (gr < M && gc < N)
                    {
                        const int out_idx = gr * N + gc;
                        float val = acc[wi][wj][e] * alpha;

                        if constexpr (SPLIT_K > 1)
                        {
                            if (blockIdx.z == 0)
                            {
                                if (beta != 0.0f && C_existing)
                                    val += beta * C_existing[out_idx];
                                if (bias)
                                    val += (e & 1) ? bias1 : bias0;
                            }
                            atomicAdd(&C[out_idx], val);
                        }
                        else
                        {
                            if (beta != 0.0f && C_existing)
                                val += beta * C_existing[out_idx];
                            if (bias)
                                val += (e & 1) ? bias1 : bias0;
                            C[out_idx] = val;
                        }
                    }
                }
            }
        }
#else
        (void)A;
        (void)payload;
        (void)scales_B;
        (void)C;
        (void)scales_A;
        (void)C_existing;
        (void)bias;
        (void)M;
        (void)N;
        (void)K;
        (void)alpha;
        (void)beta;
#endif
    }

    constexpr int q40PrefillShapeIndex(Q40PrefillShape shape)
    {
        switch (shape)
        {
        case Q40PrefillShape::TpNarrow:
            return 0;
        case Q40PrefillShape::Balanced:
            return 1;
        case Q40PrefillShape::Wide:
            return 2;
        }
        return 1;
    }

    const char *q40PrefillShapeName(Q40PrefillShape shape)
    {
        switch (shape)
        {
        case Q40PrefillShape::TpNarrow:
            return "native_q40_tc_tp_narrow";
        case Q40PrefillShape::Balanced:
            return "native_q40_tc_balanced";
        case Q40PrefillShape::Wide:
            return "native_q40_tc_wide";
        }
        return "native_q40_tc_balanced";
    }

    void recordQ40PrefillShape(Q40PrefillShape shape)
    {
        g_last_q40_prefill_shape = shape;
        g_q40_prefill_shape_counts[q40PrefillShapeIndex(shape)].fetch_add(1, std::memory_order_relaxed);
    }

    int roundUpPow2Clamped(int value, int max_value)
    {
        int out = 1;
        while (out < value && out < max_value)
            out <<= 1;
        return out;
    }

    int chooseSplitK(Q40PrefillShape shape, int M, int N, int K, int bm, int bn)
    {
        const int grid_blocks = ((M + bm - 1) / bm) * ((N + bn - 1) / bn);
        const int num_k_blocks = K / BK;
        const bool k_rich = (K >= 2 * N);
        const bool recover_underfill =
            grid_blocks < MIN_GRID_BLOCKS &&
            num_k_blocks >= 32;

        if (!k_rich && !recover_underfill)
            return 1;
        if (!recover_underfill && grid_blocks >= MIN_GRID_BLOCKS)
            return 1;
        if (num_k_blocks < 4)
            return 1;

        const int needed = (MIN_GRID_BLOCKS + grid_blocks - 1) / grid_blocks;
        int split_k = roundUpPow2Clamped(needed, 8);
        const int min_k_blocks_per_partition = k_rich ? 2 : 8;
        while (split_k > 1 && (num_k_blocks / split_k) < min_k_blocks_per_partition)
            split_k >>= 1;
        return std::max(1, split_k);
    }

    Q40PrefillShape classifyQ40PrefillShape(int M, int N, int K)
    {
        (void)M;
        if (N <= 1536)
            return Q40PrefillShape::TpNarrow;
        if (N <= 3072 && (2 * N) <= K)
            return Q40PrefillShape::TpNarrow;
        if (N >= 8192)
            return Q40PrefillShape::Wide;
        if ((4 * N) >= (9 * K))
            return Q40PrefillShape::Wide;
        return Q40PrefillShape::Balanced;
    }

    // SM count query (cached)
    int getSmCount()
    {
        static int count = 0;
        if (count == 0)
        {
            int dev = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, dev);
            if (count <= 0)
                count = 82;
        }
        return count;
    }

    // ─── Sweep-derived tile + split_k heuristic ───────────────────────
    // Fills-first strategy: prefer the largest tile family that fills the
    // GPU without split_k, only adding split_k when no family fills.
    // Within 64×128, warp config depends on real tile count:
    //   tiles ≥ 112 → w2x2 (more blocks/SM at high tile count)
    //   ki ≤ 7      → w4x2 (small K, 4 warps in M-dim)
    //   otherwise   → w2x4 (default, 8 warps for ILP)
    TileChoice chooseQ40PrefillTile(int M, int N, int K)
    {
        const int SM = getSmCount();
        constexpr int HBK = 128; // heuristic block-K unit (analysis granularity)
        const int ki = K / HBK;
        const int t64 = ((M + 63) / 64) * ((N + 63) / 64);
        const int t64x128 = ((M + 63) / 64) * ((N + 127) / 128);
        const int t128 = ((M + 127) / 128) * ((N + 127) / 128);

        const bool fills_128 = (t128 >= SM) && (M >= 128);
        const bool fills_64x128 = (t64x128 >= SM);
        const bool fills_64 = (t64 >= SM);

        // Helper: pick 64×128 warp config from real tile count
        auto pick_warp = [&](int real_tiles, int k_iters) -> TileId
        {
            if (real_tiles >= 112)
                return TileId::T64x128_w2x2;
            if (k_iters <= 7)
                return TileId::T64x128_w4x2;
            return TileId::T64x128_w2x4;
        };

        // ═══ TIER 1: 128×128 fills ═══
        if (fills_128)
        {
            bool use_128 = true;
            // Small K + many tiles → occupancy limited, smaller tile better
            if (ki <= 7 && t128 > 2 * SM)
                use_128 = false;
            else if (ki <= 16 && t128 > 3 * SM)
                use_128 = false;

            if (use_128)
            {
                // w4x4 for adequate tiles + K + M
                if ((t128 >= 56 && ki >= 7 && M >= 512) ||
                    (t128 >= 128 && ki >= 16))
                    return {TileId::T128x128_w4x4, 1};
                return {TileId::T128x128_w4x2, 1};
            }
            // Demote to 64×128
            return {pick_warp(t64x128, ki), 1};
        }

        // ═══ TIER 2: 64×128 fills ═══
        if (fills_64x128)
        {
            // Prefer 64×64 for small K + high tile parallelism
            if (ki <= 7 && t64 >= (5 * SM / 2))
                return {TileId::T64x64_w2x2, 1};
            if (ki <= 14 && t64x128 < (13 * SM / 10) && t64 >= 2 * SM)
                return {TileId::T64x64_w2x2, 1};

            int sk = 1;
            // Marginal wave fill (< 1.5 waves) + sufficient K → sk=2
            if (t64x128 < (3 * SM / 2) && ki >= 28)
                sk = 2;

            TileId warp = pick_warp(t64x128 * sk, ki);

            // 128×128+sk override for very large K at moderate M
            if (M >= 256 && ki >= 40 && t128 >= 32 && t64x128 <= (3 * SM / 2))
                return {TileId::T128x128_w4x2, 2};

            // 128×128_sk2 when tiles fill and K is very large
            if (M >= 128 && ki >= 40 && t128 >= 32 && t64x128 >= (3 * SM / 2))
            {
                if (t128 * 2 >= SM)
                    return {TileId::T128x128_w4x2, 2};
            }

            return {warp, sk};
        }

        // ═══ TIER 3: 64×64 fills ═══
        if (fills_64)
        {
            // Try upgrading to 64×128 with split_k for large K
            if (ki >= 14 && t64x128 >= 28)
            {
                const int target = 3 * SM / 2; // ~1.5 waves
                int sk = 1;
                for (int s = 1; s <= 4; s *= 2)
                {
                    if (ki < s * 7) // min 7 K-iters per partition
                        break;
                    sk = s;
                    if (t64x128 * s >= target)
                        break;
                }
                TileId warp = pick_warp(t64x128 * sk, ki);

                // Further upgrade: 128×128+sk for very large K
                if (t128 >= 16 && ki >= 40 && M >= 128)
                {
                    int sk128 = 1;
                    for (int s = 1; s <= 8; s *= 2)
                    {
                        if (ki < s * 7)
                            break;
                        sk128 = s;
                        if (t128 * s >= SM)
                            break;
                    }
                    if (t128 * sk128 >= (7 * SM / 10))
                        return {TileId::T128x128_w4x2, sk128};
                }

                return {warp, sk};
            }
            return {TileId::T64x64_w2x2, 1};
        }

        // ═══ TIER 4: Nothing fills → split_k required ═══
        TileId tile;
        int bm, bn;
        if (t128 >= 16 && ki >= 40 && M >= 128)
        {
            tile = TileId::T128x128_w4x2;
            bm = 128;
            bn = 128;
        }
        else if (t64x128 >= 8 && ki >= 8)
        {
            tile = pick_warp(0, ki); // will yield w2x4 or w4x2
            bm = 64;
            bn = 128;
        }
        else
        {
            tile = TileId::T64x64_w2x2;
            bm = 64;
            bn = 64;
        }

        const int base = ((M + bm - 1) / bm) * ((N + bn - 1) / bn);
        const int target = 3 * SM / 2; // ~1.5 waves
        int sk = 1;
        for (int s = 1; s <= 8; s *= 2)
        {
            if (base >= 8 && ki < s)
                break;
            sk = s;
            if (base * s >= target)
                break;
        }
        return {tile, sk};
    }

    bool isAmperePlus(int device_id)
    {
        static int cached_device = -1;
        static bool cached_result = false;
        if (cached_device == device_id)
            return cached_result;

        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess)
            return false;

        cached_result = (prop.major >= 8);
        cached_device = device_id;
        return cached_result;
    }

    template <int BM, int BN, int WM, int WN, int SPLIT_K = 1, bool SINGLE_PASS_MATERIALIZE = false>
    bool launchQ40TensorCoreVariant(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        cudaStream_t cuda_stream)
    {
        const dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, SPLIT_K);
        const dim3 block(WM * WN * 32);

        if constexpr (SPLIT_K > 1)
            cudaMemsetAsync(d_C_fp32, 0, static_cast<size_t>(M) * N * sizeof(float), cuda_stream);

        q40NativeVNNITensorCoreKernel<BM, BN, WM, WN, SPLIT_K, SINGLE_PASS_MATERIALIZE><<<grid, block, 0, cuda_stream>>>(
            d_A_int8,
            d_payload,
            d_scales,
            d_C_fp32,
            d_scales_A_block,
            d_C_existing,
            d_bias,
            M,
            N,
            K,
            alpha,
            beta);
        return cudaGetLastError() == cudaSuccess;
    }

    template <uint8_t CODEBOOK_ID, int BM, int BN, int WM, int WN, int SPLIT_K = 1>
    bool launchNativeVNNITC_BK64(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        const uint16_t *d_mins,
        const uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        cudaStream_t cuda_stream)
    {
        const int num_k_tiles = (K / 32 + 1) / 2; // ceil: handles K%64!=0
        int kt_per_part = num_k_tiles;
        if constexpr (SPLIT_K > 1)
            kt_per_part = (num_k_tiles + SPLIT_K - 1) / SPLIT_K;
        if (kt_per_part <= 0)
            return false;

        const dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, SPLIT_K);
        const dim3 block(WM * WN * 32);

        if constexpr (SPLIT_K > 1)
            cudaMemsetAsync(d_C_fp32, 0, static_cast<size_t>(M) * N * sizeof(float), cuda_stream);

        // Clear any stale CUDA error from prior operations (e.g. CUTLASS reference path)
        (void)cudaGetLastError();

        nativeVnniTC_BK64<CODEBOOK_ID, BM, BN, WM, WN, SPLIT_K><<<grid, block, 0, cuda_stream>>>(
            d_A_int8,
            d_payload,
            d_scales,
            d_mins,
            d_emins,
            d_C_fp32,
            d_scales_A_block,
            d_C_existing,
            d_bias,
            M,
            N,
            K,
            alpha,
            beta,
            nullptr);
        return cudaGetLastError() == cudaSuccess;
    }

    // Single-buffered BK=64 launch helper (STAGES_=1, higher occupancy target)
    template <uint8_t CODEBOOK_ID, int BM, int BN, int WM, int WN, int SPLIT_K = 1>
    bool launchNativeVNNITC_BK64_SB(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        const uint16_t *d_mins,
        const uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        cudaStream_t cuda_stream)
    {
        const int num_k_tiles = (K / 32 + 1) / 2;
        int kt_per_part = num_k_tiles;
        if constexpr (SPLIT_K > 1)
            kt_per_part = (num_k_tiles + SPLIT_K - 1) / SPLIT_K;
        if (kt_per_part <= 0)
            return false;

        const dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, SPLIT_K);
        const dim3 block(WM * WN * 32);

        if constexpr (SPLIT_K > 1)
            cudaMemsetAsync(d_C_fp32, 0, static_cast<size_t>(M) * N * sizeof(float), cuda_stream);

        (void)cudaGetLastError();

        // STAGES_=1: single-buffered, half smem, higher occupancy
        nativeVnniTC_BK64<CODEBOOK_ID, BM, BN, WM, WN, SPLIT_K, /*STAGES_=*/1><<<grid, block, 0, cuda_stream>>>(
            d_A_int8,
            d_payload,
            d_scales,
            d_mins,
            d_emins,
            d_C_fp32,
            d_scales_A_block,
            d_C_existing,
            d_bias,
            M,
            N,
            K,
            alpha,
            beta,
            nullptr);
        return cudaGetLastError() == cudaSuccess;
    }

    // =========================================================================
    // Stream-K fixup kernel: accumulates partial sums from the fixup buffer
    // into the final output matrix C. Each fixup block corresponds to a main
    // kernel block (same blockIdx.x). If that main kernel block completed a
    // tile starting from mid-K, we walk backwards to find previous blocks
    // whose fixup slots contain partial sums for the same tile.
    //
    // Key insight: a block's fixup slot always contains data for its LAST
    // tile (the only tile that can be partial). We identify the fixup tile
    // using (prev_kbc_stop - 1) / num_k_tiles, not prev_kbc / num_k_tiles.
    // =========================================================================
    template <int BM, int BN>
    __global__ __launch_bounds__(256) void streamK_fixup(
        float *__restrict__ C,
        const float *__restrict__ tmp_fixup,
        int M, int N, int K,
        int ntx, int nty, int num_k_tiles)
    {
        const long long total_work = (long long)nty * ntx * num_k_tiles;
        const long long kbc0 = (long long)blockIdx.x * total_work / gridDim.x;
        const long long kbc0_stop = (long long)(blockIdx.x + 1) * total_work / gridDim.x;

        // Early exit: empty block or started at tile boundary
        if (kbc0 >= kbc0_stop)
            return;
        if (kbc0 % num_k_tiles == 0)
            return; // started at K=0 for a tile — wrote epilogue, no fixup needed

        // Check if we complete our first tile (the one we started mid-K on)
        const int first_tile_idx = static_cast<int>(kbc0 / num_k_tiles);
        const int local_kt_begin = static_cast<int>(kbc0 % num_k_tiles);
        const int local_kt_end = min(num_k_tiles,
                                     local_kt_begin + static_cast<int>(kbc0_stop - kbc0));

        if (local_kt_end != num_k_tiles)
            return; // didn't complete the tile — wrote to fixup, not epilogue

        // This block completed a tile starting from mid-K (wrote epilogue to C).
        // We need to accumulate partial sums from earlier blocks that also
        // contributed to this tile but wrote to fixup instead of C.
        const int tile_idx = first_tile_idx;
        const int ty = tile_idx / ntx;
        const int tx = tile_idx % ntx;
        const int block_m = ty * BM;
        const int block_n = tx * BN;

        constexpr int TILE_SIZE = BM * BN;
        constexpr int ELEMS_PER_THREAD = (TILE_SIZE + 255) / 256;
        float local_sum[ELEMS_PER_THREAD];
#pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i)
            local_sum[i] = 0.0f;

        // Walk backwards through previous blocks to collect fixup contributions.
        // A block's fixup slot contains its LAST tile's partial (only the last
        // tile in a block's range can be partial — see tile loop analysis).
        int bidx = static_cast<int>(blockIdx.x) - 1;
        while (bidx >= 0)
        {
            const long long prev_kbc = (long long)bidx * total_work / gridDim.x;
            const long long prev_kbc_stop = (long long)(bidx + 1) * total_work / gridDim.x;

            if (prev_kbc >= prev_kbc_stop)
            {
                bidx--;
                continue;
            } // empty block

            // The fixup slot contains the block's LAST tile. Determine it from
            // the block's end position, not its start position.
            const int fixup_tile = static_cast<int>((prev_kbc_stop - 1) / num_k_tiles);

            if (fixup_tile < tile_idx)
                break; // this block (and all earlier) are in earlier tiles

            if (fixup_tile == tile_idx)
            {
                // Check if this block actually wrote a partial (not completed)
                const int end_k_pos = static_cast<int>((prev_kbc_stop - 1) % num_k_tiles) + 1;
                if (end_k_pos < num_k_tiles)
                {
                    // This block wrote a partial for our tile — accumulate it
                    const float *fix = tmp_fixup + bidx * TILE_SIZE;
#pragma unroll
                    for (int i = 0; i < ELEMS_PER_THREAD; ++i)
                    {
                        const int idx = threadIdx.x + i * 256;
                        if (idx < TILE_SIZE)
                            local_sum[i] += fix[idx];
                    }
                }

                // Check if this block started at K=0 for our tile — if so,
                // no earlier blocks contributed (all K-work accounted for)
                const long long tile_start = (long long)tile_idx * num_k_tiles;
                if (prev_kbc <= tile_start)
                    break;
            }
            // fixup_tile > tile_idx: this block's fixup is for a later tile, skip

            bidx--;
        }

        // Write accumulated fixup to C
#pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i)
        {
            const int idx = threadIdx.x + i * 256;
            if (idx < TILE_SIZE && local_sum[i] != 0.0f)
            {
                const int row = idx / BN;
                const int col = idx % BN;
                const int gr = block_m + row;
                const int gc = block_n + col;
                if (gr < M && gc < N)
                    C[gr * N + gc] += local_sum[i];
            }
        }
    }

    // =========================================================================
    // Stream-K force mode: 0 = auto (heuristic), 1 = force ON, -1 = force OFF
    // Controllable from tests via extern "C" cudaNativeVNNIPrefill_setStreamKMode()
    // =========================================================================
    static int g_stream_k_force_mode = 0;

    // =========================================================================
    // Persistent fixup buffer: pre-allocated once, reused across launches.
    // Avoids cudaMallocAsync/cudaFreeAsync overhead per GEMM call.
    // =========================================================================
    static float *g_stream_k_fixup_buf = nullptr;
    static size_t g_stream_k_fixup_buf_bytes = 0;

    float *getStreamKFixupBuffer(size_t required_bytes, cudaStream_t stream)
    {
        if (g_stream_k_fixup_buf && g_stream_k_fixup_buf_bytes >= required_bytes)
        {
            cudaMemsetAsync(g_stream_k_fixup_buf, 0, required_bytes, stream);
            return g_stream_k_fixup_buf;
        }
        // Grow the buffer (free old, allocate new with 2x headroom)
        if (g_stream_k_fixup_buf)
        {
            cudaDeviceSynchronize();
            cudaFree(g_stream_k_fixup_buf);
            g_stream_k_fixup_buf = nullptr;
            g_stream_k_fixup_buf_bytes = 0;
        }
        const size_t alloc_bytes = required_bytes * 2; // 2x headroom
        if (cudaMalloc(&g_stream_k_fixup_buf, alloc_bytes) != cudaSuccess)
            return nullptr;
        g_stream_k_fixup_buf_bytes = alloc_bytes;
        cudaMemsetAsync(g_stream_k_fixup_buf, 0, required_bytes, stream);
        return g_stream_k_fixup_buf;
    }

    // =========================================================================
    // Stream-K launch helper: launches the persistent GEMM kernel with nsm
    // blocks, then a fixup kernel to accumulate partial sums.
    // =========================================================================
    template <uint8_t CODEBOOK_ID, int BM, int BN, int WM, int WN>
    bool launchNativeVNNITC_BK64_StreamK(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        const uint16_t *d_mins,
        const uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        cudaStream_t cuda_stream)
    {
        const int nsm = getSmCount();
        const int ntx = (N + BN - 1) / BN;
        const int nty = (M + BM - 1) / BM;
        const int num_k_tiles = (K / 32 + 1) / 2;
        const int total_tiles = ntx * nty;

        // Grid size: 1 block/SM for stream-K. The relaxed __launch_bounds__
        // (MIN_BLOCKS_HINT=1) gives the compiler 128 regs/thread, eliminating
        // the devastating local memory spilling that occurred with 2 blocks/SM.
        // With 1 persistent block/SM, each block handles ~total_tiles/82 tiles.
        const int grid_blocks = nsm;

        // Check if fixup is needed (tiles don't divide evenly among blocks)
        const bool fixup_needed = ((long long)total_tiles * num_k_tiles) % grid_blocks != 0;

        // Get pre-allocated fixup buffer (reused across launches)
        float *d_fixup = nullptr;
        if (fixup_needed)
        {
            const size_t fixup_bytes = static_cast<size_t>(grid_blocks) * BM * BN * sizeof(float);
            d_fixup = getStreamKFixupBuffer(fixup_bytes, cuda_stream);
            if (!d_fixup)
                return false;
        }

        (void)cudaGetLastError();

        // Main kernel: grid_blocks persistent blocks (fills all SM slots)
        const dim3 grid(grid_blocks, 1, 1);
        const dim3 block(WM * WN * 32);

        nativeVnniTC_BK64<CODEBOOK_ID, BM, BN, WM, WN, /*SPLIT_K=*/1, /*STAGES_=*/2, /*STREAM_K=*/true><<<grid, block, 0, cuda_stream>>>(
            d_A_int8,
            d_payload,
            d_scales,
            d_mins,
            d_emins,
            d_C_fp32,
            d_scales_A_block,
            d_C_existing,
            d_bias,
            M, N, K,
            alpha, beta,
            d_fixup);

        if (cudaGetLastError() != cudaSuccess)
            return false;

        // Fixup kernel: accumulate partial sums
        if (fixup_needed)
        {
            streamK_fixup<BM, BN><<<dim3(grid_blocks, 1, 1), dim3(256), 0, cuda_stream>>>(
                d_C_fp32, d_fixup, M, N, K, ntx, nty, num_k_tiles);

            if (cudaGetLastError() != cudaSuccess)
                return false;
        }

        return true;
    }

    // =========================================================================
    // Stream-K profitability heuristic: determines whether stream-K is likely
    // to outperform standard tiling for a given GEMM shape.
    // Stream-K wins when wave-tail inefficiency is significant (>10%).
    // =========================================================================
    bool shouldUseStreamK(int M, int N, int K, int bm, int bn)
    {
        // Respect force mode from test harness
        if (g_stream_k_force_mode < 0)
            return false;
        if (g_stream_k_force_mode > 0)
            return true;

        // Auto mode: data-driven heuristic from A/B profiling (v2_perf_cuda_streamk_ab).
        // Stream-K wins when the standard tiling's last wave is poorly filled,
        // wasting SM slots. With 128×128 tiles and 82 SMs (RTX 3090), the sweet
        // spot is wave efficiency between ~40-70% (significant tail waste).
        //
        // Profiled results (Qwen2.5-7B shapes, T128x128_w4x4):
        //   QKV  M=596: tiles=180, wave_eff=54.9% → SK wins 1.044x
        //   Down M=512: tiles=112, wave_eff=68.3% → SK wins 1.179x
        //   Wo   M=596: tiles=140, wave_eff=85.4% → STD wins (already efficient)
        //   GateUp M=*: tiles=148-740, wave_eff≥60% → STD wins (high occupancy)
        //
        // Stream-K also loses when tiles >> nsm (many full waves), because
        // the standard kernel already has high utilization and stream-K's
        // lower occupancy (33% vs 67%) hurts.
        const int nsm = getSmCount();
        const int total_tiles = ((M + bm - 1) / bm) * ((N + bn - 1) / bn);

        // Stream-K only helps with 128×128 tiles (where it's compiled with
        // relaxed launch_bounds). Skip for smaller tiles.
        if (bm < 128 || bn < 128)
            return false;

        // Need at least 1 wave of tiles (otherwise standard is fine)
        if (total_tiles < nsm)
            return false;

        // Compute wave tail efficiency (how full is the last wave)
        const float waves = static_cast<float>(total_tiles) / static_cast<float>(nsm);
        const float frac = waves - static_cast<float>(static_cast<int>(waves));
        const float wave_eff = (frac < 0.001f) ? 1.0f : frac;

        // Stream-K wins when wave_eff < 70% (significant tail waste)
        // AND total tiles don't exceed ~3 waves (diminishing returns)
        return wave_eff < 0.70f && waves < 3.0f;
    }

    int chooseSplitK_BK64(Q40PrefillShape shape, int M, int N, int K, int bm, int bn)
    {
        const int grid_blocks = ((M + bm - 1) / bm) * ((N + bn - 1) / bn);
        const int num_k_tiles = (K / 32 + 1) / 2; // ceil: handles K%64!=0
        const bool k_rich = (K >= 2 * N);

        // Estimate max concurrent blocks per SM from block size.
        // BM=128 → 256 threads, 128 regs, ~44KB smem → 2 blocks/SM.
        // BM=64  → 128 threads, 168 regs, ~32KB smem → 3 blocks/SM.
        const int num_warps = (bm >= 128) ? 8 : 4;
        const int max_blocks_per_sm = (num_warps >= 8) ? 2 : 3;
        const int num_sms = getSmCount();
        const int total_slots = num_sms * max_blocks_per_sm;
        const float waves = static_cast<float>(grid_blocks) / static_cast<float>(total_slots);

        // Wave-efficiency split_k: when grid doesn't fill 2 full waves,
        // use split_k to improve SM utilization even if grid_blocks >= MIN_GRID_BLOCKS.
        // Only for k_rich shapes (large K) where atomicAdd overhead is amortized.
        const bool low_wave_fill = k_rich && waves < 2.0f && num_k_tiles >= 16;

        const bool recover_underfill =
            grid_blocks < MIN_GRID_BLOCKS &&
            num_k_tiles >= 16;

        if (!k_rich && !recover_underfill && !low_wave_fill)
            return 1;
        if (!recover_underfill && !low_wave_fill && grid_blocks >= MIN_GRID_BLOCKS)
            return 1;
        if (num_k_tiles < 2)
            return 1;

        int needed;
        if (low_wave_fill && !recover_underfill)
        {
            // Target ~4 waves for better amortization of wave tails
            // and higher sustained SM utilization.
            const int target_blocks = total_slots * 4;
            needed = (target_blocks + grid_blocks - 1) / grid_blocks;
        }
        else
        {
            needed = (MIN_GRID_BLOCKS + grid_blocks - 1) / grid_blocks;
        }
        int split_k = roundUpPow2Clamped(needed, 8);
        const int min_tiles_per_part = k_rich ? 1 : 4;
        while (split_k > 1 && (num_k_tiles / split_k) < min_tiles_per_part)
            split_k >>= 1;
        return std::max(1, split_k);
    }

    // BK=256 launch helper: sets >48KB dynamic smem opt-in before first launch
    template <int BM, int BN, int WM, int WN, int SPLIT_K = 1>
    bool launchNativeVNNITC_BK256(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        cudaStream_t cuda_stream)
    {
        // Compute dynamic smem size (must match kernel layout: A uses K=128 half)
        constexpr int SCALES_B_OFF = BM * BK128_STRIDE + BN * BK256_STRIDE;
        constexpr int SA_OFF = SCALES_B_OFF + 8 * BN * static_cast<int>(sizeof(uint16_t));
        constexpr int SA_ALIGNED = (SA_OFF + 3) & ~3;
        constexpr int smem_bytes = SA_ALIGNED + BM * 8 * static_cast<int>(sizeof(float));

        // Opt-in for >48KB dynamic shared memory (once per kernel template)
        static bool smem_configured = false;
        if (!smem_configured)
        {
            auto fn_ptr = nativeVnniTC_BK256<BM, BN, WM, WN, SPLIT_K>;
            cudaFuncSetAttribute(fn_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
            smem_configured = true;
        }

        const int num_outer_tiles = (K / 32 + 7) / 8;
        int ot_per_part = num_outer_tiles;
        if constexpr (SPLIT_K > 1)
            ot_per_part = (num_outer_tiles + SPLIT_K - 1) / SPLIT_K;
        if (ot_per_part <= 0)
            return false;

        const dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, SPLIT_K);
        const dim3 block(WM * WN * 32);

        if constexpr (SPLIT_K > 1)
            cudaMemsetAsync(d_C_fp32, 0, static_cast<size_t>(M) * N * sizeof(float), cuda_stream);

        (void)cudaGetLastError(); // clear stale errors

        nativeVnniTC_BK256<BM, BN, WM, WN, SPLIT_K><<<grid, block, smem_bytes, cuda_stream>>>(
            d_A_int8,
            d_payload,
            d_scales,
            d_C_fp32,
            d_scales_A_block,
            d_C_existing,
            d_bias,
            M,
            N,
            K,
            alpha,
            beta);
        return cudaGetLastError() == cudaSuccess;
    }

    int chooseSplitK_BK256(Q40PrefillShape shape, int M, int N, int K, int bm, int bn)
    {
        const int grid_blocks = ((M + bm - 1) / bm) * ((N + bn - 1) / bn);
        const int num_outer_tiles = (K / 32 + 7) / 8;
        const bool k_rich = (K >= 2 * N);
        const bool recover_underfill =
            grid_blocks < MIN_GRID_BLOCKS &&
            num_outer_tiles >= 4;

        if (!k_rich && !recover_underfill)
            return 1;
        if (!recover_underfill && grid_blocks >= MIN_GRID_BLOCKS)
            return 1;
        if (num_outer_tiles < 2)
            return 1;

        const int needed = (MIN_GRID_BLOCKS + grid_blocks - 1) / grid_blocks;
        int split_k = roundUpPow2Clamped(needed, 8);
        while (split_k > 1 && (num_outer_tiles / split_k) < 2)
            split_k >>= 1;
        return std::max(1, split_k);
    }
}

extern "C"
{
    bool cudaNativeVNNIPrefillQ40_fp32(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        int cuda_device_id,
        void *stream)
    {
        if (!d_A_int8 || !d_payload || !d_scales || !d_C_fp32 || !d_scales_A_block)
            return false;
        if (M <= 0 || N <= 0 || K <= 0 || (K % BK) != 0)
            return false;
        if (!isAmperePlus(cuda_device_id))
            return false;
        if (cudaSetDevice(cuda_device_id) != cudaSuccess)
            return false;

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        // Legacy shape tracking (kept for diagnostics / family counters)
        const Q40PrefillShape shape = classifyQ40PrefillShape(M, N, K);
        recordQ40PrefillShape(shape);

        // ─── Sweep-derived dispatch ───────────────────────────────────
        const TileChoice tc = chooseQ40PrefillTile(M, N, K);

// Macro to fan out split_k for a given tile config (avoids 24-way copy-paste)
#define Q40_DISPATCH_SK(CB, BM_, BN_, WM_, WN_)                             \
    do                                                                      \
    {                                                                       \
        if (tc.split_k == 1 && shouldUseStreamK(M, N, K, BM_, BN_))         \
            return launchNativeVNNITC_BK64_StreamK<CB, BM_, BN_, WM_, WN_>( \
                d_A_int8, d_payload, d_scales, nullptr, nullptr,            \
                d_C_fp32, d_scales_A_block,                                 \
                M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);   \
        switch (tc.split_k)                                                 \
        {                                                                   \
        case 8:                                                             \
            return launchNativeVNNITC_BK64<CB, BM_, BN_, WM_, WN_, 8>(      \
                d_A_int8, d_payload, d_scales, nullptr, nullptr,            \
                d_C_fp32, d_scales_A_block,                                 \
                M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);   \
        case 4:                                                             \
            return launchNativeVNNITC_BK64<CB, BM_, BN_, WM_, WN_, 4>(      \
                d_A_int8, d_payload, d_scales, nullptr, nullptr,            \
                d_C_fp32, d_scales_A_block,                                 \
                M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);   \
        case 2:                                                             \
            return launchNativeVNNITC_BK64<CB, BM_, BN_, WM_, WN_, 2>(      \
                d_A_int8, d_payload, d_scales, nullptr, nullptr,            \
                d_C_fp32, d_scales_A_block,                                 \
                M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);   \
        default:                                                            \
            return launchNativeVNNITC_BK64<CB, BM_, BN_, WM_, WN_, 1>(      \
                d_A_int8, d_payload, d_scales, nullptr, nullptr,            \
                d_C_fp32, d_scales_A_block,                                 \
                M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);   \
        }                                                                   \
    } while (0)

        switch (tc.tile)
        {
        case TileId::T64x64_w2x2:
            Q40_DISPATCH_SK(0, 64, 64, 2, 2);
        case TileId::T64x128_w2x2:
            Q40_DISPATCH_SK(0, 64, 128, 2, 2);
        case TileId::T64x128_w4x2:
            Q40_DISPATCH_SK(0, 64, 128, 4, 2);
        case TileId::T64x128_w2x4:
            Q40_DISPATCH_SK(0, 64, 128, 2, 4);
        case TileId::T128x128_w4x2:
            Q40_DISPATCH_SK(0, 128, 128, 4, 2);
        case TileId::T128x128_w4x4:
            Q40_DISPATCH_SK(0, 128, 128, 4, 4);
        }

#undef Q40_DISPATCH_SK
        return false; // unreachable
    }

    const char *cudaNativeVNNIPrefillQ40_lastSelectedFamily()
    {
        return q40PrefillShapeName(g_last_q40_prefill_shape);
    }

    unsigned long long cudaNativeVNNIPrefillQ40_getFamilyCount(int family_index)
    {
        if (family_index < 0 || family_index >= 3)
            return 0;
        return g_q40_prefill_shape_counts[family_index].load(std::memory_order_relaxed);
    }

    void cudaNativeVNNIPrefillQ40_resetFamilyCounts()
    {
        for (auto &count : g_q40_prefill_shape_counts)
            count.store(0, std::memory_order_relaxed);
    }

    // Stream-K force mode: 0=auto(heuristic), 1=force ON, -1=force OFF
    void cudaNativeVNNIPrefill_setStreamKMode(int mode)
    {
        g_stream_k_force_mode = mode;
    }

    int cudaNativeVNNIPrefill_getStreamKMode()
    {
        return g_stream_k_force_mode;
    }

    void cudaNativeVNNIPrefill_freeStreamKFixup()
    {
        if (g_stream_k_fixup_buf)
        {
            cudaDeviceSynchronize();
            cudaFree(g_stream_k_fixup_buf);
            g_stream_k_fixup_buf = nullptr;
            g_stream_k_fixup_buf_bytes = 0;
        }
    }

    bool cudaNativeVNNIPrefillIQ4NL_fp32(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        int cuda_device_id,
        void *stream)
    {
        if (!d_A_int8 || !d_payload || !d_scales || !d_C_fp32 || !d_scales_A_block)
            return false;
        if (M <= 0 || N <= 0 || K <= 0 || (K % BK) != 0)
            return false;
        if (!isAmperePlus(cuda_device_id))
            return false;
        if (cudaSetDevice(cuda_device_id) != cudaSuccess)
            return false;

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        const Q40PrefillShape shape = classifyQ40PrefillShape(M, N, K);
        recordQ40PrefillShape(shape);

        // IQ4_NL uses BK=64 only (same payload layout as Q4_0).
        // K-tail handling supports K%64!=0 (K%32==0 is sufficient).

        switch (shape)
        {
        case Q40PrefillShape::TpNarrow:
        {
            const int split_k = chooseSplitK_BK64(shape, M, N, K, 64, 128);
            switch (split_k)
            {
            case 8:
                return launchNativeVNNITC_BK64<4, 64, 128, 2, 2, 8>(
                    d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            case 4:
                return launchNativeVNNITC_BK64<4, 64, 128, 2, 2, 4>(
                    d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            case 2:
                return launchNativeVNNITC_BK64<4, 64, 128, 2, 2, 2>(
                    d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            default:
                return launchNativeVNNITC_BK64<4, 64, 128, 2, 2, 1>(
                    d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            }
        }

        case Q40PrefillShape::Wide:
        {
            const int split_k = chooseSplitK_BK64(shape, M, N, K, 64, 128);
            switch (split_k)
            {
            case 8:
                return launchNativeVNNITC_BK64<4, 64, 128, 2, 2, 8>(
                    d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            case 4:
                return launchNativeVNNITC_BK64<4, 64, 128, 2, 2, 4>(
                    d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            case 2:
                return launchNativeVNNITC_BK64<4, 64, 128, 2, 2, 2>(
                    d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            default:
                return launchNativeVNNITC_BK64<4, 64, 128, 2, 2, 1>(
                    d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            }
        }

        case Q40PrefillShape::Balanced:
        default:
        {
            const int split_k = chooseSplitK_BK64(shape, M, N, K, 64, 128);
            switch (split_k)
            {
            case 8:
                return launchNativeVNNITC_BK64<4, 64, 128, 2, 2, 8>(
                    d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            case 4:
                return launchNativeVNNITC_BK64<4, 64, 128, 2, 2, 4>(
                    d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            case 2:
                return launchNativeVNNITC_BK64<4, 64, 128, 2, 2, 2>(
                    d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            default:
                return launchNativeVNNITC_BK64<4, 64, 128, 2, 2, 1>(
                    d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            }
        }
        }
    }
} // extern "C"

// =========================================================================
// Generic prefill dispatch: routes by codebook_id to the correct BK=64
// template instantiation. Supports all single-scale and asymmetric formats.
// Dual-scale formats (Q6_K, Q3_K, Q2_K, IQ2_S, IQ2_XS, IQ1_M) are not
// supported here — they need a half-block MMA approach (future work).
// =========================================================================

namespace
{
    // Helper: launch BK=64 with split-K for a given codebook
    template <uint8_t CB>
    bool launchGenericPrefillBK64(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        const uint16_t *d_mins,
        const uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int M, int N, int K,
        float alpha, float beta,
        const float *d_C_existing,
        const float *d_bias,
        cudaStream_t cuda_stream)
    {
        const Q40PrefillShape shape = classifyQ40PrefillShape(M, N, K);
        const int split_k = chooseSplitK_BK64(shape, M, N, K, 64, 128);
        switch (split_k)
        {
        case 8:
            return launchNativeVNNITC_BK64<CB, 64, 128, 2, 2, 8>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C_fp32, d_scales_A_block,
                M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
        case 4:
            return launchNativeVNNITC_BK64<CB, 64, 128, 2, 2, 4>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C_fp32, d_scales_A_block,
                M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
        case 2:
            return launchNativeVNNITC_BK64<CB, 64, 128, 2, 2, 2>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C_fp32, d_scales_A_block,
                M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
        default:
            return launchNativeVNNITC_BK64<CB, 64, 128, 2, 2, 1>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C_fp32, d_scales_A_block,
                M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
        }
    }
} // namespace

// Profitability gate for dual-scale formats.
// Split-MMA doubles IMMA count per K-tile. NativeVNNI only wins over
// CUTLASS expanded path when the shape is memory-bound (high K/N, small M).
// Empirical data (RTX 3090, BK64, M∈{32,64,128}):
//   K/N ≥ 2 && M ≤ 64  → 81% profitable, avg 1.15x speedup
//   K/N < 2 || M > 64  → only ~8% profitable, avg 0.67x
//   IQ1_M (CB17)       → 3.89x instruction count, 16.7% occupancy,
//                         only 3% profitable (mean 0.38x) — always gate out
static bool isDualScalePrefillProfitable(int M, int N, int K, uint8_t codebook_id)
{
    // IQ1_M: delta correction overhead is too extreme (13424 insns vs 3448 baseline)
    if (codebook_id == 17)
        return false;

    // Dual-scale formats: CB 8 (Q6_K), 9 (Q3_K), 10 (Q2_K), 13 (IQ2_S), 14 (IQ2_XS)
    // Profitable when K-rich (memory-bound) and M is not too large (not compute-bound)
    return (K >= 2 * N && M <= 64);
}

extern "C" bool cudaNativeVNNIPrefill_fp32(
    const int8_t *d_A_int8,
    const uint8_t *d_payload,
    const uint16_t *d_scales,
    const uint16_t *d_mins,
    const uint32_t *d_emins,
    float *d_C_fp32,
    const float *d_scales_A_block,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    const float *d_C_existing,
    const float *d_bias,
    uint8_t codebook_id,
    int cuda_device_id,
    void *stream)
{
    if (!d_A_int8 || !d_payload || !d_scales || !d_C_fp32 || !d_scales_A_block)
        return false;
    if (M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
        return false;
    if (!isAmperePlus(cuda_device_id))
        return false;
    if (cudaSetDevice(cuda_device_id) != cudaSuccess)
        return false;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    switch (codebook_id)
    {
    // --- Single-scale formats (no min correction) ---
    case 0:
        return launchGenericPrefillBK64<0>(
            d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
            M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
    case 4:
        return launchGenericPrefillBK64<4>(
            d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
            M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
    case 6: // Q5_0
        return launchGenericPrefillBK64<6>(
            d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
            M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
    case 11: // IQ3_S
        return launchGenericPrefillBK64<11>(
            d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
            M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
    case 12: // IQ3_XXS
        return launchGenericPrefillBK64<12>(
            d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
            M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
    case 15: // IQ2_XXS
        return launchGenericPrefillBK64<15>(
            d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
            M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);

    // --- Asymmetric formats (need min correction, d_mins required) ---
    case 5: // Q4_1 / Q4_K / Q5_K
        if (!d_mins)
            return false;
        return launchGenericPrefillBK64<5>(
            d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block,
            M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
    case 7: // Q5_1
        if (!d_mins)
            return false;
        return launchGenericPrefillBK64<7>(
            d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block,
            M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
    case 16: // IQ1_S
        if (!d_mins)
            return false;
        return launchGenericPrefillBK64<16>(
            d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block,
            M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);

    // --- Dual-scale formats (separate lo/hi scales via split MMA) ---
    // Profitability gate: only launch for shapes where NativeVNNI beats CUTLASS.
    case 8: // Q6_K
        if (!d_mins || !isDualScalePrefillProfitable(M, N, K, 8))
            return false;
        return launchGenericPrefillBK64<8>(
            d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block,
            M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
    case 9: // Q3_K
        if (!d_mins || !isDualScalePrefillProfitable(M, N, K, 9))
            return false;
        return launchGenericPrefillBK64<9>(
            d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block,
            M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
    case 10: // Q2_K (dual-scale + asymmetric via emins)
        if (!d_mins || !isDualScalePrefillProfitable(M, N, K, 10))
            return false;
        return launchGenericPrefillBK64<10>(
            d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C_fp32, d_scales_A_block,
            M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
    case 13: // IQ2_S
        if (!d_mins || !isDualScalePrefillProfitable(M, N, K, 13))
            return false;
        return launchGenericPrefillBK64<13>(
            d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block,
            M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
    case 14: // IQ2_XS
        if (!d_mins || !isDualScalePrefillProfitable(M, N, K, 14))
            return false;
        return launchGenericPrefillBK64<14>(
            d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block,
            M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
    case 17: // IQ1_M (dual-scale + delta correction)
        if (!d_mins || !isDualScalePrefillProfitable(M, N, K, 17))
            return false;
        return launchGenericPrefillBK64<17>(
            d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block,
            M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);

    // --- 8-bit format (no decode overhead, single-scale) ---
    case 18: // Q8_0
        return launchGenericPrefillBK64<18>(
            d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
            M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);

    default:
        return false;
    }
}

// =========================================================================
// GEMM Sweep Benchmark API
// =========================================================================
// Exposes all tile-config variants via numbered config IDs so an external
// benchmark driver can systematically sweep (BM, BN, WARPS_M, WARPS_N,
// SPLIT_K) for any (M, N, K) shape.
// =========================================================================

namespace
{
    struct SweepConfig
    {
        int bm, bn, warps_m, warps_n;
        const char *name;
    };

    // 17 tile configs covering the practical design space.
    // Constraint: WARP_M = BM/WARPS_M >= 16, WARP_N = BN/WARPS_N >= 8.
    static constexpr SweepConfig kSweepConfigs[] = {
        {32, 128, 1, 2, "32x128_w1x2"},   //  0:  64 thr, ~27KB smem
        {32, 128, 1, 4, "32x128_w1x4"},   //  1: 128 thr, ~27KB smem
        {64, 64, 2, 1, "64x64_w2x1"},     //  2:  64 thr, ~22KB smem
        {64, 64, 2, 2, "64x64_w2x2"},     //  3: 128 thr, ~22KB smem
        {64, 128, 2, 2, "64x128_w2x2"},   //  4: 128 thr, ~33KB smem (DEFAULT small-M)
        {64, 128, 4, 2, "64x128_w4x2"},   //  5: 256 thr, ~33KB smem
        {64, 128, 2, 4, "64x128_w2x4"},   //  6: 256 thr, ~33KB smem
        {128, 64, 4, 1, "128x64_w4x1"},   //  7: 128 thr, ~33KB smem
        {128, 64, 2, 1, "128x64_w2x1"},   //  8:  64 thr, ~33KB smem (high ILP)
        {128, 128, 4, 2, "128x128_w4x2"}, //  9: 256 thr, ~44KB smem (DEFAULT large-M)
        {128, 128, 2, 2, "128x128_w2x2"}, // 10: 128 thr, ~44KB smem (high ILP)
        {128, 128, 4, 4, "128x128_w4x4"}, // 11: 512 thr, ~44KB smem
    };
    static constexpr int kNumSweepConfigs = sizeof(kSweepConfigs) / sizeof(kSweepConfigs[0]);

    template <int BM, int BN, int WM, int WN>
    bool sweepLaunchQ40(int split_k,
                        const int8_t *A, const uint8_t *payload, const uint16_t *scales,
                        float *C, const float *scales_A,
                        int M, int N, int K, float alpha, float beta,
                        const float *C_existing, const float *bias,
                        cudaStream_t stream)
    {
        switch (split_k)
        {
        case 8:
            return launchNativeVNNITC_BK64<0, BM, BN, WM, WN, 8>(
                A, payload, scales, nullptr, nullptr, C, scales_A,
                M, N, K, alpha, beta, C_existing, bias, stream);
        case 4:
            return launchNativeVNNITC_BK64<0, BM, BN, WM, WN, 4>(
                A, payload, scales, nullptr, nullptr, C, scales_A,
                M, N, K, alpha, beta, C_existing, bias, stream);
        case 2:
            return launchNativeVNNITC_BK64<0, BM, BN, WM, WN, 2>(
                A, payload, scales, nullptr, nullptr, C, scales_A,
                M, N, K, alpha, beta, C_existing, bias, stream);
        default:
            return launchNativeVNNITC_BK64<0, BM, BN, WM, WN, 1>(
                A, payload, scales, nullptr, nullptr, C, scales_A,
                M, N, K, alpha, beta, C_existing, bias, stream);
        }
    }

} // namespace

extern "C"
{
    int cudaGemmSweepNumConfigs() { return kNumSweepConfigs; }

    void cudaGemmSweepGetConfig(int idx, int *bm, int *bn, int *warps_m, int *warps_n, const char **name)
    {
        if (idx < 0 || idx >= kNumSweepConfigs)
            return;
        if (bm)
            *bm = kSweepConfigs[idx].bm;
        if (bn)
            *bn = kSweepConfigs[idx].bn;
        if (warps_m)
            *warps_m = kSweepConfigs[idx].warps_m;
        if (warps_n)
            *warps_n = kSweepConfigs[idx].warps_n;
        if (name)
            *name = kSweepConfigs[idx].name;
    }

    bool cudaGemmSweepLaunch(
        int config_idx, int split_k,
        const int8_t *A, const uint8_t *payload, const uint16_t *scales,
        float *C, const float *scales_A,
        int M, int N, int K,
        float alpha, float beta,
        const float *C_existing, const float *bias,
        int device_id, void *stream)
    {
        if (config_idx < 0 || config_idx >= kNumSweepConfigs)
            return false;
        if (!A || !payload || !scales || !C || !scales_A)
            return false;
        if (M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
            return false;
        if (!isAmperePlus(device_id))
            return false;
        if (cudaSetDevice(device_id) != cudaSuccess)
            return false;

        cudaStream_t cs = static_cast<cudaStream_t>(stream);

        switch (config_idx)
        {
        case 0:
            return sweepLaunchQ40<32, 128, 1, 2>(split_k, A, payload, scales, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 1:
            return sweepLaunchQ40<32, 128, 1, 4>(split_k, A, payload, scales, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 2:
            return sweepLaunchQ40<64, 64, 2, 1>(split_k, A, payload, scales, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 3:
            return sweepLaunchQ40<64, 64, 2, 2>(split_k, A, payload, scales, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 4:
            return sweepLaunchQ40<64, 128, 2, 2>(split_k, A, payload, scales, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 5:
            return sweepLaunchQ40<64, 128, 4, 2>(split_k, A, payload, scales, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 6:
            return sweepLaunchQ40<64, 128, 2, 4>(split_k, A, payload, scales, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 7:
            return sweepLaunchQ40<128, 64, 4, 1>(split_k, A, payload, scales, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 8:
            return sweepLaunchQ40<128, 64, 2, 1>(split_k, A, payload, scales, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 9:
            return sweepLaunchQ40<128, 128, 4, 2>(split_k, A, payload, scales, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 10:
            return sweepLaunchQ40<128, 128, 2, 2>(split_k, A, payload, scales, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 11:
            return sweepLaunchQ40<128, 128, 4, 4>(split_k, A, payload, scales, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        default:
            return false;
        }
    }

    // Stream-K variant of sweep launch: calls launchNativeVNNITC_BK64_StreamK
    // directly for a given tile config. Used by the A/B perf harness.
    bool cudaGemmSweepLaunchStreamK(
        int config_idx,
        const int8_t *A, const uint8_t *payload, const uint16_t *scales,
        float *C, const float *scales_A,
        int M, int N, int K,
        float alpha, float beta,
        const float *C_existing, const float *bias,
        int device_id, void *stream)
    {
        if (config_idx < 0 || config_idx >= kNumSweepConfigs)
            return false;
        if (!A || !payload || !scales || !C || !scales_A)
            return false;
        if (M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
            return false;
        if (!isAmperePlus(device_id))
            return false;
        if (cudaSetDevice(device_id) != cudaSuccess)
            return false;

        cudaStream_t cs = static_cast<cudaStream_t>(stream);

        switch (config_idx)
        {
        case 0:
            return launchNativeVNNITC_BK64_StreamK<0, 32, 128, 1, 2>(A, payload, scales, nullptr, nullptr, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 1:
            return launchNativeVNNITC_BK64_StreamK<0, 32, 128, 1, 4>(A, payload, scales, nullptr, nullptr, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 2:
            return launchNativeVNNITC_BK64_StreamK<0, 64, 64, 2, 1>(A, payload, scales, nullptr, nullptr, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 3:
            return launchNativeVNNITC_BK64_StreamK<0, 64, 64, 2, 2>(A, payload, scales, nullptr, nullptr, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 4:
            return launchNativeVNNITC_BK64_StreamK<0, 64, 128, 2, 2>(A, payload, scales, nullptr, nullptr, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 5:
            return launchNativeVNNITC_BK64_StreamK<0, 64, 128, 4, 2>(A, payload, scales, nullptr, nullptr, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 6:
            return launchNativeVNNITC_BK64_StreamK<0, 64, 128, 2, 4>(A, payload, scales, nullptr, nullptr, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 7:
            return launchNativeVNNITC_BK64_StreamK<0, 128, 64, 4, 1>(A, payload, scales, nullptr, nullptr, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 8:
            return launchNativeVNNITC_BK64_StreamK<0, 128, 64, 2, 1>(A, payload, scales, nullptr, nullptr, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 9:
            return launchNativeVNNITC_BK64_StreamK<0, 128, 128, 4, 2>(A, payload, scales, nullptr, nullptr, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 10:
            return launchNativeVNNITC_BK64_StreamK<0, 128, 128, 2, 2>(A, payload, scales, nullptr, nullptr, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        case 11:
            return launchNativeVNNITC_BK64_StreamK<0, 128, 128, 4, 4>(A, payload, scales, nullptr, nullptr, C, scales_A, M, N, K, alpha, beta, C_existing, bias, cs);
        default:
            return false;
        }
    }
}
