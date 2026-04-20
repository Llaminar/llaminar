/**
 * @file CUDAFusedTCGemm_Ampere.cu
 * @brief Ampere+ fused tensor-core GEMM with mma.sync.m16n8k32 for INT8.
 *
 * V2 architecture targeting SM80+ (Ampere / Ada / Hopper).
 * Structural improvements over V1 (CUDAFusedTCGemm_Turing.cu, WMMA m16n16k16):
 *
 *   1. mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 inline PTX
 *      — one MMA instruction per K=32 quantisation block (vs 2 WMMA calls)
 *      — direct register fragment management (no opaque WMMA types)
 *      — K=32 aligns with quantisation block size → cleaner scheduling
 *
 *   2. cp.async.ca.shared.global for global→shared memory transfers
 *      — hardware async copy bypasses L1, uses L2 efficiently
 *      — no register intermediaries → less register pressure
 *      — enables true compute/memory overlap with STAGES >= 3
 *
 *   3. Multi-stage software pipeline (STAGES=2,3,4)
 *      — STAGES-deep circular shared memory buffer
 *      — commit_group / wait_group for pipeline synchronization
 *      — STAGES=2: equivalent to double-buffer (compute overlaps 1 load)
 *      — STAGES=3: compute overlaps 2 loads (hides more memory latency)
 *
 *   4. Split-K (SPLIT_K > 1)
 *      — K-dimension partitioned across CTAs via grid.z
 *      — Each CTA writes its FP32 partial into a dedicated scratch plane
 *      — A fixed-order reduction kernel combines split-K planes into C
 *      — Avoids FP32 atomicAdd non-determinism while preserving the fast path
 *      — Useful for tall-K bandwidth-bound shapes (e.g., FFN_Down)
 *
 * Falls back gracefully on SM75 (Turing) — returns false, V1 handles it.
 *
 * Data layout contract (same as V1):
 *   A:        [M × K]       row-major INT8 (blockwise quantised, block=32)
 *   B:        [K/32][N][32] tc-blocked INT8 (column-major per 32-element group)
 *   C:        [M × N]       row-major FP32
 *   scales_A: [M × K/32]    per-row per-block activation scales
 *   scales_B: [N]           per-column weight scales
 *
 * Template parameters:
 *   BM, BN:            CTA tile dimensions (output rows × cols)
 *   WARPS_M, WARPS_N:  Warp grid layout within the CTA
 *   STAGES:            Pipeline depth (2=double buffer; 3+=multi-stage)
 *   SPLIT_K:           K-partitions (1=standard; >1=split-K partial reduction)
 *
 * Per-warp tile (m16n8k32 units):
 *   WARP_M = BM / WARPS_M,   WARP_N = BN / WARPS_N
 *   WM = WARP_M / 16,        WN = WARP_N / 8
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace
{

    // ─── Shared memory stride with padding ───
    // 32 bytes data + 16 bytes padding = 48 bytes per row/col.
    // Bank analysis: stride 48 ÷ 4 = 12 words. For mma.sync A fragment loads
    // (gid=0..7, tid=0..3), bank = (gid*12 + tid) % 32 → all 32 unique banks.
    // Zero bank conflicts empirically verified for all 4 A fragment registers.
    constexpr int BK = 32;                                      // quantisation block size (fixed)
    constexpr int SMEM_PAD = 16;                                // padding bytes per row
    [[maybe_unused]] constexpr int SMEM_STRIDE = BK + SMEM_PAD; // = 48
    constexpr int MAX_SPLIT_K_PARTITIONS = 8;
    constexpr size_t SPLIT_K_PARTIAL_SCRATCH_BUDGET_BYTES = 256ull * 1024ull * 1024ull;

    // ─── mma.sync.m16n8k32 output fragment mapping (PTX ISA §9.7.13.4.10) ───
    //
    // Each thread in a warp holds 4 INT32 elements of the m16×n8 output.
    //
    // For element index l ∈ {0,1,2,3} and lane_id (0-31):
    //   groupID        = lane_id / 4    (0-7)
    //   tid_in_group   = lane_id % 4    (0-3)
    //   row = (l / 2) * 8 + groupID     → l<2: rows 0-7;  l≥2: rows 8-15
    //   col = tid_in_group * 2 + (l % 2) → 0-7

    [[maybe_unused]] __device__ __forceinline__ int v2_frag_row(int lane_id, int elem)
    {
        return (elem >> 1) * 8 + (lane_id >> 2);
    }

    [[maybe_unused]] __device__ __forceinline__ int v2_frag_col(int lane_id, int elem)
    {
        return (lane_id & 3) * 2 + (elem & 1);
    }

    // ─── cp.async inline PTX helpers ───

    [[maybe_unused]] __device__ __forceinline__ void cp_async_cg_16_zfill_128(
        void *smem_dst, const void *gmem_src, int src_size)
    {
        const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::"r"(smem_addr), "l"(gmem_src), "r"(src_size) : "memory");
    }

    [[maybe_unused]] __device__ __forceinline__ void cp_async_commit()
    {
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    template <int N>
    __device__ __forceinline__ void cp_async_wait()
    {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
    }

    [[maybe_unused]] __device__ __forceinline__ void load_ldmatrix_a_m16n8k32(
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

    [[maybe_unused]] __device__ __forceinline__ void load_ldmatrix_b_m16n8k32(
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

    // ─── Inline PTX MMA: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ───
    //
    // D += A × B  (accumulate-in-place)
    //
    // Register requirements per thread:
    //   A: 4 × uint32_t  (16 INT8 values, m16×k32 row-major)
    //   B: 2 × uint32_t  (8 INT8 values, k32×n8 col-major)
    //   D: 4 × int32_t   (4 INT32 output/accumulator, m16×n8)

    [[maybe_unused]] __device__ __forceinline__ void mma_m16n8k32_s8(
        int32_t D[4],
        const uint32_t A[4],
        const uint32_t B[2])
    {
#if __CUDA_ARCH__ >= 800
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%0,  %1,  %2,  %3};\n"
            : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
              "r"(B[0]), "r"(B[1]));
#else
        (void)D;
        (void)A;
        (void)B;
#endif
    }

    // ════════════════════════════════════════════════════════════════════════
    // V2 Fused tensor-core GEMM kernel (SM80+ Ampere)
    //
    // One mma.sync.m16n8k32 per K=32 quantisation block per (wi,wj) tile.
    // STAGES-deep shared memory pipeline with cp.async transfers.
    // Optional split-K via grid.z partitioning + fixed-order reduction epilogue.
    // ════════════════════════════════════════════════════════════════════════

    template <int BM, int BN, int WARPS_M, int WARPS_N,
              int STAGES = 2, int SPLIT_K = 1>
    __global__ void __launch_bounds__(WARPS_M * WARPS_N * 32,
                                      (WARPS_M * WARPS_N <= 4 && STAGES <= 2) ? 2 : 1)
        fusedTCGemmV2Kernel(
            const int8_t *__restrict__ A,
            const int8_t *__restrict__ B_tc,
            float *__restrict__ C,
            float *__restrict__ splitk_partials,
            const float *__restrict__ scales_A,
            const float *__restrict__ scales_B,
            const float *__restrict__ C_existing,
            const float *__restrict__ bias,
            int M, int N, int K,
            float alpha, float beta)
    {
#if __CUDA_ARCH__ >= 800
        // ─── Compile-time tile geometry ───
        constexpr int NUM_WARPS = WARPS_M * WARPS_N;
        constexpr int BLOCK_SIZE = NUM_WARPS * 32;
        constexpr int WARP_M = BM / WARPS_M;
        constexpr int WARP_N = BN / WARPS_N;
        constexpr int WM = WARP_M / 16; // MMA tiles vertically per warp
        constexpr int WN = WARP_N / 8;  // MMA tiles horizontally per warp

        static_assert(BM % WARPS_M == 0 && BN % WARPS_N == 0);
        static_assert(WARP_M % 16 == 0, "WARP_M must be a multiple of 16 (MMA m-dim)");
        static_assert(WARP_N % 8 == 0, "WARP_N must be a multiple of 8 (MMA n-dim)");
        static_assert(STAGES >= 2 && STAGES <= 4, "STAGES must be 2, 3, or 4");
        static_assert(SPLIT_K >= 1 && SPLIT_K <= 8, "SPLIT_K must be 1..8");

        // ─── Thread / warp identification ───
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 31;
        const int wr = warp_id / WARPS_N; // warp row in CTA grid
        const int wc = warp_id % WARPS_N; // warp col in CTA grid
        const int gid = lane_id >> 2;     // groupID for MMA (0-7)
        // Note: tid_in_group (lane_id & 3) is implicit inside the ldmatrix/mma
        // helpers; not needed as a separate local here.

        const int block_m = blockIdx.x * BM;
        const int block_n = blockIdx.y * BN;
        const size_t splitk_plane_stride = static_cast<size_t>(M) * static_cast<size_t>(N);
        float *splitk_plane = nullptr;
        if constexpr (SPLIT_K > 1)
        {
            splitk_plane = splitk_partials + static_cast<size_t>(blockIdx.z) * splitk_plane_stride;
        }

        // ─── K-dimension range (split-K partitioning) ───
        const int num_k_blocks_total = K / BK;
        int kb_begin, kb_end;
        if constexpr (SPLIT_K > 1)
        {
            const int blocks_per_part = (num_k_blocks_total + SPLIT_K - 1) / SPLIT_K;
            kb_begin = static_cast<int>(blockIdx.z) * blocks_per_part;
            kb_end = min(kb_begin + blocks_per_part, num_k_blocks_total);
        }
        else
        {
            kb_begin = 0;
            kb_end = num_k_blocks_total;
        }
        const int num_k_iters = kb_end - kb_begin;

        // ─── Multi-stage shared memory (padded, circular) ───
        __shared__ int8_t smem_A[STAGES][BM * SMEM_STRIDE];
        __shared__ int8_t smem_B[STAGES][BN * SMEM_STRIDE];
        __shared__ float smem_scales_B[BN];

        // ─── FP32 accumulators (persist across all K blocks) ───
        float acc[WM][WN][4];
#pragma unroll
        for (int i = 0; i < WM; i++)
#pragma unroll
            for (int j = 0; j < WN; j++)
#pragma unroll
                for (int e = 0; e < 4; e++)
                    acc[i][j][e] = 0.0f;

        // Bail early if this split-K partition has no work
        if (num_k_iters <= 0)
            return;

        for (int idx = threadIdx.x; idx < BN; idx += BLOCK_SIZE)
        {
            const int gcol = block_n + idx;
            smem_scales_B[idx] = (gcol < N) ? scales_B[gcol] : 0.0f;
        }
        __syncthreads();

        // ─── Tile load helpers: global → shared via cp.async ───
        //
        // cp.async.cg.shared.global.L2::128B copies 16 bytes (int4) from global
        // to shared memory through L2 while bypassing L1.  src_size={16,0}
        // provides zero-fill for OOB lanes without a separate branch/store path.

        // A: [M × K] row-major.  BM rows × 32 bytes per K-block.
        constexpr int A_VEC_LOADS = BM * BK / 16;

        auto load_A_tile = [&](int stage, int kb) __attribute__((always_inline))
        {
#pragma unroll 4
            for (int idx = threadIdx.x; idx < A_VEC_LOADS; idx += BLOCK_SIZE)
            {
                const int linear = idx << 4; // byte offset in dense tile
                const int row = linear >> 5; // / 32
                const int col = linear & 31; // % 32
                const int grow = block_m + row;
                void *dst = &smem_A[stage][row * SMEM_STRIDE + col];
                const bool valid = grow < M;
                const void *src = valid
                                      ? static_cast<const void *>(&A[static_cast<size_t>(grow) * K + kb * BK + col])
                                      : static_cast<const void *>(A);
                cp_async_cg_16_zfill_128(dst, src, valid ? 16 : 0);
            }
        };

        // B: [K/32][N][32] tc-blocked.  Contiguous BN×32 bytes per K-block.
        constexpr int B_VEC_LOADS = BN * BK / 16;

        auto load_B_tile = [&](int stage, int kb) __attribute__((always_inline))
        {
            const int8_t *b_base = B_tc + static_cast<size_t>(kb) * N * BK + static_cast<size_t>(block_n) * BK;
#pragma unroll 4
            for (int idx = threadIdx.x; idx < B_VEC_LOADS; idx += BLOCK_SIZE)
            {
                const int linear = idx << 4;
                const int col = linear >> 5;  // N-dimension column
                const int elem = linear & 31; // K-dimension element
                const int gcol = block_n + col;
                void *dst = &smem_B[stage][col * SMEM_STRIDE + elem];
                const bool valid = gcol < N;
                const void *src = valid
                                      ? static_cast<const void *>(&b_base[linear])
                                      : static_cast<const void *>(B_tc);
                cp_async_cg_16_zfill_128(dst, src, valid ? 16 : 0);
            }
        };

        auto compute_k_block = [&](int stage, int kb) __attribute__((always_inline))
        {
        // ─── Compute: load MMA fragments from smem → registers → MMA ───
        //
        // PTX ISA §9.7.13.4.10 — mma.m16n8k32 .s8 fragment layout:
        //   A_frag[0] (a0..a3):   row=gid,      k=tid*4+{0..3}       (k: 0..15)
        //   A_frag[1] (a4..a7):   row=gid+8,    k=tid*4+{0..3}       (k: 0..15)
        //   A_frag[2] (a8..a11):  row=gid,      k=tid*4+{0..3}+16    (k: 16..31)
        //   A_frag[3] (a12..a15): row=gid+8,    k=tid*4+{0..3}+16    (k: 16..31)
        //
        // B_frag[0] (b0..b3):  row=tid*4+{0..3}     (k: 0..15),  col=gid
        // B_frag[1] (b4..b7):  row=tid*4+{0..3}+16  (k: 16..31), col=gid

#pragma unroll
            for (int wi = 0; wi < WM; wi++)
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
                const float sa0 = (grow0 < M)
                                      ? scales_A[grow0 * num_k_blocks_total + kb]
                                      : 0.0f;
                const float sa1 = (grow1 < M)
                                      ? scales_A[grow1 * num_k_blocks_total + kb]
                                      : 0.0f;

#pragma unroll
                for (int wj = 0; wj < WN; wj++)
                {
                    const int b_col_base = wc * WARP_N + wj * 8;

                    uint32_t B_frag[2];
                    load_ldmatrix_b_m16n8k32(
                        B_frag,
                        reinterpret_cast<const int *>(&smem_B[stage][b_col_base * SMEM_STRIDE]),
                        SMEM_STRIDE / 4,
                        lane_id);

                    int32_t D_frag[4] = {0, 0, 0, 0};
                    mma_m16n8k32_s8(D_frag, A_frag, B_frag);

#pragma unroll
                    for (int e = 0; e < 4; e++)
                    {
                        const float s = (e < 2) ? sa0 : sa1;
                        acc[wi][wj][e] += static_cast<float>(D_frag[e]) * s;
                    }
                }
            }
        };

        if constexpr (STAGES == 2)
        {
            // Phase 2: pair two K=32 blocks per outer loop body. This preserves the
            // existing K=32 data layout and per-block scaling contract, while cutting
            // loop/control overhead roughly in half on the hot STAGES=2 path.
            load_A_tile(0, kb_begin);
            load_B_tile(0, kb_begin);
            cp_async_commit();

            for (int ki = 0; ki < num_k_iters; ki += 2)
            {
                cp_async_wait<0>();
                __syncthreads();

                const int kb0 = kb_begin + ki;

                if (ki + 1 < num_k_iters)
                {
                    load_A_tile(1, kb_begin + ki + 1);
                    load_B_tile(1, kb_begin + ki + 1);
                    cp_async_commit();
                }

                compute_k_block(0, kb0);

                if (ki + 1 < num_k_iters)
                {
                    cp_async_wait<0>();
                    __syncthreads();

                    if (ki + 2 < num_k_iters)
                    {
                        load_A_tile(0, kb_begin + ki + 2);
                        load_B_tile(0, kb_begin + ki + 2);
                        cp_async_commit();
                    }

                    compute_k_block(1, kb_begin + ki + 1);
                }
            }
        }
        else
        {
// ─── Prologue: fill pipeline with (STAGES - 1) stages ───
#pragma unroll
            for (int s = 0; s < STAGES - 1; s++)
            {
                if (s < num_k_iters)
                {
                    load_A_tile(s, kb_begin + s);
                    load_B_tile(s, kb_begin + s);
                }
                cp_async_commit();
            }

            for (int ki = 0; ki < num_k_iters; ki++)
            {
                cp_async_wait<STAGES - 2>();
                __syncthreads();

                const int cur = ki % STAGES;
                const int kb = kb_begin + ki;
                const int load_ki = ki + STAGES - 1;
                if (load_ki < num_k_iters)
                {
                    load_A_tile(load_ki % STAGES, kb_begin + load_ki);
                    load_B_tile(load_ki % STAGES, kb_begin + load_ki);
                }
                cp_async_commit();

                compute_k_block(cur, kb);
                __syncthreads();
            }
        }

        // ─── Epilogue: scale_B · alpha + beta · C_existing + bias → C ───
        //
        // Fragment layout for m16×n8 output:
        //   row = (elem / 2) * 8 + groupID
        //   col = tid_in_group * 2 + (elem % 2)
        //
        // SPLIT_K > 1: each partition writes into its own FP32 scratch plane.
        //   A follow-up reduction kernel applies beta*C_existing and bias once.
        // SPLIT_K = 1: direct store (standard path).

        const bool simple_epilogue = (beta == 0.0f) && (bias == nullptr);

#pragma unroll
        for (int wj = 0; wj < WN; wj++)
        {
            const int tile_n = block_n + wc * WARP_N + wj * 8;
            const int gc0 = tile_n + v2_frag_col(lane_id, 0);
            const int gc1 = tile_n + v2_frag_col(lane_id, 1);
            const bool gc0_valid = gc0 < N;
            const bool gc1_valid = gc1 < N;
            const float scale0 = gc0_valid ? (alpha * smem_scales_B[gc0 - block_n]) : 0.0f;
            const float scale1 = gc1_valid ? (alpha * smem_scales_B[gc1 - block_n]) : 0.0f;
            const float bias0 = (bias && gc0_valid) ? bias[gc0] : 0.0f;
            const float bias1 = (bias && gc1_valid) ? bias[gc1] : 0.0f;

#pragma unroll
            for (int wi = 0; wi < WM; wi++)
            {
                const int tile_m = block_m + wr * WARP_M + wi * 16;
                const bool interior = (tile_m + 15 < M) && (tile_n + 7 < N);

                if (interior && simple_epilogue)
                {
                    const int out_idx0 = (tile_m + v2_frag_row(lane_id, 0)) * N + gc0;
                    const int out_idx1 = (tile_m + v2_frag_row(lane_id, 1)) * N + gc1;
                    const int out_idx2 = (tile_m + v2_frag_row(lane_id, 2)) * N + gc0;
                    const int out_idx3 = (tile_m + v2_frag_row(lane_id, 3)) * N + gc1;

                    if constexpr (SPLIT_K > 1)
                    {
                        splitk_plane[out_idx0] = acc[wi][wj][0] * scale0;
                        splitk_plane[out_idx1] = acc[wi][wj][1] * scale1;
                        splitk_plane[out_idx2] = acc[wi][wj][2] * scale0;
                        splitk_plane[out_idx3] = acc[wi][wj][3] * scale1;
                    }
                    else
                    {
                        C[out_idx0] = acc[wi][wj][0] * scale0;
                        C[out_idx1] = acc[wi][wj][1] * scale1;
                        C[out_idx2] = acc[wi][wj][2] * scale0;
                        C[out_idx3] = acc[wi][wj][3] * scale1;
                    }
                    continue;
                }

#pragma unroll
                for (int e = 0; e < 4; e++)
                {
                    const int gr = tile_m + v2_frag_row(lane_id, e);
                    const int gc = (e & 1) ? gc1 : gc0;

                    if (gr < M && gc < N)
                    {
                        const int out_idx = gr * N + gc;
                        float val = acc[wi][wj][e] * ((e & 1) ? scale1 : scale0);

                        if constexpr (SPLIT_K > 1)
                        {
                            splitk_plane[out_idx] = val;
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
        // SM75 (Turing): this kernel is not supported — return without writing.
        // The host dispatch function returns false for non-Ampere GPUs.
        (void)A;
        (void)B_tc;
        (void)C;
        (void)scales_A;
        (void)scales_B;
        (void)C_existing;
        (void)bias;
        (void)M;
        (void)N;
        (void)K;
        (void)alpha;
        (void)beta;
#endif // __CUDA_ARCH__ >= 800
    }

    template <int SPLIT_K, bool APPLY_BETA, bool APPLY_BIAS>
    __global__ void reduceSplitKPartialsKernel(
        const float *__restrict__ splitk_partials,
        float *__restrict__ C,
        const float *__restrict__ C_existing,
        const float *__restrict__ bias,
        int total,
        int N,
        float beta)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total)
            return;

        const size_t plane_stride = static_cast<size_t>(total);
        float sum = 0.0f;
#pragma unroll
        for (int part = 0; part < SPLIT_K; ++part)
            sum += splitk_partials[static_cast<size_t>(part) * plane_stride + idx];

        if constexpr (APPLY_BETA)
            sum += beta * C_existing[idx];
        if constexpr (APPLY_BIAS)
            sum += bias[idx % N];

        C[idx] = sum;
    }

    // ════════════════════════════════════════════════════════════════════════
    // Shape classification and dispatch
    // ════════════════════════════════════════════════════════════════════════

    enum class ShapeFamily
    {
        SkinnyM,  // M ≤ 32           → BM=32,  BN=128 (1×2,  64 threads)
        NarrowN,  // N ≤ 2048         → BM=128, BN=64  (2×1,  64 threads)
        DeepK,    // K > 2·N          → BM=64,  BN=128 (2×2, 128 threads)
        WideN,    // N ≥ 8192         → BM=64,  BN=128 (2×2, 128 threads)
        Balanced, // default           → BM=128, BN=128 (2×2, 128 threads)
        Compact,  // grid fallback     → BM=32,  BN=64  (1×1,  32 threads)
    };

    static const char *shapeFamilyName(ShapeFamily f)
    {
        switch (f)
        {
        case ShapeFamily::SkinnyM:
            return "v2_skinny_m";
        case ShapeFamily::NarrowN:
            return "v2_narrow_n";
        case ShapeFamily::DeepK:
            return "v2_deep_k";
        case ShapeFamily::WideN:
            return "v2_wide_n";
        case ShapeFamily::Balanced:
            return "v2_balanced";
        case ShapeFamily::Compact:
            return "v2_compact";
        }
        return "v2_unknown";
    }

    static bool tryParseForcedShapeFamily(const char *value, ShapeFamily &out)
    {
        if (!value || *value == '\0')
            return false;

        if (std::strcmp(value, "skinny") == 0 || std::strcmp(value, "skinny_m") == 0 || std::strcmp(value, "v2_skinny_m") == 0)
            out = ShapeFamily::SkinnyM;
        else if (std::strcmp(value, "narrow") == 0 || std::strcmp(value, "narrow_n") == 0 || std::strcmp(value, "v2_narrow_n") == 0)
            out = ShapeFamily::NarrowN;
        else if (std::strcmp(value, "deep") == 0 || std::strcmp(value, "deep_k") == 0 || std::strcmp(value, "v2_deep_k") == 0)
            out = ShapeFamily::DeepK;
        else if (std::strcmp(value, "wide") == 0 || std::strcmp(value, "wide_n") == 0 || std::strcmp(value, "v2_wide_n") == 0)
            out = ShapeFamily::WideN;
        else if (std::strcmp(value, "balanced") == 0 || std::strcmp(value, "v2_balanced") == 0)
            out = ShapeFamily::Balanced;
        else if (std::strcmp(value, "compact") == 0 || std::strcmp(value, "v2_compact") == 0)
            out = ShapeFamily::Compact;
        else
            return false;

        return true;
    }

    static bool getForcedShapeFamily(ShapeFamily &out)
    {
        return tryParseForcedShapeFamily(std::getenv("LLAMINAR_CUDA_FUSEDTC_V2_FORCE_FAMILY"), out);
    }

    // Last-selected family for perf harness introspection (host-side only).
    static thread_local int g_last_family_v2 = 4; // Balanced

    static constexpr int MIN_GRID_BLOCKS = 64;
    static thread_local int g_last_split_k_v2 = 1;

    static int getSplitKScratchPlanesForWorkspace(int M, int N, int K)
    {
        const int num_k_blocks = K / BK;
        if (num_k_blocks <= 1)
            return 1;

        const size_t partial_plane_bytes = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);
        const int budget_limited_chunk_count = (partial_plane_bytes == 0)
                                                   ? 1
                                                   : static_cast<int>(SPLIT_K_PARTIAL_SCRATCH_BUDGET_BYTES / partial_plane_bytes);
        const int max_chunk_count = std::max(1, std::min(MAX_SPLIT_K_PARTITIONS, budget_limited_chunk_count));
        const int deepk_grid_blocks = ((M + 64 - 1) / 64) * ((N + 128 - 1) / 128);
        const int balanced_grid_blocks = ((M + 128 - 1) / 128) * ((N + 128 - 1) / 128);
        const int effective_grid_blocks = min(deepk_grid_blocks, balanced_grid_blocks);
        const bool k_rich = K > 2 * N;
        const bool recover_underfill = effective_grid_blocks < MIN_GRID_BLOCKS;
        const uint64_t total_output_elements = static_cast<uint64_t>(M) * static_cast<uint64_t>(N);
        if (num_k_blocks >= 64 && (k_rich || recover_underfill))
            return min(num_k_blocks, max_chunk_count);
        if (M >= 64 || N >= 16384 || total_output_elements >= (1ull << 20))
            return std::min(num_k_blocks, max_chunk_count);
        if (total_output_elements >= (1ull << 18))
            return std::min(4, max_chunk_count);
        return 1;
    }

    static int roundUpPow2Clamped(int value, int max_value)
    {
        int out = 1;
        while (out < value && out < max_value)
            out <<= 1;
        return out;
    }

    static int chooseSplitK(ShapeFamily family, int M, int N, int K, int bm, int bn, int max_available_split_k)
    {
        // Deterministic mode still forces split_k=1 to preserve the strictest
        // parity behavior even though the split-K path is now reduction-based.
        static const bool s_deterministic = []()
        {
            const char *env = std::getenv("LLAMINAR_DETERMINISTIC");
            return env && std::atoi(env) != 0;
        }();
        if (s_deterministic || max_available_split_k <= 1)
            return 1;

        const int grid_blocks = ((M + bm - 1) / bm) * ((N + bn - 1) / bn);
        const int num_k_blocks = K / BK;
        const bool k_rich = (family == ShapeFamily::DeepK) || (K > 2 * N);
        const bool recover_underfill =
            family != ShapeFamily::SkinnyM &&
            family != ShapeFamily::NarrowN &&
            grid_blocks < MIN_GRID_BLOCKS &&
            num_k_blocks >= 64;

        if (!k_rich && !recover_underfill)
            return 1;
        if (!recover_underfill && grid_blocks >= MIN_GRID_BLOCKS)
            return 1;
        if (num_k_blocks < 4)
            return 1;

        const int needed = (MIN_GRID_BLOCKS + grid_blocks - 1) / grid_blocks;
        int split_k = roundUpPow2Clamped(needed, 8);
        while (split_k > max_available_split_k)
            split_k >>= 1;

        const int min_k_blocks_per_partition = k_rich ? 2 : 8;
        while (split_k > 1 && (num_k_blocks / split_k) < min_k_blocks_per_partition)
            split_k >>= 1;

        return max(1, split_k);
    }

    static ShapeFamily classifyShape(int M, int N, int K)
    {
        const int num_k_blocks = K / BK;
        if (K > 2 * N && num_k_blocks >= 64)
            return ShapeFamily::DeepK;
        if (M <= 32)
            return ShapeFamily::SkinnyM;

        ShapeFamily family;
        if (K > 2 * N)
            family = ShapeFamily::DeepK;
        else if (N < 2048)
            family = ShapeFamily::NarrowN;
        else if (N >= 8192 || (N >= 4096 && K >= 2048))
            family = ShapeFamily::WideN;
        else
            family = ShapeFamily::Balanced;

        // K-rich shapes will rely on split-K to recover CTA count instead of
        // being demoted to skinny tiles purely for occupancy reasons.
        if (family == ShapeFamily::DeepK)
            return family;

        int bm, bn;
        switch (family)
        {
        case ShapeFamily::NarrowN:
            bm = 128;
            bn = 64;
            break;
        case ShapeFamily::DeepK:
            bm = 64;
            bn = 128;
            break;
        case ShapeFamily::WideN:
            bm = 64;
            bn = 256;
            break;
        default:
            bm = 128;
            bn = 128;
            break;
        }
        int grid_blocks = ((M + bm - 1) / bm) * ((N + bn - 1) / bn);
        const bool preserve_family_with_splitk =
            family != ShapeFamily::NarrowN &&
            family != ShapeFamily::SkinnyM &&
            num_k_blocks >= 64;

        if (grid_blocks < MIN_GRID_BLOCKS && !preserve_family_with_splitk)
            return ShapeFamily::SkinnyM;

        return family;
    }

    // ─── Architecture check: SM80+ (Ampere) ───
    static bool isAmperePlus(int device_id)
    {
        static int cached_device = -1;
        static bool cached_result = false;
        if (cached_device == device_id)
            return cached_result;

        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess)
            return false;

        cached_result = (prop.major >= 8);
        cached_device = device_id;
        return cached_result;
    }

    // ─── Launch helper (STAGES=2 default, SPLIT_K=1 default) ───
    template <int BM, int BN, int WM, int WN, int STAGES = 2, int SPLIT_K = 1>
    static bool launchV2Kernel(
        const int8_t *A, const int8_t *B_tc,
        float *C, float *splitk_partials, const float *scales_A, const float *scales_B,
        const float *C_existing, const float *bias,
        int M, int N, int K, float alpha, float beta,
        cudaStream_t stream)
    {
        const dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, SPLIT_K);
        const dim3 block(WM * WN * 32);

        if constexpr (SPLIT_K > 1)
        {
            if (!splitk_partials)
                return false;
        }

        fusedTCGemmV2Kernel<BM, BN, WM, WN, STAGES, SPLIT_K><<<grid, block, 0, stream>>>(
            A, B_tc, C, splitk_partials, scales_A, scales_B, C_existing, bias,
            M, N, K, alpha, beta);
        if (cudaGetLastError() != cudaSuccess)
            return false;

        if constexpr (SPLIT_K > 1)
        {
            constexpr int THREADS = 256;
            const int total = M * N;
            const dim3 reduce_grid((total + THREADS - 1) / THREADS);
            const bool apply_beta = (beta != 0.0f) && (C_existing != nullptr);
            const bool apply_bias = (bias != nullptr);

            if (apply_beta && apply_bias)
                reduceSplitKPartialsKernel<SPLIT_K, true, true><<<reduce_grid, THREADS, 0, stream>>>(splitk_partials, C, C_existing, bias, total, N, beta);
            else if (apply_beta)
                reduceSplitKPartialsKernel<SPLIT_K, true, false><<<reduce_grid, THREADS, 0, stream>>>(splitk_partials, C, C_existing, bias, total, N, beta);
            else if (apply_bias)
                reduceSplitKPartialsKernel<SPLIT_K, false, true><<<reduce_grid, THREADS, 0, stream>>>(splitk_partials, C, C_existing, bias, total, N, beta);
            else
                reduceSplitKPartialsKernel<SPLIT_K, false, false><<<reduce_grid, THREADS, 0, stream>>>(splitk_partials, C, C_existing, bias, total, N, beta);

            return cudaGetLastError() == cudaSuccess;
        }

        return true;
    }

} // anonymous namespace

// ════════════════════════════════════════════════════════════════════════
// Public C API
// ════════════════════════════════════════════════════════════════════════

extern "C"
{
    bool cudaFusedTCGemmV2_blockwiseGemm(
        const int8_t *d_A_int8,
        const int8_t *d_weights_int8_tc_blocked,
        int32_t *d_partial_int32,
        float *d_C_fp32,
        const float *d_scales_A_block,
        const float *d_scales_B,
        int M, int N, int K,
        float alpha, float beta,
        const float *d_C_existing,
        const float *d_bias,
        int cuda_device_id,
        void *stream)
    {
        if (!d_A_int8 || !d_weights_int8_tc_blocked || !d_C_fp32 || !d_scales_A_block || !d_scales_B)
            return false;
        if (M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
            return false;
        if (!isAmperePlus(cuda_device_id))
            return false;

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        if (cudaSetDevice(cuda_device_id) != cudaSuccess)
            return false;

        float *d_splitk_partials = reinterpret_cast<float *>(d_partial_int32);
        const int max_split_k = d_splitk_partials ? getSplitKScratchPlanesForWorkspace(M, N, K) : 1;

        ShapeFamily family = classifyShape(M, N, K);
        ShapeFamily forced_family;
        if (getForcedShapeFamily(forced_family))
            family = forced_family;
        g_last_family_v2 = static_cast<int>(family);
        g_last_split_k_v2 = 1;

        switch (family)
        {
        case ShapeFamily::SkinnyM:
            return launchV2Kernel<32, 128, 1, 2>(
                d_A_int8, d_weights_int8_tc_blocked,
                d_C_fp32, nullptr, d_scales_A_block, d_scales_B,
                d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);

        case ShapeFamily::NarrowN:
            return launchV2Kernel<128, 64, 2, 1>(
                d_A_int8, d_weights_int8_tc_blocked,
                d_C_fp32, nullptr, d_scales_A_block, d_scales_B,
                d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);

        case ShapeFamily::DeepK:
        {
            const int split_k = chooseSplitK(family, M, N, K, 64, 128, max_split_k);
            g_last_split_k_v2 = split_k;
            switch (split_k)
            {
            case 8:
                return launchV2Kernel<64, 128, 2, 2, 2, 8>(
                    d_A_int8, d_weights_int8_tc_blocked,
                    d_C_fp32, d_splitk_partials, d_scales_A_block, d_scales_B,
                    d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);
            case 4:
                return launchV2Kernel<64, 128, 2, 2, 2, 4>(
                    d_A_int8, d_weights_int8_tc_blocked,
                    d_C_fp32, d_splitk_partials, d_scales_A_block, d_scales_B,
                    d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);
            case 2:
                return launchV2Kernel<64, 128, 2, 2, 2, 2>(
                    d_A_int8, d_weights_int8_tc_blocked,
                    d_C_fp32, d_splitk_partials, d_scales_A_block, d_scales_B,
                    d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);
            default:
                return launchV2Kernel<64, 128, 2, 2>(
                    d_A_int8, d_weights_int8_tc_blocked,
                    d_C_fp32, nullptr, d_scales_A_block, d_scales_B,
                    d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);
            }
        }

        case ShapeFamily::WideN:
        {
            const int split_k = chooseSplitK(family, M, N, K, 64, 128, max_split_k);
            g_last_split_k_v2 = split_k;
            switch (split_k)
            {
            case 8:
                return launchV2Kernel<64, 128, 2, 2, 2, 8>(
                    d_A_int8, d_weights_int8_tc_blocked,
                    d_C_fp32, d_splitk_partials, d_scales_A_block, d_scales_B,
                    d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);
            case 4:
                return launchV2Kernel<64, 128, 2, 2, 2, 4>(
                    d_A_int8, d_weights_int8_tc_blocked,
                    d_C_fp32, d_splitk_partials, d_scales_A_block, d_scales_B,
                    d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);
            case 2:
                return launchV2Kernel<64, 128, 2, 2, 2, 2>(
                    d_A_int8, d_weights_int8_tc_blocked,
                    d_C_fp32, d_splitk_partials, d_scales_A_block, d_scales_B,
                    d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);
            default:
                return launchV2Kernel<64, 128, 2, 2>(
                    d_A_int8, d_weights_int8_tc_blocked,
                    d_C_fp32, nullptr, d_scales_A_block, d_scales_B,
                    d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);
            }
        }

        case ShapeFamily::Balanced:
        {
            const int split_k = chooseSplitK(family, M, N, K, 128, 128, max_split_k);
            g_last_split_k_v2 = split_k;
            switch (split_k)
            {
            case 8:
                return launchV2Kernel<128, 128, 2, 2, 2, 8>(
                    d_A_int8, d_weights_int8_tc_blocked,
                    d_C_fp32, d_splitk_partials, d_scales_A_block, d_scales_B,
                    d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);
            case 4:
                return launchV2Kernel<128, 128, 2, 2, 2, 4>(
                    d_A_int8, d_weights_int8_tc_blocked,
                    d_C_fp32, d_splitk_partials, d_scales_A_block, d_scales_B,
                    d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);
            case 2:
                return launchV2Kernel<128, 128, 2, 2, 2, 2>(
                    d_A_int8, d_weights_int8_tc_blocked,
                    d_C_fp32, d_splitk_partials, d_scales_A_block, d_scales_B,
                    d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);
            default:
                return launchV2Kernel<128, 128, 2, 2>(
                    d_A_int8, d_weights_int8_tc_blocked,
                    d_C_fp32, nullptr, d_scales_A_block, d_scales_B,
                    d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);
            }
        }

        case ShapeFamily::Compact:
            return launchV2Kernel<32, 64, 1, 1>(
                d_A_int8, d_weights_int8_tc_blocked,
                d_C_fp32, nullptr, d_scales_A_block, d_scales_B,
                d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);
        }

        return false;
    }

    const char *cudaFusedTCGemmV2_lastSelectedFamily()
    {
        return shapeFamilyName(static_cast<ShapeFamily>(g_last_family_v2));
    }

    int cudaFusedTCGemmV2_lastSelectedSplitK()
    {
        return g_last_split_k_v2;
    }
}
